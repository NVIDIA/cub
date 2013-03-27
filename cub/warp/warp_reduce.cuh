/******************************************************************************
 * Copyright (c) 2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2013, NVIDIA CORPORATION.  All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

/**
 * \file
 * cub::WarpReduce provides variants of parallel reduction across CUDA warps.
 */

#pragma once

#include "../thread/thread_load.cuh"
#include "../thread/thread_store.cuh"
#include "../util_device.cuh"
#include "../util_type.cuh"
#include "../thread/thread_operators.cuh"
#include "../util_namespace.cuh"

/// Optional outer namespace(s)
CUB_NS_PREFIX

/// CUB namespace
namespace cub {

/**
 * \addtogroup WarpModule
 * @{
 */

/**
 * \brief WarpReduce provides variants of parallel reduction across CUDA warps.  ![](warp_reduce_logo.png)
 *
 * \par Overview
 * A <a href="http://en.wikipedia.org/wiki/Reduce_(higher-order_function)"><em>reduction</em> (or <em>fold</em>)</a>
 * uses a binary combining operator to compute a single aggregate from a list of input elements.
 *
 * \par
 * For convenience, WarpReduce exposes a spectrum of entrypoints that differ by:
 * - Operator (generic reduction <em>vs.</em> summation for numeric types)
 * - Input (full warps <em>vs.</em> partially-full warps having some undefined elements)
 *
 * \tparam T                        The reduction input/output element type
 * \tparam WARPS                    The number of "logical" warps performing concurrent warp reductions
 * \tparam LOGICAL_WARP_THREADS     <b>[optional]</b> The number of threads per "logical" warp (may be less than the number of hardware warp threads).  Default is the warp size associated with the CUDA Compute Capability targeted by the compiler (e.g., 32 threads for SM20).
 *
 * \par Usage Considerations
 * - Supports non-commutative reduction operators
 * - Supports "logical" warps smaller than the physical warp size (e.g., a logical warp of 8 threads)
 * - Warp reductions are concurrent if more than one warp is participating
 * - The warp-wide scalar reduction output is only considered valid in <em>warp-lane</em><sub>0</sub>
 * - \smemreuse{WarpReduce::SmemStorage}

 * \par Performance Considerations
 * - Uses special instructions when applicable (e.g., warp \p SHFL)
 * - Uses synchronization-free communication between warp lanes when applicable
 * - Zero bank conflicts for most types.
 * - Computation is slightly more efficient (i.e., having lower instruction overhead) for:
 *     - Reduction parameterizations where \p T is a built-in C++ primitive or CUDA vector type (e.g.,
 *       \p short, \p int2, \p double, \p float2, etc.)
 *     - Reduction parameterizations where \p LOGICAL_WARP_THREADS is a multiple of the architecture's warp size
 *
 * \par Algorithm
 * These parallel reduction variants implement a warp-synchronous
 * Kogge-Stone algorithm having <em>O</em>(log<em>n</em>)
 * steps and <em>O</em>(<em>n</em>log<em>n</em>) work complexity,
 * where <em>n</em> = \p LOGICAL_WARP_THREADS (which defaults to the warp
 * size associated with the CUDA Compute Capability targeted by the compiler).
 * <br><br>
 * \image html kogge_stone_reduction.png
 * <div class="centercaption">Data flow within a 16-thread Kogge-Stone reduction construction.  Junctions represent binary operators.</div>
 * <br>
 *
 * \par Examples
 * <em>Example 1.</em> Perform a simple sum reduction for one warp
 * \code
 * #include <cub.cuh>
 *
 * __global__ void SomeKernel(...)
 * {
 *     // A parameterized int-based WarpReduce type for use with one warp.
 *     typedef cub::WarpReduce<int, 1> WarpReduce;
 *
 *     // Opaque shared memory for WarpReduce
 *     __shared__ typename WarpReduce::SmemStorage smem_storage;
 *
 *     // Perform prefix sum of threadIds in first warp
 *     if (threadIdx.x < 32)
 *     {
 *         int input = threadIdx.x;
 *         int output = WarpReduce::Sum(smem_storage, input, output);
 *
 *         if (threadIdx.x == 0)
 *             printf("tid(%d) output(%d)\n\n", threadIdx.x, output);
 *     }
 *
 *     ...
 * \endcode
 * Printed output:
 * \code
 * tid(0) output(496)
 * \endcode
 *
 */
template <
    typename    T,
    int         WARPS,
    int         LOGICAL_WARP_THREADS = PtxArchProps::WARP_THREADS>
class WarpReduce
{
    /// BlockReduce is a friend class that has access to the WarpReduceInternal classes
    template <typename T, int BLOCK_THERADS>
    friend class BlockReduce;

    //---------------------------------------------------------------------
    // Constants and typedefs
    //---------------------------------------------------------------------

private:

    /// WarpReduce algorithmic variants
    enum WarpReducePolicy
    {
        SHFL_REDUCE,          // Warp-synchronous SHFL-based reduction
        SMEM_REDUCE,          // Warp-synchronous smem-based reduction
    };

    /// Constants
    enum
    {
        POW_OF_TWO = ((LOGICAL_WARP_THREADS & (LOGICAL_WARP_THREADS - 1)) == 0),

        /// Use SHFL_REDUCE if (architecture is >= SM30) and (T is a primitive) and (T is 4-bytes or smaller) and (LOGICAL_WARP_THREADS is a power-of-two)
        POLICY = ((CUB_PTX_ARCH >= 300) && Traits<T>::PRIMITIVE && (sizeof(T) <= 4) && POW_OF_TWO) ?
            SHFL_REDUCE :
            SMEM_REDUCE,
    };


    /** \cond INTERNAL */


    /**
     * WarpReduce specialized for SHFL_REDUCE variant
     */
    template <int POLICY, int DUMMY = 0>
    struct WarpReduceInternal
    {
        /// Constants
        enum
        {
            /// The number of warp reduction steps
            STEPS = Log2<LOGICAL_WARP_THREADS>::VALUE,

            // The 5-bit SHFL mask for logically splitting warps into sub-segments
            SHFL_MASK = (-1 << STEPS) & 31,

            // The 5-bit SFHL clamp
            SHFL_CLAMP = LOGICAL_WARP_THREADS - 1,

            // The packed C argument (mask starts 8 bits up)
            SHFL_C = (SHFL_MASK << 8) | SHFL_CLAMP,
        };

        /// Shared memory storage layout type
        typedef NullType SmemStorage;

        /// Reduction (specialized for unsigned int)
        template <
            bool    FULL_TILE,
            int     VALID_PER_LANE>
        static __device__ __forceinline__ unsigned int Sum(
            SmemStorage         &smem_storage,      ///< [in] Shared reference to opaque SmemStorage layout
            unsigned int        input,              ///< [in] Calling thread's input
            const unsigned int  &valid)             ///< [in] Number of valid lanes in the calling thread's logical warp (may be less than \p LOGICAL_WARP_THREADS)
        {
            unsigned int lane_id = (WARPS == 1) ? threadIdx.x : (threadIdx.x & (LOGICAL_WARP_THREADS - 1));

            // Iterate reduction steps
            #pragma unroll
            for (int STEP = 0; STEP < STEPS; STEP++)
            {
                const int OFFSET = 1 << STEP;

                if (!FULL_TILE)
                {
                    asm(
                        "{"
                        "  .reg .u32 r0;"
                        "  .reg .pred p;"
                        "  shfl.down.b32 r0|p, %1, %2, %3;"
                        "  setp.lt.and.u32 p, %5, %6, p;"
                        "  mov.u32 %0, %1;"
                        "  @p add.u32 %0, %0, r0;"
                        "}"
                        : "=r"(input) : "r"(input), "r"(OFFSET), "r"(SHFL_C), "r"(input), "r"((lane_id + OFFSET) * VALID_PER_LANE), "r"(valid));
                }
                else
                {
                    asm(
                        "{"
                        "  .reg .u32 r0;"
                        "  .reg .pred p;"
                        "  shfl.down.b32 r0|p, %1, %2, %3;"
                        "  @p add.u32 r0, r0, %4;"
                        "  mov.u32 %0, r0;"
                        "}"
                        : "=r"(input) : "r"(input), "r"(OFFSET), "r"(SHFL_C), "r"(input));
                }
            }

            return input;
        }


        /// Reduction (specialized for float)
        template <
            bool    FULL_TILE,
            int     VALID_PER_LANE>
        static __device__ __forceinline__ float Sum(
            SmemStorage         &smem_storage,      ///< [in] Shared reference to opaque SmemStorage layout
            float               input,              ///< [in] Calling thread's input
            const unsigned int  &valid)             ///< [in] Number of valid lanes in the calling thread's logical warp (may be less than \p LOGICAL_WARP_THREADS)
        {
            unsigned int lane_id = (WARPS == 1) ? threadIdx.x : (threadIdx.x & (LOGICAL_WARP_THREADS - 1));

            // Iterate reduction steps
            #pragma unroll
            for (int STEP = 0; STEP < STEPS; STEP++)
            {
                const int OFFSET = 1 << STEP;

                if (!FULL_TILE)
                {
                    asm(
                        "{"
                        "  .reg .f32 r0;"
                        "  .reg .pred p;"
                        "  shfl.down.b32 r0|p, %1, %2, %3;"
                        "  setp.lt.and.u32 p, %5, %6, p;"
                        "  mov.f32 %0, %1;"
                        "  @p add.f32 %0, %0, r0;"
                        "}"
                        : "=f"(input) : "f"(input), "r"(OFFSET), "r"(SHFL_C), "f"(input), "r"((lane_id + OFFSET) * VALID_PER_LANE), "r"(valid));
                }
                else
                {
                    asm(
                        "{"
                        "  .reg .f32 r0;"
                        "  .reg .pred p;"
                        "  shfl.down.b32 r0|p, %1, %2, %3;"
                        "  @p add.f32 r0, r0, %4;"
                        "  mov.u32 %0, r0;"
                        "}"
                        : "=f"(input) : "f"(input), "r"(OFFSET), "r"(SHFL_C), "f"(input));
                }
            }

            return input;
        }

        /// Summation
        template <
            bool        FULL_TILE,
            int         VALID_PER_LANE,
            typename    _T>
        static __device__ __forceinline__ _T Sum(
            SmemStorage         &smem_storage,      ///< [in] Shared reference to opaque SmemStorage layout
            _T                  input,              ///< [in] Calling thread's input
            const unsigned int  &valid)             ///< [in] Number of valid lanes in the calling thread's logical warp (may be less than \p LOGICAL_WARP_THREADS)
        {
            // Cast as unsigned int
            _T output;
            unsigned int    &uinput             = reinterpret_cast<unsigned int&>(input);
            unsigned int    &uoutput            = reinterpret_cast<unsigned int&>(output);

            uoutput = Sum<FULL_TILE, VALID_PER_LANE>(smem_storage, uinput, valid);
            return output;
        }

        /// Reduction
        template <
            bool            FULL_TILE,
            int             VALID_PER_LANE,
            typename        ReductionOp>
        static __device__ __forceinline__ T Reduce(
            SmemStorage         &smem_storage,      ///< [in] Shared reference to opaque SmemStorage layout
            T                   input,              ///< [in] Calling thread's input
            const unsigned int  &valid,             ///< [in] Number of valid lanes in the calling thread's logical warp (may be less than \p LOGICAL_WARP_THREADS)
            ReductionOp         reduction_op)       ///< [in] Binary reduction operator
        {
            unsigned int lane_id = (WARPS == 1) ? threadIdx.x : (threadIdx.x & (LOGICAL_WARP_THREADS - 1));

            T               temp;
            unsigned int    &utemp              = reinterpret_cast<unsigned int&>(temp);
            unsigned int    &uinput             = reinterpret_cast<unsigned int&>(input);

            // Iterate scan steps
            #pragma unroll
            for (int STEP = 0; STEP < STEPS; STEP++)
            {
                const int OFFSET = 1 << STEP;

                if (!FULL_TILE)
                {
                    // Grab addend from peer and set predicate if it's in valid-item range
                    const int OFFSET = 1 << STEP;
                    asm(
                        "{"
                        "  .reg .pred p;"
                        "  shfl.down.b32 %0|p, %1, %2, %3;"
                        "  setp.lt.and.u32 p, %4, %5, p;"
                        : "=r"(utemp) : "r"(uinput), "r"(OFFSET), "r"(SHFL_C), "r"((lane_id + OFFSET) * VALID_PER_LANE), "r"(valid));

                    // Reduce
                    temp = reduction_op(input, temp);

                    asm(
                        "  selp.b32 %0, %1, %2, p;"
                        "}"
                        : "=r"(uinput) : "r"(utemp), "r"(uinput));
                }
                else
                {
                    asm(
                        "{"
                        "  .reg .pred p;"
                        "  shfl.down.b32 %0|p, %1, %2, %3;"
                        : "=r"(utemp) : "r"(uinput), "r"(OFFSET), "r"(SHFL_C));

                    // Reduce
                    temp = reduction_op(input, temp);

                    asm(
                        "  selp.b32 %0, %1, %2, p;"
                        "}"
                        : "=r"(uinput) : "r"(utemp), "r"(uinput));
                }
            }

            return input;
        }

    };


    /**
     * WarpReduce specialized for SMEM_REDUCE
     */
    template <int DUMMY>
    struct WarpReduceInternal<SMEM_REDUCE, DUMMY>
    {
        /// Constants
        enum
        {
            /// The number of warp scan steps
            STEPS = Log2<LOGICAL_WARP_THREADS>::VALUE,

            /// The number of threads in half a warp
            HALF_WARP_THREADS = 1 << (STEPS - 1),

            /// The number of shared memory elements per warp
            WARP_SMEM_ELEMENTS =  LOGICAL_WARP_THREADS + HALF_WARP_THREADS,

            /// Whether or not warp-synchronous reduction should be unguarded (i.e., the warp-reduction elements is a power of two
            WARP_SYNCHRONOUS_UNGUARDED = ((LOGICAL_WARP_THREADS & (LOGICAL_WARP_THREADS - 1)) == 0),
        };


        /// Shared memory storage layout type
        struct SmemStorage
        {
            /// WarpReduce layout: 1.5 warps-worth of elements for each warp.
            T warp_buffer[WARPS][WARP_SMEM_ELEMENTS];
        };

        /**
         * Reduction
         */
        template <
            bool                FULL_TILE,
            int                 VALID_PER_LANE,
            typename            ReductionOp>
        static __device__ __forceinline__ T Reduce(
            SmemStorage         &smem_storage,      ///< [in] Shared reference to opaque SmemStorage layout
            T                   input,              ///< [in] Calling thread's input
            const unsigned int  &valid,             ///< [in] Number of valid lanes in the calling thread's logical warp (may be less than \p LOGICAL_WARP_THREADS)
            ReductionOp         reduction_op)       ///< [in] Reduction operator
        {
            // Warp, thread-lane-IDs
            unsigned int warp_id = (WARPS == 1) ? 0 : (threadIdx.x / LOGICAL_WARP_THREADS);
            unsigned int lane_id = (WARPS == 1) ? threadIdx.x : (threadIdx.x & (LOGICAL_WARP_THREADS - 1));

            for (int STEP = 0; STEP < STEPS; STEP++)
            {
                const int OFFSET = 1 << STEP;

                // Share input into buffer
                ThreadStore<PTX_STORE_VS>(&smem_storage.warp_buffer[warp_id][lane_id], input);

                // Update input if addend is in range
                if ((FULL_TILE && WARP_SYNCHRONOUS_UNGUARDED) || ((lane_id + OFFSET) * VALID_PER_LANE < valid))
                {
                    T addend = ThreadLoad<PTX_LOAD_VS>(&smem_storage.warp_buffer[warp_id][lane_id + OFFSET]);
                    input = reduction_op(input, addend);
                }
            }

            return input;
        }


        /**
         * Summation
         */
        template <
            bool    FULL_TILE,
            int     VALID_PER_LANE>
        static __device__ __forceinline__ T Sum(
            SmemStorage         &smem_storage,      ///< [in] Shared reference to opaque SmemStorage layout
            T                   input,              ///< [in] Calling thread's input
            const unsigned int  &valid)             ///< [in] Number of valid lanes in the calling thread's logical warp (may be less than \p LOGICAL_WARP_THREADS)
        {
            return Reduce<FULL_TILE, VALID_PER_LANE>(smem_storage, input, valid, cub::Sum<T>());
        }


    };


    /** \endcond */     // INTERNAL

    typedef typename WarpReduceInternal<POLICY> Internal;

    /// Shared memory storage layout type for WarpReduce
    typedef typename Internal::SmemStorage _SmemStorage;


public:

    /// \smemstorage{WarpReduce}
    typedef _SmemStorage SmemStorage;


    /******************************************************************//**
     * \name Summation reductions
     *********************************************************************/
    //@{

    /**
     * \brief Computes a warp-wide sum for <em>warp-lane</em><sub>0</sub> in each active warp.
     *
     * The return value is undefined in threads other than thread<sub>0</sub>.
     *
     * \smemreuse
     */
    static __device__ __forceinline__ T Sum(
        SmemStorage         &smem_storage,      ///< [in] Shared reference to opaque SmemStorage layout
        T                   input)              ///< [in] Calling thread's input
    {
        return Internal::Sum<true, 1>(smem_storage, input, LOGICAL_WARP_THREADS);
    }

    /**
     * \brief Computes a warp-wide sum for <em>warp-lane</em><sub>0</sub> in each active warp.
     *
     * All threads in each logical warp must agree on the same value for \p valid_lanes.  Otherwise the result is undefined.
     *
     * The return value is undefined in threads other than <em>warp-lane</em><sub>0</sub>.
     *
     * \smemreuse
     */
    static __device__ __forceinline__ T Sum(
        SmemStorage         &smem_storage,          ///< [in] Shared reference to opaque SmemStorage layout
        T                   input,                  ///< [in] Calling thread's input
        const unsigned int  &valid_lanes)           ///< [in] Number of valid lanes in the calling thread's logical warp (may be less than \p LOGICAL_WARP_THREADS)
    {
        // Determine if we don't need bounds checking
        if (valid_lanes == LOGICAL_WARP_THREADS)
        {
            return Internal::Sum<true, 1>(smem_storage, input, valid_lanes);
        }
        else
        {
            return Internal::Sum<false, 1>(smem_storage, input, valid_lanes);
        }
    }


    //@}  end member group
    /******************************************************************//**
     * \name Generic reductions
     *********************************************************************/
    //@{

    /**
     * \brief Computes a warp-wide reduction for <em>warp-lane</em><sub>0</sub> in each active warp using the specified binary reduction functor.
     *
     * The return value is undefined in threads other than <em>warp-lane</em><sub>0</sub>.
     *
     * \smemreuse
     *
     * \tparam ReductionOp     <b>[inferred]</b> Binary scan operator type having member <tt>T operator()(const T &a, const T &b)</tt>
     */
    template <typename ReductionOp>
    static __device__ __forceinline__ T Reduce(
        SmemStorage         &smem_storage,      ///< [in] Shared reference to opaque SmemStorage layout
        T                   input,              ///< [in] Calling thread's input
        ReductionOp         reduction_op)       ///< [in] Binary reduction operator
    {
        return Internal::Reduce<true, 1>(smem_storage, input, LOGICAL_WARP_THREADS, reduction_op);
    }

    /**
     * \brief Computes a warp-wide reduction for <em>warp-lane</em><sub>0</sub> in each active warp using the specified binary reduction functor.
     *
     * All threads in each logical warp must agree on the same value for \p valid_lanes.  Otherwise the result is undefined.
     *
     * The return value is undefined in threads other than <em>warp-lane</em><sub>0</sub>.
     *
     * \smemreuse
     *
     * \tparam ReductionOp     <b>[inferred]</b> Binary scan operator type having member <tt>T operator()(const T &a, const T &b)</tt>
     */
    template <typename ReductionOp>
    static __device__ __forceinline__ T Reduce(
        SmemStorage         &smem_storage,      ///< [in] Shared reference to opaque SmemStorage layout
        T                   input,              ///< [in] Calling thread's input
        ReductionOp         reduction_op,       ///< [in] Binary reduction operator
        const unsigned int  &valid_lanes)       ///< [in] Number of valid lanes in the calling thread's logical warp (may be less than \p LOGICAL_WARP_THREADS)
    {
        // Determine if we don't need bounds checking
        if (valid_lanes == LOGICAL_WARP_THREADS)
        {
            return Internal::Reduce<true, 1>(smem_storage, input, valid_lanes, reduction_op);
        }
        else
        {
            return Internal::Reduce<false, 1>(smem_storage, input, valid_lanes, reduction_op);
        }
    }





    //@}  end member group
};

/** @} */       // end group WarpModule

}               // CUB namespace
CUB_NS_POSTFIX  // Optional outer namespace(s)
