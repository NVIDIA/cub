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
 * A <a href="http://en.wikipedia.org/wiki/Reduce_(higher-order_function)"><em>reduction</em></a> (or <em>fold</em>)
 * uses a binary combining operator to compute a single aggregate from a list of input elements.
 *
 * \par
 * For convenience, WarpReduce provides alternative entrypoints that differ by:
 * - Operator (generic reduction <em>vs.</em> summation of numeric types)
 * - Input validity (full warps <em>vs.</em> partially-full warps having some undefined elements)
 *
 * \tparam T                        The reduction input/output element type
 * \tparam ACTIVE_WARPS             <b>[optional]</b> The number of entrant "logical" warps performing concurrent warp reductions.  Default is 1.
 * \tparam LOGICAL_WARP_THREADS     <b>[optional]</b> The number of threads per "logical" warp (may be less than the number of hardware warp threads).  Default is the warp size of the targeted CUDA compute-capability (e.g., 32 threads for SM20).
 *
 * \par Usage Considerations
 * - Supports non-commutative reduction operators
 * - Supports "logical" warps smaller than the physical warp size (e.g., a logical warp of 8 threads)
 * - The number of entrant threads must be an multiple of \p LOGICAL_WARP_THREADS
 * - Warp reductions are concurrent if more than one warp is participating
 * - The warp-wide scalar reduction output is only considered valid in warp <em>lane</em><sub>0</sub>
 * - \smemreuse{WarpReduce::TempStorage}

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
 * #include <cub/cub.cuh>
 *
 * __global__ void SomeKernel(...)
 * {
 *     // Parameterize WarpReduce for 1 warp on type int
 *     typedef cub::WarpReduce<int, 1> WarpReduce;
 *
 *     // Opaque shared memory for WarpReduce
 *     __shared__ typename WarpReduce::TempStorage temp_storage;
 *
 *     // Perform a reduction of thread ranks in first warp
 *     if (threadIdx.x < 32)
 *     {
 *         int input = threadIdx.x;
 *         int output = WarpReduce::Sum(temp_storage, input);
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
    int         ACTIVE_WARPS                   = 1,
    int         LOGICAL_WARP_THREADS    = PtxArchProps::WARP_THREADS>
class WarpReduce
{
private:

    /******************************************************************************
     * Constants and typedefs
     ******************************************************************************/

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
    };

    /// Use shuffle-based reduction if (architecture is >= SM30) and (LOGICAL_WARP_THREADS is a power-of-two)
    static const WarpReducePolicy POLICY = ((CUB_PTX_ARCH >= 300) && POW_OF_TWO) ?
                                            SHFL_REDUCE :
                                            SMEM_REDUCE;


    #ifndef DOXYGEN_SHOULD_SKIP_THIS    // Do not document


    /******************************************************************************
     * Algorithmic variants
     ******************************************************************************/

    /**
     * SHFL_REDUCE algorithmic variant
     */
    template <int _ALGORITHM, int DUMMY = 0>
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
        typedef NullType TempStorage;


        // Thread fields
        TempStorage     &temp_storage;
        int             warp_id;
        int             lane_id;


        /// Constructor
        __device__ __forceinline__ WarpReduceInternal(
            TempStorage &temp_storage,
            int warp_id,
            int lane_id)
        :
            temp_storage(temp_storage),
            warp_id(warp_id),
            lane_id(lane_id)
        {}


        /******************************************************************************
         * Summation specializations
         ******************************************************************************/

        /// Summation (upcast input to uint32)
        template <
            bool                FULL_TILE,
            int                 FOLDED_ITEMS_PER_LANE>
        __device__ __forceinline__ T Sum(
            T                   input,              ///< [in] Calling thread's input
            int                 valid_items,        ///< [in] Total number of valid items in the calling thread's logical warp (may be less than \p LOGICAL_WARP_THREADS)
            Int2Type<true>      upcast_to_uint32)
        {
            unsigned int &input_alias = reinterpret_cast<unsigned int &>(input);

            // Iterate reduction steps
            #pragma unroll
            for (int STEP = 0; STEP < STEPS; STEP++)
            {
                const int OFFSET = 1 << STEP;

                if (FULL_TILE)
                {
                    asm(
                        "{"
                        "  .reg .u32 r0;"
                        "  .reg .pred p;"
                        "  shfl.down.b32 r0|p, %1, %2, %3;"
                        "  @p add.u32 r0, r0, %4;"
                        "  mov.u32 %0, r0;"
                        "}"
                        : "=r"(input_alias) : "r"(input_alias), "r"(OFFSET), "r"(SHFL_C), "r"(input_alias));
                }
                else
                {
                    asm(
                        "{"
                        "  .reg .u32 r0;"
                        "  .reg .pred p;"
                        "  shfl.down.b32 r0|p, %1, %2, %3;"
                        "  setp.lt.and.u32 p, %5, %6, p;"
                        "  mov.u32 %0, %1;"
                        "  @p add.u32 %0, %1, r0;"
                        "}"
                        : "=r"(input_alias) : "r"(input_alias), "r"(OFFSET), "r"(SHFL_C), "r"(input_alias), "r"((lane_id + OFFSET) * FOLDED_ITEMS_PER_LANE), "r"(valid_items));
                }
            }

            return input;
        }


        /// Summation (cast input as array of uint32)
        template <
            bool                FULL_TILE,
            int                 FOLDED_ITEMS_PER_LANE>
        __device__ __forceinline__ T Sum(
            T                   input,              ///< [in] Calling thread's input
            int                 valid_items,        ///< [in] Total number of valid items in the calling thread's logical warp (may be less than \p LOGICAL_WARP_THREADS)
            Int2Type<false>     upcast_to_uint32)
        {
            return Reduce<FULL_TILE, FOLDED_ITEMS_PER_LANE>(input, valid_items, cub::Sum<T>());
        }


        /// Summation (float)
        template <
            bool                FULL_TILE,
            int                 FOLDED_ITEMS_PER_LANE>
        __device__ __forceinline__ float Sum(
            float               input,              ///< [in] Calling thread's input
            int                 valid_items)        ///< [in] Total number of valid items in the calling thread's logical warp (may be less than \p LOGICAL_WARP_THREADS)
        {
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
                        : "=f"(input) : "f"(input), "r"(OFFSET), "r"(SHFL_C), "f"(input), "r"((lane_id + OFFSET) * FOLDED_ITEMS_PER_LANE), "r"(valid_items));
                }
                else
                {
                    asm(
                        "{"
                        "  .reg .f32 r0;"
                        "  .reg .pred p;"
                        "  shfl.down.b32 r0|p, %1, %2, %3;"
                        "  @p add.f32 r0, r0, %4;"
                        "  mov.f32 %0, r0;"
                        "}"
                        : "=f"(input) : "f"(input), "r"(OFFSET), "r"(SHFL_C), "f"(input));
                }
            }

            return input;
        }

        /// Summation (generic)
        template <
            bool                FULL_TILE,
            int                 FOLDED_ITEMS_PER_LANE,
            typename            _T>
        __device__ __forceinline__ _T Sum(
            _T                  input,              ///< [in] Calling thread's input
            int                 valid_items)        ///< [in] Total number of valid items in the calling thread's logical warp (may be less than \p LOGICAL_WARP_THREADS)
        {
            Int2Type<(Traits<_T>::PRIMITIVE) && (sizeof(_T) <= sizeof(unsigned int))> upcast_to_uint32;

            return Sum<FULL_TILE, FOLDED_ITEMS_PER_LANE>(input, valid_items, upcast_to_uint32);
        }


        /******************************************************************************
         * Reduction specializations
         ******************************************************************************/

        /// Reduction
        template <
            bool            FULL_TILE,
            int             FOLDED_ITEMS_PER_LANE,
            typename        ReductionOp>
        __device__ __forceinline__ T Reduce(
            T               input,              ///< [in] Calling thread's input
            int             valid_items,        ///< [in] Total number of valid items in the calling thread's logical warp (may be less than \p LOGICAL_WARP_THREADS)
            ReductionOp     reduction_op)       ///< [in] Binary reduction operator
        {
            typedef typename WordAlignment<T>::ShuffleWord ShuffleWord;

            const int       WORDS           = (sizeof(T) + sizeof(ShuffleWord) - 1) / sizeof(ShuffleWord);
            T               temp;
            ShuffleWord     *temp_alias     = reinterpret_cast<ShuffleWord *>(&temp);
            ShuffleWord     *input_alias    = reinterpret_cast<ShuffleWord *>(&input);

            // Iterate scan steps
            #pragma unroll
            for (int STEP = 0; STEP < STEPS; STEP++)
            {
                // Grab addend from peer
                const int OFFSET = 1 << STEP;

                #pragma unroll
                for (int WORD = 0; WORD < WORDS; ++WORD)
                {
                    unsigned int shuffle_word = input_alias[WORD];
                    asm(
                        "  shfl.down.b32 %0, %1, %2, %3;"
                        : "=r"(shuffle_word) : "r"(shuffle_word), "r"(OFFSET), "r"(SHFL_C));
                    temp_alias[WORD] = (ShuffleWord) shuffle_word;
                }

                // Select reduction if valid
                if (FULL_TILE)
                {
                    if (lane_id < LOGICAL_WARP_THREADS - OFFSET)
                        input = reduction_op(input, temp);
                }
                else
                {
                    if (((lane_id + OFFSET) * FOLDED_ITEMS_PER_LANE) < valid_items)
                        input = reduction_op(input, temp);
                }
            }

            return input;

        }


        /// Segmented reduction
        template <
            bool            HEAD_SEGMENTED,
            typename        Flag,
            typename        ReductionOp>
        __device__ __forceinline__ T SegmentedReduce(
            T               input,              ///< [in] Calling thread's input
            Flag            flag,               ///< [in] Whether or not the current lane is a segment head/tail
            ReductionOp     reduction_op)       ///< [in] Binary reduction operator
        {
            typedef typename WordAlignment<T>::ShuffleWord ShuffleWord;

            const int       WORDS           = (sizeof(T) + sizeof(ShuffleWord) - 1) / sizeof(ShuffleWord);
            T               temp;
            ShuffleWord     *temp_alias     = reinterpret_cast<ShuffleWord *>(&temp);
            ShuffleWord     *input_alias    = reinterpret_cast<ShuffleWord *>(&input);

            // Get the start flags for each thread in the warp.
            int warp_flags = __ballot(flag);

            if (!HEAD_SEGMENTED)
                warp_flags <<= 1;

            // Keep bits above the current thread.
            warp_flags &= LaneMaskGt();

            // Accommodate packing of multiple logical warps in a single physical warp
            if ((ACTIVE_WARPS > 1) && (LOGICAL_WARP_THREADS < 32))
                warp_flags >>= (warp_id * LOGICAL_WARP_THREADS);

            // Find next flag
            int next_flag = __clz(__brev(warp_flags));

            // Clip the next segment at the warp boundary if necessary
            if (LOGICAL_WARP_THREADS != 32)
                next_flag = CUB_MIN(next_flag, LOGICAL_WARP_THREADS);

            // Iterate scan steps
            #pragma unroll
            for (int STEP = 0; STEP < STEPS; STEP++)
            {
                const int OFFSET = 1 << STEP;

                // Grab addend from peer
                #pragma unroll
                for (int WORD = 0; WORD < WORDS; ++WORD)
                {
                    unsigned int shuffle_word = input_alias[WORD];
                    asm(
                        "  shfl.down.b32 %0, %1, %2, %3;"
                        : "=r"(shuffle_word) : "r"(shuffle_word), "r"(OFFSET), "r"(SHFL_C));
                    temp_alias[WORD] = (ShuffleWord) shuffle_word;
                }

                // Select reduction if valid
                if (OFFSET < next_flag - lane_id)
                    input = reduction_op(input, temp);
            }

            return input;
        }

    };


    /**
     * SMEM_REDUCE algorithmic variant
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
        };


        /// Shared memory storage layout type (1.5 warps-worth of elements for each warp)
        typedef T TempStorage[ACTIVE_WARPS][WARP_SMEM_ELEMENTS];


        // Thread fields
        TempStorage     &temp_storage;
        int             warp_id;
        int             lane_id;


        /// Constructor
        __device__ __forceinline__ WarpReduceInternal(
            TempStorage     &temp_storage,
            int             warp_id,
            int             lane_id)
        :
            temp_storage(temp_storage),
            warp_id(warp_id),
            lane_id(lane_id)
        {}


        /**
         * Reduction
         */
        template <
            bool                FULL_TILE,
            int                 FOLDED_ITEMS_PER_LANE,
            typename            ReductionOp>
        __device__ __forceinline__ T Reduce(
            T                   input,              ///< [in] Calling thread's input
            int                 valid_items,        ///< [in] Total number of valid items in the calling thread's logical warp (may be less than \p LOGICAL_WARP_THREADS)
            ReductionOp         reduction_op)       ///< [in] Reduction operator
        {
            for (int STEP = 0; STEP < STEPS; STEP++)
            {
                const int OFFSET = 1 << STEP;

                // Share input into buffer
                ThreadStore<STORE_VOLATILE>(&temp_storage[warp_id][lane_id], input);

                // Update input if addend is in range
                if ((FULL_TILE && POW_OF_TWO) || ((lane_id + OFFSET) * FOLDED_ITEMS_PER_LANE < valid_items))
                {
                    T addend = ThreadLoad<LOAD_VOLATILE>(&temp_storage[warp_id][lane_id + OFFSET]);
                    input = reduction_op(input, addend);
                }
            }

            return input;
        }


        /**
         * Segmented reduction
         */
        template <
            bool            HEAD_SEGMENTED,
            typename        Flag,
            typename        ReductionOp>
        __device__ __forceinline__ T SegmentedReduce(
            T               input,              ///< [in] Calling thread's input
            Flag            flag,               ///< [in] Whether or not the current lane is a segment head/tail
            ReductionOp     reduction_op)       ///< [in] Reduction operator
        {
            // Get the start flags for each thread in the warp.
            int warp_flags = __ballot(flag);

            if (!HEAD_SEGMENTED)
                warp_flags <<= 1;

            // Keep bits above the current thread.
            warp_flags &= LaneMaskGt();

            // Accommodate packing of multiple logical warps in a single physical warp
            if ((ACTIVE_WARPS > 1) && (LOGICAL_WARP_THREADS < 32))
                warp_flags >>= (warp_id * LOGICAL_WARP_THREADS);

            // Find next flag
            int next_flag = __clz(__brev(warp_flags));

            // Clip the next segment at the warp boundary if necessary
            if (LOGICAL_WARP_THREADS != 32)
                next_flag = CUB_MIN(next_flag, LOGICAL_WARP_THREADS);

            for (int STEP = 0; STEP < STEPS; STEP++)
            {
                const int OFFSET = 1 << STEP;

                // Share input into buffer
                ThreadStore<STORE_VOLATILE>(&temp_storage[warp_id][lane_id], input);

                // Update input if addend is in range
                if (OFFSET < next_flag - lane_id)
                {
                    T addend = ThreadLoad<LOAD_VOLATILE>(&temp_storage[warp_id][lane_id + OFFSET]);
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
            int     FOLDED_ITEMS_PER_LANE>
        __device__ __forceinline__ T Sum(
            T                   input,              ///< [in] Calling thread's input
            int                 valid_items)        ///< [in] Total number of valid items in the calling thread's logical warp (may be less than \p LOGICAL_WARP_THREADS)
        {
            return Reduce<FULL_TILE, FOLDED_ITEMS_PER_LANE>(input, valid_items, cub::Sum<T>());
        }

    };


    /******************************************************************************
     * Type definitions
     ******************************************************************************/

public:

    typedef WarpReduceInternal<POLICY> InternalWarpReduce;


    #endif // DOXYGEN_SHOULD_SKIP_THIS


    /// \smemstorage{WarpReduce}
    typedef typename InternalWarpReduce::TempStorage TempStorage;


private:

    /******************************************************************************
     * Utility methods
     ******************************************************************************/

    /// Internal storage allocator
    __device__ __forceinline__ TempStorage& PrivateStorage()
    {
        __shared__ TempStorage private_storage;
        return private_storage;
    }


    /******************************************************************************
     * Thread fields
     ******************************************************************************/

    /// Shared storage reference
    TempStorage &temp_storage;

    /// Warp ID
    int warp_id;

    /// Lane ID
    int lane_id;

public:

    /******************************************************************//**
     * \name Collective construction
     *********************************************************************/

    /**
     * \brief Collective constructor for 1D thread blocks using a private static allocation of shared memory as temporary storage.  Logical warp and lane identifiers are constructed from <tt>threadIdx.x</tt>.
     */
    __device__ __forceinline__ WarpReduce()
    :
        temp_storage(PrivateStorage()),
        warp_id((ACTIVE_WARPS == 1) ?
            0 :
            threadIdx.x / LOGICAL_WARP_THREADS),
        lane_id(((ACTIVE_WARPS == 1) || (LOGICAL_WARP_THREADS == PtxArchProps::WARP_THREADS)) ?
            LaneId() :
            threadIdx.x % LOGICAL_WARP_THREADS)
    {}


    /**
     * \brief Collective constructor for 1D thread blocks using the specified memory allocation as temporary storage.  Logical warp and lane identifiers are constructed from <tt>threadIdx.x</tt>.
     */
    __device__ __forceinline__ WarpReduce(
        TempStorage &temp_storage)             ///< [in] Reference to memory allocation having layout type TempStorage
    :
        temp_storage(temp_storage),
        warp_id((ACTIVE_WARPS == 1) ?
            0 :
            threadIdx.x / LOGICAL_WARP_THREADS),
        lane_id(((ACTIVE_WARPS == 1) || (LOGICAL_WARP_THREADS == PtxArchProps::WARP_THREADS)) ?
            LaneId() :
            threadIdx.x % LOGICAL_WARP_THREADS)
    {}


    /**
     * \brief Collective constructor using a private static allocation of shared memory as temporary storage.  Threads are identified using the given warp and lane identifiers.
     */
    __device__ __forceinline__ WarpReduce(
        int warp_id,                           ///< [in] A suitable warp membership identifier
        int lane_id)                           ///< [in] A lane identifier within the warp
    :
        temp_storage(PrivateStorage()),
        warp_id(warp_id),
        lane_id(lane_id)
    {}


    /**
     * \brief Collective constructor using the specified memory allocation as temporary storage.  Threads are identified using the given warp and lane identifiers.
     */
    __device__ __forceinline__ WarpReduce(
        TempStorage &temp_storage,             ///< [in] Reference to memory allocation having layout type TempStorage
        int warp_id,                           ///< [in] A suitable warp membership identifier
        int lane_id)                           ///< [in] A lane identifier within the warp
    :
        temp_storage(temp_storage),
        warp_id(warp_id),
        lane_id(lane_id)
    {}



    //@}  end member group
    /******************************************************************//**
     * \name Summation reductions
     *********************************************************************/
    //@{


    /**
     * \brief Computes a warp-wide sum in each active warp.  The output is valid in warp <em>lane</em><sub>0</sub>.
     *
     * The return value is undefined in threads other than thread<sub>0</sub>.
     *
     * \smemreuse
     */
    __device__ __forceinline__ T Sum(
        T                   input)              ///< [in] Calling thread's input
    {
        return InternalWarpReduce(temp_storage, warp_id, lane_id).Sum<true, 1>(input, LOGICAL_WARP_THREADS);
    }

    /**
     * \brief Computes a partially-full warp-wide sum in each active warp.  The output is valid in warp <em>lane</em><sub>0</sub>.
     *
     * All threads in each logical warp must agree on the same value for \p valid_items.  Otherwise the result is undefined.
     *
     * The return value is undefined in threads other than warp <em>lane</em><sub>0</sub>.
     *
     * \smemreuse
     */
    __device__ __forceinline__ T Sum(
        T                   input,                  ///< [in] Calling thread's input
        int                 valid_items)        ///< [in] Total number of valid items in the calling thread's logical warp (may be less than \p LOGICAL_WARP_THREADS)
    {
        // Determine if we don't need bounds checking
        if (valid_items >= LOGICAL_WARP_THREADS)
        {
            return InternalWarpReduce(temp_storage, warp_id, lane_id).Sum<true, 1>(input, valid_items);
        }
        else
        {
            return InternalWarpReduce(temp_storage, warp_id, lane_id).Sum<false, 1>(input, valid_items);
        }
    }


    /**
     * \brief Computes a segmented sum in each active warp where segments are defined by head-flags.  The sum of each segment is returned to the first lane in that segment (which always includes <em>lane</em><sub>0</sub>).
     *
     * \smemreuse
     *
     * \tparam ReductionOp     <b>[inferred]</b> Binary scan operator type having member <tt>T operator()(const T &a, const T &b)</tt>
     */
    template <
        typename            Flag>
    __device__ __forceinline__ T HeadSegmentedSum(
        T                   input,              ///< [in] Calling thread's input
        Flag                head_flag)          ///< [in] Head flag denoting whether or not \p input is the start of a new segment
    {
        return HeadSegmentedReduce(input, head_flag, cub::Sum<T>());
    }


    /**
     * \brief Computes a segmented sum in each active warp where segments are defined by tail-flags.  The sum of each segment is returned to the first lane in that segment (which always includes <em>lane</em><sub>0</sub>).
     *
     * \smemreuse
     *
     * \tparam ReductionOp     <b>[inferred]</b> Binary scan operator type having member <tt>T operator()(const T &a, const T &b)</tt>
     */
    template <
        typename            Flag>
    __device__ __forceinline__ T TailSegmentedSum(
        T                   input,              ///< [in] Calling thread's input
        Flag                tail_flag)          ///< [in] Head flag denoting whether or not \p input is the start of a new segment
    {
        return TailSegmentedReduce(input, tail_flag, cub::Sum<T>());
    }



    //@}  end member group
    /******************************************************************//**
     * \name Generic reductions
     *********************************************************************/
    //@{

    /**
     * \brief Computes a warp-wide reduction in each active warp using the specified binary reduction functor.  The output is valid in warp <em>lane</em><sub>0</sub>.
     *
     * The return value is undefined in threads other than warp <em>lane</em><sub>0</sub>.
     *
     * \smemreuse
     *
     * \tparam ReductionOp     <b>[inferred]</b> Binary scan operator type having member <tt>T operator()(const T &a, const T &b)</tt>
     */
    template <typename ReductionOp>
    __device__ __forceinline__ T Reduce(
        T                   input,              ///< [in] Calling thread's input
        ReductionOp         reduction_op)       ///< [in] Binary reduction operator
    {
        return InternalWarpReduce(temp_storage, warp_id, lane_id).Reduce<true, 1>(input, LOGICAL_WARP_THREADS, reduction_op);
    }

    /**
     * \brief Computes a partially-full warp-wide reduction in each active warp using the specified binary reduction functor.  The output is valid in warp <em>lane</em><sub>0</sub>.
     *
     * All threads in each logical warp must agree on the same value for \p valid_items.  Otherwise the result is undefined.
     *
     * The return value is undefined in threads other than warp <em>lane</em><sub>0</sub>.
     *
     * \smemreuse
     *
     * \tparam ReductionOp     <b>[inferred]</b> Binary scan operator type having member <tt>T operator()(const T &a, const T &b)</tt>
     */
    template <typename ReductionOp>
    __device__ __forceinline__ T Reduce(
        T                   input,              ///< [in] Calling thread's input
        ReductionOp         reduction_op,       ///< [in] Binary reduction operator
        int                 valid_items)        ///< [in] Total number of valid items in the calling thread's logical warp (may be less than \p LOGICAL_WARP_THREADS)
    {
        // Determine if we don't need bounds checking
        if (valid_items >= LOGICAL_WARP_THREADS)
        {
            return InternalWarpReduce(temp_storage, warp_id, lane_id).Reduce<true, 1>(input, valid_items, reduction_op);
        }
        else
        {
            return InternalWarpReduce(temp_storage, warp_id, lane_id).Reduce<false, 1>(input, valid_items, reduction_op);
        }
    }


    /**
     * \brief Computes a segmented reduction in each active warp where segments are defined by head-flags.  The reduction of each segment is returned to the first lane in that segment (which always includes <em>lane</em><sub>0</sub>).
     *
     * \smemreuse
     *
     * \tparam ReductionOp     <b>[inferred]</b> Binary scan operator type having member <tt>T operator()(const T &a, const T &b)</tt>
     */
    template <
        typename            ReductionOp,
        typename            Flag>
    __device__ __forceinline__ T HeadSegmentedReduce(
        T                   input,              ///< [in] Calling thread's input
        Flag                head_flag,          ///< [in] Head flag denoting whether or not \p input is the start of a new segment
        ReductionOp         reduction_op)       ///< [in] Reduction operator
    {
        return InternalWarpReduce(temp_storage, warp_id, lane_id).template SegmentedReduce<true>(input, head_flag, reduction_op);
    }


    /**
     * \brief Computes a segmented reduction in each active warp where segments are defined by tail-flags.  The reduction of each segment is returned to the first lane in that segment (which always includes <em>lane</em><sub>0</sub>).
     *
     * \smemreuse
     *
     * \tparam ReductionOp     <b>[inferred]</b> Binary scan operator type having member <tt>T operator()(const T &a, const T &b)</tt>
     */
    template <
        typename            ReductionOp,
        typename            Flag>
    __device__ __forceinline__ T TailSegmentedReduce(
        T                   input,              ///< [in] Calling thread's input
        Flag                tail_flag,          ///< [in] Tail flag denoting whether or not \p input is the end of the current segment
        ReductionOp         reduction_op)       ///< [in] Reduction operator
    {
        return InternalWarpReduce(temp_storage, warp_id, lane_id).template SegmentedReduce<false>(input, tail_flag, reduction_op);
    }



    //@}  end member group
};

/** @} */       // end group WarpModule

}               // CUB namespace
CUB_NS_POSTFIX  // Optional outer namespace(s)
