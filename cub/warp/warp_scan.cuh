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
 * cub::WarpScan provides variants of parallel prefix scan across CUDA warps.
 */

#pragma once

#include "../thread/thread_load.cuh"
#include "../thread/thread_store.cuh"
#include "../util_arch.cuh"
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
 * \brief WarpScan provides variants of parallel prefix scan across CUDA warps.  ![](warp_scan_logo.png)
 *
 * \par Overview
 * Given a list of input elements and a binary reduction operator, a [<em>prefix scan</em>](http://en.wikipedia.org/wiki/Prefix_sum)
 * produces an output list where each element is computed to be the reduction
 * of the elements occurring earlier in the input list.  <em>Prefix sum</em>
 * connotes a prefix scan with the addition operator. The term \em inclusive indicates
 * that the <em>i</em><sup>th</sup> output reduction incorporates the <em>i</em><sup>th</sup> input.
 * The term \em exclusive indicates the <em>i</em><sup>th</sup> input is not incorporated into
 * the <em>i</em><sup>th</sup> output reduction.
 *
 * \par
 * For convenience, WarpScan provides alternative entrypoints that differ by:
 * - Operator (generic scan <em>vs.</em> prefix sum of numeric types)
 * - Output ordering (inclusive <em>vs.</em> exclusive)
 * - Warp-wide prefix (identity <em>vs.</em> call-back functor)
 * - What is computed (scanned elements only <em>vs.</em> scanned elements and the total aggregate)
 *
 * \tparam T                        The scan input/output element type
 * \tparam WARPS                    <b>[optional]</b> The number of "logical" warps performing concurrent warp scans. Default is 1.
 * \tparam LOGICAL_WARP_THREADS     <b>[optional]</b> The number of threads per "logical" warp (may be less than the number of hardware warp threads).  Default is the warp size associated with the CUDA Compute Capability targeted by the compiler (e.g., 32 threads for SM20).
 *
 * \par Usage Considerations
 * - Supports non-commutative scan operators
 * - Supports "logical" warps smaller than the physical warp size (e.g., a logical warp of 8 threads)
 * - Warp scans are concurrent if more than one warp is participating
 * - \smemreuse{WarpScan::SmemStorage}

 * \par Performance Considerations
 * - Uses special instructions when applicable (e.g., warp \p SHFL)
 * - Uses synchronization-free communication between warp lanes when applicable
 * - Zero bank conflicts for most types.
 * - Computation is slightly more efficient (i.e., having lower instruction overhead) for:
 *     - Prefix sum variants (vs. generic scan)
 *     - Exclusive variants (vs. inclusive)
 *     - Basic scan variants that don't require scalar inputs and outputs (e.g., \p warp_prefix_op and \p warp_aggregate)
 *     - Scan parameterizations where \p T is a built-in C++ primitive or CUDA vector type (e.g.,
 *       \p short, \p int2, \p double, \p float2, etc.)
 *     - Scan parameterizations where \p LOGICAL_WARP_THREADS is a multiple of the architecture's warp size
 *
 * \par Algorithm
 * These parallel prefix scan variants implement a warp-synchronous
 * Kogge-Stone algorithm having <em>O</em>(log<em>n</em>)
 * steps and <em>O</em>(<em>n</em>log<em>n</em>) work complexity,
 * where <em>n</em> = \p LOGICAL_WARP_THREADS (which defaults to the warp
 * size associated with the CUDA Compute Capability targeted by the compiler).
 * <br><br>
 * \image html kogge_stone_scan.png
 * <div class="centercaption">Data flow within a 16-thread Kogge-Stone scan construction.  Junctions represent binary operators.</div>
 * <br>
 *
 * \par Examples
 * <em>Example 1.</em> Perform a simple exclusive prefix sum for one warp
 * \code
 * #include <cub/cub.cuh>
 *
 * __global__ void SomeKernel(...)
 * {
 *     // Parameterize WarpScan for 1 warp on type int
 *     typedef cub::WarpScan<int> WarpScan;
 *
 *     // Opaque shared memory for WarpScan
 *     __shared__ typename WarpScan::SmemStorage smem_storage;
 *
 *     // Perform prefix sum of threadIds in first warp
 *     if (threadIdx.x < 32)
 *     {
 *         int input = threadIdx.x;
 *         int output;
 *         WarpScan::ExclusiveSum(smem_storage, input, output);
 *
 *         printf("tid(%d) output(%d)\n\n", threadIdx.x, output);
 *     }
 * \endcode
 * Printed output:
 * \code
 * tid(0) output(0)
 * tid(1) output(0)
 * tid(2) output(1)
 * tid(3) output(3)
 * tid(4) output(6)
 * ...
 * tid(31) output(465)
 * \endcode
 *
 * \par
 * <em>Example 2.</em> Use a single warp to iteratively compute an exclusive prefix sum over a larger input using a prefix functor to maintain a running total between scans.
 *      \code
 *      #include <cub/cub.cuh>
 *
 *      // Stateful functor that maintains a running prefix that can be applied to
 *      // consecutive scan operations.
 *      struct WarpPrefixOp
 *      {
 *          // Running prefix
 *          int running_total;
 *
 *          // Functor constructor
 *          __device__ WarpPrefixOp(int running_total) : running_total(running_total) {}
 *
 *          // Functor operator.  Lane-0 produces a value for seeding the warp-wide scan given
 *          // the local aggregate.
 *          __device__ int operator()(int warp_aggregate)
 *          {
 *              int old_prefix = running_total;
 *              running_total += warp_aggregate;
 *              return old_prefix;
 *          }
 *      }
 *
 *      __global__ void SomeKernel(int *d_data, int num_elements)
 *      {
 *          // Parameterize WarpScan for 1 warp on type int
 *          typedef cub::WarpScan<int> WarpScan;
 *
 *          // Opaque shared memory for WarpScan
 *          __shared__ typename WarpScan::SmemStorage smem_storage;
 *
 *          // The first warp iteratively computes a prefix sum over d_data
 *          if (threadIdx.x < 32)
 *          {
 *              // Running total
 *              WarpPrefixOp prefix_op(0);
 *
 *              // Iterate in strips of 32 items
 *              for (int warp_offset = 0; warp_offset < num_elements; warp_offset += 32)
 *              {
 *                  // Read item
 *                  int datum = d_data[warp_offset + threadIdx.x];
 *
 *                  // Scan the tile of items
 *                  int tile_aggregate;
 *                  WarpScan::ExclusiveSum(smem_storage, datum, datum,
 *                      tile_aggregate, prefix_op);
 *
 *                  // Write item
 *                  d_data[warp_offset + threadIdx.x] = datum;
 *              }
 *          }
 *      \endcode
 */
template <
    typename    T,
    int         WARPS                   = 1,
    int         LOGICAL_WARP_THREADS    = PtxArchProps::WARP_THREADS>
class WarpScan
{
private:

     /******************************************************************************
      * Constants and typedefs
      ******************************************************************************/

    /// Constants
    enum
    {
        POW_OF_TWO = ((LOGICAL_WARP_THREADS & (LOGICAL_WARP_THREADS - 1)) == 0),
    };

    /// WarpScan algorithmic variants (would use an enum, but it causes GCC crash as of CUDA5)
    static const int SHFL_SCAN = 0;          // Warp-synchronous SHFL-based scan
    static const int SMEM_SCAN = 1;          // Warp-synchronous smem-based scan


    /// Use SHFL_SCAN if (architecture is >= SM30) and (T is a primitive) and (T is 4-bytes or smaller) and (LOGICAL_WARP_THREADS is a power-of-two)
    static const int POLICY = ((CUB_PTX_ARCH >= 300) && Traits<T>::PRIMITIVE && (sizeof(T) <= 4) && POW_OF_TWO) ?
        SHFL_SCAN :
        SMEM_SCAN;



    #ifndef DOXYGEN_SHOULD_SKIP_THIS    // Do not document


    /******************************************************************************
     * Algorithmic variants
     ******************************************************************************/

    /**
     * SHFL_SCAN algorithmic variant
     */
    template <int _ALGORITHM, int DUMMY = 0>
    struct WarpScanInternal
    {
        /// Constants
        enum
        {
            /// The number of warp scan steps
            STEPS = Log2<LOGICAL_WARP_THREADS>::VALUE,

            // The 5-bit SHFL mask for logically splitting warps into sub-segments starts 8-bits up
            SHFL_MASK = ((-1 << STEPS) & 31) << 8,
        };

        /// Shared memory storage layout type
        typedef NullType SmemStorage;


        /// Broadcast
        static __device__ __forceinline__ T Broadcast(
            SmemStorage     &smem_storage,      ///< [in] Reference to shared memory allocation having layout type SmemStorage
            T               input,              ///< [in] The value to broadcast
            unsigned int    src_lane)           ///< [in] Which warp lane is to do the broacasting
        {
            T               output;
            unsigned int    &uinput     = reinterpret_cast<unsigned int&>(input);
            unsigned int    &uoutput    = reinterpret_cast<unsigned int&>(output);

            asm("shfl.idx.b32 %0, %1, %2, %3;"
                : "=r"(uoutput) : "r"(uinput), "r"(src_lane), "r"(LOGICAL_WARP_THREADS - 1));

            return output;
        }

        /// Inclusive prefix sum with aggregate (specialized for unsigned int)
        static __device__ __forceinline__ void InclusiveSum(
            SmemStorage     &smem_storage,      ///< [in] Reference to shared memory allocation having layout type SmemStorage
            unsigned int    input,              ///< [in] Calling thread's input item.
            unsigned int    &output,            ///< [out] Calling thread's output item.  May be aliased with \p input.
            unsigned int    &warp_aggregate)    ///< [out] Warp-wide aggregate reduction of input items.
        {
            // Iterate scan steps
            #pragma unroll
            for (int STEP = 0; STEP < STEPS; STEP++)
            {
                asm(
                    "{"
                    "  .reg .u32 r0;"
                    "  .reg .pred p;"
                    "  shfl.up.b32 r0|p, %1, %2, %3;"
                    "  @p add.u32 r0, r0, %4;"
                    "  mov.u32 %0, r0;"
                    "}"
                    : "=r"(input) : "r"(input), "r"(1 << STEP), "r"(SHFL_MASK), "r"(input));
            }

            // Grab aggregate from last warp lane
            warp_aggregate = Broadcast(smem_storage, input, LOGICAL_WARP_THREADS - 1);

            // Update output
            output = input;
        }


        /// Inclusive prefix sum with aggregate (specialized for float)
        static __device__ __forceinline__ void InclusiveSum(
            SmemStorage     &smem_storage,      ///< [in] Reference to shared memory allocation having layout type SmemStorage
            float           input,              ///< [in] Calling thread's input item.
            float           &output,            ///< [out] Calling thread's output item.  May be aliased with \p input.
            float           &warp_aggregate)    ///< [out] Warp-wide aggregate reduction of input items.
        {
            // Iterate scan steps
            #pragma unroll
            for (int STEP = 0; STEP < STEPS; STEP++)
            {
                asm(
                    "{"
                    "  .reg .f32 r0;"
                    "  .reg .pred p;"
                    "  shfl.up.b32 r0|p, %1, %2, %3;"
                    "  @p add.f32 r0, r0, %4;"
                    "  mov.f32 %0, r0;"
                    "}"
                    : "=f"(input) : "f"(input), "r"(1 << STEP), "r"(SHFL_MASK), "f"(input));
            }

            // Grab aggregate from last warp lane
            warp_aggregate = Broadcast(smem_storage, input, LOGICAL_WARP_THREADS - 1);

            // Update output
            output = input;
        }

        /// Inclusive prefix sum with aggregate
        template <typename _T>
        static __device__ __forceinline__ void InclusiveSum(
            SmemStorage     &smem_storage,      ///< [in] Reference to shared memory allocation having layout type SmemStorage
            _T               input,             ///< [in] Calling thread's input item.
            _T               &output,           ///< [out] Calling thread's output item.  May be aliased with \p input.
            _T               &warp_aggregate)   ///< [out] Warp-wide aggregate reduction of input items.
        {
            unsigned int uinput = (unsigned int) input;
            unsigned int uoutput;
            unsigned int uwarp_aggregate;

            InclusiveSum(smem_storage, uinput, uoutput, uwarp_aggregate);

            warp_aggregate = (T) uwarp_aggregate;
            output = (T) uoutput;
        }

        /// Inclusive prefix sum
        static __device__ __forceinline__ void InclusiveSum(
            SmemStorage     &smem_storage,      ///< [in] Reference to shared memory allocation having layout type SmemStorage
            T               input,              ///< [in] Calling thread's input item.
            T               &output)            ///< [out] Calling thread's output item.  May be aliased with \p input.
        {
            T warp_aggregate;
            InclusiveSum(smem_storage, input, output, warp_aggregate);
        }


        /// Inclusive scan with aggregate
        template <typename ScanOp>
        static __device__ __forceinline__ void InclusiveScan(
            SmemStorage     &smem_storage,      ///< [in] Reference to shared memory allocation having layout type SmemStorage
            T               input,              ///< [in] Calling thread's input item.
            T               &output,            ///< [out] Calling thread's output item.  May be aliased with \p input.
            ScanOp          scan_op,            ///< [in] Binary scan operator
            T               &warp_aggregate)    ///< [out] Warp-wide aggregate reduction of input items.
        {
            T               temp;
            unsigned int    &utemp              = reinterpret_cast<unsigned int&>(temp);
            unsigned int    &uinput             = reinterpret_cast<unsigned int&>(input);

            // Iterate scan steps
            #pragma unroll
            for (int STEP = 0; STEP < STEPS; STEP++)
            {
                asm(
                    "{"
                    "  .reg .pred p;"
                    "  shfl.up.b32 %0|p, %1, %2, %3;"
                    : "=r"(utemp) : "r"(uinput), "r"(1 << STEP), "r"(SHFL_MASK));

                temp = scan_op(temp, input);

                asm(
                    "  selp.b32 %0, %1, %2, p;"
                    "}"
                    : "=r"(uinput) : "r"(utemp), "r"(uinput));
            }

            // Grab aggregate from last warp lane
            warp_aggregate = Broadcast(smem_storage, input, LOGICAL_WARP_THREADS - 1);

            // Update output
            output = input;
        }


        /// Inclusive scan
        template <typename ScanOp>
        static __device__ __forceinline__ void InclusiveScan(
            SmemStorage     &smem_storage,      ///< [in] Reference to shared memory allocation having layout type SmemStorage
            T               input,              ///< [in] Calling thread's input item.
            T               &output,            ///< [out] Calling thread's output item.  May be aliased with \p input.
            ScanOp          scan_op)            ///< [in] Binary scan operator
        {
            T warp_aggregate;
            InclusiveScan(smem_storage, input, output, scan_op, warp_aggregate);
        }

        /// Exclusive scan with aggregate
        template <typename ScanOp>
        static __device__ __forceinline__ void ExclusiveScan(
            SmemStorage     &smem_storage,      ///< [in] Reference to shared memory allocation having layout type SmemStorage
            T               input,              ///< [in] Calling thread's input item.
            T               &output,            ///< [out] Calling thread's output item.  May be aliased with \p input.
            const T         &identity,          ///< [in] Identity value
            ScanOp          scan_op,            ///< [in] Binary scan operator
            T               &warp_aggregate)    ///< [out] Warp-wide aggregate reduction of input items.
        {
            // Compute inclusive scan
            T inclusive;
            InclusiveScan(smem_storage, input, inclusive, scan_op, warp_aggregate);

            unsigned int    &uinclusive     = reinterpret_cast<unsigned int&>(inclusive);
            unsigned int    &uidentity      = reinterpret_cast<unsigned int&>(const_cast<T&>(identity));
            unsigned int    &uoutput        = reinterpret_cast<unsigned int&>(output);

            asm(
                "{"
                "  .reg .u32 r0;"
                "  .reg .pred p;"
                "  shfl.up.b32 r0|p, %1, 1, %2;"
                "  selp.b32 %0, r0, %3, p;"
                "}"
                : "=r"(uoutput) : "r"(uinclusive), "r"(SHFL_MASK), "r"(uidentity));
        }


        /// Exclusive scan
        template <typename ScanOp>
        static __device__ __forceinline__ void ExclusiveScan(
            SmemStorage     &smem_storage,      ///< [in] Reference to shared memory allocation having layout type SmemStorage
            T               input,              ///< [in] Calling thread's input item.
            T               &output,            ///< [out] Calling thread's output item.  May be aliased with \p input.
            const T         &identity,          ///< [in] Identity value
            ScanOp          scan_op)            ///< [in] Binary scan operator
        {
            T warp_aggregate;
            ExclusiveScan(smem_storage, input, output, identity, scan_op, warp_aggregate);
        }


        /// Exclusive scan with aggregate, without identity
        template <typename ScanOp>
        static __device__ __forceinline__ void ExclusiveScan(
            SmemStorage     &smem_storage,      ///< [in] Reference to shared memory allocation having layout type SmemStorage
            T               input,              ///< [in] Calling thread's input item.
            T               &output,            ///< [out] Calling thread's output item.  May be aliased with \p input.
            ScanOp          scan_op,            ///< [in] Binary scan operator
            T               &warp_aggregate)    ///< [out] Warp-wide aggregate reduction of input items.
        {
            // Compute inclusive scan
            T inclusive;
            InclusiveScan(smem_storage, input, inclusive, scan_op, warp_aggregate);

            unsigned int    &uinclusive     = reinterpret_cast<unsigned int&>(inclusive);
            unsigned int    &uoutput        = reinterpret_cast<unsigned int&>(output);

            asm(
                "shfl.up.b32 %0, %1, 1, %2;"
                : "=r"(uoutput) : "r"(uinclusive), "r"(SHFL_MASK));
        }


        /// Exclusive scan without identity
        template <typename ScanOp>
        static __device__ __forceinline__ void ExclusiveScan(
            SmemStorage     &smem_storage,      ///< [in] Reference to shared memory allocation having layout type SmemStorage
            T               input,              ///< [in] Calling thread's input item.
            T               &output,            ///< [out] Calling thread's output item.  May be aliased with \p input.
            ScanOp          scan_op)            ///< [in] Binary scan operator
        {
            T warp_aggregate;
            ExclusiveScan(smem_storage, input, output, scan_op, warp_aggregate);
        }
    };


    /**
     * SMEM_SCAN algorithmic variant
     */
    template <int DUMMY>
    struct WarpScanInternal<SMEM_SCAN, DUMMY>
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


        /// Shared memory storage layout type
        struct SmemStorage
        {
            /// Warpscan layout: 1.5 warps-worth of elements for each warp.
            T warp_scan[WARPS][WARP_SMEM_ELEMENTS];
        };


        /// Broadcast
        static __device__ __forceinline__ T Broadcast(
            SmemStorage     &smem_storage,      ///< [in] Reference to shared memory allocation having layout type SmemStorage
            T               input,              ///< [in] The value to broadcast
            unsigned int    src_lane)           ///< [in] Which warp lane is to do the broadcasting
        {
            unsigned int lane_id = (WARPS == 1) ? threadIdx.x : (threadIdx.x & (LOGICAL_WARP_THREADS - 1));
            unsigned int warp_id = (WARPS == 1) ? 0 : (threadIdx.x / LOGICAL_WARP_THREADS);

            if (lane_id == src_lane)
            {
                ThreadStore<PTX_STORE_VS>(smem_storage.warp_scan[warp_id], input);
            }

#if (CUB_PTX_ARCH <= 110)
            __threadfence_block();
#endif
            return ThreadLoad<PTX_LOAD_VS>(smem_storage.warp_scan[warp_id]);
        }



        /// Basic inclusive scan iteration (template unrolled, inductive-case specialization)
        template <
            bool HAS_IDENTITY,
            bool SHARE_FINAL,
            int STEP>
        struct Iteration
        {
            template <typename ScanOp>
            static __device__ __forceinline__ void ScanStep(SmemStorage &smem_storage, unsigned int warp_id, unsigned int lane_id, T &partial, ScanOp scan_op)
            {
                const int OFFSET = 1 << STEP;

                // Share partial into buffer
                ThreadStore<PTX_STORE_VS>(&smem_storage.warp_scan[warp_id][HALF_WARP_THREADS + lane_id], partial);

                // Update partial if addend is in range
                if (HAS_IDENTITY || (lane_id >= OFFSET))
                {
                    T addend = ThreadLoad<PTX_LOAD_VS>(&smem_storage.warp_scan[warp_id][HALF_WARP_THREADS + lane_id - OFFSET]);
                    partial = scan_op(addend, partial);
                }

                Iteration<HAS_IDENTITY, SHARE_FINAL, STEP + 1>::ScanStep(smem_storage, warp_id, lane_id, partial, scan_op);
            }
        };


        /// Basic inclusive scan iteration(template unrolled, base-case specialization)
        template <
            bool HAS_IDENTITY,
            bool SHARE_FINAL>
        struct Iteration<HAS_IDENTITY, SHARE_FINAL, STEPS>
        {
            template <typename ScanOp>
            static __device__ __forceinline__ void ScanStep(SmemStorage &smem_storage, unsigned int warp_id, unsigned int lane_id, T &partial, ScanOp scan_op) {}
        };


        /// Basic inclusive scan
        template <
            bool HAS_IDENTITY,
            bool SHARE_FINAL,
            typename ScanOp>
        static __device__ __forceinline__ T BasicScan(
            SmemStorage     &smem_storage,      ///< Reference to shared memory allocation having layout type SmemStorage
            unsigned int    warp_id,            ///< Warp id
            unsigned int    lane_id,            ///< thread-lane id
            T               partial,            ///< Calling thread's input partial reduction
            ScanOp          scan_op)            ///< Binary associative scan functor
        {
            // Iterate scan steps
            Iteration<HAS_IDENTITY, SHARE_FINAL, 0>::ScanStep(smem_storage, warp_id, lane_id, partial, scan_op);

            if (SHARE_FINAL)
            {
                // Share partial into buffer
                ThreadStore<PTX_STORE_VS>(&smem_storage.warp_scan[warp_id][HALF_WARP_THREADS + lane_id], partial);
            }

            return partial;
        }


        /// Inclusive prefix sum
        static __device__ __forceinline__ void InclusiveSum(
            SmemStorage     &smem_storage,      ///< [in] Reference to shared memory allocation having layout type SmemStorage
            T               input,              ///< [in] Calling thread's input item.
            T               &output)            ///< [out] Calling thread's output item.  May be aliased with \p input.
        {
            // Warp, thread-lane-IDs
            unsigned int warp_id = (WARPS == 1) ? 0 : (threadIdx.x / LOGICAL_WARP_THREADS);
            unsigned int lane_id = (WARPS == 1) ? threadIdx.x : (threadIdx.x & (LOGICAL_WARP_THREADS - 1));

            // Initialize identity region
            ThreadStore<PTX_STORE_VS>(&smem_storage.warp_scan[warp_id][lane_id], T(0));

            // Compute inclusive warp scan (has identity, don't share final)
            output = BasicScan<true, false>(
                smem_storage,
                warp_id,
                lane_id,
                input,
                Sum<T>());
        }


        /// Inclusive prefix sum with aggregate
        static __device__ __forceinline__ void InclusiveSum(
            SmemStorage     &smem_storage,      ///< [in] Reference to shared memory allocation having layout type SmemStorage
            T               input,              ///< [in] Calling thread's input item.
            T               &output,            ///< [out] Calling thread's output item.  May be aliased with \p input.
            T               &warp_aggregate)    ///< [out] Warp-wide aggregate reduction of input items.
        {
            // Warp, thread-lane-IDs
            unsigned int warp_id = (WARPS == 1) ? 0 : (threadIdx.x / LOGICAL_WARP_THREADS);
            unsigned int lane_id = (WARPS == 1) ? threadIdx.x : (threadIdx.x & (LOGICAL_WARP_THREADS - 1));

            // Initialize identity region
            ThreadStore<PTX_STORE_VS>(&smem_storage.warp_scan[warp_id][lane_id], T(0));

            // Compute inclusive warp scan (has identity, share final)
            output = BasicScan<true, true>(
                smem_storage,
                warp_id,
                lane_id,
                input,
                Sum<T>());

            // Retrieve aggregate in <em>warp-lane</em><sub>0</sub>
            warp_aggregate = ThreadLoad<PTX_LOAD_VS>(&smem_storage.warp_scan[warp_id][WARP_SMEM_ELEMENTS - 1]);
        }


        /// Inclusive scan
        template <typename ScanOp>
        static __device__ __forceinline__ void InclusiveScan(
            SmemStorage     &smem_storage,      ///< [in] Reference to shared memory allocation having layout type SmemStorage
            T               input,              ///< [in] Calling thread's input item.
            T               &output,            ///< [out] Calling thread's output item.  May be aliased with \p input.
            ScanOp          scan_op)            ///< [in] Binary scan operator
        {
            // Warp, thread-lane-IDs
            unsigned int warp_id = (WARPS == 1) ? 0 : (threadIdx.x / LOGICAL_WARP_THREADS);
            unsigned int lane_id = (WARPS == 1) ? threadIdx.x : (threadIdx.x & (LOGICAL_WARP_THREADS - 1));

            // Compute inclusive warp scan (no identity, don't share final)
            output = BasicScan<false, false>(
                smem_storage,
                warp_id,
                lane_id,
                input,
                scan_op);
        }


        /// Inclusive scan with aggregate
        template <typename ScanOp>
        static __device__ __forceinline__ void InclusiveScan(
            SmemStorage     &smem_storage,      ///< [in] Reference to shared memory allocation having layout type SmemStorage
            T               input,              ///< [in] Calling thread's input item.
            T               &output,            ///< [out] Calling thread's output item.  May be aliased with \p input.
            ScanOp          scan_op,            ///< [in] Binary scan operator
            T               &warp_aggregate)    ///< [out] Warp-wide aggregate reduction of input items.
        {
            // Warp, thread-lane-IDs
            unsigned int warp_id = (WARPS == 1) ? 0 : (threadIdx.x / LOGICAL_WARP_THREADS);
            unsigned int lane_id = (WARPS == 1) ? threadIdx.x : (threadIdx.x & (LOGICAL_WARP_THREADS - 1));

            // Compute inclusive warp scan (no identity, share final)
            output = BasicScan<false, true>(
                smem_storage,
                warp_id,
                lane_id,
                input,
                scan_op);

            // Retrieve aggregate
            warp_aggregate = ThreadLoad<PTX_LOAD_VS>(&smem_storage.warp_scan[warp_id][WARP_SMEM_ELEMENTS - 1]);
        }

        /// Exclusive scan
        template <typename ScanOp>
        static __device__ __forceinline__ void ExclusiveScan(
            SmemStorage     &smem_storage,      ///< [in] Reference to shared memory allocation having layout type SmemStorage
            T               input,              ///< [in] Calling thread's input item.
            T               &output,            ///< [out] Calling thread's output item.  May be aliased with \p input.
            const T         &identity,          ///< [in] Identity value
            ScanOp          scan_op)            ///< [in] Binary scan operator
        {
            // Warp, thread-lane-IDs
            unsigned int warp_id = (WARPS == 1) ? 0 : (threadIdx.x / LOGICAL_WARP_THREADS);
            unsigned int lane_id = (WARPS == 1) ? threadIdx.x : (threadIdx.x & (LOGICAL_WARP_THREADS - 1));

            // Initialize identity region
            ThreadStore<PTX_STORE_VS>(&smem_storage.warp_scan[warp_id][lane_id], identity);

            // Compute inclusive warp scan (identity, share final)
            T inclusive = BasicScan<true, true>(
                smem_storage,
                warp_id,
                lane_id,
                input,
                scan_op);

            // Retrieve exclusive scan
            output = ThreadLoad<PTX_LOAD_VS>(&smem_storage.warp_scan[warp_id][HALF_WARP_THREADS + lane_id - 1]);
        }


        /// Exclusive scan with aggregate
        template <typename ScanOp>
        static __device__ __forceinline__ void ExclusiveScan(
            SmemStorage     &smem_storage,      ///< [in] Reference to shared memory allocation having layout type SmemStorage
            T               input,              ///< [in] Calling thread's input item.
            T               &output,            ///< [out] Calling thread's output item.  May be aliased with \p input.
            const T         &identity,          ///< [in] Identity value
            ScanOp          scan_op,            ///< [in] Binary scan operator
            T               &warp_aggregate)    ///< [out] Warp-wide aggregate reduction of input items.
        {
            // Warp id
            unsigned int warp_id = (WARPS == 1) ? 0 : (threadIdx.x / LOGICAL_WARP_THREADS);

            // Exclusive warp scan (which does share final)
            ExclusiveScan(smem_storage, input, output, identity, scan_op);

            // Retrieve aggregate
            warp_aggregate = ThreadLoad<PTX_LOAD_VS>(&smem_storage.warp_scan[warp_id][WARP_SMEM_ELEMENTS - 1]);
        }


        /// Exclusive scan without identity
        template <typename ScanOp>
        static __device__ __forceinline__ void ExclusiveScan(
            SmemStorage     &smem_storage,      ///< [in] Reference to shared memory allocation having layout type SmemStorage
            T               input,              ///< [in] Calling thread's input item.
            T               &output,            ///< [out] Calling thread's output item.  May be aliased with \p input.
            ScanOp          scan_op)            ///< [in] Binary scan operator
        {
            // Warp, thread-lane-IDs
            unsigned int warp_id = (WARPS == 1) ? 0 : (threadIdx.x / LOGICAL_WARP_THREADS);
            unsigned int lane_id = (WARPS == 1) ? threadIdx.x : (threadIdx.x & (LOGICAL_WARP_THREADS - 1));

            // Compute inclusive warp scan (no identity, share final)
            T inclusive = BasicScan<false, true>(
                smem_storage,
                warp_id,
                lane_id,
                input,
                scan_op);

            // Retrieve exclusive scan
            output = ThreadLoad<PTX_LOAD_VS>(&smem_storage.warp_scan[warp_id][HALF_WARP_THREADS + lane_id - 1]);
        }


        /// Exclusive scan with aggregate, without identity
        template <typename ScanOp>
        static __device__ __forceinline__ void ExclusiveScan(
            SmemStorage     &smem_storage,      ///< [in] Reference to shared memory allocation having layout type SmemStorage
            T               input,              ///< [in] Calling thread's input item.
            T               &output,            ///< [out] Calling thread's output item.  May be aliased with \p input.
            ScanOp          scan_op,            ///< [in] Binary scan operator
            T               &warp_aggregate)    ///< [out] Warp-wide aggregate reduction of input items.
        {
            // Warp id
            unsigned int warp_id = (WARPS == 1) ? 0 : (threadIdx.x / LOGICAL_WARP_THREADS);

            // Exclusive warp scan (which does share final)
            ExclusiveScan(smem_storage, input, output, scan_op);

            // Retrieve aggregate
            warp_aggregate = ThreadLoad<PTX_LOAD_VS>(&smem_storage.warp_scan[warp_id][WARP_SMEM_ELEMENTS - 1]);
        }

    };


    #endif // DOXYGEN_SHOULD_SKIP_THIS


    /// Shared memory storage layout type for WarpScan
    typedef typename WarpScanInternal<POLICY>::SmemStorage _SmemStorage;


public:

    /// \smemstorage{WarpScan}
    typedef _SmemStorage SmemStorage;


    /******************************************************************//**
     * \name Inclusive prefix sums
     *********************************************************************/
    //@{

private:

    /// Computes an exclusive prefix sum in each logical warp.
    static __device__ __forceinline__ void InclusiveSum(SmemStorage &smem_storage, T input, T &output, Int2Type<true> is_primitive)
    {
        WarpScanInternal<POLICY>::InclusiveSum(smem_storage, input, output);
    }

    /// Computes an exclusive prefix sum in each logical warp.  Specialized for non-primitive types.
    static __device__ __forceinline__ void InclusiveSum(SmemStorage &smem_storage, T input, T &output, Int2Type<false> is_primitive)
    {
        // Delegate to regular scan for non-primitive types (because we won't be able to use subtraction or 0 as identity)
        InclusiveScan(smem_storage, input, output, Sum<T>());
    }

    /// Computes an exclusive prefix sum in each logical warp.  Also provides every thread with the warp-wide \p warp_aggregate of all inputs.
    static __device__ __forceinline__ void InclusiveSum(SmemStorage &smem_storage, T input, T &output, T &warp_aggregate, Int2Type<true> is_primitive)
    {
        WarpScanInternal<POLICY>::InclusiveSum(smem_storage, input, output, warp_aggregate);
    }

    /// Computes an exclusive prefix sum in each logical warp.  Also provides every thread with the warp-wide \p warp_aggregate of all inputs.  Specialized for non-primitive types.
    static __device__ __forceinline__ void InclusiveSum(SmemStorage &smem_storage, T input, T &output, T &warp_aggregate, Int2Type<false> is_primitive)
    {
        // Delegate to regular scan for non-primitive types (because we won't be able to use subtraction or 0 as identity)
        InclusiveScan(smem_storage, input, output, Sum<T>(), warp_aggregate);
    }

    /// Computes an exclusive prefix sum in each logical warp.  Instead of using 0 as the warp-wide prefix, the call-back functor \p warp_prefix_op is invoked to provide the "seed" value that logically prefixes the warp's scan inputs.  Also provides every thread with the warp-wide \p warp_aggregate of all inputs.
    template <typename WarpPrefixOp>
    static __device__ __forceinline__ void InclusiveSum(SmemStorage &smem_storage, T input, T &output, T &warp_aggregate, WarpPrefixOp &warp_prefix_op, Int2Type<true> is_primitive)
    {
        // Compute inclusive warp scan
        InclusiveSum(smem_storage, input, output, warp_aggregate);

        // Compute warp-wide prefix from aggregate, then broadcast to other lanes
        T prefix;
        prefix = warp_prefix_op(warp_aggregate);
        prefix = WarpScanInternal<POLICY>::Broadcast(smem_storage, prefix, 0);

        // Update output
        output = prefix + output;
    }

    /// Computes an exclusive prefix sum in each logical warp.  Instead of using 0 as the warp-wide prefix, the call-back functor \p warp_prefix_op is invoked to provide the "seed" value that logically prefixes the warp's scan inputs.  Also provides every thread with the warp-wide \p warp_aggregate of all inputs.  Specialized for non-primitive types.
    template <typename WarpPrefixOp>
    static __device__ __forceinline__ void InclusiveSum(SmemStorage &smem_storage, T input, T &output, T &warp_aggregate, WarpPrefixOp &warp_prefix_op, Int2Type<false> is_primitive)
    {
        // Delegate to regular scan for non-primitive types (because we won't be able to use subtraction or 0 as identity)
        InclusiveScan(smem_storage, input, output, Sum<T>(), warp_aggregate, warp_prefix_op);
    }

public:


    /**
     * \brief Computes an inclusive prefix sum in each logical warp.
     *
     * \smemreuse
     */
    static __device__ __forceinline__ void InclusiveSum(
        SmemStorage     &smem_storage,      ///< [in] Reference to shared memory allocation having layout type SmemStorage
        T               input,              ///< [in] Calling thread's input item.
        T               &output)            ///< [out] Calling thread's output item.  May be aliased with \p input.
    {
        InclusiveSum(smem_storage, input, output, Int2Type<Traits<T>::PRIMITIVE>());
    }


    /**
     * \brief Computes an inclusive prefix sum in each logical warp.  Also provides every thread with the warp-wide \p warp_aggregate of all inputs.
     *
     * The \p warp_aggregate is undefined in threads other than <em>warp-lane</em><sub>0</sub>.
     *
     * \smemreuse
     */
    static __device__ __forceinline__ void InclusiveSum(
        SmemStorage     &smem_storage,      ///< [in] Reference to shared memory allocation having layout type SmemStorage
        T               input,              ///< [in] Calling thread's input item.
        T               &output,            ///< [out] Calling thread's output item.  May be aliased with \p input.
        T               &warp_aggregate)    ///< [out] Warp-wide aggregate reduction of input items.
    {
        InclusiveSum(smem_storage, input, output, warp_aggregate, Int2Type<Traits<T>::PRIMITIVE>());
    }


    /**
     * \brief Computes an inclusive prefix sum in each logical warp.  Instead of using 0 as the warp-wide prefix, the call-back functor \p warp_prefix_op is invoked to provide the "seed" value that logically prefixes the warp's scan inputs.  Also provides every thread with the warp-wide \p warp_aggregate of all inputs.
     *
     * The \p warp_aggregate is undefined in threads other than <em>warp-lane</em><sub>0</sub>.
     *
     * The \p warp_prefix_op functor must implement a member function <tt>T operator()(T warp_aggregate)</tt>.
     * The functor's input parameter \p warp_aggregate is the same value also returned by the scan operation.
     * The functor will be invoked by the entire warp of threads, however only the return value from
     * <em>lane</em><sub>0</sub> is applied as the threadblock-wide prefix.  Can be stateful.
     *
     * \smemreuse
     *
     * \tparam WarpPrefixOp                 <b>[inferred]</b> Call-back functor type having member <tt>T operator()(T warp_aggregate)</tt>
     */
    template <typename WarpPrefixOp>
    static __device__ __forceinline__ void InclusiveSum(
        SmemStorage     &smem_storage,      ///< [in] Reference to shared memory allocation having layout type SmemStorage
        T               input,              ///< [in] Calling thread's input item.
        T               &output,            ///< [out] Calling thread's output item.  May be aliased with \p input.
        T               &warp_aggregate,    ///< [out] <b>[<em>warp-lane</em><sub>0</sub> only]</b> Warp-wide aggregate reduction of input items, exclusive of the \p warp_prefix_op value
        WarpPrefixOp    &warp_prefix_op)    ///< [in-out] <b>[<em>warp-lane</em><sub>0</sub> only]</b> Call-back functor for specifying a warp-wide prefix to be applied to all inputs.
    {
        InclusiveSum(smem_storage, input, output, warp_aggregate, warp_prefix_op, Int2Type<Traits<T>::PRIMITIVE>());
    }


    //@}  end member group

    /******************************************************************//**
     * \name Exclusive prefix sums
     *********************************************************************/
    //@{


private:

    /// Computes an exclusive prefix sum in each logical warp.
    static __device__ __forceinline__ void ExclusiveSum(SmemStorage &smem_storage, T input, T &output, Int2Type<true> is_primitive)
    {
        // Compute exclusive warp scan from inclusive warp scan
        T inclusive;
        InclusiveSum(smem_storage, input, inclusive);
        output = inclusive - input;
    }

    /// Computes an exclusive prefix sum in each logical warp.  Specialized for non-primitive types.
    static __device__ __forceinline__ void ExclusiveSum(SmemStorage &smem_storage, T input, T &output, Int2Type<false> is_primitive)
    {
        // Delegate to regular scan for non-primitive types (because we won't be able to use subtraction or 0 as identity)
        T identity = T();
        ExclusiveScan(smem_storage, input, output, identity, Sum<T>());
    }

    /// Computes an exclusive prefix sum in each logical warp.  Also provides every thread with the warp-wide \p warp_aggregate of all inputs.
    static __device__ __forceinline__ void ExclusiveSum(SmemStorage &smem_storage, T input, T &output, T &warp_aggregate, Int2Type<true> is_primitive)
    {
        // Compute exclusive warp scan from inclusive warp scan
        T inclusive;
        InclusiveSum(smem_storage, input, inclusive, warp_aggregate);
        output = inclusive - input;
    }

    /// Computes an exclusive prefix sum in each logical warp.  Also provides every thread with the warp-wide \p warp_aggregate of all inputs.  Specialized for non-primitive types.
    static __device__ __forceinline__ void ExclusiveSum(SmemStorage &smem_storage, T input, T &output, T &warp_aggregate, Int2Type<false> is_primitive)
    {
        // Delegate to regular scan for non-primitive types (because we won't be able to use subtraction or 0 as identity)
        T identity = T();
        ExclusiveScan(smem_storage, input, output, identity, Sum<T>(), warp_aggregate);
    }

    /// Computes an exclusive prefix sum in each logical warp.  Instead of using 0 as the warp-wide prefix, the call-back functor \p warp_prefix_op is invoked to provide the "seed" value that logically prefixes the warp's scan inputs.  Also provides every thread with the warp-wide \p warp_aggregate of all inputs.
    template <typename WarpPrefixOp>
    static __device__ __forceinline__ void ExclusiveSum(SmemStorage &smem_storage, T input, T &output, T &warp_aggregate, WarpPrefixOp &warp_prefix_op, Int2Type<true> is_primitive)
    {
        // Compute exclusive warp scan from inclusive warp scan
        T inclusive;
        InclusiveSum(smem_storage, input, inclusive, warp_aggregate, warp_prefix_op);
        output = inclusive - input;
    }

    /// Computes an exclusive prefix sum in each logical warp.  Instead of using 0 as the warp-wide prefix, the call-back functor \p warp_prefix_op is invoked to provide the "seed" value that logically prefixes the warp's scan inputs.  Also provides every thread with the warp-wide \p warp_aggregate of all inputs.  Specialized for non-primitive types.
    template <typename WarpPrefixOp>
    static __device__ __forceinline__ void ExclusiveSum(SmemStorage &smem_storage, T input, T &output, T &warp_aggregate, WarpPrefixOp &warp_prefix_op, Int2Type<false> is_primitive)
    {
        // Delegate to regular scan for non-primitive types (because we won't be able to use subtraction or 0 as identity)
        T identity = T();
        ExclusiveScan(smem_storage, input, output, identity, Sum<T>(), warp_aggregate, warp_prefix_op);
    }

public:


    /**
     * \brief Computes an exclusive prefix sum in each logical warp.
     *
     * This operation assumes the value of obtained by the <tt>T</tt>'s default
     * constructor (or by zero-initialization if no user-defined default
     * constructor exists) is suitable as the identity value "zero" for
     * addition.
     *
     * \smemreuse
     */
    static __device__ __forceinline__ void ExclusiveSum(
        SmemStorage     &smem_storage,      ///< [in] Reference to shared memory allocation having layout type SmemStorage
        T               input,              ///< [in] Calling thread's input item.
        T               &output)            ///< [out] Calling thread's output item.  May be aliased with \p input.
    {
        ExclusiveSum(smem_storage, input, output, Int2Type<Traits<T>::PRIMITIVE>());
    }


    /**
     * \brief Computes an exclusive prefix sum in each logical warp.  Also provides every thread with the warp-wide \p warp_aggregate of all inputs.
     *
     * This operation assumes the value of obtained by the <tt>T</tt>'s default
     * constructor (or by zero-initialization if no user-defined default
     * constructor exists) is suitable as the identity value "zero" for
     * addition.
     *
     * \smemreuse
     */
    static __device__ __forceinline__ void ExclusiveSum(
        SmemStorage     &smem_storage,      ///< [in] Reference to shared memory allocation having layout type SmemStorage
        T               input,              ///< [in] Calling thread's input item.
        T               &output,            ///< [out] Calling thread's output item.  May be aliased with \p input.
        T               &warp_aggregate)    ///< [out] Warp-wide aggregate reduction of input items.
    {
        ExclusiveSum(smem_storage, input, output, warp_aggregate, Int2Type<Traits<T>::PRIMITIVE>());
    }


    /**
     * \brief Computes an exclusive prefix sum in each logical warp.  Instead of using 0 as the warp-wide prefix, the call-back functor \p warp_prefix_op is invoked to provide the "seed" value that logically prefixes the warp's scan inputs.  Also provides every thread with the warp-wide \p warp_aggregate of all inputs.
     *
     * This operation assumes the value of obtained by the <tt>T</tt>'s default
     * constructor (or by zero-initialization if no user-defined default
     * constructor exists) is suitable as the identity value "zero" for
     * addition.
     *
     * The \p warp_prefix_op functor must implement a member function <tt>T operator()(T warp_aggregate)</tt>.
     * The functor's input parameter \p warp_aggregate is the same value also returned by the scan operation.
     * The functor will be invoked by the entire warp of threads, however only the return value from
     * <em>lane</em><sub>0</sub> is applied as the threadblock-wide prefix.  Can be stateful.
     *
     * \smemreuse
     *
     * \tparam WarpPrefixOp                 <b>[inferred]</b> Call-back functor type having member <tt>T operator()(T warp_aggregate)</tt>
     */
    template <typename WarpPrefixOp>
    static __device__ __forceinline__ void ExclusiveSum(
        SmemStorage     &smem_storage,      ///< [in] Reference to shared memory allocation having layout type SmemStorage
        T               input,              ///< [in] Calling thread's input item.
        T               &output,            ///< [out] Calling thread's output item.  May be aliased with \p input.
        T               &warp_aggregate,    ///< [out] <b>[<em>warp-lane</em><sub>0</sub> only]</b> Warp-wide aggregate reduction of input items (exclusive of the \p warp_prefix_op value).
        WarpPrefixOp    &warp_prefix_op)    ///< [in-out] <b>[<em>warp-lane</em><sub>0</sub> only]</b> Call-back functor for specifying a warp-wide prefix to be applied to all inputs.
    {
        ExclusiveSum(smem_storage, input, output, warp_aggregate, warp_prefix_op, Int2Type<Traits<T>::PRIMITIVE>());
    }


    //@}  end member group
    /******************************************************************//**
     * \name Inclusive prefix scans
     *********************************************************************/
    //@{

    /**
     * \brief Computes an inclusive prefix sum using the specified binary scan functor in each logical warp.
     *
     * \smemreuse
     *
     * \tparam ScanOp     <b>[inferred]</b> Binary scan operator type having member <tt>T operator()(const T &a, const T &b)</tt>
     */
    template <typename ScanOp>
    static __device__ __forceinline__ void InclusiveScan(
        SmemStorage     &smem_storage,      ///< [in] Reference to shared memory allocation having layout type SmemStorage
        T               input,              ///< [in] Calling thread's input item.
        T               &output,            ///< [out] Calling thread's output item.  May be aliased with \p input.
        ScanOp          scan_op)            ///< [in] Binary scan operator
    {
        WarpScanInternal<POLICY>::InclusiveScan(smem_storage, input, output, scan_op);
    }


    /**
     * \brief Computes an inclusive prefix sum using the specified binary scan functor in each logical warp.  Also provides every thread with the warp-wide \p warp_aggregate of all inputs.
     *
     * The \p warp_aggregate is undefined in threads other than <em>warp-lane</em><sub>0</sub>.
     *
     * \smemreuse
     *
     * \tparam ScanOp     <b>[inferred]</b> Binary scan operator type having member <tt>T operator()(const T &a, const T &b)</tt>
     */
    template <typename ScanOp>
    static __device__ __forceinline__ void InclusiveScan(
        SmemStorage     &smem_storage,      ///< [in] Reference to shared memory allocation having layout type SmemStorage
        T               input,              ///< [in] Calling thread's input item.
        T               &output,            ///< [out] Calling thread's output item.  May be aliased with \p input.
        ScanOp          scan_op,            ///< [in] Binary scan operator
        T               &warp_aggregate)    ///< [out] Warp-wide aggregate reduction of input items.
    {
        WarpScanInternal<POLICY>::InclusiveScan(smem_storage, input, output, scan_op, warp_aggregate);
    }


    /**
     * \brief Computes an inclusive prefix sum using the specified binary scan functor in each logical warp.  The call-back functor \p warp_prefix_op is invoked to provide the "seed" value that logically prefixes the warp's scan inputs.  Also provides every thread with the warp-wide \p warp_aggregate of all inputs.
     *
     * The \p warp_aggregate is undefined in threads other than <em>warp-lane</em><sub>0</sub>.
     *
     * The \p warp_prefix_op functor must implement a member function <tt>T operator()(T warp_aggregate)</tt>.
     * The functor's input parameter \p warp_aggregate is the same value also returned by the scan operation.
     * The functor will be invoked by the entire warp of threads, however only the return value from
     * <em>lane</em><sub>0</sub> is applied as the threadblock-wide prefix.  Can be stateful.
     *
     * \smemreuse
     *
     * \tparam ScanOp                       <b>[inferred]</b> Binary scan operator type having member <tt>T operator()(const T &a, const T &b)</tt>
     * \tparam WarpPrefixOp                 <b>[inferred]</b> Call-back functor type having member <tt>T operator()(T warp_aggregate)</tt>
     */
    template <
        typename ScanOp,
        typename WarpPrefixOp>
    static __device__ __forceinline__ void InclusiveScan(
        SmemStorage     &smem_storage,      ///< [in] Reference to shared memory allocation having layout type SmemStorage
        T               input,              ///< [in] Calling thread's input item.
        T               &output,            ///< [out] Calling thread's output item.  May be aliased with \p input.
        ScanOp          scan_op,            ///< [in] Binary scan operator
        T               &warp_aggregate,    ///< [out] <b>[<em>warp-lane</em><sub>0</sub> only]</b> Warp-wide aggregate reduction of input items (exclusive of the \p warp_prefix_op value).
        WarpPrefixOp    &warp_prefix_op)    ///< [in-out] <b>[<em>warp-lane</em><sub>0</sub> only]</b> Call-back functor for specifying a warp-wide prefix to be applied to all inputs.
    {
        // Compute inclusive warp scan
        InclusiveScan(smem_storage, input, output, scan_op, warp_aggregate);

        // Compute warp-wide prefix from aggregate, then broadcast to other lanes
        T prefix;
        prefix = warp_prefix_op(warp_aggregate);
        prefix = WarpScanInternal<POLICY>::Broadcast(smem_storage, prefix, 0);

        // Update output
        output = scan_op(prefix, output);
    }


    //@}  end member group
    /******************************************************************//**
     * \name Exclusive prefix scans
     *********************************************************************/
    //@{

    /**
     * \brief Computes an exclusive prefix scan using the specified binary scan functor in each logical warp.
     *
     * \smemreuse
     *
     * \tparam ScanOp     <b>[inferred]</b> Binary scan operator type having member <tt>T operator()(const T &a, const T &b)</tt>
     */
    template <typename ScanOp>
    static __device__ __forceinline__ void ExclusiveScan(
        SmemStorage     &smem_storage,      ///< [in] Reference to shared memory allocation having layout type SmemStorage
        T               input,              ///< [in] Calling thread's input item.
        T               &output,            ///< [out] Calling thread's output item.  May be aliased with \p input.
        const T         &identity,          ///< [in] Identity value
        ScanOp          scan_op)            ///< [in] Binary scan operator
    {
        WarpScanInternal<POLICY>::ExclusiveScan(smem_storage, input, output, identity, scan_op);
    }


    /**
     * \brief Computes an exclusive prefix scan using the specified binary scan functor in each logical warp.  Also provides every thread with the warp-wide \p warp_aggregate of all inputs.
     *
     * The \p warp_aggregate is undefined in threads other than <em>warp-lane</em><sub>0</sub>.
     *
     * \smemreuse
     *
     * \tparam ScanOp     <b>[inferred]</b> Binary scan operator type having member <tt>T operator()(const T &a, const T &b)</tt>
     */
    template <typename ScanOp>
    static __device__ __forceinline__ void ExclusiveScan(
        SmemStorage     &smem_storage,      ///< [in] Reference to shared memory allocation having layout type SmemStorage
        T               input,              ///< [in] Calling thread's input item.
        T               &output,            ///< [out] Calling thread's output item.  May be aliased with \p input.
        const T         &identity,          ///< [in] Identity value
        ScanOp          scan_op,            ///< [in] Binary scan operator
        T               &warp_aggregate)    ///< [out] Warp-wide aggregate reduction of input items.
    {
        WarpScanInternal<POLICY>::ExclusiveScan(smem_storage, input, output, identity, scan_op, warp_aggregate);
    }


    /**
     * \brief Computes an exclusive prefix scan using the specified binary scan functor in each logical warp.  The call-back functor \p warp_prefix_op is invoked to provide the "seed" value that logically prefixes the warp's scan inputs.  Also provides every thread with the warp-wide \p warp_aggregate of all inputs.
     *
     * The \p warp_aggregate is undefined in threads other than <em>warp-lane</em><sub>0</sub>.
     *
     * The \p warp_prefix_op functor must implement a member function <tt>T operator()(T warp_aggregate)</tt>.
     * The functor's input parameter \p warp_aggregate is the same value also returned by the scan operation.
     * The functor will be invoked by the entire warp of threads, however only the return value from
     * <em>lane</em><sub>0</sub> is applied as the threadblock-wide prefix.  Can be stateful.
     *
     * \smemreuse
     *
     * \tparam ScanOp                       <b>[inferred]</b> Binary scan operator type having member <tt>T operator()(const T &a, const T &b)</tt>
     * \tparam WarpPrefixOp                 <b>[inferred]</b> Call-back functor type having member <tt>T operator()(T warp_aggregate)</tt>
     */
    template <
        typename ScanOp,
        typename WarpPrefixOp>
    static __device__ __forceinline__ void ExclusiveScan(
        SmemStorage     &smem_storage,      ///< [in] Reference to shared memory allocation having layout type SmemStorage
        T               input,              ///< [in] Calling thread's input item.
        T               &output,            ///< [out] Calling thread's output item.  May be aliased with \p input.
        const T         &identity,          ///< [in] Identity value
        ScanOp          scan_op,            ///< [in] Binary scan operator
        T               &warp_aggregate,    ///< [out] <b>[<em>warp-lane</em><sub>0</sub> only]</b> Warp-wide aggregate reduction of input items (exclusive of the \p warp_prefix_op value).
        WarpPrefixOp    &warp_prefix_op)    ///< [in-out] <b>[<em>warp-lane</em><sub>0</sub> only]</b> Call-back functor for specifying a warp-wide prefix to be applied to all inputs.
    {
        // Exclusive warp scan
        ExclusiveScan(smem_storage, input, output, identity, scan_op, warp_aggregate);

        // Compute warp-wide prefix from aggregate, then broadcast to other lanes
        T prefix;
        unsigned int lane_id = (WARPS == 1) ? threadIdx.x : (threadIdx.x & (LOGICAL_WARP_THREADS - 1));
        prefix = warp_prefix_op(warp_aggregate);
        prefix = WarpScanInternal<POLICY>::Broadcast(smem_storage, prefix, 0);

        // Update output
        output = (lane_id == 0) ?
            prefix :
            scan_op(prefix, output);
    }


    //@}  end member group
    /******************************************************************//**
     * \name Exclusive prefix scans (without supplied identity)
     *********************************************************************/
    //@{


    /**
     * \brief Computes an exclusive prefix scan using the specified binary scan functor in each logical warp.  Because no identity value is supplied, the \p output computed for <em>warp-lane</em><sub>0</sub> is undefined.
     *
     * \smemreuse
     *
     * \tparam ScanOp     <b>[inferred]</b> Binary scan operator type having member <tt>T operator()(const T &a, const T &b)</tt>
     */
    template <typename ScanOp>
    static __device__ __forceinline__ void ExclusiveScan(
        SmemStorage     &smem_storage,      ///< [in] Reference to shared memory allocation having layout type SmemStorage
        T               input,              ///< [in] Calling thread's input item.
        T               &output,            ///< [out] Calling thread's output item.  May be aliased with \p input.
        ScanOp          scan_op)            ///< [in] Binary scan operator
    {
        WarpScanInternal<POLICY>::ExclusiveScan(smem_storage, input, output, scan_op);
    }


    /**
     * \brief Computes an exclusive prefix scan using the specified binary scan functor in each logical warp.  Because no identity value is supplied, the \p output computed for <em>warp-lane</em><sub>0</sub> is undefined.  Also provides every thread with the warp-wide \p warp_aggregate of all inputs.
     *
     * The \p warp_aggregate is undefined in threads other than <em>warp-lane</em><sub>0</sub>.
     *
     * \smemreuse
     *
     * \tparam ScanOp     <b>[inferred]</b> Binary scan operator type having member <tt>T operator()(const T &a, const T &b)</tt>
     */
    template <typename ScanOp>
    static __device__ __forceinline__ void ExclusiveScan(
        SmemStorage     &smem_storage,      ///< [in] Reference to shared memory allocation having layout type SmemStorage
        T               input,              ///< [in] Calling thread's input item.
        T               &output,            ///< [out] Calling thread's output item.  May be aliased with \p input.
        ScanOp          scan_op,            ///< [in] Binary scan operator
        T               &warp_aggregate)    ///< [out] Warp-wide aggregate reduction of input items.
    {
        WarpScanInternal<POLICY>::ExclusiveScan(smem_storage, input, output, scan_op, warp_aggregate);
    }


    /**
     * \brief Computes an exclusive prefix scan using the specified binary scan functor in each logical warp.  The \p warp_prefix_op value from thread-thread-lane<sub>0</sub> is applied to all scan outputs.  Also computes the warp-wide \p warp_aggregate of all inputs for thread-thread-lane<sub>0</sub>.
     *
     * The \p warp_aggregate is undefined in threads other than <em>warp-lane</em><sub>0</sub>.
     *
     * The \p warp_prefix_op functor must implement a member function <tt>T operator()(T warp_aggregate)</tt>.
     * The functor's input parameter \p warp_aggregate is the same value also returned by the scan operation.
     * The functor will be invoked by the entire warp of threads, however only the return value from
     * <em>lane</em><sub>0</sub> is applied as the threadblock-wide prefix.  Can be stateful.
     *
     * \smemreuse
     *
     * \tparam ScanOp                       <b>[inferred]</b> Binary scan operator type having member <tt>T operator()(const T &a, const T &b)</tt>
     * \tparam WarpPrefixOp                 <b>[inferred]</b> Call-back functor type having member <tt>T operator()(T warp_aggregate)</tt>
     */
    template <
        typename ScanOp,
        typename WarpPrefixOp>
    static __device__ __forceinline__ void ExclusiveScan(
        SmemStorage     &smem_storage,      ///< [in] Reference to shared memory allocation having layout type SmemStorage
        T               input,              ///< [in] Calling thread's input item.
        T               &output,            ///< [out] Calling thread's output item.  May be aliased with \p input.
        ScanOp          scan_op,            ///< [in] Binary scan operator
        T               &warp_aggregate,    ///< [out] <b>[<em>warp-lane</em><sub>0</sub> only]</b> Warp-wide aggregate reduction of input items (exclusive of the \p warp_prefix_op value).
        WarpPrefixOp    &warp_prefix_op)    ///< [in-out] <b>[<em>warp-lane</em><sub>0</sub> only]</b> Call-back functor for specifying a warp-wide prefix to be applied to all inputs.
    {
        // Exclusive warp scan
        ExclusiveScan(smem_storage, input, output, scan_op, warp_aggregate);

        // Compute warp-wide prefix from aggregate, then broadcast to other lanes
        T prefix;
        unsigned int lane_id = (WARPS == 1) ? threadIdx.x : (threadIdx.x & (LOGICAL_WARP_THREADS - 1));
        prefix = warp_prefix_op(warp_aggregate);
        prefix = WarpScanInternal<POLICY>::Broadcast(smem_storage, prefix, 0);

        // Update output with prefix
        output = (lane_id == 0) ?
            prefix :
            scan_op(prefix, output);
    }

    //@}  end member group
};

/** @} */       // end group WarpModule

}               // CUB namespace
CUB_NS_POSTFIX  // Optional outer namespace(s)
