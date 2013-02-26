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
 * The cub::WarpScan type provides variants of parallel prefix scan across threads within a CUDA warp.
 */

#pragma once

#include "../thread/thread_load.cuh"
#include "../thread/thread_store.cuh"
#include "../device_props.cuh"
#include "../type_utils.cuh"
#include "../operators.cuh"
#include "../ns_wrapper.cuh"

CUB_NS_PREFIX

/// CUB namespace
namespace cub {

/**
 * \addtogroup SimtCoop
 * @{
 */

/**
 * \brief The WarpScan type provides variants of parallel prefix scan across threads within a CUDA warp.  ![](warp_scan_logo.png)
 *
 * <b>Overview</b>
 * \par
 * Given a list of input elements and a binary reduction operator, <em>prefix scan</em>
 * produces an output list where each element is computed to be the reduction
 * of the elements occurring earlier in the input list.  <em>Prefix sum</em>
 * connotes a prefix scan with the addition operator. The term \em inclusive means
 * that each result includes the corresponding input operand in the partial sum.
 * The term \em exclusive means that each result does not include the corresponding
 * input operand in the partial reduction.
 *
 * \par
 * These parallel prefix scan variants assume the input and output lists to
 * be logically partitioned among threads with warp thread-lane-<em>i</em>
 * having the <em>i</em><sup>th</sup> input and output elements.  To minimize
 * synchronization overhead for operations involving the cumulative
 * \p aggregate and \p warp_prefix_op, these values are only valid in
 * thread-lane<sub>0</sub>.
 *
 * \tparam T                        The scan input/output element type
 * \tparam WARPS                    The number of "logical" warps performing concurrent warp scans
 * \tparam LOGICAL_WARP_THREADS     <b>[optional]</b> The number of threads per "logical" warp (may be less than the number of hardware warp threads).  Default is the warp size associated with the CUDA Compute Capability targeted by the compiler (e.g., 32 warps for SM20).
 *
 * <b>Performance Features and Considerations</b>
 * \par
 * - Support for "logical" warps smaller than the physical warp size (e.g., 8 threads).
 * - Support for non-commutative binary associative scan functors.
 * - Support for concurrent scans within multiple warps.
 * - Zero bank conflicts for most types.
 * - After any operation, a subsequent threadblock barrier (<tt>__syncthreads</tt>) is
 *   required if the supplied WarpScan::SmemStorage is to be reused/repurposed by the threadblock.
 * - The operations are most efficient (lowest instruction overhead) when:
 *      - The data type \p T is a built-in primitive or CUDA vector type (e.g.,
 *        \p short, \p int2, \p double, \p float2, etc.)  Otherwise the implementation
 *        may use memory fences to prevent reference reordering of
 *        non-primitive types.
 *      - Performing exclusive scans. The implementation may use guarded
 *        shared memory accesses for inclusive scans (other than prefix sum)
 *        because no identity element is provided.
 *
 * <b>Algorithm</b>
 * \par
 * These parallel prefix scan variants implement a warp-synchronous
 * Kogge-Stone algorithm having <em>O</em>(log<em>n</em>)
 * steps and <em>O</em>(<em>n</em>log<em>n</em>) work complexity,
 * where <em>n</em> = \p LOGICAL_WARP_THREADS (which defaults to the warp
 * size associated with the CUDA Compute Capability targeted by the compiler).
 * <br><br>
 * \image html kogge_stone_scan.png
 * <center><b>Data flow within a 16-thread Kogge-Stone scan construction.  Junctions represent binary operators.</b></center>
 * <br>
 *
 * <b>Examples</b>
 *
 * \par
 * - <b>Example 1:</b> Simple exclusive prefix sum for one warp
 *      \code
 *      #include <cub.cuh>
 *
 *      __global__ void SomeKernel(...)
 *      {
 *          // A parameterized WarpScan type for use with one warp on type int.
 *          typedef cub::WarpScan<int, 1> WarpScan;
 *
 *          // Opaque shared memory for WarpScan
 *          __shared__ typename WarpScan::SmemStorage smem_storage;
 *
 *          // Perform prefix sum of threadIds in first warp
 *          if (threadIdx.x < 32)
 *          {
 *              int output, input = threadIdx.x;
 *              WarpScan::ExclusiveSum(smem_storage, input, output);
 *
 *              printf("tid(%d) output(%d)\n\n", threadIdx.x, output);
 *          }
 *      \endcode
 *
 *      Printed output:
 *      \code
 *      tid(0) output(0)
 *      tid(1) output(0)
 *      tid(2) output(1)
 *      tid(3) output(3)
 *      tid(4) output(6)
 *      ...
 *      tid(31) output(465)
 *      \endcode
 *
 * \par
 * - <b>Example 2:</b> More sophisticated exclusive prefix sum for one warp
 *      \code
 *      #include <cub.cuh>
 *
 *      __global__ void SomeKernel(...)
 *      {
 *          // A parameterized WarpScan type for use with one warp on type int.
 *          typedef cub::WarpScan<int, 1> WarpScan;
 *
 *          // Opaque shared memory for WarpScan
 *          __shared__ typename WarpScan::SmemStorage smem_storage;
 *
 *          // Perform prefix sum of 2s, all seeded with a warp prefix value of 10
 *          if (threadIdx.x < 32)
 *          {
 *              int output, aggregate, input = 2, warp_prefix_op = 10;
 *              WarpScan::ExclusiveSum(smem_storage, input, output, warp_prefix_op, warp_aggregate);
 *
 *              printf("tid(%d) output(%d)\n\n", threadIdx.x, output);
 *              if (threadIdx.x == 0)
 *                  printf("computed aggregate(%d), updated warp_prefix_op(%d)\n", aggregate, warp_prefix_op);
 *          }
 *          \endcode
 *
 *      Printed output:
 *      \code
 *      tid(0) output(10)
 *      tid(1) output(12)
 *      tid(2) output(14)
 *      tid(3) output(16)
 *      tid(4) output(18)
 *      ...
 *      tid(31) output(72)
 *
 *      computed aggregate(74), udpated warp_prefix_op(84)
 *      \endcode
 */
template <
    typename    T,
    int         WARPS,
    int         LOGICAL_WARP_THREADS = DeviceProps::WARP_THREADS>
class WarpScan
{
    //---------------------------------------------------------------------
    // Constants and typedefs
    //---------------------------------------------------------------------

private:

    /// WarpScan algorithmic variants
    enum WarpScanPolicy
    {
        SHFL_SCAN,          // Warp-synchronous SHFL-based scan
        SMEM_SCAN,          // Warp-synchronous smem-based scan
    };

    /// Constants
    enum
    {
        POW_OF_TWO = ((LOGICAL_WARP_THREADS & (LOGICAL_WARP_THREADS - 1)) == 0),

        /// Use SHFL-scan if (architecture is >= SM30) and (T is a primitive) and (T is 4-bytes or smaller) and (LOGICAL_WARP_THREADS is a power-of-two)
        POLICY = ((PTX_ARCH >= 300) && Traits<T>::PRIMITIVE && (sizeof(T) <= 4) && POW_OF_TWO) ?
            SHFL_SCAN :
            SMEM_SCAN,
    };


    /**
     * Specialized WarpScan implementations
     */
    template <int POLICY, int DUMMY = 0>
    struct WarpScanInternal;


    /**
     * Warpscan specialized for SHFL_SCAN variant
     */
    template <int DUMMY>
    struct WarpScanInternal<SHFL_SCAN, DUMMY>
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
        template <typename T>
        static __device__ __forceinline__ T Broadcast(
            SmemStorage     &smem_storage,      ///< [in] Shared reference to opaque SmemStorage layout
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
            SmemStorage     &smem_storage,      ///< [in] Shared reference to opaque SmemStorage layout
            unsigned int    input,              ///< [in] Calling thread's input item.
            unsigned int    &output,            ///< [out] Calling thread's output item.  May be aliased with \p input.
            unsigned int    &warp_aggregate)   ///< [out] <b>[<em>warp-lane</em><sub>0</sub> only]</b> Warp-wide aggregate reduction of input items.
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
            SmemStorage     &smem_storage,      ///< [in] Shared reference to opaque SmemStorage layout
            float           input,              ///< [in] Calling thread's input item.
            float           &output,            ///< [out] Calling thread's output item.  May be aliased with \p input.
            float           &warp_aggregate)   ///< [out] <b>[<em>warp-lane</em><sub>0</sub> only]</b> Warp-wide aggregate reduction of input items.
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
        template <typename T>
        static __device__ __forceinline__ void InclusiveSum(
            SmemStorage     &smem_storage,      ///< [in] Shared reference to opaque SmemStorage layout
            T               input,              ///< [in] Calling thread's input item.
            T               &output,            ///< [out] Calling thread's output item.  May be aliased with \p input.
            T               &warp_aggregate)   ///< [out] <b>[<em>warp-lane</em><sub>0</sub> only]</b> Warp-wide aggregate reduction of input items.
        {
            // Cast as unsigned int
            unsigned int    &uinput             = reinterpret_cast<unsigned int&>(input);
            unsigned int    &uoutput            = reinterpret_cast<unsigned int&>(output);
            unsigned int    &uwarp_aggregate   = reinterpret_cast<unsigned int&>(warp_aggregate);

            InclusiveSum(smem_storage, uinput, uoutput, uwarp_aggregate);
        }

        /// Inclusive prefix sum
        static __device__ __forceinline__ void InclusiveSum(
            SmemStorage     &smem_storage,      ///< [in] Shared reference to opaque SmemStorage layout
            T               input,              ///< [in] Calling thread's input item.
            T               &output)            ///< [out] Calling thread's output item.  May be aliased with \p input.
        {
            T warp_aggregate;
            InclusiveSum(smem_storage, input, output, warp_aggregate);
        }


        /// Inclusive scan with aggregate
        template <typename ScanOp>
        static __device__ __forceinline__ void InclusiveScan(
            SmemStorage     &smem_storage,      ///< [in] Shared reference to opaque SmemStorage layout
            T               input,              ///< [in] Calling thread's input item.
            T               &output,            ///< [out] Calling thread's output item.  May be aliased with \p input.
            ScanOp          scan_op,            ///< [in] Binary associative scan functor.
            T               &warp_aggregate)   ///< [out] <b>[<em>warp-lane</em><sub>0</sub> only]</b> Warp-wide aggregate reduction of input items.
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
            SmemStorage     &smem_storage,      ///< [in] Shared reference to opaque SmemStorage layout
            T               input,              ///< [in] Calling thread's input item.
            T               &output,            ///< [out] Calling thread's output item.  May be aliased with \p input.
            ScanOp          scan_op)            ///< [in] Binary associative scan functor.
        {
            T warp_aggregate;
            InclusiveScan(smem_storage, input, output, scan_op, warp_aggregate);
        }

        /// Exclusive scan with aggregate
        template <typename ScanOp>
        static __device__ __forceinline__ void ExclusiveScan(
            SmemStorage     &smem_storage,      ///< [in] Shared reference to opaque SmemStorage layout
            T               input,              ///< [in] Calling thread's input item.
            T               &output,            ///< [out] Calling thread's output item.  May be aliased with \p input.
            const T         &identity,          ///< [in] Identity value
            ScanOp          scan_op,            ///< [in] Binary associative scan functor.
            T               &warp_aggregate)   ///< [out] <b>[<em>warp-lane</em><sub>0</sub> only]</b> Warp-wide aggregate reduction of input items.
        {
            // Compute inclusive scan
            T inclusive;
            InclusiveScan(smem_storage, input, inclusive, scan_op, warp_aggregate);

            unsigned int    &uinclusive     = reinterpret_cast<unsigned int&>(inclusive);
            unsigned int    &uidentity      = reinterpret_cast<unsigned int&>(identity);
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
            SmemStorage     &smem_storage,      ///< [in] Shared reference to opaque SmemStorage layout
            T               input,              ///< [in] Calling thread's input item.
            T               &output,            ///< [out] Calling thread's output item.  May be aliased with \p input.
            const T         &identity,          ///< [in] Identity value
            ScanOp          scan_op)            ///< [in] Binary associative scan functor.
        {
            T warp_aggregate;
            ExclusiveScan(smem_storage, input, output, identity, scan_op, warp_aggregate);
        }


        /// Exclusive scan with aggregate, without identity
        template <typename ScanOp>
        static __device__ __forceinline__ void ExclusiveScan(
            SmemStorage     &smem_storage,      ///< [in] Shared reference to opaque SmemStorage layout
            T               input,              ///< [in] Calling thread's input item.
            T               &output,            ///< [out] Calling thread's output item.  May be aliased with \p input.
            ScanOp          scan_op,            ///< [in] Binary associative scan functor.
            T               &warp_aggregate)   ///< [out] <b>[<em>warp-lane</em><sub>0</sub> only]</b> Warp-wide aggregate reduction of input items.
        {
            // Compute inclusive scan
            T inclusive;
            InclusiveScan(smem_storage, input, inclusive, scan_op);

            unsigned int    &uinclusive     = reinterpret_cast<unsigned int&>(inclusive);
            unsigned int    &uoutput        = reinterpret_cast<unsigned int&>(output);

            asm(
                "shfl.up.b32 %0, %1, 1, %2;"
                : "=r"(uoutput) : "r"(uinclusive), "r"(SHFL_MASK));
        }


        /// Exclusive scan without identity
        template <typename ScanOp>
        static __device__ __forceinline__ void ExclusiveScan(
            SmemStorage     &smem_storage,      ///< [in] Shared reference to opaque SmemStorage layout
            T               input,              ///< [in] Calling thread's input item.
            T               &output,            ///< [out] Calling thread's output item.  May be aliased with \p input.
            ScanOp          scan_op)            ///< [in] Binary associative scan functor.
        {
            T warp_aggregate;
            ExclusiveScan(smem_storage, input, output, scan_op, warp_aggregate);
        }
    };


    /**
     * Warpscan specialized for SMEM_SCAN
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

            /// Single variable for broadcasting aggregate, etc.
            T broadcast;
        };


        /// Broadcast
        static __device__ __forceinline__ T Broadcast(
            SmemStorage     &smem_storage,      ///< [in] Shared reference to opaque SmemStorage layout
            T               input,              ///< [in] The value to broadcast
            unsigned int    src_lane)           ///< [in] Which warp lane is to do the broacasting
        {
            unsigned int lane_id = (WARPS == 1) ? threadIdx.x : (threadIdx.x & (LOGICAL_WARP_THREADS - 1));

            if (lane_id == src_lane)
            {
                ThreadStore<PTX_STORE_VS>(&smem_storage.broadcast, input);
            }

            return ThreadLoad<PTX_LOAD_VS>(&smem_storage.broadcast);
        }

        /// Basic inclusive scan
        template <
            bool HAS_IDENTITY,
            bool SHARE_FINAL,
            typename ScanOp>
        static __device__ __forceinline__ T BasicScan(
            SmemStorage     &smem_storage,      ///< Shared reference to opaque SmemStorage layout
            unsigned int    warp_id,            ///< Warp id
            unsigned int    lane_id,            ///< thread-lane id
            T               partial,            ///< Calling thread's input partial reduction
            ScanOp          scan_op)            ///< Binary associative scan functor
        {
            // Iterate scan steps
            #pragma unroll
            for (int STEP = 0; STEP < STEPS; STEP++)
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
            }


            if (SHARE_FINAL)
            {
                // Share partial into buffer
                ThreadStore<PTX_STORE_VS>(&smem_storage.warp_scan[warp_id][HALF_WARP_THREADS + lane_id], partial);
            }

            return partial;
        }


        /// Inclusive prefix sum
        static __device__ __forceinline__ void InclusiveSum(
            SmemStorage     &smem_storage,      ///< [in] Shared reference to opaque SmemStorage layout
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
            SmemStorage     &smem_storage,      ///< [in] Shared reference to opaque SmemStorage layout
            T               input,              ///< [in] Calling thread's input item.
            T               &output,            ///< [out] Calling thread's output item.  May be aliased with \p input.
            T               &warp_aggregate)   ///< [out] <b>[<em>warp-lane</em><sub>0</sub> only]</b> Warp-wide aggregate reduction of input items.
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

            // Retrieve aggregate in thread-lane<sub>0</sub>
            warp_aggregate = ThreadLoad<PTX_LOAD_VS>(&smem_storage.warp_scan[warp_id][WARP_SMEM_ELEMENTS - 1]);
        }


        /// Inclusive scan
        template <typename ScanOp>
        static __device__ __forceinline__ void InclusiveScan(
            SmemStorage     &smem_storage,      ///< [in] Shared reference to opaque SmemStorage layout
            T               input,              ///< [in] Calling thread's input item.
            T               &output,            ///< [out] Calling thread's output item.  May be aliased with \p input.
            ScanOp          scan_op)            ///< [in] Binary associative scan functor.
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
            SmemStorage     &smem_storage,      ///< [in] Shared reference to opaque SmemStorage layout
            T               input,              ///< [in] Calling thread's input item.
            T               &output,            ///< [out] Calling thread's output item.  May be aliased with \p input.
            ScanOp          scan_op,            ///< [in] Binary associative scan functor.
            T               &warp_aggregate)   ///< [out] <b>[<em>warp-lane</em><sub>0</sub> only]</b> Warp-wide aggregate reduction of input items.
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
            SmemStorage     &smem_storage,      ///< [in] Shared reference to opaque SmemStorage layout
            T               input,              ///< [in] Calling thread's input item.
            T               &output,            ///< [out] Calling thread's output item.  May be aliased with \p input.
            const T         &identity,          ///< [in] Identity value
            ScanOp          scan_op)            ///< [in] Binary associative scan functor.
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
            SmemStorage     &smem_storage,      ///< [in] Shared reference to opaque SmemStorage layout
            T               input,              ///< [in] Calling thread's input item.
            T               &output,            ///< [out] Calling thread's output item.  May be aliased with \p input.
            const T         &identity,          ///< [in] Identity value
            ScanOp          scan_op,            ///< [in] Binary associative scan functor.
            T               &warp_aggregate)   ///< [out] <b>[<em>warp-lane</em><sub>0</sub> only]</b> Warp-wide aggregate reduction of input items.
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
            SmemStorage     &smem_storage,      ///< [in] Shared reference to opaque SmemStorage layout
            T               input,              ///< [in] Calling thread's input item.
            T               &output,            ///< [out] Calling thread's output item.  May be aliased with \p input.
            ScanOp          scan_op)            ///< [in] Binary associative scan functor.
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
            SmemStorage     &smem_storage,      ///< [in] Shared reference to opaque SmemStorage layout
            T               input,              ///< [in] Calling thread's input item.
            T               &output,            ///< [out] Calling thread's output item.  May be aliased with \p input.
            ScanOp          scan_op,            ///< [in] Binary associative scan functor.
            T               &warp_aggregate)   ///< [out] <b>[<em>warp-lane</em><sub>0</sub> only]</b> Warp-wide aggregate reduction of input items.
        {
            // Warp id
            unsigned int warp_id = (WARPS == 1) ? 0 : (threadIdx.x / LOGICAL_WARP_THREADS);

            // Exclusive warp scan (which does share final)
            ExclusiveScan(smem_storage, input, output, scan_op);

            // Retrieve aggregate
            warp_aggregate = ThreadLoad<PTX_LOAD_VS>(&smem_storage.warp_scan[warp_id][WARP_SMEM_ELEMENTS - 1]);
        }

    };


    /// Shared memory storage layout type for WarpScan
    typedef typename WarpScanInternal<POLICY>::SmemStorage _SmemStorage;


public:

    /// The operations exposed by WarpScan require shared memory of this
    /// type.  This opaque storage can be allocated directly using the
    /// <tt>__shared__</tt> keyword.  Alternatively, it can be aliased to
    /// externally allocated shared memory or <tt>union</tt>'d with other types
    /// to facilitate shared memory reuse.
    typedef _SmemStorage SmemStorage;


    /******************************************************************//**
     * \name Inclusive prefix sums
     *********************************************************************/
    //@{

    /**
     * \brief Computes an inclusive prefix sum in each logical warp.
     *
     * \smemreuse
     */
    static __device__ __forceinline__ void InclusiveSum(
        SmemStorage     &smem_storage,      ///< [in] Shared reference to opaque SmemStorage layout
        T               input,              ///< [in] Calling thread's input item.
        T               &output)            ///< [out] Calling thread's output item.  May be aliased with \p input.
    {
        WarpScanInternal<POLICY>::InclusiveSum(smem_storage, input, output);
    }


    /**
     * \brief Computes an inclusive prefix sum in each logical warp.  Also computes the warp-wide \p aggregate of all inputs for thread-lane<sub>0</sub>.
     *
     * The \p aggregate is undefined in threads other than thread-lane<sub>0</sub>.
     *
     * \smemreuse
     */
    static __device__ __forceinline__ void InclusiveSum(
        SmemStorage     &smem_storage,      ///< [in] Shared reference to opaque SmemStorage layout
        T               input,              ///< [in] Calling thread's input item.
        T               &output,            ///< [out] Calling thread's output item.  May be aliased with \p input.
        T               &warp_aggregate)   ///< [out] <b>[<em>warp-lane</em><sub>0</sub> only]</b> Warp-wide aggregate reduction of input items.
    {
        WarpScanInternal<POLICY>::InclusiveSum(smem_storage, input, output, warp_aggregate);
    }


    /**
     * \brief Computes an inclusive prefix sum in each logical warp.  The functor \p warp_prefix_op is evaluated by <em>warp-lane</em><sub>0</sub> to provide the preceding (or "base") value that logically prefixes the warp's scan inputs.  Also computes the warp-wide \p aggregate of all inputs for thread-lane<sub>0</sub>.  The \p warp_prefix_op is further updated by the value of \p aggregate.
     *
     * The \p aggregate and \p warp_prefix_op are undefined in threads other than thread-lane<sub>0</sub>.
     *
     * \smemreuse
     *
     * \tparam WarpPrefixOp                 <b>[inferred]</b> A call-back unary functor of the model <em>operator()(T warp_aggregate)</em>
     */
    template <typename WarpPrefixOp>
    static __device__ __forceinline__ void InclusiveSum(
        SmemStorage     &smem_storage,      ///< [in] Shared reference to opaque SmemStorage layout
        T               input,              ///< [in] Calling thread's input item.
        T               &output,            ///< [out] Calling thread's output item.  May be aliased with \p input.
        T               &warp_aggregate,   ///< [out] <b>[<em>warp-lane</em><sub>0</sub> only]</b> Warp-wide aggregate reduction of input items, exclusive of the \p warp_prefix_op value
        WarpPrefixOp    &warp_prefix_op)   ///< [in-out] <b>[<em>warp-lane</em><sub>0</sub> only]</b> A call-back unary functor of the model <em>operator()(T warp_aggregate)</em> for specifying the appropriate external prefix (rather than \p identity), to be invoked internally by <em>warp-lane</em><sub>0</sub>.  When provided the warp-wide aggregate of input items (also returned above), this functor is expected to return the logical warp-wide prefix to be applied during the scan operation.  Can be stateful.
    {
        // Compute inclusive warp scan
        InclusiveSum(smem_storage, input, output, warp_aggregate);

        // Compute warp-wide prefix from aggregate, then broadcast to other lanes
        T prefix;
        unsigned int lane_id = (WARPS == 1) ? threadIdx.x : (threadIdx.x & (LOGICAL_WARP_THREADS - 1));
        if (lane_id == 0)
        {
            prefix = warp_prefix_op(warp_aggregate);
        }
        prefix = WarpScanInternal<POLICY>::Broadcast(smem_storage, prefix, 0);

        // Update output
        output = prefix + output;
    }


    //@}

    /******************************************************************//**
     * \name Exclusive prefix sums
     *********************************************************************/
    //@{

    /**
     * \brief Computes an exclusive prefix sum in each logical warp.
     *
     * \smemreuse
     */
    static __device__ __forceinline__ void ExclusiveSum(
        SmemStorage     &smem_storage,      ///< [in] Shared reference to opaque SmemStorage layout
        T               input,              ///< [in] Calling thread's input item.
        T               &output)            ///< [out] Calling thread's output item.  May be aliased with \p input.
    {
        // Compute exclusive warp scan from inclusive warp scan
        T inclusive;
        InclusiveSum(smem_storage, input, inclusive);
        output = inclusive - input;
    }


    /**
     * \brief Computes an exclusive prefix sum in each logical warp.  Also computes the warp-wide \p aggregate of all inputs for thread-lane<sub>0</sub>.
     *
     * The \p aggregate is undefined in threads other than thread-lane<sub>0</sub>.
     *
     * \smemreuse
     */
    static __device__ __forceinline__ void ExclusiveSum(
        SmemStorage     &smem_storage,      ///< [in] Shared reference to opaque SmemStorage layout
        T               input,              ///< [in] Calling thread's input item.
        T               &output,            ///< [out] Calling thread's output item.  May be aliased with \p input.
        T               &warp_aggregate)   ///< [out] <b>[<em>warp-lane</em><sub>0</sub> only]</b> Warp-wide aggregate reduction of input items.
    {
        // Compute exclusive warp scan from inclusive warp scan
        T inclusive;
        InclusiveSum(smem_storage, input, inclusive, warp_aggregate);
        output = inclusive - input;
    }


    /**
     * \brief Computes an exclusive prefix sum in each logical warp.  The functor \p warp_prefix_op is evaluated by <em>warp-lane</em><sub>0</sub> to provide the preceding (or "base") value that logically prefixes the warp's scan inputs.  Also computes the warp-wide \p aggregate of all inputs for thread-lane<sub>0</sub>.  The \p warp_prefix_op is further updated by the value of \p aggregate.
     *
     * The \p aggregate and \p warp_prefix_op are undefined in threads other than thread-lane<sub>0</sub>.
     *
     * \smemreuse
     *
     * \tparam WarpPrefixOp                 <b>[inferred]</b> A call-back unary functor of the model <em>operator()(T warp_aggregate)</em>
     */
    template <typename WarpPrefixOp>
    static __device__ __forceinline__ void ExclusiveSum(
        SmemStorage     &smem_storage,      ///< [in] Shared reference to opaque SmemStorage layout
        T               input,              ///< [in] Calling thread's input item.
        T               &output,            ///< [out] Calling thread's output item.  May be aliased with \p input.
        T               &warp_aggregate,   ///< [out] <b>[<em>warp-lane</em><sub>0</sub> only]</b> Warp-wide aggregate reduction of input items, exclusive of the \p warp_prefix_op value.
        WarpPrefixOp    &warp_prefix_op)   ///< [in-out] <b>[<em>warp-lane</em><sub>0</sub> only]</b> A call-back unary functor of the model <em>operator()(T warp_aggregate)</em> for specifying the appropriate external prefix (rather than \p identity), to be invoked internally by <em>warp-lane</em><sub>0</sub>.  When provided the warp-wide aggregate of input items (also returned above), this functor is expected to return the logical warp-wide prefix to be applied during the scan operation.  Can be stateful.
    {
        // Compute exclusive warp scan from inclusive warp scan
        T inclusive;
        InclusiveSum(smem_storage, input, inclusive, warp_aggregate, warp_prefix_op);
        output = inclusive - input;
    }


    //@}
    /******************************************************************//**
     * \name Inclusive prefix scans
     *********************************************************************/
    //@{

    /**
     * \brief Computes an inclusive prefix sum using the specified binary scan functor in each logical warp.
     *
     * \smemreuse
     *
     * \tparam ScanOp     <b>[inferred]</b> Binary scan functor type
     */
    template <typename ScanOp>
    static __device__ __forceinline__ void InclusiveScan(
        SmemStorage     &smem_storage,      ///< [in] Shared reference to opaque SmemStorage layout
        T               input,              ///< [in] Calling thread's input item.
        T               &output,            ///< [out] Calling thread's output item.  May be aliased with \p input.
        ScanOp          scan_op)            ///< [in] Binary associative scan functor.
    {
        WarpScanInternal<POLICY>::InclusiveScan(smem_storage, input, output, scan_op);
    }


    /**
     * \brief Computes an inclusive prefix sum using the specified binary scan functor in each logical warp.  Also computes the warp-wide \p aggregate of all inputs for thread-lane<sub>0</sub>.
     *
     * The \p aggregate is undefined in threads other than thread-lane<sub>0</sub>.
     *
     * \smemreuse
     *
     * \tparam ScanOp     <b>[inferred]</b> Binary scan functor type
     */
    template <typename ScanOp>
    static __device__ __forceinline__ void InclusiveScan(
        SmemStorage     &smem_storage,      ///< [in] Shared reference to opaque SmemStorage layout
        T               input,              ///< [in] Calling thread's input item.
        T               &output,            ///< [out] Calling thread's output item.  May be aliased with \p input.
        ScanOp          scan_op,            ///< [in] Binary associative scan functor.
        T               &warp_aggregate)   ///< [out] <b>[<em>warp-lane</em><sub>0</sub> only]</b> Warp-wide aggregate reduction of input items.
    {
        WarpScanInternal<POLICY>::InclusiveScan(smem_storage, input, output, scan_op, warp_aggregate);
    }


    /**
     * \brief Computes an inclusive prefix sum using the specified binary scan functor in each logical warp.  The functor \p warp_prefix_op is evaluated by <em>warp-lane</em><sub>0</sub> to provide the preceding (or "base") value that logically prefixes the warp's scan inputs.  Also computes the warp-wide \p aggregate of all inputs for thread-lane<sub>0</sub>.  The \p warp_prefix_op is further updated by the value of \p aggregate.
     *
     * The \p aggregate and \p warp_prefix_op are undefined in threads other than thread-lane<sub>0</sub>.
     *
     * \smemreuse
     *
     * \tparam ScanOp                       <b>[inferred]</b> Binary scan functor type
     * \tparam WarpPrefixOp                 <b>[inferred]</b> A call-back unary functor of the model <em>operator()(T warp_aggregate)</em>
     */
    template <
        typename ScanOp,
        typename WarpPrefixOp>
    static __device__ __forceinline__ void InclusiveScan(
        SmemStorage     &smem_storage,      ///< [in] Shared reference to opaque SmemStorage layout
        T               input,              ///< [in] Calling thread's input item.
        T               &output,            ///< [out] Calling thread's output item.  May be aliased with \p input.
        ScanOp          scan_op,            ///< [in] Binary associative scan functor.
        T               &warp_aggregate,   ///< [out] <b>[<em>warp-lane</em><sub>0</sub> only]</b> Warp-wide aggregate reduction of input items, exclusive of the \p warp_prefix_op value.
        WarpPrefixOp    &warp_prefix_op)   ///< [in-out] <b>[<em>warp-lane</em><sub>0</sub> only]</b> A call-back unary functor of the model <em>operator()(T warp_aggregate)</em> for specifying the appropriate external prefix (rather than \p identity), to be invoked internally by <em>warp-lane</em><sub>0</sub>.  When provided the warp-wide aggregate of input items (also returned above), this functor is expected to return the logical warp-wide prefix to be applied during the scan operation.  Can be stateful.
    {
        // Compute inclusive warp scan
        InclusiveScan(smem_storage, input, output, scan_op, warp_aggregate);

        // Compute warp-wide prefix from aggregate, then broadcast to other lanes
        T prefix;
        unsigned int lane_id = (WARPS == 1) ? threadIdx.x : (threadIdx.x & (LOGICAL_WARP_THREADS - 1));
        if (lane_id == 0)
        {
            prefix = warp_prefix_op(warp_aggregate);
        }
        prefix = WarpScanInternal<POLICY>::Broadcast(smem_storage, prefix, 0);

        // Update output
        output = scan_op(prefix, output);
    }


    //@}
    /******************************************************************//**
     * \name Exclusive prefix scans
     *********************************************************************/
    //@{

    /**
     * \brief Computes an exclusive prefix scan using the specified binary scan functor in each logical warp.
     *
     * \smemreuse
     *
     * \tparam ScanOp     <b>[inferred]</b> Binary scan functor type
     */
    template <typename ScanOp>
    static __device__ __forceinline__ void ExclusiveScan(
        SmemStorage     &smem_storage,      ///< [in] Shared reference to opaque SmemStorage layout
        T               input,              ///< [in] Calling thread's input item.
        T               &output,            ///< [out] Calling thread's output item.  May be aliased with \p input.
        const T         &identity,          ///< [in] Identity value
        ScanOp          scan_op)            ///< [in] Binary associative scan functor.
    {
        WarpScanInternal<POLICY>::ExclusiveScan(smem_storage, input, output, identity, scan_op);
    }


    /**
     * \brief Computes an exclusive prefix scan using the specified binary scan functor in each logical warp.  Also computes the warp-wide \p aggregate of all inputs for thread-lane<sub>0</sub>.
     *
     * The \p aggregate is undefined in threads other than thread-lane<sub>0</sub>.
     *
     * \smemreuse
     *
     * \tparam ScanOp     <b>[inferred]</b> Binary scan functor type
     */
    template <typename ScanOp>
    static __device__ __forceinline__ void ExclusiveScan(
        SmemStorage     &smem_storage,      ///< [in] Shared reference to opaque SmemStorage layout
        T               input,              ///< [in] Calling thread's input item.
        T               &output,            ///< [out] Calling thread's output item.  May be aliased with \p input.
        const T         &identity,          ///< [in] Identity value
        ScanOp          scan_op,            ///< [in] Binary associative scan functor.
        T               &warp_aggregate)   ///< [out] <b>[<em>warp-lane</em><sub>0</sub> only]</b> Warp-wide aggregate reduction of input items.
    {
        WarpScanInternal<POLICY>::ExclusiveScan(smem_storage, input, output, identity, scan_op, warp_aggregate);
    }


    /**
     * \brief Computes an exclusive prefix scan using the specified binary scan functor in each logical warp.  The functor \p warp_prefix_op is evaluated by <em>warp-lane</em><sub>0</sub> to provide the preceding (or "base") value that logically prefixes the warp's scan inputs.  Also computes the warp-wide \p aggregate of all inputs for thread-lane<sub>0</sub>.  The \p warp_prefix_op is further updated by the value of \p aggregate.
     *
     * The \p aggregate and \p warp_prefix_op are undefined in threads other than thread-lane<sub>0</sub>.
     *
     * \smemreuse
     *
     * \tparam ScanOp                       <b>[inferred]</b> Binary scan functor type
     * \tparam WarpPrefixOp                 <b>[inferred]</b> A call-back unary functor of the model <em>operator()(T warp_aggregate)</em>
     */
    template <
        typename ScanOp,
        typename WarpPrefixOp>
    static __device__ __forceinline__ void ExclusiveScan(
        SmemStorage     &smem_storage,      ///< [in] Shared reference to opaque SmemStorage layout
        T               input,              ///< [in] Calling thread's input item.
        T               &output,            ///< [out] Calling thread's output item.  May be aliased with \p input.
        const T         &identity,          ///< [in] Identity value
        ScanOp          scan_op,            ///< [in] Binary associative scan functor.
        T               &warp_aggregate,   ///< [out] <b>[<em>warp-lane</em><sub>0</sub> only]</b> Warp-wide aggregate reduction of input items, exclusive of the \p warp_prefix_op value.
        WarpPrefixOp    &warp_prefix_op)    ///< [in-out] <b>[<em>warp-lane</em><sub>0</sub> only]</b> A call-back unary functor of the model <em>operator()(T warp_aggregate)</em> for specifying the appropriate external prefix (rather than \p identity), to be invoked internally by <em>warp-lane</em><sub>0</sub>.  When provided the warp-wide aggregate of input items (also returned above), this functor is expected to return the logical warp-wide prefix to be applied during the scan operation.  Can be stateful.
    {
        // Exclusive warp scan
        ExclusiveScan(smem_storage, input, output, identity, scan_op, warp_aggregate);

        // Compute warp-wide prefix from aggregate, then broadcast to other lanes
        T prefix;
        unsigned int lane_id = (WARPS == 1) ? threadIdx.x : (threadIdx.x & (LOGICAL_WARP_THREADS - 1));
        if (lane_id == 0)
        {
            prefix = warp_prefix_op(warp_aggregate);
        }
        prefix = WarpScanInternal<POLICY>::Broadcast(smem_storage, prefix, 0);

        // Update output
        output = (lane_id == 0) ?
            prefix :
            scan_op(prefix, output);
    }


    //@}
    /******************************************************************//**
     * \name Exclusive prefix scans (without supplied identity)
     *********************************************************************/
    //@{


    /**
     * \brief Computes an exclusive prefix scan using the specified binary scan functor in each logical warp.  Because no identity value is supplied, the \p output computed for thread-lane<sub>0</sub> is invalid.
     *
     * \smemreuse
     *
     * \tparam ScanOp     <b>[inferred]</b> Binary scan functor type
     */
    template <typename ScanOp>
    static __device__ __forceinline__ void ExclusiveScan(
        SmemStorage     &smem_storage,      ///< [in] Shared reference to opaque SmemStorage layout
        T               input,              ///< [in] Calling thread's input item.
        T               &output,            ///< [out] Calling thread's output item.  May be aliased with \p input.
        ScanOp          scan_op)            ///< [in] Binary associative scan functor.
    {
        WarpScanInternal<POLICY>::ExclusiveScan(smem_storage, input, output, scan_op);
    }


    /**
     * \brief Computes an exclusive prefix scan using the specified binary scan functor in each logical warp.  Because no identity value is supplied, the \p output computed for thread-lane<sub>0</sub> is invalid.  Also computes the warp-wide \p aggregate of all inputs for thread-lane<sub>0</sub>.
     *
     * The \p aggregate is undefined in threads other than thread-lane<sub>0</sub>.
     *
     * \smemreuse
     *
     * \tparam ScanOp     <b>[inferred]</b> Binary scan functor type
     */
    template <typename ScanOp>
    static __device__ __forceinline__ void ExclusiveScan(
        SmemStorage     &smem_storage,      ///< [in] Shared reference to opaque SmemStorage layout
        T               input,              ///< [in] Calling thread's input item.
        T               &output,            ///< [out] Calling thread's output item.  May be aliased with \p input.
        ScanOp          scan_op,            ///< [in] Binary associative scan functor.
        T               &warp_aggregate)   ///< [out] <b>[<em>warp-lane</em><sub>0</sub> only]</b> Warp-wide aggregate reduction of input items.
    {
        WarpScanInternal<POLICY>::ExclusiveScan(smem_storage, input, output, scan_op, warp_aggregate);
    }


    /**
     * \brief Computes an exclusive prefix scan using the specified binary scan functor in each logical warp.  The \p warp_prefix_op value from thread-thread-lane<sub>0</sub> is applied to all scan outputs.  Also computes the warp-wide \p aggregate of all inputs for thread-thread-lane<sub>0</sub>.  The \p warp_prefix_op is further updated by the value of \p aggregate.
     *
     * The \p aggregate and \p warp_prefix_op are undefined in threads other than thread-lane<sub>0</sub>.
     *
     * \smemreuse
     *
     * \tparam ScanOp                       <b>[inferred]</b> Binary scan functor type
     * \tparam WarpPrefixOp                 <b>[inferred]</b> A call-back unary functor of the model <em>operator()(T warp_aggregate)</em>
     */
    template <
        typename ScanOp,
        typename WarpPrefixOp>
    static __device__ __forceinline__ void ExclusiveScan(
        SmemStorage     &smem_storage,      ///< [in] Shared reference to opaque SmemStorage layout
        T               input,              ///< [in] Calling thread's input item.
        T               &output,            ///< [out] Calling thread's output item.  May be aliased with \p input.
        ScanOp          scan_op,            ///< [in] Binary associative scan functor.
        T               &warp_aggregate,    ///< [out] <b>[<em>warp-lane</em><sub>0</sub> only]</b> Warp-wide aggregate reduction of input items, exclusive of the \p warp_prefix_op value.
        WarpPrefixOp    &warp_prefix_op)    ///< [in-out] <b>[<em>warp-lane</em><sub>0</sub> only]</b> A call-back unary functor of the model <em>operator()(T warp_aggregate)</em> for specifying the appropriate external prefix (rather than \p identity), to be invoked internally by <em>warp-lane</em><sub>0</sub>.  When provided the warp-wide aggregate of input items (also returned above), this functor is expected to return the logical warp-wide prefix to be applied during the scan operation.  Can be stateful.
    {
        // Exclusive warp scan
        ExclusiveScan(smem_storage, input, output, scan_op, warp_aggregate);

        // Compute warp-wide prefix from aggregate, then broadcast to other lanes
        T prefix;
        unsigned int lane_id = (WARPS == 1) ? threadIdx.x : (threadIdx.x & (LOGICAL_WARP_THREADS - 1));
        if (lane_id == 0)
        {
            prefix = warp_prefix_op(warp_aggregate);
        }
        prefix = WarpScanInternal<POLICY>::Broadcast(smem_storage, prefix, 0);

        // Update output with prefix
        output = (lane_id == 0) ?
            prefix :
            scan_op(prefix, output);
    }

    //@}
};

/** @} */       // end of SimtCoop group

} // namespace cub
CUB_NS_POSTFIX
