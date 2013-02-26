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
 * The cub::BlockScan type provides variants of parallel prefix scan across threads within a threadblock.
 */

#pragma once

#include "../device_props.cuh"
#include "../type_utils.cuh"
#include "../operators.cuh"
#include "../warp/warp_scan.cuh"
#include "../thread/thread_reduce.cuh"
#include "../thread/thread_scan.cuh"
#include "../ns_wrapper.cuh"

CUB_NS_PREFIX

/// CUB namespace
namespace cub {

/// Tuning policy for cub::BlockScan
enum BlockScanPolicy
{

    /**
     * \brief An efficient "raking reduce-then-scan" prefix scan algorithm.
     *
     * Although this variant may suffer longer turnaround latencies when the
     * GPU is under-occupied, it can often provide higher overall throughput
     * across the GPU when suitably occupied.
     *
     * It execution is comprised of five phases:
     *   <br><br>
     *     -# Upsweep sequential reduction in registers (if threads contribute more than one input each).  Each thread then places the partial reduction of its item(s) into shared memory.
     *     -# Upsweep sequential reduction in shared memory.  Threads within a single warp rake across segments of shared partial reductions.
     *     -# A warp-synchronous Kogge-Stone style exclusive scan within the raking warp.
     *     -# Downsweep sequential exclusive scan in shared memory.  Threads within a single warp rake across segments of shared partial reductions, seeded with the warp-scan output.
     *     -# Downsweep sequential scan in registers (if threads contribute more than one input), seeded with the raking scan output.
     *     <br><br>
     *     \image html block_scan_raking.png
     *     <center><b>\p BLOCK_SCAN_RAKING data flow for a hypothetical 16-thread threadblock and 4-thread raking warp.</b></center>
     */
    BLOCK_SCAN_RAKING,


    /**
     * \brief A quick "tiled warpscans" prefix scan algorithm.
     *
     * Although this variant may suffer lower overall throughput across the
     * GPU because due to a heavy reliance on inefficient warpscans, it can
     * often provide lower turnaround latencies when the GPU is under-occupied.
     *
     * It execution is comprised of two phases:
     *   <br><br>
     *     -# Compute a shallow, but inefficient warp-synchronous Kogge-Stone style scan within each warp.
     *     -# A propagation phase where the warp scan outputs in each warp are updated with the aggregate from each preceding warp.
     *     <br><br>
     *     \image html block_scan_warpscans.png
     *     <center><b>\p BLOCK_SCAN_WARPSCANS data flow for a hypothetical 16-thread threadblock and 4-thread raking warp.</b></center>
     */
    BLOCK_SCAN_WARPSCANS,
};

/**
 * \addtogroup SimtCoop
 * @{
 */

/**
 * \brief The BlockScan type provides variants of parallel prefix scan across threads within a threadblock. ![](scan_logo.png)
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
 * \tparam T                The reduction input/output element type
 * \tparam BLOCK_THREADS    The threadblock size in threads
 * \tparam POLICY           <b>[optional]</b> cub::BlockScanPolicy tuning policy enumeration.  Default = cub::BLOCK_SCAN_RAKING.
 *
 * <b>Usage Considerations</b>
 * \par
 * - Supports non-commutative scan operators.
 * - The scan operations assume a [<b><em>blocked arrangement</em></b>](index.html#sec3sec3) of elements across
 *   threads, i.e., <em>n</em>-element lists that are partitioned evenly across
 *   the threadblock, with thread<sub><em>i</em></sub> owning the
 *   <em>i</em><sup>th</sup> element (or <em>i</em><sup>th</sup> segment of
 *   consecutive elements).
 * - After any scan operation, a subsequent threadblock barrier (<tt>__syncthreads()</tt>)
 *   is required if the supplied BlockScan::SmemStorage is to be reused/repurposed
 *   by the threadblock.
 * - Scalar inputs and outputs (e.g., \p block_prefix_op and \p aggregate) are
 *   only valid in <em>thread</em><sub>0</sub>.

 * <b>Performance Considerations</b>
 * \par
 * - Uses special instructions (e.g., warp \p SHFL) when applicable
 * - Uses synchronization-free communication between warp lanes when applicable
 * - Only one or two threadblock-wide synchronization barriers, depending on
 *   algorithm selection.
 * - Zero bank conflicts for most types.
 * - Operations are most efficient (i.e., lowest instruction overhead) when:
 *      - Addition is the reduction operator (viz. prefix sum variants)
 *      - The data type \p T is a built-in primitive or CUDA vector type, e.g.,
 *        \p short, \p int2, \p double, \p float2, etc.  (Otherwise the implementation
 *        may use memory fences to prevent reference reordering of non-primitive types.)
 *      - \p BLOCK_THREADS is a multiple of the architecture's warp size
 *
 * <b>Algorithm</b>
 * \par
 * The BlockScan type can be configured to use one of two alternative algorithms:
 *   -# <b>cub::BLOCK_SCAN_RAKING</b>.  An efficient "raking reduce-then-scan" prefix scan algorithm.
 *   -# <b>cub::BLOCK_SCAN_WARPSCANS</b>.  A quick "tiled warpscans" prefix scan algorithm.
 *
 * <b>Examples</b>
 * \par
 * - <b>Example 1:</b> Simple exclusive prefix sum of 32-bit integer keys (128 threads, 4 keys per thread, blocked arrangement)
 *      \code
 *      #include <cub.cuh>
 *
 *      __global__ void SomeKernel(...)
 *      {
 *          // Parameterize a BlockScan type for use in the current execution context
 *          typedef cub::BlockScan<int, 128> BlockScan;
 *
 *          // Declare shared memory for BlockScan
 *          __shared__ typename BlockScan::SmemStorage smem_storage;
 *
 *          // A segment of consecutive input items per thread
 *          int data[4];
 *
 *          // Obtain items in blocked order
 *          ...
 *
 *          // Compute the threadblock-wide exclusve prefix sum
 *          BlockScan::ExclusiveSum(smem_storage, data, data);
 *
 *      \endcode
 *
 * \par
 * - <b>Example 2:</b> Use of local prefix sum and global atomic-add for performing cooperative allocation within a global data structure
 *      \code
 *      #include <cub.cuh>
 *
 *      /// Simple functor for producing a value for which to seed the entire local scan.
 *      struct BlockPrefixOp
 *      {
 *          int *d_global_counter;
 *
 *          /// Functor constructor
 *          BlockPrefix(int *d_global_counter) : d_global_counter(d_global_counter) {}
 *
 *          /// Functor operator.  Produces a value for seeding the threadblock-wide scan given
 *          /// the local aggregate (called only by thread-0).
 *          int operator(int block_aggregate)
 *          {
 *              return atomicAdd(d_global_counter, block_aggregate);
 *          }
 *      }
 *
 *      template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
 *      __global__ void SomeKernel(int *d_global_counter, ...)
 *      {
 *          // Parameterize a BlockScan type for use in the current execution context
 *          typedef cub::BlockScan<int, BLOCK_THREADS> BlockScan;
 *
 *          // Declare shared memory for BlockScan
 *          __shared__ typename BlockScan::SmemStorage smem_storage;
 *
 *          // A segment of consecutive input items per thread
 *          int data[ITEMS_PER_THREAD];

 *          // Obtain keys in blocked order
 *          ...
 *
 *          // Compute the threadblock-wide exclusive prefix sum, seeded with a threadblock-wide prefix
 *          int aggregate;
 *          BlockScan::ExclusiveSum(smem_storage, data, data, block_aggregate, BlockPrefix(d_global_counter));
 *      \endcode
 */
template <
typename            T,
int                 BLOCK_THREADS,
BlockScanPolicy     POLICY = BLOCK_SCAN_RAKING>
class BlockScan
{
private:

    enum
    {
        SAFE_POLICY =
            ((POLICY == BLOCK_SCAN_WARPSCANS) && (BLOCK_THREADS % DeviceProps::WARP_THREADS != 0)) ?    // BLOCK_SCAN_WARPSCANS policy cannot be used with threadblock sizes not a multiple of the architectural warp size
                BLOCK_SCAN_RAKING :
        POLICY
    };


    /**
     * Specialized BlockScan implementations
     */
    template <int POLICY, int DUMMY = 0>
    struct BlockScanInternal;


    /**
     * Warpscan specialized for BLOCK_SCAN_RAKING variant
     */
    template <int DUMMY>
    struct BlockScanInternal<BLOCK_SCAN_RAKING, DUMMY>
    {
        /// Layout type for padded threadblock raking grid
        typedef BlockRakingGrid<BLOCK_THREADS, T> BlockRakingGrid;

        /// Constants
        enum
        {
            /// Number of active warps
            WARPS = (BLOCK_THREADS + DeviceProps::WARP_THREADS - 1) / DeviceProps::WARP_THREADS,

            /// Number of raking threads
            RAKING_THREADS = BlockRakingGrid::RAKING_THREADS,

            /// Number of raking elements per warp synchronous raking thread
            RAKING_LENGTH = BlockRakingGrid::RAKING_LENGTH,

            /// Cooperative work can be entirely warp synchronous
            WARP_SYNCHRONOUS = (BLOCK_THREADS == RAKING_THREADS),
        };

        ///  Raking warp-scan utility type
        typedef WarpScan<T, 1, RAKING_THREADS> WarpScan;

        /// Shared memory storage layout type
        struct SmemStorage
        {
            typename WarpScan::SmemStorage          warp_scan;      ///< Buffer for warp-synchronous scan
            typename BlockRakingGrid::SmemStorage     raking_grid;    ///< Padded threadblock raking grid
        };

        /// Computes an exclusive threadblock-wide prefix scan using the specified binary \p scan_op functor.  Each thread contributes one input element.  The inclusive threadblock-wide \p aggregate of all inputs is computed for <em>thread</em><sub>0</sub>.
        template <typename ScanOp>
        static __device__ __forceinline__ void ExclusiveScan(
            SmemStorage     &smem_storage,      ///< [in] Shared reference to opaque SmemStorage layout
            T               input,              ///< [in] Calling thread's input items
            T               &output,            ///< [out] Calling thread's output items (may be aliased to \p input)
            const T         &identity,          ///< [in] Identity value
            ScanOp          scan_op,            ///< [in] Binary scan operator having member <tt>T operator()(const T &a, const T &b)</tt>
            T               &block_aggregate)   ///< [out] <b>[<em>thread</em><sub>0</sub> only]</b> threadblock-wide aggregate reduction of input items
        {
            if (WARP_SYNCHRONOUS)
            {
                // Short-circuit directly to warp scan
                WarpScan::ExclusiveScan(
                    smem_storage.warp_scan,
                    input,
                    output,
                    identity,
                    scan_op,
                    block_aggregate);
            }
            else
            {
                // Place thread partial into shared memory raking grid
                T *placement_ptr = BlockRakingGrid::PlacementPtr(smem_storage.raking_grid);
                *placement_ptr = input;

                __syncthreads();

                // Reduce parallelism down to just raking threads
                if (threadIdx.x < RAKING_THREADS)
                {
                    // Raking upsweep reduction in grid
                    T *raking_ptr = BlockRakingGrid::RakingPtr(smem_storage.raking_grid);
                    T raking_partial = raking_ptr[0];

                    #pragma unroll
                    for (int i = 1; i < RAKING_LENGTH; i++)
                    {
                        if ((BlockRakingGrid::UNGUARDED) || (((threadIdx.x * RAKING_LENGTH) + i) < BLOCK_THREADS))
                        {
                            raking_partial = scan_op(raking_partial, raking_ptr[i]);
                        }
                    }

                    // Exclusive warp synchronous scan
                    WarpScan::ExclusiveScan(
                        smem_storage.warp_scan,
                        raking_partial,
                        raking_partial,
                        identity,
                        scan_op,
                        block_aggregate);

                    // Exclusive raking downsweep scan
                    ThreadScanExclusive<RAKING_LENGTH>(raking_ptr, raking_ptr, scan_op, raking_partial);
                }

                __syncthreads();

                // Grab thread prefix from shared memory
                output = *placement_ptr;

            }
        }


        /// Computes an exclusive threadblock-wide prefix scan using the specified binary \p scan_op functor.  Each thread contributes one input element.  Instead of using \p identity as the threadblock-wide prefix, the call-back functor \p block_prefix_op is invoked by <em>thread</em><sub>0</sub> to provide the "seed" value that logically prefixes the threadblock's scan inputs.  The inclusive threadblock-wide \p aggregate of all inputs is computed for <em>thread</em><sub>0</sub>.
        template <
        typename ScanOp,
        typename BlockPrefixOp>
        static __device__ __forceinline__ void ExclusiveScan(
            SmemStorage     &smem_storage,                  ///< [in] Shared reference to opaque SmemStorage layout
            T               input,                          ///< [in] Calling thread's input item
            T               &output,                        ///< [out] Calling thread's output item (may be aliased to \p input)
            T               identity,                       ///< [in] Identity value
            ScanOp          scan_op,                        ///< [in] Binary scan operator having member <tt>T operator()(const T &a, const T &b)</tt>
            T               &block_aggregate,               ///< [out] <b>[<em>thread</em><sub>0</sub> only]</b> threadblock-wide aggregate reduction of input items, exclusive of the \p block_prefix_op value
            BlockPrefixOp   &block_prefix_op)               ///< [in-out] <b>[<em>thread</em><sub>0</sub> only]</b> Call-back functor having member <tt>T operator()(T block_aggregate)</tt> for specifying the appropriate external prefix (rather than \p identity), to be invoked internally by <em>thread</em><sub>0</sub>.  When provided the threadblock-wide aggregate of input items (also returned above), this functor is expected to return the logical threadblock-wide prefix to be applied during the scan operation.  Can be stateful.
        {
            if (WARP_SYNCHRONOUS)
            {
                // Short-circuit directly to warp scan
                WarpScan::ExclusiveScan(
                    smem_storage.warp_scan,
                    input,
                    output,
                    identity,
                    scan_op,
                    block_aggregate,
                    block_prefix_op);
            }
            else
            {
                // Place thread partial into shared memory raking grid
                T *placement_ptr = BlockRakingGrid::PlacementPtr(smem_storage.raking_grid);
                *placement_ptr = input;

                __syncthreads();

                // Reduce parallelism down to just raking threads
                if (threadIdx.x < RAKING_THREADS)
                {
                    // Raking upsweep reduction in grid
                    T *raking_ptr = BlockRakingGrid::RakingPtr(smem_storage.raking_grid);
                    T raking_partial = raking_ptr[0];

                    #pragma unroll
                    for (int i = 1; i < RAKING_LENGTH; i++)
                    {
                        if ((BlockRakingGrid::UNGUARDED) || (((threadIdx.x * RAKING_LENGTH) + i) < BLOCK_THREADS))
                        {
                            raking_partial = scan_op(raking_partial, raking_ptr[i]);
                        }
                    }

                    // Exclusive warp synchronous scan
                    WarpScan::ExclusiveScan(
                        smem_storage.warp_scan,
                        raking_partial,
                        raking_partial,
                        identity,
                        scan_op,
                        block_aggregate,
                        block_prefix_op);

                    // Exclusive raking downsweep scan
                    ThreadScanExclusive<RAKING_LENGTH>(raking_ptr, raking_ptr, scan_op, raking_partial);
                }

                __syncthreads();

                // Grab thread prefix from shared memory
                output = *placement_ptr;
            }
        }


        /// Computes an exclusive threadblock-wide prefix scan using the specified binary \p scan_op functor.  Each thread contributes one input element.  The inclusive threadblock-wide \p aggregate of all inputs is computed for <em>thread</em><sub>0</sub>.  With no identity value, the output computed for <em>thread</em><sub>0</sub> is invalid.
        template <typename ScanOp>
        static __device__ __forceinline__ void ExclusiveScan(
            SmemStorage     &smem_storage,                  ///< [in] Shared reference to opaque SmemStorage layout
            T               input,                          ///< [in] Calling thread's input item
            T               &output,                        ///< [out] Calling thread's output item (may be aliased to \p input)
            ScanOp          scan_op,                        ///< [in] Binary scan operator having member <tt>T operator()(const T &a, const T &b)</tt>
            T               &block_aggregate)               ///< [out] <b>[<em>thread</em><sub>0</sub> only]</b> threadblock-wide aggregate reduction of input items
        {
            if (WARP_SYNCHRONOUS)
            {
                // Short-circuit directly to warp scan
                WarpScan::ExclusiveScan(
                    smem_storage.warp_scan,
                    input,
                    output,
                    scan_op,
                    block_aggregate);
            }
            else
            {
                // Place thread partial into shared memory raking grid
                T *placement_ptr = BlockRakingGrid::PlacementPtr(smem_storage.raking_grid);
                *placement_ptr = input;

                __syncthreads();

                // Reduce parallelism down to just raking threads
                if (threadIdx.x < RAKING_THREADS)
                {
                    // Raking upsweep reduction in grid
                    T *raking_ptr = BlockRakingGrid::RakingPtr(smem_storage.raking_grid);
                    T raking_partial = raking_ptr[0];

                    #pragma unroll
                    for (int i = 1; i < RAKING_LENGTH; i++)
                    {
                        if ((BlockRakingGrid::UNGUARDED) || (((threadIdx.x * RAKING_LENGTH) + i) < BLOCK_THREADS))
                        {
                            raking_partial = scan_op(raking_partial, raking_ptr[i]);
                        }
                    }

                    // Exclusive warp synchronous scan
                    WarpScan::ExclusiveScan(
                        smem_storage.warp_scan,
                        raking_partial,
                        raking_partial,
                        scan_op,
                        block_aggregate);

                    // Exclusive raking downsweep scan
                    ThreadScanExclusive<RAKING_LENGTH>(raking_ptr, raking_ptr, scan_op, raking_partial, (threadIdx.x != 0));
                }

                __syncthreads();

                // Grab thread prefix from shared memory
                output = *placement_ptr;
            }
        }


        /// Computes an exclusive threadblock-wide prefix scan using the specified binary \p scan_op functor.  Each thread contributes one input element.  Instead of using \p identity as the threadblock-wide prefix, the call-back functor \p block_prefix_op is invoked by <em>thread</em><sub>0</sub> to provide the "seed" value that logically prefixes the threadblock's scan inputs.  The inclusive threadblock-wide \p aggregate of all inputs is computed for <em>thread</em><sub>0</sub>.
        template <
        typename ScanOp,
        typename BlockPrefixOp>
        static __device__ __forceinline__ void ExclusiveScan(
            SmemStorage     &smem_storage,                  ///< [in] Shared reference to opaque SmemStorage layout
            T               input,                          ///< [in] Calling thread's input item
            T               &output,                        ///< [out] Calling thread's output item (may be aliased to \p input)
            ScanOp          scan_op,                        ///< [in] Binary scan operator having member <tt>T operator()(const T &a, const T &b)</tt>
            T               &block_aggregate,               ///< [out] <b>[<em>thread</em><sub>0</sub> only]</b> threadblock-wide aggregate reduction of input items, exclusive of the \p block_prefix_op value
            BlockPrefixOp   &block_prefix_op)                   ///< [in-out] <b>[<em>thread</em><sub>0</sub> only]</b> Call-back functor having member <tt>T operator()(T block_aggregate)</tt> for specifying the appropriate external prefix (rather than \p identity), to be invoked internally by <em>thread</em><sub>0</sub>.  When provided the threadblock-wide aggregate of input items (also returned above), this functor is expected to return the logical threadblock-wide prefix to be applied during the scan operation.  Can be stateful.
        {
            if (WARP_SYNCHRONOUS)
            {
                // Short-circuit directly to warp scan
                WarpScan::ExclusiveScan(
                    smem_storage.warp_scan,
                    input,
                    output,
                    scan_op,
                    block_aggregate,
                    block_prefix_op);
            }
            else
            {
                // Place thread partial into shared memory raking grid
                T *placement_ptr = BlockRakingGrid::PlacementPtr(smem_storage.raking_grid);
                *placement_ptr = input;

                __syncthreads();

                // Reduce parallelism down to just raking threads
                if (threadIdx.x < RAKING_THREADS)
                {
                    // Raking upsweep reduction in grid
                    T *raking_ptr = BlockRakingGrid::RakingPtr(smem_storage.raking_grid);
                    T raking_partial = raking_ptr[0];

                    #pragma unroll
                    for (int i = 1; i < RAKING_LENGTH; i++)
                    {
                        if ((BlockRakingGrid::UNGUARDED) || (((threadIdx.x * RAKING_LENGTH) + i) < BLOCK_THREADS))
                        {
                            raking_partial = scan_op(raking_partial, raking_ptr[i]);
                        }
                    }

                    // Exclusive warp synchronous scan
                    WarpScan::ExclusiveScan(
                        smem_storage.warp_scan,
                        raking_partial,
                        raking_partial,
                        scan_op,
                        block_aggregate,
                        block_prefix_op);

                    // Exclusive raking downsweep scan
                    ThreadScanExclusive<RAKING_LENGTH>(raking_ptr, raking_ptr, scan_op, raking_partial);
                }

                __syncthreads();

                // Grab thread prefix from shared memory
                output = *placement_ptr;
            }
        }


        /// Computes an exclusive threadblock-wide prefix scan using addition (+) as the scan operator.  Each thread contributes one input element.  The inclusive threadblock-wide \p aggregate of all inputs is computed for <em>thread</em><sub>0</sub>.
        static __device__ __forceinline__ void ExclusiveSum(
            SmemStorage     &smem_storage,                  ///< [in] Shared reference to opaque SmemStorage layout
            T               input,                          ///< [in] Calling thread's input item
            T               &output,                        ///< [out] Calling thread's output item (may be aliased to \p input)
            T               &block_aggregate)               ///< [out] <b>[<em>thread</em><sub>0</sub> only]</b> threadblock-wide aggregate reduction of input items
        {
            if (WARP_SYNCHRONOUS)
            {
                // Short-circuit directly to warp scan
                WarpScan::ExclusiveSum(
                    smem_storage.warp_scan,
                    input,
                    output,
                    block_aggregate);
            }
            else
            {
                // Raking scan
                Sum<T> scan_op;

                // Place thread partial into shared memory raking grid
                T *placement_ptr = BlockRakingGrid::PlacementPtr(smem_storage.raking_grid);
                *placement_ptr = input;

                __syncthreads();

                // Reduce parallelism down to just raking threads
                if (threadIdx.x < RAKING_THREADS)
                {
                    // Raking upsweep reduction in grid
                    T *raking_ptr = BlockRakingGrid::RakingPtr(smem_storage.raking_grid);
                    T raking_partial = raking_ptr[0];

                    #pragma unroll
                    for (int i = 1; i < RAKING_LENGTH; i++)
                    {
                        if ((BlockRakingGrid::UNGUARDED) || (((threadIdx.x * RAKING_LENGTH) + i) < BLOCK_THREADS))
                        {
                            raking_partial = scan_op(raking_partial, raking_ptr[i]);
                        }
                    }

                    // Exclusive warp synchronous scan
                    WarpScan::ExclusiveSum(
                        smem_storage.warp_scan,
                        raking_partial,
                        raking_partial,
                        block_aggregate);

                    // Exclusive raking downsweep scan
                    ThreadScanExclusive<RAKING_LENGTH>(raking_ptr, raking_ptr, scan_op, raking_partial);
                }

                __syncthreads();

                // Grab thread prefix from shared memory
                output = *placement_ptr;
            }
        }


        /// Computes an exclusive threadblock-wide prefix scan using addition (+) as the scan operator.  Each thread contributes one input element.  Instead of using \p identity as the threadblock-wide prefix, the call-back functor \p block_prefix_op is invoked by <em>thread</em><sub>0</sub> to provide the "seed" value that logically prefixes the threadblock's scan inputs.  The inclusive threadblock-wide \p aggregate of all inputs is computed for <em>thread</em><sub>0</sub>.
        template <typename BlockPrefixOp>
        static __device__ __forceinline__ void ExclusiveSum(
            SmemStorage     &smem_storage,                  ///< [in] Shared reference to opaque SmemStorage layout
            T               input,                          ///< [in] Calling thread's input item
            T               &output,                        ///< [out] Calling thread's output item (may be aliased to \p input)
            T               &block_aggregate,               ///< [out] <b>[<em>thread</em><sub>0</sub> only]</b> threadblock-wide aggregate reduction of input items, exclusive of the \p block_prefix_op value
            BlockPrefixOp   &block_prefix_op)               ///< [in-out] <b>[<em>thread</em><sub>0</sub> only]</b> Call-back functor of the model <em>T block_prefix_op(T block_aggregate)</em> to be run <em>thread</em><sub>0</sub> for providing the operation with a threadblock-wide prefix value to seed the scan with.  Can be stateful.
        {
            if (WARP_SYNCHRONOUS)
            {
                // Short-circuit directly to warp scan
                WarpScan::ExclusiveSum(
                    smem_storage.warp_scan,
                    input,
                    output,
                    block_aggregate,
                    block_prefix_op);
            }
            else
            {
                // Raking scan
                Sum<T> scan_op;

                // Place thread partial into shared memory raking grid
                T *placement_ptr = BlockRakingGrid::PlacementPtr(smem_storage.raking_grid);
                *placement_ptr = input;

                __syncthreads();

                // Reduce parallelism down to just raking threads
                if (threadIdx.x < RAKING_THREADS)
                {
                    // Raking upsweep reduction in grid
                    T *raking_ptr = BlockRakingGrid::RakingPtr(smem_storage.raking_grid);
                    T raking_partial = raking_ptr[0];

                    #pragma unroll
                    for (int i = 1; i < RAKING_LENGTH; i++)
                    {
                        if ((BlockRakingGrid::UNGUARDED) || (((threadIdx.x * RAKING_LENGTH) + i) < BLOCK_THREADS))
                        {
                            raking_partial = scan_op(raking_partial, raking_ptr[i]);
                        }
                    }

                    // Exclusive warp synchronous scan
                    WarpScan::ExclusiveSum(
                        smem_storage.warp_scan,
                        raking_partial,
                        raking_partial,
                        block_aggregate,
                        block_prefix_op);

                    // Exclusive raking downsweep scan
                    ThreadScanExclusive<RAKING_LENGTH>(raking_ptr, raking_ptr, scan_op, raking_partial);
                }

                __syncthreads();

                // Grab thread prefix from shared memory
                output = *placement_ptr;
            }
        }


        /// Computes an inclusive threadblock-wide prefix scan using the specified binary \p scan_op functor.  Each thread contributes one input element.  The inclusive threadblock-wide \p aggregate of all inputs is computed for <em>thread</em><sub>0</sub>.
        template <typename ScanOp>
        static __device__ __forceinline__ void InclusiveScan(
            SmemStorage     &smem_storage,                  ///< [in] Shared reference to opaque SmemStorage layout
            T               input,                          ///< [in] Calling thread's input item
            T               &output,                        ///< [out] Calling thread's output item (may be aliased to \p input)
            ScanOp          scan_op,                        ///< [in] Binary scan operator having member <tt>T operator()(const T &a, const T &b)</tt>
            T               &block_aggregate)               ///< [out] <b>[<em>thread</em><sub>0</sub> only]</b> threadblock-wide aggregate reduction of input items
        {
            if (WARP_SYNCHRONOUS)
            {
                // Short-circuit directly to warp scan
                WarpScan::InclusiveScan(
                    smem_storage.warp_scan,
                    input,
                    output,
                    scan_op,
                    block_aggregate);
            }
            else
            {
                // Place thread partial into shared memory raking grid
                T *placement_ptr = BlockRakingGrid::PlacementPtr(smem_storage.raking_grid);
                *placement_ptr = input;

                __syncthreads();

                // Reduce parallelism down to just raking threads
                if (threadIdx.x < RAKING_THREADS)
                {
                    // Raking upsweep reduction in grid
                    T *raking_ptr = BlockRakingGrid::RakingPtr(smem_storage.raking_grid);
                    T raking_partial = raking_ptr[0];

                    #pragma unroll
                    for (int i = 1; i < RAKING_LENGTH; i++)
                    {
                        if ((BlockRakingGrid::UNGUARDED) || (((threadIdx.x * RAKING_LENGTH) + i) < BLOCK_THREADS))
                        {
                            raking_partial = scan_op(raking_partial, raking_ptr[i]);
                        }
                    }

                    // Exclusive warp synchronous scan
                    WarpScan::ExclusiveScan(
                        smem_storage.warp_scan,
                        raking_partial,
                        raking_partial,
                        scan_op,
                        block_aggregate);

                    // Exclusive raking downsweep scan
                    ThreadScanExclusive<RAKING_LENGTH>(raking_ptr, raking_ptr, scan_op, raking_partial, (threadIdx.x != 0));
                }

                __syncthreads();

                // Grab thread prefix from shared memory
                output = *placement_ptr;
            }
        }


        /// Computes an inclusive threadblock-wide prefix scan using the specified binary \p scan_op functor.  Each thread contributes one input element.  Instead of using \p identity as the threadblock-wide prefix, the call-back functor \p block_prefix_op is invoked by <em>thread</em><sub>0</sub> to provide the "seed" value that logically prefixes the threadblock's scan inputs.  The inclusive threadblock-wide \p aggregate of all inputs is computed for <em>thread</em><sub>0</sub>.
        template <
        typename ScanOp,
        typename BlockPrefixOp>
        static __device__ __forceinline__ void InclusiveScan(
            SmemStorage     &smem_storage,                  ///< [in] Shared reference to opaque SmemStorage layout
            T               input,                          ///< [in] Calling thread's input item
            T               &output,                        ///< [out] Calling thread's output item (may be aliased to \p input)
            ScanOp          scan_op,                        ///< [in] Binary scan operator having member <tt>T operator()(const T &a, const T &b)</tt>
            T               &block_aggregate,               ///< [out] <b>[<em>thread</em><sub>0</sub> only]</b> threadblock-wide aggregate reduction of input items, exclusive of the \p block_prefix_op value
            BlockPrefixOp   &block_prefix_op)               ///< [in-out] <b>[<em>thread</em><sub>0</sub> only]</b> Call-back functor having member <tt>T operator()(T block_aggregate)</tt> for specifying the appropriate external prefix (rather than \p identity), to be invoked internally by <em>thread</em><sub>0</sub>.  When provided the threadblock-wide aggregate of input items (also returned above), this functor is expected to return the logical threadblock-wide prefix to be applied during the scan operation.  Can be stateful.
        {
            if (WARP_SYNCHRONOUS)
            {
                // Short-circuit directly to warp scan
                WarpScan::InclusiveScan(
                    smem_storage.warp_scan,
                    input,
                    output,
                    scan_op,
                    block_aggregate,
                    block_prefix_op);
            }
            else
            {
                // Place thread partial into shared memory raking grid
                T *placement_ptr = BlockRakingGrid::PlacementPtr(smem_storage.raking_grid);
                *placement_ptr = input;

                __syncthreads();

                // Reduce parallelism down to just raking threads
                if (threadIdx.x < RAKING_THREADS)
                {
                    // Raking upsweep reduction in grid
                    T *raking_ptr = BlockRakingGrid::RakingPtr(smem_storage.raking_grid);
                    T raking_partial = raking_ptr[0];

                    #pragma unroll
                    for (int i = 1; i < RAKING_LENGTH; i++)
                    {
                        if ((BlockRakingGrid::UNGUARDED) || (((threadIdx.x * RAKING_LENGTH) + i) < BLOCK_THREADS))
                        {
                            raking_partial = scan_op(raking_partial, raking_ptr[i]);
                        }
                    }

                    // Warp synchronous scan
                    WarpScan::ExclusiveScan(
                        smem_storage.warp_scan,
                        raking_partial,
                        raking_partial,
                        scan_op,
                        block_aggregate,
                        block_prefix_op);

                    // Exclusive raking downsweep scan
                    ThreadScanExclusive<RAKING_LENGTH>(raking_ptr, raking_ptr, scan_op, raking_partial);
                }

                __syncthreads();

                // Grab thread prefix from shared memory
                output = *placement_ptr;
            }
        }


        /// Computes an inclusive threadblock-wide prefix scan using the specified binary \p scan_op functor.  Each thread contributes one input element.  The inclusive threadblock-wide \p aggregate of all inputs is computed for <em>thread</em><sub>0</sub>.
        static __device__ __forceinline__ void InclusiveSum(
            SmemStorage     &smem_storage,                  ///< [in] Shared reference to opaque SmemStorage layout
            T               input,                          ///< [in] Calling thread's input item
            T               &output,                        ///< [out] Calling thread's output item (may be aliased to \p input)
            T               &block_aggregate)               ///< [out] <b>[<em>thread</em><sub>0</sub> only]</b> threadblock-wide aggregate reduction of input items
        {
            if (WARP_SYNCHRONOUS)
            {
                // Short-circuit directly to warp scan
                WarpScan::InclusiveSum(
                    smem_storage.warp_scan,
                    input,
                    output,
                    block_aggregate);
            }
            else
            {
                // Raking scan
                Sum<T> scan_op;

                // Place thread partial into shared memory raking grid
                T *placement_ptr = BlockRakingGrid::PlacementPtr(smem_storage.raking_grid);
                *placement_ptr = input;

                __syncthreads();

                // Reduce parallelism down to just raking threads
                if (threadIdx.x < RAKING_THREADS)
                {
                    // Raking upsweep reduction in grid
                    T *raking_ptr = BlockRakingGrid::RakingPtr(smem_storage.raking_grid);
                    T raking_partial = raking_ptr[0];

                    #pragma unroll
                    for (int i = 1; i < RAKING_LENGTH; i++)
                    {
                        if ((BlockRakingGrid::UNGUARDED) || (((threadIdx.x * RAKING_LENGTH) + i) < BLOCK_THREADS))
                        {
                            raking_partial = scan_op(raking_partial, raking_ptr[i]);
                        }
                    }

                    // Exclusive warp synchronous scan
                    WarpScan::ExclusiveSum(
                        smem_storage.warp_scan,
                        raking_partial,
                        raking_partial,
                        block_aggregate);

                    // Exclusive raking downsweep scan
                    ThreadScanExclusive<RAKING_LENGTH>(raking_ptr, raking_ptr, scan_op, raking_partial, (threadIdx.x != 0));
                }

                __syncthreads();

                // Grab thread prefix from shared memory
                output = *placement_ptr;
            }
        }


        /// Computes an inclusive threadblock-wide prefix scan using the specified binary \p scan_op functor.  Each thread contributes one input element.  Instead of using \p identity as the threadblock-wide prefix, the call-back functor \p block_prefix_op is invoked by <em>thread</em><sub>0</sub> to provide the "seed" value that logically prefixes the threadblock's scan inputs.  The inclusive threadblock-wide \p aggregate of all inputs is computed for <em>thread</em><sub>0</sub>.
        template <typename BlockPrefixOp>
        static __device__ __forceinline__ void InclusiveSum(
            SmemStorage     &smem_storage,                  ///< [in] Shared reference to opaque SmemStorage layout
            T               input,                          ///< [in] Calling thread's input item
            T               &output,                        ///< [out] Calling thread's output item (may be aliased to \p input)
            T               &block_aggregate,               ///< [out] <b>[<em>thread</em><sub>0</sub> only]</b> threadblock-wide aggregate reduction of input items, exclusive of the \p block_prefix_op value
            BlockPrefixOp   &block_prefix_op)                   ///< [in-out] <b>[<em>thread</em><sub>0</sub> only]</b> Call-back functor having member <tt>T operator()(T block_aggregate)</tt> for specifying the appropriate external prefix (rather than \p identity), to be invoked internally by <em>thread</em><sub>0</sub>.  When provided the threadblock-wide aggregate of input items (also returned above), this functor is expected to return the logical threadblock-wide prefix to be applied during the scan operation.  Can be stateful.
        {
            if (WARP_SYNCHRONOUS)
            {
                // Short-circuit directly to warp scan
                WarpScan::InclusiveSum(
                    smem_storage.warp_scan,
                    input,
                    output,
                    block_aggregate,
                    block_prefix_op);
            }
            else
            {
                // Raking scan
                Sum<T> scan_op;

                // Place thread partial into shared memory raking grid
                T *placement_ptr = BlockRakingGrid::PlacementPtr(smem_storage.raking_grid);
                *placement_ptr = input;

                __syncthreads();

                // Reduce parallelism down to just raking threads
                if (threadIdx.x < RAKING_THREADS)
                {
                    // Raking upsweep reduction in grid
                    T *raking_ptr = BlockRakingGrid::RakingPtr(smem_storage.raking_grid);
                    T raking_partial = raking_ptr[0];

                    #pragma unroll
                    for (int i = 1; i < RAKING_LENGTH; i++)
                    {
                        if ((BlockRakingGrid::UNGUARDED) || (((threadIdx.x * RAKING_LENGTH) + i) < BLOCK_THREADS))
                        {
                            raking_partial = scan_op(raking_partial, raking_ptr[i]);
                        }
                    }

                    // Warp synchronous scan
                    WarpScan::ExclusiveSum(
                        smem_storage.warp_scan,
                        raking_partial,
                        raking_partial,
                        block_aggregate,
                        block_prefix_op);

                    // Exclusive raking downsweep scan
                    ThreadScanExclusive<RAKING_LENGTH>(raking_ptr, raking_ptr, scan_op, raking_partial);
                }

                __syncthreads();

                // Grab thread prefix from shared memory
                output = *placement_ptr;
            }
        }

    };



    /**
     * Warpscan specialized for BLOCK_SCAN_WARPSCANS variant
     *
     * Can only be used when BLOCK_THREADS is a multiple of the architecture's warp size
     */
    template <int DUMMY>
    struct BlockScanInternal<BLOCK_SCAN_WARPSCANS, DUMMY>
    {
        /// Constants
        enum
        {
            /// Number of active warps
            WARPS = (BLOCK_THREADS + DeviceProps::WARP_THREADS - 1) / DeviceProps::WARP_THREADS,
        };

        ///  Raking warp-scan utility type
        typedef WarpScan<T, WARPS, DeviceProps::WARP_THREADS> WarpScan;

        /// Shared memory storage layout type
        struct SmemStorage
        {
            typename WarpScan::SmemStorage      warp_scan;                  ///< Buffer for warp-synchronous scan
            T                                   warp_aggregates[WARPS];     ///< Shared totals from each warp-synchronous scan
            T                                   block_prefix;                 ///< Shared prefix for the entire threadblock
        };


        /// Update outputs and block_aggregate with warp-wide aggregates from lane-0s
        template <typename ScanOp>
        static __device__ __forceinline__ void PrefixUpdate(
            SmemStorage     &smem_storage,      ///< [in] Shared reference to opaque SmemStorage layout
            T               &output,            ///< [out] Calling thread's output items
            ScanOp          scan_op,            ///< [in] Binary scan operator having member <tt>T operator()(const T &a, const T &b)</tt>
            T               warp_aggregate,     ///< [in] <b>[<em>lane</em><sub>0</sub>s only]</b> Warp-wide aggregate reduction of input items
            T               &block_aggregate)   ///< [out] <b>[<em>thread</em><sub>0</sub> only]</b> threadblock-wide aggregate reduction of input items
        {
            unsigned int warp_id = threadIdx.x / DeviceProps::WARP_THREADS;
            unsigned int lane_id = threadIdx.x & (DeviceProps::WARP_THREADS - 1);

            // Share lane aggregates
            if (lane_id == 0)
            {
                smem_storage.warp_aggregates[warp_id] = warp_aggregate;
            }

            __syncthreads();

            // Incorporate aggregates from preceding warps into outputs
            #pragma unroll
            for (int PREFIX_WARP = 0; PREFIX_WARP < WARPS - 1; PREFIX_WARP++)
            {
                if (warp_id > PREFIX_WARP)
                {
                    output = scan_op(smem_storage.warp_aggregates[PREFIX_WARP], output);
                }
            }

            // Update total aggregate in warp 0, lane 0
            block_aggregate = warp_aggregate;
            if (threadIdx.x == 0)
            {
                #pragma unroll
                for (int SUCCESSOR_WARP = 1; SUCCESSOR_WARP < WARPS; SUCCESSOR_WARP++)
                {
                    block_aggregate = scan_op(block_aggregate, smem_storage.warp_aggregates[SUCCESSOR_WARP]);
                }
            }
        }


        /// Computes an exclusive threadblock-wide prefix scan using the specified binary \p scan_op functor.  Each thread contributes one input element.  The inclusive threadblock-wide \p aggregate of all inputs is computed for <em>thread</em><sub>0</sub>.
        template <typename ScanOp>
        static __device__ __forceinline__ void ExclusiveScan(
            SmemStorage     &smem_storage,      ///< [in] Shared reference to opaque SmemStorage layout
            T               input,              ///< [in] Calling thread's input items
            T               &output,            ///< [out] Calling thread's output items (may be aliased to \p input)
            const T         &identity,          ///< [in] Identity value
            ScanOp          scan_op,            ///< [in] Binary scan operator having member <tt>T operator()(const T &a, const T &b)</tt>
            T               &block_aggregate)   ///< [out] <b>[<em>thread</em><sub>0</sub> only]</b> threadblock-wide aggregate reduction of input items
        {
            T warp_aggregate;       // Valid in lane-0s
            WarpScan::ExclusiveScan(smem_storage.warp_scan, input, output, identity, scan_op, warp_aggregate);

            // Update outputs and block_aggregate with warp-wide aggregates from lane-0s
            PrefixUpdate(smem_storage, output, scan_op, warp_aggregate, block_aggregate);
        }


        /// Computes an exclusive threadblock-wide prefix scan using the specified binary \p scan_op functor.  Each thread contributes one input element.  Instead of using \p identity as the threadblock-wide prefix, the call-back functor \p block_prefix_op is invoked by <em>thread</em><sub>0</sub> to provide the "seed" value that logically prefixes the threadblock's scan inputs.  The inclusive threadblock-wide \p aggregate of all inputs is computed for <em>thread</em><sub>0</sub>.
        template <
        typename ScanOp,
        typename BlockPrefixOp>
        static __device__ __forceinline__ void ExclusiveScan(
            SmemStorage     &smem_storage,                  ///< [in] Shared reference to opaque SmemStorage layout
            T               input,                          ///< [in] Calling thread's input item
            T               &output,                        ///< [out] Calling thread's output item (may be aliased to \p input)
            T               identity,                       ///< [in] Identity value
            ScanOp          scan_op,                        ///< [in] Binary scan operator having member <tt>T operator()(const T &a, const T &b)</tt>
            T               &block_aggregate,               ///< [out] <b>[<em>thread</em><sub>0</sub> only]</b> threadblock-wide aggregate reduction of input items, exclusive of the \p block_prefix_op value
            BlockPrefixOp   &block_prefix_op)               ///< [in-out] <b>[<em>thread</em><sub>0</sub> only]</b> Call-back functor having member <tt>T operator()(T block_aggregate)</tt> for specifying the appropriate external prefix (rather than \p identity), to be invoked internally by <em>thread</em><sub>0</sub>.  When provided the threadblock-wide aggregate of input items (also returned above), this functor is expected to return the logical threadblock-wide prefix to be applied during the scan operation.  Can be stateful.
        {
            ExclusiveScan(smem_storage, input, output, identity, scan_op, block_aggregate);

            // Compute and share threadblock prefix
            if (threadIdx.x == 0)
            {
                smem_storage.block_prefix = block_prefix_op(block_aggregate);
            }

            __syncthreads();

            // Incorporate threadblock prefix into outputs
            output = scan_op(smem_storage.block_prefix, output);
        }


        /// Computes an exclusive threadblock-wide prefix scan using the specified binary \p scan_op functor.  Each thread contributes one input element.  The inclusive threadblock-wide \p aggregate of all inputs is computed for <em>thread</em><sub>0</sub>.  With no identity value, the output computed for <em>thread</em><sub>0</sub> is invalid.
        template <typename ScanOp>
        static __device__ __forceinline__ void ExclusiveScan(
            SmemStorage     &smem_storage,                  ///< [in] Shared reference to opaque SmemStorage layout
            T               input,                          ///< [in] Calling thread's input item
            T               &output,                        ///< [out] Calling thread's output item (may be aliased to \p input)
            ScanOp          scan_op,                        ///< [in] Binary scan operator having member <tt>T operator()(const T &a, const T &b)</tt>
            T               &block_aggregate)               ///< [out] <b>[<em>thread</em><sub>0</sub> only]</b> threadblock-wide aggregate reduction of input items
        {
            T warp_aggregate;       // Valid in lane-0s
            WarpScan::ExclusiveScan(smem_storage.warp_scan, input, output, scan_op, warp_aggregate);

            unsigned int warp_id = threadIdx.x / DeviceProps::WARP_THREADS;
            unsigned int lane_id = threadIdx.x & (DeviceProps::WARP_THREADS - 1);

            // Share lane aggregates
            if (lane_id == 0)
            {
                smem_storage.warp_aggregates[warp_id] = warp_aggregate;
            }

            __syncthreads();

            // Incorporate aggregates from preceding warps into outputs

            // Other warps grab aggregate from first warp
            if ((WARPS > 1) && (warp_id > 0))
            {
                T addend = smem_storage.warp_aggregates[0];
                output = (lane_id == 0) ?
                    addend :
                    scan_op(addend, output);
            }

            // Continue grabbing from predecessor warps
            #pragma unroll
            for (int PREDECESSOR = 1; PREDECESSOR < WARPS - 1; PREDECESSOR++)
            {
                if (warp_id > PREDECESSOR)
                {
                    T addend = smem_storage.warp_aggregates[PREDECESSOR];
                    output = scan_op(addend, output);
                }
            }

            // Update total aggregate in warp 0, lane 0
            block_aggregate = warp_aggregate;
            if (threadIdx.x == 0)
            {
                #pragma unroll
                for (int SUCESSOR = 1; SUCESSOR < WARPS; SUCESSOR++)
                {
                    block_aggregate = scan_op(block_aggregate, smem_storage.warp_aggregates[SUCESSOR]);
                }
            }
        }


        /// Computes an exclusive threadblock-wide prefix scan using the specified binary \p scan_op functor.  Each thread contributes one input element.  Instead of using \p identity as the threadblock-wide prefix, the call-back functor \p block_prefix_op is invoked by <em>thread</em><sub>0</sub> to provide the "seed" value that logically prefixes the threadblock's scan inputs.  The inclusive threadblock-wide \p aggregate of all inputs is computed for <em>thread</em><sub>0</sub>.
        template <
        typename ScanOp,
        typename BlockPrefixOp>
        static __device__ __forceinline__ void ExclusiveScan(
            SmemStorage     &smem_storage,                  ///< [in] Shared reference to opaque SmemStorage layout
            T               input,                          ///< [in] Calling thread's input item
            T               &output,                        ///< [out] Calling thread's output item (may be aliased to \p input)
            ScanOp          scan_op,                        ///< [in] Binary scan operator having member <tt>T operator()(const T &a, const T &b)</tt>
            T               &block_aggregate,               ///< [out] <b>[<em>thread</em><sub>0</sub> only]</b> threadblock-wide aggregate reduction of input items, exclusive of the \p block_prefix_op value
            BlockPrefixOp   &block_prefix_op)               ///< [in-out] <b>[<em>thread</em><sub>0</sub> only]</b> Call-back functor having member <tt>T operator()(T block_aggregate)</tt> for specifying the appropriate external prefix (rather than \p identity), to be invoked internally by <em>thread</em><sub>0</sub>.  When provided the threadblock-wide aggregate of input items (also returned above), this functor is expected to return the logical threadblock-wide prefix to be applied during the scan operation.  Can be stateful.
        {
            ExclusiveScan(smem_storage, input, output, scan_op, block_aggregate);

            // Compute and share threadblock prefix
            if (threadIdx.x == 0)
            {
                smem_storage.block_prefix = block_prefix_op(block_aggregate);
            }

            __syncthreads();

            // Incorporate threadblock prefix into outputs
            output = (threadIdx.x == 0) ?
                smem_storage.block_prefix :
                scan_op(smem_storage.block_prefix, output);
        }


        /// Computes an exclusive threadblock-wide prefix scan using addition (+) as the scan operator.  Each thread contributes one input element.  The inclusive threadblock-wide \p aggregate of all inputs is computed for <em>thread</em><sub>0</sub>.
        static __device__ __forceinline__ void ExclusiveSum(
            SmemStorage     &smem_storage,                  ///< [in] Shared reference to opaque SmemStorage layout
            T               input,                          ///< [in] Calling thread's input item
            T               &output,                        ///< [out] Calling thread's output item (may be aliased to \p input)
            T               &block_aggregate)               ///< [out] <b>[<em>thread</em><sub>0</sub> only]</b> threadblock-wide aggregate reduction of input items
        {
            T warp_aggregate;       // Valid in lane-0s
            WarpScan::ExclusiveSum(smem_storage.warp_scan, input, output, warp_aggregate);

            // Update outputs and block_aggregate with warp-wide aggregates from lane-0s
            PrefixUpdate(smem_storage, output, Sum<T>(), warp_aggregate, block_aggregate);
        }


        /// Computes an exclusive threadblock-wide prefix scan using addition (+) as the scan operator.  Each thread contributes one input element.  Instead of using \p identity as the threadblock-wide prefix, the call-back functor \p block_prefix_op is invoked by <em>thread</em><sub>0</sub> to provide the "seed" value that logically prefixes the threadblock's scan inputs.  The inclusive threadblock-wide \p aggregate of all inputs is computed for <em>thread</em><sub>0</sub>.
        template <typename BlockPrefixOp>
        static __device__ __forceinline__ void ExclusiveSum(
            SmemStorage     &smem_storage,                  ///< [in] Shared reference to opaque SmemStorage layout
            T               input,                          ///< [in] Calling thread's input item
            T               &output,                        ///< [out] Calling thread's output item (may be aliased to \p input)
            T               &block_aggregate,               ///< [out] <b>[<em>thread</em><sub>0</sub> only]</b> threadblock-wide aggregate reduction of input items, exclusive of the \p block_prefix_op value
            BlockPrefixOp   &block_prefix_op)               ///< [in-out] <b>[<em>thread</em><sub>0</sub> only]</b> Call-back functor of the model <em>T block_prefix_op(T block_aggregate)</em> to be run <em>thread</em><sub>0</sub> for providing the operation with a threadblock-wide prefix value to seed the scan with.  Can be stateful.
        {
            ExclusiveSum(smem_storage, input, output, block_aggregate);

            // Compute and share threadblock prefix
            if (threadIdx.x == 0)
            {
                smem_storage.block_prefix = block_prefix_op(block_aggregate);
            }

            __syncthreads();

            // Incorporate threadblock prefix into outputs
            Sum<T> scan_op;
            output = scan_op(smem_storage.block_prefix, output);
        }


        /// Computes an inclusive threadblock-wide prefix scan using the specified binary \p scan_op functor.  Each thread contributes one input element.  The inclusive threadblock-wide \p aggregate of all inputs is computed for <em>thread</em><sub>0</sub>.
        template <typename ScanOp>
        static __device__ __forceinline__ void InclusiveScan(
            SmemStorage     &smem_storage,                  ///< [in] Shared reference to opaque SmemStorage layout
            T               input,                          ///< [in] Calling thread's input item
            T               &output,                        ///< [out] Calling thread's output item (may be aliased to \p input)
            ScanOp          scan_op,                        ///< [in] Binary scan operator having member <tt>T operator()(const T &a, const T &b)</tt>
            T               &block_aggregate)               ///< [out] <b>[<em>thread</em><sub>0</sub> only]</b> threadblock-wide aggregate reduction of input items
        {
            T warp_aggregate;       // Valid in lane-0s
            WarpScan::InclusiveScan(smem_storage.warp_scan, input, output, scan_op, warp_aggregate);

            // Update outputs and block_aggregate with warp-wide aggregates from lane-0s
            PrefixUpdate(smem_storage, output, scan_op, warp_aggregate, block_aggregate);

        }


        /// Computes an inclusive threadblock-wide prefix scan using the specified binary \p scan_op functor.  Each thread contributes one input element.  Instead of using \p identity as the threadblock-wide prefix, the call-back functor \p block_prefix_op is invoked by <em>thread</em><sub>0</sub> to provide the "seed" value that logically prefixes the threadblock's scan inputs.  The inclusive threadblock-wide \p aggregate of all inputs is computed for <em>thread</em><sub>0</sub>.
        template <
        typename ScanOp,
        typename BlockPrefixOp>
        static __device__ __forceinline__ void InclusiveScan(
            SmemStorage     &smem_storage,                  ///< [in] Shared reference to opaque SmemStorage layout
            T               input,                          ///< [in] Calling thread's input item
            T               &output,                        ///< [out] Calling thread's output item (may be aliased to \p input)
            ScanOp          scan_op,                        ///< [in] Binary scan operator having member <tt>T operator()(const T &a, const T &b)</tt>
            T               &block_aggregate,               ///< [out] <b>[<em>thread</em><sub>0</sub> only]</b> threadblock-wide aggregate reduction of input items, exclusive of the \p block_prefix_op value
            BlockPrefixOp   &block_prefix_op)               ///< [in-out] <b>[<em>thread</em><sub>0</sub> only]</b> Call-back functor having member <tt>T operator()(T block_aggregate)</tt> for specifying the appropriate external prefix (rather than \p identity), to be invoked internally by <em>thread</em><sub>0</sub>.  When provided the threadblock-wide aggregate of input items (also returned above), this functor is expected to return the logical threadblock-wide prefix to be applied during the scan operation.  Can be stateful.
        {
            InclusiveScan(smem_storage, input, output, scan_op, block_aggregate);

            // Compute and share threadblock prefix
            if (threadIdx.x == 0)
            {
                smem_storage.block_prefix = block_prefix_op(block_aggregate);
            }

            __syncthreads();

            // Incorporate threadblock prefix into outputs
            output = scan_op(smem_storage.block_prefix, output);
        }


        /// Computes an inclusive threadblock-wide prefix scan using the specified binary \p scan_op functor.  Each thread contributes one input element.  The inclusive threadblock-wide \p aggregate of all inputs is computed for <em>thread</em><sub>0</sub>.
        static __device__ __forceinline__ void InclusiveSum(
            SmemStorage     &smem_storage,                  ///< [in] Shared reference to opaque SmemStorage layout
            T               input,                          ///< [in] Calling thread's input item
            T               &output,                        ///< [out] Calling thread's output item (may be aliased to \p input)
            T               &block_aggregate)               ///< [out] <b>[<em>thread</em><sub>0</sub> only]</b> threadblock-wide aggregate reduction of input items
        {
            T warp_aggregate;       // Valid in lane-0s
            WarpScan::InclusiveSum(smem_storage.warp_scan, input, output, warp_aggregate);

            // Update outputs and block_aggregate with warp-wide aggregates from lane-0s
            PrefixUpdate(smem_storage, output, Sum<T>(), warp_aggregate, block_aggregate);
        }


        /// Computes an inclusive threadblock-wide prefix scan using the specified binary \p scan_op functor.  Each thread contributes one input element.  Instead of using \p identity as the threadblock-wide prefix, the call-back functor \p block_prefix_op is invoked by <em>thread</em><sub>0</sub> to provide the "seed" value that logically prefixes the threadblock's scan inputs.  The inclusive threadblock-wide \p aggregate of all inputs is computed for <em>thread</em><sub>0</sub>.
        template <typename BlockPrefixOp>
        static __device__ __forceinline__ void InclusiveSum(
            SmemStorage     &smem_storage,                  ///< [in] Shared reference to opaque SmemStorage layout
            T               input,                          ///< [in] Calling thread's input item
            T               &output,                        ///< [out] Calling thread's output item (may be aliased to \p input)
            T               &block_aggregate,               ///< [out] <b>[<em>thread</em><sub>0</sub> only]</b> threadblock-wide aggregate reduction of input items, exclusive of the \p block_prefix_op value
            BlockPrefixOp   &block_prefix_op)               ///< [in-out] <b>[<em>thread</em><sub>0</sub> only]</b> Call-back functor having member <tt>T operator()(T block_aggregate)</tt> for specifying the appropriate external prefix (rather than \p identity), to be invoked internally by <em>thread</em><sub>0</sub>.  When provided the threadblock-wide aggregate of input items (also returned above), this functor is expected to return the logical threadblock-wide prefix to be applied during the scan operation.  Can be stateful.
        {
            InclusiveSum(smem_storage, input, output, block_aggregate);

            // Compute and share threadblock prefix
            if (threadIdx.x == 0)
            {
                smem_storage.block_prefix = block_prefix_op(block_aggregate);
            }

            __syncthreads();

            // Incorporate threadblock prefix into outputs
            Sum<T> scan_op;
            output = scan_op(smem_storage.block_prefix, output);
        }

    };



    /// Shared memory storage layout type for BlockScan
    typedef typename BlockScanInternal<SAFE_POLICY>::SmemStorage _SmemStorage;

public:

    /// The operations exposed by BlockScan require shared memory of this
    /// type.  This opaque storage can be allocated directly using the
    /// <tt>__shared__</tt> keyword.  Alternatively, it can be aliased to
    /// externally allocated shared memory or <tt>union</tt>'d with other types
    /// to facilitate shared memory reuse.
    typedef _SmemStorage SmemStorage;



    /******************************************************************//**
     * \name Exclusive prefix scans
     *********************************************************************/
    //@{


    /**
     * \brief Computes an exclusive threadblock-wide prefix scan using the specified binary \p scan_op functor.  Each thread contributes one input element.  The inclusive threadblock-wide \p aggregate of all inputs is computed for <em>thread</em><sub>0</sub>.
     *
     * The \p aggregate is undefined in threads other than <em>thread</em><sub>0</sub>.
     *
     * \smemreuse
     *
     * \tparam ScanOp   <b>[inferred]</b> Binary scan operator type having member <tt>T operator()(const T &a, const T &b)</tt>
     */
    template <typename ScanOp>
    static __device__ __forceinline__ void ExclusiveScan(
        SmemStorage     &smem_storage,      ///< [in] Shared reference to opaque SmemStorage layout
        T               input,              ///< [in] Calling thread's input items
        T               &output,            ///< [out] Calling thread's output items (may be aliased to \p input)
        const T         &identity,          ///< [in] Identity value
        ScanOp          scan_op,            ///< [in] Binary scan operator having member <tt>T operator()(const T &a, const T &b)</tt>
        T               &block_aggregate)   ///< [out] <b>[<em>thread</em><sub>0</sub> only]</b> threadblock-wide aggregate reduction of input items
    {
        BlockScanInternal<SAFE_POLICY>::ExclusiveScan(smem_storage, input, output, identity, scan_op, block_aggregate);
    }


    /**
     * \brief Computes an exclusive threadblock-wide prefix scan using the specified binary \p scan_op functor.  Each thread contributes an array of consecutive input elements.  The inclusive threadblock-wide \p aggregate of all inputs is computed for <em>thread</em><sub>0</sub>.
     *
     * The \p aggregate is undefined in threads other than <em>thread</em><sub>0</sub>.
     *
     * \smemreuse
     *
     * \tparam ITEMS_PER_THREAD     <b>[inferred]</b> The number of consecutive items partitioned onto each thread.
     * \tparam ScanOp               <b>[inferred]</b> Binary scan operator type having member <tt>T operator()(const T &a, const T &b)</tt>
     */
    template <
    int             ITEMS_PER_THREAD,
    typename        ScanOp>
    static __device__ __forceinline__ void ExclusiveScan(
        SmemStorage       &smem_storage,                ///< [in] Shared reference to opaque SmemStorage layout
        T                 (&input)[ITEMS_PER_THREAD],   ///< [in] Calling thread's input items
        T                 (&output)[ITEMS_PER_THREAD],  ///< [out] Calling thread's output items (may be aliased to \p input)
        const T           &identity,                    ///< [in] Identity value
        ScanOp            scan_op,                      ///< [in] Binary scan operator having member <tt>T operator()(const T &a, const T &b)</tt>
        T                 &block_aggregate)             ///< [out] <b>[<em>thread</em><sub>0</sub> only]</b> threadblock-wide aggregate reduction of input items
    {
        // Reduce consecutive thread items in registers
        T thread_partial = ThreadReduce(input, scan_op);

        // Exclusive threadblock-scan
        ExclusiveScan(smem_storage, thread_partial, thread_partial, identity, scan_op, block_aggregate);

        // Exclusive scan in registers with prefix
        ThreadScanExclusive(input, output, scan_op, thread_partial);
    }


    /**
     * \brief Computes an exclusive threadblock-wide prefix scan using the specified binary \p scan_op functor.  Each thread contributes one input element.  Instead of using \p identity as the threadblock-wide prefix, the call-back functor \p block_prefix_op is invoked by <em>thread</em><sub>0</sub> to provide the "seed" value that logically prefixes the threadblock's scan inputs.  The inclusive threadblock-wide \p aggregate of all inputs is computed for <em>thread</em><sub>0</sub>.
     *
     * The \p aggregate and \p block_prefix_op are undefined in threads other than <em>thread</em><sub>0</sub>.
     *
     * \smemreuse
     *
     * \tparam ScanOp               <b>[inferred]</b> Binary scan operator type having member <tt>T operator()(const T &a, const T &b)</tt>
     * \tparam BlockPrefixOp        <b>[inferred]</b> Call-back functor type having member <tt>T operator()(T block_aggregate)</tt>
     */
    template <
    typename ScanOp,
    typename BlockPrefixOp>
    static __device__ __forceinline__ void ExclusiveScan(
        SmemStorage     &smem_storage,                  ///< [in] Shared reference to opaque SmemStorage layout
        T               input,                          ///< [in] Calling thread's input item
        T               &output,                        ///< [out] Calling thread's output item (may be aliased to \p input)
        T               identity,                       ///< [in] Identity value
        ScanOp          scan_op,                        ///< [in] Binary scan operator having member <tt>T operator()(const T &a, const T &b)</tt>
        T               &block_aggregate,               ///< [out] <b>[<em>thread</em><sub>0</sub> only]</b> threadblock-wide aggregate reduction of input items, exclusive of the \p block_prefix_op value
        BlockPrefixOp   &block_prefix_op)               ///< [in-out] <b>[<em>thread</em><sub>0</sub> only]</b> Call-back functor having member <tt>T operator()(T block_aggregate)</tt> for specifying the appropriate external prefix (rather than \p identity), to be invoked internally by <em>thread</em><sub>0</sub>.  When provided the threadblock-wide aggregate of input items (also returned above), this functor is expected to return the logical threadblock-wide prefix to be applied during the scan operation.  Can be stateful.
    {
        BlockScanInternal<SAFE_POLICY>::ExclusiveScan(smem_storage, input, output, identity, scan_op, block_aggregate, block_prefix_op);
    }


    /**
     * \brief Computes an exclusive threadblock-wide prefix scan using the specified binary \p scan_op functor.  Each thread contributes an array of consecutive input elements.  Instead of using \p identity as the threadblock-wide prefix, the call-back functor \p block_prefix_op is invoked by <em>thread</em><sub>0</sub> to provide the "seed" value that logically prefixes the threadblock's scan inputs.  The inclusive threadblock-wide \p aggregate of all inputs is computed for <em>thread</em><sub>0</sub>.
     *
     * The \p aggregate and \p block_prefix_op are undefined in threads other than <em>thread</em><sub>0</sub>.
     *
     * \smemreuse
     *
     * \tparam ITEMS_PER_THREAD     <b>[inferred]</b> The number of consecutive items partitioned onto each thread.
     * \tparam ScanOp               <b>[inferred]</b> Binary scan operator type having member <tt>T operator()(const T &a, const T &b)</tt>
     * \tparam BlockPrefixOp        <b>[inferred]</b> Call-back functor type having member <tt>T operator()(T block_aggregate)</tt>
     */
    template <
    int             ITEMS_PER_THREAD,
    typename        ScanOp,
    typename        BlockPrefixOp>
    static __device__ __forceinline__ void ExclusiveScan(
        SmemStorage     &smem_storage,                  ///< [in] Shared reference to opaque SmemStorage layout
        T               (&input)[ITEMS_PER_THREAD],     ///< [in] Calling thread's input items
        T               (&output)[ITEMS_PER_THREAD],    ///< [out] Calling thread's output items (may be aliased to \p input)
        T               identity,                       ///< [in] Identity value
        ScanOp          scan_op,                        ///< [in] Binary scan operator having member <tt>T operator()(const T &a, const T &b)</tt>
        T               &block_aggregate,               ///< [out] <b>[<em>thread</em><sub>0</sub> only]</b> threadblock-wide aggregate reduction of input items, exclusive of the \p block_prefix_op value
        BlockPrefixOp   &block_prefix_op)               ///< [in-out] <b>[<em>thread</em><sub>0</sub> only]</b> Call-back functor having member <tt>T operator()(T block_aggregate)</tt> for specifying the appropriate external prefix (rather than \p identity), to be invoked internally by <em>thread</em><sub>0</sub>.  When provided the threadblock-wide aggregate of input items (also returned above), this functor is expected to return the logical threadblock-wide prefix to be applied during the scan operation.  Can be stateful.
    {
        // Reduce consecutive thread items in registers
        T thread_partial = ThreadReduce(input, scan_op);

        // Exclusive threadblock-scan
        ExclusiveScan(smem_storage, thread_partial, thread_partial, identity, scan_op, block_aggregate, block_prefix_op);

        // Exclusive scan in registers with prefix
        ThreadScanExclusive(input, output, scan_op, thread_partial);
    }


    /**
     * \brief Computes an exclusive threadblock-wide prefix scan using the specified binary \p scan_op functor.  Each thread contributes one input element.
     *
     * \smemreuse
     *
     * \tparam ScanOp               <b>[inferred]</b> Binary scan operator type having member <tt>T operator()(const T &a, const T &b)</tt>
     */
    template <typename ScanOp>
    static __device__ __forceinline__ void ExclusiveScan(
        SmemStorage     &smem_storage,                  ///< [in] Shared reference to opaque SmemStorage layout
        T               input,                          ///< [in] Calling thread's input item
        T               &output,                        ///< [out] Calling thread's output item (may be aliased to \p input)
        T               identity,                       ///< [in] Identity value
        ScanOp          scan_op)                        ///< [in] Binary scan operator having member <tt>T operator()(const T &a, const T &b)</tt>
    {
        T block_aggregate;
        BlockScanInternal<SAFE_POLICY>::ExclusiveScan(smem_storage, input, output, identity, scan_op, block_aggregate);
    }



    /**
     * \brief Computes an exclusive threadblock-wide prefix scan using the specified binary \p scan_op functor.  Each thread contributes an array of consecutive input elements.
     *
     * \smemreuse
     *
     * \tparam ITEMS_PER_THREAD     <b>[inferred]</b> The number of consecutive items partitioned onto each thread.
     * \tparam ScanOp               <b>[inferred]</b> Binary scan operator type having member <tt>T operator()(const T &a, const T &b)</tt>
     */
    template <
    int             ITEMS_PER_THREAD,
    typename         ScanOp>
    static __device__ __forceinline__ void ExclusiveScan(
        SmemStorage       &smem_storage,                ///< [in] Shared reference to opaque SmemStorage layout
        T                 (&input)[ITEMS_PER_THREAD],   ///< [in] Calling thread's input items
        T                 (&output)[ITEMS_PER_THREAD],  ///< [out] Calling thread's output items (may be aliased to \p input)
        const T           &identity,                    ///< [in] Identity value
        ScanOp            scan_op)                      ///< [in] Binary scan operator having member <tt>T operator()(const T &a, const T &b)</tt>
    {
        // Reduce consecutive thread items in registers
        T thread_partial = ThreadReduce(input, scan_op);

        // Exclusive threadblock-scan
        ExclusiveScan(smem_storage, thread_partial, thread_partial, identity, scan_op);

        // Exclusive scan in registers with prefix
        ThreadScanExclusive(input, output, scan_op, thread_partial);
    }


    //@}
    /******************************************************************//**
     * \name Exclusive prefix scans (without supplied identity)
     *********************************************************************/
    //@{


    /**
     * \brief Computes an exclusive threadblock-wide prefix scan using the specified binary \p scan_op functor.  Each thread contributes one input element.  The inclusive threadblock-wide \p aggregate of all inputs is computed for <em>thread</em><sub>0</sub>.  With no identity value, the output computed for <em>thread</em><sub>0</sub> is invalid.
     *
     * The \p aggregate is undefined in threads other than <em>thread</em><sub>0</sub>.
     *
     * \smemreuse
     *
     * \tparam ScanOp   <b>[inferred]</b> Binary scan operator type having member <tt>T operator()(const T &a, const T &b)</tt>
     */
    template <typename ScanOp>
    static __device__ __forceinline__ void ExclusiveScan(
        SmemStorage     &smem_storage,                  ///< [in] Shared reference to opaque SmemStorage layout
        T               input,                          ///< [in] Calling thread's input item
        T               &output,                        ///< [out] Calling thread's output item (may be aliased to \p input)
        ScanOp          scan_op,                        ///< [in] Binary scan operator having member <tt>T operator()(const T &a, const T &b)</tt>
        T               &block_aggregate)               ///< [out] <b>[<em>thread</em><sub>0</sub> only]</b> threadblock-wide aggregate reduction of input items
    {
        BlockScanInternal<SAFE_POLICY>::ExclusiveScan(smem_storage, input, output, scan_op, block_aggregate);
    }


    /**
     * \brief Computes an exclusive threadblock-wide prefix scan using the specified binary \p scan_op functor.  Each thread contributes an array of consecutive input elements.  The inclusive threadblock-wide \p aggregate of all inputs is computed for <em>thread</em><sub>0</sub>.  With no identity value, the output computed for <em>thread</em><sub>0</sub> is invalid.
     *
     * The \p aggregate is undefined in threads other than <em>thread</em><sub>0</sub>.
     *
     * \smemreuse
     *
     * \tparam ITEMS_PER_THREAD     <b>[inferred]</b> The number of consecutive items partitioned onto each thread.
     * \tparam ScanOp               <b>[inferred]</b> Binary scan operator type having member <tt>T operator()(const T &a, const T &b)</tt>
     */
    template <
    int             ITEMS_PER_THREAD,
    typename         ScanOp>
    static __device__ __forceinline__ void ExclusiveScan(
        SmemStorage     &smem_storage,                  ///< [in] Shared reference to opaque SmemStorage layout
        T               (&input)[ITEMS_PER_THREAD],     ///< [in] Calling thread's input items
        T               (&output)[ITEMS_PER_THREAD],    ///< [out] Calling thread's output items (may be aliased to \p input)
        ScanOp          scan_op,                        ///< [in] Binary scan operator having member <tt>T operator()(const T &a, const T &b)</tt>
        T               &block_aggregate)               ///< [out] <b>[<em>thread</em><sub>0</sub> only]</b> threadblock-wide aggregate reduction of input items
    {
        // Reduce consecutive thread items in registers
        T thread_partial = ThreadReduce(input, scan_op);

        // Exclusive threadblock-scan
        ExclusiveScan(smem_storage, thread_partial, thread_partial, scan_op, block_aggregate);

        // Exclusive scan in registers with prefix
        ThreadScanExclusive(input, output, scan_op, thread_partial, (threadIdx.x != 0));
    }


    /**
     * \brief Computes an exclusive threadblock-wide prefix scan using the specified binary \p scan_op functor.  Each thread contributes one input element.  Instead of using \p identity as the threadblock-wide prefix, the call-back functor \p block_prefix_op is invoked by <em>thread</em><sub>0</sub> to provide the "seed" value that logically prefixes the threadblock's scan inputs.  The inclusive threadblock-wide \p aggregate of all inputs is computed for <em>thread</em><sub>0</sub>.
     *
     * The \p aggregate and \p block_prefix_op are undefined in threads other than <em>thread</em><sub>0</sub>.
     *
     * \smemreuse
     *
     * \tparam ScanOp               <b>[inferred]</b> Binary scan operator type having member <tt>T operator()(const T &a, const T &b)</tt>
     * \tparam BlockPrefixOp        <b>[inferred]</b> Call-back functor type having member <tt>T operator()(T block_aggregate)</tt>
     */
    template <
    typename ScanOp,
    typename BlockPrefixOp>
    static __device__ __forceinline__ void ExclusiveScan(
        SmemStorage     &smem_storage,                  ///< [in] Shared reference to opaque SmemStorage layout
        T               input,                          ///< [in] Calling thread's input item
        T               &output,                        ///< [out] Calling thread's output item (may be aliased to \p input)
        ScanOp          scan_op,                        ///< [in] Binary scan operator having member <tt>T operator()(const T &a, const T &b)</tt>
        T               &block_aggregate,               ///< [out] <b>[<em>thread</em><sub>0</sub> only]</b> threadblock-wide aggregate reduction of input items, exclusive of the \p block_prefix_op value
        BlockPrefixOp   &block_prefix_op)               ///< [in-out] <b>[<em>thread</em><sub>0</sub> only]</b> Call-back functor having member <tt>T operator()(T block_aggregate)</tt> for specifying the appropriate external prefix (rather than \p identity), to be invoked internally by <em>thread</em><sub>0</sub>.  When provided the threadblock-wide aggregate of input items (also returned above), this functor is expected to return the logical threadblock-wide prefix to be applied during the scan operation.  Can be stateful.
    {
        BlockScanInternal<SAFE_POLICY>::ExclusiveScan(smem_storage, input, output, scan_op, block_aggregate, block_prefix_op);
    }


    /**
     * \brief Computes an exclusive threadblock-wide prefix scan using the specified binary \p scan_op functor.  Each thread contributes an array of consecutive input elements.  Instead of using \p identity as the threadblock-wide prefix, the call-back functor \p block_prefix_op is invoked by <em>thread</em><sub>0</sub> to provide the "seed" value that logically prefixes the threadblock's scan inputs.  The inclusive threadblock-wide \p aggregate of all inputs is computed for <em>thread</em><sub>0</sub>.
     *
     * The \p aggregate and \p block_prefix_op are undefined in threads other than <em>thread</em><sub>0</sub>.
     *
     * \smemreuse
     *
     * \tparam ITEMS_PER_THREAD     <b>[inferred]</b> The number of consecutive items partitioned onto each thread.
     * \tparam ScanOp               <b>[inferred]</b> Binary scan operator type having member <tt>T operator()(const T &a, const T &b)</tt>
     * \tparam BlockPrefixOp        <b>[inferred]</b> Call-back functor type having member <tt>T operator()(T block_aggregate)</tt>
     */
    template <
    int             ITEMS_PER_THREAD,
    typename        ScanOp,
    typename        BlockPrefixOp>
    static __device__ __forceinline__ void ExclusiveScan(
        SmemStorage      &smem_storage,               ///< [in] Shared reference to opaque SmemStorage layout
        T               (&input)[ITEMS_PER_THREAD],   ///< [in] Calling thread's input items
        T               (&output)[ITEMS_PER_THREAD],  ///< [out] Calling thread's output items (may be aliased to \p input)
        ScanOp          scan_op,                      ///< [in] Binary scan operator having member <tt>T operator()(const T &a, const T &b)</tt>
        T               &block_aggregate,             ///< [out] <b>[<em>thread</em><sub>0</sub> only]</b> threadblock-wide aggregate reduction of input items, exclusive of the \p block_prefix_op value
        BlockPrefixOp   &block_prefix_op)             ///< [in-out] <b>[<em>thread</em><sub>0</sub> only]</b> Call-back functor having member <tt>T operator()(T block_aggregate)</tt> for specifying the appropriate external prefix (rather than \p identity), to be invoked internally by <em>thread</em><sub>0</sub>.  When provided the threadblock-wide aggregate of input items (also returned above), this functor is expected to return the logical threadblock-wide prefix to be applied during the scan operation.  Can be stateful.
    {
        // Reduce consecutive thread items in registers
        T thread_partial = ThreadReduce(input, scan_op);

        // Exclusive threadblock-scan
        ExclusiveScan(smem_storage, thread_partial, thread_partial, scan_op, block_aggregate, block_prefix_op);

        // Exclusive scan in registers with prefix
        ThreadScanExclusive(input, output, scan_op, thread_partial);
    }


    /**
     * \brief Computes an exclusive threadblock-wide prefix scan using the specified binary \p scan_op functor.  Each thread contributes one input element.  With no identity value, the output computed for <em>thread</em><sub>0</sub> is invalid.
     *
     * \smemreuse
     *
     * \tparam ScanOp               <b>[inferred]</b> Binary scan operator type having member <tt>T operator()(const T &a, const T &b)</tt>
     */
    template <typename ScanOp>
    static __device__ __forceinline__ void ExclusiveScan(
        SmemStorage     &smem_storage,                  ///< [in] Shared reference to opaque SmemStorage layout
        T               input,                          ///< [in] Calling thread's input item
        T               &output,                        ///< [out] Calling thread's output item (may be aliased to \p input)
        ScanOp          scan_op)                        ///< [in] Binary scan operator having member <tt>T operator()(const T &a, const T &b)</tt>
    {
        T block_aggregate;
        BlockScanInternal<SAFE_POLICY>::ExclusiveScan(smem_storage, input, output, scan_op, block_aggregate);
    }


    /**
     * \brief Computes an exclusive threadblock-wide prefix scan using the specified binary \p scan_op functor.  Each thread contributes an array of consecutive input elements.  With no identity value, the output computed for <em>thread</em><sub>0</sub> is invalid.
     *
     * \smemreuse
     *
     * \tparam ITEMS_PER_THREAD     <b>[inferred]</b> The number of consecutive items partitioned onto each thread.
     * \tparam ScanOp               <b>[inferred]</b> Binary scan operator type having member <tt>T operator()(const T &a, const T &b)</tt>
     */
    template <
    int             ITEMS_PER_THREAD,
    typename         ScanOp>
    static __device__ __forceinline__ void ExclusiveScan(
        SmemStorage        &smem_storage,               ///< [in] Shared reference to opaque SmemStorage layout
        T                 (&input)[ITEMS_PER_THREAD],   ///< [in] Calling thread's input items
        T                 (&output)[ITEMS_PER_THREAD],  ///< [out] Calling thread's output items (may be aliased to \p input)
        ScanOp            scan_op)                      ///< [in] Binary scan operator having member <tt>T operator()(const T &a, const T &b)</tt>
    {
        // Reduce consecutive thread items in registers
        T thread_partial = ThreadReduce(input, scan_op);

        // Exclusive threadblock-scan
        ExclusiveScan(smem_storage, thread_partial, thread_partial, scan_op);

        // Exclusive scan in registers with prefix
        ThreadScanExclusive(input, output, scan_op, thread_partial, (threadIdx.x != 0));
    }


    //@}
    /******************************************************************//**
     * \name Exclusive prefix sums
     *********************************************************************/
    //@{


    /**
     * \brief Computes an exclusive threadblock-wide prefix scan using addition (+) as the scan operator.  Each thread contributes one input element.  The inclusive threadblock-wide \p aggregate of all inputs is computed for <em>thread</em><sub>0</sub>.
     *
     * The \p aggregate is undefined in threads other than <em>thread</em><sub>0</sub>.
     *
     * \smemreuse
     */
    static __device__ __forceinline__ void ExclusiveSum(
        SmemStorage     &smem_storage,                  ///< [in] Shared reference to opaque SmemStorage layout
        T               input,                          ///< [in] Calling thread's input item
        T               &output,                        ///< [out] Calling thread's output item (may be aliased to \p input)
        T               &block_aggregate)               ///< [out] <b>[<em>thread</em><sub>0</sub> only]</b> threadblock-wide aggregate reduction of input items
    {
        BlockScanInternal<SAFE_POLICY>::ExclusiveSum(smem_storage, input, output, block_aggregate);
    }


    /**
     * \brief Computes an exclusive threadblock-wide prefix scan using addition (+) as the scan operator.  Each thread contributes an array of consecutive input elements.  The inclusive threadblock-wide \p aggregate of all inputs is computed for <em>thread</em><sub>0</sub>.
     *
     * The \p aggregate is undefined in threads other than <em>thread</em><sub>0</sub>.
     *
     * \smemreuse
     *
     * \tparam ITEMS_PER_THREAD     <b>[inferred]</b> The number of consecutive items partitioned onto each thread.
     */
    template <int ITEMS_PER_THREAD>
    static __device__ __forceinline__ void ExclusiveSum(
        SmemStorage        &smem_storage,                   ///< [in] Shared reference to opaque SmemStorage layout
        T                 (&input)[ITEMS_PER_THREAD],       ///< [in] Calling thread's input items
        T                 (&output)[ITEMS_PER_THREAD],      ///< [out] Calling thread's output items (may be aliased to \p input)
        T                 &block_aggregate)                 ///< [out] <b>[<em>thread</em><sub>0</sub> only]</b> threadblock-wide aggregate reduction of input items
    {
        // Reduce consecutive thread items in registers
        Sum<T> scan_op;
        T thread_partial = ThreadReduce(input, scan_op);

        // Exclusive threadblock-scan
        ExclusiveSum(smem_storage, thread_partial, thread_partial, block_aggregate);

        // Exclusive scan in registers with prefix
        ThreadScanExclusive(input, output, scan_op, thread_partial);
    }


    /**
     * \brief Computes an exclusive threadblock-wide prefix scan using addition (+) as the scan operator.  Each thread contributes one input element.  Instead of using \p identity as the threadblock-wide prefix, the call-back functor \p block_prefix_op is invoked by <em>thread</em><sub>0</sub> to provide the "seed" value that logically prefixes the threadblock's scan inputs.  The inclusive threadblock-wide \p aggregate of all inputs is computed for <em>thread</em><sub>0</sub>.
     *
     * The \p aggregate and \p block_prefix_op are undefined in threads other than <em>thread</em><sub>0</sub>.
     *
     * \smemreuse
     *
     * \tparam BlockPrefixOp        <b>[inferred]</b> Call-back functor type having member <tt>T operator()(T block_aggregate)</tt>
     */
    template <typename BlockPrefixOp>
    static __device__ __forceinline__ void ExclusiveSum(
        SmemStorage     &smem_storage,                  ///< [in] Shared reference to opaque SmemStorage layout
        T               input,                          ///< [in] Calling thread's input item
        T               &output,                        ///< [out] Calling thread's output item (may be aliased to \p input)
        T               &block_aggregate,               ///< [out] <b>[<em>thread</em><sub>0</sub> only]</b> threadblock-wide aggregate reduction of input items, exclusive of the \p block_prefix_op value
        BlockPrefixOp   &block_prefix_op)               ///< [in-out] <b>[<em>thread</em><sub>0</sub> only]</b> Call-back functor of the model <em>T block_prefix_op(T block_aggregate)</em> to be run <em>thread</em><sub>0</sub> for providing the operation with a threadblock-wide prefix value to seed the scan with.  Can be stateful.
    {
        BlockScanInternal<SAFE_POLICY>::ExclusiveSum(smem_storage, input, output, block_aggregate, block_prefix_op);
    }


    /**
     * \brief Computes an exclusive threadblock-wide prefix scan using addition (+) as the scan operator.  Each thread contributes an array of consecutive input elements.  Instead of using \p identity as the threadblock-wide prefix, the call-back functor \p block_prefix_op is invoked by <em>thread</em><sub>0</sub> to provide the "seed" value that logically prefixes the threadblock's scan inputs.  The inclusive threadblock-wide \p aggregate of all inputs is computed for <em>thread</em><sub>0</sub>.
     *
     * The \p aggregate and \p block_prefix_op are undefined in threads other than <em>thread</em><sub>0</sub>.
     *
     * \smemreuse
     *
     * \tparam ITEMS_PER_THREAD     <b>[inferred]</b> The number of consecutive items partitioned onto each thread.
     * \tparam BlockPrefixOp        <b>[inferred]</b> Call-back functor type having member <tt>T operator()(T block_aggregate)</tt>
     */
    template <
    int ITEMS_PER_THREAD,
    typename BlockPrefixOp>
    static __device__ __forceinline__ void ExclusiveSum(
        SmemStorage       &smem_storage,                ///< [in] Shared reference to opaque SmemStorage layout
        T                 (&input)[ITEMS_PER_THREAD],   ///< [in] Calling thread's input items
        T                 (&output)[ITEMS_PER_THREAD],  ///< [out] Calling thread's output items (may be aliased to \p input)
        T                 &block_aggregate,             ///< [out] <b>[<em>thread</em><sub>0</sub> only]</b> threadblock-wide aggregate reduction of input items, exclusive of the \p block_prefix_op value
        BlockPrefixOp       &block_prefix_op)           ///< [in-out] <b>[<em>thread</em><sub>0</sub> only]</b> Call-back functor having member <tt>T operator()(T block_aggregate)</tt> for specifying the appropriate external prefix (rather than \p identity), to be invoked internally by <em>thread</em><sub>0</sub>.  When provided the threadblock-wide aggregate of input items (also returned above), this functor is expected to return the logical threadblock-wide prefix to be applied during the scan operation.  Can be stateful.
    {
        // Reduce consecutive thread items in registers
        Sum<T> scan_op;
        T thread_partial = ThreadReduce(input, scan_op);

        // Exclusive threadblock-scan
        ExclusiveSum(smem_storage, thread_partial, thread_partial, block_aggregate, block_prefix_op);

        // Exclusive scan in registers with prefix
        ThreadScanExclusive(input, output, scan_op, thread_partial);
    }


    /**
     * \brief Computes an exclusive threadblock-wide prefix scan using addition (+) as the scan operator.  Each thread contributes one input element.
     *
     * \smemreuse
     */
    static __device__ __forceinline__ void ExclusiveSum(
        SmemStorage     &smem_storage,                  ///< [in] Shared reference to opaque SmemStorage layout
        T               input,                          ///< [in] Calling thread's input item
        T               &output)                        ///< [out] Calling thread's output item (may be aliased to \p input)
    {
        T block_aggregate;
        BlockScanInternal<SAFE_POLICY>::ExclusiveSum(smem_storage, input, output, block_aggregate);
    }


    /**
     * \brief Computes an exclusive threadblock-wide prefix scan using addition (+) as the scan operator.  Each thread contributes an array of consecutive input elements.
     *
     * \smemreuse
     *
     * \tparam ITEMS_PER_THREAD     <b>[inferred]</b> The number of consecutive items partitioned onto each thread.
     */
    template <int ITEMS_PER_THREAD>
    static __device__ __forceinline__ void ExclusiveSum(
        SmemStorage        &smem_storage,               ///< [in] Shared reference to opaque SmemStorage layout
        T                 (&input)[ITEMS_PER_THREAD],   ///< [in] Calling thread's input items
        T                 (&output)[ITEMS_PER_THREAD])  ///< [out] Calling thread's output items (may be aliased to \p input)
    {
        // Reduce consecutive thread items in registers
        Sum<T> scan_op;
        T thread_partial = ThreadReduce(input, scan_op);

        // Exclusive threadblock-scan
        ExclusiveSum(smem_storage, thread_partial, thread_partial);

        // Exclusive scan in registers with prefix
        ThreadScanExclusive(input, output, scan_op, thread_partial);
    }


    //@}
    /******************************************************************//**
     * \name Inclusive prefix scans
     *********************************************************************/
    //@{


    /**
     * \brief Computes an inclusive threadblock-wide prefix scan using the specified binary \p scan_op functor.  Each thread contributes one input element.  The inclusive threadblock-wide \p aggregate of all inputs is computed for <em>thread</em><sub>0</sub>.
     *
     * The \p aggregate is undefined in threads other than <em>thread</em><sub>0</sub>.
     *
     * \smemreuse
     *
     * \tparam ScanOp   <b>[inferred]</b> Binary scan operator type having member <tt>T operator()(const T &a, const T &b)</tt>
     */
    template <typename ScanOp>
    static __device__ __forceinline__ void InclusiveScan(
        SmemStorage     &smem_storage,                  ///< [in] Shared reference to opaque SmemStorage layout
        T               input,                          ///< [in] Calling thread's input item
        T               &output,                        ///< [out] Calling thread's output item (may be aliased to \p input)
        ScanOp          scan_op,                        ///< [in] Binary scan operator having member <tt>T operator()(const T &a, const T &b)</tt>
        T               &block_aggregate)               ///< [out] <b>[<em>thread</em><sub>0</sub> only]</b> threadblock-wide aggregate reduction of input items
    {
        BlockScanInternal<SAFE_POLICY>::InclusiveScan(smem_storage, input, output, scan_op, block_aggregate);
    }


    /**
     * \brief Computes an inclusive threadblock-wide prefix scan using the specified binary \p scan_op functor.  Each thread contributes an array of consecutive input elements.  The inclusive threadblock-wide \p aggregate of all inputs is computed for <em>thread</em><sub>0</sub>.
     *
     * The \p aggregate is undefined in threads other than <em>thread</em><sub>0</sub>.
     *
     * \smemreuse
     *
     * \tparam ITEMS_PER_THREAD     <b>[inferred]</b> The number of consecutive items partitioned onto each thread.
     * \tparam ScanOp               <b>[inferred]</b> Binary scan operator type having member <tt>T operator()(const T &a, const T &b)</tt>
     */
    template <
    int             ITEMS_PER_THREAD,
    typename         ScanOp>
    static __device__ __forceinline__ void InclusiveScan(
        SmemStorage     &smem_storage,                  ///< [in] Shared reference to opaque SmemStorage layout
        T               (&input)[ITEMS_PER_THREAD],     ///< [in] Calling thread's input items
        T               (&output)[ITEMS_PER_THREAD],    ///< [out] Calling thread's output items (may be aliased to \p input)
        ScanOp          scan_op,                        ///< [in] Binary scan operator having member <tt>T operator()(const T &a, const T &b)</tt>
        T               &block_aggregate)               ///< [out] <b>[<em>thread</em><sub>0</sub> only]</b> threadblock-wide aggregate reduction of input items
    {
        // Reduce consecutive thread items in registers
        T thread_partial = ThreadReduce(input, scan_op);

        // Exclusive threadblock-scan
        ExclusiveScan(smem_storage, thread_partial, thread_partial, scan_op, block_aggregate);

        // Inclusive scan in registers with prefix
        ThreadScanInclusive(input, output, scan_op, thread_partial, (threadIdx.x != 0));
    }


    /**
     * \brief Computes an inclusive threadblock-wide prefix scan using the specified binary \p scan_op functor.  Each thread contributes one input element.  Instead of using \p identity as the threadblock-wide prefix, the call-back functor \p block_prefix_op is invoked by <em>thread</em><sub>0</sub> to provide the "seed" value that logically prefixes the threadblock's scan inputs.  The inclusive threadblock-wide \p aggregate of all inputs is computed for <em>thread</em><sub>0</sub>.
     *
     * The \p aggregate and \p block_prefix_op are undefined in threads other than <em>thread</em><sub>0</sub>.
     *
     * \smemreuse
     *
     * \tparam ScanOp               <b>[inferred]</b> Binary scan operator type having member <tt>T operator()(const T &a, const T &b)</tt>
     * \tparam BlockPrefixOp        <b>[inferred]</b> Call-back functor type having member <tt>T operator()(T block_aggregate)</tt>
     */
    template <
    typename ScanOp,
    typename BlockPrefixOp>
    static __device__ __forceinline__ void InclusiveScan(
        SmemStorage     &smem_storage,                  ///< [in] Shared reference to opaque SmemStorage layout
        T               input,                          ///< [in] Calling thread's input item
        T               &output,                        ///< [out] Calling thread's output item (may be aliased to \p input)
        ScanOp          scan_op,                        ///< [in] Binary scan operator having member <tt>T operator()(const T &a, const T &b)</tt>
        T               &block_aggregate,               ///< [out] <b>[<em>thread</em><sub>0</sub> only]</b> threadblock-wide aggregate reduction of input items, exclusive of the \p block_prefix_op value
        BlockPrefixOp   &block_prefix_op)               ///< [in-out] <b>[<em>thread</em><sub>0</sub> only]</b> Call-back functor having member <tt>T operator()(T block_aggregate)</tt> for specifying the appropriate external prefix (rather than \p identity), to be invoked internally by <em>thread</em><sub>0</sub>.  When provided the threadblock-wide aggregate of input items (also returned above), this functor is expected to return the logical threadblock-wide prefix to be applied during the scan operation.  Can be stateful.
    {
        BlockScanInternal<SAFE_POLICY>::InclusiveScan(smem_storage, input, output, scan_op, block_aggregate, block_prefix_op);
    }


    /**
     * \brief Computes an inclusive threadblock-wide prefix scan using the specified binary \p scan_op functor.  Each thread contributes an array of consecutive input elements.  Instead of using \p identity as the threadblock-wide prefix, the call-back functor \p block_prefix_op is invoked by <em>thread</em><sub>0</sub> to provide the "seed" value that logically prefixes the threadblock's scan inputs.  The inclusive threadblock-wide \p aggregate of all inputs is computed for <em>thread</em><sub>0</sub>.
     *
     * The \p aggregate and \p block_prefix_op are undefined in threads other than <em>thread</em><sub>0</sub>.
     *
     * \smemreuse
     *
     * \tparam ITEMS_PER_THREAD     <b>[inferred]</b> The number of consecutive items partitioned onto each thread.
     * \tparam ScanOp               <b>[inferred]</b> Binary scan operator type having member <tt>T operator()(const T &a, const T &b)</tt>
     * \tparam BlockPrefixOp        <b>[inferred]</b> Call-back functor type having member <tt>T operator()(T block_aggregate)</tt>
     */
    template <
    int             ITEMS_PER_THREAD,
    typename        ScanOp,
    typename        BlockPrefixOp>
    static __device__ __forceinline__ void InclusiveScan(
        SmemStorage     &smem_storage,                  ///< [in] Shared reference to opaque SmemStorage layout
        T               (&input)[ITEMS_PER_THREAD],     ///< [in] Calling thread's input items
        T               (&output)[ITEMS_PER_THREAD],    ///< [out] Calling thread's output items (may be aliased to \p input)
        ScanOp          scan_op,                        ///< [in] Binary scan operator having member <tt>T operator()(const T &a, const T &b)</tt>
        T               &block_aggregate,               ///< [out] <b>[<em>thread</em><sub>0</sub> only]</b> threadblock-wide aggregate reduction of input items, exclusive of the \p block_prefix_op value
        BlockPrefixOp   &block_prefix_op)               ///< [in-out] <b>[<em>thread</em><sub>0</sub> only]</b> Call-back functor having member <tt>T operator()(T block_aggregate)</tt> for specifying the appropriate external prefix (rather than \p identity), to be invoked internally by <em>thread</em><sub>0</sub>.  When provided the threadblock-wide aggregate of input items (also returned above), this functor is expected to return the logical threadblock-wide prefix to be applied during the scan operation.  Can be stateful.
    {
        // Reduce consecutive thread items in registers
        T thread_partial = ThreadReduce(input, scan_op);

        // Exclusive threadblock-scan
        ExclusiveScan(smem_storage, thread_partial, thread_partial, scan_op, block_aggregate, block_prefix_op);

        // Inclusive scan in registers with prefix
        ThreadScanInclusive(input, output, scan_op, thread_partial);
    }


    /**
     * \brief Computes an inclusive threadblock-wide prefix scan using the specified binary \p scan_op functor.  Each thread contributes one input element.
     *
     * \smemreuse
     *
     * \tparam ScanOp               <b>[inferred]</b> Binary scan operator type having member <tt>T operator()(const T &a, const T &b)</tt>
     */
    template <typename ScanOp>
    static __device__ __forceinline__ void InclusiveScan(
        SmemStorage     &smem_storage,                  ///< [in] Shared reference to opaque SmemStorage layout
        T               input,                          ///< [in] Calling thread's input item
        T               &output,                        ///< [out] Calling thread's output item (may be aliased to \p input)
        ScanOp          scan_op)                        ///< [in] Binary scan operator having member <tt>T operator()(const T &a, const T &b)</tt>
    {
        T block_aggregate;
        InclusiveScan(smem_storage, input, output, scan_op, block_aggregate);
    }


    /**
     * \brief Computes an inclusive threadblock-wide prefix scan using the specified binary \p scan_op functor.  Each thread contributes an array of consecutive input elements.
     *
     * \smemreuse
     *
     * \tparam ITEMS_PER_THREAD     <b>[inferred]</b> The number of consecutive items partitioned onto each thread.
     * \tparam ScanOp               <b>[inferred]</b> Binary scan operator type having member <tt>T operator()(const T &a, const T &b)</tt>
     */
    template <
    int             ITEMS_PER_THREAD,
    typename        ScanOp>
    static __device__ __forceinline__ void InclusiveScan(
        SmemStorage     &smem_storage,                  ///< [in] Shared reference to opaque SmemStorage layout
        T               (&input)[ITEMS_PER_THREAD],     ///< [in] Calling thread's input items
        T               (&output)[ITEMS_PER_THREAD],    ///< [out] Calling thread's output items (may be aliased to \p input)
        ScanOp          scan_op)                        ///< [in] Binary scan operator having member <tt>T operator()(const T &a, const T &b)</tt>
    {
        // Reduce consecutive thread items in registers
        T thread_partial = ThreadReduce(input, scan_op);

        // Exclusive threadblock-scan
        ExclusiveScan(smem_storage, thread_partial, thread_partial, scan_op);

        // Inclusive scan in registers with prefix
        ThreadScanInclusive(input, output, scan_op, thread_partial, (threadIdx.x != 0));
    }


    //@}
    /******************************************************************//**
     * \name Inclusive prefix sums
     *********************************************************************/
    //@{


    /**
     * \brief Computes an inclusive threadblock-wide prefix scan using addition (+) as the scan operator.  Each thread contributes one input element.  The inclusive threadblock-wide \p aggregate of all inputs is computed for <em>thread</em><sub>0</sub>.
     *
     * The \p aggregate is undefined in threads other than <em>thread</em><sub>0</sub>.
     *
     * \smemreuse
     */
    static __device__ __forceinline__ void InclusiveSum(
        SmemStorage     &smem_storage,                  ///< [in] Shared reference to opaque SmemStorage layout
        T               input,                          ///< [in] Calling thread's input item
        T               &output,                        ///< [out] Calling thread's output item (may be aliased to \p input)
        T               &block_aggregate)               ///< [out] <b>[<em>thread</em><sub>0</sub> only]</b> threadblock-wide aggregate reduction of input items
    {
        BlockScanInternal<SAFE_POLICY>::InclusiveSum(smem_storage, input, output, block_aggregate);
    }


    /**
     * \brief Computes an inclusive threadblock-wide prefix scan using addition (+) as the scan operator.  Each thread contributes an array of consecutive input elements.  The inclusive threadblock-wide \p aggregate of all inputs is computed for <em>thread</em><sub>0</sub>.
     *
     * The \p aggregate is undefined in threads other than <em>thread</em><sub>0</sub>.
     *
     * \smemreuse
     *
     * \tparam ITEMS_PER_THREAD     <b>[inferred]</b> The number of consecutive items partitioned onto each thread.
     * \tparam ScanOp               <b>[inferred]</b> Binary scan operator type having member <tt>T operator()(const T &a, const T &b)</tt>
     */
    template <int ITEMS_PER_THREAD>
    static __device__ __forceinline__ void InclusiveSum(
        SmemStorage     &smem_storage,                  ///< [in] Shared reference to opaque SmemStorage layout
        T               (&input)[ITEMS_PER_THREAD],     ///< [in] Calling thread's input items
        T               (&output)[ITEMS_PER_THREAD],    ///< [out] Calling thread's output items (may be aliased to \p input)
        T               &block_aggregate)               ///< [out] <b>[<em>thread</em><sub>0</sub> only]</b> threadblock-wide aggregate reduction of input items
    {
        // Reduce consecutive thread items in registers
        Sum<T> scan_op;
        T thread_partial = ThreadReduce(input, scan_op);

        // Exclusive threadblock-scan
        ExclusiveSum(smem_storage, thread_partial, thread_partial, block_aggregate);

        // Inclusive scan in registers with prefix
        ThreadScanInclusive(input, output, scan_op, thread_partial, (threadIdx.x != 0));
    }


    /**
     * \brief Computes an inclusive threadblock-wide prefix scan using addition (+) as the scan operator.  Each thread contributes one input element.  Instead of using \p identity as the threadblock-wide prefix, the call-back functor \p block_prefix_op is invoked by <em>thread</em><sub>0</sub> to provide the "seed" value that logically prefixes the threadblock's scan inputs.  The inclusive threadblock-wide \p aggregate of all inputs is computed for <em>thread</em><sub>0</sub>.
     *
     * The \p aggregate and \p block_prefix_op are undefined in threads other than <em>thread</em><sub>0</sub>.
     *
     * \smemreuse
     *
     * \tparam BlockPrefixOp          <b>[inferred]</b> Call-back functor type having member <tt>T operator()(T block_aggregate)</tt>
     */
    template <typename BlockPrefixOp>
    static __device__ __forceinline__ void InclusiveSum(
        SmemStorage     &smem_storage,                  ///< [in] Shared reference to opaque SmemStorage layout
        T               input,                          ///< [in] Calling thread's input item
        T               &output,                        ///< [out] Calling thread's output item (may be aliased to \p input)
        T               &block_aggregate,               ///< [out] <b>[<em>thread</em><sub>0</sub> only]</b> threadblock-wide aggregate reduction of input items, exclusive of the \p block_prefix_op value
        BlockPrefixOp   &block_prefix_op)               ///< [in-out] <b>[<em>thread</em><sub>0</sub> only]</b> Call-back functor having member <tt>T operator()(T block_aggregate)</tt> for specifying the appropriate external prefix (rather than \p identity), to be invoked internally by <em>thread</em><sub>0</sub>.  When provided the threadblock-wide aggregate of input items (also returned above), this functor is expected to return the logical threadblock-wide prefix to be applied during the scan operation.  Can be stateful.
    {
        BlockScanInternal<SAFE_POLICY>::InclusiveSum(smem_storage, input, output, block_aggregate, block_prefix_op);
    }


    /**
     * \brief Computes an inclusive threadblock-wide prefix scan using addition (+) as the scan operator.  Each thread contributes an array of consecutive input elements.  Instead of using \p identity as the threadblock-wide prefix, the call-back functor \p block_prefix_op is invoked by <em>thread</em><sub>0</sub> to provide the "seed" value that logically prefixes the threadblock's scan inputs.  The inclusive threadblock-wide \p aggregate of all inputs is computed for <em>thread</em><sub>0</sub>.
     *
     * The \p aggregate and \p block_prefix_op are undefined in threads other than <em>thread</em><sub>0</sub>.
     *
     * \smemreuse
     *
     * \tparam ITEMS_PER_THREAD     <b>[inferred]</b> The number of consecutive items partitioned onto each thread.
     * \tparam BlockPrefixOp        <b>[inferred]</b> Call-back functor type having member <tt>T operator()(T block_aggregate)</tt>
     */
    template <
    int ITEMS_PER_THREAD,
    typename BlockPrefixOp>
    static __device__ __forceinline__ void InclusiveSum(
        SmemStorage     &smem_storage,                  ///< [in] Shared reference to opaque SmemStorage layout
        T               (&input)[ITEMS_PER_THREAD],     ///< [in] Calling thread's input items
        T               (&output)[ITEMS_PER_THREAD],    ///< [out] Calling thread's output items (may be aliased to \p input)
        T               &block_aggregate,               ///< [out] <b>[<em>thread</em><sub>0</sub> only]</b> threadblock-wide aggregate reduction of input items, exclusive of the \p block_prefix_op value
        BlockPrefixOp   &block_prefix_op)               ///< [in-out] <b>[<em>thread</em><sub>0</sub> only]</b> Call-back functor having member <tt>T operator()(T block_aggregate)</tt> for specifying the appropriate external prefix (rather than \p identity), to be invoked internally by <em>thread</em><sub>0</sub>.  When provided the threadblock-wide aggregate of input items (also returned above), this functor is expected to return the logical threadblock-wide prefix to be applied during the scan operation.  Can be stateful.
    {
        // Reduce consecutive thread items in registers
        Sum<T> scan_op;
        T thread_partial = ThreadReduce(input, scan_op);

        // Exclusive threadblock-scan
        ExclusiveSum(smem_storage, thread_partial, thread_partial, block_aggregate, block_prefix_op);

        // Inclusive scan in registers with prefix
        ThreadScanInclusive(input, output, scan_op, thread_partial);
    }


    /**
     * \brief Computes an inclusive threadblock-wide prefix scan using addition (+) as the scan operator.  Each thread contributes one input element.
     *
     * \smemreuse
     */
    static __device__ __forceinline__ void InclusiveSum(
        SmemStorage     &smem_storage,                  ///< [in] Shared reference to opaque SmemStorage layout
        T               input,                          ///< [in] Calling thread's input item
        T               &output)                        ///< [out] Calling thread's output item (may be aliased to \p input)
    {
        T block_aggregate;
        BlockScanInternal<SAFE_POLICY>::InclusiveSum(smem_storage, input, output, block_aggregate);
    }


    /**
     * \brief Computes an inclusive threadblock-wide prefix scan using addition (+) as the scan operator.  Each thread contributes an array of consecutive input elements.
     *
     * \smemreuse
     *
     * \tparam ITEMS_PER_THREAD     <b>[inferred]</b> The number of consecutive items partitioned onto each thread.
     */
    template <int ITEMS_PER_THREAD>
    static __device__ __forceinline__ void InclusiveSum(
        SmemStorage     &smem_storage,                  ///< [in] Shared reference to opaque SmemStorage layout
        T               (&input)[ITEMS_PER_THREAD],     ///< [in] Calling thread's input items
        T               (&output)[ITEMS_PER_THREAD])    ///< [out] Calling thread's output items (may be aliased to \p input)
    {
        // Reduce consecutive thread items in registers
        Sum<T> scan_op;
        T thread_partial = ThreadReduce(input, scan_op);

        // Exclusive threadblock-scan
        ExclusiveSum(smem_storage, thread_partial, thread_partial);

        // Inclusive scan in registers with prefix
        ThreadScanInclusive(input, output, scan_op, thread_partial, (threadIdx.x != 0));
    }

    //@}        // Inclusive prefix sums

};

/** @} */       // SimtCoop

} // namespace cub
CUB_NS_POSTFIX
