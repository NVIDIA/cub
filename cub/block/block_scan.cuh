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
 * cub::BlockScan provides variants of parallel prefix scan across a CUDA threadblock.
 */

#pragma once

#include "../util_arch.cuh"
#include "../util_type.cuh"
#include "../block/block_raking_layout.cuh"
#include "../thread/thread_operators.cuh"
#include "../thread/thread_reduce.cuh"
#include "../thread/thread_scan.cuh"
#include "../warp/warp_scan.cuh"
#include "../util_namespace.cuh"

/// Optional outer namespace(s)
CUB_NS_PREFIX

/// CUB namespace
namespace cub {


/******************************************************************************
 * Algorithmic variants
 ******************************************************************************/

/**
 * \brief BlockScanAlgorithm enumerates alternative algorithms for cub::BlockScan to compute a parallel prefix scan across a CUDA thread block.
 */
enum BlockScanAlgorithm
{

    /**
     * \par Overview
     * An efficient "raking reduce-then-scan" prefix scan algorithm.  Execution is comprised of five phases:
     * -# Upsweep sequential reduction in registers (if threads contribute more than one input each).  Each thread then places the partial reduction of its item(s) into shared memory.
     * -# Upsweep sequential reduction in shared memory.  Threads within a single warp rake across segments of shared partial reductions.
     * -# A warp-synchronous Kogge-Stone style exclusive scan within the raking warp.
     * -# Downsweep sequential exclusive scan in shared memory.  Threads within a single warp rake across segments of shared partial reductions, seeded with the warp-scan output.
     * -# Downsweep sequential scan in registers (if threads contribute more than one input), seeded with the raking scan output.
     *
     * \par
     * \image html block_scan_raking.png
     * <div class="centercaption">\p BLOCK_SCAN_RAKING data flow for a hypothetical 16-thread threadblock and 4-thread raking warp.</div>
     *
     * \par Performance Considerations
     * - Although this variant may suffer longer turnaround latencies when the
     *   GPU is under-occupied, it can often provide higher overall throughput
     *   across the GPU when suitably occupied.
     */
    BLOCK_SCAN_RAKING,


    /**
     * \par Overview
     * Similar to cub::BLOCK_SCAN_RAKING, but with fewer shared memory reads at
     * the expense of higher register pressure.  Raking threads preserve their
     * "upsweep" segment of values in registers while performing warp-synchronous
     * scan, allowing the "downsweep" not to re-read them from shared memory.
     */
    BLOCK_SCAN_RAKING_MEMOIZE,


    /**
     * \par Overview
     * A quick "tiled warpscans" prefix scan algorithm.  Execution is comprised of four phases:
     * -# Upsweep sequential reduction in registers (if threads contribute more than one input each).  Each thread then places the partial reduction of its item(s) into shared memory.
     * -# Compute a shallow, but inefficient warp-synchronous Kogge-Stone style scan within each warp.
     * -# A propagation phase where the warp scan outputs in each warp are updated with the aggregate from each preceding warp.
     * -# Downsweep sequential scan in registers (if threads contribute more than one input), seeded with the raking scan output.
     *
     * \par
     * \image html block_scan_warpscans.png
     * <div class="centercaption">\p BLOCK_SCAN_WARP_SCANS data flow for a hypothetical 16-thread threadblock and 4-thread raking warp.</div>
     *
     * \par Performance Considerations
     * - Although this variant may suffer lower overall throughput across the
     *   GPU because due to a heavy reliance on inefficient warpscans, it can
     *   often provide lower turnaround latencies when the GPU is under-occupied.
     */
    BLOCK_SCAN_WARP_SCANS,
};


/******************************************************************************
 * Block scan
 ******************************************************************************/

/**
 * \brief BlockScan provides variants of parallel prefix scan (and prefix sum) across a CUDA threadblock. ![](scan_logo.png)
 * \ingroup BlockModule
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
 * For convenience, BlockScan provides alternative entrypoints that differ by:
 * - Operator (generic scan <em>vs.</em> prefix sum of numeric types)
 * - Granularity (single <em>vs.</em> multiple data items per thread)
 * - Output ordering (inclusive <em>vs.</em> exclusive)
 * - Block-wide prefix (identity <em>vs.</em> call-back functor)
 * - What is computed (scanned elements only <em>vs.</em> scanned elements and the total aggregate)
 *
 * \tparam T                The reduction input/output element type
 * \tparam BLOCK_THREADS    The threadblock size in threads
 * \tparam ALGORITHM        <b>[optional]</b> cub::BlockScanAlgorithm enumerator specifying the underlying algorithm to use (default = cub::BLOCK_SCAN_RAKING)
 *
 * \par Algorithm
 * BlockScan provides a single prefix scan abstraction whose performance behavior can be tuned
 * for different usage scenarios.  BlockScan can be (optionally) configured to use different algorithms that cater
 * to different latency/throughput needs:
 *   -# <b>cub::BLOCK_SCAN_RAKING</b>.  An efficient "raking reduce-then-scan" prefix scan algorithm. [More...](\ref cub::BlockScanAlgorithm)
 *   -# <b>cub::BLOCK_SCAN_WARP_SCANS</b>.  A quick "tiled warpscans" prefix scan algorithm. [More...](\ref cub::BlockScanAlgorithm)
 *
 * \par Usage Considerations
 * - Supports non-commutative scan operators
 * - Assumes a [<em>blocked arrangement</em>](index.html#sec3sec3) of elements across threads
 * - \smemreuse{BlockScan::TempStorage}
 *
 * \par Performance Considerations
 * - Uses special instructions when applicable (e.g., warp \p SHFL)
 * - Uses synchronization-free communication between warp lanes when applicable
 * - Uses only one or two threadblock-wide synchronization barriers (depending on
 *   algorithm selection)
 * - Zero bank conflicts for most types
 * - Computation is slightly more efficient (i.e., having lower instruction overhead) for:
 *   - Prefix sum variants (<em>vs.</em> generic scan)
 *   - Exclusive variants (<em>vs.</em> inclusive)
 *   - Simple scan variants that do not also input a \p block_prefix_op or compute a \p block_aggregate
 *   - \p T is a built-in C++ primitive or CUDA vector type (e.g., \p short, \p int2, \p double, \p float2, etc.)
 *   - \p BLOCK_THREADS is a multiple of the architecture's warp size
 * - See cub::BlockScanAlgorithm for performance details regarding algorithmic alternatives
 *
 * \par Examples
 * <em>Example 1.</em> Perform a simple exclusive prefix sum of 512 integer keys that
 * are partitioned in a blocked arrangement across a 128-thread threadblock (where each thread holds 4 keys).
 * \code
 * #include <cub/cub.cuh>
 *
 * __global__ void SomeKernel(...)
 * {
 *     // Parameterize BlockScan for 128 threads on type int
 *     typedef cub::BlockScan<int, 128> BlockScan;
 *
 *     // Declare shared memory for BlockScan
 *     __shared__ typename BlockScan::TempStorage temp_storage;
 *
 *     // A segment of consecutive input items per thread
 *     int data[4];
 *
 *     // Obtain items in blocked order
 *     ...
 *
 *     // Compute the threadblock-wide exclusive prefix sum
 *     BlockScan::ExclusiveSum(data, data);
 *
 *     ...
 * \endcode
 *
 * \par
 * <em>Example 2.</em> Use a single thread block to iteratively compute an exclusive prefix sum over a larger input using a prefix functor to maintain a running total between scans.
 *      \code
 *      #include <cub/cub.cuh>
 *
 *      // Stateful functor that maintains a running prefix that can be applied to
 *      // consecutive scan operations.
 *      struct BlockPrefixOp
 *      {
 *          // Running prefix
 *          int running_total;
 *
 *          // Functor constructor
 *          __device__ BlockPrefixOp(int running_total) : running_total(running_total) {}
 *
 *          // Functor operator.  Thread-0 produces a value for seeding the block-wide scan given
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
 *          // Parameterize BlockScan for 1 warp on type int
 *          typedef cub::BlockScan<int> BlockScan;
 *
 *          // Opaque shared memory for BlockScan
 *          __shared__ typename BlockScan::TempStorage temp_storage;
 *
 *          // Running total
 *          BlockPrefixOp prefix_op(0);
 *
 *          // Iterate in strips of BLOCK_THREADS items
 *          for (int block_offset = 0; block_offset < num_elements; block_offset += BLOCK_THREADS)
 *          {
 *              // Read item
 *              int datum = d_data[block_offset + linear_tid];
 *
 *              // Scan the tile of items
 *              int tile_aggregate;
 *              BlockScan::ExclusiveSum(datum, datum,
 *                  tile_aggregate, prefix_op);
 *
 *              // Write item
 *              d_data[block_offset + linear_tid] = datum;
 *          }
 *      \endcode
 *
 * \par
 * <em>Example 3:</em> Perform dynamic, contended allocation within a global data array \p d_data.
 * The global counter is only atomically updated once per block.
 * \code
 * #include <cub/cub.cuh>
 *
 * // Simple functor for producing a value for which to seed the entire block scan.
 * struct BlockPrefixOp
 * {
 *     int *d_global_counter;
 *
 *     // Functor constructor
 *     __device__ BlockPrefixOp(int *d_global_counter) : d_global_counter(d_global_counter) {}
 *
 *     // Functor operator.  Produces a value for seeding the threadblock-wide
 *     // scan given the local aggregate (only valid in thread-0).
 *     __device__ int operator()(int aggregate_block_request)
 *     {
 *         return (linear_tid == 0) ?
 *             atomicAdd(d_global_counter, aggregate_block_request) :  // thread0
 *             0;                                                      // anybody else
 *     }
 * }
 *
 * template <typename T, int BLOCK_THREADS>
 * __global__ void SomeKernel(int *d_global_counter, T *d_data, ...)
 * {
 *     // Parameterize BlockScan on type int
 *     typedef cub::BlockScan<int, BLOCK_THREADS> BlockScan;
 *
 *     // Declare shared memory for BlockScan
 *     __shared__ typename BlockScan::TempStorage temp_storage;
 *
 *     // Allocation request size for each thread
 *     int allocation_request = ...
 *
 *     // Determine a unique offset int d_out for each thread to
 *     // write its allocation
 *     int allocation_offset;
 *     int aggregate_block_request;  // Unused
 *     BlockScan::ExclusiveSum(allocation_request, allocation_offset,
 *         aggregate_block_request, BlockPrefix(d_global_counter));
 *
 *
 * \endcode
 */
template <
    typename            T,
    int                 BLOCK_THREADS,
    BlockScanAlgorithm  ALGORITHM = BLOCK_SCAN_RAKING>
class BlockScan
{
private:

    /******************************************************************************
     * Constants
     ******************************************************************************/

    /**
     * Ensure the template parameterization meets the requirements of the
     * specified algorithm. Currently, the BLOCK_SCAN_WARP_SCANS policy
     * cannot be used with threadblock sizes not a multiple of the
     * architectural warp size.
     */
    static const BlockScanAlgorithm SAFE_ALGORITHM =
        ((ALGORITHM == BLOCK_SCAN_WARP_SCANS) && (BLOCK_THREADS % PtxArchProps::WARP_THREADS != 0)) ?
            BLOCK_SCAN_RAKING :
            ALGORITHM;


    #ifndef DOXYGEN_SHOULD_SKIP_THIS    // Do not document


    /******************************************************************************
     * Algorithmic variants
     ******************************************************************************/

    /**
     * BLOCK_SCAN_RAKING and BLOCK_SCAN_RAKING_MEMOIZE algorithmic variants
     */
    template <BlockScanAlgorithm _ALGORITHM, int DUMMY = 0>
    struct BlockScanInternal
    {
        /// Layout type for padded threadblock raking grid
        typedef BlockRakingLayout<T, BLOCK_THREADS> BlockRakingLayout;

        /// Constants
        enum
        {
            /// Number of active warps
            WARPS = (BLOCK_THREADS + PtxArchProps::WARP_THREADS - 1) / PtxArchProps::WARP_THREADS,

            /// Number of raking threads
            RAKING_THREADS = BlockRakingLayout::RAKING_THREADS,

            /// Number of raking elements per warp synchronous raking thread
            SEGMENT_LENGTH = BlockRakingLayout::SEGMENT_LENGTH,

            /// Cooperative work can be entirely warp synchronous
            WARP_SYNCHRONOUS = (BLOCK_THREADS == RAKING_THREADS),
        };

        ///  WarpScan utility type
        typedef WarpScan<T, 1, RAKING_THREADS> WarpScan;

        /// Shared memory storage layout type
        struct TempStorage
        {
            typename WarpScan::TempStorage              warp_scan;          ///< Buffer for warp-synchronous scan
            typename BlockRakingLayout::TempStorage     raking_grid;        ///< Padded threadblock raking grid
            T                                           block_aggregate;    ///< Block aggregate
        };


        // Thread fields
        TempStorage &temp_storage;


        /// Constructor
        BlockScanInternal(TempStorage &temp_storage) : temp_storage(temp_storage) {}


        /// Raking helper structure
        struct RakingHelper
        {
            /// Copy of raking segment, promoted to registers
            T cached_segment[SEGMENT_LENGTH];

            // Performs upsweep raking reduction, returning the aggregate
            template <typename ScanOp>
            __device__ __forceinline__ T Upsweep(
                TempStorage&    temp_storage,
                ScanOp          scan_op)
            {
                T *smem_raking_ptr = BlockRakingLayout::RakingPtr(temp_storage.raking_grid);
                T *raking_ptr;

                if (_ALGORITHM == BLOCK_SCAN_RAKING_MEMOIZE)
                {
                    // Copy data into registers
                    #pragma unroll
                    for (int i = 0; i < SEGMENT_LENGTH; i++)
                    {
                        cached_segment[i] = smem_raking_ptr[i];
                    }
                    raking_ptr = cached_segment;
                }
                else
                {
                    raking_ptr = smem_raking_ptr;
                }

                T raking_partial = raking_ptr[0];

                #pragma unroll
                for (int i = 1; i < SEGMENT_LENGTH; i++)
                {
                    if ((BlockRakingLayout::UNGUARDED) || (((linear_tid * SEGMENT_LENGTH) + i) < BLOCK_THREADS))
                    {
                        raking_partial = scan_op(raking_partial, raking_ptr[i]);
                    }
                }

                return raking_partial;
            }


            /// Performs exclusive downsweep raking scan
            template <typename ScanOp>
            __device__ __forceinline__ void ExclusiveDownsweep(
                TempStorage&    temp_storage,
                ScanOp          scan_op,
                T               raking_partial,
                bool            apply_prefix = true)
            {
                T *smem_raking_ptr = BlockRakingLayout::RakingPtr(temp_storage.raking_grid);

                T *raking_ptr = (_ALGORITHM == BLOCK_SCAN_RAKING_MEMOIZE) ?
                    cached_segment :
                    smem_raking_ptr;

                ThreadScanExclusive<SEGMENT_LENGTH>(raking_ptr, raking_ptr, scan_op, raking_partial, apply_prefix);

                if (_ALGORITHM == BLOCK_SCAN_RAKING_MEMOIZE)
                {
                    // Copy data back to smem
                    #pragma unroll
                    for (int i = 0; i < SEGMENT_LENGTH; i++)
                    {
                        smem_raking_ptr[i] = cached_segment[i];
                    }
                }
            }


            /// Performs inclusive downsweep raking scan
            template <typename ScanOp>
            __device__ __forceinline__ void InclusiveDownsweep(
                TempStorage&    temp_storage,
                ScanOp          scan_op,
                T               raking_partial,
                bool            apply_prefix = true)
            {
                T *smem_raking_ptr = BlockRakingLayout::RakingPtr(temp_storage.raking_grid);

                T *raking_ptr = (_ALGORITHM == BLOCK_SCAN_RAKING_MEMOIZE) ?
                    cached_segment :
                    smem_raking_ptr;

                ThreadScanInclusive<SEGMENT_LENGTH>(raking_ptr, raking_ptr, scan_op, raking_partial, apply_prefix);

                if (_ALGORITHM == BLOCK_SCAN_RAKING_MEMOIZE)
                {
                    // Copy data back to smem
                    #pragma unroll
                    for (int i = 0; i < SEGMENT_LENGTH; i++)
                    {
                        smem_raking_ptr[i] = cached_segment[i];
                    }
                }
            }
        };

        /// Computes an exclusive threadblock-wide prefix scan using the specified binary \p scan_op functor.  Each thread contributes one input element.  Also provides every thread with the block-wide \p block_aggregate of all inputs.
        template <typename ScanOp>
        __device__ __forceinline__ void ExclusiveScan(
            T               input,              ///< [in] Calling thread's input items
            T               &output,            ///< [out] Calling thread's output items (may be aliased to \p input)
            const T         &identity,          ///< [in] Identity value
            ScanOp          scan_op,            ///< [in] Binary scan operator
            T               &block_aggregate)   ///< [out] Threadblock-wide aggregate reduction of input items
        {
            if (WARP_SYNCHRONOUS)
            {
                // Short-circuit directly to warp scan
                WarpScan::ExclusiveScan(
                    temp_storage.warp_scan,
                    input,
                    output,
                    identity,
                    scan_op,
                    block_aggregate);
            }
            else
            {
                // Place thread partial into shared memory raking grid
                T *placement_ptr = BlockRakingLayout::PlacementPtr(temp_storage.raking_grid);
                *placement_ptr = input;

                __syncthreads();

                // Reduce parallelism down to just raking threads
                if (linear_tid < RAKING_THREADS)
                {
                    // Raking upsweep reduction in grid
                    RakingHelper helper;
                    T raking_partial = helper.Upsweep(temp_storage, scan_op);

                    // Exclusive warp synchronous scan
                    WarpScan::ExclusiveScan(
                        temp_storage.warp_scan,
                        raking_partial,
                        raking_partial,
                        identity,
                        scan_op,
                        temp_storage.block_aggregate);

                    // Exclusive raking downsweep scan
                    helper.ExclusiveDownsweep(temp_storage, scan_op, raking_partial);
                }

                __syncthreads();

                // Grab thread prefix from shared memory
                output = *placement_ptr;

                // Retrieve block aggregate
                block_aggregate = temp_storage.block_aggregate;
            }
        }


        /// Computes an exclusive threadblock-wide prefix scan using the specified binary \p scan_op functor.  Each thread contributes one input element.  the call-back functor \p block_prefix_op is invoked by the first warp in the block, and the value returned by <em>lane</em><sub>0</sub> in that warp is used as the "seed" value that logically prefixes the threadblock's scan inputs.  Also provides every thread with the block-wide \p block_aggregate of all inputs.
        template <
            typename        ScanOp,
            typename        BlockPrefixOp>
        __device__ __forceinline__ void ExclusiveScan(
            T               input,                          ///< [in] Calling thread's input item
            T               &output,                        ///< [out] Calling thread's output item (may be aliased to \p input)
            T               identity,                       ///< [in] Identity value
            ScanOp          scan_op,                        ///< [in] Binary scan operator
            T               &block_aggregate,               ///< [out] Threadblock-wide aggregate reduction of input items (exclusive of the \p block_prefix_op value)
            BlockPrefixOp   &block_prefix_op)               ///< [in-out] <b>[<em>warp</em><sub>0</sub> only]</b> Call-back functor for specifying a threadblock-wide prefix to be applied to all inputs.
        {
            if (WARP_SYNCHRONOUS)
            {
                // Short-circuit directly to warp scan
                WarpScan::ExclusiveScan(
                    temp_storage.warp_scan,
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
                T *placement_ptr = BlockRakingLayout::PlacementPtr(temp_storage.raking_grid);
                *placement_ptr = input;

                __syncthreads();

                // Reduce parallelism down to just raking threads
                if (linear_tid < RAKING_THREADS)
                {
                    // Raking upsweep reduction in grid
                    RakingHelper helper;
                    T raking_partial = helper.Upsweep(temp_storage, scan_op);

                    // Exclusive warp synchronous scan
                    WarpScan::ExclusiveScan(
                        temp_storage.warp_scan,
                        raking_partial,
                        raking_partial,
                        identity,
                        scan_op,
                        temp_storage.block_aggregate,
                        block_prefix_op);

                    // Exclusive raking downsweep scan
                    helper.ExclusiveDownsweep(temp_storage, scan_op, raking_partial);
                }

                __syncthreads();

                // Grab thread prefix from shared memory
                output = *placement_ptr;

                // Retrieve block aggregate
                block_aggregate = temp_storage.block_aggregate;
            }
        }


        /// Computes an exclusive threadblock-wide prefix scan using the specified binary \p scan_op functor.  Each thread contributes one input element.  Also provides every thread with the block-wide \p block_aggregate of all inputs.  With no identity value, the output computed for <em>thread</em><sub>0</sub> is undefined.
        template <typename ScanOp>
        __device__ __forceinline__ void ExclusiveScan(
            T               input,                          ///< [in] Calling thread's input item
            T               &output,                        ///< [out] Calling thread's output item (may be aliased to \p input)
            ScanOp          scan_op,                        ///< [in] Binary scan operator
            T               &block_aggregate)               ///< [out] Threadblock-wide aggregate reduction of input items
        {
            if (WARP_SYNCHRONOUS)
            {
                // Short-circuit directly to warp scan
                WarpScan::ExclusiveScan(
                    temp_storage.warp_scan,
                    input,
                    output,
                    scan_op,
                    block_aggregate);
            }
            else
            {
                // Place thread partial into shared memory raking grid
                T *placement_ptr = BlockRakingLayout::PlacementPtr(temp_storage.raking_grid);
                *placement_ptr = input;

                __syncthreads();

                // Reduce parallelism down to just raking threads
                if (linear_tid < RAKING_THREADS)
                {
                    // Raking upsweep reduction in grid
                    RakingHelper helper;
                    T raking_partial = helper.Upsweep(temp_storage, scan_op);

                    // Exclusive warp synchronous scan
                    WarpScan::ExclusiveScan(
                        temp_storage.warp_scan,
                        raking_partial,
                        raking_partial,
                        scan_op,
                        temp_storage.block_aggregate);

                    // Exclusive raking downsweep scan
                    helper.ExclusiveDownsweep(scan_op, raking_partial, (linear_tid != 0));
                }

                __syncthreads();

                // Grab thread prefix from shared memory
                output = *placement_ptr;

                // Retrieve block aggregate
                block_aggregate = temp_storage.block_aggregate;
            }
        }


        /// Computes an exclusive threadblock-wide prefix scan using the specified binary \p scan_op functor.  Each thread contributes one input element.  the call-back functor \p block_prefix_op is invoked by the first warp in the block, and the value returned by <em>lane</em><sub>0</sub> in that warp is used as the "seed" value that logically prefixes the threadblock's scan inputs.  Also provides every thread with the block-wide \p block_aggregate of all inputs.
        template <
            typename ScanOp,
            typename BlockPrefixOp>
        __device__ __forceinline__ void ExclusiveScan(
            T               input,                          ///< [in] Calling thread's input item
            T               &output,                        ///< [out] Calling thread's output item (may be aliased to \p input)
            ScanOp          scan_op,                        ///< [in] Binary scan operator
            T               &block_aggregate,               ///< [out] Threadblock-wide aggregate reduction of input items (exclusive of the \p block_prefix_op value)
            BlockPrefixOp   &block_prefix_op)               ///< [in-out] <b>[<em>warp</em><sub>0</sub> only]</b> Call-back functor for specifying a threadblock-wide prefix to be applied to all inputs.
        {
            if (WARP_SYNCHRONOUS)
            {
                // Short-circuit directly to warp scan
                WarpScan::ExclusiveScan(
                    temp_storage.warp_scan,
                    input,
                    output,
                    scan_op,
                    block_aggregate,
                    block_prefix_op);
            }
            else
            {
                // Place thread partial into shared memory raking grid
                T *placement_ptr = BlockRakingLayout::PlacementPtr(temp_storage.raking_grid);
                *placement_ptr = input;

                __syncthreads();

                // Reduce parallelism down to just raking threads
                if (linear_tid < RAKING_THREADS)
                {
                    // Raking upsweep reduction in grid
                    RakingHelper helper;
                    T raking_partial = helper.Upsweep(temp_storage, scan_op);

                    // Exclusive warp synchronous scan
                    WarpScan::ExclusiveScan(
                        temp_storage.warp_scan,
                        raking_partial,
                        raking_partial,
                        scan_op,
                        temp_storage.block_aggregate,
                        block_prefix_op);

                    // Exclusive raking downsweep scan
                    helper.ExclusiveDownsweep(temp_storage, scan_op, raking_partial);
                }

                __syncthreads();

                // Grab thread prefix from shared memory
                output = *placement_ptr;

                // Retrieve block aggregate
                block_aggregate = temp_storage.block_aggregate;
            }
        }


        /// Computes an exclusive threadblock-wide prefix scan using addition (+) as the scan operator.  Each thread contributes one input element.  Also provides every thread with the block-wide \p block_aggregate of all inputs.
        __device__ __forceinline__ void ExclusiveSum(
            T               input,                          ///< [in] Calling thread's input item
            T               &output,                        ///< [out] Calling thread's output item (may be aliased to \p input)
            T               &block_aggregate)               ///< [out] Threadblock-wide aggregate reduction of input items
        {
            if (WARP_SYNCHRONOUS)
            {
                // Short-circuit directly to warp scan
                WarpScan::ExclusiveSum(
                    temp_storage.warp_scan,
                    input,
                    output,
                    block_aggregate);
            }
            else
            {
                // Raking scan
                Sum<T> scan_op;

                // Place thread partial into shared memory raking grid
                T *placement_ptr = BlockRakingLayout::PlacementPtr(temp_storage.raking_grid);
                *placement_ptr = input;

                __syncthreads();

                // Reduce parallelism down to just raking threads
                if (linear_tid < RAKING_THREADS)
                {
                    // Raking upsweep reduction in grid
                    RakingHelper helper;
                    T raking_partial = helper.Upsweep(temp_storage, scan_op);

                    // Exclusive warp synchronous scan
                    WarpScan::ExclusiveSum(
                        temp_storage.warp_scan,
                        raking_partial,
                        raking_partial,
                        temp_storage.block_aggregate);

                    // Exclusive raking downsweep scan
                    helper.ExclusiveDownsweep(temp_storage, scan_op, raking_partial);
                }

                __syncthreads();

                // Grab thread prefix from shared memory
                output = *placement_ptr;

                // Retrieve block aggregate
                block_aggregate = temp_storage.block_aggregate;
            }
        }


        /// Computes an exclusive threadblock-wide prefix scan using addition (+) as the scan operator.  Each thread contributes one input element.  Instead of using 0 as the threadblock-wide prefix, the call-back functor \p block_prefix_op is invoked by the first warp in the block, and the value returned by <em>lane</em><sub>0</sub> in that warp is used as the "seed" value that logically prefixes the threadblock's scan inputs.  Also provides every thread with the block-wide \p block_aggregate of all inputs.
        template <typename BlockPrefixOp>
        __device__ __forceinline__ void ExclusiveSum(
            T               input,                          ///< [in] Calling thread's input item
            T               &output,                        ///< [out] Calling thread's output item (may be aliased to \p input)
            T               &block_aggregate,               ///< [out] Threadblock-wide aggregate reduction of input items (exclusive of the \p block_prefix_op value)
            BlockPrefixOp   &block_prefix_op)               ///< [in-out] <b>[<em>warp</em><sub>0</sub> only]</b> Call-back functor for specifying a threadblock-wide prefix to be applied to all inputs.
        {
            if (WARP_SYNCHRONOUS)
            {
                // Short-circuit directly to warp scan
                WarpScan::ExclusiveSum(
                    temp_storage.warp_scan,
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
                T *placement_ptr = BlockRakingLayout::PlacementPtr(temp_storage.raking_grid);
                *placement_ptr = input;

                __syncthreads();

                // Reduce parallelism down to just raking threads
                if (linear_tid < RAKING_THREADS)
                {
                    // Raking upsweep reduction in grid
                    RakingHelper helper;
                    T raking_partial = helper.Upsweep(temp_storage, scan_op);

                    // Exclusive warp synchronous scan
                    WarpScan::ExclusiveSum(
                        temp_storage.warp_scan,
                        raking_partial,
                        raking_partial,
                        temp_storage.block_aggregate,
                        block_prefix_op);

                    // Exclusive raking downsweep scan
                    helper.ExclusiveDownsweep(temp_storage, scan_op, raking_partial);
                }

                __syncthreads();

                // Grab thread prefix from shared memory
                output = *placement_ptr;

                // Retrieve block aggregate
                block_aggregate = temp_storage.block_aggregate;
            }
        }


        /// Computes an inclusive threadblock-wide prefix scan using the specified binary \p scan_op functor.  Each thread contributes one input element.  Also provides every thread with the block-wide \p block_aggregate of all inputs.
        template <typename ScanOp>
        __device__ __forceinline__ void InclusiveScan(
            T               input,                          ///< [in] Calling thread's input item
            T               &output,                        ///< [out] Calling thread's output item (may be aliased to \p input)
            ScanOp          scan_op,                        ///< [in] Binary scan operator
            T               &block_aggregate)               ///< [out] Threadblock-wide aggregate reduction of input items
        {
            if (WARP_SYNCHRONOUS)
            {
                // Short-circuit directly to warp scan
                WarpScan::InclusiveScan(
                    temp_storage.warp_scan,
                    input,
                    output,
                    scan_op,
                    block_aggregate);
            }
            else
            {
                // Place thread partial into shared memory raking grid
                T *placement_ptr = BlockRakingLayout::PlacementPtr(temp_storage.raking_grid);
                *placement_ptr = input;

                __syncthreads();

                // Reduce parallelism down to just raking threads
                if (linear_tid < RAKING_THREADS)
                {
                    // Raking upsweep reduction in grid
                    RakingHelper helper;
                    T raking_partial = helper.Upsweep(temp_storage, scan_op);

                    // Exclusive warp synchronous scan
                    WarpScan::ExclusiveScan(
                        temp_storage.warp_scan,
                        raking_partial,
                        raking_partial,
                        scan_op,
                        temp_storage.block_aggregate);

                    // Inclusive raking downsweep scan
                    helper.InclusiveDownsweep(temp_storage, scan_op, raking_partial, (linear_tid != 0));
                }

                __syncthreads();

                // Grab thread prefix from shared memory
                output = *placement_ptr;

                // Retrieve block aggregate
                block_aggregate = temp_storage.block_aggregate;
            }
        }


        /// Computes an inclusive threadblock-wide prefix scan using the specified binary \p scan_op functor.  Each thread contributes one input element.  the call-back functor \p block_prefix_op is invoked by the first warp in the block, and the value returned by <em>lane</em><sub>0</sub> in that warp is used as the "seed" value that logically prefixes the threadblock's scan inputs.  Also provides every thread with the block-wide \p block_aggregate of all inputs.
        template <
            typename ScanOp,
            typename BlockPrefixOp>
        __device__ __forceinline__ void InclusiveScan(
            T               input,                          ///< [in] Calling thread's input item
            T               &output,                        ///< [out] Calling thread's output item (may be aliased to \p input)
            ScanOp          scan_op,                        ///< [in] Binary scan operator
            T               &block_aggregate,               ///< [out] Threadblock-wide aggregate reduction of input items (exclusive of the \p block_prefix_op value)
            BlockPrefixOp   &block_prefix_op)               ///< [in-out] <b>[<em>warp</em><sub>0</sub> only]</b> Call-back functor for specifying a threadblock-wide prefix to be applied to all inputs.
        {
            if (WARP_SYNCHRONOUS)
            {
                // Short-circuit directly to warp scan
                WarpScan::InclusiveScan(
                    temp_storage.warp_scan,
                    input,
                    output,
                    scan_op,
                    block_aggregate,
                    block_prefix_op);
            }
            else
            {
                // Place thread partial into shared memory raking grid
                T *placement_ptr = BlockRakingLayout::PlacementPtr(temp_storage.raking_grid);
                *placement_ptr = input;

                __syncthreads();

                // Reduce parallelism down to just raking threads
                if (linear_tid < RAKING_THREADS)
                {
                    // Raking upsweep reduction in grid
                    RakingHelper helper;
                    T raking_partial = helper.Upsweep(temp_storage, scan_op);

                    // Warp synchronous scan
                    WarpScan::ExclusiveScan(
                        temp_storage.warp_scan,
                        raking_partial,
                        raking_partial,
                        scan_op,
                        temp_storage.block_aggregate,
                        block_prefix_op);

                    // Inclusive raking downsweep scan
                    helper.InclusiveDownsweep(temp_storage, scan_op, raking_partial);
                }

                __syncthreads();

                // Grab thread prefix from shared memory
                output = *placement_ptr;

                // Retrieve block aggregate
                block_aggregate = temp_storage.block_aggregate;
            }
        }


        /// Computes an inclusive threadblock-wide prefix scan using the specified binary \p scan_op functor.  Each thread contributes one input element.  Also provides every thread with the block-wide \p block_aggregate of all inputs.
        __device__ __forceinline__ void InclusiveSum(
            T               input,                          ///< [in] Calling thread's input item
            T               &output,                        ///< [out] Calling thread's output item (may be aliased to \p input)
            T               &block_aggregate)               ///< [out] Threadblock-wide aggregate reduction of input items
        {
            if (WARP_SYNCHRONOUS)
            {
                // Short-circuit directly to warp scan
                WarpScan::InclusiveSum(
                    temp_storage.warp_scan,
                    input,
                    output,
                    block_aggregate);
            }
            else
            {
                // Raking scan
                Sum<T> scan_op;

                // Place thread partial into shared memory raking grid
                T *placement_ptr = BlockRakingLayout::PlacementPtr(temp_storage.raking_grid);
                *placement_ptr = input;

                __syncthreads();

                // Reduce parallelism down to just raking threads
                if (linear_tid < RAKING_THREADS)
                {
                    // Raking upsweep reduction in grid
                    RakingHelper helper;
                    T raking_partial = helper.Upsweep(temp_storage, scan_op);

                    // Exclusive warp synchronous scan
                    WarpScan::ExclusiveSum(
                        temp_storage.warp_scan,
                        raking_partial,
                        raking_partial,
                        temp_storage.block_aggregate);

                    // Inclusive raking downsweep scan
                    helper.InclusiveDownsweep(temp_storage, scan_op, raking_partial, (linear_tid != 0));
                }

                __syncthreads();

                // Grab thread prefix from shared memory
                output = *placement_ptr;

                // Retrieve block aggregate
                block_aggregate = temp_storage.block_aggregate;
            }
        }


        /// Computes an inclusive threadblock-wide prefix scan using the specified binary \p scan_op functor.  Each thread contributes one input element.  Instead of using 0 as the threadblock-wide prefix, the call-back functor \p block_prefix_op is invoked by the first warp in the block, and the value returned by <em>lane</em><sub>0</sub> in that warp is used as the "seed" value that logically prefixes the threadblock's scan inputs.  Also provides every thread with the block-wide \p block_aggregate of all inputs.
        template <typename BlockPrefixOp>
        __device__ __forceinline__ void InclusiveSum(
            T               input,                          ///< [in] Calling thread's input item
            T               &output,                        ///< [out] Calling thread's output item (may be aliased to \p input)
            T               &block_aggregate,               ///< [out] Threadblock-wide aggregate reduction of input items (exclusive of the \p block_prefix_op value)
            BlockPrefixOp   &block_prefix_op)               ///< [in-out] <b>[<em>warp</em><sub>0</sub> only]</b> Call-back functor for specifying a threadblock-wide prefix to be applied to all inputs.
        {
            if (WARP_SYNCHRONOUS)
            {
                // Short-circuit directly to warp scan
                WarpScan::InclusiveSum(
                    temp_storage.warp_scan,
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
                T *placement_ptr = BlockRakingLayout::PlacementPtr(temp_storage.raking_grid);
                *placement_ptr = input;

                __syncthreads();

                // Reduce parallelism down to just raking threads
                if (linear_tid < RAKING_THREADS)
                {
                    // Raking upsweep reduction in grid
                    RakingHelper helper;
                    T raking_partial = helper.Upsweep(temp_storage, scan_op);

                    // Warp synchronous scan
                    WarpScan::ExclusiveSum(
                        temp_storage.warp_scan,
                        raking_partial,
                        raking_partial,
                        temp_storage.block_aggregate,
                        block_prefix_op);

                    // Inclusive raking downsweep scan
                    helper.InclusiveDownsweep(temp_storage, scan_op, raking_partial);
                }

                __syncthreads();

                // Grab thread prefix from shared memory
                output = *placement_ptr;

                // Retrieve block aggregate
                block_aggregate = temp_storage.block_aggregate;
            }
        }

    };



    /**
     * BLOCK_SCAN_WARP_SCANS algorithmic variant
     *
     * Can only be used when BLOCK_THREADS is a multiple of the architecture's warp size
     */
    template <int DUMMY>
    struct BlockScanInternal<BLOCK_SCAN_WARP_SCANS, DUMMY>
    {
        /// Constants
        enum
        {
            /// Number of active warps
            WARPS = (BLOCK_THREADS + PtxArchProps::WARP_THREADS - 1) / PtxArchProps::WARP_THREADS,
        };

        ///  WarpScan utility type
        typedef WarpScan<T, WARPS, PtxArchProps::WARP_THREADS> WarpScan;

        /// Shared memory storage layout type
        struct TempStorage
        {
            typename WarpScan::TempStorage      warp_scan;                  ///< Buffer for warp-synchronous scan
            T                                   warp_aggregates[WARPS];     ///< Shared totals from each warp-synchronous scan
            T                                   block_prefix;               ///< Shared prefix for the entire threadblock
        };


        // Thread fields
        TempStorage &temp_storage;
        unsigned int linear_tid;


        /// Constructor
        BlockScanInternal(
            TempStorage &temp_storage,
            unsigned int linear_tid)
        :
            temp_storage(temp_storage),
            linear_tid(linear_tid) {}


        /// Update the calling thread's partial reduction with the warp-wide aggregates from preceeding warps.  Also returns block-wide aggregate in <em>thread</em><sub>0</sub>.
        template <typename ScanOp>
        __device__ __forceinline__ void ApplyWarpAggregates(
            T               &partial,           ///< [out] The calling thread's partial reduction
            ScanOp          scan_op,            ///< [in] Binary scan operator
            T               warp_aggregate,     ///< [in] <b>[<em>lane</em><sub>0</sub>s only]</b> Warp-wide aggregate reduction of input items
            T               &block_aggregate,   ///< [out] Threadblock-wide aggregate reduction of input items
            bool            lane_valid = true)  ///< [in] Whether or not the partial belonging to the current thread is valid
        {
            unsigned int warp_id = linear_tid / PtxArchProps::WARP_THREADS;

            // Share lane aggregates
            temp_storage.warp_aggregates[warp_id] = warp_aggregate;

            __syncthreads();

            block_aggregate = temp_storage.warp_aggregates[0];

            #pragma unroll
            for (int WARP = 1; WARP < WARPS; WARP++)
            {
                if (warp_id == WARP)
                {
                    partial = (lane_valid) ?
                        scan_op(block_aggregate, partial) :     // fold it in our valid partial
                        block_aggregate;                        // replace our invalid partial with the aggregate
                }

                block_aggregate = scan_op(block_aggregate, temp_storage.warp_aggregates[WARP]);
            }
        }


        /// Computes an exclusive threadblock-wide prefix scan using the specified binary \p scan_op functor.  Each thread contributes one input element.  Also provides every thread with the block-wide \p block_aggregate of all inputs.
        template <typename ScanOp>
        __device__ __forceinline__ void ExclusiveScan(
            T               input,              ///< [in] Calling thread's input items
            T               &output,            ///< [out] Calling thread's output items (may be aliased to \p input)
            const T         &identity,          ///< [in] Identity value
            ScanOp          scan_op,            ///< [in] Binary scan operator
            T               &block_aggregate)   ///< [out] Threadblock-wide aggregate reduction of input items
        {
            T warp_aggregate;
            WarpScan::ExclusiveScan(temp_storage.warp_scan, input, output, identity, scan_op, warp_aggregate);

            // Update outputs and block_aggregate with warp-wide aggregates
            ApplyWarpAggregates(output, scan_op, warp_aggregate, block_aggregate);
        }


        /// Computes an exclusive threadblock-wide prefix scan using the specified binary \p scan_op functor.  Each thread contributes one input element.  the call-back functor \p block_prefix_op is invoked by the first warp in the block, and the value returned by <em>lane</em><sub>0</sub> in that warp is used as the "seed" value that logically prefixes the threadblock's scan inputs.  Also provides every thread with the block-wide \p block_aggregate of all inputs.
        template <
            typename ScanOp,
            typename BlockPrefixOp>
        __device__ __forceinline__ void ExclusiveScan(
            T               input,                          ///< [in] Calling thread's input item
            T               &output,                        ///< [out] Calling thread's output item (may be aliased to \p input)
            T               identity,                       ///< [in] Identity value
            ScanOp          scan_op,                        ///< [in] Binary scan operator
            T               &block_aggregate,               ///< [out] Threadblock-wide aggregate reduction of input items (exclusive of the \p block_prefix_op value)
            BlockPrefixOp   &block_prefix_op)               ///< [in-out] <b>[<em>warp</em><sub>0</sub> only]</b> Call-back functor for specifying a threadblock-wide prefix to be applied to all inputs.
        {
            ExclusiveScan(input, output, identity, scan_op, block_aggregate);

            // Compute and share threadblock prefix
            if (linear_tid == 0)
            {
                temp_storage.block_prefix = block_prefix_op(block_aggregate);
            }

            __syncthreads();

            // Incorporate threadblock prefix into outputs
            output = scan_op(temp_storage.block_prefix, output);
        }


        /// Computes an exclusive threadblock-wide prefix scan using the specified binary \p scan_op functor.  Each thread contributes one input element.  Also provides every thread with the block-wide \p block_aggregate of all inputs.  With no identity value, the output computed for <em>thread</em><sub>0</sub> is undefined.
        template <typename ScanOp>
        __device__ __forceinline__ void ExclusiveScan(
            T               input,                          ///< [in] Calling thread's input item
            T               &output,                        ///< [out] Calling thread's output item (may be aliased to \p input)
            ScanOp          scan_op,                        ///< [in] Binary scan operator
            T               &block_aggregate)               ///< [out] Threadblock-wide aggregate reduction of input items
        {
            T warp_aggregate;
            WarpScan::ExclusiveScan(temp_storage.warp_scan, input, output, scan_op, warp_aggregate);

            // Update outputs and block_aggregate with warp-wide aggregates
            unsigned int lane_id = linear_tid & (PtxArchProps::WARP_THREADS - 1);
            ApplyWarpAggregates(output, scan_op, warp_aggregate, block_aggregate, (lane_id > 0));
        }


        /// Computes an exclusive threadblock-wide prefix scan using the specified binary \p scan_op functor.  Each thread contributes one input element.  the call-back functor \p block_prefix_op is invoked by the first warp in the block, and the value returned by <em>lane</em><sub>0</sub> in that warp is used as the "seed" value that logically prefixes the threadblock's scan inputs.  Also provides every thread with the block-wide \p block_aggregate of all inputs.
        template <
            typename ScanOp,
            typename BlockPrefixOp>
        __device__ __forceinline__ void ExclusiveScan(
            T               input,                          ///< [in] Calling thread's input item
            T               &output,                        ///< [out] Calling thread's output item (may be aliased to \p input)
            ScanOp          scan_op,                        ///< [in] Binary scan operator
            T               &block_aggregate,               ///< [out] Threadblock-wide aggregate reduction of input items (exclusive of the \p block_prefix_op value)
            BlockPrefixOp   &block_prefix_op)               ///< [in-out] <b>[<em>warp</em><sub>0</sub> only]</b> Call-back functor for specifying a threadblock-wide prefix to be applied to all inputs.
        {
            ExclusiveScan(input, output, scan_op, block_aggregate);

            // Compute and share threadblock prefix
            unsigned int warp_id = linear_tid / PtxArchProps::WARP_THREADS;
            if (warp_id == 0)
            {
                temp_storage.block_prefix = block_prefix_op(block_aggregate);
            }

            __syncthreads();

            // Incorporate threadblock prefix into outputs
            output = (linear_tid == 0) ?
                temp_storage.block_prefix :
                scan_op(temp_storage.block_prefix, output);
        }


        /// Computes an exclusive threadblock-wide prefix scan using addition (+) as the scan operator.  Each thread contributes one input element.  Also provides every thread with the block-wide \p block_aggregate of all inputs.
        __device__ __forceinline__ void ExclusiveSum(
            T               input,                          ///< [in] Calling thread's input item
            T               &output,                        ///< [out] Calling thread's output item (may be aliased to \p input)
            T               &block_aggregate)               ///< [out] Threadblock-wide aggregate reduction of input items
        {
            T warp_aggregate;
            WarpScan::ExclusiveSum(temp_storage.warp_scan, input, output, warp_aggregate);

            // Update outputs and block_aggregate with warp-wide aggregates from lane-0s
            ApplyWarpAggregates(output, Sum<T>(), warp_aggregate, block_aggregate);
        }


        /// Computes an exclusive threadblock-wide prefix scan using addition (+) as the scan operator.  Each thread contributes one input element.  Instead of using 0 as the threadblock-wide prefix, the call-back functor \p block_prefix_op is invoked by the first warp in the block, and the value returned by <em>lane</em><sub>0</sub> in that warp is used as the "seed" value that logically prefixes the threadblock's scan inputs.  Also provides every thread with the block-wide \p block_aggregate of all inputs.
        template <typename BlockPrefixOp>
        __device__ __forceinline__ void ExclusiveSum(
            T               input,                          ///< [in] Calling thread's input item
            T               &output,                        ///< [out] Calling thread's output item (may be aliased to \p input)
            T               &block_aggregate,               ///< [out] Threadblock-wide aggregate reduction of input items (exclusive of the \p block_prefix_op value)
            BlockPrefixOp   &block_prefix_op)               ///< [in-out] <b>[<em>warp</em><sub>0</sub> only]</b> Call-back functor for specifying a threadblock-wide prefix to be applied to all inputs.
        {
            ExclusiveSum(input, output, block_aggregate);

            // Compute and share threadblock prefix
            unsigned int warp_id = linear_tid / PtxArchProps::WARP_THREADS;
            if (warp_id == 0)
            {
                temp_storage.block_prefix = block_prefix_op(block_aggregate);
            }

            __syncthreads();

            // Incorporate threadblock prefix into outputs
            Sum<T> scan_op;
            output = scan_op(temp_storage.block_prefix, output);
        }


        /// Computes an inclusive threadblock-wide prefix scan using the specified binary \p scan_op functor.  Each thread contributes one input element.  Also provides every thread with the block-wide \p block_aggregate of all inputs.
        template <typename ScanOp>
        __device__ __forceinline__ void InclusiveScan(
            T               input,                          ///< [in] Calling thread's input item
            T               &output,                        ///< [out] Calling thread's output item (may be aliased to \p input)
            ScanOp          scan_op,                        ///< [in] Binary scan operator
            T               &block_aggregate)               ///< [out] Threadblock-wide aggregate reduction of input items
        {
            T warp_aggregate;
            WarpScan::InclusiveScan(temp_storage.warp_scan, input, output, scan_op, warp_aggregate);

            // Update outputs and block_aggregate with warp-wide aggregates from lane-0s
            ApplyWarpAggregates(output, scan_op, warp_aggregate, block_aggregate);

        }


        /// Computes an inclusive threadblock-wide prefix scan using the specified binary \p scan_op functor.  Each thread contributes one input element.  the call-back functor \p block_prefix_op is invoked by the first warp in the block, and the value returned by <em>lane</em><sub>0</sub> in that warp is used as the "seed" value that logically prefixes the threadblock's scan inputs.  Also provides every thread with the block-wide \p block_aggregate of all inputs.
        template <
            typename ScanOp,
            typename BlockPrefixOp>
        __device__ __forceinline__ void InclusiveScan(
            T               input,                          ///< [in] Calling thread's input item
            T               &output,                        ///< [out] Calling thread's output item (may be aliased to \p input)
            ScanOp          scan_op,                        ///< [in] Binary scan operator
            T               &block_aggregate,               ///< [out] Threadblock-wide aggregate reduction of input items (exclusive of the \p block_prefix_op value)
            BlockPrefixOp   &block_prefix_op)               ///< [in-out] <b>[<em>warp</em><sub>0</sub> only]</b> Call-back functor for specifying a threadblock-wide prefix to be applied to all inputs.
        {
            InclusiveScan(input, output, scan_op, block_aggregate);

            // Compute and share threadblock prefix
            unsigned int warp_id = linear_tid / PtxArchProps::WARP_THREADS;
            if (warp_id == 0)
            {
                temp_storage.block_prefix = block_prefix_op(block_aggregate);
            }

            __syncthreads();

            // Incorporate threadblock prefix into outputs
            output = scan_op(temp_storage.block_prefix, output);
        }


        /// Computes an inclusive threadblock-wide prefix scan using the specified binary \p scan_op functor.  Each thread contributes one input element.  Also provides every thread with the block-wide \p block_aggregate of all inputs.
        __device__ __forceinline__ void InclusiveSum(
            T               input,                          ///< [in] Calling thread's input item
            T               &output,                        ///< [out] Calling thread's output item (may be aliased to \p input)
            T               &block_aggregate)               ///< [out] Threadblock-wide aggregate reduction of input items
        {
            T warp_aggregate;
            WarpScan::InclusiveSum(temp_storage.warp_scan, input, output, warp_aggregate);

            // Update outputs and block_aggregate with warp-wide aggregates from lane-0s
            ApplyWarpAggregates(output, Sum<T>(), warp_aggregate, block_aggregate);
        }


        /// Computes an inclusive threadblock-wide prefix scan using the specified binary \p scan_op functor.  Each thread contributes one input element.  Instead of using 0 as the threadblock-wide prefix, the call-back functor \p block_prefix_op is invoked by the first warp in the block, and the value returned by <em>lane</em><sub>0</sub> in that warp is used as the "seed" value that logically prefixes the threadblock's scan inputs.  Also provides every thread with the block-wide \p block_aggregate of all inputs.
        template <typename BlockPrefixOp>
        __device__ __forceinline__ void InclusiveSum(
            T               input,                          ///< [in] Calling thread's input item
            T               &output,                        ///< [out] Calling thread's output item (may be aliased to \p input)
            T               &block_aggregate,               ///< [out] Threadblock-wide aggregate reduction of input items (exclusive of the \p block_prefix_op value)
            BlockPrefixOp   &block_prefix_op)               ///< [in-out] <b>[<em>warp</em><sub>0</sub> only]</b> Call-back functor for specifying a threadblock-wide prefix to be applied to all inputs.
        {
            InclusiveSum(input, output, block_aggregate);

            // Compute and share threadblock prefix
            unsigned int warp_id = linear_tid / PtxArchProps::WARP_THREADS;
            if (warp_id == 0)
            {
                temp_storage.block_prefix = block_prefix_op(block_aggregate);
            }

            __syncthreads();

            // Incorporate threadblock prefix into outputs
            Sum<T> scan_op;
            output = scan_op(temp_storage.block_prefix, output);
        }

    };


    #endif // DOXYGEN_SHOULD_SKIP_THIS


    /******************************************************************************
     * Type definitions
     ******************************************************************************/

    /// Shared memory storage layout type for BlockScan
    typedef typename BlockScanInternal<SAFE_ALGORITHM>::TempStorage _TempStorage;


    /******************************************************************************
     * Utility methods
     ******************************************************************************/

    /// Internal storage allocator
    __device__ __forceinline__ _TempStorage& PrivateStorage()
    {
        __shared__ _TempStorage private_storage;
        return private_storage;
    }


    /******************************************************************************
     * Thread fields
     ******************************************************************************/

    /// Shared storage reference
    _TempStorage &temp_storage;

    /// Linear thread-id
    unsigned int linear_tid;


public:

    /// \smemstorage{BlockScan}
    typedef _TempStorage TempStorage;


    /******************************************************************//**
     * \name Collective construction
     *********************************************************************/

    /**
     * \brief Collective constructor for 1D thread blocks using a private static allocation of shared memory as temporary storage.
     */
    __device__ __forceinline__ BlockScan()
    :
        temp_storage(PrivateStorage()),
        linear_tid(threadIdx.x){}


    /**
     * \brief Collective constructor for 1D thread blocks using the specified memory allocation as temporary storage.
     */
    __device__ __forceinline__ BlockScan(
        TempStorage &temp_storage)                      ///< [in] Reference to memory allocation having layout type TempStorage
    :
        temp_storage(temp_storage),
        linear_tid(threadIdx.x){}


    /**
     * \brief Collective constructor for multidimensional thread blocks using a private static allocation of shared memory as temporary storage.
     */
    __device__ __forceinline__ BlockScan(
        unsigned int linear_tid)                        ///< [in] A suitable 1D thread-identifier for the calling thread (e.g., <tt>(threadIdx.y * blockDim.x) + linear_tid</tt> for 2D thread blocks)
    :
        temp_storage(PrivateStorage()),
        linear_tid(linear_tid){}


    /**
     * \brief Collective constructor for multidimensional thread blocks using the specified memory allocation as temporary storage.
     */
    __device__ __forceinline__ BlockScan(
        TempStorage &temp_storage,                      ///< [in] Reference to memory allocation having layout type TempStorage
        unsigned int linear_tid)                        ///< [in] <b>[optional]</b> A suitable 1D thread-identifier for the calling thread (e.g., <tt>(threadIdx.y * blockDim.x) + linear_tid</tt> for 2D thread blocks)
    :
        temp_storage(temp_storage),
        linear_tid(linear_tid){}



    //@}  end member group
    /******************************************************************//**
     * \name Exclusive prefix sums (single datum per thread)
     *********************************************************************/
    //@{


    /**
     * \brief Computes an exclusive threadblock-wide prefix scan using addition (+) as the scan operator.  Each thread contributes one input element.
     *
     * \smemreuse
     */
    __device__ __forceinline__ void ExclusiveSum(
        T               input,                          ///< [in] Calling thread's input item
        T               &output)                        ///< [out] Calling thread's output item (may be aliased to \p input)
    {
        T block_aggregate;
        BlockScanInternal<SAFE_ALGORITHM>(temp_storage, linear_tid).ExclusiveSum(input, output, block_aggregate);
    }


    /**
     * \brief Computes an exclusive threadblock-wide prefix scan using addition (+) as the scan operator.  Each thread contributes one input element.  Also provides every thread with the block-wide \p block_aggregate of all inputs.
     *
     * \smemreuse
     */
    __device__ __forceinline__ void ExclusiveSum(
        T               input,                          ///< [in] Calling thread's input item
        T               &output,                        ///< [out] Calling thread's output item (may be aliased to \p input)
        T               &block_aggregate)               ///< [out] Threadblock-wide aggregate reduction of input items
    {
        BlockScanInternal<SAFE_ALGORITHM>(temp_storage, linear_tid).ExclusiveSum(input, output, block_aggregate);
    }


    /**
     * \brief Computes an exclusive threadblock-wide prefix scan using addition (+) as the scan operator.  Each thread contributes one input element.  Instead of using 0 as the threadblock-wide prefix, the call-back functor \p block_prefix_op is invoked by the first warp in the block, and the value returned by <em>lane</em><sub>0</sub> in that warp is used as the "seed" value that logically prefixes the threadblock's scan inputs.  Also provides every thread with the block-wide \p block_aggregate of all inputs.
     *
     * The \p block_prefix_op functor must implement a member function <tt>T operator()(T block_aggregate)</tt>.
     * The functor's input parameter \p block_aggregate is the same value also returned by the scan operation.
     * The functor will be invoked by the first warp of threads in the block, however only the return value from
     * <em>lane</em><sub>0</sub> is applied as the threadblock-wide prefix.  Can be stateful.
     *
     * \smemreuse
     *
     * \tparam BlockPrefixOp        <b>[inferred]</b> Call-back functor type having member <tt>T operator()(T block_aggregate)</tt>
     */
    template <typename BlockPrefixOp>
    __device__ __forceinline__ void ExclusiveSum(
        T               input,                          ///< [in] Calling thread's input item
        T               &output,                        ///< [out] Calling thread's output item (may be aliased to \p input)
        T               &block_aggregate,               ///< [out] Threadblock-wide aggregate reduction of input items (exclusive of the \p block_prefix_op value)
        BlockPrefixOp   &block_prefix_op)               ///< [in-out] <b>[<em>warp</em><sub>0</sub> only]</b> Call-back functor for specifying a threadblock-wide prefix to be applied to all inputs.
    {
        BlockScanInternal<SAFE_ALGORITHM>(temp_storage, linear_tid).ExclusiveSum(input, output, block_aggregate, block_prefix_op);
    }


    //@}  end member group
    /******************************************************************//**
     * \name Exclusive prefix sums (multiple data per thread)
     *********************************************************************/
    //@{


    /**
     * \brief Computes an exclusive threadblock-wide prefix scan using addition (+) as the scan operator.  Each thread contributes an array of consecutive input elements.
     *
     * \smemreuse
     *
     * \tparam ITEMS_PER_THREAD     <b>[inferred]</b> The number of consecutive items partitioned onto each thread.
     */
    template <int ITEMS_PER_THREAD>
    __device__ __forceinline__ void ExclusiveSum(
        T                 (&input)[ITEMS_PER_THREAD],   ///< [in] Calling thread's input items
        T                 (&output)[ITEMS_PER_THREAD])  ///< [out] Calling thread's output items (may be aliased to \p input)
    {
        // Reduce consecutive thread items in registers
        Sum<T> scan_op;
        T thread_partial = ThreadReduce(input, scan_op);

        // Exclusive threadblock-scan
        ExclusiveSum(thread_partial, thread_partial);

        // Exclusive scan in registers with prefix
        ThreadScanExclusive(input, output, scan_op, thread_partial);
    }


    /**
     * \brief Computes an exclusive threadblock-wide prefix scan using addition (+) as the scan operator.  Each thread contributes an array of consecutive input elements.  Also provides every thread with the block-wide \p block_aggregate of all inputs.
     *
     * \smemreuse
     *
     * \tparam ITEMS_PER_THREAD     <b>[inferred]</b> The number of consecutive items partitioned onto each thread.
     */
    template <int ITEMS_PER_THREAD>
    __device__ __forceinline__ void ExclusiveSum(
        T                 (&input)[ITEMS_PER_THREAD],       ///< [in] Calling thread's input items
        T                 (&output)[ITEMS_PER_THREAD],      ///< [out] Calling thread's output items (may be aliased to \p input)
        T                 &block_aggregate)                 ///< [out] Threadblock-wide aggregate reduction of input items
    {
        // Reduce consecutive thread items in registers
        Sum<T> scan_op;
        T thread_partial = ThreadReduce(input, scan_op);

        // Exclusive threadblock-scan
        ExclusiveSum(thread_partial, thread_partial, block_aggregate);

        // Exclusive scan in registers with prefix
        ThreadScanExclusive(input, output, scan_op, thread_partial);
    }




    /**
     * \brief Computes an exclusive threadblock-wide prefix scan using addition (+) as the scan operator.  Each thread contributes an array of consecutive input elements.  Instead of using 0 as the threadblock-wide prefix, the call-back functor \p block_prefix_op is invoked by the first warp in the block, and the value returned by <em>lane</em><sub>0</sub> in that warp is used as the "seed" value that logically prefixes the threadblock's scan inputs.  Also provides every thread with the block-wide \p block_aggregate of all inputs.
     *
     * The \p block_prefix_op functor must implement a member function <tt>T operator()(T block_aggregate)</tt>.
     * The functor's input parameter \p block_aggregate is the same value also returned by the scan operation.
     * The functor will be invoked by the first warp of threads in the block, however only the return value from
     * <em>lane</em><sub>0</sub> is applied as the threadblock-wide prefix.  Can be stateful.
     *
     * \smemreuse
     *
     * \tparam ITEMS_PER_THREAD     <b>[inferred]</b> The number of consecutive items partitioned onto each thread.
     * \tparam BlockPrefixOp        <b>[inferred]</b> Call-back functor type having member <tt>T operator()(T block_aggregate)</tt>
     */
    template <
        int ITEMS_PER_THREAD,
        typename BlockPrefixOp>
    __device__ __forceinline__ void ExclusiveSum(
        T                 (&input)[ITEMS_PER_THREAD],   ///< [in] Calling thread's input items
        T                 (&output)[ITEMS_PER_THREAD],  ///< [out] Calling thread's output items (may be aliased to \p input)
        T                 &block_aggregate,             ///< [out] Threadblock-wide aggregate reduction of input items (exclusive of the \p block_prefix_op value)
        BlockPrefixOp     &block_prefix_op)             ///< [in-out] <b>[<em>warp</em><sub>0</sub> only]</b> Call-back functor for specifying a threadblock-wide prefix to be applied to all inputs.
    {
        // Reduce consecutive thread items in registers
        Sum<T> scan_op;
        T thread_partial = ThreadReduce(input, scan_op);

        // Exclusive threadblock-scan
        ExclusiveSum(thread_partial, thread_partial, block_aggregate, block_prefix_op);

        // Exclusive scan in registers with prefix
        ThreadScanExclusive(input, output, scan_op, thread_partial);
    }


    //@}  end member group
    /******************************************************************//**
     * \name Inclusive prefix sums (single datum per thread)
     *********************************************************************/
    //@{


    /**
     * \brief Computes an inclusive threadblock-wide prefix scan using addition (+) as the scan operator.  Each thread contributes one input element.
     *
     * \smemreuse
     */
    __device__ __forceinline__ void InclusiveSum(
        T               input,                          ///< [in] Calling thread's input item
        T               &output)                        ///< [out] Calling thread's output item (may be aliased to \p input)
    {
        T block_aggregate;
        BlockScanInternal<SAFE_ALGORITHM>(temp_storage, linear_tid).InclusiveSum(input, output, block_aggregate);
    }


    /**
     * \brief Computes an inclusive threadblock-wide prefix scan using addition (+) as the scan operator.  Each thread contributes one input element.  Also provides every thread with the block-wide \p block_aggregate of all inputs.
     *
     * \smemreuse
     */
    __device__ __forceinline__ void InclusiveSum(
        T               input,                          ///< [in] Calling thread's input item
        T               &output,                        ///< [out] Calling thread's output item (may be aliased to \p input)
        T               &block_aggregate)               ///< [out] Threadblock-wide aggregate reduction of input items
    {
        BlockScanInternal<SAFE_ALGORITHM>(temp_storage, linear_tid).InclusiveSum(input, output, block_aggregate);
    }



    /**
     * \brief Computes an inclusive threadblock-wide prefix scan using addition (+) as the scan operator.  Each thread contributes one input element.  Instead of using 0 as the threadblock-wide prefix, the call-back functor \p block_prefix_op is invoked by the first warp in the block, and the value returned by <em>lane</em><sub>0</sub> in that warp is used as the "seed" value that logically prefixes the threadblock's scan inputs.  Also provides every thread with the block-wide \p block_aggregate of all inputs.
     *
     * The \p block_prefix_op functor must implement a member function <tt>T operator()(T block_aggregate)</tt>.
     * The functor's input parameter \p block_aggregate is the same value also returned by the scan operation.
     * The functor will be invoked by the first warp of threads in the block, however only the return value from
     * <em>lane</em><sub>0</sub> is applied as the threadblock-wide prefix.  Can be stateful.
     *
     * \smemreuse
     *
     * \tparam BlockPrefixOp          <b>[inferred]</b> Call-back functor type having member <tt>T operator()(T block_aggregate)</tt>
     */
    template <typename BlockPrefixOp>
    __device__ __forceinline__ void InclusiveSum(
        T               input,                          ///< [in] Calling thread's input item
        T               &output,                        ///< [out] Calling thread's output item (may be aliased to \p input)
        T               &block_aggregate,               ///< [out] Threadblock-wide aggregate reduction of input items (exclusive of the \p block_prefix_op value)
        BlockPrefixOp   &block_prefix_op)               ///< [in-out] <b>[<em>warp</em><sub>0</sub> only]</b> Call-back functor for specifying a threadblock-wide prefix to be applied to all inputs.
    {
        BlockScanInternal<SAFE_ALGORITHM>(temp_storage, linear_tid).InclusiveSum(input, output, block_aggregate, block_prefix_op);
    }


    //@}  end member group
    /******************************************************************//**
     * \name Inclusive prefix sums (multiple data per thread)
     *********************************************************************/
    //@{



    /**
     * \brief Computes an inclusive threadblock-wide prefix scan using addition (+) as the scan operator.  Each thread contributes an array of consecutive input elements.
     *
     * \smemreuse
     *
     * \tparam ITEMS_PER_THREAD     <b>[inferred]</b> The number of consecutive items partitioned onto each thread.
     */
    template <int ITEMS_PER_THREAD>
    __device__ __forceinline__ void InclusiveSum(
        T               (&input)[ITEMS_PER_THREAD],     ///< [in] Calling thread's input items
        T               (&output)[ITEMS_PER_THREAD])    ///< [out] Calling thread's output items (may be aliased to \p input)
    {
        if (ITEMS_PER_THREAD == 1)
        {
            InclusiveSum(input[0], output[0]);
        }
        else
        {
            // Reduce consecutive thread items in registers
            Sum<T> scan_op;
            T thread_partial = ThreadReduce(input, scan_op);

            // Exclusive threadblock-scan
            ExclusiveSum(thread_partial, thread_partial);

            // Inclusive scan in registers with prefix
            ThreadScanInclusive(input, output, scan_op, thread_partial, (linear_tid != 0));
        }
    }


    /**
     * \brief Computes an inclusive threadblock-wide prefix scan using addition (+) as the scan operator.  Each thread contributes an array of consecutive input elements.  Also provides every thread with the block-wide \p block_aggregate of all inputs.
     *
     * \smemreuse
     *
     * \tparam ITEMS_PER_THREAD     <b>[inferred]</b> The number of consecutive items partitioned onto each thread.
     * \tparam ScanOp               <b>[inferred]</b> Binary scan operator type having member <tt>T operator()(const T &a, const T &b)</tt>
     */
    template <int ITEMS_PER_THREAD>
    __device__ __forceinline__ void InclusiveSum(
        T               (&input)[ITEMS_PER_THREAD],     ///< [in] Calling thread's input items
        T               (&output)[ITEMS_PER_THREAD],    ///< [out] Calling thread's output items (may be aliased to \p input)
        T               &block_aggregate)               ///< [out] Threadblock-wide aggregate reduction of input items
    {
        if (ITEMS_PER_THREAD == 1)
        {
            InclusiveSum(input[0], output[0], block_aggregate);
        }
        else
        {
            // Reduce consecutive thread items in registers
            Sum<T> scan_op;
            T thread_partial = ThreadReduce(input, scan_op);

            // Exclusive threadblock-scan
            ExclusiveSum(thread_partial, thread_partial, block_aggregate);

            // Inclusive scan in registers with prefix
            ThreadScanInclusive(input, output, scan_op, thread_partial, (linear_tid != 0));
        }
    }


    /**
     * \brief Computes an inclusive threadblock-wide prefix scan using addition (+) as the scan operator.  Each thread contributes an array of consecutive input elements.  Instead of using 0 as the threadblock-wide prefix, the call-back functor \p block_prefix_op is invoked by the first warp in the block, and the value returned by <em>lane</em><sub>0</sub> in that warp is used as the "seed" value that logically prefixes the threadblock's scan inputs.  Also provides every thread with the block-wide \p block_aggregate of all inputs.
     *
     * The \p block_prefix_op functor must implement a member function <tt>T operator()(T block_aggregate)</tt>.
     * The functor's input parameter \p block_aggregate is the same value also returned by the scan operation.
     * The functor will be invoked by the first warp of threads in the block, however only the return value from
     * <em>lane</em><sub>0</sub> is applied as the threadblock-wide prefix.  Can be stateful.
     *
     * \smemreuse
     *
     * \tparam ITEMS_PER_THREAD     <b>[inferred]</b> The number of consecutive items partitioned onto each thread.
     * \tparam BlockPrefixOp        <b>[inferred]</b> Call-back functor type having member <tt>T operator()(T block_aggregate)</tt>
     */
    template <
        int ITEMS_PER_THREAD,
        typename BlockPrefixOp>
    __device__ __forceinline__ void InclusiveSum(
        T               (&input)[ITEMS_PER_THREAD],     ///< [in] Calling thread's input items
        T               (&output)[ITEMS_PER_THREAD],    ///< [out] Calling thread's output items (may be aliased to \p input)
        T               &block_aggregate,               ///< [out] Threadblock-wide aggregate reduction of input items (exclusive of the \p block_prefix_op value)
        BlockPrefixOp   &block_prefix_op)               ///< [in-out] <b>[<em>warp</em><sub>0</sub> only]</b> Call-back functor for specifying a threadblock-wide prefix to be applied to all inputs.
    {
        if (ITEMS_PER_THREAD == 1)
        {
            InclusiveSum(input[0], output[0], block_aggregate, block_prefix_op);
        }
        else
        {
            // Reduce consecutive thread items in registers
            Sum<T> scan_op;
            T thread_partial = ThreadReduce(input, scan_op);

            // Exclusive threadblock-scan
            ExclusiveSum(thread_partial, thread_partial, block_aggregate, block_prefix_op);

            // Inclusive scan in registers with prefix
            ThreadScanInclusive(input, output, scan_op, thread_partial);
        }
    }


    //@}  end member group        // Inclusive prefix sums
    /******************************************************************//**
     * \name Exclusive prefix scans (single datum per thread)
     *********************************************************************/
    //@{


    /**
     * \brief Computes an exclusive threadblock-wide prefix scan using the specified binary \p scan_op functor.  Each thread contributes one input element.
     *
     * \smemreuse
     *
     * \tparam ScanOp               <b>[inferred]</b> Binary scan operator type having member <tt>T operator()(const T &a, const T &b)</tt>
     */
    template <typename ScanOp>
    __device__ __forceinline__ void ExclusiveScan(
        T               input,                          ///< [in] Calling thread's input item
        T               &output,                        ///< [out] Calling thread's output item (may be aliased to \p input)
        T               identity,                       ///< [in] Identity value
        ScanOp          scan_op)                        ///< [in] Binary scan operator
    {
        T block_aggregate;
        BlockScanInternal<SAFE_ALGORITHM>(temp_storage, linear_tid).ExclusiveScan(input, output, identity, scan_op, block_aggregate);
    }


    /**
     * \brief Computes an exclusive threadblock-wide prefix scan using the specified binary \p scan_op functor.  Each thread contributes one input element.  Also provides every thread with the block-wide \p block_aggregate of all inputs.
     *
     * \smemreuse
     *
     * \tparam ScanOp   <b>[inferred]</b> Binary scan operator type having member <tt>T operator()(const T &a, const T &b)</tt>
     */
    template <typename ScanOp>
    __device__ __forceinline__ void ExclusiveScan(
        T               input,              ///< [in] Calling thread's input items
        T               &output,            ///< [out] Calling thread's output items (may be aliased to \p input)
        const T         &identity,          ///< [in] Identity value
        ScanOp          scan_op,            ///< [in] Binary scan operator
        T               &block_aggregate)   ///< [out] Threadblock-wide aggregate reduction of input items
    {
        BlockScanInternal<SAFE_ALGORITHM>(temp_storage, linear_tid).ExclusiveScan(input, output, identity, scan_op, block_aggregate);
    }


    /**
     * \brief Computes an exclusive threadblock-wide prefix scan using the specified binary \p scan_op functor.  Each thread contributes one input element.  the call-back functor \p block_prefix_op is invoked by the first warp in the block, and the value returned by <em>lane</em><sub>0</sub> in that warp is used as the "seed" value that logically prefixes the threadblock's scan inputs.  Also provides every thread with the block-wide \p block_aggregate of all inputs.
     *
     * The \p block_prefix_op functor must implement a member function <tt>T operator()(T block_aggregate)</tt>.
     * The functor's input parameter \p block_aggregate is the same value also returned by the scan operation.
     * The functor will be invoked by the first warp of threads in the block, however only the return value from
     * <em>lane</em><sub>0</sub> is applied as the threadblock-wide prefix.  Can be stateful.
     *
     * \smemreuse
     *
     * \tparam ScanOp               <b>[inferred]</b> Binary scan operator type having member <tt>T operator()(const T &a, const T &b)</tt>
     * \tparam BlockPrefixOp        <b>[inferred]</b> Call-back functor type having member <tt>T operator()(T block_aggregate)</tt>
     */
    template <
        typename ScanOp,
        typename BlockPrefixOp>
    __device__ __forceinline__ void ExclusiveScan(
        T               input,                          ///< [in] Calling thread's input item
        T               &output,                        ///< [out] Calling thread's output item (may be aliased to \p input)
        T               identity,                       ///< [in] Identity value
        ScanOp          scan_op,                        ///< [in] Binary scan operator
        T               &block_aggregate,               ///< [out] Threadblock-wide aggregate reduction of input items (exclusive of the \p block_prefix_op value)
        BlockPrefixOp   &block_prefix_op)               ///< [in-out] <b>[<em>warp</em><sub>0</sub> only]</b> Call-back functor for specifying a threadblock-wide prefix to be applied to all inputs.
    {
        BlockScanInternal<SAFE_ALGORITHM>(temp_storage, linear_tid).ExclusiveScan(input, output, identity, scan_op, block_aggregate, block_prefix_op);
    }


    //@}  end member group        // Inclusive prefix sums
    /******************************************************************//**
     * \name Exclusive prefix scans (multiple data per thread)
     *********************************************************************/
    //@{


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
        typename        ScanOp>
    __device__ __forceinline__ void ExclusiveScan(
        T                 (&input)[ITEMS_PER_THREAD],   ///< [in] Calling thread's input items
        T                 (&output)[ITEMS_PER_THREAD],  ///< [out] Calling thread's output items (may be aliased to \p input)
        const T           &identity,                    ///< [in] Identity value
        ScanOp            scan_op)                      ///< [in] Binary scan operator
    {
        // Reduce consecutive thread items in registers
        T thread_partial = ThreadReduce(input, scan_op);

        // Exclusive threadblock-scan
        ExclusiveScan(thread_partial, thread_partial, identity, scan_op);

        // Exclusive scan in registers with prefix
        ThreadScanExclusive(input, output, scan_op, thread_partial);
    }


    /**
     * \brief Computes an exclusive threadblock-wide prefix scan using the specified binary \p scan_op functor.  Each thread contributes an array of consecutive input elements.  Also provides every thread with the block-wide \p block_aggregate of all inputs.
     *
     * \smemreuse
     *
     * \tparam ITEMS_PER_THREAD     <b>[inferred]</b> The number of consecutive items partitioned onto each thread.
     * \tparam ScanOp               <b>[inferred]</b> Binary scan operator type having member <tt>T operator()(const T &a, const T &b)</tt>
     */
    template <
        int             ITEMS_PER_THREAD,
        typename        ScanOp>
    __device__ __forceinline__ void ExclusiveScan(
        T                 (&input)[ITEMS_PER_THREAD],   ///< [in] Calling thread's input items
        T                 (&output)[ITEMS_PER_THREAD],  ///< [out] Calling thread's output items (may be aliased to \p input)
        const T           &identity,                    ///< [in] Identity value
        ScanOp            scan_op,                      ///< [in] Binary scan operator
        T                 &block_aggregate)             ///< [out] Threadblock-wide aggregate reduction of input items
    {
        // Reduce consecutive thread items in registers
        T thread_partial = ThreadReduce(input, scan_op);

        // Exclusive threadblock-scan
        ExclusiveScan(thread_partial, thread_partial, identity, scan_op, block_aggregate);

        // Exclusive scan in registers with prefix
        ThreadScanExclusive(input, output, scan_op, thread_partial);
    }


    /**
     * \brief Computes an exclusive threadblock-wide prefix scan using the specified binary \p scan_op functor.  Each thread contributes an array of consecutive input elements.  the call-back functor \p block_prefix_op is invoked by the first warp in the block, and the value returned by <em>lane</em><sub>0</sub> in that warp is used as the "seed" value that logically prefixes the threadblock's scan inputs.  Also provides every thread with the block-wide \p block_aggregate of all inputs.
     *
     * The \p block_prefix_op functor must implement a member function <tt>T operator()(T block_aggregate)</tt>.
     * The functor's input parameter \p block_aggregate is the same value also returned by the scan operation.
     * The functor will be invoked by the first warp of threads in the block, however only the return value from
     * <em>lane</em><sub>0</sub> is applied as the threadblock-wide prefix.  Can be stateful.
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
    __device__ __forceinline__ void ExclusiveScan(
        T               (&input)[ITEMS_PER_THREAD],     ///< [in] Calling thread's input items
        T               (&output)[ITEMS_PER_THREAD],    ///< [out] Calling thread's output items (may be aliased to \p input)
        T               identity,                       ///< [in] Identity value
        ScanOp          scan_op,                        ///< [in] Binary scan operator
        T               &block_aggregate,               ///< [out] Threadblock-wide aggregate reduction of input items (exclusive of the \p block_prefix_op value)
        BlockPrefixOp   &block_prefix_op)               ///< [in-out] <b>[<em>warp</em><sub>0</sub> only]</b> Call-back functor for specifying a threadblock-wide prefix to be applied to all inputs.
    {
        // Reduce consecutive thread items in registers
        T thread_partial = ThreadReduce(input, scan_op);

        // Exclusive threadblock-scan
        ExclusiveScan(thread_partial, thread_partial, identity, scan_op, block_aggregate, block_prefix_op);

        // Exclusive scan in registers with prefix
        ThreadScanExclusive(input, output, scan_op, thread_partial);
    }


    //@}  end member group
    /******************************************************************//**
     * \name Identityless exclusive prefix scans (single datum per thread)
     *********************************************************************/
    //@{


    /**
     * \brief Computes an exclusive threadblock-wide prefix scan using the specified binary \p scan_op functor.  Each thread contributes one input element.  With no identity value, the output computed for <em>thread</em><sub>0</sub> is undefined.
     *
     * \smemreuse
     *
     * \tparam ScanOp               <b>[inferred]</b> Binary scan operator type having member <tt>T operator()(const T &a, const T &b)</tt>
     */
    template <typename ScanOp>
    __device__ __forceinline__ void ExclusiveScan(
        T               input,                          ///< [in] Calling thread's input item
        T               &output,                        ///< [out] Calling thread's output item (may be aliased to \p input)
        ScanOp          scan_op)                        ///< [in] Binary scan operator
    {
        T block_aggregate;
        BlockScanInternal<SAFE_ALGORITHM>(temp_storage, linear_tid).ExclusiveScan(input, output, scan_op, block_aggregate);
    }


    /**
     * \brief Computes an exclusive threadblock-wide prefix scan using the specified binary \p scan_op functor.  Each thread contributes one input element.  Also provides every thread with the block-wide \p block_aggregate of all inputs.  With no identity value, the output computed for <em>thread</em><sub>0</sub> is undefined.
     *
     * \smemreuse
     *
     * \tparam ScanOp   <b>[inferred]</b> Binary scan operator type having member <tt>T operator()(const T &a, const T &b)</tt>
     */
    template <typename ScanOp>
    __device__ __forceinline__ void ExclusiveScan(
        T               input,                          ///< [in] Calling thread's input item
        T               &output,                        ///< [out] Calling thread's output item (may be aliased to \p input)
        ScanOp          scan_op,                        ///< [in] Binary scan operator
        T               &block_aggregate)               ///< [out] Threadblock-wide aggregate reduction of input items
    {
        BlockScanInternal<SAFE_ALGORITHM>(temp_storage, linear_tid).ExclusiveScan(input, output, scan_op, block_aggregate);
    }


    /**
     * \brief Computes an exclusive threadblock-wide prefix scan using the specified binary \p scan_op functor.  Each thread contributes one input element.  the call-back functor \p block_prefix_op is invoked by the first warp in the block, and the value returned by <em>lane</em><sub>0</sub> in that warp is used as the "seed" value that logically prefixes the threadblock's scan inputs.  Also provides every thread with the block-wide \p block_aggregate of all inputs.
     *
     * The \p block_prefix_op functor must implement a member function <tt>T operator()(T block_aggregate)</tt>.
     * The functor's input parameter \p block_aggregate is the same value also returned by the scan operation.
     * The functor will be invoked by the first warp of threads in the block, however only the return value from
     * <em>lane</em><sub>0</sub> is applied as the threadblock-wide prefix.  Can be stateful.
     *
     * \smemreuse
     *
     * \tparam ScanOp               <b>[inferred]</b> Binary scan operator type having member <tt>T operator()(const T &a, const T &b)</tt>
     * \tparam BlockPrefixOp        <b>[inferred]</b> Call-back functor type having member <tt>T operator()(T block_aggregate)</tt>
     */
    template <
        typename ScanOp,
        typename BlockPrefixOp>
    __device__ __forceinline__ void ExclusiveScan(
        T               input,                          ///< [in] Calling thread's input item
        T               &output,                        ///< [out] Calling thread's output item (may be aliased to \p input)
        ScanOp          scan_op,                        ///< [in] Binary scan operator
        T               &block_aggregate,               ///< [out] Threadblock-wide aggregate reduction of input items (exclusive of the \p block_prefix_op value)
        BlockPrefixOp   &block_prefix_op)               ///< [in-out] <b>[<em>warp</em><sub>0</sub> only]</b> Call-back functor for specifying a threadblock-wide prefix to be applied to all inputs.
    {
        BlockScanInternal<SAFE_ALGORITHM>(temp_storage, linear_tid).ExclusiveScan(input, output, scan_op, block_aggregate, block_prefix_op);
    }


    //@}  end member group
    /******************************************************************//**
     * \name Identityless exclusive prefix scans (multiple data per thread)
     *********************************************************************/
    //@{


    /**
     * \brief Computes an exclusive threadblock-wide prefix scan using the specified binary \p scan_op functor.  Each thread contributes an array of consecutive input elements.  With no identity value, the output computed for <em>thread</em><sub>0</sub> is undefined.
     *
     * \smemreuse
     *
     * \tparam ITEMS_PER_THREAD     <b>[inferred]</b> The number of consecutive items partitioned onto each thread.
     * \tparam ScanOp               <b>[inferred]</b> Binary scan operator type having member <tt>T operator()(const T &a, const T &b)</tt>
     */
    template <
        int             ITEMS_PER_THREAD,
        typename        ScanOp>
    __device__ __forceinline__ void ExclusiveScan(
        T                 (&input)[ITEMS_PER_THREAD],   ///< [in] Calling thread's input items
        T                 (&output)[ITEMS_PER_THREAD],  ///< [out] Calling thread's output items (may be aliased to \p input)
        ScanOp            scan_op)                      ///< [in] Binary scan operator
    {
        // Reduce consecutive thread items in registers
        T thread_partial = ThreadReduce(input, scan_op);

        // Exclusive threadblock-scan
        ExclusiveScan(thread_partial, thread_partial, scan_op);

        // Exclusive scan in registers with prefix
        ThreadScanExclusive(input, output, scan_op, thread_partial, (linear_tid != 0));
    }


    /**
     * \brief Computes an exclusive threadblock-wide prefix scan using the specified binary \p scan_op functor.  Each thread contributes an array of consecutive input elements.  Also provides every thread with the block-wide \p block_aggregate of all inputs.  With no identity value, the output computed for <em>thread</em><sub>0</sub> is undefined.
     *
     * The scalar \p block_aggregate is undefined in threads other than <em>thread</em><sub>0</sub>.
     *
     * \smemreuse
     *
     * \tparam ITEMS_PER_THREAD     <b>[inferred]</b> The number of consecutive items partitioned onto each thread.
     * \tparam ScanOp               <b>[inferred]</b> Binary scan operator type having member <tt>T operator()(const T &a, const T &b)</tt>
     */
    template <
        int             ITEMS_PER_THREAD,
        typename        ScanOp>
    __device__ __forceinline__ void ExclusiveScan(
        T               (&input)[ITEMS_PER_THREAD],     ///< [in] Calling thread's input items
        T               (&output)[ITEMS_PER_THREAD],    ///< [out] Calling thread's output items (may be aliased to \p input)
        ScanOp          scan_op,                        ///< [in] Binary scan operator
        T               &block_aggregate)               ///< [out] Threadblock-wide aggregate reduction of input items
    {
        // Reduce consecutive thread items in registers
        T thread_partial = ThreadReduce(input, scan_op);

        // Exclusive threadblock-scan
        ExclusiveScan(thread_partial, thread_partial, scan_op, block_aggregate);

        // Exclusive scan in registers with prefix
        ThreadScanExclusive(input, output, scan_op, thread_partial, (linear_tid != 0));
    }


    /**
     * \brief Computes an exclusive threadblock-wide prefix scan using the specified binary \p scan_op functor.  Each thread contributes an array of consecutive input elements.  the call-back functor \p block_prefix_op is invoked by the first warp in the block, and the value returned by <em>lane</em><sub>0</sub> in that warp is used as the "seed" value that logically prefixes the threadblock's scan inputs.  Also provides every thread with the block-wide \p block_aggregate of all inputs.
     *
     * The \p block_prefix_op functor must implement a member function <tt>T operator()(T block_aggregate)</tt>.
     * The functor's input parameter \p block_aggregate is the same value also returned by the scan operation.
     * The functor will be invoked by the first warp of threads in the block, however only the return value from
     * <em>lane</em><sub>0</sub> is applied as the threadblock-wide prefix.  Can be stateful.
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
    __device__ __forceinline__ void ExclusiveScan(
        T               (&input)[ITEMS_PER_THREAD],   ///< [in] Calling thread's input items
        T               (&output)[ITEMS_PER_THREAD],  ///< [out] Calling thread's output items (may be aliased to \p input)
        ScanOp          scan_op,                      ///< [in] Binary scan operator
        T               &block_aggregate,             ///< [out] Threadblock-wide aggregate reduction of input items (exclusive of the \p block_prefix_op value)
        BlockPrefixOp   &block_prefix_op)             ///< [in-out] <b>[<em>warp</em><sub>0</sub> only]</b> Call-back functor for specifying a threadblock-wide prefix to be applied to all inputs.
    {
        // Reduce consecutive thread items in registers
        T thread_partial = ThreadReduce(input, scan_op);

        // Exclusive threadblock-scan
        ExclusiveScan(thread_partial, thread_partial, scan_op, block_aggregate, block_prefix_op);

        // Exclusive scan in registers with prefix
        ThreadScanExclusive(input, output, scan_op, thread_partial);
    }


    //@}  end member group
    /******************************************************************//**
     * \name Inclusive prefix scans (single datum per thread)
     *********************************************************************/
    //@{


    /**
     * \brief Computes an inclusive threadblock-wide prefix scan using the specified binary \p scan_op functor.  Each thread contributes one input element.
     *
     * \smemreuse
     *
     * \tparam ScanOp               <b>[inferred]</b> Binary scan operator type having member <tt>T operator()(const T &a, const T &b)</tt>
     */
    template <typename ScanOp>
    __device__ __forceinline__ void InclusiveScan(
        T               input,                          ///< [in] Calling thread's input item
        T               &output,                        ///< [out] Calling thread's output item (may be aliased to \p input)
        ScanOp          scan_op)                        ///< [in] Binary scan operator
    {
        T block_aggregate;
        InclusiveScan(input, output, scan_op, block_aggregate);
    }


    /**
     * \brief Computes an inclusive threadblock-wide prefix scan using the specified binary \p scan_op functor.  Each thread contributes one input element.  Also provides every thread with the block-wide \p block_aggregate of all inputs.
     *
     * \smemreuse
     *
     * \tparam ScanOp   <b>[inferred]</b> Binary scan operator type having member <tt>T operator()(const T &a, const T &b)</tt>
     */
    template <typename ScanOp>
    __device__ __forceinline__ void InclusiveScan(
        T               input,                          ///< [in] Calling thread's input item
        T               &output,                        ///< [out] Calling thread's output item (may be aliased to \p input)
        ScanOp          scan_op,                        ///< [in] Binary scan operator
        T               &block_aggregate)               ///< [out] Threadblock-wide aggregate reduction of input items
    {
        BlockScanInternal<SAFE_ALGORITHM>(temp_storage, linear_tid).InclusiveScan(input, output, scan_op, block_aggregate);
    }


    /**
     * \brief Computes an inclusive threadblock-wide prefix scan using the specified binary \p scan_op functor.  Each thread contributes one input element.  the call-back functor \p block_prefix_op is invoked by the first warp in the block, and the value returned by <em>lane</em><sub>0</sub> in that warp is used as the "seed" value that logically prefixes the threadblock's scan inputs.  Also provides every thread with the block-wide \p block_aggregate of all inputs.
     *
     * The \p block_prefix_op functor must implement a member function <tt>T operator()(T block_aggregate)</tt>.
     * The functor's input parameter \p block_aggregate is the same value also returned by the scan operation.
     * The functor will be invoked by the first warp of threads in the block, however only the return value from
     * <em>lane</em><sub>0</sub> is applied as the threadblock-wide prefix.  Can be stateful.
     *
     * \smemreuse
     *
     * \tparam ScanOp               <b>[inferred]</b> Binary scan operator type having member <tt>T operator()(const T &a, const T &b)</tt>
     * \tparam BlockPrefixOp        <b>[inferred]</b> Call-back functor type having member <tt>T operator()(T block_aggregate)</tt>
     */
    template <
        typename ScanOp,
        typename BlockPrefixOp>
    __device__ __forceinline__ void InclusiveScan(
        T               input,                          ///< [in] Calling thread's input item
        T               &output,                        ///< [out] Calling thread's output item (may be aliased to \p input)
        ScanOp          scan_op,                        ///< [in] Binary scan operator
        T               &block_aggregate,               ///< [out] Threadblock-wide aggregate reduction of input items (exclusive of the \p block_prefix_op value)
        BlockPrefixOp   &block_prefix_op)               ///< [in-out] <b>[<em>warp</em><sub>0</sub> only]</b> Call-back functor for specifying a threadblock-wide prefix to be applied to all inputs.
    {
        BlockScanInternal<SAFE_ALGORITHM>(temp_storage, linear_tid).InclusiveScan(input, output, scan_op, block_aggregate, block_prefix_op);
    }


    //@}  end member group
    /******************************************************************//**
     * \name Inclusive prefix scans (multiple data per thread)
     *********************************************************************/
    //@{


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
    __device__ __forceinline__ void InclusiveScan(
        T               (&input)[ITEMS_PER_THREAD],     ///< [in] Calling thread's input items
        T               (&output)[ITEMS_PER_THREAD],    ///< [out] Calling thread's output items (may be aliased to \p input)
        ScanOp          scan_op)                        ///< [in] Binary scan operator
    {
        if (ITEMS_PER_THREAD == 1)
        {
            InclusiveScan(input[0], output[0], scan_op);
        }
        else
        {
            // Reduce consecutive thread items in registers
            T thread_partial = ThreadReduce(input, scan_op);

            // Exclusive threadblock-scan
            ExclusiveScan(thread_partial, thread_partial, scan_op);

            // Inclusive scan in registers with prefix
            ThreadScanInclusive(input, output, scan_op, thread_partial, (linear_tid != 0));
        }
    }


    /**
     * \brief Computes an inclusive threadblock-wide prefix scan using the specified binary \p scan_op functor.  Each thread contributes an array of consecutive input elements.  Also provides every thread with the block-wide \p block_aggregate of all inputs.
     *
     * \smemreuse
     *
     * \tparam ITEMS_PER_THREAD     <b>[inferred]</b> The number of consecutive items partitioned onto each thread.
     * \tparam ScanOp               <b>[inferred]</b> Binary scan operator type having member <tt>T operator()(const T &a, const T &b)</tt>
     */
    template <
        int             ITEMS_PER_THREAD,
        typename         ScanOp>
    __device__ __forceinline__ void InclusiveScan(
        T               (&input)[ITEMS_PER_THREAD],     ///< [in] Calling thread's input items
        T               (&output)[ITEMS_PER_THREAD],    ///< [out] Calling thread's output items (may be aliased to \p input)
        ScanOp          scan_op,                        ///< [in] Binary scan operator
        T               &block_aggregate)               ///< [out] Threadblock-wide aggregate reduction of input items
    {
        if (ITEMS_PER_THREAD == 1)
        {
            InclusiveScan(input[0], output[0], scan_op, block_aggregate);
        }
        else
        {
            // Reduce consecutive thread items in registers
            T thread_partial = ThreadReduce(input, scan_op);

            // Exclusive threadblock-scan
            ExclusiveScan(thread_partial, thread_partial, scan_op, block_aggregate);

            // Inclusive scan in registers with prefix
            ThreadScanInclusive(input, output, scan_op, thread_partial, (linear_tid != 0));
        }
    }


    /**
     * \brief Computes an inclusive threadblock-wide prefix scan using the specified binary \p scan_op functor.  Each thread contributes an array of consecutive input elements.  the call-back functor \p block_prefix_op is invoked by the first warp in the block, and the value returned by <em>lane</em><sub>0</sub> in that warp is used as the "seed" value that logically prefixes the threadblock's scan inputs.  Also provides every thread with the block-wide \p block_aggregate of all inputs.
     *
     * The \p block_prefix_op functor must implement a member function <tt>T operator()(T block_aggregate)</tt>.
     * The functor's input parameter \p block_aggregate is the same value also returned by the scan operation.
     * The functor will be invoked by the first warp of threads in the block, however only the return value from
     * <em>lane</em><sub>0</sub> is applied as the threadblock-wide prefix.  Can be stateful.
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
    __device__ __forceinline__ void InclusiveScan(
        T               (&input)[ITEMS_PER_THREAD],     ///< [in] Calling thread's input items
        T               (&output)[ITEMS_PER_THREAD],    ///< [out] Calling thread's output items (may be aliased to \p input)
        ScanOp          scan_op,                        ///< [in] Binary scan operator
        T               &block_aggregate,               ///< [out] Threadblock-wide aggregate reduction of input items (exclusive of the \p block_prefix_op value)
        BlockPrefixOp   &block_prefix_op)               ///< [in-out] <b>[<em>warp</em><sub>0</sub> only]</b> Call-back functor for specifying a threadblock-wide prefix to be applied to all inputs.
    {
        if (ITEMS_PER_THREAD == 1)
        {
            InclusiveScan(input[0], output[0], scan_op, block_aggregate, block_prefix_op);
        }
        else
        {
            // Reduce consecutive thread items in registers
            T thread_partial = ThreadReduce(input, scan_op);

            // Exclusive threadblock-scan
            ExclusiveScan(thread_partial, thread_partial, scan_op, block_aggregate, block_prefix_op);

            // Inclusive scan in registers with prefix
            ThreadScanInclusive(input, output, scan_op, thread_partial);
        }
    }

    //@}  end member group


};

}               // CUB namespace
CUB_NS_POSTFIX  // Optional outer namespace(s)

