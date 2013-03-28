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
#include "../thread/thread_operators.cuh"
#include "../warp/warp_scan.cuh"
#include "../thread/thread_reduce.cuh"
#include "../thread/thread_scan.cuh"
#include "../util_namespace.cuh"

/// Optional outer namespace(s)
CUB_NS_PREFIX

/// CUB namespace
namespace cub {

/**
 * BlockScanAlgorithm enumerates alternative algorithms for parallel prefix
 * scan across a CUDA threadblock.
 */
enum BlockScanAlgorithm
{

    /**
     * \par Overview
     * An efficient "raking reduce-then-scan" prefix scan algorithm.  Scan execution is comprised of five phases:
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
     * A quick "tiled warpscans" prefix scan algorithm.  Scan execution is comprised of four phases:
     * -# Upsweep sequential reduction in registers (if threads contribute more than one input each).  Each thread then places the partial reduction of its item(s) into shared memory.
     * -# Compute a shallow, but inefficient warp-synchronous Kogge-Stone style scan within each warp.
     * -# A propagation phase where the warp scan outputs in each warp are updated with the aggregate from each preceding warp.
     * -# Downsweep sequential scan in registers (if threads contribute more than one input), seeded with the raking scan output.
     *
     * \par
     * \image html block_scan_warpscans.png
     * <div class="centercaption">\p BLOCK_SCAN_WARPSCANS data flow for a hypothetical 16-thread threadblock and 4-thread raking warp.</div>
     *
     * \par Performance Considerations
     * - Although this variant may suffer lower overall throughput across the
     *   GPU because due to a heavy reliance on inefficient warpscans, it can
     *   often provide lower turnaround latencies when the GPU is under-occupied.
     */
    BLOCK_SCAN_WARPSCANS,
};

/**
 * \addtogroup BlockModule
 * @{
 */

/**
 * \brief BlockScan provides variants of parallel prefix scan (and prefix sum) across a CUDA threadblock. ![](scan_logo.png)
 *
 * \par Overview
 * Given a list of input elements and a binary reduction operator, a [<em>prefix scan</em>](http://en.wikipedia.org/wiki/Prefix_sum)
 * produces an output list where each element is computed to be the reduction
 * of the elements occurring earlier in the input list.  <em>Prefix sum</em>
 * connotes a prefix scan with the addition operator. The term \em inclusive indicates
 * that the <em>i</em><sup>th</sup> output reduction includes the <em>i</em><sup>th</sup> input.
 * The term \em exclusive indicates the <em>i</em><sup>th</sup> input is not computed into
 * the <em>i</em><sup>th</sup> output reduction.
 *
 * \par
 * For convenience, BlockScan exposes a spectrum of entrypoints that differ by:
 * - Operator (generic scan <em>vs.</em> prefix sum for numeric types)
 * - Granularity (single <em>vs.</em> multiple items per thread)
 * - Output ordering (inclusive <em>vs.</em> exclusive)
 * - Block-wide prefix (identity <em>vs.</em> call-back functor)
 * - Output (scanned elements only <em>vs.</em> scanned elements and the total aggregate)
 *
 * \par
 * Furthermore, BlockScan provides a single prefix scan abstraction whose performance behavior can be tuned
 * externally.  In particular, BlockScan implements alternative cub::BlockScanAlgorithm strategies
 * catering to different latency/throughput needs.
 *
 * \tparam T                The reduction input/output element type
 * \tparam BLOCK_THREADS    The threadblock size in threads
 * \tparam ALGORITHM           <b>[optional]</b> cub::BlockScanAlgorithm tuning policy.  Default = cub::BLOCK_SCAN_RAKING.
 *
 * \par Algorithm
 * BlockScan can be (optionally) configured to use different algorithms:
 *   -# <b>cub::BLOCK_SCAN_RAKING</b>.  An efficient "raking reduce-then-scan" prefix scan algorithm. [More...](\ref cub::BlockScanAlgorithm)
 *   -# <b>cub::BLOCK_SCAN_WARPSCANS</b>.  A quick "tiled warpscans" prefix scan algorithm. [More...](\ref cub::BlockScanAlgorithm)
 *
 * \par Usage Considerations
 * - Supports non-commutative scan operators
 * - Assumes a [<em>blocked arrangement</em>](index.html#sec3sec3) of elements across threads
 * - Any threadblock-wide scalar inputs and outputs (e.g., \p block_prefix_op and \p block_aggregate) are
 *   only considered valid in <em>thread</em><sub>0</sub>
 * - \smemreuse{BlockScan::SmemStorage}
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
 *   - Basic scan variants that don't require scalar inputs and outputs (e.g., \p block_prefix_op and \p block_aggregate)
 *   - \p T is a built-in C++ primitive or CUDA vector type (e.g., \p short, \p int2, \p double, \p float2, etc.)
 *   - \p BLOCK_THREADS is a multiple of the architecture's warp size
 * - See cub::BlockScanAlgorithm for more performance details regarding algorithmic alternatives
 *
 * \par Examples
 * <em>Example 1.</em> Perform a simple exclusive prefix sum of 512 integer keys that
 * are partitioned in a blocked arrangement across a 128-thread threadblock (where each thread holds 4 keys).
 * \code
 * #include <cub.cuh>
 *
 * __global__ void SomeKernel(...)
 * {
 *     // Parameterize BlockScan for 128 threads on type int
 *     typedef cub::BlockScan<int, 128> BlockScan;
 *
 *     // Declare shared memory for BlockScan
 *     __shared__ typename BlockScan::SmemStorage smem_storage;
 *
 *     // A segment of consecutive input items per thread
 *     int data[4];
 *
 *     // Obtain items in blocked order
 *     ...
 *
 *     // Compute the threadblock-wide exclusve prefix sum
 *     BlockScan::ExclusiveSum(smem_storage, data, data);
 *
 *     ...
 * \endcode
 *
 * \par
 * <em>Example 2:</em> Perform inter-threadblock allocation within a global data structure by using local prefix sum that incorporates a single global atomic-add.
 * \code
 * #include <cub.cuh>
 *
 * /// Simple functor for producing a value for which to seed the entire block scan.
 * struct BlockPrefixOp
 * {
 *     int *d_global_counter;
 *
 *     /// Functor constructor
 *     __device__ BlockPrefixOp(int *d_global_counter) : d_global_counter(d_global_counter) {}
 *
 *     /// Functor operator.  Produces a value for seeding the threadblock-wide scan given
 *     /// the local aggregate (only valid in thread-0).
 *     __device__ int operator(int block_aggregate)
 *     {
 *         return (threadIdx.x == 0) ?
 *             atomicAdd(d_global_counter, block_aggregate) :      // thread0
 *             0;                                                  // anybody else
 *     }
 * }
 *
 * template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
 * __global__ void SomeKernel(int *d_global_counter, ...)
 * {
 *     // Parameterize BlockScan on type int
 *     typedef cub::BlockScan<int, BLOCK_THREADS> BlockScan;
 *
 *     // Declare shared memory for BlockScan
 *     __shared__ typename BlockScan::SmemStorage smem_storage;
 *
 *     // A segment of consecutive input items per thread
 *     int data[ITEMS_PER_THREAD];

 *     // Obtain keys in blocked order
 *     ...
 *
 *     // Compute the threadblock-wide exclusive prefix sum, seeded with a threadblock-wide prefix
 *     int aggregate;
 *     BlockScan::ExclusiveSum(smem_storage, data, data, block_aggregate, BlockPrefix(d_global_counter));
 * \endcode
 */
template <
    typename            T,
    int                 BLOCK_THREADS,
    BlockScanAlgorithm  ALGORITHM = BLOCK_SCAN_RAKING>
class BlockScan
{
private:

    enum
    {
        /// Ensure the parameterization meets the requirements of the specified algorithm
        SAFE_ALGORITHM =
            ((ALGORITHM == BLOCK_SCAN_WARPSCANS) && (BLOCK_THREADS % PtxArchProps::WARP_THREADS != 0)) ?    // BLOCK_SCAN_WARPSCANS policy cannot be used with threadblock sizes not a multiple of the architectural warp size
                BLOCK_SCAN_RAKING :
                ALGORITHM
    };


    #ifndef DOXYGEN_SHOULD_SKIP_THIS    // Do not document

    /**
     * Warpscan specialized for BLOCK_SCAN_RAKING variant
     */
    template <int _ALGORITHM, int DUMMY = 0>
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
            RAKING_LENGTH = BlockRakingLayout::RAKING_LENGTH,

            /// Cooperative work can be entirely warp synchronous
            WARP_SYNCHRONOUS = (BLOCK_THREADS == RAKING_THREADS),
        };

        ///  Raking warp-scan utility type
        typedef WarpScan<T, 1, RAKING_THREADS> WarpScan;

        /// Shared memory storage layout type
        struct SmemStorage
        {
            typename WarpScan::SmemStorage          warp_scan;      ///< Buffer for warp-synchronous scan
            typename BlockRakingLayout::SmemStorage   raking_grid;    ///< Padded threadblock raking grid
        };

        /// Computes an exclusive threadblock-wide prefix scan using the specified binary \p scan_op functor.  Each thread contributes one input element.  The inclusive threadblock-wide \p block_aggregate of all inputs is computed for <em>thread</em><sub>0</sub>.
        template <typename ScanOp>
        static __device__ __forceinline__ void ExclusiveScan(
            SmemStorage     &smem_storage,      ///< [in] Shared reference to opaque SmemStorage layout
            T               input,              ///< [in] Calling thread's input items
            T               &output,            ///< [out] Calling thread's output items (may be aliased to \p input)
            const T         &identity,          ///< [in] Identity value
            ScanOp          scan_op,            ///< [in] Binary scan operator
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
                T *placement_ptr = BlockRakingLayout::PlacementPtr(smem_storage.raking_grid);
                *placement_ptr = input;

                __syncthreads();

                // Reduce parallelism down to just raking threads
                if (threadIdx.x < RAKING_THREADS)
                {
                    // Raking upsweep reduction in grid
                    T *raking_ptr = BlockRakingLayout::RakingPtr(smem_storage.raking_grid);
                    T raking_partial = raking_ptr[0];

                    #pragma unroll
                    for (int i = 1; i < RAKING_LENGTH; i++)
                    {
                        if ((BlockRakingLayout::UNGUARDED) || (((threadIdx.x * RAKING_LENGTH) + i) < BLOCK_THREADS))
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


        /// Computes an exclusive threadblock-wide prefix scan using the specified binary \p scan_op functor.  Each thread contributes one input element.  The call-back functor \p block_prefix_op is invoked to provide the "seed" value that logically prefixes the threadblock's scan inputs.  The inclusive threadblock-wide \p block_aggregate of all inputs is computed for <em>thread</em><sub>0</sub>.
        template <
        typename ScanOp,
        typename BlockPrefixOp>
        static __device__ __forceinline__ void ExclusiveScan(
            SmemStorage     &smem_storage,                  ///< [in] Shared reference to opaque SmemStorage layout
            T               input,                          ///< [in] Calling thread's input item
            T               &output,                        ///< [out] Calling thread's output item (may be aliased to \p input)
            T               identity,                       ///< [in] Identity value
            ScanOp          scan_op,                        ///< [in] Binary scan operator
            T               &block_aggregate,               ///< [out] <b>[<em>thread</em><sub>0</sub> only]</b> Threadblock-wide aggregate reduction of input items (exclusive of the \p block_prefix_op value)
            BlockPrefixOp   &block_prefix_op)               ///< [in-out] <b>[<em>thread</em><sub>0</sub> only]</b> Call-back functor for specifying a threadblock-wide prefix to be applied to all inputs.
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
                T *placement_ptr = BlockRakingLayout::PlacementPtr(smem_storage.raking_grid);
                *placement_ptr = input;

                __syncthreads();

                // Reduce parallelism down to just raking threads
                if (threadIdx.x < RAKING_THREADS)
                {
                    // Raking upsweep reduction in grid
                    T *raking_ptr = BlockRakingLayout::RakingPtr(smem_storage.raking_grid);
                    T raking_partial = raking_ptr[0];

                    #pragma unroll
                    for (int i = 1; i < RAKING_LENGTH; i++)
                    {
                        if ((BlockRakingLayout::UNGUARDED) || (((threadIdx.x * RAKING_LENGTH) + i) < BLOCK_THREADS))
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


        /// Computes an exclusive threadblock-wide prefix scan using the specified binary \p scan_op functor.  Each thread contributes one input element.  The inclusive threadblock-wide \p block_aggregate of all inputs is computed for <em>thread</em><sub>0</sub>.  With no identity value, the output computed for <em>thread</em><sub>0</sub> is invalid.
        template <typename ScanOp>
        static __device__ __forceinline__ void ExclusiveScan(
            SmemStorage     &smem_storage,                  ///< [in] Shared reference to opaque SmemStorage layout
            T               input,                          ///< [in] Calling thread's input item
            T               &output,                        ///< [out] Calling thread's output item (may be aliased to \p input)
            ScanOp          scan_op,                        ///< [in] Binary scan operator
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
                T *placement_ptr = BlockRakingLayout::PlacementPtr(smem_storage.raking_grid);
                *placement_ptr = input;

                __syncthreads();

                // Reduce parallelism down to just raking threads
                if (threadIdx.x < RAKING_THREADS)
                {
                    // Raking upsweep reduction in grid
                    T *raking_ptr = BlockRakingLayout::RakingPtr(smem_storage.raking_grid);
                    T raking_partial = raking_ptr[0];

                    #pragma unroll
                    for (int i = 1; i < RAKING_LENGTH; i++)
                    {
                        if ((BlockRakingLayout::UNGUARDED) || (((threadIdx.x * RAKING_LENGTH) + i) < BLOCK_THREADS))
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


        /// Computes an exclusive threadblock-wide prefix scan using the specified binary \p scan_op functor.  Each thread contributes one input element.  The call-back functor \p block_prefix_op is invoked to provide the "seed" value that logically prefixes the threadblock's scan inputs.  The inclusive threadblock-wide \p block_aggregate of all inputs is computed for <em>thread</em><sub>0</sub>.
        template <
        typename ScanOp,
        typename BlockPrefixOp>
        static __device__ __forceinline__ void ExclusiveScan(
            SmemStorage     &smem_storage,                  ///< [in] Shared reference to opaque SmemStorage layout
            T               input,                          ///< [in] Calling thread's input item
            T               &output,                        ///< [out] Calling thread's output item (may be aliased to \p input)
            ScanOp          scan_op,                        ///< [in] Binary scan operator
            T               &block_aggregate,               ///< [out] <b>[<em>thread</em><sub>0</sub> only]</b> Threadblock-wide aggregate reduction of input items (exclusive of the \p block_prefix_op value)
            BlockPrefixOp   &block_prefix_op)               ///< [in-out] <b>[<em>thread</em><sub>0</sub> only]</b> Call-back functor for specifying a threadblock-wide prefix to be applied to all inputs.
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
                T *placement_ptr = BlockRakingLayout::PlacementPtr(smem_storage.raking_grid);
                *placement_ptr = input;

                __syncthreads();

                // Reduce parallelism down to just raking threads
                if (threadIdx.x < RAKING_THREADS)
                {
                    // Raking upsweep reduction in grid
                    T *raking_ptr = BlockRakingLayout::RakingPtr(smem_storage.raking_grid);
                    T raking_partial = raking_ptr[0];

                    #pragma unroll
                    for (int i = 1; i < RAKING_LENGTH; i++)
                    {
                        if ((BlockRakingLayout::UNGUARDED) || (((threadIdx.x * RAKING_LENGTH) + i) < BLOCK_THREADS))
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


        /// Computes an exclusive threadblock-wide prefix scan using addition (+) as the scan operator.  Each thread contributes one input element.  The inclusive threadblock-wide \p block_aggregate of all inputs is computed for <em>thread</em><sub>0</sub>.
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
                T *placement_ptr = BlockRakingLayout::PlacementPtr(smem_storage.raking_grid);
                *placement_ptr = input;

                __syncthreads();

                // Reduce parallelism down to just raking threads
                if (threadIdx.x < RAKING_THREADS)
                {
                    // Raking upsweep reduction in grid
                    T *raking_ptr = BlockRakingLayout::RakingPtr(smem_storage.raking_grid);
                    T raking_partial = raking_ptr[0];

                    #pragma unroll
                    for (int i = 1; i < RAKING_LENGTH; i++)
                    {
                        if ((BlockRakingLayout::UNGUARDED) || (((threadIdx.x * RAKING_LENGTH) + i) < BLOCK_THREADS))
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


        /// Computes an exclusive threadblock-wide prefix scan using addition (+) as the scan operator.  Each thread contributes one input element.  Instead of using 0 as the threadblock-wide prefix, the call-back functor \p block_prefix_op is invoked to provide the "seed" value that logically prefixes the threadblock's scan inputs.  The inclusive threadblock-wide \p block_aggregate of all inputs is computed for <em>thread</em><sub>0</sub>.
        template <typename BlockPrefixOp>
        static __device__ __forceinline__ void ExclusiveSum(
            SmemStorage     &smem_storage,                  ///< [in] Shared reference to opaque SmemStorage layout
            T               input,                          ///< [in] Calling thread's input item
            T               &output,                        ///< [out] Calling thread's output item (may be aliased to \p input)
            T               &block_aggregate,               ///< [out] <b>[<em>thread</em><sub>0</sub> only]</b> Threadblock-wide aggregate reduction of input items (exclusive of the \p block_prefix_op value)
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
                T *placement_ptr = BlockRakingLayout::PlacementPtr(smem_storage.raking_grid);
                *placement_ptr = input;

                __syncthreads();

                // Reduce parallelism down to just raking threads
                if (threadIdx.x < RAKING_THREADS)
                {
                    // Raking upsweep reduction in grid
                    T *raking_ptr = BlockRakingLayout::RakingPtr(smem_storage.raking_grid);
                    T raking_partial = raking_ptr[0];

                    #pragma unroll
                    for (int i = 1; i < RAKING_LENGTH; i++)
                    {
                        if ((BlockRakingLayout::UNGUARDED) || (((threadIdx.x * RAKING_LENGTH) + i) < BLOCK_THREADS))
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


        /// Computes an inclusive threadblock-wide prefix scan using the specified binary \p scan_op functor.  Each thread contributes one input element.  The inclusive threadblock-wide \p block_aggregate of all inputs is computed for <em>thread</em><sub>0</sub>.
        template <typename ScanOp>
        static __device__ __forceinline__ void InclusiveScan(
            SmemStorage     &smem_storage,                  ///< [in] Shared reference to opaque SmemStorage layout
            T               input,                          ///< [in] Calling thread's input item
            T               &output,                        ///< [out] Calling thread's output item (may be aliased to \p input)
            ScanOp          scan_op,                        ///< [in] Binary scan operator
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
                T *placement_ptr = BlockRakingLayout::PlacementPtr(smem_storage.raking_grid);
                *placement_ptr = input;

                __syncthreads();

                // Reduce parallelism down to just raking threads
                if (threadIdx.x < RAKING_THREADS)
                {
                    // Raking upsweep reduction in grid
                    T *raking_ptr = BlockRakingLayout::RakingPtr(smem_storage.raking_grid);
                    T raking_partial = raking_ptr[0];

                    #pragma unroll
                    for (int i = 1; i < RAKING_LENGTH; i++)
                    {
                        if ((BlockRakingLayout::UNGUARDED) || (((threadIdx.x * RAKING_LENGTH) + i) < BLOCK_THREADS))
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


        /// Computes an inclusive threadblock-wide prefix scan using the specified binary \p scan_op functor.  Each thread contributes one input element.  The call-back functor \p block_prefix_op is invoked to provide the "seed" value that logically prefixes the threadblock's scan inputs.  The inclusive threadblock-wide \p block_aggregate of all inputs is computed for <em>thread</em><sub>0</sub>.
        template <
        typename ScanOp,
        typename BlockPrefixOp>
        static __device__ __forceinline__ void InclusiveScan(
            SmemStorage     &smem_storage,                  ///< [in] Shared reference to opaque SmemStorage layout
            T               input,                          ///< [in] Calling thread's input item
            T               &output,                        ///< [out] Calling thread's output item (may be aliased to \p input)
            ScanOp          scan_op,                        ///< [in] Binary scan operator
            T               &block_aggregate,               ///< [out] <b>[<em>thread</em><sub>0</sub> only]</b> Threadblock-wide aggregate reduction of input items (exclusive of the \p block_prefix_op value)
            BlockPrefixOp   &block_prefix_op)               ///< [in-out] <b>[<em>thread</em><sub>0</sub> only]</b> Call-back functor for specifying a threadblock-wide prefix to be applied to all inputs.
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
                T *placement_ptr = BlockRakingLayout::PlacementPtr(smem_storage.raking_grid);
                *placement_ptr = input;

                __syncthreads();

                // Reduce parallelism down to just raking threads
                if (threadIdx.x < RAKING_THREADS)
                {
                    // Raking upsweep reduction in grid
                    T *raking_ptr = BlockRakingLayout::RakingPtr(smem_storage.raking_grid);
                    T raking_partial = raking_ptr[0];

                    #pragma unroll
                    for (int i = 1; i < RAKING_LENGTH; i++)
                    {
                        if ((BlockRakingLayout::UNGUARDED) || (((threadIdx.x * RAKING_LENGTH) + i) < BLOCK_THREADS))
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


        /// Computes an inclusive threadblock-wide prefix scan using the specified binary \p scan_op functor.  Each thread contributes one input element.  The inclusive threadblock-wide \p block_aggregate of all inputs is computed for <em>thread</em><sub>0</sub>.
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
                T *placement_ptr = BlockRakingLayout::PlacementPtr(smem_storage.raking_grid);
                *placement_ptr = input;

                __syncthreads();

                // Reduce parallelism down to just raking threads
                if (threadIdx.x < RAKING_THREADS)
                {
                    // Raking upsweep reduction in grid
                    T *raking_ptr = BlockRakingLayout::RakingPtr(smem_storage.raking_grid);
                    T raking_partial = raking_ptr[0];

                    #pragma unroll
                    for (int i = 1; i < RAKING_LENGTH; i++)
                    {
                        if ((BlockRakingLayout::UNGUARDED) || (((threadIdx.x * RAKING_LENGTH) + i) < BLOCK_THREADS))
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


        /// Computes an inclusive threadblock-wide prefix scan using the specified binary \p scan_op functor.  Each thread contributes one input element.  Instead of using 0 as the threadblock-wide prefix, the call-back functor \p block_prefix_op is invoked to provide the "seed" value that logically prefixes the threadblock's scan inputs.  The inclusive threadblock-wide \p block_aggregate of all inputs is computed for <em>thread</em><sub>0</sub>.
        template <typename BlockPrefixOp>
        static __device__ __forceinline__ void InclusiveSum(
            SmemStorage     &smem_storage,                  ///< [in] Shared reference to opaque SmemStorage layout
            T               input,                          ///< [in] Calling thread's input item
            T               &output,                        ///< [out] Calling thread's output item (may be aliased to \p input)
            T               &block_aggregate,               ///< [out] <b>[<em>thread</em><sub>0</sub> only]</b> Threadblock-wide aggregate reduction of input items (exclusive of the \p block_prefix_op value)
            BlockPrefixOp   &block_prefix_op)                   ///< [in-out] <b>[<em>thread</em><sub>0</sub> only]</b> Call-back functor for specifying a threadblock-wide prefix to be applied to all inputs.
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
                T *placement_ptr = BlockRakingLayout::PlacementPtr(smem_storage.raking_grid);
                *placement_ptr = input;

                __syncthreads();

                // Reduce parallelism down to just raking threads
                if (threadIdx.x < RAKING_THREADS)
                {
                    // Raking upsweep reduction in grid
                    T *raking_ptr = BlockRakingLayout::RakingPtr(smem_storage.raking_grid);
                    T raking_partial = raking_ptr[0];

                    #pragma unroll
                    for (int i = 1; i < RAKING_LENGTH; i++)
                    {
                        if ((BlockRakingLayout::UNGUARDED) || (((threadIdx.x * RAKING_LENGTH) + i) < BLOCK_THREADS))
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
            WARPS = (BLOCK_THREADS + PtxArchProps::WARP_THREADS - 1) / PtxArchProps::WARP_THREADS,
        };

        ///  Raking warp-scan utility type
        typedef WarpScan<T, WARPS, PtxArchProps::WARP_THREADS> WarpScan;

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
            ScanOp          scan_op,            ///< [in] Binary scan operator
            T               warp_aggregate,     ///< [in] <b>[<em>lane</em><sub>0</sub>s only]</b> Warp-wide aggregate reduction of input items
            T               &block_aggregate)   ///< [out] <b>[<em>thread</em><sub>0</sub> only]</b> threadblock-wide aggregate reduction of input items
        {
            unsigned int warp_id = threadIdx.x / PtxArchProps::WARP_THREADS;
            unsigned int lane_id = threadIdx.x & (PtxArchProps::WARP_THREADS - 1);

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


        /// Computes an exclusive threadblock-wide prefix scan using the specified binary \p scan_op functor.  Each thread contributes one input element.  The inclusive threadblock-wide \p block_aggregate of all inputs is computed for <em>thread</em><sub>0</sub>.
        template <typename ScanOp>
        static __device__ __forceinline__ void ExclusiveScan(
            SmemStorage     &smem_storage,      ///< [in] Shared reference to opaque SmemStorage layout
            T               input,              ///< [in] Calling thread's input items
            T               &output,            ///< [out] Calling thread's output items (may be aliased to \p input)
            const T         &identity,          ///< [in] Identity value
            ScanOp          scan_op,            ///< [in] Binary scan operator
            T               &block_aggregate)   ///< [out] <b>[<em>thread</em><sub>0</sub> only]</b> threadblock-wide aggregate reduction of input items
        {
            T warp_aggregate;       // Valid in lane-0s
            WarpScan::ExclusiveScan(smem_storage.warp_scan, input, output, identity, scan_op, warp_aggregate);

            // Update outputs and block_aggregate with warp-wide aggregates from lane-0s
            PrefixUpdate(smem_storage, output, scan_op, warp_aggregate, block_aggregate);
        }


        /// Computes an exclusive threadblock-wide prefix scan using the specified binary \p scan_op functor.  Each thread contributes one input element.  The call-back functor \p block_prefix_op is invoked to provide the "seed" value that logically prefixes the threadblock's scan inputs.  The inclusive threadblock-wide \p block_aggregate of all inputs is computed for <em>thread</em><sub>0</sub>.
        template <
        typename ScanOp,
        typename BlockPrefixOp>
        static __device__ __forceinline__ void ExclusiveScan(
            SmemStorage     &smem_storage,                  ///< [in] Shared reference to opaque SmemStorage layout
            T               input,                          ///< [in] Calling thread's input item
            T               &output,                        ///< [out] Calling thread's output item (may be aliased to \p input)
            T               identity,                       ///< [in] Identity value
            ScanOp          scan_op,                        ///< [in] Binary scan operator
            T               &block_aggregate,               ///< [out] <b>[<em>thread</em><sub>0</sub> only]</b> Threadblock-wide aggregate reduction of input items (exclusive of the \p block_prefix_op value)
            BlockPrefixOp   &block_prefix_op)               ///< [in-out] <b>[<em>thread</em><sub>0</sub> only]</b> Call-back functor for specifying a threadblock-wide prefix to be applied to all inputs.
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


        /// Computes an exclusive threadblock-wide prefix scan using the specified binary \p scan_op functor.  Each thread contributes one input element.  The inclusive threadblock-wide \p block_aggregate of all inputs is computed for <em>thread</em><sub>0</sub>.  With no identity value, the output computed for <em>thread</em><sub>0</sub> is invalid.
        template <typename ScanOp>
        static __device__ __forceinline__ void ExclusiveScan(
            SmemStorage     &smem_storage,                  ///< [in] Shared reference to opaque SmemStorage layout
            T               input,                          ///< [in] Calling thread's input item
            T               &output,                        ///< [out] Calling thread's output item (may be aliased to \p input)
            ScanOp          scan_op,                        ///< [in] Binary scan operator
            T               &block_aggregate)               ///< [out] <b>[<em>thread</em><sub>0</sub> only]</b> threadblock-wide aggregate reduction of input items
        {
            T warp_aggregate;       // Valid in lane-0s
            WarpScan::ExclusiveScan(smem_storage.warp_scan, input, output, scan_op, warp_aggregate);

            unsigned int warp_id = threadIdx.x / PtxArchProps::WARP_THREADS;
            unsigned int lane_id = threadIdx.x & (PtxArchProps::WARP_THREADS - 1);

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


        /// Computes an exclusive threadblock-wide prefix scan using the specified binary \p scan_op functor.  Each thread contributes one input element.  The call-back functor \p block_prefix_op is invoked to provide the "seed" value that logically prefixes the threadblock's scan inputs.  The inclusive threadblock-wide \p block_aggregate of all inputs is computed for <em>thread</em><sub>0</sub>.
        template <
        typename ScanOp,
        typename BlockPrefixOp>
        static __device__ __forceinline__ void ExclusiveScan(
            SmemStorage     &smem_storage,                  ///< [in] Shared reference to opaque SmemStorage layout
            T               input,                          ///< [in] Calling thread's input item
            T               &output,                        ///< [out] Calling thread's output item (may be aliased to \p input)
            ScanOp          scan_op,                        ///< [in] Binary scan operator
            T               &block_aggregate,               ///< [out] <b>[<em>thread</em><sub>0</sub> only]</b> Threadblock-wide aggregate reduction of input items (exclusive of the \p block_prefix_op value)
            BlockPrefixOp   &block_prefix_op)               ///< [in-out] <b>[<em>thread</em><sub>0</sub> only]</b> Call-back functor for specifying a threadblock-wide prefix to be applied to all inputs.
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


        /// Computes an exclusive threadblock-wide prefix scan using addition (+) as the scan operator.  Each thread contributes one input element.  The inclusive threadblock-wide \p block_aggregate of all inputs is computed for <em>thread</em><sub>0</sub>.
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


        /// Computes an exclusive threadblock-wide prefix scan using addition (+) as the scan operator.  Each thread contributes one input element.  Instead of using 0 as the threadblock-wide prefix, the call-back functor \p block_prefix_op is invoked to provide the "seed" value that logically prefixes the threadblock's scan inputs.  The inclusive threadblock-wide \p block_aggregate of all inputs is computed for <em>thread</em><sub>0</sub>.
        template <typename BlockPrefixOp>
        static __device__ __forceinline__ void ExclusiveSum(
            SmemStorage     &smem_storage,                  ///< [in] Shared reference to opaque SmemStorage layout
            T               input,                          ///< [in] Calling thread's input item
            T               &output,                        ///< [out] Calling thread's output item (may be aliased to \p input)
            T               &block_aggregate,               ///< [out] <b>[<em>thread</em><sub>0</sub> only]</b> Threadblock-wide aggregate reduction of input items (exclusive of the \p block_prefix_op value)
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


        /// Computes an inclusive threadblock-wide prefix scan using the specified binary \p scan_op functor.  Each thread contributes one input element.  The inclusive threadblock-wide \p block_aggregate of all inputs is computed for <em>thread</em><sub>0</sub>.
        template <typename ScanOp>
        static __device__ __forceinline__ void InclusiveScan(
            SmemStorage     &smem_storage,                  ///< [in] Shared reference to opaque SmemStorage layout
            T               input,                          ///< [in] Calling thread's input item
            T               &output,                        ///< [out] Calling thread's output item (may be aliased to \p input)
            ScanOp          scan_op,                        ///< [in] Binary scan operator
            T               &block_aggregate)               ///< [out] <b>[<em>thread</em><sub>0</sub> only]</b> threadblock-wide aggregate reduction of input items
        {
            T warp_aggregate;       // Valid in lane-0s
            WarpScan::InclusiveScan(smem_storage.warp_scan, input, output, scan_op, warp_aggregate);

            // Update outputs and block_aggregate with warp-wide aggregates from lane-0s
            PrefixUpdate(smem_storage, output, scan_op, warp_aggregate, block_aggregate);

        }


        /// Computes an inclusive threadblock-wide prefix scan using the specified binary \p scan_op functor.  Each thread contributes one input element.  The call-back functor \p block_prefix_op is invoked to provide the "seed" value that logically prefixes the threadblock's scan inputs.  The inclusive threadblock-wide \p block_aggregate of all inputs is computed for <em>thread</em><sub>0</sub>.
        template <
        typename ScanOp,
        typename BlockPrefixOp>
        static __device__ __forceinline__ void InclusiveScan(
            SmemStorage     &smem_storage,                  ///< [in] Shared reference to opaque SmemStorage layout
            T               input,                          ///< [in] Calling thread's input item
            T               &output,                        ///< [out] Calling thread's output item (may be aliased to \p input)
            ScanOp          scan_op,                        ///< [in] Binary scan operator
            T               &block_aggregate,               ///< [out] <b>[<em>thread</em><sub>0</sub> only]</b> Threadblock-wide aggregate reduction of input items (exclusive of the \p block_prefix_op value)
            BlockPrefixOp   &block_prefix_op)               ///< [in-out] <b>[<em>thread</em><sub>0</sub> only]</b> Call-back functor for specifying a threadblock-wide prefix to be applied to all inputs.
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


        /// Computes an inclusive threadblock-wide prefix scan using the specified binary \p scan_op functor.  Each thread contributes one input element.  The inclusive threadblock-wide \p block_aggregate of all inputs is computed for <em>thread</em><sub>0</sub>.
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


        /// Computes an inclusive threadblock-wide prefix scan using the specified binary \p scan_op functor.  Each thread contributes one input element.  Instead of using 0 as the threadblock-wide prefix, the call-back functor \p block_prefix_op is invoked to provide the "seed" value that logically prefixes the threadblock's scan inputs.  The inclusive threadblock-wide \p block_aggregate of all inputs is computed for <em>thread</em><sub>0</sub>.
        template <typename BlockPrefixOp>
        static __device__ __forceinline__ void InclusiveSum(
            SmemStorage     &smem_storage,                  ///< [in] Shared reference to opaque SmemStorage layout
            T               input,                          ///< [in] Calling thread's input item
            T               &output,                        ///< [out] Calling thread's output item (may be aliased to \p input)
            T               &block_aggregate,               ///< [out] <b>[<em>thread</em><sub>0</sub> only]</b> Threadblock-wide aggregate reduction of input items (exclusive of the \p block_prefix_op value)
            BlockPrefixOp   &block_prefix_op)               ///< [in-out] <b>[<em>thread</em><sub>0</sub> only]</b> Call-back functor for specifying a threadblock-wide prefix to be applied to all inputs.
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


    #endif // DOXYGEN_SHOULD_SKIP_THIS


    /// Shared memory storage layout type for BlockScan
    typedef typename BlockScanInternal<SAFE_ALGORITHM>::SmemStorage _SmemStorage;

public:

    /// \smemstorage{BlockScan}
    typedef _SmemStorage SmemStorage;



    /******************************************************************//**
     * \name Exclusive prefix scans
     *********************************************************************/
    //@{


    /**
     * \brief Computes an exclusive threadblock-wide prefix scan using the specified binary \p scan_op functor.  Each thread contributes one input element.  The inclusive threadblock-wide \p block_aggregate of all inputs is computed for <em>thread</em><sub>0</sub>.
     *
     * The scalar \p block_aggregate is undefined in threads other than <em>thread</em><sub>0</sub>.
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
        ScanOp          scan_op,            ///< [in] Binary scan operator
        T               &block_aggregate)   ///< [out] <b>[<em>thread</em><sub>0</sub> only]</b> threadblock-wide aggregate reduction of input items
    {
        BlockScanInternal<SAFE_ALGORITHM>::ExclusiveScan(smem_storage, input, output, identity, scan_op, block_aggregate);
    }


    /**
     * \brief Computes an exclusive threadblock-wide prefix scan using the specified binary \p scan_op functor.  Each thread contributes an array of consecutive input elements.  The inclusive threadblock-wide \p block_aggregate of all inputs is computed for <em>thread</em><sub>0</sub>.
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
    static __device__ __forceinline__ void ExclusiveScan(
        SmemStorage       &smem_storage,                ///< [in] Shared reference to opaque SmemStorage layout
        T                 (&input)[ITEMS_PER_THREAD],   ///< [in] Calling thread's input items
        T                 (&output)[ITEMS_PER_THREAD],  ///< [out] Calling thread's output items (may be aliased to \p input)
        const T           &identity,                    ///< [in] Identity value
        ScanOp            scan_op,                      ///< [in] Binary scan operator
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
     * \brief Computes an exclusive threadblock-wide prefix scan using the specified binary \p scan_op functor.  Each thread contributes one input element.  The call-back functor \p block_prefix_op is invoked to provide the "seed" value that logically prefixes the threadblock's scan inputs.  The inclusive threadblock-wide \p block_aggregate of all inputs is computed for <em>thread</em><sub>0</sub>.
     *
     * The scalar \p block_aggregate is undefined in threads other than <em>thread</em><sub>0</sub>.

     * The \p block_prefix_op functor must implement a member function <tt>T operator()(T block_aggregate)</tt>.
     * The functor's input parameter \p block_aggregate is the same value also returned by the scan operation.
     * This functor is expected to return a threadblock-wide prefix to be applied to all inputs.  The functor
     * will be invoked by the entire first warp of threads, however the input and output are undefined in threads other
     * than <em>thread</em><sub>0</sub>.  Can be stateful.
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
        ScanOp          scan_op,                        ///< [in] Binary scan operator
        T               &block_aggregate,               ///< [out] <b>[<em>thread</em><sub>0</sub> only]</b> Threadblock-wide aggregate reduction of input items (exclusive of the \p block_prefix_op value)
        BlockPrefixOp   &block_prefix_op)               ///< [in-out] <b>[<em>thread</em><sub>0</sub> only]</b> Call-back functor for specifying a threadblock-wide prefix to be applied to all inputs.
    {
        BlockScanInternal<SAFE_ALGORITHM>::ExclusiveScan(smem_storage, input, output, identity, scan_op, block_aggregate, block_prefix_op);
    }


    /**
     * \brief Computes an exclusive threadblock-wide prefix scan using the specified binary \p scan_op functor.  Each thread contributes an array of consecutive input elements.  The call-back functor \p block_prefix_op is invoked to provide the "seed" value that logically prefixes the threadblock's scan inputs.  The inclusive threadblock-wide \p block_aggregate of all inputs is computed for <em>thread</em><sub>0</sub>.
     *
     * The scalar \p block_aggregate is undefined in threads other than <em>thread</em><sub>0</sub>.

     * The \p block_prefix_op functor must implement a member function <tt>T operator()(T block_aggregate)</tt>.
     * The functor's input parameter \p block_aggregate is the same value also returned by the scan operation.
     * This functor is expected to return a threadblock-wide prefix to be applied to all inputs.  The functor
     * will be invoked by the entire first warp of threads, however the input and output are undefined in threads other
     * than <em>thread</em><sub>0</sub>.  Can be stateful.
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
        ScanOp          scan_op,                        ///< [in] Binary scan operator
        T               &block_aggregate,               ///< [out] <b>[<em>thread</em><sub>0</sub> only]</b> Threadblock-wide aggregate reduction of input items (exclusive of the \p block_prefix_op value)
        BlockPrefixOp   &block_prefix_op)               ///< [in-out] <b>[<em>thread</em><sub>0</sub> only]</b> Call-back functor for specifying a threadblock-wide prefix to be applied to all inputs.
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
        ScanOp          scan_op)                        ///< [in] Binary scan operator
    {
        T block_aggregate;
        BlockScanInternal<SAFE_ALGORITHM>::ExclusiveScan(smem_storage, input, output, identity, scan_op, block_aggregate);
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
        ScanOp            scan_op)                      ///< [in] Binary scan operator
    {
        // Reduce consecutive thread items in registers
        T thread_partial = ThreadReduce(input, scan_op);

        // Exclusive threadblock-scan
        ExclusiveScan(smem_storage, thread_partial, thread_partial, identity, scan_op);

        // Exclusive scan in registers with prefix
        ThreadScanExclusive(input, output, scan_op, thread_partial);
    }


    //@}  end member group
    /******************************************************************//**
     * \name Exclusive prefix scans (without supplied identity)
     *********************************************************************/
    //@{


    /**
     * \brief Computes an exclusive threadblock-wide prefix scan using the specified binary \p scan_op functor.  Each thread contributes one input element.  The inclusive threadblock-wide \p block_aggregate of all inputs is computed for <em>thread</em><sub>0</sub>.  With no identity value, the output computed for <em>thread</em><sub>0</sub> is invalid.
     *
     * The scalar \p block_aggregate is undefined in threads other than <em>thread</em><sub>0</sub>.
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
        ScanOp          scan_op,                        ///< [in] Binary scan operator
        T               &block_aggregate)               ///< [out] <b>[<em>thread</em><sub>0</sub> only]</b> threadblock-wide aggregate reduction of input items
    {
        BlockScanInternal<SAFE_ALGORITHM>::ExclusiveScan(smem_storage, input, output, scan_op, block_aggregate);
    }


    /**
     * \brief Computes an exclusive threadblock-wide prefix scan using the specified binary \p scan_op functor.  Each thread contributes an array of consecutive input elements.  The inclusive threadblock-wide \p block_aggregate of all inputs is computed for <em>thread</em><sub>0</sub>.  With no identity value, the output computed for <em>thread</em><sub>0</sub> is invalid.
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
    typename         ScanOp>
    static __device__ __forceinline__ void ExclusiveScan(
        SmemStorage     &smem_storage,                  ///< [in] Shared reference to opaque SmemStorage layout
        T               (&input)[ITEMS_PER_THREAD],     ///< [in] Calling thread's input items
        T               (&output)[ITEMS_PER_THREAD],    ///< [out] Calling thread's output items (may be aliased to \p input)
        ScanOp          scan_op,                        ///< [in] Binary scan operator
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
     * \brief Computes an exclusive threadblock-wide prefix scan using the specified binary \p scan_op functor.  Each thread contributes one input element.  The call-back functor \p block_prefix_op is invoked to provide the "seed" value that logically prefixes the threadblock's scan inputs.  The inclusive threadblock-wide \p block_aggregate of all inputs is computed for <em>thread</em><sub>0</sub>.
     *
     * The scalar \p block_aggregate is undefined in threads other than <em>thread</em><sub>0</sub>.

     * The \p block_prefix_op functor must implement a member function <tt>T operator()(T block_aggregate)</tt>.
     * The functor's input parameter \p block_aggregate is the same value also returned by the scan operation.
     * This functor is expected to return a threadblock-wide prefix to be applied to all inputs.  The functor
     * will be invoked by the entire first warp of threads, however the input and output are undefined in threads other
     * than <em>thread</em><sub>0</sub>.  Can be stateful.
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
        ScanOp          scan_op,                        ///< [in] Binary scan operator
        T               &block_aggregate,               ///< [out] <b>[<em>thread</em><sub>0</sub> only]</b> Threadblock-wide aggregate reduction of input items (exclusive of the \p block_prefix_op value)
        BlockPrefixOp   &block_prefix_op)               ///< [in-out] <b>[<em>thread</em><sub>0</sub> only]</b> Call-back functor for specifying a threadblock-wide prefix to be applied to all inputs.
    {
        BlockScanInternal<SAFE_ALGORITHM>::ExclusiveScan(smem_storage, input, output, scan_op, block_aggregate, block_prefix_op);
    }


    /**
     * \brief Computes an exclusive threadblock-wide prefix scan using the specified binary \p scan_op functor.  Each thread contributes an array of consecutive input elements.  The call-back functor \p block_prefix_op is invoked to provide the "seed" value that logically prefixes the threadblock's scan inputs.  The inclusive threadblock-wide \p block_aggregate of all inputs is computed for <em>thread</em><sub>0</sub>.
     *
     * The scalar \p block_aggregate is undefined in threads other than <em>thread</em><sub>0</sub>.

     * The \p block_prefix_op functor must implement a member function <tt>T operator()(T block_aggregate)</tt>.
     * The functor's input parameter \p block_aggregate is the same value also returned by the scan operation.
     * This functor is expected to return a threadblock-wide prefix to be applied to all inputs.  The functor
     * will be invoked by the entire first warp of threads, however the input and output are undefined in threads other
     * than <em>thread</em><sub>0</sub>.  Can be stateful.
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
        ScanOp          scan_op,                      ///< [in] Binary scan operator
        T               &block_aggregate,             ///< [out] <b>[<em>thread</em><sub>0</sub> only]</b> Threadblock-wide aggregate reduction of input items (exclusive of the \p block_prefix_op value)
        BlockPrefixOp   &block_prefix_op)             ///< [in-out] <b>[<em>thread</em><sub>0</sub> only]</b> Call-back functor for specifying a threadblock-wide prefix to be applied to all inputs.
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
        ScanOp          scan_op)                        ///< [in] Binary scan operator
    {
        T block_aggregate;
        BlockScanInternal<SAFE_ALGORITHM>::ExclusiveScan(smem_storage, input, output, scan_op, block_aggregate);
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
        ScanOp            scan_op)                      ///< [in] Binary scan operator
    {
        // Reduce consecutive thread items in registers
        T thread_partial = ThreadReduce(input, scan_op);

        // Exclusive threadblock-scan
        ExclusiveScan(smem_storage, thread_partial, thread_partial, scan_op);

        // Exclusive scan in registers with prefix
        ThreadScanExclusive(input, output, scan_op, thread_partial, (threadIdx.x != 0));
    }


    //@}  end member group
    /******************************************************************//**
     * \name Exclusive prefix sums
     *********************************************************************/
    //@{


    /**
     * \brief Computes an exclusive threadblock-wide prefix scan using addition (+) as the scan operator.  Each thread contributes one input element.  The inclusive threadblock-wide \p block_aggregate of all inputs is computed for <em>thread</em><sub>0</sub>.
     *
     * The scalar \p block_aggregate is undefined in threads other than <em>thread</em><sub>0</sub>.
     *
     * \smemreuse
     */
    static __device__ __forceinline__ void ExclusiveSum(
        SmemStorage     &smem_storage,                  ///< [in] Shared reference to opaque SmemStorage layout
        T               input,                          ///< [in] Calling thread's input item
        T               &output,                        ///< [out] Calling thread's output item (may be aliased to \p input)
        T               &block_aggregate)               ///< [out] <b>[<em>thread</em><sub>0</sub> only]</b> threadblock-wide aggregate reduction of input items
    {
        BlockScanInternal<SAFE_ALGORITHM>::ExclusiveSum(smem_storage, input, output, block_aggregate);
    }


    /**
     * \brief Computes an exclusive threadblock-wide prefix scan using addition (+) as the scan operator.  Each thread contributes an array of consecutive input elements.  The inclusive threadblock-wide \p block_aggregate of all inputs is computed for <em>thread</em><sub>0</sub>.
     *
     * The scalar \p block_aggregate is undefined in threads other than <em>thread</em><sub>0</sub>.
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
     * \brief Computes an exclusive threadblock-wide prefix scan using addition (+) as the scan operator.  Each thread contributes one input element.  Instead of using 0 as the threadblock-wide prefix, the call-back functor \p block_prefix_op is invoked to provide the "seed" value that logically prefixes the threadblock's scan inputs.  The inclusive threadblock-wide \p block_aggregate of all inputs is computed for <em>thread</em><sub>0</sub>.
     *
     * The scalar \p block_aggregate is undefined in threads other than <em>thread</em><sub>0</sub>.

     * The \p block_prefix_op functor must implement a member function <tt>T operator()(T block_aggregate)</tt>.
     * The functor's input parameter \p block_aggregate is the same value also returned by the scan operation.
     * This functor is expected to return a threadblock-wide prefix to be applied to all inputs.  The functor
     * will be invoked by the entire first warp of threads, however the input and output are undefined in threads other
     * than <em>thread</em><sub>0</sub>.  Can be stateful.
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
        T               &block_aggregate,               ///< [out] <b>[<em>thread</em><sub>0</sub> only]</b> Threadblock-wide aggregate reduction of input items (exclusive of the \p block_prefix_op value)
        BlockPrefixOp   &block_prefix_op)               ///< [in-out] <b>[<em>thread</em><sub>0</sub> only]</b> Call-back functor of the model <em>T block_prefix_op(T block_aggregate)</em> to be run <em>thread</em><sub>0</sub> for providing the operation with a threadblock-wide prefix value to seed the scan with.  Can be stateful.
    {
        BlockScanInternal<SAFE_ALGORITHM>::ExclusiveSum(smem_storage, input, output, block_aggregate, block_prefix_op);
    }


    /**
     * \brief Computes an exclusive threadblock-wide prefix scan using addition (+) as the scan operator.  Each thread contributes an array of consecutive input elements.  Instead of using 0 as the threadblock-wide prefix, the call-back functor \p block_prefix_op is invoked to provide the "seed" value that logically prefixes the threadblock's scan inputs.  The inclusive threadblock-wide \p block_aggregate of all inputs is computed for <em>thread</em><sub>0</sub>.
     *
     * The scalar \p block_aggregate is undefined in threads other than <em>thread</em><sub>0</sub>.

     * The \p block_prefix_op functor must implement a member function <tt>T operator()(T block_aggregate)</tt>.
     * The functor's input parameter \p block_aggregate is the same value also returned by the scan operation.
     * This functor is expected to return a threadblock-wide prefix to be applied to all inputs.  The functor
     * will be invoked by the entire first warp of threads, however the input and output are undefined in threads other
     * than <em>thread</em><sub>0</sub>.  Can be stateful.
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
        T                 &block_aggregate,             ///< [out] <b>[<em>thread</em><sub>0</sub> only]</b> Threadblock-wide aggregate reduction of input items (exclusive of the \p block_prefix_op value)
        BlockPrefixOp     &block_prefix_op)             ///< [in-out] <b>[<em>thread</em><sub>0</sub> only]</b> Call-back functor for specifying a threadblock-wide prefix to be applied to all inputs.
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
        BlockScanInternal<SAFE_ALGORITHM>::ExclusiveSum(smem_storage, input, output, block_aggregate);
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


    //@}  end member group
    /******************************************************************//**
     * \name Inclusive prefix scans
     *********************************************************************/
    //@{


    /**
     * \brief Computes an inclusive threadblock-wide prefix scan using the specified binary \p scan_op functor.  Each thread contributes one input element.  The inclusive threadblock-wide \p block_aggregate of all inputs is computed for <em>thread</em><sub>0</sub>.
     *
     * The scalar \p block_aggregate is undefined in threads other than <em>thread</em><sub>0</sub>.
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
        ScanOp          scan_op,                        ///< [in] Binary scan operator
        T               &block_aggregate)               ///< [out] <b>[<em>thread</em><sub>0</sub> only]</b> threadblock-wide aggregate reduction of input items
    {
        BlockScanInternal<SAFE_ALGORITHM>::InclusiveScan(smem_storage, input, output, scan_op, block_aggregate);
    }


    /**
     * \brief Computes an inclusive threadblock-wide prefix scan using the specified binary \p scan_op functor.  Each thread contributes an array of consecutive input elements.  The inclusive threadblock-wide \p block_aggregate of all inputs is computed for <em>thread</em><sub>0</sub>.
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
    typename         ScanOp>
    static __device__ __forceinline__ void InclusiveScan(
        SmemStorage     &smem_storage,                  ///< [in] Shared reference to opaque SmemStorage layout
        T               (&input)[ITEMS_PER_THREAD],     ///< [in] Calling thread's input items
        T               (&output)[ITEMS_PER_THREAD],    ///< [out] Calling thread's output items (may be aliased to \p input)
        ScanOp          scan_op,                        ///< [in] Binary scan operator
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
     * \brief Computes an inclusive threadblock-wide prefix scan using the specified binary \p scan_op functor.  Each thread contributes one input element.  The call-back functor \p block_prefix_op is invoked to provide the "seed" value that logically prefixes the threadblock's scan inputs.  The inclusive threadblock-wide \p block_aggregate of all inputs is computed for <em>thread</em><sub>0</sub>.
     *
     * The scalar \p block_aggregate is undefined in threads other than <em>thread</em><sub>0</sub>.

     * The \p block_prefix_op functor must implement a member function <tt>T operator()(T block_aggregate)</tt>.
     * The functor's input parameter \p block_aggregate is the same value also returned by the scan operation.
     * This functor is expected to return a threadblock-wide prefix to be applied to all inputs.  The functor
     * will be invoked by the entire first warp of threads, however the input and output are undefined in threads other
     * than <em>thread</em><sub>0</sub>.  Can be stateful.
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
        ScanOp          scan_op,                        ///< [in] Binary scan operator
        T               &block_aggregate,               ///< [out] <b>[<em>thread</em><sub>0</sub> only]</b> Threadblock-wide aggregate reduction of input items (exclusive of the \p block_prefix_op value)
        BlockPrefixOp   &block_prefix_op)               ///< [in-out] <b>[<em>thread</em><sub>0</sub> only]</b> Call-back functor for specifying a threadblock-wide prefix to be applied to all inputs.
    {
        BlockScanInternal<SAFE_ALGORITHM>::InclusiveScan(smem_storage, input, output, scan_op, block_aggregate, block_prefix_op);
    }


    /**
     * \brief Computes an inclusive threadblock-wide prefix scan using the specified binary \p scan_op functor.  Each thread contributes an array of consecutive input elements.  The call-back functor \p block_prefix_op is invoked to provide the "seed" value that logically prefixes the threadblock's scan inputs.  The inclusive threadblock-wide \p block_aggregate of all inputs is computed for <em>thread</em><sub>0</sub>.
     *
     * The scalar \p block_aggregate is undefined in threads other than <em>thread</em><sub>0</sub>.

     * The \p block_prefix_op functor must implement a member function <tt>T operator()(T block_aggregate)</tt>.
     * The functor's input parameter \p block_aggregate is the same value also returned by the scan operation.
     * This functor is expected to return a threadblock-wide prefix to be applied to all inputs.  The functor
     * will be invoked by the entire first warp of threads, however the input and output are undefined in threads other
     * than <em>thread</em><sub>0</sub>.  Can be stateful.
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
        ScanOp          scan_op,                        ///< [in] Binary scan operator
        T               &block_aggregate,               ///< [out] <b>[<em>thread</em><sub>0</sub> only]</b> Threadblock-wide aggregate reduction of input items (exclusive of the \p block_prefix_op value)
        BlockPrefixOp   &block_prefix_op)               ///< [in-out] <b>[<em>thread</em><sub>0</sub> only]</b> Call-back functor for specifying a threadblock-wide prefix to be applied to all inputs.
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
        ScanOp          scan_op)                        ///< [in] Binary scan operator
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
        ScanOp          scan_op)                        ///< [in] Binary scan operator
    {
        // Reduce consecutive thread items in registers
        T thread_partial = ThreadReduce(input, scan_op);

        // Exclusive threadblock-scan
        ExclusiveScan(smem_storage, thread_partial, thread_partial, scan_op);

        // Inclusive scan in registers with prefix
        ThreadScanInclusive(input, output, scan_op, thread_partial, (threadIdx.x != 0));
    }


    //@}  end member group
    /******************************************************************//**
     * \name Inclusive prefix sums
     *********************************************************************/
    //@{


    /**
     * \brief Computes an inclusive threadblock-wide prefix scan using addition (+) as the scan operator.  Each thread contributes one input element.  The inclusive threadblock-wide \p block_aggregate of all inputs is computed for <em>thread</em><sub>0</sub>.
     *
     * The scalar \p block_aggregate is undefined in threads other than <em>thread</em><sub>0</sub>.
     *
     * \smemreuse
     */
    static __device__ __forceinline__ void InclusiveSum(
        SmemStorage     &smem_storage,                  ///< [in] Shared reference to opaque SmemStorage layout
        T               input,                          ///< [in] Calling thread's input item
        T               &output,                        ///< [out] Calling thread's output item (may be aliased to \p input)
        T               &block_aggregate)               ///< [out] <b>[<em>thread</em><sub>0</sub> only]</b> threadblock-wide aggregate reduction of input items
    {
        BlockScanInternal<SAFE_ALGORITHM>::InclusiveSum(smem_storage, input, output, block_aggregate);
    }


    /**
     * \brief Computes an inclusive threadblock-wide prefix scan using addition (+) as the scan operator.  Each thread contributes an array of consecutive input elements.  The inclusive threadblock-wide \p block_aggregate of all inputs is computed for <em>thread</em><sub>0</sub>.
     *
     * The scalar \p block_aggregate is undefined in threads other than <em>thread</em><sub>0</sub>.
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
     * \brief Computes an inclusive threadblock-wide prefix scan using addition (+) as the scan operator.  Each thread contributes one input element.  Instead of using 0 as the threadblock-wide prefix, the call-back functor \p block_prefix_op is invoked to provide the "seed" value that logically prefixes the threadblock's scan inputs.  The inclusive threadblock-wide \p block_aggregate of all inputs is computed for <em>thread</em><sub>0</sub>.
     *
     * The scalar \p block_aggregate is undefined in threads other than <em>thread</em><sub>0</sub>.

     * The \p block_prefix_op functor must implement a member function <tt>T operator()(T block_aggregate)</tt>.
     * The functor's input parameter \p block_aggregate is the same value also returned by the scan operation.
     * This functor is expected to return a threadblock-wide prefix to be applied to all inputs.  The functor
     * will be invoked by the entire first warp of threads, however the input and output are undefined in threads other
     * than <em>thread</em><sub>0</sub>.  Can be stateful.
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
        T               &block_aggregate,               ///< [out] <b>[<em>thread</em><sub>0</sub> only]</b> Threadblock-wide aggregate reduction of input items (exclusive of the \p block_prefix_op value)
        BlockPrefixOp   &block_prefix_op)               ///< [in-out] <b>[<em>thread</em><sub>0</sub> only]</b> Call-back functor for specifying a threadblock-wide prefix to be applied to all inputs.
    {
        BlockScanInternal<SAFE_ALGORITHM>::InclusiveSum(smem_storage, input, output, block_aggregate, block_prefix_op);
    }


    /**
     * \brief Computes an inclusive threadblock-wide prefix scan using addition (+) as the scan operator.  Each thread contributes an array of consecutive input elements.  Instead of using 0 as the threadblock-wide prefix, the call-back functor \p block_prefix_op is invoked to provide the "seed" value that logically prefixes the threadblock's scan inputs.  The inclusive threadblock-wide \p block_aggregate of all inputs is computed for <em>thread</em><sub>0</sub>.
     *
     * The scalar \p block_aggregate is undefined in threads other than <em>thread</em><sub>0</sub>.

     * The \p block_prefix_op functor must implement a member function <tt>T operator()(T block_aggregate)</tt>.
     * The functor's input parameter \p block_aggregate is the same value also returned by the scan operation.
     * This functor is expected to return a threadblock-wide prefix to be applied to all inputs.  The functor
     * will be invoked by the entire first warp of threads, however the input and output are undefined in threads other
     * than <em>thread</em><sub>0</sub>.  Can be stateful.
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
        T               &block_aggregate,               ///< [out] <b>[<em>thread</em><sub>0</sub> only]</b> Threadblock-wide aggregate reduction of input items (exclusive of the \p block_prefix_op value)
        BlockPrefixOp   &block_prefix_op)               ///< [in-out] <b>[<em>thread</em><sub>0</sub> only]</b> Call-back functor for specifying a threadblock-wide prefix to be applied to all inputs.
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
        BlockScanInternal<SAFE_ALGORITHM>::InclusiveSum(smem_storage, input, output, block_aggregate);
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

    //@}  end member group        // Inclusive prefix sums

};

/** @} */       // BlockModule

}               // CUB namespace
CUB_NS_POSTFIX  // Optional outer namespace(s)

