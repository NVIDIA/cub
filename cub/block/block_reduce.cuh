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
 * cub::BlockReduce provides variants of parallel reduction across a CUDA threadblock
 */

#pragma once

#include "../block/block_raking_layout.cuh"
#include "../warp/warp_reduce.cuh"
#include "../util_arch.cuh"
#include "../util_type.cuh"
#include "../thread/thread_operators.cuh"
#include "../thread/thread_reduce.cuh"
#include "../thread/thread_load.cuh"
#include "../thread/thread_store.cuh"
#include "../util_namespace.cuh"

/// Optional outer namespace(s)
CUB_NS_PREFIX

/// CUB namespace
namespace cub {



/******************************************************************************
 * Algorithmic variants
 ******************************************************************************/

/**
 * BlockReduceAlgorithm enumerates alternative algorithms for parallel
 * reduction across a CUDA threadblock.
 */
enum BlockReduceAlgorithm
{

    /**
     * \par Overview
     * An efficient "raking" reduction algorithm.  Execution is comprised of
     * three phases:
     * -# Upsweep sequential reduction in registers (if threads contribute more
     *    than one input each).  Each thread then places the partial reduction
     *    of its item(s) into shared memory.
     * -# Upsweep sequential reduction in shared memory.  Threads within a
     *    single warp rake across segments of shared partial reductions.
     * -# A warp-synchronous Kogge-Stone style reduction within the raking warp.
     *
     * \par
     * \image html block_reduce.png
     * <div class="centercaption">\p BLOCK_REDUCE_RAKING data flow for a hypothetical 16-thread threadblock and 4-thread raking warp.</div>
     *
     * \par Performance Considerations
     * - Although this variant may suffer longer turnaround latencies when the
     *   GPU is under-occupied, it can often provide higher overall throughput
     *   across the GPU when suitably occupied.
     */
    BLOCK_REDUCE_RAKING,


    /**
     * \par Overview
     * A quick "tiled warp-reductions" reduction algorithm.  Execution is
     * comprised of four phases:
     * -# Upsweep sequential reduction in registers (if threads contribute more
     *    than one input each).  Each thread then places the partial reduction
     *    of its item(s) into shared memory.
     * -# Compute a shallow, but inefficient warp-synchronous Kogge-Stone style
     *    reduction within each warp.
     * -# A propagation phase where the warp reduction outputs in each warp are
     *    updated with the aggregate from each preceding warp.
     *
     * \par
     * \image html block_scan_warpscans.png
     * <div class="centercaption">\p BLOCK_REDUCE_WARP_REDUCTIONS data flow for a hypothetical 16-thread threadblock and 4-thread raking warp.</div>
     *
     * \par Performance Considerations
     * - Although this variant may suffer lower overall throughput across the
     *   GPU because due to a heavy reliance on inefficient warp-reductions, it
     *   can often provide lower turnaround latencies when the GPU is
     *   under-occupied.
     */
    BLOCK_REDUCE_WARP_REDUCTIONS,
};


/******************************************************************************
 * Block reduce
 ******************************************************************************/

/**
 * \brief BlockReduce provides variants of parallel reduction across a CUDA threadblock. ![](reduce_logo.png)
 * \ingroup BlockModule
 *
 * \par Overview
 * A <a href="http://en.wikipedia.org/wiki/Reduce_(higher-order_function)"><em>reduction</em></a> (or <em>fold</em>)
 * uses a binary combining operator to compute a single aggregate from a list of input elements.
 *
 * \par
 * For convenience, BlockReduce provides alternative entrypoints that differ by:
 * - Operator (generic reduction <em>vs.</em> summation of numeric types)
 * - Granularity (single <em>vs.</em> multiple data items per thread)
 * - Input validity (full data tile <em>vs.</em> partially-full data tile having some undefined elements)
 *
 * \tparam T                The reduction input/output element type
 * \tparam BLOCK_THREADS    The threadblock size in threads
 * \tparam ALGORITHM        <b>[optional]</b> cub::BlockReduceAlgorithm enumerator specifying the underlying algorithm to use (default = cub::BLOCK_REDUCE_RAKING)
 *
 * \par Algorithm
 * BlockReduce provides a single prefix scan abstraction whose performance behavior can be tuned
 * for different usage scenarios.  BlockReduce can be (optionally) configured to use different algorithms that cater
 * to different latency/throughput needs:
 *   -# <b>cub::BLOCK_REDUCE_RAKING</b>.  An efficient "raking" reduction algorithm. [More...](\ref cub::BlockReduceAlgorithm)
 *   -# <b>cub::BLOCK_REDUCE_WARP_REDUCTIONS</b>.  A quick "tiled warp-reductions" reduction algorithm. [More...](\ref cub::BlockReduceAlgorithm)
 *
 * \par Usage Considerations
 * - Supports non-commutative reduction operators
 * - Supports partially-full threadblocks (i.e., the most-significant thread ranks having undefined values).
 * - Assumes a [<em>blocked arrangement</em>](index.html#sec3sec3) of elements across threads
 * - The threadblock-wide scalar reduction output is only considered valid in <em>thread</em><sub>0</sub>
 * - \smemreuse{BlockReduce::TempStorage}
 *
 * \par Performance Considerations
 * - Very efficient (only one synchronization barrier).
 * - Zero bank conflicts for most types.
 * - Computation is slightly more efficient (i.e., having lower instruction overhead) for:
 *   - \p T is a built-in C++ primitive or CUDA vector type (e.g., \p short, \p int2, \p double, \p float2, etc.)
 *   - \p BLOCK_THREADS is a multiple of the architecture's warp size
 *   - Every thread has a valid input (i.e., full <em>vs.</em> partial-tiles)
 * - See cub::BlockReduceAlgorithm for performance details regarding algorithmic alternatives
 *
 * \par Examples
 * \par
 * <em>Example 1.</em> Perform a simple reduction of 512 integer keys that
 * are partitioned in a blocked arrangement across a 128-thread threadblock (where each thread holds 4 keys).
 * \code
 * #include <cub/cub.cuh>
 *
 * __global__ void SomeKernel(...)
 * {
 *      // Parameterize BlockReduce for 128 threads on type int
 *      typedef cub::BlockReduce<int, 128> BlockReduce;
 *
 *      // Declare shared memory for BlockReduce
 *      __shared__ typename BlockReduce::TempStorage temp_storage;
 *
 *      // A segment of consecutive input items per thread
 *      int data[4];
 *
 *      // Obtain items in blocked order
 *      ...
 *
 *      // Compute the threadblock-wide sum for thread0
 *      int aggregate = BlockReduce::Sum(temp_storage, data);
 *
 *      ...
 * \endcode
 *
 * \par
 * <em>Example 2:</em> Perform a guarded reduction of only the first
 * \p num_items keys that are partitioned in a blocked arrangement
 * across \p BLOCK_THREADS threads.
 * \code
 * #include <cub/cub.cuh>
 *
 * template <int BLOCK_THREADS>
 * __global__ void SomeKernel(..., int num_items)
 * {
 *      // Parameterize BlockReduce on type int
 *      typedef cub::BlockReduce<int, BLOCK_THREADS> BlockReduce;
 *
 *      // Declare shared memory for BlockReduce
 *      __shared__ typename BlockReduce::TempStorage temp_storage;
 *
 *      // Guarded load of input item
 *      int data;
 *      if (threadIdx.x < num_items) data = ...;
 *
 *      // Compute the threadblock-wide sum of valid elements in thread0
 *      int aggregate = BlockReduce::Sum(temp_storage, data, num_items);
 *
 *      ...
 * \endcode
 *
 */
template <
    typename                T,
    int                     BLOCK_THREADS,
    BlockReduceAlgorithm    ALGORITHM = BLOCK_REDUCE_RAKING>
class BlockReduce
{
private:

    #ifndef DOXYGEN_SHOULD_SKIP_THIS    // Do not document

    /******************************************************************************
     * Algorithmic variants
     ******************************************************************************/

    /**
     * BLOCK_REDUCE_RAKING algorithmic variant
     */
    template <BlockReduceAlgorithm _ALGORITHM, int DUMMY = 0>
    struct BlockReduceInternal
    {
        /// Layout type for padded threadblock raking grid
        typedef BlockRakingLayout<T, BLOCK_THREADS, 1> BlockRakingLayout;

        ///  WarpReduce utility type
        typedef typename WarpReduce<T, 1, BlockRakingLayout::RAKING_THREADS>::InternalWarpReduce WarpReduce;

        /// Constants
        enum
        {
            /// Number of raking threads
            RAKING_THREADS = BlockRakingLayout::RAKING_THREADS,

            /// Number of raking elements per warp synchronous raking thread
            SEGMENT_LENGTH = BlockRakingLayout::SEGMENT_LENGTH,

            /// Cooperative work can be entirely warp synchronous
            WARP_SYNCHRONOUS = (RAKING_THREADS == BLOCK_THREADS),

            /// Whether or not warp-synchronous reduction should be unguarded (i.e., the warp-reduction elements is a power of two
            WARP_SYNCHRONOUS_UNGUARDED = ((RAKING_THREADS & (RAKING_THREADS - 1)) == 0),

            /// Whether or not accesses into smem are unguarded
            RAKING_UNGUARDED = BlockRakingLayout::UNGUARDED,

        };


        /// Shared memory storage layout type
        struct TempStorage
        {
            typename WarpReduce::TempStorage            warp_storage;        ///< Storage for warp-synchronous reduction
            typename BlockRakingLayout::TempStorage     raking_grid;         ///< Padded threadblock raking grid
        };


        // Thread fields
        TempStorage &temp_storage;
        int linear_tid;


        /// Constructor
        __device__ __forceinline__ BlockReduceInternal(
            TempStorage &temp_storage,
            int linear_tid)
        :
            temp_storage(temp_storage),
            linear_tid(linear_tid)
        {}


        /// Computes a threadblock-wide reduction using addition (+) as the reduction operator. The first num_valid threads each contribute one reduction partial.  The return value is only valid for thread<sub>0</sub>.
        template <bool FULL_TILE>
        __device__ __forceinline__ T Sum(
            T                   partial,            ///< [in] Calling thread's input partial reductions
            int                 num_valid)          ///< [in] Number of valid elements (may be less than BLOCK_THREADS)
        {
            cub::Sum<T> reduction_op;

            if (WARP_SYNCHRONOUS)
            {
                // Short-circuit directly to warp synchronous reduction (unguarded if active threads is a power-of-two)
                partial = WarpReduce(temp_storage.warp_storage, 0, linear_tid).template Sum<FULL_TILE, SEGMENT_LENGTH>(
                    partial,
                    num_valid);
            }
            else
            {
                // Place partial into shared memory grid.
                *BlockRakingLayout::PlacementPtr(temp_storage.raking_grid) = partial;

                __syncthreads();

                // Reduce parallelism to one warp
                if (threadIdx.x < RAKING_THREADS)
                {
                    // Raking reduction in grid
                    T *raking_segment = BlockRakingLayout::RakingPtr(temp_storage.raking_grid, linear_tid);
                    partial = raking_segment[0];

                    #pragma unroll
                    for (int ITEM = 1; ITEM < SEGMENT_LENGTH; ITEM++)
                    {
                        // Update partial if addend is in range
                        if ((FULL_TILE && RAKING_UNGUARDED) || ((threadIdx.x * SEGMENT_LENGTH) + ITEM < num_valid))
                        {
                            partial = reduction_op(partial, raking_segment[ITEM]);
                        }
                    }

                    partial = WarpReduce(temp_storage.warp_storage, 0, linear_tid).template Sum<FULL_TILE && RAKING_UNGUARDED, SEGMENT_LENGTH>(
                        partial,
                        num_valid);
                }
            }

            return partial;
        }


        /// Computes a threadblock-wide reduction using the specified reduction operator. The first num_valid threads each contribute one reduction partial.  The return value is only valid for thread<sub>0</sub>.
        template <
            bool                FULL_TILE,
            typename            ReductionOp>
        __device__ __forceinline__ T Reduce(
            T                   partial,            ///< [in] Calling thread's input partial reductions
            int                 num_valid,          ///< [in] Number of valid elements (may be less than BLOCK_THREADS)
            ReductionOp         reduction_op)       ///< [in] Binary reduction operator
        {
            if (WARP_SYNCHRONOUS)
            {
                // Short-circuit directly to warp synchronous reduction (unguarded if active threads is a power-of-two)
                partial = WarpReduce(temp_storage.warp_storage, 0, linear_tid).template Reduce<FULL_TILE, SEGMENT_LENGTH>(
                    partial,
                    num_valid,
                    reduction_op);
            }
            else
            {
                // Place partial into shared memory grid.
                *BlockRakingLayout::PlacementPtr(temp_storage.raking_grid) = partial;

                __syncthreads();

                // Reduce parallelism to one warp
                if (threadIdx.x < RAKING_THREADS)
                {
                    // Raking reduction in grid
                    T *raking_segment = BlockRakingLayout::RakingPtr(temp_storage.raking_grid, linear_tid);
                    partial = raking_segment[0];

                    #pragma unroll
                    for (int ITEM = 1; ITEM < SEGMENT_LENGTH; ITEM++)
                    {
                        // Update partial if addend is in range
                        if ((FULL_TILE && RAKING_UNGUARDED) || ((threadIdx.x * SEGMENT_LENGTH) + ITEM < num_valid))
                        {
                            partial = reduction_op(partial, raking_segment[ITEM]);
                        }
                    }

                    partial = WarpReduce(temp_storage.warp_storage, 0, linear_tid).template Reduce<FULL_TILE && RAKING_UNGUARDED, SEGMENT_LENGTH>(
                        partial,
                        num_valid,
                        reduction_op);
                }
            }

            return partial;
        }

    };


    /**
     * BLOCK_REDUCE_WARP_REDUCTIONS algorithmic variant
     *
     * Can only be used when BLOCK_THREADS is a multiple of the architecture's warp size
     */
    template <int DUMMY>
    struct BlockReduceInternal<BLOCK_REDUCE_WARP_REDUCTIONS, DUMMY>
    {
        /// Constants
        enum
        {
            /// Number of active warps
            WARPS = (BLOCK_THREADS + PtxArchProps::WARP_THREADS - 1) / PtxArchProps::WARP_THREADS,

            /// The logical warp size for warp reductions
            LOGICAL_WARP_SIZE = CUB_MIN(BLOCK_THREADS, PtxArchProps::WARP_THREADS),

            /// Whether or not the logical warp size evenly divides the threadblock size
            EVEN_WARP_MULTIPLE = (BLOCK_THREADS % LOGICAL_WARP_SIZE == 0)
        };


        ///  WarpReduce utility type
        typedef typename WarpReduce<T, WARPS, LOGICAL_WARP_SIZE>::InternalWarpReduce WarpReduce;


        /// Shared memory storage layout type
        struct TempStorage
        {
            typename WarpReduce::TempStorage    warp_reduce;                ///< Buffer for warp-synchronous scan
            T                                   warp_aggregates[WARPS];     ///< Shared totals from each warp-synchronous scan
            T                                   block_prefix;               ///< Shared prefix for the entire threadblock
        };

        // Thread fields
        TempStorage &temp_storage;
        int linear_tid;
        int warp_id;
        int lane_id;


        /// Constructor
        __device__ __forceinline__ BlockReduceInternal(
            TempStorage &temp_storage,
            int linear_tid)
        :
            temp_storage(temp_storage),
            linear_tid(linear_tid),
            warp_id((BLOCK_THREADS <= PtxArchProps::WARP_THREADS) ?
                0 :
                linear_tid / PtxArchProps::WARP_THREADS),
            lane_id((BLOCK_THREADS <= PtxArchProps::WARP_THREADS) ?
                linear_tid :
                linear_tid % PtxArchProps::WARP_THREADS)
        {}


        /// Returns block-wide aggregate in <em>thread</em><sub>0</sub>.
        template <
            bool                FULL_TILE,
            typename            ReductionOp>
        __device__ __forceinline__ T ApplyWarpAggregates(
            ReductionOp         reduction_op,       ///< [in] Binary scan operator
            T                   warp_aggregate,     ///< [in] <b>[<em>lane</em><sub>0</sub>s only]</b> Warp-wide aggregate reduction of input items
            int                 num_valid)          ///< [in] Number of valid elements (may be less than BLOCK_THREADS)
        {
            // Share lane aggregates
            if (lane_id == 0)
            {
                temp_storage.warp_aggregates[warp_id] = warp_aggregate;
            }

            __syncthreads();

            // Update total aggregate in warp 0, lane 0
            if (threadIdx.x == 0)
            {
                #pragma unroll
                for (int SUCCESSOR_WARP = 1; SUCCESSOR_WARP < WARPS; SUCCESSOR_WARP++)
                {
                    if (FULL_TILE || (SUCCESSOR_WARP * LOGICAL_WARP_SIZE < num_valid))
                    {
                        warp_aggregate = reduction_op(warp_aggregate, temp_storage.warp_aggregates[SUCCESSOR_WARP]);
                    }
                }
            }

            return warp_aggregate;
        }


        /// Computes a threadblock-wide reduction using addition (+) as the reduction operator. The first num_valid threads each contribute one reduction partial.  The return value is only valid for thread<sub>0</sub>.
        template <bool FULL_TILE>
        __device__ __forceinline__ T Sum(
            T                   input,          ///< [in] Calling thread's input partial reductions
            int                 num_valid)      ///< [in] Number of valid elements (may be less than BLOCK_THREADS)
        {
            cub::Sum<T>     reduction_op;
            unsigned int    warp_offset = warp_id * LOGICAL_WARP_SIZE;
            unsigned int    warp_num_valid = (FULL_TILE && EVEN_WARP_MULTIPLE) ?
                                LOGICAL_WARP_SIZE :
                                (warp_offset < num_valid) ?
                                    num_valid - warp_offset :
                                    0;

            // Warp reduction in every warp
            T warp_aggregate = WarpReduce(temp_storage.warp_reduce, warp_id, lane_id).template Sum<(FULL_TILE && EVEN_WARP_MULTIPLE), 1>(
                input,
                warp_num_valid);

            // Update outputs and block_aggregate with warp-wide aggregates from lane-0s
            return ApplyWarpAggregates<FULL_TILE>(reduction_op, warp_aggregate, num_valid);
        }


        /// Computes a threadblock-wide reduction using the specified reduction operator. The first num_valid threads each contribute one reduction partial.  The return value is only valid for thread<sub>0</sub>.
        template <
            bool                FULL_TILE,
            typename            ReductionOp>
        __device__ __forceinline__ T Reduce(
            T                   input,              ///< [in] Calling thread's input partial reductions
            int                 num_valid,          ///< [in] Number of valid elements (may be less than BLOCK_THREADS)
            ReductionOp         reduction_op)       ///< [in] Binary reduction operator
        {
            unsigned int    warp_id = (WARPS == 1) ? 0 : (threadIdx.x / LOGICAL_WARP_SIZE);
            unsigned int    warp_offset = warp_id * LOGICAL_WARP_SIZE;
            unsigned int    warp_num_valid = (FULL_TILE && EVEN_WARP_MULTIPLE) ?
                                LOGICAL_WARP_SIZE :
                                (warp_offset < num_valid) ?
                                    num_valid - warp_offset :
                                    0;

            // Warp reduction in every warp
            T warp_aggregate = WarpReduce(temp_storage.warp_reduce, warp_id, lane_id).template Reduce<(FULL_TILE && EVEN_WARP_MULTIPLE), 1>(
                input,
                warp_num_valid,
                reduction_op);

            // Update outputs and block_aggregate with warp-wide aggregates from lane-0s
            return ApplyWarpAggregates<FULL_TILE>(reduction_op, warp_aggregate, num_valid);
        }

    };


    #endif // DOXYGEN_SHOULD_SKIP_THIS

    /******************************************************************************
     * Type definitions
     ******************************************************************************/

    /// Internal block reduce implementation to use
    typedef BlockReduceInternal<ALGORITHM> InternalBlockReduce;

    /// Shared memory storage layout type for BlockReduce
    typedef typename InternalBlockReduce::TempStorage _TempStorage;


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
    int linear_tid;


public:

    /// \smemstorage{BlockReduce}
    typedef _TempStorage TempStorage;


    /******************************************************************//**
     * \name Collective construction
     *********************************************************************/
    //@{

    /**
     * \brief Collective constructor for 1D thread blocks using a private static allocation of shared memory as temporary storage.  Threads are identified using <tt>threadIdx.x</tt>.
     */
    __device__ __forceinline__ BlockReduce()
    :
        temp_storage(PrivateStorage()),
        linear_tid(threadIdx.x)
    {}


    /**
     * \brief Collective constructor for 1D thread blocks using the specified memory allocation as temporary storage.  Threads are identified using <tt>threadIdx.x</tt>.
     */
    __device__ __forceinline__ BlockReduce(
        TempStorage &temp_storage)             ///< [in] Reference to memory allocation having layout type TempStorage
    :
        temp_storage(temp_storage),
        linear_tid(threadIdx.x)
    {}


    /**
     * \brief Collective constructor using a private static allocation of shared memory as temporary storage.  Threads are identified using the given linear thread identifier
     */
    __device__ __forceinline__ BlockReduce(
        int linear_tid)                        ///< [in] A suitable 1D thread-identifier for the calling thread (e.g., <tt>(threadIdx.y * blockDim.x) + linear_tid</tt> for 2D thread blocks)
    :
        temp_storage(PrivateStorage()),
        linear_tid(linear_tid)
    {}


    /**
     * \brief Collective constructor using the specified memory allocation as temporary storage.  Threads are identified using the given linear thread identifier.
     */
    __device__ __forceinline__ BlockReduce(
        TempStorage &temp_storage,             ///< [in] Reference to memory allocation having layout type TempStorage
        int linear_tid)                        ///< [in] <b>[optional]</b> A suitable 1D thread-identifier for the calling thread (e.g., <tt>(threadIdx.y * blockDim.x) + linear_tid</tt> for 2D thread blocks)
    :
        temp_storage(temp_storage),
        linear_tid(linear_tid)
    {}



    //@}  end member group
    /******************************************************************//**
     * \name Generic reductions
     *********************************************************************/
    //@{


    /**
     * \brief Computes a threadblock-wide reduction for thread<sub>0</sub> using the specified binary reduction functor.  Each thread contributes one input element.
     *
     * The return value is undefined in threads other than thread<sub>0</sub>.
     *
     * \smemreuse
     *
     * \tparam ReductionOp          <b>[inferred]</b> Binary reduction operator type having member <tt>T operator()(const T &a, const T &b)</tt>
     */
    template <typename ReductionOp>
    __device__ __forceinline__ T Reduce(
        T               input,                      ///< [in] Calling thread's input
        ReductionOp     reduction_op)               ///< [in] Binary reduction operator
    {
        return InternalBlockReduce(temp_storage, linear_tid).template Reduce<true>(input, BLOCK_THREADS, reduction_op);
    }


    /**
     * \brief Computes a threadblock-wide reduction for thread<sub>0</sub> using the specified binary reduction functor.  Each thread contributes an array of consecutive input elements.
     *
     * The return value is undefined in threads other than thread<sub>0</sub>.
     *
     * \smemreuse
     *
     * \tparam ITEMS_PER_THREAD     <b>[inferred]</b> The number of consecutive items partitioned onto each thread.
     * \tparam ReductionOp          <b>[inferred]</b> Binary reduction operator type having member <tt>T operator()(const T &a, const T &b)</tt>
     */
    template <
        int ITEMS_PER_THREAD,
        typename ReductionOp>
    __device__ __forceinline__ T Reduce(
        T               (&inputs)[ITEMS_PER_THREAD],    ///< [in] Calling thread's input segment
        ReductionOp     reduction_op)                   ///< [in] Binary reduction operator
    {
        // Reduce partials
        T partial = ThreadReduce(inputs, reduction_op);
        return Reduce(partial, reduction_op);
    }


    /**
     * \brief Computes a threadblock-wide reduction for thread<sub>0</sub> using the specified binary reduction functor.  The first \p num_valid threads each contribute one input element.
     *
     * The return value is undefined in threads other than thread<sub>0</sub>.
     *
     * \smemreuse
     *
     * \tparam ReductionOp          <b>[inferred]</b> Binary reduction operator type having member <tt>T operator()(const T &a, const T &b)</tt>
     */
    template <typename ReductionOp>
    __device__ __forceinline__ T Reduce(
        T                   input,                  ///< [in] Calling thread's input
        ReductionOp         reduction_op,           ///< [in] Binary reduction operator
        int                 num_valid)              ///< [in] Number of threads containing valid elements (may be less than BLOCK_THREADS)
    {
        // Determine if we scan skip bounds checking
        if (num_valid >= BLOCK_THREADS)
        {
            return InternalBlockReduce(temp_storage, linear_tid).template Reduce<true>(input, num_valid, reduction_op);
        }
        else
        {
            return InternalBlockReduce(temp_storage, linear_tid).template Reduce<false>(input, num_valid, reduction_op);
        }
    }


    //@}  end member group
    /******************************************************************//**
     * \name Summation reductions
     *********************************************************************/
    //@{


    /**
     * \brief Computes a threadblock-wide reduction for thread<sub>0</sub> using addition (+) as the reduction operator.  Each thread contributes one input element.
     *
     * The return value is undefined in threads other than thread<sub>0</sub>.
     *
     * \smemreuse
     */
    __device__ __forceinline__ T Sum(
        T               input)                      ///< [in] Calling thread's input
    {
        return InternalBlockReduce(temp_storage, linear_tid).template Sum<true>(input, BLOCK_THREADS);
    }

    /**
     * \brief Computes a threadblock-wide reduction for thread<sub>0</sub> using addition (+) as the reduction operator.  Each thread contributes an array of consecutive input elements.
     *
     * The return value is undefined in threads other than thread<sub>0</sub>.
     *
     * \smemreuse
     *
     * \tparam ITEMS_PER_THREAD     <b>[inferred]</b> The number of consecutive items partitioned onto each thread.
     */
    template <int ITEMS_PER_THREAD>
    __device__ __forceinline__ T Sum(
        T               (&inputs)[ITEMS_PER_THREAD])    ///< [in] Calling thread's input segment
    {
        // Reduce partials
        T partial = ThreadReduce(inputs, cub::Sum<T>());
        return Sum(partial);
    }


    /**
     * \brief Computes a threadblock-wide reduction for thread<sub>0</sub> using addition (+) as the reduction operator.  The first \p num_valid threads each contribute one input element.
     *
     * \smemreuse
     *
     * The return value is undefined in threads other than thread<sub>0</sub>.
     */
    __device__ __forceinline__ T Sum(
        T                   input,                  ///< [in] Calling thread's input
        int                 num_valid)              ///< [in] Number of threads containing valid elements (may be less than BLOCK_THREADS)
    {
        // Determine if we scan skip bounds checking
        if (num_valid >= BLOCK_THREADS)
        {
            return InternalBlockReduce(temp_storage, linear_tid).template Sum<true>(input, num_valid);
        }
        else
        {
            return InternalBlockReduce(temp_storage, linear_tid).template Sum<false>(input, num_valid);
        }
    }


    //@}  end member group
};

}               // CUB namespace
CUB_NS_POSTFIX  // Optional outer namespace(s)

