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
 * cub::BlockByteHisto constructs 256-valued histograms from 8b data partitioned across threads within a CUDA thread block.
 */

#pragma once

#include "../util_arch.cuh"
#include "../block/block_radix_sort.cuh"
#include "../util_namespace.cuh"

/// Optional outer namespace(s)
CUB_NS_PREFIX

/// CUB namespace
namespace cub {


/**
 * BlockByteHistoAlgorithm enumerates alternative algorithms for the parallel
 * construction of 8b histograms.
 */
enum BlockByteHistoAlgorithm
{

    /**
     * \par Overview
     * Sorting followed by differentiation.  Execution is comprised of two phases:
     * -# Sort the data using efficient radix sort
     * -# Look for "runs" of same-valued 8b keys by detecting discontinuities; the run-lengths are histogram bin counts.
     */
    BLOCK_BYTE_HISTO_SORT,


    /**
     * \par Overview
     * Use atomic addition to update byte counts directly
     *
     * \par Usage Considerations
     * BLOCK_BYTE_HISTO_ATOMIC can only be used on version SM120 or later. Otherwise BLOCK_BYTE_HISTO_SORT is used regardless.
     */
    BLOCK_BYTE_HISTO_ATOMIC,
};


/**
 * \addtogroup BlockModule
 * @{
 */

/**
 * \brief BlockByteHisto constructs 256-valued histograms from 8b data partitioned across threads within a CUDA thread block.
 *
 * \par Overview
 * A <a href="http://en.wikipedia.org/wiki/Reduce_(higher-order_function)"><em>reduction</em> (or <em>fold</em>)</a>
 * uses a binary combining operator to compute a single aggregate from a list of input elements.
 *
 * \par
 * For convenience, BlockByteHisto exposes a spectrum of entrypoints that differ by:
 * - Granularity (single <em>vs.</em> multiple data items per thread)
 * - Complete/incremental composition (compute a new histogram vs. update existing histogram data)
 *
 * \tparam BLOCK_THREADS    The threadblock size in threads
 * \tparam ALGORITHM        <b>[optional]</b> cub::BlockByteHistoAlgorithm tuning policy.  Default = cub::BLOCK_BYTE_HISTO_SORT.
 *
 * \par Algorithm
 * BlockByteHisto can be (optionally) configured to use different algorithms:
 *   -# <b>cub::BLOCK_BYTE_HISTO_SORT</b>.  An efficient An efficient "raking" reduction algorithm. [More...](\ref cub::BlockByteHistoAlgorithm)
 *   -# <b>cub::BLOCK_BYTE_HISTO_ATOMIC</b>.  A quick "tiled warp-reductions" reduction algorithm. [More...](\ref cub::BlockByteHistoAlgorithm)
 *
 * \par Usage Considerations
 * - The histogram output can be constructed in shared or global memory
 * - Supports partially-full threadblocks (i.e., the most-significant thread ranks having undefined values).
 * - \smemreuse{BlockByteHisto::SmemStorage}
 *
 * \par Performance Considerations
 * - Computation is slightly more efficient (i.e., having lower instruction overhead) for:
 *   - \p BLOCK_THREADS is a multiple of the architecture's warp size
 *   - Every thread has a valid input (i.e., full <em>vs.</em> partial-tiles)
 * - See cub::BlockByteHistoAlgorithm for performance details regarding algorithmic alternatives
 *
 * \par Examples
 * \par
 * <em>Example 1.</em> Compute a simple 8b histogram in shared memory from 512 byte values that
 * are partitioned across a 128-thread threadblock (where each thread holds 4 values).
 * \code
 * #include <cub/cub.cuh>
 *
 * __global__ void SomeKernel(...)
 * {
 *      // Parameterize BlockByteHisto for 128 threads
 *      typedef cub::BlockByteHisto<128> BlockByteHisto;
 *
 *      // Declare shared memory for BlockByteHisto
 *      __shared__ typename BlockByteHisto::SmemStorage smem_storage;
 *
 *      // Declare shared memory for histogram
 *      __shared__ unsigned char smem_histo[256];
 *
 *      // Input items per thread
 *      unsigned char data[4];
 *
 *      // Obtain items
 *      ...
 *
 *      // Compute the threadblock-wide histogram
 *      BlockByteHisto::Sum(smem_storage, smem_histo, data);
 *
 *      ...
 * \endcode
 *
 * \par
 * <em>Example 2:</em> Composite an incremental round of 8b histogram data onto
 * an existing histogram in global memory.  The composition is guarded: not all
 * threads have valid inputs.
 * \code
 * #include <cub/cub.cuh>
 *
 * template <int BLOCK_THREADS>
 * __global__ void SomeKernel(..., int *d_histogram, int num_items)
 * {
 *      // Parameterize BlockByteHisto
 *      typedef cub::BlockByteHisto<BLOCK_THREADS> BlockByteHisto;
 *
 *      // Declare shared memory for BlockByteHisto
 *      __shared__ typename BlockByteHisto::SmemStorage smem_storage;
 *
 *      // Guarded load of input item
 *      int data;
 *      if (threadIdx.x < num_items) data = ...;
 *
 *      // Compute the threadblock-wide sum of valid elements in thread0
 *      BlockByteHisto::Sum(smem_storage, d_histogram, data, num_items);
 *
 *      ...
 * \endcode
 *
 */
template <
    typename                    T,
    int                         BLOCK_THREADS,
    BlockByteHistoAlgorithm     ALGORITHM = BLOCK_BYTE_HISTO_SORT>
class BlockByteHisto
{
private:

    /******************************************************************************
     * Constants
     ******************************************************************************/

    /**
     * Ensure the template parameterization meets the requirements of the
     * targeted device architecture.  BLOCK_BYTE_HISTO_ATOMIC can only be used
     * on version SM120 or later.  Otherwise BLOCK_BYTE_HISTO_SORT is used
     * regardless.
     */
    static const BlockReduceAlgorithm SAFE_ALGORITHM =
        ((ALGORITHM == BLOCK_BYTE_HISTO_ATOMIC) && (CUB_PTX_ARCH < 120)) ?
            BLOCK_BYTE_HISTO_SORT :
            ALGORITHM;

    #ifndef DOXYGEN_SHOULD_SKIP_THIS    // Do not document


    /******************************************************************************
     * Algorithmic variants
     ******************************************************************************/

    /**
     * BLOCK_BYTE_HISTO_SORT algorithmic variant
     */
    template <BlockByteHistoAlgorithm _ALGORITHM, int DUMMY = 0>
    struct BlockByteHistoInternal
    {
        /// Layout type for padded threadblock raking grid
        typedef BlockRakingLayout<T, BLOCK_THREADS, 1> BlockRakingLayout;

        ///  WarpReduce utility type
        typedef typename WarpReduce<T, 1, BlockRakingLayout::RAKING_THREADS>::Internal WarpReduce;

        /// Constants
        enum
        {
            /// Number of raking threads
            RAKING_THREADS = BlockRakingLayout::RAKING_THREADS,

            /// Number of raking elements per warp synchronous raking thread
            RAKING_LENGTH = BlockRakingLayout::RAKING_LENGTH,

            /// Cooperative work can be entirely warp synchronous
            WARP_SYNCHRONOUS = (RAKING_THREADS == BLOCK_THREADS),

            /// Whether or not warp-synchronous reduction should be unguarded (i.e., the warp-reduction elements is a power of two
            WARP_SYNCHRONOUS_UNGUARDED = ((RAKING_THREADS & (RAKING_THREADS - 1)) == 0),

            /// Whether or not accesses into smem are unguarded
            RAKING_UNGUARDED = BlockRakingLayout::UNGUARDED,

        };


        /// Shared memory storage layout type
        struct SmemStorage
        {
            typename WarpReduce::SmemStorage            warp_storage;        ///< Storage for warp-synchronous reduction
            typename BlockRakingLayout::SmemStorage     raking_grid;         ///< Padded threadblock raking grid
        };


        /// Computes a threadblock-wide reduction using addition (+) as the reduction operator. The first num_valid threads each contribute one reduction partial.  The return value is only valid for thread<sub>0</sub>.
        template <bool FULL_TILE>
        static __device__ __forceinline__ T Sum(
            SmemStorage         &smem_storage,      ///< [in] Shared reference to opaque SmemStorage layout
            T                   partial,            ///< [in] Calling thread's input partial reductions
            const unsigned int  &num_valid)         ///< [in] Number of valid elements (may be less than BLOCK_THREADS)
        {
            cub::Sum<T> reduction_op;

            if (WARP_SYNCHRONOUS)
            {
                // Short-circuit directly to warp synchronous reduction (unguarded if active threads is a power-of-two)
                partial = WarpReduce::Sum<FULL_TILE, RAKING_LENGTH>(
                    smem_storage.warp_storage,
                    partial,
                    num_valid);
            }
            else
            {
                // Place partial into shared memory grid.
                *BlockRakingLayout::PlacementPtr(smem_storage.raking_grid) = partial;

                __syncthreads();

                // Reduce parallelism to one warp
                if (threadIdx.x < RAKING_THREADS)
                {
                    // Raking reduction in grid
                    T *raking_segment = BlockRakingLayout::RakingPtr(smem_storage.raking_grid);
                    partial = raking_segment[0];

                    #pragma unroll
                    for (int ITEM = 1; ITEM < RAKING_LENGTH; ITEM++)
                    {
                        // Update partial if addend is in range
                        if ((FULL_TILE && RAKING_UNGUARDED) || ((threadIdx.x * RAKING_LENGTH) + ITEM < num_valid))
                        {
                            partial = reduction_op(partial, raking_segment[ITEM]);
                        }
                    }

                    partial = WarpReduce::Sum<FULL_TILE && RAKING_UNGUARDED, RAKING_LENGTH>(
                        smem_storage.warp_storage,
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
        static __device__ __forceinline__ T Reduce(
            SmemStorage         &smem_storage,      ///< [in] Shared reference to opaque SmemStorage layout
            T                   partial,            ///< [in] Calling thread's input partial reductions
            const unsigned int  &num_valid,         ///< [in] Number of valid elements (may be less than BLOCK_THREADS)
            ReductionOp         reduction_op)       ///< [in] Binary reduction operator
        {
            if (WARP_SYNCHRONOUS)
            {
                // Short-circuit directly to warp synchronous reduction (unguarded if active threads is a power-of-two)
                partial = WarpReduce::Reduce<FULL_TILE, RAKING_LENGTH>(
                    smem_storage.warp_storage,
                    partial,
                    num_valid,
                    reduction_op);
            }
            else
            {
                // Place partial into shared memory grid.
                *BlockRakingLayout::PlacementPtr(smem_storage.raking_grid) = partial;

                __syncthreads();

                // Reduce parallelism to one warp
                if (threadIdx.x < RAKING_THREADS)
                {
                    // Raking reduction in grid
                    T *raking_segment = BlockRakingLayout::RakingPtr(smem_storage.raking_grid);
                    partial = raking_segment[0];

                    #pragma unroll
                    for (int ITEM = 1; ITEM < RAKING_LENGTH; ITEM++)
                    {
                        // Update partial if addend is in range
                        if ((FULL_TILE && RAKING_UNGUARDED) || ((threadIdx.x * RAKING_LENGTH) + ITEM < num_valid))
                        {
                            partial = reduction_op(partial, raking_segment[ITEM]);
                        }
                    }

                    partial = WarpReduce::Reduce<FULL_TILE && RAKING_UNGUARDED, RAKING_LENGTH>(
                        smem_storage.warp_storage,
                        partial,
                        num_valid,
                        reduction_op);
                }
            }

            return partial;
        }

    };


    /**
     * BLOCK_BYTE_HISTO_ATOMIC algorithmic variant
     */
    template <int DUMMY>
    struct BlockByteHistoInternal<BLOCK_BYTE_HISTO_ATOMIC, DUMMY>
    {
        /// Shared memory storage layout type
        struct SmemStorage {};

        /// Returns block-wide aggregate in <em>thread</em><sub>0</sub>.
        template <int SizeT>
        static __device__ __forceinline__ void Composite(
            SmemStorage         &smem_storage,      ///< [in] Shared reference to opaque SmemStorage layout
            SizeT               *histogram,
            unsigned char       data)
        {
            atomicInc(historgram[data])



            // Warp, thread-lane-IDs
            unsigned int warp_id = (WARPS == 1) ? 0 : (threadIdx.x / PtxArchProps::WARP_THREADS);
            unsigned int lane_id = (WARPS == 1) ? threadIdx.x : (threadIdx.x & (PtxArchProps::WARP_THREADS - 1));

            // Share lane aggregates
            if (lane_id == 0)
            {
                smem_storage.warp_aggregates[warp_id] = warp_aggregate;
            }

            __syncthreads();

            // Update total aggregate in warp 0, lane 0
            if (threadIdx.x == 0)
            {
                #pragma unroll
                for (int SUCCESSOR_WARP = 1; SUCCESSOR_WARP < WARPS; SUCCESSOR_WARP++)
                {
                    if (FULL_TILE || (SUCCESSOR_WARP * PtxArchProps::WARP_THREADS < num_valid))
                    {
                        warp_aggregate = reduction_op(warp_aggregate, smem_storage.warp_aggregates[SUCCESSOR_WARP]);
                    }
                }
            }

            return warp_aggregate;
        }


        /// Computes a threadblock-wide reduction using addition (+) as the reduction operator. The first num_valid threads each contribute one reduction partial.  The return value is only valid for thread<sub>0</sub>.
        template <int FULL_TILE>
        static __device__ __forceinline__ T Sum(
            SmemStorage         &smem_storage,  ///< [in] Shared reference to opaque SmemStorage layout
            T                   input,          ///< [in] Calling thread's input partial reductions
            const unsigned int  &num_valid)     ///< [in] Number of valid elements (may be less than BLOCK_THREADS)
        {
            cub::Sum<T>     reduction_op;
            unsigned int    warp_id = (WARPS == 1) ? 0 : (threadIdx.x / PtxArchProps::WARP_THREADS);
            unsigned int    warp_offset = warp_id * PtxArchProps::WARP_THREADS;
            unsigned int    warp_num_valid = (FULL_TILE) ?
                                PtxArchProps::WARP_THREADS :
                                (warp_offset < num_valid) ?
                                    num_valid - warp_offset :
                                    0;

            // Warp reduction in every warp
            T warp_aggregate = WarpReduce::template Sum<FULL_TILE, 1>(
                smem_storage.warp_reduce,
                input,
                warp_num_valid);

            // Update outputs and block_aggregate with warp-wide aggregates from lane-0s
            return ApplyWarpAggregates<FULL_TILE>(smem_storage, reduction_op, warp_aggregate, num_valid);
        }


        /// Computes a threadblock-wide reduction using the specified reduction operator. The first num_valid threads each contribute one reduction partial.  The return value is only valid for thread<sub>0</sub>.
        template <
            int                 FULL_TILE,
            typename            ReductionOp>
        static __device__ __forceinline__ T Reduce(
            SmemStorage         &smem_storage,      ///< [in] Shared reference to opaque SmemStorage layout
            T                   input,              ///< [in] Calling thread's input partial reductions
            const unsigned int  &num_valid,         ///< [in] Number of valid elements (may be less than BLOCK_THREADS)
            ReductionOp         reduction_op)       ///< [in] Binary reduction operator
        {
            unsigned int    warp_id = (WARPS == 1) ? 0 : (threadIdx.x / PtxArchProps::WARP_THREADS);
            unsigned int    warp_offset = warp_id * PtxArchProps::WARP_THREADS;
            unsigned int    warp_num_valid = (FULL_TILE) ?
                                PtxArchProps::WARP_THREADS :
                                (warp_offset < num_valid) ?
                                    num_valid - warp_offset :
                                    0;

            // Warp reduction in every warp
            T warp_aggregate = WarpReduce::template Reduce<FULL_TILE, 1>(
                smem_storage.warp_reduce,
                input,
                warp_num_valid,
                reduction_op);

            // Update outputs and block_aggregate with warp-wide aggregates from lane-0s
            return ApplyWarpAggregates<FULL_TILE>(smem_storage, reduction_op, warp_aggregate, num_valid);
        }

    };


    #endif // DOXYGEN_SHOULD_SKIP_THIS


    /// Shared memory storage layout type for BlockByteHisto
    typedef typename BlockByteHistoInternal<SAFE_ALGORITHM>::SmemStorage _SmemStorage;


public:

    /// \smemstorage{BlockByteHisto}
    typedef _SmemStorage SmemStorage;


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
    static __device__ __forceinline__ T Reduce(
        SmemStorage     &smem_storage,              ///< [in] Shared reference to opaque SmemStorage layout
        T               input,                      ///< [in] Calling thread's input
        ReductionOp     reduction_op)               ///< [in] Binary reduction operator
    {
        return BlockByteHistoInternal<SAFE_ALGORITHM>::template Reduce<true>(smem_storage, input, BLOCK_THREADS, reduction_op);
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
    static __device__ __forceinline__ T Reduce(
        SmemStorage     &smem_storage,                  ///< [in] Shared reference to opaque SmemStorage layout
        T               (&inputs)[ITEMS_PER_THREAD],    ///< [in] Calling thread's input segment
        ReductionOp     reduction_op)                   ///< [in] Binary reduction operator
    {
        // Reduce partials
        T partial = ThreadReduce(inputs, reduction_op);
        return Reduce(smem_storage, partial, reduction_op);
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
    static __device__ __forceinline__ T Reduce(
        SmemStorage         &smem_storage,          ///< [in] Shared reference to opaque SmemStorage layout
        T                   input,                  ///< [in] Calling thread's input
        ReductionOp         reduction_op,           ///< [in] Binary reduction operator
        const unsigned int  &num_valid)             ///< [in] Number of threads containing valid elements (may be less than BLOCK_THREADS)
    {
        // Determine if we don't need bounds checking
        if (num_valid >= BLOCK_THREADS)
        {
            return BlockByteHistoInternal<SAFE_ALGORITHM>::template Reduce<true>(smem_storage, input, num_valid, reduction_op);
        }
        else
        {
            return BlockByteHistoInternal<SAFE_ALGORITHM>::template Reduce<false>(smem_storage, input, num_valid, reduction_op);
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
    static __device__ __forceinline__ T Sum(
        SmemStorage     &smem_storage,              ///< [in] Shared reference to opaque SmemStorage layout
        T               input)                      ///< [in] Calling thread's input
    {
        return BlockByteHistoInternal<SAFE_ALGORITHM>::template Sum<true>(smem_storage, input, BLOCK_THREADS);
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
    static __device__ __forceinline__ T Sum(
        SmemStorage     &smem_storage,                  ///< [in] Shared reference to opaque SmemStorage layout
        T               (&inputs)[ITEMS_PER_THREAD])    ///< [in] Calling thread's input segment
    {
        // Reduce partials
        T partial = ThreadReduce(inputs, cub::Sum<T>());
        return Sum(smem_storage, partial);
    }


    /**
     * \brief Computes a threadblock-wide reduction for thread<sub>0</sub> using addition (+) as the reduction operator.  The first \p num_valid threads each contribute one input element.
     *
     * \smemreuse
     *
     * The return value is undefined in threads other than thread<sub>0</sub>.
     */
    static __device__ __forceinline__ T Sum(
        SmemStorage         &smem_storage,          ///< [in] Shared reference to opaque SmemStorage layout
        T                   input,                  ///< [in] Calling thread's input
        const unsigned int  &num_valid)             ///< [in] Number of threads containing valid elements (may be less than BLOCK_THREADS)
    {
        // Determine if we don't need bounds checking
        if (num_valid >= BLOCK_THREADS)
        {
            return BlockByteHistoInternal<SAFE_ALGORITHM>::template Sum<true>(smem_storage, input, num_valid);
        }
        else
        {
            return BlockByteHistoInternal<SAFE_ALGORITHM>::template Sum<false>(smem_storage, input, num_valid);
        }
    }


    //@}  end member group
};

/** @} */       // end group BlockModule

}               // CUB namespace
CUB_NS_POSTFIX  // Optional outer namespace(s)

