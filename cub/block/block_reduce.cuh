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

/**
 * \addtogroup BlockModule
 * @{
 */

/**
 * \brief BlockReduce provides variants of parallel reduction across a CUDA threadblock. ![](reduce_logo.png)
 *
 * \par Overview
 * A <a href="http://en.wikipedia.org/wiki/Reduce_(higher-order_function)"><em>reduction</em> (or <em>fold</em>)</a>
 * uses a binary combining operator to compute a single aggregate from a list of input elements.
 *
 * \par
 * For convenience, BlockReduce exposes a spectrum of entrypoints that differ by:
 * - Operator (generic reduction <em>vs.</em> summation for numeric types)
 * - Granularity (single <em>vs.</em> multiple items per thread)
 * - Input (full data tile <em>vs.</em> partially-full data tile having some undefined elements)
 *
 * \tparam T                        The reduction input/output element type
 * \tparam BLOCK_THREADS            The threadblock size in threads
 *
 * \par Algorithm
 * BlockReduce entrypoints have <em>O</em>(<em>n</em>) work complexity and are implemented in three phases:
 * -# Sequential reduction in registers (if threads contribute more than one input each).  Each thread then places the partial reduction of its item(s) into shared memory.
 * -# A single-warp performs a raking upsweep across partial reductions shared each thread in the threadblock.
 * -# A warp-synchronous Kogge-Stone style reduction within the raking warp to produce the total aggregate.
 * \image html block_reduce.png
 * <div class="centercaption">Data flow for a hypothetical 16-thread threadblock and 4-thread raking warp.</div>
 *
 * \par Usage Considerations
 * - Supports non-commutative reduction operators
 * - Supports partially-full threadblocks (i.e., the most-significant thread ranks having undefined values).
 * - Assumes a [<em>blocked arrangement</em>](index.html#sec3sec3) of elements across threads
 * - The threadblock-wide scalar reduction output is only considered valid in <em>thread</em><sub>0</sub>
 * - \smemreuse{BlockReduce::SmemStorage}
 *
 * \par Performance Considerations
 * - Very efficient (only one synchronization barrier).
 * - Zero bank conflicts for most types.
 * - Computation is slightly more efficient (i.e., having lower instruction overhead) for:
 *   - \p T is a built-in C++ primitive or CUDA vector type (e.g., \p short, \p int2, \p double, \p float2, etc.)
 *   - \p BLOCK_THREADS is a multiple of the architecture's warp size
 *   - Every thread has a valid input (i.e., full <em>vs.</em> partial-tiles)
 *
 * \par Examples
 * \par
 * <em>Example 1.</em> Perform a simple reduction of 512 integer keys that
 * are partitioned in a blocked arrangement across a 128-thread threadblock (where each thread holds 4 keys).
 * \code
 * #include <cub.cuh>
 *
 * __global__ void SomeKernel(...)
 * {
 *      // Parameterize BlockReduce for 128 threads on type int
 *      typedef cub::BlockReduce<int, 128> BlockReduce;
 *
 *      // Declare shared memory for BlockReduce
 *      __shared__ typename BlockReduce::SmemStorage smem_storage;
 *
 *      // A segment of consecutive input items per thread
 *      int data[4];
 *
 *      // Obtain items in blocked order
 *      ...
 *
 *      // Compute the threadblock-wide sum for thread0
 *      int aggregate = BlockReduce::Sum(smem_storage, data);
 *
 *      ...
 * \endcode
 *
 * \par
 * <em>Example 2:</em> Perform a guarded reduction of only \p num_items keys that
 * are partitioned in a partially-full blocked arrangement across \p BLOCK_THREADS threads.
 * \code
 * #include <cub.cuh>
 *
 * template <int BLOCK_THREADS>
 * __global__ void SomeKernel(..., int num_items)
 * {
 *      // Parameterize BlockReduce on type int
 *      typedef cub::BlockReduce<int, BLOCK_THREADS> BlockReduce;
 *
 *      // Declare shared memory for BlockReduce
 *      __shared__ typename BlockReduce::SmemStorage smem_storage;
 *
 *      // Guarded load
 *      int data;
 *      if (threadIdx.x < num_items) data = ...;
 *
 *      // Compute the threadblock-wide sum of valid elements in thread0
 *      int aggregate = BlockReduce::Sum(smem_storage, data, num_items);
 *
 *      ...
 * \endcode
 *
 */
template <
    typename    T,
    int         BLOCK_THREADS>
class BlockReduce
{
private:

    //---------------------------------------------------------------------
    // Constants and typedefs
    //---------------------------------------------------------------------

    /// Layout type for padded threadblock raking grid
    typedef BlockRakingLayout<T, BLOCK_THREADS, 1> BlockRakingLayout;

    /// Warp reduction type
    typedef typename WarpReduce<T, 1, BlockRakingLayout::RAKING_THREADS>::Internal WarpReduce;

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
    struct _SmemStorage
    {
        typename WarpReduce::SmemStorage        warp_storage;        ///< Storage for warp-synchronous reduction
        typename BlockRakingLayout::SmemStorage   raking_grid;         ///< Padded threadblock raking grid
    };

public:

    /// \smemstorage{BlockReduce}
    typedef _SmemStorage SmemStorage;

private:

    //---------------------------------------------------------------------
    // Utility methods
    //---------------------------------------------------------------------


    /**
     * Perform a cooperative, threadblock-wide reduction. The first num_valid
     * threads each contribute one reduction partial.
     *
     * The return value is only valid for thread<sub>0</sub> (and is undefined for
     * other threads).
     */
    template <
        bool                FULL_TILE,
        typename            ReductionOp>
    static __device__ __forceinline__ T ReduceInternal(
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

public:


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
        return Reduce(smem_storage, input, reduction_op, BLOCK_THREADS);
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
            return ReduceInternal<true>(smem_storage, input, num_valid, reduction_op);
        }
        else
        {
            return ReduceInternal<false>(smem_storage, input, num_valid, reduction_op);
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
        cub::Sum<T> reduction_op;
        return Reduce(smem_storage, input, reduction_op);
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
        cub::Sum<T> reduction_op;
        return Reduce(smem_storage, inputs, reduction_op);
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
        cub::Sum<T> reduction_op;
        return Reduce(smem_storage, input, reduction_op, num_valid);
    }


    //@}  end member group
};

/** @} */       // end group BlockModule

}               // CUB namespace
CUB_NS_POSTFIX  // Optional outer namespace(s)

