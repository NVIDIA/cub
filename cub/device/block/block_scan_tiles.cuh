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
 * cub::BlockScanTiles implements an abstraction of CUDA thread blocks for
 * participating in device-wide prefix scan.
 */

#pragma once

#include <iterator>

#include "../grid/grid_mapping.cuh"
#include "../grid/grid_even_share.cuh"
#include "../grid/grid_queue.cuh"
#include "../../block/block_load.cuh"
#include "../../block/block_reduce.cuh"
#include "../../util_vector.cuh"
#include "../../util_namespace.cuh"

/// Optional outer namespace(s)
CUB_NS_PREFIX

/// CUB namespace
namespace cub {


/**
 * Tuning policy for BlockReduceTiles
 */
template <
    int                         _BLOCK_THREADS,
    int                         _ITEMS_PER_THREAD,
    int                         _VECTOR_LOAD_LENGTH,
    PtxLoadModifier             _LOAD_MODIFIER,
    GridMappingStrategy         _GRID_MAPPING,
    int                         _OVERSUBSCRIPTION>
struct BlockScanTilesPolicy
{
    enum
    {
        BLOCK_THREADS       = _BLOCK_THREADS,
        ITEMS_PER_THREAD    = _ITEMS_PER_THREAD,
        VECTOR_LOAD_LENGTH  = _VECTOR_LOAD_LENGTH,
        OVERSUBSCRIPTION    = _OVERSUBSCRIPTION,
    };

    static const GridMappingStrategy   GRID_MAPPING       = _GRID_MAPPING;
    static const PtxLoadModifier       LOAD_MODIFIER      = _LOAD_MODIFIER;
};


/**
 * \brief BlockReduceTiles implements an abstraction of CUDA thread blocks for
 * participating in device-wide reduction.
 */
template <
    typename BlockScanTilesPolicy,
    typename InputIterator,
    typename SizeT>
class BlockReduceTiles
{
private:

    //---------------------------------------------------------------------
    // Types and constants
    //---------------------------------------------------------------------

    // Data type of input iterator
    typedef typename std::iterator_traits<InputIterator>::value_type T;

    // Constants
    enum
    {
        BLOCK_THREADS       = BlockScanTilesPolicy::BLOCK_THREADS,
        ITEMS_PER_THREAD    = BlockScanTilesPolicy::ITEMS_PER_THREAD,
        TILE_ITEMS          = BLOCK_THREADS * ITEMS_PER_THREAD,

        // Actual vector load length must evenly divide ITEMS_PER_THREAD
        VECTOR_LOAD_LENGTH  = (ITEMS_PER_THREAD % BlockScanTilesPolicy::VECTOR_LOAD_LENGTH == 0) ?
                                BlockScanTilesPolicy::VECTOR_LOAD_LENGTH :
                                1,

    };

    static const PtxLoadModifier LOAD_MODIFIER      = BlockScanTilesPolicy::LOAD_MODIFIER;

    // Parameterized BlockReduce primitive
    typedef BlockReduce<T, BLOCK_THREADS> BlockReduceT;

    // Shared memory type for this threadblock
    struct _SmemStorage
    {
        SizeT block_offset;                                 // Location where to dequeue input for dynamic operation
        typename BlockReduceT::SmemStorage  reduce;         // Smem needed for cooperative reduction
    };

public:

    /// \smemstorage{BlockReduceTiles}
    typedef _SmemStorage SmemStorage;

private:

    //---------------------------------------------------------------------
    // Utility operations
    //---------------------------------------------------------------------

    /**
     * Process a single, full tile.  Specialized for d_in is an iterator (not a native pointer)
     *
     * Each thread reduces only the values it loads.  If \p FIRST_TILE,
     * this partial reduction is stored into \p thread_aggregate.  Otherwise
     * it is accumulated into \p thread_aggregate.
     */
    template <
        bool FIRST_TILE,
        typename InputIterator,
        typename SizeT,
        typename ReductionOp>
    static __device__ __forceinline__ void ConsumeFullTile(
        SmemStorage             &smem_storage,
        InputIterator           d_in,
        SizeT                   block_offset,
        ReductionOp             &reduction_op,
        T                       &thread_aggregate)
    {
        T items[ITEMS_PER_THREAD];

        BlockLoadDirectStriped(
            d_in + block_offset,
            items);

        T partial = ThreadReduce(items, reduction_op);

        thread_aggregate = (FIRST_TILE) ?
            partial :
            reduction_op(thread_aggregate, partial);
    }


    /**
     * Process a single, full tile.  Specialized for native pointers
     *
     * Each thread reduces only the values it loads.  If \p FIRST_TILE,
     * this partial reduction is stored into \p thread_aggregate.  Otherwise
     * it is accumulated into \p thread_aggregate.
     *
     * Performs a block-wide barrier synchronization
     */
    template <
        bool FIRST_TILE,
        typename SizeT,
        typename ReductionOp>
    static __device__ __forceinline__ void ConsumeFullTile(
        SmemStorage             &smem_storage,
        T                       *d_in,
        SizeT                   block_offset,
        ReductionOp             &reduction_op,
        T                       &thread_aggregate)
    {
        if ((size_t(d_in) & (VECTOR_LOAD_LENGTH - 1)) == 0)
        {
            T items[ITEMS_PER_THREAD];

            typedef typename VectorType<T, VECTOR_LOAD_LENGTH>::Type VectorT;

            // Alias items as an array of VectorT and load it in striped fashion
            BlockLoadDirectStriped(
                reinterpret_cast<VectorT*>(d_in + block_offset),
                reinterpret_cast<VectorT (&)[ITEMS_PER_THREAD / VECTOR_LOAD_LENGTH]>(items));

            // Prevent hoisting
            __syncthreads();

            T partial = ThreadReduce(items, reduction_op);

            thread_aggregate = (FIRST_TILE) ?
                partial :
                reduction_op(thread_aggregate, partial);
        }
        else
        {
            T items[ITEMS_PER_THREAD];

            BlockLoadDirectStriped(
                d_in + block_offset,
                items);

            // Prevent hoisting
            __syncthreads();

            T partial = ThreadReduce(items, reduction_op);

            thread_aggregate = (FIRST_TILE) ?
                partial :
                reduction_op(thread_aggregate, partial);
        }

    }


    /**
     * Process a single, partial tile.
     *
     * Each thread reduces only the values it loads.  If \p FIRST_TILE,
     * this partial reduction is stored into \p thread_aggregate.  Otherwise
     * it is accumulated into \p thread_aggregate.
     */
    template <
        bool FIRST_TILE,
        typename SizeT,
        typename ReductionOp>
    static __device__ __forceinline__ void ConsumePartialTile(
        SmemStorage             &smem_storage,
        InputIterator           d_in,
        SizeT                   block_offset,
        const SizeT             &block_oob,
        ReductionOp             &reduction_op,
        T                       &thread_aggregate)
    {
        SizeT thread_offset = block_offset + threadIdx.x;

        if ((FIRST_TILE) && (thread_offset < block_oob))
        {
            thread_aggregate = ThreadLoad<LOAD_MODIFIER>(d_in + thread_offset);
            thread_offset += BLOCK_THREADS;
        }

        while (thread_offset < block_oob)
        {
            T item = ThreadLoad<LOAD_MODIFIER>(d_in + thread_offset);
            thread_aggregate = reduction_op(thread_aggregate, item);
            thread_offset += BLOCK_THREADS;
        }
    }

public:

    //---------------------------------------------------------------------
    // Interface
    //---------------------------------------------------------------------

    /**
     * \brief Consumes input tiles using an even-share policy, computing a threadblock-wide reduction for thread<sub>0</sub> using the specified binary reduction functor.
     *
     * The return value is undefined in threads other than thread<sub>0</sub>.
     */
    template <typename SizeT, typename ReductionOp>
    static __device__ __forceinline__ T ProcessTilesEvenShare(
        SmemStorage             &smem_storage,
        InputIterator           d_in,
        SizeT                   block_offset,
        const SizeT             &block_oob,
        ReductionOp             &reduction_op)
    {
        if (block_offset + TILE_ITEMS <= block_oob)
        {
            // We have at least one full tile to consume
            T thread_aggregate;
            ConsumeFullTile<true>(smem_storage, d_in, block_offset, reduction_op, thread_aggregate);
            block_offset += TILE_ITEMS;

            // Consume any other full tiles
            while (block_offset + TILE_ITEMS <= block_oob)
            {
                ConsumeFullTile<false>(smem_storage, d_in, block_offset, reduction_op, thread_aggregate);
                block_offset += TILE_ITEMS;
            }

            // Consume any remaining input
            ConsumePartialTile<false>(smem_storage, d_in, block_offset, block_oob, reduction_op, thread_aggregate);

            // Compute the block-wide reduction (every thread has a valid input)
            return BlockReduceT::Reduce(smem_storage.reduce, thread_aggregate, reduction_op);
        }
        else
        {
            // We have less than a full tile to consume
            T thread_aggregate;
            ConsumePartialTile<true>(smem_storage, d_in, block_offset, block_oob, reduction_op, thread_aggregate);

            // Compute the block-wide reduction  (up to block_items threads have valid inputs)
            SizeT block_items = block_oob - block_offset;
            return BlockReduceT::Reduce(smem_storage.reduce, thread_aggregate, reduction_op, block_items);
        }
    }


    /**
     * \brief Consumes input tiles using a dynamic queue policy, computing a threadblock-wide reduction for thread<sub>0</sub> using the specified binary reduction functor.
     *
     * The return value is undefined in threads other than thread<sub>0</sub>.
     */
    template <typename SizeT, typename ReductionOp>
    static __device__ __forceinline__ T ProcessTilesDynamic(
        SmemStorage             &smem_storage,
        InputIterator           d_in,
        SizeT                   num_items,
        GridQueue<SizeT>        &queue,
        ReductionOp             &reduction_op)
    {
        // Each thread block is statically assigned at some input, otherwise its
        // block_aggregate will be undefined.
        SizeT block_offset = blockIdx.x * TILE_ITEMS;

        if (block_offset + TILE_ITEMS <= num_items)
        {
            // We have a full tile to consume
            T thread_aggregate;
            ConsumeFullTile<true>(smem_storage, d_in, block_offset, reduction_op, thread_aggregate);

            // Dynamically consume other tiles
            SizeT even_share_base = gridDim.x * TILE_ITEMS;

            if (even_share_base < num_items)
            {
                // There are tiles left to consume
                while (true)
                {
                    // Dequeue up to TILE_ITEMS
                    if (threadIdx.x == 0)
                    {
                        smem_storage.block_offset = queue.Drain(TILE_ITEMS) + even_share_base;
                    }

                    __syncthreads();

                    block_offset = smem_storage.block_offset;

                    if (block_offset + TILE_ITEMS > num_items)
                    {
                        if (block_offset < num_items)
                        {
                            // We have less than a full tile to consume
                            ConsumePartialTile<false>(smem_storage, d_in, block_offset, num_items, reduction_op, thread_aggregate);
                        }

                        // No more work to do
                        break;
                    }

                    // We have a full tile to consume (which performs a barrier to protect smem_storage.block_offset WARs)
                    ConsumeFullTile<false>(smem_storage, d_in, block_offset, reduction_op, thread_aggregate);
                }
            }

            // Compute the block-wide reduction (every thread has a valid input)
            return BlockReduceT::Reduce(smem_storage.reduce, thread_aggregate, reduction_op);
        }
        else
        {
            // We have less than a full tile to consume
            T thread_aggregate;
            SizeT block_items = num_items - block_offset;
            ConsumePartialTile<true>(smem_storage, d_in, block_offset, num_items, reduction_op, thread_aggregate);

            // Compute the block-wide reduction  (up to block_items threads have valid inputs)
            return BlockReduceT::Reduce(smem_storage.reduce, thread_aggregate, reduction_op, block_items);
        }
    }


    /**
     * \brief Consumes input tiles according to <tt>BlockScanTilesPolicy::GRID_MAPPING</tt>, computing a threadblock-wide reduction for thread<sub>0</sub> using the specified binary reduction functor.
     *
     * The return value is undefined in threads other than thread<sub>0</sub>.
     */
    template <typename SizeT, typename ReductionOp>
    static __device__ __forceinline__ T ProcessTiles(
        SmemStorage             &smem_storage,
        InputIterator           d_in,
        SizeT                   num_items,
        GridEvenShare<SizeT>    &even_share,
        GridQueue<SizeT>        &queue,
        ReductionOp             &reduction_op)
    {
        if (BlockScanTilesPolicy::GRID_MAPPING == GRID_MAPPING_EVEN_SHARE)
        {
            // Even share
            even_share.BlockInit();

            return ProcessTilesEvenShare(
                smem_storage,
                d_in,
                even_share.block_offset,
                even_share.block_oob,
                reduction_op);
        }
        else
        {
            // Dynamically dequeue
            return ProcessTilesDynamic(
                smem_storage,
                d_in,
                num_items,
                queue,
                reduction_op);
        }
    }

};


}               // CUB namespace
CUB_NS_POSTFIX  // Optional outer namespace(s)

