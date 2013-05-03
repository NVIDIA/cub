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
 * cub::GridMappingStrategy enumerates alternative strategies for mapping
 * constant-sized tiles of device-wide data onto a grid of CUDA thread
 * blocks.
 */

#pragma once

#include "../util_namespace.cuh"

/// Optional outer namespace(s)
CUB_NS_PREFIX

/// CUB namespace
namespace cub {


/**
 * \addtogroup GridModule
 * @{
 */


/**
 * GridMappingStrategy enumerates alternative strategies for mapping
 * constant-sized tiles of device-wide data onto a grid of CUDA thread
 * blocks.
 */
enum GridMappingStrategy
{
    /**
     * \brief An "even-share" strategy for assigning input tiles to thread blocks.
     *
     * \par Overview
     * The input is evenly partitioned into \p p segments, where \p p is
     * constant and corresponds loosely to the number of thread blocks that may
     * actively reside on the target device. Each segment is comprised of
     * consecutive tiles, where a tile is a small, constant-sized unit of input
     * to be processed to completion before the thread block terminates or
     * obtains more work.  The kernel invokes \p p thread blocks, each
     * of which iteratively consumes a segment of <em>n</em>/<em>p</em> elements
     * in tile-size increments.
     */
    GRID_MAPPING_EVEN_SHARE,

    /**
     * \brief A dynamic "queue-based" strategy for assigning input tiles to thread blocks.
     *
     * \par Overview
     * The input is treated as a queue to be dynamically consumed by a grid of
     * thread blocks.  Work is atomically dequeued in tiles, where a tile is a
     * unit of input to be processed to completion before the thread block
     * terminates or obtains more work.  The grid size \p p is constant,
     * loosely corresponding to the number of thread blocks that may actively
     * reside on the target device.
     */
    GRID_MAPPING_DYNAMIC,
};



/**
 * \brief Thread blocks (abstracted as \p grid_block instances) consume input tiles using an GridEvenShare descriptor. (See GridMappingStrategy::GRID_MAPPING_EVEN_SHARE.)
 *
 * Expects the \p GridBlock type to have the following member functions for
 * - Full-tile and partial-tile processing, respectively:
 *   - <tt>template <bool FIRST_TILE> void ConsumeTile(SizeT block_offset);</tt>
 *   - <tt>template <bool FIRST_TILE> void ConsumeTile(SizeT block_offset, SizeT valid_items);</tt>
 * - Full-block and partial-block finalization, respectively:
 *   - <tt>template <typename Result> void Finalize(Result &result);</tt>
 *   - <tt>template <typename Result> void Finalize(Result &result, SizeT valid_items);</tt>
 * - Getting the number of items processed per call to \p ConsumeTile:
 *   - <tt>int TileItems();</tt>
 *
 * \tparam GridBlock    <b>[inferred]</b> Thread block abstraction type for tile processing
 * \tparam SizeT        <b>[inferred]</b> Integral type used for global array indexing
 * \tparam Result       <b>[inferred]</b> Result type to be returned by the GridBlock instance
 */
template <
    typename                GridBlock,
    typename                SizeT,
    typename                Result>
__device__ __forceinline__ void BlockConsumeTiles(
    GridBlock               &grid_block,        ///< [in,out] Threadblock abstraction for tile processing
    SizeT                   num_items,          ///< [in] Total number of global input items
    GridEvenShare<SizeT>    &even_share,        ///< [in] GridEvenShare descriptor (assumed to have already been block-initialized).
    Result                  &result)            ///< [out] Result returned by <tt>grid_block::Finalize()</tt>
{
    SizeT block_offset = even_share.block_offset;

    // Number of items per tile that can be processed by grid_block
    int tile_items = grid_block.TileItems();

    if (block_offset + tile_items <= even_share.block_oob)
    {
        // We have at least one full tile to consume
        grid_block.ConsumeTile<true>(block_offset);
        block_offset += tile_items;

        // Consume any other full tiles
        while (block_offset + tile_items <= even_share.block_oob)
        {
            grid_block.ConsumeTile<false>(block_offset);
            block_offset += tile_items;
        }

        // Consume any remaining input
        grid_block.ConsumeTile<false>(block_offset, even_share.block_oob - block_offset);

        // Compute the block-wide reduction (every thread has a valid input)
        return grid_block.Finalize(result);
    }
    else
    {
        // We have less than a full tile to consume
        SizeT block_items = even_share.block_oob - block_offset;
        grid_block.ConsumeTile<true>(block_offset, block_items);
        grid_block.Finalize(result, block_items);
    }
}



/**
 * \brief Thread blocks (abstracted as \p grid_block instances) consume input tiles using an GridQueue descriptor. (See GridMappingStrategy::GRID_MAPPING_DYNAMIC.)
 *
 * Expects the \p GridBlock type to have the following member functions for
 * - Full-tile and partial-tile processing, respectively:
 *   - <tt>template <bool FIRST_TILE> void ConsumeTile(SizeT block_offset);</tt>
 *   - <tt>template <bool FIRST_TILE> void ConsumeTile(SizeT block_offset, SizeT valid_items);</tt>
 * - Full-block and partial-block finalization, respectively:
 *   - <tt>template <typename Result> void Finalize(Result &result);</tt>
 *   - <tt>template <typename Result> void Finalize(Result &result, SizeT valid_items);</tt>
 * - Getting the number of items processed per call to \p ConsumeTile:
 *   - <tt>int TileItems();</tt>
 *
 * \tparam GridBlock    <b>[inferred]</b> Thread block abstraction type for tile processing
 * \tparam SizeT        <b>[inferred]</b> Integral type used for global array indexing
 * \tparam Result       <b>[inferred]</b> Result type to be returned by the GridBlock instance
 */
template <
    typename                GridBlock,
    typename                SizeT,
    typename                Result>
__device__ __forceinline__ void BlockConsumeTiles(
    GridBlock               &grid_block,        ///< [in,out] Threadblock abstraction for tile processing
    SizeT                   num_items,          ///< [in] Total number of global input items
    GridQueue<SizeT>        &queue,             ///< [in,out] GridQueue descriptor
    Result                  &result)            ///< [out] Result returned by <tt>grid_block::Finalize()</tt>
{
    // Shared tile-processing offset obtained dynamically from queue
    __shared__ SizeT dynamic_block_offset;

    // Number of items per tile that can be processed by grid_block
    int tile_items = grid_block.TileItems();

    // We give each thread block at least one tile of input.
    SizeT block_offset = blockIdx.x * tile_items;

    // Check if we have a full tile to consume
    if (block_offset + tile_items <= num_items)
    {
        grid_block.ConsumeTile<true>(block_offset);

        // Now that every block in the kernel has gotten a tile, attempt to dynamically consume any remaining
        SizeT even_share_base = gridDim.x * tile_items;
        if (even_share_base < num_items)
        {
            // There are tiles left to consume
            while (true)
            {
                // Dequeue up to tile_items
                if (threadIdx.x == 0)
                {
                    dynamic_block_offset = queue.Drain(tile_items) + even_share_base;
                }

                __syncthreads();

                block_offset = dynamic_block_offset;

                __syncthreads();

                if (block_offset + tile_items > num_items)
                {
                    if (block_offset < num_items)
                    {
                        // We have less than a full tile to consume
                        grid_block.ConsumeTile<false>(block_offset, num_items - block_offset);
                    }

                    // No more work to do
                    break;
                }

                // We have a full tile to consume
                grid_block.ConsumeTile<false>(block_offset);
            }
        }

        // Compute the block-wide reduction (every thread has a valid input)
        grid_block.Finalize(result);
    }
    else
    {
        // We have less than a full tile to consume
        SizeT block_items = num_items - block_offset;
        grid_block.ConsumeTile<true>(block_offset, block_items);
        grid_block.Finalize(result, block_items);
    }
}







/** @} */       // end group GridModule

}               // CUB namespace
CUB_NS_POSTFIX  // Optional outer namespace(s)

