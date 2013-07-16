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
 * cub::BlockPivotTiles implements a stateful abstraction of CUDA thread blocks for participating in device-wide pivot.
 */

#pragma once

#include <iterator>

#include "scan_tiles_types.cuh"
#include "../../block/block_load.cuh"
#include "../../block/block_store.cuh"
#include "../../block/block_scan.cuh"
#include "../../util_vector.cuh"
#include "../../util_namespace.cuh"

/// Optional outer namespace(s)
CUB_NS_PREFIX

/// CUB namespace
namespace cub {


/******************************************************************************
 * Tuning policy types
 ******************************************************************************/

/**
 * Tuning policy for BlockPivotTiles
 */
template <
    int                         _BLOCK_THREADS,
    int                         _ITEMS_PER_THREAD,
    PtxLoadModifier             _LOAD_MODIFIER,
    BlockScanAlgorithm          _SCAN_ALGORITHM>
struct BlockPivotTilesPolicy
{
    enum
    {
        BLOCK_THREADS           = _BLOCK_THREADS,
        ITEMS_PER_THREAD        = _ITEMS_PER_THREAD,
    };

    static const PtxLoadModifier        LOAD_MODIFIER       = _LOAD_MODIFIER;
    static const BlockScanAlgorithm     SCAN_ALGORITHM      = _SCAN_ALGORITHM;
};


/******************************************************************************
 * Thread block abstractions
 ******************************************************************************/

/**
 * \brief BlockPivotTiles implements a stateful abstraction of CUDA thread blocks for participating in device-wide pivot.
 *
 * Implements a single-pass "domino" strategy with adaptive prefix lookback.
 */
template <
    typename BlockPivotTilesPolicy,     ///< Tuning policy
    typename InputIteratorRA,           ///< Input iterator type
    typename OutputIteratorRA,          ///< Output iterator type
    typename PredicateOp,               ///< Pivot predicate functor type
    typename SizeT>                     ///< Offset integer type
struct BlockPivotTiles
{
    //---------------------------------------------------------------------
    // Types and constants
    //---------------------------------------------------------------------

    // Data type of input iterator
    typedef typename std::iterator_traits<InputIteratorRA>::value_type T;

    // Scan data type (pair of SizeT: first is offset from beginning, second is offset from end)
    typedef typename VectorHelper<SizeT, 2>::Type ScanTuple;

    // Constants
    enum
    {
        BLOCK_THREADS       = BlockPivotTilesPolicy::BLOCK_THREADS,
        ITEMS_PER_THREAD    = BlockPivotTilesPolicy::ITEMS_PER_THREAD,
        TILE_ITEMS          = BLOCK_THREADS * ITEMS_PER_THREAD,
    };

    // Load modifier
    static const PtxLoadModifier LOAD_MODIFIER = BlockPivotTilesPolicy::LOAD_MODIFIER;

    // Device tile status descriptor type
    typedef ScanTileDescriptor<ScanTuple> ScanTileDescriptorT;

    // Block scan type
    typedef BlockScan<
        ScanTuple,
        BlockPivotTilesPolicy::BLOCK_THREADS,
        BlockPivotTilesPolicy::SCAN_ALGORITHM> BlockScanT;

    // Scan functor
    struct ScanOp
    {
        __device__ __forceinline__ ScanTuple operator()(const ScanTuple &a, const ScanTuple &b)
        {
            ScanTuple retval;
            retval.x = a.x + b.x;
            retval.y = a.y + b.y;
            return retval;
        }
    };

    // Device scan prefix callback type for inter-block scans
    typedef DeviceScanBlockPrefixOp<ScanTuple, ScanOp> InterblockPrefixOp;

    // Shared memory type for this threadblock
    struct TempStorage
    {
        typename InterblockPrefixOp::TempStorage    prefix;         // Smem needed for cooperative prefix callback
        typename BlockScanT::TempStorage            scan;           // Smem needed for tile scanning
        SizeT                                       tile_idx;           // Shared tile index
    };


    //---------------------------------------------------------------------
    // Per-thread fields
    //---------------------------------------------------------------------

    TempStorage                 &temp_storage;      ///< Reference to temp_storage
    InputIteratorRA             d_in;               ///< Input data
    OutputIteratorRA            d_out;              ///< Output data
    ScanTileDescriptorT         *d_tile_status;     ///< Global list of tile status
    PredicateOp                 pred_op;            ///< Binary scan operator
    SizeT                       num_items;          ///< Total number of input items


    //---------------------------------------------------------------------
    // Constructor
    //---------------------------------------------------------------------

    // Constructor
    __device__ __forceinline__
    BlockPivotTiles(
        TempStorage                 &temp_storage,      ///< Reference to temp_storage
        InputIteratorRA             d_in,               ///< Input data
        OutputIteratorRA            d_out,              ///< Output data
        ScanTileDescriptorT         *d_tile_status,     ///< Global list of tile status
        PredicateOp                 pred_op,            ///< Binary scan operator
        SizeT                       num_items)
    :
        temp_storage(temp_storage),
        d_in(d_in),
        d_out(d_out),
        d_tile_status(d_tile_status),
        pred_op(pred_op),
        num_items(num_items)
    {}


    //---------------------------------------------------------------------
    // Domino scan
    //---------------------------------------------------------------------

    /**
     * Process a tile of input
     */
    template <bool FULL_TILE>
    __device__ __forceinline__ void ConsumeTile(
        int         tile_idx,           ///< Tile index
        SizeT       block_offset,       ///< Tile offset
        ScanTuple   &running_total)     ///< Running total
    {
        T           items[ITEMS_PER_THREAD];
        ScanTuple   tuples[ITEMS_PER_THREAD];

        // Load items
        int valid_items = num_items - block_offset;
        if (FULL_TILE)
            LoadStriped<LOAD_MODIFIER, BLOCK_THREADS>(threadIdx.x, d_in + block_offset, items);
        else
            LoadStriped<LOAD_MODIFIER, BLOCK_THREADS>(threadIdx.x, d_in + block_offset, items, valid_items);

        // Prevent hoisting
//        __syncthreads();
//        __threadfence_block();

        // Apply predicate to form scan tuples
        #pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
        {
            tuples[ITEM].x = pred_op(items[ITEM]);
            tuples[ITEM].y = !tuples[ITEM].x;
        }

        // Perform inclusive scan over scan tuples
        ScanTuple block_aggregate;
        if (tile_idx == 0)
        {
            BlockScanT(temp_storage.scan).InclusiveScan(tuples, tuples, ScanOp(), block_aggregate);
            running_total = block_aggregate;

            // Update tile status if there are successor tiles
            if (FULL_TILE && (threadIdx.x == 0))
                ScanTileDescriptorT::SetPrefix(d_tile_status, block_aggregate);
        }
        else
        {
            InterblockPrefixOp prefix_op(d_tile_status, temp_storage.prefix, ScanOp(), tile_idx);
            BlockScanT(temp_storage.scan).InclusiveScan(tuples, tuples, ScanOp(), block_aggregate, prefix_op);
            running_total = prefix_op.inclusive_prefix;
        }

        // Scatter items
        #pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
        {
            // Scatter if not out-of-bounds
            if (FULL_TILE || (threadIdx.x + (ITEM * BLOCK_THREADS) < valid_items))
            {
                SizeT scatter_offset = (pred_op(items[ITEM])) ?
                    tuples[ITEM].x - 1 :
                    num_items - tuples[ITEM];

                d_out[scatter_offset] = items[ITEM];
            }
        }
    }


    /**
     * Dequeue and scan tiles of items as part of a domino scan
     */
    __device__ __forceinline__ void ConsumeTiles(
        GridQueue<int>          queue,              ///< Queue descriptor for assigning tiles of work to thread blocks
        ScanTuple               &running_total)     ///< Running total
    {
#if CUB_PTX_ARCH < 200

        // No concurrent kernels allowed and blocks are launched in increasing order, so just assign one tile per block (up to 65K blocks)
        int     tile_idx        = blockIdx.x;
        SizeT   block_offset    = SizeT(TILE_ITEMS) * tile_idx;

        if (block_offset + TILE_ITEMS <= num_items)
        {
            ConsumeTile<true>(tile_idx, block_offset, running_total);
        }
        else if (block_offset < num_items)
        {
            ConsumeTile<false>(tile_idx, block_offset, running_total);
        }

#else

        // Get first tile
        if (threadIdx.x == 0)
            temp_storage.tile_idx = queue.Drain(1);

        __syncthreads();

        int tile_idx = temp_storage.tile_idx;
        SizeT block_offset = SizeT(TILE_ITEMS) * tile_idx;

        while (block_offset + TILE_ITEMS <= num_items)
        {
            // Consume full tile
            ConsumeTile<true>(tile_idx, block_offset, running_total);

            // Get next tile
            if (threadIdx.x == 0)
                temp_storage.tile_idx = queue.Drain(1);

            __syncthreads();

            tile_idx = temp_storage.tile_idx;
            block_offset = SizeT(TILE_ITEMS) * tile_idx;
        }

        // Consume a partially-full tile
        if (block_offset < num_items)
        {
            ConsumeTile<false>(tile_idx, block_offset, running_total);
        }
#endif
    }
};


}               // CUB namespace
CUB_NS_POSTFIX  // Optional outer namespace(s)

