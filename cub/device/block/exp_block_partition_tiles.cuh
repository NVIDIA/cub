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
 * cub::BlockPartitionTiles implements a stateful abstraction of CUDA thread blocks for participating in device-wide list partitioning.
 */

#pragma once

#include <iterator>

#include "device_scan_types.cuh"
#include "../../block/block_load.cuh"
#include "../../block/block_store.cuh"
#include "../../block/block_scan.cuh"
#include "../../grid/grid_queue.cuh"
#include "../../util_namespace.cuh"

/// Optional outer namespace(s)
CUB_NS_PREFIX

/// CUB namespace
namespace cub {


/******************************************************************************
 * Tuning policy types
 ******************************************************************************/

/**
 * Tuning policy for BlockPartitionTiles
 */
template <
    int                         _BLOCK_THREADS,
    int                         _ITEMS_PER_THREAD,
    BlockLoadAlgorithm          _LOAD_ALGORITHM,
    bool                        _LOAD_WARP_TIME_SLICING,
    PtxLoadModifier             _LOAD_MODIFIER,
    BlockStoreAlgorithm         _STORE_ALGORITHM,
    bool                        _STORE_WARP_TIME_SLICING,
    BlockScanAlgorithm          _SCAN_ALGORITHM,
    bool                        _BACKFILL_SECOND_PARTITION>
struct BlockPartitionTilesPolicy
{
    enum
    {
        BLOCK_THREADS               = _BLOCK_THREADS,
        ITEMS_PER_THREAD            = _ITEMS_PER_THREAD,
        LOAD_WARP_TIME_SLICING      = _LOAD_WARP_TIME_SLICING,
        STORE_WARP_TIME_SLICING     = _STORE_WARP_TIME_SLICING,
        BACKFILL_SECOND_PARTITION   = _BACKFILL_SECOND_PARTITION,
    };

    static const BlockLoadAlgorithm     LOAD_ALGORITHM      = _LOAD_ALGORITHM;
    static const PtxLoadModifier        LOAD_MODIFIER       = _LOAD_MODIFIER;
    static const BlockStoreAlgorithm    STORE_ALGORITHM     = _STORE_ALGORITHM;
    static const BlockScanAlgorithm     SCAN_ALGORITHM      = _SCAN_ALGORITHM;
};





/******************************************************************************
 * Thread block abstractions
 ******************************************************************************/

/**
 * \brief BlockPartitionTiles implements a stateful abstraction of CUDA thread blocks for participating in device-wide list partitioning.
 *
 * Implements a single-pass "domino" strategy with variable predecessor look-back.
 */
template <
    typename BlockPartitionTilesPolicy,     ///< Tuning policy
    typename InputIterator,               ///< Input iterator type
    typename FirstOutputIterator,         ///< Output iterator type for first partition
    typename SecondOutputIterator,        ///< Output iterator type for second partition
    typename SelectOp,                      ///< Unary partition selection functor type specifying whether an input item goes into the first partition
    typename SizeT>                         ///< Offset integer type
struct BlockPartitionTiles
{
    //---------------------------------------------------------------------
    // Types and constants
    //---------------------------------------------------------------------

    // Data type of input iterator
    typedef typename std::iterator_traits<InputIterator>::value_type T;

    // Constants
    enum
    {
        BLOCK_THREADS               = BlockPartitionTilesPolicy::BLOCK_THREADS,
        ITEMS_PER_THREAD            = BlockPartitionTilesPolicy::ITEMS_PER_THREAD,
        TILE_ITEMS                  = BLOCK_THREADS * ITEMS_PER_THREAD,

        BACKFILL_SECOND_PARTITION   = BlockPartitionTilesPolicy::BACKFILL_SECOND_PARTITION,
        COMPACT_ONLY                = Int2Type<Equals<SecondOutputIterator, NullType>::VALUE,
    };

    // Block load type
    typedef BlockLoad<
        InputIterator,
        BlockPartitionTilesPolicy::BLOCK_THREADS,
        BlockPartitionTilesPolicy::ITEMS_PER_THREAD,
        BlockPartitionTilesPolicy::LOAD_ALGORITHM,
        BlockPartitionTilesPolicy::LOAD_MODIFIER,
        BlockPartitionTilesPolicy::LOAD_WARP_TIME_SLICING> BlockLoadT;

    // Scan data type
    typedef DevicePartitionScanTuple<SizeT> ScanTuple;

    // Tile status descriptor type
    typedef LookbackTileDescriptor<ScanTuple> TileDescriptor;

    // Callback type for obtaining inter-tile prefix during block scan
    typedef LookbackBlockPrefixCallbackOp<ScanTuple, cub::Sum> PrefixCallbackOp;

    // Block scan type
    typedef BlockScan<
        ScanTuple,
        BlockPartitionTilesPolicy::BLOCK_THREADS,
        BlockPartitionTilesPolicy::SCAN_ALGORITHM> BlockScanT;

    // Shared memory type for this threadblock
    struct _TempStorage
    {
        union
        {
            typename BlockLoadT::TempStorage            load;       // Smem needed for tile loading
            struct
            {
                typename PrefixCallbackOp::TempStorage          prefix;     // Smem needed for cooperative prefix callback
                typename BlockScanT::TempStorage        scan;       // Smem needed for tile scanning
            };
        };

        SizeT                                           tile_idx;   // Shared tile index
    };

    // Alias wrapper allowing storage to be unioned
    struct TempStorage : Uninitialized<_TempStorage> {};


    //---------------------------------------------------------------------
    // Per-thread fields
    //---------------------------------------------------------------------

    _TempStorage                &temp_storage;      ///< Reference to temp_storage
    InputIterator             d_in;               ///< Input data
    FirstOutputIterator       d_out_a;            ///< First partition output data
    SecondOutputIterator      d_out_b;            ///< Second partition output data
    SelectOp                    select_op;          ///< Unary partition selection operator



    //---------------------------------------------------------------------
    // Methods
    //---------------------------------------------------------------------

    // Constructor
    __device__ __forceinline__
    BlockPartitionTiles(
        TempStorage                 &temp_storage,      ///< Reference to temp_storage
        InputIterator             d_in,               ///< Input data
        FirstOutputIterator       d_out_a,            ///< First partition output data
        SecondOutputIterator      d_out_b,            ///< Second partition output data
        SelectOp                    select_op)          ///< Unary partition selection operator
    :
        temp_storage(temp_storage.Alias()),
        d_in(d_in),
        d_out_a(d_out_a),
        d_out_b(d_out_b),
        select_op(select_op)
    {}


    /**
     * Scatter a tile of output.  Specialized for compaction (only keep first partition)
     */
    template <bool FULL_TILE>
    __device__ __forceinline__ void ScatterTile(
        T               items[ITEMS_PER_THREAD],
        bool            selectors[ITEMS_PER_THREAD],
        ScanTuple       tuples[ITEMS_PER_THREAD],
        int             valid_items,
        Int2Type<true>  compact_only)
    {
        #pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
        {
            if (FULL_TILE || ((ITEM < valid_items - threadIdx.x * ITEMS_PER_THREAD)))
            {
                if (selectors[ITEM])
                {
                    // Scatter to first partition
                    d_out_a[tuples[ITEM].first_count] = items[ITEM];
                }
            }
        }
    }


    /**
     * Scatter a tile of output.  Specialized for 2-partition pivoting
     */
    template <bool FULL_TILE>
    __device__ __forceinline__ void ScatterTile(
        T               items[ITEMS_PER_THREAD],
        bool            selectors[ITEMS_PER_THREAD],
        ScanTuple       tuples[ITEMS_PER_THREAD],
        int             valid_items,
        Int2Type<false> compact_only)
    {
        #pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
        {
            if (FULL_TILE || ((ITEM < valid_items - threadIdx.x * ITEMS_PER_THREAD)))
            {
                if (selectors[ITEM])
                {
                    // Scatter to first partition
                    d_out_a[tuples[ITEM].first_count] = items[ITEM];
                }
                else
                {
                    // Store second partition items if the second partition is valid
                    if (BACKFILL_SECOND_PARTITION)
                    {
                        // Offset from second iterator is negative
                        *(d_out_b - tuples[ITEM].second_count - 1) = items[ITEM];
                    }
                    else
                    {
                        // Offset from second iterator is positive
                        d_out_b[tuples[ITEM].second_count] = items[ITEM];
                    }
                }
            }
        }
    }


    /**
     * Process a tile of input (dynamic domino scan)
     */
    template <bool FULL_TILE>
    __device__ __forceinline__ void ConsumeTile(
        SizeT                       num_items,          ///< Total number of input items
        int                         tile_idx,           ///< Tile index
        SizeT                       block_offset,       ///< Tile offset
        TileDescriptor              *d_tile_status)     ///< Global list of tile status
    {
        int valid_items = num_items - block_offset;

        // Load items
        T items[ITEMS_PER_THREAD];

        if (FULL_TILE)
            BlockLoadT(temp_storage.load).Load(d_in + block_offset, items);
        else
            BlockLoadT(temp_storage.load).Load(d_in + block_offset, items, valid_items);

        __syncthreads();

        // Initialize partition selectors and scan tuples
        bool        selectors[ITEMS_PER_THREAD];
        ScanTuple   tuples[ITEMS_PER_THREAD];

        #pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
        {
            tuples[ITEM].first_count = 0;
            tuples[ITEM].first_count = 0;

            if (FULL_TILE || ((ITEM < valid_items - threadIdx.x * ITEMS_PER_THREAD)))
            {
                selectors[ITEM] = select_op(items[ITEM]);
                tuples[ITEM].first_count = selectors[ITEM];
                tuples[ITEM].second_count = !selectors[ITEM];
            }
        }

        // Perform tile scan
        ScanTuple block_aggregate;
        if (tile_idx == 0)
        {
            // Scan first tile
            BlockScanT(temp_storage.scan).ExclusiveSum(tuples, tuples, block_aggregate);

            // Update tile status if this tile is full (i.e., there may be successor tiles)
            if (FULL_TILE && (threadIdx.x == 0))
                TileDescriptor::SetPrefix(d_tile_status, block_aggregate);
        }
        else
        {
            PrefixCallbackOp prefix_op(d_tile_status, temp_storage.prefix, cub::Sum, tile_idx);
            BlockScanT(temp_storage.scan).ExclusiveSum(items, items, block_aggregate, prefix_op);
        }

        __syncthreads();

        // Store items
        ScatterTile<FULL_TILE>(items, selectors, tuples, valid_items, Int2Type<COMPACT_ONLY>());
    }


    /**
     * Dequeue and scan tiles of items as part of a dynamic domino scan
     */
    __device__ __forceinline__ void ConsumeTiles(
        int                         num_items,          ///< Total number of input items
        GridQueue<int>              queue,              ///< Queue descriptor for assigning tiles of work to thread blocks
        TileDescriptor              *d_tile_status)     ///< Global list of tile status
    {
#if CUB_PTX_VERSION < 200

        // No concurrent kernels allowed and blocks are launched in increasing order, so just assign one tile per block (up to 65K blocks)
        int     tile_idx        = blockIdx.x;
        SizeT   block_offset    = SizeT(TILE_ITEMS) * tile_idx;

        if (block_offset + TILE_ITEMS <= num_items)
            ConsumeTile<true>(num_items, tile_idx, block_offset, d_tile_status);
        else if (block_offset < num_items)
            ConsumeTile<false>(num_items, tile_idx, block_offset, d_tile_status);

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
            ConsumeTile<true>(num_items, tile_idx, block_offset, d_tile_status);

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
            ConsumeTile<false>(num_items, tile_idx, block_offset, d_tile_status);
        }
#endif

    }


};


}               // CUB namespace
CUB_NS_POSTFIX  // Optional outer namespace(s)

