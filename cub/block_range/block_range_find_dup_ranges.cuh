/******************************************************************************
 * Copyright (c) 2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2014, NVIDIA CORPORATION.  All rights reserved.
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
 * cub::BlockRangeFindDupRanges implements a stateful abstraction of CUDA thread blocks for participating in device-wide reduce-value-by-key.
 */

#pragma once

#include <iterator>

#include "block_scan_prefix_operators.cuh"
#include "../block/block_load.cuh"
#include "../block/block_store.cuh"
#include "../block/block_scan.cuh"
#include "../block/block_exchange.cuh"
#include "../block/block_discontinuity.cuh"
#include "../grid/grid_queue.cuh"
#include "../iterator/cache_modified_input_iterator.cuh"
#include "../iterator/constant_input_iterator.cuh"
#include "../util_namespace.cuh"

/// Optional outer namespace(s)
CUB_NS_PREFIX

/// CUB namespace
namespace cub {


/******************************************************************************
 * Tuning policy types
 ******************************************************************************/

/**
 * Parameterizable tuning policy type for BlockRangeFindDupRanges
 */
template <
    int                         _BLOCK_THREADS,                 ///< Threads per thread block
    int                         _ITEMS_PER_THREAD,              ///< Items per thread (per tile of input)
    BlockLoadAlgorithm          _LOAD_ALGORITHM,                ///< The BlockLoad algorithm to use
    CacheLoadModifier           _LOAD_MODIFIER,                 ///< Cache load modifier for reading input elements
    bool                        _STORE_WARP_TIME_SLICING,       ///< Whether or not only one warp's worth of shared memory should be allocated and time-sliced among block-warps during any store-related data transpositions (versus each warp having its own storage)
    BlockScanAlgorithm          _SCAN_ALGORITHM>                ///< The BlockScan algorithm to use
struct BlockRangeFindDupRangesPolicy
{
    enum
    {
        BLOCK_THREADS           = _BLOCK_THREADS,               ///< Threads per thread block
        ITEMS_PER_THREAD        = _ITEMS_PER_THREAD,            ///< Items per thread (per tile of input)
        STORE_WARP_TIME_SLICING = _STORE_WARP_TIME_SLICING,     ///< Whether or not only one warp's worth of shared memory should be allocated and time-sliced among block-warps during any store-related data transpositions (versus each warp having its own storage)
    };

    static const BlockLoadAlgorithm     LOAD_ALGORITHM          = _LOAD_ALGORITHM;      ///< The BlockLoad algorithm to use
    static const CacheLoadModifier      LOAD_MODIFIER           = _LOAD_MODIFIER;       ///< Cache load modifier for reading input elements
    static const BlockScanAlgorithm     SCAN_ALGORITHM          = _SCAN_ALGORITHM;      ///< The BlockScan algorithm to use
};





/******************************************************************************
 * Thread block abstractions
 ******************************************************************************/

/**
 * \brief BlockRangeFindDupRanges implements a stateful abstraction of CUDA thread blocks for participating in device-wide reduce-value-by-key across a range of tiles
 */
template <
    typename    BlockRangeFindDupRangesPolicy,      ///< Parameterized BlockRangeFindDupRangesPolicy tuning policy type
    typename    InputIterator,                      ///< Random-access input iterator type for data
    typename    OffsetOutputIterator,               ///< Random-access output iterator type for offset values
    typename    ValueOutputIterator,               ///< Random-access output iterator type for value values
    typename    EqualityOp>                         ///< T equality operator type
struct BlockRangeFindDupRanges
{
    //---------------------------------------------------------------------
    // Types and constants
    //---------------------------------------------------------------------

    // Data type of key iterator
    typedef typename std::iterator_traits<InputIterator>::value_type T;

    // Signed integer type for global offsets
    typedef typename std::iterator_traits<OffsetOutputIterator>::value_type Offset;

    // Signed integer type for segment indices
    typedef Offset Index;

    // Signed integer type for segment values
    typedef typename std::iterator_traits<ValueOutputIterator>::value_type Value;

    // Tuple type for scanning (pairs segment-value and segment-index)
    typedef ItemOffsetPair<Value, Index> ValueOffsetPair;

    // Tile status descriptor interface type
    typedef ReduceByKeyScanTileState<Value, Index> ScanTileState;

    // Constants
    enum
    {
        BLOCK_THREADS           = BlockRangeFindDupRangesPolicy::BLOCK_THREADS,
        ITEMS_PER_THREAD        = BlockRangeFindDupRangesPolicy::ITEMS_PER_THREAD,
        TILE_ITEMS              = BLOCK_THREADS * ITEMS_PER_THREAD,

        /// Number of warp threads
        WARP_THREADS            = CUB_WARP_THREADS(PTX_ARCH),

        /// Number of active warps
        WARPS                   = (BLOCK_THREADS + WARP_THREADS - 1) / WARP_THREADS,

        /// Whether or not to sync after loading data
        SYNC_AFTER_LOAD         = (BlockRangeFindDupRangesPolicy::LOAD_ALGORITHM != BLOCK_LOAD_DIRECT),

        /// Whether or not only one warp's worth of shared memory should be allocated and time-sliced among block-warps during any store-related data transpositions (versus each warp having its own storage)
        STORE_WARP_TIME_SLICING = BlockRangeFindDupRangesPolicy::STORE_WARP_TIME_SLICING,
        ACTIVE_EXCHANGE_WARPS   = (STORE_WARP_TIME_SLICING) ? 1 : WARPS,
    };

    // Cache-modified input iterator wrapper type for data
    typedef typename If<IsPointer<InputIterator>::VALUE,
            CacheModifiedInputIterator<BlockRangeFindDupRangesPolicy::LOAD_MODIFIER, T, Offset>,      // Wrap the native input pointer with CacheModifiedValueInputIterator
            InputIterator>::Type                                                                     // Directly use the supplied input iterator type
        WrappedInputIterator;

    // Parameterized BlockLoad type for data
    typedef BlockLoad<
            WrappedInputIterator,
            BlockRangeFindDupRangesPolicy::BLOCK_THREADS,
            BlockRangeFindDupRangesPolicy::ITEMS_PER_THREAD,
            BlockRangeFindDupRangesPolicy::LOAD_ALGORITHM>
        BlockLoadT;

    // Parameterized BlockDiscontinuity type for data
    typedef BlockDiscontinuity<T, BLOCK_THREADS> BlockDiscontinuityT;

    // Parameterized WarpScan type
    typedef WarpScan<ValueOffsetPair> WarpScanPairs;

    // Reduce-value-by-segment scan operator
    typedef ReduceBySegmentOp<cub::Sum, ValueOffsetPair> ReduceBySegmentOp;

    // Callback type for obtaining tile prefix during block scan
    typedef BlockScanLookbackPrefixOp<
            ValueOffsetPair,
            ReduceBySegmentOp,
            ScanTileState>
        LookbackPrefixCallbackOp;

    // Shared memory type for this threadblock
    struct _TempStorage
    {
        union
        {
            struct
            {
                typename BlockDiscontinuityT::TempStorage       discontinuity;              // Smem needed for discontinuity detection
                typename WarpScanPairs::TempStorage         warp_scan[WARPS];           // Smem needed for warp-synchronous scans
                ValueOffsetPair                                warp_aggregates[WARPS];     // Smem needed for sharing warp-wide aggregates
                typename LookbackPrefixCallbackOp::TempStorage  prefix;                     // Smem needed for cooperative prefix callback
            };

            // Smem needed for input loading
            typename BlockLoadT::TempStorage        load_items;

            // Smem needed for two-phase scatter
            ValueOffsetPair exchange[ACTIVE_EXCHANGE_WARPS][(WARP_THREADS * ITEMS_PER_THREAD) + 1];
        };

        Offset      tile_idx;                   // Shared tile index
        Offset      tile_inclusive;             // Inclusive tile prefix
        Offset      exclusive_in_tile;             // Exclusive tile prefix
    };

    // Alias wrapper allowing storage to be unioned
    struct TempStorage : Uninitialized<_TempStorage> {};


    //---------------------------------------------------------------------
    // Per-thread fields
    //---------------------------------------------------------------------

    _TempStorage                    &temp_storage;      ///< Reference to temp_storage

    WrappedInputIterator            d_in;               ///< Input data
    OffsetOutputIterator            d_offsets_out;      ///< Input segment offsets
    ValueOutputIterator             d_values_out;       ///< Output segment values

    InequalityWrapper<EqualityOp>   inequality_op;      ///< T inequality operator
    ReduceBySegmentOp               scan_op;            ///< Reduce-value-by-flag scan operator
    Offset                          num_items;          ///< Total number of input items


    //---------------------------------------------------------------------
    // Constructor
    //---------------------------------------------------------------------

    // Constructor
    __device__ __forceinline__
    BlockRangeFindDupRanges(
        TempStorage                 &temp_storage,      ///< Reference to temp_storage
        InputIterator               d_in,               ///< Input data
        OffsetOutputIterator        d_offsets_out,      ///< Input segment offsets
        ValueOutputIterator        d_values_out,      ///< Output segment values
        EqualityOp                  equality_op,        ///< T equality operator
        Offset                      num_items)          ///< Total number of input items
    :
        temp_storage(temp_storage.Alias()),
        d_in(d_in),
        d_offsets_out(d_offsets_out),
        d_values_out(d_values_out),
        inequality_op(equality_op),
        scan_op(cub::Sum),
        num_items(num_items)
    {}




    //---------------------------------------------------------------------
    // Utility methods for initializing the selections
    //---------------------------------------------------------------------

    template <bool FIRST_TILE, bool LAST_TILE>
    __device__ __forceinline__ void InitializeSelections(
        Offset              block_offset,
        Offset              num_remaining,
        T                   (&items)[ITEMS_PER_THREAD],
        int                 (&head_flags)[ITEMS_PER_THREAD],
        ValueOffsetPair    (&values_and_segments)[ITEMS_PER_THREAD])
    {
        int                 tail_flags[ITEMS_PER_THREAD];

        if (FIRST_TILE && LAST_TILE)
        {
            // First-and-last-tile always head-flags the first item and tail-flags the last item

            BlockDiscontinuityT(temp_storage.discontinuity).FlagHeadsAndTails(
                head_flags, tail_flags, items, inequality_op);
        }
        else if (FIRST_TILE)
        {
            // First-tile always head-flags the first item

            // Get the first item from the next tile
            T tile_successor_item;
            if (threadIdx.x == BLOCK_THREADS - 1)
                tile_successor_item = d_in[block_offset + TILE_ITEMS];

            BlockDiscontinuityT(temp_storage.discontinuity).FlagHeadsAndTails(
                head_flags, tail_flags, tile_successor_item, items, inequality_op);
        }
        else if (LAST_TILE)
        {
            // Last-tile always flags the last item

            // Get the last item from the previous tile
            T tile_predecessor_item;
            if (threadIdx.x == 0)
                tile_predecessor_item = d_in[block_offset - 1];

            BlockDiscontinuityT(temp_storage.discontinuity).FlagHeadsAndTails(
                head_flags, tile_predecessor_item, tail_flags, items, inequality_op);
        }
        else
        {
            // Get the first item from the next tile
            T tile_successor_item;
            if (threadIdx.x == BLOCK_THREADS - 1)
                tile_successor_item = d_in[block_offset + TILE_ITEMS];

            // Get the last item from the previous tile
            T tile_predecessor_item;
            if (threadIdx.x == 0)
                tile_predecessor_item = d_in[block_offset - 1];

            BlockDiscontinuityT(temp_storage.discontinuity).FlagHeadsAndTails(
                head_flags, tile_predecessor_item, tail_flags, items, inequality_op);
        }

        // Zip counts and segments
        #pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
        {
            bool    is_trivial                  = head_flags[ITEM] & tail_flags[ITEM];
            bool    oob                         = (!LAST_TILE) || (Offset(threadIdx.x * ITEMS_PER_THREAD) + ITEM >= num_remaining);

            values_and_segments[ITEM].offset   = (is_trivial) ? head_flags[ITEM] : 0;                      //
            values_and_segments[ITEM].value    = (oob || is_trivial) ? 0 : 1;
        }
    }

    //---------------------------------------------------------------------
    // Scan utility methods
    //---------------------------------------------------------------------

    /**
     * Scan of allocations
     */
    __device__ __forceinline__ void WarpScanAllocations(
        ValueOffsetPair    &tile_aggregate,
        ValueOffsetPair    &warp_aggregate,
        ValueOffsetPair    &exclusive_in_warp,
        ValueOffsetPair    &thread_exclusive,
        ValueOffsetPair    (&values_and_segments)[ITEMS_PER_THREAD])
    {
        // Perform warpscans
        int warp_id = ((WARPS == 1) ? 0 : threadIdx.x / WARP_THREADS);
        int lane_id = LaneId();

        ValueOffsetPair thread_inclusive;
        ValueOffsetPair thread_aggregate = ThreadReduce(values_and_segments, scan_op);
        WarpScanPairs(temp_storage.warp_scan[warp_id]).Sum(thread_aggregate, thread_inclusive, thread_exclusive);

        // Last lane in each warp shares its warp-aggregate
        if (lane_id == WARP_THREADS - 1)
            temp_storage.warp_aggregates[warp_id] = thread_inclusive;

        __syncthreads();

        // Accumulate total selected and the warp-wide prefix
        exclusive_in_warp   = 0;
        warp_aggregate   = temp_storage.warp_aggregates[warp_id];
        tile_aggregate   = temp_storage.warp_aggregates[0];

        #pragma unroll
        for (int WARP = 1; WARP < WARPS; ++WARP)
        {
            if (warp_id == WARP)
                exclusive_in_warp = tile_aggregate;

            tile_aggregate = scan_op(tile_aggregate, temp_storage.warp_aggregates[WARP]);
        }
    }


    //---------------------------------------------------------------------
    // Utility methods for scattering selections
    //---------------------------------------------------------------------

    /**
     * Two-phase scatter, specialized for warp time-slicing
     */
    template <bool FIRST_TILE>
    __device__ __forceinline__ void ScatterTwoPhase(
        Offset              tile_segments_exclusive_in_global,
        Offset              warp_segments_aggregate,
        Offset              warp_segments_exclusive_in_tile,
        Offset              (&thread_segments_exclusive_in_warp)[ITEMS_PER_THREAD],
        ValueOffsetPair     (&values_and_offsets)[ITEMS_PER_THREAD],
        Int2Type<true>      is_warp_time_slice)
    {
        int warp_id = ((WARPS == 1) ? 0 : threadIdx.x / WARP_THREADS);
        int lane_id = LaneId();

        // Locally compact items, warp-by-warp
        if (warp_id == 0)
        {
            #pragma unroll
            for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
            {
                temp_storage.exchange[0][thread_segments_exclusive_in_warp[ITEM]] = values_and_offsets[ITEM];
            }

            __threadfence_block();

            #pragma unroll
            for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
            {
                int item_offset = (ITEM * WARP_THREADS) + lane_id;
                values_and_offsets[ITEM] = temp_storage.exchange[0][item_offset];
            }
        }

        // Locally compact items, warp-by-warp
        #pragma unroll
        for (int SLICE = 1; SLICE < WARPS; ++SLICE)
        {
            __syncthreads();

            if (warp_id == SLICE)
            {
                #pragma unroll
                for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
                {
                    temp_storage.exchange[0][thread_segments_exclusive_in_warp[ITEM]] = values_and_offsets[ITEM];
                }

                __threadfence_block();

                #pragma unroll
                for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
                {
                    int item_offset = (ITEM * WARP_THREADS) + lane_id;
                    values_and_offsets[ITEM] = temp_storage.exchange[0][item_offset];
                }
            }
        }

        // Global scatter
        #pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
        {
            if ((ITEM * WARP_THREADS) < warp_segments_aggregate - lane_id)
            {
                Offset item_offset =
                    tile_segments_exclusive_in_global +
                    warp_segments_exclusive_in_tile +
                    (ITEM * WARP_THREADS) + lane_id;

                // Scatter offset
                d_offsets_out[item_offset] = values_and_offsets[ITEM].offset;

                // Scatter value if not the first (global) value
                if ((!FIRST_TILE) || (ITEM != 0) || (threadIdx.x > 0))
                {
                    d_values_out[item_offset - 1] = values_and_offsets[ITEM].value;
                }
            }
        }
    }



    /**
     * Two-phase scatter
     * /
    __device__ __forceinline__ void ScatterTwoPhase(
        ValueOffsetPair    tile_aggregate,
        ValueOffsetPair    exclusive_in_tile,
        ValueOffsetPair    warp_aggregate,
        ValueOffsetPair    exclusive_in_warp,
        ValueOffsetPair    (&thread_exclusives)[ITEMS_PER_THREAD],
        T                   (&items)[ITEMS_PER_THREAD],
        Int2Type<false>     is_warp_time_slice)
    {
        int warp_id = ((WARPS == 1) ? 0 : threadIdx.x / WARP_THREADS);
        int lane_id = LaneId();

        // Locally compact items in each warp
        #pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
        {
            temp_storage.exchange[warp_id][thread_exclusives[ITEM]] = items[ITEM];
        }

        __threadfence_block();

        #pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
        {
            int item_offset = (ITEM * WARP_THREADS) + lane_id;
            items[ITEM] = temp_storage.exchange[warp_id][item_offset];
        }

        #pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
        {
            if ((ITEM * WARP_THREADS) < warp_aggregate - lane_id)
            {
                d_out[exclusive_in_tile + exclusive_in_warp + (ITEM * WARP_THREADS) + lane_id] = items[ITEM];
            }
        }
    }
*/

    template <bool FIRST_TILE>
    __device__ __forceinline__ void ScatterDirect(
        Offset              tile_segments_exclusive_in_global,
        Offset              warp_segments_aggregate,
        Offset              warp_segments_exclusive_in_tile,
        Offset              (&thread_segments_exclusive_in_warp)[ITEMS_PER_THREAD],
        ValueOffsetPair     (&values_and_offsets)[ITEMS_PER_THREAD])
    {
        #pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
        {
            if (thread_segments_exclusive_in_warp[ITEM] < warp_segments_aggregate)
            {
                Offset item_offset =
                    tile_segments_exclusive_in_global +
                    warp_segments_exclusive_in_tile +
                    thread_segments_exclusive_in_warp[ITEM];

                // Scatter offset
                d_offsets_out[item_offset] = values_and_offsets[ITEM].offset;

                // Scatter value if not the first (global) value
                if ((!FIRST_TILE) || (ITEM != 0) || (threadIdx.x > 0))
                {
                    d_values_out[item_offset - 1] = values_and_offsets[ITEM].value;
                }
            }
        }
    }


    template <bool FIRST_TILE>
    __device__ __forceinline__ void Scatter(
        Offset              tile_segments_aggregate,
        Offset              tile_segments_exclusive_in_global,
        Offset              warp_segments_aggregate,
        Offset              warp_segments_exclusive_in_tile,
        Offset              (&thread_segments_exclusive_in_warp)[ITEMS_PER_THREAD],
        ValueOffsetPair     (&values_and_offsets)[ITEMS_PER_THREAD])
    {
        if ((ITEMS_PER_THREAD == 1) || (tile_segments_aggregate < BLOCK_THREADS))
        {
            // Direct scatter if the warp has any items
            if (warp_segments_aggregate)
            {
                ScatterDirect<FIRST_TILE>(
                    tile_segments_exclusive_in_global,
                    warp_segments_aggregate,
                    warp_segments_exclusive_in_tile,
                    thread_segments_exclusive_in_warp,
                    values_and_offsets);
            }
        }
        else
        {
            // Scatter two phase
            ScatterTwoPhase(
                tile_segments_exclusive_in_global,
                warp_segments_aggregate,
                warp_segments_exclusive_in_tile,
                thread_segments_exclusive_in_warp,
                values_and_offsets);
                Int2Type<STORE_WARP_TIME_SLICING>());
        }
    }



    //---------------------------------------------------------------------
    // Cooperatively scan a device-wide sequence of tiles with other CTAs
    //---------------------------------------------------------------------

    /**
     * Process a tile of input (dynamic domino scan)
     */
    template <
        bool                LAST_TILE>
    __device__ __forceinline__ ValueOffsetPair ConsumeTile(
        Offset              num_items,          ///< Total number of global input items
        Offset              num_remaining,      ///< Number of global input items remaining (including this tile)
        int                 tile_idx,           ///< Tile index
        Offset              block_offset,       ///< Tile offset
        ScanTileState  &tile_status)       ///< Global list of tile status
    {
        if (tile_idx == 0)
        {
            // Load items
            T items[ITEMS_PER_THREAD];
            if (LAST_TILE)
            {
                // Repeat last item
                T oob_item = d_in[num_items - 1];
                BlockLoadT(temp_storage.load_items).Load(d_in + block_offset, items, num_remaining, oob_item);
            }
            else
            {
                BlockLoadT(temp_storage.load_items).Load(d_in + block_offset, items);
            }

            if (SYNC_AFTER_LOAD)
                __syncthreads();

            // Set flags
            int                head_flags[ITEMS_PER_THREAD];
            ValueOffsetPair    values_and_segments[ITEMS_PER_THREAD];

            InitializeSelections<true, LAST_TILE>(
                block_offset,
                num_remaining,
                items,
                head_flags,
                values_and_segments);

            // Exclusive scan of values and segments
            ValueOffsetPair tile_aggregate;
            ValueOffsetPair warp_aggregate;
            ValueOffsetPair warp_exclusive_in_tile;
            ValueOffsetPair thread_exclusive_in_warp;

            WarpScanAllocations(
                tile_aggregate,
                warp_aggregate,
                warp_exclusive_in_tile,
                thread_exclusive_in_warp,
                values_and_segments);

            // Update tile status if this is not the last tile
            if (!LAST_TILE && (threadIdx.x == 0))
                tile_status.SetInclusive(0, tile_aggregate);

            // Update values (global scope) and segment indices (block scope)
            ValueOffsetPair thread_exclusive = scan_op(warp_exclusive_in_tile, thread_exclusive_in_warp);
            ThreadScanExclusive(values_and_segments, values_and_segments, scan_op, thread_exclusive);

            // Zip
            ValueOffsetPair    values_and_offsets[ITEMS_PER_THREAD];
            Offset             thread_segments_exclusive_in_warp[ITEMS_PER_THREAD];

            #pragma unroll
            for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
            {
                values_and_offsets[ITEM].value              = values_and_segments[ITEM].value;
                values_and_offsets[ITEM].offset             = block_offset + (threadIdx.x * ITEMS_PER_THREAD) + ITEM;
                thread_segments_exclusive_in_warp[ITEM]     = (head_flags[ITEM]) ?
                                                                values_and_segments[ITEM].offset :      // keep
                                                                WARP_THREADS * ITEMS_PER_THREAD;        // discard
            }

            Offset tile_segments_aggregate              = tile_aggregate.offset;
            Offset tile_segments_exclusive_in_global    = 0;
            Offset warp_segments_aggregate              = warp_aggregate.offset;
            Offset warp_segments_exclusive_in_tile      = warp_exclusive_in_tile.offset;

            // Scatter
            Scatter(
                tile_segments_aggregate,
                tile_segments_exclusive_in_global,
                warp_segments_aggregate,
                warp_segments_exclusive_in_tile,
                thread_segments_exclusive_in_warp,
                values_and_offsets);

            // Return total number of segments found so far (inclusive of this tile)
            return tile_segments_aggregate;
        }
        else
        {
            // Not first tile

            T tile_predecessor_key = (threadIdx.x == 0) ?
                d_in[block_offset - 1] :
                ZeroInitialize<T>();

            __syncthreads();

            // Load values
            if (LAST_TILE)
                BlockLoadValues(temp_storage.load_values).Load(d_values_in + block_offset, values, num_remaining);
            else
                BlockLoadValues(temp_storage.load_values).Load(d_values_in + block_offset, values);

            __syncthreads();

            // Set head flags
            BlockDiscontinuityT(temp_storage.discontinuity).FlagHeads(flags, data, inequality_op, tile_predecessor_key);

            // Zip values and flags
            ZipValuesAndFlags<LAST_TILE>(num_remaining, values, flags, values_and_segments);

            // Exclusive scan of values and flags
            ValueOffsetPair block_aggregate;
            LookbackPrefixCallbackOp prefix_op(tile_status, temp_storage.prefix, scan_op, tile_idx);

            ScanBlock(values_and_segments, block_aggregate, prefix_op, Int2Type<HAS_IDENTITY_ZERO>());
            running_total = prefix_op.inclusive_prefix;

            // Scatter flagged items
            Scatter<LAST_TILE, false>(num_remaining, data, values_and_segments, flags, block_aggregate.offset, prefix_op.exclusive_prefix.offset);
        }

        return running_total;
    }


    /**
     * Dequeue and scan tiles of items as part of a dynamic domino scan
     */
    template <typename NumSegmentsIterator>         ///< Output iterator type for recording number of items selected
    __device__ __forceinline__ void ConsumeRange(
        int                     num_tiles,          ///< Total number of input tiles
        GridQueue<int>          queue,              ///< Queue descriptor for assigning tiles of work to thread blocks
        ScanTileState           &tile_status,       ///< Global list of tile status
        NumSegmentsIterator     d_num_segments)     ///< Output pointer for total number of segments identified
    {
#if (CUB_PTX_ARCH <= 130)
        // Blocks are launched in increasing order, so just assign one tile per block

        int     tile_idx        = (blockIdx.y * 32 * 1024) + blockIdx.x;    // Current tile index
        Offset  block_offset    = Offset(TILE_ITEMS) * tile_idx;            // Global offset for the current tile
        Offset  num_remaining   = num_items - block_offset;                 // Remaining items (including this tile)

        if (num_remaining > TILE_ITEMS)
        {
            // Full tile
            ConsumeTile<false>(num_items, num_remaining, tile_idx, block_offset, tile_status);
        }
        else if (num_remaining > 0)
        {
            // Last tile
            ValueOffsetPair running_total = ConsumeTile<true>(num_items, num_remaining, tile_idx, block_offset, tile_status);

            // Output the total number of items selected
            if (threadIdx.x == 0)
            {
                *d_num_segments = running_total.offset;

                // If the last tile is a whole tile, the inclusive prefix contains accumulated value reduction for the last segment
                if (num_remaining == TILE_ITEMS)
                {
                    d_values_out[running_total.offset - 1] = running_total.value;
                }
            }
        }
#else
        // Blocks may not be launched in increasing order, so work-steal tiles

        // Get first tile index
        if (threadIdx.x == 0)
            temp_storage.tile_idx = queue.Drain(1);

        __syncthreads();

        int     tile_idx        = temp_storage.tile_idx;
        Offset  block_offset    = Offset(TILE_ITEMS) * tile_idx;    // Global offset for the current tile
        Offset  num_remaining   = num_items - block_offset;         // Remaining items (including this tile)

        while (num_remaining > TILE_ITEMS)
        {
            // Consume full tile
            ConsumeTile<false>(num_items, num_remaining, tile_idx, block_offset, tile_status);

            // Get tile index
            if (threadIdx.x == 0)
                temp_storage.tile_idx = queue.Drain(1);

            __syncthreads();

            tile_idx        = temp_storage.tile_idx;
            block_offset    = Offset(TILE_ITEMS) * tile_idx;
            num_remaining   = num_items - block_offset;
        }

        if (num_remaining > 0)
        {
            // Consume last tile (treat as partially-full)
            ValueOffsetPair running_total = ConsumeTile<true>(num_items, num_remaining, tile_idx, block_offset, tile_status);

            if ((threadIdx.x == 0))
            {
                // Output the total number of items selected
                *d_num_segments = running_total.offset;

                // If the last tile is a whole tile, the inclusive prefix contains accumulated value reduction for the last segment
                if (num_remaining == TILE_ITEMS)
                {
                    d_values_out[running_total.offset - 1] = running_total.value;
                }
            }
        }
#endif
    }

};


}               // CUB namespace
CUB_NS_POSTFIX  // Optional outer namespace(s)

