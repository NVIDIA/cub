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
 * cub::BlockSelectRegion implements a stateful abstraction of CUDA thread blocks for participating in device-wide select.
 */

#pragma once

#include <iterator>

#include "device_scan_types.cuh"
#include "../../block/block_load.cuh"
#include "../../block/block_store.cuh"
#include "../../block/block_scan.cuh"
#include "../../block/block_exchange.cuh"
#include "../../block/block_discontinuity.cuh"
#include "../../grid/grid_queue.cuh"
#include "../../util_iterator.cuh"
#include "../../util_namespace.cuh"

/// Optional outer namespace(s)
CUB_NS_PREFIX

/// CUB namespace
namespace cub {


/******************************************************************************
 * Tuning policy types
 ******************************************************************************/

/**
 * Parameterizable tuning policy type for BlockSelectRegion
 */
template <
    int                         _BLOCK_THREADS,                 ///< Threads per thread block
    int                         _ITEMS_PER_THREAD,              ///< Items per thread (per tile of input)
    BlockLoadAlgorithm          _LOAD_ALGORITHM,                ///< The BlockLoad algorithm to use
    bool                        _LOAD_WARP_TIME_SLICING,        ///< Whether or not only one warp's worth of shared memory should be allocated and time-sliced among block-warps during any load-related data transpositions (versus each warp having its own storage)
    CacheLoadModifier           _LOAD_MODIFIER,                 ///< Cache load modifier for reading input elements
    bool                        _TWO_PHASE_SCATTER,             ///< Whether or not to coalesce output values in shared memory before scattering them to global
    BlockScanAlgorithm          _SCAN_ALGORITHM>                ///< The BlockScan algorithm to use
struct BlockSelectRegionPolicy
{
    enum
    {
        BLOCK_THREADS           = _BLOCK_THREADS,               ///< Threads per thread block
        ITEMS_PER_THREAD        = _ITEMS_PER_THREAD,            ///< Items per thread (per tile of input)
        LOAD_WARP_TIME_SLICING  = _LOAD_WARP_TIME_SLICING,      ///< Whether or not only one warp's worth of shared memory should be allocated and time-sliced among block-warps during any load-related data transpositions (versus each warp having its own storage)
        TWO_PHASE_SCATTER       = _TWO_PHASE_SCATTER,           ///< Whether or not to coalesce output values in shared memory before scattering them to global
    };

    static const BlockLoadAlgorithm     LOAD_ALGORITHM          = _LOAD_ALGORITHM;      ///< The BlockLoad algorithm to use
    static const CacheLoadModifier      LOAD_MODIFIER           = _LOAD_MODIFIER;       ///< Cache load modifier for reading input elements
    static const BlockScanAlgorithm     SCAN_ALGORITHM          = _SCAN_ALGORITHM;      ///< The BlockScan algorithm to use
};




/******************************************************************************
 * Thread block abstractions
 ******************************************************************************/

/**
 * \brief BlockSelectRegion implements a stateful abstraction of CUDA thread blocks for participating in device-wide selection across a region of tiles
 *
 * Performs functor-based selection if SelectOp functor type != NullType
 * Otherwise performs flag-based selection if FlagIterator's value type != NullType
 * Otherwise performs discontinuity selection (keep unique)
 */
template <
    typename    BlockSelectRegionPolicy,        ///< Parameterized BlockSelectRegionPolicy tuning policy type
    typename    InputIterator,                  ///< Random-access input iterator type for selection items
    typename    FlagIterator,                   ///< Random-access input iterator type for selection flags (NullType* if a selection functor or discontinuity flagging is to be used for selection)
    typename    OutputIterator,                 ///< Random-access input iterator type for selected items
    typename    NumSelectedIterator,            ///< Output iterator type for recording number of items selected
    typename    SelectOp,                       ///< Selection operator type (NullType if selection flags or discontinuity flagging is to be used for selection)
    typename    OffsetTuple>                    ///< Signed integer tuple type for global scatter offsets (selections and rejections)
struct BlockSelectRegion
{
    //---------------------------------------------------------------------
    // Types and constants
    //---------------------------------------------------------------------

    // Data type of input iterator
    typedef typename std::iterator_traits<InputIterator>::value_type T;

    // Data type of flag iterator
    typedef typename std::iterator_traits<FlagIterator>::value_type Flag;

    // Signed integer type for global offsets
    typedef typename OffsetTuple::BaseType Offset;

    // Input iterator wrapper type
    typedef typename If<IsPointer<InputIterator>::VALUE,
            CacheModifiedInputIterator<BlockSelectRegionPolicy::LOAD_MODIFIER, T, Offset>,      // Wrap the native input pointer with CacheModifiedInputIterator
            InputIterator>::Type                                                                // Directly use the supplied input iterator type
        WrappedInputIterator;

    // Flag iterator wrapper type
    typedef typename If<IsPointer<FlagIterator>::VALUE,
            CacheModifiedInputIterator<BlockSelectRegionPolicy::LOAD_MODIFIER, Flag, Offset>,   // Wrap the native input pointer with CacheModifiedInputIterator
            InputIterator>::Type                                                                // Directly use the supplied input iterator type
        WrappedFlagIterator;

    // Constants
    enum
    {
        USE_SELECT_OP,
        USE_SELECT_FLAGS,
        USE_DISCONTINUITY,

        BLOCK_THREADS       = BlockSelectRegionPolicy::BLOCK_THREADS,
        ITEMS_PER_THREAD    = BlockSelectRegionPolicy::ITEMS_PER_THREAD,
        TWO_PHASE_SCATTER   = BlockSelectRegionPolicy::TWO_PHASE_SCATTER,
        TILE_ITEMS          = BLOCK_THREADS * ITEMS_PER_THREAD,
        KEEP_REJECTS        = sizeof(OffsetTuple) > sizeof(Offset),                 // Whether or not we push rejected items to the back of the output

        SELECT_METHOD       = (!Equals<SelectOp, NullType>::VALUE) ?
                                USE_SELECT_OP :
                                (!Equals<Flag, NullType>::VALUE) ?
                                    USE_SELECT_FLAGS :
                                    USE_DISCONTINUITY
    };

    // Parameterized BlockLoad type for input items
    typedef BlockLoad<
            WrappedInputIterator,
            BlockSelectRegionPolicy::BLOCK_THREADS,
            BlockSelectRegionPolicy::ITEMS_PER_THREAD,
            BlockSelectRegionPolicy::LOAD_ALGORITHM,
            BlockSelectRegionPolicy::LOAD_WARP_TIME_SLICING>
        BlockLoadT;

    // Parameterized BlockLoad type for flags
    typedef BlockLoad<
            WrappedFlagIterator,
            BlockSelectRegionPolicy::BLOCK_THREADS,
            BlockSelectRegionPolicy::ITEMS_PER_THREAD,
            BlockSelectRegionPolicy::LOAD_ALGORITHM,
            BlockSelectRegionPolicy::LOAD_WARP_TIME_SLICING>
        BlockLoadFlags;

    // Parameterized BlockExchange type for input items
    typedef BlockExchange<
            T,
            BLOCK_THREADS,
            ITEMS_PER_THREAD>
        BlockExchangeT;

    // Parameterized BlockDiscontinuity type for input items
    typedef BlockDiscontinuity<T, BLOCK_THREADS> BlockDiscontinuityT;

    // Tile status descriptor type
    typedef LookbackTileDescriptor<OffsetTuple> TileDescriptor;

    // Parameterized BlockScan type
    typedef BlockScan<
            OffsetTuple,
            BlockSelectRegionPolicy::BLOCK_THREADS,
            BlockSelectRegionPolicy::SCAN_ALGORITHM>
        BlockScanAllocations;

    // Callback type for obtaining tile prefix during block scan
    typedef LookbackBlockPrefixCallbackOp<
            OffsetTuple,
            Sum>
        LookbackPrefixCallbackOp;

    // Stateful BlockScan prefix callback type for managing a running total while scanning consecutive tiles
    typedef RunningBlockPrefixCallbackOp<
            OffsetTuple,
            Sum>
        RunningPrefixCallbackOp;

    // Shared memory type for this threadblock
    struct _TempStorage
    {
        union
        {
            struct
            {
                typename LookbackPrefixCallbackOp::TempStorage  prefix;     // Smem needed for cooperative prefix callback
                typename BlockScanAllocations::TempStorage      scan;       // Smem needed for tile scanning
            };

            // Smem needed for input loading
            typename BlockLoadT::TempStorage load_items;

            // Smem needed for flag loading
            typename BlockLoadFlags::TempStorage load_flags;

            // Smem needed for discontinuity detection
            typename BlockDiscontinuityT::TempStorage discontinuity;

            // Smem needed for two-phase scatter
            typename If<TWO_PHASE_SCATTER, typename BlockExchangeT::TempStorage, NullType>::Type exchange;
        };

        Offset tile_idx;   // Shared tile index
    };

    // Alias wrapper allowing storage to be unioned
    struct TempStorage : Uninitialized<_TempStorage> {};


    //---------------------------------------------------------------------
    // Per-thread fields
    //---------------------------------------------------------------------

    _TempStorage                &temp_storage;      ///< Reference to temp_storage
    WrappedInputIterator        d_in;               ///< Input data
    WrappedFlagIterator         d_flags;            ///< Input flags
    OutputIterator              d_out;              ///< Output data
    SelectOp                    select_op;          ///< Selection operator
    Offset                      num_items;          ///< Total number of input items


    //---------------------------------------------------------------------
    // Constructor
    //---------------------------------------------------------------------

    // Constructor
    __device__ __forceinline__
    BlockSelectRegion(
        TempStorage                 &temp_storage,      ///< Reference to temp_storage
        InputIterator               d_in,               ///< Input data
        FlagIterator                d_flags,            ///< Input flags
        OutputIterator              d_out,              ///< Output data
        SelectOp                    select_op,          ///< Selection operator
        Offset                      num_items)          ///< Total number of input items
    :
        temp_storage(temp_storage.Alias()),
        d_in(d_in),
        d_flags(d_flags),
        d_out(d_out),
        select_op(select_op),
        num_items(num_items)
    {}


    //---------------------------------------------------------------------
    // Utility methods for initializing the selection allocations
    //---------------------------------------------------------------------

    /**
     * Initialize selection allocations (specialized for selection operator and for discarding rejected items)
     */
    template <bool FIRST_TILE, bool FULL_TILE>
    __device__ __forceinline__ void InitializeAllocations(
        Offset                      block_offset,
        int                         valid_items,
        T                           (&items)[ITEMS_PER_THREAD],
        OffsetTuple                 (&allocations)[ITEMS_PER_THREAD],
        Int2Type<false>             keep_rejects,
        Int2Type<USE_SELECT_OP>     select_method)
    {
        #pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
        {
            if (FULL_TILE || ((threadIdx.x * ITEMS_PER_THREAD) + ITEM < valid_items))
            {
                allocations[ITEM].x = select_op(items[ITEM]);
            }
        }
    }


    /**
     * Initialize selection allocations (specialized for valid flags and for discarding rejected items)
     */
    template <bool FIRST_TILE, bool FULL_TILE>
    __device__ __forceinline__ void InitializeAllocations(
        Offset                      block_offset,
        int                         valid_items,
        T                           (&items)[ITEMS_PER_THREAD],
        OffsetTuple                 (&allocations)[ITEMS_PER_THREAD],
        Int2Type<false>             keep_rejects,
        Int2Type<USE_SELECT_FLAGS>  select_method)
    {
        __syncthreads();

        Flag flags[ITEMS_PER_THREAD];
        if (FULL_TILE)
            BlockLoadFlags(temp_storage.load_flags).Load(d_in + block_offset, flags);
        else
            BlockLoadFlags(temp_storage.load_flags).Load(d_in + block_offset, flags, valid_items);

        #pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
        {
            if (FULL_TILE || ((threadIdx.x * ITEMS_PER_THREAD) + ITEM < valid_items))
            {
                allocations[ITEM].x = flags[ITEM];
            }
        }
    }


    /**
     * Initialize selection allocations (specialized for discontinuity detection and for discarding rejected items)
     */
    template <bool FIRST_TILE, bool FULL_TILE>
    __device__ __forceinline__ void InitializeAllocations(
        Offset                      block_offset,
        int                         valid_items,
        T                           (&items)[ITEMS_PER_THREAD],
        OffsetTuple                 (&allocations)[ITEMS_PER_THREAD],
        Int2Type<false>             keep_rejects,
        Int2Type<USE_DISCONTINUITY> select_method)
    {
        __syncthreads();

        int flags[ITEMS_PER_THREAD];

        if (FIRST_TILE)
        {
            // First tile always flags the first item
            BlockDiscontinuityT(temp_storage.discontinuity).FlagHeads(flags, items, Inequality());
        }
        else
        {
            // Subsequent tiles require the last item from the previous tile
            T tile_predecessor_item;
            if (threadIdx.x == 0)
                tile_predecessor_item = d_in[block_offset - 1];

            BlockDiscontinuityT(temp_storage.discontinuity).FlagHeads(flags, items, Inequality(), tile_predecessor_item);
        }

        #pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
        {
            if (FULL_TILE || ((threadIdx.x * ITEMS_PER_THREAD) + ITEM < valid_items))
            {
                allocations[ITEM].x = flags[ITEM];
            }
        }
    }


    /**
     * Initialize selection allocations (for partitioning rejected items after selected items)
     */
    template <bool FIRST_TILE, bool FULL_TILE, int _SELECT_METHOD>
    __device__ __forceinline__ void InitializeAllocations(
        Offset                      block_offset,
        int                         valid_items,
        T                           (&items)[ITEMS_PER_THREAD],
        OffsetTuple                 (&allocations)[ITEMS_PER_THREAD],
        Int2Type<true>              keep_rejects,
        Int2Type<_SELECT_METHOD>    select_method)
    {
        // Initialize allocations for selected items
        InitializeAllocations<FIRST_TILE, FULL_TILE>(
            block_offset,
            valid_items,
            items,
            allocations,
            Int2Type<false>(),
            select_method);

        // Initialize allocations for rejected items
        #pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
        {
            if (FULL_TILE || ((threadIdx.x * ITEMS_PER_THREAD) + ITEM < valid_items))
            {
                allocations[ITEM].y = !allocations[ITEM].x;
            }
        }
    }


    //---------------------------------------------------------------------
    // Utility methods for scattering selections
    //---------------------------------------------------------------------

    /**
     * Scatter data items to select offsets (specialized for direct scattering and for discarding rejected items)
     */
    __device__ __forceinline__ void Scatter(
        T               (&items)[ITEMS_PER_THREAD],
        OffsetTuple     allocations[ITEMS_PER_THREAD],
        OffsetTuple     allocation_offsets[ITEMS_PER_THREAD],
        OffsetTuple     tile_exclusive_prefix,
        OffsetTuple     valid_items,
        Int2Type<false> keep_rejects,
        Int2Type<false> two_phase_scatter)
    {
        #pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
        {
            // Selected items are placed front-to-back
            if (allocations[ITEM].x)
            {
                d_out[allocation_offsets[ITEM].x] = items[ITEM];
            }
        }
    }


    /**
     * Scatter data items to select offsets (specialized for direct scattering and for partitioning rejected items after selected items)
     */
    __device__ __forceinline__ void Scatter(
        T               (&items)[ITEMS_PER_THREAD],
        OffsetTuple     allocations[ITEMS_PER_THREAD],
        OffsetTuple     allocation_offsets[ITEMS_PER_THREAD],
        OffsetTuple     tile_exclusive_prefix,
        OffsetTuple     valid_items,
        Int2Type<true>  keep_rejects,
        Int2Type<false> two_phase_scatter)
    {
        #pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
        {
            // Selected items are placed front-to-back
            if (allocations[ITEM].x)
            {
                d_out[allocation_offsets[ITEM].x] = items[ITEM];
            }

            // Rejected items are placed back-to-front
            if (allocations[ITEM].y)
            {
                d_out[num_items - allocation_offsets[ITEM].y] = items[ITEM];
            }
        }
    }


    /**
     * Scatter data items to select offsets (specialized for two-phase scattering and for discarding rejected items)
     */
    __device__ __forceinline__ void Scatter(
        T               (&items)[ITEMS_PER_THREAD],
        OffsetTuple     allocations[ITEMS_PER_THREAD],
        OffsetTuple     allocation_offsets[ITEMS_PER_THREAD],
        OffsetTuple     tile_exclusive_prefix,
        OffsetTuple     valid_items,
        Int2Type<false> keep_rejects,
        Int2Type<true>  two_phase_scatter)
    {
        Offset local_ranks[ITEMS_PER_THREAD];
        Offset is_valid[ITEMS_PER_THREAD];

        #pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
        {
            local_ranks[ITEM]   = allocation_offsets[ITEM].x - tile_exclusive_prefix.x;
            is_valid[ITEM]      = allocations[ITEM].x;
        }

        __syncthreads();

        BlockExchangeT(temp_storage.exchange).ScatterToStriped(items, local_ranks, is_valid, valid_items.x);

        // Selected items are placed front-to-back
        StoreDirectStriped<BLOCK_THREADS>(threadIdx.x, d_out + tile_exclusive_prefix.x, items, valid_items.x);
    }


    /**
     * Scatter data items to select offsets (specialized for two-phase scattering and for partitioning rejected items after selected items)
     */
    __device__ __forceinline__ void Scatter(
        T               (&items)[ITEMS_PER_THREAD],
        OffsetTuple     allocations[ITEMS_PER_THREAD],
        OffsetTuple     allocation_offsets[ITEMS_PER_THREAD],
        OffsetTuple     tile_exclusive_prefix,
        OffsetTuple     valid_items,
        Int2Type<true>  keep_rejects,
        Int2Type<true>  two_phase_scatter)
    {
        Offset local_ranks[ITEMS_PER_THREAD];
        Offset is_valid[ITEMS_PER_THREAD];

        #pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
        {
            local_ranks[ITEM]   = allocation_offsets[ITEM].x - tile_exclusive_prefix.x;
            is_valid[ITEM]      = allocations[ITEM].x;

            if (allocations[ITEM].y)
            {
                local_ranks[ITEM]   = allocation_offsets[ITEM].y - tile_exclusive_prefix.y + valid_items.x;
                is_valid[ITEM]      = 1;
            }
        }

        __syncthreads();

        // Coalesce selected and rejected items in shared memory, gathering in striped arrangements
        BlockExchangeT(temp_storage.exchange).ScatterToStriped(items, local_ranks, is_valid, valid_items);

        // Store in striped order
        #pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
        {
            Offset item_offset = (ITEM * BLOCK_THREADS) + threadIdx.x;
            if (item_offset < valid_items.x)
            {
                // Scatter selected items front-to-back
                d_out[tile_exclusive_prefix.x + item_offset] = items[ITEM];
            }
            else
            {
                item_offset -= valid_items.x;
                if (item_offset < valid_items.y)
                {
                    // Scatter rejected items back-to-front
                    d_out[num_items - (tile_exclusive_prefix.y + item_offset)] = items[ITEM];
                }
            }
        }
    }


    //---------------------------------------------------------------------
    // Cooperatively scan a device-wide sequence of tiles with other CTAs
    //---------------------------------------------------------------------

    /**
     * Process a tile of input (dynamic domino scan)
     */
    template <bool FULL_TILE>
    __device__ __forceinline__ OffsetTuple ConsumeTile(
        Offset                      num_items,          ///< Total number of input items
        int                         tile_idx,           ///< Tile index
        Offset                      block_offset,       ///< Tile offset
        TileDescriptor              *d_tile_status)     ///< Global list of tile status
    {
        int valid_items = num_items - block_offset;

        // Load items
        T items[ITEMS_PER_THREAD];

        if (FULL_TILE)
            BlockLoadT(temp_storage.load_items).Load(d_in + block_offset, items);
        else
            BlockLoadT(temp_storage.load_items).Load(d_in + block_offset, items, valid_items);

        // Zero-initialize output allocations
        OffsetTuple allocations[ITEMS_PER_THREAD];
        #pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
        {
            allocations[ITEM] = OffsetTuple();
        }

        // Compute allocation offsets
        OffsetTuple allocation_offsets[ITEMS_PER_THREAD];       // Allocation offsets
        OffsetTuple tile_exclusive_prefix;                                // The tile prefixes for selected (and rejected) items
        OffsetTuple block_aggregate;                            // Total number of items selected (and rejected)
        if (tile_idx == 0)
        {
            // Initialize selected/rejected output allocations for first tile
            InitializeAllocations<true, FULL_TILE>(
                block_offset,
                valid_items,
                items,
                allocations,
                Int2Type<KEEP_REJECTS>(),
                Int2Type<SELECT_METHOD>());

            __syncthreads();

            // Scan allocations
            BlockScanAllocations(temp_storage.scan).ExclusiveSum(allocations, allocation_offsets, block_aggregate);
            tile_exclusive_prefix = OffsetTuple();        // Zeros

            // Update tile status if there may be successor tiles (i.e., this tile is full)
            if (FULL_TILE && (threadIdx.x == 0))
                TileDescriptor::SetPrefix(d_tile_status, block_aggregate);
        }
        else
        {
            // Initialize selected/rejected output allocations for non-first tile
            InitializeAllocations<false, FULL_TILE>(
                block_offset,
                valid_items,
                items,
                allocations,
                Int2Type<KEEP_REJECTS>(),
                Int2Type<SELECT_METHOD>());

            __syncthreads();

            // Scan allocations
            LookbackPrefixCallbackOp prefix_op(d_tile_status, temp_storage.prefix, Sum(), tile_idx);

            BlockScanAllocations(temp_storage.scan).ExclusiveSum(allocations, allocation_offsets, block_aggregate, prefix_op);
            tile_exclusive_prefix = prefix_op.exclusive_prefix;
        }

        // Store selected items
        Scatter(
            items,
            allocations,
            allocation_offsets,
            tile_exclusive_prefix,
            block_aggregate,
            Int2Type<KEEP_REJECTS>(),
            Int2Type<TWO_PHASE_SCATTER>());

        // Return inclusive prefix
        return tile_exclusive_prefix + block_aggregate;
    }


    /**
     * Dequeue and scan tiles of items as part of a dynamic domino scan
     */
    template <typename NumSelectedIterator>         ///< Output iterator type having Offset value type
    __device__ __forceinline__ void ConsumeRegion(
        int                     num_tiles,          ///< Total number of input tiles
        GridQueue<int>          queue,              ///< Queue descriptor for assigning tiles of work to thread blocks
        TileDescriptor          *d_tile_status,     ///< Global list of tile status
        NumSelectedIterator     d_num_selected)     ///< Output total number selected
    {
        OffsetTuple total_selected;

#if CUB_PTX_VERSION < 200

        // No concurrent kernels allowed and blocks are launched in increasing order, so just assign one tile per block (up to 65K blocks)
        int     tile_idx        = blockIdx.x;
        Offset  block_offset    = Offset(TILE_ITEMS) * tile_idx;


        if (block_offset + TILE_ITEMS <= num_items)
            total_selected = ConsumeTile<true>(num_items, tile_idx, block_offset, d_tile_status);
        else if (block_offset < num_items)
            total_selected = ConsumeTile<false>(num_items, tile_idx, block_offset, d_tile_status);

#else

        // Get first tile
        if (threadIdx.x == 0)
            temp_storage.tile_idx = queue.Drain(1);

        __syncthreads();

        int tile_idx = temp_storage.tile_idx;
        Offset block_offset = Offset(TILE_ITEMS) * tile_idx;

        while (block_offset + TILE_ITEMS <= num_items)
        {
            // Consume full tile
            total_selected = ConsumeTile<true>(num_items, tile_idx, block_offset, d_tile_status);

            // Get next tile
            if (threadIdx.x == 0)
                temp_storage.tile_idx = queue.Drain(1);

            __syncthreads();

            tile_idx = temp_storage.tile_idx;
            block_offset = Offset(TILE_ITEMS) * tile_idx;
        }

        // Consume a partially-full tile
        if (block_offset < num_items)
        {
            total_selected = ConsumeTile<false>(num_items, tile_idx, block_offset, d_tile_status);
        }
#endif

        // The thread block processing the last tile must output the total number of items selected
        if ((threadIdx.x == 0) && (tile_idx == num_tiles - 1))
        {
            *d_num_selected = total_selected.x;
        }
    }

};


}               // CUB namespace
CUB_NS_POSTFIX  // Optional outer namespace(s)

