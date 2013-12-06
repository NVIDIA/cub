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
#include "../../iterator/cache_modified_input_iterator.cuh"
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
    typename    SelectOp,                       ///< Selection operator type (NullType if selection flags or discontinuity flagging is to be used for selection)
    typename    EqualityOp,                     ///< Equality operator type (NullType if selection functor or selection flags is to be used for selection)
    typename    Offset,                         ///< Signed integer tuple type for global scatter offsets (selections and rejections)
    bool        KEEP_REJECTS>                   ///< Whether or not we push rejected items to the back of the output
struct BlockSelectRegion
{
    //---------------------------------------------------------------------
    // Types and constants
    //---------------------------------------------------------------------

    // Data type of input iterator
    typedef typename std::iterator_traits<InputIterator>::value_type T;

    // Data type of flag iterator
    typedef typename std::iterator_traits<FlagIterator>::value_type Flag;

    // Input iterator wrapper type
    typedef typename If<IsPointer<InputIterator>::VALUE,
            CacheModifiedInputIterator<BlockSelectRegionPolicy::LOAD_MODIFIER, T, Offset>,      // Wrap the native input pointer with CacheModifiedInputIterator
            InputIterator>::Type                                                                // Directly use the supplied input iterator type
        WrappedInputIterator;

    // Flag iterator wrapper type
    typedef typename If<IsPointer<FlagIterator>::VALUE,
            CacheModifiedInputIterator<BlockSelectRegionPolicy::LOAD_MODIFIER, Flag, Offset>,   // Wrap the native input pointer with CacheModifiedInputIterator
            FlagIterator>::Type                                                                 // Directly use the supplied input iterator type
        WrappedFlagIterator;

    // Constants
    enum
    {
        USE_SELECT_OP,
        USE_SELECT_FLAGS,
        USE_DISCONTINUITY,

        BLOCK_THREADS       = BlockSelectRegionPolicy::BLOCK_THREADS,
        ITEMS_PER_THREAD    = BlockSelectRegionPolicy::ITEMS_PER_THREAD,
        TWO_PHASE_SCATTER   = (BlockSelectRegionPolicy::TWO_PHASE_SCATTER) && (ITEMS_PER_THREAD > 1),
        TILE_ITEMS          = BLOCK_THREADS * ITEMS_PER_THREAD,

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
    typedef LookbackTileDescriptor<Offset> TileDescriptor;

    // Parameterized BlockScan type
    typedef BlockScan<
            Offset,
            BlockSelectRegionPolicy::BLOCK_THREADS,
            BlockSelectRegionPolicy::SCAN_ALGORITHM>
        BlockScanAllocations;

    // Callback type for obtaining tile prefix during block scan
    typedef LookbackBlockPrefixCallbackOp<
            Offset,
            Sum>
        LookbackPrefixCallbackOp;

    // Stateful BlockScan prefix callback type for managing a running total while scanning consecutive tiles
    typedef RunningBlockPrefixCallbackOp<
            Offset,
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

        Offset      tile_idx;               // Shared tile index
        Offset      tile_exclusive_prefix;  // Exclusive tile prefix
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
    EqualityOp                  equality_op;        ///< Equality operator
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
        EqualityOp                  equality_op,        ///< Equality operator
        Offset                      num_items)          ///< Total number of input items
    :
        temp_storage(temp_storage.Alias()),
        d_in(d_in),
        d_flags(d_flags),
        d_out(d_out),
        select_op(select_op),
        equality_op(equality_op),
        num_items(num_items)
    {}


    //---------------------------------------------------------------------
    // Utility methods for initializing the selection allocations
    //---------------------------------------------------------------------

    /**
     * Initialize selection allocations (specialized for selection operator)
     */
    template <bool FIRST_TILE, bool FULL_TILE>
    __device__ __forceinline__ void InitializeAllocations(
        Offset                      block_offset,
        int                         valid_items,
        T                           (&items)[ITEMS_PER_THREAD],
        Offset                      (&allocations)[ITEMS_PER_THREAD],
        Int2Type<USE_SELECT_OP>     select_method)
    {
        #pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
        {
            if (FULL_TILE || ((threadIdx.x * ITEMS_PER_THREAD) + ITEM < valid_items))
            {
                allocations[ITEM] = select_op(items[ITEM]);
            }
        }
    }


    /**
     * Initialize selection allocations (specialized for valid flags)
     */
    template <bool FIRST_TILE, bool FULL_TILE>
    __device__ __forceinline__ void InitializeAllocations(
        Offset                      block_offset,
        int                         valid_items,
        T                           (&items)[ITEMS_PER_THREAD],
        Offset                      (&allocations)[ITEMS_PER_THREAD],
        Int2Type<USE_SELECT_FLAGS>  select_method)
    {
        __syncthreads();

        Flag flags[ITEMS_PER_THREAD];

        #pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
        {
            flags[ITEM] = 0;
        }

        if (FULL_TILE)
            BlockLoadFlags(temp_storage.load_flags).Load(d_flags + block_offset, flags);
        else
            BlockLoadFlags(temp_storage.load_flags).Load(d_flags + block_offset, flags, valid_items);

        #pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
        {
            allocations[ITEM] = flags[ITEM];
        }
    }


    /**
     * Initialize selection allocations (specialized for discontinuity detection)
     */
    template <bool FIRST_TILE, bool FULL_TILE>
    __device__ __forceinline__ void InitializeAllocations(
        Offset                      block_offset,
        int                         valid_items,
        T                           (&items)[ITEMS_PER_THREAD],
        Offset                      (&allocations)[ITEMS_PER_THREAD],
        Int2Type<USE_DISCONTINUITY> select_method)
    {
        __syncthreads();

        int flags[ITEMS_PER_THREAD];

        InequalityWrapper<EqualityOp> inequality_op(equality_op);

        if (FIRST_TILE)
        {
            // First tile always flags the first item
            BlockDiscontinuityT(temp_storage.discontinuity).FlagHeads(flags, items, inequality_op);
        }
        else
        {
            // Subsequent tiles require the last item from the previous tile
            T tile_predecessor_item;
            if (threadIdx.x == 0)
                tile_predecessor_item = d_in[block_offset - 1];

            BlockDiscontinuityT(temp_storage.discontinuity).FlagHeads(flags, items, inequality_op, tile_predecessor_item);
        }

        #pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
        {
            if (FULL_TILE || ((threadIdx.x * ITEMS_PER_THREAD) + ITEM < valid_items))
            {
                allocations[ITEM] = flags[ITEM];
            }
        }
    }


    //---------------------------------------------------------------------
    // Utility methods for scattering selections
    //---------------------------------------------------------------------

    /**
     * Scatter data items to select offsets (specialized for direct scattering and for discarding rejected items)
     */
    template <bool FULL_TILE>
    __device__ __forceinline__ void Scatter(
        Offset          tile_idx,
        T               (&items)[ITEMS_PER_THREAD],
        Offset          allocations[ITEMS_PER_THREAD],
        Offset          allocation_offsets[ITEMS_PER_THREAD],
        Offset          tile_exclusive_prefix,
        Offset          tile_num_selected,
        Offset          valid_items,
        Int2Type<false> keep_rejects,
        Int2Type<false> two_phase_scatter)
    {
        #pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
        {
            if (allocations[ITEM])
            {
                // Selected items are placed front-to-back
                d_out[allocation_offsets[ITEM]] = items[ITEM];
            }
        }
    }


    /**
     * Scatter data items to select offsets (specialized for direct scattering and for partitioning rejected items after selected items)
     */
    template <bool FULL_TILE>
    __device__ __forceinline__ void Scatter(
        Offset          tile_idx,
        T               (&items)[ITEMS_PER_THREAD],
        Offset          allocations[ITEMS_PER_THREAD],
        Offset          allocation_offsets[ITEMS_PER_THREAD],
        Offset          tile_exclusive_prefix,
        Offset          tile_num_selected,
        Offset          valid_items,
        Int2Type<true>  keep_rejects,
        Int2Type<false> two_phase_scatter)
    {
        #pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
        {
            if (allocations[ITEM])
            {
                // Selected items are placed front-to-back
                d_out[allocation_offsets[ITEM]] = items[ITEM];
            }
            else if (FULL_TILE || ((threadIdx.x * ITEMS_PER_THREAD) + ITEM < valid_items))
            {
                Offset global_idx = (tile_idx * TILE_ITEMS) + (threadIdx.x * ITEMS_PER_THREAD) + ITEM;
                Offset reject_idx = global_idx - allocation_offsets[ITEM];

                // Rejected items are placed back-to-front
                d_out[num_items - reject_idx - 1] = items[ITEM];
            }
        }
    }


    /**
     * Scatter data items to select offsets (specialized for two-phase scattering and for discarding rejected items)
     */
    template <bool FULL_TILE>
    __device__ __forceinline__ void Scatter(
        Offset          tile_idx,
        T               (&items)[ITEMS_PER_THREAD],
        Offset          allocations[ITEMS_PER_THREAD],
        Offset          allocation_offsets[ITEMS_PER_THREAD],
        Offset          tile_exclusive_prefix,
        Offset          tile_num_selected,
        Offset          valid_items,
        Int2Type<false> keep_rejects,
        Int2Type<true>  two_phase_scatter)
    {
        if ((tile_num_selected >> Log2<BLOCK_THREADS>::VALUE) == 0)
        {
            // Average number of selected items per thread is less than one, so just do a one-phase scatter
            Scatter<FULL_TILE>(
                tile_idx,
                items,
                allocations,
                allocation_offsets,
                tile_exclusive_prefix,
                tile_num_selected,
                valid_items,
                keep_rejects,
                Int2Type<false>());
        }
        else
        {
            // Share exclusive tile prefix
            if (threadIdx.x == 0)
            {
                temp_storage.tile_exclusive_prefix = tile_exclusive_prefix;
            }

            __syncthreads();

            // Load exclusive tile prefix in all threads
            tile_exclusive_prefix = temp_storage.tile_exclusive_prefix;

            int local_ranks[ITEMS_PER_THREAD];

            #pragma unroll
            for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
            {
                local_ranks[ITEM] = allocation_offsets[ITEM] - tile_exclusive_prefix;
            }

            BlockExchangeT(temp_storage.exchange).ScatterToStriped(items, local_ranks, allocations, tile_num_selected);

            // Selected items are placed front-to-back
            StoreDirectStriped<BLOCK_THREADS>(threadIdx.x, d_out + tile_exclusive_prefix, items, tile_num_selected);
        }
    }


    /**
     * Scatter data items to select offsets (specialized for two-phase scattering and for partitioning rejected items after selected items)
     */
    template <bool FULL_TILE>
    __device__ __forceinline__ void Scatter(
        Offset          tile_idx,
        T               (&items)[ITEMS_PER_THREAD],
        Offset          allocations[ITEMS_PER_THREAD],
        Offset          allocation_offsets[ITEMS_PER_THREAD],
        Offset          tile_exclusive_prefix,
        Offset          tile_num_selected,
        Offset          valid_items,
        Int2Type<true>  keep_rejects,
        Int2Type<true>  two_phase_scatter)
    {
        // Share exclusive tile prefix
        if (threadIdx.x == 0)
        {
            temp_storage.tile_exclusive_prefix = tile_exclusive_prefix;
        }

        __syncthreads();

        // Load exclusive tile prefix in all threads
        tile_exclusive_prefix = temp_storage.tile_exclusive_prefix;

        int     local_ranks[ITEMS_PER_THREAD];
        bool    is_valid[ITEMS_PER_THREAD];

        Offset tile_rejected_exclusive_prefix = (tile_idx * TILE_ITEMS) - tile_exclusive_prefix;

        // Determine local scatter offsets
        #pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
        {
            is_valid[ITEM] = true;
            if (allocations[ITEM])
            {
                // Selected items
                local_ranks[ITEM] = allocation_offsets[ITEM] - tile_exclusive_prefix;
            }
            else if (FULL_TILE || ((threadIdx.x * ITEMS_PER_THREAD) + ITEM < valid_items))
            {
                // Rejected items
                Offset global_idx = (tile_idx * TILE_ITEMS) + (threadIdx.x * ITEMS_PER_THREAD) + ITEM;
                Offset reject_idx = global_idx - allocation_offsets[ITEM];
                local_ranks[ITEM] = (reject_idx - tile_rejected_exclusive_prefix) + tile_num_selected;
            }
            else
            {
                is_valid[ITEM] = false;
            }
        }

        // Coalesce selected and rejected items in shared memory, gathering in striped arrangements
        BlockExchangeT(temp_storage.exchange).ScatterToStriped(items, local_ranks, is_valid, valid_items);

        // Store in striped order
        #pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
        {
            Offset local_idx = (ITEM * BLOCK_THREADS) + threadIdx.x;
            if (local_idx < tile_num_selected)
            {
                // Scatter selected items front-to-back
                d_out[tile_exclusive_prefix + local_idx] = items[ITEM];
            }
            else if (local_idx < valid_items)
            {
                // Scatter rejected items back-to-front
                d_out[num_items - (tile_rejected_exclusive_prefix + (local_idx - tile_num_selected)) - 1] = items[ITEM];
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
    __device__ __forceinline__ Offset ConsumeTile(
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
        Offset allocations[ITEMS_PER_THREAD];
        #pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
        {
            allocations[ITEM] = ZeroInitialize<Offset>();
        }

        // Compute allocation offsets
        Offset allocation_offsets[ITEMS_PER_THREAD];       // Allocation offsets
        Offset tile_exclusive_prefix;                      // The tile prefixes for selected (and rejected) items
        Offset tile_num_selected;                            // Total number of items selected (and rejected)
        if (tile_idx == 0)
        {
            // Initialize selected/rejected output allocations for first tile
            InitializeAllocations<true, FULL_TILE>(
                block_offset,
                valid_items,
                items,
                allocations,
                Int2Type<SELECT_METHOD>());

            __syncthreads();

            // Scan allocations
            BlockScanAllocations(temp_storage.scan).ExclusiveSum(allocations, allocation_offsets, tile_num_selected);
            tile_exclusive_prefix = ZeroInitialize<Offset>();        // Zeros

            // Update tile status if there may be successor tiles (i.e., this tile is full)
            if (FULL_TILE && (threadIdx.x == 0))
                TileDescriptor::SetPrefix(d_tile_status, tile_num_selected);
        }
        else
        {
            // Initialize selected/rejected output allocations for non-first tile
            InitializeAllocations<false, FULL_TILE>(
                block_offset,
                valid_items,
                items,
                allocations,
                Int2Type<SELECT_METHOD>());

            __syncthreads();

            // Scan allocations
            LookbackPrefixCallbackOp prefix_op(d_tile_status, temp_storage.prefix, Sum(), tile_idx);

            BlockScanAllocations(temp_storage.scan).ExclusiveSum(allocations, allocation_offsets, tile_num_selected, prefix_op);
            tile_exclusive_prefix = prefix_op.exclusive_prefix;
        }

        // Store selected items
        Scatter<FULL_TILE>(
            tile_idx,
            items,
            allocations,
            allocation_offsets,
            tile_exclusive_prefix,
            tile_num_selected,
            valid_items,
            Int2Type<KEEP_REJECTS>(),
            Int2Type<TWO_PHASE_SCATTER>());

        // Return inclusive prefix
        return tile_exclusive_prefix + tile_num_selected;
    }


    /**
     * Dequeue and scan tiles of items as part of a dynamic domino scan
     */
    template <typename NumSelectedIterator>         ///< Output iterator type for recording number of items selected
    __device__ __forceinline__ void ConsumeRegion(
        int                     num_tiles,          ///< Total number of input tiles
        GridQueue<int>          queue,              ///< Queue descriptor for assigning tiles of work to thread blocks
        TileDescriptor          *d_tile_status,     ///< Global list of tile status
        NumSelectedIterator     d_num_selected)     ///< Output total number selected
    {
#if CUB_PTX_VERSION < 200

        // No concurrent kernels are allowed, and blocks are launched in increasing order, so just assign one tile per block (up to 65K blocks)
        int     tile_idx        = blockIdx.x;
        Offset  block_offset    = Offset(TILE_ITEMS) * tile_idx;
        Offset  total_selected;

        if (block_offset + TILE_ITEMS <= num_items)
            total_selected = ConsumeTile<true>(num_items, tile_idx, block_offset, d_tile_status);
        else if (block_offset < num_items)
            total_selected = ConsumeTile<false>(num_items, tile_idx, block_offset, d_tile_status);

        // Output the total number of items selected
        if ((tile_idx == num_tiles - 1) && (threadIdx.x == 0))
        {
            *d_num_selected = total_selected;
        }

#else

        // Get first tile
        if (threadIdx.x == 0)
            temp_storage.tile_idx = queue.Drain(1);

        __syncthreads();

        int tile_idx = temp_storage.tile_idx;

        while (tile_idx < num_tiles - 1)
        {
            Offset block_offset = Offset(TILE_ITEMS) * tile_idx;

            // Consume full tile
            ConsumeTile<true>(num_items, tile_idx, block_offset, d_tile_status);

            // Get next tile
            if (threadIdx.x == 0)
                temp_storage.tile_idx = queue.Drain(1);

            __syncthreads();

            tile_idx = temp_storage.tile_idx;
        }

        // Consume the last tile and output the total number of items selected
        if (tile_idx == num_tiles - 1)
        {
            Offset block_offset     = Offset(TILE_ITEMS) * tile_idx;
            Offset total_selected   = ConsumeTile<false>(num_items, tile_idx, block_offset, d_tile_status);

            if (threadIdx.x == 0)
            {
                *d_num_selected = total_selected;
            }
        }

#endif
    }

};


}               // CUB namespace
CUB_NS_POSTFIX  // Optional outer namespace(s)

