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
 * cub::AgentSelectIf implements a stateful abstraction of CUDA thread blocks for participating in device-wide select.
 */

#pragma once

#include <iterator>

#include "single_pass_scan_operators.cuh"
#include "../block/block_load.cuh"
#include "../block/block_store.cuh"
#include "../block/block_scan.cuh"
#include "../block/block_exchange.cuh"
#include "../block/block_discontinuity.cuh"
#include "../grid/grid_queue.cuh"
#include "../iterator/cache_modified_input_iterator.cuh"
#include "../util_namespace.cuh"

/// Optional outer namespace(s)
CUB_NS_PREFIX

/// CUB namespace
namespace cub {


/******************************************************************************
 * Tuning policy types
 ******************************************************************************/

/**
 * Parameterizable tuning policy type for AgentSelectIf
 */
template <
    int                         _BLOCK_THREADS,                 ///< Threads per thread block
    int                         _ITEMS_PER_THREAD,              ///< Items per thread (per tile of input)
    BlockLoadAlgorithm          _LOAD_ALGORITHM,                ///< The BlockLoad algorithm to use
    CacheLoadModifier           _LOAD_MODIFIER,                 ///< Cache load modifier for reading input elements
    BlockScanAlgorithm          _SCAN_ALGORITHM>                ///< The BlockScan algorithm to use
struct AgentSelectIfPolicy
{
    enum
    {
        BLOCK_THREADS           = _BLOCK_THREADS,               ///< Threads per thread block
        ITEMS_PER_THREAD        = _ITEMS_PER_THREAD,            ///< Items per thread (per tile of input)
    };

    static const BlockLoadAlgorithm     LOAD_ALGORITHM          = _LOAD_ALGORITHM;      ///< The BlockLoad algorithm to use
    static const CacheLoadModifier      LOAD_MODIFIER           = _LOAD_MODIFIER;       ///< Cache load modifier for reading input elements
    static const BlockScanAlgorithm     SCAN_ALGORITHM          = _SCAN_ALGORITHM;      ///< The BlockScan algorithm to use
};




/******************************************************************************
 * Thread block abstractions
 ******************************************************************************/


/**
 * \brief AgentSelectIf implements a stateful abstraction of CUDA thread blocks for participating in device-wide selection
 *
 * Performs functor-based selection if SelectOpT functor type != NullType
 * Otherwise performs flag-based selection if FlagsInputIterator's value type != NullType
 * Otherwise performs discontinuity selection (keep unique)
 */
template <
    typename    AgentSelectIfPolicyT,           ///< Parameterized AgentSelectIfPolicy tuning policy type
    typename    InputIteratorT,                 ///< Random-access input iterator type for selection items
    typename    FlagsInputIteratorT,            ///< Random-access input iterator type for selections (NullType* if a selection functor or discontinuity flagging is to be used for selection)
    typename    SelectedOutputIteratorT,        ///< Random-access input iterator type for selection_flags items
    typename    SelectOpT,                      ///< Selection operator type (NullType if selections or discontinuity flagging is to be used for selection)
    typename    EqualityOpT,                    ///< Equality operator type (NullType if selection functor or selections is to be used for selection)
    typename    OffsetT,                        ///< Signed integer type for global offsets
    bool        KEEP_REJECTS>                   ///< Whether or not we push rejected items to the back of the output
struct AgentSelectIf
{
    //---------------------------------------------------------------------
    // Types and constants
    //---------------------------------------------------------------------

    // Data type of input iterator
    typedef typename std::iterator_traits<InputIteratorT>::value_type T;

    // Data type of flag iterator
    typedef typename std::iterator_traits<FlagsInputIteratorT>::value_type FlagT;

    // Tile status descriptor interface type
    typedef ScanTileState<OffsetT> ScanTileStateT;

    // Constants
    enum
    {
        USE_SELECT_OP,
        USE_SELECT_FLAGS,
        USE_DISCONTINUITY,

        BLOCK_THREADS           = AgentSelectIfPolicyT::BLOCK_THREADS,
        ITEMS_PER_THREAD        = AgentSelectIfPolicyT::ITEMS_PER_THREAD,
        TILE_ITEMS              = BLOCK_THREADS * ITEMS_PER_THREAD,
        TWO_PHASE_SCATTER       = (ITEMS_PER_THREAD > 1),

        SELECT_METHOD           = (!Equals<SelectOpT, NullType>::VALUE) ?
                                    USE_SELECT_OP :
                                    (!Equals<FlagT, NullType>::VALUE) ?
                                        USE_SELECT_FLAGS :
                                        USE_DISCONTINUITY
    };

    // Cache-modified Input iterator wrapper type (for applying cache modifier) for items
    typedef typename If<IsPointer<InputIteratorT>::VALUE,
            CacheModifiedInputIterator<AgentSelectIfPolicyT::LOAD_MODIFIER, T, OffsetT>,        // Wrap the native input pointer with CacheModifiedValuesInputIterator
            InputIteratorT>::Type                                                               // Directly use the supplied input iterator type
        WrappedInputIteratorT;

    // Cache-modified Input iterator wrapper type (for applying cache modifier) for values
    typedef typename If<IsPointer<FlagsInputIteratorT>::VALUE,
            CacheModifiedInputIterator<AgentSelectIfPolicyT::LOAD_MODIFIER, FlagT, OffsetT>,    // Wrap the native input pointer with CacheModifiedValuesInputIterator
            FlagsInputIteratorT>::Type                                                          // Directly use the supplied input iterator type
        WrappedFlagsInputIteratorT;

    // Parameterized BlockLoad type for input data
    typedef BlockLoad<
            WrappedInputIteratorT,
            BLOCK_THREADS,
            ITEMS_PER_THREAD,
            AgentSelectIfPolicyT::LOAD_ALGORITHM>
        BlockLoadT;

    // Parameterized BlockLoad type for flags
    typedef BlockLoad<
            WrappedFlagsInputIteratorT,
            BLOCK_THREADS,
            ITEMS_PER_THREAD,
            AgentSelectIfPolicyT::LOAD_ALGORITHM>
        BlockLoadFlags;

    // Parameterized BlockDiscontinuity type for items
    typedef BlockDiscontinuity<
            T,
            BLOCK_THREADS>
        BlockDiscontinuityT;

    // Parameterized BlockScan type
    typedef BlockScan<
            OffsetT,
            BLOCK_THREADS,
            AgentSelectIfPolicyT::SCAN_ALGORITHM>
        BlockScanT;

    // Callback type for obtaining tile prefix during block scan
    typedef TilePrefixCallbackOp<
            OffsetT,
            cub::Sum,
            ScanTileStateT>
        TilePrefixCallbackOpT;

    // Shared memory type for this threadblock
    union _TempStorage
    {
        struct
        {
            typename BlockScanT::TempStorage                scan;           // Smem needed for tile scanning
            typename TilePrefixCallbackOpT::TempStorage     prefix;         // Smem needed for cooperative prefix callback
            typename BlockDiscontinuityT::TempStorage       discontinuity;  // Smem needed for discontinuity detection
        };

        // Smem needed for loading items
        typename BlockLoadT::TempStorage load_items;

        // Smem needed for loading values
        typename BlockLoadFlags::TempStorage load_flags;

        // Smem needed for compacting items
        T exchange_items[TILE_ITEMS];
    };

    // Alias wrapper allowing storage to be unioned
    struct TempStorage : Uninitialized<_TempStorage> {};


    //---------------------------------------------------------------------
    // Per-thread fields
    //---------------------------------------------------------------------

    _TempStorage&                   temp_storage;       ///< Reference to temp_storage
    WrappedInputIteratorT           d_in;               ///< Input items
    SelectedOutputIteratorT         d_selected_out;     ///< Unique output items
    WrappedFlagsInputIteratorT      d_flags_in;         ///< Input values
    InequalityWrapper<EqualityOpT>  inequality_op;      ///< T inequality operator
    SelectOpT                       select_op;          ///< Selection operator


    //---------------------------------------------------------------------
    // Constructor
    //---------------------------------------------------------------------

    // Constructor
    __device__ __forceinline__
    AgentSelectIf(
        TempStorage                 &temp_storage,      ///< Reference to temp_storage
        InputIteratorT              d_in,               ///< Input data
        FlagsInputIteratorT         d_flags_in,         ///< Input flags
        SelectedOutputIteratorT     d_selected_out,     ///< Output data
        SelectOpT                   select_op,          ///< Selection operator
        EqualityOpT                 equality_op)        ///< Equality operator
    :
        temp_storage(temp_storage.Alias()),
        d_in(d_in),
        d_flags_in(d_flags_in),
        d_selected_out(d_selected_out),
        select_op(select_op),
        inequality_op(equality_op)
    {}


    //---------------------------------------------------------------------
    // Utility methods for initializing the selections
    //---------------------------------------------------------------------

    /**
     * Initialize selections (specialized for selection operator)
     */
    template <bool IS_FIRST_TILE, bool IS_LAST_TILE>
    __device__ __forceinline__ void InitializeSelections(
        OffsetT                     tile_offset,
        OffsetT                     num_tile_items,
        T                           (&items)[ITEMS_PER_THREAD],
        OffsetT                     (&selection_flags)[ITEMS_PER_THREAD],
        Int2Type<USE_SELECT_OP>     select_method)
    {
        #pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
        {
            // Out-of-bounds items are selection_flags
            selection_flags[ITEM] = 1;

            if (!IS_LAST_TILE || (OffsetT(threadIdx.x * ITEMS_PER_THREAD) + ITEM < num_tile_items))
                selection_flags[ITEM] = select_op(items[ITEM]);
        }
    }


    /**
     * Initialize selections (specialized for valid flags)
     */
    template <bool IS_FIRST_TILE, bool IS_LAST_TILE>
    __device__ __forceinline__ void InitializeSelections(
        OffsetT                     tile_offset,
        OffsetT                     num_tile_items,
        T                           (&items)[ITEMS_PER_THREAD],
        OffsetT                     (&selection_flags)[ITEMS_PER_THREAD],
        Int2Type<USE_SELECT_FLAGS>  select_method)
    {
        __syncthreads();

        FlagT flags[ITEMS_PER_THREAD];

        if (IS_LAST_TILE)
        {
            // Out-of-bounds items are selection_flags
            BlockLoadFlags(temp_storage.load_flags).Load(d_flags_in + tile_offset, flags, num_tile_items, 1);
        }
        else
        {
            BlockLoadFlags(temp_storage.load_flags).Load(d_flags_in + tile_offset, flags);
        }

        // Convert flag type to selection_flags type
        #pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
        {
            selection_flags[ITEM] = flags[ITEM];
        }
    }


    /**
     * Initialize selections (specialized for discontinuity detection)
     */
    template <bool IS_FIRST_TILE, bool IS_LAST_TILE>
    __device__ __forceinline__ void InitializeSelections(
        OffsetT                     tile_offset,
        OffsetT                     num_tile_items,
        T                           (&items)[ITEMS_PER_THREAD],
        OffsetT                     (&selection_flags)[ITEMS_PER_THREAD],
        Int2Type<USE_DISCONTINUITY> select_method)
    {
        if (IS_FIRST_TILE)
        {
            __syncthreads();

            // Set head selection_flags.  First tile sets the first flag for the first item
            BlockDiscontinuityT(temp_storage.discontinuity).FlagHeads(selection_flags, items, inequality_op);
        }
        else
        {
            T tile_predecessor = (threadIdx.x == 0) ?
                d_in[tile_offset - 1] :
                ZeroInitialize<T>();

            __syncthreads();

            BlockDiscontinuityT(temp_storage.discontinuity).FlagHeads(selection_flags, items, inequality_op, tile_predecessor);
        }

        // Set selection flags for out-of-bounds items
        #pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
        {
            // Set selection_flags for out-of-bounds items
            if ((IS_LAST_TILE) && (OffsetT(threadIdx.x * ITEMS_PER_THREAD) + ITEM >= num_tile_items))
                selection_flags[ITEM] = 1;
        }
    }


    //---------------------------------------------------------------------
    // Scatter utility methods
    //---------------------------------------------------------------------

    /**
     * Scatter flagged items to output offsets (specialized for direct scattering)
     */
    template <bool IS_LAST_TILE, bool IS_FIRST_TILE>
    __device__ __forceinline__ void ScatterDirect(
        T       (&items)[ITEMS_PER_THREAD],
        OffsetT (&selection_flags)[ITEMS_PER_THREAD],
        OffsetT (&selection_indices)[ITEMS_PER_THREAD],
        OffsetT num_selections)
    {
        // Scatter flagged items
        #pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
        {
            if (selection_flags[ITEM])
            {
                if ((!IS_LAST_TILE) || selection_indices[ITEM] < num_selections)
                {
                    d_selected_out[selection_indices[ITEM]] = items[ITEM];
                }
            }
        }
    }


    /**
     * Scatter flagged items to output offsets (specialized for two-phase scattering)
     */
    template <bool IS_LAST_TILE, bool IS_FIRST_TILE>
    __device__ __forceinline__ void ScatterTwoPhase(
        T               (&items)[ITEMS_PER_THREAD],
        OffsetT         (&selection_flags)[ITEMS_PER_THREAD],
        OffsetT         (&selection_indices)[ITEMS_PER_THREAD],
        OffsetT         num_tile_selections,
        OffsetT         num_tile_selections_prefix,
        int             num_tile_items,
        Int2Type<false> is_keep_rejects
        )
    {
        __syncthreads();

        // Compact and scatter items
        #pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
        {
            if (selection_flags[ITEM])
            {
                temp_storage.exchange_items[selection_indices[ITEM] - num_tile_selections_prefix] = items[ITEM];
            }
        }

        __syncthreads();

        for (int item = threadIdx.x; item < num_tile_selections; item += BLOCK_THREADS)
        {
            d_selected_out[num_tile_selections_prefix + item] = temp_storage.exchange_items[item];
        }
    }


    /**
     * Scatter flagged items to output offsets (specialized for two-phase scattering)
     */
    template <bool IS_LAST_TILE, bool IS_FIRST_TILE>
    __device__ __forceinline__ void ScatterTwoPhase(
        T               (&items)[ITEMS_PER_THREAD],
        OffsetT         (&selection_flags)[ITEMS_PER_THREAD],
        OffsetT         (&selection_indices)[ITEMS_PER_THREAD],
        OffsetT         num_tile_selections,
        OffsetT         num_tile_selections_prefix,
        int             num_tile_items,
        Int2Type<true>  is_keep_rejects)
    {
        __syncthreads();

        // Compact and scatter items
        #pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
        {
            int item = (threadIdx.x * ITEMS_PER_THREAD) + ITEM;
            int scatter_offset = selection_indices[ITEM] - num_tile_selections_prefix;
            if (selection_flags[ITEM])
                scatter_offset = num_tile_selections + (item - scatter_offset);

            temp_storage.exchange_items[scatter_offset] = items[ITEM];
        }

        __syncthreads();

        for (int item = threadIdx.x; item < num_tile_selections; item += BLOCK_THREADS)
        {
            d_selected_out[num_tile_selections_prefix + item] = temp_storage.exchange_items[item];
        }

        OffsetT num_rejected = (tile_idx * TILE_ITEMS)

        for (int item = num_tile_selections + threadIdx.x; item < num_tile_items; item += BLOCK_THREADS)
        {
            d_selected_out[num_items - num_tile_selections_prefix + item] = temp_storage.exchange_items[item];
        }

    }


    /**
     * Scatter flagged items
     */
    template <bool IS_LAST_TILE, bool IS_FIRST_TILE>
    __device__ __forceinline__ void Scatter(
        T               (&items)[ITEMS_PER_THREAD],
        OffsetT         (&selection_flags)[ITEMS_PER_THREAD],
        OffsetT         (&selection_indices)[ITEMS_PER_THREAD],
        OffsetT         num_tile_selections,
        OffsetT         num_tile_selections_prefix,
        int             num_tile_items,
        OffsetT         num_selections,
)
    {
        // Do a two-phase scatter if (a) keeping both partitions or (b) two-phase is enabled and the average number of selection_flags items per thread is greater than one
        if (KEEP_REJECTS || (TWO_PHASE_SCATTER && (num_tile_selections > BLOCK_THREADS)))
        {
            ScatterTwoPhase<IS_LAST_TILE, IS_FIRST_TILE>(
                items,
                selection_flags,
                selection_indices,
                num_tile_selections,
                num_tile_selections_prefix,
                num_tile_items,
                Int2Type<KEEP_REJECTS>());
        }
        else
        {
            ScatterDirect<IS_LAST_TILE, IS_FIRST_TILE>(
                items,
                selection_flags,
                selection_indices,
                num_selections);
        }
    }

    //---------------------------------------------------------------------
    // Cooperatively scan a device-wide sequence of tiles with other CTAs
    //---------------------------------------------------------------------


    /**
     * Process first tile of input (dynamic chained scan).  Returns the running count of selections (including this tile)
     */
    template <bool IS_LAST_TILE>
    __device__ __forceinline__ OffsetT ConsumeFirstTile(
        int                 num_tile_items,      ///< Number of input items comprising this tile
        OffsetT             tile_offset,        ///< Tile offset
        ScanTileStateT&     tile_state)         ///< Global tile state descriptor
    {
        T           items[ITEMS_PER_THREAD];
        OffsetT     selection_flags[ITEMS_PER_THREAD];
        OffsetT     selection_indices[ITEMS_PER_THREAD];

        // Load items
        if (IS_LAST_TILE)
            BlockLoadT(temp_storage.load_items).Load(d_in + tile_offset, items, num_tile_items);
        else
            BlockLoadT(temp_storage.load_items).Load(d_in + tile_offset, items);

        // Initialize selection_flags
        InitializeSelections<true, IS_LAST_TILE>(
            tile_offset,
            num_tile_items,
            items,
            selection_flags,
            Int2Type<SELECT_METHOD>());

        __syncthreads();

        // Exclusive scan of selection_flags
        OffsetT tile_aggregate;
        BlockScanT(temp_storage.scan).ExclusiveSum(selection_flags, selection_indices, tile_aggregate);

        if (threadIdx.x == 0)
        {
            // Update tile status if this is not the last tile
            if (!IS_LAST_TILE)
                tile_state.SetInclusive(0, tile_aggregate);
        }

        // Discount any out-of-bounds selections
        if (IS_LAST_TILE)
            tile_aggregate -= (TILE_ITEMS - num_tile_items);

        // Scatter flagged items
        Scatter<IS_LAST_TILE, true>(
            items,
            selection_flags,
            selection_indices,
            tile_aggregate,
            0,
            num_tile_items,
            tile_aggregate)

        return tile_aggregate;
    }


    /**
     * Process subsequent tile of input (dynamic chained scan).  Returns the running count of selections (including this tile)
     */
    template <bool IS_LAST_TILE>
    __device__ __forceinline__ OffsetT ConsumeSubsequentTile(
        int                 num_tile_items,      ///< Number of input items comprising this tile
        int                 tile_idx,           ///< Tile index
        OffsetT             tile_offset,        ///< Tile offset
        ScanTileStateT&     tile_state)         ///< Global tile state descriptor
    {
        T           items[ITEMS_PER_THREAD];
        OffsetT     selection_flags[ITEMS_PER_THREAD];
        OffsetT     selection_indices[ITEMS_PER_THREAD];

        // Load items
        if (IS_LAST_TILE)
            BlockLoadT(temp_storage.load_items).Load(d_in + tile_offset, items, num_tile_items);
        else
            BlockLoadT(temp_storage.load_items).Load(d_in + tile_offset, items);

        // Initialize selection_flags
        InitializeSelections<false, IS_LAST_TILE>(
            tile_offset,
            num_tile_items,
            items,
            selection_flags,
            Int2Type<SELECT_METHOD>());

        __syncthreads();

        // Exclusive scan of values and selection_flags
        OffsetT tile_aggregate;
        TilePrefixCallbackOpT prefix_op(tile_state, temp_storage.prefix, cub::Sum(), tile_idx);
        BlockScanT(temp_storage.scan).ExclusiveSum(selection_flags, selection_indices, tile_aggregate, prefix_op);
        OffsetT tile_inclusive_prefix = prefix_op.GetInclusivePrefix();

        // Discount any out-of-bounds selections
        if (IS_LAST_TILE)
        {
            int num_discount        = TILE_ITEMS - num_tile_items;
            tile_inclusive_prefix   -= num_discount;
            tile_aggregate          -= num_discount;
        }

        // Scatter flagged items
        Scatter<IS_LAST_TILE, false>(
            items,
            selection_flags,
            selection_indices,
            tile_aggregate,
            prefix_op.GetExclusivePrefix(),
            num_tile_items,
            tile_inclusive_prefix)

        return tile_inclusive_prefix;
    }


    /**
     * Process a tile of input
     */
    template <bool IS_LAST_TILE>
    __device__ __forceinline__ OffsetT ConsumeTile(
        int                 num_tile_items,         ///< Number of input items comprising this tile
        int                 tile_idx,           ///< Tile index
        OffsetT             tile_offset,        ///< Tile offset
        ScanTileStateT&     tile_state)         ///< Global tile state descriptor
    {
        OffsetT tile_inclusive_prefix;
        if (tile_idx == 0)
        {
            tile_inclusive_prefix = ConsumeFirstTile<IS_LAST_TILE>(num_tile_items, tile_offset, tile_state);
        }
        else
        {
            tile_inclusive_prefix = ConsumeSubsequentTile<IS_LAST_TILE>(num_tile_items, tile_idx, tile_offset, tile_state);
        }

        return tile_inclusive_prefix;
    }


    /**
     * Scan tiles of items as part of a dynamic chained scan
     */
    template <typename NumSelectedIteratorT>        ///< Output iterator type for recording number of items selection_flags
    __device__ __forceinline__ void ConsumeRange(
        int                     num_items,          ///< Total number of input items
        int                     num_tiles,          ///< Total number of input tiles
        ScanTileStateT&         tile_state,         ///< Global tile state descriptor
        NumSelectedIteratorT    d_num_selected_out) ///< Output total number selection_flags
    {
        // Blocks are launched in increasing order, so just assign one tile per block
        int     tile_idx        = (blockIdx.y * gridDim.x) + blockIdx.x;    // Current tile index
        OffsetT tile_offset     = tile_idx * TILE_ITEMS;                    // Global offset for the current tile

        if (tile_idx < num_tiles - 1)
        {
            // Not the last tile (full)
            ConsumeTile<false>(TILE_ITEMS, tile_idx, tile_offset, tile_state);
        }
        else
        {
            // The last tile (possibly partially-full)
            OffsetT num_remaining           = num_items - tile_offset;
            OffsetT tile_inclusive_prefix   = ConsumeTile<true>(num_remaining, tile_idx, tile_offset, tile_state);

            if (threadIdx.x == 0)
            {
                // Output the total number of items selection_flags
                *d_num_selected_out = tile_inclusive_prefix;
            }
        }
    }

};



}               // CUB namespace
CUB_NS_POSTFIX  // Optional outer namespace(s)

