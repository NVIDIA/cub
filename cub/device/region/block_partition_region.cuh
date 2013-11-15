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
 * cub::BlockPartitionRegion implements a stateful abstraction of CUDA thread blocks for participating in device-wide partition.
 */

#pragma once

#include <iterator>

#include "device_scan_types.cuh"
#include "../../block/block_load.cuh"
#include "../../block/block_store.cuh"
#include "../../block/block_scan.cuh"
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
 * Parameterizable tuning policy type for BlockPartitionRegion
 */
template <
    int                         _BLOCK_THREADS,                 ///< Threads per thread block
    int                         _ITEMS_PER_THREAD,              ///< Items per thread (per tile of input)
    BlockLoadAlgorithm          _LOAD_ALGORITHM,                ///< The BlockLoad algorithm to use
    bool                        _LOAD_WARP_TIME_SLICING,        ///< Whether or not only one warp's worth of shared memory should be allocated and time-sliced among block-warps during any load-related data transpositions (versus each warp having its own storage)
    CacheLoadModifier           _LOAD_MODIFIER,                 ///< Cache load modifier for reading input elements
    bool                        _TWO_PHASE_SCATTER,             ///< Whether or not to coalesce output values in shared memory before scattering them to global
    BlockScanAlgorithm          _SCAN_ALGORITHM>                ///< The BlockScan algorithm to use
struct BlockPartitionRegionPolicy
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
 * \brief BlockPartitionRegion implements a stateful abstraction of CUDA thread blocks for participating in device-wide partition.
 */
template <
    typename BlockPartitionRegionPolicy,    ///< Parameterized BlockPartitionRegionPolicy tuning policy type
    typename InputIterator,                 ///< Random-access input iterator type
    typename OutputIterator,                ///< Random-access output iterator type
    typename SelectOp,                      ///< Selection operator type
    typename OffsetTuple>                   ///< Signed integer tuple type for global offsets
struct BlockPartitionRegion
{
    //---------------------------------------------------------------------
    // Types and constants
    //---------------------------------------------------------------------

    // Data type of input iterator
    typedef typename std::iterator_traits<InputIterator>::value_type T;

    // Signed integer type for global offsets
    typedef typename OffsetTuple::BaseType Offset;

    // Input iterator wrapper type
    typedef typename If<IsPointer<InputIterator>::VALUE,
            CacheModifiedInputIterator<BlockPartitionRegionPolicy::LOAD_MODIFIER, T, Offset>,   // Wrap the native input pointer with CacheModifiedInputIterator
            InputIterator>::Type                                                                // Directly use the supplied input iterator type
        WrappedInputIterator;

    // Constants
    enum
    {
        BLOCK_THREADS       = BlockPartitionRegionPolicy::BLOCK_THREADS,
        ITEMS_PER_THREAD    = BlockPartitionRegionPolicy::ITEMS_PER_THREAD,
        TWO_PHASE_SCATTER   = BlockPartitionRegionPolicy::TWO_PHASE_SCATTER,
        TILE_ITEMS          = BLOCK_THREADS * ITEMS_PER_THREAD,
        PIVOT_REJECTS       = sizeof(OffsetTuple) > sizeof(Offset),               // Whether or not we push rejected items to the back of the output
    };

    // Parameterized BlockLoad type
    typedef BlockLoad<
            WrappedInputIterator,
            BlockPartitionRegionPolicy::BLOCK_THREADS,
            BlockPartitionRegionPolicy::ITEMS_PER_THREAD,
            BlockPartitionRegionPolicy::LOAD_ALGORITHM,
            BlockPartitionRegionPolicy::LOAD_WARP_TIME_SLICING>
        BlockLoad;

    // Parameterized BlockExchange type
    typedef BlockExchange<
            T,
            BLOCK_THREADS,
            ITEMS_PER_THREAD>
        BlockExchange;

    // Tile status descriptor type
    typedef LookbackTileDescriptor<OffsetTuple> TileDescriptor;

    // Parameterized BlockScan type
    typedef BlockScan<
            OffsetTuple,
            BlockPartitionRegionPolicy::BLOCK_THREADS,
            BlockPartitionRegionPolicy::SCAN_ALGORITHM>
        BlockScan;

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
                typename BlockScan::TempStorage                 scan;       // Smem needed for tile scanning
            };

            // Smem needed for tile loading
            typename BlockLoad::TempStorage load;

            // Smem needed for two-phase scatter
            typename If<TWO_PHASE_SCATTER, typename BlockExchange::TempStorage, NullType>::Type exchange;
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
    OutputIterator              d_out;              ///< Output data
    SelectOp                    select_op;          ///< Selection operator
    Offset                      num_items;          ///< Total number of input items


    //---------------------------------------------------------------------
    // Constructor
    //---------------------------------------------------------------------

    // Constructor
    __device__ __forceinline__
    BlockPartitionRegion(
        TempStorage                 &temp_storage,      ///< Reference to temp_storage
        InputIterator               d_in,               ///< Input data
        OutputIterator              d_out,              ///< Output data
        SelectOp                    select_op,          ///< Selection operator
        Offset                      num_items)          ///< Total number of input items
    :
        temp_storage(temp_storage.Alias()),
        d_in(d_in),
        d_out(d_out),
        select_op(select_op),
        num_items(num_items)
    {}


    //---------------------------------------------------------------------
    // Allocation initialization methods
    //---------------------------------------------------------------------

    /**
     * Initialize partition allocations (specialized for pushing rejected items to the back of the output)
     */
    template <bool FULL_TILE>
    __device__ __forceinline__ void InitializeAllocations(
        T               items[ITEMS_PER_THREAD],
        OffsetTuple       partition_allocations[ITEMS_PER_THREAD],
        int             valid_items,
        Int2Type<true>  pivot_rejects)
    {
        #pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
        {
            if (FULL_TILE || ((threadIdx.x * ITEMS_PER_THREAD) + ITEM < valid_items))
            {
                partition_allocations[ITEM].x = select_op(items[ITEM]);
            }
        }
    }


    /**
     * Initialize partition allocations (specialized for pushing rejected items to the back of the output)
     */
    template <bool FULL_TILE>
    __device__ __forceinline__ void InitializeAllocations(
        T               items[ITEMS_PER_THREAD],
        OffsetTuple       partition_allocations[ITEMS_PER_THREAD],
        int             valid_items,
        Int2Type<false> pivot_rejects)
    {
        #pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
        {
            if (FULL_TILE || ((threadIdx.x * ITEMS_PER_THREAD) + ITEM < valid_items))
            {
                partition_allocations[ITEM].x = select_op(items[ITEM]);
                partition_allocations[ITEM].y = !partition_allocations[ITEM].x;
            }
        }

    }


    //---------------------------------------------------------------------
    // Scatter utility methods
    //---------------------------------------------------------------------

    /**
     * Scatter data items to partition offsets (specialized for drop-rejects, direct-scattering)
     */
    __device__ __forceinline__ void Scatter(
        T               items[ITEMS_PER_THREAD],
        OffsetTuple       partition_allocations[ITEMS_PER_THREAD],
        OffsetTuple       partition_offsets[ITEMS_PER_THREAD],
        OffsetTuple       tile_offset,
        OffsetTuple       valid_items,
        Int2Type<false> pivot_rejects,
        Int2Type<false> two_phase_scatter)
    {
        #pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
        {
            // Selected items insert front-to-back
            if (partition_allocations[ITEM].x)
            {
                d_out[partition_offsets[ITEM].x] = items[ITEM];
            }
        }
    }


    /**
     * Scatter data items to partition offsets (specialized for drop-rejects, two-phase-scattering)
     */
    __device__ __forceinline__ void Scatter(
        T               items[ITEMS_PER_THREAD],
        OffsetTuple       partition_allocations[ITEMS_PER_THREAD],
        OffsetTuple       partition_offsets[ITEMS_PER_THREAD],
        OffsetTuple       tile_offset,
        OffsetTuple       valid_items,
        Int2Type<false> pivot_rejects,
        Int2Type<true>  two_phase_scatter)
    {
        Offset local_ranks[ITEMS_PER_THREAD];
        Offset is_valid[ITEMS_PER_THREAD];

        #pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
        {
            local_ranks[ITEM]   = partition_offsets[ITEM].x - tile_offset.x;
            is_valid[ITEM]      = partition_allocations[ITEM].x;
        }

        __syncthreads();

        BlockExchange(temp_storage.exchange).ScatterToStriped(items, local_ranks, is_valid, valid_items.x);

        // Selected items insert front-to-back
        StoreStriped<BLOCK_THREADS>(threadIdx.x, d_out + tile_offset.x, items, valid_items.x);
    }


    /**
     * Scatter data items to partition offsets (specialized for keep-rejects and direct-scattering)
     */
    __device__ __forceinline__ void Scatter(
        T               items[ITEMS_PER_THREAD],
        OffsetTuple       partition_allocations[ITEMS_PER_THREAD],
        OffsetTuple       partition_offsets[ITEMS_PER_THREAD],
        OffsetTuple       tile_offset,
        OffsetTuple       valid_items,
        Int2Type<true>  pivot_rejects,
        Int2Type<false> two_phase_scatter)
    {
        #pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
        {
            // Selected items insert front-to-back
            if (partition_allocations[ITEM].x)
            {
                d_out[partition_offsets[ITEM].x] = items[ITEM];
            }

            // Rejected items insert back-to-front
            if (partition_allocations[ITEM].y)
            {
                d_out[num_items - partition_offsets[ITEM].y] = items[ITEM];
            }
        }
    }


    /**
     * Scatter data items to partition offsets (specialized for keep-rejects and two-phase-scattering)
     */
    __device__ __forceinline__ void Scatter(
        T               items[ITEMS_PER_THREAD],
        OffsetTuple       partition_allocations[ITEMS_PER_THREAD],
        OffsetTuple       partition_offsets[ITEMS_PER_THREAD],
        OffsetTuple       tile_offset,
        OffsetTuple       valid_items,
        Int2Type<true>  pivot_rejects,
        Int2Type<true>  two_phase_scatter)
    {
        Offset local_ranks[ITEMS_PER_THREAD];
        Offset is_valid[ITEMS_PER_THREAD];

        #pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
        {
            local_ranks[ITEM]   = partition_offsets[ITEM].x - tile_offset.x;
            is_valid[ITEM]      = partition_allocations[ITEM].x;

            if (partition_allocations[ITEM].y)
            {
                local_ranks[ITEM]   = partition_offsets[ITEM].y - tile_offset.y + valid_items.x;
                is_valid[ITEM]      = 1;
            }
        }

        __syncthreads();

        // Coalesce selected and rejected items in shared memory, gathering in striped arrangements
        BlockExchange(temp_storage.exchange).ScatterToStriped(items, local_ranks, is_valid, valid_items);

        // Store in striped order
        #pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
        {
            Offset item_offset = (ITEM * BLOCK_THREADS) + threadIdx.x;
            if (item_offset < valid_items.x)
            {
                // Scatter selected items front-to-back
                d_out[tile_offset.x + item_offset] = items[ITEM];
            }
            else
            {
                item_offset -= valid_items.x;
                if (item_offset < valid_items.y)
                {
                    // Scatter rejected items back-to-front
                    d_out[num_items - (tile_offset.y + item_offset)] = items[ITEM];
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
    __device__ __forceinline__ void ConsumeTile(
        Offset                      num_items,          ///< Total number of input items
        int                         tile_idx,           ///< Tile index
        Offset                      block_offset,       ///< Tile offset
        TileDescriptor              *d_tile_status)     ///< Global list of tile status
    {
        int valid_items = num_items - block_offset;

        // Load items
        T items[ITEMS_PER_THREAD];

        if (FULL_TILE)
            BlockLoad(temp_storage.load).Load(d_in + block_offset, items);
        else
            BlockLoad(temp_storage.load).Load(d_in + block_offset, items, valid_items);

        __syncthreads();

        // Initialize partition allocations
        OffsetTuple partition_allocations[ITEMS_PER_THREAD];
        #pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
        {
            // Zero-initialize
            partition_allocations[ITEM] = OffsetTuple();
        }
        InitializeAllocations<FULL_TILE>(items, partition_allocations, valid_items, Int2Type<PIVOT_REJECTS>());

        // Compute partition offsets
        OffsetTuple partition_offsets[ITEMS_PER_THREAD];

        OffsetTuple tile_offset;          // Prefix offset in each partition
        OffsetTuple block_aggregate;      // Total number of items in each partition
        if (tile_idx == 0)
        {
            // Scan first tile
            BlockScan(temp_storage.scan).ExclusiveSum(partition_allocations, partition_offsets, block_aggregate);
            tile_offset = OffsetTuple();

            // Update tile status if this tile is full (i.e., there may be successor tiles)
            if (FULL_TILE && (threadIdx.x == 0))
                TileDescriptor::SetPrefix(d_tile_status, block_aggregate);
        }
        else
        {
            LookbackPrefixCallbackOp prefix_op(d_tile_status, temp_storage.prefix, Sum(), tile_idx);
            BlockScan(temp_storage.scan).ExclusiveSum(partition_allocations, partition_offsets, block_aggregate, prefix_op);
            tile_offset = prefix_op.exclusive_prefix;
        }

        __syncthreads();

        // Store selected items
        Scatter(
            items,
            partition_allocations,
            partition_offsets,
            tile_offset,
            block_aggregate,
            Int2Type<PIVOT_REJECTS>(),
            Int2Type<TWO_PHASE_SCATTER>());
    }


    /**
     * Dequeue and scan tiles of items as part of a dynamic domino scan
     */
    __device__ __forceinline__ void ConsumeRegion(
        int                     num_items,          ///< Total number of input items
        GridQueue<int>          queue,              ///< Queue descriptor for assigning tiles of work to thread blocks
        TileDescriptor          *d_tile_status)     ///< Global list of tile status
    {
#if CUB_PTX_VERSION < 200

        // No concurrent kernels allowed and blocks are launched in increasing order, so just assign one tile per block (up to 65K blocks)
        int     tile_idx        = blockIdx.x;
        Offset   block_offset    = Offset(TILE_ITEMS) * tile_idx;

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
        Offset block_offset = Offset(TILE_ITEMS) * tile_idx;

        while (block_offset + TILE_ITEMS <= num_items)
        {
            // Consume full tile
            ConsumeTile<true>(num_items, tile_idx, block_offset, d_tile_status);

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
            ConsumeTile<false>(num_items, tile_idx, block_offset, d_tile_status);
        }
#endif

    }

};


}               // CUB namespace
CUB_NS_POSTFIX  // Optional outer namespace(s)

