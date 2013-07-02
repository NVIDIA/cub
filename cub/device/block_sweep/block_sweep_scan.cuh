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
 * cub::BlockSweepScan implements a stateful abstraction of CUDA thread blocks for participating in device-wide prefix scan.
 */

#pragma once

#include <iterator>

#include "device_scan_types.cuh"
#include "../../block/block_load.cuh"
#include "../../block/block_store.cuh"
#include "../../block/block_scan.cuh"
#include "../../util_namespace.cuh"

/// Optional outer namespace(s)
CUB_NS_PREFIX

/// CUB namespace
namespace cub {


/******************************************************************************
 * Tuning policy types
 ******************************************************************************/

/**
 * Tuning policy for BlockSweepScan
 */
template <
    int                         _BLOCK_THREADS,
    int                         _ITEMS_PER_THREAD,
    BlockLoadAlgorithm          _LOAD_ALGORITHM,
    bool                        _LOAD_WARP_TIME_SLICING,
    PtxLoadModifier             _LOAD_MODIFIER,
    BlockStoreAlgorithm         _STORE_ALGORITHM,
    bool                        _STORE_WARP_TIME_SLICING,
    BlockScanAlgorithm          _SCAN_ALGORITHM>
struct BlockSweepScanPolicy
{
    enum
    {
        BLOCK_THREADS           = _BLOCK_THREADS,
        ITEMS_PER_THREAD        = _ITEMS_PER_THREAD,
        LOAD_WARP_TIME_SLICING  = _LOAD_WARP_TIME_SLICING,
        STORE_WARP_TIME_SLICING = _STORE_WARP_TIME_SLICING,
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
 * \brief BlockSweepScan implements a stateful abstraction of CUDA thread blocks for participating in device-wide prefix scan.
 */
template <
    typename BlockSweepScanPolicy,     ///< Tuning policy
    typename InputIteratorRA,               ///< Input iterator type
    typename OutputIteratorRA,              ///< Output iterator type
    typename ScanOp,                        ///< Scan functor type
    typename Identity,                      ///< Identity element type (cub::NullType for inclusive scan)
    typename SizeT>                         ///< Offset integer type
struct BlockSweepScan
{
    //---------------------------------------------------------------------
    // Types and constants
    //---------------------------------------------------------------------

    // Data type of input iterator
    typedef typename std::iterator_traits<InputIteratorRA>::value_type T;

    // Constants
    enum
    {
        INCLUSIVE           = Equals<Identity, NullType>::VALUE,            // Inclusive scan if no identity type is provided
        BLOCK_THREADS       = BlockSweepScanPolicy::BLOCK_THREADS,
        ITEMS_PER_THREAD    = BlockSweepScanPolicy::ITEMS_PER_THREAD,
        TILE_ITEMS          = BLOCK_THREADS * ITEMS_PER_THREAD,
    };

    // Block load type
    typedef BlockLoad<
        InputIteratorRA,
        BlockSweepScanPolicy::BLOCK_THREADS,
        BlockSweepScanPolicy::ITEMS_PER_THREAD,
        BlockSweepScanPolicy::LOAD_ALGORITHM,
        BlockSweepScanPolicy::LOAD_MODIFIER,
        BlockSweepScanPolicy::LOAD_WARP_TIME_SLICING>   BlockLoadT;

    // Block store type
    typedef BlockStore<
        OutputIteratorRA,
        BlockSweepScanPolicy::BLOCK_THREADS,
        BlockSweepScanPolicy::ITEMS_PER_THREAD,
        BlockSweepScanPolicy::STORE_ALGORITHM,
        STORE_DEFAULT,
        BlockSweepScanPolicy::STORE_WARP_TIME_SLICING>  BlockStoreT;

    // Device tile status descriptor type
    typedef DeviceScanTileDescriptor<T>                 DeviceScanTileDescriptorT;

    // Block scan type
    typedef BlockScan<
        T,
        BlockSweepScanPolicy::BLOCK_THREADS,
        BlockSweepScanPolicy::SCAN_ALGORITHM> BlockScanT;

    // Device scan prefix callback type for inter-block scans
    typedef DeviceScanBlockPrefixOp<T, ScanOp> InterblockPrefixOp;

    // Running scan prefix callback type for single-block scans.
    // Maintains a running prefix that can be applied to consecutive
    // scan operations.
    struct RunningBlockPrefixOp
    {
        // Running prefix
        T running_total;

        // Callback operator.
        __device__ T operator()(T block_aggregate)
        {
            T old_prefix = running_total;
            running_total += block_aggregate;
            return old_prefix;
        }
    };

    // Shared memory type for this threadblock
    struct TempStorage
    {
        union
        {
            typename BlockLoadT::TempStorage            load;               // Smem needed for tile loading
            typename BlockStoreT::TempStorage           store;              // Smem needed for tile storing
            struct
            {
                typename InterblockPrefixOp::TempStorage    prefix;         // Smem needed for cooperative prefix callback
                typename BlockScanT::TempStorage            scan;           // Smem needed for tile scanning
            };
        };

        SizeT                                           tile_idx;           // Shared tile index
    };


    //---------------------------------------------------------------------
    // Per-thread fields
    //---------------------------------------------------------------------

    TempStorage                 &temp_storage;      ///< Reference to temp_storage
    InputIteratorRA             d_in;               ///< Input data
    OutputIteratorRA            d_out;              ///< Output data
    ScanOp                      scan_op;            ///< Binary scan operator
    Identity                    identity;           ///< Identity element



    //---------------------------------------------------------------------
    // Block scan utility methods (first tile)
    //---------------------------------------------------------------------

    /**
     * Exclusive scan specialization
     */
    template <typename _ScanOp, typename _Identity>
    __device__ __forceinline__
    void ScanBlock(T (&items)[ITEMS_PER_THREAD], _ScanOp scan_op, _Identity identity, T& block_aggregate)
    {
        BlockScanT(temp_storage.scan).ExclusiveScan(items, items, identity, scan_op, block_aggregate);
    }

    /**
     * Exclusive sum specialization
     */
    template <typename _Identity>
    __device__ __forceinline__
    void ScanBlock(T (&items)[ITEMS_PER_THREAD], Sum scan_op, _Identity identity, T& block_aggregate)
    {
        BlockScanT(temp_storage.scan).ExclusiveSum(items, items, block_aggregate);
    }

    /**
     * Inclusive scan specialization
     */
    template <typename _ScanOp>
    __device__ __forceinline__
    void ScanBlock(T (&items)[ITEMS_PER_THREAD], _ScanOp scan_op, NullType identity, T& block_aggregate)
    {
        BlockScanT(temp_storage.scan).InclusiveScan(items, items, scan_op, block_aggregate);
    }

    /**
     * Inclusive sum specialization
     */
    __device__ __forceinline__
    void ScanBlock(T (&items)[ITEMS_PER_THREAD], Sum scan_op, NullType identity, T& block_aggregate)
    {
        BlockScanT(temp_storage.scan).InclusiveSum(items, items, block_aggregate);
    }

    //---------------------------------------------------------------------
    // Block scan utility methods (subsequent tiles)
    //---------------------------------------------------------------------

    /**
     * Exclusive scan specialization (with prefix from predecessors)
     */
    template <typename _ScanOp, typename _Identity, typename PrefixCallback>
    __device__ __forceinline__
    void ScanBlock(T (&items)[ITEMS_PER_THREAD], _ScanOp scan_op, _Identity identity, T& block_aggregate, PrefixCallback &prefix_op)
    {
        BlockScanT(temp_storage.scan).ExclusiveScan(items, items, identity, scan_op, block_aggregate, prefix_op);
    }

    /**
     * Exclusive sum specialization (with prefix from predecessors)
     */
    template <typename _Identity, typename PrefixCallback>
    __device__ __forceinline__
    void ScanBlock(T (&items)[ITEMS_PER_THREAD], Sum scan_op, _Identity identity, T& block_aggregate, PrefixCallback &prefix_op)
    {
        BlockScanT(temp_storage.scan).ExclusiveSum(items, items, block_aggregate, prefix_op);
    }

    /**
     * Inclusive scan specialization (with prefix from predecessors)
     */
    template <typename _ScanOp, typename PrefixCallback>
    __device__ __forceinline__
    void ScanBlock(T (&items)[ITEMS_PER_THREAD], _ScanOp scan_op, NullType identity, T& block_aggregate, PrefixCallback &prefix_op)
    {
        BlockScanT(temp_storage.scan).InclusiveScan(items, items, scan_op, block_aggregate, prefix_op);
    }

    /**
     * Inclusive sum specialization (with prefix from predecessors)
     */
    template <typename PrefixCallback>
    __device__ __forceinline__
    void ScanBlock(T (&items)[ITEMS_PER_THREAD], Sum scan_op, NullType identity, T& block_aggregate, PrefixCallback &prefix_op)
    {
        BlockScanT(temp_storage.scan).InclusiveSum(items, items, block_aggregate, prefix_op);
    }

    //---------------------------------------------------------------------
    // Interface
    //---------------------------------------------------------------------

    // Constructor
    __device__ __forceinline__
    BlockSweepScan(
        TempStorage                 &temp_storage,      ///< Reference to temp_storage
        InputIteratorRA             d_in,               ///< Input data
        OutputIteratorRA            d_out,              ///< Output data
        ScanOp                      scan_op,            ///< Binary scan operator
        Identity                    identity)           ///< Identity element
    :
        temp_storage(temp_storage),
        d_in(d_in),
        d_out(d_out),
        scan_op(scan_op),
        identity(identity)
    {}


    /**
     * Process a tile of input
     */
    template <bool FULL_TILE>
    __device__ __forceinline__ void ConsumeTile(
        SizeT                       num_items,          ///< Total number of input items
        int                         tile_idx,           ///< Tile index
        SizeT                       block_offset,       ///< Tile offset
        DeviceScanTileDescriptorT   *d_tile_status)     ///< Global list of tile status
    {
        // Load items
        T items[ITEMS_PER_THREAD];

        if (FULL_TILE)
            BlockLoadT(temp_storage.load).Load(d_in + block_offset, items);
        else
            BlockLoadT(temp_storage.load).Load(d_in + block_offset, items, num_items - block_offset);

        __syncthreads();

        T block_aggregate;
        if (tile_idx == 0)
        {
            ScanBlock(items, scan_op, identity, block_aggregate);

            // Update tile status if there are successor tiles
            if (FULL_TILE && (threadIdx.x == 0))
                DeviceScanTileDescriptorT::SetPrefix(d_tile_status, block_aggregate);
        }
        else
        {
            InterblockPrefixOp prefix_op(d_tile_status, temp_storage.prefix, scan_op, tile_idx);
            ScanBlock(items, scan_op, identity, block_aggregate, prefix_op);
        }

        __syncthreads();

        // Store items
        if (FULL_TILE)
            BlockStoreT(temp_storage.store).Store(d_out + block_offset, items);
        else
            BlockStoreT(temp_storage.store).Store(d_out + block_offset, items, num_items - block_offset);
    }


    /**
     * Process a tile of input
     */
    template <
        bool FULL_TILE,
        bool FIRST_TILE>
    __device__ __forceinline__ void ConsumeTile(
        SizeT                   block_offset,               ///< Tile offset
        RunningBlockPrefixOp    &prefix_op,                 ///< Running prefix operator
        int                     valid_items = TILE_ITEMS)   ///< Number of valid items in the tile
    {
        // Load items
        T items[ITEMS_PER_THREAD];

        if (FULL_TILE)
            BlockLoadT(temp_storage.load).Load(d_in + block_offset, items);
        else
            BlockLoadT(temp_storage.load).Load(d_in + block_offset, items, valid_items);

        __syncthreads();

        // Block scan
        if (FIRST_TILE)
        {
            T block_aggregate;
            ScanBlock(items, scan_op, identity, block_aggregate);
            prefix_op.running_total = block_aggregate;
        }
        else
        {
            ScanBlock(items, scan_op, identity, prefix_op);
        }

        __syncthreads();

        // Store items
        if (FULL_TILE)
            BlockStoreT(temp_storage.store).Store(d_out + block_offset, items);
        else
            BlockStoreT(temp_storage.store).Store(d_out + block_offset, items, valid_items);
    }


    /**
     * Dequeue and scan tiles of items as part of a inter-block scan
     */
    __device__ __forceinline__ void ConsumeTiles(
        int                     num_items,          ///< Total number of input items
        GridQueue<int>          queue,              ///< Queue descriptor for assigning tiles of work to thread blocks
        DeviceScanTileDescriptorT    *d_tile_status)    ///< Global list of tile status
    {
        // We give each thread block at least one tile of input.
        int tile_idx = blockIdx.x;

        SizeT block_offset = SizeT(TILE_ITEMS) * tile_idx;
        while (block_offset + TILE_ITEMS <= num_items)
        {
            ConsumeTile<true>(num_items, tile_idx, block_offset, d_tile_status);

            // Get next tile
#if CUB_PTX_ARCH < 200
            // No concurrent kernels allowed, so just stripe tiles
            tile_idx += gridDim.x;
#else
            // Concurrent kernels are allowed, so we must only use active blocks to dequeue tile indices
            if (threadIdx.x == 0)
                temp_storage.tile_idx = queue.Drain(1) + gridDim.x;

            __syncthreads();

            tile_idx = temp_storage.tile_idx;
#endif
            block_offset = SizeT(TILE_ITEMS) * tile_idx;
        }

        // Consume a partially-full tile
        if (block_offset < num_items)
        {
            ConsumeTile<false>(num_items, tile_idx, block_offset, d_tile_status);
        }
    }


    /**
     * Scan a consecutive segment of input tiles
     */
    __device__ __forceinline__ void ConsumeTiles(
        SizeT   block_offset,                       ///< [in] Threadblock begin offset (inclusive)
        SizeT   block_oob)                          ///< [in] Threadblock end offset (exclusive)
    {
        RunningBlockPrefixOp prefix_op;

        if (block_offset + TILE_ITEMS <= block_oob)
        {
            // Consume first tile of input (full)
            ConsumeTile<true, true>(block_offset, prefix_op);
            block_offset += TILE_ITEMS;

            // Consume subsequent full tiles of input
            while (block_offset + TILE_ITEMS <= block_oob)
            {
                ConsumeTile<true, false>(block_offset, prefix_op);
                block_offset += TILE_ITEMS;
            }

            // Consume a partially-full tile
            if (block_offset < block_oob)
            {
                int valid_items = block_oob - block_offset;
                ConsumeTile<false, false>(block_offset, prefix_op, valid_items);
            }
        }
        else
        {
            // Consume the first tile of input (partially-full)
            int valid_items = block_oob - block_offset;
            ConsumeTile<true, false>(block_offset, prefix_op, valid_items);
        }
    }


    /**
     * Scan a consecutive segment of input tiles, seeded with the specified prefix value
     */
    __device__ __forceinline__ void ConsumeTiles(
        SizeT   block_offset,                       ///< [in] Threadblock begin offset (inclusive)
        SizeT   block_oob,                          ///< [in] Threadblock end offset (exclusive)
        T       prefix)                             ///< [in] The prefix to apply to the scan segment
    {
        RunningBlockPrefixOp prefix_op;
        prefix_op.running_total = prefix;

        // Consume full tiles of input
        while (block_offset + TILE_ITEMS <= block_oob)
        {
            ConsumeTile<true, false>(block_offset, prefix_op);
            block_offset += TILE_ITEMS;
        }

        // Consume a partially-full tile
        if (block_offset < block_oob)
        {
            int valid_items = block_oob - block_offset;
            ConsumeTile<false, false>(block_offset, prefix_op, valid_items);
        }
    }

};


}               // CUB namespace
CUB_NS_POSTFIX  // Optional outer namespace(s)

