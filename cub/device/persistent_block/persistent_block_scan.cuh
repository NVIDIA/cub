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
 * cub::PersistentBlockScan implements a stateful abstraction of CUDA thread blocks for participating in device-wide prefix scan.
 */

#pragma once

#include <iterator>

#include "../../block/block_load.cuh"
#include "../../block/block_scan.cuh"
#include "../../warp/warp_reduce.cuh"
#include "../../util_namespace.cuh"

/// Optional outer namespace(s)
CUB_NS_PREFIX

/// CUB namespace
namespace cub {


/**
 * Enumerations of tile status
 */
enum
{
    DEVICE_SCAN_TILE_OOB,          // Out-of-bounds (e.g., padding)
    DEVICE_SCAN_TILE_INVALID,      // Not yet processed
    DEVICE_SCAN_TILE_PARTIAL,      // Tile aggregate is available
    DEVICE_SCAN_TILE_PREFIX,       // Inclusive tile prefix is available
};


/**
 * Data type of tile status word
 */
typedef int DeviceScanTileStatus;


/**
 * Tuning policy for PersistentBlockScan
 */
template <
    int                         _BLOCK_THREADS,
    int                         _ITEMS_PER_THREAD,
    BlockLoadAlgorithm             _LOAD_POLICY,
    BlockStorePolicy            _STORE_POLICY,
    BlockScanAlgorithm          _SCAN_ALGORITHM>
struct PersistentBlockScanPolicy
{
    enum
    {
        BLOCK_THREADS       = _BLOCK_THREADS,
        ITEMS_PER_THREAD    = _ITEMS_PER_THREAD,
    };

    static const BlockLoadAlgorithm        LOAD_POLICY      = _LOAD_POLICY;
    static const BlockStorePolicy       STORE_POLICY     = _STORE_POLICY;
    static const BlockScanAlgorithm     SCAN_ALGORITHM   = _SCAN_ALGORITHM;
};


/**
 * \brief PersistentBlockScan implements a stateful abstraction of CUDA thread blocks for participating in device-wide prefix scan.
 */
template <
    typename PersistentBlockScanPolicy,
    typename InputIteratorRA,
    typename OutputIteratorRA,
    typename ScanOp,
    typename Identity,
    typename SizeT>
struct PersistentBlockScan
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
        BLOCK_THREADS       = PersistentBlockScanPolicy::BLOCK_THREADS,
        ITEMS_PER_THREAD    = PersistentBlockScanPolicy::ITEMS_PER_THREAD,
        TILE_ITEMS          = BLOCK_THREADS * ITEMS_PER_THREAD,
        STATUS_PADDING      = PtxArchProps::WARP_THREADS,
    };

    // Parameterized block load
    typedef BlockLoad<
        InputIteratorRA,
        PersistentBlockScanPolicy::BLOCK_THREADS,
        PersistentBlockScanPolicy::ITEMS_PER_THREAD,
        PersistentBlockScanPolicy::LOAD_POLICY>         BlockLoadT;

    // Parameterized block store
    typedef BlockStore<
        OutputIteratorRA,
        PersistentBlockScanPolicy::BLOCK_THREADS,
        PersistentBlockScanPolicy::ITEMS_PER_THREAD,
        PersistentBlockScanPolicy::STORE_POLICY>        BlockStoreT;

    // Parameterized block scan
    typedef BlockScan<
        T,
        PersistentBlockScanPolicy::BLOCK_THREADS,
        PersistentBlockScanPolicy::SCAN_ALGORITHM>      BlockScanT;

    // Parameterized warp reduce
    typedef WarpReduce<T>                               WarpReduceT;

    // Shared memory type for this threadblock
    struct SmemStorage
    {
        union
        {
            typename BlockLoadT::SmemStorage    load;   // Smem needed for tile loading
            typename BlockStoreT::SmemStorage   store;  // Smem needed for tile storing
            typename BlockScanT::SmemStorage    scan;   // Smem needed for tile scanning
        };

        typename WarpReduceT::SmemStorage       warp_reduce;    // Smem needed for warp reduction
    };


    /**
     * Stateful prefix functor that provides the the running prefix for
     * the current tile by waiting on aggregates/prefixes from predecessor tiles
     * to become available
     */
    struct BlockPrefixOp
    {
        PersistentBlockScan     *block_scan;    ///< Reference to the persistent block scan abstraction
        int                     tile_idx;     ///< The current tile index

        // Constructor
        __device__ __forceinline__
        BlockPrefixOp(
            PersistentBlockScan     *block_scan,
            int                     tile_idx) :
                block_scan(block_scan),
                tile_idx(tile_idx) {}


/*
        // Prefix functor (called by the first warp)
        __device__ __forceinline__
        T operator()(T block_aggregate)
        {
            // Update our status with our tile-aggregate
            if (threadIdx.x == 0)
            {
                block_scan->d_tile_aggregates[tile_idx] = block_aggregate;
                __threadfence_block();
                block_scan->d_tile_status[tile_idx] = DEVICE_SCAN_TILE_PARTIAL;
            }

            // Wait for all predecessor blocks to become valid and at least one prefix to show up.
            PredecessorTile predecessor_status;
            unsigned int predecessor_idx = tile_idx - threadIdx.x - 1;
            do
            {
                predecessor_status.status = Load<LOAD_CG>(block_scan->d_tile_status + predecessor_idx);
                __threadfence_block();
            }
            while (__all(predecessor_status.status != DEVICE_SCAN_TILE_PREFIX) || __any(predecessor_status.status == DEVICE_SCAN_TILE_INVALID));

            // Grab predecessor block's corresponding partial/prefix
            predecessor_status.value = (predecessor_status.status == DEVICE_SCAN_TILE_PREFIX) ?
                Load<LOAD_CG>(block_scan->d_tile_prefixes + predecessor_idx) :
                Load<LOAD_CG>(block_scan->d_tile_aggregates + predecessor_idx);

            int flag = (predecessor_status.status != DEVICE_SCAN_TILE_PARTIAL);

            T prefix = WarpReduceT::SegmentedReduce(
                block_scan->smem_storage.warp_reduce,
                predecessor_status.value,
                flag,
                block_scan->scan_op);

            // Update our status with our inclusive prefix
            if (threadIdx.x == 0)
            {
                T inclusive_prefix = block_scan->scan_op(prefix, block_aggregate);

                block_scan->d_tile_prefixes[tile_idx] = inclusive_prefix;
                __threadfence_block();
                block_scan->d_tile_status[tile_idx] = DEVICE_SCAN_TILE_PREFIX;
            }

            // Return block-wide exclusive prefix
            return prefix;
        }
*/



        // Prefix functor (called by the first warp)
        __device__ __forceinline__
        T operator()(T block_aggregate)
        {
            // Update our status with our tile-aggregate
            if (threadIdx.x == 0)
            {
                block_scan->d_tile_aggregates[tile_idx] = block_aggregate;
                __threadfence_block();
                block_scan->d_tile_status[tile_idx] = DEVICE_SCAN_TILE_PARTIAL;
            }

            // Wait for the window of predecessor blocks to become valid
            int predecessor_idx = tile_idx - threadIdx.x - 1;

            DeviceScanTileStatus predecessor_status;
            do
            {
                predecessor_status = Load<LOAD_CG>(block_scan->d_tile_status + predecessor_idx);
                __threadfence_block();
            }
            while (predecessor_status == DEVICE_SCAN_TILE_INVALID);

            // Grab predecessor block's corresponding partial/prefix
            T predecessor_value = (predecessor_status == DEVICE_SCAN_TILE_PREFIX) ?
                Load<LOAD_CG>(block_scan->d_tile_prefixes + predecessor_idx) :
                Load<LOAD_CG>(block_scan->d_tile_aggregates + predecessor_idx);

            // Perform a segmented reduction to get the prefix for the current window
            int flag = (predecessor_status != DEVICE_SCAN_TILE_PARTIAL);
            T window_prefix = WarpReduceT::SegmentedReduce(
                block_scan->smem_storage.warp_reduce,
                predecessor_value,
                flag,
                block_scan->scan_op);

            // The block prefix starts out as the current window prefix
            T block_prefix = window_prefix;

            // Keep sliding the window back until we come across a tile whose prefix (not aggregate) is known
            while (__all(predecessor_status != DEVICE_SCAN_TILE_PREFIX))
            {
                predecessor_idx -= PtxArchProps::WARP_THREADS;

                // Wait for the window of predecessor blocks to become valid
                do
                {
                    predecessor_status = Load<LOAD_CG>(block_scan->d_tile_status + predecessor_idx);
                    __threadfence_block();
                }
                while (predecessor_status == DEVICE_SCAN_TILE_INVALID);

                // Grab predecessor block's corresponding partial/prefix
                predecessor_value = (predecessor_status == DEVICE_SCAN_TILE_PREFIX) ?
                    Load<LOAD_CG>(block_scan->d_tile_prefixes + predecessor_idx) :
                    Load<LOAD_CG>(block_scan->d_tile_aggregates + predecessor_idx);

                // Perform a segmented reduction to get the prefix for the current window
                flag = (predecessor_status != DEVICE_SCAN_TILE_PARTIAL);
                window_prefix = WarpReduceT::SegmentedReduce(
                    block_scan->smem_storage.warp_reduce,
                    predecessor_value,
                    flag,
                    block_scan->scan_op);

                // Update block prefix with the window prefix
                block_prefix = block_scan->scan_op(window_prefix, block_prefix);
            }

            // Update our status with our inclusive block_prefix
            if (threadIdx.x == 0)
            {
                T inclusive_prefix = block_scan->scan_op(block_prefix, block_aggregate);

                block_scan->d_tile_prefixes[tile_idx] = inclusive_prefix;
                __threadfence_block();
                block_scan->d_tile_status[tile_idx] = DEVICE_SCAN_TILE_PREFIX;
            }

            // Return block-wide exclusive block_prefix
            return block_prefix;
        }
    };


    //---------------------------------------------------------------------
    // Per-thread fields
    //---------------------------------------------------------------------

    SmemStorage             &smem_storage;      ///< Reference to smem_storage
    InputIteratorRA         d_in;               ///< Input data
    OutputIteratorRA        d_out;              ///< Output data
    T                       *d_tile_aggregates; ///< Global list of block aggregates
    T                       *d_tile_prefixes;   ///< Global list of block inclusive prefixes
    DeviceScanTileStatus    *d_tile_status;     ///< Global list of tile status
    ScanOp                  scan_op;            ///< Binary scan operator
    Identity                identity;           ///< Identity element
    SizeT                   num_items;          ///< Total number of scan items for the entire problem



    //---------------------------------------------------------------------
    // Interface
    //---------------------------------------------------------------------

    // Constructor
    __device__ __forceinline__
    PersistentBlockScan(
        SmemStorage             &smem_storage,      ///< Reference to smem_storage
        InputIteratorRA         d_in,               ///< Input data
        OutputIteratorRA        d_out,              ///< Output data
        T                       *d_tile_aggregates, ///< Global list of block aggregates
        T                       *d_tile_prefixes,   ///< Global list of block inclusive prefixes
        DeviceScanTileStatus    *d_tile_status,     ///< Global list of tile status
        ScanOp                  scan_op,            ///< Binary scan operator
        Identity                identity,           ///< Identity element
        SizeT                   num_items) :        ///< Total number of scan items for the entire problem
            smem_storage(smem_storage),
            d_in(d_in),
            d_out(d_out),
            d_tile_aggregates(d_tile_aggregates),
            d_tile_prefixes(d_tile_prefixes),
            d_tile_status(d_tile_status + STATUS_PADDING),
            scan_op(scan_op),
            identity(identity),
            num_items(num_items) {}


    /**
     * Exclusive scan specialization
     */
    template <typename _ScanOp, typename _Identity>
    __device__ __forceinline__
    void ScanBlock(T (&items)[ITEMS_PER_THREAD], _ScanOp scan_op, _Identity identity, T& block_aggregate)
    {
        BlockScanT::ExclusiveScan(smem_storage.scan, items, items, identity, scan_op, block_aggregate);
    }

    /**
     * Exclusive sum specialization
     */
    template <typename _Identity>
    __device__ __forceinline__
    void ScanBlock(T (&items)[ITEMS_PER_THREAD], Sum<T> scan_op, _Identity identity, T& block_aggregate)
    {
        BlockScanT::ExclusiveSum(smem_storage.scan, items, items, block_aggregate);
    }

    /**
     * Inclusive scan specialization
     */
    template <typename _ScanOp>
    __device__ __forceinline__
    void ScanBlock(T (&items)[ITEMS_PER_THREAD], _ScanOp scan_op, NullType identity, T& block_aggregate)
    {
        BlockScanT::InclusiveScan(smem_storage.scan, items, items, scan_op, block_aggregate);
    }

    /**
     * Inclusive sum specialization
     */
    template <typename _ScanOp>
    __device__ __forceinline__
    void ScanBlock(T (&items)[ITEMS_PER_THREAD], Sum<T> scan_op, NullType identity, T& block_aggregate)
    {
        BlockScanT::InclusiveSum(smem_storage.scan, items, items, block_aggregate);
    }

    /**
     * Exclusive scan specialization (with prefix from predecessors)
     */
    template <typename _ScanOp, typename _Identity>
    __device__ __forceinline__
    void ScanBlock(T (&items)[ITEMS_PER_THREAD], _ScanOp scan_op, _Identity identity, T& block_aggregate, int tile_idx)
    {
        BlockPrefixOp prefix_op(this, tile_idx);
        BlockScanT::ExclusiveScan(smem_storage.scan, items, items, identity, scan_op, block_aggregate, prefix_op);
    }

    /**
     * Exclusive sum specialization (with prefix from predecessors)
     */
    template <typename _Identity>
    __device__ __forceinline__
    void ScanBlock(T (&items)[ITEMS_PER_THREAD], Sum<T> scan_op, _Identity identity, T& block_aggregate, int tile_idx)
    {
        BlockPrefixOp prefix_op(this, tile_idx);
        BlockScanT::ExclusiveSum(smem_storage.scan, items, items, block_aggregate, prefix_op);
    }

    /**
     * Inclusive scan specialization (with prefix from predecessors)
     */
    template <typename _ScanOp>
    __device__ __forceinline__
    void ScanBlock(T (&items)[ITEMS_PER_THREAD], _ScanOp scan_op, NullType identity, T& block_aggregate, int tile_idx)
    {
        BlockPrefixOp prefix_op(this, tile_idx);
        BlockScanT::InclusiveScan(smem_storage.scan, items, items, scan_op, block_aggregate, prefix_op);
    }

    /**
     * Inclusive sum specialization (with prefix from predecessors)
     */
    template <typename _ScanOp>
    __device__ __forceinline__
    void ScanBlock(T (&items)[ITEMS_PER_THREAD], Sum<T> scan_op, NullType identity, T& block_aggregate, int tile_idx)
    {
        BlockPrefixOp prefix_op(this, tile_idx);
        BlockScanT::InclusiveSum(smem_storage.scan, items, items, block_aggregate, prefix_op);
    }


    /**
     * Process the tile (full)
     */
    __device__ __forceinline__ void ConsumeTileFull(int tile_idx)
    {
        SizeT block_offset = SizeT(TILE_ITEMS) * tile_idx;

        // Load items
        T items[ITEMS_PER_THREAD];
        BlockLoadT::Load(smem_storage.load, d_in + block_offset, items);

        __syncthreads();

        if (tile_idx == 0)
        {
            T block_aggregate;
            ScanBlock(items, scan_op, identity, block_aggregate);

            // Update tile status
            if (threadIdx.x == 0)
            {
                d_tile_prefixes[0] = block_aggregate;
                __threadfence_block();
                d_tile_status[0] = DEVICE_SCAN_TILE_PREFIX;
            }
        }
        else
        {
            T block_aggregate;
            ScanBlock(items, scan_op, identity, block_aggregate, tile_idx);
        }


        __syncthreads();

        // Store items
        BlockStoreT::Store(smem_storage.store, d_out + block_offset, items);
    }


    /**
     * Process the tile (partial)
     */
    __device__ __forceinline__ void ConsumeTilePartial(int tile_idx)
    {
        SizeT block_offset = SizeT(TILE_ITEMS) * tile_idx;
        int num_valid = num_items - block_offset;

        // Load items
        T items[ITEMS_PER_THREAD];
        BlockLoadT::Load(smem_storage.load, d_in + block_offset, num_valid, items);

        __syncthreads();

        if (tile_idx == 0)
        {
            // Block scan
            T block_aggregate;
            ScanBlock(items, scan_op, identity, block_aggregate);
        }
        else
        {
            // Block scan
            T block_aggregate;
            ScanBlock(items, scan_op, identity, block_aggregate, tile_idx);
        }


        __syncthreads();

        // Store items
        BlockStoreT::Store(smem_storage.store, d_out + block_offset, num_valid, items);
    }


};


}               // CUB namespace
CUB_NS_POSTFIX  // Optional outer namespace(s)

