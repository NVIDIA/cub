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
    BlockLoadPolicy             _LOAD_POLICY,
    BlockStorePolicy            _STORE_POLICY,
    BlockScanAlgorithm          _SCAN_ALGORITHM>
struct PersistentBlockScanPolicy
{
    enum
    {
        BLOCK_THREADS       = _BLOCK_THREADS,
        ITEMS_PER_THREAD    = _ITEMS_PER_THREAD,
    };

    static const BlockLoadPolicy        LOAD_POLICY      = _LOAD_POLICY;
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

    // Pairing of a tile aggregate/prefix and tile status word
    struct PredecessorTile
    {
        T                       value;
        DeviceScanTileStatus    status;
/*
        typedef void ThreadLoadTag;
        typedef void ThreadStoreTag;

        // ThreadLoad
        template <cub::PtxLoadModifier MODIFIER>
        __device__ __forceinline__
        void ThreadLoad(PredecessorTile *ptr)
        {
            value = cub::ThreadLoad<MODIFIER>(&(ptr->value));
            status = cub::ThreadLoad<MODIFIER>(&(ptr->status));
        }

         // ThreadStore
        template <cub::PtxStoreModifier MODIFIER>
        __device__ __forceinline__ void ThreadStore(PredecessorTile *ptr) const
        {
            cub::ThreadStore<MODIFIER>(&(ptr->value), value);
            cub::ThreadStore<MODIFIER>(&(ptr->status), status);
        }
*/
    };

    // Reduction operator for computing the exclusive prefix from predecessor tile status
    struct PredecessorTileReduceOp
    {
        ScanOp scan_op;

        // Constructor
        __device__ __forceinline__
        PredecessorTileReduceOp(ScanOp scan_op) : scan_op(scan_op) {}

        __device__ __forceinline__
        PredecessorTile operator()(const PredecessorTile& first, const PredecessorTile& second)
        {
            if ((second.status == DEVICE_SCAN_TILE_OOB) || (first.status == DEVICE_SCAN_TILE_PREFIX))
                return first;

            PredecessorTile retval;
            retval.status = second.status;
            retval.value = scan_op(second.value, first.value);
            return retval;
        }
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
    typedef WarpReduce<PredecessorTile>                 WarpReduceT;

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
        int                     tile_index;     ///< The current tile index

        // Constructor
        __device__ __forceinline__
        BlockPrefixOp(
            PersistentBlockScan     *block_scan,
            int                     tile_index) :
                block_scan(block_scan),
                tile_index(tile_index) {}

        // Prefix functor (called by the first warp)
        __device__ __forceinline__
        T operator()(T block_aggregate)
        {

            // Update our status with our tile-aggregate
            if (threadIdx.x == 0)
            {
                block_scan->d_tile_aggregates[tile_index] = block_aggregate;

                __threadfence();

                block_scan->d_tile_status[tile_index] = DEVICE_SCAN_TILE_PARTIAL;
            }

            // Wait for all predecessor blocks to become valid and at least one prefix to show up.
            PredecessorTile predecessor_status;
            unsigned int predecessor_idx = tile_index - threadIdx.x - 1;
            do
            {
                predecessor_status.status = block_scan->d_tile_status[predecessor_idx];
            }
            while (__any(predecessor_status.status == DEVICE_SCAN_TILE_INVALID) || __all(predecessor_status.status != DEVICE_SCAN_TILE_PREFIX));

            // Grab predecessor block's corresponding partial/prefix
            predecessor_status.value = (predecessor_status.status == DEVICE_SCAN_TILE_PREFIX) ?
                block_scan->d_tile_prefixes[predecessor_idx] :
                block_scan->d_tile_aggregates[predecessor_idx];

            // Reduce predecessor partials/prefixes to get our block-wide exclusive prefix
            PredecessorTile prefix_status = WarpReduceT::Reduce(
                block_scan->smem_storage.warp_reduce,
                predecessor_status,
                PredecessorTileReduceOp(block_scan->scan_op));

            // Update our status with our inclusive prefix
            if (threadIdx.x == 0)
            {
                T inclusive_prefix = block_scan->scan_op(prefix_status.value, block_aggregate);

                block_scan->d_tile_prefixes[tile_index] = inclusive_prefix;

                __threadfence();

                block_scan->d_tile_status[tile_index] = DEVICE_SCAN_TILE_PREFIX;
            }

            // Return block-wide exclusive prefix
            return prefix_status.value;

/*


            // Update our status with our tile-aggregate
            if (threadIdx.x == 0)
            {
                block_scan->d_tile_aggregates[tile_index] = block_aggregate;

                __threadfence();

                block_scan->d_tile_status[tile_index] = DEVICE_SCAN_TILE_PARTIAL;
            }

            // Wait for all predecessor blocks to become valid and at least one prefix to show up.
            unsigned int predecessor_idx = tile_index - threadIdx.x - 1;

            PredecessorTile predecessor_status;
            do
            {
                predecessor_status.status = block_scan->d_tile_status[predecessor_idx];
            }
            while (__any(predecessor_status.status == DEVICE_SCAN_TILE_INVALID));

            // Grab predecessor block's corresponding partial/prefix
            predecessor_status.value = (predecessor_status.status == DEVICE_SCAN_TILE_PREFIX) ?
                block_scan->d_tile_prefixes[predecessor_idx] :
                block_scan->d_tile_aggregates[predecessor_idx];

            // Reduce predecessor partials/prefixes to get our block-wide exclusive prefix
            PredecessorTileReduceOp status_reduce_op(block_scan->scan_op);

            PredecessorTile window_prefix_status = WarpReduceT::Reduce(
                block_scan->smem_storage.warp_reduce,
                predecessor_status,
                status_reduce_op);

            // Continue looking at more predecessors if necessary
            PredecessorTile prefix_status = window_prefix_status;
            while (__all(predecessor_status.status != DEVICE_SCAN_TILE_PREFIX))
            {
                predecessor_idx -= PtxArchProps::WARP_THREADS;

                do
                {
                    predecessor_status.status = block_scan->d_tile_status[predecessor_idx];
                }
                while (__any(predecessor_status.status == DEVICE_SCAN_TILE_INVALID));

                // Grab predecessor block's corresponding partial/prefix
                predecessor_status.value = (predecessor_status.status == DEVICE_SCAN_TILE_PREFIX) ?
                    block_scan->d_tile_prefixes[predecessor_idx] :
                    block_scan->d_tile_aggregates[predecessor_idx];

                // Reduce predecessor partials/prefixes to get our block-wide exclusive prefix
                PredecessorTile window_prefix_status = WarpReduceT::Reduce(
                    block_scan->smem_storage.warp_reduce,
                    predecessor_status,
                    status_reduce_op);

                prefix_status = status_reduce_op(prefix_status, window_prefix_status);
            }

            // Update our status with our inclusive prefix
            if (threadIdx.x == 0)
            {
                T inclusive_prefix = block_scan->scan_op(prefix_status.value, block_aggregate);

                block_scan->d_tile_prefixes[tile_index] = inclusive_prefix;

                __threadfence();

                block_scan->d_tile_status[tile_index] = DEVICE_SCAN_TILE_PREFIX;
            }

            // Return block-wide exclusive prefix
            return prefix_status.value;

 */
        }
    };


    //---------------------------------------------------------------------
    // Per-thread fields
    //---------------------------------------------------------------------

    SmemStorage             &smem_storage;      ///< Reference to smem_storage
    InputIteratorRA         d_in;               ///< Input data
    OutputIteratorRA        d_out;              ///< Output data
    volatile T                       *d_tile_aggregates; ///< Global list of block aggregates
    volatile T                       *d_tile_prefixes;   ///< Global list of block inclusive prefixes
    volatile DeviceScanTileStatus    *d_tile_status;     ///< Global list of tile status
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
            identity(identity) {}


    /**
     * The number of tiles processed per tile
     */
    __device__ __forceinline__ int TileItems()
    {
        return 1;
    }


    /**
     * Process a single tile of input
     */
    __device__ __forceinline__ void ConsumeTile(
        bool    &sync_after,    ///< Whether or not the caller needs to synchronize before repurposing the shared memory used by this instance
        int     tile_index,     ///< The index of the tile to consume
        int     num_valid)      ///< Unused
    {
        SizeT block_offset = SizeT(TILE_ITEMS) * tile_index;

        if (block_offset + TILE_ITEMS > num_items)
        {
            // We are the last, partially-full tile
            int valid_items = num_items - block_offset;

            // Load items
            T items[ITEMS_PER_THREAD];
            BlockLoadT::Load(smem_storage.load, d_in + block_offset, valid_items, items);

            __syncthreads();

            // Block scan
            if (INCLUSIVE)
                BlockScanT::InclusiveScan(smem_storage.scan, items, items, scan_op);
            else
                BlockScanT::ExclusiveScan(smem_storage.scan, items, items, identity, scan_op);

            __syncthreads();

            // Store items
            BlockStoreT::Store(smem_storage.store, d_out + block_offset, valid_items, items);
        }
        else if (tile_index == 0)
        {
            //
            // We are the first, full tile
            //

            // Load items
            T items[ITEMS_PER_THREAD];
            BlockLoadT::Load(smem_storage.load, d_in + block_offset, items);

            __syncthreads();

            // Block scan
            T block_aggregate;
            if (INCLUSIVE)
                BlockScanT::InclusiveScan(smem_storage.scan, items, items, scan_op, block_aggregate);
            else
                BlockScanT::ExclusiveScan(smem_storage.scan, items, items, identity, scan_op, block_aggregate);

            // Use barrier also as thread fence
            __syncthreads();

            if (threadIdx.x == 0)
            {
                // Update tile prefix value
                d_tile_prefixes[tile_index] = block_aggregate;

                __threadfence();

                // Update tile prefix status
                d_tile_status[tile_index] = DEVICE_SCAN_TILE_PREFIX;
            }

            // Store items
            BlockStoreT::Store(smem_storage.store, d_out + block_offset, items);
        }
        else
        {
            //
            // We are a full tile with predecessors
            //

            // Load items
            T items[ITEMS_PER_THREAD];
            BlockLoadT::Load(smem_storage.load, d_in + block_offset, items);

            __syncthreads();

            // Block scan
            T block_aggregate;
            if (INCLUSIVE)
                BlockScanT::InclusiveScan(smem_storage.scan, items, items, scan_op, block_aggregate, BlockPrefixOp(this, tile_index));
            else
                BlockScanT::ExclusiveScan(smem_storage.scan, items, items, identity, scan_op, block_aggregate, BlockPrefixOp(this, tile_index));

            __syncthreads();

            // Store items
            BlockStoreT::Store(smem_storage.store, d_out + block_offset, items);
        }

        sync_after = true;
    }


    /**
     * Finalize the computation.
     */
    __device__ __forceinline__ void Finalize(
        int dummy_result)
    {}

};


}               // CUB namespace
CUB_NS_POSTFIX  // Optional outer namespace(s)

