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


/******************************************************************************
 * Utility data types
 ******************************************************************************/

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
 * Data type of tile status word.  Specialized for scan types that can be
 * combined with a status flag to form a single word that can be singly read/written
 * (i.e., power-of-two-sized data less than or equal to 8B).
 */
template <
    typename T,
    bool SINGLE_WORD = (((sizeof(T) & (sizeof(T)  - 1)) == 0) && sizeof(T) <= 8)>
struct DeviceScanTileStatus
{
    T                                               value;
    typename WordAlignment<T>::VolatileAlignWord    status;

    static __device__ __forceinline__ void SetPrefix(DeviceScanTileStatus *ptr, T prefix)
    {
        DeviceScanTileStatus tile_status;
        tile_status.status = DEVICE_SCAN_TILE_PREFIX;
        tile_status.value = prefix;
        ThreadStore<STORE_CG>(ptr, tile_status);
    }

    static __device__ __forceinline__ void SetPartial(DeviceScanTileStatus *ptr, T partial)
    {
        DeviceScanTileStatus tile_status;
        tile_status.status = DEVICE_SCAN_TILE_PARTIAL;
        tile_status.value = partial;
        ThreadStore<STORE_CG>(ptr, tile_status);
    }

    static __device__ __forceinline__ void WaitForValid(
        DeviceScanTileStatus    *ptr,
        int                     &status,
        T                       &value)
    {
        DeviceScanTileStatus tile_status;
        while (true)
        {
            tile_status = ThreadLoad<LOAD_CG>(ptr);
            if (__all(tile_status.status != DEVICE_SCAN_TILE_INVALID)) break;
            __threadfence_block();

            tile_status = ThreadLoad<LOAD_CG>(ptr);
            if (__all(tile_status.status != DEVICE_SCAN_TILE_INVALID)) break;
            __threadfence_block();
        }

        status = tile_status.status;
        value = tile_status.value;
    }

};


/**
 * Data type of tile status word.  Specialized for scan types that cannot be
 * combined with a status flag to form a single word that can be singly read/written
 * (i.e., non-power-of-two-sized data or data greater than 8B).
 */
template <typename T>
struct DeviceScanTileStatus<T, false>
{
    T       prefix_value;
    T       partial_value;
    int     status;

    static __device__ __forceinline__ void SetPrefix(DeviceScanTileStatus *ptr, T prefix)
    {
        ThreadStore<STORE_CG>(&ptr->prefix_value, prefix);
        __threadfence_block();  // We can get away with not using a global fence because the fields will all be on the same cache line
        ThreadStore<STORE_CG>(&ptr->status, DEVICE_SCAN_TILE_PREFIX);
    }

    static __device__ __forceinline__ void SetPartial(DeviceScanTileStatus *ptr, T partial)
    {
        ThreadStore<STORE_CG>(&ptr->partial_value, partial);
        __threadfence_block();  // We can get away with not using a global fence because the fields will all be on the same cache line
        ThreadStore<STORE_CG>(&ptr->status, DEVICE_SCAN_TILE_PARTIAL);
    }

    static __device__ __forceinline__ void WaitForValid(
        DeviceScanTileStatus    *ptr,
        int                     &status,
        T                       &value)
    {
        while (true)
        {
            status = ThreadLoad<LOAD_CG>(&ptr->status);
            if (__all(status != DEVICE_SCAN_TILE_INVALID)) break;
            __threadfence_block();

            status = ThreadLoad<LOAD_CG>(&ptr->status);
            if (__all(status != DEVICE_SCAN_TILE_INVALID)) break;
            __threadfence_block();
        }

        T partial   = ThreadLoad<LOAD_CG>(&ptr->partial_value);
        T prefix    = ThreadLoad<LOAD_CG>(&ptr->prefix_value);

        value = (status == DEVICE_SCAN_TILE_PARTIAL) ? partial : prefix;
    }
};


/******************************************************************************
 * Tuning policy types
 ******************************************************************************/

/**
 * Tuning policy for PersistentBlockScan
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
struct PersistentBlockScanPolicy
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
        PersistentBlockScanPolicy::LOAD_ALGORITHM,
        PersistentBlockScanPolicy::LOAD_MODIFIER,
        PersistentBlockScanPolicy::LOAD_WARP_TIME_SLICING>  BlockLoadT;

    // Parameterized block store
    typedef BlockStore<
        OutputIteratorRA,
        PersistentBlockScanPolicy::BLOCK_THREADS,
        PersistentBlockScanPolicy::ITEMS_PER_THREAD,
        PersistentBlockScanPolicy::STORE_ALGORITHM,
        STORE_DEFAULT,
        PersistentBlockScanPolicy::STORE_WARP_TIME_SLICING> BlockStoreT;

    // Parameterized block scan
    typedef BlockScan<
        T,
        PersistentBlockScanPolicy::BLOCK_THREADS,
        PersistentBlockScanPolicy::SCAN_ALGORITHM>          BlockScanT;

    // Parameterized warp reduce
    typedef WarpReduce<T>                                   WarpReduceT;

    // Shared memory type for this threadblock
    struct TempStorage
    {
        union
        {
            typename BlockLoadT::TempStorage    load;   // Smem needed for tile loading
            typename BlockStoreT::TempStorage   store;  // Smem needed for tile storing
            typename BlockScanT::TempStorage    scan;   // Smem needed for tile scanning
        };

        typename WarpReduceT::TempStorage       warp_reduce;    // Smem needed for warp reduction
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


        // Block until all predecessors within the specified window have non-invalid status
        __device__ __forceinline__
        void ProcessWindow(
            int                         predecessor_idx,
            int                         &predecessor_status,
            T                           &window_prefix)
        {
            T value;
            DeviceScanTileStatus<T>::WaitForValid(block_scan->d_tile_status + predecessor_idx, predecessor_status, value);

            // Perform a segmented reduction to get the prefix for the current window
            int flag = (predecessor_status != DEVICE_SCAN_TILE_PARTIAL);
            window_prefix = WarpReduceT(block_scan->temp_storage.warp_reduce).TailSegmentedReduce(
                value, flag, block_scan->scan_op);
        }

        // Prefix functor (called by the first warp)
        __device__ __forceinline__
        T operator()(T block_aggregate)
        {
            // Update our status with our tile-aggregate
            if (threadIdx.x == 0)
            {
                DeviceScanTileStatus<T>::SetPartial(
                    block_scan->d_tile_status + tile_idx,
                    block_aggregate);
            }

            // Wait for the window of predecessor blocks to become valid
            int predecessor_idx = tile_idx - threadIdx.x - 1;
            int predecessor_status;
            T window_prefix;
            ProcessWindow(predecessor_idx, predecessor_status, window_prefix);

            // The block prefix starts out as the current window prefix
            T block_prefix = window_prefix;

            // Keep sliding the window back until we come across a tile whose prefix (not aggregate) is known
            while (__all(predecessor_status != DEVICE_SCAN_TILE_PREFIX))
            {
                predecessor_idx -= PtxArchProps::WARP_THREADS;

                // Update block prefix with the window prefix
                ProcessWindow(predecessor_idx, predecessor_status, window_prefix);
                block_prefix = block_scan->scan_op(window_prefix, block_prefix);
            }

            // Update our status with our inclusive block_prefix
            if (threadIdx.x == 0)
            {
                DeviceScanTileStatus<T>::SetPrefix(
                    block_scan->d_tile_status + tile_idx,
                    block_scan->scan_op(block_prefix, block_aggregate));
            }

            // Return block-wide exclusive block_prefix
            return block_prefix;
        }
    };


    //---------------------------------------------------------------------
    // Per-thread fields
    //---------------------------------------------------------------------

    TempStorage                 &temp_storage;      ///< Reference to temp_storage
    InputIteratorRA             d_in;               ///< Input data
    OutputIteratorRA            d_out;              ///< Output data
    DeviceScanTileStatus<T>     *d_tile_status;     ///< Global list of tile status
    ScanOp                      scan_op;            ///< Binary scan operator
    Identity                    identity;           ///< Identity element
    SizeT                       num_items;          ///< Total number of scan items for the entire problem



    //---------------------------------------------------------------------
    // Interface
    //---------------------------------------------------------------------

    // Constructor
    __device__ __forceinline__
    PersistentBlockScan(
        TempStorage                 &temp_storage,      ///< Reference to temp_storage
        InputIteratorRA             d_in,               ///< Input data
        OutputIteratorRA            d_out,              ///< Output data
        DeviceScanTileStatus<T>     *d_tile_status,     ///< Global list of tile status
        ScanOp                      scan_op,            ///< Binary scan operator
        Identity                    identity,           ///< Identity element
        SizeT                       num_items) :        ///< Total number of scan items for the entire problem
            temp_storage(temp_storage),
            d_in(d_in),
            d_out(d_out),
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
        BlockScanT(temp_storage.scan).ExclusiveScan(items, items, identity, scan_op, block_aggregate);
    }

    /**
     * Exclusive sum specialization
     */
    template <typename _Identity>
    __device__ __forceinline__
    void ScanBlock(T (&items)[ITEMS_PER_THREAD], Sum<T> scan_op, _Identity identity, T& block_aggregate)
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
        BlockScanT::InclusiveScan(temp_storage.scan, items, items, scan_op, block_aggregate);
    }

    /**
     * Inclusive sum specialization
     */
    template <typename _ScanOp>
    __device__ __forceinline__
    void ScanBlock(T (&items)[ITEMS_PER_THREAD], Sum<T> scan_op, NullType identity, T& block_aggregate)
    {
        BlockScanT::InclusiveSum(temp_storage.scan, items, items, block_aggregate);
    }

    /**
     * Exclusive scan specialization (with prefix from predecessors)
     */
    template <typename _ScanOp, typename _Identity>
    __device__ __forceinline__
    void ScanBlock(T (&items)[ITEMS_PER_THREAD], _ScanOp scan_op, _Identity identity, T& block_aggregate, int tile_idx)
    {
        BlockPrefixOp prefix_op(this, tile_idx);
        BlockScanT(temp_storage.scan).ExclusiveScan(items, items, identity, scan_op, block_aggregate, prefix_op);
    }

    /**
     * Exclusive sum specialization (with prefix from predecessors)
     */
    template <typename _Identity>
    __device__ __forceinline__
    void ScanBlock(T (&items)[ITEMS_PER_THREAD], Sum<T> scan_op, _Identity identity, T& block_aggregate, int tile_idx)
    {
        BlockPrefixOp prefix_op(this, tile_idx);
        BlockScanT(temp_storage.scan).ExclusiveSum(items, items, block_aggregate, prefix_op);
    }

    /**
     * Inclusive scan specialization (with prefix from predecessors)
     */
    template <typename _ScanOp>
    __device__ __forceinline__
    void ScanBlock(T (&items)[ITEMS_PER_THREAD], _ScanOp scan_op, NullType identity, T& block_aggregate, int tile_idx)
    {
        BlockPrefixOp prefix_op(this, tile_idx);
        BlockScanT::InclusiveScan(temp_storage.scan, items, items, scan_op, block_aggregate, prefix_op);
    }

    /**
     * Inclusive sum specialization (with prefix from predecessors)
     */
    template <typename _ScanOp>
    __device__ __forceinline__
    void ScanBlock(T (&items)[ITEMS_PER_THREAD], Sum<T> scan_op, NullType identity, T& block_aggregate, int tile_idx)
    {
        BlockPrefixOp prefix_op(this, tile_idx);
        BlockScanT::InclusiveSum(temp_storage.scan, items, items, block_aggregate, prefix_op);
    }


    /**
     * Process the tile (full)
     */
    __device__ __forceinline__ void ConsumeTileFull(int tile_idx)
    {
        SizeT block_offset = SizeT(TILE_ITEMS) * tile_idx;

        // Load items
        T items[ITEMS_PER_THREAD];
        BlockLoadT(temp_storage.load).Load(d_in + block_offset, items);

        __syncthreads();

        if (tile_idx == 0)
        {
            T block_aggregate;
            ScanBlock(items, scan_op, identity, block_aggregate);

            // Update tile status
            if (threadIdx.x == 0)
            {
                DeviceScanTileStatus<T>::SetPrefix(d_tile_status, block_aggregate);
            }
        }
        else
        {
            T block_aggregate;
            ScanBlock(items, scan_op, identity, block_aggregate, tile_idx);
        }


        __syncthreads();

        // Store items
        BlockStoreT(temp_storage.store).Store(d_out + block_offset, items);
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
        BlockLoadT(temp_storage.load).Load(d_in + block_offset, items, num_valid);

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
        BlockStoreT(temp_storage.store).Store(d_out + block_offset, items, num_valid);
    }


};


}               // CUB namespace
CUB_NS_POSTFIX  // Optional outer namespace(s)

