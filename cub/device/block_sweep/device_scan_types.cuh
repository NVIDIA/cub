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
 * Utility types for device-wide scan
 */

#pragma once

#include <iterator>

#include "../../thread/thread_load.cuh"
#include "../../thread/thread_store.cuh"
#include "../../warp/warp_reduce.cuh"
#include "../../util_ptx.cuh"
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
 * Data type of tile status word.
 *
 * Specialized for scan status and value types that can be combined into the same
 * machine word that can be read/written coherently in a single access.
 */
template <
    typename T,
    bool SINGLE_WORD =
        ((sizeof(T) & (sizeof(T)  - 1)) == 0) &&        // T is a power-of-two
        (sizeof(T) <= 8)>                               // T is less than 16B (16B total is the max word size)
struct DeviceScanTileStatus
{
    T                                       value;
    typename WordAlignment<T>::AlignWord    status;

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
            if (tile_status.status != DEVICE_SCAN_TILE_INVALID) break;
        }

        status = tile_status.status;
        value = tile_status.value;
    }

};


/**
 * Data type of tile status word.
 *
 * Specialized for scan status and value types that cannot fused into
 * the same machine word.
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
            if (status != DEVICE_SCAN_TILE_INVALID) break;
        }

        value = (status == DEVICE_SCAN_TILE_PARTIAL) ?
            ThreadLoad<LOAD_CG>(&ptr->partial_value) :
            ThreadLoad<LOAD_CG>(&ptr->prefix_value);
    }
};


/**
 * Stateful prefix functor that provides the the running prefix for
 * the current tile by using the callback warp to wait on on
 * aggregates/prefixes from predecessor tiles to become available
 */
template <
    typename T,
    typename ScanOp>
struct DeviceScanBlockPrefixOp
{
    // Parameterized warp reduce
    typedef WarpReduce<T> WarpReduceT;

    // Storage type
    typedef WarpReduceT::TempStorage TempStorage;

    // Fields
    DeviceScanTileStatus<T>     *d_tile_status; ///< Pointer to array of tile status
    TempStorage                 &temp_storage;  ///< Reference to a warp-reduction instance
    ScanOp                      scan_op;        ///< Binary scan operator
    int                         tile_idx;       ///< The current tile index

    // Constructor
    __device__ __forceinline__
    DeviceScanBlockPrefixOp(
        DeviceScanTileStatus<T>     *d_tile_status,
        TempStorage                 &temp_storage,
        ScanOp                      scan_op,
        int                         tile_idx) :
            d_tile_status(d_tile_status),
            temp_storage(temp_storage),
            scan_op(scan_op),
            tile_idx(tile_idx) {}


    // Block until all predecessors within the specified window have non-invalid status
    __device__ __forceinline__
    void ProcessWindow(
        int                         predecessor_idx,
        int                         &predecessor_status,
        T                           &window_prefix)
    {
        T value;
        DeviceScanTileStatus<T>::WaitForValid(d_tile_status + predecessor_idx, predecessor_status, value);

        // Perform a segmented reduction to get the prefix for the current window
        int flag = (predecessor_status != DEVICE_SCAN_TILE_PARTIAL);
        window_prefix = WarpReduceT(temp_storage).TailSegmentedReduce(value, flag, scan_op);
    }


    // Prefix functor (called by the first warp)
    __device__ __forceinline__
    T operator()(T block_aggregate)
    {
        // Update our status with our tile-aggregate
        if (threadIdx.x == 0)
        {
            DeviceScanTileStatus<T>::SetPartial(d_tile_status + tile_idx, block_aggregate);
        }

        // Wait for the window of predecessor blocks to become valid
        int predecessor_idx = tile_idx - threadIdx.x - 1;
        int predecessor_status;
        T window_prefix;
        ProcessWindow(predecessor_idx, predecessor_status, window_prefix);

        // The block prefix starts out as the current window prefix
        T block_prefix = window_prefix;

        // Keep sliding the window back until we come across a tile whose prefix (not aggregate) is known
        while (WarpAll(predecessor_status != DEVICE_SCAN_TILE_PREFIX))
        {
            predecessor_idx -= PtxArchProps::WARP_THREADS;

            // Update block prefix with the window prefix
            ProcessWindow(predecessor_idx, predecessor_status, window_prefix);
            block_prefix = scan_op(window_prefix, block_prefix);
        }

        // Update our status with our inclusive block_prefix
        if (threadIdx.x == 0)
        {
            DeviceScanTileStatus<T>::SetPrefix(
                d_tile_status + tile_idx,
                scan_op(block_prefix, block_aggregate));
        }

        // Return block-wide exclusive block_prefix
        return block_prefix;
    }
};


}               // CUB namespace
CUB_NS_POSTFIX  // Optional outer namespace(s)

