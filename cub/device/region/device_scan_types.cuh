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
 * Utility types for device-wide scan and similar primitives
 */

#pragma once

#include <iterator>

#include "../../thread/thread_load.cuh"
#include "../../thread/thread_store.cuh"
#include "../../warp/warp_reduce.cuh"
#include "../../util_arch.cuh"
#include "../../util_namespace.cuh"

/// Optional outer namespace(s)
CUB_NS_PREFIX

/// CUB namespace
namespace cub {


/******************************************************************************
 * Callback operator types for supplying BlockScan prefixes
 ******************************************************************************/

/**
 * Stateful callback operator type for supplying BlockScan prefixes.
 * Maintains a running prefix that can be applied to consecutive
 * BlockScan operations.
 */
template <
    typename T,                 ///< BlockScan value type
    typename ScanOp>            ///< Wrapped scan operator type
struct RunningBlockPrefixCallbackOp
{
    ScanOp  op;                 ///< Wrapped scan operator
    T       running_total;      ///< Running block-wide prefix

    /// Constructor
    __device__ __forceinline__ RunningBlockPrefixCallbackOp(ScanOp op)
    :
        op(op)
    {}

    /// Constructor
    __device__ __forceinline__ RunningBlockPrefixCallbackOp(
        T starting_prefix,
        ScanOp op)
    :
        op(op),
        running_total(starting_prefix)
    {}

    /**
     * Prefix callback operator.  Returns the block-wide running_total in thread-0.
     */
    __device__ __forceinline__ T operator()(
        const T &block_aggregate)              ///< The aggregate sum of the BlockScan inputs
    {
        T retval = running_total;
        running_total = op(running_total, block_aggregate);
        return retval;
    }
};



/******************************************************************************
 * Scan data types
 ******************************************************************************/

/**
 * Scan data type for two-way partitioning
 */
template <typename Offset>          ///< Signed integer type for global offsets
struct DevicePartitionScanTuple
{
    Offset   first_count;
    Offset   second_count;

    __device__ __host__ __forceinline__ DevicePartitionScanTuple operator+(
        const DevicePartitionScanTuple &other)
    {
        DevicePartitionScanTuple retval;
        retval.first_count = first_count + other.first_count;
        retval.second_count = second_count + other.second_count;
        return retval;
    }
};


/******************************************************************************
 * Bookkeeping types for single-pass device-wide scan with dynamic lookback
 ******************************************************************************/

/**
 * Enumerations of tile status
 */
enum LookbackTileStatus
{
    LOOKBACK_TILE_OOB,          // Out-of-bounds (e.g., padding)
    LOOKBACK_TILE_INVALID,      // Not yet processed
    LOOKBACK_TILE_PARTIAL,      // Tile aggregate is available
    LOOKBACK_TILE_PREFIX,       // Inclusive tile prefix is available
};




/**
 * Data type of tile status descriptor.
 *
 * Specialized for scan status and value types that can be combined into the same
 * machine word that can be read/written coherently in a single access.
 */
template <
    typename    T,
    bool        SINGLE_WORD = (PowerOfTwo<sizeof(T)>::VALUE && (sizeof(T) <= 8))>
struct LookbackTileDescriptor
{
    // Status word type
    typedef typename UnitWord<T>::VolatileWord StatusWord;

    // Unit word type
    typedef typename If<(sizeof(T) == 8),
        longlong2,
        typename If<(sizeof(T) == 4),
            int2,
            typename If<(sizeof(T) == 2),
                int,
                char2>::Type>::Type>::Type TxnWord;

    StatusWord  status;
    T           value;

    static __device__ __forceinline__ void SetPrefix(LookbackTileDescriptor *ptr, T prefix)
    {
        LookbackTileDescriptor tile_descriptor;
        tile_descriptor.status = LOOKBACK_TILE_PREFIX;
        tile_descriptor.value = prefix;

        TxnWord alias;
        *reinterpret_cast<LookbackTileDescriptor*>(&alias) = tile_descriptor;
        ThreadStore<STORE_CG>(reinterpret_cast<TxnWord*>(ptr), alias);
    }

    static __device__ __forceinline__ void SetPartial(LookbackTileDescriptor *ptr, T partial)
    {
        LookbackTileDescriptor tile_descriptor;
        tile_descriptor.status = LOOKBACK_TILE_PARTIAL;
        tile_descriptor.value = partial;

        TxnWord alias;
        *reinterpret_cast<LookbackTileDescriptor*>(&alias) = tile_descriptor;
        ThreadStore<STORE_CG>(reinterpret_cast<TxnWord*>(ptr), alias);
    }

    static __device__ __forceinline__ void WaitForValid(
        LookbackTileDescriptor    *ptr,
        int                     &status,
        T                       &value)
    {
        LookbackTileDescriptor tile_descriptor;
        tile_descriptor.status = LOOKBACK_TILE_INVALID;

#if CUB_PTX_VERSION == 100

        // Use shared memory to determine when all threads have valid status
        __shared__ volatile int done;

        do
        {
            if (threadIdx.x == 0) done = 1;

            TxnWord alias = ThreadLoad<LOAD_CG>(reinterpret_cast<TxnWord*>(ptr));
            __threadfence_block();

            tile_descriptor = reinterpret_cast<LookbackTileDescriptor&>(alias);
            if (tile_descriptor.status == LOOKBACK_TILE_INVALID)
                done = 0;

        } while (done == 0);

#else

        // Use warp-any to determine when all threads have valid status
        bool invalid = true;
        do
        {
            if (invalid)
            {
                TxnWord alias = ThreadLoad<LOAD_CG>(reinterpret_cast<TxnWord*>(ptr));
                __threadfence_block();

                tile_descriptor = reinterpret_cast<LookbackTileDescriptor&>(alias);
                invalid = tile_descriptor.status == LOOKBACK_TILE_INVALID;
            }
        } while (__any(invalid));
#endif

        status = tile_descriptor.status;
        value = tile_descriptor.value;
    }

};



template <typename T>
struct LookbackTileDescriptor<T, false>
{
    T       prefix_value;
    T       partial_value;
    T       value;

    /// Workaround for the fact that win32 doesn't guarantee 16B alignment 16B values of T
    union
    {
        int                     status;
        Uninitialized<T>        padding;
    };

    static __device__ __forceinline__ void SetPrefix(LookbackTileDescriptor *ptr, T prefix)
    {
        ThreadStore<STORE_CG>(&ptr->prefix_value, prefix);
        __threadfence_block();
//        __threadfence();        // __threadfence_block seems sufficient on current architectures to prevent reordeing
        ThreadStore<STORE_CG>(&ptr->status, (int) LOOKBACK_TILE_PREFIX);

    }

    static __device__ __forceinline__ void SetPartial(LookbackTileDescriptor *ptr, T partial)
    {
        ThreadStore<STORE_CG>(&ptr->partial_value, partial);
        __threadfence_block();
//        __threadfence();        // __threadfence_block seems sufficient on current architectures to prevent reordeing
        ThreadStore<STORE_CG>(&ptr->status, (int) LOOKBACK_TILE_PARTIAL);
    }

    static __device__ __forceinline__ void WaitForValid(
        LookbackTileDescriptor    *ptr,
        int                         &status,
        T                           &value)
    {

        while (true)
        {
            status = ThreadLoad<LOAD_CG>(&ptr->status);
            if (WarpAll(status != LOOKBACK_TILE_INVALID)) break;

            __threadfence_block();
        }

        value = (status == LOOKBACK_TILE_PARTIAL) ?
            ThreadLoad<LOAD_CG>(&ptr->partial_value) :
            ThreadLoad<LOAD_CG>(&ptr->prefix_value);

    }
};



/**
 * Stateful block-scan prefix functor.  Provides the the running prefix for
 * the current tile by using the call-back warp to wait on on
 * aggregates/prefixes from predecessor tiles to become available.
 */
template <
    typename T,
    typename ScanOp>
struct LookbackBlockPrefixCallbackOp
{
    // Parameterized warp reduce
    typedef WarpReduce<T>                       WarpReduceT;

    // Storage type
    typedef typename WarpReduceT::TempStorage   _TempStorage;

    // Alias wrapper allowing storage to be unioned
    typedef Uninitialized<_TempStorage>         TempStorage;

    // Tile status descriptor type
    typedef LookbackTileDescriptor<T>           LookbackTileDescriptorT;

    // Fields
    LookbackTileDescriptorT     *d_tile_status;     ///< Pointer to array of tile status
    _TempStorage                &temp_storage;      ///< Reference to a warp-reduction instance
    ScanOp                      scan_op;            ///< Binary scan operator
    int                         tile_idx;           ///< The current tile index
    T                           exclusive_prefix;   ///< Exclusive prefix for the tile
    T                           inclusive_prefix;   ///< Inclusive prefix for the tile

    // Constructor
    __device__ __forceinline__
    LookbackBlockPrefixCallbackOp(
        LookbackTileDescriptorT     *d_tile_status,
        TempStorage                 &temp_storage,
        ScanOp                      scan_op,
        int                         tile_idx) :
            d_tile_status(d_tile_status),
            temp_storage(temp_storage.Alias()),
            scan_op(scan_op),
            tile_idx(tile_idx) {}


    // Block until all predecessors within the warp-wide window have non-invalid status
    __device__ __forceinline__
    void ProcessWindow(
        int predecessor_idx,        ///< Preceding tile index to inspect
        int &predecessor_status,    ///< [out] Preceding tile status
        T   &window_aggregate)      ///< [out] Relevant partial reduction from this window of preceding tiles
    {
        T value;
        LookbackTileDescriptorT::WaitForValid(d_tile_status + predecessor_idx, predecessor_status, value);

        // Perform a segmented reduction to get the prefix for the current window
        int tail_flag = (predecessor_status == LOOKBACK_TILE_PREFIX);
        window_aggregate = WarpReduceT(temp_storage).TailSegmentedReduce(value, tail_flag, scan_op);
    }


    // BlockScan prefix callback functor (called by the first warp)
    __device__ __forceinline__
    T operator()(T block_aggregate)
    {
        // Update our status with our tile-aggregate
        if (threadIdx.x == 0)
        {
            LookbackTileDescriptorT::SetPartial(d_tile_status + tile_idx, block_aggregate);
        }

        // Wait for the warp-wide window of predecessor tiles to become valid
        int predecessor_idx = tile_idx - threadIdx.x - 1;
        int predecessor_status;
        T window_aggregate;
        ProcessWindow(predecessor_idx, predecessor_status, window_aggregate);

        // The exclusive tile prefix starts out as the current window aggregate
        exclusive_prefix = window_aggregate;

        // Keep sliding the window back until we come across a tile whose inclusive prefix is known
        while (WarpAll(predecessor_status != LOOKBACK_TILE_PREFIX))
        {
            predecessor_idx -= CUB_PTX_WARP_THREADS;

            // Update exclusive tile prefix with the window prefix
            ProcessWindow(predecessor_idx, predecessor_status, window_aggregate);
            exclusive_prefix = scan_op(window_aggregate, exclusive_prefix);
        }

        // Compute the inclusive tile prefix and update the status for this tile
        if (threadIdx.x == 0)
        {
            inclusive_prefix = scan_op(exclusive_prefix, block_aggregate);
            LookbackTileDescriptorT::SetPrefix(
                d_tile_status + tile_idx,
                inclusive_prefix);
        }

        // Return exclusive_prefix
        return exclusive_prefix;
    }
};


}               // CUB namespace
CUB_NS_POSTFIX  // Optional outer namespace(s)

