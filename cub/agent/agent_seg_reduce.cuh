/******************************************************************************
 * Copyright (c) 2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2015, NVIDIA CORPORATION.  All rights reserved.
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
 * cub::AgentSegReduce implements a stateful abstraction of CUDA thread blocks for participating in device-wide segmented reduction.
 */

#pragma once

#include <iterator>

#include "../util_type.cuh"
#include "../block/block_reduce.cuh"
#include "../block/block_scan.cuh"
#include "../thread/thread_search.cuh"
#include "../thread/thread_operators.cuh"
#include "../iterator/cache_modified_input_iterator.cuh"
#include "../iterator/counting_input_iterator.cuh"
#include "../util_namespace.cuh"

/// Optional outer namespace(s)
CUB_NS_PREFIX

/// CUB namespace
namespace cub {


/******************************************************************************
 * Tuning policy
 ******************************************************************************/

/**
 * Parameterizable tuning policy type for AgentSegReduce
 */
template <
    int                             _BLOCK_THREADS,                         ///< Threads per thread block
    int                             _ITEMS_PER_THREAD,                      ///< Items per thread (per tile of input)
    CacheLoadModifier               _SEARCH_ROW_OFFSETS_LOAD_MODIFIER,      ///< Cache load modifier for reading CSR row-offsets during search
    CacheLoadModifier               _ROW_OFFSETS_LOAD_MODIFIER,             ///< Cache load modifier for reading CSR row-offsets
    CacheLoadModifier               _VALUES_LOAD_MODIFIER,                  ///< Cache load modifier for reading CSR values
    bool                            _DIRECT_LOAD_NONZEROS,                  ///< Whether to load values directly from global during sequential merging (pre-staged through shared memory)
    BlockScanAlgorithm              _SCAN_ALGORITHM>                        ///< The BlockScan algorithm to use
struct AgentSegReducePolicy
{
    enum
    {
        BLOCK_THREADS                                                   = _BLOCK_THREADS,                       ///< Threads per thread block
        ITEMS_PER_THREAD                                                = _ITEMS_PER_THREAD,                    ///< Items per thread (per tile of input)
        DIRECT_LOAD_NONZEROS                                            = _DIRECT_LOAD_NONZEROS,                ///< Whether to load values directly from global during sequential merging (vs. pre-staged through shared memory)
    };

    static const CacheLoadModifier  SEARCH_ROW_OFFSETS_LOAD_MODIFIER    = _SEARCH_ROW_OFFSETS_LOAD_MODIFIER;    ///< Cache load modifier for reading CSR row-offsets
    static const CacheLoadModifier  ROW_OFFSETS_LOAD_MODIFIER           = _ROW_OFFSETS_LOAD_MODIFIER;           ///< Cache load modifier for reading CSR row-offsets
    static const CacheLoadModifier  VALUES_LOAD_MODIFIER                = _VALUES_LOAD_MODIFIER;                ///< Cache load modifier for reading CSR values
    static const BlockScanAlgorithm SCAN_ALGORITHM                      = _SCAN_ALGORITHM;                      ///< The BlockScan algorithm to use

};


/******************************************************************************
 * Thread block abstractions
 ******************************************************************************/

template <
    typename        T,                      ///< Value type
    typename        OffsetT,                ///< Signed integer type for sequence offsets
    typename        ReductionOpT>           ///< Binary reduction functor type having member <tt>T operator()(const T &a, const T &b)</tt> (e.g., cub::Sum, cub::Min, cub::Max, etc.)
struct SegReduceParams
{
    T*              d_values;               ///< Pointer to the array of \p num_values values of the corresponding nonzero elements of matrix <b>A</b>.
    OffsetT*        d_row_end_offsets;      ///< Pointer to the array of \p m offsets demarcating the end of every row in \p d_column_indices and \p d_values
    T*              d_out;                  ///< Pointer to the array of \p num_rows values corresponding to the dense output vector <em>y</em>
    int             num_rows;               ///< Number of rows of matrix <b>A</b>.
    int             num_values;             ///< Number of nonzero elements of matrix <b>A</b>.
    ReductionOpT    reduction_op;           ///< Binary reduction functor
    T               identity;               ///< Identity value to use for zero-length rows
};


/**
 * \brief AgentSegReduce implements a stateful abstraction of CUDA thread blocks for participating in device-wide segmented reduction.
 */
template <
    typename    AgentSegReducePolicyT,      ///< Parameterized AgentSegReducePolicy tuning policy type
    typename    T,                          ///< Value type
    typename    OffsetT,                    ///< Signed integer type for sequence offsets
    typename    ReductionOpT,               ///< Binary reduction operator type having member <tt>T operator()(const T &a, const T &b)</tt>
    int         PTX_ARCH = CUB_PTX_ARCH>    ///< PTX compute capability
struct AgentSegReduce
{
    //---------------------------------------------------------------------
    // Types and constants
    //---------------------------------------------------------------------

    /// Constants
    enum
    {
        BLOCK_THREADS           = AgentSegReducePolicyT::BLOCK_THREADS,
        ITEMS_PER_THREAD        = AgentSegReducePolicyT::ITEMS_PER_THREAD,
        TILE_ITEMS              = BLOCK_THREADS * ITEMS_PER_THREAD,
    };

    /// 2D merge path coordinate type
    typedef typename CubVector<OffsetT, 2>::Type CoordinateT;

    /// Input iterator wrapper types (for applying cache modifiers)

    typedef CacheModifiedInputIterator<
            AgentSegReducePolicyT::ROW_OFFSETS_LOAD_MODIFIER,
            OffsetT,
            OffsetT>
        RowOffsetsIteratorT;

    typedef CacheModifiedInputIterator<
            AgentSegReducePolicyT::VALUES_LOAD_MODIFIER,
            T,
            OffsetT>
        ValueIteratorT;

    // Tuple type for scanning (pairs accumulated segment-value with segment-index)
    typedef ItemOffsetPair<T, OffsetT> ItemOffsetPairT;

    // Tile status descriptor interface type
    typedef ReduceByKeyScanTileState<T, OffsetT> ScanTileStateT;

    // Reduce-value-by-segment scan operator
    typedef ReduceBySegmentOp<
            cub::Sum,
            ItemOffsetPairT>
        ReduceBySegmentOpT;

    // BlockReduce specialization
    typedef BlockReduce<
            T,
            BLOCK_THREADS,
            BLOCK_REDUCE_WARP_REDUCTIONS>
        BlockReduceT;

    // BlockScan specialization
    typedef BlockScan<
            ItemOffsetPairT,
            BLOCK_THREADS,
            AgentSegReducePolicyT::SCAN_ALGORITHM>
        BlockScanT;

    /// Merge item type (either a non-zero value or a row-end offset)
    union MergeItem
    {
        // Value type to pair with index type OffsetT (NullType if loading values directly during merge)
        typedef typename If<AgentSegReducePolicyT::DIRECT_LOAD_NONZEROS, NullType, T>::Type MergeValueT;

        OffsetT row_end_offset;
        MergeValueT nonzero;
    };

    /// Shared memory type required by this thread block
    struct _TempStorage
    {
        // Starting and ending global merge path coordinates for the tile
        CoordinateT tile_coordinates[2];

        // Smem needed for tile of merge items
        MergeItem merge_items[ITEMS_PER_THREAD + TILE_ITEMS + 1];

        union
        {

            // Smem needed for block-wide reduction
            typename BlockReduceT::TempStorage reduce;

            // Smem needed for tile scanning
            typename BlockScanT::TempStorage scan;
        };
    };

    /// Temporary storage type (unionable)
    struct TempStorage : Uninitialized<_TempStorage> {};


    //---------------------------------------------------------------------
    // Per-thread fields
    //---------------------------------------------------------------------


    _TempStorage&                               temp_storage;           /// Reference to temp_storage
    SegReduceParams<T, OffsetT, ReductionOpT>&  params;
    ValueIteratorT                              wd_values;              ///< Wrapped pointer to the array of \p num_values values of the corresponding nonzero elements of matrix <b>A</b>.
    RowOffsetsIteratorT                         wd_row_end_offsets;     ///< Wrapped Pointer to the array of \p m offsets demarcating the end of every row in \p d_column_indices and \p d_values


    //---------------------------------------------------------------------
    // Interface
    //---------------------------------------------------------------------

    /**
     * Constructor
     */
    __device__ __forceinline__ AgentSegReduce(
        TempStorage&                                temp_storage,       ///< Reference to temp_storage
        SegReduceParams<T, OffsetT, ReductionOpT>&  params)             ///< SpMV input parameter bundle
    :
        temp_storage(temp_storage.Alias()),
        params(params),
        wd_values(params.d_values),
        wd_row_end_offsets(params.d_row_end_offsets)
    {}


    /**
     * Consume a merge tile, specialized for direct-load of values
     */
    __device__ __forceinline__ ItemOffsetPairT ConsumeTile(
        int             tile_idx,
        CoordinateT     tile_start_coord,
        CoordinateT     tile_end_coord,
        Int2Type<true>  is_direct_load)     ///< Marker type indicating whether to load values directly during path-discovery or beforehand in batch
    {
        ItemOffsetPairT tile_aggregate;

        int         tile_num_rows           = tile_end_coord.x - tile_start_coord.x;
        int         tile_num_values         = tile_end_coord.y - tile_start_coord.y;
        OffsetT*    s_tile_row_end_offsets  = reinterpret_cast<OffsetT*>(temp_storage.merge_items);

        // Gather the row end-offsets for the merge tile into shared memory
        for (int item = threadIdx.x; item < tile_num_rows; item += BLOCK_THREADS)
        {
            s_tile_row_end_offsets[item] = wd_row_end_offsets[tile_start_coord.x + item];
        }

        // Pad the end of the merge tile's axis of row-end offsets
        if (threadIdx.x < ITEMS_PER_THREAD)
        {
            s_tile_row_end_offsets[tile_num_rows + threadIdx.x] = params.num_values;
        }

        __syncthreads();

        // Search for the thread's starting coordinate within the merge tile
        CountingInputIterator<OffsetT>  tile_value_indices(tile_start_coord.y);
        CoordinateT                     thread_start_coord;

        MergePathSearch(
            OffsetT(threadIdx.x * ITEMS_PER_THREAD),    // Diagonal
            s_tile_row_end_offsets,                     // List A
            tile_value_indices,                         // List B
            tile_num_rows,
            tile_num_values,
            thread_start_coord);

        __syncthreads();    // Perf-sync

        // Compute the thread's merge path segment
        CoordinateT         thread_current_coord = thread_start_coord;
        ItemOffsetPairT     thread_segment[ITEMS_PER_THREAD];
        OffsetT             row_indices[ITEMS_PER_THREAD];

        #pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
        {
            OffsetT value_idx   = CUB_MIN(tile_value_indices[thread_current_coord.y], params.num_values - 1);
            T  value            = wd_values[value_idx];

            bool accumulate = (tile_value_indices[thread_current_coord.y] < s_tile_row_end_offsets[thread_current_coord.x]);
            if (accumulate)
            {
                // Move down (accumulate)
                thread_segment[ITEM].value = value;
                thread_segment[ITEM].offset = 0;
                row_indices[ITEM] = tile_num_rows;
                ++thread_current_coord.y;
            }
            else
            {
                // Move right (reset)
                thread_segment[ITEM].value = params.identity;
                thread_segment[ITEM].offset = 1;
                row_indices[ITEM] = thread_current_coord.x;
                ++thread_current_coord.x;
            }
        }

        __syncthreads();    // Perf-sync

        // Block-wide reduce-value-by-segment
        ItemOffsetPairT     scan_identity(params.identity, 0);
        ReduceBySegmentOpT  scan_op;

        BlockScanT(temp_storage.scan).ExclusiveScan(thread_segment, thread_segment, scan_identity, scan_op, tile_aggregate);

        // Direct scatter
        #pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
        {
            if (row_indices[ITEM] < tile_num_rows)
            {
                // Set the output vector element
                params.d_out[tile_start_coord.x + thread_segment[ITEM].offset] = thread_segment[ITEM].value;
//                  params.d_out[tile_start_coord.x + row_indices[ITEM]] = thread_segment[ITEM].value;
            }
        }

        // Return the tile's running carry-out
        return tile_aggregate;
    }



    /**
     * Consume a merge tile, specialized for indirect load of values
     */
    __device__ __forceinline__ ItemOffsetPairT ConsumeTile(
        int             tile_idx,
        CoordinateT     tile_start_coord,
        CoordinateT     tile_end_coord,
        Int2Type<false> is_direct_load)     ///< Marker type indicating whether to load values directly during path-discovery or beforehand in batch
    {
        int         tile_num_rows           = tile_end_coord.x - tile_start_coord.x;
        int         tile_num_values         = tile_end_coord.y - tile_start_coord.y;
        OffsetT*    s_tile_row_end_offsets  = reinterpret_cast<OffsetT*>(temp_storage.merge_items);
        T*          s_tile_values           = reinterpret_cast<T*>(temp_storage.merge_items + tile_num_rows + ITEMS_PER_THREAD);

        if (tile_num_values > 0)
        {
            // Gather the values for the merge tile into shared memory
            #pragma unroll
            for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
            {
                int value_idx               = threadIdx.x + (ITEM * BLOCK_THREADS);
                value_idx                   = CUB_MIN(value_idx, tile_num_values - 1);
                T value                     = wd_values[tile_start_coord.y + value_idx];
                s_tile_values[value_idx]    = value;
            }

            __syncthreads();    // Perf-sync
        }

        // Gather the row end-offsets for the merge tile into shared memory
        #pragma unroll 1
        for (int item = threadIdx.x; item < tile_num_rows; item += BLOCK_THREADS)
        {
            s_tile_row_end_offsets[item] = wd_row_end_offsets[tile_start_coord.x + item];
        }

        // Pad the end of the merge tile's axis of row-end offsets
        if (threadIdx.x < ITEMS_PER_THREAD)
        {
            s_tile_row_end_offsets[tile_num_rows + threadIdx.x] = params.num_values;
        }

        __syncthreads();

        // Search for the thread's starting coordinate within the merge tile
        CountingInputIterator<OffsetT>  tile_value_indices(tile_start_coord.y);
        CoordinateT                     thread_start_coord;

        MergePathSearch(
            OffsetT(threadIdx.x * ITEMS_PER_THREAD),    // Diagonal
            s_tile_row_end_offsets,                     // List A
            tile_value_indices,                         // List B
            tile_num_rows,
            tile_num_values,
            thread_start_coord);

        __syncthreads();    // Perf-sync

        // Compute the thread's merge path segment
        CoordinateT         thread_current_coord = thread_start_coord;
        ItemOffsetPairT     thread_segment[ITEMS_PER_THREAD];
        T                   running_total = params.identity;

        #pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
        {
            bool accumulate = (tile_value_indices[thread_current_coord.y] < s_tile_row_end_offsets[thread_current_coord.x]);

            if (accumulate)
            {
                // Move down (accumulate)
                running_total += s_tile_values[thread_current_coord.y];
                thread_segment[ITEM].offset = tile_num_rows;
                ++thread_current_coord.y;
            }
            else
            {
                // Move right (reset)
                thread_segment[ITEM].value  = running_total;
                running_total = params.identity;
                thread_segment[ITEM].offset = thread_current_coord.x;
                ++thread_current_coord.x;
            }
        }

        // Block-wide reduce-value-by-key
        // Compute the inter-thread fix-up from thread-carries (for when a row spans more than one thread-segment)
        int                                             thread_num_rows = thread_current_coord.x - thread_start_coord.x;
        ItemOffsetPairT                                 tile_carry;
        ItemOffsetPairT                                 scan_item(running_total, thread_num_rows);
        ItemOffsetPairT                                 scan_identity(params.identity, 0);
        ReduceBySegmentOp<cub::Sum, ItemOffsetPairT>    scan_op;
        BlockScanT(temp_storage.scan).ExclusiveScan(scan_item, scan_item, scan_identity, scan_op, tile_carry);

        // Two phase scatter
        T* s_partials = reinterpret_cast<T*>(temp_storage.merge_items);

        #pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
        {
            if (thread_segment[ITEM].offset < tile_num_rows)
            {
                thread_segment[ITEM].value += scan_item.value;
                scan_item.value = params.identity;

                s_partials[thread_segment[ITEM].offset] = thread_segment[ITEM].value;
            }
        }

        __syncthreads();

        #pragma unroll 1
        for (int item = threadIdx.x; item < tile_num_rows; item += BLOCK_THREADS)
        {
            params.d_out[tile_start_coord.x + item] = s_partials[item];
        }

        // Return the tile's running carry-out
        return tile_carry;
    }



    /**
     * Consume input tile
     */
    __device__ __forceinline__ void ConsumeTile(
        CoordinateT*    d_tile_coordinates,     ///< [in] Pointer to the temporary array of tile starting coordinates
        OffsetT*        d_tile_carry_rows,      ///< [out] Pointer to the array of carry-out row-ids, one per tile
        T*              d_tile_carry_values,    ///< [out] Pointer to the array of carry-out partial-sums, one per tile
        int             num_tiles)              ///< [in] Number of merge tiles
    {
        // Get current tile index
        int tile_idx = (blockIdx.x * gridDim.y) + blockIdx.y;
        if (tile_idx >= num_tiles)
            return;

        // Get the 2D start and end coordinates from the merge kernel
        CoordinateT tile_start_coord     = d_tile_coordinates[tile_idx + 0];
        CoordinateT tile_end_coord       = d_tile_coordinates[tile_idx + 1];

        // Consume multi-segment tile
         ItemOffsetPairT tile_aggregate = ConsumeTile(
            tile_idx,
            tile_start_coord,
            tile_end_coord,
            Int2Type<AgentSegReducePolicyT::DIRECT_LOAD_NONZEROS>());

        // Output the tile's carry-out
        if (threadIdx.x == 0)
        {
            d_tile_carry_rows[tile_idx]     = tile_aggregate.offset + tile_start_coord.x;
            d_tile_carry_values[tile_idx]   = tile_aggregate.value;
        }
    }


};




}               // CUB namespace
CUB_NS_POSTFIX  // Optional outer namespace(s)

