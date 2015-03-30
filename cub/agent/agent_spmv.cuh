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
 * cub::AgentSpmv implements a stateful abstraction of CUDA thread blocks for participating in device-wide SpMV.
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
 * Parameterizable tuning policy type for AgentSpmv
 */
template <
    int                             _BLOCK_THREADS,                         ///< Threads per thread block
    int                             _ITEMS_PER_THREAD,                      ///< Items per thread (per tile of input)
    CacheLoadModifier               _SEARCH_ROW_OFFSETS_LOAD_MODIFIER,      ///< Cache load modifier for reading CSR row-offsets during search
    CacheLoadModifier               _ROW_OFFSETS_LOAD_MODIFIER,             ///< Cache load modifier for reading CSR row-offsets
    CacheLoadModifier               _COLUMN_INDICES_LOAD_MODIFIER,          ///< Cache load modifier for reading CSR column-indices
    CacheLoadModifier               _VALUES_LOAD_MODIFIER,                  ///< Cache load modifier for reading CSR values
    CacheLoadModifier               _VECTOR_VALUES_LOAD_MODIFIER,           ///< Cache load modifier for reading vector values
    bool                            _DIRECT_LOAD_NONZEROS,                  ///< Whether to load nonzeros directly from global during sequential merging (vs. pre-staged through shared memory)
    BlockScanAlgorithm              _SCAN_ALGORITHM>                        ///< The BlockScan algorithm to use
struct AgentSpmvPolicy
{
    enum
    {
        BLOCK_THREADS                                                   = _BLOCK_THREADS,                       ///< Threads per thread block
        ITEMS_PER_THREAD                                                = _ITEMS_PER_THREAD,                    ///< Items per thread (per tile of input)
        DIRECT_LOAD_NONZEROS                                            = _DIRECT_LOAD_NONZEROS,                ///< Whether to load nonzeros directly from global during sequential merging (pre-staged through shared memory) 
    };

    static const CacheLoadModifier  SEARCH_ROW_OFFSETS_LOAD_MODIFIER    = _SEARCH_ROW_OFFSETS_LOAD_MODIFIER;    ///< Cache load modifier for reading CSR row-offsets
    static const CacheLoadModifier  ROW_OFFSETS_LOAD_MODIFIER           = _ROW_OFFSETS_LOAD_MODIFIER;           ///< Cache load modifier for reading CSR row-offsets
    static const CacheLoadModifier  COLUMN_INDICES_LOAD_MODIFIER        = _COLUMN_INDICES_LOAD_MODIFIER;        ///< Cache load modifier for reading CSR column-indices
    static const CacheLoadModifier  VALUES_LOAD_MODIFIER                = _VALUES_LOAD_MODIFIER;                ///< Cache load modifier for reading CSR values
    static const CacheLoadModifier  VECTOR_VALUES_LOAD_MODIFIER         = _VECTOR_VALUES_LOAD_MODIFIER;         ///< Cache load modifier for reading vector values
    static const BlockScanAlgorithm SCAN_ALGORITHM                      = _SCAN_ALGORITHM;                      ///< The BlockScan algorithm to use

};


/******************************************************************************
 * Thread block abstractions
 ******************************************************************************/

template <
    typename        ValueT,              ///< Matrix and vector value type
    typename        OffsetT>             ///< Signed integer type for sequence offsets
struct SpmvParams
{
    ValueT*         d_values;            ///< Pointer to the array of \p num_nonzeros values of the corresponding nonzero elements of matrix <b>A</b>.
    OffsetT*        d_row_end_offsets;   ///< Pointer to the array of \p m offsets demarcating the end of every row in \p d_column_indices and \p d_values
    OffsetT*        d_column_indices;    ///< Pointer to the array of \p num_nonzeros column-indices of the corresponding nonzero elements of matrix <b>A</b>.  (Indices are zero-valued.)
    ValueT*         d_vector_x;          ///< Pointer to the array of \p num_cols values corresponding to the dense input vector <em>x</em>
    ValueT*         d_vector_y;          ///< Pointer to the array of \p num_rows values corresponding to the dense output vector <em>y</em>
    int             num_rows;            ///< Number of rows of matrix <b>A</b>.
    int             num_cols;            ///< Number of columns of matrix <b>A</b>.
    int             num_nonzeros;        ///< Number of nonzero elements of matrix <b>A</b>.
    ValueT          alpha;               ///< Alpha multiplicand
    ValueT          beta;                ///< Beta addend-multiplicand
};


/**
 * \brief AgentSpmv implements a stateful abstraction of CUDA thread blocks for participating in device-wide SpMV.
 */
template <
    typename    AgentSpmvPolicyT,           ///< Parameterized AgentSpmvPolicy tuning policy type
    typename    ValueT,                     ///< Matrix and vector value type
    typename    OffsetT,                    ///< Signed integer type for sequence offsets
    bool        HAS_ALPHA,                  ///< Whether the input parameter \p alpha is 1
    bool        HAS_BETA,                   ///< Whether the input parameter \p beta is 0 
    int         PTX_ARCH = CUB_PTX_ARCH>    ///< PTX compute capability
struct AgentSpmv
{
    //---------------------------------------------------------------------
    // Types and constants
    //---------------------------------------------------------------------

    /// Constants
    enum
    {
        BLOCK_THREADS           = AgentSpmvPolicyT::BLOCK_THREADS,
        ITEMS_PER_THREAD        = AgentSpmvPolicyT::ITEMS_PER_THREAD,
        TILE_ITEMS              = BLOCK_THREADS * ITEMS_PER_THREAD,
    };

    /// 2D merge path coordinate type
    typedef typename CubVector<OffsetT, 2>::Type CoordinateT;

    /// Input iterator wrapper types (for applying cache modifiers)

    typedef CacheModifiedInputIterator<
            AgentSpmvPolicyT::ROW_OFFSETS_LOAD_MODIFIER,
            OffsetT,
            OffsetT>
        RowOffsetsIteratorT;

    typedef CacheModifiedInputIterator<
            AgentSpmvPolicyT::COLUMN_INDICES_LOAD_MODIFIER,
            OffsetT,
            OffsetT>
        ColumnIndicesIteratorT;

    typedef CacheModifiedInputIterator<
            AgentSpmvPolicyT::VALUES_LOAD_MODIFIER,
            ValueT,
            OffsetT>
        ValueIteratorT;

    typedef CacheModifiedInputIterator<
            AgentSpmvPolicyT::VECTOR_VALUES_LOAD_MODIFIER,
            ValueT,
            OffsetT>
        VectorValueIteratorT;

    // Tuple type for scanning (pairs accumulated segment-value with segment-index)
    typedef ItemOffsetPair<ValueT, OffsetT> ItemOffsetPairT;

    // Tile status descriptor interface type
    typedef ReduceByKeyScanTileState<ValueT, OffsetT> ScanTileStateT;

    // Reduce-value-by-segment scan operator
    typedef ReduceBySegmentOp<
            cub::Sum,
            ItemOffsetPairT>
        ReduceBySegmentOpT;

    // BlockReduce specialization
    typedef BlockReduce<
            ValueT,
            BLOCK_THREADS,
            BLOCK_REDUCE_WARP_REDUCTIONS>
        BlockReduceT;

    // BlockScan specialization
    typedef BlockScan<
            ItemOffsetPairT,
            BLOCK_THREADS,
            AgentSpmvPolicyT::SCAN_ALGORITHM>
        BlockScanT;

    /// Merge item type (either a non-zero value or a row-end offset)
    union MergeItem
    {
        // Value type to pair with index type OffsetT (NullType if loading values directly during merge)
        typedef typename If<AgentSpmvPolicyT::DIRECT_LOAD_NONZEROS, NullType, ValueT>::Type MergeValueT;

        OffsetT     row_end_offset;
        MergeValueT nonzero;
    };

    /// Shared memory type required by this thread block
    struct _TempStorage
    {
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


    _TempStorage&                   temp_storage;               /// Reference to temp_storage

    SpmvParams<ValueT, OffsetT>&    spmv_params;

    ValueIteratorT            wd_values;            ///< Wrapped pointer to the array of \p num_nonzeros values of the corresponding nonzero elements of matrix <b>A</b>.
    RowOffsetsIteratorT       wd_row_end_offsets;   ///< Wrapped Pointer to the array of \p m offsets demarcating the end of every row in \p d_column_indices and \p d_values
    ColumnIndicesIteratorT    wd_column_indices;    ///< Wrapped Pointer to the array of \p num_nonzeros column-indices of the corresponding nonzero elements of matrix <b>A</b>.  (Indices are zero-valued.)
    VectorValueIteratorT            wd_vector_x;                 ///< Wrapped Pointer to the array of \p num_cols values corresponding to the dense input vector <em>x</em>
    VectorValueIteratorT            wd_vector_y;                 ///< Wrapped Pointer to the array of \p num_cols values corresponding to the dense input vector <em>x</em>


    //---------------------------------------------------------------------
    // Interface
    //---------------------------------------------------------------------

    /**
     * Constructor
     */
    __device__ __forceinline__ AgentSpmv(
        TempStorage&                    temp_storage,           ///< Reference to temp_storage
        SpmvParams<ValueT, OffsetT>&    spmv_params)            ///< SpMV input parameter bundle
    :
        temp_storage(temp_storage.Alias()),
        spmv_params(spmv_params),
        wd_values(spmv_params.d_values),
        wd_row_end_offsets(spmv_params.d_row_end_offsets),
        wd_column_indices(spmv_params.d_column_indices),
        wd_vector_x(spmv_params.d_vector_x),
        wd_vector_y(spmv_params.d_vector_y)
    {}




    /**
     * Consume a merge tile, specialized for direct-load of nonzeros
     */
    __device__ __forceinline__ ItemOffsetPairT ConsumeTile(
        int             tile_idx,
        CoordinateT     tile_start_coord,
        CoordinateT     tile_end_coord,
        Int2Type<true>  is_direct_load)     ///< Marker type indicating whether to load nonzeros directly during path-discovery or beforehand in batch
    {
        ItemOffsetPairT tile_carry;

        int         tile_num_rows           = tile_end_coord.x - tile_start_coord.x;
        int         tile_num_nonzeros       = tile_end_coord.y - tile_start_coord.y;
        OffsetT*    s_tile_row_end_offsets  = reinterpret_cast<OffsetT*>(temp_storage.merge_items);

        // Gather the row end-offsets for the merge tile into shared memory
        for (int item = threadIdx.x; item < tile_num_rows; item += BLOCK_THREADS)
        {
            s_tile_row_end_offsets[item] = wd_row_end_offsets[tile_start_coord.x + item];
        }

        // Pad the end of the merge tile's axis of row-end offsets
        if (threadIdx.x < ITEMS_PER_THREAD)
        {
            s_tile_row_end_offsets[tile_num_rows + threadIdx.x] = spmv_params.num_nonzeros;
        }

        __syncthreads();

        // Search for the thread's starting coordinate within the merge tile
        CountingInputIterator<OffsetT>  tile_nonzero_indices(tile_start_coord.y);
        CoordinateT                     thread_start_coord;

        MergePathSearch(
            OffsetT(threadIdx.x * ITEMS_PER_THREAD),    // Diagonal
            s_tile_row_end_offsets,                     // List A
            tile_nonzero_indices,                       // List B
            tile_num_rows,
            tile_num_nonzeros,
            thread_start_coord);

        __syncthreads();    // Perf-sync

        // Compute the thread's merge path segment
        CoordinateT         thread_current_coord = thread_start_coord;
        ItemOffsetPairT     thread_segment[ITEMS_PER_THREAD];
        OffsetT             flush[ITEMS_PER_THREAD];

        #pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
        {
            OffsetT nonzero_idx         = CUB_MIN(tile_nonzero_indices[thread_current_coord.y], spmv_params.num_nonzeros - 1);
            OffsetT column_idx          = wd_column_indices[nonzero_idx];
            ValueT  value               = wd_values[nonzero_idx];
            ValueT  vector_value        = wd_vector_x[column_idx];
            ValueT  nonzero             = value * vector_value;

            if (HAS_ALPHA)
                nonzero *= spmv_params.alpha;

            bool accumulate = (tile_nonzero_indices[thread_current_coord.y] < s_tile_row_end_offsets[thread_current_coord.x]);
            if (accumulate)
            {
                // Move down (accumulate)
                thread_segment[ITEM].value = nonzero;
                thread_segment[ITEM].offset = 0;
                flush[ITEM] = 0;
                ++thread_current_coord.y;
            }
            else
            {
                // Move right (reset)
                thread_segment[ITEM].value = 0.0;
                thread_segment[ITEM].offset = 1;
                flush[ITEM] = 1;
                ++thread_current_coord.x;
            }
        }

        __syncthreads();    // Perf-sync

        // Block-wide reduce-value-by-segment
        ItemOffsetPairT     scan_identity(0.0, 0);
        ReduceBySegmentOpT  scan_op;

        BlockScanT(temp_storage.scan).ExclusiveScan(thread_segment, thread_segment, scan_identity, scan_op, tile_carry);

        // Direct scatter
        #pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
        {
            if (flush[ITEM])
            {
                if (HAS_BETA)
                {
                    // Update the output vector element
                    ValueT addend = spmv_params.beta * wd_vector_y[tile_start_coord.x + thread_segment[ITEM].offset];
                    thread_segment[ITEM].value += addend;
                }

                // Set the output vector element
                spmv_params.d_vector_y[tile_start_coord.x + thread_segment[ITEM].offset] = thread_segment[ITEM].value;
            }
        }

        // Return the tile's running carry-out
        return tile_carry;
    }



    /**
     * Consume a merge tile, specialized for indirect load of nonzeros
     */
    __device__ __forceinline__ ItemOffsetPairT ConsumeTile(
        int             tile_idx,
        CoordinateT     tile_start_coord,
        CoordinateT     tile_end_coord,
        Int2Type<false> is_direct_load)     ///< Marker type indicating whether to load nonzeros directly during path-discovery or beforehand in batch
    {
        int         tile_num_rows           = tile_end_coord.x - tile_start_coord.x;
        int         tile_num_nonzeros       = tile_end_coord.y - tile_start_coord.y;
        OffsetT*    s_tile_row_end_offsets  = reinterpret_cast<OffsetT*>(temp_storage.merge_items);
        ValueT*     s_tile_nonzeros         = reinterpret_cast<ValueT*>(temp_storage.merge_items + tile_num_rows + ITEMS_PER_THREAD);

        if (tile_num_nonzeros > 0)
        {
            // Gather the nonzeros for the merge tile into shared memory
            #pragma unroll
            for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
            {
                int     nonzero_idx         = threadIdx.x + (ITEM * BLOCK_THREADS);
                nonzero_idx                 = CUB_MIN(nonzero_idx, tile_num_nonzeros - 1);

                OffsetT column_idx          = wd_column_indices[tile_start_coord.y + nonzero_idx];
                ValueT  value               = wd_values[tile_start_coord.y + nonzero_idx];

                ValueT  vector_value        = wd_vector_x[column_idx];
                ValueT  nonzero             = value * vector_value;

                if (HAS_ALPHA)
                    nonzero *= spmv_params.alpha;

                s_tile_nonzeros[nonzero_idx]       = nonzero;
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
            s_tile_row_end_offsets[tile_num_rows + threadIdx.x] = spmv_params.num_nonzeros;
        }

        __syncthreads();

        // Search for the thread's starting coordinate within the merge tile
        CountingInputIterator<OffsetT>  tile_nonzero_indices(tile_start_coord.y);
        CoordinateT                     thread_start_coord;

        MergePathSearch(
            OffsetT(threadIdx.x * ITEMS_PER_THREAD),            // Diagonal
            s_tile_row_end_offsets,                               // List A
            tile_nonzero_indices,                               // List B
            tile_num_rows,
            tile_num_nonzeros,
            thread_start_coord);

        __syncthreads();    // Perf-sync
/*
        // Compute the thread's merge path segment
        CoordinateT         thread_current_coord = thread_start_coord;
        ItemOffsetPairT     thread_segment[ITEMS_PER_THREAD];
        OffsetT             flush[ITEMS_PER_THREAD];

        #pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
        {
            bool accumulate = (tile_nonzero_indices[thread_current_coord.y] < s_tile_row_end_offsets[thread_current_coord.x]);
            if (accumulate)
            {
                // Move down (accumulate)
                thread_segment[ITEM].value = s_tile_nonzeros[thread_current_coord.y];
                thread_segment[ITEM].offset = tile_num_rows;
                flush[ITEM] = tile_num_rows;
                ++thread_current_coord.y;
            }
            else
            {
                // Move right (reset)
                thread_segment[ITEM].value = 0.0;
                thread_segment[ITEM].offset = 1;
                flush[ITEM] = thread_current_coord.x;
                ++thread_current_coord.x;
            }
        }

        __syncthreads();    // Perf-sync

        // Block-wide reduce-value-by-segment
        ItemOffsetPairT     scan_identity(0.0, 0);
        ReduceBySegmentOpT  scan_op;
        ItemOffsetPairT     tile_carry;
        BlockScanT(temp_storage.scan).ExclusiveScan(thread_segment, thread_segment, scan_identity, scan_op, tile_carry);

        // Two phase scatter
        ValueT* s_partials = reinterpret_cast<ValueT*>(temp_storage.merge_items);

        #pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
        {
            if (flush[ITEM] < tile_num_rows)
            {
                if (HAS_BETA)
                {
                    // Update the output vector element
                    ValueT addend = spmv_params.beta * wd_vector_y[tile_start_coord.x + thread_segment[ITEM].offset];
                    thread_segment[ITEM].value += addend;
                }

                // Set the output vector element
                s_partials[flush[ITEM]] = thread_segment[ITEM].value;
            }
        }

*/

        // Compute the thread's merge path segment
        CoordinateT         thread_current_coord = thread_start_coord;
        ItemOffsetPairT     thread_segment[ITEMS_PER_THREAD];
        ValueT              running_total = 0.0;

        #pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
        {
            bool accumulate = (tile_nonzero_indices[thread_current_coord.y] < s_tile_row_end_offsets[thread_current_coord.x]);

            thread_segment[ITEM].value  = running_total;
            thread_segment[ITEM].offset = thread_current_coord.x;
            running_total += s_tile_nonzeros[thread_current_coord.y];

            if (accumulate)
            {
                // Move down (accumulate)
                thread_segment[ITEM].offset = tile_num_rows;
                ++thread_current_coord.y;
            }
            else
            {
                // Move right (reset)
                running_total = 0.0;
                ++thread_current_coord.x;
            }
        }

        // Block-wide reduce-value-by-key
        // Compute the inter-thread fix-up from thread-carries (for when a row spans more than one thread-segment)
        int                                             thread_num_rows = thread_current_coord.x - thread_start_coord.x;
        ItemOffsetPairT                                 tile_carry;
        ItemOffsetPairT                                 scan_item(running_total, thread_num_rows);
        ItemOffsetPairT                                 scan_identity(0.0, 0);
        ReduceBySegmentOp<cub::Sum, ItemOffsetPairT>    scan_op;
        BlockScanT(temp_storage.scan).ExclusiveScan(scan_item, scan_item, scan_identity, scan_op, tile_carry);

        // Two phase scatter
        ValueT* s_partials = reinterpret_cast<ValueT*>(temp_storage.merge_items);

        #pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
        {
            if (thread_segment[ITEM].offset < tile_num_rows)
            {
                thread_segment[ITEM].value += scan_item.value;
                scan_item.value = 0.0;

                s_partials[thread_segment[ITEM].offset] = thread_segment[ITEM].value;
            }
        }

        __syncthreads();

        #pragma unroll 1
        for (int item = threadIdx.x; item < tile_num_rows; item += BLOCK_THREADS)
        {
            spmv_params.d_vector_y[tile_start_coord.x + item] = s_partials[item];
        }

        // Return the tile's running carry-out
        return tile_carry;
    }



    /**
     * Consume input tile
     */
    __device__ __forceinline__ void ConsumeTile(
        CoordinateT*    d_tile_coordinates,     ///< [in] Pointer to the temporary array of tile starting coordinates
        OffsetT*        d_tile_carry_rows,      ///< [out] Pointer to the temporary array carry-out dot product row-ids, one per block
        ValueT*         d_tile_carry_values,    ///< [out] Pointer to the temporary array carry-out dot product partial-sums, one per block
        int             num_merge_tiles)        ///< [in] Number of merge tiles
    {
        int tile_idx = (blockIdx.x * gridDim.y) + blockIdx.y;    // Current tile index

        if (tile_idx >= num_merge_tiles)
            return;

        CoordinateT tile_start_coord     = d_tile_coordinates[tile_idx + 0];
        CoordinateT tile_end_coord       = d_tile_coordinates[tile_idx + 1];

        // Consume multi-segment tile
        ItemOffsetPairT tile_carry = ConsumeTile(
            tile_idx,
            tile_start_coord,
            tile_end_coord,
            Int2Type<AgentSpmvPolicyT::DIRECT_LOAD_NONZEROS>());

        // Output the tile's carry-out
        if (threadIdx.x == 0)
        {
            d_tile_carry_rows[tile_idx]     = tile_start_coord.x + tile_carry.offset;
            d_tile_carry_values[tile_idx]   = tile_carry.value;
        }
    }


};




}               // CUB namespace
CUB_NS_POSTFIX  // Optional outer namespace(s)

