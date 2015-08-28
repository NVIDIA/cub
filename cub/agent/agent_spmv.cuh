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
#include "../block/block_exchange.cuh"
#include "../thread/thread_search.cuh"
#include "../thread/thread_operators.cuh"
#include "../iterator/cache_modified_input_iterator.cuh"
#include "../iterator/counting_input_iterator.cuh"
#include "../iterator/tex_ref_input_iterator.cuh"
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
    CacheLoadModifier               _ROW_OFFSETS_SEARCH_LOAD_MODIFIER,      ///< Cache load modifier for reading CSR row-offsets during search
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

    static const CacheLoadModifier  ROW_OFFSETS_SEARCH_LOAD_MODIFIER    = _ROW_OFFSETS_SEARCH_LOAD_MODIFIER;    ///< Cache load modifier for reading CSR row-offsets
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

    TexRefInputIterator<ValueT, 66778899, OffsetT>  t_vector_x;
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
            AgentSpmvPolicyT::ROW_OFFSETS_SEARCH_LOAD_MODIFIER,
            OffsetT,
            OffsetT>
        RowOffsetsSearchIteratorT;

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
    typedef KeyValuePair<OffsetT, ValueT> KeyValuePairT;

    // Reduce-value-by-segment scan operator
    typedef ReduceBySegmentOp<cub::Sum> ReduceBySegmentOpT;

    // Prefix functor type
    typedef BlockScanRunningPrefixOp<KeyValuePairT, ReduceBySegmentOpT> PrefixOpT;

    // BlockReduce specialization
    typedef BlockReduce<
            ValueT,
            BLOCK_THREADS,
            BLOCK_REDUCE_WARP_REDUCTIONS>
        BlockReduceT;

    // BlockScan specialization
    typedef BlockScan<
            KeyValuePairT,
            BLOCK_THREADS,
            AgentSpmvPolicyT::SCAN_ALGORITHM>
        BlockScanT;
    /// Shared memory type required by this thread block
    struct _TempStorage
    {
            ValueT running_total;

//        union
//        {
            // Smem needed for tile of merge items
            ValueT values[TILE_ITEMS + 1];

/*
            // Smem needed for block exchange
            typename BlockExchangeT::TempStorage exchange;

            // Smem needed for block-wide reduction
            typename BlockReduceT::TempStorage reduce;
*/
            // Smem needed for tile scanning
            typename BlockScanT::TempStorage scan;

//        };
    };

    /// Temporary storage type (unionable)
    struct TempStorage : Uninitialized<_TempStorage> {};


    //---------------------------------------------------------------------
    // Per-thread fields
    //---------------------------------------------------------------------


    _TempStorage&                   temp_storage;         /// Reference to temp_storage

    SpmvParams<ValueT, OffsetT>&    spmv_params;

    ValueIteratorT                  wd_values;            ///< Wrapped pointer to the array of \p num_nonzeros values of the corresponding nonzero elements of matrix <b>A</b>.
    RowOffsetsIteratorT             wd_row_end_offsets;   ///< Wrapped Pointer to the array of \p m offsets demarcating the end of every row in \p d_column_indices and \p d_values
    ColumnIndicesIteratorT          wd_column_indices;    ///< Wrapped Pointer to the array of \p num_nonzeros column-indices of the corresponding nonzero elements of matrix <b>A</b>.  (Indices are zero-valued.)
    VectorValueIteratorT            wd_vector_x;          ///< Wrapped Pointer to the array of \p num_cols values corresponding to the dense input vector <em>x</em>
    VectorValueIteratorT            wd_vector_y;          ///< Wrapped Pointer to the array of \p num_cols values corresponding to the dense input vector <em>x</em>


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



    __device__ __forceinline__ void InitNan(double& nan_token)
    {
        long long NAN_BITS  = 0xFFF0000000000001;
        nan_token           = reinterpret_cast<ValueT&>(NAN_BITS); // ValueT(0.0) / ValueT(0.0);
    }


    __device__ __forceinline__ void InitNan(float& nan_token)
    {
        int NAN_BITS        = 0xFF800001;
        nan_token           = reinterpret_cast<ValueT&>(NAN_BITS); // ValueT(0.0) / ValueT(0.0);
    }


    /**
     * Consume input range
     */
    __device__ __forceinline__ void ConsumeRange(
        CoordinateT*    d_tile_coordinates,     ///< [in] Pointer to the temporary array of tile starting coordinates
        KeyValuePairT*  d_tile_carry_pairs,
        OffsetT         tiles_per_block,
        OffsetT         num_merge_tiles)
    {
        ValueT NAN_TOKEN;
        InitNan(NAN_TOKEN);

        if (threadIdx.x == 0)
            temp_storage.running_total = 0.0;

        KeyValuePairT       tile_prefix = {0, 0.0};
        ReduceBySegmentOpT  scan_op;
        PrefixOpT           prefix_op(tile_prefix, scan_op);

        // Read our starting coordinates
        int tile_idx_begin      = blockIdx.x * tiles_per_block;
        int tile_idx_end        = CUB_MIN(tile_idx_begin + tiles_per_block, num_merge_tiles);
        CoordinateT tile_coord  = d_tile_coordinates[tile_idx_begin];

        #pragma unroll 1
        for (int tile_idx = tile_idx_begin; tile_idx < tile_idx_end; ++tile_idx)
        {
            CoordinateT tile_coord_end      = d_tile_coordinates[tile_idx + 1];

            int         tile_num_rows       = tile_coord_end.x - tile_coord.x;
            int         tile_num_nonzeros   = tile_coord_end.y - tile_coord.y;

            ValueT *s_saved = temp_storage.values + TILE_ITEMS + 1 - tile_num_rows;
//            ValueT *s_saved = temp_storage.values + tile_num_nonzeros + 1;

/*
            if (threadIdx.x == 0)
                CubLog("\n\nTile %d tile_coord(%d,%d), tile_coord_end(%d,%d), tile_num_rows%d, tile_num_nonzeros %d\n\n",
                    tile_idx,
                    tile_coord.x, tile_coord.y,
                    tile_coord_end.x, tile_coord_end.y,
                    tile_num_rows, tile_num_nonzeros);
*/
            // Nonzeros tile

            ValueT nonzeros[ITEMS_PER_THREAD];

            #pragma unroll
            for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
            {
                OffsetT local_nonzero_idx   = (ITEM * BLOCK_THREADS) + threadIdx.x;
                OffsetT nonzero_idx         = tile_coord.y + local_nonzero_idx;
                nonzero_idx = CUB_MIN(nonzero_idx, tile_coord_end.y - 1);

                OffsetT column_idx          = wd_column_indices[nonzero_idx];
                ValueT  value               = wd_values[nonzero_idx];
                ValueT  vector_value        = wd_vector_x[column_idx];
                nonzeros[ITEM]              = value * vector_value;

                temp_storage.values[local_nonzero_idx] = nonzeros[ITEM];
            }

            __syncthreads();


            // Replace row-ends with NAN tokens
            #pragma unroll 1
            for (int row = threadIdx.x; row < tile_num_rows; row += BLOCK_THREADS)
            {
                OffsetT row_offset              = wd_row_end_offsets[tile_coord.x + row - 1];
                OffsetT row_end_offset          = wd_row_end_offsets[tile_coord.x + row];
                int     local_row_end_offset    = row_end_offset - tile_coord.y - 1;

                ValueT value = temp_storage.values[local_row_end_offset];
                temp_storage.values[local_row_end_offset] = NAN_TOKEN;

                if (local_row_end_offset < 0)
                {
                    prefix_op.running_total.key = 1;
                    prefix_op.running_total.value = 0.0;
                }

                s_saved[row] = (row_offset != row_end_offset) ? value : 0.0;

//                CubLog("\tlocal_row_end_offset %d: %f, save_offset %d: %f\n",
//                    local_row_end_offset, NAN_TOKEN, row, value);
            }

            __syncthreads();

            // Read nonzeros into thread-blocked order, setup segment flags
            KeyValuePairT scan_items[ITEMS_PER_THREAD];
            for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
            {
                int     local_nonzero_idx   = (threadIdx.x * ITEMS_PER_THREAD) + ITEM;
                ValueT  value               = temp_storage.values[local_nonzero_idx];
                bool    is_nan              = (value != value);

                scan_items[ITEM].key    = is_nan;
                scan_items[ITEM].value  = (is_nan || (local_nonzero_idx >= tile_num_nonzeros)) ?
                                            0.0 :
                                            value;
            }

            KeyValuePairT tile_aggregate;
            KeyValuePairT scan_items_out[ITEMS_PER_THREAD];

            BlockScanT(temp_storage.scan).ExclusiveScan(
                scan_items,
                scan_items_out,
                scan_op,
                tile_aggregate,
                prefix_op);

            // Reset running key
            if (threadIdx.x == 0)
            {
                prefix_op.running_total.key = 0;
                temp_storage.running_total = prefix_op.running_total.value;
            }

            // Compact segment totals
            for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
            {
                if (scan_items[ITEM].key)
                {
                    s_saved[scan_items_out[ITEM].key] += scan_items_out[ITEM].value;
                }
            }

            __syncthreads();

            // Store row totals
            #pragma unroll 1
            for (int row = threadIdx.x; row < tile_num_rows; row += BLOCK_THREADS)
            {
//                CubLog("Storing row %d: %f\n", tile_coord.x + row, value);
                spmv_params.d_vector_y[tile_coord.x + row] = s_saved[row];
            }

            __syncthreads();

            tile_coord = tile_coord_end;
        }
    }


};




}               // CUB namespace
CUB_NS_POSTFIX  // Optional outer namespace(s)

