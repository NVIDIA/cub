/******************************************************************************
 * Copyright (c) 2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2014, NVIDIA CORPORATION.  All rights reserved.
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
#include "../thread/thread_search.cuh"
#include "../grid/grid_queue.cuh"
#include "../iterator/cache_modified_input_iterator.cuh"
#include "../iterator/counting_input_iterator.cuh"
#include "../iterator/transform_input_iterator.cuh"
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
    CacheLoadModifier               _MATRIX_ROW_OFFSETS_LOAD_MODIFIER,      ///< Cache load modifier for reading CSR row-offsets
    CacheLoadModifier               _MATRIX_COLUMN_INDICES_LOAD_MODIFIER,   ///< Cache load modifier for reading CSR column-indices
    CacheLoadModifier               _MATRIX_VALUES_LOAD_MODIFIER,           ///< Cache load modifier for reading CSR values
    CacheLoadModifier               _VECTOR_VALUES_LOAD_MODIFIER>           ///< Cache load modifier for reading vector values
struct AgentSpmvPolicy
{
    enum
    {
        BLOCK_THREADS                                                   = _BLOCK_THREADS,                           ///< Threads per thread block
        ITEMS_PER_THREAD                                                = _ITEMS_PER_THREAD,                        ///< Items per thread (per tile of input)
    };

    static const CacheLoadModifier MATRIX_ROW_OFFSETS_LOAD_MODIFIER    = _MATRIX_ROW_OFFSETS_LOAD_MODIFIER;       ///< Cache load modifier for reading CSR row-offsets
    static const CacheLoadModifier MATRIX_COLUMN_INDICES_LOAD_MODIFIER = _MATRIX_COLUMN_INDICES_LOAD_MODIFIER;    ///< Cache load modifier for reading CSR column-indices
    static const CacheLoadModifier MATRIX_VALUES_LOAD_MODIFIER         = _MATRIX_VALUES_LOAD_MODIFIER;            ///< Cache load modifier for reading CSR values
    static const CacheLoadModifier VECTOR_VALUES_LOAD_MODIFIER         = _VECTOR_VALUES_LOAD_MODIFIER;            ///< Cache load modifier for reading vector values
};


/******************************************************************************
 * Thread block abstractions
 ******************************************************************************/

/**
 * \brief AgentSpmv implements a stateful abstraction of CUDA thread blocks for participating in device-wide SpMV.
 */
template <
    typename    AgentSpmvPolicyT,           ///< Parameterized AgentSpmvPolicy tuning policy type
    typename    ValueT,                     ///< Matrix and vector value type
    typename    OffsetT,                    ///< Signed integer type for sequence offsets
    typename    CoordinateT,                ///< Merge path coordinate type
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

    /// Input iterator wrapper types (for applying cache modifiers)

    typedef CacheModifiedInputIterator<
            AgentSpmvPolicyT::MATRIX_ROW_OFFSETS_LOAD_MODIFIER,
            OffsetT,
            OffsetT>
        MatrixRowOffsetsIteratorT;

    typedef CacheModifiedInputIterator<
            AgentSpmvPolicyT::MATRIX_COLUMN_INDICES_LOAD_MODIFIER,
            OffsetT,
            OffsetT>
        MatrixColumnIndicesIteratorT;

    typedef CacheModifiedInputIterator<
            AgentSpmvPolicyT::MATRIX_VALUES_LOAD_MODIFIER,
            ValueT,
            OffsetT>
        MatrixValueIteratorT;

    typedef CacheModifiedInputIterator<
            AgentSpmvPolicyT::VECTOR_VALUES_LOAD_MODIFIER,
            ValueT,
            OffsetT>
        VectorValueIteratorT;

    // BlockReduce specialization
    typedef BlockReduce<
            ValueT,
            BLOCK_THREADS,
            BLOCK_REDUCE_WARP_REDUCTIONS>
        BlockReduceT;

    /// Shared memory type required by this thread block
    struct _TempStorage
    {
        // Tile of merge items
        union MergeItem
        {
            OffsetT    row_end_offset;
            ValueT     nonzero;

        } merge_items[TILE_ITEMS + 1];

        // Starting and ending global merge path coordinates for the tile
        CoordinateT tile_coordinates[2];
    };

    /// Temporary storage type (unionable)
    struct TempStorage : Uninitialized<_TempStorage> {};


    //---------------------------------------------------------------------
    // Per-thread fields
    //---------------------------------------------------------------------

    /// Reference to temp_storage
    _TempStorage &temp_storage;

    MatrixValueIteratorT            d_matrix_values;            ///< Pointer to the array of \p num_nonzeros values of the corresponding nonzero elements of matrix <b>A</b>.
    MatrixRowOffsetsIteratorT       d_matrix_row_end_offsets;   ///< Pointer to the array of \p m offsets demarcating the end of every row in \p d_matrix_column_indices and \p d_matrix_values
    MatrixColumnIndicesIteratorT    d_matrix_column_indices;    ///< Pointer to the array of \p num_nonzeros column-indices of the corresponding nonzero elements of matrix <b>A</b>.  (Indices are zero-valued.)
    VectorValueIteratorT            d_vector_x;                 ///< Pointer to the array of \p num_cols values corresponding to the dense input vector <em>x</em>
    ValueT*                         d_vector_y;                 ///< Pointer to the array of \p num_rows values corresponding to the dense output vector <em>y</em>
    OffsetT*                        d_block_carryout_rows;      ///< Pointer to the temporary array carry-out dot product row-ids, one per block
    ValueT*                         d_block_carryout_values;    ///< Pointer to the temporary array carry-out dot product partial-sums, one per block
    int                             num_rows;                   ///< The number of rows of matrix <b>A</b>.
    int                             num_cols;                   ///< The number of columns of matrix <b>A</b>.
    int                             num_nonzeros;               ///< The number of nonzero elements of matrix <b>A</b>.

    //---------------------------------------------------------------------
    // Utility methods
    //---------------------------------------------------------------------



    //---------------------------------------------------------------------
    // Interface
    //---------------------------------------------------------------------

    /**
     * Constructor
     */
    __device__ __forceinline__ AgentSpmv(
        TempStorage &temp_storage,              ///< Reference to temp_storage
        ValueT*     d_matrix_values,            ///< [in] Pointer to the array of \p num_nonzeros values of the corresponding nonzero elements of matrix <b>A</b>.
        OffsetT*    d_matrix_row_end_offsets,   ///< [in] Pointer to the array of \p m offsets demarcating the end of every row in \p d_matrix_column_indices and \p d_matrix_values
        OffsetT*    d_matrix_column_indices,    ///< [in] Pointer to the array of \p num_nonzeros column-indices of the corresponding nonzero elements of matrix <b>A</b>.  (Indices are zero-valued.)
        ValueT*     d_vector_x,                 ///< [in] Pointer to the array of \p num_cols values corresponding to the dense input vector <em>x</em>
        ValueT*     d_vector_y,                 ///< [out] Pointer to the array of \p num_rows values corresponding to the dense output vector <em>y</em>
        OffsetT*    d_block_carryout_rows,      ///< [out] Pointer to the temporary array carry-out dot product row-ids, one per block
        ValueT*     d_block_carryout_values,    ///< [out] Pointer to the temporary array carry-out dot product partial-sums, one per block
        int         num_rows,                   ///< [in] number of rows of matrix <b>A</b>.
        int         num_cols,                   ///< [in] number of columns of matrix <b>A</b>.
        int         num_nonzeros)               ///< [in] number of nonzero elements of matrix <b>A</b>.
    :
        temp_storage(temp_storage.Alias()),
        d_matrix_values(d_matrix_values),
        d_matrix_row_end_offsets(d_matrix_row_end_offsets),
        d_matrix_column_indices(d_matrix_column_indices),
        d_vector_x(d_vector_x),
        d_vector_y(d_vector_y),
        d_block_carryout_rows(d_block_carryout_rows),
        d_block_carryout_values(d_block_carryout_values),
        num_rows(num_rows),
        num_cols(num_cols),
        num_nonzeros(num_nonzeros)
    {}


    /**
     * Consume input tile
     */
    __device__ __forceinline__ void ConsumeTile(
        CoordinateT tile_start,
        CoordinateT tile_end)
    {
        int                             tile_num_rows           = tile_end.x - tile_start.x;
        int                             tile_num_nonzeros       = tile_end.y - tile_start.y;
        OffsetT*                        tile_row_end_offsets    = reinterpret_cast<OffsetT*>(temp_storage.merge_items);
        ValueT*                         tile_nonzeros           = reinterpret_cast<ValueT*>(temp_storage.merge_items + tile_num_rows);
        ValueT*                         tile_partial_sums       = reinterpret_cast<ValueT*>(temp_storage.merge_items);

        // Gather a merge tile of (A) row end-offsets and (B) nonzeros
        for (int item = threadIdx.x; item < tile_num_rows; item += BLOCK_THREADS)
        {
            tile_row_end_offsets[item] = d_matrix_row_end_offsets[tile_start.x + item];
        }

        __syncthreads();

        for (int item = threadIdx.x; item < tile_num_nonzeros; item += BLOCK_THREADS)
        {
            OffsetT column_index        = d_matrix_column_indices[tile_start.y + item];
            ValueT  matrix_value        = d_matrix_values[tile_start.y + item];
            ValueT  vector_value        = d_vector_x[column_index];

            tile_nonzeros[item]         = matrix_value * vector_value;
        }

        __syncthreads();

        // Search for the thread's starting coordinate within the merge tile
        CountingInputIterator<OffsetT>  tile_nonzero_indices(tile_start.y);
        CoordinateT                     thread_coordinate;
     
        MergePathSearch(
            OffsetT(threadIdx.x * ITEMS_PER_THREAD),            // Diagonal
            tile_row_end_offsets,                               // List A
            tile_nonzero_indices,                               // List B
            tile_num_rows,
            tile_num_nonzeros,
            thread_coordinate);

        __syncthreads();

        // Run merge and reduce value by key

        OffsetT     row_ids[ITEMS_PER_THREAD];
        ValueT      non_zeros[ITEMS_PER_THREAD];
    
        ValueT running_sum = 0.0;

        #pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
        {
            row_ids[ITEM]   = -1;
            non_zeros[ITEM] = running_sum;
// Mooch
//            if (thread_coordinate.x < tile_num_rows)
            {
                if ((tile_nonzero_indices[thread_coordinate.y] < tile_row_end_offsets[thread_coordinate.x]))
                {
                    // Move down (accumulate)
                    running_sum += tile_nonzeros[thread_coordinate.y];
                    ++thread_coordinate.y;
                }
                else
                {
                    // Move right
                    row_ids[ITEM] = thread_coordinate.x;
                    ++thread_coordinate.x;
                    running_sum = 0.0;
                }
            }
        }

        __syncthreads();

        // Local scatter
        #pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
        {
            if (row_ids[ITEM] >= 0)
            {
                tile_partial_sums[row_ids[ITEM]] = non_zeros[ITEM];
            }
        }

        __syncthreads();

        // Update with thread_carry_out values
        if (thread_coordinate.x < tile_num_rows)
            atomicAdd(tile_partial_sums + thread_coordinate.x, running_sum);

        __syncthreads();

        // Output row partial sums
        for (int item = threadIdx.x; item < tile_num_rows; item += BLOCK_THREADS)
        {
            d_vector_y[tile_start.x + item] = tile_partial_sums[item];
        }
    }



    /**
     * Consume input tile
     */
    __device__ __forceinline__ void ConsumeTile(
        CoordinateT* d_tile_coordinates)
    {
        int tile_idx = blockIdx.x;

        // Find the starting and ending coordinates for this block's merge path segment
        if (threadIdx.x < 2)
        {
            temp_storage.tile_coordinates[threadIdx.x] = d_tile_coordinates[tile_idx + threadIdx.x];
        }

        __syncthreads();

        CoordinateT tile_start     = temp_storage.tile_coordinates[0];
        CoordinateT tile_end       = temp_storage.tile_coordinates[1];

        ConsumeTile(tile_start, tile_end);
    }



};




}               // CUB namespace
CUB_NS_POSTFIX  // Optional outer namespace(s)

