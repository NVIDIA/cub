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
#include "../block/block_load.cuh"
#include "../thread/thread_search.cuh"
#include "../grid/grid_queue.cuh"
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


    /// Shared memory type required by this thread block
    struct _TempStorage
    {
        union
        {
            OffsetT    row_end_offset;
            ValueT     partial_sum;

        } merge[TILE_ITEMS];

        // Starting and ending merge path coordinates
        CoordinateT agent_coordinates[2];
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
        CoordinateT agent_start,
        CoordinateT agent_end)
    {
        int                             num_row_items           = agent_end.x - agent_start.x;
        int                             num_nonzero_items       = agent_end.y - agent_start.y;
        OffsetT*                        local_row_end_offsets   = reinterpret_cast<OffsetT*>(temp_storage.merge + 0);
        ValueT*                         local_partial_sums      = reinterpret_cast<ValueT*>(temp_storage.merge + num_row_items);
        ValueT*                         output_partial_sums     = reinterpret_cast<ValueT*>(temp_storage.merge + 0);
        CountingInputIterator<OffsetT>  local_nonzero_indices(agent_start.y);

        // Gather partial sums
        #pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
        {
            int item = (ITEM * BLOCK_THREADS) + threadIdx.x;
            if (item < num_row_items)
            {
                local_row_end_offsets[item] = d_matrix_row_end_offsets[agent_start.x + item];
            }
            if (item < num_nonzero_items)
            {
                OffsetT column_index        = d_matrix_column_indices[agent_start.y + item];
                ValueT  matrix_value        = d_matrix_values[agent_start.y + item];
                ValueT  vector_value        = d_vector_x[column_index];

                local_partial_sums[item]    = matrix_value * vector_value;
            }
        }

        __syncthreads();

        // Search for the thread's local starting coordinate
        CoordinateT local_coordinate;
        MergePathSearch(
            OffsetT(threadIdx.x * ITEMS_PER_THREAD),            // Diagonal
            local_row_end_offsets,                              // List A
            local_nonzero_indices,                              // List B
            num_row_items,
            num_nonzero_items,
            local_coordinate);

        __syncthreads();

        // Run SPMV merge
        ValueT partial_sum = 0.0;

        #pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
        {
            if (local_nonzero_indices[local_coordinate.y] < local_row_end_offsets[local_coordinate.x])
            {
                // Move down (accumulate)
                partial_sum += local_partial_sums[local_coordinate.y];
                ++local_coordinate.y;
            }
            else
            {
                // Move right (flush) TODO: this will clobber for sizeof(OffsetT) != sizeof(ValueT)!!!!!!
                output_partial_sums[local_coordinate.x] = partial_sum;
                partial_sum = 0.0;
                ++local_coordinate.x;
            }
        }

        __syncthreads();

        // Output row partial sums
        for (int item = threadIdx.x; item < num_row_items; item += BLOCK_THREADS)
        {
            d_vector_y[agent_start.x + item] = output_partial_sums[item];
        }
    }



    /**
     * Consume input tile
     */
    __device__ __forceinline__ void ConsumeTile(
        CoordinateT* d_agent_coordinates)
    {
        int agent_idx = blockIdx.x;

        // Find the starting and ending coordinates for this block's merge path segment
        if (threadIdx.x < 2)
        {
            temp_storage.agent_coordinates[threadIdx.x] = d_agent_coordinates[agent_idx + threadIdx.x];
            temp_storage.agent_coordinates[threadIdx.x] = d_agent_coordinates[agent_idx + threadIdx.x];
        }

        __syncthreads();

        CoordinateT agent_start     = temp_storage.agent_coordinates[0];
        CoordinateT agent_end       = temp_storage.agent_coordinates[1];

        ConsumeTile(agent_start, agent_end);
    }



};




}               // CUB namespace
CUB_NS_POSTFIX  // Optional outer namespace(s)

