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
#include "../block/block_scan.cuh"
#include "../thread/thread_search.cuh"
#include "../thread/thread_operators.cuh"
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
    CacheLoadModifier               _VECTOR_VALUES_LOAD_MODIFIER,           ///< Cache load modifier for reading vector values
    bool                            _UNROLL_NONZERO_LOADS,                  ///< Whether to fully unroll the loading of the non-zero merge tile segment
    BlockScanAlgorithm              _SCAN_ALGORITHM>                        ///< The BlockScan algorithm to use
struct AgentSpmvPolicy
{
    enum
    {
        BLOCK_THREADS                                                   = _BLOCK_THREADS,                       ///< Threads per thread block
        ITEMS_PER_THREAD                                                = _ITEMS_PER_THREAD,                    ///< Items per thread (per tile of input)
        UNROLL_NONZERO_LOADS                                            = _UNROLL_NONZERO_LOADS,                ///< Whether to fully unroll the loading of the non-zero merge tile segment
    };

    static const CacheLoadModifier  MATRIX_ROW_OFFSETS_LOAD_MODIFIER    = _MATRIX_ROW_OFFSETS_LOAD_MODIFIER;    ///< Cache load modifier for reading CSR row-offsets
    static const CacheLoadModifier  MATRIX_COLUMN_INDICES_LOAD_MODIFIER = _MATRIX_COLUMN_INDICES_LOAD_MODIFIER; ///< Cache load modifier for reading CSR column-indices
    static const CacheLoadModifier  MATRIX_VALUES_LOAD_MODIFIER         = _MATRIX_VALUES_LOAD_MODIFIER;         ///< Cache load modifier for reading CSR values
    static const CacheLoadModifier  VECTOR_VALUES_LOAD_MODIFIER         = _VECTOR_VALUES_LOAD_MODIFIER;         ///< Cache load modifier for reading vector values
    static const BlockScanAlgorithm SCAN_ALGORITHM                      = _SCAN_ALGORITHM;                      ///< The BlockScan algorithm to use

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

    // BlockScan specialization
    typedef BlockScan<
            ItemOffsetPair<ValueT, OffsetT>,
            BLOCK_THREADS,
            AgentSpmvPolicyT::SCAN_ALGORITHM>
        BlockScanT;

    /// Merge item type (either a non-zero value or a row-end offset)
    union MergeItem
    {
        OffsetT    row_end_offset;
        ValueT     nonzero;
    };

    /// Shared memory type required by this thread block
    struct _TempStorage
    {
        // Starting and ending global merge path coordinates for the tile
        CoordinateT tile_coordinates[2];

        // Tile of merge items
        MergeItem merge_items[TILE_ITEMS + ITEMS_PER_THREAD];

        union
        {
            typename BlockReduceT::TempStorage reduce;
            typename BlockScanT::TempStorage scan;
        };
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
    OffsetT*                        d_tile_carry_rows;          ///< Pointer to the temporary array carry-out dot product row-ids, one per block
    ValueT*                         d_tile_carry_values;        ///< Pointer to the temporary array carry-out dot product partial-sums, one per block
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
        OffsetT*    d_tile_carry_rows,          ///< [out] Pointer to the temporary array carry-out dot product row-ids, one per block
        ValueT*     d_tile_carry_values,        ///< [out] Pointer to the temporary array carry-out dot product partial-sums, one per block
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
        d_tile_carry_rows(d_tile_carry_rows),
        d_tile_carry_values(d_tile_carry_values),
        num_rows(num_rows),
        num_cols(num_cols),
        num_nonzeros(num_nonzeros)
    {}


    /**
     * Consume a merge tile (when comprised entirely of non-zeros)
     */
    __device__ __forceinline__ ItemOffsetPair<ValueT, OffsetT> ConsumeTileNonZeros(
        int tile_idx,
        CoordinateT tile_start_coord,
        CoordinateT tile_end_coord)
    {
        ItemOffsetPair<ValueT, OffsetT>     tile_carry;
        ValueT                              non_zeros[ITEMS_PER_THREAD];

        // Gather the nonzeros for the merge tile
        #pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
        {
            int item = (ITEM * BLOCK_THREADS) + threadIdx.x;

            OffsetT column_index        = d_matrix_column_indices[tile_start_coord.y + item];
            ValueT  matrix_value        = d_matrix_values[tile_start_coord.y + item];
            ValueT  vector_value        = d_vector_x[column_index];
            non_zeros[ITEM]             = matrix_value * vector_value;
        }

        __syncthreads();    // Perf-sync

        ValueT tile_sum = BlockReduceT(temp_storage.reduce).Sum(non_zeros);

        tile_carry.offset = 0;
        tile_carry.value = tile_sum;

        // Return the tile's running carry-out
        return tile_carry;
    }


    /**
     * Consume a merge tile
     */
    __device__ __forceinline__ ItemOffsetPair<ValueT, OffsetT> ConsumeTile(
        int tile_idx,
        CoordinateT tile_start_coord,
        CoordinateT tile_end_coord)
    {
        int                             tile_num_rows           = tile_end_coord.x - tile_start_coord.x;
        int                             tile_num_nonzeros       = tile_end_coord.y - tile_start_coord.y;
        OffsetT*                        tile_row_end_offsets    = reinterpret_cast<OffsetT*>(temp_storage.merge_items);
        ValueT*                         tile_nonzeros           = reinterpret_cast<ValueT*>(temp_storage.merge_items + tile_num_rows);
        ValueT*                         tile_partial_sums       = reinterpret_cast<ValueT*>(temp_storage.merge_items);

        // Gather the row end-offsets for the merge tile into shared memory
        for (int item = threadIdx.x; item < tile_num_rows; item += BLOCK_THREADS)
        {
            tile_row_end_offsets[item] = d_matrix_row_end_offsets[tile_start_coord.x + item];
        }

        __syncthreads();              // Perf-sync

        // Gather the nonzeros for the merge tile into shared memory
        if (AgentSpmvPolicyT::UNROLL_NONZERO_LOADS)
        {
            #pragma unroll
            for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
            {
                int item = threadIdx.x + (ITEM * BLOCK_THREADS);

                if (item < tile_num_nonzeros)
                {
                    OffsetT column_index        = d_matrix_column_indices[tile_start_coord.y + item];
                    ValueT  matrix_value        = d_matrix_values[tile_start_coord.y + item];
                    ValueT  vector_value        = d_vector_x[column_index];

                    tile_nonzeros[item]         = matrix_value * vector_value;
                }
            }
        }
        else
        {
            for (int item = threadIdx.x; item < tile_num_nonzeros; item += BLOCK_THREADS)
            {
                OffsetT column_index        = d_matrix_column_indices[tile_start_coord.y + item];
                ValueT  matrix_value        = d_matrix_values[tile_start_coord.y + item];
                ValueT  vector_value        = d_vector_x[column_index];

                tile_nonzeros[item]         = matrix_value * vector_value;
            }
        }

        __syncthreads();   

        // Search for the thread's starting coordinate within the merge tile
        CountingInputIterator<OffsetT>  tile_nonzero_indices(tile_start_coord.y);
        CoordinateT                     thread_start_coord;

        MergePathSearch(
            OffsetT(threadIdx.x * ITEMS_PER_THREAD),            // Diagonal
            tile_row_end_offsets,                               // List A
            tile_nonzero_indices,                               // List B
            tile_num_rows,
            tile_num_nonzeros,
            thread_start_coord);

        __syncthreads();    // Perf-sync

        // Compute the thread's merge path segment
        ItemOffsetPair<ValueT, OffsetT> thread_segment[ITEMS_PER_THREAD];
        CoordinateT                     thread_current_coord = thread_start_coord;

        #pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
        {
            bool accumulate = (tile_nonzero_indices[thread_current_coord.y] < tile_row_end_offsets[thread_current_coord.x]);

            if (accumulate)
            {
                // Move down (accumulate)
                thread_segment[ITEM].offset = 0;
                thread_segment[ITEM].value  = tile_nonzeros[thread_current_coord.y];
                ++thread_current_coord.y;
            }
            else
            {
                // Move right (reset)
                thread_segment[ITEM].offset = 1;
                thread_segment[ITEM].value  = 0.0;
                ++thread_current_coord.x;
            }
        }

        __syncthreads();

        // Compact the accumulated value for each row-end into shared memory
        OffsetT running_offset = 0;
        ValueT  running_total = 0.0;

        #pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
        {
            running_total += thread_segment[ITEM].value;

            if (thread_segment[ITEM].offset)
            {
                tile_partial_sums[thread_start_coord.x + running_offset] = running_total;
                running_total = 0.0;
            }

            running_offset += thread_segment[ITEM].offset;
        }

        // Compute the inter-thread fix-up from thread-carries (for when a row spans more than one thread-segment)
        int num_thread_rows = thread_current_coord.x - thread_start_coord.x;

        ItemOffsetPair<ValueT, OffsetT>                                 tile_carry;
        ItemOffsetPair<ValueT, OffsetT>                                 scan_identity(0.0, 0);
        ItemOffsetPair<ValueT, OffsetT>                                 thread_carry(running_total, num_thread_rows);
        ReduceBySegmentOp<cub::Sum, ItemOffsetPair<ValueT, OffsetT> >   scan_op;

        // Block-wide reduce-value-by-key
        BlockScanT(temp_storage.scan).ExclusiveScan(thread_carry, thread_carry, scan_identity, scan_op, tile_carry);

        // Flush the fix-up if this thread encountered a row-transition
        if (num_thread_rows > 0)
            tile_partial_sums[thread_carry.offset] += thread_carry.value;

        __syncthreads();

        // Scatter compacted partial sums
        for (int item = threadIdx.x; item < tile_num_rows; item += BLOCK_THREADS)
        {
            d_vector_y[tile_start_coord.x + item] = tile_partial_sums[item];
        }

        // Return the tile's running carry-out
        return tile_carry;
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

        CoordinateT tile_start_coord     = temp_storage.tile_coordinates[0];
        CoordinateT tile_end_coord       = temp_storage.tile_coordinates[1];

        // Consume a merge tile
        ItemOffsetPair<ValueT, OffsetT> tile_carry;

        if (tile_start_coord.x < tile_end_coord.x)
        {
            tile_carry = ConsumeTile(tile_idx, tile_start_coord, tile_end_coord);
        }
        else
        {
            // Fast-path when the merge segment is comprised of all non-zeros
            tile_carry = ConsumeTileNonZeros(tile_idx, tile_start_coord, tile_end_coord);
        }

        // Output the tile's merge-path carry
        if (threadIdx.x == 0)
        {
            d_tile_carry_rows[tile_idx] = tile_carry.offset;
            d_tile_carry_values[tile_idx] = tile_carry.value;
        }
    }


};




}               // CUB namespace
CUB_NS_POSTFIX  // Optional outer namespace(s)

