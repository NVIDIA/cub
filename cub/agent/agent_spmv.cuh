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
#include "../grid/grid_queue.cuh"
#include "../iterator/cache_modified_input_iterator.cuh"
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
        BLOCK_THREADS                                           = _BLOCK_THREADS,                   ///< Threads per thread block
        ITEMS_PER_THREAD                                        = _ITEMS_PER_THREAD,                ///< Items per thread (per tile of input)
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
    typename    AgentSpmvPolicyT,      ///< Parameterized AgentSpmvPolicy tuning policy type
    typename    VertexT,                    ///< Integer type for vertex identifiers
    typename    ValueT,                     ///< Matrix and vector value type
    typename    OffsetT,                    ///< Signed integer type for sequence offsets
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
        MatrixValueIteratorT;

    typedef CacheModifiedInputIterator<
            AgentSpmvPolicyT::MATRIX_COLUMN_INDICES_LOAD_MODIFIER,
            VertexT,
            OffsetT>
        MatrixRowOffsetsIteratorT;

    typedef CacheModifiedInputIterator<
            AgentSpmvPolicyT::MATRIX_VALUES_LOAD_MODIFIER,
            ValueT,
            OffsetT>
        MatrixColumnIndicesIteratorT;

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
            OffsetT row_offset;
            ValueT  partial_sum;

        } merge[TILE_ITEMS];


        OffsetT block_row_offsets_start_idx;
        OffsetT block_column_indices_start_idx;
    };


    /// Temporary storage type (unionable)
    struct TempStorage : Uninitialized<_TempStorage> {};


    //---------------------------------------------------------------------
    // Per-thread fields
    //---------------------------------------------------------------------

    /// Reference to temp_storage
    _TempStorage &temp_storage;

    MatrixValueIteratorT            d_matrix_values;            ///< Pointer to the array of \p num_nonzeros values of the corresponding nonzero elements of matrix <b>A</b>.
    MatrixRowOffsetsIteratorT       d_matrix_row_offsets;       ///< Pointer to the array of \p m + 1 offsets demarcating the start of every row in \p d_matrix_column_indices and \p d_matrix_values (with the final entry being equal to \p num_nonzeros)
    MatrixColumnIndicesIteratorT    d_matrix_column_indices;    ///< Pointer to the array of \p num_nonzeros column-indices of the corresponding nonzero elements of matrix <b>A</b>.  (Indices are zero-valued.)
    VectorValueIteratorT            d_vector_x;                 ///< Pointer to the array of \p num_cols values corresponding to the dense input vector <em>x</em>
    ValueT*                         d_vector_y;                 ///< Pointer to the array of \p num_rows values corresponding to the dense output vector <em>y</em>
    VertexT*                        d_block_carryout_rows;      ///< Pointer to the temporary array carry-out dot product row-ids, one per block
    ValueT*                         d_block_runout_values;      ///< Pointer to the temporary array carry-out dot product partial-sums, one per block
    int                             num_rows;                   ///< The number of rows of matrix <b>A</b>.
    int                             num_cols;                   ///< The number of columns of matrix <b>A</b>.
    int                             num_nonzeros;               ///< The number of nonzero elements of matrix <b>A</b>.


    //---------------------------------------------------------------------
    // Interface
    //---------------------------------------------------------------------

    /**
     * Constructor
     */
    __device__ __forceinline__ AgentSpmv(
        TempStorage &temp_storage,              ///< Reference to temp_storage
        ValueT*     d_matrix_values,            ///< [in] Pointer to the array of \p num_nonzeros values of the corresponding nonzero elements of matrix <b>A</b>.
        OffsetT*    d_matrix_row_offsets,       ///< [in] Pointer to the array of \p m + 1 offsets demarcating the start of every row in \p d_matrix_column_indices and \p d_matrix_values (with the final entry being equal to \p num_nonzeros)
        VertexT*    d_matrix_column_indices,    ///< [in] Pointer to the array of \p num_nonzeros column-indices of the corresponding nonzero elements of matrix <b>A</b>.  (Indices are zero-valued.)
        ValueT*     d_vector_x,                 ///< [in] Pointer to the array of \p num_cols values corresponding to the dense input vector <em>x</em>
        ValueT*     d_vector_y,                 ///< [out] Pointer to the array of \p num_rows values corresponding to the dense output vector <em>y</em>
        VertexT*    d_block_carryout_rows,      ///< [out] Pointer to the temporary array carry-out dot product row-ids, one per block
        ValueT*     d_block_runout_values,      ///< [out] Pointer to the temporary array carry-out dot product partial-sums, one per block
        int         num_rows,                   ///< [in] number of rows of matrix <b>A</b>.
        int         num_cols,                   ///< [in] number of columns of matrix <b>A</b>.
        int         num_nonzeros)               ///< [in] number of nonzero elements of matrix <b>A</b>.
    :
        temp_storage(temp_storage.Alias()),
        d_wrapped_samples(d_samples),
        num_output_bins(num_output_bins),
        num_privatized_bins(num_privatized_bins),
        d_output_histograms(d_output_histograms),
        privatized_decode_op(privatized_decode_op),
        output_decode_op(output_decode_op),
        d_native_samples(NativePointer(d_wrapped_samples)),
        prefer_smem((MEM_PREFERENCE == SMEM) ?
            true :                              // prefer smem privatized histograms
            (MEM_PREFERENCE == GMEM) ?
                false :                         // prefer gmem privatized histograms
                blockIdx.x & 1)                 // prefer blended privatized histograms
    {
        int blockId = (blockIdx.y * gridDim.x) + blockIdx.x;

        for (int CHANNEL = 0; CHANNEL < NUM_ACTIVE_CHANNELS; ++CHANNEL)
            this->d_privatized_histograms[CHANNEL] = d_privatized_histograms[CHANNEL] + (blockId * num_privatized_bins[CHANNEL]);
    }


    /**
     * Consume image
     */
    __device__ __forceinline__ void ConsumeTiles(
        OffsetT             num_row_pixels,             ///< The number of multi-channel pixels per row in the region of interest
        OffsetT             num_rows,                   ///< The number of rows in the region of interest
        OffsetT             row_stride_samples,         ///< The number of samples between starts of consecutive rows in the region of interest
        int                 tiles_per_row,              ///< Number of image tiles per row
        GridQueue<int>      tile_queue)                 ///< Queue descriptor for assigning tiles of work to thread blocks
    {
/*

        // Whether all row starting offsets are quad-aligned (in single-channel)
        // Whether all row starting offsets are pixel-aligned (in multi-channel)
        int     offsets             = int(d_native_samples) | (int(row_stride_samples) * int(sizeof(SampleT)));
        int     quad_mask           = sizeof(SampleT) * 4 - 1;
        int     pixel_mask          = AlignBytes<PixelT>::ALIGN_BYTES - 1;

        bool    quad_aligned_rows   = (NUM_CHANNELS == 1) && ((offsets & quad_mask) == 0);
        bool    pixel_aligned_rows  = (NUM_CHANNELS > 1) && ((offsets & pixel_mask) == 0);

        // Whether rows are aligned and can be vectorized
        if ((d_native_samples != NULL) && (quad_aligned_rows || pixel_aligned_rows))
            ConsumeRows<true>(num_row_pixels, num_rows, row_stride_samples, d_native_samples);
        else
            ConsumeRows<false>(num_row_pixels, num_rows, row_stride_samples, d_native_samples);
*/

//        ConsumeRows<true>(num_row_pixels, num_rows, row_stride_samples);

        ConsumeTiles<true>(
            0,
            num_row_pixels * NUM_CHANNELS,
            tiles_per_row,
            tile_queue,
            Int2Type<WORK_STEALING>());
    }


    /**
     * Initialize privatized bin counters.  Specialized for privatized shared-memory counters
     */
    __device__ __forceinline__ void InitBinCounters()
    {
        if (prefer_smem)
            InitSmemBinCounters();
        else
            InitGmemBinCounters();
    }


    /**
     * Store privatized histogram to global memory.  Specialized for privatized shared-memory counters
     */
    __device__ __forceinline__ void StoreOutput()
    {
        if (prefer_smem)
            StoreSmemOutput();
        else
            StoreGmemOutput();
    }


};




}               // CUB namespace
CUB_NS_POSTFIX  // Optional outer namespace(s)

