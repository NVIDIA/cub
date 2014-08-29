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
 * cub::BlockHistogramSweep implements a stateful abstraction of CUDA thread blocks for participating in device-wide histogram across a range of tiles.
 */

#pragma once

#include <iterator>

#include "specializations/block_histogram_satomic_sweep.cuh"
#include "specializations/block_histogram_sort_sweep.cuh"
#include "../util_type.cuh"
#include "../grid/grid_mapping.cuh"
#include "../grid/grid_even_share.cuh"
#include "../grid/grid_queue.cuh"
#include "../util_namespace.cuh"

/// Optional outer namespace(s)
CUB_NS_PREFIX

/// CUB namespace
namespace cub {


/******************************************************************************
 * Tuning policy
 ******************************************************************************/

/**
 * Parameterizable tuning policy type for BlockHistogramSweep
 */
template <
    int _BLOCK_THREADS,                             ///< Threads per thread block
    int _PIXELS_PER_THREAD>                         ///< Pixels per thread (per tile of input)
struct BlockHistogramSweepPolicy
{
    enum
    {
        BLOCK_THREADS       = _BLOCK_THREADS,       ///< Threads per thread block
        PIXELS_PER_THREAD   = _PIXELS_PER_THREAD,   ///< Pixels per thread (per tile of input)
    };
};



/******************************************************************************
 * Thread block abstractions
 ******************************************************************************/

/**
 * \brief BlockHistogramSweep implements a stateful abstraction of CUDA thread blocks for participating in device-wide histogram across a range of tiles.
 */
template <
    typename    BlockHistogramSweepPolicy,      ///< Parameterized BlockHistogramSweepPolicy tuning policy type
    int         MAX_PRIVATIZED_BINS,            ///< Number of histogram bins per channel (e.g., up to 256)
    int         NUM_CHANNELS,                   ///< Number of channels interleaved in the input data (may be greater than the number of active channels being histogrammed)
    int         NUM_ACTIVE_CHANNELS,            ///< Number of channels actively being histogrammed
    typename    InputIteratorT,                 ///< Random-access input iterator type for reading samples.
    typename    CounterT,                       ///< Integer type for counting sample occurrences per histogram bin
    typename    SampleTransformOpT,             ///< Transform operator type for determining bin-ids from samples for each channel
    typename    OffsetT,                        ///< Signed integer type for global offsets
    int         PTX_ARCH = CUB_PTX_ARCH>        ///< PTX version
struct BlockHistogramSweep
{
    //---------------------------------------------------------------------
    // Types and constants
    //---------------------------------------------------------------------

    // Alternative internal implementation types
    typedef BlockHistogramSweepSort<            BlockHistogramSweepPolicy, MAX_PRIVATIZED_BINS, NUM_CHANNELS, NUM_ACTIVE_CHANNELS, InputIteratorT, CounterT, SampleTransformOpT, OffsetT>  BlockHistogramSweepSortT;
    typedef BlockHistogramSweepSharedAtomic<    BlockHistogramSweepPolicy, MAX_PRIVATIZED_BINS, NUM_CHANNELS, NUM_ACTIVE_CHANNELS, InputIteratorT, CounterT, SampleTransformOpT, OffsetT>  BlockHistogramSweepSharedAtomicT;

    // Internal block sweep histogram type
    typedef typename If<(PTX_ARCH < 200),
        BlockHistogramSweepSortT,                                           // Use sort strategy for SM1x because it doesn't support (very well) shared atomics
        BlockHistogramSweepSharedAtomicT>::Type InternalBlockDelegate;

    enum
    {
        TILE_PIXELS     = InternalBlockDelegate::TILE_PIXELS,
        TILE_SAMPLES    = TILE_PIXELS * NUM_CHANNELS,
    };


    // Temporary storage type
    typedef typename InternalBlockDelegate::TempStorage TempStorage;

    //---------------------------------------------------------------------
    // Per-thread fields
    //---------------------------------------------------------------------

    // Internal block delegate
    InternalBlockDelegate internal_delegate;


    //---------------------------------------------------------------------
    // Interface
    //---------------------------------------------------------------------

    /**
     * Constructor
     */
    __device__ __forceinline__ BlockHistogramSweep(
        TempStorage         &temp_storage,                                  ///< Reference to temp_storage
        InputIteratorT    	d_in,                                           ///< Input data to reduce
        CounterT*           (&d_out_histograms)[NUM_ACTIVE_CHANNELS],       ///< Reference to output histograms
        SampleTransformOpT  (&transform_op)[NUM_ACTIVE_CHANNELS])           ///< Transform operators for determining bin-ids from samples, one for each channel
    :
        internal_delegate(temp_storage, d_in, d_out_histograms, transform_op)
    {}


    /**
     * \brief Consume striped tiles
     */
    __device__ __forceinline__ void ConsumeStriped(
        OffsetT  block_offset,                       ///< [in] Threadblock begin offset (inclusive)
        OffsetT  block_end)                          ///< [in] Threadblock end offset (exclusive)
    {
        // Consume subsequent full tiles of input
        while (block_offset + TILE_SAMPLES <= block_end)
        {
            internal_delegate.ConsumeTile<true>(block_offset);
            block_offset += TILE_SAMPLES;
        }

        // Consume a partially-full tile
        if (block_offset < block_end)
        {
            int valid_pixels = (block_end - block_offset) / NUM_CHANNELS;
            internal_delegate.ConsumeTile<false>(block_offset, valid_pixels);
        }

        // Aggregate output
        internal_delegate.AggregateOutput();
    }

};




}               // CUB namespace
CUB_NS_POSTFIX  // Optional outer namespace(s)

