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
 * cub::BlockHistogramSweepSharedAtomic implements a stateful abstraction of CUDA thread blocks for histogramming multiple tiles as part of device-wide histogram using shared atomics
 */

#pragma once

#include <iterator>

#include "../../util_type.cuh"
#include "../../util_namespace.cuh"

/// Optional outer namespace(s)
CUB_NS_PREFIX

/// CUB namespace
namespace cub {


/**
 * BlockHistogramSweepSharedAtomic implements a stateful abstraction of CUDA thread blocks for histogramming multiple tiles as part of device-wide histogram using shared atomics
 */
template <
    typename    BlockHistogramSweepPolicy,		///< Tuning policy
    int         MAX_PRIVATIZED_BINS,            ///< Number of histogram bins per channel (e.g., up to 256)
    int         NUM_CHANNELS,                   ///< Number of channels interleaved in the input data (may be greater than the number of active channels being histogrammed)
    int         NUM_ACTIVE_CHANNELS,            ///< Number of channels actively being histogrammed
    typename    InputIteratorT,               	///< The input iterator type. \iterator
    typename    CounterT,                       ///< Integer type for counting sample occurrences per histogram bin
    typename    SampleTransformOpT,             ///< Transform operator type for determining bin-ids from samples for each channel
    typename    OffsetT>                        ///< Signed integer type for global offsets
struct BlockHistogramSweepSharedAtomic
{
    //---------------------------------------------------------------------
    // Types and constants
    //---------------------------------------------------------------------

    // Sample type
    typedef typename std::iterator_traits<InputIteratorT>::value_type SampleT;

    // Constants
    enum
    {
        BLOCK_THREADS           = BlockHistogramSweepPolicy::BLOCK_THREADS,
        PIXELS_PER_THREAD       = BlockHistogramSweepPolicy::PIXELS_PER_THREAD,
        TILE_PIXELS             = BLOCK_THREADS * PIXELS_PER_THREAD,
    };

    /// Shared memory type required by this thread block
    struct _TempStorage
    {
        CounterT histograms[NUM_ACTIVE_CHANNELS][MAX_PRIVATIZED_BINS + 1];  // Accommodate out-of-bounds samples
    };

    /// Alias wrapper allowing storage to be unioned
    struct TempStorage : Uninitialized<_TempStorage> {};


    //---------------------------------------------------------------------
    // Per-thread fields
    //---------------------------------------------------------------------

    /// Reference to temp_storage
    _TempStorage &temp_storage;

    /// Reference to output histograms
    CounterT* (&d_out_histograms)[NUM_ACTIVE_CHANNELS];

    /// Transform operators for determining bin-ids from samples, one for each channel
    SampleTransformOpT (&transform_op)[NUM_ACTIVE_CHANNELS];

    /// Input data to reduce
    InputIteratorT d_in;


    //---------------------------------------------------------------------
    // Interface
    //---------------------------------------------------------------------

    /**
     * Constructor
     */
    __device__ __forceinline__ BlockHistogramSweepSharedAtomic(
        TempStorage         &temp_storage,                                  ///< Reference to temp_storage
        InputIteratorT      d_in,                                           ///< Input data to reduce
        CounterT*           (&d_out_histograms)[NUM_ACTIVE_CHANNELS],       ///< Reference to output histograms
        SampleTransformOpT  (&transform_op)[NUM_ACTIVE_CHANNELS])           ///< Transform operators for determining bin-ids from samples, one for each channel
    :
        temp_storage(temp_storage.Alias()),
        d_in(d_in),
        d_out_histograms(d_out_histograms),
        transform_op(transform_op)
    {
        // Initialize histogram bin counts to zeros
        #pragma unroll
        for (int CHANNEL = 0; CHANNEL < NUM_ACTIVE_CHANNELS; ++CHANNEL)
        {
            int histo_offset = 0;

            #pragma unroll
            for(; histo_offset + BLOCK_THREADS <= MAX_PRIVATIZED_BINS; histo_offset += BLOCK_THREADS)
            {
                this->temp_storage.histograms[CHANNEL][histo_offset + threadIdx.x] = 0;
            }

            // Finish up with guarded initialization if necessary
            if ((MAX_PRIVATIZED_BINS % BLOCK_THREADS != 0) && (histo_offset + threadIdx.x < MAX_PRIVATIZED_BINS))
            {
                this->temp_storage.histograms[CHANNEL][histo_offset + threadIdx.x] = 0;
            }
        }

        __syncthreads();
    }


    /**
     * Process a single tile of input
     */
    template <bool FULL_TILE>
    __device__ __forceinline__ void ConsumeTile(
        OffsetT     block_offset,                       ///< The offset the tile to consume
        int         valid_pixels = TILE_PIXELS)         ///< The number of valid pixels in the tile
    {
        if (FULL_TILE)
        {
            // Full tile of samples to read and composite
            SampleT samples[PIXELS_PER_THREAD][NUM_ACTIVE_CHANNELS];

            #pragma unroll
            for (int PIXEL = 0; PIXEL < PIXELS_PER_THREAD; PIXEL++)
            {
                #pragma unroll
                for (int CHANNEL = 0; CHANNEL < NUM_ACTIVE_CHANNELS; ++CHANNEL)
                {
                    samples[PIXEL][CHANNEL] = d_in[block_offset + (PIXEL * BLOCK_THREADS * NUM_CHANNELS) + (threadIdx.x * NUM_CHANNELS) + CHANNEL];
                }
            }

            __threadfence_block();

            #pragma unroll
            for (int PIXEL = 0; PIXEL < PIXELS_PER_THREAD; PIXEL++)
            {
                #pragma unroll
                for (int CHANNEL = 0; CHANNEL < NUM_ACTIVE_CHANNELS; ++CHANNEL)
                {
                    int bin = (int) transform_op[CHANNEL](samples[PIXEL][CHANNEL]);
                    atomicAdd(temp_storage.histograms[CHANNEL] + bin, 1);
                }
            }

            __threadfence_block();
        }
        else
        {
            // Only a partially-full tile of samples to read and composite
            #pragma unroll
            for (int PIXEL = 0; PIXEL < PIXELS_PER_THREAD; ++PIXEL)
            {
                if (PIXEL * BLOCK_THREADS < valid_pixels - threadIdx.x)
                {
                    #pragma unroll
                    for (int CHANNEL = 0; CHANNEL < NUM_ACTIVE_CHANNELS; ++CHANNEL)
                    {
                        SampleT sample = d_in[block_offset + (PIXEL * BLOCK_THREADS * NUM_CHANNELS) + (threadIdx.x * NUM_CHANNELS) + CHANNEL];
                        int bin = (int) transform_op[CHANNEL](sample);
                        atomicAdd(temp_storage.histograms[CHANNEL] + bin, 1);
                    }
                }
            }
        }
    }


    /**
     * Aggregate results into output
     */
    __device__ __forceinline__ void AggregateOutput()
    {
        // Barrier to ensure shared memory histograms are coherent
        __syncthreads();

        // Copy shared memory histograms to output
        int block_id = (blockIdx.y * gridDim.x) + blockIdx.x;
        int channel_offset = (block_id * MAX_PRIVATIZED_BINS);

        #pragma unroll
        for (int CHANNEL = 0; CHANNEL < NUM_ACTIVE_CHANNELS; ++CHANNEL)
        {
            int histo_offset = 0;

            #pragma unroll
            for(; histo_offset + BLOCK_THREADS <= MAX_PRIVATIZED_BINS; histo_offset += BLOCK_THREADS)
            {
                CounterT count = temp_storage.histograms[CHANNEL][histo_offset + threadIdx.x];

                d_out_histograms[CHANNEL][channel_offset + histo_offset + threadIdx.x] = count;
            }

            // Finish up with guarded initialization if necessary
            if ((MAX_PRIVATIZED_BINS % BLOCK_THREADS != 0) && (histo_offset + threadIdx.x < MAX_PRIVATIZED_BINS))
            {
                CounterT count = temp_storage.histograms[CHANNEL][histo_offset + threadIdx.x];

                d_out_histograms[CHANNEL][channel_offset + histo_offset + threadIdx.x] = count;
            }
        }
    }



};



}               // CUB namespace
CUB_NS_POSTFIX  // Optional outer namespace(s)

