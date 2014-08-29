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
 * cub::BlockUnzipChannelsSweep implements a stateful abstraction of CUDA thread blocks for unzipping "pixels" of interleaved data channels
 */

#pragma once

#include <iterator>

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
 * Parameterizable tuning policy type for BlockScanSweep
 */
template <
    int                     _BLOCK_THREADS,                 ///< Threads per thread block
    int                     _PIXELS_PER_THREAD>             ///< Pixels per thread (per tile of input)
struct BlockUnzipChannelsSweepPolicy
{
    enum
    {
        BLOCK_THREADS       = _BLOCK_THREADS,               ///< Threads per thread block
        PIXELS_PER_THREAD   = _PIXELS_PER_THREAD,           ///< Pixels per thread (per tile of input)
    };
};


/******************************************************************************
 * Thread block abstractions
 ******************************************************************************/


/**
 * \brief BlockUnzipChannelsSweep implements a stateful abstraction of CUDA thread blocks for unzipping interleaved data channels
 */
template <
    typename    BlockUnzipChannelsSweepPolicyT, ///< Parameterized BlockUnzipChannelsSweepPolicy tuning policy type
    int         NUM_CHANNELS,                   ///< Number of channels interleaved in the input data (may be greater than the number of active channels being histogrammed)
    int         NUM_ACTIVE_CHANNELS,            ///< Number of channels actively being histogrammed
    typename    InputIteratorT,                 ///< Random-access input iterator type for reading samples.
    typename    OutputIteratorT,                ///< Random-access output iterator type
    typename    OffsetT>                        ///< Signed integer type for global offsets
struct BlockUnzipChannelsSweep
{
    //---------------------------------------------------------------------
    // Types and constants
    //---------------------------------------------------------------------

    // Data type of input iterator
    typedef typename std::iterator_traits<InputIteratorT>::value_type SampleT;

    enum
    {
        BLOCK_THREADS       = BlockUnzipChannelsSweepPolicyT::TILE_PIXELS,
        PIXELS_PER_THREAD   = BlockUnzipChannelsSweepPolicyT::PIXELS_PER_THREAD,
        TILE_PIXELS         = BLOCK_THREADS * PIXELS_PER_THREAD,
        TILE_SAMPLES        = TILE_PIXELS * NUM_CHANNELS,
    };

    //---------------------------------------------------------------------
    // Per-thread fields
    //---------------------------------------------------------------------

    InputIteratorT          d_in;                           ///< Pointer to input data
    OutputIteratorT         (&d_out)[NUM_ACTIVE_CHANNELS];  ///< Pointers to output data

    //---------------------------------------------------------------------
    // Interface
    //---------------------------------------------------------------------

    /**
     * Constructor
     */
    __device__ __forceinline__ BlockUnzipChannelsSweep(
        InputIteratorT       d_in,                           ///< Pointer to input data
        OutputIteratorT      (&d_out)[NUM_ACTIVE_CHANNELS])  ///< Pointers to output data
    :
        d_in(d_in), d_out(d_out)
    {}


    /**
     * Process a single tile of input
     */
    template <bool FULL_TILE>
    __device__ __forceinline__ void ConsumeTile(
        OffsetT             block_offset_in,                                    ///< The offset the tile to consume
        OffsetT             block_offset_out,                                   ///< The offset the tile to produce in each channel
        int                 valid_pixels = TILE_PIXELS)                         ///< The number of valid pixels in the tile
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
                    samples[PIXEL][CHANNEL] = d_in[block_offset_in + (PIXEL * BLOCK_THREADS * NUM_CHANNELS) + (threadIdx.x * NUM_CHANNELS) + CHANNEL];
                }
            }

            __threadfence_block();

            #pragma unroll
            for (int PIXEL = 0; PIXEL < PIXELS_PER_THREAD; PIXEL++)
            {
                #pragma unroll
                for (int CHANNEL = 0; CHANNEL < NUM_ACTIVE_CHANNELS; ++CHANNEL)
                {
                    d_out[CHANNEL][block_offset_out + (PIXEL * BLOCK_THREADS) + threadIdx.x] = samples[PIXEL][CHANNEL];
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
                        SampleT sample = d_in[block_offset_in + (PIXEL * BLOCK_THREADS * NUM_CHANNELS) + (threadIdx.x * NUM_CHANNELS) + CHANNEL];
                        d_out[CHANNEL][block_offset_out + (PIXEL * BLOCK_THREADS) + threadIdx.x] = sample;
                    }
                }
            }
        }
    }


    /**
     * \brief Consume striped tiles
     */
    __device__ __forceinline__ void ConsumeStriped(
        OffsetT             block_offset_in,                ///< [in] Threadblock input begin offset
        OffsetT             block_end,                      ///< [in] Threadblock input end offset (exclusive)
        OffsetT             block_offset_out)               ///< [in] Threadblock out begin offset
    {
        // Consume subsequent full tiles of input
        while (block_offset_in + TILE_SAMPLES <= block_end)
        {
            ConsumeTile<true>(block_offset_in, block_offset_out);
            block_offset_in += TILE_SAMPLES;
            block_offset_out += TILE_PIXELS;
        }

        // Consume a partially-full tile
        if (block_offset_in < block_end)
        {
            int valid_pixels = (block_end - block_offset_in) / NUM_CHANNELS;
            ConsumeTile<false>(block_offset_in, block_offset_out, valid_pixels);
        }
    }

};





}               // CUB namespace
CUB_NS_POSTFIX  // Optional outer namespace(s)

