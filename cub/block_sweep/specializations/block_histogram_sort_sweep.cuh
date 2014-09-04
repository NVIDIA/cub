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
 * cub::BlockHistogramSweepSort implements a stateful abstraction of CUDA thread blocks for histogramming multiple tiles as part of device-wide histogram using local sorting
 */

#pragma once

#include <iterator>

#include "../../block/block_radix_sort.cuh"
#include "../../block/block_discontinuity.cuh"
#include "../../util_namespace.cuh"

/// Optional outer namespace(s)
CUB_NS_PREFIX

/// CUB namespace
namespace cub {


/**
 * BlockHistogramSweepSort implements a stateful abstraction of CUDA thread blocks for histogramming multiple tiles as part of device-wide histogram using local sorting
 */
template <
    typename    BlockHistogramSweepPolicy,      ///< Tuning policy
    int         MAX_PRIVATIZED_BINS,            ///< Maximum number of privatized shared-memory histogram bins of any channel.  Zero indicates privatized counters to be maintained in global memory.
    int         NUM_CHANNELS,                   ///< Number of channels interleaved in the input data
    int         NUM_ACTIVE_CHANNELS,            ///< Number of channels actively being histogrammed
    typename    InputIteratorT,                 ///< The input iterator type. \iterator
    typename    CounterT,                       ///< Integer type for counting sample occurrences per histogram bin
    typename    SampleTransformOpT,             ///< Transform operator type for determining bin-ids from samples for each channel
    typename    OffsetT>                        ///< Signed integer type for global offsets
struct BlockHistogramSweepSort
{
    //---------------------------------------------------------------------
    // Types and constants
    //---------------------------------------------------------------------

    // Sample type
    typedef typename std::iterator_traits<InputIteratorT>::value_type SampleT;

    // BinId type
    typedef If<(sizeof(SampleT) > 1), unsigned short, unsigned char>::Type BinId;

    // Constants
    enum
    {
        BLOCK_THREADS               = BlockHistogramSweepPolicy::BLOCK_THREADS,
        PIXELS_PER_THREAD           = BlockHistogramSweepPolicy::PIXELS_PER_THREAD,
        TILE_PIXELS                 = BLOCK_THREADS * PIXELS_PER_THREAD,
        STRIPED_COUNTERS_PER_THREAD = (MAX_PRIVATIZED_BINS + BLOCK_THREADS - 1) / BLOCK_THREADS,
    };

    // Parameterize BlockRadixSort type for our thread block
    typedef BlockRadixSort<BinId, BLOCK_THREADS, PIXELS_PER_THREAD> BlockRadixSortT;

    // Parameterize BlockDiscontinuity type for our thread block
    typedef BlockDiscontinuity<BinId, BLOCK_THREADS> BlockDiscontinuityT;

    /// Shared memory type required by this thread block
    union _TempStorage
    {
        // Storage for sorting bin values
        typename BlockRadixSortT::TempStorage sort;

        struct
        {
            // Storage for detecting discontinuities in the tile of sorted bin values
            typename BlockDiscontinuityT::TempStorage flag;

            // Storage for noting begin/end offsets of bin runs in the tile of sorted bin values
            BinId run_begin[(BLOCK_THREADS * STRIPED_COUNTERS_PER_THREAD) + 1];     // Accommodate out-of-bounds samples
            BinId run_end[(BLOCK_THREADS * STRIPED_COUNTERS_PER_THREAD) + 1];       // Accommodate out-of-bounds samples
        };
    };


    /// Alias wrapper allowing storage to be unioned
    struct TempStorage : Uninitialized<_TempStorage> {};


    // Discontinuity functor
    struct DiscontinuityOp
    {
        // Reference to temp_storage
        _TempStorage &temp_storage;

        // Constructor
        __device__ __forceinline__ DiscontinuityOp(_TempStorage &temp_storage) :
            temp_storage(temp_storage)
        {}

        // Discontinuity predicate
        __device__ __forceinline__ bool operator()(const SampleT &a, const SampleT &b, int b_index)
        {
            if (a != b)
            {
                // Note the begin/end offsets in shared storage
                temp_storage.run_begin[b] = b_index;
                temp_storage.run_end[a] = b_index;

                return true;
            }
            else
            {
                return false;
            }
        }
    };


    //---------------------------------------------------------------------
    // Per-thread fields
    //---------------------------------------------------------------------

    /// Reference to temp_storage
    _TempStorage &temp_storage;

    /// Histogram counters striped across threads
    CounterT thread_counters[NUM_ACTIVE_CHANNELS][STRIPED_COUNTERS_PER_THREAD];

    /// Reference to output histograms
    CounterT* (&d_out_histograms)[NUM_ACTIVE_CHANNELS];

    /// Transform operators for determining bin-ids from samples, one for each channel
    SampleTransformOpT* (&transform_op)[NUM_ACTIVE_CHANNELS];

    /// Input data to reduce
    InputIteratorT d_in;


    //---------------------------------------------------------------------
    // Interface
    //---------------------------------------------------------------------

    /**
     * Constructor
     */
    __device__ __forceinline__ BlockHistogramSweepSort(
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
        // Initialize histogram counters striped across threads
        #pragma unroll
        for (int CHANNEL = 0; CHANNEL < NUM_ACTIVE_CHANNELS; ++CHANNEL)
        {
            #pragma unroll
            for (int COUNTER = 0; COUNTER < STRIPED_COUNTERS_PER_THREAD; ++COUNTER)
            {
                thread_counters[CHANNEL][COUNTER] = 0;
            }
        }
    }


    /**
     * Composite a tile of input bin_ids
     */
    __device__ __forceinline__ void Composite(
        BinId           bin_ids[PIXELS_PER_THREAD],                     ///< Tile of bin-ids
        CounterT        thread_counters[STRIPED_COUNTERS_PER_THREAD])   ///< Histogram counters striped across threads
    {
        // Sort bytes in blocked arrangement
        BlockRadixSortT(temp_storage.sort).Sort(bin_ids);

        __syncthreads();

        // Initialize the shared memory's run_begin and run_end for each bin
        #pragma unroll
        for (int COUNTER = 0; COUNTER < STRIPED_COUNTERS_PER_THREAD; ++COUNTER)
        {
            temp_storage.run_begin[(COUNTER * BLOCK_THREADS) + threadIdx.x] = TILE_PIXELS;
            temp_storage.run_end[(COUNTER * BLOCK_THREADS) + threadIdx.x]   = TILE_PIXELS;
        }

        __syncthreads();

        // Note the begin/end run offsets of bin runs in the sorted tile
        int flags[PIXELS_PER_THREAD];                // unused
        DiscontinuityOp flag_op(temp_storage);
        BlockDiscontinuityT(temp_storage.flag).FlagHeads(flags, bin_ids, flag_op);

        // Update begin for first item
        if (threadIdx.x == 0)
            temp_storage.run_begin[bin_ids[0]] = 0;

        __syncthreads();

        // Composite into histogram
        // Initialize the shared memory's run_begin and run_end for each bin
        #pragma unroll
        for (int COUNTER = 0; COUNTER < STRIPED_COUNTERS_PER_THREAD; ++COUNTER)
        {
            BinId       bin            = (COUNTER * BLOCK_THREADS) + threadIdx.x;
            CounterT    run_length     = temp_storage.run_end[bin] - temp_storage.run_begin[bin];

            thread_counters[COUNTER] += run_length;
        }
    }


    /**
     * Process one channel within a tile.  Inductive step.
     */
    template <bool FULL_TILE, int CHANNEL>
    __device__ __forceinline__ void ConsumeTileChannel(
        OffsetT             block_offset,
        int                 valid_pixels,
        Int2Type<CHANNEL>   channel)
    {
        __syncthreads();

        // Load bin_ids in striped fashion
        if (FULL_TILE)
        {
            // Full tile of samples to read and composite
            BinId bin_ids[PIXELS_PER_THREAD];

            // Unguarded loads
            #pragma unroll
            for (int PIXEL = 0; PIXEL < PIXELS_PER_THREAD; PIXEL++)
            {
                SampleT sample = d_in[CHANNEL + block_offset + (PIXEL * BLOCK_THREADS * NUM_CHANNELS) + (threadIdx.x * NUM_CHANNELS)];
                bin_ids[PIXEL] = (BinId) transform_op[CHANNEL](sample);
            }

            // Composite our histogram data
            Composite(bin_ids, thread_counters[CHANNEL]);
        }
        else
        {
            // Only a partially-full tile of samples to read and composite
            BinId bin_ids[PIXELS_PER_THREAD];

            #pragma unroll
            for (int PIXEL = 0; PIXEL < PIXELS_PER_THREAD; PIXEL++)
            {
                if (PIXEL * BLOCK_THREADS < valid_pixels - threadIdx.x)
                {
                    SampleT sample  = d_in[CHANNEL + block_offset + (PIXEL * BLOCK_THREADS * NUM_CHANNELS) + (threadIdx.x * NUM_CHANNELS)];
                    bin_ids[PIXEL]   = (BinId) transform_op[CHANNEL](sample);
                }
                else
                {
                    bin_ids[PIXEL] = 0;
                }
            }

            // Composite our histogram data
            Composite(bin_ids, thread_counters[CHANNEL]);

            __syncthreads();

            // Correct the overcounting in the zero-bin from invalid (out-of-bounds) bin_ids
            if (threadIdx.x == 0)
            {
                int extra_zeros = TILE_PIXELS - valid_pixels;
                thread_counters[CHANNEL][0] -= extra_zeros;
            }
        }

        // Consume next channel
        ConsumeTileChannel<FULL_TILE>(block_offset, valid_pixels, Int2Type<CHANNEL + 1>());
    }


    /**
     * Process one channel within a tile.  Base step.
     */
    template <bool FULL_TILE, int CHANNEL>
    __device__ __forceinline__ void ConsumeTileChannel(
        OffsetT                         block_offset,
        int                             valid_pixels,
        Int2Type<NUM_ACTIVE_CHANNELS>   channel)
    {}


    /**
     * Process a single tile of input
     */
    template <bool FULL_TILE>
    __device__ __forceinline__ void ConsumeTile(
        OffsetT     block_offset,                       ///< The offset the tile to consume
        int         valid_pixels = TILE_PIXELS)         ///< The number of valid pixels in the tile
    {
        // Iterate channels
        ConsumeTileChannel<FULL_TILE>(block_offset, valid_pixels, Int2Type<0>());
    }


    /**
     * Aggregate results into output
     */
    __device__ __forceinline__ void AggregateOutput()
    {
        // Copy counters striped across threads into the histogram output
        #pragma unroll
        for (int CHANNEL = 0; CHANNEL < NUM_ACTIVE_CHANNELS; ++CHANNEL)
        {
            int block_id = (blockIdx.y * gridDim.x) + blockIdx.x;
            int channel_offset  = (block_id * MAX_PRIVATIZED_BINS);

            #pragma unroll
            for (int COUNTER = 0; COUNTER < STRIPED_COUNTERS_PER_THREAD; ++COUNTER)
            {
                int bin = (COUNTER * BLOCK_THREADS) + threadIdx.x;

                if ((STRIPED_COUNTERS_PER_THREAD * BLOCK_THREADS == MAX_PRIVATIZED_BINS) || (bin < MAX_PRIVATIZED_BINS))
                {
                    d_out_histograms[CHANNEL][channel_offset + bin] = thread_counters[CHANNEL][COUNTER];
                }
            }
        }
    }
};




}               // CUB namespace
CUB_NS_POSTFIX  // Optional outer namespace(s)

