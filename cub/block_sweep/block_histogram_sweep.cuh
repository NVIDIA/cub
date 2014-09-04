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
    int                     _BLOCK_THREADS,                         ///< Threads per thread block
    int                     _PIXELS_PER_THREAD,                     ///< Pixels per thread (per tile of input)
    CacheLoadModifier       _LOAD_MODIFIER>                         ///< Cache load modifier for reading input elements
struct BlockHistogramSweepPolicy
{
    enum
    {
        BLOCK_THREADS       = _BLOCK_THREADS,                       ///< Threads per thread block
        PIXELS_PER_THREAD   = _PIXELS_PER_THREAD,                   ///< Pixels per thread (per tile of input)
    };

    static const CacheLoadModifier LOAD_MODIFIER = _LOAD_MODIFIER;  ///< Cache load modifier for reading input elements
};



/******************************************************************************
 * Thread block abstractions
 ******************************************************************************/

/**
 * \brief BlockHistogramSweep implements a stateful abstraction of CUDA thread blocks for participating in device-wide histogram across a range of tiles.
 */
template <
    typename    BlockHistogramSweepPolicy,      ///< Parameterized BlockHistogramSweepPolicy tuning policy type
    int         MAX_PRIVATIZED_BINS,            ///< Maximum number of privatized shared-memory histogram bins of any channel.  Zero indicates privatized counters to be maintained in global memory.
    int         NUM_CHANNELS,                   ///< Number of channels interleaved in the input data
    int         NUM_ACTIVE_CHANNELS,            ///< Number of channels actively being histogrammed
    typename    InputIteratorT,                 ///< Random-access input iterator type for reading samples.
    typename    CounterT,                       ///< Integer type for counting sample occurrences per histogram bin
    typename    SampleTransformOpT,             ///< Transform operator type for determining bin-ids from samples for each channel
    typename    OffsetT>                        ///< Signed integer type for global offsets
struct BlockHistogramSweep
{
    //---------------------------------------------------------------------
    // Types and constants
    //---------------------------------------------------------------------

    /// The sample value type of the input iterator
    typedef typename std::iterator_traits<InputIteratorT>::value_type SampleT;

    /// Constants
    enum
    {
        BLOCK_THREADS           = BlockHistogramSweepPolicy::BLOCK_THREADS,
        PIXELS_PER_THREAD       = BlockHistogramSweepPolicy::PIXELS_PER_THREAD,
        TILE_PIXELS             = PIXEL * BLOCK_THREADSS_PER_THREAD,
        TILE_SAMPLES            = TILE_PIXELS * NUM_CHANNELS,
        USE_SHARED_MEM          = MAX_PRIVATIZED_BINS > 0,

        VECTOR_LOAD_LENGTH      = (NUM_CHANNELS > 1) ?
                                    ((NUM_CHANNELS > 4) ?                   // Multi-channel: Use one vector per pixel if the number of samples-per-pixel less than or equal to four
                                        1 :
                                        NUM_CHANNELS) :
                                    ((PIXELS_PER_THREAD % 2 != 0) ?         // Single-channel: Use vectors to load multiple samples if samples-per-thread is a multiple of two
                                        1 :
                                        ((PIXELS_PER_THREAD % 4 == 0) ?
                                            4 :
                                            2)),

        // Whether or not the type system will let us load with vector types
        IS_VECTOR_SUITABLE      = (VECTOR_LOAD_LENGTH > 1) && (IsPointer<InputIteratorT>::VALUE) && Traits<SampleT>::PRIMITIVE,

        // Number of vector loads per thread
        VECTOR_LOADS = (PIXELS_PER_THREAD * NUM_CHANNELS) / VECTOR_LOAD_LENGTH
    };

    /// Vector type of SampleT for aligned data movement
    typedef typename CubVector<SampleT, BlockHistogramSweepPolicy::VECTOR_LOAD_LENGTH>::Type VectorT;

    /// Input iterator wrapper type
    typedef typename If<IsPointer<InputIteratorT>::VALUE,
            CacheModifiedInputIterator<BlockHistogramSweepPolicy::LOAD_MODIFIER, SampleT, OffsetT>,     // Wrap the native input pointer with CacheModifiedInputIterator
            InputIteratorT>::Type                                                                       // Directly use the supplied input iterator type
        WrappedInputIteratorSampleT;

    /// Vector input iterator type
    typedef CacheModifiedInputIterator<BlockHistogramSweepPolicy::LOAD_MODIFIER, VectorT, OffsetT> InputIteratorVectorT;

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

    /// The transform operator for determining bin-ids from samples, one for each channel
    SampleTransformOpT (&transform_op)[NUM_ACTIVE_CHANNELS];

    /// Input data to reduce
    WrappedInputIteratorSampleT d_in;

    /// The number of boundaries (levels) for delineating histogram samples, one for each channel
    int (&num_levels)[NUM_ACTIVE_CHANNELS];

    /// Whether or not input is vector-aligned
    bool can_vectorize;

    //---------------------------------------------------------------------
    // Utility
    //---------------------------------------------------------------------

    // Whether or not the input is aligned with the vector type.  Specialized for types we can vectorize
    template <typename Iterator>
    static __device__ __forceinline__ bool IsAligned(
        Iterator        d_in,
        Int2Type<true>  is_vector_suitable)
    {
        return (size_t(d_in) & (sizeof(VectorT) - 1)) == 0;
    }


    // Whether or not the input is aligned with the vector type.  Specialized for types we cannot vectorize
    template <typename Iterator>
    static __device__ __forceinline__ bool IsAligned(
        Iterator        d_in,
        Int2Type<false> is_vector_suitable)
    {
        return false;
    }


    // Initialize privatized bin counters.  Specialized for privatized shared-memory counters
    __device__ __forceinline__ void InitBinCounters(
        Int2Type<true> use_shared_mem)
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
    }

    // Initialize privatized bin counters.  Specialized for privatized global-memory counters
    __device__ __forceinline__ void InitBinCounters(
        Int2Type<false> use_shared_mem)
    {
        // Initialize histogram bin counts to zeros
        #pragma unroll
        for (int CHANNEL = 0; CHANNEL < NUM_ACTIVE_CHANNELS; ++CHANNEL)
        {
            int histo_offset = 0;

            #pragma unroll
            for(; histo_offset + BLOCK_THREADS <= MAX_PRIVATIZED_BINS; histo_offset += BLOCK_THREADS)
            {
                d_out_histograms[CHANNEL][histo_offset + threadIdx.x] = 0;
            }

            // Finish up with guarded initialization if necessary
            if ((MAX_PRIVATIZED_BINS % BLOCK_THREADS != 0) && (histo_offset + threadIdx.x < MAX_PRIVATIZED_BINS))
            {
                d_out_histograms[CHANNEL][histo_offset + threadIdx.x] = 0;
            }
        }
    }

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
        SampleTransformOpT  (&transform_op)[NUM_ACTIVE_CHANNELS],           ///< Transform operators for determining bin-ids from samples, one for each channel
        int                 (&num_levels)[NUM_ACTIVE_CHANNELS])             ///< The number of boundaries (levels) for delineating histogram samples, one for each channel
    :
            temp_storage(temp_storage.Alias()),
            d_in(d_in),
            d_out_histograms(d_out_histograms),
            transform_op(transform_op),
            num_levels(num_levels),
            can_vectorize(IsAligned(d_in, Int2Type<IS_VECTOR_SUITABLE>()))
    {
        InitBinCounters(Int2Type<USE_SHARED_MEM>());

        __syncthreads();
    }


    /**
     * Load a full tile of data samples.  Specialized for non-pointer types (no vectorization)
     */
    __device__ __forceinline__ void ConsumeFullTile(
        OffsetT         block_offset,
        Int2Type<false> is_vector_suitable)
    {
        SampleT samples[PIXELS_PER_THREAD][NUM_CHANNELS];

        // Read striped pixels
        #pragma unroll
        for (int PIXEL = 0; PIXEL < PIXELS_PER_THREAD; ++PIXEL)
        {
            OffsetT pixel_offset = (PIXEL * BLOCK_THREADS * NUM_CHANNELS) + (threadIdx.x * NUM_CHANNELS);

            #pragma unroll
            for (int CHANNEL = 0; CHANNEL < NUM_ACTIVE_CHANNELS; ++CHANNEL)
            {
                samples[PIXEL][CHANNEL] = d_in[block_offset + pixel_offset + CHANNEL];
            }
        }

        __threadfence_block();

        // Accumulate samples
    }


    /**
     * Load a full tile of data samples.  Specialized for pointer types (possible vectorization)
     */
    __device__ __forceinline__ void ConsumeFullTile(
        OffsetT         block_offset,
        Int2Type<true>  is_vector_suitable)
    {
        if (!can_vectorize)
        {
            // Not aligned
            ConsumeFullTile(block_offset, Int2Type<false>());
        }
        else
        {
            // Alias items as an array of VectorT and load it in striped fashion
            SampleT                 samples[PIXELS_PER_THREAD][NUM_CHANNELS];
            VectorT                 *vec_items = reinterpret_cast<VectorT*>(samples);

            // Vector input iterator wrapper at starting pixel
            InputIteratorVectorT    d_vec_in(reinterpret_cast<VectorT*>(d_in + block_offset + (threadIdx.x * VECTOR_LOAD_LENGTH)));

            if (NUM_CHANNELS > 1)
            {
                // Multi-channel: Read striped pixels
                #pragma unroll
                for (int PIXEL = 0; PIXEL < PIXELS_PER_THREAD; ++PIXEL)
                    vec_items[PIXEL] = d_vec_in[PIXEL * BLOCK_THREADS];
            }
            else
            {
                // Single-channel: read striped vectors
                #pragma unroll
                for (int VECTOR = 0; VECTOR < (PIXELS_PER_THREAD / VECTOR_LOAD_LENGTH); ++VECTOR)
                    vec_items[VECTOR] = d_vec_in[VECTOR * BLOCK_THREADS];
            }

            __threadfence_block();

            // Accumulate samples
        }
    }


    /**
     * Load a full tile of data samples.  Specialized for non-pointer types (no vectorization)
     */
    __device__ __forceinline__ void ConsumePartialTile(
        OffsetT         block_offset,
        int             valid_pixels,
        Int2Type<false> is_vector_suitable)
    {
        // Read striped pixels
        for (int pixel_idx = threadIdx.x; pixel_idx < valid_pixels; pixel_idx += BLOCK_THREADS)
        {
            SampleT pixel[NUM_ACTIVE_CHANNELS];

            #pragma unroll
            for (int CHANNEL = 0; CHANNEL < NUM_ACTIVE_CHANNELS; ++CHANNEL)
            {
                pixel[CHANNEL] = d_in[block_offset + (pixel_idx * NUM_CHANNELS) + CHANNEL];
            }

            // Accumulate pixel

        }

    }


    /**
     * Load a full tile of data samples.  Specialized for pointer types (possible vectorization)
     */
    __device__ __forceinline__ void ConsumePartialTile(
        OffsetT         block_offset,
        int             valid_pixels,
        Int2Type<true>  is_vector_suitable)
    {
        if ((NUM_CHANNELS == 1) || (!can_vectorize))
        {
            // Not aligned
            ConsumePartialTile(block_offset, valid_pixels, Int2Type<false>());
        }
        else
        {
            // Multi-channel: read striped pixels

            // Vector input iterator wrapper at starting pixel
            InputIteratorVectorT d_vec_in(
                reinterpret_cast<VectorT*>(d_in + block_offset + (threadIdx.x * NUM_CHANNELS)));

            for (int pixel_idx = threadIdx.x; pixel_idx < valid_pixels; pixel_idx += BLOCK_THREADS)
            {
                SampleT pixel[NUM_ACTIVE_CHANNELS];
                VectorT *vec_items = reinterpret_cast<VectorT*>(pixel);
                *vec_items = d_vec_in[pixel_idx];

                // Accumulate pixel

            }
        }
    }


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
            ConsumeFullTile(block_offset, Int2Type<IS_VECTOR_SUITABLE>());
            block_offset += TILE_SAMPLES;
        }

        // Consume a partially-full tile
        if (block_offset < block_end)
        {
            int valid_pixels = (block_end - block_offset) / NUM_CHANNELS;
            ConsumePartialTile(block_offset, valid_pixels, Int2Type<IS_VECTOR_SUITABLE>());
        }

        // Aggregate output
        AggregateOutput(Int2Type<MAX_PRIVATIZED_BINS>());
    }

};




}               // CUB namespace
CUB_NS_POSTFIX  // Optional outer namespace(s)

