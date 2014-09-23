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

#include "../util_type.cuh"
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
 * Parameterizable tuning policy type for BlockHistogramSweep
 */
template <
    int                     _BLOCK_THREADS,                         ///< Threads per thread block
    int                     _PIXELS_PER_THREAD,                     ///< Pixels per thread (per tile of input)
    CacheLoadModifier       _LOAD_MODIFIER,                         ///< Cache load modifier for reading input elements
    bool                    _LOAD_FENCE>                            ///< Whether to prevent hoisting computation into loads
struct BlockHistogramSweepPolicy
{
    enum
    {
        BLOCK_THREADS       = _BLOCK_THREADS,                       ///< Threads per thread block
        PIXELS_PER_THREAD   = _PIXELS_PER_THREAD,                   ///< Pixels per thread (per tile of input)
        LOAD_FENCE          = _LOAD_FENCE,                          ///< Whether to prevent hoisting computation into loads
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
    typename    SampleIteratorT,                ///< Random-access input iterator type for reading samples.
    typename    CounterT,                       ///< Integer type for counting sample occurrences per histogram bin
    typename    SampleTransformOpT,             ///< Transform operator type for determining bin-ids from samples for each channel
    typename    OffsetT,                        ///< Signed integer type for global offsets
    int         PTX_ARCH = CUB_PTX_ARCH>        ///< PTX compute capability
struct BlockHistogramSweep
{
    //---------------------------------------------------------------------
    // Types and constants
    //---------------------------------------------------------------------

    /// The sample value type of the input iterator
    typedef typename std::iterator_traits<SampleIteratorT>::value_type SampleT;

    /// Constants
    enum
    {
        BLOCK_THREADS           = BlockHistogramSweepPolicy::BLOCK_THREADS,
        PIXELS_PER_THREAD       = BlockHistogramSweepPolicy::PIXELS_PER_THREAD,
        TILE_PIXELS             = PIXELS_PER_THREAD * BLOCK_THREADS,
        TILE_SAMPLES            = TILE_PIXELS * NUM_CHANNELS,
        USE_SHARED_MEM          = ((MAX_PRIVATIZED_BINS > 0) && (PTX_ARCH >= 120)),

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
        IS_VECTOR_SUITABLE      = (VECTOR_LOAD_LENGTH > 1) && (IsPointer<SampleIteratorT>::VALUE) && Traits<SampleT>::PRIMITIVE,

        // Number of vector loads per thread
        VECTOR_LOADS = (PIXELS_PER_THREAD * NUM_CHANNELS) / VECTOR_LOAD_LENGTH,

        // Whether to prevent hoisting computation into loads
        LOAD_FENCE = BlockHistogramSweepPolicy::LOAD_FENCE,
    };

    /// Vector type of SampleT for aligned data movement
    typedef typename CubVector<SampleT, VECTOR_LOAD_LENGTH>::Type VectorT;

    /// Input iterator wrapper type
    typedef typename If<IsPointer<SampleIteratorT>::VALUE,
            CacheModifiedInputIterator<BlockHistogramSweepPolicy::LOAD_MODIFIER, SampleT, OffsetT>,     // Wrap the native input pointer with CacheModifiedInputIterator
            SampleIteratorT>::Type                                                                       // Directly use the supplied input iterator type
        WrappedSampleIteratorT;

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
    SampleIteratorT d_samples;

    /// Cache-modified input data to reduce
    WrappedSampleIteratorT d_wrapped_samples;

    /// The number of bins for each channel
    int (&num_bins)[NUM_ACTIVE_CHANNELS];

    /// Whether or not input is vector-aligned
    bool can_vectorize;

    //---------------------------------------------------------------------
    // Utility
    //---------------------------------------------------------------------

    // Whether or not the input is aligned with the vector type.  Specialized for types we can vectorize
    template <typename Iterator>
    static __device__ __forceinline__ bool IsAligned(
        Iterator        d_samples,
        OffsetT         row_stride_samples,
        Int2Type<true>  is_vector_suitable)
    {
        // both row stride and starting pointer must be vector-aligned
        return ((size_t(d_samples) | (sizeof(SampleT) * row_stride_samples)) & (sizeof(VectorT) - 1)) == 0;
    }


    // Whether or not the input is aligned with the vector type.  Specialized for types we cannot vectorize
    template <typename Iterator>
    static __device__ __forceinline__ bool IsAligned(
        Iterator        d_samples,
        OffsetT         row_stride_samples,
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
            int bin_base = 0;

            #pragma unroll
            for (; bin_base + BLOCK_THREADS <= MAX_PRIVATIZED_BINS; bin_base += BLOCK_THREADS)
            {
                temp_storage.histograms[CHANNEL][bin_base + threadIdx.x] = 0;
            }

            // Finish up with guarded initialization if necessary
            if ((MAX_PRIVATIZED_BINS % BLOCK_THREADS != 0) && (bin_base + threadIdx.x < MAX_PRIVATIZED_BINS))
            {
                temp_storage.histograms[CHANNEL][bin_base + threadIdx.x] = 0;
            }
        }

        // Barrier to make sure all threads are done updating counters
        __syncthreads();
    }

    // Initialize privatized bin counters.  Specialized for privatized global-memory counters
    __device__ __forceinline__ void InitBinCounters(
        Int2Type<false> use_shared_mem)
    {
        // Initialize histogram bin counts to zeros
        #pragma unroll
        for (int CHANNEL = 0; CHANNEL < NUM_ACTIVE_CHANNELS; ++CHANNEL)
        {
            OffsetT block_histo_offset = ((blockIdx.y * gridDim.x) + blockIdx.x) * num_bins[CHANNEL];

            for (int bin = threadIdx.x; bin < num_bins[CHANNEL]; bin += BLOCK_THREADS)
            {
                d_out_histograms[CHANNEL][block_histo_offset + bin] = 0;
            }
        }

        // Barrier to make sure all threads are done updating counters
        __syncthreads();
    }


    // Store privatized histogram to global memory.  Specialized for privatized shared-memory counters
    __device__ __forceinline__ void StoreOutput(
        Int2Type<true> use_shared_mem)
    {
        // Barrier to make sure all threads are done updating counters
        __syncthreads();

        // Copy bin counts
        #pragma unroll
        for (int CHANNEL = 0; CHANNEL < NUM_ACTIVE_CHANNELS; ++CHANNEL)
        {
            OffsetT block_histo_offset = ((blockIdx.y * gridDim.x) + blockIdx.x) * num_bins[CHANNEL];
            for (int bin = threadIdx.x; bin < num_bins[CHANNEL]; bin += BLOCK_THREADS)
            {
                d_out_histograms[CHANNEL][block_histo_offset + bin] = temp_storage.histograms[CHANNEL][bin];
            }
        }
    }

    // Store privatized histogram to global memory.  Specialized for privatized global-memory counters
    __device__ __forceinline__ void StoreOutput(
        Int2Type<false> use_shared_mem)
    {
        // No work to be done
    }


    /**
     * Accumulate pixel, specialized for gmem privatized histogram
     */
    __device__ __forceinline__ void AccumulatePixels(
        SampleT             samples[PIXELS_PER_THREAD][NUM_CHANNELS],
        bool                is_valid[PIXELS_PER_THREAD],
        Int2Type<false>     use_shared_mem)
    {
        #pragma unroll
        for (int PIXEL = 0; PIXEL < PIXELS_PER_THREAD; ++PIXEL)
        {
            #pragma unroll
            for (int CHANNEL = 0; CHANNEL < NUM_ACTIVE_CHANNELS; ++CHANNEL)
            {
                OffsetT block_histo_offset = ((blockIdx.y * gridDim.x) + blockIdx.x) * num_bins[CHANNEL];

                int bin;
                bool valid_sample = is_valid[PIXEL];
                transform_op[CHANNEL].BinSelect<BlockHistogramSweepPolicy::LOAD_MODIFIER>(samples[PIXEL][CHANNEL], bin, valid_sample);
                if (valid_sample)
                    atomicAdd(d_out_histograms[CHANNEL] + block_histo_offset + bin, 1);
            }
        }
    }


    /**
     * Accumulate pixel, specialized for smem privatized histogram
     */
    __device__ __forceinline__ void AccumulatePixels(
        SampleT             samples[PIXELS_PER_THREAD][NUM_CHANNELS],
        bool                is_valid[PIXELS_PER_THREAD],
        Int2Type<true>      use_shared_mem)
    {
        #pragma unroll
        for (int PIXEL = 0; PIXEL < PIXELS_PER_THREAD; ++PIXEL)
        {
            #pragma unroll
            for (int CHANNEL = 0; CHANNEL < NUM_ACTIVE_CHANNELS; ++CHANNEL)
            {
                int bin;
                bool valid_sample = is_valid[PIXEL];
                transform_op[CHANNEL].BinSelect<BlockHistogramSweepPolicy::LOAD_MODIFIER>(samples[PIXEL][CHANNEL], bin, valid_sample);
                if (valid_sample)
                {
                    atomicAdd(temp_storage.histograms[CHANNEL] + bin, 1);
                }
            }
        }
    }


    /**
     * Load a full tile of data samples.  Specialized for non-pointer types (no vectorization)
     */
    template <bool IS_FULL_TILE>
    __device__ __forceinline__ void ConsumeTile(
        OffsetT         block_row_offset,
        int             valid_pixels,
        Int2Type<false> is_vector_suitable)
    {
        SampleT     samples[PIXELS_PER_THREAD][NUM_CHANNELS];
        bool        is_valid[PIXELS_PER_THREAD];

        // Read striped pixels
        #pragma unroll
        for (int PIXEL = 0; PIXEL < PIXELS_PER_THREAD; ++PIXEL)
        {
            int pixel_idx = (PIXEL * BLOCK_THREADS) + threadIdx.x;
            if ((is_valid[PIXEL] = (IS_FULL_TILE || (pixel_idx < valid_pixels))))
            {
                #pragma unroll
                for (int CHANNEL = 0; CHANNEL < NUM_ACTIVE_CHANNELS; ++CHANNEL)
                {
                    samples[PIXEL][CHANNEL] = d_wrapped_samples[block_row_offset + (pixel_idx * NUM_CHANNELS) + CHANNEL];
                }
            }
        }

        if (LOAD_FENCE)
            __threadfence_block();

        // Accumulate samples
        AccumulatePixels(samples, is_valid, Int2Type<USE_SHARED_MEM>());
    }


    /**
     * Load a full tile of data samples.  Specialized for pointer types (possible vectorization)
     */
    template <bool IS_FULL_TILE>
    __device__ __forceinline__ void ConsumeTile(
        OffsetT         block_row_offset,
        int             valid_pixels,
        Int2Type<true>  is_vector_suitable)
    {
        if (((NUM_CHANNELS == 1) && (!IS_FULL_TILE)) || !can_vectorize)
        {
            // Not a full tile of single-channel samples, or not aligned.  Load un-vectorized
            ConsumeTile<IS_FULL_TILE>(block_row_offset, valid_pixels, Int2Type<false>());
        }
        else if (NUM_CHANNELS == 1)
        {
            // Full single-channel tile
            SampleT     samples[PIXELS_PER_THREAD][NUM_CHANNELS];
            bool        is_valid[PIXELS_PER_THREAD];

            // Alias items as an array of VectorT and load it in striped fashion
            VectorT *vec_items = reinterpret_cast<VectorT*>(samples);

            // Vector input iterator wrapper at starting pixel
            SampleT *d_samples_unqualified = const_cast<SampleT*>(d_samples) + block_row_offset + (threadIdx.x * VECTOR_LOAD_LENGTH);
            VectorT *ptr = reinterpret_cast<VectorT*>(d_samples_unqualified);
            InputIteratorVectorT d_vec_in(ptr);

            // Read striped vectors
            #pragma unroll
            for (int VECTOR = 0; VECTOR < (PIXELS_PER_THREAD / VECTOR_LOAD_LENGTH); ++VECTOR)
            {
                vec_items[VECTOR] = d_vec_in[VECTOR * BLOCK_THREADS];
            }

            if (LOAD_FENCE)
                 __threadfence_block();

            #pragma unroll
            for (int PIXEL = 0; PIXEL < PIXELS_PER_THREAD; ++PIXEL)
                is_valid[PIXEL] = true;

            // Accumulate samples
            AccumulatePixels(samples, is_valid, Int2Type<USE_SHARED_MEM>());
        }
        else
        {
            // Multi-channel tile
            SampleT     samples[PIXELS_PER_THREAD][NUM_CHANNELS];
            bool        is_valid[PIXELS_PER_THREAD];

            // Alias items as an array of VectorT and load it in striped fashion
            VectorT *vec_items = reinterpret_cast<VectorT*>(samples);

            // Vector input iterator wrapper at starting pixel
            SampleT *d_samples_unqualified = const_cast<SampleT*>(d_samples) + block_row_offset + (threadIdx.x * VECTOR_LOAD_LENGTH);
            VectorT *ptr = reinterpret_cast<VectorT*>(d_samples_unqualified);
            InputIteratorVectorT d_vec_in(ptr);

            // Read striped pixels
            #pragma unroll
            for (int PIXEL = 0; PIXEL < PIXELS_PER_THREAD; ++PIXEL)
            {
                int pixel_idx = (PIXEL * BLOCK_THREADS) + threadIdx.x;
                if ((is_valid[PIXEL] = (IS_FULL_TILE || (pixel_idx < valid_pixels))))
                {
                    vec_items[PIXEL] = d_vec_in[PIXEL * BLOCK_THREADS];
                }
            }

            if (LOAD_FENCE)
                 __threadfence_block();

            // Accumulate samples
            AccumulatePixels(samples, is_valid, Int2Type<USE_SHARED_MEM>());
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
        SampleIteratorT    	d_samples,                                      ///< Input data to reduce
        OffsetT             row_stride_samples,                             ///< The number of samples between starts of consecutive rows in the region of interest
        CounterT*           (&d_out_histograms)[NUM_ACTIVE_CHANNELS],       ///< Reference to output histograms
        SampleTransformOpT  (&transform_op)[NUM_ACTIVE_CHANNELS],           ///< Transform operators for determining bin-ids from samples, one for each channel
        int                 (&num_bins)[NUM_ACTIVE_CHANNELS])               ///< The number of boundaries (levels) for delineating histogram samples, one for each channel
    :
        temp_storage(temp_storage.Alias()),
        d_samples(d_samples),
        d_wrapped_samples(d_samples),
        d_out_histograms(d_out_histograms),
        transform_op(transform_op),
        num_bins(num_bins),
        can_vectorize(IsAligned(d_samples, row_stride_samples, Int2Type<IS_VECTOR_SUITABLE>()))
    {
        InitBinCounters(Int2Type<USE_SHARED_MEM>());

        // Barrier to make sure all counters are initialized
        __syncthreads();
    }


    /**
     * \brief Consume striped tiles
     */
    __device__ __forceinline__ void ConsumeStriped(
        OffsetT  row_offset,                       ///< [in] Row begin offset (inclusive)
        OffsetT  row_end)                          ///< [in] Row end offset (exclusive)
    {
        for (
            OffsetT block_row_offset = row_offset + (blockIdx.x * TILE_SAMPLES);
            block_row_offset < row_end;
            block_row_offset += gridDim.x * TILE_SAMPLES)
        {
            if (block_row_offset + TILE_SAMPLES <= row_end)
            {
                ConsumeTile<true>(block_row_offset, TILE_SAMPLES, Int2Type<IS_VECTOR_SUITABLE>());
            }
            else
            {
                int valid_pixels = (row_end - block_row_offset) / NUM_CHANNELS;
                ConsumeTile<false>(block_row_offset, valid_pixels, Int2Type<IS_VECTOR_SUITABLE>());
            }
        }
    }


    /**
     * Initialize privatized bin counters.  Specialized for privatized shared-memory counters
     */
    __device__ __forceinline__ void InitBinCounters()
    {
        InitBinCounters(Int2Type<USE_SHARED_MEM>());
    }


    /**
     * Store privatized histogram to global memory.  Specialized for privatized shared-memory counters
     */
    __device__ __forceinline__ void StoreOutput()
    {
        StoreOutput(Int2Type<USE_SHARED_MEM>());
    }


};




}               // CUB namespace
CUB_NS_POSTFIX  // Optional outer namespace(s)

