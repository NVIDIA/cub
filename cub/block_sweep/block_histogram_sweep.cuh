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
    BlockLoadAlgorithm      _LOAD_ALGORITHM,                        ///< The BlockLoad algorithm to use
    CacheLoadModifier       _LOAD_MODIFIER,                         ///< Cache load modifier for reading input elements
    bool                    _RLE_COMPRESS>                          ///< Whether to perform localized RLE to compress samples before histogramming
struct BlockHistogramSweepPolicy
{
    enum
    {
        BLOCK_THREADS       = _BLOCK_THREADS,                       ///< Threads per thread block
        PIXELS_PER_THREAD   = _PIXELS_PER_THREAD,                   ///< Pixels per thread (per tile of input)
        RLE_COMPRESS        = _RLE_COMPRESS,                        ///< Whether to perform localized RLE to compress samples before histogramming
    };

    static const BlockLoadAlgorithm     LOAD_ALGORITHM          = _LOAD_ALGORITHM;          ///< The BlockLoad algorithm to use
    static const CacheLoadModifier      LOAD_MODIFIER           = _LOAD_MODIFIER;           ///< Cache load modifier for reading input elements
};



/******************************************************************************
 * Thread block abstractions
 ******************************************************************************/

/**
 * \brief BlockHistogramSweep implements a stateful abstraction of CUDA thread blocks for participating in device-wide histogram across a range of tiles.
 */
template <
    typename    BlockHistogramSweepPolicyT,     ///< Parameterized BlockHistogramSweepPolicy tuning policy type
    int         PRIVATIZED_SMEM_BINS,           ///< Number of privatized shared-memory histogram bins of any channel.  Zero indicates privatized counters to be maintained in global memory.
    int         NUM_CHANNELS,                   ///< Number of channels interleaved in the input data.  Supports up to four channels.
    int         NUM_ACTIVE_CHANNELS,            ///< Number of channels actively being histogrammed
    typename    SampleIteratorT,                ///< Random-access input iterator type for reading samples
    typename    CounterT,                       ///< Integer type for counting sample occurrences per histogram bin
    typename    BinDecodeOpT,                   ///< Transform operator type for determining bin-ids from samples for each channel
    typename    OffsetT,                        ///< Signed integer type for global offsets
    int         PTX_ARCH = CUB_PTX_ARCH>        ///< PTX compute capability
struct BlockHistogramSweep
{
    //---------------------------------------------------------------------
    // Types and constants
    //---------------------------------------------------------------------

    /// The sample type of the input iterator
    typedef typename std::iterator_traits<SampleIteratorT>::value_type SampleT;

    /// The pixel type of SampleT
    typedef typename CubVector<SampleT, NUM_CHANNELS>::Type PixelT;

    /// Constants
    enum
    {
        BLOCK_THREADS           = BlockHistogramSweepPolicyT::BLOCK_THREADS,

        PIXELS_PER_THREAD       = BlockHistogramSweepPolicyT::PIXELS_PER_THREAD,
        TILE_PIXELS             = PIXELS_PER_THREAD * BLOCK_THREADS,

        SAMPLES_PER_THREAD      = PIXELS_PER_THREAD * NUM_CHANNELS,
        TILE_SAMPLES            = SAMPLES_PER_THREAD * BLOCK_THREADS,

        USE_SHARED_MEM          = ((PRIVATIZED_SMEM_BINS > 0) && (PTX_ARCH >= 120)),
    };

    /// Input iterator wrapper type (for applying cache modifier)
    typedef typename If<IsPointer<SampleIteratorT>::VALUE,
            CacheModifiedInputIterator<BlockHistogramSweepPolicyT::LOAD_MODIFIER, SampleT, OffsetT>,     // Wrap the native input pointer with CacheModifiedInputIterator
            SampleIteratorT>::Type                                                                       // Directly use the supplied input iterator type
        WrappedSampleIteratorT;

    /// Pixel input iterator type (for applying cache modifier)
    typedef CacheModifiedInputIterator<BlockHistogramSweepPolicyT::LOAD_MODIFIER, PixelT, OffsetT>
        WrappedPixelIteratorT;

    /// Parameterized BlockLoad type for samples
    typedef BlockLoad<
            WrappedSampleIteratorT,
            BlockScanSweepPolicy::BLOCK_THREADS,
            BlockScanSweepPolicy::ITEMS_PER_THREAD,
            BlockScanSweepPolicy::LOAD_ALGORITHM>
        BlockLoadSampleT;

    /// Parameterized BlockLoad type for pixels
    typedef BlockLoad<
            WrappedPixelIteratorT,
            BlockScanSweepPolicy::BLOCK_THREADS,
            BlockScanSweepPolicy::ITEMS_PER_THREAD,
            BlockScanSweepPolicy::LOAD_ALGORITHM>
        BlockLoadPixelT;


    /// Shared memory type required by this thread block
    struct _TempStorage
    {
        CounterT histograms[NUM_ACTIVE_CHANNELS][PRIVATIZED_SMEM_BINS + 1];     // Smem needed for block-privatized smem histogram (with 1 word of padding)

        union
        {
            typename BlockLoadSampleT::TempStorage sample_load;                 // Smem needed for loading a tile of samples
            typename BlockLoadPixelT::TempStorage pixel_load;                   // Smem needed for loading a tile of pixels
        };
    };


    /// Temporary storage type (unionable)
    struct TempStorage : Uninitialized<_TempStorage> {};


    //---------------------------------------------------------------------
    // Per-thread fields
    //---------------------------------------------------------------------

    /// Reference to temp_storage
    _TempStorage &temp_storage;

    /// Reference to final output histograms (gmem)
    CounterT* (&d_out_histograms)[NUM_ACTIVE_CHANNELS];

    /// Reference to privatized output histograms (gmem)
    CounterT* (&d_private_histograms)[NUM_ACTIVE_CHANNELS];

    /// The transform operator for determining bin-ids from samples, one for each channel
    BinDecodeOpT (&decode_op)[NUM_ACTIVE_CHANNELS];

    /// Cache-modified sample input data
    WrappedSampleIteratorT d_wrapped_samples;

    /// Cache-modified pixel input data
    WrappedPixelIteratorT d_wrapped_pixels;

    /// The number of bins for each channel
    int (&num_bins)[NUM_ACTIVE_CHANNELS];

    //---------------------------------------------------------------------
    // Utility
    //---------------------------------------------------------------------

    // Whether or not the input is quad-aligned (specialized for wrapped pointer types)
    template <
        CacheLoadModifier   MODIFIER,
        typename            ValueType,
        typename            OffsetT>
    __device__ __forceinline__ bool IsAligned(CacheModifiedInputIterator<MODIFIER, ValueType, OffsetT> d_wrapped_samples)
    {
        return (Traits<SampleT>::PRIMITIVE) &&
            (int(d_wrapped_samples.ptr) & (int(sizeof(SampleT) * 4) - 1) == 0);
    }


    // Whether or not the input is quad-aligned (specialized for iterators)
    template <typename Iterator>
    static __device__ __forceinline__ bool IsAligned(Iterator d_samples)
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
            int block_histo_offset = 0;

            #pragma unroll
            for (; block_histo_offset + BLOCK_THREADS <= PRIVATIZED_SMEM_BINS; block_histo_offset += BLOCK_THREADS)
            {
                temp_storage.histograms[CHANNEL][block_histo_offset + threadIdx.x] = 0;
            }

            // Finish up with guarded initialization if necessary
            if ((PRIVATIZED_SMEM_BINS % BLOCK_THREADS != 0) && (block_histo_offset + threadIdx.x < PRIVATIZED_SMEM_BINS))
            {
                temp_storage.histograms[CHANNEL][block_histo_offset + threadIdx.x] = 0;
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
//            OffsetT block_histo_offset = ((blockIdx.y * gridDim.x) + blockIdx.x) * num_bins[CHANNEL];
            for (int bin = threadIdx.x; bin < num_bins[CHANNEL]; bin += BLOCK_THREADS)
            {
//                d_out_histograms[CHANNEL][block_histo_offset + bin] = temp_storage.histograms[CHANNEL][bin];
                atomicAdd(&d_out_histograms[CHANNEL][bin], temp_storage.histograms[CHANNEL][bin]);
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
     * Accumulate pixels
     */
    __device__ __forceinline__ void AccumulatePixels(
        SampleT             samples[PIXELS_PER_THREAD][NUM_CHANNELS],
        bool                is_valid[PIXELS_PER_THREAD],
        CounterT*           histograms[NUM_ACTIVE_CHANNELS])
    {

        #pragma unroll
        for (int PIXEL = 0; PIXEL < PIXELS_PER_THREAD; ++PIXEL)
        {
            #pragma unroll
            for (int CHANNEL = 0; CHANNEL < NUM_ACTIVE_CHANNELS; ++CHANNEL)
            {
                // Bin next pixel
                int bin;
                decode_op[CHANNEL].BinSelect<BlockHistogramSweepPolicyT::LOAD_MODIFIER>(
                    samples[PIXEL][CHANNEL],
                    bin,
                    is_valid[PIXEL]);

                if (bin >= 0)
                    atomicAdd(histograms[CHANNEL] + bin, 1);
            }
        }

/*
        #pragma unroll
        for (int CHANNEL = 0; CHANNEL < NUM_ACTIVE_CHANNELS; ++CHANNEL)
        {
            // Bin pixels
            int bins[PIXELS_PER_THREAD];

            // Bin current pixel
            decode_op[CHANNEL].BinSelect<BlockHistogramSweepPolicyT::LOAD_MODIFIER>(
                samples[0][CHANNEL],
                bins[0],
                is_valid[0]);

            int accumulator = 1;

            #pragma unroll
            for (int PIXEL = 0; PIXEL < PIXELS_PER_THREAD - 1; ++PIXEL)
            {
                // Bin next pixel
                decode_op[CHANNEL].BinSelect<BlockHistogramSweepPolicyT::LOAD_MODIFIER>(
                    samples[PIXEL + 1][CHANNEL],
                    bins[PIXEL + 1],
                    is_valid[PIXEL + 1]);

                if (bins[PIXEL] == bins[PIXEL + 1])
                {
                    accumulator++;
                }
                else
                {
                    if (bins[PIXEL] >= 0)
                        atomicAdd(histograms[CHANNEL] + bins[PIXEL], accumulator);
                    accumulator = 1;
                }
            }

            // Last pixel
            if (bins[PIXELS_PER_THREAD - 1] >= 0)
                atomicAdd(histograms[CHANNEL] + bins[PIXELS_PER_THREAD - 1], accumulator);
        }
*/
    }


    /**
     * Accumulate pixel, specialized for gmem privatized histogram
     */
    __device__ __forceinline__ void AccumulatePixels(
        SampleT             samples[PIXELS_PER_THREAD][NUM_CHANNELS],
        bool                is_valid[PIXELS_PER_THREAD],
        Int2Type<false>     use_shared_mem)
    {
        AccumulatePixels(samples, is_valid, d_out_histograms);
    }


    /**
     * Accumulate pixel, specialized for smem privatized histogram
     */
    __device__ __forceinline__ void AccumulatePixels(
        SampleT             samples[PIXELS_PER_THREAD][NUM_CHANNELS],
        bool                is_valid[PIXELS_PER_THREAD],
        Int2Type<true>      use_shared_mem)
    {
        CounterT*           histograms[NUM_ACTIVE_CHANNELS];
        for (int CHANNEL = 0; CHANNEL < NUM_ACTIVE_CHANNELS; ++CHANNEL)
            histograms[CHANNEL] = temp_storage.histograms[CHANNEL];

        AccumulatePixels(samples, is_valid, histograms);
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
            __syncthreads();

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
/*
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
                 __syncthreads();

            #pragma unroll
            for (int PIXEL = 0; PIXEL < PIXELS_PER_THREAD; ++PIXEL)
                is_valid[PIXEL] = true;

            // Accumulate samples
            AccumulatePixels(samples, is_valid, Int2Type<USE_SHARED_MEM>());
        }
        else
*/
        {
            // Multi-channel tile
            SampleT     samples[PIXELS_PER_THREAD][NUM_CHANNELS];
            bool        is_valid[PIXELS_PER_THREAD];

            // Alias items as an array of VectorT and load it in striped fashion
            VectorT *vec_items = reinterpret_cast<VectorT*>(samples);

            // Vector input iterator wrapper at starting pixel
/*
            // Read striped pixels
            SampleT *d_samples_unqualified = const_cast<SampleT*>(d_samples) + block_row_offset + (threadIdx.x * VECTOR_LOAD_LENGTH);
            VectorT *ptr = reinterpret_cast<VectorT*>(d_samples_unqualified);
            InputIteratorVectorT d_vec_in(ptr);

            #pragma unroll
            for (int PIXEL = 0; PIXEL < PIXELS_PER_THREAD; ++PIXEL)
            {
                int pixel_idx = (PIXEL * BLOCK_THREADS) + threadIdx.x;
                if ((is_valid[PIXEL] = (IS_FULL_TILE || (pixel_idx < valid_pixels))))
                {
                    vec_items[PIXEL] = d_vec_in[PIXEL * BLOCK_THREADS];
                }
            }
*/

            // Read blocked pixels
            SampleT *d_samples_unqualified = const_cast<SampleT*>(d_samples) + block_row_offset;
            VectorT *ptr = reinterpret_cast<VectorT*>(d_samples_unqualified);
            InputIteratorVectorT d_vec_in(ptr);

            #pragma unroll
            for (int PIXEL = 0; PIXEL < PIXELS_PER_THREAD; ++PIXEL)
            {
                int pixel_idx = (threadIdx.x * PIXELS_PER_THREAD) + PIXEL;
                if ((is_valid[PIXEL] = (IS_FULL_TILE || (pixel_idx < valid_pixels))))
                {
                    vec_items[PIXEL] = d_vec_in[pixel_idx];
                }
            }

            if (LOAD_FENCE)
                 __syncthreads();

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
        BinDecodeOpT  (&decode_op)[NUM_ACTIVE_CHANNELS],           ///< Transform operators for determining bin-ids from samples, one for each channel
        int                 (&num_bins)[NUM_ACTIVE_CHANNELS])               ///< The number of boundaries (levels) for delineating histogram samples, one for each channel
    :
        temp_storage(temp_storage.Alias()),
        d_samples(d_samples),
        d_wrapped_samples(d_samples),
        d_out_histograms(d_out_histograms),
        decode_op(decode_op),
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

        if (IsAligned(d_wrapped_samples + row_offset))
        {
            // Read pixels
        }
        else
        {
            // Read samples
        }


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

