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
    typename    PrivatizedDecodeOpT,            ///< The transform operator type for determining privatized counter indices from samples, one for each channel
    typename    OutputDecodeOpT,                ///< The transform operator type for determining output bin-ids from privatized counter indices, one for each channel
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

        RLE_COMPRESS            = BlockHistogramSweepPolicyT::RLE_COMPRESS,
        USE_SHARED_MEM          = ((PRIVATIZED_SMEM_BINS > 0) && (PTX_ARCH >= 120)),

    };

    /// Cache load modifier for reading input elements
    static const CacheLoadModifier LOAD_MODIFIER = BlockHistogramSweepPolicyT::LOAD_MODIFIER;


    /// Input iterator wrapper type (for applying cache modifier)
    typedef typename If<IsPointer<SampleIteratorT>::VALUE,
            CacheModifiedInputIterator<LOAD_MODIFIER, SampleT, OffsetT>,     // Wrap the native input pointer with CacheModifiedInputIterator
            SampleIteratorT>::Type                                           // Directly use the supplied input iterator type
        WrappedSampleIteratorT;

    /// Pixel input iterator type (for applying cache modifier)
    typedef CacheModifiedInputIterator<LOAD_MODIFIER, PixelT, OffsetT>
        WrappedPixelIteratorT;

    /// Parameterized BlockLoad type for samples
    typedef BlockLoad<
            WrappedSampleIteratorT,
            BLOCK_THREADS,
            SAMPLES_PER_THREAD,
            BlockHistogramSweepPolicyT::LOAD_ALGORITHM>
        BlockLoadSampleT;

    /// Parameterized BlockLoad type for pixels
    typedef BlockLoad<
            WrappedPixelIteratorT,
            BLOCK_THREADS,
            PIXELS_PER_THREAD,
            BlockHistogramSweepPolicyT::LOAD_ALGORITHM>
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

    /// Sample input iterator (with cache modifier applied, if possible)
    WrappedSampleIteratorT d_wrapped_samples;

    /// The number of output bins for each channel
    int (&num_output_bins)[NUM_ACTIVE_CHANNELS];

    /// The number of privatized bins for each channel
    int (&num_privatized_bins)[NUM_ACTIVE_CHANNELS];

    /// Reference to final output histograms (gmem)
    CounterT* (&d_output_histograms)[NUM_ACTIVE_CHANNELS];

    /// Reference to block-private temporary histograms (gmem)
    CounterT* (&d_privatized_histograms)[NUM_ACTIVE_CHANNELS];

    /// The transform operator for determining output bin-ids from privatized counter indices, one for each channel
    OutputDecodeOpT (&output_decode_op)[NUM_ACTIVE_CHANNELS];

    /// The transform operator for determining privatized counter indices from samples, one for each channel
    PrivatizedDecodeOpT (&privatized_decode_op)[NUM_ACTIVE_CHANNELS];


    //---------------------------------------------------------------------
    // Initialize privatized bin counters
    //---------------------------------------------------------------------

    // Initialize privatized bin counters
    __device__ __forceinline__ void InitBinCounters(CounterT* privatized_histograms[NUM_ACTIVE_CHANNELS])
    {
        // Initialize histogram bin counts to zeros
        #pragma unroll
        for (int CHANNEL = 0; CHANNEL < NUM_ACTIVE_CHANNELS; ++CHANNEL)
        {
            for (int privatized_bin = threadIdx.x; privatized_bin < num_privatized_bins[CHANNEL]; privatized_bin += BLOCK_THREADS)
            {
                privatized_histograms[CHANNEL][privatized_bin] = 0;
            }
        }

        // Barrier to make sure all threads are done updating counters
        __syncthreads();
    }


    // Initialize privatized bin counters.  Specialized for privatized shared-memory counters
    __device__ __forceinline__ void InitBinCounters(Int2Type<true> use_shared_mem)
    {
        CounterT* privatized_histograms[NUM_ACTIVE_CHANNELS];

        for (int CHANNEL = 0; CHANNEL < NUM_ACTIVE_CHANNELS; ++CHANNEL)
            privatized_histograms[CHANNEL] = temp_storage.histograms[CHANNEL];

        InitBinCounters(privatized_histograms);
    }


    // Initialize privatized bin counters.  Specialized for privatized global-memory counters
    __device__ __forceinline__ void InitBinCounters(Int2Type<false> use_shared_mem)
    {
        CounterT* privatized_histograms[NUM_ACTIVE_CHANNELS];

        for (int CHANNEL = 0; CHANNEL < NUM_ACTIVE_CHANNELS; ++CHANNEL)
            privatized_histograms[CHANNEL] = d_privatized_histograms[CHANNEL] + (blockIdx.x * num_privatized_bins[CHANNEL]);

        InitBinCounters(privatized_histograms);
    }


    //---------------------------------------------------------------------
    // Update final output histograms
    //---------------------------------------------------------------------

    // Update final output histograms from privatized histograms
    __device__ __forceinline__ void StoreOutput(CounterT* privatized_histograms[NUM_ACTIVE_CHANNELS])
    {
        // Barrier to make sure all threads are done updating counters
        __syncthreads();

        // Apply privatized bin counts to output bin counts
        #pragma unroll
        for (int CHANNEL = 0; CHANNEL < NUM_ACTIVE_CHANNELS; ++CHANNEL)
        {
            for (int privatized_bin = threadIdx.x; privatized_bin < num_privatized_bins[CHANNEL]; privatized_bin += BLOCK_THREADS)
            {
                CounterT count = privatized_histograms[CHANNEL][privatized_bin];

                // Determine output bin from privatized counter index
                int output_bin;
                output_decode_op[CHANNEL].BinSelect<LOAD_MODIFIER>((SampleT) privatized_bin, output_bin, true);

                if (output_bin > 0)
                    atomicAdd(&d_output_histograms[CHANNEL][output_bin], count);
            }
        }
    }


    // Update final output histograms from privatized histograms.  Specialized for privatized shared-memory counters
    __device__ __forceinline__ void StoreOutput(Int2Type<true> use_shared_mem)
    {
        CounterT* privatized_histograms[NUM_ACTIVE_CHANNELS];

        for (int CHANNEL = 0; CHANNEL < NUM_ACTIVE_CHANNELS; ++CHANNEL)
            privatized_histograms[CHANNEL] = temp_storage.histograms[CHANNEL];

        StoreOutput(privatized_histograms);
    }


    // Update final output histograms from privatized histograms.  Specialized for privatized global-memory counters
    __device__ __forceinline__ void StoreOutput(Int2Type<false> use_shared_mem)
    {
        CounterT* privatized_histograms[NUM_ACTIVE_CHANNELS];

        for (int CHANNEL = 0; CHANNEL < NUM_ACTIVE_CHANNELS; ++CHANNEL)
            privatized_histograms[CHANNEL] = d_privatized_histograms[CHANNEL] + (blockIdx.x * num_privatized_bins[CHANNEL]);

        StoreOutput(privatized_histograms);
    }


    //---------------------------------------------------------------------
    // Accumulate privatized histograms
    //---------------------------------------------------------------------

    // Accumulate pixels.  Specialized for RLE compression.
    __device__ __forceinline__ void AccumulatePixels(
        SampleT             samples[PIXELS_PER_THREAD][NUM_CHANNELS],
        bool                is_valid[PIXELS_PER_THREAD],
        CounterT*           privatized_histograms[NUM_ACTIVE_CHANNELS],
        Int2Type<true>      is_rle_compress)
    {
        #pragma unroll
        for (int CHANNEL = 0; CHANNEL < NUM_ACTIVE_CHANNELS; ++CHANNEL)
        {
            // Bin pixels
            int bins[PIXELS_PER_THREAD];

            // Bin current pixel
            privatized_decode_op[CHANNEL].BinSelect<LOAD_MODIFIER>(
                samples[0][CHANNEL],
                bins[0],
                is_valid[0]);

            CounterT accumulator = 1;

            #pragma unroll
            for (int PIXEL = 0; PIXEL < PIXELS_PER_THREAD - 1; ++PIXEL)
            {
                // Bin next pixel
                privatized_decode_op[CHANNEL].BinSelect<LOAD_MODIFIER>( samples[PIXEL + 1][CHANNEL], bins[PIXEL + 1], is_valid[PIXEL + 1]);

                if (bins[PIXEL] == bins[PIXEL + 1])
                {
                    accumulator++;
                }
                else
                {
                    if (bins[PIXEL] >= 0)
                        atomicAdd(privatized_histograms[CHANNEL] + bins[PIXEL], accumulator);
                    accumulator = 1;
                }
            }

            // Last pixel
            if (bins[PIXELS_PER_THREAD - 1] >= 0)
                atomicAdd(privatized_histograms[CHANNEL] + bins[PIXELS_PER_THREAD - 1], accumulator);
        }
    }


    // Accumulate pixels.  Specialized for individual accumulation of each pixel.
    __device__ __forceinline__ void AccumulatePixels(
        SampleT             samples[PIXELS_PER_THREAD][NUM_CHANNELS],
        bool                is_valid[PIXELS_PER_THREAD],
        CounterT*           privatized_histograms[NUM_ACTIVE_CHANNELS],
        Int2Type<false>     is_rle_compress)
    {
        #pragma unroll
        for (int PIXEL = 0; PIXEL < PIXELS_PER_THREAD; ++PIXEL)
        {
            #pragma unroll
            for (int CHANNEL = 0; CHANNEL < NUM_ACTIVE_CHANNELS; ++CHANNEL)
            {
                int bin;
                privatized_decode_op[CHANNEL].BinSelect<LOAD_MODIFIER>(samples[PIXEL][CHANNEL], bin, is_valid[PIXEL]);
                if (bin >= 0)
                    atomicAdd(privatized_histograms[CHANNEL] + bin, 1);
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
        CounterT* privatized_histograms[NUM_ACTIVE_CHANNELS];

        for (int CHANNEL = 0; CHANNEL < NUM_ACTIVE_CHANNELS; ++CHANNEL)
            privatized_histograms[CHANNEL] = temp_storage.histograms[CHANNEL];

        AccumulatePixels(samples, is_valid, privatized_histograms, Int2Type<RLE_COMPRESS>());
    }


    /**
     * Accumulate pixel, specialized for gmem privatized histogram
     */
    __device__ __forceinline__ void AccumulatePixels(
        SampleT             samples[PIXELS_PER_THREAD][NUM_CHANNELS],
        bool                is_valid[PIXELS_PER_THREAD],
        Int2Type<false>     use_shared_mem)
    {
        CounterT* privatized_histograms[NUM_ACTIVE_CHANNELS];

        for (int CHANNEL = 0; CHANNEL < NUM_ACTIVE_CHANNELS; ++CHANNEL)
            privatized_histograms[CHANNEL] = d_privatized_histograms[CHANNEL] + (blockIdx.x * num_privatized_bins[CHANNEL]);

        AccumulatePixels(samples, is_valid, privatized_histograms, Int2Type<RLE_COMPRESS>());
    }



    //---------------------------------------------------------------------
    // Tile processing
    //---------------------------------------------------------------------


    /**
     * Load a full tile of data samples.
     */
    template <
        bool IS_ALIGNED,        // Whether the tile offset is aligned (quad-aligned for single-channel, pixel-aligned for multi-channel)
        bool IS_FULL_TILE>      // Whether the tile is full
    __device__ __forceinline__ void ConsumeTile(
        OffsetT         block_offset,
        int             valid_samples,
        SampleT*        d_native_samples)
    {
        SampleT     samples[PIXELS_PER_THREAD][NUM_CHANNELS];
        bool        is_valid[PIXELS_PER_THREAD];

        typedef PixelT AliasedSamples[SAMPLES_PER_THREAD];
        typedef PixelT AliasedPixels[PIXELS_PER_THREAD];

        WrappedPixelIteratorT d_wrapped_pixels((PixelT*) (d_native_samples + block_offset));

        int valid_pixels = valid_samples / NUM_CHANNELS;

        if (IS_FULL_TILE)
        {
            // Full tile
//            if (IS_ALIGNED)
            {
                // Load using a wrapped pixel iterator
                BlockLoadPixelT(temp_storage.pixel_load).Load(
                    d_wrapped_pixels,
                    reinterpret_cast<AliasedPixels&>(samples));
            }
/*
            else
            {
                // Load using a wrapped sample iterator
                BlockLoadSampleT(temp_storage.sample_load).Load(
                    d_wrapped_samples + block_offset,
                    reinterpret_cast<AliasedSamples&>(samples));
            }
*/
        }
        else
        {
            // Partial tile
//            if (IS_ALIGNED)
            {
                // Load using a wrapped pixel iterator
                BlockLoadPixelT(temp_storage.pixel_load).Load(
                    d_wrapped_pixels,
                    reinterpret_cast<AliasedPixels&>(samples),
                    valid_pixels);
            }
/*
            else
            {
                // Load using a wrapped sample iterator
                BlockLoadSampleT(temp_storage.pixel_load).Load(
                    d_wrapped_samples + block_offset,
                    reinterpret_cast<AliasedSamples&>(samples),
                    valid_samples);
            }
*/
        }

        // Set valid flags
        #pragma unroll
        for (int PIXEL = 0; PIXEL < PIXELS_PER_THREAD; ++PIXEL)
            is_valid[PIXEL] = IS_FULL_TILE || ((threadIdx.x * PIXELS_PER_THREAD) + PIXEL < valid_pixels);

        // Accumulate samples
        AccumulatePixels(samples, is_valid, Int2Type<USE_SHARED_MEM>());
    }


    // Consume row tiles (striped across thread blocks)
    template <bool IS_ALIGNED>
    __device__ __forceinline__ void ConsumeRowTiles(
        OffsetT  row_offset,
        OffsetT  row_end,
        SampleT* d_native_samples)
    {
        for (
            OffsetT block_offset = row_offset + (blockIdx.x * TILE_SAMPLES);
            block_offset < row_end;
            block_offset += gridDim.x * TILE_SAMPLES)
        {
            int valid_samples = row_end - block_offset;

            if (valid_samples >= TILE_SAMPLES)
                ConsumeTile<IS_ALIGNED, true>(block_offset, TILE_SAMPLES, d_native_samples);
            else
                ConsumeTile<IS_ALIGNED, false>(block_offset, valid_samples, d_native_samples);
        }
    }


    // Consume rows
    template <bool IS_ALIGNED>
    __device__ __forceinline__ void ConsumeRows(
        OffsetT     num_row_pixels,
        OffsetT     num_rows,
        OffsetT     row_stride_samples,
        SampleT*    d_native_samples)
    {
/* mooch
        for (int row = blockIdx.y; row < num_rows; row += gridDim.y)
        {
            OffsetT row_offset     = row * row_stride_samples;
            OffsetT row_end        = row_offset + (num_row_pixels * NUM_CHANNELS);
            ConsumeRowTiles<IS_ALIGNED>(row_offset, row_end, d_native_samples);
        }
*/

        ConsumeRowTiles<IS_ALIGNED>(0, row_stride_samples * NUM_CHANNELS, d_native_samples);
    }


    //---------------------------------------------------------------------
    // Parameter extraction
    //---------------------------------------------------------------------

    // Return a native pixel pointer (specialized for CacheModifiedInputIterator types)
    template <
        CacheLoadModifier   MODIFIER,
        typename            ValueT,
        typename            OffsetT>
    __device__ __forceinline__ SampleT* NativePointer(CacheModifiedInputIterator<MODIFIER, ValueT, OffsetT> itr)
    {
        return itr.ptr;
    }

    // Return a native pixel pointer (specialized for other types)
    template <typename IteratorT>
    __device__ __forceinline__ SampleT* NativePointer(IteratorT itr)
    {
        return NULL;
    }



    //---------------------------------------------------------------------
    // Interface
    //---------------------------------------------------------------------

    /**
     * Constructor
     */
    __device__ __forceinline__ BlockHistogramSweep(
        TempStorage         &temp_storage,                                      ///< Reference to temp_storage
        SampleIteratorT    	d_samples,                                          ///< Input data to reduce
        int                 (&num_output_bins)[NUM_ACTIVE_CHANNELS],            ///< The number bins per final output histogram
        int                 (&num_privatized_bins)[NUM_ACTIVE_CHANNELS],        ///< The number bins per privatized histogram
        CounterT*           (&d_output_histograms)[NUM_ACTIVE_CHANNELS],        ///< Reference to final output histograms
        CounterT*           (&d_privatized_histograms)[NUM_ACTIVE_CHANNELS],    ///< Reference to privatized histograms
        OutputDecodeOpT     (&output_decode_op)[NUM_ACTIVE_CHANNELS],           ///< The transform operator for determining output bin-ids from privatized counter indices, one for each channel
        PrivatizedDecodeOpT (&privatized_decode_op)[NUM_ACTIVE_CHANNELS])       ///< The transform operator for determining privatized counter indices from samples, one for each channel
    :
        temp_storage(temp_storage.Alias()),
        d_wrapped_samples(d_samples),
        num_output_bins(num_output_bins),
        num_privatized_bins(num_privatized_bins),
        d_output_histograms(d_output_histograms),
        d_privatized_histograms(d_privatized_histograms),
        privatized_decode_op(privatized_decode_op),
        output_decode_op(output_decode_op)
    {}


    /**
     * Consume image
     */
    __device__ __forceinline__ void ConsumeTiles(
        OffsetT                                                 num_row_pixels,                     ///< The number of multi-channel pixels per row in the region of interest
        OffsetT                                                 num_rows,                           ///< The number of rows in the region of interest
        OffsetT                                                 row_stride_samples)                 ///< The number of samples between starts of consecutive rows in the region of interest
    {
        // Get native pointer for input samples (possibly NULL if unavailable)
        SampleT* d_native_samples = NativePointer(d_wrapped_samples);

        // Whether all row starting offsets are quad-aligned (in single-channel)
//        bool quad_aligned_rows   = (NUM_CHANNELS == 1) && ((OffsetT(d_native_samples) | row_stride_samples) & ((int(sizeof(SampleT)) * 4) - 1) == 0);

        // Whether all row starting offsets are pixel-aligned (in multi-channel)
//        bool pixel_aligned_rows  = (NUM_CHANNELS > 1)  && ((OffsetT(d_native_samples) | row_stride_samples) & (AlignBytes<PixelT>::ALIGN_BYTES - 1) == 0);         // pixel-aligned

        // Whether rows are aligned and can be vectorized
/*
Mooch
        if ((d_native_samples != NULL) && (quad_aligned_rows || pixel_aligned_rows))
            ConsumeRows<true>(num_row_pixels, num_rows, row_stride_samples, d_native_samples);
        else
            ConsumeRows<false>(num_row_pixels, num_rows, row_stride_samples, d_native_samples);
*/

        ConsumeRows<true>(num_row_pixels, num_rows, row_stride_samples, d_native_samples);

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

