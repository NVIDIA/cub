
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
 * cub::DeviceHistogram provides device-wide parallel operations for constructing histogram(s) from a sequence of samples data residing within global memory.
 */

#pragma once

#include <stdio.h>
#include <iterator>

#include "../../block_sweep/block_histogram_sweep.cuh"
#include "../device_radix_sort.cuh"
#include "../../iterator/tex_ref_input_iterator.cuh"
#include "../../util_debug.cuh"
#include "../../util_device.cuh"
#include "../../util_namespace.cuh"
#include "../../thread/thread_search.cuh"

/// Optional outer namespace(s)
CUB_NS_PREFIX

/// CUB namespace
namespace cub {



/******************************************************************************
 * Histogram kernel entry points
 *****************************************************************************/


/**
 * Histogram privatized sweep kernel entry point (multi-block).  Computes privatized histograms, one per thread block.
 */
template <
    typename                                            BlockHistogramSweepPolicyT, ///< Parameterized BlockHistogramSweepPolicy tuning policy type
    int                                                 MAX_PRIVATIZED_BINS,        ///< Maximum number of histogram bins per channel (e.g., up to 256)
    int                                                 NUM_CHANNELS,               ///< Number of channels interleaved in the input data (may be greater than the number of channels being actively histogrammed)
    int                                                 NUM_ACTIVE_CHANNELS,        ///< Number of channels actively being histogrammed
    typename                                            SampleIteratorT,             ///< The input iterator type. \iterator.
    typename                                            CounterT,                   ///< Integer type for counting sample occurrences per histogram bin
    typename                                            SampleTransformOpT,         ///< Transform operator type for determining bin-ids from samples for each channel
    typename                                            OffsetT>                    ///< Signed integer type for global offsets
__launch_bounds__ (int(BlockHistogramSweepPolicyT::BLOCK_THREADS))
__global__ void DeviceHistogramSweepKernel(
    SampleIteratorT                                         d_samples,              ///< [in] Array of sample data. The samples from different channels are assumed to be interleaved (e.g., an array of 32b pixels where each pixel consists of four RGBA 8b samples).
    ArrayWrapper<CounterT*, NUM_ACTIVE_CHANNELS>            d_temp_histo_wrapper,   ///< [out] Histogram counter data having logical dimensions <tt>CounterT[NUM_ACTIVE_CHANNELS][gridDim.x][MAX_PRIVATIZED_BINS]</tt>
    ArrayWrapper<SampleTransformOpT, NUM_ACTIVE_CHANNELS>   transform_op_wrapper,   ///< [in] Transform operators for determining bin-ids from samples, one for each channel
    OffsetT                                                 num_row_pixels,         ///< [in] The number of multi-channel pixels per row in the region of interest
    OffsetT                                                 num_rows,               ///< [in] The number of rows in the region of interest
    OffsetT                                                 row_pixel_stride)       ///< [in] The number of multi-channel pixels between starts of consecutive rows in the region of interest
{
    // Thread block type for compositing input tiles
    typedef BlockHistogramSweep<BlockHistogramSweepPolicyT, MAX_PRIVATIZED_BINS, NUM_CHANNELS, NUM_ACTIVE_CHANNELS, SampleIteratorT, CounterT, SampleTransformOpT, OffsetT> BlockHistogramSweepT;

    // Shared memory for BlockHistogramSweep
    __shared__ typename BlockHistogramSweepT::TempStorage temp_storage;

    // Consume input tiles
    for (OffsetT row = blockIdx.y; row < num_rows; row += gridDim.y)
    {
        OffsetT block_offset_in     = row * row_pixel_stride * NUM_CHANNELS;
        OffsetT block_end           = block_offset_in + num_row_pixels;

        BlockHistogramSweepT(temp_storage, d_samples, d_temp_histo_wrapper.array, transform_op_wrapper.array).ConsumeRange(
            block_offset_in, block_end);
    }
}


/**
 * Histogram aggregation kernel entry point (one block per channel).  Aggregates privatized threadblock histograms from a previous multi-block histogram pass.
 */
template <
    int                                             MAX_PRIVATIZED_BINS,        ///< Maximum number of histogram bins per channel
    int                                             NUM_ACTIVE_CHANNELS,        ///< Number of channels actively being histogrammed
    typename                                        CounterT>                   ///< Integer type for counting sample occurrences per histogram bin
__launch_bounds__ (MAX_PRIVATIZED_BINS)
__global__ void DeviceHistogramAggregateKernel(
    ArrayWrapper<int, NUM_ACTIVE_CHANNELS>          num_bins_wrapper,           ///< [in] Number of histogram bins per channel
    ArrayWrapper<CounterT*, NUM_ACTIVE_CHANNELS>    d_temp_histo_wrapper,       ///< [out] Histogram counter data having logical dimensions <tt>CounterT[NUM_ACTIVE_CHANNELS][gridDim.x][MAX_PRIVATIZED_BINS]</tt>
    ArrayWrapper<CounterT*, NUM_ACTIVE_CHANNELS>    d_histo_wrapper,            ///< [out] Histogram counter data having logical dimensions <tt>CounterT[NUM_ACTIVE_CHANNELS][num_bins.array[CHANNEL]]</tt>
    int                                             num_threadblocks)           ///< [in] Number of threadblock histograms per channel in \p d_block_histograms
{
    // Accumulate threadblock-histograms from the channel
    int         channel             = blockIdx.x;
    int         bin                 = threadIdx.x;
    CounterT    bin_aggregate       = 0;

#if CUB_PTX_ARCH >= 200
    #pragma unroll 32
#endif
    for (
        int block_offset = 0;
        block_offset < num_threadblocks * MAX_PRIVATIZED_BINS;
        block_offset += MAX_PRIVATIZED_BINS)
    {
        bin_aggregate += d_temp_histo_wrapper.array[channel][block_offset + bin];
    }

    // Output
    if (bin < num_bins_wrapper.array[channel])
        d_histo_wrapper.array[channel][bin] = bin_aggregate;
}




/******************************************************************************
 * Dispatch
 ******************************************************************************/

/**
 * Utility class for dispatching the appropriately-tuned kernels for DeviceHistogram
 */
template <
    int         NUM_CHANNELS,               ///< Number of channels interleaved in the input data (may be greater than the number of channels being actively histogrammed)
    int         NUM_ACTIVE_CHANNELS,        ///< Number of channels actively being histogrammed
    typename    SampleT,                    ///< The data sample type
    typename    CounterT,                   ///< Integer type for counting sample occurrences per histogram bin
    typename    LevelT,                     ///< Type for specifying bin level boundaries
    typename    OffsetT>                    ///< Signed integer type for global offsets
struct DeviceHistogramDispatch
{
    /******************************************************************************
     * Types and constants
     ******************************************************************************/

    enum
    {
        // Maximum number of bins for which we will use a privatized strategy
        MAX_PRIVATIZED_BINS = 256
    };


    /******************************************************************************
     * Transform functors for converting samples to bin-ids
     ******************************************************************************/

    // Searches for bin given a list of bin-boundary levels
    template <typename LevelIteratorT>
    struct SearchTransform
    {
        LevelIteratorT  d_levels;      // Pointer to levels array
        int             num_levels;    // Number of levels in array

        // Functor for converting samples to bin-ids
        __device__ __forceinline__ int operator()(SampleT sample)
        {
            int retval = ((int) UpperBound(d_levels, num_levels, (LevelT) sample)) - 1;
            if (retval < 0)
            {
                // Sample out of range
                return num_levels;
            }
            return retval;
        }

        // Return inclusive lower level for the given bin
        __device__ __forceinline__ LevelT LowerInclusive(int bin)
        {
            return d_levels[bin];
        }

        // Return exclusive upper level for the given bin
        __device__ __forceinline__ LevelT UpperExclusive(int bin)
        {
            return d_levels[bin + 1];
        }
    };


    // Scales samples to evenly-spaced bins
    struct ScaleTransform
    {
        int    num_levels;  // Number of levels in array
        LevelT max;         // Max sample level (exclusive)
        LevelT min;         // Min sample level (inclusive)
        LevelT scale;       // Bin scaling factor

        // Initializer
        __device__ __forceinline__ void Init(
            int    num_levels,  // Number of levels in array
            LevelT max,         // Max sample level (exclusive)
            LevelT min,         // Min sample level (inclusive)
            LevelT scale)       // Bin scaling factor
        {
            this->num_levels = num_levels;
            this->max = max;
            this->min = min;
            this->scale = scale;
        }

        // Functor for converting samples to bin-ids
        __device__ __forceinline__ int operator()(SampleT sample)
        {
            if ((sample < min) || (sample <= max))
            {
                // Sample out of range
                return num_levels;
            }

            return (int) ((((LevelT) sample) - min) / scale);
        }

        // Return inclusive lower level for the given bin
        __device__ __forceinline__ LevelT LowerInclusive(int bin)
        {
            return min + (scale * bin);
        }

        // Return exclusive upper level for the given bin
        __device__ __forceinline__ LevelT UpperExclusive(int bin)
        {
            return LowerInclusive(bin) + scale;
        }
    };


    // Scale-free sample transform used for the the common-case of 8b samples and 256 evenly-spaced bins (specialized for int8)
    template <typename SampleT, int DUMMY = 0>
    struct ScaleFreeTransform
    {
        // Functor for converting samples to bin-ids
        int operator()(SampleT sample)
        {
            return ((int) sample) + 128;
        }
    };

    // Scale-free sample transform used for the the common-case of 8b samples and 256 evenly-spaced bins (specialized for unsigend int8)
    template <int DUMMY>
    struct ScaleFreeTransform<unsigned char, DUMMY>
    {
        // Functor for converting samples to bin-ids
        int operator()(unsigned char sample)
        {
            return (int) sample;
        }
    };



    /******************************************************************************
     * Tuning policies
     ******************************************************************************/

    /// SM35
    struct Policy350
    {
        // HistogramSweepPolicy
        typedef BlockHistogramSweepPolicy<
                256,
                (30 / NUM_ACTIVE_CHANNELS)>
            HistogramSweepPolicy;

        // HistogramUnzipPolicy
        typedef BlockUnzipChannelsSweepPolicy<
                256,
                (30 / NUM_ACTIVE_CHANNELS)>
            HistogramUnzipPolicy;
    };

    /// SM30
    struct Policy300
    {
        // HistogramSweepPolicy
        typedef BlockHistogramSweepPolicy<
                128,
                (22 / NUM_ACTIVE_CHANNELS)>
            HistogramSweepPolicy;

        // HistogramUnzipPolicy
        typedef BlockUnzipChannelsSweepPolicy<
                128,
                (22 / NUM_ACTIVE_CHANNELS)>
            HistogramUnzipPolicy;
    };

    /// SM20
    struct Policy200
    {
        // HistogramSweepPolicy
        typedef BlockHistogramSweepPolicy<
                128,
                (23 / NUM_ACTIVE_CHANNELS)>
            HistogramSweepPolicy;

        // HistogramUnzipPolicy
        typedef BlockUnzipChannelsSweepPolicy<
                128,
                (23 / NUM_ACTIVE_CHANNELS)>
            HistogramUnzipPolicy;
    };

    /// SM10
    struct Policy100
    {
        // HistogramSweepPolicy
        typedef BlockHistogramSweepPolicy<
                128,
                7>
            HistogramSweepPolicy;

        // HistogramUnzipPolicy
        typedef BlockUnzipChannelsSweepPolicy<
                128,
                7>
            HistogramUnzipPolicy;
    };


    /******************************************************************************
     * Tuning policies of current PTX compiler pass
     ******************************************************************************/

#if (CUB_PTX_ARCH >= 350)
    typedef Policy350 PtxPolicy;

#elif (CUB_PTX_ARCH >= 300)
    typedef Policy300 PtxPolicy;

#elif (CUB_PTX_ARCH >= 200)
    typedef Policy200 PtxPolicy;

#else
    typedef Policy100 PtxPolicy;

#endif

    // "Opaque" policies (whose parameterizations aren't reflected in the type signature)
    struct PtxHistogramSweepPolicy : PtxPolicy::HistogramSweepPolicy {};
    struct PtxHistogramUnzipPolicy : PtxPolicy::HistogramUnzipPolicy {};


    /******************************************************************************
     * Utilities
     ******************************************************************************/


    /**
     * Initialize kernel dispatch configurations with the policies corresponding to the PTX assembly we will use
     */
    template <typename KernelConfig>
    CUB_RUNTIME_FUNCTION __forceinline__
    static void InitConfigs(
        int             ptx_version,
        KernelConfig    &histogram_sweep_config,
        KernelConfig    &histogram_unzip_config)
    {
    #if (CUB_PTX_ARCH > 0)

        // We're on the device, so initialize the kernel dispatch configurations with the current PTX policy
        histogram_sweep_config.template Init<PtxHistogramSweepPolicy>();
        histogram_unzip_config.template Init<PtxHistogramUnzipPolicy>();

    #else

        // We're on the host, so lookup and initialize the kernel dispatch configurations with the policies that match the device's PTX version
        if (ptx_version >= 350)
        {
            histogram_sweep_config.template Init<typename Policy350::HistogramSweepPolicy>();
            histogram_unzip_config.template Init<typename Policy350::HistogramUnzipPolicy>();
        }
        else if (ptx_version >= 300)
        {
            histogram_sweep_config.template Init<typename Policy300::HistogramSweepPolicy>();
            histogram_unzip_config.template Init<typename Policy300::HistogramUnzipPolicy>();
        }
        else if (ptx_version >= 200)
        {
            histogram_sweep_config.template Init<typename Policy200::HistogramSweepPolicy>();
            histogram_unzip_config.template Init<typename Policy200::HistogramUnzipPolicy>();
        }
        else
        {
            histogram_sweep_config.template Init<typename Policy100::HistogramSweepPolicy>();
            histogram_unzip_config.template Init<typename Policy100::HistogramUnzipPolicy>();
        }

    #endif
    }


    /**
     * Kernel kernel dispatch configuration
     */
    struct KernelConfig
    {
        int                             block_threads;
        int                             pixels_per_thread;

        template <typename BlockPolicy>
        CUB_RUNTIME_FUNCTION __forceinline__
        void Init()
        {
            block_threads               = BlockPolicy::BLOCK_THREADS;
            pixels_per_thread           = BlockPolicy::PIXELS_PER_THREAD;
        }

        CUB_RUNTIME_FUNCTION __forceinline__
        void Print()
        {
            printf("%d, %d", block_threads, pixels_per_thread);
        }

    };






    /******************************************************************************
     * Dispatch entrypoints
     ******************************************************************************/


    /**
     * Privatization-based dispatch routine
     */
    template <
        typename                            SampleIteratorT,
        typename                            SampleTransformOpT,                     ///< Transform operator type for determining bin-ids from samples for each channel
        typename                            DeviceHistogramSweepKernelT>            ///< Function type of cub::DeviceHistogramSweepKernel
    CUB_RUNTIME_FUNCTION __forceinline__
    static cudaError_t PrivatizedDispatch(
        void                                *d_temp_storage,                        ///< [in] %Device allocation of temporary storage.  When NULL, the required allocation size is written to \p temp_storage_bytes and no work is done.
        size_t                              &temp_storage_bytes,                    ///< [in,out] Reference to size in bytes of \p d_temp_storage allocation
        SampleIteratorT                     d_samples,                              ///< [in] The pointer to the input sequence of sample items. The samples from different channels are assumed to be interleaved (e.g., an array of 32-bit pixels where each pixel consists of four RGBA 8-bit samples).
        CounterT                            *d_histogram[NUM_ACTIVE_CHANNELS],      ///< [out] The pointers to the histogram counter output arrays, one for each active channel.  For channel<sub><em>i</em></sub>, the allocation length of <tt>d_histograms[i]</tt> should be <tt>num_levels[i]</tt> - 1.
        int                                 num_levels[NUM_ACTIVE_CHANNELS],        ///< [in] The number of bin level boundaries for delineating histogram samples in each active channel.  Implies that the number of bins for channel<sub><em>i</em></sub> is <tt>num_levels[i]</tt> - 1.
        SampleTransformOpT                  transform_op[NUM_ACTIVE_CHANNELS],      ///< [in] Transform operators for determining bin-ids from samples, one for each channel
        int                                 num_row_pixels,                         ///< [in] The number of multi-channel pixels per row in the region of interest
        int                                 num_rows,                               ///< [in] The number of rows in the region of interest
        int                                 row_pixel_stride,                       ///< [in] The number of multi-channel pixels between starts of consecutive rows in the region of interest
        DeviceHistogramSweepKernelT         histogram_sweep_kernel,                 ///< [in] Kernel function pointer to parameterization of cub::DeviceHistogramSweepKernel
        KernelConfig                        histogram_sweep_config,                 ///< [in] Dispatch parameters that match the policy that \p histogram_sweep_kernel was compiled for
        cudaStream_t                        stream,                                 ///< [in] CUDA stream to launch kernels within.  Default is stream<sub>0</sub>.
        bool                                debug_synchronous)                      ///< [in] Whether or not to synchronize the stream after every kernel launch to check for errors.  May cause significant slowdown.  Default is \p false.
    {
    #ifndef CUB_RUNTIME_ENABLED

        // Kernel launch not supported from this device
        return CubDebug(cudaErrorNotSupported);

    #else

        cudaError error = cudaSuccess;
        do
        {
            // Get device ordinal
            int device_ordinal;
            if (CubDebug(error = cudaGetDevice(&device_ordinal))) break;

            // Get device SM version
            int sm_version;
            if (CubDebug(error = SmVersion(sm_version, device_ordinal))) break;

            // Get SM count
            int sm_count;
            if (CubDebug(error = cudaDeviceGetAttribute (&sm_count, cudaDevAttrMultiProcessorCount, device_ordinal))) break;

            // Get SM occupancy for histogram_sweep_kernel
            int histogram_sweep_sm_occupancy;
            if (CubDebug(error = MaxSmOccupancy(
                histogram_sweep_sm_occupancy,
                sm_version,
                histogram_sweep_kernel,
                histogram_sweep_config.block_threads))) break;

            // Get device occupancy for histogram_sweep_kernel
            int histogram_sweep_occupancy = histogram_sweep_sm_occupancy * sm_count;

            // Get grid dimensions
            int sweep_pixels_per_block = histogram_sweep_config.block_threads * histogram_sweep_config.pixels_per_thread;
            dim3 histogram_sweep_grid_size;
            histogram_sweep_grid_size.x = CUB_MIN(histogram_sweep_occupancy, (num_row_pixels + sweep_pixels_per_block - 1) / sweep_pixels_per_block);
            histogram_sweep_grid_size.y = CUB_MIN(histogram_sweep_occupancy, num_rows);
            histogram_sweep_grid_size.z = 1;
            int histogram_sweep_grid_blocks = histogram_sweep_grid_size.x * histogram_sweep_grid_size.y;

            // Temporary storage allocation requirements
            void* allocations[1];
            size_t allocation_sizes[1] =
            {
                NUM_ACTIVE_CHANNELS * histogram_sweep_grid_blocks * sizeof(CounterT) * MAX_PRIVATIZED_BINS,     // bytes needed for privatized histograms
            };

            // Alias the temporary allocations from the single storage blob (or compute the necessary size of the blob)
            if (CubDebug(error = AliasTemporaries(d_temp_storage, temp_storage_bytes, allocations, allocation_sizes))) break;
            if (d_temp_storage == NULL)
            {
                // Return if the caller is simply requesting the size of the storage allocation
                return cudaSuccess;
            }

            // Alias the allocation for the privatized per-block reductions
            CounterT *d_block_histograms = (CounterT*) allocations[0];

            // Setup array wrapper for histogram channel output (because we can't pass static arrays as kernel parameters)
            ArrayWrapper<CounterT*, NUM_ACTIVE_CHANNELS> d_histo_wrapper;
            for (int CHANNEL = 0; CHANNEL < NUM_ACTIVE_CHANNELS; ++CHANNEL)
                d_histo_wrapper.array[CHANNEL] = d_histogram[CHANNEL];

            // Setup array wrapper for privatized per-block histogram channel output (because we can't pass static arrays as kernel parameters)
            ArrayWrapper<CounterT*, NUM_ACTIVE_CHANNELS> d_temp_histo_wrapper;
            for (int CHANNEL = 0; CHANNEL < NUM_ACTIVE_CHANNELS; ++CHANNEL)
                d_temp_histo_wrapper.array[CHANNEL] = d_block_histograms + (CHANNEL * histogram_sweep_grid_size * MAX_PRIVATIZED_BINS);

            // Setup array wrapper for bin transformers (because we can't pass static arrays as kernel parameters)
            ArrayWrapper<SampleTransformOpT, NUM_ACTIVE_CHANNELS> transform_op_wrapper;
            for (int CHANNEL = 0; CHANNEL < NUM_ACTIVE_CHANNELS; ++CHANNEL)
                transform_op_wrapper.array[CHANNEL] = transform_op[CHANNEL];

            // Setup array wrapper for num bins (because we can't pass static arrays as kernel parameters)
            ArrayWrapper<int, NUM_ACTIVE_CHANNELS> num_bins_wrapper;
            for (int CHANNEL = 0; CHANNEL < NUM_ACTIVE_CHANNELS; ++CHANNEL)
                num_bins_wrapper.array[CHANNEL] = num_levels[CHANNEL] - 1;

            // Log histogram_sweep_kernel configuration
            if (debug_synchronous) CubLog("Invoking histogram_sweep_kernel<<<{%d, %d, %d}, %d, 0, %lld>>>(), %d pixels per thread, %d SM occupancy\n",
                histogram_sweep_grid_size.x, histogram_sweep_grid_size.y, histogram_sweep_grid_size.z,
                histogram_sweep_config.block_threads, (long long) stream, histogram_sweep_config.pixels_per_thread, histogram_sweep_sm_occupancy);

            // Invoke histogram_sweep_kernel
            histogram_sweep_kernel<<<histogram_sweep_grid_size, histogram_sweep_config.block_threads, 0, stream>>>(
                d_samples,
                d_temp_histo_wrapper,
                transform_op_wrapper,
                num_row_pixels,
                num_rows,
                row_pixel_stride);

            // Check for failure to launch
            if (CubDebug(error = cudaPeekAtLastError())) break;

            // Sync the stream if specified to flush runtime errors
            if (debug_synchronous && (CubDebug(error = SyncStream(stream)))) break;

            // Log DeviceHistogramEvenAggregateKernel configuration
            if (debug_synchronous) CubLog("Invoking DeviceHistogramEvenAggregateKernel<<<%d, %d, 0, %lld>>>()\n",
                NUM_ACTIVE_CHANNELS, MAX_PRIVATIZED_BINS, (long long) stream);

            // Invoke kernel to aggregate the privatized histograms
            DeviceHistogramEvenAggregateKernel<MAX_PRIVATIZED_BINS, NUM_ACTIVE_CHANNELS, CounterT><<<NUM_ACTIVE_CHANNELS, MAX_PRIVATIZED_BINS, 0, stream>>>(
                num_bins_wrapper,
                d_temp_histo_wrapper,
                d_histo_wrapper,
                histogram_sweep_grid_blocks);

            // Check for failure to launch
            if (CubDebug(error = cudaPeekAtLastError())) break;

            // Sync the stream if specified to flush runtime errors
            if (debug_synchronous && (CubDebug(error = SyncStream(stream)))) break;
        }
        while (0);

        return error;

    #endif // CUB_RUNTIME_ENABLED
    }



    /**
     * Dispatch routine for HistogramRange (specialized for iterators)
     */
    template <typename SampleIteratorT>
    CUB_RUNTIME_FUNCTION
    static cudaError_t DispatchRange(
        void                *d_temp_storage,                        ///< [in] %Device allocation of temporary storage.  When NULL, the required allocation size is written to \p temp_storage_bytes and no work is done.
        size_t              &temp_storage_bytes,                    ///< [in,out] Reference to size in bytes of \p d_temp_storage allocation
        SampleIteratorT     d_samples,                              ///< [in] The pointer to the multi-channel input sequence of data samples. The samples from different channels are assumed to be interleaved (e.g., an array of 32-bit pixels where each pixel consists of four RGBA 8-bit samples).
        CounterT            *d_histogram[NUM_ACTIVE_CHANNELS],      ///< [out] The pointers to the histogram counter output arrays, one for each active channel.  For channel<sub><em>i</em></sub>, the allocation length of <tt>d_histograms[i]</tt> should be <tt>num_levels[i]</tt> - 1.
        int                 num_levels[NUM_ACTIVE_CHANNELS],        ///< [in] The number of boundaries (levels) for delineating histogram samples in each active channel.  Implies that the number of bins for channel<sub><em>i</em></sub> is <tt>num_levels[i]</tt> - 1.
        LevelT              *d_levels[NUM_ACTIVE_CHANNELS],         ///< [in] The pointers to the arrays of boundaries (levels), one for each active channel.  Bin ranges are defined by consecutive boundary pairings: lower sample value boundaries are inclusive and upper sample value boundaries are exclusive.
        int                 num_row_pixels,                         ///< [in] The number of multi-channel pixels per row in the region of interest
        int                 num_rows,                               ///< [in] The number of rows in the region of interest
        int                 row_pixel_stride,                       ///< [in] The number of multi-channel pixels between starts of consecutive rows in the region of interest
        cudaStream_t        stream,                                 ///< [in] CUDA stream to launch kernels within.  Default is stream<sub>0</sub>.
        bool                debug_synchronous,                      ///< [in] Whether or not to synchronize the stream after every kernel launch to check for errors.  May cause significant slowdown.  Default is \p false.
        Int2Type<false>     is_pointer)                             ///< [in] Marker whether the input samples are a native pointer
    {
        cudaError error = cudaSuccess;
        do
        {
            // Get PTX version
            int ptx_version;
    #if (CUB_PTX_ARCH == 0)
            if (CubDebug(error = PtxVersion(ptx_version))) break;
    #else
            ptx_version = CUB_PTX_ARCH;
    #endif

            // Get kernel kernel dispatch configurations
            KernelConfig histogram_sweep_config;
            KernelConfig histogram_unzip_config;
            InitConfigs(ptx_version, histogram_sweep_config, histogram_unzip_config);

            // Determine the minimum and maximum number of levels in any channel
            int min_levels = num_levels[0];
            int max_levels = num_levels[0];
            for (int channel = 1; channel < NUM_ACTIVE_CHANNELS; ++channel)
            {
                if (num_levels[channel] > max_levels)
                    max_levels = num_levels[channel];
                if (num_levels[channel] < min_levels)
                    min_levels = num_levels[channel];
            }

            // Minimum and maximum number of bins in any channel
            int max_bins = max_levels - 1;
            int min_bins = min_levels - 1;

        } while (0);

        return error;

    }



    /**
     * Dispatch routine for HistogramRange (specialized for native pointers)
     */
    template <typename SampleIteratorT>
    CUB_RUNTIME_FUNCTION
    static cudaError_t DispatchRange(
        void                *d_temp_storage,                        ///< [in] %Device allocation of temporary storage.  When NULL, the required allocation size is written to \p temp_storage_bytes and no work is done.
        size_t              &temp_storage_bytes,                    ///< [in,out] Reference to size in bytes of \p d_temp_storage allocation
        SampleIteratorT     d_samples,                              ///< [in] The pointer to the multi-channel input sequence of data samples. The samples from different channels are assumed to be interleaved (e.g., an array of 32-bit pixels where each pixel consists of four RGBA 8-bit samples).
        CounterT            *d_histogram[NUM_ACTIVE_CHANNELS],      ///< [out] The pointers to the histogram counter output arrays, one for each active channel.  For channel<sub><em>i</em></sub>, the allocation length of <tt>d_histograms[i]</tt> should be <tt>num_levels[i]</tt> - 1.
        int                 num_levels[NUM_ACTIVE_CHANNELS],        ///< [in] The number of boundaries (levels) for delineating histogram samples in each active channel.  Implies that the number of bins for channel<sub><em>i</em></sub> is <tt>num_levels[i]</tt> - 1.
        LevelT              *d_levels[NUM_ACTIVE_CHANNELS],         ///< [in] The pointers to the arrays of boundaries (levels), one for each active channel.  Bin ranges are defined by consecutive boundary pairings: lower sample value boundaries are inclusive and upper sample value boundaries are exclusive.
        int                 num_row_pixels,                         ///< [in] The number of multi-channel pixels per row in the region of interest
        int                 num_rows,                               ///< [in] The number of rows in the region of interest
        int                 row_pixel_stride,                       ///< [in] The number of multi-channel pixels between starts of consecutive rows in the region of interest
        cudaStream_t        stream,                                 ///< [in] CUDA stream to launch kernels within.  Default is stream<sub>0</sub>.
        bool                debug_synchronous,                      ///< [in] Whether or not to synchronize the stream after every kernel launch to check for errors.  May cause significant slowdown.  Default is \p false.
        Int2Type<true>      is_pointer)                             ///< [in] Marker whether the input samples are a native pointer
    {
#ifdef CUB_CDP
        typedef TexObjInputIterator<SampleT, OffsetT> TexSampleIterator;
#else
        typedef TexRefInputIterator<SampleT, __LINE__, OffsetT> TexSampleIterator;
#endif

        TexSampleIterator tex_samples;
        int bound = false;

        cudaError error = cudaSuccess;
        do
        {
            // Wrap pointer in texture
            if (CubDebug(error = tex_samples.BindTexture(d_samples, sizeof(SampleT) * num_rows * row_pixel_stride))) break;
            bound = true;

            if (CubDebug(error = DispatchRange(
                    d_temp_storage,
                    temp_storage_bytes,
                    tex_samples,
                    d_histogram,
                    num_levels,
                    d_levels,
                    num_row_pixels,
                    num_rows,
                    row_pixel_stride,
                    stream,
                    debug_synchronous,
                    Int2Type<false>()))) break;
        }
        while (0);

        if (bound)
            CubDebug(tex_samples.UnbindTexture());

        return error;
    }



    /**
     * Dispatch routine for HistogramEven (specialized for iterators)
     */
    template <typename SampleIteratorT>
    CUB_RUNTIME_FUNCTION __forceinline__
    static cudaError_t DispatchEven(
        void                *d_temp_storage,                        ///< [in] %Device allocation of temporary storage.  When NULL, the required allocation size is written to \p temp_storage_bytes and no work is done.
        size_t              &temp_storage_bytes,                    ///< [in,out] Reference to size in bytes of \p d_temp_storage allocation
        SampleIteratorT     d_samples,                              ///< [in] The pointer to the input sequence of sample items. The samples from different channels are assumed to be interleaved (e.g., an array of 32-bit pixels where each pixel consists of four RGBA 8-bit samples).
        CounterT            *d_histogram[NUM_ACTIVE_CHANNELS],      ///< [out] The pointers to the histogram counter output arrays, one for each active channel.  For channel<sub><em>i</em></sub>, the allocation length of <tt>d_histograms[i]</tt> should be <tt>num_levels[i]</tt> - 1.
        int                 num_levels[NUM_ACTIVE_CHANNELS],        ///< [in] The number of bin level boundaries for delineating histogram samples in each active channel.  Implies that the number of bins for channel<sub><em>i</em></sub> is <tt>num_levels[i]</tt> - 1.
        LevelT              lower_level[NUM_ACTIVE_CHANNELS],       ///< [in] The lower sample value bound (inclusive) for the lowest histogram bin in each active channel.
        LevelT              upper_level[NUM_ACTIVE_CHANNELS],       ///< [in] The upper sample value bound (exclusive) for the highest histogram bin in each active channel.
        int                 num_row_pixels,                         ///< [in] The number of multi-channel pixels per row in the region of interest
        int                 num_rows,                               ///< [in] The number of rows in the region of interest
        int                 row_pixel_stride,                       ///< [in] The number of multi-channel pixels between starts of consecutive rows in the region of interest
        cudaStream_t        stream,                                 ///< [in] CUDA stream to launch kernels within.  Default is stream<sub>0</sub>.
        bool                debug_synchronous,                      ///< [in] Whether or not to synchronize the stream after every kernel launch to check for errors.  May cause significant slowdown.  Default is \p false.
        Int2Type<false>     is_pointer)                             ///< [in] Marker whether the input samples are a native pointer
    {
        cudaError error = cudaSuccess;
        do
        {
            // Get PTX version
            int ptx_version;
    #if (CUB_PTX_ARCH == 0)
            if (CubDebug(error = PtxVersion(ptx_version))) break;
    #else
            ptx_version = CUB_PTX_ARCH;
    #endif

            // Get kernel kernel dispatch configurations
            KernelConfig histogram_sweep_config;
            KernelConfig histogram_unzip_config;
            InitConfigs(ptx_version, histogram_sweep_config, histogram_unzip_config);

            // Determine the minimum and maximum number of levels in any channel
            int min_levels = num_levels[0];
            int max_levels = num_levels[0];
            for (int channel = 1; channel < NUM_ACTIVE_CHANNELS; ++channel)
            {
                if (num_levels[channel] > max_levels)
                    max_levels = num_levels[channel];
                if (num_levels[channel] < min_levels)
                    min_levels = num_levels[channel];
            }

            // Minimum and maximum number of bins in any channel
            int max_bins = max_levels - 1;
            int min_bins = min_levels - 1;

            if ((sizeof(SampleT) == 1) && (max_bins == 256) && (min_bins == 256))
            {
                // Dispatch privatized approach for the common scenario (8-bit samples with 256 bins in every channel) using efficient scale-free transformer
                ScaleFreeTransform<SampleT> transform_op[NUM_ACTIVE_CHANNELS];

                if (CubDebug(error = PrivatizedDispatch(
                    d_temp_storage,
                    temp_storage_bytes,
                    d_samples,
                    d_histogram,
                    num_levels,
                    transform_op,
                    num_row_pixels,
                    num_rows,
                    row_pixel_stride,
                    DeviceHistogramSweepKernel<PtxHistogramSweepPolicy, MAX_PRIVATIZED_BINS, NUM_CHANNELS, NUM_ACTIVE_CHANNELS, SampleIteratorT, CounterT, ScaleFreeTransform<SampleT>, OffsetT>,
                    histogram_sweep_config,
                    stream,
                    debug_synchronous))) break;
            }
            else
            {
                // Use the default sample transformer for scaling samples
                ScaleTransform transform_op[NUM_ACTIVE_CHANNELS];
                for (int channel = 0; channel < NUM_ACTIVE_CHANNELS; ++channel)
                {
                    transform_op[channel].Init(
                        num_levels[channel],
                        upper_level[channel],
                        lower_level[channel],
                        ((upper_level[channel] - lower_level[channel]) / (num_levels[channel] - 1)));
                }

                if (max_bins <= MAX_PRIVATIZED_BINS)
                {
                    // Too many bins for a privatized approach.  Dispatch global sorting approach.
                    if (CubDebug(error = SortingDispatch(
                        d_temp_storage,
                        temp_storage_bytes,
                        d_samples,
                        d_histogram,
                        num_levels,
                        max_bins,
                        transform_op,
                        num_row_pixels,
                        num_rows,
                        row_pixel_stride,
                        DeviceHistogramUnzipKernel<PtxHistogramUnzipPolicy, NUM_CHANNELS, NUM_ACTIVE_CHANNELS, SampleIteratorT, SampleT, OffsetT>,
                        histogram_unzip_config,
                        stream,
                        debug_synchronous))) break;
                }
                else
                {
                    // Dispatch privatized approach
                    if (CubDebug(error = PrivatizedDispatch(
                        d_temp_storage,
                        temp_storage_bytes,
                        d_samples,
                        d_histogram,
                        num_levels,
                        num_row_pixels,
                        num_rows,
                        row_pixel_stride,
                        transform_op,
                        DeviceHistogramSweepKernel<PtxHistogramSweepPolicy, MAX_PRIVATIZED_BINS, NUM_CHANNELS, NUM_ACTIVE_CHANNELS, SampleIteratorT, CounterT, ScaleTransform, OffsetT>,
                        histogram_sweep_config,
                        stream,
                        debug_synchronous))) break;

                }
            }
        }
        while (0);

        return error;
    }


    /**
     * Dispatch routine for HistogramEven (specialized for native pointers)
     */
    template <typename SampleIteratorT>
    CUB_RUNTIME_FUNCTION __forceinline__
    static cudaError_t DispatchEven(
        void                *d_temp_storage,                        ///< [in] %Device allocation of temporary storage.  When NULL, the required allocation size is written to \p temp_storage_bytes and no work is done.
        size_t              &temp_storage_bytes,                    ///< [in,out] Reference to size in bytes of \p d_temp_storage allocation
        SampleIteratorT     d_samples,                              ///< [in] The pointer to the input sequence of sample items. The samples from different channels are assumed to be interleaved (e.g., an array of 32-bit pixels where each pixel consists of four RGBA 8-bit samples).
        CounterT            *d_histogram[NUM_ACTIVE_CHANNELS],      ///< [out] The pointers to the histogram counter output arrays, one for each active channel.  For channel<sub><em>i</em></sub>, the allocation length of <tt>d_histograms[i]</tt> should be <tt>num_levels[i]</tt> - 1.
        int                 num_levels[NUM_ACTIVE_CHANNELS],        ///< [in] The number of bin level boundaries for delineating histogram samples in each active channel.  Implies that the number of bins for channel<sub><em>i</em></sub> is <tt>num_levels[i]</tt> - 1.
        LevelT              lower_level[NUM_ACTIVE_CHANNELS],       ///< [in] The lower sample value bound (inclusive) for the lowest histogram bin in each active channel.
        LevelT              upper_level[NUM_ACTIVE_CHANNELS],       ///< [in] The upper sample value bound (exclusive) for the highest histogram bin in each active channel.
        int                 num_row_pixels,                         ///< [in] The number of multi-channel pixels per row in the region of interest
        int                 num_rows,                               ///< [in] The number of rows in the region of interest
        int                 row_pixel_stride,                       ///< [in] The number of multi-channel pixels between starts of consecutive rows in the region of interest
        cudaStream_t        stream,                                 ///< [in] CUDA stream to launch kernels within.  Default is stream<sub>0</sub>.
        bool                debug_synchronous,                      ///< [in] Whether or not to synchronize the stream after every kernel launch to check for errors.  May cause significant slowdown.  Default is \p false.
        Int2Type<true>      is_pointer)                             ///< [in] Marker whether the input samples are a native pointer
    {
#ifdef CUB_CDP
        typedef TexObjInputIterator<SampleT, OffsetT> TexSampleIterator;
#else
        typedef TexRefInputIterator<SampleT, __LINE__, OffsetT> TexSampleIterator;
#endif

        TexSampleIterator tex_samples;
        int bound = false;

        cudaError error = cudaSuccess;
        do
        {
            // Wrap pointer in texture
            if (CubDebug(error = tex_samples.BindTexture(d_samples, sizeof(SampleT) * num_rows * row_pixel_stride))) break;
            bound = true;

            if (CubDebug(error = DispatchEven(
                    d_temp_storage,
                    temp_storage_bytes,
                    tex_samples,
                    d_histogram,
                    num_levels,
                    lower_level,
                    upper_level,
                    num_row_pixels,
                    num_rows,
                    row_pixel_stride,
                    stream,
                    debug_synchronous,
                    Int2Type<false>()))) break;
        }
        while (0);

        if (bound)
            CubDebug(tex_samples.UnbindTexture());

        return error;
    }
};


}               // CUB namespace
CUB_NS_POSTFIX  // Optional outer namespace(s)


