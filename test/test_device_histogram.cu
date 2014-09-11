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

/******************************************************************************
 * Test of DeviceHistogram utilities
 ******************************************************************************/

// Ensure printing of CUDA runtime errors to console
#define CUB_STDERR

#include <stdio.h>
#include <limits>
#include <algorithm>

#include <npp.h>

#include <cub/util_allocator.cuh>
#include <cub/device/device_histogram.cuh>

#include "test_util.h"

using namespace cub;


//---------------------------------------------------------------------
// Globals, constants and typedefs
//---------------------------------------------------------------------


// Dispatch types
enum Backend
{
    CUB,        // CUB method
    NPP,        // NPP method
    CDP,        // GPU-based (dynamic parallelism) dispatch to CUB method
};


bool                    g_verbose_input     = false;
bool                    g_verbose           = false;
int                     g_timing_iterations = 0;
int                     g_repeat            = 0;
CachingDeviceAllocator  g_allocator(true);




//---------------------------------------------------------------------
// Dispatch to NPP histogram
//---------------------------------------------------------------------

/**
 * Dispatch to NPP
 * /
template <int BINS, int NUM_CHANNELS, int NUM_ACTIVE_CHANNELS, typename SampleIteratorT, typename CounterT>
cudaError_t Dispatch(
    Int2Type<NPP_HISTO> algorithm,
    Int2Type<false>     use_cdp,
    int                 timing_timing_iterations,
    size_t              *d_temp_storage_bytes,
    cudaError_t         *d_cdp_error,

    void                *d_temp_storage,
    size_t              &temp_storage_bytes,
    char                *d_samples,
    SampleIteratorT      d_sample_itr,
    CounterT            *d_histograms[NUM_ACTIVE_CHANNELS],
    int                 num_samples,
    cudaStream_t        stream,
    bool                debug_synchronous)
{
    cudaError_t error       = cudaSuccess;
    int binCount            = BINS;
    int levelCount          = BINS + 1;

    int rows = 1;
    NppiSize oSizeROI = {
        num_samples / rows,
        rows
    };

    if (d_temp_storage_bytes == NULL)
    {
        int nDeviceBufferSize;
        nppiHistogramEvenGetBufferSize_8u_C1R(oSizeROI, levelCount ,&nDeviceBufferSize);
        temp_storage_bytes = nDeviceBufferSize;
    }
    else
    {
        for (int i = 0; i < timing_timing_iterations; ++i)
        {
            // compute the histogram
            nppiHistogramEven_8u_C1R(
                d_samples,
                num_samples / rows,
                oSizeROI,
                d_histograms[0],
                levelCount,
                0,
                binCount,
                (Npp8u*) d_temp_storage);
        }
    }

    return error;
}


//---------------------------------------------------------------------
// Dispatch to different DeviceHistogram entrypoints
//---------------------------------------------------------------------

/**
 * Dispatch to CUB histogram-even entrypoint
 */
template <int NUM_CHANNELS, int NUM_ACTIVE_CHANNELS, typename SampleIteratorT, typename CounterT, typename LevelT>
CUB_RUNTIME_FUNCTION __forceinline__
cudaError_t DispatchEven(
    Int2Type<CUB>       dispatch_to,
    int                 timing_timing_iterations,
    size_t              *d_temp_storage_bytes,
    cudaError_t         *d_cdp_error,

    void                *d_temp_storage,
    size_t              &temp_storage_bytes,
    SampleIteratorT     d_samples,                                  ///< [in] The pointer to the multi-channel input sequence of data samples. The samples from different channels are assumed to be interleaved (e.g., an array of 32-bit pixels where each pixel consists of four RGBA 8-bit samples).
    CounterT            *d_histogram[NUM_ACTIVE_CHANNELS],          ///< [out] The pointers to the histogram counter output arrays, one for each active channel.  For channel<sub><em>i</em></sub>, the allocation length of <tt>d_histograms[i]</tt> should be <tt>num_levels[i]</tt> - 1.
    int                 num_levels[NUM_ACTIVE_CHANNELS],            ///< [in] The number of boundaries (levels) for delineating histogram samples in each active channel.  Implies that the number of bins for channel<sub><em>i</em></sub> is <tt>num_levels[i]</tt> - 1.
    LevelT              lower_level[NUM_ACTIVE_CHANNELS],           ///< [in] The lower sample value bound (inclusive) for the lowest histogram bin in each active channel.
    LevelT              upper_level[NUM_ACTIVE_CHANNELS],           ///< [in] The upper sample value bound (exclusive) for the highest histogram bin in each active channel.
    int                 num_row_pixels,                             ///< [in] The number of multi-channel pixels per row in the region of interest
    int                 num_rows,                                   ///< [in] The number of rows in the region of interest
    int                 row_stride,                                 ///< [in] The number of multi-channel pixels between starts of consecutive rows in the region of interest
    cudaStream_t        stream,
    bool                debug_synchronous)
{
    cudaError_t error = cudaSuccess;
    for (int i = 0; i < timing_timing_iterations; ++i)
    {
        error = DeviceHistogram::MultiHistogramEven<NUM_CHANNELS, NUM_ACTIVE_CHANNELS>(
            d_temp_storage,
            temp_storage_bytes,
            d_samples,
            d_histogram,
            num_levels,
            lower_level,
            upper_level,
            num_row_pixels,
            num_rows,
            row_stride,
            stream,
            debug_synchronous);
    }
    return error;
}


template <int NUM_CHANNELS, int NUM_ACTIVE_CHANNELS, typename SampleIteratorT, typename CounterT, typename LevelT>
CUB_RUNTIME_FUNCTION __forceinline__
cudaError_t DispatchRange(
    Int2Type<CUB>       dispatch_to,
    int                 timing_timing_iterations,
    size_t              *d_temp_storage_bytes,
    cudaError_t         *d_cdp_error,

    void                *d_temp_storage,
    size_t              &temp_storage_bytes,
    SampleIteratorT     d_samples,                                  ///< [in] The pointer to the multi-channel input sequence of data samples. The samples from different channels are assumed to be interleaved (e.g., an array of 32-bit pixels where each pixel consists of four RGBA 8-bit samples).
    CounterT            *d_histogram[NUM_ACTIVE_CHANNELS],          ///< [out] The pointers to the histogram counter output arrays, one for each active channel.  For channel<sub><em>i</em></sub>, the allocation length of <tt>d_histograms[i]</tt> should be <tt>num_levels[i]</tt> - 1.
    int                 num_levels[NUM_ACTIVE_CHANNELS],            ///< [in] The number of boundaries (levels) for delineating histogram samples in each active channel.  Implies that the number of bins for channel<sub><em>i</em></sub> is <tt>num_levels[i]</tt> - 1.
    LevelT              *d_levels[NUM_ACTIVE_CHANNELS],         ///< [in] The pointers to the arrays of boundaries (levels), one for each active channel.  Bin ranges are defined by consecutive boundary pairings: lower sample value boundaries are inclusive and upper sample value boundaries are exclusive.
    int                 num_row_pixels,                             ///< [in] The number of multi-channel pixels per row in the region of interest
    int                 num_rows,                                   ///< [in] The number of rows in the region of interest
    int                 row_stride,                                 ///< [in] The number of multi-channel pixels between starts of consecutive rows in the region of interest
    cudaStream_t        stream,
    bool                debug_synchronous)
{
    cudaError_t error = cudaSuccess;
    for (int i = 0; i < timing_timing_iterations; ++i)
    {
        error = DeviceHistogram::MultiHistogramRange<NUM_CHANNELS, NUM_ACTIVE_CHANNELS>(
            d_temp_storage,
            temp_storage_bytes,
            d_samples,
            d_histogram,
            num_levels,
            d_levels,
            num_row_pixels,
            num_rows,
            row_stride,
            stream,
            debug_synchronous);
    }
    return error;
}




//---------------------------------------------------------------------
// CUDA nested-parallelism test kernel
//---------------------------------------------------------------------

/**
 * Simple wrapper kernel to invoke DeviceHistogram
 * /
template <int BINS, int NUM_CHANNELS, int NUM_ACTIVE_CHANNELS, typename SampleT, typename SampleIteratorT, typename CounterT, int ALGORITHM>
__global__ void CnpDispatchKernel(
    Int2Type<ALGORITHM> algorithm,
    int                 timing_timing_iterations,
    size_t              *d_temp_storage_bytes,
    cudaError_t         *d_cdp_error,

    void                *d_temp_storage,
    size_t              temp_storage_bytes,
    SampleT             *d_samples,
    SampleIteratorT      d_sample_itr,
    ArrayWrapper<CounterT*, NUM_ACTIVE_CHANNELS> d_out_histograms,
    int                 num_samples,
    bool                debug_synchronous)
{
#ifndef CUB_CDP
    *d_cdp_error = cudaErrorNotSupported;
#else
    *d_cdp_error = Dispatch<BINS, NUM_CHANNELS, NUM_ACTIVE_CHANNELS>(algorithm, Int2Type<false>(), timing_timing_iterations, d_temp_storage_bytes, d_cdp_error, d_temp_storage, temp_storage_bytes, d_samples, d_sample_itr, d_out_histograms.array, num_samples, 0, debug_synchronous);
    *d_temp_storage_bytes = temp_storage_bytes;
#endif
}


/ **
 * Dispatch to CDP kernel
 * /
template <int BINS, int NUM_CHANNELS, int NUM_ACTIVE_CHANNELS, typename SampleT, typename SampleIteratorT, typename CounterT, int ALGORITHM>
cudaError_t Dispatch(
    Int2Type<ALGORITHM> algorithm,
    Int2Type<true>      use_cdp,
    int                 timing_timing_iterations,
    size_t              *d_temp_storage_bytes,
    cudaError_t         *d_cdp_error,

    void                *d_temp_storage,
    size_t              &temp_storage_bytes,
    SampleT             *d_samples,
    SampleIteratorT      d_sample_itr,
    CounterT        *d_histograms[NUM_ACTIVE_CHANNELS],
    int                 num_samples,
    cudaStream_t        stream,
    bool                debug_synchronous)
{
    // Setup array wrapper for histogram channel output (because we can't pass static arrays as kernel parameters)
    ArrayWrapper<CounterT*, NUM_ACTIVE_CHANNELS> d_histo_wrapper;
    for (int CHANNEL = 0; CHANNEL < NUM_ACTIVE_CHANNELS; ++CHANNEL)
        d_histo_wrapper.array[CHANNEL] = d_histograms[CHANNEL];

    // Invoke kernel to invoke device-side dispatch
    CnpDispatchKernel<BINS, NUM_CHANNELS, NUM_ACTIVE_CHANNELS, SampleIteratorT, CounterT, ALGORITHM><<<1,1>>>(algorithm, timing_timing_iterations, d_temp_storage_bytes, d_cdp_error, d_temp_storage, temp_storage_bytes, d_samples, d_sample_itr, d_histo_wrapper, num_samples, debug_synchronous);

    // Copy out temp_storage_bytes
    CubDebugExit(cudaMemcpy(&temp_storage_bytes, d_temp_storage_bytes, sizeof(size_t) * 1, cudaMemcpyDeviceToHost));

    // Copy out error
    cudaError_t retval;
    CubDebugExit(cudaMemcpy(&retval, d_cdp_error, sizeof(cudaError_t) * 1, cudaMemcpyDeviceToHost));
    return retval;
}
*/


//---------------------------------------------------------------------
// Test generation
//---------------------------------------------------------------------

// Searches for bin given a list of bin-boundary levels
template <typename LevelT>
struct SearchTransform
{
    LevelT          *levels;      // Pointer to levels array
    int             num_levels;   // Number of levels in array

    // Functor for converting samples to bin-ids (num_levels is returned if sample is out of range)
    template <typename SampleT>
    int operator()(SampleT sample)
    {
        int bin = std::upper_bound(levels, levels + num_levels, (LevelT) sample) - levels - 1;
        if (bin < 0)
        {
            // Sample out of range
            return num_levels;
        }
        return bin;
    }
};


// Scales samples to evenly-spaced bins
template <typename LevelT>
struct ScaleTransform
{
    int    num_levels;  // Number of levels in array
    LevelT max;         // Max sample level (exclusive)
    LevelT min;         // Min sample level (inclusive)
    LevelT scale;       // Bin scaling factor

    void Init(
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

    // Functor for converting samples to bin-ids  (num_levels is returned if sample is out of range)
    template <typename SampleT>
    int operator()(SampleT sample)
    {
        if ((sample < min) || (sample >= max))
        {
            // Sample out of range
            return num_levels;
        }

        return (int) ((((LevelT) sample) - min) / scale);
    }
};


/**
 * Generate sample
 */
template <typename T, typename LevelT>
void Sample(T &datum, LevelT max_value, int entropy_reduction)
{
    unsigned int max = (unsigned int) -1;
    unsigned int bits;
    RandomBits(bits, entropy_reduction);
    float fraction = (float(bits) / max);

    datum = (T) (fraction * max_value);
}


/**
 * Initialize histogram problem (and solution)
 */
template <
    int             NUM_CHANNELS,
    int             NUM_ACTIVE_CHANNELS,
    typename        LevelT,
    typename        SampleT,
    typename        CounterT,
    typename        TransformOp>
void Initialize(
    LevelT          max_value,
    int             entropy_reduction,
    SampleT         *h_samples,
    int             num_levels[NUM_ACTIVE_CHANNELS],        ///< [in] The number of boundaries (levels) for delineating histogram samples in each active channel.  Implies that the number of bins for channel<sub><em>i</em></sub> is <tt>num_levels[i]</tt> - 1.
    TransformOp     transform_op[NUM_ACTIVE_CHANNELS],      ///< [in] The lower sample value bound (inclusive) for the lowest histogram bin in each active channel.
    CounterT        *h_histogram[NUM_ACTIVE_CHANNELS],      ///< [out] The pointers to the histogram counter output arrays, one for each active channel.  For channel<sub><em>i</em></sub>, the allocation length of <tt>d_histograms[i]</tt> should be <tt>num_levels[i]</tt> - 1.
    int             num_row_pixels,                         ///< [in] The number of multi-channel pixels per row in the region of interest
    int             num_rows,                               ///< [in] The number of rows in the region of interest
    int             row_stride)                             ///< [in] The number of multi-channel pixels between starts of consecutive rows in the region of interest
{
    // Init bins
    for (int CHANNEL = 0; CHANNEL < NUM_ACTIVE_CHANNELS; ++CHANNEL)
    {
        for (int bin = 0; bin < num_levels[CHANNEL] - 1; ++bin)
        {
            h_histogram[CHANNEL][bin] = 0;
        }
    }

    // Initialize samples
    if (g_verbose_input) printf("Samples: \n");
    for (int row = 0; row < num_rows; ++row)
    {
        for (int pixel = 0; pixel < num_row_pixels; ++pixel)
        {
            if (g_verbose_input) printf("[");
            for (int channel = 0; channel < NUM_ACTIVE_CHANNELS; ++channel)
            {
                // Sample offset
                size_t offset = (row * row_stride * NUM_CHANNELS) + (pixel * NUM_CHANNELS) + channel;

                // Init sample value
                Sample(h_samples[offset], max_value, entropy_reduction);
                if (g_verbose_input)
                {
                    if (channel > 0) printf(", ");
                    std::cout << CoutCast(h_samples[offset]);
                }


                // Update sample bin
                int bin = transform_op[channel](h_samples[offset]);
                if (g_verbose_input) printf(" (%d)", bin); fflush(stdout);
                h_histogram[channel][bin]++;
            }
            if (g_verbose_input) printf("]");
        }
        if (g_verbose_input) printf("\n\n");
    }
}


/**
 * Test histogram-even
 */
template <
    Backend         BACKEND,
    int             NUM_CHANNELS,
    int             NUM_ACTIVE_CHANNELS,
    typename        SampleT,
    typename        CounterT,
    typename        LevelT>
void Test(
    LevelT          max_value,
    int             entropy_reduction,
    int             num_levels[NUM_ACTIVE_CHANNELS],            ///< [in] The number of boundaries (levels) for delineating histogram samples in each active channel.  Implies that the number of bins for channel<sub><em>i</em></sub> is <tt>num_levels[i]</tt> - 1.
    LevelT          lower_level[NUM_ACTIVE_CHANNELS],           ///< [in] The lower sample value bound (inclusive) for the lowest histogram bin in each active channel.
    LevelT          upper_level[NUM_ACTIVE_CHANNELS],           ///< [in] The upper sample value bound (exclusive) for the highest histogram bin in each active channel.
    int             num_row_pixels,                             ///< [in] The number of multi-channel pixels per row in the region of interest
    int             num_rows,                                   ///< [in] The number of rows in the region of interest
    int             row_stride,                           ///< [in] The number of multi-channel pixels between starts of consecutive rows in the region of interest
    char*           type_string)
{
    int total_samples =  num_rows * row_stride * NUM_CHANNELS;

    printf("%s cub::DeviceHistogram %d pixels (%d height, %d width, %d stride), %d %d-byte %s samples, %d/%d channels, max sample ",
        (BACKEND == CDP) ? "CDP CUB" : (BACKEND == NPP) ? "NPP" : "CUB",
        num_row_pixels * num_rows, num_rows, num_row_pixels, row_stride,
        total_samples, (int) sizeof(SampleT), type_string,
        NUM_ACTIVE_CHANNELS, NUM_CHANNELS);
    std::cout << CoutCast(max_value) << "\n";
    for (int channel = 0; channel < NUM_ACTIVE_CHANNELS; ++channel)
        std::cout << "Channel " << channel << ": " << num_levels[channel] - 1 << " bins [" << lower_level[channel] << ", " << upper_level[channel] << ")\n";
    fflush(stdout);

    // Allocate and initialize host and device data

    SampleT*                    h_samples = new SampleT[total_samples];
    CounterT*                   h_histogram[NUM_ACTIVE_CHANNELS];
    ScaleTransform<LevelT>      transform_op[NUM_ACTIVE_CHANNELS];

    for (int channel = 0; channel < NUM_ACTIVE_CHANNELS; ++channel)
    {
        int bins = num_levels[channel] - 1;
        h_histogram[channel] = new CounterT[bins];

        transform_op[channel].Init(
            num_levels[channel],
            upper_level[channel],
            lower_level[channel],
            ((upper_level[channel] - lower_level[channel]) / bins));
    }

    Initialize<NUM_CHANNELS, NUM_ACTIVE_CHANNELS>(
        max_value, entropy_reduction, h_samples, num_levels, transform_op, h_histogram, num_row_pixels, num_rows, row_stride);

    // Allocate and initialize device data

    SampleT*        d_samples = NULL;
    CounterT*       d_histogram[NUM_ACTIVE_CHANNELS];

    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_samples, sizeof(SampleT) * total_samples));
    CubDebugExit(cudaMemcpy(d_samples, h_samples, sizeof(SampleT) * total_samples, cudaMemcpyHostToDevice));
    for (int channel = 0; channel < NUM_ACTIVE_CHANNELS; ++channel)
    {
        CubDebugExit(g_allocator.DeviceAllocate((void**)&d_histogram[channel], sizeof(CounterT) * (num_levels[channel] - 1)));
        CubDebugExit(cudaMemset(d_histogram[channel], 0, sizeof(CounterT) * (num_levels[channel] - 1)));
    }

    // Allocate CDP device arrays
    size_t          *d_temp_storage_bytes = NULL;
    cudaError_t     *d_cdp_error = NULL;
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_temp_storage_bytes,  sizeof(size_t) * 1));
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_cdp_error,           sizeof(cudaError_t) * 1));

    // Allocate temporary storage
    void            *d_temp_storage = NULL;
    size_t          temp_storage_bytes = 0;

    DispatchEven<NUM_CHANNELS, NUM_ACTIVE_CHANNELS>(
        Int2Type<BACKEND>(), 1, d_temp_storage_bytes, d_cdp_error,
        d_temp_storage, temp_storage_bytes,
        d_samples, d_histogram, num_levels, lower_level, upper_level,
        num_row_pixels, num_rows, row_stride,
        0, true);

    CubDebugExit(g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes));

    // Run warmup/correctness iteration
    DispatchEven<NUM_CHANNELS, NUM_ACTIVE_CHANNELS>(
        Int2Type<BACKEND>(), 1, d_temp_storage_bytes, d_cdp_error,
        d_temp_storage, temp_storage_bytes,
        d_samples, d_histogram, num_levels, lower_level, upper_level,
        num_row_pixels, num_rows, row_stride,
        0, true);

    // Flush any stdout/stderr
    CubDebugExit(cudaPeekAtLastError());
    CubDebugExit(cudaDeviceSynchronize());
    fflush(stdout);
    fflush(stderr);

    // Check for correctness (and display results, if specified)
    int error = 0;
    for (int channel = 0; channel < NUM_ACTIVE_CHANNELS; ++channel)
    {
        int channel_error = CompareDeviceResults(h_histogram[channel], d_histogram[channel], num_levels[channel] - 1, g_verbose, g_verbose);
        printf("Channel %d %s", channel, channel_error ? "FAIL" : "PASS\n");
        error |= channel_error;
    }

    // Performance
    GpuTimer gpu_timer;
    gpu_timer.Start();

    DispatchEven<NUM_CHANNELS, NUM_ACTIVE_CHANNELS>(
        Int2Type<BACKEND>(), g_timing_iterations, d_temp_storage_bytes, d_cdp_error,
        d_temp_storage, temp_storage_bytes,
        d_samples, d_histogram, num_levels, lower_level, upper_level,
        num_row_pixels, num_rows, row_stride,
        0, false);

    gpu_timer.Stop();
    float elapsed_millis = gpu_timer.ElapsedMillis();

    // Display performance
    if (g_timing_iterations > 0)
    {
        float avg_millis = elapsed_millis / g_timing_iterations;
        float grate = float(total_samples) / avg_millis / 1000.0 / 1000.0;
        float gbandwidth = grate * sizeof(SampleT);
        printf(", %.3f avg ms, %.3f billion samples/s, %.3f billion bins/s, %.3f billion pixels/s, %.3f logical GB/s",
            avg_millis,
            grate,
            grate * NUM_ACTIVE_CHANNELS / NUM_CHANNELS,
            grate / NUM_CHANNELS,
            gbandwidth);
    }

    printf("\n\n");

    // Cleanup
    if (h_samples) delete[] h_samples;

    for (int channel = 0; channel < NUM_ACTIVE_CHANNELS; ++channel)
    {
        if (h_histogram[channel])
            delete[] h_histogram[channel];

        if (d_histogram[channel])
            CubDebugExit(g_allocator.DeviceFree(d_histogram[channel]));
    }

    if (d_samples) CubDebugExit(g_allocator.DeviceFree(d_samples));
    if (d_temp_storage_bytes) CubDebugExit(g_allocator.DeviceFree(d_temp_storage_bytes));
    if (d_cdp_error) CubDebugExit(g_allocator.DeviceFree(d_cdp_error));
    if (d_temp_storage) CubDebugExit(g_allocator.DeviceFree(d_temp_storage));

    // Correctness asserts
    AssertEquals(0, error);
}





/**
 * Test histogram-range
 */
template <
    Backend         BACKEND,
    int             NUM_CHANNELS,
    int             NUM_ACTIVE_CHANNELS,
    typename        SampleT,
    typename        CounterT,
    typename        LevelT>
void Test(
    LevelT          max_value,
    int             entropy_reduction,
    int             num_levels[NUM_ACTIVE_CHANNELS],            ///< [in] The number of boundaries (levels) for delineating histogram samples in each active channel.  Implies that the number of bins for channel<sub><em>i</em></sub> is <tt>num_levels[i]</tt> - 1.
    LevelT*         levels[NUM_ACTIVE_CHANNELS],                ///< [in] The lower sample value bound (inclusive) for the lowest histogram bin in each active channel.
    int             num_row_pixels,                             ///< [in] The number of multi-channel pixels per row in the region of interest
    int             num_rows,                                   ///< [in] The number of rows in the region of interest
    int             row_stride,                                 ///< [in] The number of multi-channel pixels between starts of consecutive rows in the region of interest
    char*           type_string)
{
    int total_samples =  num_rows * row_stride * NUM_CHANNELS;

    printf("%s cub::DeviceHistogram %d pixels (%d height, %d width, %d stride), %d %d-byte %s samples, %d/%d channels, max sample ",
        (BACKEND == CDP) ? "CDP CUB" : (BACKEND == NPP) ? "NPP" : "CUB",
        num_row_pixels * num_rows, num_rows, num_row_pixels, row_stride,
        total_samples, (int) sizeof(SampleT), type_string,
        NUM_ACTIVE_CHANNELS, NUM_CHANNELS);
    std::cout << CoutCast(max_value) << "\n";
    for (int channel = 0; channel < NUM_ACTIVE_CHANNELS; ++channel)
    {
        printf("Channel %d: %d bins [", channel, num_levels[channel] - 1);
        std::cout << levels[channel][0];
        for (int level = 1; level < num_levels[channel]; ++level)
            std::cout << ", " << levels[channel][level];
        printf("]\n");
    }
    fflush(stdout);

    // Allocate and initialize host and device data
    SampleT*                    h_samples = new SampleT[total_samples];
    CounterT*                   h_histogram[NUM_ACTIVE_CHANNELS];
    SearchTransform<LevelT>     transform_op[NUM_ACTIVE_CHANNELS];

    for (int channel = 0; channel < NUM_ACTIVE_CHANNELS; ++channel)
    {
        transform_op[channel].levels = levels[channel];
        transform_op[channel].num_levels = num_levels[channel];

        int bins = num_levels[channel] - 1;
        h_histogram[channel] = new CounterT[bins];
    }

    Initialize<NUM_CHANNELS, NUM_ACTIVE_CHANNELS>(
        max_value, entropy_reduction, h_samples, num_levels, transform_op, h_histogram, num_row_pixels, num_rows, row_stride);

    // Allocate and initialize device data

    SampleT*        d_samples = NULL;
    LevelT*         d_levels[NUM_ACTIVE_CHANNELS];
    CounterT*       d_histogram[NUM_ACTIVE_CHANNELS];

    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_samples, sizeof(SampleT) * total_samples));
    CubDebugExit(cudaMemcpy(d_samples, h_samples, sizeof(SampleT) * total_samples, cudaMemcpyHostToDevice));
    for (int channel = 0; channel < NUM_ACTIVE_CHANNELS; ++channel)
    {
        CubDebugExit(g_allocator.DeviceAllocate((void**)&d_levels[channel], sizeof(LevelT) * num_levels[channel]));
        CubDebugExit(cudaMemcpy(d_levels[channel], levels[channel],         sizeof(LevelT) * num_levels[channel], cudaMemcpyHostToDevice));

        CubDebugExit(g_allocator.DeviceAllocate((void**)&d_histogram[channel],  sizeof(CounterT) * (num_levels[channel] - 1)));
        CubDebugExit(cudaMemset(d_histogram[channel], 0,                        sizeof(CounterT) * (num_levels[channel] - 1)));
    }

    // Allocate CDP device arrays
    size_t          *d_temp_storage_bytes = NULL;
    cudaError_t     *d_cdp_error = NULL;
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_temp_storage_bytes,  sizeof(size_t) * 1));
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_cdp_error,           sizeof(cudaError_t) * 1));

    // Allocate temporary storage
    void            *d_temp_storage = NULL;
    size_t          temp_storage_bytes = 0;

    DispatchRange<NUM_CHANNELS, NUM_ACTIVE_CHANNELS>(
        Int2Type<BACKEND>(), 1, d_temp_storage_bytes, d_cdp_error,
        d_temp_storage, temp_storage_bytes,
        d_samples, d_histogram, num_levels, d_levels,
        num_row_pixels, num_rows, row_stride,
        0, true);

    CubDebugExit(g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes));

    // Run warmup/correctness iteration
    DispatchRange<NUM_CHANNELS, NUM_ACTIVE_CHANNELS>(
        Int2Type<BACKEND>(), 1, d_temp_storage_bytes, d_cdp_error,
        d_temp_storage, temp_storage_bytes,
        d_samples, d_histogram, num_levels, d_levels,
        num_row_pixels, num_rows, row_stride,
        0, true);

    // Flush any stdout/stderr
    CubDebugExit(cudaPeekAtLastError());
    CubDebugExit(cudaDeviceSynchronize());
    fflush(stdout);
    fflush(stderr);

    // Check for correctness (and display results, if specified)
    int error = 0;
    for (int channel = 0; channel < NUM_ACTIVE_CHANNELS; ++channel)
    {
        int channel_error = CompareDeviceResults(h_histogram[channel], d_histogram[channel], num_levels[channel] - 1, g_verbose, g_verbose);
        printf("Channel %d %s", channel, channel_error ? "FAIL" : "PASS\n");
        error |= channel_error;
    }

    // Performance
    GpuTimer gpu_timer;
    gpu_timer.Start();

    DispatchRange<NUM_CHANNELS, NUM_ACTIVE_CHANNELS>(
        Int2Type<BACKEND>(), g_timing_iterations, d_temp_storage_bytes, d_cdp_error,
        d_temp_storage, temp_storage_bytes,
        d_samples, d_histogram, num_levels, d_levels,
        num_row_pixels, num_rows, row_stride,
        0, false);

    gpu_timer.Stop();
    float elapsed_millis = gpu_timer.ElapsedMillis();

    // Display performance
    if (g_timing_iterations > 0)
    {
        float avg_millis = elapsed_millis / g_timing_iterations;
        float grate = float(total_samples) / avg_millis / 1000.0 / 1000.0;
        float gbandwidth = grate * sizeof(SampleT);
        printf(", %.3f avg ms, %.3f billion samples/s, %.3f billion bins/s, %.3f billion pixels/s, %.3f logical GB/s",
            avg_millis,
            grate,
            grate * NUM_ACTIVE_CHANNELS / NUM_CHANNELS,
            grate / NUM_CHANNELS,
            gbandwidth);
    }

    printf("\n\n");

    // Cleanup
    if (h_samples) delete[] h_samples;

    for (int channel = 0; channel < NUM_ACTIVE_CHANNELS; ++channel)
    {
        if (h_histogram[channel])
            delete[] h_histogram[channel];

        if (d_histogram[channel])
            CubDebugExit(g_allocator.DeviceFree(d_histogram[channel]));

        if (d_levels[channel])
            CubDebugExit(g_allocator.DeviceFree(d_levels[channel]));
    }

    if (d_samples) CubDebugExit(g_allocator.DeviceFree(d_samples));
    if (d_temp_storage_bytes) CubDebugExit(g_allocator.DeviceFree(d_temp_storage_bytes));
    if (d_cdp_error) CubDebugExit(g_allocator.DeviceFree(d_cdp_error));
    if (d_temp_storage) CubDebugExit(g_allocator.DeviceFree(d_temp_storage));

    // Correctness asserts
    AssertEquals(0, error);
}





#if 0

/**
 * Test different dispatch
 */
template <
    int                         BINS,
    int                         NUM_CHANNELS,
    int                         NUM_ACTIVE_CHANNELS,
    typename                    SampleT,
    typename                    IteratorValue,
    typename                    CounterT,
    typename                    BinOp,
    int                         ALGORITHM>
void TestCnp(
    Int2Type<ALGORITHM>         algorithm,
    GenMode                     gen_mode,
    BinOp                       bin_op,
    int                         num_samples,
    char*                       type_string)
{
    Test<false, BINS, NUM_CHANNELS, NUM_ACTIVE_CHANNELS, SampleT, IteratorValue, CounterT>(algorithm, gen_mode, bin_op, num_samples, type_string);
#ifdef CUB_CDP
    Test<true, BINS, NUM_CHANNELS, NUM_ACTIVE_CHANNELS, SampleT, IteratorValue, CounterT>(algorithm, gen_mode, bin_op, num_samples, type_string);
#endif
}


/**
 * Test different algorithms
 */
template <
    int             BINS,
    int             NUM_CHANNELS,
    int             NUM_ACTIVE_CHANNELS,
    typename        SampleT,
    typename        IteratorValue,
    typename        CounterT,
    typename        BinOp>
void Test(
    GenMode         gen_mode,
    BinOp           bin_op,
    int             num_samples,
    char*           type_string)
{
    TestCnp<BINS, NUM_CHANNELS, NUM_ACTIVE_CHANNELS, SampleT, IteratorValue, CounterT>(Int2Type<DEVICE_HISTO_SORT>(),          gen_mode, bin_op, num_samples * NUM_CHANNELS, type_string);
    TestCnp<BINS, NUM_CHANNELS, NUM_ACTIVE_CHANNELS, SampleT, IteratorValue, CounterT>(Int2Type<DEVICE_HISTO_SHARED_ATOMIC>(), gen_mode, bin_op, num_samples * NUM_CHANNELS, type_string);
    TestCnp<BINS, NUM_CHANNELS, NUM_ACTIVE_CHANNELS, SampleT, IteratorValue, CounterT>(Int2Type<DEVICE_HISTO_GLOBAL_ATOMIC>(), gen_mode, bin_op, num_samples * NUM_CHANNELS, type_string);
}


/**
 * Iterate over different channel configurations
 */
template <
    int             BINS,
    typename        SampleT,
    typename        IteratorValue,
    typename        CounterT,
    typename        BinOp>
void Test(
    GenMode         gen_mode,
    BinOp           bin_op,
    int             num_samples,
    char*           type_string)
{
    Test<BINS, 1, 1, SampleT, IteratorValue, CounterT>(gen_mode, bin_op, num_samples, type_string);
    Test<BINS, 4, 3, SampleT, IteratorValue, CounterT>(gen_mode, bin_op, num_samples, type_string);
}


/**
 * Iterate over different gen modes
 */
template <
    int             BINS,
    typename        SampleT,
    typename        IteratorValue,
    typename        CounterT,
    typename        BinOp>
void TestModes(
    BinOp           bin_op,
    int             num_samples,
    char*           type_string)
{
    Test<BINS, SampleT, IteratorValue, CounterT>(RANDOM, bin_op, num_samples, type_string);
    Test<BINS, SampleT, IteratorValue, CounterT>(UNIFORM, bin_op, num_samples, type_string);
}


/**
 * Iterate input sizes
 */
template <
    int             BINS,
    typename        SampleT,
    typename        IteratorValue,
    typename        CounterT,
    typename        BinOp>
void Test(
    BinOp           bin_op,
    int             num_samples,
    char*           type_string)
{
    if (num_samples < 0)
    {
        TestModes<BINS, SampleT, IteratorValue, CounterT>(bin_op, 1,       type_string);
        TestModes<BINS, SampleT, IteratorValue, CounterT>(bin_op, 100,     type_string);
        TestModes<BINS, SampleT, IteratorValue, CounterT>(bin_op, 10000,   type_string);
        TestModes<BINS, SampleT, IteratorValue, CounterT>(bin_op, 1000000, type_string);
    }
    else
    {
        TestModes<BINS, SampleT, IteratorValue, CounterT>(bin_op, num_samples, type_string);
    }
}

#endif


//---------------------------------------------------------------------
// Main
//---------------------------------------------------------------------

/**
 * Main
 */
int main(int argc, char** argv)
{
    int num_row_pixels = -1;
    int entropy_reduction = 0;
    int num_rows = 1;

    // Initialize command line
    CommandLineArgs args(argc, argv);
    g_verbose = args.CheckCmdLineFlag("v");
    g_verbose_input = args.CheckCmdLineFlag("v2");
    args.GetCmdLineArgument("n", num_row_pixels);

    int row_stride = num_row_pixels;

    args.GetCmdLineArgument("rows", num_rows);
    args.GetCmdLineArgument("stride", row_stride);
    args.GetCmdLineArgument("i", g_timing_iterations);
    args.GetCmdLineArgument("repeat", g_repeat);
    args.GetCmdLineArgument("entropy", entropy_reduction);

    // Print usage
    if (args.CheckCmdLineFlag("help"))
    {
        printf("%s "
            "[--n=<pixels per row> "
            "[--rows=<number of rows> "
            "[--stride=<row stride in pixels> "
            "[--i=<timing iterations> "
            "[--device=<device-id>] "
            "[--repeat=<repetitions of entire test suite>]"
            "[--entropy=<entropy-reduction factor (default 0)>]"
            "[--v] "
            "[--cdp]"
            "\n", argv[0]);
        exit(0);
    }

    // Initialize device
    CubDebugExit(args.DeviceInit());


#ifdef QUICKER_TEST

    // Compile/run quick tests
    if (num_row_pixels < 0) num_row_pixels = 32000000;

    {
        // HistogramRange: unsigned char 256 bins
        enum {
            NUM_CHANNELS = 1,
            NUM_ACTIVE_CHANNELS = 1,
        };

        int max_value = 256;
        int num_levels[NUM_ACTIVE_CHANNELS] = {257};
        int* levels[NUM_ACTIVE_CHANNELS];
        for (int channel = 0; channel < NUM_ACTIVE_CHANNELS; ++channel)
        {
            levels[channel] = new int[num_levels[channel]];
            for (int level = 0; level < num_levels[channel]; ++level)
                levels[channel][level] = level;
        }

        Test<CUB, NUM_CHANNELS, NUM_ACTIVE_CHANNELS, unsigned char, int>(max_value, entropy_reduction, num_levels, levels, num_row_pixels, num_rows, row_stride, "unsigned char");

        for (int channel = 0; channel < NUM_ACTIVE_CHANNELS; ++channel)
            delete[] levels[channel];
    }

/*
    {
        // HistogramEven: float [0,1.0] 256 bins
        float max_value = 1.0;
        int num_levels[1] = {257};
        float lower_level[1] = {0.0};
        float upper_level[1] = {1.0};
        Test<CUB, 1, 1, float, int>(max_value, entropy_reduction, num_levels, lower_level, upper_level, num_row_pixels, num_rows, row_stride, "float");
    }

    {
        // HistogramEven: unsigned char 256 bins
        int max_value = 256;
        int num_levels[1] = {257};
        int lower_level[1] = {0};
        int upper_level[1] = {256};
        Test<CUB, 1, 1, unsigned char, int>(max_value, entropy_reduction, num_levels, lower_level, upper_level, num_row_pixels, num_rows, row_stride, "unsigned char");
    }

    {
        // HistogramEven: 4/4 multichannel Unsigned char 256 bins
        int max_value = 256;
        int num_levels[4] = {257, 257, 257, 257};
        int lower_level[4] = {0, 0, 0, 0};
        int upper_level[4] = {256, 256, 256, 256};
        Test<CUB, 4, 4, unsigned char, int>(max_value, entropy_reduction, num_levels, lower_level, upper_level, num_row_pixels, num_rows, row_stride, "unsigned char");
    }

    {
        // HistogramEven: 3/4 multichannel Unsigned char 256 bins
        int max_value = 256;
        int num_levels[3] = {257, 257, 257};
        int lower_level[3] = {0, 0, 0};
        int upper_level[3] = {256, 256, 256};
        Test<CUB, 4, 3, unsigned char, int>(max_value, entropy_reduction, num_levels, lower_level, upper_level, num_row_pixels, num_rows, row_stride, "unsigned char");
    }

    {
        // HistogramEven: unsigned char 16 bins
        int max_value = 256;
        int num_levels[1] = {17};
        int lower_level[1] = {0};
        int upper_level[1] = {256};
        Test<CUB, 1, 1, unsigned char, int>(max_value, entropy_reduction, num_levels, lower_level, upper_level, num_row_pixels, num_rows, row_stride, "unsigned char");
    }
    {
        // HistogramEven: unsigned char 16 bins [64,192)
        int max_value = 256;
        int num_levels[1] = {16};
        int lower_level[1] = {64};
        int upper_level[1] = {192};
        Test<CUB, 1, 1, unsigned char, int>(max_value, entropy_reduction, num_levels, lower_level, upper_level, num_row_pixels, num_rows, row_stride, "unsigned char");
    }
*/
/*
    printf("SINGLE CHANNEL:\n\n");
    TestCnp<256, 1, 1, unsigned char, unsigned char, int>(Int2Type<DEVICE_HISTO_SORT>(),          RANDOM,  Cast<unsigned char>(), num_samples, CUB_TYPE_STRING(unsigned char));
    TestCnp<256, 1, 1, unsigned char, unsigned char, int>(Int2Type<DEVICE_HISTO_SORT>(),          UNIFORM, Cast<unsigned char>(), num_samples, CUB_TYPE_STRING(unsigned char));
    printf("\n");
    TestCnp<256, 1, 1, unsigned char, unsigned char, int>(Int2Type<DEVICE_HISTO_SHARED_ATOMIC>(), RANDOM,  Cast<unsigned char>(), num_samples, CUB_TYPE_STRING(unsigned char));
    TestCnp<256, 1, 1, unsigned char, unsigned char, int>(Int2Type<DEVICE_HISTO_SHARED_ATOMIC>(), UNIFORM, Cast<unsigned char>(), num_samples, CUB_TYPE_STRING(unsigned char));
    printf("\n");
    TestCnp<256, 1, 1, unsigned char, unsigned char, int>(Int2Type<DEVICE_HISTO_GLOBAL_ATOMIC>(), RANDOM,  Cast<unsigned char>(), num_samples, CUB_TYPE_STRING(unsigned char));
    TestCnp<256, 1, 1, unsigned char, unsigned char, int>(Int2Type<DEVICE_HISTO_GLOBAL_ATOMIC>(), UNIFORM, Cast<unsigned char>(), num_samples, CUB_TYPE_STRING(unsigned char));
    printf("\n");

    printf("3/4 CHANNEL (RGB/RGBA):\n\n");
    TestCnp<256, 4, 3, unsigned char, unsigned char, int>(Int2Type<DEVICE_HISTO_SORT>(),          RANDOM,  Cast<unsigned char>(), num_samples * 4, CUB_TYPE_STRING(unsigned char));
    TestCnp<256, 4, 3, unsigned char, unsigned char, int>(Int2Type<DEVICE_HISTO_SORT>(),          UNIFORM, Cast<unsigned char>(), num_samples * 4, CUB_TYPE_STRING(unsigned char));
    printf("\n");
    TestCnp<256, 4, 3, unsigned char, unsigned char, int>(Int2Type<DEVICE_HISTO_SHARED_ATOMIC>(), RANDOM,  Cast<unsigned char>(), num_samples * 4, CUB_TYPE_STRING(unsigned char));
    TestCnp<256, 4, 3, unsigned char, unsigned char, int>(Int2Type<DEVICE_HISTO_SHARED_ATOMIC>(), UNIFORM, Cast<unsigned char>(), num_samples * 4, CUB_TYPE_STRING(unsigned char));
    printf("\n");
    TestCnp<256, 4, 3, unsigned char, unsigned char, int>(Int2Type<DEVICE_HISTO_GLOBAL_ATOMIC>(), RANDOM,  Cast<unsigned char>(), num_samples * 4, CUB_TYPE_STRING(unsigned char));
    TestCnp<256, 4, 3, unsigned char, unsigned char, int>(Int2Type<DEVICE_HISTO_GLOBAL_ATOMIC>(), UNIFORM, Cast<unsigned char>(), num_samples * 4, CUB_TYPE_STRING(unsigned char));
    printf("\n");
*/

#elif defined(QUICK_TEST)

#else

    // Compile/run thorough tests
    for (int i = 0; i <= g_repeat; ++i)
    {
/*
        // 256-bin tests
        Test<256, unsigned char,    unsigned char, int>(Cast<unsigned char>(),              num_samples, CUB_TYPE_STRING(unsigned char));
        Test<256, unsigned short,   unsigned char, int>(Cast<unsigned char>(),              num_samples, CUB_TYPE_STRING(unsigned short));
        Test<256, unsigned int,     unsigned char, int>(Cast<unsigned char>(),              num_samples, CUB_TYPE_STRING(unsigned int));
        Test<256, float,            unsigned char, int>(FloatScaleOp<unsigned char, 256>(), num_samples, CUB_TYPE_STRING(float));

        // 512-bin tests
        Test<512, unsigned char,    unsigned short, int>(Cast<unsigned short>(),              num_samples, CUB_TYPE_STRING(unsigned char));
        Test<512, unsigned short,   unsigned short, int>(Cast<unsigned short>(),              num_samples, CUB_TYPE_STRING(unsigned short));
        Test<512, unsigned int,     unsigned short, int>(Cast<unsigned short>(),              num_samples, CUB_TYPE_STRING(unsigned int));
        Test<512, float,            unsigned short, int>(FloatScaleOp<unsigned short, 512>(), num_samples, CUB_TYPE_STRING(float));
*/
    }

#endif

    return 0;
}



