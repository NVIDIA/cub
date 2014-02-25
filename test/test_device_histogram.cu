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

#include <npp.h>

#include <cub/util_allocator.cuh>
#include <cub/device/device_histogram.cuh>
#include <cub/iterator/tex_ref_input_iterator.cuh>
#include <cub/iterator/tex_obj_input_iterator.cuh>
#include <cub/iterator/transform_input_iterator.cuh>

#include "test_util.h"

using namespace cub;


//---------------------------------------------------------------------
// Globals, constants and typedefs
//---------------------------------------------------------------------

bool                    g_verbose_input     = false;
bool                    g_verbose           = false;
int                     g_timing_iterations = 0;
int                     g_repeat            = 0;
CachingDeviceAllocator  g_allocator(true);

enum
{
    NPP_HISTO = 99
};


//---------------------------------------------------------------------
// Dispatch to NPP histogram
//---------------------------------------------------------------------

/**
 * Dispatch to NPP
 */
template <int BINS, int CHANNELS, int ACTIVE_CHANNELS, typename InputIterator, typename HistoCounter>
cudaError_t Dispatch(
    Int2Type<NPP_HISTO> algorithm,
    Int2Type<false>     use_cdp,
    int                 timing_timing_iterations,
    size_t              *d_temp_storage_bytes,
    cudaError_t         *d_cdp_error,

    void                *d_temp_storage,
    size_t              &temp_storage_bytes,
    char                *d_samples,
    InputIterator       d_sample_itr,
    HistoCounter        *d_histograms[ACTIVE_CHANNELS],
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
 * Dispatch to shared sorting entrypoint
 */
template <int BINS, int CHANNELS, int ACTIVE_CHANNELS, typename SampleT, typename InputIterator, typename HistoCounter>
__host__ __device__ __forceinline__
cudaError_t Dispatch(
    Int2Type<DEVICE_HISTO_SORT> algorithm,
    Int2Type<false>     use_cdp,
    int                 timing_timing_iterations,
    size_t              *d_temp_storage_bytes,
    cudaError_t         *d_cdp_error,

    void                *d_temp_storage,
    size_t              &temp_storage_bytes,
    SampleT             *d_samples,
    InputIterator       d_sample_itr,
    HistoCounter        *d_histograms[ACTIVE_CHANNELS],
    int                 num_samples,
    cudaStream_t        stream,
    bool                debug_synchronous)
{
    cudaError_t error = cudaSuccess;
    for (int i = 0; i < timing_timing_iterations; ++i)
    {
        error = DeviceHistogram::MultiChannelSorting<BINS, CHANNELS, ACTIVE_CHANNELS>(d_temp_storage, temp_storage_bytes, d_sample_itr, d_histograms, num_samples, stream, debug_synchronous);
    }
    return error;
}


/**
 *
 *
 * Dispatch to shared atomic entrypoint
 */
template <int BINS, int CHANNELS, int ACTIVE_CHANNELS, typename SampleT, typename InputIterator, typename HistoCounter>
__host__ __device__ __forceinline__
cudaError_t Dispatch(
    Int2Type<DEVICE_HISTO_SHARED_ATOMIC> algorithm,
    Int2Type<false>     use_cdp,
    int                 timing_timing_iterations,
    size_t              *d_temp_storage_bytes,
    cudaError_t         *d_cdp_error,

    void                *d_temp_storage,
    size_t              &temp_storage_bytes,
    SampleT             *d_samples,
    InputIterator       d_sample_itr,
    HistoCounter        *d_histograms[ACTIVE_CHANNELS],
    int                 num_samples,
    cudaStream_t        stream,
    bool                debug_synchronous)
{
    cudaError_t error = cudaSuccess;
    for (int i = 0; i < timing_timing_iterations; ++i)
    {
        error = DeviceHistogram::MultiChannelSharedAtomic<BINS, CHANNELS, ACTIVE_CHANNELS>(d_temp_storage, temp_storage_bytes, d_sample_itr, d_histograms, num_samples, stream, debug_synchronous);
    }
    return error;
}


/**
 * Dispatch to global atomic entrypoint
 */
template <int BINS, int CHANNELS, int ACTIVE_CHANNELS, typename SampleT, typename InputIterator, typename HistoCounter>
__host__ __device__ __forceinline__
cudaError_t Dispatch(
    Int2Type<DEVICE_HISTO_GLOBAL_ATOMIC> algorithm,
    Int2Type<false>     use_cdp,
    int                 timing_timing_iterations,
    size_t              *d_temp_storage_bytes,
    cudaError_t         *d_cdp_error,

    void                *d_temp_storage,
    size_t              &temp_storage_bytes,
    SampleT             *d_samples,
    InputIterator       d_sample_itr,
    HistoCounter        *d_histograms[ACTIVE_CHANNELS],
    int                 num_samples,
    cudaStream_t        stream,
    bool                debug_synchronous)
{
    cudaError_t error = cudaSuccess;
    for (int i = 0; i < timing_timing_iterations; ++i)
    {
        error = DeviceHistogram::MultiChannelGlobalAtomic<BINS, CHANNELS, ACTIVE_CHANNELS>(d_temp_storage, temp_storage_bytes, d_sample_itr, d_histograms, num_samples, stream, debug_synchronous);
    }
    return error;
}


//---------------------------------------------------------------------
// CUDA nested-parallelism test kernel
//---------------------------------------------------------------------

/**
 * Simple wrapper kernel to invoke DeviceScan
 */
template <int BINS, int CHANNELS, int ACTIVE_CHANNELS, typename SampleT, typename InputIterator, typename HistoCounter, int ALGORITHM>
__global__ void CnpDispatchKernel(
    Int2Type<ALGORITHM> algorithm,
    int                 timing_timing_iterations,
    size_t              *d_temp_storage_bytes,
    cudaError_t         *d_cdp_error,

    void                *d_temp_storage,
    size_t              temp_storage_bytes,
    SampleT             *d_samples,
    InputIterator       d_sample_itr,
    ArrayWrapper<HistoCounter*, ACTIVE_CHANNELS> d_out_histograms,
    int                 num_samples,
    bool                debug_synchronous)
{
#ifndef CUB_CDP
    *d_cdp_error = cudaErrorNotSupported;
#else
    *d_cdp_error = Dispatch<BINS, CHANNELS, ACTIVE_CHANNELS>(algorithm, Int2Type<false>(), timing_timing_iterations, d_temp_storage_bytes, d_cdp_error, d_temp_storage, temp_storage_bytes, d_samples, d_sample_itr, d_out_histograms.array, num_samples, 0, debug_synchronous);
    *d_temp_storage_bytes = temp_storage_bytes;
#endif
}


/**
 * Dispatch to CDP kernel
 */
template <int BINS, int CHANNELS, int ACTIVE_CHANNELS, typename SampleT, typename InputIterator, typename HistoCounter, int ALGORITHM>
cudaError_t Dispatch(
    Int2Type<ALGORITHM> algorithm,
    Int2Type<true>      use_cdp,
    int                 timing_timing_iterations,
    size_t              *d_temp_storage_bytes,
    cudaError_t         *d_cdp_error,

    void                *d_temp_storage,
    size_t              &temp_storage_bytes,
    SampleT             *d_samples,
    InputIterator       d_sample_itr,
    HistoCounter        *d_histograms[ACTIVE_CHANNELS],
    int                 num_samples,
    cudaStream_t        stream,
    bool                debug_synchronous)
{
    // Setup array wrapper for histogram channel output (because we can't pass static arrays as kernel parameters)
    ArrayWrapper<HistoCounter*, ACTIVE_CHANNELS> d_histo_wrapper;
    for (int CHANNEL = 0; CHANNEL < ACTIVE_CHANNELS; ++CHANNEL)
        d_histo_wrapper.array[CHANNEL] = d_histograms[CHANNEL];

    // Invoke kernel to invoke device-side dispatch
    CnpDispatchKernel<BINS, CHANNELS, ACTIVE_CHANNELS, InputIterator, HistoCounter, ALGORITHM><<<1,1>>>(algorithm, timing_timing_iterations, d_temp_storage_bytes, d_cdp_error, d_temp_storage, temp_storage_bytes, d_samples, d_sample_itr, d_histo_wrapper, num_samples, debug_synchronous);

    // Copy out temp_storage_bytes
    CubDebugExit(cudaMemcpy(&temp_storage_bytes, d_temp_storage_bytes, sizeof(size_t) * 1, cudaMemcpyDeviceToHost));

    // Copy out error
    cudaError_t retval;
    CubDebugExit(cudaMemcpy(&retval, d_cdp_error, sizeof(cudaError_t) * 1, cudaMemcpyDeviceToHost));
    return retval;
}



//---------------------------------------------------------------------
// Test generation
//---------------------------------------------------------------------

/**
 * Scaling operator for binning f32 types
 */
template <typename T, int BINS>
struct FloatScaleOp
{
    __host__ __device__ __forceinline__ T operator()(float datum) const
    {
        float datum_scale = datum * float(BINS - 1);
        return (T) datum_scale;
    }
};


/**
 * Generate integer sample
 */
template <int BINS, typename T>
void Sample(GenMode gen_mode, T &datum, int i)
{
    InitValue(gen_mode, datum, i);
    datum = datum % BINS;
}


/**
 * Generate float sample [0..1]
 */
template <int BINS>
void Sample(GenMode gen_mode, float &datum, int i)
{
    unsigned int bits;
    unsigned int max = (unsigned int) -1;

    InitValue(gen_mode, bits, i);
    datum = float(bits) / max;
}


/**
 * Initialize problem (and solution)
 */
template <
    int             BINS,
    int             CHANNELS,
    int             ACTIVE_CHANNELS,
    typename        IteratorValue,
    typename        SampleT,
    typename        HistoCounterT,
    typename        BinOp>
void Initialize(
    GenMode         gen_mode,
    SampleT         *h_samples,
    BinOp           bin_op,
    HistoCounterT   *h_histograms_linear,
    int             num_samples)
{
    // Init bins
    for (int bin = 0; bin < ACTIVE_CHANNELS * BINS; ++bin)
    {
        h_histograms_linear[bin] = 0;
    }

    if (g_verbose_input) printf("Samples: \n");

    // Initialize interleaved channel samples and histogram them correspondingly
    for (int i = 0; i < num_samples; ++i)
    {
        Sample<BINS>(gen_mode, h_samples[i], i);

        IteratorValue bin = bin_op(h_samples[i]);

        int channel = i % CHANNELS;

        if (g_verbose_input)
        {
            if (channel == 0) printf("<");
            if (channel == CHANNELS - 1)
                std::cout << h_samples[i] << ">, ";
            else
                std::cout << h_samples[i] << ", ";
        }

        if (channel < ACTIVE_CHANNELS)
        {
            h_histograms_linear[(channel * BINS) + bin]++;
        }
    }

    if (g_verbose_input) printf("\n\n");
}


/**
 * Test DeviceHistogram
 */
template <
    bool                        CDP,
    int                         BINS,
    int                         CHANNELS,
    int                         ACTIVE_CHANNELS,
    typename                    SampleT,
    typename                    IteratorValue,
    typename                    HistoCounterT,
    typename                    BinOp,
    int                         ALGORITHM>
void Test(
    Int2Type<ALGORITHM>         algorithm,
    GenMode                     gen_mode,
    BinOp                       bin_op,
    int                         num_samples,
    char*                       type_string)
{
    // Texture iterator type for loading samples through tex
#ifdef CUB_CDP
    typedef TexObjInputIterator<SampleT, int> TexIterator;
#else
    typedef TexRefInputIterator<SampleT, __LINE__, int> TexIterator;
#endif

    int compare         = 0;
    int cdp_compare     = 0;
    int total_bins      = ACTIVE_CHANNELS * BINS;

    printf("%s cub::DeviceHistogram %s %d %s samples (%dB), %d bins, %d channels, %d active channels, gen-mode %s\n",
        (CDP) ? "CDP device invoked" : "Host-invoked",
        (ALGORITHM == NPP_HISTO) ? "NPP" : (ALGORITHM == DEVICE_HISTO_SHARED_ATOMIC) ? "satomic" : (ALGORITHM == DEVICE_HISTO_GLOBAL_ATOMIC) ? "gatomic" : "sort",
        num_samples,
        type_string,
        (int) sizeof(SampleT),
        BINS,
        CHANNELS,
        ACTIVE_CHANNELS,
        (gen_mode == RANDOM) ? "RANDOM" : (gen_mode == INTEGER_SEED) ? "SEQUENTIAL" : "HOMOGENOUS");
    fflush(stdout);

    // Allocate host arrays
    SampleT         *h_samples          = new SampleT[num_samples];
    HistoCounterT   *h_reference_linear = new HistoCounterT[total_bins];

    // Initialize problem
    Initialize<BINS, CHANNELS, ACTIVE_CHANNELS, IteratorValue>(gen_mode, h_samples, bin_op, h_reference_linear, num_samples);

    // Allocate problem device arrays
    SampleT         *d_samples = NULL;
    HistoCounterT   *d_histograms_linear = NULL;
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_samples,             sizeof(SampleT) * num_samples));
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_histograms_linear,   sizeof(HistoCounterT) * total_bins));

    // Allocate CDP device arrays
    size_t          *d_temp_storage_bytes = NULL;
    cudaError_t     *d_cdp_error = NULL;
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_temp_storage_bytes,  sizeof(size_t) * 1));
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_cdp_error,           sizeof(cudaError_t) * 1));

    // Initialize/clear device arrays
    CubDebugExit(cudaMemcpy(d_samples, h_samples, sizeof(SampleT) * num_samples, cudaMemcpyHostToDevice));
    CubDebugExit(cudaMemset(d_histograms_linear, 0, sizeof(HistoCounterT) * total_bins));

    // Structure of channel histograms
    HistoCounterT *d_histograms[ACTIVE_CHANNELS];
    for (int CHANNEL = 0; CHANNEL < ACTIVE_CHANNELS; ++CHANNEL)
    {
        d_histograms[CHANNEL] = d_histograms_linear + (CHANNEL * BINS);
    }

    // Create a texture iterator wrapper
    TexIterator tex_itr;
    CubDebugExit(tex_itr.BindTexture(d_samples, sizeof(SampleT) * num_samples))

    // Create a transform iterator wrapper for SampleT -> IteratorValue conversion
    TransformInputIterator<IteratorValue, BinOp, TexIterator, int> d_sample_itr(tex_itr, bin_op);

    // Allocate temporary storage
    void            *d_temp_storage = NULL;
    size_t          temp_storage_bytes = 0;
    Dispatch<BINS, CHANNELS, ACTIVE_CHANNELS>(algorithm, Int2Type<CDP>(), 1, d_temp_storage_bytes, d_cdp_error, d_temp_storage, temp_storage_bytes, d_samples, d_sample_itr, d_histograms, num_samples, 0, true);
    CubDebugExit(g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes));

    // Run warmup/correctness iteration
    Dispatch<BINS, CHANNELS, ACTIVE_CHANNELS>(algorithm, Int2Type<CDP>(), 1, d_temp_storage_bytes, d_cdp_error, d_temp_storage, temp_storage_bytes, d_samples, d_sample_itr, d_histograms, num_samples, 0, true);

    // Check for correctness (and display results, if specified)
    compare = CompareDeviceResults((HistoCounterT*) h_reference_linear, d_histograms_linear, total_bins, g_verbose, g_verbose);
    printf("%s", compare ? "FAIL" : "PASS");

    // Flush any stdout/stderr
    CubDebugExit(cudaDeviceSynchronize());
    fflush(stdout);
    fflush(stderr);

    // Performance
    GpuTimer gpu_timer;
    gpu_timer.Start();
    Dispatch<BINS, CHANNELS, ACTIVE_CHANNELS>(algorithm, Int2Type<CDP>(), g_timing_iterations, d_temp_storage_bytes, d_cdp_error, d_temp_storage, temp_storage_bytes, d_samples, d_sample_itr, d_histograms, num_samples, 0, false);
    gpu_timer.Stop();
    float elapsed_millis = gpu_timer.ElapsedMillis();

    // Display performance
    if (g_timing_iterations > 0)
    {
        float avg_millis = elapsed_millis / g_timing_iterations;
        float grate = float(num_samples) / avg_millis / 1000.0 / 1000.0;
        float gbandwidth = grate * sizeof(SampleT);
        printf(", %.3f avg ms, %.3f billion samples/s, %.3f billion bins/s, %.3f billion pixels/s, %.3f logical GB/s",
            avg_millis,
            grate,
            grate * ACTIVE_CHANNELS / CHANNELS,
            grate / CHANNELS,
            gbandwidth);
    }

    printf("\n\n");

    // Cleanup
    CubDebugExit(tex_itr.UnbindTexture());
    if (h_samples) delete[] h_samples;
    if (h_reference_linear) delete[] h_reference_linear;
    if (d_samples) CubDebugExit(g_allocator.DeviceFree(d_samples));
    if (d_histograms_linear) CubDebugExit(g_allocator.DeviceFree(d_histograms_linear));
    if (d_temp_storage_bytes) CubDebugExit(g_allocator.DeviceFree(d_temp_storage_bytes));
    if (d_cdp_error) CubDebugExit(g_allocator.DeviceFree(d_cdp_error));
    if (d_temp_storage) CubDebugExit(g_allocator.DeviceFree(d_temp_storage));

    // Correctness asserts
    AssertEquals(0, compare);
    AssertEquals(0, cdp_compare);
}


/**
 * Test different dispatch
 */
template <
    int                         BINS,
    int                         CHANNELS,
    int                         ACTIVE_CHANNELS,
    typename                    SampleT,
    typename                    IteratorValue,
    typename                    HistoCounterT,
    typename                    BinOp,
    int                         ALGORITHM>
void TestCnp(
    Int2Type<ALGORITHM>         algorithm,
    GenMode                     gen_mode,
    BinOp                       bin_op,
    int                         num_samples,
    char*                       type_string)
{
    Test<false, BINS, CHANNELS, ACTIVE_CHANNELS, SampleT, IteratorValue, HistoCounterT>(algorithm, gen_mode, bin_op, num_samples, type_string);
#ifdef CUB_CDP
    Test<true, BINS, CHANNELS, ACTIVE_CHANNELS, SampleT, IteratorValue, HistoCounterT>(algorithm, gen_mode, bin_op, num_samples, type_string);
#endif
}


/**
 * Test different algorithms
 */
template <
    int             BINS,
    int             CHANNELS,
    int             ACTIVE_CHANNELS,
    typename        SampleT,
    typename        IteratorValue,
    typename        HistoCounterT,
    typename        BinOp>
void Test(
    GenMode         gen_mode,
    BinOp           bin_op,
    int             num_samples,
    char*           type_string)
{
    TestCnp<BINS, CHANNELS, ACTIVE_CHANNELS, SampleT, IteratorValue, HistoCounterT>(Int2Type<DEVICE_HISTO_SORT>(),          gen_mode, bin_op, num_samples * CHANNELS, type_string);
    TestCnp<BINS, CHANNELS, ACTIVE_CHANNELS, SampleT, IteratorValue, HistoCounterT>(Int2Type<DEVICE_HISTO_SHARED_ATOMIC>(), gen_mode, bin_op, num_samples * CHANNELS, type_string);
    TestCnp<BINS, CHANNELS, ACTIVE_CHANNELS, SampleT, IteratorValue, HistoCounterT>(Int2Type<DEVICE_HISTO_GLOBAL_ATOMIC>(), gen_mode, bin_op, num_samples * CHANNELS, type_string);
}


/**
 * Iterate over different channel configurations
 */
template <
    int             BINS,
    typename        SampleT,
    typename        IteratorValue,
    typename        HistoCounterT,
    typename        BinOp>
void Test(
    GenMode         gen_mode,
    BinOp           bin_op,
    int             num_samples,
    char*           type_string)
{
    Test<BINS, 1, 1, SampleT, IteratorValue, HistoCounterT>(gen_mode, bin_op, num_samples, type_string);
    Test<BINS, 4, 3, SampleT, IteratorValue, HistoCounterT>(gen_mode, bin_op, num_samples, type_string);
}


/**
 * Iterate over different gen modes
 */
template <
    int             BINS,
    typename        SampleT,
    typename        IteratorValue,
    typename        HistoCounterT,
    typename        BinOp>
void TestModes(
    BinOp           bin_op,
    int             num_samples,
    char*           type_string)
{
    Test<BINS, SampleT, IteratorValue, HistoCounterT>(RANDOM, bin_op, num_samples, type_string);
    Test<BINS, SampleT, IteratorValue, HistoCounterT>(UNIFORM, bin_op, num_samples, type_string);
}


/**
 * Iterate input sizes
 */
template <
    int             BINS,
    typename        SampleT,
    typename        IteratorValue,
    typename        HistoCounterT,
    typename        BinOp>
void Test(
    BinOp           bin_op,
    int             num_samples,
    char*           type_string)
{
    if (num_samples < 0)
    {
        TestModes<BINS, SampleT, IteratorValue, HistoCounterT>(bin_op, 1,       type_string);
        TestModes<BINS, SampleT, IteratorValue, HistoCounterT>(bin_op, 100,     type_string);
        TestModes<BINS, SampleT, IteratorValue, HistoCounterT>(bin_op, 10000,   type_string);
        TestModes<BINS, SampleT, IteratorValue, HistoCounterT>(bin_op, 1000000, type_string);
    }
    else
    {
        TestModes<BINS, SampleT, IteratorValue, HistoCounterT>(bin_op, num_samples, type_string);
    }
}



//---------------------------------------------------------------------
// Main
//---------------------------------------------------------------------

/**
 * Main
 */
int main(int argc, char** argv)
{
    int num_samples = -1;

    // Initialize command line
    CommandLineArgs args(argc, argv);
    g_verbose = args.CheckCmdLineFlag("v");
    args.GetCmdLineArgument("n", num_samples);
    args.GetCmdLineArgument("i", g_timing_iterations);
    args.GetCmdLineArgument("repeat", g_repeat);

    // Print usage
    if (args.CheckCmdLineFlag("help"))
    {
        printf("%s "
            "[--n=<total input pixels> "
            "[--i=<timing iterations> "
            "[--device=<device-id>] "
            "[--repeat=<repetitions of entire test suite>]"
            "[--v] "
            "[--cdp]"
            "\n", argv[0]);
        exit(0);
    }

    // Initialize device
    CubDebugExit(args.DeviceInit());

#ifdef QUICK_TEST

    // Compile/run quick tests
    if (num_samples < 0) num_samples = 32000000;

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

#else

    // Compile/run thorough tests
    for (int i = 0; i <= g_repeat; ++i)
    {
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
    }

#endif

    return 0;
}



