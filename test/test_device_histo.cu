/******************************************************************************
 * Copyright (c) 2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2013, NVIDIA CORPORATION.  All rights reserved.
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
 * Test of DeviceHisto utilities
 ******************************************************************************/

// Ensure printing of CUDA runtime errors to console
#define CUB_STDERR

#include <stdio.h>
#include <limits>
#include <string>

#include <cub/cub.cuh>
#include "test_util.h"

using namespace cub;


//---------------------------------------------------------------------
// Globals, constants and typedefs
//---------------------------------------------------------------------

bool                        g_verbose           = false;
int                         g_timing_iterations = 0;
bool                        g_verbose_input     = false;
CachingDeviceAllocator      g_allocator;


/**
 * Scaling operator for binning f32 types
 */
template <typename T, int BINS>
struct FloatScaleOp
{
    __host__ __device__ __forceinline__ T operator()(float datum)
    {
        float datum_scale = datum * float(BINS - 1);
        return (T) datum_scale;
    }
};



//---------------------------------------------------------------------
// CUDA Nested Parallelism Test Kernel
//---------------------------------------------------------------------

/**
 * Simple wrapper kernel to invoke DeviceHisto
 */
/*
template <
    bool                STREAM_SYNCHRONOUS,
    typename            InputIteratorRA,
    typename            OutputIteratorRA,
    typename            ReductionOp>
__global__ void CnpHisto(
    InputIteratorRA     d_samples,
    OutputIteratorRA    d_out,
    int                 num_samples,
    ReductionOp         reduction_op,
    int                 iterations,
    cudaError_t*        d_cnp_error)
{
    cudaError_t error = cudaSuccess;

#ifdef CUB_RUNTIME_ENABLED
    for (int i = 0; i < iterations; ++i)
    {
        error = DeviceHisto::SingleChannel(d_samples, d_out, num_samples, reduction_op, 0, STREAM_SYNCHRONOUS);
    }
#else
    error = cudaErrorInvalidConfiguration;
#endif

    *d_cnp_error = error;
}
*/

//---------------------------------------------------------------------
// Host utility subroutines
//---------------------------------------------------------------------

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
    typename        IteratorValueT,
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

        IteratorValueT bin = bin_op(h_samples[i]);

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


//---------------------------------------------------------------------
// Full tile test generation
//---------------------------------------------------------------------


/**
 * Test DeviceHisto
 */
template <
    int             BINS,
    int             CHANNELS,
    int             ACTIVE_CHANNELS,
    typename        SampleT,
    typename        IteratorValueT,
    typename        HistoCounterT,
    typename        BinOp>
void Test(
    BlockSweepHistoAlgorithm    g_algorithm,
    GenMode                     gen_mode,
    BinOp                       bin_op,
    int                         num_samples,
    char*                       type_string)
{
    // Binning iterator type for loading through tex and applying a transform operator
    typedef TexTransformIteratorRA<IteratorValueT, BinOp, SampleT> BinningIterator;

    int compare         = 0;
    int cnp_compare     = 0;
    int total_bins      = ACTIVE_CHANNELS * BINS;

    printf("cub::DeviceHisto %s %d %s samples (%dB), %d bins, %d channels, %d active channels, gen-mode %d\n",
        (g_algorithm == GRID_HISTO_SHARED_ATOMIC) ? "satomic" : (g_algorithm == GRID_HISTO_GLOBAL_ATOMIC) ? "gatomic" : "sort",
        num_samples,
        type_string,
        (int) sizeof(SampleT),
        BINS,
        CHANNELS,
        ACTIVE_CHANNELS,
        gen_mode);
    fflush(stdout);

    // Allocate host arrays
    SampleT         *h_samples          = new SampleT[num_samples];
    HistoCounterT   *h_reference_linear = new HistoCounterT[total_bins];

    // Initialize problem
    Initialize<BINS, CHANNELS, ACTIVE_CHANNELS, IteratorValueT>(gen_mode, h_samples, bin_op, h_reference_linear, num_samples);

    // Allocate device arrays
    SampleT*        d_samples = NULL;
    HistoCounterT*  d_histograms_linear = NULL;
    cudaError_t*    d_cnp_error = NULL;
    void            *d_temporary_storage = NULL;
    size_t          temporary_storage_bytes = 0;
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_samples,             sizeof(SampleT) * num_samples));
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_histograms_linear,   sizeof(HistoCounterT) * total_bins));
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_cnp_error,           sizeof(cudaError_t) * 1));

    // Initialize device arrays
    CubDebugExit(cudaMemcpy(d_samples, h_samples, sizeof(SampleT) * num_samples, cudaMemcpyHostToDevice));
    CubDebugExit(cudaMemset(d_histograms_linear, 0, sizeof(HistoCounterT) * total_bins));

    // Structure of channel histograms
    HistoCounterT *d_histograms[ACTIVE_CHANNELS];
    for (int CHANNEL = 0; CHANNEL < ACTIVE_CHANNELS; ++CHANNEL)
    {
        d_histograms[CHANNEL] = d_histograms_linear + (CHANNEL * BINS);
    }

    // Create iterator wrapper for SampleT -> unsigned char conversion
    BinningIterator d_sample_itr(d_samples, bin_op);

    // Run warmup/correctness iteration
    printf("Host dispatch:\n"); fflush(stdout);
    if (g_algorithm == GRID_HISTO_SHARED_ATOMIC)
    {
        // Allocate temporary storage
        CubDebugExit((DeviceHisto::MultiChannelAtomic<BINS, CHANNELS>(d_temporary_storage, temporary_storage_bytes, d_sample_itr, d_histograms, num_samples, 0, true)));
        CubDebugExit(g_allocator.DeviceAllocate(&d_temporary_storage, temporary_storage_bytes));

        // Run test
        CubDebugExit((DeviceHisto::MultiChannelAtomic<BINS, CHANNELS>(d_temporary_storage, temporary_storage_bytes, d_sample_itr, d_histograms, num_samples, 0, true)));
    }
    else if (g_algorithm == GRID_HISTO_GLOBAL_ATOMIC)
    {
        // Allocate temporary storage
        CubDebugExit((DeviceHisto::MultiChannelGlobalAtomic<BINS, CHANNELS>(d_temporary_storage, temporary_storage_bytes, d_sample_itr, d_histograms, num_samples, 0, true)));
        CubDebugExit(g_allocator.DeviceAllocate(&d_temporary_storage, temporary_storage_bytes));

        // Run test
        CubDebugExit((DeviceHisto::MultiChannelGlobalAtomic<BINS, CHANNELS>(d_temporary_storage, temporary_storage_bytes, d_sample_itr, d_histograms, num_samples, 0, true)));
    }
    else
    {
        // Allocate temporary storage
        CubDebugExit((DeviceHisto::MultiChannelSorting<BINS, CHANNELS>(d_temporary_storage, temporary_storage_bytes, d_sample_itr, d_histograms, num_samples, 0, true)));
        CubDebugExit(g_allocator.DeviceAllocate(&d_temporary_storage, temporary_storage_bytes));

        // Run test
        CubDebugExit((DeviceHisto::MultiChannelSorting<BINS, CHANNELS>(d_temporary_storage, temporary_storage_bytes, d_sample_itr, d_histograms, num_samples, 0, true)));
    }

    // Check for correctness (and display results, if specified)
    compare = CompareDeviceResults((HistoCounterT*) h_reference_linear, d_histograms_linear, total_bins, g_verbose, g_verbose);
    printf("%s", compare ? "FAIL" : "PASS");

    // Flush any stdout/stderr
    CubDebugExit(cudaDeviceSynchronize());
    fflush(stdout);
    fflush(stderr);

    // Performance
    GpuTimer gpu_timer;
    float elapsed_millis = 0.0;
    for (int i = 0; i < g_timing_iterations; i++)
    {
        gpu_timer.Start();

        if (g_algorithm == GRID_HISTO_SHARED_ATOMIC)
        {
            CubDebugExit((DeviceHisto::MultiChannelAtomic<BINS, CHANNELS>(d_temporary_storage, temporary_storage_bytes, d_sample_itr, d_histograms, num_samples, 0)));
        }
        else if (g_algorithm == GRID_HISTO_GLOBAL_ATOMIC)
        {
            CubDebugExit((DeviceHisto::MultiChannelGlobalAtomic<BINS, CHANNELS>(d_temporary_storage, temporary_storage_bytes, d_sample_itr, d_histograms, num_samples, 0)));
        }
        else
        {
            CubDebugExit((DeviceHisto::MultiChannelSorting<BINS, CHANNELS>(d_temporary_storage, temporary_storage_bytes, d_sample_itr, d_histograms, num_samples, 0)));
        }

        gpu_timer.Stop();
        elapsed_millis += gpu_timer.ElapsedMillis();
    }
    if (g_timing_iterations > 0)
    {
        float avg_millis = elapsed_millis / g_timing_iterations;
        float grate = float(num_samples) / avg_millis / 1000.0 / 1000.0;
        float gbandwidth = grate * sizeof(SampleT);
        printf(", %.3f avg ms, %.3f billion samples/s, %.3f billion bins/s, %.3f billion pixels/s, %.3f GB/s",
            avg_millis,
            grate,
            grate * ACTIVE_CHANNELS / CHANNELS,
            grate / CHANNELS,
            gbandwidth);
    }

    printf("\n\n");

/*
    // Evaluate using CUDA nested parallelism
#if (TEST_CNP == 1)

    CubDebugExit(cudaMemset(d_out, 0, sizeof(SampleT) * 1));

    // Run warmup/correctness iteration
    printf("\nDevice dispatch:\n"); fflush(stdout);
    CnpHisto<true><<<1,1>>>(d_samples, d_out, num_samples, reduction_op, 1, d_cnp_error);

    // Flush any stdout/stderr
    fflush(stdout);
    fflush(stderr);

    // Check if we were compiled and linked for CNP
    cudaError_t h_cnp_error;
    CubDebugExit(cudaMemcpy(&h_cnp_error, d_cnp_error, sizeof(cudaError_t) * 1, cudaMemcpyDeviceToHost));
    if (h_cnp_error == cudaErrorInvalidConfiguration)
    {
        printf("CNP not supported");
    }
    else
    {
        CubDebugExit(h_cnp_error);

        // Check for correctness (and display results, if specified)
        cnp_compare = CompareDeviceResults(h_reference_linear, d_out, 1, g_verbose, g_verbose);
        printf("\n%s", cnp_compare ? "FAIL" : "PASS");

        // Performance
        gpu_timer.Start();

        CnpHisto<false><<<1,1>>>(d_samples, d_out, num_samples, reduction_op, g_timing_iterations, d_cnp_error);

        gpu_timer.Stop();
        elapsed_millis = gpu_timer.ElapsedMillis();

        if (g_timing_iterations > 0)
        {
            float avg_millis = elapsed_millis / g_timing_iterations;
            float grate = float(num_samples) / avg_millis / 1000.0 / 1000.0;
            float gbandwidth = grate * sizeof(SampleT);
            printf(", %.3f avg ms, %.3f billion items/s, %.3f GB/s\n", avg_millis, grate, gbandwidth);
        }
        else
        {
            printf("\n");
        }
    }

#endif
*/

    // Cleanup
    if (h_samples) delete[] h_samples;
    if (h_reference_linear) delete[] h_reference_linear;
    if (d_samples) CubDebugExit(g_allocator.DeviceFree(d_samples));
    if (d_histograms_linear) CubDebugExit(g_allocator.DeviceFree(d_histograms_linear));
    if (d_cnp_error) CubDebugExit(g_allocator.DeviceFree(d_cnp_error));
    if (d_temporary_storage) CubDebugExit(g_allocator.DeviceFree(d_temporary_storage));

    // Correctness asserts
    AssertEquals(0, compare);
    AssertEquals(0, cnp_compare);
}



/**
 * Iterate over different algorithms
 */
template <
    int             BINS,
    int             CHANNELS,
    int             ACTIVE_CHANNELS,
    typename        SampleT,
    typename        IteratorValueT,
    typename        HistoCounterT,
    typename        BinOp>
void Test(
    GenMode         gen_mode,
    BinOp           bin_op,
    int             num_samples,
    char*           type_string)
{
    Test<BINS, CHANNELS, ACTIVE_CHANNELS, SampleT, IteratorValueT, HistoCounterT>(GRID_HISTO_SORT,          gen_mode, bin_op, num_samples, type_string);
    Test<BINS, CHANNELS, ACTIVE_CHANNELS, SampleT, IteratorValueT, HistoCounterT>(GRID_HISTO_SHARED_ATOMIC, gen_mode, bin_op, num_samples, type_string);
    Test<BINS, CHANNELS, ACTIVE_CHANNELS, SampleT, IteratorValueT, HistoCounterT>(GRID_HISTO_GLOBAL_ATOMIC, gen_mode, bin_op, num_samples, type_string);
}


/**
 * Iterate over different channel configurations
 */
template <
    int             BINS,
    typename        SampleT,
    typename        IteratorValueT,
    typename        HistoCounterT,
    typename        BinOp>
void Test(
    GenMode         gen_mode,
    BinOp           bin_op,
    int             num_samples,
    char*           type_string)
{
    Test<BINS, 1, 1, SampleT, IteratorValueT, HistoCounterT>(gen_mode, bin_op, num_samples, type_string);
    Test<BINS, 4, 3, SampleT, IteratorValueT, HistoCounterT>(gen_mode, bin_op, num_samples, type_string);
}


/**
 * Iterate over different gen modes
 */
template <
    int             BINS,
    typename        SampleT,
    typename        IteratorValueT,
    typename        HistoCounterT,
    typename        BinOp>
void Test(
    BinOp           bin_op,
    int             num_samples,
    char*           type_string)
{
    Test<BINS, SampleT, IteratorValueT, HistoCounterT>(RANDOM, bin_op, num_samples, type_string);
    Test<BINS, SampleT, IteratorValueT, HistoCounterT>(UNIFORM, bin_op, num_samples, type_string);
}



//---------------------------------------------------------------------
// Main
//---------------------------------------------------------------------

/**
 * Main
 */
int main(int argc, char** argv)
{
    int num_samples = 1 * 1024 * 1024;

    // Initialize command line
    CommandLineArgs args(argc, argv);
    args.GetCmdLineArgument("n", num_samples);                  // Total number of samples across all channels
    args.GetCmdLineArgument("i", g_timing_iterations);          // Timing iterations
    g_verbose = args.CheckCmdLineFlag("v");                     // Display input/output data

    // Print usage
    if (args.CheckCmdLineFlag("help"))
    {
        printf("%s "
            "[--device=<device-id>] "
            "[--v] "
            "[--n=<total number of samples across all channels>]"
            "[--i=<timing iterations>]"
            "\n", argv[0]);
        exit(0);
    }

    // Initialize device
    CubDebugExit(args.DeviceInit());

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

    return 0;
}



