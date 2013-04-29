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
 * Test of DeviceHisto256 utilities
 ******************************************************************************/

// Ensure printing of CUDA runtime errors to console
#define CUB_STDERR

#include <stdio.h>
#include <cub/cub.cuh>
#include "test_util.h"

using namespace cub;


//---------------------------------------------------------------------
// Globals, constants and typedefs
//---------------------------------------------------------------------

bool    g_verbose       = false;
int     g_iterations    = 100;
bool    g_atomic        = false;


//---------------------------------------------------------------------
// CUDA Nested Parallelism Test Kernel
//---------------------------------------------------------------------

/**
 * Simple wrapper kernel to invoke DeviceHisto256
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

#if CUB_CNP_ENABLED
    for (int i = 0; i < iterations; ++i)
    {
        error = DeviceHisto256::SingleChannel(d_samples, d_out, num_samples, reduction_op, 0, STREAM_SYNCHRONOUS);
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
 * Initialize problem (and solution)
 */
template <
    int             CHANNELS,
    int             ACTIVE_CHANNELS,
    typename        SampleType,
    typename        BinOp,
    typename        HistoCounter>
void Initialize(
    int             gen_mode,
    SampleType      *h_samples,
    BinOp           bin_op,
    HistoCounter    *h_histograms_linear,
    int             num_samples)
{
    // Init bins
    for (int bin = 0; bin < ACTIVE_CHANNELS * 256; ++bin)
    {
        h_histograms_linear[bin] = 0;
    }

    if (g_verbose) printf("Samples: \n");

    // Initialize interleaved channel samples and histogram them correspondingly
    for (int i = 0; i < num_samples; ++i)
    {
        InitValue(gen_mode, h_samples[i], i);

        unsigned char bin = bin_op(h_samples[i]);
        int channel = i % CHANNELS;

        if (g_verbose)
        {
            if (channel == 0) printf("<");
            if (channel == CHANNELS - 1)
                printf("%d>, ", (int) h_samples[i]);
            else
                printf("%d, ", (int) h_samples[i]);
        }

        if (channel < ACTIVE_CHANNELS)
        {
            h_histograms_linear[(channel * 256) + bin]++;
        }
    }

    if (g_verbose) printf("\n\n");
}


//---------------------------------------------------------------------
// Full tile test generation
//---------------------------------------------------------------------


/**
 * Test DeviceHisto256
 */
template <
    int             CHANNELS,
    int             ACTIVE_CHANNELS,
    typename        SampleType,
    typename        HistoCounter,
    typename        BinOp>
void Test(
    int             gen_mode,
    BinOp           bin_op,
    int             num_samples,
    char*           type_string)
{
    int compare         = 0;
    int cnp_compare     = 0;
    int total_bins      = ACTIVE_CHANNELS * 256;

    printf("cub::DeviceHisto256 %d %s samples (%dB), %d channels, %d active channels, gen-mode %d\n\n",
        num_samples,
        type_string,
        (int) sizeof(SampleType),
        CHANNELS,
        ACTIVE_CHANNELS,
        gen_mode);
    fflush(stdout);

    // Allocate host arrays
    SampleType      *h_samples = new SampleType[num_samples];
    HistoCounter    *h_reference_linear = new HistoCounter[total_bins];

    // Initialize problem
    Initialize<CHANNELS, ACTIVE_CHANNELS>(gen_mode, h_samples, bin_op, h_reference_linear, num_samples);

    // Allocate device arrays
    SampleType*     d_samples = NULL;
    HistoCounter*   d_histograms_linear = NULL;
    cudaError_t*    d_cnp_error = NULL;
    CubDebugExit(DeviceAllocate((void**)&d_samples,             sizeof(SampleType) * num_samples));
    CubDebugExit(DeviceAllocate((void**)&d_histograms_linear,   sizeof(HistoCounter) * total_bins));
    CubDebugExit(DeviceAllocate((void**)&d_cnp_error,           sizeof(cudaError_t) * 1));

    // Initialize device arrays
    CubDebugExit(cudaMemcpy(d_samples, h_samples, sizeof(SampleType) * num_samples, cudaMemcpyHostToDevice));
    CubDebugExit(cudaMemset(d_histograms_linear, 0, sizeof(HistoCounter) * total_bins));

    // Structure of channel histograms
    HistoCounter *d_histograms[ACTIVE_CHANNELS];
    for (int CHANNEL = 0; CHANNEL < ACTIVE_CHANNELS; ++CHANNEL)
    {
        d_histograms[CHANNEL] = d_histograms_linear + (CHANNEL * 256);
    }

    // Run warmup/correctness iteration
    printf("Host dispatch:\n"); fflush(stdout);
    if (g_atomic)
    {
        CubDebugExit(DeviceHisto256::MultiChannelAtomic<CHANNELS>(d_samples, d_histograms, num_samples, 0, true));
    }
    else
    {
        CubDebugExit(DeviceHisto256::MultiChannel<CHANNELS>(d_samples, d_histograms, num_samples, 0, true));
    }

    // Check for correctness (and display results, if specified)
    compare = CompareDeviceResults((HistoCounter*) h_reference_linear, d_histograms_linear, total_bins, g_verbose, g_verbose);
    printf("\n%s", compare ? "FAIL" : "PASS");

    // Flush any stdout/stderr
    fflush(stdout);
    fflush(stderr);

    // Performance
    GpuTimer gpu_timer;
    float elapsed_millis = 0.0;
    for (int i = 0; i < g_iterations; i++)
    {
        gpu_timer.Start();

        if (g_atomic)
        {
            CubDebugExit(DeviceHisto256::MultiChannelAtomic<CHANNELS>(d_samples, d_histograms, num_samples));
        }
        else
        {
            CubDebugExit(DeviceHisto256::MultiChannel<CHANNELS>(d_samples, d_histograms, num_samples));
        }

        gpu_timer.Stop();
        elapsed_millis += gpu_timer.ElapsedMillis();
    }
    if (g_iterations > 0)
    {
        float avg_millis = elapsed_millis / g_iterations;
        float grate = float(num_samples) / avg_millis / 1000.0 / 1000.0;
        float gbandwidth = grate * sizeof(SampleType);
        printf(", %.3f avg ms, %.3f billion samples/s, %.3f billion bins/s, %.3f billion pixels/s, %.3f GB/s\n",
            avg_millis,
            grate,
            grate * ACTIVE_CHANNELS / CHANNELS,
            grate / CHANNELS,
            gbandwidth);
    }
    else
    {
        printf("\n");
    }


    // Evaluate using CUDA nested parallelism
#if (TEST_CNP == 1)

    CubDebugExit(cudaMemset(d_out, 0, sizeof(SampleType) * 1));

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
        printf("CNP reduction not supported");
    }
    else
    {
        CubDebugExit(h_cnp_error);

        // Check for correctness (and display results, if specified)
        cnp_compare = CompareDeviceResults(h_reference_linear, d_out, 1, g_verbose, g_verbose);
        printf("\n%s", cnp_compare ? "FAIL" : "PASS");

        // Performance
        gpu_timer.Start();

        CnpHisto<false><<<1,1>>>(d_samples, d_out, num_samples, reduction_op, g_iterations, d_cnp_error);

        gpu_timer.Stop();
        elapsed_millis = gpu_timer.ElapsedMillis();

        if (g_iterations > 0)
        {
            float avg_millis = elapsed_millis / g_iterations;
            float grate = float(num_samples) / avg_millis / 1000.0 / 1000.0;
            float gbandwidth = grate * sizeof(SampleType);
            printf(", %.3f avg ms, %.3f billion items/s, %.3f GB/s\n", avg_millis, grate, gbandwidth);
        }
        else
        {
            printf("\n");
        }
    }

#endif

    // Cleanup
    if (h_samples) delete[] h_samples;
    if (h_reference_linear) delete[] h_reference_linear;
    if (d_samples) CubDebugExit(DeviceFree(d_samples));
    if (d_histograms_linear) CubDebugExit(DeviceFree(d_histograms_linear));
    if (d_cnp_error) CubDebugExit(DeviceFree(d_cnp_error));

    // Correctness asserts
    AssertEquals(0, compare);
    AssertEquals(0, cnp_compare);
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
    args.GetCmdLineArgument("n", num_samples);
    args.GetCmdLineArgument("i", g_iterations);
    g_verbose = args.CheckCmdLineFlag("v");             // Display input/output data
    g_atomic = args.CheckCmdLineFlag("atomic");         // Use atomic or regular (sorting) algorithm
    bool rgba = args.CheckCmdLineFlag("rgba");          // Single channel vs. 4-channel (3 histograms)
    bool uniform = args.CheckCmdLineFlag("uniform");    // Random data vs. uniform (homogeneous)

    // Print usage
    if (args.CheckCmdLineFlag("help"))
    {
        printf("%s "
            "[--device=<device-id>] "
            "[--v] "
            "[--cnp]"
            "[--rgba]"
            "[--uniform]"
            "\n", argv[0]);
        exit(0);
    }

    // Initialize device
    CubDebugExit(args.DeviceInit());

    if (rgba)
    {
        // Quad samples, first three channels active
        Test<4, 3, unsigned char, int>(
            (uniform) ? UNIFORM : RANDOM,
            Cast<unsigned char>(),
            num_samples,
            CUB_TYPE_STRING(unsigned char));
    }
    else
    {
        // Single stream of byte sample data
        Test<1, 1, unsigned char, int>(
            (uniform) ? UNIFORM : RANDOM,
            Cast<unsigned char>(),
            num_samples,
            CUB_TYPE_STRING(unsigned char));
    }

    return 0;
}



