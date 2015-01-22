/******************************************************************************
 * Copyright (c) 2011-2015, NVIDIA CORPORATION.  All rights reserved.
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

#include <stdio.h>
#include <map>
#include <vector>
#include <algorithm>

#include "histogram/readers.h"
#include "histogram/histogram_gmem_atomics.h"
#include "histogram/histogram_smem_atomics.h"
#include "histogram/histogram_cub.h"

#include <cub/util_allocator.cuh>
#include <cub/device/device_scan.cuh>
#include <test/test_util.h>

using namespace cub;

//---------------------------------------------------------------------
// Globals, constants, and type declarations
//---------------------------------------------------------------------

// Ensure printing of CUDA runtime errors to console
#define CUB_STDERR

bool                    g_verbose = false;  // Whether to display input/output to console
CachingDeviceAllocator  g_allocator(true);  // Caching allocator for device memory

struct less_than_value
{
    inline bool operator()(
        const std::pair<std::string, double> &a,
        const std::pair<std::string, double> &b)
    {
        return a.second < b.second;
    }
};


//---------------------------------------------------------------------
// Test generation
//---------------------------------------------------------------------


// Compute reference histogram.
template <
    int         ACTIVE_CHANNELS,
    int         NUM_BINS,
    typename    PixelType>
void histogram_gold(PixelType *image, int width, int height, unsigned int* hist);


// Compute reference histogram.  Specialized for uchar4
template <
    int         ACTIVE_CHANNELS,
    int         NUM_BINS>
void histogram_gold(uchar4 *image, int width, int height, unsigned int* hist)
{
    memset(hist, 0, ACTIVE_CHANNELS * NUM_BINS * sizeof(unsigned int));

    for (int i = 0; i < width; i++)
    {
        for (int j = 0; j < height; j++)
        {
            uchar4 pixel = image[i + j * width];

            unsigned int r_bin = (unsigned int) pixel.x;
            unsigned int g_bin = (unsigned int) pixel.y;
            unsigned int b_bin = (unsigned int) pixel.z;
            unsigned int a_bin = (unsigned int) pixel.w;

            if (ACTIVE_CHANNELS > 0)
                hist[(NUM_BINS * 0) + r_bin]++;
            if (ACTIVE_CHANNELS > 1)
                hist[(NUM_BINS * 1) + g_bin]++;
            if (ACTIVE_CHANNELS > 2)
                hist[(NUM_BINS * 2) + b_bin]++;
            if (ACTIVE_CHANNELS > 3)
                hist[(NUM_BINS * 3) + a_bin]++;
        }
    }
}


// Compute reference histogram.  Specialized for float4
template <
    int         ACTIVE_CHANNELS,
    int         NUM_BINS>
void histogram_gold(float4 *image, int width, int height, unsigned int* hist)
{
    memset(hist, 0, ACTIVE_CHANNELS * NUM_BINS * sizeof(unsigned int));

    for (int i = 0; i < width; i++)
    {
        for (int j = 0; j < height; j++)
        {
            float4 pixel = image[i + j * width];

            unsigned int r_bin = (unsigned int) (pixel.x * NUM_BINS);
            unsigned int g_bin = (unsigned int) (pixel.y * NUM_BINS);
            unsigned int b_bin = (unsigned int) (pixel.z * NUM_BINS);
            unsigned int a_bin = (unsigned int) (pixel.w * NUM_BINS);

            if (ACTIVE_CHANNELS > 0)
                hist[(NUM_BINS * 0) + r_bin]++;
            if (ACTIVE_CHANNELS > 1)
                hist[(NUM_BINS * 1) + g_bin]++;
            if (ACTIVE_CHANNELS > 2)
                hist[(NUM_BINS * 2) + b_bin]++;
            if (ACTIVE_CHANNELS > 3)
                hist[(NUM_BINS * 3) + a_bin]++;
        }
    }
}


/**
 * Run a specific histogram implementation
 */
template <
    int         ACTIVE_CHANNELS,
    int         NUM_BINS,
    typename    PixelType>
void RunTest(
    std::vector<std::pair<std::string, double> >&   timings,
    PixelType*                                      d_pixels,
    const int                                       width,
    const int                                       height,
    unsigned int *                                  d_hist,
    unsigned int *                                  h_hist,
    int                                             timing_iterations,
    const char *                                    long_name,
    const char *                                    short_name,
    double (*f)(PixelType*, int, int, unsigned int*, bool))
{
    printf("%s ", long_name);

    // Run single test to verify (and code cache)
    (*f)(d_pixels, width, height, d_hist, true);

    int compare = CompareDeviceResults(h_hist, d_hist, ACTIVE_CHANNELS * NUM_BINS, true, g_verbose);
    printf("\t%s\n", compare ? "FAIL" : "PASS");

    double elapsed_time = 0;
    for (int i = 0; i < timing_iterations; i++)
    {
        elapsed_time += (*f)(d_pixels, width, height, d_hist, false);
    }
    double avg_time = elapsed_time /= timing_iterations;    // average
    timings.push_back(std::pair<std::string, double>(short_name, avg_time));

    printf("Avg time %.3f ms (%d iterations)\n", avg_time, timing_iterations);
    AssertEquals(0, compare);

}


/**
 * Evaluate a variety of different histogram implementations
 */
template <
    int         ACTIVE_CHANNELS,
    int         NUM_BINS,
    typename    PixelType>
void RunTests(
    PixelType*  h_pixels,
    int         height,
    int         width,
    int         timing_iterations)
{
    // Copy data to gpu
    PixelType* d_pixels;
    size_t pixel_bytes = width * height * sizeof(PixelType);
    CubDebugExit(g_allocator.DeviceAllocate((void**) &d_pixels, pixel_bytes));
    CubDebugExit(cudaMemcpy(d_pixels, h_pixels, pixel_bytes, cudaMemcpyHostToDevice));

    // Allocate results arrays on cpu/gpu
    unsigned int *h_hist;
    unsigned int *d_hist;
    size_t channel_bytes = NUM_BINS * sizeof(unsigned int);
    h_hist = (unsigned int *) malloc(channel_bytes * ACTIVE_CHANNELS);
    g_allocator.DeviceAllocate((void **) &d_hist, channel_bytes * ACTIVE_CHANNELS);

    // Compute reference cpu histogram
    histogram_gold<ACTIVE_CHANNELS, NUM_BINS>(h_pixels, width, height, h_hist);

    // Store timings
    std::vector<std::pair<std::string, double> > timings;

    // Run experiments
    RunTest<ACTIVE_CHANNELS, NUM_BINS>(timings, d_pixels, width, height, d_hist, h_hist, timing_iterations,
        "Global memory atomics", "gmem atomics", run_gmem_atomics<ACTIVE_CHANNELS, NUM_BINS, PixelType>);
    RunTest<ACTIVE_CHANNELS, NUM_BINS>(timings, d_pixels, width, height, d_hist, h_hist, timing_iterations,
        "Shared memory atomics", "smem atomics", run_smem_atomics<ACTIVE_CHANNELS, NUM_BINS, PixelType>);
    RunTest<ACTIVE_CHANNELS, NUM_BINS>(timings, d_pixels, width, height, d_hist, h_hist, timing_iterations,
        "CUB", "CUB", run_cub_histogram<ACTIVE_CHANNELS, NUM_BINS, PixelType>);

    // Report timings
    std::sort(timings.begin(), timings.end(), less_than_value());
    printf("Timings (ms):\n");
    for (int i = 0; i < timings.size(); i++)
    {
        double bandwidth = height * width * sizeof(PixelType) / timings[i].second / 1000 / 1000;
        printf("  %.3f %s (%.3f GB/s)\n", timings[i].second, timings[i].first.c_str(), bandwidth);
    }

    // Free data
    CubDebugExit(g_allocator.DeviceFree(d_pixels));
    CubDebugExit(g_allocator.DeviceFree(d_hist));
    free(h_hist);
}


/**
 * Main
 */
int main(int argc, char **argv)
{
    enum {
        ACTIVE_CHANNELS = 3,
        NUM_BINS        = 256

    };

    // Initialize command line
    CommandLineArgs     args(argc, argv);
    int                 timing_iterations   = 100;
    std::string         filename;
    int                 height              = -1;
    int                 width               = -1;
    g_verbose = args.CheckCmdLineFlag("v");
    args.GetCmdLineArgument("i", timing_iterations);
    args.GetCmdLineArgument("file", filename);
    args.GetCmdLineArgument("height", height);
    args.GetCmdLineArgument("width", width);

    // Print usage
    if (args.CheckCmdLineFlag("help"))
    {
        printf("%s "
            "[--device=<device-id>] "
            "[--v] "
            "[--i=<timing iterations>] "
            "--file=<.tga filename> "
            "[--height=<binfile height>] "
            "[--width=<binfile width>] "
            "\n", argv[0]);
        exit(0);
    }

    // Initialize device
    CubDebugExit(args.DeviceInit());

    // Parse targa file
    uchar4* byte_pixels = NULL;
    ReadTga(byte_pixels, width, height, filename.c_str());
/*
    // uchar4 tests
    printf("Targa (uchar4):\n");
    RunTests<ACTIVE_CHANNELS, NUM_BINS>(byte_pixels, width, height, timing_iterations);
*/
    // Convert uchar4 to float4 pixels
    float4* float_pixels = NULL;
    if ((float_pixels = (float4*) malloc(width * height * sizeof(float4))) == NULL)
    {
        fprintf(stderr, "malloc of image failed\n");
        exit(-1);
    }
    for (int i = 0; i < width * height; ++i)
    {
        float_pixels[i].x = ((float) byte_pixels[i].x) / NUM_BINS;
        float_pixels[i].y = ((float) byte_pixels[i].y) / NUM_BINS;
        float_pixels[i].z = ((float) byte_pixels[i].z) / NUM_BINS;
        float_pixels[i].w = ((float) byte_pixels[i].w) / NUM_BINS;
    }

    // float4 tests
    printf("\n\nTarga (float4):\n");
    RunTests<ACTIVE_CHANNELS, NUM_BINS>(float_pixels, width, height, timing_iterations);

    // Free pixel data
    free(float_pixels);
    free(byte_pixels);

    CubDebugExit(cudaDeviceSynchronize());
    printf("\n\n");

    return 0;
}
