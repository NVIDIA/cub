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
 * Simple example of DeviceHistogram::MultiChannelSharedAtomic().
 *
 * Computes three 256-bin RGB histograms from quad-channel pixels (interleaved
 * 8b RGBA samples) using an algorithm based upon shared-memory atomic instructions.
 *
 * To compile using the command line:
 *   nvcc -arch=sm_XX example_device_histogram.cu -I../.. -lcudart -O3
 *
 ******************************************************************************/

// Ensure printing of CUDA runtime errors to console
#define CUB_STDERR

#include <stdio.h>
#include <limits>

#include <cub/util_allocator.cuh>
#include <cub/iterator/tex_ref_input_iterator.cuh>
#include <cub/device/device_histogram.cuh>

#include "../../test/test_util.h"

using namespace cub;


//---------------------------------------------------------------------
// Globals, constants and typedefs
//---------------------------------------------------------------------

bool                    g_verbose = false;  // Whether to display input/output to console
CachingDeviceAllocator  g_allocator(true);  // Caching allocator for device memory


//---------------------------------------------------------------------
// Test generation
//---------------------------------------------------------------------

/**
 * Initialize problem (and solution)
 */
template <
    int             BINS,
    int             CHANNELS,
    int             ACTIVE_CHANNELS>
void Initialize(
    unsigned char   *h_samples,
    int             h_histograms[ACTIVE_CHANNELS][BINS],
    int             num_pixels)
{
    // Init bins
    for (int channel = 0; channel < ACTIVE_CHANNELS; ++channel)
        for (int bin = 0; bin < BINS; ++bin)
            h_histograms[channel][bin] = 0;

    if (g_verbose) printf("Samples: \n");

    // Initialize interleaved channel samples and histogram them correspondingly
    for (int i = 0; i < num_pixels * CHANNELS; i += CHANNELS)
    {
        if (g_verbose) std::cout << "<";

        for (int channel = 0; channel < ACTIVE_CHANNELS; ++channel)
        {
            RandomBits(h_samples[i + channel]);
            h_histograms[channel][h_samples[i + channel]]++;
            if (g_verbose) {
                std::cout << int(h_samples[i + channel]);
                if (channel < CHANNELS - 1) std::cout << ",";
            }
        }

        for (int channel = ACTIVE_CHANNELS; channel < CHANNELS; ++channel)
        {
            RandomBits(h_samples[i + channel]);
            if (g_verbose) {
                std::cout << int(h_samples[i + channel]);
                if (channel < CHANNELS - 1) std::cout << ",";
            }
        }

        if (g_verbose) std::cout << "> ";
    }

    if (g_verbose) printf("\n\n");
}


//---------------------------------------------------------------------
// Main
//---------------------------------------------------------------------

/**
 * Main
 */
int main(int argc, char** argv)
{
    const int CHANNELS          = 4;    /// Four interleaved RGBA 8b samples per pixel
    const int ACTIVE_CHANNELS   = 3;    /// We want histograms for only RGB channels
    const int BINS              = 256;  /// 256 bins per sample

    int num_pixels = 150;

    // Initialize command line
    CommandLineArgs args(argc, argv);
    g_verbose = args.CheckCmdLineFlag("v");
    bool quick = args.CheckCmdLineFlag("quick");
    args.GetCmdLineArgument("n", num_pixels);

    // Print usage
    if (args.CheckCmdLineFlag("help"))
    {
        printf("%s "
            "[--n=<number of RGBA pixels> "
            "[--device=<device-id>] "
            "[--v] "
            "\n", argv[0]);
        exit(0);
    }

    // Initialize device
    CubDebugExit(args.DeviceInit());

    int num_samples = num_pixels * CHANNELS;

    printf("cub::DeviceHistogram::MultiChannelSharedAtomic() RGB histograms from %d RGBA pixels (%d %d-byte channel samples)\n",
        num_pixels, num_samples, (int) sizeof(unsigned char));
    fflush(stdout);

    int total_bins = ACTIVE_CHANNELS * BINS;

    // Allocate host arrays
    unsigned char   *h_samples = new unsigned char[num_samples];
    int             h_histograms[ACTIVE_CHANNELS][BINS];

    // Initialize problem
    Initialize<BINS, CHANNELS, ACTIVE_CHANNELS>(h_samples, h_histograms, num_pixels);

    // Allocate problem device arrays
    unsigned char   *d_samples = NULL;
    int             *d_histograms_linear = NULL;
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_samples,             sizeof(unsigned char) * num_samples));
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_histograms_linear,   sizeof(int) * total_bins));

    // Initialize device arrays
    CubDebugExit(cudaMemcpy(d_samples, h_samples, sizeof(unsigned char) * num_samples, cudaMemcpyHostToDevice));

    // Structure of channel histograms
    int *d_histograms[ACTIVE_CHANNELS];
    for (int CHANNEL = 0; CHANNEL < ACTIVE_CHANNELS; ++CHANNEL)
        d_histograms[CHANNEL] = d_histograms_linear + (CHANNEL * BINS);

    // Texture iterator type for loading samples through tex
    TexRefInputIterator<unsigned char, __LINE__, int> d_sample_itr;
    CubDebugExit(d_sample_itr.BindTexture(d_samples, sizeof(unsigned char) * num_samples));

    // Allocate temporary storage
    void            *d_temp_storage = NULL;
    size_t          temp_storage_bytes = 0;
    CubDebugExit((DeviceHistogram::MultiChannelSharedAtomic<BINS, CHANNELS, ACTIVE_CHANNELS>(d_temp_storage, temp_storage_bytes, d_sample_itr, d_histograms, num_samples)));
    CubDebugExit(g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes));

    // Run
    CubDebugExit((DeviceHistogram::MultiChannelSharedAtomic<BINS, CHANNELS, ACTIVE_CHANNELS>(d_temp_storage, temp_storage_bytes, d_sample_itr, d_histograms, num_samples)));

    // Check for correctness (and display results, if specified)
    printf("R channel: ");
    int compare = CompareDeviceResults(h_histograms[0], d_histograms[0], BINS, g_verbose, g_verbose);
    printf("%s\n", compare ? "FAIL" : "PASS");
    AssertEquals(0, compare);

    printf("G channel: ");
    compare = CompareDeviceResults(h_histograms[1], d_histograms[1], BINS, g_verbose, g_verbose);
    printf("%s\n", compare ? "FAIL" : "PASS");
    AssertEquals(0, compare);

    printf("B channel: ");
    compare = CompareDeviceResults(h_histograms[2], d_histograms[2], BINS, g_verbose, g_verbose);
    printf("%s\n", compare ? "FAIL" : "PASS");
    AssertEquals(0, compare);

    // Cleanup
    CubDebugExit(d_sample_itr.UnbindTexture());
    if (h_samples) delete[] h_samples;
    if (d_samples) CubDebugExit(g_allocator.DeviceFree(d_samples));
    if (d_histograms_linear) CubDebugExit(g_allocator.DeviceFree(d_histograms_linear));
    if (d_temp_storage) CubDebugExit(g_allocator.DeviceFree(d_temp_storage));

    printf("\n\n");

    return 0;
}



