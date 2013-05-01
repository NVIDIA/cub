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
 * Simple demonstration of cub::BlockSort
 *
 * Example compilation string:
 *
 * nvcc example_block_radix_sort.cu -gencode=arch=compute_20,code=\"sm_20,compute_20\" -o example_block_radix_sort -m32 -Xptxas -v -I../cub
 *
 ******************************************************************************/

// Ensure printing of CUDA runtime errors to console (define before including cub.h)
#define CUB_STDERR

#if defined(_WIN32) || defined(_WIN64)
    #include <windows.h>
    #undef small            // Windows is terrible for polluting macro namespace
#else
    #include <sys/resource.h>
#endif

#include <stdio.h>
#include <iostream>
#include <algorithm>

#include <cub/cub.cuh>

#include "../test/test_util.h"

using namespace cub;

//---------------------------------------------------------------------
// Globals, constants and typedefs
//---------------------------------------------------------------------

/// Verbose output
bool g_verbose = false;

/// Timing iterations
int g_iterations = 100;

/// Default grid size
int g_grid_size = 1;

/// Uniform key samples
bool g_uniform_keys;


//---------------------------------------------------------------------
// Kernels
//---------------------------------------------------------------------

/**
 * Simple kernel for performing a block-wide sorting over integers
 */
template <
    typename    KeyType,
    int         BLOCK_THREADS,
    int         ITEMS_PER_THREAD>
__launch_bounds__ (BLOCK_THREADS)
__global__ void BlockSortKernel(
    KeyType     *d_in,          // Tile of input
    KeyType     *d_out,         // Tile of output
    clock_t     *d_elapsed)     // Elapsed cycle count of block scan
{
    enum { TILE_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD };

    // Parameterize BlockRadixSort type for our thread block
    typedef BlockRadixSort<KeyType, BLOCK_THREADS, ITEMS_PER_THREAD> BlockRadixSortT;

    // Shared memory
    __shared__ typename BlockRadixSortT::SmemStorage smem_storage;

    // Per-thread tile items
    KeyType items[ITEMS_PER_THREAD];

    // Load items in striped fashion
    int block_offset = blockIdx.x * TILE_SIZE;
    BlockLoadDirectStriped(d_in + block_offset, items);

    // Start cycle timer
    clock_t start = clock();

    // Sort keys
    BlockRadixSortT::SortStriped(smem_storage, items);

    // Stop cycle timer
    clock_t stop = clock();

    // Store output
    BlockStoreDirectStriped(d_out + block_offset, items);

    // Store elapsed clocks
    if (threadIdx.x == 0)
    {
        d_elapsed[blockIdx.x] = (start > stop) ? start - stop : stop - start;
    }
}



//---------------------------------------------------------------------
// Host utilities
//---------------------------------------------------------------------


/**
 * Initialize sorting problem (and solution).
 */
template <typename KeyType>
void Initialize(
    KeyType *h_in,
    KeyType *h_reference,
    int num_items,
    int tile_size)
{
    for (int i = 0; i < num_items; ++i)
    {
        if (g_uniform_keys)
        {
            h_in[i] = 0;
        }
        else
        {
            RandomBits(h_in[i]);
        }
        h_reference[i] = h_in[i];
    }

    // Only sort the first tile
    std::sort(h_reference, h_reference + tile_size);
}


/**
 * Test BlockScan
 */
template <
    typename    KeyType,
    int         BLOCK_THREADS,
    int         ITEMS_PER_THREAD>
void Test()
{
    const int TILE_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD;

    // Allocate host arrays
    KeyType *h_in           = new KeyType[TILE_SIZE * g_grid_size];
    KeyType *h_reference    = new KeyType[TILE_SIZE * g_grid_size];
    clock_t *h_elapsed      = new clock_t[g_grid_size];

    // Initialize problem and reference output on host
    Initialize(h_in, h_reference, TILE_SIZE * g_grid_size, TILE_SIZE);

    // Initialize device arrays
    KeyType *d_in       = NULL;
    KeyType *d_out      = NULL;
    clock_t *d_elapsed  = NULL;
    CubDebugExit(cudaMalloc((void**)&d_in,          sizeof(KeyType) * TILE_SIZE * g_grid_size));
    CubDebugExit(cudaMalloc((void**)&d_out,         sizeof(KeyType) * TILE_SIZE * g_grid_size));
    CubDebugExit(cudaMalloc((void**)&d_elapsed,     sizeof(clock_t) * g_grid_size));

    // Display input problem data
    if (g_verbose)
    {
        printf("Input data: ");
        for (int i = 0; i < TILE_SIZE; i++)
            std::cout << h_in[i] << ", ";
        printf("\n\n");
    }

    // CUDA device props
    Device device;
    int max_sm_occupancy;
    CubDebugExit(device.Init());
    CubDebugExit(device.MaxSmOccupancy(max_sm_occupancy, BlockSortKernel<KeyType, BLOCK_THREADS, ITEMS_PER_THREAD>, BLOCK_THREADS));

    // Copy problem to device
    CubDebugExit(cudaMemcpy(d_in, h_in, sizeof(KeyType) * TILE_SIZE * g_grid_size, cudaMemcpyHostToDevice));

    printf("BlockRadixSort %d items (%d timing iterations, %d blocks, %d threads, %d items per thread, %d SM occupancy):\n",
        TILE_SIZE * g_grid_size, g_iterations, g_grid_size, BLOCK_THREADS, ITEMS_PER_THREAD, max_sm_occupancy);
    fflush(stdout);

    // Run kernel once to prime caches and check result
    BlockSortKernel<KeyType, BLOCK_THREADS, ITEMS_PER_THREAD><<<g_grid_size, BLOCK_THREADS>>>(
        d_in,
        d_out,
        d_elapsed);

    // Check for kernel errors and STDIO from the kernel, if any
    CubDebugExit(cudaDeviceSynchronize());

    // Check results
    printf("\tOutput items: ");
    int compare = CompareDeviceResults(h_reference, d_out, TILE_SIZE, g_verbose, g_verbose);
    printf("%s\n", compare ? "FAIL" : "PASS");
    AssertEquals(0, compare);
    fflush(stdout);

    // Run this several times and average the performance results
    GpuTimer            timer;
    float               elapsed_millis          = 0.0;
    unsigned long long  elapsed_clocks          = 0;

    for (int i = 0; i < g_iterations; ++i)
    {
        timer.Start();

        // Run kernel
        BlockSortKernel<KeyType, BLOCK_THREADS, ITEMS_PER_THREAD><<<g_grid_size, BLOCK_THREADS>>>(
            d_in,
            d_out,
            d_elapsed);

        timer.Stop();
        elapsed_millis += timer.ElapsedMillis();

        // Copy clocks from device
        CubDebugExit(cudaMemcpy(h_elapsed, d_elapsed, sizeof(clock_t) * g_grid_size, cudaMemcpyDeviceToHost));
        for (int i = 0; i < g_grid_size; i++)
            elapsed_clocks += h_elapsed[i];
    }

    // Check for kernel errors and STDIO from the kernel, if any
    CubDebugExit(cudaDeviceSynchronize());

    // Display timing results
    float avg_millis            = elapsed_millis / g_iterations;
    float avg_items_per_sec     = float(TILE_SIZE * g_grid_size) / avg_millis / 1000.0;
    double avg_clocks           = double(elapsed_clocks) / g_iterations / g_grid_size;
    double avg_clocks_per_item  = avg_clocks / TILE_SIZE;

    printf("\tAverage BlockRadixSort::SortBlocked clocks: %.3f\n", avg_clocks);
    printf("\tAverage BlockRadixSort::SortBlocked clocks per item: %.3f\n", avg_clocks_per_item);
    printf("\tAverage kernel millis: %.4f\n", avg_millis);
    printf("\tAverage million items / sec: %.4f\n", avg_items_per_sec);
    fflush(stdout);

    // Cleanup
    if (h_in) delete[] h_in;
    if (h_reference) delete[] h_reference;
    if (h_elapsed) delete[] h_elapsed;
    if (d_in) CubDebugExit(cudaFree(d_in));
    if (d_out) CubDebugExit(cudaFree(d_out));
    if (d_elapsed) CubDebugExit(cudaFree(d_elapsed));
}


/**
 * Main
 */
int main(int argc, char** argv)
{
    // Initialize command line
    CommandLineArgs args(argc, argv);
    g_verbose = args.CheckCmdLineFlag("v");
    g_uniform_keys = args.CheckCmdLineFlag("uniform");
    args.GetCmdLineArgument("i", g_iterations);
    args.GetCmdLineArgument("grid-size", g_grid_size);

    // Print usage
    if (args.CheckCmdLineFlag("help"))
    {
        printf("%s "
            "[--device=<device-id>] "
            "[--i=<timing iterations (default:%d)>]"
            "[--grid-size=<grid size (default:%d)>]"
            "[--v] "
            "\n", argv[0], g_iterations, g_grid_size);
        exit(0);
    }

    // Initialize device
    CubDebugExit(args.DeviceInit());
    fflush(stdout);

    // Run tests
    printf("\nuint32:\n"); fflush(stdout);
    Test<unsigned int, 128, 17>();
    printf("\n"); fflush(stdout);

    printf("\nfp32:\n"); fflush(stdout);
    Test<float, 128, 17>();
    printf("\n"); fflush(stdout);

    printf("\nuint8:\n"); fflush(stdout);
    Test<unsigned char, 128, 17>();
    printf("\n"); fflush(stdout);

    return 0;
}

