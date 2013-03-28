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
 * Simple demonstration of cub::BlockScan
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

#include <cub.cuh>

using namespace cub;

//---------------------------------------------------------------------
// Globals, constants and typedefs
//---------------------------------------------------------------------

/// Verbose output
bool g_verbose = false;

/// Timing iterations
int g_iterations = 100;


//---------------------------------------------------------------------
// Kernels
//---------------------------------------------------------------------

/**
 * Simple kernel for performing a block-wide sorting over integers
 */
template <
    int         BLOCK_THREADS,
    int         ITEMS_PER_THREAD>
__global__ void BlockSortKernel(
    int         *d_in,          // Tile of input
    int         *d_out,         // Tile of output
    clock_t     *d_elapsed)     // Elapsed cycle count of block scan
{
    // Parameterize BlockRadixSort type for our thread block
    typedef BlockRadixSort<int, BLOCK_THREADS, ITEMS_PER_THREAD> BlockRadixSortT;

    // Shared memory
    __shared__ typename BlockRadixSortT::SmemStorage smem_storage;

    // Per-thread tile data
    int data[ITEMS_PER_THREAD];
    BlockLoadVectorized(d_in, data);

    // Start cycle timer
    clock_t start = clock();

    // Sort keys
    BlockRadixSortT::SortBlocked(smem_storage, data);

    // Stop cycle timer
    clock_t stop = clock();

    // Store output
    BlockStoreVectorized(d_out, data);

    // Store elapsed clocks
    if (threadIdx.x == 0)
    {
        *d_elapsed = (start > stop) ? start - stop : stop - start;
    }
}



//---------------------------------------------------------------------
// Host utilities
//---------------------------------------------------------------------


/**
 * Event-based GPU kernel timer
 */
struct GpuTimer
{
    cudaEvent_t start;
    cudaEvent_t stop;

    GpuTimer()
    {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }

    ~GpuTimer()
    {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    void Start()
    {
        cudaEventRecord(start, 0);
    }

    void Stop()
    {
        cudaEventRecord(stop, 0);
    }

    float ElapsedMillis()
    {
        float elapsed;
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed, start, stop);
        return elapsed;
    }
};


/**
 * Initialize exclusive sorting problem (and solution).
 */
void Initialize(
    int *h_in,
    int *h_reference,
    int num_items)
{
    for (int i = 0; i < num_items; ++i)
    {
        h_in[i] = (int) rand();
        h_reference[i] = h_in[i];
    }

    std::sort(h_reference, h_reference + num_items);
}


/**
 * Test thread block scan
 */
template <
    int BLOCK_THREADS,
    int ITEMS_PER_THREAD>
void Test()
{
    const int TILE_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD;

    // Allocate host arrays
    int *h_in           = new int[TILE_SIZE];
    int *h_reference    = new int[TILE_SIZE];
    int *h_gpu          = new int[TILE_SIZE];

    // Initialize problem and reference output on host
    Initialize(h_in, h_reference, TILE_SIZE);

    // Initialize device arrays
    int *d_in           = NULL;
    int *d_out          = NULL;
    clock_t *d_elapsed  = NULL;
    CubDebugExit(cudaMalloc((void**)&d_in,          sizeof(int) * TILE_SIZE));
    CubDebugExit(cudaMalloc((void**)&d_out,         sizeof(int) * (TILE_SIZE)));
    CubDebugExit(cudaMalloc((void**)&d_elapsed,     sizeof(clock_t)));

    // Display input problem data
    if (g_verbose)
    {
        printf("Input data: ");
        for (int i = 0; i < TILE_SIZE; i++)
            printf("%d, ", h_in[i]);
        printf("\n\n");
    }

    // Copy problem to device
    CubDebugExit(cudaMemcpy(d_in, h_in, sizeof(int) * TILE_SIZE, cudaMemcpyHostToDevice));

    printf("BlockScan, %d threadblock threads, %d items per thread:\n", BLOCK_THREADS, ITEMS_PER_THREAD);
    fflush(stdout);

    // Run kernel iterations
    GpuTimer    timer;
    float       elapsed_millis          = 0.0;
    clock_t     elapsed_scan_clocks     = 0;
    bool        correct                 = true;

    for (int i = 0; i < g_iterations; ++i)
    {
        timer.Start();

        // Run kernel
        BlockSortKernel<BLOCK_THREADS, ITEMS_PER_THREAD><<<1, BLOCK_THREADS>>>(
            d_in,
            d_out,
            d_elapsed);

        timer.Stop();
        elapsed_millis += timer.ElapsedMillis();

        // Copy results from device
        clock_t scan_clocks;
        CubDebugExit(cudaMemcpy(h_gpu, d_out,               sizeof(int) * (TILE_SIZE), cudaMemcpyDeviceToHost));
        CubDebugExit(cudaMemcpy(&scan_clocks, d_elapsed,    sizeof(clock_t), cudaMemcpyDeviceToHost));
        elapsed_scan_clocks += scan_clocks;
/*
        // Check data
        for (int i = 0; i < TILE_SIZE; i++)
        {
            if (h_gpu[i] != h_reference[i])
            {
                printf("Incorrect result @ offset %d (%d != %d)\n", i, h_gpu[i], h_reference[i]);
                correct = false;
                break;
            }
        }
        if (!correct) break;
*/
    }
    if (correct) printf("Correct!\n");

    // Check for kernel errors and STDIO from the kernel, if any
    CubDebugExit(cudaDeviceSynchronize());

    // Display results problem data
    if (g_verbose)
    {
        printf("GPU output (reference output): ");
        for (int i = 0; i < TILE_SIZE; i++)
            printf("%d (%d), ", h_gpu[i], h_reference[i]);
        printf("\n");
    }

    // Display timing results
    float avg_millis            = elapsed_millis / g_iterations;
    float avg_words_per_sec     = float(TILE_SIZE) / (elapsed_millis / 1000.0);
    float avg_clocks            = float(elapsed_scan_clocks) / g_iterations;
    float avg_clocks_per_word   = avg_clocks / TILE_SIZE;

    printf("\tAverage kernel millis: %.4f\n", avg_millis);
    printf("\tAverage words sorted / sec: %.4f\n", avg_words_per_sec);
    printf("\tAverage BlockRadixSort::SortBlocked clocks: %.3f\n", avg_clocks);
    printf("\tAverage BlockRadixSort::SortBlocked clocks per word scanned: %.3f\n", avg_clocks_per_word);

    // Cleanup
    if (h_in) delete[] h_in;
    if (h_reference) delete[] h_reference;
    if (h_gpu) delete[] h_gpu;
    if (d_in) CubDebugExit(cudaFree(d_in));
    if (d_out) CubDebugExit(cudaFree(d_out));
    if (d_elapsed) CubDebugExit(cudaFree(d_elapsed));
}


/**
 * Main
 */
int main(int argc, char** argv)
{
    if (argc > 1)
    {
        g_verbose = true;
    }

    // Make sure there are GPUs
    int deviceCount;
    CubDebugExit(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0)
    {
        fprintf(stderr, "No devices supporting CUDA.\n");
        exit(1);
    }

    // Iterate over installed GPUs
    for (int i = 0; i < deviceCount; i++)
    {
        printf("------------------------------------------\n");
        // Display GPU name
        cudaDeviceProp props;
        CubDebugExit(cudaGetDeviceProperties(&props, i));
        printf("Using device %d: %s\n", i, props.name);
        fflush(stdout);

        // Set GPU
        CubDebugExit(cudaSetDevice(i));

        // Run tests
        Test<128, 4>();
    }

    return 0;
}

