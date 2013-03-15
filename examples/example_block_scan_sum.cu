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
 * nvcc example_block_scan_sum.cu -gencode=arch=compute_20,code=\"sm_20,compute_20\" -o example_block_scan_sum
 *
 ******************************************************************************/

// Ensure printing of CUDA runtime errors to console (define before including cub.h)
#define CUB_STDERR

#include <stdio.h>
#include <iostream>

#include <cub/cub.cuh>

using namespace cub;

//---------------------------------------------------------------------
// Globals, constants and typedefs
//---------------------------------------------------------------------

bool g_verbose      = false;
int g_iterations    = 100;


//---------------------------------------------------------------------
// Kernels
//---------------------------------------------------------------------

/**
 * Simple kernel for performing a block-wide exclusive prefix sum over integers
 */
template <
    int         BLOCK_THREADS,
    int         ITEMS_PER_THREAD>
__global__ void BlockPrefixSumKernel(
    int         *d_in,          // Tile of input
    int         *d_out,         // Tile of output
    clock_t     *d_elapsed)     // Elapsed cycle count of block scan
{
    // Parameterize BlockScan type for our thread block
    typedef BlockScan<int, BLOCK_THREADS> BlockScanT;

    // Shared memory
    __shared__ typename BlockScanT::SmemStorage smem_storage;

    // Per-thread tile data
    int data[ITEMS_PER_THREAD];
    BlockLoadVectorized(d_in, data);

    // Start cycle timer
    clock_t start = clock();

    // Compute exclusive prefix sum
    int aggregate;
    BlockScanT::ExclusiveSum(smem_storage, data, data, aggregate);

    // Stop cycle timer
    clock_t stop = clock();

    // Store output
    BlockStoreVectorized(d_out, data);

    // Store aggregate and elapsed clocks
    if (threadIdx.x == 0)
    {
        *d_elapsed = (start > stop) ? start - stop : stop - start;
        d_out[BLOCK_THREADS * ITEMS_PER_THREAD] = aggregate;
    }
}



//---------------------------------------------------------------------
// Host utilities
//---------------------------------------------------------------------

/**
 * Initialize exclusive prefix sum problem (and solution).
 * Returns the aggregate
 */
int Initialize(
    int *h_in,
    int *h_reference,
    int num_elements)
{
    int inclusive = 0;

    for (int i = 0; i < num_elements; ++i)
    {
        h_in[i] = i % 17;

        h_reference[i] = inclusive;
        inclusive += h_in[i];
    }

    return inclusive;
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
    int *h_gpu          = new int[TILE_SIZE + 1];

    // Initialize problem and reference output on host
    int h_aggregate = Initialize(h_in, h_reference, TILE_SIZE);

    // Initialize device arrays
    int *d_in           = NULL;
    int *d_out          = NULL;
    clock_t *d_elapsed  = NULL;
    cudaMalloc((void**)&d_in,          sizeof(int) * TILE_SIZE);
    cudaMalloc((void**)&d_out,         sizeof(int) * (TILE_SIZE + 1));
    cudaMalloc((void**)&d_elapsed,     sizeof(clock_t));

    // Display input problem data
    if (g_verbose)
    {
        printf("Input data: ");
        for (int i = 0; i < TILE_SIZE; i++)
            printf("%d, ", h_in[i]);
        printf("\n\n");
    }

    // Copy problem to device
    cudaMemcpy(d_in, h_in, sizeof(int) * TILE_SIZE, cudaMemcpyHostToDevice);

    printf("BlockScan %d items (%d threads, %d items per thread): ",
        TILE_SIZE, BLOCK_THREADS, ITEMS_PER_THREAD);

    // Run this several times and average the performance results
    clock_t elapsed_scan_clocks     = 0;
    for (int i = 0; i < g_iterations; ++i)
    {
        // Run aggregate/prefix kernel
        BlockPrefixSumKernel<BLOCK_THREADS, ITEMS_PER_THREAD><<<1, BLOCK_THREADS>>>(
            d_in,
            d_out,
            d_elapsed);

        // Copy results from device
        clock_t scan_clocks;
        cudaMemcpy(h_gpu, d_out, sizeof(int) * (TILE_SIZE + 1), cudaMemcpyDeviceToHost);
        cudaMemcpy(&scan_clocks, d_elapsed, sizeof(clock_t), cudaMemcpyDeviceToHost);
        elapsed_scan_clocks += scan_clocks;
    }

    // Check scanned items
    bool correct = true;
    for (int i = 0; i < TILE_SIZE; i++)
    {
        if (h_gpu[i] != h_reference[i])
        {
            printf("Incorrect result @ offset %d (%d != %d)\n",
                i, h_gpu[i], h_reference[i]);
            correct = false;
            break;
        }
    }

    // Check total aggregate
    if (h_gpu[TILE_SIZE] != h_aggregate)
    {
        printf("Incorrect aggregate (%d != %d)\n", h_gpu[TILE_SIZE], h_aggregate);
        correct = false;
    }
    if (correct) printf("Correct!\n");

    // Display results problem data
    if (g_verbose)
    {
        printf("GPU output (reference output): ");
        for (int i = 0; i < TILE_SIZE; i++)
            printf("%d (%d), ", h_gpu[i], h_reference[i]);
        printf("\n");
        printf("GPU aggregate (reference aggregate)", h_gpu[TILE_SIZE], h_aggregate);
        printf("\n\n");
    }

    // Display timing results
    printf("Average clocks per 32-bit int scanned: %.3f\n\n", float(elapsed_scan_clocks) / TILE_SIZE / g_iterations);

    // Cleanup
    if (h_in) delete[] h_in;
    if (h_reference) delete[] h_reference;
    if (h_gpu) delete[] h_gpu;
    if (d_in) cudaFree(d_in);
    if (d_out) cudaFree(d_out);
    if (d_elapsed) cudaFree(d_elapsed);
}


/**
 * Main
 */
int main(int argc, char** argv)
{
    // Display GPU name
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, 0);
    printf("Using device %s\n", props.name);

/** Add tests here **/

    // Run tests
    Test<1024, 1>();
    Test<512, 2>();
    Test<256, 4>();
    Test<128, 8>();
    Test<64, 16>();
    Test<32, 32>();
    Test<16, 64>();

/****/

    return 0;
}

