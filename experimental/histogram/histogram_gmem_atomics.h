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

#include <test/test_util.h>

#define NUM_PARTS       1024

template <
    int         ACTIVE_CHANNELS,
    int         NUM_BINS,
    typename    PixelType>
__global__ void histogram_gmem_atomics(
    const PixelType *in,
    int width,
    int height,
    unsigned int *out)
{
    // global position and size
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int nx = blockDim.x * gridDim.x;
    int ny = blockDim.y * gridDim.y;

    // threads in workgroup
    int t = threadIdx.x + threadIdx.y * blockDim.x; // thread index in workgroup, linear in 0..nt-1
    int nt = blockDim.x * blockDim.y; // total threads in workgroup

    // group index in 0..ngroups-1
    int g = blockIdx.x + blockIdx.y * gridDim.x;

    // initialize smem
    unsigned int *gmem = out + g * NUM_PARTS;
    for (int i = t; i < ACTIVE_CHANNELS * NUM_BINS; i += nt)
        gmem[i] = 0;
    __syncthreads();

    // process pixels
    // updates our group's partial histogram in smem
    for (int col = x; col < width; col += nx)
    {
        for (int row = y; row < height; row += ny)
        {
            PixelType pixel = in[row * width + col];

            unsigned int r = (unsigned int) (256 * pixel.x);
            unsigned int g = (unsigned int) (256 * pixel.y);
            unsigned int b = (unsigned int) (256 * pixel.z);
            unsigned int a = (unsigned int) (256 * pixel.w);

            if (ACTIVE_CHANNELS > 0)
                atomicAdd(&gmem[NUM_BINS * 0 + r], 1);
            if (ACTIVE_CHANNELS > 1)
                atomicAdd(&gmem[NUM_BINS * 1 + g], 1);
            if (ACTIVE_CHANNELS > 2)
                atomicAdd(&gmem[NUM_BINS * 2 + b], 1);
            if (ACTIVE_CHANNELS > 3)
                atomicAdd(&gmem[NUM_BINS * 3 + a], 1);
        }
    }
}

template <
    int         ACTIVE_CHANNELS,
    int         NUM_BINS>
__global__ void histogram_gmem_accum(
    const unsigned int *in,
    int n,
    unsigned int *out)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i > ACTIVE_CHANNELS * NUM_BINS)
        return; // out of range

    unsigned int total = 0;
    for (int j = 0; j < n; j++)
        total += in[i + NUM_PARTS * j];

    out[i] = total;
}

template <
    int         ACTIVE_CHANNELS,
    int         NUM_BINS,
    typename    PixelType>
double run_gmem_atomics(
    PixelType *d_image,
    int width,
    int height,
    unsigned int *d_hist, 
    bool warmup)
{
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, 0);

    dim3 block(32, 4);
    dim3 grid(16, 16);
    int total_blocks = grid.x * grid.y;

    // allocate partial histogram
    unsigned int *d_part_hist;
    cudaMalloc(&d_part_hist, total_blocks * NUM_PARTS * sizeof(unsigned int));

    dim3 block2(128);
    dim3 grid2((3 * NUM_BINS + block.x - 1) / block.x);

    GpuTimer gpu_timer;
    gpu_timer.Start();

    histogram_gmem_atomics<ACTIVE_CHANNELS, NUM_BINS><<<grid, block>>>(d_image, width, height, d_part_hist);
    histogram_gmem_accum<ACTIVE_CHANNELS, NUM_BINS><<<grid2, block2>>>(d_part_hist, total_blocks, d_hist);

    gpu_timer.Stop();
    float elapsed_millis = gpu_timer.ElapsedMillis();

    cudaFree(d_part_hist);

    return elapsed_millis;
}

