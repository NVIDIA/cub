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

#define NUM_PARTS       1024

static __device__ __forceinline__ int get_lane_id()
{
    int id;
    asm( "mov.u32 %0, %%laneid;" : "=r"(id) );
    return id;
}

static __device__ __forceinline__ void warp_coalescing_add(
    int data,
    unsigned int *smem)
{
    int lane_id = get_lane_id();
    int lane_bit = 1 << lane_id;
    int mask = -1;          // all 32 bits set to 1
    int tinc = 0;
    int tref = 0;
    while (mask)
    {
        int index = __ffs(mask) - 1;          // find leading bit -> thread id
        int ref = __shfl(data, index); // broadcast thread ref value to all threads
        unsigned int ballot = __ballot(
            (ref == data ? 1 : 0) && (mask & lane_bit)); // generate a ballot of threads having the same value
        if (lane_id == index)
        {                           // only 1 thread writes the result
            tinc = __popc(ballot);           // num of bits set -> bin increment
            tref = ref;
        }
        mask &= ~ballot;      // move on to the next set
    }
    if (tinc) atomicAdd(&smem[tref], tinc);
}

template<class IN_TYPE>
__global__ void histogram_smem_warp_coalescing(
    const IN_TYPE *in,
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
    __shared__ unsigned int smem[3 * NUM_BINS + 3];
    for (int i = t; i < 3 * NUM_BINS + 3; i += nt)
        smem[i] = 0;
    __syncthreads();

    // process pixels
    // updates our group's partial histogram in smem
    for (int col = x; col < width; col += nx)
        for (int row = y; row < height; row += ny)
        {
            unsigned int r = (unsigned int) (256 * in[row * width + col].x);
            unsigned int g = (unsigned int) (256 * in[row * width + col].y);
            unsigned int b = (unsigned int) (256 * in[row * width + col].z);
            warp_coalescing_add(r, &smem[NUM_BINS * 0 + 0]);
            warp_coalescing_add(g, &smem[NUM_BINS * 1 + 1]);
            warp_coalescing_add(b, &smem[NUM_BINS * 2 + 2]);
        }
    __syncthreads();

    // move to our workgroup's slice of output
    out += g * NUM_PARTS;

    // store local output to global
    for (int i = t; i < NUM_BINS; i += nt)
    {
        out[i + NUM_BINS * 0] = smem[i + NUM_BINS * 0];
        out[i + NUM_BINS * 1] = smem[i + NUM_BINS * 1 + 1];
        out[i + NUM_BINS * 2] = smem[i + NUM_BINS * 2 + 2];
    }
}

template<class in_type>
double run_smem_warp_coalescing(
    in_type *d_image,
    int width,
    int height,
    unsigned int *d_hist)
{
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, 0);

    dim3 block(8, 32);
    dim3 grid(16, 16);
    int total_blocks = grid.x * grid.y;

    // allocate partial histogram
    unsigned int *d_part_hist;
    cudaMalloc(&d_part_hist, total_blocks * NUM_PARTS * sizeof(unsigned int));

    dim3 block2(128);
    dim3 grid2((3 * NUM_BINS + block.x - 1) / block.x);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    histogram_smem_warp_coalescing<<<grid, block>>>(d_image, width, height, d_part_hist);
    histogram_smem_accum<<<grid2, block2>>>(d_part_hist, total_blocks, d_hist);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float time;
    cudaEventElapsedTime(&time, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(d_part_hist);

    return time * 1e3;
}

