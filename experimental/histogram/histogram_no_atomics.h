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

#define NPP_SMEM_BANKS			16
#define BLOCK_LENGTH		64
#define ITER_BIN_NUM		128
#define MERGE_BLOCK_SIZE	256

#define NUM_SEGS	2

inline __device__ int getThreadPosition(int nThreadId)
{
    return ((nThreadId & ~(NPP_SMEM_BANKS * 4 - 1)) << 0)
        | ((nThreadId & (NPP_SMEM_BANKS - 1)) << 2)
        | ((nThreadId & (NPP_SMEM_BANKS * 3)) >> 4);
}

template<int CHANNEL, class IN_TYPE>
inline __device__ unsigned int load(IN_TYPE &in)
{
    switch (CHANNEL) {
    case 0:
        return (unsigned int) (256 * in.x);
    case 1:
        return (unsigned int) (256 * in.y);
    case 2:
        return (unsigned int) (256 * in.z);
    }
    return 0;
}

template<int CHANNEL, class IN_TYPE>
__global__ void histogram_no_atomics(
    IN_TYPE *in,
    int width,
    int height,
    unsigned int *out,
    int max_length)
{
    const int threadLinearId = threadIdx.x + threadIdx.y * blockDim.x;
    const int blockLinearId = blockIdx.x + blockIdx.y * gridDim.x;
    const int nThreadPos = getThreadPosition(threadLinearId % BLOCK_LENGTH);
    const int stride_8u = BLOCK_LENGTH;
    const int stride_32s = stride_8u / 4;

    __shared__ unsigned char pHist[stride_8u * ITER_BIN_NUM];
    unsigned char *pThreadBase = pHist + nThreadPos;

    unsigned int nMinBin = 0;
    unsigned int nMaxBin;

    // process multiple segments
    for (int seg = 0; seg < NUM_SEGS; seg++)
    {
        const int seg_bin_count = ITER_BIN_NUM;
        nMaxBin = nMinBin + seg_bin_count;

        unsigned int nSum = 0;
        IN_TYPE *pTemp = in;
        int nBase = 0;
        while (nBase < width)
        {
            int pix_x_base = blockIdx.x * blockDim.x;
            int pix_y_base = blockIdx.y * blockDim.y;
            int seg_length =
                (width - nBase) > max_length ? max_length : (width - nBase);

            // flush smem, write integer instead of 4 chars
            if (threadLinearId < ITER_BIN_NUM)
            {
#pragma unroll
                for (unsigned int i = 0; i < BLOCK_LENGTH / 4; i++)
                    ((unsigned int *) pHist)[threadLinearId * stride_32s + i] =
                        0;
            }

            // iterate both x and y directions on the image
            int nBinPos;	// the thread's write position
            while (pix_y_base < height)
            {
                const int pix_y = pix_y_base + threadIdx.y;
                while (pix_x_base < seg_length)
                {
                    const int pix_x = pix_x_base + threadIdx.x;

                    // read data
                    unsigned int data = nMaxBin;
                    bool valid = (pix_x < seg_length && pix_y < height);
                    if (valid) data = load<CHANNEL, IN_TYPE>(
                        pTemp[pix_x + pix_y * width]);

                    // select bin
                    if (data >= nMinBin && data < nMaxBin)
                    {
                        nBinPos = data - nMinBin;
                        if (threadLinearId < BLOCK_LENGTH) pThreadBase[nBinPos
                            * stride_8u]++;
                    }
                    __syncthreads();

                    if (data >= nMinBin && data < nMaxBin) if (threadLinearId
                        >= BLOCK_LENGTH && threadLinearId < BLOCK_LENGTH * 2) pThreadBase[nBinPos
                        * stride_8u]++;
                    __syncthreads();

                    if (data >= nMinBin && data < nMaxBin) if (threadLinearId
                        >= BLOCK_LENGTH * 2
                        && threadLinearId < BLOCK_LENGTH * 3) pThreadBase[nBinPos
                        * stride_8u]++;
                    __syncthreads();

                    if (data >= nMinBin && data < nMaxBin) if (threadLinearId
                        >= BLOCK_LENGTH * 3
                        && threadLinearId < BLOCK_LENGTH * 4) pThreadBase[nBinPos
                        * stride_8u]++;
                    __syncthreads();

                    pix_x_base += blockDim.x * gridDim.x;
                }

                pix_y_base += blockDim.y * gridDim.y;
            }

            if (threadLinearId < seg_bin_count)
            {
                unsigned char *pHistBase = pHist + threadLinearId * stride_8u;
                unsigned int pos = 4 * (threadLinearId & (NPP_SMEM_BANKS - 1));
#pragma unroll
                for (unsigned int i = 0; i < (BLOCK_LENGTH / 4); i++)
                {
                    nSum += pHistBase[pos + 0] + pHistBase[pos + 1]
                        + pHistBase[pos + 2] + pHistBase[pos + 3];
                    pos = (pos + 4) & (BLOCK_LENGTH - 1);
                }
            }
            __syncthreads();

            // update seg length to iterate on the image
            pTemp += seg_length;
            nBase += seg_length;
        }

        if (threadLinearId < seg_bin_count) out[blockLinearId * ITER_BIN_NUM
            * NUM_SEGS + threadLinearId + seg * ITER_BIN_NUM] = nSum;
        nMinBin = nMaxBin;
    }
}

__global__ void histogram_no_atomics_accum(
    unsigned int *out,
    const unsigned int *in,
    int n)
{
    __shared__ unsigned int data[MERGE_BLOCK_SIZE];

    unsigned int nSum = 0;
    for (unsigned int i = threadIdx.x; i < n; i += MERGE_BLOCK_SIZE)
        nSum += in[blockIdx.x + i * ITER_BIN_NUM * NUM_SEGS];
    data[threadIdx.x] = nSum;

    for (unsigned int stride = MERGE_BLOCK_SIZE / 2; stride > 0; stride >>= 1)
    {
        __syncthreads();
        if (threadIdx.x < stride) data[threadIdx.x] +=
            data[threadIdx.x + stride];
    }

    if (threadIdx.x == 0) out[blockIdx.x] = data[0];
}

template<class in_type>
double run_no_atomics(
    in_type *d_image,
    int width,
    int height,
    unsigned int *d_hist)
{
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, 0);
    int max_threads_sm = props.maxThreadsPerMultiProcessor;
    int sm_count = props.multiProcessorCount;

    dim3 block(32, 8);
    dim3 grid(1,
        min(max_threads_sm / block.y * sm_count,
            (height + block.y - 1) / block.y));
    int max_length = grid.x * block.x * 256 / 4; // 64 slots, 256 threads, each position is written by 4 threads
    int total_blocks = grid.x * grid.y;

    // allocate partial histogram
    unsigned int *d_part_hist;
    int inter_hist_size = 64 * ((NUM_BINS + 63) / 64);
    int channel_size = (64 + 4 * inter_hist_size * total_blocks + 256 * 24);
    cudaMalloc(&d_part_hist, 3 * channel_size * sizeof(unsigned int));

    dim3 block2(MERGE_BLOCK_SIZE);
    dim3 grid2(NUM_BINS);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    histogram_no_atomics<0><<<grid, block>>>(d_image, width, height, d_part_hist + 0 * channel_size, max_length);
    histogram_no_atomics<1><<<grid, block>>>(d_image, width, height, d_part_hist + 1 * channel_size, max_length);
    histogram_no_atomics<2><<<grid, block>>>(d_image, width, height, d_part_hist + 2 * channel_size, max_length);

    histogram_no_atomics_accum<<<grid2, block2>>>(d_hist + 0 * NUM_BINS, d_part_hist + 0 * channel_size, total_blocks);
    histogram_no_atomics_accum<<<grid2, block2>>>(d_hist + 1 * NUM_BINS, d_part_hist + 1 * channel_size, total_blocks);
    histogram_no_atomics_accum<<<grid2, block2>>>(d_hist + 2 * NUM_BINS, d_part_hist + 2 * channel_size, total_blocks);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float time;
    cudaEventElapsedTime(&time, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(d_part_hist);

    return time * 1e3;
}

