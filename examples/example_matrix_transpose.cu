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
 * Example matrix transpose
 ******************************************************************************/

// Ensure printing of CUDA runtime errors to console
#define CUB_STDERR

#include <stdio.h>
#include <cub/cub.cuh>
#include <test_util.h>

using namespace cub;


//---------------------------------------------------------------------
// Globals, constants and typedefs
//---------------------------------------------------------------------

bool    g_verbose = false;
int     g_iterations = 100;


//---------------------------------------------------------------------
// Transpose kernel
//---------------------------------------------------------------------

template <
    int                 BLOCK_THREADS,
    int                 ITEMS_PER_THREAD,
    PtxLoadModifier     LOAD_MODIFIER,
    typename            T>
__global__ void TransposeKernel(
    T *d_in,
    T *d_out,
    int ydim,
    int xdim)
{
    enum
    {
        TILE_XDIM   = BLOCK_THREADS,
        TILE_YDIM   = ITEMS_PER_THREAD,
        TILE_SIZE   = TILE_XDIM * TILE_YDIM
    };

    typedef cub::BlockExchange<T, BLOCK_THREADS, ITEMS_PER_THREAD> BlockExchangeT;

    __shared__ typename BlockExchangeT::SmemStorage smem_storage;

    // Items per thread
    T items[ITEMS_PER_THREAD];

    // Load
    int block_offset = (blockIdx.y * gridDim.x * TILE_SIZE) + (blockIdx.x * TILE_XDIM);
    cub::BlockLoadDirectStriped<LOAD_MODIFIER>(d_in + block_offset, items, xdim);

    T data = __ldg(d_in);

    BlockExchangeT::StripedToBlocked(smem_storage, items);

    // Store
    block_offset = (blockIdx.x * gridDim.y * TILE_SIZE) + (blockIdx.y * TILE_YDIM);
    cub::BlockStoreDirectStriped(d_out + block_offset, items, ydim);
}




//---------------------------------------------------------------------
// Host utility subroutines
//---------------------------------------------------------------------

/**
 * Initialize problem (and solution)
 */
template <typename T>
void Initialize(
    T   *h_in,
    T   *h_reference,
    int ydim,
    int xdim)
{
    for (int y = 0; y < ydim; ++y)
    {
        for (int x = 0; x < xdim; ++x)
        {
            int val = (y * xdim) + x;
            h_in[(y * xdim) + x] = val;
            h_reference[(x * ydim) + y] = val;
        }
    }
}


//---------------------------------------------------------------------
// Full tile test generation
//---------------------------------------------------------------------


/**
 * Test matrix transpose
 */
template <
    int                 BLOCK_THREADS,
    int                 ITEMS_PER_THREAD,
    PtxLoadModifier     LOAD_MODIFIER,
    typename            T>
void TestTranspose(
    int                 ydim,
    int                 xdim)
{
    enum
    {
        TILE_XDIM   = BLOCK_THREADS,
        TILE_YDIM   = ITEMS_PER_THREAD,
        TILE_SIZE   = TILE_XDIM * TILE_YDIM
    };

    int num_items = ydim * xdim;

    // Allocate host arrays
    T *h_in = new T[num_items];
    T *h_reference = new T[num_items];

    // Initialize problem
    Initialize(h_in, h_reference, ydim, xdim);

    // Initialize device arrays
    T *d_in = NULL;
    T *d_out = NULL;
    CubDebugExit(DeviceAllocate((void**)&d_in, sizeof(T) * num_items));
    CubDebugExit(DeviceAllocate((void**)&d_out, sizeof(T) * num_items));
    CubDebugExit(cudaMemcpy(d_in, h_in, sizeof(T) * num_items, cudaMemcpyHostToDevice));

    dim3 grid_dims;
    grid_dims.y = ydim / TILE_YDIM;
    grid_dims.x = xdim / TILE_XDIM;
    grid_dims.z = 1;

    typedef void (*TransposeKernelPtr)(T*, T*, int, int);
    TransposeKernelPtr kernel_ptr = TransposeKernel<BLOCK_THREADS, ITEMS_PER_THREAD, LOAD_MODIFIER>;

    Device device;
    device.Init();

    int sm_occupancy;
    device.MaxSmOccupancy(sm_occupancy, kernel_ptr, BLOCK_THREADS);
    printf("sm_occupancy(%d)\n", sm_occupancy);

    printf("TransposeKernel<%d, %d, %d><<<{y(%d), x{%d}, %d>>>(ydim(%d), xdim(%d)) (%d elements, %d bytes each, TILE_YDIM(%d), TILE_XDIM(%d), TILE_SIZE(%d), sm_occupancy(%d)):\n",
        BLOCK_THREADS, ITEMS_PER_THREAD, LOAD_MODIFIER,
        grid_dims.y, grid_dims.x, BLOCK_THREADS,
        ydim, xdim, num_items, (int) sizeof(T),
        TILE_YDIM, TILE_XDIM, TILE_SIZE, sm_occupancy);
    fflush(stdout);

    // Run warmup/correctness iteration
    TransposeKernel<BLOCK_THREADS, ITEMS_PER_THREAD, LOAD_MODIFIER><<<grid_dims, BLOCK_THREADS>>>(
        d_in, d_out, ydim, xdim);

    CubDebugExit(cudaDeviceSynchronize());
    fflush(stdout);

    // Copy out and display results
    printf("\nTranspose results: ");
    int compare = CompareDeviceResults(h_reference, d_out, num_items, g_verbose, g_verbose);
    printf("\n");

    // Performance
    GpuTimer gpu_timer;
    float elapsed_millis = 0.0;
    for (int i = 0; i < g_iterations; i++)
    {
        gpu_timer.Start();

        TransposeKernel<BLOCK_THREADS, ITEMS_PER_THREAD, LOAD_MODIFIER><<<grid_dims, BLOCK_THREADS>>>(
            d_in, d_out, ydim, xdim);

        gpu_timer.Stop();
        elapsed_millis += gpu_timer.ElapsedMillis();
    }
    if (g_iterations > 0)
    {
        float avg_millis = elapsed_millis / g_iterations;
        float grate = float(num_items) / avg_millis / 1000.0 / 1000.0;
        float gbandwidth = grate * sizeof(T) * 2;
        printf("\nPerformance: %.3f avg ms, %.3f billion items/s, %.3f GB/s\n", avg_millis, grate, gbandwidth);
    }

    // Cleanup
    if (h_in) delete[] h_in;
    if (h_in) delete[] h_reference;
    if (d_in) CubDebugExit(DeviceFree(d_in));
    if (d_out) CubDebugExit(DeviceFree(d_out));

    AssertEquals(0, compare);
}




//---------------------------------------------------------------------
// Main
//---------------------------------------------------------------------

/**
 * Main
 */
int main(int argc, char** argv)
{
    typedef int T;

    int ydim = 1024;
    int xdim = 1024;

    // Initialize command line
    CommandLineArgs args(argc, argv);
    args.GetCmdLineArgument("y", ydim);
    args.GetCmdLineArgument("x", xdim);
    args.GetCmdLineArgument("i", g_iterations);
    g_verbose = args.CheckCmdLineFlag("v");
    bool quick = args.CheckCmdLineFlag("quick");

    // Print usage
    if (args.CheckCmdLineFlag("help"))
    {
        printf("%s "
            "[--device=<device-id>] "
            "[--v] "
            "[--y=<ydim>]"
            "[--x=<xdim>]"
            "[--i=<timing iterations>]"
            "\n", argv[0]);
        exit(0);
    }

    // Initialize device
    CubDebugExit(args.DeviceInit());

    TestTranspose<32, 32, PTX_LOAD_NONE, T>(ydim, xdim);

    return 0;
}



