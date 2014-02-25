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
 * Simple demonstration of cub::BlockTridiagonalSolve
 ******************************************************************************/

// Ensure printing of CUDA runtime errors to console (define before including cub.h)
#define CUB_STDERR

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
int g_timing_iterations = 100;

/// Default grid size
int g_grid_size = 1;



//---------------------------------------------------------------------
// Kernels
//---------------------------------------------------------------------

/**
 * Simple kernel for solving a block-wide tridiagonal linear system of equations Ax = d
 */
template <
    int                 BLOCK_THREADS,      /// The thread block size in threads
    int                 ITEMS_PER_THREAD,   /// The number of consecutive unknowns partitioned onto each thread
    BlockScanAlgorithm  SCAN_ALGORITHM,     /// cub::BlockScanAlgorithm enumerator specifying the underlying algorithm to use (e.g., cub::BLOCK_SCAN_RAKING)
    typename            T>                  /// Data type of numeric coefficients (e.g., \p float or \p double)
__global__ void BlockSolveKernel(
    T       *d_a,                           ///< [in] Coefficients within the subdiagonal of the matrix multipicand A
    T       *d_b,                           ///< [in] Coefficients within the main diagonal of the matrix multipicand A
    T       *d_c,                           ///< [in] Coefficients within the superdiagonal of the matrix multipicand A
    T       *d_d,                           ///< [in] The vector product
    T       *d_x,                           ///< [out] The vector multiplier (solution)
    clock_t *d_elapsed)                     ///< [out] Elapsed cycle count of block reduction
{
    // Specialize BlockTridiagonalSolve type for our thread block
    typedef BlockTridiagonalSolve<T, BLOCK_THREADS, SCAN_ALGORITHM> BlockTridiagonalSolveT;

    // Specialize BlockLoad type for our thread block
    typedef BlockLoad<T*, BLOCK_THREADS, ITEMS_PER_THREAD, BLOCK_LOAD_WARP_TRANSPOSE> BlockLoadT;

    // Specialize BlockStore type for our thread block
    typedef BlockStore<T*, BLOCK_THREADS, ITEMS_PER_THREAD, BLOCK_STORE_WARP_TRANSPOSE> BlockStoreT;

    // Allocate shared memory
    __shared__ union
    {
        typename BlockTridiagonalSolveT::TempStorage    solve;
        typename BlockLoadT::TempStorage                load;
        typename BlockStoreT::TempStorage               store;

    } temp_storage;

    // Declare per-thread data
    T a[ITEMS_PER_THREAD];  // Each thread's segment of coefficients within the subdiagonal of the matrix multipicand A
    T b[ITEMS_PER_THREAD];  // Each thread's segment of coefficients within the main diagonal of the matrix multipicand A
    T c[ITEMS_PER_THREAD];  // Each thread's segment of coefficients within the superdiagonal of the matrix multipicand A
    T d[ITEMS_PER_THREAD];  // Each thread's segment of the vector product
    T x[ITEMS_PER_THREAD];  // Each thread's segment of the vector multiplier

    // Load system of equations
    int block_offset = blockIdx.x * BLOCK_THREADS * ITEMS_PER_THREAD;
    BlockLoadT(temp_storage.load).Load(d_a + block_offset, a);
    BlockLoadT(temp_storage.load).Load(d_b + block_offset, b);
    BlockLoadT(temp_storage.load).Load(d_c + block_offset, c);
    BlockLoadT(temp_storage.load).Load(d_d + block_offset, d);

    __syncthreads();

    // Start cycle timer
    clock_t start = clock();

    // Compute solution
    BlockTridiagonalSolveT(temp_storage.solve).Solve(a, b, c, d, x);

    // Stop cycle timer
    clock_t stop = clock();

    __syncthreads();

    // Store output
    BlockStoreT(temp_storage.store).Store(d_x + block_offset, x);

    // Store aggregate and elapsed clocks
    if (threadIdx.x == 0)
    {
        *d_elapsed = (start > stop) ? start - stop : stop - start;
    }
}



//---------------------------------------------------------------------
// Host utilities
//---------------------------------------------------------------------

/**
 * Simple tridiagonal matrix equation tuple
 */
template <typename T>
struct MatrixEquation
{
    T a;
    T b;
    T c;
    T d;
    T x;

    T f;
    T e;
    T y;

    static T Abs(T val)
    {
        return (val > 0) ? val : val * -1.0;
    }

    void Init(T abs_max)
    {

        // Select non-zero coefficients
        int x, y, z, w;
        do
        {
            RandomBits(x);
            RandomBits(y);
            RandomBits(z);

        } while ((x == 0) || (y == 0) || (z == 0));
        RandomBits(w);

        // Scale
        a = T(x) / T(INT_MAX) * abs_max;
        b = T(y) / T(INT_MAX) * abs_max;
        c = T(z) / T(INT_MAX) * abs_max;
        d = T(w) / T(INT_MAX) * abs_max;

        // Make diagonally dominant
        if (Abs(a) > Abs(b))
        {
            T temp = a;
            a = b;
            b = temp;
        }
        if (Abs(c) > Abs(b))
        {
            T temp = c;
            c = b;
            b = temp;
        }
    }

    // Less than operator for diagonal pivoting (actually greater-than for sorting by descending order)
    bool operator<(const MatrixEquation &other) const
    {
        return (Abs(b) > Abs(other.b));
    }
};


/**
 * Initialize tridiagonal system (and solution).
 */
template <typename T>
void Initialize(
    T *a,
    T *b,
    T *c,
    T *d,
    T *x,
    int system_size,
    int num_systems)
{
    float abs_max = 9.0;        // somewhat arbitrary maximum coefficient

    // Loop over systems
    for (int j = 0; j < num_systems; ++j)
    {
        // Initialize system
        MatrixEquation<T> *rows = new MatrixEquation<T>[system_size];
        for (int i = 0; i < system_size; ++i)
            rows[i].Init(abs_max);

        // Pivot system (so main diagonal is ordered largest to smallest)
        std::stable_sort(rows, rows + system_size);

        // Fix "out-of-bounds" coefficients
        rows[0].a                   = 0;    // first item in subdiagonal out of bounds (0)
        rows[system_size - 1].c     = 1;    // last item in superdiagonal is out of bounds (1)

        // Solve system using LU decomposition
        rows[0].f = rows[0].b;
        for (int i = 1; i < system_size; ++i)
        {
            rows[i].e = rows[i].a / rows[i - 1].f;
            rows[i].f = rows[i].b - rows[i].e * rows[i - 1].c;
        }

        rows[0].y = rows[0].d;
        for (int i = 1; i < system_size; ++i)
        {
            rows[i].y = rows[i].d - rows[i].e * rows[i - 1].y;
        }

        rows[system_size - 1].x = rows[system_size - 1].y / rows[system_size - 1].f;
        for (int i = system_size - 2; i >= 0; --i)
        {
            rows[i].x = (rows[i].y - rows[i].c * rows[i + 1].x) / rows[i].f;
        }

        // Display solved system
        if (g_verbose)
        {
            printf("x[%d]: %.3f \t\t(%.3f*%.3f + %.3f*%.3f = %.3f)\n",
                0, rows[0].x,
                rows[0].b, rows[0].x,
                rows[0].c, rows[1].x,
                rows[0].d);

            for (int i = 1; i < system_size - 1; ++i)
                printf("x[%d]: %.3f \t\t(%.3f*%.3f + %.3f*%.3f + %.3f*%.3f = %.3f)\n",
                    i, rows[i].x,
                    rows[i].a, rows[i - 1].x,
                    rows[i].b, rows[i].x,
                    rows[i].c, rows[i + 1].x,
                    rows[i].d);

            printf("x[%d]: %.3f \t\t(%.3f*%.3f + %.3f*%.3f = %.3f)\n",
                system_size - 1, rows[system_size - 1].x,
                rows[system_size - 1].a, rows[system_size - 2].x,
                rows[system_size - 1].b, rows[system_size - 1].x,
                rows[system_size - 1].d);
        }

        // Copy system
        int base = j * system_size;
        for (int i = 0; i < system_size; ++i)
        {
            a[base + i] = rows[i].a;
            b[base + i] = rows[i].b;
            c[base + i] = rows[i].c;
            d[base + i] = rows[i].d;
            x[base + i] = rows[i].x;
        }

        // Cleanup
        delete[] rows;
    }
}


/**
 * Test thread block tridiagonal solve
 */
template <
    typename            T,
    int                 BLOCK_THREADS,
    int                 ITEMS_PER_THREAD,
    BlockScanAlgorithm  SCAN_ALGORITHM>
void Test()
{
    const int TILE_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD;

    // Allocate host arrays
    T *h_a = new T[TILE_SIZE * g_grid_size];
    T *h_b = new T[TILE_SIZE * g_grid_size];
    T *h_c = new T[TILE_SIZE * g_grid_size];
    T *h_d = new T[TILE_SIZE * g_grid_size];
    T *h_x = new T[TILE_SIZE * g_grid_size];

    // Initialize problem and reference output on host
    Initialize(h_a, h_b, h_c, h_d, h_x, TILE_SIZE, g_grid_size);

    // Declare device arrays
    T *d_a              = NULL;
    T *d_b              = NULL;
    T *d_c              = NULL;
    T *d_d              = NULL;
    T *d_x              = NULL;
    clock_t *d_elapsed  = NULL;

    // Initialize device arrays
    cudaMalloc((void**)&d_a,        sizeof(T) * TILE_SIZE * g_grid_size);
    cudaMalloc((void**)&d_b,        sizeof(T) * TILE_SIZE * g_grid_size);
    cudaMalloc((void**)&d_c,        sizeof(T) * TILE_SIZE * g_grid_size);
    cudaMalloc((void**)&d_d,        sizeof(T) * TILE_SIZE * g_grid_size);
    cudaMalloc((void**)&d_x,        sizeof(T) * TILE_SIZE * g_grid_size);
    cudaMalloc((void**)&d_elapsed,  sizeof(clock_t) * g_grid_size);

    // CUDA device props
    Device device;
    int max_sm_occupancy;
    CubDebugExit(device.Init());
    CubDebugExit(device.MaxSmOccupancy(
        max_sm_occupancy,
        BlockSolveKernel<BLOCK_THREADS, ITEMS_PER_THREAD, SCAN_ALGORITHM, T>,
        BLOCK_THREADS));

    // Copy problem to device
    cudaMemcpy(d_a, h_a, sizeof(T) * TILE_SIZE * g_grid_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, sizeof(T) * TILE_SIZE * g_grid_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, h_c, sizeof(T) * TILE_SIZE * g_grid_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_d, h_d, sizeof(T) * TILE_SIZE * g_grid_size, cudaMemcpyHostToDevice);
    cudaMemset(d_x, 0, sizeof(T) * TILE_SIZE * g_grid_size);

    printf("BlockTridiagonalSolve %d items (%d timing iterations, %d blocks, %d threads, %d items per thread, %d SM occupancy):\n",
        TILE_SIZE, g_timing_iterations, g_grid_size, BLOCK_THREADS, ITEMS_PER_THREAD, max_sm_occupancy);

    // Run aggregate/prefix kernel
    BlockSolveKernel<BLOCK_THREADS, ITEMS_PER_THREAD, SCAN_ALGORITHM><<<g_grid_size, BLOCK_THREADS>>>(
        d_a,
        d_b,
        d_c,
        d_d,
        d_x,
        d_elapsed);

    // Check output from first system
    printf("\tSolve: ");
    int compare = CompareDeviceResults(h_x, d_x, TILE_SIZE, g_verbose, g_verbose);
    printf("%s\n", compare ? "FAIL" : "PASS");

    // Run this several times and average the performance results
    if (g_timing_iterations > 0)
    {
        GpuTimer    timer;
        float       elapsed_millis          = 0.0;
        clock_t     elapsed_clocks          = 0;

        for (int i = 0; i < g_timing_iterations; ++i)
        {
            timer.Start();

            // Run aggregate/prefix kernel
            BlockSolveKernel<BLOCK_THREADS, ITEMS_PER_THREAD, SCAN_ALGORITHM><<<g_grid_size, BLOCK_THREADS>>>(
                d_a,
                d_b,
                d_c,
                d_d,
                d_x,
                d_elapsed);

            timer.Stop();
            elapsed_millis += timer.ElapsedMillis();

            // Copy clocks from device
            clock_t clocks;
            CubDebugExit(cudaMemcpy(&clocks, d_elapsed, sizeof(clock_t), cudaMemcpyDeviceToHost));
            elapsed_clocks += clocks;
        }

        // Check for kernel errors and STDIO from the kernel, if any
        CubDebugExit(cudaDeviceSynchronize());

        // Display timing results
        float avg_millis            = elapsed_millis / g_timing_iterations;
        float avg_items_per_sec     = float(TILE_SIZE * g_grid_size) / avg_millis / 1000.0;
        float avg_clocks            = float(elapsed_clocks) / g_timing_iterations;
        float avg_clocks_per_item   = avg_clocks / TILE_SIZE;

        printf("\tAverage BlockTridiagonalSolve::Solve clocks: %.3f\n", avg_clocks);
        printf("\tAverage BlockTridiagonalSolve::Solve clocks per item: %.3f\n", avg_clocks_per_item);
        printf("\tAverage kernel millis: %.4f\n", avg_millis);
        printf("\tAverage million items / sec: %.4f\n", avg_items_per_sec);
    }

    // Cleanup
    if (h_a) delete[] h_a;
    if (h_b) delete[] h_b;
    if (h_c) delete[] h_c;
    if (h_d) delete[] h_d;
    if (h_x) delete[] h_x;

    if (d_a) cudaFree(d_a);
    if (d_b) cudaFree(d_b);
    if (d_c) cudaFree(d_c);
    if (d_d) cudaFree(d_d);
    if (d_x) cudaFree(d_x);
    if (d_elapsed) cudaFree(d_elapsed);

    AssertEquals(0, compare);
}


/**
 * Main
 */
int main(int argc, char** argv)
{
    // Initialize command line
    CommandLineArgs args(argc, argv);
    g_verbose = args.CheckCmdLineFlag("v");
    args.GetCmdLineArgument("i", g_timing_iterations);
    args.GetCmdLineArgument("grid-size", g_grid_size);

    // Print usage
    if (args.CheckCmdLineFlag("help"))
    {
        printf("%s "
            "[--device=<device-id>] "
            "[--i=<timing iterations>] "
            "[--grid-size=<grid size>] "
            "[--v] "
            "\n", argv[0]);
        exit(0);
    }

    // Initialize device
    CubDebugExit(args.DeviceInit());


/** Add tests here **/

    // Run tests
    Test<float, 32, 1, BLOCK_SCAN_RAKING>();

/****/

    return 0;
}

