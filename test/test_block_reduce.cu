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
 * Test of BlockReduce utilities
 ******************************************************************************/

// Ensure printing of CUDA runtime errors to console
#define CUB_STDERR

#include <stdio.h>

#include <device_functions.h>

#include <cub/util_allocator.cuh>
#include <cub/block/block_load.cuh>
#include <cub/block/block_reduce.cuh>

#include "test_util.h"

using namespace cub;


//---------------------------------------------------------------------
// Globals, constants and typedefs
//---------------------------------------------------------------------

bool                    g_verbose       = false;
int                     g_repeat        = 0;
CachingDeviceAllocator  g_allocator(true);



//---------------------------------------------------------------------
// Test kernels
//---------------------------------------------------------------------


/// Generic reduction (full, 1)
template <typename BlockReduce, typename T, typename ReductionOp>
__device__ __forceinline__ T DeviceTest(
    typename BlockReduce::TempStorage &temp_storage, T (&data)[1], ReductionOp &reduction_op)
{
    return BlockReduce(temp_storage).Reduce(data[0], reduction_op);
}

/// Generic reduction (full, ITEMS_PER_THREAD)
template <typename BlockReduce, typename T, int ITEMS_PER_THREAD, typename ReductionOp>
__device__ __forceinline__ T DeviceTest(
    typename BlockReduce::TempStorage &temp_storage, T (&data)[ITEMS_PER_THREAD], ReductionOp &reduction_op)
{
    return BlockReduce(temp_storage).Reduce(data, reduction_op);
}

/// Generic reduction (partial, 1)
template <typename BlockReduce, typename T, typename ReductionOp>
__device__ __forceinline__ T DeviceTest(
    typename BlockReduce::TempStorage &temp_storage, T &data, ReductionOp &reduction_op, int valid_threads)
{
    return BlockReduce(temp_storage).Reduce(data, reduction_op, valid_threads);
}

/// Sum reduction (full, 1)
template <typename BlockReduce, typename T>
__device__ __forceinline__ T DeviceTest(
    typename BlockReduce::TempStorage &temp_storage, T (&data)[1], Sum &reduction_op)
{
    return BlockReduce(temp_storage).Sum(data[0]);
}

/// Sum reduction (full, ITEMS_PER_THREAD)
template <typename BlockReduce, typename T, int ITEMS_PER_THREAD>
__device__ __forceinline__ T DeviceTest(
    typename BlockReduce::TempStorage &temp_storage, T (&data)[ITEMS_PER_THREAD], Sum &reduction_op)
{
    return BlockReduce(temp_storage).Sum(data);
}

/// Sum reduction (partial, 1)
template <typename BlockReduce, typename T>
__device__ __forceinline__ T DeviceTest(
    typename BlockReduce::TempStorage &temp_storage, T &data, Sum &reduction_op, int valid_threads)
{
    return BlockReduce(temp_storage).Sum(data, valid_threads);
}


/**
 * Test full-tile reduction kernel (where num_items is an even
 * multiple of BLOCK_THREADS)
 */
template <
    BlockReduceAlgorithm    ALGORITHM,
    int                     BLOCK_THREADS,
    int                     ITEMS_PER_THREAD,
    typename                T,
    typename                ReductionOp>
__launch_bounds__ (BLOCK_THREADS, 1)
__global__ void FullTileReduceKernel(
    T                       *d_in,
    T                       *d_out,
    ReductionOp             reduction_op,
    int                     tiles,
    clock_t                 *d_elapsed)
{
    const int TILE_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD;

    // Cooperative threadblock reduction utility type (returns aggregate in thread 0)
    typedef BlockReduce<T, BLOCK_THREADS, ALGORITHM> BlockReduce;

    // Allocate temp storage in shared memory
    __shared__ typename BlockReduce::TempStorage temp_storage;

    // Per-thread tile data
    T data[ITEMS_PER_THREAD];

    // Load first tile of data
    int block_offset = 0;
    LoadDirectBlocked(threadIdx.x, d_in + block_offset, data);
    block_offset += TILE_SIZE;

    // Start cycle timer
    clock_t start = clock();

    // Cooperative reduce first tile
    T block_aggregate = DeviceTest<BlockReduce>(temp_storage, data, reduction_op);

    // Stop cycle timer
#if CUB_PTX_VERSION == 100
    // Bug: recording stop clock causes mis-write of running prefix value
    clock_t stop = 0;
#else
    clock_t stop = clock();
#endif // CUB_PTX_VERSION == 100
    clock_t elapsed = (start > stop) ? start - stop : stop - start;

    // Loop over input tiles
    while (block_offset < TILE_SIZE * tiles)
    {
        // TestBarrier between threadblock reductions
        __syncthreads();

        // Load tile of data
        LoadDirectBlocked(threadIdx.x, d_in + block_offset, data);
        block_offset += TILE_SIZE;

        // Start cycle timer
        clock_t start = clock();

        // Cooperatively reduce the tile's aggregate
        T tile_aggregate = DeviceTest<BlockReduce>(temp_storage, data, reduction_op);

        // Stop cycle timer
    #if CUB_PTX_VERSION == 100
        // Bug: recording stop clock causes mis-write of running prefix value
        clock_t stop = 0;
    #else
        clock_t stop = clock();
    #endif // CUB_PTX_VERSION == 100
        elapsed += (start > stop) ? start - stop : stop - start;

        // Reduce threadblock aggregate
        block_aggregate = reduction_op(block_aggregate, tile_aggregate);
    }

    // Store data
    if (threadIdx.x == 0)
    {
        d_out[0] = block_aggregate;
        *d_elapsed = elapsed;
    }
}



/**
 * Test partial-tile reduction kernel (where num_items < BLOCK_THREADS)
 */
template <
    BlockReduceAlgorithm    ALGORITHM,
    int                     BLOCK_THREADS,
    typename                T,
    typename                ReductionOp>
__launch_bounds__ (BLOCK_THREADS, 1)
__global__ void PartialTileReduceKernel(
    T                       *d_in,
    T                       *d_out,
    int                     num_items,
    ReductionOp             reduction_op,
    clock_t                 *d_elapsed)
{
    // Cooperative threadblock reduction utility type (returns aggregate only in thread-0)
    typedef BlockReduce<T, BLOCK_THREADS, ALGORITHM> BlockReduce;

    // Allocate temp storage in shared memory
    __shared__ typename BlockReduce::TempStorage temp_storage;

    // Per-thread tile data
    T partial;

    // Load partial tile data
    if (threadIdx.x < num_items)
    {
        partial = d_in[threadIdx.x];
    }

    // Start cycle timer
    clock_t start = clock();

    // Cooperatively reduce the tile's aggregate
    T tile_aggregate = DeviceTest<BlockReduce>(temp_storage, partial, reduction_op, num_items);

    // Stop cycle timer
#if CUB_PTX_VERSION == 100
    // Bug: recording stop clock causes mis-write of running prefix value
    clock_t stop = 0;
#else
    clock_t stop = clock();
#endif // CUB_PTX_VERSION == 100
    clock_t elapsed = (start > stop) ? start - stop : stop - start;

    // Store data
    if (threadIdx.x == 0)
    {
        d_out[0] = tile_aggregate;
        *d_elapsed = elapsed;
    }
}


//---------------------------------------------------------------------
// Host utility subroutines
//---------------------------------------------------------------------

/**
 * Initialize problem (and solution)
 */
template <
    typename    T,
    typename    ReductionOp>
void Initialize(
    GenMode     gen_mode,
    T           *h_in,
    T           h_reference[1],
    ReductionOp reduction_op,
    int         num_items)
{
    for (int i = 0; i < num_items; ++i)
    {
        InitValue(gen_mode, h_in[i], i);
        if (i == 0)
            h_reference[0] = h_in[0];
        else
            h_reference[0] = reduction_op(h_reference[0], h_in[i]);
    }
}


//---------------------------------------------------------------------
// Full tile test generation
//---------------------------------------------------------------------


/**
 * Test full-tile reduction
 */
template <
    BlockReduceAlgorithm    ALGORITHM,
    int                     BLOCK_THREADS,
    int                     ITEMS_PER_THREAD,
    typename                T,
    typename                ReductionOp>
void TestFullTile(
    GenMode                 gen_mode,
    int                     tiles,
    ReductionOp             reduction_op,
    char                    *type_string)
{
    const int TILE_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD;

    int num_items = TILE_SIZE * tiles;

    // Allocate host arrays
    T *h_in = new T[num_items];
    T h_reference[1];

    // Initialize problem
    Initialize(gen_mode, h_in, h_reference, reduction_op, num_items);

    // Initialize/clear device arrays
    T       *d_in = NULL;
    T       *d_out = NULL;
    clock_t *d_elapsed = NULL;
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_elapsed, sizeof(unsigned long long)));
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_in, sizeof(T) * num_items));
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_out, sizeof(T) * 1));
    CubDebugExit(cudaMemcpy(d_in, h_in, sizeof(T) * num_items, cudaMemcpyHostToDevice));
    CubDebugExit(cudaMemset(d_out, 0, sizeof(T) * 1));

    // Test multi-tile (unguarded)
    printf("TestFullTile %s, gen-mode %d, num_items(%d), BLOCK_THREADS(%d), ITEMS_PER_THREAD(%d), tiles(%d), %s (%d bytes) elements:\n",
        (ALGORITHM == BLOCK_REDUCE_RAKING) ? "BLOCK_REDUCE_RAKING" : (ALGORITHM == BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY) ? "BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY" : "BLOCK_REDUCE_WARP_REDUCTIONS",
        gen_mode,
        num_items,
        BLOCK_THREADS,
        ITEMS_PER_THREAD,
        tiles,
        type_string,
        (int) sizeof(T));
    fflush(stdout);

    FullTileReduceKernel<ALGORITHM, BLOCK_THREADS, ITEMS_PER_THREAD><<<1, BLOCK_THREADS>>>(
        d_in,
        d_out,
        reduction_op,
        tiles,
        d_elapsed);

    CubDebugExit(cudaDeviceSynchronize());

    // Copy out and display results
    printf("\tReduction results: ");
    int compare = CompareDeviceResults(h_reference, d_out, 1, g_verbose, g_verbose);
    printf("%s\n", compare ? "FAIL" : "PASS");
    AssertEquals(0, compare);

    printf("\tElapsed clocks: ");
    DisplayDeviceResults(d_elapsed, 1);

    // Cleanup
    if (h_in) delete[] h_in;
    if (d_in) CubDebugExit(g_allocator.DeviceFree(d_in));
    if (d_out) CubDebugExit(g_allocator.DeviceFree(d_out));
    if (d_elapsed) CubDebugExit(g_allocator.DeviceFree(d_elapsed));
}

/**
 * Run battery of tests for different thread items
 */
template <
    BlockReduceAlgorithm    ALGORITHM,
    int                     BLOCK_THREADS,
    typename                T,
    typename                ReductionOp>
void TestFullTile(
    GenMode                 gen_mode,
    int                     tiles,
    ReductionOp             reduction_op,
    char                    *type_string)
{
    TestFullTile<ALGORITHM, BLOCK_THREADS, 1, T>(gen_mode, tiles, reduction_op, type_string);
    TestFullTile<ALGORITHM, BLOCK_THREADS, 4, T>(gen_mode, tiles, reduction_op, type_string);
}


/**
 * Run battery of full-tile tests for different numbers of tiles
 */
template <
    BlockReduceAlgorithm    ALGORITHM,
    int                     BLOCK_THREADS,
    typename                T,
    typename                ReductionOp>
void TestFullTile(
    GenMode                 gen_mode,
    ReductionOp             reduction_op,
    char                    *type_string)
{
    for (int tiles = 1; tiles < 3; tiles++)
    {
        TestFullTile<ALGORITHM, BLOCK_THREADS, T>(gen_mode, tiles, reduction_op, type_string);
    }
}


//---------------------------------------------------------------------
// Partial-tile test generation
//---------------------------------------------------------------------

/**
 * Test partial-tile reduction
 */
template <
    BlockReduceAlgorithm    ALGORITHM,
    int                     BLOCK_THREADS,
    typename                T,
    typename                ReductionOp>
void TestPartialTile(
    GenMode                 gen_mode,
    int                     num_items,
    ReductionOp             reduction_op,
    char                    *type_string)
{
    const int TILE_SIZE = BLOCK_THREADS;

    // Allocate host arrays
    T *h_in = new T[num_items];
    T h_reference[1];

    // Initialize problem
    Initialize(gen_mode, h_in, h_reference, reduction_op, num_items);

    // Initialize/clear device arrays
    T       *d_in = NULL;
    T       *d_out = NULL;
    clock_t *d_elapsed = NULL;
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_elapsed, sizeof(unsigned long long)));
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_in, sizeof(T) * TILE_SIZE));
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_out, sizeof(T) * 1));
    CubDebugExit(cudaMemcpy(d_in, h_in, sizeof(T) * num_items, cudaMemcpyHostToDevice));
    CubDebugExit(cudaMemset(d_out, 0, sizeof(T) * 1));

    printf("TestPartialTile %s, gen-mode %d, num_items(%d), BLOCK_THREADS(%d), %s (%d bytes) elements:\n",
        (ALGORITHM == BLOCK_REDUCE_RAKING) ? "BLOCK_REDUCE_RAKING" : (ALGORITHM == BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY) ? "BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY" : "BLOCK_REDUCE_WARP_REDUCTIONS",
        gen_mode,
        num_items,
        BLOCK_THREADS,
        type_string,
        (int) sizeof(T));
    fflush(stdout);

    PartialTileReduceKernel<ALGORITHM, BLOCK_THREADS><<<1, BLOCK_THREADS>>>(
        d_in,
        d_out,
        num_items,
        reduction_op,
        d_elapsed);

    CubDebugExit(cudaDeviceSynchronize());

    // Copy out and display results
    printf("\tReduction results: ");
    int compare = CompareDeviceResults(h_reference, d_out, 1, g_verbose, g_verbose);
    printf("%s\n", compare ? "FAIL" : "PASS");
    AssertEquals(0, compare);

    printf("\tElapsed clocks: ");
    DisplayDeviceResults(d_elapsed, 1);

    // Cleanup
    if (h_in) delete[] h_in;
    if (d_in) CubDebugExit(g_allocator.DeviceFree(d_in));
    if (d_out) CubDebugExit(g_allocator.DeviceFree(d_out));
    if (d_elapsed) CubDebugExit(g_allocator.DeviceFree(d_elapsed));
}


/**
 *  Run battery of partial-tile tests for different numbers of effective threads
 */
template <
    BlockReduceAlgorithm    ALGORITHM,
    int                     BLOCK_THREADS,
    typename                T,
    typename                ReductionOp>
void TestPartialTile(
    GenMode                 gen_mode,
    ReductionOp             reduction_op,
    char                    *type_string)
{
    for (
        int num_items = 1;
        num_items < BLOCK_THREADS;
        num_items += CUB_MAX(1, BLOCK_THREADS / 5))
    {
        TestPartialTile<ALGORITHM, BLOCK_THREADS, T>(gen_mode, num_items, reduction_op, type_string);
    }
}



//---------------------------------------------------------------------
// Main
//---------------------------------------------------------------------

/**
 * Run battery of full-tile tests for different gen modes
 */
template <
    BlockReduceAlgorithm    ALGORITHM,
    int                     BLOCK_THREADS,
    typename                T,
    typename                ReductionOp>
void Test(
    ReductionOp             reduction_op,
    char                    *type_string)
{
    TestFullTile<ALGORITHM, BLOCK_THREADS, T>(UNIFORM, reduction_op, type_string);
    TestPartialTile<ALGORITHM, BLOCK_THREADS, T>(UNIFORM, reduction_op, type_string);

    TestFullTile<ALGORITHM, BLOCK_THREADS, T>(INTEGER_SEED, reduction_op, type_string);
    TestPartialTile<ALGORITHM, BLOCK_THREADS, T>(INTEGER_SEED, reduction_op, type_string);

    if (Traits<T>::CATEGORY != FLOATING_POINT)
    {
        // Don't test randomly-generated floats b/c of stability
        TestFullTile<ALGORITHM, BLOCK_THREADS, T>(RANDOM, reduction_op, type_string);
        TestPartialTile<ALGORITHM, BLOCK_THREADS, T>(RANDOM, reduction_op, type_string);
    }
}


/**
 * Run battery of tests for different block-reduction algorithmic variants
 */
template <
    int             BLOCK_THREADS,
    typename        T,
    typename        ReductionOp>
void Test(
    ReductionOp     reduction_op,
    char            *type_string)
{
    Test<BLOCK_REDUCE_RAKING, BLOCK_THREADS, T>(reduction_op, type_string);
    Test<BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY, BLOCK_THREADS, T>(reduction_op, type_string);
    Test<BLOCK_REDUCE_WARP_REDUCTIONS, BLOCK_THREADS, T>(reduction_op, type_string);
}


/**
 * Run battery of tests for different block sizes
 */
template <
    typename        T,
    typename        ReductionOp>
void Test(
    ReductionOp     reduction_op,
    char            *type_string)
{
    Test<7, T>(reduction_op, type_string);
    Test<32, T>(reduction_op, type_string);
    Test<63, T>(reduction_op, type_string);
    Test<97, T>(reduction_op, type_string);
    Test<128, T>(reduction_op, type_string);
    Test<238, T>(reduction_op, type_string);
}


/**
 * Run battery of tests for different block sizes
 */
template <typename T>
void Test(char* type_string)
{
    Test<T>(Sum(), type_string);
    Test<T>(Max(), type_string);
}


/**
 * Main
 */
int main(int argc, char** argv)
{
    // Initialize command line
    CommandLineArgs args(argc, argv);
    g_verbose = args.CheckCmdLineFlag("v");
    args.GetCmdLineArgument("repeat", g_repeat);

    // Print usage
    if (args.CheckCmdLineFlag("help"))
    {
        printf("%s "
            "[--device=<device-id>] "
            "[--repeat=<repetitions of entire test suite>]"
            "[--v] "
            "\n", argv[0]);
        exit(0);
    }

    // Initialize device
    CubDebugExit(args.DeviceInit());

    // Get ptx version
    int ptx_version;
    CubDebugExit(PtxVersion(ptx_version));

#ifdef QUICK_TEST

    // Compile/run quick tests
    typedef int T;

    printf("\n full tile ------------------------\n\n");

    TestFullTile<BLOCK_REDUCE_RAKING,                   128, 4, int>(UNIFORM, 1, Sum(), CUB_TYPE_STRING(int));
    TestFullTile<BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY,  128, 4, int>(UNIFORM, 1, Sum(), CUB_TYPE_STRING(int));
    TestFullTile<BLOCK_REDUCE_WARP_REDUCTIONS,          128, 4, int>(UNIFORM, 1, Sum(), CUB_TYPE_STRING(int));

    TestFullTile<BLOCK_REDUCE_RAKING,                   128, 1, int>(UNIFORM, 1, Sum(), CUB_TYPE_STRING(int));
    TestFullTile<BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY,  128, 1, int>(UNIFORM, 1, Sum(), CUB_TYPE_STRING(int));
    TestFullTile<BLOCK_REDUCE_WARP_REDUCTIONS,          128, 1, int>(UNIFORM, 1, Sum(), CUB_TYPE_STRING(int));

    printf("\n partial tile ------------------------\n\n");

    TestPartialTile<BLOCK_REDUCE_RAKING,                   128, int>(UNIFORM, 1, Sum(), CUB_TYPE_STRING(int));
    TestPartialTile<BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY,  128, int>(UNIFORM, 1, Sum(), CUB_TYPE_STRING(int));
    TestPartialTile<BLOCK_REDUCE_WARP_REDUCTIONS,          128, int>(UNIFORM, 1, Sum(), CUB_TYPE_STRING(int));

#else

    // Compile/run thorough tests
    for (int i = 0; i <= g_repeat; ++i)
    {
        // primitives
        Test<char>(CUB_TYPE_STRING(char));
        Test<short>(CUB_TYPE_STRING(short));
        Test<int>(CUB_TYPE_STRING(int));
        Test<long long>(CUB_TYPE_STRING(long long));
        if (ptx_version > 100)                          // Don't check doubles on PTX100 because they're down-converted
            Test<double>(CUB_TYPE_STRING(double));

        // vector types
        Test<char2>(CUB_TYPE_STRING(char2));
        Test<short2>(CUB_TYPE_STRING(short2));
        Test<int2>(CUB_TYPE_STRING(int2));
        Test<longlong2>(CUB_TYPE_STRING(longlong2));

        Test<char4>(CUB_TYPE_STRING(char4));
        Test<short4>(CUB_TYPE_STRING(short4));
        Test<int4>(CUB_TYPE_STRING(int4));
        Test<longlong4>(CUB_TYPE_STRING(longlong4));

        // Complex types
        Test<TestFoo>(CUB_TYPE_STRING(TestFoo));
        Test<TestBar>(CUB_TYPE_STRING(TestBar));
    }

#endif

    return 0;
}



