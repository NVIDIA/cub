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
 * Test of BlockReduce utilities
 ******************************************************************************/

// Ensure printing of CUDA runtime errors to console
#define CUB_STDERR

#include <stdio.h>
#include <cub/cub.cuh>
#include "test_util.h"

using namespace cub;


//---------------------------------------------------------------------
// Globals, constants and typedefs
//---------------------------------------------------------------------

bool g_verbose = false;


//---------------------------------------------------------------------
// Test kernels
//---------------------------------------------------------------------


/**
 * Generic reduction
 */
template <
    typename    T,
    typename    ReductionOp,
    bool        PRIMITIVE = Traits<T>::PRIMITIVE>
struct DeviceTest
{
    template <
        typename    BlockReduce,
        int         ITEMS_PER_THREAD>
    static __device__ __forceinline__ T Test(
        typename BlockReduce::SmemStorage   &smem_storage,
        T                                   (&data)[ITEMS_PER_THREAD],
        ReductionOp                         &reduction_op)
    {
        return BlockReduce::Reduce(smem_storage, data, reduction_op);
    }

    template < typename BlockReduce>
    static __device__ __forceinline__ T Test(
        typename BlockReduce::SmemStorage   &smem_storage,
        T                                   &data,
        ReductionOp                         &reduction_op,
        int                                 valid_threads)
    {
        return BlockReduce::Reduce(smem_storage, data, reduction_op, valid_threads);
    }
};


/**
 * Sum reduction (only compile for primitive, built-ins only)
 */
template <typename T>
struct DeviceTest<T, Sum<T>, true>
{
    template <
        typename    BlockReduce,
        int         ITEMS_PER_THREAD>
    static __device__ __forceinline__ T Test(
        typename BlockReduce::SmemStorage   &smem_storage,
        T                                   (&data)[ITEMS_PER_THREAD],
        Sum<T>                              &reduction_op)
    {
        return BlockReduce::Sum(smem_storage, data);
    }

    template <typename BlockReduce>
    static __device__ __forceinline__ T Test(
        typename BlockReduce::SmemStorage   &smem_storage,
        T                                   &data,
        Sum<T>                              &reduction_op,
        int                                 valid_threads)
    {
        return BlockReduce::Sum(smem_storage, data, valid_threads);
    }
};


/**
 * Test full-tile reduction kernel (where num_elements is an even
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
    int                     tiles)
{
    const int TILE_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD;

    // Cooperative threadblock reduction utility type (returns aggregate in thread 0)
    typedef BlockReduce<T, BLOCK_THREADS, ALGORITHM> BlockReduce;

    // Shared memory
    __shared__ typename BlockReduce::SmemStorage smem_storage;

    // Per-thread tile data
    T data[ITEMS_PER_THREAD];

    // Load first tile of data
    int block_offset = 0;
    BlockLoadDirect(d_in + block_offset, data);
    block_offset += TILE_SIZE;

    // Cooperative reduce first tile
    T block_aggregate = DeviceTest<T, ReductionOp>::template Test<BlockReduce>(smem_storage, data, reduction_op);

    // Loop over input tiles
    while (block_offset < TILE_SIZE * tiles)
    {
        // TestBarrier between threadblock reductions
        __syncthreads();

        // Load tile of data
        BlockLoadDirect(d_in + block_offset, data);
        block_offset += TILE_SIZE;

        // Cooperatively reduce the tile's aggregate
        T tile_aggregate = DeviceTest<T, ReductionOp>::template Test<BlockReduce>(smem_storage, data, reduction_op);

        // Reduce threadblock aggregate
        block_aggregate = reduction_op(block_aggregate, tile_aggregate);
    }

    // Store data
    if (threadIdx.x == 0)
    {
        d_out[0] = block_aggregate;
    }
}



/**
 * Test partial-tile reduction kernel (where num_elements < BLOCK_THREADS)
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
    int                     num_elements,
    ReductionOp             reduction_op)
{
    // Cooperative threadblock reduction utility type (returns aggregate only in thread-0)
    typedef BlockReduce<T, BLOCK_THREADS, ALGORITHM> BlockReduce;

    // Shared memory
    __shared__ typename BlockReduce::SmemStorage smem_storage;

    // Per-thread tile data
    T partial;

    // Load partial tile data
    if (threadIdx.x < num_elements)
    {
        partial = d_in[threadIdx.x];
    }

    // Cooperatively reduce the tile's aggregate
    T tile_aggregate = DeviceTest<T, ReductionOp>::template Test<BlockReduce>(smem_storage, partial, reduction_op, num_elements);

    // Store data
    if (threadIdx.x == 0)
    {
        d_out[0] = tile_aggregate;
    }
}


//---------------------------------------------------------------------
// Host utility subroutines
//---------------------------------------------------------------------

/**
 * Initialize problem (and solution)
 */
template <
    typename        T,
    typename        ReductionOp>
void Initialize(
    int             gen_mode,
    T               *h_in,
    T               h_reference[1],
    ReductionOp     reduction_op,
    int             num_elements)
{
    for (int i = 0; i < num_elements; ++i)
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
    int                     gen_mode,
    int                     tiles,
    ReductionOp             reduction_op,
    char                    *type_string)
{
    const int TILE_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD;

    int num_elements = TILE_SIZE * tiles;

    // Allocate host arrays
    T *h_in = new T[num_elements];
    T h_reference[1];

    // Initialize problem
    Initialize(gen_mode, h_in, h_reference, reduction_op, num_elements);

    // Initialize device arrays
    T *d_in = NULL;
    T *d_out = NULL;
    CubDebugExit(DeviceAllocate((void**)&d_in, sizeof(T) * num_elements));
    CubDebugExit(DeviceAllocate((void**)&d_out, sizeof(T) * 1));
    CubDebugExit(cudaMemcpy(d_in, h_in, sizeof(T) * num_elements, cudaMemcpyHostToDevice));
    CubDebugExit(cudaMemset(d_out, 0, sizeof(T) * 1));

    // Test multi-tile (unguarded)
    printf("TestFullTile, gen-mode %d, num_elements(%d), BLOCK_THREADS(%d), ITEMS_PER_THREAD(%d), tiles(%d), %s (%d bytes) elements:\n",
        gen_mode,
        num_elements,
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
        tiles);

    CubDebugExit(cudaDeviceSynchronize());

    // Copy out and display results
    printf("\tReduction results: ");
    int compare = CompareDeviceResults(h_reference, d_out, 1, g_verbose, g_verbose);
    printf("%s\n", compare ? "FAIL" : "PASS");
    AssertEquals(0, compare);

    // Cleanup
    if (h_in) delete[] h_in;
    if (d_in) CubDebugExit(DeviceFree(d_in));
    if (d_out) CubDebugExit(DeviceFree(d_out));
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
    int                     gen_mode,
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
    int                     gen_mode,
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
    int                     gen_mode,
    int                     num_elements,
    ReductionOp             reduction_op,
    char                    *type_string)
{
    const int TILE_SIZE = BLOCK_THREADS;

    // Allocate host arrays
    T *h_in = new T[num_elements];
    T h_reference[1];

    // Initialize problem
    Initialize(gen_mode, h_in, h_reference, reduction_op, num_elements);

    // Initialize device arrays
    T *d_in = NULL;
    T *d_out = NULL;
    CubDebugExit(DeviceAllocate((void**)&d_in, sizeof(T) * TILE_SIZE));
    CubDebugExit(DeviceAllocate((void**)&d_out, sizeof(T) * 1));
    CubDebugExit(cudaMemcpy(d_in, h_in, sizeof(T) * num_elements, cudaMemcpyHostToDevice));
    CubDebugExit(cudaMemset(d_out, 0, sizeof(T) * 1));

    printf("TestPartialTile, gen-mode %d, num_elements(%d), BLOCK_THREADS(%d), %s (%d bytes) elements:\n",
        gen_mode,
        num_elements,
        BLOCK_THREADS,
        type_string,
        (int) sizeof(T));
    fflush(stdout);

    PartialTileReduceKernel<ALGORITHM, BLOCK_THREADS><<<1, BLOCK_THREADS>>>(
        d_in,
        d_out,
        num_elements,
        reduction_op);

    CubDebugExit(cudaDeviceSynchronize());

    // Copy out and display results
    printf("\tReduction results: ");
    int compare = CompareDeviceResults(h_reference, d_out, 1, g_verbose, g_verbose);
    printf("%s\n", compare ? "FAIL" : "PASS");
    AssertEquals(0, compare);

    // Cleanup
    if (h_in) delete[] h_in;
    if (d_in) CubDebugExit(DeviceFree(d_in));
    if (d_out) CubDebugExit(DeviceFree(d_out));
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
    int                     gen_mode,
    ReductionOp             reduction_op,
    char                    *type_string)
{
    for (
        int num_elements = 1;
        num_elements < BLOCK_THREADS;
        num_elements += CUB_MAX(1, BLOCK_THREADS / 5))
    {
        TestPartialTile<ALGORITHM, BLOCK_THREADS, T>(gen_mode, num_elements, reduction_op, type_string);
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

    TestFullTile<ALGORITHM, BLOCK_THREADS, T>(SEQ_INC, reduction_op, type_string);
    TestPartialTile<ALGORITHM, BLOCK_THREADS, T>(SEQ_INC, reduction_op, type_string);

    TestFullTile<ALGORITHM, BLOCK_THREADS, T>(RANDOM, reduction_op, type_string);
    TestPartialTile<ALGORITHM, BLOCK_THREADS, T>(RANDOM, reduction_op, type_string);
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
    Test<65, T>(reduction_op, type_string);
    Test<128, T>(reduction_op, type_string);
}


/**
 * Run battery of tests for different block sizes
 */
template <typename T>
void Test(char* type_string)
{
    Test<T>(Sum<T>(), type_string);
    Test<T>(Max<T>(), type_string);
}


/**
 * Main
 */
int main(int argc, char** argv)
{
    // Initialize command line
    CommandLineArgs args(argc, argv);
    g_verbose = args.CheckCmdLineFlag("v");
    bool quick = args.CheckCmdLineFlag("quick");

    // Print usage
    if (args.CheckCmdLineFlag("help"))
    {
        printf("%s "
            "[--device=<device-id>] "
            "[--v] "
            "[--quick]"
            "\n", argv[0]);
        exit(0);
    }

    // Initialize device
    CubDebugExit(args.DeviceInit());

    if (quick)
    {
        // Quick test
        typedef int T;
        TestFullTile<BLOCK_REDUCE_RAKING, 128, 4, T>(UNIFORM, 1, Sum<T>(), CUB_TYPE_STRING(T));
    }
    else
    {
        // primitives
        Test<char>(CUB_TYPE_STRING(char));
        Test<short>(CUB_TYPE_STRING(short));
        Test<int>(CUB_TYPE_STRING(int));
        Test<long long>(CUB_TYPE_STRING(long long));

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

    return 0;
}



