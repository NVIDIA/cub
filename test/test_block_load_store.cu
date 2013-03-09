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
 * Test of BlockLoad and BlockStore utilities
 ******************************************************************************/

// Ensure printing of CUDA runtime errors to console
#define CUB_STDERR

#include <iterator>
#include <stdio.h>
#include <test_util.h>

#include <thrust/iterator/counting_iterator.h>

#include "../cub.cuh"

using namespace cub;

//---------------------------------------------------------------------
// Globals, constants and typedefs
//---------------------------------------------------------------------

bool g_verbose = false;


//---------------------------------------------------------------------
// Test kernels
//---------------------------------------------------------------------


/**
 * Test load/store kernel.
 */
template <
    int                 BLOCK_THREADS,
    int                 ITEMS_PER_THREAD,
    BlockLoadPolicy       LOAD_POLICY,
    BlockStorePolicy      STORE_POLICY,
    PtxLoadModifier     LOAD_MODIFIER,
    PtxStoreModifier    STORE_MODIFIER,
    typename            InputIterator,
    typename            OutputIterator>
__launch_bounds__ (BLOCK_THREADS, 1)
__global__ void Kernel(
    InputIterator       d_in,
    OutputIterator      d_out_unguarded,
    OutputIterator      d_out_guarded_range,
    int                 num_elements)
{
    enum
    {
        TILE_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD
    };

    // Data type of input/output iterators
    typedef typename std::iterator_traits<InputIterator>::value_type T;

    // Threadblock load/store abstraction types
    typedef BlockLoad<InputIterator, BLOCK_THREADS, ITEMS_PER_THREAD, LOAD_POLICY, LOAD_MODIFIER> BlockLoad;
    typedef BlockStore<OutputIterator, BLOCK_THREADS, ITEMS_PER_THREAD, STORE_POLICY, STORE_MODIFIER> BlockStore;

    // Shared memory type for this threadblock
    union SmemStorage
    {
        typename BlockLoad::SmemStorage     load;
        typename BlockStore::SmemStorage     store;
    };

    // Shared memory
    __shared__ SmemStorage smem_storage;

    // Threadblock work bounds
    int block_offset = blockIdx.x * TILE_SIZE;
    int guarded_elements = num_elements - block_offset;

    // Test unguarded
    {
        // Tile of items
        T data[ITEMS_PER_THREAD];

        // Load data
        BlockLoad::Load(smem_storage.load, d_in + block_offset, data);

        __syncthreads();

        // Store data
        BlockStore::Store(smem_storage.store, d_out_unguarded + block_offset, data);
    }

    __syncthreads();

    // Test guarded by range
    {
        // Tile of items
        T data[ITEMS_PER_THREAD];

        // Load data
        BlockLoad::Load(smem_storage.load, d_in + block_offset, guarded_elements, data);

        __syncthreads();

        // Store data
        BlockStore::Store(smem_storage.store, d_out_guarded_range + block_offset, guarded_elements, data);
    }
}


//---------------------------------------------------------------------
// Host testing subroutines
//---------------------------------------------------------------------


/**
 * Test load/store variants
 */
template <
    typename            T,
    int                 BLOCK_THREADS,
    int                 ITEMS_PER_THREAD,
    BlockLoadPolicy       LOAD_POLICY,
    BlockStorePolicy      STORE_POLICY,
    PtxLoadModifier     LOAD_MODIFIER,
    PtxStoreModifier    STORE_MODIFIER,
    typename            InputIterator,
    typename            OutputIterator>
void TestKernel(
    T                   *h_in,
    InputIterator       d_in,
    OutputIterator      d_out_unguarded,
    OutputIterator      d_out_guarded_range,
    int                 grid_size,
    int                 guarded_elements)
{
    int unguarded_elements = grid_size * BLOCK_THREADS * ITEMS_PER_THREAD;

    // Run kernel
    Kernel<BLOCK_THREADS, ITEMS_PER_THREAD, LOAD_POLICY, STORE_POLICY, LOAD_MODIFIER, STORE_MODIFIER>
        <<<grid_size, BLOCK_THREADS>>>(
            d_in,
            d_out_unguarded,
            d_out_guarded_range,
            guarded_elements);

    CubDebugExit(cudaDeviceSynchronize());

    // Check results
    printf("\tUnguarded: ");
    AssertEquals(0, CompareDeviceResults(h_in, d_out_unguarded, unguarded_elements, g_verbose, g_verbose));
    printf("\n");

    printf("\tGuarded range: ");
    AssertEquals(0, CompareDeviceResults(h_in, d_out_guarded_range, guarded_elements, g_verbose, g_verbose));
    printf("\n");
}


/**
 * Test native pointer
 */
template <
    typename            T,
    int                 BLOCK_THREADS,
    int                 ITEMS_PER_THREAD,
    BlockLoadPolicy       LOAD_POLICY,
    BlockStorePolicy      STORE_POLICY,
    PtxLoadModifier     LOAD_MODIFIER,
    PtxStoreModifier    STORE_MODIFIER>
void TestNative(
    int                 grid_size,
    float               fraction_valid)
{
    int unguarded_elements = grid_size * BLOCK_THREADS * ITEMS_PER_THREAD;
    int guarded_elements = int(fraction_valid * float(unguarded_elements));

    // Allocate host arrays
    T *h_in = (T*) malloc(unguarded_elements * sizeof(T));

    // Allocate device arrays
    T *d_in = NULL;
    T *d_out_unguarded = NULL;
    T *d_out_guarded_range = NULL;
    CubDebugExit(cudaMalloc((void**)&d_in, sizeof(T) * unguarded_elements));
    CubDebugExit(cudaMalloc((void**)&d_out_unguarded, sizeof(T) * unguarded_elements));
    CubDebugExit(cudaMalloc((void**)&d_out_guarded_range, sizeof(T) * guarded_elements));

    // Initialize problem on host and device
    for (int i = 0; i < unguarded_elements; ++i)
    {
        RandomBits(h_in[i]);
    }
    CubDebugExit(cudaMemcpy(d_in, h_in, sizeof(T) * unguarded_elements, cudaMemcpyHostToDevice));

    printf("TestNative "
        "grid_size(%d) "
        "guarded_elements(%d) "
        "unguarded_elements(%d) "
        "BLOCK_THREADS(%d) "
        "ITEMS_PER_THREAD(%d) "
        "LOAD_POLICY(%d) "
        "STORE_POLICY(%d) "
        "LOAD_MODIFIER(%d) "
        "STORE_MODIFIER(%d) "
        "sizeof(T)(%d)\n",
            grid_size, guarded_elements, unguarded_elements, BLOCK_THREADS, ITEMS_PER_THREAD, LOAD_POLICY, STORE_POLICY, LOAD_MODIFIER, STORE_MODIFIER, (int) sizeof(T));

    TestKernel<T, BLOCK_THREADS, ITEMS_PER_THREAD, LOAD_POLICY, STORE_POLICY, LOAD_MODIFIER, STORE_MODIFIER>(
        h_in,
        d_in,
        d_out_unguarded,
        d_out_guarded_range,
        grid_size,
        guarded_elements);

    // Cleanup
    if (h_in) free(h_in);
    if (d_in) CubDebugExit(cudaFree(d_in));
    if (d_out_unguarded) CubDebugExit(cudaFree(d_out_unguarded));
    if (d_out_guarded_range) CubDebugExit(cudaFree(d_out_guarded_range));
}


/**
 * Test iterator
 */
template <
    typename        T,
    int             BLOCK_THREADS,
    int             ITEMS_PER_THREAD,
    BlockLoadPolicy   LOAD_POLICY,
    BlockStorePolicy  STORE_POLICY>
void TestIterator(
    int             grid_size,
    float           fraction_valid)
{
    int unguarded_elements = grid_size * BLOCK_THREADS * ITEMS_PER_THREAD;
    int guarded_elements = int(fraction_valid * float(unguarded_elements));

    // Allocate host arrays
    T *h_in = (T*) malloc(unguarded_elements * sizeof(T));

    // Allocate device arrays
    T *d_out_unguarded = NULL;
    T *d_out_guarded_range = NULL;
    CubDebugExit(cudaMalloc((void**)&d_out_unguarded, sizeof(T) * unguarded_elements));
    CubDebugExit(cudaMalloc((void**)&d_out_guarded_range, sizeof(T) * guarded_elements));

    // Initialize problem on host and device
    thrust::counting_iterator<T> counting_itr(0);
    for (int i = 0; i < unguarded_elements; ++i)
    {
        h_in[i] = counting_itr[i];
    }

    printf("TestIterator "
        "grid_size(%d) "
        "guarded_elements(%d) "
        "unguarded_elements(%d) "
        "BLOCK_THREADS(%d) "
        "ITEMS_PER_THREAD(%d) "
        "LOAD_POLICY(%d) "
        "STORE_POLICY(%d) "
        "sizeof(T)(%d)\n",
            grid_size, guarded_elements, unguarded_elements, BLOCK_THREADS, ITEMS_PER_THREAD, LOAD_POLICY, STORE_POLICY, (int) sizeof(T));

    TestKernel<T, BLOCK_THREADS, ITEMS_PER_THREAD, LOAD_POLICY, STORE_POLICY, PTX_LOAD_NONE, PTX_STORE_NONE>(
        h_in,
        counting_itr,
        d_out_unguarded,
        d_out_guarded_range,
        grid_size,
        guarded_elements);

    // Cleanup
    if (h_in) free(h_in);
    if (d_out_unguarded) CubDebugExit(cudaFree(d_out_unguarded));
    if (d_out_guarded_range) CubDebugExit(cudaFree(d_out_guarded_range));
}


/**
 * Evaluate different pointer access types
 */
template <
    typename        T,
    int             BLOCK_THREADS,
    int             ITEMS_PER_THREAD,
    BlockLoadPolicy   LOAD_POLICY,
    BlockStorePolicy  STORE_POLICY>
void TestPointerAccess(
    int             grid_size,
    float           fraction_valid)
{
    TestNative<T, BLOCK_THREADS, ITEMS_PER_THREAD, LOAD_POLICY, STORE_POLICY, PTX_LOAD_NONE, PTX_STORE_NONE>(grid_size, fraction_valid);
    TestNative<T, BLOCK_THREADS, ITEMS_PER_THREAD, LOAD_POLICY, STORE_POLICY, PTX_LOAD_CG, PTX_STORE_CG>(grid_size, fraction_valid);
    TestNative<T, BLOCK_THREADS, ITEMS_PER_THREAD, LOAD_POLICY, STORE_POLICY, PTX_LOAD_CS, PTX_STORE_CS>(grid_size, fraction_valid);
    TestIterator<T, BLOCK_THREADS, ITEMS_PER_THREAD, LOAD_POLICY, STORE_POLICY>(grid_size, fraction_valid);
}


/**
 * Evaluate different load/store strategies
 */
template <
    typename        T,
    int             BLOCK_THREADS,
    int             ITEMS_PER_THREAD>
void TestStrategy(
    int             grid_size,
    float           fraction_valid)
{
    TestPointerAccess<T, BLOCK_THREADS, ITEMS_PER_THREAD, BLOCK_LOAD_DIRECT, BLOCK_STORE_DIRECT>(grid_size, fraction_valid);
    TestPointerAccess<T, BLOCK_THREADS, ITEMS_PER_THREAD, BLOCK_LOAD_TRANSPOSE, BLOCK_STORE_TRANSPOSE>(grid_size, fraction_valid);
    TestPointerAccess<T, BLOCK_THREADS, ITEMS_PER_THREAD, BLOCK_LOAD_VECTORIZE, BLOCK_STORE_VECTORIZE>(grid_size, fraction_valid);
}


/**
 * Evaluate different register blocking
 */
template <
    typename T,
    int BLOCK_THREADS>
void TestItemsPerThread(
    int grid_size,
    float fraction_valid)
{
    TestStrategy<T, BLOCK_THREADS, 1>(grid_size, fraction_valid);
    TestStrategy<T, BLOCK_THREADS, 3>(grid_size, fraction_valid);
    TestStrategy<T, BLOCK_THREADS, 4>(grid_size, fraction_valid);
    TestStrategy<T, BLOCK_THREADS, 8>(grid_size, fraction_valid);
}


/**
 * Evaluate different threadblock sizes
 */
template <typename T>
void TestThreads(
    int grid_size,
    float fraction_valid)
{
    TestItemsPerThread<T, 15>(grid_size, fraction_valid);
    TestItemsPerThread<T, 32>(grid_size, fraction_valid);
    TestItemsPerThread<T, 96>(grid_size, fraction_valid);
    TestItemsPerThread<T, 128>(grid_size, fraction_valid);
}


/**
 * Main
 */
int main(int argc, char** argv)
{
    // Initialize command line
    CommandLineArgs args(argc, argv);
    g_verbose = args.CheckCmdLineFlag("v");

    // Print usage
    if (args.CheckCmdLineFlag("help"))
    {
        printf("%s "
            "[--device=<device-id>] "
            "[--v] "
            "\n", argv[0]);
        exit(0);
    }

    // Initialize device
    CubDebugExit(args.DeviceInit());

    // Evaluate different data types
    TestThreads<int>(2, 0.8);

    return 0;
}



