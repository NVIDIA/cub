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

#include <thrust/iterator/counting_iterator.h>

#include <cub/util_allocator.cuh>
#include <cub/block/block_load.cuh>
#include <cub/block/block_store.cuh>

#include "test_util.h"

using namespace cub;

//---------------------------------------------------------------------
// Globals, constants and typedefs
//---------------------------------------------------------------------

bool                    g_verbose = false;
CachingDeviceAllocator  g_allocator;


//---------------------------------------------------------------------
// Test kernels
//---------------------------------------------------------------------


/**
 * Test load/store kernel (unguarded)
 */
template <
    int                 BLOCK_THREADS,
    int                 ITEMS_PER_THREAD,
    BlockLoadAlgorithm  LOAD_ALGORITHM,
    BlockStoreAlgorithm STORE_ALGORITHM,
    PtxLoadModifier     LOAD_MODIFIER,
    PtxStoreModifier    STORE_MODIFIER,
    int                 WARP_TIME_SLICING,
    typename            InputIteratorRA,
    typename            OutputIteratorRA>
__launch_bounds__ (BLOCK_THREADS, 1)
__global__ void Kernel(
    InputIteratorRA     d_in,
    OutputIteratorRA    d_out_unguarded)
{
    enum
    {
        TILE_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD
    };

    // Data type of input/output iterators
    typedef typename std::iterator_traits<InputIteratorRA>::value_type T;

    // Threadblock load/store abstraction types
    typedef BlockLoad<InputIteratorRA, BLOCK_THREADS, ITEMS_PER_THREAD, LOAD_ALGORITHM, LOAD_MODIFIER, WARP_TIME_SLICING> BlockLoad;
    typedef BlockStore<OutputIteratorRA, BLOCK_THREADS, ITEMS_PER_THREAD, STORE_ALGORITHM, STORE_MODIFIER, WARP_TIME_SLICING> BlockStore;

    // Shared memory type for this threadblock
    union TempStorage
    {
        typename BlockLoad::TempStorage     load;
        typename BlockStore::TempStorage    store;
    };

    // Allocate temp storage in shared memory
    __shared__ TempStorage temp_storage;

    // Threadblock work bounds
    int block_offset = blockIdx.x * TILE_SIZE;

    // Tile of items
    T data[ITEMS_PER_THREAD];

    // Load data
    BlockLoad(temp_storage.load).Load(d_in + block_offset, data);

    __syncthreads();

    // Store data
    BlockStore(temp_storage.store).Store(d_out_unguarded + block_offset, data);

}


/**
 * Test load/store kernel.
 */
template <
    int                 BLOCK_THREADS,
    int                 ITEMS_PER_THREAD,
    BlockLoadAlgorithm  LOAD_ALGORITHM,
    BlockStoreAlgorithm STORE_ALGORITHM,
    PtxLoadModifier     LOAD_MODIFIER,
    PtxStoreModifier    STORE_MODIFIER,
    int                 WARP_TIME_SLICING,
    typename            InputIteratorRA,
    typename            OutputIteratorRA>
__launch_bounds__ (BLOCK_THREADS, 1)
__global__ void KernelGuarded(
    InputIteratorRA     d_in,
    OutputIteratorRA    d_out_guarded,
    int                 num_items)
{
    enum
    {
        TILE_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD
    };

    // Data type of input/output iterators
    typedef typename std::iterator_traits<InputIteratorRA>::value_type T;

    // Threadblock load/store abstraction types
    typedef BlockLoad<InputIteratorRA, BLOCK_THREADS, ITEMS_PER_THREAD, LOAD_ALGORITHM, LOAD_MODIFIER, WARP_TIME_SLICING> BlockLoad;
    typedef BlockStore<OutputIteratorRA, BLOCK_THREADS, ITEMS_PER_THREAD, STORE_ALGORITHM, STORE_MODIFIER, WARP_TIME_SLICING> BlockStore;

    // Shared memory type for this threadblock
    union TempStorage
    {
        typename BlockLoad::TempStorage     load;
        typename BlockStore::TempStorage    store;
    };

    // Allocate temp storage in shared memory
    __shared__ TempStorage temp_storage;

    // Threadblock work bounds
    int block_offset = blockIdx.x * TILE_SIZE;
    int guarded_elements = num_items - block_offset;

    // Tile of items
    T data[ITEMS_PER_THREAD];

    // Load data
    BlockLoad(temp_storage.load).Load(d_in + block_offset, data, guarded_elements);

    __syncthreads();

    // Store data
    BlockStore(temp_storage.store).Store(d_out_guarded + block_offset, data, guarded_elements);
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
    BlockLoadAlgorithm  LOAD_ALGORITHM,
    BlockStoreAlgorithm STORE_ALGORITHM,
    PtxLoadModifier     LOAD_MODIFIER,
    PtxStoreModifier    STORE_MODIFIER,
    int                 WARP_TIME_SLICING,
    typename            InputIteratorRA,
    typename            OutputIteratorRA>
void TestKernel(
    T                   *h_in,
    InputIteratorRA     d_in,
    OutputIteratorRA    d_out_unguarded,
    OutputIteratorRA    d_out_guarded,
    int                 grid_size,
    int                 guarded_elements)
{
    int compare;

    int unguarded_elements = grid_size * BLOCK_THREADS * ITEMS_PER_THREAD;

    // Run unguarded kernel
    Kernel<BLOCK_THREADS, ITEMS_PER_THREAD, LOAD_ALGORITHM, STORE_ALGORITHM, LOAD_MODIFIER, STORE_MODIFIER, WARP_TIME_SLICING>
        <<<grid_size, BLOCK_THREADS>>>(
            d_in,
            d_out_unguarded);

    CubDebugExit(cudaDeviceSynchronize());

    // Check results
    compare = CompareDeviceResults(h_in, d_out_unguarded, unguarded_elements, g_verbose, g_verbose);
    printf("\tUnguarded: %s\n", (compare) ? "FAIL" : "PASS");
    AssertEquals(0, compare);

    // Run guarded kernel
    KernelGuarded<BLOCK_THREADS, ITEMS_PER_THREAD, LOAD_ALGORITHM, STORE_ALGORITHM, LOAD_MODIFIER, STORE_MODIFIER, WARP_TIME_SLICING>
        <<<grid_size, BLOCK_THREADS>>>(
            d_in,
            d_out_guarded,
            guarded_elements);

    CubDebugExit(cudaDeviceSynchronize());

    // Check results
    compare = CompareDeviceResults(h_in, d_out_guarded, guarded_elements, g_verbose, g_verbose);
    printf("\tGuarded: %s\n", (compare) ? "FAIL" : "PASS");
    AssertEquals(0, compare);
}


/**
 * Test native pointer
 */
template <
    typename            T,
    int                 BLOCK_THREADS,
    int                 ITEMS_PER_THREAD,
    BlockLoadAlgorithm  LOAD_ALGORITHM,
    BlockStoreAlgorithm STORE_ALGORITHM,
    PtxLoadModifier     LOAD_MODIFIER,
    PtxStoreModifier    STORE_MODIFIER,
    int                 WARP_TIME_SLICING>
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
    T *d_out_guarded = NULL;
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_in, sizeof(T) * unguarded_elements));
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_out_unguarded, sizeof(T) * unguarded_elements));
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_out_guarded, sizeof(T) * guarded_elements));
    CubDebugExit(cudaMemset(d_out_unguarded, 0, sizeof(T) * unguarded_elements));
    CubDebugExit(cudaMemset(d_out_guarded, 0, sizeof(T) * guarded_elements));

    // Initialize problem on host and device
    for (int i = 0; i < unguarded_elements; ++i)
    {
        InitValue(SEQ_INC, h_in[i], i);
    }
    CubDebugExit(cudaMemcpy(d_in, h_in, sizeof(T) * unguarded_elements, cudaMemcpyHostToDevice));

    printf("TestNative(%d) "
        "grid_size(%d) "
        "guarded_elements(%d) "
        "unguarded_elements(%d) "
        "BLOCK_THREADS(%d) "
        "ITEMS_PER_THREAD(%d) "
        "LOAD_ALGORITHM(%d) "
        "STORE_ALGORITHM(%d) "
        "LOAD_MODIFIER(%d) "
        "STORE_MODIFIER(%d) "
        "WARP_TIME_SLICING(%d) "
        "sizeof(T)(%d)\n",
            IsPointer<T*>::VALUE, grid_size, guarded_elements, unguarded_elements, BLOCK_THREADS, ITEMS_PER_THREAD, LOAD_ALGORITHM, STORE_ALGORITHM, LOAD_MODIFIER, STORE_MODIFIER, WARP_TIME_SLICING, (int) sizeof(T));

    TestKernel<T, BLOCK_THREADS, ITEMS_PER_THREAD, LOAD_ALGORITHM, STORE_ALGORITHM, LOAD_MODIFIER, STORE_MODIFIER, WARP_TIME_SLICING>(
        h_in,
        d_in,
        d_out_unguarded,
        d_out_guarded,
        grid_size,
        guarded_elements);

    // Cleanup
    if (h_in) free(h_in);
    if (d_in) CubDebugExit(g_allocator.DeviceFree(d_in));
    if (d_out_unguarded) CubDebugExit(g_allocator.DeviceFree(d_out_unguarded));
    if (d_out_guarded) CubDebugExit(g_allocator.DeviceFree(d_out_guarded));
}


/**
 * Test iterator (uses thrust counting iterator)
 */
template <
    typename            T,
    int                 BLOCK_THREADS,
    int                 ITEMS_PER_THREAD,
    BlockLoadAlgorithm  LOAD_ALGORITHM,
    BlockStoreAlgorithm STORE_ALGORITHM,
    int                 WARP_TIME_SLICING>
void TestIterator(
    int                 grid_size,
    float               fraction_valid,
    Int2Type<false>     is_integer)
{}


/**
 * Test iterator (uses thrust counting iterator)
 */
template <
    typename            T,
    int                 BLOCK_THREADS,
    int                 ITEMS_PER_THREAD,
    BlockLoadAlgorithm  LOAD_ALGORITHM,
    BlockStoreAlgorithm STORE_ALGORITHM,
    int                 WARP_TIME_SLICING>
void TestIterator(
    int                 grid_size,
    float               fraction_valid,
    Int2Type<true>      is_integer)
{
    typedef thrust::counting_iterator<T> Iterator;

    int unguarded_elements = grid_size * BLOCK_THREADS * ITEMS_PER_THREAD;
    int guarded_elements = int(fraction_valid * float(unguarded_elements));

    // Allocate host arrays
    T *h_in = (T*) malloc(unguarded_elements * sizeof(T));

    // Allocate device arrays
    T *d_out_unguarded = NULL;
    T *d_out_guarded = NULL;
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_out_unguarded, sizeof(T) * unguarded_elements));
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_out_guarded, sizeof(T) * guarded_elements));
    CubDebugExit(cudaMemset(d_out_unguarded, 0, sizeof(T) * unguarded_elements));
    CubDebugExit(cudaMemset(d_out_guarded, 0, sizeof(T) * guarded_elements));

    // Initialize problem on host and device
    Iterator counting_itr(0);
    for (int i = 0; i < unguarded_elements; ++i)
    {
        h_in[i] = counting_itr[i];
    }

    printf("TestIterator(%d) "
        "grid_size(%d) "
        "guarded_elements(%d) "
        "unguarded_elements(%d) "
        "BLOCK_THREADS(%d) "
        "ITEMS_PER_THREAD(%d) "
        "LOAD_ALGORITHM(%d) "
        "STORE_ALGORITHM(%d) "
        "WARP_TIME_SLICING(%d) "
        "sizeof(T)(%d)\n",
            !IsPointer<Iterator>::VALUE, grid_size, guarded_elements, unguarded_elements, BLOCK_THREADS, ITEMS_PER_THREAD, LOAD_ALGORITHM, STORE_ALGORITHM, WARP_TIME_SLICING, (int) sizeof(T));

    TestKernel<T, BLOCK_THREADS, ITEMS_PER_THREAD, LOAD_ALGORITHM, STORE_ALGORITHM, LOAD_DEFAULT, STORE_DEFAULT, WARP_TIME_SLICING>(
        h_in,
        counting_itr,
        d_out_unguarded,
        d_out_guarded,
        grid_size,
        guarded_elements);

    // Cleanup
    if (h_in) free(h_in);
    if (d_out_unguarded) CubDebugExit(g_allocator.DeviceFree(d_out_unguarded));
    if (d_out_guarded) CubDebugExit(g_allocator.DeviceFree(d_out_guarded));
}


/**
 * Evaluate different pointer access types
 */
template <
    typename                T,
    int                     BLOCK_THREADS,
    int                     ITEMS_PER_THREAD,
    BlockLoadAlgorithm      LOAD_ALGORITHM,
    BlockStoreAlgorithm     STORE_ALGORITHM,
    bool                    WARP_TIME_SLICING>
void TestPointerAccess(
    int             grid_size,
    float           fraction_valid)
{
    TestNative<T, BLOCK_THREADS, ITEMS_PER_THREAD, LOAD_ALGORITHM, STORE_ALGORITHM, LOAD_DEFAULT, STORE_DEFAULT, WARP_TIME_SLICING>(grid_size, fraction_valid);
    TestNative<T, BLOCK_THREADS, ITEMS_PER_THREAD, LOAD_ALGORITHM, STORE_ALGORITHM, LOAD_CG, STORE_CG, WARP_TIME_SLICING>(grid_size, fraction_valid);
    TestIterator<T, BLOCK_THREADS, ITEMS_PER_THREAD, LOAD_ALGORITHM, STORE_ALGORITHM, WARP_TIME_SLICING>(grid_size, fraction_valid, Int2Type<((Traits<T>::CATEGORY == SIGNED_INTEGER) || (Traits<T>::CATEGORY == UNSIGNED_INTEGER))>());
}


/**
 * Evaluate different time-slicing strategies
 */
template <
    typename                T,
    int                     BLOCK_THREADS,
    int                     ITEMS_PER_THREAD,
    BlockLoadAlgorithm      LOAD_ALGORITHM,
    BlockStoreAlgorithm     STORE_ALGORITHM>
void TestSlicedStrategy(
    int             grid_size,
    float           fraction_valid)
{
    TestPointerAccess<T, BLOCK_THREADS, ITEMS_PER_THREAD, LOAD_ALGORITHM, STORE_ALGORITHM, true>(grid_size, fraction_valid);
    TestPointerAccess<T, BLOCK_THREADS, ITEMS_PER_THREAD, LOAD_ALGORITHM, STORE_ALGORITHM, false>(grid_size, fraction_valid);
}



/**
 * Evaluate different load/store strategies (specialized for block sizes that are not a multiple of 32)
 */
template <
    typename        T,
    int             BLOCK_THREADS,
    int             ITEMS_PER_THREAD>
void TestStrategy(
    int             grid_size,
    float           fraction_valid,
    Int2Type<false> is_warp_multiple)
{
    TestPointerAccess<T, BLOCK_THREADS, ITEMS_PER_THREAD, BLOCK_LOAD_DIRECT, BLOCK_STORE_DIRECT, false>(grid_size, fraction_valid);
    TestPointerAccess<T, BLOCK_THREADS, ITEMS_PER_THREAD, BLOCK_LOAD_VECTORIZE, BLOCK_STORE_VECTORIZE, false>(grid_size, fraction_valid);
    TestSlicedStrategy<T, BLOCK_THREADS, ITEMS_PER_THREAD, BLOCK_LOAD_TRANSPOSE, BLOCK_STORE_TRANSPOSE>(grid_size, fraction_valid);
}


/**
 * Evaluate different load/store strategies (specialized for block sizes that are a multiple of 32)
 */
template <
    typename        T,
    int             BLOCK_THREADS,
    int             ITEMS_PER_THREAD>
void TestStrategy(
    int             grid_size,
    float           fraction_valid,
    Int2Type<true>  is_warp_multiple)
{
    TestStrategy<T, BLOCK_THREADS, ITEMS_PER_THREAD>(grid_size, fraction_valid, Int2Type<false>());
    TestSlicedStrategy<T, BLOCK_THREADS, ITEMS_PER_THREAD, BLOCK_LOAD_WARP_TRANSPOSE, BLOCK_STORE_WARP_TRANSPOSE>(grid_size, fraction_valid);
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
    TestStrategy<T, BLOCK_THREADS, 1>(grid_size, fraction_valid, Int2Type<BLOCK_THREADS % 32 == 0>());
    TestStrategy<T, BLOCK_THREADS, 3>(grid_size, fraction_valid, Int2Type<BLOCK_THREADS % 32 == 0>());
    TestStrategy<T, BLOCK_THREADS, 4>(grid_size, fraction_valid, Int2Type<BLOCK_THREADS % 32 == 0>());
    TestStrategy<T, BLOCK_THREADS, 17>(grid_size, fraction_valid, Int2Type<BLOCK_THREADS % 32 == 0>());
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
    TestItemsPerThread<T, 72>(grid_size, fraction_valid);
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

    // Simple test
   TestNative<int, 64, 2, BLOCK_LOAD_WARP_TRANSPOSE, BLOCK_STORE_WARP_TRANSPOSE, LOAD_DEFAULT, STORE_DEFAULT, true>(1, 0.8);

    // Evaluate different data types
    TestThreads<char>(2, 0.8);
    TestThreads<int>(2, 0.8);
    TestThreads<double2>(2, 0.8);
    TestThreads<TestFoo>(2, 0.8);
    TestThreads<TestBar>(2, 0.8);

    return 0;
}



