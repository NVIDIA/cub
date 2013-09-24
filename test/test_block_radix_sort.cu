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
 * Test of BlockRadixSort utilities
 ******************************************************************************/

// Ensure printing of CUDA runtime errors to console
#define CUB_STDERR

#include <stdio.h>
#include <algorithm>
#include <iostream>

#include <cub/util_allocator.cuh>
#include <cub/block/block_load.cuh>
#include <cub/block/block_store.cuh>
#include <cub/block/block_radix_sort.cuh>

#include "test_util.h"

using namespace cub;

//---------------------------------------------------------------------
// Globals, constants and typedefs
//---------------------------------------------------------------------

bool                    g_verbose = false;
CachingDeviceAllocator  g_allocator(true);


//---------------------------------------------------------------------
// Test kernels
//---------------------------------------------------------------------


/**
 * BlockRadixSort key-value pair kernel
 */
template <
    int                     BLOCK_THREADS,
    int                     ITEMS_PER_THREAD,
    int                     RADIX_BITS,
    bool                    MEMOIZE_OUTER_SCAN,
    BlockScanAlgorithm      INNER_SCAN_ALGORITHM,
    cudaSharedMemConfig     SMEM_CONFIG,
    typename                Key,
    typename                Value>
__launch_bounds__ (BLOCK_THREADS, 1)
__global__ void Kernel(
    Key                 *d_keys,
    Value               *d_values,
    int                     begin_bit,
    int                     end_bit,
    clock_t                 *d_elapsed)
{
    enum
    {
        TILE_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD,
    };

    // Threadblock load/store abstraction types
    typedef BlockRadixSort<
        Key,
        BLOCK_THREADS,
        ITEMS_PER_THREAD,
        Value,
        RADIX_BITS,
        MEMOIZE_OUTER_SCAN,
        INNER_SCAN_ALGORITHM,
        SMEM_CONFIG> BlockRadixSort;

    // Allocate temp storage in shared memory
    __shared__ typename BlockRadixSort::TempStorage temp_storage;

    // Keys per thread
    Key keys[ITEMS_PER_THREAD];
    Value values[ITEMS_PER_THREAD];

    LoadBlocked<LOAD_DEFAULT>(threadIdx.x, d_keys, keys);
    LoadBlocked<LOAD_DEFAULT>(threadIdx.x, d_values, values);

    // Start cycle timer
    clock_t start = clock();

    // Test keys-value sorting (in striped arrangement)
    BlockRadixSort(temp_storage).SortBlockedToStriped(keys, values, begin_bit, end_bit);

    // Stop cycle timer
    clock_t stop = clock();

    StoreStriped<STORE_DEFAULT, BLOCK_THREADS>(threadIdx.x, d_keys, keys);
    StoreStriped<STORE_DEFAULT, BLOCK_THREADS>(threadIdx.x, d_values, values);

    // Store time
    if (threadIdx.x == 0)
        *d_elapsed = (start > stop) ? start - stop : stop - start;
}


/**
 * BlockRadixSort keys-only kernel
 */
template <
    int                     BLOCK_THREADS,
    int                     ITEMS_PER_THREAD,
    int                     RADIX_BITS,
    bool                    MEMOIZE_OUTER_SCAN,
    BlockScanAlgorithm      INNER_SCAN_ALGORITHM,
    cudaSharedMemConfig     SMEM_CONFIG,
    typename                Key>
__launch_bounds__ (BLOCK_THREADS, 1)
__global__ void Kernel(
    Key                 *d_keys,
    NullType                *d_values,
    int                     begin_bit,
    int                     end_bit,
    clock_t                 *d_elapsed)
{
    enum
    {
        TILE_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD,
    };

    // Threadblock load/store abstraction types
    typedef BlockRadixSort<
        Key,
        BLOCK_THREADS,
        ITEMS_PER_THREAD,
        NullType,
        RADIX_BITS,
        MEMOIZE_OUTER_SCAN,
        INNER_SCAN_ALGORITHM,
        SMEM_CONFIG> BlockRadixSort;

    // Allocate temp storage in shared memory
    __shared__ typename BlockRadixSort::TempStorage temp_storage;

    // Keys per thread
    Key keys[ITEMS_PER_THREAD];

    LoadBlocked<LOAD_DEFAULT>(threadIdx.x, d_keys, keys);

    // Start cycle timer
    clock_t start = clock();

    // Test keys-only sorting (in striped arrangement)
    BlockRadixSort(temp_storage).SortBlockedToStriped(keys, begin_bit, end_bit);

    // Stop cycle timer
    clock_t stop = clock();

    StoreStriped<STORE_DEFAULT, BLOCK_THREADS>(threadIdx.x, d_keys, keys);

    // Store time
    if (threadIdx.x == 0)
        *d_elapsed = (start > stop) ? start - stop : stop - start;
}


//---------------------------------------------------------------------
// Host testing subroutines
//---------------------------------------------------------------------

/**
 * Drive BlockRadixSort kernel
 */
template <
    int                     BLOCK_THREADS,
    int                     ITEMS_PER_THREAD,
    int                     RADIX_BITS,
    bool                    MEMOIZE_OUTER_SCAN,
    BlockScanAlgorithm      INNER_SCAN_ALGORITHM,
    cudaSharedMemConfig     SMEM_CONFIG,
    typename                Key,
    typename                Value>
void TestDriver(
    int                     entropy_reduction,
    int                     begin_bit,
    int                     end_bit)
{
    enum
    {
        TILE_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD,
        KEYS_ONLY = Equals<Value, NullType>::VALUE,
    };

    // Allocate host arrays
    Key     *h_keys             = (Key*) malloc(TILE_SIZE * sizeof(Key));
    Key     *h_reference_keys   = (Key*) malloc(TILE_SIZE * sizeof(Key));
    Value   *h_values           = (Value*) malloc(TILE_SIZE * sizeof(Value));

    // Allocate device arrays
    Key     *d_keys     = NULL;
    Value   *d_values   = NULL;
    clock_t     *d_elapsed = NULL;
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_keys, sizeof(Key) * TILE_SIZE));
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_values, sizeof(Value) * TILE_SIZE));
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_elapsed, sizeof(clock_t)));

    // Initialize problem on host and device
    for (int i = 0; i < TILE_SIZE; ++i)
    {
        RandomBits(h_keys[i], entropy_reduction, begin_bit, end_bit);
        h_reference_keys[i] = h_keys[i];
        h_values[i] = i;
    }
    CubDebugExit(cudaMemcpy(d_keys, h_keys, sizeof(Key) * TILE_SIZE, cudaMemcpyHostToDevice));
    CubDebugExit(cudaMemcpy(d_values, h_values, sizeof(Value) * TILE_SIZE, cudaMemcpyHostToDevice));

    printf("%s "
        "BLOCK_THREADS(%d) "
        "ITEMS_PER_THREAD(%d) "
        "RADIX_BITS(%d) "
        "INNER_SCAN_ALGORITHM(%d) "
        "SMEM_CONFIG(%d) "
        "sizeof(Key)(%d) "
        "sizeof(Value)(%d) "
        "entropy_reduction(%d) "
        "begin_bit(%d) "
        "end_bit(%d)\n",
            ((KEYS_ONLY) ? "Keys-only" : "Key-value"),
            BLOCK_THREADS,
            ITEMS_PER_THREAD,
            RADIX_BITS,
            MEMOIZE_OUTER_SCAN,
            INNER_SCAN_ALGORITHM,
            SMEM_CONFIG,
            (int) sizeof(Key),
            (int) sizeof(Value),
            entropy_reduction,
            begin_bit,
            end_bit);

    // Compute reference solution
    printf("\tComputing reference solution on CPU..."); fflush(stdout);
    std::sort(h_reference_keys, h_reference_keys + TILE_SIZE);
    printf(" Done.\n"); fflush(stdout);

    cudaDeviceSetSharedMemConfig(SMEM_CONFIG);

    // Run kernel
    Kernel<BLOCK_THREADS, ITEMS_PER_THREAD, RADIX_BITS, MEMOIZE_OUTER_SCAN, INNER_SCAN_ALGORITHM, SMEM_CONFIG><<<1, BLOCK_THREADS>>>(
        d_keys, d_values, begin_bit, end_bit, d_elapsed);

    // Flush kernel output / errors
    CubDebugExit(cudaDeviceSynchronize());

    // Check keys results
    printf("\tKeys: ");
    int compare = CompareDeviceResults(h_reference_keys, d_keys, TILE_SIZE, g_verbose, g_verbose);
    printf("%s\n", compare ? "FAIL" : "PASS");
    AssertEquals(0, compare);

    // Check value results (which aren't valid for 8-bit values and tile size >= 256 because they can't fully index into the starting array)
    if (!KEYS_ONLY && ((sizeof(Value) > 1) || (TILE_SIZE < 256)))
    {
        CubDebugExit(cudaMemcpy(h_values, d_values, sizeof(Value) * TILE_SIZE, cudaMemcpyDeviceToHost));

        printf("\tValues: ");
        if (g_verbose)
        {
            DisplayResults(h_values, TILE_SIZE);
            printf("\n");
        }

        for (int i = 0; i < TILE_SIZE; ++i)
        {
            int value = reinterpret_cast<int&>(h_values[i]);
            if (h_keys[value] != h_reference_keys[i])
            {
                std::cout << "Incorrect: [" << i << "]: " << h_keys[value] << " != " << h_reference_keys[i] << "\n\n";
                compare = 1;
                break;
            }
        }
        printf("%s\n", compare ? "FAIL" : "PASS");
        AssertEquals(0, compare);
    }
    printf("\n");

    printf("\tElapsed clocks: ");
    DisplayDeviceResults(d_elapsed, 1);
    printf("\n");

    // Cleanup
    if (h_keys)             free(h_keys);
    if (h_reference_keys)   free(h_reference_keys);
    if (h_values)           free(h_values);
    if (d_keys)             CubDebugExit(g_allocator.DeviceFree(d_keys));
    if (d_values)           CubDebugExit(g_allocator.DeviceFree(d_values));
    if (d_elapsed)          CubDebugExit(g_allocator.DeviceFree(d_elapsed));
}


/**
 * Test driver (valid tile size < 48KB)
 */
template <
    int                     BLOCK_THREADS,
    int                     ITEMS_PER_THREAD,
    int                     RADIX_BITS,
    bool                    MEMOIZE_OUTER_SCAN,
    BlockScanAlgorithm      INNER_SCAN_ALGORITHM,
    cudaSharedMemConfig     SMEM_CONFIG,
    typename                Key,
    typename                Value,
    typename                BlockRadixSortT = BlockRadixSort<Key, BLOCK_THREADS, ITEMS_PER_THREAD, Value, RADIX_BITS, MEMOIZE_OUTER_SCAN, INNER_SCAN_ALGORITHM, SMEM_CONFIG>,
    bool                    VALID = (sizeof(typename BlockRadixSortT::TempStorage) <= (1024 * 48))>
struct Valid
{
    static void Test()
    {
        // Iterate entropy_reduction
        for (int entropy_reduction = 0; entropy_reduction <= 9; entropy_reduction += 3)
        {
            // Iterate begin_bit
            for (int begin_bit = 0; begin_bit <= 1; begin_bit++)
            {
                // Iterate end bit
                for (int end_bit = begin_bit + 1; end_bit <= sizeof(Key) * 8; end_bit = end_bit * 2 + begin_bit)
                {
                    TestDriver<BLOCK_THREADS, ITEMS_PER_THREAD, RADIX_BITS, MEMOIZE_OUTER_SCAN, INNER_SCAN_ALGORITHM, SMEM_CONFIG, Key, Value>(
                        entropy_reduction,
                        begin_bit,
                        end_bit);
                }
            }
        }
    }
};


/**
 * Test driver (invalid tile size)
 */
template <
    int                     BLOCK_THREADS,
    int                     ITEMS_PER_THREAD,
    int                     RADIX_BITS,
    bool                    MEMOIZE_OUTER_SCAN,
    BlockScanAlgorithm      INNER_SCAN_ALGORITHM,
    cudaSharedMemConfig     SMEM_CONFIG,
    typename                Key,
    typename                Value,
    typename                BlockRadixSortT>
struct Valid<BLOCK_THREADS, ITEMS_PER_THREAD, RADIX_BITS, MEMOIZE_OUTER_SCAN, INNER_SCAN_ALGORITHM, SMEM_CONFIG, Key, Value, BlockRadixSortT, false>
{
    // Do nothing
    static void Test() {}
};


/**
 * Test value type
 */
template <
    int                     BLOCK_THREADS,
    int                     ITEMS_PER_THREAD,
    int                     RADIX_BITS,
    bool                    MEMOIZE_OUTER_SCAN,
    BlockScanAlgorithm      INNER_SCAN_ALGORITHM,
    cudaSharedMemConfig     SMEM_CONFIG,
    typename                Key>
void Test()
{
    Valid<BLOCK_THREADS, ITEMS_PER_THREAD, RADIX_BITS, MEMOIZE_OUTER_SCAN, INNER_SCAN_ALGORITHM, SMEM_CONFIG, Key, NullType>::Test();         // Keys-only

    Valid<BLOCK_THREADS, ITEMS_PER_THREAD, RADIX_BITS, MEMOIZE_OUTER_SCAN, INNER_SCAN_ALGORITHM, SMEM_CONFIG, Key, unsigned char>::Test();
    Valid<BLOCK_THREADS, ITEMS_PER_THREAD, RADIX_BITS, MEMOIZE_OUTER_SCAN, INNER_SCAN_ALGORITHM, SMEM_CONFIG, Key, unsigned short>::Test();
    Valid<BLOCK_THREADS, ITEMS_PER_THREAD, RADIX_BITS, MEMOIZE_OUTER_SCAN, INNER_SCAN_ALGORITHM, SMEM_CONFIG, Key, unsigned int>::Test();
    Valid<BLOCK_THREADS, ITEMS_PER_THREAD, RADIX_BITS, MEMOIZE_OUTER_SCAN, INNER_SCAN_ALGORITHM, SMEM_CONFIG, Key, unsigned long long>::Test();
}


/**
 * Test key type
 */
template <
    int                     BLOCK_THREADS,
    int                     ITEMS_PER_THREAD,
    int                     RADIX_BITS,
    bool                    MEMOIZE_OUTER_SCAN,
    BlockScanAlgorithm      INNER_SCAN_ALGORITHM,
    cudaSharedMemConfig     SMEM_CONFIG>
void Test()
{
    Test<BLOCK_THREADS, ITEMS_PER_THREAD, RADIX_BITS, MEMOIZE_OUTER_SCAN, INNER_SCAN_ALGORITHM, SMEM_CONFIG, unsigned char>();
    Test<BLOCK_THREADS, ITEMS_PER_THREAD, RADIX_BITS, MEMOIZE_OUTER_SCAN, INNER_SCAN_ALGORITHM, SMEM_CONFIG, unsigned short>();
    Test<BLOCK_THREADS, ITEMS_PER_THREAD, RADIX_BITS, MEMOIZE_OUTER_SCAN, INNER_SCAN_ALGORITHM, SMEM_CONFIG, unsigned int>();
    Test<BLOCK_THREADS, ITEMS_PER_THREAD, RADIX_BITS, MEMOIZE_OUTER_SCAN, INNER_SCAN_ALGORITHM, SMEM_CONFIG, unsigned long long>();

    Test<BLOCK_THREADS, ITEMS_PER_THREAD, RADIX_BITS, MEMOIZE_OUTER_SCAN, INNER_SCAN_ALGORITHM, SMEM_CONFIG, char>();
    Test<BLOCK_THREADS, ITEMS_PER_THREAD, RADIX_BITS, MEMOIZE_OUTER_SCAN, INNER_SCAN_ALGORITHM, SMEM_CONFIG, short>();
    Test<BLOCK_THREADS, ITEMS_PER_THREAD, RADIX_BITS, MEMOIZE_OUTER_SCAN, INNER_SCAN_ALGORITHM, SMEM_CONFIG, int>();
    Test<BLOCK_THREADS, ITEMS_PER_THREAD, RADIX_BITS, MEMOIZE_OUTER_SCAN, INNER_SCAN_ALGORITHM, SMEM_CONFIG, float>();
    Test<BLOCK_THREADS, ITEMS_PER_THREAD, RADIX_BITS, MEMOIZE_OUTER_SCAN, INNER_SCAN_ALGORITHM, SMEM_CONFIG, double>();
}


/**
 * Test smem config
 */
template <
    int                     BLOCK_THREADS,
    int                     ITEMS_PER_THREAD,
    int                     RADIX_BITS,
    bool                    MEMOIZE_OUTER_SCAN,
    BlockScanAlgorithm      INNER_SCAN_ALGORITHM>
void Test()
{
    Test<BLOCK_THREADS, ITEMS_PER_THREAD, RADIX_BITS, MEMOIZE_OUTER_SCAN, INNER_SCAN_ALGORITHM, cudaSharedMemBankSizeFourByte>();
    Test<BLOCK_THREADS, ITEMS_PER_THREAD, RADIX_BITS, MEMOIZE_OUTER_SCAN, INNER_SCAN_ALGORITHM, cudaSharedMemBankSizeEightByte>();
}


/**
 * Test inner scan algorithm
 */
template <
    int                     BLOCK_THREADS,
    int                     ITEMS_PER_THREAD,
    int                     RADIX_BITS,
    bool                    MEMOIZE_OUTER_SCAN>
void Test()
{
    Test<BLOCK_THREADS, ITEMS_PER_THREAD, RADIX_BITS, MEMOIZE_OUTER_SCAN, BLOCK_SCAN_RAKING>();
    Test<BLOCK_THREADS, ITEMS_PER_THREAD, RADIX_BITS, MEMOIZE_OUTER_SCAN, BLOCK_SCAN_WARP_SCANS>();
}


/**
 * Test outer scan algorithm
 */
template <
    int                     BLOCK_THREADS,
    int                     ITEMS_PER_THREAD,
    int                     RADIX_BITS>
void Test()
{
    Test<BLOCK_THREADS, ITEMS_PER_THREAD, RADIX_BITS, true>();
    Test<BLOCK_THREADS, ITEMS_PER_THREAD, RADIX_BITS, false>();
}


/**
 * Test radix bits
 */
template <
    int BLOCK_THREADS,
    int ITEMS_PER_THREAD>
void Test()
{
    Test<BLOCK_THREADS, ITEMS_PER_THREAD, 1>();
    Test<BLOCK_THREADS, ITEMS_PER_THREAD, 2>();
    Test<BLOCK_THREADS, ITEMS_PER_THREAD, 5>();
}


/**
 * Test items per thread
 */
template <int BLOCK_THREADS>
void Test()
{
    Test<BLOCK_THREADS, 1>();
    Test<BLOCK_THREADS, 8>();
    Test<BLOCK_THREADS, 11>();
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

    // Quick test
    typedef unsigned int T;
    TestDriver<64, 2, 5, true, BLOCK_SCAN_WARP_SCANS, cudaSharedMemBankSizeFourByte, T, NullType>(0, 0, sizeof(T) * 8);
/*
    // Test threads
    Test<32>();
    Test<64>();
    Test<128>();
    Test<256>();
*/
    return 0;
}



