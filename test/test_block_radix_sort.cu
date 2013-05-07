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

#include "test_util.h"
#include <cub/cub.cuh>

using namespace cub;

//---------------------------------------------------------------------
// Globals, constants and typedefs
//---------------------------------------------------------------------

bool g_verbose = false;


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
    typename                KeyType,
    typename                ValueType>
__launch_bounds__ (BLOCK_THREADS, 1)
__global__ void Kernel(
    KeyType             *d_keys,
    ValueType           *d_values,
    unsigned int        begin_bit,
    unsigned int        end_bit)
{
    enum
    {
        TILE_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD,
    };

    // Threadblock load/store abstraction types
    typedef BlockRadixSort<
        KeyType,
        BLOCK_THREADS,
        ITEMS_PER_THREAD,
        ValueType,
        RADIX_BITS,
        MEMOIZE_OUTER_SCAN,
        INNER_SCAN_ALGORITHM,
        SMEM_CONFIG> BlockRadixSort;

    // Shared memory
    __shared__ typename BlockRadixSort::SmemStorage smem_storage;

    // Keys per thread
    KeyType keys[ITEMS_PER_THREAD];
    ValueType values[ITEMS_PER_THREAD];

    BlockLoadDirectStriped(d_keys, keys, BLOCK_THREADS);
    BlockLoadDirectStriped(d_values, values, BLOCK_THREADS);

    // Test keys-value sorting (in striped arrangement)
    BlockRadixSort::SortStriped(smem_storage, keys, values, begin_bit, end_bit);

    BlockStoreDirectStriped(d_keys, keys, BLOCK_THREADS);
    BlockStoreDirectStriped(d_values, values, BLOCK_THREADS);
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
    typename                KeyType>
__launch_bounds__ (BLOCK_THREADS, 1)
__global__ void Kernel(
    KeyType             *d_keys,
    NullType            *d_values,
    unsigned int        begin_bit,
    unsigned int        end_bit)
{
    enum
    {
        TILE_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD,
    };

    // Threadblock load/store abstraction types
    typedef BlockRadixSort<
        KeyType,
        BLOCK_THREADS,
        ITEMS_PER_THREAD,
        NullType,
        RADIX_BITS,
        MEMOIZE_OUTER_SCAN,
        INNER_SCAN_ALGORITHM,
        SMEM_CONFIG> BlockRadixSort;

    // Shared memory
    __shared__ typename BlockRadixSort::SmemStorage smem_storage;

    // Keys per thread
    KeyType keys[ITEMS_PER_THREAD];

    BlockLoadDirectStriped(d_keys, keys, BLOCK_THREADS);

    // Test keys-only sorting (in striped arrangement)
    BlockRadixSort::SortStriped(smem_storage, keys, begin_bit, end_bit);

    BlockStoreDirectStriped(d_keys, keys, BLOCK_THREADS);
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
    typename                KeyType,
    typename                ValueType>
void TestDriver(
    unsigned int            entropy_reduction,
    unsigned int            begin_bit,
    unsigned int            end_bit)
{
    enum
    {
        TILE_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD,
        KEYS_ONLY = Equals<ValueType, NullType>::VALUE,
    };

    // Allocate host arrays
    KeyType     *h_keys             = (KeyType*) malloc(TILE_SIZE * sizeof(KeyType));
    KeyType     *h_reference_keys   = (KeyType*) malloc(TILE_SIZE * sizeof(KeyType));
    ValueType   *h_values           = (ValueType*) malloc(TILE_SIZE * sizeof(ValueType));

    // Allocate device arrays
    KeyType     *d_keys     = NULL;
    ValueType   *d_values   = NULL;
    CubDebugExit(DeviceAllocate((void**)&d_keys, sizeof(KeyType) * TILE_SIZE));
    CubDebugExit(DeviceAllocate((void**)&d_values, sizeof(ValueType) * TILE_SIZE));

    // Initialize problem on host and device
    for (int i = 0; i < TILE_SIZE; ++i)
    {
        RandomBits(h_keys[i], entropy_reduction, begin_bit, end_bit);
        h_reference_keys[i] = h_keys[i];
        h_values[i] = i;
    }
    CubDebugExit(cudaMemcpy(d_keys, h_keys, sizeof(KeyType) * TILE_SIZE, cudaMemcpyHostToDevice));
    CubDebugExit(cudaMemcpy(d_values, h_values, sizeof(ValueType) * TILE_SIZE, cudaMemcpyHostToDevice));

    printf("%s "
        "BLOCK_THREADS(%d) "
        "ITEMS_PER_THREAD(%d) "
        "RADIX_BITS(%d) "
        "INNER_SCAN_ALGORITHM(%d) "
        "SMEM_CONFIG(%d) "
        "sizeof(KeyType)(%d) "
        "sizeof(ValueType)(%d) "
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
            (int) sizeof(KeyType),
            (int) sizeof(ValueType),
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
        d_keys, d_values, begin_bit, end_bit);

    // Flush kernel output / errors
    CubDebugExit(cudaDeviceSynchronize());

    // Check keys results
    printf("\tKeys: ");
    int compare = CompareDeviceResults(h_reference_keys, d_keys, TILE_SIZE, g_verbose, g_verbose);
    printf("%s\n", compare ? "FAIL" : "PASS");
    AssertEquals(0, compare);

    // Check value results (which aren't valid for 8-bit values and tile size >= 256 because they can't fully index into the starting array)
    if (!KEYS_ONLY && ((sizeof(ValueType) > 1) || (TILE_SIZE < 256)))
    {
        CubDebugExit(cudaMemcpy(h_values, d_values, sizeof(ValueType) * TILE_SIZE, cudaMemcpyDeviceToHost));

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

    // Cleanup
    if (h_keys)             free(h_keys);
    if (h_reference_keys)   free(h_reference_keys);
    if (h_values)           free(h_values);
    if (d_keys)             CubDebugExit(DeviceFree(d_keys));
    if (d_values)           CubDebugExit(DeviceFree(d_values));
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
    typename                KeyType,
    typename                ValueType,
    typename                BlockRadixSortT = BlockRadixSort<KeyType, BLOCK_THREADS, ITEMS_PER_THREAD, ValueType, RADIX_BITS, MEMOIZE_OUTER_SCAN, INNER_SCAN_ALGORITHM, SMEM_CONFIG>,
    bool                    VALID = (sizeof(typename BlockRadixSortT::SmemStorage) <= (1024 * 48))>
struct Valid
{
    static void Test()
    {
        // Iterate entropy_reduction
        for (unsigned int entropy_reduction = 0; entropy_reduction <= 9; entropy_reduction += 3)
        {
            // Iterate begin_bit
            for (unsigned int begin_bit = 0; begin_bit <= 1; begin_bit++)
            {
                // Iterate end bit
                for (unsigned int end_bit = begin_bit + 1; end_bit <= sizeof(KeyType) * 8; end_bit = end_bit * 2 + begin_bit)
                {
                    TestDriver<BLOCK_THREADS, ITEMS_PER_THREAD, RADIX_BITS, MEMOIZE_OUTER_SCAN, INNER_SCAN_ALGORITHM, SMEM_CONFIG, KeyType, ValueType>(
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
    typename                KeyType,
    typename                ValueType,
    typename                BlockRadixSortT>
struct Valid<BLOCK_THREADS, ITEMS_PER_THREAD, RADIX_BITS, MEMOIZE_OUTER_SCAN, INNER_SCAN_ALGORITHM, SMEM_CONFIG, KeyType, ValueType, BlockRadixSortT, false>
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
    typename                KeyType>
void Test()
{
    Valid<BLOCK_THREADS, ITEMS_PER_THREAD, RADIX_BITS, MEMOIZE_OUTER_SCAN, INNER_SCAN_ALGORITHM, SMEM_CONFIG, KeyType, NullType>::Test();         // Keys-only

    Valid<BLOCK_THREADS, ITEMS_PER_THREAD, RADIX_BITS, MEMOIZE_OUTER_SCAN, INNER_SCAN_ALGORITHM, SMEM_CONFIG, KeyType, unsigned char>::Test();
    Valid<BLOCK_THREADS, ITEMS_PER_THREAD, RADIX_BITS, MEMOIZE_OUTER_SCAN, INNER_SCAN_ALGORITHM, SMEM_CONFIG, KeyType, unsigned short>::Test();
    Valid<BLOCK_THREADS, ITEMS_PER_THREAD, RADIX_BITS, MEMOIZE_OUTER_SCAN, INNER_SCAN_ALGORITHM, SMEM_CONFIG, KeyType, unsigned int>::Test();
    Valid<BLOCK_THREADS, ITEMS_PER_THREAD, RADIX_BITS, MEMOIZE_OUTER_SCAN, INNER_SCAN_ALGORITHM, SMEM_CONFIG, KeyType, unsigned long long>::Test();
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
    Test<BLOCK_THREADS, ITEMS_PER_THREAD, 3>();
    Test<BLOCK_THREADS, ITEMS_PER_THREAD, 4>();
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
    Test<BLOCK_THREADS, 15>();
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

    // Test threads
    Test<32>();
    Test<64>();
    Test<128>();
    Test<256>();

    return 0;
}



