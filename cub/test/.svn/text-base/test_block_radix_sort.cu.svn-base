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

#include <test_util.h>
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
 * BlockRadixSort kernel
 */
template <
    int                     BLOCK_THREADS,
    int                     ITEMS_PER_THREAD,
    int                     RADIX_BITS,
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
        KEYS_ONLY = Equals<ValueType, NullType>::VALUE,
    };

    // Threadblock load/store abstraction types
    typedef BlockRadixSort<
        KeyType,
        BLOCK_THREADS,
        ITEMS_PER_THREAD,
        ValueType,
        RADIX_BITS,
        SMEM_CONFIG> BlockRadixSort;

    // Shared memory
    __shared__ typename BlockRadixSort::SmemStorage smem_storage;

    // Keys per thread
    KeyType keys[ITEMS_PER_THREAD];

    if (KEYS_ONLY)
    {
        // Test keys-only sorting (in striped arrangement)
        BlockLoadDirectStriped(d_keys, keys, BLOCK_THREADS);

        BlockRadixSort::SortStriped(smem_storage, keys, begin_bit, end_bit);

        BlockStoreDirectStriped(d_keys, keys, BLOCK_THREADS);
    }
    else
    {
        // Test keys-value sorting (in striped arrangement)
        ValueType values[ITEMS_PER_THREAD];

        BlockLoadDirectStriped(d_keys, keys, BLOCK_THREADS);
        BlockLoadDirectStriped(d_values, values, BLOCK_THREADS);

        BlockRadixSort::SortStriped(smem_storage, keys, values, begin_bit, end_bit);

        BlockStoreDirectStriped(d_keys, keys, BLOCK_THREADS);
        BlockStoreDirectStriped(d_values, values, BLOCK_THREADS);
    }
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
    cudaSharedMemConfig     SMEM_CONFIG,
    typename                KeyType,
    typename                ValueType>
void TestDriver(
    bool                    keys_only,
    unsigned int            entropy_reduction,
    unsigned int            begin_bit,
    unsigned int            end_bit)
{
    enum
    {
        TILE_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD,
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
        "SMEM_CONFIG(%d) "
        "sizeof(KeyType)(%d) "
        "sizeof(ValueType)(%d) "
        "entropy_reduction(%d) "
        "begin_bit(%d) "
        "end_bit(%d)\n",
            ((keys_only) ? "keys-only" : "key-value"),
            BLOCK_THREADS,
            ITEMS_PER_THREAD,
            RADIX_BITS,
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
    if (keys_only)
    {
        // Keys-only
        Kernel<BLOCK_THREADS, ITEMS_PER_THREAD, RADIX_BITS, SMEM_CONFIG><<<1, BLOCK_THREADS>>>(
            d_keys, (NullType*) d_values, begin_bit, end_bit);
    }
    else
    {
        // Key-value pairs
        Kernel<BLOCK_THREADS, ITEMS_PER_THREAD, RADIX_BITS, SMEM_CONFIG><<<1, BLOCK_THREADS>>>(
            d_keys, d_values, begin_bit, end_bit);
    }

    // Flush kernel output / errors
    CubDebugExit(cudaDeviceSynchronize());

    // Check keys results
    printf("\tKeys: ");
    AssertEquals(0, CompareDeviceResults(h_reference_keys, d_keys, TILE_SIZE, g_verbose, g_verbose));
    printf("\n");

    // Check value results (which aren't valid for 8-bit values and tile size >= 256 because they can't fully index into the starting array)
    if (!keys_only && ((sizeof(ValueType) > 1) || (TILE_SIZE < 256)))
    {
        CubDebugExit(cudaMemcpy(h_values, d_values, sizeof(ValueType) * TILE_SIZE, cudaMemcpyDeviceToHost));

        printf("\tValues: ");
        if (g_verbose)
        {
            DisplayResults(h_values, TILE_SIZE);
            printf("\n");
        }

        bool correct = true;
        for (int i = 0; i < TILE_SIZE; ++i)
        {
            if (h_keys[h_values[i]] != h_reference_keys[i])
            {
                std::cout << "Incorrect: [" << i << "]: " << h_keys[h_values[i]] << " != " << h_reference_keys[i] << std::endl << std::endl;
                correct = false;
                break;
            }
        }
        if (correct) printf("Correct\n");
        AssertEquals(true, correct);
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
    cudaSharedMemConfig     SMEM_CONFIG,
    typename                KeyType,
    typename                ValueType,
    bool                    VALID = (CUB_MAX(sizeof(KeyType), sizeof(ValueType)) * BLOCK_THREADS * ITEMS_PER_THREAD < (1024 * 48))>
struct Valid
{
    static void Test()
    {
        // Iterate keys vs. key-value pairs
        for (unsigned int keys_only = 0; keys_only <= 1; keys_only++)
        {
            // Iterate entropy_reduction
            for (unsigned int entropy_reduction = 0; entropy_reduction <= 8; entropy_reduction += 2)
            {
                // Iterate begin_bit
                for (unsigned int begin_bit = 0; begin_bit <= 1; begin_bit++)
                {
                    // Iterate passes
                    for (unsigned int passes = 1; passes <= (sizeof(KeyType) * 8) / RADIX_BITS; passes++)
                    {
                        // Iterate relative_end
                        for (int relative_end = -1; relative_end <= 1; relative_end++)
                        {
                            int end_bit = begin_bit + (passes * RADIX_BITS) + relative_end;
                            if ((end_bit > begin_bit) && (end_bit <= sizeof(KeyType) * 8))
                            {
                                TestDriver<BLOCK_THREADS, ITEMS_PER_THREAD, RADIX_BITS, SMEM_CONFIG, KeyType, ValueType>(
                                    (bool) keys_only,
                                    entropy_reduction,
                                    begin_bit,
                                    end_bit);
                            }
                        }
                    }
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
    cudaSharedMemConfig     SMEM_CONFIG,
    typename                KeyType,
    typename                ValueType>
struct Valid<BLOCK_THREADS, ITEMS_PER_THREAD, RADIX_BITS, SMEM_CONFIG, KeyType, ValueType, false>
{
    // Do nothing
    static void Test() {}
};


/**
 * Test value type
 */
template <
    int BLOCK_THREADS,
    int ITEMS_PER_THREAD,
    int RADIX_BITS,
    cudaSharedMemConfig SMEM_CONFIG,
    typename KeyType>
void Test()
{
    Valid<BLOCK_THREADS, ITEMS_PER_THREAD, RADIX_BITS, SMEM_CONFIG, KeyType, unsigned char>::Test();
    Valid<BLOCK_THREADS, ITEMS_PER_THREAD, RADIX_BITS, SMEM_CONFIG, KeyType, unsigned short>::Test();
    Valid<BLOCK_THREADS, ITEMS_PER_THREAD, RADIX_BITS, SMEM_CONFIG, KeyType, unsigned int>::Test();
    Valid<BLOCK_THREADS, ITEMS_PER_THREAD, RADIX_BITS, SMEM_CONFIG, KeyType, unsigned long long>::Test();
}


/**
 * Test key type
 */
template <
    int BLOCK_THREADS,
    int ITEMS_PER_THREAD,
    int RADIX_BITS,
    cudaSharedMemConfig SMEM_CONFIG>
void Test()
{
    Test<BLOCK_THREADS, ITEMS_PER_THREAD, RADIX_BITS, SMEM_CONFIG, unsigned char>();
    Test<BLOCK_THREADS, ITEMS_PER_THREAD, RADIX_BITS, SMEM_CONFIG, unsigned short>();
    Test<BLOCK_THREADS, ITEMS_PER_THREAD, RADIX_BITS, SMEM_CONFIG, unsigned int>();
    Test<BLOCK_THREADS, ITEMS_PER_THREAD, RADIX_BITS, SMEM_CONFIG, unsigned long long>();
}


/**
 * Test smem config
 */
template <
    int BLOCK_THREADS,
    int ITEMS_PER_THREAD,
    int RADIX_BITS>
void Test()
{
    Test<BLOCK_THREADS, ITEMS_PER_THREAD, RADIX_BITS, cudaSharedMemBankSizeFourByte>();
    Test<BLOCK_THREADS, ITEMS_PER_THREAD, RADIX_BITS, cudaSharedMemBankSizeEightByte>();
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
    Test<BLOCK_THREADS, 19>();
}

/**
 * Test threads
 */
void Test()
{
    Test<32>();
    Test<64>();
    Test<128>();
    Test<256>();
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
        printf("%s [--device=<device-id>] [--v] [--quick]\n", argv[0]);
        exit(0);
    }

    // Initialize device
    CubDebugExit(args.DeviceInit());

    if (quick)
    {
        // Quick test
        TestDriver<64, 2, 5, cudaSharedMemBankSizeFourByte, unsigned int, unsigned int>(
            true,
            0,
            0,
            sizeof(unsigned int) * 8);
    }
    else
    {
        Test();
    }

    return 0;
}



