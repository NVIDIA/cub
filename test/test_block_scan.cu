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
 * Test of BlockScan utilities
 ******************************************************************************/

// Ensure printing of CUDA runtime errors to console
#define CUB_STDERR

#include <stdio.h>
#include <iostream>

#include <cub/util_allocator.cuh>
#include <cub/block/block_load.cuh>
#include <cub/block/block_store.cuh>
#include <cub/block/block_scan.cuh>

#include "test_util.h"


using namespace cub;

//---------------------------------------------------------------------
// Globals, constants and typedefs
//---------------------------------------------------------------------

bool                    g_verbose       = false;
int                     g_repeat        = 0;
CachingDeviceAllocator  g_allocator(true);


/**
 * Primitive variant to test
 */
enum TestMode
{
    BASIC,           //!< BASIC
    AGGREGATE,       //!< AGGREGATE
    PREFIX_AGGREGATE,//!< PREFIX_AGGREGATE
};



/**
 * Stateful prefix functor
 */
template <
    typename T,
    typename ScanOp>
struct BlockPrefixCallbackOp
{
    T       prefix;
    ScanOp  scan_op;

    __device__ __forceinline__
    BlockPrefixCallbackOp(T prefix, ScanOp scan_op) : prefix(prefix), scan_op(scan_op) {}

    __device__ __forceinline__
    T operator()(T block_aggregate)
    {
        // For testing purposes
        T retval = (threadIdx.x == 0) ? prefix  : T();
        prefix = scan_op(prefix, block_aggregate);
        return retval;
    }
};


//---------------------------------------------------------------------
// Exclusive scan
//---------------------------------------------------------------------

/// Exclusive scan (BASIC, 1)
template <typename BlockScan, typename T, typename ScanOp, typename IdentityT, typename PrefixCallbackOp>
__device__ __forceinline__ void DeviceTest(
    typename BlockScan::TempStorage &temp_storage, T (&data)[1], IdentityT &identity, ScanOp &scan_op, T &aggregate, PrefixCallbackOp &prefix_op, Int2Type<BASIC> test_mode)
{
    BlockScan(temp_storage).ExclusiveScan(data[0], data[0], identity, scan_op);
}

/// Exclusive scan (BASIC, ITEMS_PER_THREAD)
template <typename BlockScan, typename T, typename ScanOp, typename IdentityT, typename PrefixCallbackOp, int ITEMS_PER_THREAD>
__device__ __forceinline__ void DeviceTest(
    typename BlockScan::TempStorage &temp_storage, T (&data)[ITEMS_PER_THREAD], IdentityT &identity, ScanOp &scan_op, T &aggregate, PrefixCallbackOp &prefix_op, Int2Type<BASIC> test_mode)
{
    BlockScan(temp_storage).ExclusiveScan(data, data, identity, scan_op);
}

/// Exclusive scan (AGGREGATE, 1)
template <typename BlockScan, typename T, typename ScanOp, typename IdentityT, typename PrefixCallbackOp>
__device__ __forceinline__ void DeviceTest(
    typename BlockScan::TempStorage &temp_storage, T (&data)[1], IdentityT &identity, ScanOp &scan_op, T &aggregate, PrefixCallbackOp &prefix_op, Int2Type<AGGREGATE> test_mode)
{
    BlockScan(temp_storage).ExclusiveScan(data[0], data[0], identity, scan_op, aggregate);
}

/// Exclusive scan (AGGREGATE, ITEMS_PER_THREAD)
template <typename BlockScan, typename T, typename ScanOp, typename IdentityT, typename PrefixCallbackOp, int ITEMS_PER_THREAD>
__device__ __forceinline__ void DeviceTest(
    typename BlockScan::TempStorage &temp_storage, T (&data)[ITEMS_PER_THREAD], IdentityT &identity, ScanOp &scan_op, T &aggregate, PrefixCallbackOp &prefix_op, Int2Type<AGGREGATE> test_mode)
{
    BlockScan(temp_storage).ExclusiveScan(data, data, identity, scan_op, aggregate);
}

/// Exclusive scan (PREFIX_AGGREGATE, 1)
template <typename BlockScan, typename T, typename ScanOp, typename IdentityT, typename PrefixCallbackOp>
__device__ __forceinline__ void DeviceTest(
    typename BlockScan::TempStorage &temp_storage, T (&data)[1], IdentityT &identity, ScanOp &scan_op, T &aggregate, PrefixCallbackOp &prefix_op, Int2Type<PREFIX_AGGREGATE> test_mode)
{
    BlockScan(temp_storage).ExclusiveScan(data[0], data[0], identity, scan_op, aggregate, prefix_op);
}

/// Exclusive scan (PREFIX_AGGREGATE, ITEMS_PER_THREAD)
template <typename BlockScan, typename T, typename ScanOp, typename IdentityT, typename PrefixCallbackOp, int ITEMS_PER_THREAD>
__device__ __forceinline__ void DeviceTest(
    typename BlockScan::TempStorage &temp_storage, T (&data)[ITEMS_PER_THREAD], IdentityT &identity, ScanOp &scan_op, T &aggregate, PrefixCallbackOp &prefix_op, Int2Type<PREFIX_AGGREGATE> test_mode)
{
    BlockScan(temp_storage).ExclusiveScan(data, data, identity, scan_op, aggregate, prefix_op);
}


//---------------------------------------------------------------------
// Exclusive sum
//---------------------------------------------------------------------

/// Exclusive sum (BASIC, 1)
template <typename BlockScan, typename T, typename IdentityT, typename PrefixCallbackOp>
__device__ __forceinline__ void DeviceTest(
    typename BlockScan::TempStorage &temp_storage, T (&data)[1], IdentityT &identity, Sum &scan_op, T &aggregate, PrefixCallbackOp &prefix_op, Int2Type<BASIC> test_mode)
{
    BlockScan(temp_storage).ExclusiveSum(data[0], data[0]);
}

/// Exclusive sum (BASIC, ITEMS_PER_THREAD)
template <typename BlockScan, typename T, typename IdentityT, typename PrefixCallbackOp, int ITEMS_PER_THREAD>
__device__ __forceinline__ void DeviceTest(
    typename BlockScan::TempStorage &temp_storage, T (&data)[ITEMS_PER_THREAD], IdentityT &identity, Sum &scan_op, T &aggregate, PrefixCallbackOp &prefix_op, Int2Type<BASIC> test_mode)
{
    BlockScan(temp_storage).ExclusiveSum(data, data);
}

/// Exclusive sum (AGGREGATE, 1)
template <typename BlockScan, typename T, typename IdentityT, typename PrefixCallbackOp>
__device__ __forceinline__ void DeviceTest(
    typename BlockScan::TempStorage &temp_storage, T (&data)[1], IdentityT &identity, Sum &scan_op, T &aggregate, PrefixCallbackOp &prefix_op, Int2Type<AGGREGATE> test_mode)
{
    BlockScan(temp_storage).ExclusiveSum(data[0], data[0], aggregate);
}

/// Exclusive sum (AGGREGATE, ITEMS_PER_THREAD)
template <typename BlockScan, typename T, typename IdentityT, typename PrefixCallbackOp, int ITEMS_PER_THREAD>
__device__ __forceinline__ void DeviceTest(
    typename BlockScan::TempStorage &temp_storage, T (&data)[ITEMS_PER_THREAD], IdentityT &identity, Sum &scan_op, T &aggregate, PrefixCallbackOp &prefix_op, Int2Type<AGGREGATE> test_mode)
{
    BlockScan(temp_storage).ExclusiveSum(data, data, aggregate);
}

/// Exclusive sum (PREFIX_AGGREGATE, 1)
template <typename BlockScan, typename T, typename IdentityT, typename PrefixCallbackOp>
__device__ __forceinline__ void DeviceTest(
    typename BlockScan::TempStorage &temp_storage, T (&data)[1], IdentityT &identity, Sum &scan_op, T &aggregate, PrefixCallbackOp &prefix_op, Int2Type<PREFIX_AGGREGATE> test_mode)
{
    BlockScan(temp_storage).ExclusiveSum(data[0], data[0], aggregate, prefix_op);
}

/// Exclusive sum (PREFIX_AGGREGATE, ITEMS_PER_THREAD)
template <typename BlockScan, typename T, typename IdentityT, typename PrefixCallbackOp, int ITEMS_PER_THREAD>
__device__ __forceinline__ void DeviceTest(
    typename BlockScan::TempStorage &temp_storage, T (&data)[ITEMS_PER_THREAD], IdentityT &identity, Sum &scan_op, T &aggregate, PrefixCallbackOp &prefix_op, Int2Type<PREFIX_AGGREGATE> test_mode)
{
    BlockScan(temp_storage).ExclusiveSum(data, data, aggregate, prefix_op);
}


//---------------------------------------------------------------------
// Inclusive scan
//---------------------------------------------------------------------

/// Inclusive scan (BASIC, 1)
template <typename BlockScan, typename T, typename ScanOp, typename PrefixCallbackOp>
__device__ __forceinline__ void DeviceTest(
    typename BlockScan::TempStorage &temp_storage, T (&data)[1], NullType &identity, ScanOp &scan_op, T &aggregate, PrefixCallbackOp &prefix_op, Int2Type<BASIC> test_mode)
{
    BlockScan(temp_storage).InclusiveScan(data[0], data[0], scan_op);
}

/// Inclusive scan (BASIC, ITEMS_PER_THREAD)
template <typename BlockScan, typename T, typename ScanOp, typename PrefixCallbackOp, int ITEMS_PER_THREAD>
__device__ __forceinline__ void DeviceTest(
    typename BlockScan::TempStorage &temp_storage, T (&data)[ITEMS_PER_THREAD], NullType &identity, ScanOp &scan_op, T &aggregate, PrefixCallbackOp &prefix_op, Int2Type<BASIC> test_mode)
{
    BlockScan(temp_storage).InclusiveScan(data, data, scan_op);
}

/// Inclusive scan (AGGREGATE, 1)
template <typename BlockScan, typename T, typename ScanOp, typename PrefixCallbackOp>
__device__ __forceinline__ void DeviceTest(
    typename BlockScan::TempStorage &temp_storage, T (&data)[1], NullType &identity, ScanOp &scan_op, T &aggregate, PrefixCallbackOp &prefix_op, Int2Type<AGGREGATE> test_mode)
{
    BlockScan(temp_storage).InclusiveScan(data[0], data[0], scan_op, aggregate);
}

/// Inclusive scan (AGGREGATE, ITEMS_PER_THREAD)
template <typename BlockScan, typename T, typename ScanOp, typename PrefixCallbackOp, int ITEMS_PER_THREAD>
__device__ __forceinline__ void DeviceTest(
    typename BlockScan::TempStorage &temp_storage, T (&data)[ITEMS_PER_THREAD], NullType &identity, ScanOp &scan_op, T &aggregate, PrefixCallbackOp &prefix_op, Int2Type<AGGREGATE> test_mode)
{
    BlockScan(temp_storage).InclusiveScan(data, data, scan_op, aggregate);
}

/// Inclusive scan (PREFIX_AGGREGATE, 1)
template <typename BlockScan, typename T, typename ScanOp, typename PrefixCallbackOp>
__device__ __forceinline__ void DeviceTest(
    typename BlockScan::TempStorage &temp_storage, T (&data)[1], NullType &identity, ScanOp &scan_op, T &aggregate, PrefixCallbackOp &prefix_op, Int2Type<PREFIX_AGGREGATE> test_mode)
{
    BlockScan(temp_storage).InclusiveScan(data[0], data[0], scan_op, aggregate, prefix_op);
}

/// Inclusive scan (PREFIX_AGGREGATE, ITEMS_PER_THREAD)
template <typename BlockScan, typename T, typename ScanOp, typename PrefixCallbackOp, int ITEMS_PER_THREAD>
__device__ __forceinline__ void DeviceTest(
    typename BlockScan::TempStorage &temp_storage, T (&data)[ITEMS_PER_THREAD], NullType &identity, ScanOp &scan_op, T &aggregate, PrefixCallbackOp &prefix_op, Int2Type<PREFIX_AGGREGATE> test_mode)
{
    BlockScan(temp_storage).InclusiveScan(data, data, scan_op, aggregate, prefix_op);
}


//---------------------------------------------------------------------
// Inclusive sum
//---------------------------------------------------------------------

/// Inclusive sum (BASIC, 1)
template <typename BlockScan, typename T, typename PrefixCallbackOp>
__device__ __forceinline__ void DeviceTest(
    typename BlockScan::TempStorage &temp_storage, T (&data)[1], NullType &identity, Sum &scan_op, T &aggregate, PrefixCallbackOp &prefix_op, Int2Type<BASIC> test_mode)
{
    BlockScan(temp_storage).InclusiveSum(data[0], data[0]);
}

/// Inclusive sum (BASIC, ITEMS_PER_THREAD)
template <typename BlockScan, typename T, typename PrefixCallbackOp, int ITEMS_PER_THREAD>
__device__ __forceinline__ void DeviceTest(
    typename BlockScan::TempStorage &temp_storage, T (&data)[ITEMS_PER_THREAD], NullType &identity, Sum &scan_op, T &aggregate, PrefixCallbackOp &prefix_op, Int2Type<BASIC> test_mode)
{
    BlockScan(temp_storage).InclusiveSum(data, data);
}

/// Inclusive sum (AGGREGATE, 1)
template <typename BlockScan, typename T, typename PrefixCallbackOp>
__device__ __forceinline__ void DeviceTest(
    typename BlockScan::TempStorage &temp_storage, T (&data)[1], NullType &identity, Sum &scan_op, T &aggregate, PrefixCallbackOp &prefix_op, Int2Type<AGGREGATE> test_mode)
{
    BlockScan(temp_storage).InclusiveSum(data[0], data[0], aggregate);
}

/// Inclusive sum (AGGREGATE, ITEMS_PER_THREAD)
template <typename BlockScan, typename T, typename PrefixCallbackOp, int ITEMS_PER_THREAD>
__device__ __forceinline__ void DeviceTest(
    typename BlockScan::TempStorage &temp_storage, T (&data)[ITEMS_PER_THREAD], NullType &identity, Sum &scan_op, T &aggregate, PrefixCallbackOp &prefix_op, Int2Type<AGGREGATE> test_mode)
{
    BlockScan(temp_storage).InclusiveSum(data, data, aggregate);
}

/// Inclusive sum (PREFIX_AGGREGATE, 1)
template <typename BlockScan, typename T, typename PrefixCallbackOp>
__device__ __forceinline__ void DeviceTest(
    typename BlockScan::TempStorage &temp_storage, T (&data)[1], NullType &identity, Sum &scan_op, T &aggregate, PrefixCallbackOp &prefix_op, Int2Type<PREFIX_AGGREGATE> test_mode)
{
    BlockScan(temp_storage).InclusiveSum(data[0], data[0], aggregate, prefix_op);
}

/// Inclusive sum (PREFIX_AGGREGATE, ITEMS_PER_THREAD)
template <typename BlockScan, typename T, typename PrefixCallbackOp, int ITEMS_PER_THREAD>
__device__ __forceinline__ void DeviceTest(
    typename BlockScan::TempStorage &temp_storage, T (&data)[ITEMS_PER_THREAD], NullType &identity, Sum &scan_op, T &aggregate, PrefixCallbackOp &prefix_op, Int2Type<PREFIX_AGGREGATE> test_mode)
{
    BlockScan(temp_storage).InclusiveSum(data, data, aggregate, prefix_op);
}



//---------------------------------------------------------------------
// Test kernels
//---------------------------------------------------------------------

/**
 * BlockScan test kernel.
 */
template <
    int                 BLOCK_THREADS,
    int                 ITEMS_PER_THREAD,
    TestMode            TEST_MODE,
    BlockScanAlgorithm  ALGORITHM,
    typename            T,
    typename            ScanOp,
    typename            IdentityT>
__global__ void BlockScanKernel(
    T                   *d_in,
    T                   *d_out,
    T                   *d_aggregate,
    ScanOp              scan_op,
    IdentityT           identity,
    T                   prefix,
    clock_t             *d_elapsed)
{
    const int TILE_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD;

    // Parameterize BlockScan type for our thread block
    typedef BlockScan<T, BLOCK_THREADS, ALGORITHM> BlockScan;

    // Allocate temp storage in shared memory
    __shared__ typename BlockScan::TempStorage temp_storage;

    // Per-thread tile data
    T data[ITEMS_PER_THREAD];
    LoadDirectBlocked(threadIdx.x, d_in, data);

    // Start cycle timer
    clock_t start = clock();

    // Test scan
    T aggregate;
    BlockPrefixCallbackOp<T, ScanOp> prefix_op(prefix, scan_op);
    DeviceTest<BlockScan>(temp_storage, data, identity, scan_op, aggregate, prefix_op, Int2Type<TEST_MODE>());

    // Stop cycle timer
#if CUB_PTX_VERSION == 100
    // Bug: recording stop clock causes mis-write of running prefix value
    clock_t stop = 0;
#else
    clock_t stop = clock();
#endif // CUB_PTX_VERSION == 100

    // Store output
    StoreDirectBlocked(threadIdx.x, d_out, data);

    // Store aggregate
    d_aggregate[threadIdx.x] = aggregate;

    // Store prefix and time
    if (threadIdx.x == 0)
    {
        d_out[TILE_SIZE] = prefix_op.prefix;
        *d_elapsed = (start > stop) ? start - stop : stop - start;
    }
}



//---------------------------------------------------------------------
// Host utility subroutines
//---------------------------------------------------------------------

/**
 * Initialize exclusive-scan problem (and solution)
 */
template <
    typename    T,
    typename    ScanOp,
    typename    IdentityT>
T Initialize(
    GenMode     gen_mode,
    T           *h_in,
    T           *h_reference,
    int         num_items,
    ScanOp      scan_op,
    IdentityT   identity,
    T           *prefix)
{
    T inclusive = (prefix != NULL) ? *prefix : identity;
    T aggregate = identity;

    for (int i = 0; i < num_items; ++i)
    {
        InitValue(gen_mode, h_in[i], i);
        h_reference[i] = inclusive;
        inclusive = scan_op(inclusive, h_in[i]);
        aggregate = scan_op(aggregate, h_in[i]);
    }

    return aggregate;
}


/**
 * Initialize inclusive-scan problem (and solution)
 */
template <
    typename    T,
    typename    ScanOp>
T Initialize(
    GenMode     gen_mode,
    T           *h_in,
    T           *h_reference,
    int         num_items,
    ScanOp      scan_op,
    NullType,
    T           *prefix)
{
    T inclusive;
    T aggregate;
    for (int i = 0; i < num_items; ++i)
    {
        InitValue(gen_mode, h_in[i], i);
        if (i == 0)
        {
            inclusive = (prefix != NULL) ?
                scan_op(*prefix, h_in[0]) :
                h_in[0];
            aggregate = h_in[0];
        }
        else
        {
            inclusive = scan_op(inclusive, h_in[i]);
            aggregate = scan_op(aggregate, h_in[i]);
        }
        h_reference[i] = inclusive;
    }

    return aggregate;
}


/**
 * Test threadblock scan
 */
template <
    int                 BLOCK_THREADS,
    int                 ITEMS_PER_THREAD,
    TestMode            TEST_MODE,
    BlockScanAlgorithm  ALGORITHM,
    typename            ScanOp,
    typename            IdentityT,        // NullType implies inclusive-scan, otherwise inclusive scan
    typename            T>
void Test(
    GenMode         gen_mode,
    ScanOp          scan_op,
    IdentityT       identity,
    T               prefix,
    const char      *type_string)
{
    const int TILE_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD;

    // Allocate host arrays
    T *h_in = new T[TILE_SIZE];
    T *h_reference = new T[TILE_SIZE];
    T *h_aggregate = new T[BLOCK_THREADS];

    // Initialize problem
    T *p_prefix = (TEST_MODE == PREFIX_AGGREGATE) ? &prefix : NULL;
    T aggregate = Initialize(
        gen_mode,
        h_in,
        h_reference,
        TILE_SIZE,
        scan_op,
        identity,
        p_prefix);

    // Test reference aggregate is returned in all threads
    for (int i = 0; i < BLOCK_THREADS; ++i)
    {
        h_aggregate[i] = aggregate;
    }

    // Run kernel
    printf("Test-mode %d, gen-mode %d, policy %d, %s BlockScan, %d threadblock threads, %d items per thread, %d tile size, %s (%d bytes) elements:\n",
        TEST_MODE,
        gen_mode,
        ALGORITHM,
        (Equals<IdentityT, NullType>::VALUE) ? "Inclusive" : "Exclusive",
        BLOCK_THREADS,
        ITEMS_PER_THREAD,
        TILE_SIZE,
        type_string,
        (int) sizeof(T));
    fflush(stdout);

    // Initialize/clear device arrays
    T       *d_in = NULL;
    T       *d_out = NULL;
    T       *d_aggregate = NULL;
    clock_t *d_elapsed = NULL;
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_elapsed, sizeof(unsigned long long)));
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_in, sizeof(T) * TILE_SIZE));
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_out, sizeof(T) * (TILE_SIZE + 2)));
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_aggregate, sizeof(T) * BLOCK_THREADS));
    CubDebugExit(cudaMemcpy(d_in, h_in, sizeof(T) * TILE_SIZE, cudaMemcpyHostToDevice));
    CubDebugExit(cudaMemset(d_out, 0, sizeof(T) * (TILE_SIZE + 1)));
    CubDebugExit(cudaMemset(d_aggregate, 0, sizeof(T) * BLOCK_THREADS));

    // Display input problem data
    if (g_verbose)
    {
        printf("Input data: ");
        for (int i = 0; i < TILE_SIZE; i++)
        {
            std::cout << CoutCast(h_in[i]) << ", ";
        }
        printf("\n\n");
    }

    // Run aggregate/prefix kernel
    BlockScanKernel<BLOCK_THREADS, ITEMS_PER_THREAD, TEST_MODE, ALGORITHM><<<1, BLOCK_THREADS>>>(
        d_in,
        d_out,
        d_aggregate,
        scan_op,
        identity,
        prefix,
        d_elapsed);

    CubDebugExit(cudaDeviceSynchronize());

    // Copy out and display results
    printf("\tScan results: ");
    int compare = CompareDeviceResults(h_reference, d_out, TILE_SIZE, g_verbose, g_verbose);
    printf("%s\n", compare ? "FAIL" : "PASS");
    AssertEquals(0, compare);

    // Copy out and display aggregate
    if ((TEST_MODE == AGGREGATE) || (TEST_MODE == PREFIX_AGGREGATE))
    {
        printf("\tScan aggregate: ");
        compare = CompareDeviceResults(h_aggregate, d_aggregate, BLOCK_THREADS, g_verbose, g_verbose);
        printf("%s\n", compare ? "FAIL" : "PASS");
        AssertEquals(0, compare);

        // Copy out and display updated prefix
        if (TEST_MODE == PREFIX_AGGREGATE)
        {
            printf("\tScan prefix: ");
            T updated_prefix = scan_op(prefix, aggregate);
            compare = CompareDeviceResults(&updated_prefix, d_out + TILE_SIZE, 1, g_verbose, g_verbose);
            printf("%s\n", compare ? "FAIL" : "PASS");
            AssertEquals(0, compare);
        }
    }

    printf("\tElapsed clocks: ");
    DisplayDeviceResults(d_elapsed, 1);

    // Cleanup
    if (h_in) delete[] h_in;
    if (h_reference) delete[] h_reference;
    if (h_aggregate) delete[] h_aggregate;
    if (d_in) CubDebugExit(g_allocator.DeviceFree(d_in));
    if (d_out) CubDebugExit(g_allocator.DeviceFree(d_out));
    if (d_aggregate) CubDebugExit(g_allocator.DeviceFree(d_aggregate));
    if (d_elapsed) CubDebugExit(g_allocator.DeviceFree(d_elapsed));
}


/**
 * Run test for different policy types
 */
template <
    int         BLOCK_THREADS,
    int         ITEMS_PER_THREAD,
    TestMode    TEST_MODE,
    typename    ScanOp,
    typename    IdentityT,
    typename    T>
void Test(
    GenMode     gen_mode,
    ScanOp      scan_op,
    IdentityT   identity,
    T           prefix,
    const char  *type_string)
{
    Test<BLOCK_THREADS, ITEMS_PER_THREAD, TEST_MODE, BLOCK_SCAN_RAKING>(gen_mode, scan_op, identity, prefix, type_string);
    Test<BLOCK_THREADS, ITEMS_PER_THREAD, TEST_MODE, BLOCK_SCAN_RAKING_MEMOIZE>(gen_mode, scan_op, identity, prefix, type_string);
    Test<BLOCK_THREADS, ITEMS_PER_THREAD, TEST_MODE, BLOCK_SCAN_WARP_SCANS>(gen_mode, scan_op, identity, prefix, type_string);
}


/**
 * Run tests for different primitive variants
 */
template <
    int         BLOCK_THREADS,
    int         ITEMS_PER_THREAD,
    typename    ScanOp,
    typename    T>
void Test(
    GenMode     gen_mode,
    ScanOp      scan_op,
    T           identity,
    T           prefix,
    const char  *type_string)
{
    // Exclusive
    Test<BLOCK_THREADS, ITEMS_PER_THREAD, BASIC>(gen_mode, scan_op, identity, prefix, type_string);
    Test<BLOCK_THREADS, ITEMS_PER_THREAD, AGGREGATE>(gen_mode, scan_op, identity, prefix, type_string);
    Test<BLOCK_THREADS, ITEMS_PER_THREAD, PREFIX_AGGREGATE>(gen_mode, scan_op, identity, prefix, type_string);

    // Inclusive
    Test<BLOCK_THREADS, ITEMS_PER_THREAD, BASIC>(gen_mode, scan_op, NullType(), prefix, type_string);
    Test<BLOCK_THREADS, ITEMS_PER_THREAD, AGGREGATE>(gen_mode, scan_op, NullType(), prefix, type_string);
    Test<BLOCK_THREADS, ITEMS_PER_THREAD, PREFIX_AGGREGATE>(gen_mode, scan_op, NullType(), prefix, type_string);
}


/**
 * Run tests for different data types and scan ops
 */
template <
    int BLOCK_THREADS,
    int ITEMS_PER_THREAD>
void Test(GenMode gen_mode)
{
/*
    // primitive
    Test<BLOCK_THREADS, ITEMS_PER_THREAD>(gen_mode, Sum(), (unsigned char) 0, (unsigned char) 99, CUB_TYPE_STRING(Sum<unsigned char>));
    Test<BLOCK_THREADS, ITEMS_PER_THREAD>(gen_mode, Sum(), (unsigned short) 0, (unsigned short) 99, CUB_TYPE_STRING(Sum<unsigned short>));
    Test<BLOCK_THREADS, ITEMS_PER_THREAD>(gen_mode, Sum(), (unsigned int) 0, (unsigned int) 99, CUB_TYPE_STRING(Sum<unsigned int>));
    Test<BLOCK_THREADS, ITEMS_PER_THREAD>(gen_mode, Sum(), (unsigned long long) 0, (unsigned long long) 99, CUB_TYPE_STRING(Sum<unsigned long long>));

    // primitive (alternative scan op)
    Test<BLOCK_THREADS, ITEMS_PER_THREAD>(gen_mode, Max(), (unsigned char) 0, (unsigned char) 99, CUB_TYPE_STRING(Max<unsigned char>));
    Test<BLOCK_THREADS, ITEMS_PER_THREAD>(gen_mode, Max(), (unsigned short) 0, (unsigned short) 99, CUB_TYPE_STRING(Max<unsigned short>));
    Test<BLOCK_THREADS, ITEMS_PER_THREAD>(gen_mode, Max(), (unsigned int) 0, (unsigned int) 99, CUB_TYPE_STRING(Max<unsigned int>));
    Test<BLOCK_THREADS, ITEMS_PER_THREAD>(gen_mode, Max(), (unsigned long long) 0, (unsigned long long) 99, CUB_TYPE_STRING(Max<unsigned long long>));

    // vec-1
    Test<BLOCK_THREADS, ITEMS_PER_THREAD>(gen_mode, Sum(), make_uchar1(0), make_uchar1(17), CUB_TYPE_STRING(Sum<uchar1>));

    // vec-2
    Test<BLOCK_THREADS, ITEMS_PER_THREAD>(gen_mode, Sum(), make_uchar2(0, 0), make_uchar2(17, 21), CUB_TYPE_STRING(Sum<uchar2>));
    Test<BLOCK_THREADS, ITEMS_PER_THREAD>(gen_mode, Sum(), make_ushort2(0, 0), make_ushort2(17, 21), CUB_TYPE_STRING(Sum<ushort2>));
    Test<BLOCK_THREADS, ITEMS_PER_THREAD>(gen_mode, Sum(), make_uint2(0, 0), make_uint2(17, 21), CUB_TYPE_STRING(Sum<uint2>));
    Test<BLOCK_THREADS, ITEMS_PER_THREAD>(gen_mode, Sum(), make_ulonglong2(0, 0), make_ulonglong2(17, 21), CUB_TYPE_STRING(Sum<ulonglong2>));

    // vec-4
    Test<BLOCK_THREADS, ITEMS_PER_THREAD>(gen_mode, Sum(), make_uchar4(0, 0, 0, 0), make_uchar4(17, 21, 32, 85), CUB_TYPE_STRING(Sum<uchar4>));
    Test<BLOCK_THREADS, ITEMS_PER_THREAD>(gen_mode, Sum(), make_ushort4(0, 0, 0, 0), make_ushort4(17, 21, 32, 85), CUB_TYPE_STRING(Sum<ushort4>));
    Test<BLOCK_THREADS, ITEMS_PER_THREAD>(gen_mode, Sum(), make_uint4(0, 0, 0, 0), make_uint4(17, 21, 32, 85), CUB_TYPE_STRING(Sum<uint4>));
    Test<BLOCK_THREADS, ITEMS_PER_THREAD>(gen_mode, Sum(), make_ulonglong4(0, 0, 0, 0), make_ulonglong4(17, 21, 32, 85), CUB_TYPE_STRING(Sum<ulonglong4>));
*/
    // complex
    Test<BLOCK_THREADS, ITEMS_PER_THREAD>(gen_mode, Sum(), TestFoo::MakeTestFoo(0, 0, 0, 0), TestFoo::MakeTestFoo(17, 21, 32, 85), CUB_TYPE_STRING(Sum<TestFoo>));
    Test<BLOCK_THREADS, ITEMS_PER_THREAD>(gen_mode, Sum(), TestBar(0, 0), TestBar(17, 21), CUB_TYPE_STRING(Sum<TestBar>));
}


/**
 * Run tests for different problem generation options
 */
template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
void Test()
{
    Test<BLOCK_THREADS, ITEMS_PER_THREAD>(UNIFORM);
    Test<BLOCK_THREADS, ITEMS_PER_THREAD>(INTEGER_SEED);
    Test<BLOCK_THREADS, ITEMS_PER_THREAD>(RANDOM);
}


/**
 * Run tests for different items per thread
 */
template <int BLOCK_THREADS>
void Test()
{
    Test<BLOCK_THREADS, 1>();

#if defined(SM100) || defined(SM110) || defined(SM130)
    // Open64 compiler can't handle the number of test cases
#else
    Test<BLOCK_THREADS, 2>();
#endif

    Test<BLOCK_THREADS, 9>();
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

#ifdef QUICK_TEST

    // Compile/run quick tests
    Test<128, 1, AGGREGATE, BLOCK_SCAN_WARP_SCANS>(UNIFORM, Sum(), int(0), int(10), CUB_TYPE_STRING(Sum<int>));
    Test<128, 4, AGGREGATE, BLOCK_SCAN_RAKING_MEMOIZE>(UNIFORM, Sum(), int(0), int(10), CUB_TYPE_STRING(Sum<int>));

    TestFoo prefix = TestFoo::MakeTestFoo(17, 21, 32, 85);
    Test<128, 2, PREFIX_AGGREGATE, BLOCK_SCAN_RAKING>(INTEGER_SEED, Sum(), NullType(), prefix, CUB_TYPE_STRING(Sum<TestFoo>));

#else

    // Compile/run thorough tests
    for (int i = 0; i <= g_repeat; ++i)
    {
        // Run tests for different threadblock sizes
        Test<17>();
        Test<32>();
        Test<62>();
        Test<65>();
//            Test<96>();             // TODO: file bug for UNREACHABLE error for Test<96, 9, BASIC, BLOCK_SCAN_RAKING>(UNIFORM, Sum(), NullType(), make_ulonglong2(17, 21), CUB_TYPE_STRING(Sum<ulonglong2>));
        Test<128>();
    }

#endif

    return 0;
}




