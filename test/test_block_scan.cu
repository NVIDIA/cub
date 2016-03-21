/******************************************************************************
 * Copyright (c) 2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2016, NVIDIA CORPORATION.  All rights reserved.
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
#include <limits>
#include <typeinfo>

#include <cub/block/block_scan.cuh>
#include <cub/block/block_load.cuh>
#include <cub/block/block_store.cuh>
#include <cub/util_ptx.cuh>
#include <cub/util_allocator.cuh>

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
    BASIC,
    AGGREGATE,
    PREFIX_AGGREGATE,
};



/**
 * Stateful prefix functor
 */
template <
    typename T,
    typename ScanOp>
struct BlockPrefixCallbackOp
{
    int     linear_tid;
    T       prefix;
    ScanOp  scan_op;

    __device__ __forceinline__
    BlockPrefixCallbackOp(int linear_tid, T prefix, ScanOp scan_op) :
        linear_tid(linear_tid),
        prefix(prefix),
        scan_op(scan_op)
    {}

    __device__ __forceinline__
    T operator()(T block_aggregate)
    {
        // For testing purposes
        T retval = (linear_tid == 0) ? prefix  : T();
        prefix = scan_op(prefix, block_aggregate);
        return retval;
    }
};


//---------------------------------------------------------------------
// Exclusive scan
//---------------------------------------------------------------------

/// Exclusive scan (BASIC, 1)
template <typename BlockScanT, typename T, typename ScanOp, typename IdentityT, typename PrefixCallbackOp>
__device__ __forceinline__ void DeviceTest(
    BlockScanT &block_scan, T (&data)[1], IdentityT &identity, ScanOp &scan_op, T &aggregate, PrefixCallbackOp &prefix_op, Int2Type<BASIC> test_mode)
{
    block_scan.ExclusiveScan(data[0], data[0], identity, scan_op);
}

/// Exclusive scan (BASIC, ITEMS_PER_THREAD)
template <typename BlockScanT, typename T, typename ScanOp, typename IdentityT, typename PrefixCallbackOp, int ITEMS_PER_THREAD>
__device__ __forceinline__ void DeviceTest(
    BlockScanT &block_scan, T (&data)[ITEMS_PER_THREAD], IdentityT &identity, ScanOp &scan_op, T &aggregate, PrefixCallbackOp &prefix_op, Int2Type<BASIC> test_mode)
{
    block_scan.ExclusiveScan(data, data, identity, scan_op);
}

/// Exclusive scan (AGGREGATE, 1)
template <typename BlockScanT, typename T, typename ScanOp, typename IdentityT, typename PrefixCallbackOp>
__device__ __forceinline__ void DeviceTest(
    BlockScanT &block_scan, T (&data)[1], IdentityT &identity, ScanOp &scan_op, T &aggregate, PrefixCallbackOp &prefix_op, Int2Type<AGGREGATE> test_mode)
{
    block_scan.ExclusiveScan(data[0], data[0], identity, scan_op, aggregate);
}

/// Exclusive scan (AGGREGATE, ITEMS_PER_THREAD)
template <typename BlockScanT, typename T, typename ScanOp, typename IdentityT, typename PrefixCallbackOp, int ITEMS_PER_THREAD>
__device__ __forceinline__ void DeviceTest(
    BlockScanT &block_scan, T (&data)[ITEMS_PER_THREAD], IdentityT &identity, ScanOp &scan_op, T &aggregate, PrefixCallbackOp &prefix_op, Int2Type<AGGREGATE> test_mode)
{
    block_scan.ExclusiveScan(data, data, identity, scan_op, aggregate);
}

/// Exclusive scan (PREFIX_AGGREGATE, 1)
template <typename BlockScanT, typename T, typename ScanOp, typename IdentityT, typename PrefixCallbackOp>
__device__ __forceinline__ void DeviceTest(
    BlockScanT &block_scan, T (&data)[1], IdentityT &identity, ScanOp &scan_op, T &aggregate, PrefixCallbackOp &prefix_op, Int2Type<PREFIX_AGGREGATE> test_mode)
{
    block_scan.ExclusiveScan(data[0], data[0], identity, scan_op, aggregate, prefix_op);
}

/// Exclusive scan (PREFIX_AGGREGATE, ITEMS_PER_THREAD)
template <typename BlockScanT, typename T, typename ScanOp, typename IdentityT, typename PrefixCallbackOp, int ITEMS_PER_THREAD>
__device__ __forceinline__ void DeviceTest(
    BlockScanT &block_scan, T (&data)[ITEMS_PER_THREAD], IdentityT &identity, ScanOp &scan_op, T &aggregate, PrefixCallbackOp &prefix_op, Int2Type<PREFIX_AGGREGATE> test_mode)
{
    block_scan.ExclusiveScan(data, data, identity, scan_op, aggregate, prefix_op);
}


//---------------------------------------------------------------------
// Exclusive sum
//---------------------------------------------------------------------

/// Exclusive sum (BASIC, 1)
template <typename BlockScanT, typename T, typename IdentityT, typename PrefixCallbackOp>
__device__ __forceinline__ void DeviceTest(
    BlockScanT &block_scan, T (&data)[1], IdentityT &identity, Sum &scan_op, T &aggregate, PrefixCallbackOp &prefix_op, Int2Type<BASIC> test_mode)
{
    block_scan.ExclusiveSum(data[0], data[0]);
}

/// Exclusive sum (BASIC, ITEMS_PER_THREAD)
template <typename BlockScanT, typename T, typename IdentityT, typename PrefixCallbackOp, int ITEMS_PER_THREAD>
__device__ __forceinline__ void DeviceTest(
    BlockScanT &block_scan, T (&data)[ITEMS_PER_THREAD], IdentityT &identity, Sum &scan_op, T &aggregate, PrefixCallbackOp &prefix_op, Int2Type<BASIC> test_mode)
{
    block_scan.ExclusiveSum(data, data);
}

/// Exclusive sum (AGGREGATE, 1)
template <typename BlockScanT, typename T, typename IdentityT, typename PrefixCallbackOp>
__device__ __forceinline__ void DeviceTest(
    BlockScanT &block_scan, T (&data)[1], IdentityT &identity, Sum &scan_op, T &aggregate, PrefixCallbackOp &prefix_op, Int2Type<AGGREGATE> test_mode)
{
    block_scan.ExclusiveSum(data[0], data[0], aggregate);
}

/// Exclusive sum (AGGREGATE, ITEMS_PER_THREAD)
template <typename BlockScanT, typename T, typename IdentityT, typename PrefixCallbackOp, int ITEMS_PER_THREAD>
__device__ __forceinline__ void DeviceTest(
    BlockScanT &block_scan, T (&data)[ITEMS_PER_THREAD], IdentityT &identity, Sum &scan_op, T &aggregate, PrefixCallbackOp &prefix_op, Int2Type<AGGREGATE> test_mode)
{
    block_scan.ExclusiveSum(data, data, aggregate);
}

/// Exclusive sum (PREFIX_AGGREGATE, 1)
template <typename BlockScanT, typename T, typename IdentityT, typename PrefixCallbackOp>
__device__ __forceinline__ void DeviceTest(
    BlockScanT &block_scan, T (&data)[1], IdentityT &identity, Sum &scan_op, T &aggregate, PrefixCallbackOp &prefix_op, Int2Type<PREFIX_AGGREGATE> test_mode)
{
    block_scan.ExclusiveSum(data[0], data[0], aggregate, prefix_op);
}

/// Exclusive sum (PREFIX_AGGREGATE, ITEMS_PER_THREAD)
template <typename BlockScanT, typename T, typename IdentityT, typename PrefixCallbackOp, int ITEMS_PER_THREAD>
__device__ __forceinline__ void DeviceTest(
    BlockScanT &block_scan, T (&data)[ITEMS_PER_THREAD], IdentityT &identity, Sum &scan_op, T &aggregate, PrefixCallbackOp &prefix_op, Int2Type<PREFIX_AGGREGATE> test_mode)
{
    block_scan.ExclusiveSum(data, data, aggregate, prefix_op);
}


//---------------------------------------------------------------------
// Inclusive scan
//---------------------------------------------------------------------

/// Inclusive scan (BASIC, 1)
template <typename BlockScanT, typename T, typename ScanOp, typename PrefixCallbackOp>
__device__ __forceinline__ void DeviceTest(
    BlockScanT &block_scan, T (&data)[1], NullType &identity, ScanOp &scan_op, T &aggregate, PrefixCallbackOp &prefix_op, Int2Type<BASIC> test_mode)
{
    block_scan.InclusiveScan(data[0], data[0], scan_op);
}

/// Inclusive scan (BASIC, ITEMS_PER_THREAD)
template <typename BlockScanT, typename T, typename ScanOp, typename PrefixCallbackOp, int ITEMS_PER_THREAD>
__device__ __forceinline__ void DeviceTest(
    BlockScanT &block_scan, T (&data)[ITEMS_PER_THREAD], NullType &identity, ScanOp &scan_op, T &aggregate, PrefixCallbackOp &prefix_op, Int2Type<BASIC> test_mode)
{
    block_scan.InclusiveScan(data, data, scan_op);
}

/// Inclusive scan (AGGREGATE, 1)
template <typename BlockScanT, typename T, typename ScanOp, typename PrefixCallbackOp>
__device__ __forceinline__ void DeviceTest(
    BlockScanT &block_scan, T (&data)[1], NullType &identity, ScanOp &scan_op, T &aggregate, PrefixCallbackOp &prefix_op, Int2Type<AGGREGATE> test_mode)
{
    block_scan.InclusiveScan(data[0], data[0], scan_op, aggregate);
}

/// Inclusive scan (AGGREGATE, ITEMS_PER_THREAD)
template <typename BlockScanT, typename T, typename ScanOp, typename PrefixCallbackOp, int ITEMS_PER_THREAD>
__device__ __forceinline__ void DeviceTest(
    BlockScanT &block_scan, T (&data)[ITEMS_PER_THREAD], NullType &identity, ScanOp &scan_op, T &aggregate, PrefixCallbackOp &prefix_op, Int2Type<AGGREGATE> test_mode)
{
    block_scan.InclusiveScan(data, data, scan_op, aggregate);
}

/// Inclusive scan (PREFIX_AGGREGATE, 1)
template <typename BlockScanT, typename T, typename ScanOp, typename PrefixCallbackOp>
__device__ __forceinline__ void DeviceTest(
    BlockScanT &block_scan, T (&data)[1], NullType &identity, ScanOp &scan_op, T &aggregate, PrefixCallbackOp &prefix_op, Int2Type<PREFIX_AGGREGATE> test_mode)
{
    block_scan.InclusiveScan(data[0], data[0], scan_op, aggregate, prefix_op);
}

/// Inclusive scan (PREFIX_AGGREGATE, ITEMS_PER_THREAD)
template <typename BlockScanT, typename T, typename ScanOp, typename PrefixCallbackOp, int ITEMS_PER_THREAD>
__device__ __forceinline__ void DeviceTest(
    BlockScanT &block_scan, T (&data)[ITEMS_PER_THREAD], NullType &identity, ScanOp &scan_op, T &aggregate, PrefixCallbackOp &prefix_op, Int2Type<PREFIX_AGGREGATE> test_mode)
{
    block_scan.InclusiveScan(data, data, scan_op, aggregate, prefix_op);
}


//---------------------------------------------------------------------
// Inclusive sum
//---------------------------------------------------------------------

/// Inclusive sum (BASIC, 1)
template <typename BlockScanT, typename T, typename PrefixCallbackOp>
__device__ __forceinline__ void DeviceTest(
    BlockScanT &block_scan, T (&data)[1], NullType &identity, Sum &scan_op, T &aggregate, PrefixCallbackOp &prefix_op, Int2Type<BASIC> test_mode)
{
    block_scan.InclusiveSum(data[0], data[0]);
}

/// Inclusive sum (BASIC, ITEMS_PER_THREAD)
template <typename BlockScanT, typename T, typename PrefixCallbackOp, int ITEMS_PER_THREAD>
__device__ __forceinline__ void DeviceTest(
    BlockScanT &block_scan, T (&data)[ITEMS_PER_THREAD], NullType &identity, Sum &scan_op, T &aggregate, PrefixCallbackOp &prefix_op, Int2Type<BASIC> test_mode)
{
    block_scan.InclusiveSum(data, data);
}

/// Inclusive sum (AGGREGATE, 1)
template <typename BlockScanT, typename T, typename PrefixCallbackOp>
__device__ __forceinline__ void DeviceTest(
    BlockScanT &block_scan, T (&data)[1], NullType &identity, Sum &scan_op, T &aggregate, PrefixCallbackOp &prefix_op, Int2Type<AGGREGATE> test_mode)
{
    block_scan.InclusiveSum(data[0], data[0], aggregate);
}

/// Inclusive sum (AGGREGATE, ITEMS_PER_THREAD)
template <typename BlockScanT, typename T, typename PrefixCallbackOp, int ITEMS_PER_THREAD>
__device__ __forceinline__ void DeviceTest(
    BlockScanT &block_scan, T (&data)[ITEMS_PER_THREAD], NullType &identity, Sum &scan_op, T &aggregate, PrefixCallbackOp &prefix_op, Int2Type<AGGREGATE> test_mode)
{
    block_scan.InclusiveSum(data, data, aggregate);
}

/// Inclusive sum (PREFIX_AGGREGATE, 1)
template <typename BlockScanT, typename T, typename PrefixCallbackOp>
__device__ __forceinline__ void DeviceTest(
    BlockScanT &block_scan, T (&data)[1], NullType &identity, Sum &scan_op, T &aggregate, PrefixCallbackOp &prefix_op, Int2Type<PREFIX_AGGREGATE> test_mode)
{
    block_scan.InclusiveSum(data[0], data[0], aggregate, prefix_op);
}

/// Inclusive sum (PREFIX_AGGREGATE, ITEMS_PER_THREAD)
template <typename BlockScanT, typename T, typename PrefixCallbackOp, int ITEMS_PER_THREAD>
__device__ __forceinline__ void DeviceTest(
    BlockScanT &block_scan, T (&data)[ITEMS_PER_THREAD], NullType &identity, Sum &scan_op, T &aggregate, PrefixCallbackOp &prefix_op, Int2Type<PREFIX_AGGREGATE> test_mode)
{
    block_scan.InclusiveSum(data, data, aggregate, prefix_op);
}



//---------------------------------------------------------------------
// Test kernels
//---------------------------------------------------------------------

/**
 * BlockScan test kernel.
 */
template <
    int                 BLOCK_DIM_X,
    int                 BLOCK_DIM_Y,
    int                 BLOCK_DIM_Z,
    int                 ITEMS_PER_THREAD,
    TestMode            TEST_MODE,
    BlockScanAlgorithm  ALGORITHM,
    typename            T,
    typename            ScanOp,
    typename            IdentityT>
__launch_bounds__ (BLOCK_DIM_X * BLOCK_DIM_Y * BLOCK_DIM_Z)
__global__ void BlockScanKernel(
    T                   *d_in,
    T                   *d_out,
    T                   *d_aggregate,
    ScanOp              scan_op,
    IdentityT           identity,
    T                   prefix,
    clock_t             *d_elapsed)
{
    const int BLOCK_THREADS     = BLOCK_DIM_X * BLOCK_DIM_Y * BLOCK_DIM_Z;
    const int TILE_SIZE         = BLOCK_THREADS * ITEMS_PER_THREAD;

    // Parameterize BlockScan type for our thread block
    typedef BlockScan<T, BLOCK_DIM_X, ALGORITHM, BLOCK_DIM_Y, BLOCK_DIM_Z> BlockScanT;

    // Allocate temp storage in shared memory
    __shared__ typename BlockScanT::TempStorage temp_storage;

    int linear_tid = RowMajorTid(BLOCK_DIM_X, BLOCK_DIM_Y, BLOCK_DIM_Z);

    // Per-thread tile data
    T data[ITEMS_PER_THREAD];
    LoadDirectBlocked(linear_tid, d_in, data);

#if CUB_PTX_ARCH > 100
    // Bug: using the clock on SM10 causes codegen issues (e.g., uchar4 sum)
    clock_t start = clock();
#endif

    // Test scan
    T                                   aggregate;
    BlockScanT                          block_scan(temp_storage);
    BlockPrefixCallbackOp<T, ScanOp>    prefix_op(linear_tid, prefix, scan_op);

    DeviceTest(block_scan, data, identity, scan_op, aggregate, prefix_op, Int2Type<TEST_MODE>());

    // Stop cycle timer
#if CUB_PTX_ARCH > 100
    clock_t stop = clock();
#endif

    // Store output
    StoreDirectBlocked(linear_tid, d_out, data);

    if (TEST_MODE != BASIC)
    {
        // Store aggregate
        d_aggregate[linear_tid] = aggregate;
    }

    if (linear_tid == 0)
    {
        if (TEST_MODE == PREFIX_AGGREGATE)
        {
            // Store prefix
            d_out[TILE_SIZE] = prefix_op.prefix;
        }

        // Store time
#if CUB_PTX_ARCH > 100
        *d_elapsed = (start > stop) ? start - stop : stop - start;
#endif
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
 * Test threadblock scan.  (Specialized for sufficient resources)
 */
template <
    int                 BLOCK_DIM_X,
    int                 BLOCK_DIM_Y,
    int                 BLOCK_DIM_Z,
    int                 ITEMS_PER_THREAD,
    TestMode            TEST_MODE,
    BlockScanAlgorithm  ALGORITHM,
    typename            ScanOp,
    typename            IdentityT,        // NullType implies inclusive-scan, otherwise inclusive scan
    typename            T>
void Test(
    GenMode             gen_mode,
    ScanOp              scan_op,
    IdentityT           identity,
    T                   prefix,
    Int2Type<true>      sufficient_resources)
{
    const int BLOCK_THREADS     = BLOCK_DIM_X * BLOCK_DIM_Y * BLOCK_DIM_Z;
    const int TILE_SIZE         = BLOCK_THREADS * ITEMS_PER_THREAD;

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
    printf("Test-mode %d, gen-mode %d, policy %d, %s BlockScan, %d (%d,%d,%d) threadblock threads, %d items per thread, %d tile size, %s (%d bytes) elements:\n",
        TEST_MODE, gen_mode, ALGORITHM,
        (Equals<IdentityT, NullType>::VALUE) ? "Inclusive" : "Exclusive",
        BLOCK_THREADS, BLOCK_DIM_X, BLOCK_DIM_Y, BLOCK_DIM_Z,
        ITEMS_PER_THREAD,  TILE_SIZE,
        typeid(T).name(), (int) sizeof(T));
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
    dim3 block_dims(BLOCK_DIM_X, BLOCK_DIM_Y, BLOCK_DIM_Z);
    BlockScanKernel<BLOCK_DIM_X, BLOCK_DIM_Y, BLOCK_DIM_Z, ITEMS_PER_THREAD, TEST_MODE, ALGORITHM><<<1, block_dims>>>(
        d_in,
        d_out,
        d_aggregate,
        scan_op,
        identity,
        prefix,
        d_elapsed);

    CubDebugExit(cudaPeekAtLastError());
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
 * Test threadblock scan.  (Specialized for insufficient resources)
 */
template <
    int                 BLOCK_DIM_X,
    int                 BLOCK_DIM_Y,
    int                 BLOCK_DIM_Z,
    int                 ITEMS_PER_THREAD,
    TestMode            TEST_MODE,
    BlockScanAlgorithm  ALGORITHM,
    typename            ScanOp,
    typename            IdentityT,        // NullType implies inclusive-scan, otherwise inclusive scan
    typename            T>
void Test(
    GenMode             gen_mode,
    ScanOp              scan_op,
    IdentityT           identity,
    T                   prefix,
    Int2Type<false>     sufficient_resources)
{}


/**
 * Test threadblock scan.
 */
template <
    int                 BLOCK_DIM_X,
    int                 BLOCK_DIM_Y,
    int                 BLOCK_DIM_Z,
    int                 ITEMS_PER_THREAD,
    TestMode            TEST_MODE,
    BlockScanAlgorithm  ALGORITHM,
    typename            ScanOp,
    typename            IdentityT,        // NullType implies inclusive-scan, otherwise inclusive scan
    typename            T>
void Test(
    GenMode             gen_mode,
    ScanOp              scan_op,
    IdentityT           identity,
    T                   prefix)
{
    // Check size of smem storage for the target arch to make sure it will fit
    typedef BlockScan<T, BLOCK_DIM_X, ALGORITHM, BLOCK_DIM_Y, BLOCK_DIM_Z> BlockScanT;

    enum
    {
#if defined(SM100) || defined(SM110) || defined(SM130)
        sufficient_smem         = (sizeof(typename BlockScanT::TempStorage)     <= 16 * 1024),
        sufficient_threads      = ((BLOCK_DIM_X * BLOCK_DIM_Y * BLOCK_DIM_Z)    <= 512),
#else
        sufficient_smem         = (sizeof(typename BlockScanT::TempStorage)     <= 16 * 1024),
        sufficient_threads      = ((BLOCK_DIM_X * BLOCK_DIM_Y * BLOCK_DIM_Z)    <= 1024),
#endif

#if defined(_WIN32) || defined(_WIN64)
        // Accommodate ptxas crash bug (access violation) on Windows
        special_skip            = ((TEST_ARCH <= 130) && (Equals<T, TestBar>::VALUE) && (BLOCK_DIM_Z > 1)),
#else
        special_skip            = false,
#endif
        sufficient_resources    = (sufficient_smem && sufficient_threads && !special_skip),
    };

    Test<BLOCK_DIM_X, BLOCK_DIM_Y, BLOCK_DIM_Z, ITEMS_PER_THREAD, TEST_MODE, ALGORITHM>(gen_mode, scan_op, identity, prefix, Int2Type<sufficient_resources>());
}



/**
 * Run test for different threadblock dimensions
 */
template <
    int                 BLOCK_THREADS,
    int                 ITEMS_PER_THREAD,
    TestMode            TEST_MODE,
    BlockScanAlgorithm  ALGORITHM,
    typename            ScanOp,
    typename            IdentityT,
    typename            T>
void Test(
    GenMode     gen_mode,
    ScanOp      scan_op,
    IdentityT   identity,
    T           prefix)
{
    Test<BLOCK_THREADS, 1, 1, ITEMS_PER_THREAD, TEST_MODE, ALGORITHM>(gen_mode, scan_op, identity, prefix);
    Test<BLOCK_THREADS, 2, 2, ITEMS_PER_THREAD, TEST_MODE, ALGORITHM>(gen_mode, scan_op, identity, prefix);
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
    T           prefix)
{
#ifdef TEST_RAKING
    Test<BLOCK_THREADS, ITEMS_PER_THREAD, TEST_MODE, BLOCK_SCAN_RAKING>(gen_mode, scan_op, identity, prefix);
#endif
#ifdef TEST_RAKING_MEMOIZE
    Test<BLOCK_THREADS, ITEMS_PER_THREAD, TEST_MODE, BLOCK_SCAN_RAKING_MEMOIZE>(gen_mode, scan_op, identity, prefix);
#endif
#ifdef TEST_WARP_SCANS
    Test<BLOCK_THREADS, ITEMS_PER_THREAD, TEST_MODE, BLOCK_SCAN_WARP_SCANS>(gen_mode, scan_op, identity, prefix);
#endif
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
    T           prefix)
{
    // Exclusive
    Test<BLOCK_THREADS, ITEMS_PER_THREAD, BASIC>(gen_mode, scan_op, identity, prefix);
    Test<BLOCK_THREADS, ITEMS_PER_THREAD, AGGREGATE>(gen_mode, scan_op, identity, prefix);
    Test<BLOCK_THREADS, ITEMS_PER_THREAD, PREFIX_AGGREGATE>(gen_mode, scan_op, identity, prefix);

    // Inclusive
    Test<BLOCK_THREADS, ITEMS_PER_THREAD, BASIC>(gen_mode, scan_op, NullType(), prefix);
    Test<BLOCK_THREADS, ITEMS_PER_THREAD, AGGREGATE>(gen_mode, scan_op, NullType(), prefix);
    Test<BLOCK_THREADS, ITEMS_PER_THREAD, PREFIX_AGGREGATE>(gen_mode, scan_op, NullType(), prefix);
}


/**
 * Run tests for different problem-generation options
 */
template <
    int         BLOCK_THREADS,
    int         ITEMS_PER_THREAD,
    typename    ScanOp,
    typename    T>
void Test(
    ScanOp      scan_op,
    T           identity,
    T           prefix)
{
    Test<BLOCK_THREADS, ITEMS_PER_THREAD>(UNIFORM, scan_op, identity, prefix);
    Test<BLOCK_THREADS, ITEMS_PER_THREAD>(INTEGER_SEED, scan_op, identity, prefix);

    // Don't test randomly-generated floats b/c of stability
    if (Traits<T>::CATEGORY != FLOATING_POINT)
        Test<BLOCK_THREADS, ITEMS_PER_THREAD>(RANDOM, scan_op, identity, prefix);
}


/**
 * Run tests for different data types and scan ops
 */
template <
    int BLOCK_THREADS,
    int ITEMS_PER_THREAD>
void Test()
{
    // Get ptx version
    int ptx_version;
    CubDebugExit(PtxVersion(ptx_version));

    // primitive
    Test<BLOCK_THREADS, ITEMS_PER_THREAD>(Sum(), (unsigned char) 0, (unsigned char) 99);
    Test<BLOCK_THREADS, ITEMS_PER_THREAD>(Sum(), (unsigned short) 0, (unsigned short) 99);
    Test<BLOCK_THREADS, ITEMS_PER_THREAD>(Sum(), (unsigned int) 0, (unsigned int) 99);
    Test<BLOCK_THREADS, ITEMS_PER_THREAD>(Sum(), (unsigned long long) 0, (unsigned long long) 99);
    Test<BLOCK_THREADS, ITEMS_PER_THREAD>(Sum(), (float) 0, (float) 99);

    // primitive (alternative scan op)
    Test<BLOCK_THREADS, ITEMS_PER_THREAD>(Max(), std::numeric_limits<char>::min(), (char) 99);
    Test<BLOCK_THREADS, ITEMS_PER_THREAD>(Max(), std::numeric_limits<short>::min(), (short) 99);
    Test<BLOCK_THREADS, ITEMS_PER_THREAD>(Max(), std::numeric_limits<int>::min(), (int) 99);
    Test<BLOCK_THREADS, ITEMS_PER_THREAD>(Max(), std::numeric_limits<long long>::min(), (long long) 99);

    if (ptx_version > 120)                          // Don't check doubles on PTX120 or below because they're down-converted
        Test<BLOCK_THREADS, ITEMS_PER_THREAD>(Max(), std::numeric_limits<double>::max() * -1, (double) 99);

    // vec-1
    Test<BLOCK_THREADS, ITEMS_PER_THREAD>(Sum(), make_uchar1(0), make_uchar1(17));

    // vec-2
    Test<BLOCK_THREADS, ITEMS_PER_THREAD>(Sum(), make_uchar2(0, 0), make_uchar2(17, 21));
    Test<BLOCK_THREADS, ITEMS_PER_THREAD>(Sum(), make_ushort2(0, 0), make_ushort2(17, 21));
    Test<BLOCK_THREADS, ITEMS_PER_THREAD>(Sum(), make_uint2(0, 0), make_uint2(17, 21));
    Test<BLOCK_THREADS, ITEMS_PER_THREAD>(Sum(), make_ulonglong2(0, 0), make_ulonglong2(17, 21));

    // vec-4
    Test<BLOCK_THREADS, ITEMS_PER_THREAD>(Sum(), make_char4(0, 0, 0, 0), make_char4(17, 21, 32, 85));
    Test<BLOCK_THREADS, ITEMS_PER_THREAD>(Sum(), make_short4(0, 0, 0, 0), make_short4(17, 21, 32, 85));
    Test<BLOCK_THREADS, ITEMS_PER_THREAD>(Sum(), make_int4(0, 0, 0, 0), make_int4(17, 21, 32, 85));
    Test<BLOCK_THREADS, ITEMS_PER_THREAD>(Sum(), make_longlong4(0, 0, 0, 0), make_longlong4(17, 21, 32, 85));

    // complex
    Test<BLOCK_THREADS, ITEMS_PER_THREAD>(Sum(), TestFoo::MakeTestFoo(0, 0, 0, 0), TestFoo::MakeTestFoo(17, 21, 32, 85));
    Test<BLOCK_THREADS, ITEMS_PER_THREAD>(Sum(), TestBar(0, 0), TestBar(17, 21));
}


/**
 * Run tests for different items per thread
 */
template <int BLOCK_THREADS>
void Test()
{
    Test<BLOCK_THREADS, 1>();
    Test<BLOCK_THREADS, 2>();
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
    Test<128, 1, 1, 1, AGGREGATE, BLOCK_SCAN_WARP_SCANS>(UNIFORM, Sum(), int(0), int(10));
    Test<128, 1, 1, 4, AGGREGATE, BLOCK_SCAN_RAKING_MEMOIZE>(UNIFORM, Sum(), int(0), int(10));
    Test<128, 1, 1, 2, PREFIX_AGGREGATE, BLOCK_SCAN_RAKING>(INTEGER_SEED, Sum(), NullType(), TestFoo::MakeTestFoo(17, 21, 32, 85));
    Test<128, 1, 1, 1, AGGREGATE, BLOCK_SCAN_WARP_SCANS>(UNIFORM, Sum(), make_longlong4(0, 0, 0, 0), make_longlong4(17, 21, 32, 85));

#else

    // Compile/run thorough tests
    for (int i = 0; i <= g_repeat; ++i)
    {
        // Run tests for different threadblock sizes
        Test<17>();
        Test<32>();
        Test<62>();
        Test<65>();
//            Test<96>();             // TODO: file bug for UNREACHABLE error for Test<96, 9, BASIC, BLOCK_SCAN_RAKING>(UNIFORM, Sum(), NullType(), make_ulonglong2(17, 21));
        Test<128>();
    }

#endif

    return 0;
}




