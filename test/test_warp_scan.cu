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
 * Test of WarpScan utilities
 ******************************************************************************/

// Ensure printing of CUDA runtime errors to console
#define CUB_STDERR

#include <stdio.h>
#include <typeinfo>

#include <cub/warp/warp_scan.cuh>
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
};



//---------------------------------------------------------------------
// Test kernels
//---------------------------------------------------------------------

/**
 * Exclusive scan
 */
template <
    typename    T,
    typename    ScanOp,
    typename    IdentityT>
struct DeviceTest
{
    template <
        TestMode TEST_MODE,
        typename WarpScan>
    static __device__ __forceinline__ void Test(
        typename WarpScan::TempStorage  &temp_storage,
        T                               &data,
        IdentityT                       &identity,
        ScanOp                          &scan_op,
        T                               &aggregate)
    {
        if (TEST_MODE == BASIC)
        {
            // Test basic warp scan
            WarpScan(temp_storage).ExclusiveScan(data, data, identity, scan_op);
        }
        else if (TEST_MODE == AGGREGATE)
        {
            // Test with cumulative aggregate
            WarpScan(temp_storage).ExclusiveScan(data, data, identity, scan_op, aggregate);
        }
    }
};


/**
 * Exclusive sum
 */
template <
    typename T,
    typename IdentityT>
struct DeviceTest<T, Sum, IdentityT>
{
    template <
        TestMode TEST_MODE,
        typename WarpScan>
    static __device__ __forceinline__ void Test(
        typename WarpScan::TempStorage  &temp_storage,
        T                               &data,
        T                               &identity,
        Sum                             &scan_op,
        T                               &aggregate)
    {
        if (TEST_MODE == BASIC)
        {
            // Test basic warp scan
            WarpScan(temp_storage).ExclusiveSum(data, data);
        }
        else if (TEST_MODE == AGGREGATE)
        {
            // Test with cumulative aggregate
            WarpScan(temp_storage).ExclusiveSum(data, data, aggregate);
        }
    }
};


/**
 * Inclusive scan
 */
template <
    typename    T,
    typename    ScanOp>
struct DeviceTest<T, ScanOp, NullType>
{
    template <
        TestMode TEST_MODE,
        typename WarpScan>
    static __device__ __forceinline__ void Test(
        typename WarpScan::TempStorage  &temp_storage,
        T                               &data,
        NullType                        &identity,
        ScanOp                          &scan_op,
        T                               &aggregate)
    {
        if (TEST_MODE == BASIC)
        {
            // Test basic warp scan
            WarpScan(temp_storage).InclusiveScan(data, data, scan_op);
        }
        else if (TEST_MODE == AGGREGATE)
        {
            // Test with cumulative aggregate
            WarpScan(temp_storage).InclusiveScan(data, data, scan_op, aggregate);
        }
    }
};


/**
 * Inclusive sum
 */
template <typename T>
struct DeviceTest<T, Sum, NullType>
{
    template <
        TestMode TEST_MODE,
        typename WarpScan>
    static __device__ __forceinline__ void Test(
        typename WarpScan::TempStorage  &temp_storage,
        T                               &data,
        NullType                        &identity,
        Sum                             &scan_op,
        T                               &aggregate)
    {
        if (TEST_MODE == BASIC)
        {
            // Test basic warp scan
            WarpScan(temp_storage).InclusiveSum(data, data);
        }
        else if (TEST_MODE == AGGREGATE)
        {
            // Test with cumulative aggregate
            WarpScan(temp_storage).InclusiveSum(data, data, aggregate);
        }
    }
};



/**
 * WarpScan test kernel
 */
template <
    int         LOGICAL_WARP_THREADS,
    TestMode    TEST_MODE,
    typename    T,
    typename    ScanOp,
    typename    IdentityT>
__global__ void WarpScanKernel(
    T           *d_in,
    T           *d_out,
    T           *d_aggregate,
    ScanOp      scan_op,
    IdentityT   identity,
    T           prefix,
    clock_t     *d_elapsed)
{
    // Cooperative warp-scan utility type (1 warp)
    typedef WarpScan<T, LOGICAL_WARP_THREADS> WarpScan;

    // Allocate temp storage in shared memory
    __shared__ typename WarpScan::TempStorage temp_storage;

    // Per-thread tile data
    T data = d_in[threadIdx.x];

    // Start cycle timer
    clock_t start = clock();

    T aggregate;

    // Test scan
    DeviceTest<T, ScanOp, IdentityT>::template Test<TEST_MODE, WarpScan>(
        temp_storage, data, identity, scan_op, aggregate);

    // Stop cycle timer
    clock_t stop = clock();

    // Store data
    d_out[threadIdx.x] = data;

    if (TEST_MODE != BASIC)
    {
        // Store aggregate
        d_aggregate[threadIdx.x] = aggregate;
    }

    // Store prefix and time
    if (threadIdx.x == 0)
    {
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
 * Test warp scan
 */
template <
    int         LOGICAL_WARP_THREADS,
    TestMode    TEST_MODE,
    typename    ScanOp,
    typename    IdentityT,        // NullType implies inclusive-scan, otherwise inclusive scan
    typename    T>
void Test(
    GenMode     gen_mode,
    ScanOp      scan_op,
    IdentityT   identity,
    T           prefix)
{
    // Allocate host arrays
    T *h_in = new T[LOGICAL_WARP_THREADS];
    T *h_reference = new T[LOGICAL_WARP_THREADS];
    T *h_aggregate = new T[LOGICAL_WARP_THREADS];

    // Initialize problem
    T *p_prefix = NULL;
    T aggregate = Initialize(gen_mode, h_in, h_reference, LOGICAL_WARP_THREADS, scan_op, identity, p_prefix);

    if (g_verbose)
    {
        printf("Input: \n");
        DisplayResults(h_in, LOGICAL_WARP_THREADS);
        printf("\n");
    }

    for (int i = 0; i < LOGICAL_WARP_THREADS; ++i)
    {
        h_aggregate[i] = aggregate;
    }

    // Initialize/clear device arrays
    T *d_in = NULL;
    T *d_out = NULL;
    T *d_aggregate = NULL;
    clock_t *d_elapsed = NULL;
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_in, sizeof(T) * LOGICAL_WARP_THREADS));
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_out, sizeof(T) * (LOGICAL_WARP_THREADS + 1)));
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_aggregate, sizeof(T) * LOGICAL_WARP_THREADS));
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_elapsed, sizeof(clock_t)));
    CubDebugExit(cudaMemcpy(d_in, h_in, sizeof(T) * LOGICAL_WARP_THREADS, cudaMemcpyHostToDevice));
    CubDebugExit(cudaMemset(d_out, 0, sizeof(T) * (LOGICAL_WARP_THREADS + 1)));
    CubDebugExit(cudaMemset(d_aggregate, 0, sizeof(T) * LOGICAL_WARP_THREADS));

    // Run kernel
    printf("Test-mode %d (%s), gen-mode %d (%s), %s warpscan, %d warp threads, %s (%d bytes) elements:\n",
        TEST_MODE, typeid(TEST_MODE).name(),
        gen_mode, typeid(gen_mode).name(),
        (Equals<IdentityT, NullType>::VALUE) ? "Inclusive" : "Exclusive",
        LOGICAL_WARP_THREADS,
        typeid(T).name(),
        (int) sizeof(T));
    fflush(stdout);

    // Run aggregate/prefix kernel
    WarpScanKernel<LOGICAL_WARP_THREADS, TEST_MODE><<<1, LOGICAL_WARP_THREADS>>>(
        d_in,
        d_out,
        d_aggregate,
        scan_op,
        identity,
        prefix,
        d_elapsed);

    printf("\tElapsed clocks: ");
    DisplayDeviceResults(d_elapsed, 1);

    CubDebugExit(cudaPeekAtLastError());
    CubDebugExit(cudaDeviceSynchronize());

    // Copy out and display results
    printf("\tScan results: ");
    int compare = CompareDeviceResults(h_reference, d_out, LOGICAL_WARP_THREADS, g_verbose, g_verbose);
    printf("%s\n", compare ? "FAIL" : "PASS");
    AssertEquals(0, compare);

    // Copy out and display aggregate
    if (TEST_MODE == AGGREGATE)
    {
        printf("\tScan aggregate: ");
        compare = CompareDeviceResults(h_aggregate, d_aggregate, LOGICAL_WARP_THREADS, g_verbose, g_verbose);
        printf("%s\n", compare ? "FAIL" : "PASS");
        AssertEquals(0, compare);
    }

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
 * Run battery of tests for different primitive variants
 */
template <
    int         LOGICAL_WARP_THREADS,
    typename    ScanOp,
    typename    T>
void Test(
    GenMode     gen_mode,
    ScanOp      scan_op,
    T           identity,
    T           prefix)
{
    // Exclusive
    Test<LOGICAL_WARP_THREADS, BASIC>(gen_mode, scan_op, identity, prefix);
    Test<LOGICAL_WARP_THREADS, AGGREGATE>(gen_mode, scan_op, identity, prefix);

    // Inclusive
    Test<LOGICAL_WARP_THREADS, BASIC>(gen_mode, scan_op, NullType(), prefix);
    Test<LOGICAL_WARP_THREADS, AGGREGATE>(gen_mode, scan_op, NullType(), prefix);
}


/**
 * Run battery of tests for different data types and scan ops
 */
template <int LOGICAL_WARP_THREADS>
void Test(GenMode gen_mode)
{
    // Get device ordinal
    int device_ordinal;
    CubDebugExit(cudaGetDevice(&device_ordinal));

    // Get ptx version
    int ptx_version;
    CubDebugExit(PtxVersion(ptx_version));

    // primitive
    Test<LOGICAL_WARP_THREADS>(gen_mode, Sum(), (char) 0, (char) 99);
    Test<LOGICAL_WARP_THREADS>(gen_mode, Sum(), (short) 0, (short) 99);
    Test<LOGICAL_WARP_THREADS>(gen_mode, Sum(), (int) 0, (int) 99);
    Test<LOGICAL_WARP_THREADS>(gen_mode, Sum(), (long) 0, (long) 99);
    Test<LOGICAL_WARP_THREADS>(gen_mode, Sum(), (long long) 0, (long long) 99);
    if (gen_mode != RANDOM) {
        // Only test numerically stable inputs
        Test<LOGICAL_WARP_THREADS>(gen_mode, Sum(), (float) 0, (float) 99);
        if (ptx_version > 100)
            Test<LOGICAL_WARP_THREADS>(gen_mode, Sum(), (double) 0, (double) 99);
    }

    // primitive (alternative scan op)
    Test<LOGICAL_WARP_THREADS>(gen_mode, Max(), (unsigned char) 0, (unsigned char) 99);
    Test<LOGICAL_WARP_THREADS>(gen_mode, Max(), (unsigned short) 0, (unsigned short) 99);
    Test<LOGICAL_WARP_THREADS>(gen_mode, Max(), (unsigned int) 0, (unsigned int) 99);
    Test<LOGICAL_WARP_THREADS>(gen_mode, Max(), (unsigned long long) 0, (unsigned long long) 99);

    // vec-2
    Test<LOGICAL_WARP_THREADS>(gen_mode, Sum(), make_uchar2(0, 0), make_uchar2(17, 21));
    Test<LOGICAL_WARP_THREADS>(gen_mode, Sum(), make_ushort2(0, 0), make_ushort2(17, 21));
    Test<LOGICAL_WARP_THREADS>(gen_mode, Sum(), make_uint2(0, 0), make_uint2(17, 21));
    Test<LOGICAL_WARP_THREADS>(gen_mode, Sum(), make_ulong2(0, 0), make_ulong2(17, 21));
    Test<LOGICAL_WARP_THREADS>(gen_mode, Sum(), make_ulonglong2(0, 0), make_ulonglong2(17, 21));
    if (gen_mode != RANDOM) {
        // Only test numerically stable inputs
        Test<LOGICAL_WARP_THREADS>(gen_mode, Sum(), make_float2(0, 0), make_float2(17, 21));
        if (ptx_version > 100)
            Test<LOGICAL_WARP_THREADS>(gen_mode, Sum(), make_double2(0, 0), make_double2(17, 21));
    }

    // vec-4
    Test<LOGICAL_WARP_THREADS>(gen_mode, Sum(), make_char4(0, 0, 0, 0), make_char4(17, 21, 32, 85));
    Test<LOGICAL_WARP_THREADS>(gen_mode, Sum(), make_short4(0, 0, 0, 0), make_short4(17, 21, 32, 85));
    Test<LOGICAL_WARP_THREADS>(gen_mode, Sum(), make_int4(0, 0, 0, 0), make_int4(17, 21, 32, 85));
    Test<LOGICAL_WARP_THREADS>(gen_mode, Sum(), make_long4(0, 0, 0, 0), make_long4(17, 21, 32, 85));
    Test<LOGICAL_WARP_THREADS>(gen_mode, Sum(), make_longlong4(0, 0, 0, 0), make_longlong4(17, 21, 32, 85));
    if (gen_mode != RANDOM) {
        // Only test numerically stable inputs
        Test<LOGICAL_WARP_THREADS>(gen_mode, Sum(), make_float4(0, 0, 0, 0), make_float4(17, 21, 32, 85));
        if (ptx_version > 100)
            Test<LOGICAL_WARP_THREADS>(gen_mode, Sum(), make_double4(0, 0, 0, 0), make_double4(17, 21, 32, 85));
    }

    // complex
    Test<LOGICAL_WARP_THREADS>(gen_mode, Sum(), TestFoo::MakeTestFoo(0, 0, 0, 0), TestFoo::MakeTestFoo(17, 21, 32, 85));
    Test<LOGICAL_WARP_THREADS>(gen_mode, Sum(), TestBar(0, 0), TestBar(17, 21));
}


/**
 * Run battery of tests for different problem generation options
 */
template <int LOGICAL_WARP_THREADS>
void Test()
{
    Test<LOGICAL_WARP_THREADS>(UNIFORM);
    Test<LOGICAL_WARP_THREADS>(INTEGER_SEED);
    Test<LOGICAL_WARP_THREADS>(RANDOM);
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
    Test<32, AGGREGATE>(UNIFORM, Sum(), (int) 0, (int) 99);
    Test<32, AGGREGATE>(UNIFORM, Sum(), (float) 0, (float) 99);
    Test<32, AGGREGATE>(UNIFORM, Sum(), (long long) 0, (long long) 99);
    Test<32, AGGREGATE>(UNIFORM, Sum(), (double) 0, (double) 99);

    typedef KeyValuePair<int, float> T;
    cub::Sum sum_op;
    Test<32, AGGREGATE>(UNIFORM, ReduceBySegmentOp<cub::Sum>(sum_op), ZeroInitialize<T>(), ZeroInitialize<T>());

#else

    // Compile/run thorough tests
    for (int i = 0; i <= g_repeat; ++i)
    {
        // Test logical warp sizes
        Test<32>();
        Test<16>();
        Test<9>();
        Test<7>();
    }

#endif

    return 0;
}




