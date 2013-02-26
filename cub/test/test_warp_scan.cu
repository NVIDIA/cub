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
 * Test of WarpScan utilities
 ******************************************************************************/

// Ensure printing of CUDA runtime errors to console
#define CUB_STDERR

#include <stdio.h>
#include <iostream>
#include <test_util.h>
#include "../cub.cuh"

using namespace cub;

//---------------------------------------------------------------------
// Globals, constants and typedefs
//---------------------------------------------------------------------

/**
 * Verbose output
 */
bool g_verbose = false;


/**
 * Primitive variant to test
 */
enum TestMode
{
    BASIC,
    AGGREGATE,
    PREFIX_AGGREGATE,
};



//---------------------------------------------------------------------
// Test kernels
//---------------------------------------------------------------------

// Stateful prefix functor
template <
    typename T,
    typename ScanOp>
struct WarpPrefixOp
{
    T       prefix;
    ScanOp  scan_op;

    __device__ __forceinline__
    WarpPrefixOp(T prefix, ScanOp scan_op) : prefix(prefix), scan_op(scan_op) {}

    __device__ __forceinline__
    T operator()(T block_aggregate)
    {
        T retval = prefix;
        prefix = scan_op(prefix, block_aggregate);
        return retval;
    }
};

/**
 * Exclusive WarpScan test kernel.
 */
template <
    int         LOGICAL_WARP_THREADS,
    TestMode    TEST_MODE,
    typename     T,
    typename     ScanOp,
    typename     IdentityT>
__global__ void WarpScanKernel(
    T             *d_in,
    T             *d_out,
    ScanOp         scan_op,
    IdentityT     identity,
    T            prefix,
    clock_t        *d_elapsed)
{
    // Cooperative warp-scan utility type (1 warp)
    typedef WarpScan<T, 1, LOGICAL_WARP_THREADS> WarpScan;

    // Shared memory
    __shared__ typename WarpScan::SmemStorage smem_storage;

    // Per-thread tile data
    T data = d_in[threadIdx.x];

    // Record elapsed clocks
    clock_t start = clock();

    // Test scan
    T aggregate;
    WarpPrefixOp<T, ScanOp> prefix_op(prefix, scan_op);
    if (TEST_MODE == BASIC)
    {
        // Test basic warp scan
        WarpScan::ExclusiveScan(smem_storage, data, data, identity, scan_op);
    }
    else if (TEST_MODE == AGGREGATE)
    {
        // Test with cumulative aggregate
        WarpScan::ExclusiveScan(smem_storage, data, data, identity, scan_op, aggregate);
    }
    else if (TEST_MODE == PREFIX_AGGREGATE)
    {
        // Test with warp-prefix and cumulative aggregate
        WarpScan::ExclusiveScan(smem_storage, data, data, identity, scan_op, aggregate, prefix_op);
    }

    // Record elapsed clocks
    *d_elapsed = clock() - start;

    // Store data
    d_out[threadIdx.x] = data;

    // Store aggregate and prefix
    if (threadIdx.x == 0)
    {
        d_out[LOGICAL_WARP_THREADS] = aggregate;
        d_out[LOGICAL_WARP_THREADS + 1] = prefix_op.prefix;
    }
}


/**
 * Exclusive WarpScan test kernel (specialized for prefix sum)
 */
template <
    int         LOGICAL_WARP_THREADS,
    TestMode    TEST_MODE,
    typename     T,
    typename     IdentityT>
__global__ void WarpScanKernel(
    T                                                 *d_in,
    T                                                 *d_out,
    Sum<T>,
    IdentityT,
    T                                                prefix,
    clock_t                                            *d_elapsed,
    typename EnableIf<Traits<T>::PRIMITIVE>::Type     *dummy = NULL)
{
    // Cooperative warp-scan utility type (1 warp)
    typedef WarpScan<T, 1, LOGICAL_WARP_THREADS> WarpScan;

    // Shared memory
    __shared__ typename WarpScan::SmemStorage smem_storage;

    // Per-thread tile data
    T data = d_in[threadIdx.x];

    // Record elapsed clocks
    clock_t start = clock();

    // Test scan
    T aggregate;
    WarpPrefixOp<T, Sum<T> > prefix_op(prefix, Sum<T>());
    if (TEST_MODE == BASIC)
    {
        // Test basic warp scan
        WarpScan::ExclusiveSum(smem_storage, data, data);
    }
    else if (TEST_MODE == AGGREGATE)
    {
        // Test with cumulative aggregate
        WarpScan::ExclusiveSum(smem_storage, data, data, aggregate);
    }
    else if (TEST_MODE == PREFIX_AGGREGATE)
    {
        // Test with warp-prefix and cumulative aggregate
        WarpScan::ExclusiveSum(smem_storage, data, data, aggregate, prefix_op);
    }

    // Record elapsed clocks
    *d_elapsed = clock() - start;

    // Store data
    d_out[threadIdx.x] = data;

    // Store aggregate and prefix
    if (threadIdx.x == 0)
    {
        d_out[LOGICAL_WARP_THREADS] = aggregate;
        d_out[LOGICAL_WARP_THREADS + 1] = prefix_op.prefix;
    }
}


/**
 * Inclusive WarpScan test kernel.
 */
template <
    int         LOGICAL_WARP_THREADS,
    TestMode    TEST_MODE,
    typename     T,
    typename     ScanOp>
__global__ void WarpScanKernel(
    T             *d_in,
    T             *d_out,
    ScanOp         scan_op,
    NullType,
    T            prefix,
    clock_t        *d_elapsed)
{
    // Cooperative warp-scan utility type (1 warp)
    typedef WarpScan<T, 1, LOGICAL_WARP_THREADS> WarpScan;

    // Shared memory
    __shared__ typename WarpScan::SmemStorage smem_storage;

    // Per-thread tile data
    T data = d_in[threadIdx.x];

    // Record elapsed clocks
    clock_t start = clock();

    T aggregate;
    WarpPrefixOp<T, ScanOp> prefix_op(prefix, scan_op);
    if (TEST_MODE == BASIC)
    {
        // Test basic warp scan
        WarpScan::InclusiveScan(smem_storage, data, data, scan_op);
    }
    else if (TEST_MODE == AGGREGATE)
    {
        // Test with cumulative aggregate
        WarpScan::InclusiveScan(smem_storage, data, data, scan_op, aggregate);
    }
    else if (TEST_MODE == PREFIX_AGGREGATE)
    {
        // Test with warp-prefix and cumulative aggregate
        WarpScan::InclusiveScan(smem_storage, data, data, scan_op, aggregate, prefix_op);
    }

    // Record elapsed clocks
    *d_elapsed = clock() - start;

    // Store data
    d_out[threadIdx.x] = data;

    // Store aggregate and prefix
    if (threadIdx.x == 0)
    {
        d_out[LOGICAL_WARP_THREADS] = aggregate;
        d_out[LOGICAL_WARP_THREADS + 1] = prefix_op.prefix;
    }
}


/**
 * Inclusive WarpScan test kernel (specialized for prefix sum).
 */
template <
    int         LOGICAL_WARP_THREADS,
    TestMode    TEST_MODE,
    typename     T>
__global__ void WarpScanKernel(
    T                                                 *d_in,
    T                                                 *d_out,
    Sum<T>,
    NullType,
    T                                                prefix,
    clock_t                                            *d_elapsed,
    typename EnableIf<Traits<T>::PRIMITIVE>::Type     *dummy = NULL)

{
    // Cooperative warp-scan utility type (1 warp)
    typedef WarpScan<T, 1, LOGICAL_WARP_THREADS> WarpScan;

    // Shared memory
    __shared__ typename WarpScan::SmemStorage smem_storage;

    // Per-thread tile data
    T data = d_in[threadIdx.x];

    // Record elapsed clocks
    clock_t start = clock();

    T aggregate;
    WarpPrefixOp<T, Sum<T> > prefix_op(prefix, Sum<T>() );
    if (TEST_MODE == BASIC)
    {
        // Test basic warp scan
        WarpScan::InclusiveSum(smem_storage, data, data);
    }
    else if (TEST_MODE == AGGREGATE)
    {
        // Test with cumulative aggregate
        WarpScan::InclusiveSum(smem_storage, data, data, aggregate);
    }
    else if (TEST_MODE == PREFIX_AGGREGATE)
    {
        // Test with warp-prefix and cumulative aggregate
        WarpScan::InclusiveSum(smem_storage, data, data, aggregate, prefix_op);
    }

    // Record elapsed clocks
    *d_elapsed = clock() - start;

    // Store data
    d_out[threadIdx.x] = data;

    // Store aggregate and prefix
    if (threadIdx.x == 0)
    {
        d_out[LOGICAL_WARP_THREADS] = aggregate;
        d_out[LOGICAL_WARP_THREADS + 1] = prefix_op.prefix;
    }
}


//---------------------------------------------------------------------
// Host utility subroutines
//---------------------------------------------------------------------

/**
 * Initialize exclusive-scan problem (and solution)
 */
template <
    typename     T,
    typename     ScanOp,
    typename     IdentityT>
T Initialize(
    int             gen_mode,
    T             *h_in,
    T             *h_reference,
    int         num_elements,
    ScanOp         scan_op,
    IdentityT     identity,
    T            *prefix)
{
    T inclusive = (prefix != NULL) ? *prefix : identity;
    T aggregate = identity;

    for (int i = 0; i < num_elements; ++i)
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
    typename     T,
    typename     ScanOp>
T Initialize(
    int             gen_mode,
    T             *h_in,
    T             *h_reference,
    int         num_elements,
    ScanOp         scan_op,
    NullType,
    T            *prefix)
{
    T inclusive;
    T aggregate;
    for (int i = 0; i < num_elements; ++i)
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
    TestMode     TEST_MODE,
    typename     ScanOp,
    typename     IdentityT,        // NullType implies inclusive-scan, otherwise inclusive scan
    typename     T>
void Test(
    int         gen_mode,
    ScanOp         scan_op,
    IdentityT     identity,
    T            prefix,
    char        *type_string)
{
    // Allocate host arrays
    T *h_in = new T[LOGICAL_WARP_THREADS];
    T *h_reference = new T[LOGICAL_WARP_THREADS];

    // Initialize problem
    T *p_prefix = (TEST_MODE == PREFIX_AGGREGATE) ? &prefix : NULL;
    T aggregate = Initialize(gen_mode, h_in, h_reference, LOGICAL_WARP_THREADS, scan_op, identity, p_prefix);

    // Initialize device arrays
    T *d_in = NULL;
    T *d_out = NULL;
    clock_t *d_elapsed = NULL;
    CubDebugExit(cudaMalloc((void**)&d_in, sizeof(T) * LOGICAL_WARP_THREADS));
    CubDebugExit(cudaMalloc((void**)&d_out, sizeof(T) * (LOGICAL_WARP_THREADS + 2)));
    CubDebugExit(cudaMalloc((void**)&d_elapsed, sizeof(clock_t)));
    CubDebugExit(cudaMemcpy(d_in, h_in, sizeof(T) * LOGICAL_WARP_THREADS, cudaMemcpyHostToDevice));

    // Run kernel
    printf("Test-mode %d, gen-mode %d, %s warpscan, %d warp threads, %s (%d bytes) elements:\n",
        TEST_MODE,
        gen_mode,
        (Equals<IdentityT, NullType>::VALUE) ? "Inclusive" : "Exclusive",
        LOGICAL_WARP_THREADS,
        type_string,
        (int) sizeof(T));
    fflush(stdout);

    // Run aggregate/prefix kernel
    WarpScanKernel<LOGICAL_WARP_THREADS, TEST_MODE><<<1, LOGICAL_WARP_THREADS>>>(
        d_in,
        d_out,
        scan_op,
        identity,
        prefix,
        d_elapsed);

    printf("\tElapsed clocks: ");
    DisplayDeviceResults(d_elapsed, 1);

    CubDebugExit(cudaDeviceSynchronize());

    // Copy out and display results
    printf("\tScan results: ");
    AssertEquals(0, CompareDeviceResults(h_reference, d_out, LOGICAL_WARP_THREADS, g_verbose, g_verbose));
    printf("\n");

    // Copy out and display aggregate
    if ((TEST_MODE == AGGREGATE) || (TEST_MODE == PREFIX_AGGREGATE))
    {
        printf("\tScan aggregate: ");
        AssertEquals(0, CompareDeviceResults(&aggregate, d_out + LOGICAL_WARP_THREADS, 1, g_verbose, g_verbose));
        printf("\n");

        if (TEST_MODE == PREFIX_AGGREGATE)
        {
            printf("\tScan prefix: ");
            T new_prefix = scan_op(prefix, aggregate);
            AssertEquals(0, CompareDeviceResults(&new_prefix, d_out + LOGICAL_WARP_THREADS + 1, 1, g_verbose, g_verbose));
            printf("\n");
        }
    }

    // Cleanup
    if (h_in) delete[] h_in;
    if (h_reference) delete[] h_in;
    if (d_in) CubDebugExit(cudaFree(d_in));
    if (d_out) CubDebugExit(cudaFree(d_out));
}


/**
 * Run battery of tests for different primitive variants
 */
template <
    int         LOGICAL_WARP_THREADS,
    typename     ScanOp,
    typename     T>
void Test(
    int         gen_mode,
    ScanOp         scan_op,
    T             identity,
    T            prefix,
    char *        type_string)
{
    // Exclusive
    Test<LOGICAL_WARP_THREADS, BASIC>(gen_mode, scan_op, identity, prefix, type_string);
    Test<LOGICAL_WARP_THREADS, AGGREGATE>(gen_mode, scan_op, identity, prefix, type_string);
    Test<LOGICAL_WARP_THREADS, PREFIX_AGGREGATE>(gen_mode, scan_op, identity, prefix, type_string);

    // Inclusive
    Test<LOGICAL_WARP_THREADS, BASIC>(gen_mode, scan_op, NullType(), prefix, type_string);
    Test<LOGICAL_WARP_THREADS, AGGREGATE>(gen_mode, scan_op, NullType(), prefix, type_string);
    Test<LOGICAL_WARP_THREADS, PREFIX_AGGREGATE>(gen_mode, scan_op, NullType(), prefix, type_string);
}


/**
 * Run battery of tests for different data types and scan ops
 */
template <int LOGICAL_WARP_THREADS>
void Test(int gen_mode)
{
    // primitive
    Test<LOGICAL_WARP_THREADS>(gen_mode, Sum<unsigned char>(), (unsigned char) 0, (unsigned char) 99, CUB_TYPE_STRING(unsigned char));
    Test<LOGICAL_WARP_THREADS>(gen_mode, Sum<unsigned short>(), (unsigned short) 0, (unsigned short) 99, CUB_TYPE_STRING(unsigned short));
    Test<LOGICAL_WARP_THREADS>(gen_mode, Sum<unsigned int>(), (unsigned int) 0, (unsigned int) 99, CUB_TYPE_STRING(unsigned int));
    Test<LOGICAL_WARP_THREADS>(gen_mode, Sum<unsigned long long>(), (unsigned long long) 0, (unsigned long long) 99, CUB_TYPE_STRING(unsigned long long));

    // primitive (alternative scan op)
    Test<LOGICAL_WARP_THREADS>(gen_mode, Max<unsigned char>(), (unsigned char) 0, (unsigned char) 99, CUB_TYPE_STRING(unsigned char));
    Test<LOGICAL_WARP_THREADS>(gen_mode, Max<unsigned short>(), (unsigned short) 0, (unsigned short) 99, CUB_TYPE_STRING(unsigned short));
    Test<LOGICAL_WARP_THREADS>(gen_mode, Max<unsigned int>(), (unsigned int) 0, (unsigned int) 99, CUB_TYPE_STRING(unsigned int));
    Test<LOGICAL_WARP_THREADS>(gen_mode, Max<unsigned long long>(), (unsigned long long) 0, (unsigned long long) 99, CUB_TYPE_STRING(unsigned long long));

    // vec-2
    Test<LOGICAL_WARP_THREADS>(gen_mode, Sum<uchar2>(), make_uchar2(0, 0), make_uchar2(17, 21), CUB_TYPE_STRING(uchar2));
    Test<LOGICAL_WARP_THREADS>(gen_mode, Sum<ushort2>(), make_ushort2(0, 0), make_ushort2(17, 21), CUB_TYPE_STRING(ushort2));
    Test<LOGICAL_WARP_THREADS>(gen_mode, Sum<uint2>(), make_uint2(0, 0), make_uint2(17, 21), CUB_TYPE_STRING(uint2));
    Test<LOGICAL_WARP_THREADS>(gen_mode, Sum<ulonglong2>(), make_ulonglong2(0, 0), make_ulonglong2(17, 21), CUB_TYPE_STRING(ulonglong2));

    // vec-4
    Test<LOGICAL_WARP_THREADS>(gen_mode, Sum<uchar4>(), make_uchar4(0, 0, 0, 0), make_uchar4(17, 21, 32, 85), CUB_TYPE_STRING(uchar4));
    Test<LOGICAL_WARP_THREADS>(gen_mode, Sum<ushort4>(), make_ushort4(0, 0, 0, 0), make_ushort4(17, 21, 32, 85), CUB_TYPE_STRING(ushort4));
    Test<LOGICAL_WARP_THREADS>(gen_mode, Sum<uint4>(), make_uint4(0, 0, 0, 0), make_uint4(17, 21, 32, 85), CUB_TYPE_STRING(uint4));
    Test<LOGICAL_WARP_THREADS>(gen_mode, Sum<ulonglong4>(), make_ulonglong4(0, 0, 0, 0), make_ulonglong4(17, 21, 32, 85), CUB_TYPE_STRING(ulonglong4));

    // complex
    Test<LOGICAL_WARP_THREADS>(gen_mode, Sum<TestFoo>(), TestFoo::MakeTestFoo(0, 0, 0, 0), TestFoo::MakeTestFoo(17, 21, 32, 85), CUB_TYPE_STRING(TestFoo));
    Test<LOGICAL_WARP_THREADS>(gen_mode, Sum<TestBar>(), TestBar::MakeTestBar(0, 0), TestBar::MakeTestBar(17, 21), CUB_TYPE_STRING(TestBar));
}


/**
 * Run battery of tests for different problem generation options
 */
template <int LOGICAL_WARP_THREADS>
void Test()
{
    Test<LOGICAL_WARP_THREADS>(UNIFORM);
    Test<LOGICAL_WARP_THREADS>(SEQ_INC);
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
        // Quick exclusive test
        Test<32, PREFIX_AGGREGATE>(UNIFORM, Sum<int>(), (int) 0, (int) 99, CUB_TYPE_STRING(int));
    }
    else
    {
        // Test logical warp sizes
        Test<32>();
        Test<16>();
        Test<9>();
        Test<7>();
    }

    return 0;
}



