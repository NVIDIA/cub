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
 * Test of DeviceScan utilities
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

bool                    g_verbose           = false;
int                     g_timing_iterations = 0;
int                     g_repeat            = 0;
CachingDeviceAllocator  g_allocator;


//---------------------------------------------------------------------
// Dispatch to different DeviceScan entrypoints
//---------------------------------------------------------------------


/**
 * Exclusive scan specialization
 */
template <
    typename InputIteratorRA,
    typename OutputIteratorRA,
    typename ScanOp,
    typename Identity,
    typename SizeT>
__host__ __device__ __forceinline__
cudaError_t Dispatch(
    void                        *d_temp_storage,                    ///< [in] %Device allocation of temporary storage.  When NULL, the required allocation size is returned in \p temp_storage_bytes and no work is done.
    size_t                      &temp_storage_bytes,                ///< [in,out] Size in bytes of \t d_temp_storage allocation.
    InputIteratorRA             d_in,
    OutputIteratorRA            d_out,
    ScanOp                      scan_op,
    Identity                    identity,
    SizeT                       num_items,
    cudaStream_t                stream              = 0,
    bool                        stream_synchronous  = false)
{
    return DeviceScan::ExclusiveScan(d_temp_storage, temp_storage_bytes, d_in, d_out, scan_op, identity, num_items, stream, stream_synchronous);
}


/**
 * Exclusive sum specialization
 */
template <
    typename InputIteratorRA,
    typename OutputIteratorRA,
    typename T,
    typename Identity,
    typename SizeT>
__host__ __device__ __forceinline__
cudaError_t Dispatch(
    void                        *d_temp_storage,                    ///< [in] %Device allocation of temporary storage.  When NULL, the required allocation size is returned in \p temp_storage_bytes and no work is done.
    size_t                      &temp_storage_bytes,                ///< [in,out] Size in bytes of \t d_temp_storage allocation.
    InputIteratorRA             d_in,
    OutputIteratorRA            d_out,
    Sum                         scan_op,
    Identity                    identity,
    SizeT                       num_items,
    cudaStream_t                stream              = 0,
    bool                        stream_synchronous  = false)
{
    return DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items, stream, stream_synchronous);
}


/**
 * Inclusive scan specialization
 */
template <
    typename InputIteratorRA,
    typename OutputIteratorRA,
    typename ScanOp,
    typename SizeT>
__host__ __device__ __forceinline__
cudaError_t Dispatch(
    void                        *d_temp_storage,                    ///< [in] %Device allocation of temporary storage.  When NULL, the required allocation size is returned in \p temp_storage_bytes and no work is done.
    size_t                      &temp_storage_bytes,                ///< [in,out] Size in bytes of \t d_temp_storage allocation.
    InputIteratorRA             d_in,
    OutputIteratorRA            d_out,
    ScanOp                      scan_op,
    NullType                    identity,
    SizeT                       num_items,
    cudaStream_t                stream              = 0,
    bool                        stream_synchronous  = false)
{
    return DeviceScan::InclusiveScan(d_temp_storage, temp_storage_bytes, d_in, d_out, scan_op, num_items, stream, stream_synchronous);
}


/**
 * Inclusive sum specialization
 */
template <
    typename InputIteratorRA,
    typename OutputIteratorRA,
    typename T,
    typename SizeT>
__host__ __device__ __forceinline__
cudaError_t Dispatch(
    void                        *d_temp_storage,                    ///< [in] %Device allocation of temporary storage.  When NULL, the required allocation size is returned in \p temp_storage_bytes and no work is done.
    size_t                      &temp_storage_bytes,                ///< [in,out] Size in bytes of \t d_temp_storage allocation.
    InputIteratorRA             d_in,
    OutputIteratorRA            d_out,
    Sum                         scan_op,
    NullType                    identity,
    SizeT                       num_items,
    cudaStream_t                stream              = 0,
    bool                        stream_synchronous  = false)
{
    return DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items, stream, stream_synchronous);
}

//---------------------------------------------------------------------
// CUDA Nested Parallelism Test Kernel
//---------------------------------------------------------------------

/**
 * Simple wrapper kernel to invoke DeviceScan
 */
/*
template <
    bool                STREAM_SYNCHRONOUS,
    typename            InputIteratorRA,
    typename            OutputIteratorRA,
    typename            ScanOp>
__global__ void CnpScan(
    InputIteratorRA     d_in,
    OutputIteratorRA    d_out,
    int                 num_items,
    ScanOp              scan_op,
    int                 iterations,
    cudaError_t*        d_cnp_error)
{
    cudaError_t error = cudaSuccess;

#ifdef CUB_RUNTIME_ENABLED
    for (int i = 0; i < iterations; ++i)
    {
        error = DeviceScan::Scan(d_in, d_out, num_items, scan_op, 0, STREAM_SYNCHRONOUS);
    }
#else
    error = cudaErrorNotSupported;
#endif

    *d_cnp_error = error;
}
*/

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
    GenMode      gen_mode,
    T            *h_in,
    T            *h_reference,
    int          num_items,
    ScanOp       scan_op,
    IdentityT    identity)
{
    T inclusive = identity;
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
    typename     T,
    typename     ScanOp>
T Initialize(
    GenMode      gen_mode,
    T            *h_in,
    T            *h_reference,
    int          num_items,
    ScanOp       scan_op,
    NullType)
{
    T inclusive;
    T aggregate;
    for (int i = 0; i < num_items; ++i)
    {
        InitValue(gen_mode, h_in[i], i);
        if (i == 0)
        {
            inclusive = h_in[0];
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




//---------------------------------------------------------------------
// Test generation
//---------------------------------------------------------------------

/**
 * Test DeviceScan
 */
template <
    typename        T,
    typename        ScanOp,
    typename        IdentityT>
void Test(
    int             num_items,
    GenMode         gen_mode,
    ScanOp          scan_op,
    IdentityT       identity,
    char*           type_string)
{
    int compare = 0;
    int cnp_compare = 0;

    printf("%s cub::DeviceScan::%s %d items, %s %d-byte elements, gen-mode %s\n",
        (Equals<IdentityT, NullType>::VALUE) ? "Inclusive" : "Exclusive",
        (Equals<ScanOp, Sum>::VALUE) ? "Sum" : "Scan",
        num_items,
        type_string,
        (int) sizeof(T),
        (gen_mode == RANDOM) ? "RANDOM" : (gen_mode == SEQ_INC) ? "SEQUENTIAL" : "HOMOGENOUS");
    fflush(stdout);

    // Allocate host arrays
    T*  h_in = new T[num_items];
    T*  h_reference = new T[num_items];

    // Initialize problem
    Initialize(gen_mode, h_in, h_reference, num_items, scan_op, identity);

    // Allocate device arrays
    T*              d_in = NULL;
    T*              d_out = NULL;
    cudaError_t*    d_cnp_error = NULL;
    void            *d_temporary_storage = NULL;
    size_t          temporary_storage_bytes = 0;
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_in,          sizeof(T) * num_items));
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_out,         sizeof(T) * num_items));
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_cnp_error,   sizeof(cudaError_t) * 1));

    // Initialize device arrays
    CubDebugExit(cudaMemcpy(d_in, h_in, sizeof(T) * num_items, cudaMemcpyHostToDevice));
    CubDebugExit(cudaMemset(d_out, 0, sizeof(T) * num_items));

    // Allocate temporary storage
    CubDebugExit(Dispatch(d_temporary_storage, temporary_storage_bytes, d_in, d_out, scan_op, identity, num_items, 0, true));
    CubDebugExit(g_allocator.DeviceAllocate(&d_temporary_storage, temporary_storage_bytes));

    // Run warmup/correctness iteration
    printf("Host dispatch:\n"); fflush(stdout);
    CubDebugExit(Dispatch(d_temporary_storage, temporary_storage_bytes, d_in, d_out, scan_op, identity, num_items, 0, true));

    // Check for correctness (and display results, if specified)
    compare = CompareDeviceResults(h_reference, d_out, num_items, true, g_verbose);
    printf("\t%s", compare ? "FAIL" : "PASS");

    // Flush any stdout/stderr
    fflush(stdout);
    fflush(stderr);

    // Performance
    GpuTimer gpu_timer;
    float elapsed_millis = 0.0;
    for (int i = 0; i < g_timing_iterations; i++)
    {
        gpu_timer.Start();

        CubDebugExit(Dispatch(d_temporary_storage, temporary_storage_bytes, d_in, d_out, scan_op, identity, num_items));

        gpu_timer.Stop();
        elapsed_millis += gpu_timer.ElapsedMillis();
    }
    if (g_timing_iterations > 0)
    {
        float avg_millis = elapsed_millis / g_timing_iterations;
        float grate = float(num_items) / avg_millis / 1000.0 / 1000.0;
        float gbandwidth = grate * sizeof(T) * 2;
        printf(", %.3f avg ms, %.3f billion items/s, %.3f GB/s", avg_millis, grate, gbandwidth);
    }

    printf("\n\n");


    // Evaluate using CUDA nested parallelism
/*
#if (CUB_CNP == 1)

    CubDebugExit(cudaMemset(d_out, 0, sizeof(T) * 1));

    // Run warmup/correctness iteration
    printf("\nDevice dispatch:\n"); fflush(stdout);
    CnpScan<true><<<1,1>>>(d_in, d_out, num_items, scan_op, 1, d_cnp_error);

    // Flush any stdout/stderr
    fflush(stdout);
    fflush(stderr);

    // Check if we were compiled and linked for CNP
    cudaError_t h_cnp_error;
    CubDebugExit(cudaMemcpy(&h_cnp_error, d_cnp_error, sizeof(cudaError_t) * 1, cudaMemcpyDeviceToHost));
    if (h_cnp_error == cudaErrorInvalidConfiguration)
    {
        printf("CNP not supported");
    }
    else
    {
        CubDebugExit(h_cnp_error);

        // Check for correctness (and display results, if specified)
        cnp_compare = CompareDeviceResults(h_reference, d_out, 1, g_verbose, g_verbose);
        printf("\n%s", cnp_compare ? "FAIL" : "PASS");

        // Performance
        gpu_timer.Start();

        CnpScan<false><<<1,1>>>(d_in, d_out, num_items, scan_op, g_timing_iterations, d_cnp_error);

        gpu_timer.Stop();
        elapsed_millis = gpu_timer.ElapsedMillis();

        if (g_timing_iterations > 0)
        {
            float avg_millis = elapsed_millis / g_timing_iterations;
            float grate = float(num_items) / avg_millis / 1000.0 / 1000.0;
            float gbandwidth = grate * sizeof(T) * 2;
            printf(", %.3f avg ms, %.3f billion items/s, %.3f GB/s\n", avg_millis, grate, gbandwidth);
        }
        else
        {
            printf("\n");
        }
    }

#endif
*/

    // Cleanup
    if (h_in) delete[] h_in;
    if (h_reference) delete[] h_reference;
    if (d_in) CubDebugExit(g_allocator.DeviceFree(d_in));
    if (d_out) CubDebugExit(g_allocator.DeviceFree(d_out));
    if (d_cnp_error) CubDebugExit(g_allocator.DeviceFree(d_cnp_error));
    if (d_temporary_storage) CubDebugExit(g_allocator.DeviceFree(d_temporary_storage));

    // Correctness asserts
    AssertEquals(0, compare);
    AssertEquals(0, cnp_compare);
}


/**
 * Iterative different gen modes
 */
template <
    typename        T,
    typename        ScanOp,
    typename        Identity>
void Test(
    int             num_items,
    ScanOp          scan_op,
    Identity        identity,
    char*           type_string)
{
    Test<T>(num_items, UNIFORM, scan_op, identity, type_string);
    Test<T>(num_items, RANDOM, scan_op, identity, type_string);
}


/**
 * Iterate inclusive/exclusive
 */
template <
    typename        T,
    typename        ScanOp>
void Test(
    int             num_items,
    ScanOp          scan_op,
    char*           type_string)
{
    Test<T>(num_items, scan_op, T(), type_string);          // exclusive
    Test<T>(num_items, scan_op, NullType(), type_string);   // inclusive
}


/**
 * Iterate sum/scan
 */
template <
    typename        T>
void TestOp(
    int             num_items,
    char*           type_string)
{
    Test<T>(num_items, Sum(), type_string);
    Test<T>(num_items, Max(), type_string);
}


/**
 * Iterate input sizes
 */
template <
    typename        T>
void Test(
    int             num_items,
    char*           type_string)
{
    if (num_items < 0)
    {
        TestOp<T>(1,        type_string);
        TestOp<T>(100,      type_string);
        TestOp<T>(10000,    type_string);
        TestOp<T>(1000000,  type_string);
    }
    else
    {
        TestOp<T>(num_items, type_string);
    }
}


//---------------------------------------------------------------------
// Main
//---------------------------------------------------------------------

/**
 * Main
 */
int main(int argc, char** argv)
{
    int num_items = -1;

    // Initialize command line
    CommandLineArgs args(argc, argv);
    g_verbose = args.CheckCmdLineFlag("v");
    bool quick = args.CheckCmdLineFlag("quick");
    args.GetCmdLineArgument("n", num_items);
    args.GetCmdLineArgument("i", g_timing_iterations);
    args.GetCmdLineArgument("repeat", g_repeat);

    // Print usage
    if (args.CheckCmdLineFlag("help"))
    {
        printf("%s "
            "[--n=<input items> "
            "[--i=<timing iterations> "
            "[--device=<device-id>] "
            "[--repeat=<times to repeat tests>]"
            "[--quick]"
            "[--v] "
            "[--cnp]"
            "\n", argv[0]);
        exit(0);
    }

    // Initialize device
    CubDebugExit(args.DeviceInit());
    printf("\n");

    if (quick)
    {
        // Quick test
        if (num_items < 0) num_items = 32000000;

        Test<char>(        num_items * 4, RANDOM, Sum(), int(0), CUB_TYPE_STRING(int));
        Test<short>(       num_items * 2, RANDOM, Sum(), short(0), CUB_TYPE_STRING(short));
        Test<int>(         num_items    , RANDOM, Sum(), (int) (0), CUB_TYPE_STRING(int));
        Test<long long>(   num_items / 2, RANDOM, Sum(), (long long) (0), CUB_TYPE_STRING(long long));
        Test<TestFoo>(     num_items / 4, RANDOM, Sum(), TestFoo(), CUB_TYPE_STRING(TestFoo));
    }
    else
    {
        // Repeat test sequence
        for (int i = 0; i <= g_repeat; ++i)
        {
            // Test different input types
            Test<unsigned char>(num_items, CUB_TYPE_STRING(unsigned char));
            Test<unsigned short>(num_items, CUB_TYPE_STRING(unsigned short));
            Test<unsigned int>(num_items, CUB_TYPE_STRING(unsigned int));
            Test<unsigned long long>(num_items, CUB_TYPE_STRING(unsigned long long));

            Test<uchar2>(num_items, CUB_TYPE_STRING(uchar2));
            Test<uint2>(num_items, CUB_TYPE_STRING(uint2));
            Test<ulonglong2>(num_items, CUB_TYPE_STRING(ulonglong2));
            Test<ulonglong4>(num_items, CUB_TYPE_STRING(ulonglong4));

            Test<TestFoo>(num_items, CUB_TYPE_STRING(TestFoo));
            Test<TestBar>(num_items, CUB_TYPE_STRING(TestBar));
        }
    }

    return 0;
}



