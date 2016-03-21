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
 * Test of DeviceScan utilities
 ******************************************************************************/

// Ensure printing of CUDA runtime errors to console
#define CUB_STDERR

#include <stdio.h>
#include <typeinfo>

#include <cub/util_allocator.cuh>
#include <cub/iterator/constant_input_iterator.cuh>
#include <cub/device/device_scan.cuh>

#include <thrust/device_ptr.h>
#include <thrust/scan.h>

#include "test_util.h"

using namespace cub;


//---------------------------------------------------------------------
// Globals, constants and typedefs
//---------------------------------------------------------------------

bool                    g_verbose           = false;
int                     g_timing_iterations = 0;
int                     g_repeat            = 0;
double                  g_device_giga_bandwidth;
CachingDeviceAllocator  g_allocator(true);

// Dispatch types
enum Backend
{
    CUB,        // CUB method
    THRUST,     // Thrust method
    CDP,        // GPU-based (dynamic parallelism) dispatch to CUB method
};


//---------------------------------------------------------------------
// Dispatch to different CUB DeviceScan entrypoints
//---------------------------------------------------------------------

/**
 * Dispatch to exclusive scan entrypoint
 */
template <typename InputIteratorT, typename OutputIteratorT, typename ScanOp, typename Identity, typename OffsetT>
CUB_RUNTIME_FUNCTION __forceinline__
cudaError_t Dispatch(
    Int2Type<CUB>       dispatch_to,
    int                 timing_timing_iterations,
    size_t              *d_temp_storage_bytes,
    cudaError_t         *d_cdp_error,

    void*               d_temp_storage,
    size_t&             temp_storage_bytes,
    InputIteratorT      d_in,
    OutputIteratorT      d_out,
    ScanOp              scan_op,
    Identity            identity,
    OffsetT             num_items,
    cudaStream_t        stream,
    bool                debug_synchronous)
{
    cudaError_t error = cudaSuccess;
    for (int i = 0; i < timing_timing_iterations; ++i)
    {
        error = DeviceScan::ExclusiveScan(d_temp_storage, temp_storage_bytes, d_in, d_out, scan_op, identity, num_items, stream, debug_synchronous);
    }
    return error;
}


/**
 * Dispatch to exclusive sum entrypoint
 */
template <typename InputIteratorT, typename OutputIteratorT, typename Identity, typename OffsetT>
CUB_RUNTIME_FUNCTION __forceinline__
cudaError_t Dispatch(
    Int2Type<CUB>       dispatch_to,
    int                 timing_timing_iterations,
    size_t              *d_temp_storage_bytes,
    cudaError_t         *d_cdp_error,

    void*               d_temp_storage,
    size_t&             temp_storage_bytes,
    InputIteratorT      d_in,
    OutputIteratorT      d_out,
    Sum                 scan_op,
    Identity            identity,
    OffsetT             num_items,
    cudaStream_t        stream,
    bool                debug_synchronous)
{
    cudaError_t error = cudaSuccess;
    for (int i = 0; i < timing_timing_iterations; ++i)
    {
        error = DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items, stream, debug_synchronous);
    }
    return error;
}


/**
 * Dispatch to inclusive scan entrypoint
 */
template <typename InputIteratorT, typename OutputIteratorT, typename ScanOp, typename OffsetT>
CUB_RUNTIME_FUNCTION __forceinline__
cudaError_t Dispatch(
    Int2Type<CUB>       dispatch_to,
    int                 timing_timing_iterations,
    size_t              *d_temp_storage_bytes,
    cudaError_t         *d_cdp_error,

    void*               d_temp_storage,
    size_t&             temp_storage_bytes,
    InputIteratorT      d_in,
    OutputIteratorT      d_out,
    ScanOp              scan_op,
    NullType            identity,
    OffsetT             num_items,
    cudaStream_t        stream,
    bool                debug_synchronous)
{
    cudaError_t error = cudaSuccess;
    for (int i = 0; i < timing_timing_iterations; ++i)
    {
        error = DeviceScan::InclusiveScan(d_temp_storage, temp_storage_bytes, d_in, d_out, scan_op, num_items, stream, debug_synchronous);
    }
    return error;
}


/**
 * Dispatch to inclusive sum entrypoint
 */
template <typename InputIteratorT, typename OutputIteratorT, typename OffsetT>
CUB_RUNTIME_FUNCTION __forceinline__
cudaError_t Dispatch(
    Int2Type<CUB>       dispatch_to,
    int                 timing_timing_iterations,
    size_t              *d_temp_storage_bytes,
    cudaError_t         *d_cdp_error,

    void*               d_temp_storage,
    size_t&             temp_storage_bytes,
    InputIteratorT      d_in,
    OutputIteratorT      d_out,
    Sum                 scan_op,
    NullType            identity,
    OffsetT             num_items,
    cudaStream_t        stream,
    bool                debug_synchronous)
{
    cudaError_t error = cudaSuccess;
    for (int i = 0; i < timing_timing_iterations; ++i)
    {
        error = DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items, stream, debug_synchronous);
    }
    return error;
}

//---------------------------------------------------------------------
// Dispatch to different Thrust entrypoints
//---------------------------------------------------------------------

/**
 * Dispatch to exclusive scan entrypoint
 */
template <typename InputIteratorT, typename OutputIteratorT, typename ScanOp, typename Identity, typename OffsetT>
cudaError_t Dispatch(
    Int2Type<THRUST>    dispatch_to,
    int                 timing_timing_iterations,
    size_t              *d_temp_storage_bytes,
    cudaError_t         *d_cdp_error,

    void*               d_temp_storage,
    size_t&             temp_storage_bytes,
    InputIteratorT      d_in,
    OutputIteratorT      d_out,
    ScanOp              scan_op,
    Identity            identity,
    OffsetT             num_items,
    cudaStream_t        stream,
    bool                debug_synchronous)
{
    typedef typename std::iterator_traits<InputIteratorT>::value_type T;

    if (d_temp_storage == 0)
    {
        temp_storage_bytes = 1;
    }
    else
    {
        thrust::device_ptr<T> d_in_wrapper(d_in);
        thrust::device_ptr<T> d_out_wrapper(d_out);
        for (int i = 0; i < timing_timing_iterations; ++i)
        {
            thrust::exclusive_scan(d_in_wrapper, d_in_wrapper + num_items, d_out_wrapper, identity, scan_op);
        }
    }

    return cudaSuccess;
}


/**
 * Dispatch to exclusive sum entrypoint
 */
template <typename InputIteratorT, typename OutputIteratorT, typename Identity, typename OffsetT>
cudaError_t Dispatch(
    Int2Type<THRUST>    dispatch_to,
    int                 timing_timing_iterations,
    size_t              *d_temp_storage_bytes,
    cudaError_t         *d_cdp_error,

    void*               d_temp_storage,
    size_t&             temp_storage_bytes,
    InputIteratorT      d_in,
    OutputIteratorT      d_out,
    Sum                 scan_op,
    Identity            identity,
    OffsetT             num_items,
    cudaStream_t        stream,
    bool                debug_synchronous)
{
    typedef typename std::iterator_traits<InputIteratorT>::value_type T;

    if (d_temp_storage == 0)
    {
        temp_storage_bytes = 1;
    }
    else
    {
        thrust::device_ptr<T> d_in_wrapper(d_in);
        thrust::device_ptr<T> d_out_wrapper(d_out);
        for (int i = 0; i < timing_timing_iterations; ++i)
        {
            thrust::exclusive_scan(d_in_wrapper, d_in_wrapper + num_items, d_out_wrapper);
        }
    }

    return cudaSuccess;
}


/**
 * Dispatch to inclusive scan entrypoint
 */
template <typename InputIteratorT, typename OutputIteratorT, typename ScanOp, typename OffsetT>
cudaError_t Dispatch(
    Int2Type<THRUST>    dispatch_to,
    int                 timing_timing_iterations,
    size_t              *d_temp_storage_bytes,
    cudaError_t         *d_cdp_error,

    void*               d_temp_storage,
    size_t&             temp_storage_bytes,
    InputIteratorT      d_in,
    OutputIteratorT      d_out,
    ScanOp              scan_op,
    NullType            identity,
    OffsetT             num_items,
    cudaStream_t        stream,
    bool                debug_synchronous)
{
    typedef typename std::iterator_traits<InputIteratorT>::value_type T;

    if (d_temp_storage == 0)
    {
        temp_storage_bytes = 1;
    }
    else
    {
        thrust::device_ptr<T> d_in_wrapper(d_in);
        thrust::device_ptr<T> d_out_wrapper(d_out);
        for (int i = 0; i < timing_timing_iterations; ++i)
        {
            thrust::inclusive_scan(d_in_wrapper, d_in_wrapper + num_items, d_out_wrapper, scan_op);
        }
    }

    return cudaSuccess;
}


/**
 * Dispatch to inclusive sum entrypoint
 */
template <typename InputIteratorT, typename OutputIteratorT, typename OffsetT>
cudaError_t Dispatch(
    Int2Type<THRUST>    dispatch_to,
    int                 timing_timing_iterations,
    size_t              *d_temp_storage_bytes,
    cudaError_t         *d_cdp_error,

    void*               d_temp_storage,
    size_t&             temp_storage_bytes,
    InputIteratorT      d_in,
    OutputIteratorT      d_out,
    Sum                 scan_op,
    NullType            identity,
    OffsetT             num_items,
    cudaStream_t        stream,
    bool                debug_synchronous)
{
    typedef typename std::iterator_traits<InputIteratorT>::value_type T;

    if (d_temp_storage == 0)
    {
        temp_storage_bytes = 1;
    }
    else
    {
        thrust::device_ptr<T> d_in_wrapper(d_in);
        thrust::device_ptr<T> d_out_wrapper(d_out);
        for (int i = 0; i < timing_timing_iterations; ++i)
        {
            thrust::inclusive_scan(d_in_wrapper, d_in_wrapper + num_items, d_out_wrapper);
        }
    }

    return cudaSuccess;
}



//---------------------------------------------------------------------
// CUDA Nested Parallelism Test Kernel
//---------------------------------------------------------------------

/**
 * Simple wrapper kernel to invoke DeviceScan
 */
template <typename InputIteratorT, typename OutputIteratorT, typename ScanOp, typename Identity, typename OffsetT>
__global__ void CnpDispatchKernel(
    int                 timing_timing_iterations,
    size_t              *d_temp_storage_bytes,
    cudaError_t         *d_cdp_error,

    void*               d_temp_storage,
    size_t              temp_storage_bytes,
    InputIteratorT      d_in,
    OutputIteratorT      d_out,
    ScanOp              scan_op,
    Identity            identity,
    OffsetT             num_items,
    bool                debug_synchronous)
{
#ifndef CUB_CDP
    *d_cdp_error = cudaErrorNotSupported;
#else
    *d_cdp_error = Dispatch(Int2Type<CUB>(), timing_timing_iterations, d_temp_storage_bytes, d_cdp_error, d_temp_storage, temp_storage_bytes, d_in, d_out, scan_op, identity, num_items, 0, debug_synchronous);
    *d_temp_storage_bytes = temp_storage_bytes;
#endif
}


/**
 * Dispatch to CDP kernel
 */
template <typename InputIteratorT, typename OutputIteratorT, typename ScanOp, typename Identity, typename OffsetT>
cudaError_t Dispatch(
    Int2Type<CDP>       dispatch_to,
    int                 timing_timing_iterations,
    size_t              *d_temp_storage_bytes,
    cudaError_t         *d_cdp_error,

    void*               d_temp_storage,
    size_t&             temp_storage_bytes,
    InputIteratorT      d_in,
    OutputIteratorT      d_out,
    ScanOp              scan_op,
    Identity            identity,
    OffsetT             num_items,
    cudaStream_t        stream,
    bool                debug_synchronous)
{
    // Invoke kernel to invoke device-side dispatch
    CnpDispatchKernel<<<1,1>>>(timing_timing_iterations, d_temp_storage_bytes, d_cdp_error, d_temp_storage, temp_storage_bytes, d_in, d_out, scan_op, identity, num_items, debug_synchronous);

    // Copy out temp_storage_bytes
    CubDebugExit(cudaMemcpy(&temp_storage_bytes, d_temp_storage_bytes, sizeof(size_t) * 1, cudaMemcpyDeviceToHost));

    // Copy out error
    cudaError_t retval;
    CubDebugExit(cudaMemcpy(&retval, d_cdp_error, sizeof(cudaError_t) * 1, cudaMemcpyDeviceToHost));
    return retval;
}



//---------------------------------------------------------------------
// Test generation
//---------------------------------------------------------------------


/**
 * Initialize problem
 */
template <typename T>
void Initialize(
    GenMode      gen_mode,
    T            *h_in,
    int          num_items)
{
    for (int i = 0; i < num_items; ++i)
    {
        InitValue(gen_mode, h_in[i], i);
    }

    if (g_verbose)
    {
        printf("Input:\n");
        DisplayResults(h_in, num_items);
        printf("\n\n");
    }
}

/**
 * Solve exclusive-scan problem
 */
template <
    typename        InputIteratorT,
    typename        T,
    typename        ScanOp,
    typename        IdentityT>
T Solve(
    InputIteratorT  h_in,
    T               *h_reference,
    int             num_items,
    ScanOp          scan_op,
    IdentityT       identity)
{
    T inclusive = identity;
    T aggregate = identity;

    for (int i = 0; i < num_items; ++i)
    {
        h_reference[i] = inclusive;
        inclusive = scan_op(inclusive, h_in[i]);
        aggregate = scan_op(aggregate, h_in[i]);
    }

    return aggregate;
}


/**
 * Solve inclusive-scan problem
 */
template <
    typename        InputIteratorT,
    typename        T,
    typename        ScanOp>
T Solve(
    InputIteratorT  h_in,
    T               *h_reference,
    int             num_items,
    ScanOp          scan_op,
    NullType)
{
    T inclusive;
    T aggregate;
    for (int i = 0; i < num_items; ++i)
    {
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


/**
 * Test DeviceScan for a given problem input
 */
template <
    Backend             BACKEND,
    typename            DeviceInputIteratorT,
    typename            T,
    typename            ScanOp,
    typename            IdentityT>
void Test(
    DeviceInputIteratorT d_in,
    T                   *h_reference,
    int                 num_items,
    ScanOp              scan_op,
    IdentityT           identity)
{
    // Allocate device output array
    T *d_out = NULL;
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_out,         sizeof(T) * num_items));

    // Allocate CDP device arrays
    size_t          *d_temp_storage_bytes = NULL;
    cudaError_t     *d_cdp_error = NULL;
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_temp_storage_bytes,  sizeof(size_t) * 1));
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_cdp_error,   sizeof(cudaError_t) * 1));

    // Allocate temporary storage
    void            *d_temp_storage = NULL;
    size_t          temp_storage_bytes = 0;
    CubDebugExit(Dispatch(Int2Type<BACKEND>(), 1, d_temp_storage_bytes, d_cdp_error, d_temp_storage, temp_storage_bytes, d_in, d_out, scan_op, identity, num_items, 0, true));
    CubDebugExit(g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes));

    // Clear device output array
    CubDebugExit(cudaMemset(d_out, 0, sizeof(T) * num_items));

    // Run warmup/correctness iteration
    CubDebugExit(Dispatch(Int2Type<BACKEND>(), 1, d_temp_storage_bytes, d_cdp_error, d_temp_storage, temp_storage_bytes, d_in, d_out, scan_op, identity, num_items, 0, true));

    // Check for correctness (and display results, if specified)
    int compare = CompareDeviceResults(h_reference, d_out, num_items, true, g_verbose);
    printf("\t%s", compare ? "FAIL" : "PASS");

    // Flush any stdout/stderr
    fflush(stdout);
    fflush(stderr);

    // Performance
    GpuTimer gpu_timer;
    gpu_timer.Start();
    CubDebugExit(Dispatch(Int2Type<BACKEND>(), g_timing_iterations, d_temp_storage_bytes, d_cdp_error, d_temp_storage, temp_storage_bytes, d_in, d_out, scan_op, identity, num_items, 0, false));
    gpu_timer.Stop();
    float elapsed_millis = gpu_timer.ElapsedMillis();

    // Display performance
    if (g_timing_iterations > 0)
    {
        float avg_millis = elapsed_millis / g_timing_iterations;
        float giga_rate = float(num_items) / avg_millis / 1000.0 / 1000.0;
        float giga_bandwidth = giga_rate * sizeof(T) * 2;
        printf(", %.3f avg ms, %.3f billion items/s, %.3f logical GB/s, %.1f%% peak", avg_millis, giga_rate, giga_bandwidth, giga_bandwidth / g_device_giga_bandwidth * 100.0);
    }

    printf("\n\n");

    // Cleanup
    if (d_out) CubDebugExit(g_allocator.DeviceFree(d_out));
    if (d_temp_storage_bytes) CubDebugExit(g_allocator.DeviceFree(d_temp_storage_bytes));
    if (d_cdp_error) CubDebugExit(g_allocator.DeviceFree(d_cdp_error));
    if (d_temp_storage) CubDebugExit(g_allocator.DeviceFree(d_temp_storage));

    // Correctness asserts
    AssertEquals(0, compare);
}


/**
 * Test DeviceScan on pointer type
 */
template <
    Backend         BACKEND,
    typename        T,
    typename        ScanOp,
    typename        IdentityT>
void TestPointer(
    int             num_items,
    GenMode         gen_mode,
    ScanOp          scan_op,
    IdentityT       identity)
{
    printf("\nPointer %s %s cub::DeviceScan::%s %d items, %s %d-byte elements, gen-mode %s\n",
        (BACKEND == CDP) ? "CDP CUB" : (BACKEND == THRUST) ? "Thrust" : "CUB",
        (Equals<IdentityT, NullType>::VALUE) ? "Inclusive" : "Exclusive",
        (Equals<ScanOp, Sum>::VALUE) ? "Sum" : "Scan",
        num_items,
        typeid(T).name(),
        (int) sizeof(T),
        (gen_mode == RANDOM) ? "RANDOM" : (gen_mode == INTEGER_SEED) ? "SEQUENTIAL" : "HOMOGENOUS");
    fflush(stdout);

    // Allocate host arrays
    T*  h_in = new T[num_items];
    T*  h_reference = new T[num_items];

    // Initialize problem and solution
    Initialize(gen_mode, h_in, num_items);
    Solve(h_in, h_reference, num_items, scan_op, identity);

    // Allocate problem device arrays
    T *d_in = NULL;
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_in, sizeof(T) * num_items));

    // Initialize device input
    CubDebugExit(cudaMemcpy(d_in, h_in, sizeof(T) * num_items, cudaMemcpyHostToDevice));

    // Run Test
    Test<BACKEND>(d_in, h_reference, num_items, scan_op, identity);

    // Cleanup
    if (h_in) delete[] h_in;
    if (h_reference) delete[] h_reference;
    if (d_in) CubDebugExit(g_allocator.DeviceFree(d_in));
}


/**
 * Test DeviceScan on iterator type
 */
template <
    Backend         BACKEND,
    typename        T,
    typename        ScanOp,
    typename        IdentityT>
void TestIterator(
    int             num_items,
    ScanOp          scan_op,
    IdentityT       identity)
{
    printf("\nIterator %s %s cub::DeviceScan::%s %d items, %s %d-byte elements\n",
        (BACKEND == CDP) ? "CDP CUB" : (BACKEND == THRUST) ? "Thrust" : "CUB",
        (Equals<IdentityT, NullType>::VALUE) ? "Inclusive" : "Exclusive",
        (Equals<ScanOp, Sum>::VALUE) ? "Sum" : "Scan",
        num_items,
        typeid(T).name(),
        (int) sizeof(T));
    fflush(stdout);

    // Use a constant iterator as the input
    T val = T();
    ConstantInputIterator<T, int> h_in(val);

    // Allocate host arrays
    T*  h_reference = new T[num_items];

    // Initialize problem and solution
    Solve(h_in, h_reference, num_items, scan_op, identity);

    // Run Test
    Test<BACKEND>(h_in, h_reference, num_items, scan_op, identity);

    // Cleanup
    if (h_reference) delete[] h_reference;
}


/**
 * Test different gen modes
 */
template <
    Backend         BACKEND,
    typename        T,
    typename        ScanOp,
    typename        Identity>
void Test(
    int             num_items,
    ScanOp          scan_op,
    Identity        identity)
{
    TestPointer<BACKEND, T>(num_items, UNIFORM, scan_op, identity);
    TestPointer<BACKEND, T>(num_items, RANDOM, scan_op, identity);

    TestIterator<BACKEND, T>(num_items, scan_op, identity);
}


/**
 * Test different dispatch
 */
template <
    typename        T,
    typename        ScanOp,
    typename        IdentityT>
void Test(
    int             num_items,
    ScanOp          scan_op,
    IdentityT       identity)
{
    Test<CUB, T>(num_items, scan_op, identity);
#ifdef CUB_CDP
    Test<CDP, T>(num_items, scan_op, identity);
#endif
}




/**
 * Test inclusive/exclusive
 */
template <
    typename        T,
    typename        ScanOp>
void TestVariant(
    int             num_items,
    ScanOp          scan_op,
    T               identity)
{
    Test<T>(num_items, scan_op, identity);     // exclusive
    Test<T>(num_items, scan_op, NullType());   // inclusive
}


/**
 * Test different operators
 */
template <typename T>
void TestOp(
    int             num_items,
    T               max_identity)
{
    TestVariant(num_items, Sum(), T());
    TestVariant(num_items, Max(), max_identity);
}


/**
 * Test different input sizes
 */
template <typename T>
void TestSize(
    int             num_items,
    T               max_identity)
{
    if (num_items < 0)
    {
        TestOp(0,        max_identity);
        TestOp(1,        max_identity);
        TestOp(100,      max_identity);
        TestOp(10000,    max_identity);
        TestOp(1000000,  max_identity);

        // Randomly select problem size between 1:10,000,000
        unsigned int max_int = (unsigned int) -1;
        for (int i = 0; i < 10; ++i)
        {
            unsigned int num_items;
            RandomBits(num_items);
            num_items = (unsigned int) ((double(num_items) * double(10000000)) / double(max_int));
            num_items = CUB_MAX(1, num_items);
            TestOp(num_items,  max_identity);
        }
    }
    else
    {
        TestOp(num_items, max_identity);
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
            "[--repeat=<repetitions of entire test suite>]"
            "[--v] "
            "[--cdp]"
            "\n", argv[0]);
        exit(0);
    }

    // Initialize device
    CubDebugExit(args.DeviceInit());
    g_device_giga_bandwidth = args.device_giga_bandwidth;
    printf("\n");

#ifdef QUICKER_TEST

    // Compile/run basic CUB test
    if (num_items < 0) num_items = 32000000;
    TestPointer<CUB, int>(         num_items    , UNIFORM, Sum(), (int) (0));

#elif defined(QUICK_TEST)

    // Get device ordinal
    int device_ordinal;
    CubDebugExit(cudaGetDevice(&device_ordinal));

    // Get device SM version
    int sm_version;
    CubDebugExit(SmVersion(sm_version, device_ordinal));

    // Compile/run quick tests
    if (num_items < 0) num_items = 32000000;

    TestPointer<CUB, char>(        num_items * ((sm_version <= 130) ? 1 : 4), UNIFORM, Sum(), char(0));
    TestPointer<THRUST, char>(     num_items * ((sm_version <= 130) ? 1 : 4), UNIFORM, Sum(), char(0));

    printf("----------------------------\n");
    TestPointer<CUB, short>(       num_items * ((sm_version <= 130) ? 1 : 2), UNIFORM, Sum(), short(0));
    TestPointer<THRUST, short>(    num_items * ((sm_version <= 130) ? 1 : 2), UNIFORM, Sum(), short(0));

    printf("----------------------------\n");
    TestPointer<CUB, int>(         num_items    , UNIFORM, Sum(), (int) (0));
    TestPointer<THRUST, int>(      num_items    , UNIFORM, Sum(), (int) (0));

    printf("----------------------------\n");
    TestPointer<CUB, long long>(   num_items / 2, UNIFORM, Sum(), (long long) (0));
    TestPointer<THRUST, long long>(num_items / 2, UNIFORM, Sum(), (long long) (0));

    printf("----------------------------\n");
    TestPointer<CUB, TestBar>(     num_items / 4, UNIFORM, Sum(), TestBar());
    TestPointer<THRUST, TestBar>(  num_items / 4, UNIFORM, Sum(), TestBar());

#else

    // Compile/run thorough tests
    for (int i = 0; i <= g_repeat; ++i)
    {
        // Test different input types
        TestSize<unsigned char>(num_items, 0);
        TestSize<char>(num_items, 0);
        TestSize<unsigned short>(num_items, 0);
        TestSize<unsigned int>(num_items, 0);
        TestSize<unsigned long long>(num_items, 0);

        TestSize<uchar2>(num_items, make_uchar2(0, 0));
        TestSize<char2>(num_items, make_char2(0, 0));
        TestSize<ushort2>(num_items, make_ushort2(0, 0));
        TestSize<uint2>(num_items, make_uint2(0, 0));
        TestSize<ulonglong2>(num_items, make_ulonglong2(0, 0));

        TestSize<uchar4>(num_items, make_uchar4(0, 0, 0, 0));
        TestSize<char4>(num_items, make_char4(0, 0, 0, 0));
        TestSize<ushort4>(num_items, make_ushort4(0, 0, 0, 0));
        TestSize<uint4>(num_items, make_uint4(0, 0, 0, 0));
        TestSize<ulonglong4>(num_items, make_ulonglong4(0, 0, 0, 0));

        TestSize<TestFoo>(num_items, TestFoo::MakeTestFoo(1ll << 63, 1 << 31, short(1 << 15), char(1 << 7)));
        TestSize<TestBar>(num_items, TestBar(1ll << 63, 1 << 31));
    }

#endif

    return 0;
}



