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
 * Test of DeviceReduce::ReduceByKey utilities
 ******************************************************************************/

// Ensure printing of CUDA runtime errors to console
#define CUB_STDERR

#include <stdio.h>

#include <cub/util_allocator.cuh>
#include <cub/iterator/counting_input_iterator.cuh>
#include <cub/device/device_select.cuh>

#include <thrust/device_ptr.h>
#include <thrust/unique.h>

#include "test_util.h"

using namespace cub;


//---------------------------------------------------------------------
// Globals, constants and typedefs
//---------------------------------------------------------------------

bool                    g_verbose           = false;
int                     g_timing_iterations = 0;
int                     g_repeat            = 0;
CachingDeviceAllocator  g_allocator(true);



//---------------------------------------------------------------------
// Dispatch to different CUB DeviceSelect entrypoints
//---------------------------------------------------------------------


/**
 * Dispatch to unique entrypoint
 */
template <
    typename                    KeyInputIterator,
    typename                    KeyOutputIterator,
    typename                    ValueInputIterator,
    typename                    ValueOutputIterator,
    typename                    NumSegmentsIterator,
    typename                    EqualityOp,
    typename                    ReductionOp,
    typename                    Offset>
__host__ __device__ __forceinline__
cudaError_t Dispatch(
    Int2Type<CUB>               dispatch_to,
    int                         timing_timing_iterations,
    size_t                      *d_temp_storage_bytes,
    cudaError_t                 *d_cdp_error,

    void                        *d_temp_storage,
    size_t                      &temp_storage_bytes,
    KeyInputIterator            d_keys_in,
    KeyOutputIterator           d_keys_out,
    ValueInputIterator          d_values_in,
    ValueOutputIterator         d_values_out,
    NumSegmentsIterator         d_num_segments,
    EqualityOp                  equality_op,
    ReductionOp                 reduction_op,
    int                         num_items,
    cudaStream_t                stream,
    bool                        debug_synchronous)
{
    cudaError_t error = cudaSuccess;
    for (int i = 0; i < timing_timing_iterations; ++i)
    {
        error = DeviceReduce::ReduceByKey(
            d_temp_storage,
            temp_storage_bytes,
            d_keys_in,
            d_keys_out,
            d_values_in,
            d_values_out,
            d_num_segments,
            reduction_op,
            num_items,
            stream,
            debug_synchronous);
    }
    return error;
}


//---------------------------------------------------------------------
// Dispatch to different Thrust entrypoints
//---------------------------------------------------------------------


/**
 * Dispatch to unique entrypoint
 * /
template <typename InputIterator, typename OutputIterator, typename NumSelectedIterator, typename Offset>
__host__ __forceinline__
cudaError_t Dispatch(
    Int2Type<THRUST>            dispatch_to,
    int                         timing_timing_iterations,
    size_t                      *d_temp_storage_bytes,
    cudaError_t                 *d_cdp_error,

    void                        *d_temp_storage,
    size_t                      &temp_storage_bytes,
    InputIterator               d_in,
    OutputIterator              d_out,
    NumSelectedIterator         d_num_segments,
    Offset                      num_items,
    cudaStream_t                stream,
    bool                        debug_synchronous)
{
    typedef typename std::iterator_traits<InputIterator>::value_type T;

    if (d_temp_storage == 0)
    {
        temp_storage_bytes = 1;
    }
    else
    {
        thrust::device_ptr<T> d_out_wrapper_end;
        thrust::device_ptr<T> d_in_wrapper(d_in);
        thrust::device_ptr<T> d_out_wrapper(d_out);
        for (int i = 0; i < timing_timing_iterations; ++i)
        {
            d_out_wrapper_end = thrust::unique_copy(d_in_wrapper, d_in_wrapper + num_items, d_out_wrapper);
        }

        Offset num_segments = d_out_wrapper_end - d_out_wrapper;
        CubDebugExit(cudaMemcpy(d_num_segments, &num_segments, sizeof(Offset), cudaMemcpyHostToDevice));

    }

    return cudaSuccess;
}



//---------------------------------------------------------------------
// CUDA Nested Parallelism Test Kernel
//---------------------------------------------------------------------

/ **
 * Simple wrapper kernel to invoke DeviceSelect
 * /
template <typename InputIterator, typename OutputIterator, typename NumSelectedIterator, typename Offset>
__global__ void CnpDispatchKernel(
    int                         timing_timing_iterations,
    size_t                      *d_temp_storage_bytes,
    cudaError_t                 *d_cdp_error,

    void                        *d_temp_storage,
    size_t                      temp_storage_bytes,
    InputIterator               d_in,
    OutputIterator              d_out,
    NumSelectedIterator         d_num_segments,
    Offset                      num_items,
    bool                        debug_synchronous)
{

#ifndef CUB_CDP
    *d_cdp_error = cudaErrorNotSupported;
#else
    *d_cdp_error = Dispatch(Int2Type<CUB>(), timing_timing_iterations, d_temp_storage_bytes, d_cdp_error,
        d_temp_storage, temp_storage_bytes, d_in, d_out, d_num_segments, num_items, 0, debug_synchronous);
    *d_temp_storage_bytes = temp_storage_bytes;
#endif
}


/ **
 * Dispatch to CDP kernel
 * /
template <typename InputIterator, typename OutputIterator, typename NumSelectedIterator, typename Offset>
cudaError_t Dispatch(
    Int2Type<CDP>               dispatch_to,
    int                         timing_timing_iterations,
    size_t                      *d_temp_storage_bytes,
    cudaError_t                 *d_cdp_error,

    void                        *d_temp_storage,
    size_t                      &temp_storage_bytes,
    InputIterator               d_in,
    OutputIterator              d_out,
    NumSelectedIterator         d_num_segments,
    Offset                      num_items,
    cudaStream_t                stream,
    bool                        debug_synchronous)
{
    // Invoke kernel to invoke device-side dispatch
    CnpDispatchKernel<<<1,1>>>(timing_timing_iterations, d_temp_storage_bytes, d_cdp_error,
        d_temp_storage, temp_storage_bytes, d_in, d_out, d_num_segments, num_items, debug_synchronous);

    // Copy out temp_storage_bytes
    CubDebugExit(cudaMemcpy(&temp_storage_bytes, d_temp_storage_bytes, sizeof(size_t) * 1, cudaMemcpyDeviceToHost));

    // Copy out error
    cudaError_t retval;
    CubDebugExit(cudaMemcpy(&retval, d_cdp_error, sizeof(cudaError_t) * 1, cudaMemcpyDeviceToHost));
    return retval;
}

*/

//---------------------------------------------------------------------
// Test generation
//---------------------------------------------------------------------


/**
 * Initialize problem.  Keys are initialized to segment number
 * and values are initialized to 1
 */
template <
    typename Key,
    typename Value>
void Initialize(
    int         entropy_reduction,
    Key         *h_keys_in,
    Value       *h_values_in,
    int         num_items,
    int         max_segment)
{
    unsigned short max_short = (unsigned short) -1;

    int segment_id = 0;
    int i = 0;
    while (i < num_items)
    {
        // Select number of repeating occurrences

        unsigned short repeat;
        RandomBits(repeat, entropy_reduction);
        repeat = (unsigned short) ((float(repeat) * (float(max_segment) / float(max_short))));
        repeat = CUB_MAX(1, repeat);

        int j = i;
        while (j < CUB_MIN(i + repeat, num_items))
        {
            InitValue(SEQ_INC, h_keys_in[j], segment_id);
            InitValue(SEQ_INC, h_values_in[j], 1);
            j++;
        }

        i = j;
        segment_id++;
    }

    if (g_verbose)
    {
        printf("Input keys:\n");
        DisplayResults(h_keys_in, num_items);
        printf("\n\n");
        printf("Input values:\n");
        DisplayResults(h_keys_in, num_items);
        printf("\n\n");
    }
}


/**
 * Solve problem.  Returns total number of segments identified
 */
template <
    typename        KeyInputIterator,
    typename        ValueInputIterator,
    typename        Key,
    typename        Value,
    typename        EqualityOp,
    typename        ReductionOp>
int Solve(
    KeyInputIterator    h_keys_in,
    Key                 *h_keys_reference,
    ValueInputIterator  h_values_in,
    Value               *h_values_reference,
    EqualityOp          equality_op,
    ReductionOp         reduction_op,
    int                 num_items)
{
    // First item
    Key previous        = h_keys_in[0];
    Value aggregate     = h_values_in[0];
    int num_segments    = 0;

    // Subsequent items
    for (int i = 1; i < num_items; ++i)
    {
        if (!equality_op(previous, h_keys_in[i]))
        {
            h_keys_reference[num_segments] = previous;
            h_values_reference[num_segments] = aggregate;
            num_segments++;
            aggregate = h_values_in[i];
        }
        else
        {
            aggregate = reduction_op(aggregate, h_values_in[i]);
        }
        previous = h_keys_in[i];
    }

    h_keys_reference[num_segments] = previous;
    h_values_reference[num_segments] = aggregate;
    num_segments++;

    return num_segments;
}



/**
 * Test DeviceSelect for a given problem input
 */
template <
    Backend             BACKEND,
    typename            DeviceKeyInputIterator,
    typename            DeviceValueInputIterator,
    typename            Key,
    typename            Value,
    typename            EqualityOp,
    typename            ReductionOp>
void Test(
    DeviceKeyInputIterator      d_keys_in,
    DeviceValueInputIterator    d_values_in,
    Key                         *h_keys_reference,
    Value                       *h_values_reference,
    EqualityOp                  equality_op,
    ReductionOp                 reduction_op,
    int                         num_segments,
    int                         num_items,
    char*                       key_type_string,
    char*                       value_type_string)
{
    // Allocate device output arrays and number of segments
    Key     *d_keys_out             = NULL;
    Value   *d_values_out           = NULL;
    int     *d_num_segments         = NULL;
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_keys_out, sizeof(Key) * num_items));
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_keys_out, sizeof(Value) * num_items));
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_num_segments, sizeof(int)));

    // Allocate CDP device arrays
    size_t          *d_temp_storage_bytes = NULL;
    cudaError_t     *d_cdp_error = NULL;
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_temp_storage_bytes,  sizeof(size_t) * 1));
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_cdp_error,           sizeof(cudaError_t) * 1));

    // Allocate temporary storage
    void            *d_temp_storage = NULL;
    size_t          temp_storage_bytes = 0;
    CubDebugExit(Dispatch(Int2Type<BACKEND>(), 1, d_temp_storage_bytes, d_cdp_error, d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, d_values_in, d_values_out, d_num_segments, equality_op, reduction_op, num_items, 0, true));
    CubDebugExit(g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes));

    // Clear device output arrays
    CubDebugExit(cudaMemset(d_keys_out, 0, sizeof(Key) * num_items));
    CubDebugExit(cudaMemset(d_values_out, 0, sizeof(Value) * num_items));
    CubDebugExit(cudaMemset(d_num_segments, 0, sizeof(int)));

    // Run warmup/correctness iteration
    CubDebugExit(Dispatch(Int2Type<BACKEND>(), 1, d_temp_storage_bytes, d_cdp_error, d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, d_values_in, d_values_out, d_num_segments, equality_op, reduction_op, num_items, 0, true));

    // Check for correctness (and display results, if specified)
    int compare = CompareDeviceResults(h_keys_reference, d_keys_out, num_segments, true, g_verbose);
    printf("\t Keys %s ", compare ? "FAIL" : "PASS");

    compare |= CompareDeviceResults(h_values_reference, d_values_out, num_segments, true, g_verbose);
    printf("\t Values %s ", compare ? "FAIL" : "PASS");

    compare |= CompareDeviceResults(&num_segments, d_num_segments, 1, true, g_verbose);
    printf("\t Count %s ", compare ? "FAIL" : "PASS");

    // Flush any stdout/stderr
    fflush(stdout);
    fflush(stderr);

    // Performance
    GpuTimer gpu_timer;
    gpu_timer.Start();
    CubDebugExit(Dispatch(Int2Type<BACKEND>(), g_timing_iterations, d_temp_storage_bytes, d_cdp_error, d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, d_values_in, d_values_out, d_num_segments, equality_op, reduction_op, num_items, 0, false));
    gpu_timer.Stop();
    float elapsed_millis = gpu_timer.ElapsedMillis();

    // Display performance
    if (g_timing_iterations > 0)
    {
        float avg_millis = elapsed_millis / g_timing_iterations;
        float grate = float(num_items) / avg_millis / 1000.0 / 1000.0;
        int bytes_moved = (num_items + num_segments) * (sizeof(Key) + sizeof(Value));
        float gbandwidth = float(bytes_moved) / avg_millis / 1000.0 / 1000.0;
        printf(", %.3f avg ms, %.3f billion items/s, %.3f logical GB/s", avg_millis, grate, gbandwidth);
    }
    printf("\n\n");

    // Flush any stdout/stderr
    fflush(stdout);
    fflush(stderr);

    // Cleanup
    if (d_keys_out) CubDebugExit(g_allocator.DeviceFree(d_keys_out));
    if (d_values_out) CubDebugExit(g_allocator.DeviceFree(d_values_out));
    if (d_num_segments) CubDebugExit(g_allocator.DeviceFree(d_num_segments));
    if (d_temp_storage_bytes) CubDebugExit(g_allocator.DeviceFree(d_temp_storage_bytes));
    if (d_cdp_error) CubDebugExit(g_allocator.DeviceFree(d_cdp_error));
    if (d_temp_storage) CubDebugExit(g_allocator.DeviceFree(d_temp_storage));

    // Correctness asserts
    AssertEquals(0, compare);
}


/**
 * Test DeviceSelect on pointer type
 */
template <
    Backend         BACKEND,
    typename        Key,
    typename        Value,
    typename        ReductionOp>
void TestPointer(
    int             num_items,
    int             entropy_reduction,
    int             max_segment,
    ReductionOp     reduction_op,
    char*           key_type_string,
    char*           value_type_string)
{
    // Allocate host arrays
    Key* h_keys_in        = new Key[num_items];
    Key* h_keys_reference = new Key[num_items];

    Value* h_values_in        = new Value[num_items];
    Value* h_values_reference = new Value[num_items];

    // Initialize problem and solution
    Equality equality_op;
    Initialize(entropy_reduction, h_keys_in, h_values_in, num_items, max_segment);
    int num_segments = Solve(h_keys_in, h_keys_reference, h_values_in, h_values_reference, equality_op, reduction_op, num_items);

    printf("\nPointer %s cub::DeviceReduce::ReduceByKey %d items, %d segments (avg run length %d), {%s,%s} key value pairs, entropy_reduction %d\n",
        (BACKEND == CDP) ? "CDP CUB" : (BACKEND == THRUST) ? "Thrust" : "CUB",
        num_items, num_segments, num_items / num_segments,
        key_type_string, value_type_string,
        entropy_reduction);
    fflush(stdout);

    // Allocate problem device arrays
    Key     *d_keys_in = NULL;
    Value   *d_values_in = NULL;
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_keys_in, sizeof(Key) * num_items));
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_values_in, sizeof(Value) * num_items));

    // Initialize device input
    CubDebugExit(cudaMemcpy(d_keys_in, h_keys_in, sizeof(Key) * num_items, cudaMemcpyHostToDevice));
    CubDebugExit(cudaMemcpy(d_values_in, h_values_in, sizeof(Value) * num_items, cudaMemcpyHostToDevice));

    // Run Test
    Test<BACKEND>(d_keys_in, d_values_in, h_keys_reference, h_values_reference, equality_op, reduction_op, num_segments, num_items, key_type_string, value_type_string);

    // Cleanup
    if (h_keys_in) delete[] h_keys_in;
    if (h_values_in) delete[] h_values_in;
    if (h_keys_reference) delete[] h_keys_reference;
    if (h_values_reference) delete[] h_values_reference;
    if (d_keys_in) CubDebugExit(g_allocator.DeviceFree(d_keys_in));
    if (d_values_in) CubDebugExit(g_allocator.DeviceFree(d_values_in));
}


/**
 * Test DeviceSelect on iterator type
 * /
template <
    Backend         BACKEND,
    typename        Key,
    typename        Value>
void TestIterator(
    int             num_items,
    char*           key_type_string,
    char*           value_type_string,
    Int2Type<true>  is_number)
{
    // Use a counting iterator as the input
    CountingInputIterator<T, int> h_in(0);

    // Allocate host arrays
    T*  h_reference = new T[num_items];

    // Initialize problem and solution
    int num_segments = Solve(h_in, h_reference, num_items);

    printf("\nIterator %s cub::DeviceReduce::ReduceByKey %d items, %d selected (avg run length %d), %s %d-byte elements\n",
        (BACKEND == CDP) ? "CDP CUB" : (BACKEND == THRUST) ? "Thrust" : "CUB",
        num_items, num_segments, num_items / num_segments,
        type_string,
        (int) sizeof(T));
    fflush(stdout);

    // Run Test
    Test<BACKEND>(h_in, h_reference, num_segments, num_items, key_type_string, value_type_string);

    // Cleanup
    if (h_reference) delete[] h_reference;
}


/ **
 * Test DeviceSelect on iterator type
 * /
template <
    Backend         BACKEND,
    typename        Key,
    typename        Value>
void TestIterator(
    int             num_items,
    char*           key_type_string,
    char*           value_type_string,
    Int2Type<false> is_number)
{}


/ **
 * Test different gen modes
 * /
template <
    Backend         BACKEND,
    typename        Key,
    typename        Value>
void Test(
    int             num_items,
    char*           type_string)
{
    for (int max_segment = 1; max_segment < CUB_MIN(num_items, (unsigned short) -1); max_segment *= 11)
    {
        TestPointer<BACKEND, Key, Value>(num_items, 0, max_segment, key_type_string, value_type_string);
        TestPointer<BACKEND, Key, Value>(num_items, 2, max_segment, key_type_string, value_type_string);
        TestPointer<BACKEND, Key, Value>(num_items, 7, max_segment, key_type_string, value_type_string);
    }

    TestIterator<BACKEND, Key, Value>(num_items, type_string, Int2Type<Traits<T>::CATEGORY != NOT_A_NUMBER>());

}


/ **
 * Test different dispatch
 * /
template <
    typename        Key,
    typename        Value>
void TestOp(
    int             num_items,
    char*           key_type_string,
    char*           value_type_string)
{
    Test<CUB, Key, Value>(num_items, key_type_string, value_type_string);
#ifdef CUB_CDP
    Test<CDP, Key, Value>(num_items, key_type_string, value_type_string);
#endif
}


/ **
 * Test different input sizes
 * /
template <
    typename        Key,
    typename        Value>
void Test(
    int             num_items,
    char*           key_type_string,
    char*           value_type_string)
{
    if (num_items < 0)
    {
        TestOp<Key, Value>(1,        key_type_string, value_type_string);
        TestOp<Key, Value>(100,      key_type_string, value_type_string);
        TestOp<Key, Value>(10000,    key_type_string, value_type_string);
        TestOp<Key, Value>(1000000,  key_type_string, value_type_string);
    }
    else
    {
        TestOp<Key, Value>(num_items, key_type_string, value_type_string);
    }
}
*/


//---------------------------------------------------------------------
// Main
//---------------------------------------------------------------------

/**
 * Main
 */
int main(int argc, char** argv)
{
    int num_items           = -1;
    int entropy_reduction   = 0;
    int maxseg              = 1000;

    // Initialize command line
    CommandLineArgs args(argc, argv);
    g_verbose = args.CheckCmdLineFlag("v");
    args.GetCmdLineArgument("n", num_items);
    args.GetCmdLineArgument("i", g_timing_iterations);
    args.GetCmdLineArgument("repeat", g_repeat);
    args.GetCmdLineArgument("maxseg", maxseg);
    args.GetCmdLineArgument("entropy", entropy_reduction);

    // Print usage
    if (args.CheckCmdLineFlag("help"))
    {
        printf("%s "
            "[--n=<input items> "
            "[--i=<timing iterations> "
            "[--device=<device-id>] "
            "[--maxseg=<max segment length>]"
            "[--entropy=<segment length bit entropy reduction rounds>]"
            "[--repeat=<repetitions of entire test suite>]"
            "[--v] "
            "[--cdp]"
            "\n", argv[0]);
        exit(0);
    }

    // Initialize device
    CubDebugExit(args.DeviceInit());
    printf("\n");

    // Get device ordinal
    int device_ordinal;
    CubDebugExit(cudaGetDevice(&device_ordinal));

    // Get device SM version
    int sm_version;
    CubDebugExit(SmVersion(sm_version, device_ordinal));

    TestPointer<CUB, int, int>(num_items, entropy_reduction, maxseg, cub::Sum(), CUB_TYPE_STRING(int), CUB_TYPE_STRING(int));

/*
#ifdef QUICK_TEST

    // Compile/run quick tests
    if (num_items < 0) num_items = 32000000;

    TestPointer<CUB, char>(        num_items * ((sm_version <= 130) ? 1 : 4), entropy_reduction, maxseg, CUB_TYPE_STRING(char));
    TestPointer<THRUST, char>(     num_items * ((sm_version <= 130) ? 1 : 4), entropy_reduction, maxseg, CUB_TYPE_STRING(char));

    printf("----------------------------\n");
    TestPointer<CUB, short>(       num_items * ((sm_version <= 130) ? 1 : 2), entropy_reduction, maxseg, CUB_TYPE_STRING(short));
    TestPointer<THRUST, short>(    num_items * ((sm_version <= 130) ? 1 : 2), entropy_reduction, maxseg, CUB_TYPE_STRING(short));

    printf("----------------------------\n");
    TestPointer<CUB, int>(         num_items,                                 entropy_reduction, maxseg, CUB_TYPE_STRING(int));
    TestPointer<THRUST, int>(      num_items,                                 entropy_reduction, maxseg, CUB_TYPE_STRING(int));

    printf("----------------------------\n");
    TestPointer<CUB, long long>(   num_items / 2,                             entropy_reduction, maxseg, CUB_TYPE_STRING(long long));
    TestPointer<THRUST, long long>(num_items / 2,                             entropy_reduction, maxseg, CUB_TYPE_STRING(long long));

    printf("----------------------------\n");
    TestPointer<CUB, TestFoo>(     num_items / 4,                             entropy_reduction, maxseg, CUB_TYPE_STRING(TestFoo));
    TestPointer<THRUST, TestFoo>(  num_items / 4,                             entropy_reduction, maxseg, CUB_TYPE_STRING(TestFoo));

#else

    // Compile/run thorough tests
    for (int i = 0; i <= g_repeat; ++i)
    {
        // Test different input types
        Test<unsigned char>(num_items, CUB_TYPE_STRING(unsigned char));
        Test<unsigned short>(num_items, CUB_TYPE_STRING(unsigned short));
        Test<unsigned int>(num_items, CUB_TYPE_STRING(unsigned int));
        Test<unsigned long long>(num_items, CUB_TYPE_STRING(unsigned long long));

        Test<uchar2>(num_items, CUB_TYPE_STRING(uchar2));
        Test<ushort2>(num_items, CUB_TYPE_STRING(ushort2));
        Test<uint2>(num_items, CUB_TYPE_STRING(uint2));
        Test<ulonglong2>(num_items, CUB_TYPE_STRING(ulonglong2));

        Test<uchar4>(num_items, CUB_TYPE_STRING(uchar4));
        Test<ushort4>(num_items, CUB_TYPE_STRING(ushort4));
        Test<uint4>(num_items, CUB_TYPE_STRING(uint4));
        Test<ulonglong4>(num_items, CUB_TYPE_STRING(ulonglong4));

        Test<TestFoo>(num_items, CUB_TYPE_STRING(TestFoo));
        Test<TestBar>(num_items, CUB_TYPE_STRING(TestBar));
    }

#endif
*/
    return 0;
}



