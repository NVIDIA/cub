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
 * Test of DeviceReduce::ReduceByKey utilities
 ******************************************************************************/

// Ensure printing of CUDA runtime errors to console
#define CUB_STDERR

#include <stdio.h>

#include <cub/util_allocator.cuh>
#include <cub/iterator/constant_input_iterator.cuh>
#include <cub/device/device_reduce.cuh>
#include <cub/thread/thread_operators.cuh>

#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/iterator/constant_iterator.h>

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
 * Dispatch to reduce-by-key entrypoint
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
    Offset                      num_items,
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


/**
 * Dispatch to run-length encode entrypoint
 */
template <
    typename                    KeyInputIterator,
    typename                    KeyOutputIterator,
    typename                    ValueOutputIterator,
    typename                    NumSegmentsIterator,
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
    ConstantInputIterator<typename std::iterator_traits<ValueOutputIterator>::value_type, Offset> d_values_in,
    ValueOutputIterator         d_values_out,
    NumSegmentsIterator         d_num_segments,
    cub::Equality               equality_op,
    cub::Sum                    reduction_op,
    Offset                      num_items,
    cudaStream_t                stream,
    bool                        debug_synchronous)
{
    cudaError_t error = cudaSuccess;
    for (int i = 0; i < timing_timing_iterations; ++i)
    {
        error = DeviceReduce::RunLengthEncode(
            d_temp_storage,
            temp_storage_bytes,
            d_keys_in,
            d_keys_out,
            d_values_out,
            d_num_segments,
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
 * Dispatch to reduce-by-key entrypoint
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
cudaError_t Dispatch(
    Int2Type<THRUST>            dispatch_to,
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
    Offset                      num_items,
    cudaStream_t                stream,
    bool                        debug_synchronous)
{
    typedef typename std::iterator_traits<KeyInputIterator>::value_type Key;
    typedef typename std::iterator_traits<ValueInputIterator>::value_type Value;

    if (d_temp_storage == 0)
    {
        temp_storage_bytes = 1;
    }
    else
    {
        thrust::device_ptr<Key> d_keys_in_wrapper(d_keys_in);
        thrust::device_ptr<Key> d_keys_out_wrapper(d_keys_out);

        thrust::device_ptr<Value> d_values_in_wrapper(d_values_in);
        thrust::device_ptr<Value> d_values_out_wrapper(d_values_out);

        thrust::pair<thrust::device_ptr<Key>, thrust::device_ptr<Value> > d_out_ends;

        for (int i = 0; i < timing_timing_iterations; ++i)
        {
            d_out_ends = thrust::reduce_by_key(
                d_keys_in_wrapper,
                d_keys_in_wrapper + num_items,
                d_values_in_wrapper,
                d_keys_out_wrapper,
                d_values_out_wrapper);
        }

        Offset num_segments = d_out_ends.first - d_keys_out_wrapper;
        CubDebugExit(cudaMemcpy(d_num_segments, &num_segments, sizeof(Offset), cudaMemcpyHostToDevice));

    }

    return cudaSuccess;
}


/**
 * Dispatch to run-length encode entrypoint
 */
template <
    typename                    KeyInputIterator,
    typename                    KeyOutputIterator,
    typename                    ValueOutputIterator,
    typename                    NumSegmentsIterator,
    typename                    Offset>
cudaError_t Dispatch(
    Int2Type<THRUST>            dispatch_to,
    int                         timing_timing_iterations,
    size_t                      *d_temp_storage_bytes,
    cudaError_t                 *d_cdp_error,

    void                        *d_temp_storage,
    size_t                      &temp_storage_bytes,
    KeyInputIterator            d_keys_in,
    KeyOutputIterator           d_keys_out,
    ConstantInputIterator<typename std::iterator_traits<ValueOutputIterator>::value_type, Offset> d_values_in,
    ValueOutputIterator         d_values_out,
    NumSegmentsIterator         d_num_segments,
    cub::Equality               equality_op,
    cub::Sum                    reduction_op,
    Offset                      num_items,
    cudaStream_t                stream,
    bool                        debug_synchronous)
{
    typedef typename std::iterator_traits<KeyInputIterator>::value_type Key;
    typedef typename std::iterator_traits<ValueOutputIterator>::value_type Value;

    if (d_temp_storage == 0)
    {
        temp_storage_bytes = 1;
    }
    else
    {
        thrust::device_ptr<Key>     d_keys_in_wrapper(d_keys_in);
        thrust::device_ptr<Key>     d_keys_out_wrapper(d_keys_out);
        thrust::device_ptr<Value>   d_values_out_wrapper(d_values_out);

        thrust::pair<thrust::device_ptr<Key>, thrust::device_ptr<Value> > d_out_ends;

        Value one_val;
        InitValue(INTEGER_SEED, one_val, 1);
        thrust::constant_iterator<Value> constant_one(one_val);

        for (int i = 0; i < timing_timing_iterations; ++i)
        {
            d_out_ends = thrust::reduce_by_key(
                d_keys_in_wrapper,
                d_keys_in_wrapper + num_items,
//                d_values_in,
                constant_one,
                d_keys_out_wrapper,
                d_values_out_wrapper);
        }

        Offset num_segments = d_out_ends.first - d_keys_out_wrapper;
        CubDebugExit(cudaMemcpy(d_num_segments, &num_segments, sizeof(Offset), cudaMemcpyHostToDevice));
    }

    return cudaSuccess;
}



//---------------------------------------------------------------------
// CUDA Nested Parallelism Test Kernel
//---------------------------------------------------------------------

/**
 * Simple wrapper kernel to invoke DeviceSelect
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
__global__ void CnpDispatchKernel(
    int                         timing_timing_iterations,
    size_t                      *d_temp_storage_bytes,
    cudaError_t                 *d_cdp_error,

    void                        *d_temp_storage,
    size_t                      temp_storage_bytes,
    KeyInputIterator            d_keys_in,
    KeyOutputIterator           d_keys_out,
    ValueInputIterator          d_values_in,
    ValueOutputIterator         d_values_out,
    NumSegmentsIterator         d_num_segments,
    EqualityOp                  equality_op,
    ReductionOp                 reduction_op,
    Offset                      num_items,
    cudaStream_t                stream,
    bool                        debug_synchronous)
{

#ifndef CUB_CDP
    *d_cdp_error = cudaErrorNotSupported;
#else
    *d_cdp_error = Dispatch(Int2Type<CUB>(), timing_timing_iterations, d_temp_storage_bytes, d_cdp_error,
        d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, d_values_in, d_values_out, d_num_segments, equality_op, reduction_op, num_items, 0, debug_synchronous);

    *d_temp_storage_bytes = temp_storage_bytes;
#endif
}


/**
 * Dispatch to CDP kernel
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
    Int2Type<CDP>               dispatch_to,
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
    Offset                      num_items,
    cudaStream_t                stream,
    bool                        debug_synchronous)
{
    // Invoke kernel to invoke device-side dispatch
    CnpDispatchKernel<<<1,1>>>(timing_timing_iterations, d_temp_storage_bytes, d_cdp_error,
        d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, d_values_in, d_values_out, d_num_segments, equality_op, reduction_op, num_items, 0, debug_synchronous);

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
 * Initialize problem.  Keys are initialized to segment number
 * and values are initialized to 1
 */
template <typename KeyIterator>
void Initialize(
    int             entropy_reduction,
    KeyIterator     h_keys_in,
    int             num_items,
    int             max_segment)
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
            InitValue(INTEGER_SEED, h_keys_in[j], segment_id);
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
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_values_out, sizeof(Value) * num_items));
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
    int compare1 = CompareDeviceResults(h_keys_reference, d_keys_out, num_segments, true, g_verbose);
    printf("\t Keys %s ", compare1 ? "FAIL" : "PASS");

    int compare2 = CompareDeviceResults(h_values_reference, d_values_out, num_segments, true, g_verbose);
    printf("\t Values %s ", compare2 ? "FAIL" : "PASS");

    int compare3 = CompareDeviceResults(&num_segments, d_num_segments, 1, true, g_verbose);
    printf("\t Count %s ", compare3 ? "FAIL" : "PASS");

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
    AssertEquals(0, compare1 | compare2 | compare3);
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

    for (int i = 0; i < num_items; ++i)
        InitValue(INTEGER_SEED, h_values_in[i], 1);

    // Initialize problem and solution
    Equality equality_op;
    Initialize(entropy_reduction, h_keys_in, num_items, max_segment);
    int num_segments = Solve(h_keys_in, h_keys_reference, h_values_in, h_values_reference, equality_op, reduction_op, num_items);

    printf("\nPointer %s cub::DeviceReduce::ReduceByKey %s reduction of %d items, %d segments (avg run length %d), {%s,%s} key value pairs, max_segment %d, entropy_reduction %d\n",
        (BACKEND == CDP) ? "CDP CUB" : (BACKEND == THRUST) ? "Thrust" : "CUB",
        (Equals<ReductionOp, Sum>::VALUE) ? "Sum" : "Max",
        num_items, num_segments, num_items / num_segments,
        key_type_string, value_type_string,
        max_segment, entropy_reduction);
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
 */
template <
    Backend         BACKEND,
    typename        Key,
    typename        Value,
    typename        ReductionOp>
void TestIterator(
    int             num_items,
    int             entropy_reduction,
    int             max_segment,
    ReductionOp     reduction_op,
    char*           key_type_string,
    char*           value_type_string,
    Int2Type<true>  is_primitive)
{
    // Allocate host arrays
    Key* h_keys_in        = new Key[num_items];
    Key* h_keys_reference = new Key[num_items];

    Value one_val;
    InitValue(INTEGER_SEED, one_val, 1);
    ConstantInputIterator<Value, int> h_values_in(one_val);
    Value* h_values_reference = new Value[num_items];

    // Initialize problem and solution
    Equality equality_op;
    Initialize(entropy_reduction, h_keys_in, num_items, max_segment);
    int num_segments = Solve(h_keys_in, h_keys_reference, h_values_in, h_values_reference, equality_op, reduction_op, num_items);

    printf("\nIterator %s cub::DeviceReduce::ReduceByKey %s reduction of %d items, %d segments (avg run length %d), {%s,%s} key value pairs, max_segment %d, entropy_reduction %d\n",
        (BACKEND == CDP) ? "CDP CUB" : (BACKEND == THRUST) ? "Thrust" : "CUB",
        (Equals<ReductionOp, Sum>::VALUE) ? "Sum" : "Max",
        num_items, num_segments, num_items / num_segments,
        key_type_string, value_type_string,
        max_segment, entropy_reduction);
    fflush(stdout);

    // Allocate problem device arrays
    Key     *d_keys_in = NULL;
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_keys_in, sizeof(Key) * num_items));

    // Initialize device input
    CubDebugExit(cudaMemcpy(d_keys_in, h_keys_in, sizeof(Key) * num_items, cudaMemcpyHostToDevice));

    // Run Test
    Test<BACKEND>(d_keys_in, h_values_in, h_keys_reference, h_values_reference, equality_op, reduction_op, num_segments, num_items, key_type_string, value_type_string);

    // Cleanup
    if (h_keys_in) delete[] h_keys_in;
    if (h_keys_reference) delete[] h_keys_reference;
    if (h_values_reference) delete[] h_values_reference;
    if (d_keys_in) CubDebugExit(g_allocator.DeviceFree(d_keys_in));
}

/**
 * Test DeviceSelect on iterator type
 */
template <
    Backend         BACKEND,
    typename        Key,
    typename        Value,
    typename        ReductionOp>
void TestIterator(
    int             num_items,
    int             entropy_reduction,
    int             max_segment,
    ReductionOp     reduction_op,
    char*           key_type_string,
    char*           value_type_string,
    Int2Type<false> is_primitive)
{}


/**
 * Test different gen modes
 */
template <
    Backend         BACKEND,
    typename        Key,
    typename        Value,
    typename        ReductionOp>
void Test(
    int             num_items,
    ReductionOp     reduction_op,
    char*           key_type_string,
    char*           value_type_string)
{
    // Evaluate different max-segment lengths
    for (int max_segment = 1; max_segment < CUB_MIN(num_items, (unsigned short) -1); max_segment *= 11)
    {
        // 0 key-bit entropy reduction rounds
        TestPointer<BACKEND, Key, Value>(num_items, 0, max_segment, reduction_op, key_type_string, value_type_string);
        TestIterator<BACKEND, Key, Value>(num_items, 0, max_segment, reduction_op, key_type_string, value_type_string, Int2Type<Traits<Value>::PRIMITIVE>());

        if (max_segment > 1)
        {
            // 2 key-bit entropy reduction rounds
            TestPointer<BACKEND, Key, Value>(num_items, 2, max_segment, reduction_op, key_type_string, value_type_string);
            TestIterator<BACKEND, Key, Value>(num_items, 2, max_segment, reduction_op, key_type_string, value_type_string, Int2Type<Traits<Value>::PRIMITIVE>());

            // 7 key-bit entropy reduction rounds
            TestPointer<BACKEND, Key, Value>(num_items, 7, max_segment, reduction_op, key_type_string, value_type_string);
            TestIterator<BACKEND, Key, Value>(num_items, 7, max_segment, reduction_op, key_type_string, value_type_string, Int2Type<Traits<Value>::PRIMITIVE>());
        }
    }
}


/**
 * Test different dispatch
 */
template <
    typename        Key,
    typename        Value,
    typename        ReductionOp>
void TestDispatch(
    int             num_items,
    ReductionOp     reduction_op,
    char*           key_type_string,
    char*           value_type_string)
{
    Test<CUB, Key, Value>(num_items, reduction_op, key_type_string, value_type_string);
#ifdef CUB_CDP
    Test<CDP, Key, Value>(num_items, reduction_op, key_type_string, value_type_string);
#endif
}


/**
 * Test different input sizes
 */
template <
    typename        Key,
    typename        Value,
    typename        ReductionOp>
void TestSize(
    int             num_items,
    ReductionOp     reduction_op,
    char*           key_type_string,
    char*           value_type_string)
{
    if (num_items < 0)
    {
        TestDispatch<Key, Value>(1,        reduction_op, key_type_string, value_type_string);
        TestDispatch<Key, Value>(100,      reduction_op, key_type_string, value_type_string);
        TestDispatch<Key, Value>(10000,    reduction_op, key_type_string, value_type_string);
        TestDispatch<Key, Value>(1000000,  reduction_op, key_type_string, value_type_string);
    }
    else
    {
        TestDispatch<Key, Value>(num_items, reduction_op, key_type_string, value_type_string);
    }

}


template <
    typename        Key,
    typename        Value>
void TestOp(
    int             num_items,
    char*           key_type_string,
    char*           value_type_string)
{
    TestSize<Key, Value>(num_items, cub::Sum(), key_type_string, value_type_string);
    TestSize<Key, Value>(num_items, cub::Max(), key_type_string, value_type_string);
}



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

    // Get ptx version
    int ptx_version;
    CubDebugExit(PtxVersion(ptx_version));

#ifdef QUICK_TEST

    // Compile/run quick tests
    if (num_items < 0) num_items = 32000000;

    printf("---- RLE int ---- \n");
    TestIterator<CUB, int, int>(num_items, entropy_reduction, maxseg, cub::Sum(), CUB_TYPE_STRING(int), CUB_TYPE_STRING(int), Int2Type<Traits<int>::PRIMITIVE>());
    TestIterator<THRUST, int, int>(num_items, entropy_reduction, maxseg, cub::Sum(), CUB_TYPE_STRING(int), CUB_TYPE_STRING(int), Int2Type<Traits<int>::PRIMITIVE>());

    printf("---- RLE long long ---- \n");
    TestIterator<CUB, long long, int>(num_items, entropy_reduction, maxseg, cub::Sum(), CUB_TYPE_STRING(long long), CUB_TYPE_STRING(int), Int2Type<Traits<int>::PRIMITIVE>());
    TestIterator<THRUST, long long, int>(num_items, entropy_reduction, maxseg, cub::Sum(), CUB_TYPE_STRING(long long), CUB_TYPE_STRING(int), Int2Type<Traits<int>::PRIMITIVE>());

    printf("---- int ---- \n");
    TestPointer<CUB, int, int>(num_items, entropy_reduction, maxseg, cub::Sum(), CUB_TYPE_STRING(int), CUB_TYPE_STRING(int));
    TestPointer<THRUST, int, int>(num_items, entropy_reduction, maxseg, cub::Sum(), CUB_TYPE_STRING(int), CUB_TYPE_STRING(int));

    printf("---- float ---- \n");
    TestPointer<CUB, int, float>(num_items, entropy_reduction, maxseg, cub::Sum(), CUB_TYPE_STRING(int), CUB_TYPE_STRING(float));
    TestPointer<THRUST, int, float>(num_items, entropy_reduction, maxseg, cub::Sum(), CUB_TYPE_STRING(int), CUB_TYPE_STRING(float));

    if (ptx_version > 100)                          // Don't check doubles on PTX100 because they're down-converted
    {
        printf("---- double ---- \n");
        TestPointer<CUB, int, double>(num_items, entropy_reduction, maxseg, cub::Sum(), CUB_TYPE_STRING(int), CUB_TYPE_STRING(double));
        TestPointer<THRUST, int, double>(num_items, entropy_reduction, maxseg, cub::Sum(), CUB_TYPE_STRING(int), CUB_TYPE_STRING(double));
    }

#else

    // Compile/run thorough tests
    for (int i = 0; i <= g_repeat; ++i)
    {

        // Test different input types

        TestOp<int, char>(num_items, CUB_TYPE_STRING(int), CUB_TYPE_STRING(int));
        TestOp<int, short>(num_items, CUB_TYPE_STRING(int), CUB_TYPE_STRING(short));
        TestOp<int, int>(num_items, CUB_TYPE_STRING(int), CUB_TYPE_STRING(int));
        TestOp<int, long>(num_items, CUB_TYPE_STRING(int), CUB_TYPE_STRING(long));
        TestOp<int, long long>(num_items, CUB_TYPE_STRING(int), CUB_TYPE_STRING(long long));
        TestOp<int, float>(num_items, CUB_TYPE_STRING(int), CUB_TYPE_STRING(float));
        if (ptx_version > 100)                          // Don't check doubles on PTX100 because they're down-converted
            TestOp<int, double>(num_items, CUB_TYPE_STRING(int), CUB_TYPE_STRING(double));

        TestOp<int, uchar2>(num_items, CUB_TYPE_STRING(int), CUB_TYPE_STRING(uchar2));
        TestOp<int, uint2>(num_items, CUB_TYPE_STRING(int), CUB_TYPE_STRING(uint2));
        TestOp<int, uint3>(num_items, CUB_TYPE_STRING(int), CUB_TYPE_STRING(uint3));
        TestOp<int, uint4>(num_items, CUB_TYPE_STRING(int), CUB_TYPE_STRING(uint4));
        TestOp<int, ulonglong4>(num_items, CUB_TYPE_STRING(int), CUB_TYPE_STRING(ulonglong4));

        TestOp<int, TestFoo>(num_items, CUB_TYPE_STRING(int), CUB_TYPE_STRING(TestFoo));
        TestOp<int, TestBar>(num_items, CUB_TYPE_STRING(int), CUB_TYPE_STRING(TestBar));

        TestOp<char, int>(num_items, CUB_TYPE_STRING(char), CUB_TYPE_STRING(int));
        TestOp<long long, int>(num_items, CUB_TYPE_STRING(long long), CUB_TYPE_STRING(int));
        TestOp<TestFoo, int>(num_items, CUB_TYPE_STRING(TestFoo), CUB_TYPE_STRING(int));
        TestOp<TestBar, int>(num_items, CUB_TYPE_STRING(TestBar), CUB_TYPE_STRING(int));
    }

#endif

    return 0;
}



