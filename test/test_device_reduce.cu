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
 * Test of DeviceReduce utilities
 ******************************************************************************/

// Ensure printing of CUDA runtime errors to console
#define CUB_STDERR

#include <stdio.h>

#include <cub/util_allocator.cuh>
#include <cub/device/device_reduce.cuh>
#include <cub/iterator/constant_input_iterator.cuh>

#include <thrust/device_ptr.h>
#include <thrust/reduce.h>

#include "test_util.h"

using namespace cub;


//---------------------------------------------------------------------
// Globals, constants and typedefs
//---------------------------------------------------------------------

bool                    g_verbose           = false;
int                     g_timing_iterations = 0;
int                     g_repeat            = 0;
CachingDeviceAllocator  g_allocator(true);


// Custom max functor
struct CustomMax
{
    /// Boolean max operator, returns <tt>(a > b) ? a : b</tt>
    template <typename T>
    __host__ __device__ __forceinline__ T operator()(const T &a, const T &b)
    {
        return CUB_MAX(a, b);
    }
};


//---------------------------------------------------------------------
// Dispatch to different CUB DeviceReduce entrypoints
//---------------------------------------------------------------------

/**
 * Dispatch to reduce entrypoint
 */
template <typename InputIterator, typename OutputIterator, typename ReductionOp>
__host__ __device__ __forceinline__
cudaError_t Dispatch(
    Int2Type<CUB>       dispatch_to,
    int                 timing_timing_iterations,
    size_t              *d_temp_storage_bytes,
    cudaError_t         *d_cdp_error,

    void                *d_temp_storage,
    size_t              &temp_storage_bytes,
    InputIterator       d_in,
    OutputIterator      d_out,
    int                 num_items,
    ReductionOp         reduction_op,
    cudaStream_t        stream,
    bool                debug_synchronous)
{
    // Invoke kernel to device reduction directly
    cudaError_t error = cudaSuccess;
    for (int i = 0; i < timing_timing_iterations; ++i)
    {
        error = DeviceReduce::Reduce(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items, reduction_op, 0, debug_synchronous);
    }
    return error;
}

/**
 * Dispatch to sum entrypoint
 */
template <typename InputIterator, typename OutputIterator>
__host__ __device__ __forceinline__
cudaError_t Dispatch(
    Int2Type<CUB>       dispatch_to,
    int                 timing_timing_iterations,
    size_t              *d_temp_storage_bytes,
    cudaError_t         *d_cdp_error,

    void                *d_temp_storage,
    size_t              &temp_storage_bytes,
    InputIterator       d_in,
    OutputIterator      d_out,
    int                 num_items,
    cub::Sum            reduction_op,
    cudaStream_t        stream,
    bool                debug_synchronous)
{
    // Invoke kernel to device reduction directly
    cudaError_t error = cudaSuccess;
    for (int i = 0; i < timing_timing_iterations; ++i)
    {
        error = DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items, 0, debug_synchronous);
    }
    return error;
}

/**
 * Dispatch to min entrypoint
 */
template <typename InputIterator, typename OutputIterator>
__host__ __device__ __forceinline__
cudaError_t Dispatch(
    Int2Type<CUB>       dispatch_to,
    int                 timing_timing_iterations,
    size_t              *d_temp_storage_bytes,
    cudaError_t         *d_cdp_error,

    void                *d_temp_storage,
    size_t              &temp_storage_bytes,
    InputIterator       d_in,
    OutputIterator      d_out,
    int                 num_items,
    cub::Min            reduction_op,
    cudaStream_t        stream,
    bool                debug_synchronous)
{
    // Invoke kernel to device reduction directly
    cudaError_t error = cudaSuccess;
    for (int i = 0; i < timing_timing_iterations; ++i)
    {
        error = DeviceReduce::Min(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items, 0, debug_synchronous);
    }
    return error;
}

/**
 * Dispatch to max entrypoint
 */
template <typename InputIterator, typename OutputIterator>
__host__ __device__ __forceinline__
cudaError_t Dispatch(
    Int2Type<CUB>       dispatch_to,
    int                 timing_timing_iterations,
    size_t              *d_temp_storage_bytes,
    cudaError_t         *d_cdp_error,

    void                *d_temp_storage,
    size_t              &temp_storage_bytes,
    InputIterator       d_in,
    OutputIterator      d_out,
    int                 num_items,
    cub::Max            reduction_op,
    cudaStream_t        stream,
    bool                debug_synchronous)
{
    // Invoke kernel to device reduction directly
    cudaError_t error = cudaSuccess;
    for (int i = 0; i < timing_timing_iterations; ++i)
    {
        error = DeviceReduce::Max(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items, 0, debug_synchronous);
    }
    return error;
}

/**
 * Dispatch to argmin entrypoint
 */
template <typename InputIterator, typename OutputIterator>
__host__ __device__ __forceinline__
cudaError_t Dispatch(
    Int2Type<CUB>       dispatch_to,
    int                 timing_timing_iterations,
    size_t              *d_temp_storage_bytes,
    cudaError_t         *d_cdp_error,

    void                *d_temp_storage,
    size_t              &temp_storage_bytes,
    InputIterator       d_in,
    OutputIterator      d_out,
    int                 num_items,
    cub::ArgMin         reduction_op,
    cudaStream_t        stream,
    bool                debug_synchronous)
{
    // Invoke kernel to device reduction directly
    cudaError_t error = cudaSuccess;
    for (int i = 0; i < timing_timing_iterations; ++i)
    {
        error = DeviceReduce::ArgMin(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items, 0, debug_synchronous);
    }
    return error;
}

/**
 * Dispatch to argmax entrypoint
 */
template <typename InputIterator, typename OutputIterator>
__host__ __device__ __forceinline__
cudaError_t Dispatch(
    Int2Type<CUB>       dispatch_to,
    int                 timing_timing_iterations,
    size_t              *d_temp_storage_bytes,
    cudaError_t         *d_cdp_error,

    void                *d_temp_storage,
    size_t              &temp_storage_bytes,
    InputIterator       d_in,
    OutputIterator      d_out,
    int                 num_items,
    cub::ArgMax         reduction_op,
    cudaStream_t        stream,
    bool                debug_synchronous)
{
    // Invoke kernel to device reduction directly
    cudaError_t error = cudaSuccess;
    for (int i = 0; i < timing_timing_iterations; ++i)
    {
        error = DeviceReduce::ArgMax(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items, 0, debug_synchronous);
    }
    return error;
}


//---------------------------------------------------------------------
// Dispatch to different Thrust entrypoints
//---------------------------------------------------------------------

/**
 * Dispatch to reduction entrypoint (min or max specialization)
 */
template <typename InputIterator, typename OutputIterator, typename ReductionOp>
cudaError_t Dispatch(
    Int2Type<THRUST>    dispatch_to,
    int                 timing_timing_iterations,
    size_t              *d_temp_storage_bytes,
    cudaError_t         *d_cdp_error,

    void                *d_temp_storage,
    size_t              &temp_storage_bytes,
    InputIterator       d_in,
    OutputIterator      d_out,
    int                 num_items,
    ReductionOp         reduction_op,
    cudaStream_t        stream,
    bool                debug_synchronous)
{
    typedef typename std::iterator_traits<InputIterator>::value_type T;

    if (d_temp_storage == 0)
    {
        temp_storage_bytes = 1;
    }
    else
    {
        T init;
        CubDebugExit(cudaMemcpy(&init, d_in + 0, sizeof(T), cudaMemcpyDeviceToHost));

        thrust::device_ptr<T> d_in_wrapper(d_in);
        T retval;
        for (int i = 0; i < timing_timing_iterations; ++i)
        {
            retval = thrust::reduce(d_in_wrapper, d_in_wrapper + num_items, init, reduction_op);
        }

        CubDebugExit(cudaMemcpy(d_out, &retval, sizeof(T), cudaMemcpyHostToDevice));
    }

    return cudaSuccess;
}

/**
 * Dispatch to reduction entrypoint (sum specialization)
 */
template <typename InputIterator, typename OutputIterator>
cudaError_t Dispatch(
    Int2Type<THRUST>    dispatch_to,
    int                 timing_timing_iterations,
    size_t              *d_temp_storage_bytes,
    cudaError_t         *d_cdp_error,

    void                *d_temp_storage,
    size_t              &temp_storage_bytes,
    InputIterator       d_in,
    OutputIterator      d_out,
    int                 num_items,
    Sum                 reduction_op,
    cudaStream_t        stream,
    bool                debug_synchronous)
{
    typedef typename std::iterator_traits<InputIterator>::value_type T;

    if (d_temp_storage == 0)
    {
        temp_storage_bytes = 1;
    }
    else
    {
        thrust::device_ptr<T> d_in_wrapper(d_in);
        T retval;
        for (int i = 0; i < timing_timing_iterations; ++i)
        {
            retval = thrust::reduce(d_in_wrapper, d_in_wrapper + num_items);
        }

        CubDebugExit(cudaMemcpy(d_out, &retval, sizeof(T), cudaMemcpyHostToDevice));
    }

    return cudaSuccess;
}





//---------------------------------------------------------------------
// CUDA nested-parallelism test kernel
//---------------------------------------------------------------------

/**
 * Simple wrapper kernel to invoke DeviceReduce
 */
template <
    typename            InputIterator,
    typename            OutputIterator,
    typename            ReductionOp>
__global__ void CnpDispatchKernel(
    int                 timing_timing_iterations,
    size_t              *d_temp_storage_bytes,
    cudaError_t         *d_cdp_error,

    void                *d_temp_storage,
    size_t              temp_storage_bytes,
    InputIterator       d_in,
    OutputIterator      d_out,
    int                 num_items,
    ReductionOp         reduction_op,
    bool                debug_synchronous)
{
#ifndef CUB_CDP
    *d_cdp_error = cudaErrorNotSupported;
#else
    *d_cdp_error = Dispatch(Int2Type<CUB>(), timing_timing_iterations, d_temp_storage_bytes, d_cdp_error, d_temp_storage, temp_storage_bytes, d_in, d_out, num_items, reduction_op, 0, debug_synchronous);
    *d_temp_storage_bytes = temp_storage_bytes;
#endif
}


/**
 * Dispatch to CDP kernel
 */
template <typename InputIterator, typename OutputIterator, typename ReductionOp>
__host__ __device__ __forceinline__
cudaError_t Dispatch(
    Int2Type<CDP>       dispatch_to,
    int                 timing_timing_iterations,
    size_t              *d_temp_storage_bytes,
    cudaError_t         *d_cdp_error,

    void                *d_temp_storage,
    size_t              &temp_storage_bytes,
    InputIterator       d_in,
    OutputIterator      d_out,
    int                 num_items,
    ReductionOp         reduction_op,
    cudaStream_t        stream,
    bool                debug_synchronous)
{
    // Invoke kernel to invoke device-side dispatch
    CnpDispatchKernel<<<1,1>>>(timing_timing_iterations, d_temp_storage_bytes, d_cdp_error, d_temp_storage, temp_storage_bytes, d_in, d_out, num_items, reduction_op, debug_synchronous);

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
    GenMode         gen_mode,
    T               *h_in,
    int             num_items)
{
    for (int i = 0; i < num_items; ++i)
        InitValue(gen_mode, h_in[i], i);

    if (g_verbose)
    {
        printf("Input:\n");
        DisplayResults(h_in, num_items);
        printf("\n\n");
    }
}


/**
 * Compute solution
 */
template <
    typename        InputIterator,
    typename        T,
    typename        ReductionOp>
void Solve(
    InputIterator   h_in,
    T               &h_reference,
    ReductionOp     reduction_op,
    int             num_items)
{
    for (int i = 0; i < num_items; ++i)
    {
        if (i == 0)
            h_reference = h_in[0];
        else
            h_reference = reduction_op(h_reference, h_in[i]);
    }
}


/**
 * Test DeviceReduce for a given problem input
 */
template <
    Backend     BACKEND,
    typename    DeviceInputIterator,
    typename    T,
    typename    ReductionOp>
void Test(
    DeviceInputIterator     d_in,
    T                       &h_reference,
    int                     num_items,
    ReductionOp             reduction_op)
{
    // Allocate device output array
    T *d_out = NULL;
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_out, sizeof(T) * 1));

    // Allocate CDP device arrays for temp storage size and error
    size_t          *d_temp_storage_bytes = NULL;
    cudaError_t     *d_cdp_error = NULL;
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_temp_storage_bytes,  sizeof(size_t) * 1));
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_cdp_error,           sizeof(cudaError_t) * 1));

    // Request and allocate temporary storage
    void            *d_temp_storage = NULL;
    size_t          temp_storage_bytes = 0;
    CubDebugExit(Dispatch(Int2Type<BACKEND>(), 1, d_temp_storage_bytes, d_cdp_error, d_temp_storage, temp_storage_bytes, d_in, d_out, num_items, reduction_op, 0, true));
    CubDebugExit(g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes));

    // Clear device output
    CubDebugExit(cudaMemset(d_out, 0, sizeof(T) * 1));

    // Run warmup/correctness iteration
    CubDebugExit(Dispatch(Int2Type<BACKEND>(), 1, d_temp_storage_bytes, d_cdp_error, d_temp_storage, temp_storage_bytes, d_in, d_out, num_items, reduction_op, 0, true));

    // Check for correctness (and display results, if specified)
    int compare = CompareDeviceResults(&h_reference, d_out, 1, g_verbose, g_verbose);
    printf("\t%s", compare ? "FAIL" : "PASS");

    // Flush any stdout/stderr
    fflush(stdout);
    fflush(stderr);

    // Performance
    GpuTimer gpu_timer;
    gpu_timer.Start();
    CubDebugExit(Dispatch(Int2Type<BACKEND>(), g_timing_iterations, d_temp_storage_bytes, d_cdp_error, d_temp_storage, temp_storage_bytes, d_in, d_out, num_items, reduction_op, 0, false));
    gpu_timer.Stop();
    float elapsed_millis = gpu_timer.ElapsedMillis();

    // Display performance
    if (g_timing_iterations > 0)
    {
        float avg_millis = elapsed_millis / g_timing_iterations;
        float grate = float(num_items) / avg_millis / 1000.0 / 1000.0;
        float gbandwidth = grate * sizeof(T);
        printf(", %.3f avg ms, %.3f billion items/s, %.3f logical GB/s", avg_millis, grate, gbandwidth);
    }

    if (d_out) CubDebugExit(g_allocator.DeviceFree(d_out));
    if (d_temp_storage_bytes) CubDebugExit(g_allocator.DeviceFree(d_temp_storage_bytes));
    if (d_cdp_error) CubDebugExit(g_allocator.DeviceFree(d_cdp_error));
    if (d_temp_storage) CubDebugExit(g_allocator.DeviceFree(d_temp_storage));

    // Correctness asserts
    AssertEquals(0, compare);
}


/**
 * Test DeviceReduce on pointer type
 */
template <
    Backend     BACKEND,
    typename    T,
    typename    ReductionOp>
void TestPointer(
    int         num_items,
    GenMode     gen_mode,
    ReductionOp reduction_op,
    char*       type_string)
{
    printf("\n\nPointer %s cub::DeviceReduce::%s %d items, %s %d-byte elements, gen-mode %s\n",
        (BACKEND == CDP) ? "CDP CUB" : (BACKEND == THRUST) ? "Thrust" : "CUB",
        (Equals<ReductionOp, Sum>::VALUE) ? "Sum" : (Equals<ReductionOp, Min>::VALUE) ? "Min" : "Max",
        num_items, type_string, (int) sizeof(T),
        (gen_mode == RANDOM) ? "RANDOM" : (gen_mode == INTEGER_SEED) ? "SEQUENTIAL" : "HOMOGENOUS");
    fflush(stdout);

    // Allocate host arrays
    T* h_in = new T[num_items];
    T  h_reference;

    // Initialize problem and solution
    Initialize(gen_mode, h_in, num_items);
    Solve(h_in, h_reference, reduction_op, num_items);

    // Allocate problem device arrays
    T *d_in = NULL;
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_in, sizeof(T) * num_items));

    // Initialize device input
    CubDebugExit(cudaMemcpy(d_in, h_in, sizeof(T) * num_items, cudaMemcpyHostToDevice));

    // Run test
    Test<BACKEND>(d_in, h_reference, num_items, reduction_op);

    // Cleanup
    if (h_in) delete[] h_in;
    if (d_in) CubDebugExit(g_allocator.DeviceFree(d_in));
}


/**
 * Test DeviceReduce on pointer type (argmin specialization)
 */
template <
    Backend     BACKEND,
    typename    T>
void TestPointer(
    int         num_items,
    GenMode     gen_mode,
    ArgMin      reduction_op,
    char*       type_string)
{
    printf("\n\nPointer %s cub::DeviceReduce::%s %d items, %s %d-byte elements, gen-mode %s\n",
        (BACKEND == CDP) ? "CDP CUB" : (BACKEND == THRUST) ? "Thrust" : "CUB",
        "ArgMin", num_items, type_string, (int) sizeof(T),
        (gen_mode == RANDOM) ? "RANDOM" : (gen_mode == INTEGER_SEED) ? "SEQUENTIAL" : "HOMOGENOUS");
    fflush(stdout);

    // Allocate host arrays
    T* h_in = new T[num_items];
    ItemOffsetPair<T, int> h_reference;

    // Initialize problem and solution
    Initialize(gen_mode, h_in, num_items);
    Solve(h_in, h_reference.value, Min(), num_items);
    for (int i = 0; i < num_items; ++i)
    {
        if (!(h_reference.value != h_in[i]))
        {
            h_reference.offset = i;
            break;
        }
    }

    // Allocate problem device arrays
    T *d_in = NULL;
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_in, sizeof(T) * num_items));

    // Initialize device input
    CubDebugExit(cudaMemcpy(d_in, h_in, sizeof(T) * num_items, cudaMemcpyHostToDevice));

    // Run test
    Test<BACKEND>(d_in, h_reference, num_items, reduction_op);

    // Cleanup
    if (h_in) delete[] h_in;
    if (d_in) CubDebugExit(g_allocator.DeviceFree(d_in));
}


/**
 * Test DeviceReduce on pointer type (argmax specialization)
 */
template <
    Backend     BACKEND,
    typename    T>
void TestPointer(
    int         num_items,
    GenMode     gen_mode,
    ArgMax      reduction_op,
    char*       type_string)
{
    printf("\n\nPointer %s cub::DeviceReduce::%s %d items, %s %d-byte elements, gen-mode %s\n",
        (BACKEND == CDP) ? "CDP CUB" : (BACKEND == THRUST) ? "Thrust" : "CUB",
        "ArgMax", num_items, type_string, (int) sizeof(T),
        (gen_mode == RANDOM) ? "RANDOM" : (gen_mode == INTEGER_SEED) ? "SEQUENTIAL" : "HOMOGENOUS");
    fflush(stdout);

    // Allocate host arrays
    T* h_in = new T[num_items];
    ItemOffsetPair<T, int> h_reference;

    // Initialize problem and solution
    Initialize(gen_mode, h_in, num_items);
    Solve(h_in, h_reference.value, Max(), num_items);
    for (int i = 0; i < num_items; ++i)
    {
        if (!(h_reference.value != h_in[i]))
        {
            h_reference.offset = i;
            break;
        }
    }

    // Allocate problem device arrays
    T *d_in = NULL;
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_in, sizeof(T) * num_items));

    // Initialize device input
    CubDebugExit(cudaMemcpy(d_in, h_in, sizeof(T) * num_items, cudaMemcpyHostToDevice));

    // Run test
    Test<BACKEND>(d_in, h_reference, num_items, reduction_op);

    // Cleanup
    if (h_in) delete[] h_in;
    if (d_in) CubDebugExit(g_allocator.DeviceFree(d_in));
}


/**
 * Test DeviceReduce on iterator type
 */
template <
    Backend     BACKEND,
    typename    T,
    typename    ReductionOp>
void TestIterator(
    int         num_items,
    ReductionOp reduction_op,
    char*       type_string)
{
    printf("\n\nIterator %s cub::DeviceReduce::%s %d items, %s %d-byte elements\n",
        (BACKEND == CDP) ? "CDP CUB" : (BACKEND == THRUST) ? "Thrust" : "CUB",
        (Equals<ReductionOp, Sum>::VALUE) ? "Sum" : (Equals<ReductionOp, Min>::VALUE) ? "Min" : "Max",
        num_items, type_string, (int) sizeof(T));
    fflush(stdout);

    // Use a constant iterator as the input
    T val = T();
    ConstantInputIterator<T, int> h_in(val);
    T  h_reference;

    // Initialize problem and solution
    Solve(h_in, h_reference, reduction_op, num_items);

    // Run test
    Test<BACKEND>(h_in, h_reference, num_items, reduction_op);
}


/**
 * Test DeviceReduce on iterator type (argmin specialization)
 */
template <
    Backend     BACKEND,
    typename    T>
void TestIterator(
    int         num_items,
    ArgMin      reduction_op,
    char*       type_string)
{
    printf("\n\nIterator %s cub::DeviceReduce::%s %d items, %s %d-byte elements\n",
        (BACKEND == CDP) ? "CDP CUB" : (BACKEND == THRUST) ? "Thrust" : "CUB",
        "ArgMin",
        num_items, type_string, (int) sizeof(T));
    fflush(stdout);

    // Use a constant iterator as the input
    T val = T();
    ConstantInputIterator<T, int> h_in(val);
    ItemOffsetPair<T, int> h_reference;

    // Initialize problem and solution
    Solve(h_in, h_reference.value, Min(), num_items);
    for (int i = 0; i < num_items; ++i)
    {
        if (!(h_reference.value != h_in[i]))
        {
            h_reference.offset = i;
            break;
        }
    }

    // Run test
    Test<BACKEND>(h_in, h_reference, num_items, reduction_op);
}


/**
 * Test DeviceReduce on iterator type (argmax specialization)
 */
template <
    Backend     BACKEND,
    typename    T>
void TestIterator(
    int         num_items,
    ArgMax      reduction_op,
    char*       type_string)
{
    printf("\n\nIterator %s cub::DeviceReduce::%s %d items, %s %d-byte elements\n",
        (BACKEND == CDP) ? "CDP CUB" : (BACKEND == THRUST) ? "Thrust" : "CUB",
        "ArgMax",
        num_items, type_string, (int) sizeof(T));
    fflush(stdout);

    // Use a constant iterator as the input
    T val = T();
    ConstantInputIterator<T, int> h_in(val);
    ItemOffsetPair<T, int> h_reference;

    // Initialize problem and solution
    Solve(h_in, h_reference.value, Max(), num_items);
    for (int i = 0; i < num_items; ++i)
    {
        if (!(h_reference.value != h_in[i]))
        {
            h_reference.offset = i;
            break;
        }
    }

    // Run test
    Test<BACKEND>(h_in, h_reference, num_items, reduction_op);
}


/**
 * Test different gen modes
 */
template <
    Backend         BACKEND,
    typename        T,
    typename        ReductionOp>
void Test(
    int             num_items,
    ReductionOp     reduction_op,
    char*           type_string)
{

    TestPointer<BACKEND, T>(num_items, UNIFORM, reduction_op, type_string);
    TestPointer<BACKEND, T>(num_items, INTEGER_SEED, reduction_op, type_string);
    TestPointer<BACKEND, T>(num_items, RANDOM, reduction_op, type_string);

    TestIterator<BACKEND, T>(num_items, reduction_op, type_string);
}


/**
 * Test different dispatch
 */
template <
    typename    T,
    typename    ReductionOp>
void Test(
    int         num_items,
    ReductionOp reduction_op,
    char*       type_string)
{
    Test<CUB, T>(num_items, reduction_op, type_string);
#ifdef CUB_CDP
    Test<CDP, T>(num_items, reduction_op, type_string);
#endif
}


/**
 * Test different operators
 */
template <
    typename        T>
void TestOp(
    int             num_items,
    char*           type_string)
{
    Test<T>(num_items, CustomMax(), type_string);
    Test<T>(num_items, Sum(), type_string);
    Test<T>(num_items, Min(), type_string);
    Test<T>(num_items, ArgMin(), type_string);
    Test<T>(num_items, Max(), type_string);
    Test<T>(num_items, ArgMax(), type_string);
}


/**
 * Test different input sizes
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
    printf("\n");

#ifdef QUICK_TEST

    // Compile/run quick tests
    if (num_items < 0) num_items = 32000000;

    TestPointer<CUB, char>(         num_items * 4, UNIFORM, Sum(), CUB_TYPE_STRING(char));
    TestPointer<THRUST, char>(      num_items * 4, UNIFORM, Sum(), CUB_TYPE_STRING(char));

    printf("\n----------------------------\n");
    TestPointer<CUB, short>(        num_items * 2, UNIFORM, Sum(), CUB_TYPE_STRING(short));
    TestPointer<THRUST, short>(     num_items * 2, UNIFORM, Sum(), CUB_TYPE_STRING(short));

    printf("\n----------------------------\n");
    TestPointer<CUB, int>(          num_items,     UNIFORM, Sum(), CUB_TYPE_STRING(int));
    TestPointer<THRUST, int>(       num_items,     UNIFORM, Sum(), CUB_TYPE_STRING(int));

    printf("\n----------------------------\n");
    TestPointer<CUB, long long>(    num_items / 2, UNIFORM, Sum(), CUB_TYPE_STRING(long long));
    TestPointer<THRUST, long long>( num_items / 2, UNIFORM, Sum(), CUB_TYPE_STRING(long long));

    printf("\n----------------------------\n");
    TestPointer<CUB, TestFoo>(      num_items / 4, UNIFORM, Max(), CUB_TYPE_STRING(TestFoo));
    TestPointer<THRUST, TestFoo>(   num_items / 4, UNIFORM, Max(), CUB_TYPE_STRING(TestFoo));

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
        Test<uint2>(num_items, CUB_TYPE_STRING(uint2));
        Test<ulonglong2>(num_items, CUB_TYPE_STRING(ulonglong2));
        Test<ulonglong4>(num_items, CUB_TYPE_STRING(ulonglong4));

        Test<TestFoo>(num_items, CUB_TYPE_STRING(TestFoo));
        Test<TestBar>(num_items, CUB_TYPE_STRING(TestBar));
    }

#endif

    return 0;
}



