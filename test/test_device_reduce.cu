/******************************************************************************
 * Copyright (c) 2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2015, NVIDIA CORPORATION.  All rights reserved.
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

int                     g_ptx_version;
int                     g_sm_count;
bool                    g_verbose           = false;
int                     g_timing_iterations = 0;
int                     g_repeat            = 0;
CachingDeviceAllocator  g_allocator(true);


// Dispatch types
enum Backend
{
    CUB,            // CUB method
    CUB_SEGMENTED,  // CUB segmented method
    CUB_CDP,        // GPU-based (dynamic parallelism) dispatch to CUB method
    THRUST,         // Thrust method
};


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
template <typename InputIteratorT, typename OutputIteratorT, typename ReductionOpT>
CUB_RUNTIME_FUNCTION __forceinline__
cudaError_t Dispatch(
    Int2Type<CUB>       dispatch_to,
    int                 timing_timing_iterations,
    size_t              *d_temp_storage_bytes,
    cudaError_t         *d_cdp_error,

    void*               d_temp_storage,
    size_t&             temp_storage_bytes,
    InputIteratorT      d_in,
    OutputIteratorT     d_out,
    int                 num_items,
    int                 max_segments,
    int                 *d_segment_offsets,
    ReductionOpT         reduction_op,
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
template <typename InputIteratorT, typename OutputIteratorT>
CUB_RUNTIME_FUNCTION __forceinline__
cudaError_t Dispatch(
    Int2Type<CUB>       dispatch_to,
    int                 timing_timing_iterations,
    size_t              *d_temp_storage_bytes,
    cudaError_t         *d_cdp_error,

    void*               d_temp_storage,
    size_t&             temp_storage_bytes,
    InputIteratorT      d_in,
    OutputIteratorT     d_out,
    int                 num_items,
    int                 max_segments,
    int                 *d_segment_offsets,
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
template <typename InputIteratorT, typename OutputIteratorT>
CUB_RUNTIME_FUNCTION __forceinline__
cudaError_t Dispatch(
    Int2Type<CUB>       dispatch_to,
    int                 timing_timing_iterations,
    size_t              *d_temp_storage_bytes,
    cudaError_t         *d_cdp_error,

    void*               d_temp_storage,
    size_t&             temp_storage_bytes,
    InputIteratorT      d_in,
    OutputIteratorT     d_out,
    int                 num_items,
    int                 max_segments,
    int                 *d_segment_offsets,
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
template <typename InputIteratorT, typename OutputIteratorT>
CUB_RUNTIME_FUNCTION __forceinline__
cudaError_t Dispatch(
    Int2Type<CUB>       dispatch_to,
    int                 timing_timing_iterations,
    size_t              *d_temp_storage_bytes,
    cudaError_t         *d_cdp_error,

    void*               d_temp_storage,
    size_t&             temp_storage_bytes,
    InputIteratorT      d_in,
    OutputIteratorT     d_out,
    int                 num_items,
    int                 max_segments,
    int                 *d_segment_offsets,
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
template <typename InputIteratorT, typename OutputIteratorT>
CUB_RUNTIME_FUNCTION __forceinline__
cudaError_t Dispatch(
    Int2Type<CUB>       dispatch_to,
    int                 timing_timing_iterations,
    size_t              *d_temp_storage_bytes,
    cudaError_t         *d_cdp_error,

    void*               d_temp_storage,
    size_t&             temp_storage_bytes,
    InputIteratorT      d_in,
    OutputIteratorT     d_out,
    int                 num_items,
    int                 max_segments,
    int                 *d_segment_offsets,
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
template <typename InputIteratorT, typename OutputIteratorT>
CUB_RUNTIME_FUNCTION __forceinline__
cudaError_t Dispatch(
    Int2Type<CUB>       dispatch_to,
    int                 timing_timing_iterations,
    size_t              *d_temp_storage_bytes,
    cudaError_t         *d_cdp_error,

    void*               d_temp_storage,
    size_t&             temp_storage_bytes,
    InputIteratorT      d_in,
    OutputIteratorT     d_out,
    int                 num_items,
    int                 max_segments,
    int                 *d_segment_offsets,
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
template <typename InputIteratorT, typename OutputIteratorT, typename ReductionOpT>
cudaError_t Dispatch(
    Int2Type<THRUST>    dispatch_to,
    int                 timing_timing_iterations,
    size_t              *d_temp_storage_bytes,
    cudaError_t         *d_cdp_error,

    void*               d_temp_storage,
    size_t&             temp_storage_bytes,
    InputIteratorT      d_in,
    OutputIteratorT     d_out,
    int                 num_items,
    int                 max_segments,
    int                 *d_segment_offsets,
    ReductionOpT         reduction_op,
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
template <typename InputIteratorT, typename OutputIteratorT>
cudaError_t Dispatch(
    Int2Type<THRUST>    dispatch_to,
    int                 timing_timing_iterations,
    size_t              *d_temp_storage_bytes,
    cudaError_t         *d_cdp_error,

    void*               d_temp_storage,
    size_t&             temp_storage_bytes,
    InputIteratorT      d_in,
    OutputIteratorT     d_out,
    int                 num_items,
    int                 max_segments,
    int                 *d_segment_offsets,
    Sum                 reduction_op,
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
    typename            InputIteratorT,
    typename            OutputIteratorT,
    typename            ReductionOpT>
__global__ void CnpDispatchKernel(
    int                 timing_timing_iterations,
    size_t              *d_temp_storage_bytes,
    cudaError_t         *d_cdp_error,

    void*               d_temp_storage,
    size_t              temp_storage_bytes,
    InputIteratorT      d_in,
    OutputIteratorT     d_out,
    int                 num_items,
    int                 max_segments,
    int                 *d_segment_offsets,
    ReductionOpT         reduction_op,
    bool                debug_synchronous)
{
#ifndef CUB_CDP
    *d_cdp_error = cudaErrorNotSupported;
#else
    *d_cdp_error = Dispatch(Int2Type<CUB>(), timing_timing_iterations, d_temp_storage_bytes, d_cdp_error, d_temp_storage, temp_storage_bytes,
        d_in, d_out, num_items, max_segments, d_segment_offsets, reduction_op, 0, debug_synchronous);
    *d_temp_storage_bytes = temp_storage_bytes;
#endif
}


/**
 * Dispatch to CUB_CDP kernel
 */
template <typename InputIteratorT, typename OutputIteratorT, typename ReductionOpT>
CUB_RUNTIME_FUNCTION __forceinline__
cudaError_t Dispatch(
    Int2Type<CUB_CDP>       dispatch_to,
    int                 timing_timing_iterations,
    size_t              *d_temp_storage_bytes,
    cudaError_t         *d_cdp_error,

    void*               d_temp_storage,
    size_t&             temp_storage_bytes,
    InputIteratorT      d_in,
    OutputIteratorT     d_out,
    int                 num_items,
    int                 max_segments,
    int                 *d_segment_offsets,
    ReductionOpT         reduction_op,
    cudaStream_t        stream,
    bool                debug_synchronous)
{
    // Invoke kernel to invoke device-side dispatch
    CnpDispatchKernel<<<1,1>>>(timing_timing_iterations, d_temp_storage_bytes, d_cdp_error, d_temp_storage, temp_storage_bytes,
        d_in, d_out, num_items, max_segments, d_segment_offsets, reduction_op, debug_synchronous);

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


/// Test DeviceReduce for a given problem input
template <
    typename                BackendT,
    typename                DeviceInputIteratorT,
    typename                HostReferenceIteratorT,
    typename                ReductionOpT>
void Test(
    BackendT                backend,
    DeviceInputIteratorT    d_in,
    int                     num_items,
    int                     num_segments,
    int                     *d_segment_offsets,
    ReductionOpT            reduction_op,
    HostReferenceIteratorT  h_reference)
{
    // Input data type
    typedef typename std::iterator_traits<DeviceInputIteratorT>::value_type InputT;

    // Output data type
    typedef typename std::iterator_traits<HostReferenceIteratorT>::value_type OutputT;

    // Allocate CUB_CDP device arrays for temp storage size and error
    OutputT         *d_out = NULL;
    size_t          *d_temp_storage_bytes = NULL;
    cudaError_t     *d_cdp_error = NULL;
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_out,                 sizeof(OutputT) * num_segments));
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_temp_storage_bytes,  sizeof(size_t) * 1));
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_cdp_error,           sizeof(cudaError_t) * 1));

    // Inquire temp device storage
    void            *d_temp_storage = NULL;
    size_t          temp_storage_bytes = 0;
    CubDebugExit(Dispatch(backend, 1,
        d_temp_storage_bytes, d_cdp_error, d_temp_storage, temp_storage_bytes,
        d_in, d_out, num_items, reduction_op, 0, true));

    // Allocate temp device storage
    CubDebugExit(g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes));

    // Clear device output
    CubDebugExit(cudaMemset(d_out, 0, sizeof(OutputT) * num_segments));

    // Run warmup/correctness iteration
    CubDebugExit(Dispatch(backend, 1,
        d_temp_storage_bytes, d_cdp_error, d_temp_storage, temp_storage_bytes,
        d_in, d_out, num_items, reduction_op, 0, true));

    // Check for correctness (and display results, if specified)
    int compare = CompareDeviceResults(h_reference, d_out, num_segments, g_verbose, g_verbose);
    printf("\t%s", compare ? "FAIL" : "PASS");

    // Flush any stdout/stderr
    fflush(stdout);
    fflush(stderr);

    // Performance
    if (g_timing_iterations > 0)
    {
        GpuTimer gpu_timer;
        gpu_timer.Start();

        CubDebugExit(Dispatch(backend, g_timing_iterations,
            d_temp_storage_bytes, d_cdp_error, d_temp_storage, temp_storage_bytes,
            d_in, d_out, num_items, reduction_op, 0, false));

        gpu_timer.Stop();
        float elapsed_millis = gpu_timer.ElapsedMillis();

        // Display performance
        float avg_millis = elapsed_millis / g_timing_iterations;
        float giga_rate = float(num_items) / avg_millis / 1000.0 / 1000.0;
        float giga_bandwidth = giga_rate * sizeof(InputT);
        printf(", %.3f avg ms, %.3f billion items/s, %.3f logical GB/s", avg_millis, giga_rate, giga_bandwidth);
    }

    if (d_out) CubDebugExit(g_allocator.DeviceFree(d_out));
    if (d_temp_storage_bytes) CubDebugExit(g_allocator.DeviceFree(d_temp_storage_bytes));
    if (d_cdp_error) CubDebugExit(g_allocator.DeviceFree(d_cdp_error));
    if (d_temp_storage) CubDebugExit(g_allocator.DeviceFree(d_temp_storage));

    // Correctness asserts
    AssertEquals(0, compare);
}


/// Test DeviceReduce
template <
    Backend                 BACKEND,
    typename                HostInputIteratorT,
    typename                DeviceInputIteratorT,
    typename                ReductionOpT>
void SolveAndTest(
    HostInputIteratorT      h_in,
    DeviceInputIteratorT    d_in,
    int                     num_items,
    int                     num_segments,
    int                     *h_segment_offsets,
    int                     *d_segment_offsets,
    ReductionOpT            reduction_op)
{
    // Data type of input iterator
    typedef typename std::iterator_traits<HostInputIteratorT>::value_type T;

    printf("\n\n%s cub::DeviceReduce::%s %d items, %d segments\n",
        (BACKEND == CUB_CDP) ? "CUB_CDP" : (BACKEND == THRUST) ? "Thrust" : (BACKEND == CUB_SEGMENTED) ? "CUB_SEGMENTED" : "CUB",
        (Equals<ReductionOpT, Sum>::VALUE) ? "Sum" : (Equals<ReductionOpT, Min>::VALUE) ? "Min" : "Max",
        num_items, num_segments);
    fflush(stdout);

    // Allocate host arrays
    T* h_reference = new T[num_segments];

    // Solve solution
    for (int i = 0; i < num_segments; ++i)
    {
        for (int j = h_segment_offsets[i]; j < h_segment_offsets[i + 1]; ++j)
        {
            if (j == 0)
                h_reference[i] = h_in[j];
            else
                h_reference[i] = reduction_op(h_reference[i], h_in[j]);
        }
    }
    printf("Reference done\n"); fflush(stdout);

    // Run test
    Test(Int2Type<BACKEND>(), d_in, num_items, num_segments, d_segment_offsets, reduction_op, h_reference);

    // Cleanup
    if (h_reference) delete[] h_reference;
}


/// Test DeviceReduce (argmin specialization)
/*
template <
    Backend             BACKEND,
    typename            HostInputIteratorT,
    typename            DeviceInputIteratorT,
    typename            ReductionOpT>
void SolveAndTest(
    HostInputIteratorT      h_in,
    DeviceInputIteratorT    d_in,
    int                     num_items,
    int                     num_segments,
    int                     *h_segment_offsets,
    int                     *d_segment_offsets,
    ReductionOpT            reduction_op,
    const char              *type_string)
{
    // Data type of input iterator
    typedef typename std::iterator_traits<HostInputIteratorT>::value_type T;

    printf("\n\n%s cub::DeviceReduce::%s %d items, %d segments, %s %d-byte elements\n",
        (BACKEND == CUB_CDP) ? "CUB_CDP" : (BACKEND == THRUST) ? "Thrust" : (BACKEND == CUB_SEGMENTED) ? "CUB_SEGMENTED" : "CUB",
        (Equals<ReductionOpT, Sum>::VALUE) ? "Sum" : (Equals<ReductionOpT, Min>::VALUE) ? "Min" : "Max",
        num_items, num_segments, type_string, (int) sizeof(T));
    fflush(stdout);

    // Allocate host arrays
    typedef KeyValuePair<int, T> KeyValuePairT;
    KeyValuePairT *h_reference = new KeyValuePairT[num_segments];

    // Solve solution
    for (int i = 0; i < num_segments; ++i)
    {
        for (int j = h_segment_offsets[i]; j < h_segment_offsets[i + 1]; ++j)
        {
            if (j == 0)
                h_reference[i] = h_in[j];
            else
                h_reference[i] = reduction_op(h_reference[i], h_in[j]);
        }
    }
    printf("Reference done\n"); fflush(stdout);

    // Run test
    Test<BACKEND>(d_in, h_reference, num_items, num_segments, d_segment_offsets, reduction_op);

    // Cleanup
    if (h_reference) delete[] h_reference;
}
*/


/*
/// Test DeviceReduce on pointer type (argmax specialization)
template <
    Backend     BACKEND,
    typename    T>
void SolveAndTest(
    int         num_items,
    GenMode     gen_mode,
    ArgMax      reduction_op,
    const char* type_string)
{
    printf("\n\n%s cub::DeviceReduce::%s %d items, %s %d-byte elements, gen-mode %s\n",
        (BACKEND == CUB_CDP) ? "CUB_CDP" : (BACKEND == THRUST) ? "Thrust" : "CUB",
        "ArgMax", num_items, type_string, (int) sizeof(T),
        (gen_mode == RANDOM) ? "RANDOM" : (gen_mode == INTEGER_SEED) ? "SEQUENTIAL" : "HOMOGENOUS");
    fflush(stdout);

    // Allocate host arrays
    T* h_in = new T[num_items];
    KeyValuePair<int, T> h_reference;

    // Initialize problem and solution
    printf("Initializing... "); fflush(stdout);
    Initialize(gen_mode, h_in, num_items);
    printf("Solving... "); fflush(stdout);
    Solve(h_in, h_reference.value, Max(), num_items);
    for (int i = 0; i < num_items; ++i)
    {
        if (!(h_reference.value != h_in[i]))
        {
            h_reference.key = i;
            break;
        }
    }
    printf("Reference done\n"); fflush(stdout);

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
*/



/// Test specific problem type
template <
    Backend         BACKEND,
    typename        T,
    typename        ReductionOpT>
void TestProblem(
    int             num_items,
    int             num_segments,
    GenMode         gen_mode,
    ReductionOpT    reduction_op,
    const char*     type_string)
{
    // Initialize host data
    printf("Initializing %s (gen mode %d)... ", type_string, gen_mode); fflush(stdout);
    T* h_in = new T[num_items];
    Initialize(gen_mode, h_in, num_items);

    // Initialize segment data
    int *h_segment_offsets = new int[num_segments + 1];
    InitializeSegments(num_items, num_segments, h_segment_offsets, g_verbose);

    // Initialize device data
    int *d_segment_offsets      = NULL;
    T   *d_in                   = NULL;
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_in,              sizeof(T) * num_items));
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_segment_offsets, sizeof(int) * (num_segments + 1)));
    CubDebugExit(cudaMemcpy(d_in,               h_in,                   sizeof(T) * num_items, cudaMemcpyHostToDevice));
    CubDebugExit(cudaMemcpy(d_segment_offsets,  h_segment_offsets,      sizeof(int) * (num_segments + 1), cudaMemcpyHostToDevice));

    SolveAndTest<BACKEND>(h_in, d_in, num_items, num_segments, h_segment_offsets, d_segment_offsets, reduction_op);

    if (h_segment_offsets)  delete[] h_segment_offsets;
    if (d_segment_offsets)  CubDebugExit(g_allocator.DeviceFree(d_segment_offsets));
    if (h_in)               delete[] h_in;
    if (d_in)               CubDebugExit(g_allocator.DeviceFree(d_in));
}


/// Test different operators
template <
    Backend             BACKEND,
    typename            HostInputIteratorT,
    typename            DeviceInputIteratorT>
void TestByOp(
    HostInputIteratorT      h_in,
    DeviceInputIteratorT    d_in,
    int                     num_items,
    int                     num_segments,
    int                     *h_segment_offsets,
    int                     *d_segment_offsets)
{
    SolveAndTest<BACKEND>(h_in, d_in, num_items, num_segments, h_segment_offsets, d_segment_offsets, CustomMax());
    SolveAndTest<BACKEND>(h_in, d_in, num_items, num_segments, h_segment_offsets, d_segment_offsets, Sum());
    SolveAndTest<BACKEND>(h_in, d_in, num_items, num_segments, h_segment_offsets, d_segment_offsets, Min());
    SolveAndTest<BACKEND>(h_in, d_in, num_items, num_segments, h_segment_offsets, d_segment_offsets, ArgMin());
    SolveAndTest<BACKEND>(h_in, d_in, num_items, num_segments, h_segment_offsets, d_segment_offsets, Max());
    SolveAndTest<BACKEND>(h_in, d_in, num_items, num_segments, h_segment_offsets, d_segment_offsets, ArgMax());
}


/// Test different backends
template <typename T>
void TestByBackend(
    int             num_items,
    int             max_segments,
    GenMode         gen_mode,
    const char*     type_string)
{
    // Initialize host data
    printf("Initializing %s (gen mode %d)... ", type_string, gen_mode); fflush(stdout);
    T* h_in = new T[num_items];
    Initialize(gen_mode, h_in, num_items);

    // Initialize device data
    T *d_in = NULL;
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_in, sizeof(T) * num_items));
    CubDebugExit(cudaMemcpy(d_in, h_in, sizeof(T) * num_items, cudaMemcpyHostToDevice));

    // Test single-segment implementations

    TestByOp<CUB, T>(h_in, d_in, num_items, 1, NULL);
#ifdef CUB_CDP
    TestByOp<CUB_CDP, T>(h_in, d_in, num_items, 1, NULL);
#endif

    // Test segmented implementation
    if (max_segments > 0)
    {

        // Right now we assign a single thread block to each segment, so lets keep it to under 128K items per segment
        int max_items_per_segment   = 128000;
        int *h_segment_offsets      = new int[max_segments + 1];
        int *d_segment_offsets      = NULL;
        CubDebugExit(g_allocator.DeviceAllocate((void**)&d_segment_offsets, sizeof(int) * (max_segments + 1)));

        for (int num_segments = (num_items + max_items_per_segment - 1) / max_items_per_segment;
            num_segments < max_segments;
            num_segments *= 32)
        {
            InitializeSegments(num_items, num_segments, h_segment_offsets);
            CubDebugExit(cudaMemcpy(d_segment_offsets, h_segment_offsets, sizeof(int) * (num_segments + 1), cudaMemcpyHostToDevice));
            TestByOp<CUB_SEGMENTED, T>(h_in, d_in, num_items, num_segments, h_segment_offsets, d_segment_offsets);
        }

        if (h_segment_offsets)  delete[] h_segment_offsets;
        if (d_segment_offsets)  CubDebugExit(g_allocator.DeviceFree(d_segment_offsets));
    }

    if (h_in)               delete[] h_in;
    if (d_in)               CubDebugExit(g_allocator.DeviceFree(d_in));
}


/// Test different input-generation modes
template <typename T>
void TestByGenMode(
    int             num_items,
    int             max_segments,
    const char*     type_string)
{
    TestByBackend<T>(num_items, max_segments, UNIFORM,         type_string);
    TestByBackend<T>(num_items, max_segments, INTEGER_SEED,    type_string);
    TestByBackend<T>(num_items, max_segments, RANDOM,          type_string);

    // Test iterator support using a constant-iterator and SUM
    T val;
    InitValue(UNIFORM, val, 0);
    ConstantInputIterator<T, int> h_in(val);
    Test<CUB>(      h_in, h_in, num_items, 1, NULL, NULL, Sum(), UNIFORM, type_string);
#ifdef CUB_CDP
    Test<CUB_CDP>(  h_in, h_in, num_items, 1, NULL, NULL, Sum(), UNIFORM, type_string);
#endif
}


/// Test differnet problem sizes
template <typename T>
void TestBySize(
    int             num_items,
    int             max_segments,
    const char*     type_string)
{
    // Policy dispatch functor
    struct DispatchPolicy
    {
        int             max_items;
        int             max_segments;
        const char*     type_string;

        DispatchPolicy(int max_items, int max_segments, const char* type_string) :
            max_items(max_items), max_segments(max_segments), type_string(type_string) {}

        template <typename ActivePolicyT>
        CUB_RUNTIME_FUNCTION __forceinline__
        cudaError_t Invoke()
        {
            // Black-box testing on all backends

            // Test 0, 1, many
            TestByGenMode<T>(0,           max_segments, type_string);
            TestByGenMode<T>(1,           max_segments, type_string);
            TestByGenMode<T>(max_items,   max_segments, type_string);

            // Test random sizes
            int num_iterations = 8;
            for (int i = 0; i < num_iterations; ++i)
            {
                unsigned int num_items;
                RandomValue((unsigned int&) num_items, max_items);
                TestByGenMode<T>(num_items, max_segments, type_string);
            }

            // White-box testing of single-segment problems around specific sizes

            // multiple blocks, one tile per block
            int tile_size = ActivePolicyT::ReducePolicy::BLOCK_THREADS * ActivePolicyT::ReducePolicy::ITEMS_PER_THREAD;
            TestByGenMode<T>(tile_size * 4,     -1, type_string);
            TestByGenMode<T>(tile_size * 4 + 1, -1, type_string);
            TestByGenMode<T>(tile_size * 4 - 1, -1, type_string);

            // multiple blocks, multiple tiles per block
            int sm_occupancy = 32;
            int occupancy = tile_size * sm_occupancy * g_sm_count;
            TestByGenMode<T>(occupancy,         -1, type_string);
            TestByGenMode<T>(occupancy + 1,     -1, type_string);
            TestByGenMode<T>(occupancy - 1,     -1, type_string);

            return cudaSuccess;
        }
    };

    DeviceReducePolicy::MaxPolicy::Invoke(
        g_ptx_version,
        DispatchPolicy<T>(num_items, max_segments, type_string));
}


//---------------------------------------------------------------------
// Main
//---------------------------------------------------------------------


/**
 * Main
 */
int main(int argc, char** argv)
{
    unsigned int max_items       = 48000000;
    unsigned int max_segments    = 5000;

    // Initialize command line
    CommandLineArgs args(argc, argv);
    g_verbose = args.CheckCmdLineFlag("v");
    args.GetCmdLineArgument("n", max_items);
    args.GetCmdLineArgument("s", max_segments);
    args.GetCmdLineArgument("i", g_timing_iterations);
    args.GetCmdLineArgument("repeat", g_repeat);

    // Print usage
    if (args.CheckCmdLineFlag("help"))
    {
        printf("%s "
            "[--n=<input items> "
            "[--s=<num segments> "
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

    // Get ptx version
    CubDebugExit(PtxVersion(g_ptx_version));

    // Get SM count
    g_sm_count = args.deviceProp.multiProcessorCount;

    printf("\n");

#ifdef QUICKER_TEST

    // Compile/run basic test
    TestProblem<CUB, int>(max_items, 1, RANDOM, Sum(), CUB_TYPE_STRING(int));

/*
    TestPointer<CUB, int>(          max_items, 1,            RANDOM, Sum(), CUB_TYPE_STRING(int));
    TestPointer<CUB, int>(          max_items, 1,            RANDOM, ArgMax(), CUB_TYPE_STRING(int));

    TestPointer<CUB, float>(        max_items, 1,            RANDOM, Sum(), CUB_TYPE_STRING(float));
    TestPointer<CUB, float>(        max_items, 1,            RANDOM, ArgMax(), CUB_TYPE_STRING(float));
*/

#elif defined(QUICK_TEST)

    // Compile/run quick comparison tests

    TestPointer<CUB, char>(         max_items * 4, 1, UNIFORM, Sum(), CUB_TYPE_STRING(char));
    TestPointer<THRUST, char>(      max_items * 4, 1, UNIFORM, Sum(), CUB_TYPE_STRING(char));

    printf("\n----------------------------\n");
    TestPointer<CUB, short>(        max_items * 2, 1, UNIFORM, Sum(), CUB_TYPE_STRING(short));
    TestPointer<THRUST, short>(     max_items * 2, 1, UNIFORM, Sum(), CUB_TYPE_STRING(short));

    printf("\n----------------------------\n");
    TestPointer<CUB, int>(          max_items,     1, UNIFORM, Sum(), CUB_TYPE_STRING(int));
    TestPointer<THRUST, int>(       max_items,     1, UNIFORM, Sum(), CUB_TYPE_STRING(int));

    printf("\n----------------------------\n");
    TestPointer<CUB, long long>(    max_items / 2, 1, UNIFORM, Sum(), CUB_TYPE_STRING(long long));
    TestPointer<THRUST, long long>( max_items / 2, 1, UNIFORM, Sum(), CUB_TYPE_STRING(long long));

    printf("\n----------------------------\n");
    TestPointer<CUB, TestFoo>(      max_items / 4, 1, UNIFORM, Max(), CUB_TYPE_STRING(TestFoo));
    TestPointer<THRUST, TestFoo>(   max_items / 4, 1, UNIFORM, Max(), CUB_TYPE_STRING(TestFoo));

#else

    // Compile/run thorough tests
    for (int i = 0; i <= g_repeat; ++i)
    {
        // Test different input types
        Test<unsigned char>(        max_items, max_segments, CUB_TYPE_STRING(unsigned char));
        Test<unsigned short>(       max_items, max_segments, CUB_TYPE_STRING(unsigned short));
        Test<unsigned int>(         max_items, max_segments, CUB_TYPE_STRING(unsigned int));
        Test<unsigned long long>(   max_items, max_segments, CUB_TYPE_STRING(unsigned long long));

        Test<uchar2>(               max_items, max_segments, CUB_TYPE_STRING(uchar2));
        Test<uint2>(                max_items, max_segments, CUB_TYPE_STRING(uint2));
        Test<ulonglong2>(           max_items, max_segments, CUB_TYPE_STRING(ulonglong2));
        Test<ulonglong4>(           max_items, max_segments, CUB_TYPE_STRING(ulonglong4));

        Test<TestFoo>(              max_items, max_segments, CUB_TYPE_STRING(TestFoo));
        Test<TestBar>(              max_items, max_segments, CUB_TYPE_STRING(TestBar));
    }

#endif

    return 0;
}



