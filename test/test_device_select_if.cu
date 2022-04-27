/******************************************************************************
 * Copyright (c) 2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2018, NVIDIA CORPORATION.  All rights reserved.
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
 * Test of DeviceSelect::If and DevicePartition::If utilities
 ******************************************************************************/

// Ensure printing of CUDA runtime errors to console
#define CUB_STDERR

#include <stdio.h>
#include <typeinfo>

#include <cub/util_allocator.cuh>
#include <cub/device/device_select.cuh>
#include <cub/device/device_partition.cuh>
#include <cub/iterator/counting_input_iterator.cuh>

#include <thrust/count.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/transform_output_iterator.h>
#include <thrust/iterator/zip_iterator.h>

#include "test_util.h"

using namespace cub;


//---------------------------------------------------------------------
// Globals, constants and typedefs
//---------------------------------------------------------------------

bool                    g_verbose               = false;
int                     g_timing_iterations     = 0;
float                   g_device_giga_bandwidth;
CachingDeviceAllocator  g_allocator(true);

// Dispatch types
enum Backend
{
    CUB,        // CUB method
    CDP,        // GPU-based (dynamic parallelism) dispatch to CUB method
};


// Selection functor type
template <typename T>
struct LessThan
{
    T compare;

    __host__ __device__ __forceinline__
    LessThan(T compare) : compare(compare) {}

    __host__ __device__ __forceinline__
    bool operator()(const T &a) const {
        return (a < compare);
    }
};

//---------------------------------------------------------------------
// Dispatch to different CUB DeviceSelect entrypoints
//---------------------------------------------------------------------


/**
 * Dispatch to select if entrypoint
 */
template <typename InputIteratorT, typename FlagIteratorT, typename SelectOpT, typename OutputIteratorT, typename NumSelectedIteratorT, typename OffsetT>
CUB_RUNTIME_FUNCTION __forceinline__
cudaError_t Dispatch(
    Int2Type<CUB>               /*dispatch_to*/,
    Int2Type<false>             /*is_flagged*/,
    Int2Type<false>             /*is_partition*/,
    int                         timing_timing_iterations,
    size_t*                     /*d_temp_storage_bytes*/,
    cudaError_t*                /*d_cdp_error*/,

    void*                       d_temp_storage,
    size_t&                     temp_storage_bytes,
    InputIteratorT              d_in,
    FlagIteratorT               /*d_flags*/,
    OutputIteratorT             d_out,
    NumSelectedIteratorT        d_num_selected_out,
    OffsetT                     num_items,
    SelectOpT                   select_op,
    cudaStream_t                stream,
    bool                        debug_synchronous)
{
    cudaError_t error = cudaSuccess;
    for (int i = 0; i < timing_timing_iterations; ++i)
    {
        error = DeviceSelect::If(d_temp_storage, temp_storage_bytes, d_in, d_out, d_num_selected_out, num_items, select_op, stream, debug_synchronous);
    }
    return error;
}


/**
 * Dispatch to partition if entrypoint
 */
template <typename InputIteratorT, typename FlagIteratorT, typename SelectOpT, typename OutputIteratorT, typename NumSelectedIteratorT, typename OffsetT>
CUB_RUNTIME_FUNCTION __forceinline__
cudaError_t Dispatch(
    Int2Type<CUB>               /*dispatch_to*/,
    Int2Type<false>             /*is_flagged*/,
    Int2Type<true>              /*is_partition*/,
    int                         timing_timing_iterations,
    size_t*                     /*d_temp_storage_bytes*/,
    cudaError_t*                /*d_cdp_error*/,

    void*                       d_temp_storage,
    size_t&                     temp_storage_bytes,
    InputIteratorT              d_in,
    FlagIteratorT               /*d_flags*/,
    OutputIteratorT             d_out,
    NumSelectedIteratorT        d_num_selected_out,
    OffsetT                     num_items,
    SelectOpT                   select_op,
    cudaStream_t                stream,
    bool                        debug_synchronous)
{
    cudaError_t error = cudaSuccess;
    for (int i = 0; i < timing_timing_iterations; ++i)
    {
        error = DevicePartition::If(d_temp_storage, temp_storage_bytes, d_in, d_out, d_num_selected_out, num_items, select_op, stream, debug_synchronous);
    }
    return error;
}


/**
 * Dispatch to select flagged entrypoint
 */
template <typename InputIteratorT, typename FlagIteratorT, typename SelectOpT, typename OutputIteratorT, typename NumSelectedIteratorT, typename OffsetT>
CUB_RUNTIME_FUNCTION __forceinline__
cudaError_t Dispatch(
    Int2Type<CUB>               /*dispatch_to*/,
    Int2Type<true>              /*is_flagged*/,
    Int2Type<false>             /*partition*/,
    int                         timing_timing_iterations,
    size_t*                     /*d_temp_storage_bytes*/,
    cudaError_t*                /*d_cdp_error*/,

    void*                       d_temp_storage,
    size_t&                     temp_storage_bytes,
    InputIteratorT              d_in,
    FlagIteratorT               d_flags,
    OutputIteratorT             d_out,
    NumSelectedIteratorT        d_num_selected_out,
    OffsetT                     num_items,
    SelectOpT                   /*select_op*/,
    cudaStream_t                stream,
    bool                        debug_synchronous)
{
    cudaError_t error = cudaSuccess;
    for (int i = 0; i < timing_timing_iterations; ++i)
    {
        error = DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, d_in, d_flags, d_out, d_num_selected_out, num_items, stream, debug_synchronous);
    }
    return error;
}


/**
 * Dispatch to partition flagged entrypoint
 */
template <typename InputIteratorT, typename FlagIteratorT, typename SelectOpT, typename OutputIteratorT, typename NumSelectedIteratorT, typename OffsetT>
CUB_RUNTIME_FUNCTION __forceinline__
cudaError_t Dispatch(
    Int2Type<CUB>               /*dispatch_to*/,
    Int2Type<true>              /*is_flagged*/,
    Int2Type<true>              /*partition*/,
    int                         timing_timing_iterations,
    size_t*                     /*d_temp_storage_bytes*/,
    cudaError_t*                /*d_cdp_error*/,

    void*                       d_temp_storage,
    size_t&                     temp_storage_bytes,
    InputIteratorT              d_in,
    FlagIteratorT               d_flags,
    OutputIteratorT             d_out,
    NumSelectedIteratorT        d_num_selected_out,
    OffsetT                     num_items,
    SelectOpT                   /*select_op*/,
    cudaStream_t                stream,
    bool                        debug_synchronous)
{
    cudaError_t error = cudaSuccess;
    for (int i = 0; i < timing_timing_iterations; ++i)
    {
        error = DevicePartition::Flagged(d_temp_storage, temp_storage_bytes, d_in, d_flags, d_out, d_num_selected_out, num_items, stream, debug_synchronous);
    }
    return error;
}

//---------------------------------------------------------------------
// CUDA Nested Parallelism Test Kernel
//---------------------------------------------------------------------

/**
 * Simple wrapper kernel to invoke DeviceSelect
 */
template <typename InputIteratorT, typename FlagIteratorT, typename SelectOpT, typename OutputIteratorT, typename NumSelectedIteratorT, typename OffsetT, typename IsFlaggedTag, typename IsPartitionTag>
__global__ void CnpDispatchKernel(
    IsFlaggedTag                is_flagged,
    IsPartitionTag              is_partition,
    int                         timing_timing_iterations,
    size_t*                     d_temp_storage_bytes,
    cudaError_t*                d_cdp_error,

    void*                       d_temp_storage,
    size_t                      temp_storage_bytes,
    InputIteratorT              d_in,
    FlagIteratorT               d_flags,
    OutputIteratorT             d_out,
    NumSelectedIteratorT        d_num_selected_out,
    OffsetT                     num_items,
    SelectOpT                   select_op,
    bool                        debug_synchronous)
{

#ifndef CUB_CDP
    (void)is_flagged;
    (void)is_partition;
    (void)timing_timing_iterations;
    (void)d_temp_storage_bytes;
    (void)d_temp_storage;
    (void)temp_storage_bytes;
    (void)d_in;
    (void)d_flags;
    (void)d_out;
    (void)d_num_selected_out;
    (void)num_items;
    (void)select_op;
    (void)debug_synchronous;
    *d_cdp_error = cudaErrorNotSupported;
#else
    *d_cdp_error = Dispatch(Int2Type<CUB>(), is_flagged, is_partition, timing_timing_iterations, d_temp_storage_bytes, d_cdp_error,
        d_temp_storage, temp_storage_bytes, d_in, d_flags, d_out, d_num_selected_out, num_items, select_op, 0, debug_synchronous);
    *d_temp_storage_bytes = temp_storage_bytes;
#endif
}


/**
 * Dispatch to CDP kernel
 */
template <typename InputIteratorT, typename FlagIteratorT, typename SelectOpT, typename OutputIteratorT, typename NumSelectedIteratorT, typename OffsetT, typename IsFlaggedTag, typename IsPartitionTag>
cudaError_t Dispatch(
    Int2Type<CDP>               dispatch_to,
    IsFlaggedTag                is_flagged,
    IsPartitionTag              is_partition,
    int                         timing_timing_iterations,
    size_t*                     d_temp_storage_bytes,
    cudaError_t*                d_cdp_error,

    void*                       d_temp_storage,
    size_t&                     temp_storage_bytes,
    InputIteratorT              d_in,
    FlagIteratorT               d_flags,
    OutputIteratorT             d_out,
    NumSelectedIteratorT        d_num_selected_out,
    OffsetT                     num_items,
    SelectOpT                   select_op,
    cudaStream_t                stream,
    bool                        debug_synchronous)
{
    // Invoke kernel to invoke device-side dispatch
    CnpDispatchKernel<<<1,1>>>(is_flagged, is_partition, timing_timing_iterations, d_temp_storage_bytes, d_cdp_error,
        d_temp_storage, temp_storage_bytes, d_in, d_flags, d_out, d_num_selected_out, num_items, select_op, debug_synchronous);

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
    T*  h_in,
    int num_items)
{
    for (int i = 0; i < num_items; ++i)
    {
        // Initialize each item to a randomly selected value from [0..126]
        unsigned int value;
        RandomBits(value, 0, 0, 7);
        if (value == 127)
            value = 126;
        InitValue(INTEGER_SEED, h_in[i], value);
    }

    if (g_verbose)
    {
        printf("Input:\n");
        DisplayResults(h_in, num_items);
        printf("\n\n");
    }
}


/**
 * Solve selection problem (and set corresponding flags)
 */
template <
    typename        InputIteratorT,
    typename        FlagIteratorT,
    typename        SelectOpT,
    typename        T>
int Solve(
    InputIteratorT  h_in,
    SelectOpT       select_op,
    T*              h_reference,
    FlagIteratorT   h_flags,
    int             num_items)
{
    int num_selected = 0;
    for (int i = 0; i < num_items; ++i)
    {
        if ((h_flags[i] = select_op(h_in[i])))
        {
            h_reference[num_selected] = h_in[i];
            num_selected++;
        }
        else
        {
            h_reference[num_items - (i - num_selected) - 1] = h_in[i];
        }
    }

    return num_selected;
}



/**
 * Test DeviceSelect for a given problem input
 */
template <
    Backend             BACKEND,
    bool                IS_FLAGGED,
    bool                IS_PARTITION,
    typename            DeviceInputIteratorT,
    typename            FlagT,
    typename            SelectOpT,
    typename            T>
void Test(
    DeviceInputIteratorT    d_in,
    FlagT*                  h_flags,
    SelectOpT               select_op,
    T*                      h_reference,
    int                     num_selected,
    int                     num_items)
{
    // Allocate device flags, output, and num-selected
    FlagT*      d_flags = NULL;
    T*          d_out = NULL;
    int*        d_num_selected_out = NULL;
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_flags, sizeof(FlagT) * num_items));
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_out, sizeof(T) * num_items));
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_num_selected_out, sizeof(int)));

    // Allocate CDP device arrays
    size_t*         d_temp_storage_bytes = NULL;
    cudaError_t*    d_cdp_error = NULL;
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_temp_storage_bytes,  sizeof(size_t) * 1));
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_cdp_error,           sizeof(cudaError_t) * 1));

    // Allocate temporary storage
    void            *d_temp_storage = NULL;
    size_t          temp_storage_bytes = 0;
    CubDebugExit(Dispatch(Int2Type<BACKEND>(), Int2Type<IS_FLAGGED>(), Int2Type<IS_PARTITION>(), 1, d_temp_storage_bytes, d_cdp_error,
    d_temp_storage, temp_storage_bytes, d_in, d_flags, d_out, d_num_selected_out, num_items, select_op, 0, true));
    CubDebugExit(g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes));

    // Copy flags and clear device output array
    CubDebugExit(cudaMemcpy(d_flags, h_flags, sizeof(FlagT) * num_items, cudaMemcpyHostToDevice));
    CubDebugExit(cudaMemset(d_out, 0, sizeof(T) * num_items));
    CubDebugExit(cudaMemset(d_num_selected_out, 0, sizeof(int)));

    // Run warmup/correctness iteration
    CubDebugExit(Dispatch(Int2Type<BACKEND>(), Int2Type<IS_FLAGGED>(), Int2Type<IS_PARTITION>(), 1, d_temp_storage_bytes, d_cdp_error,
        d_temp_storage, temp_storage_bytes, d_in, d_flags, d_out, d_num_selected_out, num_items, select_op, 0, true));

    // Check for correctness (and display results, if specified)
    int compare1 = (IS_PARTITION) ?
        CompareDeviceResults(h_reference, d_out, num_items, true, g_verbose) :
        CompareDeviceResults(h_reference, d_out, num_selected, true, g_verbose);
    printf("\t Data %s\n", compare1 ? "FAIL" : "PASS");

    int compare2 = CompareDeviceResults(&num_selected, d_num_selected_out, 1, true, g_verbose);
    printf("\t Count %s\n", compare2 ? "FAIL" : "PASS");

    // Flush any stdout/stderr
    fflush(stdout);
    fflush(stderr);

    // Performance
    GpuTimer gpu_timer;
    gpu_timer.Start();
    CubDebugExit(Dispatch(Int2Type<BACKEND>(), Int2Type<IS_FLAGGED>(), Int2Type<IS_PARTITION>(), g_timing_iterations, d_temp_storage_bytes, d_cdp_error,
        d_temp_storage, temp_storage_bytes, d_in, d_flags, d_out, d_num_selected_out, num_items, select_op, 0, false));
    gpu_timer.Stop();
    float elapsed_millis = gpu_timer.ElapsedMillis();

    // Display performance
    if (g_timing_iterations > 0)
    {
        float   avg_millis          = elapsed_millis / g_timing_iterations;
        float   giga_rate           = float(num_items) / avg_millis / 1000.0f / 1000.0f;
        int     num_output_items    = (IS_PARTITION) ? num_items : num_selected;
        int     num_flag_items      = (IS_FLAGGED) ? num_items : 0;
        size_t  num_bytes           = sizeof(T) * (num_items + num_output_items) + sizeof(FlagT) * num_flag_items;
        float   giga_bandwidth      = float(num_bytes) / avg_millis / 1000.0f / 1000.0f;

        printf(", %.3f avg ms, %.3f billion items/s, %.3f logical GB/s, %.1f%% peak", avg_millis, giga_rate, giga_bandwidth, giga_bandwidth / g_device_giga_bandwidth * 100.0);
    }
    printf("\n\n");

    // Flush any stdout/stderr
    fflush(stdout);
    fflush(stderr);

    // Cleanup
    if (d_flags) CubDebugExit(g_allocator.DeviceFree(d_flags));
    if (d_out) CubDebugExit(g_allocator.DeviceFree(d_out));
    if (d_num_selected_out) CubDebugExit(g_allocator.DeviceFree(d_num_selected_out));
    if (d_temp_storage_bytes) CubDebugExit(g_allocator.DeviceFree(d_temp_storage_bytes));
    if (d_cdp_error) CubDebugExit(g_allocator.DeviceFree(d_cdp_error));
    if (d_temp_storage) CubDebugExit(g_allocator.DeviceFree(d_temp_storage));

    // Correctness asserts
    AssertEquals(0, compare1 | compare2);
}


/**
 * Test on pointer type
 */
template <
    Backend         BACKEND,
    bool            IS_FLAGGED,
    bool            IS_PARTITION,
    typename        T>
void TestPointer(
    int             num_items,
    float           select_ratio)
{
    typedef char FlagT;

    // Allocate host arrays
    T*      h_in        = new T[num_items];
    FlagT*  h_flags     = new FlagT[num_items];
    T*      h_reference = new T[num_items];

    // Initialize input
    Initialize(h_in, num_items);

    // Select a comparison value that is select_ratio through the space of [0,127]
    T compare;
    if (select_ratio <= 0.0)
        InitValue(INTEGER_SEED, compare, 0);        // select none
    else if (select_ratio >= 1.0)
        InitValue(INTEGER_SEED, compare, 127);      // select all
    else
        InitValue(INTEGER_SEED, compare, int(double(double(127) * select_ratio)));

    LessThan<T> select_op(compare);
    int num_selected = Solve(h_in, select_op, h_reference, h_flags, num_items);

    if (g_verbose) std::cout << "\nComparison item: " << compare << "\n";
    printf("\nPointer %s cub::%s::%s %d items, %d selected (select ratio %.3f), %s %d-byte elements\n",
        (IS_PARTITION) ? "DevicePartition" : "DeviceSelect",
        (IS_FLAGGED) ? "Flagged" : "If",
        (BACKEND == CDP) ? "CDP CUB" : "CUB",
        num_items, num_selected, float(num_selected) / num_items, typeid(T).name(), (int) sizeof(T));
    fflush(stdout);

    // Allocate problem device arrays
    T *d_in = NULL;

    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_in, sizeof(T) * num_items));

    // Initialize device input
    CubDebugExit(cudaMemcpy(d_in, h_in, sizeof(T) * num_items, cudaMemcpyHostToDevice));

    // Run Test
    Test<BACKEND, IS_FLAGGED, IS_PARTITION>(d_in, h_flags, select_op, h_reference, num_selected, num_items);

    // Cleanup
    if (h_in) delete[] h_in;
    if (h_reference) delete[] h_reference;
    if (h_flags) delete[] h_flags;
    if (d_in) CubDebugExit(g_allocator.DeviceFree(d_in));
}


/**
 * Test on iterator type
 */
template <
    Backend         BACKEND,
    bool            IS_FLAGGED,
    bool            IS_PARTITION,
    typename        T>
void TestIterator(
    int             num_items,
    float           select_ratio)
{
    typedef char FlagT;

    // Allocate host arrays
    T*      h_reference = new T[num_items];
    FlagT*  h_flags = new FlagT[num_items];

    // Use counting iterator as the input
    CountingInputIterator<T, int> h_in(0);

    // Select a comparison value that is select_ratio through the space of [0,127]
    T compare;
    if (select_ratio <= 0.0)
        InitValue(INTEGER_SEED, compare, 0);        // select none
    else if (select_ratio >= 1.0)
        InitValue(INTEGER_SEED, compare, 127);      // select all
    else
        InitValue(INTEGER_SEED, compare, int(double(double(127) * select_ratio)));

    LessThan<T> select_op(compare);
    int num_selected = Solve(h_in, select_op, h_reference, h_flags, num_items);

    if (g_verbose) std::cout << "\nComparison item: " << compare << "\n";
    printf("\nIterator %s cub::%s::%s %d items, %d selected (select ratio %.3f), %s %d-byte elements\n",
        (IS_PARTITION) ? "DevicePartition" : "DeviceSelect",
        (IS_FLAGGED) ? "Flagged" : "If",
        (BACKEND == CDP) ? "CDP CUB" : "CUB",
        num_items, num_selected, float(num_selected) / num_items, typeid(T).name(), (int) sizeof(T));
    fflush(stdout);

    // Run Test
    Test<BACKEND, IS_FLAGGED, IS_PARTITION>(h_in, h_flags, select_op, h_reference, num_selected, num_items);

    // Cleanup
    if (h_reference) delete[] h_reference;
    if (h_flags) delete[] h_flags;
}


/**
 * Test different selection ratios
 */
template <
    Backend         BACKEND,
    bool            IS_FLAGGED,
    bool            IS_PARTITION,
    typename        T>
void Test(
    int             num_items)
{
    for (float select_ratio = 0.0f; select_ratio <= 1.0f; select_ratio += 0.2f)
    {
        TestPointer<BACKEND, IS_FLAGGED, IS_PARTITION, T>(num_items, select_ratio);
    }
}


/**
 * Test (select vs. partition) and (flagged vs. functor)
 */
template <
    Backend         BACKEND,
    typename        T>
void TestMethod(
    int             num_items)
{
    // Functor
    Test<BACKEND, false, false, T>(num_items);
    Test<BACKEND, false, true, T>(num_items);

    // Flagged
    Test<BACKEND, true, false, T>(num_items);
    Test<BACKEND, true, true, T>(num_items);
}


/**
 * Test different dispatch
 */
template <
    typename        T>
void TestOp(
    int             num_items)
{
    TestMethod<CUB, T>(num_items);
#ifdef CUB_CDP
    TestMethod<CDP, T>(num_items);
#endif
}


/**
 * Test different input sizes
 */
template <typename T>
void Test(
    int             num_items)
{
    if (num_items < 0)
    {
        TestOp<T>(0);
        TestOp<T>(1);
        TestOp<T>(100);
        TestOp<T>(10000);
        TestOp<T>(1000000);
    }
    else
    {
        TestOp<T>(num_items);
    }
}

template<class T0, class T1>
struct pair_to_col_t
{
    __host__ __device__ T0 operator()(const thrust::tuple<T0, T1> &in)
    {
        return thrust::get<0>(in);
    }
};

template<class T0, class T1>
struct select_t {
    __host__ __device__ bool operator()(const thrust::tuple<T0, T1> &in) {
        return static_cast<T1>(thrust::get<0>(in)) > thrust::get<1>(in);
    }
};

template <typename T0, typename T1>
void TestMixedOp(int num_items)
{
    const T0 target_value = static_cast<T0>(42);
    thrust::device_vector<T0> col_a(num_items, target_value);
    thrust::device_vector<T1> col_b(num_items, static_cast<T1>(4.2));

    thrust::device_vector<T0> result(num_items);

    auto in = thrust::make_zip_iterator(col_a.begin(), col_b.begin());
    auto out = thrust::make_transform_output_iterator(result.begin(), pair_to_col_t<T0, T1>{});

    void *d_tmp_storage {};
    std::size_t tmp_storage_size{};
    cub::DeviceSelect::If(
            d_tmp_storage, tmp_storage_size,
            in, out, thrust::make_discard_iterator(),
            num_items, select_t<T0, T1>{},
            0, true);

    thrust::device_vector<char> tmp_storage(tmp_storage_size);
    d_tmp_storage = thrust::raw_pointer_cast(tmp_storage.data());

    cub::DeviceSelect::If(
            d_tmp_storage, tmp_storage_size,
            in, out, thrust::make_discard_iterator(),
            num_items, select_t<T0, T1>{},
            0, true);

    AssertEquals(num_items, thrust::count(result.begin(), result.end(), target_value));
}

/**
 * Test different input sizes
 */
template <typename T0, typename T1>
void TestMixed(int num_items)
{
    if (num_items < 0)
    {
        TestMixedOp<T0, T1>(0);
        TestMixedOp<T0, T1>(1);
        TestMixedOp<T0, T1>(100);
        TestMixedOp<T0, T1>(10000);
        TestMixedOp<T0, T1>(1000000);
    }
    else
    {
        TestMixedOp<T0, T1>(num_items);
    }
}

void TestFlagsNormalization()
{
  const int num_items = 1024 * 1024;
  thrust::device_vector<int> result(num_items);

  void *d_tmp_storage{};
  std::size_t tmp_storage_size{};
  CubDebugExit(
    cub::DeviceSelect::Flagged(d_tmp_storage,
                               tmp_storage_size,
                               cub::CountingInputIterator<int>(0),      // in
                               cub::CountingInputIterator<int>(1),      // flags
                               thrust::raw_pointer_cast(result.data()), // out
                               thrust::make_discard_iterator(), // num_out
                               num_items,
                               0,
                               true));

  thrust::device_vector<char> tmp_storage(tmp_storage_size);
  d_tmp_storage = thrust::raw_pointer_cast(tmp_storage.data());

  CubDebugExit(
    cub::DeviceSelect::Flagged(d_tmp_storage,
                               tmp_storage_size,
                               cub::CountingInputIterator<int>(0),      // in
                               cub::CountingInputIterator<int>(1),      // flags
                               thrust::raw_pointer_cast(result.data()), // out
                               thrust::make_discard_iterator(), // num_out
                               num_items,
                               0,
                               true));

  AssertTrue(thrust::equal(result.begin(),
                           result.end(),
                           thrust::make_counting_iterator(0)));
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
    float select_ratio      = 0.5;

    // Initialize command line
    CommandLineArgs args(argc, argv);
    g_verbose = args.CheckCmdLineFlag("v");
    args.GetCmdLineArgument("n", num_items);
    args.GetCmdLineArgument("i", g_timing_iterations);
    args.GetCmdLineArgument("ratio", select_ratio);

    // Print usage
    if (args.CheckCmdLineFlag("help"))
    {
        printf("%s "
            "[--n=<input items> "
            "[--i=<timing iterations> "
            "[--device=<device-id>] "
            "[--ratio=<selection ratio, default 0.5>] "
            "[--v] "
            "[--cdp] "
            "\n", argv[0]);
        exit(0);
    }

    // Initialize device
    CubDebugExit(args.DeviceInit());
    g_device_giga_bandwidth = args.device_giga_bandwidth;
    printf("\n");

    Test<unsigned char>(num_items);
    Test<unsigned short>(num_items);
    Test<unsigned int>(num_items);
    Test<unsigned long long>(num_items);

    Test<uchar2>(num_items);
    Test<ushort2>(num_items);
    Test<uint2>(num_items);
    Test<ulonglong2>(num_items);

    Test<uchar4>(num_items);
    Test<ushort4>(num_items);
    Test<uint4>(num_items);
    Test<ulonglong4>(num_items);

    Test<TestFoo>(num_items);
    Test<TestBar>(num_items);

    TestMixed<int, double>(num_items);
    TestFlagsNormalization();

    return 0;
}



