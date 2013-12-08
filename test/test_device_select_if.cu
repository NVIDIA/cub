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
 * Test of DeviceSelect::If and DevicePartition::If utilities
 ******************************************************************************/

// Ensure printing of CUDA runtime errors to console
#define CUB_STDERR

#include <stdio.h>

#include <cub/util_allocator.cuh>
#include <cub/device/device_select.cuh>
#include <cub/device/device_partition.cuh>
#include <cub/iterator/counting_input_iterator.cuh>

#include <thrust/device_ptr.h>
#include <thrust/copy.h>
#include <thrust/partition.h>
#include <thrust/iterator/reverse_iterator.h>

#include "test_util.h"

using namespace cub;


//---------------------------------------------------------------------
// Globals, constants and typedefs
//---------------------------------------------------------------------

bool                    g_verbose           = false;
int                     g_timing_iterations = 0;
int                     g_repeat            = 0;
CachingDeviceAllocator  g_allocator(true);

// Selection functor type
template <typename T>
struct GreaterThan
{
    T compare;

    __host__ __device__ __forceinline__
    GreaterThan(T compare) : compare(compare) {}

    __host__ __device__ __forceinline__
    bool operator()(const T &a) const {
        return (a > compare);
    }
};

//---------------------------------------------------------------------
// Dispatch to different CUB DeviceSelect entrypoints
//---------------------------------------------------------------------


/**
 * Dispatch to select if entrypoint
 */
template <typename InputIterator, typename SelectOp, typename OutputIterator, typename NumSelectedIterator, typename Offset>
__host__ __device__ __forceinline__
cudaError_t Dispatch(
    Int2Type<CUB>               dispatch_to,
    Int2Type<false>             partition,
    int                         timing_timing_iterations,
    size_t                      *d_temp_storage_bytes,
    cudaError_t                 *d_cdp_error,

    void                        *d_temp_storage,
    size_t                      &temp_storage_bytes,
    InputIterator               d_in,
    OutputIterator              d_out,
    NumSelectedIterator         d_num_selected,
    Offset                      num_items,
    SelectOp                    select_op,
    cudaStream_t                stream,
    bool                        debug_synchronous)
{
    cudaError_t error = cudaSuccess;
    for (int i = 0; i < timing_timing_iterations; ++i)
    {
        error = DeviceSelect::If(d_temp_storage, temp_storage_bytes, d_in, d_out, d_num_selected, num_items, select_op, stream, debug_synchronous);
    }
    return error;
}


/**
 * Dispatch to partition if entrypoint
 */
template <typename InputIterator, typename SelectOp, typename OutputIterator, typename NumSelectedIterator, typename Offset>
__host__ __device__ __forceinline__
cudaError_t Dispatch(
    Int2Type<CUB>               dispatch_to,
    Int2Type<true>              partition,
    int                         timing_timing_iterations,
    size_t                      *d_temp_storage_bytes,
    cudaError_t                 *d_cdp_error,

    void                        *d_temp_storage,
    size_t                      &temp_storage_bytes,
    InputIterator               d_in,
    OutputIterator              d_out,
    NumSelectedIterator         d_num_selected,
    Offset                      num_items,
    SelectOp                    select_op,
    cudaStream_t                stream,
    bool                        debug_synchronous)
{
    cudaError_t error = cudaSuccess;
    for (int i = 0; i < timing_timing_iterations; ++i)
    {
        error = DevicePartition::If(d_temp_storage, temp_storage_bytes, d_in, d_out, d_num_selected, num_items, select_op, stream, debug_synchronous);
    }
    return error;
}



//---------------------------------------------------------------------
// Dispatch to different Thrust entrypoints
//---------------------------------------------------------------------

/**
 * Dispatch to select if entrypoint
 */
template <typename InputIterator, typename SelectOp, typename OutputIterator, typename NumSelectedIterator, typename Offset>
__host__ __forceinline__
cudaError_t Dispatch(
    Int2Type<THRUST>            dispatch_to,
    Int2Type<false>             partition,
    int                         timing_timing_iterations,
    size_t                      *d_temp_storage_bytes,
    cudaError_t                 *d_cdp_error,

    void                        *d_temp_storage,
    size_t                      &temp_storage_bytes,
    InputIterator               d_in,
    OutputIterator              d_out,
    NumSelectedIterator         d_num_selected,
    Offset                      num_items,
    SelectOp                    select_op,
    cudaStream_t                stream,
    bool                        debug_synchronous)
{
    typedef typename std::iterator_traits<InputIterator>::value_type    T;

    if (d_temp_storage == 0)
    {
        temp_storage_bytes = 1;
    }
    else
    {
        thrust::device_ptr<T>       d_out_wrapper_end;
        thrust::device_ptr<T>       d_in_wrapper(d_in);
        thrust::device_ptr<T>       d_out_wrapper(d_out);

        for (int i = 0; i < timing_timing_iterations; ++i)
        {
            d_out_wrapper_end = thrust::copy_if(d_in_wrapper, d_in_wrapper + num_items, d_out_wrapper, select_op);
        }

        Offset num_selected = d_out_wrapper_end - d_out_wrapper;
        CubDebugExit(cudaMemcpy(d_num_selected, &num_selected, sizeof(Offset), cudaMemcpyHostToDevice));
    }

    return cudaSuccess;
}


/**
 * Dispatch to partition if entrypoint
 */
template <typename InputIterator, typename SelectOp, typename OutputIterator, typename NumSelectedIterator, typename Offset>
__host__ __forceinline__
cudaError_t Dispatch(
    Int2Type<THRUST>            dispatch_to,
    Int2Type<true>              partition,
    int                         timing_timing_iterations,
    size_t                      *d_temp_storage_bytes,
    cudaError_t                 *d_cdp_error,

    void                        *d_temp_storage,
    size_t                      &temp_storage_bytes,
    InputIterator               d_in,
    OutputIterator              d_out,
    NumSelectedIterator         d_num_selected,
    Offset                      num_items,
    SelectOp                    select_op,
    cudaStream_t                stream,
    bool                        debug_synchronous)
{
    typedef typename std::iterator_traits<InputIterator>::value_type    T;

    typedef thrust::reverse_iterator<thrust::device_ptr<T> > ReverseOutputIterator;

    if (d_temp_storage == 0)
    {
        temp_storage_bytes = 1;
    }
    else
    {
        thrust::pair<thrust::device_ptr<T>, ReverseOutputIterator> d_out_wrapper_end;

        thrust::device_ptr<T>       d_in_wrapper(d_in);
        thrust::device_ptr<T>       d_out_wrapper(d_out);

        ReverseOutputIterator d_out_unselected(d_out_wrapper + num_items);

        for (int i = 0; i < timing_timing_iterations; ++i)
        {
            d_out_wrapper_end = thrust::partition_copy(
                d_in_wrapper,
                d_in_wrapper + num_items,
                d_out_wrapper,
                d_out_unselected,
                select_op);
        }

        Offset num_selected = d_out_wrapper_end.first - d_out_wrapper;
        CubDebugExit(cudaMemcpy(d_num_selected, &num_selected, sizeof(Offset), cudaMemcpyHostToDevice));
    }

    return cudaSuccess;
}



//---------------------------------------------------------------------
// CUDA Nested Parallelism Test Kernel
//---------------------------------------------------------------------

/**
 * Simple wrapper kernel to invoke DeviceSelect
 */
template <typename InputIterator, typename SelectOp, typename OutputIterator, typename NumSelectedIterator, typename Offset, typename PartitionTag>
__global__ void CnpDispatchKernel(
    PartitionTag                partition,
    int                         timing_timing_iterations,
    size_t                      *d_temp_storage_bytes,
    cudaError_t                 *d_cdp_error,

    void                        *d_temp_storage,
    size_t                      temp_storage_bytes,
    InputIterator               d_in,
    OutputIterator              d_out,
    NumSelectedIterator         d_num_selected,
    Offset                      num_items,
    SelectOp                    select_op,
    bool                        debug_synchronous)
{

#ifndef CUB_CDP
    *d_cdp_error = cudaErrorNotSupported;
#else
    *d_cdp_error = Dispatch(Int2Type<CUB>(), partition, timing_timing_iterations, d_temp_storage_bytes, d_cdp_error,
        d_temp_storage, temp_storage_bytes, d_in, d_out, d_num_selected, num_items, select_op, 0, debug_synchronous);
    *d_temp_storage_bytes = temp_storage_bytes;
#endif
}


/**
 * Dispatch to CDP kernel
 */
template <typename InputIterator, typename SelectOp, typename OutputIterator, typename NumSelectedIterator, typename Offset, typename PartitionTag>
cudaError_t Dispatch(
    Int2Type<CDP>               dispatch_to,
    PartitionTag                partition,
    int                         timing_timing_iterations,
    size_t                      *d_temp_storage_bytes,
    cudaError_t                 *d_cdp_error,

    void                        *d_temp_storage,
    size_t                      &temp_storage_bytes,
    InputIterator               d_in,
    OutputIterator              d_out,
    NumSelectedIterator         d_num_selected,
    Offset                      num_items,
    SelectOp                    select_op,
    cudaStream_t                stream,
    bool                        debug_synchronous)
{
    // Invoke kernel to invoke device-side dispatch
    CnpDispatchKernel<<<1,1>>>(partition, timing_timing_iterations, d_temp_storage_bytes, d_cdp_error,
        d_temp_storage, temp_storage_bytes, d_in, d_out, d_num_selected, num_items, select_op, debug_synchronous);

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
    int         entropy_reduction,
    T           *h_in,
    int         num_items,
    int         max_segment)
{
    unsigned short max_short = (unsigned short) -1;

    int key = 0;
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
            InitValue(SEQ_INC, h_in[j], key);
            j++;
        }

        i = j;
        key++;
    }

    if (g_verbose)
    {
        printf("Input:\n");
        DisplayResults(h_in, num_items);
        printf("\n\n");
    }
}


/**
 * Solve unique problem
 */
template <
    typename        InputIterator,
    typename        SelectOp,
    typename        T>
int Solve(
    InputIterator   h_in,
    SelectOp        select_op,
    T               *h_reference,
    int             num_items)
{
    int num_selected = 0;
    for (int i = 0; i < num_items; ++i)
    {
        if (select_op(h_in[i]))
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
    bool                PARTITION,
    typename            DeviceInputIterator,
    typename            SelectOp,
    typename            T>
void Test(
    DeviceInputIterator d_in,
    SelectOp            select_op,
    T                   *h_reference,
    int                 num_selected,
    int                 num_items,
    char*               type_string)
{
    // Allocate device output array and num selected
    T       *d_out            = NULL;
    int     *d_num_selected   = NULL;
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_out, sizeof(T) * num_items));
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_num_selected, sizeof(int)));

    // Allocate CDP device arrays
    size_t          *d_temp_storage_bytes = NULL;
    cudaError_t     *d_cdp_error = NULL;
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_temp_storage_bytes,  sizeof(size_t) * 1));
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_cdp_error,           sizeof(cudaError_t) * 1));

    // Allocate temporary storage
    void            *d_temp_storage = NULL;
    size_t          temp_storage_bytes = 0;
    CubDebugExit(Dispatch(Int2Type<BACKEND>(), Int2Type<PARTITION>(), 1, d_temp_storage_bytes, d_cdp_error,
        d_temp_storage, temp_storage_bytes, d_in, d_out, d_num_selected, num_items, select_op, 0, true));
    CubDebugExit(g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes));

    // Clear device output array
    CubDebugExit(cudaMemset(d_out, 0, sizeof(T) * num_items));
    CubDebugExit(cudaMemset(d_num_selected, 0, sizeof(int)));

    // Run warmup/correctness iteration
    CubDebugExit(Dispatch(Int2Type<BACKEND>(), Int2Type<PARTITION>(), 1, d_temp_storage_bytes, d_cdp_error,
        d_temp_storage, temp_storage_bytes, d_in, d_out, d_num_selected, num_items, select_op, 0, true));

    // Check for correctness (and display results, if specified)
    int compare = (PARTITION) ?
        CompareDeviceResults(h_reference, d_out, num_items, true, g_verbose) :
        CompareDeviceResults(h_reference, d_out, num_selected, true, g_verbose);
    printf("\t Data %s ", compare ? "FAIL" : "PASS");

    compare = compare | CompareDeviceResults(&num_selected, d_num_selected, 1, true, g_verbose);
    printf("\t Count %s ", compare ? "FAIL" : "PASS");

    // Flush any stdout/stderr
    fflush(stdout);
    fflush(stderr);

    // Performance
    GpuTimer gpu_timer;
    gpu_timer.Start();
    CubDebugExit(Dispatch(Int2Type<BACKEND>(), Int2Type<PARTITION>(), g_timing_iterations, d_temp_storage_bytes, d_cdp_error,
        d_temp_storage, temp_storage_bytes, d_in, d_out, d_num_selected, num_items, select_op, 0, false));
    gpu_timer.Stop();
    float elapsed_millis = gpu_timer.ElapsedMillis();

    // Display performance
    if (g_timing_iterations > 0)
    {
        float avg_millis = elapsed_millis / g_timing_iterations;
        float grate = float(num_items) / avg_millis / 1000.0 / 1000.0;
        float gbandwidth = float((num_items + num_selected) * sizeof(T)) / avg_millis / 1000.0 / 1000.0;
        printf(", %.3f avg ms, %.3f billion items/s, %.3f logical GB/s", avg_millis, grate, gbandwidth);
    }
    printf("\n\n");

    // Flush any stdout/stderr
    fflush(stdout);
    fflush(stderr);

    // Cleanup
    if (d_out) CubDebugExit(g_allocator.DeviceFree(d_out));
    if (d_num_selected) CubDebugExit(g_allocator.DeviceFree(d_num_selected));
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
    bool            PARTITION,
    typename        T>
void TestPointer(
    int             num_items,
    int             pivot_idx,
    int             entropy_reduction,
    int             max_segment,
    char*           type_string)
{
    // Allocate host arrays
    T       *h_in        = new T[num_items];
    T       *h_reference = new T[num_items];

    // Initialize problem and solution
    Initialize(entropy_reduction, h_in, num_items, max_segment);
    GreaterThan<T> select_op(h_in[pivot_idx]);
    int num_selected = Solve(h_in, select_op, h_reference, num_items);

    printf("\nPointer %s cub::%s::If %d items, %d selected (avg run length %d), %s %d-byte elements, entropy_reduction %d\n",
        (PARTITION) ? "DevicePartition" : "DeviceSelect",
        (BACKEND == CDP) ? "CDP CUB" : (BACKEND == THRUST) ? "Thrust" : "CUB",
        num_items, num_selected, (num_selected > 0) ? num_items / num_selected : 0,
        type_string,
        (int) sizeof(T),
        entropy_reduction);
    fflush(stdout);

    // Allocate problem device arrays
    T       *d_in = NULL;

    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_in, sizeof(T) * num_items));

    // Initialize device input
    CubDebugExit(cudaMemcpy(d_in, h_in, sizeof(T) * num_items, cudaMemcpyHostToDevice));

    // Run Test
    Test<BACKEND, PARTITION>(d_in, select_op, h_reference, num_selected, num_items, type_string);

    // Cleanup
    if (h_in) delete[] h_in;
    if (h_reference) delete[] h_reference;
    if (d_in) CubDebugExit(g_allocator.DeviceFree(d_in));
}


/**
 * Test DeviceSelect on pointer type
 */
template <typename T>
void ComparePointer(
    int             num_items,
    int             entropy_reduction,
    int             max_segment,
    char*           type_string)
{
    unsigned int pivot_idx;
    unsigned int max_int = (unsigned int) -1;
    RandomBits(pivot_idx);
    pivot_idx = (unsigned int) ((float(pivot_idx) * (float(num_items - 1) / float(max_int))));
    printf("\nPivot idx: %d\n", pivot_idx); fflush(stdout);

    printf("-- Select ----------------------------\n");
    TestPointer<CUB, false, T>(num_items, pivot_idx, entropy_reduction, max_segment, type_string);
    TestPointer<THRUST, false, T>(num_items, pivot_idx, entropy_reduction, max_segment, type_string);

    printf("-- Partition ----------------------------\n");
    TestPointer<CUB, true, T>(num_items, pivot_idx, entropy_reduction, max_segment, type_string);
    TestPointer<THRUST, true, T>(num_items, pivot_idx, entropy_reduction, max_segment, type_string);
}


/**
 * Test DeviceSelect on iterator type
 */
template <
    Backend         BACKEND,
    bool            PARTITION,
    typename        T>
void TestIterator(
    int             num_items,
    int             pivot_idx,
    char*           type_string,
    Int2Type<true>  is_number)
{
    // Allocate host arrays
    T*  h_reference = new T[num_items];

    // Initialize problem and solution using a counting iterator as the input
    CountingInputIterator<T, int> h_in(0);
    GreaterThan<T> select_op(h_in[pivot_idx]);
    int num_selected = Solve(h_in, select_op, h_reference, num_items);

    printf("\nIterator %s cub::DeviceSelect::If %d items, %d selected (avg run length %d), %s %d-byte elements\n",
        (BACKEND == CDP) ? "CDP CUB" : (BACKEND == THRUST) ? "Thrust" : "CUB",
        num_items, num_selected, (num_selected > 0) ? num_items / num_selected : 0,
        type_string,
        (int) sizeof(T));
    fflush(stdout);

    // Run Test
    Test<BACKEND, PARTITION>(h_in, select_op, h_reference, num_selected, num_items, type_string);

    // Cleanup
    if (h_reference) delete[] h_reference;
}


/**
 * Test DeviceSelect on iterator type
 */
template <
    Backend         BACKEND,
    bool            PARTITION,
    typename        T>
void TestIterator(
    int             num_items,
    int             pivot_idx,
    char*           type_string,
    Int2Type<false> is_number)
{}


/**
 * Test different gen modes
 */
template <
    Backend         BACKEND,
    bool            PARTITION,
    typename        T>
void Test(
    int             num_items,
    int             pivot_idx,
    char*           type_string)
{
    for (int max_segment = 1; max_segment < CUB_MIN(num_items, (unsigned short) -1); max_segment *= 11)
    {
        TestPointer<BACKEND, PARTITION, T>(num_items, pivot_idx, 0, max_segment, type_string);
        TestPointer<BACKEND, PARTITION, T>(num_items, pivot_idx, 2, max_segment, type_string);
        TestPointer<BACKEND, PARTITION, T>(num_items, pivot_idx, 7, max_segment, type_string);
    }

    TestIterator<BACKEND, PARTITION, T>(num_items, pivot_idx, type_string, Int2Type<Traits<T>::CATEGORY != NOT_A_NUMBER>());

}


/**
 * Test select vs. partition
 */
template <
    Backend         BACKEND,
    typename        T>
void TestMethod(
    int             num_items,
    int             pivot_idx,
    char*           type_string)
{
    Test<BACKEND, false, T>(num_items, pivot_idx, type_string);
    Test<BACKEND, true, T>(num_items, pivot_idx, type_string);
}


/**
 * Test different dispatch
 */
template <
    typename        T>
void TestOp(
    int             num_items,
    char*           type_string)
{
    unsigned int pivot_idx;
    unsigned int max_int = (unsigned int) -1;
    RandomBits(pivot_idx);
    pivot_idx = (unsigned int) ((float(pivot_idx) * (float(num_items - 1) / float(max_int))));
    printf("Pivot idx: %d\n", pivot_idx); fflush(stdout);

    TestMethod<CUB, T>(num_items, pivot_idx, type_string);
#ifdef CUB_CDP
    TestMethod<CDP, T>(num_items, pivot_idx, type_string);
#endif
}


/**
 * Test different input sizes
 */
template <typename T>
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
    int num_items           = -1;
    int entropy_reduction   = 0;
    int max_segment              = 1000;

    // Initialize command line
    CommandLineArgs args(argc, argv);
    g_verbose = args.CheckCmdLineFlag("v");
    bool quick = args.CheckCmdLineFlag("quick");
    args.GetCmdLineArgument("n", num_items);
    args.GetCmdLineArgument("i", g_timing_iterations);
    args.GetCmdLineArgument("repeat", g_repeat);
    args.GetCmdLineArgument("maxseg", max_segment);
    args.GetCmdLineArgument("entropy", entropy_reduction);

    // Print usage
    if (args.CheckCmdLineFlag("help"))
    {
        printf("%s "
            "[--n=<input items> "
            "[--i=<timing iterations> "
            "[--device=<device-id>] "
            "[--maxseg=<max segment length>] "
            "[--entropy=<segment length bit entropy reduction rounds>] "
            "[--repeat=<repetitions of entire test suite>] "
            "[--quick] "
            "[--v] "
            "[--cdp] "
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

    if (quick)
    {
        // Quick test
        if (num_items < 0) num_items = 32000000;

        ComparePointer<char>(       num_items * ((sm_version <= 130) ? 1 : 4),  entropy_reduction, max_segment, CUB_TYPE_STRING(char));
        ComparePointer<short>(      num_items * ((sm_version <= 130) ? 1 : 2),  entropy_reduction, max_segment, CUB_TYPE_STRING(short));
        ComparePointer<int>(        num_items,                                  entropy_reduction, max_segment, CUB_TYPE_STRING(int));
        ComparePointer<long long>(  num_items / 2,                              entropy_reduction, max_segment, CUB_TYPE_STRING(long long));
        ComparePointer<TestFoo>(    num_items / 4,                              entropy_reduction, max_segment, CUB_TYPE_STRING(TestFoo));

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
    }

    return 0;
}



