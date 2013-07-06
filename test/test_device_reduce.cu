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
 * Test of DeviceReduce utilities
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
// CUDA Nested Parallelism Test Kernel
//---------------------------------------------------------------------

/**
 * Simple wrapper kernel to invoke DeviceReduce
 */
template <
    typename            InputIteratorRA,
    typename            OutputIteratorRA,
    typename            ReductionOp>
__global__ void CnpDispatchKernel(
    void                *d_temp_storage,
    size_t              temp_storage_bytes,
    InputIteratorRA     d_in,
    OutputIteratorRA    d_out,
    int                 num_items,
    ReductionOp         reduction_op,
    bool                stream_synchronous,
    int                 timing_iterations,
    cudaError_t*        d_cnp_error)
{
#ifndef CUB_CNP
    *d_cnp_error = cudaErrorNotSupported;
#else
    cudaError_t error = cudaSuccess;
    for (int i = 0; i < timing_iterations; ++i)
    {
        error = DeviceReduce::Reduce(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items, reduction_op, 0, stream_synchronous);
    }
    *d_cnp_error = error;
#endif
}


//---------------------------------------------------------------------
// Host utility subroutines
//---------------------------------------------------------------------

/**
 * Initialize problem (and solution)
 */
template <
    typename        T,
    typename        ReductionOp>
void Initialize(
    GenMode         gen_mode,
    T               *h_in,
    T               h_reference[1],
    ReductionOp     reduction_op,
    int             num_items)
{
    for (int i = 0; i < num_items; ++i)
    {
        InitValue(gen_mode, h_in[i], i);
        if (i == 0)
            h_reference[0] = h_in[0];
        else
            h_reference[0] = reduction_op(h_reference[0], h_in[i]);
    }
}


/**
 * Dispatch from host
 */
template <typename InputIteratorRA, typename OutputIteratorRA, typename ReductionOp>
__host__ __device__ __forceinline__
cudaError_t Dispatch(
    Int2Type<false>             use_cnp,
    int                         timing_iterations,
    cudaError_t                 *d_cnp_error,
    void                        *d_temp_storage,                    ///< [in] %Device allocation of temporary storage.  When NULL, the required allocation size is returned in \p temp_storage_bytes and no work is done.
    size_t                      &temp_storage_bytes,                ///< [in,out] Size in bytes of \p d_temp_storage allocation.
    InputIteratorRA             d_in,                               ///< [in] Input data to reduce
    OutputIteratorRA            d_out,                              ///< [out] Output location for result
    int                         num_items,                          ///< [in] Number of items to reduce
    ReductionOp                 reduction_op,                       ///< [in] Binary reduction operator
    cudaStream_t                stream              = 0,            ///< [in] <b>[optional]</b> CUDA stream to launch kernels within.  Default is stream<sub>0</sub>.
    bool                        stream_synchronous  = false)        ///< [in] <b>[optional]</b> Whether or not to synchronize the stream after every kernel launch to check for errors.  Default is \p false.
{
    // Invoke kernel to device reduction directly
    cudaError_t error = cudaSuccess;
    for (int i = 0; i < timing_iterations; ++i)
    {
        error = DeviceReduce::Reduce(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items, reduction_op, 0, stream_synchronous);
    }
    return error;
}


/**
 * Dispatch from device (CNP)
 */
template <typename InputIteratorRA, typename OutputIteratorRA, typename ReductionOp>
__host__ __device__ __forceinline__
cudaError_t Dispatch(
    Int2Type<true>              use_cnp,
    int                         timing_iterations,
    cudaError_t                 *d_cnp_error,
    void                        *d_temp_storage,                    ///< [in] %Device allocation of temporary storage.  When NULL, the required allocation size is returned in \p temp_storage_bytes and no work is done.
    size_t                      &temp_storage_bytes,                ///< [in,out] Size in bytes of \p d_temp_storage allocation.
    InputIteratorRA             d_in,                               ///< [in] Input data to reduce
    OutputIteratorRA            d_out,                              ///< [out] Output location for result
    int                         num_items,                          ///< [in] Number of items to reduce
    ReductionOp                 reduction_op,                       ///< [in] Binary reduction operator
    cudaStream_t                stream              = 0,            ///< [in] <b>[optional]</b> CUDA stream to launch kernels within.  Default is stream<sub>0</sub>.
    bool                        stream_synchronous  = false)        ///< [in] <b>[optional]</b> Whether or not to synchronize the stream after every kernel launch to check for errors.  Default is \p false.
{
    // Invoke kernel to invoke device-side dispatch
    CnpDispatchKernel<<<1,1>>>(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items, reduction_op, stream_synchronous, timing_iterations, d_cnp_error);

    // Copy out error
    cudaError_t retval;
    CubDebugExit(cudaMemcpy(&retval, d_cnp_error, sizeof(cudaError_t) * 1, cudaMemcpyDeviceToHost));
    return retval;
}



/**
 * Test DeviceReduce
 */
template <
    bool        CNP,
    typename    T,
    typename    ReductionOp>
void Test(
    int         num_items,
    GenMode     gen_mode,
    ReductionOp reduction_op,
    char*       type_string)
{
    printf("%s cub::DeviceReduce::%s %d items, %s %d-byte elements, gen-mode %s\n",
        (CNP) ? "CNP device invoked" : "Host-invoked",
        (Equals<ReductionOp, Sum>::VALUE) ? "Sum" : "Reduce",
        num_items, type_string, (int) sizeof(T),
        (gen_mode == RANDOM) ? "RANDOM" : (gen_mode == SEQ_INC) ? "SEQUENTIAL" : "HOMOGENOUS");
    fflush(stdout);

    // Allocate host arrays
    T*              h_in = new T[num_items];
    T               h_reference[1];

    // Initialize problem
    Initialize(gen_mode, h_in, h_reference, reduction_op, num_items);

    // Allocate device arrays
    T               *d_in = NULL;
    T               *d_out = NULL;
    cudaError_t     *d_cnp_error = NULL;
    void            *d_temp_storage = NULL;
    size_t          temp_storage_bytes = 0;
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_in,          sizeof(T) * num_items));
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_out,         sizeof(T) * 1));
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_cnp_error,   sizeof(cudaError_t) * 1));

    // Allocate temporary storage
    CubDebugExit(DeviceReduce::Reduce(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items, reduction_op));
    CubDebugExit(g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes));

    // Initialize device arrays
    CubDebugExit(cudaMemcpy(d_in, h_in, sizeof(T) * num_items, cudaMemcpyHostToDevice));
    CubDebugExit(cudaMemset(d_out, 0, sizeof(T) * 1));

    // Run warmup/correctness iteration
    printf("Host dispatch:\n"); fflush(stdout);
    CubDebugExit(Dispatch(Int2Type<CNP>(), 1, d_cnp_error, d_temp_storage, temp_storage_bytes, d_in, d_out, num_items, reduction_op, 0, true));

    // Check for correctness (and display results, if specified)
    int compare = CompareDeviceResults(h_reference, d_out, 1, g_verbose, g_verbose);
    printf("\t%s", compare ? "FAIL" : "PASS");

    // Flush any stdout/stderr
    fflush(stdout);
    fflush(stderr);

    // Performance
    GpuTimer gpu_timer;
    gpu_timer.Start();
    CubDebugExit(Dispatch(Int2Type<CNP>(), g_timing_iterations, d_cnp_error, d_temp_storage, temp_storage_bytes, d_in, d_out, num_items, reduction_op));
    gpu_timer.Stop();
    float elapsed_millis = gpu_timer.ElapsedMillis();

    // Display performance
    if (g_timing_iterations > 0)
    {
        float avg_millis = elapsed_millis / g_timing_iterations;
        float grate = float(num_items) / avg_millis / 1000.0 / 1000.0;
        float gbandwidth = grate * sizeof(T);
        printf(", %.3f avg ms, %.3f billion items/s, %.3f GB/s", avg_millis, grate, gbandwidth);
    }

    printf("\n\n");

    // Cleanup
    if (h_in) delete[] h_in;
    if (d_in) CubDebugExit(g_allocator.DeviceFree(d_in));
    if (d_out) CubDebugExit(g_allocator.DeviceFree(d_out));
    if (d_cnp_error) CubDebugExit(g_allocator.DeviceFree(d_cnp_error));
    if (d_temp_storage) CubDebugExit(g_allocator.DeviceFree(d_temp_storage));

    // Correctness asserts
    AssertEquals(0, compare);
}


/**
 * Test different dispatch
 */
template <
    typename    T,
    typename    ReductionOp>
void TestCnp(
    int         num_items,
    GenMode     gen_mode,
    ReductionOp reduction_op,
    char*       type_string)
{
    Test<false, T>(num_items, gen_mode, reduction_op, type_string);
#ifdef CUB_CNP
    Test<true, T>(num_items, gen_mode, reduction_op, type_string);
#endif
}


/**
 * Test different gen modes
 */
template <
    typename        T,
    typename        ReductionOp>
void Test(
    int             num_items,
    ReductionOp     reduction_op,
    char*           type_string)
{
    TestCnp<T>(num_items, UNIFORM, reduction_op, type_string);
    TestCnp<T>(num_items, RANDOM, reduction_op, type_string);
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
    Test<T>(num_items, Sum(), type_string);
    Test<T>(num_items, Max(), type_string);
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

        TestCnp<char>(     num_items * 4, UNIFORM, Sum(), CUB_TYPE_STRING(char));
        TestCnp<short>(    num_items * 2, UNIFORM, Sum(), CUB_TYPE_STRING(short));
        TestCnp<int>(      num_items,     UNIFORM, Sum(), CUB_TYPE_STRING(int));
        TestCnp<long long>(num_items / 2, UNIFORM, Sum(), CUB_TYPE_STRING(long long));
        TestCnp<TestFoo>(  num_items / 4, UNIFORM, Max(), CUB_TYPE_STRING(TestFoo));
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



