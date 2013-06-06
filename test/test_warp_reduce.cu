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
 * Test of WarpReduce utilities
 ******************************************************************************/

// Ensure printing of CUDA runtime errors to console
#define CUB_STDERR

#include <stdio.h>
#include <iostream>
#include "test_util.h"
#include <cub/cub.cuh>

using namespace cub;

//---------------------------------------------------------------------
// Globals, constants and typedefs
//---------------------------------------------------------------------

/**
 * Verbose output
 */
bool g_verbose = false;



//---------------------------------------------------------------------
// Test kernels
//---------------------------------------------------------------------

/**
 * Generic reduction
 */
template <
    typename    T,
    typename    ReductionOp,
    typename    WarpReduce,
    bool        PRIMITIVE = Traits<T>::PRIMITIVE>
struct DeviceTest
{
    static __device__ __forceinline__ T Reduce(
        typename WarpReduce::TempStorage    &temp_storage,
        T                                   &data,
        ReductionOp                         &reduction_op)
    {
        return WarpReduce(temp_storage).Reduce(data, reduction_op);
    }

    static __device__ __forceinline__ T Reduce(
        typename WarpReduce::TempStorage    &temp_storage,
        T                                   &data,
        ReductionOp                         &reduction_op,
        const int                           &valid_lanes)
    {
        return WarpReduce(temp_storage).Reduce(data, reduction_op, valid_lanes);
    }

    template <typename Flag>
    static __device__ __forceinline__ T SegmentedReduce(
        typename WarpReduce::TempStorage    &temp_storage,
        T                                   &data,
        Flag                                &flag,
        ReductionOp                         &reduction_op)
    {
        return WarpReduce(temp_storage).SegmentedReduce(data, flag, reduction_op);
    }
};


/**
 * Summation
 */
template <
    typename    T,
    typename    WarpReduce>
struct DeviceTest<T, Sum<T>, WarpReduce, true>
{
    static __device__ __forceinline__ T Reduce(
        typename WarpReduce::TempStorage    &temp_storage,
        T                                   &data,
        Sum<T>                              &reduction_op)
    {
        return WarpReduce(temp_storage).Sum(data);
    }

    static __device__ __forceinline__ T Reduce(
        typename WarpReduce::TempStorage    &temp_storage,
        T                                   &data,
        Sum<T>                              &reduction_op,
        const int                           &valid_lanes)
    {
        return WarpReduce(temp_storage).Sum(data, valid_lanes);
    }

    template <typename Flag>
    static __device__ __forceinline__ T SegmentedReduce(
        typename WarpReduce::TempStorage    &temp_storage,
        T                                   &data,
        Flag                                &flag,
        Sum<T>                              &reduction_op)
    {
        return WarpReduce(temp_storage).SegmentedSum(data, flag);
    }
};


/**
 * WarpReduce test kernel (full tile)
 */
template <
    int         LOGICAL_WARP_THREADS,
    typename    T,
    typename    ReductionOp>
__global__ void WarpReduceKernel(
    T               *d_in,
    T               *d_out,
    ReductionOp     reduction_op,
    clock_t         *d_elapsed)
{
    // Cooperative warp-reduce utility type (1 warp)
    typedef WarpReduce<T, 1, LOGICAL_WARP_THREADS> WarpReduce;

    // Shared memory
    __shared__ typename WarpReduce::TempStorage temp_storage;

    // Per-thread tile data
    T input = d_in[threadIdx.x];

    // Record elapsed clocks
    clock_t start = clock();

    // Test warp reduce
    T output = DeviceTest<T, ReductionOp, WarpReduce>::Reduce(
        temp_storage, input, reduction_op);

    // Record elapsed clocks
    *d_elapsed = clock() - start;

    // Store aggregate
    if (threadIdx.x == 0)
    {
        *d_out = output;
    }
}

/**
 * WarpReduce test kernel (partially-full tile)
 */
template <
    int         LOGICAL_WARP_THREADS,
    typename    T,
    typename    ReductionOp>
__global__ void PartialWarpReduceKernel(
    T               *d_in,
    T               *d_out,
    ReductionOp     reduction_op,
    clock_t         *d_elapsed,
    int             valid_lanes)
{
    // Cooperative warp-reduce utility type (1 warp)
    typedef WarpReduce<T, 1, LOGICAL_WARP_THREADS> WarpReduce;

    // Shared memory
    __shared__ typename WarpReduce::TempStorage temp_storage;

    // Per-thread tile data
    T input = d_in[threadIdx.x];

    // Record elapsed clocks
    clock_t start = clock();

    // Test partial-warp reduce
    T output = DeviceTest<T, ReductionOp, WarpReduce>::Reduce(
        temp_storage, input, reduction_op, valid_lanes);

    // Record elapsed clocks
    *d_elapsed = clock() - start;

    // Store aggregate
    if (threadIdx.x == 0)
    {
        *d_out = output;
    }
}


/**
 * WarpReduce test kernel (segmented)
 */
template <
    int         LOGICAL_WARP_THREADS,
    typename    T,
    typename    Flag,
    typename    ReductionOp>
__global__ void SegmentedWarpReduceKernel(
    T               *d_in,
    Flag            *d_head_flags,
    T               *d_out,
    ReductionOp     reduction_op,
    clock_t         *d_elapsed)
{
    // Cooperative warp-reduce utility type (1 warp)
    typedef WarpReduce<T, 1, LOGICAL_WARP_THREADS> WarpReduce;

    // Shared memory
    __shared__ typename WarpReduce::TempStorage temp_storage;

    // Per-thread tile data
    T       input   = d_in[threadIdx.x];
    Flag    flag    = d_head_flags[threadIdx.x];

    // Record elapsed clocks
    clock_t start = clock();

    // Test segmented warp reduce
    T output = DeviceTest<T, ReductionOp, WarpReduce>::SegmentedReduce(
        temp_storage, input, flag, reduction_op);

    // Record elapsed clocks
    *d_elapsed = clock() - start;

    // Store aggregate
    d_out[threadIdx.x] = output;
}


//---------------------------------------------------------------------
// Host utility subroutines
//---------------------------------------------------------------------

/**
 * Initialize reduction problem (and solution)
 */
template <
    typename    T,
    typename    ReductionOp>
T Initialize(
    int             gen_mode,
    T               *h_in,
    int             num_items,
    ReductionOp     reduction_op)
{
    InitValue(gen_mode, h_in[0], 0);
    T aggregate = h_in[0];

    for (int i = 1; i < num_items; ++i)
    {
        InitValue(gen_mode, h_in[i], i);
        aggregate = reduction_op(aggregate, h_in[i]);
    }

    return aggregate;
}

/**
 * Test warp reduction
 */
template <
    int         LOGICAL_WARP_THREADS,
    typename    T,
    typename    ReductionOp>
void Test(
    int         gen_mode,
    ReductionOp reduction_op,
    const char  *type_string,
    int         valid_lanes)
{
    // Allocate host arrays
    T *h_in = new T[LOGICAL_WARP_THREADS];

    // Initialize problem
    T aggregate = Initialize(gen_mode, h_in, valid_lanes, reduction_op);

    // Initialize device arrays
    T *d_in = NULL;
    T *d_out = NULL;
    clock_t *d_elapsed = NULL;
    CubDebugExit(cudaMalloc((void**)&d_in, sizeof(T) * LOGICAL_WARP_THREADS));
    CubDebugExit(cudaMalloc((void**)&d_out, sizeof(T) * 1));
    CubDebugExit(cudaMalloc((void**)&d_elapsed, sizeof(clock_t)));
    CubDebugExit(cudaMemcpy(d_in, h_in, sizeof(T) * LOGICAL_WARP_THREADS, cudaMemcpyHostToDevice));
    CubDebugExit(cudaMemset(d_out, 0, sizeof(T) * 1));

    // Run kernel
    printf("Gen-mode %d, %d warp threads, %d valid lanes, %s (%d bytes) elements:\n",
        gen_mode,
        LOGICAL_WARP_THREADS,
        valid_lanes,
        type_string,
        (int) sizeof(T));
    fflush(stdout);

    if (valid_lanes == LOGICAL_WARP_THREADS)
    {
        // Run full-tile kernel
        WarpReduceKernel<LOGICAL_WARP_THREADS><<<1, LOGICAL_WARP_THREADS>>>(
            d_in,
            d_out,
            reduction_op,
            d_elapsed);
    }
    else
    {
        // Run partial-tile kernel
        PartialWarpReduceKernel<LOGICAL_WARP_THREADS><<<1, LOGICAL_WARP_THREADS>>>(
            d_in,
            d_out,
            reduction_op,
            d_elapsed,
            valid_lanes);
    }

    printf("\tElapsed clocks: ");
    DisplayDeviceResults(d_elapsed, 1);

    CubDebugExit(cudaDeviceSynchronize());

    // Copy out and display results
    printf("\tReduction results: ");
    int compare = CompareDeviceResults(&aggregate, d_out, 1, g_verbose, g_verbose);
    printf("%s\n", compare ? "FAIL" : "PASS");
    AssertEquals(0, compare);

    // Cleanup
    if (h_in) delete[] h_in;
    if (d_in) CubDebugExit(cudaFree(d_in));
    if (d_out) CubDebugExit(cudaFree(d_out));
    if (d_elapsed) CubDebugExit(cudaFree(d_elapsed));
}


/**
 * Run battery of tests for different full and partial tile sizes
 */
template <
    int         LOGICAL_WARP_THREADS,
    typename    T,
    typename    ReductionOp>
void Test(
    int             gen_mode,
    ReductionOp     reduction_op,
    const char*     type_string)
{
    // Partial tiles
    for (
        int valid_lanes = 1;
        valid_lanes < LOGICAL_WARP_THREADS;
        valid_lanes += CUB_MAX(1, LOGICAL_WARP_THREADS / 5))
    {
        Test<LOGICAL_WARP_THREADS, T>(gen_mode, reduction_op, type_string, valid_lanes);
    }

    // Full tile
    Test<LOGICAL_WARP_THREADS, T>(gen_mode, reduction_op, type_string, LOGICAL_WARP_THREADS);
}


/**
 * Run battery of tests for different data types and reduce ops
 */
template <int LOGICAL_WARP_THREADS>
void Test(int gen_mode)
{
    // primitive
    Test<LOGICAL_WARP_THREADS, unsigned char>(      gen_mode, Sum<unsigned char>(),         CUB_TYPE_STRING(unsigned char));
    Test<LOGICAL_WARP_THREADS, unsigned short>(     gen_mode, Sum<unsigned short>(),        CUB_TYPE_STRING(unsigned short));
    Test<LOGICAL_WARP_THREADS, unsigned int>(       gen_mode, Sum<unsigned int>(),          CUB_TYPE_STRING(unsigned int));
    Test<LOGICAL_WARP_THREADS, unsigned long long>( gen_mode, Sum<unsigned long long>(),    CUB_TYPE_STRING(unsigned long long));

    // primitive (alternative reduce op)
    Test<LOGICAL_WARP_THREADS, unsigned char>(      gen_mode, Max<unsigned char>(),         CUB_TYPE_STRING(unsigned char));
    Test<LOGICAL_WARP_THREADS, unsigned short>(     gen_mode, Max<unsigned short>(),        CUB_TYPE_STRING(unsigned short));
    Test<LOGICAL_WARP_THREADS, unsigned int>(       gen_mode, Max<unsigned int>(),          CUB_TYPE_STRING(unsigned int));
    Test<LOGICAL_WARP_THREADS, unsigned long long>( gen_mode, Max<unsigned long long>(),    CUB_TYPE_STRING(unsigned long long));

    // vec-2
    Test<LOGICAL_WARP_THREADS, uchar2>(             gen_mode, Sum<uchar2>(),                CUB_TYPE_STRING(uchar2));
    Test<LOGICAL_WARP_THREADS, ushort2>(            gen_mode, Sum<ushort2>(),               CUB_TYPE_STRING(ushort2));
    Test<LOGICAL_WARP_THREADS, uint2>(              gen_mode, Sum<uint2>(),                 CUB_TYPE_STRING(uint2));
    Test<LOGICAL_WARP_THREADS, ulonglong2>(         gen_mode, Sum<ulonglong2>(),            CUB_TYPE_STRING(ulonglong2));

    // vec-4
    Test<LOGICAL_WARP_THREADS, uchar4>(             gen_mode, Sum<uchar4>(),                CUB_TYPE_STRING(uchar4));
    Test<LOGICAL_WARP_THREADS, ushort4>(            gen_mode, Sum<ushort4>(),               CUB_TYPE_STRING(ushort4));
    Test<LOGICAL_WARP_THREADS, uint4>(              gen_mode, Sum<uint4>(),                 CUB_TYPE_STRING(uint4));
    Test<LOGICAL_WARP_THREADS, ulonglong4>(         gen_mode, Sum<ulonglong4>(),            CUB_TYPE_STRING(ulonglong4));

    // complex
    Test<LOGICAL_WARP_THREADS, TestFoo>(            gen_mode, Sum<TestFoo>(),               CUB_TYPE_STRING(TestFoo));
    Test<LOGICAL_WARP_THREADS, TestBar>(            gen_mode, Sum<TestBar>(),               CUB_TYPE_STRING(TestBar));
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

    // Print usage
    if (args.CheckCmdLineFlag("help"))
    {
        printf("%s "
            "[--device=<device-id>] "
            "[--v] "
            "\n", argv[0]);
        exit(0);
    }

    // Initialize device
    CubDebugExit(args.DeviceInit());

    // Quick exclusive test
    Test<32, int>(UNIFORM, Sum<int>(), CUB_TYPE_STRING(int), 32);

    // Test logical warp sizes
    Test<32>();
    Test<16>();
    Test<9>();
    Test<7>();

    return 0;
}




