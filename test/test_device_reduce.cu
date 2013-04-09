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
#include <cub.cuh>
#include "test_util.h"

using namespace cub;


//---------------------------------------------------------------------
// Globals, constants and typedefs
//---------------------------------------------------------------------

bool    g_verbose = false;
int     g_iterations = 100;




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
    int             gen_mode,
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


//---------------------------------------------------------------------
// Full tile test generation
//---------------------------------------------------------------------


/**
 * Test full-tile reduction
 */
template <
    typename        T,
    typename        ReductionOp>
void Test(
    int             num_items,
    int             gen_mode,
    int             tiles,
    ReductionOp     reduction_op,
    char            *type_string)
{
    // Allocate host arrays
    T *h_in = new T[num_items];
    T h_reference[1];

    // Initialize problem
    Initialize(gen_mode, h_in, h_reference, reduction_op, num_items);

    // Initialize device arrays
    T *d_in = NULL;
    T *d_out = NULL;
    CubDebugExit(DeviceAllocate((void**)&d_in, sizeof(T) * num_items));
    CubDebugExit(DeviceAllocate((void**)&d_out, sizeof(T) * 1));
    CubDebugExit(cudaMemcpy(d_in, h_in, sizeof(T) * num_items, cudaMemcpyHostToDevice));

    printf("DeviceReduce gen-mode %d, num_items(%d), %s (%d bytes) elements:\n",
        gen_mode,
        num_items,
        type_string,
        (int) sizeof(T));
    fflush(stdout);

    // Run warmup/correctness iteration
    DeviceReduce::Reduce(d_in, d_out, num_items, reduction_op, 0, true);
    CubDebugExit(cudaDeviceSynchronize());

    // Copy out and display results
    printf("\nReduction results: ");
    int compare = CompareDeviceResults(h_reference, d_out, 1, g_verbose, g_verbose);
    printf("%s\n", compare ? "FAIL" : "PASS");

    // Performance
    GpuTimer gpu_timer;
    float elapsed_millis = 0.0;
    for (int i = 0; i < g_iterations; i++)
    {
        gpu_timer.Start();

        DeviceReduce::Reduce(d_in, d_out, num_items, reduction_op);

        gpu_timer.Stop();
        elapsed_millis += gpu_timer.ElapsedMillis();
    }
    if (g_iterations > 0)
    {
        float avg_millis = elapsed_millis / g_iterations;
        float grate = float(num_items) / avg_millis / 1000.0 / 1000.0;
        float gbandwidth = grate * sizeof(T);
        printf("\nPerformance: %.3f avg ms, %.3f billion items/s, %.3f GB/s\n", avg_millis, grate, gbandwidth);
    }

    // Cleanup
    if (h_in) delete[] h_in;
    if (d_in) CubDebugExit(DeviceFree(d_in));
    if (d_out) CubDebugExit(DeviceFree(d_out));

    AssertEquals(0, compare);
}




//---------------------------------------------------------------------
// Main
//---------------------------------------------------------------------

/**
 * Run battery of full-tile tests for different gen modes
 */
template <
    typename        T,
    typename        ReductionOp>
void Test(
    int             num_items,
    ReductionOp     reduction_op,
    char*           type_string)
{
    Test<T>(num_items, UNIFORM, reduction_op, type_string);
    Test<T>(num_items, SEQ_INC, reduction_op, type_string);
    Test<T>(num_items, RANDOM, reduction_op, type_string);
}


/**
 * Main
 */
int main(int argc, char** argv)
{
    int num_items = 1 * 1024 * 1024;

    // Initialize command line
    CommandLineArgs args(argc, argv);
    args.GetCmdLineArgument("n", num_items);
    args.GetCmdLineArgument("i", g_iterations);
    g_verbose = args.CheckCmdLineFlag("v");
    bool quick = args.CheckCmdLineFlag("quick");

    // Print usage
    if (args.CheckCmdLineFlag("help"))
    {
        printf("%s "
            "[--device=<device-id>] "
            "[--v] "
            "[--quick]"
            "\n", argv[0]);
        exit(0);
    }

    // Initialize device
    CubDebugExit(args.DeviceInit());

    if (quick)
    {
        // Quick test
        typedef int T;
        Test<T>(num_items, UNIFORM, 1, Sum<T>(), CUB_TYPE_STRING(T));
    }
    else
    {
/*
        // primitives
        Test<char>(Sum<char>(), CUB_TYPE_STRING(char));
        Test<short>(Sum<short>(), CUB_TYPE_STRING(short));
        Test<int>(Sum<int>(), CUB_TYPE_STRING(int));
        Test<long long>(Sum<long long>(), CUB_TYPE_STRING(long long));

        // vector types
        Test<char2>(Sum<char2>(), CUB_TYPE_STRING(char2));
        Test<short2>(Sum<short2>(), CUB_TYPE_STRING(short2));
        Test<int2>(Sum<int2>(), CUB_TYPE_STRING(int2));
        Test<longlong2>(Sum<longlong2>(), CUB_TYPE_STRING(longlong2));

        Test<char4>(Sum<char4>(), CUB_TYPE_STRING(char4));
        Test<short4>(Sum<short4>(), CUB_TYPE_STRING(short4));
        Test<int4>(Sum<int4>(), CUB_TYPE_STRING(int4));
        Test<longlong4>(Sum<longlong4>(), CUB_TYPE_STRING(longlong4));

        // Complex types
        Test<TestFoo>(Sum<TestFoo>(), CUB_TYPE_STRING(TestFoo));
        Test<TestBar>(Sum<TestBar>(), CUB_TYPE_STRING(TestBar));
*/
    }

    return 0;
}



