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

#include <vector>
#include <algorithm>
#include <stdio.h>
#include <cub.cuh>
#include "../test/test_util.h"

using namespace cub;
using namespace std;


//---------------------------------------------------------------------
// Globals, constants and typedefs
//---------------------------------------------------------------------

#ifndef TUNE_ARCH
#define TUNE_ARCH 100
#endif

int g_iterations = 100;


//---------------------------------------------------------------------
// Host utility subroutines
//---------------------------------------------------------------------

/**
 * Initialize problem (and solution)
 */
template <typename T, typename ReductionOp>
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
 * Performance data for a given tuning configuration
 */
struct Result
{
    DeviceReduce::KernelDispachParams   multi_params;
    DeviceReduce::KernelDispachParams   single_params;
    bool                                correct;
    float                               avg_elapsed;
    float                               avg_throughput;
    float                               avg_bandwidth;

    void Print()
    {
        printf("multi( ");
        multi_params.Print();
        printf(" ), single( ");
        single_params.Print();
        printf(" ), correct: %d, avg_ms: %.3f, avg_gitems/s: %.3f, avg_gb/s: %.3f\n", correct, avg_elapsed, avg_throughput, avg_bandwidth);
    }


};


/**
 * Comparison operator for Result
 */
bool operator< (const Result &a, const Result &b)
{
    return (a.avg_throughput < b.avg_throughput);
}


/**
 * Wrapper structure for generating and running different tuning configurations
 */
template <
    typename T,
    typename SizeT,
    typename ReductionOp>
struct Schmoo
{
    //---------------------------------------------------------------------
    // Types
    //---------------------------------------------------------------------

    /// Pairing of dispatch params and kernel function pointer
    template <typename KernelPtr>
    struct DispatchTuple
    {
        DeviceReduce::KernelDispachParams   params;
        KernelPtr                           kernel_ptr;
    };

    /// Multi-block reduction kernel type and dispatch tuple type
    typedef void (*MultiReduceKernelPtr)(T*, T*, SizeT, GridEvenShare<SizeT>, GridQueue<SizeT>, ReductionOp);
    typedef DispatchTuple<MultiReduceKernelPtr> MultiDispatchTuple;

    /// Single-block reduction kernel type and dispatch tuple type
    typedef void (*SingleReduceKernelPtr)(T*, T*, SizeT, ReductionOp);
    typedef DispatchTuple<SingleReduceKernelPtr> SingleDispatchTuple;

    //---------------------------------------------------------------------
    // Fields
    //---------------------------------------------------------------------

    vector<MultiDispatchTuple> multi_kernels;       // List of generated multi-block kernels
    vector<SingleDispatchTuple> single_kernels;     // List of generated single-block kernels
    vector<Result> runs;

    //---------------------------------------------------------------------
    // Methods
    //---------------------------------------------------------------------

    /**
     * Test reduction
     */
    void Test(
        MultiDispatchTuple      multi_dispatch,
        SingleDispatchTuple     single_dispatch,
        T*                      d_in,
        T*                      d_out,
        T*                      h_reference,
        SizeT                   num_items,
        ReductionOp             reduction_op)
    {
        Result run;
        run.multi_params = multi_dispatch.params;
        run.single_params = single_dispatch.params;

        // Warmup/correctness iteration
        DeviceReduce::Dispatch(
            multi_dispatch.kernel_ptr,
            single_dispatch.kernel_ptr,
            multi_dispatch.params,
            single_dispatch.params,
            d_in,
            d_out,
            num_items,
            reduction_op);

        CubDebugExit(cudaDeviceSynchronize());

        // Copy out and display results
        run.correct = CompareDeviceResults(h_reference, d_out, 1, false, false);

        // Performance
        GpuTimer gpu_timer;
        float elapsed_millis = 0.0;
        for (int i = 0; i < g_iterations; i++)
        {
            gpu_timer.Start();

            DeviceReduce::Dispatch(
                multi_dispatch.kernel_ptr,
                single_dispatch.kernel_ptr,
                multi_dispatch.params,
                single_dispatch.params,
                d_in,
                d_out,
                num_items,
                reduction_op);

            gpu_timer.Stop();
            elapsed_millis += gpu_timer.ElapsedMillis();
        }

        if (g_iterations > 0)
        {
            run.avg_elapsed = elapsed_millis / g_iterations;
            run.avg_throughput = float(num_items) / run.avg_elapsed / 1000.0 / 1000.0;
            run.avg_bandwidth = run.avg_throughput * sizeof(T);
        }

        run.Print();
        fflush(stdout);
        runs.push_back(run);

        AssertEquals(0, run.correct);
    }


    /**
     * Specialization that allows kernel generation with the specified BlockReduceTilesPolicy
     */
    template <
        typename BlockReduceTilesPolicy,
        bool IsOk = (sizeof(typename BlockReduceTiles<BlockReduceTilesPolicy, T*, SizeT>::SmemStorage) < ArchProps<TUNE_ARCH>::SMEM_BYTES)>
    struct Ok
    {
        /// Generate multi-block kernel and add to the list
        template <typename KernelsVector>
        static void GenerateMulti(KernelsVector &multi_kernels)
        {
            MultiDispatchTuple tuple;
            tuple.params.template Init<BlockReduceTilesPolicy>();
            tuple.kernel_ptr = MultiReduceKernel<BlockReduceTilesPolicy, T*, T*, SizeT, ReductionOp>;
            multi_kernels.push_back(tuple);
        }


        /// Generate single-block kernel and add to the list
        template <typename KernelsVector>
        static void GenerateSingle(KernelsVector &single_kernels)
        {
            SingleDispatchTuple tuple;
            tuple.params.template Init<BlockReduceTilesPolicy>();
            tuple.kernel_ptr = SingleReduceKernel<BlockReduceTilesPolicy, T*, T*, SizeT, ReductionOp>;
            single_kernels.push_back(tuple);
        }
    };

    /**
     * Specialization that rejects kernel generation with the specified BlockReduceTilesPolicy
     */
    template <typename BlockReduceTilesPolicy>
    struct Ok<BlockReduceTilesPolicy, false>
    {
        template <typename KernelsVector>
        static void GenerateMulti(KernelsVector &multi_kernels) {}

        template <typename KernelsVector>
        static void GenerateSingle(KernelsVector &single_kernels) {}
    };


    /// Generate block-scheduling variations
    template <
        int             BLOCK_THREADS,
        int             ITEMS_PER_THREAD,
        int             VECTOR_LOAD_LENGTH,
        PtxLoadModifier LOAD_MODIFIER>
    void Generate()
    {
//      Ok<BlockReduceTilesPolicy<BLOCK_THREADS, ITEMS_PER_THREAD, VECTOR_LOAD_LENGTH, LOAD_MODIFIER, GRID_MAPPING_EVEN_SHARE, 1> >::GenerateMulti(multi_kernels);
//      Ok<BlockReduceTilesPolicy<BLOCK_THREADS, ITEMS_PER_THREAD, VECTOR_LOAD_LENGTH, LOAD_MODIFIER, GRID_MAPPING_EVEN_SHARE, 2> >::GenerateMulti(multi_kernels);
        Ok<BlockReduceTilesPolicy<BLOCK_THREADS, ITEMS_PER_THREAD, VECTOR_LOAD_LENGTH, LOAD_MODIFIER, GRID_MAPPING_EVEN_SHARE, 4> >::GenerateMulti(multi_kernels);
//      Ok<BlockReduceTilesPolicy<BLOCK_THREADS, ITEMS_PER_THREAD, VECTOR_LOAD_LENGTH, LOAD_MODIFIER, GRID_MAPPING_EVEN_SHARE, 8> >::GenerateMulti(multi_kernels);
#if TUNE_ARCH >= 130
        Ok<BlockReduceTilesPolicy<BLOCK_THREADS, ITEMS_PER_THREAD, VECTOR_LOAD_LENGTH, LOAD_MODIFIER, GRID_MAPPING_DYNAMIC, 1> >::GenerateMulti(multi_kernels);
#endif

        Ok<BlockReduceTilesPolicy<BLOCK_THREADS, ITEMS_PER_THREAD, VECTOR_LOAD_LENGTH, LOAD_MODIFIER, GRID_MAPPING_EVEN_SHARE, 1> >::GenerateSingle(single_kernels);
    }


    /// Generate load modifier variations
    template <
        int BLOCK_THREADS,
        int ITEMS_PER_THREAD,
        int VECTOR_LOAD_LENGTH>
    void Generate()
    {
        Generate<BLOCK_THREADS, ITEMS_PER_THREAD, VECTOR_LOAD_LENGTH, PTX_LOAD_NONE>();
#if TUNE_ARCH >= 350
        Generate<BLOCK_THREADS, ITEMS_PER_THREAD, VECTOR_LOAD_LENGTH, PTX_LOAD_LDG>();
#endif
    }


    /// Generate vectorization variations
    template <
        int BLOCK_THREADS,
        int ITEMS_PER_THREAD>
    void Generate()
    {
        Generate<BLOCK_THREADS, ITEMS_PER_THREAD, 1>();
        Generate<BLOCK_THREADS, ITEMS_PER_THREAD, 2>();
        Generate<BLOCK_THREADS, ITEMS_PER_THREAD, 4>();
    }


    /// Generate thread-granularity variations
    template <int BLOCK_THREADS>
    void Generate()
    {
        Generate<BLOCK_THREADS, 1>();
        Generate<BLOCK_THREADS, 2>();
        Generate<BLOCK_THREADS, 4>();
//        Generate<BLOCK_THREADS, 7>();
        Generate<BLOCK_THREADS, 8>();
//        Generate<BLOCK_THREADS, 11>();
//        Generate<BLOCK_THREADS, 12>();
//        Generate<BLOCK_THREADS, 15>();
        Generate<BLOCK_THREADS, 16>();
//        Generate<BLOCK_THREADS, 19>();
        Generate<BLOCK_THREADS, 20>();
    }


    /// Generate block size variations
    void Generate()
    {
        Generate<32>();
        Generate<64>();
        Generate<96>();
        Generate<128>();
        Generate<164>();
        Generate<192>();
        Generate<256>();
        Generate<512>();
    }


    /**
     * Generate and run tests
     */
    void Test(
        SizeT           num_items,
        ReductionOp     reduction_op)
    {

        typedef BlockReduceTilesPolicy<32, 4, 4, PTX_LOAD_NONE, GRID_MAPPING_EVEN_SHARE, 1> SimpleSinglePolicy;

        SingleDispatchTuple simple_single_tuple;
        simple_single_tuple.params.template Init<SimpleSinglePolicy>();
        simple_single_tuple.kernel_ptr = SingleReduceKernel<SimpleSinglePolicy, T*, T*, SizeT, ReductionOp>;

        // Allocate host arrays
        T *h_in = new T[num_items];
        T h_reference[1];

        // Initialize problem
        Initialize(UNIFORM, h_in, h_reference, reduction_op, num_items);

        // Initialize device arrays
        T *d_in = NULL;
        T *d_out = NULL;
        CubDebugExit(DeviceAllocate((void**)&d_in, sizeof(T) * num_items));
        CubDebugExit(DeviceAllocate((void**)&d_out, sizeof(T) * 1));
        CubDebugExit(cudaMemcpy(d_in, h_in, sizeof(T) * num_items, cudaMemcpyHostToDevice));

        for (int i = 0; i < multi_kernels.size(); ++i)
        {
            Test(multi_kernels[i], simple_single_tuple, d_in, d_out, h_reference, num_items, reduction_op);
        }

        // Cleanup
        if (h_in) delete[] h_in;
        if (d_in) CubDebugExit(DeviceFree(d_in));
        if (d_out) CubDebugExit(DeviceFree(d_out));
    }

};



//---------------------------------------------------------------------
// Main
//---------------------------------------------------------------------

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
    bool quick = args.CheckCmdLineFlag("quick");

    // Print usage
    if (args.CheckCmdLineFlag("help"))
    {
        printf("%s "
            "[--device=<device-id>] "
            "[--n=<num items>]"
            "[--i=<timing iterations>]"
            "\n", argv[0]);
        exit(0);
    }

    // Initialize device
    CubDebugExit(args.DeviceInit());

    typedef unsigned int SizeT;
    typedef unsigned int T;
    Sum<T> reduction_op;

    printf("\nGenerating kernels\n"); fflush(stdout);
    Schmoo<T, SizeT, Sum<T> > schmoo;
    schmoo.Generate();

    printf("\nRunning tests, iterations: %d, num_items: %d, sizeof(T): %d\n", g_iterations, num_items, (int) sizeof(T)); fflush(stdout);
    schmoo.Test(num_items, reduction_op);

    printf("\nSorted results:\n");
    sort(schmoo.runs.begin(), schmoo.runs.end());
    for (int i = 0; i < schmoo.runs.size(); i++)
    {
        schmoo.runs[i].Print();
    }

    return 0;
}



