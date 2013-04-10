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

int     g_max_items         = 48 * 1024 * 1024;
int     g_samples           = 100;
int     g_iterations        = 2;
int     g_min_exponent      = 14;       // min sampled problem size is 2^14 (16384)
bool    g_verbose           = false;
bool    g_single            = false;


//---------------------------------------------------------------------
// Host utility subroutines
//---------------------------------------------------------------------

/**
 * Initialize problem
 */
template <typename T>
void Initialize(
    int             gen_mode,
    T               *h_in,
    int             num_items)
{
    for (int i = 0; i < num_items; ++i)
    {
        InitValue(gen_mode, h_in[i], i);
    }
}

/**
 * Sequential reduction
 */
template <typename T, typename ReductionOp>
T Reduce(
    T               *h_in,
    ReductionOp     reduction_op,
    int             num_items)
{
    T retval = h_in[0];
    for (int i = 1; i < num_items; ++i)
        retval = reduction_op(retval, h_in[i]);
    return retval;
}



//---------------------------------------------------------------------
// Full tile test generation
//---------------------------------------------------------------------



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

    /// Pairing of kernel function pointer and corresponding dispatch params
    template <typename KernelPtr>
    struct DispatchTuple
    {
        KernelPtr                           kernel_ptr;
        DeviceReduce::KernelDispachParams   params;

        float                               avg_throughput;
        float                               max_throughput;
        int                                 cumulative_rank;

        DispatchTuple() : kernel_ptr(0), avg_throughput(0.0), max_throughput(0.0), cumulative_rank(0) {}
    };

    /**
     * Comparison operator for DispatchTuple.avg_throughput
     */
    template <typename Tuple>
    static bool MinThroughput(const Tuple &a, const Tuple &b)
    {
        return (a.avg_throughput < b.avg_throughput);
    }

    /**
     * Comparison operator for DispatchTuple.cumulative_rank
     */
    template <typename Tuple>
    static bool MinRank(const Tuple &a, const Tuple &b)
    {
        return (a.cumulative_rank < b.cumulative_rank);
    }


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


    //---------------------------------------------------------------------
    // Kernel enumeration methods
    //---------------------------------------------------------------------

    /**
     * Specialization that allows kernel generation with the specified BlockReduceTilesPolicy
     */
    template <
        typename BlockReduceTilesPolicy,
        bool IsOk = (sizeof(typename BlockReduceTiles<BlockReduceTilesPolicy, T*, SizeT>::SmemStorage) < ArchProps<TUNE_ARCH>::SMEM_BYTES)>
    struct Ok
    {
        /// Enumerate multi-block kernel and add to the list
        template <typename KernelsVector>
        static void GenerateMulti(
            KernelsVector &multi_kernels,
            int subscription_factor)
        {
            MultiDispatchTuple tuple;
            tuple.params.template Init<BlockReduceTilesPolicy>(subscription_factor);
            tuple.kernel_ptr = MultiReduceKernel<BlockReduceTilesPolicy, T*, T*, SizeT, ReductionOp>;
            multi_kernels.push_back(tuple);
        }


        /// Enumerate single-block kernel and add to the list
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


    /// Enumerate block-scheduling variations
    template <
        int             BLOCK_THREADS,
        int             ITEMS_PER_THREAD,
        int             VECTOR_LOAD_LENGTH,
        PtxLoadModifier LOAD_MODIFIER>
    void Enumerate()
    {
        // Multi-block kernels
        Ok<BlockReduceTilesPolicy<BLOCK_THREADS, ITEMS_PER_THREAD, VECTOR_LOAD_LENGTH, LOAD_MODIFIER, GRID_MAPPING_EVEN_SHARE> >::GenerateMulti(multi_kernels, 1);
        Ok<BlockReduceTilesPolicy<BLOCK_THREADS, ITEMS_PER_THREAD, VECTOR_LOAD_LENGTH, LOAD_MODIFIER, GRID_MAPPING_EVEN_SHARE> >::GenerateMulti(multi_kernels, 2);
        Ok<BlockReduceTilesPolicy<BLOCK_THREADS, ITEMS_PER_THREAD, VECTOR_LOAD_LENGTH, LOAD_MODIFIER, GRID_MAPPING_EVEN_SHARE> >::GenerateMulti(multi_kernels, 4);
        Ok<BlockReduceTilesPolicy<BLOCK_THREADS, ITEMS_PER_THREAD, VECTOR_LOAD_LENGTH, LOAD_MODIFIER, GRID_MAPPING_EVEN_SHARE> >::GenerateMulti(multi_kernels, 8);
#if TUNE_ARCH >= 200
        Ok<BlockReduceTilesPolicy<BLOCK_THREADS, ITEMS_PER_THREAD, VECTOR_LOAD_LENGTH, LOAD_MODIFIER, GRID_MAPPING_DYNAMIC> >::GenerateMulti(multi_kernels, 1);
#endif

        // Single-block kernels
        Ok<BlockReduceTilesPolicy<BLOCK_THREADS, ITEMS_PER_THREAD, VECTOR_LOAD_LENGTH, LOAD_MODIFIER, GRID_MAPPING_EVEN_SHARE> >::GenerateSingle(single_kernels);
    }


    /// Enumerate load modifier variations
    template <
        int BLOCK_THREADS,
        int ITEMS_PER_THREAD,
        int VECTOR_LOAD_LENGTH>
    void Enumerate()
    {
        Enumerate<BLOCK_THREADS, ITEMS_PER_THREAD, VECTOR_LOAD_LENGTH, PTX_LOAD_NONE>();
#if TUNE_ARCH >= 350
        Enumerate<BLOCK_THREADS, ITEMS_PER_THREAD, VECTOR_LOAD_LENGTH, PTX_LOAD_LDG>();
#endif
    }


    /// Enumerate vectorization variations
    template <
        int BLOCK_THREADS,
        int ITEMS_PER_THREAD>
    void Enumerate()
    {
        Enumerate<BLOCK_THREADS, ITEMS_PER_THREAD, 1>();
        Enumerate<BLOCK_THREADS, ITEMS_PER_THREAD, 2>();
        Enumerate<BLOCK_THREADS, ITEMS_PER_THREAD, 4>();
    }


    /// Enumerate thread-granularity variations
    template <int BLOCK_THREADS>
    void Enumerate()
    {
        Enumerate<BLOCK_THREADS, 1>();
        Enumerate<BLOCK_THREADS, 2>();
        Enumerate<BLOCK_THREADS, 4>();
//      Enumerate<BLOCK_THREADS, 7>();
        Enumerate<BLOCK_THREADS, 8>();
//      Enumerate<BLOCK_THREADS, 11>();
//      Enumerate<BLOCK_THREADS, 12>();
//      Enumerate<BLOCK_THREADS, 15>();
        Enumerate<BLOCK_THREADS, 16>();
//      Enumerate<BLOCK_THREADS, 19>();
        Enumerate<BLOCK_THREADS, 20>();
    }


    /// Enumerate block size variations
    void Enumerate()
    {
        printf("\nEnumerating kernels\n"); fflush(stdout);

        Enumerate<32>();
        Enumerate<64>();
        Enumerate<96>();
        Enumerate<128>();
        Enumerate<160>();
        Enumerate<192>();
        Enumerate<256>();
        Enumerate<512>();
    }


    //---------------------------------------------------------------------
    // Multi-block test methods
    //---------------------------------------------------------------------

    /**
     * Test multi reduction
     */
    void Test(
        MultiDispatchTuple      &multi_dispatch,
        SingleDispatchTuple     &single_dispatch,
        T*                      d_in,
        T*                      d_out,
        T*                      h_reference,
        SizeT                   num_items,
        ReductionOp             reduction_op)
    {
        // Clear output
        CubDebugExit(cudaMemset(d_out, 0, sizeof(T)));

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
        int correct = CompareDeviceResults(h_reference, d_out, 1, true, false);

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

        float avg_elapsed = elapsed_millis / g_iterations;
        float avg_throughput = float(num_items) / avg_elapsed / 1000.0 / 1000.0;
        float avg_bandwidth = avg_throughput * sizeof(T);

        multi_dispatch.avg_throughput = CUB_MAX(avg_throughput, multi_dispatch.avg_throughput);
        multi_dispatch.max_throughput = CUB_MAX(avg_throughput, multi_dispatch.max_throughput);

        single_dispatch.avg_throughput = CUB_MAX(avg_throughput, single_dispatch.avg_throughput);
        single_dispatch.max_throughput = CUB_MAX(avg_throughput, single_dispatch.max_throughput);

        if (g_verbose)
        {
            printf("\t%.2f GB/s, multi_dispatch( ", avg_bandwidth);
            multi_dispatch.params.Print();
            printf(" ), single_dispatch( ");
            single_dispatch.params.Print();
            printf(" )\n");
            fflush(stdout);
        }

        AssertEquals(0, correct);
    }


    /**
     * Evaluate multi-block configurations
     */
    void TestMulti(
        T*                      h_in,
        T*                      d_in,
        T*                      d_out,
        ReductionOp             reduction_op)
    {
        // Simple single kernel tuple for use with multi kernel sweep
        typedef typename DeviceReduce::TunedPolicies<T, SizeT, TUNE_ARCH>::SinglePolicy SimpleSinglePolicy;
        SingleDispatchTuple simple_single_tuple;
        simple_single_tuple.params.template Init<SimpleSinglePolicy>();
        simple_single_tuple.kernel_ptr = SingleReduceKernel<SimpleSinglePolicy, T*, T*, SizeT, ReductionOp>;

        double max_exponent      = log2(double(g_max_items));
        unsigned int max_int     = (unsigned int) -1;

        for (int sample = 0; sample < g_samples; ++sample)
        {
            printf("\nMulti-block sample %d,", sample);

            int num_items;
            if (sample == 0)
            {
                // First sample: use max items
                num_items = g_max_items;
                printf("num_items: %d", num_items); fflush(stdout);
            }
            else
            {
                // Sample a problem size from [2^g_min_exponent, g_max_items].  First 2/3 of the samples are log-distributed, the other 1/3 are uniformly-distributed.
                unsigned int bits;
                RandomBits(bits);
                double scale = double(bits) / max_int;

                if (sample < (2 * g_samples) / 3)
                {
                    // log bias
                    double exponent = ((max_exponent - g_min_exponent) * scale) + g_min_exponent;
                    num_items = pow(2.0, exponent);
                    printf("num_items: %d (2^%.2f)", num_items, exponent); fflush(stdout);
                }
                else
                {
                    // uniform bias
                    num_items = CUB_MAX(pow(2.0, g_min_exponent), scale * g_max_items);
                    printf("num_items: %d (%.2f * %d)", num_items, scale, g_max_items); fflush(stdout);
                }
            }
            if (g_verbose)
                printf("\n");
            else
                printf(", ");

            // Compute reference
            T h_reference = Reduce(h_in, reduction_op, num_items);

            // Run test on each multi-kernel configuration
            for (int j = 0; j < multi_kernels.size(); ++j)
            {
                multi_kernels[j].avg_throughput = 0.0;
                Test(multi_kernels[j], simple_single_tuple, d_in, d_out, &h_reference, num_items, reduction_op);
            }

            // Sort by throughput
            sort(multi_kernels.begin(), multi_kernels.end(), MinThroughput<MultiDispatchTuple>);

            // Print best throughput for this problem size
            float best_throughput = (multi_kernels.size() > 0) ? multi_kernels.back().avg_throughput : 0;
            printf("Best: %.2fe9 items/s (%.2f GB/s)\n", best_throughput, best_throughput * sizeof(T));

            // Update cumulative rank
            for (int j = 0; j < multi_kernels.size(); ++j)
                multi_kernels[j].cumulative_rank += j;
        }

        // Sort by cumulative rank
        sort(multi_kernels.begin(), multi_kernels.end(), MinRank<MultiDispatchTuple>);

        // Find max overall throughput
        float overall_max_throughput = 0.0;
        for (int j = 0; j < multi_kernels.size(); ++j)
            overall_max_throughput = CUB_MAX(overall_max_throughput, multi_kernels[j].max_throughput);

        // Print ranked multi configurations
        printf("\nRanked multi_kernels:\n");
        for (int j = 0; j < multi_kernels.size(); ++j)
        {
            printf("\t (%d) params( ", multi_kernels.size() - j);
            multi_kernels[j].params.Print();
            printf(" ) avg rank: %.2f, max throughput %.2f (%.2f GB/s, %.2f%%)\n",
                float(multi_kernels[j].cumulative_rank) / (g_samples * multi_kernels.size()),
                multi_kernels[j].max_throughput,
                multi_kernels[j].max_throughput * sizeof(T),
                multi_kernels[j].max_throughput / overall_max_throughput);
        }

        printf("\nMax multi-block throughput %.2f (%.2f GB/s)\n", overall_max_throughput, overall_max_throughput * sizeof(T));
    }


    //---------------------------------------------------------------------
    // Single-block test methods
    //---------------------------------------------------------------------

    /**
     * Test single reduction
     */
    void Test(
        SingleDispatchTuple     &single_dispatch,
        T*                      d_in,
        T*                      d_out,
        T*                      h_reference,
        SizeT                   num_items,
        ReductionOp             reduction_op)
    {
        // Clear output
        CubDebugExit(cudaMemset(d_out, 0, sizeof(T)));

        // Warmup/correctness iteration
        DeviceReduce::DispatchSingle(
            single_dispatch.kernel_ptr,
            single_dispatch.params,
            d_in,
            d_out,
            num_items,
            reduction_op);

        CubDebugExit(cudaDeviceSynchronize());

        // Copy out and display results
        int correct = CompareDeviceResults(h_reference, d_out, 1, true, false);

        // Performance
        GpuTimer gpu_timer;
        float elapsed_millis = 0.0;
        for (int i = 0; i < g_iterations; i++)
        {
            gpu_timer.Start();

            DeviceReduce::DispatchSingle(
                single_dispatch.kernel_ptr,
                single_dispatch.params,
                d_in,
                d_out,
                num_items,
                reduction_op);

            gpu_timer.Stop();
            elapsed_millis += gpu_timer.ElapsedMillis();
        }

        float avg_elapsed = elapsed_millis / g_iterations;
        float avg_throughput = float(num_items) / avg_elapsed / 1000.0 / 1000.0;
        float avg_bandwidth = avg_throughput * sizeof(T);

        single_dispatch.avg_throughput = CUB_MAX(avg_throughput, single_dispatch.avg_throughput);
        single_dispatch.max_throughput = CUB_MAX(avg_throughput, single_dispatch.max_throughput);

        if (g_verbose)
        {
            printf("\t%.2f GB/s, single_dispatch( ", avg_bandwidth);
            single_dispatch.params.Print();
            printf(" )\n");
            fflush(stdout);
        }

        AssertEquals(0, correct);
    }


    /**
     * Evaluate single-block configurations
     */
    void TestSingle(
        T*                      h_in,
        T*                      d_in,
        T*                      d_out,
        ReductionOp             reduction_op)
     {
        double max_exponent     = log2(double(g_max_items));
        unsigned int max_int    = (unsigned int) -1;

        for (int sample = 0; sample < g_samples; ++sample)
        {
            printf("\nSingle-block sample %d,", sample);

            int num_items;
            if (sample == 0)
            {
                // First sample: use max items
                num_items = g_max_items;
                printf("num_items: %d", num_items); fflush(stdout);
            }
            else
            {
                // Sample a problem size from [2, g_max_items], log-distributed
                unsigned int bits;
                RandomBits(bits);
                double scale = double(bits) / max_int;
                double exponent = ((max_exponent - 1) * scale) + 1;
                num_items = pow(2.0, exponent);
                printf("num_items: %d (2^%.2f)", num_items, exponent); fflush(stdout);
            }

            if (g_verbose)
                printf("\n");
            else
                printf(", ");

            // Compute reference
            T h_reference = Reduce(h_in, reduction_op, num_items);

            // Run test on each single-kernel configuration
            for (int j = 0; j < single_kernels.size(); ++j)
            {
                single_kernels[j].avg_throughput = 0.0;
                Test(single_kernels[j], d_in, d_out, &h_reference, num_items, reduction_op);
            }

            // Sort by throughput
            sort(single_kernels.begin(), single_kernels.end(), MinThroughput<SingleDispatchTuple>);

            // Print best throughput for this problem size
            float best_throughput = (single_kernels.size() > 0) ? single_kernels.back().avg_throughput : 0;
            printf("Best: %.2fe9 items/s (%.2f GB/s)\n", best_throughput, best_throughput * sizeof(T));

            // Update cumulative rank
            for (int j = 0; j < single_kernels.size(); ++j)
                single_kernels[j].cumulative_rank += j;
        }

        // Sort by cumulative rank
        sort(single_kernels.begin(), single_kernels.end(), MinRank<SingleDispatchTuple>);

        // Find max overall throughput
        float overall_max_throughput = 0.0;
        for (int j = 0; j < single_kernels.size(); ++j)
            overall_max_throughput = CUB_MAX(overall_max_throughput, single_kernels[j].max_throughput);

        // Print ranked single configurations
        printf("\nRanked single_kernels:\n");
        for (int j = 0; j < single_kernels.size(); ++j)
        {
            printf("\t (%d) params( ", single_kernels.size() - j);
            single_kernels[j].params.Print();
            printf(" ) avg rank: %.2f, max throughput %.2f (%.2f GB/s, %.2f%%)\n",
                float(single_kernels[j].cumulative_rank) / (g_samples * single_kernels.size()),
                single_kernels[j].max_throughput,
                single_kernels[j].max_throughput * sizeof(T),
                single_kernels[j].max_throughput / overall_max_throughput);
        }

        printf("\nMax single-block throughput %.2f (%.2f GB/s)\n", overall_max_throughput, overall_max_throughput * sizeof(T));
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
    // Initialize command line
    CommandLineArgs args(argc, argv);
    args.GetCmdLineArgument("n", g_max_items);
    args.GetCmdLineArgument("s", g_samples);
    args.GetCmdLineArgument("i", g_iterations);
    g_verbose = args.CheckCmdLineFlag("v");
    g_single = args.CheckCmdLineFlag("single");

    // Print usage
    if (args.CheckCmdLineFlag("help"))
    {
        printf("%s "
            "[--device=<device-id>] "
            "[--n=<max items>]"
            "[--s=<samples>]"
            "[--i=<timing iterations>]"
            "[--v]"
            "[--single]"
            "\n", argv[0]);
        exit(0);
    }

    // Initialize device
    CubDebugExit(args.DeviceInit());

    typedef unsigned int SizeT;
    typedef unsigned int T;
    Sum<T> reduction_op;

    // Enumerate kernels
    Schmoo<T, SizeT, Sum<T> > schmoo;
    schmoo.Enumerate();

    // Allocate host arrays
    T *h_in = new T[g_max_items];

    // Initialize problem
    Initialize(UNIFORM, h_in, g_max_items);

    // Initialize device arrays
    T *d_in = NULL;
    T *d_out = NULL;
    CubDebugExit(DeviceAllocate((void**)&d_in, sizeof(T) * g_max_items));
    CubDebugExit(DeviceAllocate((void**)&d_out, sizeof(T) * 1));
    CubDebugExit(cudaMemcpy(d_in, h_in, sizeof(T) * g_max_items, cudaMemcpyHostToDevice));

    // Test kernels
    if (g_single)
        schmoo.TestSingle(h_in, d_in, d_out, reduction_op);
    else
        schmoo.TestMulti(h_in, d_in, d_out, reduction_op);

    // Cleanup
    if (h_in) delete[] h_in;
    if (d_in) CubDebugExit(DeviceFree(d_in));
    if (d_out) CubDebugExit(DeviceFree(d_out));

    return 0;
}



