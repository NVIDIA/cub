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
 * Evaluates different tuning configurations of DeviceReduce.
 *
 * The best way to use this program:
 * (1) Find the best all-around single-block tune for a given arch.
 *     For example, 1000 samples [1 ..512], 10 timing iterations per config per sample:
 *         ./bin/tune_device_reduce_sm200_nvvm_5.0_abi_i386 --i=10 --s=1000 --n=512 --single --device=0
 * (2) Update the single tune in device_reduce.cuh
 * (3) Find the best all-around multi-block tune for a given arch.
 *     For example, 1000 samples [single-block tile-size ..  50,331,648], 2 timing iterations per config per sample:
 *         ./bin/tune_device_reduce_sm200_nvvm_5.0_abi_i386 --i=2 --s=1000 --device=0
 * (4) Update the multi-block tune in device_reduce.cuh
 *
 ******************************************************************************/

// Ensure printing of CUDA runtime errors to console
#define CUB_STDERR

#include <vector>
#include <algorithm>
#include <stdio.h>
#include <cub/cub.cuh>
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
bool    g_verbose           = false;
bool    g_single            = false;
bool    g_verify            = true;


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
        float                               best_avg_throughput;
        SizeT                               best_size;
        float                               hmean_speedup;


        DispatchTuple() :
            kernel_ptr(0),
            avg_throughput(0.0),
            best_avg_throughput(0.0),
            hmean_speedup(0.0),
            best_size(0) {}
    };

    /**
     * Comparison operator for DispatchTuple.avg_throughput
     */
    template <typename Tuple>
    static bool MinSpeedup(const Tuple &a, const Tuple &b)
    {
        float delta = a.hmean_speedup - b.hmean_speedup;

        return ((delta < 0.005) && (delta > -0.005)) ?
            (a.best_avg_throughput < b.best_avg_throughput) :       // Negligible average performance differences: defer to best performance
            (a.hmean_speedup < b.hmean_speedup);
    }



    /// Multi-block reduction kernel type and dispatch tuple type
    typedef void (*MultiBlockDeviceReduceKernelPtr)(T*, T*, SizeT, GridEvenShare<SizeT>, GridQueue<SizeT>, ReductionOp);
    typedef DispatchTuple<MultiBlockDeviceReduceKernelPtr> MultiDispatchTuple;

    /// Single-block reduction kernel type and dispatch tuple type
    typedef void (*SingleBlockDeviceReduceKernelPtr)(T*, T*, SizeT, ReductionOp);
    typedef DispatchTuple<SingleBlockDeviceReduceKernelPtr> SingleDispatchTuple;


    //---------------------------------------------------------------------
    // Fields
    //---------------------------------------------------------------------

    vector<MultiDispatchTuple> multi_kernels;       // List of generated multi-block kernels
    vector<SingleDispatchTuple> single_kernels;     // List of generated single-block kernels


    //---------------------------------------------------------------------
    // Kernel enumeration methods
    //---------------------------------------------------------------------

    /**
     * Must have smem that fits in the SM
     * Must have vector load length that divides items per thread
     */
    template <typename TilesReducePolicy, typename ReductionOp>
    struct SmemSize
    {
        enum
        {
            BYTES = sizeof(typename TilesReduce<TilesReducePolicy, T*, SizeT, ReductionOp>::SmemStorage),
            IS_OK = ((BYTES < ArchProps<TUNE_ARCH>::SMEM_BYTES) &&
                     (TilesReducePolicy::ITEMS_PER_THREAD % TilesReducePolicy::VECTOR_LOAD_LENGTH == 0))
        };
    };


    /**
     * Specialization that allows kernel generation with the specified TilesReducePolicy
     */
    template <
        typename    TilesReducePolicy,
        bool        IsOk = SmemSize<TilesReducePolicy, ReductionOp>::IS_OK>
    struct Ok
    {
        /// Enumerate multi-block kernel and add to the list
        template <typename KernelsVector>
        static void GenerateMulti(
            KernelsVector &multi_kernels,
            int subscription_factor)
        {
            MultiDispatchTuple tuple;
            tuple.params.template Init<TilesReducePolicy>(subscription_factor);
            tuple.kernel_ptr = MultiBlockReduceKernel<TilesReducePolicy, T*, T*, SizeT, ReductionOp>;
            multi_kernels.push_back(tuple);
        }


        /// Enumerate single-block kernel and add to the list
        template <typename KernelsVector>
        static void GenerateSingle(KernelsVector &single_kernels)
        {
            SingleDispatchTuple tuple;
            tuple.params.template Init<TilesReducePolicy>();
            tuple.kernel_ptr = SingleBlockReduceKernel<TilesReducePolicy, T*, T*, SizeT, ReductionOp>;
            single_kernels.push_back(tuple);
        }
    };

    /**
     * Specialization that rejects kernel generation with the specified TilesReducePolicy
     */
    template <typename TilesReducePolicy>
    struct Ok<TilesReducePolicy, false>
    {
        template <typename KernelsVector>
        static void GenerateMulti(KernelsVector &multi_kernels, int subscription_factor) {}

        template <typename KernelsVector>
        static void GenerateSingle(KernelsVector &single_kernels) {}
    };


    /// Enumerate block-scheduling variations
    template <
        int                     BLOCK_THREADS,
        int                     ITEMS_PER_THREAD,
        int                     VECTOR_LOAD_LENGTH,
        BlockReduceAlgorithm    BLOCK_ALGORITHM,
        PtxLoadModifier         LOAD_MODIFIER>
    void Enumerate()
    {
        // Multi-block kernels
        Ok<TilesReducePolicy<BLOCK_THREADS, ITEMS_PER_THREAD, VECTOR_LOAD_LENGTH, BLOCK_ALGORITHM, LOAD_MODIFIER, GRID_MAPPING_EVEN_SHARE> >::GenerateMulti(multi_kernels, 1);
        Ok<TilesReducePolicy<BLOCK_THREADS, ITEMS_PER_THREAD, VECTOR_LOAD_LENGTH, BLOCK_ALGORITHM, LOAD_MODIFIER, GRID_MAPPING_EVEN_SHARE> >::GenerateMulti(multi_kernels, 2);
        Ok<TilesReducePolicy<BLOCK_THREADS, ITEMS_PER_THREAD, VECTOR_LOAD_LENGTH, BLOCK_ALGORITHM, LOAD_MODIFIER, GRID_MAPPING_EVEN_SHARE> >::GenerateMulti(multi_kernels, 4);
        Ok<TilesReducePolicy<BLOCK_THREADS, ITEMS_PER_THREAD, VECTOR_LOAD_LENGTH, BLOCK_ALGORITHM, LOAD_MODIFIER, GRID_MAPPING_EVEN_SHARE> >::GenerateMulti(multi_kernels, 8);
#if TUNE_ARCH >= 200
        Ok<TilesReducePolicy<BLOCK_THREADS, ITEMS_PER_THREAD, VECTOR_LOAD_LENGTH, BLOCK_ALGORITHM, LOAD_MODIFIER, GRID_MAPPING_DYNAMIC> >::GenerateMulti(multi_kernels, 1);
#endif

        // Single-block kernels
        Ok<TilesReducePolicy<BLOCK_THREADS, ITEMS_PER_THREAD, VECTOR_LOAD_LENGTH, BLOCK_ALGORITHM, LOAD_MODIFIER, GRID_MAPPING_EVEN_SHARE> >::GenerateSingle(single_kernels);
    }


    /// Enumerate load modifier variations
    template <
        int                     BLOCK_THREADS,
        int                     ITEMS_PER_THREAD,
        int                     VECTOR_LOAD_LENGTH,
        BlockReduceAlgorithm    BLOCK_ALGORITHM>
    void Enumerate()
    {
        Enumerate<BLOCK_THREADS, ITEMS_PER_THREAD, VECTOR_LOAD_LENGTH, BLOCK_ALGORITHM, LOAD_DEFAULT>();
#if TUNE_ARCH >= 350
        Enumerate<BLOCK_THREADS, ITEMS_PER_THREAD, VECTOR_LOAD_LENGTH, BLOCK_ALGORITHM, LOAD_LDG>();
#endif
    }


    /// Enumerate block algorithms
    template <
        int BLOCK_THREADS,
        int ITEMS_PER_THREAD,
        int VECTOR_LOAD_LENGTH>
    void Enumerate()
    {
        Enumerate<BLOCK_THREADS, ITEMS_PER_THREAD, VECTOR_LOAD_LENGTH, BLOCK_REDUCE_RAKING>();
        Enumerate<BLOCK_THREADS, ITEMS_PER_THREAD, VECTOR_LOAD_LENGTH, BLOCK_REDUCE_WARP_REDUCTIONS>();
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
        Enumerate<BLOCK_THREADS, 9>();
        Enumerate<BLOCK_THREADS, 11>();
        Enumerate<BLOCK_THREADS, 12>();
//      Enumerate<BLOCK_THREADS, 13>();
//      Enumerate<BLOCK_THREADS, 15>();
        Enumerate<BLOCK_THREADS, 16>();
        Enumerate<BLOCK_THREADS, 17>();
//      Enumerate<BLOCK_THREADS, 19>();
        Enumerate<BLOCK_THREADS, 20>();
//      Enumerate<BLOCK_THREADS, 21>();
        Enumerate<BLOCK_THREADS, 23>();
        Enumerate<BLOCK_THREADS, 24>();
        Enumerate<BLOCK_THREADS, 25>();
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
        if (g_verify) CubDebugExit(cudaMemset(d_out, 0, sizeof(T)));

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

        if (g_verify) CubDebugExit(cudaDeviceSynchronize());

        // Copy out and display results
        int compare = (g_verify) ?
            CompareDeviceResults(h_reference, d_out, 1, true, false) :
            0;

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
        if (avg_throughput > multi_dispatch.best_avg_throughput)
        {
            multi_dispatch.best_avg_throughput = avg_throughput;
            multi_dispatch.best_size = num_items;
        }

        single_dispatch.avg_throughput = CUB_MAX(avg_throughput, single_dispatch.avg_throughput);
        if (avg_throughput > single_dispatch.best_avg_throughput)
        {
            single_dispatch.best_avg_throughput = avg_throughput;
            single_dispatch.best_size = num_items;
        }

        if (g_verbose)
        {
            printf("\t%.2f GB/s, multi_dispatch( ", avg_bandwidth);
            multi_dispatch.params.Print();
            printf(" ), single_dispatch( ");
            single_dispatch.params.Print();
            printf(" )\n");
            fflush(stdout);
        }

        AssertEquals(0, compare);
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
        typedef typename DeviceReduce::TunedPolicies<T, SizeT, TUNE_ARCH>::SingleBlockPolicy SimpleSingleBlockPolicy;
        SingleDispatchTuple simple_single_tuple;
        simple_single_tuple.params.template Init<SimpleSingleBlockPolicy>();
        simple_single_tuple.kernel_ptr = SingleBlockReduceKernel<SimpleSingleBlockPolicy, T*, T*, SizeT, ReductionOp>;

        double max_exponent      = log2(double(g_max_items));
        double min_exponent      = log2(double(simple_single_tuple.params.tile_size));
        unsigned int max_int     = (unsigned int) -1;

        for (int sample = 0; sample < g_samples; ++sample)
        {
            printf("\nMulti-block sample %d, ", sample);

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
                    double exponent = ((max_exponent - min_exponent) * scale) + min_exponent;
                    num_items = pow(2.0, exponent);
                    printf("num_items: %d (2^%.2f)", num_items, exponent); fflush(stdout);
                }
                else
                {
                    // uniform bias
                    num_items = CUB_MAX(pow(2.0, min_exponent), scale * g_max_items);
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
            float best_avg_throughput = 0.0;
            for (int j = 0; j < multi_kernels.size(); ++j)
            {
                multi_kernels[j].avg_throughput = 0.0;

                Test(multi_kernels[j], simple_single_tuple, d_in, d_out, &h_reference, num_items, reduction_op);

                best_avg_throughput = CUB_MAX(best_avg_throughput, multi_kernels[j].avg_throughput);
            }

            // Print best throughput for this problem size
            printf("Best: %.2fe9 items/s (%.2f GB/s)\n", best_avg_throughput, best_avg_throughput * sizeof(T));

            // Accumulate speedup (inverse for harmonic mean)
            for (int j = 0; j < multi_kernels.size(); ++j)
                multi_kernels[j].hmean_speedup += best_avg_throughput / multi_kernels[j].avg_throughput;
        }

        // Find max overall throughput and compute hmean speedups
        float overall_max_throughput = 0.0;
        for (int j = 0; j < multi_kernels.size(); ++j)
        {
            overall_max_throughput = CUB_MAX(overall_max_throughput, multi_kernels[j].best_avg_throughput);
            multi_kernels[j].hmean_speedup = float(g_samples) / multi_kernels[j].hmean_speedup;
        }

        // Sort by cumulative speedup
        sort(multi_kernels.begin(), multi_kernels.end(), MinSpeedup<MultiDispatchTuple>);

        // Print ranked multi configurations
        printf("\nRanked multi_kernels:\n");
        for (int j = 0; j < multi_kernels.size(); ++j)
        {
            printf("\t (%d) params( ", multi_kernels.size() - j);
            multi_kernels[j].params.Print();
            printf(" ) hmean speedup: %.3f, best throughput %.2f @ %d elements (%.2f GB/s, %.2f%%)\n",
                multi_kernels[j].hmean_speedup,
                multi_kernels[j].best_avg_throughput,
                (int) multi_kernels[j].best_size,
                multi_kernels[j].best_avg_throughput * sizeof(T),
                multi_kernels[j].best_avg_throughput / overall_max_throughput);
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
        if (g_verify) CubDebugExit(cudaMemset(d_out, 0, sizeof(T)));

        // Warmup/correctness iteration
        DeviceReduce::DispatchSingle(
            single_dispatch.kernel_ptr,
            single_dispatch.params,
            d_in,
            d_out,
            num_items,
            reduction_op);

        if (g_verify) CubDebugExit(cudaDeviceSynchronize());

        // Copy out and display results
        int compare = (g_verify) ?
            CompareDeviceResults(h_reference, d_out, 1, true, false) :
            0;

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
        if (avg_throughput > single_dispatch.best_avg_throughput)
        {
            single_dispatch.best_avg_throughput = avg_throughput;
            single_dispatch.best_size = num_items;
        }

        if (g_verbose)
        {
            printf("\t%.2f GB/s, single_dispatch( ", avg_bandwidth);
            single_dispatch.params.Print();
            printf(" )\n");
            fflush(stdout);
        }

        AssertEquals(0, compare);
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
            printf("\nSingle-block sample %d, ", sample);

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
            float best_avg_throughput = 0.0;
            for (int j = 0; j < single_kernels.size(); ++j)
            {
                single_kernels[j].avg_throughput = 0.0;

                Test(single_kernels[j], d_in, d_out, &h_reference, num_items, reduction_op);

                best_avg_throughput = CUB_MAX(best_avg_throughput, single_kernels[j].avg_throughput);
            }

            // Print best throughput for this problem size
            printf("Best: %.2fe9 items/s (%.2f GB/s)\n", best_avg_throughput, best_avg_throughput * sizeof(T));

            // Accumulate speedup (inverse for harmonic mean)
            for (int j = 0; j < single_kernels.size(); ++j)
                single_kernels[j].hmean_speedup += best_avg_throughput / single_kernels[j].avg_throughput;
        }

        // Find max overall throughput and compute hmean speedups
        float overall_max_throughput = 0.0;
        for (int j = 0; j < single_kernels.size(); ++j)
        {
            overall_max_throughput = CUB_MAX(overall_max_throughput, single_kernels[j].best_avg_throughput);
            single_kernels[j].hmean_speedup = float(g_samples) / single_kernels[j].hmean_speedup;
        }

        // Sort by cumulative speedup
        sort(single_kernels.begin(), single_kernels.end(), MinSpeedup<SingleDispatchTuple>);

        // Print ranked single configurations
        printf("\nRanked single_kernels:\n");
        for (int j = 0; j < single_kernels.size(); ++j)
        {
            printf("\t (%d) params( ", single_kernels.size() - j);
            single_kernels[j].params.Print();
            printf(" ) hmean speedup: %.3f, best throughput %.2f @ %d elements (%.2f GB/s, %.2f%%)\n",
                single_kernels[j].hmean_speedup,
                single_kernels[j].best_avg_throughput,
                (int) single_kernels[j].best_size,
                single_kernels[j].best_avg_throughput * sizeof(T),
                single_kernels[j].best_avg_throughput / overall_max_throughput);
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
    g_verify = !args.CheckCmdLineFlag("noverify");

    // Print usage
    if (args.CheckCmdLineFlag("help"))
    {
        printf("%s "
            "[--device=<device-id>] "
            "[--n=<max items>]"
            "[--s=<samples>]"
            "[--i=<timing iterations>]"
            "[--single]"
            "[--v]"
            "[--noverify]"
            "\n", argv[0]);
        exit(0);
    }

    // Initialize device
    CubDebugExit(args.DeviceInit());

#if (TUNE_SIZE == 1)
    typedef unsigned char T;
#elif (TUNE_SIZE == 2)
    typedef unsigned short T;
#elif (TUNE_SIZE == 4)
    typedef unsigned int T;
#elif (TUNE_SIZE == 8)
    typedef unsigned long long T;
#else
    // Default
    typedef unsigned int T;
#endif

    typedef unsigned int SizeT;
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



