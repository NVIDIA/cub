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
#include <stdio.h>
#include <cub.cuh>
#include <test_util.h>

using namespace cub;
using namespace std;


//---------------------------------------------------------------------
// Globals, constants and typedefs
//---------------------------------------------------------------------

bool    g_verbose = false;
int     g_iterations = 100;


/**
 * Tuning policy for BlockReduceTiles
 */
template <
    typename T,
    typename SizeT,
    typename ReductionOp>
struct KernelDispatch
{
    int                     block_threads;
    int                     items_per_thread;
    int                     vector_load_length;
    int                     oversubscription;
    GridMappingStrategy     grid_mapping;
    PtxLoadModifier         load_modifier;

    void (reduce_kernel_ptr)(
        T                       *d_in,
        T                       *d_out,
        SizeT                   num_items,
        GridEvenShare<SizeT>    even_share,
        GridQueue<SizeT>        queue,
        ReductionOp             reduction_op);

    void (single_kernel_ptr)(
        T                       *d_in,
        T                       *d_out,
        SizeT                   num_items,
        ReductionOp             reduction_op);

    template <typename BlockReduceTilesPolicy>
    KernelDispatch(BlockReduceTilesPolicy policy) :
        block_threads(BlockReduceTilesPolicy::BLOCK_THREADS),
        items_per_thread(BlockReduceTilesPolicy::ITEMS_PER_THREAD),
        vector_load_length(BlockReduceTilesPolicy::VECTOR_LOAD_LENGTH),
        oversubscription(BlockReduceTilesPolicy::OVERSUBSCRIPTION),
        grid_mapping(BlockReduceTilesPolicy::GRID_MAPPING),
        load_modifier(BlockReduceTilesPolicy::LOAD_MODIFIER)
    {
        reduce_kernel_ptr = ReduceKernel<BlockReduceTilesPolicy, T*, T*, SizeT, ReductionOp>;
        single_kernel_ptr = SingleReduceKernel<BlockReduceTilesPolicy, T*, T*, SizeT, ReductionOp>;
    }
};








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
 * Test full-tile reduction
 */
template <
    typename    T,
    typename    ReductionOp>
void Test(
    int             num_items,
    int             gen_mode,
    int             tiles,
    ReductionOp     reduction_op)
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

    printf("DeviceReduce gen-mode %d, num_items(%d), %d-byte elements, elements:\n",
        gen_mode,
        num_items,
        (int) sizeof(T));
    fflush(stdout);

    // Run warmup/correctness iteration
    DeviceReduce::Reduce(d_in, d_out, num_items, reduction_op, 0, true);
    CubDebugExit(cudaDeviceSynchronize());

    // Copy out and display results
    printf("\nReduction results: ");
    int compare = CompareDeviceResults(h_reference, d_out, 1, g_verbose, g_verbose);
    printf("\n");

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
 * Generate kernels for different CTA sizes
 */
template <
    typename T,
    typename SizeT,
    typename ReductionOp>
void GenerateKernels(vector<KernelDispatch<T, SizeT, ReductionOp> > &kernels)
{

}


/**
 * Main
 */
int main(int argc, char** argv)
{
    int     num_items = 1 * 1024 * 1024;
    int     gen_mode = UNIFORM,

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

    typedef unsigned int SizeT;

#if 1
    typedef unsigned int T;
#endif

    vector<KernelDispatch<T, SizeT, Sum<T> > > kernels;

    GenerateKernels(kernels);

    Sum<T> binary_op;


    return 0;
}



