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
 * Test of DeviceRadixSort utilities
 ******************************************************************************/

// Ensure printing of CUDA runtime errors to console
#define CUB_STDERR

#include <stdio.h>
#include <algorithm>

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
// Dispatch to different DeviceRadixSort entrypoints
//---------------------------------------------------------------------

/**
 * Dispatch to key-value pair sorting entrypoint
 */
template <typename Key, typename Value>
__host__ __device__ __forceinline__
cudaError_t Dispatch(
    Int2Type<false>     use_cnp,
    int                 timing_iterations,
    size_t              *d_temp_storage_bytes,
    int                 *d_selectors,
    cudaError_t         *d_cnp_error,

    void                *d_temp_storage,
    size_t              &temp_storage_bytes,
    DoubleBuffer<Key>   &d_keys,
    DoubleBuffer<Value> &d_values,
    int                 num_items,
    int                 begin_bit,
    int                 end_bit,
    cudaStream_t        stream,
    bool                stream_synchronous)
{
    cudaError_t error = cudaSuccess;
    for (int i = 0; i < timing_iterations; ++i)
    {
        error = DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_keys, d_values, num_items, begin_bit, end_bit, stream, stream_synchronous);
    }
    return error;
}


//---------------------------------------------------------------------
// CUDA Nested Parallelism Test Kernel
//---------------------------------------------------------------------

/**
 * Simple wrapper kernel to invoke DeviceRadixSort
 */
template <typename Key, typename Value>
__global__ void CnpDispatchKernel(
    int                 timing_iterations,
    size_t              *d_temp_storage_bytes,
    int                 *d_selectors,
    cudaError_t         *d_cnp_error,

    void                *d_temp_storage,
    size_t              temp_storage_bytes,
    DoubleBuffer<Key>   d_keys,
    DoubleBuffer<Value> d_values,
    int                 num_items,
    int                 begin_bit,
    int                 end_bit,
    bool                stream_synchronous)
{
#ifndef CUB_CNP
    *d_cnp_error = cudaErrorNotSupported;
#else
    *d_cnp_error = Dispatch(Int2Type<false>(), timing_iterations, d_temp_storage_bytes, d_selectors, d_cnp_error, d_temp_storage, temp_storage_bytes, d_keys, d_values, num_items, begin_bit, end_bit, 0, stream_synchronous);
    *d_temp_storage_bytes = temp_storage_bytes;
    d_selectors[0] = d_keys.selector;
    d_selectors[1] = d_values.selector;
#endif
}


/**
 * Dispatch to CNP kernel
 */
template <typename Key, typename Value>
cudaError_t Dispatch(
    Int2Type<true>      use_cnp,
    int                 timing_iterations,
    size_t              *d_temp_storage_bytes,
    int                 *d_selectors,
    cudaError_t         *d_cnp_error,

    void                *d_temp_storage,
    size_t              &temp_storage_bytes,
    DoubleBuffer<Key>   &d_keys,
    DoubleBuffer<Value> &d_values,
    int                 num_items,
    int                 begin_bit,
    int                 end_bit,
    cudaStream_t        stream,
    bool                stream_synchronous)
{
    // Invoke kernel to invoke device-side dispatch
    CnpDispatchKernel<<<1,1>>>(timing_iterations, d_temp_storage_bytes, d_selectors, d_cnp_error, d_temp_storage, temp_storage_bytes, d_keys, d_values, num_items, begin_bit, end_bit, stream_synchronous);

    // Copy out temp_storage_bytes
    CubDebugExit(cudaMemcpy(&temp_storage_bytes, d_temp_storage_bytes, sizeof(size_t) * 1, cudaMemcpyDeviceToHost));

    // Copy out selectors
    CubDebugExit(cudaMemcpy(&d_keys.selector, d_selectors + 0, sizeof(int) * 1, cudaMemcpyDeviceToHost));
    CubDebugExit(cudaMemcpy(&d_values.selector, d_selectors + 1, sizeof(int) * 1, cudaMemcpyDeviceToHost));

    // Copy out error
    cudaError_t retval;
    CubDebugExit(cudaMemcpy(&retval, d_cnp_error, sizeof(cudaError_t) * 1, cudaMemcpyDeviceToHost));
    return retval;
}



//---------------------------------------------------------------------
// Test generation
//---------------------------------------------------------------------


/**
 * Simple key-value pairing
 */
template <typename Key, typename Value>
struct Pair
{
    Key     key;
    Value   value;

    bool operator<(const Pair &b)
    {
        return (key < b.key);
    }
};


/**
 * Initialize key-value sorting problem
 */
template <typename Key, typename Value>
void Initialize(
    GenMode      gen_mode,
    Key          *h_keys,
    Value        *h_values,
    Key          *h_sorted_keys,
    Value        *h_sorted_values,
    int          num_items)
{

    Pair<Key, Value> *pairs = new Pair<Key, Value>[num_items];
    for (int i = 0; i < num_items; ++i)
    {
//        InitValue(gen_mode, h_keys[i], i);
//        h_keys[i]       = i % 4;
        h_keys[i]       = i % 32;
        h_values[i]     = i;
        pairs[i].key    = h_keys[i];
        pairs[i].value  = h_keys[i];
    }

    std::sort(pairs, pairs + num_items);

    for (int i = 0; i < num_items; ++i)
    {
        h_sorted_keys[i]    = pairs[i].key;
        h_sorted_values[i]  = pairs[i].value;
    }

    delete[] pairs;
}


/**
 * Initialize keys-only sorting problem
 */
template <typename Key, typename Value>
void Initialize(
    GenMode      gen_mode,
    Key          *h_keys,
    NullType     *h_values,
    Key          *h_sorted_keys,
    NullType     *h_sorted_values,
    int          num_items)
{
    for (int i = 0; i < num_items; ++i)
    {
//        InitValue(gen_mode, h_keys[i], i);
//        h_keys[i]           = i % 4;
        h_keys[i]           = i % 32;
        h_sorted_keys[i]    = h_keys[i];
    }

    std::sort(h_sorted_keys, h_sorted_keys + num_items);
}


/**
 * Test DeviceRadixSort
 */
template <
    bool            CNP,
    typename        Key,
    typename        Value>
void Test(
    int             num_items,
    GenMode         gen_mode,
    int             begin_bit,
    int             end_bit,
    char*           type_string)
{
    const bool KEYS_ONLY = Equals<Value, NullType>::VALUE;

    if (KEYS_ONLY)
        printf("%s keys-only cub::DeviceRadixSort %d items, %s %d-byte keys, gen-mode %s\n",
            (CNP) ? "CNP device invoked" : "Host-invoked", num_items, type_string, (int) sizeof(Key),
            (gen_mode == RANDOM) ? "RANDOM" : (gen_mode == SEQ_INC) ? "SEQUENTIAL" : "HOMOGENOUS");
    else
        printf("%s keys-value cub::DeviceRadixSort %d items, %s %d-byte keys + %d-byte values, gen-mode %s\n",
            (CNP) ? "CNP device invoked" : "Host-invoked", num_items, type_string, (int) sizeof(Key), (int) sizeof(Value),
            (gen_mode == RANDOM) ? "RANDOM" : (gen_mode == SEQ_INC) ? "SEQUENTIAL" : "HOMOGENOUS");
    fflush(stdout);

    // Allocate host arrays
    Key     *h_keys             = new Key[num_items];
    Key     *h_sorted_keys      = new Key[num_items];
    Value   *h_values           = (KEYS_ONLY) ? NULL : new Value[num_items];
    Value   *h_sorted_values    = (KEYS_ONLY) ? NULL : new Value[num_items];

    // Initialize problem
    Initialize(gen_mode, h_keys, h_values, h_sorted_keys, h_sorted_values, num_items);

    // Allocate device arrays
    DoubleBuffer<Key>   d_keys;
    DoubleBuffer<Value> d_values;
    size_t          *d_temp_storage_bytes;
    int             *d_selectors;
    cudaError_t     *d_cnp_error;
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_keys.d_buffers[0], sizeof(Key) * num_items));
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_keys.d_buffers[1], sizeof(Key) * num_items));
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_temp_storage_bytes, sizeof(size_t) * 1));
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_selectors, sizeof(int) * 2));
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_cnp_error, sizeof(cudaError_t) * 1));
    if (!KEYS_ONLY)
    {
        CubDebugExit(g_allocator.DeviceAllocate((void**)&d_values.d_buffers[0], sizeof(Value) * num_items));
        CubDebugExit(g_allocator.DeviceAllocate((void**)&d_values.d_buffers[1], sizeof(Value) * num_items));
    }

    // Initialize device arrays
    CubDebugExit(cudaMemcpy(d_keys.d_buffers[0], h_keys, sizeof(Key) * num_items, cudaMemcpyHostToDevice));
    CubDebugExit(cudaMemset(d_keys.d_buffers[1], 0, sizeof(Key) * num_items));
    if (!KEYS_ONLY)
    {
        CubDebugExit(cudaMemcpy(d_keys.d_buffers[0], h_keys, sizeof(Key) * num_items, cudaMemcpyHostToDevice));
        CubDebugExit(cudaMemset(d_values.d_buffers[1], 0, sizeof(Value) * num_items));
    }

    // Allocate temporary storage
    size_t  temp_storage_bytes  = 0;
    void    *d_temp_storage     = NULL;
    CubDebugExit(Dispatch(Int2Type<CNP>(), 1, d_temp_storage_bytes, d_selectors, d_cnp_error, d_temp_storage, temp_storage_bytes, d_keys, d_values, num_items, begin_bit, end_bit, 0, true));
    CubDebugExit(g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes));

    // Run warmup/correctness iteration
    CubDebugExit(Dispatch(Int2Type<CNP>(), 1, d_temp_storage_bytes, d_selectors, d_cnp_error, d_temp_storage, temp_storage_bytes, d_keys, d_values, num_items, begin_bit, end_bit, 0, true));

    // Check for correctness (and display results, if specified)
    int compare = CompareDeviceResults(h_sorted_keys, d_keys.Current(), num_items, true, g_verbose);
    printf("\t%s", compare ? "FAIL" : "PASS");
    if (!KEYS_ONLY)
    {
        int values_compare = CompareDeviceResults(h_sorted_values, d_values.Current(), num_items, true, g_verbose);
        compare |= values_compare;
        printf("\t%s", values_compare ? "FAIL" : "PASS");
    }

    // Flush any stdout/stderr
    fflush(stdout);
    fflush(stderr);

    // Performance
    GpuTimer gpu_timer;
    gpu_timer.Start();
    CubDebugExit(Dispatch(Int2Type<CNP>(), g_timing_iterations, d_temp_storage_bytes, d_selectors, d_cnp_error, d_temp_storage, temp_storage_bytes, d_keys, d_values, num_items, begin_bit, end_bit, 0, false));
    gpu_timer.Stop();
    float elapsed_millis = gpu_timer.ElapsedMillis();

    // Display performance
    if (g_timing_iterations > 0)
    {
        float avg_millis = elapsed_millis / g_timing_iterations;
        float grate = float(num_items) / avg_millis / 1000.0 / 1000.0;
        float gbandwidth = (KEYS_ONLY) ?
            grate * sizeof(Key) * 2 :
            grate * (sizeof(Key) + sizeof(Value)) * 2;
        printf(", %.3f avg ms, %.3f billion items/s, %.3f logical GB/s", avg_millis, grate, gbandwidth);
    }

    printf("\n\n");

    // Cleanup
    if (h_keys) delete[] h_keys;
    if (h_sorted_keys) delete[] h_sorted_keys;
    if (h_values) delete[] h_values;
    if (h_sorted_values) delete[] h_sorted_values;

    if (d_keys.d_buffers[0]) CubDebugExit(g_allocator.DeviceFree(d_keys.d_buffers[0]));
    if (d_keys.d_buffers[1]) CubDebugExit(g_allocator.DeviceFree(d_keys.d_buffers[1]));
    if (d_values.d_buffers[0]) CubDebugExit(g_allocator.DeviceFree(d_values.d_buffers[0]));
    if (d_values.d_buffers[1]) CubDebugExit(g_allocator.DeviceFree(d_values.d_buffers[1]));
    if (d_selectors) CubDebugExit(g_allocator.DeviceFree(d_selectors));
    if (d_temp_storage_bytes) CubDebugExit(g_allocator.DeviceFree(d_temp_storage_bytes));
    if (d_cnp_error) CubDebugExit(g_allocator.DeviceFree(d_cnp_error));

    // Correctness asserts
    AssertEquals(0, compare);
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

//    Test<false, unsigned int, NullType>(num_items, UNIFORM, 0, sizeof(unsigned int) * 8, CUB_TYPE_STRING(unsigned int));
    Test<false, unsigned int, NullType>(num_items, UNIFORM, 0, 5, CUB_TYPE_STRING(unsigned int));

    return 0;
}



