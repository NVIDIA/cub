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

#include <cub/util_allocator.cuh>
#include <cub/device/device_radix_sort.cuh>

#include <thrust/device_ptr.h>
#include <thrust/sort.h>

#include "test_util.h"

using namespace cub;


//---------------------------------------------------------------------
// Globals, constants and typedefs
//---------------------------------------------------------------------

bool                    g_verbose           = false;
int                     g_timing_iterations = 0;
int                     g_repeat            = 0;
int                     g_bits              = -1;
CachingDeviceAllocator  g_allocator;

// Dispatch types
enum Backend
{
    CUB,
    THRUST,
    CDP,
};

//---------------------------------------------------------------------
// Dispatch to different DeviceRadixSort entrypoints
//---------------------------------------------------------------------

/**
 * Dispatch to CUB sorting entrypoint
 */
template <typename Key, typename Value>
__host__ __device__ __forceinline__
cudaError_t Dispatch(
    Int2Type<CUB>       dispatch_to,
    int                 *d_selector,
    size_t              *d_temp_storage_bytes,
    cudaError_t         *d_cdp_error,

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
    return DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_keys, d_values, num_items, begin_bit, end_bit, stream, stream_synchronous);
}


/**
 * Dispatch keys-only to Thrust sorting entrypoint
 */
template <typename Key>
cudaError_t Dispatch(
    Int2Type<THRUST>        dispatch_to,
    int                     *d_selector,
    size_t                  *d_temp_storage_bytes,
    cudaError_t             *d_cdp_error,
    void                    *d_temp_storage,
    size_t                  &temp_storage_bytes,
    DoubleBuffer<Key>       &d_keys,
    DoubleBuffer<NullType>  &d_values,
    int                     num_items,
    int                     begin_bit,
    int                     end_bit,
    cudaStream_t            stream,
    bool                    stream_synchronous)
{
    if (d_temp_storage == 0)
    {
        temp_storage_bytes = 1;
    }
    else
    {
        thrust::device_ptr<Key>     d_keys_wrapper(d_keys.Current());
        thrust::sort(d_keys_wrapper, d_keys_wrapper + num_items);
    }

    return cudaSuccess;
}


/**
 * Dispatch key-value pairs to Thrust sorting entrypoint
 */
template <typename Key, typename Value>
cudaError_t Dispatch(
    Int2Type<THRUST>    dispatch_to,
    int                 *d_selector,
    size_t              *d_temp_storage_bytes,
    cudaError_t         *d_cdp_error,
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
    if (d_temp_storage == 0)
    {
        temp_storage_bytes = 1;
    }
    else
    {
        thrust::device_ptr<Key>     d_keys_wrapper(d_keys.Current());
        thrust::device_ptr<Value>   d_values_wrapper(d_values.Current());
        thrust::sort_by_key(d_keys_wrapper, d_keys_wrapper + num_items, d_values_wrapper);
    }

    return cudaSuccess;
}


//---------------------------------------------------------------------
// CUDA Nested Parallelism Test Kernel
//---------------------------------------------------------------------

/**
 * Simple wrapper kernel to invoke DeviceRadixSort
 */
template <typename Key, typename Value>
__global__ void CnpDispatchKernel(
    int                 *d_selector,
    size_t              *d_temp_storage_bytes,
    cudaError_t         *d_cdp_error,

    void                *d_temp_storage,
    size_t              temp_storage_bytes,
    DoubleBuffer<Key>   d_keys,
    DoubleBuffer<Value> d_values,
    int                 num_items,
    int                 begin_bit,
    int                 end_bit,
    bool                stream_synchronous)
{
#ifndef CUB_CDP
    *d_cdp_error = cudaErrorNotSupported;
#else
    *d_cdp_error            = Dispatch(Int2Type<CUB>(), d_selector, d_temp_storage_bytes, d_cdp_error, d_temp_storage, temp_storage_bytes, d_keys, d_values, num_items, begin_bit, end_bit, 0, stream_synchronous);
    *d_temp_storage_bytes   = temp_storage_bytes;
    *d_selector             = d_keys.selector;
#endif
}


/**
 * Dispatch to CDP kernel
 */
template <typename Key, typename Value>
cudaError_t Dispatch(
    Int2Type<CDP>       dispatch_to,
    int                 *d_selector,
    size_t              *d_temp_storage_bytes,
    cudaError_t         *d_cdp_error,

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
    CnpDispatchKernel<<<1,1>>>(d_selector, d_temp_storage_bytes, d_cdp_error, d_temp_storage, temp_storage_bytes, d_keys, d_values, num_items, begin_bit, end_bit, stream_synchronous);

    // Copy out selector
    CubDebugExit(cudaMemcpy(&d_keys.selector, d_selector, sizeof(int) * 1, cudaMemcpyDeviceToHost));
    d_values.selector = d_keys.selector;

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
 * Simple key-value pairing
 */
template <typename Key, typename Value>
struct Pair
{
    Key     key;
    Value   value;

    bool operator<(const Pair &b) const
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
        if (gen_mode == RANDOM) {
            RandomBits(h_keys[i], 0, 0, g_bits);
        } else if (gen_mode == UNIFORM) {
            h_keys[i] = 2;
        } else {
            h_keys[i] = i;
        }

        h_values[i]     = i;

        pairs[i].key    = h_keys[i];
        pairs[i].value  = h_values[i];
    }

    std::stable_sort(pairs, pairs + num_items);

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
template <typename Key>
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
        if (gen_mode == RANDOM) {
            RandomBits(h_keys[i], 0, 0, g_bits);
        } else if (gen_mode == UNIFORM) {
            h_keys[i] = 2;
        } else {
            h_keys[i] = i;
        }

        h_sorted_keys[i] = h_keys[i];
    }

    std::stable_sort(h_sorted_keys, h_sorted_keys + num_items);
}


/**
 * Test DeviceRadixSort
 */
template <
    Backend         BACKEND,
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

    if (end_bit < 0) end_bit = sizeof(Key) * 8;

    if (KEYS_ONLY)
        printf("%s keys-only cub::DeviceRadixSort %d items, %s %d-byte keys, gen-mode %s\n",
            (BACKEND == CDP) ? "CDP CUB" : (BACKEND == THRUST) ? "Thrust" : "CUB",
            num_items, type_string, (int) sizeof(Key),
            (gen_mode == RANDOM) ? "RANDOM" : (gen_mode == SEQ_INC) ? "SEQUENTIAL" : "HOMOGENOUS");
    else
        printf("%s keys-value cub::DeviceRadixSort %d items, %s %d-byte keys + %d-byte values, gen-mode %s\n",
            (BACKEND == CDP) ? "CDP CUB" : (BACKEND == THRUST) ? "Thrust" : "CUB",
            num_items, type_string, (int) sizeof(Key), (int) sizeof(Value),
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
    int                 *d_selector;
    size_t              *d_temp_storage_bytes;
    cudaError_t         *d_cdp_error;
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_keys.d_buffers[0], sizeof(Key) * num_items));
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_keys.d_buffers[1], sizeof(Key) * num_items));
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_selector, sizeof(int) * 1));
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_temp_storage_bytes, sizeof(size_t) * 1));
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_cdp_error, sizeof(cudaError_t) * 1));
    if (!KEYS_ONLY)
    {
        CubDebugExit(g_allocator.DeviceAllocate((void**)&d_values.d_buffers[0], sizeof(Value) * num_items));
        CubDebugExit(g_allocator.DeviceAllocate((void**)&d_values.d_buffers[1], sizeof(Value) * num_items));
    }

    // Allocate temporary storage
    size_t  temp_storage_bytes  = 0;
    void    *d_temp_storage     = NULL;
    CubDebugExit(Dispatch(Int2Type<BACKEND>(), d_selector, d_temp_storage_bytes, d_cdp_error, d_temp_storage, temp_storage_bytes, d_keys, d_values, num_items, begin_bit, end_bit, 0, true));
    CubDebugExit(g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes));

    // Initialize/clear device arrays
    CubDebugExit(cudaMemcpy(d_keys.d_buffers[d_keys.selector], h_keys, sizeof(Key) * num_items, cudaMemcpyHostToDevice));
    CubDebugExit(cudaMemset(d_keys.d_buffers[d_keys.selector ^ 1], 0, sizeof(Key) * num_items));
    if (!KEYS_ONLY)
    {
        CubDebugExit(cudaMemcpy(d_values.d_buffers[d_values.selector], h_values, sizeof(Value) * num_items, cudaMemcpyHostToDevice));
        CubDebugExit(cudaMemset(d_values.d_buffers[d_values.selector ^ 1], 0, sizeof(Value) * num_items));
    }

    // Run warmup/correctness iteration
    CubDebugExit(Dispatch(Int2Type<BACKEND>(), d_selector, d_temp_storage_bytes, d_cdp_error, d_temp_storage, temp_storage_bytes, d_keys, d_values, num_items, begin_bit, end_bit, 0, true));

    // Check for correctness (and display results, if specified)
    int compare = CompareDeviceResults(h_sorted_keys, d_keys.Current(), num_items, true, g_verbose);
    printf("\t Compare keys (selector %d): %s ", d_keys.selector, compare ? "FAIL" : "PASS");
    if (!KEYS_ONLY)
    {
        int values_compare = CompareDeviceResults(h_sorted_values, d_values.Current(), num_items, true, g_verbose);
        compare |= values_compare;
        printf("\t Compare values (selector %d): %s ", d_values.selector, values_compare ? "FAIL" : "PASS");
    }

    // Flush any stdout/stderr
    fflush(stdout);
    fflush(stderr);

    // Performance
    GpuTimer gpu_timer;
    float elapsed_millis = 0.0f;
    for (int i = 0; i < g_timing_iterations; ++i)
    {
        // Initialize/clear device arrays
        CubDebugExit(cudaMemcpy(d_keys.d_buffers[d_keys.selector], h_keys, sizeof(Key) * num_items, cudaMemcpyHostToDevice));
        CubDebugExit(cudaMemset(d_keys.d_buffers[d_keys.selector ^ 1], 0, sizeof(Key) * num_items));
        if (!KEYS_ONLY)
        {
            CubDebugExit(cudaMemcpy(d_values.d_buffers[d_values.selector], h_values, sizeof(Value) * num_items, cudaMemcpyHostToDevice));
            CubDebugExit(cudaMemset(d_values.d_buffers[d_values.selector ^ 1], 0, sizeof(Value) * num_items));
        }

        gpu_timer.Start();
        CubDebugExit(Dispatch(Int2Type<BACKEND>(), d_selector, d_temp_storage_bytes, d_cdp_error, d_temp_storage, temp_storage_bytes, d_keys, d_values, num_items, begin_bit, end_bit, 0, false));
        gpu_timer.Stop();
        elapsed_millis += gpu_timer.ElapsedMillis();
    }

    // Display performance
    if (g_timing_iterations > 0)
    {
        float avg_millis = elapsed_millis / g_timing_iterations;
        float grate = float(num_items) / avg_millis / 1000.0 / 1000.0;
        float gbandwidth = (KEYS_ONLY) ?
            grate * sizeof(Key) * 2 :
            grate * (sizeof(Key) + sizeof(Value)) * 2;
        printf("\n%.3f elapsed ms, %.3f avg ms, %.3f billion items/s, %.3f logical GB/s", elapsed_millis, avg_millis, grate, gbandwidth);
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
    if (d_temp_storage_bytes) CubDebugExit(g_allocator.DeviceFree(d_temp_storage_bytes));
    if (d_cdp_error) CubDebugExit(g_allocator.DeviceFree(d_cdp_error));

    // Correctness asserts
    AssertEquals(0, compare);
}


/**
 * Test problem generation
 */
template <
    Backend         BACKEND,
    typename        Key,
    typename        Value>
void Test(
    int             num_items,
    int             begin_bit,
    int             end_bit,
    char*           type_string)
{
    Test<BACKEND, Key, Value>(num_items, RANDOM, begin_bit, end_bit, type_string);
    Test<BACKEND, Key, Value>(num_items, UNIFORM, begin_bit, end_bit, type_string);
    Test<BACKEND, Key, Value>(num_items, SEQ_INC, begin_bit, end_bit, type_string);
}

/**
 * Test CDP and num items
 */
template <
    typename        Key,
    typename        Value>
void Test(
    int             num_items,
    int             begin_bit,
    int             end_bit,
    char*           type_string)
{
    Test<CUB, Key, Value>(num_items, begin_bit, end_bit, type_string);

#ifdef CUB_CDP
    Test<CDP, Key, Value>(num_items, begin_bit, end_bit, type_string);
#endif
}


/**
 * Test CDP and num items
 */
template <
    typename        Key,
    typename        Value>
void TestItems(
    int             num_items,
    int             begin_bit,
    int             end_bit,
    char*           type_string)
{
    if (num_items < 0)
    {
        Test<Key, Value>(1, begin_bit, end_bit, type_string);
        Test<Key, Value>(32, begin_bit, end_bit, type_string);
        Test<Key, Value>(3200, begin_bit, end_bit, type_string);
        Test<Key, Value>(320000, begin_bit, end_bit, type_string);
        Test<Key, Value>(32000000, begin_bit, end_bit, type_string);
    }
    else
    {
        Test<Key, Value>(num_items, begin_bit, end_bit, type_string);
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
    args.GetCmdLineArgument("bits", g_bits);

    // Print usage
    if (args.CheckCmdLineFlag("help"))
    {
        printf("%s "
            "[--bits=<valid key bits>]"
            "[--n=<input items> "
            "[--i=<timing iterations> "
            "[--device=<device-id>] "
            "[--repeat=<times to repeat tests>]"
            "[--quick]"
            "[--v] "
            "[--cdp]"
            "\n", argv[0]);
        exit(0);
    }

    // Initialize device
    CubDebugExit(args.DeviceInit());
    printf("\n");

    if (quick)
    {
        if (num_items < 0) num_items = 32000000;

        Test<CUB, unsigned int, NullType> (num_items, RANDOM, 0, g_bits, CUB_TYPE_STRING(unsigned int));
        Test<THRUST, unsigned int, NullType> (num_items, RANDOM, 0, g_bits, CUB_TYPE_STRING(unsigned int));

        Test<CUB, unsigned int, unsigned int> (num_items, RANDOM, 0, g_bits, CUB_TYPE_STRING(unsigned int));
        Test<THRUST, unsigned int, unsigned int> (num_items, RANDOM, 0, g_bits, CUB_TYPE_STRING(unsigned int));
    }
    else
    {
        TestItems<unsigned int, NullType>        (num_items, 0, g_bits, CUB_TYPE_STRING(unsigned int));
        TestItems<unsigned long long, NullType>  (num_items, 0, g_bits, CUB_TYPE_STRING(unsigned long long));
        TestItems<unsigned int, unsigned int>    (num_items, 0, g_bits, CUB_TYPE_STRING(unsigned int));
        TestItems<int, NullType>                 (num_items, 0, g_bits, CUB_TYPE_STRING(int));
        TestItems<double, NullType>              (num_items, 0, g_bits, CUB_TYPE_STRING(double));
        TestItems<float, NullType>               (num_items, 0, g_bits, CUB_TYPE_STRING(unsigned int));
    }

    return 0;
}



