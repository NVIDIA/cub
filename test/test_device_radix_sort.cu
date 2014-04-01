/******************************************************************************
 * Copyright (c) 2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2014, NVIDIA CORPORATION.  All rights reserved.
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
#include <thrust/reverse.h>

#include "test_util.h"

using namespace cub;


//---------------------------------------------------------------------
// Globals, constants and typedefs
//---------------------------------------------------------------------

bool                    g_verbose           = false;
int                     g_timing_iterations = 0;
int                     g_repeat            = 0;
int                     g_bits              = -1;
CachingDeviceAllocator  g_allocator(true);

//---------------------------------------------------------------------
// Dispatch to different DeviceRadixSort entrypoints
//---------------------------------------------------------------------

/**
 * Dispatch to CUB sorting entrypoint (specialized for ascending)
 */
template <typename Key, typename Value>
__host__ __device__ __forceinline__
cudaError_t Dispatch(
    Int2Type<false>     is_descending,
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
    bool                debug_synchronous)
{
    return DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_keys, d_values, num_items, begin_bit, end_bit, stream, debug_synchronous);
}


/**
 * Dispatch to CUB sorting entrypoint (specialized for descending)
 */
template <typename Key, typename Value>
__host__ __device__ __forceinline__
cudaError_t Dispatch(
    Int2Type<true>      is_descending,
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
    bool                debug_synchronous)
{
    return DeviceRadixSort::SortPairsDescending(d_temp_storage, temp_storage_bytes, d_keys, d_values, num_items, begin_bit, end_bit, stream, debug_synchronous);
}


/**
 * Dispatch keys-only to Thrust sorting entrypoint
 */
template <int DESCENDING, typename Key>
cudaError_t Dispatch(
    Int2Type<DESCENDING>    is_descending,
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
    bool                    debug_synchronous)
{

    if (d_temp_storage == 0)
    {
        temp_storage_bytes = 1;
    }
    else
    {
        thrust::device_ptr<Key> d_keys_wrapper(d_keys.Current());

        if (DESCENDING) thrust::reverse(d_keys_wrapper, d_keys_wrapper + num_items);
        thrust::sort(d_keys_wrapper, d_keys_wrapper + num_items);
        if (DESCENDING) thrust::reverse(d_keys_wrapper, d_keys_wrapper + num_items);
    }

    return cudaSuccess;
}


/**
 * Dispatch key-value pairs to Thrust sorting entrypoint
 */
template <int DESCENDING, typename Key, typename Value>
cudaError_t Dispatch(
    Int2Type<DESCENDING>    is_descending,
    Int2Type<THRUST>        dispatch_to,
    int                     *d_selector,
    size_t                  *d_temp_storage_bytes,
    cudaError_t             *d_cdp_error,
    void                    *d_temp_storage,
    size_t                  &temp_storage_bytes,
    DoubleBuffer<Key>       &d_keys,
    DoubleBuffer<Value>     &d_values,
    int                     num_items,
    int                     begin_bit,
    int                     end_bit,
    cudaStream_t            stream,
    bool                    debug_synchronous)
{

    if (d_temp_storage == 0)
    {
        temp_storage_bytes = 1;
    }
    else
    {
        thrust::device_ptr<Key>     d_keys_wrapper(d_keys.Current());
        thrust::device_ptr<Value>   d_values_wrapper(d_values.Current());

        if (DESCENDING) {
            thrust::reverse(d_keys_wrapper, d_keys_wrapper + num_items);
            thrust::reverse(d_values_wrapper, d_values_wrapper + num_items);
        }

        thrust::sort_by_key(d_keys_wrapper, d_keys_wrapper + num_items, d_values_wrapper);

        if (DESCENDING) {
            thrust::reverse(d_keys_wrapper, d_keys_wrapper + num_items);
            thrust::reverse(d_values_wrapper, d_values_wrapper + num_items);
        }
    }

    return cudaSuccess;
}


//---------------------------------------------------------------------
// CUDA Nested Parallelism Test Kernel
//---------------------------------------------------------------------

/**
 * Simple wrapper kernel to invoke DeviceRadixSort
 */
template <int DESCENDING, typename Key, typename Value>
__global__ void CnpDispatchKernel(
    Int2Type<DESCENDING>    is_descending,
    int                     *d_selector,
    size_t                  *d_temp_storage_bytes,
    cudaError_t             *d_cdp_error,

    void                    *d_temp_storage,
    size_t                  temp_storage_bytes,
    DoubleBuffer<Key>       d_keys,
    DoubleBuffer<Value>     d_values,
    int                     num_items,
    int                     begin_bit,
    int                     end_bit,
    bool                    debug_synchronous)
{
#ifndef CUB_CDP
    *d_cdp_error = cudaErrorNotSupported;
#else
    *d_cdp_error            = Dispatch(is_descending, Int2Type<CUB>(), d_selector, d_temp_storage_bytes, d_cdp_error, d_temp_storage, temp_storage_bytes, d_keys, d_values, num_items, begin_bit, end_bit, 0, debug_synchronous);
    *d_temp_storage_bytes   = temp_storage_bytes;
    *d_selector             = d_keys.selector;
#endif
}


/**
 * Dispatch to CDP kernel
 */
template <int DESCENDING, typename Key, typename Value>
cudaError_t Dispatch(
    Int2Type<DESCENDING>    is_descending,
    Int2Type<CDP>           dispatch_to,
    int                     *d_selector,
    size_t                  *d_temp_storage_bytes,
    cudaError_t             *d_cdp_error,

    void                    *d_temp_storage,
    size_t                  &temp_storage_bytes,
    DoubleBuffer<Key>       &d_keys,
    DoubleBuffer<Value>     &d_values,
    int                     num_items,
    int                     begin_bit,
    int                     end_bit,
    cudaStream_t            stream,
    bool                    debug_synchronous)
{
    // Invoke kernel to invoke device-side dispatch
    CnpDispatchKernel<<<1,1>>>(is_descending, d_selector, d_temp_storage_bytes, d_cdp_error, d_temp_storage, temp_storage_bytes, d_keys, d_values, num_items, begin_bit, end_bit, debug_synchronous);

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
template <
    typename Key,
    typename Value,
    bool IS_FLOAT = (Traits<Key>::CATEGORY == FLOATING_POINT)>
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
 * Simple key-value pairing (specialized for floating point types)
 */
template <typename Key, typename Value>
struct Pair<Key, Value, true>
{
    Key     key;
    Value   value;

    bool operator<(const Pair &b) const
    {
        if (key < b.key)
            return true;

        if (key > b.key)
            return false;

        // Key in unsigned bits
        typedef typename Traits<Key>::UnsignedBits UnsignedBits;

        // Return true if key is negative zero and b.key is positive zero
        UnsignedBits key_bits   = *reinterpret_cast<UnsignedBits*>(const_cast<Key*>(&key));
        UnsignedBits b_key_bits = *reinterpret_cast<UnsignedBits*>(const_cast<Key*>(&b.key));
        UnsignedBits HIGH_BIT   = Traits<Key>::HIGH_BIT;

        return ((key_bits & HIGH_BIT) != 0) && ((b_key_bits & HIGH_BIT) == 0);
    }
};


/**
 * Initialize key-value sorting problem.
 */
template <int DESCENDING, typename Key, typename Value>
void Initialize(
    GenMode         gen_mode,
    Key             *h_keys,
    Value           *h_values,
    Key             **h_reference_keys,
    Value           **h_reference_values,
    int             num_items,
    int             entropy_reduction,
    int             begin_bit,
    int             end_bit)
{
    const bool KEYS_ONLY = Equals<Value, NullType>::VALUE;

    Pair<Key, Value> *h_pairs = new Pair<Key, Value>[num_items];

    for (int i = 0; i < num_items; ++i)
    {
        if (gen_mode == RANDOM) {
            RandomBits(h_keys[i], entropy_reduction);
        } else if (gen_mode == UNIFORM) {
            h_keys[i] = 1;
        } else {
            h_keys[i] = i;
        }

        if (h_values != NULL)
            RandomBits(h_values[i]);

        // Mask off unwanted portions
        int num_bits = end_bit - begin_bit;
        if ((begin_bit > 0) || (end_bit < sizeof(Key) * 8))
        {
            unsigned long long base = 0;
            memcpy(&base, &h_keys[i], sizeof(Key));
            base &= ((1ull << num_bits) - 1) << begin_bit;
            memcpy(&h_keys[i], &base, sizeof(Key));
        }

        h_pairs[i].key    = h_keys[i];
        h_pairs[i].value  = h_values[i];
    }

    if (g_verbose)
    {
        printf("Input keys:\n");
        DisplayResults(h_keys, num_items);
        printf("\n\n");
        if (!KEYS_ONLY)
        {
            printf("Input values:\n");
            DisplayResults(h_values, num_items);
            printf("\n\n");
        }
    }


    if (DESCENDING) std::reverse(h_pairs, h_pairs + num_items);
    std::stable_sort(h_pairs, h_pairs + num_items);
    if (DESCENDING) std::reverse(h_pairs, h_pairs + num_items);

    *h_reference_keys   = new Key[num_items];
    *h_reference_values = (KEYS_ONLY) ? NULL : new Value[num_items];

    for (int i = 0; i < num_items; ++i)
    {
        (*h_reference_keys)[i]     = h_pairs[i].key;

        if ((*h_reference_values) != NULL)
            (*h_reference_values)[i]   = h_pairs[i].value;
    }

    delete[] h_pairs;
}




/**
 * Test DeviceRadixSort
 */
template <
    Backend         BACKEND,
    typename        Key,
    typename        Value,
    bool            DESCENDING>
void Test(
    int             num_items,
    GenMode         gen_mode,
    int             entropy_reduction,
    int             begin_bit,
    int             end_bit,
    char*           type_string)
{
    const bool KEYS_ONLY = Equals<Value, NullType>::VALUE;

    if (end_bit < 0) end_bit = sizeof(Key) * 8;

    printf("%s %s cub::DeviceRadixSort %d items, %s %d-byte keys %d-byte values, gen-mode %s, descending %d, entropy_reduction %d, begin_bit %d, end_bit %d\n",
        (BACKEND == CDP) ? "CDP CUB" : (BACKEND == THRUST) ? "Thrust" : "CUB",
        (KEYS_ONLY) ? "keys-only" : "key-value",
        num_items, type_string,
        (int) sizeof(Key), (KEYS_ONLY) ? 0 : (int) sizeof(Value),
        (gen_mode == RANDOM) ? "RANDOM" : (gen_mode == INTEGER_SEED) ? "SEQUENTIAL" : "HOMOGENOUS",
        DESCENDING, entropy_reduction, begin_bit, end_bit);
    fflush(stdout);

    // Allocate host arrays
    Key     *h_keys             = new Key[num_items];
    Value   *h_values           = (KEYS_ONLY) ? NULL : new Value[num_items];

    Key     *h_reference_keys;
    Value   *h_reference_values;

    // Initialize problem and solution on host
    Initialize<DESCENDING>(
        gen_mode,
        h_keys,
        h_values,
        &h_reference_keys,
        &h_reference_values,
        num_items,
        entropy_reduction,
        begin_bit,
        end_bit);

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
    CubDebugExit(Dispatch(Int2Type<DESCENDING>(), Int2Type<BACKEND>(), d_selector, d_temp_storage_bytes, d_cdp_error, d_temp_storage, temp_storage_bytes, d_keys, d_values, num_items, begin_bit, end_bit, 0, true));
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
    CubDebugExit(Dispatch(Int2Type<DESCENDING>(), Int2Type<BACKEND>(), d_selector, d_temp_storage_bytes, d_cdp_error, d_temp_storage, temp_storage_bytes, d_keys, d_values, num_items, begin_bit, end_bit, 0, true));

    // Check for correctness (and display results, if specified)
    int compare = CompareDeviceResults(h_reference_keys, d_keys.Current(), num_items, true, g_verbose);
    printf("\t Compare keys (selector %d): %s ", d_keys.selector, compare ? "FAIL" : "PASS");
    if (!KEYS_ONLY)
    {
        int values_compare = CompareDeviceResults(h_reference_values, d_values.Current(), num_items, true, g_verbose);
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
        CubDebugExit(Dispatch(Int2Type<DESCENDING>(), Int2Type<BACKEND>(), d_selector, d_temp_storage_bytes, d_cdp_error, d_temp_storage, temp_storage_bytes, d_keys, d_values, num_items, begin_bit, end_bit, 0, false));
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
    if (h_reference_keys) delete[] h_reference_keys;
    if (h_values) delete[] h_values;
    if (h_reference_values) delete[] h_reference_values;

    if (d_keys.d_buffers[0]) CubDebugExit(g_allocator.DeviceFree(d_keys.d_buffers[0]));
    if (d_keys.d_buffers[1]) CubDebugExit(g_allocator.DeviceFree(d_keys.d_buffers[1]));
    if (d_values.d_buffers[0]) CubDebugExit(g_allocator.DeviceFree(d_values.d_buffers[0]));
    if (d_values.d_buffers[1]) CubDebugExit(g_allocator.DeviceFree(d_values.d_buffers[1]));
    if (d_temp_storage) CubDebugExit(g_allocator.DeviceFree(d_temp_storage));
    if (d_cdp_error) CubDebugExit(g_allocator.DeviceFree(d_cdp_error));
    if (d_selector) CubDebugExit(g_allocator.DeviceFree(d_selector));
    if (d_temp_storage_bytes) CubDebugExit(g_allocator.DeviceFree(d_temp_storage_bytes));

    // Correctness asserts
    AssertEquals(0, compare);
}

/**
 * Test ascending/descending
 */
template <
    Backend         BACKEND,
    typename        Key,
    typename        Value>
void Test(
    int             num_items,
    GenMode         gen_mode,
    int             entropy_reduction,
    int             begin_bit,
    int             end_bit,
    char*           type_string)
{
    Test<BACKEND, Key, Value, false>(num_items, gen_mode, entropy_reduction, begin_bit, end_bit, type_string);
    Test<BACKEND, Key, Value, true>(num_items, gen_mode, entropy_reduction, begin_bit, end_bit, type_string);
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
    for (int entropy_reduction = 0; entropy_reduction <= 6; entropy_reduction += 3)
    {
        Test<BACKEND, Key, Value>(num_items, RANDOM, entropy_reduction, begin_bit, end_bit, type_string);
    }

    Test<BACKEND, Key, Value>(num_items, UNIFORM, 0, begin_bit, end_bit, type_string);
    Test<BACKEND, Key, Value>(num_items, INTEGER_SEED, 0, begin_bit, end_bit, type_string);
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
    // Get ptx version
    int ptx_version;
    CubDebugExit(PtxVersion(ptx_version));

    if (num_items < 0)
    {
        Test<Key, Value>(1, begin_bit, end_bit, type_string);
        Test<Key, Value>(32, begin_bit, end_bit, type_string);
        Test<Key, Value>(3203, begin_bit, end_bit, type_string);
        Test<Key, Value>(320003, begin_bit, end_bit, type_string);
        if (ptx_version > 100)
            Test<Key, Value>(9000003, begin_bit, end_bit, type_string);
        else
            Test<Key, Value>(5000003, begin_bit, end_bit, type_string);
    }
    else
    {
        Test<Key, Value>(num_items, begin_bit, end_bit, type_string);
    }
}


/**
 * Test value type
 */
template <typename Key>
void TestItems(
    int             num_items,
    int             begin_bit,
    int             end_bit,
    char*           type_string)
{
    TestItems<Key, NullType>(num_items, begin_bit, end_bit, type_string);
    TestItems<Key, Key>(num_items, begin_bit, end_bit, type_string);
    TestItems<Key, unsigned int>(num_items, begin_bit, end_bit, type_string);
    TestItems<Key, unsigned long long>(num_items, begin_bit, end_bit, type_string);
    TestItems<Key, TestFoo>(num_items, begin_bit, end_bit, type_string);
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
    int entropy_reduction = 0;

    // Initialize command line
    CommandLineArgs args(argc, argv);
    g_verbose = args.CheckCmdLineFlag("v");
    args.GetCmdLineArgument("n", num_items);
    args.GetCmdLineArgument("i", g_timing_iterations);
    args.GetCmdLineArgument("repeat", g_repeat);
    args.GetCmdLineArgument("bits", g_bits);
    args.GetCmdLineArgument("entropy", entropy_reduction);

    // Print usage
    if (args.CheckCmdLineFlag("help"))
    {
        printf("%s "
            "[--bits=<valid key bits>]"
            "[--n=<input items> "
            "[--i=<timing iterations> "
            "[--device=<device-id>] "
            "[--repeat=<repetitions of entire test suite>]"
            "[--v] "
            "[--entropy=<entropy-reduction factor (default 0)>]"
            "\n", argv[0]);
        exit(0);
    }

    // Initialize device
    CubDebugExit(args.DeviceInit());

    // Get ptx version
    int ptx_version;
    CubDebugExit(PtxVersion(ptx_version));

#ifdef QUICK_TEST

    // Compile/run quick tests
    if (num_items < 0) num_items = 20000000;

    // Compare CUB and thrust on 32b keys-only
    Test<CUB, unsigned int, NullType, false> (num_items, RANDOM, entropy_reduction, 0, g_bits, CUB_TYPE_STRING(unsigned int));
    Test<THRUST, unsigned int, NullType, false> (num_items, RANDOM, entropy_reduction, 0, g_bits, CUB_TYPE_STRING(unsigned int));

    // Compare CUB and thrust on 64b keys-only
    Test<CUB, unsigned long long, NullType, false> (num_items, RANDOM, entropy_reduction, 0, g_bits, CUB_TYPE_STRING(unsigned long long));
    Test<THRUST, unsigned long long, NullType, false> (num_items, RANDOM, entropy_reduction, 0, g_bits, CUB_TYPE_STRING(unsigned long long));


    // Compare CUB and thrust on 32b key-value pairs
    Test<CUB, unsigned int, unsigned int, false> (num_items, RANDOM, entropy_reduction, 0, g_bits, CUB_TYPE_STRING(unsigned int));
    Test<THRUST, unsigned int, unsigned int, false> (num_items, RANDOM, entropy_reduction, 0, g_bits, CUB_TYPE_STRING(unsigned int));

    // Compare CUB and thrust on 64b key-value pairs
    Test<CUB, unsigned long long, unsigned long long, false> (num_items, RANDOM, entropy_reduction, 0, g_bits, CUB_TYPE_STRING(unsigned long long));
    Test<THRUST, unsigned long long, unsigned long long, false> (num_items, RANDOM, entropy_reduction, 0, g_bits, CUB_TYPE_STRING(unsigned long long));

#else

    // Compile/run thorough tests
    for (int i = 0; i <= g_repeat; ++i)
    {
        TestItems<char>                 (num_items, 0, g_bits, CUB_TYPE_STRING(char));
        TestItems<signed char>          (num_items, 0, g_bits, CUB_TYPE_STRING(signed char));
        TestItems<unsigned char>        (num_items, 0, g_bits, CUB_TYPE_STRING(unsigned char));

        TestItems<short>                (num_items, 0, g_bits, CUB_TYPE_STRING(short));
        TestItems<unsigned short>       (num_items, 0, g_bits, CUB_TYPE_STRING(unsigned short));

        TestItems<int>                  (num_items, 0, g_bits, CUB_TYPE_STRING(int));
        TestItems<unsigned int>         (num_items, 0, g_bits, CUB_TYPE_STRING(unsigned int));

        TestItems<long>                 (num_items, 0, g_bits, CUB_TYPE_STRING(long));
        TestItems<unsigned long>        (num_items, 0, g_bits, CUB_TYPE_STRING(unsigned long));

        TestItems<long long>            (num_items, 0, g_bits, CUB_TYPE_STRING(long long));
        TestItems<unsigned long long>   (num_items, 0, g_bits, CUB_TYPE_STRING(unsigned long long));

        TestItems<float>                (num_items, 0, g_bits, CUB_TYPE_STRING(float));

        if (ptx_version > 100)                          // Don't check doubles on PTX100 because they're down-converted
            TestItems<double>               (num_items, 0, g_bits, CUB_TYPE_STRING(double));
    }

#endif

    return 0;
}



