/******************************************************************************
 * Copyright (c) 2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2015, NVIDIA CORPORATION.  All rights reserved.
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
CachingDeviceAllocator  g_allocator(true);

// Dispatch types
enum Backend
{
    CUB,        // CUB method using large temp storage
    CUB_DB,     // CUB method using alternating double-buffer
    THRUST,     // Thrust method
    CDP,        // GPU-based (dynamic parallelism) dispatch to CUB method
};


//---------------------------------------------------------------------
// Dispatch to different DeviceRadixSort entrypoints
//---------------------------------------------------------------------

/**
 * Dispatch to CUB sorting entrypoint (specialized for ascending)
 */
template <typename Key, typename Value>
CUB_RUNTIME_FUNCTION __forceinline__
cudaError_t Dispatch(
    Int2Type<false>     is_descending,
    Int2Type<CUB>       dispatch_to,
    int                 *d_selector,
    size_t              *d_temp_storage_bytes,
    cudaError_t         *d_cdp_error,

    void*               d_temp_storage,
    size_t&             temp_storage_bytes,
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
 * Dispatch to CUB_DB sorting entrypoint (specialized for ascending)
 */
template <typename Key, typename Value>
CUB_RUNTIME_FUNCTION __forceinline__
cudaError_t Dispatch(
    Int2Type<false>     is_descending,
    Int2Type<CUB_DB>    dispatch_to,
    int                 *d_selector,
    size_t              *d_temp_storage_bytes,
    cudaError_t         *d_cdp_error,

    void*               d_temp_storage,
    size_t&             temp_storage_bytes,
    DoubleBuffer<Key>   &d_keys,
    DoubleBuffer<Value> &d_values,
    int                 num_items,
    int                 begin_bit,
    int                 end_bit,
    cudaStream_t        stream,
    bool                debug_synchronous)
{
    cudaError_t retval = DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
        d_keys.Current(), d_keys.Alternate(),
        d_values.Current(), d_values.Alternate(),
        num_items, begin_bit, end_bit, stream, debug_synchronous);

    d_keys.selector ^= 1;
    d_values.selector ^= 1;
    return retval;
}

/**
 * Dispatch to CUB sorting entrypoint (specialized for descending)
 */
template <typename Key, typename Value>
CUB_RUNTIME_FUNCTION __forceinline__
cudaError_t Dispatch(
    Int2Type<true>      is_descending,
    Int2Type<CUB>       dispatch_to,
    int                 *d_selector,
    size_t              *d_temp_storage_bytes,
    cudaError_t         *d_cdp_error,

    void*               d_temp_storage,
    size_t&             temp_storage_bytes,
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
 * Dispatch to CUB_DB sorting entrypoint (specialized for descending)
 */
template <typename Key, typename Value>
CUB_RUNTIME_FUNCTION __forceinline__
cudaError_t Dispatch(
    Int2Type<true>      is_descending,
    Int2Type<CUB_DB>       dispatch_to,
    int                 *d_selector,
    size_t              *d_temp_storage_bytes,
    cudaError_t         *d_cdp_error,

    void*               d_temp_storage,
    size_t&             temp_storage_bytes,
    DoubleBuffer<Key>   &d_keys,
    DoubleBuffer<Value> &d_values,
    int                 num_items,
    int                 begin_bit,
    int                 end_bit,
    cudaStream_t        stream,
    bool                debug_synchronous)
{
    cudaError_t retval = DeviceRadixSort::SortPairsDescending(d_temp_storage, temp_storage_bytes,
        d_keys.Current(), d_keys.Alternate(),
        d_values.Current(), d_values.Alternate(),
        num_items, begin_bit, end_bit, stream, debug_synchronous);

    d_keys.selector ^= 1;
    d_values.selector ^= 1;
    return retval;
}

/**
 * Dispatch keys-only to Thrust sorting entrypoint
 */
template <int IS_DESCENDING, typename Key>
cudaError_t Dispatch(
    Int2Type<IS_DESCENDING>    is_descending,
    Int2Type<THRUST>        dispatch_to,
    int                     *d_selector,
    size_t                  *d_temp_storage_bytes,
    cudaError_t             *d_cdp_error,
    void*               d_temp_storage,
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

        if (IS_DESCENDING) thrust::reverse(d_keys_wrapper, d_keys_wrapper + num_items);
        thrust::sort(d_keys_wrapper, d_keys_wrapper + num_items);
        if (IS_DESCENDING) thrust::reverse(d_keys_wrapper, d_keys_wrapper + num_items);
    }

    return cudaSuccess;
}


/**
 * Dispatch key-value pairs to Thrust sorting entrypoint
 */
template <int IS_DESCENDING, typename Key, typename Value>
cudaError_t Dispatch(
    Int2Type<IS_DESCENDING>    is_descending,
    Int2Type<THRUST>        dispatch_to,
    int                     *d_selector,
    size_t                  *d_temp_storage_bytes,
    cudaError_t             *d_cdp_error,
    void*               d_temp_storage,
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

        if (IS_DESCENDING) {
            thrust::reverse(d_keys_wrapper, d_keys_wrapper + num_items);
            thrust::reverse(d_values_wrapper, d_values_wrapper + num_items);
        }

        thrust::sort_by_key(d_keys_wrapper, d_keys_wrapper + num_items, d_values_wrapper);

        if (IS_DESCENDING) {
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
template <int IS_DESCENDING, typename Key, typename Value>
__global__ void CnpDispatchKernel(
    Int2Type<IS_DESCENDING>    is_descending,
    int                     *d_selector,
    size_t                  *d_temp_storage_bytes,
    cudaError_t             *d_cdp_error,

    void*               d_temp_storage,
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
template <int IS_DESCENDING, typename Key, typename Value>
cudaError_t Dispatch(
    Int2Type<IS_DESCENDING>    is_descending,
    Int2Type<CDP>           dispatch_to,
    int                     *d_selector,
    size_t                  *d_temp_storage_bytes,
    cudaError_t             *d_cdp_error,

    void*               d_temp_storage,
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
 * Initialize key data
 */
template <typename Key>
void InitializeKeyBits(
    GenMode         gen_mode,
    Key             *h_keys,
    int             num_items,
    int             entropy_reduction)
{
    for (int i = 0; i < num_items; ++i)
    {
        if (gen_mode == RANDOM) {
            RandomBits(h_keys[i], entropy_reduction);
        } else if (gen_mode == UNIFORM) {
            h_keys[i] = 1;
        } else {
            h_keys[i] = i;
        }
    }
}


/**
 * Initialize solution
 */
template <bool IS_DESCENDING, typename Key>
void InitializeSolution(
    Key     *h_keys,
    int     num_items,
    int     begin_bit,
    int     end_bit,
    int     *&h_reference_ranks,
    Key     *&h_reference_keys)
{
    Pair<Key, int> *h_pairs = new Pair<Key, int>[num_items];

    int num_bits = end_bit - begin_bit;
    for (int i = 0; i < num_items; ++i)
    {

        // Mask off unwanted portions
        if (num_bits < sizeof(Key) * 8)
        {
            unsigned long long base = 0;
            memcpy(&base, &h_keys[i], sizeof(Key));
            base &= ((1ull << num_bits) - 1) << begin_bit;
            memcpy(&h_pairs[i].key, &base, sizeof(Key));
        }
        else
        {
            h_pairs[i].key = h_keys[i];
        }

        h_pairs[i].value = i;
    }

    printf("\nSorting reference solution on CPU..."); fflush(stdout);
    if (IS_DESCENDING) std::reverse(h_pairs, h_pairs + num_items);
    std::stable_sort(h_pairs, h_pairs + num_items);
    if (IS_DESCENDING) std::reverse(h_pairs, h_pairs + num_items);
    printf(" Done.\n"); fflush(stdout);

    h_reference_ranks  = new int[num_items];
    h_reference_keys   = new Key[num_items];

    for (int i = 0; i < num_items; ++i)
    {
        h_reference_ranks[i]    = h_pairs[i].value;
        h_reference_keys[i]     = h_keys[h_pairs[i].value];
    }

    delete[] h_pairs;
}



/**
 * Test DeviceRadixSort
 */
template <
    Backend     BACKEND,
    bool        IS_DESCENDING,
    typename    Key,
    typename    Value>
void Test(
    Key         *h_keys,
    Value       *h_values,
    int         num_items,
    int         begin_bit,
    int         end_bit,
    Key         *h_reference_keys,
    Value       *h_reference_values)
{
    const bool KEYS_ONLY = Equals<Value, NullType>::VALUE;

    printf("%s %s cub::DeviceRadixSort %d items, %d-byte keys %d-byte values, descending %d, begin_bit %d, end_bit %d\n",
        (BACKEND == CUB_DB) ? "CUB_DB" : (BACKEND == CDP) ? "CDP CUB" : (BACKEND == THRUST) ? "Thrust" : "CUB",
        (KEYS_ONLY) ? "keys-only" : "key-value",
        num_items, (int) sizeof(Key), (KEYS_ONLY) ? 0 : (int) sizeof(Value),
        IS_DESCENDING, begin_bit, end_bit);
    fflush(stdout);

    if (g_verbose)
    {
        printf("Input keys:\n");
        DisplayResults(h_keys, num_items);
        printf("\n\n");
    }

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
    CubDebugExit(Dispatch(Int2Type<IS_DESCENDING>(), Int2Type<BACKEND>(), d_selector, d_temp_storage_bytes, d_cdp_error, d_temp_storage, temp_storage_bytes, d_keys, d_values, num_items, begin_bit, end_bit, 0, true));
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
    CubDebugExit(Dispatch(Int2Type<IS_DESCENDING>(), Int2Type<BACKEND>(), d_selector, d_temp_storage_bytes, d_cdp_error, d_temp_storage, temp_storage_bytes, d_keys, d_values, num_items, begin_bit, end_bit, 0, true));

    // Flush any stdout/stderr
    fflush(stdout);
    fflush(stderr);

    // Check for correctness (and display results, if specified)
    printf("Warmup done.  Checking results:\n"); fflush(stdout);
    int compare = CompareDeviceResults(h_reference_keys, d_keys.Current(), num_items, true, g_verbose);
    printf("\t Compare keys (selector %d): %s ", d_keys.selector, compare ? "FAIL" : "PASS"); fflush(stdout);
    if (!KEYS_ONLY)
    {
        int values_compare = CompareDeviceResults(h_reference_values, d_values.Current(), num_items, true, g_verbose);
        compare |= values_compare;
        printf("\t Compare values (selector %d): %s ", d_values.selector, values_compare ? "FAIL" : "PASS"); fflush(stdout);
    }

    // Performance
    if (g_timing_iterations)
        printf("\nPerforming timing iterations:\n"); fflush(stdout);

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
        CubDebugExit(Dispatch(Int2Type<IS_DESCENDING>(), Int2Type<BACKEND>(), d_selector, d_temp_storage_bytes, d_cdp_error, d_temp_storage, temp_storage_bytes, d_keys, d_values, num_items, begin_bit, end_bit, 0, false));
        gpu_timer.Stop();
        elapsed_millis += gpu_timer.ElapsedMillis();
    }

    // Display performance
    if (g_timing_iterations > 0)
    {
        float avg_millis = elapsed_millis / g_timing_iterations;
        float giga_rate = float(num_items) / avg_millis / 1000.0 / 1000.0;
        float giga_bandwidth = (KEYS_ONLY) ?
            giga_rate * sizeof(Key) * 2 :
            giga_rate * (sizeof(Key) + sizeof(Value)) * 2;
        printf("\n%.3f elapsed ms, %.3f avg ms, %.3f billion items/s, %.3f logical GB/s", elapsed_millis, avg_millis, giga_rate, giga_bandwidth);
    }

    printf("\n\n");

    // Cleanup
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
 * Test backend
 */
template <bool IS_DESCENDING, typename Key, typename Value>
void TestBackend(
    Key     *h_keys,
    int     num_items,
    int     begin_bit,
    int     end_bit,
    Key     *h_reference_keys,
    int     *h_reference_ranks)
{
    const bool KEYS_ONLY = Equals<Value, NullType>::VALUE;

    Value *h_values             = NULL;
    Value *h_reference_values   = NULL;

    if (!KEYS_ONLY)
    {
        h_values            = new Value[num_items];
        h_reference_values  = new Value[num_items];

        for (int i = 0; i < num_items; ++i)
        {
            InitValue(INTEGER_SEED, h_values[i], i);
            InitValue(INTEGER_SEED, h_reference_values[i], h_reference_ranks[i]);
        }
    }

    Test<CUB, IS_DESCENDING>(h_keys, h_values, num_items, begin_bit, end_bit, h_reference_keys, h_reference_values);
    Test<CUB_DB, IS_DESCENDING>(h_keys, h_values, num_items, begin_bit, end_bit, h_reference_keys, h_reference_values);

#ifdef CUB_CDP
    Test<CDP, IS_DESCENDING>(h_keys, h_values, num_items, begin_bit, end_bit, h_reference_keys, h_reference_values);
#endif

    if (h_values) delete[] h_values;
    if (h_reference_values) delete[] h_reference_values;
}




/**
 * Test value type
 */
template <bool IS_DESCENDING, typename Key>
void TestValueTypes(
    Key     *h_keys,
    int     num_items,
    int     begin_bit,
    int     end_bit)
{
    // Initialize the solution

    int *h_reference_ranks = NULL;
    Key *h_reference_keys = NULL;
    InitializeSolution<IS_DESCENDING>(h_keys, num_items, begin_bit, end_bit, h_reference_ranks, h_reference_keys);

    // Test value types

    TestBackend<IS_DESCENDING, Key, NullType>              (h_keys, num_items, begin_bit, end_bit, h_reference_keys, h_reference_ranks);

    TestBackend<IS_DESCENDING, Key, Key>                   (h_keys, num_items, begin_bit, end_bit, h_reference_keys, h_reference_ranks);

    if (!Equals<Key, unsigned int>::VALUE)
        TestBackend<IS_DESCENDING, Key, unsigned int>      (h_keys, num_items, begin_bit, end_bit, h_reference_keys, h_reference_ranks);

    if (!Equals<Key, unsigned long long>::VALUE)
        TestBackend<IS_DESCENDING, Key, unsigned long long>(h_keys, num_items, begin_bit, end_bit, h_reference_keys, h_reference_ranks);

    TestBackend<IS_DESCENDING, Key, TestFoo>               (h_keys, num_items, begin_bit, end_bit, h_reference_keys, h_reference_ranks);

    // Cleanup

    if (h_reference_ranks) delete[] h_reference_ranks;
    if (h_reference_keys) delete[] h_reference_keys;
}



/**
 * Test ascending/descending
 */
template <typename Key>
void TestDirection(
    Key     *h_keys,
    int     num_items,
    int     begin_bit,
    int     end_bit)
{
    TestValueTypes<true>(h_keys, num_items, begin_bit, end_bit);
    TestValueTypes<false>(h_keys, num_items, begin_bit, end_bit);
}


/**
 * Test different bit ranges
 */
template <typename Key>
void TestBits(
    Key *h_keys,
    int num_items)
{
    if (Traits<Key>::CATEGORY == UNSIGNED_INTEGER)
    {
        // Don't test partial-word sorting for fp or signed types (the bit-flipping techniques get in the way)
        int mid_bit = sizeof(Key) * 4;
        printf("Testing key bits [%d,%d)\n", mid_bit - 1, mid_bit); fflush(stdout);
        TestDirection(h_keys, num_items, mid_bit - 1, mid_bit);
    }

    printf("Testing key bits [%d,%d)\n", 0, int(sizeof(Key)) * 8); fflush(stdout);
    TestDirection(h_keys, num_items, 0, sizeof(Key) * 8);
}


/**
 * Test different (sub)lengths
 */
template <typename Key>
void TestSizes(
    Key *h_keys,
    int max_items)
{
    while (true)
    {
        TestBits(h_keys, max_items);

        if (max_items == 1)
            break;

        max_items = (max_items + 31) / 32;
    }
}


/**
 * Test key sampling distributions
 */
template <typename Key>
void TestGen(
    int             max_items,
    const char*     type_string)
{
    if (max_items < 0)
    {
        int ptx_version;
        CubDebugExit(PtxVersion(ptx_version));
        max_items = (ptx_version > 100) ? 9000003 : max_items = 5000003;
    }

    Key *h_keys = new Key[max_items];

    for (int entropy_reduction = 0; entropy_reduction <= 6; entropy_reduction += 3)
    {
        printf("\nTesting random %s keys with entropy reduction factor %d\n", type_string, entropy_reduction); fflush(stdout);
        InitializeKeyBits(RANDOM, h_keys, max_items, entropy_reduction);
        TestSizes(h_keys, max_items);
    }

    printf("\nTesting uniform %s keys\n", type_string); fflush(stdout);
    InitializeKeyBits(UNIFORM, h_keys, max_items, 0);
    TestSizes(h_keys, max_items);

    printf("\nTesting natural number %s keys\n", type_string); fflush(stdout);
    InitializeKeyBits(INTEGER_SEED, h_keys, max_items, 0);
    TestSizes(h_keys, max_items);

    if (h_keys) delete[] h_keys;
}



template <
    Backend     BACKEND,
    typename    Key,
    typename    Value,
    bool        IS_DESCENDING>
void Test(
    int         num_items,
    GenMode     gen_mode,
    int         entropy_reduction,
    int         begin_bit,
    int         end_bit,
    char        *type_string)
{
    const bool KEYS_ONLY = Equals<Value, NullType>::VALUE;

    Key     *h_keys             = new Key[num_items];
    int     *h_reference_ranks  = NULL;
    Key     *h_reference_keys   = NULL;
    Value   *h_values           = NULL;
    Value   *h_reference_values = NULL;

    if (end_bit < 0)
        end_bit = sizeof(Key) * 8;

    InitializeKeyBits(gen_mode, h_keys, num_items, entropy_reduction);
    InitializeSolution<IS_DESCENDING>(h_keys, num_items, begin_bit, end_bit, h_reference_ranks, h_reference_keys);

    if (!KEYS_ONLY)
    {
        h_values            = new Value[num_items];
        h_reference_values  = new Value[num_items];

        for (int i = 0; i < num_items; ++i)
        {
            InitValue(INTEGER_SEED, h_values[i], i);
            InitValue(INTEGER_SEED, h_reference_values[i], h_reference_ranks[i]);
        }
    }
    if (h_reference_ranks) delete[] h_reference_ranks;

    printf("\nTesting bits [%d,%d) of %s keys with gen-mode %d\n", begin_bit, end_bit, type_string, gen_mode); fflush(stdout);
    Test<BACKEND, IS_DESCENDING>(h_keys, h_values, num_items, begin_bit, end_bit, h_reference_keys, h_reference_values);

    if (h_keys) delete[] h_keys;
    if (h_reference_keys) delete[] h_reference_keys;
    if (h_values) delete[] h_values;
    if (h_reference_values) delete[] h_reference_values;
}



//---------------------------------------------------------------------
// Main
//---------------------------------------------------------------------

/**
 * Main
 */
int main(int argc, char** argv)
{
    int bits = -1;
    int num_items = -1;
    int entropy_reduction = 0;

    // Initialize command line
    CommandLineArgs args(argc, argv);
    g_verbose = args.CheckCmdLineFlag("v");
    args.GetCmdLineArgument("n", num_items);
    args.GetCmdLineArgument("i", g_timing_iterations);
    args.GetCmdLineArgument("repeat", g_repeat);
    args.GetCmdLineArgument("bits", bits);
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

#ifdef QUICKER_TEST

    // Compile/run basic CUB test
    if (num_items < 0) num_items = 32000000;

    Test<CUB, unsigned int, NullType, false> (num_items, RANDOM, entropy_reduction, 0, bits, CUB_TYPE_STRING(unsigned int));
    Test<CUB, unsigned long long, NullType, false> (num_items, RANDOM, entropy_reduction, 0, bits, CUB_TYPE_STRING(unsigned int));

    Test<CUB, unsigned int, unsigned int, false> (num_items, RANDOM, entropy_reduction, 0, bits, CUB_TYPE_STRING(unsigned int));
    Test<CUB, unsigned long long, unsigned int, false> (num_items, RANDOM, entropy_reduction, 0, bits, CUB_TYPE_STRING(unsigned int));

#elif defined(QUICK_TEST)

    // Compile/run quick tests
    if (num_items < 0) num_items = 20000000;

    // Compare CUB and thrust on 32b keys-only
    Test<CUB, unsigned int, NullType, false> (num_items, RANDOM, entropy_reduction, 0, bits, CUB_TYPE_STRING(unsigned int));
    Test<THRUST, unsigned int, NullType, false> (num_items, RANDOM, entropy_reduction, 0, bits, CUB_TYPE_STRING(unsigned int));

    // Compare CUB and thrust on 64b keys-only
    Test<CUB, unsigned long long, NullType, false> (num_items, RANDOM, entropy_reduction, 0, bits, CUB_TYPE_STRING(unsigned long long));
    Test<THRUST, unsigned long long, NullType, false> (num_items, RANDOM, entropy_reduction, 0, bits, CUB_TYPE_STRING(unsigned long long));


    // Compare CUB and thrust on 32b key-value pairs
    Test<CUB, unsigned int, unsigned int, false> (num_items, RANDOM, entropy_reduction, 0, bits, CUB_TYPE_STRING(unsigned int));
    Test<THRUST, unsigned int, unsigned int, false> (num_items, RANDOM, entropy_reduction, 0, bits, CUB_TYPE_STRING(unsigned int));

    // Compare CUB and thrust on 64b key-value pairs
    Test<CUB, unsigned long long, unsigned long long, false> (num_items, RANDOM, entropy_reduction, 0, bits, CUB_TYPE_STRING(unsigned long long));
    Test<THRUST, unsigned long long, unsigned long long, false> (num_items, RANDOM, entropy_reduction, 0, bits, CUB_TYPE_STRING(unsigned long long));


#else

    // Compile/run thorough tests
    for (int i = 0; i <= g_repeat; ++i)
    {
        TestGen<char>                 (num_items, CUB_TYPE_STRING(char));
        TestGen<signed char>          (num_items, CUB_TYPE_STRING(signed char));
        TestGen<unsigned char>        (num_items, CUB_TYPE_STRING(unsigned char));

        TestGen<short>                (num_items, CUB_TYPE_STRING(short));
        TestGen<unsigned short>       (num_items, CUB_TYPE_STRING(unsigned short));

        TestGen<int>                  (num_items, CUB_TYPE_STRING(int));
        TestGen<unsigned int>         (num_items, CUB_TYPE_STRING(unsigned int));

        TestGen<long>                 (num_items, CUB_TYPE_STRING(long));
        TestGen<unsigned long>        (num_items, CUB_TYPE_STRING(unsigned long));

        TestGen<long long>            (num_items, CUB_TYPE_STRING(long long));
        TestGen<unsigned long long>   (num_items, CUB_TYPE_STRING(unsigned long long));

        TestGen<float>                (num_items, CUB_TYPE_STRING(float));

        if (ptx_version > 120)                          // Don't check doubles on PTX120 or below because they're down-converted
            TestGen<double>               (num_items, CUB_TYPE_STRING(double));
    }

#endif

    return 0;
}



