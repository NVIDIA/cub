/******************************************************************************
 * Copyright (c) 2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2018, NVIDIA CORPORATION.  All rights reserved.
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
#include <random>
#include <typeinfo>
#include <vector>

#if (__CUDACC_VER_MAJOR__ >= 9 || CUDA_VERSION >= 9000) && !_NVHPC_CUDA
    #include <cuda_fp16.h>
#endif

#if (__CUDACC_VER_MAJOR__ >= 11 || CUDA_VERSION >= 11000) && !_NVHPC_CUDA
    #include <cuda_bf16.h>
#endif

#include <cub/util_allocator.cuh>
#include <cub/util_math.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_segmented_radix_sort.cuh>
#include <cub/iterator/transform_input_iterator.cuh>

#include "test_util.h"

#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/reverse.h>

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
    CUB,                        // CUB method (allows overwriting of input)
    CUB_NO_OVERWRITE,           // CUB method (disallows overwriting of input)

    CUB_SEGMENTED,              // CUB method (allows overwriting of input)
    CUB_SEGMENTED_NO_OVERWRITE, // CUB method (disallows overwriting of input)

    THRUST,                     // Thrust method
    CDP,                        // GPU-based (dynamic parallelism) dispatch to CUB method
};


//---------------------------------------------------------------------
// Dispatch to different DeviceRadixSort entrypoints
//---------------------------------------------------------------------

/**
 * Dispatch to CUB sorting entrypoint (specialized for ascending)
 */
template <typename KeyT, typename ValueT, typename BeginOffsetIteratorT, typename EndOffsetIteratorT>
CUB_RUNTIME_FUNCTION
__forceinline__
cudaError_t Dispatch(
    Int2Type<false>         /*is_descending*/,
    Int2Type<CUB>           /*dispatch_to*/,
    int                     */*d_selector*/,
    size_t                  */*d_temp_storage_bytes*/,
    cudaError_t             */*d_cdp_error*/,

    void*                   d_temp_storage,
    size_t&                 temp_storage_bytes,
    DoubleBuffer<KeyT>      &d_keys,
    DoubleBuffer<ValueT>    &d_values,
    std::size_t             num_items,
    int                     /*num_segments*/,
    BeginOffsetIteratorT    /*d_segment_begin_offsets*/,
    EndOffsetIteratorT      /*d_segment_end_offsets*/,
    int                     begin_bit,
    int                     end_bit,
    cudaStream_t            stream,
    bool                    debug_synchronous)
{
    return DeviceRadixSort::SortPairs(
        d_temp_storage, temp_storage_bytes,
        d_keys, d_values,
        static_cast<int>(num_items), begin_bit, end_bit, stream, debug_synchronous);
}

/**
 * Dispatch to CUB_NO_OVERWRITE sorting entrypoint (specialized for ascending)
 */
template <typename KeyT, typename ValueT, typename BeginOffsetIteratorT, typename EndOffsetIteratorT>
CUB_RUNTIME_FUNCTION
__forceinline__
cudaError_t Dispatch(
    Int2Type<false>             /*is_descending*/,
    Int2Type<CUB_NO_OVERWRITE>  /*dispatch_to*/,
    int                         */*d_selector*/,
    size_t                      */*d_temp_storage_bytes*/,
    cudaError_t                 */*d_cdp_error*/,

    void*                   d_temp_storage,
    size_t&                 temp_storage_bytes,
    DoubleBuffer<KeyT>      &d_keys,
    DoubleBuffer<ValueT>    &d_values,
    std::size_t             num_items,
    int                     /*num_segments*/,
    BeginOffsetIteratorT    /*d_segment_begin_offsets*/,
    EndOffsetIteratorT      /*d_segment_end_offsets*/,
    int                     begin_bit,
    int                     end_bit,
    cudaStream_t            stream,
    bool                    debug_synchronous)
{
    KeyT      const *const_keys_itr     = d_keys.Current();
    ValueT    const *const_values_itr   = d_values.Current();

    cudaError_t retval = DeviceRadixSort::SortPairs(
        d_temp_storage, temp_storage_bytes,
        const_keys_itr, d_keys.Alternate(), const_values_itr, d_values.Alternate(),
        static_cast<int>(num_items), begin_bit, end_bit, stream, debug_synchronous);

    d_keys.selector ^= 1;
    d_values.selector ^= 1;
    return retval;
}

/**
 * Dispatch to CUB sorting entrypoint (specialized for descending)
 */
template <typename KeyT, typename ValueT, typename BeginOffsetIteratorT, typename EndOffsetIteratorT>
CUB_RUNTIME_FUNCTION
__forceinline__
cudaError_t Dispatch(
    Int2Type<true>          /*is_descending*/,
    Int2Type<CUB>           /*dispatch_to*/,
    int                     */*d_selector*/,
    size_t                  */*d_temp_storage_bytes*/,
    cudaError_t             */*d_cdp_error*/,

    void*                   d_temp_storage,
    size_t&                 temp_storage_bytes,
    DoubleBuffer<KeyT>      &d_keys,
    DoubleBuffer<ValueT>    &d_values,
    std::size_t             num_items,
    int                     /*num_segments*/,
    BeginOffsetIteratorT    /*d_segment_begin_offsets*/,
    EndOffsetIteratorT      /*d_segment_end_offsets*/,
    int                     begin_bit,
    int                     end_bit,
    cudaStream_t            stream,
    bool                    debug_synchronous)
{
    return DeviceRadixSort::SortPairsDescending(
        d_temp_storage, temp_storage_bytes,
        d_keys, d_values, static_cast<int>(num_items),
        begin_bit, end_bit, stream, debug_synchronous);
}


/**
 * Dispatch to CUB_NO_OVERWRITE sorting entrypoint (specialized for descending)
 */
template <typename KeyT, typename ValueT, typename BeginOffsetIteratorT, typename EndOffsetIteratorT>
CUB_RUNTIME_FUNCTION
__forceinline__
cudaError_t Dispatch(
    Int2Type<true>              /*is_descending*/,
    Int2Type<CUB_NO_OVERWRITE>  /*dispatch_to*/,
    int                         */*d_selector*/,
    size_t                      */*d_temp_storage_bytes*/,
    cudaError_t                 */*d_cdp_error*/,

    void*                   d_temp_storage,
    size_t&                 temp_storage_bytes,
    DoubleBuffer<KeyT>      &d_keys,
    DoubleBuffer<ValueT>    &d_values,
    std::size_t             num_items,
    int                     /*num_segments*/,
    BeginOffsetIteratorT    /*d_segment_begin_offsets*/,
    EndOffsetIteratorT      /*d_segment_end_offsets*/,
    int                     begin_bit,
    int                     end_bit,
    cudaStream_t            stream,
    bool                    debug_synchronous)
{
    KeyT      const *const_keys_itr     = d_keys.Current();
    ValueT    const *const_values_itr   = d_values.Current();

    cudaError_t retval = DeviceRadixSort::SortPairsDescending(
        d_temp_storage, temp_storage_bytes,
        const_keys_itr, d_keys.Alternate(), const_values_itr, d_values.Alternate(),
        static_cast<int>(num_items), begin_bit, end_bit, stream, debug_synchronous);

    d_keys.selector ^= 1;
    d_values.selector ^= 1;
    return retval;
}

//---------------------------------------------------------------------
// Dispatch to different DeviceRadixSort entrypoints
//---------------------------------------------------------------------

/**
 * Dispatch to CUB_SEGMENTED sorting entrypoint (specialized for ascending)
 */
template <typename KeyT, typename ValueT, typename BeginOffsetIteratorT, typename EndOffsetIteratorT>
CUB_RUNTIME_FUNCTION
__forceinline__
cudaError_t Dispatch(
    Int2Type<false>         /*is_descending*/,
    Int2Type<CUB_SEGMENTED> /*dispatch_to*/,
    int                     */*d_selector*/,
    size_t                  */*d_temp_storage_bytes*/,
    cudaError_t             */*d_cdp_error*/,

    void*                   d_temp_storage,
    size_t&                 temp_storage_bytes,
    DoubleBuffer<KeyT>      &d_keys,
    DoubleBuffer<ValueT>    &d_values,
    std::size_t             num_items,
    int                     num_segments,
    BeginOffsetIteratorT    d_segment_begin_offsets,
    EndOffsetIteratorT      d_segment_end_offsets,
    int                     begin_bit,
    int                     end_bit,
    cudaStream_t            stream,
    bool                    debug_synchronous)
{
    return DeviceSegmentedRadixSort::SortPairs(
        d_temp_storage, temp_storage_bytes,
        d_keys, d_values, static_cast<int>(num_items),
        num_segments, d_segment_begin_offsets, d_segment_end_offsets,
        begin_bit, end_bit, stream, debug_synchronous);
}

/**
 * Dispatch to CUB_SEGMENTED_NO_OVERWRITE sorting entrypoint (specialized for ascending)
 */
template <typename KeyT, typename ValueT, typename BeginOffsetIteratorT, typename EndOffsetIteratorT>
CUB_RUNTIME_FUNCTION
__forceinline__
cudaError_t Dispatch(
    Int2Type<false>                         /*is_descending*/,
    Int2Type<CUB_SEGMENTED_NO_OVERWRITE>    /*dispatch_to*/,
    int                                     */*d_selector*/,
    size_t                                  */*d_temp_storage_bytes*/,
    cudaError_t                             */*d_cdp_error*/,

    void*                   d_temp_storage,
    size_t&                 temp_storage_bytes,
    DoubleBuffer<KeyT>      &d_keys,
    DoubleBuffer<ValueT>    &d_values,
    std::size_t             num_items,
    int                     num_segments,
    BeginOffsetIteratorT    d_segment_begin_offsets,
    EndOffsetIteratorT      d_segment_end_offsets,
    int                     begin_bit,
    int                     end_bit,
    cudaStream_t            stream,
    bool                    debug_synchronous)
{
    KeyT      const *const_keys_itr     = d_keys.Current();
    ValueT    const *const_values_itr   = d_values.Current();

    cudaError_t retval = DeviceSegmentedRadixSort::SortPairs(
        d_temp_storage, temp_storage_bytes,
        const_keys_itr, d_keys.Alternate(), const_values_itr, d_values.Alternate(),
        static_cast<int>(num_items), num_segments, d_segment_begin_offsets, d_segment_end_offsets,
        begin_bit, end_bit, stream, debug_synchronous);

    d_keys.selector ^= 1;
    d_values.selector ^= 1;
    return retval;
}


/**
 * Dispatch to CUB_SEGMENTED sorting entrypoint (specialized for descending)
 */
template <typename KeyT, typename ValueT, typename BeginOffsetIteratorT, typename EndOffsetIteratorT>
CUB_RUNTIME_FUNCTION
__forceinline__
cudaError_t Dispatch(
    Int2Type<true>          /*is_descending*/,
    Int2Type<CUB_SEGMENTED> /*dispatch_to*/,
    int                     */*d_selector*/,
    size_t                  */*d_temp_storage_bytes*/,
    cudaError_t             */*d_cdp_error*/,

    void*                   d_temp_storage,
    size_t&                 temp_storage_bytes,
    DoubleBuffer<KeyT>      &d_keys,
    DoubleBuffer<ValueT>    &d_values,
    std::size_t             num_items,
    int                     num_segments,
    BeginOffsetIteratorT    d_segment_begin_offsets,
    EndOffsetIteratorT      d_segment_end_offsets,
    int                     begin_bit,
    int                     end_bit,
    cudaStream_t            stream,
    bool                    debug_synchronous)
{
    return DeviceSegmentedRadixSort::SortPairsDescending(
        d_temp_storage, temp_storage_bytes,
        d_keys, d_values, static_cast<int>(num_items),
        num_segments, d_segment_begin_offsets, d_segment_end_offsets,
        begin_bit, end_bit, stream, debug_synchronous);
}

/**
 * Dispatch to CUB_SEGMENTED_NO_OVERWRITE sorting entrypoint (specialized for descending)
 */
template <typename KeyT, typename ValueT, typename BeginOffsetIteratorT, typename EndOffsetIteratorT>
CUB_RUNTIME_FUNCTION
__forceinline__
cudaError_t Dispatch(
    Int2Type<true>                          /*is_descending*/,
    Int2Type<CUB_SEGMENTED_NO_OVERWRITE>    /*dispatch_to*/,
    int                                     */*d_selector*/,
    size_t                                  */*d_temp_storage_bytes*/,
    cudaError_t                             */*d_cdp_error*/,

    void*                   d_temp_storage,
    size_t&                 temp_storage_bytes,
    DoubleBuffer<KeyT>      &d_keys,
    DoubleBuffer<ValueT>    &d_values,
    std::size_t             num_items,
    int                     num_segments,
    BeginOffsetIteratorT    d_segment_begin_offsets,
    EndOffsetIteratorT      d_segment_end_offsets,
    int                     begin_bit,
    int                     end_bit,
    cudaStream_t            stream,
    bool                    debug_synchronous)
{
    KeyT      const *const_keys_itr     = d_keys.Current();
    ValueT    const *const_values_itr   = d_values.Current();

    cudaError_t retval = DeviceSegmentedRadixSort::SortPairsDescending(
        d_temp_storage, temp_storage_bytes,
        const_keys_itr, d_keys.Alternate(), const_values_itr, d_values.Alternate(),
        static_cast<int>(num_items), num_segments, d_segment_begin_offsets, d_segment_end_offsets,
        begin_bit, end_bit, stream, debug_synchronous);

    d_keys.selector ^= 1;
    d_values.selector ^= 1;
    return retval;
}


//---------------------------------------------------------------------
// Dispatch to different Thrust entrypoints
//---------------------------------------------------------------------

/**
 * Dispatch keys-only to Thrust sorting entrypoint
 */
template <int IS_DESCENDING, typename KeyT, typename BeginOffsetIteratorT, typename EndOffsetIteratorT>
cudaError_t Dispatch(
    Int2Type<IS_DESCENDING> /*is_descending*/,
    Int2Type<THRUST>        /*dispatch_to*/,
    int                     */*d_selector*/,
    size_t                  */*d_temp_storage_bytes*/,
    cudaError_t             */*d_cdp_error*/,

    void                    *d_temp_storage,
    size_t                  &temp_storage_bytes,
    DoubleBuffer<KeyT>      &d_keys,
    DoubleBuffer<NullType>  &/*d_values*/,
    std::size_t             num_items,
    int                     /*num_segments*/,
    BeginOffsetIteratorT    /*d_segment_begin_offsets*/,
    EndOffsetIteratorT      /*d_segment_end_offsets*/,
    int                     /*begin_bit*/,
    int                     /*end_bit*/,
    cudaStream_t            /*stream*/,
    bool                    /*debug_synchronous*/)
{

    if (d_temp_storage == 0)
    {
        temp_storage_bytes = 1;
    }
    else
    {
        THRUST_NS_QUALIFIER::device_ptr<KeyT> d_keys_wrapper(d_keys.Current());

        if (IS_DESCENDING) THRUST_NS_QUALIFIER::reverse(d_keys_wrapper, d_keys_wrapper + num_items);
        THRUST_NS_QUALIFIER::sort(d_keys_wrapper, d_keys_wrapper + num_items);
        if (IS_DESCENDING) THRUST_NS_QUALIFIER::reverse(d_keys_wrapper, d_keys_wrapper + num_items);
    }

    return cudaSuccess;
}


/**
 * Dispatch key-value pairs to Thrust sorting entrypoint
 */
template <int IS_DESCENDING, typename KeyT, typename ValueT, typename BeginOffsetIteratorT, typename EndOffsetIteratorT>
cudaError_t Dispatch(
    Int2Type<IS_DESCENDING> /*is_descending*/,
    Int2Type<THRUST>        /*dispatch_to*/,
    int                     */*d_selector*/,
    size_t                  */*d_temp_storage_bytes*/,
    cudaError_t             */*d_cdp_error*/,

    void                    *d_temp_storage,
    size_t                  &temp_storage_bytes,
    DoubleBuffer<KeyT>      &d_keys,
    DoubleBuffer<ValueT>    &d_values,
    std::size_t             num_items,
    int                     /*num_segments*/,
    BeginOffsetIteratorT    /*d_segment_begin_offsets*/,
    EndOffsetIteratorT      /*d_segment_end_offsets*/,
    int                     /*begin_bit*/,
    int                     /*end_bit*/,
    cudaStream_t            /*stream*/,
    bool                    /*debug_synchronous*/)
{

    if (d_temp_storage == 0)
    {
        temp_storage_bytes = 1;
    }
    else
    {
        THRUST_NS_QUALIFIER::device_ptr<KeyT>     d_keys_wrapper(d_keys.Current());
        THRUST_NS_QUALIFIER::device_ptr<ValueT>   d_values_wrapper(d_values.Current());

        if (IS_DESCENDING) {
            THRUST_NS_QUALIFIER::reverse(d_keys_wrapper, d_keys_wrapper + num_items);
            THRUST_NS_QUALIFIER::reverse(d_values_wrapper, d_values_wrapper + num_items);
        }

        THRUST_NS_QUALIFIER::sort_by_key(d_keys_wrapper, d_keys_wrapper + num_items, d_values_wrapper);

        if (IS_DESCENDING) {
            THRUST_NS_QUALIFIER::reverse(d_keys_wrapper, d_keys_wrapper + num_items);
            THRUST_NS_QUALIFIER::reverse(d_values_wrapper, d_values_wrapper + num_items);
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
template <int IS_DESCENDING, typename KeyT, typename ValueT, typename BeginOffsetIteratorT, typename EndOffsetIteratorT>
__global__ void CnpDispatchKernel(
    Int2Type<IS_DESCENDING> is_descending,
    int                     *d_selector,
    size_t                  *d_temp_storage_bytes,
    cudaError_t             *d_cdp_error,

    void                    *d_temp_storage,
    size_t                  temp_storage_bytes,
    DoubleBuffer<KeyT>      d_keys,
    DoubleBuffer<ValueT>    d_values,
    std::size_t             num_items,
    int                     num_segments,
    BeginOffsetIteratorT    d_segment_begin_offsets,
    EndOffsetIteratorT      d_segment_end_offsets,
    int                     begin_bit,
    int                     end_bit,
    bool                    debug_synchronous)
{
#ifndef CUB_CDP
  (void)is_descending;
  (void)d_selector;
  (void)d_temp_storage_bytes;
  (void)d_cdp_error;
  (void)d_temp_storage;
  (void)temp_storage_bytes;
  (void)d_keys;
  (void)d_values;
  (void)num_items;
  (void)num_segments;
  (void)d_segment_begin_offsets;
  (void)d_segment_end_offsets;
  (void)begin_bit;
  (void)end_bit;
  (void)debug_synchronous;
    *d_cdp_error            = cudaErrorNotSupported;
#else
    *d_cdp_error            = Dispatch(
                                is_descending, Int2Type<CUB>(), d_selector, d_temp_storage_bytes, d_cdp_error,
                                d_temp_storage, temp_storage_bytes, d_keys, d_values,
                                num_items, num_segments, d_segment_begin_offsets, d_segment_end_offsets,
                                begin_bit, end_bit, 0, debug_synchronous);
    *d_temp_storage_bytes   = temp_storage_bytes;
    *d_selector             = d_keys.selector;
#endif
}


/**
 * Dispatch to CDP kernel
 */
template <int IS_DESCENDING, typename KeyT, typename ValueT, typename BeginOffsetIteratorT, typename EndOffsetIteratorT>
cudaError_t Dispatch(
    Int2Type<IS_DESCENDING> is_descending,
    Int2Type<CDP>           dispatch_to,
    int                     *d_selector,
    size_t                  *d_temp_storage_bytes,
    cudaError_t             *d_cdp_error,

    void                    *d_temp_storage,
    size_t                  &temp_storage_bytes,
    DoubleBuffer<KeyT>      &d_keys,
    DoubleBuffer<ValueT>    &d_values,
    std::size_t             num_items,
    int                     num_segments,
    BeginOffsetIteratorT    d_segment_begin_offsets,
    EndOffsetIteratorT      d_segment_end_offsets,
    int                     begin_bit,
    int                     end_bit,
    cudaStream_t            stream,
    bool                    debug_synchronous)
{
    // Invoke kernel to invoke device-side dispatch
    CnpDispatchKernel<<<1,1>>>(
        is_descending, d_selector, d_temp_storage_bytes, d_cdp_error,
        d_temp_storage, temp_storage_bytes, d_keys, d_values,
        num_items, num_segments, d_segment_begin_offsets, d_segment_end_offsets,
        begin_bit, end_bit, debug_synchronous);

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
// Problem generation
//---------------------------------------------------------------------


/**
 * Simple key-value pairing
 */
template <
    typename KeyT,
    typename ValueT>
struct Pair
{
    KeyT     key;
    ValueT   value;

    bool operator<(const Pair &b) const
    {
        return (key < b.key);
    }
};


/**
 * Simple key-value pairing (specialized for bool types)
 */
template <typename ValueT>
struct Pair<bool, ValueT>
{
    bool     key;
    ValueT   value;

    bool operator<(const Pair &b) const
    {
        return (!key && b.key);
    }
};


/**
 * Initialize key data
 */
template <typename KeyT>
void InitializeKeyBits(
    GenMode         gen_mode,
    KeyT            *h_keys,
    std::size_t     num_items,
    int             /*entropy_reduction*/)
{
    for (std::size_t i = 0; i < num_items; ++i)
        InitValue(gen_mode, h_keys[i], i);
}

template <typename KeyT,
          typename UnsignedBits = typename cub::Traits<KeyT>::UnsignedBits>
UnsignedBits KeyBits(KeyT key)
{
    UnsignedBits bits;
    memcpy(&bits, &key, sizeof(KeyT));
    return bits;
}

/** Initialize the reference array monotonically. */
template <typename KeyT>
void InitializeKeysSorted(
    KeyT            *h_keys,
    std::size_t     num_items)
{
    using TraitsT = cub::Traits<KeyT>;
    using UnsignedBits = typename TraitsT::UnsignedBits;

    // Numbers to generate random runs.
    UnsignedBits max_inc = 1 << (sizeof(UnsignedBits) < 4 ? 3 :
                                 (sizeof(UnsignedBits) < 8 ? 14 : 24));
    UnsignedBits min_bits = TraitsT::TwiddleIn(KeyBits(TraitsT::Lowest()));
    UnsignedBits max_bits = TraitsT::TwiddleIn(KeyBits(TraitsT::Max()));
    std::size_t max_run = std::max(
        std::size_t(double(num_items) * (max_inc + 1) / max_bits),
        std::size_t(1 << 14));

    UnsignedBits *h_key_bits = reinterpret_cast<UnsignedBits*>(h_keys);
    std::size_t i = 0;
    // Start with the minimum twiddled key.
    UnsignedBits twiddled_key = min_bits;
    while (i < num_items)
    {
        // Generate random increment (avoid overflow).
        UnsignedBits inc_bits = 0;
        RandomBits(inc_bits);
        // twiddled_key < max_bits at this point.
        UnsignedBits inc = static_cast<UnsignedBits>(std::min(1 + inc_bits % max_inc, max_bits - twiddled_key));
        twiddled_key += inc;
        
        // Generate random run length (ensure there are enough values to fill the rest).
        std::size_t run_bits = 0;
        RandomBits(run_bits);
        std::size_t run_length = std::min(1 + run_bits % max_run, num_items - i);
        if (twiddled_key == max_bits) run_length = num_items - i;
        std::size_t run_end = i + run_length;
        
        // Fill the array.
        UnsignedBits key = TraitsT::TwiddleOut(twiddled_key);
        // Avoid -0.0 for floating-point keys.
        UnsignedBits negative_zero = UnsignedBits(1) << UnsignedBits(sizeof(UnsignedBits) * 8 - 1);
        if (TraitsT::CATEGORY == cub::FLOATING_POINT && key == negative_zero)
        {
            key = 0;
        }

        for (; i < run_end; ++i)
        {
            h_key_bits[i] = key;
        }
    }
}


/**
 * Initialize solution
 */
template <bool IS_DESCENDING, typename KeyT, typename OffsetT>
void InitializeSolution(
    KeyT    *h_keys,
    OffsetT num_items,
    int     num_segments,
    bool    pre_sorted,
    OffsetT *h_segment_offsets,
    int     begin_bit,
    int     end_bit,
    OffsetT *&h_reference_ranks,
    KeyT    *&h_reference_keys)
{
    if (pre_sorted)
    {
        printf("Shuffling reference solution on CPU\n");
        // Note: begin_bit and end_bit are ignored here, and assumed to have the
        // default values (begin_bit == 0, end_bit == 8 * sizeof(KeyT)).
        // Otherwise, pre-sorting won't work, as it doesn't necessarily
        // correspond to the order of keys sorted by a subrange of bits.
        // num_segments is also ignored as assumed to be 1, as pre-sorted tests
        // are currently not supported for multiple segments.
        //
        // Pre-sorted tests with non-default begin_bit, end_bit or num_segments
        // != 1 are skipped in TestBits() and TestSegments(), respectively.

        // Copy to the reference solution.
        h_reference_keys = new KeyT[num_items];
        if (IS_DESCENDING)
        {
            // Copy in reverse.
            for (OffsetT i = 0; i < num_items; ++i)
            {
                h_reference_keys[i] = h_keys[num_items - 1 - i];
            }
            // Copy back.
            memcpy(h_keys, h_reference_keys, num_items * sizeof(KeyT));
        }
        else
        {
            memcpy(h_reference_keys, h_keys, num_items * sizeof(KeyT));
        }

        // Summarize the pre-sorted array (element, 1st position, count).
        struct Element
        {
            KeyT key;
            std::size_t num;
            std::size_t index;
        };

        std::vector<Element> summary;
        KeyT cur_key = h_reference_keys[0];
        summary.push_back(Element{cur_key, 1, 0});
        for (std::size_t i = 1; i < num_items; ++i)
        {
            KeyT key = h_reference_keys[i];
            if (key == cur_key)
            {
                // Same key.
                summary.back().num++;
                continue;
            }

            // Different key.
            cur_key = key;
            summary.push_back(Element{cur_key, 1, i});
        }
        
        // Generate a random permutation from the summary. Such a complicated
        // approach is used to permute the array and compute ranks in a
        // cache-friendly way and in a short time.
        h_reference_ranks = new std::size_t[num_items];
        std::size_t max_run = 32, run = 0, i = 0;
        while (summary.size() > 0)
        {
            // Pick up a random element and a run.
            std::size_t bits = 0;
            RandomBits(bits);
            std::size_t summary_id = bits % summary.size();
            Element& element = summary[summary_id];
            run = std::min(1 + bits % (max_run - 1), element.num);
            for (std::size_t j = 0; j < run; ++j)
            {
                h_keys[i + j] = element.key;
                h_reference_ranks[element.index + j] = i + j;
            }
            i += run;
            element.index += run;
            element.num -= run;
            if (element.num == 0)
            {
                // Remove the empty entry.
                std::swap(summary[summary_id], summary.back());
                summary.pop_back();
            }
        }
        printf(" Done.\n");
    }
    else
    {
        typedef Pair<KeyT, OffsetT> PairT;

        PairT *h_pairs = new PairT[num_items];

        int num_bits = end_bit - begin_bit;
        for (OffsetT i = 0; i < num_items; ++i)
        {

            // Mask off unwanted portions
            if (num_bits < static_cast<int>(sizeof(KeyT) * 8))
            {
                unsigned long long base = 0;
                memcpy(&base, &h_keys[i], sizeof(KeyT));
                base &= ((1ull << num_bits) - 1) << begin_bit;
                memcpy(&h_pairs[i].key, &base, sizeof(KeyT));
            }
            else
            {
                h_pairs[i].key = h_keys[i];
            }

            h_pairs[i].value = i;
        }

        printf("\nSorting reference solution on CPU (%d segments)...", num_segments); fflush(stdout);

        for (int i = 0; i < num_segments; ++i)
        {
            if (IS_DESCENDING) std::reverse(h_pairs + h_segment_offsets[i], h_pairs + h_segment_offsets[i + 1]);
            std::stable_sort(               h_pairs + h_segment_offsets[i], h_pairs + h_segment_offsets[i + 1]);
            if (IS_DESCENDING) std::reverse(h_pairs + h_segment_offsets[i], h_pairs + h_segment_offsets[i + 1]);
        }

        printf(" Done.\n"); fflush(stdout);

        h_reference_ranks  = new OffsetT[num_items];
        h_reference_keys   = new KeyT[num_items];

        for (OffsetT i = 0; i < num_items; ++i)
        {
            h_reference_ranks[i]    = h_pairs[i].value;
            h_reference_keys[i]     = h_keys[h_pairs[i].value];
        }

        if (h_pairs) delete[] h_pairs;
    }
}

template <bool IS_DESCENDING, typename KeyT>
void ResetKeys(KeyT *h_keys, std::size_t num_items, bool pre_sorted, KeyT *reference_keys)
{
    if (!pre_sorted) return;

    // Copy the reference keys back.
    if (IS_DESCENDING)
    {
        // Keys need to be copied in reverse.
        for (std::size_t i = 0; i < num_items; ++i)
        {
            h_keys[i] = reference_keys[num_items - 1 - i];
        }
    }
    else
    {
        memcpy(h_keys, reference_keys, num_items * sizeof(KeyT));
    }
}


//---------------------------------------------------------------------
// Test generation
//---------------------------------------------------------------------

template <typename T>
struct UnwrapHalfAndBfloat16 {
    using Type = T;
};

#if (__CUDACC_VER_MAJOR__ >= 9 || CUDA_VERSION >= 9000) && !_NVHPC_CUDA
template <>
struct UnwrapHalfAndBfloat16<half_t> {
    using Type = __half;
};
#endif

#if (__CUDACC_VER_MAJOR__ >= 11 || CUDA_VERSION >= 11000) && !_NVHPC_CUDA
template <>
struct UnwrapHalfAndBfloat16<bfloat16_t> {
    using Type = __nv_bfloat16;
};
#endif

/**
 * Test DeviceRadixSort
 */
template <
    Backend     BACKEND,
    bool        IS_DESCENDING,
    typename    KeyT,
    typename    ValueT,
    typename    BeginOffsetIteratorT,
    typename    EndOffsetIteratorT>
void Test(
    KeyT                 *h_keys,
    ValueT               *h_values,
    std::size_t          num_items,
    int                  num_segments,
    BeginOffsetIteratorT d_segment_begin_offsets,
    EndOffsetIteratorT   d_segment_end_offsets,
    int                  begin_bit,
    int                  end_bit,
    KeyT                 *h_reference_keys,
    ValueT               *h_reference_values)
{
    // Key alias type
    using KeyAliasT = typename UnwrapHalfAndBfloat16<KeyT>::Type;

    const bool KEYS_ONLY = Equals<ValueT, NullType>::VALUE;

    printf("%s %s cub::DeviceRadixSort %zd items, %d segments, %d-byte keys (%s) %d-byte values (%s), descending %d, begin_bit %d, end_bit %d\n",
        (BACKEND == CUB_NO_OVERWRITE) ? "CUB_NO_OVERWRITE" : (BACKEND == CDP) ? "CDP CUB" : (BACKEND == THRUST) ? "Thrust" : "CUB",
        (KEYS_ONLY) ? "keys-only" : "key-value",
        num_items, num_segments,
        (int) sizeof(KeyT), typeid(KeyT).name(), (KEYS_ONLY) ? 0 : (int) sizeof(ValueT), typeid(ValueT).name(),
        IS_DESCENDING, begin_bit, end_bit);
    fflush(stdout);

    if (g_verbose)
    {
        printf("Input keys:\n");
        DisplayResults(h_keys, num_items);
        printf("\n\n");
    }

    // Allocate device arrays
    DoubleBuffer<KeyAliasT> d_keys;
    DoubleBuffer<ValueT>    d_values;
    int                     *d_selector;
    size_t                  *d_temp_storage_bytes;
    cudaError_t             *d_cdp_error;
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_keys.d_buffers[0], sizeof(KeyT) * num_items));
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_keys.d_buffers[1], sizeof(KeyT) * num_items));
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_selector, sizeof(int) * 1));
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_temp_storage_bytes, sizeof(size_t) * 1));
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_cdp_error, sizeof(cudaError_t) * 1));
    if (!KEYS_ONLY)
    {
        CubDebugExit(g_allocator.DeviceAllocate((void**)&d_values.d_buffers[0], sizeof(ValueT) * num_items));
        CubDebugExit(g_allocator.DeviceAllocate((void**)&d_values.d_buffers[1], sizeof(ValueT) * num_items));
    }

    // Allocate temporary storage (and make it un-aligned)
    size_t  temp_storage_bytes  = 0;
    void    *d_temp_storage     = NULL;
    CubDebugExit(Dispatch(
        Int2Type<IS_DESCENDING>(), Int2Type<BACKEND>(), d_selector, d_temp_storage_bytes, d_cdp_error,
        d_temp_storage, temp_storage_bytes, d_keys, d_values,
        num_items, num_segments, d_segment_begin_offsets, d_segment_end_offsets,
        begin_bit, end_bit, 0, true));

    CubDebugExit(g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes + 1));
    void* mis_aligned_temp = static_cast<char*>(d_temp_storage) + 1;

    // Initialize/clear device arrays
    d_keys.selector = 0;
    CubDebugExit(cudaMemcpy(d_keys.d_buffers[0], h_keys, sizeof(KeyT) * num_items, cudaMemcpyHostToDevice));
    CubDebugExit(cudaMemset(d_keys.d_buffers[1], 0, sizeof(KeyT) * num_items));
    if (!KEYS_ONLY)
    {
        d_values.selector = 0;
        CubDebugExit(cudaMemcpy(d_values.d_buffers[0], h_values, sizeof(ValueT) * num_items, cudaMemcpyHostToDevice));
        CubDebugExit(cudaMemset(d_values.d_buffers[1], 0, sizeof(ValueT) * num_items));
    }

    // Run warmup/correctness iteration
    CubDebugExit(Dispatch(
        Int2Type<IS_DESCENDING>(), Int2Type<BACKEND>(), d_selector, d_temp_storage_bytes, d_cdp_error,
        mis_aligned_temp, temp_storage_bytes, d_keys, d_values,
        num_items, num_segments, d_segment_begin_offsets, d_segment_end_offsets,
        begin_bit, end_bit, 0, true));

    // Flush any stdout/stderr
    fflush(stdout);
    fflush(stderr);

    // Check for correctness (and display results, if specified)
    printf("Warmup done.  Checking results:\n"); fflush(stdout);
    int compare = CompareDeviceResults(h_reference_keys, reinterpret_cast<KeyT*>(d_keys.Current()), num_items, true, g_verbose);
    printf("\t Compare keys (selector %d): %s ", d_keys.selector, compare ? "FAIL" : "PASS"); fflush(stdout);
    if (!KEYS_ONLY)
    {
        int values_compare = CompareDeviceResults(h_reference_values, d_values.Current(), num_items, true, g_verbose);
        compare |= values_compare;
        printf("\t Compare values (selector %d): %s ", d_values.selector, values_compare ? "FAIL" : "PASS"); fflush(stdout);
    }
    if (BACKEND == CUB_NO_OVERWRITE)
    {
        // Check that input isn't overwritten
        int input_compare = CompareDeviceResults(h_keys, reinterpret_cast<KeyT*>(d_keys.d_buffers[0]), num_items, true, g_verbose);
        compare |= input_compare;
        printf("\t Compare input keys: %s ", input_compare ? "FAIL" : "PASS"); fflush(stdout);
    }

    // Performance
    if (g_timing_iterations)
        printf("\nPerforming timing iterations:\n"); fflush(stdout);

    GpuTimer gpu_timer;
    float elapsed_millis = 0.0f;
    for (int i = 0; i < g_timing_iterations; ++i)
    {
        // Initialize/clear device arrays
        CubDebugExit(cudaMemcpy(d_keys.d_buffers[d_keys.selector], h_keys, sizeof(KeyT) * num_items, cudaMemcpyHostToDevice));
        CubDebugExit(cudaMemset(d_keys.d_buffers[d_keys.selector ^ 1], 0, sizeof(KeyT) * num_items));
        if (!KEYS_ONLY)
        {
            CubDebugExit(cudaMemcpy(d_values.d_buffers[d_values.selector], h_values, sizeof(ValueT) * num_items, cudaMemcpyHostToDevice));
            CubDebugExit(cudaMemset(d_values.d_buffers[d_values.selector ^ 1], 0, sizeof(ValueT) * num_items));
        }

        gpu_timer.Start();
        CubDebugExit(Dispatch(
            Int2Type<IS_DESCENDING>(), Int2Type<BACKEND>(), d_selector, d_temp_storage_bytes, d_cdp_error,
            mis_aligned_temp, temp_storage_bytes, d_keys, d_values,
            num_items, num_segments, d_segment_begin_offsets, d_segment_end_offsets,
            begin_bit, end_bit, 0, false));
        gpu_timer.Stop();
        elapsed_millis += gpu_timer.ElapsedMillis();
    }

    // Display performance
    if (g_timing_iterations > 0)
    {
        float avg_millis = elapsed_millis / g_timing_iterations;
        float giga_rate = float(num_items) / avg_millis / 1000.0f / 1000.0f;
        float giga_bandwidth = (KEYS_ONLY) ?
            giga_rate * sizeof(KeyT) * 2 :
            giga_rate * (sizeof(KeyT) + sizeof(ValueT)) * 2;
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

// Returns whether there is enough memory for the test.
template <typename KeyT, typename ValueT>
bool HasEnoughMemory(std::size_t num_items)
{
    std::size_t total_mem = TotalGlobalMem();
    std::size_t value_size = Equals<ValueT, NullType>::VALUE ? 0 : sizeof(ValueT);
    // A conservative estimate of the amount of memory required.
    std::size_t test_mem = 4 * num_items * (sizeof(KeyT) + value_size);
    return test_mem < total_mem;
}

/**
 * Test backend
 */
template <bool IS_DESCENDING, typename KeyT, typename ValueT, typename BeginOffsetIteratorT, typename EndOffsetIteratorT>
void TestBackend(
    KeyT                 *h_keys,
    std::size_t          num_items,
    int                  num_segments,
    BeginOffsetIteratorT d_segment_begin_offsets,
    EndOffsetIteratorT   d_segment_end_offsets,
    int                  begin_bit,
    int                  end_bit,
    KeyT                 *h_reference_keys,
    std::size_t          *h_reference_ranks)
{
    const bool KEYS_ONLY = Equals<ValueT, NullType>::VALUE;

    ValueT *h_values             = NULL;
    ValueT *h_reference_values   = NULL;

    // Skip tests for which we don't have enough memory.
    if (!HasEnoughMemory<KeyT, ValueT>(num_items)) return;

    if (!KEYS_ONLY)
    {
        h_values            = new ValueT[num_items];
        h_reference_values  = new ValueT[num_items];

        for (std::size_t i = 0; i < num_items; ++i)
        {
            InitValue(INTEGER_SEED, h_values[i], i);
            InitValue(INTEGER_SEED, h_reference_values[i], h_reference_ranks[i]);
        }
    }

#ifdef SEGMENTED_SORT
    // Test multi-segment implementations
    Test<CUB_SEGMENTED, IS_DESCENDING>(               h_keys, h_values, num_items, num_segments, d_segment_begin_offsets, d_segment_end_offsets, begin_bit, end_bit, h_reference_keys, h_reference_values);
    Test<CUB_SEGMENTED_NO_OVERWRITE, IS_DESCENDING>(  h_keys, h_values, num_items, num_segments, d_segment_begin_offsets, d_segment_end_offsets, begin_bit, end_bit, h_reference_keys, h_reference_values);
#else   // SEGMENTED_SORT
    if (num_segments == 1)
    {
        // Test single-segment implementations
        Test<CUB, IS_DESCENDING>(               h_keys, h_values, num_items, num_segments, d_segment_begin_offsets, d_segment_end_offsets, begin_bit, end_bit, h_reference_keys, h_reference_values);
        Test<CUB_NO_OVERWRITE, IS_DESCENDING>(  h_keys, h_values, num_items, num_segments, d_segment_begin_offsets, d_segment_end_offsets, begin_bit, end_bit, h_reference_keys, h_reference_values);
    #ifdef CUB_CDP
        Test<CDP, IS_DESCENDING>(               h_keys, h_values, num_items, num_segments, d_segment_begin_offsets, d_segment_end_offsets, begin_bit, end_bit, h_reference_keys, h_reference_values);
    #endif
    }
#endif  // SEGMENTED_SORT

    if (h_values) delete[] h_values;
    if (h_reference_values) delete[] h_reference_values;
}




/**
 * Test value type
 */
template <bool IS_DESCENDING, typename KeyT, typename BeginOffsetIteratorT, typename EndOffsetIteratorT>
void TestValueTypes(
    KeyT                 *h_keys,
    std::size_t          num_items,
    int                  num_segments,
    bool                 pre_sorted,
    std::size_t          *h_segment_offsets,
    BeginOffsetIteratorT d_segment_begin_offsets,
    EndOffsetIteratorT   d_segment_end_offsets,
    int                  begin_bit,
    int                  end_bit)
{
    // Initialize the solution
    std::size_t *h_reference_ranks = NULL;
    KeyT *h_reference_keys = NULL;
    InitializeSolution<IS_DESCENDING>(h_keys, num_items, num_segments, pre_sorted, h_segment_offsets, begin_bit, end_bit, h_reference_ranks, h_reference_keys);

    // Test keys-only
    TestBackend<IS_DESCENDING, KeyT, NullType>          (h_keys, num_items, num_segments, d_segment_begin_offsets, d_segment_end_offsets, begin_bit, end_bit, h_reference_keys, h_reference_ranks);

    // Test with 8b value
    TestBackend<IS_DESCENDING, KeyT, unsigned char>     (h_keys, num_items, num_segments, d_segment_begin_offsets, d_segment_end_offsets, begin_bit, end_bit, h_reference_keys, h_reference_ranks);

    // Test with 32b value
    TestBackend<IS_DESCENDING, KeyT, unsigned int>      (h_keys, num_items, num_segments, d_segment_begin_offsets, d_segment_end_offsets, begin_bit, end_bit, h_reference_keys, h_reference_ranks);

    // Test with 64b value
    TestBackend<IS_DESCENDING, KeyT, unsigned long long>(h_keys, num_items, num_segments, d_segment_begin_offsets, d_segment_end_offsets, begin_bit, end_bit, h_reference_keys, h_reference_ranks);

    // Test with non-trivially-constructable value
    TestBackend<IS_DESCENDING, KeyT, TestBar>           (h_keys, num_items, num_segments, d_segment_begin_offsets, d_segment_end_offsets, begin_bit, end_bit, h_reference_keys, h_reference_ranks);

    // Cleanup
    ResetKeys<IS_DESCENDING>(h_keys, num_items, pre_sorted, h_reference_keys);
    if (h_reference_ranks) delete[] h_reference_ranks;
    if (h_reference_keys) delete[] h_reference_keys;
}



/**
 * Test ascending/descending
 */
template <typename KeyT, typename BeginOffsetIteratorT, typename EndOffsetIteratorT>
void TestDirection(
    KeyT                 *h_keys,
    std::size_t          num_items,
    int                  num_segments,
    bool                 pre_sorted,
    std::size_t          *h_segment_offsets,
    BeginOffsetIteratorT d_segment_begin_offsets,
    EndOffsetIteratorT   d_segment_end_offsets,
    int                  begin_bit,
    int                  end_bit)
{
    TestValueTypes<true>(h_keys, num_items, num_segments, pre_sorted, h_segment_offsets, d_segment_begin_offsets, d_segment_end_offsets, begin_bit, end_bit);
    TestValueTypes<false>(h_keys, num_items, num_segments, pre_sorted, h_segment_offsets, d_segment_begin_offsets, d_segment_end_offsets, begin_bit, end_bit);
}


/**
 * Test different bit ranges
 */
template <typename KeyT, typename BeginOffsetIteratorT, typename EndOffsetIteratorT>
void TestBits(
    KeyT                 *h_keys,
    std::size_t          num_items,
    int                  num_segments,
    bool                 pre_sorted,
    std::size_t          *h_segment_offsets,
    BeginOffsetIteratorT d_segment_begin_offsets,
    EndOffsetIteratorT   d_segment_end_offsets)
{
    // Don't test partial-word sorting for boolean, fp, or signed types (the bit-flipping techniques get in the way) or pre-sorted keys
    if ((Traits<KeyT>::CATEGORY == UNSIGNED_INTEGER) && (!Equals<KeyT, bool>::VALUE)
        && !pre_sorted)
    {
        // Partial bits
        int begin_bit = 1;
        int end_bit = (sizeof(KeyT) * 8) - 1;
        printf("Testing key bits [%d,%d)\n", begin_bit, end_bit); fflush(stdout);
        TestDirection(h_keys, num_items, num_segments, pre_sorted, h_segment_offsets, d_segment_begin_offsets, d_segment_end_offsets, begin_bit, end_bit);

        // Across subword boundaries
        int mid_bit = sizeof(KeyT) * 4;
        printf("Testing key bits [%d,%d)\n", mid_bit - 1, mid_bit + 1); fflush(stdout);
        TestDirection(h_keys, num_items, num_segments, pre_sorted, h_segment_offsets, d_segment_begin_offsets, d_segment_end_offsets, mid_bit - 1, mid_bit + 1);
    }

    printf("Testing key bits [%d,%d)\n", 0, int(sizeof(KeyT)) * 8); fflush(stdout);
    TestDirection(h_keys, num_items, num_segments, pre_sorted, h_segment_offsets, d_segment_begin_offsets, d_segment_end_offsets, 0, sizeof(KeyT) * 8);
}


template<typename OffsetT>
struct TransformFunctor1
{
    __host__ __device__ __forceinline__ OffsetT operator()(OffsetT offset) const
    {
        return offset;
    }
};

template<typename OffsetT>
struct TransformFunctor2
{
    __host__ __device__ __forceinline__ OffsetT operator()(OffsetT offset) const
    {
        return offset;
    }
};


/**
* Test different segment iterators
*/
template <typename KeyT>
void TestSegmentIterators(
    KeyT           *h_keys,
    std::size_t    num_items,
    int            num_segments,
    bool           pre_sorted,
    std::size_t    *h_segment_offsets,
    std::size_t    *d_segment_offsets)
{
    InitializeSegments(num_items, num_segments, h_segment_offsets);
    CubDebugExit(cudaMemcpy(d_segment_offsets, h_segment_offsets, sizeof(std::size_t) * (num_segments + 1), cudaMemcpyHostToDevice));

    // Test with segment pointer
    TestBits(h_keys, num_items, num_segments, pre_sorted, h_segment_offsets, d_segment_offsets, d_segment_offsets + 1);

#ifdef SEGMENTED_SORT
    // Test with segment iterator
    typedef CastOp<std::size_t> IdentityOpT;
    IdentityOpT identity_op;
    TransformInputIterator<std::size_t, IdentityOpT, std::size_t*, std::size_t> d_segment_offsets_itr(d_segment_offsets, identity_op);

    TestBits(h_keys, num_items, num_segments, pre_sorted, h_segment_offsets, d_segment_offsets_itr, d_segment_offsets_itr + 1);

    // Test with transform iterators of different types
    typedef TransformFunctor1<std::size_t> TransformFunctor1T;
    typedef TransformFunctor2<std::size_t> TransformFunctor2T;

    TransformInputIterator<std::size_t, TransformFunctor1T, std::size_t*, std::size_t> d_segment_begin_offsets_itr(d_segment_offsets, TransformFunctor1T());
    TransformInputIterator<std::size_t, TransformFunctor2T, std::size_t*, std::size_t> d_segment_end_offsets_itr(d_segment_offsets + 1, TransformFunctor2T());

    TestBits(h_keys, num_items, num_segments, pre_sorted, h_segment_offsets, d_segment_begin_offsets_itr, d_segment_end_offsets_itr);
#endif
}


/**
 * Test different segment compositions
 */
template <typename KeyT>
void TestSegments(
    KeyT         *h_keys,
    std::size_t  num_items,
    int          max_segments,
    bool         pre_sorted)
{
    std::size_t *h_segment_offsets = new size_t[max_segments + 1];

    std::size_t *d_segment_offsets = nullptr;
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_segment_offsets, sizeof(std::size_t) * (max_segments + 1)));

#ifdef SEGMENTED_SORT
    for (int num_segments = max_segments; num_segments > 1; num_segments = (num_segments + 32 - 1) / 32)
    {
        // Pre-sorted tests are not supported for segmented sort
        if (num_items / num_segments < 128 * 1000 && !pre_sorted) {
            // Right now we assign a single thread block to each segment, so lets keep it to under 128K items per segment
            TestSegmentIterators(h_keys, num_items, num_segments, pre_sorted, h_segment_offsets, d_segment_offsets);
        }
    }
#else
    // Test single segment
    if (num_items < 128 * 1000 || pre_sorted) {
        // Right now we assign a single thread block to each segment, so lets keep it to under 128K items per segment
        TestSegmentIterators(h_keys, num_items, 1, pre_sorted, h_segment_offsets, d_segment_offsets);
    }
#endif

    if (h_segment_offsets) delete[] h_segment_offsets;
    if (d_segment_offsets) CubDebugExit(g_allocator.DeviceFree(d_segment_offsets));
}


/**
 * Test different (sub)lengths and number of segments
 */
template <typename KeyT>
void TestSizes(
    KeyT         *h_keys,
    std::size_t  max_items,
    int          max_segments,
    bool         pre_sorted)
{
    for (std::size_t num_items = max_items;
         num_items > 1;
         num_items = cub::DivideAndRoundUp(num_items, 32))
    {
        TestSegments(h_keys, num_items, max_segments, pre_sorted);
    }
    TestSegments(h_keys, 1, max_segments, pre_sorted);
    TestSegments(h_keys, 0, max_segments, pre_sorted);
}


/**
 * Test key sampling distributions
 */
template <typename KeyT>
void TestGen(
    std::size_t     max_items,
    int             max_segments)
{
    if (max_items == ~std::size_t(0))
        max_items = 200000003;

    if (max_segments < 0)
        max_segments = 5003;

    KeyT *h_keys = new KeyT[max_items];

    for (int entropy_reduction = 0; entropy_reduction <= 6; entropy_reduction += 3)
    {
        printf("\nTesting random %s keys with entropy reduction factor %d\n", typeid(KeyT).name(), entropy_reduction); fflush(stdout);
        InitializeKeyBits(RANDOM, h_keys, max_items, entropy_reduction);
        TestSizes(h_keys, max_items, max_segments, false);
    }

    if (cub::Traits<KeyT>::CATEGORY == cub::FLOATING_POINT)
    {
        printf("\nTesting random %s keys with some replaced with -0.0 or +0.0 \n", typeid(KeyT).name());
        fflush(stdout);
        InitializeKeyBits(RANDOM_MINUS_PLUS_ZERO, h_keys, max_items, 0);
        TestSizes(h_keys, max_items, max_segments, false);
    }

    printf("\nTesting uniform %s keys\n", typeid(KeyT).name()); fflush(stdout);
    InitializeKeyBits(UNIFORM, h_keys, max_items, 0);
    TestSizes(h_keys, max_items, max_segments, false);

    printf("\nTesting natural number %s keys\n", typeid(KeyT).name()); fflush(stdout);
    InitializeKeyBits(INTEGER_SEED, h_keys, max_items, 0);
    TestSizes(h_keys, max_items, max_segments, false);

    if (!cub::Equals<KeyT, bool>::VALUE)
    {
        printf("\nTesting pre-sorted and randomly permuted %s keys\n", typeid(KeyT).name());
        fflush(stdout);
        InitializeKeysSorted(h_keys, max_items);
        fflush(stdout);
        TestSizes(h_keys, max_items, max_segments, true);
        fflush(stdout);
    }

    if (h_keys) delete[] h_keys;
}


//---------------------------------------------------------------------
// Simple test
//---------------------------------------------------------------------

template <
    Backend     BACKEND,
    typename    KeyT,
    typename    ValueT,
    bool        IS_DESCENDING>
void Test(
    std::size_t num_items,
    int         num_segments,
    GenMode     gen_mode,
    int         entropy_reduction,
    int         begin_bit,
    int         end_bit)
{
    const bool KEYS_ONLY = Equals<ValueT, NullType>::VALUE;

    KeyT         *h_keys             = new KeyT[num_items];
    std::size_t  *h_reference_ranks  = NULL;
    KeyT         *h_reference_keys   = NULL;
    ValueT       *h_values           = NULL;
    ValueT       *h_reference_values = NULL;
    size_t       *h_segment_offsets  = new std::size_t[num_segments + 1];

    std::size_t* d_segment_offsets = nullptr;
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_segment_offsets, sizeof(std::size_t) * (num_segments + 1)));

    if (end_bit < 0)
        end_bit = sizeof(KeyT) * 8;

    InitializeKeyBits(gen_mode, h_keys, num_items, entropy_reduction);
    InitializeSegments(num_items, num_segments, h_segment_offsets);
    CubDebugExit(cudaMemcpy(d_segment_offsets, h_segment_offsets, sizeof(std::size_t) * (num_segments + 1), cudaMemcpyHostToDevice));
    InitializeSolution<IS_DESCENDING>(
        h_keys, num_items, num_segments, false, h_segment_offsets,
        begin_bit, end_bit, h_reference_ranks, h_reference_keys);

    if (!KEYS_ONLY)
    {
        h_values            = new ValueT[num_items];
        h_reference_values  = new ValueT[num_items];

        for (std::size_t i = 0; i < num_items; ++i)
        {
            InitValue(INTEGER_SEED, h_values[i], i);
            InitValue(INTEGER_SEED, h_reference_values[i], h_reference_ranks[i]);
        }
    }
    if (h_reference_ranks) delete[] h_reference_ranks;

    printf("\nTesting bits [%d,%d) of %s keys with gen-mode %d\n", begin_bit, end_bit, typeid(KeyT).name(), gen_mode); fflush(stdout);
    Test<BACKEND, IS_DESCENDING>(
        h_keys, h_values,
        num_items, num_segments, d_segment_offsets, d_segment_offsets + 1,
        begin_bit, end_bit, h_reference_keys, h_reference_values);

    if (h_keys)             delete[] h_keys;
    if (h_reference_keys)   delete[] h_reference_keys;
    if (h_values)           delete[] h_values;
    if (h_reference_values) delete[] h_reference_values;
    if (h_segment_offsets)  delete[] h_segment_offsets;
    if (d_segment_offsets) CubDebugExit(g_allocator.DeviceFree(d_segment_offsets));
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
    std::size_t num_items = ~std::size_t(0);
    int num_segments = -1;
    int entropy_reduction = 0;

    // Initialize command line
    CommandLineArgs args(argc, argv);
    g_verbose = args.CheckCmdLineFlag("v");
    args.GetCmdLineArgument("n", num_items);
    args.GetCmdLineArgument("s", num_segments);
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
            "[--s=<num segments> "
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
    int ptx_version = 0;
    CubDebugExit(PtxVersion(ptx_version));

#ifdef CUB_TEST_MINIMAL

    enum {
        IS_DESCENDING   = false
    };

    // Compile/run basic CUB test
    if (num_items == ~std::size_t(0))  num_items       = 24000000;
    if (num_segments < 0)              num_segments    = 5000;

    Test<CUB_SEGMENTED, unsigned int,       NullType, IS_DESCENDING>(num_items, num_segments, RANDOM, entropy_reduction, 0, bits);

    printf("\n-------------------------------\n");

    Test<CUB,           unsigned char,      NullType, IS_DESCENDING>(num_items, 1, RANDOM, entropy_reduction, 0, bits);
    Test<CUB,           unsigned int,       NullType, IS_DESCENDING>(num_items, 1, RANDOM, entropy_reduction, 0, bits);
    Test<CUB,           unsigned long long, NullType, IS_DESCENDING>(num_items, 1, RANDOM, entropy_reduction, 0, bits);

    printf("\n-------------------------------\n");

#if (__CUDACC_VER_MAJOR__ >= 9 || CUDA_VERSION >= 9000) && !_NVHPC_CUDA
    Test<CUB,           half_t,             NullType, IS_DESCENDING>(num_items, 1, RANDOM, entropy_reduction, 0, bits);
#endif

#if (__CUDACC_VER_MAJOR__ >= 11 || CUDA_VERSION >= 11000) && !_NVHPC_CUDA
#if !defined(__ICC)
    // Fails with `-0 != 0` with ICC for unknown reasons. See #333.
    Test<CUB,           bfloat16_t,         NullType, IS_DESCENDING>(num_items, 1, RANDOM, entropy_reduction, 0, bits);
#endif
#endif
    Test<CUB,           float,              NullType, IS_DESCENDING>(num_items, 1, RANDOM, entropy_reduction, 0, bits);
    Test<CUB,           double,             NullType, IS_DESCENDING>(num_items, 1, RANDOM, entropy_reduction, 0, bits);

    printf("\n-------------------------------\n");

    Test<CUB,           unsigned char,      unsigned int, IS_DESCENDING>(num_items, 1, RANDOM, entropy_reduction, 0, bits);
    Test<CUB,           unsigned int,       unsigned int, IS_DESCENDING>(num_items, 1, RANDOM, entropy_reduction, 0, bits);
    Test<CUB,           unsigned long long, unsigned int, IS_DESCENDING>(num_items, 1, RANDOM, entropy_reduction, 0, bits);

#elif defined(CUB_TEST_BENCHMARK)

    // Compile/run quick tests
    if (num_items == ~std::size_t(0))  num_items       = 48000000;
    if (num_segments < 0)              num_segments    = 5000;

    // Compare CUB and thrust on 32b keys-only
    Test<CUB, unsigned int, NullType, false> (                      num_items, 1, RANDOM, entropy_reduction, 0, bits);
    Test<THRUST, unsigned int, NullType, false> (                   num_items, 1, RANDOM, entropy_reduction, 0, bits);

    // Compare CUB and thrust on 64b keys-only
    Test<CUB, unsigned long long, NullType, false> (                num_items, 1, RANDOM, entropy_reduction, 0, bits);
    Test<THRUST, unsigned long long, NullType, false> (             num_items, 1, RANDOM, entropy_reduction, 0, bits);


    // Compare CUB and thrust on 32b key-value pairs
    Test<CUB, unsigned int, unsigned int, false> (                  num_items, 1, RANDOM, entropy_reduction, 0, bits);
    Test<THRUST, unsigned int, unsigned int, false> (               num_items, 1, RANDOM, entropy_reduction, 0, bits);

    // Compare CUB and thrust on 64b key + 32b value pairs
    Test<CUB, unsigned long long, unsigned int, false> (      num_items, 1, RANDOM, entropy_reduction, 0, bits);
    Test<THRUST, unsigned long long, unsigned int, false> (   num_items, 1, RANDOM, entropy_reduction, 0, bits);


#else

    // Compile/run thorough tests
    for (int i = 0; i <= g_repeat; ++i)
    {
        TestGen<bool>                 (num_items, num_segments);

        TestGen<char>                 (num_items, num_segments);
        TestGen<signed char>          (num_items, num_segments);
        TestGen<unsigned char>        (num_items, num_segments);

        TestGen<short>                (num_items, num_segments);
        TestGen<unsigned short>       (num_items, num_segments);

        TestGen<int>                  (num_items, num_segments);
        TestGen<unsigned int>         (num_items, num_segments);

        TestGen<long>                 (num_items, num_segments);
        TestGen<unsigned long>        (num_items, num_segments);

        TestGen<long long>            (num_items, num_segments);
        TestGen<unsigned long long>   (num_items, num_segments);

#if (__CUDACC_VER_MAJOR__ >= 9 || CUDA_VERSION >= 9000) && !_NVHPC_CUDA
        TestGen<half_t>               (num_items, num_segments);
#endif

#if (__CUDACC_VER_MAJOR__ >= 11 || CUDA_VERSION >= 11000) && !_NVHPC_CUDA
#if !defined(__ICC)
        // Fails with `-0 != 0` with ICC for unknown reasons. See #333.
        TestGen<bfloat16_t>           (num_items, num_segments);
#endif
#endif
        TestGen<float>                (num_items, num_segments);

        if (ptx_version > 120)                          // Don't check doubles on PTX120 or below because they're down-converted
            TestGen<double>           (num_items, num_segments);

    }

#endif

    return 0;
}
