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

#include <algorithm>
#include <climits>
#include <cstdio>
#include <limits>
#include <memory>
#include <random>
#include <type_traits>
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

using namespace cub;


//---------------------------------------------------------------------
// Globals, constants and typedefs
//---------------------------------------------------------------------

bool                    g_verbose                       = false;
int                     g_timing_iterations             = 0;
std::size_t             g_smallest_pre_sorted_num_items = (std::size_t(1) << 32) - 42;
CachingDeviceAllocator  g_allocator(true);

// Dispatch types
enum Backend
{
    CUB,                        // CUB method (allows overwriting of input)
    CUB_NO_OVERWRITE,           // CUB method (disallows overwriting of input)

    CUB_SEGMENTED,              // CUB method (allows overwriting of input)
    CUB_SEGMENTED_NO_OVERWRITE, // CUB method (disallows overwriting of input)

    CDP,                        // GPU-based (dynamic parallelism) dispatch to CUB method
};

static const char* BackendToString(Backend b)
{
  switch (b)
  {
    case CUB:
      return "CUB";
    case CUB_NO_OVERWRITE:
      return "CUB_NO_OVERWRITE";
    case CUB_SEGMENTED:
      return "CUB_SEGMENTED";
    case CUB_SEGMENTED_NO_OVERWRITE:
      return "CUB_SEGMENTED_NO_OVERWRITE";
    case CDP:
      return "CDP";
    default:
      break;
  }

  return "";
}

//---------------------------------------------------------------------
// Dispatch to different DeviceRadixSort entrypoints
//---------------------------------------------------------------------

/**
 * Dispatch to CUB sorting entrypoint (specialized for ascending)
 */
template <typename KeyT, typename ValueT, typename BeginOffsetIteratorT, typename EndOffsetIteratorT,
          typename NumItemsT>
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
    NumItemsT               num_items,
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
        num_items, begin_bit, end_bit, stream, debug_synchronous);
}

/**
 * Dispatch to CUB_NO_OVERWRITE sorting entrypoint (specialized for ascending)
 */
template <typename KeyT, typename ValueT, typename BeginOffsetIteratorT, typename EndOffsetIteratorT,
          typename NumItemsT>
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
    NumItemsT               num_items,
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
        num_items, begin_bit, end_bit, stream, debug_synchronous);

    d_keys.selector ^= 1;
    d_values.selector ^= 1;
    return retval;
}

/**
 * Dispatch to CUB sorting entrypoint (specialized for descending)
 */
template <typename KeyT, typename ValueT, typename BeginOffsetIteratorT, typename EndOffsetIteratorT,
          typename NumItemsT>
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
    NumItemsT               num_items,
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
        d_keys, d_values, num_items,
        begin_bit, end_bit, stream, debug_synchronous);
}


/**
 * Dispatch to CUB_NO_OVERWRITE sorting entrypoint (specialized for descending)
 */
template <typename KeyT, typename ValueT, typename BeginOffsetIteratorT, typename EndOffsetIteratorT,
          typename NumItemsT>
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
    NumItemsT               num_items,
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
        num_items, begin_bit, end_bit, stream, debug_synchronous);

    d_keys.selector ^= 1;
    d_values.selector ^= 1;
    return retval;
}

//---------------------------------------------------------------------
// Dispatch to different DeviceRadixSort entrypoints
//---------------------------------------------------------------------

// Validates that `num_items` fits into `int`
// TODO(canonizer): remove this check once num_items is templated for segmented sort.
template <typename NumItemsT>
__host__ __device__ bool ValidateNumItemsForSegmentedSort(NumItemsT num_items)
{
  if (static_cast<long long int>(num_items) <
      static_cast<long long int>(INT_MAX))
  {
    return true;
  }
  else
  {
    printf("cub::DeviceSegmentedRadixSort is currently limited by %d items but "
           "%lld were provided\n",
           INT_MAX,
           static_cast<long long int>(num_items));
  }

  return false;
}

/**
 * Dispatch to CUB_SEGMENTED sorting entrypoint (specialized for ascending)
 */
template <typename KeyT, typename ValueT, typename BeginOffsetIteratorT, typename EndOffsetIteratorT,
          typename NumItemsT>
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
    NumItemsT               num_items,
    int                     num_segments,
    BeginOffsetIteratorT    d_segment_begin_offsets,
    EndOffsetIteratorT      d_segment_end_offsets,
    int                     begin_bit,
    int                     end_bit,
    cudaStream_t            stream,
    bool                    debug_synchronous)
{
  if (ValidateNumItemsForSegmentedSort(num_items))
  {
    return DeviceSegmentedRadixSort::SortPairs(d_temp_storage,
                                               temp_storage_bytes,
                                               d_keys,
                                               d_values,
                                               static_cast<int>(num_items),
                                               num_segments,
                                               d_segment_begin_offsets,
                                               d_segment_end_offsets,
                                               begin_bit,
                                               end_bit,
                                               stream,
                                               debug_synchronous);
  }

  return cudaErrorInvalidValue;
}

/**
 * Dispatch to CUB_SEGMENTED_NO_OVERWRITE sorting entrypoint (specialized for ascending)
 */
template <typename KeyT, typename ValueT, typename BeginOffsetIteratorT, typename EndOffsetIteratorT,
          typename NumItemsT>
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
    NumItemsT               num_items,
    int                     num_segments,
    BeginOffsetIteratorT    d_segment_begin_offsets,
    EndOffsetIteratorT      d_segment_end_offsets,
    int                     begin_bit,
    int                     end_bit,
    cudaStream_t            stream,
    bool                    debug_synchronous)
{
  if (ValidateNumItemsForSegmentedSort(num_items))
  {
    KeyT const *const_keys_itr     = d_keys.Current();
    ValueT const *const_values_itr = d_values.Current();

    cudaError_t retval =
      DeviceSegmentedRadixSort::SortPairs(d_temp_storage,
                                          temp_storage_bytes,
                                          const_keys_itr,
                                          d_keys.Alternate(),
                                          const_values_itr,
                                          d_values.Alternate(),
                                          static_cast<int>(num_items),
                                          num_segments,
                                          d_segment_begin_offsets,
                                          d_segment_end_offsets,
                                          begin_bit,
                                          end_bit,
                                          stream,
                                          debug_synchronous);

    d_keys.selector ^= 1;
    d_values.selector ^= 1;
    return retval;
  }

  return cudaErrorInvalidValue;
}


/**
 * Dispatch to CUB_SEGMENTED sorting entrypoint (specialized for descending)
 */
template <typename KeyT, typename ValueT, typename BeginOffsetIteratorT, typename EndOffsetIteratorT,
          typename NumItemsT>
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
    NumItemsT               num_items,
    int                     num_segments,
    BeginOffsetIteratorT    d_segment_begin_offsets,
    EndOffsetIteratorT      d_segment_end_offsets,
    int                     begin_bit,
    int                     end_bit,
    cudaStream_t            stream,
    bool                    debug_synchronous)
{
  if (ValidateNumItemsForSegmentedSort(num_items))
  {
    return DeviceSegmentedRadixSort::SortPairsDescending(
      d_temp_storage,
      temp_storage_bytes,
      d_keys,
      d_values,
      static_cast<int>(num_items),
      num_segments,
      d_segment_begin_offsets,
      d_segment_end_offsets,
      begin_bit,
      end_bit,
      stream,
      debug_synchronous);
  }

  return cudaErrorInvalidValue;
}

/**
 * Dispatch to CUB_SEGMENTED_NO_OVERWRITE sorting entrypoint (specialized for descending)
 */
template <typename KeyT, typename ValueT, typename BeginOffsetIteratorT, typename EndOffsetIteratorT,
          typename NumItemsT>
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
    NumItemsT               num_items,
    int                     num_segments,
    BeginOffsetIteratorT    d_segment_begin_offsets,
    EndOffsetIteratorT      d_segment_end_offsets,
    int                     begin_bit,
    int                     end_bit,
    cudaStream_t            stream,
    bool                    debug_synchronous)
{
  if (ValidateNumItemsForSegmentedSort(num_items))
  {
    KeyT const *const_keys_itr     = d_keys.Current();
    ValueT const *const_values_itr = d_values.Current();

    cudaError_t retval =
      DeviceSegmentedRadixSort::SortPairsDescending(d_temp_storage,
                                                    temp_storage_bytes,
                                                    const_keys_itr,
                                                    d_keys.Alternate(),
                                                    const_values_itr,
                                                    d_values.Alternate(),
                                                    static_cast<int>(num_items),
                                                    num_segments,
                                                    d_segment_begin_offsets,
                                                    d_segment_end_offsets,
                                                    begin_bit,
                                                    end_bit,
                                                    stream,
                                                    debug_synchronous);

    d_keys.selector ^= 1;
    d_values.selector ^= 1;
    return retval;
  }

  return cudaErrorInvalidValue;
}

//---------------------------------------------------------------------
// CUDA Nested Parallelism Test Kernel
//---------------------------------------------------------------------

/**
 * Simple wrapper kernel to invoke DeviceRadixSort
 */
template <int IS_DESCENDING, typename KeyT, typename ValueT, typename BeginOffsetIteratorT,
          typename EndOffsetIteratorT, typename NumItemsT>
__global__ void CnpDispatchKernel(
    Int2Type<IS_DESCENDING> is_descending,
    int                     *d_selector,
    size_t                  *d_temp_storage_bytes,
    cudaError_t             *d_cdp_error,

    void                    *d_temp_storage,
    size_t                  temp_storage_bytes,
    DoubleBuffer<KeyT>      d_keys,
    DoubleBuffer<ValueT>    d_values,
    NumItemsT               num_items,
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
template <int IS_DESCENDING, typename KeyT, typename ValueT, typename BeginOffsetIteratorT,
          typename EndOffsetIteratorT, typename NumItemsT>
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
    NumItemsT               num_items,
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
template <typename KeyT, typename NumItemsT>
void InitializeKeyBits(
    GenMode         gen_mode,
    KeyT            *h_keys,
    NumItemsT       num_items,
    int             /*entropy_reduction*/)
{
    for (NumItemsT i = 0; i < num_items; ++i)
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
template <typename KeyT, typename NumItemsT>
void InitializeKeysSorted(
    KeyT            *h_keys,
    NumItemsT       num_items)
{
    using TraitsT = cub::Traits<KeyT>;
    using UnsignedBits = typename TraitsT::UnsignedBits;

    // Numbers to generate random runs.
    UnsignedBits max_inc = 1 << (sizeof(UnsignedBits) < 4 ? 3 :
                                 (sizeof(UnsignedBits) < 8 ? 14 : 24));
    UnsignedBits min_bits = TraitsT::TwiddleIn(KeyBits(TraitsT::Lowest()));
    UnsignedBits max_bits = TraitsT::TwiddleIn(KeyBits(TraitsT::Max()));
    NumItemsT max_run = std::max(
        NumItemsT(double(num_items) * (max_inc + 1) / max_bits),
        NumItemsT(1 << 14));

    UnsignedBits *h_key_bits = reinterpret_cast<UnsignedBits*>(h_keys);
    NumItemsT i = 0;
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
        NumItemsT run_bits = 0;
        RandomBits(run_bits);
        NumItemsT run_length = std::min(1 + run_bits % max_run, num_items - i);
        if (twiddled_key == max_bits) run_length = num_items - i;
        NumItemsT run_end = i + run_length;

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
template <bool IS_DESCENDING, bool WANT_RANKS, typename KeyT, typename NumItemsT>
void InitializeSolution(
    KeyT       *h_keys,
    NumItemsT  num_items,
    int        num_segments,
    bool       pre_sorted,
    NumItemsT  *h_segment_offsets,
    int        begin_bit,
    int        end_bit,
    NumItemsT  *&h_reference_ranks,
    KeyT       *&h_reference_keys)
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
        AssertEquals(begin_bit, 0);
        AssertEquals(end_bit, static_cast<int>(8 * sizeof(KeyT)));
        AssertEquals(num_segments, 1);

        // Copy to the reference solution.
        h_reference_keys = new KeyT[num_items];
        if (IS_DESCENDING)
        {
            // Copy in reverse.
            for (NumItemsT i = 0; i < num_items; ++i)
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
            NumItemsT num;
            NumItemsT index;
        };

        std::vector<Element> summary;
        KeyT cur_key = h_reference_keys[0];
        summary.push_back(Element{cur_key, 1, 0});
        for (NumItemsT i = 1; i < num_items; ++i)
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
        if (WANT_RANKS)
        {
            h_reference_ranks = new NumItemsT[num_items];
        }
        NumItemsT max_run = 32, run = 0, i = 0;
        while (summary.size() > 0)
        {
            // Pick up a random element and a run.
            NumItemsT bits = 0;
            RandomBits(bits);
            NumItemsT summary_id = bits % summary.size();
            Element& element = summary[summary_id];
            run = std::min(1 + bits % (max_run - 1), element.num);
            for (NumItemsT j = 0; j < run; ++j)
            {
                h_keys[i + j] = element.key;
                if (WANT_RANKS)
                {
                    h_reference_ranks[element.index + j] = i + j;
                }
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
        typedef Pair<KeyT, NumItemsT> PairT;

        PairT *h_pairs = new PairT[num_items];

        int num_bits = end_bit - begin_bit;
        for (NumItemsT i = 0; i < num_items; ++i)
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

        if (WANT_RANKS)
        {
            h_reference_ranks  = new NumItemsT[num_items];
        }
        h_reference_keys   = new KeyT[num_items];

        for (NumItemsT i = 0; i < num_items; ++i)
        {
            if (WANT_RANKS)
            {
                h_reference_ranks[i]    = h_pairs[i].value;
            }
            h_reference_keys[i]     = h_keys[h_pairs[i].value];
        }

        if (h_pairs) delete[] h_pairs;
    }
}

template <bool IS_DESCENDING, typename KeyT, typename NumItemsT>
void ResetKeys(KeyT *h_keys, NumItemsT num_items, bool pre_sorted, KeyT *reference_keys)
{
    if (!pre_sorted) return;

    // Copy the reference keys back.
    if (IS_DESCENDING)
    {
        // Keys need to be copied in reverse.
        for (NumItemsT i = 0; i < num_items; ++i)
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
    typename    EndOffsetIteratorT,
    typename    NumItemsT>
void Test(
    KeyT                 *h_keys,
    ValueT               *h_values,
    NumItemsT            num_items,
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

    const bool KEYS_ONLY = std::is_same<ValueT, NullType>::value;

    printf("%s %s cub::DeviceRadixSort %zd items, %d segments, "
           "%d-byte keys (%s) %d-byte values (%s), descending %d, "
           "begin_bit %d, end_bit %d\n",
           BackendToString(BACKEND),
           (KEYS_ONLY) ? "keys-only" : "key-value",
           static_cast<std::size_t>(num_items),
           num_segments,
           static_cast<int>(sizeof(KeyT)),
           typeid(KeyT).name(),
           (KEYS_ONLY) ? 0 : static_cast<int>(sizeof(ValueT)),
           typeid(ValueT).name(),
           IS_DESCENDING,
           begin_bit,
           end_bit);

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
bool HasEnoughMemory(std::size_t num_items, bool overwrite)
{
    std::size_t total_mem = TotalGlobalMem();
    std::size_t value_size = std::is_same<ValueT, NullType>::value
                           ? 0
                           : sizeof(ValueT);
    // A conservative estimate of the amount of memory required.
    double factor = overwrite ? 2.25 : 3.25;
    std::size_t test_mem = static_cast<std::size_t>
      (num_items * (sizeof(KeyT) + value_size) * factor);
    return test_mem < total_mem;
}

/**
 * Test backend
 */
template <bool IS_DESCENDING, typename KeyT, typename ValueT, typename BeginOffsetIteratorT, typename EndOffsetIteratorT, typename NumItemsT>
void TestBackend(
    KeyT                 *h_keys,
    NumItemsT            num_items,
    int                  num_segments,
    BeginOffsetIteratorT d_segment_begin_offsets,
    EndOffsetIteratorT   d_segment_end_offsets,
    int                  begin_bit,
    int                  end_bit,
    KeyT                 *h_reference_keys,
    NumItemsT            *h_reference_ranks)
{
    const bool KEYS_ONLY = std::is_same<ValueT, NullType>::value;

    // A conservative check assuming overwrite is allowed.
    if (!HasEnoughMemory<KeyT, ValueT>(static_cast<std::size_t>(num_items), true))
    {
        printf("Skipping the test due to insufficient device memory\n");
        return;
    }

    ValueT *h_values             = NULL;
    ValueT *h_reference_values   = NULL;
    
    if (!KEYS_ONLY)
    {
        h_values            = new ValueT[num_items];
        h_reference_values  = new ValueT[num_items];

        for (NumItemsT i = 0; i < num_items; ++i)
        {
            InitValue(INTEGER_SEED, h_values[i], i);
            InitValue(INTEGER_SEED, h_reference_values[i], h_reference_ranks[i]);
        }
    }

    // Skip segmented sort if num_items isn't int.
    // TODO(canonizer): re-enable these tests once num_items is templated for segmented sort.
    if (std::is_same<NumItemsT, int>::value)
    {
        Test<CUB_SEGMENTED, IS_DESCENDING>(               h_keys, h_values, num_items, num_segments, d_segment_begin_offsets, d_segment_end_offsets, begin_bit, end_bit, h_reference_keys, h_reference_values);
        Test<CUB_SEGMENTED_NO_OVERWRITE, IS_DESCENDING>(  h_keys, h_values, num_items, num_segments, d_segment_begin_offsets, d_segment_end_offsets, begin_bit, end_bit, h_reference_keys, h_reference_values);
    }

    if (num_segments == 1)
    {
        Test<CUB, IS_DESCENDING>(               h_keys, h_values, num_items, num_segments, d_segment_begin_offsets, d_segment_end_offsets, begin_bit, end_bit, h_reference_keys, h_reference_values);
        if (HasEnoughMemory<KeyT, ValueT>(static_cast<std::size_t>(num_items), false))
        {
            Test<CUB_NO_OVERWRITE, IS_DESCENDING>(  h_keys, h_values, num_items, num_segments, d_segment_begin_offsets, d_segment_end_offsets, begin_bit, end_bit, h_reference_keys, h_reference_values);
        }
        else
        {
            printf("Skipping CUB_NO_OVERWRITE tests due to insufficient memory\n");
        }
    #ifdef CUB_CDP
            Test<CDP, IS_DESCENDING>(               h_keys, h_values, num_items, num_segments, d_segment_begin_offsets, d_segment_end_offsets, begin_bit, end_bit, h_reference_keys, h_reference_values);
    #endif
    }

    if (h_values) delete[] h_values;
    if (h_reference_values) delete[] h_reference_values;
}

// Smallest value type for TEST_VALUE_TYPE.
// Unless TEST_VALUE_TYPE == 3, this is the only value type tested.
#if TEST_VALUE_TYPE == 0
// Test keys-only
using SmallestValueT = NullType;
#elif TEST_VALUE_TYPE == 1
// Test with 8b value
using SmallestValueT = unsigned char;
#elif TEST_VALUE_TYPE == 2
// Test with 32b value
using SmallestValueT = unsigned int;
// Test with 64b value
#elif TEST_VALUE_TYPE == 3
using SmallestValueT = unsigned long long;
#endif


/**
 * Test value type
 */
template <bool IS_DESCENDING, typename KeyT, typename BeginOffsetIteratorT, typename EndOffsetIteratorT, typename NumItemsT>
void TestValueTypes(
    KeyT                 *h_keys,
    NumItemsT            num_items,
    int                  num_segments,
    bool                 pre_sorted,
    NumItemsT            *h_segment_offsets,
    BeginOffsetIteratorT d_segment_begin_offsets,
    EndOffsetIteratorT   d_segment_end_offsets,
    int                  begin_bit,
    int                  end_bit)
{
    // Initialize the solution
    NumItemsT *h_reference_ranks = NULL;
    KeyT *h_reference_keys = NULL;
    // If TEST_VALUE_TYPE == 0, no values are sorted, only keys.
    // Since ranks are only necessary when checking for values,
    // they are not computed in this case.
    InitializeSolution<IS_DESCENDING, TEST_VALUE_TYPE != 0>(h_keys, num_items, num_segments, pre_sorted, h_segment_offsets, begin_bit, end_bit, h_reference_ranks, h_reference_keys);

    TestBackend<IS_DESCENDING, KeyT, SmallestValueT>          (h_keys, num_items, num_segments, d_segment_begin_offsets, d_segment_end_offsets, begin_bit, end_bit, h_reference_keys, h_reference_ranks);

#if TEST_VALUE_TYPE == 3
    // Test with non-trivially-constructable value
    // These are cheap to build, so lump them in with the 64b value tests.
    TestBackend<IS_DESCENDING, KeyT, TestBar>           (h_keys, num_items, num_segments, d_segment_begin_offsets, d_segment_end_offsets, begin_bit, end_bit, h_reference_keys, h_reference_ranks);
#endif

    // Cleanup
    ResetKeys<IS_DESCENDING>(h_keys, num_items, pre_sorted, h_reference_keys);
    if (h_reference_ranks) delete[] h_reference_ranks;
    if (h_reference_keys) delete[] h_reference_keys;
}



/**
 * Test ascending/descending
 */
template <typename KeyT, typename BeginOffsetIteratorT, typename EndOffsetIteratorT,
          typename NumItemsT>
void TestDirection(
    KeyT                 *h_keys,
    NumItemsT            num_items,
    int                  num_segments,
    bool                 pre_sorted,
    NumItemsT            *h_segment_offsets,
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
template <typename KeyT, typename BeginOffsetIteratorT, typename EndOffsetIteratorT,
          typename NumItemsT>
void TestBits(
    KeyT                 *h_keys,
    NumItemsT            num_items,
    int                  num_segments,
    bool                 pre_sorted,
    NumItemsT            *h_segment_offsets,
    BeginOffsetIteratorT d_segment_begin_offsets,
    EndOffsetIteratorT   d_segment_end_offsets)
{
    // Don't test partial-word sorting for boolean, fp, or signed types (the bit-flipping techniques get in the way) or pre-sorted keys
    if ((Traits<KeyT>::CATEGORY == UNSIGNED_INTEGER)
        && (!std::is_same<KeyT, bool>::value)
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
template <typename KeyT, typename NumItemsT>
void TestSegmentIterators(
    KeyT           *h_keys,
    NumItemsT      num_items,
    int            num_segments,
    bool           pre_sorted,
    NumItemsT     *h_segment_offsets,
    NumItemsT     *d_segment_offsets)
{
    InitializeSegments(num_items, num_segments, h_segment_offsets);
    CubDebugExit(cudaMemcpy(d_segment_offsets, h_segment_offsets, sizeof(NumItemsT) * (num_segments + 1), cudaMemcpyHostToDevice));

    // Test with segment pointer.
    // This is also used to test non-segmented sort.
    TestBits(h_keys, num_items, num_segments, pre_sorted, h_segment_offsets, d_segment_offsets, d_segment_offsets + 1);

    if (num_segments > 1)
    {
        // Test with transform iterators of different types
        typedef TransformFunctor1<NumItemsT> TransformFunctor1T;
        typedef TransformFunctor2<NumItemsT> TransformFunctor2T;

        TransformInputIterator<NumItemsT, TransformFunctor1T, NumItemsT*, NumItemsT> d_segment_begin_offsets_itr(d_segment_offsets, TransformFunctor1T());
        TransformInputIterator<NumItemsT, TransformFunctor2T, NumItemsT*, NumItemsT> d_segment_end_offsets_itr(d_segment_offsets + 1, TransformFunctor2T());

        TestBits(h_keys, num_items, num_segments, pre_sorted, h_segment_offsets, d_segment_begin_offsets_itr, d_segment_end_offsets_itr);
    }
}


/**
 * Test different segment compositions
 */
template <typename KeyT, typename NumItemsT>
void TestSegments(
    KeyT         *h_keys,
    NumItemsT    num_items,
    int          max_segments,
    bool         pre_sorted)
{
    max_segments = static_cast<int>(CUB_MIN(num_items, static_cast<NumItemsT>(max_segments)));
    NumItemsT *h_segment_offsets = new NumItemsT[max_segments + 1];

    NumItemsT *d_segment_offsets = nullptr;
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_segment_offsets, sizeof(NumItemsT) * (max_segments + 1)));

    for (int num_segments = max_segments; num_segments > 1; num_segments = cub::DivideAndRoundUp(num_segments, 64))
    {
        // Pre-sorted tests are not supported for segmented sort
        if (num_items / num_segments < 128 * 1000 && !pre_sorted) {
            // Right now we assign a single thread block to each segment, so lets keep it to under 128K items per segment
            TestSegmentIterators(h_keys, num_items, num_segments, pre_sorted, h_segment_offsets, d_segment_offsets);
        }
    }

    // Test single segment
    if (num_items > 0)
    {
      if (num_items < 128 * 1000 || pre_sorted)
      {
        // Right now we assign a single thread block to each segment, so lets
        // keep it to under 128K items per segment
        TestSegmentIterators(h_keys,
                             num_items,
                             1,
                             pre_sorted,
                             h_segment_offsets,
                             d_segment_offsets);
      }
    }

    if (h_segment_offsets) delete[] h_segment_offsets;
    if (d_segment_offsets) CubDebugExit(g_allocator.DeviceFree(d_segment_offsets));
}

/** 
 * Test different NumItemsT, i.e. types of num_items 
 */
template <typename KeyT>
void TestNumItems(KeyT         *h_keys,
                  std::size_t  num_items,
                  int          max_segments,
                  bool         pre_sorted)
{
    if (!pre_sorted && num_items <= std::size_t(std::numeric_limits<int>::max()))
    {
        TestSegments<KeyT, int>(h_keys, static_cast<int>(num_items), max_segments, pre_sorted);
    }
    if (pre_sorted && num_items <= std::size_t(std::numeric_limits<std::uint32_t>::max()))
    {
        TestSegments<KeyT, std::uint32_t>(h_keys, static_cast<std::uint32_t>(num_items), max_segments, pre_sorted);
    }
    TestSegments<KeyT, std::size_t>(h_keys, num_items, max_segments, pre_sorted);
}


/**
 * Test different (sub)lengths and number of segments
 */
template <typename KeyT>
void TestSizes(KeyT* h_keys,
               std::size_t max_items,
               int max_segments,
               bool pre_sorted)
{
    if (pre_sorted)
    {
        // run a specific list of sizes, up to max_items
        std::size_t sizes[] = {g_smallest_pre_sorted_num_items, 4350000007ull};
        for (std::size_t num_items : sizes)
        {
            if (num_items > max_items) break;
            TestNumItems(h_keys, num_items, max_segments, pre_sorted);
        }
    }
    else
    {
        for (std::size_t num_items = max_items;
             num_items > 1;
             num_items = cub::DivideAndRoundUp(num_items, 64))
        {
            TestNumItems(h_keys, num_items, max_segments, pre_sorted);
        }
    }
}

/**
 * Test key sampling distributions
 */
template <typename KeyT, bool WITH_PRE_SORTED>
void TestGen(
    std::size_t     max_items,
    int             max_segments)
{
    if (max_items == ~std::size_t(0))
    {
        max_items = 9000003;
    }

    if (max_segments < 0)
    {
        max_segments = 5003;
    }

    std::unique_ptr<KeyT[]> h_keys(new KeyT[max_items]);

    // Test trivial problems sizes
    h_keys[0] = static_cast<KeyT>(42);
    TestNumItems(h_keys.get(), 0, 0, false);
    TestNumItems(h_keys.get(), 1, 1, false);

    for (int entropy_reduction = 0; entropy_reduction <= 6; entropy_reduction += 6)
    {
        printf("\nTesting random %s keys with entropy reduction factor %d\n", typeid(KeyT).name(), entropy_reduction); fflush(stdout);
        InitializeKeyBits(RANDOM, h_keys.get(), max_items, entropy_reduction);
        TestSizes(h_keys.get(), max_items, max_segments, false);
    }

    if (cub::Traits<KeyT>::CATEGORY == cub::FLOATING_POINT)
    {
        printf("\nTesting random %s keys with some replaced with -0.0 or +0.0 \n", typeid(KeyT).name());
        fflush(stdout);
        InitializeKeyBits(RANDOM_MINUS_PLUS_ZERO, h_keys.get(), max_items, 0);
        // This just tests +/- 0 handling -- don't need to test multiple sizes
        TestNumItems(h_keys.get(), max_items, max_segments, false);
    }

    printf("\nTesting uniform %s keys\n", typeid(KeyT).name()); fflush(stdout);
    InitializeKeyBits(UNIFORM, h_keys.get(), max_items, 0);
    TestSizes(h_keys.get(), max_items, max_segments, false);

    printf("\nTesting natural number %s keys\n", typeid(KeyT).name()); fflush(stdout);
    InitializeKeyBits(INTEGER_SEED, h_keys.get(), max_items, 0);
    TestSizes(h_keys.get(), max_items, max_segments, false);

    if (WITH_PRE_SORTED)
    {
        // Presorting is only used for testing large input arrays.
        const std::size_t large_num_items = std::size_t(4350000007ull);

        // A conservative check for memory, as we don't know ValueT or whether
        // the overwrite is allowed until later.
        // For ValueT, the check is actually exact unless TEST_VALUE_TYPE == 3.
        if (!HasEnoughMemory<KeyT, SmallestValueT>(large_num_items, true))
        {
            printf("Skipping the permutation-based test due to insufficient device memory\n");
            return;
        }

        h_keys.reset(nullptr); // Explicitly free old buffer before allocating.
        h_keys.reset(new KeyT[large_num_items]);

        printf("\nTesting pre-sorted and randomly permuted %s keys\n", typeid(KeyT).name());
        fflush(stdout);
        InitializeKeysSorted(h_keys.get(), large_num_items);
        fflush(stdout);
        TestSizes(h_keys.get(), large_num_items, max_segments, true);
        fflush(stdout);
    }

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
    const bool KEYS_ONLY = std::is_same<ValueT, NullType>::value;

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
    InitializeSolution<IS_DESCENDING, !KEYS_ONLY>(
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
    std::size_t num_items = ~std::size_t(0);
    int num_segments = -1;

    // Initialize command line
    CommandLineArgs args(argc, argv);
    g_verbose = args.CheckCmdLineFlag("v");
    args.GetCmdLineArgument("n", num_items);
    args.GetCmdLineArgument("s", num_segments);
    args.GetCmdLineArgument("i", g_timing_iterations);

    // Print usage
    if (args.CheckCmdLineFlag("help"))
    {
        printf("%s "
            "[--n=<input items> "
            "[--s=<num segments> "
            "[--i=<timing iterations> "
            "[--device=<device-id>] "
            "[--v] "
            "\n", argv[0]);
        exit(0);
    }

    // Initialize device
    CubDebugExit(args.DeviceInit());

    // %PARAM% TEST_KEY_BYTES bytes 1:2:4:8
    // %PARAM% TEST_VALUE_TYPE pairs 0:1:2:3
    //   0->Keys only
    //   1->uchar
    //   2->uint
    //   3->[ull,TestBar] (TestBar is cheap to build, included here to
    //                     reduce total number of targets)

    // To reduce testing time, some key types are only tested when not
    // testing pairs:
#if TEST_VALUE_TYPE == 0
#define TEST_EXTENDED_KEY_TYPES
#endif

    // Compile/run thorough tests
#if TEST_KEY_BYTES == 1

    TestGen<char, true>               (num_items, num_segments);

#ifdef TEST_EXTENDED_KEY_TYPES
    TestGen<bool, false>              (num_items, num_segments);
    TestGen<signed char, false>       (num_items, num_segments);
    TestGen<unsigned char, false>     (num_items, num_segments);
#endif // TEST_EXTENDED_KEY_TYPES

#elif TEST_KEY_BYTES == 2
    TestGen<unsigned short, true>     (num_items, num_segments);

#ifdef TEST_EXTENDED_KEY_TYPES
    TestGen<short, false>             (num_items, num_segments);

#if (__CUDACC_VER_MAJOR__ >= 9 || CUDA_VERSION >= 9000) && !_NVHPC_CUDA
    TestGen<half_t, false>            (num_items, num_segments);
#endif // CTK >= 9

#if (__CUDACC_VER_MAJOR__ >= 11 || CUDA_VERSION >= 11000) && !_NVHPC_CUDA
#if !defined(__ICC)
    // Fails with `-0 != 0` with ICC for unknown reasons. See #333.
    TestGen<bfloat16_t, false>        (num_items, num_segments);
#endif // !ICC
#endif // CTK >= 11

#endif // TEST_EXTENDED_KEY_TYPES

#elif TEST_KEY_BYTES == 4

    TestGen<int, true>                (num_items, num_segments);

#ifdef TEST_EXTENDED_KEY_TYPES
    TestGen<float, false>             (num_items, num_segments);
    TestGen<unsigned int, false>      (num_items, num_segments);
#endif // TEST_EXTENDED_KEY_TYPES

#elif TEST_KEY_BYTES == 8

    TestGen<double, true>             (num_items, num_segments);

#ifdef TEST_EXTENDED_KEY_TYPES
    TestGen<long long, false>         (num_items, num_segments);
    TestGen<unsigned long long, false>(num_items, num_segments);
#endif // TEST_EXTENDED_KEY_TYPES

#endif // TEST_KEY_BYTES switch

    return 0;
}
