/******************************************************************************
 * Copyright (c) 2011-2021, NVIDIA CORPORATION.  All rights reserved.
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

#pragma once

#include "../config.cuh"
#include "../util_type.cuh"
#include "../util_namespace.cuh"
#include "../block/block_radix_sort.cuh"
#include "../agent/agent_radix_sort_upsweep.cuh"
#include "../agent/agent_radix_sort_downsweep.cuh"

CUB_NAMESPACE_BEGIN


template <
  bool      IS_DESCENDING,                  ///< Whether or not the sorted-order is high-to-low
  typename  SegmentedPolicyT,               ///< Chained tuning policy
  typename  KeyT,                           ///< Key type
  typename  ValueT,                         ///< Value type
  typename  OffsetT>                        ///< Signed integer type for global offsets
struct AgentSegmentedRadixSort
{
  OffsetT segment_begin;
  OffsetT segment_end;
  OffsetT num_items;

  static constexpr int ITEMS_PER_THREAD = SegmentedPolicyT::ITEMS_PER_THREAD;
  static constexpr int BLOCK_THREADS    = SegmentedPolicyT::BLOCK_THREADS;
  static constexpr int RADIX_BITS       = SegmentedPolicyT::RADIX_BITS;
  static constexpr int RADIX_DIGITS     = 1 << RADIX_BITS;
  static constexpr int KEYS_ONLY        = Equals<ValueT, NullType>::VALUE;

  // Huge segment handlers
  using BlockUpsweepT = AgentRadixSortUpsweep<SegmentedPolicyT, KeyT, OffsetT>;
  using DigitScanT    = BlockScan<OffsetT, BLOCK_THREADS>;
  using BlockDownsweepT =
    AgentRadixSortDownsweep<SegmentedPolicyT, IS_DESCENDING, KeyT, ValueT, OffsetT>;

  /// Number of bin-starting offsets tracked per thread
  static constexpr int BINS_TRACKED_PER_THREAD = BlockDownsweepT::BINS_TRACKED_PER_THREAD;

  // Small segment handlers
  using BlockRadixSortT =
    BlockRadixSort<KeyT,
                   BLOCK_THREADS,
                   ITEMS_PER_THREAD,
                   ValueT,
                   RADIX_BITS,
                   (SegmentedPolicyT::RANK_ALGORITHM == RADIX_RANK_MEMOIZE),
                   SegmentedPolicyT::SCAN_ALGORITHM>;

  using BlockKeyLoadT = BlockLoad<KeyT,
                                  BLOCK_THREADS,
                                  ITEMS_PER_THREAD,
                                  SegmentedPolicyT::LOAD_ALGORITHM>;

  using BlockValueLoadT = BlockLoad<ValueT,
                                    BLOCK_THREADS,
                                    ITEMS_PER_THREAD,
                                    SegmentedPolicyT::LOAD_ALGORITHM>;

  union _TempStorage
  {
    // Huge segment handlers
    typename BlockUpsweepT::TempStorage upsweep;
    typename BlockDownsweepT::TempStorage downsweep;

    struct UnboundBlockSort
    {
      volatile OffsetT reverse_counts_in[RADIX_DIGITS];
      volatile OffsetT reverse_counts_out[RADIX_DIGITS];
      typename DigitScanT::TempStorage scan;
    } unbound_sort;

    // Small segment handlers
    typename BlockKeyLoadT::TempStorage keys_load;
    typename BlockValueLoadT::TempStorage values_load;
    typename BlockRadixSortT::TempStorage sort;
  };

  using TempStorage = Uninitialized<_TempStorage>;
  _TempStorage &temp_storage;

  __device__ __forceinline__
  AgentSegmentedRadixSort(OffsetT segment_begin,
                          OffsetT segment_end,
                          OffsetT num_items,
                          TempStorage &temp_storage)
      : segment_begin(segment_begin)
      , segment_end(segment_end)
      , num_items(num_items)
      , temp_storage(temp_storage.Alias())
  {}

  __device__ __forceinline__ void ProcessSmallSegment(int begin_bit,
                                                      int end_bit,
                                                      const KeyT *d_keys_in,
                                                      const ValueT *d_values_in,
                                                      KeyT *d_keys_out,
                                                      ValueT *d_values_out)
  {
    KeyT thread_keys[ITEMS_PER_THREAD];
    ValueT thread_values[ITEMS_PER_THREAD];

    // For FP64 the difference is:
    // Lowest() -> -1.79769e+308 = 00...00b -> TwiddleIn -> -0 = 10...00b
    // LOWEST   -> -nan          = 11...11b -> TwiddleIn ->  0 = 00...00b

    using UnsignedBitsT = typename Traits<KeyT>::UnsignedBits;
    UnsignedBitsT default_key_bits = IS_DESCENDING ? Traits<KeyT>::LOWEST_KEY
                                                   : Traits<KeyT>::MAX_KEY;
    KeyT oob_default = reinterpret_cast<KeyT &>(default_key_bits);

    if (!KEYS_ONLY)
    {
      BlockValueLoadT(temp_storage.values_load).Load(
        d_values_in + segment_begin,
        thread_values,
        num_items);

      CTA_SYNC();
    }

    {
      BlockKeyLoadT(temp_storage.keys_load).Load(
        d_keys_in + segment_begin,
        thread_keys,
        num_items,
        oob_default);

      CTA_SYNC();
    }

    BlockRadixSortT(temp_storage.sort).SortBlockedToStriped(
      thread_keys,
      thread_values,
      begin_bit,
      end_bit,
      Int2Type<IS_DESCENDING>(),
      Int2Type<KEYS_ONLY>());

    // Store keys and values
    d_keys_out   += segment_begin;
    d_values_out += segment_begin;

    #pragma unroll
    for (int item = 0; item < ITEMS_PER_THREAD; item++)
    {
      int item_offset = item * BLOCK_THREADS + threadIdx.x;

      if (item_offset < num_items)
      {
        d_keys_out[item_offset] = thread_keys[item];

        if (!KEYS_ONLY)
        {
          d_values_out[item_offset] = thread_values[item];
        }
      }
    }
  }

  __device__ __forceinline__ void
  ProcessLargeSegment(int current_bit,
                      int pass_bits,
                      const KeyT *d_keys_in,
                      const ValueT *d_values_in,
                      KeyT *d_keys_out,
                      ValueT *d_values_out)
  {
    // Upsweep
    BlockUpsweepT upsweep(temp_storage.upsweep,
                          d_keys_in,
                          current_bit,
                          pass_bits);
    upsweep.ProcessRegion(segment_begin, segment_end);

    CTA_SYNC();

    // The count of each digit value in this pass (valid in the first RADIX_DIGITS threads)
    OffsetT bin_count[BINS_TRACKED_PER_THREAD];
    upsweep.ExtractCounts(bin_count);

    CTA_SYNC();

    if (IS_DESCENDING)
    {
      // Reverse bin counts
      #pragma unroll
      for (int track = 0; track < BINS_TRACKED_PER_THREAD; ++track)
      {
        int bin_idx = (threadIdx.x * BINS_TRACKED_PER_THREAD) + track;

        if ((BLOCK_THREADS == RADIX_DIGITS) || (bin_idx < RADIX_DIGITS))
        {
          temp_storage.unbound_sort.reverse_counts_in[bin_idx] = bin_count[track];
        }
      }

      CTA_SYNC();

      #pragma unroll
      for (int track = 0; track < BINS_TRACKED_PER_THREAD; ++track)
      {
        int bin_idx = (threadIdx.x * BINS_TRACKED_PER_THREAD) + track;

        if ((BLOCK_THREADS == RADIX_DIGITS) || (bin_idx < RADIX_DIGITS))
        {
          bin_count[track] = temp_storage.unbound_sort.reverse_counts_in[RADIX_DIGITS - bin_idx - 1];
        }
      }
    }

    // Scan
    OffsetT bin_offset[BINS_TRACKED_PER_THREAD]; // The global scatter base offset for each digit value in this pass (valid in the first RADIX_DIGITS threads)
    DigitScanT(temp_storage.unbound_sort.scan).ExclusiveSum(bin_count, bin_offset);

    #pragma unroll
    for (int track = 0; track < BINS_TRACKED_PER_THREAD; ++track)
    {
      bin_offset[track] += segment_begin;
    }

    if (IS_DESCENDING)
    {
      // Reverse bin offsets
      #pragma unroll
      for (int track = 0; track < BINS_TRACKED_PER_THREAD; ++track)
      {
        int bin_idx = (threadIdx.x * BINS_TRACKED_PER_THREAD) + track;

        if ((BLOCK_THREADS == RADIX_DIGITS) || (bin_idx < RADIX_DIGITS))
        {
          temp_storage.unbound_sort.reverse_counts_out[threadIdx.x] = bin_offset[track];
        }
      }

      CTA_SYNC();

      #pragma unroll
      for (int track = 0; track < BINS_TRACKED_PER_THREAD; ++track)
      {
        int bin_idx = (threadIdx.x * BINS_TRACKED_PER_THREAD) + track;

        if ((BLOCK_THREADS == RADIX_DIGITS) || (bin_idx < RADIX_DIGITS))
        {
          bin_offset[track] = temp_storage.unbound_sort.reverse_counts_out[RADIX_DIGITS - bin_idx - 1];
        }
      }
    }

    CTA_SYNC();

    // Downsweep
    BlockDownsweepT downsweep(temp_storage.downsweep,
                              bin_offset,
                              num_items,
                              d_keys_in,
                              d_keys_out,
                              d_values_in,
                              d_values_out,
                              current_bit,
                              pass_bits);
    downsweep.ProcessRegion(segment_begin, segment_end);
  }
};


CUB_NAMESPACE_END
