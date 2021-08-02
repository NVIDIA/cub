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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 *IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
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
#include "../util_ptx.cuh"
#include "../util_type.cuh"
#include "block_scan.cuh"
#include <limits>

/// Optional outer namespace(s)
CUB_NS_PREFIX

/// CUB namespace
namespace cub
{

/**
 * BlockRunLengthDecodeAlgorithm enumerates alternative algorithms for parallel
 * reduction across a CUDA thread block.
 */
enum class BlockRunLengthDecodeAlgorithm
{
  /// If there's an unused item (amongst items representable by UniqueItemT) that will never be seen amongst the unique
  /// items, the algorithm can internally use such unused item and provide a more efficient implementation that also
  /// requires less shared memory
  UNUSED,

  /// This is a regular run-length decoding implementation
  NORMAL,

  /// This implementation also provides information about the relative offset within each run.
  /// E.g., if 4, 4, 4, 1, 7, 7 was decoded, the the relative offsets are: 0, 1, 2, 0, 0, 1
  OFFSETS,
};

/**
 * @brief
 * TODO: provide detailed desdcription on usage and examples
 *
 * @tparam UniqueItemT The data type of the
 * @tparam BLOCK_DIM_X The thread block length in threads along the X dimension
 * @tparam RUNS_PER_THREAD The number of consecutive runs that each thread contributes
 * @tparam DECODED_ITEMS_PER_THREAD The maximum number of decoded items that each thread holds
 * @tparam RelativeOffsetT Type used to index into the block's decoded items (large enough to hold the sum over all the
 * runs' lengths)
 * @tparam BLOCK_DIM_Y The thread block length in threads along the Y dimension
 * @tparam BLOCK_DIM_Z The thread block length in threads along the Z dimension
 */
template <typename UniqueItemT,
          int BLOCK_DIM_X,
          int RUNS_PER_THREAD,
          int DECODED_ITEMS_PER_THREAD,
          BlockRunLengthDecodeAlgorithm DECODE_ALGORITHM = BlockRunLengthDecodeAlgorithm::NORMAL,
          typename RelativeOffsetT                       = uint32_t,
          int BLOCK_DIM_Y                                = 1,
          int BLOCK_DIM_Z                                = 1>
class BlockRunLengthDecode
{
  //---------------------------------------------------------------------
  // CONFIGS & TYPE ALIASES
  //---------------------------------------------------------------------
private:
  /// The thread block size in threads
  static constexpr int BLOCK_THREADS = BLOCK_DIM_X * BLOCK_DIM_Y * BLOCK_DIM_Z;

  /// The number of runs that the block decodes (out-of-bounds items may be padded with run lengths of '0')
  static constexpr int BLOCK_RUNS = BLOCK_THREADS * RUNS_PER_THREAD;

  /// The number of decoded items. If the actually run-length decoded items exceed BLOCK_DECODED_ITEMS, the user can
  /// retrieve the full run-length decoded data through multiple invocations.
  static constexpr int BLOCK_DECODED_ITEMS = BLOCK_THREADS * DECODED_ITEMS_PER_THREAD;

  /// BlockScan used to determine the beginning of each run (i.e., prefix sum over the runs' length)
  using RunOffsetScanT =
    cub::BlockScan<RelativeOffsetT, BLOCK_DIM_X, BLOCK_SCAN_RAKING_MEMOIZE, BLOCK_DIM_Y, BLOCK_DIM_Z>;

  /// Data type used to either
  /// (a) distinguish between beginning-of-run versus continuation-of-run or
  /// (b) track the run offset within a run but also distinguish between beginning-of-run versus continuation-of-run
  using RelativeRunOffsetT = uint32_t;

  /**
   * [SPECIALIZATION: NORMAL OR OFFSETS ]
   * Helper data structure to facilitate run-length decoding using a prefix scan
   */
  struct ZippedRunLengthDecode
  {
    // holding the run-length decoded items
    UniqueItemT unique;
    // the offset within the current run
    RelativeRunOffsetT offset;
  };

  /**
   * [SPECIALIZATION: NORMAL OR OFFSETS ]
   * Helper data structure to facilitate scattering <beginning-of-run> into a shared memory buffer that is
   * pre-initialized with <continuation-of-run> items. Prior to utilising the prefix scan to perform the run-length
   * decoding.
   */
  struct UnzippedRunLengthDecodeBuffer
  {
    UniqueItemT uniques[BLOCK_DECODED_ITEMS];
    RelativeRunOffsetT offsets[BLOCK_DECODED_ITEMS];
  };

  /**
   * [SPECIALIZATION: UNUSED ]
   */
  struct DecodedBuffer
  {
    UniqueItemT uniques[BLOCK_DECODED_ITEMS];
  };

  /// The bit-offset of the is-unresolved bit-flag that indicates whether this item within the decode buffer still needs
  /// to be resolved (i.e., yet-to-be-populated with the repetition of the correct unique item)
  static constexpr RelativeRunOffsetT IS_UNRESOLVED_BIT_OFFSET = (8U * sizeof(RelativeRunOffsetT)) - 1;

  /// Bit-mask to check the is-unresolved bit-flag
  static constexpr RelativeRunOffsetT IS_UNRESOLVED_BIT = 0x01U << IS_UNRESOLVED_BIT_OFFSET;

  /// Bit-mask to extract the item's offset relative to its run
  /// E.g., if 4, 4, 4, 1, 7, 7 was decoded, the the relative offsets are: 0, 1, 2, 0, 0, 1
  static constexpr RelativeRunOffsetT OFFSET_PAYLOAD_BIT_MASK = IS_UNRESOLVED_BIT - 1U;

  /// Bit-mask to extract the item's offset relative to its run
  static constexpr RelativeRunOffsetT IS_UNRESOLVED_ITEM = IS_UNRESOLVED_BIT | 0x01U;

  /**
   * [SPECIALIZATION: OFFSETS ]
   */
  struct DecodeScanOpWithRelativeOffset
  {
    __device__ __forceinline__ ZippedRunLengthDecode operator()(const ZippedRunLengthDecode &lhs,
                                                                const ZippedRunLengthDecode &rhs)
    {
      // If the rhs is still unresolved, propagate the lhs (which may or may not be resolved _at this point_, but
      // eventually _some_ lhs will provide the answer)
      return cub::BFE(rhs.offset, IS_UNRESOLVED_BIT_OFFSET, 1U)
               ? ZippedRunLengthDecode{lhs.unique, lhs.offset + (rhs.offset & OFFSET_PAYLOAD_BIT_MASK)}
               : ZippedRunLengthDecode{rhs.unique, rhs.offset};
    }
  };

  /**
   * [SPECIALIZATION: NORMAL ]
   */
  struct DecodeScanNormal
  {
    __device__ __forceinline__ ZippedRunLengthDecode operator()(const ZippedRunLengthDecode &lhs,
                                                                const ZippedRunLengthDecode &rhs)
    {
      // If the rhs is still unresolved, propagate the lhs (which may or may not be resolved _at this point_, but
      // eventually _some_ lhs will provide the answer)
      return rhs.offset ? lhs : rhs;
    }
  };

  /// The actual prefix scan oeprator being used for the run-length decoding
  using DecodeScanOpT = typename cub::If<(DECODE_ALGORITHM == BlockRunLengthDecodeAlgorithm::NORMAL),
                                         DecodeScanNormal,
                                         DecodeScanOpWithRelativeOffset>::Type;

  /// The meta item type that is used to distinguish between <already-run-length-decoded> item and
  /// <yet-to-be-resolved> items. If the run-length decode may use a unique unused item from the set representable by
  /// UniqueItemT, we can simply use UniqueItemT. Otherwise, we'll have to use a meta type to distinguish between the
  /// two item types.
  using MetaItemT = typename cub::
    If<(DECODE_ALGORITHM == BlockRunLengthDecodeAlgorithm::UNUSED), UniqueItemT, ZippedRunLengthDecode>::Type;

  using DecodedBufferT = typename cub::
    If<(DECODE_ALGORITHM == BlockRunLengthDecodeAlgorithm::UNUSED), DecodedBuffer, UnzippedRunLengthDecodeBuffer>::Type;

  /// BlockScan used to replace <continuation-of-run> items with the run that they're a continuation of
  using DecodeScanT = cub::BlockScan<MetaItemT, BLOCK_DIM_X, BLOCK_SCAN_WARP_SCANS, BLOCK_DIM_Y, BLOCK_DIM_Z>;

  /// Type used to index into the block's runs
  using RunOffsetT = uint32_t;

  /// Shared memory type required by this thread block
  union _TempStorage
  {
    typename RunOffsetScanT::TempStorage offset_scan;
    struct
    {
      UniqueItemT unique_items[BLOCK_RUNS];
      RelativeOffsetT run_offsets[BLOCK_RUNS + 1];
      union
      {
        typename DecodeScanT::TempStorage decode_scan;
        DecodedBufferT decode_buffer;
      };
    } runs;
  }; // union TempStorage

  /// Internal storage allocator (used when the user does not provide pre-allocated shared memory)
  __device__ __forceinline__ _TempStorage &PrivateStorage()
  {
    __shared__ _TempStorage private_storage;
    return private_storage;
  }

  /// Shared storage reference
  _TempStorage &temp_storage;

  /// Linear thread-id
  uint32_t linear_tid;

  //---------------------------------------------------------------------
  // HELPERS: SPECIALIZATIONS IF ALGORITHM IS NORMAL OR OFFSETS
  //---------------------------------------------------------------------
  /**
   * [SPECIALIZATION: NORMAL OR OFFSETS]
   * Populates the run-length decode buffer with <continuation-of-current-run> items
   */
  __device__ __forceinline__ void InitDecodeBuffer(UnzippedRunLengthDecodeBuffer &decode_buffer)
  {
    RelativeOffsetT buffer_offset = linear_tid;
#pragma unroll
    for (uint32_t i = 0; i < DECODED_ITEMS_PER_THREAD; i++)
    {
      decode_buffer.offsets[buffer_offset] = IS_UNRESOLVED_ITEM;
      buffer_offset += BLOCK_THREADS;
    }
  }

  /**
   * [SPECIALIZATION: NORMAL OR OFFSETS]
   * Populates the run-length decode buffer with <continuation-of-current-run> items
   */
  template <typename DecodeBufferOffsetT>
  __device__ __forceinline__ void WriteBeginningOfRun(UnzippedRunLengthDecodeBuffer &decode_buffer,
                                                      DecodeBufferOffsetT buffer_offset,
                                                      const UniqueItemT &unique_item,
                                                      const RelativeRunOffsetT &relative_offset)
  {
    decode_buffer.uniques[buffer_offset] = unique_item;
    decode_buffer.offsets[buffer_offset] = relative_offset;
  }

  /**
   * [SPECIALIZATION: NORMAL OR OFFSETS]
   * Populates the run-length decode buffer with <continuation-of-current-run> items
   */
  __device__ __forceinline__ void LoadFromDecodeBuffer(ZippedRunLengthDecode (&scan_items)[DECODED_ITEMS_PER_THREAD],
                                                       UnzippedRunLengthDecodeBuffer &decode_buffer)
  {
    RelativeOffsetT run_offset = linear_tid * DECODED_ITEMS_PER_THREAD;
#pragma unroll
    for (uint32_t i = 0; i < DECODED_ITEMS_PER_THREAD; i++)
    {
      scan_items[i].unique = decode_buffer.uniques[run_offset];
      scan_items[i].offset = decode_buffer.offsets[run_offset];
      run_offset++;
    }
  }

  /**
   * [SPECIALIZATION: NORMAL OR OFFSETS]
   * Populate registers (SOA) to return the results to the user
   */
  template <typename UserRelativeOffsetT>
  __device__ __forceinline__ void PopulateReturnValues(ZippedRunLengthDecode (&scan_items)[DECODED_ITEMS_PER_THREAD],
                                                       UniqueItemT (&uniques)[DECODED_ITEMS_PER_THREAD],
                                                       UserRelativeOffsetT (&relative_offset)[DECODED_ITEMS_PER_THREAD])
  {
#pragma unroll
    for (uint32_t i = 0; i < DECODED_ITEMS_PER_THREAD; i++)
    {
      uniques[i]         = scan_items[i].unique;
      relative_offset[i] = scan_items[i].offset;
    }
  }

public:
  struct TempStorage : Uninitialized<_TempStorage>
  {};

  //---------------------------------------------------------------------
  // CONSTRUCTOR
  //---------------------------------------------------------------------
  __device__ __forceinline__ BlockRunLengthDecode()
      : temp_storage(PrivateStorage())
      , linear_tid(RowMajorTid(BLOCK_DIM_X, BLOCK_DIM_Y, BLOCK_DIM_Z))
  {}

  __device__ __forceinline__ BlockRunLengthDecode(TempStorage &temp_storage)
      : temp_storage(temp_storage.Alias())
      , linear_tid(RowMajorTid(BLOCK_DIM_X, BLOCK_DIM_Y, BLOCK_DIM_Z))
  {}

  /**
   * @brief Initializes the run-length decoding instances with the given sequence of runs and returns the total
   * run-length decoded size. Subsequent calls to <b>RunLengthDecode</b> can be used to retrieve the run-length
   * decoded data.
   */
  template <typename RunLengthT, typename OffsetT>
  __device__ __forceinline__ void Init(UniqueItemT (&unique_items)[RUNS_PER_THREAD],
                                       RunLengthT (&run_lengths)[RUNS_PER_THREAD],
                                       OffsetT &total_decoded_size)
  {
    // Compute the offset for the beginning of each run
    RelativeOffsetT run_offsets[RUNS_PER_THREAD];
#pragma unroll
    for (uint32_t i = 0; i < RUNS_PER_THREAD; i++)
    {
      run_offsets[i] = run_lengths[i];
    }
    RelativeOffsetT decoded_size_aggregate;
    RunOffsetScanT(temp_storage.offset_scan).ExclusiveSum(run_offsets, run_offsets, decoded_size_aggregate);
    total_decoded_size = decoded_size_aggregate;

    // Ensure the prefix scan's temporary storage can be reused (may be superfluous, but depends on scan implementaiton)
    __syncthreads();

    // Keep the runs' unique items and the offsets of each run's beginning in the temporary storage
    RunOffsetT thread_dst_offset = linear_tid * RUNS_PER_THREAD;
#pragma unroll
    for (uint32_t i = 0; i < RUNS_PER_THREAD; i++)
    {
      temp_storage.runs.unique_items[thread_dst_offset] = unique_items[i];
      temp_storage.runs.run_offsets[thread_dst_offset]  = run_offsets[i];
      thread_dst_offset++;
    }

    // Write the runs' total decoded size into the last element
    if (linear_tid == 0)
    {
      temp_storage.runs.run_offsets[BLOCK_RUNS] = total_decoded_size;
    }

    // Ensure run offsets have been writen to shared memory
    __syncthreads();
  }

  /**
   * @brief Run-length decodes the runs previously passed via a call to Init(...) and returns the run-length decoded
   * items in a blocked arrangement to \p decoded_items. If the number of run-length decoded items exceeds the
   * run-length decode buffer (i.e., <b>DECODED_ITEMS_PER_THREAD * BLOCK_THREADS</b>), only the items that fit within
   * the buffer are returned. Subsequent calls to <b>RunLengthDecode</b> adjusting \p from_decoded_offset can be
   * used to retrieve the remaining run-length decoded items.
   *
   * @param from_decoded_offset If invoked with from_decoded_offset that is larger than total_decoded_size results in
   * undefined behavior.
   */
  template <typename UserRelativeOffsetT>
  __device__ __forceinline__ void RunLengthDecode(UniqueItemT (&decoded_items)[DECODED_ITEMS_PER_THREAD],
                                                  UserRelativeOffsetT (&item_offsets)[DECODED_ITEMS_PER_THREAD],
                                                  int32_t from_decoded_offset = 0)
  {
    // Populate run-length decode buffer with continuation-of-current-run items
    InitDecodeBuffer(temp_storage.runs.decode_buffer);

    // Ensure that the continuation-of-current-run items have been written to shared memory
    __syncthreads();

    // Scatter the beginning of each run into the run-length decode buffer
    RunOffsetT run_offset = linear_tid;
#pragma unroll
    for (uint32_t i = 0; i < RUNS_PER_THREAD; i++)
    {
      // Fetch this thread's run and subsequently check whether it falls into this pass' run-length decode buffer
      int32_t run_target_idx      = temp_storage.runs.run_offsets[run_offset];
      int32_t next_run_target_idx = temp_storage.runs.run_offsets[run_offset + 1];
      int32_t run_length          = next_run_target_idx - run_target_idx;

      // Whether this run begins before the from_decoded_offset but has trailing items that still fall within the
      // beginning of this buffer
      bool is_a_trailing_run = (run_target_idx < from_decoded_offset) && (next_run_target_idx > from_decoded_offset);

      // In particular for the trailing run, we want to scatter into index 0 (not a negative scatter index)
      int32_t scatter_target_idx = max(0, run_target_idx - from_decoded_offset);
      int32_t relative_offset    = is_a_trailing_run ? from_decoded_offset - run_target_idx : 0;

      // Only write this beginning-of-run if the run length falls within the decoded buffer OR
      // the run is a trailing run.
      if (run_length > 0 && (run_target_idx >= from_decoded_offset || is_a_trailing_run) &&
          scatter_target_idx < BLOCK_DECODED_ITEMS)
      {
        WriteBeginningOfRun(temp_storage.runs.decode_buffer,
                            scatter_target_idx,
                            temp_storage.runs.unique_items[run_offset],
                            relative_offset);
      }
      run_offset += BLOCK_THREADS;
    }

    // Ensure that the beginning of all relevant runs has been written
    __syncthreads();

    // Read in blocked-arrangement in preparation for the run-length decoding prefix scan
    ZippedRunLengthDecode scan_items[DECODED_ITEMS_PER_THREAD];
    LoadFromDecodeBuffer(scan_items, temp_storage.runs.decode_buffer);

    // Ensure the decoded_buffer can be repurposed for the scan's temporary storage
    __syncthreads();

    // Perform the run-length decoding prefix scan (populating continuation-of-runs with the correct run's value)
    DecodeScanT(temp_storage.runs.decode_scan).InclusiveScan(scan_items, scan_items, DecodeScanOpT());

    // Prepare result to return it to the user within user-provided registers
    PopulateReturnValues(scan_items, decoded_items, item_offsets);

    // Ensure we're done using the the temporary storage (e.g., before the user is invoking another round of
    // RunLengthDecode(...))
    __syncthreads();
  }
};
} // namespace cub
CUB_NS_POSTFIX // Optional outer namespace(s)