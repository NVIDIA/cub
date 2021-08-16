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
#include "../thread/thread_search.cuh"
#include "../util_ptx.cuh"
#include "../util_type.cuh"
#include "block_scan.cuh"
#include <limits>

/// CUB namespace
CUB_NAMESPACE_BEGIN

/**
 * BlockRunLengthDecodeAlgorithm enumerates alternative specialisations of the BlockRunLengthDecode algorithm.
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
 * @tparam DecodedOffsetT Type used to index into the block's decoded items (large enough to hold the sum over all the
 * runs' lengths)
 * @tparam BLOCK_DIM_Y The thread block length in threads along the Y dimension
 * @tparam BLOCK_DIM_Z The thread block length in threads along the Z dimension
 */
template <typename UniqueItemT,
          int BLOCK_DIM_X,
          int RUNS_PER_THREAD,
          int DECODED_ITEMS_PER_THREAD,
          BlockRunLengthDecodeAlgorithm DECODE_ALGORITHM = BlockRunLengthDecodeAlgorithm::NORMAL,
          typename DecodedOffsetT                        = uint32_t,
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
    cub::BlockScan<DecodedOffsetT, BLOCK_DIM_X, BLOCK_SCAN_RAKING_MEMOIZE, BLOCK_DIM_Y, BLOCK_DIM_Z>;

  /// Type used to index into the block's runs
  using RunOffsetT = uint32_t;

  /// Shared memory type required by this thread block
  union _TempStorage
  {
    typename RunOffsetScanT::TempStorage offset_scan;
    struct
    {
      UniqueItemT unique_items[BLOCK_RUNS];
      DecodedOffsetT run_offsets[BLOCK_RUNS + 1];
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
    DecodedOffsetT run_offsets[RUNS_PER_THREAD];
#pragma unroll
    for (uint32_t i = 0; i < RUNS_PER_THREAD; i++)
    {
      run_offsets[i] = run_lengths[i];
    }
    DecodedOffsetT decoded_size_aggregate;
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
  template <typename RelativeOffsetT>
  __device__ __forceinline__ void RunLengthDecode(UniqueItemT (&decoded_items)[DECODED_ITEMS_PER_THREAD],
                                                  RelativeOffsetT (&item_offsets)[DECODED_ITEMS_PER_THREAD],
                                                  DecodedOffsetT from_decoded_offset = 0)
  {
    // The offset of the first item decoded by this thread
    DecodedOffsetT thread_decoded_offset = from_decoded_offset + linear_tid * DECODED_ITEMS_PER_THREAD;

    // The run that the first decoded item of this thread belongs to
    // If this thread's <thread_decoded_offset> is already beyond the total decoded size, it will be assigned to the
    // last run
    RunOffsetT assigned_run = cub::UpperBound(temp_storage.runs.run_offsets, BLOCK_RUNS + 1, thread_decoded_offset) -
                              static_cast<RunOffsetT>(1U);

    DecodedOffsetT assigned_run_begin = temp_storage.runs.run_offsets[assigned_run];
    DecodedOffsetT assigned_run_end   = temp_storage.runs.run_offsets[assigned_run + 1];

    UniqueItemT val = temp_storage.runs.unique_items[assigned_run];

#pragma unroll
    for (DecodedOffsetT i = 0; i < DECODED_ITEMS_PER_THREAD; i++)
    {
      decoded_items[i] = val;
      item_offsets[i]  = thread_decoded_offset - assigned_run_begin;
      if (thread_decoded_offset == assigned_run_end - 1)
      {
        // If this thread is already beyond the total_decoded_size, we keep it assigned to the very last run
        assigned_run       = min(BLOCK_THREADS * RUNS_PER_THREAD, assigned_run + 1);
        assigned_run_begin = temp_storage.runs.run_offsets[assigned_run];
        assigned_run_end   = temp_storage.runs.run_offsets[assigned_run + 1];
        val                = temp_storage.runs.unique_items[assigned_run];
      }
      thread_decoded_offset++;
    }
  }

  __device__ __forceinline__ void RunLengthDecode(UniqueItemT (&decoded_items)[DECODED_ITEMS_PER_THREAD],
                                                  DecodedOffsetT from_decoded_offset = 0)
  {
    DecodedOffsetT item_offsets[DECODED_ITEMS_PER_THREAD];
    RunLengthDecode(decoded_items, item_offsets, from_decoded_offset);
  }
};

CUB_NAMESPACE_END