/******************************************************************************
 * Copyright (c) 2011-2021, NVIDIA CORPORATION. All rights reserved.
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

#include <cstdint>
#include <iterator>
#include <stdio.h>
#include <type_traits>
#include <utility>

#include <cub/block/block_load.cuh>
#include <cub/block/block_run_length_decode.cuh>
#include <cub/block/block_store.cuh>
#include <cub/device/device_scan.cuh>
#include <cub/iterator/counting_input_iterator.cuh>
#include <cub/iterator/transform_input_iterator.cuh>
#include <cub/util_allocator.cuh>

#include "test_util.h"

using namespace cub;

/******************************************************************************
 * HELPER CLASS FOR RUN-LENGTH DECODING TESTS
 ******************************************************************************/

/**
 * \brief Class template to facilitate testing the BlockRunLengthDecode algorithm for all its template parameter
 * specialisations.
 *
 * \tparam ItemItT The item type being run-length decoded
 * \tparam RunLengthsItT Iterator type providing the runs' lengths
 * \tparam RUNS_PER_THREAD The number of runs that each thread is getting assigned to
 * \tparam DECODED_ITEMS_PER_THREAD The number of run-length decoded items that each thread is decoding
 * \tparam TEST_RELATIVE_OFFSETS_ Whether to also retrieve each decoded item's relative offset within its run
 * \tparam TEST_RUN_OFFSETS_ Whether to pass in each run's offset instead of each run's length
 * \tparam BLOCK_DIM_X The thread block length in threads along the X dimension
 * \tparam BLOCK_DIM_Y The thread block length in threads along the Y dimension
 * \tparam BLOCK_DIM_Z The thread block length in threads along the Z dimension
 */
template <typename ItemItT,
          typename RunLengthsItT,
          int RUNS_PER_THREAD,
          int DECODED_ITEMS_PER_THREAD,
          bool TEST_RELATIVE_OFFSETS_,
          bool TEST_RUN_OFFSETS_,
          int BLOCK_DIM_X,
          int BLOCK_DIM_Y = 1,
          int BLOCK_DIM_Z = 1>
class AgentTestBlockRunLengthDecode
{
public:
  constexpr static uint32_t BLOCK_THREADS     = BLOCK_DIM_X * BLOCK_DIM_Y * BLOCK_DIM_Z;
  constexpr static uint32_t RUNS_PER_BLOCK    = RUNS_PER_THREAD * BLOCK_THREADS;
  constexpr static bool TEST_RELATIVE_OFFSETS = TEST_RELATIVE_OFFSETS_;

private:
  using RunItemT   = cub::detail::value_t<ItemItT>;
  using RunLengthT = cub::detail::value_t<RunLengthsItT>;

  using BlockRunOffsetScanT = cub::BlockScan<RunLengthT,
                                             BLOCK_DIM_X,
                                             BLOCK_SCAN_RAKING,
                                             BLOCK_DIM_Y,
                                             BLOCK_DIM_Z>;

  using BlockRunLengthDecodeT =
    cub::BlockRunLengthDecode<RunItemT,
                              BLOCK_DIM_X,
                              RUNS_PER_THREAD,
                              DECODED_ITEMS_PER_THREAD>;

  using BlockLoadRunItemT = cub::BlockLoad<RunItemT,
                                           BLOCK_DIM_X,
                                           RUNS_PER_THREAD,
                                           BLOCK_LOAD_WARP_TRANSPOSE,
                                           BLOCK_DIM_Y,
                                           BLOCK_DIM_Z>;

  using BlockLoadRunLengthsT = cub::BlockLoad<RunLengthT,
                                              BLOCK_DIM_X,
                                              RUNS_PER_THREAD,
                                              BLOCK_LOAD_WARP_TRANSPOSE,
                                              BLOCK_DIM_Y,
                                              BLOCK_DIM_Z>;

  using BlockStoreDecodedItemT = cub::BlockStore<RunItemT,
                                                 BLOCK_DIM_X,
                                                 DECODED_ITEMS_PER_THREAD,
                                                 BLOCK_STORE_WARP_TRANSPOSE,
                                                 BLOCK_DIM_Y,
                                                 BLOCK_DIM_Z>;

  using BlockStoreRelativeOffsetT = cub::BlockStore<RunLengthT,
                                                    BLOCK_DIM_X,
                                                    DECODED_ITEMS_PER_THREAD,
                                                    BLOCK_STORE_WARP_TRANSPOSE,
                                                    BLOCK_DIM_Y,
                                                    BLOCK_DIM_Z>;

  __device__ __forceinline__ BlockRunLengthDecodeT InitBlockRunLengthDecode(RunItemT (&unique_items)[RUNS_PER_THREAD],
                                                                            RunLengthT (&run_lengths)[RUNS_PER_THREAD],
                                                                            RunLengthT &decoded_size,
                                                                            cub::Int2Type<true> /*test_run_offsets*/)
  {
    RunLengthT run_offsets[RUNS_PER_THREAD];
    BlockRunOffsetScanT(temp_storage.run_offsets_scan_storage).ExclusiveSum(run_lengths, run_offsets, decoded_size);

    // Ensure temporary shared memory can be repurposed
    CTA_SYNC();

    // Construct BlockRunLengthDecode and initialize with the run offsets
    return BlockRunLengthDecodeT(temp_storage.decode.run_length_decode_storage, unique_items, run_offsets);
  }

  __device__ __forceinline__ BlockRunLengthDecodeT InitBlockRunLengthDecode(RunItemT (&unique_items)[RUNS_PER_THREAD],
                                                                            RunLengthT (&run_lengths)[RUNS_PER_THREAD],
                                                                            RunLengthT &decoded_size,
                                                                            cub::Int2Type<false> /*test_run_offsets*/)
  {
    // Construct BlockRunLengthDecode and initialize with the run lengths
    return BlockRunLengthDecodeT(temp_storage.decode.run_length_decode_storage, unique_items, run_lengths, decoded_size);
  }

  __device__ __forceinline__ void LoadRuns(ItemItT d_block_unique_items,
                                           RunLengthsItT d_block_run_lengths,
                                           RunItemT (&unique_items)[RUNS_PER_THREAD],
                                           RunLengthT (&run_lengths)[RUNS_PER_THREAD],
                                           size_t num_valid_items)
  {
    if (num_valid_items < RUNS_PER_BLOCK)
    {
      BlockLoadRunItemT(temp_storage.load_uniques_storage).Load(d_block_unique_items, unique_items, num_valid_items);
    }
    else
    {
      BlockLoadRunItemT(temp_storage.load_uniques_storage).Load(d_block_unique_items, unique_items);
    }

    // Ensure BlockLoad's temporary shared memory can be repurposed
    CTA_SYNC();

    // Load this block's tile of run lengths
    if (num_valid_items < RUNS_PER_BLOCK)
      BlockLoadRunLengthsT(temp_storage.load_run_lengths_storage)
        .Load(d_block_run_lengths, run_lengths, num_valid_items, static_cast<RunLengthT>(0));
    else
      BlockLoadRunLengthsT(temp_storage.load_run_lengths_storage).Load(d_block_run_lengths, run_lengths);

    // Ensure temporary shared memory can be repurposed
    CTA_SYNC();
  }

public:
  union TempStorage
  {
    typename BlockLoadRunItemT::TempStorage load_uniques_storage;
    typename BlockLoadRunLengthsT::TempStorage load_run_lengths_storage;
    cub::detail::conditional_t<TEST_RUN_OFFSETS_,
                               typename BlockRunOffsetScanT::TempStorage,
                               cub::NullType>
      run_offsets_scan_storage;
    struct
    {
      typename BlockRunLengthDecodeT::TempStorage run_length_decode_storage;
      typename BlockStoreDecodedItemT::TempStorage store_decoded_runs_storage;
      typename BlockStoreRelativeOffsetT::TempStorage store_relative_offsets;
    } decode;
  };

  TempStorage &temp_storage;

  __device__ __forceinline__ AgentTestBlockRunLengthDecode(TempStorage &temp_storage)
      : temp_storage(temp_storage)
  {}

  /**
   * \brief Loads the given block (or tile) of runs, and computes their "decompressed" (run-length decoded) size.
   */
  __device__ __forceinline__ uint32_t GetDecodedSize(ItemItT d_block_unique_items,
                                                     RunLengthsItT d_block_run_lengths,
                                                     size_t num_valid_runs)
  {
    // Load this block's tile of encoded runs
    RunItemT unique_items[RUNS_PER_THREAD];
    RunLengthT run_lengths[RUNS_PER_THREAD];
    LoadRuns(d_block_unique_items, d_block_run_lengths, unique_items, run_lengths, num_valid_runs);

    // Init the BlockRunLengthDecode and get the total decoded size of this block's tile (i.e., the "decompressed" size)
    uint32_t decoded_size = 0U;
    BlockRunLengthDecodeT run_length_decode =
      InitBlockRunLengthDecode(unique_items, run_lengths, decoded_size, cub::Int2Type<TEST_RUN_OFFSETS_>());
    return decoded_size;
  }

  /**
   * \brief Loads the given block (or tile) of runs, run-length decodes them, and writes the results to \p
   * d_block_decoded_out.
   */
  template <typename UniqueItemOutItT, typename RelativeOffsetOutItT>
  __device__ __forceinline__ uint32_t WriteDecodedRuns(ItemItT d_block_unique_items,
                                                       RunLengthsItT d_block_run_lengths,
                                                       UniqueItemOutItT d_block_decoded_out,
                                                       RelativeOffsetOutItT d_block_rel_out,
                                                       size_t num_valid_runs)
  {
    // Load this block's tile of encoded runs
    RunItemT unique_items[RUNS_PER_THREAD];
    RunLengthT run_lengths[RUNS_PER_THREAD];
    LoadRuns(d_block_unique_items, d_block_run_lengths, unique_items, run_lengths, num_valid_runs);

    // Init the BlockRunLengthDecode and get the total decoded size of this block's tile (i.e., the "decompressed" size)
    uint32_t decoded_size = 0U;
    BlockRunLengthDecodeT run_length_decode =
      InitBlockRunLengthDecode(unique_items, run_lengths, decoded_size, cub::Int2Type<TEST_RUN_OFFSETS_>());

    // Run-length decode ("decompress") the runs into a window buffer of limited size. This is repeated until all runs
    // have been decoded.
    uint32_t decoded_window_offset = 0U;
    while (decoded_window_offset < decoded_size)
    {
      RunLengthT relative_offsets[DECODED_ITEMS_PER_THREAD];
      RunItemT decoded_items[DECODED_ITEMS_PER_THREAD];

      // The number of decoded items that are valid within this window (aka pass) of run-length decoding
      uint32_t num_valid_items = decoded_size - decoded_window_offset;
      run_length_decode.RunLengthDecode(decoded_items, relative_offsets, decoded_window_offset);
      BlockStoreDecodedItemT(temp_storage.decode.store_decoded_runs_storage)
        .Store(d_block_decoded_out + decoded_window_offset, decoded_items, num_valid_items);

      if (TEST_RELATIVE_OFFSETS)
      {
        BlockStoreRelativeOffsetT(temp_storage.decode.store_relative_offsets)
          .Store(d_block_rel_out + decoded_window_offset, relative_offsets, num_valid_items);
      }

      decoded_window_offset += DECODED_ITEMS_PER_THREAD * BLOCK_THREADS;
    }
    return decoded_size;
  }
};

/******************************************************************************
 * [STAGE 1] RUN-LENGTH DECODING TEST KERNEL
 ******************************************************************************/
template <typename AgentTestBlockRunLengthDecode,
          typename ItemItT,
          typename RunLengthsItT,
          typename OffsetT,
          typename DecodedSizesOutT>
__launch_bounds__(AgentTestBlockRunLengthDecode::BLOCK_THREADS) __global__
  void BlockRunLengthDecodeGetSizeKernel(const ItemItT d_unique_items,
                                         const RunLengthsItT d_run_lengths,
                                         const OffsetT num_runs,
                                         DecodedSizesOutT d_decoded_sizes)
{
  constexpr OffsetT RUNS_PER_BLOCK = AgentTestBlockRunLengthDecode::RUNS_PER_BLOCK;

  __shared__ typename AgentTestBlockRunLengthDecode::TempStorage temp_storage;

  OffsetT block_offset   = blockIdx.x * RUNS_PER_BLOCK;
  OffsetT num_valid_runs = (block_offset + RUNS_PER_BLOCK >= num_runs) ? (num_runs - block_offset) : RUNS_PER_BLOCK;

  AgentTestBlockRunLengthDecode run_length_decode_agent(temp_storage);
  uint64_t num_decoded_items =
    run_length_decode_agent.GetDecodedSize(d_unique_items + block_offset, d_run_lengths + block_offset, num_valid_runs);

  d_decoded_sizes[blockIdx.x] = num_decoded_items;
}

/******************************************************************************
 * [STAGE 2] RUN-LENGTH DECODING TEST KERNEL
 ******************************************************************************/
template <typename AgentTestBlockRunLengthDecode,
          typename ItemItT,
          typename RunLengthsItT,
          typename DecodedSizesOutT,
          typename OffsetT,
          typename DecodedItemsOutItT,
          typename RelativeOffsetOutItT>
__launch_bounds__(AgentTestBlockRunLengthDecode::BLOCK_THREADS) __global__
  void BlockRunLengthDecodeTestKernel(const ItemItT d_unique_items,
                                      const RunLengthsItT d_run_lengths,
                                      const DecodedSizesOutT d_decoded_offsets,
                                      const OffsetT num_runs,
                                      DecodedItemsOutItT d_decoded_items,
                                      RelativeOffsetOutItT d_relative_offsets)

{
  constexpr OffsetT RUNS_PER_BLOCK = AgentTestBlockRunLengthDecode::RUNS_PER_BLOCK;

  __shared__ typename AgentTestBlockRunLengthDecode::TempStorage temp_storage;

  OffsetT block_offset   = blockIdx.x * RUNS_PER_BLOCK;
  OffsetT num_valid_runs = (block_offset + RUNS_PER_BLOCK >= num_runs) ? (num_runs - block_offset) : RUNS_PER_BLOCK;

  AgentTestBlockRunLengthDecode run_length_decode_agent(temp_storage);
  run_length_decode_agent.WriteDecodedRuns(d_unique_items + block_offset,
                                           d_run_lengths + block_offset,
                                           d_decoded_items + d_decoded_offsets[blockIdx.x],
                                           d_relative_offsets + d_decoded_offsets[blockIdx.x],
                                           num_valid_runs);
}

struct ModOp
{
  using T = uint32_t;
  __host__ __device__ __forceinline__ T operator()(const T &x) const { return 1 + (x % 100); }
};

template <uint32_t RUNS_PER_THREAD,
          uint32_t DECODED_ITEMS_PER_THREAD,
          uint32_t BLOCK_DIM_X,
          uint32_t BLOCK_DIM_Y,
          uint32_t BLOCK_DIM_Z,
          bool TEST_RUN_OFFSETS,
          bool TEST_RELATIVE_OFFSETS>
void TestAlgorithmSpecialisation()
{
  constexpr uint32_t THREADS_PER_BLOCK = BLOCK_DIM_X * BLOCK_DIM_Y * BLOCK_DIM_Z;
  constexpr uint32_t RUNS_PER_BLOCK    = RUNS_PER_THREAD * THREADS_PER_BLOCK;

  using RunItemT      = float;
  using RunLengthT    = uint32_t;
  using ItemItT       = cub::CountingInputIterator<RunItemT>;
  using RunLengthsItT = cub::TransformInputIterator<RunLengthT, ModOp, cub::CountingInputIterator<RunLengthT>>;

  ItemItT d_unique_items(1000U);
  RunLengthsItT d_run_lengths(cub::CountingInputIterator<RunLengthT>(0), ModOp{});

  constexpr uint32_t num_runs   = 10000;
  constexpr uint32_t num_blocks = (num_runs + (RUNS_PER_BLOCK - 1U)) / RUNS_PER_BLOCK;

  size_t temp_storage_bytes      = 0ULL;
  void *temp_storage             = nullptr;
  uint32_t *h_num_decoded_total  = nullptr;
  uint32_t *d_decoded_sizes      = nullptr;
  uint32_t *d_decoded_offsets    = nullptr;
  RunItemT *d_decoded_out        = nullptr;
  RunLengthT *d_relative_offsets = nullptr;
  RunItemT *h_decoded_out        = nullptr;
  RunLengthT *h_relative_offsets = nullptr;

  using AgentTestBlockRunLengthDecodeT = AgentTestBlockRunLengthDecode<ItemItT,
                                                                       RunLengthsItT,
                                                                       RUNS_PER_THREAD,
                                                                       DECODED_ITEMS_PER_THREAD,
                                                                       TEST_RELATIVE_OFFSETS,
                                                                       TEST_RUN_OFFSETS,
                                                                       THREADS_PER_BLOCK,
                                                                       1,
                                                                       1>;

  enum : uint32_t
  {
    TIMER_SIZE_BEGIN = 0,
    TIMER_SIZE_END,
    TIMER_DECODE_BEGIN,
    TIMER_DECODE_END,
    NUM_TIMERS,
  };

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  cudaEvent_t cuda_evt_timers[NUM_TIMERS];
  for (uint32_t i = 0; i < NUM_TIMERS; i++)
  {
    cudaEventCreate(&cuda_evt_timers[i]);
  }

  // Get temporary storage requirements for the scan (for computing offsets for the per-block run-length decoded items)
  cub::DeviceScan::InclusiveSum(nullptr, temp_storage_bytes, d_decoded_sizes, d_decoded_offsets, num_blocks, stream);

  // Allocate device memory
  CubDebugExit(cudaMalloc(&temp_storage, temp_storage_bytes));
  CubDebugExit(cudaMalloc(&d_decoded_sizes, num_blocks * sizeof(*d_decoded_sizes)));
  // Allocate for the exclusive sum PLUS the overall aggregate
  CubDebugExit(cudaMalloc(&d_decoded_offsets, (num_blocks + 1) * sizeof(*d_decoded_offsets)));
  CubDebugExit(cudaMallocHost(&h_num_decoded_total, sizeof(*h_num_decoded_total)));

  // Get the per-block number of items being decoded (i-th thread block writing size to d_decoded_sizes[i])
  CubDebugExit(cudaEventRecord(cuda_evt_timers[TIMER_SIZE_BEGIN], stream));
  BlockRunLengthDecodeGetSizeKernel<AgentTestBlockRunLengthDecodeT>
    <<<num_blocks, THREADS_PER_BLOCK, 0U, stream>>>(d_unique_items, d_run_lengths, num_runs, d_decoded_sizes);
  CubDebugExit(cudaEventRecord(cuda_evt_timers[TIMER_SIZE_END], stream));

  // Compute offsets for the runs decoded by each block (exclusive sum + aggregate)
  CubDebugExit(cudaMemsetAsync(d_decoded_offsets, 0, sizeof(d_decoded_offsets[0]), stream));
  CubDebugExit(cub::DeviceScan::InclusiveSum(temp_storage,
                                             temp_storage_bytes,
                                             d_decoded_sizes,
                                             &d_decoded_offsets[1],
                                             num_blocks,
                                             stream));

  // Copy the total decoded size to CPU in order to allocate just the right amount of device memory
  CubDebugExit(cudaMemcpyAsync(h_num_decoded_total,
                               &d_decoded_offsets[num_blocks],
                               sizeof(*h_num_decoded_total),
                               cudaMemcpyDeviceToHost,
                               stream));

  // Ensure the total decoded size has been copied from GPU to CPU
  CubDebugExit(cudaStreamSynchronize(stream));

  // Allocate device memory for the run-length decoded output
  CubDebugExit(cudaMallocHost(&h_decoded_out, (*h_num_decoded_total) * sizeof(RunItemT)));
  CubDebugExit(cudaMalloc(&d_decoded_out, (*h_num_decoded_total) * sizeof(RunItemT)));
  if (TEST_RELATIVE_OFFSETS)
  {
    CubDebugExit(cudaMalloc(&d_relative_offsets, (*h_num_decoded_total) * sizeof(RunLengthT)));
    CubDebugExit(cudaMallocHost(&h_relative_offsets, (*h_num_decoded_total) * sizeof(RunLengthT)));
  }

  // Perform the block-wise run-length decoding (each block taking its offset from d_decoded_offsets)
  CubDebugExit(cudaEventRecord(cuda_evt_timers[TIMER_DECODE_BEGIN], stream));
  BlockRunLengthDecodeTestKernel<AgentTestBlockRunLengthDecodeT>
    <<<num_blocks, THREADS_PER_BLOCK, 0U, stream>>>(d_unique_items,
                                                    d_run_lengths,
                                                    d_decoded_offsets,
                                                    num_runs,
                                                    d_decoded_out,
                                                    d_relative_offsets);
  CubDebugExit(cudaEventRecord(cuda_evt_timers[TIMER_DECODE_END], stream));

  // Copy back results for verification
  CubDebugExit(cudaMemcpyAsync(h_decoded_out,
                               d_decoded_out,
                               (*h_num_decoded_total) * sizeof(*h_decoded_out),
                               cudaMemcpyDeviceToHost,
                               stream));

  if (TEST_RELATIVE_OFFSETS)
  {
    // Copy back the relative offsets
    CubDebugExit(cudaMemcpyAsync(h_relative_offsets,
                                 d_relative_offsets,
                                 (*h_num_decoded_total) * sizeof(*h_relative_offsets),
                                 cudaMemcpyDeviceToHost,
                                 stream));
  }

  // Generate host-side run-length decoded data for verification
  std::vector<std::pair<RunItemT, RunLengthT>> host_golden;
  host_golden.reserve(*h_num_decoded_total);
  for (uint32_t run = 0; run < num_runs; run++)
  {
    for (RunLengthT i = 0; i < d_run_lengths[run]; i++)
    {
      host_golden.push_back({d_unique_items[run], i});
    }
  }

  // Ensure the run-length decoded result has been copied to the host
  CubDebugExit(cudaStreamSynchronize(stream));

  // Verify the total run-length decoded size is correct
  AssertEquals(host_golden.size(), h_num_decoded_total[0]);

  float duration_size   = 0.0f;
  float duration_decode = 0.0f;
  cudaEventElapsedTime(&duration_size, cuda_evt_timers[TIMER_SIZE_BEGIN], cuda_evt_timers[TIMER_SIZE_END]);
  cudaEventElapsedTime(&duration_decode, cuda_evt_timers[TIMER_DECODE_BEGIN], cuda_evt_timers[TIMER_DECODE_END]);

  size_t decoded_bytes          = host_golden.size() * sizeof(RunItemT);
  size_t relative_offsets_bytes = TEST_RELATIVE_OFFSETS ? host_golden.size() * sizeof(RunLengthT) : 0ULL;
  size_t total_bytes_written    = decoded_bytes + relative_offsets_bytes;

  std::cout << "MODE: " << (TEST_RELATIVE_OFFSETS ? "offsets, " : "normal,  ")    //
            << "INIT: " << (TEST_RUN_OFFSETS ? "run offsets, " : "run lengths, ") //
            << "RUNS_PER_THREAD: " << RUNS_PER_THREAD                             //
            << ", DECODED_ITEMS_PER_THREAD: " << DECODED_ITEMS_PER_THREAD         //
            << ", THREADS_PER_BLOCK: " << THREADS_PER_BLOCK                       //
            << ", decoded size (bytes): " << decoded_bytes                        //
            << ", relative offsets (bytes): " << relative_offsets_bytes           //
            << ", time_size (ms): " << duration_size                              //
            << ", time_decode (ms): " << duration_decode                          //
            << ", achieved decode BW (GB/s): "
            << ((static_cast<double>(total_bytes_written) / 1.0e9) * (1000.0 / duration_decode)) << "\n";

  // Verify the run-length decoded data is correct
  bool cmp_eq = true;
  for (uint32_t i = 0; i < host_golden.size(); i++)
  {
    if (host_golden[i].first != h_decoded_out[i])
    {
      std::cout << "Mismatch at #" << i << ": CPU item: " << host_golden[i].first << ", GPU: " << h_decoded_out[i]
                << "\n";
      cmp_eq = false;
    }
    if (TEST_RELATIVE_OFFSETS)
    {
      if (host_golden[i].second != h_relative_offsets[i])
      {
        std::cout << "Mismatch of relative offset at #" << i << ": CPU item: " << host_golden[i].first
                  << ", GPU: " << h_decoded_out[i] << "; relative offsets: CPU: " << host_golden[i].second
                  << ", GPU: " << h_relative_offsets[i] << "\n";
        cmp_eq = false;
        break;
      }
    }
  }
  AssertEquals(cmp_eq, true);

  // Clean up memory allocations
  CubDebugExit(cudaFree(temp_storage));
  CubDebugExit(cudaFree(d_decoded_sizes));
  CubDebugExit(cudaFree(d_decoded_offsets));
  CubDebugExit(cudaFree(d_decoded_out));
  CubDebugExit(cudaFreeHost(h_num_decoded_total));
  CubDebugExit(cudaFreeHost(h_decoded_out));
  if (TEST_RELATIVE_OFFSETS)
  {
    CubDebugExit(cudaFree(d_relative_offsets));
    CubDebugExit(cudaFreeHost(h_relative_offsets));
  }

  // Clean up events
  for (uint32_t i = 0; i < NUM_TIMERS; i++)
  {
    CubDebugExit(cudaEventDestroy(cuda_evt_timers[i]));
  }

  // Clean up streams
  CubDebugExit(cudaStreamDestroy(stream));
}

template <uint32_t RUNS_PER_THREAD,
          uint32_t DECODED_ITEMS_PER_THREAD,
          uint32_t BLOCK_DIM_X,
          uint32_t BLOCK_DIM_Y = 1U,
          uint32_t BLOCK_DIM_Z = 1U>
void TestForTuningParameters()
{
  constexpr bool DO_TEST_RELATIVE_OFFSETS     = true;
  constexpr bool DO_NOT_TEST_RELATIVE_OFFSETS = false;

  constexpr bool TEST_WITH_RUN_OFFSETS = true;
  constexpr bool TEST_WITH_RUN_LENGTHS = false;
  // Run BlockRunLengthDecode that uses run lengths and generates offsets relative to each run
  TestAlgorithmSpecialisation<RUNS_PER_THREAD,
                              DECODED_ITEMS_PER_THREAD,
                              BLOCK_DIM_X,
                              BLOCK_DIM_Y,
                              BLOCK_DIM_Z,
                              TEST_WITH_RUN_LENGTHS,
                              DO_TEST_RELATIVE_OFFSETS>();

  // Run BlockRunLengthDecode that uses run lengths and performs normal run-length decoding
  TestAlgorithmSpecialisation<RUNS_PER_THREAD,
                              DECODED_ITEMS_PER_THREAD,
                              BLOCK_DIM_X,
                              BLOCK_DIM_Y,
                              BLOCK_DIM_Z,
                              TEST_WITH_RUN_LENGTHS,
                              DO_NOT_TEST_RELATIVE_OFFSETS>();

  // Run BlockRunLengthDecode that uses run offsets and generates offsets relative to each run
  TestAlgorithmSpecialisation<RUNS_PER_THREAD,
                              DECODED_ITEMS_PER_THREAD,
                              BLOCK_DIM_X,
                              BLOCK_DIM_Y,
                              BLOCK_DIM_Z,
                              TEST_WITH_RUN_OFFSETS,
                              DO_TEST_RELATIVE_OFFSETS>();

  // Run BlockRunLengthDecode that uses run offsets and performs normal run-length decoding
  TestAlgorithmSpecialisation<RUNS_PER_THREAD,
                              DECODED_ITEMS_PER_THREAD,
                              BLOCK_DIM_X,
                              BLOCK_DIM_Y,
                              BLOCK_DIM_Z,
                              TEST_WITH_RUN_OFFSETS,
                              DO_NOT_TEST_RELATIVE_OFFSETS>();
}

int main(int argc, char **argv)
{
  CommandLineArgs args(argc, argv);

  // Initialize device
  CubDebugExit(args.DeviceInit());

  // Instantiate test template instances for various configurations (tuning parameter dimensions)
  // <RUNS_PER_THREAD, DECODED_ITEMS_PER_THREAD, BLOCK_DIM_X[, BLOCK_DIM_Y, BLOCK_DIM_Z]>
  TestForTuningParameters<1U, 1U, 64U>();
  TestForTuningParameters<1U, 3U, 32U, 2U, 3U>();
  TestForTuningParameters<1U, 1U, 128U>();
  TestForTuningParameters<1U, 8U, 128U>();
  TestForTuningParameters<2U, 8U, 128U>();
  TestForTuningParameters<3U, 1U, 256U>();
  TestForTuningParameters<1U, 8U, 256U>();
  TestForTuningParameters<8U, 1U, 256U>();
  TestForTuningParameters<1U, 1U, 256U>();
  TestForTuningParameters<2U, 2U, 384U>();

  return 0;
}
