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

#include "test_util.h"
#include "cub/warp/warp_exchange.cuh"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

template <typename InputT,
          typename OutputT,
          int LogicalWarpThreads,
          int ItemsPerThread,
          int BlockThreads,
          typename ActionT>
__global__ void kernel(const InputT *input_data,
                       OutputT *output_data,
                       ActionT action,
                       cub::Int2Type<true> same_type)
{
  using WarpExchangeT =
    cub::WarpExchange<InputT, ItemsPerThread, LogicalWarpThreads>;

  constexpr int tile_size = ItemsPerThread * LogicalWarpThreads;
  constexpr int warps_per_block = BlockThreads / LogicalWarpThreads;
  __shared__ typename WarpExchangeT::TempStorage temp_storage[warps_per_block];

  const int warp_id = threadIdx.x / LogicalWarpThreads;
  WarpExchangeT exchange(temp_storage[warp_id]);

  InputT input[ItemsPerThread];

  input_data += warp_id * tile_size;
  output_data += warp_id * tile_size;

  for (int item = 0; item < ItemsPerThread; item++)
  {
    input[item] = input_data[exchange.lane_id * ItemsPerThread + item];
  }

  action(input, input, exchange);

  for (int item = 0; item < ItemsPerThread; item++)
  {
    output_data[exchange.lane_id * ItemsPerThread + item] = input[item];
  }
}

template <typename InputT,
          typename OutputT,
          int LogicalWarpThreads,
          int ItemsPerThread,
          int BlockThreads,
          typename ActionT>
__global__ void kernel(const InputT *input_data,
                       OutputT *output_data,
                       ActionT action,
                       cub::Int2Type<false> different_types)
{
  using WarpExchangeT =
  cub::WarpExchange<InputT, ItemsPerThread, LogicalWarpThreads>;

  constexpr int tile_size = ItemsPerThread * LogicalWarpThreads;
  constexpr int warps_per_block = BlockThreads / LogicalWarpThreads;
  __shared__ typename WarpExchangeT::TempStorage temp_storage[warps_per_block];

  const int warp_id = threadIdx.x / LogicalWarpThreads;
  WarpExchangeT exchange(temp_storage[warp_id]);

  InputT input[ItemsPerThread];
  OutputT output[ItemsPerThread];

  input_data += warp_id * tile_size;
  output_data += warp_id * tile_size;

  for (int item = 0; item < ItemsPerThread; item++)
  {
    input[item] = input_data[exchange.lane_id * ItemsPerThread + item];
  }

  action(input, output, exchange);

  for (int item = 0; item < ItemsPerThread; item++)
  {
    output_data[exchange.lane_id * ItemsPerThread + item] = output[item];
    }
                         }

struct StripedToBlocked
{
  template <typename InputT,
            typename OutputT,
            int LogicalWarpThreads,
            int ItemsPerThread,
            int ITEMS_PER_THREAD>
  __device__ void operator()(
    InputT (&input)[ITEMS_PER_THREAD],
    OutputT (&output)[ITEMS_PER_THREAD],
    cub::WarpExchange<InputT, ItemsPerThread, LogicalWarpThreads> &exchange)
  {
    exchange.StripedToBlocked(input, output);
  }
};

struct BlockedToStriped
{
  template <typename InputT,
            typename OutputT,
            int LogicalWarpThreads,
            int ItemsPerThread,
            int ITEMS_PER_THREAD>
  __device__ void operator()(
    InputT (&input)[ITEMS_PER_THREAD],
    OutputT (&output)[ITEMS_PER_THREAD],
    cub::WarpExchange<InputT, ItemsPerThread, LogicalWarpThreads> &exchange)
  {
    exchange.BlockedToStriped(input, output);
  }
};


template <typename T>
bool Compare(
  const thrust::device_vector<T> &lhs,
  const thrust::device_vector<T> &rhs)
{
  auto err = thrust::mismatch(lhs.begin(), lhs.end(), rhs.begin());

  if (err.first != lhs.end())
  {
    auto i = thrust::distance(lhs.begin(), err.first);

    std::cerr << "Mismatch at " << i << ": " << lhs[i] << " != " << rhs[i]
              << std::endl;

    return false;
  }

  return true;
}


template <typename InputT,
          typename OutputT,
          int LogicalWarpThreads,
          int ItemsPerThread,
          int BlockThreads>
void TestStripedToBlocked(thrust::device_vector<InputT> &input,
                          thrust::device_vector<OutputT> &output)
{
  thrust::fill(output.begin(), output.end(), OutputT{0});

  thrust::host_vector<InputT> h_input(input.size());
  FillStriped<LogicalWarpThreads, ItemsPerThread, BlockThreads>(
    h_input.begin());

  input = h_input;

  kernel<InputT, OutputT, LogicalWarpThreads, ItemsPerThread, BlockThreads>
    <<<1, BlockThreads>>>(thrust::raw_pointer_cast(input.data()),
                          thrust::raw_pointer_cast(output.data()),
                          StripedToBlocked{},
                          cub::Int2Type<std::is_same<InputT, OutputT>::value>{});
  cudaDeviceSynchronize();

  thrust::device_vector<OutputT> expected_output(output.size());
  thrust::sequence(expected_output.begin(), expected_output.end());

  AssertTrue(Compare(expected_output, output));
}

template <typename InputT,
          typename OutputT,
          int LogicalWarpThreads,
          int ItemsPerThread,
          int BlockThreads>
void TestBlockedToStriped(thrust::device_vector<InputT> &input,
                          thrust::device_vector<OutputT> &output)
{
  thrust::fill(output.begin(), output.end(), OutputT{0});

  thrust::host_vector<OutputT> expected_output(input.size());
  FillStriped<LogicalWarpThreads, ItemsPerThread, BlockThreads>(
    expected_output.begin());

  thrust::sequence(input.begin(), input.end());

  kernel<InputT, OutputT, LogicalWarpThreads, ItemsPerThread, BlockThreads>
    <<<1, BlockThreads>>>(thrust::raw_pointer_cast(input.data()),
                          thrust::raw_pointer_cast(output.data()),
                          BlockedToStriped{},
                          cub::Int2Type<std::is_same<InputT, OutputT>::value>{});
  cudaDeviceSynchronize();

  thrust::device_vector<OutputT> d_expected_output(expected_output);
  AssertTrue(Compare(d_expected_output, output));
}

template <typename InputT,
          typename OutputT,
          int LogicalWarpThreads,
          int ItemsPerThread,
          int BlockThreads>
void Test()
{
  static_assert(BlockThreads % LogicalWarpThreads == 0,
                "BlockThreads must be a multiple of LogicalWarpThreads");

  const int warps_in_block = BlockThreads / LogicalWarpThreads;
  const int items_per_warp = LogicalWarpThreads * ItemsPerThread;
  const int items_per_block = items_per_warp * warps_in_block;

  thrust::device_vector<InputT> input(items_per_block);
  thrust::device_vector<OutputT> output(items_per_block);

  TestStripedToBlocked<InputT,
                       OutputT,
                       LogicalWarpThreads,
                       ItemsPerThread,
                       BlockThreads>(input, output);

  TestBlockedToStriped<InputT,
                       OutputT,
                       LogicalWarpThreads,
                       ItemsPerThread,
                       BlockThreads>(input, output);
}

template <int LogicalWarpThreads,
  int ItemsPerThread,
  int BlockThreads>
void Test()
{
  Test<std::uint16_t, std::uint32_t, LogicalWarpThreads, ItemsPerThread, BlockThreads>();
  Test<std::uint32_t, std::uint32_t, LogicalWarpThreads, ItemsPerThread, BlockThreads>();
  Test<std::uint64_t, std::uint32_t, LogicalWarpThreads, ItemsPerThread, BlockThreads>();
}

template <int LogicalWarpThreads,
          int ItemsPerThread>
void Test()
{
  Test<LogicalWarpThreads, ItemsPerThread, 128>();
  Test<LogicalWarpThreads, ItemsPerThread, 256>();
}

template <int LogicalWarpThreads>
void Test()
{
  Test<LogicalWarpThreads, 1>();
  Test<LogicalWarpThreads, 4>();
  Test<LogicalWarpThreads, 7>();
}

int main(int argc, char** argv)
{
  CommandLineArgs args(argc, argv);

  // Initialize device
  CubDebugExit(args.DeviceInit());

  Test<4>();
  Test<16>();
  Test<32>();

  return 0;
}
