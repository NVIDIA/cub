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

// Ensure printing of CUDA runtime errors to console
#define CUB_STDERR

#include <iterator>

#include <cub/warp/warp_store.cuh>
#include <cub/iterator/cache_modified_output_iterator.cuh>
#include <cub/iterator/discard_output_iterator.cuh>
#include <cub/util_allocator.cuh>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>

#include "test_util.h"

using namespace cub;


const int MAX_ITERATIONS = 30;


template <int                 BlockThreads,
          int                 WarpThreads,
          int                 ItemsPerThread,
          WarpStoreAlgorithm  StoreAlgorithm,
          typename            OutputIteratorT,
          typename            OutputT>
__global__ void kernel(OutputIteratorT output)
{
  using WarpStoreT =
    WarpStore<OutputT, ItemsPerThread, StoreAlgorithm, WarpThreads>;

  constexpr int warps_in_block = BlockThreads / WarpThreads;
  constexpr int tile_size = ItemsPerThread * WarpThreads;
  const int warp_id = static_cast<int>(threadIdx.x) / WarpThreads;

  __shared__ typename WarpStoreT::TempStorage temp_storage[warps_in_block];

  OutputT reg[ItemsPerThread];

  for (int item = 0; item < ItemsPerThread; item++)
  {
    reg[item] = static_cast<OutputT>(threadIdx.x * ItemsPerThread + item);
  }

  WarpStoreT(temp_storage[warp_id]).Store(output + warp_id * tile_size, reg);
}


template <int                 BlockThreads,
          int                 WarpThreads,
          int                 ItemsPerThread,
          WarpStoreAlgorithm  StoreAlgorithm,
          typename            OutputIteratorT,
          typename            OutputT>
__global__ void kernel(int valid_items,
                       OutputIteratorT output)
{
  using WarpStoreT =
    WarpStore<OutputT, ItemsPerThread, StoreAlgorithm, WarpThreads>;

  constexpr int warps_in_block = BlockThreads / WarpThreads;
  constexpr int tile_size = ItemsPerThread * WarpThreads;

  const int tid = static_cast<int>(threadIdx.x);
  const int warp_id = tid / WarpThreads;

  __shared__ typename WarpStoreT::TempStorage temp_storage[warps_in_block];

  OutputT reg[ItemsPerThread];

  for (int item = 0; item < ItemsPerThread; item++)
  {
    reg[item] = static_cast<OutputT>(threadIdx.x * ItemsPerThread + item);
  }

  WarpStoreT(temp_storage[warp_id])
    .Store(output + warp_id * tile_size, reg, valid_items);
}


template <typename            T,
          int                 BlockThreads,
          int                 WarpThreads,
          int                 ItemsPerThread,
          WarpStoreAlgorithm  StoreAlgorithm>
thrust::device_vector<T> GenExpectedOutput(int valid_items)
{
  const int tile_size = WarpThreads * ItemsPerThread;
  const int total_warps = BlockThreads / WarpThreads;
  const int elements = total_warps * tile_size;

  thrust::device_vector<T> input(elements);

  if (StoreAlgorithm == WarpStoreAlgorithm::WARP_STORE_STRIPED)
  {
    thrust::host_vector<T> h_input(elements);

    // In this case we need different stripe pattern, so the
    // items/threads parameters are swapped

    constexpr int fake_block_size = ItemsPerThread *
                                    (BlockThreads / WarpThreads);

    FillStriped<ItemsPerThread, WarpThreads, fake_block_size>(h_input.begin());
    input = h_input;
  }
  else
  {
    thrust::sequence(input.begin(), input.end());
  }

  if (valid_items != elements)
  {
    for (int warp_id = 0; warp_id < total_warps; warp_id++)
    {
      thrust::fill(input.begin() + warp_id * tile_size + valid_items,
                   input.begin() + (warp_id + 1) * tile_size,
                   T{});
    }
  }

  return input;
}


template <
  typename            T,
  int                 BlockThreads,
  int                 WarpThreads,
  int                 ItemsPerThread,
  WarpStoreAlgorithm  StoreAlgorithm>
void CheckResults(int valid_items,
                  const thrust::device_vector<T> &output)
{
  const thrust::device_vector<T> expected_output =
    GenExpectedOutput<T,
                      BlockThreads,
                      WarpThreads,
                      ItemsPerThread,
                      StoreAlgorithm>(valid_items);

  AssertEquals(expected_output, output);
}


template <typename            T,
  int                 BlockThreads,
  int                 WarpThreads,
  int                 ItemsPerThread,
  WarpStoreAlgorithm  StoreAlgorithm,
  typename            OutputIteratorT>
void TestImplementation(OutputIteratorT output)
{
  kernel<BlockThreads,
         WarpThreads,
         ItemsPerThread,
         StoreAlgorithm,
         OutputIteratorT,
         T><<<1, BlockThreads>>>(output);

  CubDebugExit(cudaPeekAtLastError());
  CubDebugExit(cudaDeviceSynchronize());
}


template <typename            T,
          int                 BlockThreads,
          int                 WarpThreads,
          int                 ItemsPerThread,
          WarpStoreAlgorithm  StoreAlgorithm,
          typename            OutputIteratorT>
void TestImplementation(int valid_items,
                        OutputIteratorT output)
{
  kernel<BlockThreads,
         WarpThreads,
         ItemsPerThread,
         StoreAlgorithm,
         OutputIteratorT,
         T><<<1, BlockThreads>>>(valid_items, output);

  CubDebugExit(cudaPeekAtLastError());
  CubDebugExit(cudaDeviceSynchronize());
}


template <typename            T,
  int                 BlockThreads,
  int                 WarpThreads,
  int                 ItemsPerThread,
  WarpStoreAlgorithm  StoreAlgorithm>
void TestPointer()
{
  const int tile_size = WarpThreads * ItemsPerThread;
  const int total_warps = BlockThreads / WarpThreads;
  const int elements = total_warps * tile_size;

  thrust::device_vector<T> output(elements);
  thrust::fill(output.begin(), output.end(), T{});

  TestImplementation<T, BlockThreads, WarpThreads, ItemsPerThread, StoreAlgorithm>(
    thrust::raw_pointer_cast(output.data()));

  CheckResults<T, BlockThreads, WarpThreads, ItemsPerThread, StoreAlgorithm>(
    elements, output);

  const unsigned int max_valid_items = WarpThreads * ItemsPerThread;

  for (int iteration = 0; iteration < MAX_ITERATIONS; iteration++)
  {
    const int valid_items = static_cast<int>(RandomValue(max_valid_items));

    thrust::fill(output.begin(), output.end(), T{});
    TestImplementation<T, BlockThreads, WarpThreads, ItemsPerThread, StoreAlgorithm>(
      valid_items, thrust::raw_pointer_cast(output.data()));

    CheckResults<T, BlockThreads, WarpThreads, ItemsPerThread, StoreAlgorithm>(
      valid_items, output);
  }
}


template <typename            T,
  int                 BlockThreads,
  int                 WarpThreads,
  int                 ItemsPerThread,
  WarpStoreAlgorithm  StoreAlgorithm,
  CacheStoreModifier  StoreModifier>
void TestIterator()
{
  const int tile_size = WarpThreads * ItemsPerThread;
  const int total_warps = BlockThreads / WarpThreads;
  const int elements = total_warps * tile_size;

  thrust::device_vector<T> output(elements);

  thrust::fill(output.begin(), output.end(), T{});
  TestImplementation<T, BlockThreads, WarpThreads, ItemsPerThread, StoreAlgorithm>(
    CacheModifiedOutputIterator<StoreModifier, T>(
      thrust::raw_pointer_cast(output.data())));

  CheckResults<T, BlockThreads, WarpThreads, ItemsPerThread, StoreAlgorithm>(
    elements, output);

  const int max_valid_items = WarpThreads * ItemsPerThread;

  for (int iteration = 0; iteration < MAX_ITERATIONS; iteration++)
  {
    const int valid_items = RandomValue(max_valid_items);

    thrust::fill(output.begin(), output.end(), T{});
    TestImplementation<T,
                       BlockThreads,
                       WarpThreads,
                       ItemsPerThread,
                       StoreAlgorithm>(
      valid_items,
      CacheModifiedOutputIterator<StoreModifier, T>(
        thrust::raw_pointer_cast(output.data())));

    CheckResults<T, BlockThreads, WarpThreads, ItemsPerThread, StoreAlgorithm>(
      valid_items, output);
  }
}


template <typename            T,
          int                 BlockThreads,
          int                 WarpThreads,
          int                 ItemsPerThread,
          WarpStoreAlgorithm  StoreAlgorithm>
void TestIterator()
{
  TestIterator<T, BlockThreads, WarpThreads, ItemsPerThread, StoreAlgorithm, CacheStoreModifier::STORE_DEFAULT>();
  TestIterator<T, BlockThreads, WarpThreads, ItemsPerThread, StoreAlgorithm, CacheStoreModifier::STORE_WB>();
  TestIterator<T, BlockThreads, WarpThreads, ItemsPerThread, StoreAlgorithm, CacheStoreModifier::STORE_CG>();
  TestIterator<T, BlockThreads, WarpThreads, ItemsPerThread, StoreAlgorithm, CacheStoreModifier::STORE_CS>();
  TestIterator<T, BlockThreads, WarpThreads, ItemsPerThread, StoreAlgorithm, CacheStoreModifier::STORE_WT>();
  TestIterator<T, BlockThreads, WarpThreads, ItemsPerThread, StoreAlgorithm, CacheStoreModifier::STORE_VOLATILE>();
}


template <typename            T,
          int                 BlockThreads,
          int                 WarpThreads,
          int                 ItemsPerThread,
          WarpStoreAlgorithm  StoreAlgorithm>
void Test()
{
  TestPointer<T, BlockThreads, WarpThreads, ItemsPerThread, StoreAlgorithm>();
  TestIterator<T, BlockThreads, WarpThreads, ItemsPerThread, StoreAlgorithm>();
}


template <typename  T,
          int       BlockThreads,
          int       WarpThreads,
          int       ItemsPerThread>
void Test()
{
  Test<T, BlockThreads, WarpThreads, ItemsPerThread, WarpStoreAlgorithm::WARP_STORE_DIRECT>();
  Test<T, BlockThreads, WarpThreads, ItemsPerThread, WarpStoreAlgorithm::WARP_STORE_STRIPED>();
  Test<T, BlockThreads, WarpThreads, ItemsPerThread, WarpStoreAlgorithm::WARP_STORE_TRANSPOSE>();
  Test<T, BlockThreads, WarpThreads, ItemsPerThread, WarpStoreAlgorithm::WARP_STORE_VECTORIZE>();
}


template <typename T,
          int      BlockThreads,
          int      WarpThreads>
void Test()
{
  Test<T, BlockThreads, WarpThreads, 1>();
  Test<T, BlockThreads, WarpThreads, 4>();
  Test<T, BlockThreads, WarpThreads, 7>();
}


template <typename T,
          int BlockThreads>
void Test()
{
  Test<T, BlockThreads, 4>();
  Test<T, BlockThreads, 16>();
  Test<T, BlockThreads, 32>();
}


template <int BlockThreads>
void Test()
{
  Test<std::uint16_t, BlockThreads>();
  Test<std::uint32_t, BlockThreads>();
  Test<std::uint64_t, BlockThreads>();
}


int main(int argc, char** argv)
{
  // Initialize command line
  CommandLineArgs args(argc, argv);

  // Initialize device
  CubDebugExit(args.DeviceInit());

  Test<256>();
}
