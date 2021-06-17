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

/******************************************************************************
 * Test of BlockAdjacentDifference utilities
 ******************************************************************************/

// Ensure printing of CUDA runtime errors to console
#define CUB_STDERR

#include <stdio.h>
#include <limits>
#include <typeinfo>
#include <memory>

#include <cub/util_allocator.cuh>
#include <cub/block/block_adjacent_difference.cuh>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/random.h>
#include <thrust/sequence.h>
#include <thrust/shuffle.h>
#include <thrust/sort.h>

#include "test_util.h"

using namespace cub;

struct CustomType
{
  unsigned int key;
  unsigned int value;

  __device__ __host__ CustomType()
    : key(0)
    , value(0)
  {}

  __device__ __host__ CustomType(unsigned int key, unsigned int value)
    : key(key) // overflow
    , value(value)
  {}
};


__device__ __host__ bool operator==(const CustomType& lhs, const CustomType& rhs)
{
  return lhs.key == rhs.key && lhs.value == rhs.value;
}

__device__ __host__ bool operator!=(const CustomType& lhs, const CustomType& rhs)
{
  return !(lhs == rhs);
}

__device__ __host__ CustomType operator-(const CustomType& lhs, const CustomType& rhs)
{
  return CustomType{lhs.key - rhs.key, lhs.value - rhs.value};
}

struct CustomDifference
{
  template <typename DataType>
  __device__ DataType operator()(DataType &lhs, DataType &rhs)
  {
    return lhs - rhs;
  }
};

template <
  typename DataType,
  unsigned int ThreadsInBlock,
  unsigned int ItemsPerThread,
  bool ReadLeft = false>
__global__ void BlockAdjacentDifferenceLastTileTestKernel(DataType *data,
                                                          unsigned int valid_items)
{
  using BlockAdjacentDifferenceT =
    cub::BlockAdjacentDifference<DataType, ThreadsInBlock>;

  __shared__ typename BlockAdjacentDifferenceT::TempStorage temp_storage;

  DataType thread_data[ItemsPerThread];
  DataType thread_result[ItemsPerThread];

  const unsigned int thread_offset = threadIdx.x * ItemsPerThread;

  for (unsigned int item = 0; item < ItemsPerThread; item++)
  {
    const unsigned int idx = thread_offset + item;
    thread_data[item] = idx < valid_items ? data[idx] : DataType();
  }
  __syncthreads();

  if (ReadLeft)
  {
    DataType thread_preds[ItemsPerThread];

    BlockAdjacentDifferenceT(temp_storage).SubtractLeft(
      thread_result,
      thread_data,
      thread_preds,
      CustomDifference());
  }
  else
  {
    BlockAdjacentDifferenceT(temp_storage).SubtractRightPartialTile(
      thread_result,
      thread_data,
      CustomDifference(),
      valid_items);
  }

  for (unsigned int item = 0; item < ItemsPerThread; item++)
  {
    const unsigned int idx = thread_offset + item;

    if (idx >= valid_items)
      break;

    data[idx] = thread_result[item];
  }
}

template <typename DataType,
          unsigned int ThreadsInBlock,
          unsigned int ItemsPerThread,
          bool ReadLeft = false>
__global__ void BlockAdjacentDifferenceTestKernel(DataType *data,
                                                  unsigned int valid_items,
                                                  DataType oob_default)
{
  using BlockAdjacentDifferenceT =
    cub::BlockAdjacentDifference<DataType, ThreadsInBlock>;

  __shared__
    typename BlockAdjacentDifferenceT::TempStorage temp_storage;

  DataType thread_data[ItemsPerThread];
  DataType thread_result[ItemsPerThread];

  const unsigned int thread_offset = threadIdx.x * ItemsPerThread;

  for (unsigned int item = 0; item < ItemsPerThread; item++)
  {
    const unsigned int idx = thread_offset + item;
    thread_data[item]      = idx < valid_items ? data[idx] : DataType();
  }
  __syncthreads();

  if (ReadLeft)
  {
    DataType thread_preds[ItemsPerThread];

    BlockAdjacentDifferenceT(temp_storage)
      .SubtractLeft(thread_result,
                    thread_data,
                    thread_preds,
                    CustomDifference(),
                    oob_default);
  }
  else
  {
    BlockAdjacentDifferenceT(temp_storage)
      .SubtractRight(thread_result,
                     thread_data,
                     CustomDifference(),
                     oob_default);
  }

  for (unsigned int item = 0; item < ItemsPerThread; item++)
  {
    const unsigned int idx = thread_offset + item;

    if (idx >= valid_items)
      break;

    data[idx] = thread_result[item];
  }
}

template <typename DataType,
          unsigned int ItemsPerThread,
          unsigned int ThreadsInBlock,
          bool ReadLeft = false>
void BlockAdjacentDifferenceLastTileTest(DataType *data, unsigned int valid_items)
{
  BlockAdjacentDifferenceLastTileTestKernel<DataType,
                                            ThreadsInBlock,
                                            ItemsPerThread,
                                            ReadLeft>
    <<<1, ThreadsInBlock>>>(data, valid_items);

  CubDebugExit(cudaPeekAtLastError());
  CubDebugExit(cudaDeviceSynchronize());
}

template <typename DataType,
          unsigned int ItemsPerThread,
          unsigned int ThreadsInBlock,
          bool ReadLeft = false>
void BlockAdjacentDifferenceTest(DataType *data,
                                 unsigned int valid_items,
                                 DataType oob_default)
{
  BlockAdjacentDifferenceTestKernel<DataType,
                                    ThreadsInBlock,
                                    ItemsPerThread,
                                    ReadLeft>
    <<<1, ThreadsInBlock>>>(data, valid_items, oob_default);

  CubDebugExit(cudaPeekAtLastError());
  CubDebugExit(cudaDeviceSynchronize());
}

template <
  typename DataType,
  unsigned int ItemsPerThread,
  unsigned int ThreadsInBlock>
void TestLastTile(unsigned int num_items,
                  thrust::device_vector<DataType> &d_data)
{
  thrust::sequence(d_data.begin(), d_data.end());

  constexpr bool read_left = true;
  constexpr bool read_right = false;

  BlockAdjacentDifferenceLastTileTest<DataType,
                                      ItemsPerThread,
                                      ThreadsInBlock,
                                      read_left>(thrust::raw_pointer_cast(
                                                   d_data.data()),
                                                 num_items);

  {
    const std::size_t expected_count = num_items - 1;
    const std::size_t actual_count =
      thrust::count(d_data.begin(), d_data.end(), static_cast<DataType>(1));

    AssertEquals(expected_count, actual_count);
    AssertEquals(d_data.front(), DataType{0});
  }

  thrust::sequence(d_data.begin(), d_data.end());

  BlockAdjacentDifferenceLastTileTest<DataType,
                                      ItemsPerThread,
                                      ThreadsInBlock,
                                      read_right>(thrust::raw_pointer_cast(
                                                    d_data.data()),
                                                  num_items);

  {
    AssertEquals(thrust::count(d_data.begin(), d_data.end() - 1,
                 static_cast<DataType>(-1)), num_items - 1);

    const auto expected_value = static_cast<DataType>(num_items - 1);
    const DataType actual_value = d_data.back();

    AssertEquals(actual_value, expected_value);
  }
}

struct IntToCustomType
{
  unsigned int offset;

  IntToCustomType()
      : offset(0)
  {}

  explicit IntToCustomType(unsigned int offset)
      : offset(offset)
  {}

  __device__ __host__ CustomType operator()(unsigned int idx) const
  {
    return { idx + offset, idx + offset };
  }
};

template <unsigned int ItemsPerThread,
          unsigned int ThreadsInBlock>
void TestLastTile(unsigned int num_items,
                  thrust::device_vector<CustomType> &d_data)
{
  thrust::tabulate(d_data.begin(), d_data.end(), IntToCustomType());

  constexpr bool read_left  = true;
  constexpr bool read_right = false;

  BlockAdjacentDifferenceLastTileTest<CustomType,
                                      ItemsPerThread,
                                      ThreadsInBlock,
                                      read_left>(thrust::raw_pointer_cast(
                                                   d_data.data()),
                                                 num_items);

  {
    const std::size_t expected_count = num_items - 1;
    const std::size_t actual_count =
      thrust::count(d_data.begin(), d_data.end(), CustomType{1, 1});

    AssertEquals(expected_count, actual_count);
    AssertEquals(d_data.front(), CustomType{});
  }

  thrust::tabulate(d_data.begin(), d_data.end(), IntToCustomType());

  BlockAdjacentDifferenceLastTileTest<CustomType,
                                      ItemsPerThread,
                                      ThreadsInBlock,
                                      read_right>(thrust::raw_pointer_cast(
                                                    d_data.data()),
                                                  num_items);

  {
    const auto unsigned_minus_one = static_cast<unsigned int>(-1);
    const auto expected_count = static_cast<std::size_t>(num_items - 1);
    const std::size_t actual_count =
      thrust::count(d_data.begin(),
                    d_data.end() - 1,
                    CustomType{unsigned_minus_one, unsigned_minus_one});

    AssertEquals(actual_count, expected_count);
  }

  {
    const CustomType expected_value = CustomType{num_items - 1, num_items - 1};
    const CustomType actual_value   = d_data.back();

    AssertEquals(actual_value, expected_value);
  }
}

template <typename DataType,
  unsigned int ItemsPerThread,
  unsigned int ThreadsInBlock>
void Test(unsigned int num_items,
          thrust::device_vector<DataType> &d_data)
{
  thrust::sequence(d_data.begin(), d_data.end(), DataType{1});

  constexpr bool read_left  = true;
  constexpr bool read_right = false;

  BlockAdjacentDifferenceTest<DataType, ItemsPerThread, ThreadsInBlock, read_left>(
    thrust::raw_pointer_cast(d_data.data()),
    num_items,
    DataType{0});

  {
    const std::size_t expected_count = num_items;
    const std::size_t actual_count =
      thrust::count(d_data.begin(), d_data.end(), DataType{1});

    AssertEquals(expected_count, actual_count);
  }

  thrust::sequence(d_data.begin(), d_data.end());

  BlockAdjacentDifferenceTest<DataType,
                              ItemsPerThread,
                              ThreadsInBlock,
                              read_right>(thrust::raw_pointer_cast(
                                            d_data.data()),
                                          num_items,
                                          static_cast<DataType>(num_items));

  {
    const std::size_t expected_count = num_items;
    const std::size_t actual_count =
      thrust::count(d_data.begin(), d_data.end(), static_cast<DataType>(-1));

    AssertEquals(expected_count, actual_count);
  }
}

template <unsigned int ItemsPerThread,
          unsigned int ThreadsInBlock>
void TestBaseCase(unsigned int num_items,
                  thrust::device_vector<CustomType> &d_data)
{
  thrust::tabulate(d_data.begin(), d_data.end(), IntToCustomType{1});

  constexpr bool read_left  = true;
  constexpr bool read_right = false;

  BlockAdjacentDifferenceTest<CustomType,
                              ItemsPerThread,
                              ThreadsInBlock,
                              read_left>(thrust::raw_pointer_cast(
                                           d_data.data()),
                                         num_items,
                                         CustomType{});

  {
    const std::size_t expected_count = num_items;
    const std::size_t actual_count =
      thrust::count(d_data.begin(), d_data.end(), CustomType{1, 1});

    AssertEquals(expected_count, actual_count);
  }

  thrust::tabulate(d_data.begin(), d_data.end(), IntToCustomType{});

  BlockAdjacentDifferenceTest<CustomType,
                              ItemsPerThread,
                              ThreadsInBlock,
                              read_right>(thrust::raw_pointer_cast(
                                            d_data.data()),
                                          num_items,
                                          CustomType{num_items, num_items});

  {
    const auto unsigned_minus_one = static_cast<unsigned int>(-1);

    const std::size_t expected_count = num_items;
    const std::size_t actual_count =
      thrust::count(d_data.begin(),
                    d_data.end(),
                    CustomType{unsigned_minus_one, unsigned_minus_one});

    AssertEquals(expected_count, actual_count);
  }
}

template <
  typename ValueType,
  unsigned int ItemsPerThread,
  unsigned int ThreadsInBlock>
void Test()
{
  constexpr int tile_size = ItemsPerThread * ThreadsInBlock;
  thrust::device_vector<ValueType> d_values(tile_size);

  for (unsigned int num_items = tile_size; num_items > 1; num_items /= 2)
  {
    d_values.resize(num_items);
    TestLastTile<ValueType, ItemsPerThread, ThreadsInBlock>(num_items, d_values);
  }

  d_values.resize(tile_size);
  Test<ValueType, ItemsPerThread, ThreadsInBlock>(tile_size, d_values);
}

template <
  unsigned int ItemsPerThread,
  unsigned int ThreadsInBlock>
void TestCustomType()
{
  constexpr int tile_size = ItemsPerThread * ThreadsInBlock;
  thrust::device_vector<CustomType> d_values(tile_size);

  for (unsigned int num_items = tile_size; num_items > 1; num_items /= 2)
  {
    d_values.resize(num_items);
    TestLastTile<ItemsPerThread, ThreadsInBlock>(num_items, d_values);
  }

  d_values.resize(tile_size);
  TestBaseCase<ItemsPerThread, ThreadsInBlock>(tile_size, d_values);
}

template <unsigned int ItemsPerThread, unsigned int ThreadsPerBlock>
void Test()
{
  Test<std::uint8_t,  ItemsPerThread, ThreadsPerBlock>();
  Test<std::uint16_t, ItemsPerThread, ThreadsPerBlock>();
  Test<std::uint32_t, ItemsPerThread, ThreadsPerBlock>();
  Test<std::uint64_t, ItemsPerThread, ThreadsPerBlock>();

  TestCustomType<ItemsPerThread, ThreadsPerBlock>();
}

template <unsigned int ItemsPerThread>
void Test()
{
  Test<ItemsPerThread, 32>();
  Test<ItemsPerThread, 256>();
}

int main(int argc, char** argv)
{
  CommandLineArgs args(argc, argv);

  // Initialize device
  CubDebugExit(args.DeviceInit());

  Test<1>();
  Test<2>();
  Test<10>();
  Test<15>();

  return 0;
}
