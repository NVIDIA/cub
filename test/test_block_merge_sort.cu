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
 * Test of BlockMergeSort utilities
 ******************************************************************************/

// Ensure printing of CUDA runtime errors to console
#define CUB_STDERR

#include <stdio.h>
#include <limits>
#include <typeinfo>
#include <memory>

#include <cub/util_allocator.cuh>
#include <cub/block/block_merge_sort.cuh>

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
  std::uint8_t key;
  std::uint64_t count;

  __device__ __host__ CustomType()
    : key(0)
    , count(0)
  {}

  __device__ __host__ CustomType(std::uint64_t value)
    : key(value) // overflow
    , count(value)
  {}

  __device__ __host__ void operator=(std::uint64_t value)
  {
    key = value; // overflow
    count = value;
  }
};


struct CustomLess
{
  template <typename DataType>
  __device__ bool operator()(DataType &lhs, DataType &rhs)
  {
    return lhs < rhs;
  }

  __device__ bool operator()(CustomType &lhs, CustomType &rhs)
  {
    return lhs.key < rhs.key;
  }
};

template <
  typename DataType,
  unsigned int ThreadsInBlock,
  unsigned int ItemsPerThread,
  bool Stable = false>
__global__ void BlockMergeSortTestKernel(DataType *data, unsigned int valid_items)
{
  using BlockMergeSort =
    cub::BlockMergeSort<DataType, ThreadsInBlock, ItemsPerThread>;

  __shared__ typename BlockMergeSort::TempStorage temp_storage_shuffle;

  DataType thread_data[ItemsPerThread];

  const unsigned int thread_offset = threadIdx.x * ItemsPerThread;

  for (unsigned int item = 0; item < ItemsPerThread; item++)
  {
    const unsigned int idx = thread_offset + item;
    thread_data[item] = idx < valid_items ? data[idx] : DataType();
  }
  __syncthreads();

  // Tests below use sequence to fill the data.
  // Therefore the following value should be greater than any that
  // is present in the input data.
  const DataType oob_default =
    static_cast<std::uint64_t>(ThreadsInBlock * ItemsPerThread + 1);

  if (Stable)
  {
    if (valid_items == ThreadsInBlock * ItemsPerThread)
    {
      BlockMergeSort(temp_storage_shuffle).StableSort(
        thread_data,
        CustomLess());
    }
    else
    {
      BlockMergeSort(temp_storage_shuffle).StableSort(
        thread_data,
        CustomLess(),
        valid_items,
        oob_default);
    }
  }
  else
  {
    if (valid_items == ThreadsInBlock * ItemsPerThread)
    {
      BlockMergeSort(temp_storage_shuffle).Sort(
        thread_data,
        CustomLess());
    }
    else
    {
      BlockMergeSort(temp_storage_shuffle).Sort(
        thread_data,
        CustomLess(),
        valid_items,
        oob_default);
    }
  }

  for (unsigned int item = 0; item < ItemsPerThread; item++)
  {
    const unsigned int idx = thread_offset + item;

    if (idx >= valid_items)
      break;

    data[idx] = thread_data[item];
  }
}

template <
  typename KeyType,
  typename ValueType,
  unsigned int ThreadsInBlock,
  unsigned int ItemsPerThread,
  bool Stable = false>
__global__ void BlockMergeSortTestKernel(KeyType *keys,
                                         ValueType *values,
                                         unsigned int valid_items)
{
  using BlockMergeSort =
    cub::BlockMergeSort<KeyType, ThreadsInBlock, ItemsPerThread, ValueType>;

  __shared__ typename BlockMergeSort::TempStorage temp_storage_shuffle;

  KeyType thread_keys[ItemsPerThread];
  ValueType thread_values[ItemsPerThread];

  const unsigned int thread_offset = threadIdx.x * ItemsPerThread;

  for (unsigned int item = 0; item < ItemsPerThread; item++)
  {
    const unsigned int idx = thread_offset + item;
    thread_keys[item] = idx < valid_items ? keys[idx] : KeyType();
    thread_values[item] = idx < valid_items ? values[idx] : ValueType();
  }
  __syncthreads();

  // Tests below use sequence to fill the data.
  // Therefore the following value should be greater than any that
  // is present in the input data.
  const KeyType oob_default = ThreadsInBlock * ItemsPerThread + 1;

  if (Stable)
  {
    if (valid_items == ThreadsInBlock * ItemsPerThread)
    {
      BlockMergeSort(temp_storage_shuffle).StableSort(
        thread_keys,
        thread_values,
        CustomLess());
    }
    else
    {
      BlockMergeSort(temp_storage_shuffle).StableSort(
        thread_keys,
        thread_values,
        CustomLess(),
        valid_items,
        oob_default);
    }
  }
  else
  {
    if (valid_items == ThreadsInBlock * ItemsPerThread)
    {
      BlockMergeSort(temp_storage_shuffle).Sort(
        thread_keys,
        thread_values,
        CustomLess());
    }
    else
    {
      BlockMergeSort(temp_storage_shuffle).Sort(
        thread_keys,
        thread_values,
        CustomLess(),
        valid_items,
        oob_default);
    }
  }

  for (unsigned int item = 0; item < ItemsPerThread; item++)
  {
    const unsigned int idx = thread_offset + item;

    if (idx >= valid_items)
      break;

    keys[idx] = thread_keys[item];
    values[idx] = thread_values[item];
  }
}

template<
  typename DataType,
  unsigned int ItemsPerThread,
  unsigned int ThreadsInBlock,
  bool Stable = false>
void BlockMergeSortTest(DataType *data, unsigned int valid_items)
{
  BlockMergeSortTestKernel<DataType, ThreadsInBlock, ItemsPerThread, Stable>
    <<<1, ThreadsInBlock>>>(data, valid_items);

  CubDebugExit(cudaPeekAtLastError());
  CubDebugExit(cudaDeviceSynchronize());
}

template<
  typename KeyType,
  typename ValueType,
  unsigned int ItemsPerThread,
  unsigned int ThreadsInBlock>
void BlockMergeSortTest(KeyType *keys, ValueType *values, unsigned int valid_items)
{
  BlockMergeSortTestKernel<KeyType, ValueType, ThreadsInBlock, ItemsPerThread>
    <<<1, ThreadsInBlock>>>(keys, values, valid_items);

  CubDebugExit(cudaPeekAtLastError());
  CubDebugExit(cudaDeviceSynchronize());
}

template <typename DataType>
bool CheckResult(int num_items,
                 thrust::device_vector<DataType> &d_data,
                 thrust::host_vector<DataType> &h_data)
{
  thrust::copy_n(d_data.begin(), num_items, h_data.begin());

  for (int i = 0; i < num_items; i++)
  {
    if (h_data[i] != i)
    {
      return false;
    }
  }

  return true;
}

template <
  typename DataType,
  unsigned int ItemsPerThread,
  unsigned int ThreadsInBlock>
void Test(unsigned int num_items,
          thrust::default_random_engine &rng,
          thrust::device_vector<DataType> &d_data,
          thrust::host_vector<DataType> &h_data)
{
  thrust::sequence(d_data.begin(), d_data.end());
  thrust::shuffle(d_data.begin(), d_data.end(), rng);

  BlockMergeSortTest<DataType, ItemsPerThread, ThreadsInBlock>(
      thrust::raw_pointer_cast(d_data.data()), num_items);

  AssertTrue(CheckResult(num_items, d_data, h_data));
}

template <
  typename KeyType,
  typename ValueType,
  unsigned int ItemsPerThread,
  unsigned int ThreadsInBlock>
void Test(unsigned int num_items,
          thrust::default_random_engine &rng,
          thrust::device_vector<KeyType> &d_keys,
          thrust::device_vector<ValueType> &d_values,
          thrust::host_vector<ValueType> &h_data)
{
  thrust::sequence(d_keys.begin(), d_keys.end());
  thrust::shuffle(d_keys.begin(), d_keys.end(), rng);
  thrust::copy_n(d_keys.begin(), num_items, d_values.begin());

  BlockMergeSortTest<KeyType, ValueType, ItemsPerThread, ThreadsInBlock>(
      thrust::raw_pointer_cast(d_keys.data()),
      thrust::raw_pointer_cast(d_values.data()),
      num_items);

  AssertTrue(CheckResult(num_items, d_values, h_data));
}

template <
  typename KeyType,
  typename ValueType,
  unsigned int ItemsPerThread,
  unsigned int ThreadsInBlock>
void Test(thrust::default_random_engine &rng)
{
  for (unsigned int num_items = ItemsPerThread * ThreadsInBlock;
       num_items > 1;
       num_items /= 2)
  {
    thrust::device_vector<KeyType> d_keys(num_items);
    thrust::device_vector<ValueType> d_values(num_items);
    thrust::host_vector<KeyType> h_keys(num_items);
    thrust::host_vector<ValueType> h_values(num_items);

    Test<KeyType, ItemsPerThread, ThreadsInBlock>(num_items,
                                                  rng,
                                                  d_keys,
                                                  h_keys);

    Test<KeyType, ValueType, ItemsPerThread, ThreadsInBlock>(num_items,
                                                             rng,
                                                             d_keys,
                                                             d_values,
                                                             h_values);
  }
}

template <unsigned int ItemsPerThread, unsigned int ThreadsPerBlock>
void Test(thrust::default_random_engine &rng)
{
  Test<std::int32_t, std::int32_t, ItemsPerThread, ThreadsPerBlock>(rng);
  Test<std::int64_t, std::int64_t, ItemsPerThread, ThreadsPerBlock>(rng);

  // Mixed types
  Test<std::int16_t, std::int64_t, ItemsPerThread, ThreadsPerBlock>(rng);
  Test<std::int32_t, std::int64_t, ItemsPerThread, ThreadsPerBlock>(rng);
}

template <unsigned int ItemsPerThread>
void Test(thrust::default_random_engine &rng)
{
  Test<ItemsPerThread, 32>(rng);
  Test<ItemsPerThread, 256>(rng);
}

struct CountToType
{
  __device__ __host__ CustomType operator()(std::uint64_t val)
  {
    return { val };
  }
};

struct CountComparator
{
  __device__ __host__ bool operator()(const CustomType &lhs, const CustomType &rhs)
  {
    if (lhs.key == rhs.key)
      return lhs.count < rhs.count;

    return lhs.key < rhs.key;
  }
};

void TestStability()
{
  constexpr unsigned int items_per_thread = 10;
  constexpr unsigned int threads_per_block = 128;
  constexpr unsigned int elements = items_per_thread * threads_per_block;
  constexpr bool stable = true;

  thrust::device_vector<CustomType> d_keys(elements);
  thrust::device_vector<std::uint64_t> d_counts(elements);
  thrust::sequence(d_counts.begin(), d_counts.end());
  thrust::transform(d_counts.begin(), d_counts.end(), d_keys.begin(), CountToType{});

  // Sort keys
  BlockMergeSortTest<CustomType, items_per_thread, threads_per_block, stable>(
    thrust::raw_pointer_cast(d_keys.data()),
    elements);

  // Check counts
  AssertTrue(thrust::is_sorted(d_keys.begin(), d_keys.end(), CountComparator{}));
}

int main(int argc, char** argv)
{
  CommandLineArgs args(argc, argv);

  // Initialize device
  CubDebugExit(args.DeviceInit());

  thrust::default_random_engine rng;

  Test<1>(rng);
  Test<2>(rng);
  Test<10>(rng);
  Test<15>(rng);

  Test<std::int32_t, std::int32_t, 1, 512>(rng);
  Test<std::int64_t, std::int64_t, 2, 512>(rng);

  TestStability();

  return 0;
}
