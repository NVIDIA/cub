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

#include <cub/block/block_adjacent_difference.cuh>
#include <cub/util_allocator.cuh>

#include <thrust/count.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/mismatch.h>
#include <thrust/random.h>
#include <thrust/sequence.h>
#include <thrust/shuffle.h>
#include <thrust/sort.h>
#include <thrust/tabulate.h>

#include <limits>
#include <memory>
#include <typeinfo>

#include "test_util.h"


using namespace cub;

template <typename DataType>
using CountingIteratorT =
  typename thrust::counting_iterator<DataType,
    thrust::use_default,
    std::size_t,
    std::size_t>;

/**
 * \brief Generates integer sequence \f$S_n=i(i-1)/2\f$.
 *
 * The adjacent difference of this sequence produce consecutive numbers:
 * \f[
 *   p = \frac{i(i - 1)}{2} \\
 *   n = \frac{(i + 1) i}{2} \\
 *   n - p = i \\
 *   \frac{(i + 1) i}{2} - \frac{i (i - 1)}{2} = i \\
 *   (i + 1) i - i (i - 1) = 2 i \\
 *   (i + 1) - (i - 1) = 2 \\
 *   2 = 2
 * \f]
 */
template <typename DestT>
struct TestSequenceGenerator
{
  std::size_t offset;

  TestSequenceGenerator(std::size_t offset = 0)
      : offset(offset)
  {}

  template <typename SourceT>
  __device__ __host__ DestT operator()(SourceT index) const
  {
    index += static_cast<SourceT>(offset);
    return static_cast<DestT>(index * (index - 1) / SourceT(2));
  }
};



struct CustomType
{
  unsigned int key;
  unsigned int value;

  __device__ __host__ CustomType()
    : key(0)
    , value(0)
  {}

  __device__ __host__ CustomType(unsigned int key, unsigned int value)
    : key(key)
    , value(value)
  {}
};


__device__ __host__ bool operator==(const CustomType& lhs,
                                    const CustomType& rhs)
{
  return lhs.key == rhs.key && lhs.value == rhs.value;
}

__device__ __host__ bool operator!=(const CustomType& lhs,
                                    const CustomType& rhs)
{
  return !(lhs == rhs);
}

__device__ __host__ CustomType operator-(const CustomType& lhs,
                                         const CustomType& rhs)
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

template <typename DataType,
          unsigned int ThreadsInBlock,
          unsigned int ItemsPerThread,
          typename ActionT>
__global__ void AdjDiffKernel(const DataType *input,
                              DataType *output,
                              ActionT action,
                              bool in_place)
{
  using BlockAdjacentDifferenceT =
    cub::BlockAdjacentDifference<DataType, ThreadsInBlock>;

  __shared__ typename BlockAdjacentDifferenceT::TempStorage temp_storage;

  DataType thread_data[ItemsPerThread];
  DataType thread_result[ItemsPerThread];

  const unsigned int thread_offset = threadIdx.x * ItemsPerThread;

  for (unsigned int item = 0; item < ItemsPerThread; item++)
  {
    thread_data[item] = input[thread_offset + item];
  }
  __syncthreads();

  BlockAdjacentDifferenceT block_adj_diff(temp_storage);

  if (in_place)
  {
    action(thread_data, thread_data, block_adj_diff);

    for (unsigned int item = 0; item < ItemsPerThread; item++)
    {
      output[thread_offset + item] = thread_data[item];
    }
  }
  else 
  {
    action(thread_data, thread_result, block_adj_diff);

    for (unsigned int item = 0; item < ItemsPerThread; item++)
    {
      output[thread_offset + item] = thread_result[item];
    }
  }
}

template <typename DataType,
          unsigned int ItemsPerThread,
          unsigned int ThreadsInBlock,
          typename ActionT>
void AdjDiffTest(const DataType *input,
                 DataType *output,
                 ActionT action,
                 bool in_place = false)
{
  AdjDiffKernel<DataType, ThreadsInBlock, ItemsPerThread, ActionT>
    <<<1, ThreadsInBlock>>>(input, output, action, in_place);

  CubDebugExit(cudaPeekAtLastError());
  CubDebugExit(cudaDeviceSynchronize());
}

template <bool ReadLeft>
struct LastTileOpT
{
  unsigned int m_valid_items{};

  __host__ LastTileOpT(unsigned int valid_items)
    : m_valid_items(valid_items)
  {}

  template <int ITEMS_PER_THREAD,
            typename T,
            typename BlockAdjDiff>
  __device__ void operator()(T (&input)[ITEMS_PER_THREAD],
                             T (&output)[ITEMS_PER_THREAD],
                             BlockAdjDiff &block_adj_diff) const
  {
    if (ReadLeft)
    {
      block_adj_diff.SubtractLeftPartialTile(input,
                                             output,
                                             CustomDifference(),
                                             m_valid_items);
    }
    else
    {
      block_adj_diff.SubtractRightPartialTile(input,
                                              output,
                                              CustomDifference(),
                                              m_valid_items);
    }
  }
};

template <typename DataType, 
          bool ReadLeft>
struct MiddleTileOpT
{
  DataType m_neighbour_tile_value;

  __host__ MiddleTileOpT(DataType neighbour_tile_value)
    : m_neighbour_tile_value(neighbour_tile_value)
  {}

  template <int ITEMS_PER_THREAD, typename T, typename BlockAdjDiff>
  __device__ void operator()(T (&input)[ITEMS_PER_THREAD],
                             T (&output)[ITEMS_PER_THREAD],
                             BlockAdjDiff &block_adj_diff) const
  {
    if (ReadLeft)
    {
      block_adj_diff.SubtractLeft(input,
                                  output,
                                  CustomDifference(),
                                  m_neighbour_tile_value);
    }
    else
    {
      block_adj_diff.SubtractRight(input,
                                   output,
                                   CustomDifference(),
                                   m_neighbour_tile_value);
    }
  }
};

template <bool ReadLeft>
struct BaseOpT
{
  template <int ITEMS_PER_THREAD, typename T, typename BlockAdjDiff>
  __device__ void operator()(T (&input)[ITEMS_PER_THREAD],
                             T (&output)[ITEMS_PER_THREAD],
                             BlockAdjDiff &block_adj_diff) const
  {
    if (ReadLeft)
    {
      block_adj_diff.SubtractLeft(input,
                                  output,
                                  CustomDifference());
    }
    else
    {
      block_adj_diff.SubtractRight(input,
                                   output,
                                   CustomDifference());
    }
  }
};


template <typename DataType,
          unsigned int ItemsPerThread,
          unsigned int ThreadsInBlock,
          bool ReadLeft = false>
void LastTileTest(const DataType *input,
                  DataType *output,
                  unsigned int valid_items,
                  bool in_place)
{
  AdjDiffTest<DataType, ItemsPerThread, ThreadsInBlock>(input,
                                                        output,
                                                        LastTileOpT<ReadLeft>{
                                                          valid_items},
                                                        in_place);
}

template <typename DataType>
struct LastTileWithPredOpT
{
  unsigned int m_valid_items;
  DataType m_neighbour_tile_value;

  __host__ LastTileWithPredOpT(
      unsigned int valid_items,
      DataType neighbour_tile_value)
    : m_valid_items(valid_items)
    , m_neighbour_tile_value(neighbour_tile_value)
  {
  }

  template <int ITEMS_PER_THREAD, typename T, typename BlockAdjDiff>
  __device__ void operator()(T (&input)[ITEMS_PER_THREAD],
                             T (&output)[ITEMS_PER_THREAD],
                             BlockAdjDiff &block_adj_diff) const
  {
    block_adj_diff.SubtractLeftPartialTile(input,
                                           output,
                                           CustomDifference(),
                                           m_valid_items,
                                           m_neighbour_tile_value);
  }
};

template <typename DataType,
          unsigned int ItemsPerThread,
          unsigned int ThreadsInBlock>
void LastTileWithPredTest(const DataType *input,
                          DataType *output,
                          unsigned int valid_items,
                          DataType neighbour_tile_value,
                          bool in_place)
{
  AdjDiffTest<DataType, ItemsPerThread, ThreadsInBlock>(
    input,
    output,
    LastTileWithPredOpT<DataType>{valid_items, neighbour_tile_value},
    in_place);
}

template <typename DataType,
          unsigned int ItemsPerThread,
          unsigned int ThreadsInBlock,
          bool ReadLeft = false>
void Test(DataType *data,
          bool in_place)
{
  AdjDiffTest<DataType, ItemsPerThread, ThreadsInBlock>(
    data,
    data,
    BaseOpT<ReadLeft>{},
    in_place);
}


template <typename DataType,
          unsigned int ItemsPerThread,
          unsigned int ThreadsInBlock,
          bool ReadLeft = false>
void MiddleTileTest(const DataType *input,
                    DataType *output,
                    DataType neighbour_tile_value,
                    bool in_place)
{
  AdjDiffTest<DataType, ItemsPerThread, ThreadsInBlock>(
    input,
    output,
    MiddleTileOpT<DataType, ReadLeft>{neighbour_tile_value},
    in_place);
}


template <typename FirstIteratorT,
          typename SecondOperatorT>
bool CheckResult(FirstIteratorT first_begin,
                 FirstIteratorT first_end,
                 SecondOperatorT second_begin)
{
  auto err = thrust::mismatch(first_begin, first_end, second_begin);

  if (err.first != first_end)
  {
    return false;
  }

  return true;
}


template <typename DataType,
          unsigned int ItemsPerThread,
          unsigned int ThreadsInBlock>
void TestLastTile(bool inplace,
                  unsigned int num_items,
                  thrust::device_vector<DataType> &d_input)
{
  thrust::tabulate(d_input.begin(),
                   d_input.end(),
                   TestSequenceGenerator<DataType>{});
  thrust::device_vector<DataType> d_output(d_input.size());

  constexpr bool read_left = true;
  constexpr bool read_right = false;

  DataType *d_input_ptr = thrust::raw_pointer_cast(d_input.data());
  DataType *d_output_ptr = thrust::raw_pointer_cast(d_output.data());

  LastTileTest<DataType, ItemsPerThread, ThreadsInBlock, read_left>(
    d_input_ptr,
    d_output_ptr,
    num_items,
    inplace);

  {
    AssertEquals(d_output.front(), d_input.front());
    AssertTrue(CheckResult(d_output.begin() + 1,
                           d_output.begin() + num_items,
                           CountingIteratorT<DataType>(DataType{0})));
    AssertTrue(CheckResult(d_output.begin() + num_items,
                           d_output.end(),
                           d_input.begin() + num_items));
  }

  if (num_items > 0)
  {
    LastTileWithPredTest<DataType, ItemsPerThread, ThreadsInBlock>(
      d_input_ptr +  1,
      d_output_ptr,
      num_items - 1,
      TestSequenceGenerator<DataType>{}(0),
      inplace);

    AssertTrue(CheckResult(d_output.begin(),
                           d_output.begin() + num_items - 1,
                           CountingIteratorT<DataType>(DataType{0})));
    AssertTrue(CheckResult(d_output.begin() + num_items - 1,
                           d_output.end() - 1,
                           d_input.begin() + num_items));
  }

  thrust::tabulate(d_input.begin(),
                   d_input.end(),
                   TestSequenceGenerator<DataType>{});

  LastTileTest<DataType, ItemsPerThread, ThreadsInBlock, read_right>(
    d_input_ptr,
    d_output_ptr,
    num_items,
    inplace);

  {
    thrust::device_vector<DataType> reference(num_items);
    thrust::sequence(reference.begin(),
                     reference.end(),
                     static_cast<DataType>(0),
                     static_cast<DataType>(-1));

    AssertTrue(CheckResult(d_output.begin(),
                           d_output.begin() + num_items - 1,
                           reference.begin()));
    AssertTrue(CheckResult(d_output.begin() + num_items - 1,
                           d_output.end(),
                           d_input.begin() + num_items - 1));
  }
}


template <typename DataType,
          unsigned int ItemsPerThread,
          unsigned int ThreadsInBlock>
void TestMiddleTile(bool inplace,
                    thrust::device_vector<DataType> &d_input)
{
  thrust::tabulate(d_input.begin(),
                   d_input.end(),
                   TestSequenceGenerator<DataType>{std::size_t{1}});
  thrust::device_vector<DataType> d_output(d_input.size());

  constexpr bool read_left  = true;
  constexpr bool read_right = false;

  DataType *d_input_ptr  = thrust::raw_pointer_cast(d_input.data());
  DataType *d_output_ptr = thrust::raw_pointer_cast(d_output.data());

  const DataType left_tile_last_value{0};
  const DataType right_tile_first_value{
    TestSequenceGenerator<DataType>{}(d_input.size())
  };

  MiddleTileTest<DataType, ItemsPerThread, ThreadsInBlock, read_left>(
    d_input_ptr,
    d_output_ptr,
    left_tile_last_value,
    inplace);

  {
    AssertTrue(CheckResult(d_output.begin(),
                           d_output.end(),
                           CountingIteratorT<DataType>(DataType{0})));
  }

  thrust::tabulate(d_input.begin(),
                   d_input.end(),
                   TestSequenceGenerator<DataType>{});

  MiddleTileTest<DataType, ItemsPerThread, ThreadsInBlock, read_right>(
    d_input_ptr,
    d_output_ptr,
    right_tile_first_value,
    inplace);

  {
    thrust::device_vector<DataType> reference(d_input.size());
    thrust::sequence(reference.begin(),
                     reference.end(),
                     static_cast<DataType>(0),
                     static_cast<DataType>(-1));

    AssertTrue(CheckResult(d_output.begin(),
                           d_output.end(),
                           reference.begin()));
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


template <typename DataType,
          unsigned int ItemsPerThread,
          unsigned int ThreadsInBlock>
void TestFullTile(bool inplace,
                  thrust::device_vector<DataType> &d_data)
{
  thrust::tabulate(d_data.begin(),
                   d_data.end(),
                   TestSequenceGenerator<DataType>{});

  constexpr bool read_left  = true;
  constexpr bool read_right = false;

  DataType *d_data_ptr = thrust::raw_pointer_cast(d_data.data());

  Test<DataType, ItemsPerThread, ThreadsInBlock, read_left>(d_data_ptr,
                                                            inplace);
  {
    AssertEquals(d_data.front(), TestSequenceGenerator<DataType>{}(0));
    AssertTrue(CheckResult(d_data.begin() + 1,
                           d_data.end(),
                           CountingIteratorT<DataType>(DataType{0})));
  }

  thrust::tabulate(d_data.begin(),
                   d_data.end(),
                   TestSequenceGenerator<DataType>{});

  Test<DataType, ItemsPerThread, ThreadsInBlock, read_right>(d_data_ptr,
                                                             inplace);

  {
    thrust::device_vector<DataType> reference(d_data.size());
    thrust::sequence(reference.begin(),
                     reference.end(),
                     static_cast<DataType>(0),
                     static_cast<DataType>(-1));

    AssertTrue(CheckResult(d_data.begin(),
                           d_data.end() - 1,
                           reference.begin()));
    AssertEquals(d_data.back(),
                 TestSequenceGenerator<DataType>{}(d_data.size() - 1));
  }
}


template <unsigned int ItemsPerThread,
          unsigned int ThreadsInBlock>
void TestCustomType(bool inplace,
                    thrust::device_vector<CustomType> &d_data)
{
  thrust::tabulate(d_data.begin(), d_data.end(), IntToCustomType{1});
  CustomType *d_data_ptr = thrust::raw_pointer_cast(d_data.data());

  constexpr bool read_left  = true;
  constexpr bool read_right = false;

  Test<CustomType, ItemsPerThread, ThreadsInBlock, read_left>(d_data_ptr,
                                                              inplace);

  {
    const std::size_t expected_count = d_data.size();
    const std::size_t actual_count =
      thrust::count(d_data.begin(), d_data.end(), CustomType{1, 1});

    AssertEquals(expected_count, actual_count);
  }

  thrust::tabulate(d_data.begin(), d_data.end(), IntToCustomType{});

  Test<CustomType, ItemsPerThread, ThreadsInBlock, read_right>(d_data_ptr,
                                                               inplace);

  {
    const auto unsigned_minus_one = static_cast<unsigned int>(-1);

    const std::size_t expected_count = d_data.size() - 1;
    const std::size_t actual_count =
      thrust::count(d_data.begin(),
                    d_data.end() - 1,
                    CustomType{unsigned_minus_one, unsigned_minus_one});

    AssertEquals(expected_count, actual_count);
  }
}


template <
  typename ValueType,
  unsigned int ItemsPerThread,
  unsigned int ThreadsInBlock>
void Test(bool inplace)
{
  constexpr int tile_size = ItemsPerThread * ThreadsInBlock;
  thrust::device_vector<ValueType> d_values(tile_size);

  for (unsigned int num_items = tile_size; num_items > 1; num_items /= 2)
  {
    TestLastTile<ValueType, ItemsPerThread, ThreadsInBlock>(inplace,
                                                            num_items,
                                                            d_values);
  }

  TestFullTile<ValueType, ItemsPerThread, ThreadsInBlock>(inplace, d_values);
  TestMiddleTile<ValueType, ItemsPerThread, ThreadsInBlock>(inplace, d_values);
}


template <unsigned int ItemsPerThread,
          unsigned int ThreadsInBlock>
void TestCustomType(bool inplace)
{
  constexpr int tile_size = ItemsPerThread * ThreadsInBlock;
  thrust::device_vector<CustomType> d_values(tile_size);
  TestCustomType<ItemsPerThread, ThreadsInBlock>(inplace, d_values);
}


template <unsigned int ItemsPerThread, unsigned int ThreadsPerBlock>
void Test(bool inplace)
{
  Test<std::uint8_t,  ItemsPerThread, ThreadsPerBlock>(inplace);
  Test<std::uint16_t, ItemsPerThread, ThreadsPerBlock>(inplace);
  Test<std::uint32_t, ItemsPerThread, ThreadsPerBlock>(inplace);
  Test<std::uint64_t, ItemsPerThread, ThreadsPerBlock>(inplace);
}


template <unsigned int ItemsPerThread>
void Test(bool inplace)
{
  Test<ItemsPerThread, 32>(inplace);
  Test<ItemsPerThread, 256>(inplace);
}


template <unsigned int ItemsPerThread>
void Test()
{
  Test<ItemsPerThread>(false);
  Test<ItemsPerThread>(true);
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

  // More of a compilation check
  TestCustomType<5, 256>(true);

  return 0;
}

