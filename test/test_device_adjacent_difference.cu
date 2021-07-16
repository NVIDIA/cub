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

#include <limits>
#include <memory>

#include <cub/util_allocator.cuh>
#include "cub/device/device_adjacent_difference.cuh"

#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>
#include <thrust/shuffle.h>
#include <thrust/random.h>

#include "test_util.h"

using namespace cub;

template <bool ReadLeft,
          typename ValueT,
          typename OffsetT>
void test(OffsetT elements, thrust::default_random_engine &rng)
{
  thrust::device_vector<ValueT> input(elements, ValueT{});
  thrust::device_vector<ValueT> output(elements, ValueT{42});

  if (ReadLeft)
  {
    cub::DeviceAdjacentDifference::SubtractLeftCopy(
      thrust::raw_pointer_cast(input.data()),
      thrust::raw_pointer_cast(output.data()),
      elements);

    AssertEquals(elements, thrust::count(output.begin(), output.end(), ValueT{}));
  }
  else
  {
    cub::DeviceAdjacentDifference::SubtractRightCopy(
      thrust::raw_pointer_cast(input.data()),
      thrust::raw_pointer_cast(output.data()),
      elements);

    AssertEquals(elements, thrust::count(output.begin(), output.end(), ValueT{}));
  }

  thrust::sequence(input.begin(), input.end(), ValueT{1});

  if (ReadLeft)
  {
    cub::DeviceAdjacentDifference::SubtractLeftCopy(
      thrust::raw_pointer_cast(input.data()),
      thrust::raw_pointer_cast(output.data()),
      elements);

    AssertEquals(elements, thrust::count(output.begin(), output.end(), ValueT(1)));

    thrust::shuffle(input.begin(), input.end(), rng);

    cub::DeviceAdjacentDifference::SubtractLeftCopy(
      thrust::raw_pointer_cast(input.data()),
      thrust::raw_pointer_cast(output.data()),
      elements);

    thrust::host_vector<ValueT> tmp(input);
    thrust::adjacent_difference(tmp.begin(), tmp.end(), tmp.begin());

    AssertEquals(tmp, output);
  }
  else
  {
    cub::DeviceAdjacentDifference::SubtractRightCopy(
      thrust::raw_pointer_cast(input.data()),
      thrust::raw_pointer_cast(output.data()),
      elements);

    AssertEquals(elements - 1, thrust::count(output.begin(), output.end(), static_cast<ValueT>(-1)));
    AssertEquals(output.back(), static_cast<ValueT>(elements));
  }
}

template <bool ReadLeft,
          typename ValueT,
          typename OffsetT>
void test_iterator(OffsetT elements)
{
  thrust::counting_iterator<ValueT> count_iter(ValueT(1));
  thrust::device_vector<ValueT> output(elements, ValueT(42));

  if (ReadLeft)
  {
    cub::DeviceAdjacentDifference::SubtractLeftCopy(count_iter,
                                                    thrust::raw_pointer_cast(
                                                      output.data()),
                                                    elements);

    AssertEquals(elements, thrust::count(output.begin(), output.end(), ValueT(1)));
  }
  else
  {
    cub::DeviceAdjacentDifference::SubtractRightCopy(count_iter,
                                                     thrust::raw_pointer_cast(
                                                       output.data()),
                                                     elements);

    AssertEquals(elements - 1, thrust::count(output.begin(), output.end() - 1, static_cast<ValueT>(-1)));
    AssertEquals(output.back(), static_cast<ValueT>(elements));
  }

  thrust::constant_iterator<ValueT> const_iter(ValueT(0));

  if(ReadLeft)
  {
    cub::DeviceAdjacentDifference::SubtractLeftCopy(const_iter,
                                                    thrust::raw_pointer_cast(
                                                      output.data()),
                                                    elements);

    AssertEquals(elements, thrust::count(output.begin(), output.end(), ValueT(0)));
  }
  else
  {
    cub::DeviceAdjacentDifference::SubtractRightCopy(const_iter,
                                                     thrust::raw_pointer_cast(
                                                       output.data()),
                                                     elements);

    AssertEquals(elements, thrust::count(output.begin(), output.end(), ValueT(0)));
  }
}

template <bool ReadLeft,
          typename ValueT,
          typename OffsetT>
void test_iterator_with_custom_op(OffsetT elements)
{
  thrust::device_vector<ValueT> output(elements, ValueT(42));

  if (ReadLeft)
  {
    thrust::constant_iterator<ValueT> iter(ValueT{1});
    cub::DeviceAdjacentDifference::SubtractLeftCopy(iter,
                                                    thrust::raw_pointer_cast(
                                                      output.data()),
                                                    elements,
                                                    cub::Sum());

    AssertEquals(elements - 1, thrust::count(output.begin(), output.end(), ValueT{2}));
    AssertEquals(ValueT{1}, thrust::count(output.begin(), output.end(), ValueT{1}));
  }
  else
  {
    thrust::counting_iterator<ValueT> iter{ValueT(0)};

    cub::DeviceAdjacentDifference::SubtractRightCopy(
      iter,
      thrust::raw_pointer_cast(output.data()),
      elements,
      cub::MakeBinaryFlip(cub::Difference()));

    AssertEquals(elements - 1, thrust::count(output.begin(), output.end(), ValueT{1}));

    {
      const auto expected_value = static_cast<ValueT>(elements - 1);
      const ValueT actual_value = output[elements - 1];

      if (actual_value != expected_value)
      {
        std::cout << ":(";
        cub::DeviceAdjacentDifference::SubtractRightCopy(
          iter,
          thrust::raw_pointer_cast(output.data()),
          elements,
          cub::MakeBinaryFlip(cub::Difference()));
      }

      AssertEquals(actual_value, expected_value);
    }
  }
}

template <bool ReadLeft,
          typename ValueT,
          typename OffsetT>
void test_inplace(OffsetT elements)
{
  thrust::device_vector<ValueT> input(elements, ValueT());

  size_t temp_storage_size{};
  cub::DeviceAdjacentDifference::SubtractLeft(
    nullptr,
    temp_storage_size,
    thrust::raw_pointer_cast(input.data()),
    elements);

  thrust::device_vector<uint8_t> tmp_storage(temp_storage_size);

  if (ReadLeft)
  {
    cub::DeviceAdjacentDifference::SubtractLeft(
      thrust::raw_pointer_cast(tmp_storage.data()),
      temp_storage_size,
      thrust::raw_pointer_cast(input.data()),
      elements);

    AssertEquals(elements, thrust::count(input.begin(), input.end(), ValueT()));

    thrust::sequence(input.begin(), input.end(), ValueT(1));

    cub::DeviceAdjacentDifference::SubtractLeft(
      thrust::raw_pointer_cast(tmp_storage.data()),
      temp_storage_size,
      thrust::raw_pointer_cast(input.data()),
      elements);

    AssertEquals(elements, thrust::count(input.begin(), input.end(), ValueT(1)));
  }
  else
  {
    cub::DeviceAdjacentDifference::SubtractRight(
      thrust::raw_pointer_cast(tmp_storage.data()),
      temp_storage_size,
      thrust::raw_pointer_cast(input.data()),
      elements);

    AssertEquals(elements, thrust::count(input.begin(), input.end(), ValueT()));

    thrust::sequence(input.begin(), input.end());

    cub::DeviceAdjacentDifference::SubtractRight(
      thrust::raw_pointer_cast(tmp_storage.data()),
      temp_storage_size,
      thrust::raw_pointer_cast(input.data()),
      elements);

    {
      const auto target_value = static_cast<ValueT>(-1);
      const OffsetT actual_count = thrust::count(input.begin(), input.end() - 1, target_value);
      const OffsetT target_count = elements - 1;

      AssertEquals(actual_count, target_count);
    }

    {
      const auto target_value = static_cast<ValueT>(elements - 1);
      const ValueT actual_value = input[elements - 1];

      AssertEquals(target_value, actual_value);
    }

    thrust::sequence(input.begin(), input.end(), ValueT(1));

    CubDebugExit(cub::DeviceAdjacentDifference::SubtractRight(
      thrust::raw_pointer_cast(tmp_storage.data()),
      temp_storage_size,
      thrust::raw_pointer_cast(input.data()),
      elements,
      cub::MakeBinaryFlip(cub::Difference())));

    {
      const auto target_value = ValueT(1);
      const OffsetT actual_count = thrust::count(input.begin(), input.end() - 1, target_value);
      const OffsetT target_count = elements - 1;

      AssertEquals(actual_count, target_count);
    }

    {
      const auto target_value = ValueT(elements);
      const ValueT actual_value = input[elements - 1];

      AssertEquals(target_value, actual_value);
    }
  }
}

template <typename ValueT, typename OffsetT>
void test_type(OffsetT elements, thrust::default_random_engine &rng)
{
  test<true, ValueT>(elements, rng);
  test_iterator<true, ValueT>(elements);
  test_inplace<true, ValueT>(elements);
  test_iterator_with_custom_op<true, ValueT, OffsetT>(elements);

  test<false, ValueT>(elements, rng);
  test_iterator<false, ValueT>(elements);
  test_inplace<false, ValueT>(elements);
  test_iterator_with_custom_op<false, ValueT, OffsetT>(elements);
}

template <typename OffsetT>
void test_types(OffsetT elements, thrust::default_random_engine &rng)
{
  test_type<uint8_t,  OffsetT>(elements, rng);
  test_type<uint16_t, OffsetT>(elements, rng);
  test_type<uint32_t, OffsetT>(elements, rng);
  test_type<uint64_t, OffsetT>(elements, rng);
}

int main(int argc, char** argv)
{
  CommandLineArgs args(argc, argv);

  // Initialize device
  CubDebugExit(args.DeviceInit());

  thrust::default_random_engine rng;

  for (int power_of_two = 2; power_of_two < 20; power_of_two += 2)
  {
    unsigned int elements = 1 << power_of_two;

    test_types<unsigned int>(elements, rng);
    test_types<long long int>(elements, rng);
  }

  return 0;
}
