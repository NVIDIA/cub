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

#include <fstream>

#include <cub/device/device_segmented_sort.cuh>
#include <test_util.h>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/shuffle.h>
#include <thrust/random.h>
#include <thrust/reduce.h>

using namespace cub;


constexpr static int MAX_ITERATIONS = 4;


class SizeGroupDescription
{
public:
  SizeGroupDescription(const int segments,
                       const int segment_size)
      : segments(segments)
      , segment_size(segment_size)
  {}

  int segments {};
  int segment_size {};
};

template <typename KeyT,
          typename OffsetT>
struct SegmentChecker
{
  const KeyT *sorted_keys {};
  const OffsetT *offsets {};

  explicit SegmentChecker(const KeyT *sorted_keys,
                          const OffsetT *offsets)
    : sorted_keys(sorted_keys)
    , offsets(offsets)
  {}

  __device__ bool operator()(unsigned int segment_id)
  {
    const unsigned int segment_begin = offsets[segment_id];
    const unsigned int segment_end = offsets[segment_id + 1];

    unsigned int counter = 0;
    for (unsigned int i = segment_begin; i < segment_end; i++)
    {
      if (sorted_keys[i] != static_cast<KeyT>(counter++))
      {
        return false;
      }
    }

    return true;
  }
};

template <typename KeyT,
          typename OffsetT>
struct DescendingSegmentChecker
{
  const KeyT *sorted_keys{};
  const OffsetT *offsets{};

  explicit DescendingSegmentChecker(const KeyT *sorted_keys,
                                    const OffsetT *offsets)
      : sorted_keys(sorted_keys)
      , offsets(offsets)
  {}

  __device__ bool operator()(unsigned int segment_id)
  {
    const int segment_begin = static_cast<int>(offsets[segment_id]);
    const int segment_end   = static_cast<int>(offsets[segment_id + 1]);

    unsigned int counter = 0;
    for (int i = segment_end - 1; i >= segment_begin; i--)
    {
      if (sorted_keys[i] != static_cast<KeyT>(counter++))
      {
        return false;
      }
    }

    return true;
  }
};

template <typename KeyT,
          typename OffsetT>
struct ReversedIOTA
{
  KeyT *data {};
  const OffsetT *offsets {};

  ReversedIOTA(KeyT *data,
               const OffsetT *offsets)
    : data(data)
    , offsets(offsets)
  {}

  __device__ void operator()(unsigned int segment_id) const
  {
    const unsigned int segment_begin = offsets[segment_id];
    const unsigned int segment_end = offsets[segment_id + 1];
    const unsigned int segment_size = segment_end - segment_begin;

    unsigned int count = 0;
    for (unsigned int i = segment_begin; i < segment_end; i++)
    {
      data[i] = static_cast<KeyT>(segment_size - 1 - count++);
    }
  }
};


template <typename KeyT,
          typename OffsetT>
struct IOTA
{
  KeyT *data{};
  const OffsetT *offsets{};

  IOTA(KeyT *data, const OffsetT *offsets)
      : data(data)
      , offsets(offsets)
  {}

  __device__ void operator()(unsigned int segment_id) const
  {
    const unsigned int segment_begin = offsets[segment_id];
    const unsigned int segment_end   = offsets[segment_id + 1];

    unsigned int count = 0;
    for (unsigned int i = segment_begin; i < segment_end; i++)
    {
      data[i] = static_cast<KeyT>(count++);
    }
  }
};


template <typename KeyT,
          typename OffsetT,
          typename ValueT = cub::NullType>
class Input
{
  thrust::default_random_engine random_engine;
  thrust::device_vector<OffsetT> d_segment_sizes;
  thrust::device_vector<OffsetT> d_offsets;
  thrust::host_vector<OffsetT> h_offsets;

  using MaskedValueT = typename std::conditional<
    std::is_same<ValueT, cub::NullType>::value,
    KeyT,
    ValueT>::type;

  bool reverse {};
  OffsetT num_items {};
  thrust::device_vector<KeyT> d_keys;
  thrust::device_vector<MaskedValueT> d_values;

public:
  Input(bool reverse, const thrust::host_vector<OffsetT> &h_segment_sizes)
      : d_segment_sizes(h_segment_sizes)
      , d_offsets(d_segment_sizes.size() + 1)
      , h_offsets(d_segment_sizes.size() + 1)
      , reverse(reverse)
      , num_items(static_cast<OffsetT>(
          thrust::reduce(d_segment_sizes.begin(), d_segment_sizes.end())))
      , d_keys(num_items)
      , d_values(num_items)
  {
    update();
  }

  Input(thrust::host_vector<OffsetT> &h_offsets)
    : d_offsets(h_offsets)
    , h_offsets(h_offsets)
    , reverse(false)
    , num_items(h_offsets.back())
    , d_keys(num_items)
    , d_values(num_items)
  {
  }

  void shuffle()
  {
    thrust::shuffle(d_segment_sizes.begin(), d_segment_sizes.end(), random_engine);

    update();
  }

  OffsetT get_num_items() const
  {
    return num_items;
  }

  unsigned int get_num_segments() const
  {
    return static_cast<unsigned int>(d_segment_sizes.size());
  }

  const KeyT *get_d_keys() const
  {
    return thrust::raw_pointer_cast(d_keys.data());
  }

  thrust::device_vector<KeyT> &get_d_keys_vec()
  {
    return d_keys;
  }

  thrust::device_vector<MaskedValueT> &get_d_values_vec()
  {
    return d_values;
  }

  KeyT *get_d_keys()
  {
    return thrust::raw_pointer_cast(d_keys.data());
  }

  const thrust::host_vector<OffsetT>& get_h_offsets()
  {
    return h_offsets;
  }

  MaskedValueT *get_d_values()
  {
    return thrust::raw_pointer_cast(d_values.data());
  }

  const OffsetT *get_d_offsets() const
  {
    return thrust::raw_pointer_cast(d_offsets.data());
  }

  template <typename T>
  bool check_output_implementation(const T *keys_output)
  {
    thrust::device_vector<bool> is_segment_sorted(get_num_segments(), true);

    if (reverse)
    {
      thrust::transform(
        thrust::counting_iterator<unsigned int>(0),
                        thrust::counting_iterator<unsigned int>(
                          static_cast<unsigned int>(get_num_segments())),
                        is_segment_sorted.begin(),
                        DescendingSegmentChecker<T, OffsetT>(keys_output,
                                                             get_d_offsets()));
    }
    else
    {
      thrust::transform(
        thrust::counting_iterator<unsigned int>(0),
                        thrust::counting_iterator<unsigned int>(
                          static_cast<unsigned int>(get_num_segments())),
                        is_segment_sorted.begin(),
                        SegmentChecker<T, OffsetT>(keys_output,
                                                   get_d_offsets()));
    }

    return thrust::reduce(is_segment_sorted.begin(),
                          is_segment_sorted.end(),
                          true,
                          thrust::logical_and<bool>());
  }

  bool check_output(const KeyT *keys_output,
                    const MaskedValueT *values_output = nullptr)
  {
    const bool keys_ok = check_output_implementation(keys_output);
    const bool values_ok = (std::is_same<ValueT, cub::NullType>::value ||
                            values_output == nullptr)
                             ? true
                             : check_output_implementation(values_output);

    return keys_ok && values_ok;
  }

private:
  void update()
  {
    fill_offsets();
    gen_keys();
  }

  void fill_offsets()
  {
    thrust::copy(d_segment_sizes.begin(), d_segment_sizes.end(), d_offsets.begin());
    thrust::exclusive_scan(d_offsets.begin(), d_offsets.end(), d_offsets.begin(), 0u);
    thrust::copy(d_offsets.begin(), d_offsets.end(), h_offsets.begin());
  }

  void gen_keys()
  {
    const unsigned int total_segments =
      static_cast<unsigned int>(get_num_segments());

    if (reverse)
    {
      thrust::for_each(thrust::counting_iterator<unsigned int>(0),
                       thrust::counting_iterator<unsigned int>(total_segments),
                       IOTA<KeyT, OffsetT>(get_d_keys(),
                                           get_d_offsets()));
    }
    else
    {
      thrust::for_each(thrust::counting_iterator<unsigned int>(0),
                       thrust::counting_iterator<unsigned int>(total_segments),
                       ReversedIOTA<KeyT, OffsetT>(get_d_keys(),
                                                   get_d_offsets()));
    }

    thrust::copy(d_keys.begin(), d_keys.end(), d_values.begin());
  }
};

template <typename KeyT, typename OffsetT>
class InputDescription
{
  thrust::host_vector<OffsetT> segment_sizes;

public:
  InputDescription& add(const SizeGroupDescription &group)
  {
    if (static_cast<std::size_t>(group.segment_size) <
        static_cast<std::size_t>(std::numeric_limits<KeyT>::max()))
    {
      for (int i = 0; i < group.segments; i++)
      {
        segment_sizes.push_back(group.segment_size);
      }
    }

    return *this;
  }

  template <typename ValueT = cub::NullType>
  Input<KeyT, OffsetT, ValueT> gen(bool reverse)
  {
    return Input<KeyT, OffsetT, ValueT>(reverse, segment_sizes);
  }
};


template <typename KeyT,
          typename ValueT,
          typename OffsetT>
void Sort(bool pairs,
          bool descending,
          bool double_buffer,
          bool stable_sort,

          void *tmp_storage,
          std::size_t &temp_storage_bytes,

          KeyT *input_keys,
          KeyT *output_keys,

          ValueT *input_values,
          ValueT *output_values,

          OffsetT num_items,
          unsigned int num_segments,
          const OffsetT *d_offsets,

          int *keys_selector = nullptr,
          int *values_selector = nullptr)
{
  if (stable_sort)
  {
    if (pairs)
    {
      if (descending)
      {
        if (double_buffer)
        {
          cub::DoubleBuffer<KeyT> keys_buffer(input_keys, output_keys);
          keys_buffer.selector = *keys_selector;

          cub::DoubleBuffer<ValueT> values_buffer(input_values, output_values);
          values_buffer.selector = *values_selector;

          CubDebugExit(cub::DeviceSegmentedSort::StableSortPairsDescending(
            tmp_storage,
            temp_storage_bytes,
            keys_buffer,
            values_buffer,
            num_items,
            num_segments,
            d_offsets,
            d_offsets + 1,
            0,
            true));

          *keys_selector   = keys_buffer.selector;
          *values_selector = values_buffer.selector;
        }
        else
        {
          CubDebugExit(cub::DeviceSegmentedSort::StableSortPairsDescending(
            tmp_storage,
            temp_storage_bytes,
            input_keys,
            output_keys,
            input_values,
            output_values,
            num_items,
            num_segments,
            d_offsets,
            d_offsets + 1,
            0,
            true));
        }
      }
      else
      {
        if (double_buffer)
        {
          cub::DoubleBuffer<KeyT> keys_buffer(input_keys, output_keys);
          keys_buffer.selector = *keys_selector;

          cub::DoubleBuffer<ValueT> values_buffer(input_values, output_values);
          values_buffer.selector = *values_selector;

          CubDebugExit(
            cub::DeviceSegmentedSort::StableSortPairs(tmp_storage,
                                                      temp_storage_bytes,
                                                      keys_buffer,
                                                      values_buffer,
                                                      num_items,
                                                      num_segments,
                                                      d_offsets,
                                                      d_offsets + 1,
                                                      0,
                                                      true));

          *keys_selector   = keys_buffer.selector;
          *values_selector = values_buffer.selector;
        }
        else
        {
          CubDebugExit(
            cub::DeviceSegmentedSort::StableSortPairs(tmp_storage,
                                                      temp_storage_bytes,
                                                      input_keys,
                                                      output_keys,
                                                      input_values,
                                                      output_values,
                                                      num_items,
                                                      num_segments,
                                                      d_offsets,
                                                      d_offsets + 1,
                                                      0,
                                                      true));
        }
      }
    }
    else
    {
      if (descending)
      {
        if (double_buffer)
        {
          cub::DoubleBuffer<KeyT> keys_buffer(input_keys, output_keys);
          keys_buffer.selector = *keys_selector;

          CubDebugExit(cub::DeviceSegmentedSort::StableSortKeysDescending(
            tmp_storage,
            temp_storage_bytes,
            keys_buffer,
            num_items,
            num_segments,
            d_offsets,
            d_offsets + 1,
            0,
            true));

          *keys_selector = keys_buffer.selector;
        }
        else
        {
          CubDebugExit(cub::DeviceSegmentedSort::StableSortKeysDescending(
            tmp_storage,
            temp_storage_bytes,
            input_keys,
            output_keys,
            num_items,
            num_segments,
            d_offsets,
            d_offsets + 1,
            0,
            true));
        }
      }
      else
      {
        if (double_buffer)
        {
          cub::DoubleBuffer<KeyT> keys_buffer(input_keys, output_keys);
          keys_buffer.selector = *keys_selector;

          CubDebugExit(
            cub::DeviceSegmentedSort::StableSortKeys(tmp_storage,
                                                     temp_storage_bytes,
                                                     keys_buffer,
                                                     num_items,
                                                     num_segments,
                                                     d_offsets,
                                                     d_offsets + 1,
                                                     0,
                                                     true));

          *keys_selector = keys_buffer.selector;
        }
        else
        {
          CubDebugExit(
            cub::DeviceSegmentedSort::StableSortKeys(tmp_storage,
                                                     temp_storage_bytes,
                                                     input_keys,
                                                     output_keys,
                                                     num_items,
                                                     num_segments,
                                                     d_offsets,
                                                     d_offsets + 1,
                                                     0,
                                                     true));
        }
      }
    }
  }
  else
  {
    if (pairs)
    {
      if (descending)
      {
        if (double_buffer)
        {
          cub::DoubleBuffer<KeyT> keys_buffer(input_keys, output_keys);
          keys_buffer.selector = *keys_selector;

          cub::DoubleBuffer<ValueT> values_buffer(input_values, output_values);
          values_buffer.selector = *values_selector;

          CubDebugExit(
            cub::DeviceSegmentedSort::SortPairsDescending(tmp_storage,
                                                          temp_storage_bytes,
                                                          keys_buffer,
                                                          values_buffer,
                                                          num_items,
                                                          num_segments,
                                                          d_offsets,
                                                          d_offsets + 1,
                                                          0,
                                                          true));

          *keys_selector   = keys_buffer.selector;
          *values_selector = values_buffer.selector;
        }
        else
        {
          CubDebugExit(
            cub::DeviceSegmentedSort::SortPairsDescending(tmp_storage,
                                                          temp_storage_bytes,
                                                          input_keys,
                                                          output_keys,
                                                          input_values,
                                                          output_values,
                                                          num_items,
                                                          num_segments,
                                                          d_offsets,
                                                          d_offsets + 1,
                                                          0,
                                                          true));
        }
      }
      else
      {
        if (double_buffer)
        {
          cub::DoubleBuffer<KeyT> keys_buffer(input_keys, output_keys);
          keys_buffer.selector = *keys_selector;

          cub::DoubleBuffer<ValueT> values_buffer(input_values, output_values);
          values_buffer.selector = *values_selector;

          CubDebugExit(cub::DeviceSegmentedSort::SortPairs(tmp_storage,
                                                           temp_storage_bytes,
                                                           keys_buffer,
                                                           values_buffer,
                                                           num_items,
                                                           num_segments,
                                                           d_offsets,
                                                           d_offsets + 1,
                                                           0,
                                                           true));

          *keys_selector   = keys_buffer.selector;
          *values_selector = values_buffer.selector;
        }
        else
        {
          CubDebugExit(cub::DeviceSegmentedSort::SortPairs(tmp_storage,
                                                           temp_storage_bytes,
                                                           input_keys,
                                                           output_keys,
                                                           input_values,
                                                           output_values,
                                                           num_items,
                                                           num_segments,
                                                           d_offsets,
                                                           d_offsets + 1,
                                                           0,
                                                           true));
        }
      }
    }
    else
    {
      if (descending)
      {
        if (double_buffer)
        {
          cub::DoubleBuffer<KeyT> keys_buffer(input_keys, output_keys);
          keys_buffer.selector = *keys_selector;

          CubDebugExit(
            cub::DeviceSegmentedSort::SortKeysDescending(tmp_storage,
                                                         temp_storage_bytes,
                                                         keys_buffer,
                                                         num_items,
                                                         num_segments,
                                                         d_offsets,
                                                         d_offsets + 1,
                                                         0,
                                                         true));

          *keys_selector = keys_buffer.selector;
        }
        else
        {
          CubDebugExit(
            cub::DeviceSegmentedSort::SortKeysDescending(tmp_storage,
                                                         temp_storage_bytes,
                                                         input_keys,
                                                         output_keys,
                                                         num_items,
                                                         num_segments,
                                                         d_offsets,
                                                         d_offsets + 1,
                                                         0,
                                                         true));
        }
      }
      else
      {
        if (double_buffer)
        {
          cub::DoubleBuffer<KeyT> keys_buffer(input_keys, output_keys);
          keys_buffer.selector = *keys_selector;

          CubDebugExit(cub::DeviceSegmentedSort::SortKeys(tmp_storage,
                                                          temp_storage_bytes,
                                                          keys_buffer,
                                                          num_items,
                                                          num_segments,
                                                          d_offsets,
                                                          d_offsets + 1,
                                                          0,
                                                          true));

          *keys_selector = keys_buffer.selector;
        }
        else
        {
          CubDebugExit(cub::DeviceSegmentedSort::SortKeys(tmp_storage,
                                                          temp_storage_bytes,
                                                          input_keys,
                                                          output_keys,
                                                          num_items,
                                                          num_segments,
                                                          d_offsets,
                                                          d_offsets + 1,
                                                          0,
                                                          true));
        }
      }
    }
  }
}

template <typename KeyT,
          typename ValueT,
          typename OffsetT>
std::size_t Sort(bool pairs,
                 bool descending,
                 bool double_buffer,
                 bool stable_sort,

                 KeyT *input_keys,
                 KeyT *output_keys,

                 ValueT *input_values,
                 ValueT *output_values,

                 OffsetT num_items,
                 unsigned int num_segments,
                 const OffsetT *d_offsets,

                 int *keys_selector   = nullptr,
                 int *values_selector = nullptr)
{
  std::size_t temp_storage_bytes = 42ul;

  Sort<KeyT, ValueT, OffsetT>(pairs,
                              descending,
                              double_buffer,
                              stable_sort,
                              nullptr,
                              temp_storage_bytes,
                              input_keys,
                              output_keys,
                              input_values,
                              output_values,
                              num_items,
                              num_segments,
                              d_offsets,
                              keys_selector,
                              values_selector);

  thrust::device_vector<std::uint8_t> temp_storage(temp_storage_bytes);
  std::uint8_t *d_temp_storage = thrust::raw_pointer_cast(temp_storage.data());

  Sort<KeyT, ValueT, OffsetT>(pairs,
                              descending,
                              double_buffer,
                              stable_sort,
                              d_temp_storage,
                              temp_storage_bytes,
                              input_keys,
                              output_keys,
                              input_values,
                              output_values,
                              num_items,
                              num_segments,
                              d_offsets,
                              keys_selector,
                              values_selector);

  return temp_storage_bytes;
}


constexpr bool keys = false;
constexpr bool pairs = true;

constexpr bool ascending = false;
constexpr bool descending = true;

constexpr bool pointers = false;
constexpr bool double_buffer = true;

constexpr bool unstable = false;
constexpr bool stable = true;


void TestZeroSegments()
{
  // Type doesn't affect the escape logic, so it should be fine
  // to test only one set of types here.

  using KeyT = std::uint8_t;
  using ValueT = std::uint64_t;
  using OffsetT = std::uint32_t;

  for (bool stable_sort: { unstable, stable })
  {
    for (bool sort_pairs: { keys, pairs })
    {
      for (bool sort_descending: { ascending, descending })
      {
        for (bool sort_buffer: { pointers, double_buffer })
        {
          cub::DoubleBuffer<KeyT> keys_buffer(nullptr, nullptr);
          cub::DoubleBuffer<ValueT> values_buffer(nullptr, nullptr);
          values_buffer.selector = 1;

          const std::size_t temp_storage_bytes =
            Sort<KeyT, ValueT, OffsetT>(sort_pairs,
                                        sort_descending,
                                        sort_buffer,
                                        stable_sort,
                                        nullptr,
                                        nullptr,
                                        nullptr,
                                        nullptr,
                                        OffsetT{},
                                        OffsetT{},
                                        nullptr,
                                        &keys_buffer.selector,
                                        &values_buffer.selector);

          AssertEquals(keys_buffer.selector, 0);
          AssertEquals(values_buffer.selector, 1);
          AssertEquals(temp_storage_bytes, 0ul);
        }
      }
    }
  }
}


void TestEmptySegments(unsigned int segments)
{
  // Type doesn't affect the escape logic, so it should be fine
  // to test only one set of types here.

  using KeyT = std::uint8_t;
  using ValueT = std::uint64_t;
  using OffsetT = std::uint32_t;

  thrust::device_vector<OffsetT> offsets(segments + 1, OffsetT{});
  const OffsetT *d_offsets = thrust::raw_pointer_cast(offsets.data());

  for (bool sort_stable: { unstable, stable })
  {
    for (bool sort_pairs: { keys, pairs })
    {
      for (bool sort_descending: { ascending, descending })
      {
        for (bool sort_buffer: { pointers, double_buffer })
        {
          cub::DoubleBuffer<KeyT> keys_buffer(nullptr, nullptr);
          cub::DoubleBuffer<ValueT> values_buffer(nullptr, nullptr);
          values_buffer.selector = 1;

          const std::size_t temp_storage_bytes =
            Sort<KeyT, ValueT, OffsetT>(sort_pairs,
                                        sort_descending,
                                        sort_buffer,
                                        sort_stable,
                                        nullptr,
                                        nullptr,
                                        nullptr,
                                        nullptr,
                                        OffsetT{},
                                        segments,
                                        d_offsets,
                                        &keys_buffer.selector,
                                        &values_buffer.selector);

          AssertEquals(keys_buffer.selector, 0);
          AssertEquals(values_buffer.selector, 1);
          AssertEquals(temp_storage_bytes, 0ul);
        }
      }
    }
  }
}


template <typename KeyT,
          typename ValueT,
          typename OffsetT>
void TestSameSizeSegments(OffsetT segment_size,
                          unsigned int segments,
                          bool skip_values = false)
{
  const OffsetT num_items = segment_size * segments;

  thrust::device_vector<OffsetT> offsets(segments + 1);
  thrust::sequence(offsets.begin(),
                   offsets.end(),
                   OffsetT{},
                   OffsetT{segment_size});

  const OffsetT *d_offsets = thrust::raw_pointer_cast(offsets.data());

  const KeyT target_key = KeyT{42};
  const ValueT target_value = ValueT{42};

  thrust::device_vector<KeyT> keys_input(num_items);
  thrust::device_vector<KeyT> keys_output(num_items);

  KeyT *d_keys_input  = thrust::raw_pointer_cast(keys_input.data());
  KeyT *d_keys_output = thrust::raw_pointer_cast(keys_output.data());

  thrust::device_vector<ValueT> values_input(num_items);
  thrust::device_vector<ValueT> values_output(num_items);

  ValueT *d_values_input  = thrust::raw_pointer_cast(values_input.data());
  ValueT *d_values_output = thrust::raw_pointer_cast(values_output.data());

  for (bool stable_sort: { unstable, stable })
  {
    for (bool sort_pairs: { keys, pairs })
    {
      if (sort_pairs)
      {
        if (skip_values)
        {
          continue;
        }
      }

      for (bool sort_descending: { ascending, descending })
      {
        for (bool sort_buffers: { pointers, double_buffer })
        {
          cub::DoubleBuffer<KeyT> keys_buffer(nullptr, nullptr);
          cub::DoubleBuffer<ValueT> values_buffer(nullptr, nullptr);
          values_buffer.selector = 1;

          thrust::fill(keys_input.begin(), keys_input.end(), target_key);
          thrust::fill(keys_output.begin(), keys_output.end(), KeyT{});

          if (sort_pairs)
          {
            if (sort_buffers)
            {
              thrust::fill(values_input.begin(), values_input.end(), ValueT{});
              thrust::fill(values_output.begin(), values_output.end(), target_value);
            }
            else
            {
              thrust::fill(values_input.begin(), values_input.end(), target_value);
              thrust::fill(values_output.begin(), values_output.end(), ValueT{});
            }
          }

          const std::size_t temp_storage_bytes =
            Sort<KeyT, ValueT, OffsetT>(sort_pairs,
                                        sort_descending,
                                        sort_buffers,
                                        stable_sort,
                                        d_keys_input,
                                        d_keys_output,
                                        d_values_input,
                                        d_values_output,
                                        num_items,
                                        segments,
                                        d_offsets,
                                        &keys_buffer.selector,
                                        &values_buffer.selector);

          // If temporary storage size is defined by extra keys storage
          if (sort_buffers)
          {
            if (2 * segments * sizeof(unsigned int) < num_items * sizeof(KeyT))
            {
              std::size_t extra_temp_storage_bytes{};

              Sort(sort_pairs,
                   sort_descending,
                   pointers,
                   stable_sort,
                   nullptr,
                   extra_temp_storage_bytes,
                   d_keys_input,
                   d_keys_output,
                   d_values_input,
                   d_values_output,
                   num_items,
                   segments,
                   d_offsets,
                   &keys_buffer.selector,
                   &values_buffer.selector);

              AssertTrue(extra_temp_storage_bytes > temp_storage_bytes);
            }
          }

          {
            const std::size_t items_selected =
              keys_buffer.selector || !sort_buffers
              ? thrust::count(keys_output.begin(),
                              keys_output.end(),
                              target_key)
              : thrust::count(keys_input.begin(), keys_input.end(), target_key);
            AssertEquals(items_selected, num_items);
          }

          if (sort_pairs)
          {
            const std::size_t items_selected = [&]() -> std::size_t {
              if (sort_buffers)
              {
                return values_buffer.selector
                       ? thrust::count(values_output.begin(),
                                       values_output.end(),
                                       target_value)
                       : thrust::count(values_input.begin(),
                                       values_input.end(),
                                       target_value);
              }

              return thrust::count(values_output.begin(),
                                   values_output.end(),
                                   target_value);
            } ();

            AssertEquals(items_selected, num_items);
          }
        }
      }
    }
  }
}


template <typename KeyT,
          typename ValueT,
          typename OffsetT>
void InputTest(bool sort_descending,
               Input<KeyT, OffsetT, ValueT> &input)
{
  thrust::device_vector<KeyT> keys_output(input.get_num_items());
  KeyT *d_keys_output = thrust::raw_pointer_cast(keys_output.data());

  thrust::device_vector<ValueT> values_output(input.get_num_items());
  ValueT *d_values_output = thrust::raw_pointer_cast(values_output.data());

  for (bool stable_sort: { unstable, stable })
  {
    for (bool sort_pairs : {keys, pairs})
    {
      for (bool sort_buffers : {pointers, double_buffer})
      {
        for (int iteration = 0; iteration < MAX_ITERATIONS; iteration++)
        {
          thrust::fill(keys_output.begin(), keys_output.end(), KeyT{});
          thrust::fill(values_output.begin(), values_output.end(), ValueT{});

          cub::DoubleBuffer<KeyT> keys_buffer(input.get_d_keys(),
                                              d_keys_output);
          cub::DoubleBuffer<ValueT> values_buffer(input.get_d_values(),
                                                  d_values_output);

          Sort<KeyT, ValueT, OffsetT>(sort_pairs,
                                      sort_descending,
                                      sort_buffers,
                                      stable_sort,
                                      input.get_d_keys(),
                                      d_keys_output,
                                      input.get_d_values(),
                                      d_values_output,
                                      input.get_num_items(),
                                      input.get_num_segments(),
                                      input.get_d_offsets(),
                                      &keys_buffer.selector,
                                      &values_buffer.selector);

          if (sort_buffers)
          {
            if (sort_pairs)
            {
              AssertTrue(input.check_output(keys_buffer.Current(),
                                            values_buffer.Current()));
            }
            else
            {
              AssertTrue(input.check_output(keys_buffer.Current()));
            }
          }
          else
          {
            if (sort_pairs)
            {
              AssertTrue(input.check_output(d_keys_output, d_values_output));
            }
            else
            {
              AssertTrue(input.check_output(d_keys_output));
            }
          }

          input.shuffle();
        }
      }
    }
  }
}


template <typename T,
          typename OffsetT>
bool compare_two_outputs(const thrust::host_vector<OffsetT> &offsets,
                         const thrust::host_vector<T> &lhs,
                         const thrust::host_vector<T> &rhs)
{
  const auto num_segments = static_cast<unsigned int>(offsets.size() - 1);

  for (std::size_t segment_id = 0; segment_id < num_segments; segment_id++)
  {
    auto lhs_begin = lhs.cbegin() + offsets[segment_id];
    auto lhs_end = lhs.cbegin() + offsets[segment_id + 1];
    auto rhs_begin = rhs.cbegin() + offsets[segment_id];

    auto err = thrust::mismatch(lhs_begin, lhs_end, rhs_begin);

    if (err.first != lhs_end)
    {
      const auto idx = thrust::distance(lhs_begin, err.first);
      const auto segment_size = std::distance(lhs_begin, lhs_end);

      std::cerr << "Mismatch in segment " << segment_id
                << " at position " << idx << " / " << segment_size
                << ": "
                << static_cast<std::uint64_t>(lhs_begin[idx]) << " vs "
                << static_cast<std::uint64_t>(rhs_begin[idx]) << " ("
                << typeid(lhs_begin[idx]).name() << ")" << std::endl;

      return false;
    }
  }

  return true;
}


template <typename KeyT,
          typename ValueT>
void RandomizeInput(thrust::host_vector<KeyT> &h_keys,
                    thrust::host_vector<ValueT> &h_values)
{
  for (std::size_t i = 0; i < h_keys.size(); i++)
  {
    h_keys[i] = RandomValue(std::numeric_limits<KeyT>::max());
    h_values[i] = RandomValue(std::numeric_limits<ValueT>::max());
  }
}


template <typename KeyT,
          typename ValueT,
          typename OffsetT>
void HostReferenceSort(bool sort_pairs,
                       bool sort_descending,
                       unsigned int num_segments,
                       const thrust::host_vector<OffsetT> &h_offsets,
                       thrust::host_vector<KeyT> &h_keys,
                       thrust::host_vector<ValueT> &h_values)
{
  for (unsigned int segment_i = 0;
       segment_i < num_segments;
       segment_i++)
  {
    const OffsetT segment_begin = h_offsets[segment_i];
    const OffsetT segment_end   = h_offsets[segment_i + 1];

    if (sort_pairs)
    {
      if (sort_descending)
      {
        thrust::stable_sort_by_key(h_keys.begin() + segment_begin,
                                   h_keys.begin() + segment_end,
                                   h_values.begin() + segment_begin,
                                   thrust::greater<KeyT>{});
      }
      else
      {
        thrust::stable_sort_by_key(h_keys.begin() + segment_begin,
                                   h_keys.begin() + segment_end,
                                   h_values.begin() + segment_begin);
      }
    }
    else
    {
      if (sort_descending)
      {
        thrust::stable_sort(h_keys.begin() + segment_begin,
                            h_keys.begin() + segment_end,
                            thrust::greater<KeyT>{});
      }
      else
      {
        thrust::stable_sort(h_keys.begin() + segment_begin,
                            h_keys.begin() + segment_end);
      }
    }
  }
}


#if STORE_ON_FAILURE
template <typename KeyT,
          typename ValueT,
          typename OffsetT>
void DumpInput(bool sort_pairs,
               bool sort_descending,
               bool sort_buffers,
               Input<KeyT, OffsetT, ValueT> &input,
               thrust::host_vector<KeyT> &h_keys,
               thrust::host_vector<ValueT> &h_values)
{
  const thrust::host_vector<OffsetT> &h_offsets = input.get_h_offsets();

  std::cout << "sort pairs: " << sort_pairs << "\n";
  std::cout << "sort descending: " << sort_descending << "\n";
  std::cout << "sort buffers: " << sort_buffers << "\n";
  std::cout << "num_items: " << input.get_num_items() << "\n";
  std::cout << "num_segments: " << input.get_num_segments() << "\n";
  std::cout << "key type: " << typeid(h_keys[0]).name() << "\n";
  std::cout << "value type: " << typeid(h_values[0]).name() << "\n";
  std::cout << "offset type: " << typeid(h_offsets[0]).name() << "\n";

  std::ofstream offsets_dump("offsets", std::ios::binary);
  offsets_dump.write(reinterpret_cast<const char *>(
                       thrust::raw_pointer_cast(h_offsets.data())),
                     sizeof(OffsetT) * h_offsets.size());

  std::ofstream keys_dump("keys", std::ios::binary);
  keys_dump.write(reinterpret_cast<const char *>(
                    thrust::raw_pointer_cast(h_keys.data())),
                  sizeof(KeyT) * h_keys.size());

  std::ofstream values_dump("values", std::ios::binary);
  values_dump.write(reinterpret_cast<const char *>(
                      thrust::raw_pointer_cast(h_values.data())),
                    sizeof(ValueT) * h_values.size());
}
#endif


template <typename KeyT,
          typename ValueT,
          typename OffsetT>
void InputTestRandom(Input<KeyT, OffsetT, ValueT> &input)
{
  thrust::host_vector<KeyT> h_keys_output(input.get_num_items());
  thrust::device_vector<KeyT> keys_output(input.get_num_items());

  thrust::host_vector<ValueT> h_values_output(input.get_num_items());
  thrust::device_vector<ValueT> values_output(input.get_num_items());

  KeyT *d_keys_output = thrust::raw_pointer_cast(keys_output.data());
  ValueT *d_values_output = thrust::raw_pointer_cast(values_output.data());

  thrust::host_vector<KeyT> h_keys(input.get_num_items());
  thrust::host_vector<ValueT> h_values(input.get_num_items());

  const thrust::host_vector<OffsetT> &h_offsets = input.get_h_offsets();

  for (bool stable_sort: { unstable, stable })
  {
    for (bool sort_pairs: { keys, pairs })
    {
      for (bool sort_descending: { ascending, descending })
      {
        for (bool sort_buffers: { pointers, double_buffer })
        {
          for (int iteration = 0; iteration < MAX_ITERATIONS / 10; iteration++)
          {
            RandomizeInput(h_keys, h_values);

#if STORE_ON_FAILURE
            auto h_keys_backup = h_keys;
            auto h_values_backup = h_values;
#endif

            input.get_d_keys_vec()   = h_keys;
            input.get_d_values_vec() = h_values;

            cub::DoubleBuffer<KeyT> keys_buffer(input.get_d_keys(), d_keys_output);
            cub::DoubleBuffer<ValueT> values_buffer(input.get_d_values(), d_values_output);

            Sort<KeyT, ValueT, OffsetT>(
              sort_pairs,
              sort_descending,
              sort_buffers,
              stable_sort,
              input.get_d_keys(),
              d_keys_output,
              input.get_d_values(),
              d_values_output,
              input.get_num_items(),
              input.get_num_segments(),
              input.get_d_offsets(),
              &keys_buffer.selector,
              &values_buffer.selector);

            HostReferenceSort(sort_pairs,
                              sort_descending,
                              input.get_num_segments(),
                              h_offsets,
                              h_keys,
                              h_values);

            if (sort_buffers)
            {
              if (keys_buffer.selector)
              {
                h_keys_output = keys_output;
              }
              else
              {
                h_keys_output = input.get_d_keys_vec();
              }

              if (values_buffer.selector)
              {
                h_values_output = values_output;
              }
              else
              {
                h_values_output = input.get_d_values_vec();
              }
            }
            else
            {
              h_keys_output = keys_output;
              h_values_output = values_output;
            }

            const bool keys_ok =
              compare_two_outputs(h_offsets, h_keys, h_keys_output);

            const bool values_ok =
              sort_pairs
              ? compare_two_outputs(h_offsets, h_values, h_values_output)
              : true;

#if STORE_ON_FAILURE
            if (!keys_ok || !values_ok)
            {
              DumpInput<KeyT, ValueT, OffsetT>(sort_pairs,
                                               sort_descending,
                                               sort_buffers,
                                               input,
                                               h_keys_backup,
                                               h_values_backup);
            }
#endif

AssertTrue(keys_ok);
            AssertTrue(values_ok);

            input.shuffle();
          }
        }
      }
    }
  }
}


template <typename KeyT,
          typename ValueT,
          typename OffsetT>
void EdgePatternsTest()
{
  for (bool sort_descending: { ascending, descending })
  {
    Input<KeyT, OffsetT, ValueT> edge_cases =
      InputDescription<KeyT, OffsetT>()
        .add({420, 0})
        .add({420, 1})
        .add({420, 2})
        .add({420, 8})
        .add({420, 9})
        .add({420, 10})
        .add({420, 17})
        .add({42, 18})
        .add({42, 19})
        .add({42, 26})
        .add({42, 27})
        .add({42, 28})
        .add({42, 35})
        .add({42, 36})
        .add({42, 37})
        .add({42, 286})
        .add({42, 287})
        .add({42, 288})
        .add({42, 5887})
        .add({42, 5888})
        .add({42, 5889})
        .add({2, 23552})
        .template gen<ValueT>(sort_descending);

    InputTest<KeyT, ValueT, OffsetT>(sort_descending, edge_cases);
  }
}

template <typename KeyT,
          typename ValueT,
          typename OffsetT>
Input<KeyT, OffsetT, ValueT> GenRandomInput(OffsetT max_items,
                                            OffsetT min_segments,
                                            OffsetT max_segments,
                                            bool descending)
{
  std::size_t items_generated {};
  const std::size_t segments_num = RandomValue(max_segments) + min_segments;

  thrust::host_vector<OffsetT> segment_sizes;
  segment_sizes.reserve(segments_num);

  const OffsetT max_segment_size = 6000;

  for (std::size_t segment_id = 0; segment_id < segments_num; segment_id++)
  {
    const OffsetT segment_size_raw = RandomValue(max_segment_size);
    const OffsetT segment_size = segment_size_raw > OffsetT{0} ? segment_size_raw
                                                               : OffsetT{0};

    if (segment_size + items_generated > max_items)
    {
      break;
    }

    items_generated += segment_size;
    segment_sizes.push_back(segment_size);
  }

  return Input<KeyT, OffsetT, ValueT>{descending, segment_sizes};
}

template <typename KeyT,
          typename ValueT,
          typename OffsetT>
void RandomTest(OffsetT min_segments,
                OffsetT max_segments)
{
  const OffsetT max_items = 10000000;

  for (int iteration = 0; iteration < 10 * MAX_ITERATIONS; iteration++)
  {
    Input<KeyT, OffsetT, ValueT> edge_cases =
      GenRandomInput<KeyT, ValueT, OffsetT>(max_items,
                                            min_segments,
                                            max_segments,
                                            descending);

    InputTestRandom(edge_cases);
  }
}

template <typename KeyT,
          typename OffsetT>
void TestKeys()
{
  const bool skip_values = true;

  for (OffsetT segment_size : {1, 1024, 24 * 1024})
  {
    for (int segments : {1, 1024})
    {
      TestSameSizeSegments<KeyT, KeyT, OffsetT>(segment_size,
                                                segments,
                                                skip_values);
    }
  }
}


template <typename KeyT,
          typename ValueT,
          typename OffsetT>
void TestPairs()
{
  for (OffsetT segment_size: { 1, 1024, 24 * 1024 })
  {
    for (int segments: { 1, 1024 })
    {
      TestSameSizeSegments<KeyT, ValueT, OffsetT>(segment_size, segments);
    }
  }

  RandomTest<KeyT, ValueT, OffsetT>(1 << 2, 1 << 8);
  RandomTest<KeyT, ValueT, OffsetT>(1 << 9, 1 << 19);
}


template <typename T,
          typename OffsetT>
void TestKeysAndPairs()
{
  TestKeys<T, OffsetT>();
  TestPairs<T, T, OffsetT>();
}


int main(int argc, char** argv)
{
  CommandLineArgs args(argc, argv);

  // Initialize device
  CubDebugExit(args.DeviceInit());

  TestZeroSegments();
  TestEmptySegments(1 << 2);
  TestEmptySegments(1 << 22);

  TestKeysAndPairs<std::uint8_t,  std::uint32_t>();
  TestKeysAndPairs<std::uint16_t, std::uint32_t>();
  TestKeysAndPairs<std::uint32_t, std::uint32_t>();
  TestKeysAndPairs<std::uint64_t, std::uint32_t>();
  TestKeysAndPairs<std::uint64_t, std::uint64_t>();
  TestPairs<std::uint8_t, std::uint64_t, std::uint32_t>();
  TestPairs<std::int64_t, std::uint64_t, std::uint32_t>();

  return 0;
}
