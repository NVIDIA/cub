/******************************************************************************
 * Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

#include <cub/device/device_radix_sort.cuh>
#include <thrust/detail/raw_pointer_cast.h>

#include <algorithm>

// Has to go after all cub headers. Otherwise, this test won't catch unused
// variables in cub kernels.
#include "catch2_test_helper.h"
#include "thrust/device_vector.h"
#include "thrust/host_vector.h"

struct key_decomposer_t
{
  template <template <typename> class... Ps>
  __host__ __device__ ::cuda::std::tuple<std::size_t &>
  operator()(c2h::custom_type_t<Ps...> &key) const
  {
    return {key.key};
  }
};

template <class KeyT, class DecomposerT>
thrust::host_vector<KeyT> sort_keys(const thrust::device_vector<KeyT> &keys_in,
                                    DecomposerT decomposer)
{
  const auto num_items = static_cast<int>(keys_in.size());
  thrust::device_vector<KeyT> keys_out(num_items);

  const auto d_in_keys = thrust::raw_pointer_cast(keys_in.data());
  auto d_out_keys      = thrust::raw_pointer_cast(keys_out.data());

  std::uint8_t *d_temp_storage{};
  std::size_t temp_storage_size{};
  cudaError_t status = cub::DeviceRadixSort::SortKeys(d_temp_storage,
                                                      temp_storage_size,
                                                      d_in_keys,
                                                      d_out_keys,
                                                      num_items,
                                                      decomposer);
  REQUIRE(cudaSuccess == status);

  thrust::device_vector<std::uint8_t> temp_storage(temp_storage_size);
  d_temp_storage = thrust::raw_pointer_cast(temp_storage.data());

  status = cub::DeviceRadixSort::SortKeys(d_temp_storage,
                                          temp_storage_size,
                                          d_in_keys,
                                          d_out_keys,
                                          num_items,
                                          decomposer);
  REQUIRE(cudaSuccess == status);
  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());

  return keys_out;
}

template <class KeyT>
thrust::device_vector<KeyT> reference_sort_keys(const thrust::device_vector<KeyT> &keys_in)
{
  thrust::host_vector<KeyT> host_keys = keys_in;
  std::stable_sort(host_keys.begin(), host_keys.end());
  return host_keys;
}

TEST_CASE("Device radix sort works with parts of custom i128_t", "[radix][sort][device]")
{
  using key_t = c2h::custom_type_t<c2h::equal_comparable_t, c2h::less_comparable_t>;

  const int max_items = 1 << 18;
  const int num_items = GENERATE_COPY(take(4, random(max_items / 2, max_items)));

  thrust::device_vector<key_t> in_keys(num_items);
  c2h::gen(CUB_SEED(10), in_keys);

  auto reference_keys = reference_sort_keys(in_keys);
  auto out_keys = sort_keys(in_keys, key_decomposer_t{});

  REQUIRE(reference_keys == out_keys);
}

struct pair_decomposer_t
{
  template <template <typename> class... Ps>
  __host__ __device__ ::cuda::std::tuple<std::size_t &, std::size_t &>
  operator()(c2h::custom_type_t<Ps...> &key) const
  {
    return {key.key, key.val};
  }
};

TEST_CASE("Device radix sort works with custom i128_t", "[radix][sort][device]")
{
  using key_t = c2h::custom_type_t<c2h::equal_comparable_t, c2h::lexicographical_less_comparable_t>;

  const int max_items = 1 << 18;
  const int num_items = GENERATE_COPY(take(4, random(max_items / 2, max_items)));

  thrust::device_vector<key_t> in_keys(num_items);
  c2h::gen(CUB_SEED(10), in_keys);

  auto reference_keys = reference_sort_keys(in_keys);
  auto out_keys = sort_keys(in_keys, pair_decomposer_t{});

  REQUIRE(reference_keys == out_keys);
}
