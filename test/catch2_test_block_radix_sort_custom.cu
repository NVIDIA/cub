/******************************************************************************
 * Copyright (c) 2011-2022, NVIDIA CORPORATION.  All rights reserved.
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

#include "catch2_test_block_radix_sort.cuh"
#include "cub/block/radix_rank_sort_operations.cuh"

#include <algorithm>
#include <type_traits>
#include <utility>

#include <thrust/host_vector.h>
#include <thrust/sequence.h>

struct pair_decomposer_t
{
  template <template <typename> class... Ps>
  __host__ __device__ ::cuda::std::tuple<std::size_t &, std::size_t &>
  operator()(c2h::custom_type_t<Ps...> &key) const
  {
    return {key.key, key.val};
  }
};

struct pair_custom_sort_op_t
{
  template <class BlockRadixSortT, class KeysT>
  __device__ void operator()(BlockRadixSortT &block_radix_sort,
                             KeysT &keys,
                             int /* begin_bit */,
                             int /* end_bit */,
                             cub::Int2Type<0> /* striped */)
  {
    block_radix_sort.Sort(keys, pair_decomposer_t{});
  }

  template <class BlockRadixSortT, class KeysT>
  __device__ void operator()(BlockRadixSortT &block_radix_sort,
                             KeysT &keys,
                             int /* begin_bit */,
                             int /* end_bit */,
                             cub::Int2Type<1> /* striped */)
  {
    block_radix_sort.SortBlockedToStriped(keys, pair_decomposer_t{});
  }
};

constexpr int items_per_thread = 1;
constexpr int threads_in_block = 2;
constexpr int tile_size = items_per_thread * threads_in_block;

constexpr int radix_bits = 8;
constexpr bool memoize = true;
constexpr cub::BlockScanAlgorithm algorithm = cub::BLOCK_SCAN_WARP_SCANS;
constexpr cudaSharedMemConfig shmem_config = cudaSharedMemBankSizeFourByte;

template <class KeyT>
thrust::device_vector<KeyT> reference_sort_keys(const thrust::device_vector<KeyT> &keys_in)
{
  thrust::host_vector<KeyT> host_keys = keys_in;
  std::sort(host_keys.begin(), host_keys.end());
  return host_keys;
}

TEST_CASE("Block radix sort can sort custom pairs", "[radix][sort][block]")
{
  using key_t = c2h::custom_type_t<c2h::equal_comparable_t,                 //
                                   c2h::lexicographical_less_comparable_t,  //
                                   c2h::lexicographical_greater_comparable_t>;

  // const bool is_descending = false;
  const bool striped = GENERATE(false, true);
  const int begin_bit = 0;
  const int end_bit = sizeof(std::size_t) * 8 * 2;

  thrust::device_vector<key_t> in_keys(tile_size);
  thrust::device_vector<key_t> out_keys(tile_size);
  c2h::gen(CUB_SEED(2), in_keys);

  auto reference_keys = reference_sort_keys(in_keys);

  block_radix_sort<items_per_thread, threads_in_block, radix_bits, memoize, algorithm, shmem_config>(
    pair_custom_sort_op_t{},
    thrust::raw_pointer_cast(in_keys.data()),
    thrust::raw_pointer_cast(out_keys.data()),
    begin_bit,
    end_bit,
    striped);

  INFO( "striped = " << striped );
  REQUIRE( reference_keys == out_keys );
}
