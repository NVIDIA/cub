/******************************************************************************
 * Copyright (c) 2011-2023, NVIDIA CORPORATION.  All rights reserved.
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

#include <cub/device/device_scan.cuh>

#include <type_traits>

#if !TUNE_BASE
template <typename AccumT>
struct policy_hub_t
{
  struct policy_t : cub::ChainedPolicy<300, policy_t, policy_t>
  {
    using ScanPolicyT = cub::AgentScanPolicy<TUNE_THREADS,
                                             TUNE_ITEMS,
                                             AccumT,
                                             cub::BLOCK_LOAD_WARP_TRANSPOSE,
                                             cub::LOAD_DEFAULT,
                                             cub::BLOCK_STORE_WARP_TRANSPOSE,
                                             cub::BLOCK_SCAN_WARP_SCANS>;
  };

  using MaxPolicy = policy_t;
};

template <typename T, typename OffsetT>
constexpr std::size_t max_temp_storage_size()
{
  using accum_t     = T;
  using input_it_t  = const T *;
  using output_it_t = T *;
  using offset_t    = OffsetT;
  using init_t      = cub::detail::InputValue<T>;
  using policy_t    = typename policy_hub_t<accum_t>::policy_t;
  using real_init_t = typename init_t::value_type;

  using agent_scan_t = cub::AgentScan<typename policy_t::ScanPolicyT,
                                      input_it_t,
                                      output_it_t,
                                      op_t,
                                      real_init_t,
                                      offset_t,
                                      accum_t>;

  return sizeof(typename agent_scan_t::TempStorage);
}

template <typename T, typename OffsetT>
constexpr bool fits_in_default_shared_memory()
{
  return max_temp_storage_size<T, OffsetT>() < 48 * 1024;
}
#else // TUNE_BASE
template <typename T, typename OffsetT>
constexpr bool fits_in_default_shared_memory()
{
  return true;
}
#endif // TUNE_BASE

template <typename T, typename OffsetT>
static void basic(std::integral_constant<bool, true>,
                  nvbench::state &state,
                  nvbench::type_list<T, OffsetT>)
{
  using init_t      = cub::detail::InputValue<T>;
  using accum_t     = cub::detail::accumulator_t<op_t, T, T>;
  using input_it_t  = const T *;
  using output_it_t = T *;
  using offset_t    = OffsetT;

#if !TUNE_BASE
  using policy_t = policy_hub_t<accum_t>;
  using dispatch_t =
    cub::DispatchScan<input_it_t, output_it_t, op_t, init_t, offset_t, accum_t, policy_t>;
#else
  using dispatch_t = cub::DispatchScan<input_it_t, output_it_t, op_t, init_t, offset_t, accum_t>;
#endif

  const auto elements = static_cast<std::size_t>(state.get_int64("Elements{io}"));

  thrust::device_vector<T> input(elements);
  thrust::device_vector<T> output(elements);

  gen(seed_t{}, input);

  T *d_input  = thrust::raw_pointer_cast(input.data());
  T *d_output = thrust::raw_pointer_cast(output.data());

  state.add_element_count(elements);
  state.add_global_memory_reads<T>(elements, "Size");
  state.add_global_memory_writes<T>(elements);

  size_t tmp_size;
  dispatch_t::Dispatch(nullptr,
                       tmp_size,
                       d_input,
                       d_output,
                       op_t{},
                       init_t{T{}},
                       static_cast<int>(input.size()),
                       0 /* stream */);

  thrust::device_vector<nvbench::uint8_t> tmp(tmp_size);
  nvbench::uint8_t *d_tmp = thrust::raw_pointer_cast(tmp.data());

  state.exec([&](nvbench::launch &launch) {
    dispatch_t::Dispatch(thrust::raw_pointer_cast(tmp.data()),
                         tmp_size,
                         d_input,
                         d_output,
                         op_t{},
                         init_t{T{}},
                         static_cast<int>(input.size()),
                         launch.get_stream());
  });
}

template <typename T, typename OffsetT>
static void basic(std::integral_constant<bool, false>,
                  nvbench::state &,
                  nvbench::type_list<T, OffsetT>)
{
  // TODO Support
}

template <typename T, typename OffsetT>
static void basic(nvbench::state &state, nvbench::type_list<T, OffsetT> tl)
{
  basic(std::integral_constant < bool,
        (sizeof(OffsetT) == 4) && fits_in_default_shared_memory<T, OffsetT>() > {},
        state,
        tl);
}

using some_offset_types = nvbench::type_list<nvbench::int32_t>;

NVBENCH_BENCH_TYPES(basic, NVBENCH_TYPE_AXES(all_types, some_offset_types))
  .set_name("cub::DeviceScan::ExclusiveSum")
  .set_type_axes_names({"T{ct}", "OffsetT{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", nvbench::range(16, 28, 4));
