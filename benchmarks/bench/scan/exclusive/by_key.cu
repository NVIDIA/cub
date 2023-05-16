#include <nvbench_helper.cuh>

// %RANGE% TUNE_ITEMS ipt 7:24:1
// %RANGE% TUNE_THREADS tpb 128:1024:32
// %RANGE% CUB_DETAIL_L2_BACKOFF_NS l2b 0:1200:5
// %RANGE% CUB_DETAIL_L2_WRITE_LATENCY_NS l2w 0:1200:5

#include <cub/device/device_scan.cuh>

#include <type_traits>

#if !TUNE_BASE
struct policy_hub_t
{
  struct policy_t : cub::ChainedPolicy<300, policy_t, policy_t>
  {
    using ScanByKeyPolicyT = cub::AgentScanByKeyPolicy<TUNE_THREADS,
                                                       TUNE_ITEMS,
                                                       // TODO Tune
                                                       cub::BLOCK_LOAD_WARP_TRANSPOSE,
                                                       cub::LOAD_CA,
                                                       cub::BLOCK_SCAN_WARP_SCANS,
                                                       cub::BLOCK_STORE_WARP_TRANSPOSE>;
  };

  using MaxPolicy = policy_t;
};
#endif // !TUNE_BASE

template <typename KeyT, typename ValueT, typename OffsetT>
static void scan(nvbench::state &state, nvbench::type_list<KeyT, ValueT, OffsetT>)
{
  using init_value_t    = ValueT;
  using op_t            = cub::Sum;
  using accum_t         = cub::detail::accumulator_t<op_t, init_value_t, ValueT>;
  using key_input_it_t  = const KeyT *;
  using val_input_it_t  = const ValueT *;
  using val_output_it_t = ValueT *;
  using equality_op_t   = cub::Equality;
  using offset_t        = OffsetT;

  #if !TUNE_BASE
  using policy_t   = policy_hub_t;
  using dispatch_t = cub::DispatchScanByKey<key_input_it_t,
                                            val_input_it_t,
                                            val_output_it_t,
                                            equality_op_t,
                                            op_t,
                                            init_value_t,
                                            offset_t,
                                            accum_t,
                                            policy_t>;
  #else // TUNE_BASE
  using dispatch_t = cub::DispatchScanByKey<key_input_it_t,
                                            val_input_it_t,
                                            val_output_it_t,
                                            equality_op_t,
                                            op_t,
                                            init_value_t,
                                            offset_t,
                                            accum_t>;
  #endif // TUNE_BASE

  const auto elements = static_cast<std::size_t>(state.get_int64("Elements{io}"));

  thrust::device_vector<ValueT> in_vals(elements);
  thrust::device_vector<ValueT> out_vals(elements);
  thrust::device_vector<KeyT> keys = gen_uniform_key_segments<KeyT>(seed_t{}, elements, 0, 5200);

  KeyT *d_keys       = thrust::raw_pointer_cast(keys.data());
  ValueT *d_in_vals  = thrust::raw_pointer_cast(in_vals.data());
  ValueT *d_out_vals = thrust::raw_pointer_cast(out_vals.data());

  state.add_element_count(elements);
  state.add_global_memory_reads<KeyT>(elements);
  state.add_global_memory_reads<ValueT>(elements);
  state.add_global_memory_writes<ValueT>(elements);

  size_t tmp_size;
  dispatch_t::Dispatch(nullptr,
                       tmp_size,
                       d_keys,
                       d_in_vals,
                       d_out_vals,
                       equality_op_t{},
                       op_t{},
                       init_value_t{},
                       static_cast<int>(elements),
                       0 /* stream */);

  thrust::device_vector<nvbench::uint8_t> tmp(tmp_size);
  nvbench::uint8_t *d_tmp = thrust::raw_pointer_cast(tmp.data());

  state.exec([&](nvbench::launch &launch) {
    dispatch_t::Dispatch(d_tmp,
                         tmp_size,
                         d_keys,
                         d_in_vals,
                         d_out_vals,
                         equality_op_t{},
                         op_t{},
                         init_value_t{},
                         static_cast<int>(elements),
                         launch.get_stream());
  });
}

using some_offset_types = nvbench::type_list<nvbench::int32_t>;

#ifdef TUNE_KeyT
using key_types = nvbench::type_list<TUNE_KeyT>;
#else // !defined(TUNE_KeyT)
using key_types = all_types;
#endif // TUNE_KeyT

#ifdef TUNE_ValueT
using value_types = nvbench::type_list<TUNE_ValueT>;
#else // !defined(TUNE_ValueT)
using value_types = nvbench::type_list<int8_t, int16_t, int32_t, int64_t, int128_t>;
#endif // TUNE_ValueT

NVBENCH_BENCH_TYPES(scan, NVBENCH_TYPE_AXES(key_types, value_types, some_offset_types))
  .set_name("cub::DeviceScan::ExclusiveSumByKey")
  .set_type_axes_names({"KeyT{ct}", "ValueT{ct}", "OffsetT{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", nvbench::range(16, 28, 4));
