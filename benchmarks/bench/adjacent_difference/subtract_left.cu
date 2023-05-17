#include <nvbench_helper.cuh>

// %RANGE% TUNE_ITEMS_PER_THREAD ipt 7:24:1
// %RANGE% TUNE_THREADS_PER_BLOCK tpb 128:1024:32

#include <cub/device/device_adjacent_difference.cuh>

#if !TUNE_BASE
struct policy_hub_t
{
  struct Policy350 : cub::ChainedPolicy<350, Policy350, Policy350>
  {
    using AdjacentDifferencePolicy =
      cub::AgentAdjacentDifferencePolicy<TUNE_THREADS_PER_BLOCK,
                                         TUNE_ITEMS_PER_THREAD,
                                         cub::BLOCK_LOAD_WARP_TRANSPOSE,
                                         cub::LOAD_CA,
                                         cub::BLOCK_STORE_WARP_TRANSPOSE>;
  };

  using MaxPolicy = Policy350;
};
#endif // !TUNE_BASE

template <class T, class OffsetT>
void adjacent_difference(nvbench::state& state, nvbench::type_list<T, OffsetT>)
{
  constexpr bool may_alias = false;
  constexpr bool read_left = true;

  using input_it_t = const T*;
  using output_it_t = T*;
  using difference_op_t = cub::Difference;
  using offset_t = typename cub::detail::ChooseOffsetT<OffsetT>::Type;

#if !TUNE_BASE
  using dispatch_t = cub::DispatchAdjacentDifference<input_it_t,
                                                     output_it_t,
                                                     difference_op_t,
                                                     offset_t,
                                                     may_alias,
                                                     read_left,
                                                     policy_hub_t>;
#else
  using dispatch_t = cub::DispatchAdjacentDifference<input_it_t,
                                                     output_it_t,
                                                     difference_op_t,
                                                     offset_t,
                                                     may_alias,
                                                     read_left>;
#endif // TUNE_BASE

  const auto elements = static_cast<std::size_t>(state.get_int64("Elements{io}"));
  thrust::device_vector<T> in(elements);
  thrust::device_vector<T> out(elements);
  gen(seed_t{}, in);

  input_it_t d_in   = thrust::raw_pointer_cast(in.data());
  output_it_t d_out = thrust::raw_pointer_cast(out.data());

  state.add_element_count(elements);
  state.add_global_memory_reads<T>(elements);
  state.add_global_memory_writes<T>(elements);

  std::size_t temp_storage_bytes{};
  dispatch_t::Dispatch(nullptr,
                       temp_storage_bytes,
                       d_in,
                       d_out,
                       static_cast<offset_t>(elements),
                       difference_op_t{},
                       0);

  thrust::device_vector<std::uint8_t> temp_storage(temp_storage_bytes);
  std::uint8_t* d_temp_storage = thrust::raw_pointer_cast(temp_storage.data());

  state.exec([&](nvbench::launch &launch) {
    dispatch_t::Dispatch(d_temp_storage,
                         temp_storage_bytes,
                         d_in,
                         d_out,
                         static_cast<offset_t>(elements),
                         difference_op_t{},
                         launch.get_stream());
  });
}


using types = nvbench::type_list<int32_t>;

NVBENCH_BENCH_TYPES(adjacent_difference, NVBENCH_TYPE_AXES(types, offset_types))
  .set_name("cub::DeviceAdjacentDifference::SubtractLeftCopy")
  .set_type_axes_names({"T{ct}", "OffsetT{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", nvbench::range(16, 28, 4));
