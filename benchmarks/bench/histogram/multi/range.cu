#include <cub/device/device_histogram.cuh>

#include <thrust/sequence.h>

#include <nvbench_helper.cuh>

// %RANGE% TUNE_ITEMS ipt 7:24:1
// %RANGE% TUNE_THREADS tpb 128:1024:32
// %RANGE% TUNE_RLE_COMPRESS rle 0:1:1
// %RANGE% TUNE_WORK_STEALING ws 0:1:1
// %RANGE% TUNE_MEM_PREFERENCE mem 0:2:1
// %RANGE% TUNE_LOAD ld 0:2:1

#if !TUNE_BASE

#if TUNE_LOAD == 0
#define TUNE_LOAD_MODIFIER cub::LOAD_DEFAULT
#elif TUNE_LOAD == 1
#define TUNE_LOAD_MODIFIER cub::LOAD_LDG
#else // TUNE_LOAD == 2
#define TUNE_LOAD_MODIFIER cub::LOAD_CA
#endif // TUNE_LOAD

#if TUNE_MEM_PREFERENCE == 0
constexpr cub::BlockHistogramMemoryPreference MEM_PREFERENCE = cub::GMEM;
#elif TUNE_MEM_PREFERENCE == 1
constexpr cub::BlockHistogramMemoryPreference MEM_PREFERENCE = cub::SMEM;
#else  // TUNE_MEM_PREFERENCE == 2
constexpr cub::BlockHistogramMemoryPreference MEM_PREFERENCE = cub::BLEND;
#endif // TUNE_MEM_PREFERENCE

template <typename SampleT, int NUM_ACTIVE_CHANNELS>
struct policy_hub_t
{
  template <int NOMINAL_ITEMS_PER_THREAD>
  struct TScale
  {
    enum
    {
      V_SCALE = (sizeof(SampleT) + sizeof(int) - 1) / sizeof(int),
      VALUE   = CUB_MAX((NOMINAL_ITEMS_PER_THREAD / NUM_ACTIVE_CHANNELS / V_SCALE), 1)
    };
  };

  struct policy_t : cub::ChainedPolicy<350, policy_t, policy_t>
  {
    using AgentHistogramPolicyT = cub::AgentHistogramPolicy<TUNE_THREADS,
                                                            TScale<TUNE_ITEMS>::VALUE,
                                                            cub::BLOCK_LOAD_DIRECT,
                                                            TUNE_LOAD_MODIFIER,
                                                            TUNE_RLE_COMPRESS,
                                                            MEM_PREFERENCE,
                                                            TUNE_WORK_STEALING>;
  };

  using MaxPolicy = policy_t;
};
#endif // !TUNE_BASE

template <class SampleT, class OffsetT>
SampleT get_upper_level(OffsetT bins, OffsetT elements)
{
  if constexpr (cuda::std::is_integral_v<SampleT>)
  {
    if constexpr (sizeof(SampleT) < sizeof(OffsetT))
    {
      const SampleT max_key = std::numeric_limits<SampleT>::max();
      return static_cast<SampleT>(std::min(bins, static_cast<OffsetT>(max_key)));
    }
    else
    {
      return static_cast<SampleT>(bins);
    }
  }

  return static_cast<SampleT>(elements);
}

template <typename SampleT, typename CounterT, typename OffsetT>
static void histogram(nvbench::state &state, nvbench::type_list<SampleT, CounterT, OffsetT>)
{
  constexpr int num_channels        = 4;
  constexpr int num_active_channels = 3;

  using sample_iterator_t = SampleT *;

#if !TUNE_BASE
  using policy_t   = policy_hub_t<key_t, num_active_channels>;
  using dispatch_t = cub::DispatchHistogram<num_channels, //
                                            num_active_channels,
                                            sample_iterator_t,
                                            CounterT,
                                            SampleT,
                                            OffsetT,
                                            policy_t>;
#else  // TUNE_BASE
  using dispatch_t = cub::DispatchHistogram<num_channels, //
                                            num_active_channels,
                                            sample_iterator_t,
                                            CounterT,
                                            SampleT,
                                            OffsetT>;
#endif // TUNE_BASE

  const auto entropy     = str_to_entropy(state.get_string("Entropy"));
  const auto elements    = state.get_int64("Elements{io}");
  const auto num_bins    = state.get_int64("Bins");
  const int num_levels_r = static_cast<int>(num_bins) + 1;
  const int num_levels_g = num_levels_r;
  const int num_levels_b = num_levels_g;

  const SampleT lower_level = 0;
  const SampleT upper_level = get_upper_level<SampleT>(num_bins, elements);

  SampleT step = (upper_level - lower_level) / num_bins;
  thrust::device_vector<SampleT> levels_r(num_bins + 1);

  // TODO Extract sequence to the helper TU
  thrust::sequence(levels_r.begin(), levels_r.end(), lower_level, step);
  thrust::device_vector<SampleT> levels_g = levels_r;
  thrust::device_vector<SampleT> levels_b = levels_g;

  SampleT *d_levels_r = thrust::raw_pointer_cast(levels_r.data());
  SampleT *d_levels_g = thrust::raw_pointer_cast(levels_g.data());
  SampleT *d_levels_b = thrust::raw_pointer_cast(levels_b.data());

  thrust::device_vector<SampleT> input(elements * num_channels);
  thrust::device_vector<CounterT> hist_r(num_bins);
  thrust::device_vector<CounterT> hist_g(num_bins);
  thrust::device_vector<CounterT> hist_b(num_bins);
  gen(seed_t{}, input, entropy, lower_level, upper_level);

  SampleT *d_input        = thrust::raw_pointer_cast(input.data());
  CounterT *d_histogram_r = thrust::raw_pointer_cast(hist_r.data());
  CounterT *d_histogram_g = thrust::raw_pointer_cast(hist_g.data());
  CounterT *d_histogram_b = thrust::raw_pointer_cast(hist_b.data());

  CounterT *d_histogram[num_active_channels] = {d_histogram_r, d_histogram_g, d_histogram_b};
  int num_levels[num_active_channels]        = {num_levels_r, num_levels_g, num_levels_b};
  SampleT *d_levels[num_active_channels]     = {d_levels_r, d_levels_g, d_levels_b};

  std::uint8_t *d_temp_storage = nullptr;
  std::size_t temp_storage_bytes{};

  cub::Int2Type<sizeof(SampleT) == 1> is_byte_sample;
  OffsetT num_row_pixels     = static_cast<OffsetT>(elements);
  OffsetT num_rows           = 1;
  OffsetT row_stride_samples = num_row_pixels;

  state.add_element_count(elements);
  state.add_global_memory_reads<SampleT>(elements * num_active_channels);
  state.add_global_memory_writes<CounterT>(num_bins * num_active_channels);

  dispatch_t::DispatchRange(d_temp_storage,
                            temp_storage_bytes,
                            d_input,
                            d_histogram,
                            num_levels,
                            d_levels,
                            num_row_pixels,
                            num_rows,
                            row_stride_samples,
                            0,
                            is_byte_sample);

  thrust::device_vector<nvbench::uint8_t> tmp(temp_storage_bytes);
  d_temp_storage = thrust::raw_pointer_cast(tmp.data());

  state.exec([&](nvbench::launch &launch) {
    dispatch_t::DispatchRange(d_temp_storage,
                              temp_storage_bytes,
                              d_input,
                              d_histogram,
                              num_levels,
                              d_levels,
                              num_row_pixels,
                              num_rows,
                              row_stride_samples,
                              launch.get_stream(),
                              is_byte_sample);
  });
}

using bin_types         = nvbench::type_list<int32_t>;
using some_offset_types = nvbench::type_list<int32_t>;

#ifdef TUNE_SampleT
using sample_types = nvbench::type_list<TUNE_SampleT>;
#else  // !defined(TUNE_SampleT)
using sample_types = nvbench::type_list<int8_t, int16_t, int32_t, int64_t, float, double>;
#endif // TUNE_SampleT

NVBENCH_BENCH_TYPES(histogram, NVBENCH_TYPE_AXES(sample_types, bin_types, some_offset_types))
  .set_name("cub::DeviceHistogram::MultiHistogramRange")
  .set_type_axes_names({"SampleT{ct}", "BinT{ct}", "OffsetT{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", nvbench::range(16, 28, 4))
  .add_int64_axis("Bins", {128, 2048, 2097152})
  .add_string_axis("Entropy", {"1.000", "0.544", "0.000"});
