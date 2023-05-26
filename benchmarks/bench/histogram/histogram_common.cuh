#pragma once

#include <cub/device/device_histogram.cuh>

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
#else // TUNE_MEM_PREFERENCE == 2
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
