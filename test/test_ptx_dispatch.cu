/******************************************************************************
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

#include <cub/detail/detect_cuda_runtime.cuh>
#include <cub/detail/exec_check_disable.cuh>
#include <cub/detail/ptx_dispatch.cuh>
#include <cub/detail/ptx_targets.cuh>
#include <cub/util_debug.cuh>
#include <cub/util_ptx.cuh>

#include <cuda/std/type_traits>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

#include "test_util.h"

// %PARAM% TEST_CDP cdp 0:1

// ptx_dispatch functor that records the dispatched arch.
struct record_arch
{
  int arch{-1};

  template <int PtxArch>
  __host__ __device__ void operator()(cub::detail::type_wrapper<cub::detail::ptx<PtxArch>>)
  {
    this->arch = PtxArch;
  }
};

//--------------------------------------------------------------------------------------------------
// Unit tests:

void test_ptx()
{
  using cub::detail::ptx;
  StaticAssertEquals((ptx<0>::ptx_arch), (0));
  StaticAssertEquals((ptx<800>::ptx_arch), (800));
}

void test_runtime_exec_space()
{
#if TEST_CDP == 0
  StaticAssertEquals((cub::detail::runtime_exec_space), (cub::detail::exec_space::host));
#else
  StaticAssertEquals((cub::detail::runtime_exec_space), (cub::detail::exec_space::host_device));
#endif
}

void test_ptx_arch_lookup()
{
  using cub::detail::no_ptx_type;
  using cub::detail::ptx;
  using cub::detail::ptx_arch_lookup;
  using cub::detail::type_list;

  StaticAssertSame((typename ptx_arch_lookup<0, type_list<>>::type), (no_ptx_type));
  StaticAssertSame((typename ptx_arch_lookup<800, type_list<>>::type), (no_ptx_type));

  {
    using ptx_types = type_list<ptx<700>, ptx<500>, ptx<350>>;

    StaticAssertSame((typename ptx_arch_lookup<100, ptx_types>::type), (no_ptx_type));
    StaticAssertSame((typename ptx_arch_lookup<350, ptx_types>::type), (ptx<350>));
    StaticAssertSame((typename ptx_arch_lookup<370, ptx_types>::type), (ptx<350>));
    StaticAssertSame((typename ptx_arch_lookup<500, ptx_types>::type), (ptx<500>));
    StaticAssertSame((typename ptx_arch_lookup<600, ptx_types>::type), (ptx<500>));
    StaticAssertSame((typename ptx_arch_lookup<700, ptx_types>::type), (ptx<700>));
    StaticAssertSame((typename ptx_arch_lookup<800, ptx_types>::type), (ptx<700>));
  }
}

void test_restrict_ptx_types()
{
  using cub::detail::no_ptx_type;
  using cub::detail::ptx;
  using cub::detail::restrict_ptx_types_impl;
  using cub::detail::type_list;

  using ptx_types = type_list<ptx<800>, ptx<700>, ptx<600>, ptx<500>, ptx<350>>;

  StaticAssertSame((typename restrict_ptx_types_impl<ptx_types, 200>::type), //
                   (type_list<>));
  StaticAssertSame((typename restrict_ptx_types_impl<ptx_types, 350>::type), //
                   (type_list<ptx<350>>));
  StaticAssertSame((typename restrict_ptx_types_impl<ptx_types, 370>::type), //
                   (type_list<ptx<350>>));
  StaticAssertSame((typename restrict_ptx_types_impl<ptx_types, 350, 370>::type),
                   (type_list<ptx<350>>));
  StaticAssertSame((typename restrict_ptx_types_impl<ptx_types, 350, 370, 500>::type),
                   (type_list<ptx<500>, ptx<350>>));
  StaticAssertSame((typename restrict_ptx_types_impl<ptx_types, 860>::type), //
                   (type_list<ptx<800>>));
  StaticAssertSame((typename restrict_ptx_types_impl<ptx_types, 500, 860>::type),
                   (type_list<ptx<800>, ptx<500>>));
  StaticAssertSame((typename restrict_ptx_types_impl<ptx_types, 520, 860>::type),
                   (type_list<ptx<800>, ptx<500>>));
  StaticAssertSame((typename restrict_ptx_types_impl<ptx_types, 500, 520, 860>::type),
                   (type_list<ptx<800>, ptx<500>>));
  StaticAssertSame((typename restrict_ptx_types_impl<ptx_types, 500, 520, 860>::type),
                   (type_list<ptx<800>, ptx<500>>));
  // clang-format off
  StaticAssertSame((typename restrict_ptx_types_impl<ptx_types, 350, 370, 500, 520, 530, 600, 610,
                                                                620, 700, 720, 750, 800, 860>
                    ::type),
                   (ptx_types));
  // clang-format on
}

void test_check_ptx_type_list()
{
  using cub::detail::ptx;
  using cub::detail::type_list;
  using cub::detail::check_ptx_type_list_impl::is_valid;

  struct invalid
  {}; // does not define ptx_arch

  // valid:
  using ptx_types_empty      = type_list<>;
  using ptx_types_descending = type_list<ptx<800>, ptx<700>, ptx<500>, ptx<350>>;

  // invalid:
  using ptx_types_ascending        = type_list<ptx<350>, ptx<500>, ptx<700>, ptx<800>>;
  using ptx_types_invalid_single   = type_list<invalid>;
  using ptx_types_invalid_first    = type_list<invalid, ptx<700>, ptx<350>>;
  using ptx_types_invalid_mid      = type_list<ptx<800>, invalid, ptx<350>>;
  using ptx_types_invalid_last     = type_list<ptx<800>, ptx<700>, invalid>;
  using ptx_types_invalid_all      = type_list<invalid, invalid, invalid>;
  using ptx_types_duplicate_single = type_list<ptx<800>, ptx<800>>;
  using ptx_types_duplicate_first  = type_list<ptx<800>, ptx<800>, ptx<700>, ptx<500>>;
  using ptx_types_duplicate_mid    = type_list<ptx<800>, ptx<700>, ptx<700>, ptx<500>>;
  using ptx_types_duplicate_last   = type_list<ptx<800>, ptx<700>, ptx<500>, ptx<500>>;
  using ptx_types_duplicate_all =
    type_list<ptx<800>, ptx<800>, ptx<700>, ptx<700>, ptx<500>, ptx<500>>;

  StaticAssertEquals((is_valid<ptx_types_empty>::value), (true));
  StaticAssertEquals((is_valid<ptx_types_descending>::value), (true));
  StaticAssertEquals((is_valid<ptx_types_ascending>::value), (false));
  StaticAssertEquals((is_valid<ptx_types_invalid_single>::value), (false));
  StaticAssertEquals((is_valid<ptx_types_invalid_first>::value), (false));
  StaticAssertEquals((is_valid<ptx_types_invalid_mid>::value), (false));
  StaticAssertEquals((is_valid<ptx_types_invalid_last>::value), (false));
  StaticAssertEquals((is_valid<ptx_types_invalid_all>::value), (false));
  StaticAssertEquals((is_valid<ptx_types_duplicate_single>::value), (false));
  StaticAssertEquals((is_valid<ptx_types_duplicate_first>::value), (false));
  StaticAssertEquals((is_valid<ptx_types_duplicate_mid>::value), (false));
  StaticAssertEquals((is_valid<ptx_types_duplicate_last>::value), (false));
  StaticAssertEquals((is_valid<ptx_types_duplicate_all>::value), (false));
}

// Execute a ptx_dispatch_host_impl with record_arch and verify the runtime result.
template <typename DispatchImpl>
void validate_dispatch_host_impl(int target_ptx_arch, int expected_result)
{
  record_arch functor{};
  DispatchImpl::exec(target_ptx_arch, functor);
  if (functor.arch != expected_result)
  {
    std::printf("%s:%d: Expected dispatch to %d, actually dispatched to %d.\n",
                __FILE__,
                __LINE__,
                expected_result,
                functor.arch);
    std::exit(1);
  }
}

void test_ptx_dispatch_host_impl()
{
  using cub::detail::ptx;
  using cub::detail::ptx_dispatch_host_impl;
  using cub::detail::type_list;

  {
    using ptx_types = type_list<>;

    validate_dispatch_host_impl<ptx_dispatch_host_impl<ptx_types>>(0, -1);
    validate_dispatch_host_impl<ptx_dispatch_host_impl<ptx_types>>(800, -1);
  }

  {
    using ptx_types = type_list<ptx<600>>;

    validate_dispatch_host_impl<ptx_dispatch_host_impl<ptx_types>>(300, -1);
    validate_dispatch_host_impl<ptx_dispatch_host_impl<ptx_types>>(600, 600);
    validate_dispatch_host_impl<ptx_dispatch_host_impl<ptx_types>>(800, 600);
  }

  {
    using ptx_types = type_list<ptx<800>, ptx<700>, ptx<600>, ptx<520>, ptx<350>>;

    validate_dispatch_host_impl<ptx_dispatch_host_impl<ptx_types>>(300, -1);
    validate_dispatch_host_impl<ptx_dispatch_host_impl<ptx_types>>(350, 350);
    validate_dispatch_host_impl<ptx_dispatch_host_impl<ptx_types>>(500, 350);
    validate_dispatch_host_impl<ptx_dispatch_host_impl<ptx_types>>(520, 520);
    validate_dispatch_host_impl<ptx_dispatch_host_impl<ptx_types>>(530, 520);
    validate_dispatch_host_impl<ptx_dispatch_host_impl<ptx_types>>(600, 600);
    validate_dispatch_host_impl<ptx_dispatch_host_impl<ptx_types>>(620, 600);
    validate_dispatch_host_impl<ptx_dispatch_host_impl<ptx_types>>(700, 700);
    validate_dispatch_host_impl<ptx_dispatch_host_impl<ptx_types>>(750, 700);
    validate_dispatch_host_impl<ptx_dispatch_host_impl<ptx_types>>(800, 800);
    validate_dispatch_host_impl<ptx_dispatch_host_impl<ptx_types>>(860, 800);
  }
}

// Execute a ptx_dispatch_device_impl with record_arch and verify the runtime result.
// The target_ptx_arch is statically encoded in the DispatchImpl
template <typename DispatchImpl>
__device__ void validate_dispatch_device_impl(int expected_result)
{
  record_arch functor{};
  DispatchImpl::exec(functor);
  if (functor.arch != expected_result)
  {
    std::printf("%s:%d: Expected dispatch to %d, actually dispatched to %d.\n",
                __FILE__,
                __LINE__,
                expected_result,
                functor.arch);
    cub::ThreadTrap();
  }
}

__global__ void test_ptx_dispatch_device_impl_kernel()
{
  using cub::detail::ptx;
  using cub::detail::ptx_dispatch_device_impl;
  using cub::detail::type_list;

  {
    using ptx_types = type_list<>;

    validate_dispatch_device_impl<ptx_dispatch_device_impl<0, ptx_types>>(-1);
    validate_dispatch_device_impl<ptx_dispatch_device_impl<800, ptx_types>>(-1);
  }

  {
    using ptx_types = type_list<ptx<600>>;

    validate_dispatch_device_impl<ptx_dispatch_device_impl<300, ptx_types>>(-1);
    validate_dispatch_device_impl<ptx_dispatch_device_impl<600, ptx_types>>(600);
    validate_dispatch_device_impl<ptx_dispatch_device_impl<800, ptx_types>>(600);
  }

  {
    using ptx_types = type_list<ptx<800>, ptx<700>, ptx<600>, ptx<520>, ptx<350>>;

    validate_dispatch_device_impl<ptx_dispatch_device_impl<300, ptx_types>>(-1);
    validate_dispatch_device_impl<ptx_dispatch_device_impl<350, ptx_types>>(350);
    validate_dispatch_device_impl<ptx_dispatch_device_impl<500, ptx_types>>(350);
    validate_dispatch_device_impl<ptx_dispatch_device_impl<520, ptx_types>>(520);
    validate_dispatch_device_impl<ptx_dispatch_device_impl<530, ptx_types>>(520);
    validate_dispatch_device_impl<ptx_dispatch_device_impl<600, ptx_types>>(600);
    validate_dispatch_device_impl<ptx_dispatch_device_impl<620, ptx_types>>(600);
    validate_dispatch_device_impl<ptx_dispatch_device_impl<700, ptx_types>>(700);
    validate_dispatch_device_impl<ptx_dispatch_device_impl<750, ptx_types>>(700);
    validate_dispatch_device_impl<ptx_dispatch_device_impl<800, ptx_types>>(800);
    validate_dispatch_device_impl<ptx_dispatch_device_impl<860, ptx_types>>(800);
  }
}

void test_ptx_dispatch_device_impl()
{
  test_ptx_dispatch_device_impl_kernel<<<1, 1>>>();
  CubDebugExit(cudaDeviceSynchronize());
}

// Executes a ptx_dispatch with record_arch and verifies the runtime result.
CUB_EXEC_CHECK_DISABLE
template <typename DispatchT>
__host__ __device__ void validate_ptx_dispatch(int expected_result)
{
  record_arch functor{};
  DispatchT::exec(functor);
  if (functor.arch != expected_result)
  {
    std::printf("%s:%d: Expected dispatch to %d, actually dispatched to %d.\n",
                __FILE__,
                __LINE__,
                expected_result,
                functor.arch);
    NV_IF_TARGET(NV_IS_HOST, (std::exit(1);), (cub::ThreadTrap();));
  }
}

template <typename PtxTypes>
void test_ptx_dispatch_host(int expected_arch)
{
  std::printf(" - Testing host-side dispatch...\n");
  {
    using dispatcher_t = cub::detail::ptx_dispatch<PtxTypes, cub::detail::exec_space::host>;
    validate_ptx_dispatch<dispatcher_t>(expected_arch);
  }

  {
    using dispatcher_t = cub::detail::ptx_dispatch<PtxTypes, cub::detail::runtime_exec_space>;
    validate_ptx_dispatch<dispatcher_t>(expected_arch);
  }
}

template <typename PtxTypes>
__global__ void test_ptx_dispatch_device_kernel(int expected_arch)
{
  {
    using dispatcher_t = cub::detail::ptx_dispatch<PtxTypes, cub::detail::exec_space::device>;
    validate_ptx_dispatch<dispatcher_t>(expected_arch);
  }

#if TEST_CDP == 1
  {
    using dispatcher_t = cub::detail::ptx_dispatch<PtxTypes, cub::detail::runtime_exec_space>;
    validate_ptx_dispatch<dispatcher_t>(expected_arch);
  }
#endif
}

template <typename PtxTypes>
void test_ptx_dispatch_device(int expected_arch)
{
  std::printf(" - Testing device-side dispatch...\n");
  test_ptx_dispatch_device_kernel<PtxTypes><<<1, 1>>>(expected_arch);
  CubDebugExit(cudaDeviceSynchronize());
}

void test_ptx_dispatch()
{
  using cub::detail::ptx;
  using cub::detail::type_list;

  const std::vector<int> cub_ptx_targets{CUB_PTX_TARGETS};

  std::printf("CUB_PTX_TARGETS: ");
  for (const auto arch : cub_ptx_targets)
  {
    std::printf("%d ", arch);
  }
  std::printf("\n");

  // These need to stay synced:
  using ptx_types = type_list<ptx<800>, ptx<700>, ptx<600>, ptx<500>, ptx<350>>; // descending
  const std::vector<int> ptx_arches{350, 500, 600, 700, 800};                    // ascending

  std::printf("ptx_types: ");
  for (const auto arch : ptx_arches)
  {
    std::printf("ptx<%d> ", arch);
  }
  std::printf("\n");

  int device_count{};
  CubDebugExit(cudaGetDeviceCount(&device_count));
  std::printf("Detected %d CUDA devices.\n", device_count);

  for (int device_id = 0; device_id < device_count; ++device_id)
  {
    CubDebugExit(cudaSetDevice(device_id));

    cudaDeviceProp prop;
    CubDebugExit(cudaGetDeviceProperties(&prop, device_id));

    // Find the best matching arch in arches that supports target.
    auto lookup_ptx = [&](int target, const std::vector<int> &arches) {
      int result = -1;
      for (const auto arch : arches)
      {
        if (arch <= target)
        {
          result = arch;
        }
        else
        {
          break;
        }
      }
      return result;
    };

    // Actual device architecture
    int device_arch = prop.major * 100 + prop.minor * 10;

    // Best matching arch to device_arch in CUB_PTX_TARGETS (__CUDA_ARCH_LIST__)
    int target_arch = lookup_ptx(device_arch, cub_ptx_targets);

    // Best matching arch to target_arch in the dispatched ptx_types
    int tuning_arch = lookup_ptx(target_arch, ptx_arches);

    std::printf("Testing Device %d (device_arch=%d target_arch=%d tuning_arch=%d) '%s'...\n",
                device_id,
                device_arch,
                target_arch,
                tuning_arch,
                prop.name);

    test_ptx_dispatch_host<ptx_types>(tuning_arch);
    test_ptx_dispatch_device<ptx_types>(tuning_arch);
  }
}

int main()
{
  test_ptx();
  test_runtime_exec_space();
  test_ptx_arch_lookup();
  test_restrict_ptx_types();
  test_check_ptx_type_list();
  test_ptx_dispatch_host_impl();
  test_ptx_dispatch_device_impl();
  test_ptx_dispatch();
}
