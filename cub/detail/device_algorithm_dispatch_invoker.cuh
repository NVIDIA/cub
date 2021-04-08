/******************************************************************************
 * Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 ******************************************************************************/

#pragma once

#include <cub/detail/ptx_dispatch.cuh>

#include <cub/util_arch.cuh> // for CUB_RUNTIME_FUNCTION

CUB_NAMESPACE_BEGIN
namespace detail
{

/**
 * Helper dispatch functor to adapt the legacy "ChainedPolicy"
 * `Dispatch.Invoke<Policy>(...)` pattern to use `ptx_dispatch`.
 */
template <cub::detail::exec_space ExecSpace>
struct device_algorithm_dispatch_invoker
{
  cudaError_t status{cudaErrorInvalidPtx};

#pragma nv_exec_check_disable
  template <typename Policy, typename DeviceAlgorithmDispatch, typename... Args>
  __host__ __device__ __forceinline__
  void operator()(cub::detail::type_wrapper<Policy> policy_wrapper,
                  DeviceAlgorithmDispatch &&dispatch,
                  Args &&...args)
  {
    this->impl(exec_space_tag<ExecSpace>{},
               policy_wrapper,
               std::forward<DeviceAlgorithmDispatch>(dispatch),
               std::forward<Args>(args)...);
  }

private:
  template <typename Policy, typename DeviceAlgorithmDispatch, typename... Args>
  __host__ __forceinline__
  void impl(host_tag,
            cub::detail::type_wrapper<Policy>,
            DeviceAlgorithmDispatch &&dispatch,
            Args &&...args)
  {
    status = dispatch.template Invoke<Policy>(std::forward<Args>(args)...);
  }

  template <typename Policy, typename DeviceAlgorithmDispatch, typename... Args>
  __device__ __forceinline__
  void impl(device_tag,
            cub::detail::type_wrapper<Policy>,
            DeviceAlgorithmDispatch &&dispatch,
            Args &&...args)
  {
    status = dispatch.template Invoke<Policy>(std::forward<Args>(args)...);
  }

  template <typename Policy, typename DeviceAlgorithmDispatch, typename... Args>
  __host__ __device__ __forceinline__
  void impl(host_device_tag,
            cub::detail::type_wrapper<Policy>,
            DeviceAlgorithmDispatch &&dispatch,
            Args &&...args)
  {
    status = dispatch.template Invoke<Policy>(std::forward<Args>(args)...);
  }
};

} // namespace detail
CUB_NAMESPACE_END
