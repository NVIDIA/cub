/*
 *  Copyright 2021 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#pragma once

#include <cub/detail/exec_check_disable.cuh>
#include <cub/util_arch.cuh>
#include <cub/util_namespace.cuh>

#include <nv/target>

#include <cuda_runtime_api.h>

CUB_NAMESPACE_BEGIN

namespace detail
{

/**
 * Call `cudaDeviceSynchronize()` using the proper API for the current CUB and
 * CUDA configuration.
 */
CUB_EXEC_CHECK_DISABLE
CUB_RUNTIME_FUNCTION inline cudaError_t device_synchronize()
{
  cudaError_t result = cudaErrorUnknown;

#ifdef CUB_RUNTIME_ENABLED

#if defined(__CUDACC__) &&                                                     \
  ((__CUDACC_VER_MAJOR__ > 11) ||                                              \
   ((__CUDACC_VER_MAJOR__ == 11) && (__CUDACC_VER_MINOR__ >= 6)))
  // CUDA >= 11.6
#define CUB_TMP_DEVICE_SYNC_IMPL                                               \
  result = __cudaDeviceSynchronizeDeprecationAvoidance();
#else // CUDA < 11.6
#define CUB_TMP_DEVICE_SYNC_IMPL result = cudaDeviceSynchronize();
#endif

#else // Device code without the CUDA runtime.
  // Device side CUDA API calls are not supported in this configuration.
#define CUB_TMP_DEVICE_SYNC_IMPL result = cudaErrorInvalidConfiguration;
#endif

  NV_IF_TARGET(NV_IS_HOST,
               (result = cudaDeviceSynchronize();),
               (CUB_TMP_DEVICE_SYNC_IMPL));

#undef CUB_TMP_DEVICE_SYNC_IMPL

  return result;
}

} // namespace detail

CUB_NAMESPACE_END
