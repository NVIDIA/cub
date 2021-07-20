/******************************************************************************
 * Copyright (c) 2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2021, NVIDIA CORPORATION.  All rights reserved.
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

/**
 * \file
 * Utilities for CUDA dynamic parallelism.
 */

#pragma once

#include <cub/util_namespace.cuh>

#include <cub/detail/detect_cuda_runtime.cuh>
#include <cub/detail/target.cuh>
#include <cub/detail/type_traits.cuh>

#include <cuda_runtime_api.h>

CUB_NAMESPACE_BEGIN
namespace detail {

/**
 * If CUDA Dynamic Parallelism / CUDA Nested Parallelism is available, always
 * run the parallel implementation. Otherwise, run the parallel implementation
 * when called from the host, and fallback to the sequential implementation on
 * the device.
 *
 * `run_par` and `run_seq` are both parameterless lambdas functions that may
 * or may not return a value. Both lambdas must have the same return type (if
 * any), and any lambda result will be returned to the caller.
 *
 * `run_seq` will not be used when CDP is disabled. To avoid unnecessary
 * codegen, the body of `run_seq` may be a no-op in this case. The
 * CUB_RUNTIME_ENABLED macro is only defined when CDP is available and
 * may be used to remove `run_seq`'s implementation.
 * @{
 */
#pragma nv_exec_check_disable
template <typename ParallelImplT,
          typename SequentialImplT>
__host__ __device__
auto cdp_dispatch(ParallelImplT &&run_par, SequentialImplT &&run_seq)
  // Enable if the lambdas have a return type:
 -> typename std::enable_if<
      !std::is_same<cub::detail::invoke_result_t<ParallelImplT>, void>::value,
      cub::detail::invoke_result_t<ParallelImplT>
    >::type
{
  static_assert(
    std::is_same<cub::detail::invoke_result_t<ParallelImplT>,
                   cub::detail::invoke_result_t<SequentialImplT>>::value,
    "Parallel and Sequential implementation result types must match.");

  (void)run_seq; // maybe-unused
  (void)run_par; // maybe-unused

#ifdef CUB_RUNTIME_ENABLED
  return run_par();
#else // CUB_RUNTIME_ENABLED
  NV_IF_TARGET(NV_IS_DEVICE, (return run_seq();), (return run_par();));
#endif // CUB_RUNTIME_ENABLED
}

#pragma nv_exec_check_disable
template <typename ParallelImplT,
          typename SequentialImplT>
__host__ __device__
auto cdp_dispatch(ParallelImplT &&run_par, SequentialImplT &&run_seq)
  // Enable if the lambdas return void:
 -> typename std::enable_if<
      std::is_same<cub::detail::invoke_result_t<ParallelImplT>, void>::value
    >::type
{
  (void)run_seq; // maybe-unused
  (void)run_par; // maybe-unused

#ifdef CUB_RUNTIME_ENABLED
  run_par();
#else // CUB_RUNTIME_ENABLED
  NV_IF_TARGET(NV_IS_DEVICE, (run_seq();), (run_par();));
#endif // CUB_RUNTIME_ENABLED
}
/** @} */

}               // detail namespace
CUB_NAMESPACE_END
