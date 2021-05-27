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

#include <cub/config.cuh>
#include <cub/detail/detect_cuda_runtime.cuh>
#include <cub/detail/target.cuh>

/**
 * \def CUB_CDP_DISPATCH
 *
 * If CUDA Dynamic Parallelism / CUDA Nested Parallelism is available, always
 * run the parallel implementation. Otherwise, run the parallel implementation
 * when called from the host, and fallback to the sequential implementation on
 * the device.
 *
 * `par_impl` and `seq_impl` are blocks of C++ statements enclosed in
 * parentheses, similar to NV_IF_TARGET blocks:
 *
 * \code
 * CUB_CDP_DISPATCH((launch_parallel_kernel();), (run_serial_impl();));
 * \endcode
 */

#ifdef CUB_RUNTIME_ENABLED

// seq_impl unused.
#define CUB_CDP_DISPATCH(par_impl, seq_impl)                                   \
  NV_IF_TARGET(NV_ANY_TARGET, par_impl)

#else // CUB_RUNTIME_ENABLED

// Special case for NVCC -- need to inform the device path about the kernels
// that are launched from the host path.
#if defined(__CUDACC__) && defined(__CUDA_ARCH__)

// Device-side launch not supported, fallback to sequential in device code.
#define CUB_CDP_DISPATCH(par_impl, seq_impl)                                   \
  if (false)                                                                   \
  { /* Without this, the device pass won't compile any kernels. */             \
    NV_IF_TARGET(NV_ANY_TARGET, par_impl);                                     \
  }                                                                            \
  NV_IF_TARGET(NV_IS_HOST, par_impl, seq_impl)

#else // NVCC device pass

#define CUB_CDP_DISPATCH(par_impl, seq_impl)                                   \
  NV_IF_TARGET(NV_IS_HOST, par_impl, seq_impl)

#endif // NVCC device pass

#endif // CUB_RUNTIME_ENABLED
