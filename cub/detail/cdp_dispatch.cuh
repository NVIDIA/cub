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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 *IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
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

// FIXME These reimplement portions of the NV_IF_TARGET macros. Once we have a
// dependency on libcu++, these should be replaced by proper if-target macros.
#define CUB_STRIP_PAREN2(...) __VA_ARGS__
#define CUB_STRIP_PAREN1(...) CUB_STRIP_PAREN2 __VA_ARGS__
#define CUB_STRIP_PAREN(...) CUB_STRIP_PAREN1(__VA_ARGS__)
#define CUB_BLOCK_EXPAND(...) CUB_STRIP_PAREN(__VA_ARGS__)

#ifdef CUB_RUNTIME_ENABLED

// seq_impl unused.
#define CUB_CDP_DISPATCH(par_impl, seq_impl)                                   \
  /* FIXME Just use: NV_IF_TARGET(NV_ANY_TARGET, par_impl) */                  \
  do                                                                           \
  {                                                                            \
    CUB_BLOCK_EXPAND(par_impl)                                                 \
  } while (false)

#else // CUB_RUNTIME_ENABLED

// Special case for NVCC -- need to inform the device path about the kernels
// that are launched from the host path.
#if defined(__CUDACC__) && defined(__CUDA_ARCH__)

// Device-side launch not supported, fallback to sequential in device code.
// Since this is a special case for NVCC's device pass, we just execute the
// serial branch. The parallel implementation is still instantiated in a dead
// branch to make sure that the kernels get compiled for each device.
#define CUB_CDP_DISPATCH(par_impl, seq_impl)                                   \
  if (false)                                                                   \
  { /* Without this, the device pass won't compile any kernels. */             \
    /* FIXME Just use: NV_IF_TARGET(NV_ANY_TARGET, par_impl); */               \
    CUB_BLOCK_EXPAND(par_impl)                                                 \
  }                                                                            \
  /* FIXME: Just use: NV_IF_TARGET(NV_ANY_TARGET, seq_impl) */                 \
  do                                                                           \
  {                                                                            \
    CUB_BLOCK_EXPAND(seq_impl)                                                 \
  } while (false)

#else // NVCC device pass

// Launch parallel implementation on host, serial implementation on device.
#define CUB_CDP_DISPATCH(par_impl, seq_impl)                                   \
  /* FIXME Just use: NV_IF_TARGET(NV_IS_HOST, par_impl, seq_impl) */           \
  do                                                                           \
  {                                                                            \
    if (CUB_IS_HOST_CODE)                                                      \
    {                                                                          \
      CUB_CDP_HOST_DISPATCH_IMPL(par_impl)                                     \
    }                                                                          \
    else                                                                       \
    {                                                                          \
      CUB_CDP_DEVICE_DISPATCH_IMPL(seq_impl)                                   \
    }                                                                          \
  } while (false)

// Host implementation:
#if CUB_INCLUDE_HOST_CODE
#define CUB_CDP_HOST_DISPATCH_IMPL(par_impl) CUB_BLOCK_EXPAND(par_impl)
#else // CUB_INCLUDE_HOST_CODE
#define CUB_CDP_HOST_DISPATCH_IMPL(par_impl)
#endif // CUB_INCLUDE_HOST_CODE

// Device implementation
#if CUB_INCLUDE_DEVICE_CODE
#define CUB_CDP_DEVICE_DISPATCH_IMPL(seq_impl) CUB_BLOCK_EXPAND(seq_impl)
#else // CUB_INCLUDE_DEVICE_CODE
#define CUB_CDP_DEVICE_DISPATCH_IMPL(seq_impl)
#endif // CUB_INCLUDE_DEVICE_CODE

#endif // NVCC device pass

#endif // CUB_RUNTIME_ENABLED
