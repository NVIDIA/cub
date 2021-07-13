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

#include <cub/util_compiler.cuh>

// NVBug 2431416: If a kernel is only launched on the host (e.g. the launch is
// ifdef'd out when !defined(__CUDA_ARCH__)), spurious unused parameter warnings
// will be emitted from the __wrapper_device_stub_[kernel name] stub.
//
// We regularly hit this due to per-architecture tunings, and there is no known
// workaround that can be easily applied. We just disable unused parameter
// warnings around the kernel definitions.

// clang-format off

#if defined(__CUDACC__) // nvcc only

#if CUB_HOST_COMPILER == CUB_HOST_COMPILER_GCC ||                              \
    CUB_HOST_COMPILER == CUB_HOST_COMPILER_CLANG

// Clang honors GCC pragmas
#define CUB_KERNEL_BEGIN                                                       \
  _Pragma("GCC diagnostic push")                                               \
  _Pragma("GCC diagnostic ignored \"-Wunused-parameter\"")
#define CUB_KERNEL_END \
  _Pragma("GCC diagnostic pop")

#elif CUB_HOST_COMPILER == CUB_HOST_COMPILER_MSVC

#define CUB_KERNEL_BEGIN                                                       \
  __pragma(warning(push))                                                      \
  __pragma(warning(disable : 4100)) /* "unreferenced formal parameter" */
#define CUB_KERNEL_END  \
  __pragma(warning(pop))

#endif // host compiler switch
#endif // is nvcc

// Fallback
#ifndef CUB_KERNEL_BEGIN
#define CUB_KERNEL_BEGIN
#define CUB_KERNEL_END
#endif


// clang-format on
