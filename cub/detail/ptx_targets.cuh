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

/**
 * \file
 * Utilities for selecting a type corresponding to a target PTX version.
 */

#pragma once

#include <cub/util_namespace.cuh>

#include <cuda_runtime_api.h>

CUB_NAMESPACE_BEGIN
namespace detail {

#ifdef DOXYGEN_SHOULD_SKIP_THIS // Only parse this during doxygen passes:

/**
 * A comma separated list of all virtual PTX architectures targeted by the
 * current compilation. This is used to restrict the number of kernel tunings
 * considered by `cub::detail::ptx_dispatch`. PTX architectures are specified
 * using the 3-digit form.
 *
 * This is effectively all of the possible values of `__CUDA_ARCH__` that
 * would be defined by nvcc during device passes, with no duplicates and
 * in ascending order. This is defined during all compilation passes.
 *
 * Users may define this directly to override any detected/default settings.
 *
 * By default, `__CUDA_ARCH_LIST__` will be used if it is defined. Otherwise,
 * `CUB_PTX_TARGETS` will contain all architectures used by `ptx_dispatch`.
 */
#define CUB_PTX_TARGETS

#else // Non-doxygen pass:

#ifndef CUB_PTX_TARGETS

#ifdef __CUDA_ARCH_LIST__
#define CUB_PTX_TARGETS __CUDA_ARCH_LIST__
#else
#define CUB_PTX_TARGETS 350,370,500,520,530,600,610,620,700,720,750,800,860
#endif

#endif // CUB_PTX_TARGETS predefined

#endif  // Do not document

} // namespace detail
CUB_NAMESPACE_END
