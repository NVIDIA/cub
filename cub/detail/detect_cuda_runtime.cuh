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

#include <cuda_runtime_api.h>

CUB_NAMESPACE_BEGIN
namespace detail
{

#ifdef DOXYGEN_SHOULD_SKIP_THIS // Only parse this during doxygen passes:

/** Defined if RDC is enabled. */
#define CUB_RUNTIME_ENABLED

/**
 * Execution space for RDC implementations (`__host__` when RDC is off,
 * `__host__ __device__` when RDC is on).
 */
#define CUB_RUNTIME_FUNCTION

#else // Non-doxygen pass:

#ifndef CUB_RUNTIME_FUNCTION

#if defined(__CUDACC_RDC__)

#define CUB_RUNTIME_ENABLED
#define CUB_RUNTIME_FUNCTION __host__ __device__

#else // RDC disabled:

#define CUB_RUNTIME_FUNCTION __host__

#endif // RDC enabled
#endif // CUB_RUNTIME_FUNCTION predefined

#endif // Do not document

} // namespace detail
CUB_NAMESPACE_END
