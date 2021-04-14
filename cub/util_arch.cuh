/******************************************************************************
 * Copyright (c) 2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2018, NVIDIA CORPORATION.  All rights reserved.
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
 * Static architectural properties by SM version.
 */

#pragma once

#include "util_cpp_dialect.cuh"
#include "util_namespace.cuh"
#include "util_macro.cuh"

// Legacy include; this used to be defined in here.
#include "detail/detect_cuda_runtime.cuh"

CUB_NAMESPACE_BEGIN

#ifndef DOXYGEN_SHOULD_SKIP_THIS    // Do not document

#if ((__CUDACC_VER_MAJOR__ >= 9) || defined(_NVHPC_CUDA) ||            \
     CUDA_VERSION >= 9000) &&                                                  \
  !defined(CUB_USE_COOPERATIVE_GROUPS)
#define CUB_USE_COOPERATIVE_GROUPS
#endif

/// Maximum number of devices supported.
#ifndef CUB_MAX_DEVICES
    #define CUB_MAX_DEVICES (128)
#endif

static_assert(CUB_MAX_DEVICES > 0, "CUB_MAX_DEVICES must be greater than 0.");

/// Number of threads per warp
#ifndef CUB_LOG_WARP_THREADS
    #define CUB_LOG_WARP_THREADS (5)
    #define CUB_WARP_THREADS (1 << CUB_LOG_WARP_THREADS)

    #define CUB_PTX_WARP_THREADS        CUB_WARP_THREADS
    #define CUB_PTX_LOG_WARP_THREADS    CUB_LOG_WARP_THREADS
#endif


/// Number of smem banks
#ifndef CUB_LOG_SMEM_BANKS
    #define CUB_LOG_SMEM_BANKS (5)
    #define CUB_SMEM_BANKS (1 << CUB_LOG_SMEM_BANKS)

    #define CUB_PTX_LOG_SMEM_BANKS      CUB_LOG_SMEM_BANKS
    #define CUB_PTX_SMEM_BANKS          CUB_SMEM_BANKS
#endif


/// Oversubscription factor
#ifndef CUB_SUBSCRIPTION_FACTOR
    #define CUB_SUBSCRIPTION_FACTOR (5)
    #define CUB_PTX_SUBSCRIPTION_FACTOR CUB_SUBSCRIPTION_FACTOR
#endif


/// Prefer padding overhead vs X-way conflicts greater than this threshold
#ifndef CUB_PREFER_CONFLICT_OVER_PADDING
    #define CUB_PREFER_CONFLICT_OVER_PADDING (1)
    #define CUB_PTX_PREFER_CONFLICT_OVER_PADDING CUB_PREFER_CONFLICT_OVER_PADDING
#endif


template <
    int NOMINAL_4B_BLOCK_THREADS,
    int NOMINAL_4B_ITEMS_PER_THREAD,
    typename T>
struct RegBoundScaling
{
    enum {
        ITEMS_PER_THREAD    = CUB_MAX(1, NOMINAL_4B_ITEMS_PER_THREAD * 4 / CUB_MAX(4, sizeof(T))),
        BLOCK_THREADS       = CUB_MIN(NOMINAL_4B_BLOCK_THREADS, (((1024 * 48) / (sizeof(T) * ITEMS_PER_THREAD)) + 31) / 32 * 32),
    };
};


template <
    int NOMINAL_4B_BLOCK_THREADS,
    int NOMINAL_4B_ITEMS_PER_THREAD,
    typename T>
struct MemBoundScaling
{
    enum {
        ITEMS_PER_THREAD    = CUB_MAX(1, CUB_MIN(NOMINAL_4B_ITEMS_PER_THREAD * 4 / sizeof(T), NOMINAL_4B_ITEMS_PER_THREAD * 2)),
        BLOCK_THREADS       = CUB_MIN(NOMINAL_4B_BLOCK_THREADS, (((1024 * 48) / (sizeof(T) * ITEMS_PER_THREAD)) + 31) / 32 * 32),
    };
};




#endif  // Do not document

CUB_NAMESPACE_END
