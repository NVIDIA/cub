/******************************************************************************
 * Copyright (c) 2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2013, NVIDIA CORPORATION.  All rights reserved.
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
 * Error and event logging routines.
 *
 * The following macros definitions are supported:
 * - \p CUB_LOG.  Simple event messages are printed to \p stdout.
 * - \p CUB_STDERR.  Error messages are printed to \p stderr (or \p stdout in device code).
 */

#pragma once

#include <stdio.h>
#include "util_namespace.cuh"
#include "util_arch.cuh"

/// Optional outer namespace(s)
CUB_NS_PREFIX

/// CUB namespace
namespace cub {


/**
 * \addtogroup UtilModule
 * @{
 */


/// CUB error reporting macro (prints error messages to stderr)
#if (defined(DEBUG) || defined(_DEBUG))
    #define CUB_STDERR
#endif


/**
 * \brief %If \p CUB_STDERR is defined and \p error is not \p cudaSuccess, \p message is printed to \p stderr (or \p stdout in device code) along with the supplied source context.
 *
 * \return The CUDA error.
 */
__host__ __device__ __forceinline__ cudaError_t Debug(
    cudaError_t     error,
    const char*     message,
    const char*     filename,
    int             line)
{
#ifdef CUB_STDERR
    if (error)
    {
    #if (CUB_PTX_ARCH == 0)
        fprintf(stderr, "[%s, %d] %s (CUDA error %d: %s)\n", filename, line, message, error, cudaGetErrorString(error));
        fflush(stderr);
    #elif (CUB_PTX_ARCH >= 200)
        printf("[block %d, thread %d, %s, %d] %s (CUDA error %d: %s)\n", blockIdx.x, threadIdx.x, filename, line, message, error, cudaGetErrorString(error));
    #endif
    }
#endif
    return error;
}


/**
 * \brief %If \p CUB_STDERR is defined and \p error is not \p cudaSuccess, the corresponding error message is printed to \p stderr (or \p stdout in device code) along with the supplied source context.
 *
 * \return The CUDA error.
 */
__host__ __device__ __forceinline__ cudaError_t Debug(
    cudaError_t     error,
    const char*     filename,
    int             line)
{
#ifdef CUB_STDERR
    if (error)
    {
    #if (CUB_PTX_ARCH == 0)
        fprintf(stderr, "[%s, %d] (CUDA error %d: %s)\n", filename, line, error, cudaGetErrorString(error));
        fflush(stderr);
    #elif (CUB_PTX_ARCH >= 200)
        printf("[block %d, thread %d, %s, %d] (CUDA error %d: %s)\n", blockIdx.x, threadIdx.x, filename, line, error, cudaGetErrorString(error));
    #endif
    }
#endif
    return error;
}


/**
 * Debug macro
 */
#define CubDebug(e) cub::Debug(e, __FILE__, __LINE__)


/**
 * Debug macro with exit
 */
#define CubDebugExit(e) if (cub::Debug(e, __FILE__, __LINE__)) exit(1)


/**
 * Log macro for printf statements.
 */
#ifdef CUB_LOG
    #if (CUB_PTX_ARCH == 0)
        #define CubLog(p) p; fflush(stdout);
    #elif (CUB_PTX_ARCH >= 200)
        #define CubLog(p) p;
    #endif
#else
    #define CubLog(p)
#endif



/** @} */       // end group UtilModule

}               // CUB namespace
CUB_NS_POSTFIX  // Optional outer namespace(s)
