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

/******************************************************************************
 * Abstraction for grid-wide queue management
 ******************************************************************************/

#pragma once

#include "../ns_wrapper.cuh"
#include "../macro_utils.cuh"
#include "../debug.cuh"
#include "../device_props.cuh"
#include "../allocator.cuh"

CUB_NS_PREFIX
namespace cub {


/**
 * Abstraction for grid-wide queue management.  Provides calling
 * threads with unique filling/draining offsets which can be used to
 * write/read from globally-shared vectors.
 *
 * Filling works by atomically-incrementing a zero-initialized counter, giving the
 * offset for writing items.
 *
 * Draining works by atomically-incrementing a different zero-initialized counter until
 * the previous fill-size is exceeded.
 */
template <typename SizeT>
class GridQueue
{
private:

    /// Counter indices
    enum
    {
        FILL    = 0,
        DRAIN   = 1,
    };

    SizeT *d_counters;  /// Pair of counters

public:


    /// Constructor
    __host__ __device__ __forceinline__ GridQueue() : d_counters(NULL) {}


    /// Allocate the resources necessary for this GridQueue.
    __host__ __device__ __forceinline__ cudaError_t Allocate()
    {
        if (d_counters) return cudaErrorInvalidValue;
        return CubDebug(DeviceAllocate((void**)&d_counters, sizeof(SizeT) * 2));
    }


    /// DeviceFree the resources used by this GridQueue.
    __host__ __device__ __forceinline__ cudaError_t Free()
    {
        if (!d_counters) return cudaErrorInvalidValue;
        cudaError_t error = CubDebug(DeviceFree(d_counters));
        d_counters = NULL;
        return error;
    }


    /// Prepares the queue for draining in the next kernel instance using
    /// \p fill_size as amount to drain.
    __host__ __device__ __forceinline__ cudaError_t PrepareDrain(SizeT fill_size)
    {
#ifdef __CUDA_ARCH__
        d_counters[FILL] = fill_size;
        d_counters[DRAIN] = 0;
        return cudaSuccess;
#else
        SizeT counters[2];
        counters[FILL] = fill_size;
        counters[DRAIN] = 0;
        return CubDebug(cudaMemcpy(d_counters, counters, sizeof(SizeT) * 2, cudaMemcpyHostToDevice));
#endif
    }


    /// Prepares the queue for draining in the next kernel instance using
    /// the current fill counter as the amount to drain.
    __host__ __device__ __forceinline__ cudaError_t PrepareDrain()
    {
#ifdef __CUDA_ARCH__
        d_counters[DRAIN] = 0;
        return cudaSuccess;
#else
        return PrepareDrain(0);
#endif
    }


    /// Prepares the queue for filling in the next kernel instance
    __host__ __device__ __forceinline__ cudaError_t PrepareFill()
    {
#ifdef __CUDA_ARCH__
        d_counters[FILL] = 0;
        return cudaSuccess;
#else
        return CubDebug(cudaMemset(d_counters + FILL, 0, sizeof(SizeT)));
#endif
    }


    /// Returns number of items filled in the previous kernel.
    __host__ __device__ __forceinline__ cudaError_t FillSize(SizeT &fill_size)
    {
#ifdef __CUDA_ARCH__
        fill_size = d_counters[FILL];
#else
        return CubDebug(cudaMemcpy(&fill_size, d_counters + FILL, sizeof(SizeT), cudaMemcpyDeviceToHost));
#endif
    }


    /// Drain num_items.  Returns offset from which to read items.
    __device__ __forceinline__ SizeT Drain(SizeT num_items)
    {
        return atomicAdd(d_counters + DRAIN, num_items);
    }


    /// Fill num_items.  Returns offset from which to write items.
    __device__ __forceinline__ SizeT Fill(SizeT num_items)
    {
        return atomicAdd(d_counters + FILL, num_items);
    }
};




} // namespace cub
CUB_NS_POSTFIX

