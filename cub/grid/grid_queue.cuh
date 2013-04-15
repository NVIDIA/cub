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

#include "../util_namespace.cuh"
#include "../util_macro.cuh"
#include "../util_debug.cuh"
#include "../util_arch.cuh"
#include "../util_allocator.cuh"

/// Optional outer namespace(s)
CUB_NS_PREFIX

/// CUB namespace
namespace cub {


/**
 * \addtogroup GridModule
 * @{
 */


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
    __host__ __device__ __forceinline__ cudaError_t Allocate(
//        DeviceAllocator *device_allocator = DefaultDeviceAllocator())
          DeviceAllocator *device_allocator = NULL)
    {
        if (d_counters) return cudaErrorInvalidValue;
        return CubDebug(device_allocator->DeviceAllocate((void**)&d_counters, sizeof(SizeT) * 2));
    }


    /// DeviceFree the resources used by this GridQueue.
    __host__ __device__ __forceinline__ cudaError_t Free(
//        DeviceAllocator *device_allocator = DefaultDeviceAllocator())
        DeviceAllocator *device_allocator = NULL)
    {
        if (!d_counters) return cudaErrorInvalidValue;
        cudaError_t error = CubDebug(device_allocator->DeviceFree(d_counters));
        d_counters = NULL;
        return error;
    }


    /// Prepares the queue for draining in the next kernel instance
    __host__ __device__ __forceinline__ cudaError_t PrepareDrain(
        SizeT fill_size,
        cudaStream_t stream = 0)
    {
#ifdef __CUDA_ARCH__
        d_counters[FILL] = fill_size;
        d_counters[DRAIN] = 0;
        return cudaSuccess;
#else
        SizeT counters[2];
        counters[FILL] = fill_size;
        counters[DRAIN] = 0;
        return CubDebug(cudaMemcpyAsync(d_counters, counters, sizeof(SizeT) * 2, cudaMemcpyHostToDevice, stream));
#endif
    }


    /// Prepares the queue for draining in the next kernel instance
    __host__ __device__ __forceinline__ cudaError_t PrepareDrainAfterFill(cudaStream_t stream = 0)
    {
#ifdef __CUDA_ARCH__
        d_counters[DRAIN] = 0;
        return cudaSuccess;
#else
        return PrepareDrain(0, stream);
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
    __host__ __device__ __forceinline__ cudaError_t FillSize(
        SizeT &fill_size,
        cudaStream_t stream = 0)
    {
#ifdef __CUDA_ARCH__
        fill_size = d_counters[FILL];
#else
        return CubDebug(cudaMemcpyAsync(&fill_size, d_counters + FILL, sizeof(SizeT), cudaMemcpyDeviceToHost, stream));
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


#ifndef DOXYGEN_SHOULD_SKIP_THIS    // Do not document


/**
 * Reset grid queue (call with 1 block of 1 thread)
 */
template <typename SizeT>
__global__ void PrepareDrainKernel(
    GridQueue<SizeT>    grid_queue,
    SizeT               num_items)
{
    grid_queue.PrepareDrain(num_items);
}



#endif // DOXYGEN_SHOULD_SKIP_THIS


/** @} */       // end group GridModule

}               // CUB namespace
CUB_NS_POSTFIX  // Optional outer namespace(s)


