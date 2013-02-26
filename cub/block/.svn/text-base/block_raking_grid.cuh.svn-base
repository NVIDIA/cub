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
 * Threadblock raking grid abstraction.
 *
 * Threadblock threads place elements into shared "grid" and then reduce
 * parallelism to one "raking" warp whose threads can perform sequential
 * aggregation operations on consecutive sequences of shared items.  Padding
 * is provided to eliminate bank conflicts (for most data types).
 ******************************************************************************/

#pragma once

#include "../device_props.cuh"
#include "../type_utils.cuh"
#include "../ns_wrapper.cuh"

CUB_NS_PREFIX
namespace cub {


/**
 * Threadblock raking grid abstraction.
 *
 * Threadblock threads place elements into shared "grid" and then reduce
 * parallelism to one "raking" warp whose threads can perform sequential
 * aggregation operations on consecutive sequences of shared items.  Padding
 * is provided to eliminate bank conflicts (for most data types).
 */
template <
    int         BLOCK_THREADS,        // The threadblock size in threads
    typename     T,                    // The reduction type
    int         BLOCK_STRIPS = 1>        // When strip-mining, the number of threadblock-strips per tile
struct BlockRakingGrid
{
    //---------------------------------------------------------------------
    // Constants and typedefs
    //---------------------------------------------------------------------

    enum
    {
        // The total number of elements that need to be cooperatively reduced
        SHARED_ELEMENTS = BLOCK_THREADS * BLOCK_STRIPS,

        // Maximum number of warp-synchronous raking threads
        MAX_RAKING_THREADS = CUB_MIN(BLOCK_THREADS, DeviceProps::WARP_THREADS),

        // Number of raking elements per warp-synchronous raking thread (rounded up)
        RAKING_LENGTH = (SHARED_ELEMENTS + MAX_RAKING_THREADS - 1) / MAX_RAKING_THREADS,

        // Number of warp-synchronous raking threads
        RAKING_THREADS = (SHARED_ELEMENTS + RAKING_LENGTH - 1) / RAKING_LENGTH,

        // Total number of raking elements in the grid
        RAKING_ELEMENTS = RAKING_THREADS * RAKING_LENGTH,

        // Number of bytes per shared memory segment
        SEGMENT_BYTES = DeviceProps::SMEM_BANKS * DeviceProps::SMEM_BANK_BYTES,

        // Number of elements per shared memory segment (rounded up)
        SEGMENT_LENGTH = (SEGMENT_BYTES + sizeof(T) - 1) / sizeof(T),

        // Stride in elements between padding blocks (insert a padding block after each), must be a multiple of raking elements
        PADDING_STRIDE = CUB_ROUND_UP_NEAREST(SEGMENT_LENGTH, RAKING_LENGTH),

        // Number of elements per padding block
        PADDING_ELEMENTS = (DeviceProps::SMEM_BANK_BYTES + sizeof(T) - 1) / sizeof(T),

        // Total number of elements in the raking grid
        GRID_ELEMENTS = RAKING_ELEMENTS + (RAKING_ELEMENTS / PADDING_STRIDE),

        // Whether or not we need bounds checking during raking (the number of
        // reduction elements is not a multiple of the warp size)
        UNGUARDED = (SHARED_ELEMENTS % DeviceProps::WARP_THREADS == 0),
    };


    /**
     * Shared memory storage type
     */
    typedef T SmemStorage[BlockRakingGrid::GRID_ELEMENTS];


    /**
     * Pointer for placement into raking grid (with padding)
     */
    static __device__ __forceinline__ T* PlacementPtr(
        SmemStorage &smem_storage,
        int block_strip = 0,
        int tid = threadIdx.x)
    {
        // Offset for partial
        unsigned int offset = (block_strip * BLOCK_THREADS) + tid;

        // Incorporating a block of padding partials every shared memory segment
        return smem_storage + offset + (offset / PADDING_STRIDE) * PADDING_ELEMENTS;
    }


    /**
     * Pointer for sequential warp-synchronous raking within grid (with padding)
     */
    static __device__ __forceinline__ T* RakingPtr(
        SmemStorage &smem_storage,
        int tid = threadIdx.x)
    {
        unsigned int raking_begin_bytes     = tid * RAKING_LENGTH * sizeof(T);
        unsigned int padding_bytes             = (raking_begin_bytes / (PADDING_STRIDE * sizeof(T))) * PADDING_ELEMENTS * sizeof(T);

        return reinterpret_cast<T*>(
            reinterpret_cast<char*>(smem_storage) +
            raking_begin_bytes +
            padding_bytes);
    }
};



} // namespace cub
CUB_NS_POSTFIX
