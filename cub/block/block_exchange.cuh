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
 * cub::BlockExchange provides operations for reorganizing the partitioning of ordered data across a CUDA threadblock.
 */

#pragma once

#include "../util_debug.cuh"
#include "../util_arch.cuh"
#include "../util_macro.cuh"
#include "../util_ptx.cuh"
#include "../util_type.cuh"
#include "../util_namespace.cuh"

/// Optional outer namespace(s)
CUB_NS_PREFIX

/// CUB namespace
namespace cub {

/**
 * \brief BlockExchange provides operations for reorganizing the partitioning of ordered data across a CUDA threadblock. ![](transpose_logo.png)
 * \ingroup BlockModule
 *
 * \par Overview
 * BlockExchange allows threadblocks to reorganize data items between
 * threads. More specifically, BlockExchange supports the following types of data
 * exchanges:
 * - Transposing between [<em>blocked</em>](index.html#sec3sec3) and [<em>striped</em>](index.html#sec3sec3) arrangements
 * - Transposing between [<em>blocked</em>](index.html#sec3sec3) and [<em>warp-striped</em>](index.html#sec3sec3) arrangements
 * - Scattering to a [<em>blocked arrangement</em>](index.html#sec3sec3)
 * - Scattering to a [<em>striped arrangement</em>](index.html#sec3sec3)
 *
 * \tparam T                    The data type to be exchanged.
 * \tparam BLOCK_THREADS        The threadblock size in threads.
 * \tparam ITEMS_PER_THREAD     The number of items partitioned onto each thread.
 * \tparam WARP_TIME_SLICING    <b>[optional]</b> When \p true, only use enough shared memory for a single warp's worth of tile data, time-slicing the block-wide exchange over multiple synchronized rounds (default = false)
 *
 * \par Algorithm
 * Threads scatter items by item-order into shared memory, allowing one item of padding
 * for every memory bank's worth of items.  After a barrier, items are gathered in the desired arrangement.
 * \image html raking.png
 * <div class="centercaption">A threadblock of 16 threads reading a blocked arrangement of 64 items in a parallel "raking" fashion.</div>
 *
 * \par Usage Considerations
 * - \smemreuse{BlockExchange::TempStorage}
 *
 * \par Performance Considerations
 * - Proper device-specific padding ensures zero bank conflicts for most types.
 *
 */
template <
    typename        T,
    int             BLOCK_THREADS,
    int             ITEMS_PER_THREAD,
    bool            WARP_TIME_SLICING = false>
class BlockExchange
{
private:

    /******************************************************************************
     * Constants
     ******************************************************************************/

    enum
    {
        LOG_WARP_THREADS            = PtxArchProps::LOG_WARP_THREADS,
        WARP_THREADS                = 1 << LOG_WARP_THREADS,
        WARPS                       = (BLOCK_THREADS + PtxArchProps::WARP_THREADS - 1) / PtxArchProps::WARP_THREADS,

        LOG_SMEM_BANKS              = PtxArchProps::LOG_SMEM_BANKS,
        SMEM_BANKS                  = 1 << LOG_SMEM_BANKS,

        TILE_ITEMS                  = BLOCK_THREADS * ITEMS_PER_THREAD,

        TIME_SLICES                 = (WARP_TIME_SLICING) ? WARPS : 1,

        TIME_SLICED_THREADS         = (WARP_TIME_SLICING) ? CUB_MIN(BLOCK_THREADS, WARP_THREADS) : BLOCK_THREADS,
        TIME_SLICED_ITEMS           = TIME_SLICED_THREADS * ITEMS_PER_THREAD,

        WARP_TIME_SLICED_THREADS    = CUB_MIN(BLOCK_THREADS, WARP_THREADS),
        WARP_TIME_SLICED_ITEMS      = WARP_TIME_SLICED_THREADS * ITEMS_PER_THREAD,

        // Insert padding if the number of items per thread is a power of two
        INSERT_PADDING              = ((ITEMS_PER_THREAD & (ITEMS_PER_THREAD - 1)) == 0),
        PADDING_ITEMS               = (INSERT_PADDING) ? (TIME_SLICED_ITEMS >> LOG_SMEM_BANKS) : 0,
    };

    /******************************************************************************
     * Type definitions
     ******************************************************************************/

    /// Shared memory storage layout type
    typedef T _TempStorage[TIME_SLICED_ITEMS + PADDING_ITEMS];

public:

    /// \smemstorage{BlockExchange}
    typedef _TempStorage TempStorage;

private:


    /******************************************************************************
     * Thread fields
     ******************************************************************************/

    /// Shared storage reference
    _TempStorage &temp_storage;

    /// Linear thread-id
    int linear_tid;


    /******************************************************************************
     * Utility methods
     ******************************************************************************/

    /// Internal storage allocator
    __device__ __forceinline__ _TempStorage& PrivateStorage()
    {
        __shared__ _TempStorage private_storage;
        return private_storage;
    }


    /**
     * Transposes data items from <em>blocked</em> arrangement to <em>striped</em> arrangement.  Specialized for no timeslicing.
     */
    __device__ __forceinline__ void BlockedToStriped(
        T               items[ITEMS_PER_THREAD],    ///< [in-out] Items to exchange, converting between <em>blocked</em> and <em>striped</em> arrangements.
        Int2Type<false> time_slicing)
    {
        #pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
        {
            int item_offset = (threadIdx.x * ITEMS_PER_THREAD) + ITEM;
            if (INSERT_PADDING) item_offset += item_offset >> LOG_SMEM_BANKS;
            temp_storage[item_offset] = items[ITEM];
        }

        __syncthreads();

        #pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
        {
            int item_offset = int(ITEM * BLOCK_THREADS) + threadIdx.x;
            if (INSERT_PADDING) item_offset += item_offset >> LOG_SMEM_BANKS;
            items[ITEM] = temp_storage[item_offset];
        }
    }


    /**
     * Transposes data items from <em>blocked</em> arrangement to <em>striped</em> arrangement.  Specialized for warp-timeslicing.
     */
    __device__ __forceinline__ void BlockedToStriped(
        T               items[ITEMS_PER_THREAD],    ///< [in-out] Items to exchange, converting between <em>blocked</em> and <em>striped</em> arrangements.
        Int2Type<true>  time_slicing)
    {
        T temp_items[ITEMS_PER_THREAD];

        int warp_lane   = threadIdx.x & (WARP_THREADS - 1);
        int warp_id     = threadIdx.x >> LOG_WARP_THREADS;

        #pragma unroll
        for (int SLICE = 0; SLICE < TIME_SLICES; SLICE++)
        {
            const int SLICE_OFFSET  = SLICE * TIME_SLICED_ITEMS;
            const int SLICE_OOB     = SLICE_OFFSET + TIME_SLICED_ITEMS;

            __syncthreads();

            if (warp_id == SLICE)
            {
                #pragma unroll
                for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
                {
                    int item_offset = (warp_lane * ITEMS_PER_THREAD) + ITEM;
                    if (INSERT_PADDING) item_offset += item_offset >> LOG_SMEM_BANKS;
                    temp_storage[item_offset] = items[ITEM];
                }
            }

            __syncthreads();

            #pragma unroll
            for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
            {
                // Read a strip of items
                const int STRIP_OFFSET  = ITEM * BLOCK_THREADS;
                const int STRIP_OOB     = STRIP_OFFSET + BLOCK_THREADS;

                if ((SLICE_OFFSET < STRIP_OOB) && (SLICE_OOB > STRIP_OFFSET))
                {
                    int item_offset = STRIP_OFFSET + threadIdx.x - SLICE_OFFSET;
                    if ((item_offset >= 0) && (item_offset < TIME_SLICED_ITEMS))
                    {
                        if (INSERT_PADDING) item_offset += item_offset >> LOG_SMEM_BANKS;
                        temp_items[ITEM] = temp_storage[item_offset];
                    }
                }
            }
        }

        // Copy
        #pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
        {
            items[ITEM] = temp_items[ITEM];
        }
    }


    /**
     * Transposes data items from <em>blocked</em> arrangement to <em>warp-striped</em> arrangement. Specialized for no timeslicing
     */
    __device__ __forceinline__ void BlockedToWarpStriped(
        T               items[ITEMS_PER_THREAD],   ///< [in-out] Items to exchange, converting between <em>blocked</em> and <em>warp-striped</em> arrangements.
        Int2Type<false> time_slicing)
    {
        int warp_lane   = threadIdx.x & (WARP_THREADS - 1);
        int warp_id     = threadIdx.x >> LOG_WARP_THREADS;
        int warp_offset = warp_id * WARP_TIME_SLICED_ITEMS;

        #pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
        {
            int item_offset = warp_offset + ITEM + (warp_lane * ITEMS_PER_THREAD);
            if (INSERT_PADDING) item_offset += item_offset >> LOG_SMEM_BANKS;
            temp_storage[item_offset] = items[ITEM];
        }

        // Prevent hoisting
        __threadfence_block();

        #pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
        {
            int item_offset = warp_offset + (ITEM * WARP_TIME_SLICED_THREADS) + warp_lane;
            if (INSERT_PADDING) item_offset += item_offset >> LOG_SMEM_BANKS;
            items[ITEM] = temp_storage[item_offset];
        }
    }

    /**
     * Transposes data items from <em>blocked</em> arrangement to <em>warp-striped</em> arrangement. Specialized for warp-timeslicing
     */
    __device__ __forceinline__ void BlockedToWarpStriped(
        T               items[ITEMS_PER_THREAD],   ///< [in-out] Items to exchange, converting between <em>blocked</em> and <em>warp-striped</em> arrangements.
        Int2Type<true>  time_slicing)
    {
        int warp_lane   = threadIdx.x & (WARP_THREADS - 1);
        int warp_id     = threadIdx.x >> LOG_WARP_THREADS;

        #pragma unroll
        for (int SLICE = 0; SLICE < TIME_SLICES; ++SLICE)
        {
            __syncthreads();

            if (warp_id == SLICE)
            {
                #pragma unroll
                for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
                {
                    int item_offset = ITEM + (warp_lane * ITEMS_PER_THREAD);
                    if (INSERT_PADDING) item_offset += item_offset >> LOG_SMEM_BANKS;
                    temp_storage[item_offset] = items[ITEM];
                }

                // Prevent hoisting
                __threadfence_block();

                #pragma unroll
                for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
                {
                    int item_offset = (ITEM * WARP_TIME_SLICED_THREADS) + warp_lane;
                    if (INSERT_PADDING) item_offset += item_offset >> LOG_SMEM_BANKS;
                    items[ITEM] = temp_storage[item_offset];
                }
            }
        }
    }


    /**
     * Transposes data items from <em>striped</em> arrangement to <em>blocked</em> arrangement.  Specialized for no timeslicing.
     */
    __device__ __forceinline__ void StripedToBlocked(
        T               items[ITEMS_PER_THREAD],   ///< [in-out] Items to exchange, converting between <em>striped</em> and <em>blocked</em> arrangements.
        Int2Type<false> time_slicing)
    {
        #pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
        {
            int item_offset = int(ITEM * BLOCK_THREADS) + threadIdx.x;
            if (INSERT_PADDING) item_offset += item_offset >> LOG_SMEM_BANKS;
            temp_storage[item_offset] = items[ITEM];
        }

        __syncthreads();

        // No timeslicing
        #pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
        {
            int item_offset = (threadIdx.x * ITEMS_PER_THREAD) + ITEM;
            if (INSERT_PADDING) item_offset += item_offset >> LOG_SMEM_BANKS;
            items[ITEM] = temp_storage[item_offset];
        }
    }


    /**
     * Transposes data items from <em>striped</em> arrangement to <em>blocked</em> arrangement.  Specialized for warp-timeslicing.
     */
    __device__ __forceinline__ void StripedToBlocked(
        T               items[ITEMS_PER_THREAD],   ///< [in-out] Items to exchange, converting between <em>striped</em> and <em>blocked</em> arrangements.
        Int2Type<true>  time_slicing)
    {
        // Warp time-slicing
        T temp_items[ITEMS_PER_THREAD];

        int warp_lane   = threadIdx.x & (WARP_THREADS - 1);
        int warp_id     = threadIdx.x >> LOG_WARP_THREADS;

        #pragma unroll
        for (int SLICE = 0; SLICE < TIME_SLICES; SLICE++)
        {
            const int SLICE_OFFSET  = SLICE * TIME_SLICED_ITEMS;
            const int SLICE_OOB     = SLICE_OFFSET + TIME_SLICED_ITEMS;

            __syncthreads();

            #pragma unroll
            for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
            {
                // Write a strip of items
                const int STRIP_OFFSET  = ITEM * BLOCK_THREADS;
                const int STRIP_OOB     = STRIP_OFFSET + BLOCK_THREADS;

                if ((SLICE_OFFSET < STRIP_OOB) && (SLICE_OOB > STRIP_OFFSET))
                {
                    int item_offset = STRIP_OFFSET + threadIdx.x - SLICE_OFFSET;
                    if ((item_offset >= 0) && (item_offset < TIME_SLICED_ITEMS))
                    {
                        if (INSERT_PADDING) item_offset += item_offset >> LOG_SMEM_BANKS;
                        temp_storage[item_offset] = items[ITEM];
                    }
                }
            }

            __syncthreads();

            if (warp_id == SLICE)
            {
                #pragma unroll
                for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
                {
                    int item_offset = (warp_lane * ITEMS_PER_THREAD) + ITEM;
                    if (INSERT_PADDING) item_offset += item_offset >> LOG_SMEM_BANKS;
                    temp_items[ITEM] = temp_storage[item_offset];
                }
            }
        }

        // Copy
        #pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
        {
            items[ITEM] = temp_items[ITEM];
        }
    }


    /**
     * Transposes data items from <em>warp-striped</em> arrangement to <em>blocked</em> arrangement.  Specialized for no timeslicing
     */
    __device__ __forceinline__ void WarpStripedToBlocked(
        T               items[ITEMS_PER_THREAD],   ///< [in-out] Items to exchange, converting between <em>warp-striped</em> and <em>blocked</em> arrangements.
        Int2Type<false> time_slicing)
    {
        int warp_lane   = threadIdx.x & (WARP_THREADS - 1);
        int warp_id     = threadIdx.x >> LOG_WARP_THREADS;
        int warp_offset = warp_id * WARP_TIME_SLICED_ITEMS;

        #pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
        {
            int item_offset = warp_offset + (ITEM * WARP_TIME_SLICED_THREADS) + warp_lane;
            if (INSERT_PADDING) item_offset += item_offset >> LOG_SMEM_BANKS;
            temp_storage[item_offset] = items[ITEM];
        }

        // Prevent hoisting
        __threadfence_block();

        #pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
        {
            int item_offset = warp_offset + ITEM + (warp_lane * ITEMS_PER_THREAD);
            if (INSERT_PADDING) item_offset += item_offset >> LOG_SMEM_BANKS;
            items[ITEM] = temp_storage[item_offset];
        }
    }


    /**
     * Transposes data items from <em>warp-striped</em> arrangement to <em>blocked</em> arrangement.  Specialized for warp-timeslicing
     */
    __device__ __forceinline__ void WarpStripedToBlocked(
        T               items[ITEMS_PER_THREAD],   ///< [in-out] Items to exchange, converting between <em>warp-striped</em> and <em>blocked</em> arrangements.
        Int2Type<true>  time_slicing)
    {
        int warp_lane   = threadIdx.x & (WARP_THREADS - 1);
        int warp_id     = threadIdx.x >> LOG_WARP_THREADS;

        #pragma unroll
        for (int SLICE = 0; SLICE < TIME_SLICES; ++SLICE)
        {
            __syncthreads();

            if (warp_id == SLICE)
            {
                #pragma unroll
                for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
                {
                    int item_offset = (ITEM * WARP_TIME_SLICED_THREADS) + warp_lane;
                    if (INSERT_PADDING) item_offset += item_offset >> LOG_SMEM_BANKS;
                    temp_storage[item_offset] = items[ITEM];
                }

                // Prevent hoisting
                __threadfence_block();

                #pragma unroll
                for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
                {
                    int item_offset = ITEM + (warp_lane * ITEMS_PER_THREAD);
                    if (INSERT_PADDING) item_offset += item_offset >> LOG_SMEM_BANKS;
                    items[ITEM] = temp_storage[item_offset];
                }
            }
        }
    }


    /**
     * Exchanges data items annotated by rank into <em>blocked</em> arrangement.  Specialized for no timeslicing.
     */
    __device__ __forceinline__ void ScatterToBlocked(
        T               items[ITEMS_PER_THREAD],    ///< [in-out] Items to exchange
        int             ranks[ITEMS_PER_THREAD],    ///< [in] Corresponding scatter ranks
        Int2Type<false> time_slicing)
    {
        #pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
        {
            int item_offset = ranks[ITEM];
            if (INSERT_PADDING) item_offset = SHR_ADD(item_offset, LOG_SMEM_BANKS, item_offset);
            temp_storage[item_offset] = items[ITEM];
        }

        __syncthreads();

        #pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
        {
            int item_offset = (threadIdx.x * ITEMS_PER_THREAD) + ITEM;
            if (INSERT_PADDING) item_offset = SHR_ADD(item_offset, LOG_SMEM_BANKS, item_offset);
            items[ITEM] = temp_storage[item_offset];
        }
    }

    /**
     * Exchanges data items annotated by rank into <em>blocked</em> arrangement.  Specialized for warp-timeslicing.
     */
    __device__ __forceinline__ void ScatterToBlocked(
        T               items[ITEMS_PER_THREAD],    ///< [in-out] Items to exchange
        int             ranks[ITEMS_PER_THREAD],    ///< [in] Corresponding scatter ranks
        Int2Type<true>  time_slicing)
    {
        int warp_lane   = threadIdx.x & (WARP_THREADS - 1);
        int warp_id     = threadIdx.x >> LOG_WARP_THREADS;

        T temp_items[ITEMS_PER_THREAD];

        #pragma unroll
        for (int SLICE = 0; SLICE < TIME_SLICES; SLICE++)
        {
            __syncthreads();

            const int SLICE_OFFSET = TIME_SLICED_ITEMS * SLICE;

            #pragma unroll
            for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
            {
                int item_offset = ranks[ITEM] - SLICE_OFFSET;
                if ((item_offset >= 0) && (item_offset < WARP_TIME_SLICED_ITEMS))
                {
                    if (INSERT_PADDING) item_offset = SHR_ADD(item_offset, LOG_SMEM_BANKS, item_offset);
                    temp_storage[item_offset] = items[ITEM];
                }
            }

            __syncthreads();

            if (warp_id == SLICE)
            {
                #pragma unroll
                for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
                {
                    int item_offset = (warp_lane * ITEMS_PER_THREAD) + ITEM;
                    if (INSERT_PADDING) item_offset = SHR_ADD(item_offset, LOG_SMEM_BANKS, item_offset);
                    temp_items[ITEM] = temp_storage[item_offset];
                }
            }
        }

        // Copy
        #pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
        {
            items[ITEM] = temp_items[ITEM];
        }
    }


    /**
     * Exchanges data items annotated by rank into <em>striped</em> arrangement.  Specialized for no timeslicing.
     */
    __device__ __forceinline__ void ScatterToStriped(
        T               items[ITEMS_PER_THREAD],    ///< [in-out] Items to exchange
        int             ranks[ITEMS_PER_THREAD],    ///< [in] Corresponding scatter ranks
        Int2Type<false> time_slicing)
    {
        #pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
        {
            int item_offset = ranks[ITEM];
            if (INSERT_PADDING) item_offset = SHR_ADD(item_offset, LOG_SMEM_BANKS, item_offset);
            temp_storage[item_offset] = items[ITEM];
        }

        __syncthreads();

        #pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
        {
            int item_offset = int(ITEM * BLOCK_THREADS) + threadIdx.x;
            if (INSERT_PADDING) item_offset = SHR_ADD(item_offset, LOG_SMEM_BANKS, item_offset);
            items[ITEM] = temp_storage[item_offset];
        }
    }


    /**
     * Exchanges data items annotated by rank into <em>striped</em> arrangement.  Specialized for warp-timeslicing.
     */
    __device__ __forceinline__ void ScatterToStriped(
        T               items[ITEMS_PER_THREAD],    ///< [in-out] Items to exchange
        int             ranks[ITEMS_PER_THREAD],    ///< [in] Corresponding scatter ranks
        Int2Type<true> time_slicing)
    {
        T temp_items[ITEMS_PER_THREAD];

        #pragma unroll
        for (int SLICE = 0; SLICE < TIME_SLICES; SLICE++)
        {
            const int SLICE_OFFSET  = SLICE * TIME_SLICED_ITEMS;
            const int SLICE_OOB     = SLICE_OFFSET + TIME_SLICED_ITEMS;

            __syncthreads();

            #pragma unroll
            for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
            {
                int item_offset = ranks[ITEM] - SLICE_OFFSET;
                if ((item_offset >= 0) && (item_offset < WARP_TIME_SLICED_ITEMS))
                {
                    if (INSERT_PADDING) item_offset = SHR_ADD(item_offset, LOG_SMEM_BANKS, item_offset);
                    temp_storage[item_offset] = items[ITEM];
                }
            }

            __syncthreads();

            #pragma unroll
            for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
            {
                // Read a strip of items
                const int STRIP_OFFSET  = ITEM * BLOCK_THREADS;
                const int STRIP_OOB     = STRIP_OFFSET + BLOCK_THREADS;

                if ((SLICE_OFFSET < STRIP_OOB) && (SLICE_OOB > STRIP_OFFSET))
                {
                    int item_offset = STRIP_OFFSET + threadIdx.x - SLICE_OFFSET;
                    if ((item_offset >= 0) && (item_offset < TIME_SLICED_ITEMS))
                    {
                        if (INSERT_PADDING) item_offset += item_offset >> LOG_SMEM_BANKS;
                        temp_items[ITEM] = temp_storage[item_offset];
                    }
                }
            }
        }

        // Copy
        #pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
        {
            items[ITEM] = temp_items[ITEM];
        }
    }


public:

    /******************************************************************//**
     * \name Collective construction
     *********************************************************************/
    //@{

    /**
     * \brief Collective constructor for 1D thread blocks using a private static allocation of shared memory as temporary storage.  Threads are identified using <tt>threadIdx.x</tt>.
     */
    __device__ __forceinline__ BlockExchange()
    :
        temp_storage(PrivateStorage()),
        linear_tid(threadIdx.x)
    {}


    /**
     * \brief Collective constructor for 1D thread blocks using the specified memory allocation as temporary storage.  Threads are identified using <tt>threadIdx.x</tt>.
     */
    __device__ __forceinline__ BlockExchange(
        TempStorage &temp_storage)             ///< [in] Reference to memory allocation having layout type TempStorage
    :
        temp_storage(temp_storage),
        linear_tid(threadIdx.x)
    {}


    /**
     * \brief Collective constructor using a private static allocation of shared memory as temporary storage.  Threads are identified using the given linear thread identifier
     */
    __device__ __forceinline__ BlockExchange(
        int linear_tid)                        ///< [in] A suitable 1D thread-identifier for the calling thread (e.g., <tt>(threadIdx.y * blockDim.x) + linear_tid</tt> for 2D thread blocks)
    :
        temp_storage(PrivateStorage()),
        linear_tid(linear_tid)
    {}


    /**
     * \brief Collective constructor using the specified memory allocation as temporary storage.  Threads are identified using the given linear thread identifier.
     */
    __device__ __forceinline__ BlockExchange(
        TempStorage &temp_storage,              ///< [in] Reference to memory allocation having layout type TempStorage
        int         linear_tid)                 ///< [in] <b>[optional]</b> A suitable 1D thread-identifier for the calling thread (e.g., <tt>(threadIdx.y * blockDim.x) + linear_tid</tt> for 2D thread blocks)
    :
        temp_storage(temp_storage),
        linear_tid(linear_tid)
    {}


    //@}  end member group
    /******************************************************************//**
     * \name Blocked exchanges
     *********************************************************************/
    //@{

    /**
     * \brief Transposes data items from <em>blocked</em> arrangement to <em>striped</em> arrangement.
     *
     * \smemreuse
     */
    __device__ __forceinline__ void BlockedToStriped(
        T               items[ITEMS_PER_THREAD])    ///< [in-out] Items to exchange, converting between <em>blocked</em> and <em>striped</em> arrangements.
    {
        BlockedToStriped(items, Int2Type<WARP_TIME_SLICING>());
    }


    /**
     * \brief Transposes data items from <em>blocked</em> arrangement to <em>warp-striped</em> arrangement.
     *
     * \smemreuse
     */
    __device__ __forceinline__ void BlockedToWarpStriped(
        T                items[ITEMS_PER_THREAD])   ///< [in-out] Items to exchange, converting between <em>blocked</em> and <em>warp-striped</em> arrangements.
    {
        BlockedToWarpStriped(items, Int2Type<WARP_TIME_SLICING>());
    }


    //@}  end member group
    /******************************************************************//**
     * \name Striped exchanges
     *********************************************************************/
    //@{


    /**
     * \brief Transposes data items from <em>striped</em> arrangement to <em>blocked</em> arrangement.
     *
     * \smemreuse
     */
    __device__ __forceinline__ void StripedToBlocked(
        T                items[ITEMS_PER_THREAD])   ///< [in-out] Items to exchange, converting between <em>striped</em> and <em>blocked</em> arrangements.
    {
        StripedToBlocked(items, Int2Type<WARP_TIME_SLICING>());
    }


    //@}  end member group
    /******************************************************************//**
     * \name Warp-striped exchanges
     *********************************************************************/
    //@{


    /**
     * \brief Transposes data items from <em>warp-striped</em> arrangement to <em>blocked</em> arrangement.
     *
     * \smemreuse
     */
    __device__ __forceinline__ void WarpStripedToBlocked(
        T                items[ITEMS_PER_THREAD])   ///< [in-out] Items to exchange, converting between <em>warp-striped</em> and <em>blocked</em> arrangements.
    {
        WarpStripedToBlocked(items, Int2Type<WARP_TIME_SLICING>());
    }


    //@}  end member group
    /******************************************************************//**
     * \name Scatter exchanges
     *********************************************************************/
    //@{


    /**
     * \brief Exchanges data items annotated by rank into <em>blocked</em> arrangement.
     *
     * \smemreuse
     */
    __device__ __forceinline__ void ScatterToBlocked(
        T               items[ITEMS_PER_THREAD],    ///< [in-out] Items to exchange
        int             ranks[ITEMS_PER_THREAD])    ///< [in] Corresponding scatter ranks
    {
        ScatterToBlocked(items, ranks, Int2Type<WARP_TIME_SLICING>());
    }


    /**
     * \brief Exchanges data items annotated by rank into <em>striped</em> arrangement.
     *
     * \smemreuse
     */
    __device__ __forceinline__ void ScatterToStriped(
        T               items[ITEMS_PER_THREAD],    ///< [in-out] Items to exchange
        int             ranks[ITEMS_PER_THREAD])    ///< [in] Corresponding scatter ranks
    {
        ScatterToStriped(items, ranks, Int2Type<WARP_TIME_SLICING>());
    }

    //@}  end member group


};

}               // CUB namespace
CUB_NS_POSTFIX  // Optional outer namespace(s)

