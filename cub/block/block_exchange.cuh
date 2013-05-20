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

#include "../util_namespace.cuh"
#include "../util_arch.cuh"
#include "../util_ptx.cuh"
#include "../util_type.cuh"

/// Optional outer namespace(s)
CUB_NS_PREFIX

/// CUB namespace
namespace cub {

/**
 * \addtogroup BlockModule
 * @{
 */

/**
 * \brief BlockExchange provides operations for reorganizing the partitioning of ordered data across a CUDA threadblock. ![](transpose_logo.png)
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
 * \tparam TIME_SLICES          <b>[optional]</b> When > 1, the thread block uses a smaller amount of shared memory that is time-sliced over multiple synchronization rounds to complete the all-to-all exchange (default = 1)
 *
 * \par Algorithm
 * Threads scatter items by item-order into shared memory, allowing one item of padding
 * for every memory bank's worth of items.  After a barrier, items are gathered in the desired arrangement.
 * \image html raking.png
 * <div class="centercaption">A threadblock of 16 threads reading a blocked arrangement of 64 items in a parallel "raking" fashion.</div>
 *
 * \par Usage Considerations
 * - \smemreuse{BlockExchange::SmemStorage}
 *
 * \par Performance Considerations
 * - Proper device-specific padding ensures zero bank conflicts for most types.
 *
 */
template <
    typename        T,
    int             BLOCK_THREADS,
    int             ITEMS_PER_THREAD,
    int             TIME_SLICES = 1>
class BlockExchange
{
    //---------------------------------------------------------------------
    // Type definitions and constants
    //---------------------------------------------------------------------

private:

    enum
    {
        LOG_SMEM_BANKS      = PtxArchProps::LOG_SMEM_BANKS,
        SMEM_BANKS          = 1 << LOG_SMEM_BANKS,

        SLICE_THREADS       = (BLOCK_THREADS + TIME_SLICES - 1) / TIME_SLICES,
        SLICE_ITEMS         = SLICE_THREADS * ITEMS_PER_THREAD,

        // Insert padding if the number of items per thread is a power of two
        PADDING             = ((ITEMS_PER_THREAD & (ITEMS_PER_THREAD - 1)) == 0),
        PADDING_ELEMENTS    = (PADDING) ? (SLICE_ITEMS >> LOG_SMEM_BANKS) : 0,
    };

    /// Shared memory storage layout type
    struct SmemStorage
    {
        T exchange[SLICE_ITEMS + PADDING_ELEMENTS];
    };

public:

    /// \smemstorage{BlockExchange}
    typedef SmemStorage SmemStorage;


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
    static __device__ __forceinline__ void BlockedToStriped(
        SmemStorage     &smem_storage,              ///< [in] Reference to shared memory allocation having layout type SmemStorage
        T               items[ITEMS_PER_THREAD])    ///< [in-out] Items to exchange, converting between <em>blocked</em> and <em>striped</em> arrangements.
    {
        int tid         = (TIME_SLICES == 1) ? threadIdx.x : threadIdx.x % SLICE_THREADS;
        int slice_id    = (TIME_SLICES == 1) ? 0 : threadIdx.x / SLICE_THREADS;

        // First slice
        if (slice_id == 0)
        {
            #pragma unroll
            for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
            {
                int item_offset = (tid * ITEMS_PER_THREAD) + ITEM;
                if (PADDING) item_offset += item_offset >> LOG_SMEM_BANKS;
                smem_storage.exchange[item_offset] = items[ITEM];
            }
        }

        __syncthreads();

        #pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
        {
            int item_offset = int(ITEM * BLOCK_THREADS) + threadIdx.x;
            if ((TIME_SLICES == 1) || (item_offset < SLICE_ITEMS))
            {
                if (PADDING) item_offset += item_offset >> LOG_SMEM_BANKS;
                items[ITEM] = smem_storage.exchange[item_offset];
            }
        }

        // Remaining slices
        #pragma unroll
        for (int SLICE = 1; SLICE < TIME_SLICES; SLICE++)
        {
            __syncthreads();

            if (slice_id == SLICE)
            {
                #pragma unroll
                for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
                {
                    int item_offset = (tid * ITEMS_PER_THREAD) + ITEM;
                    if (PADDING) item_offset += item_offset >> LOG_SMEM_BANKS;
                    smem_storage.exchange[item_offset] = items[ITEM];
                }
            }

            __syncthreads();

            #pragma unroll
            for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
            {
                int item_offset = int(ITEM * BLOCK_THREADS) + threadIdx.x - (SLICE_ITEMS * SLICE);
                if ((item_offset >= 0) && (item_offset < SLICE_ITEMS))
                {
                    if (PADDING) item_offset += item_offset >> LOG_SMEM_BANKS;
                    items[ITEM] = smem_storage.exchange[item_offset];
                }
            }
        }
    }


    /**
     * \brief Transposes data items from <em>blocked</em> arrangement to <em>warp-striped</em> arrangement.
     *
     * \smemreuse
     */
    static __device__ __forceinline__ void BlockedToWarpStriped(
        SmemStorage      &smem_storage,             ///< [in] Reference to shared memory allocation having layout type SmemStorage
        T                items[ITEMS_PER_THREAD])   ///< [in-out] Items to exchange, converting between <em>blocked</em> and <em>warp-striped</em> arrangements.
    {
        int tid         = (TIME_SLICES == 1) ? threadIdx.x : threadIdx.x % SLICE_THREADS;
        int slice_id    = (TIME_SLICES == 1) ? 0 : threadIdx.x / SLICE_THREADS;

        // First slice
        if (slice_id == 0)
        {
            #pragma unroll
            for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
            {
                int item_offset = (tid * ITEMS_PER_THREAD) + ITEM;
                if (PADDING) item_offset += item_offset >> LOG_SMEM_BANKS;
                smem_storage.exchange[item_offset] = items[ITEM];
            }

            // Prevent hoisting
            __threadfence_block();

            #pragma unroll
            for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
            {
                int item_offset = (ITEM * SLICE_THREADS) + tid;
                if (PADDING) item_offset += item_offset >> LOG_SMEM_BANKS;
                items[ITEM] = smem_storage.exchange[item_offset];
            }
        }

        // Continue unrolling slices
        #pragma unroll
        for (int SLICE = 1; SLICE < TIME_SLICES; SLICE++)
        {
            __syncthreads();

            if (slice_id == SLICE)
            {
                #pragma unroll
                for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
                {
                    int item_offset = (tid * ITEMS_PER_THREAD) + ITEM;
                    if (PADDING) item_offset += item_offset >> LOG_SMEM_BANKS;
                    smem_storage.exchange[item_offset] = items[ITEM];
                }

                // Prevent hoisting
                __threadfence_block();

                #pragma unroll
                for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
                {
                    int item_offset = (ITEM * SLICE_THREADS) + tid;
                    if (PADDING) item_offset += item_offset >> LOG_SMEM_BANKS;
                    items[ITEM] = smem_storage.exchange[item_offset];
                }
            }
        }
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
    static __device__ __forceinline__ void StripedToBlocked(
        SmemStorage      &smem_storage,             ///< [in] Reference to shared memory allocation having layout type SmemStorage
        T                items[ITEMS_PER_THREAD])   ///< [in-out] Items to exchange, converting between <em>striped</em> and <em>blocked</em> arrangements.
    {
        int tid         = (TIME_SLICES == 1) ? threadIdx.x : threadIdx.x % SLICE_THREADS;
        int slice_id    = (TIME_SLICES == 1) ? 0 : threadIdx.x / SLICE_THREADS;

        // First slice
        #pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
        {
            int item_offset = int(ITEM * BLOCK_THREADS) + threadIdx.x;
            if ((TIME_SLICES == 1) || (item_offset < SLICE_ITEMS))
            {
                if (PADDING) item_offset = SHR_ADD(item_offset, LOG_SMEM_BANKS, item_offset);
                smem_storage.exchange[item_offset] = items[ITEM];
            }
        }

        __syncthreads();

        if (slice_id == 0)
        {
            #pragma unroll
            for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
            {
                int item_offset = (tid * ITEMS_PER_THREAD) + ITEM;
                if (PADDING) item_offset = SHR_ADD(item_offset, LOG_SMEM_BANKS, item_offset);
                items[ITEM] = smem_storage.exchange[item_offset];
            }
        }

        // Remaining slices
        #pragma unroll
        for (int SLICE = 1; SLICE < TIME_SLICES; SLICE++)
        {
            __syncthreads();

            #pragma unroll
            for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
            {
                int item_offset = int(ITEM * BLOCK_THREADS) + threadIdx.x - (SLICE_ITEMS * SLICE);
                if ((item_offset >= 0) && (item_offset < SLICE_ITEMS))
                {
                    if (PADDING) item_offset = SHR_ADD(item_offset, LOG_SMEM_BANKS, item_offset);
                    smem_storage.exchange[item_offset] = items[ITEM];
                }
            }

            __syncthreads();

            if (slice_id == SLICE)
            {
                #pragma unroll
                for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
                {
                    int item_offset = (tid * ITEMS_PER_THREAD) + ITEM;
                    if (PADDING) item_offset = SHR_ADD(item_offset, LOG_SMEM_BANKS, item_offset);
                    items[ITEM] = smem_storage.exchange[item_offset];
                }
            }
        }
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
    static __device__ __forceinline__ void WarpStripedToBlocked(
        SmemStorage      &smem_storage,             ///< [in] Reference to shared memory allocation having layout type SmemStorage
        T                items[ITEMS_PER_THREAD])   ///< [in-out] Items to exchange, converting between <em>warp-striped</em> and <em>blocked</em> arrangements.
    {
        int tid         = (TIME_SLICES == 1) ? threadIdx.x : threadIdx.x % SLICE_THREADS;
        int slice_id    = (TIME_SLICES == 1) ? 0 : threadIdx.x / SLICE_THREADS;

        // First slice
        if (slice_id == 0)
        {
            #pragma unroll
            for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
            {
                int item_offset = (ITEM * SLICE_THREADS) + tid;
                if (PADDING) item_offset = SHR_ADD(item_offset, LOG_SMEM_BANKS, item_offset);
                smem_storage.exchange[item_offset] = items[ITEM];
            }

            // Prevent hoisting
            __threadfence_block();

            #pragma unroll
            for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
            {
                int item_offset = (tid * ITEMS_PER_THREAD) + ITEM;
                if (PADDING) item_offset = SHR_ADD(item_offset, LOG_SMEM_BANKS, item_offset);
                items[ITEM] = smem_storage.exchange[item_offset];
            }
        }

        // Continue unrolling slices
        #pragma unroll
        for (int SLICE = 1; SLICE < TIME_SLICES; SLICE++)
        {
            __syncthreads();

            if (slice_id == SLICE)
            {
                #pragma unroll
                for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
                {
                    int item_offset = (ITEM * SLICE_THREADS) + tid;
                    if (PADDING) item_offset = SHR_ADD(item_offset, LOG_SMEM_BANKS, item_offset);
                    smem_storage.exchange[item_offset] = items[ITEM];
                }

                // Prevent hoisting
                __threadfence_block();

                #pragma unroll
                for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
                {
                    int item_offset = (tid * ITEMS_PER_THREAD) + ITEM;
                    if (PADDING) item_offset = SHR_ADD(item_offset, LOG_SMEM_BANKS, item_offset);
                    items[ITEM] = smem_storage.exchange[item_offset];
                }
            }
        }
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
    static __device__ __forceinline__ void ScatterToBlocked(
        SmemStorage     &smem_storage,              ///< [in] Reference to shared memory allocation having layout type SmemStorage
        T               items[ITEMS_PER_THREAD],    ///< [in-out] Items to exchange
        int             ranks[ITEMS_PER_THREAD])    ///< [in] Corresponding scatter ranks
    {
        int tid         = (TIME_SLICES == 1) ? threadIdx.x : threadIdx.x % SLICE_THREADS;
        int slice_id    = (TIME_SLICES == 1) ? 0 : threadIdx.x / SLICE_THREADS;

        // First slice
        #pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
        {
            int item_offset = ranks[ITEM];
            if ((TIME_SLICES == 1) || (item_offset < SLICE_ITEMS))
            {
                if (PADDING) item_offset = SHR_ADD(item_offset, LOG_SMEM_BANKS, item_offset);
                smem_storage.exchange[item_offset] = items[ITEM];
            }
        }

        __syncthreads();

        if (slice_id == 0)
        {
            #pragma unroll
            for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
            {
                int item_offset = (tid * ITEMS_PER_THREAD) + ITEM;
                if (PADDING) item_offset = SHR_ADD(item_offset, LOG_SMEM_BANKS, item_offset);
                items[ITEM] = smem_storage.exchange[item_offset];
            }
        }

        // Remaining slices
        #pragma unroll
        for (int SLICE = 1; SLICE < TIME_SLICES; SLICE++)
        {
            __syncthreads();

            #pragma unroll
            for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
            {
                int item_offset = ranks[ITEM] - (SLICE_ITEMS * SLICE);
                if ((item_offset >= 0) && (item_offset < SLICE_ITEMS))
                {
                    if (PADDING) item_offset = SHR_ADD(item_offset, LOG_SMEM_BANKS, item_offset);
                    smem_storage.exchange[item_offset] = items[ITEM];
                }
            }

            __syncthreads();

            if (slice_id == SLICE)
            {
                #pragma unroll
                for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
                {
                    int item_offset = (tid * ITEMS_PER_THREAD) + ITEM;
                    if (PADDING) item_offset = SHR_ADD(item_offset, LOG_SMEM_BANKS, item_offset);
                    items[ITEM] = smem_storage.exchange[item_offset];
                }
            }
        }
    }


    /**
     * \brief Exchanges data items annotated by rank into <em>striped</em> arrangement.
     *
     * \smemreuse
     */
    static __device__ __forceinline__ void ScatterToStriped(
        SmemStorage     &smem_storage,              ///< [in] Reference to shared memory allocation having layout type SmemStorage
        T               items[ITEMS_PER_THREAD],    ///< [in-out] Items to exchange
        int             ranks[ITEMS_PER_THREAD])    ///< [in] Corresponding scatter ranks
    {
        int tid         = (TIME_SLICES == 1) ? threadIdx.x : threadIdx.x % SLICE_THREADS;
        int slice_id    = (TIME_SLICES == 1) ? 0 : threadIdx.x / SLICE_THREADS;

        // First slice
        #pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
        {
            int item_offset = ranks[ITEM];
            if ((TIME_SLICES == 1) || (item_offset < SLICE_ITEMS))
            {
                if (PADDING) item_offset = SHR_ADD(item_offset, LOG_SMEM_BANKS, item_offset);
                smem_storage.exchange[item_offset] = items[ITEM];
            }
        }

        __syncthreads();

        #pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
        {
            int item_offset = int(ITEM * BLOCK_THREADS) + threadIdx.x;
            if ((TIME_SLICES == 1) || (item_offset < SLICE_ITEMS))
            {
                if (PADDING) item_offset = SHR_ADD(item_offset, LOG_SMEM_BANKS, item_offset);
                items[ITEM] = smem_storage.exchange[item_offset];
            }
        }

        // Remaining slices
        #pragma unroll
        for (int SLICE = 1; SLICE < TIME_SLICES; SLICE++)
        {
            __syncthreads();

            #pragma unroll
            for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
            {
                int item_offset = ranks[ITEM] - (SLICE_ITEMS * SLICE);
                if ((item_offset >= 0) && (item_offset < SLICE_ITEMS))
                {
                    if (PADDING) item_offset = SHR_ADD(item_offset, LOG_SMEM_BANKS, item_offset);
                    smem_storage.exchange[item_offset] = items[ITEM];
                }
            }

            __syncthreads();

            #pragma unroll
            for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
            {
                int item_offset = int(ITEM * BLOCK_THREADS) + threadIdx.x - (SLICE_ITEMS * SLICE);
                if ((item_offset >= 0) && (item_offset < SLICE_ITEMS))
                {
                    if (PADDING) item_offset = SHR_ADD(item_offset, LOG_SMEM_BANKS, item_offset);
                    items[ITEM] = smem_storage.exchange[item_offset];
                }
            }
        }
    }

    //@}  end member group


};

/** @} */       // end group BlockModule

}               // CUB namespace
CUB_NS_POSTFIX  // Optional outer namespace(s)

