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
 * \tparam WARP_TIME_SLICING          <b>[optional]</b> The number of communication rounds needed to complete the all-to-all exchange; more rounds can be traded for a smaller shared memory footprint (default = 1)
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
    bool            WARP_TIME_SLICING = false>          // if true, the number of items per thread must be greater than or equal to the number of warps
class BlockExchange
{
    //---------------------------------------------------------------------
    // Type definitions and constants
    //---------------------------------------------------------------------

private:



    enum
    {
        LOG_SMEM_BANKS              = PtxArchProps::LOG_SMEM_BANKS,
        SMEM_BANKS                  = 1 << LOG_SMEM_BANKS,
        TILE_ITEMS                  = BLOCK_THREADS * ITEMS_PER_THREAD,

        WARPS                       = (BLOCK_THREADS + PtxArchProps::WARP_THREADS - 1) / PtxArchProps::WARP_THREADS,
        TIME_SLICED_THREADS         = (WARP_TIME_SLICING && (BLOCK_THREADS > PtxArchProps::WARP_THREADS)) ? PtxArchProps::WARP_THREADS : BLOCK_THREADS,
        TIME_SLICED_ITEMS           = TIME_SLICED_THREADS * ITEMS_PER_THREAD,
        TIME_SLICES                 = (BLOCK_THREADS + TIME_SLICED_THREADS - 1) / TIME_SLICED_THREADS,

        // Insert padding if the number of items per thread is a power of two
        PADDING                     = ((ITEMS_PER_THREAD & (ITEMS_PER_THREAD - 1)) == 0),
    };

    /// Shared memory storage layout type
    union SmemStorage
    {
        T exchange[TIME_SLICED_ITEMS + (PADDING) ? (TIME_SLICED_ITEMS >> LOG_SMEM_BANKS) : 0];
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
        if (WARP_TIME_SLICING == 0)
        {
            // No timeslicing
            #pragma unroll
            for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
            {
                int item_offset = (threadIdx.x * ITEMS_PER_THREAD) + ITEM;
                if (PADDING) item_offset += item_offset >> LOG_SMEM_BANKS;
                smem_storage.exchange[item_offset] = items[ITEM];
            }

            __syncthreads();

            #pragma unroll
            for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
            {
                int item_offset = int(ITEM * BLOCK_THREADS) + threadIdx.x;
                if (PADDING) item_offset += item_offset >> LOG_SMEM_BANKS;
                items[ITEM] = smem_storage.exchange[item_offset];
            }
        }
        else
        {
            T temp_items[ITEMS_PER_THREAD];

            int slice_lane  = threadIdx.x % TIME_SLICED_THREADS;
            int slice_id    = threadIdx.x / TIME_SLICED_THREADS;

            #pragma unroll
            for (int SLICE = 0; SLICE < TIME_SLICES; SLICE++)
            {
                __syncthreads();

                if (slice_id == SLICE)
                {
                    #pragma unroll
                    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
                    {
                        int item_offset = (slice_lane * ITEMS_PER_THREAD) + ITEM;
                        if (PADDING) item_offset += item_offset >> LOG_SMEM_BANKS;
                        smem_storage.exchange[item_offset] = items[ITEM];
                    }
                }

                __syncthreads();

                #pragma unroll
                for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
                {
                    int item_offset = int(ITEM * BLOCK_THREADS) + threadIdx.x - (TIME_SLICED_ITEMS * SLICE);
                    if ((item_offset >= 0) && (item_offset < TIME_SLICED_ITEMS))
                    {
                        if (PADDING) item_offset += item_offset >> LOG_SMEM_BANKS;
                        temp_items[ITEM] = smem_storage.exchange[item_offset];
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

        CubLog("items [%d, %d]\n", items[0], items[1]);
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
        CubLog("Out begin items [%d, %d]\n", items[0], items[1]);

        int slice_lane         = (WARP_TIME_SLICING == 1) ? threadIdx.x : threadIdx.x % TIME_SLICED_THREADS;
        int slice_id    = (WARP_TIME_SLICING == 1) ? 0 : threadIdx.x / TIME_SLICED_THREADS;

        // First slice
        if (slice_id == 0)
        {
            #pragma unroll
            for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
            {
                int item_offset = (slice_lane * ITEMS_PER_THREAD) + ITEM;
                if (PADDING) item_offset += item_offset >> LOG_SMEM_BANKS;
                smem_storage.exchange[item_offset] = items[ITEM];
            }

            // Prevent hoisting
            __threadfence_block();

            #pragma unroll
            for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
            {
                int item_offset = (ITEM * TIME_SLICED_THREADS) + slice_lane;
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
                    int item_offset = (slice_lane * ITEMS_PER_THREAD) + ITEM;
                    if (PADDING) item_offset += item_offset >> LOG_SMEM_BANKS;
                    smem_storage.exchange[item_offset] = items[ITEM];
                }

                // Prevent hoisting
                __threadfence_block();

                #pragma unroll
                for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
                {
                    int item_offset = (ITEM * TIME_SLICED_THREADS) + slice_lane;
                    if (PADDING) item_offset += item_offset >> LOG_SMEM_BANKS;
                    items[ITEM] = smem_storage.exchange[item_offset];
                }
            }
        }

        CubLog("\t\tOut end items [%d, %d]\n", items[0], items[1]);
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
        if (WARP_TIME_SLICING == 1)
        {
            #pragma unroll
            for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
            {
                int item_offset = int(ITEM * BLOCK_THREADS) + threadIdx.x;
                if (PADDING) item_offset = SHR_ADD(item_offset, LOG_SMEM_BANKS, item_offset);
                smem_storage.exchange[item_offset] = items[ITEM];
            }

            __syncthreads();

            #pragma unroll
            for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
            {
                int item_offset = (threadIdx.x * ITEMS_PER_THREAD) + ITEM;
                if (PADDING) item_offset = SHR_ADD(item_offset, LOG_SMEM_BANKS, item_offset);
                items[ITEM] = smem_storage.exchange[item_offset];
            }
        }
        else
        {
            T temp_items[ITEMS_PER_THREAD];

            int slice_lane         = (WARP_TIME_SLICING == 1) ? threadIdx.x : threadIdx.x % TIME_SLICED_THREADS;
            int slice_id    = (WARP_TIME_SLICING == 1) ? 0 : threadIdx.x / TIME_SLICED_THREADS;

            #pragma unroll
            for (int SLICE = 0; SLICE < TIME_SLICES; SLICE++)
            {
                __syncthreads();

                #pragma unroll
                for (
                    int ITEM = SLICE * ITEMS_PER_THREAD_PER_SLICE;
                    ITEM < CUB_MIN((SLICE + 1)* ITEMS_PER_THREAD_PER_SLICE, ITEMS_PER_THREAD);
                    ITEM++)
                {
                    int item_offset = int(ITEM * BLOCK_THREADS) + threadIdx.x - (TIME_SLICED_ITEMS * SLICE);
                    if (PADDING) item_offset = SHR_ADD(item_offset, LOG_SMEM_BANKS, item_offset);
                    smem_storage.exchange[item_offset] = items[ITEM];
                }

                __syncthreads();

                if (slice_id == SLICE)
                {
                    #pragma unroll
                    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
                    {
                        int item_offset = (slice_lane * ITEMS_PER_THREAD) + ITEM;
                        if (PADDING) item_offset = SHR_ADD(item_offset, LOG_SMEM_BANKS, item_offset);
                        temp_items[ITEM] = smem_storage.exchange[item_offset];
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

        CubLog("\t\titems [%d, %d]\n", items[0], items[1]);
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
        int warp_lane               = threadIdx.x % PtxArchProps::WARP_THREADS;
        int wid                     = threadIdx.x / PtxArchProps::WARP_THREADS;
        int slice_id                = (WARP_TIME_SLICING == 1) ? 0 : wid / WARP_TIME_SLICING;
        int slice_offset             = slice_id * WARP_TILE_ITEMS;



        int slice_lane         = (WARP_TIME_SLICING == 1) ? threadIdx.x : threadIdx.x % TIME_SLICED_THREADS;
        int slice_id    = (WARP_TIME_SLICING == 1) ? 0 : threadIdx.x / TIME_SLICED_THREADS;

        // First slice
        if (slice_id == 0)
        {



            #pragma unroll
            for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
            {
                int item_offset = (ITEM * PtxArchProps::WARP_THREADS) +  + slice_lane;
                if (PADDING) item_offset = SHR_ADD(item_offset, LOG_SMEM_BANKS, item_offset);

                CubLog("In begin items[%d](%d) slice_lane %d slice_id %d item_offset %d\n", ITEM, items[ITEM], slice_lane, slice_id, item_offset);

                smem_storage.exchange[item_offset] = items[ITEM];
            }

            // Prevent hoisting
            __threadfence_block();

            #pragma unroll
            for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
            {
                int item_offset = (slice_lane * ITEMS_PER_THREAD) + ITEM;
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
                    int item_offset = (ITEM * TIME_SLICED_THREADS) + slice_lane;
                    if (PADDING) item_offset = SHR_ADD(item_offset, LOG_SMEM_BANKS, item_offset);
                    smem_storage.exchange[item_offset] = items[ITEM];
                }

                // Prevent hoisting
                __threadfence_block();

                #pragma unroll
                for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
                {
                    int item_offset = (slice_lane * ITEMS_PER_THREAD) + ITEM;
                    if (PADDING) item_offset = SHR_ADD(item_offset, LOG_SMEM_BANKS, item_offset);
                    items[ITEM] = smem_storage.exchange[item_offset];
                }
            }
        }

        CubLog("\t\tIn end items [%d, %d]\n", items[0], items[1]);
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
        if (WARP_TIME_SLICING == 1)
        {
            #pragma unroll
            for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
            {
                int item_offset = ranks[ITEM];
                if (PADDING) item_offset = SHR_ADD(item_offset, LOG_SMEM_BANKS, item_offset);
                smem_storage.exchange[item_offset] = items[ITEM];
            }

            __syncthreads();

            #pragma unroll
            for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
            {
                int item_offset = (threadIdx.x * ITEMS_PER_THREAD) + ITEM;
                if (PADDING) item_offset = SHR_ADD(item_offset, LOG_SMEM_BANKS, item_offset);
                items[ITEM] = smem_storage.exchange[item_offset];
            }
        }
        else
        {
            T temp_items[ITEMS_PER_THREAD];

            int slice_lane  = (WARP_TIME_SLICING == 1) ? threadIdx.x : threadIdx.x % TIME_SLICED_THREADS;
            int slice_id    = (WARP_TIME_SLICING == 1) ? 0 : threadIdx.x / TIME_SLICED_THREADS;

            #pragma unroll
            for (int SLICE = 0; SLICE < TIME_SLICES; SLICE++)
            {
                __syncthreads();

                #pragma unroll
                for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
                {
                    if ((ranks[ITEM] >= TIME_SLICED_ITEMS * SLICE) && (ranks[ITEM] < TIME_SLICED_ITEMS * (SLICE + 1)))
                    {
                        int item_offset = ranks[ITEM] - (TIME_SLICED_ITEMS * SLICE);
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
                        int item_offset = (slice_lane * ITEMS_PER_THREAD) + ITEM;
                        if (PADDING) item_offset = SHR_ADD(item_offset, LOG_SMEM_BANKS, item_offset);
                        temp_items[ITEM] = smem_storage.exchange[item_offset];
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
        if (WARP_TIME_SLICING == 1)
        {
            #pragma unroll
            for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
            {
                int item_offset = ranks[ITEM];
                if (PADDING) item_offset = SHR_ADD(item_offset, LOG_SMEM_BANKS, item_offset);
                smem_storage.exchange[item_offset] = items[ITEM];
            }

            __syncthreads();

            #pragma unroll
            for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
            {
                int item_offset = int(ITEM * BLOCK_THREADS) + threadIdx.x;
                if (PADDING) item_offset = SHR_ADD(item_offset, LOG_SMEM_BANKS, item_offset);
                items[ITEM] = smem_storage.exchange[item_offset];
            }
        }
        else
        {
            T temp_items[ITEMS_PER_THREAD];

            int slice_lane  = (WARP_TIME_SLICING == 1) ? threadIdx.x : threadIdx.x % TIME_SLICED_THREADS;
            int slice_id    = (WARP_TIME_SLICING == 1) ? 0 : threadIdx.x / TIME_SLICED_THREADS;

            #pragma unroll
            for (int SLICE = 0; SLICE < TIME_SLICES; SLICE++)
            {
                __syncthreads();

                #pragma unroll
                for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
                {
                    if ((ranks[ITEM] >= TIME_SLICED_ITEMS * SLICE) && (ranks[ITEM] < TIME_SLICED_ITEMS * (SLICE + 1)))
                    {
                        int item_offset = ranks[ITEM] - (TIME_SLICED_ITEMS * SLICE);
                        if (PADDING) item_offset = SHR_ADD(item_offset, LOG_SMEM_BANKS, item_offset);
                        smem_storage.exchange[item_offset] = items[ITEM];
                    }
                }

                __syncthreads();

                #pragma unroll
                for (
                    int ITEM = SLICE * ITEMS_PER_THREAD_PER_SLICE;
                    ITEM < CUB_MIN((SLICE + 1)* ITEMS_PER_THREAD_PER_SLICE, ITEMS_PER_THREAD);
                    ITEM++)
                {
                    int item_offset = int(ITEM * BLOCK_THREADS) + threadIdx.x - (TIME_SLICED_ITEMS * SLICE);
                    if (PADDING) item_offset = SHR_ADD(item_offset, LOG_SMEM_BANKS, item_offset);
                    temp_items[ITEM] = smem_storage.exchange[item_offset];
                }
            }

            // Copy
            #pragma unroll
            for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
            {
                items[ITEM] = temp_items[ITEM];
            }
        }
    }

    //@}  end member group


};

/** @} */       // end group BlockModule

}               // CUB namespace
CUB_NS_POSTFIX  // Optional outer namespace(s)

