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
 * cub::TilesHisto256 implements an abstraction of CUDA thread blocks for histogramming multiple tiles as part of device-wide 256-bin histogram.
 */

#pragma once

#include <iterator>

#include "../../util_arch.cuh"
#include "../../block/block_radix_sort.cuh"
#include "../../block/block_discontinuity.cuh"
#include "../../grid/grid_mapping.cuh"
#include "../../grid/grid_even_share.cuh"
#include "../../grid/grid_queue.cuh"
#include "../../block/block_load.cuh"
#include "../../block/block_histo_256.cuh"
#include "../../util_vector.cuh"
#include "../../util_namespace.cuh"

/// Optional outer namespace(s)
CUB_NS_PREFIX

/// CUB namespace
namespace cub {



/******************************************************************************
 * Tuning policy
 ******************************************************************************/

/**
 * Tuning policy for TilesHisto256
 */
template <
    int                     _BLOCK_THREADS,
    int                     _ITEMS_PER_THREAD,
    BlockHisto256Algorithm  _BLOCK_ALGORITHM,
    GridMappingStrategy     _GRID_MAPPING>
struct TilesHisto256Policy
{
    enum
    {
        BLOCK_THREADS       = _BLOCK_THREADS,
        ITEMS_PER_THREAD    = _ITEMS_PER_THREAD,
    };

    static const BlockHisto256Algorithm     BLOCK_ALGORITHM      = _BLOCK_ALGORITHM;
    static const GridMappingStrategy        GRID_MAPPING         = _GRID_MAPPING;
};



/******************************************************************************
 * TilesHisto256
 ******************************************************************************/

/**
 * \brief TilesHisto256 implements an abstraction of CUDA thread blocks for participating in device-wide histogram.
 */
template <
    typename    TilesHisto256Policy,      ///< Tuning policy
    int         CHANNELS,                 ///< Number of channels interleaved in the input data (may be greater than the number of active channels being histogrammed)
    int         ACTIVE_CHANNELS,          ///< Number of channels actively being histogrammed
    typename    HistoCounter,             ///< Integral type for counting sample occurrences per histogram bin
    typename    SizeT>                    ///< Integer type for offsets
class TilesHisto256
{
private:

    //---------------------------------------------------------------------
    // Types and constants
    //---------------------------------------------------------------------

    // Constants
    enum
    {
        BLOCK_THREADS       = TilesHisto256Policy::BLOCK_THREADS,
        ITEMS_PER_THREAD    = TilesHisto256Policy::ITEMS_PER_THREAD,
        TILE_CHANNEL_ITEMS  = BLOCK_THREADS * ITEMS_PER_THREAD,
        TILE_ITEMS          = TILE_CHANNEL_ITEMS * CHANNELS,
    };

    static const BlockHisto256Algorithm BLOCK_ALGORITHM = TilesHisto256Policy::BLOCK_ALGORITHM;

private:

    //---------------------------------------------------------------------
    // Utility operations
    //---------------------------------------------------------------------

    /**
     * BLOCK_BYTE_HISTO_SORT algorithmic variant.
     *
     * Keeps the histogram(s) in registers as tiles are processed
     */
    template <
        BlockHisto256Algorithm _BLOCK_ALGORITHM, int DUMMY = 0>
    struct Internal
    {
        //---------------------------------------------------------------------
        // Types
        //---------------------------------------------------------------------

        // Constants
        enum
        {
            STRIPED_BINS_PER_THREAD = (256 + BLOCK_THREADS - 1) / BLOCK_THREADS,
        };

        // Parameterize BlockRadixSort type for our thread block
        typedef BlockRadixSort<unsigned char, BLOCK_THREADS, ITEMS_PER_THREAD> BlockRadixSortT;

        // Parameterize BlockDiscontinuity type for our thread block
        typedef BlockDiscontinuity<unsigned char, BLOCK_THREADS> BlockDiscontinuityT;

        // Shared memory
        union SmemStorage
        {
            // Storage for sorting bin values
            typename BlockRadixSortT::SmemStorage sort_storage;

            struct
            {
                // Storage for detecting discontinuities in the tile of sorted bin values
                typename BlockDiscontinuityT::SmemStorage discont_storage;

                // Storage for noting begin/end offsets of bin runs in the tile of sorted bin values
                unsigned int run_begin[256];
                unsigned int run_end[256];
            };
        };


        // Discontinuity functor
        struct DiscontinuityOp
        {
            // Reference to smem_storage
            SmemStorage &smem_storage;

            // Constructor
            __device__ __forceinline__ DiscontinuityOp(SmemStorage &smem_storage) : smem_storage(smem_storage) {}

            // Discontinuity predicate
            __device__ __forceinline__ bool operator()(const unsigned char &a, const unsigned char &b, unsigned int b_index)
            {
                if (a != b)
                {
                    // Note the begin/end offsets in shared storage
                    smem_storage.run_begin[b] = b_index;
                    smem_storage.run_end[a] = b_index;

                    return true;
                }
                else
                {
                    return false;
                }
            }
        };


        //---------------------------------------------------------------------
        // Fields
        //---------------------------------------------------------------------

        // Shared memory storage reference
        SmemStorage &smem_storage;

        // Histogram counters striped across threads
        HistoCounter bins[ACTIVE_CHANNELS][STRIPED_BINS_PER_THREAD];


        //---------------------------------------------------------------------
        // Operations
        //---------------------------------------------------------------------

        // Constructor
        Internal(
            SmemStorage     &smem_storage,
            HistoCounter    (&histograms)[ACTIVE_CHANNELS][256])




        template <
            typename            HistoCounter>
        static __device__ __forceinline__ void Composite(
            SmemStorage         &smem_storage,
            unsigned char       (&items)[ITEMS_PER_THREAD],
            HistoCounter        histogram[256])
        {
            // Sort bytes in blocked arrangement
            BlockRadixSortT::SortBlocked(smem_storage.sort_storage, items);

            __syncthreads();

            // Initialize the shared memory's run_begin and run_end for each bin
            int histo_offset = 0;

            #pragma unroll
            for(; histo_offset + BLOCK_THREADS <= 256; histo_offset += BLOCK_THREADS)
            {
                smem_storage.run_begin[histo_offset + threadIdx.x] = TILE_CHANNEL_ITEMS;
                smem_storage.run_end[histo_offset + threadIdx.x] = TILE_CHANNEL_ITEMS;
            }
            // Finish up with guarded initialization if necessary
            if ((histo_offset < BLOCK_THREADS) && (histo_offset + threadIdx.x < 256))
            {
                smem_storage.run_begin[histo_offset + threadIdx.x] = TILE_CHANNEL_ITEMS;
                smem_storage.run_end[histo_offset + threadIdx.x] = TILE_CHANNEL_ITEMS;
            }

            __syncthreads();

            int flags[ITEMS_PER_THREAD];    // unused

            // Note the begin/end run offsets of bin runs in the sorted tile
            DiscontinuityOp flag_op(smem_storage);
            BlockDiscontinuityT::Flag(smem_storage.discont_storage, items, flag_op, flags);

            // Update begin for first item
            if (threadIdx.x == 0) smem_storage.run_begin[items[0]] = 0;

            __syncthreads();

            // Composite into histogram
            histo_offset = 0;

            #pragma unroll
            for(; histo_offset + BLOCK_THREADS <= 256; histo_offset += BLOCK_THREADS)
            {
                int thread_offset = histo_offset + threadIdx.x;
                HistoCounter count = smem_storage.run_end[thread_offset] - smem_storage.run_begin[thread_offset];
                histogram[thread_offset] += count;
            }
            // Finish up with guarded composition if necessary
            if ((histo_offset < BLOCK_THREADS) && (histo_offset + threadIdx.x < 256))
            {
                int thread_offset = histo_offset + threadIdx.x;
                HistoCounter count = smem_storage.run_end[thread_offset] - smem_storage.run_begin[thread_offset];
                histogram[thread_offset] += count;
            }
        }


        /**
         * Process one channel within a tile.
         */
        template <
            typename        InputIteratorRA,
            typename        HistoCounter,
            int             ACTIVE_CHANNELS>
        static __device__ __forceinline__ void ConsumeTileChannel(
            SmemStorage     &smem_storage,
            int             channel,
            InputIteratorRA d_in,
            SizeT           block_offset,
            HistoCounter    (&histograms)[ACTIVE_CHANNELS][256],
            const int       &guarded_items = TILE_ITEMS)
        {
            // Load items in striped fashion
            if (guarded_items < TILE_ITEMS)
            {
                unsigned char items[ITEMS_PER_THREAD];

                // Assign our tid as the bin for out-of-bounds items (to give an even distribution), and keep track of how oob items to subtract out later
                int bounds = (guarded_items - (threadIdx.x * CHANNELS));

                #pragma unroll
                for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
                {
                    items[ITEM] = ((ITEM * BLOCK_THREADS * CHANNELS) < bounds) ?
                        d_in[channel + block_offset + (ITEM * BLOCK_THREADS * CHANNELS) + (threadIdx.x * CHANNELS)] :
                        0;
                }

                // Composite our histogram data
                Composite(smem_storage, items, histograms[channel]);

                __syncthreads();

                // Correct the overcounting in the zero-bin from invalid (out-of-bounds) items
                if (threadIdx.x == 0)
                {
                    int extra = (TILE_ITEMS - guarded_items) / CHANNELS;
                    histograms[channel][0] -= extra;
                }
            }
            else
            {
                unsigned char items[ITEMS_PER_THREAD];

                // Unguarded loads
                #pragma unroll
                for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
                {
                    items[ITEM] = d_in[channel + block_offset + (ITEM * BLOCK_THREADS * CHANNELS) + (threadIdx.x * CHANNELS)];
                }

                // Composite our histogram data
                Composite(smem_storage, items, histograms[channel]);
            }
        }


        /**
         * Process one tile.
         */
        template <
            typename        InputIteratorRA,
            typename        HistoCounter,
            int             ACTIVE_CHANNELS>
        static __device__ __forceinline__ void ConsumeTile(
            SmemStorage     &smem_storage,
            InputIteratorRA d_in,
            SizeT           block_offset,
            HistoCounter    (&histograms)[ACTIVE_CHANNELS][256],
            const int       &guarded_items = TILE_ITEMS)
        {
            // We take several passes through the tile in this variant, extracting the samples for one channel at a time

            // First channel
            ConsumeTileChannel(smem_storage, 0, d_in, block_offset, histograms, guarded_items);

            // Iterate through remaining channels
            #pragma unroll
            for (int CHANNEL = 1; CHANNEL < ACTIVE_CHANNELS; ++CHANNEL)
            {
                __syncthreads();

                ConsumeTileChannel(smem_storage, CHANNEL, d_in, block_offset, histograms, guarded_items);
            }
        }
    };



    /**
     * BLOCK_BYTE_HISTO_ATOMIC algorithmic variant
     */
    template <int DUMMY>
    struct Internal<BLOCK_BYTE_HISTO_ATOMIC, DUMMY>
    {
        struct SmemStorage {};

        /**
         * Process one tile.
         */
        template <
            typename        InputIteratorRA,
            typename        HistoCounter,
            int             ACTIVE_CHANNELS>
        static __device__ __forceinline__ void ConsumeTile(
            SmemStorage     &smem_storage,
            InputIteratorRA d_in,
            SizeT           block_offset,
            HistoCounter    (&histograms)[ACTIVE_CHANNELS][256],
            const int       &guarded_items = TILE_ITEMS)
        {

            if (guarded_items < TILE_ITEMS)
            {
                int bounds = guarded_items - (threadIdx.x * CHANNELS);

                #pragma unroll
                for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
                {
                    #pragma unroll
                    for (int CHANNEL = 0; CHANNEL < CHANNELS; ++CHANNEL)
                    {
                        if (((ACTIVE_CHANNELS == CHANNELS) || (CHANNEL < ACTIVE_CHANNELS)) && ((ITEM * BLOCK_THREADS * CHANNELS) + CHANNEL < bounds))
                        {
                            unsigned char item  = d_in[block_offset + (ITEM * BLOCK_THREADS * CHANNELS) + (threadIdx.x * CHANNELS) + CHANNEL];
                            atomicAdd(histograms[CHANNEL] + item, 1);
                        }
                    }
                }

            }
            else
            {
                unsigned char items[ITEMS_PER_THREAD][CHANNELS];

                #pragma unroll
                for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
                {
                    #pragma unroll
                    for (int CHANNEL = 0; CHANNEL < CHANNELS; ++CHANNEL)
                    {
                        if (CHANNEL < ACTIVE_CHANNELS)
                        {
                            items[ITEM][CHANNEL] = d_in[block_offset + (ITEM * BLOCK_THREADS * CHANNELS) + (threadIdx.x * CHANNELS) + CHANNEL];
                        }
                    }
                }

                __threadfence_block();

                #pragma unroll
                for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
                {
                    #pragma unroll
                    for (int CHANNEL = 0; CHANNEL < CHANNELS; ++CHANNEL)
                    {
                        if (CHANNEL < ACTIVE_CHANNELS)
                        {
                            atomicAdd(histograms[CHANNEL] + items[ITEM][CHANNEL], 1);
                        }
                    }
                }
            }
        }
    };

    /**
     * Initialize shared histogram
     */
    template <typename HistoCounter>
    static __device__ __forceinline__ void InitHistogram(HistoCounter histogram[256])
    {
        // Initialize histogram bin counts to zeros
        int histo_offset = 0;

        #pragma unroll
        for(; histo_offset + BLOCK_THREADS <= 256; histo_offset += BLOCK_THREADS)
        {
            histogram[histo_offset + threadIdx.x] = 0;
        }
        // Finish up with guarded initialization if necessary
        if ((histo_offset < BLOCK_THREADS) && (histo_offset + threadIdx.x < 256))
        {
            histogram[histo_offset + threadIdx.x] = 0;
        }
    }


    // The internal implementation we will use
    typedef Internal<BLOCK_ALGORITHM> Internal;

    // Shared memory type for this threadblock
    struct _SmemStorage
    {
        SizeT                           block_offset;   // Location where to dequeue input for dynamic operation
        typename Internal::SmemStorage  block_histo;    // Smem needed for cooperative histogramming
    };

public:

    /// \smemstorage{TilesHisto256}
    typedef _SmemStorage SmemStorage;




public:

    //---------------------------------------------------------------------
    // Interface
    //---------------------------------------------------------------------

    /**
     * \brief Consumes input tiles using an even-share policy
     */
    template <
        typename        InputIteratorRA,
        typename        HistoCounter,
        int             ACTIVE_CHANNELS>
    static __device__ __forceinline__ void ProcessTilesEvenShare(
        SmemStorage     &smem_storage,
        InputIteratorRA d_in,
        SizeT           block_offset,
        const SizeT     &block_oob,
        HistoCounter    (&histograms)[ACTIVE_CHANNELS][256])
    {
        // Initialize histograms
        #pragma unroll
        for (int CHANNEL = 0; CHANNEL < ACTIVE_CHANNELS; ++CHANNEL)
        {
            InitHistogram(histograms[CHANNEL]);
        }

        __syncthreads();

        // Consume full tiles
        while (block_offset + TILE_ITEMS <= block_oob)
        {
            Internal<BLOCK_ALGORITHM>::ConsumeTile(smem_storage, d_in, block_offset, histograms);

            block_offset += TILE_ITEMS;

            // Skip synchro for atomic version since we know it doesn't use any smem
            if (BLOCK_ALGORITHM !=  BLOCK_BYTE_HISTO_ATOMIC)
            {
                __syncthreads();
            }
        }

        // Consume any remaining partial-tile
        if (block_offset < block_oob)
        {
            Internal<BLOCK_ALGORITHM>::ConsumeTile(smem_storage, d_in, block_offset, histograms, block_oob - block_offset);
        }
    }


    /**
     * \brief Consumes input tiles using a dynamic queue policy
     */
    template <
        typename            InputIteratorRA,
        typename            HistoCounter,
        int                 ACTIVE_CHANNELS>
    static __device__ __forceinline__ void ProcessTilesDynamic(
        SmemStorage         &smem_storage,
        InputIteratorRA     d_in,
        SizeT               num_items,
        GridQueue<SizeT>    &queue,
        HistoCounter        (&histograms)[ACTIVE_CHANNELS][256])
    {

        // Initialize histograms
        #pragma unroll
        for (int CHANNEL = 0; CHANNEL < ACTIVE_CHANNELS; ++CHANNEL)
        {
            InitHistogram(histograms[CHANNEL]);
        }

        // Dynamically consume tiles
        while (true)
        {
            // Dequeue up to TILE_ITEMS
            if (threadIdx.x == 0)
            {
                smem_storage.block_offset = queue.Drain(TILE_ITEMS);
            }

            __syncthreads();

            SizeT block_offset = smem_storage.block_offset;

            __syncthreads();

            if (block_offset + TILE_ITEMS > num_items)
            {
                if (block_offset < num_items)
                {
                    // We have less than a full tile to consume
                    Internal<BLOCK_ALGORITHM>::ConsumeTile(smem_storage, d_in, block_offset, histograms, num_items - block_offset);
                }

                // No more work to do
                break;
            }

            // We have a full tile to consume
            Internal<BLOCK_ALGORITHM>::ConsumeTile(smem_storage, d_in, block_offset, histograms);
        }
    }


    /**
     * Specialized for GRID_MAPPING_EVEN_SHARE
     */
    template <GridMappingStrategy GRID_MAPPING, int DUMMY = 0>
    struct Mapping
    {
        template <
            typename                InputIteratorRA,
            typename                HistoCounter,
            int                     ACTIVE_CHANNELS>
        static __device__ __forceinline__ void ProcessTiles(
            SmemStorage             &smem_storage,
            InputIteratorRA         d_in,
            SizeT                   num_items,
            GridEvenShare<SizeT>    &even_share,
            GridQueue<SizeT>        &queue,
            HistoCounter            (&histograms)[ACTIVE_CHANNELS][256])
        {
            even_share.BlockInit();
            return ProcessTilesEvenShare(smem_storage, d_in, even_share.block_offset, even_share.block_oob, histograms);
        }

    };


    /**
     * Specialized for GRID_MAPPING_DYNAMIC
     */
    template <int DUMMY>
    struct Mapping<GRID_MAPPING_DYNAMIC, DUMMY>
    {
        template <
            typename                InputIteratorRA,
            typename                HistoCounter,
            int                     ACTIVE_CHANNELS>
        static __device__ __forceinline__ void ProcessTiles(
            SmemStorage             &smem_storage,
            InputIteratorRA         d_in,
            SizeT                   num_items,
            GridEvenShare<SizeT>    &even_share,
            GridQueue<SizeT>        &queue,
            HistoCounter            (&histograms)[ACTIVE_CHANNELS][256])
        {
            ProcessTilesDynamic(smem_storage, d_in, num_items, queue, histograms);
        }

    };

};


}               // CUB namespace
CUB_NS_POSTFIX  // Optional outer namespace(s)

