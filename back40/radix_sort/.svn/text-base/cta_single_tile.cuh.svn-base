/******************************************************************************
 * 
 * Copyright (c) 2010-2012, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2012, NVIDIA CORPORATION.  All rights reserved.
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License. 
 * 
 ******************************************************************************/

/******************************************************************************
 * CTA-wide abstraction for sorting a single tile of input
 ******************************************************************************/

#pragma once

#include "../../cub/cub.cuh"
#include "../ns_wrapper.cuh"
#include "sort_utils.cuh"

BACK40_NS_PREFIX
namespace back40 {
namespace radix_sort {


//---------------------------------------------------------------------
// Tuning policy types
//---------------------------------------------------------------------

/**
 * Single tile CTA tuning policy
 */
template <
    int                     _RADIX_BITS,        // The number of radix bits, i.e., log2(bins)
    int                     _BLOCK_THREADS,       // The number of threads per CTA
    int                     _THREAD_ITEMS,      // The number of consecutive items to load per thread per tile
    cub::PtxLoadModifier    _LOAD_MODIFIER,     // Load cache-modifier
    cub::PtxStoreModifier   _STORE_MODIFIER,    // Store cache-modifier
    cudaSharedMemConfig     _SMEM_CONFIG>       // Shared memory bank size
struct BlockSingleTilePolicy
{
    enum
    {
        RADIX_BITS      = _RADIX_BITS,
        BLOCK_THREADS     = _BLOCK_THREADS,
        THREAD_ITEMS    = _THREAD_ITEMS,
        TILE_ITEMS      = BLOCK_THREADS * THREAD_ITEMS,
    };

    static const cub::PtxLoadModifier   LOAD_MODIFIER       = _LOAD_MODIFIER;
    static const cub::PtxStoreModifier  STORE_MODIFIER      = _STORE_MODIFIER;
    static const cudaSharedMemConfig    SMEM_CONFIG         = _SMEM_CONFIG;
};



//---------------------------------------------------------------------
// CTA-wide abstractions
//---------------------------------------------------------------------

/**
 * CTA-wide abstraction for sorting a single tile of input
 */
template <
    typename BlockSingleTilePolicy,
    typename KeyType,
    typename ValueType>
class BlockSingleTile
{
private:

    //---------------------------------------------------------------------
    // Type definitions and constants
    //---------------------------------------------------------------------

    // Appropriate unsigned-bits representation of KeyType
    typedef typename KeyTraits<KeyType>::UnsignedBits UnsignedBits;

    static const UnsignedBits     MIN_KEY     = KeyTraits<KeyType>::MIN_KEY;
    static const UnsignedBits     MAX_KEY     = KeyTraits<KeyType>::MAX_KEY;

    enum
    {
        BLOCK_THREADS         = BlockSingleTilePolicy::BLOCK_THREADS,
        RADIX_BITS          = BlockSingleTilePolicy::RADIX_BITS,
        KEYS_PER_THREAD     = BlockSingleTilePolicy::THREAD_ITEMS,
        TILE_ITEMS          = BlockSingleTilePolicy::TILE_ITEMS,
    };


public:

    /**
     * Shared memory storage layout
     */
    typedef typename BlockRadixSortT::SmemStorage SmemStorage;


    //---------------------------------------------------------------------
    // Interface
    //---------------------------------------------------------------------

    /**
     * Sort a tile.  (Specialized for keys-only sorting.)
     */
    static __device__ __forceinline__ void Sort(
        SmemStorage         &smem_storage,
        KeyType             *d_keys_in,
        KeyType             *d_keys_out,
        cub::NullType       *d_values_in,
        cub::NullType       *d_values_out,
        unsigned int        begin_bit,
        const unsigned int  &end_bit,
        const int           &num_elements)
    {
        UnsignedBits* d_bits_in = reinterpret_cast<UnsignedBits*>(d_keys_in);
        UnsignedBits* d_bits_out = reinterpret_cast<UnsignedBits*>(d_keys_out);

        UnsignedBits keys[KEYS_PER_THREAD];

        // Load striped
        cub::BlockLoadDirectStriped(d_bits_in, num_elements, MAX_KEY, keys, BLOCK_THREADS);

        // Twiddle key bits if necessary
        #pragma unroll
        for (int KEY = 0; KEY < KEYS_PER_THREAD; KEY++)
        {
            keys[KEY] = KeyTraits<KeyType>::TwiddleIn(keys[KEY]);
        }

        // Sort
        BlockRadixSortT::SortStriped(smem_storage, keys, begin_bit, end_bit);

        // TODO: Twiddle key bits if necessary
        #pragma unroll
        for (int KEY = 0; KEY < KEYS_PER_THREAD; KEY++)
        {
            keys[KEY] = KeyTraits<KeyType>::TwiddleOut(keys[KEY]);
        }

        // Store keys
        cub::BlockStoreDirectStriped(d_keys_out, num_elements, keys, BLOCK_THREADS);
        StoreTile(
            keys,
            reinterpret_cast<UnsignedBits*>(d_bits_in),
            num_elements);
    }



    /**
     * Sort a tile.  (Specialized for key-value sorting.)
     */
    static __device__ __forceinline__ void Sort(
        SmemStorage         &smem_storage,
        KeyType             *d_keys_in,
        KeyType             *d_keys_out,
        cub::NullType         *d_values_in,
        cub::NullType         *d_values_out,
        unsigned int         current_bit,
        const unsigned int     &bits_remaining,
        const int             &num_elements)
    {
        UnsignedBits keys[KEYS_PER_THREAD];

        // Initialize keys to default key value
        #pragma unroll
        for (int KEY = 0; KEY < KEYS_PER_THREAD; KEY++)
        {
            keys[KEY] = MAX_KEY;
        }

        // Load keys
        LoadTile(
            smem_storage.key_exchange,
            keys,
            reinterpret_cast<UnsignedBits*>(d_keys_in),
            num_elements);

        __syncthreads();

        // Twiddle key bits if necessary
        #pragma unroll
        for (int KEY = 0; KEY < KEYS_PER_THREAD; KEY++)
        {
            keys[KEY] = KeyTraits<KeyType>::TwiddleIn(keys[KEY]);
        }

        // Sort
        BlockRadixSortT::SortThreadToBlockStride(
            smem_storage.sorting_storage,
            keys,
            current_bit,
            bits_remaining);

        // Twiddle key bits if necessary
        #pragma unroll
        for (int KEY = 0; KEY < KEYS_PER_THREAD; KEY++)
        {
            keys[KEY] = KeyTraits<KeyType>::TwiddleOut(keys[KEY]);
        }

        // Store keys
        StoreTile(
            keys,
            reinterpret_cast<UnsignedBits*>(d_keys_out),
            num_elements);
    }

};


} // namespace radix_sort
} // namespace back40
BACK40_NS_POSTFIX
