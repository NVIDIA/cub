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
 * cub::BlockRadixSort provides variants of parallel radix sorting across a CUDA threadblock.
 */


#pragma once

#include "../util_namespace.cuh"
#include "../util_arch.cuh"
#include "../util_type.cuh"
#include "block_exchange.cuh"
#include "block_radix_rank.cuh"

/// Optional outer namespace(s)
CUB_NS_PREFIX

/// CUB namespace
namespace cub {

/**
 * \addtogroup BlockModule
 * @{
 */

/**
 * \brief BlockRadixSort provides variants of parallel radix sorting across a CUDA threadblock.  ![](sorting_logo.png)
 *
 * \par Overview
 * The [<em>radix sorting method</em>](http://en.wikipedia.org/wiki/Radix_sort) relies upon a positional representation for
 * keys, i.e., each key is comprised of an ordered sequence of symbols (e.g., digits,
 * characters, etc.) specified from least-significant to most-significant.  For a
 * given input sequence of keys and a set of rules specifying a total ordering
 * of the symbolic alphabet, the radix sorting method produces a lexicographic
 * ordering of those keys.
 *
 * \par
 * BlockRadixSort can sort all of the built-in C++ numeric primitive types, e.g.:
 * <tt>unsigned char</tt>, \p int, \p double, etc.  Within each key, the implementation treats fixed-length
 * bit-sequences of \p RADIX_BITS as radix digit places.  Although the direct radix sorting
 * method can only be applied to unsigned integral types, BlockRadixSort
 * is able to sort signed and floating-point types via simple bit-wise transformations
 * that ensure lexicographic key ordering.
 *
 * \par
 * For convenience, BlockRadixSort exposes a spectrum of entrypoints that differ by:
 * - Value association (keys-only <em>vs.</em> key-value-pairs)
 * - Input/output data arrangements (combinations of [<em>blocked</em>](index.html#sec3sec3) and [<em>striped</em>](index.html#sec3sec3) arrangements)
 *
 * \tparam KeyType              Key type
 * \tparam BLOCK_THREADS        The threadblock size in threads
 * \tparam ITEMS_PER_THREAD     The number of items per thread
 * \tparam ValueType            <b>[optional]</b> Value type (default: cub::NullType)
 * \tparam RADIX_BITS           <b>[optional]</b> The number of radix bits per digit place (default: 5 bits)
 * \tparam MEMOIZE_OUTER_SCAN   <b>[optional]</b> Whether or not to buffer outer raking scan partials to incur fewer shared memory reads at the expense of higher register pressure (default: true for architectures SM35 and newer, false otherwise).  See BlockScanAlgorithm::BLOCK_SCAN_RAKING_MEMOIZE for more details.
 * \tparam INNER_SCAN_ALGORITHM <b>[optional]</b> The cub::BlockScanAlgorithm algorithm to use (default: cub::BLOCK_SCAN_WARP_SCANS)
 * \tparam SMEM_CONFIG          <b>[optional]</b> Shared memory bank mode (default: \p cudaSharedMemBankSizeFourByte)
 *
 * \par Usage Considerations
 * - After any sorting operation, a subsequent <tt>__syncthreads()</tt> barrier
 *   is required if the supplied BlockRadixSort::SmemStorage is to be reused or repurposed
 *   by the threadblock.
 * - BlockRadixSort can only accommodate one associated tile of values. To "truck along"
 *   more than one tile of values, simply perform a key-value sort of the keys paired
 *   with a temporary value array that enumerates the key indices.  The reordered indices
 *   can then be used as a gather-vector for exchanging other associated tile data through
 *   shared memory.
 *
 * \par Performance Considerations
 * - The operations are most efficient (lowest instruction overhead) when:
 *      - \p BLOCK_THREADS is a multiple of the architecture's warp size
 *      - \p KeyType is an unsigned integral type
 *      - Keys are partitioned across the threadblock in a [<em>blocked arrangement</em>](index.html#sec3sec3)
 *
 * \par Algorithm
 * BlockRadixSort is based on the method presented by Merrill et al. \cite merrill_high_2011.
 * The implementation has <em>O</em>(<em>n</em>) work complexity and iterates over digit places
 * using rounds constructed of
 *    - cub::BlockRadixRank (itself constructed from cub::BlockScan)
 *    - cub::BlockExchange
 *
 * \par Examples
 * <em>Example 1.</em> Perform a radix sort over a tile of 512 integer keys that
 * are partitioned in a blocked arrangement across a 128-thread threadblock (where each thread holds 4 keys).
 *      \code
 *      #include <cub/cub.cuh>
 *
 *      __global__ void SomeKernel(...)
 *      {
 *          // Parameterize BlockRadixSort for 128 threads (4 items each) on type unsigned int
 *          typedef cub::BlockRadixSort<unsigned int, 128, 4> BlockRadixSort;
 *
 *          // Declare shared memory for BlockRadixSort
 *          __shared__ typename BlockRadixSort::SmemStorage smem_storage;
 *
 *          // A segment of consecutive input items per thread
 *          int keys[4];
 *
 *          // Obtain keys in blocked order
 *          ...
 *
 *          // Sort keys in ascending order
 *          BlockRadixSort::SortBlocked(smem_storage, keys);
 *
 *      \endcode
 *
 * \par
 * <em>Example 2.</em> Perform a key-value radix sort over the lower 20-bits of a tile of 32-bit integer
 * keys paired with floating-point values.  The data are partitioned in a striped arrangement across the threadblock.
 *      \code
 *      #include <cub/cub.cuh>
 *
 *      template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
 *      __global__ void SomeKernel(...)
 *      {
 *          // Parameterize BlockRadixSort on key-value pairs of type unsigned int, float
 *          typedef cub::BlockRadixSort<unsigned int, BLOCK_THREADS, ITEMS_PER_THREAD, float> BlockRadixSort;
 *
 *          // Declare shared memory for BlockRadixSort
 *          __shared__ typename BlockRadixSort::SmemStorage smem_storage;
 *
 *          // Input keys and values per thread (striped across the threadblock)
 *          int keys[ITEMS_PER_THREAD];
 *          float values[ITEMS_PER_THREAD];
 *
 *          // Obtain keys and values in striped order
 *          ...
 *
 *          // Sort pairs in ascending order (using only the lower 20 distinguishing key bits)
 *          BlockRadixSort::SortStriped(smem_storage, keys, values, 0, 20);
 *      }
 *
 *      \endcode
 */
template <
    typename                KeyType,
    int                     BLOCK_THREADS,
    int                     ITEMS_PER_THREAD,
    typename                ValueType               = NullType,
    int                     RADIX_BITS              = 4, //(sizeof(KeyType) < 8) ? 4 : 5,
    bool                    MEMOIZE_OUTER_SCAN      = (CUB_PTX_ARCH >= 350) ? true : false,
    BlockScanAlgorithm      INNER_SCAN_ALGORITHM    = BLOCK_SCAN_WARP_SCANS,
    cudaSharedMemConfig     SMEM_CONFIG             = cudaSharedMemBankSizeFourByte>
class BlockRadixSort
{
    //---------------------------------------------------------------------
    // Type definitions and constants
    //---------------------------------------------------------------------

private:

    // Key traits and unsigned bits type
    typedef NumericTraits<KeyType>              KeyTraits;
    typedef typename KeyTraits::UnsignedBits    UnsignedBits;

    /// BlockRadixRank utility type
    typedef BlockRadixRank<BLOCK_THREADS, RADIX_BITS, MEMOIZE_OUTER_SCAN, INNER_SCAN_ALGORITHM, SMEM_CONFIG> BlockRadixRank;

    /// BlockExchange utility type for keys
    typedef BlockExchange<KeyType, BLOCK_THREADS, ITEMS_PER_THREAD> KeyBlockExchange;

    /// BlockExchange utility type for values
    typedef BlockExchange<ValueType, BLOCK_THREADS, ITEMS_PER_THREAD> ValueBlockExchange;

    /// Shared memory storage layout type
    struct _SmemStorage
    {
        union
        {
            typename BlockRadixRank::SmemStorage          ranking_storage;
            typename KeyBlockExchange::SmemStorage        key_storage;
            typename ValueBlockExchange::SmemStorage      value_storage;
        };
    };

public:

    /// \smemstorage{BlockRadixSort}
    typedef _SmemStorage SmemStorage;


    /******************************************************************//**
     * \name Keys-only sorting
     *********************************************************************/
    //@{

    /**
     * \brief Performs a threadblock-wide radix sort over a [<em>blocked arrangement</em>](index.html#sec3sec3) of keys.
     *
     * \smemreuse
     */
    static __device__ __forceinline__ void SortBlocked(
        SmemStorage         &smem_storage,                      ///< [in] Shared reference to opaque SmemStorage layout
        KeyType             (&keys)[ITEMS_PER_THREAD],          ///< [in-out] Keys to sort
        unsigned int        begin_bit = 0,                      ///< [in] <b>[optional]</b> The beginning (least-significant) bit index needed for key comparison
        const unsigned int  &end_bit = sizeof(KeyType) * 8)     ///< [in] <b>[optional]</b> The past-the-end (most-significant) bit index needed for key comparison
    {
        UnsignedBits (&unsigned_keys)[ITEMS_PER_THREAD] =
            reinterpret_cast<UnsignedBits (&)[ITEMS_PER_THREAD]>(keys);

        // Twiddle bits if necessary
        #pragma unroll
        for (int KEY = 0; KEY < ITEMS_PER_THREAD; KEY++)
        {
            unsigned_keys[KEY] = KeyTraits::TwiddleIn(unsigned_keys[KEY]);
        }

        // Radix sorting passes
        while (true)
        {
            // Rank the blocked keys
            unsigned int ranks[ITEMS_PER_THREAD];
            BlockRadixRank::RankKeys(smem_storage.ranking_storage, unsigned_keys, ranks, begin_bit);
            begin_bit += RADIX_BITS;

            __syncthreads();

            // Exchange keys through shared memory in blocked arrangement
            KeyBlockExchange::ScatterToBlocked(smem_storage.key_storage, keys, ranks);

            // Quit if done
            if (begin_bit >= end_bit) break;

            __syncthreads();
        }

        // Untwiddle bits if necessary
        #pragma unroll
        for (int KEY = 0; KEY < ITEMS_PER_THREAD; KEY++)
        {
            unsigned_keys[KEY] = KeyTraits::TwiddleOut(unsigned_keys[KEY]);
        }
    }


    /**
     * \brief Performs a radix sort across a [<em>blocked arrangement</em>](index.html#sec3sec3) of keys, leaving them in a [<em>striped arrangement</em>](index.html#sec3sec3).
     *
     * \smemreuse
     */
    static __device__ __forceinline__ void SortBlockedToStriped(
        SmemStorage         &smem_storage,                      ///< [in] Shared reference to opaque SmemStorage layout
        KeyType             (&keys)[ITEMS_PER_THREAD],          ///< [in-out] Keys to sort
        unsigned int        begin_bit = 0,                      ///< [in] <b>[optional]</b> The beginning (least-significant) bit index needed for key comparison
        const unsigned int  &end_bit = sizeof(KeyType) * 8)     ///< [in] <b>[optional]</b> The past-the-end (most-significant) bit index needed for key comparison
    {
        UnsignedBits (&unsigned_keys)[ITEMS_PER_THREAD] =
            reinterpret_cast<UnsignedBits (&)[ITEMS_PER_THREAD]>(keys);

        // Twiddle bits if necessary
        #pragma unroll
        for (int KEY = 0; KEY < ITEMS_PER_THREAD; KEY++)
        {
            unsigned_keys[KEY] = KeyTraits::TwiddleIn(unsigned_keys[KEY]);
        }

        // Radix sorting passes
        while (true)
        {
            // Rank the blocked keys
            unsigned int ranks[ITEMS_PER_THREAD];
            BlockRadixRank::RankKeys(smem_storage.ranking_storage, unsigned_keys, ranks, begin_bit);
            begin_bit += RADIX_BITS;

            __syncthreads();

            // Check if this is the last pass
            if (begin_bit >= end_bit)
            {
                // Last pass exchanges keys through shared memory in striped arrangement
                KeyBlockExchange::ScatterToStriped(smem_storage.key_storage, keys, ranks);

                // Quit
                break;
            }

            // Exchange keys through shared memory in blocked arrangement
            KeyBlockExchange::ScatterToBlocked(
                smem_storage.key_storage,
                keys,
                ranks);

            __syncthreads();
        }

        // Untwiddle bits if necessary
        #pragma unroll
        for (int KEY = 0; KEY < ITEMS_PER_THREAD; KEY++)
        {
            unsigned_keys[KEY] = KeyTraits::TwiddleOut(unsigned_keys[KEY]);
        }
    }


    /**
     * \brief Performs a radix sort across a [<em>striped arrangement</em>](index.html#sec3sec3) of keys.
     *
     * \smemreuse
     */
    static __device__ __forceinline__ void SortStriped(
        SmemStorage         &smem_storage,                      ///< [in] Shared reference to opaque SmemStorage layout
        KeyType             (&keys)[ITEMS_PER_THREAD],          ///< [in-out] Keys to sort
        unsigned int        begin_bit = 0,                      ///< [in] <b>[optional]</b> The beginning (least-significant) bit index needed for key comparison
        const unsigned int  &end_bit = sizeof(KeyType) * 8)     ///< [in] <b>[optional]</b> The past-the-end (most-significant) bit index needed for key comparison
    {
        // Transpose keys from striped to blocked arrangement
        KeyBlockExchange::StripedToBlocked(smem_storage.key_storage, keys);

        __syncthreads();

        // Sort blocked-to-striped
        SortBlockedToStriped(smem_storage, keys, begin_bit, end_bit);
    }

    //@}  end member group
    /******************************************************************//**
     * \name Key-value pair sorting
     *********************************************************************/
    //@{

    /**
     * \brief Performs a radix sort across a [<em>blocked arrangement</em>](index.html#sec3sec3) of keys and values.
     *
     * \smemreuse
     */
    static __device__ __forceinline__ void SortBlocked(
        SmemStorage         &smem_storage,                      ///< [in] Shared reference to opaque SmemStorage layout
        KeyType             (&keys)[ITEMS_PER_THREAD],          ///< [in-out] Keys to sort
        ValueType           (&values)[ITEMS_PER_THREAD],        ///< [in-out] Values to sort
        unsigned int        begin_bit = 0,                      ///< [in] <b>[optional]</b> The beginning (least-significant) bit index needed for key comparison
        const unsigned int  &end_bit = sizeof(KeyType) * 8)     ///< [in] <b>[optional]</b> The past-the-end (most-significant) bit index needed for key comparison
    {
        UnsignedBits (&unsigned_keys)[ITEMS_PER_THREAD] =
            reinterpret_cast<UnsignedBits (&)[ITEMS_PER_THREAD]>(keys);

        // Twiddle bits if necessary
        #pragma unroll
        for (int KEY = 0; KEY < ITEMS_PER_THREAD; KEY++)
        {
            unsigned_keys[KEY] = KeyTraits::TwiddleIn(unsigned_keys[KEY]);
        }

        // Radix sorting passes
        while (true)
        {
            // Rank the blocked keys
            unsigned int ranks[ITEMS_PER_THREAD];
            BlockRadixRank::RankKeys(smem_storage.ranking_storage, unsigned_keys, ranks, begin_bit);
            begin_bit += RADIX_BITS;

            __syncthreads();

            // Exchange keys through shared memory in blocked arrangement
            KeyBlockExchange::ScatterToBlocked(smem_storage.key_storage, keys, ranks);

            __syncthreads();

            // Exchange values through shared memory in blocked arrangement
            ValueBlockExchange::ScatterToBlocked(smem_storage.value_storage, values, ranks);

            // Quit if done
            if (begin_bit >= end_bit) break;

            __syncthreads();
        }

        // Untwiddle bits if necessary
        #pragma unroll
        for (int KEY = 0; KEY < ITEMS_PER_THREAD; KEY++)
        {
            unsigned_keys[KEY] = KeyTraits::TwiddleOut(unsigned_keys[KEY]);
        }
    }


    /**
     * \brief Performs a radix sort across a [<em>blocked arrangement</em>](index.html#sec3sec3) of keys and values, leaving them in a [<em>striped arrangement</em>](index.html#sec3sec3).
     *
     * \smemreuse
     */
    static __device__ __forceinline__ void SortBlockedToStriped(
        SmemStorage         &smem_storage,                      ///< [in] Shared reference to opaque SmemStorage layout
        KeyType             (&keys)[ITEMS_PER_THREAD],          ///< [in-out] Keys to sort
        ValueType           (&values)[ITEMS_PER_THREAD],        ///< [in-out] Values to sort
        unsigned int        begin_bit = 0,                      ///< [in] <b>[optional]</b> The beginning (least-significant) bit index needed for key comparison
        const unsigned int  &end_bit = sizeof(KeyType) * 8)     ///< [in] <b>[optional]</b> The past-the-end (most-significant) bit index needed for key comparison
    {
        UnsignedBits (&unsigned_keys)[ITEMS_PER_THREAD] =
            reinterpret_cast<UnsignedBits (&)[ITEMS_PER_THREAD]>(keys);

        // Twiddle bits if necessary
        #pragma unroll
        for (int KEY = 0; KEY < ITEMS_PER_THREAD; KEY++)
        {
            unsigned_keys[KEY] = KeyTraits::TwiddleIn(unsigned_keys[KEY]);
        }

        // Radix sorting passes
        while (true)
        {
            // Rank the blocked keys
            unsigned int ranks[ITEMS_PER_THREAD];
            BlockRadixRank::RankKeys(smem_storage.ranking_storage, unsigned_keys, ranks, begin_bit);
            begin_bit += RADIX_BITS;

            __syncthreads();

            // Check if this is the last pass
            if (begin_bit >= end_bit)
            {
                // Last pass exchanges keys through shared memory in striped arrangement
                KeyBlockExchange::ScatterToStriped(smem_storage.key_storage, keys, ranks);

                __syncthreads();

                // Last pass exchanges through shared memory in striped arrangement
                ValueBlockExchange::ScatterToStriped(smem_storage.value_storage, values, ranks);

                // Quit
                break;
            }

            // Exchange keys through shared memory in blocked arrangement
            KeyBlockExchange::ScatterToBlocked(smem_storage.key_storage, keys, ranks);

            __syncthreads();

            // Exchange values through shared memory in blocked arrangement
            ValueBlockExchange::ScatterToBlocked(smem_storage.value_storage, values, ranks);

            __syncthreads();
        }

        // Untwiddle bits if necessary
        #pragma unroll
        for (int KEY = 0; KEY < ITEMS_PER_THREAD; KEY++)
        {
            unsigned_keys[KEY] = KeyTraits::TwiddleOut(unsigned_keys[KEY]);
        }
    }


    /**
     * \brief Performs a radix sort across a [<em>striped arrangement</em>](index.html#sec3sec3) of keys and values.
     *
     * \smemreuse
     */
    static __device__ __forceinline__ void SortStriped(
        SmemStorage         &smem_storage,                      ///< [in] Shared reference to opaque SmemStorage layout
        KeyType             (&keys)[ITEMS_PER_THREAD],          ///< [in-out] Keys to sort
        ValueType           (&values)[ITEMS_PER_THREAD],        ///< [in-out] Values to sort
        unsigned int        begin_bit = 0,                      ///< [in] <b>[optional]</b> The beginning (least-significant) bit index needed for key comparison
        const unsigned int  &end_bit = sizeof(KeyType) * 8)     ///< [in] <b>[optional]</b> The past-the-end (most-significant) bit index needed for key comparison
    {
        // Transpose keys from striped to blocked arrangement
        KeyBlockExchange::StripedToBlocked(smem_storage.key_storage, keys);

        __syncthreads();

        // Transpose values from striped to blocked arrangement
        ValueBlockExchange::StripedToBlocked(smem_storage.value_storage, values);

        __syncthreads();

        // Sort blocked-to-striped
        SortBlockedToStriped(smem_storage, keys, values, begin_bit, end_bit);
    }


    /** @} */   // Key-value pair sorting

};

/** @} */       // BlockModule

}               // CUB namespace
CUB_NS_POSTFIX  // Optional outer namespace(s)

