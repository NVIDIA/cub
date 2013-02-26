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
 * The cub::BlockRadixSort type provides variants of parallel radix sorting of unsigned numeric types across threads within a threadblock.
 */


#pragma once

#include "../ns_wrapper.cuh"
#include "../device_props.cuh"
#include "../type_utils.cuh"
#include "block_exchange.cuh"
#include "block_radix_rank.cuh"

CUB_NS_PREFIX

/// CUB namespace
namespace cub {

/**
 * \addtogroup SimtCoop
 * @{
 */

/**
 * \brief The BlockRadixSort type provides variants of parallel radix sorting of unsigned numeric types across threads within a threadblock.  ![](sorting_logo.png)
 *
 * <b>Overview</b>
 * \par
 * The <em>radix sort</em> method relies upon a positional representation for
 * keys, i.e., each key is comprised of an ordered sequence of numeral symbols
 * (i.e., digits) specified from least-significant to most-significant.  For a
 * given input sequence of keys and a set of rules specifying a total ordering
 * of the symbolic alphabet, the radix sorting method produces a lexicographic
 * ordering of those keys.
 *
 * \par
 * BlockRadixSort accommodates the following arrangements of data items among threads:
 * -# <b><em>blocked</em> arrangement</b>.  The aggregate tile of items is partitioned
 *   evenly across threads in "blocked" fashion with thread<sub><em>i</em></sub>
 *   owning the <em>i</em><sup>th</sup> segment of consecutive elements.
 * -# <b><em>striped</em> arrangement</b>.  The aggregate tile of items is partitioned across
 *   threads in "striped" fashion, i.e., the \p ITEMS_PER_THREAD items owned by
 *   each thread have logical stride \p BLOCK_THREADS between them.
 *
 * \tparam KeyType              Key type
 * \tparam BLOCK_THREADS          The threadblock size in threads
 * \tparam ITEMS_PER_THREAD     The number of items per thread
 * \tparam ValueType            <b>[optional]</b> Value type (default: cub::NullType)
 * \tparam RADIX_BITS           <b>[optional]</b> The number of radix bits per digit place (default: 5 bits)
 * \tparam SMEM_CONFIG          <b>[optional]</b> Shared memory bank mode (default: \p cudaSharedMemBankSizeFourByte)
 *
 * <b>Performance Features and Considerations</b>
 * \par
 * - After any BlockRadixSort operation, a subsequent threadblock barrier (<tt>__syncthreads()</tt>) is
 *   required if the supplied BlockRadixSort::SmemStorage is to be reused/repurposed by the threadblock.
 * - Keys must be in a form suitable for radix ranking (i.e., unsigned integer types).
 * - The operations are most efficient (lowest instruction overhead) when:
 *      - \p BLOCK_THREADS is a multiple of the architecture's warp size
 *
 * <b>Algorithm</b>
 * \par
 * These parallel radix sorting variants have <em>O</em>(<em>n</em>) work complexity and are implemented in XXX phases:
 * -# blah
 * -# blah
 *
 * <b>Examples</b>
 * \par
 * - <b>Example 1:</b> Simple radix sort of 32-bit integer keys (128 threads, 4 keys per thread, blocked arrangement)
 *      \code
 *      #include <cub.cuh>
 *
 *      __global__ void SomeKernel(...)
 *      {
 *          // Parameterize a BlockRadixSort type for use in the current execution context
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
 * - <b>Example 2:</b> Lower 20-bit key-value radix sort of 32-bit integer keys and fp values (striped arrangement)
 *      \code
 *      #include <cub.cuh>
 *
 *      template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
 *      __global__ void SomeKernel(...)
 *      {
 *          // Parameterize a BlockRadixSort type for use in the current execution context
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
    typename                ValueType = NullType,
    int                     RADIX_BITS = 5,
    cudaSharedMemConfig     SMEM_CONFIG = cudaSharedMemBankSizeFourByte>
class BlockRadixSort
{
    //---------------------------------------------------------------------
    // Type definitions and constants
    //---------------------------------------------------------------------

private:

    /// BlockRadixRank utility type
    typedef BlockRadixRank<BLOCK_THREADS, RADIX_BITS, SMEM_CONFIG>      BlockRadixRank;

    /// BlockExchange utility type for keys
    typedef BlockExchange<KeyType, BLOCK_THREADS, ITEMS_PER_THREAD>     KeyBlockExchange;

    /// BlockExchange utility type for values
    typedef BlockExchange<ValueType, BLOCK_THREADS, ITEMS_PER_THREAD>   ValueBlockExchange;

    /// Shared memory storage layout type
    struct SmemStorage
    {
        union
        {
            typename BlockRadixRank::SmemStorage          ranking_storage;
            typename KeyBlockExchange::SmemStorage        key_storage;
            typename ValueBlockExchange::SmemStorage      value_storage;
        };
    };

public:

    /// The operations exposed by BlockRadixSort require shared memory of this
    /// type.  This opaque storage can be allocated directly using the
    /// <tt>__shared__</tt> keyword.  Alternatively, it can be aliased to
    /// externally allocated shared memory or <tt>union</tt>'d with other types
    /// to facilitate shared memory reuse.
    typedef SmemStorage SmemStorage;


    /******************************************************************//**
     * \name Keys-only sorting
     *********************************************************************/
    //@{

    /**
     * \brief Performs a threadblock-wide radix sort over a <em>blocked</em> arrangement of keys.
     *
     * \smemreuse
     */
    static __device__ __forceinline__ void SortBlocked(
        SmemStorage         &smem_storage,                      ///< [in] Shared reference to opaque SmemStorage layout
        KeyType             (&keys)[ITEMS_PER_THREAD],          ///< [in-out] Keys to sort
        unsigned int        begin_bit = 0,                      ///< [in] <b>[optional]</b> The beginning (least-significant) bit index needed for key comparison
        const unsigned int  &end_bit = sizeof(KeyType) * 8)     ///< [in] <b>[optional]</b> The past-the-end (most-significant) bit index needed for key comparison
    {
        // Radix sorting passes
        while (true)
        {
            // Rank the blocked keys
            unsigned int ranks[ITEMS_PER_THREAD];
            BlockRadixRank::RankKeys(smem_storage.ranking_storage, keys, ranks, begin_bit);
            begin_bit += RADIX_BITS;

            __syncthreads();

            // Exchange keys through shared memory in blocked arrangement
            KeyBlockExchange::ScatterToBlocked(smem_storage.key_storage, keys, ranks);

            // Quit if done
            if (begin_bit >= end_bit) break;

            __syncthreads();
        }
    }


    /**
     * \brief Performs a radix sort across a <em>blocked</em> arrangement of keys, leaving them in a <em>striped</em> arrangement.
     *
     * \smemreuse
     */
    static __device__ __forceinline__ void SortBlockedToStriped(
        SmemStorage         &smem_storage,                      ///< [in] Shared reference to opaque SmemStorage layout
        KeyType             (&keys)[ITEMS_PER_THREAD],          ///< [in-out] Keys to sort
        unsigned int        begin_bit = 0,                      ///< [in] <b>[optional]</b> The beginning (least-significant) bit index needed for key comparison
        const unsigned int  &end_bit = sizeof(KeyType) * 8)     ///< [in] <b>[optional]</b> The past-the-end (most-significant) bit index needed for key comparison
    {
        // Radix sorting passes
        while (true)
        {
            // Rank the blocked keys
            unsigned int ranks[ITEMS_PER_THREAD];
            BlockRadixRank::RankKeys(smem_storage.ranking_storage, keys, ranks, begin_bit);
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
    }


    /**
     * \brief Performs a radix sort across a <em>striped</em> arrangement of keys.
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

    //@}
    /******************************************************************//**
     * \name Key-value pair sorting
     *********************************************************************/
    //@{

    /**
     * \brief Performs a radix sort across a <em>blocked</em> arrangement of keys and values.
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
        // Radix sorting passes
        while (true)
        {
            // Rank the blocked keys
            unsigned int ranks[ITEMS_PER_THREAD];
            BlockRadixRank::RankKeys(smem_storage.ranking_storage, keys, ranks, begin_bit);
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
    }


    /**
     * \brief Performs a radix sort across a <em>blocked</em> arrangement of keys and values, leaving them in a <em>striped</em> arrangement.
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
        // Radix sorting passes
        while (true)
        {
            // Rank the blocked keys
            unsigned int ranks[ITEMS_PER_THREAD];
            BlockRadixRank::RankKeys(smem_storage.ranking_storage, keys, ranks, begin_bit);
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
    }


    /**
     * \brief Performs a radix sort across a <em>striped</em> arrangement of keys and values.
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

/** @} */       // SimtCoop

} // namespace cub
CUB_NS_POSTFIX
