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
 * \brief BlockRadixSort provides variants of parallel radix sorting across a CUDA threadblock.  ![](sorting_logo.png)
 * \ingroup BlockModule
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
 * For convenience, BlockRadixSort provides alternative entrypoints that differ by:
 * - Value association (keys-only <b><em>vs.</em></b> key-value-pairs)
 *
 * \tparam Key              Key type
 * \tparam BLOCK_THREADS        The threadblock size in threads
 * \tparam ITEMS_PER_THREAD     The number of items per thread
 * \tparam Value                <b>[optional]</b> Value type (default: cub::NullType)
 * \tparam RADIX_BITS           <b>[optional]</b> The number of radix bits per digit place (default: 4 bits)
 * \tparam MEMOIZE_OUTER_SCAN   <b>[optional]</b> Whether or not to buffer outer raking scan partials to incur fewer shared memory reads at the expense of higher register pressure (default: true for architectures SM35 and newer, false otherwise).
 * \tparam INNER_SCAN_ALGORITHM <b>[optional]</b> The cub::BlockScanAlgorithm algorithm to use (default: cub::BLOCK_SCAN_WARP_SCANS)
 * \tparam SMEM_CONFIG          <b>[optional]</b> Shared memory bank mode (default: \p cudaSharedMemBankSizeFourByte)
 *
 * \par Usage Considerations
 * - After any sorting operation, a subsequent <tt>__syncthreads()</tt> barrier
 *   is required if the supplied BlockRadixSort::TempStorage is to be reused or repurposed
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
 *      - \p Key is an unsigned integral type
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
 * <em>Example 1.</em> Perform a radix sort over a tile of 512 integer keys.
 * The data are partitioned in a blocked arrangement across a block of 128
 * threads where each thread holds 4 keys.
 *      \code
 *      #include <cub/cub.cuh>
 *
 *      __global__ void SomeKernel(...)
 *      {
 *          // Parameterize BlockRadixSort for 128 threads (4 items each) on type unsigned int
 *          typedef cub::BlockRadixSort<unsigned int, 128, 4> BlockRadixSort;
 *
 *          // Declare shared memory for BlockRadixSort
 *          __shared__ typename BlockRadixSort::TempStorage temp_storage;
 *
 *          // A segment of consecutive input items per thread
 *          int keys[4];
 *
 *          // Obtain keys in blocked order
 *          ...
 *
 *          // Sort keys in ascending order
 *          BlockRadixSort(temp_storage).Sort(keys);
 *
 *      \endcode
 *
 * \par
 * <em>Example 2.</em> Perform a key-value radix sort over the lower 20-bits
 * of a tile of 32-bit integer keys paired with floating-point values.
 * The data are partitioned in a blocked arrangement across a block of BLOCK_THREADS
 * threads where each thread holds ITEMS_PER_THREAD keys.
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
 *          __shared__ typename BlockRadixSort::TempStorage temp_storage;
 *
 *          // Input keys and values per thread
 *          int keys[ITEMS_PER_THREAD];
 *          float values[ITEMS_PER_THREAD];
 *
 *          // Obtain keys and values in blocked order
 *          ...
 *
 *          // Sort pairs in ascending order (using only the lower 20 distinguishing key bits)
 *          BlockRadixSort(temp_storage).Sort(keys, values, 0, 20);
 *      }
 *
 *      \endcode
 */
template <
    typename                Key,
    int                     BLOCK_THREADS,
    int                     ITEMS_PER_THREAD,
    typename                Value                   = NullType,
    int                     RADIX_BITS              = 4,
    bool                    MEMOIZE_OUTER_SCAN      = (CUB_PTX_ARCH >= 350) ? true : false,
    BlockScanAlgorithm      INNER_SCAN_ALGORITHM    = BLOCK_SCAN_WARP_SCANS,
    cudaSharedMemConfig     SMEM_CONFIG             = cudaSharedMemBankSizeFourByte>
class BlockRadixSort
{
private:

    /******************************************************************************
     * Constants and type definitions
     ******************************************************************************/

    // Key traits and unsigned bits type
    typedef NumericTraits<Key>              Keyraits;
    typedef typename Keyraits::UnsignedBits    UnsignedBits;

    /// BlockRadixRank utility type
    typedef BlockRadixRank<BLOCK_THREADS, RADIX_BITS, MEMOIZE_OUTER_SCAN, INNER_SCAN_ALGORITHM, SMEM_CONFIG> BlockRadixRank;

    /// BlockExchange utility type for keys
    typedef BlockExchange<Key, BLOCK_THREADS, ITEMS_PER_THREAD> BlockExchangeKeys;

    /// BlockExchange utility type for values
    typedef BlockExchange<Value, BLOCK_THREADS, ITEMS_PER_THREAD> BlockExchangeValues;

    /// Shared memory storage layout type
    struct _TempStorage
    {
        union
        {
            typename BlockRadixRank::TempStorage          ranking_storage;
            typename BlockExchangeKeys::TempStorage        exchange_keys;
            typename BlockExchangeValues::TempStorage      exchange_values;
        };
    };

    /******************************************************************************
     * Utility methods
     ******************************************************************************/

    /// Internal storage allocator
    __device__ __forceinline__ _TempStorage& PrivateStorage()
    {
        __shared__ _TempStorage private_storage;
        return private_storage;
    }


    /******************************************************************************
     * Thread fields
     ******************************************************************************/

    /// Shared storage reference
    _TempStorage &temp_storage;

    /// Linear thread-id
    int linear_tid;


public:

    /// \smemstorage{BlockScan}
    typedef _TempStorage TempStorage;


    /******************************************************************//**
     * \name Collective construction
     *********************************************************************/
    //@{

    /**
     * \brief Collective constructor for 1D thread blocks using a private static allocation of shared memory as temporary storage.  Threads are identified using <tt>threadIdx.x</tt>.
     */
    __device__ __forceinline__ BlockRadixSort()
    :
        temp_storage(PrivateStorage()),
        linear_tid(threadIdx.x)
    {}


    /**
     * \brief Collective constructor for 1D thread blocks using the specified memory allocation as temporary storage.  Threads are identified using <tt>threadIdx.x</tt>.
     */
    __device__ __forceinline__ BlockRadixSort(
        TempStorage &temp_storage)             ///< [in] Reference to memory allocation having layout type TempStorage
    :
        temp_storage(temp_storage),
        linear_tid(threadIdx.x)
    {}


    /**
     * \brief Collective constructor using a private static allocation of shared memory as temporary storage.  Threads are identified using the given linear thread identifier
     */
    __device__ __forceinline__ BlockRadixSort(
        int linear_tid)                        ///< [in] A suitable 1D thread-identifier for the calling thread (e.g., <tt>(threadIdx.y * blockDim.x) + linear_tid</tt> for 2D thread blocks)
    :
        temp_storage(PrivateStorage()),
        linear_tid(linear_tid)
    {}


    /**
     * \brief Collective constructor using the specified memory allocation as temporary storage.  Threads are identified using the given linear thread identifier.
     */
    __device__ __forceinline__ BlockRadixSort(
        TempStorage &temp_storage,             ///< [in] Reference to memory allocation having layout type TempStorage
        int linear_tid)                        ///< [in] <b>[optional]</b> A suitable 1D thread-identifier for the calling thread (e.g., <tt>(threadIdx.y * blockDim.x) + linear_tid</tt> for 2D thread blocks)
    :
        temp_storage(temp_storage),
        linear_tid(linear_tid)
    {}



    //@}  end member group
    /******************************************************************//**
     * \name Sorting (blocked arrangements)
     *********************************************************************/
    //@{

    /**
     * \brief Performs a threadblock-wide radix sort over a [<em>blocked arrangement</em>](index.html#sec3sec3) of keys.
     *
     * \smemreuse
     */
    __device__ __forceinline__ void Sort(
        Key     (&keys)[ITEMS_PER_THREAD],          ///< [in-out] Keys to sort
        int     begin_bit   = 0,                    ///< [in] <b>[optional]</b> The beginning (least-significant) bit index needed for key comparison
        int     end_bit     = sizeof(Key) * 8)      ///< [in] <b>[optional]</b> The past-the-end (most-significant) bit index needed for key comparison
    {
        UnsignedBits (&unsigned_keys)[ITEMS_PER_THREAD] =
            reinterpret_cast<UnsignedBits (&)[ITEMS_PER_THREAD]>(keys);

        // Twiddle bits if necessary
        #pragma unroll
        for (int KEY = 0; KEY < ITEMS_PER_THREAD; KEY++)
        {
            unsigned_keys[KEY] = Keyraits::TwiddleIn(unsigned_keys[KEY]);
        }

        // Radix sorting passes
        while (true)
        {
            // Rank the blocked keys
            int ranks[ITEMS_PER_THREAD];
            BlockRadixRank(temp_storage.ranking_storage, linear_tid).RankKeys(unsigned_keys, ranks, begin_bit);
            begin_bit += RADIX_BITS;

            __syncthreads();

            // Exchange keys through shared memory in blocked arrangement
            BlockExchangeKeys(temp_storage.exchange_keys, linear_tid).ScatterToBlocked(keys, ranks);

            // Quit if done
            if (begin_bit >= end_bit) break;

            __syncthreads();
        }

        // Untwiddle bits if necessary
        #pragma unroll
        for (int KEY = 0; KEY < ITEMS_PER_THREAD; KEY++)
        {
            unsigned_keys[KEY] = Keyraits::TwiddleOut(unsigned_keys[KEY]);
        }
    }


    /**
     * \brief Performs a radix sort across a [<em>blocked arrangement</em>](index.html#sec3sec3) of keys and values.
     *
     * \smemreuse
     */
    __device__ __forceinline__ void Sort(
        Key     (&keys)[ITEMS_PER_THREAD],          ///< [in-out] Keys to sort
        Value   (&values)[ITEMS_PER_THREAD],        ///< [in-out] Values to sort
        int     begin_bit   = 0,                    ///< [in] <b>[optional]</b> The beginning (least-significant) bit index needed for key comparison
        int     end_bit     = sizeof(Key) * 8)      ///< [in] <b>[optional]</b> The past-the-end (most-significant) bit index needed for key comparison
    {
        UnsignedBits (&unsigned_keys)[ITEMS_PER_THREAD] =
            reinterpret_cast<UnsignedBits (&)[ITEMS_PER_THREAD]>(keys);

        // Twiddle bits if necessary
        #pragma unroll
        for (int KEY = 0; KEY < ITEMS_PER_THREAD; KEY++)
        {
            unsigned_keys[KEY] = Keyraits::TwiddleIn(unsigned_keys[KEY]);
        }

        // Radix sorting passes
        while (true)
        {
            // Rank the blocked keys
            int ranks[ITEMS_PER_THREAD];
            BlockRadixRank(temp_storage.ranking_storage, linear_tid).RankKeys(unsigned_keys, ranks, begin_bit);
            begin_bit += RADIX_BITS;

            __syncthreads();

            // Exchange keys through shared memory in blocked arrangement
            BlockExchangeKeys(temp_storage.exchange_keys, linear_tid).ScatterToBlocked(keys, ranks);

            __syncthreads();

            // Exchange values through shared memory in blocked arrangement
            BlockExchangeValues(temp_storage.exchange_values, linear_tid).ScatterToBlocked(values, ranks);

            // Quit if done
            if (begin_bit >= end_bit) break;

            __syncthreads();
        }

        // Untwiddle bits if necessary
        #pragma unroll
        for (int KEY = 0; KEY < ITEMS_PER_THREAD; KEY++)
        {
            unsigned_keys[KEY] = Keyraits::TwiddleOut(unsigned_keys[KEY]);
        }
    }


    //@}  end member group
    /******************************************************************//**
     * \name Sorting (blocked arrangement -> striped arrangement)
     *********************************************************************/
    //@{


    /**
     * \brief Performs a radix sort across a [<em>blocked arrangement</em>](index.html#sec3sec3) of keys, leaving them in a [<em>striped arrangement</em>](index.html#sec3sec3).
     *
     * \smemreuse
     */
    __device__ __forceinline__ void SortBlockedToStriped(
        Key     (&keys)[ITEMS_PER_THREAD],          ///< [in-out] Keys to sort
        int     begin_bit   = 0,                    ///< [in] <b>[optional]</b> The beginning (least-significant) bit index needed for key comparison
        int     end_bit     = sizeof(Key) * 8)      ///< [in] <b>[optional]</b> The past-the-end (most-significant) bit index needed for key comparison
    {
        UnsignedBits (&unsigned_keys)[ITEMS_PER_THREAD] =
            reinterpret_cast<UnsignedBits (&)[ITEMS_PER_THREAD]>(keys);

        // Twiddle bits if necessary
        #pragma unroll
        for (int KEY = 0; KEY < ITEMS_PER_THREAD; KEY++)
        {
            unsigned_keys[KEY] = Keyraits::TwiddleIn(unsigned_keys[KEY]);
        }

        // Radix sorting passes
        while (true)
        {
            // Rank the blocked keys
            int ranks[ITEMS_PER_THREAD];
            BlockRadixRank(temp_storage.ranking_storage, linear_tid).RankKeys(unsigned_keys, ranks, begin_bit);
            begin_bit += RADIX_BITS;

            __syncthreads();

            // Check if this is the last pass
            if (begin_bit >= end_bit)
            {
                // Last pass exchanges keys through shared memory in striped arrangement
                BlockExchangeKeys(temp_storage.exchange_keys, linear_tid).ScatterToStriped(keys, ranks);

                // Quit
                break;
            }

            // Exchange keys through shared memory in blocked arrangement
            BlockExchangeKeys(temp_storage.exchange_keys, linear_tid).ScatterToBlocked(keys, ranks);

            __syncthreads();
        }

        // Untwiddle bits if necessary
        #pragma unroll
        for (int KEY = 0; KEY < ITEMS_PER_THREAD; KEY++)
        {
            unsigned_keys[KEY] = Keyraits::TwiddleOut(unsigned_keys[KEY]);
        }
    }


    /**
     * \brief Performs a radix sort across a [<em>blocked arrangement</em>](index.html#sec3sec3) of keys and values, leaving them in a [<em>striped arrangement</em>](index.html#sec3sec3).
     *
     * \smemreuse
     */
    __device__ __forceinline__ void SortBlockedToStriped(
        Key     (&keys)[ITEMS_PER_THREAD],          ///< [in-out] Keys to sort
        Value   (&values)[ITEMS_PER_THREAD],        ///< [in-out] Values to sort
        int     begin_bit   = 0,                    ///< [in] <b>[optional]</b> The beginning (least-significant) bit index needed for key comparison
        int     end_bit     = sizeof(Key) * 8)      ///< [in] <b>[optional]</b> The past-the-end (most-significant) bit index needed for key comparison
    {
        UnsignedBits (&unsigned_keys)[ITEMS_PER_THREAD] =
            reinterpret_cast<UnsignedBits (&)[ITEMS_PER_THREAD]>(keys);

        // Twiddle bits if necessary
        #pragma unroll
        for (int KEY = 0; KEY < ITEMS_PER_THREAD; KEY++)
        {
            unsigned_keys[KEY] = Keyraits::TwiddleIn(unsigned_keys[KEY]);
        }

        // Radix sorting passes
        while (true)
        {
            // Rank the blocked keys
            int ranks[ITEMS_PER_THREAD];
            BlockRadixRank(temp_storage.ranking_storage, linear_tid).RankKeys(unsigned_keys, ranks, begin_bit);
            begin_bit += RADIX_BITS;

            __syncthreads();

            // Check if this is the last pass
            if (begin_bit >= end_bit)
            {
                // Last pass exchanges keys through shared memory in striped arrangement
                BlockExchangeKeys(temp_storage.exchange_keys, linear_tid).ScatterToStriped(keys, ranks);

                __syncthreads();

                // Last pass exchanges through shared memory in striped arrangement
                BlockExchangeValues(temp_storage.exchange_values, linear_tid).ScatterToStriped(values, ranks);

                // Quit
                break;
            }

            // Exchange keys through shared memory in blocked arrangement
            BlockExchangeKeys(temp_storage.exchange_keys, linear_tid).ScatterToBlocked(keys, ranks);

            __syncthreads();

            // Exchange values through shared memory in blocked arrangement
            BlockExchangeValues(temp_storage.exchange_values, linear_tid).ScatterToBlocked(values, ranks);

            __syncthreads();
        }

        // Untwiddle bits if necessary
        #pragma unroll
        for (int KEY = 0; KEY < ITEMS_PER_THREAD; KEY++)
        {
            unsigned_keys[KEY] = Keyraits::TwiddleOut(unsigned_keys[KEY]);
        }
    }


    //@}  end member group

};

}               // CUB namespace
CUB_NS_POSTFIX  // Optional outer namespace(s)

