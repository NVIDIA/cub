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
 * The cub::BlockRadixRank type provides operations for raking unsigned integer types across threads within a threadblock
 */

#pragma once

#include "../device_props.cuh"
#include "../type_utils.cuh"
#include "../ptx_intrinsics.cuh"
#include "../thread/thread_reduce.cuh"
#include "../thread/thread_scan.cuh"
#include "../ns_wrapper.cuh"


CUB_NS_PREFIX

/// CUB namespace
namespace cub {

/**
 * \addtogroup SimtCoop
 * @{
 */

/**
 * \brief The cub::BlockRadixRank type provides operations for raking unsigned integer types across threads within a threadblock.
 *
 * <B>Overview</b>
 * \par
 * Blah...
 *
 * \tparam BLOCK_THREADS          The threadblock size in threads
 * \tparam RADIX_BITS           <b>[optional]</b> The number of radix bits per digit place (default: 5 bits)
 * \tparam SMEM_CONFIG          <b>[optional]</b> Shared memory bank mode (default: \p cudaSharedMemBankSizeFourByte)
 *
 * <b>Performance Features and Considerations</b>
 * \par
 * - Keys must be in a form suitable for radix ranking (i.e., unsigned bits).
 * - After any operation, a subsequent threadblock barrier (<tt>__syncthreads()</tt>) is
 *   required if the supplied BlockRadixRank::SmemStorage is to be reused/repurposed by the threadblock.
 * - Blah...
 *
 * <b>Algorithm</b>
 * \par
 * These parallel radix ranking variants have <em>O</em>(<em>n</em>) work complexity and are implemented in XXX phases:
 * -# blah
 * -# blah
 *
 * <b>Examples</b>
 * \par
 * - <b>Example 1:</b> Simple radix rank of 32-bit integer keys
 *      \code
 *      #include <cub.cuh>
 *
 *      template <int BLOCK_THREADS>
 *      __global__ void SomeKernel(...)
 *      {
 *
 *      \endcode
 */
template <
    int                     BLOCK_THREADS,
    int                     RADIX_BITS,
    cudaSharedMemConfig     SMEM_CONFIG = cudaSharedMemBankSizeFourByte>    // Shared memory bank size
class BlockRadixRank
{
private:

    //---------------------------------------------------------------------
    // Type definitions and constants
    //---------------------------------------------------------------------

    // Integer type for digit counters (to be packed into words of type PackedCounters)
    typedef unsigned short DigitCounter;

    // Integer type for packing DigitCounters into columns of shared memory banks
    typedef typename If<(SMEM_CONFIG == cudaSharedMemBankSizeEightByte),
        unsigned long long,
        unsigned int>::Type PackedCounter;

    enum {
        RADIX_DIGITS                 = 1 << RADIX_BITS,

        LOG_WARP_THREADS             = DeviceProps::LOG_WARP_THREADS,
        WARP_THREADS                = 1 << LOG_WARP_THREADS,
        WARPS                        = (BLOCK_THREADS + WARP_THREADS - 1) / WARP_THREADS,

        BYTES_PER_COUNTER            = sizeof(DigitCounter),
        LOG_BYTES_PER_COUNTER        = Log2<BYTES_PER_COUNTER>::VALUE,

        PACKING_RATIO                = sizeof(PackedCounter) / sizeof(DigitCounter),
        LOG_PACKING_RATIO            = Log2<PACKING_RATIO>::VALUE,

        LOG_COUNTER_LANES            = CUB_MAX((RADIX_BITS - LOG_PACKING_RATIO), 0),                // Always at least one lane
        COUNTER_LANES                = 1 << LOG_COUNTER_LANES,

        // The number of packed counters per thread (plus one for padding)
        RAKING_SEGMENT                = COUNTER_LANES + 1,

        LOG_SMEM_BANKS                = DeviceProps::LOG_SMEM_BANKS,
        SMEM_BANKS                    = 1 << LOG_SMEM_BANKS,
    };

public:

    /**
     * Shared memory storage layout
     */
    struct SmemStorage
    {
        // Storage for scanning local ranks
        volatile PackedCounter        warpscan[WARPS][WARP_THREADS * 3 / 2];

        union
        {
            DigitCounter            digit_counters[COUNTER_LANES + 1][BLOCK_THREADS][PACKING_RATIO];
            PackedCounter            raking_grid[BLOCK_THREADS][RAKING_SEGMENT];
        };
    };

private :

    //---------------------------------------------------------------------
    // Template iteration structures.  (Regular iteration cannot always be
    // unrolled due to conditionals or ABI procedure calls within
    // functors).
    //---------------------------------------------------------------------

    // General template iteration
    template <int COUNT, int MAX>
    struct Iterate
    {
        /**
         * Decode keys.  Decodes the radix digit from the current digit place
         * and increments the thread's corresponding counter in shared
         * memory for that digit.
         *
         * Saves both (1) the prior value of that counter (the key's
         * thread-local exclusive prefix sum for that digit), and (2) the shared
         * memory offset of the counter (for later use).
         */
        template <typename UnsignedBits, int KEYS_PER_THREAD>
        static __device__ __forceinline__ void DecodeKeys(
            SmemStorage        &smem_storage,                       // Shared memory storage
            UnsignedBits     (&keys)[KEYS_PER_THREAD],              // Key to decode
            DigitCounter     (&thread_prefixes)[KEYS_PER_THREAD],   // Prefix counter value (out parameter)
            DigitCounter*     (&digit_counters)[KEYS_PER_THREAD],   // Counter smem offset (out parameter)
            unsigned int     current_bit)                           // The least-significant bit position of the current digit to extract
        {
            // Add in sub-counter offset
            UnsignedBits sub_counter = BFE(keys[COUNT], current_bit + LOG_COUNTER_LANES, LOG_PACKING_RATIO);

            // Add in row offset
            UnsignedBits row_offset = BFE(keys[COUNT], current_bit, LOG_COUNTER_LANES);

            // Pointer to smem digit counter
            digit_counters[COUNT] = &smem_storage.digit_counters[row_offset][threadIdx.x][sub_counter];

            // Load thread-exclusive prefix
            thread_prefixes[COUNT] = *digit_counters[COUNT];

            // Store inclusive prefix
            *digit_counters[COUNT] = thread_prefixes[COUNT] + 1;

            // Iterate next key
            Iterate<COUNT + 1, MAX>::DecodeKeys(smem_storage, keys, thread_prefixes, digit_counters, current_bit);
        }


        // Termination
        template <int KEYS_PER_THREAD>
        static __device__ __forceinline__ void UpdateRanks(
            SmemStorage        &smem_storage,                       // Shared memory storage
            unsigned int     (&ranks)[KEYS_PER_THREAD],             // Local ranks (out parameter)
            DigitCounter     (&thread_prefixes)[KEYS_PER_THREAD],   // Prefix counter value
            DigitCounter*     (&digit_counters)[KEYS_PER_THREAD])   // Counter smem offset
        {
            // Add in threadblock exclusive prefix
            ranks[COUNT] = thread_prefixes[COUNT] + *digit_counters[COUNT];

            // Iterate next key
            Iterate<COUNT + 1, MAX>::UpdateRanks(smem_storage, ranks, thread_prefixes, digit_counters);
        }
    };


    // Termination
    template <int MAX>
    struct Iterate<MAX, MAX>
    {
        // DecodeKeys
        template <typename UnsignedBits, int KEYS_PER_THREAD>
        static __device__ __forceinline__ void DecodeKeys(
            SmemStorage        &smem_storage,
            UnsignedBits     (&keys)[KEYS_PER_THREAD],
            DigitCounter     (&thread_prefixes)[KEYS_PER_THREAD],
            DigitCounter*    (&digit_counters)[KEYS_PER_THREAD],
            unsigned int     current_bit) {}


        // UpdateRanks
        template <int KEYS_PER_THREAD>
        static __device__ __forceinline__ void UpdateRanks(
            SmemStorage        &smem_storage,
            unsigned int     (&ranks)[KEYS_PER_THREAD],
            DigitCounter     (&thread_prefixes)[KEYS_PER_THREAD],
            DigitCounter*    (&digit_counters)[KEYS_PER_THREAD]) {}
    };


    //---------------------------------------------------------------------
    // Utility methods
    //---------------------------------------------------------------------

    /**
     * Reset shared memory digit counters
     */
    static __device__ __forceinline__ void ResetCounters(SmemStorage &smem_storage)
    {
        // Reset shared memory digit counters
        #pragma unroll
        for (int LANE = 0; LANE < COUNTER_LANES + 1; LANE++)
        {
            *((PackedCounter*) smem_storage.digit_counters[LANE][threadIdx.x]) = 0;
        }
    }


    /**
     * Scan shared memory digit counters.
     */
    static __device__ __forceinline__ void ScanCounters(SmemStorage &smem_storage)
    {
        PackedCounter *raking_segment = smem_storage.raking_grid[threadIdx.x];

        // Upsweep reduce
        PackedCounter raking_partial         = ThreadReduce<RAKING_SEGMENT>(raking_segment, Sum<PackedCounter>());

        int warp_id                         = threadIdx.x >> LOG_WARP_THREADS;
        int tid                             = threadIdx.x & (WARP_THREADS - 1);
        volatile PackedCounter *warpscan     = &smem_storage.warpscan[warp_id][(WARP_THREADS / 2) + tid];

        // Initialize warpscan identity regions
        warpscan[0 - (WARP_THREADS / 2)] = 0;

        // Warpscan
        PackedCounter partial = raking_partial;
        warpscan[0] = partial;

        #pragma unroll
        for (int STEP = 0; STEP < LOG_WARP_THREADS; STEP++)
        {
            partial += warpscan[0 - (1 << STEP)];
            warpscan[0] = partial;
        }

        // TestBarrier
        __syncthreads();

        // Scan across warpscan totals
        PackedCounter warpscan_totals = 0;

        #pragma unroll
        for (int WARP = 0; WARP < WARPS; WARP++)
        {
            // Add totals from all previous warpscans into our partial
            PackedCounter warpscan_total = smem_storage.warpscan[WARP][(WARP_THREADS * 3 / 2) - 1];
            if (warp_id == WARP) {
                partial += warpscan_totals;
            }

            // Increment warpscan totals
            warpscan_totals += warpscan_total;
        }

        // Add lower totals from all warpscans into partial's upper
        #pragma unroll
        for (int PACKED = 1; PACKED < PACKING_RATIO; PACKED++)
        {
            partial += warpscan_totals << (sizeof(DigitCounter) * 8 * PACKED);
        }

        // Downsweep scan with exclusive partial
        PackedCounter exclusive_partial = partial - raking_partial;

        exclusive_partial = ThreadScanExclusive<RAKING_SEGMENT>(
            raking_segment,
            raking_segment,
            Sum<PackedCounter>(),
            exclusive_partial);
    }

public:

    //---------------------------------------------------------------------
    // Interface
    //---------------------------------------------------------------------

    /**
     * Rank keys.
     */
    template <
        typename UnsignedBits,
        int KEYS_PER_THREAD>
    static __device__ __forceinline__ void RankKeys(
        SmemStorage     &smem_storage,                      // Shared memory storage
        UnsignedBits    (&keys)[KEYS_PER_THREAD],           // Keys for this tile
        unsigned int    (&ranks)[KEYS_PER_THREAD],          // For each key, the local rank within the tile
        unsigned int    current_bit)                        // The least-significant bit position of the current digit to extract
    {
        DigitCounter    thread_prefixes[KEYS_PER_THREAD];   // For each key, the count of previous keys in this tile having the same digit
        DigitCounter*     digit_counters[KEYS_PER_THREAD];  // For each key, the byte-offset of its corresponding digit counter in smem

        // Reset shared memory digit counters
        ResetCounters(smem_storage);

        // Decode keys and update digit counters
        Iterate<0, KEYS_PER_THREAD>::DecodeKeys(smem_storage, keys, thread_prefixes, digit_counters, current_bit);

        __syncthreads();

        // Scan shared memory counters
        ScanCounters(smem_storage);

        __syncthreads();

        // Extract the local ranks of each key
        Iterate<0, KEYS_PER_THREAD>::UpdateRanks(smem_storage, ranks, thread_prefixes, digit_counters);
    }


    /**
     * Rank keys.  For the lower RADIX_DIGITS threads, digit counts for each
     * digit are provided for the corresponding thread-id.
     */
    template <
        typename UnsignedBits,
        int KEYS_PER_THREAD>
    static __device__ __forceinline__ void RankKeys(
        SmemStorage     &smem_storage,                      // Shared memory storage
        UnsignedBits    (&keys)[KEYS_PER_THREAD],           // Keys for this tile
        unsigned int    (&ranks)[KEYS_PER_THREAD],          // For each key, the local rank within the tile (out parameter)
        unsigned int    digit_prefixes[RADIX_DIGITS],
        unsigned int    current_bit)                        // The least-significant bit position of the current digit to extract
    {
        // Rank keys
        RankKeys(smem_storage, keys, ranks, current_bit);

        // Get the inclusive and exclusive digit totals corresponding to the calling thread.
        if ((BLOCK_THREADS == RADIX_DIGITS) || (threadIdx.x < RADIX_DIGITS))
        {
            // Initialize digit scan's identity value
            digit_prefixes[threadIdx.x] = 0;

            // Obtain ex/inclusive digit counts.  (Unfortunately these all reside in the
            // first counter column, resulting in unavoidable bank conflicts.)
            int counter_lane = (threadIdx.x & (COUNTER_LANES - 1));
            int sub_counter = threadIdx.x >> (LOG_COUNTER_LANES);
            digit_prefixes[threadIdx.x + 1] = smem_storage.digit_counters[counter_lane + 1][0][sub_counter];
        }
    }
};

/** @} */       // SimtCoop

} // namespace cub
CUB_NS_POSTFIX

