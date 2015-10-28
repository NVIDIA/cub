
/******************************************************************************
 * Copyright (c) 2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2015, NVIDIA CORPORATION.  All rights reserved.
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
 * cub::DeviceRadixSort provides device-wide, parallel operations for computing a radix sort across a sequence of data items residing within device-accessible memory.
 */

#pragma once

#include <stdio.h>
#include <iterator>

#include "dispatch_radix_sort.cuh"
#include "../../agent/agent_radix_sort_upsweep.cuh"
#include "../../agent/agent_radix_sort_downsweep.cuh"
#include "../../warp/warp_scan.cuh"
#include "../../util_type.cuh"
#include "../../util_debug.cuh"
#include "../../util_device.cuh"
#include "../../util_namespace.cuh"

/// Optional outer namespace(s)
CUB_NS_PREFIX

/// CUB namespace
namespace cub {

/******************************************************************************
 * Kernel entry points
 *****************************************************************************/

/**
 * Radix sorting pass (one segment per block)
 */
template <
    typename    AgentRadixSortDownsweepPolicyT,         ///< Parameterized AgentRadixSortDownsweepPolicy tuning policy type
    bool        DESCENDING,                             ///< Whether or not the sorted-order is high-to-low
    typename    KeyT,                                   ///< Key type
    typename    ValueT,                                 ///< Value type
    typename    OffsetT>                                ///< Signed integer type for global offsets
__launch_bounds__ (int(AgentRadixSortDownsweepPolicyT::BLOCK_THREADS))
__global__ void DeviceSegmentedRadixSortKernel(
    KeyT        *d_keys_in,                             ///< [in] Input keys ping buffer
    KeyT        *d_keys_out,                            ///< [in] Output keys pong buffer
    ValueT      *d_values_in,                           ///< [in] Input values ping buffer
    ValueT      *d_values_out,                          ///< [in] Output values pong buffer
    int         *d_begin_offsets,                       ///< [in] %Device-accessible pointer to the sequence of beginning offsets of length \p num_segments, such that <tt>d_begin_offsets[i]</tt> is the first element of the <em>i</em><sup>th</sup> data segment in <tt>d_keys_*</tt> and <tt>d_values_*</tt>
    int         *d_end_offsets,                         ///< [in] %Device-accessible pointer to the sequence of ending offsets of length \p num_segments, such that <tt>d_end_offsets[i]-1</tt> is the last element of the <em>i</em><sup>th</sup> data segment in <tt>d_keys_*</tt> and <tt>d_values_*</tt>.  If <tt>d_end_offsets[i]-1</tt> <= <tt>d_begin_offsets[i]</tt>, the <em>i</em><sup>th</sup> is considered empty.
    int         num_segments,                           ///< [in] The number of segments that comprise the sorting data
    int         current_bit,                            ///< [in] Bit position of current radix digit
    int         pass_bits)                              ///< [in] Number of bits of current radix digit
{
    //
    // Constants
    //

    enum
    {
        BLOCK_THREADS       = AgentRadixSortDownsweepPolicyT::BLOCK_THREADS,
        ITEMS_PER_THREAD    = AgentRadixSortDownsweepPolicyT::ITEMS_PER_THREAD,
        RADIX_BITS          = AgentRadixSortDownsweepPolicyT::RADIX_BITS,
        TILE_ITEMS          = BLOCK_THREADS * ITEMS_PER_THREAD,
        RADIX_DIGITS        = 1 << RADIX_BITS,
        KEYS_ONLY           = Equals<ValueT, NullType>::VALUE,
    };

    static const BlockScanAlgorithm     SCAN_ALGORITHM  = AgentRadixSortDownsweepPolicyT::INNER_SCAN_ALGORITHM;
    static const BlockLoadAlgorithm     LOAD_ALGORITHM  = AgentRadixSortDownsweepPolicyT::LOAD_ALGORITHM;
    static const CacheLoadModifier      LOAD_MODIFIER   = AgentRadixSortDownsweepPolicyT::LOAD_MODIFIER;

    //
    // Parameterize collective types
    //

    // Upsweep policy
    typedef AgentRadixSortUpsweepPolicy <BLOCK_THREADS, ITEMS_PER_THREAD, LOAD_MODIFIER, RADIX_BITS> UpsweepPolicyT;

    // Upsweep type
    typedef AgentRadixSortUpsweep<UpsweepPolicyT, KeyT, OffsetT> BlockUpsweepT;

    // Digit-scan type
    typedef WarpScan<OffsetT, RADIX_DIGITS> DigitScanT;

    // Downsweep type
    typedef AgentRadixSortDownsweep<AgentRadixSortDownsweepPolicyT, DESCENDING, KeyT, ValueT, OffsetT> BlockDownsweepT;

    //
    // Shared memory storage
    //

    __shared__ union
    {
        typename BlockUpsweepT::TempStorage     upsweep;
        typename DigitScanT::TempStorage        scan;
        typename BlockDownsweepT::TempStorage   downsweep;

    } temp_storage;

    //
    // Process input tiles
    //

    OffsetT segment_begin   = d_begin_offsets[blockIdx.x];
    OffsetT segment_end     = d_end_offsets[blockIdx.x];
    OffsetT num_items       = segment_end - segment_begin;

    // Check if empty segment
    if (num_items <= 0)
        return;

    // Upsweep
    OffsetT bin_count;      // The count of each digit value in this pass (valid in the first RADIX_DIGITS threads)
    BlockUpsweepT(temp_storage.upsweep, d_keys_in, current_bit, pass_bits).ProcessRegion(
        segment_begin, segment_end, bin_count);

    __syncthreads();

    // Scan
    OffsetT bin_offset;     // The global scatter base offset for each digit value in this pass (valid in the first RADIX_DIGITS threads)
    DigitScanT(temp_storage.scan).ExclusiveSum(bin_count, bin_offset);

    __syncthreads();

    // Downsweep
    BlockDownsweepT(temp_storage.downsweep, num_items, bin_offset, d_keys_in, d_keys_out, d_values_in, d_values_out, current_bit, pass_bits).ProcessRegion(
        segment_begin, segment_end);
}




/******************************************************************************
 * Dispatch
 ******************************************************************************/

/**
 * Utility class for dispatching the appropriately-tuned kernels for DeviceRadixSort
 */
template <
    bool     DESCENDING,    ///< Whether or not the sorted-order is high-to-low
    bool     ALT_STORAGE,   ///< Whether or not we need a third buffer to either (a) prevent modification to input buffer, or (b) place output into a specific buffer (instead of a pointer to one of the double buffers)
    typename KeyT,          ///< Key type
    typename ValueT,        ///< Value type
    typename OffsetT>       ///< Signed integer type for global offsets
struct DispatchSegmentedRadixSort : DispatchRadixSort<DESCENDING, ALT_STORAGE, KeyT, ValueT, OffsetT>
{
    /******************************************************************************
     * Constants
     ******************************************************************************/

    enum
    {
        // Whether this is a keys-only (or key-value) sort
        KEYS_ONLY = (Equals<ValueT, NullType>::VALUE),

        // Relative size of key type to a 4-byte word
        SCALE_FACTOR_4B = (CUB_MAX(sizeof(KeyT), sizeof(ValueT)) + 3) / 4,
    };

    /******************************************************************************
     * Tuning policies
     ******************************************************************************/

    /// SM52
    struct Policy520
    {
        enum {
            PRIMARY_RADIX_BITS      = 5,
            ALT_RADIX_BITS          = PRIMARY_RADIX_BITS - 1,
        };

        typedef AgentRadixSortUpsweepPolicy <256,   CUB_MAX(1, 16 / SCALE_FACTOR_4B), LOAD_DEFAULT, PRIMARY_RADIX_BITS> UpsweepPolicy;
        typedef AgentRadixSortUpsweepPolicy <256,   CUB_MAX(1, 16 / SCALE_FACTOR_4B), LOAD_DEFAULT, ALT_RADIX_BITS> AltUpsweepPolicy;

        typedef AgentScanPolicy <512, 23, BLOCK_LOAD_WARP_TRANSPOSE, LOAD_DEFAULT, BLOCK_STORE_WARP_TRANSPOSE, BLOCK_SCAN_RAKING_MEMOIZE> ScanPolicy;

        typedef AgentRadixSortDownsweepPolicy <256, CUB_MAX(1, 16 / SCALE_FACTOR_4B),  BLOCK_LOAD_DIRECT, LOAD_LDG, true, BLOCK_SCAN_RAKING_MEMOIZE, RADIX_SORT_SCATTER_TWO_PHASE, cudaSharedMemBankSizeFourByte, PRIMARY_RADIX_BITS> DownsweepPolicy;
        typedef AgentRadixSortDownsweepPolicy <256, CUB_MAX(1, 16 / SCALE_FACTOR_4B),  BLOCK_LOAD_DIRECT, LOAD_LDG, true, BLOCK_SCAN_RAKING_MEMOIZE, RADIX_SORT_SCATTER_TWO_PHASE, cudaSharedMemBankSizeFourByte, ALT_RADIX_BITS> AltDownsweepPolicy;

        typedef AgentRadixSortDownsweepPolicy <256, CUB_MAX(1, 19 / SCALE_FACTOR_4B),  BLOCK_LOAD_DIRECT, LOAD_LDG, true, BLOCK_SCAN_WARP_SCANS, RADIX_SORT_SCATTER_TWO_PHASE, cudaSharedMemBankSizeFourByte, PRIMARY_RADIX_BITS> SinglePolicy;
    };


    /// SM35
    struct Policy350
    {
        enum {
            PRIMARY_RADIX_BITS      = 5,
            ALT_RADIX_BITS          = PRIMARY_RADIX_BITS - 1,
        };

        // Primary UpsweepPolicy (passes having digit-length RADIX_BITS)
        typedef AgentRadixSortUpsweepPolicy <64,     CUB_MAX(1, 18 / SCALE_FACTOR_4B), LOAD_LDG, PRIMARY_RADIX_BITS> UpsweepPolicyKeys;
        typedef AgentRadixSortUpsweepPolicy <128,    CUB_MAX(1, 15 / SCALE_FACTOR_4B), LOAD_LDG, PRIMARY_RADIX_BITS> UpsweepPolicyPairs;
        typedef typename If<KEYS_ONLY, UpsweepPolicyKeys, UpsweepPolicyPairs>::Type UpsweepPolicy;

        // Alternate UpsweepPolicy (passes having digit-length ALT_RADIX_BITS)
        typedef AgentRadixSortUpsweepPolicy <64,     CUB_MAX(1, 22 / SCALE_FACTOR_4B), LOAD_LDG, ALT_RADIX_BITS> AltUpsweepPolicyKeys;
        typedef AgentRadixSortUpsweepPolicy <128,    CUB_MAX(1, 15 / SCALE_FACTOR_4B), LOAD_LDG, ALT_RADIX_BITS> AltUpsweepPolicyPairs;
        typedef typename If<KEYS_ONLY, AltUpsweepPolicyKeys, AltUpsweepPolicyPairs>::Type AltUpsweepPolicy;

        // ScanPolicy
        typedef AgentScanPolicy <1024, 4, BLOCK_LOAD_VECTORIZE, LOAD_DEFAULT, BLOCK_STORE_VECTORIZE, BLOCK_SCAN_WARP_SCANS> ScanPolicy;

        // Primary DownsweepPolicy
        typedef AgentRadixSortDownsweepPolicy <64,   CUB_MAX(1, 18 / SCALE_FACTOR_4B), BLOCK_LOAD_DIRECT, LOAD_LDG, true, BLOCK_SCAN_WARP_SCANS, RADIX_SORT_SCATTER_TWO_PHASE, cudaSharedMemBankSizeFourByte, PRIMARY_RADIX_BITS> DownsweepPolicyKeys;
        typedef AgentRadixSortDownsweepPolicy <128,  CUB_MAX(1, 15 / SCALE_FACTOR_4B), BLOCK_LOAD_DIRECT, LOAD_LDG, true, BLOCK_SCAN_WARP_SCANS, RADIX_SORT_SCATTER_TWO_PHASE, cudaSharedMemBankSizeFourByte, PRIMARY_RADIX_BITS> DownsweepPolicyPairs;
        typedef typename If<KEYS_ONLY, DownsweepPolicyKeys, DownsweepPolicyPairs>::Type DownsweepPolicy;

        // Alternate DownsweepPolicy for ALT_RADIX_BITS-bit passes
        typedef AgentRadixSortDownsweepPolicy <128,  CUB_MAX(1, 11 / SCALE_FACTOR_4B), BLOCK_LOAD_DIRECT, LOAD_LDG, true, BLOCK_SCAN_WARP_SCANS, RADIX_SORT_SCATTER_TWO_PHASE, cudaSharedMemBankSizeFourByte, ALT_RADIX_BITS> AltDownsweepPolicyKeys;
        typedef AgentRadixSortDownsweepPolicy <128,  CUB_MAX(1, 15 / SCALE_FACTOR_4B), BLOCK_LOAD_DIRECT, LOAD_LDG, true, BLOCK_SCAN_WARP_SCANS, RADIX_SORT_SCATTER_TWO_PHASE, cudaSharedMemBankSizeFourByte, ALT_RADIX_BITS> AltDownsweepPolicyPairs;
        typedef typename If<KEYS_ONLY, AltDownsweepPolicyKeys, AltDownsweepPolicyPairs>::Type AltDownsweepPolicy;

        typedef DownsweepPolicy SinglePolicy;
    };


    /// SM30
    struct Policy300
    {
        enum {
            PRIMARY_RADIX_BITS      = 5,
            ALT_RADIX_BITS          = PRIMARY_RADIX_BITS - 1,
        };

        // UpsweepPolicy
        typedef AgentRadixSortUpsweepPolicy <256, CUB_MAX(1, 7 / SCALE_FACTOR_4B), LOAD_DEFAULT, PRIMARY_RADIX_BITS> UpsweepPolicyKeys;
        typedef AgentRadixSortUpsweepPolicy <256, CUB_MAX(1, 5 / SCALE_FACTOR_4B), LOAD_DEFAULT, PRIMARY_RADIX_BITS> UpsweepPolicyPairs;
        typedef typename If<KEYS_ONLY, UpsweepPolicyKeys, UpsweepPolicyPairs>::Type UpsweepPolicy;

        // Alternate UpsweepPolicy for ALT_RADIX_BITS-bit passes
        typedef AgentRadixSortUpsweepPolicy <256, CUB_MAX(1, 7 / SCALE_FACTOR_4B), LOAD_DEFAULT, ALT_RADIX_BITS> AltUpsweepPolicyKeys;
        typedef AgentRadixSortUpsweepPolicy <256, CUB_MAX(1, 5 / SCALE_FACTOR_4B), LOAD_DEFAULT, ALT_RADIX_BITS> AltUpsweepPolicyPairs;
        typedef typename If<KEYS_ONLY, AltUpsweepPolicyKeys, AltUpsweepPolicyPairs>::Type AltUpsweepPolicy;

        // ScanPolicy
        typedef AgentScanPolicy <1024, 4, BLOCK_LOAD_VECTORIZE, LOAD_DEFAULT, BLOCK_STORE_VECTORIZE, BLOCK_SCAN_RAKING_MEMOIZE> ScanPolicy;

        // DownsweepPolicy
        typedef AgentRadixSortDownsweepPolicy <128, CUB_MAX(1, 14 / SCALE_FACTOR_4B), BLOCK_LOAD_WARP_TRANSPOSE, LOAD_DEFAULT, false, BLOCK_SCAN_WARP_SCANS, RADIX_SORT_SCATTER_TWO_PHASE, cudaSharedMemBankSizeFourByte, PRIMARY_RADIX_BITS> DownsweepPolicyKeys;
        typedef AgentRadixSortDownsweepPolicy <128, CUB_MAX(1, 10 / SCALE_FACTOR_4B), BLOCK_LOAD_TRANSPOSE, LOAD_DEFAULT, false, BLOCK_SCAN_WARP_SCANS, RADIX_SORT_SCATTER_TWO_PHASE, cudaSharedMemBankSizeFourByte, PRIMARY_RADIX_BITS> DownsweepPolicyPairs;
        typedef typename If<KEYS_ONLY, DownsweepPolicyKeys, DownsweepPolicyPairs>::Type DownsweepPolicy;

        // Alternate DownsweepPolicy for ALT_RADIX_BITS-bit passes
        typedef AgentRadixSortDownsweepPolicy <128, CUB_MAX(1, 14 / SCALE_FACTOR_4B), BLOCK_LOAD_WARP_TRANSPOSE, LOAD_DEFAULT, false, BLOCK_SCAN_WARP_SCANS, RADIX_SORT_SCATTER_TWO_PHASE, cudaSharedMemBankSizeFourByte, ALT_RADIX_BITS> AltDownsweepPolicyKeys;
        typedef AgentRadixSortDownsweepPolicy <128, CUB_MAX(1, 10 / SCALE_FACTOR_4B), BLOCK_LOAD_TRANSPOSE, LOAD_DEFAULT, false, BLOCK_SCAN_WARP_SCANS, RADIX_SORT_SCATTER_TWO_PHASE, cudaSharedMemBankSizeFourByte, ALT_RADIX_BITS> AltDownsweepPolicyPairs;
        typedef typename If<KEYS_ONLY, AltDownsweepPolicyKeys, AltDownsweepPolicyPairs>::Type AltDownsweepPolicy;

        typedef DownsweepPolicy SinglePolicy;
    };


    /// SM20
    struct Policy200
    {
        enum {
            PRIMARY_RADIX_BITS      = 5,
            ALT_RADIX_BITS          = PRIMARY_RADIX_BITS - 1,
        };

        // Primary UpsweepPolicy (passes having digit-length RADIX_BITS)
        typedef AgentRadixSortUpsweepPolicy <64, CUB_MAX(1, 18 / SCALE_FACTOR_4B), LOAD_DEFAULT, PRIMARY_RADIX_BITS> UpsweepPolicyKeys;
        typedef AgentRadixSortUpsweepPolicy <128, CUB_MAX(1, 13 / SCALE_FACTOR_4B), LOAD_DEFAULT, PRIMARY_RADIX_BITS> UpsweepPolicyPairs;
        typedef typename If<KEYS_ONLY, UpsweepPolicyKeys, UpsweepPolicyPairs>::Type UpsweepPolicy;

        // Alternate UpsweepPolicy for ALT_RADIX_BITS-bit passes
        typedef AgentRadixSortUpsweepPolicy <64, CUB_MAX(1, 18 / SCALE_FACTOR_4B), LOAD_DEFAULT, ALT_RADIX_BITS> AltUpsweepPolicyKeys;
        typedef AgentRadixSortUpsweepPolicy <128, CUB_MAX(1, 13 / SCALE_FACTOR_4B), LOAD_DEFAULT, ALT_RADIX_BITS> AltUpsweepPolicyPairs;
        typedef typename If<KEYS_ONLY, AltUpsweepPolicyKeys, AltUpsweepPolicyPairs>::Type AltUpsweepPolicy;

        // ScanPolicy
        typedef AgentScanPolicy <512, 4, BLOCK_LOAD_VECTORIZE, LOAD_DEFAULT, BLOCK_STORE_VECTORIZE, BLOCK_SCAN_RAKING_MEMOIZE> ScanPolicy;

        // DownsweepPolicy
        typedef AgentRadixSortDownsweepPolicy <64, CUB_MAX(1, 18 / SCALE_FACTOR_4B), BLOCK_LOAD_WARP_TRANSPOSE, LOAD_DEFAULT, false, BLOCK_SCAN_WARP_SCANS, RADIX_SORT_SCATTER_TWO_PHASE, cudaSharedMemBankSizeFourByte, PRIMARY_RADIX_BITS> DownsweepPolicyKeys;
        typedef AgentRadixSortDownsweepPolicy <128, CUB_MAX(1, 13 / SCALE_FACTOR_4B), BLOCK_LOAD_WARP_TRANSPOSE, LOAD_DEFAULT, false, BLOCK_SCAN_WARP_SCANS, RADIX_SORT_SCATTER_TWO_PHASE, cudaSharedMemBankSizeFourByte, PRIMARY_RADIX_BITS> DownsweepPolicyPairs;
        typedef typename If<KEYS_ONLY, DownsweepPolicyKeys, DownsweepPolicyPairs>::Type DownsweepPolicy;

        // Alternate DownsweepPolicy for ALT_RADIX_BITS-bit passes
        typedef AgentRadixSortDownsweepPolicy <64, CUB_MAX(1, 18 / SCALE_FACTOR_4B), BLOCK_LOAD_WARP_TRANSPOSE, LOAD_DEFAULT, false, BLOCK_SCAN_WARP_SCANS, RADIX_SORT_SCATTER_TWO_PHASE, cudaSharedMemBankSizeFourByte, ALT_RADIX_BITS> AltDownsweepPolicyKeys;
        typedef AgentRadixSortDownsweepPolicy <128, CUB_MAX(1, 13 / SCALE_FACTOR_4B), BLOCK_LOAD_WARP_TRANSPOSE, LOAD_DEFAULT, false, BLOCK_SCAN_WARP_SCANS, RADIX_SORT_SCATTER_TWO_PHASE, cudaSharedMemBankSizeFourByte, ALT_RADIX_BITS> AltDownsweepPolicyPairs;
        typedef typename If<KEYS_ONLY, AltDownsweepPolicyKeys, AltDownsweepPolicyPairs>::Type AltDownsweepPolicy;

        typedef DownsweepPolicy SinglePolicy;
    };


    /// SM13
    struct Policy130
    {
        enum {
            PRIMARY_RADIX_BITS      = 5,
            ALT_RADIX_BITS          = PRIMARY_RADIX_BITS - 1,
        };

        // UpsweepPolicy
        typedef AgentRadixSortUpsweepPolicy <128, CUB_MAX(1, 19 / SCALE_FACTOR_4B), LOAD_DEFAULT, PRIMARY_RADIX_BITS> UpsweepPolicyKeys;
        typedef AgentRadixSortUpsweepPolicy <128, CUB_MAX(1, 19 / SCALE_FACTOR_4B), LOAD_DEFAULT, PRIMARY_RADIX_BITS> UpsweepPolicyPairs;
        typedef typename If<KEYS_ONLY, UpsweepPolicyKeys, UpsweepPolicyPairs>::Type UpsweepPolicy;

        // Alternate UpsweepPolicy for ALT_RADIX_BITS-bit passes
        typedef AgentRadixSortUpsweepPolicy <128, CUB_MAX(1, 15 / SCALE_FACTOR_4B), LOAD_DEFAULT, ALT_RADIX_BITS> AltUpsweepPolicyKeys;
        typedef AgentRadixSortUpsweepPolicy <128, CUB_MAX(1, 15 / SCALE_FACTOR_4B), LOAD_DEFAULT, ALT_RADIX_BITS> AltUpsweepPolicyPairs;
        typedef typename If<KEYS_ONLY, AltUpsweepPolicyKeys, AltUpsweepPolicyPairs>::Type AltUpsweepPolicy;

        // ScanPolicy
        typedef AgentScanPolicy <256, 4, BLOCK_LOAD_VECTORIZE, LOAD_DEFAULT, BLOCK_STORE_VECTORIZE, BLOCK_SCAN_WARP_SCANS> ScanPolicy;

        // DownsweepPolicy
        typedef AgentRadixSortDownsweepPolicy <64, CUB_MAX(1, 19 / SCALE_FACTOR_4B), BLOCK_LOAD_WARP_TRANSPOSE, LOAD_DEFAULT, false, BLOCK_SCAN_WARP_SCANS, RADIX_SORT_SCATTER_TWO_PHASE, cudaSharedMemBankSizeFourByte, PRIMARY_RADIX_BITS> DownsweepPolicyKeys;
        typedef AgentRadixSortDownsweepPolicy <64, CUB_MAX(1, 19 / SCALE_FACTOR_4B), BLOCK_LOAD_WARP_TRANSPOSE, LOAD_DEFAULT, false, BLOCK_SCAN_WARP_SCANS, RADIX_SORT_SCATTER_TWO_PHASE, cudaSharedMemBankSizeFourByte, PRIMARY_RADIX_BITS> DownsweepPolicyPairs;
        typedef typename If<KEYS_ONLY, DownsweepPolicyKeys, DownsweepPolicyPairs>::Type DownsweepPolicy;

        // Alternate DownsweepPolicy for ALT_RADIX_BITS-bit passes
        typedef AgentRadixSortDownsweepPolicy <128, CUB_MAX(1, 15 / SCALE_FACTOR_4B), BLOCK_LOAD_WARP_TRANSPOSE, LOAD_DEFAULT, false, BLOCK_SCAN_WARP_SCANS, RADIX_SORT_SCATTER_TWO_PHASE, cudaSharedMemBankSizeFourByte, ALT_RADIX_BITS> AltDownsweepPolicyKeys;
        typedef AgentRadixSortDownsweepPolicy <128, CUB_MAX(1, 15 / SCALE_FACTOR_4B), BLOCK_LOAD_WARP_TRANSPOSE, LOAD_DEFAULT, false, BLOCK_SCAN_WARP_SCANS, RADIX_SORT_SCATTER_TWO_PHASE, cudaSharedMemBankSizeFourByte, ALT_RADIX_BITS> AltDownsweepPolicyPairs;
        typedef typename If<KEYS_ONLY, AltDownsweepPolicyKeys, AltDownsweepPolicyPairs>::Type AltDownsweepPolicy;

        typedef DownsweepPolicy SinglePolicy;
    };


    /// SM10
    struct Policy100
    {
        enum {
            PRIMARY_RADIX_BITS      = 4,
            ALT_RADIX_BITS          = PRIMARY_RADIX_BITS - 1,
        };

        // UpsweepPolicy
        typedef AgentRadixSortUpsweepPolicy <64, CUB_MAX(1, 9 / SCALE_FACTOR_4B), LOAD_DEFAULT, PRIMARY_RADIX_BITS> UpsweepPolicy;

        // Alternate UpsweepPolicy for ALT_RADIX_BITS-bit passes
        typedef AgentRadixSortUpsweepPolicy <64, CUB_MAX(1, 9 / SCALE_FACTOR_4B), LOAD_DEFAULT, ALT_RADIX_BITS> AltUpsweepPolicy;

        // ScanPolicy
        typedef AgentScanPolicy <256, 4, BLOCK_LOAD_VECTORIZE, LOAD_DEFAULT, BLOCK_STORE_VECTORIZE, BLOCK_SCAN_RAKING_MEMOIZE> ScanPolicy;

        // DownsweepPolicy
        typedef AgentRadixSortDownsweepPolicy <64, CUB_MAX(1, 9 / SCALE_FACTOR_4B), BLOCK_LOAD_WARP_TRANSPOSE, LOAD_DEFAULT, false, BLOCK_SCAN_WARP_SCANS, RADIX_SORT_SCATTER_TWO_PHASE, cudaSharedMemBankSizeFourByte, PRIMARY_RADIX_BITS> DownsweepPolicy;

        // Alternate DownsweepPolicy for ALT_RADIX_BITS-bit passes
        typedef AgentRadixSortDownsweepPolicy <64, CUB_MAX(1, 9 / SCALE_FACTOR_4B), BLOCK_LOAD_WARP_TRANSPOSE, LOAD_DEFAULT, false, BLOCK_SCAN_WARP_SCANS, RADIX_SORT_SCATTER_TWO_PHASE, cudaSharedMemBankSizeFourByte, ALT_RADIX_BITS> AltDownsweepPolicy;

        typedef DownsweepPolicy SinglePolicy;
    };


    /******************************************************************************
     * Tuning policies of current PTX compiler pass
     ******************************************************************************/

#if (CUB_PTX_ARCH >= 520)
    typedef Policy520 PtxPolicy;

#elif (CUB_PTX_ARCH >= 350)
    typedef Policy350 PtxPolicy;

#elif (CUB_PTX_ARCH >= 300)
    typedef Policy300 PtxPolicy;

#elif (CUB_PTX_ARCH >= 200)
    typedef Policy200 PtxPolicy;

#elif (CUB_PTX_ARCH >= 130)
    typedef Policy130 PtxPolicy;

#else
    typedef Policy100 PtxPolicy;

#endif

    // "Opaque" policies (whose parameterizations aren't reflected in the type signature)
    struct PtxUpsweepPolicy         : PtxPolicy::UpsweepPolicy {};
    struct PtxAltUpsweepPolicy      : PtxPolicy::AltUpsweepPolicy {};
    struct PtxScanPolicy            : PtxPolicy::ScanPolicy {};
    struct PtxDownsweepPolicy       : PtxPolicy::DownsweepPolicy {};
    struct PtxAltDownsweepPolicy    : PtxPolicy::AltDownsweepPolicy {};
    struct PtxSinglePolicy          : PtxPolicy::SinglePolicy {};


    /******************************************************************************
     * Utilities
     ******************************************************************************/

    /**
     * Initialize kernel dispatch configurations with the policies corresponding to the PTX assembly we will use
     */
    template <
        typename Policy,
        typename KernelConfig,
        typename UpsweepKernelPtr,          ///< Function type of cub::DeviceRadixSortUpsweepKernel
        typename ScanKernelPtr,             ///< Function type of cub::SpineScanKernel
        typename DownsweepKernelPtr,        ///< Function type of cub::DeviceRadixSortDownsweepKernel
        typename SingleKernelPtr>           ///< Function type of cub::DeviceRadixSortSingleKernel
    CUB_RUNTIME_FUNCTION __forceinline__
    static cudaError_t InitConfigs(
        int                     sm_version,
        int                     sm_count,
        KernelConfig            &upsweep_config,
        KernelConfig            &alt_upsweep_config,
        KernelConfig            &scan_config,
        KernelConfig            &downsweep_config,
        KernelConfig            &alt_downsweep_config,
        KernelConfig            &single_config,
        UpsweepKernelPtr        upsweep_kernel,
        UpsweepKernelPtr        alt_upsweep_kernel,
        ScanKernelPtr           scan_kernel,
        DownsweepKernelPtr      downsweep_kernel,
        DownsweepKernelPtr      alt_downsweep_kernel,
        SingleKernelPtr         single_kernel)
    {
        cudaError_t error;
        do {
            if (CubDebug(error = upsweep_config.template        InitUpsweepPolicy<typename Policy::UpsweepPolicy>(          sm_version, sm_count, upsweep_kernel))) break;
            if (CubDebug(error = alt_upsweep_config.template    InitUpsweepPolicy<typename Policy::AltUpsweepPolicy>(       sm_version, sm_count, alt_upsweep_kernel))) break;
            if (CubDebug(error = scan_config.template           InitScanPolicy<typename Policy::ScanPolicy>(                sm_version, sm_count, scan_kernel))) break;
            if (CubDebug(error = downsweep_config.template      InitDownsweepPolicy<typename Policy::DownsweepPolicy>(      sm_version, sm_count, downsweep_kernel))) break;
            if (CubDebug(error = alt_downsweep_config.template  InitDownsweepPolicy<typename Policy::AltDownsweepPolicy>(   sm_version, sm_count, alt_downsweep_kernel))) break;
            if (CubDebug(error = single_config.template         InitSinglePolicy<typename Policy::SinglePolicy>(            sm_version, sm_count, single_kernel))) break;

        } while (0);

        return error;
    }


    /**
     * Initialize kernel dispatch configurations with the policies corresponding to the PTX assembly we will use
     */
    template <
        typename KernelConfig,
        typename UpsweepKernelPtr,          ///< Function type of cub::DeviceRadixSortUpsweepKernel
        typename ScanKernelPtr,             ///< Function type of cub::SpineScanKernel
        typename DownsweepKernelPtr,        ///< Function type of cub::DeviceRadixSortDownsweepKernel
        typename SingleKernelPtr>           ///< Function type of cub::DeviceRadixSortSingleKernel
    CUB_RUNTIME_FUNCTION __forceinline__
    static cudaError_t InitConfigs(
        int                     ptx_version,
        int                     sm_version,
        int                     sm_count,
        KernelConfig            &upsweep_config,
        KernelConfig            &alt_upsweep_config,
        KernelConfig            &scan_config,
        KernelConfig            &downsweep_config,
        KernelConfig            &alt_downsweep_config,
        KernelConfig            &single_config,
        UpsweepKernelPtr        upsweep_kernel,
        UpsweepKernelPtr        alt_upsweep_kernel,
        ScanKernelPtr           scan_kernel,
        DownsweepKernelPtr      downsweep_kernel,
        DownsweepKernelPtr      alt_downsweep_kernel,
        SingleKernelPtr         single_kernel)
    {
    #if (CUB_PTX_ARCH > 0)

        // We're on the device, so initialize the kernel dispatch configurations with the current PTX policy
        cudaError_t error;
        do {

            if (CubDebug(error = upsweep_config.template InitUpsweepPolicy<PtxUpsweepPolicy>(               sm_version, sm_count, upsweep_kernel))) break;
            if (CubDebug(error = alt_upsweep_config.template InitUpsweepPolicy<PtxAltUpsweepPolicy>(        sm_version, sm_count, alt_upsweep_kernel))) break;
            if (CubDebug(error = scan_config.template InitScanPolicy<PtxScanPolicy>(                        sm_version, sm_count, scan_kernel))) break;
            if (CubDebug(error = downsweep_config.template InitDownsweepPolicy<PtxDownsweepPolicy>(         sm_version, sm_count, downsweep_kernel))) break;
            if (CubDebug(error = alt_downsweep_config.template InitDownsweepPolicy<PtxAltDownsweepPolicy>(  sm_version, sm_count, alt_downsweep_kernel))) break;
            if (CubDebug(error = single_config.template InitSinglePolicy<PtxSinglePolicy>(                  sm_version, sm_count, single_kernel))) break;

        } while (0);

        return error;

    #else

        // We're on the host, so lookup and initialize the kernel dispatch configurations with the policies that match the device's PTX version
        cudaError_t error;
        if (ptx_version >= 520)
        {
            error = InitConfigs<Policy520>(sm_version, sm_count, upsweep_config, alt_upsweep_config, scan_config, downsweep_config, alt_downsweep_config, single_config, upsweep_kernel, alt_upsweep_kernel, scan_kernel, downsweep_kernel, alt_downsweep_kernel, single_kernel);
        }
        else if (ptx_version >= 350)
        {
            error = InitConfigs<Policy350>(sm_version, sm_count, upsweep_config, alt_upsweep_config, scan_config, downsweep_config, alt_downsweep_config, single_config, upsweep_kernel, alt_upsweep_kernel, scan_kernel, downsweep_kernel, alt_downsweep_kernel, single_kernel);
        }
        else if (ptx_version >= 300)
        {
            error = InitConfigs<Policy300>(sm_version, sm_count, upsweep_config, alt_upsweep_config, scan_config, downsweep_config, alt_downsweep_config, single_config, upsweep_kernel, alt_upsweep_kernel, scan_kernel, downsweep_kernel, alt_downsweep_kernel, single_kernel);
        }
        else if (ptx_version >= 200)
        {
            error = InitConfigs<Policy200>(sm_version, sm_count, upsweep_config, alt_upsweep_config, scan_config, downsweep_config, alt_downsweep_config, single_config, upsweep_kernel, alt_upsweep_kernel, scan_kernel, downsweep_kernel, alt_downsweep_kernel, single_kernel);
        }
        else if (ptx_version >= 130)
        {
            error = InitConfigs<Policy130>(sm_version, sm_count, upsweep_config, alt_upsweep_config, scan_config, downsweep_config, alt_downsweep_config, single_config, upsweep_kernel, alt_upsweep_kernel, scan_kernel, downsweep_kernel, alt_downsweep_kernel, single_kernel);
        }
        else
        {
            error = InitConfigs<Policy100>(sm_version, sm_count, upsweep_config, alt_upsweep_config, scan_config, downsweep_config, alt_downsweep_config, single_config, upsweep_kernel, alt_upsweep_kernel, scan_kernel, downsweep_kernel, alt_downsweep_kernel, single_kernel);
        }

        return error;

    #endif
    }



    /**
     * Kernel kernel dispatch configurations
     */
    struct KernelConfig
    {
        int                     block_threads;
        int                     items_per_thread;
        int                     tile_size;
        int                     radix_bits;
        int                     sm_occupancy;
        int                     max_grid_size;
        int                     subscription_factor;

        template <typename UpsweepPolicy, typename UpsweepKernelPtr>
        CUB_RUNTIME_FUNCTION __forceinline__ cudaError_t InitUpsweepPolicy(
            int sm_version, int sm_count, UpsweepKernelPtr upsweep_kernel)
        {
            block_threads               = UpsweepPolicy::BLOCK_THREADS;
            items_per_thread            = UpsweepPolicy::ITEMS_PER_THREAD;
            radix_bits                  = UpsweepPolicy::RADIX_BITS;
            tile_size                   = block_threads * items_per_thread;
            cudaError_t retval          = MaxSmOccupancy(sm_occupancy, sm_version, upsweep_kernel, block_threads);
            subscription_factor         = CUB_SUBSCRIPTION_FACTOR(sm_version);
            max_grid_size               = (sm_occupancy * sm_count) * subscription_factor;

            return retval;
        }

        template <typename ScanPolicy, typename ScanKernelPtr>
        CUB_RUNTIME_FUNCTION __forceinline__ cudaError_t InitScanPolicy(
            int sm_version, int sm_count, ScanKernelPtr scan_kernel)
        {
            block_threads               = ScanPolicy::BLOCK_THREADS;
            items_per_thread            = ScanPolicy::ITEMS_PER_THREAD;
            radix_bits                  = 0;
            tile_size                   = block_threads * items_per_thread;
            sm_occupancy                = 1;
            subscription_factor         = 1;
            max_grid_size               = 1;

            return cudaSuccess;
        }

        template <typename DownsweepPolicy, typename DownsweepKernelPtr>
        CUB_RUNTIME_FUNCTION __forceinline__ cudaError_t InitDownsweepPolicy(
            int sm_version, int sm_count, DownsweepKernelPtr downsweep_kernel)
        {
            block_threads               = DownsweepPolicy::BLOCK_THREADS;
            items_per_thread            = DownsweepPolicy::ITEMS_PER_THREAD;
            radix_bits                  = DownsweepPolicy::RADIX_BITS;
            tile_size                   = block_threads * items_per_thread;
            cudaError_t retval          = MaxSmOccupancy(sm_occupancy, sm_version, downsweep_kernel, block_threads);
            subscription_factor         = CUB_SUBSCRIPTION_FACTOR(sm_version);
            max_grid_size               = (sm_occupancy * sm_count) * subscription_factor;

            return retval;
        }

        template <typename SinglePolicy, typename SingleKernelPtr>
        CUB_RUNTIME_FUNCTION __forceinline__ cudaError_t InitSinglePolicy(
            int sm_version, int sm_count, SingleKernelPtr single_kernel)
        {
            block_threads               = SinglePolicy::BLOCK_THREADS;
            items_per_thread            = SinglePolicy::ITEMS_PER_THREAD;
            radix_bits                  = SinglePolicy::RADIX_BITS;
            tile_size                   = block_threads * items_per_thread;
            sm_occupancy                = 1;
            subscription_factor         = 1;
            max_grid_size               = 1;

            return cudaSuccess;
        }

    };


    /******************************************************************************
     * Dispatch entrypoints
     ******************************************************************************/

    /**
     * Internal dispatch routine for computing a device-wide radix sort using the
     * specified kernel functions.
     */
    template <
        typename                UpsweepKernelPtr,               ///< Function type of cub::DeviceRadixSortUpsweepKernel
        typename                ScanKernelPtr,                  ///< Function type of cub::SpineScanKernel
        typename                DownsweepKernelPtr>             ///< Function type of cub::DeviceRadixSortUpsweepKernel
    CUB_RUNTIME_FUNCTION __forceinline__
    static cudaError_t DispatchPass(
        KeyT                     *d_keys_in,
        KeyT                     *d_keys_out,
        ValueT                   *d_values_in,
        ValueT                   *d_values_out,
        OffsetT                 *d_spine,                       ///< [in] Digit count histograms per thread block
        int                     spine_length,                   ///< [in] Number of histogram counters
        OffsetT                 num_items,                      ///< [in] Number of items to reduce
        int                     current_bit,                    ///< [in] The beginning (least-significant) bit index needed for key comparison
        int                     pass_bits,                      ///< [in] The number of bits needed for key comparison (less than or equal to radix digit size for this pass)
        cudaStream_t            stream,                         ///< [in] CUDA stream to launch kernels within.  Default is stream<sub>0</sub>.
        bool                    debug_synchronous,              ///< [in] Whether or not to synchronize the stream after every kernel launch to check for errors.  Also causes launch configurations to be printed to the console.  Default is \p false.
        KernelConfig            &upsweep_config,                ///< [in] Dispatch parameters that match the policy that \p upsweep_kernel was compiled for
        KernelConfig            &scan_config,                   ///< [in] Dispatch parameters that match the policy that \p scan_kernel was compiled for
        KernelConfig            &downsweep_config,              ///< [in] Dispatch parameters that match the policy that \p downsweep_kernel was compiled for
        UpsweepKernelPtr        upsweep_kernel,                 ///< [in] Kernel function pointer to parameterization of cub::DeviceRadixSortUpsweepKernel
        ScanKernelPtr           scan_kernel,                    ///< [in] Kernel function pointer to parameterization of cub::SpineScanKernel
        DownsweepKernelPtr      downsweep_kernel,               ///< [in] Kernel function pointer to parameterization of cub::DeviceRadixSortUpsweepKernel
        GridEvenShare<OffsetT>  &even_share)                    ///< [in] Description of work assignment to CTAs
    {
#ifndef CUB_RUNTIME_ENABLED

        // Kernel launch not supported from this device
        return CubDebug(cudaErrorNotSupported);

#else

        cudaError error = cudaSuccess;
        do
        {
            // Log upsweep_kernel configuration
            if (debug_synchronous)
                CubLog("Invoking upsweep_kernel<<<%d, %d, 0, %lld>>>(), %d items per thread, %d SM occupancy, current bit %d, bit_grain %d\n",
                even_share.grid_size, upsweep_config.block_threads, (long long) stream, upsweep_config.items_per_thread, upsweep_config.sm_occupancy, current_bit, downsweep_config.radix_bits);

            // Invoke upsweep_kernel with same grid size as downsweep_kernel
            upsweep_kernel<<<even_share.grid_size, upsweep_config.block_threads, 0, stream>>>(
                d_keys_in,
                d_spine,
                num_items,
                current_bit,
                pass_bits,
                even_share);

            // Check for failure to launch
            if (CubDebug(error = cudaPeekAtLastError())) break;

            // Sync the stream if specified to flush runtime errors
            if (debug_synchronous && (CubDebug(error = SyncStream(stream)))) break;

            // Log scan_kernel configuration
            if (debug_synchronous) CubLog("Invoking scan_kernel<<<%d, %d, 0, %lld>>>(), %d items per thread\n",
                1, scan_config.block_threads, (long long) stream, scan_config.items_per_thread);

            // Invoke scan_kernel
            scan_kernel<<<1, scan_config.block_threads, 0, stream>>>(
                d_spine,
                spine_length);

            // Check for failure to launch
            if (CubDebug(error = cudaPeekAtLastError())) break;

            // Sync the stream if specified to flush runtime errors
            if (debug_synchronous && (CubDebug(error = SyncStream(stream)))) break;

            // Log downsweep_kernel configuration
            if (debug_synchronous) CubLog("Invoking downsweep_kernel<<<%d, %d, 0, %lld>>>(), %d items per thread, %d SM occupancy\n",
                even_share.grid_size, downsweep_config.block_threads, (long long) stream, downsweep_config.items_per_thread, downsweep_config.sm_occupancy);

            // Invoke downsweep_kernel
            downsweep_kernel<<<even_share.grid_size, downsweep_config.block_threads, 0, stream>>>(
                d_keys_in,
                d_keys_out,
                d_values_in,
                d_values_out,
                d_spine,
                num_items,
                current_bit,
                pass_bits,
                even_share);

            // Check for failure to launch
            if (CubDebug(error = cudaPeekAtLastError())) break;

            // Sync the stream if specified to flush runtime errors
            if (debug_synchronous && (CubDebug(error = SyncStream(stream)))) break;
        }
        while (0);

        return error;

#endif // CUB_RUNTIME_ENABLED
    }




    /**
     * Internal dispatch routine
     */
    template <
        typename UpsweepKernelPtr,          ///< Function type of cub::DeviceRadixSortUpsweepKernel
        typename ScanKernelPtr,             ///< Function type of cub::SpineScanKernel
        typename DownsweepKernelPtr,        ///< Function type of cub::DeviceRadixSortDownsweepKernel
        typename SingleKernelPtr>           ///< Function type of cub::DeviceRadixSortSingleKernel
    CUB_RUNTIME_FUNCTION __forceinline__
    static cudaError_t Dispatch(
        void*               d_temp_storage,                ///< [in] %Device-accessible allocation of temporary storage.  When NULL, the required allocation size is written to \p temp_storage_bytes and no work is done.
        size_t                  &temp_storage_bytes,            ///< [in,out] Reference to size in bytes of \p d_temp_storage allocation
        DoubleBuffer<KeyT>       &d_keys,                        ///< [in,out] Double-buffer whose current buffer contains the unsorted input keys and, upon return, is updated to point to the sorted output keys
        DoubleBuffer<ValueT>     &d_values,                      ///< [in,out] Double-buffer whose current buffer contains the unsorted input values and, upon return, is updated to point to the sorted output values
        OffsetT                 num_items,                      ///< [in] Number of items to reduce
        int                     begin_bit,                      ///< [in] The beginning (least-significant) bit index needed for key comparison
        int                     end_bit,                        ///< [in] The past-the-end (most-significant) bit index needed for key comparison
        cudaStream_t            stream,                         ///< [in] CUDA stream to launch kernels within.  Default is stream<sub>0</sub>.
        bool                    debug_synchronous,              ///< [in] Whether or not to synchronize the stream after every kernel launch to check for errors.  Also causes launch configurations to be printed to the console.  Default is \p false.
        UpsweepKernelPtr        upsweep_kernel,                 ///< [in] Kernel function pointer to parameterization of cub::DeviceRadixSortUpsweepKernel
        UpsweepKernelPtr        alt_upsweep_kernel,             ///< [in] Alternate kernel function pointer to parameterization of cub::DeviceRadixSortUpsweepKernel
        ScanKernelPtr           scan_kernel,                    ///< [in] Kernel function pointer to parameterization of cub::SpineScanKernel
        DownsweepKernelPtr      downsweep_kernel,               ///< [in] Kernel function pointer to parameterization of cub::DeviceRadixSortDownsweepKernel
        DownsweepKernelPtr      alt_downsweep_kernel,           ///< [in] Alternate kernel function pointer to parameterization of cub::DeviceRadixSortDownsweepKernel
        SingleKernelPtr         single_kernel)                  ///< [in] Alternate kernel function pointer to parameterization of cub::DeviceRadixSortSingleKernel
    {
#ifndef CUB_RUNTIME_ENABLED

        // Kernel launch not supported from this device
        return CubDebug(cudaErrorNotSupported );

#else

        cudaError error = cudaSuccess;

        do
        {
            // Get PTX version
            int ptx_version;
    #if (CUB_PTX_ARCH == 0)
            if (CubDebug(error = PtxVersion(ptx_version))) break;
    #else
            ptx_version = CUB_PTX_ARCH;
    #endif

            // Get device ordinal
            int device_ordinal;
            if (CubDebug(error = cudaGetDevice(&device_ordinal))) break;

            // Get device SM version
            int sm_version;
            if (CubDebug(error = SmVersion(sm_version, device_ordinal))) break;

            // Get SM count
            int sm_count;
            if (CubDebug(error = cudaDeviceGetAttribute (&sm_count, cudaDevAttrMultiProcessorCount, device_ordinal))) break;

            // Initialize kernel dispatch configurations
            KernelConfig upsweep_config, alt_upsweep_config, scan_config, downsweep_config, alt_downsweep_config, single_config;
            if (CubDebug(error = InitConfigs(ptx_version, sm_version, sm_count,
                upsweep_config, alt_upsweep_config, scan_config, downsweep_config, alt_downsweep_config, single_config,
                upsweep_kernel, alt_upsweep_kernel, scan_kernel, downsweep_kernel, alt_downsweep_kernel, single_kernel))) break;

            int num_passes;

            if (num_items <= single_config.tile_size)
            {
                // Return if the caller is simply requesting the size of the storage allocation
                if (d_temp_storage == NULL)
                {
                    temp_storage_bytes = 0;
                    return cudaSuccess;
                }

                // Sort entire problem locally within a single thread block
                num_passes = 0;

                // Log single_kernel configuration
                if (debug_synchronous)
                    CubLog("Invoking single_kernel<<<%d, %d, 0, %lld>>>(), %d items per thread, %d SM occupancy, current bit %d, bit_grain %d\n",
                        1, single_config.block_threads, (long long) stream, single_config.items_per_thread, single_config.sm_occupancy, begin_bit, single_config.radix_bits);

                // Invoke upsweep_kernel with same grid size as downsweep_kernel
                single_kernel<<<1, single_config.block_threads, 0, stream>>>(
                    d_keys.Current(),
                    (ALT_STORAGE) ? d_keys.Alternate() : d_keys.Current(),
                    d_values.Current(),
                    (ALT_STORAGE) ? d_values.Alternate() : d_values.Current(),
                    num_items,
                    begin_bit,
                    end_bit);

                // Check for failure to launch
                if (CubDebug(error = cudaPeekAtLastError())) break;

                // Sync the stream if specified to flush runtime errors
                if (debug_synchronous && (CubDebug(error = SyncStream(stream)))) break;

            }
            else
            {
                // Run multiple global digit-place passes
                // Get maximum spine length (conservatively based upon the larger, primary digit size)
                int max_grid_size   = CUB_MAX(downsweep_config.max_grid_size, alt_downsweep_config.max_grid_size);
                int spine_length    = (max_grid_size * (1 << downsweep_config.radix_bits)) + scan_config.tile_size;

                // Temporary storage allocation requirements
                void* allocations[3];
                size_t allocation_sizes[3] =
                {
                    spine_length * sizeof(OffsetT),                                     // bytes needed for privatized block digit histograms
                    (!ALT_STORAGE) ? 0 : num_items * sizeof(KeyT),                       // bytes needed for 3rd keys buffer
                    (!ALT_STORAGE || (KEYS_ONLY)) ? 0 : num_items * sizeof(ValueT),      // bytes needed for 3rd values buffer
                };

                // Alias the temporary allocations from the single storage blob (or compute the necessary size of the blob)
                if (CubDebug(error = AliasTemporaries(d_temp_storage, temp_storage_bytes, allocations, allocation_sizes))) break;

                // Return if the caller is simply requesting the size of the storage allocation
                if (d_temp_storage == NULL)
                    return cudaSuccess;

                // Alias the allocation for the privatized per-block digit histograms
                OffsetT *d_spine;
                d_spine = static_cast<OffsetT*>(allocations[0]);

                // Pass planning.  Run passes of the alternate digit-size configuration until we have an even multiple of our preferred digit size
                int num_bits        = end_bit - begin_bit;
                num_passes          = (num_bits + downsweep_config.radix_bits - 1) / downsweep_config.radix_bits;
                bool is_odd_passes  = num_passes & 1;

                int max_alt_passes  = (num_passes * downsweep_config.radix_bits) - num_bits;
                int alt_end_bit     = CUB_MIN(end_bit, begin_bit + (max_alt_passes * alt_downsweep_config.radix_bits));

                DoubleBuffer<KeyT> d_keys_remaining_passes(
                    (!ALT_STORAGE || is_odd_passes) ? d_keys.Alternate() : static_cast<KeyT*>(allocations[1]),
                    (!ALT_STORAGE) ? d_keys.Current() : (is_odd_passes) ? static_cast<KeyT*>(allocations[1]) : d_keys.Alternate());

                DoubleBuffer<ValueT> d_values_remaining_passes(
                    (!ALT_STORAGE || is_odd_passes) ? d_values.Alternate() : static_cast<ValueT*>(allocations[2]),
                    (!ALT_STORAGE) ? d_values.Current() : (is_odd_passes) ? static_cast<ValueT*>(allocations[2]) : d_values.Alternate());

                // Get even-share work distribution descriptors
                GridEvenShare<OffsetT> even_share(num_items, downsweep_config.max_grid_size, CUB_MAX(downsweep_config.tile_size, upsweep_config.tile_size));
                GridEvenShare<OffsetT> alt_even_share(num_items, alt_downsweep_config.max_grid_size, CUB_MAX(alt_downsweep_config.tile_size, alt_upsweep_config.tile_size));

                // Run first pass
                int current_bit = begin_bit;
                if (current_bit < alt_end_bit)
                {
                    // Alternate digit-length pass
                    int pass_bits = CUB_MIN(alt_downsweep_config.radix_bits, (end_bit - current_bit));
                    DispatchPass(
                        d_keys.Current(), d_keys_remaining_passes.Current(),
                        d_values.Current(), d_values_remaining_passes.Current(),
                        d_spine, spine_length, num_items, current_bit, pass_bits, stream, debug_synchronous,
                        alt_upsweep_config, scan_config, alt_downsweep_config,
                        alt_upsweep_kernel, scan_kernel, alt_downsweep_kernel,
                        alt_even_share);

                    current_bit += alt_downsweep_config.radix_bits;
                }
                else
                {
                    // Preferred digit-length pass
                    int pass_bits = CUB_MIN(downsweep_config.radix_bits, (end_bit - current_bit));
                    DispatchPass(
                        d_keys.Current(), d_keys_remaining_passes.Current(),
                        d_values.Current(), d_values_remaining_passes.Current(),
                        d_spine, spine_length, num_items, current_bit, pass_bits, stream, debug_synchronous,
                        upsweep_config, scan_config, downsweep_config,
                        upsweep_kernel, scan_kernel, downsweep_kernel,
                        even_share);

                    current_bit += downsweep_config.radix_bits;
                }

                // Run remaining passes
                while (current_bit < end_bit)
                {
                    if (current_bit < alt_end_bit)
                    {
                        // Alternate digit-length pass
                        int pass_bits = CUB_MIN(alt_downsweep_config.radix_bits, (end_bit - current_bit));
                        DispatchPass(
                            d_keys_remaining_passes.d_buffers[d_keys_remaining_passes.selector],
                            d_keys_remaining_passes.d_buffers[d_keys_remaining_passes.selector ^ 1],
                            d_values_remaining_passes.d_buffers[d_keys_remaining_passes.selector],
                            d_values_remaining_passes.d_buffers[d_keys_remaining_passes.selector ^ 1],
                            d_spine, spine_length, num_items, current_bit, pass_bits, stream, debug_synchronous,
                            alt_upsweep_config, scan_config, alt_downsweep_config,
                            alt_upsweep_kernel, scan_kernel, alt_downsweep_kernel,
                            alt_even_share);

                        current_bit += alt_downsweep_config.radix_bits;
                    }
                    else
                    {
                        // Preferred digit-length pass
                        int pass_bits = CUB_MIN(downsweep_config.radix_bits, (end_bit - current_bit));
                        DispatchPass(
                            d_keys_remaining_passes.d_buffers[d_keys_remaining_passes.selector],
                            d_keys_remaining_passes.d_buffers[d_keys_remaining_passes.selector ^ 1],
                            d_values_remaining_passes.d_buffers[d_keys_remaining_passes.selector],
                            d_values_remaining_passes.d_buffers[d_keys_remaining_passes.selector ^ 1],
                            d_spine, spine_length, num_items, current_bit, pass_bits, stream, debug_synchronous,
                            upsweep_config, scan_config, downsweep_config,
                            upsweep_kernel, scan_kernel, downsweep_kernel,
                            even_share);

                        current_bit += downsweep_config.radix_bits;
                    }

                    // Invert selectors and update current bit
                    d_keys_remaining_passes.selector ^= 1;
                    d_values_remaining_passes.selector ^= 1;
                }
            }

            // Update selector
            if (ALT_STORAGE)
            {
                // Sorted data always ends up in the other vector
                d_keys.selector ^= 1;
                d_values.selector ^= 1;
            }
            else
            {
                // Where sorted data ends up depends on the number of passes
                d_keys.selector = (d_keys.selector + num_passes) & 1;
                d_values.selector = (d_values.selector + num_passes) & 1;
            }
        }
        while (0);

        return error;

#endif // CUB_RUNTIME_ENABLED
    }


    /**
     * Internal dispatch routine
     */

    CUB_RUNTIME_FUNCTION __forceinline__
    static cudaError_t Dispatch(
        void*               d_temp_storage,                ///< [in] %Device-accessible allocation of temporary storage.  When NULL, the required allocation size is written to \p temp_storage_bytes and no work is done.
        size_t                  &temp_storage_bytes,            ///< [in,out] Reference to size in bytes of \p d_temp_storage allocation
        DoubleBuffer<KeyT>       &d_keys,                        ///< [in,out] Double-buffer whose current buffer contains the unsorted input keys and, upon return, is updated to point to the sorted output keys
        DoubleBuffer<ValueT>     &d_values,                      ///< [in,out] Double-buffer whose current buffer contains the unsorted input values and, upon return, is updated to point to the sorted output values
        OffsetT                 num_items,                      ///< [in] Number of items to reduce
        int                     begin_bit,                      ///< [in] The beginning (least-significant) bit index needed for key comparison
        int                     end_bit,                        ///< [in] The past-the-end (most-significant) bit index needed for key comparison
        cudaStream_t            stream,                         ///< [in] CUDA stream to launch kernels within.  Default is stream<sub>0</sub>.
        bool                    debug_synchronous)              ///< [in] Whether or not to synchronize the stream after every kernel launch to check for errors.  Also causes launch configurations to be printed to the console.  Default is \p false.
    {
        return Dispatch(
            d_temp_storage,
            temp_storage_bytes,
            d_keys,
            d_values,
            num_items,
            begin_bit,
            end_bit,
            stream,
            debug_synchronous,
            DeviceRadixSortUpsweepKernel<PtxUpsweepPolicy, DESCENDING, KeyT, OffsetT>,
            DeviceRadixSortUpsweepKernel<PtxAltUpsweepPolicy, DESCENDING, KeyT, OffsetT>,
            RadixSortScanBinsKernel<PtxScanPolicy, OffsetT>,
            DeviceRadixSortDownsweepKernel<PtxDownsweepPolicy, DESCENDING, KeyT, ValueT, OffsetT>,
            DeviceRadixSortDownsweepKernel<PtxAltDownsweepPolicy, DESCENDING, KeyT, ValueT, OffsetT>,
            DeviceRadixSortSingleKernel<PtxSinglePolicy, DESCENDING, KeyT, ValueT, OffsetT>);
    }

};

}               // CUB namespace
CUB_NS_POSTFIX  // Optional outer namespace(s)


