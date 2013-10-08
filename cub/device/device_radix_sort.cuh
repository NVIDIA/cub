
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
 * cub::DeviceRadixSort provides operations for computing a device-wide, parallel reduction across data items residing within global memory.
 */

#pragma once

#include <stdio.h>
#include <iterator>

#include "block/block_radix_sort_upsweep_tiles.cuh"
#include "block/block_radix_sort_downsweep_tiles.cuh"
#include "block/block_scan_tiles.cuh"
#include "../grid/grid_even_share.cuh"
#include "../util_debug.cuh"
#include "../util_device.cuh"
#include "../util_namespace.cuh"

/// Optional outer namespace(s)
CUB_NS_PREFIX

/// CUB namespace
namespace cub {

#ifndef DOXYGEN_SHOULD_SKIP_THIS    // Do not document




/******************************************************************************
 * Kernel entry points
 *****************************************************************************/

/**
 * Upsweep pass kernel entry point (multi-block).  Computes privatized digit histograms, one per block.
 */
template <
    typename                BlockRadixSortUpsweepTilesPolicy, ///< Tuning policy for cub::BlockRadixSortUpsweepTiles abstraction
    typename                Key,                            ///< Key type
    typename                SizeT>                          ///< Integer type used for global array indexing
__launch_bounds__ (int(BlockRadixSortUpsweepTilesPolicy::BLOCK_THREADS), 1)
__global__ void RadixSortUpsweepKernel(
    Key                     *d_keys,                        ///< [in] Input keys buffer
    SizeT                   *d_spine,                       ///< [out] Privatized (per block) digit histograms (striped, i.e., 0s counts from each block, then 1s counts from each block, etc.)
    SizeT                   num_items,                      ///< [in] Total number of input data items
    int                     current_bit,                    ///< [in] Bit position of current radix digit
    bool                    use_primary_bit_granularity,    ///< [in] Whether nor not to use the primary policy (or the embedded alternate policy for smaller bit granularity)
    bool                    first_pass,                     ///< [in] Whether this is the first digit pass
    GridEvenShare<SizeT>    even_share)                     ///< [in] Even-share descriptor for mapping an equal number of tiles onto each thread block
{

    // Alternate policy for when fewer bits remain
    typedef typename BlockRadixSortUpsweepTilesPolicy::AltPolicy AltPolicy;

    // Parameterize two versions of BlockRadixSortUpsweepTiles type for the current configuration
    typedef BlockRadixSortUpsweepTiles<BlockRadixSortUpsweepTilesPolicy, Key, SizeT>    BlockRadixSortUpsweepTilesT;          // Primary
    typedef BlockRadixSortUpsweepTiles<AltPolicy, Key, SizeT>                           AltBlockRadixSortUpsweepTilesT;       // Alternate (smaller bit granularity)

    // Shared memory storage
    __shared__ union
    {
        typename BlockRadixSortUpsweepTilesT::TempStorage     pass_storage;
        typename AltBlockRadixSortUpsweepTilesT::TempStorage  alt_pass_storage;
    } temp_storage;

    // Initialize even-share descriptor for this thread block
    even_share.BlockInit();

    // Process input tiles (each of the first RADIX_DIGITS threads will compute a count for that digit)
    if (use_primary_bit_granularity)
    {
        // Primary granularity
        SizeT bin_count;
        BlockRadixSortUpsweepTilesT(temp_storage.pass_storage, d_keys, current_bit).ProcessTiles(
            even_share.block_offset,
            even_share.block_end,
            bin_count);

        // Write out digit counts (striped)
        if (threadIdx.x < BlockRadixSortUpsweepTilesT::RADIX_DIGITS)
        {
            d_spine[(gridDim.x * threadIdx.x) + blockIdx.x] = bin_count;
        }
    }
    else
    {
        // Alternate granularity
        // Process input tiles (each of the first RADIX_DIGITS threads will compute a count for that digit)
        SizeT bin_count;
        AltBlockRadixSortUpsweepTilesT(temp_storage.alt_pass_storage, d_keys, current_bit).ProcessTiles(
            even_share.block_offset,
            even_share.block_end,
            bin_count);

        // Write out digit counts (striped)
        if (threadIdx.x < AltBlockRadixSortUpsweepTilesT::RADIX_DIGITS)
        {
            d_spine[(gridDim.x * threadIdx.x) + blockIdx.x] = bin_count;
        }
    }
}


/**
 * Spine scan kernel entry point (single-block).  Computes an exclusive prefix sum over the privatized digit histograms
 */
template <
    typename    BlockScanTilesPolicy,   ///< Tuning policy for cub::BlockScanTiles abstraction
    typename    SizeT>                  ///< Integer type used for global array indexing
__launch_bounds__ (int(BlockScanTilesPolicy::BLOCK_THREADS), 1)
__global__ void RadixSortScanKernel(
    SizeT       *d_spine,               ///< [in,out] Privatized (per block) digit histograms (striped, i.e., 0s counts from each block, then 1s counts from each block, etc.)
    int         num_counts)             ///< [in] Total number of bin-counts
{
    // Parameterize the BlockScanTiles type for the current configuration
    typedef BlockScanTiles<BlockScanTilesPolicy, SizeT*, SizeT*, cub::Sum, SizeT, SizeT> BlockScanTilesT;

    // Shared memory storage
    __shared__ typename BlockScanTilesT::TempStorage temp_storage;

    if (blockIdx.x > 0) return;

    // Block scan instance
    BlockScanTilesT block_scan(temp_storage, d_spine, d_spine, cub::Sum(), SizeT(0)) ;

    // Process full input tiles
    int block_offset = 0;
    RunningBlockPrefixOp<SizeT> prefix_op;
    prefix_op.running_total = 0;
    while (block_offset + BlockScanTilesT::TILE_ITEMS <= num_counts)
    {
        block_scan.ConsumeTile<true, false>(block_offset, prefix_op);
        block_offset += BlockScanTilesT::TILE_ITEMS;
    }
}


/**
 * Downsweep pass kernel entry point (multi-block).  Scatters keys (and values) into corresponding bins for the current digit place.
 */
template <
    typename                BlockRadixSortDownsweepTilesPolicy,   ///< Tuning policy for cub::BlockRadixSortUpsweepTiles abstraction
    typename                Key,                                ///< Key type
    typename                Value,                              ///< Value type
    typename                SizeT>                              ///< Integer type used for global array indexing
__launch_bounds__ (int(BlockRadixSortDownsweepTilesPolicy::BLOCK_THREADS))
__global__ void RadixSortDownsweepKernel(
    Key                     *d_keys_in,                     ///< [in] Input keys ping buffer
    Key                     *d_keys_out,                    ///< [in] Output keys pong buffer
    Value                   *d_values_in,                   ///< [in] Input values ping buffer
    Value                   *d_values_out,                  ///< [in] Output values pong buffer
    SizeT                   *d_spine,                       ///< [in] Scan of privatized (per block) digit histograms (striped, i.e., 0s counts from each block, then 1s counts from each block, etc.)
    SizeT                   num_items,                      ///< [in] Total number of input data items
    int                     current_bit,                    ///< [in] Bit position of current radix digit
    bool                    use_primary_bit_granularity,    ///< [in] Whether nor not to use the primary policy (or the embedded alternate policy for smaller bit granularity)
    bool                    first_pass,                     ///< [in] Whether this is the first digit pass
    bool                    last_pass,                      ///< [in] Whether this is the last digit pass
    GridEvenShare<SizeT>    even_share)                     ///< [in] Even-share descriptor for mapping an equal number of tiles onto each thread block
{

    // Alternate policy for when fewer bits remain
    typedef typename BlockRadixSortDownsweepTilesPolicy::AltPolicy AltPolicy;

    // Parameterize two versions of BlockRadixSortDownsweepTiles type for the current configuration
    typedef BlockRadixSortDownsweepTiles<BlockRadixSortDownsweepTilesPolicy, Key, Value, SizeT>     BlockRadixSortDownsweepTilesT;
    typedef BlockRadixSortDownsweepTiles<AltPolicy, Key, Value, SizeT>                            AltBlockRadixSortDownsweepTilesT;

    // Shared memory storage
    __shared__ union
    {
        typename BlockRadixSortDownsweepTilesT::TempStorage       pass_storage;
        typename AltBlockRadixSortDownsweepTilesT::TempStorage    alt_pass_storage;

    } temp_storage;

    // Initialize even-share descriptor for this thread block
    even_share.BlockInit();

    if (use_primary_bit_granularity)
    {
        // Process input tiles
        BlockRadixSortDownsweepTilesT(temp_storage.pass_storage, num_items, d_spine, d_keys_in, d_keys_out, d_values_in, d_values_out, current_bit).ProcessTiles(
            even_share.block_offset,
            even_share.block_end);
    }
    else
    {
        // Process input tiles
        AltBlockRadixSortDownsweepTilesT(temp_storage.alt_pass_storage, num_items, d_spine, d_keys_in, d_keys_out, d_values_in, d_values_out, current_bit).ProcessTiles(
            even_share.block_offset,
            even_share.block_end);
    }
}



/******************************************************************************
 * Dispatch
 ******************************************************************************/

/**
 * Utility class for dispatching the appropriately-tuned kernels for DeviceRadixSort
 */
template <
    typename Key,            ///< Key type
    typename Value,          ///< Value type
    typename SizeT>          ///< Integer type used for global array indexing
struct DeviceRadixSortDispatch
{
    /******************************************************************************
     * Tuning policies
     ******************************************************************************/

    /// SM35
    struct Policy350
    {
        enum {
            KEYS_ONLY       = (Equals<Value, NullType>::VALUE),
            SCALE_FACTOR    = (CUB_MAX(sizeof(Key), sizeof(Value)) + 3) / 4,
            RADIX_BITS      = 5,
        };

        // UpsweepPolicy
        typedef BlockRadixSortUpsweepTilesPolicy <64,     CUB_MAX(1, 18 / SCALE_FACTOR), LOAD_LDG, RADIX_BITS> UpsweepPolicyKeys;
        typedef BlockRadixSortUpsweepTilesPolicy <128,    CUB_MAX(1, 15 / SCALE_FACTOR), LOAD_LDG, RADIX_BITS> UpsweepPolicyPairs;
        typedef typename If<KEYS_ONLY, UpsweepPolicyKeys, UpsweepPolicyPairs>::Type UpsweepPolicy;

        // ScanPolicy
        typedef BlockScanTilesPolicy <1024, 4, BLOCK_LOAD_VECTORIZE, false, LOAD_DEFAULT, BLOCK_STORE_VECTORIZE, false, BLOCK_SCAN_RAKING_MEMOIZE> ScanPolicy;

        // DownsweepPolicy
        typedef BlockRadixSortDownsweepTilesPolicy <64,   CUB_MAX(1, 18 / SCALE_FACTOR), BLOCK_LOAD_DIRECT, LOAD_LDG, false, true, BLOCK_SCAN_WARP_SCANS, RADIX_SORT_SCATTER_TWO_PHASE, cudaSharedMemBankSizeEightByte, RADIX_BITS> DownsweepPolicyKeys;
        typedef BlockRadixSortDownsweepTilesPolicy <128,  CUB_MAX(1, 15 / SCALE_FACTOR), BLOCK_LOAD_DIRECT, LOAD_LDG, false, true, BLOCK_SCAN_WARP_SCANS, RADIX_SORT_SCATTER_TWO_PHASE, cudaSharedMemBankSizeEightByte, RADIX_BITS> DownsweepPolicyPairs;
        typedef typename If<KEYS_ONLY, DownsweepPolicyKeys, DownsweepPolicyPairs>::Type DownsweepPolicy;

/*
        // 4bit

        typedef BlockRadixSortUpsweepTilesPolicy <128, 15, LOAD_LDG, RADIX_BITS> UpsweepPolicyKeys;
        typedef BlockRadixSortUpsweepTilesPolicy <256, 13, LOAD_LDG, RADIX_BITS> UpsweepPolicyPairs;

        typedef BlockRadixSortDownsweepTilesPolicy <128, 15, BLOCK_LOAD_DIRECT, LOAD_LDG, false, true, BLOCK_SCAN_WARP_SCANS, RADIX_SORT_SCATTER_TWO_PHASE, cudaSharedMemBankSizeEightByte, RADIX_BITS> DownsweepPolicyKeys;
        typedef BlockRadixSortDownsweepTilesPolicy <256, 13, BLOCK_LOAD_DIRECT, LOAD_LDG, false, true, BLOCK_SCAN_WARP_SCANS, RADIX_SORT_SCATTER_TWO_PHASE, cudaSharedMemBankSizeEightByte, RADIX_BITS> DownsweepPolicyPairs;
*/
    };


    /// SM30
    struct Policy300
    {
        enum {
            KEYS_ONLY       = (Equals<Value, NullType>::VALUE),
            SCALE_FACTOR    = (CUB_MAX(sizeof(Key), sizeof(Value)) + 3) / 4,
            RADIX_BITS      = 5,
        };

        // UpsweepPolicy
        typedef BlockRadixSortUpsweepTilesPolicy <256, CUB_MAX(1, 7 / SCALE_FACTOR), LOAD_DEFAULT, RADIX_BITS> UpsweepPolicyKeys;
        typedef BlockRadixSortUpsweepTilesPolicy <256, CUB_MAX(1, 5 / SCALE_FACTOR), LOAD_DEFAULT, RADIX_BITS> UpsweepPolicyPairs;
        typedef typename If<KEYS_ONLY, UpsweepPolicyKeys, UpsweepPolicyPairs>::Type UpsweepPolicy;

        // ScanPolicy
        typedef BlockScanTilesPolicy <1024, 4, BLOCK_LOAD_VECTORIZE, false, LOAD_DEFAULT, BLOCK_STORE_VECTORIZE, false, BLOCK_SCAN_RAKING_MEMOIZE> ScanPolicy;

        // DownsweepPolicy
        typedef BlockRadixSortDownsweepTilesPolicy <128, CUB_MAX(1, 14 / SCALE_FACTOR), BLOCK_LOAD_WARP_TRANSPOSE, LOAD_DEFAULT, false, false, BLOCK_SCAN_WARP_SCANS, RADIX_SORT_SCATTER_TWO_PHASE, cudaSharedMemBankSizeEightByte, RADIX_BITS> DownsweepPolicyKeys;
        typedef BlockRadixSortDownsweepTilesPolicy <128, CUB_MAX(1, 10 / SCALE_FACTOR), BLOCK_LOAD_TRANSPOSE, LOAD_DEFAULT, false, false, BLOCK_SCAN_WARP_SCANS, RADIX_SORT_SCATTER_TWO_PHASE, cudaSharedMemBankSizeEightByte, RADIX_BITS> DownsweepPolicyPairs;
        typedef typename If<KEYS_ONLY, DownsweepPolicyKeys, DownsweepPolicyPairs>::Type DownsweepPolicy;
    };


    /// SM20
    struct Policy200
    {
        enum {
            KEYS_ONLY       = (Equals<Value, NullType>::VALUE),
            SCALE_FACTOR    = (CUB_MAX(sizeof(Key), sizeof(Value)) + 3) / 4,
            RADIX_BITS      = 5,
        };

        // UpsweepPolicy
        typedef BlockRadixSortUpsweepTilesPolicy <64, CUB_MAX(1, 18 / SCALE_FACTOR), LOAD_DEFAULT, RADIX_BITS> UpsweepPolicyKeys;
        typedef BlockRadixSortUpsweepTilesPolicy <128, CUB_MAX(1, 13 / SCALE_FACTOR), LOAD_DEFAULT, RADIX_BITS> UpsweepPolicyPairs;
        typedef typename If<KEYS_ONLY, UpsweepPolicyKeys, UpsweepPolicyPairs>::Type UpsweepPolicy;

        // ScanPolicy
        typedef BlockScanTilesPolicy <512, 4, BLOCK_LOAD_VECTORIZE, false, LOAD_DEFAULT, BLOCK_STORE_VECTORIZE, false, BLOCK_SCAN_RAKING_MEMOIZE> ScanPolicy;

        // DownsweepPolicy
        typedef BlockRadixSortDownsweepTilesPolicy <64, CUB_MAX(1, 18 / SCALE_FACTOR), BLOCK_LOAD_WARP_TRANSPOSE, LOAD_DEFAULT, false, false, BLOCK_SCAN_WARP_SCANS, RADIX_SORT_SCATTER_TWO_PHASE, cudaSharedMemBankSizeFourByte, RADIX_BITS> DownsweepPolicyKeys;
        typedef BlockRadixSortDownsweepTilesPolicy <128, CUB_MAX(1, 13 / SCALE_FACTOR), BLOCK_LOAD_WARP_TRANSPOSE, LOAD_DEFAULT, false, false, BLOCK_SCAN_WARP_SCANS, RADIX_SORT_SCATTER_TWO_PHASE, cudaSharedMemBankSizeFourByte, RADIX_BITS> DownsweepPolicyPairs;
        typedef typename If<KEYS_ONLY, DownsweepPolicyKeys, DownsweepPolicyPairs>::Type DownsweepPolicy;
    };


    /// SM13
    struct Policy130
    {
        enum {
            KEYS_ONLY       = (Equals<Value, NullType>::VALUE),
            SCALE_FACTOR    = (CUB_MAX(sizeof(Key), sizeof(Value)) + 3) / 4,
            RADIX_BITS      = 5,
        };

        // UpsweepPolicy
        typedef BlockRadixSortUpsweepTilesPolicy <128, CUB_MAX(1, 15 / SCALE_FACTOR), LOAD_DEFAULT, RADIX_BITS> UpsweepPolicyKeys;
        typedef BlockRadixSortUpsweepTilesPolicy <128, CUB_MAX(1, 15 / SCALE_FACTOR), LOAD_DEFAULT, RADIX_BITS> UpsweepPolicyPairs;
        typedef typename If<KEYS_ONLY, UpsweepPolicyKeys, UpsweepPolicyPairs>::Type UpsweepPolicy;

        // ScanPolicy
        typedef BlockScanTilesPolicy <256, 4, BLOCK_LOAD_VECTORIZE, false, LOAD_DEFAULT, BLOCK_STORE_VECTORIZE, false, BLOCK_SCAN_RAKING_MEMOIZE> ScanPolicy;

        // DownsweepPolicy
        typedef BlockRadixSortDownsweepTilesPolicy <64, CUB_MAX(1, 15 / SCALE_FACTOR), BLOCK_LOAD_WARP_TRANSPOSE, LOAD_DEFAULT, false, true, BLOCK_SCAN_WARP_SCANS, RADIX_SORT_SCATTER_TWO_PHASE, cudaSharedMemBankSizeFourByte, RADIX_BITS> DownsweepPolicyKeys;
        typedef BlockRadixSortDownsweepTilesPolicy <64, CUB_MAX(1, 15 / SCALE_FACTOR), BLOCK_LOAD_WARP_TRANSPOSE, LOAD_DEFAULT, false, true, BLOCK_SCAN_WARP_SCANS, RADIX_SORT_SCATTER_TWO_PHASE, cudaSharedMemBankSizeFourByte, RADIX_BITS> DownsweepPolicyPairs;
        typedef typename If<KEYS_ONLY, DownsweepPolicyKeys, DownsweepPolicyPairs>::Type DownsweepPolicy;
    };


    /// SM10
    struct Policy100
    {
        enum {
            RADIX_BITS = 4,
        };

        // UpsweepPolicy
        typedef BlockRadixSortUpsweepTilesPolicy <64, 9, LOAD_DEFAULT, RADIX_BITS> UpsweepPolicy;

        // ScanPolicy
        typedef BlockScanTilesPolicy <256, 4, BLOCK_LOAD_VECTORIZE, false, LOAD_DEFAULT, BLOCK_STORE_VECTORIZE, false, BLOCK_SCAN_RAKING_MEMOIZE> ScanPolicy;

        // DownsweepPolicy
        typedef BlockRadixSortDownsweepTilesPolicy <64, 9, BLOCK_LOAD_WARP_TRANSPOSE, LOAD_DEFAULT, false, false, BLOCK_SCAN_WARP_SCANS, RADIX_SORT_SCATTER_TWO_PHASE, cudaSharedMemBankSizeFourByte, RADIX_BITS> DownsweepPolicy;
    };


    /******************************************************************************
     * Tuning policies of current PTX compiler pass
     ******************************************************************************/

#if (CUB_PTX_VERSION >= 350)
    typedef Policy350 PtxPolicy;

#elif (CUB_PTX_VERSION >= 300)
    typedef Policy300 PtxPolicy;

#elif (CUB_PTX_VERSION >= 200)
    typedef Policy200 PtxPolicy;

#elif (CUB_PTX_VERSION >= 130)
    typedef Policy130 PtxPolicy;

#else
    typedef Policy100 PtxPolicy;

#endif

    // "Opaque" policies (whose parameterizations aren't reflected in the type signature)
    struct PtxUpsweepPolicy    : PtxPolicy::UpsweepPolicy {};
    struct PtxScanPolicy       : PtxPolicy::ScanPolicy {};
    struct PtxDownsweepPolicy  : PtxPolicy::DownsweepPolicy {};


    /******************************************************************************
     * Utilities
     ******************************************************************************/

    /**
     * Initialize dispatch configurations with the policies corresponding to the PTX assembly we will use
     */
    template <
        typename Policy,
        typename KernelDispatchConfig>
    __host__ __device__ __forceinline__
    static void InitDispatchConfigs(
        KernelDispatchConfig    &upsweep_config,
        KernelDispatchConfig    &scan_config,
        KernelDispatchConfig    &downsweep_config)
    {
        upsweep_config.template     InitUpsweepPolicy<typename Policy::UpsweepPolicy>();
        scan_config.template        InitScanPolicy<typename Policy::ScanPolicy>();
        downsweep_config.template   InitDownsweepPolicy<typename Policy::DownsweepPolicy>();
    }


    /**
     * Initialize dispatch configurations with the policies corresponding to the PTX assembly we will use
     */
    template <typename KernelDispatchConfig>
    __host__ __device__ __forceinline__
    static void InitDispatchConfigs(
        int                     ptx_version,
        KernelDispatchConfig    &upsweep_config,
        KernelDispatchConfig    &scan_config,
        KernelDispatchConfig    &downsweep_config)
    {
    #ifdef __CUDA_ARCH__

        // We're on the device, so initialize the dispatch configurations with the PtxDefaultPolicies directly
        upsweep_config.InitUpsweepPolicy<PtxUpsweepPolicy>();
        scan_config.InitScanPolicy<PtxScanPolicy>();
        downsweep_config.InitDownsweepPolicy<PtxDownsweepPolicy>();

    #else

        // We're on the host, so lookup and initialize the dispatch configurations with the policies that match the device's PTX version
        if (ptx_version >= 350)
        {
            InitDispatchConfigs<Policy350>(upsweep_config, scan_config, downsweep_config);
        }
        else if (ptx_version >= 300)
        {
            InitDispatchConfigs<Policy300>(upsweep_config, scan_config, downsweep_config);
        }
        else if (ptx_version >= 200)
        {
            InitDispatchConfigs<Policy200>(upsweep_config, scan_config, downsweep_config);
        }
        else if (ptx_version >= 130)
        {
            InitDispatchConfigs<Policy130>(upsweep_config, scan_config, downsweep_config);
        }
        else
        {
            InitDispatchConfigs<Policy100>(upsweep_config, scan_config, downsweep_config);
        }

    #endif
    }



    /**
     * Kernel dispatch configurations
     */
    struct KernelDispatchConfig
    {
        int                     block_threads;
        int                     items_per_thread;
        cudaSharedMemConfig     smem_config;
        int                     radix_bits;
        int                     alt_radix_bits;

        template <typename SortBlockPolicy>
        __host__ __device__ __forceinline__ void InitUpsweepPolicy()
        {
            block_threads               = SortBlockPolicy::BLOCK_THREADS;
            items_per_thread            = SortBlockPolicy::ITEMS_PER_THREAD;
            radix_bits                  = SortBlockPolicy::RADIX_BITS;
            alt_radix_bits              = SortBlockPolicy::AltPolicy::RADIX_BITS;
            smem_config                 = cudaSharedMemBankSizeFourByte;
        }

        template <typename ScanBlockPolicy>
        __host__ __device__ __forceinline__ void InitScanPolicy()
        {
            block_threads               = ScanBlockPolicy::BLOCK_THREADS;
            items_per_thread            = ScanBlockPolicy::ITEMS_PER_THREAD;
            radix_bits                  = 0;
            alt_radix_bits              = 0;
            smem_config                 = cudaSharedMemBankSizeFourByte;
        }

        template <typename SortBlockPolicy>
        __host__ __device__ __forceinline__ void InitDownsweepPolicy()
        {
            block_threads               = SortBlockPolicy::BLOCK_THREADS;
            items_per_thread            = SortBlockPolicy::ITEMS_PER_THREAD;
            radix_bits                  = SortBlockPolicy::RADIX_BITS;
            alt_radix_bits              = SortBlockPolicy::AltPolicy::RADIX_BITS;
            smem_config                 = SortBlockPolicy::SMEM_CONFIG;
        }
    };


    /******************************************************************************
     * Dispatch entrypoints
     ******************************************************************************/

    /**
     * Internal dispatch routine for computing a device-wide radix sort using the
     * specified kernel functions.
     */    template <
        typename            UpsweepKernelPtr,                   ///< Function type of cub::RadixSortUpsweepKernel
        typename            SpineKernelPtr,                     ///< Function type of cub::SpineScanKernel
        typename            DownsweepKernelPtr>                 ///< Function type of cub::RadixSortUpsweepKernel
    __host__ __device__ __forceinline__
    static cudaError_t Dispatch(
        void                    *d_temp_storage,                ///< [in] %Device allocation of temporary storage.  When NULL, the required allocation size is returned in \p temp_storage_bytes and no work is done.
        size_t                  &temp_storage_bytes,            ///< [in,out] Size in bytes of \p d_temp_storage allocation.
        DoubleBuffer<Key>       &d_keys,                        ///< [in,out] Double-buffer whose current buffer contains the unsorted input keys and, upon return, is updated to point to the sorted output keys
        DoubleBuffer<Value>     &d_values,                      ///< [in,out] Double-buffer whose current buffer contains the unsorted input values and, upon return, is updated to point to the sorted output values
        SizeT                   num_items,                      ///< [in] Number of items to reduce
        int                     begin_bit,                      ///< [in] <b>[optional]</b> The beginning (least-significant) bit index needed for key comparison
        int                     end_bit,                        ///< [in] <b>[optional]</b> The past-the-end (most-significant) bit index needed for key comparison
        cudaStream_t            stream,                         ///< [in] <b>[optional]</b> CUDA stream to launch kernels within.  Default is stream<sub>0</sub>.
        bool                    debug_synchronous,              ///< [in] <b>[optional]</b> Whether or not to synchronize the stream after every kernel launch to check for errors.  Also causes launch configurations to be printed to the console.  Default is \p false.
        int                     sm_version,                     ///< [in] SM version of target device to use when computing SM occupancy
        UpsweepKernelPtr        upsweep_kernel,                 ///< [in] Kernel function pointer to parameterization of cub::RadixSortUpsweepKernel
        SpineKernelPtr          scan_kernel,                    ///< [in] Kernel function pointer to parameterization of cub::SpineScanKernel
        DownsweepKernelPtr      downsweep_kernel,               ///< [in] Kernel function pointer to parameterization of cub::RadixSortUpsweepKernel
        KernelDispatchConfig    &upsweep_config,                ///< [in] Dispatch parameters that match the policy that \p upsweep_kernel was compiled for
        KernelDispatchConfig    &scan_config,                   ///< [in] Dispatch parameters that match the policy that \p scan_kernel was compiled for
        KernelDispatchConfig    &downsweep_config)              ///< [in] Dispatch parameters that match the policy that \p downsweep_kernel was compiled for
    {
#ifndef CUB_RUNTIME_ENABLED

        // Kernel launch not supported from this device
        return CubDebug(cudaErrorNotSupported );

#else

        cudaError error = cudaSuccess;
        do
        {
            int bins = 1 << downsweep_config.radix_bits;

            // Get device ordinal
            int device_ordinal;
            if (CubDebug(error = cudaGetDevice(&device_ordinal))) break;

            // Get SM count
            int sm_count;
            if (CubDebug(error = cudaDeviceGetAttribute (&sm_count, cudaDevAttrMultiProcessorCount, device_ordinal))) break;

            // Get device occupancy for kernels
            int upsweep_sm_occupancy;
            if (CubDebug(error = MaxSmOccupancy(
                upsweep_sm_occupancy,
                sm_version,
                upsweep_kernel,
                upsweep_config.block_threads))) break;

            int downsweep_sm_occupancy;
            if (CubDebug(error = MaxSmOccupancy(
                downsweep_sm_occupancy,
                sm_version,
                downsweep_kernel,
                downsweep_config.block_threads))) break;

            // Get device occupancy for downsweep_kernel
            int downsweep_occupancy = downsweep_sm_occupancy * sm_count;

            // Get tile sizes
            int downsweep_tile_size = downsweep_config.block_threads * downsweep_config.items_per_thread;
            int scan_tile_size = scan_config.block_threads * scan_config.items_per_thread;

            // Get even-share work distribution descriptor
            int subscription_factor = downsweep_sm_occupancy;     // Amount of CTAs to oversubscribe the device beyond actively-resident (heuristic)
            GridEvenShare<SizeT> even_share(
                num_items,
                downsweep_occupancy * subscription_factor,
                downsweep_tile_size);

            // Get grid size for downsweep_kernel
            int downsweep_grid_size = even_share.grid_size;

            // Get spine size (conservative)
            int spine_size = (downsweep_grid_size * bins) + scan_tile_size;

            // Temporary storage allocation requirements
            void* allocations[1];
            size_t allocation_sizes[1] =
            {
                spine_size * sizeof(SizeT),    // bytes needed for privatized block digit histograms
            };

            // Alias the temporary allocations from the single storage blob (or set the necessary size of the blob)
            if (CubDebug(error = AliasTemporaries(d_temp_storage, temp_storage_bytes, allocations, allocation_sizes))) break;
            if (d_temp_storage == NULL)
            {
                // Return if the caller is simply requesting the size of the storage allocation
                return cudaSuccess;
            }

            // Alias the allocation for the privatized per-block digit histograms
            SizeT *d_spine = (SizeT*) allocations[0];

#ifndef __CUDA_ARCH__
            // Get current smem bank configuration
            cudaSharedMemConfig original_smem_config;
            if (CubDebug(error = cudaDeviceGetSharedMemConfig(&original_smem_config))) break;
            cudaSharedMemConfig current_smem_config = original_smem_config;
#endif
            // Iterate over digit places
            int current_bit = begin_bit;
            while (current_bit < end_bit)
            {
                // Use primary bit granularity if bits remaining is a whole multiple of bit primary granularity
                int bits_remaining = end_bit - current_bit;
                bool use_primary_bit_granularity = (bits_remaining % downsweep_config.radix_bits == 0);
                int radix_bits = (use_primary_bit_granularity) ?
                    downsweep_config.radix_bits :
                    downsweep_config.alt_radix_bits;

#ifndef __CUDA_ARCH__
                // Update smem config if necessary
                if (current_smem_config != upsweep_config.smem_config)
                {
                    if (CubDebug(error = cudaDeviceSetSharedMemConfig(upsweep_config.smem_config))) break;
                    current_smem_config = upsweep_config.smem_config;
                }
#endif

                // Log upsweep_kernel configuration
                if (debug_synchronous)
                    CubLog("Invoking upsweep_kernel<<<%d, %d, 0, %lld>>>(), %d smem config, %d items per thread, %d SM occupancy, selector %d, current bit %d, bit_grain %d\n",
                    downsweep_grid_size, upsweep_config.block_threads, (long long) stream, upsweep_config.smem_config, upsweep_config.items_per_thread, upsweep_sm_occupancy, d_keys.selector, current_bit, radix_bits);

                // Invoke upsweep_kernel with same grid size as downsweep_kernel
                upsweep_kernel<<<downsweep_grid_size, upsweep_config.block_threads, 0, stream>>>(
                    d_keys.d_buffers[d_keys.selector],
                    d_spine,
                    num_items,
                    current_bit,
                    use_primary_bit_granularity,
                    (current_bit == begin_bit),
                    even_share);

                // Sync the stream if specified
                if (debug_synchronous && (CubDebug(error = SyncStream(stream)))) break;

                // Log scan_kernel configuration
                if (debug_synchronous) CubLog("Invoking scan_kernel<<<%d, %d, 0, %lld>>>(), %d items per thread\n",
                    1, scan_config.block_threads, (long long) stream, scan_config.items_per_thread);

                // Invoke scan_kernel
                scan_kernel<<<1, scan_config.block_threads, 0, stream>>>(
                    d_spine,
                    spine_size);

                // Sync the stream if specified
                if (debug_synchronous && (CubDebug(error = SyncStream(stream)))) break;

#ifndef __CUDA_ARCH__
                // Update smem config if necessary
                if (current_smem_config != downsweep_config.smem_config)
                {
                    if (CubDebug(error = cudaDeviceSetSharedMemConfig(downsweep_config.smem_config))) break;
                    current_smem_config = downsweep_config.smem_config;
                }
#endif

                // Log downsweep_kernel configuration
                if (debug_synchronous) CubLog("Invoking downsweep_kernel<<<%d, %d, 0, %lld>>>(), %d smem config, %d items per thread, %d SM occupancy\n",
                    downsweep_grid_size, downsweep_config.block_threads, (long long) stream, downsweep_config.smem_config, downsweep_config.items_per_thread, downsweep_sm_occupancy);

                // Invoke downsweep_kernel
                downsweep_kernel<<<downsweep_grid_size, downsweep_config.block_threads, 0, stream>>>(
                    d_keys.d_buffers[d_keys.selector],
                    d_keys.d_buffers[d_keys.selector ^ 1],
                    d_values.d_buffers[d_values.selector],
                    d_values.d_buffers[d_values.selector ^ 1],
                    d_spine,
                    num_items,
                    current_bit,
                    use_primary_bit_granularity,
                    (current_bit == begin_bit),
                    (current_bit + downsweep_config.radix_bits >= end_bit),
                    even_share);

                // Sync the stream if specified
                if (debug_synchronous && (CubDebug(error = SyncStream(stream)))) break;

                // Invert selectors
                d_keys.selector ^= 1;
                d_values.selector ^= 1;

                // Update current bit position
                current_bit += radix_bits;
            }

#ifndef __CUDA_ARCH__
            // Reset smem config if necessary
            if (current_smem_config != original_smem_config)
            {
                if (CubDebug(error = cudaDeviceSetSharedMemConfig(original_smem_config))) break;
            }
#endif

        }
        while (0);

        return error;

#endif // CUB_RUNTIME_ENABLED
    }


    /**
     * Internal dispatch routine
     */
    __host__ __device__ __forceinline__
    static cudaError_t Dispatch(
        void                    *d_temp_storage,                ///< [in] %Device allocation of temporary storage.  When NULL, the required allocation size is returned in \p temp_storage_bytes and no work is done.
        size_t                  &temp_storage_bytes,            ///< [in,out] Size in bytes of \p d_temp_storage allocation.
        DoubleBuffer<Key>       &d_keys,                        ///< [in,out] Double-buffer whose current buffer contains the unsorted input keys and, upon return, is updated to point to the sorted output keys
        DoubleBuffer<Value>     &d_values,                      ///< [in,out] Double-buffer whose current buffer contains the unsorted input values and, upon return, is updated to point to the sorted output values
        SizeT                   num_items,                      ///< [in] Number of items to reduce
        int                     begin_bit,                      ///< [in] <b>[optional]</b> The beginning (least-significant) bit index needed for key comparison
        int                     end_bit,                        ///< [in] <b>[optional]</b> The past-the-end (most-significant) bit index needed for key comparison
        cudaStream_t            stream,                         ///< [in] <b>[optional]</b> CUDA stream to launch kernels within.  Default is stream<sub>0</sub>.
        bool                    debug_synchronous)              ///< [in] <b>[optional]</b> Whether or not to synchronize the stream after every kernel launch to check for errors.  Also causes launch configurations to be printed to the console.  Default is \p false.
    {
        cudaError error = cudaSuccess;
        do
        {
            // Get PTX version
            int ptx_version;
    #ifndef __CUDA_ARCH__
            if (CubDebug(error = PtxVersion(ptx_version))) break;
    #else
            ptx_version = CUB_PTX_VERSION;
    #endif

            // Get kernel dispatch configurations
            KernelDispatchConfig    upsweep_config;
            KernelDispatchConfig    scan_config;
            KernelDispatchConfig    downsweep_config;
            InitDispatchConfigs(ptx_version, upsweep_config, scan_config, downsweep_config);

            // Dispatch
            if (CubDebug(error = Dispatch(
                d_temp_storage,
                temp_storage_bytes,
                d_keys,
                d_values,
                num_items,
                begin_bit,
                end_bit,
                stream,
                debug_synchronous,
                ptx_version,            // Use PTX version instead of SM version because, as a statically known quantity, this improves device-side launch dramatically but at the risk of imprecise occupancy calculation for mismatches
                RadixSortUpsweepKernel<PtxUpsweepPolicy, Key, SizeT>,
                RadixSortScanKernel<PtxScanPolicy, SizeT>,
                RadixSortDownsweepKernel<PtxDownsweepPolicy, Key, Value, SizeT>,
                upsweep_config,
                scan_config,
                downsweep_config))) break;
        }
        while (0);

        return error;
    }

};


#endif // DOXYGEN_SHOULD_SKIP_THIS



/******************************************************************************
 * DeviceRadixSort
 *****************************************************************************/

/**
 * \brief DeviceRadixSort provides operations for computing a device-wide, parallel radix sort across data items residing within global memory. ![](sorting_logo.png)
 * \ingroup DeviceModule
 *
 * \par Overview
 * The [<em>radix sorting method</em>](http://en.wikipedia.org/wiki/Radix_sort) arranges
 * items into ascending order.  It relies upon a positional representation for
 * keys, i.e., each key is comprised of an ordered sequence of symbols (e.g., digits,
 * characters, etc.) specified from least-significant to most-significant.  For a
 * given input sequence of keys and a set of rules specifying a total ordering
 * of the symbolic alphabet, the radix sorting method produces a lexicographic
 * ordering of those keys.
 *
 * \par
 * DeviceRadixSort can sort all of the built-in C++ numeric primitive types, e.g.:
 * <tt>unsigned char</tt>, \p int, \p double, etc.  Although the direct radix sorting
 * method can only be applied to unsigned integral types, BlockRadixSort
 * is able to sort signed and floating-point types via simple bit-wise transformations
 * that ensure lexicographic key ordering.
 *
 * \par Usage Considerations
 * \cdp_class{DeviceRadixSort}
 *
 * \par Performance
 *
 * \image html lsd_sort_perf.png
 *
 */
struct DeviceRadixSort
{
    /**
     * \brief Sorts key-value pairs.
     *
     * \par
     * The sorting operation requires a pair of key buffers and a pair of value
     * buffers.  Each pair is wrapped in a DoubleBuffer structure whose member
     * DoubleBuffer::Current() references the active buffer.  The currently-active
     * buffer may be changed by the sorting operation.
     *
     * \devicestorage
     *
     * \cdp
     *
     * \par
     * The code snippet below illustrates the sorting of a device vector of \p int keys
     * with associated vector of \p int values.
     * \par
     * \code
     * #include <cub/cub.cuh>
     * ...
     *
     * // Create a set of DoubleBuffers to wrap pairs of device pointers for
     * // sorting data (keys, values, and equivalently-sized alternate buffers)
     * int num_items = ...
     * cub::DoubleBuffer<int> d_keys(d_key_buf, d_key_alt_buf);
     * cub::DoubleBuffer<int> d_values(d_value_buf, d_value_alt_buf);
     *
     * // Determine temporary device storage requirements for sorting operation
     * void *d_temp_storage = NULL;
     * size_t temp_storage_bytes = 0;
     * cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_keys, num_items);
     *
     * // Allocate temporary storage for sorting operation
     * cudaMalloc(&d_temp_storage, temp_storage_bytes);
     *
     * // Run sorting operation
     * cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_keys, num_items);
     *
     * // Sorted keys and values are referenced by d_keys.Current() and d_values.Current()
     *
     * \endcode
     *
     * \tparam Key      <b>[inferred]</b> Key type
     * \tparam Value    <b>[inferred]</b> Value type
     */
    template <
        typename            Key,
        typename            Value>
    __host__ __device__ __forceinline__
    static cudaError_t SortPairs(
        void                *d_temp_storage,                        ///< [in] %Device allocation of temporary storage.  When NULL, the required allocation size is returned in \p temp_storage_bytes and no work is done.
        size_t              &temp_storage_bytes,                    ///< [in,out] Size in bytes of \p d_temp_storage allocation.
        DoubleBuffer<Key>   &d_keys,                                ///< [in,out] Double-buffer of keys whose current buffer contains the unsorted input keys and, upon return, is updated to point to the sorted output keys
        DoubleBuffer<Value> &d_values,                              ///< [in,out] Double-buffer of values whose current buffer contains the unsorted input values and, upon return, is updated to point to the sorted output values
        int                 num_items,                              ///< [in] Number of items to reduce
        int                 begin_bit           = 0,                ///< [in] <b>[optional]</b> The first (least-significant) bit index needed for key comparison
        int                 end_bit             = sizeof(Key) * 8,  ///< [in] <b>[optional]</b> The past-the-end (most-significant) bit index needed for key comparison
        cudaStream_t        stream              = 0,                ///< [in] <b>[optional]</b> CUDA stream to launch kernels within.  Default is stream<sub>0</sub>.
        bool                debug_synchronous  = false)             ///< [in] <b>[optional]</b> Whether or not to synchronize the stream after every kernel launch to check for errors.  Also causes launch configurations to be printed to the console.  Default is \p false.
    {
        typedef int SizeT;
        return DeviceRadixSortDispatch<Key, Value, SizeT>::Dispatch(
            d_temp_storage,
            temp_storage_bytes,
            d_keys,
            d_values,
            num_items,
            begin_bit,
            end_bit,
            stream,
            debug_synchronous);
    }


    /**
     * \brief Sorts keys
     *
     * \par
     * The sorting operation requires a pair of key buffers.  The pair is
     * wrapped in a DoubleBuffer structure whose member DoubleBuffer::Current()
     * references the active buffer.  The currently-active buffer may be changed
     * by the sorting operation.
     *
     * \devicestorage
     *
     * \cdp
     *
     * \par
     * The code snippet below illustrates the sorting of a device vector of \p int keys.
     * \par
     * \code
     * #include <cub/cub.cuh>
     * ...
     *
     * // Create a set of DoubleBuffers to wrap pairs of device pointers for
     * // sorting data (keys and equivalently-sized alternate buffer)
     * int num_items = ...
     * cub::DoubleBuffer<int> d_keys(d_key_buf, d_key_alt_buf);
     *
     * // Determine temporary device storage requirements for sorting operation
     * void *d_temp_storage = NULL;
     * size_t temp_storage_bytes = 0;
     * cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_keys, num_items);
     *
     * // Allocate temporary storage for sorting operation
     * cudaMalloc(&d_temp_storage, temp_storage_bytes);
     *
     * // Run sorting operation
     * cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_keys, num_items);
     *
     * // Sorted keys are referenced by d_keys.Current()
     *
     * \endcode
     *
     * \tparam Key      <b>[inferred]</b> Key type
     */
    template <typename Key>
    __host__ __device__ __forceinline__
    static cudaError_t SortKeys(
        void                *d_temp_storage,                        ///< [in] %Device allocation of temporary storage.  When NULL, the required allocation size is returned in \p temp_storage_bytes and no work is done.
        size_t              &temp_storage_bytes,                    ///< [in,out] Size in bytes of \p d_temp_storage allocation.
        DoubleBuffer<Key>   &d_keys,                                ///< [in,out] Double-buffer of keys whose current buffer contains the unsorted input keys and, upon return, is updated to point to the sorted output keys
        int                 num_items,                              ///< [in] Number of items to reduce
        int                 begin_bit           = 0,                ///< [in] <b>[optional]</b> The first (least-significant) bit index needed for key comparison
        int                 end_bit             = sizeof(Key) * 8,  ///< [in] <b>[optional]</b> The past-the-end (most-significant) bit index needed for key comparison
        cudaStream_t        stream              = 0,                ///< [in] <b>[optional]</b> CUDA stream to launch kernels within.  Default is stream<sub>0</sub>.
        bool                debug_synchronous  = false)             ///< [in] <b>[optional]</b> Whether or not to synchronize the stream after every kernel launch to check for errors.  Also causes launch configurations to be printed to the console.  Default is \p false.
    {
        DoubleBuffer<NullType> d_values;
        return SortPairs(d_temp_storage, temp_storage_bytes, d_keys, d_values, num_items, begin_bit, end_bit, stream, debug_synchronous);
    }

};


}               // CUB namespace
CUB_NS_POSTFIX  // Optional outer namespace(s)


