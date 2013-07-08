
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

#include "../block/block_radix_sort.cuh"
#include "block/block_radix_sort_histo_tiles.cuh"
#include "block/block_radix_sort_scatter_tiles.cuh"
#include "block/block_scan_tiles.cuh"
#include "../thread/thread_operators.cuh"
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
    typename                BlockRadixSortHistoTilesPolicy, ///< Tuning policy for cub::BlockRadixSortHistoTiles abstraction
    typename                Key,                        ///< Key type
    typename                SizeT>                          ///< Integer type used for global array indexing
__launch_bounds__ (int(BlockRadixSortHistoTilesPolicy::BLOCK_THREADS), 1)
__global__ void RadixSortUpsweepKernel(
    Key                     *d_keys0,                       ///< [in] Input keys ping buffer
    Key                     *d_keys1,                       ///< [in] Input keys pong buffer
    int                     *d_selector,                    ///< [in] Selector indicating which buffer to read keys from
    SizeT                   *d_spine,                       ///< [out] Privatized (per block) digit histograms (striped, i.e., 0s counts from each block, then 1s counts from each block, etc.)
    SizeT                   num_items,                      ///< [in] Total number of input data items
    int                     current_bit,                    ///< [in] Bit position of current radix digit
    bool                    first_pass,                     ///< [in] Whether this is the first digit pass
    GridEvenShare<SizeT>    even_share)                     ///< [in] Descriptor for how to map an even-share of tiles across thread blocks
{
    enum {
        RADIX_DIGITS = 1 << BlockRadixSortHistoTilesPolicy::RADIX_BITS,
    };

    // Select input buffer
    Key *d_keys = (first_pass || *d_selector == 0) ? d_keys0 : d_keys1;

    // Parameterize the BlockRadixSortHistoTiles type for the current configuration
    typedef BlockRadixSortHistoTiles<BlockRadixSortHistoTilesPolicy, Key, SizeT> BlockRadixSortHistoTilesT;

    // Shared memory storage
    __shared__ typename BlockRadixSortHistoTilesT::TempStorage temp_storage;

    // Initialize even-share descriptor for this thread block
    even_share.BlockInit();

    // Process input tiles (each of the first RADIX_DIGITS threads will compute a count for that digit)
    SizeT bin_count;
    BlockRadixSortHistoTilesT(temp_storage, d_keys, current_bit).ProcessTiles(
        even_share.block_offset,
        even_share.block_oob,
        bin_count);

    // Write out digit counts (striped)
    if (threadIdx.x < RADIX_DIGITS)
        d_spine[(gridDim.x * threadIdx.x) + blockIdx.x] = bin_count;
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

    // Process input tiles
    BlockScanTilesT(temp_storage, d_spine, d_spine, cub::Sum(), SizeT(0)).ConsumeTiles(0, num_counts);
}


/**
 * Downsweep pass kernel entry point (multi-block).  Scatters keys (and values) into corresponding bins for the current digit place.
 */
template <
    typename                BlockRadixSortHistoTilesPolicy, ///< Tuning policy for cub::BlockRadixSortHistoTiles abstraction
    typename                Key,                        ///< Key type
    typename                Value,                      ///< Value type
    typename                SizeT>                          ///< Integer type used for global array indexing
__launch_bounds__ (int(BlockRadixSortHistoTilesPolicy::BLOCK_THREADS), 1)
__global__ void RadixSortDownsweepKernel(
    Key                     *d_keys0,                       ///< [in] Input keys ping buffer
    Key                     *d_keys1,                       ///< [in] Input keys pong buffer
    Value                   *d_values0,                     ///< [in] Input values ping buffer
    Value                   *d_values1,                     ///< [in] Input values pong buffer
    int                     *d_selector,                    ///< [in] Selector indicating which buffer to read keys from
    SizeT                   *d_spine,                       ///< [in] Scan of privatized (per block) digit histograms (striped, i.e., 0s counts from each block, then 1s counts from each block, etc.)
    SizeT                   num_items,                      ///< [in] Total number of input data items
    int                     current_bit,                    ///< [in] Bit position of current radix digit
    bool                    first_pass,                     ///< [in] Whether this is the first digit pass
    bool                    last_pass,                      ///< [in] Whether this is the last digit pass
    GridEvenShare<SizeT>    even_share)                     ///< [in] Descriptor for how to map an even-share of tiles across thread blocks
{
    enum {
        RADIX_DIGITS = 1 << BlockRadixSortHistoTilesPolicy::RADIX_BITS,
    };

    Key     *d_keys_in, *d_keys_out;
    Value   *d_values_in, *d_values_out;

    // Select buffers
    if (first_pass || (*d_selector == 0))
    {
        d_keys_in       = d_keys0;
        d_values_in     = d_values0;
        d_keys_out      = d_keys1;
        d_values_out    = d_values1;
    }
    else
    {
        d_keys_in       = d_keys1;
        d_values_in     = d_values1;
        d_keys_out      = d_keys0;
        d_values_out    = d_values0;
    }

    // Parameterize the BlockRadixSortScatterTiles type for the current configuration
    typedef BlockRadixSortScatterTiles<BlockRadixSortScatterTilesPolicy, Key, Value, SizeT> BlockRadixSortScatterTilesT;

    // Shared memory storage
    __shared__ typename BlockRadixSortScatterTilesT::TempStorage temp_storage;

    // Initialize even-share descriptor for this thread block
    even_share.BlockInit();

    // Load digit bin offsets (each of the first RADIX_DIGITS threads will load an offset for that digit)
    SizeT bin_offset;
    if (threadIdx.x < RADIX_DIGITS)
        bin_offset = d_spine[(gridDim.x * threadIdx.x) + blockIdx.x];

    // Process input tiles
    BlockRadixSortScatterTilesT(temp_storage, bin_offset, d_keys_in, d_keys_out, d_values_in, d_values_out, current_bit).ProcessTiles(
        even_share.block_offset,
        even_share.block_oob);
}


#endif // DOXYGEN_SHOULD_SKIP_THIS





/******************************************************************************
 * DeviceRadixSort
 *****************************************************************************/

/**
 * \addtogroup DeviceModule
 * @{
 */

/**
 * \brief DeviceRadixSort provides operations for computing a device-wide, parallel radix sort across data items residing within global memory. ![](sorting_logo.png)
 */
struct DeviceRadixSort
{
    #ifndef DOXYGEN_SHOULD_SKIP_THIS    // Do not document


    /******************************************************************************
     * Constants and typedefs
     ******************************************************************************/

    /// Generic structure for encapsulating dispatch properties codified in block policy.
    struct KernelDispachParams
    {
        int                     block_threads;
        int                     items_per_thread;
        int                     radix_bits;
        int                     subscription_factor;
        int                     tile_size;

        template <typename SortBlockPolicy>
        __host__ __device__ __forceinline__
        void InitSortPolicy(int subscription_factor = 1)
        {
            block_threads               = SortBlockPolicy::BLOCK_THREADS;
            items_per_thread            = SortBlockPolicy::ITEMS_PER_THREAD;
            radix_bits                  = SortBlockPolicy::RADIX_BITS;
            this->subscription_factor   = subscription_factor;
            tile_size                   = block_threads * items_per_thread;
        }

        template <typename ScanBlockPolicy>
        __host__ __device__ __forceinline__
        void InitScanPolicy() :
            subscription_factor(0),
            radix_bits(0)
        {
            block_threads               = ScanBlockPolicy::BLOCK_THREADS;
            items_per_thread            = ScanBlockPolicy::ITEMS_PER_THREAD;
            tile_size                   = block_threads * items_per_thread;
        }

        __host__ __device__ __forceinline__
        void Print()
        {
            printf("%d threads, %d per thread, %d radix bits, %d subscription",
                block_threads,
                items_per_thread,
                radix_bits,
                subscription_factor);
        }

    };



    /******************************************************************************
     * Tuning policies
     ******************************************************************************/

    /// Specializations of tuned policy types for different PTX architectures
    template <typename Key, typename Value, typename SizeT, int ARCH>
    struct TunedPolicies;

    /// SM20 tune
    template <typename Key, typename Value, typename SizeT>
    struct TunedPolicies<Key, Value, SizeT, 200>
    {
        // Upsweep policy
        typedef BlockRadixSortHistoTilesPolicy <128, 17, 5, LOAD_DEFAULT> UpsweepPolicy;

        // Spine scan policy
        typedef BlockScanTilesPolicy <256, 4, BLOCK_LOAD_VECTORIZE, false, LOAD_DEFAULT, BLOCK_STORE_VECTORIZE, false, BLOCK_SCAN_WARP_SCANS> SpinePolicy;

        // Downsweep policy
        typedef BlockRadixSortScatterTilesPolicy <128, 17, BLOCK_LOAD_WARP_TRANSPOSE, LOAD_DEFAULT, false, 5, false, BLOCK_SCAN_WARP_SCANS, RADIX_SORT_SCATTER_TWO_PHASE> DownsweepPolicy;

        enum { SUBSCRIPTION_FACTOR = 4 };
    };



    /******************************************************************************
     * Default policy initializer
     ******************************************************************************/

    /// Tuning policy(ies) for the PTX architecture that DeviceRadixSort operations will get dispatched to
    template <typename Key, typename Value, typename SizeT>
    struct PtxDefaultPolicies
    {
/*
        static const int PTX_TUNE_ARCH =   (CUB_PTX_ARCH >= 350) ?
                                                350 :
                                                (CUB_PTX_ARCH >= 300) ?
                                                    300 :
                                                    (CUB_PTX_ARCH >= 200) ?
                                                        200 :
                                                        (CUB_PTX_ARCH >= 130) ?
                                                            130 :
                                                            100;
*/
        static const int PTX_TUNE_ARCH = 200;

        // Tuned policy set for the current PTX compiler pass
        typedef TunedPolicies<Key, Value, SizeT, PTX_TUNE_ARCH> PtxTunedPolicies;

        // UpsweepPolicy that opaquely derives from the specialization corresponding to the current PTX compiler pass
        struct UpsweepPolicy : PtxTunedPolicies::UpsweepPolicy {};

        // SpinePolicy that opaquely derives from the specialization corresponding to the current PTX compiler pass
        struct SpinePolicy : PtxTunedPolicies::SpinePolicy {};

        // DownsweepPolicy that opaquely derives from the specialization corresponding to the current PTX compiler pass
        struct DownsweepPolicy : PtxTunedPolicies::DownsweepPolicy {};

        // Subscription factor for the current PTX compiler pass
        enum { SUBSCRIPTION_FACTOR = PtxTunedPolicies::SUBSCRIPTION_FACTOR };


        /**
         * Initialize dispatch params with the policies corresponding to the PTX assembly we will use
         */
        static void InitDispatchParams(
            int                    ptx_version,
            KernelDispachParams    &upsweep_dispatch_params,
            KernelDispachParams    &scan_dispatch_params,
            KernelDispachParams    &downsweep_dispatch_params)
        {
/*
            if (ptx_version >= 350)
            {
            }
            else if (ptx_version >= 300)
            {
            }
            else if (ptx_version >= 200)
            {
*/
                typedef TunedPolicies<Key, Value, SizeT, 200> TunedPolicies;
                upsweep_dispatch_params.InitSortPolicy<typename TunedPolicies::UpsweepPolicy>(TunedPolicies::SUBSCRIPTION_FACTOR);
                scan_dispatch_params.InitScanPolicy<typename TunedPolicies::SpinePolicy>();
                downsweep_dispatch_params.InitSortPolicy<typename TunedPolicies::DownsweepPolicy>(TunedPolicies::SUBSCRIPTION_FACTOR);
/*
            }
            else if (ptx_version >= 130)
            {
            }
            else
            {
            }
*/
        }
    };



    /******************************************************************************
     * Utility methods
     ******************************************************************************/

    /**
     * Internal dispatch routine for computing a device-wide reduction using a two-stages of kernel invocations.
     */
    template <
        typename            UpsweepKernelPtr,                       ///< Function type of cub::RadixSortUpsweepKernel
        typename            SpineKernelPtr,                         ///< Function type of cub::SpineScanKernel
        typename            DownsweepKernelPtr,                     ///< Function type of cub::RadixSortUpsweepKernel
        typename            Key,                                    ///< Key type
        typename            Value,                                  ///< Value type
        typename            SizeT>                                  ///< Integer type used for global array indexing
    __host__ __device__ __forceinline__
    static cudaError_t Dispatch(
        void                *d_temp_storage,                        ///< [in] %Device allocation of temporary storage.  When NULL, the required allocation size is returned in \p temp_storage_bytes and no work is done.
        size_t              &temp_storage_bytes,                    ///< [in,out] Size in bytes of \p d_temp_storage allocation.
        UpsweepKernelPtr    upsweep_kernel,                         ///< [in] Kernel function pointer to parameterization of cub::RadixSortUpsweepKernel
        SpineKernelPtr      scan_kernel,                            ///< [in] Kernel function pointer to parameterization of cub::SpineScanKernel
        DownsweepKernelPtr  downsweep_kernel,                       ///< [in] Kernel function pointer to parameterization of cub::RadixSortUpsweepKernel
        KernelDispachParams &upsweep_dispatch_params,               ///< [in] Dispatch parameters that match the policy that \p upsweep_kernel was compiled for
        KernelDispachParams &scan_dispatch_params,                  ///< [in] Dispatch parameters that match the policy that \p scan_kernel was compiled for
        KernelDispachParams &downsweep_dispatch_params,             ///< [in] Dispatch parameters that match the policy that \p downsweep_kernel was compiled for
        DoubleBuffer<Key>   &d_keys,                                ///< [in,out] Double-buffer whose current buffer contains the unsorted input keys and, upon return, is updated to point to the sorted output keys
        DoubleBuffer<Value> &d_values,                              ///< [in,out] Double-buffer whose current buffer contains the unsorted input values and, upon return, is updated to point to the sorted output values
        SizeT               num_items,                              ///< [in] Number of items to reduce
        int                 begin_bit           = 0,                ///< [in] <b>[optional]</b> The beginning (least-significant) bit index needed for key comparison
        int                 end_bit             = sizeof(Key) * 8,  ///< [in] <b>[optional]</b> The past-the-end (most-significant) bit index needed for key comparison
        cudaStream_t        stream              = 0,                ///< [in] <b>[optional]</b> CUDA stream to launch kernels within.  Default is stream<sub>0</sub>.
        bool                stream_synchronous  = false)            ///< [in] <b>[optional]</b> Whether or not to synchronize the stream after every kernel launch to check for errors.  Default is \p false.
    {
#ifndef CUB_RUNTIME_ENABLED

        // Kernel launch not supported from this device
        return CubDebug(cudaErrorNotSupported );

#else

        cudaError error = cudaSuccess;
        do
        {
            // Get device ordinal
            int device_ordinal;
            if (CubDebug(error = cudaGetDevice(&device_ordinal))) break;

            // Get SM count
            int sm_count;
            if (CubDebug(error = cudaDeviceGetAttribute (&sm_count, cudaDevAttrMultiProcessorCount, device_ordinal))) break;

            // Get a rough estimate of downsweep_kernel SM occupancy based upon the maximum SM occupancy of the targeted PTX architecture
            int downsweep_sm_occupancy = CUB_MIN(
                ArchProps<CUB_PTX_ARCH>::MAX_SM_THREADBLOCKS,
                ArchProps<CUB_PTX_ARCH>::MAX_SM_THREADS / downsweep_dispatch_params.block_threads);
            int upsweep_sm_occupancy = downsweep_sm_occupancy;

#ifndef __CUDA_ARCH__
            // We're on the host, so come up with more accurate estimates of SM occupancy from actual device properties
            Device device_props;
            if (CubDebug(error = device_props.Init(device_ordinal))) break;

            if (CubDebug(error = device_props.MaxSmOccupancy(
                downsweep_sm_occupancy,
                downsweep_kernel,
                downsweep_dispatch_params.block_threads))) break;

            if (CubDebug(error = device_props.MaxSmOccupancy(
                upsweep_sm_occupancy,
                upsweep_kernel,
                upsweep_dispatch_params.block_threads))) break;
#endif
            // Get device occupancies
            int downsweep_occupancy = downsweep_sm_occupancy * sm_count;
            int upsweep_occupancy = upsweep_sm_occupancy * sm_count;

            // Get even-share work distribution descriptor
            GridEvenShare<SizeT> even_share;
            int max_downsweep_grid_size = downsweep_occupancy * downsweep_dispatch_params.subscription_factor;
            int downsweep_grid_size;
            even_share.GridInit(num_items, max_downsweep_grid_size, downsweep_dispatch_params.tile_size);
            downsweep_grid_size = even_share.grid_size;

            // Temporary storage allocation requirements
            void* allocations[2];
            size_t allocation_sizes[2] =
            {
                downsweep_grid_size * sizeof(SizeT),    // bytes needed for privatized block digit histograms
                sizeof(int)                             // bytes needed double-buffer selector
            };

            // Alias temporaries (or set the necessary size of the storage allocation)
            if (CubDebug(error = AliasTemporaries(d_temp_storage, temp_storage_bytes, allocations, allocation_sizes))) break;

            // Return if the caller is simply requesting the size of the storage allocation
            if (d_temp_storage == NULL)
                return cudaSuccess;

            // Privatized per-block digit histograms
            SizeT *d_spine = (SizeT*) allocations[0];

            // Double-buffer selector
            int *d_selector = allocations[1];

            // Iterate over digit places
            for (int current_bit = begin_bit;
                current_bit < end_bit;
                current_bit += downsweep_dispatch_params.radix_bits)
            {
                // Log upsweep_kernel configuration
                if (stream_synchronous) CubLog("Invoking upsweep_kernel<<<%d, %d, 0, %lld>>>(), %d items per thread, %d SM occupancy\n",
                    downsweep_grid_size, upsweep_dispatch_params.block_threads, (long long) stream, upsweep_dispatch_params.items_per_thread, upsweep_sm_occupancy);

                // Invoke upsweep_kernel with same grid size as downsweep_kernel
                upsweep_kernel<<<downsweep_grid_size, upsweep_dispatch_params.block_threads, 0, stream>>>(
                    d_keys.d_buffers[d_keys.selector],
                    d_keys.d_buffers[d_keys.selector ^ 1],
                    d_selector,
                    d_spine,
                    num_items,
                    current_bit,
                    (current_bit == begin_bit),
                    even_share);

                // Sync the stream if specified
                if (stream_synchronous && (CubDebug(error = SyncStream(stream)))) break;

                // Log scan_kernel configuration
                if (stream_synchronous) CubLog("Invoking scan_kernel<<<%d, %d, 0, %lld>>>(), %d items per thread\n",
                    1, scan_dispatch_params.block_threads, (long long) stream, scan_dispatch_params.items_per_thread);

                // Invoke scan_kernel
                scan_kernel<<<1, scan_dispatch_params.block_threads, 0, stream>>>(
                    d_spine,
                    downsweep_grid_size,
                    reduction_op);

                // Sync the stream if specified
                if (stream_synchronous && (CubDebug(error = SyncStream(stream)))) break;

                // Log downsweep_kernel configuration
                if (stream_synchronous) CubLog("Invoking downsweep_kernel<<<%d, %d, 0, %lld>>>(), %d items per thread, %d SM occupancy\n",
                    downsweep_grid_size, downsweep_dispatch_params.block_threads, (long long) stream, downsweep_dispatch_params.items_per_thread, downsweep_sm_occupancy);

                // Invoke downsweep_kernel
                downsweep_kernel<<<downsweep_grid_size, downsweep_dispatch_params.block_threads, 0, stream>>>(
                    d_keys.d_buffers[d_keys.selector],
                    d_keys.d_buffers[d_keys.selector ^ 1],
                    d_values.d_buffers[d_values.selector],
                    d_values.d_buffers[d_values.selector ^ 1],
                    d_selector,
                    d_spine,
                    num_items,
                    current_bit,
                    (current_bit == begin_bit),
                    (current_bit + downsweep_dispatch_params.radix_bits >= end_bit),
                    even_share);

                // Sync the stream if specified
                if (stream_synchronous && (CubDebug(error = SyncStream(stream)))) break;
            }
        }
        while (0);

        return error;

#endif // CUB_RUNTIME_ENABLED
    }



    #endif // DOXYGEN_SHOULD_SKIP_THIS

    /******************************************************************************
     * Interface
     ******************************************************************************/


    /**
     * \brief Sorts key-value pairs
     *
     * \devicestorage
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
        int                 begin_bit           = 0,                ///< [in] <b>[optional]</b> The beginning (least-significant) bit index needed for key comparison
        int                 end_bit             = sizeof(Key) * 8,  ///< [in] <b>[optional]</b> The past-the-end (most-significant) bit index needed for key comparison
        cudaStream_t        stream              = 0,                ///< [in] <b>[optional]</b> CUDA stream to launch kernels within.  Default is stream<sub>0</sub>.
        bool                stream_synchronous  = false)            ///< [in] <b>[optional]</b> Whether or not to synchronize the stream after every kernel launch to check for errors.  Default is \p false.
    {
        // Type used for array indexing
        typedef int SizeT;

        // Tuning polices
        typedef PtxDefaultPolicies<Key, Value, SizeT>           PtxDefaultPolicies; // Wrapper of default kernel policies
        typedef typename PtxDefaultPolicies::UpsweepPolicy      UpsweepPolicy;      // Upsweep kernel policy
        typedef typename PtxDefaultPolicies::ScanPolicy         ScanPolicy;         // Upsweep kernel policy
        typedef typename PtxDefaultPolicies::DownsweepPolicy    DownsweepPolicy;    // Downsweep kernel policy

        cudaError error = cudaSuccess;
        do
        {
            // Declare dispatch parameters
            KernelDispachParams upsweep_dispatch_params;
            KernelDispachParams scan_dispatch_params;
            KernelDispachParams downsweep_dispatch_params;

#ifdef __CUDA_ARCH__
            // We're on the device, so initialize the dispatch parameters with the PtxDefaultPolicies directly
            upsweep_dispatch_params.Init<UpsweepPolicy>(PtxDefaultPolicies::SUBSCRIPTION_FACTOR);
            scan_dispatch_params.Init<ScanPolicy>();
            downsweep_dispatch_params.Init<DownsweepPolicy>(PtxDefaultPolicies::SUBSCRIPTION_FACTOR);
#else
            // We're on the host, so lookup and initialize the dispatch parameters with the policies that match the device's PTX version
            int ptx_version;
            if (CubDebug(error = PtxVersion(ptx_version))) break;
            PtxDefaultPolicies::InitDispatchParams(
                ptx_version,
                upsweep_dispatch_params,
                scan_dispatch_params,
                downsweep_dispatch_params);
#endif
            // Dispatch
            if (CubDebug(error = Dispatch(
                d_temp_storage,
                temp_storage_bytes,
                RadixSortUpsweepKernel<UpsweepPolicy, Key, SizeT>,
                RadixSortScanKernel<ScanPolicy, SizeT>,
                RadixSortDownsweepKernel<DownsweepPolicy, Key, Value, SizeT>,
                upsweep_dispatch_params,
                scan_dispatch_params,
                downsweep_dispatch_params,
                d_keys,
                d_values,
                num_items,
                begin_bit,
                end_bit,
                stream,
                stream_synchronous))) break;
        }
        while (0);

        return error;
    }


};


/** @} */       // DeviceModule

}               // CUB namespace
CUB_NS_POSTFIX  // Optional outer namespace(s)


