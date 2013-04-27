
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
 * cub::DeviceHisto256 provides variants of parallel histogram over data residing
 * within a CUDA device's global memory.
 */

#pragma once

#include <stdio.h>
#include <iterator>

#include "tiles/tiles_histo_256.cuh"
#include "../util_allocator.cuh"
#include "../grid/grid_even_share.cuh"
#include "../grid/grid_queue.cuh"

/// Optional outer namespace(s)
CUB_NS_PREFIX

/// CUB namespace
namespace cub {


/******************************************************************************
 * Kernel entry points
 *****************************************************************************/

#ifndef DOXYGEN_SHOULD_SKIP_THIS    // Do not document


/**
 * Multi-block histogram kernel entry point.  Computes privatized histograms, one per thread block.
 */
template <
    typename                TilesHisto256Policy,            ///< Tuning policy for cub::TilesHisto256 abstraction
    int                     CHANNELS,                       ///< Number of channels interleaved in the input data (may be greater than the number of channels being actively histogrammed)
    int                     ACTIVE_CHANNELS,                ///< Number of channels actively being histogrammed
    typename                InputIteratorRA,                ///< The input iterator type (may be a simple pointer type).  Must have a value type that is assignable to <tt>unsigned char</tt>
    typename                HistoCounter,                   ///< Integral type for counting sample occurrences per histogram bin
    typename                SizeT>                          ///< Integral type used for global array indexing
__launch_bounds__ (TilesHisto256Policy::BLOCK_THREADS, 1)
__global__ void MultiBlockHisto256Kernel(
    InputIteratorRA         d_in,                           ///< [in] Array of sample data. (Channels, if any, are interleaved in "AOS" format)
    HistoCounter            *d_out_histograms,              ///< [out] Linearization of a 3D array of histogram counter data having logical type <tt>HistoCounter[ACTIVE_CHANNELS][gridDim.x][256]</tt>
    SizeT                   num_items,                      ///< [in] Total number of samples \p d_in for all channels
    GridEvenShare<SizeT>    even_share,                     ///< [in] Descriptor for how to map an even-share of tiles across thread blocks
    GridQueue<SizeT>        queue)                          ///< [in] Descriptor for performing dynamic mapping of tile data to thread blocks
{
    // Constants
    enum
    {
        BLOCK_THREADS = TilesHisto256Policy::BLOCK_THREADS,
    };

    // Parameterize TilesHisto256 for the parallel execution context
    typedef TilesHisto256 <TilesHisto256Policy, CHANNELS, SizeT> TilesHisto256T;

    // Parameterize which mapping of tiles -> thread blocks we will use
    typedef typename TilesHisto256T::template Mapping<TilesHisto256Policy::GRID_MAPPING> Mapping;

    // Declare shared memory
    __shared__ typename TilesHisto256T::SmemStorage block_histo;                            // Shared memory for TilesHisto256
    __shared__ HistoCounter                         histograms[ACTIVE_CHANNELS][256];       // Shared memory histograms

    // Composite samples into histogram(s)
    Mapping::ProcessTiles(
        block_histo,
        d_in,
        num_items,
        even_share,
        queue,
        histograms);

    // Barrier to ensure histograms are coherent
    __syncthreads();

    // Output histogram for each active channel
    int channel_stride = gridDim.x * 256;

    #pragma unroll
    for (int CHANNEL = 0; CHANNEL < ACTIVE_CHANNELS; ++CHANNEL)
    {
        int channel_offset = (channel_stride * CHANNEL) + (blockIdx.x * 256);
        int histo_offset = 0;

        #pragma unroll
        for(; histo_offset + BLOCK_THREADS <= 256; histo_offset += BLOCK_THREADS)
        {
            d_out_histograms[channel_offset + histo_offset + threadIdx.x] = histograms[CHANNEL][histo_offset + threadIdx.x];
        }
        // Finish up with guarded initialization if necessary
        if ((histo_offset < BLOCK_THREADS) && (histo_offset + threadIdx.x < 256))
        {
            d_out_histograms[channel_offset + histo_offset + threadIdx.x] = histograms[CHANNEL][histo_offset + threadIdx.x];
        }
    }
}


/**
 * Single-block finalization kernel for aggregating privatized threadblock histograms from a previous kernel invocation.
 */
template <
    int             ACTIVE_CHANNELS,            ///< Number of channels actively being histogrammed
    typename        HistoCounter>               ///< Integral type for counting sample occurrences per histogram bin
__launch_bounds__ (256, 1)
__global__ void FinalizeHisto256Kernel(
    HistoCounter    *d_block_histograms,        ///< [in] Linearization of a 3D array of histogram counter data having logical type <tt>HistoCounter[ACTIVE_CHANNELS][num_threadblocks][256]</tt>
    HistoCounter    *d_out_histograms,          ///< [out] Linearization of a 2D array of histogram counter data having logical type <tt>HistoCounter[ACTIVE_CHANNELS][256]</tt>
    int             num_threadblocks)           ///< [in] Number of threadblock histograms per channel in \p d_block_histograms
{
    int channel_stride = num_threadblocks * 256;

    // Output a histogram for each channel
    #pragma unroll
    for (int CHANNEL = 0; CHANNEL < ACTIVE_CHANNELS; ++CHANNEL)
    {
        // Accumulate threadblock-histograms from the channel
        HistoCounter    bin_aggregate = 0;
        int             channel_offset = (channel_stride * CHANNEL);

        for (int block = 0; block < num_threadblocks; ++block)
        {
            int block_offset = block * 256;
            bin_aggregate =+ d_block_histograms[channel_offset + block_offset + threadIdx.x];
        }

        // Output
        d_out_histograms[(CHANNEL * 256) + threadIdx.x] = bin_aggregate;
    }
}

#endif // DOXYGEN_SHOULD_SKIP_THIS


/******************************************************************************
 * DeviceHisto256
 *****************************************************************************/

/**
 * \addtogroup DeviceModule
 * @{
 */

/**
 * \brief DeviceHisto256 provides variants of parallel histogram over data residing within a CUDA device's global memory.
 */
struct DeviceHisto256
{
#ifndef DOXYGEN_SHOULD_SKIP_THIS    // Do not document

    /// Generic structure for encapsulating dispatch properties.  Mirrors the constants within TilesHisto256Policy.
    struct KernelDispachParams
    {
        // Policy fields
        int                     block_threads;
        int                     items_per_thread;
        BlockHisto256Algorithm  block_algorithm;
        GridMappingStrategy     grid_mapping;
        int                     subscription_factor;

        // Derived fields
        int                     tile_size;

        template <typename TilesHisto256Policy>
        __host__ __device__ __forceinline__
        void Init(int subscription_factor = 1)
        {
            block_threads               = TilesHisto256Policy::BLOCK_THREADS;
            items_per_thread            = TilesHisto256Policy::ITEMS_PER_THREAD;
            block_algorithm             = TilesHisto256Policy::BLOCK_ALGORITHM;
            grid_mapping                = TilesHisto256Policy::GRID_MAPPING;
            this->subscription_factor   = subscription_factor;

            tile_size = block_threads * items_per_thread;
        }

        __host__ __device__ __forceinline__
        void Print()
        {
            printf("%d, %d, %d, %d, %d",
                block_threads,
                items_per_thread,
                block_algorithm,
                grid_mapping,
                subscription_factor);
        }

    };


    /// Specializations of tuned policy types for different PTX architectures
    template <
        int                         CHANNELS,
        int                         ACTIVE_CHANNELS,
        BlockHisto256Algorithm      BLOCK_ALGORITHM     = BLOCK_BYTE_HISTO_SORT,
        int                         ARCH                = CUB_PTX_ARCH>
    struct TunedPolicies;

    /// SM35 tune
    template <int CHANNELS, int ACTIVE_CHANNELS, BlockHisto256Algorithm BLOCK_ALGORITHM>
    struct TunedPolicies<CHANNELS, ACTIVE_CHANNELS, BLOCK_ALGORITHM, 350>
    {
        typedef TilesHisto256Policy<128, 16,  BLOCK_ALGORITHM, GRID_MAPPING_EVEN_SHARE> MultiBlockPolicy;
        enum { SUBSCRIPTION_FACTOR = 1 };
    };

    /// SM30 tune
    template <int CHANNELS, int ACTIVE_CHANNELS, BlockHisto256Algorithm BLOCK_ALGORITHM>
    struct TunedPolicies<CHANNELS, ACTIVE_CHANNELS, BLOCK_ALGORITHM, 300>
    {
        typedef TilesHisto256Policy<128, 16,  BLOCK_ALGORITHM, GRID_MAPPING_EVEN_SHARE> MultiBlockPolicy;
        enum { SUBSCRIPTION_FACTOR = 1 };
    };

    /// SM20 tune
    template <int CHANNELS, int ACTIVE_CHANNELS, BlockHisto256Algorithm BLOCK_ALGORITHM>
    struct TunedPolicies<CHANNELS, ACTIVE_CHANNELS, BLOCK_ALGORITHM, 200>
    {
        typedef TilesHisto256Policy<128, 16,  BLOCK_ALGORITHM, GRID_MAPPING_EVEN_SHARE> MultiBlockPolicy;
        enum { SUBSCRIPTION_FACTOR = 1 };
    };

    /// SM10 tune
    template <int CHANNELS, int ACTIVE_CHANNELS, BlockHisto256Algorithm BLOCK_ALGORITHM>
    struct TunedPolicies<CHANNELS, ACTIVE_CHANNELS, BLOCK_ALGORITHM, 100>
    {
        typedef TilesHisto256Policy<128, 16,  BLOCK_ALGORITHM, GRID_MAPPING_EVEN_SHARE> MultiBlockPolicy;
        enum { SUBSCRIPTION_FACTOR = 1 };
    };


    /// Tuning policy(ies) for the PTX architecture that DeviceHisto256 operations will get dispatched to
    template <
        int                         CHANNELS,
        int                         ACTIVE_CHANNELS,
        BlockHisto256Algorithm      BLOCK_ALGORITHM = BLOCK_BYTE_HISTO_SORT>
    struct PtxDefaultPolicies
    {
        static const int PTX_TUNE_ARCH =   (CUB_PTX_ARCH >= 350) ?
                                                350 :
                                                (CUB_PTX_ARCH >= 300) ?
                                                    300 :
                                                    (CUB_PTX_ARCH >= 200) ?
                                                        200 :
                                                        100;

        // Tuned policy set for the current PTX compiler pass
        typedef TunedPolicies<CHANNELS, ACTIVE_CHANNELS, BLOCK_ALGORITHM, PTX_TUNE_ARCH> PtxPassTunedPolicies;

        // Subscription factor for the current PTX compiler pass
        static const int SUBSCRIPTION_FACTOR = PtxPassTunedPolicies::SUBSCRIPTION_FACTOR;

        // MultiBlockPolicy that opaquely derives from the specialization corresponding to the current PTX compiler pass
        struct MultiBlockPolicy : PtxPassTunedPolicies::MultiBlockPolicy {};

        /**
         * Initialize dispatch params with the policies corresponding to the PTX assembly we will use
         */
        static void InitDispatchParams(int ptx_version, KernelDispachParams &multi_block_dispatch_params)
        {
            if (ptx_version >= 350)
            {
                typedef TunedPolicies<CHANNELS, ACTIVE_CHANNELS, BLOCK_ALGORITHM, 350> TunedPolicies;
                multi_block_dispatch_params.Init<TunedPolicies::MultiBlockPolicy>(TunedPolicies::SUBSCRIPTION_FACTOR);
            }
            else if (ptx_version >= 300)
            {
                typedef TunedPolicies<CHANNELS, ACTIVE_CHANNELS, BLOCK_ALGORITHM, 300> TunedPolicies;
                multi_block_dispatch_params.Init<TunedPolicies::MultiBlockPolicy>(TunedPolicies::SUBSCRIPTION_FACTOR);
            }
            else if (ptx_version >= 200)
            {
                typedef TunedPolicies<CHANNELS, ACTIVE_CHANNELS, BLOCK_ALGORITHM, 200> TunedPolicies;
                multi_block_dispatch_params.Init<TunedPolicies::MultiBlockPolicy>(TunedPolicies::SUBSCRIPTION_FACTOR);
            }
            else
            {
                typedef TunedPolicies<CHANNELS, ACTIVE_CHANNELS, BLOCK_ALGORITHM, 100> TunedPolicies;
                multi_block_dispatch_params.Init<TunedPolicies::MultiBlockPolicy>(TunedPolicies::SUBSCRIPTION_FACTOR);
            }
        }
    };


    /**
     * Internal dispatch routine for invoking device-wide, multi-channel, 256-bin histogram
     */
    template <
        int                             CHANNELS,                                           ///< Number of channels interleaved in the input data (may be greater than the number of channels being actively histogrammed)
        int                             ACTIVE_CHANNELS,                                    ///< Number of channels actively being histogrammed
        typename                        MultiBlockHisto256KernelPtr,                        ///< Function type of cub::MultiBlockHisto256Kernel
        typename                        FinalizeHisto256KernelPtr,                          ///< Function type of cub::FinalizeHisto256Kernel
        typename                        PrepareDrainKernelPtr,                              ///< Function type of cub::PrepareDrainKernel
        typename                        InputIteratorRA,                                    ///< The input iterator type (may be a simple pointer type).  Must have a value type that is assignable to <tt>unsigned char</tt>
        typename                        HistoCounter,                                       ///< Integral type for counting sample occurrences per histogram bin
        typename                        SizeT>                                              ///< Integral type used for global array indexing
    __host__ __device__ __forceinline__
    static cudaError_t Dispatch(
        MultiBlockHisto256KernelPtr     multi_block_kernel_ptr,                             ///< [in] Kernel function pointer to parameterization of cub::MultiBlockHisto256Kernel
        FinalizeHisto256KernelPtr       finalize_kernel_ptr,                                ///< [in] Kernel function pointer to parameterization of cub::FinalizeHisto256Kernel
        PrepareDrainKernelPtr           prepare_drain_kernel_ptr,                           ///< [in] Kernel function pointer to parameterization of cub::PrepareDrainKernel
        KernelDispachParams             &multi_block_dispatch_params,                       ///< [in] Dispatch parameters that match the policy that \p multi_block_kernel_ptr was compiled for
        InputIteratorRA                 d_in,                                               ///< [in] Input samples to histogram
        HistoCounter                    *d_histogram,                                       ///< [out] Array of 256 counters of integral type \p HistoCounter.
        SizeT                           num_items,                                          ///< [in] Number of samples to process
        cudaStream_t                    stream              = 0,                            ///< [in] <b>[optional]</b> CUDA stream to launch kernels within.  Default is stream-0.
        bool                            stream_synchronous  = false,                        ///< [in] <b>[optional]</b> Whether or not to synchronize the stream after every kernel launch to check for errors.  Default is \p false.
        DeviceAllocator                 *device_allocator   = DefaultDeviceAllocator())     ///< [in] <b>[optional]</b> Allocator for allocating and freeing device memory.  Default is provided by DefaultDeviceAllocator.
    {
    #if !CUB_CNP_ENABLED

        // Kernel launch not supported from this device
        return CubDebug(cudaErrorInvalidConfiguration);

    #else

        HistoCounter            *d_block_histograms = NULL; // Temporary storage
        GridEvenShare<SizeT>    even_share;                 // Even-share work distribution
        GridQueue<SizeT>        queue;                      // Dynamic, queue-based work distribution

        cudaError error = cudaSuccess;
        do
        {
            // Get GPU ordinal
            int device_ordinal;
            if (CubDebug(error = cudaGetDevice(&device_ordinal))) break;

            // Get SM count
            int sm_count;
            if (CubDebug(error = cudaDeviceGetAttribute (&sm_count, cudaDevAttrMultiProcessorCount, device_ordinal))) break;

            // Rough estimate of SM occupancies based upon the maximum SM occupancy of the targeted PTX architecture
            int multi_sm_occupancy = CUB_MIN(
                ArchProps<CUB_PTX_ARCH>::MAX_SM_THREADBLOCKS,
                ArchProps<CUB_PTX_ARCH>::MAX_SM_THREADS / multi_block_dispatch_params.block_threads);

        #ifndef __CUDA_ARCH__

            // We're on the host, so come up with a more accurate estimate of SM occupancies from actual device properties
            Device device_props;
            if (CubDebug(error = device_props.Init(device_ordinal))) break;

            if (CubDebug(error = device_props.MaxSmOccupancy(
                multi_sm_occupancy,
                multi_block_kernel_ptr,
                multi_block_dispatch_params.block_threads))) break;

        #endif

            // Determine grid size for the multi-block kernel
            int multi_occupancy = multi_sm_occupancy * sm_count;
            int multi_tile_size = multi_block_dispatch_params.block_threads * multi_block_dispatch_params.items_per_thread;
            int multi_grid_size;

            switch (multi_block_dispatch_params.grid_mapping)
            {
            case GRID_MAPPING_EVEN_SHARE:

                // Work is distributed evenly
                even_share.GridInit(
                    num_items,
                    multi_occupancy * multi_block_dispatch_params.subscription_factor,
                    multi_tile_size);

                // Set MultiBlock grid size
                multi_grid_size = even_share.grid_size;
                break;

            case GRID_MAPPING_DYNAMIC:

                // Prepare queue to distribute work dynamically
                queue.Allocate(device_allocator);
                int num_tiles = (num_items + multi_tile_size - 1) / multi_tile_size;

            #ifndef __CUDA_ARCH__

                // We're on the host, so prepare queue on device (because its faster than if we prepare it here)
                if (stream_synchronous) CubLog("Invoking prepare_drain_kernel_ptr<<<1, 1, 0, %d>>>()\n", stream);
                prepare_drain_kernel_ptr<<<1, 1, 0, stream>>>(queue, num_items);

                // Sync the stream on the host
                if (stream_synchronous && CubDebug(error = cudaStreamSynchronize(stream))) break;

            #else

                // Prepare the queue here
                grid_queue.PrepareDrain(num_items);

            #endif

                // Set MultiBlock grid size
                multi_grid_size = (num_tiles < multi_occupancy) ?
                    num_tiles :                 // Not enough to fill the device with threadblocks
                    multi_occupancy;            // Fill the device with threadblocks

            break;
            };

            // Invoke MultiBlockHisto256
            if (stream_synchronous) CubLog("Invoking multi_block_kernel_ptr<<<%d, %d, 0, %d>>>(), %d items per thread, %d SM occupancy\n",
                multi_grid_size, multi_block_dispatch_params.block_threads, stream, multi_block_dispatch_params.items_per_thread, multi_sm_occupancy);

            if (multi_grid_size == 1)
            {
                // A single kernel launch will do
                multi_block_kernel_ptr<<<multi_grid_size, multi_block_dispatch_params.block_threads, 0, stream>>>(
                    d_in,
                    d_histograms,
                    num_items,
                    even_share,
                    queue);
            }
            else
            {
                // Allocate temporary storage for thread block partial reductions
                if (CubDebug(error = DeviceAllocate(
                    (void**) &d_block_histograms,
                    ACTIVE_CHANNELS * multi_grid_size * sizeof(HistoCounter) * 256,
                    device_allocator))) break;

                multi_block_kernel_ptr<<<multi_grid_size, multi_block_dispatch_params.block_threads, 0, stream>>>(
                    d_in,
                    d_block_histograms,
                    num_items,
                    even_share,
                    queue);

                    #ifndef __CUDA_ARCH__
                        // Sync the stream on the host
                        if (stream_synchronous && CubDebug(error = cudaStreamSynchronize(stream))) break;
                    #else
                        // Sync the entire device on the device (cudaStreamSynchronize doesn't exist on device)
                        if (stream_synchronous && CubDebug(error = cudaDeviceSynchronize())) break;
                    #endif

                // Invoke FinalizeHisto256
                if (stream_synchronous) CubLog("Invoking finalize_kernel_ptr<<<%d, %d, 0, %d>>>()\n",
                    1, 256, stream);

                finalize_kernel_ptr<<<1, 256, 0, stream>>>(
                    d_block_histograms,
                    d_histograms,
                    multi_grid_size);
            }

            #ifndef __CUDA_ARCH__
                // Sync the stream on the host
                if (stream_synchronous && CubDebug(error = cudaStreamSynchronize(stream))) break;
            #else
                // Sync the entire device on the device (cudaStreamSynchronize doesn't exist on device)
                if (stream_synchronous && CubDebug(error = cudaDeviceSynchronize())) break;
            #endif
        }
        while (0);

        // Free temporary storage allocation
        if (d_block_histograms) error = CubDebug(DeviceFree(d_block_histograms, device_allocator));

        // Free queue allocation
        if (multi_block_dispatch_params.grid_mapping == GRID_MAPPING_DYNAMIC) error = CubDebug(queue.Free(device_allocator));

        return error;

    #endif
    }

    #endif // DOXYGEN_SHOULD_SKIP_THIS

    //---------------------------------------------------------------------
    // Public interface
    //---------------------------------------------------------------------

    /**
     * \brief Computes a 256-bin device-wide histogram
     *
     * \tparam InputIteratorRA      <b>[inferred]</b> The random-access iterator type for input (may be a simple pointer type).  Must have a value type that is assignable to <tt>unsigned char</tt>
     * \tparam HistoCounter         <b>[inferred]</b> Integral type for counting sample occurrences per histogram bin
     */
    template <
        typename        InputIteratorRA,
        typename        HistoCounter>
    __host__ __device__ __forceinline__
    static cudaError_t Reduce(
        InputIteratorRA   d_in,                                             ///< [in] Input samples to histogram
        HistoCounter    *d_histogram,                                       ///< [out] Array of 256 counters of integral type \p HistoCounter.
        int             num_items,                                          ///< [in] Number of samples to process
        cudaStream_t    stream              = 0,                            ///< [in] <b>[optional]</b> CUDA stream to launch kernels within.  Default is stream-0.
        bool            stream_synchronous  = false,                        ///< [in] <b>[optional]</b> Whether or not to synchronize the stream after every kernel launch to check for errors.  Default is \p false.
        DeviceAllocator *device_allocator   = DefaultDeviceAllocator())     ///< [in] <b>[optional]</b> Allocator for allocating and freeing device memory.  Default is provided by DefaultDeviceAllocator.
    {
        enum
        {
            CHANNELS            = 1,        // Number of channels interleaved in the input data
            ACTIVE_CHANNELS     = 1,        // Number of channels actively being histogrammed
        };

        // Type used for array indexing
        typedef int SizeT;

        // Tuning polices for the PTX architecture that will get dispatched to
        typedef PtxDefaultPolicies<CHANNELS, ACTIVE_CHANNELS> PtxDefaultPolicies;
        typedef typename PtxDefaultPolicies::MultiBlockPolicy MultiBlockPolicy;

        cudaError error = cudaSuccess;
        do
        {
            // Declare dispatch parameters
            KernelDispachParams multi_block_dispatch_params;

        #ifdef __CUDA_ARCH__

            // We're on the device, so initialize the dispatch parameters with the PtxDefaultPolicies directly
            multi_block_dispatch_params.Init<MultiBlockPolicy>(PtxDefaultPolicies::SUBSCRIPTION_FACTOR);

        #else

            // We're on the host, so lookup and initialize the dispatch parameters with the policies that match the device's PTX version
            int ptx_version;
            if (CubDebug(error = PtxVersion(ptx_version))) break;
            PtxDefaultPolicies::InitDispatchParams(ptx_version, multi_block_dispatch_params);

        #endif

            error = Dispatch<CHANNELS, ACTIVE_CHANNELS>(
                MultiBlockHisto256Kernel<MultiBlockPolicy, CHANNELS, ACTIVE_CHANNELS, InputIteratorRA, HistoCounter*, SizeT>,
                FinalizeHisto256Kernel<ACTIVE_CHANNELS, HistoCounter>,
                multi_block_dispatch_params,
                d_in,
                d_histogram,
                num_items,
                stream,
                stream_synchronous,
                device_allocator);

            if (CubDebug(error)) break;
        }
        while (0);

        return error;
    }

};


/** @} */       // DeviceModule

}               // CUB namespace
CUB_NS_POSTFIX  // Optional outer namespace(s)


