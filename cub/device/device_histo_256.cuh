
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
 * cub::DeviceHisto256 provides variants of device-wide parallel histogram over data residing within global memory.
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
    typename                TilesHisto256Policy,                ///< Tuning policy for cub::TilesHisto256 abstraction
    int                     CHANNELS,                           ///< Number of channels interleaved in the input data (may be greater than the number of channels being actively histogrammed)
    int                     ACTIVE_CHANNELS,                    ///< Number of channels actively being histogrammed
    typename                InputIteratorRA,                    ///< The input iterator type (may be a simple pointer type).  Must have a value type that is assignable to <tt>unsigned char</tt>
    typename                HistoCounter,                       ///< Integral type for counting sample occurrences per histogram bin
    typename                SizeT>                              ///< Integral type used for global array indexing
__launch_bounds__ (TilesHisto256Policy::BLOCK_THREADS)
__global__ void MultiBlockHisto256Kernel(
    InputIteratorRA                                 d_samples,          ///< [in] Array of sample data. (Channels, if any, are interleaved in "AOS" format)
    ArrayWrapper<HistoCounter*, ACTIVE_CHANNELS>    d_out_histograms,   ///< [out] Histogram counter data having logical dimensions <tt>HistoCounter[ACTIVE_CHANNELS][gridDim.x][256]</tt>
    SizeT                                           num_samples,        ///< [in] Total number of samples \p d_samples for all channels
    GridEvenShare<SizeT>                            even_share,         ///< [in] Descriptor for how to map an even-share of tiles across thread blocks
    GridQueue<SizeT>                                queue)              ///< [in] Descriptor for performing dynamic mapping of tile data to thread blocks
{
    // Constants
    enum {
        BLOCK_THREADS       = TilesHisto256Policy::BLOCK_THREADS,
        ITEMS_PER_THREAD    = TilesHisto256Policy::ITEMS_PER_THREAD,
        TILE_SIZE           = BLOCK_THREADS * ITEMS_PER_THREAD,
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
        d_samples,
        num_samples,
        even_share,
        queue,
        histograms);

    // Barrier to ensure histograms are coherent
    __syncthreads();

    // Output histogram for each active channel

    #pragma unroll
    for (int CHANNEL = 0; CHANNEL < ACTIVE_CHANNELS; ++CHANNEL)
    {
        int channel_offset  = (blockIdx.x * 256);
        int histo_offset    = 0;

        #pragma unroll
        for(; histo_offset + BLOCK_THREADS <= 256; histo_offset += BLOCK_THREADS)
        {
            d_out_histograms.array[CHANNEL][channel_offset + histo_offset + threadIdx.x] = histograms[CHANNEL][histo_offset + threadIdx.x];
        }
        // Finish up with guarded initialization if necessary
        if ((histo_offset < BLOCK_THREADS) && (histo_offset + threadIdx.x < 256))
        {
            d_out_histograms.array[CHANNEL][channel_offset + histo_offset + threadIdx.x] = histograms[CHANNEL][histo_offset + threadIdx.x];
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
    HistoCounter*                                   d_block_histograms_linear,  ///< [in] Histogram counter data having logical dimensions <tt>HistoCounter[ACTIVE_CHANNELS][num_threadblocks][256]</tt>
    ArrayWrapper<HistoCounter*, ACTIVE_CHANNELS>    d_out_histograms,           ///< [out] Histogram counter data having logical dimensions <tt>HistoCounter[ACTIVE_CHANNELS][256]</tt>
    int                                             num_threadblocks)           ///< [in] Number of threadblock histograms per channel in \p d_block_histograms
{
    // Accumulate threadblock-histograms from the channel
    HistoCounter bin_aggregate = 0;

    HistoCounter *d_block_channel_histograms = d_block_histograms_linear + (blockIdx.x * num_threadblocks * 256);

    #pragma unroll 32
    for (int block = 0; block < num_threadblocks; ++block)
    {
        bin_aggregate += d_block_channel_histograms[(block * 256) + threadIdx.x];
    }

    // Output
    d_out_histograms.array[blockIdx.x][threadIdx.x] = bin_aggregate;
}


/******************************************************************************
 * Texture references
 *****************************************************************************/

// Anonymous namespace
namespace {

/// Templated Texture reference type for multiplicand vector
template <typename T>
struct TexDeviceHisto256
{
    // Texture reference type
    typedef texture<T, cudaTextureType1D, cudaReadModeElementType> TexRef;

    static TexRef ref;

    /**
     * Bind texture
     */
    static cudaError_t BindTexture(void *d_in, size_t &offset)
    {
        cudaChannelFormatDesc tex_desc = cudaCreateChannelDesc<T>();
        if (d_in)
        {
            return (CubDebug(cudaBindTexture(&offset, ref, d_in, tex_desc)));
        }
        return cudaSuccess;
    }

    /**
     * Unbind textures
     */
    static cudaError_t UnbindTexture()
    {
        return CubDebug(cudaUnbindTexture(ref));
    }
};

// Texture reference definitions
template <typename Value>
typename TexDeviceHisto256<Value>::TexRef TexDeviceHisto256<Value>::ref = 0;


} // Anonymous namespace

#endif // DOXYGEN_SHOULD_SKIP_THIS



/******************************************************************************
 * DeviceHisto256
 *****************************************************************************/

/**
 * \addtogroup DeviceModule
 * @{
 */

/**
 * \brief DeviceHisto256 provides variants of device-wide parallel histogram over data residing within global memory.
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
        BlockHisto256Algorithm      BLOCK_ALGORITHM,
        int                         ARCH>
    struct TunedPolicies;

    /// SM35 tune
    template <int CHANNELS, int ACTIVE_CHANNELS, BlockHisto256Algorithm BLOCK_ALGORITHM>
    struct TunedPolicies<CHANNELS, ACTIVE_CHANNELS, BLOCK_ALGORITHM, 350>
    {
        typedef TilesHisto256Policy<128, 16,  BLOCK_ALGORITHM, GRID_MAPPING_EVEN_SHARE> MultiBlockPolicy;
        enum { SUBSCRIPTION_FACTOR = 4 };
    };

    /// SM30 tune
    template <int CHANNELS, int ACTIVE_CHANNELS, BlockHisto256Algorithm BLOCK_ALGORITHM>
    struct TunedPolicies<CHANNELS, ACTIVE_CHANNELS, BLOCK_ALGORITHM, 300>
    {
        typedef TilesHisto256Policy<
            128,
            (BLOCK_ALGORITHM == BLOCK_BYTE_HISTO_SORT) ? 17 : 16,
            BLOCK_ALGORITHM,
            (BLOCK_ALGORITHM == BLOCK_BYTE_HISTO_SORT) ? GRID_MAPPING_DYNAMIC : GRID_MAPPING_EVEN_SHARE> MultiBlockPolicy;
        enum { SUBSCRIPTION_FACTOR = 4 };
    };

    /// SM20 tune
    template <int CHANNELS, int ACTIVE_CHANNELS, BlockHisto256Algorithm BLOCK_ALGORITHM>
    struct TunedPolicies<CHANNELS, ACTIVE_CHANNELS, BLOCK_ALGORITHM, 200>
    {
        typedef TilesHisto256Policy<
            128,
            (BLOCK_ALGORITHM == BLOCK_BYTE_HISTO_SORT) ? 17 : 16,
            BLOCK_ALGORITHM,
            (BLOCK_ALGORITHM == BLOCK_BYTE_HISTO_SORT) ? GRID_MAPPING_DYNAMIC : GRID_MAPPING_EVEN_SHARE> MultiBlockPolicy;
        enum { SUBSCRIPTION_FACTOR = 2 };
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
        BlockHisto256Algorithm      BLOCK_ALGORITHM>
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
        InputIteratorRA                 d_samples,                                          ///< [in] Input samples to histogram
        HistoCounter                    *(&d_histograms)[ACTIVE_CHANNELS],                  ///< [out] Array of channel histograms, each having 256 counters of integral type \p HistoCounter.
        SizeT                           num_samples,                                        ///< [in] Number of samples to process
        cudaStream_t                    stream              = 0,                            ///< [in] <b>[optional]</b> CUDA stream to launch kernels within.  Default is stream-0.
        bool                            stream_synchronous  = false,                        ///< [in] <b>[optional]</b> Whether or not to synchronize the stream after every kernel launch to check for errors.  Default is \p false.
        DeviceAllocator                 *device_allocator   = DefaultDeviceAllocator())     ///< [in] <b>[optional]</b> Allocator for allocating and freeing device memory.  Default is provided by DefaultDeviceAllocator.
    {
    #if !CUB_CNP_ENABLED

        // Kernel launch not supported from this device
        return CubDebug(cudaErrorInvalidConfiguration);

    #else

        HistoCounter            *d_block_histograms_linear = NULL;  // Temporary storage
        GridEvenShare<SizeT>    even_share;                         // Even-share work distribution
        GridQueue<SizeT>        queue;                              // Dynamic, queue-based work distribution

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
                    num_samples,
                    multi_occupancy * multi_block_dispatch_params.subscription_factor,
                    multi_tile_size);

                // Set MultiBlock grid size
                multi_grid_size = even_share.grid_size;
                break;

            case GRID_MAPPING_DYNAMIC:

                // Prepare queue to distribute work dynamically
                queue.Allocate(device_allocator);
                int num_tiles = (num_samples + multi_tile_size - 1) / multi_tile_size;

            #ifndef __CUDA_ARCH__

                // We're on the host, so prepare queue on device (because its faster than if we prepare it here)
                if (stream_synchronous) CubLog("Invoking prepare_drain_kernel_ptr<<<1, 1, 0, %d>>>()\n", stream);
                prepare_drain_kernel_ptr<<<1, 1, 0, stream>>>(queue, num_samples);

                // Sync the stream on the host
                if (stream_synchronous && CubDebug(error = cudaStreamSynchronize(stream))) break;

            #else

                // Prepare the queue here
                grid_queue.PrepareDrain(num_samples);

            #endif

                // Set MultiBlock grid size
                multi_grid_size = (num_tiles < multi_occupancy) ?
                    num_tiles :                 // Not enough to fill the device with threadblocks
                    multi_occupancy;            // Fill the device with threadblocks

            break;
            };

            // Setup array wrapper for histogram channel output because we can't pass static arrays as kernel parameters
            ArrayWrapper<HistoCounter*, ACTIVE_CHANNELS> d_histo_wrapper;
            for (int CHANNEL = 0; CHANNEL < ACTIVE_CHANNELS; ++CHANNEL)
            {
                d_histo_wrapper.array[CHANNEL] = d_histograms[CHANNEL];
            }

            // Bind texture
            if (CubDebug(error = d_samples.BindTexture())) break;

            // Invoke MultiBlockHisto256
            if (stream_synchronous) CubLog("Invoking multi_block_kernel_ptr<<<%d, %d, 0, %d>>>(), %d items per thread, %d SM occupancy\n",
                multi_grid_size, multi_block_dispatch_params.block_threads, stream, multi_block_dispatch_params.items_per_thread, multi_sm_occupancy);

            if (multi_grid_size == 1)
            {
                // A single pass will do
                multi_block_kernel_ptr<<<multi_grid_size, multi_block_dispatch_params.block_threads, 0, stream>>>(
                    d_samples,
                    d_histo_wrapper,
                    num_samples,
                    even_share,
                    queue);
            }
            else
            {
                // Use two-pass approach to compute and reduce privatized block histograms

                // Allocate temporary storage for privatized thread block histograms in each channel
                if (CubDebug(error = DeviceAllocate(
                    (void**) &d_block_histograms_linear,
                    ACTIVE_CHANNELS * multi_grid_size * sizeof(HistoCounter) * 256,
                    device_allocator))) break;

                // Setup array wrapper for temporary histogram channel output because we can't pass static arrays as kernel parameters
                ArrayWrapper<HistoCounter*, ACTIVE_CHANNELS> d_temp_histo_wrapper;
                for (int CHANNEL = 0; CHANNEL < ACTIVE_CHANNELS; ++CHANNEL)
                {
                    d_temp_histo_wrapper.array[CHANNEL] = d_block_histograms_linear + (CHANNEL * multi_grid_size * 256);
                }

                multi_block_kernel_ptr<<<multi_grid_size, multi_block_dispatch_params.block_threads, 0, stream>>>(
                    d_samples,
                    d_temp_histo_wrapper,
                    num_samples,
                    even_share,
                    queue);

                #ifndef __CUDA_ARCH__
                    // Sync the stream on the host
                    if (stream_synchronous && CubDebug(error = cudaStreamSynchronize(stream))) break;
                #else
                    // Sync the entire device on the device (cudaStreamSynchronize doesn't exist on device)
                    if (stream_synchronous && CubDebug(error = cudaDeviceSynchronize())) break;
                #endif

                if (stream_synchronous) CubLog("Invoking finalize_kernel_ptr<<<%d, %d, 0, %d>>>()\n",
                    ACTIVE_CHANNELS, 256, stream);

                finalize_kernel_ptr<<<ACTIVE_CHANNELS, 256, 0, stream>>>(
                    d_block_histograms_linear,
                    d_histo_wrapper,
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
        if (d_block_histograms_linear) error = CubDebug(DeviceFree(d_block_histograms_linear, device_allocator));

        // Free queue allocation
        if (multi_block_dispatch_params.grid_mapping == GRID_MAPPING_DYNAMIC) error = CubDebug(queue.Free(device_allocator));

        // Unbind texture
        error = CubDebug(d_samples.UnbindTexture());

        return error;

    #endif
    }


    /**
     * \brief Computes a 256-bin device-wide histogram
     *
     * \tparam BLOCK_ALGORITHM      cub::BlockHisto256Algorithm enumerator specifying the underlying algorithm to use
     * \tparam CHANNELS             Number of channels interleaved in the input data (may be greater than the number of channels being actively histogrammed)
     * \tparam ACTIVE_CHANNELS      <b>[inferred]</b> Number of channels actively being histogrammed
     * \tparam InputIteratorRA      <b>[inferred]</b> The random-access iterator type for input (may be a simple pointer type).  Must have a value type that is assignable to <tt>unsigned char</tt>
     * \tparam HistoCounter         <b>[inferred]</b> Integral type for counting sample occurrences per histogram bin
     */
    template <
        BlockHisto256Algorithm  BLOCK_ALGORITHM,
        int                     CHANNELS,                                           ///< Number of channels interleaved in the input data (may be greater than the number of channels being actively histogrammed)
        int                     ACTIVE_CHANNELS,                                    ///< Number of channels actively being histogrammed
        typename                InputIteratorRA,
        typename                HistoCounter>
    __host__ __device__ __forceinline__
    static cudaError_t Dispatch(
        InputIteratorRA         d_samples,                                          ///< [in] Input samples to histogram
        HistoCounter            *(&d_histograms)[ACTIVE_CHANNELS],                  ///< [out] Array of channel histograms, each having 256 counters of integral type \p HistoCounter.
        int                     num_samples,                                        ///< [in] Number of samples to process
        cudaStream_t            stream              = 0,                            ///< [in] <b>[optional]</b> CUDA stream to launch kernels within.  Default is stream-0.
        bool                    stream_synchronous  = false,                        ///< [in] <b>[optional]</b> Whether or not to synchronize the stream after every kernel launch to check for errors.  Default is \p false.
        DeviceAllocator*        device_allocator    = DefaultDeviceAllocator())     ///< [in] <b>[optional]</b> Allocator for allocating and freeing device memory.  Default is provided by DefaultDeviceAllocator.
    {
        // Type used for array indexing
        typedef int SizeT;

        // Tuning polices for the PTX architecture that will get dispatched to
        typedef PtxDefaultPolicies<CHANNELS, ACTIVE_CHANNELS, BLOCK_ALGORITHM> PtxDefaultPolicies;
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

            Dispatch<CHANNELS, ACTIVE_CHANNELS>(
                MultiBlockHisto256Kernel<MultiBlockPolicy, CHANNELS, ACTIVE_CHANNELS, InputIteratorRA, HistoCounter, SizeT>,
                FinalizeHisto256Kernel<ACTIVE_CHANNELS, HistoCounter>,
                PrepareDrainKernel<SizeT>,
                multi_block_dispatch_params,
                d_samples,
                d_histograms,
                num_samples,
                stream,
                stream_synchronous,
                device_allocator);

            if (CubDebug(error)) break;
        }
        while (0);

        return error;
    }



    #endif // DOXYGEN_SHOULD_SKIP_THIS

    //---------------------------------------------------------------------
    // Public interface
    //---------------------------------------------------------------------

    /**
     * \brief A simple iterator wrapper for converting non-8b sample data into 8b bins.  Loads through texture when possible.
     *
     * \tparam T            The type of sample data to wrap.
     * \tparam BinOp        Unary functor type for mapping objects of type /p T into 8b bins.  Must have member <tt>unsigned char operator()(const T &datum)</tt>.
     */
    template <typename T, typename BinOp>
    class BinningIteratorRA
    {
    public:
        typedef BinningIteratorRA                   self_type;
        typedef unsigned char                       value_type;
        typedef unsigned char                       reference;
        typedef unsigned char*                      pointer;
        typedef std::random_access_iterator_tag     iterator_category;
        typedef int                                 difference_type;

        __host__ __device__ __forceinline__ BinningIteratorRA(T* ptr, BinOp bin_op) :
            bin_op(bin_op),
            ptr(ptr) /* , offset(0) */ {}

        __host__ __forceinline__ cudaError_t BindTexture()
        {
//            return TexDeviceHisto256<T>::BindTexture(ptr, offset);
            return cudaSuccess;
        }

        __host__ __forceinline__ cudaError_t UnbindTexture()
        {
            return cudaSuccess;
//            return TexDeviceHisto256<T>::UnbindTexture();
        }

        __host__ __device__ __forceinline__ self_type operator++()
        {
            self_type i = *this;
            ptr++;
//            offset++;
            return i;
        }

        __host__ __device__ __forceinline__ self_type operator++(int junk)
        {
            ptr++;
//            offset++;
            return *this;
        }

        __host__ __device__ __forceinline__ reference operator*()
        {
//#ifndef __CUDA_ARCH__
            return bin_op(*ptr);
//#else
//            return bin_op(tex1Dfetch(TexDeviceHisto256<T>::ref, offset));
//#endif
        }

        template <typename SizeT> __host__ __device__ __forceinline__ reference operator[](SizeT n)
        {
//#ifndef __CUDA_ARCH__
            return bin_op(*(ptr + n));
//#else
//            return bin_op(tex1Dfetch(TexDeviceHisto256<T>::ref, offset + n));
//#endif
        }

        __host__ __device__ __forceinline__ pointer operator->()
        {
//#ifndef __CUDA_ARCH__
            return &(bin_op(*ptr));
//#else
//            return &(bin_op(tex1Dfetch(TexDeviceHisto256<T>::ref, offset)));
//#endif
        }

        __host__ __device__ __forceinline__ bool operator==(const self_type& rhs)
        {
//            return ((ptr == rhs.ptr) && (offset == offset));
            return (ptr == rhs.ptr);
        }

        __host__ __device__ __forceinline__ bool operator!=(const self_type& rhs)
        {
//            return ((ptr != rhs.ptr) && (offset == offset));
            return (ptr != rhs.ptr);
        }

    private:

        BinOp   bin_op;
        T*      ptr;
//        size_t  offset;
    };


    /**
     * \brief Computes a 256-bin device-wide histogram
     *
     * \tparam InputIteratorRA      <b>[inferred]</b> The random-access iterator type for input (may be a simple pointer type).  Must have a value type that is assignable to <tt>unsigned char</tt>
     * \tparam HistoCounter         <b>[inferred]</b> Integral type for counting sample occurrences per histogram bin
     */
    template <
        typename            InputIteratorRA,
        typename            HistoCounter>
    __host__ __device__ __forceinline__
    static cudaError_t SingleChannel(
        InputIteratorRA     d_samples,                                          ///< [in] Input samples
        HistoCounter*       d_histogram,                                        ///< [out] Array of 256 counters of integral type \p HistoCounter.
        int                 num_samples,                                        ///< [in] Number of samples to process
        cudaStream_t        stream              = 0,                            ///< [in] <b>[optional]</b> CUDA stream to launch kernels within.  Default is stream-0.
        bool                stream_synchronous  = false,                        ///< [in] <b>[optional]</b> Whether or not to synchronize the stream after every kernel launch to check for errors.  Default is \p false.
        DeviceAllocator*    device_allocator    = DefaultDeviceAllocator())     ///< [in] <b>[optional]</b> Allocator for allocating and freeing device memory.  Default is provided by DefaultDeviceAllocator.
    {
        return Dispatch<BLOCK_BYTE_HISTO_SORT, 1, 1>(
            d_samples, &d_histogram, num_samples, stream, stream_synchronous, device_allocator);
    }

    /**
     * \brief Computes a 256-bin device-wide histogram.  Uses atomic read-modify-write operations to compute the histogram.
     *
     * Sample input having lower diversity cause performance to be degraded.
     *
     * \tparam InputIteratorRA      <b>[inferred]</b> The random-access iterator type for input (may be a simple pointer type).  Must have a value type that is assignable to <tt>unsigned char</tt>
     * \tparam HistoCounter         <b>[inferred]</b> Integral type for counting sample occurrences per histogram bin
     */
    template <
        typename            InputIteratorRA,
        typename            HistoCounter>
    __host__ __device__ __forceinline__
    static cudaError_t SingleChannelAtomic(
        InputIteratorRA     d_samples,                                          ///< [in] Input samples
        HistoCounter*       d_histogram,                                        ///< [out] Array of 256 counters of integral type \p HistoCounter.
        int                 num_samples,                                        ///< [in] Number of samples to process
        cudaStream_t        stream              = 0,                            ///< [in] <b>[optional]</b> CUDA stream to launch kernels within.  Default is stream-0.
        bool                stream_synchronous  = false,                        ///< [in] <b>[optional]</b> Whether or not to synchronize the stream after every kernel launch to check for errors.  Default is \p false.
        DeviceAllocator*    device_allocator    = DefaultDeviceAllocator())     ///< [in] <b>[optional]</b> Allocator for allocating and freeing device memory.  Default is provided by DefaultDeviceAllocator.
    {
        return Dispatch<BLOCK_BYTE_HISTO_ATOMIC, 1, 1>(
            d_samples, &d_histogram, num_samples, stream, stream_synchronous, device_allocator);
    }


    /**
     * \brief Computes a 256-bin device-wide histogram from multi-channel data.
     *
     * \tparam CHANNELS             Number of channels interleaved in the input data (may be greater than the number of channels being actively histogrammed)
     * \tparam ACTIVE_CHANNELS      <b>[inferred]</b> Number of channels actively being histogrammed
     * \tparam InputIteratorRA      <b>[inferred]</b> The random-access iterator type for input (may be a simple pointer type).  Must have a value type that is assignable to <tt>unsigned char</tt>
     * \tparam HistoCounter         <b>[inferred]</b> Integral type for counting sample occurrences per histogram bin
     */
    template <
        int                 CHANNELS,
        int                 ACTIVE_CHANNELS,
        typename            InputIteratorRA,
        typename            HistoCounter>
    __host__ __device__ __forceinline__
    static cudaError_t MultiChannel(
        InputIteratorRA     d_samples,                                          ///< [in] Input samples. (Channels, if any, are interleaved in "AOS" format)
        HistoCounter        *(&d_histograms)[ACTIVE_CHANNELS],                  ///< [out] Array of channel histograms, each having 256 counters of integral type \p HistoCounter.
        int                 num_samples,                                        ///< [in] Number of samples to process
        cudaStream_t        stream              = 0,                            ///< [in] <b>[optional]</b> CUDA stream to launch kernels within.  Default is stream-0.
        bool                stream_synchronous  = false,                        ///< [in] <b>[optional]</b> Whether or not to synchronize the stream after every kernel launch to check for errors.  Default is \p false.
        DeviceAllocator*    device_allocator    = DefaultDeviceAllocator())     ///< [in] <b>[optional]</b> Allocator for allocating and freeing device memory.  Default is provided by DefaultDeviceAllocator.
    {
        return Dispatch<BLOCK_BYTE_HISTO_SORT, CHANNELS, ACTIVE_CHANNELS>(
            d_samples, d_histograms, num_samples, stream, stream_synchronous, device_allocator);
    }

    /**
     * \brief Computes a 256-bin device-wide histogram from multi-channel data.  Uses atomic read-modify-write operations to compute the histogram.
     *
     * Sample input having lower diversity cause performance to be degraded.
     *
     * \tparam CHANNELS             Number of channels interleaved in the input data (may be greater than the number of channels being actively histogrammed)
     * \tparam ACTIVE_CHANNELS      <b>[inferred]</b> Number of channels actively being histogrammed
     * \tparam InputIteratorRA      <b>[inferred]</b> The random-access iterator type for input (may be a simple pointer type).  Must have a value type that is assignable to <tt>unsigned char</tt>
     * \tparam HistoCounter         <b>[inferred]</b> Integral type for counting sample occurrences per histogram bin
     */
    template <
        int                 CHANNELS,                                           ///< Number of channels interleaved in the input data (may be greater than the number of channels being actively histogrammed)
        int                 ACTIVE_CHANNELS,                                    ///< Number of channels actively being histogrammed
        typename            InputIteratorRA,
        typename            HistoCounter>
    __host__ __device__ __forceinline__
    static cudaError_t MultiChannelAtomic(
        InputIteratorRA     d_samples,                                          ///< [in] Input samples. (Channels, if any, are interleaved in "AOS" format)
        HistoCounter        *(&d_histograms)[ACTIVE_CHANNELS],                  ///< [out] Array of channel histograms, each having 256 counters of integral type \p HistoCounter.
        int                 num_samples,                                        ///< [in] Number of samples to process
        cudaStream_t        stream              = 0,                            ///< [in] <b>[optional]</b> CUDA stream to launch kernels within.  Default is stream-0.
        bool                stream_synchronous  = false,                        ///< [in] <b>[optional]</b> Whether or not to synchronize the stream after every kernel launch to check for errors.  Default is \p false.
        DeviceAllocator*    device_allocator    = DefaultDeviceAllocator())     ///< [in] <b>[optional]</b> Allocator for allocating and freeing device memory.  Default is provided by DefaultDeviceAllocator.
    {
        return Dispatch<BLOCK_BYTE_HISTO_ATOMIC, CHANNELS, ACTIVE_CHANNELS>(
            d_samples, d_histograms, num_samples, stream, stream_synchronous, device_allocator);
    }


};


/** @} */       // DeviceModule

}               // CUB namespace
CUB_NS_POSTFIX  // Optional outer namespace(s)


