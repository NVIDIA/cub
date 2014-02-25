
/******************************************************************************
 * Copyright (c) 2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2014, NVIDIA CORPORATION.  All rights reserved.
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
 * cub::DeviceHistogram provides device-wide parallel operations for constructing histogram(s) from a sequence of samples data residing within global memory.
 */

#pragma once

#include <stdio.h>
#include <iterator>

#include "region/block_histo_region.cuh"
#include "../grid/grid_even_share.cuh"
#include "../grid/grid_queue.cuh"
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
 * Initialization kernel entry point (multi-block).  Prepares queue descriptors and zeroes global counters.
 */
template <
    int                                             BINS,                   ///< Number of histogram bins per channel
    int                                             ACTIVE_CHANNELS,        ///< Number of channels actively being histogrammed
    typename                                        Offset,                 ///< Signed integer type for global offsets
    typename                                        HistoCounter>           ///< Integer type for counting sample occurrences per histogram bin
__launch_bounds__ (BINS, 1)
__global__ void HistoInitKernel(
    GridQueue<Offset>                               grid_queue,             ///< [in] Drain queue descriptor for dynamically mapping tile data onto thread blocks
    ArrayWrapper<HistoCounter*, ACTIVE_CHANNELS>    d_out_histograms,       ///< [out] Histogram counter data having logical dimensions <tt>HistoCounter[ACTIVE_CHANNELS][BINS]</tt>
    Offset                                          num_samples)            ///< [in] Total number of samples \p d_samples for all channels
{
    d_out_histograms.array[blockIdx.x][threadIdx.x] = 0;
    if (threadIdx.x == 0) grid_queue.FillAndResetDrain(num_samples);
}


/**
 * Histogram tiles kernel entry point (multi-block).  Computes privatized histograms, one per thread block.
 */
template <
    typename                                        BlockHistogramRegionPolicy, ///< Parameterized BlockHistogramRegionPolicy tuning policy type
    int                                             BINS,                       ///< Number of histogram bins per channel
    int                                             CHANNELS,                   ///< Number of channels interleaved in the input data (may be greater than the number of channels being actively histogrammed)
    int                                             ACTIVE_CHANNELS,            ///< Number of channels actively being histogrammed
    typename                                        InputIterator,              ///< The input iterator type \iterator.  Must have a value type that is assignable to <tt>unsigned char</tt>
    typename                                        HistoCounter,               ///< Integer type for counting sample occurrences per histogram bin
    typename                                        Offset>                     ///< Signed integer type for global offsets
__launch_bounds__ (int(BlockHistogramRegionPolicy::BLOCK_THREADS))
__global__ void HistoRegionKernel(
    InputIterator                                   d_samples,                  ///< [in] Array of sample data. The samples from different channels are assumed to be interleaved (e.g., an array of 32b pixels where each pixel consists of four RGBA 8b samples).
    ArrayWrapper<HistoCounter*, ACTIVE_CHANNELS>    d_out_histograms,           ///< [out] Histogram counter data having logical dimensions <tt>HistoCounter[ACTIVE_CHANNELS][gridDim.x][BINS]</tt>
    Offset                                          num_samples,                ///< [in] Total number of samples \p d_samples for all channels
    GridEvenShare<Offset>                           even_share,                 ///< [in] Even-share descriptor for mapping an equal number of tiles onto each thread block
    GridQueue<Offset>                               queue)                      ///< [in] Drain queue descriptor for dynamically mapping tile data onto thread blocks
{
    // Constants
    enum
    {
        BLOCK_THREADS       = BlockHistogramRegionPolicy::BLOCK_THREADS,
        ITEMS_PER_THREAD    = BlockHistogramRegionPolicy::ITEMS_PER_THREAD,
        TILE_SIZE           = BLOCK_THREADS * ITEMS_PER_THREAD,
    };

    // Thread block type for compositing input tiles
    typedef BlockHistogramRegion<BlockHistogramRegionPolicy, BINS, CHANNELS, ACTIVE_CHANNELS, InputIterator, HistoCounter, Offset> BlockHistogramRegionT;

    // Shared memory for BlockHistogramRegion
    __shared__ typename BlockHistogramRegionT::TempStorage temp_storage;

    // Consume input tiles
    BlockHistogramRegionT(temp_storage, d_samples, d_out_histograms.array).ConsumeRegion(
        num_samples,
        even_share,
        queue,
        Int2Type<BlockHistogramRegionPolicy::GRID_MAPPING>());
}


/**
 * Aggregation kernel entry point (single-block).  Aggregates privatized threadblock histograms from a previous multi-block histogram pass.
 */
template <
    int                                             BINS,                   ///< Number of histogram bins per channel
    int                                             ACTIVE_CHANNELS,        ///< Number of channels actively being histogrammed
    typename                                        HistoCounter>           ///< Integer type for counting sample occurrences per histogram bin
__launch_bounds__ (BINS, 1)
__global__ void HistoAggregateKernel(
    HistoCounter*                                   d_block_histograms,     ///< [in] Histogram counter data having logical dimensions <tt>HistoCounter[ACTIVE_CHANNELS][num_threadblocks][BINS]</tt>
    ArrayWrapper<HistoCounter*, ACTIVE_CHANNELS>    d_out_histograms,       ///< [out] Histogram counter data having logical dimensions <tt>HistoCounter[ACTIVE_CHANNELS][BINS]</tt>
    int                                             num_threadblocks)       ///< [in] Number of threadblock histograms per channel in \p d_block_histograms
{
    // Accumulate threadblock-histograms from the channel
    HistoCounter bin_aggregate = 0;

    int block_offset = blockIdx.x * (num_threadblocks * BINS);
    int block_end = block_offset + (num_threadblocks * BINS);

#if CUB_PTX_VERSION >= 200
    #pragma unroll 32
#endif
    while (block_offset < block_end)
    {
        HistoCounter block_bin_count = d_block_histograms[block_offset + threadIdx.x];

        bin_aggregate += block_bin_count;
        block_offset += BINS;
    }

    // Output
    d_out_histograms.array[blockIdx.x][threadIdx.x] = bin_aggregate;
}



/******************************************************************************
 * Dispatch
 ******************************************************************************/

/**
 * Utility class for dispatching the appropriately-tuned kernels for DeviceHistogram
 */
template <
    DeviceHistogramAlgorithm        HISTO_ALGORITHM,            ///< Cooperative histogram algorithm to use
    int                             BINS,                       ///< Number of histogram bins per channel
    int                             CHANNELS,                   ///< Number of channels interleaved in the input data (may be greater than the number of channels being actively histogrammed)
    int                             ACTIVE_CHANNELS,            ///< Number of channels actively being histogrammed
    typename                        InputIterator,              ///< The input iterator type \iterator.  Must have a value type that is assignable to <tt>unsigned char</tt>
    typename                        HistoCounter,               ///< Integer type for counting sample occurrences per histogram bin
    typename                        Offset>                     ///< Signed integer type for global offsets
struct DeviceHistogramDispatch
{
    /******************************************************************************
     * Tuning policies
     ******************************************************************************/

    /// SM35
    struct Policy350
    {
        // HistoRegionPolicy
        typedef BlockHistogramRegionPolicy<
                (HISTO_ALGORITHM == DEVICE_HISTO_SORT) ? 128 : 256,
                (HISTO_ALGORITHM == DEVICE_HISTO_SORT) ? 12 : (30 / ACTIVE_CHANNELS),
                HISTO_ALGORITHM,
                (HISTO_ALGORITHM == DEVICE_HISTO_SORT) ? GRID_MAPPING_DYNAMIC : GRID_MAPPING_EVEN_SHARE>
            HistoRegionPolicy;
    };

    /// SM30
    struct Policy300
    {
        // HistoRegionPolicy
        typedef BlockHistogramRegionPolicy<
                128,
                (HISTO_ALGORITHM == DEVICE_HISTO_SORT) ? 20 : (22 / ACTIVE_CHANNELS),
                HISTO_ALGORITHM,
                (HISTO_ALGORITHM == DEVICE_HISTO_SORT) ? GRID_MAPPING_DYNAMIC : GRID_MAPPING_EVEN_SHARE>
            HistoRegionPolicy;
    };

    /// SM20
    struct Policy200
    {
        // HistoRegionPolicy
        typedef BlockHistogramRegionPolicy<
                128,
                (HISTO_ALGORITHM == DEVICE_HISTO_SORT) ? 21 : (23 / ACTIVE_CHANNELS),
                HISTO_ALGORITHM,
                GRID_MAPPING_DYNAMIC>
            HistoRegionPolicy;
    };

    /// SM10
    struct Policy100
    {
        // HistoRegionPolicy
        typedef BlockHistogramRegionPolicy<
                128,
                7,
                DEVICE_HISTO_SORT,        // (use sort regardless because g-atomics are unsupported and s-atomics are perf-useless)
                GRID_MAPPING_EVEN_SHARE>
            HistoRegionPolicy;
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

#else
    typedef Policy100 PtxPolicy;

#endif

    // "Opaque" policies (whose parameterizations aren't reflected in the type signature)
    struct PtxHistoRegionPolicy : PtxPolicy::HistoRegionPolicy {};


    /******************************************************************************
     * Utilities
     ******************************************************************************/

    /**
     * Initialize kernel dispatch configurations with the policies corresponding to the PTX assembly we will use
     */
    template <typename KernelConfig>
    __host__ __device__ __forceinline__
    static void InitConfigs(
        int             ptx_version,
        KernelConfig    &histo_region_config)
    {
    #if (CUB_PTX_VERSION > 0)

        // We're on the device, so initialize the kernel dispatch configurations with the current PTX policy
        histo_region_config.template Init<PtxHistoRegionPolicy>();

    #else

        // We're on the host, so lookup and initialize the kernel dispatch configurations with the policies that match the device's PTX version
        if (ptx_version >= 350)
        {
            histo_region_config.template Init<typename Policy350::HistoRegionPolicy>();
        }
        else if (ptx_version >= 300)
        {
            histo_region_config.template Init<typename Policy300::HistoRegionPolicy>();
        }
        else if (ptx_version >= 200)
        {
            histo_region_config.template Init<typename Policy200::HistoRegionPolicy>();
        }
        else
        {
            histo_region_config.template Init<typename Policy100::HistoRegionPolicy>();
        }

    #endif
    }


    /**
     * Kernel kernel dispatch configuration
     */
    struct KernelConfig
    {
        int                             block_threads;
        int                             items_per_thread;
        DeviceHistogramAlgorithm        block_algorithm;
        GridMappingStrategy             grid_mapping;

        template <typename BlockPolicy>
        __host__ __device__ __forceinline__
        void Init()
        {
            block_threads               = BlockPolicy::BLOCK_THREADS;
            items_per_thread            = BlockPolicy::ITEMS_PER_THREAD;
            block_algorithm             = BlockPolicy::HISTO_ALGORITHM;
            grid_mapping                = BlockPolicy::GRID_MAPPING;
        }

        __host__ __device__ __forceinline__
        void Print()
        {
            printf("%d, %d, %d, %d", block_threads, items_per_thread, block_algorithm, grid_mapping);
        }

    };


    /******************************************************************************
     * Dispatch entrypoints
     ******************************************************************************/


    /**
     * Internal dispatch routine
     */
    template <
        typename                    InitHistoKernelPtr,                 ///< Function type of cub::HistoInitKernel
        typename                    HistoRegionKernelPtr,               ///< Function type of cub::HistoRegionKernel
        typename                    AggregateHistoKernelPtr>            ///< Function type of cub::HistoAggregateKernel
    __host__ __device__ __forceinline__
    static cudaError_t Dispatch(
        void                        *d_temp_storage,                    ///< [in] %Device allocation of temporary storage.  When NULL, the required allocation size is written to \p temp_storage_bytes and no work is done.
        size_t                      &temp_storage_bytes,                ///< [in,out] Reference to size in bytes of \p d_temp_storage allocation
        InputIterator               d_samples,                          ///< [in] Input samples to histogram
        HistoCounter                *d_histograms[ACTIVE_CHANNELS],     ///< [out] Array of channel histograms, each having BINS counters of integral type \p HistoCounter.
        Offset                      num_samples,                        ///< [in] Number of samples to process
        cudaStream_t                stream,                             ///< [in] CUDA stream to launch kernels within.  Default is stream<sub>0</sub>.
        bool                        debug_synchronous,                  ///< [in] Whether or not to synchronize the stream after every kernel launch to check for errors.  Default is \p false.
        InitHistoKernelPtr          init_kernel,                        ///< [in] Kernel function pointer to parameterization of cub::HistoInitKernel
        HistoRegionKernelPtr        histo_region_kernel,                ///< [in] Kernel function pointer to parameterization of cub::HistoRegionKernel
        AggregateHistoKernelPtr     aggregate_kernel,                   ///< [in] Kernel function pointer to parameterization of cub::HistoAggregateKernel
        KernelConfig                histo_region_config)                ///< [in] Dispatch parameters that match the policy that \p histo_region_kernel was compiled for
    {
    #ifndef CUB_RUNTIME_ENABLED

        // Kernel launch not supported from this device
        return CubDebug(cudaErrorNotSupported);

    #else

        cudaError error = cudaSuccess;
        do
        {
            // Get device ordinal
            int device_ordinal;
            if (CubDebug(error = cudaGetDevice(&device_ordinal))) break;

            // Get device SM version
            int sm_version;
            if (CubDebug(error = SmVersion(sm_version, device_ordinal))) break;

            // Get SM count
            int sm_count;
            if (CubDebug(error = cudaDeviceGetAttribute (&sm_count, cudaDevAttrMultiProcessorCount, device_ordinal))) break;

            // Get SM occupancy for histo_region_kernel
            int histo_region_sm_occupancy;
            if (CubDebug(error = MaxSmOccupancy(
                histo_region_sm_occupancy,
                sm_version,
                histo_region_kernel,
                histo_region_config.block_threads))) break;

            // Get device occupancy for histo_region_kernel
            int histo_region_occupancy = histo_region_sm_occupancy * sm_count;

            // Get tile size for histo_region_kernel
            int channel_tile_size = histo_region_config.block_threads * histo_region_config.items_per_thread;
            int tile_size = channel_tile_size * CHANNELS;

            // Even-share work distribution
            int subscription_factor = histo_region_sm_occupancy;     // Amount of CTAs to oversubscribe the device beyond actively-resident (heuristic)
            GridEvenShare<Offset> even_share(
                num_samples,
                histo_region_occupancy * subscription_factor,
                tile_size);

            // Get grid size for histo_region_kernel
            int histo_region_grid_size;
            switch (histo_region_config.grid_mapping)
            {
            case GRID_MAPPING_EVEN_SHARE:

                // Work is distributed evenly
                histo_region_grid_size = even_share.grid_size;
                break;

            case GRID_MAPPING_DYNAMIC:

                // Work is distributed dynamically
                int num_tiles               = (num_samples + tile_size - 1) / tile_size;
                histo_region_grid_size   = (num_tiles < histo_region_occupancy) ?
                    num_tiles :                     // Not enough to fill the device with threadblocks
                    histo_region_occupancy;      // Fill the device with threadblocks
                break;
            };

            // Temporary storage allocation requirements
            void* allocations[2];
            size_t allocation_sizes[2] =
            {
                ACTIVE_CHANNELS * histo_region_grid_size * sizeof(HistoCounter) * BINS,      // bytes needed for privatized histograms
                GridQueue<int>::AllocationSize()                                                // bytes needed for grid queue descriptor
            };

            // Alias the temporary allocations from the single storage blob (or set the necessary size of the blob)
            if (CubDebug(error = AliasTemporaries(d_temp_storage, temp_storage_bytes, allocations, allocation_sizes))) break;
            if (d_temp_storage == NULL)
            {
                // Return if the caller is simply requesting the size of the storage allocation
                return cudaSuccess;
            }

            // Alias the allocation for the privatized per-block reductions
            HistoCounter *d_block_histograms = (HistoCounter*) allocations[0];

            // Alias the allocation for the grid queue descriptor
            GridQueue<Offset> queue(allocations[1]);

            // Setup array wrapper for histogram channel output (because we can't pass static arrays as kernel parameters)
            ArrayWrapper<HistoCounter*, ACTIVE_CHANNELS> d_histo_wrapper;
            for (int CHANNEL = 0; CHANNEL < ACTIVE_CHANNELS; ++CHANNEL)
                d_histo_wrapper.array[CHANNEL] = d_histograms[CHANNEL];

            // Setup array wrapper for temporary histogram channel output (because we can't pass static arrays as kernel parameters)
            ArrayWrapper<HistoCounter*, ACTIVE_CHANNELS> d_temp_histo_wrapper;
            for (int CHANNEL = 0; CHANNEL < ACTIVE_CHANNELS; ++CHANNEL)
                d_temp_histo_wrapper.array[CHANNEL] = d_block_histograms + (CHANNEL * histo_region_grid_size * BINS);

            // Log init_kernel configuration
            if (debug_synchronous) CubLog("Invoking init_kernel<<<%d, %d, 0, %lld>>>()\n", ACTIVE_CHANNELS, BINS, (long long) stream);

            // Invoke init_kernel to initialize counters and queue descriptor
            init_kernel<<<ACTIVE_CHANNELS, BINS, 0, stream>>>(queue, d_histo_wrapper, num_samples);

            // Sync the stream if specified
            if (debug_synchronous && (CubDebug(error = SyncStream(stream)))) break;

            // Whether we need privatized histograms (i.e., non-global atomics and multi-block)
            bool privatized_temporaries = (histo_region_grid_size > 1) && (histo_region_config.block_algorithm != DEVICE_HISTO_GLOBAL_ATOMIC);

            // Log histo_region_kernel configuration
            if (debug_synchronous) CubLog("Invoking histo_region_kernel<<<%d, %d, 0, %lld>>>(), %d items per thread, %d SM occupancy\n",
                histo_region_grid_size, histo_region_config.block_threads, (long long) stream, histo_region_config.items_per_thread, histo_region_sm_occupancy);

            // Invoke histo_region_kernel
            histo_region_kernel<<<histo_region_grid_size, histo_region_config.block_threads, 0, stream>>>(
                d_samples,
                (privatized_temporaries) ?
                    d_temp_histo_wrapper :
                    d_histo_wrapper,
                num_samples,
                even_share,
                queue);

            // Sync the stream if specified
            if (debug_synchronous && (CubDebug(error = SyncStream(stream)))) break;

            // Aggregate privatized block histograms if necessary
            if (privatized_temporaries)
            {
                // Log aggregate_kernel configuration
                if (debug_synchronous) CubLog("Invoking aggregate_kernel<<<%d, %d, 0, %lld>>>()\n",
                    ACTIVE_CHANNELS, BINS, (long long) stream);

                // Invoke aggregate_kernel
                aggregate_kernel<<<ACTIVE_CHANNELS, BINS, 0, stream>>>(
                    d_block_histograms,
                    d_histo_wrapper,
                    histo_region_grid_size);

                // Sync the stream if specified
                if (debug_synchronous && (CubDebug(error = SyncStream(stream)))) break;
            }
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
        void                *d_temp_storage,                    ///< [in] %Device allocation of temporary storage.  When NULL, the required allocation size is written to \p temp_storage_bytes and no work is done.
        size_t              &temp_storage_bytes,                ///< [in,out] Reference to size in bytes of \p d_temp_storage allocation
        InputIterator       d_samples,                          ///< [in] Input samples to histogram
        HistoCounter        *d_histograms[ACTIVE_CHANNELS],     ///< [out] Array of channel histograms, each having BINS counters of integral type \p HistoCounter.
        int                 num_samples,                        ///< [in] Number of samples to process
        cudaStream_t        stream,                             ///< [in] <b>[optional]</b> CUDA stream to launch kernels within.  Default is stream<sub>0</sub>.
        bool                debug_synchronous)                  ///< [in] <b>[optional]</b> Whether or not to synchronize the stream after every kernel launch to check for errors.  Also causes launch configurations to be printed to the console.  Default is \p false.
    {
        cudaError error = cudaSuccess;
        do
        {
            // Get PTX version
            int ptx_version;
    #if (CUB_PTX_VERSION == 0)
            if (CubDebug(error = PtxVersion(ptx_version))) break;
    #else
            ptx_version = CUB_PTX_VERSION;
    #endif

            // Get kernel kernel dispatch configurations
            KernelConfig histo_region_config;
            InitConfigs(ptx_version, histo_region_config);

            // Dispatch
            if (CubDebug(error = Dispatch(
                d_temp_storage,
                temp_storage_bytes,
                d_samples,
                d_histograms,
                num_samples,
                stream,
                debug_synchronous,
                HistoInitKernel<BINS, ACTIVE_CHANNELS, Offset, HistoCounter>,
                HistoRegionKernel<PtxHistoRegionPolicy, BINS, CHANNELS, ACTIVE_CHANNELS, InputIterator, HistoCounter, Offset>,
                HistoAggregateKernel<BINS, ACTIVE_CHANNELS, HistoCounter>,
                histo_region_config))) break;
        }
        while (0);

        return error;
    }
};


#endif // DOXYGEN_SHOULD_SKIP_THIS



/******************************************************************************
 * DeviceHistogram
 *****************************************************************************/

/**
 * \brief DeviceHistogram provides device-wide parallel operations for constructing histogram(s) from a sequence of samples data residing within global memory. ![](histogram_logo.png)
 * \ingroup DeviceModule
 *
 * \par Overview
 * A <a href="http://en.wikipedia.org/wiki/Histogram"><em>histogram</em></a>
 * counts the number of observations that fall into each of the disjoint categories (known as <em>bins</em>).
 *
 * \par Usage Considerations
 * \cdp_class{DeviceHistogram}
 *
 * \par Performance
 *
 * \image html histo_perf.png
 *
 */
struct DeviceHistogram
{
    /******************************************************************//**
     * \name Single-channel samples
     *********************************************************************/
    //@{


    /**
     * \brief Computes a device-wide histogram using fast block-wide sorting.
     *
     * \par
     * - The total number of samples across all channels (\p num_samples) must be a whole multiple of \p CHANNELS.
     * - Delivers consistent throughput regardless of sample diversity
     * - Histograms having a large number of bins (e.g., thousands) may adversely affect shared memory occupancy and performance (or even the ability to launch).
     * - Performance is often improved when referencing input samples through a texture-caching iterator (e.g., cub::TexObjInputIterator).
     * - \devicestorage
     * - \cdp
     *
     * \par Snippet
     * The code snippet below illustrates the computation of a 8-bin histogram of
     * single-channel <tt>unsigned char</tt> samples.
     * \par
     * \code
     * #include <cub/cub.cuh>   // or equivalently <cub/device/device_histogram.cuh>
     *
     * // Declare, allocate, and initialize device pointers for input and histogram
     * int              num_samples;    // e.g., 12
     * unsigned char    *d_samples;     // e.g., [2, 6, 7, 5, 3, 0, 2, 1, 7, 0, 6, 2]
     * unsigned int     *d_histogram;   // e.g., [ ,  ,  ,  ,  ,  ,  ,  ]
     * ...
     *
     * // Wrap d_samples device pointer in a random-access texture iterator
     * cub::TexObjInputIterator<unsigned char> d_samples_tex_itr;
     * d_samples_tex_itr.BindTexture(d_samples, num_samples * sizeof(unsigned char));
     *
     * // Determine temporary device storage requirements
     * void *d_temp_storage = NULL;
     * size_t temp_storage_bytes = 0;
     * cub::DeviceHistogram::SingleChannelSorting<8>(d_temp_storage, temp_storage_bytes, d_samples_tex_itr, d_histogram, num_samples);
     *
     * // Allocate temporary storage
     * cudaMalloc(&d_temp_storage, temp_storage_bytes);
     *
     * // Compute histogram
     * cub::DeviceHistogram::SingleChannelSorting<8>(d_temp_storage, temp_storage_bytes, d_samples_tex_itr, d_histogram, num_samples);
     *
     * // Unbind texture iterator
     * d_samples_tex_itr.UnbindTexture();
     *
     * // d_histogram   <-- [2, 1, 3, 1, 0, 1, 2, 2]
     *
     * \endcode
     *
     * \tparam BINS                 Number of histogram bins per channel
     * \tparam InputIterator        <b>[inferred]</b> Random-access input iterator type for reading input samples. (Must have an InputIterator::value_type that, when cast as an integer, falls in the range [0..BINS-1])  \iterator
     * \tparam HistoCounter         <b>[inferred]</b> Integer type for counting sample occurrences per histogram bin
     */
    template <
        int                 BINS,
        typename            InputIterator,
        typename            HistoCounter>
    __host__ __device__
    static cudaError_t SingleChannelSorting(
        void                *d_temp_storage,                    ///< [in] %Device allocation of temporary storage.  When NULL, the required allocation size is written to \p temp_storage_bytes and no work is done.
        size_t              &temp_storage_bytes,                ///< [in,out] Reference to size in bytes of \p d_temp_storage allocation
        InputIterator       d_samples,                          ///< [in] Input samples
        HistoCounter*       d_histogram,                        ///< [out] Array of BINS counters of integral type \p HistoCounter.
        int                 num_samples,                        ///< [in] Number of samples to process
        cudaStream_t        stream              = 0,            ///< [in] <b>[optional]</b> CUDA stream to launch kernels within.  Default is stream<sub>0</sub>.
        bool                debug_synchronous   = false)        ///< [in] <b>[optional]</b> Whether or not to synchronize the stream after every kernel launch to check for errors.  May cause significant slowdown.  Default is \p false.
    {
        // Signed integer type for global offsets
        typedef int Offset;

        // Dispatch type
        typedef DeviceHistogramDispatch<
                DEVICE_HISTO_SORT,
                BINS,
                1,
                1,
                InputIterator,
                HistoCounter,
                Offset>
            DeviceHistogramDispatch;

        return DeviceHistogramDispatch::Dispatch(
            d_temp_storage,
            temp_storage_bytes,
            d_samples,
            &d_histogram,
            num_samples,
            stream,
            debug_synchronous);
    }


    /**
     * \brief Computes a device-wide histogram using shared-memory atomic read-modify-write operations.
     *
     * \par
     * - Input samples having lower diversity can cause performance to be degraded due to serializations from bin-collisions.
     * - Histograms having a large number of bins (e.g., thousands) may adversely affect shared memory occupancy and performance (or even the ability to launch).
     * - Performance is often improved when referencing input samples through a texture-caching iterator (e.g., cub::TexObjInputIterator).
     * - \devicestorage
     * - \cdp
     *
     * \par Snippet
     * The code snippet below illustrates the computation of a 8-bin histogram of
     * single-channel <tt>unsigned char</tt> samples.
     * \par
     * \code
     * #include <cub/cub.cuh>   // or equivalently <cub/device/device_histogram.cuh>
     *
     * // Declare, allocate, and initialize device pointers for input and histogram
     * int              num_samples;    // e.g., 12
     * unsigned char    *d_samples;     // e.g., [2, 6, 7, 5, 3, 0, 2, 1, 7, 0, 6, 2]
     * unsigned int     *d_histogram;   // e.g., [ ,  ,  ,  ,  ,  ,  ,  ]
     * ...
     *
     * // Wrap d_samples device pointer in a random-access texture iterator
     * cub::TexObjInputIterator<unsigned char> d_samples_tex_itr;
     * d_samples_tex_itr.BindTexture(d_samples, num_samples * sizeof(unsigned char));
     *
     * // Determine temporary device storage requirements
     * void     *d_temp_storage = NULL;
     * size_t   temp_storage_bytes = 0;
     * cub::DeviceHistogram::SingleChannelSorting<8>(d_temp_storage, temp_storage_bytes, d_samples_tex_itr, d_histogram, num_samples);
     *
     * // Allocate temporary storage
     * cudaMalloc(&d_temp_storage, temp_storage_bytes);
     *
     * // Compute histogram
     * cub::DeviceHistogram::SingleChannelSharedAtomic<8>(d_temp_storage, temp_storage_bytes, d_samples_tex_itr, d_histogram, num_samples);
     *
     * // Unbind texture iterator
     * d_samples_tex_itr.UnbindTexture();
     *
     * // d_histogram   <-- [2, 1, 3, 1, 0, 1, 2, 2]
     *
     * \endcode
     *
     * \tparam BINS                 Number of histogram bins per channel
     * \tparam InputIterator        <b>[inferred]</b> Random-access input iterator type for reading input samples. (Must have an InputIterator::value_type that, when cast as an integer, falls in the range [0..BINS-1])  \iterator
     * \tparam HistoCounter         <b>[inferred]</b> Integer type for counting sample occurrences per histogram bin
     */
    template <
        int                 BINS,
        typename            InputIterator,
        typename            HistoCounter>
    __host__ __device__
    static cudaError_t SingleChannelSharedAtomic(
        void                *d_temp_storage,                    ///< [in] %Device allocation of temporary storage.  When NULL, the required allocation size is written to \p temp_storage_bytes and no work is done.
        size_t              &temp_storage_bytes,                ///< [in,out] Reference to size in bytes of \p d_temp_storage allocation
        InputIterator       d_samples,                          ///< [in] Input samples
        HistoCounter*       d_histogram,                        ///< [out] Array of BINS counters of integral type \p HistoCounter.
        int                 num_samples,                        ///< [in] Number of samples to process
        cudaStream_t        stream              = 0,            ///< [in] <b>[optional]</b> CUDA stream to launch kernels within.  Default is stream<sub>0</sub>.
        bool                debug_synchronous   = false)        ///< [in] <b>[optional]</b> Whether or not to synchronize the stream after every kernel launch to check for errors.  May cause significant slowdown.  Default is \p false.
    {
        // Signed integer type for global offsets
        typedef int Offset;

        // Dispatch type
        typedef DeviceHistogramDispatch<
                DEVICE_HISTO_SHARED_ATOMIC,
                BINS,
                1,
                1,
                InputIterator,
                HistoCounter,
                Offset>
            DeviceHistogramDispatch;

        return DeviceHistogramDispatch::Dispatch(
            d_temp_storage,
            temp_storage_bytes,
            d_samples,
            &d_histogram,
            num_samples,
            stream,
            debug_synchronous);
    }


    /**
     * \brief Computes a device-wide histogram using global-memory atomic read-modify-write operations.
     *
     * \par
     * - Input samples having lower diversity can cause performance to be degraded due to serializations from bin-collisions.
     * - Performance is not significantly impacted when computing histograms having large numbers of bins (e.g., thousands).
     * - Performance is often improved when referencing input samples through a texture-caching iterator (e.g., cub::TexObjInputIterator).
     * - \devicestorage
     * - \cdp
     *
     * \par Snippet
     * The code snippet below illustrates the computation of a 8-bin histogram of
     * single-channel <tt>unsigned char</tt> samples.
     * \par
     * \code
     * #include <cub/cub.cuh>   // or equivalently <cub/device/device_histogram.cuh>
     *
     * // Declare, allocate, and initialize device pointers for input and histogram
     * int              num_samples;    // e.g., 12
     * unsigned char    *d_samples;     // e.g., [2, 6, 7, 5, 3, 0, 2, 1, 7, 0, 6, 2]
     * unsigned int     *d_histogram;   // e.g., [ ,  ,  ,  ,  ,  ,  ,  ]
     * ...
     *
     * // Wrap d_samples device pointer in a random-access texture iterator
     * cub::TexObjInputIterator<unsigned char> d_samples_tex_itr;
     * d_samples_tex_itr.BindTexture(d_samples, num_samples * sizeof(unsigned char));
     *
     * // Determine temporary device storage requirements
     * void     *d_temp_storage = NULL;
     * size_t   temp_storage_bytes = 0;
     * cub::DeviceHistogram::SingleChannelSorting<8>(d_temp_storage, temp_storage_bytes, d_samples_tex_itr, d_histogram, num_samples);
     *
     * // Allocate temporary storage
     * cudaMalloc(&d_temp_storage, temp_storage_bytes);
     *
     * // Compute histogram
     * cub::DeviceHistogram::SingleChannelGlobalAtomic<8>(d_temp_storage, temp_storage_bytes, d_samples_tex_itr, d_histogram, num_samples);
     *
     * // Unbind texture iterator
     * d_samples_tex_itr.UnbindTexture();
     *
     * // d_histogram   <-- [2, 1, 3, 1, 0, 1, 2, 2]
     *
     * \endcode
     *
     * \tparam BINS                 Number of histogram bins per channel
     * \tparam InputIterator        <b>[inferred]</b> Random-access input iterator type for reading input samples. (Must have an InputIterator::value_type that, when cast as an integer, falls in the range [0..BINS-1])  \iterator
     * \tparam HistoCounter         <b>[inferred]</b> Integer type for counting sample occurrences per histogram bin
     */
    template <
        int                 BINS,
        typename            InputIterator,
        typename            HistoCounter>
    __host__ __device__
    static cudaError_t SingleChannelGlobalAtomic(
        void                *d_temp_storage,                    ///< [in] %Device allocation of temporary storage.  When NULL, the required allocation size is written to \p temp_storage_bytes and no work is done.
        size_t              &temp_storage_bytes,                ///< [in,out] Reference to size in bytes of \p d_temp_storage allocation
        InputIterator       d_samples,                          ///< [in] Input samples
        HistoCounter*       d_histogram,                        ///< [out] Array of BINS counters of integral type \p HistoCounter.
        int                 num_samples,                        ///< [in] Number of samples to process
        cudaStream_t        stream              = 0,            ///< [in] <b>[optional]</b> CUDA stream to launch kernels within.  Default is stream<sub>0</sub>.
        bool                debug_synchronous   = false)        ///< [in] <b>[optional]</b> Whether or not to synchronize the stream after every kernel launch to check for errors.  May cause significant slowdown.  Default is \p false.
    {
        // Signed integer type for global offsets
        typedef int Offset;

        // Dispatch type
        typedef DeviceHistogramDispatch<
                DEVICE_HISTO_GLOBAL_ATOMIC,
                BINS,
                1,
                1,
                InputIterator,
                HistoCounter,
                Offset>
            DeviceHistogramDispatch;

        return DeviceHistogramDispatch::Dispatch(
            d_temp_storage,
            temp_storage_bytes,
            d_samples,
            &d_histogram,
            num_samples,
            stream,
            debug_synchronous);
    }


    //@}  end member group
    /******************************************************************//**
     * \name Interleaved multi-channel samples
     *********************************************************************/
    //@{


    /**
     * \brief Computes a device-wide histogram from multi-channel data using fast block-sorting.
     *
     * \par
     * - The total number of samples across all channels (\p num_samples) must be a whole multiple of \p CHANNELS.
     * - Delivers consistent throughput regardless of sample diversity
     * - Histograms having a large number of bins (e.g., thousands) may adversely affect shared memory occupancy and performance (or even the ability to launch).
     * - Performance is often improved when referencing input samples through a texture-caching iterator (e.g., cub::TexObjInputIterator).
     * - \devicestorage
     * - \cdp
     *
     * \par Snippet
     * The code snippet below illustrates the computation of three 256-bin histograms from
     * an input sequence of quad-channel (interleaved) <tt>unsigned char</tt> samples.
     * (E.g., RGB histograms from RGBA pixel samples.)
     *
     * \par
     * \code
     * #include <cub/cub.cuh>   // or equivalently <cub/device/device_histogram.cuh>
     *
     * // Declare, allocate, and initialize device pointers for input and histograms
     * int           num_samples;     // e.g., 20 (five pixels with four channels each)
     * unsigned char *d_samples;      // e.g., [(2, 6, 7, 5), (3, 0, 2, 1), (7, 0, 6, 2),
     *                                //        (0, 6, 7, 5), (3, 0, 2, 6)]
     * unsigned int  *d_histogram[3]; // e.g., [ [ ,  ,  ,  ,  ,  ,  ,  ];
     *                                //         [ ,  ,  ,  ,  ,  ,  ,  ];
     *                                //         [ ,  ,  ,  ,  ,  ,  ,  ] ]
     * ...
     *
     * // Wrap d_samples device pointer in a random-access texture iterator
     * cub::TexObjInputIterator<unsigned char> d_samples_tex_itr;
     * d_samples_tex_itr.BindTexture(d_samples, num_samples * sizeof(unsigned char));
     *
     * // Determine temporary device storage requirements
     * void     *d_temp_storage = NULL;
     * size_t   temp_storage_bytes = 0;
     * cub::DeviceHistogram::MultiChannelSorting<8, 4, 3>(d_temp_storage, temp_storage_bytes, d_samples_tex_itr, d_histograms, num_samples);
     *
     * // Allocate temporary storage
     * cudaMalloc(&d_temp_storage, temp_storage_bytes);
     *
     * // Compute histograms
     * cub::DeviceHistogram::MultiChannelSorting<8, 4, 3>(d_temp_storage, temp_storage_bytes, d_samples_tex_itr, d_histograms, num_samples);
     *
     * // Unbind texture iterator
     * d_samples_tex_itr.UnbindTexture();
     *
     * // d_histogram   <-- [ [1, 0, 1, 2, 0, 0, 0, 1];
     * //                     [0, 3, 0, 0, 0, 0, 2, 0];
     * //                     [0, 0, 2, 0, 0, 0, 1, 2] ]
     *
     * \endcode
     *
     * \tparam BINS                 Number of histogram bins per channel
     * \tparam CHANNELS             Number of channels interleaved in the input data (may be greater than the number of channels being actively histogrammed)
     * \tparam ACTIVE_CHANNELS      <b>[inferred]</b> Number of channels actively being histogrammed
     * \tparam InputIterator        <b>[inferred]</b> Random-access input iterator type for reading input samples. (Must have an InputIterator::value_type that, when cast as an integer, falls in the range [0..BINS-1])  \iterator
     * \tparam HistoCounter         <b>[inferred]</b> Integer type for counting sample occurrences per histogram bin
     */
    template <
        int                 BINS,
        int                 CHANNELS,
        int                 ACTIVE_CHANNELS,
        typename            InputIterator,
        typename            HistoCounter>
    __host__ __device__
    static cudaError_t MultiChannelSorting(
        void                *d_temp_storage,                    ///< [in] %Device allocation of temporary storage.  When NULL, the required allocation size is written to \p temp_storage_bytes and no work is done.
        size_t              &temp_storage_bytes,                ///< [in,out] Reference to size in bytes of \p d_temp_storage allocation
        InputIterator       d_samples,                          ///< [in] Pointer to the input sequence of sample items. The samples from different channels are assumed to be interleaved (e.g., an array of 32b pixels where each pixel consists of four RGBA 8b samples).
        HistoCounter        *d_histograms[ACTIVE_CHANNELS],     ///< [out] Array of active channel histogram pointers, each pointing to an output array having BINS counters of integral type \p HistoCounter.
        int                 num_samples,                        ///< [in] Total number of samples to process in all channels, including non-active channels
        cudaStream_t        stream              = 0,            ///< [in] <b>[optional]</b> CUDA stream to launch kernels within.  Default is stream<sub>0</sub>.
        bool                debug_synchronous   = false)        ///< [in] <b>[optional]</b> Whether or not to synchronize the stream after every kernel launch to check for errors.  May cause significant slowdown.  Default is \p false.
    {
        // Signed integer type for global offsets
        typedef int Offset;

        // Dispatch type
        typedef DeviceHistogramDispatch<
            DEVICE_HISTO_SORT,
            BINS,
            CHANNELS,
            ACTIVE_CHANNELS,
            InputIterator,
            HistoCounter,
            Offset> DeviceHistogramDispatch;

        return DeviceHistogramDispatch::Dispatch(
            d_temp_storage,
            temp_storage_bytes,
            d_samples,
            d_histograms,
            num_samples,
            stream,
            debug_synchronous);
    }


    /**
     * \brief Computes a device-wide histogram from multi-channel data using shared-memory atomic read-modify-write operations.
     *
     * \par
     * - The total number of samples across all channels (\p num_samples) must be a whole multiple of \p CHANNELS.
     * - Input samples having lower diversity can cause performance to be degraded due to serializations from bin-collisions.
     * - Histograms having a large number of bins (e.g., thousands) may adversely affect shared memory occupancy and performance (or even the ability to launch).
     * - Performance is often improved when referencing input samples through a texture-caching iterator (e.g., cub::TexObjInputIterator).
     * - \devicestorage
     * - \cdp
     *
     * \par Snippet
     * The code snippet below illustrates the computation of three 256-bin histograms from
     * an input sequence of quad-channel (interleaved) <tt>unsigned char</tt> samples.
     * (E.g., RGB histograms from RGBA pixel samples.)
     * \par
     * \code
     * #include <cub/cub.cuh>   // or equivalently <cub/device/device_histogram.cuh>
     *
     * // Declare, allocate, and initialize device pointers for input and histograms
     * int           num_samples;     // e.g., 20 (five pixels with four channels each)
     * unsigned char *d_samples;      // e.g., [(2, 6, 7, 5), (3, 0, 2, 1), (7, 0, 6, 2),
     *                                //        (0, 6, 7, 5), (3, 0, 2, 6)]
     * unsigned int  *d_histogram[3]; // e.g., [ [ ,  ,  ,  ,  ,  ,  ,  ];
     *                                //         [ ,  ,  ,  ,  ,  ,  ,  ];
     *                                //         [ ,  ,  ,  ,  ,  ,  ,  ] ]
     * ...
     *
     * // Wrap d_samples device pointer in a random-access texture iterator
     * cub::TexObjInputIterator<unsigned char> d_samples_tex_itr;
     * d_samples_tex_itr.BindTexture(d_samples, num_samples * sizeof(unsigned char));
     *
     * // Determine temporary device storage requirements
     * void *d_temp_storage = NULL;
     * size_t temp_storage_bytes = 0;
     * cub::DeviceHistogram::MultiChannelSharedAtomic<8, 4, 3>(d_temp_storage, temp_storage_bytes, d_samples_tex_itr, d_histograms, num_samples);
     *
     * // Allocate temporary storage
     * cudaMalloc(&d_temp_storage, temp_storage_bytes);
     *
     * // Compute histograms
     * cub::DeviceHistogram::MultiChannelSharedAtomic<8, 4, 3>(d_temp_storage, temp_storage_bytes, d_samples_tex_itr, d_histograms, num_samples);
     *
     * // Unbind texture iterator
     * d_samples_tex_itr.UnbindTexture();
     *
     * // d_histogram   <-- [ [1, 0, 1, 2, 0, 0, 0, 1];
     * //                     [0, 3, 0, 0, 0, 0, 2, 0];
     * //                     [0, 0, 2, 0, 0, 0, 1, 2] ]
     *
     * \endcode
     *
     * \tparam BINS                 Number of histogram bins per channel
     * \tparam CHANNELS             Number of channels interleaved in the input data (may be greater than the number of channels being actively histogrammed)
     * \tparam ACTIVE_CHANNELS      <b>[inferred]</b> Number of channels actively being histogrammed
     * \tparam InputIterator        <b>[inferred]</b> Random-access input iterator type for reading input samples. (Must have an InputIterator::value_type that, when cast as an integer, falls in the range [0..BINS-1])  \iterator
     * \tparam HistoCounter         <b>[inferred]</b> Integer type for counting sample occurrences per histogram bin
     */
    template <
        int                 BINS,
        int                 CHANNELS,
        int                 ACTIVE_CHANNELS,
        typename            InputIterator,
        typename            HistoCounter>
    __host__ __device__
    static cudaError_t MultiChannelSharedAtomic(
        void                *d_temp_storage,                    ///< [in] %Device allocation of temporary storage.  When NULL, the required allocation size is written to \p temp_storage_bytes and no work is done.
        size_t              &temp_storage_bytes,                ///< [in,out] Reference to size in bytes of \p d_temp_storage allocation
        InputIterator       d_samples,                          ///< [in] Pointer to the input sequence of sample items. The samples from different channels are assumed to be interleaved (e.g., an array of 32b pixels where each pixel consists of four RGBA 8b samples).
        HistoCounter        *d_histograms[ACTIVE_CHANNELS],     ///< [out] Array of active channel histogram pointers, each pointing to an output array having BINS counters of integral type \p HistoCounter.
        int                 num_samples,                        ///< [in] Total number of samples to process in all channels, including non-active channels
        cudaStream_t        stream              = 0,            ///< [in] <b>[optional]</b> CUDA stream to launch kernels within.  Default is stream<sub>0</sub>.
        bool                debug_synchronous   = false)        ///< [in] <b>[optional]</b> Whether or not to synchronize the stream after every kernel launch to check for errors.  May cause significant slowdown.  Default is \p false.
    {
        // Signed integer type for global offsets
        typedef int Offset;

        // Dispatch type
        typedef DeviceHistogramDispatch<
            DEVICE_HISTO_SHARED_ATOMIC,
            BINS,
            CHANNELS,
            ACTIVE_CHANNELS,
            InputIterator,
            HistoCounter,
            Offset> DeviceHistogramDispatch;

        return DeviceHistogramDispatch::Dispatch(
            d_temp_storage,
            temp_storage_bytes,
            d_samples,
            d_histograms,
            num_samples,
            stream,
            debug_synchronous);
    }


    /**
     * \brief Computes a device-wide histogram from multi-channel data using global-memory atomic read-modify-write operations.
     *
     * \par
     * - The total number of samples across all channels (\p num_samples) must be a whole multiple of \p CHANNELS.
     * - Input samples having lower diversity can cause performance to be degraded due to serializations from bin-collisions.
     * - Performance is not significantly impacted when computing histograms having large numbers of bins (e.g., thousands).
     * - Performance is often improved when referencing input samples through a texture-caching iterator (e.g., cub::TexObjInputIterator).
     * - \devicestorage
     * - \cdp
     *
     * \par Snippet
     * The code snippet below illustrates the computation of three 256-bin histograms from
     * an input sequence of quad-channel (interleaved) <tt>unsigned char</tt> samples.
     * (E.g., RGB histograms from RGBA pixel samples.)
     *
     * \par
     * \code
     * #include <cub/cub.cuh>   // or equivalently <cub/device/device_histogram.cuh>
     *
     * // Declare, allocate, and initialize device pointers for input and histograms
     * int           num_samples;     // e.g., 20 (five pixels with four channels each)
     * unsigned char *d_samples;      // e.g., [(2, 6, 7, 5), (3, 0, 2, 1), (7, 0, 6, 2),
     *                                //        (0, 6, 7, 5), (3, 0, 2, 6)]
     * unsigned int  *d_histogram[3]; // e.g., [ [ ,  ,  ,  ,  ,  ,  ,  ];
     *                                //         [ ,  ,  ,  ,  ,  ,  ,  ];
     *                                //         [ ,  ,  ,  ,  ,  ,  ,  ] ]
     * ...
     *
     * // Wrap d_samples device pointer in a random-access texture iterator
     * cub::TexObjInputIterator<unsigned char> d_samples_tex_itr;
     * d_samples_tex_itr.BindTexture(d_samples, num_samples * sizeof(unsigned char));
     *
     * // Determine temporary device storage requirements
     * void     *d_temp_storage = NULL;
     * size_t   temp_storage_bytes = 0;
     * cub::DeviceHistogram::MultiChannelGlobalAtomic<8, 4, 3>(d_temp_storage, temp_storage_bytes, d_samples_tex_itr, d_histograms, num_samples);
     *
     * // Allocate temporary storage
     * cudaMalloc(&d_temp_storage, temp_storage_bytes);
     *
     * // Compute histograms
     * cub::DeviceHistogram::MultiChannelGlobalAtomic<8, 4, 3>(d_temp_storage, temp_storage_bytes, d_samples_tex_itr, d_histograms, num_samples);
     *
     * // Unbind texture iterator
     * d_samples_tex_itr.UnbindTexture();
     *
     * // d_histogram   <-- [ [1, 0, 1, 2, 0, 0, 0, 1];
     * //                     [0, 3, 0, 0, 0, 0, 2, 0];
     * //                     [0, 0, 2, 0, 0, 0, 1, 2] ]
     *
     * \endcode
     *
     * \tparam BINS                 Number of histogram bins per channel
     * \tparam CHANNELS             Number of channels interleaved in the input data (may be greater than the number of channels being actively histogrammed)
     * \tparam ACTIVE_CHANNELS      <b>[inferred]</b> Number of channels actively being histogrammed
     * \tparam InputIterator        <b>[inferred]</b> Random-access input iterator type for reading input samples. (Must have an InputIterator::value_type that, when cast as an integer, falls in the range [0..BINS-1])  \iterator
     * \tparam HistoCounter         <b>[inferred]</b> Integer type for counting sample occurrences per histogram bin
     */
    template <
        int                 BINS,
        int                 CHANNELS,
        int                 ACTIVE_CHANNELS,
        typename            InputIterator,
        typename            HistoCounter>
    __host__ __device__
    static cudaError_t MultiChannelGlobalAtomic(
        void                *d_temp_storage,                    ///< [in] %Device allocation of temporary storage.  When NULL, the required allocation size is written to \p temp_storage_bytes and no work is done.
        size_t              &temp_storage_bytes,                ///< [in,out] Reference to size in bytes of \p d_temp_storage allocation
        InputIterator       d_samples,                          ///< [in] Pointer to the input sequence of sample items. The samples from different channels are assumed to be interleaved (e.g., an array of 32b pixels where each pixel consists of four RGBA 8b samples).
        HistoCounter        *d_histograms[ACTIVE_CHANNELS],     ///< [out] Array of active channel histogram pointers, each pointing to an output array having BINS counters of integral type \p HistoCounter.
        int                 num_samples,                        ///< [in] Total number of samples to process in all channels, including non-active channels
        cudaStream_t        stream              = 0,            ///< [in] <b>[optional]</b> CUDA stream to launch kernels within.  Default is stream<sub>0</sub>.
        bool                debug_synchronous   = false)        ///< [in] <b>[optional]</b> Whether or not to synchronize the stream after every kernel launch to check for errors.  May cause significant slowdown.  Default is \p false.
    {
        // Signed integer type for global offsets
        typedef int Offset;

        // Dispatch type
        typedef DeviceHistogramDispatch<
                DEVICE_HISTO_GLOBAL_ATOMIC,
                BINS,
                CHANNELS,
                ACTIVE_CHANNELS,
                InputIterator,
                HistoCounter,
                Offset>
            DeviceHistogramDispatch;

        return DeviceHistogramDispatch::Dispatch(
            d_temp_storage,
            temp_storage_bytes,
            d_samples,
            d_histograms,
            num_samples,
            stream,
            debug_synchronous);
    }

    //@}  end member group

};

/**
 * \example example_device_histogram.cu
 */

}               // CUB namespace
CUB_NS_POSTFIX  // Optional outer namespace(s)


