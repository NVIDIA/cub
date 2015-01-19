
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
#include <limits>

#include "../../block_sweep/block_histogram_sweep.cuh"
#include "../device_radix_sort.cuh"
#include "../../iterator/tex_ref_input_iterator.cuh"
#include "../../util_debug.cuh"
#include "../../util_device.cuh"
#include "../../grid/grid_queue.cuh"
#include "../../thread/thread_search.cuh"
#include "../../util_namespace.cuh"

/// Optional outer namespace(s)
CUB_NS_PREFIX

/// CUB namespace
namespace cub {



/******************************************************************************
 * Histogram kernel entry points
 *****************************************************************************/


/**
 * Histogram privatized sweep kernel entry point (multi-block).  Computes privatized histograms, one per thread block.
 */
template <
    typename                                            BlockHistogramSweepPolicyT,     ///< Parameterized BlockHistogramSweepPolicy tuning policy type
    int                                                 MAX_SMEM_BINS,                  ///< Maximum number of histogram bins per channel (e.g., up to 256)
    int                                                 NUM_CHANNELS,                   ///< Number of channels interleaved in the input data (may be greater than the number of channels being actively histogrammed)
    int                                                 NUM_ACTIVE_CHANNELS,            ///< Number of channels actively being histogrammed
    typename                                            SampleIteratorT,                ///< The input iterator type. \iterator.
    typename                                            CounterT,                       ///< Integer type for counting sample occurrences per histogram bin
    typename                                            SampleTransformOpT,             ///< Transform operator type for determining privatized bins from samples for each channel
    typename                                            OffsetT>                        ///< Signed integer type for global offsets
__launch_bounds__ (int(BlockHistogramSweepPolicyT::BLOCK_THREADS))
__global__ void DeviceHistogramSweepKernel(
    SampleIteratorT                                         d_samples,                  ///< [in] Array of sample data. The samples from different channels are assumed to be interleaved (e.g., an array of 32b pixels where each pixel consists of four RGBA 8b samples).
    ArrayWrapper<CounterT*, NUM_ACTIVE_CHANNELS>            d_privatized_histo_wrapper, ///< [out] Histogram counter data having logical dimensions <tt>CounterT[NUM_ACTIVE_CHANNELS][gridDim.x][MAX_SMEM_BINS]</tt>
    ArrayWrapper<SampleTransformOpT, NUM_ACTIVE_CHANNELS>   transform_op_wrapper,       ///< [in] Transform operators for determining privatized bin-ids from samples, one for each channel
    ArrayWrapper<int, NUM_ACTIVE_CHANNELS>                  num_bins_wrapper,           ///< [in] The number of bin level boundaries for delineating histogram samples in each active channel.  Implies that the number of bins for channel<sub><em>i</em></sub> is <tt>num_output_levels[i]</tt> - 1.
    ArrayWrapper<CounterT*, NUM_ACTIVE_CHANNELS>            d_out_histo_wrapper,        ///< [out] Histogram counter data having logical dimensions <tt>CounterT[NUM_ACTIVE_CHANNELS][num_bins.array[CHANNEL]]</tt>
    OffsetT                                                 num_row_pixels,             ///< [in] The number of multi-channel pixels per row in the region of interest
    OffsetT                                                 num_rows,                   ///< [in] The number of rows in the region of interest
    OffsetT                                                 row_stride_samples)         ///< [in] The number of samples between starts of consecutive rows in the region of interest
//    GridQueue<OffsetT>                                      queue)                      ///< [in] Drain queue descriptor for dynamically mapping tile data onto thread blocks
{
    // Thread block type for compositing input tiles
    typedef BlockHistogramSweep<BlockHistogramSweepPolicyT, MAX_SMEM_BINS, NUM_CHANNELS, NUM_ACTIVE_CHANNELS, SampleIteratorT, CounterT, SampleTransformOpT, OffsetT> BlockHistogramSweepT;

    // Shared memory for BlockHistogramSweep
    __shared__ typename BlockHistogramSweepT::TempStorage temp_storage;

    BlockHistogramSweepT block_sweep(
        temp_storage,
        d_samples,
        row_stride_samples,
        d_out_histo_wrapper.array,
        transform_op_wrapper.array,
        num_bins_wrapper.array);

    // Initialize counters
    block_sweep.InitBinCounters();
/*
    // Consume input tiles
    for (OffsetT row = blockIdx.y; row < num_rows; row += gridDim.y)
    {
        OffsetT row_offset     = row * row_stride_samples;
        OffsetT row_end        = row_offset + (num_row_pixels * NUM_CHANNELS);

        block_sweep.ConsumeStriped(row_offset, row_end);
    }
*/
    block_sweep.ConsumeStriped(0, num_row_pixels * NUM_CHANNELS);
//    block_sweep.ConsumeStriped(queue, num_row_pixels);


    // Store output to global (if necessary)
    block_sweep.StoreOutput();
}


/**
 * Histogram aggregation kernel entry point.  Aggregates privatized threadblock histograms from a previous multi-block histogram pass.
 */
template <
    typename                                                BlockHistogramSweepPolicyT,     ///< Parameterized BlockHistogramSweepPolicy tuning policy type
    int                                                     NUM_ACTIVE_CHANNELS,            ///< Number of channels actively being histogrammed
    typename                                                CounterT,                       ///< Integer type for counting sample occurrences per histogram bin
    typename                                                SampleTransformOpT,             ///< Transform operator type for determining final bin-ids from privatized bin-ids for each channel
    typename                                                SampleT,                        ///< The sample type for the privatized histogram kernel
    bool                                                    SKIP_BIN_CONVERSION>            ///< Whether to skip the transformation of privatized bins to output bins
__global__ void DeviceHistogramAggregateKernel(
    ArrayWrapper<int, NUM_ACTIVE_CHANNELS>                  num_privatized_bins_wrapper,    ///< [in] Number of privatized histogram bins per channel
    ArrayWrapper<CounterT*, NUM_ACTIVE_CHANNELS>            d_privatized_histo_wrapper,     ///< [out] Histogram counter data having logical dimensions <tt>CounterT[NUM_ACTIVE_CHANNELS][gridDim.x][MAX_SMEM_BINS]</tt>
    ArrayWrapper<int, NUM_ACTIVE_CHANNELS>                  num_output_bins_wrapper,        ///< [in] Number of output histogram bins per channel
    ArrayWrapper<CounterT*, NUM_ACTIVE_CHANNELS>            d_out_histo_wrapper,            ///< [out] Histogram counter data having logical dimensions <tt>CounterT[NUM_ACTIVE_CHANNELS][num_bins.array[CHANNEL]]</tt>
    int                                                     num_sweep_grid_blocks,          ///< [in] Number of threadblock histograms per channel in \p d_block_histograms
    ArrayWrapper<SampleTransformOpT, NUM_ACTIVE_CHANNELS>   transform_op_wrapper)           ///< [in] Transform operators for determining final bin-ids from privatized bin-ids, one for each channel
{
    int             privatized_bin = (blockIdx.x * blockDim.x) + threadIdx.x;
    CounterT        bin_aggregate[NUM_ACTIVE_CHANNELS];

    #pragma unroll
    for (int CHANNEL = 0; CHANNEL < NUM_ACTIVE_CHANNELS; ++CHANNEL)
    {
        bin_aggregate[CHANNEL] = 0;
    }

    // Read and accumulate the privatized histogram from each block in the sweep kernel
    #pragma unroll
    for (int CHANNEL = 0; CHANNEL < NUM_ACTIVE_CHANNELS; ++CHANNEL)
    {
        #pragma unroll
        for (int block = 0; block < num_sweep_grid_blocks; ++block)
        {
            int block_offset = block * num_privatized_bins_wrapper.array[CHANNEL];
            if (privatized_bin < num_privatized_bins_wrapper.array[CHANNEL])
            {
                bin_aggregate[CHANNEL] += d_privatized_histo_wrapper.array[CHANNEL][block_offset + privatized_bin];
            }
        }
    }

    // Output to final bins
    #pragma unroll
    for (int CHANNEL = 0; CHANNEL < NUM_ACTIVE_CHANNELS; ++CHANNEL)
    {
        if (SKIP_BIN_CONVERSION)
        {
            // Simply output the bin counter (privatized bins match output bins)
            if (privatized_bin < num_output_bins_wrapper.array[CHANNEL])
                d_out_histo_wrapper.array[CHANNEL][privatized_bin] = bin_aggregate[CHANNEL];
        }
        else
        {
            // Get output bin for this privatized bin
            bool valid_sample = privatized_bin < num_privatized_bins_wrapper.array[CHANNEL];
            int output_bin;
            transform_op_wrapper.array[CHANNEL].BinSelect<BlockHistogramSweepPolicyT::LOAD_MODIFIER>((SampleT) ((unsigned char) privatized_bin), output_bin, valid_sample);

            // Update global histogram
            if (valid_sample)
                atomicAdd(d_out_histo_wrapper.array[CHANNEL] + output_bin, bin_aggregate[CHANNEL]);
        }
    }
}


/**
 * Histogram initialization kernel entry point.  Only necessary when atomics are used in the aggregation kernel
 */
template <
    int         NUM_ACTIVE_CHANNELS,
    typename    CounterT,
    typename    OffsetT>
__global__ void DeviceHistogramInitKernel(
    ArrayWrapper<int, NUM_ACTIVE_CHANNELS>                  num_output_bins_wrapper,        ///< [in] Number of output histogram bins per channel
    ArrayWrapper<CounterT*, NUM_ACTIVE_CHANNELS>            d_out_histo_wrapper)            ///< [out] Histogram counter data having logical dimensions <tt>CounterT[NUM_ACTIVE_CHANNELS][num_bins.array[CHANNEL]]</tt>
//    GridQueue<OffsetT>                                      grid_queue,
//    OffsetT                                                 num_pixels)
{
/*
    if ((threadIdx.x == 0) && (blockIdx.x == 0))
    {
        grid_queue.FillAndResetDrain(num_pixels);
    }
*/

    int output_bin = (blockIdx.x * blockDim.x) + threadIdx.x;
    #pragma unroll
    for (int CHANNEL = 0; CHANNEL < NUM_ACTIVE_CHANNELS; ++CHANNEL)
    {
        if (output_bin < num_output_bins_wrapper.array[CHANNEL])
            d_out_histo_wrapper.array[CHANNEL][output_bin] = 0;
    }
}



/******************************************************************************
 * Dispatch
 ******************************************************************************/

/**
 * Utility class for dispatching the appropriately-tuned kernels for DeviceHistogram
 */
template <
    int         NUM_CHANNELS,               ///< Number of channels interleaved in the input data (may be greater than the number of channels being actively histogrammed)
    int         NUM_ACTIVE_CHANNELS,        ///< Number of channels actively being histogrammed
    typename    SampleIteratorT,            ///< Random-access input iterator type for reading input items \iterator
    typename    CounterT,                   ///< Integer type for counting sample occurrences per histogram bin
    typename    LevelT,                     ///< Type for specifying bin level boundaries
    typename    OffsetT>                    ///< Signed integer type for global offsets
struct DeviceHistogramDispatch
{
    /******************************************************************************
     * Types and constants
     ******************************************************************************/

    /// The sample value type of the input iterator
    typedef typename std::iterator_traits<SampleIteratorT>::value_type SampleT;

    enum
    {
        // Maximum number of bins for which we will use a privatized strategy
        MAX_SMEM_BINS = 256
    };


    /******************************************************************************
     * Transform functors for converting samples to bin-ids
     ******************************************************************************/

    // Searches for bin given a list of bin-boundary levels
    template <typename LevelIteratorT>
    struct SearchTransform
    {
        LevelIteratorT  d_levels;                   // Pointer to levels array
        int             num_output_levels;          // Number of levels in array

        // Initializer
        __host__ __device__ __forceinline__ void Init(
            LevelIteratorT  d_levels,               // Pointer to levels array
            int             num_output_levels)      // Number of levels in array
        {
            this->d_levels          = d_levels;
            this->num_output_levels = num_output_levels;
        }

        // Method for converting samples to bin-ids
        template <CacheLoadModifier LOAD_MODIFIER, typename _SampleT>
        __host__ __device__ __forceinline__ void BinSelect(_SampleT sample, int &bin, bool &valid)
        {
            /// Level iterator wrapper type
            typedef typename If<IsPointer<LevelIteratorT>::VALUE,
                    CacheModifiedInputIterator<LOAD_MODIFIER, LevelT, OffsetT>,     // Wrap the native input pointer with CacheModifiedInputIterator
                    LevelIteratorT>::Type                                           // Directly use the supplied input iterator type
                WrappedLevelIteratorT;

            WrappedLevelIteratorT wrapped_levels(d_levels);

            int num_bins = num_output_levels - 1;
            if (valid)
            {
                bin = UpperBound(wrapped_levels, num_output_levels, (LevelT) sample) - 1;
                valid = (((unsigned int) bin) < num_bins);
            }
        }
    };


    // Scales samples to evenly-spaced bins
    struct ScaleTransform
    {
        int    num_bins;    // Number of levels in array
        LevelT max;         // Max sample level (exclusive)
        LevelT min;         // Min sample level (inclusive)
        LevelT scale;       // Bin scaling factor

        // Initializer
        template <typename _LevelT>
        __host__ __device__ __forceinline__ void Init(
            int     num_output_levels,  // Number of levels in array
            _LevelT max,         // Max sample level (exclusive)
            _LevelT min,         // Min sample level (inclusive)
            _LevelT scale)       // Bin scaling factor
        {
            this->num_bins = num_output_levels - 1;
            this->max = max;
            this->min = min;
            this->scale = scale;
        }

        // Initializer (float specialization)
        __host__ __device__ __forceinline__ void Init(
            int    num_output_levels,  // Number of levels in array
            float   max,         // Max sample level (exclusive)
            float   min,         // Min sample level (inclusive)
            float   scale)       // Bin scaling factor
        {
            this->num_bins = num_output_levels - 1;
            this->max = max;
            this->min = min;
            this->scale = 1.0 / scale;
        }

        // Initializer (double specialization)
        __host__ __device__ __forceinline__ void Init(
            int    num_output_levels,  // Number of levels in array
            double max,         // Max sample level (exclusive)
            double min,         // Min sample level (inclusive)
            double scale)       // Bin scaling factor
        {
            this->num_bins = num_output_levels - 1;
            this->max = max;
            this->min = min;
            this->scale = 1.0 / scale;
        }

        // Method for converting samples to bin-ids
        template <CacheLoadModifier LOAD_MODIFIER, typename _SampleT>
        __host__ __device__ __forceinline__ void BinSelect(_SampleT sample, int &bin, bool valid)
        {
            bin = valid && (sample >= min) && (sample < max) ?
                (int) ((((LevelT) sample) - min) / scale) :
                -1;
        }

        // Method for converting samples to bin-ids (float specialization)
        template <CacheLoadModifier LOAD_MODIFIER>
        __host__ __device__ __forceinline__ void BinSelect(float sample, int &bin, bool &valid)
        {
            bin = valid && (sample >= min) && (sample < max) ?
                (int) ((((LevelT) sample) - min) * scale) :
                -1;
        }

        // Method for converting samples to bin-ids (double specialization)
        template <CacheLoadModifier LOAD_MODIFIER>
        __host__ __device__ __forceinline__ void BinSelect(double sample, int &bin, bool &valid)
        {
            bin = valid && (sample >= min) && (sample < max) ?
                (int) ((((LevelT) sample) - min) * scale) :
                -1;
        }
    };


    // Pass-through bin transform operator
    struct PassThruTransform
    {
        // Method for converting samples to bin-ids
        template <CacheLoadModifier LOAD_MODIFIER, typename _SampleT>
        __host__ __device__ __forceinline__ void BinSelect(_SampleT sample, int &bin, bool &valid)
        {
            bin = (int) ((unsigned char) sample);
        }
    };



    /******************************************************************************
     * Tuning policies
     ******************************************************************************/

    enum {
        T_SCALE = CUB_MAX(1, (sizeof(SampleT) / 2))
    };

    /// SM52
    struct Policy520
    {
        // HistogramSweepPolicy
        typedef BlockHistogramSweepPolicy<
                256,
                8,
                LOAD_LDG,
                false>
            HistogramSweepPolicy;
    };

    /// SM35
    struct Policy350
    {
        // HistogramSweepPolicy
        typedef BlockHistogramSweepPolicy<
//                256,
//                CUB_MAX((8 / NUM_ACTIVE_CHANNELS / T_SCALE), 1),    // 20 8b samples per thread
                192,
                10,
                LOAD_LDG,
                false>
            HistogramSweepPolicy;
    };

    /// SM30
    struct Policy300
    {
        // HistogramSweepPolicy
        typedef BlockHistogramSweepPolicy<
                96,
                CUB_MAX((20 / NUM_ACTIVE_CHANNELS / T_SCALE), 1),    // 20 8b samples per thread
                LOAD_DEFAULT,
                true>
            HistogramSweepPolicy;
    };

    /// SM20
    struct Policy200
    {
        // HistogramSweepPolicy
        typedef BlockHistogramSweepPolicy<
                128,
                CUB_MAX((12 / NUM_ACTIVE_CHANNELS / T_SCALE), 1),    // 20 8b samples per thread
                LOAD_DEFAULT,
                true>
            HistogramSweepPolicy;
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

#else
    typedef Policy200 PtxPolicy;

#endif

    // "Opaque" policies (whose parameterizations aren't reflected in the type signature)
    struct PtxHistogramSweepPolicy : PtxPolicy::HistogramSweepPolicy {};


    /******************************************************************************
     * Utilities
     ******************************************************************************/


    /**
     * Initialize kernel dispatch configurations with the policies corresponding to the PTX assembly we will use
     */
    template <typename KernelConfig>
    CUB_RUNTIME_FUNCTION __forceinline__
    static void InitConfigs(
        int             ptx_version,
        KernelConfig    &histogram_sweep_config)
    {
    #if (CUB_PTX_ARCH > 0)

        // We're on the device, so initialize the kernel dispatch configurations with the current PTX policy
        histogram_sweep_config.template Init<PtxHistogramSweepPolicy>();

    #else

        // We're on the host, so lookup and initialize the kernel dispatch configurations with the policies that match the device's PTX version
        if (ptx_version >= 520)
        {
            histogram_sweep_config.template Init<typename Policy520::HistogramSweepPolicy>();
        }
        else if (ptx_version >= 350)
        {
            histogram_sweep_config.template Init<typename Policy350::HistogramSweepPolicy>();
        }
        else if (ptx_version >= 300)
        {
            histogram_sweep_config.template Init<typename Policy300::HistogramSweepPolicy>();
        }
        else
        {
            histogram_sweep_config.template Init<typename Policy200::HistogramSweepPolicy>();
        }

    #endif
    }


    /**
     * Kernel kernel dispatch configuration
     */
    struct KernelConfig
    {
        int                             block_threads;
        int                             pixels_per_thread;

        template <typename BlockPolicy>
        CUB_RUNTIME_FUNCTION __forceinline__
        void Init()
        {
            block_threads               = BlockPolicy::BLOCK_THREADS;
            pixels_per_thread           = BlockPolicy::PIXELS_PER_THREAD;
        }

        CUB_RUNTIME_FUNCTION __forceinline__
        void Print()
        {
            printf("%d, %d", block_threads, pixels_per_thread);
        }

    };






    /******************************************************************************
     * Dispatch entrypoints
     ******************************************************************************/


    /**
     * Privatization-based dispatch routine
     */
    template <
        typename                            PrivatizedTransformOpT,                         ///< Transform operator type for determining bin-ids from samples for each channel
        typename                            AggregationTransformOpT,                        ///< Transform operator type for determining bin-ids from samples for each channel
        typename                            DeviceHistogramSweepKernelT,                    ///< Function type of cub::DeviceHistogramSweepKernel
        typename                            DeviceHistogramInitKernelT,                     ///< Function type of cub::DeviceHistogramInitKernel
        typename                            DeviceHistogramAggregateKernelT>                ///< Function type of cub::DeviceHistogramAggregateKernel
    CUB_RUNTIME_FUNCTION __forceinline__
    static cudaError_t PrivatizedDispatch(
        void                                *d_temp_storage,                                ///< [in] %Device allocation of temporary storage.  When NULL, the required allocation size is written to \p temp_storage_bytes and no work is done.
        size_t                              &temp_storage_bytes,                            ///< [in,out] Reference to size in bytes of \p d_temp_storage allocation
        SampleIteratorT                     d_samples,                                      ///< [in] The pointer to the input sequence of sample items. The samples from different channels are assumed to be interleaved (e.g., an array of 32-bit pixels where each pixel consists of four RGBA 8-bit samples).
        CounterT                            *d_histogram[NUM_ACTIVE_CHANNELS],              ///< [out] The pointers to the histogram counter output arrays, one for each active channel.  For channel<sub><em>i</em></sub>, the allocation length of <tt>d_histograms[i]</tt> should be <tt>num_output_levels[i]</tt> - 1.
        int                                 num_privatized_levels[NUM_ACTIVE_CHANNELS],     ///< [in] The number of bin level boundaries for delineating histogram samples in each active channel.  Implies that the number of bins for channel<sub><em>i</em></sub> is <tt>num_output_levels[i]</tt> - 1.
        PrivatizedTransformOpT              privatized_transform_op[NUM_ACTIVE_CHANNELS],   ///< [in] Transform operators for determining bin-ids from samples, one for each channel
        int                                 num_output_levels[NUM_ACTIVE_CHANNELS],         ///< [in] The number of bin level boundaries for delineating histogram samples in each active channel.  Implies that the number of bins for channel<sub><em>i</em></sub> is <tt>num_output_levels[i]</tt> - 1.
        AggregationTransformOpT             output_transform_op[NUM_ACTIVE_CHANNELS],       ///< [in] Transform operators for determining bin-ids from samples, one for each channel
        int                                 max_privatized_bins,                            ///< [in] Maximum number of privatized bins in any channel
        bool                                skip_bin_conversion,                            ///< [in] Whether to skip the transformation of privatized bins to output bins
        OffsetT                             num_row_pixels,                                 ///< [in] The number of multi-channel pixels per row in the region of interest
        OffsetT                             num_rows,                                       ///< [in] The number of rows in the region of interest
        OffsetT                             row_stride_samples,                             ///< [in] The number of samples between starts of consecutive rows in the region of interest
        DeviceHistogramSweepKernelT         histogram_sweep_kernel,                         ///< [in] Kernel function pointer to parameterization of cub::DeviceHistogramSweepKernel
        DeviceHistogramInitKernelT          histogram_init_kernel,                          ///< [in] Kernel function pointer to parameterization of cub::DeviceHistogramInitKernel
        DeviceHistogramAggregateKernelT     histogram_aggregate_kernel,                     ///< [in] Kernel function pointer to parameterization of cub::DeviceHistogramAggregateKernel
        KernelConfig                        histogram_sweep_config,                         ///< [in] Dispatch parameters that match the policy that \p histogram_sweep_kernel was compiled for
        cudaStream_t                        stream,                                         ///< [in] CUDA stream to launch kernels within.  Default is stream<sub>0</sub>.
        bool                                debug_synchronous)                              ///< [in] Whether or not to synchronize the stream after every kernel launch to check for errors.  May cause significant slowdown.  Default is \p false.
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

            // Get SM occupancy for histogram_sweep_kernel
            int histogram_sweep_sm_occupancy;
            if (CubDebug(error = MaxSmOccupancy(
                histogram_sweep_sm_occupancy,
                sm_version,
                histogram_sweep_kernel,
                histogram_sweep_config.block_threads))) break;

            // Get device occupancy for histogram_sweep_kernel
            int histogram_sweep_occupancy = histogram_sweep_sm_occupancy * sm_count;

            if (num_row_pixels * NUM_CHANNELS == row_stride_samples)
            {
                // Treat as a single linear array of samples
                num_row_pixels *= num_rows;
                num_rows = 1;
            }

            // Get grid dimensions, trying to keep total blocks ~histogram_sweep_occupancy
            int pixels_per_tile                 = histogram_sweep_config.block_threads * histogram_sweep_config.pixels_per_thread;
            int tiles_per_row                   = (num_row_pixels + pixels_per_tile - 1) / pixels_per_tile;
            int blocks_per_row                  = CUB_MIN(histogram_sweep_occupancy, tiles_per_row);
            int blocks_per_col                  = CUB_MIN(histogram_sweep_occupancy / blocks_per_row, num_rows);
            int num_sweep_grid_blocks           = blocks_per_row * blocks_per_col;

            dim3 sweep_grid_dims;
            sweep_grid_dims.x = (unsigned int) blocks_per_row;
            sweep_grid_dims.y = (unsigned int) blocks_per_col;
            sweep_grid_dims.z = 1;

            // Temporary storage allocation requirements
            const int ALLOCATIONS = NUM_ACTIVE_CHANNELS; //  + 1;
            void* allocations[ALLOCATIONS];
            size_t allocation_sizes[ALLOCATIONS];

            for (int CHANNEL = 0; CHANNEL < NUM_ACTIVE_CHANNELS; ++CHANNEL)
                allocation_sizes[CHANNEL] = num_sweep_grid_blocks * (num_privatized_levels[CHANNEL] - 1) * sizeof(CounterT);

//			allocation_sizes[ALLOCATIONS - 1] = GridQueue<int>::AllocationSize();

            // Alias the temporary allocations from the single storage blob (or compute the necessary size of the blob)
            if (CubDebug(error = AliasTemporaries(d_temp_storage, temp_storage_bytes, allocations, allocation_sizes))) break;
            if (d_temp_storage == NULL)
            {
                // Return if the caller is simply requesting the size of the storage allocation
                return cudaSuccess;
            }

            // Alias the allocation for the grid queue descriptor
//            GridQueue<OffsetT> grid_queue(allocations[ALLOCATIONS - 1]);

            // Setup array wrapper for histogram channel output (because we can't pass static arrays as kernel parameters)
            ArrayWrapper<CounterT*, NUM_ACTIVE_CHANNELS> d_out_histo_wrapper;
            for (int CHANNEL = 0; CHANNEL < NUM_ACTIVE_CHANNELS; ++CHANNEL)
                d_out_histo_wrapper.array[CHANNEL] = d_histogram[CHANNEL];

            // Setup array wrapper for privatized per-block histogram channel output (because we can't pass static arrays as kernel parameters)
            ArrayWrapper<CounterT*, NUM_ACTIVE_CHANNELS> d_privatized_histo_wrapper;
            for (int CHANNEL = 0; CHANNEL < NUM_ACTIVE_CHANNELS; ++CHANNEL)
                d_privatized_histo_wrapper.array[CHANNEL] = (CounterT*) allocations[CHANNEL];

            // Setup array wrapper for sweep bin transforms (because we can't pass static arrays as kernel parameters)
            ArrayWrapper<PrivatizedTransformOpT, NUM_ACTIVE_CHANNELS> privatized_transform_op_wrapper;
            for (int CHANNEL = 0; CHANNEL < NUM_ACTIVE_CHANNELS; ++CHANNEL)
                privatized_transform_op_wrapper.array[CHANNEL] = privatized_transform_op[CHANNEL];

            // Setup array wrapper for aggregation bin transforms (because we can't pass static arrays as kernel parameters)
            ArrayWrapper<AggregationTransformOpT, NUM_ACTIVE_CHANNELS> output_transform_op_wrapper;
            for (int CHANNEL = 0; CHANNEL < NUM_ACTIVE_CHANNELS; ++CHANNEL)
                output_transform_op_wrapper.array[CHANNEL] = output_transform_op[CHANNEL];

            // Setup array wrapper for num privatized bins (because we can't pass static arrays as kernel parameters)
            ArrayWrapper<int, NUM_ACTIVE_CHANNELS> num_privatized_bins_wrapper;
            for (int CHANNEL = 0; CHANNEL < NUM_ACTIVE_CHANNELS; ++CHANNEL)
                num_privatized_bins_wrapper.array[CHANNEL] = num_privatized_levels[CHANNEL] - 1;

            // Setup array wrapper for num output bins (because we can't pass static arrays as kernel parameters)
            ArrayWrapper<int, NUM_ACTIVE_CHANNELS> num_output_bins_wrapper;
            for (int CHANNEL = 0; CHANNEL < NUM_ACTIVE_CHANNELS; ++CHANNEL)
                num_output_bins_wrapper.array[CHANNEL] = num_output_levels[CHANNEL] - 1;

            int histogram_aggregate_block_threads   = MAX_SMEM_BINS;
            int histogram_aggregate_grid_dims       = (max_privatized_bins + histogram_aggregate_block_threads - 1) / histogram_aggregate_block_threads;           // number of blocks per histogram channel (one thread per counter)

            // Log DeviceHistogramInitKernel configuration
            if (debug_synchronous) CubLog("Invoking DeviceHistogramInitKernel<<<%d, %d, 0, %lld>>>()\n",
                histogram_aggregate_grid_dims, histogram_aggregate_block_threads, (long long) stream);

            histogram_init_kernel<<<histogram_aggregate_grid_dims, histogram_aggregate_block_threads, 0, stream>>>(
                num_output_bins_wrapper,
                d_out_histo_wrapper);
//                grid_queue,
//                num_row_pixels);

            // Log histogram_sweep_kernel configuration
            if (debug_synchronous) CubLog("Invoking histogram_sweep_kernel<<<{%d, %d, %d}, %d, 0, %lld>>>(), %d pixels per thread, %d SM occupancy\n",
                sweep_grid_dims.x, sweep_grid_dims.y, sweep_grid_dims.z,
                histogram_sweep_config.block_threads, (long long) stream, histogram_sweep_config.pixels_per_thread, histogram_sweep_sm_occupancy);

            // Invoke histogram_sweep_kernel
            histogram_sweep_kernel<<<sweep_grid_dims, histogram_sweep_config.block_threads, 0, stream>>>(
                d_samples,
                d_privatized_histo_wrapper,
                privatized_transform_op_wrapper,
                num_privatized_bins_wrapper,
                d_out_histo_wrapper,
                num_row_pixels,
                num_rows,
                row_stride_samples);
//                grid_queue);

            // Check for failure to launch
            if (CubDebug(error = cudaPeekAtLastError())) break;

            // Sync the stream if specified to flush runtime errors
            if (debug_synchronous && (CubDebug(error = SyncStream(stream)))) break;
/*
            // Log DeviceHistogramAggregateKernel configuration
            if (debug_synchronous) CubLog("Invoking DeviceHistogramAggregateKernel<<<%d, %d, 0, %lld>>>()\n",
                histogram_aggregate_grid_dims, histogram_aggregate_block_threads, (long long) stream);

            // Invoke kernel to aggregate the privatized histograms
            histogram_aggregate_kernel<<<histogram_aggregate_grid_dims, histogram_aggregate_block_threads, 0, stream>>>(
                num_privatized_bins_wrapper,
                d_privatized_histo_wrapper,
                num_output_bins_wrapper,
                d_out_histo_wrapper,
                num_sweep_grid_blocks,
                output_transform_op_wrapper);

            // Check for failure to launch
            if (CubDebug(error = cudaPeekAtLastError())) break;

            // Sync the stream if specified to flush runtime errors
            if (debug_synchronous && (CubDebug(error = SyncStream(stream)))) break;
*/
        }
        while (0);

        return error;

    #endif // CUB_RUNTIME_ENABLED
    }



    /**
     * Dispatch routine for HistogramRange, specialized for non-1-byte types
     */
    CUB_RUNTIME_FUNCTION
    static cudaError_t DispatchRange(
        void                *d_temp_storage,                        ///< [in] %Device allocation of temporary storage.  When NULL, the required allocation size is written to \p temp_storage_bytes and no work is done.
        size_t              &temp_storage_bytes,                    ///< [in,out] Reference to size in bytes of \p d_temp_storage allocation
        SampleIteratorT     d_samples,                              ///< [in] The pointer to the multi-channel input sequence of data samples. The samples from different channels are assumed to be interleaved (e.g., an array of 32-bit pixels where each pixel consists of four RGBA 8-bit samples).
        CounterT            *d_histogram[NUM_ACTIVE_CHANNELS],      ///< [out] The pointers to the histogram counter output arrays, one for each active channel.  For channel<sub><em>i</em></sub>, the allocation length of <tt>d_histograms[i]</tt> should be <tt>num_output_levels[i]</tt> - 1.
        int                 num_output_levels[NUM_ACTIVE_CHANNELS], ///< [in] The number of boundaries (levels) for delineating histogram samples in each active channel.  Implies that the number of bins for channel<sub><em>i</em></sub> is <tt>num_output_levels[i]</tt> - 1.
        LevelT              *d_levels[NUM_ACTIVE_CHANNELS],         ///< [in] The pointers to the arrays of boundaries (levels), one for each active channel.  Bin ranges are defined by consecutive boundary pairings: lower sample value boundaries are inclusive and upper sample value boundaries are exclusive.
        OffsetT             num_row_pixels,                         ///< [in] The number of multi-channel pixels per row in the region of interest
        OffsetT             num_rows,                               ///< [in] The number of rows in the region of interest
        OffsetT             row_stride_samples,                     ///< [in] The number of samples between starts of consecutive rows in the region of interest
        cudaStream_t        stream,                                 ///< [in] CUDA stream to launch kernels within.  Default is stream<sub>0</sub>.
        bool                debug_synchronous,                      ///< [in] Whether or not to synchronize the stream after every kernel launch to check for errors.  May cause significant slowdown.  Default is \p false.
        Int2Type<false>     is_byte_sample)                         ///< [in] Marker type indicating whether or not SampleT is a 8b type
    {
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

            // Get kernel kernel dispatch configurations
            KernelConfig histogram_sweep_config;
            InitConfigs(ptx_version, histogram_sweep_config);

            // Sweep pass uses the search transform op for converting samples to privatized bins
            SearchTransform<LevelT*> privatized_transform_op[NUM_ACTIVE_CHANNELS];
            for (int channel = 0; channel < NUM_ACTIVE_CHANNELS; ++channel)
            {
                privatized_transform_op[channel].Init(
                    d_levels[channel],
                    num_output_levels[channel]);
            }

            // Aggregation pass uses the pass-thru transform op for converting privatized bins to output bins
            PassThruTransform output_transform_op[NUM_ACTIVE_CHANNELS];

            // Determine the maximum number of bins in any channel
            int max_levels = num_output_levels[0];
            for (int channel = 1; channel < NUM_ACTIVE_CHANNELS; ++channel)
            {
                if (num_output_levels[channel] > max_levels)
                    max_levels = num_output_levels[channel];
            }
            int max_privatized_bins = max_levels - 1;

            static const bool SKIP_BIN_CONVERSION = true;

            // Dispatch
            if (max_privatized_bins > MAX_SMEM_BINS)
            {
/*
                // Too many bins to keep in shared memory.
                if (CubDebug(error = PrivatizedDispatch(
                    d_temp_storage,
                    temp_storage_bytes,
                    d_samples,
                    d_histogram,
                    num_output_levels,
                    privatized_transform_op,
                    num_output_levels,
                    output_transform_op,
                    max_privatized_bins,
                    SKIP_BIN_CONVERSION,
                    num_row_pixels,
                    num_rows,
                    row_stride_samples,
                    DeviceHistogramSweepKernel<PtxHistogramSweepPolicy, 0, NUM_CHANNELS, NUM_ACTIVE_CHANNELS, SampleIteratorT, CounterT, SearchTransform<LevelT*>, OffsetT>,
                    DeviceHistogramInitKernel<NUM_ACTIVE_CHANNELS, CounterT, OffsetT>,
                    DeviceHistogramAggregateKernel<PtxHistogramSweepPolicy, NUM_ACTIVE_CHANNELS, CounterT, PassThruTransform, SampleT, SKIP_BIN_CONVERSION>,
                    histogram_sweep_config,
                    stream,
                    debug_synchronous))) break;
*/
            }
            else
            {
                // Dispatch shared-privatized approach
                if (CubDebug(error = PrivatizedDispatch(
                    d_temp_storage,
                    temp_storage_bytes,
                    d_samples,
                    d_histogram,
                    num_output_levels,
                    privatized_transform_op,
                    num_output_levels,
                    output_transform_op,
                    max_privatized_bins,
                    SKIP_BIN_CONVERSION,
                    num_row_pixels,
                    num_rows,
                    row_stride_samples,
                    DeviceHistogramSweepKernel<PtxHistogramSweepPolicy, MAX_SMEM_BINS, NUM_CHANNELS, NUM_ACTIVE_CHANNELS, SampleIteratorT, CounterT, SearchTransform<LevelT*>, OffsetT>,
                    DeviceHistogramInitKernel<NUM_ACTIVE_CHANNELS, CounterT, OffsetT>,
                    DeviceHistogramAggregateKernel<PtxHistogramSweepPolicy, NUM_ACTIVE_CHANNELS, CounterT, PassThruTransform, SampleT, SKIP_BIN_CONVERSION>,
                    histogram_sweep_config,
                    stream,
                    debug_synchronous))) break;
            }

        } while (0);

        return error;
    }


    /**
     * Dispatch routine for HistogramRange, specialized for 1-byte types (computes 256-bin privatized histograms and then reduces to user-specified levels)
     */
    CUB_RUNTIME_FUNCTION
    static cudaError_t DispatchRange(
        void                *d_temp_storage,                        ///< [in] %Device allocation of temporary storage.  When NULL, the required allocation size is written to \p temp_storage_bytes and no work is done.
        size_t              &temp_storage_bytes,                    ///< [in,out] Reference to size in bytes of \p d_temp_storage allocation
        SampleIteratorT     d_samples,                              ///< [in] The pointer to the multi-channel input sequence of data samples. The samples from different channels are assumed to be interleaved (e.g., an array of 32-bit pixels where each pixel consists of four RGBA 8-bit samples).
        CounterT            *d_histogram[NUM_ACTIVE_CHANNELS],      ///< [out] The pointers to the histogram counter output arrays, one for each active channel.  For channel<sub><em>i</em></sub>, the allocation length of <tt>d_histograms[i]</tt> should be <tt>num_output_levels[i]</tt> - 1.
        int                 num_output_levels[NUM_ACTIVE_CHANNELS], ///< [in] The number of boundaries (levels) for delineating histogram samples in each active channel.  Implies that the number of bins for channel<sub><em>i</em></sub> is <tt>num_output_levels[i]</tt> - 1.
        LevelT              *d_levels[NUM_ACTIVE_CHANNELS],         ///< [in] The pointers to the arrays of boundaries (levels), one for each active channel.  Bin ranges are defined by consecutive boundary pairings: lower sample value boundaries are inclusive and upper sample value boundaries are exclusive.
        OffsetT             num_row_pixels,                         ///< [in] The number of multi-channel pixels per row in the region of interest
        OffsetT             num_rows,                               ///< [in] The number of rows in the region of interest
        OffsetT             row_stride_samples,                     ///< [in] The number of samples between starts of consecutive rows in the region of interest
        cudaStream_t        stream,                                 ///< [in] CUDA stream to launch kernels within.  Default is stream<sub>0</sub>.
        bool                debug_synchronous,                      ///< [in] Whether or not to synchronize the stream after every kernel launch to check for errors.  May cause significant slowdown.  Default is \p false.
        Int2Type<true>      is_byte_sample)                         ///< [in] Marker type indicating whether or not SampleT is a 8b type
    {
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

            // Get kernel kernel dispatch configurations
            KernelConfig histogram_sweep_config;
            InitConfigs(ptx_version, histogram_sweep_config);

            // Configure number of privatized levels
            int num_privatized_levels[NUM_ACTIVE_CHANNELS];
            for (int channel = 0; channel < NUM_ACTIVE_CHANNELS; ++channel)
                num_privatized_levels[channel] = 257;

            // Sweep pass uses the pass-thru transform op for converting samples to privatized bins
            PassThruTransform privatized_transform_op[NUM_ACTIVE_CHANNELS];

            // Aggregation pass uses the search transform op for converting privatized bins to output bins
            SearchTransform<LevelT*> output_transform_op[NUM_ACTIVE_CHANNELS];
            for (int channel = 0; channel < NUM_ACTIVE_CHANNELS; ++channel)
            {
                output_transform_op[channel].Init(
                    d_levels[channel],
                    num_output_levels[channel]);
            }

            static const bool SKIP_BIN_CONVERSION = false;

            // Dispatch shared-privatized approach
            if (CubDebug(error = PrivatizedDispatch(
                d_temp_storage,
                temp_storage_bytes,
                d_samples,
                d_histogram,
                num_privatized_levels,
                privatized_transform_op,
                num_output_levels,
                output_transform_op,
                256,
                SKIP_BIN_CONVERSION,
                num_row_pixels,
                num_rows,
                row_stride_samples,
                DeviceHistogramSweepKernel<PtxHistogramSweepPolicy, MAX_SMEM_BINS, NUM_CHANNELS, NUM_ACTIVE_CHANNELS, SampleIteratorT, CounterT, PassThruTransform, OffsetT>,
                DeviceHistogramInitKernel<NUM_ACTIVE_CHANNELS, CounterT, OffsetT>,
                DeviceHistogramAggregateKernel<PtxHistogramSweepPolicy, NUM_ACTIVE_CHANNELS, CounterT, SearchTransform<LevelT*>, SampleT, SKIP_BIN_CONVERSION>,
                histogram_sweep_config,
                stream,
                debug_synchronous))) break;

        } while (0);

        return error;
    }

    /**
     * Dispatch routine for HistogramEven, specialized for non-1-byte types
     */
    CUB_RUNTIME_FUNCTION __forceinline__
    static cudaError_t DispatchEven(
        void                *d_temp_storage,                        ///< [in] %Device allocation of temporary storage.  When NULL, the required allocation size is written to \p temp_storage_bytes and no work is done.
        size_t              &temp_storage_bytes,                    ///< [in,out] Reference to size in bytes of \p d_temp_storage allocation
        SampleIteratorT     d_samples,                              ///< [in] The pointer to the input sequence of sample items. The samples from different channels are assumed to be interleaved (e.g., an array of 32-bit pixels where each pixel consists of four RGBA 8-bit samples).
        CounterT            *d_histogram[NUM_ACTIVE_CHANNELS],      ///< [out] The pointers to the histogram counter output arrays, one for each active channel.  For channel<sub><em>i</em></sub>, the allocation length of <tt>d_histograms[i]</tt> should be <tt>num_output_levels[i]</tt> - 1.
        int                 num_output_levels[NUM_ACTIVE_CHANNELS], ///< [in] The number of bin level boundaries for delineating histogram samples in each active channel.  Implies that the number of bins for channel<sub><em>i</em></sub> is <tt>num_output_levels[i]</tt> - 1.
        LevelT              lower_level[NUM_ACTIVE_CHANNELS],       ///< [in] The lower sample value bound (inclusive) for the lowest histogram bin in each active channel.
        LevelT              upper_level[NUM_ACTIVE_CHANNELS],       ///< [in] The upper sample value bound (exclusive) for the highest histogram bin in each active channel.
        OffsetT             num_row_pixels,                         ///< [in] The number of multi-channel pixels per row in the region of interest
        OffsetT             num_rows,                               ///< [in] The number of rows in the region of interest
        OffsetT             row_stride_samples,                     ///< [in] The number of samples between starts of consecutive rows in the region of interest
        cudaStream_t        stream,                                 ///< [in] CUDA stream to launch kernels within.  Default is stream<sub>0</sub>.
        bool                debug_synchronous,                      ///< [in] Whether or not to synchronize the stream after every kernel launch to check for errors.  May cause significant slowdown.  Default is \p false.
        Int2Type<false>     is_byte_sample)                         ///< [in] Marker type indicating whether or not SampleT is a 8b type
    {
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

            // Get kernel kernel dispatch configurations
            KernelConfig histogram_sweep_config;
            InitConfigs(ptx_version, histogram_sweep_config);

            // Sweep pass uses the scale transform op for converting samples to privatized bins
            ScaleTransform privatized_transform_op[NUM_ACTIVE_CHANNELS];
            for (int channel = 0; channel < NUM_ACTIVE_CHANNELS; ++channel)
            {
                privatized_transform_op[channel].Init(
                    num_output_levels[channel],
                    upper_level[channel],
                    lower_level[channel],
                    (LevelT) ((upper_level[channel] - lower_level[channel]) / (num_output_levels[channel] - 1)));
            }

            // Aggregation pass uses the pass-thru transform op for converting privatized bins to output bins
            PassThruTransform output_transform_op[NUM_ACTIVE_CHANNELS];

            // Determine the maximum number of bins in any channel
            int max_levels = num_output_levels[0];
            for (int channel = 1; channel < NUM_ACTIVE_CHANNELS; ++channel)
            {
                if (num_output_levels[channel] > max_levels)
                    max_levels = num_output_levels[channel];
            }
            int max_privatized_bins = max_levels - 1;

            // Dispatch
            static const bool SKIP_BIN_CONVERSION = true;
            if (max_privatized_bins > MAX_SMEM_BINS)
            {
/*
                // Too many bins to keep in shared memory.  Dispatch global-privatized approach
                if (CubDebug(error = PrivatizedDispatch(
                    d_temp_storage,
                    temp_storage_bytes,
                    d_samples,
                    d_histogram,
                    num_output_levels,
                    privatized_transform_op,
                    num_output_levels,
                    output_transform_op,
                    max_privatized_bins,
                    SKIP_BIN_CONVERSION,
                    num_row_pixels,
                    num_rows,
                    row_stride_samples,
                    DeviceHistogramSweepKernel<PtxHistogramSweepPolicy, 0, NUM_CHANNELS, NUM_ACTIVE_CHANNELS, SampleIteratorT, CounterT, ScaleTransform, OffsetT>,
                    DeviceHistogramInitKernel<NUM_ACTIVE_CHANNELS, CounterT, OffsetT>,
                    DeviceHistogramAggregateKernel<PtxHistogramSweepPolicy, NUM_ACTIVE_CHANNELS, CounterT, PassThruTransform, SampleT, SKIP_BIN_CONVERSION>,
                    histogram_sweep_config,
                    stream,
                    debug_synchronous))) break;
*/
            }
            else
            {
                // Dispatch shared-privatized approach
                if (CubDebug(error = PrivatizedDispatch(
                    d_temp_storage,
                    temp_storage_bytes,
                    d_samples,
                    d_histogram,
                    num_output_levels,
                    privatized_transform_op,
                    num_output_levels,
                    output_transform_op,
                    max_privatized_bins,
                    SKIP_BIN_CONVERSION,
                    num_row_pixels,
                    num_rows,
                    row_stride_samples,
                    DeviceHistogramSweepKernel<PtxHistogramSweepPolicy, MAX_SMEM_BINS, NUM_CHANNELS, NUM_ACTIVE_CHANNELS, SampleIteratorT, CounterT, ScaleTransform, OffsetT>,
                    DeviceHistogramInitKernel<NUM_ACTIVE_CHANNELS, CounterT, OffsetT>,
                    DeviceHistogramAggregateKernel<PtxHistogramSweepPolicy, NUM_ACTIVE_CHANNELS, CounterT, PassThruTransform, SampleT, SKIP_BIN_CONVERSION>,
                    histogram_sweep_config,
                    stream,
                    debug_synchronous))) break;
            }
        }
        while (0);

        return error;
    }


    /**
     * Dispatch routine for HistogramEven, specialized for 1-byte types (computes 256-bin privatized histograms and then reduces to user-specified levels)
     */
    CUB_RUNTIME_FUNCTION __forceinline__
    static cudaError_t DispatchEven(
        void                *d_temp_storage,                        ///< [in] %Device allocation of temporary storage.  When NULL, the required allocation size is written to \p temp_storage_bytes and no work is done.
        size_t              &temp_storage_bytes,                    ///< [in,out] Reference to size in bytes of \p d_temp_storage allocation
        SampleIteratorT     d_samples,                              ///< [in] The pointer to the input sequence of sample items. The samples from different channels are assumed to be interleaved (e.g., an array of 32-bit pixels where each pixel consists of four RGBA 8-bit samples).
        CounterT            *d_histogram[NUM_ACTIVE_CHANNELS],      ///< [out] The pointers to the histogram counter output arrays, one for each active channel.  For channel<sub><em>i</em></sub>, the allocation length of <tt>d_histograms[i]</tt> should be <tt>num_output_levels[i]</tt> - 1.
        int                 num_output_levels[NUM_ACTIVE_CHANNELS], ///< [in] The number of bin level boundaries for delineating histogram samples in each active channel.  Implies that the number of bins for channel<sub><em>i</em></sub> is <tt>num_output_levels[i]</tt> - 1.
        LevelT              lower_level[NUM_ACTIVE_CHANNELS],       ///< [in] The lower sample value bound (inclusive) for the lowest histogram bin in each active channel.
        LevelT              upper_level[NUM_ACTIVE_CHANNELS],       ///< [in] The upper sample value bound (exclusive) for the highest histogram bin in each active channel.
        OffsetT             num_row_pixels,                         ///< [in] The number of multi-channel pixels per row in the region of interest
        OffsetT             num_rows,                               ///< [in] The number of rows in the region of interest
        OffsetT             row_stride_samples,                     ///< [in] The number of samples between starts of consecutive rows in the region of interest
        cudaStream_t        stream,                                 ///< [in] CUDA stream to launch kernels within.  Default is stream<sub>0</sub>.
        bool                debug_synchronous,                      ///< [in] Whether or not to synchronize the stream after every kernel launch to check for errors.  May cause significant slowdown.  Default is \p false.
        Int2Type<true>      is_byte_sample)                         ///< [in] Marker type indicating whether or not SampleT is a 8b type
    {
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

            // Get kernel kernel dispatch configurations
            KernelConfig histogram_sweep_config;
            InitConfigs(ptx_version, histogram_sweep_config);

            // Configure number of privatized levels
            int num_privatized_levels[NUM_ACTIVE_CHANNELS];
            for (int channel = 0; channel < NUM_ACTIVE_CHANNELS; ++channel)
                num_privatized_levels[channel] = 257;

            // Sweep pass uses pass-thru transform op for converting samples to privatized bins
            PassThruTransform privatized_transform_op[NUM_ACTIVE_CHANNELS];

            // Aggregation pass uses the scale transform op for scaling privatized bins to output bins
            ScaleTransform output_transform_op[NUM_ACTIVE_CHANNELS];
            for (int channel = 0; channel < NUM_ACTIVE_CHANNELS; ++channel)
            {
                output_transform_op[channel].Init(
                    num_output_levels[channel],
                    upper_level[channel],
                    lower_level[channel],
                    (LevelT) ((upper_level[channel] - lower_level[channel]) / (num_output_levels[channel] - 1)));
            }

            // Determine the minimum number of bins in any channel
            int min_levels = num_output_levels[0];
            for (int channel = 1; channel < NUM_ACTIVE_CHANNELS; ++channel)
            {
                if (num_output_levels[channel] < min_levels)
                    min_levels = num_output_levels[channel];
            }
            int min_bins = min_levels - 1;


            if ((!std::numeric_limits<SampleT>::is_signed) && (min_bins == 256))
            {
                // We can skip the conversion of privatized bins to output bins (because they are the same)
                static const bool SKIP_BIN_CONVERSION = true;

                if (CubDebug(error = PrivatizedDispatch(
                    d_temp_storage,
                    temp_storage_bytes,
                    d_samples,
                    d_histogram,
                    num_privatized_levels,
                    privatized_transform_op,
                    num_output_levels,
                    output_transform_op,
                    256,
                    SKIP_BIN_CONVERSION,
                    num_row_pixels,
                    num_rows,
                    row_stride_samples,
                    DeviceHistogramSweepKernel<PtxHistogramSweepPolicy, MAX_SMEM_BINS, NUM_CHANNELS, NUM_ACTIVE_CHANNELS, SampleIteratorT, CounterT, PassThruTransform, OffsetT>,
                    DeviceHistogramInitKernel<NUM_ACTIVE_CHANNELS, CounterT, OffsetT>,
                    DeviceHistogramAggregateKernel<PtxHistogramSweepPolicy, NUM_ACTIVE_CHANNELS, CounterT, ScaleTransform, SampleT, SKIP_BIN_CONVERSION>,
                    histogram_sweep_config,
                    stream,
                    debug_synchronous))) break;
            }
            else
            {
                // We have to convert privatized bins to output bins and perform atomic updates to the final output counters
                static const bool SKIP_BIN_CONVERSION = false;

                if (CubDebug(error = PrivatizedDispatch(
                    d_temp_storage,
                    temp_storage_bytes,
                    d_samples,
                    d_histogram,
                    num_privatized_levels,
                    privatized_transform_op,
                    num_output_levels,
                    output_transform_op,
                    256,
                    SKIP_BIN_CONVERSION,
                    num_row_pixels,
                    num_rows,
                    row_stride_samples,
                    DeviceHistogramSweepKernel<PtxHistogramSweepPolicy, MAX_SMEM_BINS, NUM_CHANNELS, NUM_ACTIVE_CHANNELS, SampleIteratorT, CounterT, PassThruTransform, OffsetT>,
                    DeviceHistogramInitKernel<NUM_ACTIVE_CHANNELS, CounterT, OffsetT>,
                    DeviceHistogramAggregateKernel<PtxHistogramSweepPolicy, NUM_ACTIVE_CHANNELS, CounterT, ScaleTransform, SampleT, SKIP_BIN_CONVERSION>,
                    histogram_sweep_config,
                    stream,
                    debug_synchronous))) break;
            }

        }
        while (0);

        return error;
    }

};


}               // CUB namespace
CUB_NS_POSTFIX  // Optional outer namespace(s)


