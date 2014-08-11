
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

#include "dispatch/device_histogram_dispatch.cuh"
#include "../util_namespace.cuh"

/// Optional outer namespace(s)
CUB_NS_PREFIX

/// CUB namespace
namespace cub {


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
     * \name Evenly-segmented bin ranges
     *********************************************************************/
    //@{

    /**
     * \brief Computes an intensity histogram from a sequence of data samples using equal-width bins.
     *
     * \par
     * - The number of histogram bins is \p num_levels - 1
     * - The value ranges for all bins have the same width: (\p upper_level - \p lower_level) / (\p num_levels - 1)
     * - \devicestorage
     * - \cdp
     *
     * \par Snippet
     * The code snippet below illustrates the computation of an six-bin histogram
     * from a sequence of float samples
     *
     * \par
     * \code
     * #include <cub/cub.cuh>   // or equivalently <cub/device/device_histogram.cuh>
     *
     * // Declare, allocate, and initialize device pointers for input samples and
     * // output histogram
     * int           num_samples;  // e.g., 10 samples
     * float         *d_samples;   // e.g., [2.2, 6.0, 7.1, 2.9, 3.5, 0.3, 2.9, 2.0, 6.1, 999.5]
     * unsigned int  *d_histogram; // e.g., [ , , , , , , , ]
     * int           num_levels;   // e.g., 7       (seven level boundaries for six bins)
     * float         lower_level;  // e.g., 0.0     (lower boundary of lowest bin)
     * float         upper_level;  // e.g., 12.0    (upper boundary of upper bin)
     * ...
     *
     * // Determine temporary device storage requirements
     * void     *d_temp_storage = NULL;
     * size_t   temp_storage_bytes = 0;
     * cub::DeviceHistogram::HistogramEven(d_temp_storage, temp_storage_bytes,
     *     d_samples, num_samples, d_histogram, num_levels, lower_level, upper_level);
     *
     * // Allocate temporary storage
     * cudaMalloc(&d_temp_storage, temp_storage_bytes);
     *
     * // Compute histograms
     * cub::DeviceHistogram::HistogramEven(d_temp_storage, temp_storage_bytes,
     *     d_samples, num_samples, d_histogram, num_levels, lower_level, upper_level);
     *
     * // d_histogram   <-- [1, 0, 5, 0, 3, 0, 0, 0];
     *
     * \endcode
     *
     * \tparam InputIteratorT           <b>[inferred]</b> Random-access input iterator type for reading input samples. \iterator
     * \tparam CounterT                 <b>[inferred]</b> Integer type for histogram bin counters
     * \tparam LevelT                   <b>[inferred]</b> Type for specifying bin level boundaries
     */
    template <
        typename            InputIteratorT,
        typename            CounterT,
        typename            LevelT>
    CUB_RUNTIME_FUNCTION
    static cudaError_t HistogramEven(
        void                *d_temp_storage,                    ///< [in] %Device allocation of temporary storage.  When NULL, the required allocation size is written to \p temp_storage_bytes and no work is done.
        size_t              &temp_storage_bytes,                ///< [in,out] Reference to size in bytes of \p d_temp_storage allocation
        InputIteratorT      d_samples,                          ///< [in] The pointer to the input sequence of sample items.
        int                 num_samples,                        ///< [in] The total number of samples to process
        CounterT            *d_histogram,                       ///< [out] The pointer to the histogram counter output array of length <tt>num_levels</tt> - 1.
        int                 num_levels,                         ///< [in] The number of bin level boundaries for delineating histogram samples.  Implies that the number of bins is <tt>num_levels</tt> - 1.
        LevelT              lower_level,                        ///< [in] The lower bound (inclusive) for the lowest histogram bin.
        LevelT              upper_level,                        ///< [in] The upper bound (exclusive) for the highest histogram bin.
        cudaStream_t        stream              = 0,            ///< [in] <b>[optional]</b> CUDA stream to launch kernels within.  Default is stream<sub>0</sub>.
        bool                debug_synchronous   = false)        ///< [in] <b>[optional]</b> Whether or not to synchronize the stream after every kernel launch to check for errors.  May cause significant slowdown.  Default is \p false.
    {
    }


    /**
     * \brief Computes intensity histograms from a sequence of multi-channel data samples using equal-width bins.
     *
     * \par
     * - The input is a set of independent data channels that have been
     *   interleaved into a single sequence of data samples (e.g., an array
     *   of pixels where each pixel is comprised of four RGBA sample values).
     *   The total number of samples across all channels (\p num_samples) must
     *   be a whole multiple of \p NUM_CHANNELS.
     * - Of the \p NUM_CHANNELS specified, the function will only compute histograms
     *   for the first \p NUM_ACTIVE_CHANNELS (e.g., RGB histograms from RGBA
     *   pixel samples).
     * - The number of histogram bins for channel<sub><em>i</em></sub> is <tt>num_levels[i]</tt> - 1.
     * - For channel<sub><em>i</em></sub>, the value ranges for all histogram bins
     *   have the same width: (<tt>upper_level[i]</tt> - <tt>lower_level[i]</tt>) / (<tt> num_levels[i]</tt> - 1)
     * - \devicestorage
     * - \cdp
     *
     * \par Snippet
     * The code snippet below illustrates the computation of three 256-bin RGB histograms
     * from a quad-channel sequence of RGBA pixels (8 bits per channel per pixel)
     *
     * \par
     * \code
     * #include <cub/cub.cuh>   // or equivalently <cub/device/device_histogram.cuh>
     *
     * // Declare, allocate, and initialize device pointers for input samples
     * // and output histograms
     * int              num_samples;     // e.g., 20 (4 channel samples per pixel * 5 pixels)
     * unsigned char    *d_samples;      // e.g., [(2, 6, 7, 5), (3, 0, 2, 1), (7, 0, 6, 2),
     *                                   //        (0, 6, 7, 5), (3, 0, 2, 6)]
     * unsigned int     *d_histogram[3]; // e.g., three device pointers each allocated
     *                                   //       with 256 integer counters
     * int              num_levels[3];   // e.g., {257, 257, 257};
     * unsigned int     lower_level[3];  // e.g., {0, 0, 0};
     * unsigned int     upper_level[3];  // e.g., {256, 256, 256};
     * ...
     *
     * // Determine temporary device storage requirements
     * void     *d_temp_storage = NULL;
     * size_t   temp_storage_bytes = 0;
     * cub::DeviceHistogram::MultiHistogramEven<4, 3>(d_temp_storage, temp_storage_bytes,
     *     d_samples, num_samples, d_histograms, num_levels, lower_level, upper_level);
     *
     * // Allocate temporary storage
     * cudaMalloc(&d_temp_storage, temp_storage_bytes);
     *
     * // Compute histograms
     * cub::DeviceHistogram::MultiHistogramEven<4, 3>(d_temp_storage, temp_storage_bytes,
     *     d_samples, num_samples, d_histograms, num_levels, lower_level, upper_level);
     *
     * // d_histogram   <-- [ [1, 0, 1, 2, 0, 0, 0, 1, 0, 0, 0, ..., 0],
     * //                     [0, 3, 0, 0, 0, 0, 2, 0, 0, 0, 0, ..., 0],
     * //                     [0, 0, 2, 0, 0, 0, 1, 2, 0, 0, 0, ..., 0] ]
     *
     * \endcode
     *
     * \tparam NUM_CHANNELS             Number of channels interleaved in the input data (may be greater than the number of channels being actively histogrammed)
     * \tparam NUM_ACTIVE_CHANNELS      <b>[inferred]</b> Number of channels actively being histogrammed
     * \tparam InputIteratorT           <b>[inferred]</b> Random-access input iterator type for reading input samples. \iterator
     * \tparam CounterT                 <b>[inferred]</b> Integer type for histogram bin counters
     * \tparam LevelT                   <b>[inferred]</b> Type for specifying bin level boundaries
     */
    template <
        int                 NUM_CHANNELS,
        int                 NUM_ACTIVE_CHANNELS,
        typename            InputIteratorT,
        typename            CounterT,
        typename            LevelT>
    CUB_RUNTIME_FUNCTION
    static cudaError_t MultiHistogramEven(
        void                *d_temp_storage,                    ///< [in] %Device allocation of temporary storage.  When NULL, the required allocation size is written to \p temp_storage_bytes and no work is done.
        size_t              &temp_storage_bytes,                ///< [in,out] Reference to size in bytes of \p d_temp_storage allocation
        InputIteratorT      d_samples,                          ///< [in] The pointer to the input sequence of sample items. The samples from different channels are assumed to be interleaved (e.g., an array of 32-bit pixels where each pixel consists of four RGBA 8-bit samples).
        int                 num_samples,                        ///< [in] The total number of samples to process across all channels (e.g., number of channels per pixel * number of pixels)
        CounterT            *d_histogram[NUM_ACTIVE_CHANNELS],  ///< [out] The pointers to the histogram counter output arrays, one for each active channel.  For channel<sub><em>i</em></sub>, the allocation length of <tt>d_histograms[i]</tt> should be <tt>num_levels[i]</tt> - 1.
        int                 num_levels[NUM_ACTIVE_CHANNELS],    ///< [in] The number of bin level boundaries for delineating histogram samples in each active channel.  Implies that the number of bins for channel<sub><em>i</em></sub> is <tt>num_levels[i]</tt> - 1.
        LevelT              lower_level[NUM_ACTIVE_CHANNELS],   ///< [in] The lower bound (inclusive) for the lowest histogram bin in each active channel.
        LevelT              upper_level[NUM_ACTIVE_CHANNELS],   ///< [in] The upper bound (exclusive) for the highest histogram bin in each active channel.
        cudaStream_t        stream              = 0,            ///< [in] <b>[optional]</b> CUDA stream to launch kernels within.  Default is stream<sub>0</sub>.
        bool                debug_synchronous   = false)        ///< [in] <b>[optional]</b> Whether or not to synchronize the stream after every kernel launch to check for errors.  May cause significant slowdown.  Default is \p false.
    {
    }


    //@}  end member group
    /******************************************************************//**
     * \name Custom bin ranges
     *********************************************************************/
    //@{

    /**
     * \brief Computes an intensity histogram from a sequence of data samples using the specified bin boundary levels.
     *
     * \par
     * - The number of histogram bins is \p num_levels - 1
     * - The value range for bin<sub><em>i</em></sub> is [<tt>level[i]</tt>, <tt>level[i+1]</tt>)
     * - \devicestorage
     * - \cdp
     *
     * \par Snippet
     * The code snippet below illustrates the computation of an six-bin histogram
     * from a sequence of float samples
     *
     * \par
     * \code
     * #include <cub/cub.cuh>   // or equivalently <cub/device/device_histogram.cuh>
     *
     * // Declare, allocate, and initialize device pointers for input samples and
     * // output histogram
     * int           num_samples;  // e.g., 10 samples
     * float         *d_samples;   // e.g., [2.2, 6.0, 7.1, 2.9, 3.5, 0.3, 2.9, 2.0, 6.1, 999.5]
     * unsigned int  *d_histogram; // e.g., [ , , , , , , , ]
     * int           num_levels    // e.g., 7 (seven level boundaries for six bins)
     * float         *d_levels;    // e.g., [0.0, 2.0, 4.0, 6.0, 8.0, 12.0, 16.0]
     * ...
     *
     * // Determine temporary device storage requirements
     * void     *d_temp_storage = NULL;
     * size_t   temp_storage_bytes = 0;
     * cub::DeviceHistogram::HistogramRange(d_temp_storage, temp_storage_bytes,
     *     d_samples, num_samples, d_histogram, num_levels, d_levels);
     *
     * // Allocate temporary storage
     * cudaMalloc(&d_temp_storage, temp_storage_bytes);
     *
     * // Compute histograms
     * cub::DeviceHistogram::HistogramRange(d_temp_storage, temp_storage_bytes,
     *     d_samples, num_samples, d_histogram, num_levels, d_levels);
     *
     * // d_histogram   <-- [1, 0, 5, 0, 3, 0, 0, 0];
     *
     * \endcode
     *
     * \tparam InputIteratorT           <b>[inferred]</b> Random-access input iterator type for reading input samples. \iterator
     * \tparam CounterT                 <b>[inferred]</b> Integer type for histogram bin counters
     * \tparam LevelT                   <b>[inferred]</b> Type for specifying bin level boundaries
     */
    template <
        typename            InputIteratorT,
        typename            CounterT,
        typename            LevelT>
    CUB_RUNTIME_FUNCTION
    static cudaError_t HistogramRange(
        void                *d_temp_storage,                    ///< [in] %Device allocation of temporary storage.  When NULL, the required allocation size is written to \p temp_storage_bytes and no work is done.
        size_t              &temp_storage_bytes,                ///< [in,out] Reference to size in bytes of \p d_temp_storage allocation
        InputIteratorT      d_samples,                          ///< [in] The pointer to the input sequence of sample items.
        int                 num_samples,                        ///< [in] The total number of samples to process
        CounterT            *d_histogram,                       ///< [out] The pointer to the histogram counter output array of length <tt>num_levels</tt> - 1.
        int                 num_levels,                         ///< [in] The number of bin level boundaries for delineating histogram samples.  Implies that the number of bins is <tt>num_levels</tt> - 1.
        LevelT              *d_levels,                          ///< [in] The pointer to the array of bin level boundaries.  Bin ranges are defined by consecutive boundary pairings: lower boundaries are inclusive and upper boundaries are exclusive.
        cudaStream_t        stream              = 0,            ///< [in] <b>[optional]</b> CUDA stream to launch kernels within.  Default is stream<sub>0</sub>.
        bool                debug_synchronous   = false)        ///< [in] <b>[optional]</b> Whether or not to synchronize the stream after every kernel launch to check for errors.  May cause significant slowdown.  Default is \p false.
    {
    }


    /**
     * \brief Computes intensity histograms from a sequence of multi-channel data samples using the specified bin boundary levels.
     *
     * \par
     * - The input is a set of independent data channels that have been
     *   interleaved into a single sequence of data samples (e.g., an array
     *   of pixels where each pixel is comprised of four RGBA sample values).
     *   The total number of samples across all channels (\p num_samples) must
     *   be a whole multiple of \p NUM_CHANNELS.
     * - Of the \p NUM_CHANNELS specified, the function will only compute histograms
     *   for the first \p NUM_ACTIVE_CHANNELS (e.g., RGB histograms from RGBA
     *   pixel samples).
     * - The number of histogram bins for channel<sub><em>i</em></sub> is <tt>num_levels[i]</tt> - 1.
     * - For channel<sub><em>i</em></sub>, the value ranges for all histogram bins
     *   have the same width: (<tt>upper_level[i]</tt> - <tt>lower_level[i]</tt>) / (<tt> num_levels[i]</tt> - 1)
     * - \devicestorage
     * - \cdp
     *
     * \par Snippet
     * The code snippet below illustrates the computation of three 4-bin RGB histograms
     * from a quad-channel sequence of RGBA pixels (8 bits per channel per pixel)
     *
     * \par
     * \code
     * #include <cub/cub.cuh>   // or equivalently <cub/device/device_histogram.cuh>
     *
     * // Declare, allocate, and initialize device pointers for input samples
     * // and output histograms
     * int            num_samples;      // e.g., 20 (4 channel samples per pixel * 5 pixels)
     * unsigned char  *d_samples;       // e.g., [(2, 6, 7, 5), (3, 0, 2, 1),
     *                                  //        (7, 0, 6, 2), (0, 6, 7, 5), (3, 0, 2, 6)]
     * unsigned int   *d_histogram[3];  // e.g., [[ , , , ],[ , , , ],[ , , , ]];
     * int            num_levels[3];    // e.g., {5, 5, 5};
     * unsigned int   *d_levels[3];     // e.g., [ [0, 2, 4, 6, 8],
     *                                  //         [0, 2, 4, 6, 8],
     *                                  //         [0, 2, 4, 6, 8] ];
     * ...
     *
     * // Determine temporary device storage requirements
     * void     *d_temp_storage = NULL;
     * size_t   temp_storage_bytes = 0;
     * cub::DeviceHistogram::MultiHistogramEven<4, 3>(d_temp_storage, temp_storage_bytes,
     *     d_samples, num_samples, d_histograms, num_levels, d_levels);
     *
     * // Allocate temporary storage
     * cudaMalloc(&d_temp_storage, temp_storage_bytes);
     *
     * // Compute histograms
     * cub::DeviceHistogram::MultiHistogramEven<4, 3>(d_temp_storage, temp_storage_bytes,
     *     d_samples, num_samples, d_histograms, num_levels, d_levels);
     *
     * // d_histogram   <-- [ [1, 3, 0, 1],
     * //                     [3, 0, 0, 2],
     * //                     [0, 2, 0, 3] ]
     *
     * \endcode
     *
     * \tparam NUM_CHANNELS             Number of channels interleaved in the input data (may be greater than the number of channels being actively histogrammed)
     * \tparam NUM_ACTIVE_CHANNELS      <b>[inferred]</b> Number of channels actively being histogrammed
     * \tparam InputIteratorT           <b>[inferred]</b> Random-access input iterator type for reading input samples. \iterator
     * \tparam CounterT                 <b>[inferred]</b> Integer type for histogram bin counters
     * \tparam LevelT                   <b>[inferred]</b> Type for specifying bin level boundaries
     */
    template <
        int                 NUM_CHANNELS,
        int                 NUM_ACTIVE_CHANNELS,
        typename            InputIteratorT,
        typename            CounterT,
        typename            LevelT>
    CUB_RUNTIME_FUNCTION
    static cudaError_t MultiHistogramRange(
        void                *d_temp_storage,                    ///< [in] %Device allocation of temporary storage.  When NULL, the required allocation size is written to \p temp_storage_bytes and no work is done.
        size_t              &temp_storage_bytes,                ///< [in,out] Reference to size in bytes of \p d_temp_storage allocation
        InputIteratorT      d_samples,                          ///< [in] The pointer to the input sequence of sample items. The samples from different channels are assumed to be interleaved (e.g., an array of 32-bit pixels where each pixel consists of four RGBA 8-bit samples).
        int                 num_samples,                        ///< [in] The total number of samples to process across all channels (e.g., number of channels per pixel * number of pixels)
        CounterT            *d_histogram[NUM_ACTIVE_CHANNELS],  ///< [out] The pointers to the histogram counter output arrays, one for each active channel.  For channel<sub><em>i</em></sub>, the allocation length of <tt>d_histograms[i]</tt> should be <tt>num_levels[i]</tt> - 1.
        int                 num_levels[NUM_ACTIVE_CHANNELS],    ///< [in] The number of bin level boundaries for delineating histogram samples in each active channel.  Implies that the number of bins for channel<sub><em>i</em></sub> is <tt>num_levels[i]</tt> - 1.
        LevelT              *d_levels[NUM_ACTIVE_CHANNELS],     ///< [in] The pointers to the arrays of bin level boundaries, one for each active channel.  Bin ranges are defined by consecutive boundary pairings: lower boundaries are inclusive and upper boundaries are exclusive.
        cudaStream_t        stream              = 0,            ///< [in] <b>[optional]</b> CUDA stream to launch kernels within.  Default is stream<sub>0</sub>.
        bool                debug_synchronous   = false)        ///< [in] <b>[optional]</b> Whether or not to synchronize the stream after every kernel launch to check for errors.  May cause significant slowdown.  Default is \p false.
    {
    }



    //@}  end member group
};

/**
 * \example example_device_histogram.cu
 */

}               // CUB namespace
CUB_NS_POSTFIX  // Optional outer namespace(s)


