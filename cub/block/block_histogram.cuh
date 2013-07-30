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
 * cub::BlockHistogram provides methods for constructing (and compositing into) block-wide histograms from data partitioned across threads within a CUDA thread block.
 */

#pragma once

#include "../util_arch.cuh"
#include "../block/block_radix_sort.cuh"
#include "../block/block_discontinuity.cuh"
#include "../util_namespace.cuh"

/// Optional outer namespace(s)
CUB_NS_PREFIX

/// CUB namespace
namespace cub {


/******************************************************************************
 * Algorithmic variants
 ******************************************************************************/

/**
 * \brief BlockHistogramAlgorithm enumerates alternative algorithms for the parallel construction of block-wide histograms.
 */
enum BlockHistogramAlgorithm
{

    /**
     * \par Overview
     * Sorting followed by differentiation.  Execution is comprised of two phases:
     * -# Sort the data using efficient radix sort
     * -# Look for "runs" of same-valued keys by detecting discontinuities; the run-lengths are histogram bin counts.
     *
     * \par Performance Considerations
     * Delivers consistent throughput regardless of sample bin distribution.
     */
    BLOCK_HISTO_SORT,


    /**
     * \par Overview
     * Use atomic addition to update byte counts directly
     *
     * \par Performance Considerations
     * Performance is strongly tied to the hardware implementation of atomic
     * addition, and may be significantly degraded for non uniformly-random
     * input distributions where many concurrent updates are likely to be
     * made to the same bin counter.
     */
    BLOCK_HISTO_ATOMIC,
};



/******************************************************************************
 * Block histogram
 ******************************************************************************/


/**
 * \brief BlockHistogram provides methods for constructing (and compositing into) block-wide histograms from data partitioned across threads within a CUDA thread block. ![](histogram_logo.png)
 * \ingroup BlockModule
 *
 * \par Overview
 * A <a href="http://en.wikipedia.org/wiki/Histogram"><em>histogram</em></a>
 * counts the number of observations that fall into each of the disjoint categories (known as <em>bins</em>).
 *
 * \par
 * For convenience, BlockHistogram provides alternative entrypoints that differ by:
 * - Complete/incremental composition (compute a new histogram vs. update existing histogram data)
 *
 * \tparam BLOCK_THREADS        The thread block size in threads
 * \tparam ITEMS_PER_THREAD     The number of items per thread
 * \tparam BINS                 The number bins within the histogram
 * \tparam ALGORITHM            <b>[optional]</b> cub::BlockHistogramAlgorithm enumerator specifying the underlying algorithm to use (default: cub::BLOCK_HISTO_SORT)
 *
 * \par Algorithm
 * BlockHistogram can be (optionally) configured to use different algorithms:
 *   -# <b>cub::BLOCK_HISTO_SORT</b>.  Sorting followed by differentiation. [More...](\ref cub::BlockHistogramAlgorithm)
 *   -# <b>cub::BLOCK_HISTO_ATOMIC</b>.  Use atomic addition to update byte counts directly. [More...](\ref cub::BlockHistogramAlgorithm)
 *
 * \par Usage Considerations
 * - The histogram output can be constructed in shared or global memory
 * - Supports partially-full threadblocks (i.e., the most-significant thread ranks having undefined values).
 * - \smemreuse{BlockHistogram::TempStorage}
 *
 * \par Performance Considerations
 * - Computation is slightly more efficient (i.e., having lower instruction overhead) for:
 *   - \p BLOCK_THREADS is a multiple of the architecture's warp size
 *   - Every thread has a valid input (i.e., full <b><em>vs.</em></b> partial-tiles)
 * - See cub::BlockHistogramAlgorithm for performance details regarding algorithmic alternatives
 *
 * \par Examples
 * \par
 * <em>Example 1.</em> Compute a simple histogram in shared memory from 512 byte values that
 * are partitioned across a 128-thread threadblock (where each thread holds 4 values).
 * \code
 * #include <cub/cub.cuh>
 *
 * __global__ void SomeKernel(...)
 * {
 *      // Parameterize a 256-bin BlockHistogram type for 128 threads having 4 samples each
 *      typedef cub::BlockHistogram<128, 4, 256> BlockHistogram;
 *
 *      // Declare shared memory for BlockHistogram
 *      __shared__ typename BlockHistogram::TempStorage temp_storage;
 *
 *      // Declare shared memory for histogram bins
 *      __shared__ unsigned int smem_histogram[256];
 *
 *      // Input items per thread
 *      unsigned char data[4];
 *
 *      // Obtain items
 *      ...
 *
 *      // Compute the threadblock-wide histogram
 *      BlockHistogram(temp_storage).Histogram(data, smem_histogram);
 *
 *      ...
 * \endcode
 *
 * \par
 * <em>Example 2:</em> Composite an incremental round of histogram data onto
 * an existing histogram in global memory.
 * \code
 * #include <cub/cub.cuh>
 *
 * template <int BLOCK_THREADS>
 * __global__ void SomeKernel(..., int *d_histogram)
 * {
 *      // Parameterize a 256-bin BlockHistogram type where each thread composites one sample item
 *      typedef cub::BlockHistogram<BLOCK_THREADS, 1, 256> BlockHistogram;
 *
 *      // Declare shared memory for BlockHistogram
 *      __shared__ typename BlockHistogram::TempStorage temp_storage;
 *
 *      // Guarded load of input item
 *      int data[1];
 *      if (threadIdx.x < num_items) data[0] = ...;
 *
 *      // Compute the threadblock-wide sum of valid elements in thread0
 *      BlockHistogram(temp_storage).Composite(data, d_histogram);
 *
 *      ...
 * \endcode
 *
 */
template <
    int                     BLOCK_THREADS,
    int                     ITEMS_PER_THREAD,
    int                     BINS,
    BlockHistogramAlgorithm     ALGORITHM = BLOCK_HISTO_SORT>
class BlockHistogram
{
private:

    /******************************************************************************
     * Constants and type definitions
     ******************************************************************************/

    /**
     * Ensure the template parameterization meets the requirements of the
     * targeted device architecture.  BLOCK_HISTO_ATOMIC can only be used
     * on version SM120 or later.  Otherwise BLOCK_HISTO_SORT is used
     * regardless.
     */
    static const BlockHistogramAlgorithm SAFE_ALGORITHM =
        ((ALGORITHM == BLOCK_HISTO_ATOMIC) && (CUB_PTX_ARCH < 120)) ?
            BLOCK_HISTO_SORT :
            ALGORITHM;

    #ifndef DOXYGEN_SHOULD_SKIP_THIS    // Do not document


    /******************************************************************************
     * Algorithmic variants
     ******************************************************************************/

    /**
     * BLOCK_HISTO_SORT algorithmic variant
     */
    template <BlockHistogramAlgorithm _ALGORITHM, int DUMMY = 0>
    struct BlockHistogramInternal
    {
        // Parameterize BlockRadixSort type for our thread block
        typedef BlockRadixSort<unsigned char, BLOCK_THREADS, ITEMS_PER_THREAD> BlockRadixSortT;

        // Parameterize BlockDiscontinuity type for our thread block
        typedef BlockDiscontinuity<unsigned char, BLOCK_THREADS> BlockDiscontinuityT;

        // Shared memory
        union TempStorage
        {
            // Storage for sorting bin values
            typename BlockRadixSortT::TempStorage sort;

            struct
            {
                // Storage for detecting discontinuities in the tile of sorted bin values
                typename BlockDiscontinuityT::TempStorage flag;

                // Storage for noting begin/end offsets of bin runs in the tile of sorted bin values
                unsigned int run_begin[BINS];
                unsigned int run_end[BINS];
            };
        };


        // Thread fields
        TempStorage &temp_storage;
        int linear_tid;


        /// Constructor
        __device__ __forceinline__ BlockHistogramInternal(
            TempStorage &temp_storage,
            int linear_tid)
        :
            temp_storage(temp_storage),
            linear_tid(linear_tid)
        {}


        // Discontinuity functor
        struct DiscontinuityOp
        {
            // Reference to temp_storage
            TempStorage &temp_storage;

            // Constructor
            __device__ __forceinline__ DiscontinuityOp(TempStorage &temp_storage) : temp_storage(temp_storage) {}

            // Discontinuity predicate
            __device__ __forceinline__ bool operator()(const unsigned char &a, const unsigned char &b, unsigned int b_index)
            {
                if (a != b)
                {
                    // Note the begin/end offsets in shared storage
                    temp_storage.run_begin[b] = b_index;
                    temp_storage.run_end[a] = b_index;

                    return true;
                }
                else
                {
                    return false;
                }
            }
        };


        // Composite data onto an existing histogram
        template <
            typename            HistoCounter>
        __device__ __forceinline__ void Composite(
            unsigned char       (&items)[ITEMS_PER_THREAD],     ///< [in] Calling thread's input values to histogram
            HistoCounter        histogram[BINS])                 ///< [out] Reference to shared/global memory histogram
        {
            enum { TILE_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD };

            // Sort bytes in blocked arrangement
            BlockRadixSortT(temp_storage.sort, linear_tid).Sort(items);

            __syncthreads();

            // Initialize the shared memory's run_begin and run_end for each bin
            int histo_offset = 0;

            #pragma unroll
            for(; histo_offset + BLOCK_THREADS <= BINS; histo_offset += BLOCK_THREADS)
            {
                temp_storage.run_begin[histo_offset + linear_tid] = TILE_SIZE;
                temp_storage.run_end[histo_offset + linear_tid] = TILE_SIZE;
            }
            // Finish up with guarded initialization if necessary
            if ((histo_offset < BLOCK_THREADS) && (histo_offset + linear_tid < BINS))
            {
                temp_storage.run_begin[histo_offset + linear_tid] = TILE_SIZE;
                temp_storage.run_end[histo_offset + linear_tid] = TILE_SIZE;
            }

            __syncthreads();

            int flags[ITEMS_PER_THREAD];    // unused

            // Note the begin/end run offsets of bin runs in the sorted tile
            DiscontinuityOp flag_op(temp_storage);
            BlockDiscontinuityT(temp_storage.flag, linear_tid).Flag(items, flag_op, flags);

            // Update begin for first item
            if (linear_tid == 0) temp_storage.run_begin[items[0]] = 0;

            __syncthreads();

            // Composite into histogram
            histo_offset = 0;

            #pragma unroll
            for(; histo_offset + BLOCK_THREADS <= BINS; histo_offset += BLOCK_THREADS)
            {
                int thread_offset = histo_offset + linear_tid;
                HistoCounter count = temp_storage.run_end[thread_offset] - temp_storage.run_begin[thread_offset];
                histogram[thread_offset] += count;
            }
            // Finish up with guarded composition if necessary
            if ((histo_offset < BLOCK_THREADS) && (histo_offset + linear_tid < BINS))
            {
                int thread_offset = histo_offset + linear_tid;
                HistoCounter count = temp_storage.run_end[thread_offset] - temp_storage.run_begin[thread_offset];
                histogram[thread_offset] += count;
            }
        }

    };


    /**
     * BLOCK_HISTO_ATOMIC algorithmic variant
     */
    template <int DUMMY>
    struct BlockHistogramInternal<BLOCK_HISTO_ATOMIC, DUMMY>
    {
        /// Shared memory storage layout type
        struct TempStorage {};


        /// Constructor
        __device__ __forceinline__ BlockHistogramInternal(
            TempStorage &temp_storage,
            int linear_tid)
        {}


        /// Composite data onto an existing histogram
        template <
            typename            HistoCounter>
        __device__ __forceinline__ void Composite(
            unsigned char       (&items)[ITEMS_PER_THREAD],     ///< [in] Calling thread's input values to histogram
            HistoCounter        histogram[BINS])                 ///< [out] Reference to shared/global memory histogram
        {
            // Update histogram
            #pragma unroll
            for (int i = 0; i < ITEMS_PER_THREAD; ++i)
            {
                  atomicAdd(histogram + items[i], 1);
            }
        }

    };


    #endif // DOXYGEN_SHOULD_SKIP_THIS


    /******************************************************************************
     * Type definitions
     ******************************************************************************/

    /// Internal histogram implementation to use
    typedef BlockHistogramInternal<SAFE_ALGORITHM> InternalHistogram;

    /// Shared memory storage layout type for BlockHistogram
    typedef typename InternalHistogram::TempStorage _TempStorage;


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

    /// \smemstorage{BlockHistogram}
    typedef _TempStorage TempStorage;


    /******************************************************************//**
     * \name Collective constructors
     *********************************************************************/
    //@{

    /**
     * \brief Collective constructor for 1D thread blocks using a private static allocation of shared memory as temporary storage.  Threads are identified using <tt>threadIdx.x</tt>.
     */
    __device__ __forceinline__ BlockHistogram()
    :
        temp_storage(PrivateStorage()),
        linear_tid(threadIdx.x)
    {}


    /**
     * \brief Collective constructor for 1D thread blocks using the specified memory allocation as temporary storage.  Threads are identified using <tt>threadIdx.x</tt>.
     */
    __device__ __forceinline__ BlockHistogram(
        TempStorage &temp_storage)             ///< [in] Reference to memory allocation having layout type TempStorage
    :
        temp_storage(temp_storage),
        linear_tid(threadIdx.x)
    {}


    /**
     * \brief Collective constructor using a private static allocation of shared memory as temporary storage.  Each thread is identified using the supplied linear thread identifier
     */
    __device__ __forceinline__ BlockHistogram(
        int linear_tid)                        ///< [in] A suitable 1D thread-identifier for the calling thread (e.g., <tt>(threadIdx.y * blockDim.x) + linear_tid</tt> for 2D thread blocks)
    :
        temp_storage(PrivateStorage()),
        linear_tid(linear_tid)
    {}


    /**
     * \brief Collective constructor using the specified memory allocation as temporary storage.  Each thread is identified using the supplied linear thread identifier.
     */
    __device__ __forceinline__ BlockHistogram(
        TempStorage &temp_storage,             ///< [in] Reference to memory allocation having layout type TempStorage
        int linear_tid)                        ///< [in] <b>[optional]</b> A suitable 1D thread-identifier for the calling thread (e.g., <tt>(threadIdx.y * blockDim.x) + linear_tid</tt> for 2D thread blocks)
    :
        temp_storage(temp_storage),
        linear_tid(linear_tid)
    {}


    //@}  end member group
    /******************************************************************//**
     * \name Histogram operations
     *********************************************************************/
    //@{


    /**
     * \brief Initialize the shared histogram counters to zero.
     */
    template <typename HistoCounter>
    __device__ __forceinline__ void InitHistogram(HistoCounter histogram[BINS])
    {
        // Initialize histogram bin counts to zeros
        int histo_offset = 0;

        #pragma unroll
        for(; histo_offset + BLOCK_THREADS <= BINS; histo_offset += BLOCK_THREADS)
        {
            histogram[histo_offset + linear_tid] = 0;
        }
        // Finish up with guarded initialization if necessary
        if ((histo_offset < BLOCK_THREADS) && (histo_offset + linear_tid < BINS))
        {
            histogram[histo_offset + linear_tid] = 0;
        }
    }


    /**
     * \brief Constructs a threadblock-wide histogram in shared/global memory.  Each thread contributes an array of input elements.
     *
     * \smemreuse
     *
     * \tparam ITEMS_PER_THREAD     <b>[inferred]</b> The number of values per thread
     * \tparam HistoCounter         <b>[inferred]</b> Histogram counter type
     */
    template <
        typename            HistoCounter>
    __device__ __forceinline__ void Histogram(
        unsigned char       (&items)[ITEMS_PER_THREAD],     ///< [in] Calling thread's input values to histogram
        HistoCounter        histogram[BINS])                ///< [out] Reference to shared/global memory histogram
    {
        // Initialize histogram bin counts to zeros
        InitHistogram(histogram);

        // Composite the histogram
        InternalHistogram(temp_storage, linear_tid).Composite(items, histogram);
    }



    /**
     * \brief Updates an existing threadblock-wide histogram in shared/global memory.  Each thread composites an array of input elements.
     *
     * \smemreuse
     *
     * \tparam HistoCounter         <b>[inferred]</b> Histogram counter type
     */
    template <
        typename            HistoCounter>
    __device__ __forceinline__ void Composite(
        unsigned char       (&items)[ITEMS_PER_THREAD],     ///< [in] Calling thread's input values to histogram
        HistoCounter        histogram[BINS])                 ///< [out] Reference to shared/global memory histogram
    {
        InternalHistogram(temp_storage, linear_tid).Composite(items, histogram);
    }

};

}               // CUB namespace
CUB_NS_POSTFIX  // Optional outer namespace(s)

