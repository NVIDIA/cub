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
 * cub::BlockHisto256 provides methods for constructing (and compositing into) 256-valued histograms from 8b data partitioned across threads within a CUDA thread block.
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


/**
 * BlockHisto256Algorithm enumerates alternative algorithms for the parallel
 * construction of 8b histograms.
 */
enum BlockHisto256Algorithm
{

    /**
     * \par Overview
     * Sorting followed by differentiation.  Execution is comprised of two phases:
     * -# Sort the 8b data using efficient radix sort
     * -# Look for "runs" of same-valued 8b keys by detecting discontinuities; the run-lengths are histogram bin counts.
     */
    BLOCK_BYTE_HISTO_SORT,


    /**
     * \par Overview
     * Use atomic addition to update byte counts directly
     *
     * \par Usage Considerations
     * BLOCK_BYTE_HISTO_ATOMIC can only be used on version SM120 or later. Otherwise BLOCK_BYTE_HISTO_SORT is used regardless.
     */
    BLOCK_BYTE_HISTO_ATOMIC,
};


/**
 * \addtogroup BlockModule
 * @{
 */

/**
 * \brief BlockHisto256 provides methods for constructing (and compositing into) 256-valued histograms from 8b data partitioned across threads within a CUDA thread block.
 *
 * \par Overview
 * A <a href="http://en.wikipedia.org/wiki/Histogram"><em>histogram</em></a>
 * counts the number of observations that fall into each of the disjoint categories (known as <em>bins</em>).
 *
 * \par
 * For convenience, BlockHisto256 provides alternative entrypoints that differ by:
 * - Complete/incremental composition (compute a new histogram vs. update existing histogram data)
 *
 * \tparam BLOCK_THREADS    The threadblock size in threads
 * \tparam ALGORITHM        <b>[optional]</b> cub::BlockHisto256Algorithm tuning policy.  Default = cub::BLOCK_BYTE_HISTO_SORT.
 *
 * \par Algorithm
 * BlockHisto256 can be (optionally) configured to use different algorithms:
 *   -# <b>cub::BLOCK_BYTE_HISTO_SORT</b>.  An efficient An efficient "raking" reduction algorithm. [More...](\ref cub::BlockHisto256Algorithm)
 *   -# <b>cub::BLOCK_BYTE_HISTO_ATOMIC</b>.  A quick "tiled warp-reductions" reduction algorithm. [More...](\ref cub::BlockHisto256Algorithm)
 *
 * \par Usage Considerations
 * - The histogram output can be constructed in shared or global memory
 * - Supports partially-full threadblocks (i.e., the most-significant thread ranks having undefined values).
 * - \smemreuse{BlockHisto256::SmemStorage}
 *
 * \par Performance Considerations
 * - Computation is slightly more efficient (i.e., having lower instruction overhead) for:
 *   - \p BLOCK_THREADS is a multiple of the architecture's warp size
 *   - Every thread has a valid input (i.e., full <em>vs.</em> partial-tiles)
 * - See cub::BlockHisto256Algorithm for performance details regarding algorithmic alternatives
 *
 * \par Examples
 * \par
 * <em>Example 1.</em> Compute a simple 8b histogram in shared memory from 512 byte values that
 * are partitioned across a 128-thread threadblock (where each thread holds 4 values).
 * \code
 * #include <cub/cub.cuh>
 *
 * __global__ void SomeKernel(...)
 * {
 *      // Parameterize BlockHisto256 for 128 threads
 *      typedef cub::BlockHisto256<128> BlockHisto256;
 *
 *      // Declare shared memory for BlockHisto256
 *      __shared__ typename BlockHisto256::SmemStorage smem_storage;
 *
 *      // Declare shared memory for histogram
 *      __shared__ unsigned char smem_histo[256];
 *
 *      // Input items per thread
 *      unsigned char data[4];
 *
 *      // Obtain items
 *      ...
 *
 *      // Compute the threadblock-wide histogram
 *      BlockHisto256::Sum(smem_storage, smem_histo, data);
 *
 *      ...
 * \endcode
 *
 * \par
 * <em>Example 2:</em> Composite an incremental round of 8b histogram data onto
 * an existing histogram in global memory.  The composition is guarded: not all
 * threads have valid inputs.
 * \code
 * #include <cub/cub.cuh>
 *
 * template <int BLOCK_THREADS>
 * __global__ void SomeKernel(..., int *d_histogram, int num_items)
 * {
 *      // Parameterize BlockHisto256
 *      typedef cub::BlockHisto256<BLOCK_THREADS> BlockHisto256;
 *
 *      // Declare shared memory for BlockHisto256
 *      __shared__ typename BlockHisto256::SmemStorage smem_storage;
 *
 *      // Guarded load of input item
 *      int data;
 *      if (threadIdx.x < num_items) data = ...;
 *
 *      // Compute the threadblock-wide sum of valid elements in thread0
 *      BlockHisto256::Sum(smem_storage, d_histogram, data, num_items);
 *
 *      ...
 * \endcode
 *
 */
template <
    int                         BLOCK_THREADS,
    int                         ITEMS_PER_THREAD,
    BlockHisto256Algorithm     ALGORITHM = BLOCK_BYTE_HISTO_SORT>
class BlockHisto256
{
private:

    /******************************************************************************
     * Constants
     ******************************************************************************/

    /**
     * Ensure the template parameterization meets the requirements of the
     * targeted device architecture.  BLOCK_BYTE_HISTO_ATOMIC can only be used
     * on version SM120 or later.  Otherwise BLOCK_BYTE_HISTO_SORT is used
     * regardless.
     */
    static const BlockHisto256Algorithm SAFE_ALGORITHM =
        ((ALGORITHM == BLOCK_BYTE_HISTO_ATOMIC) && (CUB_PTX_ARCH < 120)) ?
            BLOCK_BYTE_HISTO_SORT :
            ALGORITHM;

    #ifndef DOXYGEN_SHOULD_SKIP_THIS    // Do not document


    /******************************************************************************
     * Algorithmic variants
     ******************************************************************************/

    /**
     * BLOCK_BYTE_HISTO_SORT algorithmic variant
     */
    template <BlockHisto256Algorithm _ALGORITHM, int DUMMY = 0>
    struct BlockHisto256Internal
    {
        // Parameterize BlockRadixSort type for our thread block
        typedef BlockRadixSort<unsigned char, BLOCK_THREADS, ITEMS_PER_THREAD> BlockRadixSortT;

        // Parameterize BlockDiscontinuity type for our thread block
        typedef BlockDiscontinuity<unsigned char, BLOCK_THREADS> BlockDiscontinuityT;

        // Shared memory
        union SmemStorage
        {
            // Storage for sorting bin values
            typename BlockRadixSortT::SmemStorage sort_storage;

            struct
            {
                // Storage for detecting discontinuities in the tile of sorted bin values
                BlockDiscontinuityT discont_storage;

                // Storage for noting begin/end offsets of bin runs in the tile of sorted bin values
                unsigned short run_begin[256];
                unsigned short run_end[256];
            };
        };


        // Discontinuity functor
        struct DiscontinuityOp
        {
            // Reference to smem_storage
            SmemStorage &smem_storage;

            // Constructor
            DiscontinuityOp(SmemStorage smem_storage) : smem_storage(smem_storage) {}

            // Discontinuity predicate
            bool operator()(const unsigned char &a, const unsigned char &b, unsigned int b_index)
            {
                if (a != b)
                {
                    // Note the begin/end offsets in shared storage
                    smem_storage.run_begin[b] = b_index;
                    smem_storage.run_end[a] = b_index - 1;
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
            int                 ITEMS_PER_THREAD,
            typename            HistoCounter>
        static __device__ __forceinline__ void Composite(
            SmemStorage         &smem_storage,                  ///< [in] Reference to shared memory allocation having layout type SmemStorage
            unsigned char       (&items)[ITEMS_PER_THREAD],     ///< [in] Calling thread's 8b input values to histogram
            HistoCounter        histogram[256])                 ///< [out] Reference to shared/global memory 256-bin histogram
        {
            enum { TILE_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD };

            // Sort bytes in blocked arrangement
            BlockRadixSortT::SortBlocked(smem_storage.sort_storage, items);

            __syncthreads();

            // Initialize the shared memory's run_begin and run_end for each bin
            int histo_offset = 0;

            #pragma unroll
            for(; histo_offset + BLOCK_THREADS <= 256; histo_offset += BLOCK_THREADS)
            {
                smem_storage.run_begin[histo_offset + threadIdx.x] = TILE_SIZE;
                smem_storage.run_end[histo_offset + threadIdx.x] = TILE_SIZE;
            }
            // Finish up with guarded initialization if necessary
            if ((histo_offset < BLOCK_THREADS) && (histo_offset + threadIdx.x < 256))
            {
                smem_storage.run_begin[histo_offset + threadIdx.x] = TILE_SIZE;
                smem_storage.run_end[histo_offset + threadIdx.x] = TILE_SIZE;
            }

            __syncthreads();

            int flags[ITEMS_PER_THREAD];    // unused

            // Note the begin/end run offsets of bin runs in the sorted tile
            BlockDiscontinuityT::Flag(smem_storage.discont_storage, items, DiscontinuityOp(smem_storage), flags);

            // Update begin and end for first item
            if (threadIdx.x == 0) smem_storage.run_begin[items[0]] = 0;

            __syncthreads();

            // Update histogram
            histo_offset = 0;

            #pragma unroll
            for(; histo_offset + BLOCK_THREADS <= 256; histo_offset += BLOCK_THREADS)
            {
                int thread_offset = histo_offset + threadIdx.x;
                histogram[thread_offset] += smem_storage.run_end[thread_offset] - smem_storage.run_begin[thread_offset];
            }
            // Finish up with guarded composition if necessary
            if ((histo_offset < BLOCK_THREADS) && (histo_offset + threadIdx.x < 256))
            {
                int thread_offset = histo_offset + threadIdx.x;
                histogram[thread_offset] += smem_storage.run_end[thread_offset] - smem_storage.run_begin[thread_offset];
            }
        }

    };


    /**
     * BLOCK_BYTE_HISTO_ATOMIC algorithmic variant
     */
    template <int DUMMY>
    struct BlockHisto256Internal<BLOCK_BYTE_HISTO_ATOMIC, DUMMY>
    {
        /// Shared memory storage layout type
        struct SmemStorage {};

        /// Composite data onto an existing histogram
        template <
            int                 ITEMS_PER_THREAD,
            typename            HistoCounter>
        static __device__ __forceinline__ void Composite(
            SmemStorage         &smem_storage,                  ///< [in] Reference to shared memory allocation having layout type SmemStorage
            unsigned char       (&items)[ITEMS_PER_THREAD],     ///< [in] Calling thread's 8b input values to histogram
            HistoCounter        histogram[256])                 ///< [out] Reference to shared/global memory 256-bin histogram
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


    /// Shared memory storage layout type for BlockHisto256
    typedef typename BlockHisto256Internal<SAFE_ALGORITHM>::SmemStorage _SmemStorage;


public:

    /// \smemstorage{BlockHisto256}
    typedef _SmemStorage SmemStorage;

    /**
     * \brief Constructs a threadblock-wide histogram in shared/global memory.  Each thread contributes an array of 8b input elements.
     *
     * \smemreuse
     *
     * \tparam ITEMS_PER_THREAD     <b>[inferred]</b> The number of 8b values per thread
     * \tparam HistoCounter         <b>[inferred]</b> Histogram counter type
     */
    template <
        int                 ITEMS_PER_THREAD,
        typename            HistoCounter>
    static __device__ __forceinline__ void Histogram(
        SmemStorage         &smem_storage,                  ///< [in] Reference to shared memory allocation having layout type SmemStorage
        unsigned char       (&items)[ITEMS_PER_THREAD],     ///< [in] Calling thread's 8b input values to histogram
        HistoCounter        histogram[256])                 ///< [out] Reference to shared/global memory 256-bin histogram
    {
        // Initialize histogram bin counts to zeros
        int histo_offset = 0;

        #pragma unroll
        for(; histo_offset + BLOCK_THREADS <= 256; histo_offset += BLOCK_THREADS)
        {
            histogram[histo_offset + threadIdx.x] = 0;
        }
        // Finish up with guarded initialization if necessary
        if ((histo_offset < BLOCK_THREADS) && (histo_offset + threadIdx.x < 256))
        {
            histogram[histo_offset + threadIdx.x] = 0;
        }

        // Composite the histogram
        BlockHisto256Internal<SAFE_ALGORITHM>::Composite(smem_storage, items, histogram);
    }


    /**
     * \brief Updates an existing threadblock-wide histogram in shared/global memory.  Each thread composites an array of 8b input elements.
     *
     * \smemreuse
     *
     * \tparam ITEMS_PER_THREAD     <b>[inferred]</b> The number of 8b values per thread
     * \tparam HistoCounter         <b>[inferred]</b> Histogram counter type
     */
    template <
        int                 ITEMS_PER_THREAD,
        typename            HistoCounter>
    static __device__ __forceinline__ void Composite(
        SmemStorage         &smem_storage,                  ///< [in] Reference to shared memory allocation having layout type SmemStorage
        unsigned char       (&items)[ITEMS_PER_THREAD],     ///< [in] Calling thread's 8b input values to histogram
        HistoCounter        histogram[256])                 ///< [out] Reference to shared/global memory 256-bin histogram
    {
        BlockHisto256Internal<SAFE_ALGORITHM>::Composite(smem_storage, items, histogram);
    }



};

/** @} */       // end group BlockModule

}               // CUB namespace
CUB_NS_POSTFIX  // Optional outer namespace(s)

