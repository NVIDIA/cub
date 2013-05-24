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
 * Operations for writing global tiles of data from the threadblock (in blocked arrangement across threads).
 */

#pragma once

#include <iterator>

#include "../util_namespace.cuh"
#include "../util_macro.cuh"
#include "../util_type.cuh"
#include "../util_vector.cuh"
#include "../thread/thread_store.cuh"
#include "block_exchange.cuh"

/// Optional outer namespace(s)
CUB_NS_PREFIX

/// CUB namespace
namespace cub {

/**
 * \addtogroup IoModule
 * @{
 */


/******************************************************************//**
 * \name Blocked I/O
 *********************************************************************/
//@{

/**
 * \brief Store a blocked arrangement of items across a thread block using the specified cache modifier.
 *
 * The aggregate tile of items is assumed to be partitioned evenly across
 * threads in <em>blocked</em> arrangement with thread<sub><em>i</em></sub> owning
 * the <em>i</em><sup>th</sup> segment of consecutive elements.
 *
 * \tparam MODIFIER             cub::PtxStoreModifier cache modifier.
 * \tparam T                    <b>[inferred]</b> The data type to store.
 * \tparam ITEMS_PER_THREAD     <b>[inferred]</b> The number of consecutive items partitioned onto each thread.
 * \tparam OutputIteratorRA     <b>[inferred]</b> The random-access iterator type for output (may be a simple pointer type).
 */
template <
    PtxStoreModifier    MODIFIER,
    typename            T,
    int                 ITEMS_PER_THREAD,
    typename            OutputIteratorRA>
__device__ __forceinline__ void StoreBlocked(
    OutputIteratorRA    block_itr,                      ///< [in] The threadblock's base output iterator for storing to
    T                   (&items)[ITEMS_PER_THREAD])     ///< [in] Data to store
{
    // Store directly in thread-blocked order
    #pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
    {
        int item_offset = (threadIdx.x * ITEMS_PER_THREAD) + ITEM;
        Store<MODIFIER>(block_itr + item_offset, items[ITEM]);
    }
}


/**
 * \brief Store a blocked arrangement of items across a thread block using the specified cache modifier, guarded by range
 *
 * The aggregate tile of items is assumed to be partitioned evenly across
 * threads in <em>blocked</em> arrangement with thread<sub><em>i</em></sub> owning
 * the <em>i</em><sup>th</sup> segment of consecutive elements.
 *
 * \tparam MODIFIER             cub::PtxStoreModifier cache modifier.
 * \tparam T                    <b>[inferred]</b> The data type to store.
 * \tparam ITEMS_PER_THREAD     <b>[inferred]</b> The number of consecutive items partitioned onto each thread.
 * \tparam OutputIteratorRA     <b>[inferred]</b> The random-access iterator type for output (may be a simple pointer type).
 */
template <
    PtxStoreModifier    MODIFIER,
    typename            T,
    int                 ITEMS_PER_THREAD,
    typename            OutputIteratorRA>
__device__ __forceinline__ void StoreBlocked(
    OutputIteratorRA    block_itr,                      ///< [in] The threadblock's base output iterator for storing to
    const int           &guarded_items,                 ///< [in] Number of valid items in the tile
    T                   (&items)[ITEMS_PER_THREAD])     ///< [in] Data to store
{
    // Store directly in thread-blocked order
    #pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
    {
        if (ITEM + (threadIdx.x * ITEMS_PER_THREAD) < guarded_items)
        {
            Store<MODIFIER>(block_itr + (threadIdx.x * ITEMS_PER_THREAD) + ITEM, items[ITEM]);
        }
    }
}



//@}  end member group
/******************************************************************//**
 * \name Striped I/O
 *********************************************************************/
//@{


/**
 * \brief Store a striped arrangement of data across the thread block using the specified cache modifier.
 *
 * The aggregate tile of items is assumed to be partitioned across
 * threads in "striped" arrangement, i.e., the \p ITEMS_PER_THREAD
 * items owned by each thread have logical stride \p BLOCK_THREADS between them.
 *
 * \tparam MODIFIER             cub::PtxStoreModifier cache modifier.
 * \tparam T                    <b>[inferred]</b> The data type to store.
 * \tparam ITEMS_PER_THREAD     <b>[inferred]</b> The number of consecutive items partitioned onto each thread.
 * \tparam OutputIteratorRA     <b>[inferred]</b> The random-access iterator type for output (may be a simple pointer type).
 */
template <
    PtxStoreModifier    MODIFIER,
    typename            T,
    int                 ITEMS_PER_THREAD,
    typename            OutputIteratorRA>
__device__ __forceinline__ void StoreStriped(
    OutputIteratorRA    block_itr,                      ///< [in] The threadblock's base output iterator for storing to
    T                   (&items)[ITEMS_PER_THREAD],     ///< [in] Data to store
    int                 block_stride = blockDim.x)      ///< [in] <b>[optional]</b> Block stripe stride.  Default is the width of the threadblock.  More efficient code can be generated if a compile-time-constant (e.g., BLOCK_THREADS) is supplied.
{
    // Store directly in striped order
    #pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
    {
        int item_offset = (ITEM * block_stride) + threadIdx.x;
        Store<MODIFIER>(block_itr + item_offset, items[ITEM]);
    }
}


/**
 * \brief  Store a striped arrangement of data across the thread block using the specified cache modifier, guarded by range
 *
 * The aggregate tile of items is assumed to be partitioned across
 * threads in <em>striped</em> arrangement, i.e., the \p ITEMS_PER_THREAD
 * items owned by each thread have logical stride \p BLOCK_THREADS between them.
 *
 * \tparam MODIFIER             cub::PtxStoreModifier cache modifier.
 * \tparam T                    <b>[inferred]</b> The data type to store.
 * \tparam ITEMS_PER_THREAD     <b>[inferred]</b> The number of consecutive items partitioned onto each thread.
 * \tparam OutputIteratorRA     <b>[inferred]</b> The random-access iterator type for output (may be a simple pointer type).
 */
template <
    PtxStoreModifier    MODIFIER,
    typename            T,
    int                 ITEMS_PER_THREAD,
    typename            OutputIteratorRA>
__device__ __forceinline__ void StoreStriped(
    OutputIteratorRA    block_itr,                      ///< [in] The threadblock's base output iterator for storing to
    const int           &guarded_items,                 ///< [in] Number of valid items in the tile
    T                   (&items)[ITEMS_PER_THREAD],     ///< [in] Data to store
    int                 block_stride = blockDim.x)      ///< [in] <b>[optional]</b> Block stripe stride.  Default is the width of the threadblock.  More efficient code can be generated if a compile-time-constant (e.g., BLOCK_THREADS) is supplied.
{
    // Store directly in striped order
    #pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
    {
        if ((ITEM * block_stride) + threadIdx.x < guarded_items)
        {
            Store<MODIFIER>(block_itr + (ITEM * block_stride) + threadIdx.x, items[ITEM]);
        }
    }
}



//@}  end member group
/******************************************************************//**
 * \name Warp-striped I/O
 *********************************************************************/
//@{


/**
 * \brief Store warp-striped tile directly using the specified cache modifier.
 *
 * The aggregate tile of items is assumed to be partitioned across threads in
 * "warp-striped" fashion, i.e., each warp owns a contiguous segment of
 * (\p WARP_THREADS * \p ITEMS_PER_THREAD) items, and the \p ITEMS_PER_THREAD
 * items owned by each thread within the same warp have logical stride
 * \p WARP_THREADS between them.
 *
 * \par Usage Considerations
 * The number of threads in the thread block must be a multiple of the architecture's warp size.
 *
 * \tparam MODIFIER             cub::PtxStoreModifier cache modifier.
 * \tparam T                    <b>[inferred]</b> The data type to store.
 * \tparam ITEMS_PER_THREAD     <b>[inferred]</b> The number of consecutive items partitioned onto each thread.
 * \tparam OutputIteratorRA     <b>[inferred]</b> The random-access iterator type for output (may be a simple pointer type).
 */
template <
    PtxStoreModifier    MODIFIER,
    typename            T,
    int                 ITEMS_PER_THREAD,
    typename            OutputIteratorRA>
__device__ __forceinline__ void StoreWarpStriped(
    OutputIteratorRA    block_itr,                                                      ///< [in] The threadblock's base output iterator for storing to
    T                   (&items)[ITEMS_PER_THREAD],                                     ///< [out] Data to load
    int                 warp_stride = PtxArchProps::WARP_THREADS)                       ///< [in] <b>[optional]</b> Warp stripe stride.  Default is the width of the warp.
{
    int tid         = threadIdx.x & (PtxArchProps::WARP_THREADS - 1);
    int wid         = threadIdx.x >> PtxArchProps::LOG_WARP_THREADS;
    int warp_offset = wid * warp_stride * ITEMS_PER_THREAD;

    // Store directly in warp-striped order
    #pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
    {
        Store<MODIFIER>(block_itr + warp_offset + tid + (ITEM * warp_stride), items[ITEM]);
    }
}


/**
 * \brief Store warp-striped directly tile using the specified cache modifier, guarded by range
 *
 * The aggregate tile of items is assumed to be partitioned across threads in
 * "warp-striped" fashion, i.e., each warp owns a contiguous segment of
 * (\p WARP_THREADS * \p ITEMS_PER_THREAD) items, and the \p ITEMS_PER_THREAD
 * items owned by each thread within the same warp have logical stride
 * \p WARP_THREADS between them.
 *
 * \par Usage Considerations
 * The number of threads in the thread block must be a multiple of the architecture's warp size.
 *
 * \tparam MODIFIER             cub::PtxStoreModifier cache modifier.
 * \tparam T                    <b>[inferred]</b> The data type to store.
 * \tparam ITEMS_PER_THREAD     <b>[inferred]</b> The number of consecutive items partitioned onto each thread.
 * \tparam OutputIteratorRA     <b>[inferred]</b> The random-access iterator type for output (may be a simple pointer type).
 */
template <
    PtxStoreModifier    MODIFIER,
    typename            T,
    int                 ITEMS_PER_THREAD,
    typename            OutputIteratorRA>
__device__ __forceinline__ void StoreWarpStriped(
    OutputIteratorRA    block_itr,                                                      ///< [in] The threadblock's base output iterator for storing to
    const int           &guarded_items,                                                 ///< [in] Number of valid items in the tile
    T                   (&items)[ITEMS_PER_THREAD],                                     ///< [out] Data to load
    int                 warp_stride = PtxArchProps::WARP_THREADS)                       ///< [in] <b>[optional]</b> Warp stripe stride.  Default is the width of the warp.
{
    int tid         = threadIdx.x & (PtxArchProps::WARP_THREADS - 1);
    int wid         = threadIdx.x >> PtxArchProps::LOG_WARP_THREADS;
    int warp_offset = wid * warp_stride * ITEMS_PER_THREAD;

    // Store directly in warp-striped order
    #pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
    {
        if (warp_offset + tid + (ITEM * warp_stride) < guarded_items)
        {
            Store<MODIFIER>(block_itr + warp_offset + tid + (ITEM * warp_stride), items[ITEM]);
        }
    }
}



//@}  end member group
/******************************************************************//**
 * \name Blocked, vectorized I/O
 *********************************************************************/
//@{

/**
 * \brief Store a blocked arrangement of items across a thread block using the specified cache modifier.
 *
 * The aggregate tile of items is assumed to be partitioned evenly across
 * threads in <em>blocked</em> arrangement with thread<sub><em>i</em></sub> owning
 * the <em>i</em><sup>th</sup> segment of consecutive elements.
 *
 * The output offset (\p block_ptr + \p block_offset) must be quad-item aligned,
 * which is the default starting offset returned by \p cudaMalloc()
 *
 * \par
 * The following conditions will prevent vectorization and storing will fall back to cub::BLOCK_STORE_DIRECT:
 *   - \p ITEMS_PER_THREAD is odd
 *   - The data type \p T is not a built-in primitive or CUDA vector type (e.g., \p short, \p int2, \p double, \p float2, etc.)
 *
 * \tparam MODIFIER             cub::PtxStoreModifier cache modifier.
 * \tparam T                    <b>[inferred]</b> The data type to store.
 * \tparam ITEMS_PER_THREAD     <b>[inferred]</b> The number of consecutive items partitioned onto each thread.
 *
 */
template <
    PtxStoreModifier    MODIFIER,
    typename            T,
    int                 ITEMS_PER_THREAD>
__device__ __forceinline__ void StoreBlockedVectorized(
    T                   *block_ptr,                     ///< [in] Input pointer for storing from
    T                   (&items)[ITEMS_PER_THREAD])     ///< [in] Data to store
{
    enum
    {
        // Maximum CUDA vector size is 4 elements
        MAX_VEC_SIZE = CUB_MIN(4, ITEMS_PER_THREAD),

        // Vector size must be a power of two and an even divisor of the items per thread
        VEC_SIZE = ((((MAX_VEC_SIZE - 1) & MAX_VEC_SIZE) == 0) && ((ITEMS_PER_THREAD % MAX_VEC_SIZE) == 0)) ?
            MAX_VEC_SIZE :
            1,

        VECTORS_PER_THREAD = ITEMS_PER_THREAD / VEC_SIZE,
    };

    // Vector type
    typedef typename VectorHelper<T, VEC_SIZE>::Type Vector;

    // Alias global pointer
    Vector *block_ptr_vectors = reinterpret_cast<Vector *>(block_ptr);

    // Alias pointers (use "raw" array here which should get optimized away to prevent conservative PTXAS lmem spilling)
    Vector raw_vector[VECTORS_PER_THREAD];
    T *raw_items = reinterpret_cast<T*>(raw_vector);

    // Copy
    #pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
    {
        raw_items[ITEM] = items[ITEM];
    }

    // Direct-store using vector types
    StoreBlocked<MODIFIER>(block_ptr_vectors, raw_vector);
}


//@}  end member group


/** @} */       // end group IoModule


//-----------------------------------------------------------------------------
// Generic BlockStore abstraction
//-----------------------------------------------------------------------------

/**
 * \brief cub::BlockStorePolicy enumerates alternative algorithms for cub::BlockStore to store a tile of data to memory from a blocked arrangement across a CUDA thread block.
 */
enum BlockStorePolicy
{
    /**
     * \par Overview
     *
     * A [<em>blocked arrangement</em>](index.html#sec3sec3) of data is written
     * directly to memory.  The threadblock writes items in a parallel "raking" fashion:
     * thread<sub><em>i</em></sub> writes the <em>i</em><sup>th</sup> segment of consecutive elements.
     *
     * \par Performance Considerations
     * - The utilization of memory transactions (coalescing) decreases as the
     *   access stride between threads increases (i.e., the number items per thread).
     */
    BLOCK_STORE_DIRECT,

    /**
     * \par Overview
     *
     * A [<em>blocked arrangement</em>](index.html#sec3sec3) of data is written directly
     * to memory using CUDA's built-in vectorized stores as a coalescing optimization.
     * The threadblock writes items in a parallel "raking" fashion: thread<sub><em>i</em></sub> uses vector stores to
     * write the <em>i</em><sup>th</sup> segment of consecutive elements.
     *
     * For example, <tt>st.global.v4.s32</tt> instructions will be generated when \p T = \p int and \p ITEMS_PER_THREAD > 4.
     *
     * \par Performance Considerations
     * - The utilization of memory transactions (coalescing) remains high until the the
     *   access stride between threads (i.e., the number items per thread) exceeds the
     *   maximum vector store width (typically 4 items or 64B, whichever is lower).
     * - The following conditions will prevent vectorization and writing will fall back to cub::BLOCK_STORE_DIRECT:
     *   - \p ITEMS_PER_THREAD is odd
     *   - The \p OutputIteratorRA is not a simple pointer type
     *   - The block output offset is not quadword-aligned
     *   - The data type \p T is not a built-in primitive or CUDA vector type (e.g., \p short, \p int2, \p double, \p float2, etc.)
     */
    BLOCK_STORE_VECTORIZE,

    /**
     * \par Overview
     * A [<em>blocked arrangement</em>](index.html#sec3sec3) is locally
     * transposed into a [<em>striped arrangement</em>](index.html#sec3sec3)
     * which is then written to memory.  More specifically, cub::BlockExchange
     * used to locally reorder the items into a
     * [<em>striped arrangement</em>](index.html#sec3sec3), after which the
     * threadblock writes items in a parallel "strip-mining" fashion: consecutive
     * items owned by thread<sub><em>i</em></sub> are written to memory with
     * stride \p BLOCK_THREADS between them.
     *
     * \par Performance Considerations
     * - The utilization of memory transactions (coalescing) remains high regardless
     *   of items written per thread.
     * - The local reordering incurs slightly longer latencies and throughput than the
     *   direct cub::BLOCK_STORE_DIRECT and cub::BLOCK_STORE_VECTORIZE alternatives.
     */
    BLOCK_STORE_TRANSPOSE,

    /**
     * \par Overview
     * A [<em>blocked arrangement</em>](index.html#sec3sec3) is locally
     * transposed into a [<em>warp-striped arrangement</em>](index.html#sec3sec3)
     * which is then written to memory.  More specifically, cub::BlockExchange used
     * to locally reorder the items into a
     * [<em>warp-striped arrangement</em>](index.html#sec3sec3), after which
     * each warp writes its own contiguous segment in a parallel "strip-mining" fashion:
     * consecutive items owned by lane<sub><em>i</em></sub> are written to memory
     * with stride \p WARP_THREADS between them.
     *
     * \par Performance Considerations
     * - The utilization of memory transactions (coalescing) remains high regardless
     *   of items written per thread.
     * - The local reordering incurs slightly longer latencies and throughput than the
     *   direct cub::BLOCK_STORE_DIRECT and cub::BLOCK_STORE_VECTORIZE alternatives.
     */
    BLOCK_STORE_WARP_TRANSPOSE,
};



/**
 * \addtogroup BlockModule
 * @{
 */


/**
 * \brief BlockStore provides data movement operations for storing a tile of data to memory from a [<em>block arrangement</em>](index.html#sec3sec3) across a CUDA thread block.  ![](block_store_logo.png)
 *
 * BlockStore provides a tile-storing abstraction that implements alternative
 * cub::BlockStoreAlgorithm strategies that can be used to optimize data
 * movement on different architectures for different granularity sizes (i.e.,
 * number of items per thread).
 *
 * \tparam OutputIteratorRA     The input iterator type (may be a simple pointer type).
 * \tparam BLOCK_THREADS        The threadblock size in threads.
 * \tparam ITEMS_PER_THREAD     The number of consecutive items partitioned onto each thread.
 * \tparam ALGORITHM            <b>[optional]</b> cub::BlockStorePolicy tuning policy enumeration.  Default = cub::BLOCK_STORE_DIRECT.
 * \tparam MODIFIER             <b>[optional]</b> cub::PtxStoreModifier cache modifier.  Default = cub::STORE_DEFAULT.
 * \tparam WARP_TIME_SLICING    <b>[optional]</b> For cooperative cub::BlockStoreAlgorithm parameterizations that utilize shared memory: When \p true, only use enough shared memory for a single warp's worth of tile data, time-slicing the block-wide exchange over multiple synchronized rounds (default = false)
 *
 * \par Algorithm
 * BlockStore can be (optionally) configured to use one of three alternative methods:
 *   -# <b>cub::BLOCK_STORE_DIRECT</b>.  A [<em>blocked arrangement</em>](index.html#sec3sec3) of data is written
        directly to memory. [More...](\ref cub::BlockStorePolicy)
 *   -# <b>cub::BLOCK_STORE_VECTORIZE</b>.  A [<em>blocked arrangement</em>](index.html#sec3sec3)
 *      of data is written directly to memory using CUDA's built-in vectorized stores as a
 *      coalescing optimization.  [More...](\ref cub::BlockStorePolicy)
 *   -# <b>cub::BLOCK_STORE_TRANSPOSE</b>.  A [<em>blocked arrangement</em>](index.html#sec3sec3)
 *      is locally transposed into a [<em>striped arrangement</em>](index.html#sec3sec3) which is
 *      then written to memory.  [More...](\ref cub::BlockStorePolicy)
 *   -# <b>cub::BLOCK_STORE_WARP_TRANSPOSE</b>.  A [<em>blocked arrangement</em>](index.html#sec3sec3)
 *      is locally transposed into a [<em>warp-striped arrangement</em>](index.html#sec3sec3) which is
 *      then written to memory.  [More...](\ref cub::BlockStorePolicy)
 *
 * \par Usage Considerations
 * - \smemreuse{BlockStore::SmemStorage}
 *
 * \par Performance Considerations
 *  - See cub::BlockStorePolicy for more performance details regarding algorithmic alternatives
 *
 * \par Examples
 * <em>Example 1.</em> Have a 128-thread thread block directly store a blocked
 * arrangement of four consecutive integers per thread.  The store operation
 * may suffer from non-coalesced memory accesses because consecutive threads are
 * referencing non-consecutive outputs as each item is written.
 * \code
 * #include <cub/cub.cuh>
 *
 * __global__ void SomeKernel(int *d_out, ...)
 * {
 *      // Parameterize BlockStore for 128 threads (4 items each) on type int
 *      typedef cub::BlockStore<int*, 128, 4> BlockStore;
 *
 *      // Declare shared memory for BlockStore
 *      __shared__ typename BlockStore::SmemStorage smem_storage;
 *
 *      // A segment of consecutive items per thread
 *      int data[4];
 *
 *      // Store a tile of data at this block's offset
 *      BlockStore::Store(smem_storage, d_out + blockIdx.x * 128 * 4, data);
 *
 *      ...
 * \endcode
 *
 * \par
 * <em>Example 2.</em> Have a thread block transpose a blocked arrangement of 21
 * consecutive integers per thread into a striped arrangement across the thread
 * block and then store the items to memory using strip-mined writes.  The store
 * operation has perfectly coalesced memory accesses because consecutive threads
 * are referencing consecutive output items.
 * \code
 * #include <cub/cub.cuh>
 *
 * template <int BLOCK_THREADS>
 * __global__ void SomeKernel(int *d_out, ...)
 * {
 *      // Parameterize BlockStore on type int
 *      typedef cub::BlockStore<int*, BLOCK_THREADS, 21, BLOCK_STORE_TRANSPOSE> BlockStore;
 *
 *      // Declare shared memory for BlockStore
 *      __shared__ typename BlockStore::SmemStorage smem_storage;
 *
 *      // A segment of consecutive items per thread
 *      int data[21];
 *
 *      // Store a tile of data at this block's offset
 *      BlockStore::Store(smem_storage, d_out + blockIdx.x * BLOCK_THREADS * 21, data);
 *
 *      ...
 * \endcode
 *
 * \par
 * <em>Example 3</em> Have a thread block store a blocked arrangement of
 * \p ITEMS_PER_THREAD consecutive integers per thread using vectorized
 * stores and global-only caching.  The store operation will have perfectly
 * coalesced memory accesses if ITEMS_PER_THREAD is 1, 2, or 4 which allows
 * consecutive threads to write consecutive int1, int2, or int4 words.
 * \code
 * #include <cub/cub.cuh>
 *
 * template <
 *     int BLOCK_THREADS,
 *     int ITEMS_PER_THREAD>
 * __global__ void SomeKernel(int *d_out, ...)
 * {
 *      // Parameterize BlockStore on type int
 *      typedef cub::BlockStore<int*, BLOCK_THREADS, ITEMS_PER_THREAD, BLOCK_STORE_VECTORIZE, STORE_CG> BlockStore;
 *
 *      // Declare shared memory for BlockStore
 *      __shared__ typename BlockStore::SmemStorage smem_storage;
 *
 *      // A segment of consecutive items per thread
 *      int data[ITEMS_PER_THREAD];
 *
 *      // Store a tile of data using vector-store instructions if possible
 *      BlockStore::Store(smem_storage, d_out + blockIdx.x * BLOCK_THREADS * ITEMS_PER_THREAD, data);
 *
 *      ...
 * \endcode
 * <br>
 */
template <
    typename            OutputIteratorRA,
    int                 BLOCK_THREADS,
    int                 ITEMS_PER_THREAD,
    BlockStorePolicy    ALGORITHM = BLOCK_STORE_DIRECT,
    PtxStoreModifier    MODIFIER = STORE_DEFAULT,
    bool                WARP_TIME_SLICING = false>
class BlockStore
{
    //---------------------------------------------------------------------
    // Type definitions and constants
    //---------------------------------------------------------------------

private:

    enum
    {
        WARP_THREADS = PtxArchProps::WARP_THREADS
    };

    // Data type of input iterator
    typedef typename std::iterator_traits<OutputIteratorRA>::value_type T;


    /// Store helper
    template <BlockStorePolicy _POLICY, int DUMMY = 0>
    struct StoreInternal;


    /**
     * BLOCK_STORE_DIRECT specialization of store helper
     */
    template <int DUMMY>
    struct StoreInternal<BLOCK_STORE_DIRECT, DUMMY>
    {
        /// Shared memory storage layout type
        typedef NullType SmemStorage;

        /// Store a tile of items across a threadblock
        static __device__ __forceinline__ void Store(
            SmemStorage         &smem_storage,              ///< [in] Reference to shared memory allocation having layout type SmemStorage
            OutputIteratorRA    block_itr,                  ///< [in] The threadblock's base output iterator for storing to
            T                   (&items)[ITEMS_PER_THREAD]) ///< [in] Data to store
        {
            StoreBlocked<MODIFIER>(block_itr, items);
        }

        /// Store a tile of items across a threadblock, guarded by range
        static __device__ __forceinline__ void Store(
            SmemStorage         &smem_storage,              ///< [in] Reference to shared memory allocation having layout type SmemStorage
            OutputIteratorRA    block_itr,                  ///< [in] The threadblock's base output iterator for storing to
            const int           &guarded_items,             ///< [in] Number of valid items in the tile
            T                   (&items)[ITEMS_PER_THREAD]) ///< [in] Data to store
        {
            StoreBlocked<MODIFIER>(block_itr, guarded_items, items);
        }
    };


    /**
     * BLOCK_STORE_VECTORIZE specialization of store helper
     */
    template <int DUMMY>
    struct StoreInternal<BLOCK_STORE_VECTORIZE, DUMMY>
    {
        /// Shared memory storage layout type
        typedef NullType SmemStorage;

        /// Store a tile of items across a threadblock, specialized for native pointer types (attempts vectorization)
        static __device__ __forceinline__ void Store(
            SmemStorage         &smem_storage,              ///< [in] Reference to shared memory allocation having layout type SmemStorage
            T                   *block_ptr,                 ///< [in] The threadblock's base output iterator for storing to
            T                   (&items)[ITEMS_PER_THREAD]) ///< [in] Data to store
        {
            StoreBlockedVectorized<MODIFIER>(block_ptr, items);
        }

        /// Store a tile of items across a threadblock, specialized for opaque input iterators (skips vectorization)
        template <typename _OutputIteratorRA>
        static __device__ __forceinline__ void Store(
            SmemStorage         &smem_storage,              ///< [in] Reference to shared memory allocation having layout type SmemStorage
            _OutputIteratorRA   block_itr,                  ///< [in] The threadblock's base output iterator for storing to
            T                   (&items)[ITEMS_PER_THREAD]) ///< [in] Data to store
        {
            StoreBlocked<MODIFIER>(block_itr, items);
        }

        /// Store a tile of items across a threadblock, guarded by range
        static __device__ __forceinline__ void Store(
            SmemStorage         &smem_storage,              ///< [in] Reference to shared memory allocation having layout type SmemStorage
            OutputIteratorRA    block_itr,                  ///< [in] The threadblock's base output iterator for storing to
            const int           &guarded_items,             ///< [in] Number of valid items in the tile
            T                   (&items)[ITEMS_PER_THREAD]) ///< [in] Data to store
        {
            StoreBlocked<MODIFIER>(block_itr, guarded_items, items);
        }
    };


    /**
     * BLOCK_STORE_TRANSPOSE specialization of store helper
     */
    template <int DUMMY>
    struct StoreInternal<BLOCK_STORE_TRANSPOSE, DUMMY>
    {
        // BlockExchange utility type for keys
        typedef BlockExchange<T, BLOCK_THREADS, ITEMS_PER_THREAD, WARP_TIME_SLICING> BlockExchange;

        /// Shared memory storage layout type
        typedef typename BlockExchange::SmemStorage SmemStorage;

        /// Store a tile of items across a threadblock
        static __device__ __forceinline__ void Store(
            SmemStorage         &smem_storage,              ///< [in] Reference to shared memory allocation having layout type SmemStorage
            OutputIteratorRA    block_itr,                  ///< [in] The threadblock's base output iterator for storing to
            T                   (&items)[ITEMS_PER_THREAD]) ///< [in] Data to store
        {
            BlockExchange::BlockedToStriped(smem_storage, items);
            StoreStriped<MODIFIER>(block_itr, items, BLOCK_THREADS);
        }

        /// Store a tile of items across a threadblock, guarded by range
        static __device__ __forceinline__ void Store(
            SmemStorage         &smem_storage,              ///< [in] Reference to shared memory allocation having layout type SmemStorage
            OutputIteratorRA    block_itr,                  ///< [in] The threadblock's base output iterator for storing to
            const int           &guarded_items,             ///< [in] Number of valid items in the tile
            T                   (&items)[ITEMS_PER_THREAD]) ///< [in] Data to store
        {
            BlockExchange::BlockedToStriped(smem_storage, items);
            StoreStriped<MODIFIER>(block_itr, guarded_items, items, BLOCK_THREADS);
        }
    };


    /**
     * BLOCK_STORE_WARP_TRANSPOSE specialization of store helper
     */
    template <int DUMMY>
    struct StoreInternal<BLOCK_STORE_WARP_TRANSPOSE, DUMMY>
    {
        // Assert BLOCK_THREADS must be a multiple of WARP_THREADS
        CUB_STATIC_ASSERT((BLOCK_THREADS % WARP_THREADS == 0), "BLOCK_THREADS must be a multiple of WARP_THREADS");

        // BlockExchange utility type for keys
        typedef BlockExchange<T, BLOCK_THREADS, ITEMS_PER_THREAD, WARP_TIME_SLICING> BlockExchange;

        /// Shared memory storage layout type
        typedef typename BlockExchange::SmemStorage SmemStorage;

        /// Store a tile of items across a threadblock
        static __device__ __forceinline__ void Store(
            SmemStorage         &smem_storage,              ///< [in] Reference to shared memory allocation having layout type SmemStorage
            OutputIteratorRA    block_itr,                  ///< [in] The threadblock's base output iterator for storing to
            T                   (&items)[ITEMS_PER_THREAD]) ///< [in] Data to store
        {
            BlockExchange::BlockedToWarpStriped(smem_storage, items);
            StoreWarpStriped<MODIFIER>(block_itr, items, WARP_THREADS);
        }

        /// Store a tile of items across a threadblock, guarded by range
        static __device__ __forceinline__ void Store(
            SmemStorage         &smem_storage,              ///< [in] Reference to shared memory allocation having layout type SmemStorage
            OutputIteratorRA    block_itr,                  ///< [in] The threadblock's base output iterator for storing to
            const int           &guarded_items,             ///< [in] Number of valid items in the tile
            T                   (&items)[ITEMS_PER_THREAD]) ///< [in] Data to store
        {
            BlockExchange::BlockedToWarpStriped(smem_storage, items);
            StoreWarpStriped<MODIFIER>(block_itr, guarded_items, items, WARP_THREADS);
        }
    };

    /// Shared memory storage layout type
    typedef typename StoreInternal<ALGORITHM>::SmemStorage _SmemStorage;

public:

    /// \smemstorage{BlockStore}
    typedef _SmemStorage SmemStorage;



    //---------------------------------------------------------------------
    // Interface
    //---------------------------------------------------------------------

    /**
     * \brief Store a tile of items across a threadblock.
     */
    static __device__ __forceinline__ void Store(
        SmemStorage         &smem_storage,                  ///< [in] Reference to shared memory allocation having layout type SmemStorage
        OutputIteratorRA    block_itr,                      ///< [in] The threadblock's base output iterator for storing to
        T                   (&items)[ITEMS_PER_THREAD])     ///< [in] Data to store
    {
        StoreInternal<ALGORITHM>::Store(smem_storage, block_itr, items);
    }

    /**
     * \brief Store a tile of items across a threadblock, guarded by range.
     */
    static __device__ __forceinline__ void Store(
        SmemStorage         &smem_storage,                  ///< [in] Reference to shared memory allocation having layout type SmemStorage
        OutputIteratorRA    block_itr,                      ///< [in] The threadblock's base output iterator for storing to
        const int           &guarded_items,                 ///< [in] Number of valid items in the tile
        T                   (&items)[ITEMS_PER_THREAD])     ///< [in] Data to store
    {
        StoreInternal<ALGORITHM>::Store(smem_storage, block_itr, guarded_items, items);
    }
};

/** @} */       // end group BlockModule

}               // CUB namespace
CUB_NS_POSTFIX  // Optional outer namespace(s)

