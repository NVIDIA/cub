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
 * The cub::BlockStore type provides store operations for writing global tiles of data from the threadblock (in blocked arrangement across threads).
 */

#pragma once

#include <iterator>

#include "../ns_wrapper.cuh"
#include "../macro_utils.cuh"
#include "../thread/thread_store.cuh"
#include "../type_utils.cuh"
#include "../vector_type.cuh"
#include "block_exchange.cuh"

CUB_NS_PREFIX

/// CUB namespace
namespace cub {

/**
 *  \addtogroup SimtUtils
 * @{
 */


/******************************************************************//**
 * \name Direct threadblock stores (blocked arrangement)
 *********************************************************************/
//@{

/**
 * \brief Store a tile of items across a threadblock directly using the specified cache modifier.
 *
 * \tparam MODIFIER             cub::PtxStoreModifier cache modifier.
 * \tparam T                    <b>[inferred]</b> The data type to store.
 * \tparam ITEMS_PER_THREAD     <b>[inferred]</b> The number of consecutive items partitioned onto each thread.
 * \tparam OutputIterator       <b>[inferred]</b> The output iterator type (may be a simple pointer).
 *
 * The aggregate tile of items is assumed to be partitioned evenly across
 * threads in <em>blocked</em> arrangement with thread<sub><em>i</em></sub> owning
 * the <em>i</em><sup>th</sup> segment of consecutive elements.
 */
template <
    PtxStoreModifier MODIFIER,
    typename        T,
    int             ITEMS_PER_THREAD,
    typename        OutputIterator>
__device__ __forceinline__ void BlockStoreDirect(
    OutputIterator  block_itr,                        ///< [in] The threadblock's base output iterator for storing to
    T               (&items)[ITEMS_PER_THREAD])     ///< [in] Data to store
{
    // Store directly in thread-blocked order
    #pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
    {
        int item_offset = (threadIdx.x * ITEMS_PER_THREAD) + ITEM;
        ThreadStore<MODIFIER>(block_itr + item_offset, items[ITEM]);
    }
}


/**
 * \brief Store a tile of items across a threadblock directly.
 *
 * \tparam T                    <b>[inferred]</b> The data type to store.
 * \tparam ITEMS_PER_THREAD     <b>[inferred]</b> The number of consecutive items partitioned onto each thread.
 * \tparam OutputIterator       <b>[inferred]</b> The output iterator type (may be a simple pointer).
 *
 * The aggregate tile of items is assumed to be partitioned evenly across
 * threads in <em>blocked</em> arrangement with thread<sub><em>i</em></sub> owning
 * the <em>i</em><sup>th</sup> segment of consecutive elements.
 */
template <
    typename        T,
    int             ITEMS_PER_THREAD,
    typename        OutputIterator>
__device__ __forceinline__ void BlockStoreDirect(
    OutputIterator  block_itr,                        ///< [in] The threadblock's base output iterator for storing to
    T               (&items)[ITEMS_PER_THREAD])     ///< [in] Data to store
{
    BlockStoreDirect<PTX_STORE_NONE>(block_itr, items);
}


/**
 * \brief Store a tile of items across a threadblock directly using the specified cache modifier, guarded by range
 *
 * \tparam MODIFIER             cub::PtxStoreModifier cache modifier.
 * \tparam T                    <b>[inferred]</b> The data type to store.
 * \tparam ITEMS_PER_THREAD     <b>[inferred]</b> The number of consecutive items partitioned onto each thread.
 * \tparam OutputIterator       <b>[inferred]</b> The output iterator type (may be a simple pointer).
 * \tparam SizeT                <b>[inferred]</b> Integer type for offsets
 *
 * The aggregate tile of items is assumed to be partitioned evenly across
 * threads in <em>blocked</em> arrangement with thread<sub><em>i</em></sub> owning
 * the <em>i</em><sup>th</sup> segment of consecutive elements.
 */
template <
    PtxStoreModifier MODIFIER,
    typename        T,
    int             ITEMS_PER_THREAD,
    typename        OutputIterator,
    typename        SizeT>
__device__ __forceinline__ void BlockStoreDirect(
    OutputIterator  block_itr,                        ///< [in] The threadblock's base output iterator for storing to
    const SizeT     &guarded_items,                 ///< [in] Number of valid items in the tile
    T               (&items)[ITEMS_PER_THREAD])     ///< [in] Data to store
{
    // Store directly in thread-blocked order
    #pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
    {
        int item_offset = (threadIdx.x * ITEMS_PER_THREAD) + ITEM;
        if (item_offset < guarded_items)
        {
            ThreadStore<MODIFIER>(block_itr + item_offset, items[ITEM]);
        }
    }
}


/**
 * \brief Store a tile of items across a threadblock directly, guarded by range
 *
 * \tparam T                    <b>[inferred]</b> The data type to store.
 * \tparam ITEMS_PER_THREAD     <b>[inferred]</b> The number of consecutive items partitioned onto each thread.
 * \tparam OutputIterator       <b>[inferred]</b> The output iterator type (may be a simple pointer).
 * \tparam SizeT                <b>[inferred]</b> Integer type for offsets
 *
 * The aggregate tile of items is assumed to be partitioned evenly across
 * threads in <em>blocked</em> arrangement with thread<sub><em>i</em></sub> owning
 * the <em>i</em><sup>th</sup> segment of consecutive elements.
 */
template <
    typename        T,
    int             ITEMS_PER_THREAD,
    typename        OutputIterator,
    typename        SizeT>
__device__ __forceinline__ void BlockStoreDirect(
    OutputIterator  block_itr,                        ///< [in] The threadblock's base output iterator for storing to
    const SizeT     &guarded_items,                 ///< [in] Number of valid items in the tile
    T               (&items)[ITEMS_PER_THREAD])     ///< [in] Data to store
{
    BlockStoreDirect<PTX_STORE_NONE>(block_itr, guarded_items, items);
}


//@}
/******************************************************************//**
 * \name Direct threadblock stores (striped arrangement)
 *********************************************************************/
//@{



/**
 * \brief Store striped tile directly using the specified cache modifier.
 *
 * \tparam MODIFIER             cub::PtxStoreModifier cache modifier.
 * \tparam T                    <b>[inferred]</b> The data type to store.
 * \tparam ITEMS_PER_THREAD     <b>[inferred]</b> The number of consecutive items partitioned onto each thread.
 * \tparam OutputIterator       <b>[inferred]</b> The output iterator type (may be a simple pointer).
 *
 * The aggregate tile of items is assumed to be partitioned across
 * threads in "striped" arrangement, i.e., the \p ITEMS_PER_THREAD
 * items owned by each thread have logical stride \p BLOCK_THREADS between them.
 */
template <
    PtxStoreModifier MODIFIER,
    typename        T,
    int             ITEMS_PER_THREAD,
    typename        OutputIterator>
__device__ __forceinline__ void BlockStoreDirectStriped(
    OutputIterator  block_itr,                        ///< [in] The threadblock's base output iterator for storing to
    T               (&items)[ITEMS_PER_THREAD],     ///< [in] Data to store
    int             stride = blockDim.x)            ///< [in] <b>[optional]</b> Stripe stride.  Default is the width of the threadblock.  More efficient code can be generated if a compile-time-constant (e.g., BLOCK_THREADS) is supplied.
{
    // Store directly in striped order
    #pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
    {
        int item_offset = (ITEM * stride) + threadIdx.x;
        ThreadStore<MODIFIER>(block_itr + item_offset, items[ITEM]);
    }
}


/**
 * \brief Store striped tile directly.
 *
 * \tparam MODIFIER             cub::PtxStoreModifier cache modifier.
 * \tparam T                    <b>[inferred]</b> The data type to store.
 * \tparam ITEMS_PER_THREAD     <b>[inferred]</b> The number of consecutive items partitioned onto each thread.
 * \tparam OutputIterator       <b>[inferred]</b> The output iterator type (may be a simple pointer).
 *
 * The aggregate tile of items is assumed to be partitioned across
 * threads in <em>striped</em> arrangement, i.e., the \p ITEMS_PER_THREAD
 * items owned by each thread have logical stride \p BLOCK_THREADS between them.
 */
template <
    typename        T,
    int             ITEMS_PER_THREAD,
    typename        OutputIterator>
__device__ __forceinline__ void BlockStoreDirectStriped(
    OutputIterator  block_itr,                        ///< [in] The threadblock's base output iterator for storing to
    T               (&items)[ITEMS_PER_THREAD],     ///< [in] Data to store
    int             stride = blockDim.x)            ///< [in] <b>[optional]</b> Stripe stride.  Default is the width of the threadblock.  More efficient code can be generated if a compile-time-constant (e.g., BLOCK_THREADS) is supplied.
{
    BlockStoreDirectStriped<PTX_STORE_NONE>(block_itr, items, stride);
}


/**
 * Store striped directly tile using the specified cache modifier, guarded by range
 *
 * \tparam MODIFIER             cub::PtxStoreModifier cache modifier.
 * \tparam T                    <b>[inferred]</b> The data type to store.
 * \tparam ITEMS_PER_THREAD     <b>[inferred]</b> The number of consecutive items partitioned onto each thread.
 * \tparam OutputIterator       <b>[inferred]</b> The output iterator type (may be a simple pointer).
 * \tparam SizeT                <b>[inferred]</b> Integer type for offsets
 *
 * The aggregate tile of items is assumed to be partitioned across
 * threads in <em>striped</em> arrangement, i.e., the \p ITEMS_PER_THREAD
 * items owned by each thread have logical stride \p BLOCK_THREADS between them.
 */
template <
    PtxStoreModifier MODIFIER,
    typename        T,
    int             ITEMS_PER_THREAD,
    typename        OutputIterator,
    typename        SizeT>
__device__ __forceinline__ void BlockStoreDirectStriped(
    OutputIterator  block_itr,                        ///< [in] The threadblock's base output iterator for storing to
    const SizeT     &guarded_items,                 ///< [in] Number of valid items in the tile
    T               (&items)[ITEMS_PER_THREAD],     ///< [in] Data to store
    int             stride = blockDim.x)            ///< [in] <b>[optional]</b> Stripe stride.  Default is the width of the threadblock.  More efficient code can be generated if a compile-time-constant (e.g., BLOCK_THREADS) is supplied.
{
    // Store directly in striped order
    #pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
    {
        int item_offset = (ITEM * stride) + threadIdx.x;
        if (item_offset < guarded_items)
        {
            ThreadStore<MODIFIER>(block_itr + item_offset, items[ITEM]);
        }
    }
}


/**
 * \brief Store striped tile directly, guarded by range
 *
 * \tparam T                    <b>[inferred]</b> The data type to store.
 * \tparam ITEMS_PER_THREAD     <b>[inferred]</b> The number of consecutive items partitioned onto each thread.
 * \tparam OutputIterator       <b>[inferred]</b> The output iterator type (may be a simple pointer).
 * \tparam SizeT                <b>[inferred]</b> Integer type for offsets
 *
 * The aggregate tile of items is assumed to be partitioned across
 * threads in <em>striped</em> arrangement, i.e., the \p ITEMS_PER_THREAD
 * items owned by each thread have logical stride \p BLOCK_THREADS between them.
 */
template <
    typename        T,
    int             ITEMS_PER_THREAD,
    typename        OutputIterator,
    typename        SizeT>
__device__ __forceinline__ void BlockStoreDirectStriped(
    OutputIterator  block_itr,                        ///< [in] The threadblock's base output iterator for storing to
    const SizeT     &guarded_items,                 ///< [in] Number of valid items in the tile
    T               (&items)[ITEMS_PER_THREAD],     ///< [in] Data to store
    int             stride = blockDim.x)            ///< [in] <b>[optional]</b> Stripe stride.  Default is the width of the threadblock.  More efficient code can be generated if a compile-time-constant (e.g., BLOCK_THREADS) is supplied.
{
    BlockStoreDirectStriped<PTX_STORE_NONE>(block_itr, guarded_items, items, stride);
}


//@}
/******************************************************************//**
 * \name Threadblock vectorized stores (blocked arrangement)
 *********************************************************************/
//@{

/**
 * \brief Store a tile of items across a threadblock directly using the specified cache modifier.
 *
 * \tparam MODIFIER             cub::PtxStoreModifier cache modifier.
 * \tparam T                    <b>[inferred]</b> The data type to store.
 * \tparam ITEMS_PER_THREAD     <b>[inferred]</b> The number of consecutive items partitioned onto each thread.
 *
 * The aggregate tile of items is assumed to be partitioned evenly across
 * threads in <em>blocked</em> arrangement with thread<sub><em>i</em></sub> owning
 * the <em>i</em><sup>th</sup> segment of consecutive elements.
 *
 * \par
 * The following conditions will prevent vectorization and storing will fall back to cub::BLOCK_STORE_DIRECT:
 *   - \p ITEMS_PER_THREAD is odd
 *   - The \p OutputIterator is not a simple pointer type
 *   - The input offset (\p block_ptr + \p block_offset) is not quad-aligned
 *   - The data type \p T is not a built-in primitive or CUDA vector type (e.g., \p short, \p int2, \p double, \p float2, etc.)
 */
template <
    PtxStoreModifier MODIFIER,
    typename        T,
    int             ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockStoreVectorized(
    T               *block_ptr,                       ///< [in] Input pointer for storing from
    T               (&items)[ITEMS_PER_THREAD])     ///< [in] Data to store
{
    enum
    {
        // Maximum CUDA vector size is 4 elements
        MAX_VEC_SIZE = CUB_MIN(4, ITEMS_PER_THREAD),

        // Vector size must be a power of two and an even divisor of the items per thread
        VEC_SIZE = ((((MAX_VEC_SIZE - 1) & MAX_VEC_SIZE) == 0) && ((ITEMS_PER_THREAD % MAX_VEC_SIZE) == 0)) ?
            MAX_VEC_SIZE :
            1,

        VECTORS_PER_THREAD     = ITEMS_PER_THREAD / VEC_SIZE,
    };

    // Vector type
    typedef typename VectorType<T, VEC_SIZE>::Type Vector;

    // Alias global pointer
    Vector *block_ptr_vectors = reinterpret_cast<Vector *>(block_ptr);

    // Vectorize if aligned
    if ((size_t(block_ptr_vectors) & (VEC_SIZE - 1)) == 0)
    {
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
        BlockStoreDirect<MODIFIER>(block_ptr_vectors, raw_vector);
    }
    else
    {
        // Unaligned: direct-store of individual items
        BlockStoreDirect<MODIFIER>(block_ptr, items);
    }
}


/**
 * \brief Store a tile of items across a threadblock directly.
 *
 * \tparam T                    <b>[inferred]</b> The data type to store.
 * \tparam ITEMS_PER_THREAD     <b>[inferred]</b> The number of consecutive items partitioned onto each thread.
 *
 * The aggregate tile of items is assumed to be partitioned evenly across
 * threads in <em>blocked</em> arrangement with thread<sub><em>i</em></sub> owning
 * the <em>i</em><sup>th</sup> segment of consecutive elements.
 *
 * \par
 * The following conditions will prevent vectorization and storing will fall back to cub::BLOCK_STORE_DIRECT:
 *   - \p ITEMS_PER_THREAD is odd
 *   - The \p OutputIterator is not a simple pointer type
 *   - The input offset (\p block_ptr + \p block_offset) is not quad-aligned
 *   - The data type \p T is not a built-in primitive or CUDA vector type (e.g., \p short, \p int2, \p double, \p float2, etc.)
 */
template <
    typename        T,
    int             ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockStoreVectorized(
    T               *block_ptr,                       ///< [in] Input pointer for storing from
    T               (&items)[ITEMS_PER_THREAD])     ///< [in] Data to store
{
    BlockStoreVectorized<PTX_STORE_NONE>(block_ptr, items);
}

//@}


/** @} */       // end of SimtUtils group


//-----------------------------------------------------------------------------
// Generic BlockStore abstraction
//-----------------------------------------------------------------------------

/// Tuning policy for cub::BlockStore
enum BlockStorePolicy
{
    BLOCK_STORE_DIRECT,        ///< Stores consecutive thread-items directly from the input
    BLOCK_STORE_VECTORIZE,     ///< Attempts to use CUDA's built-in vectorized items as a coalescing optimization
    BLOCK_STORE_TRANSPOSE,     ///< Stores striped inputs as a coalescing optimization and then transposes them through shared memory into the desired blocks of thread-consecutive items
};



/**
 * \addtogroup SimtCoop
 * @{
 */


/**
 * \brief The BlockStore type provides store operations for writing global tiles of data from the threadblock (in blocked arrangement across threads). ![](block_store_logo.png)
 *
 * <b>Overview</b>
 * \par
 * BlockStore can be configured to use one of three alternative algorithms:
 *   -# <b>cub::BLOCK_STORE_DIRECT</b>.  Stores consecutive thread-items
 *      directly from the input.
 *   <br><br>
 *   -# <b>cub::BLOCK_STORE_VECTORIZE</b>.  Attempts to use CUDA's
 *      built-in vectorized items as a coalescing optimization.  For
 *      example, <tt>st.global.v4.s32</tt> will be generated when
 *      \p T = \p int and \p ITEMS_PER_THREAD > 4.
 *   <br><br>
 *   -# <b>cub::BLOCK_STORE_TRANSPOSE</b>.  Stores striped inputs as
 *      a coalescing optimization and then transposes them through
 *      shared memory into the desired blocks of thread-consecutive items
 *
 * \par
 * The data movement operations exposed by this type assume a blocked
 * arrangement of data amongst threads, i.e., an <em>n</em>-element list (or
 * <em>tile</em>) that is partitioned evenly among \p BLOCK_THREADS threads,
 * with thread<sub><em>i</em></sub> owning the <em>i</em><sup>th</sup> segment of
 * consecutive elements.
 *
 * \tparam OutputIterator        The input iterator type (may be a simple pointer).
 * \tparam BLOCK_THREADS          The threadblock size in threads.
 * \tparam ITEMS_PER_THREAD     The number of consecutive items partitioned onto each thread.
 * \tparam POLICY               <b>[optional]</b> cub::BlockStorePolicy tuning policy enumeration.  Default = cub::BLOCK_STORE_DIRECT.
 * \tparam MODIFIER             <b>[optional]</b> cub::PtxStoreModifier cache modifier.  Default = cub::PTX_STORE_NONE.
 *
 * <b>Performance Features and Considerations</b>
 * \par
 * - After any operation, a subsequent threadblock barrier (<tt>__syncthreads()</tt>) is
 *   required if the supplied BlockStore::SmemStorage is to be reused/repurposed by the threadblock.
 * - The following conditions will prevent vectorization and storing will fall back to cub::BLOCK_STORE_DIRECT:
 *   - \p ITEMS_PER_THREAD is odd
 *   - The \p OutputIterator is not a simple pointer type
 *   - The input offset (\p block_ptr + \p block_offset) is not quad-aligned
 *   - The data type \p T is not a built-in primitive or CUDA vector type (e.g., \p short, \p int2, \p double, \p float2, etc.) *
 *
 * <b>Examples</b>
 * \par
 * - <b>Example 1:</b> Store four consecutive integers per thread:
 * \code
 * #include <cub.cuh>
 *
 * template <int BLOCK_THREADS>
 * __global__ void SomeKernel(int *d_in, ...)
 * {
 *      // Declare a parameterized BlockStore type for the kernel configuration
 *      typedef cub::BlockStore<int, BLOCK_THREADS, 4> BlockStore;
 *
 *      // Declare shared memory for BlockStore
 *      __shared__ typename BlockStore::SmemStorage smem_storage;
 *
 *      // A segment of four input items per thread
 *      int data[4];
 *
 *      // Store a tile of data
 *      BlockStore::Store(data, d_in, blockIdx.x * BLOCK_THREADS * 4);
 *
 *      ...
 * }
 * \endcode
 * - <b>Example 2:</b> Store four consecutive integers per thread using vectorized stores and global-only caching:
 * \code
 * #include <cub.cuh>
 *
 * template <int BLOCK_THREADS>
 * __global__ void SomeKernel(int *d_in, ...)
 * {
 *      const int ITEMS_PER_THREAD = 4;
 *
 *      // Declare a parameterized BlockStore type for the kernel configuration
 *      typedef cub::BlockStore<int, BLOCK_THREADS, 4, BLOCK_STORE_VECTORIZE, PTX_STORE_CG> BlockStore;
 *
 *      // Declare shared memory for BlockStore
 *      __shared__ typename BlockStore::SmemStorage smem_storage;
 *
 *      // A segment of four input items per thread
 *      int data[4];
 *
 *      // Store a tile of data using vector-store instructions if possible
 *      BlockStore::Store(data, d_in, blockIdx.x * BLOCK_THREADS * 4);
 *
 *      ...
 * }
 * \endcode
 * <br>
 */
template <
    typename            OutputIterator,
    int                 BLOCK_THREADS,
    int                 ITEMS_PER_THREAD,
    BlockStorePolicy      POLICY = BLOCK_STORE_DIRECT,
    PtxStoreModifier    MODIFIER = PTX_STORE_NONE>
class BlockStore
{
    //---------------------------------------------------------------------
    // Type definitions and constants
    //---------------------------------------------------------------------

private:

    // Data type of input iterator
    typedef typename std::iterator_traits<OutputIterator>::value_type T;


    /// Store helper
    template <BlockStorePolicy POLICY, int DUMMY = 0>
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
            SmemStorage     &smem_storage,              ///< [in] Shared reference to opaque SmemStorage layout
            OutputIterator  block_itr,                    ///< [in] The threadblock's base output iterator for storing to
            T               (&items)[ITEMS_PER_THREAD]) ///< [in] Data to store
        {
            BlockStoreDirect<MODIFIER>(block_itr, items);
        }

        /// Store a tile of items across a threadblock, guarded by range
        template <typename SizeT>
        static __device__ __forceinline__ void Store(
            SmemStorage     &smem_storage,              ///< [in] Shared reference to opaque SmemStorage layout
            OutputIterator  block_itr,                    ///< [in] The threadblock's base output iterator for storing to
            const SizeT     &guarded_items,             ///< [in] Number of valid items in the tile
            T               (&items)[ITEMS_PER_THREAD]) ///< [in] Data to store
        {
            BlockStoreDirect<PTX_STORE_NONE>(block_itr, guarded_items, items);
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
        template <typename SizeT>
        static __device__ __forceinline__ void Store(
            SmemStorage     &smem_storage,              ///< [in] Shared reference to opaque SmemStorage layout
            T               *block_ptr,                   ///< [in] The threadblock's base output iterator for storing to
            T               (&items)[ITEMS_PER_THREAD]) ///< [in] Data to store
        {
            BlockStoreVectorized<MODIFIER>(block_ptr, items);
        }

        /// Store a tile of items across a threadblock, specialized for opaque input iterators (skips vectorization)
        template <
            typename T,
            typename OutputIterator>
        static __device__ __forceinline__ void Store(
            SmemStorage     &smem_storage,              ///< [in] Shared reference to opaque SmemStorage layout
            OutputIterator  block_itr,                    ///< [in] The threadblock's base output iterator for storing to
            T               (&items)[ITEMS_PER_THREAD]) ///< [in] Data to store
        {
            BlockStoreDirect<MODIFIER>(block_itr, items);
        }

        /// Store a tile of items across a threadblock, guarded by range
        template <typename SizeT>
        static __device__ __forceinline__ void Store(
            SmemStorage     &smem_storage,              ///< [in] Shared reference to opaque SmemStorage layout
            OutputIterator  block_itr,                    ///< [in] The threadblock's base output iterator for storing to
            const SizeT     &guarded_items,             ///< [in] Number of valid items in the tile
            T               (&items)[ITEMS_PER_THREAD]) ///< [in] Data to store
        {
            BlockStoreDirect<PTX_STORE_NONE>(block_itr, guarded_items, items);
        }
    };


    /**
     * BLOCK_STORE_TRANSPOSE specialization of store helper
     */
    template <int DUMMY>
    struct StoreInternal<BLOCK_STORE_TRANSPOSE, DUMMY>
    {
        // BlockExchange utility type for keys
        typedef BlockExchange<T, BLOCK_THREADS, ITEMS_PER_THREAD> BlockExchange;

        /// Shared memory storage layout type
        typedef typename BlockExchange::SmemStorage SmemStorage;

        /// Store a tile of items across a threadblock
        static __device__ __forceinline__ void Store(
            SmemStorage     &smem_storage,              ///< [in] Shared reference to opaque SmemStorage layout
            OutputIterator  block_itr,                    ///< [in] The threadblock's base output iterator for storing to
            T               (&items)[ITEMS_PER_THREAD]) ///< [in] Data to store
        {
            // Transpose to striped order
            BlockExchange::BlockedToStriped(smem_storage, items);

            BlockStoreDirectStriped<MODIFIER>(block_itr, items, BLOCK_THREADS);
        }

        /// Store a tile of items across a threadblock, guarded by range
        template <typename SizeT>
        static __device__ __forceinline__ void Store(
            SmemStorage     &smem_storage,              ///< [in] Shared reference to opaque SmemStorage layout
            OutputIterator  block_itr,                    ///< [in] The threadblock's base output iterator for storing to
            const SizeT     &guarded_items,             ///< [in] Number of valid items in the tile
            T               (&items)[ITEMS_PER_THREAD]) ///< [in] Data to store
        {
            // Transpose to striped order
            BlockExchange::BlockedToStriped(smem_storage, items);

            BlockStoreDirectStriped<PTX_STORE_NONE>(block_itr, guarded_items, items, BLOCK_THREADS);
        }

    };

    /// Shared memory storage layout type
    typedef typename StoreInternal<POLICY>::SmemStorage SmemLayout;

public:

    /// The operations exposed by BlockStore require shared memory of this
    /// type.  This opaque storage can be allocated directly using the
    /// <tt>__shared__</tt> keyword.  Alternatively, it can be aliased to
    /// externally allocated shared memory or <tt>union</tt>'d with other types
    /// to facilitate shared memory reuse.
    typedef SmemLayout SmemStorage;



    //---------------------------------------------------------------------
    // Interface
    //---------------------------------------------------------------------

    /**
     * \brief Store a tile of items across a threadblock.
     */
    static __device__ __forceinline__ void Store(
        SmemStorage     &smem_storage,              ///< [in] Shared reference to opaque SmemStorage layout
        OutputIterator  block_itr,                    ///< [in] The threadblock's base output iterator for storing to
        T               (&items)[ITEMS_PER_THREAD]) ///< [in] Data to store
    {
        StoreInternal<POLICY>::Store(smem_storage, block_itr, items);
    }

    /**
     * \brief Store a tile of items across a threadblock, guarded by range.
     *
     * \tparam SizeT                <b>[inferred]</b> Integer type for offsets
     */
    template <typename SizeT>
    static __device__ __forceinline__ void Store(
        SmemStorage     &smem_storage,              ///< [in] Shared reference to opaque SmemStorage layout
        OutputIterator  block_itr,                    ///< [in] The threadblock's base output iterator for storing to
        const SizeT     &guarded_items,             ///< [in] Number of valid items in the tile
        T               (&items)[ITEMS_PER_THREAD]) ///< [in] Data to store
    {
        StoreInternal<POLICY>::Store(smem_storage, block_itr, guarded_items, items);
    }
};

/** @} */       // end of SimtCoop group

} // namespace cub
CUB_NS_POSTFIX
