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
 * Operations for reading global tiles of data into the threadblock (in blocked arrangement across threads).
 */

#pragma once

#include <iterator>

#include "../ns_wrapper.cuh"
#include "../macro_utils.cuh"
#include "../thread/thread_load.cuh"
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
 * \name Direct threadblock loads (blocked arrangement)
 *********************************************************************/
//@{

/**
 * \brief Load a tile of items across a threadblock directly using the specified cache modifier.
 *
 * The aggregate tile of items is assumed to be partitioned evenly across
 * threads in "blocked" fashion with thread<sub><em>i</em></sub> owning
 * the <em>i</em><sup>th</sup> segment of consecutive elements.
 *
 * \tparam MODIFIER             cub::PtxLoadModifier cache modifier.
 * \tparam T                    <b>[inferred]</b> The data type to load.
 * \tparam ITEMS_PER_THREAD     <b>[inferred]</b> The number of consecutive items partitioned onto each thread.
 * \tparam InputIterator        <b>[inferred]</b> The input iterator type (may be a simple pointer type).
 */
template <
    PtxLoadModifier MODIFIER,
    typename        T,
    int             ITEMS_PER_THREAD,
    typename        InputIterator>
__device__ __forceinline__ void BlockLoadDirect(
    InputIterator   block_itr,                        ///< [in] The threadblock's base input iterator for loading from
    T               (&items)[ITEMS_PER_THREAD])     ///< [out] Data to load
{
    // Load directly in thread-blocked order
    #pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
    {
        int item_offset = (threadIdx.x * ITEMS_PER_THREAD) + ITEM;
        items[ITEM] = ThreadLoad<MODIFIER>(block_itr + item_offset);
    }
}


/**
 * \brief Load a tile of items across a threadblock directly.
 *
 * The aggregate tile of items is assumed to be partitioned evenly across
 * threads in "blocked" fashion with thread<sub><em>i</em></sub> owning
 * the <em>i</em><sup>th</sup> segment of consecutive elements.
 *
 * \tparam T                    <b>[inferred]</b> The data type to load.
 * \tparam ITEMS_PER_THREAD     <b>[inferred]</b> The number of consecutive items partitioned onto each thread.
 * \tparam InputIterator        <b>[inferred]</b> The input iterator type (may be a simple pointer type).
 */
template <
    typename        T,
    int             ITEMS_PER_THREAD,
    typename        InputIterator>
__device__ __forceinline__ void BlockLoadDirect(
    InputIterator   block_itr,                        ///< [in] The threadblock's base input iterator for loading from
    T               (&items)[ITEMS_PER_THREAD])     ///< [out] Data to load
{
    BlockLoadDirect<PTX_LOAD_NONE>(block_itr, items);
}



/**
 * \brief Load a tile of items across a threadblock directly using the specified cache modifier, guarded by range
 *
 * The aggregate tile of items is assumed to be partitioned evenly across
 * threads in "blocked" fashion with thread<sub><em>i</em></sub> owning
 * the <em>i</em><sup>th</sup> segment of consecutive elements.
 *
 * \tparam MODIFIER             cub::PtxLoadModifier cache modifier.
 * \tparam T                    <b>[inferred]</b> The data type to load.
 * \tparam ITEMS_PER_THREAD     <b>[inferred]</b> The number of consecutive items partitioned onto each thread.
 * \tparam InputIterator        <b>[inferred]</b> The input iterator type (may be a simple pointer type).
 * \tparam SizeT                <b>[inferred]</b> Integer type for offsets
 */
template <
    PtxLoadModifier MODIFIER,
    typename        T,
    int             ITEMS_PER_THREAD,
    typename        InputIterator,
    typename        SizeT>
__device__ __forceinline__ void BlockLoadDirect(
    InputIterator   block_itr,                        ///< [in] The threadblock's base input iterator for loading from
    const SizeT     &guarded_items,                 ///< [in] Number of valid items in the tile
    T               (&items)[ITEMS_PER_THREAD])     ///< [out] Data to load
{
    // Load directly in thread-blocked order
    #pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
    {
        int item_offset = (threadIdx.x * ITEMS_PER_THREAD) + ITEM;
        if (item_offset < guarded_items)
        {
            items[ITEM] = ThreadLoad<MODIFIER>(block_itr + item_offset);
        }
    }
}


/**
 * \brief Load a tile of items across a threadblock directly, guarded by range
 *
 * The aggregate tile of items is assumed to be partitioned evenly across
 * threads in "blocked" fashion with thread<sub><em>i</em></sub> owning
 * the <em>i</em><sup>th</sup> segment of consecutive elements.
 *
 * \tparam T                    <b>[inferred]</b> The data type to load.
 * \tparam ITEMS_PER_THREAD     <b>[inferred]</b> The number of consecutive items partitioned onto each thread.
 * \tparam InputIterator        <b>[inferred]</b> The input iterator type (may be a simple pointer type).
 * \tparam SizeT                <b>[inferred]</b> Integer type for offsets
 */
template <
    typename        T,
    int             ITEMS_PER_THREAD,
    typename        InputIterator,
    typename        SizeT>
__device__ __forceinline__ void BlockLoadDirect(
    InputIterator   block_itr,                        ///< [in] The threadblock's base input iterator for loading from
    const SizeT     &guarded_items,                 ///< [in] Number of valid items in the tile
    T               (&items)[ITEMS_PER_THREAD])     ///< [out] Data to load
{
    BlockLoadDirect<PTX_LOAD_NONE>(block_itr, guarded_items, items);
}


/**
 * \brief Load a tile of items across a threadblock directly using the specified cache modifier, guarded by range, with assignment for out-of-bound elements
 *
 * The aggregate tile of items is assumed to be partitioned evenly across
 * threads in "blocked" fashion with thread<sub><em>i</em></sub> owning
 * the <em>i</em><sup>th</sup> segment of consecutive elements.
 *
 * \tparam MODIFIER             cub::PtxLoadModifier cache modifier.
 * \tparam T                    <b>[inferred]</b> The data type to load.
 * \tparam ITEMS_PER_THREAD     <b>[inferred]</b> The number of consecutive items partitioned onto each thread.
 * \tparam InputIterator        <b>[inferred]</b> The input iterator type (may be a simple pointer type).
 * \tparam SizeT                <b>[inferred]</b> Integer type for offsets
 */
template <
    PtxLoadModifier MODIFIER,
    typename        T,
    int             ITEMS_PER_THREAD,
    typename        InputIterator,
    typename        SizeT>
__device__ __forceinline__ void BlockLoadDirect(
    InputIterator   block_itr,                        ///< [in] The threadblock's base input iterator for loading from
    const SizeT     &guarded_items,                 ///< [in] Number of valid items in the tile
    T               oob_default,                    ///< [in] Default value to assign out-of-bound items
    T               (&items)[ITEMS_PER_THREAD])     ///< [out] Data to load
{
    // Load directly in thread-blocked order
    #pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
    {
        int item_offset = (threadIdx.x * ITEMS_PER_THREAD) + ITEM;
        items[ITEM] =  (item_offset < guarded_items) ?
            ThreadLoad<MODIFIER>(block_itr + item_offset) :
            oob_default;
    }
}


/**
 * \brief Load a tile of items across a threadblock directly, guarded by range, with assignment for out-of-bound elements
 *
 * The aggregate tile of items is assumed to be partitioned evenly across
 * threads in "blocked" fashion with thread<sub><em>i</em></sub> owning
 * the <em>i</em><sup>th</sup> segment of consecutive elements.
 *
 * \tparam T                    <b>[inferred]</b> The data type to load.
 * \tparam ITEMS_PER_THREAD     <b>[inferred]</b> The number of consecutive items partitioned onto each thread.
 * \tparam InputIterator        <b>[inferred]</b> The input iterator type (may be a simple pointer type).
 * \tparam SizeT                <b>[inferred]</b> Integer type for offsets
 */
template <
    typename        T,
    int             ITEMS_PER_THREAD,
    typename        InputIterator,
    typename        SizeT>
__device__ __forceinline__ void BlockLoadDirect(
    InputIterator   block_itr,                        ///< [in] The threadblock's base input iterator for loading from
    const SizeT     &guarded_items,                 ///< [in] Number of valid items in the tile
    T               oob_default,                    ///< [in] Default value to assign out-of-bound items
    T               (&items)[ITEMS_PER_THREAD])     ///< [out] Data to load
{
    BlockLoadDirect<PTX_LOAD_NONE>(block_itr, guarded_items, oob_default, items);
}


//@}
/******************************************************************//**
 * \name Direct threadblock loads (striped arrangement)
 *********************************************************************/
//@{


/**
 * \brief Load striped tile directly using the specified cache modifier.
 *
 * The aggregate tile of items is assumed to be partitioned across
 * threads in "striped" fashion, i.e., the \p ITEMS_PER_THREAD
 * items owned by each thread have logical stride \p BLOCK_THREADS between them.
 *
 * \tparam MODIFIER             cub::PtxLoadModifier cache modifier.
 * \tparam T                    <b>[inferred]</b> The data type to load.
 * \tparam ITEMS_PER_THREAD     <b>[inferred]</b> The number of consecutive items partitioned onto each thread.
 * \tparam InputIterator        <b>[inferred]</b> The input iterator type (may be a simple pointer type).
 */
template <
    PtxLoadModifier MODIFIER,
    typename        T,
    int             ITEMS_PER_THREAD,
    typename        InputIterator>
__device__ __forceinline__ void BlockLoadDirectStriped(
    InputIterator   block_itr,                        ///< [in] The threadblock's base input iterator for loading from
    T               (&items)[ITEMS_PER_THREAD],     ///< [out] Data to load
    int             stride = blockDim.x)            ///< [in] <b>[optional]</b> Stripe stride.  Default is the width of the threadblock.  More efficient code can be generated if a compile-time-constant (e.g., BLOCK_THREADS) is supplied.
{
    // Load directly in striped order
    #pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
    {
        int item_offset = (ITEM * stride) + threadIdx.x;
        items[ITEM] = ThreadLoad<MODIFIER>(block_itr + item_offset);
    }
}


/**
 * \brief Load striped tile directly.
 *
 * The aggregate tile of items is assumed to be partitioned across
 * threads in "striped" fashion, i.e., the \p ITEMS_PER_THREAD
 * items owned by each thread have logical stride \p BLOCK_THREADS between them.
 *
 * \tparam MODIFIER             cub::PtxLoadModifier cache modifier.
 * \tparam T                    <b>[inferred]</b> The data type to load.
 * \tparam ITEMS_PER_THREAD     <b>[inferred]</b> The number of consecutive items partitioned onto each thread.
 * \tparam InputIterator        <b>[inferred]</b> The input iterator type (may be a simple pointer type).
 */
template <
    typename        T,
    int             ITEMS_PER_THREAD,
    typename        InputIterator>
__device__ __forceinline__ void BlockLoadDirectStriped(
    InputIterator   block_itr,                        ///< [in] The threadblock's base input iterator for loading from
    T               (&items)[ITEMS_PER_THREAD],     ///< [out] Data to load
    int             stride = blockDim.x)            ///< [in] <b>[optional]</b> Stripe stride.  Default is the width of the threadblock.  More efficient code can be generated if a compile-time-constant (e.g., BLOCK_THREADS) is supplied.
{
    BlockLoadDirectStriped<PTX_LOAD_NONE>(block_itr, items, stride);
}

/**
 * \brief Load striped directly tile using the specified cache modifier, guarded by range
 *
 * The aggregate tile of items is assumed to be partitioned across
 * threads in "striped" fashion, i.e., the \p ITEMS_PER_THREAD
 * items owned by each thread have logical stride \p BLOCK_THREADS between them.
 *
 * \tparam BLOCK_THREADS          The threadblock size in threads
 * \tparam MODIFIER             cub::PtxLoadModifier cache modifier.
 * \tparam T                    <b>[inferred]</b> The data type to load.
 * \tparam ITEMS_PER_THREAD     <b>[inferred]</b> The number of consecutive items partitioned onto each thread.
 * \tparam InputIterator        <b>[inferred]</b> The input iterator type (may be a simple pointer type).
 * \tparam SizeT                <b>[inferred]</b> Integer type for offsets
 */
template <
    PtxLoadModifier MODIFIER,
    typename        T,
    int             ITEMS_PER_THREAD,
    typename        InputIterator,
    typename        SizeT>
__device__ __forceinline__ void BlockLoadDirectStriped(
    InputIterator   block_itr,                        ///< [in] The threadblock's base input iterator for loading from
    const SizeT     &guarded_items,                 ///< [in] Number of valid items in the tile
    T               (&items)[ITEMS_PER_THREAD],     ///< [out] Data to load
    int             stride = blockDim.x)            ///< [in] <b>[optional]</b> Stripe stride.  Default is the width of the threadblock.  More efficient code can be generated if a compile-time-constant (e.g., BLOCK_THREADS) is supplied.
{
    // Load directly in striped order
    #pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
    {
        int item_offset = (ITEM * stride) + threadIdx.x;
        if (item_offset < guarded_items)
        {
            items[ITEM] = ThreadLoad<MODIFIER>(block_itr + item_offset);
        }
    }
}


/**
 * \brief Load striped tile directly, guarded by range
 *
 * The aggregate tile of items is assumed to be partitioned across
 * threads in "striped" fashion, i.e., the \p ITEMS_PER_THREAD
 * items owned by each thread have logical stride \p BLOCK_THREADS between them.
 *
 * \tparam T                    <b>[inferred]</b> The data type to load.
 * \tparam ITEMS_PER_THREAD     <b>[inferred]</b> The number of consecutive items partitioned onto each thread.
 * \tparam InputIterator        <b>[inferred]</b> The input iterator type (may be a simple pointer type).
 * \tparam SizeT                <b>[inferred]</b> Integer type for offsets
 */
template <
    typename        T,
    int             ITEMS_PER_THREAD,
    typename        InputIterator,
    typename        SizeT>
__device__ __forceinline__ void BlockLoadDirectStriped(
    InputIterator   block_itr,                        ///< [in] The threadblock's base input iterator for loading from
    const SizeT     &guarded_items,                 ///< [in] Number of valid items in the tile
    T               (&items)[ITEMS_PER_THREAD],     ///< [out] Data to load
    int             stride = blockDim.x)            ///< [in] <b>[optional]</b> Stripe stride.  Default is the width of the threadblock.  More efficient code can be generated if a compile-time-constant (e.g., BLOCK_THREADS) is supplied.
{
    BlockLoadDirectStriped<PTX_LOAD_NONE>(block_itr, guarded_items, items, stride);
}


/**
 * \brief Load striped directly tile using the specified cache modifier, guarded by range, with assignment for out-of-bound elements
 *
 * The aggregate tile of items is assumed to be partitioned across
 * threads in "striped" fashion, i.e., the \p ITEMS_PER_THREAD
 * items owned by each thread have logical stride \p BLOCK_THREADS between them.
 *
 * \tparam MODIFIER             cub::PtxLoadModifier cache modifier.
 * \tparam T                    <b>[inferred]</b> The data type to load.
 * \tparam ITEMS_PER_THREAD     <b>[inferred]</b> The number of consecutive items partitioned onto each thread.
 * \tparam InputIterator        <b>[inferred]</b> The input iterator type (may be a simple pointer type).
 * \tparam SizeT                <b>[inferred]</b> Integer type for offsets
 */
template <
    PtxLoadModifier MODIFIER,
    typename        T,
    int             ITEMS_PER_THREAD,
    typename        InputIterator,
    typename        SizeT>
__device__ __forceinline__ void BlockLoadDirectStriped(
    InputIterator   block_itr,                        ///< [in] The threadblock's base input iterator for loading from
    const SizeT     &guarded_items,                 ///< [in] Number of valid items in the tile
    T               oob_default,                    ///< [in] Default value to assign out-of-bound items
    T               (&items)[ITEMS_PER_THREAD],     ///< [out] Data to load
    int             stride = blockDim.x)            ///< [in] <b>[optional]</b> Stripe stride.  Default is the width of the threadblock.  More efficient code can be generated if a compile-time-constant (e.g., BLOCK_THREADS) is supplied.
{
    // Load directly in striped order
    #pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
    {
        int item_offset = (ITEM * stride) + threadIdx.x;
        items[ITEM] = (item_offset < guarded_items) ?
             ThreadLoad<MODIFIER>(block_itr + item_offset) :
             oob_default;
    }
}


/**
 * \brief Load striped tile directly, guarded by range, with assignment for out-of-bound elements
 *
 * The aggregate tile of items is assumed to be partitioned across
 * threads in "striped" fashion, i.e., the \p ITEMS_PER_THREAD
 * items owned by each thread have logical stride \p BLOCK_THREADS between them.
 *
 * \tparam T                    <b>[inferred]</b> The data type to load.
 * \tparam ITEMS_PER_THREAD     <b>[inferred]</b> The number of consecutive items partitioned onto each thread.
 * \tparam InputIterator        <b>[inferred]</b> The input iterator type (may be a simple pointer type).
 * \tparam SizeT                <b>[inferred]</b> Integer type for offsets
 */
template <
    typename        T,
    int             ITEMS_PER_THREAD,
    typename        InputIterator,
    typename        SizeT>
__device__ __forceinline__ void BlockLoadDirectStriped(
    InputIterator   block_itr,                        ///< [in] The threadblock's base input iterator for loading from
    const SizeT     &guarded_items,                 ///< [in] Number of valid items in the tile
    T               oob_default,                    ///< [in] Default value to assign out-of-bound items
    T               (&items)[ITEMS_PER_THREAD],     ///< [out] Data to load
    int             stride = blockDim.x)            ///< [in] <b>[optional]</b> Stripe stride.  Default is the width of the threadblock.  More efficient code can be generated if a compile-time-constant (e.g., BLOCK_THREADS) is supplied.
{
    BlockLoadDirectStriped<PTX_LOAD_NONE>(block_itr, guarded_items, oob_default, items, stride);
}

//@}
/******************************************************************//**
 * \name Threadblock vectorized loads (blocked arrangement)
 *********************************************************************/
//@{

/**
 * \brief Load a tile of items across a threadblock directly using the specified cache modifier.
 *
 * The aggregate tile of items is assumed to be partitioned evenly across
 * threads in "blocked" fashion with thread<sub><em>i</em></sub> owning
 * the <em>i</em><sup>th</sup> segment of consecutive elements.
 *
 * The input offset (\p block_ptr + \p block_offset) must be quad-item aligned
 *
 * The following conditions will prevent vectorization and loading will fall back to cub::BLOCK_LOAD_DIRECT:
 *   - \p ITEMS_PER_THREAD is odd
 *   - The data type \p T is not a built-in primitive or CUDA vector type (e.g., \p short, \p int2, \p double, \p float2, etc.)
 *
 * \tparam MODIFIER             cub::PtxLoadModifier cache modifier.
 * \tparam T                    <b>[inferred]</b> The data type to load.
 * \tparam ITEMS_PER_THREAD     <b>[inferred]</b> The number of consecutive items partitioned onto each thread.
 * \tparam SizeT                <b>[inferred]</b> Integer type for offsets
 */
template <
    PtxLoadModifier MODIFIER,
    typename        T,
    int             ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockLoadVectorized(
    T               *block_ptr,                       ///< [in] Input pointer for loading from
    T               (&items)[ITEMS_PER_THREAD])     ///< [out] Data to load
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
    typedef typename VectorType<T, VEC_SIZE>::Type Vector;

    // Alias local data (use raw_items array here which should get optimized away to prevent conservative PTXAS lmem spilling)
    T raw_items[ITEMS_PER_THREAD];

    // Direct-load using vector types
    BlockLoadDirect<MODIFIER>(
        reinterpret_cast<Vector *>(block_ptr),
        reinterpret_cast<Vector (&)[VECTORS_PER_THREAD]>(raw_items));

    // Copy
    #pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
    {
        items[ITEM] = raw_items[ITEM];
    }
}



/**
 * \brief Load a tile of items across a threadblock directly.
 *
 * The aggregate tile of items is assumed to be partitioned evenly across
 * threads in "blocked" fashion with thread<sub><em>i</em></sub> owning
 * the <em>i</em><sup>th</sup> segment of consecutive elements.
 *
 * The input offset (\p block_ptr + \p block_offset) must be quad-item aligned
 *
 * The following conditions will prevent vectorization and loading will fall back to cub::BLOCK_LOAD_DIRECT:
 *   - \p ITEMS_PER_THREAD is odd
 *   - The data type \p T is not a built-in primitive or CUDA vector type (e.g., \p short, \p int2, \p double, \p float2, etc.)
 *
 * \tparam T                    <b>[inferred]</b> The data type to load.
 * \tparam ITEMS_PER_THREAD     <b>[inferred]</b> The number of consecutive items partitioned onto each thread.
 */
template <
    typename        T,
    int             ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockLoadVectorized(
    T               *block_ptr,                       ///< [in] Input pointer for loading from
    T               (&items)[ITEMS_PER_THREAD])     ///< [out] Data to load
{
    BlockLoadVectorized<PTX_LOAD_NONE>(block_ptr, items);
}

//@}

/** @} */       // end of SimtUtils group



//-----------------------------------------------------------------------------
// Generic BlockLoad abstraction
//-----------------------------------------------------------------------------

/// Tuning policy for cub::BlockLoad
enum BlockLoadPolicy
{
    /**
     * \par Overview
     *
     * A [<em>blocked arrangement</em>](index.html#sec3sec3) of data is read
     * directly from memory.  The threadblock reads items in a parallel "raking" fashion: thread<sub><em>i</em></sub>
     * reads the <em>i</em><sup>th</sup> segment of consecutive elements.
     *
     * \par Performance Considerations
     * - The utilization of memory transactions (coalescing) decreases as the
     *   access stride between threads increases (i.e., the number items per thread).
     */
    BLOCK_LOAD_DIRECT,

    /**
     * \par Overview
     *
     * A [<em>blocked arrangement</em>](index.html#sec3sec3) of data is read directly
     * from memory using CUDA's built-in vectorized loads as a coalescing optimization.
     * The threadblock reads items in a parallel "raking" fashion: thread<sub><em>i</em></sub> uses vector loads to
     * read the <em>i</em><sup>th</sup> segment of consecutive elements.
     *
     * For example, <tt>ld.global.v4.s32</tt> instructions will be generated when \p T = \p int and \p ITEMS_PER_THREAD > 4.
     *
     * \par Performance Considerations
     * - The utilization of memory transactions (coalescing) remains high until the the
     *   access stride between threads (i.e., the number items per thread) exceeds the
     *   maximum vector load width (typically 4 items or 64B, whichever is lower).
     * - The following conditions will prevent vectorization and loading will fall back to cub::BLOCK_LOAD_DIRECT:
     *   - \p ITEMS_PER_THREAD is odd
     *   - The \p InputIterator is not a simple pointer type
     *   - The block input offset is not quadword-aligned
     *   - The data type \p T is not a built-in primitive or CUDA vector type (e.g., \p short, \p int2, \p double, \p float2, etc.)
     */
    BLOCK_LOAD_VECTORIZE,

    /**
     * \par Overview
     *
     * A [<em>striped arrangement</em>](index.html#sec3sec3) of data is read directly from memory
     * and then is locally transposed into a [<em>blocked arrangement</em>](index.html#sec3sec3).
     * The threadblock reads items in a parallel "strip-mining" fashion: thread<sub><em>i</em></sub> reads items
     * having stride \p BLOCK_THREADS between them. cub::BlockExchange is then used to
     * locally reorder the items into a [<em>blocked arrangement</em>](index.html#sec3sec3).
     *
     * \par Performance Considerations
     * - The utilization of memory transactions (coalescing) remains high regardless
     *   of items loaded per thread.
     * - The local reordering incurs slightly longer latencies and throughput than the
     *   direct cub::BLOCK_LOAD_DIRECT and cub::BLOCK_LOAD_VECTORIZE alternatives.
     */
    BLOCK_LOAD_TRANSPOSE,


    /**
     * \par Overview
     *
     * A [<em>striped arrangement</em>](index.html#sec3sec3) of data is read directly from memory.
     * The threadblock reads items in a parallel "strip-mining" fashion: thread<sub><em>i</em></sub> reads items
     * having stride \p BLOCK_THREADS between them.
     *
     * \par Performance Considerations
     * - The utilization of memory transactions (coalescing) remains high regardless
     *   of items loaded per thread.
     */
    BLOCK_LOAD_STRIPED,
};


/**
 * \addtogroup SimtCoop
 * @{
 */


/**
 * \brief BlockLoad provides data movement operations for reading [<em>block-arranged</em>](index.html#sec3sec3) data from global memory. ![](block_load_logo.png)
 *
 * BlockLoad provides a single tile-loading abstraction whose performance behavior can be statically tuned.  In particular,
 * BlockLoad implements alternative cub::BlockLoadPolicy strategies catering to different granularity sizes (i.e.,
 * number of items per thread).
 *
 * \tparam InputIterator        The input iterator type (may be a simple pointer type).
 * \tparam BLOCK_THREADS        The threadblock size in threads.
 * \tparam ITEMS_PER_THREAD     The number of consecutive items partitioned onto each thread.
 * \tparam POLICY               <b>[optional]</b> cub::BlockLoadPolicy tuning policy.  Default = cub::BLOCK_LOAD_DIRECT.
 * \tparam MODIFIER             <b>[optional]</b> cub::PtxLoadModifier cache modifier.  Default = cub::PTX_LOAD_NONE.
 *
 * \par Algorithm
 * BlockLoad can be (optionally) configured to use one of three alternative methods:
 *   -# <b>cub::BLOCK_LOAD_DIRECT</b>.  A [<em>blocked arrangement</em>](index.html#sec3sec3)
 *      of data is read directly from memory.  [More...](\ref cub::BlockLoadPolicy)
 *   -# <b>cub::BLOCK_LOAD_VECTORIZE</b>.  A [<em>blocked arrangement</em>](index.html#sec3sec3)
 *      of data is read directly from memory using CUDA's built-in vectorized loads as a
 *      coalescing optimization.    [More...](\ref cub::BlockLoadPolicy)
 *   -# <b>cub::BLOCK_LOAD_TRANSPOSE</b>.  A [<em>striped arrangement</em>](index.html#sec3sec3)
 *      of data is read directly from memory and is then locally transposed into a
 *      [<em>blocked arrangement</em>](index.html#sec3sec3).  [More...](\ref cub::BlockLoadPolicy)
 *   -# <b>cub::BLOCK_LOAD_STRIPED</b>.  A [<em>striped arrangement</em>](index.html#sec3sec3)
 *      of data is read directly from memory.  [More...](\ref cub::BlockLoadPolicy)
 *
 * \par Usage Considerations
 * - \smemreuse{BlockLoad::SmemStorage}
 *
 * \par Performance Considerations
 *  - See cub::BlockLoadPolicy for more performance details regarding algorithmic alternatives
 *
 *
 * \par Examples
 * <em>Example 1.</em> Have a 128-thread threadblock directly load a blocked arrangement of four consecutive integers per thread.
 * \code
 * #include <cub.cuh>
 *
 * __global__ void SomeKernel(int *d_in, ...)
 * {
 *     // Parameterize BlockLoad for the parallel execution context
 *     typedef cub::BlockLoad<int*, 128, 4> BlockLoad;
 *
 *     // Declare shared memory for BlockLoad
 *     __shared__ typename BlockLoad::SmemStorage smem_storage;
 *
 *     // A segment of consecutive items per thread
 *     int data[4];
 *
 *     // Load a tile of data at this block's offset
 *     BlockLoad::Load(smem_storage, d_in + blockIdx.x * 128 * 4, data);
 *
 *     ...
 * \endcode
 *
 * \par
 * <em>Example 2.</em> Have a threadblock load a blocked arrangement of \p ITEMS_PER_THREAD consecutive
 * integers per thread using vectorized loads and global-only caching:
 * \code
 * #include <cub.cuh>
 *
 * template <
 *     int BLOCK_THREADS,
 *     int ITEMS_PER_THREAD>
 * __global__ void SomeKernel(int *d_in, ...)
 * {
 *     // Parameterize BlockLoad for the parallel execution context
 *     typedef cub::BlockLoad<int*, BLOCK_THREADS, ITEMS_PER_THREAD, BLOCK_LOAD_VECTORIZE, PTX_LOAD_CG> BlockLoad;
 *
 *     // Declare shared memory for BlockLoad
 *     __shared__ typename BlockLoad::SmemStorage smem_storage;
 *
 *     // A segment of consecutive items per thread
 *     int data[ITEMS_PER_THREAD];
 *
 *     // Load a tile of data at this block's offset
 *     BlockLoad::Load(smem_storage, d_in + blockIdx.x * BLOCK_THREADS * ITEMS_PER_THREAD, data);
 *
 *     ...
 * \endcode
 * <br>
 */
template <
    typename            InputIterator,
    int                 BLOCK_THREADS,
    int                 ITEMS_PER_THREAD,
    BlockLoadPolicy     POLICY = BLOCK_LOAD_DIRECT,
    PtxLoadModifier     MODIFIER = PTX_LOAD_NONE>
class BlockLoad
{
    //---------------------------------------------------------------------
    // Type definitions and constants
    //---------------------------------------------------------------------

private:

    // Data type of input iterator
    typedef typename std::iterator_traits<InputIterator>::value_type T;


    /// Load helper
    template <BlockLoadPolicy _POLICY, int DUMMY = 0>
    struct LoadInternal;


    /**
     * BLOCK_LOAD_DIRECT specialization of load helper
     */
    template <int DUMMY>
    struct LoadInternal<BLOCK_LOAD_DIRECT, DUMMY>
    {
        /// Shared memory storage layout type
        typedef NullType SmemStorage;

        /// Load a tile of items across a threadblock
        static __device__ __forceinline__ void Load(
            SmemStorage     &smem_storage,                  ///< [in] Shared reference to opaque SmemStorage layout
            InputIterator   block_itr,                      ///< [in] The threadblock's base input iterator for loading from
            T               (&items)[ITEMS_PER_THREAD])     ///< [out] Data to load
        {
            BlockLoadDirect<MODIFIER>(block_itr, items);
        }

        /// Load a tile of items across a threadblock, guarded by range
        template <typename SizeT>
        static __device__ __forceinline__ void Load(
            SmemStorage     &smem_storage,                  ///< [in] Shared reference to opaque SmemStorage layout
            InputIterator   block_itr,                      ///< [in] The threadblock's base input iterator for loading from
            const SizeT     &guarded_items,                 ///< [in] Number of valid items in the tile
            T               (&items)[ITEMS_PER_THREAD])     ///< [out] Data to load
        {
            BlockLoadDirect<PTX_LOAD_NONE>(block_itr, guarded_items, items);
        }
    };


    /**
     * BLOCK_LOAD_VECTORIZE specialization of load helper
     */
    template <int DUMMY>
    struct LoadInternal<BLOCK_LOAD_VECTORIZE, DUMMY>
    {
        /// Shared memory storage layout type
        typedef NullType SmemStorage;

        /// Load a tile of items across a threadblock, specialized for native pointer types (attempts vectorization)
        static __device__ __forceinline__ void Load(
            SmemStorage     &smem_storage,                  ///< [in] Shared reference to opaque SmemStorage layout
            T               *block_ptr,                     ///< [in] The threadblock's base input iterator for loading from
            T               (&items)[ITEMS_PER_THREAD])     ///< [out] Data to load
        {
            BlockLoadVectorized<MODIFIER>(block_ptr, items);
        }

        /// Load a tile of items across a threadblock, specialized for opaque input iterators (skips vectorization)
        template <
            typename T,
            typename _InputIterator>
        static __device__ __forceinline__ void Load(
            SmemStorage     &smem_storage,                  ///< [in] Shared reference to opaque SmemStorage layout
            _InputIterator  block_itr,                      ///< [in] The threadblock's base input iterator for loading from
            T               (&items)[ITEMS_PER_THREAD])     ///< [out] Data to load
        {
            BlockLoadDirect<MODIFIER>(block_itr, items);
        }

        /// Load a tile of items across a threadblock, guarded by range
        template <typename SizeT>
        static __device__ __forceinline__ void Load(
            SmemStorage     &smem_storage,                  ///< [in] Shared reference to opaque SmemStorage layout
            InputIterator   block_itr,                      ///< [in] The threadblock's base input iterator for loading from
            const SizeT     &guarded_items,                 ///< [in] Number of valid items in the tile
            T               (&items)[ITEMS_PER_THREAD])     ///< [out] Data to load{
        {
            BlockLoadDirect<PTX_LOAD_NONE>(block_itr, guarded_items, items);
        }
    };


    /**
     * BLOCK_LOAD_TRANSPOSE specialization of load helper
     */
    template <int DUMMY>
    struct LoadInternal<BLOCK_LOAD_TRANSPOSE, DUMMY>
    {
        // BlockExchange utility type for keys
        typedef BlockExchange<T, BLOCK_THREADS, ITEMS_PER_THREAD> BlockExchange;

        /// Shared memory storage layout type
        typedef typename BlockExchange::SmemStorage SmemStorage;

        /// Load a tile of items across a threadblock
        static __device__ __forceinline__ void Load(
            SmemStorage     &smem_storage,                  ///< [in] Shared reference to opaque SmemStorage layout
            InputIterator   block_itr,                      ///< [in] The threadblock's base input iterator for loading from
            T               (&items)[ITEMS_PER_THREAD])     ///< [out] Data to load{
        {
            BlockLoadDirectStriped<MODIFIER>(block_itr, items, BLOCK_THREADS);

            // Transpose to blocked order
            BlockExchange::StripedToBlocked(smem_storage, items);
        }

        /// Load a tile of items across a threadblock, guarded by range
        template <typename SizeT>
        static __device__ __forceinline__ void Load(
            SmemStorage     &smem_storage,                  ///< [in] Shared reference to opaque SmemStorage layout
            InputIterator   block_itr,                      ///< [in] The threadblock's base input iterator for loading from
            const SizeT     &guarded_items,                 ///< [in] Number of valid items in the tile
            T               (&items)[ITEMS_PER_THREAD])     ///< [out] Data to load{
        {
            BlockLoadDirectStriped<MODIFIER>(block_itr, guarded_items, items, BLOCK_THREADS);

            // Transpose to blocked order
            BlockExchange::StripedToBlocked(smem_storage, items);
        }

    };


    /**
     * BLOCK_LOAD_STRIPED specialization of load helper
     */
    template <int DUMMY>
    struct LoadInternal<BLOCK_LOAD_STRIPED, DUMMY>
    {
        /// Shared memory storage layout type
        typedef NullType SmemStorage;

        /// Load a tile of items across a threadblock
        static __device__ __forceinline__ void Load(
            SmemStorage     &smem_storage,                  ///< [in] Shared reference to opaque SmemStorage layout
            InputIterator   block_itr,                      ///< [in] The threadblock's base input iterator for loading from
            T               (&items)[ITEMS_PER_THREAD])     ///< [out] Data to load{
        {
            BlockLoadDirectStriped<MODIFIER>(block_itr, items, BLOCK_THREADS);
        }

        /// Load a tile of items across a threadblock, guarded by range
        template <typename SizeT>
        static __device__ __forceinline__ void Load(
            SmemStorage     &smem_storage,                  ///< [in] Shared reference to opaque SmemStorage layout
            InputIterator   block_itr,                      ///< [in] The threadblock's base input iterator for loading from
            const SizeT     &guarded_items,                 ///< [in] Number of valid items in the tile
            T               (&items)[ITEMS_PER_THREAD])     ///< [out] Data to load{
        {
            BlockLoadDirectStriped<MODIFIER>(block_itr, guarded_items, items, BLOCK_THREADS);
        }

    };

    /// Shared memory storage layout type
    typedef typename LoadInternal<POLICY>::SmemStorage _SmemStorage;

public:


    /// \smemstorage{BlockLoad}
    typedef _SmemStorage SmemStorage;


    //---------------------------------------------------------------------
    // Interface
    //---------------------------------------------------------------------

    /**
     * \brief Load a tile of items across a threadblock.
     */
    static __device__ __forceinline__ void Load(
        SmemStorage     &smem_storage,                  ///< [in] Shared reference to opaque SmemStorage layout
        InputIterator   block_itr,                      ///< [in] The threadblock's base input iterator for loading from
        T               (&items)[ITEMS_PER_THREAD])     ///< [out] Data to load
    {
        LoadInternal<POLICY>::Load(smem_storage, block_itr, items);
    }

    /**
     * \brief Load a tile of items across a threadblock, guarded by range.
     *
     * \tparam SizeT                <b>[inferred]</b> Integer type for offsets
     */
    template <typename SizeT>
    static __device__ __forceinline__ void Load(
        SmemStorage     &smem_storage,                  ///< [in] Shared reference to opaque SmemStorage layout
        InputIterator   block_itr,                      ///< [in] The threadblock's base input iterator for loading from
        const SizeT     &guarded_items,                 ///< [in] Number of valid items in the tile
        T               (&items)[ITEMS_PER_THREAD])     ///< [out] Data to load
    {
        LoadInternal<POLICY>::Load(smem_storage, block_itr, guarded_items, items);
    }
};

/** @} */       // end of SimtCoop group

} // namespace cub
CUB_NS_POSTFIX
