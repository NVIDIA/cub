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
 * The cub::BlockLoad type provides operations for reading global tiles of data into the threadblock (in blocked arrangement across threads).
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
 * \tparam MODIFIER             cub::PtxLoadModifier cache modifier.
 * \tparam T                    <b>[inferred]</b> The data type to load.
 * \tparam ITEMS_PER_THREAD     <b>[inferred]</b> The number of consecutive items partitioned onto each thread.
 * \tparam InputIterator        <b>[inferred]</b> The input iterator type (may be a simple pointer).
 *
 * The aggregate tile of items is assumed to be partitioned evenly across
 * threads in "blocked" fashion with thread<sub><em>i</em></sub> owning
 * the <em>i</em><sup>th</sup> segment of consecutive elements.
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
 * \tparam T                    <b>[inferred]</b> The data type to load.
 * \tparam ITEMS_PER_THREAD     <b>[inferred]</b> The number of consecutive items partitioned onto each thread.
 * \tparam InputIterator        <b>[inferred]</b> The input iterator type (may be a simple pointer).
 *
 * The aggregate tile of items is assumed to be partitioned evenly across
 * threads in "blocked" fashion with thread<sub><em>i</em></sub> owning
 * the <em>i</em><sup>th</sup> segment of consecutive elements.
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
 * \tparam MODIFIER             cub::PtxLoadModifier cache modifier.
 * \tparam T                    <b>[inferred]</b> The data type to load.
 * \tparam ITEMS_PER_THREAD     <b>[inferred]</b> The number of consecutive items partitioned onto each thread.
 * \tparam InputIterator        <b>[inferred]</b> The input iterator type (may be a simple pointer).
 * \tparam SizeT                <b>[inferred]</b> Integer type for offsets
 *
 * The aggregate tile of items is assumed to be partitioned evenly across
 * threads in "blocked" fashion with thread<sub><em>i</em></sub> owning
 * the <em>i</em><sup>th</sup> segment of consecutive elements.
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
 * \tparam T                    <b>[inferred]</b> The data type to load.
 * \tparam ITEMS_PER_THREAD     <b>[inferred]</b> The number of consecutive items partitioned onto each thread.
 * \tparam InputIterator        <b>[inferred]</b> The input iterator type (may be a simple pointer).
 * \tparam SizeT                <b>[inferred]</b> Integer type for offsets
 *
 * The aggregate tile of items is assumed to be partitioned evenly across
 * threads in "blocked" fashion with thread<sub><em>i</em></sub> owning
 * the <em>i</em><sup>th</sup> segment of consecutive elements.
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
 * \tparam MODIFIER             cub::PtxLoadModifier cache modifier.
 * \tparam T                    <b>[inferred]</b> The data type to load.
 * \tparam ITEMS_PER_THREAD     <b>[inferred]</b> The number of consecutive items partitioned onto each thread.
 * \tparam InputIterator        <b>[inferred]</b> The input iterator type (may be a simple pointer).
 * \tparam SizeT                <b>[inferred]</b> Integer type for offsets
 *
 * The aggregate tile of items is assumed to be partitioned evenly across
 * threads in "blocked" fashion with thread<sub><em>i</em></sub> owning
 * the <em>i</em><sup>th</sup> segment of consecutive elements.
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
 * \tparam T                    <b>[inferred]</b> The data type to load.
 * \tparam ITEMS_PER_THREAD     <b>[inferred]</b> The number of consecutive items partitioned onto each thread.
 * \tparam InputIterator        <b>[inferred]</b> The input iterator type (may be a simple pointer).
 * \tparam SizeT                <b>[inferred]</b> Integer type for offsets
 *
 * The aggregate tile of items is assumed to be partitioned evenly across
 * threads in "blocked" fashion with thread<sub><em>i</em></sub> owning
 * the <em>i</em><sup>th</sup> segment of consecutive elements.
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
 * \tparam MODIFIER             cub::PtxLoadModifier cache modifier.
 * \tparam T                    <b>[inferred]</b> The data type to load.
 * \tparam ITEMS_PER_THREAD     <b>[inferred]</b> The number of consecutive items partitioned onto each thread.
 * \tparam InputIterator        <b>[inferred]</b> The input iterator type (may be a simple pointer).
 *
 * The aggregate tile of items is assumed to be partitioned across
 * threads in "striped" fashion, i.e., the \p ITEMS_PER_THREAD
 * items owned by each thread have logical stride \p BLOCK_THREADS between them.
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
 * \tparam MODIFIER             cub::PtxLoadModifier cache modifier.
 * \tparam T                    <b>[inferred]</b> The data type to load.
 * \tparam ITEMS_PER_THREAD     <b>[inferred]</b> The number of consecutive items partitioned onto each thread.
 * \tparam InputIterator        <b>[inferred]</b> The input iterator type (may be a simple pointer).
 *
 * The aggregate tile of items is assumed to be partitioned across
 * threads in "striped" fashion, i.e., the \p ITEMS_PER_THREAD
 * items owned by each thread have logical stride \p BLOCK_THREADS between them.
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
 * \tparam BLOCK_THREADS          The threadblock size in threads
 * \tparam MODIFIER             cub::PtxLoadModifier cache modifier.
 * \tparam T                    <b>[inferred]</b> The data type to load.
 * \tparam ITEMS_PER_THREAD     <b>[inferred]</b> The number of consecutive items partitioned onto each thread.
 * \tparam InputIterator        <b>[inferred]</b> The input iterator type (may be a simple pointer).
 * \tparam SizeT                <b>[inferred]</b> Integer type for offsets
 *
 * The aggregate tile of items is assumed to be partitioned across
 * threads in "striped" fashion, i.e., the \p ITEMS_PER_THREAD
 * items owned by each thread have logical stride \p BLOCK_THREADS between them.
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
 * \tparam T                    <b>[inferred]</b> The data type to load.
 * \tparam ITEMS_PER_THREAD     <b>[inferred]</b> The number of consecutive items partitioned onto each thread.
 * \tparam InputIterator        <b>[inferred]</b> The input iterator type (may be a simple pointer).
 * \tparam SizeT                <b>[inferred]</b> Integer type for offsets
 *
 * The aggregate tile of items is assumed to be partitioned across
 * threads in "striped" fashion, i.e., the \p ITEMS_PER_THREAD
 * items owned by each thread have logical stride \p BLOCK_THREADS between them.
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
 * \tparam MODIFIER             cub::PtxLoadModifier cache modifier.
 * \tparam T                    <b>[inferred]</b> The data type to load.
 * \tparam ITEMS_PER_THREAD     <b>[inferred]</b> The number of consecutive items partitioned onto each thread.
 * \tparam InputIterator        <b>[inferred]</b> The input iterator type (may be a simple pointer).
 * \tparam SizeT                <b>[inferred]</b> Integer type for offsets
 *
 * The aggregate tile of items is assumed to be partitioned across
 * threads in "striped" fashion, i.e., the \p ITEMS_PER_THREAD
 * items owned by each thread have logical stride \p BLOCK_THREADS between them.
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
 * \tparam T                    <b>[inferred]</b> The data type to load.
 * \tparam ITEMS_PER_THREAD     <b>[inferred]</b> The number of consecutive items partitioned onto each thread.
 * \tparam InputIterator        <b>[inferred]</b> The input iterator type (may be a simple pointer).
 * \tparam SizeT                <b>[inferred]</b> Integer type for offsets
 *
 * The aggregate tile of items is assumed to be partitioned across
 * threads in "striped" fashion, i.e., the \p ITEMS_PER_THREAD
 * items owned by each thread have logical stride \p BLOCK_THREADS between them.
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
 * \tparam MODIFIER             cub::PtxLoadModifier cache modifier.
 * \tparam T                    <b>[inferred]</b> The data type to load.
 * \tparam ITEMS_PER_THREAD     <b>[inferred]</b> The number of consecutive items partitioned onto each thread.
 * \tparam SizeT                <b>[inferred]</b> Integer type for offsets
 *
 * The aggregate tile of items is assumed to be partitioned evenly across
 * threads in "blocked" fashion with thread<sub><em>i</em></sub> owning
 * the <em>i</em><sup>th</sup> segment of consecutive elements.
 *
 * \par
 * The following conditions will prevent vectorization and loading will fall back to cub::BLOCK_LOAD_DIRECT:
 *   - \p ITEMS_PER_THREAD is odd
 *   - The \p InputIterator is not a simple pointer type
 *   - The input offset (\p block_ptr + \p block_offset) is not quad-aligned
 *   - The data type \p T is not a built-in primitive or CUDA vector type (e.g., \p short, \p int2, \p double, \p float2, etc.)
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

    // Alias global pointer
    Vector *block_ptr_vectors = reinterpret_cast<Vector *>(block_ptr);

    // Vectorize if aligned
    if ((size_t(block_ptr_vectors) & (VEC_SIZE - 1)) == 0)
    {
        // Alias local data (use raw_items array here which should get optimized away to prevent conservative PTXAS lmem spilling)
        T raw_items[ITEMS_PER_THREAD];

        // Direct-load using vector types
        BlockLoadDirect<MODIFIER>(
            block_ptr_vectors,
            reinterpret_cast<Vector (&)[VECTORS_PER_THREAD]>(raw_items));

        // Copy
        #pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
        {
            items[ITEM] = raw_items[ITEM];
        }
    }
    else
    {
        // Unaligned: direct-load of individual items
        BlockLoadDirect<MODIFIER>(block_ptr, items);
    }
}



/**
 * \brief Load a tile of items across a threadblock directly.
 *
 * \tparam T                    <b>[inferred]</b> The data type to load.
 * \tparam ITEMS_PER_THREAD     <b>[inferred]</b> The number of consecutive items partitioned onto each thread.
 *
 * The aggregate tile of items is assumed to be partitioned evenly across
 * threads in "blocked" fashion with thread<sub><em>i</em></sub> owning
 * the <em>i</em><sup>th</sup> segment of consecutive elements.
 *
 * \par
 * The following conditions will prevent vectorization and loading will fall back to cub::BLOCK_LOAD_DIRECT:
 *   - \p ITEMS_PER_THREAD is odd
 *   - The \p InputIterator is not a simple pointer type
 *   - The input offset (\p block_ptr + \p block_offset) is not quad-aligned
 *   - The data type \p T is not a built-in primitive or CUDA vector type (e.g., \p short, \p int2, \p double, \p float2, etc.)
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
    BLOCK_LOAD_DIRECT,        ///< Loads consecutive thread-items directly from the input
    BLOCK_LOAD_VECTORIZE,     ///< Attempts to use CUDA's built-in vectorized items as a coalescing optimization
    BLOCK_LOAD_TRANSPOSE,     ///< Loads striped inputs as a coalescing optimization and then transposes them through shared memory into the desired blocks of thread-consecutive items
};


/**
 * \addtogroup SimtCoop
 * @{
 */


/**
 * \brief The BlockLoad type provides operations for reading global tiles of data into the threadblock (in blocked arrangement across threads). ![](block_load_logo.png)
 *
 * <b>Overview</b>
 * \par
 * BlockLoad can be configured to use one of three alternative algorithms:
 *   -# <b>cub::BLOCK_LOAD_DIRECT</b>.  Loads consecutive thread-items
 *      directly from the input.
 *   <br><br>
 *   -# <b>cub::BLOCK_LOAD_VECTORIZE</b>.  Attempts to use CUDA's
 *      built-in vectorized items as a coalescing optimization.  For
 *      example, <tt>ld.global.v4.s32</tt> will be generated when
 *      \p T = \p int and \p ITEMS_PER_THREAD > 4.
 *   <br><br>
 *   -# <b>cub::BLOCK_LOAD_TRANSPOSE</b>.  Loads striped inputs as
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
 * \tparam InputIterator        The input iterator type (may be a simple pointer).
 * \tparam BLOCK_THREADS          The threadblock size in threads.
 * \tparam ITEMS_PER_THREAD     The number of consecutive items partitioned onto each thread.
 * \tparam POLICY               <b>[optional]</b> cub::BlockLoadPolicy tuning policy enumeration.  Default = cub::BLOCK_LOAD_DIRECT.
 * \tparam MODIFIER             <b>[optional]</b> cub::PtxLoadModifier cache modifier.  Default = cub::PTX_LOAD_NONE.
 *
 * <b>Performance Features and Considerations</b>
 * \par
 * - After any operation, a subsequent threadblock barrier (<tt>__syncthreads()</tt>) is
 *   required if the supplied BlockLoad::SmemStorage is to be reused/repurposed by the threadblock.
 * - The following conditions will prevent vectorization and loading will fall back to cub::BLOCK_LOAD_DIRECT:
 *   - \p ITEMS_PER_THREAD is odd
 *   - The \p InputIterator is not a simple pointer type
 *   - The input offset (\p block_ptr + \p block_offset) is not quad-aligned
 *   - The data type \p T is not a built-in primitive or CUDA vector type (e.g., \p short, \p int2, \p double, \p float2, etc.)
 *
 * <b>Examples</b>
 * \par
 * - <b>Example 1:</b> Have a 128-thread threadblock load four consecutive integers per thread (blocked arrangement):
 *      \code
 *      #include <cub.cuh>
 *
 *      __global__ void SomeKernel(int *d_in, ...)
 *      {
 *          // Parameterize a BlockLoad type for use in the current execution context
 *          typedef cub::BlockLoad<int, 128, 4> BlockLoad;
 *
 *          // Declare shared memory for BlockLoad
 *          __shared__ typename BlockLoad::SmemStorage smem_storage;
 *
 *          // A segment of consecutive input items per thread
 *          int data[4];
 *
 *          // Load a tile of data at this block's offset
 *          BlockLoad::Load(data, d_in, blockIdx.x * 128 * 4);
 *
 *      \endcode
 *
 * \par
 * - <b>Example 2:</b> Have a threadblock load consecutive integers per thread (blocked arrangement) using vectorized loads and global-only caching:
 *      \code
 *      #include <cub.cuh>
 *
 *      template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
 *      __global__ void SomeKernel(int *d_in, ...)
 *      {
 *          // Parameterize a BlockLoad type for use in the current execution context
 *          typedef cub::BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD, BLOCK_LOAD_VECTORIZE, PTX_LOAD_CG> BlockLoad;
 *
 *          // Declare shared memory for BlockLoad
 *          __shared__ typename BlockLoad::SmemStorage smem_storage;
 *
 *          // A segment of consecutive input items per thread
 *          int data[ITEMS_PER_THREAD];
 *
 *          // Load a tile of data at this block's offset
 *          BlockLoad::Load(data, d_in, blockIdx.x * BLOCK_THREADS * ITEMS_PER_THREAD);
 *
 *      \endcode
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
    template <BlockLoadPolicy POLICY, int DUMMY = 0>
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
            InputIterator   block_itr,                        ///< [in] The threadblock's base input iterator for loading from
            T               (&items)[ITEMS_PER_THREAD])     ///< [out] Data to load
        {
            BlockLoadDirect<MODIFIER>(block_itr, items);
        }

        /// Load a tile of items across a threadblock, guarded by range
        template <typename SizeT>
        static __device__ __forceinline__ void Load(
            SmemStorage     &smem_storage,                  ///< [in] Shared reference to opaque SmemStorage layout
            InputIterator   block_itr,                        ///< [in] The threadblock's base input iterator for loading from
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
            T               *block_ptr,                       ///< [in] The threadblock's base input iterator for loading from
            T               (&items)[ITEMS_PER_THREAD])     ///< [out] Data to load
        {
            BlockLoadVectorized<MODIFIER>(block_ptr, items);
        }

        /// Load a tile of items across a threadblock, specialized for opaque input iterators (skips vectorization)
        template <
            typename T,
            typename InputIterator>
        static __device__ __forceinline__ void Load(
            SmemStorage     &smem_storage,                  ///< [in] Shared reference to opaque SmemStorage layout
            InputIterator   block_itr,                        ///< [in] The threadblock's base input iterator for loading from
            T               (&items)[ITEMS_PER_THREAD])     ///< [out] Data to load
        {
            BlockLoadDirect<MODIFIER>(block_itr, items);
        }

        /// Load a tile of items across a threadblock, guarded by range
        template <typename SizeT>
        static __device__ __forceinline__ void Load(
            SmemStorage     &smem_storage,                  ///< [in] Shared reference to opaque SmemStorage layout
            InputIterator   block_itr,                        ///< [in] The threadblock's base input iterator for loading from
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
            InputIterator   block_itr,                        ///< [in] The threadblock's base input iterator for loading from
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
            InputIterator   block_itr,                        ///< [in] The threadblock's base input iterator for loading from
            const SizeT     &guarded_items,                 ///< [in] Number of valid items in the tile
            T               (&items)[ITEMS_PER_THREAD])     ///< [out] Data to load{
        {
            BlockLoadDirectStriped<MODIFIER>(block_itr, guarded_items, items, BLOCK_THREADS);

            // Transpose to blocked order
            BlockExchange::StripedToBlocked(smem_storage, items);
        }

    };

    /// Shared memory storage layout type
    typedef typename LoadInternal<POLICY>::SmemStorage SmemLayout;

public:


    /// The operations exposed by BlockLoad require shared memory of this
    /// type.  This opaque storage can be allocated directly using the
    /// <tt>__shared__</tt> keyword.  Alternatively, it can be aliased to
    /// externally allocated shared memory or <tt>union</tt>'d with other types
    /// to facilitate shared memory reuse.
    typedef SmemLayout SmemStorage;


    //---------------------------------------------------------------------
    // Interface
    //---------------------------------------------------------------------

    /**
     * \brief Load a tile of items across a threadblock.
     */
    static __device__ __forceinline__ void Load(
        SmemStorage     &smem_storage,                  ///< [in] Shared reference to opaque SmemStorage layout
        InputIterator   block_itr,                        ///< [in] The threadblock's base input iterator for loading from
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
        InputIterator   block_itr,                        ///< [in] The threadblock's base input iterator for loading from
        const SizeT     &guarded_items,                 ///< [in] Number of valid items in the tile
        T               (&items)[ITEMS_PER_THREAD])     ///< [out] Data to load
    {
        LoadInternal<POLICY>::Load(smem_storage, block_itr, guarded_items, items);
    }
};

/** @} */       // end of SimtCoop group

} // namespace cub
CUB_NS_POSTFIX
