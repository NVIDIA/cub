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

#include "../util_namespace.cuh"
#include "../util_macro.cuh"
#include "../util_type.cuh"
#include "../util_vector.cuh"
#include "../thread/thread_load.cuh"
#include "block_exchange.cuh"

/// Optional outer namespace(s)
CUB_NS_PREFIX

/// CUB namespace
namespace cub {

/**
 * \addtogroup BlockModule
 * @{
 */


/******************************************************************//**
 * \name Blocked threadblock I/O
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
 * \tparam InputIteratorRA      <b>[inferred]</b> The random-access iterator type for input (may be a simple pointer type).
 */
template <
    PtxLoadModifier MODIFIER,
    typename        T,
    int             ITEMS_PER_THREAD,
    typename        InputIteratorRA>
__device__ __forceinline__ void BlockLoadBlocked(
    InputIteratorRA block_itr,                          ///< [in] The threadblock's base input iterator for loading from
    T               (&items)[ITEMS_PER_THREAD])         ///< [out] Data to load
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
 * \tparam InputIteratorRA      <b>[inferred]</b> The random-access iterator type for input (may be a simple pointer type).
 */
template <
    typename        T,
    int             ITEMS_PER_THREAD,
    typename        InputIteratorRA>
__device__ __forceinline__ void BlockLoadBlocked(
    InputIteratorRA block_itr,                          ///< [in] The threadblock's base input iterator for loading from
    T               (&items)[ITEMS_PER_THREAD])         ///< [out] Data to load
{
    BlockLoadBlocked<PTX_LOAD_NONE>(block_itr, items);
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
 * \tparam InputIteratorRA      <b>[inferred]</b> The random-access iterator type for input (may be a simple pointer type).
 */
template <
    PtxLoadModifier MODIFIER,
    typename        T,
    int             ITEMS_PER_THREAD,
    typename        InputIteratorRA>
__device__ __forceinline__ void BlockLoadBlocked(
    InputIteratorRA block_itr,                          ///< [in] The threadblock's base input iterator for loading from
    const int       &guarded_items,                     ///< [in] Number of valid items in the tile
    T               (&items)[ITEMS_PER_THREAD])         ///< [out] Data to load
{
    int bounds = guarded_items - (threadIdx.x * ITEMS_PER_THREAD);

    #pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
    {
        if (ITEM < bounds)
        {
            items[ITEM] = ThreadLoad<MODIFIER>(block_itr + (threadIdx.x * ITEMS_PER_THREAD) + ITEM);
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
 * \tparam InputIteratorRA      <b>[inferred]</b> The random-access iterator type for input (may be a simple pointer type).
 */
template <
    typename        T,
    int             ITEMS_PER_THREAD,
    typename        InputIteratorRA>
__device__ __forceinline__ void BlockLoadBlocked(
    InputIteratorRA block_itr,                          ///< [in] The threadblock's base input iterator for loading from
    const int       &guarded_items,                     ///< [in] Number of valid items in the tile
    T               (&items)[ITEMS_PER_THREAD])         ///< [out] Data to load
{
    BlockLoadBlocked<PTX_LOAD_NONE>(block_itr, guarded_items, items);
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
 * \tparam InputIteratorRA      <b>[inferred]</b> The random-access iterator type for input (may be a simple pointer type).
 */
template <
    PtxLoadModifier MODIFIER,
    typename        T,
    int             ITEMS_PER_THREAD,
    typename        InputIteratorRA>
__device__ __forceinline__ void BlockLoadBlocked(
    InputIteratorRA block_itr,                          ///< [in] The threadblock's base input iterator for loading from
    const int       &guarded_items,                     ///< [in] Number of valid items in the tile
    T               oob_default,                        ///< [in] Default value to assign out-of-bound items
    T               (&items)[ITEMS_PER_THREAD])         ///< [out] Data to load
{
    int bounds = guarded_items - (threadIdx.x * ITEMS_PER_THREAD);

    #pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
    {
        items[ITEM] = (ITEM < bounds) ?
            ThreadLoad<MODIFIER>(block_itr + (threadIdx.x * ITEMS_PER_THREAD) + ITEM) :
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
 * \tparam InputIteratorRA      <b>[inferred]</b> The random-access iterator type for input (may be a simple pointer type).
 */
template <
    typename        T,
    int             ITEMS_PER_THREAD,
    typename        InputIteratorRA>
__device__ __forceinline__ void BlockLoadBlocked(
    InputIteratorRA block_itr,                          ///< [in] The threadblock's base input iterator for loading from
    const int       &guarded_items,                     ///< [in] Number of valid items in the tile
    T               oob_default,                        ///< [in] Default value to assign out-of-bound items
    T               (&items)[ITEMS_PER_THREAD])         ///< [out] Data to load
{
    BlockLoadBlocked<PTX_LOAD_NONE>(block_itr, guarded_items, oob_default, items);
}


//@}  end member group
/******************************************************************//**
 * \name Striped threadblock I/O
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
 * \tparam InputIteratorRA      <b>[inferred]</b> The random-access iterator type for input (may be a simple pointer type).
 */
template <
    PtxLoadModifier MODIFIER,
    typename        T,
    int             ITEMS_PER_THREAD,
    typename        InputIteratorRA>
__device__ __forceinline__ void BlockLoadStriped(
    InputIteratorRA block_itr,                          ///< [in] The threadblock's base input iterator for loading from
    T               (&items)[ITEMS_PER_THREAD],         ///< [out] Data to load
    int             stride = blockDim.x)                ///< [in] <b>[optional]</b> Stripe stride.  Default is the width of the threadblock.  More efficient code can be generated if a compile-time-constant (e.g., BLOCK_THREADS) is supplied.
{
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
 * \tparam T                    <b>[inferred]</b> The data type to load.
 * \tparam ITEMS_PER_THREAD     <b>[inferred]</b> The number of consecutive items partitioned onto each thread.
 * \tparam InputIteratorRA      <b>[inferred]</b> The random-access iterator type for input (may be a simple pointer type).
 */
template <
    typename        T,
    int             ITEMS_PER_THREAD,
    typename        InputIteratorRA>
__device__ __forceinline__ void BlockLoadStriped(
    InputIteratorRA block_itr,                          ///< [in] The threadblock's base input iterator for loading from
    T               (&items)[ITEMS_PER_THREAD],         ///< [out] Data to load
    int             stride = blockDim.x)                ///< [in] <b>[optional]</b> Stripe stride.  Default is the width of the threadblock.  More efficient code can be generated if a compile-time-constant (e.g., BLOCK_THREADS) is supplied.
{
    BlockLoadStriped<PTX_LOAD_NONE>(block_itr, items, stride);
}

/**
 * \brief Load striped directly tile using the specified cache modifier, guarded by range
 *
 * The aggregate tile of items is assumed to be partitioned across
 * threads in "striped" fashion, i.e., the \p ITEMS_PER_THREAD
 * items owned by each thread have logical stride \p BLOCK_THREADS between them.
 *
 * \tparam MODIFIER             cub::PtxLoadModifier cache modifier.
 * \tparam T                    <b>[inferred]</b> The data type to load.
 * \tparam ITEMS_PER_THREAD     <b>[inferred]</b> The number of consecutive items partitioned onto each thread.
 * \tparam InputIteratorRA      <b>[inferred]</b> The random-access iterator type for input (may be a simple pointer type).
 */
template <
    PtxLoadModifier MODIFIER,
    typename        T,
    int             ITEMS_PER_THREAD,
    typename        InputIteratorRA>
__device__ __forceinline__ void BlockLoadStriped(
    InputIteratorRA block_itr,                      ///< [in] The threadblock's base input iterator for loading from
    const int       &guarded_items,                 ///< [in] Number of valid items in the tile
    T               (&items)[ITEMS_PER_THREAD],     ///< [out] Data to load
    int             stride = blockDim.x)            ///< [in] <b>[optional]</b> Stripe stride.  Default is the width of the threadblock.  More efficient code can be generated if a compile-time-constant (e.g., BLOCK_THREADS) is supplied.
{
    int bounds = guarded_items - threadIdx.x;

    #pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
    {
        if (ITEM * stride < bounds)
        {
            items[ITEM] = ThreadLoad<MODIFIER>(block_itr + threadIdx.x + (ITEM * stride));
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
 * \tparam InputIteratorRA      <b>[inferred]</b> The random-access iterator type for input (may be a simple pointer type).
 */
template <
    typename        T,
    int             ITEMS_PER_THREAD,
    typename        InputIteratorRA>
__device__ __forceinline__ void BlockLoadStriped(
    InputIteratorRA block_itr,                          ///< [in] The threadblock's base input iterator for loading from
    const int       &guarded_items,                     ///< [in] Number of valid items in the tile
    T               (&items)[ITEMS_PER_THREAD],         ///< [out] Data to load
    int             stride = blockDim.x)                ///< [in] <b>[optional]</b> Stripe stride.  Default is the width of the threadblock.  More efficient code can be generated if a compile-time-constant (e.g., BLOCK_THREADS) is supplied.
{
    BlockLoadStriped<PTX_LOAD_NONE>(block_itr, guarded_items, items, stride);
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
 * \tparam InputIteratorRA      <b>[inferred]</b> The random-access iterator type for input (may be a simple pointer type).
 */
template <
    PtxLoadModifier MODIFIER,
    typename        T,
    int             ITEMS_PER_THREAD,
    typename        InputIteratorRA>
__device__ __forceinline__ void BlockLoadStriped(
    InputIteratorRA block_itr,                      ///< [in] The threadblock's base input iterator for loading from
    const int       &guarded_items,                 ///< [in] Number of valid items in the tile
    T               oob_default,                    ///< [in] Default value to assign out-of-bound items
    T               (&items)[ITEMS_PER_THREAD],     ///< [out] Data to load
    int             stride = blockDim.x)            ///< [in] <b>[optional]</b> Stripe stride.  Default is the width of the threadblock.  More efficient code can be generated if a compile-time-constant (e.g., BLOCK_THREADS) is supplied.
{
    int bounds = guarded_items - threadIdx.x;

    #pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
    {
        items[ITEM] = (ITEM * stride < bounds) ?
             ThreadLoad<MODIFIER>(block_itr + threadIdx.x + (ITEM * stride)) :
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
 * \tparam InputIteratorRA      <b>[inferred]</b> The random-access iterator type for input (may be a simple pointer type).
 */
template <
    typename        T,
    int             ITEMS_PER_THREAD,
    typename        InputIteratorRA>
__device__ __forceinline__ void BlockLoadStriped(
    InputIteratorRA block_itr,                          ///< [in] The threadblock's base input iterator for loading from
    const int       &guarded_items,                     ///< [in] Number of valid items in the tile
    T               oob_default,                        ///< [in] Default value to assign out-of-bound items
    T               (&items)[ITEMS_PER_THREAD],         ///< [out] Data to load
    int             stride = blockDim.x)                ///< [in] <b>[optional]</b> Stripe stride.  Default is the width of the threadblock.  More efficient code can be generated if a compile-time-constant (e.g., BLOCK_THREADS) is supplied.
{
    BlockLoadStriped<PTX_LOAD_NONE>(block_itr, guarded_items, oob_default, items, stride);
}


//@}  end member group
/******************************************************************//**
 * \name Warp-striped threadblock I/O
 *********************************************************************/
//@{


/**
 * \brief Load warp-striped tile directly using the specified cache modifier.
 *
 * The aggregate tile of items is assumed to be partitioned across threads in
 * "warp-striped" fashion, i.e., each warp owns a contiguous segment of
 * (\p WARP_THREADS * \p ITEMS_PER_THREAD) items, and the \p ITEMS_PER_THREAD
 * items owned by each thread within the same warp have logical stride
 * \p WARP_THREADS between them.
 *
 * \tparam MODIFIER             cub::PtxLoadModifier cache modifier.
 * \tparam T                    <b>[inferred]</b> The data type to load.
 * \tparam ITEMS_PER_THREAD     <b>[inferred]</b> The number of consecutive items partitioned onto each thread.
 * \tparam InputIteratorRA      <b>[inferred]</b> The random-access iterator type for input (may be a simple pointer type).
 */
template <
    PtxLoadModifier MODIFIER,
    typename        T,
    int             ITEMS_PER_THREAD,
    typename        InputIteratorRA>
__device__ __forceinline__ void BlockLoadWarpStriped(
    InputIteratorRA block_itr,                          ///< [in] The threadblock's base input iterator for loading from
    T               (&items)[ITEMS_PER_THREAD])         ///< [out] Data to load
{
    const int WARP_TILE_ITEMS   = PtxArchProps::WARP_THREADS * ITEMS_PER_THREAD;
    int tid                     = threadIdx.x % PtxArchProps::WARP_THREADS;
    int wid                     = threadIdx.x / PtxArchProps::WARP_THREADS;
    int warp_offset             = wid * WARP_TILE_ITEMS;

    // Load directly in warp-striped order
    #pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
    {
        items[ITEM] = ThreadLoad<MODIFIER>(block_itr + warp_offset + tid + (ITEM * PtxArchProps::WARP_THREADS));
    }
}


/**
 * \brief Load warp-striped tile directly.
 *
 * The aggregate tile of items is assumed to be partitioned across threads in
 * "warp-striped" fashion, i.e., each warp owns a contiguous segment of
 * (\p WARP_THREADS * \p ITEMS_PER_THREAD) items, and the \p ITEMS_PER_THREAD
 * items owned by each thread within the same warp have logical stride
 * \p WARP_THREADS between them.
 *
 * \tparam T                    <b>[inferred]</b> The data type to load.
 * \tparam ITEMS_PER_THREAD     <b>[inferred]</b> The number of consecutive items partitioned onto each thread.
 * \tparam InputIteratorRA      <b>[inferred]</b> The random-access iterator type for input (may be a simple pointer type).
 */
template <
    typename        T,
    int             ITEMS_PER_THREAD,
    typename        InputIteratorRA>
__device__ __forceinline__ void BlockLoadWarpStriped(
    InputIteratorRA block_itr,                          ///< [in] The threadblock's base input iterator for loading from
    T               (&items)[ITEMS_PER_THREAD])         ///< [out] Data to load
{
    BlockLoadWarpStriped<PTX_LOAD_NONE>(block_itr, items);
}

/**
 * \brief Load warp-striped directly tile using the specified cache modifier, guarded by range
 *
 * The aggregate tile of items is assumed to be partitioned across threads in
 * "warp-striped" fashion, i.e., each warp owns a contiguous segment of
 * (\p WARP_THREADS * \p ITEMS_PER_THREAD) items, and the \p ITEMS_PER_THREAD
 * items owned by each thread within the same warp have logical stride
 * \p WARP_THREADS between them.
 *
 * \tparam MODIFIER             cub::PtxLoadModifier cache modifier.
 * \tparam T                    <b>[inferred]</b> The data type to load.
 * \tparam ITEMS_PER_THREAD     <b>[inferred]</b> The number of consecutive items partitioned onto each thread.
 * \tparam InputIteratorRA      <b>[inferred]</b> The random-access iterator type for input (may be a simple pointer type).
 */
template <
    PtxLoadModifier MODIFIER,
    typename        T,
    int             ITEMS_PER_THREAD,
    typename        InputIteratorRA>
__device__ __forceinline__ void BlockLoadWarpStriped(
    InputIteratorRA block_itr,                      ///< [in] The threadblock's base input iterator for loading from
    const int       &guarded_items,                 ///< [in] Number of valid items in the tile
    T               (&items)[ITEMS_PER_THREAD])     ///< [out] Data to load
{
    const int WARP_TILE_ITEMS   = PtxArchProps::WARP_THREADS * ITEMS_PER_THREAD;
    int tid                     = threadIdx.x % PtxArchProps::WARP_THREADS;
    int wid                     = threadIdx.x / PtxArchProps::WARP_THREADS;
    int warp_offset             = wid * WARP_TILE_ITEMS;
    int bounds                  = guarded_items - warp_offset - tid;

    // Load directly in warp-striped order
    #pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
    {
        if ((ITEM * PtxArchProps::WARP_THREADS) < bounds)
        {
            items[ITEM] = ThreadLoad<MODIFIER>(block_itr + warp_offset + tid + (ITEM * PtxArchProps::WARP_THREADS));
        }
    }
}


/**
 * \brief Load warp-striped tile directly, guarded by range
 *
 * The aggregate tile of items is assumed to be partitioned across threads in
 * "warp-striped" fashion, i.e., each warp owns a contiguous segment of
 * (\p WARP_THREADS * \p ITEMS_PER_THREAD) items, and the \p ITEMS_PER_THREAD
 * items owned by each thread within the same warp have logical stride
 * \p WARP_THREADS between them.
 *
 * \tparam T                    <b>[inferred]</b> The data type to load.
 * \tparam ITEMS_PER_THREAD     <b>[inferred]</b> The number of consecutive items partitioned onto each thread.
 * \tparam InputIteratorRA      <b>[inferred]</b> The random-access iterator type for input (may be a simple pointer type).
 */
template <
    typename        T,
    int             ITEMS_PER_THREAD,
    typename        InputIteratorRA>
__device__ __forceinline__ void BlockLoadWarpStriped(
    InputIteratorRA block_itr,                          ///< [in] The threadblock's base input iterator for loading from
    const int       &guarded_items,                     ///< [in] Number of valid items in the tile
    T               (&items)[ITEMS_PER_THREAD])         ///< [out] Data to load
{
    BlockLoadWarpStriped<PTX_LOAD_NONE>(block_itr, guarded_items, items);
}


/**
 * \brief Load warp-striped directly tile using the specified cache modifier, guarded by range, with assignment for out-of-bound elements
 *
 * The aggregate tile of items is assumed to be partitioned across threads in
 * "warp-striped" fashion, i.e., each warp owns a contiguous segment of
 * (\p WARP_THREADS * \p ITEMS_PER_THREAD) items, and the \p ITEMS_PER_THREAD
 * items owned by each thread within the same warp have logical stride
 * \p WARP_THREADS between them.
 *
 * \tparam MODIFIER             cub::PtxLoadModifier cache modifier.
 * \tparam T                    <b>[inferred]</b> The data type to load.
 * \tparam ITEMS_PER_THREAD     <b>[inferred]</b> The number of consecutive items partitioned onto each thread.
 * \tparam InputIteratorRA      <b>[inferred]</b> The random-access iterator type for input (may be a simple pointer type).
 */
template <
    PtxLoadModifier MODIFIER,
    typename        T,
    int             ITEMS_PER_THREAD,
    typename        InputIteratorRA>
__device__ __forceinline__ void BlockLoadWarpStriped(
    InputIteratorRA block_itr,                      ///< [in] The threadblock's base input iterator for loading from
    const int       &guarded_items,                 ///< [in] Number of valid items in the tile
    T               oob_default,                    ///< [in] Default value to assign out-of-bound items
    T               (&items)[ITEMS_PER_THREAD],     ///< [out] Data to load
    int             stride = blockDim.x)            ///< [in] <b>[optional]</b> Stripe stride.  Default is the width of the threadblock.  More efficient code can be generated if a compile-time-constant (e.g., BLOCK_THREADS) is supplied.
{
    const int WARP_TILE_ITEMS   = PtxArchProps::WARP_THREADS * ITEMS_PER_THREAD;
    int tid                     = threadIdx.x % PtxArchProps::WARP_THREADS;
    int wid                     = threadIdx.x / PtxArchProps::WARP_THREADS;
    int warp_offset             = wid * WARP_TILE_ITEMS;
    int bounds                  = guarded_items - warp_offset - tid;

    // Load directly in warp-striped order
    #pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
    {
        items[ITEM] = ((ITEM * PtxArchProps::WARP_THREADS) < bounds) ?
            ThreadLoad<MODIFIER>(block_itr + warp_offset + tid + (ITEM * PtxArchProps::WARP_THREADS)) :
            oob_default;
    }
}


/**
 * \brief Load warp-striped tile directly, guarded by range, with assignment for out-of-bound elements
 *
 * The aggregate tile of items is assumed to be partitioned across threads in
 * "warp-striped" fashion, i.e., each warp owns a contiguous segment of
 * (\p WARP_THREADS * \p ITEMS_PER_THREAD) items, and the \p ITEMS_PER_THREAD
 * items owned by each thread within the same warp have logical stride
 * \p WARP_THREADS between them.
 *
 * \tparam T                    <b>[inferred]</b> The data type to load.
 * \tparam ITEMS_PER_THREAD     <b>[inferred]</b> The number of consecutive items partitioned onto each thread.
 * \tparam InputIteratorRA      <b>[inferred]</b> The random-access iterator type for input (may be a simple pointer type).
 */
template <
    typename        T,
    int             ITEMS_PER_THREAD,
    typename        InputIteratorRA>
__device__ __forceinline__ void BlockLoadWarpStriped(
    InputIteratorRA block_itr,                          ///< [in] The threadblock's base input iterator for loading from
    const int       &guarded_items,                     ///< [in] Number of valid items in the tile
    T               oob_default,                        ///< [in] Default value to assign out-of-bound items
    T               (&items)[ITEMS_PER_THREAD])         ///< [out] Data to load
{
    BlockLoadWarpStriped<PTX_LOAD_NONE>(block_itr, guarded_items, oob_default, items);
}


//@}  end member group
/******************************************************************//**
 * \name Blocked, vectorized threadblock I/O
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
 */
template <
    PtxLoadModifier MODIFIER,
    typename        T,
    int             ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockLoadVectorized(
    T               *block_ptr,                         ///< [in] Input pointer for loading from
    T               (&items)[ITEMS_PER_THREAD])         ///< [out] Data to load
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

    // Alias local data (use raw_items array here which should get optimized away to prevent conservative PTXAS lmem spilling)
    T raw_items[ITEMS_PER_THREAD];

    // Direct-load using vector types
    BlockLoadBlocked<MODIFIER>(
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
    T               *block_ptr,                         ///< [in] Input pointer for loading from
    T               (&items)[ITEMS_PER_THREAD])         ///< [out] Data to load
{
    BlockLoadVectorized<PTX_LOAD_NONE>(block_ptr, items);
}

//@}  end member group

/** @} */       // end group BlockModule



//-----------------------------------------------------------------------------
// Generic BlockLoad abstraction
//-----------------------------------------------------------------------------

/**
 * \brief cub::BlockLoadAlgorithm enumerates alternative algorithms for cub::BlockLoad to load a tile of data from memory into a blocked arrangement across a CUDA thread block.
 */
enum BlockLoadAlgorithm
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
    BLOCK_LOAD_DIRECT,        //!< BLOCK_LOAD_DIRECT

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
     *   - The \p InputIteratorRA is not a simple pointer type
     *   - The block input offset is not quadword-aligned
     *   - The data type \p T is not a built-in primitive or CUDA vector type (e.g., \p short, \p int2, \p double, \p float2, etc.)
     */
    BLOCK_LOAD_VECTORIZE,     //!< BLOCK_LOAD_VECTORIZE

    /**
     * \par Overview
     *
     * A [<em>striped arrangement</em>](index.html#sec3sec3) of data is read
     * directly from memory and then is locally transposed into a
     * [<em>blocked arrangement</em>](index.html#sec3sec3). The threadblock
     * reads items in a parallel "strip-mining" fashion:
     * thread<sub><em>i</em></sub> reads items having stride \p BLOCK_THREADS
     * between them. cub::BlockExchange is then used to locally reorder the items
     * into a [<em>blocked arrangement</em>](index.html#sec3sec3).
     *
     * \par Performance Considerations
     * - The utilization of memory transactions (coalescing) remains high regardless
     *   of items loaded per thread.
     * - The local reordering incurs slightly longer latencies and throughput than the
     *   direct cub::BLOCK_LOAD_DIRECT and cub::BLOCK_LOAD_VECTORIZE alternatives.
     */
    BLOCK_LOAD_TRANSPOSE,     //!< BLOCK_LOAD_TRANSPOSE


    /**
     * \par Overview
     *
     * A [<em>warp-striped arrangement</em>](index.html#sec3sec3) of data is read
     * directly from memory and then is locally transposed into a
     * [<em>blocked arrangement</em>](index.html#sec3sec3). Each warp reads its own
     * contiguous segment in a parallel "strip-mining" fashion: lane<sub><em>i</em></sub>
     * reads items having stride \p WARP_THREADS between them. cub::BlockExchange
     * is then used to locally reorder the items into a
     * [<em>blocked arrangement</em>](index.html#sec3sec3).
     *
     * \par Performance Considerations
     * - The utilization of memory transactions (coalescing) remains high regardless
     *   of items loaded per thread.
     * - The local reordering incurs slightly longer latencies and throughput than the
     *   direct cub::BLOCK_LOAD_DIRECT and cub::BLOCK_LOAD_VECTORIZE alternatives.
     */
    BLOCK_LOAD_WARP_TRANSPOSE,//!< BLOCK_LOAD_WARP_TRANSPOSE
};


/**
 * \addtogroup BlockModule
 * @{
 */


/**
 * \brief BlockLoad provides cooperative data movement operations for loading a tile of data from memory into a [<em>block arrangement</em>](index.html#sec3sec3) across a CUDA thread block.  ![](block_load_logo.png)
 *
 * BlockLoad provides a tile-loading abstraction that implements alternative
 * cub::BlockLoadAlgorithm strategies that can be used to optimize data
 * movement on different architectures for different granularity sizes (i.e.,
 * number of items per thread).
 *
 * \tparam InputIteratorRA      The input iterator type (may be a simple pointer type).
 * \tparam BLOCK_THREADS        The threadblock size in threads.
 * \tparam ITEMS_PER_THREAD     The number of consecutive items partitioned onto each thread.
 * \tparam ALGORITHM            <b>[optional]</b> cub::BlockLoadAlgorithm tuning policy.  Default = cub::BLOCK_LOAD_DIRECT.
 * \tparam MODIFIER             <b>[optional]</b> cub::PtxLoadModifier cache modifier.  Default = cub::PTX_LOAD_NONE.
 * \tparam WARP_TIME_SLICING          <b>[optional]</b> For cooperative cub::BlockLoadAlgorithm parameterizations that utilize shared memory: the number of communication rounds needed to complete the all-to-all exchange; more rounds can be traded for a smaller shared memory footprint (default = 1)
 *
 * \par Algorithm
 * BlockLoad can be (optionally) configured to use one of three alternative methods:
 *   -# <b>cub::BLOCK_LOAD_DIRECT</b>.  A [<em>blocked arrangement</em>](index.html#sec3sec3)
 *      of data is read directly from memory.  [More...](\ref cub::BlockLoadAlgorithm)
 *   -# <b>cub::BLOCK_LOAD_VECTORIZE</b>.  A [<em>blocked arrangement</em>](index.html#sec3sec3)
 *      of data is read directly from memory using CUDA's built-in vectorized loads as a
 *      coalescing optimization.    [More...](\ref cub::BlockLoadAlgorithm)
 *   -# <b>cub::BLOCK_LOAD_TRANSPOSE</b>.  A [<em>striped arrangement</em>](index.html#sec3sec3)
 *      of data is read directly from memory and is then locally transposed into a
 *      [<em>blocked arrangement</em>](index.html#sec3sec3).  [More...](\ref cub::BlockLoadAlgorithm)
 *   -# <b>cub::BLOCK_LOAD_WARP_TRANSPOSE</b>.  A [<em>warp-striped arrangement</em>](index.html#sec3sec3)
 *      of data is read directly from memory and is then locally transposed into a
 *      [<em>blocked arrangement</em>](index.html#sec3sec3).  [More...](\ref cub::BlockLoadAlgorithm)
 *
 * \par Usage Considerations
 * - \smemreuse{BlockLoad::SmemStorage}
 *
 * \par Performance Considerations
 *  - See cub::BlockLoadAlgorithm for more performance details regarding algorithmic alternatives
 *
 *
 * \par Examples
 * <em>Example 1.</em> Have a 128-thread thread block directly load a blocked
 * arrangement of four consecutive integers per thread.  The load operation
 * may suffer from non-coalesced memory accesses because consecutive threads are
 * referencing non-consecutive inputs as each item is read.
 * \code
 * #include <cub/cub.cuh>
 *
 * __global__ void SomeKernel(int *d_in, ...)
 * {
 *     // Parameterize BlockLoad for 128 threads (4 items each) on type int
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
 * <em>Example 2.</em> Have a thread block load a blocked arrangement of 21
 * consecutive integers per thread using strip-mined loads which are
 * transposed in shared memory.  The load operation has perfectly coalesced
 * memory accesses because consecutive threads are referencing consecutive
 * input items.
 * \code
 * #include <cub/cub.cuh>
 *
 * template <int BLOCK_THREADS>
 * __global__ void SomeKernel(int *d_in, ...)
 * {
 *     // Parameterize BlockLoad on type int
 *     typedef cub::BlockLoad<int*, BLOCK_THREADS, 21, BLOCK_LOAD_TRANSPOSE> BlockLoad;
 *
 *     // Declare shared memory for BlockLoad
 *     __shared__ typename BlockLoad::SmemStorage smem_storage;
 *
 *     // A segment of consecutive items per thread
 *     int data[21];
 *
 *     // Load a tile of data at this block's offset
 *     BlockLoad::Load(smem_storage, d_in + blockIdx.x * BLOCK_THREADS * 21, data);
 *
 *     ...
 * \endcode
 *
 * \par
 * <em>Example 3.</em> Have a thread block load a blocked arrangement of
 * \p ITEMS_PER_THREAD consecutive integers per thread using vectorized
 * loads and global-only caching.  The load operation will have perfectly
 * coalesced memory accesses if ITEMS_PER_THREAD is 1, 2, or 4 which allows
 * consecutive threads to read consecutive int1, int2, or int4 words.
 * \code
 * #include <cub/cub.cuh>
 *
 * template <
 *     int BLOCK_THREADS,
 *     int ITEMS_PER_THREAD>
 * __global__ void SomeKernel(int *d_in, ...)
 * {
 *     // Parameterize BlockLoad on type int
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
 */
template <
    typename            InputIteratorRA,
    int                 BLOCK_THREADS,
    int                 ITEMS_PER_THREAD,
    BlockLoadAlgorithm  ALGORITHM = BLOCK_LOAD_DIRECT,
    PtxLoadModifier     MODIFIER = PTX_LOAD_NONE,
    int                 WARP_TIME_SLICING = 1>
class BlockLoad
{

private:

    //---------------------------------------------------------------------
    // Type definitions and constants
    //---------------------------------------------------------------------

    // Data type of input iterator
    typedef typename std::iterator_traits<InputIteratorRA>::value_type T;


    /// Load helper
    template <BlockLoadAlgorithm _POLICY, int DUMMY = 0>
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
            SmemStorage     &smem_storage,                  ///< [in] Reference to shared memory allocation having layout type SmemStorage
            InputIteratorRA block_itr,                      ///< [in] The threadblock's base input iterator for loading from
            T               (&items)[ITEMS_PER_THREAD])     ///< [out] Data to load
        {
            BlockLoadBlocked<MODIFIER>(block_itr, items);
        }

        /// Load a tile of items across a threadblock, guarded by range
        static __device__ __forceinline__ void Load(
            SmemStorage     &smem_storage,                  ///< [in] Reference to shared memory allocation having layout type SmemStorage
            InputIteratorRA block_itr,                      ///< [in] The threadblock's base input iterator for loading from
            const int       &guarded_items,                 ///< [in] Number of valid items in the tile
            T               (&items)[ITEMS_PER_THREAD])     ///< [out] Data to load
        {
            BlockLoadBlocked<MODIFIER>(block_itr, guarded_items, items);
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
            SmemStorage         &smem_storage,                  ///< [in] Reference to shared memory allocation having layout type SmemStorage
            T                   *block_ptr,                     ///< [in] The threadblock's base input iterator for loading from
            T                   (&items)[ITEMS_PER_THREAD])     ///< [out] Data to load
        {
            BlockLoadVectorized<MODIFIER>(block_ptr, items);
        }

        /// Load a tile of items across a threadblock, specialized for opaque input iterators (skips vectorization)
        template <
            typename T,
            typename _InputIteratorRA>
        static __device__ __forceinline__ void Load(
            SmemStorage         &smem_storage,                  ///< [in] Reference to shared memory allocation having layout type SmemStorage
            _InputIteratorRA    block_itr,                      ///< [in] The threadblock's base input iterator for loading from
            T                   (&items)[ITEMS_PER_THREAD])     ///< [out] Data to load
        {
            BlockLoadBlocked<MODIFIER>(block_itr, items);
        }

        /// Load a tile of items across a threadblock, guarded by range
        static __device__ __forceinline__ void Load(
            SmemStorage         &smem_storage,                  ///< [in] Reference to shared memory allocation having layout type SmemStorage
            InputIteratorRA     block_itr,                      ///< [in] The threadblock's base input iterator for loading from
            const int           &guarded_items,                 ///< [in] Number of valid items in the tile
            T                   (&items)[ITEMS_PER_THREAD])     ///< [out] Data to load{
        {
            BlockLoadBlocked<MODIFIER>(block_itr, guarded_items, items);
        }
    };


    /**
     * BLOCK_LOAD_TRANSPOSE specialization of load helper
     */
    template <int DUMMY>
    struct LoadInternal<BLOCK_LOAD_TRANSPOSE, DUMMY>
    {
        // BlockExchange utility type for keys
        typedef BlockExchange<T, BLOCK_THREADS, ITEMS_PER_THREAD, WARP_TIME_SLICING> BlockExchange;

        /// Shared memory storage layout type
        typedef typename BlockExchange::SmemStorage SmemStorage;

        /// Load a tile of items across a threadblock
        static __device__ __forceinline__ void Load(
            SmemStorage         &smem_storage,                  ///< [in] Reference to shared memory allocation having layout type SmemStorage
            InputIteratorRA     block_itr,                      ///< [in] The threadblock's base input iterator for loading from
            T                   (&items)[ITEMS_PER_THREAD])     ///< [out] Data to load{
        {
            BlockLoadStriped<MODIFIER>(block_itr, items, BLOCK_THREADS);
            BlockExchange::StripedToBlocked(smem_storage, items);
        }

        /// Load a tile of items across a threadblock, guarded by range
        static __device__ __forceinline__ void Load(
            SmemStorage         &smem_storage,                  ///< [in] Reference to shared memory allocation having layout type SmemStorage
            InputIteratorRA     block_itr,                      ///< [in] The threadblock's base input iterator for loading from
            const int           &guarded_items,                 ///< [in] Number of valid items in the tile
            T                   (&items)[ITEMS_PER_THREAD])     ///< [out] Data to load{
        {
            BlockLoadStriped<MODIFIER>(block_itr, guarded_items, items, BLOCK_THREADS);
            BlockExchange::StripedToBlocked(smem_storage, items);
        }

    };


    /**
     * BLOCK_LOAD_WARP_TRANSPOSE specialization of load helper
     */
    template <int DUMMY>
    struct LoadInternal<BLOCK_LOAD_WARP_TRANSPOSE, DUMMY>
    {
        // BlockExchange utility type for keys
        typedef BlockExchange<T, BLOCK_THREADS, ITEMS_PER_THREAD, WARP_TIME_SLICING> BlockExchange;

        /// Shared memory storage layout type
        typedef typename BlockExchange::SmemStorage SmemStorage;

        /// Load a tile of items across a threadblock
        static __device__ __forceinline__ void Load(
            SmemStorage         &smem_storage,                  ///< [in] Reference to shared memory allocation having layout type SmemStorage
            InputIteratorRA     block_itr,                      ///< [in] The threadblock's base input iterator for loading from
            T                   (&items)[ITEMS_PER_THREAD])     ///< [out] Data to load{
        {
            BlockLoadWarpStriped<MODIFIER>(block_itr, items);
            BlockExchange::WarpStripedToBlocked(smem_storage, items);
        }

        /// Load a tile of items across a threadblock, guarded by range
        static __device__ __forceinline__ void Load(
            SmemStorage         &smem_storage,                  ///< [in] Reference to shared memory allocation having layout type SmemStorage
            InputIteratorRA     block_itr,                      ///< [in] The threadblock's base input iterator for loading from
            const int           &guarded_items,                 ///< [in] Number of valid items in the tile
            T                   (&items)[ITEMS_PER_THREAD])     ///< [out] Data to load{
        {
            BlockLoadWarpStriped<MODIFIER>(block_itr, guarded_items, items);
            BlockExchange::WarpStripedToBlocked(smem_storage, items);
        }

    };

    /// Shared memory storage layout type
    typedef typename LoadInternal<ALGORITHM>::SmemStorage _SmemStorage;

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
        SmemStorage         &smem_storage,                  ///< [in] Reference to shared memory allocation having layout type SmemStorage
        InputIteratorRA     block_itr,                      ///< [in] The threadblock's base input iterator for loading from
        T                   (&items)[ITEMS_PER_THREAD])     ///< [out] Data to load
    {
        LoadInternal<ALGORITHM>::Load(smem_storage, block_itr, items);
    }

    /**
     * \brief Load a tile of items across a threadblock, guarded by range.
     */
    static __device__ __forceinline__ void Load(
        SmemStorage         &smem_storage,                  ///< [in] Reference to shared memory allocation having layout type SmemStorage
        InputIteratorRA     block_itr,                      ///< [in] The threadblock's base input iterator for loading from
        const int           &guarded_items,                 ///< [in] Number of valid items in the tile
        T                   (&items)[ITEMS_PER_THREAD])     ///< [out] Data to load
    {
        LoadInternal<ALGORITHM>::Load(smem_storage, block_itr, guarded_items, items);
    }
};

/** @} */       // end group BlockModule

}               // CUB namespace
CUB_NS_POSTFIX  // Optional outer namespace(s)

