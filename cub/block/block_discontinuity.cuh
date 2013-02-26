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
 * The cub::BlockDiscontinuity type provides operations for flagging discontinuities within a list of data items partitioned across a threadblock.
 */

#pragma once

#include "../device_props.cuh"
#include "../type_utils.cuh"
#include "../operators.cuh"
#include "../ns_wrapper.cuh"

CUB_NS_PREFIX

/// CUB namespace
namespace cub {

/**
 * \addtogroup SimtCoop
 * @{
 */


/**
 * \brief The BlockDiscontinuity type provides operations for flagging discontinuities within a list of data items partitioned across a threadblock. ![](discont_logo.png)
 *
 * <b>Overview</b>
 * \par
 * The operations exposed by BlockDiscontinuity allow threadblocks to set "head flags" for data elements that
 * are different from their predecessor (as specified by a binary boolean operator).
 *
 * \tparam T                    The data type to be exchanged.
 * \tparam BLOCK_THREADS          The threadblock size in threads.
 *
 * <b>Performance Features and Considerations</b>
 * \par
 * - After any operation, a subsequent threadblock barrier (<tt>__syncthreads()</tt>) is
 *   required if the supplied BlockDiscontinuity::SmemStorage is to be reused/repurposed by the threadblock.
 * - Zero bank conflicts for most types.
 *
 */
template <
    typename    T,
    int         BLOCK_THREADS>
class BlockDiscontinuity
{
private:

    //---------------------------------------------------------------------
    // Type definitions and constants
    //---------------------------------------------------------------------

    /// Shared memory storage layout type
    struct SmemLayout
    {
        T last_items[BLOCK_THREADS];      ///< Last element from each thread's input
    };

public:

    /// The operations exposed by BlockDiscontinuity require shared memory of this
    /// type.  This opaque storage can be allocated directly using the
    /// <tt>__shared__</tt> keyword.  Alternatively, it can be aliased to
    /// externally allocated shared memory or <tt>union</tt>'d with other types
    /// to facilitate shared memory reuse.
    typedef SmemLayout SmemStorage;


    /**
     * \brief Sets discontinuity flags for a tile of threadblock items, for which the first item has no reference (and is always flagged).  The last tile item of the last thread is also returned to thread<sub>0</sub>.
     *
     * Assuming a <em>blocked</em> arrangement of elements across threads,
     * <tt>flags</tt><sub><em>i</em></sub> is set non-zero for item
     * <tt>input</tt><sub><em>i</em></sub> when <tt>scan_op(</tt><em>previous-item</em>, <tt>input<sub><em>i</em></sub>)</tt>
     * is \p true (where <em>previous-item</em> is either <tt>input<sub><em>i-1</em></sub></tt>,
     * or <tt>input<sub><em>ITEMS_PER_THREAD</em>-1</sub></tt> in the previous thread).  Furthermore,
     * <tt>flags</tt><sub><em>i</em></sub> is always non-zero for <tt>input<sub>0</sub></tt>
     * in <em>thread</em><sub>0</sub>.
     *
     * The \p last_tile_item is undefined in threads other than <em>thread</em><sub>0</sub>.
     *
     * \smemreuse
     *
     * \tparam ITEMS_PER_THREAD     <b>[inferred]</b> The number of consecutive items partitioned onto each thread.
     * \tparam FlagT                 <b>[inferred]</b> The flag type (must be an integer type)
     * \tparam FlagOp               <b>[inferred]</b> Binary boolean functor type, having input parameters <tt>(const T &a, const T &b)</tt> and returning \p true if a discontinuity exists between \p a and \p b, otherwise \p false.
     */
    template <
        int             ITEMS_PER_THREAD,
        typename        FlagT,
        typename        FlagOp>
    static __device__ __forceinline__ void Flag(
        SmemStorage     &smem_storage,                  ///< [in] Shared reference to opaque SmemStorage layout
        T               (&input)[ITEMS_PER_THREAD],     ///< [in] Input items
        FlagOp          flag_op,                        ///< [in] Binary boolean flag predicate
        FlagT           (&flags)[ITEMS_PER_THREAD],     ///< [out] Discontinuity flags
        T               &last_tile_item)                ///< [out] <b>[<em>thread</em><sub>0</sub> only]</b> The last tile item (<tt>input<sub><em>ITEMS_PER_THREAD</em>-1</sub></tt> from thread<sub><tt><em>BLOCK_THREADS</em></tt>-1</sub>)
    {
        // Share last item
        smem_storage.last_items[threadIdx.x] = input[ITEMS_PER_THREAD - 1];

        __syncthreads();

        // Set flag for first item
        if (threadIdx.x == 0)
        {
            flags[0] = 1;
            last_tile_item = smem_storage.last_items[BLOCK_THREADS - 1];
        }
        else
        {
            flags[0] = flag_op(smem_storage.last_items[threadIdx.x - 1], input[0]);
        }


        // Set flags for remaining items
        #pragma unroll
        for (int ITEM = 1; ITEM < ITEMS_PER_THREAD; ITEM++)
        {
            flags[ITEM] = flag_op(input[ITEM - 1], input[ITEM]);
        }
    }


    /**
     * \brief Sets discontinuity flags for a tile of threadblock items, for which the first item has no reference (and is always flagged).
     *
     * Assuming a <em>blocked</em> arrangement of elements across threads,
     * <tt>flags</tt><sub><em>i</em></sub> is set non-zero for item
     * <tt>input</tt><sub><em>i</em></sub> when <tt>scan_op(</tt><em>previous-item</em>, <tt>input<sub><em>i</em></sub>)</tt>
     * is \p true (where <em>previous-item</em> is either <tt>input<sub><em>i-1</em></sub></tt>,
     * or <tt>input<sub><em>ITEMS_PER_THREAD</em>-1</sub></tt> in the previous thread).  Furthermore,
     * <tt>flags</tt><sub><em>i</em></sub> is always non-zero for <tt>input<sub>0</sub></tt>
     * in <em>thread</em><sub>0</sub>.
     *
     * The \p last_tile_item is undefined in threads other than <em>thread</em><sub>0</sub>.
     *
     * \smemreuse
     *
     * \tparam ITEMS_PER_THREAD     <b>[inferred]</b> The number of consecutive items partitioned onto each thread.
     * \tparam FlagT                 <b>[inferred]</b> The flag type (must be an integer type)
     * \tparam FlagOp               <b>[inferred]</b> Binary boolean functor type, having input parameters <tt>(const T &a, const T &b)</tt> and returning \p true if a discontinuity exists between \p a and \p b, otherwise \p false.
     */
    template <
        int             ITEMS_PER_THREAD,
        typename        FlagT,
        typename        FlagOp>
    static __device__ __forceinline__ void Flag(
        SmemStorage     &smem_storage,                  ///< [in] Shared reference to opaque SmemStorage layout
        T               (&input)[ITEMS_PER_THREAD],     ///< [in] Input items
        FlagOp          flag_op,                        ///< [in] Binary boolean flag predicate
        FlagT           (&flags)[ITEMS_PER_THREAD])     ///< [out] Discontinuity flags
    {
        T last_tile_item;   // discard
        Flag(smem_storage, input, flag_op, flags, last_tile_item);
    }



    /**
     * \brief Sets discontinuity flags for a tile of threadblock items.  The last tile item of the last thread is also returned to thread<sub>0</sub>.
     *
     * Assuming a <em>blocked</em> arrangement of elements across threads,
     * <tt>flags</tt><sub><em>i</em></sub> is set non-zero for item
     * <tt>input</tt><sub><em>i</em></sub> when <tt>scan_op(</tt><em>previous-item</em>, <tt>input<sub><em>i</em></sub>)</tt>
     * is \p true (where <em>previous-item</em> is either <tt>input<sub><em>i-1</em></sub></tt>,
     * or <tt>input<sub><em>ITEMS_PER_THREAD</em>-1</sub></tt> in the previous thread).  For
     * <em>thread</em><sub>0</sub>, item <tt>input<sub>0</sub></tt> is compared
     * against /p tile_prefix.
     *
     * The \p tile_prefix and \p last_tile_item are undefined in threads other than <em>thread</em><sub>0</sub>.
     *
     * \smemreuse
     *
     * \tparam ITEMS_PER_THREAD     <b>[inferred]</b> The number of consecutive items partitioned onto each thread.
     * \tparam FlagT                 <b>[inferred]</b> The flag type (must be an integer type)
     * \tparam FlagOp               <b>[inferred]</b> Binary boolean functor type, having input parameters <tt>(const T &a, const T &b)</tt> and returning \p true if a discontinuity exists between \p a and \p b, otherwise \p false.
     */
    template <
        int             ITEMS_PER_THREAD,
        typename        FlagT,
        typename        FlagOp>
    static __device__ __forceinline__ void Flag(
        SmemStorage     &smem_storage,                  ///< [in] Shared reference to opaque SmemStorage layout
        T               (&input)[ITEMS_PER_THREAD],     ///< [in] Input items
        T               tile_prefix,                    ///< [in] <b>[<em>thread</em><sub>0</sub> only]</b> Item with which to compare the first tile item (<tt>input<sub>0</sub></tt>from <em>thread</em><sub>0</sub>).
        FlagOp          flag_op,                        ///< [in] Binary boolean flag predicate
        FlagT           (&flags)[ITEMS_PER_THREAD],     ///< [out] Discontinuity flags
        T               &last_tile_item)                ///< [out] <b>[<em>thread</em><sub>0</sub> only]</b> The last tile item (<tt>input<sub><em>ITEMS_PER_THREAD</em>-1</sub></tt> from <em>thread</em><sub><tt><em>BLOCK_THREADS</em></tt>-1</sub>)
    {
        // Share last item
        smem_storage.last_items[threadIdx.x] = input[ITEMS_PER_THREAD - 1];

        __syncthreads();

        // Set flag for first item
        int prefix;
        if (threadIdx.x == 0)
        {
            prefix = tile_prefix;
            last_tile_item = smem_storage.last_items[BLOCK_THREADS - 1];
        }
        else
        {
            prefix = smem_storage.last_items[threadIdx.x - 1];
        }
        flags[0] = flag_op(prefix, input[0]);


        // Set flags for remaining items
        #pragma unroll
        for (int ITEM = 1; ITEM < ITEMS_PER_THREAD; ITEM++)
        {
            flags[ITEM] = flag_op(input[ITEM - 1], input[ITEM]);
        }
    }


    /**
     * \brief Sets discontinuity flags for a tile of threadblock items.
     *
     * Assuming a <em>blocked</em> arrangement of elements across threads,
     * <tt>flags</tt><sub><em>i</em></sub> is set non-zero for item
     * <tt>input</tt><sub><em>i</em></sub> when <tt>scan_op(</tt><em>previous-item</em>, <tt>input<sub><em>i</em></sub>)</tt>
     * is \p true (where <em>previous-item</em> is either <tt>input<sub><em>i-1</em></sub></tt>,
     * or <tt>input<sub><em>ITEMS_PER_THREAD</em>-1</sub></tt> in the previous thread).  For
     * <em>thread</em><sub>0</sub>, item <tt>input<sub>0</sub></tt> is compared
     * against /p tile_prefix.
     *
     * The \p tile_prefix and \p last_tile_item are undefined in threads other than <em>thread</em><sub>0</sub>.
     *
     * \smemreuse
     *
     * \tparam ITEMS_PER_THREAD     <b>[inferred]</b> The number of consecutive items partitioned onto each thread.
     * \tparam FlagT                 <b>[inferred]</b> The flag type (must be an integer type)
     * \tparam FlagOp               <b>[inferred]</b> Binary boolean functor type, having input parameters <tt>(const T &a, const T &b)</tt> and returning \p true if a discontinuity exists between \p a and \p b, otherwise \p false.
     */
    template <
        int             ITEMS_PER_THREAD,
        typename        FlagT,
        typename        FlagOp>
    static __device__ __forceinline__ void Flag(
        SmemStorage     &smem_storage,                  ///< [in] Shared reference to opaque SmemStorage layout
        T               (&input)[ITEMS_PER_THREAD],     ///< [in] Input items
        T               tile_prefix,                    ///< [in] <b>[<em>thread</em><sub>0</sub> only]</b> Item with which to compare the first tile item (<tt>input<sub>0</sub></tt>from <em>thread</em><sub>0</sub>).
        FlagOp          flag_op,                        ///< [in] Binary boolean flag predicate
        FlagT           (&flags)[ITEMS_PER_THREAD])     ///< [out] Discontinuity flags
    {
        T last_tile_item;   // discard
        Flag(smem_storage, input, tile_prefix, flag_op, flags, last_tile_item);
    }

};


/** @} */       // end of SimtCoop group

} // namespace cub
CUB_NS_POSTFIX
