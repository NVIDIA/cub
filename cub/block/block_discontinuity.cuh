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
 * cub::BlockDiscontinuity provides operations for flagging discontinuities within a list of data items partitioned across a CUDA threadblock.
 */

#pragma once

#include <cuda_runtime.h>
#include "../util_arch.cuh"
#include "../util_type.cuh"
#include "../thread/thread_operators.cuh"
#include "../util_namespace.cuh"

/// Optional outer namespace(s)
CUB_NS_PREFIX

/// CUB namespace
namespace cub {

/**
 * \brief BlockDiscontinuity provides operations for flagging discontinuities within a list of data items partitioned across a CUDA threadblock. ![](discont_logo.png)
 * \ingroup BlockModule
 *
 * \par Overview
 * The operations exposed by BlockDiscontinuity allow threadblocks to set "head flags" for data elements that
 * are different from their predecessor (as specified by a binary boolean operator).  Head flags are often useful for
 * orchestrating segmented scans and reductions.
 *
 * \par
 * For convenience, BlockDiscontinuity provides alternative entrypoints that differ by:
 * - How the first item is handled (always-flagged <em>vs.</em> compared to a specific block-wide predecessor)
 * - What is computed (discontinuity flags only <em>vs.</em> discontinuity flags and a copy of the last tile item for thread<sub><em>0</em></sub>)
 *
 * \tparam T                    The data type to be exchanged.
 * \tparam BLOCK_THREADS        The threadblock size in threads.
 *
 * \par Usage Considerations
 * - Assumes a [<em>blocked arrangement</em>](index.html#sec3sec3) of elements across threads
 * - Any threadblock-wide scalar inputs and outputs (e.g., \p tile_predecessor and \p last_tile_item) are
 *   only considered valid in <em>thread</em><sub>0</sub>
 * - \smemreuse{BlockDiscontinuity::SmemStorage}
 *
 * \par Performance Considerations
 * - Zero bank conflicts for most types.
 *
 * \par Examples
 * <em>Example 1.</em> Given a tile of 512 non-zero matrix coordinates (ordered by row) in
 * a blocked arrangement across a 128-thread threadblock, flag the first coordinate
 * element of each row.
 * \code
 * #include <cub/cub.cuh>
 *
 * // Non-zero matrix coordinates
 * struct NonZero
 * {
 *     int row;
 *     int col;
 *     float val;
 * };
 *
 * // Functor for detecting row discontinuities.
 * struct NewRowOp
 * {
 *     // Returns true if row_b is the start of a new row
 *     __device__ __forceinline__ bool operator()(
 *         const NonZero& a,
 *         const NonZero& b)
 *     {
 *         return (a.row != b.row);
 *     }
 * };
 *
 * __global__ void SomeKernel(...)
 * {
 *     // Parameterize BlockDiscontinuity for 128 threads on type NonZero
 *     typedef cub::BlockDiscontinuity<NonZero, 128> BlockDiscontinuity;
 *
 *     // Declare shared memory for BlockDiscontinuity
 *     __shared__ typename BlockDiscontinuity::SmemStorage smem_storage;
 *
 *     // A segment of consecutive non-zeroes per thread
 *     NonZero coordinates[4];
 *
 *     // Obtain items in blocked order
 *     ...
 *
 *     // Obtain the last item of the previous tile
 *     NonZero block_predecessor;
 *     if (threadIdx.x == 0)
 *     {
 *         block_predecessor = ...
 *     }
 *
 *     // Set head flags
 *     int head_flags[4];
 *     BlockDiscontinuity::Flag(smem_storage, coordinates, block_predecessor, NewRowOp(), head_flags);
 *
 * \endcode
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
    struct _SmemStorage
    {
        T last_items[BLOCK_THREADS];      ///< Last element from each thread's input
    };

    // Specialization for when FlagOp has third index param
    template <typename FlagOp, bool HAS_PARAM = BinaryOpHasIdxParam<T, FlagOp>::HAS_PARAM>
    struct ApplyOp
    {
        // Apply flag operator
        static __device__ __forceinline__ bool Flag(FlagOp flag_op, const T &a, const T &b, unsigned int idx)
        {
            return flag_op(a, b, idx);
        }
    };

    // Specialization for when FlagOp does not have a third index param
    template <typename FlagOp>
    struct ApplyOp<FlagOp, false>
    {
        // Apply flag operator
        static __device__ __forceinline__ bool Flag(FlagOp flag_op, const T &a, const T &b, unsigned int idx)
        {
            return flag_op(a, b);
        }
    };


public:

    /// \smemstorage{BlockDiscontinuity}
    typedef _SmemStorage SmemStorage;


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
     * \tparam FlagT                <b>[inferred]</b> The flag type (must be an integer type)
     * \tparam FlagOp               <b>[inferred]</b> Binary predicate functor type having member <tt>T operator()(const T &a, const T &b)</tt> or member <tt>T operator()(const T &a, const T &b, unsigned int b_index)</tt>, and returning \p true if a discontinuity exists between \p a and \p b, otherwise \p false.  \p b_index is the rank of b in the aggregate tile of data.
     */
    template <
        int             ITEMS_PER_THREAD,
        typename        FlagT,
        typename        FlagOp>
    static __device__ __forceinline__ void Flag(
        SmemStorage     &smem_storage,                  ///< [in] Reference to shared memory allocation having layout type SmemStorage
        T               (&input)[ITEMS_PER_THREAD],     ///< [in] Calling thread's input items
        FlagOp          flag_op,                        ///< [in] Binary boolean flag predicate
        FlagT           (&flags)[ITEMS_PER_THREAD],     ///< [out] Calling thread's discontinuity flags
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
            flags[0] = ApplyOp<FlagOp>::Flag(
                flag_op,
                smem_storage.last_items[threadIdx.x - 1],
                input[0],
                threadIdx.x * ITEMS_PER_THREAD);
        }

        // Set flags for remaining items
        #pragma unroll
        for (int ITEM = 1; ITEM < ITEMS_PER_THREAD; ITEM++)
        {
            flags[ITEM] = ApplyOp<FlagOp>::Flag(
                flag_op,
                input[ITEM - 1],
                input[ITEM],
                (threadIdx.x * ITEMS_PER_THREAD) + ITEM);
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
     * \tparam FlagT                <b>[inferred]</b> The flag type (must be an integer type)
     * \tparam FlagOp               <b>[inferred]</b> Binary predicate functor type having member <tt>T operator()(const T &a, const T &b)</tt> or member <tt>T operator()(const T &a, const T &b, unsigned int b_index)</tt>, and returning \p true if a discontinuity exists between \p a and \p b, otherwise \p false.  \p b_index is the rank of b in the aggregate tile of data.
     */
    template <
        int             ITEMS_PER_THREAD,
        typename        FlagT,
        typename        FlagOp>
    static __device__ __forceinline__ void Flag(
        SmemStorage     &smem_storage,                  ///< [in] Reference to shared memory allocation having layout type SmemStorage
        T               (&input)[ITEMS_PER_THREAD],     ///< [in] Calling thread's input items
        FlagOp          flag_op,                        ///< [in] Binary boolean flag predicate
        FlagT           (&flags)[ITEMS_PER_THREAD])     ///< [out] Calling thread's discontinuity flags
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
     * against \p tile_predecessor.
     *
     * The \p tile_predecessor and \p last_tile_item are undefined in threads other than <em>thread</em><sub>0</sub>.
     *
     * \smemreuse
     *
     * \tparam ITEMS_PER_THREAD     <b>[inferred]</b> The number of consecutive items partitioned onto each thread.
     * \tparam FlagT                <b>[inferred]</b> The flag type (must be an integer type)
     * \tparam FlagOp               <b>[inferred]</b> Binary predicate functor type having member <tt>T operator()(const T &a, const T &b)</tt> or member <tt>T operator()(const T &a, const T &b, unsigned int b_index)</tt>, and returning \p true if a discontinuity exists between \p a and \p b, otherwise \p false.  \p b_index is the rank of b in the aggregate tile of data.
     */
    template <
        int             ITEMS_PER_THREAD,
        typename        FlagT,
        typename        FlagOp>
    static __device__ __forceinline__ void Flag(
        SmemStorage     &smem_storage,                  ///< [in] Reference to shared memory allocation having layout type SmemStorage
        T               (&input)[ITEMS_PER_THREAD],     ///< [in] Calling thread's input items
        T               tile_predecessor,               ///< [in] <b>[<em>thread</em><sub>0</sub> only]</b> Item with which to compare the first tile item (<tt>input<sub>0</sub></tt>from <em>thread</em><sub>0</sub>).
        FlagOp          flag_op,                        ///< [in] Binary boolean flag predicate
        FlagT           (&flags)[ITEMS_PER_THREAD],     ///< [out] Calling thread's discontinuity flags
        T               &last_tile_item)                ///< [out] <b>[<em>thread</em><sub>0</sub> only]</b> The last tile item (<tt>input<sub><em>ITEMS_PER_THREAD</em>-1</sub></tt> from <em>thread</em><sub><tt><em>BLOCK_THREADS</em></tt>-1</sub>)
    {
        // Share last item
        smem_storage.last_items[threadIdx.x] = input[ITEMS_PER_THREAD - 1];

        __syncthreads();

        // Set flag for first item
        int prefix;
        if (threadIdx.x == 0)
        {
            prefix = tile_predecessor;
            last_tile_item = smem_storage.last_items[BLOCK_THREADS - 1];
        }
        else
        {
            prefix = smem_storage.last_items[threadIdx.x - 1];
        }
        flags[0] = ApplyOp<FlagOp>::Flag(
            flag_op,
            prefix,
            input[0],
            threadIdx.x * ITEMS_PER_THREAD);

        // Set flags for remaining items
        #pragma unroll
        for (int ITEM = 1; ITEM < ITEMS_PER_THREAD; ITEM++)
        {
            flags[ITEM] = ApplyOp<FlagOp>::Flag(
                flag_op,
                input[ITEM - 1],
                input[ITEM],
                (threadIdx.x * ITEMS_PER_THREAD) + ITEM);
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
     * against \p tile_predecessor.
     *
     * The \p tile_predecessor and \p last_tile_item are undefined in threads other than <em>thread</em><sub>0</sub>.
     *
     * \smemreuse
     *
     * \tparam ITEMS_PER_THREAD     <b>[inferred]</b> The number of consecutive items partitioned onto each thread.
     * \tparam FlagT                <b>[inferred]</b> The flag type (must be an integer type)
     * \tparam FlagOp               <b>[inferred]</b> Binary predicate functor type having member <tt>T operator()(const T &a, const T &b)</tt> or member <tt>T operator()(const T &a, const T &b, unsigned int b_index)</tt>, and returning \p true if a discontinuity exists between \p a and \p b, otherwise \p false.  \p b_index is the rank of b in the aggregate tile of data.
     */
    template <
        int             ITEMS_PER_THREAD,
        typename        FlagT,
        typename        FlagOp>
    static __device__ __forceinline__ void Flag(
        SmemStorage     &smem_storage,                  ///< [in] Reference to shared memory allocation having layout type SmemStorage
        T               (&input)[ITEMS_PER_THREAD],     ///< [in] Calling thread's input items
        T               tile_predecessor,               ///< [in] <b>[<em>thread</em><sub>0</sub> only]</b> Item with which to compare the first tile item (<tt>input<sub>0</sub></tt>from <em>thread</em><sub>0</sub>).
        FlagOp          flag_op,                        ///< [in] Binary boolean flag predicate
        FlagT           (&flags)[ITEMS_PER_THREAD])     ///< [out] Calling thread's discontinuity flags
    {
        T last_tile_item;   // discard
        Flag(smem_storage, input, tile_predecessor, flag_op, flags, last_tile_item);
    }

};


}               // CUB namespace
CUB_NS_POSTFIX  // Optional outer namespace(s)
