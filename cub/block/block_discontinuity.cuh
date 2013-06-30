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
 * cub::BlockDiscontinuity provides operations for flagging discontinuities within a linear tile of items partitioned across a CUDA threadblock.
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
 * \brief BlockDiscontinuity provides operations for flagging discontinuities within a linear tile of items partitioned across a CUDA threadblock. ![](discont_logo.png)
 * \ingroup BlockModule
 *
 * \par Overview
 * The operations exposed by BlockDiscontinuity allow threadblocks to set
 * "head flags" (or "tail flags") corresponding to items
 * that differ from their predecessors (or successors).  The caller specifies
 * a binary boolean operator to perform the comparison.  Computing a set of head
 * (or tail) flags across a blocked arrangement of items is often useful
 * for orchestrating block-wide segmented scans and reductions.
 *
 * \par
 * For convenience, BlockDiscontinuity provides alternative entrypoints that differ by:
 * - The item that gets flagged when a discontinuity is detected (head flagging <b><em>vs.</em></b> tail flagging)
 * - Flagging of the first/last item in the tile (always-flagged <b><em>vs.</em></b> compared to a specific block-wide predecessor/successor)
 *
 * \tparam T                    The data type to be flagged.
 * \tparam BLOCK_THREADS        The threadblock size in threads.
 *
 * \par Usage Considerations
 * - Assumes a [<em>blocked arrangement</em>](index.html#sec3sec3) of tile items across the thread block
 * - \p tile_prefix_item is only considered valid in <em>thread</em><sub>0</sub>
 * - \p tile_suffix_item is only considered valid in <em>thread</em><sub><tt>BLOCK_THREADS</tt>-1</sub>
 * - \smemreuse{BlockDiscontinuity::TempStorage}
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
 *     __shared__ typename BlockDiscontinuity::TempStorage temp_storage;
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
 *     // Set head head_flags
 *     int head_flags[4];
 *     BlockDiscontinuity::Flag(temp_storage, coordinates, block_predecessor, NewRowOp(), head_flags);
 *
 * \endcode
 */
template <
    typename    T,
    int         BLOCK_THREADS>
class BlockDiscontinuity
{
private:

    /******************************************************************************
     * Type definitions
     ******************************************************************************/

    /// Shared memory storage layout type (last element from each thread's input)
    typedef T _TempStorage[BLOCK_THREADS];


    /******************************************************************************
     * Utility methods
     ******************************************************************************/

    /// Internal storage allocator
    __device__ __forceinline__ _TempStorage& PrivateStorage()
    {
        __shared__ _TempStorage private_storage;
        return private_storage;
    }


    /// Specialization for when FlagOp has third index param
    template <typename FlagOp, bool HAS_PARAM = BinaryOpHasIdxParam<T, FlagOp>::HAS_PARAM>
    struct ApplyOp
    {
        // Apply flag operator
        static __device__ __forceinline__ bool Flag(FlagOp flag_op, const T &a, const T &b, int idx)
        {
            return flag_op(a, b, idx);
        }
    };

    /// Specialization for when FlagOp does not have a third index param
    template <typename FlagOp>
    struct ApplyOp<FlagOp, false>
    {
        // Apply flag operator
        static __device__ __forceinline__ bool Flag(FlagOp flag_op, const T &a, const T &b, int idx)
        {
            return flag_op(a, b);
        }
    };


    /******************************************************************************
     * Thread fields
     ******************************************************************************/

    /// Shared storage reference
    _TempStorage &temp_storage;

    /// Linear thread-id
    int linear_tid;


public:

    /// \smemstorage{BlockDiscontinuity}
    typedef _TempStorage TempStorage;


    /******************************************************************//**
     * \name Collective construction
     *********************************************************************/
    //@{

    /**
     * \brief Collective constructor for 1D thread blocks using a private static allocation of shared memory as temporary storage.  Threads are identified using <tt>threadIdx.x</tt>.
     */
    __device__ __forceinline__ BlockDiscontinuity()
    :
        temp_storage(PrivateStorage()),
        linear_tid(threadIdx.x)
    {}


    /**
     * \brief Collective constructor for 1D thread blocks using the specified memory allocation as temporary storage.  Threads are identified using <tt>threadIdx.x</tt>.
     */
    __device__ __forceinline__ BlockDiscontinuity(
        TempStorage &temp_storage)  ///< [in] Reference to memory allocation having layout type TempStorage
    :
        temp_storage(temp_storage),
        linear_tid(threadIdx.x)
    {}


    /**
     * \brief Collective constructor using a private static allocation of shared memory as temporary storage.  Threads are identified using the given linear thread identifier
     */
    __device__ __forceinline__ BlockDiscontinuity(
        int linear_tid)             ///< [in] A suitable 1D thread-identifier for the calling thread (e.g., <tt>(threadIdx.y * blockDim.x) + linear_tid</tt> for 2D thread blocks)
    :
        temp_storage(PrivateStorage()),
        linear_tid(linear_tid)
    {}


    /**
     * \brief Collective constructor using the specified memory allocation as temporary storage.  Threads are identified using the given linear thread identifier.
     */
    __device__ __forceinline__ BlockDiscontinuity(
        TempStorage &temp_storage,  ///< [in] Reference to memory allocation having layout type TempStorage
        int linear_tid)             ///< [in] <b>[optional]</b> A suitable 1D thread-identifier for the calling thread (e.g., <tt>(threadIdx.y * blockDim.x) + linear_tid</tt> for 2D thread blocks)
    :
        temp_storage(temp_storage),
        linear_tid(linear_tid)
    {}



    //@}  end member group
    /******************************************************************//**
     * \name Head flags
     *********************************************************************/
    //@{


    /**
     * \brief Sets head flags for a [<em>blocked arrangement</em>](index.html#sec3sec3) of tile items across the thread block, for which the first item has no reference and is always flagged.
     *
     * The flag <tt>head_flags<sub><em>i</em></sub></tt> is set for item
     * <tt>input<sub><em>i</em></sub></tt> when
     * <tt>flag_op(</tt><em>previous-item</em><tt>, input<sub><em>i</em></sub>)</tt>
     * returns \p true (where <em>previous-item</em> is either the preceding item
     * in the same thread or the last item in the previous thread).
     * Furthermore, <tt>head_flags<sub><em>i</em></sub></tt> is always set for
     * <tt>input><sub>0</sub></tt> in <em>thread</em><sub>0</sub>.
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
    __device__ __forceinline__ void FlagHeads(
        FlagT           (&head_flags)[ITEMS_PER_THREAD],    ///< [out] Calling thread's discontinuity head_flags
        T               (&input)[ITEMS_PER_THREAD],         ///< [in] Calling thread's input items
        FlagOp          flag_op)                            ///< [in] Binary boolean flag predicate
    {
        // Share last item
        temp_storage[linear_tid] = input[ITEMS_PER_THREAD - 1];

        __syncthreads();

        // Set flag for first item
        head_flags[0] = (linear_tid == 0) ?
            1 :                                 // First thread
            ApplyOp<FlagOp>::Flag(
                flag_op,
                temp_storage[linear_tid - 1],
                input[0],
                linear_tid * ITEMS_PER_THREAD);

        // Set head_flags for remaining items
        #pragma unroll
        for (int ITEM = 1; ITEM < ITEMS_PER_THREAD; ITEM++)
        {
            head_flags[ITEM] = ApplyOp<FlagOp>::Flag(
                flag_op,
                input[ITEM - 1],
                input[ITEM],
                (linear_tid * ITEMS_PER_THREAD) + ITEM);
        }
    }


    /**
     * \brief Sets head flags for a [<em>blocked arrangement</em>](index.html#sec3sec3) of tile items across the thread block
     *
     * The flag <tt>head_flags<sub><em>i</em></sub></tt> is set for item
     * <tt>input<sub><em>i</em></sub></tt> when
     * <tt>flag_op(</tt><em>previous-item</em><tt>, input<sub><em>i</em></sub>)</tt>
     * returns \p true (where <em>previous-item</em> is either the preceding item
     * in the same thread or the last item in the previous thread).
     * For <em>thread</em><sub>0</sub>, item <tt>input<sub>0</sub></tt> is compared
     * against \p tile_prefix_item.
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
    __device__ __forceinline__ void FlagHeads(
        FlagT           (&head_flags)[ITEMS_PER_THREAD],    ///< [out] Calling thread's discontinuity head_flags
        T               (&input)[ITEMS_PER_THREAD],         ///< [in] Calling thread's input items
        FlagOp          flag_op,                            ///< [in] Binary boolean flag predicate
        T               tile_prefix_item)                   ///< [in] <b>[<em>thread</em><sub>0</sub> only]</b> Item with which to compare the first tile item (<tt>input<sub>0</sub></tt> from <em>thread</em><sub>0</sub>).
    {
        // Share last item
        temp_storage[linear_tid] = input[ITEMS_PER_THREAD - 1];

        __syncthreads();

        // Set flag for first item
        int prefix = (linear_tid == 0) ?
            tile_prefix_item :              // First thread
            temp_storage[linear_tid - 1];

        head_flags[0] = ApplyOp<FlagOp>::Flag(
            flag_op,
            prefix,
            input[0],
            linear_tid * ITEMS_PER_THREAD);

        // Set flag for remaining items
        #pragma unroll
        for (int ITEM = 1; ITEM < ITEMS_PER_THREAD; ITEM++)
        {
            head_flags[ITEM] = ApplyOp<FlagOp>::Flag(
                flag_op,
                input[ITEM - 1],
                input[ITEM],
                (linear_tid * ITEMS_PER_THREAD) + ITEM);
        }
    }


    //@}  end member group
    /******************************************************************//**
     * \name Tail flags
     *********************************************************************/
    //@{


    /**
     * \brief Sets tail flags for a [<em>blocked arrangement</em>](index.html#sec3sec3) of tile items across the thread block, for which the last item has no reference and is always flagged.
     *
     * The flag <tt>tail_flags<sub><em>i</em></sub></tt> is set for item
     * <tt>input<sub><em>i</em></sub></tt> when
     * <tt>flag_op(input<sub><em>i</em></sub>, </tt><em>next-item</em><tt>)</tt>
     * returns \p true (where <em>next-item</em> is either the next item
     * in the same thread or the first item in the next thread).
     * Furthermore, <tt>tail_flags<sub>ITEMS_PER_THREAD-1</sub></tt> is always
     * set for <em>thread</em><sub><tt>BLOCK_THREADS</tt>-1</sub>.
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
    __device__ __forceinline__ void FlagTails(
        FlagT           (&tail_flags)[ITEMS_PER_THREAD],    ///< [out] Calling thread's discontinuity tail_flags
        T               (&input)[ITEMS_PER_THREAD],         ///< [in] Calling thread's input items
        FlagOp          flag_op)                            ///< [in] Binary boolean flag predicate
    {
        // Share first item
        temp_storage[linear_tid] = input[0];

        __syncthreads();

        // Set flag for last item
        tail_flags[ITEMS_PER_THREAD - 1] = (linear_tid == BLOCK_THREADS - 1) ?
            1 :                             // Last thread
            ApplyOp<FlagOp>::Flag(
                flag_op,
                input[ITEMS_PER_THREAD - 1],
                temp_storage[linear_tid + 1],
                (linear_tid * ITEMS_PER_THREAD) + (ITEMS_PER_THREAD - 1));

        // Set flags for remaining items
        #pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD - 1; ITEM++)
        {
            tail_flags[ITEM] = ApplyOp<FlagOp>::Flag(
                flag_op,
                input[ITEM],
                input[ITEM + 1],
                (linear_tid * ITEMS_PER_THREAD) + ITEM);
        }
    }


    /**
     * \brief Sets tail flags for a [<em>blocked arrangement</em>](index.html#sec3sec3) of tile items across the thread block.
     *
     * The flag <tt>tail_flags<sub><em>i</em></sub></tt> is set for item
     * <tt>input<sub><em>i</em></sub></tt> when
     * <tt>flag_op(input<sub><em>i</em></sub>, </tt><em>next-item</em><tt>)</tt>
     * returns \p true (where <em>next-item</em> is either the next item
     * in the same thread or the first item in the next thread).
     * For <em>thread</em><sub><em>BLOCK_THREADS</em>-1</sub>, item
     * <tt>input</tt><sub><em>ITEMS_PER_THREAD</em>-1</sub> is compared
     * against \p tile_prefix_item.
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
    __device__ __forceinline__ void FlagTails(
        FlagT           (&tail_flags)[ITEMS_PER_THREAD],    ///< [out] Calling thread's discontinuity tail_flags
        T               (&input)[ITEMS_PER_THREAD],         ///< [in] Calling thread's input items
        FlagOp          flag_op,                            ///< [in] Binary boolean flag predicate
        T               tile_suffix_item)                   ///< [in] <b>[<em>thread</em><sub><tt>BLOCK_THREADS</tt>-1</sub> only]</b> Item with which to compare the last tile item (<tt>input</tt><sub><em>ITEMS_PER_THREAD</em>-1</sub> from <em>thread</em><sub><em>BLOCK_THREADS</em>-1</sub>).
    {
        // Share first item
        temp_storage[linear_tid] = input[0];

        __syncthreads();

        // Set flag for last item
        int suffix_item = (linear_tid == BLOCK_THREADS - 1) ?
            tile_suffix_item :              // Last thread
            temp_storage[linear_tid + 1];

        tail_flags[ITEMS_PER_THREAD - 1] = ApplyOp<FlagOp>::Flag(
            flag_op,
            input[ITEMS_PER_THREAD - 1],
            suffix_item,
            (linear_tid * ITEMS_PER_THREAD) + (ITEMS_PER_THREAD - 1));

        // Set flags for remaining items
        #pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD - 1; ITEM++)
        {
            tail_flags[ITEM] = ApplyOp<FlagOp>::Flag(
                flag_op,
                input[ITEM],
                input[ITEM + 1],
                (linear_tid * ITEMS_PER_THREAD) + ITEM);
        }
    }

    //@}  end member group

};


}               // CUB namespace
CUB_NS_POSTFIX  // Optional outer namespace(s)
