/******************************************************************************
 * Copyright (c) 2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2021, NVIDIA CORPORATION.  All rights reserved.
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
 * The cub::BlockAdjacentDifference class provides [<em>collective</em>](index.html#sec0) methods for computing the differences of adjacent elements partitioned across a CUDA thread block.
 */

#pragma once

#include "../config.cuh"
#include "../util_type.cuh"
#include "../util_ptx.cuh"

/// Optional outer namespace(s)
CUB_NS_PREFIX

/// CUB namespace
namespace cub {

/**
 * \brief BlockAdjacentDifference provides [<em>collective</em>](index.html#sec0) methods for computing the differences of adjacent elements partitioned across a CUDA thread block.
 * \ingroup SingleModule
 *
 * \par Overview
 * - BlockAdjacentDifference calculates the differences of adjacent elements in
 *   the elements partitioned across a CUDA thread block. Because the binary
 *   operation could be noncommutative, there are two sets of methods.
 *   Methods named SubtractLeft subtract left element <tt>i - 1</tt> of
 *   input sequence from current element <tt>i</tt>. Methods named SubtractRight
 *   subtract current element <tt>i</tt> from the right one <tt>i + 1</tt>:
 *   \par
 *   \code
 *   int values[4]; // [1, 2, 3, 4]
 *   //...
 *   int subtract_left_result[4];  <-- [  1,  1,  1,  1 ]
 *   int subtract_right_result[4]; <-- [ -1, -1, -1,  4 ]
 *   \endcode
 * - For SubtractLeft, if the left element <tt>i - 1 < 0</tt> is out of bounds, the
 *   output value is assigned to <tt>input[0]</tt> without modification.
 * - For SubtractRight, if the right element is out of bounds, the iterator is
 *   assigned to corresponding input value without modification.
 *
 * \par Snippet
 * The code snippet below illustrates how to use \p BlockAdjacentDifference to
 * compute the left difference between adjacent elements.
 *
 * \par
 * \code
 * #include <cub/cub.cuh>   // or equivalently <cub/block/block_adjacent_difference.cuh>
 *
 * struct CustomDifference
 * {
 *   template <typename DataType>
 *   __device__ DataType operator()(DataType &lhs, DataType &rhs)
 *   {
 *     return lhs - rhs;
 *   }
 * };
 *
 * __global__ void ExampleKernel(...)
 * {
 *     // Specialize BlockAdjacentDifference for a 1D block of 128 threads on type int
 *     using BlockAdjacentDifferenceT =
 *        cub::BlockAdjacentDifference<int, 128>;
 *
 *     // Allocate shared memory for BlockDiscontinuity
 *     __shared__ typename BlockAdjacentDifferenceT::TempStorage temp_storage;
 *
 *     // Obtain a segment of consecutive items that are blocked across threads
 *     int thread_data[4];
 *     ...
 *
 *     // Collectively compute adjacent_difference
 *     int result[4];
 *     int thread_preds[4];
 *
 *     BlockAdjacentDifferenceT(temp_storage).SubtractLeft(
 *         result,
 *         thread_data,
 *         thread_preds,
 *         CustomDifference());
 *
 * \endcode
 * \par
 * Suppose the set of input \p thread_data across the block of threads is
 * <tt>{ [4,2,1,1], [1,1,1,1], [2,3,3,3], [3,4,1,4], ... }</tt>.
 * The corresponding output \p result in those threads will be
 * <tt>{ [4,-2,-1,0], [0,0,0,0], [1,1,0,0], [0,1,-3,3], ... }</tt>.
 *
 */
template <
    typename    T,
    int         BLOCK_DIM_X,
    int         BLOCK_DIM_Y     = 1,
    int         BLOCK_DIM_Z     = 1,
    int         PTX_ARCH        = CUB_PTX_ARCH>
class BlockAdjacentDifference
{
private:

    /******************************************************************************
     * Constants and type definitions
     ******************************************************************************/

    /// Constants

    /// The thread block size in threads
    static constexpr int BLOCK_THREADS = BLOCK_DIM_X * BLOCK_DIM_Y * BLOCK_DIM_Z;

    /// Shared memory storage layout type (last element from each thread's input)
    struct _TempStorage
    {
        T first_items[BLOCK_THREADS];
        T last_items[BLOCK_THREADS];
    };


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
        static __device__ __forceinline__ T FlagT(FlagOp flag_op, const T &a, const T &b, int idx)
        {
            return flag_op(b, a, idx);
        }
    };

    /// Specialization for when FlagOp does not have a third index param
    template <typename FlagOp>
    struct ApplyOp<FlagOp, false>
    {
        // Apply flag operator
        static __device__ __forceinline__ T FlagT(FlagOp flag_op, const T &a, const T &b, int /*idx*/)
        {
            return flag_op(b, a);
        }
    };

    /// Templated unrolling of item comparison (inductive case)
    template <int ITERATION, int MAX_ITERATIONS>
    struct Iterate
    {
        // Head flags
        template <
            int             ITEMS_PER_THREAD,
            typename        FlagT,
            typename        FlagOp>
        static __device__ __forceinline__ void FlagHeads(
            int                     linear_tid,
            FlagT                   (&flags)[ITEMS_PER_THREAD],         ///< [out] Calling thread's discontinuity head_flags
            T                       (&input)[ITEMS_PER_THREAD],         ///< [in] Calling thread's input items
            T                       (&preds)[ITEMS_PER_THREAD],         ///< [out] Calling thread's predecessor items
            FlagOp                  flag_op)                            ///< [in] Binary boolean flag predicate
        {
            preds[ITERATION] = input[ITERATION - 1];

            flags[ITERATION] = ApplyOp<FlagOp>::FlagT(
                flag_op,
                preds[ITERATION],
                input[ITERATION],
                (linear_tid * ITEMS_PER_THREAD) + ITERATION);

            Iterate<ITERATION + 1, MAX_ITERATIONS>::FlagHeads(linear_tid, flags, input, preds, flag_op);
        }

        // Tail flags
        template <
            int             ITEMS_PER_THREAD,
            typename        FlagT,
            typename        FlagOp>
        static __device__ __forceinline__ void FlagTails(
            int                     linear_tid,
            FlagT                   (&flags)[ITEMS_PER_THREAD],         ///< [out] Calling thread's discontinuity head_flags
            T                       (&input)[ITEMS_PER_THREAD],         ///< [in] Calling thread's input items
            FlagOp                  flag_op)                            ///< [in] Binary boolean flag predicate
        {
            flags[ITERATION] = ApplyOp<FlagOp>::FlagT(
                flag_op,
                input[ITERATION],
                input[ITERATION + 1],
                (linear_tid * ITEMS_PER_THREAD) + ITERATION + 1);

            Iterate<ITERATION + 1, MAX_ITERATIONS>::FlagTails(linear_tid, flags, input, flag_op);
        }

    };

    /// Templated unrolling of item comparison (termination case)
    template <int MAX_ITERATIONS>
    struct Iterate<MAX_ITERATIONS, MAX_ITERATIONS>
    {
        // Head flags
        template <
            int             ITEMS_PER_THREAD,
            typename        FlagT,
            typename        FlagOp>
        static __device__ __forceinline__ void FlagHeads(
            int                     /*linear_tid*/,
            FlagT                   (&/*flags*/)[ITEMS_PER_THREAD],         ///< [out] Calling thread's discontinuity head_flags
            T                       (&/*input*/)[ITEMS_PER_THREAD],         ///< [in] Calling thread's input items
            T                       (&/*preds*/)[ITEMS_PER_THREAD],         ///< [out] Calling thread's predecessor items
            FlagOp                  /*flag_op*/)                            ///< [in] Binary boolean flag predicate
        {}

        // Tail flags
        template <
            int             ITEMS_PER_THREAD,
            typename        FlagT,
            typename        FlagOp>
        static __device__ __forceinline__ void FlagTails(
            int                     /*linear_tid*/,
            FlagT                   (&/*flags*/)[ITEMS_PER_THREAD],         ///< [out] Calling thread's discontinuity head_flags
            T                       (&/*input*/)[ITEMS_PER_THREAD],         ///< [in] Calling thread's input items
            FlagOp                  /*flag_op*/)                            ///< [in] Binary boolean flag predicate
        {}
    };


    /******************************************************************************
     * Thread fields
     ******************************************************************************/

    /// Shared storage reference
    _TempStorage &temp_storage;

    /// Linear thread-id
    unsigned int linear_tid;


public:

    /// \smemstorage{BlockDiscontinuity}
    struct TempStorage : Uninitialized<_TempStorage> {};


    /******************************************************************//**
     * \name Collective constructors
     *********************************************************************/
    //@{

    /**
     * \brief Collective constructor using a private static allocation of shared memory as temporary storage.
     */
    __device__ __forceinline__ BlockAdjacentDifference()
    :
        temp_storage(PrivateStorage()),
        linear_tid(RowMajorTid(BLOCK_DIM_X, BLOCK_DIM_Y, BLOCK_DIM_Z))
    {}


    /**
     * \brief Collective constructor using the specified memory allocation as temporary storage.
     */
    __device__ __forceinline__ BlockAdjacentDifference(
        TempStorage &temp_storage)  ///< [in] Reference to memory allocation having layout type TempStorage
    :
        temp_storage(temp_storage.Alias()),
        linear_tid(RowMajorTid(BLOCK_DIM_X, BLOCK_DIM_Y, BLOCK_DIM_Z))
    {}


    //@}  end member group
    /******************************************************************//**
     * \name Head flag operations
     *********************************************************************/
    //@{


#ifndef DOXYGEN_SHOULD_SKIP_THIS    // Do not document

    /**
     * \brief Subtracts the left element of each adjacent pair of elements partitioned across a CUDA thread block.
     * \ingroup SingleModule
     *
     * \par Snippet
     * The code snippet below illustrates how to use \p BlockAdjacentDifference to
     * compute the left difference between adjacent elements.
     *
     * \par
     * \code
     * #include <cub/cub.cuh>   // or equivalently <cub/block/block_adjacent_difference.cuh>
     *
     * struct CustomDifference
     * {
     *   template <typename DataType>
     *   __device__ DataType operator()(DataType &lhs, DataType &rhs)
     *   {
     *     return lhs - rhs;
     *   }
     * };
     *
     * __global__ void ExampleKernel(...)
     * {
     *     // Specialize BlockAdjacentDifference for a 1D block of 128 threads on type int
     *     using BlockAdjacentDifferenceT =
     *        cub::BlockAdjacentDifference<int, 128>;
     *
     *     // Allocate shared memory for BlockDiscontinuity
     *     __shared__ typename BlockAdjacentDifferenceT::TempStorage temp_storage;
     *
     *     // Obtain a segment of consecutive items that are blocked across threads
     *     int thread_data[4];
     *     ...
     *
     *     // Collectively compute adjacent_difference
     *     int result[4];
     *     int thread_preds[4];
     *
     *     BlockAdjacentDifferenceT(temp_storage).SubtractLeft(
     *         result,
     *         thread_data,
     *         thread_preds,
     *         CustomDifference());
     *
     * \endcode
     * \par
     * Suppose the set of input \p thread_data across the block of threads is
     * <tt>{ [4,2,1,1], [1,1,1,1], [2,3,3,3], [3,4,1,4], ... }</tt>.
     * The corresponding output \p result in those threads will be
     * <tt>{ [4,-2,-1,0], [0,0,0,0], [1,1,0,0], [0,1,-3,3], ... }</tt>.
     */
    template <int ITEMS_PER_THREAD,
              typename OutputType,
              typename DifferenceOpT>
    __device__ __forceinline__ void
    SubtractLeft(OutputType (&output)[ITEMS_PER_THREAD], ///< [out] Calling thread's adjacent difference result
                 T (&input)[ITEMS_PER_THREAD],           ///< [in] Calling thread's input items
                 T (&preds)[ITEMS_PER_THREAD],           ///< [out] Calling thread's predecessor items
                 DifferenceOpT difference_op)            ///< [in] Binary difference operator
    {
      // Share last item
      temp_storage.last_items[linear_tid] = input[ITEMS_PER_THREAD - 1];

      CTA_SYNC();

      if (linear_tid == 0)
      {
        // preds[0] is undefined
        output[0] = input[0];
      }
      else
      {
        preds[0]  = temp_storage.last_items[linear_tid - 1];
        output[0] = difference_op(input[0], preds[0]);
      }

      #pragma unroll
      for (int item = 1; item < ITEMS_PER_THREAD; item++)
      {
        preds[item] = input[item - 1];
        output[item] = difference_op(input[item], preds[item]);
      }
    }

    /**
     * \brief Subtracts the left element of each adjacent pair of elements partitioned across a CUDA thread block.
     * \ingroup SingleModule
     *
     * \par Snippet
     * The code snippet below illustrates how to use \p BlockAdjacentDifference to
     * compute the left difference between adjacent elements.
     *
     * \par
     * \code
     * #include <cub/cub.cuh>   // or equivalently <cub/block/block_adjacent_difference.cuh>
     *
     * struct CustomDifference
     * {
     *   template <typename DataType>
     *   __device__ DataType operator()(DataType &lhs, DataType &rhs)
     *   {
     *     return lhs - rhs;
     *   }
     * };
     *
     * __global__ void ExampleKernel(...)
     * {
     *     // Specialize BlockAdjacentDifference for a 1D block of 128 threads on type int
     *     using BlockAdjacentDifferenceT =
     *        cub::BlockAdjacentDifference<int, 128>;
     *
     *     // Allocate shared memory for BlockDiscontinuity
     *     __shared__ typename BlockAdjacentDifferenceT::TempStorage temp_storage;
     *
     *     // Obtain a segment of consecutive items that are blocked across threads
     *     int thread_data[4];
     *     ...
     *
     *     // Collectively compute adjacent_difference
     *     int result[4];
     *     int thread_preds[4];
     *
     *     BlockAdjacentDifferenceT(temp_storage).SubtractLeft(
     *         result,
     *         thread_data,
     *         thread_preds,
     *         CustomDifference(),
     *         tile_predecessor_item);
     *
     * \endcode
     * \par
     * Suppose the set of input \p thread_data across the block of threads is
     * <tt>{ [4,2,1,1], [1,1,1,1], [2,3,3,3], [3,4,1,4], ... }</tt>.
     * and that \p tile_predecessor_item is \p 3. The corresponding output \p result in those threads will be
     * <tt>{ [1,-2,-1,0], [0,0,0,0], [1,1,0,0], [0,1,-3,3], ... }</tt>.
     */
    template <int ITEMS_PER_THREAD,
              typename OutputT,
              typename DifferenceOpT>
    __device__ __forceinline__ void
    SubtractLeft(OutputT         (&head_flags)[ITEMS_PER_THREAD],    ///< [out] Calling thread's adjacent difference result
                 T               (&input)[ITEMS_PER_THREAD],         ///< [in] Calling thread's input items
                 T               (&preds)[ITEMS_PER_THREAD],         ///< [out] Calling thread's predecessor items
                 DifferenceOpT   difference_op,                      ///< [in] Binary difference operator
                 T               tile_predecessor_item)              ///< [in] <b>[<em>thread</em><sub>0</sub> only]</b> item which is going to be subtracted from the first tile item (<tt>input<sub>0</sub></tt> from <em>thread</em><sub>0</sub>).
    {
      // Share last item
      temp_storage.last_items[linear_tid] = input[ITEMS_PER_THREAD - 1];

      CTA_SYNC();

      // Set flag for first thread-item
      preds[0] = (linear_tid == 0) ? tile_predecessor_item : // First thread
                   temp_storage.last_items[linear_tid - 1];

      head_flags[0] = difference_op(input[0], preds[0]);

      #pragma unroll
      for (int item = 1; item < ITEMS_PER_THREAD; item++)
      {
        preds[item] = input[item - 1];
        head_flags[item] = difference_op(input[item], preds[item]);
      }
    }

    /**
     * \brief Subtracts the right element of each adjacent pair of elements partitioned across a CUDA thread block.
     * \ingroup SingleModule
     *
     * \par Snippet
     * The code snippet below illustrates how to use \p BlockAdjacentDifference to
     * compute the right difference between adjacent elements.
     *
     * \par
     * \code
     * #include <cub/cub.cuh>   // or equivalently <cub/block/block_adjacent_difference.cuh>
     *
     * struct CustomDifference
     * {
     *   template <typename DataType>
     *   __device__ DataType operator()(DataType &lhs, DataType &rhs)
     *   {
     *     return lhs - rhs;
     *   }
     * };
     *
     * __global__ void ExampleKernel(...)
     * {
     *     // Specialize BlockAdjacentDifference for a 1D block of 128 threads on type int
     *     using BlockAdjacentDifferenceT =
     *        cub::BlockAdjacentDifference<int, 128>;
     *
     *     // Allocate shared memory for BlockDiscontinuity
     *     __shared__ typename BlockAdjacentDifferenceT::TempStorage temp_storage;
     *
     *     // Obtain a segment of consecutive items that are blocked across threads
     *     int thread_data[4];
     *     ...
     *
     *     // Collectively compute adjacent_difference
     *     int result[4];
     *
     *     BlockAdjacentDifferenceT(temp_storage).SubtractRight(
     *         result,
     *         thread_data,
     *         CustomDifference(),
     *         tile_successor_item);
     *
     * \endcode
     * \par
     * Suppose the set of input \p thread_data across the block of threads is
     * <tt>{ ...3], [4,2,1,1], [1,1,1,1], [2,3,3,3], [3,4,1,4] }</tt>.
     * and that \p tile_successor_item is \p 3. The corresponding output \p result in those threads will be
     * <tt>{ ..., [-1,2,1,0], [0,0,0,-1], [-1,0,0,0], [-1,3,-3,1] }</tt>.
     */
    template <int ITEMS_PER_THREAD,
    typename OutputT,
    typename DifferenceOpT>
    __device__ __forceinline__ void
    SubtractRight(OutputT       (&tail_flags)[ITEMS_PER_THREAD], ///< [out] Calling thread's adjacent difference result
                  T             (&input)[ITEMS_PER_THREAD],      ///< [in] Calling thread's input items
                  DifferenceOpT difference_op,                   ///< [in] Binary difference operator
                  T             tile_successor_item)             ///< [in] <b>[<em>thread</em><sub><tt>BLOCK_THREADS</tt>-1</sub> only]</b> item which is going to be subtracted from the last tile item (<tt>input</tt><sub><em>ITEMS_PER_THREAD</em>-1</sub> from <em>thread</em><sub><em>BLOCK_THREADS</em>-1</sub>).
    {
      // Share first item
      temp_storage.first_items[linear_tid] = input[0];

      CTA_SYNC();

      // Set flag for last thread-item
      T successor_item = (linear_tid == BLOCK_THREADS - 1)
                           ? tile_successor_item // Last thread
                           : temp_storage.first_items[linear_tid + 1];

      tail_flags[ITEMS_PER_THREAD - 1] =
        difference_op(input[ITEMS_PER_THREAD - 1], successor_item);

      #pragma unroll
      for (int item = 0; item < ITEMS_PER_THREAD - 1; item++)
      {
        tail_flags[item] = difference_op(input[item], input[item + 1]);
      }
    }

    /**
     * \brief Subtracts the right element of each adjacent pair in range of elements partitioned across a CUDA thread block.
     * \ingroup SingleModule
     *
     * \par Snippet
     * The code snippet below illustrates how to use \p BlockAdjacentDifference to
     * compute the right difference between adjacent elements.
     *
     * \par
     * \code
     * #include <cub/cub.cuh>   // or equivalently <cub/block/block_adjacent_difference.cuh>
     *
     * struct CustomDifference
     * {
     *   template <typename DataType>
     *   __device__ DataType operator()(DataType &lhs, DataType &rhs)
     *   {
     *     return lhs - rhs;
     *   }
     * };
     *
     * __global__ void ExampleKernel(...)
     * {
     *     // Specialize BlockAdjacentDifference for a 1D block of 128 threads on type int
     *     using BlockAdjacentDifferenceT =
     *        cub::BlockAdjacentDifference<int, 128>;
     *
     *     // Allocate shared memory for BlockDiscontinuity
     *     __shared__ typename BlockAdjacentDifferenceT::TempStorage temp_storage;
     *
     *     // Obtain a segment of consecutive items that are blocked across threads
     *     int thread_data[4];
     *     ...
     *
     *     // Collectively compute adjacent_difference
     *     int result[4];
     *
     *     BlockAdjacentDifferenceT(temp_storage).SubtractRightPartialTile(
     *         result,
     *         thread_data,
     *         CustomDifference(),
     *         valid_items);
     *
     * \endcode
     * \par
     * Suppose the set of input \p thread_data across the block of threads is
     * <tt>{ ...3], [4,2,1,1], [1,1,1,1], [2,3,3,3], [3,4,1,4] }</tt>.
     * and that \p valid_items is \p 507. The corresponding output \p result in those threads will be
     * <tt>{ ..., [-1,2,1,0], [0,0,0,-1], [-1,0,3,3], [3,4,1,4] }</tt>.
     */
    template <int ITEMS_PER_THREAD,
              typename OutputT,
              typename DifferenceOpT>
    __device__ __forceinline__ void
    SubtractRightPartialTile(OutputT       (&tail_flags)[ITEMS_PER_THREAD], ///< [out] Calling thread's adjacent difference result
                             T             (&input)[ITEMS_PER_THREAD],      ///< [in] Calling thread's input items
                             DifferenceOpT difference_op,                   ///< [in] Binary difference operator
                             int           valid_items)                     ///< [in] Number of valid items in thread block
    {
      // Share first item
      temp_storage.first_items[linear_tid] = input[0];

      CTA_SYNC();

      if ((linear_tid + 1) * ITEMS_PER_THREAD < valid_items)
      {
        tail_flags[ITEMS_PER_THREAD - 1] =
          difference_op(input[ITEMS_PER_THREAD - 1],
                  temp_storage.first_items[linear_tid + 1]);

        #pragma unroll
        for (int item = 0; item < ITEMS_PER_THREAD - 1; item++)
        {
           tail_flags[item] = difference_op(input[item], input[item + 1]);
        }
      }
      else
      {
        #pragma unroll
        for (int item = 0; item < ITEMS_PER_THREAD; item++)
        {
          const int idx = linear_tid * ITEMS_PER_THREAD + item;

          // Right element of input[valid_items - 1] is out of bounds.
          // According to the API it's copied into output array
          // without modification.
          if (idx < valid_items - 1)
          {
            tail_flags[item] = difference_op(input[item], input[item + 1]);
          }
          else
          {
            tail_flags[item] = input[item];
          }
        }
      }
    }

    /**
     * \deprecated [Since 1.14.0] The cub::BlockAdjacentDifference::FlagHeads
     * APIs are deprecated. Use cub::BlockAdjacentDifference::SubtractLeft instead.
     */
    template <
        int             ITEMS_PER_THREAD,
        typename        FlagT,
        typename        FlagOp>
    CUB_DEPRECATED __device__ __forceinline__ void FlagHeads(
        FlagT           (&head_flags)[ITEMS_PER_THREAD],    ///< [out] Calling thread's discontinuity head_flags
        T               (&input)[ITEMS_PER_THREAD],         ///< [in] Calling thread's input items
        T               (&preds)[ITEMS_PER_THREAD],         ///< [out] Calling thread's predecessor items
        FlagOp          flag_op)                            ///< [in] Binary boolean flag predicate
    {
        // Share last item
        temp_storage.last_items[linear_tid] = input[ITEMS_PER_THREAD - 1];

        CTA_SYNC();

        if (linear_tid == 0)
        {
            // Set flag for first thread-item (preds[0] is undefined)
            head_flags[0] = 1;
        }
        else
        {
            preds[0] = temp_storage.last_items[linear_tid - 1];
            head_flags[0] = ApplyOp<FlagOp>::FlagT(flag_op, preds[0], input[0], linear_tid * ITEMS_PER_THREAD);
        }

        // Set head_flags for remaining items
        Iterate<1, ITEMS_PER_THREAD>::FlagHeads(linear_tid, head_flags, input, preds, flag_op);
    }

    /**
     * \deprecated [Since 1.14.0] The cub::BlockAdjacentDifference::FlagHeads
     * APIs are deprecated. Use cub::BlockAdjacentDifference::SubtractLeft instead.
     */
    template <int             ITEMS_PER_THREAD,
              typename        FlagT,
              typename        FlagOp>
    CUB_DEPRECATED __device__ __forceinline__ void FlagHeads(
        FlagT           (&head_flags)[ITEMS_PER_THREAD],    ///< [out] Calling thread's discontinuity head_flags
        T               (&input)[ITEMS_PER_THREAD],         ///< [in] Calling thread's input items
        T               (&preds)[ITEMS_PER_THREAD],         ///< [out] Calling thread's predecessor items
        FlagOp          flag_op,                            ///< [in] Binary boolean flag predicate
        T               tile_predecessor_item)              ///< [in] <b>[<em>thread</em><sub>0</sub> only]</b> Item with which to compare the first tile item (<tt>input<sub>0</sub></tt> from <em>thread</em><sub>0</sub>).
    {
        // Share last item
        temp_storage.last_items[linear_tid] = input[ITEMS_PER_THREAD - 1];

        CTA_SYNC();

        // Set flag for first thread-item
        preds[0] = (linear_tid == 0) ?
            tile_predecessor_item :              // First thread
            temp_storage.last_items[linear_tid - 1];

        head_flags[0] = ApplyOp<FlagOp>::FlagT(flag_op, preds[0], input[0], linear_tid * ITEMS_PER_THREAD);

        // Set head_flags for remaining items
        Iterate<1, ITEMS_PER_THREAD>::FlagHeads(linear_tid, head_flags, input, preds, flag_op);
    }

#endif // DOXYGEN_SHOULD_SKIP_THIS

    /**
     * \deprecated [Since 1.14.0] The cub::BlockAdjacentDifference::FlagHeads
     * APIs are deprecated. Use cub::BlockAdjacentDifference::SubtractLeft instead.
     */
    template <int ITEMS_PER_THREAD,
              typename FlagT,
              typename FlagOp>
    CUB_DEPRECATED __device__ __forceinline__ void
    FlagHeads(FlagT           (&head_flags)[ITEMS_PER_THREAD],    ///< [out] Calling thread's discontinuity head_flags
              T               (&input)[ITEMS_PER_THREAD],         ///< [in] Calling thread's input items
              FlagOp          flag_op)                            ///< [in] Binary boolean flag predicate
    {
        T preds[ITEMS_PER_THREAD];
        FlagHeads(head_flags, input, preds, flag_op);
    }

    /**
     * \deprecated [Since 1.14.0] The cub::BlockAdjacentDifference::FlagHeads
     * APIs are deprecated. Use cub::BlockAdjacentDifference::SubtractLeft instead.
     */
    template <int ITEMS_PER_THREAD,
              typename FlagT,
              typename FlagOp>
    CUB_DEPRECATED __device__ __forceinline__ void
    FlagHeads(FlagT           (&head_flags)[ITEMS_PER_THREAD],    ///< [out] Calling thread's discontinuity head_flags
              T               (&input)[ITEMS_PER_THREAD],         ///< [in] Calling thread's input items
              FlagOp          flag_op,                            ///< [in] Binary boolean flag predicate
              T               tile_predecessor_item)              ///< [in] <b>[<em>thread</em><sub>0</sub> only]</b> Item with which to compare the first tile item (<tt>input<sub>0</sub></tt> from <em>thread</em><sub>0</sub>).
    {
        T preds[ITEMS_PER_THREAD];
        FlagHeads(head_flags, input, preds, flag_op, tile_predecessor_item);
    }


    /**
     * \deprecated [Since 1.14.0] The cub::BlockAdjacentDifference::FlagTails
     * APIs are deprecated. Use cub::BlockAdjacentDifference::SubtractRight instead.
     */
    template <
      int             ITEMS_PER_THREAD,
      typename        FlagT,
      typename        FlagOp>
    CUB_DEPRECATED __device__ __forceinline__ void FlagTails(
        FlagT           (&tail_flags)[ITEMS_PER_THREAD],    ///< [out] Calling thread's discontinuity tail_flags
        T               (&input)[ITEMS_PER_THREAD],         ///< [in] Calling thread's input items
        FlagOp          flag_op)                            ///< [in] Binary boolean flag predicate
    {
        // Share first item
        temp_storage.first_items[linear_tid] = input[0];

        CTA_SYNC();

        // Set flag for last thread-item
        tail_flags[ITEMS_PER_THREAD - 1] = (linear_tid == BLOCK_THREADS - 1) ?
            1 :                             // Last thread
            ApplyOp<FlagOp>::FlagT(
                flag_op,
                input[ITEMS_PER_THREAD - 1],
                temp_storage.first_items[linear_tid + 1],
                (linear_tid * ITEMS_PER_THREAD) + ITEMS_PER_THREAD);

        // Set tail_flags for remaining items
        Iterate<0, ITEMS_PER_THREAD - 1>::FlagTails(linear_tid, tail_flags, input, flag_op);
    }


    /**
     * \deprecated [Since 1.14.0] The cub::BlockAdjacentDifference::FlagTails
     * APIs are deprecated. Use cub::BlockAdjacentDifference::SubtractRight instead.
     */
    template <
      int             ITEMS_PER_THREAD,
      typename        FlagT,
      typename        FlagOp>
    CUB_DEPRECATED __device__ __forceinline__ void FlagTails(
        FlagT           (&tail_flags)[ITEMS_PER_THREAD],    ///< [out] Calling thread's discontinuity tail_flags
        T               (&input)[ITEMS_PER_THREAD],         ///< [in] Calling thread's input items
        FlagOp          flag_op,                            ///< [in] Binary boolean flag predicate
        T               tile_successor_item)                ///< [in] <b>[<em>thread</em><sub><tt>BLOCK_THREADS</tt>-1</sub> only]</b> Item with which to compare the last tile item (<tt>input</tt><sub><em>ITEMS_PER_THREAD</em>-1</sub> from <em>thread</em><sub><em>BLOCK_THREADS</em>-1</sub>).
    {
        // Share first item
        temp_storage.first_items[linear_tid] = input[0];

        CTA_SYNC();

        // Set flag for last thread-item
        T successor_item = (linear_tid == BLOCK_THREADS - 1) ?
            tile_successor_item :              // Last thread
            temp_storage.first_items[linear_tid + 1];

        tail_flags[ITEMS_PER_THREAD - 1] = ApplyOp<FlagOp>::FlagT(
            flag_op,
            input[ITEMS_PER_THREAD - 1],
            successor_item,
            (linear_tid * ITEMS_PER_THREAD) + ITEMS_PER_THREAD);

        // Set tail_flags for remaining items
        Iterate<0, ITEMS_PER_THREAD - 1>::FlagTails(linear_tid, tail_flags, input, flag_op);
    }


    /**
     * \deprecated [Since 1.14.0] The cub::BlockAdjacentDifference::FlagHeadsAndTails
     * APIs are deprecated. Use cub::BlockAdjacentDifference::SubtractLeft or
     * cub::BlockAdjacentDifference::SubtractRight instead.
     */
    template <
      int             ITEMS_PER_THREAD,
      typename        FlagT,
      typename        FlagOp>
    CUB_DEPRECATED __device__ __forceinline__ void FlagHeadsAndTails(
        FlagT           (&head_flags)[ITEMS_PER_THREAD],    ///< [out] Calling thread's discontinuity head_flags
        FlagT           (&tail_flags)[ITEMS_PER_THREAD],    ///< [out] Calling thread's discontinuity tail_flags
        T               (&input)[ITEMS_PER_THREAD],         ///< [in] Calling thread's input items
        FlagOp          flag_op)                            ///< [in] Binary boolean flag predicate
    {
        // Share first and last items
        temp_storage.first_items[linear_tid] = input[0];
        temp_storage.last_items[linear_tid] = input[ITEMS_PER_THREAD - 1];

        CTA_SYNC();

        T preds[ITEMS_PER_THREAD];

        // Set flag for first thread-item
        preds[0] = temp_storage.last_items[linear_tid - 1];
        if (linear_tid == 0)
        {
            head_flags[0] = 1;
        }
        else
        {
            head_flags[0] = ApplyOp<FlagOp>::FlagT(
                flag_op,
                preds[0],
                input[0],
                linear_tid * ITEMS_PER_THREAD);
        }


        // Set flag for last thread-item
        tail_flags[ITEMS_PER_THREAD - 1] = (linear_tid == BLOCK_THREADS - 1) ?
            1 :                             // Last thread
            ApplyOp<FlagOp>::FlagT(
                flag_op,
                input[ITEMS_PER_THREAD - 1],
                temp_storage.first_items[linear_tid + 1],
                (linear_tid * ITEMS_PER_THREAD) + ITEMS_PER_THREAD);

        // Set head_flags for remaining items
        Iterate<1, ITEMS_PER_THREAD>::FlagHeads(linear_tid, head_flags, input, preds, flag_op);

        // Set tail_flags for remaining items
        Iterate<0, ITEMS_PER_THREAD - 1>::FlagTails(linear_tid, tail_flags, input, flag_op);
    }


    /**
     * \deprecated [Since 1.14.0] The cub::BlockAdjacentDifference::FlagHeadsAndTails
     * APIs are deprecated. Use cub::BlockAdjacentDifference::SubtractLeft or
     * cub::BlockAdjacentDifference::SubtractRight instead.
     */
    template <
      int             ITEMS_PER_THREAD,
      typename        FlagT,
      typename        FlagOp>
    CUB_DEPRECATED __device__ __forceinline__ void FlagHeadsAndTails(
        FlagT           (&head_flags)[ITEMS_PER_THREAD],    ///< [out] Calling thread's discontinuity head_flags
        FlagT           (&tail_flags)[ITEMS_PER_THREAD],    ///< [out] Calling thread's discontinuity tail_flags
        T               tile_successor_item,                ///< [in] <b>[<em>thread</em><sub><tt>BLOCK_THREADS</tt>-1</sub> only]</b> Item with which to compare the last tile item (<tt>input</tt><sub><em>ITEMS_PER_THREAD</em>-1</sub> from <em>thread</em><sub><em>BLOCK_THREADS</em>-1</sub>).
        T               (&input)[ITEMS_PER_THREAD],         ///< [in] Calling thread's input items
        FlagOp          flag_op)                            ///< [in] Binary boolean flag predicate
    {
        // Share first and last items
        temp_storage.first_items[linear_tid] = input[0];
        temp_storage.last_items[linear_tid] = input[ITEMS_PER_THREAD - 1];

        CTA_SYNC();

        T preds[ITEMS_PER_THREAD];

        // Set flag for first thread-item
        if (linear_tid == 0)
        {
            head_flags[0] = 1;
        }
        else
        {
            preds[0] = temp_storage.last_items[linear_tid - 1];
            head_flags[0] = ApplyOp<FlagOp>::FlagT(
                flag_op,
                preds[0],
                input[0],
                linear_tid * ITEMS_PER_THREAD);
        }

        // Set flag for last thread-item
        T successor_item = (linear_tid == BLOCK_THREADS - 1) ?
            tile_successor_item :              // Last thread
            temp_storage.first_items[linear_tid + 1];

        tail_flags[ITEMS_PER_THREAD - 1] = ApplyOp<FlagOp>::FlagT(
            flag_op,
            input[ITEMS_PER_THREAD - 1],
            successor_item,
            (linear_tid * ITEMS_PER_THREAD) + ITEMS_PER_THREAD);

        // Set head_flags for remaining items
        Iterate<1, ITEMS_PER_THREAD>::FlagHeads(linear_tid, head_flags, input, preds, flag_op);

        // Set tail_flags for remaining items
        Iterate<0, ITEMS_PER_THREAD - 1>::FlagTails(linear_tid, tail_flags, input, flag_op);
    }

    /**
     * \deprecated [Since 1.14.0] The cub::BlockAdjacentDifference::FlagHeadsAndTails
     * APIs are deprecated. Use cub::BlockAdjacentDifference::SubtractLeft or
     * cub::BlockAdjacentDifference::SubtractRight instead.
     */
    template <
      int             ITEMS_PER_THREAD,
      typename        FlagT,
      typename        FlagOp>
    CUB_DEPRECATED __device__ __forceinline__ void FlagHeadsAndTails(
        FlagT           (&head_flags)[ITEMS_PER_THREAD],    ///< [out] Calling thread's discontinuity head_flags
        T               tile_predecessor_item,              ///< [in] <b>[<em>thread</em><sub>0</sub> only]</b> Item with which to compare the first tile item (<tt>input<sub>0</sub></tt> from <em>thread</em><sub>0</sub>).
        FlagT           (&tail_flags)[ITEMS_PER_THREAD],    ///< [out] Calling thread's discontinuity tail_flags
        T               (&input)[ITEMS_PER_THREAD],         ///< [in] Calling thread's input items
        FlagOp          flag_op)                            ///< [in] Binary boolean flag predicate
    {
        // Share first and last items
        temp_storage.first_items[linear_tid] = input[0];
        temp_storage.last_items[linear_tid] = input[ITEMS_PER_THREAD - 1];

        CTA_SYNC();

        T preds[ITEMS_PER_THREAD];

        // Set flag for first thread-item
        preds[0] = (linear_tid == 0) ?
            tile_predecessor_item :              // First thread
            temp_storage.last_items[linear_tid - 1];

        head_flags[0] = ApplyOp<FlagOp>::FlagT(
            flag_op,
            preds[0],
            input[0],
            linear_tid * ITEMS_PER_THREAD);

        // Set flag for last thread-item
        tail_flags[ITEMS_PER_THREAD - 1] = (linear_tid == BLOCK_THREADS - 1) ?
            1 :                             // Last thread
            ApplyOp<FlagOp>::FlagT(
                flag_op,
                input[ITEMS_PER_THREAD - 1],
                temp_storage.first_items[linear_tid + 1],
                (linear_tid * ITEMS_PER_THREAD) + ITEMS_PER_THREAD);

        // Set head_flags for remaining items
        Iterate<1, ITEMS_PER_THREAD>::FlagHeads(linear_tid, head_flags, input, preds, flag_op);

        // Set tail_flags for remaining items
        Iterate<0, ITEMS_PER_THREAD - 1>::FlagTails(linear_tid, tail_flags, input, flag_op);
    }


    /**
     * \deprecated [Since 1.14.0] The cub::BlockAdjacentDifference::FlagHeadsAndTails
     * APIs are deprecated. Use cub::BlockAdjacentDifference::SubtractLeft or
     * cub::BlockAdjacentDifference::SubtractRight instead.
     */
    template <
        int             ITEMS_PER_THREAD,
        typename        FlagT,
        typename        FlagOp>
    CUB_DEPRECATED __device__ __forceinline__ void FlagHeadsAndTails(
        FlagT           (&head_flags)[ITEMS_PER_THREAD],    ///< [out] Calling thread's discontinuity head_flags
        T               tile_predecessor_item,              ///< [in] <b>[<em>thread</em><sub>0</sub> only]</b> Item with which to compare the first tile item (<tt>input<sub>0</sub></tt> from <em>thread</em><sub>0</sub>).
        FlagT           (&tail_flags)[ITEMS_PER_THREAD],    ///< [out] Calling thread's discontinuity tail_flags
        T               tile_successor_item,                ///< [in] <b>[<em>thread</em><sub><tt>BLOCK_THREADS</tt>-1</sub> only]</b> Item with which to compare the last tile item (<tt>input</tt><sub><em>ITEMS_PER_THREAD</em>-1</sub> from <em>thread</em><sub><em>BLOCK_THREADS</em>-1</sub>).
        T               (&input)[ITEMS_PER_THREAD],         ///< [in] Calling thread's input items
        FlagOp          flag_op)                            ///< [in] Binary boolean flag predicate
    {
        // Share first and last items
        temp_storage.first_items[linear_tid] = input[0];
        temp_storage.last_items[linear_tid] = input[ITEMS_PER_THREAD - 1];

        CTA_SYNC();

        T preds[ITEMS_PER_THREAD];

        // Set flag for first thread-item
        preds[0] = (linear_tid == 0) ?
            tile_predecessor_item :              // First thread
            temp_storage.last_items[linear_tid - 1];

        head_flags[0] = ApplyOp<FlagOp>::FlagT(
            flag_op,
            preds[0],
            input[0],
            linear_tid * ITEMS_PER_THREAD);

        // Set flag for last thread-item
        T successor_item = (linear_tid == BLOCK_THREADS - 1) ?
            tile_successor_item :              // Last thread
            temp_storage.first_items[linear_tid + 1];

        tail_flags[ITEMS_PER_THREAD - 1] = ApplyOp<FlagOp>::FlagT(
            flag_op,
            input[ITEMS_PER_THREAD - 1],
            successor_item,
            (linear_tid * ITEMS_PER_THREAD) + ITEMS_PER_THREAD);

        // Set head_flags for remaining items
        Iterate<1, ITEMS_PER_THREAD>::FlagHeads(linear_tid, head_flags, input, preds, flag_op);

        // Set tail_flags for remaining items
        Iterate<0, ITEMS_PER_THREAD - 1>::FlagTails(linear_tid, tail_flags, input, flag_op);
    }



};


}               // CUB namespace
CUB_NS_POSTFIX  // Optional outer namespace(s)
