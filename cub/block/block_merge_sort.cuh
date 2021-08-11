/******************************************************************************
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

#pragma once

#include "../util_ptx.cuh"
#include "../util_type.cuh"
#include "../util_math.cuh"
#include "../util_namespace.cuh"

CUB_NAMESPACE_BEGIN


// Implementation of the MergePath algorithm, as described in:
// Odeh et al, "Merge Path - Parallel Merging Made Simple"
// doi:10.1109/IPDPSW.2012.202
template <typename KeyT,
          typename KeyIteratorT,
          typename OffsetT,
          typename BinaryPred>
__device__ __forceinline__ OffsetT MergePath(KeyIteratorT keys1,
                                             KeyIteratorT keys2,
                                             OffsetT keys1_count,
                                             OffsetT keys2_count,
                                             OffsetT diag,
                                             BinaryPred binary_pred)
{
  OffsetT keys1_begin = diag < keys2_count ? 0 : diag - keys2_count;
  OffsetT keys1_end   = (cub::min)(diag, keys1_count);

  while (keys1_begin < keys1_end)
  {
    OffsetT mid = cub::MidPoint<OffsetT>(keys1_begin, keys1_end);
    KeyT key1   = keys1[mid];
    KeyT key2   = keys2[diag - 1 - mid];
    bool pred   = binary_pred(key2, key1);

    if (pred)
    {
      keys1_end = mid;
    }
    else
    {
      keys1_begin = mid + 1;
    }
  }
  return keys1_begin;
}

template <typename KeyT, typename CompareOp, int ITEMS_PER_THREAD>
__device__ __forceinline__ void SerialMerge(KeyT *keys_shared,
                                            int keys1_beg,
                                            int keys2_beg,
                                            int keys1_count,
                                            int keys2_count,
                                            KeyT (&output)[ITEMS_PER_THREAD],
                                            int (&indices)[ITEMS_PER_THREAD],
                                            CompareOp compare_op)
{
  int keys1_end = keys1_beg + keys1_count;
  int keys2_end = keys2_beg + keys2_count;

  KeyT key1 = keys_shared[keys1_beg];
  KeyT key2 = keys_shared[keys2_beg];

#pragma unroll
  for (int item = 0; item < ITEMS_PER_THREAD; ++item)
  {
    bool p = (keys2_beg < keys2_end) &&
             ((keys1_beg >= keys1_end)
              || compare_op(key2, key1));

    output[item]  = p ? key2 : key1;
    indices[item] = p ? keys2_beg++ : keys1_beg++;

    if (p)
    {
      key2 = keys_shared[keys2_beg];
    }
    else
    {
      key1 = keys_shared[keys1_beg];
    }
  }
}

/**
 * \brief The BlockMergeSort class provides methods for sorting items partitioned across a CUDA thread block using a merge sorting method.
 * \ingroup BlockModule
 *
 * \tparam KeyT                 KeyT type
 * \tparam BLOCK_DIM_X          The thread block length in threads along the X dimension
 * \tparam ITEMS_PER_THREAD     The number of items per thread
 * \tparam ValueT               <b>[optional]</b> ValueT type (default: cub::NullType, which indicates a keys-only sort)
 * \tparam BLOCK_DIM_Y          <b>[optional]</b> The thread block length in threads along the Y dimension (default: 1)
 * \tparam BLOCK_DIM_Z          <b>[optional]</b> The thread block length in threads along the Z dimension (default: 1)
 *
 * \par Overview
 *   BlockMergeSort arranges items into ascending order using a comparison
 *   functor with less-than semantics. Merge sort can handle arbitrary types
 *   and comparison functors, but is slower than BlockRadixSort when sorting
 *   arithmetic types into ascending/descending order.
 *
 * \par A Simple Example
 * \blockcollective{BlockMergeSort}
 * \par
 * The code snippet below illustrates a sort of 512 integer keys that are
 * partitioned across 128 threads * where each thread owns 4 consecutive items.
 * \par
 * \code
 * #include <cub/cub.cuh>  // or equivalently <cub/block/block_merge_sort.cuh>
 *
 * struct CustomLess
 * {
 *   template <typename DataType>
 *   __device__ bool operator()(const DataType &lhs, const DataType &rhs)
 *   {
 *     return lhs < rhs;
 *   }
 * };
 *
 * __global__ void ExampleKernel(...)
 * {
 *     // Specialize BlockMergeSort for a 1D block of 128 threads owning 4 integer items each
 *     typedef cub::BlockMergeSort<int, 128, 4> BlockMergeSort;
 *
 *     // Allocate shared memory for BlockMergeSort
 *     __shared__ typename BlockMergeSort::TempStorage temp_storage_shuffle;
 *
 *     // Obtain a segment of consecutive items that are blocked across threads
 *     int thread_keys[4];
 *     ...
 *
 *     BlockMergeSort(temp_storage_shuffle).Sort(thread_data, CustomLess());
 *     ...
 * }
 * \endcode
 * \par
 * Suppose the set of input \p thread_keys across the block of threads is
 * <tt>{ [0,511,1,510], [2,509,3,508], [4,507,5,506], ..., [254,257,255,256] }</tt>.
 * The corresponding output \p thread_keys in those threads will be
 * <tt>{ [0,1,2,3], [4,5,6,7], [8,9,10,11], ..., [508,509,510,511] }</tt>.
 *
 * \par Re-using dynamically allocating shared memory
 * The following example under the examples/block folder illustrates usage of
 * dynamically shared memory with BlockReduce and how to re-purpose
 * the same memory region:
 * <a href="../../examples/block/example_block_reduce_dyn_smem.cu">example_block_reduce_dyn_smem.cu</a>
 *
 * This example can be easily adapted to the storage required by BlockMergeSort.
 */
template <
  typename  KeyT,
  int       BLOCK_DIM_X,
  int       ITEMS_PER_THREAD,
  typename  ValueT            = NullType,
  int       BLOCK_DIM_Y       = 1,
  int       BLOCK_DIM_Z       = 1>
class BlockMergeSort
{
private:

  // The thread block size in threads
  static constexpr int BLOCK_THREADS = BLOCK_DIM_X * BLOCK_DIM_Y * BLOCK_DIM_Z;
  static constexpr int ITEMS_PER_TILE = ITEMS_PER_THREAD * BLOCK_THREADS;

  // Whether or not there are values to be trucked along with keys
  static constexpr bool KEYS_ONLY = Equals<ValueT, NullType>::VALUE;

  /// Shared memory type required by this thread block
  union _TempStorage
  {
    KeyT keys_shared[ITEMS_PER_TILE + 1];
    ValueT items_shared[ITEMS_PER_TILE + 1];
  }; // union TempStorage

  /// Internal storage allocator
  __device__ __forceinline__ _TempStorage& PrivateStorage()
  {
    __shared__ _TempStorage private_storage;
    return private_storage;
  }

  /// Shared storage reference
  _TempStorage &temp_storage;

  /// Linear thread-id
  unsigned int linear_tid;

public:

  /// \smemstorage{BlockMergeSort}
  struct TempStorage : Uninitialized<_TempStorage> {};

  __device__ __forceinline__ BlockMergeSort()
      : temp_storage(PrivateStorage())
      , linear_tid(RowMajorTid(BLOCK_DIM_X, BLOCK_DIM_Y, BLOCK_DIM_Z))
  {}

  __device__ __forceinline__ BlockMergeSort(TempStorage &temp_storage)
      : temp_storage(temp_storage.Alias())
      , linear_tid(RowMajorTid(BLOCK_DIM_X, BLOCK_DIM_Y, BLOCK_DIM_Z))
  {}

private:

  template <typename T>
  __device__ __forceinline__ void Swap(T &lhs, T &rhs)
  {
    T temp = lhs;
    lhs    = rhs;
    rhs    = temp;
  }

  template <typename CompareOp>
  __device__ __forceinline__ void
  StableOddEvenSort(KeyT (&keys)[ITEMS_PER_THREAD],
                    ValueT (&items)[ITEMS_PER_THREAD],
                    CompareOp compare_op)
  {
#pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; ++i)
    {
#pragma unroll
      for (int j = 1 & i; j < ITEMS_PER_THREAD - 1; j += 2)
      {
        if (compare_op(keys[j + 1], keys[j]))
        {
          Swap(keys[j], keys[j + 1]);
          if (!KEYS_ONLY)
          {
            Swap(items[j], items[j + 1]);
          }
        }
      } // inner loop
    }   // outer loop
  }

public:

  /**
   * \brief Sorts items partitioned across a CUDA thread block using a merge sorting method.
   *
   * \par
   * - Sort is not guaranteed to be stable. That is, suppose that i and j are
   *   equivalent: neither one is less than the other. It is not guaranteed
   *   that the relative order of these two elements will be preserved by sort.
   *
   * \tparam CompareOp functor type having member <tt>bool operator()(KeyT lhs, KeyT rhs)</tt>
   *         CompareOp is a model of <a href="https://en.cppreference.com/w/cpp/concepts/strict_weak_order">Strict Weak Ordering</a>.
   */
  template <typename CompareOp>
  __device__ __forceinline__ void
  Sort(KeyT (&keys)[ITEMS_PER_THREAD], ///< [in-out] Keys to sort
       CompareOp compare_op)           ///< [in] Comparison function object which returns
                                       ///< true if the first argument is ordered before
                                       ///< the second
  {
    ValueT items[ITEMS_PER_THREAD];
    Sort<CompareOp, false>(keys, items, compare_op, ITEMS_PER_TILE, keys[0]);
  }

  /**
   * \brief Sorts items partitioned across a CUDA thread block using a merge sorting method.
   *
   * \par
   * - Sort is not guaranteed to be stable. That is, suppose that i and j are
   *   equivalent: neither one is less than the other. It is not guaranteed
   *   that the relative order of these two elements will be preserved by sort.
   * - The value of \p oob_default is assigned to all elements that are out of
   *   \p valid_items boundaries. It's expected that \p oob_default is ordered
   *   after any value in the \p valid_items boundaries. The algorithm always
   *   sorts a fixed amount of elements, which is equal to ITEMS_PER_THREAD * BLOCK_THREADS.
   *   If there is a value that is ordered after \p oob_default, it won't be
   *   placed within \p valid_items boundaries.
   *
   * \tparam CompareOp functor type having member <tt>bool operator()(KeyT lhs, KeyT rhs)</tt>
   *         CompareOp is a model of <a href="https://en.cppreference.com/w/cpp/concepts/strict_weak_order">Strict Weak Ordering</a>.
   */
  template <typename CompareOp>
  __device__ __forceinline__ void
  Sort(KeyT (&keys)[ITEMS_PER_THREAD], ///< [in-out] Keys to sort
       CompareOp compare_op,           ///< [in] Comparison function object which returns true if the first argument is ordered before the second
       int valid_items,                ///< [in] Number of valid items to sort
       KeyT oob_default)               ///< [in] Default value to assign out-of-bound items
  {
    ValueT items[ITEMS_PER_THREAD];
    Sort<CompareOp, true>(keys, items, compare_op, valid_items, oob_default);
  }

  /**
   * \brief Sorts items partitioned across a CUDA thread block using a merge sorting method.
   *
   * \par
   * - Sort is not guaranteed to be stable. That is, suppose that i and j are
   *   equivalent: neither one is less than the other. It is not guaranteed
   *   that the relative order of these two elements will be preserved by sort.
   *
   * \tparam CompareOp functor type having member <tt>bool operator()(KeyT lhs, KeyT rhs)</tt>
   *         CompareOp is a model of <a href="https://en.cppreference.com/w/cpp/concepts/strict_weak_order">Strict Weak Ordering</a>.
   */
  template <typename CompareOp>
  __device__ __forceinline__ void
  Sort(KeyT (&keys)[ITEMS_PER_THREAD],     ///< [in-out] Keys to sort
       ValueT (&items)[ITEMS_PER_THREAD],  ///< [in-out] Values to sort
       CompareOp compare_op)               ///< [in] Comparison function object which returns true if the first argument is ordered before the second
  {
    Sort<CompareOp, false>(keys, items, compare_op, ITEMS_PER_TILE, keys[0]);
  }

  /**
   * \brief Sorts items partitioned across a CUDA thread block using a merge sorting method.
   *
   * \par
   * - Sort is not guaranteed to be stable. That is, suppose that i and j are
   *   equivalent: neither one is less than the other. It is not guaranteed
   *   that the relative order of these two elements will be preserved by sort.
   * - The value of \p oob_default is assigned to all elements that are out of
   *   \p valid_items boundaries. It's expected that \p oob_default is ordered
   *   after any value in the \p valid_items boundaries. The algorithm always
   *   sorts a fixed amount of elements, which is equal to ITEMS_PER_THREAD * BLOCK_THREADS.
   *   If there is a value that is ordered after \p oob_default, it won't be
   *   placed within \p valid_items boundaries.
   *
   * \tparam CompareOp functor type having member <tt>bool operator()(KeyT lhs, KeyT rhs)</tt>
   *         CompareOp is a model of <a href="https://en.cppreference.com/w/cpp/concepts/strict_weak_order">Strict Weak Ordering</a>.
   * \tparam IS_LAST_TILE True if valid_items isn't equal to the ITEMS_PER_TILE
   */
  template <typename CompareOp,
            bool IS_LAST_TILE = true>
  __device__ __forceinline__ void
  Sort(KeyT (&keys)[ITEMS_PER_THREAD],     ///< [in-out] Keys to sort
       ValueT (&items)[ITEMS_PER_THREAD],  ///< [in-out] Values to sort
       CompareOp compare_op,               ///< [in] Comparison function object which returns true if the first argument is ordered before the second
       int valid_items,                    ///< [in] Number of valid items to sort
       KeyT oob_default)                   ///< [in] Default value to assign out-of-bound items
  {
    if (IS_LAST_TILE)
    {
      // if last tile, find valid max_key
      // and fill the remaining keys with it
      //
      KeyT max_key = oob_default;
#pragma unroll
      for (int item = 1; item < ITEMS_PER_THREAD; ++item)
      {
        if (ITEMS_PER_THREAD * linear_tid + item < valid_items)
        {
          max_key = compare_op(max_key, keys[item]) ? keys[item] : max_key;
        }
        else
        {
          keys[item] = max_key;
        }
      }
    }

    // if first element of thread is in input range, stable sort items
    //
    if (!IS_LAST_TILE || ITEMS_PER_THREAD * linear_tid < valid_items)
    {
      StableOddEvenSort(keys, items, compare_op);
    }

    // each thread has sorted keys
    // merge sort keys in shared memory
    //
#pragma unroll
    for (int target_merged_threads_number = 2;
         target_merged_threads_number <= BLOCK_THREADS;
         target_merged_threads_number *= 2)
    {
      int merged_threads_number = target_merged_threads_number / 2;
      int mask = target_merged_threads_number - 1;

      CTA_SYNC();

      // store keys in shmem
      //
#pragma unroll
      for (int item = 0; item < ITEMS_PER_THREAD; ++item)
      {
        int idx                       = ITEMS_PER_THREAD * linear_tid + item;
        temp_storage.keys_shared[idx] = keys[item];
      }

      CTA_SYNC();

      int indices[ITEMS_PER_THREAD];

      int first_thread_idx_in_thread_group_being_merged = ~mask & linear_tid;
      int start = ITEMS_PER_THREAD * first_thread_idx_in_thread_group_being_merged;
      int size  = ITEMS_PER_THREAD * merged_threads_number;

      int thread_idx_in_thread_group_being_merged = mask & linear_tid;

      int diag =
        (cub::min)(valid_items,
                   ITEMS_PER_THREAD * thread_idx_in_thread_group_being_merged);

      int keys1_beg = (cub::min)(valid_items, start);
      int keys1_end = (cub::min)(valid_items, keys1_beg + size);
      int keys2_beg = keys1_end;
      int keys2_end = (cub::min)(valid_items, keys2_beg + size);

      int keys1_count = keys1_end - keys1_beg;
      int keys2_count = keys2_end - keys2_beg;

      int partition_diag = MergePath<KeyT>(&temp_storage.keys_shared[keys1_beg],
                                           &temp_storage.keys_shared[keys2_beg],
                                           keys1_count,
                                           keys2_count,
                                           diag,
                                           compare_op);

      int keys1_beg_loc   = keys1_beg + partition_diag;
      int keys1_end_loc   = keys1_end;
      int keys2_beg_loc   = keys2_beg + diag - partition_diag;
      int keys2_end_loc   = keys2_end;
      int keys1_count_loc = keys1_end_loc - keys1_beg_loc;
      int keys2_count_loc = keys2_end_loc - keys2_beg_loc;
      SerialMerge(&temp_storage.keys_shared[0],
                  keys1_beg_loc,
                  keys2_beg_loc,
                  keys1_count_loc,
                  keys2_count_loc,
                  keys,
                  indices,
                  compare_op);

      if (!KEYS_ONLY)
      {
        CTA_SYNC();

        // store keys in shmem
        //
#pragma unroll
        for (int item = 0; item < ITEMS_PER_THREAD; ++item)
        {
          int idx = ITEMS_PER_THREAD * linear_tid + item;
          temp_storage.items_shared[idx] = items[item];
        }

        CTA_SYNC();

        // gather items from shmem
        //
#pragma unroll
        for (int item = 0; item < ITEMS_PER_THREAD; ++item)
        {
          items[item] = temp_storage.items_shared[indices[item]];
        }
      }
    }
  } // func block_merge_sort

  /**
   * \brief Sorts items partitioned across a CUDA thread block using a merge sorting method.
   *
   * \par
   * - StableSort is stable: it preserves the relative ordering of equivalent
   *   elements. That is, if x and y are elements such that x precedes y,
   *   and if the two elements are equivalent (neither x < y nor y < x) then
   *   a postcondition of stable_sort is that x still precedes y.
   *
   * \tparam CompareOp functor type having member <tt>bool operator()(KeyT lhs, KeyT rhs)</tt>
   *         CompareOp is a model of <a href="https://en.cppreference.com/w/cpp/concepts/strict_weak_order">Strict Weak Ordering</a>.
   */
  template <typename CompareOp>
  __device__ __forceinline__ void
  StableSort(KeyT (&keys)[ITEMS_PER_THREAD],   ///< [in-out] Keys to sort
             CompareOp compare_op)             ///< [in] Comparison function object which returns true if the first argument is ordered before the second
  {
    Sort(keys, compare_op);
  }

  /**
   * \brief Sorts items partitioned across a CUDA thread block using a merge sorting method.
   *
   * \par
   * - StableSort is stable: it preserves the relative ordering of equivalent
   *   elements. That is, if x and y are elements such that x precedes y,
   *   and if the two elements are equivalent (neither x < y nor y < x) then
   *   a postcondition of stable_sort is that x still precedes y.
   *
   * \tparam CompareOp functor type having member <tt>bool operator()(KeyT lhs, KeyT rhs)</tt>
   *         CompareOp is a model of <a href="https://en.cppreference.com/w/cpp/concepts/strict_weak_order">Strict Weak Ordering</a>.
   */
  template <typename CompareOp>
  __device__ __forceinline__ void
  StableSort(KeyT (&keys)[ITEMS_PER_THREAD],    ///< [in-out] Keys to sort
             ValueT (&items)[ITEMS_PER_THREAD], ///< [in-out] Values to sort
             CompareOp compare_op)              ///< [in] Comparison function object which returns true if the first argument is ordered before the second
  {
    Sort(keys, items, compare_op);
  }

  /**
   * \brief Sorts items partitioned across a CUDA thread block using a merge sorting method.
   *
   * \par
   * - StableSort is stable: it preserves the relative ordering of equivalent
   *   elements. That is, if x and y are elements such that x precedes y,
   *   and if the two elements are equivalent (neither x < y nor y < x) then
   *   a postcondition of stable_sort is that x still precedes y.
   * - The value of \p oob_default is assigned to all elements that are out of
   *   \p valid_items boundaries. It's expected that \p oob_default is ordered
   *   after any value in the \p valid_items boundaries. The algorithm always
   *   sorts a fixed amount of elements, which is equal to ITEMS_PER_THREAD * BLOCK_THREADS.
   *   If there is a value that is ordered after \p oob_default, it won't be
   *   placed within \p valid_items boundaries.
   *
   * \tparam CompareOp functor type having member <tt>bool operator()(KeyT lhs, KeyT rhs)</tt>
   *         CompareOp is a model of <a href="https://en.cppreference.com/w/cpp/concepts/strict_weak_order">Strict Weak Ordering</a>.
   */
  template <typename CompareOp>
  __device__ __forceinline__ void
  StableSort(KeyT (&keys)[ITEMS_PER_THREAD],  ///< [in-out] Keys to sort
             CompareOp compare_op,            ///< [in] Comparison function object which returns true if the first argument is ordered before the second
             int valid_items,                 ///< [in] Number of valid items to sort
             KeyT oob_default)                ///< [in] Default value to assign out-of-bound items
  {
    Sort(keys, compare_op, valid_items, oob_default);
  }

  /**
   * \brief Sorts items partitioned across a CUDA thread block using a merge sorting method.
   *
   * \par
   * - StableSort is stable: it preserves the relative ordering of equivalent
   *   elements. That is, if x and y are elements such that x precedes y,
   *   and if the two elements are equivalent (neither x < y nor y < x) then
   *   a postcondition of stable_sort is that x still precedes y.
   * - The value of \p oob_default is assigned to all elements that are out of
   *   \p valid_items boundaries. It's expected that \p oob_default is ordered
   *   after any value in the \p valid_items boundaries. The algorithm always
   *   sorts a fixed amount of elements, which is equal to ITEMS_PER_THREAD * BLOCK_THREADS.
   *   If there is a value that is ordered after \p oob_default, it won't be
   *   placed within \p valid_items boundaries.
   *
   * \tparam CompareOp functor type having member <tt>bool operator()(KeyT lhs, KeyT rhs)</tt>
   *         CompareOp is a model of <a href="https://en.cppreference.com/w/cpp/concepts/strict_weak_order">Strict Weak Ordering</a>.
   * \tparam IS_LAST_TILE True if valid_items isn't equal to the ITEMS_PER_TILE
   */
  template <typename CompareOp,
            bool IS_LAST_TILE = true>
  __device__ __forceinline__ void
  StableSort(KeyT (&keys)[ITEMS_PER_THREAD],     ///< [in-out] Keys to sort
             ValueT (&items)[ITEMS_PER_THREAD],  ///< [in-out] Values to sort
             CompareOp compare_op,               ///< [in] Comparison function object which returns true if the first argument is ordered before the second
             int valid_items,                    ///< [in] Number of valid items to sort
             KeyT oob_default)                   ///< [in] Default value to assign out-of-bound items
  {
    Sort<CompareOp, IS_LAST_TILE>(keys,
                                  items,
                                  compare_op,
                                  valid_items,
                                  oob_default);
  }
};

CUB_NAMESPACE_END
