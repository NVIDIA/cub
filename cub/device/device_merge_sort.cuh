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

#include "../config.cuh"
#include "../util_namespace.cuh"
#include "dispatch/dispatch_merge_sort.cuh"

CUB_NAMESPACE_BEGIN


/**
 * \brief DeviceMergeSort provides device-wide, parallel operations for computing a merge sort across a sequence of data items residing within device-accessible memory.
 * \ingroup SingleModule
 *
 * \par Overview
 * - DeviceMergeSort arranges items into ascending order using a comparison
 *   functor with less-than semantics. Merge sort can handle arbitrary types (as
 *   long as a value of these types is a model of
 *   <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a>)
 *   and comparison functors, but is slower than DeviceRadixSort when sorting
 *   arithmetic types into ascending/descending order.
 * - Another difference from RadixSort is the fact that DeviceMergeSort can
 *   handle arbitrary random-access iterators, as shown below.
 *
 * \par A Simple Example
 * \par
 * The code snippet below illustrates a thrust reverse iterator usage.
 * \par
 * \code
 * #include <cub/cub.cuh>  // or equivalently <cub/device/device_merge_sort.cuh>
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
 * // Declare, allocate, and initialize device-accessible pointers for sorting data
 * thrust::device_vector<KeyType> d_keys(num_items);
 * thrust::device_vector<DataType> d_values(num_items);
 * // ...
 *
 * // Initialize iterator
 * using KeyIterator = typename thrust::device_vector<KeyType>::iterator;
 * thrust::reverse_iterator<KeyIterator> reverse_iter(d_keys.end());
 *
 * // Determine temporary device storage requirements
 * std::size_t temp_storage_bytes = 0;
 * cub::DeviceMergeSort::SortPairs(
 *   nullptr,
 *   temp_storage_bytes,
 *   reverse_iter,
 *   thrust::raw_pointer_cast(d_values.data()),
 *   num_items,
 *   CustomLess());
 *
 * // Allocate temporary storage
 * cudaMalloc(&d_temp_storage, temp_storage_bytes);
 *
 * // Run sorting operation
 * cub::DeviceMergeSort::SortPairs(
 *   d_temp_storage,
 *   temp_storage_bytes,
 *   reverse_iter,
 *   thrust::raw_pointer_cast(d_values.data()),
 *   num_items,
 *   CustomLess());
 * \endcode
 */
struct DeviceMergeSort
{

  /**
   * \brief Sorts items using a merge sorting method.
   *
   * \par
   * SortPairs is not guaranteed to be stable. That is, suppose that i and j are
   * equivalent: neither one is less than the other. It is not guaranteed
   * that the relative order of these two elements will be preserved by sort.
   *
   * \par Snippet
   * The code snippet below illustrates the sorting of a device vector of \p int keys
   * with associated vector of \p int values.
   * \par
   * \code
   * #include <cub/cub.cuh>   // or equivalently <cub/device/device_merge_sort.cuh>
   *
   * // Declare, allocate, and initialize device-accessible pointers for sorting data
   * int  num_items;       // e.g., 7
   * int  *d_keys;         // e.g., [8, 6, 6, 5, 3, 0, 9]
   * int  *d_values;       // e.g., [0, 1, 2, 3, 4, 5, 6]
   * ...
   *
   * // Initialize comparator
   * CustomOpT custom_op;
   *
   * // Determine temporary device storage requirements
   * void     *d_temp_storage = NULL;
   * size_t   temp_storage_bytes = 0;
   * cub::DeviceMergeSort::SortPairs(d_temp_storage, temp_storage_bytes, d_keys, d_values, num_items, custom_op);
   *
   * // Allocate temporary storage
   * cudaMalloc(&d_temp_storage, temp_storage_bytes);
   *
   * // Run sorting operation
   * cub::DeviceMergeSort::SortPairs(d_temp_storage, temp_storage_bytes, d_keys, d_values, num_items, custom_op);
   *
   * // d_keys      <-- [0, 3, 5, 6, 6, 8, 9]
   * // d_values    <-- [5, 4, 3, 2, 1, 0, 6]
   *
   * \endcode
   *
   * \tparam KeyIteratorT is a model of <a href="https://en.cppreference.com/w/cpp/iterator/random_access_iterator">Random Access Iterator</a>,
   *         \p KeyIteratorT is mutable, and \p KeyIteratorT's \c value_type is
   *         a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a>,
   *         and the ordering relation on \p KeyIteratorT's \c value_type is a <em>strict weak ordering</em>, as defined in the
   *         <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a> requirements.
   * \tparam ValueIteratorT is a model of <a href="https://en.cppreference.com/w/cpp/iterator/random_access_iterator">Random Access Iterator</a>,
   *         \p ValueIteratorT is mutable, and \p ValueIteratorT's \c value_type is
   *         a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a>,
   *         and the ordering relation on \p ValueIteratorT's \c value_type is a <em>strict weak ordering</em>, as defined in the
   *         <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a> requirements.
   * \tparam OffsetT is an integer type for global offsets.
   * \tparam CompareOpT functor type having member <tt>bool operator()(KeyT lhs, KeyT rhs)</tt>
   *         CompareOpT is a model of <a href="https://en.cppreference.com/w/cpp/concepts/strict_weak_order">Strict Weak Ordering</a>.
   */
  template <typename KeyIteratorT,
            typename ValueIteratorT,
            typename OffsetT,
            typename CompareOpT>
  CUB_RUNTIME_FUNCTION static cudaError_t SortPairs(void *d_temp_storage,            ///< [in] Device-accessible allocation of temporary storage. When NULL, the required allocation size is written to \p temp_storage_bytes and no work is done.
                                                    std::size_t &temp_storage_bytes, ///< [in,out] Reference to size in bytes of \p d_temp_storage allocation
                                                    KeyIteratorT d_keys,             ///< [in,out] Pointer to the input sequence of unsorted input keys
                                                    ValueIteratorT d_items,          ///< [in,out] Pointer to the input sequence of unsorted input values
                                                    OffsetT num_items,               ///< [in] Number of items to sort
                                                    CompareOpT compare_op,           ///< [in] Comparison function object which returns true if the first argument is ordered before the second
                                                    cudaStream_t stream = 0,         ///< [in] <b>[optional]</b> CUDA stream to launch kernels within. Default is stream<sub>0</sub>.
                                                    bool debug_synchronous = false)  ///< [in] <b>[optional]</b> Whether or not to synchronize the stream after every kernel launch to check for errors.  Also causes launch configurations to be printed to the console.  Default is \p false.
  {
    using DispatchMergeSortT = DispatchMergeSort<KeyIteratorT,
                                                 ValueIteratorT,
                                                 KeyIteratorT,
                                                 ValueIteratorT,
                                                 OffsetT,
                                                 CompareOpT>;

    return DispatchMergeSortT::Dispatch(d_temp_storage,
                                        temp_storage_bytes,
                                        d_keys,
                                        d_items,
                                        d_keys,
                                        d_items,
                                        num_items,
                                        compare_op,
                                        stream,
                                        debug_synchronous);
  }

  /**
   * \brief Sorts items using a merge sorting method.
   *
   * \par
   * - SortPairs is not guaranteed to be stable. That is, suppose that i and j are
   *   equivalent: neither one is less than the other. It is not guaranteed
   *   that the relative order of these two elements will be preserved by sort.
   * - Input arrays d_input_keys and d_input_items are not modified.
   *
   * \par Snippet
   * The code snippet below illustrates the sorting of a device vector of \p int keys
   * with associated vector of \p int values.
   * \par
   * \code
   * #include <cub/cub.cuh>   // or equivalently <cub/device/device_merge_sort.cuh>
   *
   * // Declare, allocate, and initialize device-accessible pointers for sorting data
   * int  num_items;       // e.g., 7
   * int  *d_keys;         // e.g., [8, 6, 6, 5, 3, 0, 9]
   * int  *d_values;       // e.g., [0, 1, 2, 3, 4, 5, 6]
   * ...
   *
   * // Initialize comparator
   * CustomOpT custom_op;
   *
   * // Determine temporary device storage requirements
   * void     *d_temp_storage = NULL;
   * size_t   temp_storage_bytes = 0;
   * cub::DeviceMergeSort::SortPairsCopy(d_temp_storage, temp_storage_bytes, d_keys, d_values, num_items, custom_op);
   *
   * // Allocate temporary storage
   * cudaMalloc(&d_temp_storage, temp_storage_bytes);
   *
   * // Run sorting operation
   * cub::DeviceMergeSort::SortPairsCopy(d_temp_storage, temp_storage_bytes, d_keys, d_values, num_items, custom_op);
   *
   * // d_keys      <-- [0, 3, 5, 6, 6, 8, 9]
   * // d_values    <-- [5, 4, 3, 2, 1, 0, 6]
   *
   * \endcode
   *
   * \tparam KeyInputIteratorT is a model of <a href="https://en.cppreference.com/w/cpp/iterator/random_access_iterator">Random Access Iterator</a>,
   *         \p KeyInputIteratorT is mutable, and \p KeyInputIteratorT's \c value_type is
   *         a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a>,
   *         and the ordering relation on \p KeyInputIteratorT's \c value_type is a <em>strict weak ordering</em>, as defined in the
   *         <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a> requirements.
   * \tparam ValueInputIteratorT is a model of <a href="https://en.cppreference.com/w/cpp/iterator/random_access_iterator">Random Access Iterator</a>,
   *         \p ValueInputIteratorT is mutable, and \p ValueInputIteratorT's \c value_type is
   *         a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a>,
   *         and the ordering relation on \p ValueInputIteratorT's \c value_type is a <em>strict weak ordering</em>, as defined in the
   *         <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a> requirements.
   * \tparam KeyIteratorT is a model of <a href="https://en.cppreference.com/w/cpp/iterator/random_access_iterator">Random Access Iterator</a>,
   *         \p KeyIteratorT is mutable, and \p KeyIteratorT's \c value_type is
   *         a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a>,
   *         and the ordering relation on \p KeyIteratorT's \c value_type is a <em>strict weak ordering</em>, as defined in the
   *         <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a> requirements.
   * \tparam ValueIteratorT is a model of <a href="https://en.cppreference.com/w/cpp/iterator/random_access_iterator">Random Access Iterator</a>,
   *         \p ValueIteratorT is mutable, and \p ValueIteratorT's \c value_type is
   *         a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a>,
   *         and the ordering relation on \p ValueIteratorT's \c value_type is a <em>strict weak ordering</em>, as defined in the
   *         <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a> requirements.
   * \tparam OffsetT is an integer type for global offsets.
   * \tparam CompareOpT functor type having member <tt>bool operator()(KeyT lhs, KeyT rhs)</tt>
   *         CompareOpT is a model of <a href="https://en.cppreference.com/w/cpp/concepts/strict_weak_order">Strict Weak Ordering</a>.
   */
  template <typename KeyInputIteratorT,
            typename ValueInputIteratorT,
            typename KeyIteratorT,
            typename ValueIteratorT,
            typename OffsetT,
            typename CompareOpT>
  CUB_RUNTIME_FUNCTION static cudaError_t SortPairsCopy(void *d_temp_storage,              ///< [in] Device-accessible allocation of temporary storage. When NULL, the required allocation size is written to \p temp_storage_bytes and no work is done.
                                                        std::size_t &temp_storage_bytes,   ///< [in,out] Reference to size in bytes of \p d_temp_storage allocation
                                                        KeyInputIteratorT d_input_keys,    ///< [in] Pointer to the input sequence of unsorted input keys
                                                        ValueInputIteratorT d_input_items, ///< [in] Pointer to the input sequence of unsorted input values
                                                        KeyIteratorT d_output_keys,        ///< [out] Pointer to the output sequence of sorted input keys
                                                        ValueIteratorT d_output_items,     ///< [out] Pointer to the output sequence of sorted input values
                                                        OffsetT num_items,                 ///< [in] Number of items to sort
                                                        CompareOpT compare_op,             ///< [in] Comparison function object which returns true if the first argument is ordered before the second
                                                        cudaStream_t stream = 0,           ///< [in] <b>[optional]</b> CUDA stream to launch kernels within. Default is stream<sub>0</sub>.
                                                        bool debug_synchronous = false)    ///< [in] <b>[optional]</b> Whether or not to synchronize the stream after every kernel launch to check for errors.  Also causes launch configurations to be printed to the console.  Default is \p false.
  {
    using DispatchMergeSortT = DispatchMergeSort<KeyInputIteratorT,
                                                 ValueInputIteratorT,
                                                 KeyIteratorT,
                                                 ValueIteratorT,
                                                 OffsetT,
                                                 CompareOpT>;

    return DispatchMergeSortT::Dispatch(d_temp_storage,
                                        temp_storage_bytes,
                                        d_input_keys,
                                        d_input_items,
                                        d_output_keys,
                                        d_output_items,
                                        num_items,
                                        compare_op,
                                        stream,
                                        debug_synchronous);
  }

  /**
   * \brief Sorts items using a merge sorting method.
   *
   * \par
   * - SortKeys is not guaranteed to be stable. That is, suppose that i and j are
   *   equivalent: neither one is less than the other. It is not guaranteed
   *   that the relative order of these two elements will be preserved by sort.
   *
   * \par Snippet
   * The code snippet below illustrates the sorting of a device vector of \p int keys.
   * \par
   * \code
   * #include <cub/cub.cuh>   // or equivalently <cub/device/device_merge_sort.cuh>
   *
   * // Declare, allocate, and initialize device-accessible pointers for sorting data
   * int  num_items;       // e.g., 7
   * int  *d_keys;         // e.g., [8, 6, 7, 5, 3, 0, 9]
   * ...
   *
   * // Initialize comparator
   * CustomOpT custom_op;
   *
   * // Determine temporary device storage requirements
   * void     *d_temp_storage = NULL;
   * size_t   temp_storage_bytes = 0;
   * cub::DeviceMergeSort::SortKeys(d_temp_storage, temp_storage_bytes, d_keys, num_items, custom_op);
   *
   * // Allocate temporary storage
   * cudaMalloc(&d_temp_storage, temp_storage_bytes);
   *
   * // Run sorting operation
   * cub::DeviceMergeSort::SortKeys(d_temp_storage, temp_storage_bytes, d_keys, num_items, custom_op);
   *
   * // d_keys      <-- [0, 3, 5, 6, 7, 8, 9]
   *
   * \endcode
   *
   * \tparam KeyIteratorT is a model of <a href="https://en.cppreference.com/w/cpp/iterator/random_access_iterator">Random Access Iterator</a>,
   *         \p KeyIteratorT is mutable, and \p KeyIteratorT's \c value_type is
   *         a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a>,
   *         and the ordering relation on \p KeyIteratorT's \c value_type is a <em>strict weak ordering</em>, as defined in the
   *         <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a> requirements.
   * \tparam OffsetT is an integer type for global offsets.
   * \tparam CompareOpT functor type having member <tt>bool operator()(KeyT lhs, KeyT rhs)</tt>
   *         CompareOpT is a model of <a href="https://en.cppreference.com/w/cpp/concepts/strict_weak_order">Strict Weak Ordering</a>.
   */
  template <typename KeyIteratorT,
            typename OffsetT,
            typename CompareOpT>
  CUB_RUNTIME_FUNCTION static cudaError_t SortKeys(void *d_temp_storage,            ///< [in] Device-accessible allocation of temporary storage. When NULL, the required allocation size is written to \p temp_storage_bytes and no work is done.
                                                   std::size_t &temp_storage_bytes, ///< [in,out] Reference to size in bytes of \p d_temp_storage allocation
                                                   KeyIteratorT d_keys,             ///< [in,out] Pointer to the input sequence of unsorted input keys
                                                   OffsetT num_items,               ///< [in] Number of items to sort
                                                   CompareOpT compare_op,           ///< [in] Comparison function object which returns true if the first argument is ordered before the second
                                                   cudaStream_t stream = 0,         ///< [in] <b>[optional]</b> CUDA stream to launch kernels within. Default is stream<sub>0</sub>.
                                                   bool debug_synchronous = false)  ///< [in] <b>[optional]</b> Whether or not to synchronize the stream after every kernel launch to check for errors.  Also causes launch configurations to be printed to the console.  Default is \p false.
  {
    using DispatchMergeSortT = DispatchMergeSort<KeyIteratorT,
                                                 NullType *,
                                                 KeyIteratorT,
                                                 NullType *,
                                                 OffsetT,
                                                 CompareOpT>;

    return DispatchMergeSortT::Dispatch(d_temp_storage,
                                        temp_storage_bytes,
                                        d_keys,
                                        static_cast<NullType *>(nullptr),
                                        d_keys,
                                        static_cast<NullType *>(nullptr),
                                        num_items,
                                        compare_op,
                                        stream,
                                        debug_synchronous);
  }

  /**
   * \brief Sorts items using a merge sorting method.
   *
   * \par
   * - SortKeys is not guaranteed to be stable. That is, suppose that i and j are
   *   equivalent: neither one is less than the other. It is not guaranteed
   *   that the relative order of these two elements will be preserved by sort.
   * - Input array d_input_keys is not modified.
   *
   * \par Snippet
   * The code snippet below illustrates the sorting of a device vector of \p int keys.
   * \par
   * \code
   * #include <cub/cub.cuh>   // or equivalently <cub/device/device_merge_sort.cuh>
   *
   * // Declare, allocate, and initialize device-accessible pointers for sorting data
   * int  num_items;       // e.g., 7
   * int  *d_keys;         // e.g., [8, 6, 7, 5, 3, 0, 9]
   * ...
   *
   * // Initialize comparator
   * CustomOpT custom_op;
   *
   * // Determine temporary device storage requirements
   * void     *d_temp_storage = NULL;
   * size_t   temp_storage_bytes = 0;
   * cub::DeviceMergeSort::SortKeysCopy(d_temp_storage, temp_storage_bytes, d_keys, num_items, custom_op);
   *
   * // Allocate temporary storage
   * cudaMalloc(&d_temp_storage, temp_storage_bytes);
   *
   * // Run sorting operation
   * cub::DeviceMergeSort::SortKeysCopy(d_temp_storage, temp_storage_bytes, d_keys, num_items, custom_op);
   *
   * // d_keys      <-- [0, 3, 5, 6, 7, 8, 9]
   *
   * \endcode
   *
   * \tparam KeyInputIteratorT is a model of <a href="https://en.cppreference.com/w/cpp/iterator/random_access_iterator">Random Access Iterator</a>,
   *         \p KeyInputIteratorT is mutable, and \p KeyInputIteratorT's \c value_type is
   *         a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a>,
   *         and the ordering relation on \p KeyInputIteratorT's \c value_type is a <em>strict weak ordering</em>, as defined in the
   *         <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a> requirements.
   * \tparam KeyIteratorT is a model of <a href="https://en.cppreference.com/w/cpp/iterator/random_access_iterator">Random Access Iterator</a>,
   *         \p KeyIteratorT is mutable, and \p KeyIteratorT's \c value_type is
   *         a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a>,
   *         and the ordering relation on \p KeyIteratorT's \c value_type is a <em>strict weak ordering</em>, as defined in the
   *         <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a> requirements.
   * \tparam OffsetT is an integer type for global offsets.
   * \tparam CompareOpT functor type having member <tt>bool operator()(KeyT lhs, KeyT rhs)</tt>
   *         CompareOpT is a model of <a href="https://en.cppreference.com/w/cpp/concepts/strict_weak_order">Strict Weak Ordering</a>.
   */
  template <typename KeyInputIteratorT,
            typename KeyIteratorT,
            typename OffsetT,
            typename CompareOpT>
  CUB_RUNTIME_FUNCTION static cudaError_t SortKeysCopy(void *d_temp_storage,            ///< [in] Device-accessible allocation of temporary storage. When NULL, the required allocation size is written to \p temp_storage_bytes and no work is done.
                                                       std::size_t &temp_storage_bytes, ///< [in,out] Reference to size in bytes of \p d_temp_storage allocation
                                                       KeyInputIteratorT d_input_keys,  ///< [in] Pointer to the input sequence of unsorted input keys
                                                       KeyIteratorT d_output_keys,      ///< [out] Pointer to the output sequence of sorted input keys
                                                       OffsetT num_items,               ///< [in] Number of items to sort
                                                       CompareOpT compare_op,           ///< [in] Comparison function object which returns true if the first argument is ordered before the second
                                                       cudaStream_t stream = 0,         ///< [in] <b>[optional]</b> CUDA stream to launch kernels within. Default is stream<sub>0</sub>.
                                                       bool debug_synchronous = false)  ///< [in] <b>[optional]</b> Whether or not to synchronize the stream after every kernel launch to check for errors.  Also causes launch configurations to be printed to the console.  Default is \p false.
  {
    using DispatchMergeSortT = DispatchMergeSort<KeyInputIteratorT,
                                                 NullType *,
                                                 KeyIteratorT,
                                                 NullType *,
                                                 OffsetT,
                                                 CompareOpT>;

    return DispatchMergeSortT::Dispatch(d_temp_storage,
                                        temp_storage_bytes,
                                        d_input_keys,
                                        static_cast<NullType *>(nullptr),
                                        d_output_keys,
                                        static_cast<NullType *>(nullptr),
                                        num_items,
                                        compare_op,
                                        stream,
                                        debug_synchronous);
  }

  /**
   * \brief Sorts items using a merge sorting method.
   *
   * \par
   * StableSortPairs is stable: it preserves the relative ordering of equivalent
   * elements. That is, if x and y are elements such that x precedes y,
   * and if the two elements are equivalent (neither x < y nor y < x) then
   * a postcondition of stable_sort is that x still precedes y.
   *
   * \par Snippet
   * The code snippet below illustrates the sorting of a device vector of \p int keys
   * with associated vector of \p int values.
   * \par
   * \code
   * #include <cub/cub.cuh>   // or equivalently <cub/device/device_merge_sort.cuh>
   *
   * // Declare, allocate, and initialize device-accessible pointers for sorting data
   * int  num_items;       // e.g., 7
   * int  *d_keys;         // e.g., [8, 6, 6, 5, 3, 0, 9]
   * int  *d_values;       // e.g., [0, 1, 2, 3, 4, 5, 6]
   * ...
   *
   * // Initialize comparator
   * CustomOpT custom_op;
   *
   * // Determine temporary device storage requirements
   * void     *d_temp_storage = NULL;
   * size_t   temp_storage_bytes = 0;
   * cub::DeviceMergeSort::StableSortPairs(d_temp_storage, temp_storage_bytes, d_keys, d_values, num_items, custom_op);
   *
   * // Allocate temporary storage
   * cudaMalloc(&d_temp_storage, temp_storage_bytes);
   *
   * // Run sorting operation
   * cub::DeviceMergeSort::StableSortPairs(d_temp_storage, temp_storage_bytes, d_keys, d_values, num_items, custom_op);
   *
   * // d_keys      <-- [0, 3, 5, 6, 6, 8, 9]
   * // d_values    <-- [5, 4, 3, 1, 2, 0, 6]
   *
   * \endcode
   *
   * \tparam KeyIteratorT is a model of <a href="https://en.cppreference.com/w/cpp/iterator/random_access_iterator">Random Access Iterator</a>,
   *         \p KeyIteratorT is mutable, and \p KeyIteratorT's \c value_type is
   *         a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a>,
   *         and the ordering relation on \p KeyIteratorT's \c value_type is a <em>strict weak ordering</em>, as defined in the
   *         <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a> requirements.
   * \tparam ValueIteratorT is a model of <a href="https://en.cppreference.com/w/cpp/iterator/random_access_iterator">Random Access Iterator</a>,
   *         \p ValueIteratorT is mutable, and \p ValueIteratorT's \c value_type is
   *         a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a>,
   *         and the ordering relation on \p ValueIteratorT's \c value_type is a <em>strict weak ordering</em>, as defined in the
   *         <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a> requirements.
   * \tparam OffsetT is an integer type for global offsets.
   * \tparam CompareOpT functor type having member <tt>bool operator()(KeyT lhs, KeyT rhs)</tt>
   *         CompareOpT is a model of <a href="https://en.cppreference.com/w/cpp/concepts/strict_weak_order">Strict Weak Ordering</a>.
   */
  template <typename KeyIteratorT,
            typename ValueIteratorT,
            typename OffsetT,
            typename CompareOpT>
  CUB_RUNTIME_FUNCTION static cudaError_t
  StableSortPairs(void *d_temp_storage,            ///< [in] Device-accessible allocation of temporary storage. When NULL, the required allocation size is written to \p temp_storage_bytes and no work is done.
                  std::size_t &temp_storage_bytes, ///< [in,out] Reference to size in bytes of \p d_temp_storage allocation
                  KeyIteratorT d_keys,             ///< [in,out] Pointer to the input sequence of unsorted input keys
                  ValueIteratorT d_items,          ///< [in,out] Pointer to the input sequence of unsorted input values
                  OffsetT num_items,               ///< [in] Number of items to sort
                  CompareOpT compare_op,           ///< [in] Comparison function object which returns true if the first argument is ordered before the second
                  cudaStream_t stream = 0,         ///< [in] <b>[optional]</b> CUDA stream to launch kernels within. Default is stream<sub>0</sub>.
                  bool debug_synchronous = false)  ///< [in] <b>[optional]</b> Whether or not to synchronize the stream after every kernel launch to check for errors.  Also causes launch configurations to be printed to the console.  Default is \p false.
  {
    return SortPairs<KeyIteratorT, ValueIteratorT, OffsetT, CompareOpT>(
      d_temp_storage,
      temp_storage_bytes,
      d_keys,
      d_items,
      num_items,
      compare_op,
      stream,
      debug_synchronous);
  }

  /**
   * \brief Sorts items using a merge sorting method.
   *
   * \par
   * StableSortKeys is stable: it preserves the relative ordering of equivalent
   * elements. That is, if x and y are elements such that x precedes y,
   * and if the two elements are equivalent (neither x < y nor y < x) then
   * a postcondition of stable_sort is that x still precedes y.
   *
   * \par Snippet
   * The code snippet below illustrates the sorting of a device vector of \p int keys.
   * \par
   * \code
   * #include <cub/cub.cuh>   // or equivalently <cub/device/device_merge_sort.cuh>
   *
   * // Declare, allocate, and initialize device-accessible pointers for sorting data
   * int  num_items;       // e.g., 7
   * int  *d_keys;         // e.g., [8, 6, 7, 5, 3, 0, 9]
   * ...
   *
   * // Initialize comparator
   * CustomOpT custom_op;
   *
   * // Determine temporary device storage requirements
   * void     *d_temp_storage = NULL;
   * size_t   temp_storage_bytes = 0;
   * cub::DeviceMergeSort::StableSortKeys(d_temp_storage, temp_storage_bytes, d_keys, num_items, custom_op);
   *
   * // Allocate temporary storage
   * cudaMalloc(&d_temp_storage, temp_storage_bytes);
   *
   * // Run sorting operation
   * cub::DeviceMergeSort::StableSortKeys(d_temp_storage, temp_storage_bytes, d_keys, num_items, custom_op);
   *
   * // d_keys      <-- [0, 3, 5, 6, 7, 8, 9]
   *
   * \endcode
   *
   * \tparam KeyIteratorT is a model of <a href="https://en.cppreference.com/w/cpp/iterator/random_access_iterator">Random Access Iterator</a>,
   *         \p KeyIteratorT is mutable, and \p KeyIteratorT's \c value_type is
   *         a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a>,
   *         and the ordering relation on \p KeyIteratorT's \c value_type is a <em>strict weak ordering</em>, as defined in the
   *         <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a> requirements.
   * \tparam OffsetT is an integer type for global offsets.
   * \tparam CompareOpT functor type having member <tt>bool operator()(KeyT lhs, KeyT rhs)</tt>
   *         CompareOpT is a model of <a href="https://en.cppreference.com/w/cpp/concepts/strict_weak_order">Strict Weak Ordering</a>.
   */
  template <typename KeyIteratorT,
            typename OffsetT,
            typename CompareOpT>
  CUB_RUNTIME_FUNCTION static cudaError_t
  StableSortKeys(void *d_temp_storage,            ///< [in] Device-accessible allocation of temporary storage. When NULL, the required allocation size is written to \p temp_storage_bytes and no work is done.
                 std::size_t &temp_storage_bytes, ///< [in,out] Reference to size in bytes of \p d_temp_storage allocation
                 KeyIteratorT d_keys,             ///< [in,out] Pointer to the input sequence of unsorted input keys
                 OffsetT num_items,               ///< [in] Number of items to sort
                 CompareOpT compare_op,           ///< [in] Comparison function object which returns true if the first argument is ordered before the second
                 cudaStream_t stream = 0,         ///< [in] <b>[optional]</b> CUDA stream to launch kernels within. Default is stream<sub>0</sub>.
                 bool debug_synchronous = false)  ///< [in] <b>[optional]</b> Whether or not to synchronize the stream after every kernel launch to check for errors.  Also causes launch configurations to be printed to the console.  Default is \p false.
  {
    return SortKeys<KeyIteratorT, OffsetT, CompareOpT>(d_temp_storage,
                                                       temp_storage_bytes,
                                                       d_keys,
                                                       num_items,
                                                       compare_op,
                                                       stream,
                                                       debug_synchronous);
  }
};

CUB_NAMESPACE_END