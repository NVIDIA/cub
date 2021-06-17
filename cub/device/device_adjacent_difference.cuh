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
#include "dispatch/dispatch_adjacent_difference.cuh"

/// Optional outer namespace(s)
CUB_NS_PREFIX

/// CUB namespace
namespace cub {

/**
 * \brief DeviceAdjacentDifference provides device-wide, parallel operations for computing the differences of adjacent elements residing within device-accessible memory.
 * \ingroup SingleModule
 *
 * \par Overview
 * - DeviceAdjacentDifference calculates the differences of adjacent elements in
 *   the d_input. Because the binary operation could be noncommutative, there
 *   are two sets of methods. Methods named SubtractLeft subtract left element
 *   <tt>\*(i - 1)</tt> of input sequence from current element <tt>\*i</tt>.
 *   Methods named SubtractRight subtract current element <tt>\*i</tt> from the
 *   right one <tt>*(i + 1)</tt>:
 *   \par
 *   \code
 *   int *d_values; // [1, 2, 3, 4]
 *   //...
 *   int *d_subtract_left_result  <-- [  1,  1,  1,  1 ]
 *   int *d_subtract_right_result <-- [ -1, -1, -1,  4 ]
 *   \endcode
 * - For SubtractLeft, if the left element is out of bounds, the iterator is
 *   assigned to <tt>\*(result + (i - first))</tt> without modification.
 * - For SubtractRight, if the right element is out of bounds, the iterator is
 *   assigned to <tt>\*(result + (i - first))</tt> without modification.
 *
 * \par Snippet
 * The code snippet below illustrates how to use \p DeviceAdjacentDifference to
 * compute the left difference between adjacent elements.
 *
 * \par
 * \code
 * #include <cub/cub.cuh>   // or equivalently <cub/device/device_adjacent_difference.cuh>
 *
 * // Declare, allocate, and initialize device-accessible pointers
 * int  num_items;       // e.g., 8
 * int  *d_values;       // e.g., [1, 2, 1, 2, 1, 2, 1, 2]
 * //...
 *
 * // Determine temporary device storage requirements
 * void     *d_temp_storage = NULL;
 * size_t   temp_storage_bytes = 0;
 *
 * cub::DeviceAdjacentDifference::SubtractLeft(d_temp_storage, temp_storage_bytes, d_values, num_items);
 *
 * // Allocate temporary storage
 * cudaMalloc(&d_temp_storage, temp_storage_bytes);
 *
 * // Run operation
 * cub::DeviceAdjacentDifference::SubtractLeft(d_temp_storage, temp_storage_bytes, d_values, num_items);
 *
 * // d_values <-- [1, 1, -1, 1, -1, 1, -1, 1]
 * \endcode
 */
struct DeviceAdjacentDifference
{
  /**
   * \brief Subtracts the left element of each adjacent pair of elements residing within device-accessible memory.
   * \ingroup SingleModule
   *
   * \par Overview
   * - Calculates the differences of adjacent elements in
   *   the d_input. That is, <tt>\*d_input</tt> is assigned to
   *   <tt>\*d_output</tt>, and, for each iterator \p i in the range
   *   <tt>[d_input + 1, d_input + num_items)</tt>, the difference of
   *   <tt>\*i</tt> and <tt>*(i - 1)</tt> is assigned to
   *   <tt>\*(d_output + (i - d_input))</tt>.
   * - Note that the behavior is undefined if the input and output ranges
   *   overlap in any way.
   *
   * \par Snippet
   * The code snippet below illustrates how to use \p DeviceAdjacentDifference to
   * compute the difference between adjacent elements.
   *
   * \par
   * \code
   * #include <cub/cub.cuh>   // or equivalently <cub/device/device_adjacent_difference.cuh>
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
   * // Declare, allocate, and initialize device-accessible pointers
   * int  num_items;      // e.g., 8
   * int  *d_input;       // e.g., [1, 2, 1, 2, 1, 2, 1, 2]
   * int  *d_output;
   * ...
   *
   * // Run operation
   * cub::DeviceAdjacentDifference::SubtractLeftCopy(d_input, d_output, num_items, CustomDifference());
   *
   * // d_input; <-- [1, 2, 1, 2, 1, 2, 1, 2]
   * // d_output <-- [1, 1, -1, 1, -1, 1, -1, 1]
   * \endcode
   *
   * \tparam InputIteratorT is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>,
   *         and \c x and \c y are objects of \p InputIteratorT's \c value_type, then \c x - \c y is defined,
   *         and \p InputIteratorT's \c value_type is convertible to a type in \p OutputIteratorT's set of \c value_types,
   *         and the return type of <tt>x - y</tt> is convertible to a type in \p OutputIteratorT's set of \c value_types.
   * \tparam OutputIteratorT is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>.
   * \tparam DifferenceOpT's \c result_type is convertible to a type in \p OutputIteratorT's set of \c value_types.
   * \tparam OffsetT is an integer type for global offsets.
   */
  template <typename InputIteratorT,
            typename OutputIteratorT,
            typename DifferenceOpT,
            typename OffsetT>
  static CUB_RUNTIME_FUNCTION cudaError_t
  SubtractLeftCopy(InputIteratorT d_input,          ///< [in] Pointer to the input sequence
                   OutputIteratorT d_output,        ///< [out] Pointer to the output sequence
                   OffsetT num_items,               ///< [in] Number of items in the input sequence
                   DifferenceOpT difference_op,     ///< [in] The binary function used to compute differences.
                   cudaStream_t stream = 0,         ///< [in] <b>[optional]</b> CUDA stream to launch kernels within. Default is stream<sub>0</sub>.
                   bool debug_synchronous = false)  ///< [in] <b>[optional]</b> Whether or not to synchronize the stream after every kernel launch to check for errors.  Also causes launch configurations to be printed to the console.  Default is \p false.
  {
    constexpr bool in_place = false;
    constexpr bool read_left = true;
    using DispatchAdjacentDifferenceT =
      DispatchAdjacentDifference<InputIteratorT,
                                 OutputIteratorT,
                                 DifferenceOpT,
                                 OffsetT,
                                 in_place,
                                 read_left>;

    std::size_t temp_storage_bytes {};
    return DispatchAdjacentDifferenceT::Dispatch(nullptr,
                                                 temp_storage_bytes,
                                                 d_input,
                                                 d_output,
                                                 num_items,
                                                 difference_op,
                                                 stream,
                                                 debug_synchronous);
  }

  /**
   * \brief Subtracts the left element of each adjacent pair of elements residing within device-accessible memory.
   * \ingroup SingleModule
   *
   * \par Overview
   * - Calculates the differences of adjacent elements in
   *   the d_input. That is, <tt>\*d_input</tt> is assigned to
   *   <tt>\*d_output</tt>, and, for each iterator \p i in the range
   *   <tt>[d_input + 1, d_input + num_items)</tt>, the difference of
   *   <tt>\*i</tt> and <tt>*(i - 1)</tt> is assigned to
   *   <tt>\*(d_output + (i - d_input))</tt>.
   * - Note that the behavior is undefined if the input and output ranges
   *   overlap in any way.
   *
   * \par Snippet
   * The code snippet below illustrates how to use \p DeviceAdjacentDifference to
   * compute the difference between adjacent elements.
   *
   * \par
   * \code
   * #include <cub/cub.cuh>   // or equivalently <cub/device/device_adjacent_difference.cuh>
   *
   * // Declare, allocate, and initialize device-accessible pointers
   * int  num_items;      // e.g., 8
   * int  *d_input;       // e.g., [1, 2, 1, 2, 1, 2, 1, 2]
   * int  *d_output;
   * ...
   *
   * // Run operation
   * cub::DeviceAdjacentDifference::SubtractLeftCopy(d_input, d_output, num_items);
   *
   * // d_input; <-- [1, 2, 1, 2, 1, 2, 1, 2]
   * // d_output <-- [1, 1, -1, 1, -1, 1, -1, 1]
   * \endcode
   *
   * \tparam InputIteratorT is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>,
   *         and \c x and \c y are objects of \p InputIteratorT's \c value_type, then \c x - \c y is defined,
   *         and \p InputIteratorT's \c value_type is convertible to a type in \p OutputIteratorT's set of \c value_types,
   *         and the return type of <tt>x - y</tt> is convertible to a type in \p OutputIteratorT's set of \c value_types.
   * \tparam OutputIteratorT is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>.
   * \tparam OffsetT is an integer type for global offsets.
   */
  template <typename InputIteratorT,
            typename OutputIteratorT,
            typename OffsetT>
  static CUB_RUNTIME_FUNCTION cudaError_t
  SubtractLeftCopy(InputIteratorT d_input,          ///< [in] Pointer to the input sequence
                   OutputIteratorT d_output,        ///< [out] Pointer to the output sequence
                   OffsetT num_items,               ///< [in] Number of items in the input sequence
                   cudaStream_t stream = 0,         ///< [in] <b>[optional]</b> CUDA stream to launch kernels within. Default is stream<sub>0</sub>.
                   bool debug_synchronous = false)  ///< [in] <b>[optional]</b> Whether or not to synchronize the stream after every kernel launch to check for errors.  Also causes launch configurations to be printed to the console.  Default is \p false.
  {
    constexpr bool in_place = false;
    constexpr bool read_left = true;
    using DispatchAdjacentDifferenceT =
      DispatchAdjacentDifference<InputIteratorT,
                                 OutputIteratorT,
                                 cub::Difference,
                                 OffsetT,
                                 in_place,
                                 read_left>;

    std::size_t temp_storage_bytes{};
    return DispatchAdjacentDifferenceT::Dispatch(nullptr,
                                                 temp_storage_bytes,
                                                 d_input,
                                                 d_output,
                                                 num_items,
                                                 cub::Difference(),
                                                 stream,
                                                 debug_synchronous);
  }

  /**
   * \brief Subtracts the left element of each adjacent pair of elements residing within device-accessible memory.
   * \ingroup SingleModule
   *
   * \par Overview
   * Calculates the differences of adjacent elements in
   * the d_input. That is, for each iterator \p i in the range
   * <tt>[d_input + 1, d_input + num_items)</tt>, the difference of
   * <tt>\*i</tt> and <tt>*(i - 1)</tt> is assigned to
   * <tt>\*(d_input + (i - d_input))</tt>.
   *
   * \par Snippet
   * The code snippet below illustrates how to use \p DeviceAdjacentDifference to
   * compute the difference between adjacent elements.
   *
   * \par
   * \code
   * #include <cub/cub.cuh>   // or equivalently <cub/device/device_adjacent_difference.cuh>
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
   * // Declare, allocate, and initialize device-accessible pointers
   * int  num_items;     // e.g., 8
   * int  *d_data;       // e.g., [1, 2, 1, 2, 1, 2, 1, 2]
   * ...
   *
   * // Determine temporary device storage requirements
   * void     *d_temp_storage = NULL;
   * size_t   temp_storage_bytes = 0;
   * cub::DeviceAdjacentDifference::SubtractLeft(d_temp_storage, temp_storage_bytes, d_data, num_items, CustomDifference());
   *
   * // Allocate temporary storage
   * cudaMalloc(&d_temp_storage, temp_storage_bytes);
   *
   * // Run operation
   * cub::DeviceAdjacentDifference::SubtractLeft(d_temp_storage, temp_storage_bytes, d_data, num_items, CustomDifference());
   *
   * // d_data <-- [1, 1, -1, 1, -1, 1, -1, 1]
   * \endcode
   *
   * \tparam RandomAccessIteratorT is a model of <a href="https://en.cppreference.com/w/cpp/iterator/random_access_iterator">Random Access Iterator</a>,
   *         \p RandomAccessIteratorT is mutable. If \c x and \c y are objects of \p RandomAccessIteratorT's \c value_type, and \c x - \c y is defined,
   *         then the return type of <tt>x - y</tt> should be convertible to a type in \p RandomAccessIteratorT's set of \c value_types.
   * \tparam DifferenceOpT's \c result_type is convertible to a type in \p RandomAccessIteratorT's set of \c value_types.
   * \tparam OffsetT is an integer type for global offsets.
   */
  template <typename RandomAccessIteratorT,
            typename DifferenceOpT,
            typename OffsetT>
  static CUB_RUNTIME_FUNCTION cudaError_t
  SubtractLeft(void *d_temp_storage,            ///< [in] Device-accessible allocation of temporary storage. When NULL, the required allocation size is written to \p temp_storage_bytes and no work is done.
               std::size_t &temp_storage_bytes, ///< [in,out] Reference to size in bytes of \p d_temp_storage allocation
               RandomAccessIteratorT d_input,   ///< [in,out] Pointer to the input sequence
               OffsetT num_items,               ///< [in] Number of items in the input sequence
               DifferenceOpT difference_op,     ///< [in] The binary function used to compute differences.
               cudaStream_t stream = 0,         ///< [in] <b>[optional]</b> CUDA stream to launch kernels within. Default is stream<sub>0</sub>.
               bool debug_synchronous = false)  ///< [in] <b>[optional]</b> Whether or not to synchronize the stream after every kernel launch to check for errors.  Also causes launch configurations to be printed to the console.  Default is \p false.
  {
    constexpr bool in_place = true;
    constexpr bool read_left = true;
    using DispatchAdjacentDifferenceT =
      DispatchAdjacentDifference<RandomAccessIteratorT,
                                 RandomAccessIteratorT,
                                 DifferenceOpT,
                                 OffsetT,
                                 in_place,
                                 read_left>;

    return DispatchAdjacentDifferenceT::Dispatch(d_temp_storage,
                                                 temp_storage_bytes,
                                                 d_input,
                                                 d_input,
                                                 num_items,
                                                 difference_op,
                                                 stream,
                                                 debug_synchronous);
  }

  /**
   * \brief Subtracts the left element of each adjacent pair of elements residing within device-accessible memory.
   * \ingroup SingleModule
   *
   * \par Overview
   * Calculates the differences of adjacent elements in
   * the d_input. That is, for each iterator \p i in the range
   * <tt>[d_input + 1, d_input + num_items)</tt>, the difference of
   * <tt>\*i</tt> and <tt>*(i - 1)</tt> is assigned to
   * <tt>\*(d_input + (i - d_input))</tt>.
   *
   * \par Snippet
   * The code snippet below illustrates how to use \p DeviceAdjacentDifference to
   * compute the difference between adjacent elements.
   *
   * \par
   * \code
   * #include <cub/cub.cuh>   // or equivalently <cub/device/device_adjacent_difference.cuh>
   *
   * // Declare, allocate, and initialize device-accessible pointers
   * int  num_items;     // e.g., 8
   * int  *d_data;       // e.g., [1, 2, 1, 2, 1, 2, 1, 2]
   * ...
   *
   * // Determine temporary device storage requirements
   * void     *d_temp_storage = NULL;
   * size_t   temp_storage_bytes = 0;
   * cub::DeviceAdjacentDifference::SubtractLeft(d_temp_storage, temp_storage_bytes, d_data, num_items);
   *
   * // Allocate temporary storage
   * cudaMalloc(&d_temp_storage, temp_storage_bytes);
   *
   * // Run operation
   * cub::DeviceAdjacentDifference::SubtractLeft(d_temp_storage, temp_storage_bytes, d_data, num_items);
   *
   * // d_data <-- [1, 1, -1, 1, -1, 1, -1, 1]
   * \endcode
   *
   * \tparam RandomAccessIteratorT is a model of <a href="https://en.cppreference.com/w/cpp/iterator/random_access_iterator">Random Access Iterator</a>,
   *         \p RandomAccessIteratorT is mutable. If \c x and \c y are objects of \p RandomAccessIteratorT's \c value_type, and \c x - \c y is defined,
   *         then the return type of <tt>x - y</tt> should be convertible to a type in \p RandomAccessIteratorT's set of \c value_types.
   * \tparam OffsetT is an integer type for global offsets.
   */
  template <typename RandomAccessIteratorT,
            typename OffsetT>
  static CUB_RUNTIME_FUNCTION cudaError_t
  SubtractLeft(void *d_temp_storage,            ///< [in] Device-accessible allocation of temporary storage. When NULL, the required allocation size is written to \p temp_storage_bytes and no work is done.
               std::size_t &temp_storage_bytes, ///< [in,out] Reference to size in bytes of \p d_temp_storage allocation
               RandomAccessIteratorT d_input,   ///< [in,out] Pointer to the input sequence
               OffsetT num_items,               ///< [in] Number of items in the input sequence
               cudaStream_t stream = 0,         ///< [in] <b>[optional]</b> CUDA stream to launch kernels within. Default is stream<sub>0</sub>.
               bool debug_synchronous = false)  ///< [in] <b>[optional]</b> Whether or not to synchronize the stream after every kernel launch to check for errors.  Also causes launch configurations to be printed to the console.  Default is \p false.
  {
    constexpr bool in_place = true;
    constexpr bool read_left = true;
    using DispatchAdjacentDifferenceT =
      DispatchAdjacentDifference<RandomAccessIteratorT,
                                 RandomAccessIteratorT,
                                 cub::Difference,
                                 OffsetT,
                                 in_place,
                                 read_left>;

    return DispatchAdjacentDifferenceT::Dispatch(d_temp_storage,
                                                 temp_storage_bytes,
                                                 d_input,
                                                 d_input,
                                                 num_items,
                                                 cub::Difference(),
                                                 stream,
                                                 debug_synchronous);
  }

  /**
   * \brief Subtracts the right element of each adjacent pair of elements residing within device-accessible memory.
   * \ingroup SingleModule
   *
   * \par Overview
   * - Calculates the right differences of adjacent elements in
   *   the d_input. That is, <tt>\*(d_input + num_items - 1)</tt> is assigned to
   *   <tt>\*(d_output + num_items - 1)</tt>, and, for each iterator \p i in the range
   *   <tt>[d_input, d_input + num_items - 1)</tt>, the difference of
   *   <tt>\*i</tt> and <tt>*(i + 1)</tt> is assigned to
   *   <tt>\*(d_output + (i - d_input))</tt>.
   * - Note that the behavior is undefined if the input and output ranges
   *   overlap in any way.
   *
   * \par Snippet
   * The code snippet below illustrates how to use \p DeviceAdjacentDifference to
   * compute the difference between adjacent elements.
   *
   * \par
   * \code
   * #include <cub/cub.cuh>   // or equivalently <cub/device/device_adjacent_difference.cuh>
   *
   * // Declare, allocate, and initialize device-accessible pointers
   * int  num_items;     // e.g., 8
   * int  *d_input;      // e.g., [1, 2, 1, 2, 1, 2, 1, 2]
   * int  *d_output;
   * ...
   *
   * // Run operation
   * cub::DeviceAdjacentDifference::SubtractRightCopy(d_input, d_output, num_items);
   *
   * // d_input <-- [1, 2, 1, 2, 1, 2, 1, 2]
   * // d_data  <-- [-1, 1, -1, 1, -1, 1, -1, 2]
   * \endcode
   *
   * \tparam InputIteratorT is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>,
   *         and \c x and \c y are objects of \p InputIteratorT's \c value_type, then \c x - \c y is defined,
   *         and \p InputIteratorT's \c value_type is convertible to a type in \p OutputIteratorT's set of \c value_types,
   *         and the return type of <tt>x - y</tt> is convertible to a type in \p OutputIteratorT's set of \c value_types.
   * \tparam OutputIteratorT is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>.
   * \tparam OffsetT is an integer type for global offsets.
   */
  template <typename InputIteratorT,
            typename OutputIteratorT,
            typename OffsetT>
  static CUB_RUNTIME_FUNCTION cudaError_t
  SubtractRightCopy(InputIteratorT d_input,          ///< [in] Pointer to the input sequence
                    OutputIteratorT d_output,        ///< [out] Pointer to the output sequence
                    OffsetT num_items,               ///< [in] Number of items in the input sequence
                    cudaStream_t stream = 0,         ///< [in] <b>[optional]</b> CUDA stream to launch kernels within. Default is stream<sub>0</sub>.
                    bool debug_synchronous = false)  ///< [in] <b>[optional]</b> Whether or not to synchronize the stream after every kernel launch to check for errors.  Also causes launch configurations to be printed to the console.  Default is \p false.
  {
    constexpr bool in_place  = false;
    constexpr bool read_left = false;
    using DispatchAdjacentDifferenceT =
      DispatchAdjacentDifference<InputIteratorT,
                                 OutputIteratorT,
                                 cub::Difference,
                                 OffsetT,
                                 in_place,
                                 read_left>;

    std::size_t temp_storage_bytes;
    return DispatchAdjacentDifferenceT::Dispatch(nullptr,
                                                 temp_storage_bytes,
                                                 d_input,
                                                 d_output,
                                                 num_items,
                                                 cub::Difference(),
                                                 stream,
                                                 debug_synchronous);
  }

  /**
   * \brief Subtracts the right element of each adjacent pair of elements residing within device-accessible memory.
   * \ingroup SingleModule
   *
   * \par Overview
   * - Calculates the right differences of adjacent elements in
   *   the d_input. That is, <tt>\*(d_input + num_items - 1)</tt> is assigned to
   *   <tt>\*(d_output + num_items - 1)</tt>, and, for each iterator \p i in the range
   *   <tt>[d_input, d_input + num_items - 1)</tt>, the difference of
   *   <tt>\*i</tt> and <tt>*(i + 1)</tt> is assigned to
   *   <tt>\*(d_output + (i - d_input))</tt>.
   * - Note that the behavior is undefined if the input and output ranges
   *   overlap in any way.
   *
   * \par Snippet
   * The code snippet below illustrates how to use \p DeviceAdjacentDifference to
   * compute the difference between adjacent elements.
   *
   * \par
   * \code
   * #include <cub/cub.cuh>   // or equivalently <cub/device/device_adjacent_difference.cuh>
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
   * // Declare, allocate, and initialize device-accessible pointers
   * int  num_items;     // e.g., 8
   * int  *d_input;      // e.g., [1, 2, 1, 2, 1, 2, 1, 2]
   * int  *d_output;
   * ...
   *
   * // Run operation
   * cub::DeviceAdjacentDifference::SubtractRightCopy(d_input, d_output, num_items, CustomDifference());
   *
   * // d_input <-- [1, 2, 1, 2, 1, 2, 1, 2]
   * // d_data  <-- [-1, 1, -1, 1, -1, 1, -1, 2]
   * \endcode
   *
   * \tparam InputIteratorT is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>,
   *         and \c x and \c y are objects of \p InputIteratorT's \c value_type, then \c x - \c y is defined,
   *         and \p InputIteratorT's \c value_type is convertible to a type in \p OutputIteratorT's set of \c value_types,
   *         and the return type of <tt>x - y</tt> is convertible to a type in \p OutputIteratorT's set of \c value_types.
   * \tparam OutputIteratorT is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>.
   * \tparam DifferenceOpT's \c result_type is convertible to a type in \p RandomAccessIteratorT's set of \c value_types.
   * \tparam OffsetT is an integer type for global offsets.
   */
  template <typename InputIteratorT,
            typename OutputIteratorT,
            typename DifferenceOpT,
            typename OffsetT>
  static CUB_RUNTIME_FUNCTION cudaError_t
  SubtractRightCopy(InputIteratorT d_input,          ///< [in] Pointer to the input sequence
                    OutputIteratorT d_output,        ///< [out] Pointer to the output sequence
                    OffsetT num_items,               ///< [in] Number of items in the input sequence
                    DifferenceOpT difference_op,     ///< [in] The binary function used to compute differences.
                    cudaStream_t stream = 0,         ///< [in] <b>[optional]</b> CUDA stream to launch kernels within. Default is stream<sub>0</sub>.
                    bool debug_synchronous = false)  ///< [in] <b>[optional]</b> Whether or not to synchronize the stream after every kernel launch to check for errors.  Also causes launch configurations to be printed to the console.  Default is \p false.
  {
    constexpr bool in_place  = false;
    constexpr bool read_left = false;
    using DispatchAdjacentDifferenceT =
      DispatchAdjacentDifference<InputIteratorT,
                                 OutputIteratorT,
                                 DifferenceOpT,
                                 OffsetT,
                                 in_place,
                                 read_left>;

    std::size_t temp_storage_bytes;
    return DispatchAdjacentDifferenceT::Dispatch(nullptr,
                                                 temp_storage_bytes,
                                                 d_input,
                                                 d_output,
                                                 num_items,
                                                 difference_op,
                                                 stream,
                                                 debug_synchronous);
  }

  /**
   * \brief Subtracts the right element of each adjacent pair of elements residing within device-accessible memory.
   * \ingroup SingleModule
   *
   * \par Overview
   * Calculates the right differences of adjacent elements in
   * the d_input. That is, for each iterator \p i in the range
   * <tt>[d_input, d_input + num_items - 1)</tt>, the difference of
   * <tt>\*i</tt> and <tt>*(i + 1)</tt> is assigned to
   * <tt>\*(d_input + (i - d_input))</tt>.
   *
   * \par Snippet
   * The code snippet below illustrates how to use \p DeviceAdjacentDifference to
   * compute the difference between adjacent elements.
   *
   * \par
   * \code
   * #include <cub/cub.cuh>   // or equivalently <cub/device/device_adjacent_difference.cuh>
   *
   * // Declare, allocate, and initialize device-accessible pointers
   * int  num_items;    // e.g., 8
   * int  *d_data;      // e.g., [1, 2, 1, 2, 1, 2, 1, 2]
   * ...
   *
   * // Determine temporary device storage requirements
   * void     *d_temp_storage = NULL;
   * size_t   temp_storage_bytes = 0;
   * cub::DeviceAdjacentDifference::SubtractRight(d_temp_storage, temp_storage_bytes, d_data, num_items);
   *
   * // Allocate temporary storage
   * cudaMalloc(&d_temp_storage, temp_storage_bytes);
   *
   * // Run operation
   * cub::DeviceAdjacentDifference::SubtractRight(d_temp_storage, temp_storage_bytes, d_data, num_items);
   *
   * // d_data  <-- [-1, 1, -1, 1, -1, 1, -1, 2]
   * \endcode
   *
   * \tparam RandomAccessIteratorT is a model of <a href="https://en.cppreference.com/w/cpp/iterator/random_access_iterator">Random Access Iterator</a>,
   *         \p RandomAccessIteratorT is mutable. If \c x and \c y are objects of \p RandomAccessIteratorT's \c value_type, and \c x - \c y is defined,
   *         then the return type of <tt>x - y</tt> should be convertible to a type in \p RandomAccessIteratorT's set of \c value_types.
   * \tparam OffsetT is an integer type for global offsets.
   */
  template <typename RandomAccessIteratorT,
            typename OffsetT>
  static CUB_RUNTIME_FUNCTION cudaError_t
  SubtractRight(void *d_temp_storage,            ///< [in] Device-accessible allocation of temporary storage. When NULL, the required allocation size is written to \p temp_storage_bytes and no work is done.
                std::size_t &temp_storage_bytes, ///< [in,out] Reference to size in bytes of \p d_temp_storage allocation
                RandomAccessIteratorT d_input,   ///< [in,out] Pointer to the input sequence
                OffsetT num_items,               ///< [in] Number of items in the input sequence
                cudaStream_t stream = 0,         ///< [in] <b>[optional]</b> CUDA stream to launch kernels within. Default is stream<sub>0</sub>.
                bool debug_synchronous = false)  ///< [in] <b>[optional]</b> Whether or not to synchronize the stream after every kernel launch to check for errors.  Also causes launch configurations to be printed to the console.  Default is \p false.
  {
    constexpr bool in_place = true;
    constexpr bool read_left = false;
    using DispatchAdjacentDifferenceT =
      DispatchAdjacentDifference<RandomAccessIteratorT,
                                 RandomAccessIteratorT,
                                 cub::Difference,
                                 OffsetT,
                                 in_place,
                                 read_left>;

    return DispatchAdjacentDifferenceT::Dispatch(d_temp_storage,
                                                 temp_storage_bytes,
                                                 d_input,
                                                 d_input,
                                                 num_items,
                                                 cub::Difference(),
                                                 stream,
                                                 debug_synchronous);
  }

  /**
   * \brief Subtracts the right element of each adjacent pair of elements residing within device-accessible memory.
   * \ingroup SingleModule
   *
   * \par Overview
   * Calculates the right differences of adjacent elements in
   * the d_input. That is, for each iterator \p i in the range
   * <tt>[d_input, d_input + num_items - 1)</tt>, the difference of
   * <tt>\*i</tt> and <tt>*(i + 1)</tt> is assigned to
   * <tt>\*(d_input + (i - d_input))</tt>.
   *
   * \par Snippet
   * The code snippet below illustrates how to use \p DeviceAdjacentDifference to
   * compute the difference between adjacent elements.
   *
   * \par
   * \code
   * #include <cub/cub.cuh>   // or equivalently <cub/device/device_adjacent_difference.cuh>
   *
   * // Declare, allocate, and initialize device-accessible pointers
   * int  num_items;    // e.g., 8
   * int  *d_data;      // e.g., [1, 2, 1, 2, 1, 2, 1, 2]
   * ...
   *
   * // Determine temporary device storage requirements
   * void     *d_temp_storage = NULL;
   * size_t   temp_storage_bytes = 0;
   * cub::DeviceAdjacentDifference::SubtractRight(d_temp_storage, temp_storage_bytes, d_data, num_items);
   *
   * // Allocate temporary storage
   * cudaMalloc(&d_temp_storage, temp_storage_bytes);
   *
   * // Run operation
   * cub::DeviceAdjacentDifference::SubtractRight(d_temp_storage, temp_storage_bytes, d_data, num_items);
   *
   * // d_data  <-- [-1, 1, -1, 1, -1, 1, -1, 2]
   * \endcode
   *
   * \tparam RandomAccessIteratorT is a model of <a href="https://en.cppreference.com/w/cpp/iterator/random_access_iterator">Random Access Iterator</a>,
   *         \p RandomAccessIteratorT is mutable. If \c x and \c y are objects of \p RandomAccessIteratorT's \c value_type, and \c x - \c y is defined,
   *         then the return type of <tt>x - y</tt> should be convertible to a type in \p RandomAccessIteratorT's set of \c value_types.
   * \tparam DifferenceOpT's \c result_type is convertible to a type in \p RandomAccessIteratorT's set of \c value_types.
   * \tparam OffsetT is an integer type for global offsets.
   */
  template <typename RandomAccessIteratorT,
            typename DifferenceOpT,
            typename OffsetT>
  static CUB_RUNTIME_FUNCTION cudaError_t
  SubtractRight(void *d_temp_storage,            ///< [in] Device-accessible allocation of temporary storage. When NULL, the required allocation size is written to \p temp_storage_bytes and no work is done.
                std::size_t &temp_storage_bytes, ///< [in,out] Reference to size in bytes of \p d_temp_storage allocation
                RandomAccessIteratorT d_input,   ///< [in,out] Pointer to the input sequence
                OffsetT num_items,               ///< [in] Number of items in the input sequence
                DifferenceOpT difference_op,     ///< [in] The binary function used to compute differences.
                cudaStream_t stream = 0,         ///< [in] <b>[optional]</b> CUDA stream to launch kernels within. Default is stream<sub>0</sub>.
                bool debug_synchronous = false)  ///< [in] <b>[optional]</b> Whether or not to synchronize the stream after every kernel launch to check for errors.  Also causes launch configurations to be printed to the console.  Default is \p false.
  {
    constexpr bool in_place = true;
    constexpr bool read_left = false;
    using DispatchAdjacentDifferenceT =
      DispatchAdjacentDifference<RandomAccessIteratorT,
                                 RandomAccessIteratorT,
                                 DifferenceOpT,
                                 OffsetT,
                                 in_place,
                                 read_left>;

    return DispatchAdjacentDifferenceT::Dispatch(d_temp_storage,
                                                 temp_storage_bytes,
                                                 d_input,
                                                 d_input,
                                                 num_items,
                                                 difference_op,
                                                 stream,
                                                 debug_synchronous);
  }
};

}               // CUB namespace
CUB_NS_POSTFIX  // Optional outer namespace(s)
