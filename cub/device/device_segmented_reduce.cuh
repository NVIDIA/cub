
/******************************************************************************
 * Copyright (c) 2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2015, NVIDIA CORPORATION.  All rights reserved.
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
 * cub::DeviceSegmentedReduce provides device-wide, parallel operations for computing a batched reduction across multiple sequences of data items residing within device-accessible memory.
 */

#pragma once

#include <stdio.h>
#include <iterator>

#include "dispatch/dispatch_reduce.cuh"
#include "dispatch/dispatch_reduce_by_key.cuh"
#include "../util_namespace.cuh"

/// Optional outer namespace(s)
CUB_NS_PREFIX

/// CUB namespace
namespace cub {


/**
 * \brief DeviceSegmentedReduce provides device-wide, parallel operations for computing a reduction across multiple sequences of data items residing within device-accessible memory. ![](reduce_logo.png)
 * \ingroup SegmentedModule
 *
 * \par Overview
 * A <a href="http://en.wikipedia.org/wiki/Reduce_(higher-order_function)"><em>reduction</em></a> (or <em>fold</em>)
 * uses a binary combining operator to compute a single aggregate from a sequence of input elements.
 *
 * \par Usage Considerations
 * \cdp_class{DeviceSegmentedReduce}
 *
 */
struct DeviceSegmentedReduce
{
    /**
     * \brief Computes a device-wide segmented reduction using the specified binary \p reduction_op functor.
     *
     * \par
     * - The specified \p identity value is produced for zero-length segments.  \p identity is
     *   assumed to be arithmetically neutral for the given operation, i.e., <tt>reduction_op(XXX, identity)</tt>
     *   will return \p XXX.
     * - Does not support non-commutative reduction operators.
     * - When input a contiguous sequence of segments, a single sequence
     *   \p segment_offsets (of length <tt>num_segments+1</tt>) can be aliased
     *   for both the \p d_begin_offsets and \p d_end_offsets parameters (where
     *   the latter is specified as <tt>segment_offsets+1</tt>).
     * - \devicestorage
     *
     * \par Snippet
     * The code snippet below illustrates a custom min reduction of a device vector of \p int items.
     * \par
     * \code
     * #include <cub/cub.cuh>   // or equivalently <cub/device/device_radix_sort.cuh>
     *
     * // CustomMin functor
     * struct CustomMin
     * {
     *     template <typename T>
     *     CUB_RUNTIME_FUNCTION __forceinline__
     *     T operator()(const T &a, const T &b) const {
     *         return (b < a) ? b : a;
     *     }
     * };
     *
     * // Declare, allocate, and initialize device-accessible pointers for input and output
     * int          num_segments;   // e.g., 3
     * int          *d_offsets;     // e.g., [0, 3, 3, 7]
     * int          *d_in;          // e.g., [8, 6, 7, 5, 3, 0, 9]
     * int          *d_out;         // e.g., [-, -, -]
     * CustomMin    min_op;
     * int          identity;       // e.g., MAX_INT
     * ...
     *
     * // Determine temporary device storage requirements
     * void     *d_temp_storage = NULL;
     * size_t   temp_storage_bytes = 0;
     * cub::DeviceSegmentedReduce::Reduce(d_temp_storage, temp_storage_bytes, d_in, d_out,
     *     num_segments, d_offsets, d_offsets + 1, min_op, identity);
     *
     * // Allocate temporary storage
     * cudaMalloc(&d_temp_storage, temp_storage_bytes);
     *
     * // Run reduction
     * cub::DeviceSegmentedReduce::Reduce(d_temp_storage, temp_storage_bytes, d_in, d_out,
     *     num_segments, d_offsets, d_offsets + 1, min_op, identity);
     *
     * // d_out <-- [6, MAX_INT, 0]
     *
     * \endcode
     *
     * \tparam InputIteratorT     <b>[inferred]</b> Random-access input iterator type for reading input items \iterator
     * \tparam OutputIteratorT    <b>[inferred]</b> Output iterator type for recording the reduced aggregate \iterator
     * \tparam ReductionOp        <b>[inferred]</b> Binary reduction functor type having member <tt>T operator()(const T &a, const T &b)</tt> (e.g., cub::Sum, cub::Min, cub::Max, etc.)
     * \tparam T                  <b>[inferred]</b> Data type being reduced
     */
    template <
        typename            InputIteratorT,
        typename            OutputIteratorT,
        typename            ReductionOp,
        typename            T>
    CUB_RUNTIME_FUNCTION
    static cudaError_t Reduce(
        void                *d_temp_storage,                    ///< [in] %Device-accessible allocation of temporary storage.  When NULL, the required allocation size is written to \p temp_storage_bytes and no work is done.
        size_t              &temp_storage_bytes,                ///< [in,out] Reference to size in bytes of \p d_temp_storage allocation
        InputIteratorT      d_in,                               ///< [in] Pointer to the input sequence of data items
        OutputIteratorT     d_out,                              ///< [out] Pointer to the output aggregate
        int                 num_segments,                       ///< [in] The number of segments that comprise the sorting data
        int                 *d_begin_offsets,                   ///< [in] %Device-accessible pointer to the sequence of beginning offsets of length \p num_segments, such that <tt>d_begin_offsets[i]</tt> is the first element of the <em>i</em><sup>th</sup> data segment in <tt>d_keys_*</tt> and <tt>d_values_*</tt>
        int                 *d_end_offsets,                     ///< [in] %Device-accessible pointer to the sequence of ending offsets of length \p num_segments, such that <tt>d_end_offsets[i]-1</tt> is the last element of the <em>i</em><sup>th</sup> data segment in <tt>d_keys_*</tt> and <tt>d_values_*</tt>.  If <tt>d_end_offsets[i]-1</tt> <= <tt>d_begin_offsets[i]</tt>, the <em>i</em><sup>th</sup> is considered empty.
        ReductionOp         reduction_op,                       ///< [in] Binary reduction functor (e.g., an instance of cub::Sum, cub::Min, cub::Max, etc.)
        T                   identity,                           ///< [in] The identity value to be returned for zero-length segments
        cudaStream_t        stream              = 0,            ///< [in] <b>[optional]</b> CUDA stream to launch kernels within.  Default is stream<sub>0</sub>.
        bool                debug_synchronous   = false)        ///< [in] <b>[optional]</b> Whether or not to synchronize the stream after every kernel launch to check for errors.  Also causes launch configurations to be printed to the console.  Default is \p false.
    {
        // Signed integer type for global offsets
        typedef int OffsetT;

        return DispatchSegmentedReduce<InputIteratorT, OutputIteratorT, OffsetT, ReductionOp>::Dispatch(
            d_temp_storage,
            temp_storage_bytes,
            d_in,
            d_out,
            num_segments,
            d_begin_offsets,
            d_end_offsets,
            reduction_op,
            identity,
            stream,
            debug_synchronous);
    }


    /**
     * \brief Computes a device-wide segmented sum using the addition ('+') operator.
     *
     * \par
     * - The specified \p identity value is produced for zero-length segments.  \p identity is
     *   assumed to be arithmetically neutral for the input type's addition operation, i.e.,
     *   <tt>XXX + identity = XXX </tt>
     * - When input a contiguous sequence of segments, a single sequence
     *   \p segment_offsets (of length <tt>num_segments+1</tt>) can be aliased
     *   for both the \p d_begin_offsets and \p d_end_offsets parameters (where
     *   the latter is specified as <tt>segment_offsets+1</tt>).
     * - Does not support non-commutative reduction operators.
     * - \devicestorage
     *
     * \par Snippet
     * The code snippet below illustrates the sum reduction of a device vector of \p int items.
     * \par
     * \code
     * #include <cub/cub.cuh>   // or equivalently <cub/device/device_radix_sort.cuh>
     *
     * // Declare, allocate, and initialize device-accessible pointers for input and output
     * int num_segments;   // e.g., 3
     * int *d_offsets;     // e.g., [0, 3, 3, 7]
     * int *d_in;          // e.g., [8, 6, 7, 5, 3, 0, 9]
     * int *d_out;         // e.g., [-, -, -]
     * int identity        // e.g., 0
     * ...
     *
     * // Determine temporary device storage requirements
     * void     *d_temp_storage = NULL;
     * size_t   temp_storage_bytes = 0;
     * cub::DeviceSegmentedReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_out,
     *     num_segments, d_offsets, d_offsets + 1, identity);
     *
     * // Allocate temporary storage
     * cudaMalloc(&d_temp_storage, temp_storage_bytes);
     *
     * // Run sum-reduction
     * cub::DeviceSegmentedReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_out,
     *     num_segments, d_offsets, d_offsets + 1, identity);
     *
     * // d_out <-- [21, 0, 17]
     *
     * \endcode
     *
     * \tparam InputIteratorT     <b>[inferred]</b> Random-access input iterator type for reading input items \iterator
     * \tparam OutputIteratorT    <b>[inferred]</b> Output iterator type for recording the reduced aggregate \iterator
     * \tparam T                  <b>[inferred]</b> Data type being reduced
     */
    template <
        typename            InputIteratorT,
        typename            OutputIteratorT,
        typename            T>
    CUB_RUNTIME_FUNCTION
    static cudaError_t Sum(
        void                *d_temp_storage,                    ///< [in] %Device-accessible allocation of temporary storage.  When NULL, the required allocation size is written to \p temp_storage_bytes and no work is done.
        size_t              &temp_storage_bytes,                ///< [in,out] Reference to size in bytes of \p d_temp_storage allocation
        InputIteratorT      d_in,                               ///< [in] Pointer to the input sequence of data items
        OutputIteratorT     d_out,                              ///< [out] Pointer to the output aggregate
        int                 num_segments,                       ///< [in] The number of segments that comprise the sorting data
        int                 *d_begin_offsets,                   ///< [in] %Device-accessible pointer to the sequence of beginning offsets of length \p num_segments, such that <tt>d_begin_offsets[i]</tt> is the first element of the <em>i</em><sup>th</sup> data segment in <tt>d_keys_*</tt> and <tt>d_values_*</tt>
        int                 *d_end_offsets,                     ///< [in] %Device-accessible pointer to the sequence of ending offsets of length \p num_segments, such that <tt>d_end_offsets[i]-1</tt> is the last element of the <em>i</em><sup>th</sup> data segment in <tt>d_keys_*</tt> and <tt>d_values_*</tt>.  If <tt>d_end_offsets[i]-1</tt> <= <tt>d_begin_offsets[i]</tt>, the <em>i</em><sup>th</sup> is considered empty.
        T                   identity,                           ///< [in] The identity value to be returned for zero-length segments
        cudaStream_t        stream              = 0,            ///< [in] <b>[optional]</b> CUDA stream to launch kernels within.  Default is stream<sub>0</sub>.
        bool                debug_synchronous   = false)        ///< [in] <b>[optional]</b> Whether or not to synchronize the stream after every kernel launch to check for errors.  Also causes launch configurations to be printed to the console.  Default is \p false.
    {
        // Signed integer type for global offsets
        typedef int OffsetT;

        return DispatchSegmentedReduce<InputIteratorT, OutputIteratorT, OffsetT, cub::Sum>::Dispatch(
            d_temp_storage,
            temp_storage_bytes,
            d_in,
            d_out,
            num_segments,
            d_begin_offsets,
            d_end_offsets,
            cub::Sum(),
            identity,
            stream,
            debug_synchronous);
    }


    /**
     * \brief Computes a device-wide segmented minimum using the less-than ('<') operator.
     *
     * \par
     * - The specified \p identity value is produced for zero-length segments.  \p identity is
     *   assumed to be arithmetically neutral for the input type's \p < operation, i.e.,
     *   <tt>(identity < XXX) = false</tt>
     * - When input a contiguous sequence of segments, a single sequence
     *   \p segment_offsets (of length <tt>num_segments+1</tt>) can be aliased
     *   for both the \p d_begin_offsets and \p d_end_offsets parameters (where
     *   the latter is specified as <tt>segment_offsets+1</tt>).
     * - Does not support non-commutative minimum operators.
     * - \devicestorage
     *
     * \par Snippet
     * The code snippet below illustrates the min-reduction of a device vector of \p int items.
     * \par
     * \code
     * #include <cub/cub.cuh>   // or equivalently <cub/device/device_radix_sort.cuh>
     *
     * // Declare, allocate, and initialize device-accessible pointers for input and output
     * int num_segments;   // e.g., 3
     * int *d_offsets;     // e.g., [0, 3, 3, 7]
     * int *d_in;          // e.g., [8, 6, 7, 5, 3, 0, 9]
     * int *d_out;         // e.g., [-, -, -]
     * int identity;       // e.g., MAX_INT
     * ...
     *
     * // Determine temporary device storage requirements
     * void     *d_temp_storage = NULL;
     * size_t   temp_storage_bytes = 0;
     * cub::DeviceSegmentedReduce::Min(d_temp_storage, temp_storage_bytes, d_in, d_out,
     *     num_segments, d_offsets, d_offsets + 1, identity);
     *
     * // Allocate temporary storage
     * cudaMalloc(&d_temp_storage, temp_storage_bytes);
     *
     * // Run min-reduction
     * cub::DeviceSegmentedReduce::Min(d_temp_storage, temp_storage_bytes, d_in, d_out,
     *     num_segments, d_offsets, d_offsets + 1, identity);
     *
     * // d_out <-- [6, MAX_INT, 0]
     *
     * \endcode
     *
     * \tparam InputIteratorT     <b>[inferred]</b> Random-access input iterator type for reading input items \iterator
     * \tparam OutputIteratorT    <b>[inferred]</b> Output iterator type for recording the reduced aggregate \iterator
     * \tparam T                  <b>[inferred]</b> Data type being reduced
     */
    template <
        typename            InputIteratorT,
        typename            OutputIteratorT,
        typename            T>
    CUB_RUNTIME_FUNCTION
    static cudaError_t Min(
        void                *d_temp_storage,                    ///< [in] %Device-accessible allocation of temporary storage.  When NULL, the required allocation size is written to \p temp_storage_bytes and no work is done.
        size_t              &temp_storage_bytes,                ///< [in,out] Reference to size in bytes of \p d_temp_storage allocation
        InputIteratorT      d_in,                               ///< [in] Pointer to the input sequence of data items
        OutputIteratorT     d_out,                              ///< [out] Pointer to the output aggregate
        int                 num_segments,                       ///< [in] The number of segments that comprise the sorting data
        int                 *d_begin_offsets,                   ///< [in] %Device-accessible pointer to the sequence of beginning offsets of length \p num_segments, such that <tt>d_begin_offsets[i]</tt> is the first element of the <em>i</em><sup>th</sup> data segment in <tt>d_keys_*</tt> and <tt>d_values_*</tt>
        int                 *d_end_offsets,                     ///< [in] %Device-accessible pointer to the sequence of ending offsets of length \p num_segments, such that <tt>d_end_offsets[i]-1</tt> is the last element of the <em>i</em><sup>th</sup> data segment in <tt>d_keys_*</tt> and <tt>d_values_*</tt>.  If <tt>d_end_offsets[i]-1</tt> <= <tt>d_begin_offsets[i]</tt>, the <em>i</em><sup>th</sup> is considered empty.
        T                   identity,                           ///< [in] The identity value to be returned for zero-length segments
        cudaStream_t        stream              = 0,            ///< [in] <b>[optional]</b> CUDA stream to launch kernels within.  Default is stream<sub>0</sub>.
        bool                debug_synchronous   = false)        ///< [in] <b>[optional]</b> Whether or not to synchronize the stream after every kernel launch to check for errors.  Also causes launch configurations to be printed to the console.  Default is \p false.
    {
        // Signed integer type for global offsets
        typedef int OffsetT;

        return DispatchSegmentedReduce<InputIteratorT, OutputIteratorT, OffsetT, cub::Min>::Dispatch(
            d_temp_storage,
            temp_storage_bytes,
            d_in,
            d_out,
            num_segments,
            d_begin_offsets,
            d_end_offsets,
            cub::Min(),
            identity,
            stream,
            debug_synchronous);
    }


    /**
     * \brief Finds the first device-wide minimum in each segment using the less-than ('<') operator, also returning the in-segment index of that item.
     *
     * \par
     * - Assuming the input \p d_in has value type \p T, the output \p d_out must have value type
     *   <tt>KeyValuePair<int, T></tt>.  The minimum value is written to <tt>d_out.value</tt> and its
     *   location in the input array is written to <tt>d_out.key</tt>.
     * - The specified \p identity value is produced for zero-length segments.  \p identity is
     *   assumed to be arithmetically neutral for the input type's \p < operation, i.e.,
     *   <tt>(identity < XXX) = false</tt>
     * - When input a contiguous sequence of segments, a single sequence
     *   \p segment_offsets (of length <tt>num_segments+1</tt>) can be aliased
     *   for both the \p d_begin_offsets and \p d_end_offsets parameters (where
     *   the latter is specified as <tt>segment_offsets+1</tt>).
     * - Does not support non-commutative minimum operators.
     * - \devicestorage
     *
     * \par Snippet
     * The code snippet below illustrates the argmin-reduction of a device vector of \p int items.
     * \par
     * \code
     * #include <cub/cub.cuh>   // or equivalently <cub/device/device_radix_sort.cuh>
     *
     * // Declare, allocate, and initialize device-accessible pointers for input and output
     * int                      num_segments;   // e.g., 3
     * int                      *d_offsets;     // e.g., [0, 3, 3, 7]
     * int                      *d_in;          // e.g., [8, 6, 7, 5, 3, 0, 9]
     * KeyValuePair<int, int>   *d_out;         // e.g., [{-,-}, {-,-}, {-,-}]
     * int                      identity;       // e.g., MAX_INT
     * ...
     *
     * // Determine temporary device storage requirements
     * void     *d_temp_storage = NULL;
     * size_t   temp_storage_bytes = 0;
     * cub::DeviceSegmentedReduce::ArgMin(d_temp_storage, temp_storage_bytes, d_in, d_out,
     *     num_segments, d_offsets, d_offsets + 1, identity);
     *
     * // Allocate temporary storage
     * cudaMalloc(&d_temp_storage, temp_storage_bytes);
     *
     * // Run argmin-reduction
     * cub::DeviceSegmentedReduce::ArgMin(d_temp_storage, temp_storage_bytes, d_in, d_out,
     *     num_segments, d_offsets, d_offsets + 1, identity);
     *
     * // d_out <-- [{1,6}, {-1,MAX_INT}, {2,0}]
     *
     * \endcode
     *
     * \tparam InputIteratorT     <b>[inferred]</b> Random-access input iterator type for reading input items (of some type \p T) \iterator
     * \tparam OutputIteratorT    <b>[inferred]</b> Output iterator type for recording the reduced aggregate (having value type <tt>KeyValuePair<int, T></tt>) \iterator
     * \tparam T                  <b>[inferred]</b> Data type being reduced
     */
    template <
        typename            InputIteratorT,
        typename            OutputIteratorT,
        typename            T>
    CUB_RUNTIME_FUNCTION
    static cudaError_t ArgMin(
        void                *d_temp_storage,                    ///< [in] %Device-accessible allocation of temporary storage.  When NULL, the required allocation size is written to \p temp_storage_bytes and no work is done.
        size_t              &temp_storage_bytes,                ///< [in,out] Reference to size in bytes of \p d_temp_storage allocation
        InputIteratorT      d_in,                               ///< [in] Pointer to the input sequence of data items
        OutputIteratorT     d_out,                              ///< [out] Pointer to the output aggregate
        int                 num_segments,                       ///< [in] The number of segments that comprise the sorting data
        int                 *d_begin_offsets,                   ///< [in] %Device-accessible pointer to the sequence of beginning offsets of length \p num_segments, such that <tt>d_begin_offsets[i]</tt> is the first element of the <em>i</em><sup>th</sup> data segment in <tt>d_keys_*</tt> and <tt>d_values_*</tt>
        int                 *d_end_offsets,                     ///< [in] %Device-accessible pointer to the sequence of ending offsets of length \p num_segments, such that <tt>d_end_offsets[i]-1</tt> is the last element of the <em>i</em><sup>th</sup> data segment in <tt>d_keys_*</tt> and <tt>d_values_*</tt>.  If <tt>d_end_offsets[i]-1</tt> <= <tt>d_begin_offsets[i]</tt>, the <em>i</em><sup>th</sup> is considered empty.
        T                   identity,                           ///< [in] The identity value to be returned for zero-length segments
        cudaStream_t        stream              = 0,            ///< [in] <b>[optional]</b> CUDA stream to launch kernels within.  Default is stream<sub>0</sub>.
        bool                debug_synchronous   = false)        ///< [in] <b>[optional]</b> Whether or not to synchronize the stream after every kernel launch to check for errors.  Also causes launch configurations to be printed to the console.  Default is \p false.
    {
        // Signed integer type for global offsets
        typedef int OffsetT;

        // Wrapped input iterator
        typedef ArgIndexInputIterator<InputIteratorT, int> ArgIndexInputIteratorT;
        ArgIndexInputIteratorT d_argmin_in(d_in, 0);

        KeyValuePair<OffsetT, T> identity_pair = {-1, identity};

        return DispatchSegmentedReduce<ArgIndexInputIteratorT, OutputIteratorT, OffsetT, cub::ArgMin>::Dispatch(
            d_temp_storage,
            temp_storage_bytes,
            d_argmin_in,
            d_out,
            num_segments,
            d_begin_offsets,
            d_end_offsets,
            cub::ArgMin(),
            identity_pair,
            stream,
            debug_synchronous);
    }


    /**
     * \brief Computes a device-wide segmented maximum using the greater-than ('>') operator.
     *
     * \par
     * - The specified \p identity value is produced for zero-length segments.  \p identity is
     *   assumed to be arithmetically neutral for the input type's \p > operation, i.e.,
     *   <tt>(identity > XXX) = false</tt>
     * - When input a contiguous sequence of segments, a single sequence
     *   \p segment_offsets (of length <tt>num_segments+1</tt>) can be aliased
     *   for both the \p d_begin_offsets and \p d_end_offsets parameters (where
     *   the latter is specified as <tt>segment_offsets+1</tt>).
     * - Does not support non-commutative maximum operators.
     * - \devicestorage
     *
     * \par Snippet
     * The code snippet below illustrates the max-reduction of a device vector of \p int items.
     * \par
     * \code
     * #include <cub/cub.cuh>   // or equivalently <cub/device/device_radix_sort.cuh>
     *
     * // Declare, allocate, and initialize device-accessible pointers for input and output
     * int num_segments;   // e.g., 3
     * int *d_offsets;     // e.g., [0, 3, 3, 7]
     * int *d_in;          // e.g., [8, 6, 7, 5, 3, 0, 9]
     * int *d_out;         // e.g., [-, -, -]
     * int identity;       // e.g., MIN_INT
     * ...
     *
     * // Determine temporary device storage requirements
     * void     *d_temp_storage = NULL;
     * size_t   temp_storage_bytes = 0;
     * cub::DeviceSegmentedReduce::Max(d_temp_storage, temp_storage_bytes, d_in, d_out,
     *     num_segments, d_offsets, d_offsets + 1, identity);
     *
     * // Allocate temporary storage
     * cudaMalloc(&d_temp_storage, temp_storage_bytes);
     *
     * // Run max-reduction
     * cub::DeviceSegmentedReduce::Max(d_temp_storage, temp_storage_bytes, d_in, d_out,
     *     num_segments, d_offsets, d_offsets + 1, identity);
     *
     * // d_out <-- [8, MIN_INT, 9]
     *
     * \endcode
     *
     * \tparam InputIteratorT     <b>[inferred]</b> Random-access input iterator type for reading input items \iterator
     * \tparam OutputIteratorT    <b>[inferred]</b> Output iterator type for recording the reduced aggregate \iterator
     * \tparam T                  <b>[inferred]</b> Data type being reduced
     */
    template <
        typename            InputIteratorT,
        typename            OutputIteratorT,
        typename            T>
    CUB_RUNTIME_FUNCTION
    static cudaError_t Max(
        void                *d_temp_storage,                    ///< [in] %Device-accessible allocation of temporary storage.  When NULL, the required allocation size is written to \p temp_storage_bytes and no work is done.
        size_t              &temp_storage_bytes,                ///< [in,out] Reference to size in bytes of \p d_temp_storage allocation
        InputIteratorT      d_in,                               ///< [in] Pointer to the input sequence of data items
        OutputIteratorT     d_out,                              ///< [out] Pointer to the output aggregate
        int                 num_segments,                       ///< [in] The number of segments that comprise the sorting data
        int                 *d_begin_offsets,                   ///< [in] %Device-accessible pointer to the sequence of beginning offsets of length \p num_segments, such that <tt>d_begin_offsets[i]</tt> is the first element of the <em>i</em><sup>th</sup> data segment in <tt>d_keys_*</tt> and <tt>d_values_*</tt>
        int                 *d_end_offsets,                     ///< [in] %Device-accessible pointer to the sequence of ending offsets of length \p num_segments, such that <tt>d_end_offsets[i]-1</tt> is the last element of the <em>i</em><sup>th</sup> data segment in <tt>d_keys_*</tt> and <tt>d_values_*</tt>.  If <tt>d_end_offsets[i]-1</tt> <= <tt>d_begin_offsets[i]</tt>, the <em>i</em><sup>th</sup> is considered empty.
        T                   identity,                           ///< [in] The identity value to be returned for zero-length segments
        cudaStream_t        stream              = 0,            ///< [in] <b>[optional]</b> CUDA stream to launch kernels within.  Default is stream<sub>0</sub>.
        bool                debug_synchronous   = false)        ///< [in] <b>[optional]</b> Whether or not to synchronize the stream after every kernel launch to check for errors.  Also causes launch configurations to be printed to the console.  Default is \p false.
    {
        // Signed integer type for global offsets
        typedef int OffsetT;

        return DispatchSegmentedReduce<InputIteratorT, OutputIteratorT, OffsetT, cub::Max>::Dispatch(
            d_temp_storage,
            temp_storage_bytes,
            d_in,
            d_out,
            num_segments,
            d_begin_offsets,
            d_end_offsets,
            cub::Max(),
            identity,
            stream,
            debug_synchronous);
    }


    /**
     * \brief Finds the first device-wide maximum in each segment using the greater-than ('>') operator, also returning the in-segment index of that item
     *
     * \par
     * - Assuming the input \p d_in has value type \p T, the output \p d_out must have value type
     *   <tt>KeyValuePair<int, T></tt>.  The maximum value is written to <tt>d_out.value</tt> and its
     *   location in the input array is written to <tt>d_out.key</tt>.
     * - The specified \p identity value is produced for zero-length segments.  \p identity is
     *   assumed to be arithmetically neutral for the input type's \p > operation, i.e.,
     *   <tt>(identity > XXX) = false</tt>
     * - When input a contiguous sequence of segments, a single sequence
     *   \p segment_offsets (of length <tt>num_segments+1</tt>) can be aliased
     *   for both the \p d_begin_offsets and \p d_end_offsets parameters (where
     *   the latter is specified as <tt>segment_offsets+1</tt>).
     * - Does not support non-commutative maximum operators.
     * - \devicestorage
     *
     * \par Snippet
     * The code snippet below illustrates the argmax-reduction of a device vector of \p int items.
     * \par
     * \code
     * #include <cub/cub.cuh>   // or equivalently <cub/device/device_reduce.cuh>
     *
     * // Declare, allocate, and initialize device-accessible pointers for input and output
     * int                      num_segments;   // e.g., 3
     * int                      *d_offsets;     // e.g., [0, 3, 3, 7]
     * int                      *d_in;          // e.g., [8, 6, 7, 5, 3, 0, 9]
     * KeyValuePair<int, int>   *d_out;         // e.g., [{-,-}, {-,-}, {-,-}]
     * int                      identity;       // e.g., MIN_INT
     * ...
     *
     * // Determine temporary device storage requirements
     * void     *d_temp_storage = NULL;
     * size_t   temp_storage_bytes = 0;
     * cub::DeviceSegmentedReduce::ArgMax(d_temp_storage, temp_storage_bytes, d_in, d_out,
     *     num_segments, d_offsets, d_offsets + 1, identity);
     *
     * // Allocate temporary storage
     * cudaMalloc(&d_temp_storage, temp_storage_bytes);
     *
     * // Run argmax-reduction
     * cub::DeviceSegmentedReduce::ArgMax(d_temp_storage, temp_storage_bytes, d_in, d_out,
     *     num_segments, d_offsets, d_offsets + 1, identity);
     *
     * // d_out <-- [{0,8}, {-1,MIN_INT}, {3,9}]
     *
     * \endcode
     *
     * \tparam InputIteratorT     <b>[inferred]</b> Random-access input iterator type for reading input items (of some type \p T) \iterator
     * \tparam OutputIteratorT    <b>[inferred]</b> Output iterator type for recording the reduced aggregate (having value type <tt>KeyValuePair<int, T></tt>) \iterator
     * \tparam T                  <b>[inferred]</b> Data type being reduced
     */
    template <
        typename            InputIteratorT,
        typename            OutputIteratorT,
        typename            T>
    CUB_RUNTIME_FUNCTION
    static cudaError_t ArgMax(
        void                *d_temp_storage,                    ///< [in] %Device-accessible allocation of temporary storage.  When NULL, the required allocation size is written to \p temp_storage_bytes and no work is done.
        size_t              &temp_storage_bytes,                ///< [in,out] Reference to size in bytes of \p d_temp_storage allocation
        InputIteratorT      d_in,                               ///< [in] Pointer to the input sequence of data items
        OutputIteratorT     d_out,                              ///< [out] Pointer to the output aggregate
        int                 num_segments,                       ///< [in] The number of segments that comprise the sorting data
        int                 *d_begin_offsets,                   ///< [in] %Device-accessible pointer to the sequence of beginning offsets of length \p num_segments, such that <tt>d_begin_offsets[i]</tt> is the first element of the <em>i</em><sup>th</sup> data segment in <tt>d_keys_*</tt> and <tt>d_values_*</tt>
        int                 *d_end_offsets,                     ///< [in] %Device-accessible pointer to the sequence of ending offsets of length \p num_segments, such that <tt>d_end_offsets[i]-1</tt> is the last element of the <em>i</em><sup>th</sup> data segment in <tt>d_keys_*</tt> and <tt>d_values_*</tt>.  If <tt>d_end_offsets[i]-1</tt> <= <tt>d_begin_offsets[i]</tt>, the <em>i</em><sup>th</sup> is considered empty.
        T                   identity,                           ///< [in] The identity value to be returned for zero-length segments
        cudaStream_t        stream              = 0,            ///< [in] <b>[optional]</b> CUDA stream to launch kernels within.  Default is stream<sub>0</sub>.
        bool                debug_synchronous   = false)        ///< [in] <b>[optional]</b> Whether or not to synchronize the stream after every kernel launch to check for errors.  Also causes launch configurations to be printed to the console.  Default is \p false.
    {
        // Signed integer type for global offsets
        typedef int OffsetT;

        // Wrapped input iterator
        typedef ArgIndexInputIterator<InputIteratorT, OffsetT> ArgIndexInputIteratorT;
        ArgIndexInputIteratorT d_argmax_in(d_in, 0);

        KeyValuePair<OffsetT, T> identity_pair = {-1, identity};

        return DispatchSegmentedReduce<ArgIndexInputIteratorT, OutputIteratorT, OffsetT, cub::ArgMax>::Dispatch(
            d_temp_storage,
            temp_storage_bytes,
            d_argmax_in,
            d_out,
            num_segments,
            d_begin_offsets,
            d_end_offsets,
            cub::ArgMax(),
            identity_pair,
            stream,
            debug_synchronous);
    }

};

}               // CUB namespace
CUB_NS_POSTFIX  // Optional outer namespace(s)


