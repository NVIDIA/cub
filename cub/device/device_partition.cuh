
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
 * cub::DevicePartition provides device-wide, parallel operations for partitioning sequences of data items residing within global memory.
 */

#pragma once

#include <stdio.h>
#include <iterator>

#include "device_select.cuh"
#include "../util_namespace.cuh"

/// Optional outer namespace(s)
CUB_NS_PREFIX

/// CUB namespace
namespace cub {


/******************************************************************************
 * DevicePartition
 *****************************************************************************/

/**
 * \brief DevicePartition provides device-wide, parallel operations for partitioning sequences of data items residing within global memory. ![](partition_logo.png)
 * \ingroup DeviceModule
 *
 * \par Overview
 * These operations apply a selection criterion to construct a partitioned output sequence from items selected/unselected from
 * a specified input sequence.
 *
 * \par Usage Considerations
 * \cdp_class{DevicePartition}
 *
 * \par Performance
 *
 */
struct DevicePartition
{
    /**
     * \brief Uses the \p d_flags sequence to split the corresponding items from \p d_in into a partitioned sequence \p d_out.  The total number of items copied into the first partition is written to \p d_num_selected. ![](partition_flags_logo.png)
     *
     * \par
     * - The value type of \p d_flags must be castable to \p bool (e.g., \p bool, \p char, \p int, etc.).
     * - Copies of the selected items are compacted into \p d_out and maintain their original
     *   relative ordering, however copies of the unselected items are compacted into the
     *   rear of \p d_out in reverse order.
     * - \devicestorage
     * - \cdp
     *
     * \par
     * The code snippet below illustrates the compaction of items selected from an \p int device vector.
     * \par
     * \code
     * #include <cub/cub.cuh>   // or equivalently <cub/device/device_partition.cuh>
     *
     * // Declare, allocate, and initialize device pointers for input, flags, and output
     * int  num_items;          // e.g., 8
     * int  *d_in;              // e.g., [1, 2, 3, 4, 5, 6, 7, 8]
     * char *d_flags;           // e.g., [1, 0, 0, 1, 0, 1, 1, 0]
     * int  *d_out;             // e.g., [ ,  ,  ,  ,  ,  ,  ,  ]
     * int  *d_num_selected;    // e.g., [ ]
     * ...
     *
     * // Determine temporary device storage requirements
     * void     *d_temp_storage = NULL;
     * size_t   temp_storage_bytes = 0;
     * cub::DevicePartition::Flagged(d_temp_storage, temp_storage_bytes, d_in, d_flags, d_out, d_num_selected, num_items);
     *
     * // Allocate temporary storage
     * cudaMalloc(&d_temp_storage, temp_storage_bytes);
     *
     * // Run selection
     * cub::DevicePartition::Flagged(d_temp_storage, temp_storage_bytes, d_in, d_flags, d_out, d_num_selected, num_items);
     *
     * // d_out             <-- [1, 4, 6, 7, 8, 5, 3, 2]
     * // d_num_selected    <-- [4]
     *
     * \endcode
     *
     * \tparam InputIterator        <b>[inferred]</b> Random-access input iterator type for selection items \iterator
     * \tparam FlagIterator         <b>[inferred]</b> Random-access input iterator type for selection flags \iterator
     * \tparam OutputIterator       <b>[inferred]</b> Random-access output iterator type for selected items \iterator
     * \tparam NumSelectedIterator  <b>[inferred]</b> Output iterator type for recording number of items selected \iterator
     */
    template <
        typename                    InputIterator,
        typename                    FlagIterator,
        typename                    OutputIterator,
        typename                    NumSelectedIterator>
    __host__ __device__ __forceinline__
    static cudaError_t Flagged(
        void                        *d_temp_storage,                ///< [in] %Device allocation of temporary storage.  When NULL, the required allocation size is returned in \p temp_storage_bytes and no work is done.
        size_t                      &temp_storage_bytes,            ///< [in,out] Size in bytes of \p d_temp_storage allocation
        InputIterator               d_in,                           ///< [in] Input iterator pointing to data items
        FlagIterator                d_flags,                        ///< [in] Input iterator pointing to selection flags
        OutputIterator              d_out,                          ///< [in] Output iterator pointing to selected items
        NumSelectedIterator         d_num_selected,                 ///< [in] Output iterator pointing to total number selected
        int                         num_items,                      ///< [in] Total number of items to select from
        cudaStream_t                stream             = 0,         ///< [in] <b>[optional]</b> CUDA stream to launch kernels within.  Default is stream<sub>0</sub>.
        bool                        debug_synchronous  = false)     ///< [in] <b>[optional]</b> Whether or not to synchronize the stream after every kernel launch to check for errors.  May cause significant slowdown.  Default is \p false.
    {
        typedef int                     Offset;         // Signed integer type for global offsets
        typedef NullType                SelectOp;       // Selection op (not used)
        typedef NullType                EqualityOp;     // Equality operator (not used)

        return DeviceSelectDispatch<InputIterator, FlagIterator, OutputIterator, NumSelectedIterator, SelectOp, EqualityOp, Offset, true>::Dispatch(
            d_temp_storage,
            temp_storage_bytes,
            d_in,
            d_flags,
            d_out,
            d_num_selected,
            SelectOp(),
            EqualityOp(),
            num_items,
            stream,
            debug_synchronous);
    }


    /**
     * \brief Uses the \p select_op functor to split the corresponding items from \p d_in into a partitioned sequence \p d_out.  The total number of items copied into the first partition is written to \p d_num_selected. ![](partition_logo.png)
     *
     * \par
     * - Copies of the selected items are compacted into \p d_out and maintain their original
     *   relative ordering, however copies of the unselected items are compacted into the
     *   rear of \p d_out in reverse order.
     * - \devicestorage
     * - \cdp
     *
     * \par
     * The code snippet below illustrates the compaction of items selected from an \p int device vector.
     * \par
     * \code
     * #include <cub/cub.cuh>   // or equivalently <cub/device/device_partition.cuh>
     *
     * // Functor for selecting values that are multiples of three
     * struct IsTriple
     * {
     *     template <typename T>
     *     __host__ __device__ __forceinline__
     *     bool operator()(const T &a) const {
     *         return (a % 3 == 0);
     *     }
     * };
     *
     * // Declare, allocate, and initialize device pointers for input and output
     * int      num_items;          // e.g., 8
     * int      *d_in;              // e.g., [0, 2, 3, 9, 5, 2, 81, 8]
     * int      *d_out;             // e.g., [ ,  ,  ,  ,  ,  ,  ,  ]
     * int      *d_num_selected;    // e.g., [ ]
     * IsTriple select_op;
     * ...
     *
     * // Determine temporary device storage requirements
     * void     *d_temp_storage = NULL;
     * size_t   temp_storage_bytes = 0;
     * cub::DeviceSelect::If(d_temp_storage, temp_storage_bytes, d_in, d_out, d_num_selected, num_items, select_op);
     *
     * // Allocate temporary storage
     * cudaMalloc(&d_temp_storage, temp_storage_bytes);
     *
     * // Run selection
     * cub::DeviceSelect::If(d_temp_storage, temp_storage_bytes, d_in, d_out, d_num_selected, num_items, select_op);
     *
     * // d_out             <-- [0, 3, 9, 81, 8, 2, 5, 2]
     * // d_num_selected    <-- [4]
     *
     * \endcode
     *
     * \tparam InputIterator        <b>[inferred]</b> Random-access input iterator type for selection items \iterator
     * \tparam OutputIterator       <b>[inferred]</b> Random-access output iterator type for selected items \iterator
     * \tparam NumSelectedIterator  <b>[inferred]</b> Output iterator type for recording number of items selected \iterator
     * \tparam SelectOp             <b>[inferred]</b> Selection operator type having member <tt>bool operator()(const T &a)</tt>
     */
    template <
        typename                    InputIterator,
        typename                    OutputIterator,
        typename                    NumSelectedIterator,
        typename                    SelectOp>
    __host__ __device__ __forceinline__
    static cudaError_t If(
        void                        *d_temp_storage,                ///< [in] %Device allocation of temporary storage.  When NULL, the required allocation size is returned in \p temp_storage_bytes and no work is done.
        size_t                      &temp_storage_bytes,            ///< [in,out] Size in bytes of \p d_temp_storage allocation
        InputIterator               d_in,                           ///< [in] Input iterator pointing to data items
        OutputIterator              d_out,                          ///< [in] Output iterator pointing to selected items
        NumSelectedIterator         d_num_selected,                 ///< [in] Output iterator pointing to total number selected
        int                         num_items,                      ///< [in] Total number of items to select from
        SelectOp                    select_op,                      ///< [in] Unary selection operator
        cudaStream_t                stream             = 0,         ///< [in] <b>[optional]</b> CUDA stream to launch kernels within.  Default is stream<sub>0</sub>.
        bool                        debug_synchronous  = false)     ///< [in] <b>[optional]</b> Whether or not to synchronize the stream after every kernel launch to check for errors.  May cause significant slowdown.  Default is \p false.
    {
        typedef int                     Offset;         // Signed integer type for global offsets
        typedef NullType*               FlagIterator;   // Flag iterator type (not used)
        typedef NullType                EqualityOp;     // Equality operator (not used)

        return DeviceSelectDispatch<InputIterator, FlagIterator, OutputIterator, NumSelectedIterator, SelectOp, EqualityOp, Offset, true>::Dispatch(
            d_temp_storage,
            temp_storage_bytes,
            d_in,
            NULL,
            d_out,
            d_num_selected,
            select_op,
            EqualityOp(),
            num_items,
            stream,
            debug_synchronous);
    }

};

/**
 * \example example_device_partition_flagged.cu
 * \example example_device_partition_if.cu
 */

}               // CUB namespace
CUB_NS_POSTFIX  // Optional outer namespace(s)


