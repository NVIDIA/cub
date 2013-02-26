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

/******************************************************************************
 * Scan over thread-local array types
 ******************************************************************************/

#pragma once

#include "../operators.cuh"
#include "../ptx_intrinsics.cuh"
#include "../type_utils.cuh"
#include "../ns_wrapper.cuh"

CUB_NS_PREFIX
namespace cub {


/******************************************************************************
 * Exclusive prefix scan
 ******************************************************************************/

/**
 * Exclusive prefix scan across a thread-local array with seed prefix.
 * Returns the aggregate.
 */
template <
    int         LENGTH,                    /// Length of input/output arrays
    typename     T,                        /// (inferred) Input/output type
    typename     ScanOp>                    /// (inferred) Binary scan operator type (parameters of type T)
__device__ __forceinline__ T ThreadScanExclusive(
    T            *input,                    /// (in) Input array
    T            *output,                /// (out) Output array (may be aliased to input)
    ScanOp        scan_op,                /// (in) Scan operator
    T             prefix,                    /// (in) Prefix to seed scan with
    bool         apply_prefix = true)    /// (in) Whether or not the calling thread should apply its prefix.  If not, the first output element is undefined.  (Handy for preventing thread-0 from applying a prefix.)
{
    T inclusive = input[0];
    if (apply_prefix)
    {
        inclusive = scan_op(prefix, inclusive);
    }
    output[0] = prefix;
    T exclusive = inclusive;

    #pragma unroll
    for (int i = 1; i < LENGTH; ++i)
    {
        inclusive = scan_op(exclusive, input[i]);
        output[i] = exclusive;
        exclusive = inclusive;
    }

    return inclusive;
}


/**
 * Exclusive prefix scan across a thread-local array with seed prefix.
 * Returns the aggregate.
 */
template <
    int         LENGTH,                    /// (inferred) Length of input/output arrays
    typename     T,                        /// (inferred) Input/output type
    typename     ScanOp>                    /// (inferred) Binary scan operator type (parameters of type T)
__device__ __forceinline__ T ThreadScanExclusive(
    T            (&input)[LENGTH],        /// (in) Input array
    T            (&output)[LENGTH],        /// (out) Output array (may be aliased to input)
    ScanOp        scan_op,                /// (in) Scan operator
    T             prefix,                    /// (in) Prefix to seed scan with
    bool         apply_prefix = true)    /// (in) Whether or not the calling thread should apply its prefix.  (Handy for preventing thread-0 from applying a prefix.)
{
    return ThreadScanExclusive<LENGTH>((T*) input, (T*) output, scan_op, prefix);
}



/******************************************************************************
 * Inclusive prefix scan
 ******************************************************************************/

/**
 * Inclusive prefix scan across a thread-local array. Returns the aggregate.
 */
template <
    int         LENGTH,                    /// Length of input/output arrays
    typename     T,                        /// (inferred) Input/output type
    typename     ScanOp>                    /// (inferred) Scan operator type (functor)
__device__ __forceinline__ T ThreadScanInclusive(
    T            *input,                    /// (in) Input array
    T            *output,                /// (out) Output array (may be aliased to input)
    ScanOp         scan_op)                /// (in) Scan operator
{
    T inclusive = input[0];
    output[0] = inclusive;

    // Continue scan
    #pragma unroll
    for (int i = 0; i < LENGTH; ++i)
    {
        inclusive = scan_op(inclusive, input[i]);
        output[i] = inclusive;
    }

    return inclusive;
}


/**
 * Inclusive prefix scan across a thread-local array. Returns the aggregate.
 */
template <
    int         LENGTH,                    /// (inferred) Length of input/output arrays
    typename     T,                        /// (inferred) Input/output type
    typename     ScanOp>                    /// (inferred) Scan operator type (functor)
__device__ __forceinline__ T ThreadScanInclusive(
    T            (&input)[LENGTH],        /// (in) Input array
    T            (&output)[LENGTH],        /// (out) Output array (may be aliased to input)
    ScanOp         scan_op)                /// (in) Scan operator
{
    return ThreadScanInclusive<LENGTH>((T*) input, (T*) output, scan_op);
}


/**
 * Inclusive prefix scan across a thread-local array with seed prefix.
 * Returns the aggregate.
 */
template <
    int         LENGTH,                    /// Length of input/output arrays
    typename     T,                        /// (inferred) Input/output type
    typename     ScanOp>                    /// (inferred) Scan operator type (functor)
__device__ __forceinline__ T ThreadScanInclusive(
    T            *input,                    /// (in) Input array
    T            *output,                /// (out) Output array (may be aliased to input)
    ScanOp         scan_op,                /// (in) Scan operator
    T             prefix,                    /// (in) Prefix to seed scan with
    bool         apply_prefix = true)    /// (in) Whether or not the calling thread should apply its prefix.  (Handy for preventing thread-0 from applying a prefix.)
{
    T inclusive = input[0];
    if (apply_prefix)
    {
        inclusive = scan_op(prefix, inclusive);
    }
    output[0] = inclusive;

    // Continue scan
    #pragma unroll
    for (int i = 1; i < LENGTH; ++i)
    {
        inclusive = scan_op(inclusive, input[i]);
        output[i] = inclusive;
    }

    return inclusive;
}


/**
 * Inclusive prefix scan across a thread-local array with seed prefix.
 * Returns the aggregate.
 */
template <
    int         LENGTH,                    /// (inferred) Length of input/output arrays
    typename     T,                        /// (inferred) Input/output type
    typename     ScanOp>                    /// (inferred) Scan operator type (functor)
__device__ __forceinline__ T ThreadScanInclusive(
    T            (&input)[LENGTH],        /// (in) Input array
    T            (&output)[LENGTH],        /// (out) Output array (may be aliased to input)
    ScanOp         scan_op,                /// (in) Scan operator
    T             prefix,                    /// (in) Prefix to seed scan with
    bool         apply_prefix = true)    /// (in) Whether or not the calling thread should apply its prefix.  (Handy for preventing thread-0 from applying a prefix.)
{
    return ThreadScanInclusive<LENGTH>((T*) input, (T*) output, scan_op, prefix, apply_prefix);
}





} // namespace cub
CUB_NS_POSTFIX

