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
 * Reduction over thread-local array types.
 *
 * For example:
 *
 *    Sum<T> op;
 *
 *     int a[4] = {1, 2, 3, 4};
 *     ThreadReduce(a, op));                        // 10
 *
 *  int b[2][2] = {{1, 2}, {3, 4}};
 *     ThreadReduce(b, op));                        // 10
 *
 *     int *c = &a[1];
 *     ThreadReduce(c, op));                        // 2
 *     ThreadReduce<2>(c, op));                    // 5
 *
 *     int (*d)[2] = &b[1];
 *     ThreadReduce(d, op));                        // 7
 *
 ******************************************************************************/

#pragma once

#include "../operators.cuh"
#include "../type_utils.cuh"
#include "../ns_wrapper.cuh"

CUB_NS_PREFIX
namespace cub {

/**
 * Serial reduction with the specified operator and seed
 */
template <
    int LENGTH,
    typename T,
    typename ReductionOp>
__device__ __forceinline__ T ThreadReduce(
    T* data,
    ReductionOp reduction_op,
    T seed)
{
    #pragma unroll
    for (int i = 0; i < LENGTH; ++i)
    {
        seed = reduction_op(seed, data[i]);
    }

    return seed;
}


/**
 * Serial reduction with the specified operator
 */
template <
    int LENGTH,
    typename T,
    typename ReductionOp>
__device__ __forceinline__ T ThreadReduce(
    T* data,
    ReductionOp reduction_op)
{
    T seed = data[0];
    return ThreadReduce<LENGTH - 1>(data + 1, reduction_op, seed);
}


/**
 * Serial reduction with the specified operator and seed
 */
template <
    typename ArrayType,
    typename ReductionOp>
__device__ __forceinline__ typename ArrayTraits<ArrayType>::Type ThreadReduce(
    ArrayType &data,
    ReductionOp reduction_op,
    typename ArrayTraits<ArrayType>::Type seed)
{
    typedef typename ArrayTraits<ArrayType>::Type T;
    T* linear_array = reinterpret_cast<T*>(data);
    return ThreadReduce<ArrayTraits<ArrayType>::ELEMENTS>(linear_array, reduction_op, seed);
}


/**
 * Serial reduction with the specified operator
 */
template <
    typename ArrayType,
    typename ReductionOp>
__device__ __forceinline__ typename ArrayTraits<ArrayType>::Type ThreadReduce(
    ArrayType &data,
    ReductionOp reduction_op)
{
    typedef typename ArrayTraits<ArrayType>::Type T;
    T* linear_array = reinterpret_cast<T*>(data);
    return ThreadReduce<ArrayTraits<ArrayType>::ELEMENTS>(linear_array, reduction_op);
}


} // namespace cub
CUB_NS_POSTFIX

