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
 * Derive CUDA vector-types for primitive types
 ******************************************************************************/

#pragma once

#include "ns_wrapper.cuh"
#include "thread/thread_load.cuh"
#include "thread/thread_store.cuh"

CUB_NS_PREFIX
namespace cub {


/******************************************************************************
 * Derive CUDA vector-types for primitive types
 *
 * For example:
 *
 *     typename VectorType<unsigned int, 2>::Type    // Aliases uint2
 *
 ******************************************************************************/

enum {
    MAX_VEC_ELEMENTS = 4,    // The maximum number of elements in CUDA vector types
};

/**
 * Vector type
 */
template <typename T, int vec_elements> struct VectorType;

/**
 * Generic vector-1 type
 */
template <typename T>
struct VectorType<T, 1>
{
    T x;

    typedef VectorType<T, 1> Type;
    typedef void ThreadLoadTag;
    typedef void ThreadStoreTag;

    // ThreadLoad
    template <PtxLoadModifier MODIFIER>
    __device__ __forceinline__     void ThreadLoad(VectorType *ptr)
    {
        x = cub::ThreadLoad<MODIFIER>(&(ptr->x));
    }

     // ThreadStore
    template <PtxStoreModifier MODIFIER>
    __device__ __forceinline__ void ThreadStore(VectorType *ptr) const
    {
        cub::ThreadStore<MODIFIER>(&(ptr->x), x);
    }
};

/**
 * Generic vector-2 type
 */
template <typename T>
struct VectorType<T, 2>
{
    T x;
    T y;

    typedef VectorType<T, 2> Type;
    typedef void ThreadLoadTag;
    typedef void ThreadStoreTag;

    // ThreadLoad
    template <PtxLoadModifier MODIFIER>
    __device__ __forceinline__ void ThreadLoad(VectorType *ptr)
    {
        x = cub::ThreadLoad<MODIFIER>(&(ptr->x));
        y = cub::ThreadLoad<MODIFIER>(&(ptr->y));
    }

     // ThreadStore
    template <PtxStoreModifier MODIFIER>
    __device__ __forceinline__ void ThreadStore(VectorType *ptr) const
    {
        cub::ThreadStore<MODIFIER>(&(ptr->x), x);
        cub::ThreadStore<MODIFIER>(&(ptr->y), y);
    }
};

/**
 * Generic vector-3 type
 */
template <typename T>
struct VectorType<T, 3>
{
    T x;
    T y;
    T z;

    typedef VectorType<T, 3> Type;
    typedef void ThreadLoadTag;
    typedef void ThreadStoreTag;

    // ThreadLoad
    template <PtxLoadModifier MODIFIER>
    __device__ __forceinline__ void ThreadLoad(VectorType *ptr)
    {
        x = cub::ThreadLoad<MODIFIER>(&(ptr->x));
        y = cub::ThreadLoad<MODIFIER>(&(ptr->y));
        z = cub::ThreadLoad<MODIFIER>(&(ptr->z));
    }

     // ThreadStore
    template <PtxStoreModifier MODIFIER>
    __device__ __forceinline__ void ThreadStore(VectorType *ptr) const
    {
        cub::ThreadStore<MODIFIER>(&(ptr->x), x);
        cub::ThreadStore<MODIFIER>(&(ptr->y), y);
        cub::ThreadStore<MODIFIER>(&(ptr->z), z);
    }

};

/**
 * Generic vector-4 type
 */
template <typename T>
struct VectorType<T, 4>
{
    T x;
    T y;
    T z;
    T w;

    typedef VectorType<T, 4> Type;
    typedef void ThreadLoadTag;
    typedef void ThreadStoreTag;

    // ThreadLoad
    template <PtxLoadModifier MODIFIER>
    __device__ __forceinline__ void ThreadLoad(VectorType *ptr)
    {
        x = cub::ThreadLoad<MODIFIER>(&(ptr->x));
        y = cub::ThreadLoad<MODIFIER>(&(ptr->y));
        z = cub::ThreadLoad<MODIFIER>(&(ptr->z));
        w = cub::ThreadLoad<MODIFIER>(&(ptr->w));
    }

     // ThreadStore
    template <PtxStoreModifier MODIFIER>
    __device__ __forceinline__ void ThreadStore(VectorType *ptr) const
    {
        cub::ThreadStore<MODIFIER>(&(ptr->x), x);
        cub::ThreadStore<MODIFIER>(&(ptr->y), y);
        cub::ThreadStore<MODIFIER>(&(ptr->z), z);
        cub::ThreadStore<MODIFIER>(&(ptr->w), w);
    }
};

/**
 * Macro for expanding partially-specialized built-in vector types
 */
#define CUB_DEFINE_VECTOR_TYPE(base_type,short_type)                                  \
  template<> struct VectorType<base_type, 1> { typedef short_type##1 Type; };        \
  template<> struct VectorType<base_type, 2> { typedef short_type##2 Type; };        \
  template<> struct VectorType<base_type, 3> { typedef short_type##3 Type; };        \
  template<> struct VectorType<base_type, 4> { typedef short_type##4 Type; };

// Expand CUDA vector types for built-in primitives
CUB_DEFINE_VECTOR_TYPE(char,               char)
CUB_DEFINE_VECTOR_TYPE(signed char,        char)
CUB_DEFINE_VECTOR_TYPE(short,              short)
CUB_DEFINE_VECTOR_TYPE(int,                int)
CUB_DEFINE_VECTOR_TYPE(long,               long)
CUB_DEFINE_VECTOR_TYPE(long long,          longlong)
CUB_DEFINE_VECTOR_TYPE(unsigned char,      uchar)
CUB_DEFINE_VECTOR_TYPE(unsigned short,     ushort)
CUB_DEFINE_VECTOR_TYPE(unsigned int,       uint)
CUB_DEFINE_VECTOR_TYPE(unsigned long,      ulong)
CUB_DEFINE_VECTOR_TYPE(unsigned long long, ulonglong)
CUB_DEFINE_VECTOR_TYPE(float,              float)
CUB_DEFINE_VECTOR_TYPE(double,             double)
CUB_DEFINE_VECTOR_TYPE(bool,               uchar)

// Undefine macros
#undef CUB_DEFINE_VECTOR_TYPE


} // namespace cub
CUB_NS_POSTFIX
