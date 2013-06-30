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
 * Thread utilities for writing memory using PTX cache modifiers.
 */

#pragma once

#include <cuda.h>

#include "../util_ptx.cuh"
#include "../util_type.cuh"
#include "../util_namespace.cuh"

/// Optional outer namespace(s)
CUB_NS_PREFIX

/// CUB namespace
namespace cub {

/**
 * \addtogroup IoModule
 * @{
 */


//-----------------------------------------------------------------------------
// Tags and constants
//-----------------------------------------------------------------------------

/**
 * \brief Enumeration of PTX cache-modifiers for memory store operations.
 */
enum PtxStoreModifier
{
    STORE_DEFAULT,              ///< Default (no modifier)
    STORE_WB,                   ///< Cache write-back all coherent levels
    STORE_CG,                   ///< Cache at global level
    STORE_CS,                   ///< Cache streaming (likely to be accessed once)
    STORE_WT,                   ///< Cache write-through (to system memory)
    STORE_VOLATILE,             ///< Volatile shared
};


#ifndef DOXYGEN_SHOULD_SKIP_THIS    // Do not document

/**
 * Helper structure for giving type errors when combining non-default store
 * modifiers with (a) iterators or (b) pointers to non-primitives
 */
template <typename OutputIteratorRA>
struct ThreadStoreHelper;

#endif // DOXYGEN_SHOULD_SKIP_THIS



/**
 * \name Simple I/O
 * @{
 */

/**
 * \brief Thread utility for writing memory using cub::PtxStoreModifier cache modifiers.
 *
 * Cache modifiers will only be effected for built-in types (i.e., C++
 * primitives and CUDA vector-types).
 *
 * For example:
 * \par
 * \code
 * #include <cub/cub.cuh>
 *
 * // 32-bit store using cache-global modifier:
 * int *d_out;
 * int val;
 * cub::ThreadStore<cub::STORE_CG>(d_out + threadIdx.x, val);
 *
 * // 16-bit store using default modifier
 * short *d_out;
 * short val;
 * cub::ThreadStore<cub::STORE_DEFAULT>(d_out + threadIdx.x, val);
 *
 * // 256-bit store using write-through modifier
 * double4 *d_out;
 * double4 val;
 * cub::ThreadStore<cub::STORE_WT>(d_out + threadIdx.x, val);
 *
 * // 96-bit store using default cache modifier (ignoring STORE_CS)
 * struct TestFoo { bool a; short b; };
 * TestFoo *d_struct;
 * TestFoo val;
 * cub::ThreadStore<cub::STORE_CS>(d_out + threadIdx.x, val);
 * \endcode
 *
 */
template <
    PtxStoreModifier MODIFIER,
    typename OutputIteratorRA,
    typename T>
__device__ __forceinline__ void ThreadStore(OutputIteratorRA itr, const T& val)
{
    typedef ThreadStoreHelper<OutputIteratorRA> Helper;
    return Helper::template Store<MODIFIER>(itr, val);
}


//@}  end member group


#ifndef DOXYGEN_SHOULD_SKIP_THIS    // Do not document


/// ThreadStoreHelper specialized for iterators
template <typename OutputIteratorRA>
struct ThreadStoreHelper
{
    typedef typename std::iterator_traits<OutputIteratorRA>::value_type     T;
    typedef typename WordAlignment<T>::AlignWord                            AlignWord;

    template <PtxStoreModifier MODIFIER>
    static __device__ __forceinline__ void Store(OutputIteratorRA itr, const T& val);

#if (CUB_PTX_ARCH <= 130)
    template <>
    static __device__ __forceinline__ void Store<STORE_CG>(OutputIteratorRA itr, const T& val)
    {
        // Straightforward dereference
        *itr = val;
    }
#endif

    template <>
    static __device__ __forceinline__ void Store<STORE_DEFAULT>(OutputIteratorRA itr, const T& val)
    {
        // Straightforward dereference
        *itr = val;
    }
};


/// ThreadStoreHelper specialized for native pointer types
template <typename T>
struct ThreadStoreHelper<T*>
{
    template <PtxStoreModifier MODIFIER>
    static __device__ __forceinline__ void Store(T *ptr, const T& val)
    {
        typedef typename WordAlignment<T>::AlignWord AlignWord;

        const int ALIGN_UNROLL  = sizeof(T) / sizeof(AlignWord);

        AlignWord *alias        = reinterpret_cast<AlignWord*>(const_cast<T*>(&val));
        AlignWord *alias_ptr    = reinterpret_cast<AlignWord*>(ptr);

        #pragma unroll
        for (int i = 0; i < ALIGN_UNROLL; ++i)
            ThreadStore<MODIFIER>(alias_ptr + i, alias[i]);
    }


#if (CUB_PTX_ARCH <= 130)
    template <>
    static __device__ __forceinline__ void Store<STORE_CG>(T *ptr, const T& val)
    {
        // Straightforward dereference
        *ptr = val;
    }
#endif


    template <>
    static __device__ __forceinline__ void Store<STORE_VOLATILE>(T *ptr, const T& val)
    {

#if (CUB_PTX_ARCH <= 130)
        *ptr = val;
        __threadfence_block();
#else
        typedef typename WordAlignment<T>::VolatileAlignWord VolatileAlignWord;
        const int VOLATILE_ALIGN_UNROLL = sizeof(T) / sizeof(VolatileAlignWord);

        VolatileAlignWord *alias                = reinterpret_cast<VolatileAlignWord*>(const_cast<T*>(&val));
        volatile VolatileAlignWord *alias_ptr   = reinterpret_cast<volatile VolatileAlignWord*>(ptr);

        #pragma unroll
        for (int i = 0; i < VOLATILE_ALIGN_UNROLL; ++i)
            alias_ptr[i] = alias[i];
#endif
    }

    template <>
    static __device__ __forceinline__ void Store<STORE_DEFAULT>(T *ptr, const T& val)
    {
        // Straightforward dereference
        *ptr = val;
    }
};


/**
 * Define a int4 (16B) ThreadStore specialization for the given PTX load modifier
 */
#define CUB_STORE_16(cub_modifier, ptx_modifier)                                            \
    template<>                                                                              \
    __device__ __forceinline__ void ThreadStore<cub_modifier, int4*, int4>(int4* ptr, const int4 &val)              \
    {                                                                                       \
        asm volatile ("st."#ptx_modifier".v4.s32 [%0], {%1, %2, %3, %4};" : :               \
            _CUB_ASM_PTR_(ptr),                                                             \
            "r"(val.x),                                                                     \
            "r"(val.y),                                                                     \
            "r"(val.z),                                                                     \
            "r"(val.w));                                                                    \
    }                                                                                       \


/**
 * Define a int2 (8B) ThreadStore specialization for the given PTX load modifier
 */
#define CUB_STORE_8(cub_modifier, ptx_modifier)                                             \
    template<>                                                                              \
    __device__ __forceinline__ void ThreadStore<cub_modifier, int2*, int2>(int2* ptr, const int2 &val)              \
    {                                                                                       \
        asm volatile ("st."#ptx_modifier".v2.s32 [%0], {%1, %2};" : :                       \
            _CUB_ASM_PTR_(ptr),                                                             \
            "r"(val.x),                                                                     \
            "r"(val.y));                                                                    \
    }

/**
 * Define a int (4B) ThreadStore specialization for the given PTX load modifier
 */
#define CUB_STORE_4(cub_modifier, ptx_modifier)                                             \
    template<>                                                                              \
    __device__ __forceinline__ void ThreadStore<cub_modifier, int*, int>(int* ptr, const int &val)                 \
    {                                                                                       \
        asm volatile ("st."#ptx_modifier".s32 [%0], %1;" : :                                \
            _CUB_ASM_PTR_(ptr),                                                             \
            "r"(val));                                                                      \
    }


/**
 * Define a short (2B) ThreadStore specialization for the given PTX load modifier
 */
#define CUB_STORE_2(cub_modifier, ptx_modifier)                                             \
    template<>                                                                              \
    __device__ __forceinline__ void ThreadStore<cub_modifier, short*, short>(short* ptr, const short &val)           \
    {                                                                                       \
        asm volatile ("st."#ptx_modifier".s16 [%0], %1;" : :                                \
            _CUB_ASM_PTR_(ptr),                                                             \
            "h"(val));                                                                      \
    }


/**
 * Define a char (1B) ThreadStore specialization for the given PTX load modifier
 */
#define CUB_STORE_1(cub_modifier, ptx_modifier)                                             \
    template<>                                                                              \
    __device__ __forceinline__ void ThreadStore<cub_modifier, char*, char>(char* ptr, const char &val)              \
    {                                                                                       \
        asm volatile (                                                                      \
        "{"                                                                                 \
        "   .reg .s8 datum;"                                                                \
        "   cvt.s8.s16 datum, %1;"                                                          \
        "   st."#ptx_modifier".s8 [%0], datum;"                                             \
        "}" : :                                                                             \
            _CUB_ASM_PTR_(ptr),                                                             \
            "h"(short(val)));                                                               \
    }

/**
 * Define powers-of-two ThreadStore specializations for the given PTX load modifier
 */
#define CUB_STORE_ALL(cub_modifier, ptx_modifier)                                           \
    CUB_STORE_16(cub_modifier, ptx_modifier)                                                \
    CUB_STORE_8(cub_modifier, ptx_modifier)                                                 \
    CUB_STORE_4(cub_modifier, ptx_modifier)                                                 \
    CUB_STORE_2(cub_modifier, ptx_modifier)                                                 \
    CUB_STORE_1(cub_modifier, ptx_modifier)                                                 \

/**
 * Define ThreadStore specializations for the various PTX load modifiers
 */

#if CUB_PTX_ARCH >= 200
    CUB_STORE_ALL(STORE_WB, ca)
    CUB_STORE_ALL(STORE_CG, cg)
    CUB_STORE_ALL(STORE_CS, cs)
    CUB_STORE_ALL(STORE_WT, cv)
#endif

//CUB_STORE_ALL(STORE_VOLATILE, volatile)


#endif // DOXYGEN_SHOULD_SKIP_THIS


/** @} */       // end group IoModule


}               // CUB namespace
CUB_NS_POSTFIX  // Optional outer namespace(s)
