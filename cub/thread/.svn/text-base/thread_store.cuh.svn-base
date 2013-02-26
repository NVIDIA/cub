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
 * Thread utilities for writing memory (optionally using cache modifiers).
 * Cache modifiers will only be effected for built-in types (i.e., C++
 * primitives and CUDA vector-types).
 *
 * For example:
 *
 *     // 32-bit store using cache-global modifier:
 *
 *             int *d_out;
 *             int val;
 *             ThreadStore<PTX_STORE_CG>(d_out + threadIdx.x, val);
 *
 *
 *     // 16-bit store using default modifier
 *
 *             short *d_out;
 *             short val;
 *             ThreadStore<PTX_STORE_NONE>(d_out + threadIdx.x, val);
 *
 *
 *     // 256-bit store using write-through modifier
 *
 *             double4 *d_out;
 *             double4 val;
 *             ThreadStore<PTX_STORE_WT>(d_out + threadIdx.x, val);
 *
 *
 *     // 96-bit store using default cache modifier (ignoring PTX_STORE_CS)
 *
 *             struct TestFoo { bool a; short b; };
 *             TestFoo *d_struct;
 *             TestFoo val;
 *             ThreadStore<PTX_STORE_CS>(d_out + threadIdx.x, val);
 *
 *
 ******************************************************************************/

#pragma once

#include <cuda.h>

#include "../ptx_intrinsics.cuh"
#include "../type_utils.cuh"
#include "../ns_wrapper.cuh"

CUB_NS_PREFIX
namespace cub {


//-----------------------------------------------------------------------------
// Tags and constants
//-----------------------------------------------------------------------------

/**
 * Enumeration of store modifiers.
 */
enum PtxStoreModifier
{
    // Global store modifiers
    PTX_STORE_NONE,                 // Default (no modifier)
    PTX_STORE_WB,                   // Cache write-back all coherent levels
    PTX_STORE_CG,                   // Cache at global level
    PTX_STORE_CS,                   // Cache streaming (likely to be accessed once)
    PTX_STORE_WT,                   // Cache write-through (to system memory)

    // Shared store modifiers
    PTX_STORE_VS,                   // Volatile shared
};


//-----------------------------------------------------------------------------
// Generic ThreadStore() operation
//-----------------------------------------------------------------------------

/**
 * Define HasThreadStore structure for testing the presence of nested
 * ThreadStoreTag type names within data types
 */
CUB_HAS_NESTED_TYPE(HasThreadStore, ThreadStoreTag)


/**
 * Dispatch specializer
 */
template <PtxStoreModifier MODIFIER, bool HAS_THREAD_STORE>
struct ThreadStoreDispatch;


/**
 * Dispatch ThreadStore() to value if it exposes a ThreadStoreTag typedef
 */
template <PtxStoreModifier MODIFIER>
struct ThreadStoreDispatch<MODIFIER, true>
{
    // Pointer
    template <typename T>
    static __device__ __forceinline__ void ThreadStore(T *ptr, const T& val)
    {
        val.ThreadStore<MODIFIER>(ptr);
    }

    // Iterator
    template <typename OutputIterator, typename T>
    static __device__ __forceinline__ void ThreadStore(OutputIterator itr, const T& val)
    {
        val.ThreadStore<MODIFIER>(itr);
    }
};


/**
 * Generic PTX_STORE_NONE specialization
 */
template <>
struct ThreadStoreDispatch<PTX_STORE_NONE, false>
{
    // Pointer
    template <typename T>
    static __device__ __forceinline__ void ThreadStore(T *ptr, const T& val)
    {
        // Straightforward dereference
        *ptr = val;
    }

    // Iterator
    template <
        typename OutputIterator,
        typename T>
    static __device__ __forceinline__ void ThreadStore(OutputIterator itr, const T& val)
    {
        // Straightforward dereference
        *itr = val;
    }
};


/**
 * Generic PTX_STORE_VS specialization
 */
template <>
struct ThreadStoreDispatch<PTX_STORE_VS, false>
{
    // Pointer
    template <typename T>
    static __device__ __forceinline__ void ThreadStore(T *ptr, const T& val)
    {
        const bool USE_VOLATILE = NumericTraits<T>::PRIMITIVE;

        typedef typename If<USE_VOLATILE, volatile T, T>::Type PtrT;

        // Straightforward dereference of pointer
        *reinterpret_cast<PtrT*>(ptr) = val;

        // Prevent compiler from reordering or omitting memory accesses between rounds
        if (!USE_VOLATILE) __threadfence_block();
    }
};


/**
 * Generic ThreadStore() operation for output iterators
 */
template <
    PtxStoreModifier MODIFIER,
    typename OutputIterator,
    typename T>
__device__ __forceinline__ void ThreadStore(OutputIterator itr, const T& val)
{
    ThreadStoreDispatch<MODIFIER, HasThreadLoad<T>::VALUE>::ThreadStore(itr, val);
}


/**
 * Generic ThreadStore() operation.  Further specialized below.
 */
template <
    PtxStoreModifier MODIFIER,
    typename T>
__device__ __forceinline__ void ThreadStore(T *ptr, const T& val)
{
    ThreadStoreDispatch<MODIFIER, HasThreadLoad<T>::VALUE>::ThreadStore(ptr, val);
}


//-----------------------------------------------------------------------------
// ThreadStore() specializations by modifier and data type (i.e., primitives
// and CUDA vector types)
//-----------------------------------------------------------------------------


/**
 * Define a global ThreadStore() specialization for type
 */
#define CUB_G_STORE_0(type, asm_type, ptx_type, reg_mod, cub_modifier, ptx_modifier)    \
    template<>                                                                            \
    void ThreadStore<cub_modifier, type>(type* ptr, const type& val)                    \
    {                                                                                    \
        const asm_type raw = reinterpret_cast<const asm_type&>(val);                    \
        asm volatile ("st.global."#ptx_modifier"."#ptx_type" [%0], %1;" : :                        \
            _CUB_ASM_PTR_(ptr),                                                            \
            #reg_mod(raw));                                                             \
    }

/**
 * Define a global ThreadStore() specialization for the vector-1 type
 */
#define CUB_G_STORE_1(type, component_type, asm_type, ptx_type, reg_mod, cub_modifier, ptx_modifier)    \
    template<>                                                                            \
    void ThreadStore<cub_modifier, type>(type* ptr, const type& val)                    \
    {                                                                                    \
        const asm_type raw_x = reinterpret_cast<const asm_type&>(val.x);                \
        asm volatile ("st.global."#ptx_modifier"."#ptx_type" [%0], %1;" : :                        \
            _CUB_ASM_PTR_(ptr),                                                            \
            #reg_mod(raw_x));                                                            \
    }

/**
 * Define a volatile-shared ThreadStore() specialization for the vector-1 type
 */
#define CUB_VS_STORE_1(type, component_type, asm_type, ptx_type, reg_mod)                \
    template<>                                                                            \
    void ThreadStore<PTX_STORE_VS, type>(type* ptr, const type& val)                        \
    {                                                                                    \
        ThreadStore<PTX_STORE_VS>(                                                            \
            (asm_type*) ptr,                                                            \
            reinterpret_cast<const asm_type&>(val.x));                                    \
    }

/**
 * Define a global ThreadStore() specialization for the vector-2 type
 */
#define CUB_G_STORE_2(type, component_type, asm_type, ptx_type, reg_mod, cub_modifier, ptx_modifier)    \
    template<>                                                                            \
    void ThreadStore<cub_modifier, type>(type* ptr, const type& val)                    \
    {                                                                                    \
        const asm_type raw_x = reinterpret_cast<const asm_type&>(val.x);                \
        const asm_type raw_y = reinterpret_cast<const asm_type&>(val.y);                \
        asm volatile ("st.global."#ptx_modifier".v2."#ptx_type" [%0], {%1, %2};" : :                \
            _CUB_ASM_PTR_(ptr),                                                            \
            #reg_mod(raw_x),                                                             \
            #reg_mod(raw_y));                                                            \
    }

/**
 * Define a volatile-shared ThreadStore() specialization for the vector-2 type.
 * Performs separate references if the component_type is only 1 byte (otherwise we lose
 * performance due to the bitfield ops to disassemble the value)
 */
#define CUB_VS_STORE_2(type, component_type, asm_type, ptx_type, reg_mod)                \
    template<>                                                                            \
    void ThreadStore<PTX_STORE_VS, type>(type* ptr, const type& val)                        \
    {                                                                                    \
        if ((sizeof(component_type) == 1) || (CUDA_VERSION < 4100))                                                \
        {                                                                                \
            component_type *base_ptr = (component_type*) ptr;                            \
            ThreadStore<PTX_STORE_VS>(base_ptr, (component_type) val.x);                    \
            ThreadStore<PTX_STORE_VS>(base_ptr + 1, (component_type) val.y);                \
        }                                                                                 \
        else                                                                            \
        {                                                                                \
            const asm_type raw_x = reinterpret_cast<const asm_type&>(val.x);            \
            const asm_type raw_y = reinterpret_cast<const asm_type&>(val.y);            \
            asm volatile ("{"                                                                        \
                "    .reg ."_CUB_ASM_PTR_SIZE_" t1;"                                        \
                "    cvta.to.shared."_CUB_ASM_PTR_SIZE_" t1, %0;"                        \
                "    st.shared.volatile.v2."#ptx_type" [t1], {%1, %2};"                    \
                "}" : :                                                                    \
                _CUB_ASM_PTR_(ptr),                                                        \
                #reg_mod(raw_x),                                                         \
                #reg_mod(raw_y));                                                        \
        }                                                                                \
    }

/**
 * Define a global ThreadStore() specialization for the vector-4 type
 */
#define CUB_G_STORE_4(type, component_type, asm_type, ptx_type, reg_mod, cub_modifier, ptx_modifier)    \
    template<>                                                                            \
    void ThreadStore<cub_modifier, type>(type* ptr, const type& val)                    \
    {                                                                                    \
        const asm_type raw_x = reinterpret_cast<const asm_type&>(val.x);                \
        const asm_type raw_y = reinterpret_cast<const asm_type&>(val.y);                \
        const asm_type raw_z = reinterpret_cast<const asm_type&>(val.z);                \
        const asm_type raw_w = reinterpret_cast<const asm_type&>(val.w);                \
        asm volatile ("st.global."#ptx_modifier".v4."#ptx_type" [%0], {%1, %2, %3, %4};" : :        \
            _CUB_ASM_PTR_(ptr),                                                            \
            #reg_mod(raw_x),                                                             \
            #reg_mod(raw_y),                                                             \
            #reg_mod(raw_z),                                                             \
            #reg_mod(raw_w));                                                            \
    }

/**
 * Define a volatile-shared ThreadStore() specialization for the vector-4 type.
 * Performs separate references if the component_type is only 1 byte (otherwise we lose
 * performance due to the bitfield ops to disassemble the value)
 */
#define CUB_VS_STORE_4(type, component_type, asm_type, ptx_type, reg_mod)                \
    template<>                                                                            \
    void ThreadStore<PTX_STORE_VS, type>(type* ptr, const type& val)                        \
    {                                                                                    \
        if ((sizeof(component_type) == 1) || (CUDA_VERSION < 4100))                                                \
        {                                                                                \
            component_type *base_ptr = (component_type*) ptr;                            \
            ThreadStore<PTX_STORE_VS>(base_ptr, (component_type) val.x);                    \
            ThreadStore<PTX_STORE_VS>(base_ptr + 1, (component_type) val.y);                \
            ThreadStore<PTX_STORE_VS>(base_ptr + 2, (component_type) val.z);                \
            ThreadStore<PTX_STORE_VS>(base_ptr + 3, (component_type) val.w);                \
        }                                                                                 \
        else                                                                            \
        {                                                                                \
            const asm_type raw_x = reinterpret_cast<const asm_type&>(val.x);            \
            const asm_type raw_y = reinterpret_cast<const asm_type&>(val.y);            \
            const asm_type raw_z = reinterpret_cast<const asm_type&>(val.z);            \
            const asm_type raw_w = reinterpret_cast<const asm_type&>(val.w);            \
            asm volatile ("{"                                                                        \
                "    .reg ."_CUB_ASM_PTR_SIZE_" t1;"                                        \
                "    cvta.to.shared."_CUB_ASM_PTR_SIZE_" t1, %0;"                        \
                "    st.volatile.shared.v4."#ptx_type" [t1], {%1, %2, %3, %4};"            \
                "}" : :                                                                    \
                _CUB_ASM_PTR_(ptr),                                                        \
                #reg_mod(raw_x),                                                         \
                #reg_mod(raw_y),                                                         \
                #reg_mod(raw_z),                                                         \
                #reg_mod(raw_w));                                                        \
        }                                                                                \
    }

/**
 * Define a ThreadStore() specialization for the 64-bit vector-4 type.
 * Uses two vector-2 Stores.
 */
#define CUB_STORE_4L(type, half_type, cub_modifier)                                        \
    template<>                                                                            \
    void ThreadStore<cub_modifier, type>(type* ptr, const type& val)                    \
    {                                                                                    \
        const half_type* half_val = reinterpret_cast<const half_type*>(&val);            \
        half_type* half_ptr = reinterpret_cast<half_type*>(ptr);                        \
        ThreadStore<cub_modifier>(half_ptr, half_val[0]);                                \
        ThreadStore<cub_modifier>(half_ptr + 1, half_val[1]);                            \
    }

/**
 * Define ThreadStore() specializations for the (non-vector) type
 */
#define CUB_STORES_0(type, asm_type, ptx_type, reg_mod)                                    \
    CUB_G_STORE_0(type, asm_type, ptx_type, reg_mod, PTX_STORE_WB, wb)                        \
    CUB_G_STORE_0(type, asm_type, ptx_type, reg_mod, PTX_STORE_CG, cg)                        \
    CUB_G_STORE_0(type, asm_type, ptx_type, reg_mod, PTX_STORE_CS, cs)                        \
    CUB_G_STORE_0(type, asm_type, ptx_type, reg_mod, PTX_STORE_WT, wt)

/**
 * Define ThreadStore() specializations for the vector-1 component_type
 */
#define CUB_STORES_1(type, component_type, asm_type, ptx_type, reg_mod)                    \
    CUB_VS_STORE_1(type, component_type, asm_type, ptx_type, reg_mod)                    \
    CUB_G_STORE_1(type, component_type, asm_type, ptx_type, reg_mod, PTX_STORE_WB, wb)        \
    CUB_G_STORE_1(type, component_type, asm_type, ptx_type, reg_mod, PTX_STORE_CG, cg)        \
    CUB_G_STORE_1(type, component_type, asm_type, ptx_type, reg_mod, PTX_STORE_CS, cs)        \
    CUB_G_STORE_1(type, component_type, asm_type, ptx_type, reg_mod, PTX_STORE_WT, wt)

/**
 * Define ThreadStore() specializations for the vector-2 component_type
 */
#define CUB_STORES_2(type, component_type, asm_type, ptx_type, reg_mod)                    \
    CUB_VS_STORE_2(type, component_type, asm_type, ptx_type, reg_mod)                    \
    CUB_G_STORE_2(type, component_type, asm_type, ptx_type, reg_mod, PTX_STORE_WB, wb)        \
    CUB_G_STORE_2(type, component_type, asm_type, ptx_type, reg_mod, PTX_STORE_CG, cg)        \
    CUB_G_STORE_2(type, component_type, asm_type, ptx_type, reg_mod, PTX_STORE_CS, cs)        \
    CUB_G_STORE_2(type, component_type, asm_type, ptx_type, reg_mod, PTX_STORE_WT, wt)

/**
 * Define ThreadStore() specializations for the vector-4 component_type
 */
#define CUB_STORES_4(type, component_type, asm_type, ptx_type, reg_mod)                    \
    CUB_VS_STORE_4(type, component_type, asm_type, ptx_type, reg_mod)                    \
    CUB_G_STORE_4(type, component_type, asm_type, ptx_type, reg_mod, PTX_STORE_WB, wb)        \
    CUB_G_STORE_4(type, component_type, asm_type, ptx_type, reg_mod, PTX_STORE_CG, cg)        \
    CUB_G_STORE_4(type, component_type, asm_type, ptx_type, reg_mod, PTX_STORE_CS, cs)        \
    CUB_G_STORE_4(type, component_type, asm_type, ptx_type, reg_mod, PTX_STORE_WT, wt)

/**
 * Define ThreadStore() specializations for the 256-bit vector-4 component_type
 */
#define CUB_STORES_4L(type, half_type)                \
    CUB_STORE_4L(type, half_type, PTX_STORE_VS)            \
    CUB_STORE_4L(type, half_type, PTX_STORE_WB)            \
    CUB_STORE_4L(type, half_type, PTX_STORE_CG)            \
    CUB_STORE_4L(type, half_type, PTX_STORE_CS)            \
    CUB_STORE_4L(type, half_type, PTX_STORE_WT)

/**
 * Define vector-0/1/2 ThreadStore() specializations for the component type
 */
#define CUB_STORES_012(component_type, vec_prefix, asm_type, ptx_type, reg_mod)        \
    CUB_STORES_0(component_type, asm_type, ptx_type, reg_mod)                        \
    CUB_STORES_1(vec_prefix##1, component_type, asm_type, ptx_type, reg_mod)        \
    CUB_STORES_2(vec_prefix##2, component_type, asm_type, ptx_type, reg_mod)

/**
 * Define vector-0/1/2/4 ThreadStore() specializations for the component type
 */
#define CUB_STORES_0124(component_type, vec_prefix, asm_type, ptx_type, reg_mod)    \
    CUB_STORES_012(component_type, vec_prefix, asm_type, ptx_type, reg_mod)            \
    CUB_STORES_4(vec_prefix##4, component_type, asm_type, ptx_type, reg_mod)

/**
 * Expand ThreadStore() implementations for primitive types.
 */
// Signed
CUB_STORES_0124(char, char, short, s8, h)
CUB_STORES_0(signed char, short, s8, h)
CUB_STORES_0124(short, short, short, s16, h)
CUB_STORES_0124(int, int, int, s32, r)
CUB_STORES_012(long long, longlong, long long, u64, l)
CUB_STORES_4L(longlong4, longlong2);

// Unsigned
CUB_STORES_0124(unsigned char, uchar, unsigned short, u8, h)
CUB_STORES_0(bool, short, u8, h)
CUB_STORES_0124(unsigned short, ushort, unsigned short, u16, h)
CUB_STORES_0124(unsigned int, uint, unsigned int, u32, r)
CUB_STORES_012(unsigned long long, ulonglong, unsigned long long, u64, l)
CUB_STORES_4L(ulonglong4, ulonglong2);

// Floating point
CUB_STORES_0124(float, float, float, f32, f)
CUB_STORES_012(double, double, unsigned long long, u64, l)
CUB_STORES_4L(double4, double2);

// Signed longs / unsigned longs
#if defined(__LP64__)
    // longs are 64-bit on non-Windows 64-bit compilers
    CUB_STORES_012(long, long, long, u64, l)
    CUB_STORES_4L(long4, long2);
    CUB_STORES_012(unsigned long, ulong, unsigned long, u64, l)
    CUB_STORES_4L(ulong4, ulong2);
#else
    // longs are 32-bit on everything else
    CUB_STORES_0124(long, long, long, u32, r)
    CUB_STORES_0124(unsigned long, ulong, unsigned long, u32, r)
#endif


/**
 * Undefine macros
 */

#undef CUB_G_STORE_0
#undef CUB_G_STORE_1
#undef CUB_G_STORE_2
#undef CUB_G_STORE_4
#undef CUB_SV_STORE_1
#undef CUB_SV_STORE_2
#undef CUB_SV_STORE_4
#undef CUB_STORE_4L
#undef CUB_STORES_0
#undef CUB_STORES_1
#undef CUB_STORES_2
#undef CUB_STORES_4
#undef CUB_STORES_4L
#undef CUB_STORES_012
#undef CUB_STORES_0124


} // namespace cub
CUB_NS_POSTFIX
