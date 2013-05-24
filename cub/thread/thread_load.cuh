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
 * Thread utilities for reading memory using PTX cache modifiers.
 */

#pragma once

#include <cuda.h>

#include <iterator>

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
 * \brief Enumeration of PTX cache-modifiers for memory load operations.
 */
enum PtxLoadModifier
{
    // Global load modifiers
    LOAD_DEFAULT,       ///< Default (no modifier)
    LOAD_CA,            ///< Cache at all levels
    LOAD_CG,            ///< Cache at global level
    LOAD_CS,            ///< Cache streaming (likely to be accessed once)
    LOAD_CV,            ///< Cache as volatile (including cached system lines)
    LOAD_LDG,           ///< Cache as texture

    // Shared load modifiers
    LOAD_VS,            ///< Volatile shared

};


/**
 * \name Simple I/O
 * @{
 */


#ifndef DOXYGEN_SHOULD_SKIP_THIS    // Do not document

//-----------------------------------------------------------------------------
// Generic Load() operation
//-----------------------------------------------------------------------------

/**
 * Define HasLoad structure for testing the presence of nested
 * LoadTag type names within data types
 */
CUB_DEFINE_DETECT_NESTED_TYPE(HasLoad, LoadTag)


/**
 * Dispatch specializer
 */
template <PtxLoadModifier MODIFIER, bool HAS_THREAD_LOAD>
struct LoadDispatch;


/**
 * Dispatch Load() to value if it exposes a LoadTag typedef
 */
template <PtxLoadModifier MODIFIER>
struct LoadDispatch<MODIFIER, true>
{
    // Iterator
    template <typename InputIteratorRA>
    static __device__ __forceinline__ typename std::iterator_traits<InputIteratorRA>::value_type Load(InputIteratorRA itr)
    {
        typename std::iterator_traits<InputIteratorRA>::value_type val;
        val.Load<MODIFIER>(itr);
        return val;
    }
};


/**
 * Generic LOAD_DEFAULT specialization
 */
template <>
struct LoadDispatch<LOAD_DEFAULT, false>
{
    // Iterator
    template <typename InputIteratorRA>
    static __device__ __forceinline__ typename std::iterator_traits<InputIteratorRA>::value_type Load(InputIteratorRA itr)
    {
        // Straightforward dereference
        return *itr;
    }
};


/**
 * Generic LOAD_VS specialization
 */
template <>
struct LoadDispatch<LOAD_VS, false>
{
    // Iterator
    template <typename InputIteratorRA>
    static __device__ __forceinline__ typename std::iterator_traits<InputIteratorRA>::value_type Load(InputIteratorRA itr)
    {
        typedef typename std::iterator_traits<InputIteratorRA>::value_type T;

        const bool USE_VOLATILE = NumericTraits<T>::PRIMITIVE;

        typedef typename If<USE_VOLATILE, volatile T, T>::Type PtrT;

        // Straightforward dereference of pointer
        T val = *reinterpret_cast<PtrT*>(&*itr);

        // Prevent compiler from reordering or omitting memory accesses between rounds
        if (!USE_VOLATILE) __threadfence_block();

        return val;
    }
};



/**
 * Generic LOAD_CG specialization for SM10-SM13 architectures
 */
#if CUB_PTX_ARCH < 200
template <>
struct LoadDispatch<LOAD_CG, false>
{
    // Iterator
    template <typename InputIteratorRA>
    static __device__ __forceinline__ typename std::iterator_traits<InputIteratorRA>::value_type Load(InputIteratorRA itr)
    {
        // Straightforward dereference
        return *itr;
    }
};
#endif  // CUB_PTX_ARCH < 200



#endif // DOXYGEN_SHOULD_SKIP_THIS

/**
 * \brief Thread utility for reading memory using cub::PtxLoadModifier cache modifiers.
 *
 * Cache modifiers will only be effected for built-in types (i.e., C++
 * primitives and CUDA vector-types).
 *
 * For example:
 * \par
 * \code
 * #include <cub/cub.cuh>
 *
 * // 32-bit load using cache-global modifier:
 * int *d_in;
 * int val = cub::Load<cub::LOAD_CA>(d_in + threadIdx.x);
 *
 * // 16-bit load using default modifier
 * short *d_in;
 * short val = cub::Load<cub::LOAD_DEFAULT>(d_in + threadIdx.x);
 *
 * // 256-bit load using cache-volatile modifier
 * double4 *d_in;
 * double4 val = cub::Load<cub::LOAD_CV>(d_in + threadIdx.x);
 *
 * // 96-bit load using default cache modifier (ignoring LOAD_CS)
 * struct TestFoo { bool a; short b; };
 * TestFoo *d_struct;
 * TestFoo val = cub::Load<cub::LOAD_CS>(d_in + threadIdx.x);
 * \endcode
 *
 */
template <
    PtxLoadModifier MODIFIER,
    typename InputIteratorRA>
__device__ __forceinline__ typename std::iterator_traits<InputIteratorRA>::value_type Load(InputIteratorRA itr)
{
    typedef typename std::iterator_traits<InputIteratorRA>::value_type T;
    return LoadDispatch<MODIFIER, HasLoad<T>::VALUE>::Load(itr);
}



#ifndef DOXYGEN_SHOULD_SKIP_THIS    // Do not document

//-----------------------------------------------------------------------------
// Load() specializations by modifier and data type (i.e., primitives
// and CUDA vector types)
//-----------------------------------------------------------------------------

/**
 * Define a global Load() specialization for type
 */
#define CUB_G_LOAD_0(type, asm_type, ptx_type, reg_mod, cub_modifier, ptx_modifier)     \
    template<>                                                                          \
    type Load<cub_modifier, type*>(type* ptr)                                     \
    {                                                                                   \
        asm_type raw;                                                                   \
        asm volatile ("ld.global."#ptx_modifier"."#ptx_type" %0, [%1];" :               \
            "="#reg_mod(raw) :                                                          \
            _CUB_ASM_PTR_(ptr));                                                        \
        type val = reinterpret_cast<type&>(raw);                                        \
        return val;                                                                     \
    }

/**
 * Define a global Load() specialization for the vector-1 type
 */
#define CUB_G_LOAD_1(type, component_type, asm_type, ptx_type, reg_mod, cub_modifier, ptx_modifier)     \
    template<>                                                                          \
    type Load<cub_modifier, type*>(type* ptr)                                     \
    {                                                                                   \
        asm_type raw;                                                                   \
        asm volatile ("ld.global."#ptx_modifier"."#ptx_type" %0, [%1];" :               \
            "="#reg_mod(raw) :                                                          \
            _CUB_ASM_PTR_(ptr));                                                        \
        type val = { reinterpret_cast<component_type&>(raw) };                          \
        return val;                                                                     \
    }

/**
 * Define a volatile-shared Load() specialization for the built-in
 * vector-1 type.  Simply use the component version.
 */
#define CUB_VS_LOAD_1(type, component_type, asm_type, ptx_type, reg_mod)                \
    template<>                                                                          \
    type Load<LOAD_VS, type*>(type* ptr)                                      \
    {                                                                                   \
        type val;                                                                       \
        val.x = Load<LOAD_VS>((component_type*) ptr);                         \
        return val;                                                                     \
    }

/**
 * Define a global Load() specialization for the vector-2 type
 */
#define CUB_G_LOAD_2(type, component_type, asm_type, ptx_type, reg_mod, cub_modifier, ptx_modifier)     \
    template<>                                                                          \
    type Load<cub_modifier, type*>(type* ptr)                                     \
    {                                                                                   \
        asm_type raw_x, raw_y;                                                          \
        asm volatile ("ld.global."#ptx_modifier".v2."#ptx_type" {%0, %1}, [%2];" :      \
            "="#reg_mod(raw_x),                                                         \
            "="#reg_mod(raw_y) :                                                        \
            _CUB_ASM_PTR_(ptr));                                                        \
        type val = {                                                                    \
            reinterpret_cast<component_type&>(raw_x),                                   \
            reinterpret_cast<component_type&>(raw_y) };                                 \
        return val;                                                                     \
    }

/**
 * Define a volatile-shared Load() specialization for the vector-2 type
 * Performs separate references if the component_type is only 1 byte (otherwise we lose
 * performance due to the bitfield ops to assemble the value)
 */
#define CUB_VS_LOAD_2(type, component_type, asm_type, ptx_type, reg_mod)                \
    template<>                                                                          \
    type Load<LOAD_VS, type*>(type* ptr)                                      \
    {                                                                                   \
        type val;                                                                       \
        if ((sizeof(component_type) == 1) || (CUDA_VERSION < 4100))                     \
        {                                                                               \
            component_type *base_ptr = (component_type*) ptr;                           \
            val.x = Load<LOAD_VS>(base_ptr);                                  \
            val.y = Load<LOAD_VS>(base_ptr + 1);                              \
        }                                                                               \
        else                                                                            \
        {                                                                               \
            asm_type raw_x, raw_y;                                                      \
            asm volatile ("{"                                                           \
                "    .reg ."_CUB_ASM_PTR_SIZE_" t1;"                                    \
                "    cvta.to.shared."_CUB_ASM_PTR_SIZE_" t1, %2;"                       \
                "    ld.volatile.shared.v2."#ptx_type" {%0, %1}, [t1];"                 \
                "}" :                                                                   \
                "="#reg_mod(raw_x),                                                     \
                "="#reg_mod(raw_y) :                                                    \
                _CUB_ASM_PTR_(ptr));                                                    \
            val.x = reinterpret_cast<component_type&>(raw_x);                           \
            val.y = reinterpret_cast<component_type&>(raw_y);                           \
        }                                                                               \
        return val;                                                                     \
    }

/**
 * Define a global Load() specialization for the vector-4 type
 */
#define CUB_G_LOAD_4(type, component_type, asm_type, ptx_type, reg_mod, cub_modifier, ptx_modifier)     \
    template<>                                                                          \
    type Load<cub_modifier, type*>(type* ptr)                                     \
    {                                                                                   \
        asm_type raw_x, raw_y, raw_z, raw_w;                                            \
        asm volatile ("ld.global."#ptx_modifier".v4."#ptx_type" {%0, %1, %2, %3}, [%4];" :              \
            "="#reg_mod(raw_x),                                                         \
            "="#reg_mod(raw_y),                                                         \
            "="#reg_mod(raw_z),                                                         \
            "="#reg_mod(raw_w) :                                                        \
            _CUB_ASM_PTR_(ptr));                                                        \
        type val = {                                                                    \
            reinterpret_cast<component_type&>(raw_x),                                   \
            reinterpret_cast<component_type&>(raw_y),                                   \
            reinterpret_cast<component_type&>(raw_z),                                   \
            reinterpret_cast<component_type&>(raw_w) };                                 \
        return val;                                                                     \
    }

/**
 * Define a volatile-shared Load() specialization for the vector-4 type.
 * Performs separate references if the component_type is only 1 byte (otherwise we lose
 * performance due to the bitfield ops to assemble the value)
 */
#define CUB_VS_LOAD_4(type, component_type, asm_type, ptx_type, reg_mod)                \
    template<>                                                                          \
    type Load<LOAD_VS, type*>(type* ptr)                                      \
    {                                                                                   \
        type val;                                                                       \
        if ((sizeof(component_type) == 1) || (CUDA_VERSION < 4100))                     \
        {                                                                               \
            component_type *base_ptr = (component_type*) ptr;                           \
            val.x = Load<LOAD_VS>(base_ptr);                                  \
            val.y = Load<LOAD_VS>(base_ptr + 1);                              \
            val.z = Load<LOAD_VS>(base_ptr + 2);                              \
            val.w = Load<LOAD_VS>(base_ptr + 3);                              \
        }                                                                               \
        else                                                                            \
        {                                                                               \
            asm_type raw_x, raw_y, raw_z, raw_w;                                        \
            asm volatile ("{"                                                           \
                "    .reg ."_CUB_ASM_PTR_SIZE_" t1;"                                    \
                "    cvta.to.shared."_CUB_ASM_PTR_SIZE_" t1, %4;"                       \
                "    ld.volatile.shared.v4."#ptx_type" {%0, %1, %2, %3}, [t1];"         \
                "}" :                                                                   \
                "="#reg_mod(raw_x),                                                     \
                "="#reg_mod(raw_y),                                                     \
                "="#reg_mod(raw_z),                                                     \
                "="#reg_mod(raw_w) :                                                    \
                _CUB_ASM_PTR_(ptr));                                                    \
            val.x = reinterpret_cast<component_type&>(raw_x);                           \
            val.y = reinterpret_cast<component_type&>(raw_y);                           \
            val.z = reinterpret_cast<component_type&>(raw_z);                           \
            val.w = reinterpret_cast<component_type&>(raw_w);                           \
        }                                                                               \
        return val;                                                                     \
    }

/**
 * Define a Load() specialization for the 64-bit
 * vector-4 type
 */
#define CUB_LOAD_4L(type, half_type, cub_modifier)                                      \
    template<>                                                                          \
    type Load<cub_modifier, type*>(type* ptr)                                     \
    {                                                                                   \
        type val;                                                                       \
        half_type* half_val = reinterpret_cast<half_type*>(&val);                       \
        half_type* half_ptr = reinterpret_cast<half_type*>(ptr);                        \
        half_val[0] = Load<cub_modifier>(half_ptr);                               \
        half_val[1] = Load<cub_modifier>(half_ptr + 1);                           \
        return val;                                                                     \
    }

/**
 * Define Load() specializations for the (non-vector) type
 */
#define CUB_LOADS_0(type, asm_type, ptx_type, reg_mod)                                  \
    CUB_G_LOAD_0(type, asm_type, ptx_type, reg_mod, LOAD_LDG, nc)                   \
    CUB_G_LOAD_0(type, asm_type, ptx_type, reg_mod, LOAD_CA, ca)                    \
    CUB_G_LOAD_0(type, asm_type, ptx_type, reg_mod, LOAD_CG, cg)                    \
    CUB_G_LOAD_0(type, asm_type, ptx_type, reg_mod, LOAD_CS, cs)                    \
    CUB_G_LOAD_0(type, asm_type, ptx_type, reg_mod, LOAD_CV, cv)

/**
 * Define Load() specializations for the vector-1 component_type
 */
#define CUB_LOADS_1(type, component_type, asm_type, ptx_type, reg_mod)                  \
    CUB_VS_LOAD_1(type, component_type, asm_type, ptx_type, reg_mod)                    \
    CUB_G_LOAD_1(type, component_type, asm_type, ptx_type, reg_mod, LOAD_LDG, nc)   \
    CUB_G_LOAD_1(type, component_type, asm_type, ptx_type, reg_mod, LOAD_CA, ca)    \
    CUB_G_LOAD_1(type, component_type, asm_type, ptx_type, reg_mod, LOAD_CG, cg)    \
    CUB_G_LOAD_1(type, component_type, asm_type, ptx_type, reg_mod, LOAD_CS, cs)    \
    CUB_G_LOAD_1(type, component_type, asm_type, ptx_type, reg_mod, LOAD_CV, cv)

/**
 * Define Load() specializations for the vector-2 component_type
 */
#define CUB_LOADS_2(type, component_type, asm_type, ptx_type, reg_mod)                  \
    CUB_VS_LOAD_2(type, component_type, asm_type, ptx_type, reg_mod)                    \
    CUB_G_LOAD_2(type, component_type, asm_type, ptx_type, reg_mod, LOAD_LDG, nc)   \
    CUB_G_LOAD_2(type, component_type, asm_type, ptx_type, reg_mod, LOAD_CA, ca)    \
    CUB_G_LOAD_2(type, component_type, asm_type, ptx_type, reg_mod, LOAD_CG, cg)    \
    CUB_G_LOAD_2(type, component_type, asm_type, ptx_type, reg_mod, LOAD_CS, cs)    \
    CUB_G_LOAD_2(type, component_type, asm_type, ptx_type, reg_mod, LOAD_CV, cv)

/**
 * Define Load() specializations for the vector-4 component_type
 */
#define CUB_LOADS_4(type, component_type, asm_type, ptx_type, reg_mod)                  \
    CUB_VS_LOAD_4(type, component_type, asm_type, ptx_type, reg_mod)                    \
    CUB_G_LOAD_4(type, component_type, asm_type, ptx_type, reg_mod, LOAD_LDG, nc)   \
    CUB_G_LOAD_4(type, component_type, asm_type, ptx_type, reg_mod, LOAD_CA, ca)    \
    CUB_G_LOAD_4(type, component_type, asm_type, ptx_type, reg_mod, LOAD_CG, cg)    \
    CUB_G_LOAD_4(type, component_type, asm_type, ptx_type, reg_mod, LOAD_CS, cs)    \
    CUB_G_LOAD_4(type, component_type, asm_type, ptx_type, reg_mod, LOAD_CV, cv)

/**
 * Define Load() specializations for the 256-bit vector-4 component_type
 */
#define CUB_LOADS_4L(type, half_type)                       \
    CUB_LOAD_4L(type, half_type, LOAD_VS)               \
    CUB_LOAD_4L(type, half_type, LOAD_LDG)              \
    CUB_LOAD_4L(type, half_type, LOAD_CA)               \
    CUB_LOAD_4L(type, half_type, LOAD_CG)               \
    CUB_LOAD_4L(type, half_type, LOAD_CS)               \
    CUB_LOAD_4L(type, half_type, LOAD_CV)

/**
 * Define vector-0/1/2 Load() specializations for the component type
 */
#define CUB_LOADS_012(component_type, vec_prefix, asm_type, ptx_type, reg_mod)          \
    CUB_LOADS_0(component_type, asm_type, ptx_type, reg_mod)                            \
    CUB_LOADS_1(vec_prefix##1, component_type, asm_type, ptx_type, reg_mod)             \
    CUB_LOADS_2(vec_prefix##2, component_type, asm_type, ptx_type, reg_mod)

/**
 * Define vector-0/1/2/4 Load() specializations for the component type
 */
#define CUB_LOADS_0124(component_type, vec_prefix, asm_type, ptx_type, reg_mod)         \
    CUB_LOADS_012(component_type, vec_prefix, asm_type, ptx_type, reg_mod)              \
    CUB_LOADS_4(vec_prefix##4, component_type, asm_type, ptx_type, reg_mod)



/**
 * Expand Load() implementations for primitive types.
 */
#if CUB_PTX_ARCH >= 200

// Signed
CUB_LOADS_0124(char, char, short, s8, h)
CUB_LOADS_0(signed char, short, s8, h)
CUB_LOADS_0124(short, short, short, s16, h)
CUB_LOADS_0124(int, int, int, s32, r)
CUB_LOADS_012(long long, longlong, long long, u64, l)
CUB_LOADS_4L(longlong4, longlong2);

// Unsigned
CUB_LOADS_0(bool, short, u8, h)
CUB_LOADS_0124(unsigned char, uchar, unsigned short, u8, h)
CUB_LOADS_0124(unsigned short, ushort, unsigned short, u16, h)
CUB_LOADS_0124(unsigned int, uint, unsigned int, u32, r)
CUB_LOADS_012(unsigned long long, ulonglong, unsigned long long, u64, l)
CUB_LOADS_4L(ulonglong4, ulonglong2);

// Floating point
CUB_LOADS_0124(float, float, float, f32, f)
CUB_LOADS_012(double, double, unsigned long long, u64, l)
CUB_LOADS_4L(double4, double2);

// Signed longs / unsigned longs
#if defined(__LP64__)
    // longs are 64-bit on non-Windows 64-bit compilers
    CUB_LOADS_012(long, long, long, u64, l)
    CUB_LOADS_4L(long4, long2);
    CUB_LOADS_012(unsigned long, ulong, unsigned long, u64, l)
    CUB_LOADS_4L(ulong4, ulong2);
#else
    // longs are 32-bit on everything else
    CUB_LOADS_0124(long, long, long, u32, r)
    CUB_LOADS_0124(unsigned long, ulong, unsigned long, u32, r)
#endif

#endif  // CUB_PTX_ARCH >= 200

/**
 * Undefine macros
 */
#undef CUB_G_LOAD_0
#undef CUB_G_LOAD_1
#undef CUB_G_LOAD_2
#undef CUB_G_LOAD_4
#undef CUB_SV_LOAD_1
#undef CUB_SV_LOAD_2
#undef CUB_SV_LOAD_4
#undef CUB_LOAD_4L
#undef CUB_LOADS_0
#undef CUB_LOADS_1
#undef CUB_LOADS_2
#undef CUB_LOADS_4
#undef CUB_LOADS_4L
#undef CUB_LOADS_012
#undef CUB_LOADS_0124
#undef CUB_LOADS_012
#undef CUB_LOADS_0124

#endif // DOXYGEN_SHOULD_SKIP_THIS

//@}  end member group

/** @} */       // end group IoModule


}               // CUB namespace
CUB_NS_POSTFIX  // Optional outer namespace(s)
