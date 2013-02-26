/******************************************************************************
 * 
 * Copyright (c) 2011-2012, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2012, NVIDIA CORPORATION.  All rights reserved.
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License. 
 * 
 ******************************************************************************/

/******************************************************************************
 * Kernel utilities for reading/writing memory using cache modifiers
 ******************************************************************************/

#pragma once

#include <cub/device_intrinsics.cuh>

namespace cub {


/******************************************************************************
 * Load
 ******************************************************************************/

/**
 * Enumeration of read cache modifiers.
 */
enum ReadModifier {
	READ_NONE,		// Default (currently READ_CA for global loads, nothing for smem loads)
	READ_CA,		// Cache at all levels
	READ_CG,		// Cache at global level
	READ_CS, 		// Cache streaming (likely to be accessed once)
	READ_CV, 		// Cache as volatile (including cached system lines)
	READ_TEX,		// Texture (defaults to NONE if no tex reference is provided)

	READ_LIMIT
};


template <ReadModifier READ_MODIFIER, typename T>
__device__ __forceinline__ T Load(T *ptr)
{
	return *ptr;
}


#define CUB_LOAD_0(type, raw_type, ptx_type, reg_mod, modifier, ptx_modifier)			\
	template<>																			\
	type Load<modifier, type>(type* ptr) 												\
	{																					\
		raw_type raw;																	\
		asm("ld.global."#ptx_modifier"."#ptx_type" %0, [%1];" :							\
			"="#reg_mod(raw) : 															\
			_CUB_ASM_PTR_(ptr));														\
		type retval = raw;																\
		return retval;																	\
	}

#define CUB_LOAD_1(type, vector_type, raw_type, ptx_type, reg_mod, modifier, ptx_modifier)	\
	template<>																			\
	vector_type Load<modifier, vector_type>(vector_type* ptr) 							\
	{																					\
		raw_type raw;																	\
		asm("ld.global."#ptx_modifier"."#ptx_type" %0, [%1];" :							\
			"="#reg_mod((raw.x)) :														\
			_CUB_ASM_PTR_(ptr));														\
		vector_type retval = {reinterpret_cast<type&>(raw.x)};							\
		return retval;																	\
	}

#define CUB_LOAD_2(type, vector_type, raw_type, ptx_type, reg_mod, modifier, ptx_modifier)	\
	template<>																			\
	vector_type Load<modifier, vector_type>(vector_type* ptr) 							\
	{																					\
		raw_type raw;																	\
		asm("ld.global."#ptx_modifier".v2."#ptx_type" {%0, %1}, [%2];" :				\
			"="#reg_mod(raw.x), 														\
			"="#reg_mod(raw.y) :														\
			_CUB_ASM_PTR_(ptr));														\
			vector_type retval = {														\
				reinterpret_cast<type&>(raw.x), 										\
				reinterpret_cast<type&>(raw.y) };										\
			return retval;																\
	}

#define CUB_LOAD_4(type, vector_type, raw_type, ptx_type, reg_mod, modifier, ptx_modifier)	\
	template<>																			\
	vector_type Load<modifier, vector_type>(vector_type* ptr) 							\
	{																					\
		raw_type raw;																	\
		asm("ld.global."#ptx_modifier".v4."#ptx_type" {%0, %1, %2, %3}, [%4];" :		\
			"="#reg_mod(raw.x), 														\
			"="#reg_mod(raw.y), 														\
			"="#reg_mod(raw.z), 														\
			"="#reg_mod(raw.w) :														\
			_CUB_ASM_PTR_(ptr));														\
			vector_type retval = {														\
				reinterpret_cast<type&>(raw.x), 										\
				reinterpret_cast<type&>(raw.y), 										\
				reinterpret_cast<type&>(raw.z), 										\
				reinterpret_cast<type&>(raw.w) };										\
		return retval;																	\
	}


#define CUB_LOAD_4L(vector_type, half_type, modifier)									\
	template<>																			\
	vector_type Load<modifier, vector_type>(vector_type* ptr) 							\
	{																					\
		vector_type retval; 															\
		half_type* halves = reinterpret_cast<half_type*>(&retval);						\
		halves[0] = Load<modifier>(reinterpret_cast<half_type*>(ptr));					\
		halves[1] = Load<modifier>(reinterpret_cast<half_type*>(ptr) + 1);				\
		return retval;																	\
	}



/**
 * Defines specialized load ops for base type
 */
#define CUB_LOADS_0(type, raw_type, ptx_type, reg_mod)									\
	CUB_LOAD_0(type, raw_type, ptx_type, reg_mod, READ_CA, ca)							\
	CUB_LOAD_0(type, raw_type, ptx_type, reg_mod, READ_CG, cg)							\
	CUB_LOAD_0(type, raw_type, ptx_type, reg_mod, READ_CS, cs)							\
	CUB_LOAD_0(type, raw_type, ptx_type, reg_mod, READ_CV, cv)

#define CUB_LOADS_1(type, vector_type, raw_type, ptx_type, reg_mod)						\
	CUB_LOAD_1(type, vector_type, raw_type, ptx_type, reg_mod, READ_CA, ca)				\
	CUB_LOAD_1(type, vector_type, raw_type, ptx_type, reg_mod, READ_CG, cg)				\
	CUB_LOAD_1(type, vector_type, raw_type, ptx_type, reg_mod, READ_CS, cs)				\
	CUB_LOAD_1(type, vector_type, raw_type, ptx_type, reg_mod, READ_CV, cv)

#define CUB_LOADS_2(type, vector_type, raw_type, ptx_type, reg_mod)						\
	CUB_LOAD_2(type, vector_type, raw_type, ptx_type, reg_mod, READ_CA, ca)				\
	CUB_LOAD_2(type, vector_type, raw_type, ptx_type, reg_mod, READ_CG, cg)				\
	CUB_LOAD_2(type, vector_type, raw_type, ptx_type, reg_mod, READ_CS, cs)				\
	CUB_LOAD_2(type, vector_type, raw_type, ptx_type, reg_mod, READ_CV, cv)

#define CUB_LOADS_4(type, vector_type, raw_type, ptx_type, reg_mod)						\
	CUB_LOAD_4(type, vector_type, raw_type, ptx_type, reg_mod, READ_CA, ca)				\
	CUB_LOAD_4(type, vector_type, raw_type, ptx_type, reg_mod, READ_CG, cg)				\
	CUB_LOAD_4(type, vector_type, raw_type, ptx_type, reg_mod, READ_CS, cs)				\
	CUB_LOAD_4(type, vector_type, raw_type, ptx_type, reg_mod, READ_CV, cv)

#define CUB_LOADS_4L(vector_type, half_type)					\
	CUB_LOAD_4L(vector_type, half_type, READ_CA)				\
	CUB_LOAD_4L(vector_type, half_type, READ_CG)				\
	CUB_LOAD_4L(vector_type, half_type, READ_CS)				\
	CUB_LOAD_4L(vector_type, half_type, READ_CV)

#define CUB_LOADS_012(type, prefix, raw_type, raw_prefix, ptx_type, reg_mod)			\
	CUB_LOADS_0(type, raw_type, ptx_type, reg_mod)										\
	CUB_LOADS_1(type, prefix##1, raw_prefix##1, ptx_type, reg_mod)						\
	CUB_LOADS_2(type, prefix##2, raw_prefix##2, ptx_type, reg_mod)

#define CUB_LOADS_0124(type, prefix, raw_type, raw_prefix, ptx_type, reg_mod)			\
	CUB_LOADS_012(type, prefix, raw_type, raw_prefix, ptx_type, reg_mod)				\
	CUB_LOADS_4(type, prefix##4, raw_prefix##4, ptx_type, reg_mod)


/**
 * Expand Load() implementations for all built-in types.
 */

// Signed
CUB_LOADS_0124(char, char, short, short, s8, h)
CUB_LOADS_0124(short, short, short, short, s16, h)
CUB_LOADS_0124(int, int, int, int, s32, r)
CUB_LOADS_0(signed char, short, s8, h)
CUB_LOADS_012(long long, longlong, long long, longlong, u64, l)
CUB_LOADS_4L(longlong4, longlong2);

// Unsigned
CUB_LOADS_0124(unsigned char, uchar, unsigned short, ushort, u8, h)
CUB_LOADS_0124(unsigned short, ushort, unsigned short, ushort, u16, h)
CUB_LOADS_0124(unsigned int, uint, unsigned int, uint, u32, r)
CUB_LOADS_012(unsigned long long, ulonglong, unsigned long long, ulonglong, u64, l)
CUB_LOADS_4L(ulonglong4, ulonglong2);

// Floating point
CUB_LOADS_0124(float, float, float, float, f32, f)
CUB_LOADS_012(double, double, unsigned long long, ulonglong, u64, l)
CUB_LOADS_4L(double4, double2);


// Signed longs / unsigned longs
#if defined(__LP64__)

	// longs are 64-bit on non-Windows 64-bit compilers
	CUB_LOADS_012(long, long, long, long, u64, l)
	CUB_LOADS_4L(long4, long2);
	CUB_LOADS_012(unsigned long, ulong, unsigned long, ulong, u64, l)
	CUB_LOADS_4L(ulong4, ulong2);

#else

	// longs are 32-bit on everything else
	CUB_LOADS_0124(long, long, long, long, u32, r)
	CUB_LOADS_0124(unsigned long, ulong, unsigned long, ulong, u32, r)

#endif


/**
 * Undefine macros
 */
#undef CUB_LOAD_0
#undef CUB_LOAD_1
#undef CUB_LOAD_2
#undef CUB_LOAD_4
#undef CUB_LOAD_4L
#undef CUB_LOADS_0
#undef CUB_LOADS_1
#undef CUB_LOADS_2
#undef CUB_LOADS_4
#undef CUB_LOADS_4L
#undef CUB_LOADS_012
#undef CUB_LOADS_0124




/******************************************************************************
 * Store
 ******************************************************************************/


/**
 * Enumeration of write cache modifiers.
 */
enum WriteModifier {
	WRITE_NONE,		// Default (currently WRITE_WB)
	WRITE_WB,		// Cache write-back all coherent levels
	WRITE_CG,		// Cache at global level
	WRITE_CS, 		// Cache streaming (likely to be accessed once)
	WRITE_WT, 		// Cache write-through (to system memory)

	WRITE_LIMIT
};


} // namespace cub

