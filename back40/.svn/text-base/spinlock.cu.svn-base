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
 * Simple x86/x64 atomic spinlock
 ******************************************************************************/

namespace cub {

#if defined(_MSC_VER)

	// Microsoft VC++
	typedef long Spinlock;

	#include <intrin.h>

	/**
	 * Compiler read/write barrier
	 */
	#pragma intrinsic(_ReadWriteBarrier)

	/**
	 * Pause instruction to prevent excess processor bus usage
	 */
	__forceinline__ void CpuRelax()
	{
		__nop(); // YieldProcessor(); is reputed to generate pause instrs (instead of nops), but requires windows.h
	}

#else

	// GNU g++
	typedef int Spinlock;

	/**
	 * Compiler read/write barrier
	 */
	__forceinline__ void _ReadWriteBarrier()
	{
		__sync_synchronize();
	}

	/**
	 * Atomic exchange
	 */
	__forceinline__ long _InterlockedExchange(volatile int * const Target, const int Value)
	{
		// NOTE: __sync_lock_test_and_set would be an acquire barrier, so we force a full barrier
		_ReadWriteBarrier();
		return __sync_lock_test_and_set(Target, Value);
	}

	/**
	 * Pause instruction to prevent excess processor bus usage
	 */
	__forceinline__ void CpuRelax()
	{
		asm volatile("pause\n": : :"memory");
	}

#endif


/**
 * Return when the specified spinlock has been acquired
 */
__forceinline__ void Lock(volatile Spinlock *lock)
{
	while (1) {
		if (!_InterlockedExchange(lock, 1)) return;
		while(*lock) CpuRelax();
	}
}


/**
 * Release the specified spinlock
 */
__forceinline__ void Unlock(volatile Spinlock *lock)
{
	_ReadWriteBarrier();
	*lock = 0;
}


} // namespace cub


