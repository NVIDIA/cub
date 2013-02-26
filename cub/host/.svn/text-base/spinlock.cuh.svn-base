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
 * Simple x86/x64 atomic spinlock
 ******************************************************************************/

#pragma once

#if defined(_WIN32) || defined(_WIN64)
    #include <intrin.h>
    #include <windows.h>
    #undef small            // Windows is terrible for polluting macro namespace

    /**
     * Compiler read/write barrier
     */
    #pragma intrinsic(_ReadWriteBarrier)

#endif

#include "../ns_wrapper.cuh"

CUB_NS_PREFIX
namespace cub {


#if defined(_MSC_VER)

    // Microsoft VC++
    typedef long Spinlock;

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
    __forceinline__ void YieldProcessor()
    {
        asm volatile("pause\n": : :"memory");
    }

#endif


/**
 * Return when the specified spinlock has been acquired
 */
__forceinline__ void Lock(volatile Spinlock *lock)
{
    while (1)
    {
        if (!_InterlockedExchange(lock, 1)) return;
        while (*lock) YieldProcessor();
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
CUB_NS_POSTFIX
