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
 * Static CUDA device properties by SM architectural version.
 *
 * "PTX_ARCH" reflects the PTX arch-id targeted by the active compiler pass
 * (or zero during the host pass).
 *
 * "DeviceProps" reflects the PTX architecture targeted by the active compiler
 * pass.  It provides useful compile-time statics within device code.  E.g.,:
 *
 *     __shared__ int[DeviceProps::WARP_THREADS];
 *
 *     int padded_offset = threadIdx.x + (threadIdx.x >> DeviceProps::LOG_SMEM_BANKS);
 *
 ******************************************************************************/

#pragma once

#include "ns_wrapper.cuh"

CUB_NS_PREFIX
namespace cub {


/**
 * CUDA architecture-id targeted by the active compiler pass
 */
#ifndef __CUDA_ARCH__
    // Host path
    #define PTX_ARCH 0
#else
    // Device path
    #define PTX_ARCH __CUDA_ARCH__
#endif


/**
 * Type for representing GPU device ordinals
 */
typedef int GpuOrdinal;

enum
{
    INVALID_GPU_ORDINAL = -1,
};


/**
 * Structure for statically reporting CUDA device properties, parameterized by SM
 * architecture.
 */
template <int SM_ARCH>
struct StaticDeviceProps;


/**
 * Device properties for SM30
 */
template <>
struct StaticDeviceProps<300>
{
    enum {
        LOG_WARP_THREADS    = 5,                        // 32 threads per warp
        WARP_THREADS        = 1 << LOG_WARP_THREADS,

        LOG_SMEM_BANKS      = 5,                         // 32 banks
        SMEM_BANKS          = 1 << LOG_SMEM_BANKS,

        SMEM_BANK_BYTES     = 4,                        // 4 byte bank words
        SMEM_BYTES          = 48 * 1024,                // 48KB shared memory
        SMEM_ALLOC_UNIT     = 256,                        // 256B smem allocation segment size
        REGS_BY_BLOCK       = false,                    // Allocates registers by warp
        REG_ALLOC_UNIT      = 256,                        // 256 registers allocated at a time per warp
        WARP_ALLOC_UNIT     = 4,                        // Registers are allocated at a granularity of every 4 warps per threadblock
        MAX_SM_THREADS      = 2048,                        // 2K max threads per SM
        MAX_SM_threadblockS         = 16,                        // 16 max threadblocks per SM
        MAX_BLOCK_THREADS   = 1024,                        // 1024 max threads per threadblock
        MAX_SM_REGISTERS    = 64 * 1024,                // 64K max registers per SM
    };

    // Callback utility
    template <typename T>
    static void Callback(T &target, int sm_version)
    {
        target.template Callback<StaticDeviceProps>();
    }
};


/**
 * Device properties for SM20
 */
template <>
struct StaticDeviceProps<200>
{
    enum {
        LOG_WARP_THREADS    = 5,                        // 32 threads per warp
        WARP_THREADS        = 1 << LOG_WARP_THREADS,

        LOG_SMEM_BANKS      = 5,                         // 32 banks
        SMEM_BANKS          = 1 << LOG_SMEM_BANKS,

        SMEM_BANK_BYTES     = 4,                        // 4 byte bank words
        SMEM_BYTES          = 48 * 1024,                // 48KB shared memory
        SMEM_ALLOC_UNIT     = 128,                        // 128B smem allocation segment size
        REGS_BY_BLOCK       = false,                    // Allocates registers by warp
        REG_ALLOC_UNIT      = 64,                        // 64 registers allocated at a time per warp
        WARP_ALLOC_UNIT     = 2,                        // Registers are allocated at a granularity of every 2 warps per threadblock
        MAX_SM_THREADS      = 1536,                        // 1536 max threads per SM
        MAX_SM_threadblockS         = 8,                        // 8 max threadblocks per SM
        MAX_BLOCK_THREADS   = 1024,                        // 1024 max threads per threadblock
        MAX_SM_REGISTERS    = 32 * 1024,                // 32K max registers per SM
    };

    // Callback utility
    template <typename T>
    static void Callback(T &target, int sm_version)
    {
        if (sm_version > 200) {
            StaticDeviceProps<300>::Callback(target, sm_version);
        } else {
            target.template Callback<StaticDeviceProps>();
        }
    }
};


/**
 * Device properties for SM12
 */
template <>
struct StaticDeviceProps<120>
{
    enum {
        LOG_WARP_THREADS    = 5,                        // 32 threads per warp
        WARP_THREADS        = 1 << LOG_WARP_THREADS,

        LOG_SMEM_BANKS      = 4,                         // 16 banks
        SMEM_BANKS          = 1 << LOG_SMEM_BANKS,

        SMEM_BANK_BYTES     = 4,                        // 4 byte bank words
        SMEM_BYTES          = 16 * 1024,                // 16KB shared memory
        SMEM_ALLOC_UNIT     = 512,                        // 512B smem allocation segment size
        REGS_BY_BLOCK       = true,                        // Allocates registers by threadblock
        REG_ALLOC_UNIT      = 512,                        // 512 registers allocated at time per threadblock
        WARP_ALLOC_UNIT     = 2,                        // Registers are allocated at a granularity of every 2 warps per threadblock
        MAX_SM_THREADS      = 1024,                        // 1024 max threads per SM
        MAX_SM_threadblockS         = 8,                        // 8 max threadblocks per SM
        MAX_BLOCK_THREADS   = 512,                        // 512 max threads per threadblock
        MAX_SM_REGISTERS    = 16 * 1024,                // 16K max registers per SM
    };

    // Callback utility
    template <typename T>
    static void Callback(T &target, int sm_version)
    {
        if (sm_version > 120) {
            StaticDeviceProps<200>::Callback(target, sm_version);
        } else {
            target.template Callback<StaticDeviceProps>();
        }
    }
};


/**
 * Device properties for SM10
 */
template <>
struct StaticDeviceProps<100>
{
    enum {
        LOG_WARP_THREADS    = 5,                        // 32 threads per warp
        WARP_THREADS        = 1 << LOG_WARP_THREADS,

        LOG_SMEM_BANKS      = 4,                         // 16 banks
        SMEM_BANKS          = 1 << LOG_SMEM_BANKS,

        SMEM_BANK_BYTES     = 4,                        // 4 byte bank words
        SMEM_BYTES          = 16 * 1024,                // 16KB shared memory
        SMEM_ALLOC_UNIT     = 512,                        // 512B smem allocation segment size
        REGS_BY_BLOCK       = true,                        // Allocates registers by threadblock
        REG_ALLOC_UNIT      = 256,                        // 256 registers allocated at time per threadblock
        WARP_ALLOC_UNIT     = 2,                        // Registers are allocated at a granularity of every 2 warps per threadblock
        MAX_SM_THREADS      = 768,                        // 768 max threads per SM
        MAX_SM_threadblockS         = 8,                        // 8 max threadblocks per SM
        MAX_BLOCK_THREADS   = 512,                        // 512 max threads per threadblock
        MAX_SM_REGISTERS    = 8 * 1024,                    // 8K max registers per SM
    };

    // Callback utility
    template <typename T>
    static void Callback(T &target, int sm_version)
    {
        if (sm_version > 100) {
            StaticDeviceProps<120>::Callback(target, sm_version);
        } else {
            target.template Callback<StaticDeviceProps>();
        }
    }
};


/**
 * Device properties for SM21
 */
template <>
struct StaticDeviceProps<210> : StaticDeviceProps<200> {};        // Derives from SM20

/**
 * Device properties for SM13
 */
template <>
struct StaticDeviceProps<130> : StaticDeviceProps<120> {};        // Derives from SM12

/**
 * Device properties for SM11
 */
template <>
struct StaticDeviceProps<110> : StaticDeviceProps<100> {};        // Derives from SM10

/**
 * Unknown device properties
 */
template <int SM_ARCH>
struct StaticDeviceProps : StaticDeviceProps<100> {};            // Derives from SM10



/**
 * Device properties for the arch-id targeted by the active compiler pass.
 */
struct DeviceProps : StaticDeviceProps<PTX_ARCH> {};



} // namespace cub
CUB_NS_POSTFIX
