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
 * CUDA properties of the bundled "fatbin" assembly and attached devices
 ******************************************************************************/

#pragma once

#include "../device_props.cuh"
#include "../debug.cuh"
#include "../ns_wrapper.cuh"

CUB_NS_PREFIX
namespace cub {


/**
 * Empty Kernel
 */
template <typename T>
__global__ void EmptyKernel(void) { }


/**
 * Encapsulation of device properties for a specific device
 */
class CudaProps
{
public:

    // Version information
    int     sm_version;                // SM version of target device (SM version X.YZ in XYZ integer form)
    int     ptx_version;            // Bundled PTX version for target device (PTX version X.YZ in XYZ integer form)

    // Target device properties
    int     sm_count;                // Number of SMs
    int     warp_threads;            // Number of threads per warp
    int     smem_bank_bytes;        // Number of bytes per SM bank
    int     smem_banks;                // Number of smem banks
    int     smem_bytes;                // Smem bytes per SM
    int     smem_alloc_unit;        // Smem segment size
    bool    regs_by_block;            // Whether registers are allocated by threadblock (or by warp)
    int     reg_alloc_unit;            // Granularity of register allocation within the SM
    int     warp_alloc_unit;        // Granularity of warp allocation within the SM
    int     max_sm_threads;            // Maximum number of threads per SM
    int     max_sm_blocks;            // Maximum number of threadblocks per SM
    int     max_block_threads;        // Maximum number of threads per threadblock
    int     max_sm_registers;        // Maximum number of registers per SM
    int     max_sm_warps;            // Maximum number of warps per SM


    /**
     * Callback for initializing device properties
     */
    template <typename StaticDeviceProps>
    void Callback()
    {
        warp_threads         = StaticDeviceProps::WARP_THREADS;
        smem_bank_bytes        = StaticDeviceProps::SMEM_BANK_BYTES;
        smem_banks            = StaticDeviceProps::SMEM_BANKS;
        smem_bytes            = StaticDeviceProps::SMEM_BYTES;
        smem_alloc_unit        = StaticDeviceProps::SMEM_ALLOC_UNIT;
        regs_by_block        = StaticDeviceProps::REGS_BY_BLOCK;
        reg_alloc_unit        = StaticDeviceProps::REG_ALLOC_UNIT;
        warp_alloc_unit        = StaticDeviceProps::WARP_ALLOC_UNIT;
        max_sm_threads        = StaticDeviceProps::MAX_SM_THREADS;
        max_sm_blocks            = StaticDeviceProps::MAX_SM_threadblockS;
        max_block_threads        = StaticDeviceProps::MAX_BLOCK_THREADS;
        max_sm_registers    = StaticDeviceProps::MAX_SM_REGISTERS;
        max_sm_warps         = max_sm_threads / warp_threads;
    }

public:

    /**
     * Initializer.  Properties are retrieved for the specified GPU ordinal.
     */
    cudaError_t Init(int gpu_ordinal)
    {
        cudaError_t error = cudaSuccess;
        do
        {
            // Obtain SM version and count
            cudaDeviceProp device_props;
            if (error = cub::Debug(cudaGetDeviceProperties(&device_props, gpu_ordinal),
                "cudaGetDeviceProperties failed", __FILE__, __LINE__)) break;
            sm_version = device_props.major * 100 + device_props.minor * 10;
            sm_count = device_props.multiProcessorCount;

            // Obtain PTX version of the bundled kernel assemblies compiled for
            // the current device
            cudaFuncAttributes flush_kernel_attrs;
            if (error = cub::Debug(cudaFuncGetAttributes(&flush_kernel_attrs, EmptyKernel<void>),
                "cudaFuncGetAttributes failed", __FILE__, __LINE__)) break;
            ptx_version = flush_kernel_attrs.ptxVersion * 10;

            // Initialize our device properties via callback from static device properties
            StaticDeviceProps<100>::Callback(*this, sm_version);

        } while (0);

        return error;
    }

    /**
     * Initializer.  Properties are retrieved for the current GPU ordinal.
     */
    cudaError_t Init()
    {
        cudaError_t error = cudaSuccess;
        do
        {
            int gpu_ordinal;
            if (error = cub::Debug(cudaGetDevice(&gpu_ordinal),
                "cudaGetDevice failed", __FILE__, __LINE__)) break;

            if (error = Init(gpu_ordinal)) break;
        } while (0);
        return error;
    }

};





} // namespace cub
CUB_NS_POSTFIX
