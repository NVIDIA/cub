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
 * Properties of a given CUDA device and the corresponding PTX bundle
 */

#pragma once

#include "util_arch.cuh"
#include "util_debug.cuh"
#include "util_namespace.cuh"
#include "util_macro.cuh"

/// Optional outer namespace(s)
CUB_NS_PREFIX

/// CUB namespace
namespace cub {


/**
 * \addtogroup UtilModule
 * @{
 */


/**
 * \brief Empty kernel for querying PTX manifest metadata (e.g., version) for the current device
 */
template <typename T>
__global__ void EmptyKernel(void) { }


/**
 * \brief Type for representing GPU device ordinals
 */
typedef int DeviceOrdinal;

enum
{
    /// Invalid device ordinal
    INVALID_DEVICE_ORDINAL = -1,
};


/**
 * \brief Properties of a given CUDA device and the corresponding PTX bundle
 */
class Device
{
public:

    // Version information
    int     sm_version;             ///< SM version of target device (SM version X.YZ in XYZ integer form)
    int     ptx_version;            ///< Bundled PTX version for target device (PTX version X.YZ in XYZ integer form)

    // Target device properties
    int     sm_count;               ///< Number of SMs
    int     warp_threads;           ///< Number of threads per warp
    int     smem_bank_bytes;        ///< Number of bytes per SM bank
    int     smem_banks;             ///< Number of smem banks
    int     smem_bytes;             ///< Smem bytes per SM
    int     smem_alloc_unit;        ///< Smem segment size
    bool    regs_by_block;          ///< Whether registers are allocated by threadblock (or by warp)
    int     reg_alloc_unit;         ///< Granularity of register allocation within the SM
    int     warp_alloc_unit;        ///< Granularity of warp allocation within the SM
    int     max_sm_threads;         ///< Maximum number of threads per SM
    int     max_sm_blocks;          ///< Maximum number of threadblocks per SM
    int     max_block_threads;      ///< Maximum number of threads per threadblock
    int     max_sm_registers;       ///< Maximum number of registers per SM
    int     max_sm_warps;           ///< Maximum number of warps per SM
    int     oversubscription;       ///< Heuristic for over-subscribing the device with longer-running CTAs

    /**
     * Callback for initializing device properties
     */
    template <typename ArchProps>
    __host__ __device__ __forceinline__ void Callback()
    {
        warp_threads        = ArchProps::WARP_THREADS;
        smem_bank_bytes     = ArchProps::SMEM_BANK_BYTES;
        smem_banks          = ArchProps::SMEM_BANKS;
        smem_bytes          = ArchProps::SMEM_BYTES;
        smem_alloc_unit     = ArchProps::SMEM_ALLOC_UNIT;
        regs_by_block       = ArchProps::REGS_BY_BLOCK;
        reg_alloc_unit      = ArchProps::REG_ALLOC_UNIT;
        warp_alloc_unit     = ArchProps::WARP_ALLOC_UNIT;
        max_sm_threads      = ArchProps::MAX_SM_THREADS;
        max_sm_blocks       = ArchProps::MAX_SM_THREADBLOCKS;
        max_block_threads   = ArchProps::MAX_BLOCK_THREADS;
        max_sm_registers    = ArchProps::MAX_SM_REGISTERS;
        oversubscription    = ArchProps::OVERSUBSCRIPTION;
        max_sm_warps        = max_sm_threads / warp_threads;
    }

    /// Type definition of the EmptyKernel kernel entry point
    typedef void (*EmptyKernelPtr)();

    /// Force EmptyKernel<void> to be generated if this class is used
    __host__ __device__ __forceinline__
    EmptyKernelPtr Empty()
    {
        return EmptyKernel<void>;
    }


public:

    /**
     * Initializer.  Properties are retrieved for the specified GPU ordinal.
     */
    __host__ __device__ __forceinline__
    cudaError_t Init(int device_ordinal)
    {
    #if !CUB_CNP_ENABLED

        // CUDA API calls not supported from this device
        return cudaErrorInvalidConfiguration;

    #else

        cudaError_t error = cudaSuccess;
        do
        {
            // Fill in SM version
            int major, minor;
            if (CubDebug(error = cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device_ordinal))) break;
            if (CubDebug(error = cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device_ordinal))) break;
            sm_version = major * 100 + minor * 10;

            // Fill in static SM properties
            // Initialize our device properties via callback from static device properties
            ArchProps<100>::Callback(*this, sm_version);

            // Fill in SM count
            if (CubDebug(error = cudaDeviceGetAttribute (&sm_count, cudaDevAttrMultiProcessorCount, device_ordinal))) break;

            // Fill in PTX version
        #if CUB_PTX_ARCH > 0
            ptx_version = CUB_PTX_ARCH;
        #else
            cudaFuncAttributes flush_kernel_attrs;
            if ((error = CubDebug(cudaFuncGetAttributes(&flush_kernel_attrs, Empty())))) break;
            ptx_version = flush_kernel_attrs.ptxVersion * 10;
        #endif

        }
        while (0);

        return error;

    #endif
    }

    /**
     * Initializer.  Properties are retrieved for the current GPU ordinal.
     */
    __host__ __device__ __forceinline__
    cudaError_t Init()
    {
    #if !CUB_CNP_ENABLED

        // CUDA API calls not supported from this device
        return cudaErrorInvalidConfiguration;

    #else

        cudaError_t error = cudaSuccess;
        do
        {
            int device_ordinal;
            if ((error = CubDebug(cudaGetDevice(&device_ordinal)))) break;
            if ((error = Init(device_ordinal))) break;
        }
        while (0);
        return error;

    #endif
    }

    /**
     * Computes maximum SM occupancy in thread blocks for the given kernel
     */
    template <typename KernelPtr>
    __host__ __device__ __forceinline__
    cudaError_t MaxSmOccupancy(
        int                 &max_sm_occupancy,          ///< [out] maximum number of thread blocks that can reside on a single SM
        KernelPtr           kernel_ptr,                 ///< [in] Kernel pointer for which to compute SM occupancy
        int                 block_threads)              ///< [in] Number of threads per thread block
    {
        cudaError_t error = cudaSuccess;

        do
        {
            // Get kernel attributes
            cudaFuncAttributes kernel_attrs;
            if (CubDebug(error = cudaFuncGetAttributes(&kernel_attrs, kernel_ptr))) break;

            int block_warps = CUB_ROUND_UP_NEAREST(block_threads / warp_threads, 1);

            int block_allocated_warps = CUB_ROUND_UP_NEAREST(block_warps, warp_alloc_unit);

            int block_allocated_regs = (regs_by_block) ?
                CUB_ROUND_UP_NEAREST(
                    block_allocated_warps * kernel_attrs.numRegs * warp_threads,
                    reg_alloc_unit) :
                block_allocated_warps * CUB_ROUND_UP_NEAREST(
                    kernel_attrs.numRegs * warp_threads,
                    reg_alloc_unit);

            int block_allocated_smem = CUB_ROUND_UP_NEAREST(
                kernel_attrs.sharedSizeBytes,
                smem_alloc_unit);

            int max_sm_occupancy = max_sm_blocks;

            int max_warp_occupancy = max_sm_warps / block_warps;

            int max_smem_occupancy = (block_allocated_smem > 0) ?
                    (smem_bytes / block_allocated_smem) :
                    max_sm_occupancy;

            int max_reg_occupancy = max_sm_registers / block_allocated_regs;

            max_sm_occupancy = CUB_MIN(
                CUB_MIN(max_sm_occupancy, max_warp_occupancy),
                CUB_MIN(max_smem_occupancy, max_reg_occupancy));

        } while (0);

        return error;
    }

};


/** @} */       // end group UtilModule

}               // CUB namespace
CUB_NS_POSTFIX  // Optional outer namespace(s)
