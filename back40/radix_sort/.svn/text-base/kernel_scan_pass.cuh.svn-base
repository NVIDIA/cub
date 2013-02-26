/******************************************************************************
 * 
 * Copyright (c) 2010-2012, Duane Merrill.  All rights reserved.
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
 *
 ******************************************************************************/

#pragma once

#include "../../cub/cub.cuh"
#include "../ns_wrapper.cuh"

#include "cta_scan_pass.cuh"

BACK40_NS_PREFIX
namespace back40 {
namespace radix_sort {


/**
 * Kernel entry point
 */
template <
    typename BlockScanPassPolicy,
    typename T,
    typename SizeT>
__launch_bounds__ (BlockScanPassPolicy::BLOCK_THREADS, 1)
__global__
void ScanKernel(
    T            *d_in,
    T            *d_out,
    SizeT         spine_elements)
{
    // CTA abstraction type
    typedef BlockScanPass<BlockScanPassPolicy, T> BlockScanPassT;

    // Shared data structures
    __shared__ typename BlockScanPassT::SmemStorage smem_storage;

    // Only CTA-0 needs to run
    if (blockIdx.x > 0) return;

    BlockScanPassT::ScanPass(
        smem_storage,
        d_in,
        d_out,
        spine_elements);
}



/**
 * Spine kernel properties
 */
template <typename SizeT>
struct SpineKernelProps : cub::KernelProps
{
    // Kernel function type
    typedef void (*KernelFunc)(SizeT*, SizeT*, int);

    // Fields
    KernelFunc                     kernel_func;
    int                         tile_items;
    cudaSharedMemConfig         sm_bank_config;

    /**
     * Initializer
     */
    template <
        typename BlockScanPassPolicy,
        typename OpaqueBlockScanPassPolicy>
    cudaError_t Init(const cub::CudaProps &cuda_props)    // CUDA properties for a specific device
    {
        // Initialize fields
        kernel_func             = ScanKernel<OpaqueBlockScanPassPolicy>;
        tile_items             = BlockScanPassPolicy::TILE_ITEMS;
        sm_bank_config             = BlockScanPassPolicy::SMEM_CONFIG;

        // Initialize super class
        return cub::KernelProps::Init(
            kernel_func,
            BlockScanPassPolicy::BLOCK_THREADS,
            cuda_props);
    }

    /**
     * Initializer
     */
    template <typename KernelPolicy>
    cudaError_t Init(const cub::CudaProps &cuda_props)    // CUDA properties for a specific device
    {
        return Init<KernelPolicy, KernelPolicy>(cuda_props);
    }
};



} // namespace radix_sort
} // namespace back40
BACK40_NS_POSTFIX
