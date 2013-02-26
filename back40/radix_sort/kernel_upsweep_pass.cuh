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

#include "cta_upsweep_pass.cuh"

BACK40_NS_PREFIX
namespace back40 {
namespace radix_sort {


/**
 * Kernel entry point
 */
template <
    typename    BlockUpsweepPassPolicy,
    int         MIN_BLOCK_OCCUPANCY,
    typename    KeyType,
    typename    SizeT>
__launch_bounds__ (
    BlockUpsweepPassPolicy::BLOCK_THREADS,
    MIN_BLOCK_OCCUPANCY)
__global__
void UpsweepKernel(
    KeyType                     *d_keys_in,
    SizeT                       *d_spine,
    cub::BlockEvenShare<SizeT>    cta_even_share,
    unsigned int                current_bit)
{
    // Constants
    enum
    {
        TILE_ITEMS      = BlockUpsweepPassPolicy::TILE_ITEMS,
        RADIX_DIGITS    = 1 << BlockUpsweepPassPolicy::RADIX_BITS,
    };

    // CTA abstraction types
    typedef BlockUpsweepPass<BlockUpsweepPassPolicy, KeyType, SizeT> BlockUpsweepPassT;

    // Shared data structures
    __shared__ typename BlockUpsweepPassT::SmemStorage smem_storage;

    // Determine our threadblock's work range
    cta_even_share.Init();

    // Compute bin-count for each radix digit (valid in the first RADIX_DIGITS threads)
    SizeT bin_count;
    BlockUpsweepPassT::UpsweepPass(
        smem_storage,
        d_keys_in + cta_even_share.cta_offset,
        current_bit,
        cta_even_share.cta_items,
        bin_count);

    // Write out the bin_count reductions
    if (threadIdx.x < RADIX_DIGITS)
    {
        int spine_bin_offset = (gridDim.x * threadIdx.x) + blockIdx.x;
        d_spine[spine_bin_offset] = bin_count;
    }
}


/**
 * Upsweep kernel properties
 */
template <
    typename KeyType,
    typename SizeT>
struct UpsweepKernelProps : cub::KernelProps
{
    // Kernel function type
    typedef void (*KernelFunc)(
        KeyType*,
        SizeT*,
        cub::BlockEvenShare<SizeT>,
        unsigned int);

    // Fields
    KernelFunc              kernel_func;
    int                     tile_items;
    cudaSharedMemConfig     sm_bank_config;
    int                     radix_bits;

    /**
     * Initializer
     */
    template <
        typename BlockUpsweepPassPolicy,
        typename OpaqueBlockUpsweepPassPolicy,
        int MIN_BLOCK_OCCUPANCY>
    cudaError_t Init(const cub::CudaProps &cuda_props)    // CUDA properties for a specific device
    {
        // Initialize fields
        kernel_func             = UpsweepKernel<OpaqueBlockUpsweepPassPolicy, MIN_BLOCK_OCCUPANCY>;
        tile_items             = BlockUpsweepPassPolicy::TILE_ITEMS;
        sm_bank_config             = BlockUpsweepPassPolicy::SMEM_CONFIG;
        radix_bits                = BlockUpsweepPassPolicy::RADIX_BITS;

        // Initialize super class
        return cub::KernelProps::Init(
            kernel_func,
            BlockUpsweepPassPolicy::BLOCK_THREADS,
            cuda_props);
    }

    /**
     * Initializer
     */
    template <
        typename BlockUpsweepPassPolicy,
        int MIN_BLOCK_OCCUPANCY>
    cudaError_t Init(const cub::CudaProps &cuda_props)    // CUDA properties for a specific device
    {
        return Init<BlockUpsweepPassPolicy, BlockUpsweepPassPolicy, MIN_BLOCK_OCCUPANCY>(cuda_props);
    }
};


} // namespace radix_sort
} // namespace back40
BACK40_NS_POSTFIX
