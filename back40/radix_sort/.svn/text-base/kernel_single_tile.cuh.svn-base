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

#include "cta_single_tile.cuh"


BACK40_NS_PREFIX
namespace back40 {
namespace radix_sort {


/**
 * Kernel entry point
 */
template <
    typename BlockSingleTilePolicy,
    typename KeyType,
    typename ValueType>
__launch_bounds__ (
    BlockSingleTilePolicy::BLOCK_THREADS,
    1)
__global__
void SingleTileKernel(
    KeyType         *d_keys_in,
    KeyType         *d_keys_out,
    ValueType         *d_values_in,
    ValueType         *d_values_out,
    unsigned int    current_bit,
    unsigned int     bits_remaining,
    int             num_elements)
{
    // CTA abstraction type
    typedef BlockSingleTile<
        BlockSingleTilePolicy,
        KeyType,
        ValueType> BlockSingleTileT;

    // Shared data structures
    __shared__ typename BlockSingleTileT::SmemStorage smem_storage;

    // Sort input tile
    BlockSingleTileT::Sort(
        smem_storage,
        d_keys_in,
        d_keys_out,
        d_values_in,
        d_values_out,
        current_bit,
        bits_remaining,
        num_elements);
}


/**
 * Single-tile kernel props
 */
template <
    typename KeyType,
    typename ValueType,
    typename SizeT>
struct SingleTileKernelProps : cub::KernelProps
{
    // Kernel function type
    typedef void (*KernelFunc)(
        KeyType*,
        KeyType*,
        ValueType*,
        ValueType*,
        unsigned int,
        unsigned int,
        int);

    // Fields
    KernelFunc                     kernel_func;
    int                         tile_items;
    cudaSharedMemConfig         sm_bank_config;

    /**
     * Initializer
     */
    template <
        typename BlockSingleTilePolicy,
        typename OpaqueBlockSingleTilePolicy>
    cudaError_t Init(const cub::CudaProps &cuda_props)    // CUDA properties for a specific device
    {
        // Initialize fields
        kernel_func             = SingleTileKernel<OpaqueBlockSingleTilePolicy>;
        tile_items                 = BlockSingleTilePolicy::TILE_ITEMS;
        sm_bank_config             = BlockSingleTilePolicy::SMEM_CONFIG;

        // Initialize super class
        return cub::KernelProps::Init(
            kernel_func,
            BlockSingleTilePolicy::BLOCK_THREADS,
            cuda_props);
    }

    /**
     * Initializer
     */
    template <typename BlockSingleTilePolicy>
    cudaError_t Init(const cub::CudaProps &cuda_props)    // CUDA properties for a specific device
    {
        return Init<BlockSingleTilePolicy, BlockSingleTilePolicy>(cuda_props);
    }

};

} // namespace radix_sort
} // namespace back40
BACK40_NS_POSTFIX
