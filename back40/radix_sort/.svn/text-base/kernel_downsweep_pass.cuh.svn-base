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

#include "sort_utils.cuh"
#include "cta_upsweep_pass.cuh"

BACK40_NS_PREFIX
namespace back40 {
namespace radix_sort {


/**
 * Kernel entry point
 */
template <
    typename     BlockDownsweepPassPolicy,
    int         MIN_BLOCK_OCCUPANCY,
    typename     KeyType,
    typename     ValueType,
    typename     SizeT>
__launch_bounds__ (
    BlockDownsweepPassPolicy::BLOCK_THREADS,
    MIN_BLOCK_OCCUPANCY)
__global__ void DownsweepKernel(
    unsigned int                *d_queue_counters,
    unsigned int                *d_steal_counters,
    BinDescriptor               *d_bins_out,
    SizeT                       *d_spine,
    KeyType                     *d_keys_in,
    KeyType                     *d_keys_out,
    ValueType                   *d_values_in,
    ValueType                   *d_values_out,
    unsigned int                current_bit,
    cub::BlockEvenShare<SizeT>    cta_even_share,
    int                         iteration)
{
    enum
    {
        TILE_ITEMS      = BlockDownsweepPassPolicy::TILE_ITEMS,
        RADIX_BITS      = BlockDownsweepPassPolicy::RADIX_BITS,
        RADIX_DIGITS    = 1 << RADIX_BITS,
    };

    // CTA abstraction type
    typedef BlockDownsweepPass<
        BlockDownsweepPassPolicy,
        KeyType,
        ValueType,
        SizeT> BlockDownsweepPassT;

    // Shared data structures
    __shared__ typename BlockDownsweepPassT::SmemStorage         smem_storage;
    __shared__ cub::BlockEvenShare<SizeT, TILE_ITEMS>             cta_progress;
    __shared__ volatile unsigned int                         enqueue_offset;

    // Read exclusive bin prefixes
    SizeT bin_prefix;
    if (threadIdx.x < RADIX_DIGITS)
    {
        int spine_offset = (gridDim.x * threadIdx.x) + blockIdx.x;
        bin_prefix = d_spine[spine_offset];

        if (blockIdx.x == 0)
        {
            SizeT next_spine_offset = (threadIdx.x == RADIX_DIGITS - 1) ?
                cta_even_share.num_elements :
                d_spine[spine_offset + gridDim.x];

            SizeT bin_count = next_spine_offset - bin_prefix;

/*
            printf("Global digit %d created partition: bit(%d) bin_count(%d) offset(%d)\n",
                threadIdx.x,
                bin.current_bit,
                bin.num_elements,
                bin.offset);
*/

            unsigned int active_bins_vote         = __ballot(bin_count > 0);
            unsigned int thread_mask             = (1 << threadIdx.x) - 1;
            int active_bins                     = __popc(active_bins_vote);
            int active_bins_prefix                 = __popc(active_bins_vote & thread_mask);

            if (threadIdx.x == 0)
            {
                // Increment enqueue offset
                if (iteration == 0)
                {
                    d_queue_counters[0] = active_bins;
                    enqueue_offset = 0;
                }
                else
                {
                    enqueue_offset = atomicAdd(d_queue_counters + (iteration & 3), active_bins);
                }

                // Reset next queue counter
                d_queue_counters[(iteration + 1) & 3] = 0;
                d_steal_counters[0] = 0;
                d_steal_counters[1] = 0;
            }

            BinDescriptor bin(bin_prefix, bin_count, current_bit);
            d_bins_out[enqueue_offset + active_bins_prefix] = bin;
        }

        if (threadIdx.x == 0)
        {
            // Determine our CTA's work range
            cta_progress.Init(cta_even_share);
        }
    }


    // Sync to acquire work range
    __syncthreads();

    // Scatter keys to each radix digit bin
    BlockDownsweepPassT::DownsweepPass(
        smem_storage,
        bin_prefix,
        d_keys_in + cta_progress.cta_offset,
        d_keys_out,
        d_values_in + cta_progress.cta_offset,
        d_values_out,
        current_bit,
        cta_progress.num_elements);
}


/**
 * Downsweep kernel props
 */
template <
    typename KeyType,
    typename ValueType,
    typename SizeT>
struct DownsweepKernelProps : cub::KernelProps
{
    // Kernel function type
    typedef void (*KernelFunc)(
        unsigned int*,
        unsigned int*,
        BinDescriptor*,
        SizeT*,
        KeyType*,
        KeyType*,
        ValueType*,
        ValueType*,
        unsigned int,
        cub::BlockEvenShare<SizeT>,
        int);

    // Fields
    KernelFunc                     kernel_func;
    int                         tile_items;
    cudaSharedMemConfig         sm_bank_config;
    int                            radix_bits;

    /**
     * Initializer
     */
    template <
        typename BlockDownsweepPassPolicy,
        typename OpaqueBlockDownsweepPassPolicy,
        int MIN_BLOCK_OCCUPANCY>
    cudaError_t Init(const cub::CudaProps &cuda_props)    // CUDA properties for a specific device
    {
        // Initialize fields
        kernel_func             = DownsweepKernel<OpaqueBlockDownsweepPassPolicy, MIN_BLOCK_OCCUPANCY>;
        tile_items                 = BlockDownsweepPassPolicy::TILE_ITEMS;
        sm_bank_config             = BlockDownsweepPassPolicy::SMEM_CONFIG;
        radix_bits                = BlockDownsweepPassPolicy::RADIX_BITS;

        // Initialize super class
        return cub::KernelProps::Init(
            kernel_func,
            BlockDownsweepPassPolicy::BLOCK_THREADS,
            cuda_props);
    }

    /**
     * Initializer
     */
    template <typename BlockDownsweepPolicy>
    cudaError_t Init(const cub::CudaProps &cuda_props)    // CUDA properties for a specific device
    {
        return Init<BlockDownsweepPolicy, BlockDownsweepPolicy>(cuda_props);
    }

};


} // namespace radix_sort
} // namespace back40
BACK40_NS_POSTFIX
