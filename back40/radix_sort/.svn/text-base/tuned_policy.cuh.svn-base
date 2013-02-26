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
 * Tuned derivatives of radix sorting policy
 ******************************************************************************/

#pragma once

#include "../../cub/cub.cuh"
#include "../ns_wrapper.cuh"

#include "dispatch_policy.cuh"
#include "cta_upsweep_pass.cuh"
#include "cta_downsweep_pass.cuh"
#include "cta_scan_pass.cuh"
#include "cta_single_tile.cuh"
#include "kernel_hybrid_pass.cuh"


BACK40_NS_PREFIX
namespace back40 {
namespace radix_sort {


/**
 * Tuned policy specializations
 */
template <
    int         TUNE_ARCH,
    typename    KeyType,
    typename    ValueType,
    typename    SizeT>
struct TunedPolicy;


/**
 * SM20
 */
template <
    typename    KeyType,
    typename    ValueType,
    typename    SizeT>
struct TunedPolicy<200, KeyType, ValueType, SizeT>
{
    enum
    {
        RADIX_BITS                    = 5,
        KEYS_ONLY                     = cub::Equals<ValueType, cub::NullType>::VALUE,
        LARGE_DATA                    = (sizeof(KeyType) > 4) || (sizeof(ValueType) > 4),
    };

    // Dispatch policy
    typedef DispatchPolicy <
        8,                                          // UPSWEEP_MIN_BLOCK_OCCUPANCY
        4,                                          // DOWNSWEEP_MIN_BLOCK_OCCUPANCY
        4>                                          // HYBRID_MIN_BLOCK_OCCUPANCY
            DispatchPolicyT;

    // Upsweep pass CTA policy
    typedef BlockUpsweepPassPolicy<
        RADIX_BITS,                                 // RADIX_BITS
        128,                                        // BLOCK_THREADS
        17,                                         // THREAD_ITEMS,
        cub::PTX_LOAD_NONE,                         // LOAD_MODIFIER
        cub::PTX_STORE_NONE,                        // STORE_MODIFIER
        cudaSharedMemBankSizeFourByte>              // SMEM_CONFIG
            BlockUpsweepPassPolicyT;

    // Spine-scan pass CTA policy
    typedef BlockScanPassPolicy<
        256,                                    // BLOCK_THREADS
        4,                                        // THREAD_STRIP_ITEMS
        4,                                        // TILE_STRIPS
        cub::PTX_LOAD_NONE,                            // LOAD_MODIFIER
        cub::PTX_STORE_NONE,                        // STORE_MODIFIER
        cudaSharedMemBankSizeFourByte>            // SMEM_CONFIG
            BlockScanPassPolicyT;

    // Downsweep pass CTA policy
    typedef BlockDownsweepPassPolicy<
        RADIX_BITS,                                // RADIX_BITS
        128,                                    // BLOCK_THREADS
        17,                                        // THREAD_ITEMS
        SCATTER_TWO_PHASE,                        // SCATTER_STRATEGY
        cub::PTX_LOAD_NONE,                         // LOAD_MODIFIER
        cub::PTX_STORE_NONE,                        // STORE_MODIFIER
        cudaSharedMemBankSizeFourByte>            // SMEM_CONFIG
            BlockDownsweepPassPolicyT;

    // Single-tile CTA policy
    typedef BlockSingleTilePolicy<
        RADIX_BITS,                                // RADIX_BITS
        192,                                    // BLOCK_THREADS
        ((KEYS_ONLY) ? 17 : 9),                 // THREAD_ITEMS
        cub::PTX_LOAD_NONE,                         // LOAD_MODIFIER
        cub::PTX_STORE_NONE,                        // STORE_MODIFIER
        cudaSharedMemBankSizeFourByte>            // SMEM_CONFIG
            BlockSingleTilePolicyT;

    // Hybrid pass CTA policy
    typedef BlockHybridPassPolicy<
        RADIX_BITS,                                // RADIX_BITS
        128,                                    // BLOCK_THREADS
        17,                                     // UPSWEEP_THREAD_ITEMS
        17,                                     // DOWNSWEEP_THREAD_ITEMS
        SCATTER_TWO_PHASE,                        // DOWNSWEEP_SCATTER_STRATEGY
        ((KEYS_ONLY) ? 19 : 9),                 // SINGLE_TILE_THREAD_ITEMS
        cub::PTX_LOAD_NONE,                         // LOAD_MODIFIER
        cub::PTX_STORE_NONE,                        // STORE_MODIFIER
        cudaSharedMemBankSizeFourByte>            // SMEM_CONFIG
            BlockHybridPassPolicyT;

};





}// namespace radix_sort
}// namespace back40
BACK40_NS_POSTFIX
