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
 * Threadblock Work management.
 *
 * A given threadblock may receive one of three different amounts of
 * work: "big", "normal", and "last".  The big workloads are one
 * grain greater than the normal, and the last workload
 * does the extra work.
 *
 ******************************************************************************/

#pragma once

#include "../ns_wrapper.cuh"
#include "../macro_utils.cuh"
#include "../allocator.cuh"

CUB_NS_PREFIX
namespace cub {



/**
 * Description of work distribution amongst threadblocks
 */
template <typename SizeT>
class BlockEvenShare
{
private:

    SizeT   total_grains;
    int     grid_size;
    int     big_blocks;
    SizeT   big_share;
    SizeT   normal_share;
    SizeT   normal_base_offset;


public:

    SizeT   total_items;

    // Threadblock-specific fields
    SizeT   block_offset;
    SizeT   block_oob;

    /**
     * Constructor.
     *
     * Generally constructed in host code one time.
     */
    __host__ __device__ __forceinline__ BlockEvenShare(
        SizeT   total_items,
        int     grid_size,
        int     schedule_granularity) :
            // initializers
            total_items(total_items),
            grid_size(grid_size),
            block_offset(0),
            block_oob(0)
    {
        total_grains            = (total_items + schedule_granularity - 1) / schedule_granularity;
        SizeT grains_per_block    = total_grains / grid_size;
        big_blocks                = total_grains - (grains_per_block * grid_size);        // leftover grains go to big blocks
        normal_share            = grains_per_block * schedule_granularity;
        normal_base_offset      = big_blocks * schedule_granularity;
        big_share               = normal_share + schedule_granularity;
    }


    /**
     * Initializer.
     *
     * Generally initialized by each threadblock after construction on the host.
     */
    __device__ __forceinline__ void Init()
    {
        if (blockIdx.x < big_blocks)
        {
            // This threadblock gets a big share of grains (grains_per_block + 1)
            block_offset = (blockIdx.x * big_share);
            block_oob = block_offset + big_share;
        }
        else if (blockIdx.x < total_grains)
        {
            // This threadblock gets a normal share of grains (grains_per_block)
            block_offset = normal_base_offset + (blockIdx.x * normal_share);
            block_oob = block_offset + normal_share;
        }

        // Last threadblock
        if (blockIdx.x == grid_size - 1)
        {
            block_oob = total_items;
        }
    }


    /**
     * Print to stdout
     */
    __host__ __device__ __forceinline__ void Print()
    {
        printf(
#ifdef __CUDA_ARCH__
            "\tthreadblock(%d) "
            "block_offset(%lu) "
            "block_oob(%lu) "
#endif
            "total_items(%lu)  "
            "big_blocks(%lu)  "
            "big_share(%lu)  "
            "normal_share(%lu)\n",
#ifdef __CUDA_ARCH__
                blockIdx.x,
                (unsigned long) block_offset,
                (unsigned long) block_oob,
#endif
                (unsigned long) total_items,
                (unsigned long) big_blocks,
                (unsigned long) big_share,
                (unsigned long) normal_share);
    }
};




} // namespace cub
CUB_NS_POSTFIX

