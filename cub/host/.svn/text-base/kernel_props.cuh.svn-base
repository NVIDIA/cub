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
 * Kernel function properties
 ******************************************************************************/

#pragma once

#include "../ns_wrapper.cuh"
#include "../macro_utils.cuh"

CUB_NS_PREFIX
namespace cub {

/**
 * Encapsulation of kernel properties for a combination of {device, threadblock size}
 */
struct KernelProps
{
    //---------------------------------------------------------------------
    // Fields
    //---------------------------------------------------------------------

    int     sm_version;
    int     sm_count;
    int     smem_alloc_unit;
    int     smem_bytes;

    int     block_threads;
    int     block_warps;
    int     block_allocated_regs;
    int     block_allocated_smem;

    int     max_block_occupancy;        // Maximum threadblock occupancy per SM for the target device


    //---------------------------------------------------------------------
    // Methods
    //---------------------------------------------------------------------

    /**
     * Constructor
     */
    KernelProps() {}

    /**
     * Initializer
     */
    template <typename KernelPtr>
    cudaError_t Init(
        KernelPtr kernel_ptr,
        int block_threads,                      // Number of threads per threadblock
        const CudaProps &cuda_props)            // CUDA properties for a specific device
    {
        cudaError_t error = cudaSuccess;

        do {
            this->block_threads     = block_threads;
            this->sm_count          = cuda_props.sm_count;
            this->sm_version        = cuda_props.sm_version;
            this->smem_alloc_unit   = cuda_props.smem_alloc_unit;
            this->smem_bytes        = cuda_props.smem_bytes;

            // Get kernel attributes
            cudaFuncAttributes kernel_attrs;
            if (error = CubDebug(cudaFuncGetAttributes(&kernel_attrs, kernel_ptr))) break;

            this->block_warps = CUB_ROUND_UP_NEAREST(block_threads / cuda_props.warp_threads, 1);

            int block_allocated_warps = CUB_ROUND_UP_NEAREST(
                block_warps,
                cuda_props.warp_alloc_unit);

            this->block_allocated_regs = (cuda_props.regs_by_block) ?
                CUB_ROUND_UP_NEAREST(
                    block_allocated_warps * kernel_attrs.numRegs * cuda_props.warp_threads,
                    cuda_props.reg_alloc_unit) :
                block_allocated_warps * CUB_ROUND_UP_NEAREST(
                    kernel_attrs.numRegs * cuda_props.warp_threads,
                    cuda_props.reg_alloc_unit);

            this->block_allocated_smem = CUB_ROUND_UP_NEAREST(
                kernel_attrs.sharedSizeBytes,
                cuda_props.smem_alloc_unit);

            int max_block_occupancy = cuda_props.max_sm_blocks;

            int max_warp_occupancy = cuda_props.max_sm_warps / block_warps;

            int max_smem_occupancy = (block_allocated_smem > 0) ?
                    (cuda_props.smem_bytes / block_allocated_smem) :
                    max_block_occupancy;

            int max_reg_occupancy = cuda_props.max_sm_registers / block_allocated_regs;

            this->max_block_occupancy = CUB_MIN(
                CUB_MIN(max_block_occupancy, max_warp_occupancy),
                CUB_MIN(max_smem_occupancy, max_reg_occupancy));

        } while (0);

        return error;
    }


    /**
     *
     */
    int ResidentGridSize()
    {
        return max_block_occupancy * sm_count;
    }


    /**
     * Returns the number of threadblocks to launch for the given problem size.
     * May over/under subscribe the current device based upon heuristics.  Does not
     * the optional max_grid_size limit.
     *
     * Useful for kernels that evenly divide up the work amongst threadblocks.
     */
    int OversubscribedGridSize(
        int schedule_granularity,
        int num_elements,
        int max_grid_size = -1)
    {
        int grid_size;
        int grains = (num_elements + schedule_granularity - 1) / schedule_granularity;

        if (sm_version < 120) {

            // G80/G90: double threadblock occupancy times SM count
            grid_size = 2 * max_block_occupancy * sm_count;

        } else if (sm_version < 200) {

            // GT200: Special sauce.  Start with with full occupancy of all SMs
            grid_size = max_block_occupancy * sm_count;

            int bumps = 0;
            double cutoff = 0.005;

            while (true) {

                double quotient = double(num_elements) /
                    grid_size /
                    schedule_granularity;

                int log = log2(quotient) + 0.5;

                int primary = (1 << log) *
                    grid_size *
                    schedule_granularity;

                double ratio = double(num_elements) / primary;

                if (((ratio < 1.00 + cutoff) && (ratio > 1.00 - cutoff)) ||
                    ((ratio < 0.75 + cutoff) && (ratio > 0.75 - cutoff)) ||
                    ((ratio < 0.50 + cutoff) && (ratio > 0.50 - cutoff)) ||
                    ((ratio < 0.25 + cutoff) && (ratio > 0.25 - cutoff)))
                {
                    if (bumps == 3) {
                        // Bump it up by 33
                        grid_size += 33;
                        bumps = 0;
                    } else {
                        // Bump it down by 1
                        grid_size--;
                        bumps++;
                    }
                    continue;
                }

                break;
            }
        }
        else if (sm_version < 300)
        {
            // Fermi: quadruple threadblock occupancy times SM count
            grid_size = 4 * max_block_occupancy * sm_count;
        }
        else
        {
            // Kepler: quadruple threadblock occupancy times SM count
            grid_size = 4 * max_block_occupancy * sm_count;
        }

        grid_size = (max_grid_size > 0) ? max_grid_size : grid_size;    // Apply override, if specified
        grid_size = CUB_MIN(grains, grid_size);                            // Floor to the number of schedulable grains

        return grid_size;
    }


    /**
     * Return dynamic padding to reduce occupancy to a multiple of the specified base_occupancy
     */
    int DynamicSmemPadding(int base_occupancy)
    {
        div_t div_result = div(max_block_occupancy, base_occupancy);
        if ((!div_result.quot) || (!div_result.rem)) {
            return 0;                                                    // Perfect division (or cannot be padded)
        }

        int target_occupancy = div_result.quot * base_occupancy;
        int required_shared = CUB_ROUND_DOWN_NEAREST(smem_bytes / target_occupancy, smem_alloc_unit);
        int padding = required_shared - block_allocated_smem;

        return padding;
    }
};


} // namespace cub
CUB_NS_POSTFIX
