/******************************************************************************
 * 
 * Copyright (c) 2011-2012, Duane Merrill.  All rights reserved.
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
 * Kernel function properties
 ******************************************************************************/

#pragma once

#include <cub/host/cuda_props.cuh>
#include <cub/type_utils.cuh>

namespace cub {



/**
 * Encapsulation of kernel properties for a combination of {device, CTA size}
 */
template <typename KernelPtr>
struct KernelProps
{
	//---------------------------------------------------------------------
	// Fields
	//---------------------------------------------------------------------

	KernelPtr 					kernel_ptr;

	int 						sm_version;
	int 						sm_count;
	int							smem_alloc_unit;
	int							smem_bytes;

	int 						cta_threads;
	int 						cta_warps;
	int							cta_allocated_regs;
	int							cta_allocated_smem;

	int 						max_cta_occupancy;		// Maximum CTA occupancy per SM for the target device


	//---------------------------------------------------------------------
	// Methods
	//---------------------------------------------------------------------

	/**
	 * Initializer
	 */
	cudaError_t Init(
		KernelPtr kernel_ptr,
		int cta_threads,					// Number of threads per CTA
		const CudaProps &cuda_props)	// CUDA properties for a specific device
	{
		cudaError_t error = cudaSuccess;

		do {
			this->kernel_ptr			= kernel_ptr;
			this->cta_threads 			= cta_threads;
			this->sm_count 				= cuda_props.sm_count;
			this->sm_version 			= cuda_props.sm_version;
			this->smem_alloc_unit 		= cuda_props.smem_alloc_unit;
			this->smem_bytes 			= cuda_props.smem_bytes;

			// Get kernel attributes
			cudaFuncAttributes kernel_attrs;
			if (error = Perror(cudaFuncGetAttributes(&kernel_attrs, kernel_ptr),
				"cudaFuncGetAttributes failed", __FILE__, __LINE__)) break;

			this->cta_warps = CUB_ROUND_UP_NEAREST(cta_threads / cuda_props.warp_threads, 1);

			int cta_allocated_warps = CUB_ROUND_UP_NEAREST(
				cta_warps,
				cuda_props.warp_alloc_unit);

			this->cta_allocated_regs = (cuda_props.regs_by_block) ?
				CUB_ROUND_UP_NEAREST(
					cta_allocated_warps * kernel_attrs.numRegs * cuda_props.warp_threads,
					cuda_props.reg_alloc_unit) :
				cta_allocated_warps * CUB_ROUND_UP_NEAREST(
					kernel_attrs.numRegs * cuda_props.warp_threads,
					cuda_props.reg_alloc_unit);

			this->cta_allocated_smem = CUB_ROUND_UP_NEAREST(
				kernel_attrs.sharedSizeBytes,
				cuda_props.smem_alloc_unit);

			int max_block_occupancy = cuda_props.max_sm_ctas;

			int max_warp_occupancy = cuda_props.max_sm_warps / cta_warps;

			int max_smem_occupancy = (cta_allocated_smem > 0) ?
					(cuda_props.smem_bytes / cta_allocated_smem) :
					max_block_occupancy;

			int max_reg_occupancy = cuda_props.max_sm_registers / cta_allocated_regs;

			this->max_cta_occupancy = CUB_MIN(
				CUB_MIN(max_block_occupancy, max_warp_occupancy),
				CUB_MIN(max_smem_occupancy, max_reg_occupancy));

		} while (0);

		return error;
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
		int max_grid_size = 0)
	{
		int grid_size;
		int grains = (num_elements + schedule_granularity) / schedule_granularity;

		if (sm_version < 120) {

			// G80/G90: double CTA occupancy times SM count
			grid_size = 2 * max_cta_occupancy * sm_count;

		} else if (sm_version < 200) {

			// GT200: Special sauce.  Start with with full occupancy of all SMs
			grid_size = max_cta_occupancy * sm_count;

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

		} else {

			// GF10x: quadruple CTA occupancy times SM count
			grid_size = 4 * max_cta_occupancy * sm_count;
		}

		grid_size = (max_grid_size > 0) ? max_grid_size : grid_size;	// Apply override, if specified
		grid_size = CUB_MIN(grains, grid_size);							// Floor to the number of schedulable grains

		return grid_size;
	}


	/**
	 * Return dynamic padding to reduce occupancy to a multiple of the specified base_occupancy
	 */
	int DynamicSmemPadding(int base_occupancy)
	{
		div_t div_result = div(max_cta_occupancy, base_occupancy);
		if ((!div_result.quot) || (!div_result.rem)) {
			return 0;													// Perfect division (or cannot be padded)
		}

		int target_occupancy = div_result.quot * base_occupancy;
		int required_shared = CUB_ROUND_DOWN_NEAREST(smem_bytes / target_occupancy, smem_alloc_unit);
		int padding = required_shared - cta_allocated_smem;

		return padding;
	}
};



} // namespace cub

