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

#include "../../util/kernel_props.cuh"
#include "../../util/cta_progress.cuh"
#include "../../util/ns_wrapper.cuh"

#include "../../radix_sort/cta/cta_upsweep_pass.cuh"

BACK40_NS_PREFIX
namespace back40 {
namespace radix_sort {
namespace kernel {


/**
 * Kernel entry point
 */
template <
	typename CtaUpsweepPassPolicy,
	typename SizeT,
	typename KeyType>
__launch_bounds__ (
	CtaUpsweepPassPolicy::CTA_THREADS,
	CtaUpsweepPassPolicy::MIN_CTA_OCCUPANCY)
__global__
void UpsweepKernel(
	SizeT 								*d_spine,
	KeyType 							*d_keys_in,
	util::CtaWorkDistribution<SizeT> 	cta_work_distribution,
	unsigned int 						current_bit)
{
	// CTA abstraction type
	typedef CtaUpsweepPass<CtaUpsweepPassPolicy, SizeT, KeyType> CtaUpsweepPass;

	// Shared data structures
	__shared__ typename CtaUpsweepPass::SmemStorage 	cta_smem_storage;
	__shared__ util::CtaProgress<SizeT, TILE_ELEMENTS> 	cta_progress;

	// Determine our threadblock's work range
	if (threadIdx.x == 0)
	{
		cta_progress.Init(cta_work_distribution);
	}

	// Sync to acquire work range
	__syncthreads();

	// Compute bin-count for each radix digit (valid in the first RADIX_DIGITS threads)
	SizeT bin_count;
	CtaUpsweepPass::Upsweep(
		cta_smem_storage,
		d_keys_in + cta_progress.cta_offset,
		current_bit,
		cta_progress.num_elements,
		bin_count);

	// Write out the bin_count reductions
	if (threadIdx.x < RADIX_DIGITS)
	{
		int spine_bin_offset = (gridDim.x * threadIdx.x) + blockIdx.x;

		util::io::ModifiedStore<CtaUpsweepPassPolicy::STORE_MODIFIER>::St(
			bin_count,
			d_spine + spine_bin_offset);
	}
}


/**
 * Upsweep kernel properties
 */
template <
	typename SizeT,
	typename KeyType>
struct UpsweepKernelProps : util::KernelProps
{
	// Kernel function type
	typedef void (*KernelFunc)(
		SizeT*,
		KeyType*,
		util::CtaWorkDistribution<SizeT>,
		unsigned int);

	// Fields
	KernelFunc 					kernel_func;
	int 						tile_elements;
	cudaSharedMemConfig 		sm_bank_config;

	/**
	 * Initializer
	 */
	template <
		typename CtaUpsweepPassPolicy,
		typename OpaqueCtaUpsweepPassPolicy>
	cudaError_t Init(int sm_arch, int sm_count)
	{
		// Initialize fields
		kernel_func 			= UpsweepKernel<OpaqueCtaUpsweepPassPolicy>;
		tile_elements 			= CtaUpsweepPassPolicy::TILE_ELEMENTS;
		sm_bank_config 			= CtaUpsweepPassPolicy::SMEM_CONFIG;

		// Initialize super class
		return util::KernelProps::Init(
			kernel_func,
			CtaUpsweepPassPolicy::CTA_THREADS,
			sm_arch,
			sm_count);
	}

	/**
	 * Initializer
	 */
	template <typename CtaUpsweepPassPolicy>
	cudaError_t Init(int sm_arch, int sm_count)
	{
		return Init<CtaUpsweepPassPolicy, CtaUpsweepPassPolicy>(sm_arch, sm_count);
	}
};


} // namespace kernel
} // namespace radix_sort
} // namespace back40
BACK40_NS_POSTFIX
