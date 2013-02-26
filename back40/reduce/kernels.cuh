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
 * Reduction kernel tuning policy and entry-points
 ******************************************************************************/

#pragma once

#include <cub/cub.cuh>
#include <back40/reduce/cta.cuh>

namespace back40 {
namespace reduce {

using namespace cub;	// Fold cub namespace into back40


/**
 * Upsweep reduction kernel entry point
 */
template <
	typename KernelPolicy,
	typename InputIterator,
	typename OutputIterator,
	typename ReductionOp,
	typename SizeT>
__global__ void UpsweepKernel(
	InputIterator					d_in,
	OutputIterator			 		d_out,
	ReductionOp						reduction_op,
	WorkDistribution<SizeT> 		work_distribution)
{
	// CTA abstraction type
	typedef Block<
		KernelPolicy,
		InputIterator,
		OutputIterator,
		ReductionOp,
		SizeT> Block;

	// Declare shared memory for the CTA
	__shared__ typename Block::SmemStorage smem_storage;

	// Create CTA and have it iteratively process input tiles
	Block cta(smem_storage, d_in, d_out, reduction_op, work_distribution);
	cta.ProcessTiles();
}


/**
 * Single-CTA reduction kernel entry point
 */
template <
	typename KernelPolicy,
	typename InputIterator,
	typename OutputIterator,
	typename ReductionOp,
	typename SizeT>
__global__ void SingleKernel(
	InputIterator					d_in,
	OutputIterator			 		d_out,
	ReductionOp						reduction_op,
	SizeT							num_elements)
{
	// CTA abstraction type
	typedef Block<
		KernelPolicy,
		InputIterator,
		OutputIterator,
		ReductionOp,
		SizeT> Block;

	// Declare shared memory for the CTA
	__shared__ typename Block::SmemStorage smem_storage;

	// Create CTA and have it iteratively process input tiles
	Block cta(smem_storage, d_in, d_out, reduction_op, num_elements);
	cta.ProcessTiles();
}


} // namespace reduce
} // namespace back40

