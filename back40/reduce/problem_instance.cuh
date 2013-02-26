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
 * Encapsulated reduction problem and dispatch
 ******************************************************************************/

#pragma once

#include <cub/cub.cuh>

#include <back40/common/cuda_props.cuh>
#include <back40/reduce/kernel_policy.cuh>
#include <back40/reduce/policy.cuh>
#include <back40/reduce/kernels.cuh>


namespace back40 {
namespace reduce {

using namespace cub;	// Fold cub namespace into back40


/**
 * Encapsulated reduction problem and dispatch
 */
template <
	typename InputIterator,
	typename SizeT,
	typename ReductionOp>
struct ProblemInstance
{
	//---------------------------------------------------------------------
	// Type definitions
	//---------------------------------------------------------------------

	// Type being processed
	typedef typename std::iterator_traits<InputIterator>::value_type T;

	// Type signatures of kernel entrypoints
	typedef void (*UpsweepKernelPtr)	(InputIterator, T*, ReductionOp, WorkDistribution<SizeT>);
	typedef void (*SpineKernelPtr)		(T*, T*, ReductionOp, SizeT);
	typedef void (*SingleKernelPtr)		(InputIterator, T*, ReductionOp, SizeT);


	/**
	 * Kernel properties for a given kernel pointer, opaquely parameterized by
	 * static KernelPolicy details for that kernel
	 */
	template <typename KernelPtr>
	struct KernelProps : back40::KernelProps<KernelPtr>
	{
		int tile_items;

		/**
		 * Initializer
		 */
		template <typename KernelPolicy>
		cudaError_t Init(
			KernelPolicy 			policy,
			KernelPtr 				kernel_ptr,
			const CudaProps 		&cuda_props)
		{
			tile_items = KernelPolicy::TILE_ITEMS;

			return back40::KernelProps<KernelPtr>::Init(
				kernel_ptr,
				KernelPolicy::THREADS,
				cuda_props);
		}
	};


	//---------------------------------------------------------------------
	// Fields
	//---------------------------------------------------------------------

	CudaProps					cuda_props;

	// Problem-specific inputs
	InputIterator 				first;
	T* 							d_result;
	T* 							h_result;
	T*							h_seed;
	SizeT 						num_elements;
	ReductionOp 				reduction_op;
	int 						max_grid_size;


	//---------------------------------------------------------------------
	// Methods
	//---------------------------------------------------------------------


	/**
	 * Constructor
	 */
	ProblemInstance(
		InputIterator 			first,
		T* 						d_result,
		T* 						h_result,
		T*						h_seed,
		SizeT 					num_elements,
		ReductionOp 			reduction_op,
		int 					max_grid_size) :
			// Initializers
			first(first),
			d_result(d_result),
			h_result(h_result),
			h_seed(h_seed),
			num_elements(num_elements),
			reduction_op(reduction_op),
			max_grid_size(max_grid_size)
	{}


	/**
	 * Reduce problem using the specified kernel and dispatch details
	 */
	cudaError_t Reduce(
		const KernelProps<UpsweepKernelPtr>&	upsweep_details,
		const KernelProps<SpineKernelPtr>&		spine_details,
		const KernelProps<SingleKernelPtr>&		single_details,
		bool 									uniform_smem_allocation,
		bool 									uniform_grid_size)
	{
		cudaError_t error = cudaSuccess;
		do {


		} while (0);

		return error;
	}


	/**
	 * Reduce problem using the specified dispatch and kernel policy specializations
	 */
	template <
		typename DispatchPolicy,
		typename UpsweepKernelPolicy,
		typename SpineKernelPolicy,
		typename SingleKernelPolicy>
	cudaError_t Reduce(
		DispatchPolicy			dispatch_policy,
		UpsweepKernelPolicy 	upsweep_policy,
		SpineKernelPolicy 		spine_policy,
		SingleKernelPolicy 		single_policy)
	{
		cudaError_t error = cudaSuccess;
		do {
			// Construct kernel details from policy

			UpsweepKernelPtr upsweep_ptr = UpsweepKernel<UpsweepKernelPolicy>;
			KernelProps<UpsweepKernelPtr>	upsweep_details;
			if (error = upsweep_details.Init(upsweep_policy, upsweep_ptr, cuda_props)) break;

			SpineKernelPtr spine_ptr = SingleKernel<SpineKernelPolicy>;
			KernelProps<SpineKernelPtr> spine_details;
			if (error = spine_details.Init(spine_policy, spine_ptr, cuda_props)) break;

			SingleKernelPtr	single_ptr = SingleKernel<SingleKernelPolicy>;
			KernelProps<SingleKernelPtr> single_details;
			if (error = single_details.Init(single_policy, single_ptr, cuda_props)) break;

			// Reduce problem using the kernel and dispatch details
			if (error = Reduce(
				upsweep_details,
				spine_details,
				single_details,
				DispatchPolicy::UNIFORM_SMEM_ALLOCATION,
				DispatchPolicy::UNIFORM_GRID_SIZE))
					break;

		} while(0);

		return error;
	}


	/**
	 * Reduce problem using the specified policy specialization
	 */
	template <typename Policy>
	cudaError_t Reduce(Policy policy)
	{
		// Reduce problem using the specified dispatch-policy and
		// kernel-policy specializations
		return Reduce(
			policy,
			Policy::Upsweep(),
			Policy::Single(),
			Policy::Single());
	}


	//---------------------------------------------------------------------
	// Pre-configured policy specializations
	//---------------------------------------------------------------------

	template <int TUNED_ARCH>
	struct TunedPolicy;

	// 100
	template <>
	struct TunedPolicy<100> : Policy<
		KernelPolicy<64, 1, 1, PTX_LOAD_NONE, PTX_STORE_NONE, false>,
		KernelPolicy<64, 1, 1, PTX_LOAD_NONE, PTX_STORE_NONE, false>,
		true,
		true>
	{};

	// 130
	template <>
	struct TunedPolicy<130> : Policy<
		KernelPolicy<128, 1, 2, PTX_LOAD_NONE, PTX_STORE_NONE, false>,
		KernelPolicy<128, 1, 2, PTX_LOAD_NONE, PTX_STORE_NONE, false>,
		true,
		true>
	{};

	// 200
	template <>
	struct TunedPolicy<200> : Policy<
		KernelPolicy<128, 2, 2, PTX_LOAD_NONE, PTX_STORE_NONE, false>,
		KernelPolicy<128, 2, 2, PTX_LOAD_NONE, PTX_STORE_NONE, false>,
		true,
		true>
	{};


	// Determine the appropriate tuning arch-id from the arch-id targeted
	// by the active compiler pass.
	enum {
		TUNE_ARCH =
			(PTX_ARCH >= 200) ?
				200 :
				(PTX_ARCH >= 130) ?
					130 :
					100
	};


	// Tuning policies specific to the arch-id of the active compiler
	// pass.  (The policy's type signature is "opaque" to the target
	// architecture.)
	struct TunedUpsweep : TunedPolicy<TUNE_ARCH>::Upsweep {} tuned_upsweep;
	struct TunedSingle : TunedPolicy<TUNE_ARCH>::Single {} tuned_single;


	/**
	 * Reduce problem using pre-configured policy specializations
	 */
	cudaError_t Reduce()
	{
		if (cuda_props.ptx_version >= 200) {

			return Reduce(TunedPolicy<200>(), tuned_upsweep, tuned_single, tuned_single);

		} else if (cuda_props.ptx_version >= 130) {

			return Reduce(TunedPolicy<130>(), tuned_upsweep, tuned_single, tuned_single);

		} else {

			return Reduce(TunedPolicy<100>(), tuned_upsweep, tuned_single, tuned_single);
		}
	}
};




}// namespace reduce
}// namespace back40

