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
 * Reduction primitives
 ******************************************************************************/

#pragma once

#include <iterator>

#include <cub/cub.cuh>

#include <back40/reduce/cta.cuh>
#include <back40/reduce/kernel_policy.cuh>
#include <back40/reduce/kernels.cuh>
#include <back40/reduce/policy.cuh>
#include <back40/reduce/problem_instance.cuh>

namespace back40 {


/**
 * @return cudaSuccess on success, error enumeration otherwise
 */
template <
	typename InputIterator,
	typename ReductionOp>
cudaError_t Reduce(
	InputIterator 												d_in,
	typename std::iterator_traits<InputIterator>::value_type* 	d_result,
	typename std::iterator_traits<InputIterator>::value_type* 	h_result,
	typename std::iterator_traits<InputIterator>::value_type* 	h_seed,
	int 														num_elements,
	ReductionOp 												reduction_op,
	int 														max_grid_size = 0)
{
	reduce::ProblemInstance<InputIterator, int, ReductionOp> problem(
		d_in,
		d_result,
		h_result,
		h_seed,
		num_elements,
		reduction_op,
		max_grid_size);

	return problem.Reduce();
}


/**
 * @return cudaSuccess on success, error enumeration otherwise
 */
template <typename InputIterator>
cudaError_t Reduce(
	InputIterator 												d_in,
	typename std::iterator_traits<InputIterator>::value_type* 	d_result,
	typename std::iterator_traits<InputIterator>::value_type* 	h_result,
	typename std::iterator_traits<InputIterator>::value_type* 	h_seed,
	int 														num_elements,
	int 														max_grid_size = 0)
{
	typedef typename std::iterator_traits<InputIterator>::value_type T;
	cub::Sum<T> reduction_op;

	return Reduce(
		d_in,
		d_result,
		h_result,
		h_seed,
		num_elements,
		reduction_op,
		max_grid_size);
}

/**
 * @return cudaSuccess on success, error enumeration otherwise
 */
template <
	typename Policy,
	typename InputIterator,
	typename ReductionOp>
cudaError_t Reduce(
	InputIterator 												d_in,
	typename std::iterator_traits<InputIterator>::value_type* 	d_result,
	typename std::iterator_traits<InputIterator>::value_type* 	h_result,
	typename std::iterator_traits<InputIterator>::value_type* 	h_seed,
	int 														num_elements,
	ReductionOp 												reduction_op,
	Policy														policy,
	int 														max_grid_size = 0)
{
	reduce::ProblemInstance<InputIterator, int, ReductionOp> problem(
		d_in,
		d_result,
		h_result,
		h_seed,
		num_elements,
		reduction_op,
		max_grid_size);

	return problem.Reduce(policy);
}


}// namespace back40

