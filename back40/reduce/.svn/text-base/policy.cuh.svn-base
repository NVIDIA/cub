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
 * Policy types for GPU reduction primitives
 ******************************************************************************/

#pragma once

namespace back40 {
namespace reduce {

using namespace cub;	// Fold cub namespace into back40


/**
 * Reduction policy for GPU reduction primitives.
 *
 * Encapsulates:
 *   - Kernel tuning parameters for specializing upsweep and spine kernels
 *   - Dispatch tuning parameters.
 */
template <
	typename 		UpsweepKernelPolicy,
	typename 		SingleKernelPolicy,
	bool 			_UNIFORM_SMEM_ALLOCATION,
	bool 			_UNIFORM_GRID_SIZE>
struct Policy
{
	// Kernel policies
	typedef UpsweepKernelPolicy 		Upsweep;
	typedef SingleKernelPolicy 			Single;

	// Dispatch tuning details
	enum {
		UNIFORM_SMEM_ALLOCATION 	= _UNIFORM_SMEM_ALLOCATION,
		UNIFORM_GRID_SIZE 			= _UNIFORM_GRID_SIZE,
	};
};
		

}// namespace reduce
}// namespace back40

