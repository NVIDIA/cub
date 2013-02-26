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

#include <cub/cub.cuh>

namespace back40 {
namespace reduce {

using namespace cub;	// Fold cub namespace into back40


/**
 * Policy type for specializing reduction kernels.  Parameterizations
 * of this policy type encapsulate tuning decisions (which are reflected via
 * the static fields).
 *
 * Used to bind generic kernel code to a specific problem-type, SM-version,
 * etc.
 */
template <
	int 			_THREADS,
	int 			_STRIPS_PER_THREAD,
	int 			_ELEMENTS_PER_STRIP,
	PtxLoadModifier 	_LOAD_MODIFIER,
	PtxStoreModifier 	_STORE_MODIFIER,
	bool 			_WORK_STEALING>
struct KernelPolicy
{
	enum {
		THREADS					= _THREADS,
		STRIPS_PER_THREAD		= _STRIPS_PER_THREAD,
		ELEMENTS_PER_STRIP		= _ELEMENTS_PER_STRIP,
		TILE_ITEMS			= THREADS * STRIPS_PER_THREAD * ELEMENTS_PER_STRIP,
	};

	static const PtxLoadModifier 	LOAD_MODIFIER 	= _LOAD_MODIFIER;
	static const PtxStoreModifier 	STORE_MODIFIER 	= _STORE_MODIFIER;
	static const bool 			WORK_STEALING	= _WORK_STEALING;
};

		

}// namespace reduce
}// namespace back40

