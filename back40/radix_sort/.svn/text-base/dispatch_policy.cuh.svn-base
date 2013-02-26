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
 * Dispatch policy types
 ******************************************************************************/

#pragma once

#include "../ns_wrapper.cuh"

BACK40_NS_PREFIX
namespace back40 {
namespace radix_sort {


/**
 * Dispatch policy type
 */
template <
    int                 _UPSWEEP_MIN_BLOCK_OCCUPANCY,
    int                 _DOWNSWEEP_MIN_BLOCK_OCCUPANCY,
    int                 _HYBRID_MIN_BLOCK_OCCUPANCY>
struct DispatchPolicy
{
    enum
    {
        UPSWEEP_MIN_BLOCK_OCCUPANCY           = _UPSWEEP_MIN_BLOCK_OCCUPANCY,
        DOWNSWEEP_MIN_BLOCK_OCCUPANCY         = _DOWNSWEEP_MIN_BLOCK_OCCUPANCY,
        HYBRID_MIN_BLOCK_OCCUPANCY            = _HYBRID_MIN_BLOCK_OCCUPANCY,
    };

};




}// namespace radix_sort
}// namespace back40
BACK40_NS_POSTFIX
