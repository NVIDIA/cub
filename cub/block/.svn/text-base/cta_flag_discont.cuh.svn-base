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

/**
 * \file
 * The cub::CtaFlagDiscontinuities type provides operations for flagging discontinuities within a list of data items partitioned across CTA threads.
 */

#pragma once

#include "../device_props.cuh"
#include "../type_utils.cuh"
#include "../operators.cuh"
#include "../ns_wrapper.cuh"

CUB_NS_PREFIX

/// CUB namespace
namespace cub {

/**
 * \addtogroup SimtCoop
 * @{
 */


/**
 * \brief The CtaFlagDiscontinuities type provides operations for flagging discontinuities within a list of data items partitioned across CTA threads.
 *
 * <b>Overview</b>
 * \par
 * The operations exposed by CtaFlagDiscontinuities allow CTAs to set "head flags" for data elements that
 * are different from their predecessor (as specified by a binary boolean operator).  flag discontinuities reorganize data items between
 * threads, converting between (or scattering to) the following partitioning arrangements:
 * -# <b><em>blocked</em> arrangement</b>.  The aggregate tile of items is partitioned
 *   evenly across threads in "blocked" fashion with thread<sub><em>i</em></sub>
 *   owning the <em>i</em><sup>th</sup> segment of consecutive elements.
 * -# <b><em>striped</em> arrangement</b>.  The aggregate tile of items is partitioned across
 *   threads in "striped" fashion, i.e., the \p ITEMS_PER_THREAD items owned by
 *   each thread have logical stride \p CTA_THREADS between them.
 *
 * \tparam T                    The data type to be exchanged.
 * \tparam CTA_THREADS          The CTA size in threads.
 * \tparam ITEMS_PER_THREAD     The number of items partitioned onto each thread.
 *
 * <b>Important Features and Considerations</b>
 * \par
 * - After any operation, a subsequent CTA barrier (<tt>__syncthreads()</tt>) is
 *   required if the supplied CtaScan::CtaExchange is to be reused/repurposed by the CTA.
 * - Zero bank conflicts for most types.
 *
 * <b>Algorithm</b>
 * \par
 * Regardless of the initial blocked/striped arrangement, CTA threads scatter
 * items into shared memory in <em>blocked</em>, taking care to include
 * one item of padding for every shared memory bank's worth of items.  After a
 * barrier, items are gathered in the desired blocked/striped arrangement.
 * <br>
 * <br>
 * \image html raking.png
 * <center><b>A CTA of 16 threads performing a conflict-free <em>blocked</em> gathering of 64 exchanged items.</b></center>
 * <br>
 *
 */
template <
	int 		CTA_THREADS,			// The CTA size in threads
	typename 	T,						// The input type for which we are detecting duplicates
	int 		CTA_STRIPS = 1>			// When strip-mining, the number of CTA-strips per tile
struct CtaFlagDiscontinuities
{

};


/** @} */       // end of SimtCoop group

} // namespace cub
CUB_NS_POSTFIX
