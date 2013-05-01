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
 * Common C/C++ macro utilities
 ******************************************************************************/

#pragma once

#include "util_namespace.cuh"

/// Optional outer namespace(s)
CUB_NS_PREFIX

/// CUB namespace
namespace cub {


/**
 * \addtogroup UtilModule
 * @{
 */


/**
 * Select maximum(a, b)
 */
#define CUB_MAX(a, b) (((a) > (b)) ? (a) : (b))


/**
 * Select minimum(a, b)
 */
#define CUB_MIN(a, b) (((a) < (b)) ? (a) : (b))

/**
 * Quotient of x/y rounded down to nearest integer
 */
#define CUB_QUOTIENT_FLOOR(x, y) ((x) / (y))

/**
 * Quotient of x/y rounded up to nearest integer
 */
#define CUB_QUOTIENT_CEILING(x, y) (((x) + (y) - 1) / (y))

/**
 * x rounded up to the nearest multiple of y
 */
#define CUB_ROUND_UP_NEAREST(x, y) ((((x) + (y) - 1) / (y)) * y)


/**
 * x rounded down to the nearest multiple of y
 */
#define CUB_ROUND_DOWN_NEAREST(x, y) (((x) / (y)) * y)

/**
 * Return character string for given type
 */
#define CUB_TYPE_STRING(type) ""#type


/** @} */       // end group UtilModule

}               // CUB namespace
CUB_NS_POSTFIX  // Optional outer namespace(s)
