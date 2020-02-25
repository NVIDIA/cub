/******************************************************************************
 * Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 *IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

/*! \file
 *  \brief Detect the version of the C++ standard used by the compiler.
 */

#pragma once

#include "util_deprecated.cuh"
#include "util_namespace.cuh"

#ifndef CUB_CPP_DIALECT

// MSVC prior to 2015U3 does not expose the C++ dialect and is not supported.
// This is a hard error.
#  ifndef CUB_IGNORE_DEPRECATED_CPP_DIALECT
#    if defined(_MSC_FULL_VER) && _MSC_FULL_VER < 190024210
#      error "MSVC < 2015 Update 3 is not supported by CUB."
#    endif
#  endif

// MSVC does not define __cplusplus correctly. _MSVC_LANG is used instead
// (MSVC 2015U3+ only)
#  ifdef _MSVC_LANG
#    define CUB___CPLUSPLUS _MSVC_LANG
#  else
#    define CUB___CPLUSPLUS __cplusplus
#  endif

// Detect current standard:
#  if CUB___CPLUSPLUS < 201103L
#    define CUB_CPP_DIALECT 2003
#  elif CUB___CPLUSPLUS < 201402L
#    define CUB_CPP_DIALECT 2011
#  elif CUB___CPLUSPLUS < 201703L
#    define CUB_CPP_DIALECT 2014
#  elif CUB___CPLUSPLUS == 201703L
#    define CUB_CPP_DIALECT 2017
#  elif CUB___CPLUSPLUS > 201703L // unknown, but is higher than 2017.
#    define CUB_CPP_DIALECT 2020
#  endif

#  undef CUB___CPLUSPLUS // cleanup

#endif // CUB_CPP_DIALECT
