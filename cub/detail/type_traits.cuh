/******************************************************************************
 * Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

/**
 * \file
 * Wrappers and extensions around <type_traits> utilities.
 */

#pragma once

#include <cub/util_cpp_dialect.cuh>
#include <cub/util_namespace.cuh>

#include <cuda/std/type_traits>

#include <iterator>


CUB_NAMESPACE_BEGIN
namespace detail {

template <bool Test, class T1, class T2>
using conditional_t = typename cuda::std::conditional<Test, T1, T2>::type;

template <typename Iterator>
using value_t = typename std::iterator_traits<Iterator>::value_type;

/**
 * The output value type
 * type = (if IteratorT's value type is void) ?
 * ... then the FallbackT,
 * ... else the IteratorT's value type
 */
template <typename IteratorT, typename FallbackT>
using non_void_value_t =
  cub::detail::conditional_t<std::is_same<value_t<IteratorT>, void>::value,
                             FallbackT,
                             value_t<IteratorT>>;

template <typename Invokable, typename... Args>
using invoke_result_t =
#if CUB_CPP_DIALECT < 2017
  typename ::cuda::std::result_of<Invokable(Args...)>::type;
#else // 2017+
  ::cuda::std::invoke_result_t<Invokable, Args...>;
#endif

/// The type of intermediate accumulator (according to P2322R6)
template <typename Invokable, typename InitT, typename InputT>
using accumulator_t = 
  typename ::cuda::std::decay<invoke_result_t<Invokable, InitT, InputT>>::type;

} // namespace detail
CUB_NAMESPACE_END
