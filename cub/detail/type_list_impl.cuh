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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 ******************************************************************************/

#pragma once

#include <cub/detail/type_list.cuh>

#include <type_traits>

CUB_NAMESPACE_BEGIN
namespace detail
{
namespace tl
{

namespace contains_impl
{

template <typename Type, typename... OldTypes>
struct impl;

template <typename Type, typename HeadType, typename... TailTypes>
struct impl<Type, HeadType, TailTypes...>
{
  static constexpr bool value = std::is_same<Type, HeadType>::value ||
                                impl<Type, TailTypes...>::value;
};

template <typename Type>
struct impl<Type>
{
  static constexpr bool value = false;
};

} // namespace contains_impl

template <typename Type, typename... OldTypes>
struct contains<type_list<OldTypes...>, Type>
    : std::integral_constant<bool, contains_impl::impl<Type, OldTypes...>::value>
{};

template <typename NewType, typename... OldTypes>
struct prepend<type_list<OldTypes...>, NewType>
{
  using type = cub::detail::type_list<NewType, OldTypes...>;
};

template <typename NewType, typename... OldTypes>
struct append<type_list<OldTypes...>, NewType>
{
  using type = cub::detail::type_list<OldTypes..., NewType>;
};

} // namespace tl
} // namespace detail
CUB_NAMESPACE_END
