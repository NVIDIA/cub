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

/**
 * \file
 * Utilities for selecting a type corresponding to a target PTX version.
 */

#pragma once

#include <cub/util_namespace.cuh>

#include <cub/detail/detect_cuda_runtime.cuh>
#include <cub/detail/type_list.cuh>
#include <cub/detail/type_wrapper.cuh>

CUB_NAMESPACE_BEGIN
namespace detail
{

/**
 * Non-intrusive utility to tag a type `T` with a `ptx_arch` member. This allows
 * T to be used in with `ptx_dispatch` and doesn't modify T.
 *
 * ```
 * // Existing classes with PTX-dependent implementation details:
 * struct Foo600 { ... };
 * struct Foo800 { ... };
 *
 * // Tagged types with `ptx_arch` members:
 * using DispatchableFoo600 = cub::detail::ptx_tag<600, Foo600>;
 * using DispatchableFoo800 = cub::detail::ptx_tag<800, Foo800>;
 *
 * // `type_list` for use with `ptx_dispatch`:
 * using FooList = cub::detail::type_list<DispatchableFoo800,
 *                                        DispatchableFoo600>;
 * ```
 *
 * \tparam PtxArch 3-digit PTX architecture version (e.g. 800 for Ampere)
 * \tparam T The type to tag with `::ptx_arch = PtxArch`.
 */
template <int PtxArch, typename T>
struct ptx_tag : T
{
  static constexpr int ptx_arch = PtxArch;
};

/**
 * Intrusive version of `ptx_tag`; use as a base class to define `ptx_arch`.
 * This allows the subclass to be used with `ptx_dispatch`.
 *
 * ```
 * // Existing classes with PTX-dependent implementation details:
 * struct Foo600 : cub::detail::ptx_base<600> { ... };
 * struct Foo800 : cub::detail::ptx_base<800> { ... };
 *
 * // `type_list` for use with `ptx_dispatch`:
 * using FooList = cub::detail::type_list<Foo800, Foo600>;
 * ```
 *
 * \tparam PtxArch 3-digit PTX architecture version (e.g. 800 for Ampere)
 */
template <int PtxArch>
struct ptx_base
{
  static constexpr int ptx_arch = PtxArch;
};

/** The valid execution spaces for a `ptx_dispatch` functor. */
enum class exec_space
{
  host,
  device,
  host_device
};

/** The exec_space to use for CUB_RUNTIME_FUNCTIONS. @{ */
#ifdef CUB_RUNTIME_ENABLED
static constexpr exec_space runtime_exec_space = exec_space::host_device;
#else
static constexpr exec_space runtime_exec_space = exec_space::host;
#endif

/**
 * Specialize a functor based on the current target PTX architecture.
 *
 * Inputs:
 *
 * - `PtxTypeList`, a `cub::detail::type_list<...>` of types with `ptx_arch`
 *   members (see `ptx_tag`/`ptx_base`).
 * - `Functor`, an invokable object that will be called with a
 *   `cub::detail::type_wrapper<PtxType>` argument.
 * - `ExecSpace`, a tag identifying the execution space of `Functor` (`host`,
 *   `device`, or `host_device`). Optional; defaults to `host_device`.
 *
 * `ptx_dispatch` will determine the best matching type in `PtxTypeList` for the
 * target PTX version, and then call `Functor` with a `type_wrapper<PtxType>`.
 * The best matching `PtxType` is the type with the largest `ptx_arch` value
 * that does not exceed the target PTX version.
 *
 * The target PTX version is detected automatically.
 *
 * Additional arguments may be forwarded to `Functor` by passing them to
 * `ptx_dispatch::exec`.
 *
 * Example:
 *
 * ```
 * struct ptx_type_350 : cub::detail::ptx_base<350> { ... };
 * struct ptx_type_600 : cub::detail::ptx_base<600> { ... };
 * struct ptx_type_800 : cub::detail::ptx_base<800> { ... };
 *
 * struct functor
 * {
 *   template <typename PtxType>
 *   void operator()(cub::detail::type_wrapper<PtxType>, int var1, float var2)
 *   {
 *     // ...
 *   }
 * };
 *
 * using ptx_types = cub::detail::type_list<ptx_type_800,
 *                                          ptx_type_600,
 *                                          ptx_type_350>;
 *
 * void foo(int var1, float var2)
 * {
 *   using dispatcher_t = cub::detail::ptx_dispatch<ptx_types>;
 *   // Invoke functor with an appropriate ptx_type.
 *   dispatcher_t::exec(functor{}, var1, var2);
 * }
 * ```
 *
 * Caveats:
 *
 * For defined behavior, `PtxTypes` must:
 * - Arrange elements from highest `ptx_arch` to lowest.
 * - Contain no two elements with the same `ptx_arch` value.
 * - Contain an element suitable for the lowest ptx arch supported by the
 *   application.
 *
 * @sa CUB_PTX_TARGETS
 */
template <typename PtxTypeList,
          exec_space ExecSpace = exec_space::host_device>
struct ptx_dispatch;

} // namespace detail
CUB_NAMESPACE_END

#include <cub/detail/ptx_dispatch_impl.cuh>
