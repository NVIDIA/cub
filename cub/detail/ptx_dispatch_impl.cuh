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

#include <cub/detail/ptx_dispatch.cuh>

#include <cub/detail/ptx_targets.cuh>
#include <cub/detail/type_list.cuh>

#include <cub/util_device.cuh>
#include <cub/util_namespace.cuh>

#include <type_traits>
#include <utility>

CUB_NAMESPACE_BEGIN
namespace detail
{

/** Static dispatch tags for exec_space. @{ */
template <exec_space ExecSpace>
using exec_space_tag = std::integral_constant<exec_space, ExecSpace>;

using host_tag        = exec_space_tag<exec_space::host>;
using device_tag      = exec_space_tag<exec_space::device>;
using host_device_tag = exec_space_tag<exec_space::host_device>;
/** @} */

/** std::enable_if wrapper that detects if the two exec_spaces are equal. */
template <exec_space ES1, exec_space ES2>
using enable_exec_space_t = typename std::enable_if<ES1 == ES2>::type;

/** Placeholder for invalid ptx_arch_lookup results. */
struct no_ptx_type : ptx_base<0>
{};

/**
 * Compile-time lookup of types based on PTX version.
 *
 * Given:
 *
 * - `TargetPtxArch`, a 3-digit PTX architecture version, e.g. 750 for Turing.
 * - `PtxTypeList`, a `type_list` of types with `::ptx_arch` members defined to
 *   3-digit PTX architecture versions.
 *
 * Defines `type` to the `PtxType` with the highest `ptx_arch` that does not
 * exceed `TargetPtx`.
 *
 * Each `PtxType` must have a unique `ptx_arch` and at least one
 * `PtxType::ptx_arch` must be less-than-or-equal-to TargetPtxArch, otherwise
 * the behavior is undefined. The elements of `PtxTypes` must be sorted by
 * descending `ptx_arch` values. The results are otherwise undefined.
 *
 * @{
 */
template <int TargetPtxArch, typename PtxTypeList>
struct ptx_arch_lookup;

template <int TargetPtxArch, typename PtxType, typename... PtxTypeTail>
struct ptx_arch_lookup<TargetPtxArch, type_list<PtxType, PtxTypeTail...>>
{
private:
  using this_type =
    typename std::conditional<(PtxType::ptx_arch <= TargetPtxArch),
                              PtxType,
                              no_ptx_type>::type;

  using next_lookup = ptx_arch_lookup<TargetPtxArch, type_list<PtxTypeTail...>>;
  using next_type   = typename next_lookup::type;

public:
  using type =
    typename std::conditional<(this_type::ptx_arch > next_type::ptx_arch),
                              this_type,
                              next_type>::type;
};

template <int TargetPtxArch>
struct ptx_arch_lookup<TargetPtxArch, type_list<>>
{
  using type = no_ptx_type;
};
/** @} */

// implementation detail for restrict_ptx_types
template <typename PtxTypeList, int... PtxTargets>
struct restrict_ptx_types_impl;

template <typename PtxTypeList, int PtxTargetHead, int... PtxTargetTail>
struct restrict_ptx_types_impl<PtxTypeList, PtxTargetHead, PtxTargetTail...>
{
  using tail_ptx_types =
    typename restrict_ptx_types_impl<PtxTypeList, PtxTargetTail...>::type;
  using current_ptx_t =
    typename ptx_arch_lookup<PtxTargetHead, PtxTypeList>::type;
  static constexpr bool ptx_type_is_new =
    !cub::detail::tl::contains<tail_ptx_types, current_ptx_t>::value;
  using type = typename std::conditional<
    ptx_type_is_new,
    typename cub::detail::tl::append<tail_ptx_types, current_ptx_t>::type,
    tail_ptx_types>::type;
};

template <typename PtxTypeList>
struct restrict_ptx_types_impl<PtxTypeList>
{
  using type = cub::detail::type_list<>;
};

/**
 * Use the `CUB_PTX_TARGETS` macro to reduce the set of types in `PtxTypeList`
 * to only contain PtxTypes that are actually used by the current targets.
 * @{
 */
template <typename PtxTypeList>
struct restrict_ptx_types;

template <typename... PtxTypes>
struct restrict_ptx_types<cub::detail::type_list<PtxTypes...>>
{
  using type =
    typename restrict_ptx_types_impl<cub::detail::type_list<PtxTypes...>,
                                     CUB_PTX_TARGETS>::type;
};
/**@}*/

/** Used to silence warnings about unused variables in parameter packs.
 * @{
 */
__host__ __device__ __forceinline__
void mark_as_used()
{
}

template <typename T>
__host__ __device__ __forceinline__
void mark_as_used(T &&t)
{
  (void)t;
}

template <typename T, typename... Ts>
__host__ __device__ __forceinline__
void mark_as_used(T &&t, Ts &&...ts)
{
  (void)t;
  mark_as_used(std::forward<Ts>(ts)...);
}
/** @} */

/**
 * Instantiate (but do not execute) `Functor(type_wrapper<PtxType>{}, Args...)`
 * in the appropriate `ExecSpace`.
 *
 * To use, just instantiate and call an object of this type:
 *
 * ```
 * instantiate_functor<...>{}();
 * ```
 */
template <cub::detail::exec_space ExecSpace,
          typename Functor,
          typename PtxType,
          typename... Args>
struct instantiate_functor
{

#pragma nv_exec_check_disable
  __host__ __device__ __forceinline__
  void operator()()
  {
    // call_functor instantiates the functor's call operator with the
    // appropriate arguments in the correct exec_space.
    // We don't call call_functor; we just generate a pointer
    // so that instantiation happens without execution.
    auto func = &instantiate_functor::call_functor<ExecSpace>;
    mark_as_used(func);
  }

private:
  // never invoked, just used to instantiate the functor's operator():
  template <cub::detail::exec_space ES,
            typename = enable_exec_space_t<ES, exec_space::host>>
  __host__
  void call_functor(host_tag,
                    Functor &&functor,
                    Args &&...args)
  {
    functor(cub::detail::type_wrapper<PtxType>{}, std::forward<Args>(args)...);
  }

  template <cub::detail::exec_space ES,
            typename = enable_exec_space_t<ES, exec_space::device>>
  __device__
  void call_functor(device_tag,
                    Functor &&functor,
                    Args &&...args)
  {
    functor(cub::detail::type_wrapper<PtxType>{}, std::forward<Args>(args)...);
  }

  template <cub::detail::exec_space ES,
            typename = enable_exec_space_t<ES, exec_space::host_device>>
  __host__ __device__
  void call_functor(host_device_tag,
                    Functor &&functor,
                    Args &&...args)
  {
    functor(cub::detail::type_wrapper<PtxType>{}, std::forward<Args>(args)...);
  }
};

namespace check_ptx_type_list_impl
{

// true_type if PtxType::ptx_arch is defined and >= 0.
template <typename PtxType, typename = void>
struct has_ptx_arch : std::false_type
{};

template <typename PtxType>
struct has_ptx_arch<PtxType,
                    typename std::enable_if<(PtxType::ptx_arch >= 0)>::type>
    : std::true_type
{};

// true_type if both PtxTs define ptx_arch and PtxT1's ptx_arch is larger than
// PtxT2's ptx_arch.
template <typename PtxT1, typename PtxT2, typename = void>
struct order_valid : std::false_type
{};

template <typename PtxT1, typename PtxT2>
struct order_valid<PtxT1,
                   PtxT2,
                   typename std::enable_if<has_ptx_arch<PtxT1>::value &&
                                           has_ptx_arch<PtxT2>::value>::type>
    : std::integral_constant<bool, (PtxT1::ptx_arch > PtxT2::ptx_arch)>
{};

// true_type if the PtxTypeList is valid.
template <typename PtxTypeList>
struct is_valid;

template <>
struct is_valid<type_list<>> : std::true_type
{};

template <typename PtxType>
struct is_valid<type_list<PtxType>> : has_ptx_arch<PtxType>
{};

template <typename PtxTypeHead, typename PtxTypeNext, typename... PtxTypeTail>
struct is_valid<type_list<PtxTypeHead, PtxTypeNext, PtxTypeTail...>>
    : std::integral_constant<
        bool,
        (order_valid<PtxTypeHead, PtxTypeNext>::value &&
         is_valid<type_list<PtxTypeNext, PtxTypeTail...>>::value)>
{};

} // namespace check_ptx_type_list_impl

/**
 * Verifies that PtxTypeList is valid; must be sorted from highest-to-lowest
 * `ptx_arch` and contain no two elements with the same `ptx_arch`.
 */
template <typename PtxTypeList>
struct check_ptx_type_list
{
  __host__ __device__ __forceinline__
  void operator()()
  {
    using check_ptx_type_list_impl::is_valid;
    static_assert(is_valid<PtxTypeList>::value,
                  "Invalid PtxTypeList used with cub::detail::ptx_dispatch.\n"
                  "One of following checks failed:\n"
                  "  (a) All elements must define ptx_arch.\n"
                  "  (b) All elements must be sorted from highest-to-lowest "
                  "ptx_arch.\n"
                  "  (c) All elements must have unique ptx_arch values.");
  }
};

/**
 * Host-side implementation of ptx_dispatch. Target ptx_arch is provided at
 * runtime.
 */
template <typename PtxTypeList>
struct ptx_dispatch_host_impl;

template <typename PtxHeadType, typename... PtxTailTypes>
struct ptx_dispatch_host_impl<type_list<PtxHeadType, PtxTailTypes...>>
{
  template <typename Functor, typename... Args>
  __host__  __forceinline__
  static void exec(int target_ptx_arch, Functor &&functor, Args &&...args)
  {
    if (target_ptx_arch < PtxHeadType::ptx_arch)
    {
      using next = ptx_dispatch_host_impl<type_list<PtxTailTypes...>>;
      next::exec(target_ptx_arch,
                 std::forward<Functor>(functor),
                 std::forward<Args>(args)...);
    }
    else
    {
      functor(type_wrapper<PtxHeadType>{}, std::forward<Args>(args)...);
    }
  }
};

template <>
struct ptx_dispatch_host_impl<type_list<>>
{
  template <typename Functor, typename... Args>
  __host__ __forceinline__
  static void exec(int /* target_ptx_arch */, Functor &&, Args &&...)
  {
    // Didn't find a matching arch.
  }
};

/**
 * Executes `Functor<PtxType>{}(args...)`, after finding the best match in
 * `PtxTypes` for `TargetPtxArch`.
 */
template <int TargetPtxArch, typename PtxTypes>
struct ptx_dispatch_device_impl
{
  template <typename Functor, typename... Args>
  __device__ __forceinline__
  static void exec(Functor &&functor, Args &&...args)
  {
    using ptx_type = typename ptx_arch_lookup<TargetPtxArch, PtxTypes>::type;
    functor(type_wrapper<ptx_type>{}, std::forward<Args>(args)...);
  }
};

// Documented in main header.
//
// This can made more user-friendly at some point:
// - Sort `PtxTypes` into descending `ptx_arch` order automatically.
// - Return true/false from `exec` depending on whether or not a suitable
//   PtxType was found.
//
// These changes would remove some of the Caveats listed in the docs.
template <typename PtxTypeList, exec_space ExecSpace>
struct ptx_dispatch
{

#pragma nv_exec_check_disable
  template <typename Functor, typename... Args>
  __host__ __device__ __forceinline__
  static void exec(Functor &&functor, Args &&...args)
  {
    // Validate the PtxTypes:
    check_ptx_type_list<PtxTypeList>{}();

    // Execute the functor:
    exec_dispatch_exec_space(exec_space_tag<ExecSpace>{},
                             std::forward<Functor>(functor),
                             std::forward<Args>(args)...);
  }

private:
  // Subset of PtxTypeList that is needed to satisfy CUB_PTX_TARGETS:
  using active_ptx_types = typename restrict_ptx_types<PtxTypeList>::type;

  template <typename Functor, typename... Args>
  __host__ __forceinline__
  static void exec_dispatch_exec_space(host_tag,
                                       Functor &&functor,
                                       Args &&...args)
  {
    NV_IF_TARGET(NV_IS_HOST,
                 (exec_host(std::forward<Functor>(functor),
                            std::forward<Args>(args)...);));

    // This is necessary to ensure that the current arch's kernel is
    // instantiated in NVCC device passes.
#if defined(__CUDACC__) && defined(__CUDA_ARCH__)
    using PtxType =
      typename ptx_arch_lookup<__CUDA_ARCH__, active_ptx_types>::type;
    instantiate_functor<cub::detail::exec_space::host,
                        Functor,
                        PtxType,
                        Args...>{}();
#endif
  }

  template <typename Functor, typename... Args>
  __device__ __forceinline__
  static void exec_dispatch_exec_space(device_tag,
                                       Functor &&functor,
                                       Args&&... args)
  {
    NV_IF_TARGET(NV_IS_DEVICE,
                 (exec_device(std::forward<Functor>(functor),
                              std::forward<Args>(args)...);));
  }

#pragma nv_exec_check_disable
  template <typename Functor, typename... Args>
  __host__ __device__ __forceinline__
  static void exec_dispatch_exec_space(host_device_tag,
                                       Functor &&functor,
                                       Args&&... args)
  {
    NV_IF_TARGET(NV_IS_HOST,
                 (exec_host(std::forward<Functor>(functor),
                            std::forward<Args>(args)...);),
                 (exec_device(std::forward<Functor>(functor),
                              std::forward<Args>(args)...);));
  }

  /** Use a runtime lookup against the active device's target ptx version. */
  template <typename Functor, typename... Args>
  __host__ __forceinline__
  static void exec_host(Functor &&functor, Args &&...args)
  {
    int ptx_version = 0;
    if (CubDebug(cub::PtxVersion(ptx_version)))
    { // TODO return false
      return;
    }

    using dispatcher_t = ptx_dispatch_host_impl<active_ptx_types>;
    dispatcher_t::exec(ptx_version,
                       std::forward<Functor>(functor),
                       std::forward<Args>(args)...);
  }

  /** Dispatch to static lookup for device. */
  template <typename Functor, typename... Args>
  __device__ __forceinline__
  static void exec_device(Functor &&functor, Args &&...args)
  {
    mark_as_used(args...); // Needed for nvcc+msvc

    NV_DISPATCH_TARGET(NV_PROVIDES_SM_86,
                       (ptx_dispatch_device_impl<860, active_ptx_types>::exec(
                          std::forward<Functor>(functor),
                          std::forward<Args>(args)...);),
                       NV_PROVIDES_SM_80,
                       (ptx_dispatch_device_impl<800, active_ptx_types>::exec(
                          std::forward<Functor>(functor),
                          std::forward<Args>(args)...);),
                       NV_PROVIDES_SM_75,
                       (ptx_dispatch_device_impl<750, active_ptx_types>::exec(
                          std::forward<Functor>(functor),
                          std::forward<Args>(args)...);),
                       NV_PROVIDES_SM_72,
                       (ptx_dispatch_device_impl<720, active_ptx_types>::exec(
                          std::forward<Functor>(functor),
                          std::forward<Args>(args)...);),
                       NV_PROVIDES_SM_70,
                       (ptx_dispatch_device_impl<700, active_ptx_types>::exec(
                          std::forward<Functor>(functor),
                          std::forward<Args>(args)...);),
                       NV_PROVIDES_SM_62,
                       (ptx_dispatch_device_impl<620, active_ptx_types>::exec(
                          std::forward<Functor>(functor),
                          std::forward<Args>(args)...);),
                       NV_PROVIDES_SM_61,
                       (ptx_dispatch_device_impl<610, active_ptx_types>::exec(
                          std::forward<Functor>(functor),
                          std::forward<Args>(args)...);),
                       NV_PROVIDES_SM_60,
                       (ptx_dispatch_device_impl<600, active_ptx_types>::exec(
                          std::forward<Functor>(functor),
                          std::forward<Args>(args)...);),
                       NV_PROVIDES_SM_53,
                       (ptx_dispatch_device_impl<530, active_ptx_types>::exec(
                          std::forward<Functor>(functor),
                          std::forward<Args>(args)...);),
                       NV_PROVIDES_SM_52,
                       (ptx_dispatch_device_impl<520, active_ptx_types>::exec(
                          std::forward<Functor>(functor),
                          std::forward<Args>(args)...);),
                       NV_PROVIDES_SM_50,
                       (ptx_dispatch_device_impl<500, active_ptx_types>::exec(
                          std::forward<Functor>(functor),
                          std::forward<Args>(args)...);),
                       NV_PROVIDES_SM_37,
                       (ptx_dispatch_device_impl<370, active_ptx_types>::exec(
                          std::forward<Functor>(functor),
                          std::forward<Args>(args)...);),
                       NV_PROVIDES_SM_35,
                       (ptx_dispatch_device_impl<350, active_ptx_types>::exec(
                          std::forward<Functor>(functor),
                          std::forward<Args>(args)...);));
  }
};

} // namespace detail
CUB_NAMESPACE_END
