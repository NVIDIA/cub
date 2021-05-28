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

#include <cub/detail/preprocessor.cuh>
#include <cub/util_namespace.cuh>

#include <cuda_runtime_api.h>


#ifdef DOXYGEN_SHOULD_SKIP_THIS // Only parse this during doxygen passes:

/**
 * A comma separated list of all virtual PTX architectures targeted by the
 * current compilation. This is used to restrict the number of kernel tunings
 * considered by `cub::detail::ptx_dispatch`. PTX architectures are specified
 * using the 3-digit form.
 *
 * This is effectively all of the possible values of `__CUDA_ARCH__` that
 * would be defined by nvcc during device passes, with no duplicates and
 * in ascending order. This is defined during all compilation passes.
 *
 * Users may define this directly to override any detected/default settings.
 *
 * By default, `__CUDA_ARCH_LIST__` will be used if it is defined. Otherwise,
 * `CUB_PTX_TARGETS` will contain all architectures used by `ptx_dispatch`.
 */
#define CUB_PTX_TARGETS

#else // Non-doxygen pass:

#ifndef CUB_PTX_TARGETS

#ifdef __CUDA_ARCH_LIST__
#define CUB_PTX_TARGETS __CUDA_ARCH_LIST__
#elif defined(NV_TARGET_SM_INTEGER_LIST)

// clang-format off
#define CUB_DETAIL_PTX_TARGETS_NAME1(P1) \
    P1##0
#define CUB_DETAIL_PTX_TARGETS_NAME2(P1, P2) \
    P1##0,P2##0
#define CUB_DETAIL_PTX_TARGETS_NAME3(P1, P2, P3) \
    P1##0,P2##0,P3##0
#define CUB_DETAIL_PTX_TARGETS_NAME4(P1, P2, P3, P4) \
    P1##0,P2##0,P3##0,P4##0
#define CUB_DETAIL_PTX_TARGETS_NAME5(P1, P2, P3, P4, P5) \
    P1##0,P2##0,P3##0,P4##0,P5##0
#define CUB_DETAIL_PTX_TARGETS_NAME6(P1, P2, P3, P4, P5, P6) \
    P1##0,P2##0,P3##0,P4##0,P5##0,P6##0
#define CUB_DETAIL_PTX_TARGETS_NAME7(P1, P2, P3, P4, P5, P6, P7) \
    P1##0,P2##0,P3##0,P4##0,P5##0,P6##0,P7##0
#define CUB_DETAIL_PTX_TARGETS_NAME8(P1, P2, P3, P4, P5, P6, P7, P8) \
    P1##0,P2##0,P3##0,P4##0,P5##0,P6##0,P7##0,P8##0
#define CUB_DETAIL_PTX_TARGETS_NAME9(P1, P2, P3, P4, P5, P6, P7, P8, P9) \
    P1##0,P2##0,P3##0,P4##0,P5##0,P6##0,P7##0,P8##0,P9##0
#define CUB_DETAIL_PTX_TARGETS_NAME10(P1, P2, P3, P4, P5, P6, P7, P8, P9, P10) \
    P1##0,P2##0,P3##0,P4##0,P5##0,P6##0,P7##0,P8##0,P9##0,P10##0
#define CUB_DETAIL_PTX_TARGETS_NAME11(P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11) \
    P1##0,P2##0,P3##0,P4##0,P5##0,P6##0,P7##0,P8##0,P9##0,P10##0,P11##0
#define CUB_DETAIL_PTX_TARGETS_NAME12(P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11, P12) \
    P1##0,P2##0,P3##0,P4##0,P5##0,P6##0,P7##0,P8##0,P9##0,P10##0,P11##0,P12##0
#define CUB_DETAIL_PTX_TARGETS_NAME13(P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11, P12, P13) \
    P1##0,P2##0,P3##0,P4##0,P5##0,P6##0,P7##0,P8##0,P9##0,P10##0,P11##0,P12##0,P13##0
#define CUB_DETAIL_PTX_TARGETS_NAME14(P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11, P12, P13, P14) \
    P1##0,P2##0,P3##0,P4##0,P5##0,P6##0,P7##0,P8##0,P9##0,P10##0,P11##0,P12##0,P13##0,P14##0
#define CUB_DETAIL_PTX_TARGETS_NAME15(P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11, P12, P13, P14, P15) \
    P1##0,P2##0,P3##0,P4##0,P5##0,P6##0,P7##0,P8##0,P9##0,P10##0,P11##0,P12##0,P13##0,P14##0,P15##0
#define CUB_DETAIL_PTX_TARGETS_NAME16(P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11, P12, P13, P14, P15, P16) \
    P1##0,P2##0,P3##0,P4##0,P5##0,P6##0,P7##0,P8##0,P9##0,P10##0,P11##0,P12##0,P13##0,P14##0,P15##0,P16##0
#define CUB_DETAIL_PTX_TARGETS_NAME17(P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11, P12, P13, P14, P15, P16, P17) \
    P1##0,P2##0,P3##0,P4##0,P5##0,P6##0,P7##0,P8##0,P9##0,P10##0,P11##0,P12##0,P13##0,P14##0,P15##0,P16##0,P17##0
#define CUB_DETAIL_PTX_TARGETS_NAME18(P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11, P12, P13, P14, P15, P16, P17, P18) \
    P1##0,P2##0,P3##0,P4##0,P5##0,P6##0,P7##0,P8##0,P9##0,P10##0,P11##0,P12##0,P13##0,P14##0,P15##0,P16##0,P17##0,P18##0
#define CUB_DETAIL_PTX_TARGETS_NAME19(P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11, P12, P13, P14, P15, P16, P17, P18, P19) \
    P1##0,P2##0,P3##0,P4##0,P5##0,P6##0,P7##0,P8##0,P9##0,P10##0,P11##0,P12##0,P13##0,P14##0,P15##0,P16##0,P17##0,P18##0,P19##0
#define CUB_DETAIL_PTX_TARGETS_NAME20(P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11, P12, P13, P14, P15, P16, P17, P18, P19, P20) \
    P1##0,P2##0,P3##0,P4##0,P5##0,P6##0,P7##0,P8##0,P9##0,P10##0,P11##0,P12##0,P13##0,P14##0,P15##0,P16##0,P17##0,P18##0,P19##0,P20##0
#define CUB_DETAIL_PTX_TARGETS_DISPATCH(N) CUB_DETAIL_PTX_TARGETS_NAME ## N
#define CUB_DETAIL_PTX_TARGETS_EXPAND(...) CUB_DETAIL_IDENTITY(CUB_DETAIL_APPLY(CUB_DETAIL_PTX_TARGETS_DISPATCH, CUB_DETAIL_COUNT(__VA_ARGS__))(__VA_ARGS__))
// clang-format on

// NV_TARGET_SM_INTEGER_LIST is the same as __CUDA_ARCH_LIST__, but with two
// digit ids instead of three.
#define CUB_PTX_TARGETS CUB_DETAIL_PTX_TARGETS_EXPAND(NV_TARGET_SM_INTEGER_LIST)

#else
#define CUB_PTX_TARGETS 350,370,500,520,530,600,610,620,700,720,750,800,860
#endif

#endif // CUB_PTX_TARGETS predefined

#endif  // Do not document
