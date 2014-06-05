/******************************************************************************
 * Copyright (c) 2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2014, NVIDIA CORPORATION.  All rights reserved.
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
 * cub::WarpReduceShfl provides SHFL-based variants of parallel reduction of items partitioned across a CUDA thread warp.
 */

#pragma once

#include "../../thread/thread_operators.cuh"
#include "../../util_ptx.cuh"
#include "../../util_type.cuh"
#include "../../util_macro.cuh"
#include "../../util_namespace.cuh"

/// Optional outer namespace(s)
CUB_NS_PREFIX

/// CUB namespace
namespace cub {


/**
 * \brief WarpReduceShfl provides SHFL-based variants of parallel reduction of items partitioned across a CUDA thread warp.
 */
template <
    typename    T,                      ///< Data type being reduced
    int         LOGICAL_WARP_THREADS,   ///< Number of threads per logical warp
    int         PTX_ARCH>               ///< The PTX compute capability for which to to specialize this collective
struct WarpReduceShfl
{
    /******************************************************************************
     * Constants and type definitions
     ******************************************************************************/

    enum
    {
        /// Whether the logical warp size and the PTX warp size coincide
        IS_ARCH_WARP = (LOGICAL_WARP_THREADS == CUB_WARP_THREADS(PTX_ARCH)),

        /// The number of warp reduction steps
        STEPS = Log2<LOGICAL_WARP_THREADS>::VALUE,

        // The 5-bit SHFL mask for logically splitting warps into sub-segments
        SHFL_MASK = (-1 << STEPS) & 31,

        // The 5-bit SFHL clamp
        SHFL_CLAMP = LOGICAL_WARP_THREADS - 1,

        // The packed C argument (mask starts 8 bits up)
        SHFL_C = (SHFL_MASK << 8) | SHFL_CLAMP,

        /// Whether the data type is a primitive integer
        IS_INTEGER = (Traits<T>::CATEGORY == UNSIGNED_INTEGER) || (Traits<T>::CATEGORY == SIGNED_INTEGER),

        ///Whether the data type is a small (32b or less) integer for which we can use a single SFHL instruction per exchange
        IS_SMALL_INTEGER = IS_INTEGER && (sizeof(T) <= sizeof(unsigned int))
    };


    /// Shared memory storage layout type
    typedef NullType TempStorage;


    /******************************************************************************
     * Thread fields
     ******************************************************************************/

    int lane_id;


    /******************************************************************************
     * Construction
     ******************************************************************************/

    /// Constructor
    __device__ __forceinline__ WarpReduceShfl(
        TempStorage &temp_storage)
    :
        lane_id(IS_ARCH_WARP ?
            LaneId() :
            LaneId() % LOGICAL_WARP_THREADS)
    {}


    /******************************************************************************
     * Utility methods
     ******************************************************************************/

    /// Segmented reduction (specialized for summation across primitive integer types 32b or smaller)
    template <typename _T>
    __device__ __forceinline__ void SegmentedReduce(
        _T              input,              ///< [in] Calling thread's input item.
        _T              &output,            ///< [out] Calling thread's output item.  May be aliased with \p input.
        cub::Sum        reduction_op,       ///< [in] Binary reduction operator
        int             last_lane,          ///< [in] Index of last lane in segment
        Int2Type<true>  is_small_integer)   ///< [in] Marker type indicating whether T is a small integer
    {
        unsigned int temp = reinterpret_cast<unsigned int &>(input);

        // Iterate reduction steps
        #pragma unroll
        for (int STEP = 0; STEP < STEPS; STEP++)
        {
            // Use predicate set from SHFL to guard against invalid peers
            asm(
                "{"
                "  .reg .u32 r0;"
                "  .reg .pred p;"
                "  shfl.down.b32 r0|p, %1, %2, %3;"
                "  @p add.u32 r0, r0, %4;"
                "  mov.u32 %0, r0;"
                "}"
                : "=r"(temp) : "r"(temp), "r"(1 << STEP), "r"(last_lane), "r"(temp));
        }

        output = reinterpret_cast<_T&>(temp);
    }


    /// Segmented reduction (specialized for summation across fp32 types)
    __device__ __forceinline__ void SegmentedReduce(
        float           input,              ///< [in] Calling thread's input item.
        float           &output,            ///< [out] Calling thread's output item.  May be aliased with \p input.
        cub::Sum        reduction_op,       ///< [in] Binary reduction operator
        int             last_lane,          ///< [in] Index of last lane in segment
        Int2Type<false> is_small_integer)   ///< [in] Marker type indicating whether T is a small integer
    {
        output = input;

        // Iterate reduction steps
        #pragma unroll
        for (int STEP = 0; STEP < STEPS; STEP++)
        {
            // Use predicate set from SHFL to guard against invalid peers
            asm(
                "{"
                "  .reg .f32 r0;"
                "  .reg .pred p;"
                "  shfl.down.b32 r0|p, %1, %2, %3;"
                "  @p add.f32 r0, r0, %4;"
                "  mov.f32 %0, r0;"
                "}"
                : "=f"(output) : "f"(output), "r"(1 << STEP), "r"(last_lane), "f"(output));
        }
    }


    /// Segmented reduction (specialized for summation across unsigned long long types)
    __device__ __forceinline__ void SegmentedReduce(
        unsigned long long  input,              ///< [in] Calling thread's input item.
        unsigned long long  &output,            ///< [out] Calling thread's output item.  May be aliased with \p input.
        cub::Sum            reduction_op,       ///< [in] Binary reduction operator
        int                 last_lane,          ///< [in] Index of last lane in segment
        Int2Type<false>     is_small_integer)   ///< [in] Marker type indicating whether T is a small integer
    {
        output = input;

        // Iterate reduction steps
        #pragma unroll
        for (int STEP = 0; STEP < STEPS; STEP++)
        {
            // Use predicate set from SHFL to guard against invalid peers
            asm(
                "{"
                "  .reg .u32 lo;"
                "  .reg .u32 hi;"
                "  .reg .pred p;"
                "  mov.b64 {lo, hi}, %1;"
                "  shfl.down.b32 lo|p, lo, %2, %3;"
                "  shfl.down.b32 hi|p, hi, %2, %3;"
                "  mov.b64 %0, {lo, hi};"
                "  @p add.u64 %0, %0, %1;"
                "}"
                : "=l"(output) : "l"(output), "r"(1 << STEP), "r"(last_lane));
        }
    }


    /// Segmented reduction (specialized for summation across long long types)
    __device__ __forceinline__ void SegmentedReduce(
        long long           input,              ///< [in] Calling thread's input item.
        long long           &output,            ///< [out] Calling thread's output item.  May be aliased with \p input.
        cub::Sum            reduction_op,       ///< [in] Binary reduction operator
        int                 last_lane,          ///< [in] Index of last lane in segment
        Int2Type<false>     is_small_integer)   ///< [in] Marker type indicating whether T is a small integer
    {
        output = input;

        // Iterate reduction steps
        #pragma unroll
        for (int STEP = 0; STEP < STEPS; STEP++)
        {
            // Use predicate set from SHFL to guard against invalid peers
            asm(
                "{"
                "  .reg .u32 lo;"
                "  .reg .u32 hi;"
                "  .reg .pred p;"
                "  mov.b64 {lo, hi}, %1;"
                "  shfl.down.b32 lo|p, lo, %2, %3;"
                "  shfl.down.b32 hi|p, hi, %2, %3;"
                "  mov.b64 %0, {lo, hi};"
                "  @p add.s64 %0, %0, %1;"
                "}"
                : "=l"(output) : "l"(output), "r"(1 << STEP), "r"(last_lane));
        }
    }


    /// Segmented reduction (specialized for summation across double types)
    __device__ __forceinline__ void SegmentedReduce(
        double              input,              ///< [in] Calling thread's input item.
        double              &output,            ///< [out] Calling thread's output item.  May be aliased with \p input.
        cub::Sum            reduction_op,       ///< [in] Binary reduction operator
        int                 last_lane,          ///< [in] Index of last lane in segment
        Int2Type<false>     is_small_integer)   ///< [in] Marker type indicating whether T is a small integer
    {
        output = input;

        // Iterate reduction steps
        #pragma unroll
        for (int STEP = 0; STEP < STEPS; STEP++)
        {
            // Use predicate set from SHFL to guard against invalid peers
            asm(
                "{"
                "  .reg .u32 lo;"
                "  .reg .u32 hi;"
                "  .reg .pred p;"
                "  mov.b64 {lo, hi}, %1;"
                "  shfl.down.b32 lo|p, lo, %2, %3;"
                "  shfl.down.b32 hi|p, hi, %2, %3;"
                "  mov.b64 %0, {lo, hi};"
                "  @p add.f64 %0, %0, %1;"
                "}"
                : "=d"(output) : "d"(output), "r"(1 << STEP), "r"(last_lane));
        }
    }

    /// Segmented reduction
    template <typename _T, typename ReductionOp, int _IS_SMALL_INTEGER>
    __device__ __forceinline__ void SegmentedReduce(
        _T                              input,              ///< [in] Calling thread's input item.
        _T                              &output,            ///< [out] Calling thread's output item.  May be aliased with \p input.
        ReductionOp                     reduction_op,       ///< [in] Binary reduction operator
        int                             last_lane,          ///< [in] Index of last lane in segment
        Int2Type<_IS_SMALL_INTEGER>     is_small_integer)   ///< [in] Marker type indicating whether T is a small integer
    {
        output = input;
        int lane_id = LaneId();

        // Iterate reduction steps
        #pragma unroll
        for (int STEP = 0; STEP < STEPS; STEP++)
        {
            const int OFFSET = 1 << STEP;

            T temp = ShuffleDown(output, OFFSET);

            // Perform reduction op if valid
            if (OFFSET <= last_lane - lane_id)
            {
                output = reduction_op(output, temp);
            }
        }
    }


    /******************************************************************************
     * Interface
     ******************************************************************************/

    /// Summation (single-SHFL)
    template <
        bool                ALL_LANES_VALID,        ///< Whether all lanes in each warp are contributing a valid fold of items
        int                 FOLDED_ITEMS_PER_LANE>  ///< Number of items folded into each lane
    __device__ __forceinline__ T Sum(
        T                   input,                  ///< [in] Calling thread's input
        int                 folded_items_per_warp,  ///< [in] Total number of valid items folded into each logical warp
        Int2Type<true>      single_shfl)            ///< [in] Marker type indicating whether only one SHFL instruction is required
    {
        unsigned int output = reinterpret_cast<unsigned int &>(input);

        // Iterate reduction steps
        #pragma unroll
        for (int STEP = 0; STEP < STEPS; STEP++)
        {
            const int OFFSET = 1 << STEP;

            if (ALL_LANES_VALID)
            {
                // Use predicate set from SHFL to guard against invalid peers
                asm(
                    "{"
                    "  .reg .u32 r0;"
                    "  .reg .pred p;"
                    "  shfl.down.b32 r0|p, %1, %2, %3;"
                    "  @p add.u32 r0, r0, %4;"
                    "  mov.u32 %0, r0;"
                    "}"
                    : "=r"(output) : "r"(output), "r"(OFFSET), "r"(SHFL_C), "r"(output));
            }
            else
            {
                // Set range predicate to guard against invalid peers
                asm(
                    "{"
                    "  .reg .u32 r0;"
                    "  .reg .pred p;"
                    "  shfl.down.b32 r0, %1, %2, %3;"
                    "  setp.lt.u32 p, %5, %6;"
                    "  mov.u32 %0, %1;"
                    "  @p add.u32 %0, %1, r0;"
                    "}"
                    : "=r"(output) : "r"(output), "r"(OFFSET), "r"(SHFL_C), "r"(output), "r"((lane_id + OFFSET) * FOLDED_ITEMS_PER_LANE), "r"(folded_items_per_warp));
            }
        }

        return output;
    }


    /// Summation (multi-SHFL)
    template <
        bool                ALL_LANES_VALID,        ///< Whether all lanes in each warp are contributing a valid fold of items
        int                 FOLDED_ITEMS_PER_LANE>  ///< Number of items folded into each lane
    __device__ __forceinline__ T Sum(
        T                   input,                  ///< [in] Calling thread's input
        int                 folded_items_per_warp,  ///< [in] Total number of valid items folded into each logical warp
        Int2Type<false>     single_shfl)            ///< [in] Marker type indicating whether only one SHFL instruction is required
    {
        // Delegate to generic reduce
        return Reduce<ALL_LANES_VALID, FOLDED_ITEMS_PER_LANE>(input, folded_items_per_warp, cub::Sum());
    }


    /// Summation (float)
    template <
        bool                ALL_LANES_VALID,        ///< Whether all lanes in each warp are contributing a valid fold of items
        int                 FOLDED_ITEMS_PER_LANE>  ///< Number of items folded into each lane
    __device__ __forceinline__ float Sum(
        float               input,                  ///< [in] Calling thread's input
        int                 folded_items_per_warp)  ///< [in] Total number of valid items folded into each logical warp
    {
        T output = input;

        // Iterate reduction steps
        #pragma unroll
        for (int STEP = 0; STEP < STEPS; STEP++)
        {
            const int OFFSET = 1 << STEP;

            if (ALL_LANES_VALID)
            {
                // Use predicate set from SHFL to guard against invalid peers
                asm(
                    "{"
                    "  .reg .f32 r0;"
                    "  .reg .pred p;"
                    "  shfl.down.b32 r0|p, %1, %2, %3;"
                    "  @p add.f32 r0, r0, %4;"
                    "  mov.f32 %0, r0;"
                    "}"
                    : "=f"(output) : "f"(output), "r"(OFFSET), "r"(SHFL_C), "f"(output));
            }
            else
            {
                // Set range predicate to guard against invalid peers
                asm(
                    "{"
                    "  .reg .f32 r0;"
                    "  .reg .pred p;"
                    "  shfl.down.b32 r0, %1, %2, %3;"
                    "  setp.lt.u32 p, %5, %6;"
                    "  mov.f32 %0, %1;"
                    "  @p add.f32 %0, %0, r0;"
                    "}"
                    : "=f"(output) : "f"(output), "r"(OFFSET), "r"(SHFL_C), "f"(output), "r"((lane_id + OFFSET) * FOLDED_ITEMS_PER_LANE), "r"(folded_items_per_warp));
            }
        }

        return output;
    }

    /// Summation (generic)
    template <
        bool                ALL_LANES_VALID,        ///< Whether all lanes in each warp are contributing a valid fold of items
        int                 FOLDED_ITEMS_PER_LANE,  ///< Number of items folded into each lane
        typename            _T>
    __device__ __forceinline__ _T Sum(
        _T                  input,                  ///< [in] Calling thread's input
        int                 folded_items_per_warp)  ///< [in] Total number of valid items folded into each logical warp
    {
        // Whether sharing can be done with a single SHFL instruction (vs multiple SFHL instructions)
        Int2Type<(Traits<_T>::PRIMITIVE) && (sizeof(_T) <= sizeof(unsigned int))> single_shfl;

        return Sum<ALL_LANES_VALID, FOLDED_ITEMS_PER_LANE>(input, folded_items_per_warp, single_shfl);
    }


    /// Reduction
    template <
        bool            ALL_LANES_VALID,        ///< Whether all lanes in each warp are contributing a valid fold of items
        int             FOLDED_ITEMS_PER_LANE,  ///< Number of items folded into each lane
        typename        ReductionOp>
    __device__ __forceinline__ T Reduce(
        T               input,                  ///< [in] Calling thread's input
        int             folded_items_per_warp,  ///< [in] Total number of valid items folded into each logical warp
        ReductionOp     reduction_op)           ///< [in] Binary reduction operator
    {
        T output = input;

        // Iterate scan steps
        #pragma unroll
        for (int STEP = 0; STEP < STEPS; STEP++)
        {
            // Grab addend from peer
            const int OFFSET = 1 << STEP;

            T temp = ShuffleDown(output, OFFSET);

            // Perform reduction op if from a valid peer
            if (ALL_LANES_VALID)
            {
                if (lane_id < LOGICAL_WARP_THREADS - OFFSET)
                    output = reduction_op(output, temp);
            }
            else
            {
                if (((lane_id + OFFSET) * FOLDED_ITEMS_PER_LANE) < folded_items_per_warp)
                    output = reduction_op(output, temp);
            }
        }

        return output;
    }


    /// Segmented reduction
    template <
        bool            HEAD_SEGMENTED,     ///< Whether flags indicate a segment-head or a segment-tail
        typename        Flag,
        typename        ReductionOp>
    __device__ __forceinline__ T SegmentedReduce(
        T               input,              ///< [in] Calling thread's input
        Flag            flag,               ///< [in] Whether or not the current lane is a segment head/tail
        ReductionOp     reduction_op)       ///< [in] Binary reduction operator
    {
        T output = input;

        // Get the start flags for each thread in the warp.
        int warp_flags = __ballot(flag);

        if (HEAD_SEGMENTED)
            warp_flags >>= 1;

        // Keep bits above the current thread.
        warp_flags &= LaneMaskGe();

        // Find next flag
        int last_lane = __clz(__brev(warp_flags));

        // Get the last lane in the logical warp
        int last_warp_thread = LOGICAL_WARP_THREADS - 1;
        if (!IS_ARCH_WARP)
            last_warp_thread |= LaneId();

        // Set the last lane in the warp
        last_lane = CUB_MIN(last_lane, last_warp_thread);

        SegmentedReduce(input, output, reduction_op, last_lane, Int2Type<IS_SMALL_INTEGER>());

        return output;
    }
};


}               // CUB namespace
CUB_NS_POSTFIX  // Optional outer namespace(s)
