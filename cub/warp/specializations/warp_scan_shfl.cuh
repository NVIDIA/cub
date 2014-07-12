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
 * cub::WarpScanShfl provides SHFL-based variants of parallel prefix scan of items partitioned across a CUDA thread warp.
 */

#pragma once

#include "../../thread/thread_operators.cuh"
#include "../../util_type.cuh"
#include "../../util_ptx.cuh"
#include "../../util_namespace.cuh"

/// Optional outer namespace(s)
CUB_NS_PREFIX

/// CUB namespace
namespace cub {

/**
 * \brief WarpScanShfl provides SHFL-based variants of parallel prefix scan of items partitioned across a CUDA thread warp.
 */
template <
    typename    T,                      ///< Data type being scanned
    int         LOGICAL_WARP_THREADS,   ///< Number of threads per logical warp
    int         PTX_ARCH>               ///< The PTX compute capability for which to to specialize this collective
struct WarpScanShfl
{

    /******************************************************************************
     * Constants and type definitions
     ******************************************************************************/

    enum
    {
        /// Whether the logical warp size and the PTX warp size coincide
        IS_ARCH_WARP = (LOGICAL_WARP_THREADS == CUB_WARP_THREADS(PTX_ARCH)),

        /// The number of warp scan steps
        STEPS = Log2<LOGICAL_WARP_THREADS>::VALUE,

        /// The 5-bit SHFL mask for logically splitting warps into sub-segments starts 8-bits up
        SHFL_C = ((-1 << STEPS) & 31) << 8,
    };

    template <typename S>
    struct IsInteger
    {
        enum {
            /// Whether the data type is a primitive integer
            IS_INTEGER = (Traits<S>::CATEGORY == UNSIGNED_INTEGER) || (Traits<S>::CATEGORY == SIGNED_INTEGER),

            ///Whether the data type is a small (32b or less) integer for which we can use a single SFHL instruction per exchange
            IS_SMALL_INTEGER = IS_INTEGER && (sizeof(S) <= sizeof(unsigned int))
        };
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
    __device__ __forceinline__ WarpScanShfl(
        TempStorage &temp_storage)
    :
        lane_id(IS_ARCH_WARP ?
            LaneId() :
            LaneId() % LOGICAL_WARP_THREADS)
    {}


    /******************************************************************************
     * Utility methods
     ******************************************************************************/

    /// Inclusive prefix scan step (specialized for summation across uint32 types)
    __device__ __forceinline__ unsigned int InclusiveScanStep(
        unsigned int    input,              ///< [in] Calling thread's input item.
        cub::Sum        scan_op,            ///< [in] Binary scan operator
        int             first_lane,         ///< [in] Index of first lane in segment
        int             offset)             ///< [in] Up-offset to pull from
    {
        unsigned int output;

        // Use predicate set from SHFL to guard against invalid peers
        asm(
            "{"
            "  .reg .u32 r0;"
            "  .reg .pred p;"
            "  shfl.up.b32 r0|p, %1, %2, %3;"
            "  @p add.u32 r0, r0, %4;"
            "  mov.u32 %0, r0;"
            "}"
            : "=r"(output) : "r"(input), "r"(offset), "r"(first_lane), "r"(input));

        return output;
    }


    /// Inclusive prefix scan step (specialized for summation across fp32 types)
    __device__ __forceinline__ float InclusiveScanStep(
        float           input,              ///< [in] Calling thread's input item.
        cub::Sum        scan_op,            ///< [in] Binary scan operator
        int             first_lane,         ///< [in] Index of first lane in segment
        int             offset)             ///< [in] Up-offset to pull from
    {
        float output;

        // Use predicate set from SHFL to guard against invalid peers
        asm(
            "{"
            "  .reg .f32 r0;"
            "  .reg .pred p;"
            "  shfl.up.b32 r0|p, %1, %2, %3;"
            "  @p add.f32 r0, r0, %4;"
            "  mov.f32 %0, r0;"
            "}"
            : "=f"(output) : "f"(input), "r"(offset), "r"(first_lane), "f"(input));

        return output;
    }


    /// Inclusive prefix scan step (specialized for summation across unsigned long long types)
    __device__ __forceinline__ unsigned long long InclusiveScanStep(
        unsigned long long  input,              ///< [in] Calling thread's input item.
        cub::Sum            scan_op,            ///< [in] Binary scan operator
        int             first_lane,         ///< [in] Index of first lane in segment
        int             offset)             ///< [in] Up-offset to pull from
    {
        unsigned long long output;

        // Use predicate set from SHFL to guard against invalid peers
        asm(
            "{"
            "  .reg .u32 lo;"
            "  .reg .u32 hi;"
            "  .reg .pred p;"
            "  mov.b64 {lo, hi}, %1;"
            "  shfl.up.b32 lo|p, lo, %2, %3;"
            "  shfl.up.b32 hi|p, hi, %2, %3;"
            "  mov.b64 %0, {lo, hi};"
            "  @p add.u64 %0, %0, %1;"
            "}"
            : "=l"(output) : "l"(input), "r"(offset), "r"(first_lane));

        return output;
    }


    /// Inclusive prefix scan step (specialized for summation across long long types)
    __device__ __forceinline__ long long InclusiveScanStep(
        long long       input,              ///< [in] Calling thread's input item.
        cub::Sum        scan_op,            ///< [in] Binary scan operator
        int             first_lane,         ///< [in] Index of first lane in segment
        int             offset)             ///< [in] Up-offset to pull from
    {
        long long output;

        // Use predicate set from SHFL to guard against invalid peers
        asm(
            "{"
            "  .reg .u32 lo;"
            "  .reg .u32 hi;"
            "  .reg .pred p;"
            "  mov.b64 {lo, hi}, %1;"
            "  shfl.up.b32 lo|p, lo, %2, %3;"
            "  shfl.up.b32 hi|p, hi, %2, %3;"
            "  mov.b64 %0, {lo, hi};"
            "  @p add.s64 %0, %0, %1;"
            "}"
            : "=l"(output) : "l"(input), "r"(offset), "r"(first_lane));

        return output;
    }


    /// Inclusive prefix scan step (specialized for summation across fp64 types)
    __device__ __forceinline__ double InclusiveScanStep(
        double          input,              ///< [in] Calling thread's input item.
        cub::Sum        scan_op,            ///< [in] Binary scan operator
        int             first_lane,         ///< [in] Index of first lane in segment
        int             offset)             ///< [in] Up-offset to pull from
    {
        double output;

        // Use predicate set from SHFL to guard against invalid peers
        asm(
            "{"
            "  .reg .u32 lo;"
            "  .reg .u32 hi;"
            "  .reg .pred p;"
            "  mov.b64 {lo, hi}, %1;"
            "  shfl.up.b32 lo|p, lo, %2, %3;"
            "  shfl.up.b32 hi|p, hi, %2, %3;"
            "  mov.b64 %0, {lo, hi};"
            "  @p add.f64 %0, %0, %1;"
            "}"
            : "=d"(output) : "d"(input), "r"(offset), "r"(first_lane));

        return output;
    }


    /// Inclusive prefix scan (specialized for ReduceBySegmentOp<cub::Sum> across ItemOffsetPair<Value, Offset> types)
    template <typename Value, typename Offset>
    __device__ __forceinline__ ItemOffsetPair<Value, Offset> InclusiveScanStep(
        ItemOffsetPair<Value, Offset>                               input,              ///< [in] Calling thread's input item.
        ReduceBySegmentOp<cub::Sum, ItemOffsetPair<Value, Offset> > scan_op,            ///< [in] Binary scan operator
        int                                                         first_lane,         ///< [in] Index of first lane in segment
        int                                                         offset)             ///< [in] Up-offset to pull from
    {
        ItemOffsetPair<Value, Offset> output;

        output.value = InclusiveScanStep(input.value, cub::Sum(), first_lane, offset, Int2Type<IsInteger<Value>::IS_SMALL_INTEGER>());
        output.offset = InclusiveScanStep(input.offset, cub::Sum(), first_lane, offset, Int2Type<IsInteger<Offset>::IS_SMALL_INTEGER>());

        if (input.offset > 0)
            output.value = input.value;

/*
        int first_value_lane = (input.offset > 0) ? LOGICAL_WARP_THREADS - 1 : first_lane;
        output.value = InclusiveScanStep(input.value, cub::Sum(), first_value_lane, offset, Int2Type<IsInteger<Value>::IS_SMALL_INTEGER>());
*/
        return output;
    }


    /// Inclusive prefix scan step (generic)
    template <typename _T, typename ScanOp>
    __device__ __forceinline__ _T InclusiveScanStep(
        _T              input,              ///< [in] Calling thread's input item.
        ScanOp          scan_op,            ///< [in] Binary scan operator
        int             first_lane,         ///< [in] Index of first lane in segment
        int             offset)             ///< [in] Up-offset to pull from
    {
        T output = input;

        T temp = ShuffleUp(output, offset);

        // Perform scan op if from a valid peer
        if (lane_id >= offset)
            output = scan_op(temp, output);

        return output;
    }


    /// Inclusive prefix scan step (specialized for small integers size 32b or less)
    template <typename _T, typename ScanOp>
    __device__ __forceinline__ _T InclusiveScanStep(
        _T              input,              ///< [in] Calling thread's input item.
        ScanOp          scan_op,            ///< [in] Binary scan operator
        int             first_lane,         ///< [in] Index of first lane in segment
        int             offset,             ///< [in] Up-offset to pull from
        Int2Type<true>  is_small_integer)   ///< [in] Marker type indicating whether T is a small integer
    {
        unsigned int temp = reinterpret_cast<unsigned int &>(input);

        temp = InclusiveScanStep(temp, scan_op, first_lane, offset);

        return reinterpret_cast<_T&>(temp);
    }


    /// Inclusive prefix scan step (specialized for types other than small integers size 32b or less)
    template <typename _T, typename ScanOp>
    __device__ __forceinline__ _T InclusiveScanStep(
        _T              input,              ///< [in] Calling thread's input item.
        ScanOp          scan_op,            ///< [in] Binary scan operator
        int             first_lane,         ///< [in] Index of first lane in segment
        int             offset,             ///< [in] Up-offset to pull from
        Int2Type<false> is_small_integer)   ///< [in] Marker type indicating whether T is a small integer
    {
        return InclusiveScanStep(input, scan_op, first_lane, offset);
    }


    /// Get exclusive from inclusive (specialized for summation of integer types)
    __device__ __forceinline__ T GetExclusive(
        T               input,
        T               inclusive,
        cub::Sum        scan_op,
        Int2Type<true>  is_integer)
    {
        return inclusive - input;
    }


    /// Get exclusive from inclusive (specialized for scans other than summation of integer types)
    template <typename ScanOp, int _IS_INTEGER>
    __device__ __forceinline__ T GetExclusive(
        T                       input,
        T                       inclusive,
        ScanOp                  scan_op,
        Int2Type<_IS_INTEGER>   is_integer)
    {
        return ShuffleUp(inclusive, 1);
    }

    /// Get exclusive from inclusive (specialized for summation of integer types)
    __device__ __forceinline__ T GetExclusive(
        T               input,
        T               inclusive,
        T               identity,
        cub::Sum        scan_op,
        Int2Type<true>  is_integer)
    {
        return inclusive - input;
    }


    /// Get exclusive from inclusive (specialized for scans other than summation of integer types)
    template <typename ScanOp, int _IS_INTEGER>
    __device__ __forceinline__ T GetExclusive(
        T                       input,
        T                       inclusive,
        T                       identity,
        ScanOp                  scan_op,
        Int2Type<_IS_INTEGER>   is_integer)
    {
        T exclusive = ShuffleUp(inclusive, 1);
        return (lane_id == 0) ? identity : exclusive;
    }


    /******************************************************************************
     * Interface
     ******************************************************************************/


    /// Broadcast
    __device__ __forceinline__ T Broadcast(
        T               input,              ///< [in] The value to broadcast
        int             src_lane)           ///< [in] Which warp lane is to do the broadcasting
    {
        return ShuffleBroadcast(input, src_lane, LOGICAL_WARP_THREADS);
    }


    //---------------------------------------------------------------------
    // Inclusive operations
    //---------------------------------------------------------------------

    /// Inclusive scan
    template <typename ScanOp>
    __device__ __forceinline__ void InclusiveScan(
        T               input,              ///< [in] Calling thread's input item.
        T               &output,            ///< [out] Calling thread's output item.  May be aliased with \p input.
        ScanOp          scan_op)            ///< [in] Binary scan operator
    {
        output = input;

        // Iterate scan steps
        #pragma unroll
        for (int STEP = 0; STEP < STEPS; STEP++)
        {
            output = InclusiveScanStep(output, scan_op, SHFL_C, 1 << STEP, Int2Type<IsInteger<T>::IS_SMALL_INTEGER>());
        }
    }


    /// Inclusive scan with aggregate
    template <typename ScanOp>
    __device__ __forceinline__ void InclusiveScan(
        T               input,              ///< [in] Calling thread's input item.
        T               &output,            ///< [out] Calling thread's output item.  May be aliased with \p input.
        ScanOp          scan_op,            ///< [in] Binary scan operator
        T               &warp_aggregate)    ///< [out] Warp-wide aggregate reduction of input items.
    {
        InclusiveScan(input, output, scan_op);

        // Grab aggregate from last warp lane
        warp_aggregate = Broadcast(output, LOGICAL_WARP_THREADS - 1);
    }


    //---------------------------------------------------------------------
    // Combo (inclusive & exclusive) operations
    //---------------------------------------------------------------------

    /// Combination scan without identity
    template <typename ScanOp>
    __device__ __forceinline__ void Scan(
        T               input,              ///< [in] Calling thread's input item.
        T               &inclusive_output,  ///< [out] Calling thread's inclusive-scan output item.
        T               &exclusive_output,  ///< [out] Calling thread's exclusive-scan output item.
        ScanOp          scan_op)            ///< [in] Binary scan operator
    {
        // Compute inclusive scan
        InclusiveScan(input, inclusive_output, scan_op);

        // Grab result from predecessor
        exclusive_output = GetExclusive(input, inclusive_output, scan_op, Int2Type<IsInteger<T>::IS_INTEGER>());
    }

    /// Combination scan with identity
    template <typename ScanOp>
    __device__ __forceinline__ void Scan(
        T               input,              ///< [in] Calling thread's input item.
        T               &inclusive_output,  ///< [out] Calling thread's inclusive-scan output item.
        T               &exclusive_output,  ///< [out] Calling thread's exclusive-scan output item.
        T               identity,           ///< [in] Identity value
        ScanOp          scan_op)            ///< [in] Binary scan operator
    {
        // Compute inclusive scan
        InclusiveScan(input, inclusive_output, scan_op);

        // Grab result from predecessor
        exclusive_output = GetExclusive(input, inclusive_output, identity, scan_op, Int2Type<IsInteger<T>::IS_INTEGER>());
    }


    //---------------------------------------------------------------------
    // Exclusive operations
    //---------------------------------------------------------------------

    /// Exclusive scan with aggregate
    template <typename ScanOp>
    __device__ __forceinline__ void ExclusiveScan(
        T               input,              ///< [in] Calling thread's input item.
        T               &output,            ///< [out] Calling thread's output item.  May be aliased with \p input.
        T               identity,           ///< [in] Identity value
        ScanOp          scan_op)            ///< [in] Binary scan operator
    {
        T inclusive_output;
        Scan(input, inclusive_output, output, identity, scan_op);
    }


    /// Exclusive scan with aggregate, without identity
    template <typename ScanOp>
    __device__ __forceinline__ void ExclusiveScan(
        T               input,              ///< [in] Calling thread's input item.
        T               &output,            ///< [out] Calling thread's output item.  May be aliased with \p input.
        ScanOp          scan_op)            ///< [in] Binary scan operator
    {
        T inclusive_output;
        Scan(input, inclusive_output, output, scan_op);
    }


    /// Exclusive scan with aggregate
    template <typename ScanOp>
    __device__ __forceinline__ void ExclusiveScan(
        T               input,              ///< [in] Calling thread's input item.
        T               &output,            ///< [out] Calling thread's output item.  May be aliased with \p input.
        T               identity,           ///< [in] Identity value
        ScanOp          scan_op,            ///< [in] Binary scan operator
        T               &warp_aggregate)    ///< [out] Warp-wide aggregate reduction of input items.
    {
        T inclusive_output;
        Scan(input, inclusive_output, output, identity, scan_op);

        // Grab aggregate from last warp lane
        warp_aggregate = Broadcast(inclusive_output, LOGICAL_WARP_THREADS - 1);
    }


    /// Exclusive scan with aggregate, without identity
    template <typename ScanOp>
    __device__ __forceinline__ void ExclusiveScan(
        T               input,              ///< [in] Calling thread's input item.
        T               &output,            ///< [out] Calling thread's output item.  May be aliased with \p input.
        ScanOp          scan_op,            ///< [in] Binary scan operator
        T               &warp_aggregate)    ///< [out] Warp-wide aggregate reduction of input items.
    {
        T inclusive_output;
        Scan(input, inclusive_output, output, scan_op);

        // Grab aggregate from last warp lane
        warp_aggregate = Broadcast(inclusive_output, LOGICAL_WARP_THREADS - 1);
    }

};


}               // CUB namespace
CUB_NS_POSTFIX  // Optional outer namespace(s)
