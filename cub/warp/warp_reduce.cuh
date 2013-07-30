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

/**
 * \file
 * cub::WarpReduce provides variants of parallel reduction across CUDA warps.
 */

#pragma once

#include "specializations/warp_reduce_shfl.cuh"
#include "specializations/warp_reduce_smem.cuh"
#include "../thread/thread_operators.cuh"
#include "../util_device.cuh"
#include "../util_type.cuh"
#include "../util_namespace.cuh"

/// Optional outer namespace(s)
CUB_NS_PREFIX

/// CUB namespace
namespace cub {


/**
 * \addtogroup WarpModule
 * @{
 */

/**
 * \brief WarpReduce provides variants of parallel reduction across CUDA warps.  ![](warp_reduce_logo.png)
 *
 * \par Overview
 * A <a href="http://en.wikipedia.org/wiki/Reduce_(higher-order_function)"><em>reduction</em></a> (or <em>fold</em>)
 * uses a binary combining operator to compute a single aggregate from a list of input elements.
 *
 * \par
 * For convenience, WarpReduce provides alternative entrypoints that differ by:
 * - Operator (generic reduction <b><em>vs.</em></b> summation of numeric types)
 * - Input validity (full warps <b><em>vs.</em></b> partially-full warps having some undefined elements)
 *
 * \tparam T                        The reduction input/output element type
 * \tparam LOGICAL_WARPS             <b>[optional]</b> The number of entrant "logical" warps performing concurrent warp reductions.  Default is 1.
 * \tparam LOGICAL_WARP_THREADS     <b>[optional]</b> The number of threads per "logical" warp (may be less than the number of hardware warp threads).  Default is the warp size of the targeted CUDA compute-capability (e.g., 32 threads for SM20).
 *
 * \par Usage Considerations
 * - Supports non-commutative reduction operators
 * - Supports "logical" warps smaller than the physical warp size (e.g., a logical warp of 8 threads)
 * - The number of entrant threads must be an multiple of \p LOGICAL_WARP_THREADS
 * - Warp reductions are concurrent if more than one warp is participating
 * - The warp-wide scalar reduction output is only considered valid in warp <em>lane</em><sub>0</sub>
 * - \smemreuse{WarpReduce::TempStorage}

 * \par Performance Considerations
 * - Uses special instructions when applicable (e.g., warp \p SHFL)
 * - Uses synchronization-free communication between warp lanes when applicable
 * - Zero bank conflicts for most types.
 * - Computation is slightly more efficient (i.e., having lower instruction overhead) for:
 *     - Reduction parameterizations where \p T is a built-in C++ primitive or CUDA vector type (e.g.,
 *       \p short, \p int2, \p double, \p float2, etc.)
 *     - Reduction parameterizations where \p LOGICAL_WARP_THREADS is a multiple of the architecture's warp size
 *
 * \par Algorithm
 * These parallel reduction variants implement a warp-synchronous
 * Kogge-Stone algorithm having <em>O</em>(log<em>n</em>)
 * steps and <em>O</em>(<em>n</em>log<em>n</em>) work complexity,
 * where <em>n</em> = \p LOGICAL_WARP_THREADS (which defaults to the warp
 * size associated with the CUDA Compute Capability targeted by the compiler).
 * <br><br>
 * \image html kogge_stone_reduction.png
 * <div class="centercaption">Data flow within a 16-thread Kogge-Stone reduction construction.  Junctions represent binary operators.</div>
 * <br>
 *
 * \par Examples
 *
 * \par
 * <em><b>Example 1</b></em>. Perform a simple sum reduction for one warp
 * \code
 * #include <cub/cub.cuh>
 *
 * __global__ void ExampleKernel(...)
 * {
 *     // Parameterize WarpReduce for 1 warp on type int
 *     typedef cub::WarpReduce<int, 1> WarpReduce;
 *
 *     // Opaque shared memory for WarpReduce
 *     __shared__ typename WarpReduce::TempStorage temp_storage;
 *
 *     // Compute sum of thread ranks in first warp
 *     if (threadIdx.x < 32)
 *     {
 *         int input = threadIdx.x;
 *         int output = WarpReduce(temp_storage).Sum(input);
 *
 *         ...
 *
 * \endcode
 *
 * \par
 * <em><b>Example 2</b></em>. Perform a simple max in every warp
 * \code
 * #include <cub/cub.cuh>
 *
 * // Max functor
 * struct Max
 * {
 *     __host__ __device__ __forceinline__ T operator()(const int &a, const int &b)
 *     {
 *         return (b > a) ? b : a;
 *     }
 * };
 *
 * template <int BLOCK_THREADS>
 * __global__ void ExampleKernel(...)
 * {
 *     const int WARPS = BLOCK_THREADS / 32;
 *
 *     // Parameterize WarpReduce for all active warp on type int
 *     typedef cub::WarpReduce<int, WARPS> WarpReduce;
 *
 *     // Opaque shared memory for WarpReduce
 *     __shared__ typename WarpReduce::TempStorage temp_storage;
 *
 *     // Compute max thread rank every warp
 *     int input = threadIdx.x;
 *     int output = WarpReduce(temp_storage).Reduce(input, Max());
 *
 *     ...
 *
 * \endcode
 */
template <
    typename    T,
    int         LOGICAL_WARPS           = 1,
    int         LOGICAL_WARP_THREADS    = PtxArchProps::WARP_THREADS>
class WarpReduce
{
private:

    /******************************************************************************
     * Constants and typedefs
     ******************************************************************************/

    enum
    {
        POW_OF_TWO = ((LOGICAL_WARP_THREADS & (LOGICAL_WARP_THREADS - 1)) == 0),
    };

public:

    #ifndef DOXYGEN_SHOULD_SKIP_THIS    // Do not document

    /// Internal specialization.  Use SHFL-based reduction if (architecture is >= SM30) and ((only one logical warp) or (LOGICAL_WARP_THREADS is a power-of-two))
    typedef typename If<(CUB_PTX_ARCH >= 300) && ((LOGICAL_WARPS == 1) || POW_OF_TWO),
        WarpReduceShfl<T, LOGICAL_WARPS, LOGICAL_WARP_THREADS>,
        WarpReduceSmem<T, LOGICAL_WARPS, LOGICAL_WARP_THREADS> >::Type InternalWarpReduce;

    #endif // DOXYGEN_SHOULD_SKIP_THIS


private:

    /// Shared memory storage layout type for WarpReduce
    typedef typename InternalWarpReduce::TempStorage _TempStorage;


    /******************************************************************************
     * Thread fields
     ******************************************************************************/

    /// Shared storage reference
    _TempStorage &temp_storage;

    /// Warp ID
    int warp_id;

    /// Lane ID
    int lane_id;


    /******************************************************************************
     * Utility methods
     ******************************************************************************/

    /// Internal storage allocator
    __device__ __forceinline__ _TempStorage& PrivateStorage()
    {
        __shared__ TempStorage private_storage;
        return private_storage;
    }


public:

    /// \smemstorage{WarpReduce}
    typedef _TempStorage TempStorage;


    /******************************************************************//**
     * \name Collective constructors
     *********************************************************************/
    //@{


    /**
     * \brief Collective constructor for 1D thread blocks using a private static allocation of shared memory as temporary storage.  Logical warp and lane identifiers are constructed from <tt>threadIdx.x</tt>.
     *
     */
    __device__ __forceinline__ WarpReduce()
    :
        temp_storage(PrivateStorage()),
        warp_id((LOGICAL_WARPS == 1) ?
            0 :
            threadIdx.x / LOGICAL_WARP_THREADS),
        lane_id(((LOGICAL_WARPS == 1) || (LOGICAL_WARP_THREADS == PtxArchProps::WARP_THREADS)) ?
            LaneId() :
            threadIdx.x % LOGICAL_WARP_THREADS)
    {}


    /**
     * \brief Collective constructor for 1D thread blocks using the specified memory allocation as temporary storage.  Logical warp and lane identifiers are constructed from <tt>threadIdx.x</tt>.
     */
    __device__ __forceinline__ WarpReduce(
        TempStorage &temp_storage)             ///< [in] Reference to memory allocation having layout type TempStorage
    :
        temp_storage(temp_storage),
        warp_id((LOGICAL_WARPS == 1) ?
            0 :
            threadIdx.x / LOGICAL_WARP_THREADS),
        lane_id(((LOGICAL_WARPS == 1) || (LOGICAL_WARP_THREADS == PtxArchProps::WARP_THREADS)) ?
            LaneId() :
            threadIdx.x % LOGICAL_WARP_THREADS)
    {}


    /**
     * \brief Collective constructor using a private static allocation of shared memory as temporary storage.  Threads are identified using the given warp and lane identifiers.
     */
    __device__ __forceinline__ WarpReduce(
        int warp_id,                           ///< [in] A suitable warp membership identifier
        int lane_id)                           ///< [in] A lane identifier within the warp
    :
        temp_storage(PrivateStorage()),
        warp_id(warp_id),
        lane_id(lane_id)
    {}


    /**
     * \brief Collective constructor using the specified memory allocation as temporary storage.  Threads are identified using the given warp and lane identifiers.
     */
    __device__ __forceinline__ WarpReduce(
        TempStorage &temp_storage,             ///< [in] Reference to memory allocation having layout type TempStorage
        int warp_id,                           ///< [in] A suitable warp membership identifier
        int lane_id)                           ///< [in] A lane identifier within the warp
    :
        temp_storage(temp_storage),
        warp_id(warp_id),
        lane_id(lane_id)
    {}



    //@}  end member group
    /******************************************************************//**
     * \name Summation reductions
     *********************************************************************/
    //@{


    /**
     * \brief Computes a warp-wide sum in each active warp.  The output is valid in warp <em>lane</em><sub>0</sub>.
     *
     * \smemreuse
     */
    __device__ __forceinline__ T Sum(
        T                   input)              ///< [in] Calling thread's input
    {
        return InternalWarpReduce(temp_storage, warp_id, lane_id).Sum<true, 1>(input, LOGICAL_WARP_THREADS);
    }

    /**
     * \brief Computes a partially-full warp-wide sum in each active warp.  The output is valid in warp <em>lane</em><sub>0</sub>.
     *
     * All threads in each logical warp must agree on the same value for \p valid_items.  Otherwise the result is undefined.
     *
     * \smemreuse
     */
    __device__ __forceinline__ T Sum(
        T                   input,              ///< [in] Calling thread's input
        int                 valid_items)        ///< [in] Total number of valid items in the calling thread's logical warp (may be less than \p LOGICAL_WARP_THREADS)
    {
        // Determine if we don't need bounds checking
        if (valid_items >= LOGICAL_WARP_THREADS)
        {
            return InternalWarpReduce(temp_storage, warp_id, lane_id).Sum<true, 1>(input, valid_items);
        }
        else
        {
            return InternalWarpReduce(temp_storage, warp_id, lane_id).Sum<false, 1>(input, valid_items);
        }
    }


    /**
     * \brief Computes a segmented sum in each active warp where segments are defined by head-flags.  The sum of each segment is returned to the first lane in that segment (which always includes <em>lane</em><sub>0</sub>).
     *
     * \smemreuse
     *
     * \tparam ReductionOp     <b>[inferred]</b> Binary scan operator type having member <tt>T operator()(const T &a, const T &b)</tt>
     */
    template <
        typename            Flag>
    __device__ __forceinline__ T HeadSegmentedSum(
        T                   input,              ///< [in] Calling thread's input
        Flag                head_flag)          ///< [in] Head flag denoting whether or not \p input is the start of a new segment
    {
        return HeadSegmentedReduce(input, head_flag, cub::Sum());
    }


    /**
     * \brief Computes a segmented sum in each active warp where segments are defined by tail-flags.  The sum of each segment is returned to the first lane in that segment (which always includes <em>lane</em><sub>0</sub>).
     *
     * \smemreuse
     *
     * \tparam ReductionOp     <b>[inferred]</b> Binary scan operator type having member <tt>T operator()(const T &a, const T &b)</tt>
     */
    template <
        typename            Flag>
    __device__ __forceinline__ T TailSegmentedSum(
        T                   input,              ///< [in] Calling thread's input
        Flag                tail_flag)          ///< [in] Head flag denoting whether or not \p input is the start of a new segment
    {
        return TailSegmentedReduce(input, tail_flag, cub::Sum());
    }



    //@}  end member group
    /******************************************************************//**
     * \name Generic reductions
     *********************************************************************/
    //@{

    /**
     * \brief Computes a warp-wide reduction in each active warp using the specified binary reduction functor.  The output is valid in warp <em>lane</em><sub>0</sub>.
     *
     * \smemreuse
     *
     * \tparam ReductionOp     <b>[inferred]</b> Binary scan operator type having member <tt>T operator()(const T &a, const T &b)</tt>
     */
    template <typename ReductionOp>
    __device__ __forceinline__ T Reduce(
        T                   input,              ///< [in] Calling thread's input
        ReductionOp         reduction_op)       ///< [in] Binary reduction operator
    {
        return InternalWarpReduce(temp_storage, warp_id, lane_id).Reduce<true, 1>(input, LOGICAL_WARP_THREADS, reduction_op);
    }

    /**
     * \brief Computes a partially-full warp-wide reduction in each active warp using the specified binary reduction functor.  The output is valid in warp <em>lane</em><sub>0</sub>.
     *
     * All threads in each logical warp must agree on the same value for \p valid_items.  Otherwise the result is undefined.
     *
     * \smemreuse
     *
     * \tparam ReductionOp     <b>[inferred]</b> Binary scan operator type having member <tt>T operator()(const T &a, const T &b)</tt>
     */
    template <typename ReductionOp>
    __device__ __forceinline__ T Reduce(
        T                   input,              ///< [in] Calling thread's input
        ReductionOp         reduction_op,       ///< [in] Binary reduction operator
        int                 valid_items)        ///< [in] Total number of valid items in the calling thread's logical warp (may be less than \p LOGICAL_WARP_THREADS)
    {
        // Determine if we don't need bounds checking
        if (valid_items >= LOGICAL_WARP_THREADS)
        {
            return InternalWarpReduce(temp_storage, warp_id, lane_id).Reduce<true, 1>(input, valid_items, reduction_op);
        }
        else
        {
            return InternalWarpReduce(temp_storage, warp_id, lane_id).Reduce<false, 1>(input, valid_items, reduction_op);
        }
    }


    /**
     * \brief Computes a segmented reduction in each active warp where segments are defined by head-flags.  The reduction of each segment is returned to the first lane in that segment (which always includes <em>lane</em><sub>0</sub>).
     *
     * \smemreuse
     *
     * \tparam ReductionOp     <b>[inferred]</b> Binary scan operator type having member <tt>T operator()(const T &a, const T &b)</tt>
     */
    template <
        typename            ReductionOp,
        typename            Flag>
    __device__ __forceinline__ T HeadSegmentedReduce(
        T                   input,              ///< [in] Calling thread's input
        Flag                head_flag,          ///< [in] Head flag denoting whether or not \p input is the start of a new segment
        ReductionOp         reduction_op)       ///< [in] Reduction operator
    {
        return InternalWarpReduce(temp_storage, warp_id, lane_id).template SegmentedReduce<true>(input, head_flag, reduction_op);
    }


    /**
     * \brief Computes a segmented reduction in each active warp where segments are defined by tail-flags.  The reduction of each segment is returned to the first lane in that segment (which always includes <em>lane</em><sub>0</sub>).
     *
     * \smemreuse
     *
     * \tparam ReductionOp     <b>[inferred]</b> Binary scan operator type having member <tt>T operator()(const T &a, const T &b)</tt>
     */
    template <
        typename            ReductionOp,
        typename            Flag>
    __device__ __forceinline__ T TailSegmentedReduce(
        T                   input,              ///< [in] Calling thread's input
        Flag                tail_flag,          ///< [in] Tail flag denoting whether or not \p input is the end of the current segment
        ReductionOp         reduction_op)       ///< [in] Reduction operator
    {
        return InternalWarpReduce(temp_storage, warp_id, lane_id).template SegmentedReduce<false>(input, tail_flag, reduction_op);
    }



    //@}  end member group
};

/** @} */       // end group WarpModule

}               // CUB namespace
CUB_NS_POSTFIX  // Optional outer namespace(s)
