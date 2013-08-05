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
 * cub::WarpScan provides variants of parallel prefix scan across CUDA warps.
 */

#pragma once

#include "specializations/warp_scan_shfl.cuh"
#include "specializations/warp_scan_smem.cuh"
#include "../thread/thread_operators.cuh"
#include "../util_debug.cuh"
#include "../util_arch.cuh"
#include "../util_type.cuh"
#include "../util_ptx.cuh"
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
 * \brief WarpScan provides variants of parallel prefix scan across CUDA warps.  ![](warp_scan_logo.png)
 *
 * \par Overview
 * Given a list of input elements and a binary reduction operator, a [<em>prefix scan</em>](http://en.wikipedia.org/wiki/Prefix_sum)
 * produces an output list where each element is computed to be the reduction
 * of the elements occurring earlier in the input list.  <em>Prefix sum</em>
 * connotes a prefix scan with the addition operator. The term \em inclusive indicates
 * that the <em>i</em><sup>th</sup> output reduction incorporates the <em>i</em><sup>th</sup> input.
 * The term \em exclusive indicates the <em>i</em><sup>th</sup> input is not incorporated into
 * the <em>i</em><sup>th</sup> output reduction.
 *
 * \par
 * For convenience, WarpScan provides alternative entrypoints that differ by:
 * - Operator (generic scan <b><em>vs.</em></b> prefix sum of numeric types)
 * - Output ordering (inclusive <b><em>vs.</em></b> exclusive)
 * - Warp-wide prefix (identity <b><em>vs.</em></b> call-back functor)
 * - What is computed (scanned elements only <b><em>vs.</em></b> scanned elements and the total aggregate)
 *
 * \tparam T                        The scan input/output element type
 * \tparam LOGICAL_WARPS                    <b>[optional]</b> The number of "logical" warps performing concurrent warp scans. Default is 1.
 * \tparam LOGICAL_WARP_THREADS     <b>[optional]</b> The number of threads per "logical" warp (may be less than the number of hardware warp threads).  Default is the warp size associated with the CUDA Compute Capability targeted by the compiler (e.g., 32 threads for SM20).
 *
 * \par Usage Considerations
 * - Supports non-commutative scan operators
 * - Supports "logical" warps smaller than the physical warp size (e.g., a logical warp of 8 threads)
 * - Warp scans are concurrent if more than one warp is participating
 * - \smemreuse{WarpScan::TempStorage}

 * \par Performance Considerations
 * - Uses special instructions when applicable (e.g., warp \p SHFL)
 * - Uses synchronization-free communication between warp lanes when applicable
 * - Zero bank conflicts for most types.
 * - Computation is slightly more efficient (i.e., having lower instruction overhead) for:
 *     - Prefix sum variants (vs. generic scan)
 *     - Exclusive variants (vs. inclusive)
 *     - Basic scan variants that don't require scalar inputs and outputs (e.g., \p warp_prefix_op and \p warp_aggregate)
 *     - Scan parameterizations where \p T is a built-in C++ primitive or CUDA vector type (e.g.,
 *       \p short, \p int2, \p double, \p float2, etc.)
 *     - Scan parameterizations where \p LOGICAL_WARP_THREADS is a multiple of the architecture's warp size
 *
 * \par Algorithm
 * These parallel prefix scan variants implement a warp-synchronous
 * Kogge-Stone algorithm having <em>O</em>(log<em>n</em>)
 * steps and <em>O</em>(<em>n</em>log<em>n</em>) work complexity,
 * where <em>n</em> = \p LOGICAL_WARP_THREADS (which defaults to the warp
 * size associated with the CUDA Compute Capability targeted by the compiler).
 * <br><br>
 * \image html kogge_stone_scan.png
 * <div class="centercaption">Data flow within a 16-thread Kogge-Stone scan construction.  Junctions represent binary operators.</div>
 * <br>
 *
 * \par Examples
 * <em>Example 1.</em> Perform a simple exclusive prefix sum for one warp
 * \code
 * #include <cub/cub.cuh>
 *
 * __global__ void ExampleKernel(...)
 * {
 *     // Parameterize WarpScan for 1 warp on type int
 *     typedef cub::WarpScan<int> WarpScan;
 *
 *     // Opaque shared memory for WarpScan
 *     __shared__ typename WarpScan::TempStorage temp_storage;
 *
 *     // Perform prefix sum of threadIds in first warp
 *     if (linear_tid < 32)
 *     {
 *         int input = linear_tid;
 *         int output;
 *         WarpScan::ExclusiveSum(temp_storage, input, output);
 *
 *         printf("tid(%d) output(%d)\n\n", linear_tid, output);
 *     }
 * \endcode
 * Printed output:
 * \code
 * tid(0) output(0)
 * tid(1) output(0)
 * tid(2) output(1)
 * tid(3) output(3)
 * tid(4) output(6)
 * ...
 * tid(31) output(465)
 * \endcode
 *
 * \par
 * <em>Example 2.</em> Use a single warp to iteratively compute an exclusive prefix sum over a larger input using a prefix functor to maintain a running total between scans.
 *      \code
 *      #include <cub/cub.cuh>
 *
 *      // Stateful functor that maintains a running prefix that can be applied to
 *      // consecutive scan operations.
 *      struct WarpPrefixOp
 *      {
 *          // Running prefix
 *          int running_total;
 *
 *          // Constructor
 *          __device__ WarpPrefixOp(int running_total) : running_total(running_total) {}
 *
 *          // Callback operator, called by the entire warp of threads.
 *          // Lane-0 produces a value for seeding the warp-wide scan given
 *          // the local aggregate.
 *          __device__ int operator()(int warp_aggregate)
 *          {
 *              int old_prefix = running_total;
 *              running_total += warp_aggregate;
 *              return old_prefix;
 *          }
 *      }
 *
 *      __global__ void ExampleKernel(int *d_data, int num_items)
 *      {
 *          // Parameterize WarpScan for 1 warp on type int
 *          typedef cub::WarpScan<int> WarpScan;
 *
 *          // Opaque shared memory for WarpScan
 *          __shared__ typename WarpScan::TempStorage temp_storage;
 *
 *          // The first warp iteratively computes a prefix sum over d_data
 *          if (linear_tid < 32)
 *          {
 *              // Running total
 *              WarpPrefixOp prefix_op(0);
 *
 *              // Iterate in strips of 32 items
 *              for (int warp_offset = 0; warp_offset < num_items; warp_offset += 32)
 *              {
 *                  // Read item
 *                  int datum = d_data[warp_offset + linear_tid];
 *
 *                  // Scan the tile of items
 *                  int tile_aggregate;
 *                  WarpScan::ExclusiveSum(temp_storage, datum, datum,
 *                      tile_aggregate, prefix_op);
 *
 *                  // Write item
 *                  d_data[warp_offset + linear_tid] = datum;
 *              }
 *          }
 *      \endcode
 */
template <
    typename    T,
    int         LOGICAL_WARPS           = 1,
    int         LOGICAL_WARP_THREADS    = PtxArchProps::WARP_THREADS>
class WarpScan
{
private:

    /******************************************************************************
     * Constants and typedefs
     ******************************************************************************/

    enum
    {
        POW_OF_TWO = ((LOGICAL_WARP_THREADS & (LOGICAL_WARP_THREADS - 1)) == 0),
    };

    /// Internal specialization.  Use SHFL-based reduction if (architecture is >= SM30) and ((only one logical warp) or (LOGICAL_WARP_THREADS is a power-of-two))
    typedef typename If<(CUB_PTX_ARCH >= 300) && ((LOGICAL_WARPS == 1) || POW_OF_TWO),
        WarpScanShfl<T, LOGICAL_WARPS, LOGICAL_WARP_THREADS>,
        WarpScanSmem<T, LOGICAL_WARPS, LOGICAL_WARP_THREADS> >::Type InternalWarpScan;

    /// Shared memory storage layout type for WarpScan
    typedef typename InternalWarpScan::TempStorage _TempStorage;


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

    /// \smemstorage{WarpScan}
    typedef _TempStorage TempStorage;


    /******************************************************************//**
     * \name Collective constructors
     *********************************************************************/
    //@{

    /**
     * \brief Collective constructor for 1D thread blocks using a private static allocation of shared memory as temporary storage.  Logical warp and lane identifiers are constructed from <tt>threadIdx.x</tt>.
     */
    __device__ __forceinline__ WarpScan()
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
    __device__ __forceinline__ WarpScan(
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
    __device__ __forceinline__ WarpScan(
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
    __device__ __forceinline__ WarpScan(
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
     * \name Inclusive prefix sums
     *********************************************************************/
    //@{


    /**
     * \brief Computes an inclusive prefix sum in each logical warp.
     *
     * \smemreuse
     */
    __device__ __forceinline__ void InclusiveSum(
        T               input,              ///< [in] Calling thread's input item.
        T               &output)            ///< [out] Calling thread's output item.  May be aliased with \p input.
    {
        InternalWarpScan(temp_storage, warp_id, lane_id).InclusiveSum(input, output);
    }


    /**
     * \brief Computes an inclusive prefix sum in each logical warp.  Also provides every thread with the warp-wide \p warp_aggregate of all inputs.
     *
     * The \p warp_aggregate is undefined in threads other than <em>warp-lane</em><sub>0</sub>.
     *
     * \smemreuse
     */
    __device__ __forceinline__ void InclusiveSum(
        T               input,              ///< [in] Calling thread's input item.
        T               &output,            ///< [out] Calling thread's output item.  May be aliased with \p input.
        T               &warp_aggregate)    ///< [out] Warp-wide aggregate reduction of input items.
    {
        InternalWarpScan(temp_storage, warp_id, lane_id).InclusiveSum(input, output, warp_aggregate);
    }


    /**
     * \brief Computes an inclusive prefix sum in each logical warp.  Instead of using 0 as the warp-wide prefix, the call-back functor \p warp_prefix_op is invoked to provide the "seed" value that logically prefixes the warp's scan inputs.  Also provides every thread with the warp-wide \p warp_aggregate of all inputs.
     *
     * The \p warp_aggregate is undefined in threads other than <em>warp-lane</em><sub>0</sub>.
     *
     * The \p warp_prefix_op functor must implement a member function <tt>T operator()(T warp_aggregate)</tt>.
     * The functor's input parameter \p warp_aggregate is the same value also returned by the scan operation.
     * The functor will be invoked by the entire warp of threads, however only the return value from
     * <em>lane</em><sub>0</sub> is applied as the threadblock-wide prefix.  Can be stateful.
     *
     * \smemreuse
     *
     * \tparam WarpPrefixOp                 <b>[inferred]</b> Call-back functor type having member <tt>T operator()(T warp_aggregate)</tt>
     */
    template <typename WarpPrefixOp>
    __device__ __forceinline__ void InclusiveSum(
        T               input,              ///< [in] Calling thread's input item.
        T               &output,            ///< [out] Calling thread's output item.  May be aliased with \p input.
        T               &warp_aggregate,    ///< [out] <b>[<em>warp-lane</em><sub>0</sub> only]</b> Warp-wide aggregate reduction of input items, exclusive of the \p warp_prefix_op value
        WarpPrefixOp    &warp_prefix_op)    ///< [in-out] <b>[<em>warp-lane</em><sub>0</sub> only]</b> Call-back functor for specifying a warp-wide prefix to be applied to all inputs.
    {
        // Compute inclusive warp scan
        InclusiveSum(input, output, warp_aggregate);

        // Compute warp-wide prefix from aggregate, then broadcast to other lanes
        T prefix;
        prefix = warp_prefix_op(warp_aggregate);
        prefix = InternalWarpScan(temp_storage, warp_id, lane_id).Broadcast(prefix, 0);

        // Update output
        output = prefix + output;
    }

    //@}  end member group

private:

    /// Computes an exclusive prefix sum in each logical warp.
    __device__ __forceinline__ void ExclusiveSum(T input, T &output, Int2Type<true> is_primitive)
    {
        // Compute exclusive warp scan from inclusive warp scan
        T inclusive;
        InclusiveSum(input, inclusive);
        output = inclusive - input;
    }

    /// Computes an exclusive prefix sum in each logical warp.  Specialized for non-primitive types.
    __device__ __forceinline__ void ExclusiveSum(T input, T &output, Int2Type<false> is_primitive)
    {
        // Delegate to regular scan for non-primitive types (because we won't be able to use subtraction)
        T identity = T();
        ExclusiveScan(input, output, identity, Sum());
    }

    /// Computes an exclusive prefix sum in each logical warp.  Also provides every thread with the warp-wide \p warp_aggregate of all inputs.
    __device__ __forceinline__ void ExclusiveSum(T input, T &output, T &warp_aggregate, Int2Type<true> is_primitive)
    {
        // Compute exclusive warp scan from inclusive warp scan
        T inclusive;
        InclusiveSum(input, inclusive, warp_aggregate);
        output = inclusive - input;
    }

    /// Computes an exclusive prefix sum in each logical warp.  Also provides every thread with the warp-wide \p warp_aggregate of all inputs.  Specialized for non-primitive types.
    __device__ __forceinline__ void ExclusiveSum(T input, T &output, T &warp_aggregate, Int2Type<false> is_primitive)
    {
        // Delegate to regular scan for non-primitive types (because we won't be able to use subtraction)
        T identity = T();
        ExclusiveScan(input, output, identity, Sum(), warp_aggregate);
    }

    /// Computes an exclusive prefix sum in each logical warp.  Instead of using 0 as the warp-wide prefix, the call-back functor \p warp_prefix_op is invoked to provide the "seed" value that logically prefixes the warp's scan inputs.  Also provides every thread with the warp-wide \p warp_aggregate of all inputs.
    template <typename WarpPrefixOp>
    __device__ __forceinline__ void ExclusiveSum(T input, T &output, T &warp_aggregate, WarpPrefixOp &warp_prefix_op, Int2Type<true> is_primitive)
    {
        // Compute exclusive warp scan from inclusive warp scan
        T inclusive;
        InclusiveSum(input, inclusive, warp_aggregate, warp_prefix_op);
        output = inclusive - input;
    }

    /// Computes an exclusive prefix sum in each logical warp.  Instead of using 0 as the warp-wide prefix, the call-back functor \p warp_prefix_op is invoked to provide the "seed" value that logically prefixes the warp's scan inputs.  Also provides every thread with the warp-wide \p warp_aggregate of all inputs.  Specialized for non-primitive types.
    template <typename WarpPrefixOp>
    __device__ __forceinline__ void ExclusiveSum(T input, T &output, T &warp_aggregate, WarpPrefixOp &warp_prefix_op, Int2Type<false> is_primitive)
    {
        // Delegate to regular scan for non-primitive types (because we won't be able to use subtraction)
        T identity = T();
        ExclusiveScan(input, output, identity, Sum(), warp_aggregate, warp_prefix_op);
    }

public:


    /******************************************************************//**
     * \name Exclusive prefix sums
     *********************************************************************/
    //@{


    /**
     * \brief Computes an exclusive prefix sum in each logical warp.
     *
     * This operation assumes the value of obtained by the <tt>T</tt>'s default
     * constructor (or by zero-initialization if no user-defined default
     * constructor exists) is suitable as the identity value "zero" for
     * addition.
     *
     * \smemreuse
     */
    __device__ __forceinline__ void ExclusiveSum(
        T               input,              ///< [in] Calling thread's input item.
        T               &output)            ///< [out] Calling thread's output item.  May be aliased with \p input.
    {
        ExclusiveSum(input, output, Int2Type<Traits<T>::PRIMITIVE>());
    }


    /**
     * \brief Computes an exclusive prefix sum in each logical warp.  Also provides every thread with the warp-wide \p warp_aggregate of all inputs.
     *
     * This operation assumes the value of obtained by the <tt>T</tt>'s default
     * constructor (or by zero-initialization if no user-defined default
     * constructor exists) is suitable as the identity value "zero" for
     * addition.
     *
     * \smemreuse
     */
    __device__ __forceinline__ void ExclusiveSum(
        T               input,              ///< [in] Calling thread's input item.
        T               &output,            ///< [out] Calling thread's output item.  May be aliased with \p input.
        T               &warp_aggregate)    ///< [out] Warp-wide aggregate reduction of input items.
    {
        ExclusiveSum(input, output, warp_aggregate, Int2Type<Traits<T>::PRIMITIVE>());
    }


    /**
     * \brief Computes an exclusive prefix sum in each logical warp.  Instead of using 0 as the warp-wide prefix, the call-back functor \p warp_prefix_op is invoked to provide the "seed" value that logically prefixes the warp's scan inputs.  Also provides every thread with the warp-wide \p warp_aggregate of all inputs.
     *
     * This operation assumes the value of obtained by the <tt>T</tt>'s default
     * constructor (or by zero-initialization if no user-defined default
     * constructor exists) is suitable as the identity value "zero" for
     * addition.
     *
     * The \p warp_prefix_op functor must implement a member function <tt>T operator()(T warp_aggregate)</tt>.
     * The functor's input parameter \p warp_aggregate is the same value also returned by the scan operation.
     * The functor will be invoked by the entire warp of threads, however only the return value from
     * <em>lane</em><sub>0</sub> is applied as the threadblock-wide prefix.  Can be stateful.
     *
     * \smemreuse
     *
     * \tparam WarpPrefixOp                 <b>[inferred]</b> Call-back functor type having member <tt>T operator()(T warp_aggregate)</tt>
     */
    template <typename WarpPrefixOp>
    __device__ __forceinline__ void ExclusiveSum(
        T               input,              ///< [in] Calling thread's input item.
        T               &output,            ///< [out] Calling thread's output item.  May be aliased with \p input.
        T               &warp_aggregate,    ///< [out] <b>[<em>warp-lane</em><sub>0</sub> only]</b> Warp-wide aggregate reduction of input items (exclusive of the \p warp_prefix_op value).
        WarpPrefixOp    &warp_prefix_op)    ///< [in-out] <b>[<em>warp-lane</em><sub>0</sub> only]</b> Call-back functor for specifying a warp-wide prefix to be applied to all inputs.
    {
        ExclusiveSum(input, output, warp_aggregate, warp_prefix_op, Int2Type<Traits<T>::PRIMITIVE>());
    }


    //@}  end member group
    /******************************************************************//**
     * \name Inclusive prefix scans
     *********************************************************************/
    //@{

    /**
     * \brief Computes an inclusive prefix sum using the specified binary scan functor in each logical warp.
     *
     * \smemreuse
     *
     * \tparam ScanOp     <b>[inferred]</b> Binary scan operator type having member <tt>T operator()(const T &a, const T &b)</tt>
     */
    template <typename ScanOp>
    __device__ __forceinline__ void InclusiveScan(
        T               input,              ///< [in] Calling thread's input item.
        T               &output,            ///< [out] Calling thread's output item.  May be aliased with \p input.
        ScanOp          scan_op)            ///< [in] Binary scan operator
    {
        InternalWarpScan(temp_storage, warp_id, lane_id).InclusiveScan(input, output, scan_op);
    }


    /**
     * \brief Computes an inclusive prefix sum using the specified binary scan functor in each logical warp.  Also provides every thread with the warp-wide \p warp_aggregate of all inputs.
     *
     * The \p warp_aggregate is undefined in threads other than <em>warp-lane</em><sub>0</sub>.
     *
     * \smemreuse
     *
     * \tparam ScanOp     <b>[inferred]</b> Binary scan operator type having member <tt>T operator()(const T &a, const T &b)</tt>
     */
    template <typename ScanOp>
    __device__ __forceinline__ void InclusiveScan(
        T               input,              ///< [in] Calling thread's input item.
        T               &output,            ///< [out] Calling thread's output item.  May be aliased with \p input.
        ScanOp          scan_op,            ///< [in] Binary scan operator
        T               &warp_aggregate)    ///< [out] Warp-wide aggregate reduction of input items.
    {
        InternalWarpScan(temp_storage, warp_id, lane_id).InclusiveScan(input, output, scan_op, warp_aggregate);
    }


    /**
     * \brief Computes an inclusive prefix sum using the specified binary scan functor in each logical warp.  The call-back functor \p warp_prefix_op is invoked to provide the "seed" value that logically prefixes the warp's scan inputs.  Also provides every thread with the warp-wide \p warp_aggregate of all inputs.
     *
     * The \p warp_aggregate is undefined in threads other than <em>warp-lane</em><sub>0</sub>.
     *
     * The \p warp_prefix_op functor must implement a member function <tt>T operator()(T warp_aggregate)</tt>.
     * The functor's input parameter \p warp_aggregate is the same value also returned by the scan operation.
     * The functor will be invoked by the entire warp of threads, however only the return value from
     * <em>lane</em><sub>0</sub> is applied as the threadblock-wide prefix.  Can be stateful.
     *
     * \smemreuse
     *
     * \tparam ScanOp                       <b>[inferred]</b> Binary scan operator type having member <tt>T operator()(const T &a, const T &b)</tt>
     * \tparam WarpPrefixOp                 <b>[inferred]</b> Call-back functor type having member <tt>T operator()(T warp_aggregate)</tt>
     */
    template <
        typename ScanOp,
        typename WarpPrefixOp>
    __device__ __forceinline__ void InclusiveScan(
        T               input,              ///< [in] Calling thread's input item.
        T               &output,            ///< [out] Calling thread's output item.  May be aliased with \p input.
        ScanOp          scan_op,            ///< [in] Binary scan operator
        T               &warp_aggregate,    ///< [out] <b>[<em>warp-lane</em><sub>0</sub> only]</b> Warp-wide aggregate reduction of input items (exclusive of the \p warp_prefix_op value).
        WarpPrefixOp    &warp_prefix_op)    ///< [in-out] <b>[<em>warp-lane</em><sub>0</sub> only]</b> Call-back functor for specifying a warp-wide prefix to be applied to all inputs.
    {
        // Compute inclusive warp scan
        InclusiveScan(input, output, scan_op, warp_aggregate);

        // Compute warp-wide prefix from aggregate, then broadcast to other lanes
        T prefix;
        prefix = warp_prefix_op(warp_aggregate);
        prefix = InternalWarpScan(temp_storage, warp_id, lane_id).Broadcast(prefix, 0);

        // Update output
        output = scan_op(prefix, output);
    }


    //@}  end member group
    /******************************************************************//**
     * \name Exclusive prefix scans
     *********************************************************************/
    //@{

    /**
     * \brief Computes an exclusive prefix scan using the specified binary scan functor in each logical warp.
     *
     * \smemreuse
     *
     * \tparam ScanOp     <b>[inferred]</b> Binary scan operator type having member <tt>T operator()(const T &a, const T &b)</tt>
     */
    template <typename ScanOp>
    __device__ __forceinline__ void ExclusiveScan(
        T               input,              ///< [in] Calling thread's input item.
        T               &output,            ///< [out] Calling thread's output item.  May be aliased with \p input.
        T               identity,           ///< [in] Identity value
        ScanOp          scan_op)            ///< [in] Binary scan operator
    {
        InternalWarpScan(temp_storage, warp_id, lane_id).ExclusiveScan(input, output, identity, scan_op);
    }


    /**
     * \brief Computes an exclusive prefix scan using the specified binary scan functor in each logical warp.  Also provides every thread with the warp-wide \p warp_aggregate of all inputs.
     *
     * The \p warp_aggregate is undefined in threads other than <em>warp-lane</em><sub>0</sub>.
     *
     * \smemreuse
     *
     * \tparam ScanOp     <b>[inferred]</b> Binary scan operator type having member <tt>T operator()(const T &a, const T &b)</tt>
     */
    template <typename ScanOp>
    __device__ __forceinline__ void ExclusiveScan(
        T               input,              ///< [in] Calling thread's input item.
        T               &output,            ///< [out] Calling thread's output item.  May be aliased with \p input.
        T               identity,           ///< [in] Identity value
        ScanOp          scan_op,            ///< [in] Binary scan operator
        T               &warp_aggregate)    ///< [out] Warp-wide aggregate reduction of input items.
    {
        InternalWarpScan(temp_storage, warp_id, lane_id).ExclusiveScan(input, output, identity, scan_op, warp_aggregate);
    }


    /**
     * \brief Computes an exclusive prefix scan using the specified binary scan functor in each logical warp.  The call-back functor \p warp_prefix_op is invoked to provide the "seed" value that logically prefixes the warp's scan inputs.  Also provides every thread with the warp-wide \p warp_aggregate of all inputs.
     *
     * The \p warp_aggregate is undefined in threads other than <em>warp-lane</em><sub>0</sub>.
     *
     * The \p warp_prefix_op functor must implement a member function <tt>T operator()(T warp_aggregate)</tt>.
     * The functor's input parameter \p warp_aggregate is the same value also returned by the scan operation.
     * The functor will be invoked by the entire warp of threads, however only the return value from
     * <em>lane</em><sub>0</sub> is applied as the threadblock-wide prefix.  Can be stateful.
     *
     * \smemreuse
     *
     * \tparam ScanOp                       <b>[inferred]</b> Binary scan operator type having member <tt>T operator()(const T &a, const T &b)</tt>
     * \tparam WarpPrefixOp                 <b>[inferred]</b> Call-back functor type having member <tt>T operator()(T warp_aggregate)</tt>
     */
    template <
        typename ScanOp,
        typename WarpPrefixOp>
    __device__ __forceinline__ void ExclusiveScan(
        T               input,              ///< [in] Calling thread's input item.
        T               &output,            ///< [out] Calling thread's output item.  May be aliased with \p input.
        T               identity,           ///< [in] Identity value
        ScanOp          scan_op,            ///< [in] Binary scan operator
        T               &warp_aggregate,    ///< [out] <b>[<em>warp-lane</em><sub>0</sub> only]</b> Warp-wide aggregate reduction of input items (exclusive of the \p warp_prefix_op value).
        WarpPrefixOp    &warp_prefix_op)    ///< [in-out] <b>[<em>warp-lane</em><sub>0</sub> only]</b> Call-back functor for specifying a warp-wide prefix to be applied to all inputs.
    {
        // Exclusive warp scan
        ExclusiveScan(input, output, identity, scan_op, warp_aggregate);

        // Compute warp-wide prefix from aggregate, then broadcast to other lanes
        T prefix = warp_prefix_op(warp_aggregate);
        prefix = InternalWarpScan(temp_storage, warp_id, lane_id).Broadcast(prefix, 0);

        // Update output
        output = (lane_id == 0) ?
            prefix :
            scan_op(prefix, output);
    }


    //@}  end member group
    /******************************************************************//**
     * \name Identityless exclusive prefix scans
     *********************************************************************/
    //@{


    /**
     * \brief Computes an exclusive prefix scan using the specified binary scan functor in each logical warp.  Because no identity value is supplied, the \p output computed for <em>warp-lane</em><sub>0</sub> is undefined.
     *
     * \smemreuse
     *
     * \tparam ScanOp     <b>[inferred]</b> Binary scan operator type having member <tt>T operator()(const T &a, const T &b)</tt>
     */
    template <typename ScanOp>
    __device__ __forceinline__ void ExclusiveScan(
        T               input,              ///< [in] Calling thread's input item.
        T               &output,            ///< [out] Calling thread's output item.  May be aliased with \p input.
        ScanOp          scan_op)            ///< [in] Binary scan operator
    {
        InternalWarpScan(temp_storage, warp_id, lane_id).ExclusiveScan(input, output, scan_op);
    }


    /**
     * \brief Computes an exclusive prefix scan using the specified binary scan functor in each logical warp.  Because no identity value is supplied, the \p output computed for <em>warp-lane</em><sub>0</sub> is undefined.  Also provides every thread with the warp-wide \p warp_aggregate of all inputs.
     *
     * The \p warp_aggregate is undefined in threads other than <em>warp-lane</em><sub>0</sub>.
     *
     * \smemreuse
     *
     * \tparam ScanOp     <b>[inferred]</b> Binary scan operator type having member <tt>T operator()(const T &a, const T &b)</tt>
     */
    template <typename ScanOp>
    __device__ __forceinline__ void ExclusiveScan(
        T               input,              ///< [in] Calling thread's input item.
        T               &output,            ///< [out] Calling thread's output item.  May be aliased with \p input.
        ScanOp          scan_op,            ///< [in] Binary scan operator
        T               &warp_aggregate)    ///< [out] Warp-wide aggregate reduction of input items.
    {
        InternalWarpScan(temp_storage, warp_id, lane_id).ExclusiveScan(input, output, scan_op, warp_aggregate);
    }


    /**
     * \brief Computes an exclusive prefix scan using the specified binary scan functor in each logical warp.  The \p warp_prefix_op value from thread-thread-lane<sub>0</sub> is applied to all scan outputs.  Also computes the warp-wide \p warp_aggregate of all inputs for thread-thread-lane<sub>0</sub>.
     *
     * The \p warp_aggregate is undefined in threads other than <em>warp-lane</em><sub>0</sub>.
     *
     * The \p warp_prefix_op functor must implement a member function <tt>T operator()(T warp_aggregate)</tt>.
     * The functor's input parameter \p warp_aggregate is the same value also returned by the scan operation.
     * The functor will be invoked by the entire warp of threads, however only the return value from
     * <em>lane</em><sub>0</sub> is applied as the threadblock-wide prefix.  Can be stateful.
     *
     * \smemreuse
     *
     * \tparam ScanOp                       <b>[inferred]</b> Binary scan operator type having member <tt>T operator()(const T &a, const T &b)</tt>
     * \tparam WarpPrefixOp                 <b>[inferred]</b> Call-back functor type having member <tt>T operator()(T warp_aggregate)</tt>
     */
    template <
        typename ScanOp,
        typename WarpPrefixOp>
    __device__ __forceinline__ void ExclusiveScan(
        T               input,              ///< [in] Calling thread's input item.
        T               &output,            ///< [out] Calling thread's output item.  May be aliased with \p input.
        ScanOp          scan_op,            ///< [in] Binary scan operator
        T               &warp_aggregate,    ///< [out] <b>[<em>warp-lane</em><sub>0</sub> only]</b> Warp-wide aggregate reduction of input items (exclusive of the \p warp_prefix_op value).
        WarpPrefixOp    &warp_prefix_op)    ///< [in-out] <b>[<em>warp-lane</em><sub>0</sub> only]</b> Call-back functor for specifying a warp-wide prefix to be applied to all inputs.
    {
        // Exclusive warp scan
        ExclusiveScan(input, output, scan_op, warp_aggregate);

        // Compute warp-wide prefix from aggregate, then broadcast to other lanes
        T prefix = warp_prefix_op(warp_aggregate);
        prefix = InternalWarpScan(temp_storage, warp_id, lane_id).Broadcast(prefix, 0);

        // Update output with prefix
        output = (lane_id == 0) ?
            prefix :
            scan_op(prefix, output);
    }

    //@}  end member group
};

/** @} */       // end group WarpModule

}               // CUB namespace
CUB_NS_POSTFIX  // Optional outer namespace(s)
