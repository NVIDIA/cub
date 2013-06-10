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

#include "../thread/thread_load.cuh"
#include "../thread/thread_store.cuh"
#include "../util_debug.cuh"
#include "../util_arch.cuh"
#include "../util_type.cuh"
#include "../util_ptx.cuh"
#include "../thread/thread_operators.cuh"
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
 * - Operator (generic scan <em>vs.</em> prefix sum of numeric types)
 * - Output ordering (inclusive <em>vs.</em> exclusive)
 * - Warp-wide prefix (identity <em>vs.</em> call-back functor)
 * - What is computed (scanned elements only <em>vs.</em> scanned elements and the total aggregate)
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
 * __global__ void SomeKernel(...)
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
 *          // Functor constructor
 *          __device__ WarpPrefixOp(int running_total) : running_total(running_total) {}
 *
 *          // Functor operator.  Lane-0 produces a value for seeding the warp-wide scan given
 *          // the local aggregate.
 *          __device__ int operator()(int warp_aggregate)
 *          {
 *              int old_prefix = running_total;
 *              running_total += warp_aggregate;
 *              return old_prefix;
 *          }
 *      }
 *
 *      __global__ void SomeKernel(int *d_data, int num_elements)
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
 *              for (int warp_offset = 0; warp_offset < num_elements; warp_offset += 32)
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

    /// Constants
    enum
    {
        POW_OF_TWO = ((LOGICAL_WARP_THREADS & (LOGICAL_WARP_THREADS - 1)) == 0),
    };

    /// WarpScan algorithmic variants (would use an enum, but it causes GCC crash as of CUDA5)
    static const int SHFL_SCAN = 0;          // Warp-synchronous SHFL-based scan
    static const int SMEM_SCAN = 1;          // Warp-synchronous smem-based scan


    /// Use SHFL_SCAN if (architecture is >= SM30) and (LOGICAL_WARP_THREADS is a power-of-two)
    static const int POLICY = ((CUB_PTX_ARCH >= 300) && POW_OF_TWO) ?
                                SHFL_SCAN :
                                SMEM_SCAN;


    #ifndef DOXYGEN_SHOULD_SKIP_THIS    // Do not document


    /******************************************************************************
     * Algorithmic variants
     ******************************************************************************/

    /**
     * SHFL_SCAN algorithmic variant
     */
    template <int _ALGORITHM, int DUMMY = 0>
    struct WarpScanInternal
    {
        /// Constants
        enum
        {
            /// The number of warp scan steps
            STEPS = Log2<LOGICAL_WARP_THREADS>::VALUE,

            // The 5-bit SHFL mask for logically splitting warps into sub-segments starts 8-bits up
            SHFL_C = ((-1 << STEPS) & 31) << 8,
        };

        /// Shared memory storage layout type
        typedef NullType TempStorage;


        // Thread fields
        int             warp_id;
        int             lane_id;


        /// Constructor
        __device__ __forceinline__ WarpScanInternal(
            TempStorage &temp_storage,
            int warp_id,
            int lane_id)
        :
            warp_id(warp_id),
            lane_id(lane_id)
        {}


        /// Broadcast
        __device__ __forceinline__ T Broadcast(
            T               input,              ///< [in] The value to broadcast
            int             src_lane)           ///< [in] Which warp lane is to do the broadcasting
        {
            typedef typename WordAlignment<T>::ShuffleWord ShuffleWord;

            const int       WORDS           = (sizeof(T) + sizeof(ShuffleWord) - 1) / sizeof(ShuffleWord);
            T               output;
            ShuffleWord     *output_alias   = reinterpret_cast<ShuffleWord *>(&output);
            ShuffleWord     *input_alias    = reinterpret_cast<ShuffleWord *>(&input);

            #pragma unroll
            for (int WORD = 0; WORD < WORDS; ++WORD)
            {
                unsigned int shuffle_word = input_alias[WORD];
                asm("shfl.idx.b32 %0, %1, %2, %3;"
                    : "=r"(shuffle_word) : "r"(shuffle_word), "r"(src_lane), "r"(LOGICAL_WARP_THREADS - 1));
                output_alias[WORD] = (ShuffleWord) shuffle_word;
            }

            return output;
        }


        /// Generic shuffle-up
        __device__ __forceinline__ T ShuffleUp(
            T               input,              ///< [in] The value to broadcast
            int             src_offset)         ///< [in] The up-offset of the peer to read from
        {
            typedef typename WordAlignment<T>::ShuffleWord ShuffleWord;

            const int       WORDS           = (sizeof(T) + sizeof(ShuffleWord) - 1) / sizeof(ShuffleWord);
            T               output;
            ShuffleWord     *output_alias   = reinterpret_cast<ShuffleWord *>(&output);
            ShuffleWord     *input_alias    = reinterpret_cast<ShuffleWord *>(&input);

            #pragma unroll
            for (int WORD = 0; WORD < WORDS; ++WORD)
            {
                unsigned int shuffle_word = input_alias[WORD];
                asm(
                    "  shfl.up.b32 %0, %1, %2, %3;"
                    : "=r"(shuffle_word) : "r"(shuffle_word), "r"(src_offset), "r"(SHFL_C));
                output_alias[WORD] = (ShuffleWord) shuffle_word;
            }

            return output;
        }


        //---------------------------------------------------------------------
        // Inclusive operations
        //---------------------------------------------------------------------

        /// Inclusive prefix sum with aggregate (single-SHFL)
        __device__ __forceinline__ void InclusiveSum(
            T               input,              ///< [in] Calling thread's input item.
            T               &output,            ///< [out] Calling thread's output item.  May be aliased with \p input.
            T               &warp_aggregate,    ///< [out] Warp-wide aggregate reduction of input items.
            Int2Type<true>  single_shfl)
        {
            unsigned int temp = reinterpret_cast<unsigned int &>(input);

            // Iterate scan steps
            #pragma unroll
            for (int STEP = 0; STEP < STEPS; STEP++)
            {
                // Use predicate set from SHFL to guard against invalid peers
                asm(
                    "{"
                    "  .reg .u32 r0;"
                    "  .reg .pred p;"
                    "  shfl.up.b32 r0|p, %1, %2, %3;"
                    "  @p add.u32 r0, r0, %4;"
                    "  mov.u32 %0, r0;"
                    "}"
                    : "=r"(temp) : "r"(temp), "r"(1 << STEP), "r"(SHFL_C), "r"(temp));
            }

            output = temp;

            // Grab aggregate from last warp lane
            warp_aggregate = Broadcast(output, LOGICAL_WARP_THREADS - 1);
        }


        /// Inclusive prefix sum with aggregate (multi-SHFL)
        __device__ __forceinline__ void InclusiveSum(
            T               input,              ///< [in] Calling thread's input item.
            T               &output,            ///< [out] Calling thread's output item.  May be aliased with \p input.
            T               &warp_aggregate,    ///< [out] Warp-wide aggregate reduction of input items.
            Int2Type<false> single_shfl)        ///< [in] Marker type indicating whether only one SHFL instruction is required
        {
            // Delegate to generic scan
            InclusiveScan(input, output, Sum<T>(), warp_aggregate);
        }


        /// Inclusive prefix sum with aggregate (specialized for float)
        __device__ __forceinline__ void InclusiveSum(
            float           input,              ///< [in] Calling thread's input item.
            float           &output,            ///< [out] Calling thread's output item.  May be aliased with \p input.
            float           &warp_aggregate)    ///< [out] Warp-wide aggregate reduction of input items.
        {
            output = input;

            // Iterate scan steps
            #pragma unroll
            for (int STEP = 0; STEP < STEPS; STEP++)
            {
                // Use predicate set from SHFL to guard against invalid peers
                asm(
                    "{"
                    "  .reg .f32 r0;"
                    "  .reg .pred p;"
                    "  shfl.up.b32 r0|p, %1, %2, %3;"
                    "  @p add.f32 r0, r0, %4;"
                    "  mov.f32 %0, r0;"
                    "}"
                    : "=f"(output) : "f"(output), "r"(1 << STEP), "r"(SHFL_C), "f"(output));
            }

            // Grab aggregate from last warp lane
            warp_aggregate = Broadcast(output, LOGICAL_WARP_THREADS - 1);
        }


        /// Inclusive prefix sum with aggregate (generic)
        template <typename _T>
        __device__ __forceinline__ void InclusiveSum(
            _T               input,             ///< [in] Calling thread's input item.
            _T               &output,           ///< [out] Calling thread's output item.  May be aliased with \p input.
            _T               &warp_aggregate)   ///< [out] Warp-wide aggregate reduction of input items.
        {
            // Whether sharing can be done with a single SHFL instruction (vs multiple SFHL instructions)
            Int2Type<(Traits<_T>::PRIMITIVE) && (sizeof(_T) <= sizeof(unsigned int))> single_shfl;

            InclusiveSum(input, output, warp_aggregate, single_shfl);
        }


        /// Inclusive prefix sum
        __device__ __forceinline__ void InclusiveSum(
            T               input,              ///< [in] Calling thread's input item.
            T               &output)            ///< [out] Calling thread's output item.  May be aliased with \p input.
        {
            T warp_aggregate;
            InclusiveSum(input, output, warp_aggregate);
        }


        /// Inclusive scan with aggregate
        template <typename ScanOp>
        __device__ __forceinline__ void InclusiveScan(
            T               input,              ///< [in] Calling thread's input item.
            T               &output,            ///< [out] Calling thread's output item.  May be aliased with \p input.
            ScanOp          scan_op,            ///< [in] Binary scan operator
            T               &warp_aggregate)    ///< [out] Warp-wide aggregate reduction of input items.
        {
            output = input;

            // Iterate scan steps
            #pragma unroll
            for (int STEP = 0; STEP < STEPS; STEP++)
            {
                // Grab addend from peer
                const int OFFSET = 1 << STEP;
                T temp = ShuffleUp(output, OFFSET);

                // Perform scan op if from a valid peer
                if (lane_id >= OFFSET)
                    output = scan_op(temp, output);
            }

            // Grab aggregate from last warp lane
            warp_aggregate = Broadcast(output, LOGICAL_WARP_THREADS - 1);
        }


        /// Inclusive scan
        template <typename ScanOp>
        __device__ __forceinline__ void InclusiveScan(
            T               input,              ///< [in] Calling thread's input item.
            T               &output,            ///< [out] Calling thread's output item.  May be aliased with \p input.
            ScanOp          scan_op)            ///< [in] Binary scan operator
        {
            T warp_aggregate;
            InclusiveScan(input, output, scan_op, warp_aggregate);
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
            ScanOp          scan_op,            ///< [in] Binary scan operator
            T               &warp_aggregate)    ///< [out] Warp-wide aggregate reduction of input items.
        {
            // Compute inclusive scan
            T inclusive;
            InclusiveScan(input, inclusive, scan_op, warp_aggregate);

            // Grab result from predecessor
            T exclusive = ShuffleUp(inclusive, 1);

            output = (lane_id == 0) ?
                identity :
                exclusive;
        }


        /// Exclusive scan
        template <typename ScanOp>
        __device__ __forceinline__ void ExclusiveScan(
            T               input,              ///< [in] Calling thread's input item.
            T               &output,            ///< [out] Calling thread's output item.  May be aliased with \p input.
            T               identity,           ///< [in] Identity value
            ScanOp          scan_op)            ///< [in] Binary scan operator
        {
            T warp_aggregate;
            ExclusiveScan(input, output, identity, scan_op, warp_aggregate);
        }


        /// Exclusive scan with aggregate, without identity
        template <typename ScanOp>
        __device__ __forceinline__ void ExclusiveScan(
            T               input,              ///< [in] Calling thread's input item.
            T               &output,            ///< [out] Calling thread's output item.  May be aliased with \p input.
            ScanOp          scan_op,            ///< [in] Binary scan operator
            T               &warp_aggregate)    ///< [out] Warp-wide aggregate reduction of input items.
        {
            // Compute inclusive scan
            T inclusive;
            InclusiveScan(input, inclusive, scan_op, warp_aggregate);

            // Grab result from predecessor
            output = ShuffleUp(inclusive, 1);
        }


        /// Exclusive scan without identity
        template <typename ScanOp>
        __device__ __forceinline__ void ExclusiveScan(
            T               input,              ///< [in] Calling thread's input item.
            T               &output,            ///< [out] Calling thread's output item.  May be aliased with \p input.
            ScanOp          scan_op)            ///< [in] Binary scan operator
        {
            T warp_aggregate;
            ExclusiveScan(input, output, scan_op, warp_aggregate);
        }
    };


    /**
     * SMEM_SCAN algorithmic variant
     */
    template <int DUMMY>
    struct WarpScanInternal<SMEM_SCAN, DUMMY>
    {
        /// Constants
        enum
        {
            /// The number of warp scan steps
            STEPS = Log2<LOGICAL_WARP_THREADS>::VALUE,

            /// The number of threads in half a warp
            HALF_WARP_THREADS = 1 << (STEPS - 1),

            /// The number of shared memory elements per warp
            WARP_SMEM_ELEMENTS =  LOGICAL_WARP_THREADS + HALF_WARP_THREADS,
        };


        /// Shared memory storage layout type (1.5 warps-worth of elements for each warp)
        typedef T TempStorage[LOGICAL_WARPS][WARP_SMEM_ELEMENTS];


        // Thread fields
        TempStorage     &temp_storage;
        unsigned int    warp_id;
        unsigned int    lane_id;


        /// Constructor
        __device__ __forceinline__ WarpScanInternal(
            TempStorage     &temp_storage,
            int             warp_id,
            int             lane_id)
        :
            temp_storage(temp_storage),
            warp_id(warp_id),
            lane_id(lane_id)
        {}


        /// Basic inclusive scan iteration (template unrolled, inductive-case specialization)
        template <
            bool HAS_IDENTITY,
            bool SHARE_FINAL,
            int STEP>
        struct Iteration
        {
            template <typename ScanOp>
            static __device__ __forceinline__ void ScanStep(
                TempStorage     &temp_storage,
                unsigned int    warp_id,
                unsigned int    lane_id,
                T               &partial,
                ScanOp          scan_op)
            {
                const int OFFSET = 1 << STEP;

                // Share partial into buffer
                ThreadStore<STORE_VOLATILE>(&temp_storage[warp_id][HALF_WARP_THREADS + lane_id], partial);

                // Update partial if addend is in range
                if (HAS_IDENTITY || (lane_id >= OFFSET))
                {
                    T addend = ThreadLoad<LOAD_VOLATILE>(&temp_storage[warp_id][HALF_WARP_THREADS + lane_id - OFFSET]);
                    partial = scan_op(addend, partial);
                }

                Iteration<HAS_IDENTITY, SHARE_FINAL, STEP + 1>::ScanStep(temp_storage, warp_id, lane_id, partial, scan_op);
            }
        };


        /// Basic inclusive scan iteration(template unrolled, base-case specialization)
        template <
            bool HAS_IDENTITY,
            bool SHARE_FINAL>
        struct Iteration<HAS_IDENTITY, SHARE_FINAL, STEPS>
        {
            template <typename ScanOp>
            static __device__ __forceinline__ void ScanStep(TempStorage &temp_storage, unsigned int warp_id, unsigned int lane_id, T &partial, ScanOp scan_op) {}
        };


        /// Broadcast
        __device__ __forceinline__ T Broadcast(
            T               input,              ///< [in] The value to broadcast
            unsigned int    src_lane)           ///< [in] Which warp lane is to do the broadcasting
        {
            if (lane_id == src_lane)
            {
                ThreadStore<STORE_VOLATILE>(temp_storage[warp_id], input);
            }

            return ThreadLoad<LOAD_VOLATILE>(temp_storage[warp_id]);
        }


        /// Basic inclusive scan
        template <
            bool        HAS_IDENTITY,
            bool        SHARE_FINAL,
            typename    ScanOp>
        __device__ __forceinline__ T BasicScan(
            T               partial,            ///< Calling thread's input partial reduction
            ScanOp          scan_op)            ///< Binary associative scan functor
        {
            // Iterate scan steps
            Iteration<HAS_IDENTITY, SHARE_FINAL, 0>::ScanStep(temp_storage, warp_id, lane_id, partial, scan_op);

            if (SHARE_FINAL)
            {
                // Share partial into buffer
                ThreadStore<STORE_VOLATILE>(&temp_storage[warp_id][HALF_WARP_THREADS + lane_id], partial);
            }

            return partial;
        }


        /// Inclusive prefix sum
        __device__ __forceinline__ void InclusiveSum(
            T               input,              ///< [in] Calling thread's input item.
            T               &output)            ///< [out] Calling thread's output item.  May be aliased with \p input.
        {
            // Initialize identity region
            ThreadStore<STORE_VOLATILE>(&temp_storage[warp_id][lane_id], T(0));

            // Compute inclusive warp scan (has identity, don't share final)
            output = BasicScan<true, false>(input, Sum<T>());
        }


        /// Inclusive prefix sum with aggregate
        __device__ __forceinline__ void InclusiveSum(
            T               input,              ///< [in] Calling thread's input item.
            T               &output,            ///< [out] Calling thread's output item.  May be aliased with \p input.
            T               &warp_aggregate)    ///< [out] Warp-wide aggregate reduction of input items.
        {
            // Initialize identity region
            ThreadStore<STORE_VOLATILE>(&temp_storage[warp_id][lane_id], T(0));

            // Compute inclusive warp scan (has identity, share final)
            output = BasicScan<true, true>(input, Sum<T>());

            // Retrieve aggregate in <em>warp-lane</em><sub>0</sub>
            warp_aggregate = ThreadLoad<LOAD_VOLATILE>(&temp_storage[warp_id][WARP_SMEM_ELEMENTS - 1]);
        }


        /// Inclusive scan
        template <typename ScanOp>
        __device__ __forceinline__ void InclusiveScan(
            T               input,              ///< [in] Calling thread's input item.
            T               &output,            ///< [out] Calling thread's output item.  May be aliased with \p input.
            ScanOp          scan_op)            ///< [in] Binary scan operator
        {
            // Compute inclusive warp scan (no identity, don't share final)
            output = BasicScan<false, false>(input, scan_op);
        }


        /// Inclusive scan with aggregate
        template <typename ScanOp>
        __device__ __forceinline__ void InclusiveScan(
            T               input,              ///< [in] Calling thread's input item.
            T               &output,            ///< [out] Calling thread's output item.  May be aliased with \p input.
            ScanOp          scan_op,            ///< [in] Binary scan operator
            T               &warp_aggregate)    ///< [out] Warp-wide aggregate reduction of input items.
        {
            // Compute inclusive warp scan (no identity, share final)
            output = BasicScan<false, true>(input, scan_op);

            // Retrieve aggregate
            warp_aggregate = ThreadLoad<LOAD_VOLATILE>(&temp_storage[warp_id][WARP_SMEM_ELEMENTS - 1]);
        }

        /// Exclusive scan
        template <typename ScanOp>
        __device__ __forceinline__ void ExclusiveScan(
            T               input,              ///< [in] Calling thread's input item.
            T               &output,            ///< [out] Calling thread's output item.  May be aliased with \p input.
            T               identity,           ///< [in] Identity value
            ScanOp          scan_op)            ///< [in] Binary scan operator
        {
            // Initialize identity region
            ThreadStore<STORE_VOLATILE>(&temp_storage[warp_id][lane_id], identity);

            // Compute inclusive warp scan (identity, share final)
            T inclusive = BasicScan<true, true>(input, scan_op);

            // Retrieve exclusive scan
            output = ThreadLoad<LOAD_VOLATILE>(&temp_storage[warp_id][HALF_WARP_THREADS + lane_id - 1]);
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
            // Exclusive warp scan (which does share final)
            ExclusiveScan(input, output, identity, scan_op);

            // Retrieve aggregate
            warp_aggregate = ThreadLoad<LOAD_VOLATILE>(&temp_storage[warp_id][WARP_SMEM_ELEMENTS - 1]);
        }


        /// Exclusive scan without identity
        template <typename ScanOp>
        __device__ __forceinline__ void ExclusiveScan(
            T               input,              ///< [in] Calling thread's input item.
            T               &output,            ///< [out] Calling thread's output item.  May be aliased with \p input.
            ScanOp          scan_op)            ///< [in] Binary scan operator
        {
            // Compute inclusive warp scan (no identity, share final)
            T inclusive = BasicScan<false, true>(input, scan_op);

            // Retrieve exclusive scan
            output = ThreadLoad<LOAD_VOLATILE>(&temp_storage[warp_id][HALF_WARP_THREADS + lane_id - 1]);
        }


        /// Exclusive scan with aggregate, without identity
        template <typename ScanOp>
        __device__ __forceinline__ void ExclusiveScan(
            T               input,              ///< [in] Calling thread's input item.
            T               &output,            ///< [out] Calling thread's output item.  May be aliased with \p input.
            ScanOp          scan_op,            ///< [in] Binary scan operator
            T               &warp_aggregate)    ///< [out] Warp-wide aggregate reduction of input items.
        {
            // Exclusive warp scan (which does share final)
            ExclusiveScan(input, output, scan_op);

            // Retrieve aggregate
            warp_aggregate = ThreadLoad<LOAD_VOLATILE>(&temp_storage[warp_id][WARP_SMEM_ELEMENTS - 1]);
        }

    };


    #endif // DOXYGEN_SHOULD_SKIP_THIS


    /******************************************************************************
     * Type definitions
     ******************************************************************************/

    /// Internal warpscan implementation to use
    typedef WarpScanInternal<POLICY> InternalWarpScan;

public:

    /// Shared memory storage layout type for WarpScan
    typedef typename InternalWarpScan::TempStorage TempStorage;

private:

    /******************************************************************************
     * Utility methods
     ******************************************************************************/

    /// Internal storage allocator
    __device__ __forceinline__ TempStorage& PrivateStorage()
    {
        __shared__ TempStorage private_storage;
        return private_storage;
    }


    /******************************************************************************
     * Thread fields
     ******************************************************************************/

    /// Shared storage reference
    TempStorage &temp_storage;

    /// Warp ID
    int warp_id;

    /// Lane ID
    int lane_id;

public:

    /******************************************************************//**
     * \name Collective construction
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

private:

    /// Computes an exclusive prefix sum in each logical warp.
    __device__ __forceinline__ void InclusiveSum(T input, T &output, Int2Type<true> is_primitive)
    {
        InternalWarpScan(temp_storage, warp_id, lane_id).InclusiveSum(input, output);
    }

    /// Computes an exclusive prefix sum in each logical warp.  Specialized for non-primitive types.
    __device__ __forceinline__ void InclusiveSum(T input, T &output, Int2Type<false> is_primitive)
    {
        // Delegate to regular scan for non-primitive types (because we won't be able to use subtraction)
        InclusiveScan(input, output, Sum<T>());
    }

    /// Computes an exclusive prefix sum in each logical warp.  Also provides every thread with the warp-wide \p warp_aggregate of all inputs.
    __device__ __forceinline__ void InclusiveSum(T input, T &output, T &warp_aggregate, Int2Type<true> is_primitive)
    {
        InternalWarpScan(temp_storage, warp_id, lane_id).InclusiveSum(input, output, warp_aggregate);
    }

    /// Computes an exclusive prefix sum in each logical warp.  Also provides every thread with the warp-wide \p warp_aggregate of all inputs.  Specialized for non-primitive types.
    __device__ __forceinline__ void InclusiveSum(T input, T &output, T &warp_aggregate, Int2Type<false> is_primitive)
    {
        // Delegate to regular scan for non-primitive types (because we won't be able to use subtraction)
        InclusiveScan(input, output, Sum<T>(), warp_aggregate);
    }

    /// Computes an exclusive prefix sum in each logical warp.  Instead of using 0 as the warp-wide prefix, the call-back functor \p warp_prefix_op is invoked to provide the "seed" value that logically prefixes the warp's scan inputs.  Also provides every thread with the warp-wide \p warp_aggregate of all inputs.
    template <typename WarpPrefixOp>
    __device__ __forceinline__ void InclusiveSum(T input, T &output, T &warp_aggregate, WarpPrefixOp &warp_prefix_op, Int2Type<true> is_primitive)
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

    /// Computes an exclusive prefix sum in each logical warp.  Instead of using 0 as the warp-wide prefix, the call-back functor \p warp_prefix_op is invoked to provide the "seed" value that logically prefixes the warp's scan inputs.  Also provides every thread with the warp-wide \p warp_aggregate of all inputs.  Specialized for non-primitive types.
    template <typename WarpPrefixOp>
    __device__ __forceinline__ void InclusiveSum(T input, T &output, T &warp_aggregate, WarpPrefixOp &warp_prefix_op, Int2Type<false> is_primitive)
    {
        // Delegate to regular scan for non-primitive types (because we won't be able to use subtraction)
        InclusiveScan(input, output, Sum<T>(), warp_aggregate, warp_prefix_op);
    }

public:

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
        InclusiveSum(input, output, Int2Type<Traits<T>::PRIMITIVE>());
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
        InclusiveSum(input, output, warp_aggregate, Int2Type<Traits<T>::PRIMITIVE>());
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
        InclusiveSum(input, output, warp_aggregate, warp_prefix_op, Int2Type<Traits<T>::PRIMITIVE>());
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
        ExclusiveScan(input, output, identity, Sum<T>());
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
        ExclusiveScan(input, output, identity, Sum<T>(), warp_aggregate);
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
        ExclusiveScan(input, output, identity, Sum<T>(), warp_aggregate, warp_prefix_op);
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
