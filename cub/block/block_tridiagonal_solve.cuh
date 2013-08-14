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
 * The cub::BlockTridiagonalSolve class provides [<em>collective</em>](index.html#sec0) methods for solving a tridiagonal system of linear equations partitioned across a CUDA thread block.
 */

#pragma once

#include "block_scan.cuh"
#include "../util_arch.cuh"
#include "../util_type.cuh"
#include "../util_namespace.cuh"

/// Optional outer namespace(s)
CUB_NS_PREFIX

/// CUB namespace
namespace cub {



/******************************************************************************
 * Block tridiagonal solve
 ******************************************************************************/

/**
 * \brief The BlockTridiagonalSolve class provides [<em>collective</em>](index.html#sec0) methods for solving a tridiagonal system of linear equations partitioned across a CUDA thread block. ![](block_scan_logo.png)
 * \ingroup BlockModule
 *
 * \par Overview
 *
 * \tparam T                Data type of numeric coefficients (e.g., \p float or \p double)
 * \tparam BLOCK_THREADS    The thread block size in threads
 * \tparam ALGORITHM        <b>[optional]</b> cub::BlockScanAlgorithm enumerator specifying the underlying algorithm to use (default: cub::BLOCK_SCAN_RAKING)
 *
 * \par A Simple Example
 * \blockcollective{BlockTridiagonalSolve}
 * \par
 * The code snippet below illustrates the solution of a tridiagonal system of 512 linear equations across 128 threads
 * where each thread owns 4 consecutive items in \p x, \p d, and in each of the three diagonals in \p A.
 * \par
 * \code
 * #include <cub/cub.cuh>
 *
 * __global__ void ExampleKernel(...)
 * {
 *     // Specialize BlockTridiagonalSolve for 128 threads on type int
 *     typedef cub::BlockTridiagonalSolve<int, 128> BlockTridiagonalSolve;
 *
 *     // Allocate shared memory for BlockTridiagonalSolve
 *     __shared__ typename BlockTridiagonalSolve::TempStorage temp_storage;
 *
 *     ...
 *
 *     // Collectively solve for x
 *
 * \endcode
 * \par
 * Suppose the sets of \p a, \p b, \p c, and \p d across the block of threads is
 * <tt>{ [1,1,1,1], [1,1,1,1], ..., [1,1,1,1] }</tt>.
 *
 *
 * \par Performance Considerations
 * - Uses special instructions when applicable (e.g., warp \p SHFL)
 * - Uses synchronization-free communication between warp lanes when applicable
 * - Uses only one or two block-wide synchronization barriers (depending on
 *   algorithm selection)
 * - Zero bank conflicts for most types
 * - Computation is slightly more efficient (i.e., having lower instruction overhead) for:
 *   - Prefix sum variants (<b><em>vs.</em></b> generic scan)
 *   - Exclusive variants (<b><em>vs.</em></b> inclusive)
 *   - \p BLOCK_THREADS is a multiple of the architecture's warp size
 * - See cub::BlockTridiagonalSolveAlgorithm for performance details regarding algorithmic alternatives
 *
 */
template <
    typename            T,
    int                 BLOCK_THREADS,
    BlockScanAlgorithm  SCAN_ALGORITHM = BLOCK_SCAN_RAKING>
class BlockTridiagonalSolve
{
private:

    /******************************************************************************
     * Constants and type definitions
     ******************************************************************************/

    /// Scan data type
    struct ScanTuple
    {
        // First two rows of the 3x3 matrix used by scan
        T g[2][3];
    };

    /// Scan functor
    struct ScanOp
    {
        __device__ __forceinline__ ScanTuple operator()(const ScanTuple &first, const ScanTuple &second)
        {
            ScanTuple retval;

            // first row
            retval[0][0] = (first.g[0][0] * second.g[0][0]) + (first.g[0][1] * second.g[1][0]);
            retval[0][1] = (first.g[0][0] * second.g[0][1]) + (first.g[0][1] * second.g[1][1]);
            retval[0][2] = (first.g[0][0] * second.g[0][2]) + (first.g[0][1] * second.g[1][2]) + first.g[0][2];

            // second row
            retval[1][0] = (first.g[1][0] * second.g[0][0]) + (first.g[1][1] * second.g[1][0]);
            retval[1][1] = (first.g[1][0] * second.g[0][1]) + (first.g[1][1] * second.g[1][1]);
            retval[1][2] = (first.g[1][0] * second.g[0][2]) + (first.g[1][1] * second.g[1][2]) + first.g[1][2];

        }
    };

    /// Prefix scan implementation to use
    typedef typename BlockScan<ScanTuple, BLOCK_THREADS, SCAN_ALGORITHM> BlockScan;

    /// Shared memory storage layout type for BlockTridiagonalSolve
    typedef typename BlockScan::TempStorage _TempStorage;


    /******************************************************************************
     * Thread fields
     ******************************************************************************/

    /// Shared storage reference
    _TempStorage &temp_storage;

    /// Linear thread-id
    int linear_tid;


    /******************************************************************************
     * Utility methods
     ******************************************************************************/

    /// Internal storage allocator
    __device__ __forceinline__ _TempStorage& PrivateStorage()
    {
        __shared__ _TempStorage private_storage;
        return private_storage;
    }


public:

    /// \smemstorage{BlockTridiagonalSolve}
    struct TempStorage : Uninitialized<_TempStorage> {};


    /******************************************************************//**
     * \name Collective constructors
     *********************************************************************/
    //@{

    /**
     * \brief Collective constructor for 1D thread blocks using a private static allocation of shared memory as temporary storage.  Threads are identified using <tt>threadIdx.x</tt>.
     */
    __device__ __forceinline__ BlockTridiagonalSolve()
    :
        temp_storage(PrivateStorage()),
        linear_tid(threadIdx.x)
    {}


    /**
     * \brief Collective constructor for 1D thread blocks using the specified memory allocation as temporary storage.  Threads are identified using <tt>threadIdx.x</tt>.
     */
    __device__ __forceinline__ BlockTridiagonalSolve(
        TempStorage &temp_storage)             ///< [in] Reference to memory allocation having layout type TempStorage
    :
        temp_storage(temp_storage.Alias()),
        linear_tid(threadIdx.x)
    {}


    /**
     * \brief Collective constructor using a private static allocation of shared memory as temporary storage.  Each thread is identified using the supplied linear thread identifier
     */
    __device__ __forceinline__ BlockTridiagonalSolve(
        int linear_tid)                        ///< [in] A suitable 1D thread-identifier for the calling thread (e.g., <tt>(threadIdx.y * blockDim.x) + linear_tid</tt> for 2D thread blocks)
    :
        temp_storage(PrivateStorage()),
        linear_tid(linear_tid)
    {}


    /**
     * \brief Collective constructor using the specified memory allocation as temporary storage.  Each thread is identified using the supplied linear thread identifier.
     */
    __device__ __forceinline__ BlockTridiagonalSolve(
        TempStorage &temp_storage,             ///< [in] Reference to memory allocation having layout type TempStorage
        int linear_tid)                        ///< [in] <b>[optional]</b> A suitable 1D thread-identifier for the calling thread (e.g., <tt>(threadIdx.y * blockDim.x) + linear_tid</tt> for 2D thread blocks)
    :
        temp_storage(temp_storage.Alias()),
        linear_tid(linear_tid)
    {}



    //@}  end member group
    /******************************************************************//**
     * \name Solve operations
     *********************************************************************/
    //@{


    /**
     * \brief Solves a tridiagonal linear system <b>A</b> <b>x</b> = <b>d</b> of size \p N where \p N is \p BLOCK_THREADS * \p ITEMS_PER_THREAD.
     *
     * \blocked
     *
     * \smemreuse
     *
     * \tparam ITEMS_PER_THREAD     <b>[inferred]</b> The number of consecutive unknowns partitioned onto each thread.
     *
     */
    template <int ITEMS_PER_THREAD>
    __device__ __forceinline__ void Solve(
        T   (&a)[ITEMS_PER_THREAD],     ///< [in] Calling thread's segment of coefficients within the first diagonal.
        T   (&b)[ITEMS_PER_THREAD],     ///< [in] Calling thread's segment of coefficients within the second diagonal.
        T   (&c)[ITEMS_PER_THREAD],     ///< [in] Calling thread's segment of coefficients within the thrid diagonal.
        T   (&d)[ITEMS_PER_THREAD],     ///< [in] Calling thread's segment of the product vector
        T   (&x)[ITEMS_PER_THREAD])     ///< [out] Calling thread's segment of solution vector
    {
    }

    //@}  end member group


};

}               // CUB namespace
CUB_NS_POSTFIX  // Optional outer namespace(s)

