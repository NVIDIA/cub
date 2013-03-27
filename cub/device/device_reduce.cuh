
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
 * cub::DeviceReduce provides variants of parallel reduction across a CUDA device
 */

#pragma once

#include <stdio.h>
#include <iterator>

#include "block/block_reduce_tiles.cuh"
#include "../util_allocator.cuh"

/// Optional outer namespace(s)
CUB_NS_PREFIX

/// CUB namespace
namespace cub {



/******************************************************************************
 * Kernel entry points
 *****************************************************************************/

/** \cond INTERNAL */

/**
 * Reduction kernel entry point.
 */
template <
    typename BlockReduceTilesPolicy,
    typename InputIterator,
    typename OutputIterator,
    typename SizeT,
    typename ReductionOp>
__global__ void ReduceKernel(
    InputIterator           d_in,
    OutputIterator          d_out,
    SizeT                   num_items,
    GridEvenShare<SizeT>    even_share,
    GridQueue<SizeT>        queue,
    ReductionOp             reduction_op)
{
    // Data type of input iterator
    typedef typename std::iterator_traits<InputIterator>::value_type T;

    // Parameterize BlockReduceTiles for the parallel execution context
    typedef BlockReduceTiles <BlockReduceTilesPolicy, InputIterator, SizeT> BlockReduceTilesT;

    // Declare shared memory for BlockReduceTiles
    __shared__ typename BlockReduceTilesT::SmemStorage smem_storage;

    // Reduce tiles
    T block_aggregate = BlockReduceTilesT::ProcessTiles(
        smem_storage,
        d_in,
        num_items,
        even_share,
        queue,
        reduction_op);

    // Output result
    if (threadIdx.x == 0)
    {
        d_out[blockIdx.x] = block_aggregate;
    }
}


/**
 * Single-block reduction kernel entry point.
 */
template <
    typename BlockReduceTilesPolicy,
    typename InputIterator,
    typename OutputIterator,
    typename SizeT,
    typename ReductionOp>
__global__ void SingleReduceKernel(
    InputIterator           d_in,
    OutputIterator          d_out,
    SizeT                   num_items,
    ReductionOp             reduction_op)
{
    // Data type of input iterator
    typedef typename std::iterator_traits<InputIterator>::value_type T;

    // Parameterize BlockReduceTiles for the parallel execution context
    typedef BlockReduceTiles <BlockReduceTilesPolicy, InputIterator, SizeT> BlockReduceTilesT;

    // Declare shared memory for BlockReduceTiles
    __shared__ typename BlockReduceTilesT::SmemStorage smem_storage;

    // Reduce tiles
    T block_aggregate = BlockReduceTilesT::ProcessTilesEvenShare(
        smem_storage,
        d_in,
        0,
        num_items,
        reduction_op);

    // Output result
    if (threadIdx.x == 0)
    {
        d_out[blockIdx.x] = block_aggregate;
    }
}

/** \endcond */     // INTERNAL


/******************************************************************************
 * DeviceFoo
 *****************************************************************************/

/**
 * \addtogroup DeviceModule
 * @{
 */

/**
 * Provides Foo operations on device-global data sets.
 */
struct DeviceFoo
{
    struct DispatchPolicy
    {
        int reduce_block_threads;
        int reduce_items_per_thread;

        int single_reduce_block_threads;
        int single_reduce_items_per_thread;

        template <typename BlockReduceTilesPolicy, typename ReduceSingleKernelPolicy>
        __host__ __device__ __forceinline__
        void Init()
        {
            reduce_block_threads       = BlockReduceTilesPolicy::BLOCK_THREADS;
            reduce_items_per_thread    = BlockReduceTilesPolicy::ITEMS_PER_THREAD;
            single_reduce_block_threads     = ReduceSingleKernelPolicy::BLOCK_THREADS;
            single_reduce_items_per_thread  = ReduceSingleKernelPolicy::ITEMS_PER_THREAD;
        }
    };


    // Default tuning policy types
    template <
        typename T,
        typename SizeT>
    struct DefaultPolicy
    {

        typedef BlockReduceTilesPolicy<     64,     1>      BlockReduceTilesPolicy300;
        typedef ReduceSingleKernelPolicy<   64,     1>      ReduceSingleKernelPolicy300;

        typedef BlockReduceTilesPolicy<     128,    1>      BlockReduceTilesPolicy200;
        typedef ReduceSingleKernelPolicy<   128,    1>      ReduceSingleKernelPolicy200;

        typedef BlockReduceTilesPolicy<     256,    1>      BlockReduceTilesPolicy100;
        typedef ReduceSingleKernelPolicy<   256,    1>      ReduceSingleKernelPolicy100;

    #if CUB_PTX_ARCH >= 300
        struct PtxBlockReduceTilesPolicy :      BlockReduceTilesPolicy300 {};
        struct PtxReduceSingleKernelPolicy :    ReduceSingleKernelPolicy300 {};
    #elif CUB_PTX_ARCH >= 200
        struct PtxBlockReduceTilesPolicy :      BlockReduceTilesPolicy200 {};
        struct PtxReduceSingleKernelPolicy :    ReduceSingleKernelPolicy200 {};
    #else
        struct PtxBlockReduceTilesPolicy :      BlockReduceTilesPolicy100 {};
        struct PtxReduceSingleKernelPolicy :    ReduceSingleKernelPolicy100 {};
    #endif

        /**
         * Initialize dispatch policy
         */
        static void InitDispatchPolicy(int device_arch, DispatchPolicy &dispatch_policy)
        {
            if (device_arch >= 300)
                dispatch_policy.Init<BlockReduceTilesPolicy300, ReduceSingleKernelPolicy300>();
            else if (device_arch >= 200)
                dispatch_policy.Init<BlockReduceTilesPolicy200, ReduceSingleKernelPolicy200>();
            else
                dispatch_policy.Init<BlockReduceTilesPolicy100, ReduceSingleKernelPolicy100>();
        }
    };


    // Internal Foo dispatch
    template <
        typename ReduceKernelPtr,
        typename ReduceSingleKernelPtr,
        typename T,
        typename SizeT>
    __host__ __device__ __forceinline__
    static cudaError_t Foo(
        ReduceKernelPtr     reduce_kernel_ptr,
        ReduceSingleKernelPtr   single_reduce_kernel_ptr,
        DispatchPolicy          &dispatch_policy,
        T                       *d_in,
        T                       *d_out,
        SizeT                   num_elements)
    {
    #if !CUB_CNP_ENABLED

        // Kernel launch not supported from this device
        return cudaErrorInvalidConfiguration;

    #else

        cudaError error = cudaSuccess;
        do
        {
            // Get GPU ordinal
            int device_ordinal;
            if ((error = CubDebug(cudaGetDevice(&device_ordinal)))) break;

            // Get SM count
            int sm_count;
            if (CubDebug(error = cudaDeviceGetAttribute (&sm_count, cudaDevAttrMultiProcessorCount, device_ordinal))) break;

            // Rough estimate of maximum SM occupancies based upon PTX assembly
            int reduce_sm_occupancy        = ArchProps<CUB_PTX_ARCH>::MAX_SM_THREADBLOCKS;
            int single_reduce_sm_occupancy      = ArchProps<CUB_PTX_ARCH>::MAX_SM_THREADBLOCKS;
            int oversubscription            = ArchProps<CUB_PTX_ARCH>::OVERSUBSCRIPTION;

        #if (CUB_PTX_ARCH == 0)

            // We're on the host, so come up with a more accurate estimate of SM occupancies from actual device properties
            Device device_props;
            if (CubDebug(error = device_props.Init(device_ordinal))) break;
            oversubscription = device_props.oversubscription;

            if (CubDebug(error = device_props.MaxSmOccupancy(
                reduce_sm_occupancy,
                reduce_kernel_ptr,
                dispatch_policy.reduce_block_threads))) break;

            if (CubDebug(error = device_props.MaxSmOccupancy(
                single_reduce_sm_occupancy,
                single_reduce_kernel_ptr,
                dispatch_policy.single_reduce_block_threads))) break;
        #endif

            // Construct work distributions
            GridEvenShare<SizeT> reduce_distrib(
                num_elements,
                reduce_sm_occupancy * sm_count * oversubscription,
                dispatch_policy.reduce_block_threads * dispatch_policy.reduce_items_per_thread);

            GridEvenShare<SizeT> single_reduce_distrib(
                num_elements,
                single_reduce_sm_occupancy * sm_count * oversubscription,
                dispatch_policy.single_reduce_block_threads * dispatch_policy.single_reduce_items_per_thread);

            printf("Invoking Reduce<<<%d, %d>>>()\n",
                reduce_distrib.grid_size,
                dispatch_policy.reduce_block_threads);

            // Invoke Reduce
            reduce_kernel_ptr<<<reduce_distrib.grid_size, dispatch_policy.reduce_block_threads>>>(
                d_in, d_out, num_elements);

            printf("Invoking ReduceSingle<<<%d, %d>>>()\n",
                single_reduce_distrib.grid_size,
                dispatch_policy.single_reduce_block_threads);

            // Invoke ReduceSingle
            single_reduce_kernel_ptr<<<single_reduce_distrib.grid_size, dispatch_policy.single_reduce_block_threads>>>(
                d_in, d_out, num_elements);
        }
        while (0);
        return error;

    #endif
    }


    //---------------------------------------------------------------------
    // Public interface
    //---------------------------------------------------------------------


    /**
     * Invoke Foo operation with custom policies
     */
    template <
        typename BlockReduceTilesPolicy,
        typename ReduceSingleKernelPolicy,
        typename T,
        typename SizeT>
    __host__ __device__ __forceinline__
    static cudaError_t Foo(
        T       *d_in,
        T       *d_out,
        SizeT   num_elements)
    {
        DispatchPolicy dispatch_policy;
        dispatch_policy.Init<BlockReduceTilesPolicy, ReduceSingleKernelPolicy>();

        return Foo(
            ReduceKernel<BlockReduceTilesPolicy, T, SizeT>,
            ReduceSingleKernel<ReduceSingleKernelPolicy, T, SizeT>,
            dispatch_policy,
            d_in,
            d_out,
            num_elements);
    }


    /**
     * Invoke Foo operation with default policies
     */
    template <
        typename T,
        typename SizeT>
    __host__ __device__ __forceinline__
    static cudaError_t Foo(
        T       *d_in,
        T       *d_out,
        SizeT   num_elements)
    {
        cudaError error = cudaSuccess;
        do
        {
            typedef typename DefaultPolicy<T, SizeT>::PtxBlockReduceTilesPolicy PtxBlockReduceTilesPolicy;
            typedef typename DefaultPolicy<T, SizeT>::PtxReduceSingleKernelPolicy PtxReduceSingleKernelPolicy;

            // Initialize dispatch policy
            DispatchPolicy dispatch_policy;

        #if CUB_PTX_ARCH > 0

            // We're on the device, so initialize the tuned dispatch policy based upon PTX arch
            dispatch_policy.Init<PtxBlockReduceTilesPolicy, PtxReduceSingleKernelPolicy>();

        #else

            // We're on the host, so initialize the tuned dispatch policy based upon the device's SM arch
            int device_ordinal;
            if (CubDebug(error = cudaGetDevice(&device_ordinal))) break;

            int major, minor;
            if (CubDebug(error = cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device_ordinal))) break;
            if (CubDebug(error = cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device_ordinal))) break;
            int device_arch = major * 100 + minor * 10;

            DefaultPolicy<T, SizeT>::InitDispatchPolicy(device_arch, dispatch_policy);

        #endif

            // Dispatch
            if (CubDebug(error = Foo(
                ReduceKernel<PtxBlockReduceTilesPolicy, T, SizeT>,
                ReduceSingleKernel<PtxReduceSingleKernelPolicy, T, SizeT>,
                dispatch_policy,
                d_in,
                d_out,
                num_elements))) break;
        }
        while (0);

        return error;
    }

};


/** @} */       // DeviceModule

}               // CUB namespace
CUB_NS_POSTFIX  // Optional outer namespace(s)


