
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
#include "../grid/grid_even_share.cuh"
#include "../grid/grid_queue.cuh"

/// Optional outer namespace(s)
CUB_NS_PREFIX

/// CUB namespace
namespace cub {



/******************************************************************************
 * Kernel entry points
 *****************************************************************************/

#ifndef DOXYGEN_SHOULD_SKIP_THIS    // Do not document

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

#endif // DOXYGEN_SHOULD_SKIP_THIS


/******************************************************************************
 * DeviceReduce
 *****************************************************************************/

/**
 * \addtogroup DeviceModule
 * @{
 */

/**
 * \brief DeviceReduce provides variants of parallel reduction across a CUDA device.
 */
struct DeviceReduce
{
    #ifndef DOXYGEN_SHOULD_SKIP_THIS    // Do not document

    /// Generic structure for encapsulating dispatch properties
    struct DispatchPolicy
    {
        // Upsweep kernel policy details
        int                     upsweep_block_threads;
        int                     upsweep_items_per_thread;
        GridMappingStrategy     upsweep_mapping;

        // Single kernel policy details
        int                     single_block_threads;
        int                     single_items_per_thread;

        template <
            typename UpsweepTilesPolicy,
            typename SingleTilesPolicy>
        __host__ __device__ __forceinline__
        void Init()
        {
            upsweep_block_threads           = UpsweepTilesPolicy::BLOCK_THREADS;
            upsweep_items_per_thread        = UpsweepTilesPolicy::ITEMS_PER_THREAD;
            upsweep_mapping                 = UpsweepTilesPolicy::GRID_MAPPING;

            single_block_threads            = SingleTilesPolicy::BLOCK_THREADS;
            single_items_per_thread         = SingleTilesPolicy::ITEMS_PER_THREAD;
        }
    };


    /// Provides tuned policy default types for the targeted problem setting (and the ability to initialize a corresponding DispatchPolicy)
    template <
        typename InputIterator,
        typename SizeT>
    struct DefaultPolicy
    {
        // Data type of input iterator
        typedef typename std::iterator_traits<InputIterator>::value_type T;

        // SM30 tune
        typedef BlockReduceTilesPolicy<128,     8, GRID_MAPPING_EVEN_SHARE, BLOCK_LOAD_STRIPED, PTX_LOAD_NONE>      UpsweepTilesPolicy300;
        typedef BlockReduceTilesPolicy<128,     0, GRID_MAPPING_EVEN_SHARE, BLOCK_LOAD_STRIPED, PTX_LOAD_NONE>      SingleTilesPolicy300;

        // SM20 tune
        typedef BlockReduceTilesPolicy<128,     8, GRID_MAPPING_EVEN_SHARE, BLOCK_LOAD_STRIPED, PTX_LOAD_NONE>      UpsweepTilesPolicy200;
        typedef BlockReduceTilesPolicy<128,     0, GRID_MAPPING_EVEN_SHARE, BLOCK_LOAD_STRIPED, PTX_LOAD_NONE>      SingleTilesPolicy200;

        // SM10 tune
        typedef BlockReduceTilesPolicy<128,     8, GRID_MAPPING_EVEN_SHARE, BLOCK_LOAD_STRIPED, PTX_LOAD_NONE>      UpsweepTilesPolicy100;
        typedef BlockReduceTilesPolicy<128,     0, GRID_MAPPING_EVEN_SHARE, BLOCK_LOAD_STRIPED, PTX_LOAD_NONE>      SingleTilesPolicy100;

        // PTX tune
    #if CUB_PTX_ARCH >= 300
        struct PtxUpsweepTilesPolicy :      UpsweepTilesPolicy300 {};
        struct PtxSingleTilesPolicy :       SingleTilesPolicy300 {};
    #elif CUB_PTX_ARCH >= 200
        struct PtxUpsweepTilesPolicy :      UpsweepTilesPolicy200 {};
        struct PtxSingleTilesPolicy :       SingleTilesPolicy200 {};
    #else
        struct PtxUpsweepTilesPolicy :      UpsweepTilesPolicy100 {};
        struct PtxSingleTilesPolicy :       SingleTilesPolicy100 {};
    #endif

        /**
         * Initialize dispatch policy
         */
        static void InitDispatchPolicy(int device_arch, DispatchPolicy &dispatch_policy)
        {
            if (device_arch >= 300)
            {
                dispatch_policy.Init<UpsweepTilesPolicy300, SingleTilesPolicy300>();
            }
            else if (device_arch >= 200)
            {
                dispatch_policy.Init<UpsweepTilesPolicy200, SingleTilesPolicy200>();
            }
            else
            {
                dispatch_policy.Init<UpsweepTilesPolicy100, SingleTilesPolicy100>();
            }
        }
    };


    /**
     * Dispatch a single-block kernel to perform the device reduction
     */
    template <
        typename ReduceSingleKernelPtr,
        typename InputIterator,
        typename OutputIterator,
        typename SizeT,
        typename ReductionOp>
    __host__ __device__ __forceinline__
    static cudaError_t DispatchSingle(
        ReduceSingleKernelPtr   single_reduce_kernel_ptr,
        DispatchPolicy          &dispatch_policy,
        InputIterator           d_in,
        OutputIterator          d_out,
        SizeT                   num_items,
        ReductionOp             reduction_op,
        cudaStream_t            stream = 0,
        bool                    stream_synchronous = false)
    {
#if !CUB_CNP_ENABLED

    // Kernel launch not supported from this device
    return cudaErrorInvalidConfiguration;

#else

    cudaError error = cudaSuccess;
    do
    {
        CubLog(printf("Invoking ReduceSingle<<<%d, %d>>>()\n", 1, dispatch_policy.single_block_threads));

        // Invoke ReduceSingle
        single_reduce_kernel_ptr<<<1, dispatch_policy.single_block_threads>>>(
            d_in,
            d_out,
            num_items,
            reduction_op);

        if (stream_synchronous && CubDebug(error = cudaStreamSynchronize(stream))) break;
    }
    while (0);
    return error;

    return error;

#endif
    }


    /**
     * Dispatch two kernels (an upsweep multi-block kernel followed by a single-block kernel) to perform the device reduction
     */
    template <
        typename ReduceKernelPtr,
        typename ReduceSingleKernelPtr,
        typename InputIterator,
        typename OutputIterator,
        typename SizeT,
        typename ReductionOp>
    __host__ __device__ __forceinline__
    static cudaError_t DispatchIterative(
        ReduceKernelPtr         reduce_kernel_ptr,
        ReduceSingleKernelPtr   single_reduce_kernel_ptr,
        DispatchPolicy          &dispatch_policy,
        InputIterator           d_in,
        OutputIterator          d_out,
        SizeT                   num_items,
        ReductionOp             reduction_op,
        cudaStream_t            stream = 0,
        bool                    stream_synchronous = false)
    {
    #if !CUB_CNP_ENABLED

        // Kernel launch not supported from this device
        return cudaErrorInvalidConfiguration;

    #else

        // Data type of input iterator
        typedef typename std::iterator_traits<InputIterator>::value_type T;

        cudaError error = cudaSuccess;
        do
        {
            // Get GPU ordinal
            int device_ordinal;
            if (CubDebug(error = cudaGetDevice(&device_ordinal))) break;

            // Get SM count
            int sm_count;
            if (CubDebug(error = cudaDeviceGetAttribute (&sm_count, cudaDevAttrMultiProcessorCount, device_ordinal))) break;

            // Rough estimate of SM occupancies based upon the maximum SM occupancy of the targeted PTX architecture
            int oversubscription            = ArchProps<CUB_PTX_ARCH>::OVERSUBSCRIPTION;
            int reduce_sm_occupancy         = ArchProps<CUB_PTX_ARCH>::MAX_SM_THREADBLOCKS;
            int single_reduce_sm_occupancy  = 1;

        #if (CUB_PTX_ARCH == 0)

            // We're on the host, so come up with a more accurate estimate of SM occupancies from actual device properties
            Device device_props;
            if (CubDebug(error = device_props.Init(device_ordinal))) break;
            oversubscription = device_props.oversubscription;

            if (CubDebug(error = device_props.MaxSmOccupancy(
                reduce_sm_occupancy,
                reduce_kernel_ptr,
                dispatch_policy.upsweep_block_threads))) break;

        #endif

            GridEvenShare<SizeT>    upsweep_even_share;
            GridQueue<SizeT>        upsweep_queue;

            int reduce_occupancy = reduce_sm_occupancy * sm_count;
            int upsweep_tile_size = dispatch_policy.upsweep_block_threads * dispatch_policy.upsweep_items_per_thread;
            int upsweep_grid_size;

            switch (dispatch_policy.upsweep_mapping)
            {
            case GRID_MAPPING_EVEN_SHARE:

                // Even share
                upsweep_even_share.HostInit(
                    num_items,
                    reduce_occupancy * oversubscription,
                    upsweep_tile_size);

                upsweep_grid_size = upsweep_even_share.grid_size;
                break;

            case GRID_MAPPING_DYNAMIC:

                // Dynamic queue
                upsweep_queue.Allocate();
                int num_tiles = (num_items + upsweep_tile_size - 1) / upsweep_tile_size;
                if (num_tiles < reduce_occupancy)
                {
                    // Every thread block gets one input tile each and nothing is queued
                    upsweep_grid_size = num_tiles;
                    upsweep_queue.PrepareDrain(0);
                }
                else
                {
                    // Thread blocks get one input tile each and the rest are queued.
                    upsweep_grid_size = reduce_occupancy;
                    upsweep_queue.PrepareDrain(num_items - (reduce_occupancy * upsweep_tile_size));
                }
                break;
            };

            // Allocate temporary storage for thread block partial reductions
            T* d_block_partials;
            if (CubDebug(error = DeviceAllocate((void**) &d_block_partials, upsweep_grid_size * sizeof(T)))) break;

            CubLog(printf("Invoking Reduce<<<%d, %d>>>()\n", reduce_distrib.grid_size, dispatch_policy.upsweep_block_threads));

            // Invoke Reduce
            reduce_kernel_ptr<<<upsweep_grid_size, dispatch_policy.upsweep_block_threads>>>(
                d_in,
                d_block_partials,
                num_items,
                even_share,
                queue,
                reduction_op);

            if (stream_synchronous && CubDebug(error = cudaStreamSynchronize(stream))) break;

            CubLog(printf("Invoking ReduceSingle<<<%d, %d>>>()\n", 1, dispatch_policy.single_block_threads));

            // Invoke ReduceSingle
            single_reduce_kernel_ptr<<<1, dispatch_policy.single_block_threads>>>(
                d_block_partials,
                d_out,
                upsweep_grid_size,
                reduction_op);

            if (stream_synchronous && CubDebug(error = cudaStreamSynchronize(stream))) break;
        }
        while (0);
        return error;

    #endif
    }


    /**
     * Internal device reduction dispatch
     */
    template <
        typename ReduceKernelPtr,
        typename ReduceSingleKernelPtr,
        typename InputIterator,
        typename OutputIterator,
        typename SizeT,
        typename ReductionOp>
    __host__ __device__ __forceinline__
    static cudaError_t Dispatch(
        ReduceKernelPtr         reduce_kernel_ptr,
        ReduceSingleKernelPtr   single_reduce_kernel_ptr,
        DispatchPolicy          &dispatch_policy,
        InputIterator           d_in,
        OutputIterator          d_out,
        SizeT                   num_items,
        ReductionOp             reduction_op,
        cudaStream_t            stream = 0,
        bool                    stream_synchronous = false)
    {
        if (num_items > (dispatch_policy.single_block_threads * dispatch_policy.single_items_per_thread))
        {
            // Dispatch multiple kernels
            return DispatchIterative(reduce_kernel_ptr, single_reduce_kernel_ptr, dispatch_policy, d_in, d_out, num_items, reduction_op, stream, stream_synchronous);
        }
        else
        {
            // Dispatch a single thread block
            return DispatchSingle(single_reduce_kernel_ptr, dispatch_policy, d_in, d_out, num_items, reduction_op, stream, stream_synchronous);
        }
    }

    #endif // DOXYGEN_SHOULD_SKIP_THIS


    //---------------------------------------------------------------------
    // Public interface
    //---------------------------------------------------------------------


    /**
     * \brief Computes a device-wide reduction using the specified binary \p reduction_op functor.  The implementation is parameterized by the specified tuning policies.
     *
     * \tparam UpsweepTilesPolicy   A parameterization of BlockReduceTilesPolicy for the upsweep reduction kernel
     * \tparam SingleTilesPolicy    A parameterization of BlockReduceTilesPolicy for the single-block reduction kernel
     * \tparam InputIterator        <b>[inferred]</b> The input iterator type (may be a simple pointer type).
     * \tparam OutputIterator       <b>[inferred]</b> The output iterator type (may be a simple pointer type).
     * \tparam SizeT                <b>[inferred]</b> Integer type for offsets
     * \tparam ReductionOp          <b>[inferred]</b> Binary reduction operator type having member <tt>T operator()(const T &a, const T &b)</tt>
     */
    template <
        typename UpsweepTilesPolicy,
        typename SingleTilesPolicy,
        typename InputIterator,
        typename OutputIterator,
        typename SizeT,
        typename ReductionOp>
    __host__ __device__ __forceinline__
    static cudaError_t Reduce(
        InputIterator   d_in,
        OutputIterator  d_out,
        SizeT           num_items,
        ReductionOp     reduction_op,
        cudaStream_t    stream = 0,
        bool            stream_synchronous = false)
    {
        DispatchPolicy dispatch_policy;
        dispatch_policy.Init<UpsweepTilesPolicy, SingleTilesPolicy>();

        // Dispatch
        return CubDebug(Dispatch(
            ReduceKernel<UpsweepTilesPolicy, InputIterator, OutputIterator, SizeT, ReductionOp>,
            SingleReduceKernel<SingleTilesPolicy, InputIterator, OutputIterator, SizeT, ReductionOp>,
            dispatch_policy,
            d_in,
            d_out,
            num_items,
            reduction_op,
            stream,
            stream_synchronous));
    }


    /**
     * \brief Computes a device-wide reduction using the specified binary \p reduction_op functor.
     *
     * \tparam InputIterator        <b>[inferred]</b> The input iterator type (may be a simple pointer type).
     * \tparam OutputIterator       <b>[inferred]</b> The output iterator type (may be a simple pointer type).
     * \tparam SizeT                <b>[inferred]</b> Integer type for offsets
     * \tparam ReductionOp          <b>[inferred]</b> Binary reduction operator type having member <tt>T operator()(const T &a, const T &b)</tt>
     */
    template <
        typename InputIterator,
        typename OutputIterator,
        typename SizeT,
        typename ReductionOp>
    __host__ __device__ __forceinline__
    static cudaError_t Reduce(
        InputIterator   d_in,
        OutputIterator  d_out,
        SizeT           num_items,
        ReductionOp     reduction_op,
        cudaStream_t    stream = 0,
        bool            stream_synchronous = false)
    {
        cudaError error = cudaSuccess;
        do
        {
            // Define tuning polices for the PTX architecture of the current compiler pass
            typedef typename DefaultPolicy<InputIterator, SizeT>::PtxUpsweepTilesPolicy     PtxUpsweepTilesPolicy;
            typedef typename DefaultPolicy<InputIterator, SizeT>::PtxSingleTilesPolicy      PtxSingleTilesPolicy;

            // Declare and initialize a dispatch policy instance with the tuning policy for the target device
            DispatchPolicy dispatch_policy;

        #if CUB_PTX_ARCH > 0

            // We're on the device, so initialize the tuned dispatch policy based upon PTX arch
            dispatch_policy.Init<PtxUpsweepTilesPolicy, PtxSingleTilesPolicy>();

        #else

            // We're on the host, so initialize the tuned dispatch policy based upon the device's SM arch
            int device_ordinal;
            if (CubDebug(error = cudaGetDevice(&device_ordinal))) break;

            int major, minor;
            if (CubDebug(error = cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device_ordinal))) break;
            if (CubDebug(error = cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device_ordinal))) break;
            int device_arch = major * 100 + minor * 10;

            DefaultPolicy<InputIterator, SizeT>::InitDispatchPolicy(device_arch, dispatch_policy);

        #endif

            // Dispatch
            if (CubDebug(error = Dispatch(
                ReduceKernel<PtxUpsweepTilesPolicy, InputIterator, OutputIterator, SizeT, ReductionOp>,
                SingleReduceKernel<PtxSingleTilesPolicy, InputIterator, OutputIterator, SizeT, ReductionOp>,
                dispatch_policy,
                d_in,
                d_out,
                num_items,
                reduction_op,
                stream,
                stream_synchronous))) break;
        }
        while (0);

        return error;
    }

};


/** @} */       // DeviceModule

}               // CUB namespace
CUB_NS_POSTFIX  // Optional outer namespace(s)


