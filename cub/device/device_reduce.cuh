
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
__launch_bounds__ (BlockReduceTilesPolicy::BLOCK_THREADS, 1)
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
__launch_bounds__ (BlockReduceTilesPolicy::BLOCK_THREADS, 1)
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
        int                     upsweep_oversubscription;

        // Single kernel policy details
        int                     single_block_threads;
        int                     single_items_per_thread;

        template <typename DefaultPolicy>
        __host__ __device__ __forceinline__
        void Init()
        {
            upsweep_block_threads           = DefaultPolicy::UpsweepTilesPolicy::BLOCK_THREADS;
            upsweep_items_per_thread        = DefaultPolicy::UpsweepTilesPolicy::ITEMS_PER_THREAD;
            upsweep_mapping                 = DefaultPolicy::UpsweepTilesPolicy::GRID_MAPPING;
            upsweep_oversubscription        = DefaultPolicy::UpsweepTilesPolicy::OVERSUBSCRIPTION;

            single_block_threads            = DefaultPolicy::SingleTilesPolicy::BLOCK_THREADS;
            single_items_per_thread         = DefaultPolicy::SingleTilesPolicy::ITEMS_PER_THREAD;
        }
    };


    /// Implements tuned policy default types
    template <
        typename    T,
        typename    SizeT,
        int         ARCH = CUB_PTX_ARCH>
    struct DefaultPolicy;

    /// SM30 tune
    template <typename T, typename SizeT>
    struct DefaultPolicy<T, SizeT, 300>
    {
        typedef BlockReduceTilesPolicy<128,     8, GRID_MAPPING_DYNAMIC,     BLOCK_LOAD_STRIPED,    PTX_LOAD_NONE, 1>      UpsweepTilesPolicy;
        typedef BlockReduceTilesPolicy<32,      4, GRID_MAPPING_EVEN_SHARE,  BLOCK_LOAD_VECTORIZE,  PTX_LOAD_NONE, 1>      SingleTilesPolicy;
    };

    /// SM20 tune
    template <typename T, typename SizeT>
    struct DefaultPolicy<T, SizeT, 200>
    {
        typedef BlockReduceTilesPolicy<128,     8, GRID_MAPPING_DYNAMIC,     BLOCK_LOAD_STRIPED,    PTX_LOAD_NONE, 1>      UpsweepTilesPolicy;
        typedef BlockReduceTilesPolicy<32,      4, GRID_MAPPING_EVEN_SHARE,  BLOCK_LOAD_VECTORIZE,  PTX_LOAD_NONE, 1>      SingleTilesPolicy;
    };

    /// SM13 tune
    template <typename T, typename SizeT>
    struct DefaultPolicy<T, SizeT, 130>
    {
        typedef BlockReduceTilesPolicy<128,     8, GRID_MAPPING_EVEN_SHARE,  BLOCK_LOAD_STRIPED,    PTX_LOAD_NONE, 1>      UpsweepTilesPolicy;
        typedef BlockReduceTilesPolicy<32,      4, GRID_MAPPING_EVEN_SHARE,  BLOCK_LOAD_VECTORIZE,  PTX_LOAD_NONE, 1>      SingleTilesPolicy;
    };

    /// SM10 tune
    template <typename T, typename SizeT>
    struct DefaultPolicy<T, SizeT, 100>
    {
        typedef BlockReduceTilesPolicy<128,     8, GRID_MAPPING_EVEN_SHARE,  BLOCK_LOAD_STRIPED,    PTX_LOAD_NONE, 1>      UpsweepTilesPolicy;
        typedef BlockReduceTilesPolicy<32,      4, GRID_MAPPING_EVEN_SHARE,  BLOCK_LOAD_VECTORIZE,  PTX_LOAD_NONE, 1>      SingleTilesPolicy;
    };


    /// Selects the appropriate tuned policy types for the targeted problem setting (and the ability to initialize a corresponding DispatchPolicy)
    template <
        typename    T,
        typename    SizeT>
    struct PtxDefaultPolicy
    {
        enum
        {
            PTX_TUNE_ARCH =     (CUB_PTX_ARCH >= 300) ?
                                    300 :
                                    (CUB_PTX_ARCH >= 200) ?
                                        200 :
                                        (CUB_PTX_ARCH >= 130) ?
                                            130 :
                                            100
        };

        struct UpsweepTilesPolicy : DefaultPolicy<T, SizeT, PTX_TUNE_ARCH>::UpsweepTilesPolicy {};
        struct SingleTilesPolicy : DefaultPolicy<T, SizeT, PTX_TUNE_ARCH>::SingleTilesPolicy {};

        /**
         * Initialize dispatch policy
         */
        static void InitDispatchPolicy(int device_arch, DispatchPolicy &dispatch_policy)
        {
            if (device_arch >= 300)
            {
                dispatch_policy.Init<DefaultPolicy<T, SizeT, 300> >();
            }
            else if (device_arch >= 200)
            {
                dispatch_policy.Init<DefaultPolicy<T, SizeT, 200> >();
            }
            else if (device_arch >= 130)
            {
                dispatch_policy.Init<DefaultPolicy<T, SizeT, 130> >();
            }
            else
            {
                dispatch_policy.Init<DefaultPolicy<T, SizeT, 100> >();
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
        if (stream_synchronous) CubLog("\tInvoking ReduceSingle<<<1, %d, 0, %d>>>(), %d items per thread\n",
            dispatch_policy.single_block_threads,
            dispatch_policy.single_items_per_thread,
            stream);

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

#endif
    }


    /**
     * Dispatch two kernels (an upsweep multi-block kernel followed by a single-block kernel) to perform the device reduction
     */
    template <
        typename ReduceKernelPtr,
        typename ReduceSingleKernelPtr,
        typename PrepareDrainKernelPtr,
        typename InputIterator,
        typename OutputIterator,
        typename SizeT,
        typename ReductionOp>
    __host__ __device__ __forceinline__
    static cudaError_t DispatchIterative(
        ReduceKernelPtr         reduce_kernel_ptr,
        ReduceSingleKernelPtr   single_reduce_kernel_ptr,
        PrepareDrainKernelPtr   prepare_drain_kernel_ptr,
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

        T*                      d_block_partials = NULL;
        GridEvenShare<SizeT>    upsweep_even_share;
        GridQueue<SizeT>        upsweep_queue;

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
            int reduce_sm_occupancy = ArchProps<CUB_PTX_ARCH>::MAX_SM_THREADBLOCKS;

        #if (CUB_PTX_ARCH == 0)

            // We're on the host, so come up with a more accurate estimate of SM occupancies from actual device properties
            Device device_props;
            if (CubDebug(error = device_props.Init(device_ordinal))) break;

            if (CubDebug(error = device_props.MaxSmOccupancy(
                reduce_sm_occupancy,
                reduce_kernel_ptr,
                dispatch_policy.upsweep_block_threads))) break;

        #endif


            int reduce_occupancy    = reduce_sm_occupancy * sm_count;
            int upsweep_tile_size   = dispatch_policy.upsweep_block_threads * dispatch_policy.upsweep_items_per_thread;
            int upsweep_grid_size;

            switch (dispatch_policy.upsweep_mapping)
            {
            case GRID_MAPPING_EVEN_SHARE:

                // Even share
                upsweep_even_share.GridInit(
                    num_items,
                    reduce_occupancy * dispatch_policy.upsweep_oversubscription,
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
                }
                else
                {
                    // Thread blocks get one input tile each and the rest are queued.
                    upsweep_grid_size = reduce_occupancy;

                    // Prepare queue on device (because its faster than if we prepare it here)
                    if (stream_synchronous) CubLog("\tInvoking prepare_drain_kernel_ptr<<<1, 1, 0, %d>>>()\n", stream);

                    prepare_drain_kernel_ptr<<<1, 1, 0, stream>>>(upsweep_queue, num_items);

                    if (stream_synchronous && CubDebug(error = cudaStreamSynchronize(stream))) break;
                }
                break;
            };

            // Allocate temporary storage for thread block partial reductions
            if (CubDebug(error = DeviceAllocate((void**) &d_block_partials, upsweep_grid_size * sizeof(T)))) break;

            // Invoke Reduce
            if (stream_synchronous) CubLog("\tInvoking reduce_kernel_ptr<<<%d, %d, 0, %d>>>(), %d items per thread, %d SM occupancy\n",
                upsweep_grid_size,
                dispatch_policy.upsweep_block_threads,
                stream,
                dispatch_policy.upsweep_items_per_thread,
                reduce_sm_occupancy);

            reduce_kernel_ptr<<<upsweep_grid_size, dispatch_policy.upsweep_block_threads, 0, stream>>>(
                d_in,
                d_block_partials,
                num_items,
                upsweep_even_share,
                upsweep_queue,
                reduction_op);

            if (stream_synchronous && CubDebug(error = cudaStreamSynchronize(stream))) break;

            // Invoke ReduceSingle
            if (stream_synchronous) CubLog("\tInvoking single_reduce_kernel_ptr<<<%d, %d, 0, %d>>>(), %d items per thread\n",
                1,
                dispatch_policy.single_block_threads,
                stream,
                dispatch_policy.single_items_per_thread);

            single_reduce_kernel_ptr<<<1, dispatch_policy.single_block_threads, 0, stream>>>(
                d_block_partials,
                d_out,
                upsweep_grid_size,
                reduction_op);

            if (stream_synchronous && CubDebug(error = cudaStreamSynchronize(stream))) break;
        }
        while (0);

        // Free temporary storage allocation
        CubDebug(DeviceFree(d_block_partials));

        // Free queue allocation
        if (dispatch_policy.upsweep_mapping == GRID_MAPPING_DYNAMIC)
        {
            CubDebug(upsweep_queue.Free());
        }

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
            return DispatchIterative(
                reduce_kernel_ptr,
                single_reduce_kernel_ptr,
                PrepareDrainKernel<SizeT>,
                dispatch_policy,
                d_in,
                d_out,
                num_items,
                reduction_op,
                stream,
                stream_synchronous);
        }
        else
        {
            // Dispatch a single thread block
            return DispatchSingle(
                single_reduce_kernel_ptr,
                dispatch_policy,
                d_in,
                d_out,
                num_items,
                reduction_op,
                stream,
                stream_synchronous);
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
        // Data type of input iterator
        typedef typename std::iterator_traits<InputIterator>::value_type T;

        DispatchPolicy dispatch_policy;
        dispatch_policy.Init<UpsweepTilesPolicy, SingleTilesPolicy>();

        // Dispatch
        return CubDebug(Dispatch(
            ReduceKernel<UpsweepTilesPolicy, InputIterator, T*, SizeT, ReductionOp>,
            SingleReduceKernel<SingleTilesPolicy, T*, OutputIterator, SizeT, ReductionOp>,
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
        // Data type of input iterator
        typedef typename std::iterator_traits<InputIterator>::value_type T;

        cudaError error = cudaSuccess;
        do
        {
            // Declare and initialize a dispatch policy instance with the tuning policy for the target device
            DispatchPolicy dispatch_policy;

            // Define tuning polices for the PTX architecture of the current compiler pass
            typedef PtxDefaultPolicy<T, SizeT> TunedPolicy;

        #if CUB_PTX_ARCH > 0

            // We're on the device, so initialize the tuned dispatch policy based upon PTX arch
            dispatch_policy.Init<TunedPolicy>();

        #else

            // We're on the host, so initialize the tuned dispatch policy based upon the device's SM arch
            int device_ordinal;
            if (CubDebug(error = cudaGetDevice(&device_ordinal))) break;

            int major, minor;
            if (CubDebug(error = cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device_ordinal))) break;
            if (CubDebug(error = cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device_ordinal))) break;
            int device_arch = major * 100 + minor * 10;

            TunedPolicy::InitDispatchPolicy(device_arch, dispatch_policy);

        #endif

            // Dispatch
            if (CubDebug(error = Dispatch(
                ReduceKernel<typename TunedPolicy::UpsweepTilesPolicy, InputIterator, T*, SizeT, ReductionOp>,
                SingleReduceKernel<typename TunedPolicy::SingleTilesPolicy, T*, OutputIterator, SizeT, ReductionOp>,
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


