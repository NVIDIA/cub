
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
 * Multi-block reduction kernel entry point.
 */
template <
    typename BlockReduceTilesPolicy,
    typename InputIterator,
    typename OutputIterator,
    typename SizeT,
    typename ReductionOp>
__launch_bounds__ (BlockReduceTilesPolicy::BLOCK_THREADS, 1)
__global__ void MultiReduceKernel(
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

    /// Generic structure for encapsulating dispatch properties.  Mirrors the constants within BlockReduceTilesPolicy.
    struct KernelDispachParams
    {
        int                     block_threads;
        int                     items_per_thread;
        int                     vector_load_length;
        int                     oversubscription;
        GridMappingStrategy     grid_mapping;
        PtxLoadModifier         load_modifier;

        template <typename BlockReduceTilesPolicy>
        __host__ __device__ __forceinline__
        void Init()
        {
            block_threads       = BlockReduceTilesPolicy::BLOCK_THREADS;
            items_per_thread    = BlockReduceTilesPolicy::ITEMS_PER_THREAD;
            vector_load_length  = BlockReduceTilesPolicy::VECTOR_LOAD_LENGTH;
            oversubscription    = BlockReduceTilesPolicy::OVERSUBSCRIPTION;
            grid_mapping        = BlockReduceTilesPolicy::GRID_MAPPING;
            load_modifier       = BlockReduceTilesPolicy::LOAD_MODIFIER;
        }
    };


    /// Implements tuned policy default types
    template <
        typename    T,
        typename    SizeT,
        int         ARCH = CUB_PTX_ARCH>
    struct TunedPolicies;

    /// SM30 tune
    template <typename T, typename SizeT>
    struct TunedPolicies<T, SizeT, 300>
    {
        typedef BlockReduceTilesPolicy<128,     8, GRID_MAPPING_DYNAMIC,     2,  PTX_LOAD_NONE, 1>      MultiPolicy;
        typedef BlockReduceTilesPolicy<32,      4, GRID_MAPPING_EVEN_SHARE,  4,  PTX_LOAD_NONE, 1>      SinglePolicy;
    };

    /// SM20 tune
    template <typename T, typename SizeT>
    struct TunedPolicies<T, SizeT, 200>
    {
        typedef BlockReduceTilesPolicy<128,     8, GRID_MAPPING_DYNAMIC,     2,  PTX_LOAD_NONE, 1>      MultiPolicy;
        typedef BlockReduceTilesPolicy<32,      4, GRID_MAPPING_EVEN_SHARE,  4,  PTX_LOAD_NONE, 1>      SinglePolicy;
    };

    /// SM13 tune
    template <typename T, typename SizeT>
    struct TunedPolicies<T, SizeT, 130>
    {
        typedef BlockReduceTilesPolicy<128,     8, GRID_MAPPING_EVEN_SHARE,  1,  PTX_LOAD_NONE, 1>      MultiPolicy;
        typedef BlockReduceTilesPolicy<32,      4, GRID_MAPPING_EVEN_SHARE,  1,  PTX_LOAD_NONE, 1>      SinglePolicy;
    };

    /// SM10 tune
    template <typename T, typename SizeT>
    struct TunedPolicies<T, SizeT, 100>
    {
        typedef BlockReduceTilesPolicy<128,     8, GRID_MAPPING_EVEN_SHARE,  1,  PTX_LOAD_NONE, 1>      MultiPolicy;
        typedef BlockReduceTilesPolicy<32,      4, GRID_MAPPING_EVEN_SHARE,  1,  PTX_LOAD_NONE, 1>      SinglePolicy;
    };


    /// Selects the appropriate tuned policy types for the targeted problem setting (and the ability to initialize a corresponding DispachParams)
    template <typename T, typename SizeT>
    struct PtxDefaultPolicies
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

        struct MultiPolicy : TunedPolicies<T, SizeT, PTX_TUNE_ARCH>::MultiPolicy {};
        struct SinglePolicy : TunedPolicies<T, SizeT, PTX_TUNE_ARCH>::SinglePolicy {};

        /**
         * Initialize dispatch policy
         */
        static void InitDispatchParams(
            int                     device_arch,
            KernelDispachParams    &multi_dispatch_params,
            KernelDispachParams    &single_dispatch_params)
        {
            if (device_arch >= 300)
            {
                multi_dispatch_params.Init<TunedPolicies<T, SizeT, 300>::MultiPolicy >();
                single_dispatch_params.Init<TunedPolicies<T, SizeT, 300>::SinglePolicy >();
            }
            else if (device_arch >= 200)
            {
                multi_dispatch_params.Init<TunedPolicies<T, SizeT, 200>::MultiPolicy >();
                single_dispatch_params.Init<TunedPolicies<T, SizeT, 200>::SinglePolicy >();
            }
            else if (device_arch >= 130)
            {
                multi_dispatch_params.Init<TunedPolicies<T, SizeT, 130>::MultiPolicy >();
                single_dispatch_params.Init<TunedPolicies<T, SizeT, 130>::SinglePolicy >();
            }
            else
            {
                multi_dispatch_params.Init<TunedPolicies<T, SizeT, 100>::MultiPolicy >();
                single_dispatch_params.Init<TunedPolicies<T, SizeT, 100>::SinglePolicy >();
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
        KernelDispachParams     &single_dispatch_params,
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
                single_dispatch_params.block_threads,
                single_dispatch_params.items_per_thread,
                stream);

            // Invoke ReduceSingle
            single_reduce_kernel_ptr<<<1, single_dispatch_params.block_threads>>>(
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
     * Dispatch two kernels (an multi-block kernel followed by a single-block kernel) to perform the device reduction
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
        ReduceKernelPtr         multi_reduce_kernel_ptr,
        ReduceSingleKernelPtr   single_reduce_kernel_ptr,
        PrepareDrainKernelPtr   prepare_drain_kernel_ptr,
        KernelDispachParams     &multi_dispatch_params,
        KernelDispachParams     &single_dispatch_params,
        InputIterator           d_in,
        OutputIterator          d_out,
        SizeT                   num_items,
        ReductionOp             reduction_op,
        cudaStream_t            stream = 0,
        bool                    stream_synchronous = false,
        DeviceAllocator         *device_allocator = DefaultDeviceAllocator())
    {
    #if !CUB_CNP_ENABLED

        // Kernel launch not supported from this device
        return cudaErrorInvalidConfiguration;

    #else

        // Data type of input iterator
        typedef typename std::iterator_traits<InputIterator>::value_type T;


        T*                      d_block_partials = NULL;    // Temporary storage
        GridEvenShare<SizeT>    even_share;                 // Even-share work distribution
        GridQueue<SizeT>        queue;                      // Dynamic, queue-based work distribution

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

        #ifndef __CUDA_ARCH__

            // We're on the host, so come up with a more accurate estimate of SM occupancies from actual device properties
            Device device_props;
            if (CubDebug(error = device_props.Init(device_ordinal))) break;

            if (CubDebug(error = device_props.MaxSmOccupancy(
                reduce_sm_occupancy,
                multi_reduce_kernel_ptr,
                multi_dispatch_params.block_threads))) break;

        #endif

            // Determine grid size for the multi-block kernel
            int reduce_occupancy = reduce_sm_occupancy * sm_count;
            int multi_tile_size = multi_dispatch_params.block_threads * multi_dispatch_params.items_per_thread;
            int multi_grid_size;

            switch (multi_dispatch_params.grid_mapping)
            {
            case GRID_MAPPING_EVEN_SHARE:

                // Work is distributed evenly
                even_share.GridInit(
                    num_items,
                    reduce_occupancy * multi_dispatch_params.oversubscription,
                    multi_tile_size);

                multi_grid_size = even_share.grid_size;
                break;

            case GRID_MAPPING_DYNAMIC:

                // Work is distributed dynamically
                queue.Allocate(device_allocator);
                int num_tiles = (num_items + multi_tile_size - 1) / multi_tile_size;
                if (num_tiles < reduce_occupancy)
                {
                    // Every thread block gets one input tile each and nothing is queued
                    multi_grid_size = num_tiles;
                }
                else
                {
                    // Thread blocks get one input tile each and the rest are queued.
                    multi_grid_size = reduce_occupancy;

                    // Prepare queue on device (because its faster than if we prepare it here)
                    if (stream_synchronous) CubLog("\tInvoking prepare_drain_kernel_ptr<<<1, 1, 0, %d>>>()\n", stream);

                    prepare_drain_kernel_ptr<<<1, 1, 0, stream>>>(queue, num_items);

                    if (stream_synchronous && CubDebug(error = cudaStreamSynchronize(stream))) break;
                }
                break;
            };

            // Allocate temporary storage for thread block partial reductions
            if (CubDebug(error = device_allocator->DeviceAllocate((void**) &d_block_partials, multi_grid_size * sizeof(T)))) break;

            // Invoke Reduce
            if (stream_synchronous) CubLog("\tInvoking multi_reduce_kernel_ptr<<<%d, %d, 0, %d>>>(), %d items per thread, %d SM occupancy\n",
                multi_grid_size,
                multi_dispatch_params.block_threads,
                stream,
                multi_dispatch_params.items_per_thread,
                reduce_sm_occupancy);

            multi_reduce_kernel_ptr<<<multi_grid_size, multi_dispatch_params.block_threads, 0, stream>>>(
                d_in,
                d_block_partials,
                num_items,
                even_share,
                queue,
                reduction_op);

            if (stream_synchronous && CubDebug(error = cudaStreamSynchronize(stream))) break;

            // Invoke ReduceSingle
            if (stream_synchronous) CubLog("\tInvoking single_reduce_kernel_ptr<<<%d, %d, 0, %d>>>(), %d items per thread\n",
                1,
                single_dispatch_params.block_threads,
                stream,
                single_dispatch_params.items_per_thread);

            single_reduce_kernel_ptr<<<1, single_dispatch_params.block_threads, 0, stream>>>(
                d_block_partials,
                d_out,
                multi_grid_size,
                reduction_op);

            if (stream_synchronous && CubDebug(error = cudaStreamSynchronize(stream))) break;
        }
        while (0);

        // Free temporary storage allocation
        CubDebug(device_allocator->DeviceFree(d_block_partials));

        // Free queue allocation
        if (multi_dispatch_params.grid_mapping == GRID_MAPPING_DYNAMIC)
        {
            CubDebug(queue.Free(device_allocator));
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
        ReduceKernelPtr         multi_reduce_kernel_ptr,
        ReduceSingleKernelPtr   single_reduce_kernel_ptr,
        KernelDispachParams     &multi_dispatch_params,
        KernelDispachParams     &single_dispatch_params,
        InputIterator           d_in,
        OutputIterator          d_out,
        SizeT                   num_items,
        ReductionOp             reduction_op,
        cudaStream_t            stream = 0,
        bool                    stream_synchronous = false,
        DeviceAllocator         *device_allocator = DefaultDeviceAllocator())
    {
        if (num_items > (single_dispatch_params.block_threads * single_dispatch_params.items_per_thread))
        {
            // Dispatch multiple kernels
            return DispatchIterative(
                multi_reduce_kernel_ptr,
                single_reduce_kernel_ptr,
                PrepareDrainKernel<SizeT>,
                multi_dispatch_params,
                single_dispatch_params,
                d_in,
                d_out,
                num_items,
                reduction_op,
                stream,
                stream_synchronous,
                device_allocator);
        }
        else
        {
            // Dispatch a single thread block
            return DispatchSingle(
                single_reduce_kernel_ptr,
                single_dispatch_params,
                d_in,
                d_out,
                num_items,
                reduction_op,
                stream,
                stream_synchronous);
        }
    }

    #endif // DOXYGEN_SHOULD_SKIP_THIS

public:

    //---------------------------------------------------------------------
    // Public interface
    //---------------------------------------------------------------------


    /**
     * \brief Computes a device-wide reduction using the specified binary \p reduction_op functor.  The implementation is parameterized by the specified tuning policies.
     *
     * \tparam MultiPolicy          A parameterization of BlockReduceTilesPolicy for the upsweep reduction kernel
     * \tparam SinglePolicy         A parameterization of BlockReduceTilesPolicy for the single-block reduction kernel
     * \tparam InputIterator        <b>[inferred]</b> The input iterator type (may be a simple pointer type).
     * \tparam OutputIterator       <b>[inferred]</b> The output iterator type (may be a simple pointer type).
     * \tparam SizeT                <b>[inferred]</b> Integer type for offsets
     * \tparam ReductionOp          <b>[inferred]</b> Binary reduction operator type having member <tt>T operator()(const T &a, const T &b)</tt>
     */
    template <
        typename MultiPolicy,
        typename SinglePolicy,
        typename InputIterator,
        typename OutputIterator,
        typename SizeT,
        typename ReductionOp>
    __host__ __device__ __forceinline__
    static cudaError_t Reduce(
        InputIterator   d_in,
        OutputIterator  d_out,
        int             num_items,
        ReductionOp     reduction_op,
        cudaStream_t    stream = 0,
        bool            stream_synchronous = false,
        DeviceAllocator *device_allocator = DefaultDeviceAllocator())
    {
        // Type used for array indexing
        typedef int SizeT;

        // Data type of input iterator
        typedef typename std::iterator_traits<InputIterator>::value_type T;

        // Declare and initialize dispatch parameters
        KernelDispachParams dispatch_params;
        dispatch_params.Init<MultiPolicy, SinglePolicy>();

        // Dispatch
        return CubDebug(Dispatch(
            MultiReduceKernel<MultiPolicy, InputIterator, T*, SizeT, ReductionOp>,
            SingleReduceKernel<SinglePolicy, T*, OutputIterator, SizeT, ReductionOp>,
            dispatch_params,
            d_in,
            d_out,
            num_items,
            reduction_op,
            stream,
            stream_synchronous,
            device_allocator));
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
        typename ReductionOp>
    __host__ __device__ __forceinline__
    static cudaError_t Reduce(
        InputIterator   d_in,
        OutputIterator  d_out,
        int             num_items,
        ReductionOp     reduction_op,
        cudaStream_t    stream = 0,
        bool            stream_synchronous = false,
        DeviceAllocator *device_allocator = DefaultDeviceAllocator())
    {
        // Type used for array indexing
        typedef int SizeT;

        // Data type of input iterator
        typedef typename std::iterator_traits<InputIterator>::value_type T;

        // Tuning polices for the PTX architecture of the current compiler pass
        typedef PtxDefaultPolicies<T, SizeT> PtxDefaultPolicies;
        typedef typename PtxDefaultPolicies::MultiPolicy MultiPolicy;
        typedef typename PtxDefaultPolicies::SinglePolicy SinglePolicy;

        cudaError error = cudaSuccess;
        do
        {
            // Declare dispatch parameters
            KernelDispachParams multi_dispatch_params;
            KernelDispachParams single_dispatch_params;

        #ifdef __CUDA_ARCH__

            // We're on the device, so initialize the dispatch parameters based upon PTX arch
            multi_dispatch_params.Init<MultiPolicy>();
            single_dispatch_params.Init<SinglePolicy>();

        #else

            // We're on the host, so initialize the dispatch parameters based upon the device's SM arch
            int device_ordinal;
            if (CubDebug(error = cudaGetDevice(&device_ordinal))) break;

            int major, minor;
            if (CubDebug(error = cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device_ordinal))) break;
            if (CubDebug(error = cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device_ordinal))) break;
            int device_arch = major * 100 + minor * 10;

            PtxDefaultPolicies::InitDispatchParams(device_arch, multi_dispatch_params, single_dispatch_params);

        #endif

            // Dispatch
            if (CubDebug(error = Dispatch(
                MultiReduceKernel<MultiPolicy, InputIterator, T*, SizeT, ReductionOp>,
                SingleReduceKernel<SinglePolicy, T*, OutputIterator, SizeT, ReductionOp>,
                multi_dispatch_params,
                single_dispatch_params,
                d_in,
                d_out,
                num_items,
                reduction_op,
                stream,
                stream_synchronous,
                device_allocator))) break;
        }
        while (0);

        return error;
    }

};


/** @} */       // DeviceModule

}               // CUB namespace
CUB_NS_POSTFIX  // Optional outer namespace(s)


