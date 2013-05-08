
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
 * cub::DeviceReduce provides operations for computing a device-wide, parallel reduction across data items residing within global memory.
 */

#pragma once

#include <stdio.h>
#include <iterator>

#include "persistent_block/persistent_block_reduce.cuh"
#include "../util_allocator.cuh"
#include "../grid/grid_even_share.cuh"
#include "../grid/grid_queue.cuh"
#include "../grid/grid_mapping.cuh"

/// Optional outer namespace(s)
CUB_NS_PREFIX

/// CUB namespace
namespace cub {


/******************************************************************************
 * Kernel entry points
 *****************************************************************************/

#ifndef DOXYGEN_SHOULD_SKIP_THIS    // Do not document


/**
 * Multi-block reduction kernel entry point.  Computes privatized reductions, one per thread block.
 */
template <
    typename                PersistentBlockReducePolicy,  ///< Tuning policy for cub::PersistentBlockReduce abstraction
    typename                InputIteratorRA,        ///< The random-access iterator type for input (may be a simple pointer type).
    typename                OutputIteratorRA,       ///< The random-access iterator type for output (may be a simple pointer type).
    typename                SizeT,                  ///< Integral type used for global array indexing
    typename                ReductionOp>            ///< Binary reduction operator type having member <tt>T operator()(const T &a, const T &b)</tt>
__launch_bounds__ (int(PersistentBlockReducePolicy::BLOCK_THREADS), 1)
__global__ void MultiBlockReduceKernel(
    InputIteratorRA         d_in,                   ///< [in] Input data to reduce
    OutputIteratorRA        d_out,                  ///< [out] Output location for result
    SizeT                   num_items,              ///< [in] Total number of input data items
    GridEvenShare<SizeT>    even_share,             ///< [in] Descriptor for how to map an even-share of tiles across thread blocks
    GridQueue<SizeT>        queue,                  ///< [in] Descriptor for performing dynamic mapping of tile data to thread blocks
    ReductionOp             reduction_op)           ///< [in] Binary reduction operator
{
    // Data type
    typedef typename std::iterator_traits<InputIteratorRA>::value_type T;

    // Thread block type for reducing input tiles
    typedef PersistentBlockReduce<PersistentBlockReducePolicy, InputIteratorRA, SizeT, ReductionOp> PersistentBlockReduceT;

    // Block-wide aggregate
    T block_aggregate;

    // Shared memory storage
    __shared__ typename PersistentBlockReduceT::SmemStorage smem_storage;

    // Thread block instance
    PersistentBlockReduceT persistent_block(smem_storage, d_in, reduction_op);

    // Consume tiles using thread block instance
    GridMapping<PersistentBlockReducePolicy::GRID_MAPPING>::ConsumeTilesFlagFirst(
        persistent_block, num_items, even_share, queue, block_aggregate);

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
    typename                PersistentBlockReducePolicy,  ///< Tuning policy for cub::PersistentBlockReduce abstraction
    typename                InputIteratorRA,        ///< The random-access iterator type for input (may be a simple pointer type).
    typename                OutputIteratorRA,       ///< The random-access iterator type for output (may be a simple pointer type).
    typename                SizeT,                  ///< Integral type used for global array indexing
    typename                ReductionOp>            ///< Binary reduction operator type having member <tt>T operator()(const T &a, const T &b)</tt>
__launch_bounds__ (int(PersistentBlockReducePolicy::BLOCK_THREADS), 1)
__global__ void SingleBlockReduceKernel(
    InputIteratorRA         d_in,                   ///< [in] Input data to reduce
    OutputIteratorRA        d_out,                  ///< [out] Output location for result
    SizeT                   num_items,              ///< [in] Total number of input data items
    ReductionOp             reduction_op)           ///< [in] Binary reduction operator
{
    // Data type
    typedef typename std::iterator_traits<InputIteratorRA>::value_type T;

    // Thread block type for reducing input tiles
    typedef PersistentBlockReduce<PersistentBlockReducePolicy, InputIteratorRA, SizeT, ReductionOp> PersistentBlockReduceT;

    // Block-wide aggregate
    T block_aggregate;

    // Shared memory storage
    __shared__ typename PersistentBlockReduceT::SmemStorage smem_storage;

    // Block abstraction for reducing tiles
    PersistentBlockReduceT tiles(smem_storage, d_in, reduction_op);

    // Reduce input tiles
    ConsumeTiles(tiles, 0, num_items, block_aggregate);

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
 * \brief DeviceReduce provides operations for computing a device-wide, parallel reduction across data items residing within global memory. ![](reduce_logo.png)
 */
struct DeviceReduce
{
#ifndef DOXYGEN_SHOULD_SKIP_THIS    // Do not document

    /// Generic structure for encapsulating dispatch properties.  Mirrors the constants within PersistentBlockReducePolicy.
    struct KernelDispachParams
    {
        // Policy fields
        int                     block_threads;
        int                     items_per_thread;
        int                     vector_load_length;
        BlockReduceAlgorithm    block_algorithm;
        PtxLoadModifier         load_modifier;
        GridMappingStrategy     grid_mapping;
        int                     subscription_factor;

        // Derived fields
        int                     tile_size;

        template <typename PersistentBlockReducePolicy>
        __host__ __device__ __forceinline__
        void Init(int subscription_factor = 1)
        {
            block_threads               = PersistentBlockReducePolicy::BLOCK_THREADS;
            items_per_thread            = PersistentBlockReducePolicy::ITEMS_PER_THREAD;
            vector_load_length          = PersistentBlockReducePolicy::VECTOR_LOAD_LENGTH;
            block_algorithm             = PersistentBlockReducePolicy::BLOCK_ALGORITHM;
            load_modifier               = PersistentBlockReducePolicy::LOAD_MODIFIER;
            grid_mapping                = PersistentBlockReducePolicy::GRID_MAPPING;
            this->subscription_factor   = subscription_factor;

            tile_size = block_threads * items_per_thread;
        }

        __host__ __device__ __forceinline__
        void Print()
        {
            printf("%d, %d, %d, %d, %d, %d, %d",
                block_threads,
                items_per_thread,
                vector_load_length,
                block_algorithm,
                load_modifier,
                grid_mapping,
                subscription_factor);
        }

    };


    /// Specializations of tuned policy types for different PTX architectures
    template <
        typename    T,
        typename    SizeT,
        int         ARCH>
    struct TunedPolicies;

    /// SM35 tune
    template <typename T, typename SizeT>
    struct TunedPolicies<T, SizeT, 350>
    {
        // K20C: 182.1 @ 48M 32-bit T
        typedef PersistentBlockReducePolicy<256, 8,  2, BLOCK_REDUCE_RAKING, PTX_LOAD_NONE, GRID_MAPPING_EVEN_SHARE>             MultiBlockPolicy;
        typedef PersistentBlockReducePolicy<256, 16, 2, BLOCK_REDUCE_WARP_REDUCTIONS, PTX_LOAD_NONE, GRID_MAPPING_EVEN_SHARE>    SingleBlockPolicy;
        enum { SUBSCRIPTION_FACTOR = 4 };
    };

    /// SM30 tune
    template <typename T, typename SizeT>
    struct TunedPolicies<T, SizeT, 300>
    {
        // GTX670: 154.0 @ 48M 32-bit T
        typedef PersistentBlockReducePolicy<256, 2,  1, BLOCK_REDUCE_WARP_REDUCTIONS,  PTX_LOAD_NONE, GRID_MAPPING_EVEN_SHARE>            MultiBlockPolicy;
        typedef PersistentBlockReducePolicy<256, 24, 4, BLOCK_REDUCE_WARP_REDUCTIONS,  PTX_LOAD_NONE, GRID_MAPPING_EVEN_SHARE>   SingleBlockPolicy;
        enum { SUBSCRIPTION_FACTOR = 1 };
    };

    /// SM20 tune
    template <typename T, typename SizeT>
    struct TunedPolicies<T, SizeT, 200>
    {
        // GTX 580: 178.9 @ 48M 32-bit T
        typedef PersistentBlockReducePolicy<128, 8,  2, BLOCK_REDUCE_RAKING, PTX_LOAD_NONE, GRID_MAPPING_DYNAMIC>                MultiBlockPolicy;
        typedef PersistentBlockReducePolicy<128, 4,  1, BLOCK_REDUCE_RAKING, PTX_LOAD_NONE, GRID_MAPPING_EVEN_SHARE>             SingleBlockPolicy;
        enum { SUBSCRIPTION_FACTOR = 1 };
    };

    /// SM13 tune
    template <typename T, typename SizeT>
    struct TunedPolicies<T, SizeT, 130>
    {
        typedef PersistentBlockReducePolicy<128, 8,  2,  BLOCK_REDUCE_RAKING, PTX_LOAD_NONE, GRID_MAPPING_EVEN_SHARE>            MultiBlockPolicy;
        typedef PersistentBlockReducePolicy<32,  4,  4,  BLOCK_REDUCE_RAKING, PTX_LOAD_NONE, GRID_MAPPING_EVEN_SHARE>            SingleBlockPolicy;
        enum { SUBSCRIPTION_FACTOR = 1 };
    };

    /// SM10 tune
    template <typename T, typename SizeT>
    struct TunedPolicies<T, SizeT, 100>
    {
        typedef PersistentBlockReducePolicy<128, 8,  2,  BLOCK_REDUCE_RAKING, PTX_LOAD_NONE, GRID_MAPPING_EVEN_SHARE>            MultiBlockPolicy;
        typedef PersistentBlockReducePolicy<32,  4,  4,  BLOCK_REDUCE_RAKING, PTX_LOAD_NONE, GRID_MAPPING_EVEN_SHARE>            SingleBlockPolicy;
        enum { SUBSCRIPTION_FACTOR = 1 };
    };


    /// Tuning policy(ies) for the PTX architecture that DeviceReduce operations will get dispatched to
    template <typename T, typename SizeT>
    struct PtxDefaultPolicies
    {
        static const int PTX_TUNE_ARCH =   (CUB_PTX_ARCH >= 350) ?
                                                350 :
                                                (CUB_PTX_ARCH >= 300) ?
                                                    300 :
                                                    (CUB_PTX_ARCH >= 200) ?
                                                        200 :
                                                        (CUB_PTX_ARCH >= 130) ?
                                                            130 :
                                                            100;

        // Tuned policy set for the current PTX compiler pass
        typedef TunedPolicies<T, SizeT, PTX_TUNE_ARCH> PtxPassTunedPolicies;

        // Subscription factor for the current PTX compiler pass
        static const int SUBSCRIPTION_FACTOR = PtxPassTunedPolicies::SUBSCRIPTION_FACTOR;

        // MultiBlockPolicy that opaquely derives from the specialization corresponding to the current PTX compiler pass
        struct MultiBlockPolicy : PtxPassTunedPolicies::MultiBlockPolicy {};

        // SingleBlockPolicy that opaquely derives from the specialization corresponding to the current PTX compiler pass
        struct SingleBlockPolicy : PtxPassTunedPolicies::SingleBlockPolicy {};


        /**
         * Initialize dispatch params with the policies corresponding to the PTX assembly we will use
         */
        static void InitDispatchParams(
            int                    ptx_version,
            KernelDispachParams    &multi_block_dispatch_params,
            KernelDispachParams    &single_block_dispatch_params)
        {
            if (ptx_version >= 350)
            {
                typedef TunedPolicies<T, SizeT, 350> TunedPolicies;
                multi_block_dispatch_params.Init<typename TunedPolicies::MultiBlockPolicy>(TunedPolicies::SUBSCRIPTION_FACTOR);
                single_block_dispatch_params.Init<typename TunedPolicies::SingleBlockPolicy >();
            }
            else if (ptx_version >= 300)
            {
                typedef TunedPolicies<T, SizeT, 300> TunedPolicies;
                multi_block_dispatch_params.Init<typename TunedPolicies::MultiBlockPolicy>(TunedPolicies::SUBSCRIPTION_FACTOR);
                single_block_dispatch_params.Init<typename TunedPolicies::SingleBlockPolicy >();
            }
            else if (ptx_version >= 200)
            {
                typedef TunedPolicies<T, SizeT, 200> TunedPolicies;
                multi_block_dispatch_params.Init<typename TunedPolicies::MultiBlockPolicy>(TunedPolicies::SUBSCRIPTION_FACTOR);
                single_block_dispatch_params.Init<typename TunedPolicies::SingleBlockPolicy >();
            }
            else if (ptx_version >= 130)
            {
                typedef TunedPolicies<T, SizeT, 130> TunedPolicies;
                multi_block_dispatch_params.Init<typename TunedPolicies::MultiBlockPolicy>(TunedPolicies::SUBSCRIPTION_FACTOR);
                single_block_dispatch_params.Init<typename TunedPolicies::SingleBlockPolicy >();
            }
            else
            {
                typedef TunedPolicies<T, SizeT, 100> TunedPolicies;
                multi_block_dispatch_params.Init<typename TunedPolicies::MultiBlockPolicy>(TunedPolicies::SUBSCRIPTION_FACTOR);
                single_block_dispatch_params.Init<typename TunedPolicies::SingleBlockPolicy >();
            }
        }
    };


    /**
     * Internal dispatch routine for computing a device-wide reduction using a single thread block.
     */
    template <
        typename                ReduceSingleKernelPtr,                              ///< Function type of cub::SingleBlockReduceKernel
        typename                InputIteratorRA,                                    ///< The random-access iterator type for input (may be a simple pointer type).
        typename                OutputIteratorRA,                                   ///< The random-access iterator type for output (may be a simple pointer type).
        typename                SizeT,                                              ///< Integral type used for global array indexing
        typename                ReductionOp>                                        ///< Binary reduction operator type having member <tt>T operator()(const T &a, const T &b)</tt>
    __host__ __device__ __forceinline__
    static cudaError_t DispatchSingle(
        ReduceSingleKernelPtr   single_block_kernel,                                ///< [in] Kernel function pointer to parameterization of cub::SingleBlockReduceKernel
        KernelDispachParams     &single_block_dispatch_params,                      ///< [in] Dispatch parameters that match the policy that \p single_block_kernel was compiled for
        InputIteratorRA         d_in,                                               ///< [in] Input data to reduce
        OutputIteratorRA        d_out,                                              ///< [out] Output location for result
        SizeT                   num_items,                                          ///< [in] Number of items to reduce
        ReductionOp             reduction_op,                                       ///< [in] Binary reduction operator
        cudaStream_t            stream              = 0,                            ///< [in] <b>[optional]</b> CUDA stream to launch kernels within.  Default is stream-0.
        bool                    stream_synchronous  = false)                        ///< [in] <b>[optional]</b> Whether or not to synchronize the stream after every kernel launch to check for errors.  Default is \p false.
    {
    #ifndef CUB_RUNTIME_ENABLED

        // Kernel launch not supported from this device
        return CubDebug(cudaErrorInvalidConfiguration);

    #else

        cudaError error = cudaSuccess;
        do
        {
            if (stream_synchronous) CubLog("Invoking ReduceSingle<<<1, %d, 0, %d>>>(), %d items per thread\n",
                single_block_dispatch_params.block_threads, (int) stream, single_block_dispatch_params.items_per_thread);

            // Invoke ReduceSingle
            single_block_kernel<<<1, single_block_dispatch_params.block_threads>>>(
                d_in,
                d_out,
                num_items,
                reduction_op);

            #ifndef __CUDA_ARCH__
                // Sync the stream on the host
                if (stream_synchronous && CubDebug(error = cudaStreamSynchronize(stream))) break;
            #else
                // Sync the entire device on the device (cudaStreamSynchronize doesn't exist on device)
                if (stream_synchronous && CubDebug(error = cudaDeviceSynchronize())) break;
            #endif

        }
        while (0);
        return error;

    #endif
    }


    /**
     * Internal dispatch routine for computing a device-wide reduction using a two-stages of kernel invocations.
     */
    template <
        typename                    MultiBlockReduceKernelPtr,                          ///< Function type of cub::MultiBlockReduceKernel
        typename                    ReduceSingleKernelPtr,                              ///< Function type of cub::SingleBlockReduceKernel
        typename                    ResetDrainKernelPtr,                                ///< Function type of cub::ResetDrainKernel
        typename                    InputIteratorRA,                                    ///< The random-access iterator type for input (may be a simple pointer type).
        typename                    OutputIteratorRA,                                   ///< The random-access iterator type for output (may be a simple pointer type).
        typename                    SizeT,                                              ///< Integral type used for global array indexing
        typename                    ReductionOp>                                        ///< Binary reduction operator type having member <tt>T operator()(const T &a, const T &b)</tt>
    __host__ __device__ __forceinline__
    static cudaError_t DispatchIterative(
        MultiBlockReduceKernelPtr   multi_block_kernel,                                 ///< [in] Kernel function pointer to parameterization of cub::MultiBlockReduceKernel
        ReduceSingleKernelPtr       single_block_kernel,                                ///< [in] Kernel function pointer to parameterization of cub::SingleBlockReduceKernel
        ResetDrainKernelPtr         prepare_drain_kernel,                               ///< [in] Kernel function pointer to parameterization of cub::ResetDrainKernel
        KernelDispachParams         &multi_block_dispatch_params,                       ///< [in] Dispatch parameters that match the policy that \p multi_block_kernel_ptr was compiled for
        KernelDispachParams         &single_block_dispatch_params,                      ///< [in] Dispatch parameters that match the policy that \p single_block_kernel was compiled for
        InputIteratorRA             d_in,                                               ///< [in] Input data to reduce
        OutputIteratorRA            d_out,                                              ///< [out] Output location for result
        SizeT                       num_items,                                          ///< [in] Number of items to reduce
        ReductionOp                 reduction_op,                                       ///< [in] Binary reduction operator
        cudaStream_t                stream              = 0,                            ///< [in] <b>[optional]</b> CUDA stream to launch kernels within.  Default is stream-0.
        bool                        stream_synchronous  = false,                        ///< [in] <b>[optional]</b> Whether or not to synchronize the stream after every kernel launch to check for errors.  Default is \p false.
        DeviceAllocator             *device_allocator   = DefaultDeviceAllocator())     ///< [in] <b>[optional]</b> Allocator for allocating and freeing device memory.  Default is provided by DefaultDeviceAllocator.
    {
    #ifndef CUB_RUNTIME_ENABLED

        // Kernel launch not supported from this device
        return CubDebug(cudaErrorInvalidConfiguration);

    #else

        // Data type of input iterator
        typedef typename std::iterator_traits<InputIteratorRA>::value_type T;

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
            int multi_sm_occupancy = CUB_MIN(
                ArchProps<CUB_PTX_ARCH>::MAX_SM_THREADBLOCKS,
                ArchProps<CUB_PTX_ARCH>::MAX_SM_THREADS / multi_block_dispatch_params.block_threads);

        #ifndef __CUDA_ARCH__

            // We're on the host, so come up with a more accurate estimate of SM occupancies from actual device properties
            Device device_props;
            if (CubDebug(error = device_props.Init(device_ordinal))) break;

            if (CubDebug(error = device_props.MaxSmOccupancy(
                multi_sm_occupancy,
                multi_block_kernel,
                multi_block_dispatch_params.block_threads))) break;

        #endif

            // Determine grid size for the multi-block kernel
            int multi_occupancy = multi_sm_occupancy * sm_count;
            int multi_tile_size = multi_block_dispatch_params.block_threads * multi_block_dispatch_params.items_per_thread;
            int multi_grid_size;

            switch (multi_block_dispatch_params.grid_mapping)
            {
            case GRID_MAPPING_EVEN_SHARE:

                // Work is distributed evenly
                even_share.GridInit(
                    num_items,
                    multi_occupancy * multi_block_dispatch_params.subscription_factor,
                    multi_tile_size);

                multi_grid_size = even_share.grid_size;
                break;

            case GRID_MAPPING_DYNAMIC:

                // Work is distributed dynamically
                queue.Allocate(device_allocator);
                int num_tiles = (num_items + multi_tile_size - 1) / multi_tile_size;
                if (num_tiles < multi_occupancy)
                {
                    // Every thread block gets one input tile each and nothing is queued
                    multi_grid_size = num_tiles;
                }
                else
                {
                    // Thread blocks get one input tile each and the rest are queued.
                    multi_grid_size = multi_occupancy;

                #ifndef __CUDA_ARCH__

                    // We're on the host, so prepare queue on device (because its faster than if we prepare it here)
                    if (stream_synchronous) CubLog("Invoking prepare_drain_kernel<<<1, 1, 0, %d>>>()\n", (int) stream);

                    prepare_drain_kernel<<<1, 1, 0, stream>>>(queue, num_items);

                    // Sync the stream on the host
                    if (stream_synchronous && CubDebug(error = cudaStreamSynchronize(stream))) break;

                #else

                    // Prepare the queue here
                    queue.ResetDrain(num_items);

                #endif
                }
                break;
            };

            // Allocate temporary storage for thread block partial reductions
            if (CubDebug(error = DeviceAllocate((void**) &d_block_partials, multi_grid_size * sizeof(T), device_allocator))) break;

            // Invoke MultiBlockReduce
            if (stream_synchronous) CubLog("Invoking multi_block_kernel<<<%d, %d, 0, %d>>>(), %d items per thread, %d SM occupancy\n",
                multi_grid_size, multi_block_dispatch_params.block_threads, (int) stream, multi_block_dispatch_params.items_per_thread, multi_sm_occupancy);

            multi_block_kernel<<<multi_grid_size, multi_block_dispatch_params.block_threads, 0, stream>>>(
                d_in,
                d_block_partials,
                num_items,
                even_share,
                queue,
                reduction_op);

            #ifndef __CUDA_ARCH__
                // Sync the stream on the host
                if (stream_synchronous && CubDebug(error = cudaStreamSynchronize(stream))) break;
            #else
                // Sync the entire device on the device (cudaStreamSynchronize doesn't exist on device)
                if (stream_synchronous && CubDebug(error = cudaDeviceSynchronize())) break;
            #endif

            // Invoke SingleBlockReduce
            if (stream_synchronous) CubLog("Invoking single_block_kernel<<<%d, %d, 0, %d>>>(), %d items per thread\n",
                1, single_block_dispatch_params.block_threads, (int) stream, single_block_dispatch_params.items_per_thread);

            single_block_kernel<<<1, single_block_dispatch_params.block_threads, 0, stream>>>(
                d_block_partials,
                d_out,
                multi_grid_size,
                reduction_op);

            #ifndef __CUDA_ARCH__
                // Sync the stream on the host
                if (stream_synchronous && CubDebug(error = cudaStreamSynchronize(stream))) break;
            #else
                // Sync the entire device on the device (cudaStreamSynchronize doesn't exist on device)
                if (stream_synchronous && CubDebug(error = cudaDeviceSynchronize())) break;
            #endif
        }
        while (0);

        // Free temporary storage allocation
        if (d_block_partials) error = CubDebug(DeviceFree(d_block_partials, device_allocator));

        // Free queue allocation
        if (multi_block_dispatch_params.grid_mapping == GRID_MAPPING_DYNAMIC) error = CubDebug(queue.Free(device_allocator));

        return error;

    #endif
    }


    /**
     * Internal device reduction dispatch
     */
    template <
        typename                    MultiBlockReduceKernelPtr,                          ///< Function type of cub::MultiBlockReduceKernel
        typename                    ReduceSingleKernelPtr,                              ///< Function type of cub::SingleBlockReduceKernel
        typename                    InputIteratorRA,                                    ///< The random-access iterator type for input (may be a simple pointer type).
        typename                    OutputIteratorRA,                                   ///< The random-access iterator type for output (may be a simple pointer type).
        typename                    SizeT,                                              ///< Integral type used for global array indexing
        typename                    ReductionOp>                                        ///< Binary reduction operator type having member <tt>T operator()(const T &a, const T &b)</tt>
    __host__ __device__ __forceinline__
    static cudaError_t Dispatch(
        MultiBlockReduceKernelPtr   multi_block_kernel,                                 ///< [in] Kernel function pointer to parameterization of cub::MultiBlockReduceKernel
        ReduceSingleKernelPtr       single_block_kernel,                                ///< [in] Kernel function pointer to parameterization of cub::SingleBlockReduceKernel
        KernelDispachParams         &multi_block_dispatch_params,                       ///< [in] Dispatch parameters that match the policy that \p multi_block_kernel_ptr was compiled for
        KernelDispachParams         &single_block_dispatch_params,                      ///< [in] Dispatch parameters that match the policy that \p single_block_kernel was compiled for
        InputIteratorRA             d_in,                                               ///< [in] Input data to reduce
        OutputIteratorRA            d_out,                                              ///< [out] Output location for result
        SizeT                       num_items,                                          ///< [in] Number of items to reduce
        ReductionOp                 reduction_op,                                       ///< [in] Binary reduction operator
        cudaStream_t                stream              = 0,                            ///< [in] <b>[optional]</b> CUDA stream to launch kernels within.  Default is stream-0.
        bool                        stream_synchronous  = false,                        ///< [in] <b>[optional]</b> Whether or not to synchronize the stream after every kernel launch to check for errors.  Default is \p false.
        DeviceAllocator             *device_allocator   = DefaultDeviceAllocator())     ///< [in] <b>[optional]</b> Allocator for allocating and freeing device memory.  Default is provided by DefaultDeviceAllocator.
    {
        if (num_items > (single_block_dispatch_params.block_threads * single_block_dispatch_params.items_per_thread))
        {
            // Dispatch multiple kernels
            return CubDebug(DispatchIterative(
                multi_block_kernel,
                single_block_kernel,
                ResetDrainKernel<SizeT>,
                multi_block_dispatch_params,
                single_block_dispatch_params,
                d_in,
                d_out,
                num_items,
                reduction_op,
                stream,
                stream_synchronous,
                device_allocator));
        }
        else
        {
            // Dispatch a single thread block
            return CubDebug(DispatchSingle(
                single_block_kernel,
                single_block_dispatch_params,
                d_in,
                d_out,
                num_items,
                reduction_op,
                stream,
                stream_synchronous));
        }
    }

#endif // DOXYGEN_SHOULD_SKIP_THIS

    //---------------------------------------------------------------------
    // Public interface
    //---------------------------------------------------------------------

    /**
     * \brief Computes a device-wide reduction using the specified binary \p reduction_op functor.
     *
     * \tparam InputIteratorRA      <b>[inferred]</b> The random-access iterator type for input (may be a simple pointer type).
     * \tparam OutputIteratorRA     <b>[inferred]</b> The random-access iterator type for output (may be a simple pointer type).
     * \tparam ReductionOp          <b>[inferred]</b> Binary reduction operator type having member <tt>T operator()(const T &a, const T &b)</tt>
     */
    template <
        typename            InputIteratorRA,
        typename            OutputIteratorRA,
        typename            ReductionOp>
    __host__ __device__ __forceinline__
    static cudaError_t Reduce(
        InputIteratorRA     d_in,                                               ///< [in] Input data to reduce
        OutputIteratorRA    d_out,                                              ///< [out] Output location for result
        int                 num_items,                                          ///< [in] Number of items to reduce
        ReductionOp         reduction_op,                                       ///< [in] Binary reduction operator
        cudaStream_t        stream              = 0,                            ///< [in] <b>[optional]</b> CUDA stream to launch kernels within.  Default is stream-0.
        bool                stream_synchronous  = false,                        ///< [in] <b>[optional]</b> Whether or not to synchronize the stream after every kernel launch to check for errors.  Default is \p false.
        DeviceAllocator     *device_allocator   = DefaultDeviceAllocator())     ///< [in] <b>[optional]</b> Allocator for allocating and freeing device memory.  Default is provided by DefaultDeviceAllocator.
    {
        // Type used for array indexing
        typedef int SizeT;

        // Data type of input iterator
        typedef typename std::iterator_traits<InputIteratorRA>::value_type T;

        // Tuning polices for the PTX architecture that will get dispatched to
        typedef PtxDefaultPolicies<T, SizeT> PtxDefaultPolicies;
        typedef typename PtxDefaultPolicies::MultiBlockPolicy MultiBlockPolicy;       // Multi-block kernel policy
        typedef typename PtxDefaultPolicies::SingleBlockPolicy SingleBlockPolicy;     // Single-block kernel policy

        cudaError error = cudaSuccess;
        do
        {
            // Declare dispatch parameters
            KernelDispachParams multi_block_dispatch_params;
            KernelDispachParams single_block_dispatch_params;

        #ifdef __CUDA_ARCH__

            // We're on the device, so initialize the dispatch parameters with the PtxDefaultPolicies directly
            multi_block_dispatch_params.Init<MultiBlockPolicy>(PtxDefaultPolicies::SUBSCRIPTION_FACTOR);
            single_block_dispatch_params.Init<SingleBlockPolicy>();

        #else

            // We're on the host, so lookup and initialize the dispatch parameters with the policies that match the device's PTX version
            int ptx_version;
            if (CubDebug(error = PtxVersion(ptx_version))) break;
            PtxDefaultPolicies::InitDispatchParams(ptx_version, multi_block_dispatch_params, single_block_dispatch_params);

        #endif

            // Dispatch
            if (CubDebug(error = Dispatch(
                MultiBlockReduceKernel<MultiBlockPolicy, InputIteratorRA, T*, SizeT, ReductionOp>,
                SingleBlockReduceKernel<SingleBlockPolicy, T*, OutputIteratorRA, SizeT, ReductionOp>,
                multi_block_dispatch_params,
                single_block_dispatch_params,
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


