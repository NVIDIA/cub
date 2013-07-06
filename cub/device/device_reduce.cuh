
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

#include "block/block_reduce_tiles.cuh"
#include "../thread/thread_operators.cuh"
#include "../grid/grid_even_share.cuh"
#include "../grid/grid_queue.cuh"
#include "../util_debug.cuh"
#include "../util_device.cuh"
#include "../util_namespace.cuh"

/// Optional outer namespace(s)
CUB_NS_PREFIX

/// CUB namespace
namespace cub {


/******************************************************************************
 * Kernel entry points
 *****************************************************************************/

#ifndef DOXYGEN_SHOULD_SKIP_THIS    // Do not document


/**
 * Reduction pass kernel entry point (multi-block).  Computes privatized reductions, one per thread block.
 */
template <
    typename                BlockReduceTilesPolicy, ///< Tuning policy for cub::BlockReduceTiles abstraction
    typename                InputIteratorRA,        ///< Random-access iterator type for input (may be a simple pointer type)
    typename                OutputIteratorRA,       ///< Random-access iterator type for output (may be a simple pointer type)
    typename                SizeT,                  ///< Integral type used for global array indexing
    typename                ReductionOp>            ///< Binary reduction operator type having member <tt>T operator()(const T &a, const T &b)</tt>
__launch_bounds__ (int(BlockReduceTilesPolicy::BLOCK_THREADS), 1)
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
    typedef BlockReduceTiles<BlockReduceTilesPolicy, InputIteratorRA, SizeT, ReductionOp> BlockReduceTilesT;

    // Block-wide aggregate
    T block_aggregate;

    // Shared memory storage
    __shared__ typename BlockReduceTilesT::TempStorage temp_storage;

    // Consume input tiles
    BlockReduceTilesT(temp_storage, d_in, reduction_op).ConsumeTiles(
        num_items,
        even_share,
        queue,
        block_aggregate,
        Int2Type<BlockReduceTilesPolicy::GRID_MAPPING>());

    // Output result
    if (threadIdx.x == 0)
    {
        d_out[blockIdx.x] = block_aggregate;
    }
}


/**
 * Reduction pass kernel entry point (single-block).  Aggregates privatized threadblock reductions from a previous multi-block reduction pass.
 */
template <
    typename                BlockReduceTilesPolicy,  ///< Tuning policy for cub::BlockReduceTiles abstraction
    typename                InputIteratorRA,        ///< Random-access iterator type for input (may be a simple pointer type)
    typename                OutputIteratorRA,       ///< Random-access iterator type for output (may be a simple pointer type)
    typename                SizeT,                  ///< Integral type used for global array indexing
    typename                ReductionOp>            ///< Binary reduction operator type having member <tt>T operator()(const T &a, const T &b)</tt>
__launch_bounds__ (int(BlockReduceTilesPolicy::BLOCK_THREADS), 1)
__global__ void SingleBlockReduceKernel(
    InputIteratorRA         d_in,                   ///< [in] Input data to reduce
    OutputIteratorRA        d_out,                  ///< [out] Output location for result
    SizeT                   num_items,              ///< [in] Total number of input data items
    ReductionOp             reduction_op)           ///< [in] Binary reduction operator
{
    // Data type
    typedef typename std::iterator_traits<InputIteratorRA>::value_type T;

    // Thread block type for reducing input tiles
    typedef BlockReduceTiles<BlockReduceTilesPolicy, InputIteratorRA, SizeT, ReductionOp> BlockReduceTilesT;

    // Block-wide aggregate
    T block_aggregate;

    // Shared memory storage
    __shared__ typename BlockReduceTilesT::TempStorage temp_storage;

    // Consume input tiles
    BlockReduceTilesT(temp_storage, d_in, reduction_op).ConsumeTiles(
        SizeT(0),
        SizeT(num_items),
        block_aggregate);

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

    /******************************************************************************
     * Constants and typedefs
     ******************************************************************************/

    /// Generic structure for encapsulating dispatch properties.  Mirrors the constants within BlockReduceTilesPolicy.
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

        template <typename BlockReduceTilesPolicy>
        __host__ __device__ __forceinline__
        void Init(int subscription_factor = 1)
        {
            block_threads               = BlockReduceTilesPolicy::BLOCK_THREADS;
            items_per_thread            = BlockReduceTilesPolicy::ITEMS_PER_THREAD;
            vector_load_length          = BlockReduceTilesPolicy::VECTOR_LOAD_LENGTH;
            block_algorithm             = BlockReduceTilesPolicy::BLOCK_ALGORITHM;
            load_modifier               = BlockReduceTilesPolicy::LOAD_MODIFIER;
            grid_mapping                = BlockReduceTilesPolicy::GRID_MAPPING;
            this->subscription_factor   = subscription_factor;

            tile_size = block_threads * items_per_thread;
        }

        __host__ __device__ __forceinline__
        void Print()
        {
            printf("%d threads, %d per thread, %d veclen, %d algo, %d loadmod, %d mapping, %d subscription",
                block_threads,
                items_per_thread,
                vector_load_length,
                block_algorithm,
                load_modifier,
                grid_mapping,
                subscription_factor);
        }

    };


    /******************************************************************************
     * Tuning policies
     ******************************************************************************/

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
        // 1B Multiblock policy: GTX Titan: 206.0 GB/s @ 192M 1B items
        typedef BlockReduceTilesPolicy<128, 12,  1, BLOCK_REDUCE_RAKING, LOAD_LDG, GRID_MAPPING_DYNAMIC>                MultiBlockPolicy1B;

        // 4B Multiblock policy: GTX Titan: 254.2 GB/s @ 48M 4B items
        typedef BlockReduceTilesPolicy<512, 20,  1, BLOCK_REDUCE_RAKING, LOAD_DEFAULT, GRID_MAPPING_EVEN_SHARE>         MultiBlockPolicy4B;

        // Multiblock policy
        typedef typename If<(sizeof(T) < 4),
            MultiBlockPolicy1B,
            MultiBlockPolicy4B>::Type MultiBlockPolicy;

        // Singleblock policy
        typedef BlockReduceTilesPolicy<256, 8, 1, BLOCK_REDUCE_WARP_REDUCTIONS, LOAD_DEFAULT, GRID_MAPPING_EVEN_SHARE>  SingleBlockPolicy;

        enum { SUBSCRIPTION_FACTOR = 8 };

    };

    /// SM30 tune
    template <typename T, typename SizeT>
    struct TunedPolicies<T, SizeT, 300>
    {
        // GTX670: 154.0 @ 48M 32-bit T
        typedef BlockReduceTilesPolicy<256, 2,  1, BLOCK_REDUCE_WARP_REDUCTIONS,  LOAD_DEFAULT, GRID_MAPPING_EVEN_SHARE>    MultiBlockPolicy;
        typedef BlockReduceTilesPolicy<256, 24, 4, BLOCK_REDUCE_WARP_REDUCTIONS,  LOAD_DEFAULT, GRID_MAPPING_EVEN_SHARE>    SingleBlockPolicy;
        enum { SUBSCRIPTION_FACTOR = 1 };
    };

    /// SM20 tune
    template <typename T, typename SizeT>
    struct TunedPolicies<T, SizeT, 200>
    {
        // 1B Multiblock policy: GTX 580: 158.1 GB/s @ 192M 1B items
        typedef BlockReduceTilesPolicy<192, 24,  4, BLOCK_REDUCE_RAKING, LOAD_DEFAULT, GRID_MAPPING_EVEN_SHARE>            MultiBlockPolicy1B;

        // 4B Multiblock policy: GTX 580: 178.9 GB/s @ 48M 4B items
        typedef BlockReduceTilesPolicy<128, 8,  4, BLOCK_REDUCE_RAKING, LOAD_DEFAULT, GRID_MAPPING_DYNAMIC>                MultiBlockPolicy4B;

        // Multiblock policy
        typedef typename If<(sizeof(T) < 4),
            MultiBlockPolicy1B,
            MultiBlockPolicy4B>::Type MultiBlockPolicy;

        // Singleblock policy
        typedef BlockReduceTilesPolicy<192, 7,  1, BLOCK_REDUCE_RAKING, LOAD_DEFAULT, GRID_MAPPING_EVEN_SHARE>             SingleBlockPolicy;

        enum { SUBSCRIPTION_FACTOR = 2 };
    };

    /// SM13 tune
    template <typename T, typename SizeT>
    struct TunedPolicies<T, SizeT, 130>
    {
        typedef BlockReduceTilesPolicy<128, 8,  2,  BLOCK_REDUCE_RAKING, LOAD_DEFAULT, GRID_MAPPING_EVEN_SHARE>            MultiBlockPolicy;
        typedef BlockReduceTilesPolicy<32,  4,  4,  BLOCK_REDUCE_RAKING, LOAD_DEFAULT, GRID_MAPPING_EVEN_SHARE>            SingleBlockPolicy;
        enum { SUBSCRIPTION_FACTOR = 1 };
    };

    /// SM10 tune
    template <typename T, typename SizeT>
    struct TunedPolicies<T, SizeT, 100>
    {
        typedef BlockReduceTilesPolicy<128, 8,  2,  BLOCK_REDUCE_RAKING, LOAD_DEFAULT, GRID_MAPPING_EVEN_SHARE>            MultiBlockPolicy;
        typedef BlockReduceTilesPolicy<32,  4,  4,  BLOCK_REDUCE_RAKING, LOAD_DEFAULT, GRID_MAPPING_EVEN_SHARE>            SingleBlockPolicy;
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


    /******************************************************************************
     * Utility methods
     ******************************************************************************/


    /**
     * Internal dispatch routine for computing a device-wide reduction using a two-stages of kernel invocations.
     */
    template <
        typename                    MultiBlockReduceKernelPtr,          ///< Function type of cub::MultiBlockReduceKernel
        typename                    ReduceSingleKernelPtr,              ///< Function type of cub::SingleBlockReduceKernel
        typename                    ResetDrainKernelPtr,                ///< Function type of cub::ResetDrainKernel
        typename                    InputIteratorRA,                    ///< Random-access iterator type for input (may be a simple pointer type)
        typename                    OutputIteratorRA,                   ///< Random-access iterator type for output (may be a simple pointer type)
        typename                    SizeT,                              ///< Integral type used for global array indexing
        typename                    ReductionOp>                        ///< Binary reduction operator type having member <tt>T operator()(const T &a, const T &b)</tt>
    __host__ __device__ __forceinline__
    static cudaError_t Dispatch(
        void                        *d_temp_storage,                    ///< [in] %Device allocation of temporary storage.  When NULL, the required allocation size is returned in \p temp_storage_bytes and no work is done.
        size_t                      &temp_storage_bytes,                ///< [in,out] Size in bytes of \p d_temp_storage allocation.
        MultiBlockReduceKernelPtr   multi_block_kernel,                 ///< [in] Kernel function pointer to parameterization of cub::MultiBlockReduceKernel
        ReduceSingleKernelPtr       single_block_kernel,                ///< [in] Kernel function pointer to parameterization of cub::SingleBlockReduceKernel
        ResetDrainKernelPtr         prepare_drain_kernel,               ///< [in] Kernel function pointer to parameterization of cub::ResetDrainKernel
        KernelDispachParams         &multi_block_dispatch_params,       ///< [in] Dispatch parameters that match the policy that \p multi_block_kernel_ptr was compiled for
        KernelDispachParams         &single_block_dispatch_params,      ///< [in] Dispatch parameters that match the policy that \p single_block_kernel was compiled for
        InputIteratorRA             d_in,                               ///< [in] Input data to reduce
        OutputIteratorRA            d_out,                              ///< [out] Output location for result
        SizeT                       num_items,                          ///< [in] Number of items to reduce
        ReductionOp                 reduction_op,                       ///< [in] Binary reduction operator
        cudaStream_t                stream              = 0,            ///< [in] <b>[optional]</b> CUDA stream to launch kernels within.  Default is stream<sub>0</sub>.
        bool                        stream_synchronous  = false)        ///< [in] <b>[optional]</b> Whether or not to synchronize the stream after every kernel launch to check for errors.  Default is \p false.
    {
#ifndef CUB_RUNTIME_ENABLED

        // Kernel launch not supported from this device
        return CubDebug(cudaErrorNotSupported );

#else

        // Data type of input iterator
        typedef typename std::iterator_traits<InputIteratorRA>::value_type T;

        cudaError error = cudaSuccess;
        do
        {
            if ((multi_block_kernel == NULL) || (num_items <= (single_block_dispatch_params.tile_size)))
            {
                // Dispatch a single-block reduction kernel

                // Return if the caller is simply requesting the size of the storage allocation
                if (d_temp_storage == NULL)
                {
                    temp_storage_bytes = 1;
                    return cudaSuccess;
                }

                // Log single_block_kernel configuration
                if (stream_synchronous) CubLog("Invoking ReduceSingle<<<1, %d, 0, %lld>>>(), %d items per thread\n",
                    single_block_dispatch_params.block_threads, (long long) stream, single_block_dispatch_params.items_per_thread);

                // Invoke single_block_kernel
                single_block_kernel<<<1, single_block_dispatch_params.block_threads>>>(
                    d_in,
                    d_out,
                    num_items,
                    reduction_op);

                // Sync the stream if specified
                if (stream_synchronous && (CubDebug(error = SyncStream(stream)))) break;

            }
            else
            {
                // Dispatch two kernels: a multi-block kernel to compute
                // privatized per-block reductions, and then a single-block
                // to reduce those

                // Get device ordinal
                int device_ordinal;
                if (CubDebug(error = cudaGetDevice(&device_ordinal))) break;

                // Get SM count
                int sm_count;
                if (CubDebug(error = cudaDeviceGetAttribute (&sm_count, cudaDevAttrMultiProcessorCount, device_ordinal))) break;

                // Get a rough estimate of multi_block_kernel SM occupancy based upon the maximum SM occupancy of the targeted PTX architecture
                int multi_block_sm_occupancy = CUB_MIN(
                    ArchProps<CUB_PTX_ARCH>::MAX_SM_THREADBLOCKS,
                    ArchProps<CUB_PTX_ARCH>::MAX_SM_THREADS / multi_block_dispatch_params.block_threads);

    #ifndef __CUDA_ARCH__
                // We're on the host, so come up with a more accurate estimate of multi_block_kernel SM occupancy from actual device properties
                Device device_props;
                if (CubDebug(error = device_props.Init(device_ordinal))) break;

                if (CubDebug(error = device_props.MaxSmOccupancy(
                    multi_block_sm_occupancy,
                    multi_block_kernel,
                    multi_block_dispatch_params.block_threads))) break;
    #endif

                // Get device occupancy for multi_block_kernel
                int multi_block_occupancy = multi_block_sm_occupancy * sm_count;

                // Even-share work distribution
                GridEvenShare<SizeT> even_share;

                // Get grid size for multi_block_kernel
                int multi_block_grid_size;
                switch (multi_block_dispatch_params.grid_mapping)
                {
                case GRID_MAPPING_EVEN_SHARE:

                    // Work is distributed evenly
                    even_share.GridInit(
                        num_items,
                        multi_block_occupancy * multi_block_dispatch_params.subscription_factor,
                        multi_block_dispatch_params.tile_size);
                    multi_block_grid_size = even_share.grid_size;
                    break;

                case GRID_MAPPING_DYNAMIC:

                    // Work is distributed dynamically
                    int num_tiles = (num_items + multi_block_dispatch_params.tile_size - 1) / multi_block_dispatch_params.tile_size;
                    multi_block_grid_size   = (num_tiles < multi_block_occupancy) ?
                        num_tiles :                 // Not enough to fill the device with threadblocks
                        multi_block_occupancy;      // Fill the device with threadblocks
                    break;
                };

                // Temporary storage allocation requirements
                void* allocations[2];
                size_t allocation_sizes[2] =
                {
                    multi_block_grid_size * sizeof(T),      // bytes needed for privatized block reductions
                    GridQueue<int>::AllocationSize()        // bytes needed for grid queue descriptor
                };

                // Alias temporaries (or set the necessary size of the storage allocation)
                if (CubDebug(error = AliasTemporaries(d_temp_storage, temp_storage_bytes, allocations, allocation_sizes))) break;

                // Return if the caller is simply requesting the size of the storage allocation
                if (d_temp_storage == NULL)
                    return cudaSuccess;

                // Privatized per-block reductions
                T *d_block_reductions = (T*) allocations[0];

                // Grid queue descriptor
                GridQueue<SizeT> queue(allocations[1]);

                // Prepare the dynamic queue descriptor if necessary
                if (multi_block_dispatch_params.grid_mapping == GRID_MAPPING_DYNAMIC)
                {
                    // Prepare queue using a kernel so we know it gets prepared once per operation
                    if (stream_synchronous) CubLog("Invoking prepare_drain_kernel<<<1, 1, 0, %lld>>>()\n", (long long) stream);

                    // Invoke prepare_drain_kernel
                    prepare_drain_kernel<<<1, 1, 0, stream>>>(queue, num_items);

                    // Sync the stream if specified
                    if (stream_synchronous && (CubDebug(error = SyncStream(stream)))) break;
                }

                // Log multi_block_kernel configuration
                if (stream_synchronous) CubLog("Invoking multi_block_kernel<<<%d, %d, 0, %lld>>>(), %d items per thread, %d SM occupancy\n",
                    multi_block_grid_size, multi_block_dispatch_params.block_threads, (long long) stream, multi_block_dispatch_params.items_per_thread, multi_block_sm_occupancy);

                // Invoke multi_block_kernel
                multi_block_kernel<<<multi_block_grid_size, multi_block_dispatch_params.block_threads, 0, stream>>>(
                    d_in,
                    d_block_reductions,
                    num_items,
                    even_share,
                    queue,
                    reduction_op);

                // Sync the stream if specified
                if (stream_synchronous && (CubDebug(error = SyncStream(stream)))) break;

                // Log single_block_kernel configuration
                if (stream_synchronous) CubLog("Invoking single_block_kernel<<<%d, %d, 0, %lld>>>(), %d items per thread\n",
                    1, single_block_dispatch_params.block_threads, (long long) stream, single_block_dispatch_params.items_per_thread);

                // Invoke single_block_kernel
                single_block_kernel<<<1, single_block_dispatch_params.block_threads, 0, stream>>>(
                    d_block_reductions,
                    d_out,
                    multi_block_grid_size,
                    reduction_op);

                // Sync the stream if specified
                if (stream_synchronous && (CubDebug(error = SyncStream(stream)))) break;
            }
        }
        while (0);

        return error;

#endif // CUB_RUNTIME_ENABLED
    }



#endif // DOXYGEN_SHOULD_SKIP_THIS

    /******************************************************************************
     * Interface
     ******************************************************************************/

    /**
     * \brief Computes a device-wide sum using the addition ('+') operator.
     *
     * \devicestorage
     *
     * \tparam InputIteratorRA      <b>[inferred]</b> Random-access iterator type for input (may be a simple pointer type)
     * \tparam OutputIteratorRA     <b>[inferred]</b> Random-access iterator type for output (may be a simple pointer type)
     */
    template <
        typename                    InputIteratorRA,
        typename                    OutputIteratorRA>
    __host__ __device__ __forceinline__
    static cudaError_t Sum(
        void                        *d_temp_storage,                    ///< [in] %Device allocation of temporary storage.  When NULL, the required allocation size is returned in \p temp_storage_bytes and no work is done.
        size_t                      &temp_storage_bytes,                ///< [in,out] Size in bytes of \p d_temp_storage allocation.
        InputIteratorRA             d_in,                               ///< [in] Input data to reduce
        OutputIteratorRA            d_out,                              ///< [out] Output location for result
        int                         num_items,                          ///< [in] Number of items to reduce
        cudaStream_t                stream              = 0,            ///< [in] <b>[optional]</b> CUDA stream to launch kernels within.  Default is stream<sub>0</sub>.
        bool                        stream_synchronous  = false)        ///< [in] <b>[optional]</b> Whether or not to synchronize the stream after every kernel launch to check for errors.  Default is \p false.
    {
        return Reduce(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items, cub::Sum(), stream, stream_synchronous);
    }


    /**
     * \brief Computes a device-wide reduction using the specified binary \p reduction_op functor.
     *
     * \devicestorage
     *
     * \tparam InputIteratorRA      <b>[inferred]</b> Random-access iterator type for input (may be a simple pointer type)
     * \tparam OutputIteratorRA     <b>[inferred]</b> Random-access iterator type for output (may be a simple pointer type)
     * \tparam ReductionOp          <b>[inferred]</b> Binary reduction operator type having member <tt>T operator()(const T &a, const T &b)</tt>
     */
    template <
        typename                    InputIteratorRA,
        typename                    OutputIteratorRA,
        typename                    ReductionOp>
    __host__ __device__ __forceinline__
    static cudaError_t Reduce(
        void                        *d_temp_storage,                    ///< [in] %Device allocation of temporary storage.  When NULL, the required allocation size is returned in \p temp_storage_bytes and no work is done.
        size_t                      &temp_storage_bytes,                ///< [in,out] Size in bytes of \p d_temp_storage allocation.
        InputIteratorRA             d_in,                               ///< [in] Input data to reduce
        OutputIteratorRA            d_out,                              ///< [out] Output location for result
        int                         num_items,                          ///< [in] Number of items to reduce
        ReductionOp                 reduction_op,                       ///< [in] Binary reduction operator
        cudaStream_t                stream              = 0,            ///< [in] <b>[optional]</b> CUDA stream to launch kernels within.  Default is stream<sub>0</sub>.
        bool                        stream_synchronous  = false)        ///< [in] <b>[optional]</b> Whether or not to synchronize the stream after every kernel launch to check for errors.  Default is \p false.
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
                d_temp_storage,
                temp_storage_bytes,
                MultiBlockReduceKernel<MultiBlockPolicy, InputIteratorRA, T*, SizeT, ReductionOp>,
                SingleBlockReduceKernel<SingleBlockPolicy, T*, OutputIteratorRA, SizeT, ReductionOp>,
                ResetDrainKernel<SizeT>,
                multi_block_dispatch_params,
                single_block_dispatch_params,
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


