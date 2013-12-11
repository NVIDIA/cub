
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
 * cub::DeviceReduce provides device-wide, parallel operations for computing a reduction across a sequence of data items residing within global memory.
 */

#pragma once

#include <stdio.h>
#include <iterator>

#include "region/block_reduce_region.cuh"
#include "../thread/thread_operators.cuh"
#include "../grid/grid_even_share.cuh"
#include "../grid/grid_queue.cuh"
#include "../iterator/arg_index_input_iterator.cuh"
#include "../util_debug.cuh"
#include "../util_device.cuh"
#include "../util_namespace.cuh"

/// Optional outer namespace(s)
CUB_NS_PREFIX

/// CUB namespace
namespace cub {

#ifndef DOXYGEN_SHOULD_SKIP_THIS    // Do not document






/******************************************************************************
 * Kernel entry points
 *****************************************************************************/

/**
 * Reduce region kernel entry point (multi-block).  Computes privatized reductions, one per thread block.
 */
template <
    typename                BlockReduceRegionPolicy,    ///< Parameterized BlockReduceRegionPolicy tuning policy type
    typename                InputIterator,              ///< Random-access iterator type for input \iterator
    typename                OutputIterator,             ///< Random-access iterator type for output \iterator
    typename                Offset,                     ///< Signed integer type for global offsets
    typename                ReductionOp>                ///< Binary reduction operator type having member <tt>T operator()(const T &a, const T &b)</tt>
__launch_bounds__ (int(BlockReduceRegionPolicy::BLOCK_THREADS), 1)
__global__ void ReduceRegionKernel(
    InputIterator           d_in,                       ///< [in] Input data to reduce
    OutputIterator          d_out,                      ///< [out] Output location for result
    Offset                  num_items,                  ///< [in] Total number of input data items
    GridEvenShare<Offset>   even_share,                 ///< [in] Even-share descriptor for mapping an equal number of tiles onto each thread block
    GridQueue<Offset>       queue,                      ///< [in] Drain queue descriptor for dynamically mapping tile data onto thread blocks
    ReductionOp             reduction_op)               ///< [in] Binary reduction operator
{
    // Data type
    typedef typename std::iterator_traits<InputIterator>::value_type T;

    // Thread block type for reducing input tiles
    typedef BlockReduceRegion<BlockReduceRegionPolicy, InputIterator, Offset, ReductionOp> BlockReduceRegionT;

    // Block-wide aggregate
    T block_aggregate;

    // Shared memory storage
    __shared__ typename BlockReduceRegionT::TempStorage temp_storage;

    // Consume input tiles
    BlockReduceRegionT(temp_storage, d_in, reduction_op).ConsumeRegion(
        num_items,
        even_share,
        queue,
        block_aggregate,
        Int2Type<BlockReduceRegionPolicy::GRID_MAPPING>());

    // Output result
    if (threadIdx.x == 0)
    {
        d_out[blockIdx.x] = block_aggregate;
    }
}


/**
 * Reduce a single tile kernel entry point (single-block).  Can be used to aggregate privatized threadblock reductions from a previous multi-block reduction pass.
 */
template <
    typename                BlockReduceRegionPolicy,    ///< Parameterized BlockReduceRegionPolicy tuning policy type
    typename                InputIterator,              ///< Random-access iterator type for input \iterator
    typename                OutputIterator,             ///< Random-access iterator type for output \iterator
    typename                Offset,                     ///< Signed integer type for global offsets
    typename                ReductionOp>                ///< Binary reduction operator type having member <tt>T operator()(const T &a, const T &b)</tt>
__launch_bounds__ (int(BlockReduceRegionPolicy::BLOCK_THREADS), 1)
__global__ void SingleTileKernel(
    InputIterator           d_in,                       ///< [in] Input data to reduce
    OutputIterator          d_out,                      ///< [out] Output location for result
    Offset                  num_items,                  ///< [in] Total number of input data items
    ReductionOp             reduction_op)               ///< [in] Binary reduction operator
{
    // Data type
    typedef typename std::iterator_traits<InputIterator>::value_type T;

    // Thread block type for reducing input tiles
    typedef BlockReduceRegion<BlockReduceRegionPolicy, InputIterator, Offset, ReductionOp> BlockReduceRegionT;

    // Block-wide aggregate
    T block_aggregate;

    // Shared memory storage
    __shared__ typename BlockReduceRegionT::TempStorage temp_storage;

    // Consume input tiles
    BlockReduceRegionT(temp_storage, d_in, reduction_op).ConsumeRegion(
        Offset(0),
        Offset(num_items),
        block_aggregate);

    // Output result
    if (threadIdx.x == 0)
    {
        d_out[blockIdx.x] = block_aggregate;
    }
}




/******************************************************************************
 * Dispatch
 ******************************************************************************/

/**
 * Utility class for dispatching the appropriately-tuned kernels for DeviceReduce
 */
template <
    typename InputIterator,     ///< Random-access iterator type for input \iterator
    typename OutputIterator,    ///< Random-access iterator type for output \iterator
    typename Offset,            ///< Signed integer type for global offsets
    typename ReductionOp>       ///< Binary reduction operator type having member <tt>T operator()(const T &a, const T &b)</tt>
struct DeviceReduceDispatch
{
    // Data type of input iterator
    typedef typename std::iterator_traits<InputIterator>::value_type T;


    /******************************************************************************
     * Tuning policies
     ******************************************************************************/

    /// SM35
    struct Policy350
    {
        // ReduceRegionPolicy1B (GTX Titan: 228.7 GB/s @ 192M 1B items)
        typedef BlockReduceRegionPolicy<
                128,                                ///< Threads per thread block
                24,                                 ///< Items per thread per tile of input
                4,                                  ///< Number of items per vectorized load
                BLOCK_REDUCE_RAKING,                ///< Cooperative block-wide reduction algorithm to use
                LOAD_LDG,                           ///< Cache load modifier
                GRID_MAPPING_DYNAMIC>               ///< How to map tiles of input onto thread blocks
            ReduceRegionPolicy1B;

        enum {
            NOMINAL_4B_ITEMS_PER_THREAD = 20,
            ITEMS_PER_THREAD            = CUB_MIN(NOMINAL_4B_ITEMS_PER_THREAD, CUB_MAX(1, (NOMINAL_4B_ITEMS_PER_THREAD * 4 / sizeof(T)))),
        };

        // ReduceRegionPolicy4B (GTX Titan: 255.1 GB/s @ 48M 4B items)
        typedef BlockReduceRegionPolicy<
                256,                                ///< Threads per thread block
                ITEMS_PER_THREAD,                   ///< Items per thread per tile of input
                2,                                  ///< Number of items per vectorized load
                BLOCK_REDUCE_RAKING,                ///< Cooperative block-wide reduction algorithm to use
                LOAD_LDG,                       ///< Cache load modifier
                GRID_MAPPING_DYNAMIC>            ///< How to map tiles of input onto thread blocks
            ReduceRegionPolicy4B;

        // ReduceRegionPolicy
        typedef typename If<(sizeof(T) >= 4),
            ReduceRegionPolicy4B,
            ReduceRegionPolicy1B>::Type ReduceRegionPolicy;

        // SingleTilePolicy
        typedef BlockReduceRegionPolicy<
                256,                                ///< Threads per thread block
                8,                                  ///< Items per thread per tile of input
                1,                                  ///< Number of items per vectorized load
                BLOCK_REDUCE_WARP_REDUCTIONS,       ///< Cooperative block-wide reduction algorithm to use
                LOAD_DEFAULT,                       ///< Cache load modifier
                GRID_MAPPING_EVEN_SHARE>            ///< How to map tiles of input onto thread blocks
            SingleTilePolicy;
    };

    /// SM30
    struct Policy300
    {
        enum {
            NOMINAL_4B_ITEMS_PER_THREAD = 2,
            ITEMS_PER_THREAD            = CUB_MIN(NOMINAL_4B_ITEMS_PER_THREAD, CUB_MAX(1, (NOMINAL_4B_ITEMS_PER_THREAD * 4 / sizeof(T)))),
        };

        // ReduceRegionPolicy (GTX670: 154.0 @ 48M 4B items)
        typedef BlockReduceRegionPolicy<
                256,                                ///< Threads per thread block
                ITEMS_PER_THREAD,                   ///< Items per thread per tile of input
                1,                                  ///< Number of items per vectorized load
                BLOCK_REDUCE_WARP_REDUCTIONS,       ///< Cooperative block-wide reduction algorithm to use
                LOAD_DEFAULT,                       ///< Cache load modifier
                GRID_MAPPING_EVEN_SHARE>            ///< How to map tiles of input onto thread blocks
            ReduceRegionPolicy;

        // SingleTilePolicy
        typedef BlockReduceRegionPolicy<
                256,                                ///< Threads per thread block
                24,                                 ///< Items per thread per tile of input
                4,                                  ///< Number of items per vectorized load
                BLOCK_REDUCE_WARP_REDUCTIONS,       ///< Cooperative block-wide reduction algorithm to use
                LOAD_DEFAULT,                       ///< Cache load modifier
                GRID_MAPPING_EVEN_SHARE>            ///< How to map tiles of input onto thread blocks
            SingleTilePolicy;
    };

    /// SM20
    struct Policy200
    {
        // ReduceRegionPolicy1B (GTX 580: 158.1 GB/s @ 192M 1B items)
        typedef BlockReduceRegionPolicy<
                192,                                ///< Threads per thread block
                24,                                 ///< Items per thread per tile of input
                4,                                  ///< Number of items per vectorized load
                BLOCK_REDUCE_RAKING,                ///< Cooperative block-wide reduction algorithm to use
                LOAD_DEFAULT,                       ///< Cache load modifier
                GRID_MAPPING_EVEN_SHARE>            ///< How to map tiles of input onto thread blocks
            ReduceRegionPolicy1B;

        enum {
            NOMINAL_4B_ITEMS_PER_THREAD = 8,
            ITEMS_PER_THREAD            = CUB_MIN(NOMINAL_4B_ITEMS_PER_THREAD, CUB_MAX(1, (NOMINAL_4B_ITEMS_PER_THREAD * 4 / sizeof(T)))),
        };

        // ReduceRegionPolicy4B (GTX 580: 178.9 GB/s @ 48M 4B items)
        typedef BlockReduceRegionPolicy<
                128,                                ///< Threads per thread block
                ITEMS_PER_THREAD,                   ///< Items per thread per tile of input
                4,                                  ///< Number of items per vectorized load
                BLOCK_REDUCE_RAKING,                ///< Cooperative block-wide reduction algorithm to use
                LOAD_DEFAULT,                       ///< Cache load modifier
                GRID_MAPPING_DYNAMIC>               ///< How to map tiles of input onto thread blocks
            ReduceRegionPolicy4B;

        // ReduceRegionPolicy
        typedef typename If<(sizeof(T) < 4),
            ReduceRegionPolicy1B,
            ReduceRegionPolicy4B>::Type ReduceRegionPolicy;

        // SingleTilePolicy
        typedef BlockReduceRegionPolicy<
                192,                                ///< Threads per thread block
                7,                                  ///< Items per thread per tile of input
                1,                                  ///< Number of items per vectorized load
                BLOCK_REDUCE_RAKING,                ///< Cooperative block-wide reduction algorithm to use
                LOAD_DEFAULT,                       ///< Cache load modifier
                GRID_MAPPING_EVEN_SHARE>            ///< How to map tiles of input onto thread blocks
            SingleTilePolicy;
    };

    /// SM13
    struct Policy130
    {
        enum {
            NOMINAL_4B_ITEMS_PER_THREAD = 8,
            ITEMS_PER_THREAD            = CUB_MIN(NOMINAL_4B_ITEMS_PER_THREAD, CUB_MAX(1, (NOMINAL_4B_ITEMS_PER_THREAD * 4 / sizeof(T)))),
        };

        // ReduceRegionPolicy
        typedef BlockReduceRegionPolicy<
                128,                                ///< Threads per thread block
                ITEMS_PER_THREAD,                   ///< Items per thread per tile of input
                2,                                  ///< Number of items per vectorized load
                BLOCK_REDUCE_RAKING,                ///< Cooperative block-wide reduction algorithm to use
                LOAD_DEFAULT,                       ///< Cache load modifier
                GRID_MAPPING_EVEN_SHARE>            ///< How to map tiles of input onto thread blocks
            ReduceRegionPolicy;

        // SingleTilePolicy
        typedef BlockReduceRegionPolicy<
                32,                                 ///< Threads per thread block
                4,                                  ///< Items per thread per tile of input
                4,                                  ///< Number of items per vectorized load
                BLOCK_REDUCE_RAKING,                ///< Cooperative block-wide reduction algorithm to use
                LOAD_DEFAULT,                       ///< Cache load modifier
                GRID_MAPPING_EVEN_SHARE>            ///< How to map tiles of input onto thread blocks
            SingleTilePolicy;
    };

    /// SM10
    struct Policy100
    {
        enum {
            NOMINAL_4B_ITEMS_PER_THREAD = 8,
            ITEMS_PER_THREAD            = CUB_MIN(NOMINAL_4B_ITEMS_PER_THREAD, CUB_MAX(1, (NOMINAL_4B_ITEMS_PER_THREAD * 4 / sizeof(T)))),
        };

        // ReduceRegionPolicy
        typedef BlockReduceRegionPolicy<
                128,                                ///< Threads per thread block
                ITEMS_PER_THREAD,                   ///< Items per thread per tile of input
                2,                                  ///< Number of items per vectorized load
                BLOCK_REDUCE_RAKING,                ///< Cooperative block-wide reduction algorithm to use
                LOAD_DEFAULT,                       ///< Cache load modifier
                GRID_MAPPING_EVEN_SHARE>            ///< How to map tiles of input onto thread blocks
            ReduceRegionPolicy;

        // SingleTilePolicy
        typedef BlockReduceRegionPolicy<
                32,                                 ///< Threads per thread block
                4,                                  ///< Items per thread per tile of input
                4,                                  ///< Number of items per vectorized load
                BLOCK_REDUCE_RAKING,                ///< Cooperative block-wide reduction algorithm to use
                LOAD_DEFAULT,                       ///< Cache load modifier
                GRID_MAPPING_EVEN_SHARE>            ///< How to map tiles of input onto thread blocks
            SingleTilePolicy;
    };


    /******************************************************************************
     * Tuning policies of current PTX compiler pass
     ******************************************************************************/

#if (CUB_PTX_VERSION >= 350)
    typedef Policy350 PtxPolicy;

#elif (CUB_PTX_VERSION >= 300)
    typedef Policy300 PtxPolicy;

#elif (CUB_PTX_VERSION >= 200)
    typedef Policy200 PtxPolicy;

#elif (CUB_PTX_VERSION >= 130)
    typedef Policy130 PtxPolicy;

#else
    typedef Policy100 PtxPolicy;

#endif

    // "Opaque" policies (whose parameterizations aren't reflected in the type signature)
    struct PtxReduceRegionPolicy   : PtxPolicy::ReduceRegionPolicy {};
    struct PtxSingleTilePolicy     : PtxPolicy::SingleTilePolicy {};


    /******************************************************************************
     * Utilities
     ******************************************************************************/

    /**
     * Initialize kernel dispatch configurations with the policies corresponding to the PTX assembly we will use
     */
    template <typename KernelConfig>
    __host__ __device__ __forceinline__
    static void InitConfigs(
        int             ptx_version,
        KernelConfig    &reduce_region_config,
        KernelConfig    &single_tile_config)
    {
    #ifdef __CUDA_ARCH__

        // We're on the device, so initialize the kernel dispatch configurations with the current PTX policy
        reduce_region_config.template Init<PtxReduceRegionPolicy>();
        single_tile_config.template Init<PtxSingleTilePolicy>();

    #else

        // We're on the host, so lookup and initialize the kernel dispatch configurations with the policies that match the device's PTX version
        if (ptx_version >= 350)
        {
            reduce_region_config.template    Init<typename Policy350::ReduceRegionPolicy>();
            single_tile_config.template     Init<typename Policy350::SingleTilePolicy>();
        }
        else if (ptx_version >= 300)
        {
            reduce_region_config.template    Init<typename Policy300::ReduceRegionPolicy>();
            single_tile_config.template     Init<typename Policy300::SingleTilePolicy>();
        }
        else if (ptx_version >= 200)
        {
            reduce_region_config.template    Init<typename Policy200::ReduceRegionPolicy>();
            single_tile_config.template     Init<typename Policy200::SingleTilePolicy>();
        }
        else if (ptx_version >= 130)
        {
            reduce_region_config.template    Init<typename Policy130::ReduceRegionPolicy>();
            single_tile_config.template     Init<typename Policy130::SingleTilePolicy>();
        }
        else
        {
            reduce_region_config.template    Init<typename Policy100::ReduceRegionPolicy>();
            single_tile_config.template     Init<typename Policy100::SingleTilePolicy>();
        }

    #endif
    }


    /**
     * Kernel kernel dispatch configuration
     */
    struct KernelConfig
    {
        int                     block_threads;
        int                     items_per_thread;
        int                     vector_load_length;
        BlockReduceAlgorithm    block_algorithm;
        CacheLoadModifier       load_modifier;
        GridMappingStrategy     grid_mapping;

        template <typename BlockPolicy>
        __host__ __device__ __forceinline__
        void Init()
        {
            block_threads               = BlockPolicy::BLOCK_THREADS;
            items_per_thread            = BlockPolicy::ITEMS_PER_THREAD;
            vector_load_length          = BlockPolicy::VECTOR_LOAD_LENGTH;
            block_algorithm             = BlockPolicy::BLOCK_ALGORITHM;
            load_modifier               = BlockPolicy::LOAD_MODIFIER;
            grid_mapping                = BlockPolicy::GRID_MAPPING;
        }

        __host__ __device__ __forceinline__
        void Print()
        {
            printf("%d threads, %d per thread, %d veclen, %d algo, %d loadmod, %d mapping",
                block_threads,
                items_per_thread,
                vector_load_length,
                block_algorithm,
                load_modifier,
                grid_mapping);
        }
    };

    /******************************************************************************
     * Dispatch entrypoints
     ******************************************************************************/

    /**
     * Internal dispatch routine for computing a device-wide reduction using the
     * specified kernel functions.
     *
     * If the input is larger than a single tile, this method uses two-passes of
     * kernel invocations.
     */
    template <
        typename                    ReduceRegionKernelPtr,              ///< Function type of cub::ReduceRegionKernel
        typename                    AggregateTileKernelPtr,             ///< Function type of cub::SingleTileKernel for consuming partial reductions (T*)
        typename                    SingleTileKernelPtr,                ///< Function type of cub::SingleTileKernel for consuming input (InputIterator)
        typename                    FillAndResetDrainKernelPtr>         ///< Function type of cub::FillAndResetDrainKernel
    __host__ __device__ __forceinline__
    static cudaError_t Dispatch(
        void                        *d_temp_storage,                    ///< [in] %Device allocation of temporary storage.  When NULL, the required allocation size is returned in \p temp_storage_bytes and no work is done.
        size_t                      &temp_storage_bytes,                ///< [in,out] Size in bytes of \p d_temp_storage allocation
        InputIterator               d_in,                               ///< [in] Input data to reduce
        OutputIterator              d_out,                              ///< [out] Output location for result
        Offset                      num_items,                          ///< [in] Number of items to reduce
        ReductionOp                 reduction_op,                       ///< [in] Binary reduction operator
        cudaStream_t                stream,                             ///< [in] <b>[optional]</b> CUDA stream to launch kernels within.  Default is stream<sub>0</sub>.
        bool                        debug_synchronous,                  ///< [in] <b>[optional]</b> Whether or not to synchronize the stream after every kernel launch to check for errors.  Also causes launch configurations to be printed to the console.  Default is \p false.
        FillAndResetDrainKernelPtr  prepare_drain_kernel,               ///< [in] Kernel function pointer to parameterization of cub::FillAndResetDrainKernel
        ReduceRegionKernelPtr       reduce_region_kernel,               ///< [in] Kernel function pointer to parameterization of cub::ReduceRegionKernel
        AggregateTileKernelPtr      aggregate_kernel,                   ///< [in] Kernel function pointer to parameterization of cub::SingleTileKernel for consuming partial reductions (T*)
        SingleTileKernelPtr         single_kernel,                      ///< [in] Kernel function pointer to parameterization of cub::SingleTileKernel for consuming input (InputIterator)
        KernelConfig                &reduce_region_config,              ///< [in] Dispatch parameters that match the policy that \p reduce_region_kernel_ptr was compiled for
        KernelConfig                &single_tile_config)                ///< [in] Dispatch parameters that match the policy that \p single_kernel was compiled for
    {
#ifndef CUB_RUNTIME_ENABLED

        // Kernel launch not supported from this device
        return CubDebug(cudaErrorNotSupported );

#else
        cudaError error = cudaSuccess;
        do
        {
            // Get device ordinal
            int device_ordinal;
            if (CubDebug(error = cudaGetDevice(&device_ordinal))) break;

            // Get device SM version
            int sm_version;
            if (CubDebug(error = SmVersion(sm_version, device_ordinal))) break;

            // Get SM count
            int sm_count;
            if (CubDebug(error = cudaDeviceGetAttribute (&sm_count, cudaDevAttrMultiProcessorCount, device_ordinal))) break;

            // Tile size of reduce_region_kernel
            int tile_size = reduce_region_config.block_threads * reduce_region_config.items_per_thread;

            if ((reduce_region_kernel == NULL) || (num_items <= tile_size))
            {
                // Dispatch a single-block reduction kernel

                // Return if the caller is simply requesting the size of the storage allocation
                if (d_temp_storage == NULL)
                {
                    temp_storage_bytes = 1;
                    return cudaSuccess;
                }

                // Log single_kernel configuration
                if (debug_synchronous) CubLog("Invoking ReduceSingle<<<1, %d, 0, %lld>>>(), %d items per thread\n",
                    single_tile_config.block_threads, (long long) stream, single_tile_config.items_per_thread);

                // Invoke single_kernel
                single_kernel<<<1, single_tile_config.block_threads>>>(
                    d_in,
                    d_out,
                    num_items,
                    reduction_op);

                // Sync the stream if specified
                if (debug_synchronous && (CubDebug(error = SyncStream(stream)))) break;

            }
            else
            {
                // Dispatch two kernels: (1) a multi-block kernel to compute
                // privatized per-block reductions, and (2) a single-block
                // to reduce those partial reductions

                // Get SM occupancy for reduce_region_kernel
                int reduce_region_sm_occupancy;
                if (CubDebug(error = MaxSmOccupancy(
                    reduce_region_sm_occupancy,
                    sm_version,
                    reduce_region_kernel,
                    reduce_region_config.block_threads))) break;

                // Get device occupancy for reduce_region_kernel
                int reduce_region_occupancy = reduce_region_sm_occupancy * sm_count;

                // Even-share work distribution
                int subscription_factor = reduce_region_sm_occupancy;     // Amount of CTAs to oversubscribe the device beyond actively-resident (heuristic)
                GridEvenShare<Offset> even_share(
                    num_items,
                    reduce_region_occupancy * subscription_factor,
                    tile_size);

                // Get grid size for reduce_region_kernel
                int reduce_region_grid_size;
                switch (reduce_region_config.grid_mapping)
                {
                case GRID_MAPPING_EVEN_SHARE:

                    // Work is distributed evenly
                    reduce_region_grid_size = even_share.grid_size;
                    break;

                case GRID_MAPPING_DYNAMIC:

                    // Work is distributed dynamically
                    int num_tiles = (num_items + tile_size - 1) / tile_size;
                    reduce_region_grid_size = (num_tiles < reduce_region_occupancy) ?
                        num_tiles :                     // Not enough to fill the device with threadblocks
                        reduce_region_occupancy;         // Fill the device with threadblocks
                    break;
                };

                // Temporary storage allocation requirements
                void* allocations[2];
                size_t allocation_sizes[2] =
                {
                    reduce_region_grid_size * sizeof(T),     // bytes needed for privatized block reductions
                    GridQueue<int>::AllocationSize()        // bytes needed for grid queue descriptor
                };

                // Alias the temporary allocations from the single storage blob (or set the necessary size of the blob)
                if (CubDebug(error = AliasTemporaries(d_temp_storage, temp_storage_bytes, allocations, allocation_sizes))) break;
                if (d_temp_storage == NULL)
                {
                    // Return if the caller is simply requesting the size of the storage allocation
                    return cudaSuccess;
                }

                // Alias the allocation for the privatized per-block reductions
                T *d_block_reductions = (T*) allocations[0];

                // Alias the allocation for the grid queue descriptor
                GridQueue<Offset> queue(allocations[1]);

                // Prepare the dynamic queue descriptor if necessary
                if (reduce_region_config.grid_mapping == GRID_MAPPING_DYNAMIC)
                {
                    // Prepare queue using a kernel so we know it gets prepared once per operation
                    if (debug_synchronous) CubLog("Invoking prepare_drain_kernel<<<1, 1, 0, %lld>>>()\n", (long long) stream);

                    // Invoke prepare_drain_kernel
                    prepare_drain_kernel<<<1, 1, 0, stream>>>(queue, num_items);

                    // Sync the stream if specified
                    if (debug_synchronous && (CubDebug(error = SyncStream(stream)))) break;
                }

                // Log reduce_region_kernel configuration
                if (debug_synchronous) CubLog("Invoking reduce_region_kernel<<<%d, %d, 0, %lld>>>(), %d items per thread, %d SM occupancy\n",
                    reduce_region_grid_size, reduce_region_config.block_threads, (long long) stream, reduce_region_config.items_per_thread, reduce_region_sm_occupancy);

                // Invoke reduce_region_kernel
                reduce_region_kernel<<<reduce_region_grid_size, reduce_region_config.block_threads, 0, stream>>>(
                    d_in,
                    d_block_reductions,
                    num_items,
                    even_share,
                    queue,
                    reduction_op);

                // Sync the stream if specified
                if (debug_synchronous && (CubDebug(error = SyncStream(stream)))) break;

                // Log single_kernel configuration
                if (debug_synchronous) CubLog("Invoking single_kernel<<<%d, %d, 0, %lld>>>(), %d items per thread\n",
                    1, single_tile_config.block_threads, (long long) stream, single_tile_config.items_per_thread);

                // Invoke single_kernel
                aggregate_kernel<<<1, single_tile_config.block_threads, 0, stream>>>(
                    d_block_reductions,
                    d_out,
                    reduce_region_grid_size,
                    reduction_op);

                // Sync the stream if specified
                if (debug_synchronous && (CubDebug(error = SyncStream(stream)))) break;
            }
        }
        while (0);

        return error;

#endif // CUB_RUNTIME_ENABLED
    }


    /**
     * Internal dispatch routine for computing a device-wide reduction
     */
    __host__ __device__ __forceinline__
    static cudaError_t Dispatch(
        void                        *d_temp_storage,                    ///< [in] %Device allocation of temporary storage.  When NULL, the required allocation size is returned in \p temp_storage_bytes and no work is done.
        size_t                      &temp_storage_bytes,                ///< [in,out] Size in bytes of \p d_temp_storage allocation
        InputIterator               d_in,                               ///< [in] Input data to reduce
        OutputIterator              d_out,                              ///< [out] Output location for result
        Offset                      num_items,                          ///< [in] Number of items to reduce
        ReductionOp                 reduction_op,                       ///< [in] Binary reduction operator
        cudaStream_t                stream,                             ///< [in] <b>[optional]</b> CUDA stream to launch kernels within.  Default is stream<sub>0</sub>.
        bool                        debug_synchronous)                  ///< [in] <b>[optional]</b> Whether or not to synchronize the stream after every kernel launch to check for errors.  Also causes launch configurations to be printed to the console.  Default is \p false.
    {
        cudaError error = cudaSuccess;
        do
        {
            // Get PTX version
            int ptx_version;
    #ifndef __CUDA_ARCH__
            if (CubDebug(error = PtxVersion(ptx_version))) break;
    #else
            ptx_version = CUB_PTX_VERSION;
    #endif

            // Get kernel kernel dispatch configurations
            KernelConfig reduce_region_config;
            KernelConfig single_tile_config;
            InitConfigs(ptx_version, reduce_region_config, single_tile_config);

            // Dispatch
            if (CubDebug(error = Dispatch(
                d_temp_storage,
                temp_storage_bytes,
                d_in,
                d_out,
                num_items,
                reduction_op,
                stream,
                debug_synchronous,
                FillAndResetDrainKernel<Offset>,
                ReduceRegionKernel<PtxReduceRegionPolicy, InputIterator, T*, Offset, ReductionOp>,
                SingleTileKernel<PtxSingleTilePolicy, T*, OutputIterator, Offset, ReductionOp>,
                SingleTileKernel<PtxSingleTilePolicy, InputIterator, OutputIterator, Offset, ReductionOp>,
                reduce_region_config,
                single_tile_config))) break;
        }
        while (0);

        return error;
    }
};


#endif // DOXYGEN_SHOULD_SKIP_THIS



/******************************************************************************
 * DeviceReduce
 *****************************************************************************/

/**
 * \brief DeviceReduce provides device-wide, parallel operations for computing a reduction across a sequence of data items residing within global memory. ![](reduce_logo.png)
 * \ingroup DeviceModule
 *
 * \par Overview
 * A <a href="http://en.wikipedia.org/wiki/Reduce_(higher-order_function)"><em>reduction</em></a> (or <em>fold</em>)
 * uses a binary combining operator to compute a single aggregate from a sequence of input elements.
 *
 * \par Usage Considerations
 * \cdp_class{DeviceReduce}
 *
 * \par Performance
 *
 * \image html reduction_perf.png
 *
 */
struct DeviceReduce
{
    /**
     * \brief Computes a device-wide reduction using the specified binary \p reduction_op functor.
     *
     * \par
     * - Does not support non-commutative reduction operators.
     * - \devicestorage
     * - \cdp
     *
     * \par
     * The code snippet below illustrates a custom min reduction of a device vector of \p int items.
     * \par
     * \code
     * #include <cub/cub.cuh>   // or equivalently <cub/device/device_radix_sort.cuh>
     *
     * // MyMin functor
     * struct MyMin
     * {
     *     template <typename T>
     *     __host__ __device__ __forceinline__
     *     T operator()(const T &a, const T &b) const {
     *         return (b < a) ? b : a;
     *     }
     * };
     *
     * // Declare, allocate, and initialize device pointers for input and output
     * int      num_items;      // e.g., 7
     * int      *d_in;          // e.g., [8, 6, 7, 5, 3, 0, 9]
     * int      *d_out;         // e.g., [ ]
     * MyMin    min_op;
     * ...
     *
     * // Determine temporary device storage requirements
     * void     *d_temp_storage = NULL;
     * size_t   temp_storage_bytes = 0;
     * cub::DeviceReduce::Reduce(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items, min_op);
     *
     * // Allocate temporary storage
     * cudaMalloc(&d_temp_storage, temp_storage_bytes);
     *
     * // Run reduction
     * cub::DeviceReduce::Reduce(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items, min_op);
     *
     * // d_out <-- [0]
     *
     * \endcode
     *
     * \tparam InputIterator      <b>[inferred]</b> Random-access iterator type for input \iterator
     * \tparam OutputIterator     <b>[inferred]</b> Random-access iterator type for output \iterator
     * \tparam ReductionOp        <b>[inferred]</b> Binary reduction operator type having member <tt>T operator()(const T &a, const T &b)</tt>
     */
    template <
        typename                    InputIterator,
        typename                    OutputIterator,
        typename                    ReductionOp>
    __host__ __device__
    static cudaError_t Reduce(
        void                        *d_temp_storage,                    ///< [in] %Device allocation of temporary storage.  When NULL, the required allocation size is returned in \p temp_storage_bytes and no work is done.
        size_t                      &temp_storage_bytes,                ///< [in,out] Size in bytes of \p d_temp_storage allocation
        InputIterator               d_in,                               ///< [in] Input data to reduce
        OutputIterator              d_out,                              ///< [out] Output location for result
        int                         num_items,                          ///< [in] Number of items to reduce
        ReductionOp                 reduction_op,                       ///< [in] Binary reduction operator
        cudaStream_t                stream              = 0,            ///< [in] <b>[optional]</b> CUDA stream to launch kernels within.  Default is stream<sub>0</sub>.
        bool                        debug_synchronous   = false)        ///< [in] <b>[optional]</b> Whether or not to synchronize the stream after every kernel launch to check for errors.  Also causes launch configurations to be printed to the console.  Default is \p false.
    {
        // Signed integer type for global offsets
        typedef int Offset;

        // Dispatch type
        typedef DeviceReduceDispatch<InputIterator, OutputIterator, Offset, ReductionOp> DeviceReduceDispatch;

        return DeviceReduceDispatch::Dispatch(
            d_temp_storage,
            temp_storage_bytes,
            d_in,
            d_out,
            num_items,
            reduction_op,
            stream,
            debug_synchronous);
    }


    /**
     * \brief Computes a device-wide sum using the addition ('+') operator.
     *
     * \par
     * - Does not support non-commutative reduction operators.
     * - \devicestorage
     * - \cdp
     *
     * \par
     * The code snippet below illustrates the sum reduction of a device vector of \p int items.
     * \par
     * \code
     * #include <cub/cub.cuh>   // or equivalently <cub/device/device_radix_sort.cuh>
     *
     * // Declare, allocate, and initialize device pointers for input and output
     * int  num_items;      // e.g., 7
     * int  *d_in;          // e.g., [8, 6, 7, 5, 3, 0, 9]
     * int  *d_out;         // e.g., [ ]
     * ...
     *
     * // Determine temporary device storage requirements
     * void     *d_temp_storage = NULL;
     * size_t   temp_storage_bytes = 0;
     * cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_sum, num_items);
     *
     * // Allocate temporary storage
     * cudaMalloc(&d_temp_storage, temp_storage_bytes);
     *
     * // Run sum-reduction
     * cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_sum, num_items);
     *
     * // d_out <-- [38]
     *
     * \endcode
     *
     * \tparam InputIterator      <b>[inferred]</b> Random-access iterator type for input \iterator
     * \tparam OutputIterator     <b>[inferred]</b> Random-access iterator type for output \iterator
     */
    template <
        typename                    InputIterator,
        typename                    OutputIterator>
    __host__ __device__
    static cudaError_t Sum(
        void                        *d_temp_storage,                    ///< [in] %Device allocation of temporary storage.  When NULL, the required allocation size is returned in \p temp_storage_bytes and no work is done.
        size_t                      &temp_storage_bytes,                ///< [in,out] Size in bytes of \p d_temp_storage allocation
        InputIterator               d_in,                               ///< [in] Input data to reduce
        OutputIterator              d_out,                              ///< [out] Output location for result
        int                         num_items,                          ///< [in] Number of items to reduce
        cudaStream_t                stream              = 0,            ///< [in] <b>[optional]</b> CUDA stream to launch kernels within.  Default is stream<sub>0</sub>.
        bool                        debug_synchronous   = false)        ///< [in] <b>[optional]</b> Whether or not to synchronize the stream after every kernel launch to check for errors.  Also causes launch configurations to be printed to the console.  Default is \p false.
    {
        // Signed integer type for global offsets
        typedef int Offset;

        // Dispatch type
        typedef DeviceReduceDispatch<InputIterator, OutputIterator, Offset, cub::Sum> DeviceReduceDispatch;

        return DeviceReduceDispatch::Dispatch(
            d_temp_storage,
            temp_storage_bytes,
            d_in,
            d_out,
            num_items,
            cub::Sum(),
            stream,
            debug_synchronous);
    }


    /**
     * \brief Computes a device-wide minimum using the less-than ('<') operator.
     *
     * \par
     * - Does not support non-commutative minimum operators.
     * - \devicestorage
     * - \cdp
     *
     * \par
     * The code snippet below illustrates the min-reduction of a device vector of \p int items.
     * \par
     * \code
     * #include <cub/cub.cuh>   // or equivalently <cub/device/device_radix_sort.cuh>
     *
     * // Declare, allocate, and initialize device pointers for input and output
     * int  num_items;      // e.g., 7
     * int  *d_in;          // e.g., [8, 6, 7, 5, 3, 0, 9]
     * int  *d_out;         // e.g., [ ]
     * ...
     *
     * // Determine temporary device storage requirements
     * void     *d_temp_storage = NULL;
     * size_t   temp_storage_bytes = 0;
     * cub::DeviceReduce::Min(d_temp_storage, temp_storage_bytes, d_in, d_min, num_items);
     *
     * // Allocate temporary storage
     * cudaMalloc(&d_temp_storage, temp_storage_bytes);
     *
     * // Run min-reduction
     * cub::DeviceReduce::Min(d_temp_storage, temp_storage_bytes, d_in, d_min, num_items);
     *
     * // d_out <-- [0]
     *
     * \endcode
     *
     * \tparam InputIterator      <b>[inferred]</b> Random-access iterator type for input \iterator
     * \tparam OutputIterator     <b>[inferred]</b> Random-access iterator type for output \iterator
     */
    template <
        typename                    InputIterator,
        typename                    OutputIterator>
    __host__ __device__
    static cudaError_t Min(
        void                        *d_temp_storage,                    ///< [in] %Device allocation of temporary storage.  When NULL, the required allocation size is returned in \p temp_storage_bytes and no work is done.
        size_t                      &temp_storage_bytes,                ///< [in,out] Size in bytes of \p d_temp_storage allocation
        InputIterator               d_in,                               ///< [in] Input data to reduce
        OutputIterator              d_out,                              ///< [out] Output location for result
        int                         num_items,                          ///< [in] Number of items to reduce
        cudaStream_t                stream              = 0,            ///< [in] <b>[optional]</b> CUDA stream to launch kernels within.  Default is stream<sub>0</sub>.
        bool                        debug_synchronous   = false)        ///< [in] <b>[optional]</b> Whether or not to synchronize the stream after every kernel launch to check for errors.  Also causes launch configurations to be printed to the console.  Default is \p false.
    {
        // Signed integer type for global offsets
        typedef int Offset;

        // Dispatch type
        typedef DeviceReduceDispatch<InputIterator, OutputIterator, Offset, cub::Min> DeviceReduceDispatch;

        return DeviceReduceDispatch::Dispatch(
            d_temp_storage,
            temp_storage_bytes,
            d_in,
            d_out,
            num_items,
            cub::Min(),
            stream,
            debug_synchronous);
    }


    /**
     * \brief Finds the first device-wide minimum using the less-than ('<') operator, also returning the index of that item.
     *
     * \par
     * Assuming the input \p d_in has value type \p T, the output \p d_out must have value type
     * <tt>ItemOffsetPair<T, int></tt>.  The minimum value is returned in <tt>d_out.value</tt> and its
     * location in the input array is returned in <tt>d_out.offset</tt>.
     *
     * \par
     * - Does not support non-commutative minimum operators.
     * - \devicestorage
     * - \cdp
     *
     * \par
     * The code snippet below illustrates the argmin-reduction of a device vector of \p int items.
     * \par
     * \code
     * #include <cub/cub.cuh>   // or equivalently <cub/device/device_radix_sort.cuh>
     *
     * // Declare, allocate, and initialize device pointers for input and output
     * int                      num_items;      // e.g., 7
     * int                      *d_in;          // e.g., [8, 6, 7, 5, 3, 0, 9]
     * int                      *d_out;         // e.g., [ ]
     * ItemOffsetPair<int, int> *d_argmin;      // e.g., [{ , }]
     * ...
     *
     * // Determine temporary device storage requirements
     * void     *d_temp_storage = NULL;
     * size_t   temp_storage_bytes = 0;
     * cub::DeviceReduce::ArgMin(d_temp_storage, temp_storage_bytes, d_in, d_argmin, num_items);
     *
     * // Allocate temporary storage
     * cudaMalloc(&d_temp_storage, temp_storage_bytes);
     *
     * // Run argmin-reduction
     * cub::DeviceReduce::ArgMin(d_temp_storage, temp_storage_bytes, d_in, d_argmin, num_items);
     *
     * // d_out <-- [{0, 5}]
     *
     * \endcode
     *
     * \tparam InputIterator      <b>[inferred]</b> Random-access iterator type for input having some value type \p T \iterator
     * \tparam OutputIterator     <b>[inferred]</b> Random-access iterator type for output having value type <tt>ItemOffsetPair<T, int></tt> \iterator
     */
    template <
        typename                    InputIterator,
        typename                    OutputIterator>
    __host__ __device__
    static cudaError_t ArgMin(
        void                        *d_temp_storage,                    ///< [in] %Device allocation of temporary storage.  When NULL, the required allocation size is returned in \p temp_storage_bytes and no work is done.
        size_t                      &temp_storage_bytes,                ///< [in,out] Size in bytes of \p d_temp_storage allocation
        InputIterator               d_in,                               ///< [in] Input data to reduce
        OutputIterator              d_out,                              ///< [out] Output location for result
        int                         num_items,                          ///< [in] Number of items to reduce
        cudaStream_t                stream              = 0,            ///< [in] <b>[optional]</b> CUDA stream to launch kernels within.  Default is stream<sub>0</sub>.
        bool                        debug_synchronous   = false)        ///< [in] <b>[optional]</b> Whether or not to synchronize the stream after every kernel launch to check for errors.  Also causes launch configurations to be printed to the console.  Default is \p false.
    {
        // Signed integer type for global offsets
        typedef int Offset;

        // Wrapped input iterator
        typedef ArgIndexInputIterator<InputIterator, int> ArgIndexInputIterator;
        ArgIndexInputIterator d_argmin_in(d_in, 0);

        // Dispatch type
        typedef DeviceReduceDispatch<ArgIndexInputIterator, OutputIterator, Offset, cub::ArgMin> DeviceReduceDispatch;

        return DeviceReduceDispatch::Dispatch(
            d_temp_storage,
            temp_storage_bytes,
            d_argmin_in,
            d_out,
            num_items,
            cub::ArgMin(),
            stream,
            debug_synchronous);
    }


    /**
     * \brief Computes a device-wide maximum using the greater-than ('>') operator.
     *
     * \par
     * - Does not support non-commutative maximum operators.
     * - \devicestorage
     * - \cdp
     *
     * \par
     * The code snippet below illustrates the max-reduction of a device vector of \p int items.
     * \par
     * \code
     * #include <cub/cub.cuh>   // or equivalently <cub/device/device_radix_sort.cuh>
     *
     * // Declare, allocate, and initialize device pointers for input and output
     * int  num_items;      // e.g., 7
     * int  *d_in;          // e.g., [8, 6, 7, 5, 3, 0, 9]
     * int  *d_out;         // e.g., [ ]
     * ...
     *
     * // Determine temporary device storage requirements
     * void     *d_temp_storage = NULL;
     * size_t   temp_storage_bytes = 0;
     * cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, d_in, d_max, num_items);
     *
     * // Allocate temporary storage
     * cudaMalloc(&d_temp_storage, temp_storage_bytes);
     *
     * // Run max-reduction
     * cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, d_in, d_max, num_items);
     *
     * // d_out <-- [9]
     *
     * \endcode
     *
     * \tparam InputIterator      <b>[inferred]</b> Random-access iterator type for input \iterator
     * \tparam OutputIterator     <b>[inferred]</b> Random-access iterator type for output \iterator
     */
    template <
        typename                    InputIterator,
        typename                    OutputIterator>
    __host__ __device__
    static cudaError_t Max(
        void                        *d_temp_storage,                    ///< [in] %Device allocation of temporary storage.  When NULL, the required allocation size is returned in \p temp_storage_bytes and no work is done.
        size_t                      &temp_storage_bytes,                ///< [in,out] Size in bytes of \p d_temp_storage allocation
        InputIterator               d_in,                               ///< [in] Input data to reduce
        OutputIterator              d_out,                              ///< [out] Output location for result
        int                         num_items,                          ///< [in] Number of items to reduce
        cudaStream_t                stream              = 0,            ///< [in] <b>[optional]</b> CUDA stream to launch kernels within.  Default is stream<sub>0</sub>.
        bool                        debug_synchronous   = false)        ///< [in] <b>[optional]</b> Whether or not to synchronize the stream after every kernel launch to check for errors.  Also causes launch configurations to be printed to the console.  Default is \p false.
    {
        // Signed integer type for global offsets
        typedef int Offset;

        // Dispatch type
        typedef DeviceReduceDispatch<InputIterator, OutputIterator, Offset, cub::Max> DeviceReduceDispatch;

        return DeviceReduceDispatch::Dispatch(
            d_temp_storage,
            temp_storage_bytes,
            d_in,
            d_out,
            num_items,
            cub::Max(),
            stream,
            debug_synchronous);
    }


    /**
     * \brief Finds the first device-wide maximum using the greater-than ('>') operator, also returning the index of that item
     *
     * \par
     * Assuming the input \p d_in has value type \p T, the output \p d_out must have value type
     * <tt>ItemOffsetPair<T, int></tt>.  The maximum value is returned in <tt>d_out.value</tt> and its
     * location in the input array is returned in <tt>d_out.offset</tt>.
     *
     * \par
     * - Does not support non-commutative maximum operators.
     * - \devicestorage
     * - \cdp
     *
     * \par
     * The code snippet below illustrates the argmax-reduction of a device vector of \p int items.
     * \par
     * \code
     * #include <cub/cub.cuh>   // or equivalently <cub/device/device_reduce.cuh>
     *
     * // Declare, allocate, and initialize device pointers for input and output
     * int                      num_items;      // e.g., 7
     * int                      *d_in;          // e.g., [8, 6, 7, 5, 3, 0, 9]
     * int                      *d_out;         // e.g., [ ]
     * ItemOffsetPair<int, int> *d_argmin;      // e.g., [{ , }]
     * ...
     *
     * // Determine temporary device storage requirements
     * void     *d_temp_storage = NULL;
     * size_t   temp_storage_bytes = 0;
     * cub::DeviceReduce::ArgMax(d_temp_storage, temp_storage_bytes, d_in, d_argmax, num_items);
     *
     * // Allocate temporary storage
     * cudaMalloc(&d_temp_storage, temp_storage_bytes);
     *
     * // Run argmax-reduction
     * cub::DeviceReduce::ArgMax(d_temp_storage, temp_storage_bytes, d_in, d_argmax, num_items);
     *
     * // d_out <-- [{9, 6}]
     *
     * \endcode
     *
     * \tparam InputIterator      <b>[inferred]</b> Random-access iterator type for input having some value type \p T \iterator
     * \tparam OutputIterator     <b>[inferred]</b> Random-access iterator type for output having value type <tt>ItemOffsetPair<T, int></tt> \iterator
     */
    template <
        typename                    InputIterator,
        typename                    OutputIterator>
    __host__ __device__
    static cudaError_t ArgMax(
        void                        *d_temp_storage,                    ///< [in] %Device allocation of temporary storage.  When NULL, the required allocation size is returned in \p temp_storage_bytes and no work is done.
        size_t                      &temp_storage_bytes,                ///< [in,out] Size in bytes of \p d_temp_storage allocation
        InputIterator               d_in,                               ///< [in] Input data to reduce
        OutputIterator              d_out,                              ///< [out] Output location for result
        int                         num_items,                          ///< [in] Number of items to reduce
        cudaStream_t                stream              = 0,            ///< [in] <b>[optional]</b> CUDA stream to launch kernels within.  Default is stream<sub>0</sub>.
        bool                        debug_synchronous   = false)        ///< [in] <b>[optional]</b> Whether or not to synchronize the stream after every kernel launch to check for errors.  Also causes launch configurations to be printed to the console.  Default is \p false.
    {
        // Signed integer type for global offsets
        typedef int Offset;

        // Wrapped input iterator
        typedef ArgIndexInputIterator<InputIterator, int> ArgIndexInputIterator;
        ArgIndexInputIterator d_argmax_in(d_in, 0);

        // Dispatch type
        typedef DeviceReduceDispatch<ArgIndexInputIterator, OutputIterator, Offset, cub::ArgMax> DeviceReduceDispatch;

        return DeviceReduceDispatch::Dispatch(
            d_temp_storage,
            temp_storage_bytes,
            d_argmax_in,
            d_out,
            num_items,
            cub::ArgMax(),
            stream,
            debug_synchronous);
    }

};

/**
 * \example example_device_reduce.cu
 */

}               // CUB namespace
CUB_NS_POSTFIX  // Optional outer namespace(s)


