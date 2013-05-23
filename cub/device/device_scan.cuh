
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
 * cub::DeviceScan provides operations for computing a device-wide, parallel prefix scan across data items residing within global memory.
 */

#pragma once

#include <stdio.h>
#include <iterator>

#include "persistent_block/persistent_block_scan.cuh"
#include "../util_allocator.cuh"
#include "../grid/grid_even_share.cuh"
#include "../grid/grid_queue.cuh"
#include "../grid/grid_mapping.cuh"
#include "../util_debug.cuh"

/// Optional outer namespace(s)
CUB_NS_PREFIX

/// CUB namespace
namespace cub {


/******************************************************************************
 * Kernel entry points
 *****************************************************************************/

#ifndef DOXYGEN_SHOULD_SKIP_THIS    // Do not document


/**
 * Initialization kernel for queue descriptor preparation and for zeroing block status
 */
template <typename SizeT>                       ///< Integral type used for global array indexing
__global__ void InitScanKernel(
    GridQueue<SizeT>        grid_queue,         ///< [in] Descriptor for performing dynamic mapping of input tiles to thread blocks
    DeviceScanTileStatus    *d_tile_status,     ///< [out] Tile status words
    int                     num_tiles)          ///< [in] Number of tiles
{
    enum
    {
        STATUS_PADDING = PtxArchProps::WARP_THREADS,
    };

    // Reset queue descriptor
    if ((blockIdx.x == 0) && (threadIdx.x == 0)) grid_queue.ResetDrain(num_tiles);

    // Initialize tile status
    int tile_offset = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (tile_offset < num_tiles)
    {
        // Not-yet-set
        d_tile_status[STATUS_PADDING + tile_offset] =  DEVICE_SCAN_TILE_INVALID;
    }

    if ((blockIdx.x == 0) && (threadIdx.x < STATUS_PADDING))
    {
        // Padding
        d_tile_status[threadIdx.x] = DEVICE_SCAN_TILE_OOB;
    }
}


/**
 * Multi-block histogram kernel entry point.  Computes privatized histograms, one per thread block.
 */
template <
    typename    PersistentBlockScanPolicy,  ///< Tuning policy for cub::PersistentBlockScan abstraction
    typename    InputIteratorRA,            ///< The random-access iterator type for input (may be a simple pointer type).
    typename    OutputIteratorRA,           ///< The random-access iterator type for output (may be a simple pointer type).
    typename    ScanOp,                     ///< Binary scan operator type having member <tt>T operator()(const T &a, const T &b)</tt>
    typename    Identity,                   ///< Identity value type (cub::NullType for inclusive scans)
    typename    SizeT>                      ///< Integral type used for global array indexing
__launch_bounds__ (int(PersistentBlockScanPolicy::BLOCK_THREADS))
__global__ void MultiBlockScanKernel(
    InputIteratorRA                                             d_in,               ///< Input data
    OutputIteratorRA                                            d_out,              ///< Output data
    typename std::iterator_traits<InputIteratorRA>::value_type  *d_tile_aggregates, ///< Global list of block aggregates
    typename std::iterator_traits<InputIteratorRA>::value_type  *d_tile_prefixes,   ///< Global list of block inclusive prefixes
    DeviceScanTileStatus                                        *d_tile_status,     ///< Global list of tile status
    ScanOp                                                      scan_op,            ///< Binary scan operator
    Identity                                                    identity,           ///< Identity element
    SizeT                                                       num_items,          ///< Total number of scan items for the entire problem
    int                                                         num_tiles,          ///< Total number of input tiles
    GridQueue<SizeT>                                            queue)              ///< Descriptor for performing dynamic mapping of tile data to thread blocks
{
    // Thread block type for scanning input tiles
    typedef PersistentBlockScan<
        PersistentBlockScanPolicy,
        InputIteratorRA,
        OutputIteratorRA,
        ScanOp,
        Identity,
        SizeT> PersistentBlockScanT;

    // Shared memory for PersistentBlockScan
    __shared__ typename PersistentBlockScanT::SmemStorage smem_storage;

    // Thread block instance
    PersistentBlockScanT persistent_block(
        smem_storage,
        d_in,
        d_out,
        d_tile_aggregates,
        d_tile_prefixes,
        d_tile_status,
        scan_op,
        identity,
        num_items);

    // Shared tile-processing offset obtained dynamically from queue
    __shared__ SizeT dynamic_tile_idx;

    // We give each thread block at least one tile of input.
    SizeT tile_idx = blockIdx.x;

    // Check if we have a full tile to consume
    while (true)
    {
        if (tile_idx <= num_tiles - 1)
        {
            // Full tile to consume
            persistent_block.ConsumeTileFull(tile_idx);

            __syncthreads();

            // Dequeue up to tile_items
            if (threadIdx.x == 0)
                dynamic_tile_idx = queue.Drain(1) + gridDim.x;

            __syncthreads();

            tile_idx = dynamic_tile_idx;
        }
        else
        {
            if (tile_idx == num_tiles - 1)
            {
                // We have less than a full tile to consume
                persistent_block.ConsumeTilePartial(tile_idx);
            }

            break;
        }
    };


}


#endif // DOXYGEN_SHOULD_SKIP_THIS



/******************************************************************************
 * DeviceScan
 *****************************************************************************/

/**
 * \addtogroup DeviceModule
 * @{
 */

/**
 * \brief DeviceScan provides operations for computing a device-wide, parallel prefix scan across data items residing within global memory. ![](scan_logo.png)
 */
struct DeviceScan
{
#ifndef DOXYGEN_SHOULD_SKIP_THIS    // Do not document


    /// Generic structure for encapsulating dispatch properties.  Mirrors the constants within PersistentBlockScanPolicy.
    struct KernelDispachParams
    {
        // Policy fields
        int                             block_threads;
        int                             items_per_thread;
        BlockLoadAlgorithm                 load_policy;
        BlockStorePolicy                store_policy;
        BlockScanAlgorithm              scan_algorithm;

        // Derived fields
        int                             tile_size;

        template <typename PersistentBlockScanPolicy>
        __host__ __device__ __forceinline__
        void Init()
        {
            block_threads               = PersistentBlockScanPolicy::BLOCK_THREADS;
            items_per_thread            = PersistentBlockScanPolicy::ITEMS_PER_THREAD;
            load_policy                 = PersistentBlockScanPolicy::LOAD_POLICY;
            store_policy                = PersistentBlockScanPolicy::STORE_POLICY;
            scan_algorithm              = PersistentBlockScanPolicy::SCAN_ALGORITHM;

            tile_size                   = block_threads * items_per_thread;
        }

        __host__ __device__ __forceinline__
        void Print()
        {
            printf("%d, %d, %d, %d, %d",
                block_threads,
                items_per_thread,
                load_policy,
                store_policy,
                scan_algorithm);
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
        typedef PersistentBlockScanPolicy<128, 12,  BLOCK_LOAD_TRANSPOSE, BLOCK_STORE_TRANSPOSE, BLOCK_SCAN_RAKING_MEMOIZE> MultiBlockPolicy;
    };

    /// SM30 tune
    template <typename T, typename SizeT>
    struct TunedPolicies<T, SizeT, 300>
    {
        typedef PersistentBlockScanPolicy<256, 9,  BLOCK_LOAD_TRANSPOSE, BLOCK_STORE_TRANSPOSE, BLOCK_SCAN_RAKING_MEMOIZE> MultiBlockPolicy;
    };

    /// SM20 tune
    template <typename T, typename SizeT>
    struct TunedPolicies<T, SizeT, 200>
    {
        typedef PersistentBlockScanPolicy<128, 15,  BLOCK_LOAD_TRANSPOSE, BLOCK_STORE_TRANSPOSE, BLOCK_SCAN_RAKING_MEMOIZE> MultiBlockPolicy;
    };

    /// SM10 tune
    template <typename T, typename SizeT>
    struct TunedPolicies<T, SizeT, 100>
    {
        typedef PersistentBlockScanPolicy<128, 6,  BLOCK_LOAD_TRANSPOSE, BLOCK_STORE_TRANSPOSE, BLOCK_SCAN_RAKING_MEMOIZE> MultiBlockPolicy;
    };


    /// Tuning policy(ies) for the PTX architecture that DeviceScan operations will get dispatched to
    template <typename T, typename SizeT>
    struct PtxDefaultPolicies
    {
        static const int PTX_TUNE_ARCH =   (CUB_PTX_ARCH >= 350) ?
                                                350 :
                                                (CUB_PTX_ARCH >= 300) ?
                                                    300 :
                                                    (CUB_PTX_ARCH >= 200) ?
                                                        200 :
                                                        100;

        // Tuned policy set for the current PTX compiler pass
        typedef TunedPolicies<T, SizeT, PTX_TUNE_ARCH> PtxPassTunedPolicies;

        // MultiBlockPolicy that opaquely derives from the specialization corresponding to the current PTX compiler pass
        struct MultiBlockPolicy : PtxPassTunedPolicies::MultiBlockPolicy {};

        /**
         * Initialize dispatch params with the policies corresponding to the PTX assembly we will use
         */
        static void InitDispatchParams(int ptx_version, KernelDispachParams &multi_block_dispatch_params)
        {
            if (ptx_version >= 350)
            {
                typedef TunedPolicies<T, SizeT, 350> TunedPolicies;
                multi_block_dispatch_params.Init<typename TunedPolicies::MultiBlockPolicy>();
            }
            else if (ptx_version >= 300)
            {
                typedef TunedPolicies<T, SizeT, 300> TunedPolicies;
                multi_block_dispatch_params.Init<typename TunedPolicies::MultiBlockPolicy>();
            }
            else if (ptx_version >= 200)
            {
                typedef TunedPolicies<T, SizeT, 200> TunedPolicies;
                multi_block_dispatch_params.Init<typename TunedPolicies::MultiBlockPolicy>();
            }
            else
            {
                typedef TunedPolicies<T, SizeT, 100> TunedPolicies;
                multi_block_dispatch_params.Init<typename TunedPolicies::MultiBlockPolicy>();
            }
        }
    };





    /**
     * Internal dispatch routine for invoking device-wide, multi-channel, 256-bin histogram
     */
    template <
        typename                    InitScanKernelPtr,          ///< Function type of cub::InitScanKernel
        typename                    MultiBlockScanKernelPtr,    ///< Function type of cub::MultiBlockScanKernel
        typename                    InputIteratorRA,            ///< The random-access iterator type for input (may be a simple pointer type).
        typename                    OutputIteratorRA,           ///< The random-access iterator type for output (may be a simple pointer type).
        typename                    ScanOp,                     ///< Binary scan operator type having member <tt>T operator()(const T &a, const T &b)</tt>
        typename                    Identity,                   ///< Identity value type (cub::NullType for inclusive scans)
        typename                    SizeT>                      ///< Integral type used for global array indexing
    __host__ __device__ __forceinline__
    static cudaError_t Dispatch(
        InitScanKernelPtr           init_kernel_ptr,                                    ///< [in] Kernel function pointer to parameterization of cub::InitScanKernel
        MultiBlockScanKernelPtr     multi_block_kernel_ptr,                             ///< [in] Kernel function pointer to parameterization of cub::MultiBlockScanKernel
        KernelDispachParams         &multi_block_dispatch_params,                       ///< [in] Dispatch parameters that match the policy that \p multi_block_kernel_ptr was compiled for
        InputIteratorRA             d_in,                                               ///< [in] Input samples to histogram
        OutputIteratorRA            d_out,                                              ///< Output data
        ScanOp                      scan_op,                                            ///< Binary scan operator
        Identity                    identity,                                           ///< Identity element
        SizeT                       num_items,                                          ///< Total number of scan items for the entire problem
        cudaStream_t                stream              = 0,                            ///< [in] <b>[optional]</b> CUDA stream to launch kernels within.  Default is stream-0.
        bool                        stream_synchronous  = false,                        ///< [in] <b>[optional]</b> Whether or not to synchronize the stream after every kernel launch to check for errors.  Default is \p false.
        DeviceAllocator             *device_allocator   = DefaultDeviceAllocator())     ///< [in] <b>[optional]</b> Allocator for allocating and freeing device memory.  Default is provided by DefaultDeviceAllocator.
    {
    #ifndef CUB_RUNTIME_ENABLED

        // Kernel launch not supported from this device
        return CubDebug(cudaErrorInvalidConfiguration);

    #else

        enum
        {
            STATUS_PADDING = 32,
        };

        // Data type
        typedef typename std::iterator_traits<InputIteratorRA>::value_type T;

        T                       *d_tile_aggregates;     // Global list of block aggregates
        T                       *d_tile_prefixes;       // Global list of block inclusive prefixes
        DeviceScanTileStatus    *d_tile_status;         // Global list of tile status
        GridQueue<SizeT>        queue;                  // Dynamic, queue-based work distribution

        cudaError error = cudaSuccess;
        do
        {
            // Number of input tiles
            int num_tiles = (num_items + multi_block_dispatch_params.tile_size - 1) / multi_block_dispatch_params.tile_size;

            // Allocate temporary storage for tile aggregates, prefixes, and status
            if (CubDebug(error = DeviceAllocate((void**) &d_tile_aggregates, num_tiles * sizeof(T), device_allocator))) break;
            if (CubDebug(error = DeviceAllocate((void**) &d_tile_prefixes, num_tiles * sizeof(T), device_allocator))) break;
            if (CubDebug(error = DeviceAllocate((void**) &d_tile_status, (num_tiles + STATUS_PADDING) * sizeof(DeviceScanTileStatus), device_allocator))) break;

            // Allocate temporary storage for queue descriptor
            queue.Allocate(device_allocator);

            // Run initialization kernel
            int init_kernel_threads = 128;
            int init_grid_size = (num_tiles + init_kernel_threads - 1) / init_kernel_threads;
            if (stream_synchronous) CubLog("Invoking init_kernel_ptr<<<%d, %d, 0, %d>>>()\n", init_grid_size, init_kernel_threads, (int) stream);

            init_kernel_ptr<<<init_grid_size, init_kernel_threads, 0, stream>>>(queue, d_tile_status, num_tiles);

        #ifndef __CUDA_ARCH__
            // Sync the stream on the host
            if (stream_synchronous && CubDebug(error = cudaStreamSynchronize(stream))) break;
        #else
            // Sync the entire device on the device (cudaStreamSynchronize doesn't exist on device)
            if (stream_synchronous && CubDebug(error = cudaDeviceSynchronize())) break;
        #endif

            // Get GPU id
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
                multi_block_kernel_ptr,
                multi_block_dispatch_params.block_threads))) break;

        #endif

            // Multi-block kernel device occupancy
            int multi_occupancy = multi_sm_occupancy * sm_count;

            // Multi-block grid size
            int multi_grid_size = (num_tiles < multi_occupancy) ?
                num_tiles :                 // Not enough to fill the device with threadblocks
                multi_occupancy;            // Fill the device with threadblocks

            // Invoke MultiBlockScan
            if (stream_synchronous) CubLog("Invoking multi_block_kernel_ptr<<<%d, %d, 0, %d>>>(), %d items per thread, %d SM occupancy\n",
                multi_grid_size, multi_block_dispatch_params.block_threads, (int) stream, multi_block_dispatch_params.items_per_thread, multi_sm_occupancy);

            multi_block_kernel_ptr<<<multi_grid_size, multi_block_dispatch_params.block_threads, 0, stream>>>(
                d_in,
                d_out,
                d_tile_aggregates,
                d_tile_prefixes,
                d_tile_status,
                scan_op,
                identity,
                num_items,
                num_tiles,
                queue);

            #ifndef __CUDA_ARCH__
                // Sync the stream on the host
                if (stream_synchronous && CubDebug(error = cudaStreamSynchronize(stream))) break;
            #else
                // Sync the entire device on the device (cudaStreamSynchronize doesn't exist on device)
                if (stream_synchronous && CubDebug(error = cudaDeviceSynchronize())) break;
            #endif

        }
        while (0);

        // Free temporary storage allocations
        if (d_tile_aggregates)
            error = CubDebug(DeviceFree(d_tile_aggregates, device_allocator));
        if (d_tile_prefixes)
            error = CubDebug(DeviceFree(d_tile_prefixes, device_allocator));
        if (d_tile_status)
            error = CubDebug(DeviceFree(d_tile_status, device_allocator));

        // Free queue allocation
        error = CubDebug(queue.Free(device_allocator));

        return error;

    #endif
    }



    /**
     * Internal dispatch routine for invoking device-wide, multi-channel, 256-bin histogram
     */
    template <
        typename                    InputIteratorRA,            ///< The random-access iterator type for input (may be a simple pointer type).
        typename                    OutputIteratorRA,           ///< The random-access iterator type for output (may be a simple pointer type).
        typename                    ScanOp,                     ///< Binary scan operator type having member <tt>T operator()(const T &a, const T &b)</tt>
        typename                    Identity,                   ///< Identity value type (cub::NullType for inclusive scans)
        typename                    SizeT>                      ///< Integral type used for global array indexing
    __host__ __device__ __forceinline__
    static cudaError_t Dispatch(
        InputIteratorRA             d_in,                                               ///< [in] Input samples to histogram
        OutputIteratorRA            d_out,                                              ///< Output data
        ScanOp                      scan_op,                                            ///< Binary scan operator
        Identity                    identity,                                           ///< Identity element
        SizeT                       num_items,                                          ///< Total number of scan items for the entire problem
        cudaStream_t                stream              = 0,                            ///< [in] <b>[optional]</b> CUDA stream to launch kernels within.  Default is stream-0.
        bool                        stream_synchronous  = false,                        ///< [in] <b>[optional]</b> Whether or not to synchronize the stream after every kernel launch to check for errors.  Default is \p false.
        DeviceAllocator             *device_allocator   = DefaultDeviceAllocator())     ///< [in] <b>[optional]</b> Allocator for allocating and freeing device memory.  Default is provided by DefaultDeviceAllocator.
    {
        // Data type
        typedef typename std::iterator_traits<InputIteratorRA>::value_type T;

        // Tuning polices for the PTX architecture that will get dispatched to
        typedef PtxDefaultPolicies<T, SizeT> PtxDefaultPolicies;
        typedef typename PtxDefaultPolicies::MultiBlockPolicy MultiBlockPolicy;

        cudaError error = cudaSuccess;
        do
        {
            // Declare dispatch parameters
            KernelDispachParams multi_block_dispatch_params;

        #ifdef __CUDA_ARCH__

            // We're on the device, so initialize the dispatch parameters with the PtxDefaultPolicies directly
            multi_block_dispatch_params.Init<MultiBlockPolicy>();

        #else

            // We're on the host, so lookup and initialize the dispatch parameters with the policies that match the device's PTX version
            int ptx_version;
            if (CubDebug(error = PtxVersion(ptx_version))) break;
            PtxDefaultPolicies::InitDispatchParams(ptx_version, multi_block_dispatch_params);

        #endif

            Dispatch(
                InitScanKernel<SizeT>,
                MultiBlockScanKernel<MultiBlockPolicy, InputIteratorRA, OutputIteratorRA, ScanOp, Identity, SizeT>,
                multi_block_dispatch_params,
                d_in,
                d_out,
                scan_op,
                identity,
                num_items,
                stream,
                stream_synchronous,
                device_allocator);

            if (CubDebug(error)) break;
        }
        while (0);

        return error;
    }

    #endif // DOXYGEN_SHOULD_SKIP_THIS


    //---------------------------------------------------------------------
    // Public interface
    //---------------------------------------------------------------------

    /**
     * \brief Computes a device-wide exclusive prefix sum.
     *
     * \tparam InputIteratorRA      <b>[inferred]</b> The random-access iterator type for input (may be a simple pointer type).
     * \tparam OutputIteratorRA     <b>[inferred]</b> The random-access iterator type for output (may be a simple pointer type).
     * \tparam SizeT                <b>[inferred]</b> Integral type used for global array indexing
     */
    template <
        typename                    InputIteratorRA,
        typename                    OutputIteratorRA,
        typename                    SizeT>
    __host__ __device__ __forceinline__
    static cudaError_t ExclusiveSum(
        InputIteratorRA             d_in,                                               ///< [in] Input samples to histogram
        OutputIteratorRA            d_out,                                              ///< Output data
        SizeT                       num_items,                                          ///< Total number of scan items for the entire problem
        cudaStream_t                stream              = 0,                            ///< [in] <b>[optional]</b> CUDA stream to launch kernels within.  Default is stream-0.
        bool                        stream_synchronous  = false,                        ///< [in] <b>[optional]</b> Whether or not to synchronize the stream after every kernel launch to check for errors.  Default is \p false.
        DeviceAllocator             *device_allocator   = DefaultDeviceAllocator())
    {
        typedef typename std::iterator_traits<InputIteratorRA>::value_type T;
        return Dispatch(d_in, d_out, Sum<T>(), T(), num_items, stream, stream_synchronous, device_allocator);
    }


    /**
     * \brief Computes a device-wide exclusive prefix scan using the specified binary \p scan_op functor.
     *
     * \tparam InputIteratorRA      <b>[inferred]</b> The random-access iterator type for input (may be a simple pointer type).
     * \tparam OutputIteratorRA     <b>[inferred]</b> The random-access iterator type for output (may be a simple pointer type).
     * \tparam ScanOp               <b>[inferred]</b> Binary scan operator type having member <tt>T operator()(const T &a, const T &b)</tt>
     * \tparam Identity             <b>[inferred]</b> Type of the \p identity value used Binary scan operator type having member <tt>T operator()(const T &a, const T &b)</tt>
     * \tparam SizeT                <b>[inferred]</b> Integral type used for global array indexing
     */
    template <
        typename                    InputIteratorRA,
        typename                    OutputIteratorRA,
        typename                    ScanOp,
        typename                    Identity,
        typename                    SizeT>
    __host__ __device__ __forceinline__
    static cudaError_t ExclusiveScan(
        InputIteratorRA             d_in,                                               ///< [in] Input samples to histogram
        OutputIteratorRA            d_out,                                              ///< Output data
        ScanOp                      scan_op,                                            ///< Binary scan operator
        Identity                    identity,                                           ///< Identity element
        SizeT                       num_items,                                          ///< Total number of scan items for the entire problem
        cudaStream_t                stream              = 0,                            ///< [in] <b>[optional]</b> CUDA stream to launch kernels within.  Default is stream-0.
        bool                        stream_synchronous  = false,                        ///< [in] <b>[optional]</b> Whether or not to synchronize the stream after every kernel launch to check for errors.  Default is \p false.
        DeviceAllocator             *device_allocator   = DefaultDeviceAllocator())
    {
        return Dispatch(d_in, d_out, scan_op, identity, num_items, stream, stream_synchronous, device_allocator);
    }


    /**
     * \brief Computes a device-wide inclusive prefix sum.
     *
     * \tparam InputIteratorRA      <b>[inferred]</b> The random-access iterator type for input (may be a simple pointer type).
     * \tparam OutputIteratorRA     <b>[inferred]</b> The random-access iterator type for output (may be a simple pointer type).
     * \tparam SizeT                <b>[inferred]</b> Integral type used for global array indexing
     */
    template <
        typename                    InputIteratorRA,
        typename                    OutputIteratorRA,
        typename                    SizeT>
    __host__ __device__ __forceinline__
    static cudaError_t InclusiveSum(
        InputIteratorRA             d_in,                                               ///< [in] Input samples to histogram
        OutputIteratorRA            d_out,                                              ///< Output data
        SizeT                       num_items,                                          ///< Total number of scan items for the entire problem
        cudaStream_t                stream              = 0,                            ///< [in] <b>[optional]</b> CUDA stream to launch kernels within.  Default is stream-0.
        bool                        stream_synchronous  = false,                        ///< [in] <b>[optional]</b> Whether or not to synchronize the stream after every kernel launch to check for errors.  Default is \p false.
        DeviceAllocator             *device_allocator   = DefaultDeviceAllocator())
    {
        typedef typename std::iterator_traits<InputIteratorRA>::value_type T;
        return Dispatch(d_in, d_out, Sum<T>(), NullType(), num_items, stream, stream_synchronous, device_allocator);
    }


    /**
     * \brief Computes a device-wide exclusive prefix scan using the specified binary \p scan_op functor.
     *
     * \tparam InputIteratorRA      <b>[inferred]</b> The random-access iterator type for input (may be a simple pointer type).
     * \tparam OutputIteratorRA     <b>[inferred]</b> The random-access iterator type for output (may be a simple pointer type).
     * \tparam ScanOp               <b>[inferred]</b> Binary scan operator type having member <tt>T operator()(const T &a, const T &b)</tt>
     * \tparam SizeT                <b>[inferred]</b> Integral type used for global array indexing
     */
    template <
        typename                    InputIteratorRA,
        typename                    OutputIteratorRA,
        typename                    ScanOp,
        typename                    SizeT>
    __host__ __device__ __forceinline__
    static cudaError_t InclusiveScan(
        InputIteratorRA             d_in,                                               ///< [in] Input samples to histogram
        OutputIteratorRA            d_out,                                              ///< Output data
        ScanOp                      scan_op,                                            ///< Binary scan operator
        SizeT                       num_items,                                          ///< Total number of scan items for the entire problem
        cudaStream_t                stream              = 0,                            ///< [in] <b>[optional]</b> CUDA stream to launch kernels within.  Default is stream-0.
        bool                        stream_synchronous  = false,                        ///< [in] <b>[optional]</b> Whether or not to synchronize the stream after every kernel launch to check for errors.  Default is \p false.
        DeviceAllocator             *device_allocator   = DefaultDeviceAllocator())
    {
        return Dispatch(d_in, d_out, scan_op, NullType(), num_items, stream, stream_synchronous, device_allocator);
    }

};


/** @} */       // DeviceModule

}               // CUB namespace
CUB_NS_POSTFIX  // Optional outer namespace(s)


