
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

#include "block/block_scan_tiles.cuh"
#include "../thread/thread_operators.cuh"
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
 * Initialization kernel for tile status initialization (multi-block)
 */
template <
    typename T,                                     ///< Scan value type
    typename SizeT>                                 ///< Integral type used for global array indexing
__global__ void InitScanKernel(
    GridQueue<SizeT>            grid_queue,         ///< [in] Descriptor for performing dynamic mapping of input tiles to thread blocks
    DeviceScanTileDescriptor<T> *d_tile_status,     ///< [out] Tile status words
    int                         num_tiles)          ///< [in] Number of tiles
{
    typedef DeviceScanTileDescriptor<T> DeviceScanTileDescriptorT;

    enum
    {
        TILE_STATUS_PADDING = PtxArchProps::WARP_THREADS,
    };

    // Reset queue descriptor
    if ((blockIdx.x == 0) && (threadIdx.x == 0)) grid_queue.ResetDrain(num_tiles);

    // Initialize tile status
    int tile_offset = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (tile_offset < num_tiles)
    {
        // Not-yet-set
        d_tile_status[TILE_STATUS_PADDING + tile_offset].status = DEVICE_SCAN_TILE_INVALID;
    }

    if ((blockIdx.x == 0) && (threadIdx.x < TILE_STATUS_PADDING))
    {
        // Padding
        d_tile_status[threadIdx.x].status = DEVICE_SCAN_TILE_OOB;
    }
}


/**
 * Scan kernel entry point (multi-block)
 */
template <
    typename    BlockScanTilesPolicy,       ///< Tuning policy for cub::BlockScanTiles abstraction
    typename    InputIteratorRA,                ///< Random-access iterator type for input (may be a simple pointer type)
    typename    OutputIteratorRA,               ///< Random-access iterator type for output (may be a simple pointer type)
    typename    T,                              ///< The scan data type
    typename    ScanOp,                         ///< Binary scan operator type having member <tt>T operator()(const T &a, const T &b)</tt>
    typename    Identity,                       ///< Identity value type (cub::NullType for inclusive scans)
    typename    SizeT>                          ///< Integral type used for global array indexing
__launch_bounds__ (int(BlockScanTilesPolicy::BLOCK_THREADS))
__global__ void MultiBlockScanKernel(
    InputIteratorRA             d_in,           ///< Input data
    OutputIteratorRA            d_out,          ///< Output data
    DeviceScanTileDescriptor<T>     *d_tile_status, ///< Global list of tile status
    ScanOp                      scan_op,        ///< Binary scan operator
    Identity                    identity,       ///< Identity element
    SizeT                       num_items,      ///< Total number of scan items for the entire problem
    GridQueue<int>              queue)          ///< Descriptor for performing dynamic mapping of tile data to thread blocks
{
    enum
    {
        TILE_STATUS_PADDING = PtxArchProps::WARP_THREADS,
    };

    // Thread block type for scanning input tiles
    typedef BlockScanTiles<
        BlockScanTilesPolicy,
        InputIteratorRA,
        OutputIteratorRA,
        ScanOp,
        Identity,
        SizeT> BlockScanTilesT;

    // Shared memory for BlockScanTiles
    __shared__ typename BlockScanTilesT::TempStorage temp_storage;

    // Process tiles
    BlockScanTilesT(temp_storage, d_in, d_out, scan_op, identity).ConsumeTiles(
        num_items,
        queue,
        d_tile_status + TILE_STATUS_PADDING);
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

    /******************************************************************************
     * Constants and typedefs
     ******************************************************************************/

    /// Generic structure for encapsulating dispatch properties.  Mirrors the constants within BlockScanTilesPolicy.
    struct KernelDispachParams
    {
        // Policy fields
        int                     block_threads;
        int                     items_per_thread;
        BlockLoadAlgorithm      load_policy;
        BlockStoreAlgorithm     store_policy;
        BlockScanAlgorithm      scan_algorithm;

        // Other misc
        int                     tile_size;

        template <typename BlockScanTilesPolicy>
        __host__ __device__ __forceinline__
        void Init()
        {
            block_threads               = BlockScanTilesPolicy::BLOCK_THREADS;
            items_per_thread            = BlockScanTilesPolicy::ITEMS_PER_THREAD;
            load_policy                 = BlockScanTilesPolicy::LOAD_ALGORITHM;
            store_policy                = BlockScanTilesPolicy::STORE_ALGORITHM;
            scan_algorithm              = BlockScanTilesPolicy::SCAN_ALGORITHM;

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
        enum {
            NOMINAL_ITEMS_PER_THREAD    = 16,   // 4byte items
            ITEMS_PER_THREAD            = CUB_MIN(NOMINAL_ITEMS_PER_THREAD, CUB_MAX(1, (NOMINAL_ITEMS_PER_THREAD * 4 / sizeof(T)))),
        };
        typedef BlockScanTilesPolicy<128, ITEMS_PER_THREAD,  BLOCK_LOAD_DIRECT, false, LOAD_LDG, BLOCK_STORE_WARP_TRANSPOSE, true, BLOCK_SCAN_RAKING_MEMOIZE> MultiBlockPolicy;
    };

    /// SM30 tune
    template <typename T, typename SizeT>
    struct TunedPolicies<T, SizeT, 300>
    {
        enum {
            NOMINAL_ITEMS_PER_THREAD    = 9,   // 4byte items
            ITEMS_PER_THREAD            = CUB_MAX(1, (NOMINAL_ITEMS_PER_THREAD * 4 / sizeof(T))),
        };
        typedef BlockScanTilesPolicy<256, ITEMS_PER_THREAD,  BLOCK_LOAD_WARP_TRANSPOSE, false, LOAD_DEFAULT, BLOCK_STORE_WARP_TRANSPOSE, false, BLOCK_SCAN_RAKING_MEMOIZE> MultiBlockPolicy;
    };

    /// SM20 tune
    template <typename T, typename SizeT>
    struct TunedPolicies<T, SizeT, 200>
    {
        enum {
            NOMINAL_ITEMS_PER_THREAD    = 15,   // 4byte items
            ITEMS_PER_THREAD            = CUB_MIN(NOMINAL_ITEMS_PER_THREAD, CUB_MAX(1, (NOMINAL_ITEMS_PER_THREAD * 4 / sizeof(T)))),
        };
        typedef BlockScanTilesPolicy<128, ITEMS_PER_THREAD, BLOCK_LOAD_WARP_TRANSPOSE, false, LOAD_DEFAULT, BLOCK_STORE_WARP_TRANSPOSE, false, BLOCK_SCAN_RAKING_MEMOIZE> MultiBlockPolicy;
    };

    /// SM10 tune
    template <typename T, typename SizeT>
    struct TunedPolicies<T, SizeT, 100>
    {
        enum {
            NOMINAL_ITEMS_PER_THREAD    = 7,   // 4byte items
            ITEMS_PER_THREAD            = CUB_MAX(1, (NOMINAL_ITEMS_PER_THREAD * 4 / sizeof(T))),
        };
        typedef BlockScanTilesPolicy<128, ITEMS_PER_THREAD, BLOCK_LOAD_TRANSPOSE, false, LOAD_DEFAULT, BLOCK_STORE_TRANSPOSE, false, BLOCK_SCAN_RAKING> MultiBlockPolicy;
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


    /******************************************************************************
     * Utility methods
     ******************************************************************************/

    /**
     * Internal dispatch routine
     */
    template <
        typename                    InitScanKernelPtr,              ///< Function type of cub::InitScanKernel
        typename                    MultiBlockScanKernelPtr,        ///< Function type of cub::MultiBlockScanKernel
        typename                    InputIteratorRA,                ///< Random-access iterator type for input (may be a simple pointer type)
        typename                    OutputIteratorRA,               ///< Random-access iterator type for output (may be a simple pointer type)
        typename                    ScanOp,                         ///< Binary scan operator type having member <tt>T operator()(const T &a, const T &b)</tt>
        typename                    Identity,                       ///< Identity value type (cub::NullType for inclusive scans)
        typename                    SizeT>                          ///< Integral type used for global array indexing
    __host__ __device__ __forceinline__
    static cudaError_t Dispatch(
        void                        *d_temp_storage,                ///< [in] %Device allocation of temporary storage.  When NULL, the required allocation size is returned in \p temp_storage_bytes and no work is done.
        size_t                      &temp_storage_bytes,            ///< [in,out] Size in bytes of \p d_temp_storage allocation.
        InitScanKernelPtr           init_kernel,                    ///< [in] Kernel function pointer to parameterization of cub::InitScanKernel
        MultiBlockScanKernelPtr     multi_block_kernel,             ///< [in] Kernel function pointer to parameterization of cub::MultiBlockScanKernel
        KernelDispachParams         &multi_block_dispatch_params,   ///< [in] Dispatch parameters that match the policy that \p multi_block_kernel was compiled for
        InputIteratorRA             d_in,                           ///< [in] Iterator pointing to scan input
        OutputIteratorRA            d_out,                          ///< [in] Iterator pointing to scan output
        ScanOp                      scan_op,                        ///< [in] Binary scan operator
        Identity                    identity,                       ///< [in] Identity element
        SizeT                       num_items,                      ///< [in] Total number of items to scan
        cudaStream_t                stream              = 0,        ///< [in] <b>[optional]</b> CUDA stream to launch kernels within.  Default is stream<sub>0</sub>.
        bool                        stream_synchronous  = false)    ///< [in] <b>[optional]</b> Whether or not to synchronize the stream after every kernel launch to check for errors.  Default is \p false.
    {

#ifndef CUB_RUNTIME_ENABLED

        // Kernel launch not supported from this device
        return CubDebug(cudaErrorNotSupported);

#else

        enum
        {
            TILE_STATUS_PADDING = 32,
        };

        // Data type
        typedef typename std::iterator_traits<InputIteratorRA>::value_type T;

        // Tile status descriptor type
        typedef DeviceScanTileDescriptor<T> DeviceScanTileDescriptorT;

        cudaError error = cudaSuccess;
        do
        {
            // Number of input tiles
            int num_tiles = (num_items + multi_block_dispatch_params.tile_size - 1) / multi_block_dispatch_params.tile_size;

            // Temporary storage allocation requirements
            void* allocations[2];
            size_t allocation_sizes[2] =
            {
                (num_tiles + TILE_STATUS_PADDING) * sizeof(DeviceScanTileDescriptorT),      // bytes needed for tile status descriptors
                GridQueue<int>::AllocationSize()                                            // bytes needed for grid queue descriptor
            };

            // Alias temporaries (or set the necessary size of the storage allocation)
            if (CubDebug(error = AliasTemporaries(d_temp_storage, temp_storage_bytes, allocations, allocation_sizes))) break;

            // Return if the caller is simply requesting the size of the storage allocation
            if (d_temp_storage == NULL)
                return cudaSuccess;

            // Global list of tile status
            DeviceScanTileDescriptorT *d_tile_status = (DeviceScanTileDescriptorT*) allocations[0];

            // Grid queue descriptor
            GridQueue<int> queue(allocations[1]);

            // Get GPU id
            int device_ordinal;
            if (CubDebug(error = cudaGetDevice(&device_ordinal))) break;

            // Get SM count
            int sm_count;
            if (CubDebug(error = cudaDeviceGetAttribute (&sm_count, cudaDevAttrMultiProcessorCount, device_ordinal))) break;

            // Log init_kernel configuration
            int init_kernel_threads = 128;
            int init_grid_size = (num_tiles + init_kernel_threads - 1) / init_kernel_threads;
            if (stream_synchronous) CubLog("Invoking init_kernel<<<%d, %d, 0, %lld>>>()\n", init_grid_size, init_kernel_threads, (long long) stream);

            // Invoke init_kernel to initialize tile descriptors and queue descriptors
            init_kernel<<<init_grid_size, init_kernel_threads, 0, stream>>>(
                queue,
                d_tile_status,
                num_tiles);

            // Sync the stream if specified
            if (stream_synchronous && (CubDebug(error = SyncStream(stream)))) break;

            // Get a rough estimate of multi_block_kernel SM occupancy based upon the maximum SM occupancy of the targeted PTX architecture
            int multi_sm_occupancy = CUB_MIN(
                ArchProps<CUB_PTX_ARCH>::MAX_SM_THREADBLOCKS,
                ArchProps<CUB_PTX_ARCH>::MAX_SM_THREADS / multi_block_dispatch_params.block_threads);

#ifndef __CUDA_ARCH__

            // We're on the host, so come up with a more accurate estimate of multi_block_kernel SM occupancy from actual device properties
            Device device_props;
            if (CubDebug(error = device_props.Init(device_ordinal))) break;

            if (CubDebug(error = device_props.MaxSmOccupancy(
                multi_sm_occupancy,
                multi_block_kernel,
                multi_block_dispatch_params.block_threads))) break;

#endif
            // Get device occupancy for multi_block_kernel
            int multi_block_occupancy = multi_sm_occupancy * sm_count;

            // Get grid size for multi_block_kernel
            int multi_block_grid_size = (num_tiles < multi_block_occupancy) ?
                num_tiles :                 // Not enough to fill the device with threadblocks
                multi_block_occupancy;            // Fill the device with threadblocks

            // Log multi_block_kernel configuration
            if (stream_synchronous) CubLog("Invoking multi_block_kernel<<<%d, %d, 0, %lld>>>(), %d items per thread, %d SM occupancy\n",
                multi_block_grid_size, multi_block_dispatch_params.block_threads, (long long) stream, multi_block_dispatch_params.items_per_thread, multi_sm_occupancy);

            // Invoke multi_block_kernel
            multi_block_kernel<<<multi_block_grid_size, multi_block_dispatch_params.block_threads, 0, stream>>>(
                d_in,
                d_out,
                d_tile_status,
                scan_op,
                identity,
                num_items,
                queue);

            // Sync the stream if specified
            if (stream_synchronous && (CubDebug(error = SyncStream(stream)))) break;
        }
        while (0);

        return error;

#endif  // CUB_RUNTIME_ENABLED
    }



    /**
     * Internal scan dispatch routine for using default tuning policies
     */
    template <
        typename                    InputIteratorRA,                ///< Random-access iterator type for input (may be a simple pointer type)
        typename                    OutputIteratorRA,               ///< Random-access iterator type for output (may be a simple pointer type)
        typename                    ScanOp,                         ///< Binary scan operator type having member <tt>T operator()(const T &a, const T &b)</tt>
        typename                    Identity,                       ///< Identity value type (cub::NullType for inclusive scans)
        typename                    SizeT>                          ///< Integral type used for global array indexing
    __host__ __device__ __forceinline__
    static cudaError_t Dispatch(
        void                        *d_temp_storage,                ///< [in] %Device allocation of temporary storage.  When NULL, the required allocation size is returned in \p temp_storage_bytes and no work is done.
        size_t                      &temp_storage_bytes,            ///< [in,out] Size in bytes of \p d_temp_storage allocation.
        InputIteratorRA             d_in,                           ///< [in] Iterator pointing to scan input
        OutputIteratorRA            d_out,                          ///< [in] Iterator pointing to scan output
        ScanOp                      scan_op,                        ///< [in] Binary scan operator
        Identity                    identity,                       ///< [in] Identity element
        SizeT                       num_items,                      ///< [in] Total number of items to scan
        cudaStream_t                stream              = 0,        ///< [in] <b>[optional]</b> CUDA stream to launch kernels within.  Default is stream<sub>0</sub>.
        bool                        stream_synchronous  = false)    ///< [in] <b>[optional]</b> Whether or not to synchronize the stream after every kernel launch to check for errors.  Default is \p false.
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
                d_temp_storage,
                temp_storage_bytes,
                InitScanKernel<T, SizeT>,
                MultiBlockScanKernel<MultiBlockPolicy, InputIteratorRA, OutputIteratorRA, T, ScanOp, Identity, SizeT>,
                multi_block_dispatch_params,
                d_in,
                d_out,
                scan_op,
                identity,
                num_items,
                stream,
                stream_synchronous);

            if (CubDebug(error)) break;
        }
        while (0);

        return error;
    }

    #endif // DOXYGEN_SHOULD_SKIP_THIS


    /******************************************************************//**
     * \name Exclusive
     *********************************************************************/
    //@{

    /**
     * \brief Computes a device-wide exclusive prefix sum.
     *
     * \devicestorage
     *
     * \tparam InputIteratorRA      <b>[inferred]</b> Random-access iterator type for input (may be a simple pointer type)
     * \tparam OutputIteratorRA     <b>[inferred]</b> Random-access iterator type for output (may be a simple pointer type)
     */
    template <
        typename            InputIteratorRA,
        typename            OutputIteratorRA>
    __host__ __device__ __forceinline__
    static cudaError_t ExclusiveSum(
        void                *d_temp_storage,                    ///< [in] %Device allocation of temporary storage.  When NULL, the required allocation size is returned in \p temp_storage_bytes and no work is done.
        size_t              &temp_storage_bytes,                ///< [in,out] Size in bytes of \p d_temp_storage allocation.
        InputIteratorRA     d_in,                               ///< [in] Iterator pointing to scan input
        OutputIteratorRA    d_out,                              ///< [in] Iterator pointing to scan output
        int                 num_items,                          ///< [in] Total number of items to scan
        cudaStream_t        stream              = 0,            ///< [in] <b>[optional]</b> CUDA stream to launch kernels within.  Default is stream<sub>0</sub>.
        bool                stream_synchronous  = false)        ///< [in] <b>[optional]</b> Whether or not to synchronize the stream after every kernel launch to check for errors.  May cause significant slowdown.  Default is \p false.
    {
        typedef typename std::iterator_traits<InputIteratorRA>::value_type T;
        return Dispatch(d_temp_storage, temp_storage_bytes, d_in, d_out, Sum(), T(), num_items, stream, stream_synchronous);
    }


    /**
     * \brief Computes a device-wide exclusive prefix scan using the specified binary \p scan_op functor.
     *
     * \devicestorage
     *
     * \tparam InputIteratorRA      <b>[inferred]</b> Random-access iterator type for input (may be a simple pointer type)
     * \tparam OutputIteratorRA     <b>[inferred]</b> Random-access iterator type for output (may be a simple pointer type)
     * \tparam ScanOp               <b>[inferred]</b> Binary scan operator type having member <tt>T operator()(const T &a, const T &b)</tt>
     * \tparam Identity             <b>[inferred]</b> Type of the \p identity value used Binary scan operator type having member <tt>T operator()(const T &a, const T &b)</tt>
     */
    template <
        typename            InputIteratorRA,
        typename            OutputIteratorRA,
        typename            ScanOp,
        typename            Identity>
    __host__ __device__ __forceinline__
    static cudaError_t ExclusiveScan(
        void                *d_temp_storage,                    ///< [in] %Device allocation of temporary storage.  When NULL, the required allocation size is returned in \p temp_storage_bytes and no work is done.
        size_t              &temp_storage_bytes,                ///< [in,out] Size in bytes of \p d_temp_storage allocation.
        InputIteratorRA     d_in,                               ///< [in] Iterator pointing to scan input
        OutputIteratorRA    d_out,                              ///< [in] Iterator pointing to scan output
        ScanOp              scan_op,                            ///< [in] Binary scan operator
        Identity            identity,                           ///< [in] Identity element
        int                 num_items,                          ///< [in] Total number of items to scan
        cudaStream_t        stream              = 0,            ///< [in] <b>[optional]</b> CUDA stream to launch kernels within.  Default is stream<sub>0</sub>.
        bool                stream_synchronous  = false)        ///< [in] <b>[optional]</b> Whether or not to synchronize the stream after every kernel launch to check for errors.  May cause significant slowdown.  Default is \p false.
    {
        return Dispatch(d_temp_storage, temp_storage_bytes, d_in, d_out, scan_op, identity, num_items, stream, stream_synchronous);
    }


    //@}  end member group
    /******************************************************************//**
     * \name Inclusive
     *********************************************************************/
    //@{


    /**
     * \brief Computes a device-wide inclusive prefix sum.
     *
     * \devicestorage
     *
     * \tparam InputIteratorRA      <b>[inferred]</b> Random-access iterator type for input (may be a simple pointer type)
     * \tparam OutputIteratorRA     <b>[inferred]</b> Random-access iterator type for output (may be a simple pointer type)
     */
    template <
        typename            InputIteratorRA,
        typename            OutputIteratorRA>
    __host__ __device__ __forceinline__
    static cudaError_t InclusiveSum(
        void                *d_temp_storage,                    ///< [in] %Device allocation of temporary storage.  When NULL, the required allocation size is returned in \p temp_storage_bytes and no work is done.
        size_t              &temp_storage_bytes,                ///< [in,out] Size in bytes of \p d_temp_storage allocation.
        InputIteratorRA     d_in,                               ///< [in] Iterator pointing to scan input
        OutputIteratorRA    d_out,                              ///< [in] Iterator pointing to scan output
        int                 num_items,                          ///< [in] Total number of items to scan
        cudaStream_t        stream              = 0,            ///< [in] <b>[optional]</b> CUDA stream to launch kernels within.  Default is stream<sub>0</sub>.
        bool                stream_synchronous  = false)        ///< [in] <b>[optional]</b> Whether or not to synchronize the stream after every kernel launch to check for errors.  May cause significant slowdown.  Default is \p false.
    {
        return Dispatch(d_temp_storage, temp_storage_bytes, d_in, d_out, Sum(), NullType(), num_items, stream, stream_synchronous);
    }


    /**
     * \brief Computes a device-wide exclusive prefix scan using the specified binary \p scan_op functor.
     *
     * \devicestorage
     *
     * \tparam InputIteratorRA      <b>[inferred]</b> Random-access iterator type for input (may be a simple pointer type)
     * \tparam OutputIteratorRA     <b>[inferred]</b> Random-access iterator type for output (may be a simple pointer type)
     * \tparam ScanOp               <b>[inferred]</b> Binary scan operator type having member <tt>T operator()(const T &a, const T &b)</tt>
     */
    template <
        typename            InputIteratorRA,
        typename            OutputIteratorRA,
        typename            ScanOp>
    __host__ __device__ __forceinline__
    static cudaError_t InclusiveScan(
        void                *d_temp_storage,                    ///< [in] %Device allocation of temporary storage.  When NULL, the required allocation size is returned in \p temp_storage_bytes and no work is done.
        size_t              &temp_storage_bytes,                ///< [in,out] Size in bytes of \p d_temp_storage allocation.
        InputIteratorRA     d_in,                               ///< [in] Iterator pointing to scan input
        OutputIteratorRA    d_out,                              ///< [in] Iterator pointing to scan output
        ScanOp              scan_op,                            ///< [in] Binary scan operator
        int                 num_items,                          ///< [in] Total number of items to scan
        cudaStream_t        stream              = 0,            ///< [in] <b>[optional]</b> CUDA stream to launch kernels within.  Default is stream<sub>0</sub>.
        bool                stream_synchronous  = false)        ///< [in] <b>[optional]</b> Whether or not to synchronize the stream after every kernel launch to check for errors.  May cause significant slowdown.  Default is \p false.
    {
        return Dispatch(d_temp_storage, temp_storage_bytes, d_in, d_out, scan_op, NullType(), num_items, stream, stream_synchronous);
    }

};


/** @} */       // DeviceModule

}               // CUB namespace
CUB_NS_POSTFIX  // Optional outer namespace(s)


