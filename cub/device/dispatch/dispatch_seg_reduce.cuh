
/******************************************************************************
 * Copyright (c) 2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2015, NVIDIA CORPORATION.  All rights reserved.
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
 * cub::DeviceSegReduce provides device-wide parallel operations for performing sparse-matrix * vector multiplication (SpMV).
 */

#pragma once

#include <stdio.h>
#include <iterator>

#include "dispatch_reduce_by_key.cuh"
#include "../../agent/agent_seg_reduce.cuh"
#include "../../util_type.cuh"
#include "../../util_debug.cuh"
#include "../../util_device.cuh"
#include "../../thread/thread_search.cuh"
#include "../../grid/grid_queue.cuh"
#include "../../util_namespace.cuh"

/// Optional outer namespace(s)
CUB_NS_PREFIX

/// CUB namespace
namespace cub {


/******************************************************************************
 * Segmented reduction kernel entry points
 *****************************************************************************/

/**
 * SegReduce search kernel. Identifies merge path starting coordinates for each tile.
 */
template <
    typename    SegReducePolicyT,                   ///< Parameterized SegReducePolicy tuning policy type
    typename    ScanTileStateT,                     ///< Tile status interface type
    typename    OffsetT,                            ///< Signed integer type for sequence offsets
    typename    CoordinateT,                        ///< Merge path coordinate type
    typename    SegReduceParamsT>                   ///< SegReduceParams type
__global__ void DeviceSegReduceSearchKernel(
    ScanTileStateT      tile_state,                 ///< [in] Tile status interface for fixup reduce-by-key kernel
    int                 num_merge_tiles,            ///< [in] Number of Segmented reduction merge tiles (seg_reduce grid size)
    int                 num_reduce_by_key_tiles,    ///< [in] Number of reduce-by-key tiles (fixup grid size)
    CoordinateT*        d_tile_coordinates,         ///< [out] Pointer to the temporary array of tile starting coordinates
    SegReduceParamsT    seg_reduce_params)          ///< [in] Segmented reduction input parameter bundle
{
    /// Constants
    enum
    {
        BLOCK_THREADS           = SegReducePolicyT::BLOCK_THREADS,
        ITEMS_PER_THREAD        = SegReducePolicyT::ITEMS_PER_THREAD,
        TILE_ITEMS              = BLOCK_THREADS * ITEMS_PER_THREAD,
    };

    typedef CacheModifiedInputIterator<
            SegReducePolicyT::SEARCH_ROW_OFFSETS_LOAD_MODIFIER,
            OffsetT,
            OffsetT>
        RowOffsetsIteratorT;

    // Initialize tile status
    tile_state.InitializeStatus(num_reduce_by_key_tiles);

    // Find the starting coordinate for all tiles (plus the end coordinate of the last one)
    int tile_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (tile_idx < num_merge_tiles + 1)
    {
        OffsetT                         diagonal = (tile_idx * TILE_ITEMS);
        CoordinateT                     tile_coordinate;
        CountingInputIterator<OffsetT>  nonzero_indices(0);

        // Search the merge path
        MergePathSearch(
            diagonal,
            RowOffsetsIteratorT(seg_reduce_params.d_matrix_row_end_offsets),
            nonzero_indices,
            seg_reduce_params.num_rows,
            seg_reduce_params.num_nonzeros,
            tile_coordinate);

        // Output starting offset
        d_tile_coordinates[tile_idx] = tile_coordinate;
    }
}


/**
 * SegReduce agent entry point
 */
template <
    typename SegReducePolicyT,       ///< Parameterized SegReducePolicy tuning policy type
    typename ValueT,                 ///< Matrix and vector value type
    typename OffsetT,                ///< Signed integer type for sequence offsets
    typename ReductionOpT,           ///< Binary reduction functor type having member <tt>T operator()(const T &a, const T &b)</tt> (e.g., cub::Sum, cub::Min, cub::Max, etc.)
    typename CoordinateT>            ///< Merge path coordinate type
__launch_bounds__ (int(SegReducePolicyT::BLOCK_THREADS))
__global__ void DeviceSegReduceKernel(
    SegReduceParams<ValueT, OffsetT, ReductionOpT>  seg_reduce_params,      ///< [in] Segmented reduction input parameter bundle
    CoordinateT*                                    d_tile_coordinates,     ///< [in] Pointer to the temporary array of tile starting coordinates
    OffsetT*                                        d_tile_carry_rows,      ///< [out] Pointer to the temporary array carry-out dot product row-ids, one per block
    ValueT*                                         d_tile_carry_values,    ///< [out] Pointer to the temporary array carry-out dot product partial-sums, one per block
    int                                             num_tiles)              ///< [in] Number of merge tiles
{
    // SegReduce agent type specialization
    typedef AgentSegReduce<
            SegReducePolicyT,
            ValueT,
            OffsetT,
            ReductionOpT>
        AgentSegReduceT;

    // Shared memory for AgentSegReduce
    __shared__ typename AgentSegReduceT::TempStorage temp_storage;

    AgentSegReduceT(temp_storage, seg_reduce_params).ConsumeTile(
        d_tile_coordinates,
        d_tile_carry_rows,
        d_tile_carry_values,
        num_tiles);
}



/******************************************************************************
 * Dispatch
 ******************************************************************************/

/**
 * Utility class for dispatching the appropriately-tuned kernels for DeviceSegReduce
 */
template <
    typename ValueT,            ///< Matrix and vector value type
    typename OffsetT,           ///< Signed integer type for global offsets
    typename ReductionOpT>      ///< Binary reduction functor type having member <tt>T operator()(const T &a, const T &b)</tt> (e.g., cub::Sum, cub::Min, cub::Max, etc.)
struct DispatchSegReduce
{
    //---------------------------------------------------------------------
    // Constants and Types
    //---------------------------------------------------------------------

    enum
    {
        INIT_KERNEL_THREADS = 128
    };

    // SegReduceParams bundle type
    typedef SegReduceParams<ValueT, OffsetT, ReductionOpT> SegReduceParamsT;

    // 2D merge path coordinate type
    typedef typename CubVector<OffsetT, 2>::Type CoordinateT;

    // Reduce-by-key fixup dispatch type
    typedef DispatchReduceByKey<OffsetT*, OffsetT*, ValueT*, ValueT*, OffsetT*, cub::Equality, cub::Sum, OffsetT>
        DispatchReduceByKeyT;

    // Reduce-by-key tile status descriptor interface type
    typedef typename DispatchReduceByKeyT::ScanTileStateT ScanTileStateT;


    //---------------------------------------------------------------------
    // Tuning policies
    //---------------------------------------------------------------------

    /// SM11
    struct Policy110 : DispatchReduceByKeyT::Policy110
    {
        typedef AgentSegReducePolicy<
                128,
                1,
                LOAD_DEFAULT,
                LOAD_DEFAULT,
                LOAD_DEFAULT,
                false,
                BLOCK_SCAN_WARP_SCANS>
            SegReducePolicyT;
    };

    /// SM20
    struct Policy200 : Policy110 {};

    /// SM30
    struct Policy300 : DispatchReduceByKeyT::Policy300
    {
        typedef AgentSegReducePolicy<
                128,
                7,
                LOAD_DEFAULT,
                LOAD_DEFAULT,
                LOAD_DEFAULT,
                false,
                BLOCK_SCAN_WARP_SCANS>
            SegReducePolicyT;
    };

    /// SM35
    struct Policy350
    {
        typedef AgentSegReducePolicy<
                128,
                7,
                LOAD_LDG,
                LOAD_CA,
                LOAD_LDG,
                false,
                BLOCK_SCAN_WARP_SCANS>
            SegReducePolicyT;

        typedef AgentReduceByKeyPolicy<
                128,
                4,
                BLOCK_LOAD_VECTORIZE,
                LOAD_LDG,
                BLOCK_SCAN_WARP_SCANS>
            ReduceByKeyPolicyT;
    };

    /// SM50
    struct Policy500
    {
        typedef AgentSegReducePolicy<
                (sizeof(ValueT) > 4) ? 64 : 128,
                7,
                LOAD_LDG,
                LOAD_CA,
                LOAD_LDG,
                false,
                BLOCK_SCAN_RAKING_MEMOIZE>
//                (sizeof(ValueT) > 4) ? BLOCK_SCAN_WARP_SCANS : BLOCK_SCAN_RAKING_MEMOIZE>
            SegReducePolicyT;

        typedef AgentReduceByKeyPolicy<
                128,
                4,
                BLOCK_LOAD_VECTORIZE,
                LOAD_LDG,
                BLOCK_SCAN_WARP_SCANS>
            ReduceByKeyPolicyT;
    };



    //---------------------------------------------------------------------
    // Tuning policies of current PTX compiler pass
    //---------------------------------------------------------------------

#if (CUB_PTX_ARCH >= 500)
    typedef Policy500 PtxPolicy;

#elif (CUB_PTX_ARCH >= 350)
    typedef Policy350 PtxPolicy;

#elif (CUB_PTX_ARCH >= 300)
    typedef Policy300 PtxPolicy;

#elif (CUB_PTX_ARCH >= 200)
    typedef Policy200 PtxPolicy;

#else
    typedef Policy110 PtxPolicy;

#endif

    // "Opaque" policies (whose parameterizations aren't reflected in the type signature)
    struct PtxSegReducePolicyT : PtxPolicy::SegReducePolicyT {};
    struct PtxReduceByKeyPolicy : PtxPolicy::ReduceByKeyPolicyT {};


    //---------------------------------------------------------------------
    // Utilities
    //---------------------------------------------------------------------

    /**
     * Initialize kernel dispatch configurations with the policies corresponding to the PTX assembly we will use
     */
    template <typename KernelConfig>
    CUB_RUNTIME_FUNCTION __forceinline__
    static void InitConfigs(
        int             ptx_version,
        KernelConfig    &seg_reduce_config,
        KernelConfig    &reduce_by_key_config)
    {
    #if (CUB_PTX_ARCH > 0)

        // We're on the device, so initialize the kernel dispatch configurations with the current PTX policy
        seg_reduce_config.template Init<PtxSegReducePolicyT>();
        reduce_by_key_config.template Init<PtxReduceByKeyPolicy>();

    #else

        // We're on the host, so lookup and initialize the kernel dispatch configurations with the policies that match the device's PTX version
        if (ptx_version >= 500)
        {
            seg_reduce_config.template      Init<typename Policy500::SegReducePolicyT>();
            reduce_by_key_config.template   Init<typename Policy500::ReduceByKeyPolicyT>();
        }
        else if (ptx_version >= 350)
        {
            seg_reduce_config.template      Init<typename Policy350::SegReducePolicyT>();
            reduce_by_key_config.template   Init<typename Policy350::ReduceByKeyPolicyT>();
        }
        else if (ptx_version >= 300)
        {
            seg_reduce_config.template      Init<typename Policy300::SegReducePolicyT>();
            reduce_by_key_config.template   Init<typename Policy300::ReduceByKeyPolicyT>();

        }
        else if (ptx_version >= 200)
        {
            seg_reduce_config.template      Init<typename Policy200::SegReducePolicyT>();
            reduce_by_key_config.template   Init<typename Policy200::ReduceByKeyPolicyT>();
        }
        else
        {
            seg_reduce_config.template      Init<typename Policy110::SegReducePolicyT>();
            reduce_by_key_config.template   Init<typename Policy110::ReduceByKeyPolicyT>();
        }

    #endif
    }


    /**
     * Kernel kernel dispatch configuration.
     */
    struct KernelConfig
    {
        int block_threads;
        int items_per_thread;
        int tile_items;

        template <typename PolicyT>
        CUB_RUNTIME_FUNCTION __forceinline__
        void Init()
        {
            block_threads       = PolicyT::BLOCK_THREADS;
            items_per_thread    = PolicyT::ITEMS_PER_THREAD;
            tile_items          = block_threads * items_per_thread;
        }
    };


    //---------------------------------------------------------------------
    // Dispatch entrypoints
    //---------------------------------------------------------------------

    /**
     * Internal dispatch routine for computing a device-wide reduction using the
     * specified kernel functions.
     *
     * If the input is larger than a single tile, this method uses two-passes of
     * kernel invocations.
     */
    template <
        typename                SegReduceSearchKernelT,         ///< Function type of cub::AgentSegReduceSearchKernel
        typename                SegReduceKernelT,               ///< Function type of cub::AgentSegReduceKernel
        typename                ReduceByKeyKernelT>             ///< Function type of cub::DeviceReduceByKeyKernelT
    CUB_RUNTIME_FUNCTION __forceinline__
    static cudaError_t Dispatch(
        void*                   d_temp_storage,                 ///< [in] %Device allocation of temporary storage.  When NULL, the required allocation size is written to \p temp_storage_bytes and no work is done.
        size_t&                 temp_storage_bytes,             ///< [in,out] Reference to size in bytes of \p d_temp_storage allocation
        SegReduceParamsT&       seg_reduce_params,              ///< Segmented reduction input parameter bundle
        cudaStream_t            stream,                         ///< [in] CUDA stream to launch kernels within.  Default is stream<sub>0</sub>.
        bool                    debug_synchronous,              ///< [in] Whether or not to synchronize the stream after every kernel launch to check for errors.  Also causes launch configurations to be printed to the console.  Default is \p false.
        SegReduceSearchKernelT  seg_reduce_search_kernel,       ///< [in] Kernel function pointer to parameterization of AgentSegReduceSearchKernel
        SegReduceKernelT        seg_reduce_kernel,              ///< [in] Kernel function pointer to parameterization of AgentSegReduceKernel
        ReduceByKeyKernelT      reduce_by_key_kernel,           ///< [in] Kernel function pointer to parameterization of cub::DeviceReduceByKeyKernel
        KernelConfig            seg_reduce_config,              ///< [in] Dispatch parameters that match the policy that \p seg_reduce_kernel was compiled for
        KernelConfig            reduce_by_key_config)           ///< [in] Dispatch parameters that match the policy that \p reduce_by_key_kernel was compiled for
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

            // Get max x-dimension of grid
            int max_dim_x;
            if (CubDebug(error = cudaDeviceGetAttribute(&max_dim_x, cudaDevAttrMaxGridDimX, device_ordinal))) break;;

            // Total number of seg_reduce work items
            int num_merge_items = seg_reduce_params.num_rows + seg_reduce_params.num_nonzeros;

            // Tile sizes of kernels
            int merge_tile_size        = seg_reduce_config.block_threads * seg_reduce_config.items_per_thread;
            int reduce_by_key_tile_size     = reduce_by_key_config.block_threads * reduce_by_key_config.items_per_thread;

            // Number of tiles for kernels
            unsigned int num_merge_tiles                = (num_merge_items + merge_tile_size - 1) / merge_tile_size;
            unsigned int num_reduce_by_key_tiles    = (num_merge_tiles + reduce_by_key_tile_size - 1) / reduce_by_key_tile_size;

            // Get SM occupancy for kernels
            int seg_reduce_sm_occupancy;
            if (CubDebug(error = MaxSmOccupancy(
                seg_reduce_sm_occupancy,
                sm_version,
                seg_reduce_kernel,
                seg_reduce_config.block_threads))) break;

            int reduce_by_key_sm_occupancy;
            if (CubDebug(error = MaxSmOccupancy(
                reduce_by_key_sm_occupancy,
                sm_version,
                reduce_by_key_kernel,
                reduce_by_key_config.block_threads))) break;

            // Get grid dimensions
            dim3 seg_reduce_grid_size(
                CUB_MIN(num_merge_tiles, max_dim_x),
                (num_merge_tiles + max_dim_x - 1) / max_dim_x,
                1);

            dim3 reduce_by_key_grid_size(
                CUB_MIN(num_reduce_by_key_tiles, max_dim_x),
                (num_reduce_by_key_tiles + max_dim_x - 1) / max_dim_x,
                1);

            // Get the temporary storage allocation requirements
            size_t allocation_sizes[4];
            if (CubDebug(error = ScanTileStateT::AllocationSize(num_reduce_by_key_tiles, allocation_sizes[0]))) break;    // bytes needed for reduce-by-key tile status descriptors
            allocation_sizes[1] = num_merge_tiles * sizeof(OffsetT);             // bytes needed for block run-out row-ids
            allocation_sizes[2] = num_merge_tiles * sizeof(ValueT);              // bytes needed for block run-out partials sums
            allocation_sizes[3] = (num_merge_tiles + 1) * sizeof(CoordinateT);   // bytes needed for tile starting coordinates

            // Alias the temporary allocations from the single storage blob (or compute the necessary size of the blob)
            void* allocations[4];
            if (CubDebug(error = AliasTemporaries(d_temp_storage, temp_storage_bytes, allocations, allocation_sizes))) break;
            if (d_temp_storage == NULL)
            {
                // Return if the caller is simply requesting the size of the storage allocation
                return cudaSuccess;
            }

            // Construct the tile status interface
            ScanTileStateT tile_state;
            if (CubDebug(error = tile_state.Init(num_reduce_by_key_tiles, allocations[0], allocation_sizes[0]))) break;

            // Alias the other allocations
            OffsetT*        d_tile_carry_rows       = (OffsetT*) allocations[1];        // Agent carry-out row-ids
            ValueT*         d_tile_carry_values     = (ValueT*) allocations[2];         // Agent carry-out partial sums
            CoordinateT*    d_tile_coordinates      = (CoordinateT*) allocations[3];    // Agent starting coordinates

            // Get search/init grid dims
            int search_block_size   = INIT_KERNEL_THREADS;
            int search_grid_size    = (num_merge_tiles + 1 + search_block_size - 1) / search_block_size;

            // Log seg_reduce_search_kernel configuration
            if (debug_synchronous) CubLog("Invoking seg_reduce_search_kernel<<<%d, %d, 0, %lld>>>()\n",
                search_grid_size, search_block_size, (long long) stream);

            // Invoke seg_reduce_search_kernel
            seg_reduce_search_kernel<<<search_grid_size, search_block_size, 0, stream>>>(
                tile_state,
                num_merge_tiles,
                num_reduce_by_key_tiles,
                d_tile_coordinates,
                seg_reduce_params);

            // Log seg_reduce_kernel configuration
            if (debug_synchronous) CubLog("Invoking seg_reduce_kernel<<<{%d,%d,%d}, %d, 0, %lld>>>(), %d items per thread, %d SM occupancy\n",
                seg_reduce_grid_size.x, seg_reduce_grid_size.y, seg_reduce_grid_size.z, seg_reduce_config.block_threads, (long long) stream, seg_reduce_config.items_per_thread, seg_reduce_sm_occupancy);

            // Invoke seg_reduce_kernel
            seg_reduce_kernel<<<seg_reduce_grid_size, seg_reduce_config.block_threads, 0, stream>>>(
                seg_reduce_params,
                d_tile_coordinates,
                d_tile_carry_rows,
                d_tile_carry_values,
                num_merge_tiles);

            // Check for failure to launch
            if (CubDebug(error = cudaPeekAtLastError())) break;

            // Sync the stream if specified to flush runtime errors
            if (debug_synchronous && (CubDebug(error = SyncStream(stream)))) break;

            // Run reduce-by-key fixup if necessary
            if (num_merge_tiles > 1)
            {
                // Log reduce_by_key_kernel configuration
                if (debug_synchronous) CubLog("Invoking reduce_by_key_kernel<<<{%d,%d,%d}, %d, 0, %lld>>>(), %d items per thread, %d SM occupancy\n",
                    reduce_by_key_grid_size.x, reduce_by_key_grid_size.y, reduce_by_key_grid_size.z, reduce_by_key_config.block_threads, (long long) stream, reduce_by_key_config.items_per_thread, reduce_by_key_sm_occupancy);

                // Invoke reduce_by_key_kernel
                reduce_by_key_kernel<<<reduce_by_key_grid_size, reduce_by_key_config.block_threads, 0, stream>>>(
                    d_tile_carry_rows,
                    NULL,
                    d_tile_carry_values,
                    seg_reduce_params.d_vector_y,
                    NULL,
                    tile_state,
                    cub::Equality(),
                    cub::Sum(),
                    num_merge_tiles,
                    num_reduce_by_key_tiles);

                // Check for failure to launch
                if (CubDebug(error = cudaPeekAtLastError())) break;

                // Sync the stream if specified to flush runtime errors
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
    CUB_RUNTIME_FUNCTION __forceinline__
    static cudaError_t Dispatch(
        void*                   d_temp_storage,                     ///< [in] %Device allocation of temporary storage.  When NULL, the required allocation size is written to \p temp_storage_bytes and no work is done.
        size_t&                 temp_storage_bytes,                 ///< [in,out] Reference to size in bytes of \p d_temp_storage allocation
        SegReduceParamsT&            seg_reduce_params,                        ///< Segmented reduction input parameter bundle
        cudaStream_t            stream                  = 0,        ///< [in] <b>[optional]</b> CUDA stream to launch kernels within.  Default is stream<sub>0</sub>.
        bool                    debug_synchronous       = false)    ///< [in] <b>[optional]</b> Whether or not to synchronize the stream after every kernel launch to check for errors.  May cause significant slowdown.  Default is \p false.
    {
        cudaError error = cudaSuccess;
        do
        {
            // Get PTX version
            int ptx_version;
    #if (CUB_PTX_ARCH == 0)
            if (CubDebug(error = PtxVersion(ptx_version))) break;
    #else
            ptx_version = CUB_PTX_ARCH;
    #endif

            // Get kernel kernel dispatch configurations
            KernelConfig seg_reduce_config, reduce_by_key_config;
            InitConfigs(ptx_version, seg_reduce_config, reduce_by_key_config);
/*
            // Dispatch
            if (seg_reduce_params.beta == 0.0)
            {
                if (seg_reduce_params.alpha == 1.0)
                {
*/
                    // Dispatch y = A*x
                    if (CubDebug(error = Dispatch(
                        d_temp_storage, temp_storage_bytes, seg_reduce_params, stream, debug_synchronous,
                        DeviceSegReduceSearchKernel<PtxSegReducePolicyT, ScanTileStateT, OffsetT, CoordinateT, SegReduceParamsT>,
                        DeviceSegReduceKernel<PtxSegReducePolicyT, ValueT, OffsetT, CoordinateT, false, false>,
                        DeviceReduceByKeyKernel<PtxReduceByKeyPolicy, OffsetT*, OffsetT*, ValueT*, ValueT*, OffsetT*, ScanTileStateT, cub::Equality, cub::Sum, OffsetT, true>,
                        seg_reduce_config, reduce_by_key_config))) break;
/*
                }
                else
                {
                    // Dispatch y = alpha*A*x
                    if (CubDebug(error = Dispatch(
                        d_temp_storage, temp_storage_bytes, seg_reduce_params, stream, debug_synchronous,
                        DeviceSegReduceSearchKernel<PtxSegReducePolicyT, ScanTileStateT, OffsetT, CoordinateT, SegReduceParamsT>,
                        DeviceSegReduceKernel<PtxSegReducePolicyT, ValueT, OffsetT, CoordinateT, true, false>,
                        DeviceReduceByKeyKernel<PtxReduceByKeyPolicy, OffsetT*, OffsetT*, ValueT*, ValueT*, OffsetT*, ScanTileStateT, cub::Equality, cub::Sum, OffsetT, true>,
                        seg_reduce_config, reduce_by_key_config))) break;
                }
            }
            else
            {
                if (seg_reduce_params.alpha == 1.0)
                {
                    // Dispatch y = A*x + beta*y
                    if (CubDebug(error = Dispatch(
                        d_temp_storage, temp_storage_bytes, seg_reduce_params, stream, debug_synchronous,
                        DeviceSegReduceSearchKernel<PtxSegReducePolicyT, ScanTileStateT, OffsetT, CoordinateT, SegReduceParamsT>,
                        DeviceSegReduceKernel<PtxSegReducePolicyT, ValueT, OffsetT, CoordinateT, false, true>,
                        DeviceReduceByKeyKernel<PtxReduceByKeyPolicy, OffsetT*, OffsetT*, ValueT*, ValueT*, OffsetT*, ScanTileStateT, cub::Equality, cub::Sum, OffsetT, true>,
                        seg_reduce_config, reduce_by_key_config))) break;
                }
                else
                {
                    // Dispatch y = alpha*A*x + beta*y
                    if (CubDebug(error = Dispatch(
                        d_temp_storage, temp_storage_bytes, seg_reduce_params, stream, debug_synchronous,
                        DeviceSegReduceSearchKernel<PtxSegReducePolicyT, ScanTileStateT, OffsetT, CoordinateT, SegReduceParamsT>,
                        DeviceSegReduceKernel<PtxSegReducePolicyT, ValueT, OffsetT, CoordinateT, true, true>,
                        DeviceReduceByKeyKernel<PtxReduceByKeyPolicy, OffsetT*, OffsetT*, ValueT*, ValueT*, OffsetT*, ScanTileStateT, cub::Equality, cub::Sum, OffsetT, true>,
                        seg_reduce_config, reduce_by_key_config))) break;
                }
            }
*/
        }
        while (0);

        return error;
    }
};


}               // CUB namespace
CUB_NS_POSTFIX  // Optional outer namespace(s)


