
/******************************************************************************
 * Copyright (c) 2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2014, NVIDIA CORPORATION.  All rights reserved.
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
 * cub::DeviceSpmv provides device-wide parallel operations for performing sparse-matrix * vector multiplication (SpMV).
 */

#pragma once

#include <stdio.h>
#include <iterator>
#include <limits>

#include "../../agent/agent_spmv.cuh"
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
 * SpMV kernel entry points
 *****************************************************************************/

/**
 * Spmv search kernel. Identifies merge path starting coordinates for each tile.
 */
template <
    typename    AgentSpmvPolicyT,                   ///< Parameterized AgentSpmvPolicy tuning policy type
    typename    OffsetT,                            ///< Signed integer type for sequence offsets
    typename    CoordinateT>                        ///< Merge path coordinate type
__global__ void AgentSpmvSearchKernel(
    int             agent_spmv_grid_size,
    OffsetT*        d_matrix_row_end_offsets,       ///< [in] Pointer to the array of \p m offsets demarcating the end of every row in \p d_matrix_column_indices and \p d_matrix_values
    CoordinateT*    d_tile_coordinates,             ///< [out] Pointer to the temporary array of tile starting coordinates
    int             num_rows,                       ///< [in] number of rows of matrix <b>A</b>.
    int             num_nonzeros)                   ///< [in] number of nonzero elements of matrix <b>A</b>.
{
    /// Constants
    enum
    {
        BLOCK_THREADS           = AgentSpmvPolicyT::BLOCK_THREADS,
        ITEMS_PER_THREAD        = AgentSpmvPolicyT::ITEMS_PER_THREAD,
        TILE_ITEMS              = BLOCK_THREADS * ITEMS_PER_THREAD,
    };

    typedef CacheModifiedInputIterator<
            AgentSpmvPolicyT::MATRIX_ROW_OFFSETS_LOAD_MODIFIER,
            OffsetT,
            OffsetT>
        MatrixRowOffsetsIteratorT;

    // Find the starting coordinate for all tiles (plus the end coordinate of the last one)
    int tile_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (tile_idx < agent_spmv_grid_size + 1)
    {
        OffsetT                         diagonal = (tile_idx * TILE_ITEMS);
        CoordinateT                     tile_coordinate;
        CountingInputIterator<OffsetT>  nonzero_indices(0);

        // Search the merge path
        MergePathSearch(
            diagonal,
            MatrixRowOffsetsIteratorT(d_matrix_row_end_offsets),
            nonzero_indices,
            num_rows,
            num_nonzeros,
            tile_coordinate);

        // Output starting offset
        d_tile_coordinates[tile_idx] = tile_coordinate;
    }
}


/**
 * Spmv agent entry point
 */
template <
    typename    AgentSpmvPolicyT,           ///< Parameterized AgentSpmvPolicy tuning policy type
    typename    ValueT,                     ///< Matrix and vector value type
    typename    OffsetT,                    ///< Signed integer type for sequence offsets
    typename    CoordinateT>                ///< Merge path coordinate type
__launch_bounds__ (int(AgentSpmvPolicyT::BLOCK_THREADS))
__global__ void AgentSpmvKernel(
    ValueT*         d_matrix_values,                ///< [in] Pointer to the array of \p num_nonzeros values of the corresponding nonzero elements of matrix <b>A</b>.
    OffsetT*        d_matrix_row_end_offsets,       ///< [in] Pointer to the array of \p m offsets demarcating the end of every row in \p d_matrix_column_indices and \p d_matrix_values
    OffsetT*        d_matrix_column_indices,        ///< [in] Pointer to the array of \p num_nonzeros column-indices of the corresponding nonzero elements of matrix <b>A</b>.  (Indices are zero-valued.)
    ValueT*         d_vector_x,                     ///< [in] Pointer to the array of \p num_cols values corresponding to the dense input vector <em>x</em>
    ValueT*         d_vector_y,                     ///< [out] Pointer to the array of \p num_rows values corresponding to the dense output vector <em>y</em>
    CoordinateT*    d_tile_coordinates,             ///< [in] Pointer to the temporary array of tile starting coordinates
    OffsetT*        d_tile_carry_rows,          ///< [out] Pointer to the temporary array carry-out dot product row-ids, one per block
    ValueT*         d_tile_carry_values,        ///< [out] Pointer to the temporary array carry-out dot product partial-sums, one per block
    int             num_rows,                       ///< [in] number of rows of matrix <b>A</b>.
    int             num_cols,                       ///< [in] number of columns of matrix <b>A</b>.
    int             num_nonzeros)                   ///< [in] number of nonzero elements of matrix <b>A</b>.
{
    // Spmv agent type specialization
    typedef AgentSpmv<
            AgentSpmvPolicyT,
            ValueT,
            OffsetT,
            CoordinateT>
        AgentSpmvT;

    // Shared memory for AgentSpmv
    __shared__ typename AgentSpmvT::TempStorage temp_storage;

    AgentSpmvT agent(
        temp_storage,
        d_matrix_values,
        d_matrix_row_end_offsets,
        d_matrix_column_indices,
        d_vector_x,
        d_vector_y,
        d_tile_carry_rows,
        d_tile_carry_values,
        num_rows,
        num_cols,
        num_nonzeros);

    agent.ConsumeTile(d_tile_coordinates);
}



/******************************************************************************
 * Dispatch
 ******************************************************************************/

/**
 * Utility class for dispatching the appropriately-tuned kernels for DeviceSpmv
 */
template <
    typename    ValueT,                     ///< Matrix and vector value type
    typename    OffsetT>                    ///< Signed integer type for global offsets
struct DispatchSpmv
{
    //---------------------------------------------------------------------
    // Types
    //---------------------------------------------------------------------

    // 2D merge path coordinate type
    typedef typename CubVector<OffsetT, 2>::Type CoordinateT;


    //---------------------------------------------------------------------
    // Tuning policies
    //---------------------------------------------------------------------

    /// SM11
    struct Policy110
    {
        typedef AgentSpmvPolicy<
                128,
                7,
                LOAD_DEFAULT,
                LOAD_DEFAULT,
                LOAD_DEFAULT,
                LOAD_DEFAULT>
            AgentSpmvPolicy;
    };

    /// SM20
    struct Policy200 : Policy110 {};

    /// SM30
    struct Policy300 : Policy110 {};

    /// SM35
    struct Policy350
    {
        typedef AgentSpmvPolicy<
                128,
                5,
                LOAD_LDG,
                LOAD_DEFAULT,
                LOAD_DEFAULT,
                LOAD_LDG>
            AgentSpmvPolicy;
    };

    /// SM50
    struct Policy500
    {
        typedef AgentSpmvPolicy<
                (sizeof(ValueT) > 4) ? 64 : 128,
                7,
                LOAD_LDG,
                LOAD_DEFAULT,
                LOAD_DEFAULT,
                LOAD_LDG>
            AgentSpmvPolicy;
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
    struct PtxAgentSpmvPolicy : PtxPolicy::AgentSpmvPolicy {};


    //---------------------------------------------------------------------
    // Utilities
    //---------------------------------------------------------------------

    /**
     * Initialize kernel dispatch configurations with the policies corresponding to the PTX assembly we will use
     */
    template <typename KernelConfig>
    CUB_RUNTIME_FUNCTION __forceinline__
    static cudaError_t InitConfigs(
        int             ptx_version,
        KernelConfig    &agent_spmv_config)
    {
    #if (CUB_PTX_ARCH > 0)

        // We're on the device, so initialize the kernel dispatch configurations with the current PTX policy
        return agent_spmv_config.template Init<PtxAgentSpmvPolicy>();

    #else

        // We're on the host, so lookup and initialize the kernel dispatch configurations with the policies that match the device's PTX version
        if (ptx_version >= 500)
        {
            return agent_spmv_config.template Init<typename Policy500::AgentSpmvPolicy>();
        }
        else if (ptx_version >= 350)
        {
            return agent_spmv_config.template Init<typename Policy350::AgentSpmvPolicy>();
        }
        else if (ptx_version >= 300)
        {
            return agent_spmv_config.template Init<typename Policy300::AgentSpmvPolicy>();
        }
        else if (ptx_version >= 200)
        {
            return agent_spmv_config.template Init<typename Policy200::AgentSpmvPolicy>();
        }
        else if (ptx_version >= 110)
        {
            return agent_spmv_config.template Init<typename Policy110::AgentSpmvPolicy>();
        }
        else
        {
            // No global atomic support
            return cudaErrorNotSupported;
        }

    #endif
    }


    /**
     * Kernel kernel dispatch configuration
     */
    struct KernelConfig
    {
        int block_threads;
        int items_per_thread;

        template <typename AgentPolicyT>
        CUB_RUNTIME_FUNCTION __forceinline__
        cudaError_t Init()
        {
            block_threads       = AgentPolicyT::BLOCK_THREADS;
            items_per_thread    = AgentPolicyT::ITEMS_PER_THREAD;

            return cudaSuccess;
        }

        CUB_RUNTIME_FUNCTION __forceinline__
        void Print()
        {
            printf("%d, %d", block_threads, items_per_thread);
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
        typename            AgentSpmvSearchKernelT,                 ///< Function type of cub::AgentSpmvSearchKernel
        typename            AgentSpmvKernelT>                       ///< Function type of cub::AgentSpmvKernel
    CUB_RUNTIME_FUNCTION __forceinline__
    static cudaError_t Dispatch(
        void*                   d_temp_storage,                     ///< [in] %Device allocation of temporary storage.  When NULL, the required allocation size is written to \p temp_storage_bytes and no work is done.
        size_t&                 temp_storage_bytes,                 ///< [in,out] Reference to size in bytes of \p d_temp_storage allocation
        ValueT*                 d_matrix_values,                    ///< [in] Pointer to the array of \p num_nonzeros values of the corresponding nonzero elements of matrix <b>A</b>.
        OffsetT*                d_matrix_row_offsets,               ///< [in] Pointer to the array of \p m + 1 offsets demarcating the start of every row in \p d_matrix_column_indices and \p d_matrix_values (with the final entry being equal to \p num_nonzeros)
        OffsetT*                d_matrix_column_indices,            ///< [in] Pointer to the array of \p num_nonzeros column-indices of the corresponding nonzero elements of matrix <b>A</b>.  (Indices are zero-valued.)
        ValueT*                 d_vector_x,                         ///< [in] Pointer to the array of \p num_cols values corresponding to the dense input vector <em>x</em>
        ValueT*                 d_vector_y,                         ///< [out] Pointer to the array of \p num_rows values corresponding to the dense output vector <em>y</em>
        int                     num_rows,                           ///< [in] number of rows of matrix <b>A</b>.
        int                     num_cols,                           ///< [in] number of columns of matrix <b>A</b>.
        int                     num_nonzeros,                       ///< [in] number of nonzero elements of matrix <b>A</b>.
        cudaStream_t            stream,                             ///< [in] CUDA stream to launch kernels within.  Default is stream<sub>0</sub>.
        bool                    debug_synchronous,                  ///< [in] Whether or not to synchronize the stream after every kernel launch to check for errors.  Also causes launch configurations to be printed to the console.  Default is \p false.
        AgentSpmvSearchKernelT  agent_spmv_search_kernel,           ///< [in] Kernel function pointer to parameterization of AgentSpmvSearchKernel
        AgentSpmvKernelT        agent_spmv_kernel,                  ///< [in] Kernel function pointer to parameterization of AgentSpmvKernel
        KernelConfig            agent_spmv_config)                  ///< [in] Dispatch parameters that match the policy that \p agent_spmv_kernel was compiled for
    {
#ifndef CUB_RUNTIME_ENABLED

        // Kernel launch not supported from this device
        return CubDebug(cudaErrorNotSupported );

#else
        cudaError error = cudaSuccess;
        do
        {
            // Row end offsets
            OffsetT* d_matrix_row_end_offsets = d_matrix_row_offsets + 1;

            // Get device ordinal
            int device_ordinal;
            if (CubDebug(error = cudaGetDevice(&device_ordinal))) break;

            // Get device SM version
            int sm_version;
            if (CubDebug(error = SmVersion(sm_version, device_ordinal))) break;

            // Get SM count
            int sm_count;
            if (CubDebug(error = cudaDeviceGetAttribute (&sm_count, cudaDevAttrMultiProcessorCount, device_ordinal))) break;

            // Tile size of agent_spmv_kernel
            int tile_size = agent_spmv_config.block_threads * agent_spmv_config.items_per_thread;

            // Total number of work items
            int work_items = num_rows + num_nonzeros;

            // Get SM occupancy for agent_spmv_kernel
            int agent_spmv_sm_occupancy;
            if (CubDebug(error = MaxSmOccupancy(
                agent_spmv_sm_occupancy,
                sm_version,
                agent_spmv_kernel,
                agent_spmv_config.block_threads))) break;

            // Get device occupancy for agent_spmv_kernel
//            int agent_spmv_occupancy = agent_spmv_sm_occupancy * sm_count;

            // Get grid size for agent_spmv_kernel
            int agent_spmv_grid_size = (work_items + tile_size - 1) / tile_size;

            // Temporary storage allocation requirements
            void* allocations[4];
            size_t allocation_sizes[4] =
            {
                agent_spmv_grid_size * sizeof(OffsetT),             // bytes needed for block run-out row-ids
                agent_spmv_grid_size * sizeof(ValueT),              // bytes needed for block run-out partials sums
                (agent_spmv_grid_size + 1) * sizeof(CoordinateT),   // bytes needed for tile starting coordinates
                GridQueue<int>::AllocationSize()                    // bytes needed for grid queue descriptor
            };

            // Alias the temporary allocations from the single storage blob (or compute the necessary size of the blob)
            if (CubDebug(error = AliasTemporaries(d_temp_storage, temp_storage_bytes, allocations, allocation_sizes))) break;
            if (d_temp_storage == NULL)
            {
                // Return if the caller is simply requesting the size of the storage allocation
                return cudaSuccess;
            }

            // Alias the allocations
            OffsetT*        d_tile_carry_rows       = (OffsetT*) allocations[0];        // Agent carry-out row-ids
            ValueT*         d_tile_carry_values     = (ValueT*) allocations[1];         // Agent carry-out partial sums
            CoordinateT*    d_tile_coordinates         = (CoordinateT*) allocations[2];    // Agent starting coordinates

            // Alias the allocation for the grid queue descriptor
            GridQueue<OffsetT> queue(allocations[3]);

            int search_block_size = 128;
            int search_grid_size = (agent_spmv_grid_size + 1 + search_block_size - 1) / search_block_size;

            // Log agent_spmv_search_kernel configuration
            if (debug_synchronous) CubLog("Invoking agent_spmv_search_kernel<<<%d, %d, 0, %lld>>>()\n",
                search_grid_size, search_block_size, (long long) stream);

            // Invoke agent_spmv_search_kernel
            agent_spmv_search_kernel<<<search_grid_size, search_block_size, 0, stream>>>(
                agent_spmv_grid_size,
                d_matrix_row_end_offsets,
                d_tile_coordinates,
                num_rows,
                num_nonzeros);

            // Log agent_spmv_kernel configuration
            if (debug_synchronous) CubLog("Invoking agent_spmv_kernel<<<%d, %d, 0, %lld>>>(), %d items per thread, %d SM occupancy\n",
                agent_spmv_grid_size, agent_spmv_config.block_threads, (long long) stream, agent_spmv_config.items_per_thread, agent_spmv_sm_occupancy);

            // Invoke agent_spmv_kernel
            agent_spmv_kernel<<<agent_spmv_grid_size, agent_spmv_config.block_threads, 0, stream>>>(
                d_matrix_values,
                d_matrix_row_end_offsets,
                d_matrix_column_indices,
                d_vector_x,
                d_vector_y,
                d_tile_coordinates,
                d_tile_carry_rows,
                d_tile_carry_values,
                num_rows,
                num_cols,
                num_nonzeros);

            // Check for failure to launch
            if (CubDebug(error = cudaPeekAtLastError())) break;

            // Sync the stream if specified to flush runtime errors
            if (debug_synchronous && (CubDebug(error = SyncStream(stream)))) break;

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
        void*               d_temp_storage,                     ///< [in] %Device allocation of temporary storage.  When NULL, the required allocation size is written to \p temp_storage_bytes and no work is done.
        size_t&             temp_storage_bytes,                 ///< [in,out] Reference to size in bytes of \p d_temp_storage allocation
        ValueT*             d_matrix_values,                    ///< [in] Pointer to the array of \p num_nonzeros values of the corresponding nonzero elements of matrix <b>A</b>.
        OffsetT*            d_matrix_row_offsets,               ///< [in] Pointer to the array of \p m + 1 offsets demarcating the start of every row in \p d_matrix_column_indices and \p d_matrix_values (with the final entry being equal to \p num_nonzeros)
        OffsetT*            d_matrix_column_indices,            ///< [in] Pointer to the array of \p num_nonzeros column-indices of the corresponding nonzero elements of matrix <b>A</b>.  (Indices are zero-valued.)
        ValueT*             d_vector_x,                         ///< [in] Pointer to the array of \p num_cols values corresponding to the dense input vector <em>x</em>
        ValueT*             d_vector_y,                         ///< [out] Pointer to the array of \p num_rows values corresponding to the dense output vector <em>y</em>
        int                 num_rows,                           ///< [in] number of rows of matrix <b>A</b>.
        int                 num_cols,                           ///< [in] number of columns of matrix <b>A</b>.
        int                 num_nonzeros,                       ///< [in] number of nonzero elements of matrix <b>A</b>.
        cudaStream_t        stream                  = 0,        ///< [in] <b>[optional]</b> CUDA stream to launch kernels within.  Default is stream<sub>0</sub>.
        bool                debug_synchronous       = false)    ///< [in] <b>[optional]</b> Whether or not to synchronize the stream after every kernel launch to check for errors.  May cause significant slowdown.  Default is \p false.
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
            KernelConfig agent_spmv_config;
            InitConfigs(ptx_version, agent_spmv_config);

            // Dispatch
            if (CubDebug(error = Dispatch(
                d_temp_storage,
                temp_storage_bytes,
                d_matrix_values,
                d_matrix_row_offsets,
                d_matrix_column_indices,
                d_vector_x,
                d_vector_y,
                num_rows,
                num_cols,
                num_nonzeros,
                stream,
                debug_synchronous,
                AgentSpmvSearchKernel<PtxAgentSpmvPolicy, OffsetT, CoordinateT>,
                AgentSpmvKernel<PtxAgentSpmvPolicy, ValueT, OffsetT, CoordinateT>,
                agent_spmv_config))) break;
        }
        while (0);

        return error;
    }
};


}               // CUB namespace
CUB_NS_POSTFIX  // Optional outer namespace(s)


