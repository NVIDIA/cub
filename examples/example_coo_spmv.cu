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

/******************************************************************************
 * Experimental reduce-value-by-row COO implementation of SPMV
 ******************************************************************************/

// Ensure printing of CUDA runtime errors to console
#define CUB_STDERR

#include <iterator>
#include <vector>
#include <string>
#include <algorithm>
#include <stdio.h>

#include <cub/cub.cuh>

#include "coo_graph.cuh"
#include "../test/test_util.h"

using namespace cub;
using namespace std;


//---------------------------------------------------------------------
// Globals, constants and typedefs
//---------------------------------------------------------------------

bool    g_verbose       = false;
int     g_iterations    = 1;


//---------------------------------------------------------------------
// GPU types and device functions
//---------------------------------------------------------------------

/// Pairing of dot product partial sums and corresponding row-id
template <typename VertexId, typename Value>
struct PartialSum
{
    Value       partial;        /// PartialSum sum
    VertexId    row;            /// Row-id
};


/// Templated texture reference type for multiplicand vector
template <typename Value>
struct TexVector
{
    // Texture type to actually use (e.g., because CUDA doesn't load doubles as texture items)
    typedef typename If<(Equals<Value, double>::VALUE), uint2, Value>::Type CastType;

    // Texture reference type
    typedef texture<CastType, cudaTextureType1D, cudaReadModeElementType> TexRef;

    static TexRef ref;

    /**
     * Bind textures
     */
    static void BindTexture(void *d_in, int elements)
    {
        cudaChannelFormatDesc tex_desc = cudaCreateChannelDesc<CastType>();
        if (d_in)
        {
            size_t offset;
            size_t bytes = sizeof(CastType) * elements;
            CubDebugExit(cudaBindTexture(&offset, ref, d_in, tex_desc, bytes));
        }
    }

    /**
     * Unbind textures
     */
    static void UnbindTexture()
    {
        CubDebugExit(cudaUnbindTexture(ref));
    }

    /**
     * Load
     */
    static __device__ __forceinline__ Value Load(int offset)
    {
        Value output;
        reinterpret_cast<typename TexVector<Value>::CastType &>(output) = tex1Dfetch(TexVector<Value>::ref, offset);
        return output;
    }
};

// Texture reference definitions
template <typename Value>
typename TexVector<Value>::TexRef TexVector<Value>::ref = 0;


/// Reduce-by-row scan operator
struct ReduceByKeyOp
{
    template <typename PartialSum>
    __device__ __forceinline__ PartialSum operator()(
        const PartialSum &first,
        const PartialSum &second)
    {
        PartialSum retval;

        retval.partial = (second.row != first.row) ?
                second.partial :
                first.partial + second.partial;

        retval.row = second.row;
        return retval;
    }
};


/// Min-row reduction operator
struct MinRowOp
{
    template <typename PartialSum>
    __device__ __forceinline__ PartialSum operator()(
        const PartialSum &first,
        const PartialSum &second)
    {
        return (first.row < second.row) ?
            first :
            second;
    }
};


// Scan prefix functor for BlockScan.
template <typename PartialSum>
struct BlockPrefixOp
{
    // Running block-wide prefix
    PartialSum running_prefix;

    /**
     * Returns the block-wide running_prefix in thread-0
     */
    __device__ __forceinline__ PartialSum operator()(
        const PartialSum &block_aggregate)              ///< The aggregate sum of the local prefix sum inputs
    {
        ReduceByKeyOp scan_op;

        PartialSum retval = running_prefix;
        running_prefix = scan_op(running_prefix, block_aggregate);
        return retval;
    }
};


/// Functor for detecting row discontinuities.
struct NewRowOp
{
    /// Returns true if row_b is the start of a new row
    template <typename VertexId>
    __device__ __forceinline__ bool operator()(
        const VertexId& row_a,
        const VertexId& row_b)
    {
        return (row_a != row_b);
    }
};


/**
 * Threadblock abstraction for processing sparse SPMV tiles
 */
template <
    int             BLOCK_THREADS,
    int             ITEMS_PER_THREAD,
    typename        VertexId,
    typename        Value>
struct PersistentBlockSpmv
{
    //---------------------------------------------------------------------
    // Types and constants
    //---------------------------------------------------------------------

    // Constants
    enum
    {
        TILE_ITEMS = BLOCK_THREADS * ITEMS_PER_THREAD,
    };

    // Head flag type
    typedef int HeadFlag;

    // Dot product partial sum type
    typedef PartialSum<VertexId, Value> PartialSum;

    // Parameterized CUB types for the parallel execution context
    typedef BlockScan<PartialSum, BLOCK_THREADS, BLOCK_SCAN_RAKING_MEMOIZE>     BlockScan;
    typedef BlockExchange<VertexId, BLOCK_THREADS, ITEMS_PER_THREAD, true>      BlockExchangeRows;
    typedef BlockExchange<Value, BLOCK_THREADS, ITEMS_PER_THREAD, true>         BlockExchangeValues;
    typedef BlockDiscontinuity<HeadFlag, BLOCK_THREADS>                         BlockDiscontinuity;
    typedef BlockReduce<PartialSum, BLOCK_THREADS>                              BlockReduce;

    // Shared memory type for this threadblock
    struct TempStorage
    {
        union
        {
            typename BlockExchangeRows::TempStorage         exchange_rows;      // Smem needed for striped->blocked transpose
            typename BlockExchangeValues::TempStorage       exchange_values;    // Smem needed for striped->blocked transpose
            struct
            {
                typename BlockScan::TempStorage             scan;               // Smem needed for reduce-value-by-row scan
                typename BlockDiscontinuity::TempStorage    discontinuity;      // Smem needed for head-flagging
            };
            typename BlockReduce::TempStorage               reduce;             // Smem needed for min-finding reduction
        };

        VertexId        last_block_row;
        PartialSum      identity;
        PartialSum      first_scatter;
    };

    //---------------------------------------------------------------------
    // Thread fields
    //---------------------------------------------------------------------

    TempStorage                     &temp_storage;
    BlockPrefixOp<PartialSum>       carry;
    VertexId                        *d_rows;
    VertexId                        *d_columns;
    Value                           *d_values;
    Value                           *d_vector;
    Value                           *d_result;
    PartialSum                      *d_block_partials;
    VertexId                        last_block_row;


    /**
     * Constructor
     */
    __device__ __forceinline__
    PersistentBlockSpmv(
        TempStorage                 &temp_storage,
        VertexId                    *d_rows,
        VertexId                    *d_columns,
        Value                       *d_values,
        Value                       *d_vector,
        Value                       *d_result,
        PartialSum                  *d_block_partials,
        int                         block_offset,
        int                         block_oob)
    :
        temp_storage(temp_storage),
        d_rows(d_rows),
        d_columns(d_columns),
        d_values(d_values),
        d_vector(d_vector),
        d_result(d_result),
        d_block_partials(d_block_partials)
    {
        // Initialize scalar shared memory values
        if (threadIdx.x == 0)
        {
            VertexId first_block_row            = d_rows[block_offset];
            last_block_row                      = d_rows[block_oob - 1];

            // Initialize carry to identity
            carry.running_prefix.row            = first_block_row;
            carry.running_prefix.partial        = Value(0);
            temp_storage.identity               = carry.running_prefix;

            temp_storage.first_scatter.row      = last_block_row;
            temp_storage.first_scatter.partial  = Value(0);
            temp_storage.last_block_row         = last_block_row;
        }

        __syncthreads();

        last_block_row = temp_storage.last_block_row;
    }


    //---------------------------------------------------------------------
    // Operations
    //---------------------------------------------------------------------

    /**
     * Processes a COO input tile of edges, outputting dot products for each row
     */
    __device__ __forceinline__
    void ProcessTile(
        int block_offset,
        int guarded_items = 0)
    {
        VertexId    columns[ITEMS_PER_THREAD];
        VertexId    rows[ITEMS_PER_THREAD];
        Value       values[ITEMS_PER_THREAD];
        PartialSum  partial_sums[ITEMS_PER_THREAD];
        HeadFlag    head_flags[ITEMS_PER_THREAD];

        // Load a threadblock-striped tile of A (sparse row-ids, column-ids, and values)
        if (guarded_items)
        {
            // This is a partial-tile (e.g., the last tile of input).  Extend the coordinates of the last
            // vertex for out-of-bound items, but zero-valued
            LoadWarpStriped<LOAD_DEFAULT>(d_columns + block_offset, guarded_items, VertexId(0), columns);
            LoadWarpStriped<LOAD_DEFAULT>(d_values + block_offset, guarded_items, Value(0), values);
            LoadWarpStriped<LOAD_DEFAULT>(d_rows + block_offset, guarded_items, last_block_row, rows);
        }
        else
        {
            // Unguarded loads
            LoadWarpStriped<LOAD_DEFAULT>(d_columns + block_offset, columns);
            LoadWarpStriped<LOAD_DEFAULT>(d_values + block_offset, values);
            LoadWarpStriped<LOAD_DEFAULT>(d_rows + block_offset, rows);
        }

        // Fence to prevent hoisting any dependent code below into the loads above
        __syncthreads();

        // Load the referenced values from x and compute the dot product partials sums
        #pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
        {
#if CUB_PTX_ARCH >= 350
            values[ITEM] *= ThreadLoad<LOAD_LDG>(d_vector + columns[ITEM]);
#else
            values[ITEM] *= TexVector<Value>::Load(columns[ITEM]);
#endif
        }

        // Transpose from warp-striped to blocked arrangement
        BlockExchangeValues::WarpStripedToBlocked(temp_storage.exchange_values, values);

        __syncthreads();

        // Transpose from warp-striped to blocked arrangement
        BlockExchangeRows::WarpStripedToBlocked(temp_storage.exchange_rows, rows);

        // Barrier for smem reuse and coherence
        __syncthreads();

        // Flag row heads by looking for discontinuities
        BlockDiscontinuity(temp_storage.discontinuity).Flag(
            rows,                           // Original row ids
            carry.running_prefix.row,       // Row ID from last tile
            NewRowOp(),                     // Functor for detecting start of new rows
            head_flags);                    // (Out) Head flags

        // Assemble
        #pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
        {
            partial_sums[ITEM].partial = values[ITEM];
            partial_sums[ITEM].row = rows[ITEM];
        }

        // Compute the exclusive scan of partial_sums
        PartialSum block_aggregate;         // Threadblock-wide aggregate in thread0 (unused)
        BlockScan(temp_storage.scan).ExclusiveScan(
            partial_sums,
            partial_sums,                   // (Out)
            temp_storage.identity,
            ReduceByKeyOp(),
            block_aggregate,                // (Out)
            carry);                         // (In-out)

        // Scatter an accumulated dot product if it is the head of a valid row
        #pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
        {
            if (head_flags[ITEM])
            {
                // Scatter
                d_result[partial_sums[ITEM].row] = partial_sums[ITEM].partial;
            }
            else
            {
                // Otherwise reset the row ID to the last row in the threadblock's range
                partial_sums[ITEM].row = last_block_row;
            }
        }

        // Find the first-scattered dot product if not yet set
        if (temp_storage.first_scatter.row == last_block_row)
        {
            // Barrier for smem reuse and coherence
            __syncthreads();

            PartialSum first_scatter = BlockReduce(temp_storage.reduce).Reduce(
                partial_sums,
                MinRowOp());

            // Stash the first-scattered dot product in smem
            if (threadIdx.x == 0)
            {
                temp_storage.first_scatter = first_scatter;
            }
        }
    }


    /**
     * Processes a COO input tile of edges, outputting dot products for each row
     */
    __device__ __forceinline__
    void FinalizeBlock()
    {
        if (threadIdx.x == 0)
        {
            if (gridDim.x == 1)
            {
                // Scatter the final aggregate (this kernel contains only 1 threadblock)
                d_result[carry.running_prefix.row] = carry.running_prefix.partial;
            }
            else
            {
                // Unweight the first-output if it's the same row as the carry aggregate
                PartialSum first_scatter = temp_storage.first_scatter;
                if (first_scatter.row == carry.running_prefix.row)
                    first_scatter.partial = Value(0);

                // Write out threadblock first-item and carry aggregate
                d_block_partials[blockIdx.x * 2]          = first_scatter;
                d_block_partials[(blockIdx.x * 2) + 1]    = carry.running_prefix;
            }
        }
    }
};


/**
 * COO SpMV kernel
 */
template <
    int             BLOCK_THREADS,
    int             ITEMS_PER_THREAD,
    typename        VertexId,
    typename        Value>
__launch_bounds__ (BLOCK_THREADS)
__global__ void CooKernel(
    GridEvenShare<int>              even_share,
    PartialSum<VertexId, Value>     *d_block_partials,
    VertexId                        *d_rows,
    VertexId                        *d_columns,
    Value                           *d_values,
    Value                           *d_vector,
    Value                           *d_result)
{
    // Constants
    enum
    {
        TILE_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD
    };

    // SpMV threadblock tile-processing abstraction
    typedef PersistentBlockSpmv<BLOCK_THREADS, ITEMS_PER_THREAD, VertexId, Value> PersistentBlockSpmv;

    // Shared memory
    __shared__ typename PersistentBlockSpmv::TempStorage temp_storage;

    // Initialize threadblock even-share to tell us where to start and stop our tile-processing
    even_share.BlockInit();

    // Construct persistent SPMV thread block
    PersistentBlockSpmv persistent_block(
        temp_storage,
        d_rows,
        d_columns,
        d_values,
        d_vector,
        d_result,
        d_block_partials,
        even_share.block_offset,
        even_share.block_oob);

    // Process full tiles
    while (even_share.block_offset <= even_share.block_oob - TILE_SIZE)
    {
        persistent_block.ProcessTile(even_share.block_offset);
        even_share.block_offset += TILE_SIZE;
    }

    // Process final partial tile (if present)
    int guarded_items = even_share.block_oob - even_share.block_offset;
    if (guarded_items)
    {
        persistent_block.ProcessTile(even_share.block_offset, guarded_items);
    }

    // Finalize the block
    persistent_block.FinalizeBlock();
}


/**
 * Threadblock abstraction for processing sparse SPMV tiles
 */
template <
    int             BLOCK_THREADS,
    int             ITEMS_PER_THREAD,
    typename        VertexId,
    typename        Value>
struct FinalizeSpmvBlock
{
    //---------------------------------------------------------------------
    // Types and constants
    //---------------------------------------------------------------------

    // Constants
    enum
    {
        TILE_ITEMS = BLOCK_THREADS * ITEMS_PER_THREAD,
    };

    // Head flag type
    typedef int HeadFlag;

    // Dot product partial sum type
    typedef PartialSum<VertexId, Value> PartialSum;

    // Parameterized CUB types for the parallel execution context
    typedef BlockScan<PartialSum, BLOCK_THREADS>                        BlockScan;
    typedef BlockDiscontinuity<HeadFlag, BLOCK_THREADS>                 BlockDiscontinuity;

    // Shared memory type for this threadblock
    struct TempStorage
    {
        typename BlockScan::TempStorage           scan;               // Smem needed for reduce-value-by-row scan
        typename BlockDiscontinuity::TempStorage  discontinuity;      // Smem needed for head-flagging

        VertexId        last_block_row;
        VertexId        prev_tile_row;
        PartialSum      identity;
    };

    //---------------------------------------------------------------------
    // Operations
    //---------------------------------------------------------------------

    /**
     * Processes a COO input tile of edges, outputting dot products for each row
     */
    __device__ __forceinline__
    static void ProcessTile(
        TempStorage                     &temp_storage,
        BlockPrefixOp<PartialSum>         &carry,
        PartialSum                      *d_block_partials,
        Value                           *d_result,
        int                             block_offset,
        int                             guarded_items = 0)
    {
        VertexId    rows[ITEMS_PER_THREAD];
        PartialSum  partial_sums[ITEMS_PER_THREAD];
        HeadFlag    head_flags[ITEMS_PER_THREAD];

        // Load a threadblock-striped tile of A (sparse row-ids, column-ids, and values)
        if (guarded_items)
        {
            // This is a partial-tile (e.g., the last tile of input).  Extend the coordinates of the last
            // vertex for out-of-bound items, but zero-valued
            PartialSum default_sum;
            default_sum.row = temp_storage.last_block_row;
            default_sum.partial = Value(0);

        #if CUB_PTX_ARCH >= 350
            LoadBlocked<LOAD_LDG>(d_block_partials + block_offset, guarded_items, default_sum, partial_sums);
        #else
            LoadBlocked<LOAD_DEFAULT>(d_block_partials + block_offset, guarded_items, default_sum, partial_sums);
        #endif
        }
        else
        {
            // Unguarded loads
        #if CUB_PTX_ARCH >= 350
            LoadBlocked<LOAD_LDG>(d_block_partials + block_offset, partial_sums);
        #else
            LoadBlocked<LOAD_DEFAULT>(d_block_partials + block_offset, partial_sums);
        #endif
        }

        // Fence to prevent hoisting any dependent code below into the loads above
        __syncthreads();

        // Copy out row IDs for row-head flagging
        #pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
        {
            rows[ITEM] = partial_sums[ITEM].row;
        }

        // Flag row heads by looking for discontinuities
        BlockDiscontinuity(temp_storage.discontinuity).Flag(
            rows,                           // Original row ids
            temp_storage.prev_tile_row,     // Last row id from previous threadblock
            NewRowOp(),                     // Functor for detecting start of new rows
            head_flags);                    // (Out) Head flags

        // Store the last row in the tile (for computing head flags in the next tile)
        if (threadIdx.x == BLOCK_THREADS - 1)
        {
            temp_storage.prev_tile_row = rows[ITEMS_PER_THREAD - 1];
        }

        // Compute the exclusive scan of partial_sums
        PartialSum block_aggregate;         // Threadblock-wide aggregate in thread0 (unused)
        BlockScan(temp_storage.scan).ExclusiveScan(
            partial_sums,
            partial_sums,                   // (Out)
            temp_storage.identity,
            ReduceByKeyOp(),
            block_aggregate,                // (Out)
            carry);                         // (In-out)

        // Scatter an accumulated dot product if it is the head of a valid row
        #pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
        {
            if (head_flags[ITEM])
            {
                // Scatter
                d_result[partial_sums[ITEM].row] = partial_sums[ITEM].partial;
            }
        }

    }
};


/**
 * COO Finalize kernel.
 */
template <
    int             BLOCK_THREADS,
    int             ITEMS_PER_THREAD,
    typename        VertexId,
    typename        Value>
__launch_bounds__ (BLOCK_THREADS,  1)
__global__ void CooFinalizeKernel(
    PartialSum<VertexId, Value>     *d_block_partials,
    int                             finalize_partials,
    Value                           *d_result)
{
    // Constants
    enum
    {
        TILE_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD
    };

    // SpMV threadblock tile-processing abstraction
    typedef FinalizeSpmvBlock<BLOCK_THREADS, ITEMS_PER_THREAD, VertexId, Value> FinalizeSpmvBlock;

    // Shared memory
    __shared__ typename FinalizeSpmvBlock::TempStorage temp_storage;

    // Stateful prefix carryover from one tile to the next
    BlockPrefixOp<PartialSum<VertexId, Value> > carry;

    // Initialize scalar shared memory values
    if (threadIdx.x == 0)
    {
        VertexId first_block_row            = d_block_partials[0].row;
        VertexId last_block_row             = d_block_partials[finalize_partials - 1].row;

        // Initialize carry to identity
        carry.running_prefix.row            = first_block_row;
        carry.running_prefix.partial        = Value(0);
        temp_storage.identity               = carry.running_prefix;

        temp_storage.last_block_row         = last_block_row;
        temp_storage.prev_tile_row          = first_block_row;
    }

    // Barrier for smem coherence
    __syncthreads();

    // Process full tiles
    int block_offset = 0;
    while (block_offset <= finalize_partials - TILE_SIZE)
    {
        FinalizeSpmvBlock::ProcessTile(
            temp_storage,
            carry,
            d_block_partials,
            d_result,
            block_offset);

        block_offset += TILE_SIZE;
    }

    // Process final partial tile (if present)
    int guarded_items = finalize_partials - block_offset;
    if (guarded_items)
    {
        FinalizeSpmvBlock::ProcessTile(
            temp_storage,
            carry,
            d_block_partials,
            d_result,
            block_offset,
            guarded_items);
    }

    // Scatter the final aggregate (this kernel contains only 1 threadblock)
    if (threadIdx.x == 0)
    {
        d_result[carry.running_prefix.row] = carry.running_prefix.partial;
    }
}






//---------------------------------------------------------------------
// Host subroutines
//---------------------------------------------------------------------


/**
 * Simple test of device
 */
template <
    int                         COO_BLOCK_THREADS,
    int                         COO_ITEMS_PER_THREAD,
    int                         COO_SUBSCRIPTION_FACTOR,
    int                         FINALIZE_BLOCK_THREADS,
    int                         FINALIZE_ITEMS_PER_THREAD,
    typename                    VertexId,
    typename                    Value>
void TestDevice(
    CooGraph<VertexId, Value>&  coo_graph,
    Value*                      h_vector,
    Value*                      h_reference)
{
    typedef PartialSum<VertexId, Value> PartialSum;

    const int COO_TILE_SIZE = COO_BLOCK_THREADS * COO_ITEMS_PER_THREAD;

    if (g_iterations <= 0) return;


    // SOA device storage
    VertexId                        *d_rows;             // SOA graph row coordinates
    VertexId                        *d_columns;          // SOA graph col coordinates
    Value                           *d_values;           // SOA graph values
    Value                           *d_vector;           // Vector multiplicand
    Value                           *d_result;           // Output row
    PartialSum                      *d_block_partials;     // Temporary storage for communicating dot product partials between threadblocks

    // Create SOA version of coo_graph on host
    int                             num_edges       = coo_graph.coo_tuples.size();
    VertexId                        *h_rows          = new VertexId[num_edges];
    VertexId                        *h_columns       = new VertexId[num_edges];
    Value                           *h_values        = new Value[num_edges];
    for (int i = 0; i < num_edges; i++)
    {
        h_rows[i]       = coo_graph.coo_tuples[i].row;
        h_columns[i]    = coo_graph.coo_tuples[i].col;
        h_values[i]     = coo_graph.coo_tuples[i].val;
    }

    // Get CUDA properties
    Device device_props;
    CubDebugExit(device_props.Init());

    // Determine launch configuration from kernel properties
    int coo_sm_occupancy;
    CubDebugExit(device_props.MaxSmOccupancy(
        coo_sm_occupancy,
        CooKernel<COO_BLOCK_THREADS, COO_ITEMS_PER_THREAD, VertexId, Value>,
        COO_BLOCK_THREADS));
    int max_coo_grid_size   = device_props.sm_count * coo_sm_occupancy * COO_SUBSCRIPTION_FACTOR;

    // Construct an even-share work distribution
    GridEvenShare<int> even_share;
    even_share.GridInit(num_edges, max_coo_grid_size, COO_TILE_SIZE);
    int coo_grid_size       = even_share.grid_size;
    int finalize_partials   = coo_grid_size * 2;

    // Allocate COO device arrays
    CubDebugExit(DeviceAllocate((void**)&d_rows,            sizeof(VertexId) * num_edges));
    CubDebugExit(DeviceAllocate((void**)&d_columns,         sizeof(VertexId) * num_edges));
    CubDebugExit(DeviceAllocate((void**)&d_values,          sizeof(Value) * num_edges));
    CubDebugExit(DeviceAllocate((void**)&d_vector,          sizeof(Value) * coo_graph.col_dim));
    CubDebugExit(DeviceAllocate((void**)&d_result,          sizeof(Value) * coo_graph.row_dim));
    CubDebugExit(DeviceAllocate((void**)&d_block_partials,  sizeof(PartialSum) * finalize_partials));

    // Copy host arrays to device
    CubDebugExit(cudaMemcpy(d_rows,     h_rows,     sizeof(VertexId) * num_edges, cudaMemcpyHostToDevice));
    CubDebugExit(cudaMemcpy(d_columns,  h_columns,  sizeof(VertexId) * num_edges, cudaMemcpyHostToDevice));
    CubDebugExit(cudaMemcpy(d_values,   h_values,   sizeof(Value) * num_edges, cudaMemcpyHostToDevice));
    CubDebugExit(cudaMemcpy(d_vector,   h_vector,   sizeof(Value) * coo_graph.col_dim, cudaMemcpyHostToDevice));

    // Bind textures
    TexVector<Value>::BindTexture(d_vector, coo_graph.col_dim);


    // Print debug info
    printf("CooKernel<%d, %d><<<%d, %d>>>(...), Max SM occupancy: %d\n",
        COO_BLOCK_THREADS, COO_ITEMS_PER_THREAD, coo_grid_size, COO_BLOCK_THREADS, coo_sm_occupancy);
    if (coo_grid_size > 1)
    {
        printf("CooFinalizeKernel<<<1, %d>>>(...)\n", FINALIZE_BLOCK_THREADS);
    }
    fflush(stdout);

    CubDebugExit(cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte));

    // Run kernel
    GpuTimer gpu_timer;
    float elapsed_millis = 0.0;
    for (int i = 0; i < g_iterations; i++)
    {
        gpu_timer.Start();

        // Initialize output
        CubDebugExit(cudaMemset(d_result, 0, coo_graph.row_dim * sizeof(Value)));

        // Run the COO kernel
        CooKernel<COO_BLOCK_THREADS, COO_ITEMS_PER_THREAD><<<coo_grid_size, COO_BLOCK_THREADS>>>(
            even_share,
            d_block_partials,
            d_rows,
            d_columns,
            d_values,
            d_vector,
            d_result);

        if (coo_grid_size > 1)
        {
            // Run the COO finalize kernel
            CooFinalizeKernel<FINALIZE_BLOCK_THREADS, FINALIZE_ITEMS_PER_THREAD><<<1, FINALIZE_BLOCK_THREADS>>>(
                d_block_partials,
                finalize_partials,
                d_result);
        }

        gpu_timer.Stop();
        elapsed_millis += gpu_timer.ElapsedMillis();
    }

    // Force any kernel stdio to screen
    CubDebugExit(cudaThreadSynchronize());
    fflush(stdout);

    // Display timing
    float avg_elapsed = elapsed_millis / g_iterations;
    int total_bytes = ((sizeof(VertexId) + sizeof(VertexId)) * 2 * num_edges) + (sizeof(Value) * coo_graph.row_dim);
    printf("%d iterations, average elapsed (%.3f ms), utilized bandwidth (%.3f GB/s), GFLOPS(%.3f)\n",
        g_iterations,
        avg_elapsed,
        total_bytes / avg_elapsed / 1000.0 / 1000.0,
        num_edges * 2 / avg_elapsed / 1000.0 / 1000.0);

    // Check results
    int compare = CompareDeviceResults(h_reference, d_result, coo_graph.row_dim, true, g_verbose);
    printf("%s\n", compare ? "FAIL" : "PASS");
    AssertEquals(0, compare);

    // Cleanup
    TexVector<Value>::UnbindTexture();
    CubDebugExit(DeviceFree(d_block_partials));
    CubDebugExit(DeviceFree(d_rows));
    CubDebugExit(DeviceFree(d_columns));
    CubDebugExit(DeviceFree(d_values));
    CubDebugExit(DeviceFree(d_vector));
    CubDebugExit(DeviceFree(d_result));
    delete[] h_rows;
    delete[] h_columns;
    delete[] h_values;
}


/**
 * Compute reference answer on CPU
 */
template <typename VertexId, typename Value>
void ComputeReference(
    CooGraph<VertexId, Value>&  coo_graph,
    Value*                      h_vector,
    Value*                      h_reference)
{
    for (VertexId i = 0; i < coo_graph.row_dim; i++)
    {
        h_reference[i] = 0.0;
    }

    for (VertexId i = 0; i < coo_graph.coo_tuples.size(); i++)
    {
        h_reference[coo_graph.coo_tuples[i].row] +=
            coo_graph.coo_tuples[i].val *
            h_vector[coo_graph.coo_tuples[i].col];
    }
}


/**
 * Assign arbitrary values to vector items
 */
template <typename Value>
void AssignVectorValues(Value *vector, int col_dim)
{
    for (int i = 0; i < col_dim; i++)
    {
        vector[i] = 1.0;
    }
}


/**
 * Main
 */
int main(int argc, char** argv)
{
    // Graph of uint32s as vertex ids, double as values
    typedef int                 VertexId;
    typedef double              Value;

    // Initialize command line
    CommandLineArgs args(argc, argv);
    g_verbose = args.CheckCmdLineFlag("v");
    args.GetCmdLineArgument("i", g_iterations);

    // Print usage
    if (args.CheckCmdLineFlag("help"))
    {
        printf("%s\n [--device=<device-id>] [--v] [--iterations=<test iterations>] [--grid-size=<grid-size>]\n"
            "\t--type=wheel --spokes=<spokes>\n"
            "\t--type=grid2d --width=<width> [--no-self-loops]\n"
            "\t--type=grid3d --width=<width> [--no-self-loops]\n"
            "\t--type=market --file=<file>\n"
            "\n", argv[0]);
        exit(0);
    }

    // Initialize device
    CubDebugExit(args.DeviceInit());

    // Get graph type
    string type;
    args.GetCmdLineArgument("type", type);

    // Generate graph structure

    CpuTimer timer;
    timer.Start();
    CooGraph<VertexId, Value> coo_graph;
    if (type == string("grid2d"))
    {
        VertexId width;
        args.GetCmdLineArgument("width", width);
        bool self_loops = !args.CheckCmdLineFlag("no-self-loops");
        printf("Generating %s grid2d width(%d)... ", (self_loops) ? "5-pt" : "4-pt", width); fflush(stdout);
        if (coo_graph.InitGrid2d(width, self_loops)) exit(1);
    } else if (type == string("grid3d"))
    {
        VertexId width;
        args.GetCmdLineArgument("width", width);
        bool self_loops = !args.CheckCmdLineFlag("no-self-loops");
        printf("Generating %s grid3d width(%d)... ", (self_loops) ? "7-pt" : "6-pt", width); fflush(stdout);
        if (coo_graph.InitGrid3d(width, self_loops)) exit(1);
    }
    else if (type == string("wheel"))
    {
        VertexId spokes;
        args.GetCmdLineArgument("spokes", spokes);
        printf("Generating wheel spokes(%d)... ", spokes); fflush(stdout);
        if (coo_graph.InitWheel(spokes)) exit(1);
    }
    else if (type == string("market"))
    {
        string filename;
        args.GetCmdLineArgument("file", filename);
        printf("Generating MARKET for %s... ", filename.c_str()); fflush(stdout);
        if (coo_graph.InitMarket(filename)) exit(1);
    }
    else
    {
        printf("Unsupported graph type\n");
        exit(1);
    }
    timer.Stop();
    printf("Done (%.3fs). %d non-zeros, %d rows, %d columns\n",
        timer.ElapsedMillis() / 1000.0,
        coo_graph.coo_tuples.size(),
        coo_graph.row_dim,
        coo_graph.col_dim);
    fflush(stdout);

    if (g_verbose)
    {
        cout << coo_graph << "\n";
    }

    // Create vector
    Value *h_vector = new Value[coo_graph.col_dim];
    AssignVectorValues(h_vector, coo_graph.col_dim);
    if (g_verbose)
    {
        printf("Vector[%d]: ", coo_graph.col_dim);
        DisplayResults(h_vector, coo_graph.col_dim);
        printf("\n\n");
    }

    // Compute reference answer
    Value *h_reference = new Value[coo_graph.row_dim];
    ComputeReference(coo_graph, h_vector, h_reference);
    if (g_verbose)
    {
        printf("Results[%d]: ", coo_graph.row_dim);
        DisplayResults(h_reference, coo_graph.row_dim);
        printf("\n\n");
    }

    enum
    {
        COO_BLOCK_THREADS           = 64,
        COO_ITEMS_PER_THREAD        = 12,
        COO_SUBSCRIPTION_FACTOR     = 4,
        FINALIZE_BLOCK_THREADS      = 256,
        FINALIZE_ITEMS_PER_THREAD   = 4,
    };

    // Run GPU version
    TestDevice<
        COO_BLOCK_THREADS,
        COO_ITEMS_PER_THREAD,
        COO_SUBSCRIPTION_FACTOR,
        FINALIZE_BLOCK_THREADS,
        FINALIZE_ITEMS_PER_THREAD>(coo_graph, h_vector, h_reference);

    // Cleanup
    delete[] h_vector;
    delete[] h_reference;

    return 0;
}



