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
#include <test_util.h>

#include "../cub.cuh"

using namespace cub;
using namespace std;


//---------------------------------------------------------------------
// Globals, constants and typedefs
//---------------------------------------------------------------------

bool    g_verbose       = false;
int     g_iterations    = 1;


//---------------------------------------------------------------------
// Graph building types and utilities
//---------------------------------------------------------------------

/**
 * COO graph type.  A COO graph is just a vector of edge tuples.
 */
template<typename VertexId, typename Value>
struct CooGraph
{
    /**
     * COO edge tuple.  (A COO graph is just a vector of these.)
     */
    struct CooTuple
    {
        VertexId            row;
        VertexId            col;
        Value               val;

        CooTuple() {}
        CooTuple(VertexId row, VertexId col) : row(row), col(col) {}
        CooTuple(VertexId row, VertexId col, Value val) : row(row), col(col), val(val) {}
    };

    /**
     * Comparator for sorting COO sparse format edges
     */
    static bool CooTupleCompare (const CooTuple &elem1, const CooTuple &elem2)
    {
        if (elem1.row < elem2.row)
        {
            return true;
        }
        else if ((elem1.row == elem2.row) && (elem1.col < elem2.col))
        {
            return true;
        }

        return false;
    }


    /**
     * Fields
     */
    int                 row_dim;        // Num rows
    int                 col_dim;        // Num cols
    vector<CooTuple>    coo_tuples;     // Non-zero entries


    /**
     * CooGraph ostream operator
     */
    friend std::ostream& operator<<(std::ostream& os, const CooGraph& coo_graph)
    {
        os << "Sparse COO (" << coo_graph.row_dim << " rows, " << coo_graph.col_dim << " cols, " << coo_graph.coo_tuples.size() << " nonzeros):\n";
        os << "Ordinal, Row, Col, Val\n";
        for (int i = 0; i < coo_graph.coo_tuples.size(); i++)
        {
            os << i << ',' << coo_graph.coo_tuples[i].row << ',' << coo_graph.coo_tuples[i].col << ',' << coo_graph.coo_tuples[i].val << "\n";
        }
        return os;
    }

    /**
     * Update graph dims based upon COO tuples
     */
    void UpdateDims()
    {
        row_dim = -1;
        col_dim = -1;

        for (int i = 0; i < coo_tuples.size(); i++)
        {
            row_dim = CUB_MAX(row_dim, coo_tuples[i].row);
            col_dim = CUB_MAX(col_dim, coo_tuples[i].col);
        }

        row_dim++;
        col_dim++;
    }

    /**
     * Builds a METIS COO sparse from the given file.
     */
    int InitMetis(const string &metis_filename)
    {
        coo_tuples.clear();

        // Read from file
        FILE *f_in = fopen(metis_filename.c_str(), "r");
        if (!f_in) return -1;

        int edges_read = -1;
        int edges = 0;

        char line[1024];

        while(true)
        {
            if (fscanf(f_in, "%[^\n]\n", line) <= 0)
            {
                break;
            }
            if (line[0] == '%')
            {
                // Comment
            }
            else if (edges_read == -1)
            {
                // Problem description
                long long ll_nodes_x, ll_nodes_y, ll_edges;
                if (sscanf(line, "%lld %lld %lld", &ll_nodes_x, &ll_nodes_y, &ll_edges) != 3)
                {
                    fprintf(stderr, "Error parsing MARKET graph: invalid problem description\n");
                    return -1;
                }

                if (ll_nodes_x != ll_nodes_y)
                {
                    fprintf(stderr, "Error parsing MARKET graph: not square (%lld, %lld)\n", ll_nodes_x, ll_nodes_y);
                    return -1;
                }

                edges = ll_edges;

                printf(" (%lld nodes, %lld directed edges)... ",
                    (unsigned long long) ll_nodes_x,
                    (unsigned long long) ll_edges);
                fflush(stdout);

                // Allocate coo graph
                coo_tuples.reserve(edges);
                edges_read++;
            }
            else
            {
                if (edges_read >= edges)
                {
                    fprintf(stderr, "Error parsing MARKET graph: encountered more than %d edges\n", edges);
                    fclose(f_in);
                    return -1;
                }

                long long ll_row, ll_col;
                float val;
                if (sscanf(line, "%lld %lld %f", &ll_col, &ll_row, &val) != 3)
                {
                    fprintf(stderr, "Error parsing MARKET graph: badly formed edge\n", edges);
                    fclose(f_in);
                    return -1;
                }

                ll_row -= 1;
                ll_col -= 1;

                coo_tuples.push_back(CooTuple(ll_row, ll_col, val));    // zero-based array
                edges_read++;
            }
        }

        if (edges_read != edges)
        {
            fprintf(stderr, "Error parsing MARKET graph: only %d/%d edges read\n", edges_read, edges);
            fclose(f_in);
            return -1;
        }

        // Sort by rows, then columns, update dims
        std::stable_sort(coo_tuples.begin(), coo_tuples.end(), CooTupleCompare);
        UpdateDims();

        fclose(f_in);

        return 0;
    }


    /**
     * Builds a wheel COO sparse graph having spokes spokes.
     */
    int InitWheel(VertexId spokes)
    {
        VertexId edges  = spokes + (spokes - 1);

        coo_tuples.clear();
        coo_tuples.reserve(edges);

        // Add spoke edges
        for (VertexId i = 0; i < spokes; i++)
        {
            coo_tuples.push_back(CooTuple(0, i + 1));
        }

        // Add rim
        for (VertexId i = 0; i < spokes; i++)
        {
            VertexId dest = (i + 1) % spokes;
            coo_tuples.push_back(CooTuple(i + 1, dest + 1));
        }

        // Sort by rows, then columns, update dims
        std::stable_sort(coo_tuples.begin(), coo_tuples.end(), CooTupleCompare);
        UpdateDims();

        // Assign arbitrary values to graph vertices
        for (int i = 0; i < coo_tuples.size(); i++)
        {
            coo_tuples[i].val = i % 21;
        }

        return 0;
    }


    /**
     * Builds a square 3D grid COO sparse graph.  Interior nodes have degree 7 (including
     * a self-loop).  Values are unintialized, coo_tuples are sorted.
     */
    int InitGrid3d(VertexId width)
    {
        VertexId interior_nodes        = (width - 2) * (width - 2) * (width - 2);
        VertexId face_nodes            = (width - 2) * (width - 2) * 6;
        VertexId edge_nodes            = (width - 2) * 12;
        VertexId corner_nodes          = 8;
        VertexId nodes                 = width * width * width;
        VertexId edges                 = (interior_nodes * 6) + (face_nodes * 5) + (edge_nodes * 4) + (corner_nodes * 3) + nodes;

        coo_tuples.clear();
        coo_tuples.reserve(edges);

        for (VertexId i = 0; i < width; i++) {
            for (VertexId j = 0; j < width; j++) {
                for (VertexId k = 0; k < width; k++) {

                    VertexId me = (i * width * width) + (j * width) + k;

                    VertexId neighbor = (i * width * width) + (j * width) + (k - 1);
                    if (k - 1 >= 0) {
                        coo_tuples.push_back(CooTuple(me, neighbor));
                    }

                    neighbor = (i * width * width) + (j * width) + (k + 1);
                    if (k + 1 < width) {
                        coo_tuples.push_back(CooTuple(me, neighbor));
                    }

                    neighbor = (i * width * width) + ((j - 1) * width) + k;
                    if (j - 1 >= 0) {
                        coo_tuples.push_back(CooTuple(me, neighbor));
                    }

                    neighbor = (i * width * width) + ((j + 1) * width) + k;
                    if (j + 1 < width) {
                        coo_tuples.push_back(CooTuple(me, neighbor));
                    }

                    neighbor = ((i - 1) * width * width) + (j * width) + k;
                    if (i - 1 >= 0) {
                        coo_tuples.push_back(CooTuple(me, neighbor));
                    }

                    neighbor = ((i + 1) * width * width) + (j * width) + k;
                    if (i + 1 < width) {
                        coo_tuples.push_back(CooTuple(me, neighbor));
                    }

                    neighbor = me;
                    coo_tuples.push_back(CooTuple(me, neighbor));
                }
            }
        }

        // Sort by rows, then columns, update dims
        std::stable_sort(coo_tuples.begin(), coo_tuples.end(), CooTupleCompare);
        UpdateDims();

        // Assign arbitrary values to graph vertices
        for (int i = 0; i < coo_tuples.size(); i++)
        {
            coo_tuples[i].val = i % 21;
        }

        return 0;
    }


    /**
     * Builds a square 2D grid CSR graph.  Interior nodes have degree 5 (including
     * a self-loop)
     *
     * Returns 0 on success, 1 on failure.
     */
    int InitGrid2d(VertexId width, bool self)
    {
        VertexId interior_nodes        = (width - 2) * (width - 2);
        VertexId edge_nodes            = (width - 2) * 4;
        VertexId corner_nodes          = 4;
        VertexId edges                 = (interior_nodes * 4) + (edge_nodes * 3) + (corner_nodes * 2);

        if (self) edges += edges;

        coo_tuples.clear();
        coo_tuples.reserve(edges);

        for (VertexId j = 0; j < width; j++) {
            for (VertexId k = 0; k < width; k++) {

                VertexId me = (j * width) + k;

                VertexId neighbor = (j * width) + (k - 1);
                if (k - 1 >= 0) {
                    coo_tuples.push_back(CooTuple(me, neighbor));
                }

                neighbor = (j * width) + (k + 1);
                if (k + 1 < width) {
                    coo_tuples.push_back(CooTuple(me, neighbor));
                }

                neighbor = ((j - 1) * width) + k;
                if (j - 1 >= 0) {
                    coo_tuples.push_back(CooTuple(me, neighbor));
                }

                neighbor = ((j + 1) * width) + k;
                if (j + 1 < width) {
                    coo_tuples.push_back(CooTuple(me, neighbor));
                }

                if (self)
                {
                    neighbor = me;
                    coo_tuples.push_back(CooTuple(me, neighbor));
                }
            }
        }

        // Sort by rows, then columns, update dims
        std::stable_sort(coo_tuples.begin(), coo_tuples.end(), CooTupleCompare);
        UpdateDims();

        // Assign arbitrary values to graph vertices
        for (int i = 0; i < coo_tuples.size(); i++)
        {
            coo_tuples[i].val = i % 21;
        }

        return 0;
    }
};




//---------------------------------------------------------------------
// GPU types and device functions
//---------------------------------------------------------------------


/// Pairing of dot product partial sums and corresponding row-id
template <typename VertexId, typename Value>
struct PartialSum
{
    double      partial;        /// PartialSum sum
    int         row;            /// Row-id
    int         padding;

    /// Tags indicating this structure provides overloaded ThreadLoad and ThreadStore operations
    typedef void ThreadLoadTag;
    typedef void ThreadStoreTag;

    /// ThreadLoad (simply defer to loading individual items)
    template <PtxLoadModifier MODIFIER>
    __device__ __forceinline__ void ThreadLoad(PartialSum *ptr)
    {
        partial = cub::ThreadLoad<MODIFIER>(&(ptr->partial));
        row = cub::ThreadLoad<MODIFIER>(&(ptr->row));
    }

     /// ThreadStore (simply defer to storing individual items)
    template <PtxStoreModifier MODIFIER>
    __device__ __forceinline__ void ThreadStore(PartialSum *ptr) const
    {
        // Always write partial first
        cub::ThreadStore<MODIFIER>(&(ptr->partial), partial);
        cub::ThreadStore<MODIFIER>(&(ptr->row), row);
    }

};


/// Templated Texture reference type for multiplicand vector
template <typename Value>
struct TexVector
{
    // Texture reference type
    typedef texture<Value, cudaTextureType1D, cudaReadModeElementType> TexRef;

    static TexRef ref;

    /**
     * Bind textures
     */
    static void BindTexture(void *d_in, int elements)
    {
        cudaChannelFormatDesc tex_desc = cudaCreateChannelDesc<Value>();
        if (d_in)
        {
            size_t offset;
            size_t bytes = sizeof(Value) * elements;
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


// Callback functor for waiting on the previous threadblock to compute its partial sum (the prefix for this threadblock)
template <typename PartialSum>
struct BlockPrefixOp
{
    PartialSum prefix;

    /**
     * Threadblock-wide prefix callback functor called by thread-0 in BlockScan::ExclusiveScan().
     * Returns the threadblock-wide prefix to apply to all scan inputs.
     */
    __device__ __forceinline__ PartialSum operator()(
        const PartialSum &block_aggregate)              ///< The aggregate sum of the local prefix sum inputs
    {
        ReduceByKeyOp scan_op;

        PartialSum retval = prefix;
        prefix = scan_op(prefix, block_aggregate);
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
struct SpmvBlock
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

    // Parameterized CUB types for use in the current execution context
    typedef BlockScan<PartialSum, BLOCK_THREADS>                        BlockScan;
    typedef BlockExchange<VertexId, BLOCK_THREADS, ITEMS_PER_THREAD>    BlockExchangeRows;
    typedef BlockExchange<Value, BLOCK_THREADS, ITEMS_PER_THREAD>       BlockExchangeValues;
    typedef BlockDiscontinuity<HeadFlag, BLOCK_THREADS>                 BlockDiscontinuity;
    typedef BlockReduce<PartialSum, BLOCK_THREADS>                      BlockReduce;

    // Shared memory type for this threadblock
    struct SmemStorage
    {
        union
        {
            typename BlockExchangeRows::SmemStorage   exchange_rows;      // Smem needed for striped->blocked transpose
            typename BlockExchangeValues::SmemStorage exchange_values;    // Smem needed for striped->blocked transpose
            struct
            {
                typename BlockDiscontinuity::SmemStorage  discontinuity;      // Smem needed for head-flagging
                typename BlockScan::SmemStorage           scan;               // Smem needed for reduce-value-by-row scan
            };
            typename BlockReduce::SmemStorage         reduce;             // Smem needed for min-finding reduction
        };

        VertexId        last_block_row;
        VertexId        prev_tile_row;
        PartialSum      identity;
        PartialSum      first_scatter;
    };

    //---------------------------------------------------------------------
    // Operations
    //---------------------------------------------------------------------

    /**
     * Processes a COO input tile of edges, outputting dot products for each row
     */
    __device__ __forceinline__
    static void ProcessTile(
        SmemStorage                     &smem_storage,
        BlockPrefixOp<PartialSum>         &carry,
        VertexId*                       d_rows,
        VertexId*                       d_columns,
        Value*                          d_values,
        Value*                          d_vector,
        Value*                          d_result,
        int                             block_offset,
        int                             guarded_items = 0)
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
            BlockLoadDirectStriped(d_columns + block_offset, guarded_items, VertexId(0), columns, BLOCK_THREADS);
            BlockLoadDirectStriped(d_values + block_offset, guarded_items, Value(0), values, BLOCK_THREADS);
        }
        else
        {
            // Unguarded loads
            BlockLoadDirectStriped(d_columns + block_offset, columns, BLOCK_THREADS);
            BlockLoadDirectStriped(d_values + block_offset, values, BLOCK_THREADS);
        }

        // Fence to prevent hoisting any dependent code below into the loads above
        __syncthreads();

        // Load the referenced values from x and compute the dot product partials sums
        #pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
        {
            Value vec_item;
            uint2 *v = reinterpret_cast<uint2 *>(&vec_item);
            *v= tex1Dfetch(TexVector<uint2>::ref, columns[ITEM]);
            values[ITEM] *= vec_item;
        }

        // Load a threadblock-striped tile of A (sparse row-ids, column-ids, and values)
        if (guarded_items)
        {
            // This is a partial-tile (e.g., the last tile of input).  Extend the coordinates of the last
            // vertex for out-of-bound items, but zero-valued
            BlockLoadDirectStriped(d_rows + block_offset, guarded_items, smem_storage.last_block_row, rows, BLOCK_THREADS);
        }
        else
        {
            // Unguarded loads
            BlockLoadDirectStriped(d_rows + block_offset, rows, BLOCK_THREADS);
        }

        // Transpose from threadblock-striped to threadblock-blocked arrangement
        BlockExchangeValues::StripedToBlocked(smem_storage.exchange_values, values);

        // Barrier for smem reuse and coherence
        __syncthreads();

        // Transpose from threadblock-striped to threadblock-blocked arrangement
        BlockExchangeRows::StripedToBlocked(smem_storage.exchange_rows, rows);

        // Barrier for smem reuse and coherence
        __syncthreads();

        // Flag row heads by looking for discontinuities
        BlockDiscontinuity::Flag(
            smem_storage.discontinuity,
            rows,                           // Original row ids
            smem_storage.prev_tile_row,     // Last row id from previous threadblock
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
        BlockScan::ExclusiveScan(
            smem_storage.scan,
            partial_sums,
            partial_sums,                   // (Out)
            smem_storage.identity,
            ReduceByKeyOp(),
            block_aggregate,                // (Out)
            carry);                         // (In-out)

        // Store the last row in the tile (for computing head flags in the next tile)
        if (threadIdx.x == BLOCK_THREADS - 1)
        {
            smem_storage.prev_tile_row = rows[ITEMS_PER_THREAD - 1];
        }

        // Pull the last row ID out of smem
        VertexId last_block_row = smem_storage.last_block_row;

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
        if (smem_storage.first_scatter.row == last_block_row)
        {
            // Barrier for smem reuse and coherence
            __syncthreads();

            PartialSum candidate = BlockReduce::Reduce(
                smem_storage.reduce,
                partial_sums,
                MinRowOp());

            // Stash the first-scattered dot product in smem
            if (threadIdx.x == 0)
            {
                smem_storage.first_scatter = candidate;
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
    int             BLOCK_OCCUPANCY,
    typename        VertexId,
    typename        Value>
__launch_bounds__ (BLOCK_THREADS, BLOCK_OCCUPANCY)
__global__ void CooKernel(
    BlockEvenShare<int>               block_progress,
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
    typedef SpmvBlock<BLOCK_THREADS, ITEMS_PER_THREAD, VertexId, Value> SpmvBlock;

    // Shared memory
    __shared__ typename SpmvBlock::SmemStorage smem_storage;

    // Initialize threadblock even-share to tell us where to start and stop our tile-processing
    block_progress.Init();

    // Stateful prefix carryover from one tile to the next
    BlockPrefixOp<PartialSum<VertexId, Value> > carry;

    // Initialize scalar shared memory values
    if (threadIdx.x == 0)
    {
        VertexId first_block_row              = d_rows[block_progress.block_offset];
        VertexId last_block_row               = d_rows[block_progress.block_oob - 1];

        // Initialize carry to identity
        carry.prefix.row                    = first_block_row;
        carry.prefix.partial                = Value(0);
        smem_storage.identity               = carry.prefix;

        smem_storage.first_scatter.row      = last_block_row;
        smem_storage.first_scatter.partial  = Value(0);
        smem_storage.last_block_row           = last_block_row;
        smem_storage.prev_tile_row          = (blockIdx.x == 0) ?
                                                first_block_row :
                                                d_rows[block_progress.block_offset - 1];
    }

    // Process full tiles
    while (block_progress.block_offset <= block_progress.block_oob - TILE_SIZE)
    {
        SpmvBlock::ProcessTile(
            smem_storage,
            carry,
            d_rows,
            d_columns,
            d_values,
            d_vector,
            d_result,
            block_progress.block_offset);

        block_progress.block_offset += TILE_SIZE;
    }

    // Process final partial tile (if present)
    int guarded_items = block_progress.block_oob - block_progress.block_offset;
    if (guarded_items)
    {
        SpmvBlock::ProcessTile(
            smem_storage,
            carry,
            d_rows,
            d_columns,
            d_values,
            d_vector,
            d_result,
            block_progress.block_offset,
            guarded_items);
    }

    if (threadIdx.x == 0)
    {
        if (gridDim.x == 1)
        {
            // Scatter the final aggregate (this kernel contains only 1 threadblock)
            d_result[carry.prefix.row] = carry.prefix.partial;
        }
        else
        {
            // Write out threadblock first-item and carry aggregate

            // Unweight the first-output if it's the same row as the carry aggregate
            PartialSum<VertexId, Value> first_scatter = smem_storage.first_scatter;
            if (first_scatter.row == carry.prefix.row) first_scatter.partial = Value(0);

            d_block_partials[blockIdx.x * 2]          = first_scatter;
            d_block_partials[(blockIdx.x * 2) + 1]    = carry.prefix;
        }
    }
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

    // Parameterized CUB types for use in the current execution context
    typedef BlockScan<PartialSum, BLOCK_THREADS>                        BlockScan;
    typedef BlockDiscontinuity<HeadFlag, BLOCK_THREADS>                 BlockDiscontinuity;

    // Shared memory type for this threadblock
    struct SmemStorage
    {
        union
        {
            typename BlockScan::SmemStorage           scan;               // Smem needed for reduce-value-by-row scan
            typename BlockDiscontinuity::SmemStorage  discontinuity;      // Smem needed for head-flagging
        };

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
        SmemStorage                     &smem_storage,
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
            default_sum.row = smem_storage.last_block_row;
            default_sum.partial = Value(0);

            BlockLoadDirect(d_block_partials + block_offset, guarded_items, default_sum, partial_sums);
        }
        else
        {
            // Unguarded loads
            BlockLoadDirect(d_block_partials + block_offset, partial_sums);
        }

        // Fence to prevent hoisting any dependent code below into the loads above
//        __threadfence_block();

        // Copy out row IDs for row-head flagging
        #pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
        {
            rows[ITEM] = partial_sums[ITEM].row;
        }

        // Flag row heads by looking for discontinuities
        BlockDiscontinuity::Flag(
            smem_storage.discontinuity,
            rows,                           // Original row ids
            smem_storage.prev_tile_row,     // Last row id from previous threadblock
            NewRowOp(),                     // Functor for detecting start of new rows
            head_flags);                    // (Out) Head flags

        // Barrier for smem reuse and coherence
        __syncthreads();

        // Store the last row in the tile (for computing head flags in the next tile)
        if (threadIdx.x == BLOCK_THREADS - 1)
        {
            smem_storage.prev_tile_row = rows[ITEMS_PER_THREAD - 1];
        }

        // Compute the exclusive scan of partial_sums
        PartialSum block_aggregate;         // Threadblock-wide aggregate in thread0 (unused)
        BlockScan::ExclusiveScan(
            smem_storage.scan,
            partial_sums,
            partial_sums,                   // (Out)
            smem_storage.identity,
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
    __shared__ typename FinalizeSpmvBlock::SmemStorage smem_storage;

    // Stateful prefix carryover from one tile to the next
    BlockPrefixOp<PartialSum<VertexId, Value> > carry;

    // Initialize scalar shared memory values
    if (threadIdx.x == 0)
    {
        VertexId first_block_row              = d_block_partials[0].row;
        VertexId last_block_row               = d_block_partials[finalize_partials - 1].row;

        // Initialize carry to identity
        carry.prefix.row                    = first_block_row;
        carry.prefix.partial                = Value(0);
        smem_storage.identity               = carry.prefix;

        smem_storage.last_block_row           = last_block_row;
        smem_storage.prev_tile_row          = first_block_row;
    }

    // Barrier for smem coherence
    __syncthreads();

    // Process full tiles
    int block_offset = 0;
    while (block_offset <= finalize_partials - TILE_SIZE)
    {
        FinalizeSpmvBlock::ProcessTile(
            smem_storage,
            carry,
            d_block_partials,
            d_result,
            block_offset);

        // Barrier for smem reuse and coherence
        __syncthreads();

        block_offset += TILE_SIZE;
    }

    // Process final partial tile (if present)
    int guarded_items = finalize_partials - block_offset;
    if (guarded_items)
    {
        FinalizeSpmvBlock::ProcessTile(
            smem_storage,
            carry,
            d_block_partials,
            d_result,
            block_offset,
            guarded_items);
    }

    // Scatter the final aggregate (this kernel contains only 1 threadblock)
    if (threadIdx.x == 0)
    {
        d_result[carry.prefix.row] = carry.prefix.partial;
    }
}






//---------------------------------------------------------------------
// Host subroutines
//---------------------------------------------------------------------

/**
 * Compares the equivalence of two arrays
 */
template <typename SizeT>
int CompareResults(float* computed, float* reference, SizeT len)
{
    int retval = 0;
    for (SizeT i = 0; i < len; i++)
    {
        if (computed[i] > 0.0)
        {
            if ((computed[i] * 1.001 < reference[i]) ||
                (computed[i] * 0.999 > reference[i]))
            {
                printf("INCORRECT ([%d]: %f != %f) ", i, computed[i], reference[i]);
                return 1;
            }
        }
        else
        {
            if ((computed[i] * 1.001 > reference[i]) ||
                (computed[i] * 0.999 < reference[i]))
            {
                printf("INCORRECT ([%d]: %f != %f) ", i, computed[i], reference[i]);
                return 1;
            }
        }
    }

    if (!retval) printf("CORRECT");
    return retval;
}

/**
 * Compares the equivalence of two arrays
 */
template <typename SizeT>
int CompareResults(double* computed, double* reference, SizeT len)
{
    int retval = 0;
    for (SizeT i = 0; i < len; i++)
    {
        if (computed[i] > 0.0)
        {
            if ((computed[i] * 1.001 < reference[i]) ||
                (computed[i] * 0.999 > reference[i]))
            {
                printf("INCORRECT ([%d]: %f != %f) ", i, computed[i], reference[i]);
                return 1;
            }
        }
        else
        {
            if ((computed[i] * 1.001 > reference[i]) ||
                (computed[i] * 0.999 < reference[i]))
            {
                printf("INCORRECT ([%d]: %f != %f) ", i, computed[i], reference[i]);
                return 1;
            }
        }
    }

    if (!retval) printf("CORRECT");
    return retval;
}


/**
 * Simple test of device
 */
template <
    int                         COO_BLOCK_THREADS,
    int                         COO_ITEMS_PER_THREAD,
    int                         COO_BLOCK_OCCUPANCY,
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
    CudaProps cuda_props;
    CubDebugExit(cuda_props.Init());

    // Get kernel properties
    KernelProps coo_kernel_props;
    KernelProps finalize_kernel_props;
    CubDebugExit(coo_kernel_props.Init(
        CooKernel<COO_BLOCK_THREADS, COO_ITEMS_PER_THREAD, COO_BLOCK_OCCUPANCY, VertexId, Value>,
        COO_BLOCK_THREADS,
        cuda_props));
    CubDebugExit(finalize_kernel_props.Init(
        CooFinalizeKernel<FINALIZE_BLOCK_THREADS, FINALIZE_ITEMS_PER_THREAD, VertexId, Value>,
        FINALIZE_BLOCK_THREADS,
        cuda_props));

    // Determine launch configuration from kernel properties
    int coo_grid_size       = coo_kernel_props.OversubscribedGridSize(COO_TILE_SIZE, num_edges);
    int finalize_partials   = coo_grid_size * 2;

    // Allocate COO device arrays
    CubDebugExit(DeviceAllocate((void**)&d_rows,            sizeof(VertexId) * num_edges));
    CubDebugExit(DeviceAllocate((void**)&d_columns,         sizeof(VertexId) * num_edges));
    CubDebugExit(DeviceAllocate((void**)&d_values,          sizeof(Value) * num_edges));
    CubDebugExit(DeviceAllocate((void**)&d_vector,          sizeof(Value) * coo_graph.col_dim));
    CubDebugExit(DeviceAllocate((void**)&d_result,          sizeof(Value) * coo_graph.row_dim));
    CubDebugExit(DeviceAllocate((void**)&d_block_partials,    sizeof(PartialSum) * finalize_partials));

    // Copy host arrays to device
    CubDebugExit(cudaMemcpy(d_rows,     h_rows,     sizeof(VertexId) * num_edges, cudaMemcpyHostToDevice));
    CubDebugExit(cudaMemcpy(d_columns,  h_columns,  sizeof(VertexId) * num_edges, cudaMemcpyHostToDevice));
    CubDebugExit(cudaMemcpy(d_values,   h_values,   sizeof(Value) * num_edges, cudaMemcpyHostToDevice));
    CubDebugExit(cudaMemcpy(d_vector,   h_vector,   sizeof(Value) * coo_graph.col_dim, cudaMemcpyHostToDevice));

    // Bind textures
//    TexVector<Value>::BindTexture(d_vector, coo_graph.col_dim);
    TexVector<uint2>::BindTexture((uint2 *) d_vector, coo_graph.col_dim);

    // Construct an even-share work distribution
    BlockEvenShare<int> block_progress(num_edges, coo_grid_size, COO_TILE_SIZE);

    // Print debug info
    printf("CooKernel<%d, %d><<<%d, %d>>>(...), Max SM occupancy: %d\n",
        COO_BLOCK_THREADS, COO_ITEMS_PER_THREAD, coo_grid_size, COO_BLOCK_THREADS, coo_kernel_props.max_block_occupancy);
    if (coo_grid_size > 1)
    {
        printf("CooFinalizeKernel<<<1, %d>>>(...), Max SM occupancy: %d\n",
            FINALIZE_BLOCK_THREADS, finalize_kernel_props.max_block_occupancy);
    }
    fflush(stdout);

    // Run kernel
    GpuTimer gpu_timer;
    float elapsed_millis = 0.0;
    for (int i = 0; i < g_iterations; i++)
    {
        gpu_timer.Start();

        // Initialize output
        CubDebugExit(cudaMemset(d_result, 0, coo_graph.row_dim * sizeof(Value)));

        // Run the COO kernel
        CooKernel<COO_BLOCK_THREADS, COO_ITEMS_PER_THREAD, COO_BLOCK_OCCUPANCY><<<coo_grid_size, COO_BLOCK_THREADS>>>(
            block_progress,
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
    AssertEquals(0, CompareDeviceResults(h_reference, d_result, coo_graph.row_dim, g_verbose, g_verbose));

    // Cleanup
//    TexVector<Value>::UnbindTexture();
    TexVector<uint2>::UnbindTexture();
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
    // Graph of uint32s as vertex ids, floats as values
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
            "\t--type=grid2d --width=<width> [--7pt]\n"
            "\t--type=grid3d --width=<width>\n"
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
        bool self = args.CheckCmdLineFlag("7pt");
        printf("Generating %s grid2d width(%d)... ", (self) ? "7-pt" : "6-pt", width); fflush(stdout);
        if (coo_graph.InitGrid2d(width, self)) exit(1);
    } else if (type == string("grid3d"))
    {
        VertexId width;
        args.GetCmdLineArgument("width", width);
        printf("Generating grid3d width(%d)... ", width); fflush(stdout);
        if (coo_graph.InitGrid3d(width)) exit(1);
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
        if (coo_graph.InitMetis(filename)) exit(1);
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

    // Run GPU version
    TestDevice<
        128,
        7,
        5,
        256,
        4>(coo_graph, h_vector, h_reference);

    // Cleanup
    delete[] h_vector;
    delete[] h_reference;

    return 0;
}



