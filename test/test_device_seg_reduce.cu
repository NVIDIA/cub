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
 * An implementation of segmented reduction using a load-balanced parallelization
 * strategy based on the MergePath decision path.
 ******************************************************************************/

// Ensure printing of CUDA runtime errors to console
#define CUB_STDERR

#include <iterator>
#include <vector>
#include <string>
#include <algorithm>
#include <stdio.h>

#include <cub/cub.cuh>

#include "test_util.h"

using namespace cub;
using namespace std;


/******************************************************************************
 * Globals, constants, and typedefs
 ******************************************************************************/

bool                    g_verbose           = false;
int                     g_timing_iterations = 1;
CachingDeviceAllocator  g_allocator(true);


/******************************************************************************
 * Utility routines
 ******************************************************************************/


/**
 * Computes the begin offsets into A and B for the specified
 * location (diagonal) along the merge decision path
 */
template <
    typename    IteratorA,
    typename    IteratorB,
    typename    Offset>
__device__ __forceinline__ void MergePathSearch(
    Offset      diagonal,
    IteratorA   a,
    Offset      a_begin,
    Offset      a_end,
    Offset      &a_offset,
    IteratorB   b,
    Offset      b_begin,
    Offset      b_end,
    Offset      &b_offset)
{
    Offset split_min = CUB_MAX(diagonal - b_end, a_begin);
    Offset split_max = CUB_MIN(diagonal, a_end);

    while (split_min < split_max)
    {
        Offset split_pivot = (split_min + split_max) >> 1;
        if (a[split_pivot] <= b[diagonal - split_pivot - 1])
        {
            // Move candidate split range up A, down B
            split_min = split_pivot + 1;
        }
        else
        {
            // Move candidate split range up B, down A
            split_max = split_pivot;
        }
    }

    a_offset = CUB_MIN(split_min, a_end);
    b_offset = CUB_MIN(diagonal - split_min, b_end);
}

/**
 * Computes the begin offsets into A and B for the specified
 * location (diagonal) along the merge decision path
 */
template <
    typename    IteratorA,
    typename    IteratorB,
    typename    Offset>
__device__ __forceinline__ void MergePathSearch2(
    Offset      diagonal,
    IteratorA   a,
    Offset      a_begin,
    Offset      a_end,
    Offset      &a_offset,
    IteratorB   b,
    Offset      b_begin,
    Offset      b_end,
    Offset      &b_offset)
{
    Offset split_min = CUB_MAX(diagonal - b_end, a_begin);
    Offset split_max = CUB_MIN(diagonal, a_end);

    while (split_min < split_max)
    {
        Offset split_pivot = (split_min + split_max) >> 1;
        if (a[split_pivot] <= b[diagonal - split_pivot - 1])
        {
            // Move candidate split range up A, down B
            split_min = split_pivot + 1;
        }
        else
        {
            // Move candidate split range up B, down A
            split_max = split_pivot;
        }
    }

    a_offset = CUB_MIN(split_min, a_end);
    b_offset = CUB_MIN(diagonal - split_min, b_end);
}


/******************************************************************************
 * Tuning policy types
 ******************************************************************************/

/**
 * Parameterizable tuning policy type for BlockSegReduceRegion
 */
template <
    int                     _BLOCK_THREADS,             ///< Threads per thread block
    int                     _ITEMS_PER_THREAD,          ///< Items per thread (per tile of input)
    bool                    _USE_SMEM_SEGMENT_CACHE,    ///< Whether or not to cache incoming segment offsets in shared memory before reducing each tile
    bool                    _USE_SMEM_VALUE_CACHE,      ///< Whether or not to cache incoming values in shared memory before reducing each tile
    CacheLoadModifier       _LOAD_MODIFIER_SEGMENTS,    ///< Cache load modifier for reading segment offsets
    CacheLoadModifier       _LOAD_MODIFIER_VALUES,      ///< Cache load modifier for reading values
    BlockReduceAlgorithm    _REDUCE_ALGORITHM,          ///< The BlockReduce algorithm to use
    BlockScanAlgorithm      _SCAN_ALGORITHM>            ///< The BlockScan algorithm to use
struct BlockSegReduceRegionPolicy
{
    enum
    {
        BLOCK_THREADS           = _BLOCK_THREADS,               ///< Threads per thread block
        ITEMS_PER_THREAD        = _ITEMS_PER_THREAD,            ///< Items per thread (per tile of input)
        USE_SMEM_SEGMENT_CACHE  = _USE_SMEM_SEGMENT_CACHE,      ///< Whether or not to cache incoming segment offsets in shared memory before reducing each tile
        USE_SMEM_VALUE_CACHE    = _USE_SMEM_VALUE_CACHE,        ///< Whether or not to cache incoming upcoming values in shared memory before reducing each tile
    };

    static const CacheLoadModifier      LOAD_MODIFIER_SEGMENTS  = _LOAD_MODIFIER_SEGMENTS;  ///< Cache load modifier for reading segment offsets
    static const CacheLoadModifier      LOAD_MODIFIER_VALUES    = _LOAD_MODIFIER_VALUES;    ///< Cache load modifier for reading values
    static const BlockReduceAlgorithm   REDUCE_ALGORITHM        = _REDUCE_ALGORITHM;        ///< The BlockReduce algorithm to use
    static const BlockScanAlgorithm     SCAN_ALGORITHM          = _SCAN_ALGORITHM;          ///< The BlockScan algorithm to use
};


/******************************************************************************
 * Persistent thread block types
 ******************************************************************************/

/**
 * \brief BlockSegReduceTiles implements a stateful abstraction of CUDA thread blocks for participating in device-wide segmented reduction.
 */
template <
    typename BlockSegReduceRegionPolicy,    ///< Parameterized BlockSegReduceRegionPolicy tuning policy
    typename SegmentOffsetIterator,         ///< Random-access input iterator type for reading segment end-offsets
    typename ValueIterator,                 ///< Random-access input iterator type for reading values
    typename OutputIterator,                ///< Random-access output iterator type for writing segment reductions
    typename ReductionOp,                   ///< Binary reduction operator type having member <tt>T operator()(const T &a, const T &b)</tt>
    typename Offset>                        ///< Signed integer type for global offsets
struct BlockSegReduceRegion
{
    //---------------------------------------------------------------------
    // Types and constants
    //---------------------------------------------------------------------

    // Constants
    enum
    {
        BLOCK_THREADS       = BlockSegReduceRegionPolicy::BLOCK_THREADS,
        ITEMS_PER_THREAD    = BlockSegReduceRegionPolicy::ITEMS_PER_THREAD,
        TILE_ITEMS          = BLOCK_THREADS * ITEMS_PER_THREAD,                     /// Number of work items to be processed per tile
    };

    // Segment offset type
    typedef typename std::iterator_traits<SegmentOffsetIterator>::value_type SegmentOffset;

    // Value type
    typedef typename std::iterator_traits<ValueIterator>::value_type Value;

    // Counting iterator type
    typedef CountingInputIterator<SegmentOffset, Offset> CountingIterator;

    // Segment offsets iterator wrapper type
    typedef typename If<(IsPointer<SegmentOffsetIterator>::VALUE),
            CacheModifiedInputIterator<BlockSegReduceRegionPolicy::LOAD_MODIFIER_SEGMENTS, SegmentOffset, Offset>,  // Wrap the native input pointer with CacheModifiedInputIterator
            SegmentOffsetIterator>::Type                                                                            // Directly use the supplied input iterator type
        WrappedSegmentOffsetIterator;

    // Values iterator wrapper type
    typedef typename If<(IsPointer<ValueIterator>::VALUE),
            CacheModifiedInputIterator<BlockSegReduceRegionPolicy::LOAD_MODIFIER_VALUES, Value, Offset>,        // Wrap the native input pointer with CacheModifiedInputIterator
            ValueIterator>::Type                                                                                // Directly use the supplied input iterator type
        WrappedValueIterator;

    // Tail flag type for marking segment discontinuities
    typedef int TailFlag;

    // Reduce-by-key data type tuple (segment-ID, value)
    typedef KeyValuePair<Offset, Value> KeyValuePair;

    // BlockScan scan operator for reduction-by-segment
    typedef ReduceByKeyOp<ReductionOp> ReduceByKeyOp;

    // Stateful BlockScan prefix callback type for managing a running total while scanning consecutive tiles
    typedef RunningBlockPrefixCallbackOp<
            KeyValuePair,
            ReduceByKeyOp>
        RunningPrefixCallbackOp;

    // Parameterized BlockReduce type for block-wide reduction
    typedef BlockReduce<
            Value,
            BLOCK_THREADS,
            BlockSegReduceRegionPolicy::REDUCE_ALGORITHM>
        BlockReduce;

    // Parameterized BlockScan type for block-wide reduce-value-by-key
    typedef BlockScan<
            KeyValuePair,
            BLOCK_THREADS,
            BlockSegReduceRegionPolicy::SCAN_ALGORITHM>
        BlockScan;

    // Shared memory type for this threadblock
    struct _TempStorage
    {
        union
        {
            // Smem needed for BlockScan
            typename BlockScan::TempStorage     scan;

            // Smem needed for BlockReduce
            typename BlockReduce::TempStorage   reduce;

            struct
            {
                // Smem needed for communicating start/end indices between threads for a given work tile
                Offset thread_segment_idx[BLOCK_THREADS + 1];
                Value thread_value_idx[BLOCK_THREADS + 1];
            };
        };

        Offset block_segment_idx[2];     // The starting and ending indices of segment offsets for the threadblock's region
        Offset block_value_idx[2];       // The starting and ending indices of values for the threadblock's region

        // The first partial reduction tuple scattered by this thread block
        KeyValuePair first_tuple;
    };


    // Alias wrapper allowing storage to be unioned
    struct TempStorage : Uninitialized<_TempStorage> {};


    //---------------------------------------------------------------------
    // Thread fields
    //---------------------------------------------------------------------

    _TempStorage                    &temp_storage;          ///< Reference to shared storage
    WrappedSegmentOffsetIterator    d_segment_end_offsets;  ///< A sequence of \p num_segments segment end-offsets
    WrappedValueIterator            d_values;               ///< A sequence of \p num_values data to reduce
    OutputIterator                  d_output;               ///< A sequence of \p num_segments segment totals
    CountingIterator                d_value_offsets;        ///< A sequence of \p num_values value-offsets
    Offset                          num_values;             ///< Total number of values to reduce
    Offset                          num_segments;           ///< Number of segments being reduced
    Value                           identity;               ///< Identity value (for zero-length segments)
    ReductionOp                     reduction_op;           ///< Reduction operator
    ReduceByKeyOp                   scan_op;                ///< Reduce-by-key scan operator
    RunningPrefixCallbackOp         prefix_op;              ///< Stateful running total for block-wide prefix scan of partial reduction tuples


    //---------------------------------------------------------------------
    // Operations
    //---------------------------------------------------------------------

    /**
     * Constructor
     */
    __device__ __forceinline__
    BlockSegReduceRegion(
        TempStorage             &temp_storage,          ///< Reference to shared storage
        SegmentOffsetIterator   d_segment_end_offsets,  ///< A sequence of \p num_segments segment end-offsets
        ValueIterator           d_values,               ///< A sequence of \p num_values values
        OutputIterator          d_output,               ///< A sequence of \p num_segments segment totals
        Offset                  num_values,             ///< Number of values to reduce
        Offset                  num_segments,           ///< Number of segments being reduced
        Value                   identity,               ///< Identity value (for zero-length segments)
        ReductionOp             reduction_op)           ///< Reduction operator
    :
        temp_storage(temp_storage.Alias()),
        d_segment_end_offsets(d_segment_end_offsets),
        d_values(d_values),
        d_value_offsets(0),
        d_output(d_output),
        num_values(num_values),
        num_segments(num_segments),
        identity(identity),
        reduction_op(reduction_op),
        scan_op(reduction_op),
        prefix_op(scan_op)
    {}


    /**
     * Have the thread block process the specified region of the MergePath decision path
     */
    __device__ __forceinline__ void ProcessBlockRegion(
        Offset          block_diagonal,
        Offset          next_block_diagonal,
        KeyValuePair    &first_tuple,       // [Out] Valid in thread-0
        KeyValuePair    &last_tuple)        // [Out] Valid in thread-0
    {
        // Thread block initialization
        if (threadIdx.x < 2)
        {
            Offset diagonal = (threadIdx.x == 0) ?
                block_diagonal :        // First thread searches for start indices
                next_block_diagonal;    // Second thread searches for end indices

            // Search for block starting and ending indices
            Offset block_segment_idx;
            Offset block_value_idx;

            MergePathSearch(
                diagonal,               // Diagonal
                d_segment_end_offsets,  // A (segment end-offsets)
                0,                      // Start index into A
                num_segments,           // End index into A
                block_segment_idx,      // [out] Block index into A
                d_value_offsets,        // B (value offsets)
                0,                      // Start index into B
                num_values,             // End index into B
                block_value_idx);       // [out] Block index into B

            // Share block starting and ending indices
            temp_storage.block_segment_idx[threadIdx.x] = block_segment_idx;
            temp_storage.block_value_idx[threadIdx.x] = block_value_idx;

            // Initialize the block's running prefix
            if (threadIdx.x == 0)
            {
                prefix_op.running_total.key    = block_segment_idx;
                prefix_op.running_total.value  = identity;

                // Initialize the "first scattered partial reduction tuple" to the prefix tuple (in case we don't actually scatter one)
                temp_storage.first_tuple = prefix_op.running_total;
            }
        }

        // Ensure coherence of region indices
        __syncthreads();

        // Read block's starting indices
        Offset block_segment_idx        = temp_storage.block_segment_idx[0];
        Offset block_value_idx          = temp_storage.block_value_idx[0];

        // Have the thread block iterate over the region
        #pragma unroll 1
        while (block_diagonal < next_block_diagonal)
        {
            // Read block's ending indices (hoist?)
            Offset next_block_segment_idx   = temp_storage.block_segment_idx[1];
            Offset next_block_value_idx     = temp_storage.block_value_idx[1];
/*
            if (threadIdx.x == 0) CubLog("block diagonal %d next diagonal %d, segment idx [%d: %d], value idx [%d : %d]\n",
                block_diagonal,
                next_block_diagonal,
                block_segment_idx,
                next_block_segment_idx,
                block_value_idx,
                next_block_value_idx);
*/
            // Clamp the per-thread search range to within one work-tile of block's current indices
            Offset next_tile_segment_idx    = CUB_MIN(next_block_segment_idx,   block_segment_idx + TILE_ITEMS);
            Offset next_tile_value_idx      = CUB_MIN(next_block_value_idx,     block_value_idx + TILE_ITEMS);

            // Have each thread search for the end-indices of its subranges within the segment and value inputs
            Offset next_thread_diagonal = block_diagonal + ((threadIdx.x + 1) * ITEMS_PER_THREAD);
            Offset next_thread_segment_idx;
            Offset next_thread_value_idx;

            MergePathSearch2(
                next_thread_diagonal,           // Next thread diagonal
                d_segment_end_offsets,          // A (segment end-offsets)
                block_segment_idx,              // Start index into A
                next_tile_segment_idx,          // End index into A
                next_thread_segment_idx,        // [out] Thread index into A
                d_value_offsets,                // B (value offsets)
                block_value_idx,                // Start index into B
                next_tile_value_idx,            // End index into B
                next_thread_value_idx);         // [out] Thread index into B

            // Share thread end-indices
            temp_storage.thread_segment_idx[threadIdx.x + 1]   = next_thread_segment_idx;
            temp_storage.thread_value_idx[threadIdx.x + 1]     = next_thread_value_idx;

            // Ensure coherence of search indices
            __syncthreads();

            // Retrieve the block's starting indices for the next tile of work (i.e., the last thread's end-indices)
            next_tile_segment_idx   = temp_storage.thread_segment_idx[BLOCK_THREADS];
            next_tile_value_idx     = temp_storage.thread_value_idx[BLOCK_THREADS];

            if (block_segment_idx == next_tile_segment_idx)
            {
                // There are no segment end-offsets in this tile.  Perform a
                // simple block-wide reduction and accumulate the result into
                // the running total.

                // Load a tile's worth of values (using identity for out-of-bounds items)
                Value values[ITEMS_PER_THREAD];
                Offset num_values = next_tile_value_idx - block_value_idx;
                LoadStriped<BLOCK_THREADS>(threadIdx.x, d_values + block_value_idx, values, num_values, identity);

                // Barrier for smem reuse
                __syncthreads();

                // Reduce the tile of values and update the running total in thread-0
                KeyValuePair tile_aggregate;
                tile_aggregate.key = block_segment_idx;
                tile_aggregate.value = BlockReduce(temp_storage.reduce).Reduce(values, reduction_op);

                if (threadIdx.x == 0)
                {
                    prefix_op.running_total = scan_op(
                        prefix_op.running_total,
                        tile_aggregate);
                }
            }
/*            else if (block_value_idx == next_tile_value_idx)
            {
                // There are no values in this tile (only empty segments).  Write
                // out a tile of identity values to output.

                Value segment_reductions[ITEMS_PER_THREAD];

                if (threadIdx.x == 0)
                {
                    // The first segment gets the running segment total
                    segment_reductions[0] = prefix_op.running_total.value;

                    // Update the running prefix
                    prefix_op.running_total.value = identity;
                    prefix_op.running_total.key = next_tile_segment_idx;
                }
                else
                {
                    // Remainder of segments in this tile get identity
                    segment_reductions[0] = identity;
                }

                // Remainder of segments in this tile get identity
                #pragma unroll
                for (int ITEM = 1; ITEM < ITEMS_PER_THREAD; ++ITEM)
                    segment_reductions[ITEM] = identity;

                // Store reductions
                Offset num_segments = next_tile_segment_idx - block_segment_idx;
                StoreStriped<BLOCK_THREADS>(threadIdx.x, d_output + block_segment_idx, segment_reductions, num_segments);
            }
*/            else
            {
                // Merge the tile's segment and value indices

                // Get thread begin-index for segments
                Offset thread_segment_idx = (threadIdx.x == 0) ?
                    block_segment_idx :                             // First thread starts at the block's start
                    temp_storage.thread_segment_idx[threadIdx.x];   // Other threads start at their predecessor's end

                // Get thread begin-index for values
                Offset thread_value_idx = (threadIdx.x == 0) ?
                    block_value_idx :                               // First thread starts at the block's start
                    temp_storage.thread_value_idx[threadIdx.x];     // Other threads start at their predecessor's end
/*
                CubLog("\t thread segment idx %d:%d, value idx %d:%d\n",
                    thread_segment_idx, next_thread_segment_idx,
                    thread_value_idx, next_thread_value_idx);
*/
                // Barrier for smem reuse
                __syncthreads();

                // Check if first segment end-offset is in range
                bool valid_segment = (thread_segment_idx < next_thread_segment_idx);

                // Check if first value offset is in range
                bool valid_value = (thread_value_idx < next_thread_value_idx);

                // Load first segment end-offset
                Offset segment_end_offset = (valid_segment) ?
                    d_segment_end_offsets[thread_segment_idx] :
                    num_values;                                                     // Out of range (the last segment end-offset is one-past the last value offset)

                // Load first value offset
                Offset value_offset = (valid_value) ?
                    d_value_offsets[thread_value_idx] :
                    num_values;                                                     // Out of range (one-past the last value offset)

                // Assemble segment-demarcating tail flags and partial reduction tuples
                TailFlag        tail_flags[ITEMS_PER_THREAD];
                KeyValuePair    partial_reductions[ITEMS_PER_THREAD];

                #pragma unroll
                for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
                {
                    // Default tuple and flag values
                    partial_reductions[ITEM].key    = thread_segment_idx;
                    partial_reductions[ITEM].value  = identity;
                    tail_flags[ITEM]                = 0;

                    // Whether or not we slide (a) right along the segment path or (b) down the value path
                    if (valid_segment && (!valid_value || (segment_end_offset <= value_offset)))
                    {
                        // Consume this segment index

                        // Set tail flag noting the end of the segment
                        tail_flags[ITEM] = 1;

                        // Increment segment index
                        thread_segment_idx++;

                        // Read next segment end-offset (if valid)
                        if ((valid_segment = (thread_segment_idx < next_thread_segment_idx)))
                            segment_end_offset = d_segment_end_offsets[thread_segment_idx];
                    }
                    else if (valid_value)
                    {
                        // Consume this value index

                        // Update the tuple's value with the value at this index.
                        partial_reductions[ITEM].value = d_values[thread_value_idx];

                        // Increment value index
                        thread_value_idx++;

                        // Read next value offset (if valid)
                        if ((valid_value = (thread_value_idx < next_thread_value_idx)))
                            value_offset = d_value_offsets[thread_value_idx];
                    }
                }

/*
                CubLog("Tuples %s<%d,%.1f>, %s<%d,%.1f>\n",
                    tail_flags[0] ? "*" : "", partial_reductions[0].key, partial_reductions[0].value,
                    tail_flags[1] ? "*" : "", partial_reductions[1].key, partial_reductions[1].value);
*/
                // Use prefix scan to reduce values by segment-id.  The segment-reductions end up in items flagged as segment-tails.
                KeyValuePair block_aggregate;
                BlockScan(temp_storage.scan).InclusiveScan(
                    partial_reductions,             // Scan input
                    partial_reductions,             // Scan output
                    scan_op,                        // Scan operator
                    block_aggregate,                // Block-wide total (unused)
                    prefix_op);                     // Prefix operator for seeding the block-wide scan with the running total

/*
                CubLog("\t\t Scanned tuples %s<%d,%.1f>, %s<%d,%.1f>\n",
                    tail_flags[0] ? "*" : "", partial_reductions[0].key, partial_reductions[0].value,
                    tail_flags[1] ? "*" : "", partial_reductions[1].key, partial_reductions[1].value);
*/
                // The first segment index for this region (hoist?)
                Offset first_segment_idx = temp_storage.block_segment_idx[0];

                // Scatter an accumulated reduction if it is the head of a valid segment
                #pragma unroll
                for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
                {
                    if (tail_flags[ITEM])
                    {
                        Offset  segment_idx = partial_reductions[ITEM].key;
                        Value   value       = partial_reductions[ITEM].value;

                        // Write value reduction to corresponding segment id
                        d_output[segment_idx] = value;

                        // Save off the first value product that this thread block will scatter
                        if (segment_idx == first_segment_idx)
                        {
                            temp_storage.first_tuple.value = value;
                        }
                    }
                }
            }


            // Advance to the next region in the decision path
            block_diagonal += TILE_ITEMS;

            // Advance the block's indices in preparation for the next tile
            block_segment_idx   = next_tile_segment_idx;
            block_value_idx     = next_tile_value_idx;

            // Barrier for smem reuse
            __syncthreads();
        }

        // Get first and last tuples for the region
        if (threadIdx.x == 0)
        {
            first_tuple = temp_storage.first_tuple;
            last_tuple = prefix_op.running_total;
        }
    }


};








/******************************************************************************
 * Tuning policy types
 ******************************************************************************/

/**
 * Parameterizable tuning policy type for BlockSegReduceRegionByKey
 */
template <
    int                     _BLOCK_THREADS,             ///< Threads per thread block
    int                     _ITEMS_PER_THREAD,          ///< Items per thread (per tile of input)
    BlockLoadAlgorithm      _LOAD_ALGORITHM,            ///< The BlockLoad algorithm to use
    bool                    _LOAD_WARP_TIME_SLICING,    ///< Whether or not only one warp's worth of shared memory should be allocated and time-sliced among block-warps during any load-related data transpositions (versus each warp having its own storage)
    CacheLoadModifier       _LOAD_MODIFIER,             ///< Cache load modifier for reading input elements
    BlockScanAlgorithm      _SCAN_ALGORITHM>            ///< The BlockScan algorithm to use
struct BlockSegReduceRegionByKeyPolicy
{
    enum
    {
        BLOCK_THREADS           = _BLOCK_THREADS,               ///< Threads per thread block
        ITEMS_PER_THREAD        = _ITEMS_PER_THREAD,            ///< Items per thread (per tile of input)
        LOAD_WARP_TIME_SLICING  = _LOAD_WARP_TIME_SLICING,      ///< Whether or not only one warp's worth of shared memory should be allocated and time-sliced among block-warps during any load-related data transpositions (versus each warp having its own storage)    };
    };

    static const BlockLoadAlgorithm     LOAD_ALGORITHM          = _LOAD_ALGORITHM;      ///< The BlockLoad algorithm to use
    static const CacheLoadModifier      LOAD_MODIFIER           = _LOAD_MODIFIER;       ///< Cache load modifier for reading input elements
    static const BlockScanAlgorithm     SCAN_ALGORITHM          = _SCAN_ALGORITHM;      ///< The BlockScan algorithm to use
};


/******************************************************************************
 * Persistent thread block types
 ******************************************************************************/

/**
 * \brief BlockSegReduceRegionByKey implements a stateful abstraction of CUDA thread blocks for participating in device-wide reduce-value-by-key.
 */
template <
    typename    BlockSegReduceRegionByKeyPolicy,        ///< Parameterized BlockSegReduceRegionByKeyPolicy tuning policy
    typename    InputIterator,                          ///< Random-access iterator referencing key-value input tuples
    typename    OutputIterator,                         ///< Random-access iterator referencing segment output totals
    typename    ReductionOp>                            ///< Binary reduction operator type having member <tt>T operator()(const T &a, const T &b)</tt>
struct BlockSegReduceRegionByKey
{
    //---------------------------------------------------------------------
    // Types and constants
    //---------------------------------------------------------------------

    // Constants
    enum
    {
        BLOCK_THREADS       = BlockSegReduceRegionByKeyPolicy::BLOCK_THREADS,
        ITEMS_PER_THREAD    = BlockSegReduceRegionByKeyPolicy::ITEMS_PER_THREAD,
        TILE_ITEMS          = BLOCK_THREADS * ITEMS_PER_THREAD,
    };

    // KeyValuePair input type
    typedef typename std::iterator_traits<InputIterator>::value_type KeyValuePair;

    // Signed integer type for global offsets
    typedef typename KeyValuePair::Key Offset;

    // Value type
    typedef typename KeyValuePair::Value Value;

    // Head flag type
    typedef int HeadFlag;

    // Input iterator wrapper type for loading KeyValuePair elements through cache
    typedef CacheModifiedInputIterator<
            BlockSegReduceRegionByKeyPolicy::LOAD_MODIFIER,
            KeyValuePair,
            Offset>
        WrappedInputIterator;

    // Parameterized BlockLoad type
    typedef BlockLoad<
            WrappedInputIterator,
            BLOCK_THREADS,
            ITEMS_PER_THREAD,
            BlockSegReduceRegionByKeyPolicy::LOAD_ALGORITHM,
            BlockSegReduceRegionByKeyPolicy::LOAD_WARP_TIME_SLICING>
        BlockLoad;

    // BlockScan scan operator for reduction-by-segment
    typedef ReduceByKeyOp<ReductionOp> ReduceByKeyOp;

    // Stateful BlockScan prefix callback type for managing a running total while scanning consecutive tiles
    typedef RunningBlockPrefixCallbackOp<
            KeyValuePair,
            ReduceByKeyOp>
        RunningPrefixCallbackOp;

    // Parameterized BlockScan type for block-wide reduce-value-by-key
    typedef BlockScan<
            KeyValuePair,
            BLOCK_THREADS,
            BlockSegReduceRegionByKeyPolicy::SCAN_ALGORITHM>
        BlockScan;

    // Parameterized BlockDiscontinuity type for identifying key discontinuities
    typedef BlockDiscontinuity<
            Offset,
            BLOCK_THREADS>
        BlockDiscontinuity;

    // Operator for detecting discontinuities in a list of segment identifiers.
    struct NewSegmentOp
    {
        /// Returns true if row_b is the start of a new row
        __device__ __forceinline__ bool operator()(const Offset& b, const Offset& a)
        {
            return (a != b);
        }
    };

    // Shared memory type for this threadblock
    struct _TempStorage
    {
        union
        {
            typename BlockLoad::TempStorage                 load;           // Smem needed for tile loading
            struct {
                typename BlockScan::TempStorage             scan;           // Smem needed for reduce-value-by-segment scan
                typename BlockDiscontinuity::TempStorage    discontinuity;  // Smem needed for head-flagging
            };
        };
    };

    // Alias wrapper allowing storage to be unioned
    struct TempStorage : Uninitialized<_TempStorage> {};


    //---------------------------------------------------------------------
    // Thread fields
    //---------------------------------------------------------------------

    _TempStorage                &temp_storage;          ///< Reference to shared storage
    WrappedInputIterator        d_tuple_partials;       ///< A sequence of partial reduction tuples to scan
    OutputIterator              d_output;               ///< A sequence of segment totals
    Value                       identity;               ///< Identity value (for zero-length segments)
    ReduceByKeyOp               scan_op;                ///< Reduce-by-key scan operator
    RunningPrefixCallbackOp     prefix_op;              ///< Stateful running total for block-wide prefix scan of partial reduction tuples


    //---------------------------------------------------------------------
    // Operations
    //---------------------------------------------------------------------

    /**
     * Constructor
     */
    __device__ __forceinline__
    BlockSegReduceRegionByKey(
        TempStorage             &temp_storage,          ///< Reference to shared storage
        InputIterator           d_tuple_partials,       ///< A sequence of partial reduction tuples to scan
        OutputIterator          d_output,               ///< A sequence of segment totals
        Value                   identity,               ///< Identity value (for zero-length segments)
        ReductionOp             reduction_op)           ///< Reduction operator
    :
        temp_storage(temp_storage.Alias()),
        d_tuple_partials(d_tuple_partials),
        d_output(d_output),
        identity(identity),
        scan_op(reduction_op),
        prefix_op(scan_op)
    {}



    /**
     * Processes a reduce-value-by-key input tile, outputting reductions for each segment
     */
    template <bool FULL_TILE>
    __device__ __forceinline__
    void ProcessTile(
        Offset block_offset,
        Offset first_segment_idx,
        Offset last_segment_idx,
        int guarded_items = TILE_ITEMS)
    {
        KeyValuePair    partial_reductions[ITEMS_PER_THREAD];
        Offset          segment_ids[ITEMS_PER_THREAD];
        HeadFlag        head_flags[ITEMS_PER_THREAD];

        // Load a tile of block partials from previous kernel
        if (FULL_TILE)
        {
            // Full tile
            BlockLoad(temp_storage.load).Load(d_tuple_partials + block_offset, partial_reductions);
        }
        else
        {
            KeyValuePair oob_default;
            oob_default.key    = last_segment_idx;       // The last segment ID to be reduced
            oob_default.value  = identity;

            // Partially-full tile
            BlockLoad(temp_storage.load).Load(d_tuple_partials + block_offset, partial_reductions, guarded_items, oob_default);
        }

        // Barrier for shared memory reuse
        __syncthreads();

        // Copy the segment IDs for head-flagging
        #pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
        {
            segment_ids[ITEM] = partial_reductions[ITEM].key;
        }

        // Flag segment heads by looking for discontinuities
        BlockDiscontinuity(temp_storage.discontinuity).FlagHeads(
            head_flags,                         // [out] Head flags
            segment_ids,                        // Segment ids
            NewSegmentOp(),                     // Functor for detecting start of new rows
            prefix_op.running_total.key);       // Last segment ID from previous tile to compare with first segment ID in this tile

        // Reduce-value-by-segment across partial_reductions using exclusive prefix scan
        KeyValuePair block_aggregate;
        BlockScan(temp_storage.scan).ExclusiveScan(
            partial_reductions,                   // Scan input
            partial_reductions,                   // Scan output
            scan_op,                        // Scan operator
            block_aggregate,                // Block-wide total (unused)
            prefix_op);                     // Prefix operator for seeding the block-wide scan with the running total
/*
        CubLog("Scanned tuples %s<%d,%.1f>, %s<%d,%.1f>\n",
            head_flags[0] ? "*" : "", partial_reductions[0].key, partial_reductions[0].value,
            head_flags[1] ? "*" : "", partial_reductions[1].key, partial_reductions[1].value);
*/
        // Scatter an accumulated reduction if it is the head of a valid segment
        #pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
        {
            if (head_flags[ITEM])
            {
                d_output[partial_reductions[ITEM].key] = partial_reductions[ITEM].value;
            }
        }
    }


    /**
     * Iterate over input tiles belonging to this thread block
     */
    __device__ __forceinline__
    void ProcessRegion(
        Offset block_offset,
        Offset block_end,
        Offset first_segment_idx,
        Offset last_segment_idx)
    {
        if (threadIdx.x == 0)
        {
            // Initialize running prefix to the first segment index paired with identity
            prefix_op.running_total.key    = first_segment_idx;
            prefix_op.running_total.value  = identity;
        }

        // Process full tiles
        while (block_offset + TILE_ITEMS <= block_end)
        {
            ProcessTile<true>(block_offset, first_segment_idx, last_segment_idx);
            __syncthreads();

            block_offset += TILE_ITEMS;
        }

        // Process final value tile (if present)
        int guarded_items = block_end - block_offset;
        if (guarded_items)
        {
            ProcessTile<false>(block_offset, first_segment_idx, last_segment_idx, guarded_items);
        }
    }
};



/******************************************************************************
 * Kernel entrypoints
 ******************************************************************************/



/**
 * Segmented reduce region kernel entry point (multi-block).
 */
template <
    typename BlockSegReduceRegionPolicy,        ///< Parameterized BlockSegReduceRegionPolicy tuning policy
    typename SegmentOffsetIterator,             ///< Random-access input iterator type for reading segment end-offsets
    typename ValueIterator,                     ///< Random-access input iterator type for reading values
    typename OutputIterator,                    ///< Random-access output iterator type for writing segment reductions
    typename ReductionOp,                       ///< Binary reduction operator type having member <tt>T operator()(const T &a, const T &b)</tt>
    typename Offset,                            ///< Signed integer type for global offsets
    typename Value>                             ///< Value type
__launch_bounds__ (BlockSegReduceRegionPolicy::BLOCK_THREADS)
__global__ void SegReduceRegionKernel(
    SegmentOffsetIterator       d_segment_end_offsets,  ///< [in] A sequence of \p num_segments segment end-offsets
    ValueIterator               d_values,               ///< [in] A sequence of \p num_values values
    OutputIterator              d_output,               ///< [out] A sequence of \p num_segments segment totals
    KeyValuePair<Offset, Value> *d_tuple_partials,      ///< [out] A sequence of (gridDim.x * 2) partial reduction tuples
    Offset                      num_values,             ///< [in] Number of values to reduce
    Offset                      num_segments,           ///< [in] Number of segments being reduced
    Value                       identity,               ///< [in] Identity value (for zero-length segments)
    ReductionOp                 reduction_op,           ///< [in] Reduction operator
    GridEvenShare<Offset>       even_share)             ///< [in] Even-share descriptor for mapping an equal number of tiles onto each thread block
{
    typedef KeyValuePair<Offset, Value> KeyValuePair;

    // Specialize threadblock abstraction type for reducing a region of segmented values
    typedef BlockSegReduceRegion<
            BlockSegReduceRegionPolicy,
            SegmentOffsetIterator,
            ValueIterator,
            OutputIterator,
            ReductionOp,
            Offset>
        BlockSegReduceRegion;

    // Shared memory allocation
    __shared__ typename BlockSegReduceRegion::TempStorage temp_storage;

    // Initialize threadblock even-share to tell us where to start and stop our tile-processing
    even_share.BlockInit();

    // Construct persistent thread block
    BlockSegReduceRegion thread_block(
        temp_storage,
        d_segment_end_offsets,
        d_values,
        d_output,
        num_values,
        num_segments,
        identity,
        reduction_op);

    // First and last partial reduction tuples within the range (valid in thread-0)
    KeyValuePair first_tuple, last_tuple;

    // Consume block's region of work
    thread_block.ProcessBlockRegion(
        even_share.block_offset,
        even_share.block_end,
        first_tuple,
        last_tuple);

    if (threadIdx.x == 0)
    {
        if (gridDim.x > 1)
        {
            // Special case where the first segment written and the carry-out are for the same segment
            if (first_tuple.key == last_tuple.key)
            {
                first_tuple.value = identity;
            }

            // Write the first and last partial products from this thread block so
            // that they can be subsequently "fixed up" in the next kernel.
            d_tuple_partials[blockIdx.x * 2]          = first_tuple;
            d_tuple_partials[(blockIdx.x * 2) + 1]    = last_tuple;
        }
    }

}


/**
 * Segmented reduce region kernel entry point (single-block).
 */
template <
    typename    BlockSegReduceRegionByKeyPolicy,        ///< Parameterized BlockSegReduceRegionByKeyPolicy tuning policy
    typename    InputIterator,                          ///< Random-access iterator referencing key-value input tuples
    typename    OutputIterator,                         ///< Random-access iterator referencing segment output totals
    typename    ReductionOp,                            ///< Binary reduction operator type having member <tt>T operator()(const T &a, const T &b)</tt>
    typename    Offset,                                 ///< Signed integer type for global offsets
    typename    Value>                                  ///< Value type
__launch_bounds__ (BlockSegReduceRegionByKeyPolicy::BLOCK_THREADS, 1)
__global__ void SegReduceRegionByKeyKernel(
    InputIterator           d_tuple_partials,           ///< [in] A sequence of partial reduction tuples
    OutputIterator          d_output,                   ///< [out] A sequence of \p num_segments segment totals
    Offset                  num_segments,               ///< [in] Number of segments in the \p d_output sequence
    int                     num_tuple_partials,         ///< [in] Number of partial reduction tuples being reduced
    Value                   identity,                   ///< [in] Identity value (for zero-length segments)
    ReductionOp             reduction_op)               ///< [in] Reduction operator
{
    // Specialize threadblock abstraction type for reducing a region of values by key
    typedef BlockSegReduceRegionByKey<
            BlockSegReduceRegionByKeyPolicy,
            InputIterator,
            OutputIterator,
            ReductionOp>
        BlockSegReduceRegionByKey;

    // Shared memory allocation
    __shared__ typename BlockSegReduceRegionByKey::TempStorage temp_storage;

    // Construct persistent thread block
    BlockSegReduceRegionByKey thread_block(
        temp_storage,
        d_tuple_partials,
        d_output,
        identity,
        reduction_op);

    // Process input tiles
    thread_block.ProcessRegion(
        0,                          // Region start
        num_tuple_partials,         // Region end
        0,                          // First segment ID
        num_segments);              // Last segment ID (one-past)
}




/******************************************************************************
 * Dispatch
 ******************************************************************************/

/**
 * Utility class for dispatching the appropriately-tuned kernels for DeviceReduce
 */
template <
    typename ValueIterator,                     ///< Random-access input iterator type for reading values
    typename SegmentOffsetIterator,             ///< Random-access input iterator type for reading segment end-offsets
    typename OutputIterator,                    ///< Random-access output iterator type for writing segment reductions
    typename ReductionOp,                       ///< Binary reduction operator type having member <tt>T operator()(const T &a, const T &b)</tt>
    typename Offset>                            ///< Signed integer type for global offsets
struct DeviceSegReduceDispatch
{
    // Value type
    typedef typename std::iterator_traits<ValueIterator>::value_type Value;

    // Reduce-by-key data type tuple (segment-ID, value)
    typedef KeyValuePair<Offset, Value> KeyValuePair;



    /******************************************************************************
     * Tuning policies
     ******************************************************************************/

    /// SM35
    struct Policy350
    {
        // ReduceRegionPolicy
        typedef BlockSegReduceRegionPolicy<
                128,                            ///< Threads per thread block
                2, //12,                             ///< Items per thread (per tile of input)
                false,                          ///< Whether or not to cache incoming segment offsets in shared memory before reducing each tile
                false,                          ///< Whether or not to cache incoming values in shared memory before reducing each tile
                LOAD_LDG,                       ///< Cache load modifier for reading segment offsets
                LOAD_LDG,                       ///< Cache load modifier for reading values
                BLOCK_REDUCE_RAKING,            ///< The BlockReduce algorithm to use
                BLOCK_SCAN_RAKING_MEMOIZE>      ///< The BlockScan algorithm to use
            SegReduceRegionPolicy;

        // ReduceRegionByKeyPolicy
        typedef BlockSegReduceRegionByKeyPolicy<
                256,                            ///< Threads per thread block
                9,                             ///< Items per thread (per tile of input)
                BLOCK_LOAD_DIRECT,              ///< The BlockLoad algorithm to use
                false,                          ///< Whether or not only one warp's worth of shared memory should be allocated and time-sliced among block-warps during any load-related data transpositions (versus each warp having its own storage)
                LOAD_LDG,                       ///< Cache load modifier for reading input elements
                BLOCK_SCAN_WARP_SCANS>          ///< The BlockScan algorithm to use
            SegReduceRegionByKeyPolicy;
    };


    /// SM10
    struct Policy100
    {
        // ReduceRegionPolicy
        typedef BlockSegReduceRegionPolicy<
                128,                            ///< Threads per thread block
                3,                              ///< Items per thread (per tile of input)
                false,                          ///< Whether or not to cache incoming segment offsets in shared memory before reducing each tile
                false,                          ///< Whether or not to cache incoming values in shared memory before reducing each tile
                LOAD_DEFAULT,                   ///< Cache load modifier for reading segment offsets
                LOAD_DEFAULT,                   ///< Cache load modifier for reading values
                BLOCK_REDUCE_RAKING,            ///< The BlockReduce algorithm to use
                BLOCK_SCAN_RAKING>              ///< The BlockScan algorithm to use
            SegReduceRegionPolicy;

        // ReduceRegionByKeyPolicy
        typedef BlockSegReduceRegionByKeyPolicy<
                128,                            ///< Threads per thread block
                3,                              ///< Items per thread (per tile of input)
                BLOCK_LOAD_WARP_TRANSPOSE,      ///< The BlockLoad algorithm to use
                false,                          ///< Whether or not only one warp's worth of shared memory should be allocated and time-sliced among block-warps during any load-related data transpositions (versus each warp having its own storage)
                LOAD_DEFAULT,                   ///< Cache load modifier for reading input elements
                BLOCK_SCAN_WARP_SCANS>          ///< The BlockScan algorithm to use
            SegReduceRegionByKeyPolicy;
    };


    /******************************************************************************
     * Tuning policies of current PTX compiler pass
     ******************************************************************************/

#if (CUB_PTX_VERSION >= 350)
    typedef Policy350 PtxPolicy;
/*
#elif (CUB_PTX_VERSION >= 300)
    typedef Policy300 PtxPolicy;

#elif (CUB_PTX_VERSION >= 200)
    typedef Policy200 PtxPolicy;

#elif (CUB_PTX_VERSION >= 130)
    typedef Policy130 PtxPolicy;
*/
#else
    typedef Policy100 PtxPolicy;

#endif

    // "Opaque" policies (whose parameterizations aren't reflected in the type signature)
    struct PtxSegReduceRegionPolicy           : PtxPolicy::SegReduceRegionPolicy {};
    struct PtxSegReduceRegionByKeyPolicy      : PtxPolicy::SegReduceRegionByKeyPolicy {};


    /******************************************************************************
     * Utilities
     ******************************************************************************/

    /**
     * Initialize kernel dispatch configurations with the policies corresponding to the PTX assembly we will use
     */
    template <
        typename SegReduceKernelConfig,
        typename SegReduceByKeyKernelConfig>
    __host__ __device__ __forceinline__
    static void InitConfigs(
        int                         ptx_version,
        SegReduceKernelConfig       &seg_reduce_region_config,
        SegReduceByKeyKernelConfig  &seg_reduce_region_by_key_config)
    {
    #ifdef __CUDA_ARCH__

        // We're on the device, so initialize the kernel dispatch configurations with the current PTX policy
        seg_reduce_region_config.Init<PtxSegReduceRegionPolicy>();
        seg_reduce_region_by_key_config.Init<PtxSegReduceRegionByKeyPolicy>();

    #else

        // We're on the host, so lookup and initialize the kernel dispatch configurations with the policies that match the device's PTX version
        if (ptx_version >= 350)
        {
            seg_reduce_region_config.template          Init<typename Policy350::SegReduceRegionPolicy>();
            seg_reduce_region_by_key_config.template   Init<typename Policy350::SegReduceRegionByKeyPolicy>();
        }
/*
        else if (ptx_version >= 300)
        {
            seg_reduce_region_config.template          Init<typename Policy300::SegReduceRegionPolicy>();
            seg_reduce_region_by_key_config.template   Init<typename Policy300::SegReduceRegionByKeyPolicy>();
        }
        else if (ptx_version >= 200)
        {
            seg_reduce_region_config.template          Init<typename Policy200::SegReduceRegionPolicy>();
            seg_reduce_region_by_key_config.template   Init<typename Policy200::SegReduceRegionByKeyPolicy>();
        }
        else if (ptx_version >= 130)
        {
            seg_reduce_region_config.template          Init<typename Policy130::SegReduceRegionPolicy>();
            seg_reduce_region_by_key_config.template   Init<typename Policy130::SegReduceRegionByKeyPolicy>();
        }
*/
        else
        {
            seg_reduce_region_config.template          Init<typename Policy100::SegReduceRegionPolicy>();
            seg_reduce_region_by_key_config.template   Init<typename Policy100::SegReduceRegionByKeyPolicy>();
        }

    #endif
    }


    /**
     * SegReduceRegionKernel kernel dispatch configuration
     */
    struct SegReduceKernelConfig
    {
        int                     block_threads;
        int                     items_per_thread;
        bool                    use_smem_segment_cache;
        bool                    use_smem_value_cache;
        CacheLoadModifier       load_modifier_segments;
        CacheLoadModifier       load_modifier_values;
        BlockReduceAlgorithm    reduce_algorithm;
        BlockScanAlgorithm      scan_algorithm;

        template <typename SegReduceRegionPolicy>
        __host__ __device__ __forceinline__
        void Init()
        {
            block_threads               = SegReduceRegionPolicy::BLOCK_THREADS;
            items_per_thread            = SegReduceRegionPolicy::ITEMS_PER_THREAD;
            use_smem_segment_cache      = SegReduceRegionPolicy::USE_SMEM_SEGMENT_CACHE;
            use_smem_value_cache        = SegReduceRegionPolicy::USE_SMEM_VALUE_CACHE;
            load_modifier_segments      = SegReduceRegionPolicy::LOAD_MODIFIER_SEGMENTS;
            load_modifier_values        = SegReduceRegionPolicy::LOAD_MODIFIER_VALUES;
            reduce_algorithm            = SegReduceRegionPolicy::REDUCE_ALGORITHM;
            scan_algorithm              = SegReduceRegionPolicy::SCAN_ALGORITHM;
        }
    };

    /**
     * SegReduceRegionByKeyKernel kernel dispatch configuration
     */
    struct SegReduceByKeyKernelConfig
    {
        int                     block_threads;
        int                     items_per_thread;
        BlockLoadAlgorithm      load_algorithm;
        bool                    load_warp_time_slicing;
        CacheLoadModifier       load_modifier;
        BlockScanAlgorithm      scan_algorithm;

        template <typename SegReduceRegionByKeyPolicy>
        __host__ __device__ __forceinline__
        void Init()
        {
            block_threads               = SegReduceRegionByKeyPolicy::BLOCK_THREADS;
            items_per_thread            = SegReduceRegionByKeyPolicy::ITEMS_PER_THREAD;
            load_algorithm              = SegReduceRegionByKeyPolicy::LOAD_ALGORITHM;
            load_warp_time_slicing      = SegReduceRegionByKeyPolicy::LOAD_WARP_TIME_SLICING;
            load_modifier               = SegReduceRegionByKeyPolicy::LOAD_MODIFIER;
            scan_algorithm              = SegReduceRegionByKeyPolicy::SCAN_ALGORITHM;
        }
    };


    /******************************************************************************
     * Dispatch entrypoints
     ******************************************************************************/

    /**
     * Internal dispatch routine for computing a device-wide segmented reduction.
     */
    template <
        typename                        SegReduceRegionKernelPtr,               ///< Function type of cub::SegReduceRegionKernel
        typename                        SegReduceRegionByKeyKernelPtr>          ///< Function type of cub::SegReduceRegionByKeyKernel
    __host__ __device__ __forceinline__
    static cudaError_t Dispatch(
        void                            *d_temp_storage,                        ///< [in] %Device allocation of temporary storage.  When NULL, the required allocation size is returned in \p temp_storage_bytes and no work is done.
        size_t                          &temp_storage_bytes,                    ///< [in,out] Size in bytes of \p d_temp_storage allocation.
        ValueIterator                   d_values,                               ///< [in] A sequence of \p num_values data to reduce
        SegmentOffsetIterator           d_segment_offsets,                      ///< [in] A sequence of (\p num_segments + 1) segment offsets
        OutputIterator                  d_output,                               ///< [out] A sequence of \p num_segments segment totals
        Offset                          num_values,                             ///< [in] Total number of values to reduce
        Offset                          num_segments,                           ///< [in] Number of segments being reduced
        Value                           identity,                               ///< [in] Identity value (for zero-length segments)
        ReductionOp                     reduction_op,                           ///< [in] Reduction operator
        cudaStream_t                    stream,                                 ///< [in] <b>[optional]</b> CUDA stream to launch kernels within.  Default is stream<sub>0</sub>.
        bool                            debug_synchronous,                      ///< [in] <b>[optional]</b> Whether or not to synchronize the stream after every kernel launch to check for errors.  Also causes launch configurations to be printed to the console.  Default is \p false.
        int                             sm_version,                             ///< [in] SM version of target device to use when computing SM occupancy
        SegReduceRegionKernelPtr        seg_reduce_region_kernel,               ///< [in] Kernel function pointer to parameterization of cub::SegReduceRegionKernel
        SegReduceRegionByKeyKernelPtr   seg_reduce_region_by_key_kernel,        ///< [in] Kernel function pointer to parameterization of cub::SegReduceRegionByKeyKernel
        SegReduceKernelConfig           &seg_reduce_region_config,              ///< [in] Dispatch parameters that match the policy that \p seg_reduce_region_kernel was compiled for
        SegReduceByKeyKernelConfig      &seg_reduce_region_by_key_config)       ///< [in] Dispatch parameters that match the policy that \p seg_reduce_region_by_key_kernel was compiled for
    {
#ifndef CUB_RUNTIME_ENABLED

        // Kernel launch not supported from this device
        return CubDebug(cudaErrorNotSupported );

#else

        cudaError error = cudaSuccess;
        do
        {
            // Dispatch two kernels: (1) a multi-block segmented reduction
            // to reduce regions by block, and (2) a single-block reduce-by-key kernel
            // to "fix up" segments spanning more than one region.

            // Tile size of seg_reduce_region_kernel
            int tile_size = seg_reduce_region_config.block_threads * seg_reduce_region_config.items_per_thread;

            // Get device ordinal
            int device_ordinal;
            if (CubDebug(error = cudaGetDevice(&device_ordinal))) break;

            // Get SM count
            int sm_count;
            if (CubDebug(error = cudaDeviceGetAttribute (&sm_count, cudaDevAttrMultiProcessorCount, device_ordinal))) break;

            // Get SM occupancy for histogram_region_kernel
            int seg_reduce_region_sm_occupancy;
            if (CubDebug(error = MaxSmOccupancy(
                seg_reduce_region_sm_occupancy,
                sm_version,
                seg_reduce_region_kernel,
                seg_reduce_region_config.block_threads))) break;

            // Get device occupancy for histogram_region_kernel
            int seg_reduce_region_occupancy = seg_reduce_region_sm_occupancy * sm_count;

            // Even-share work distribution
            int num_diagonals = num_values + num_segments;                  // Total number of work items
            int subscription_factor = seg_reduce_region_sm_occupancy;       // Amount of CTAs to oversubscribe the device beyond actively-resident (heuristic)
            GridEvenShare<Offset> even_share(
                num_diagonals,
                seg_reduce_region_occupancy * subscription_factor,
                tile_size);

            // Get grid size for seg_reduce_region_kernel
            int seg_reduce_region_grid_size = even_share.grid_size;

            // Number of "fix-up" reduce-by-key tuples (2 per thread block)
            int num_tuple_partials = seg_reduce_region_grid_size * 2;

            // Temporary storage allocation requirements
            void* allocations[1];
            size_t allocation_sizes[1] =
            {
                num_tuple_partials * sizeof(KeyValuePair),     // bytes needed for "fix-up" reduce-by-key tuples
            };

            // Alias the temporary allocations from the single storage blob (or set the necessary size of the blob)
            if (CubDebug(error = AliasTemporaries(d_temp_storage, temp_storage_bytes, allocations, allocation_sizes))) break;
            if (d_temp_storage == NULL)
            {
                // Return if the caller is simply requesting the size of the storage allocation
                return cudaSuccess;
            }

            // Alias the allocation for "fix-up" tuples
            KeyValuePair *d_tuple_partials = (KeyValuePair*) allocations[0];

            // Log seg_reduce_region_kernel configuration
            if (debug_synchronous) CubLog("Invoking seg_reduce_region_kernel<<<%d, %d, 0, %lld>>>(), %d items per thread, %d SM occupancy\n",
                seg_reduce_region_grid_size, seg_reduce_region_config.block_threads, (long long) stream, seg_reduce_region_config.items_per_thread, seg_reduce_region_sm_occupancy);

            // Array of segment end-offsets
            SegmentOffsetIterator d_segment_end_offsets = d_segment_offsets + 1;

            // Invoke seg_reduce_region_kernel
            seg_reduce_region_kernel<<<seg_reduce_region_grid_size, seg_reduce_region_config.block_threads, 0, stream>>>(
                d_segment_end_offsets,
                d_values,
                d_output,
                d_tuple_partials,
                num_values,
                num_segments,
                identity,
                reduction_op,
                even_share);

            // Sync the stream if specified
            if (debug_synchronous && (CubDebug(error = SyncStream(stream)))) break;

            // Perform "fix-up" of region partial reductions if grid size is greater than one thread block
            if (seg_reduce_region_grid_size > 1)
            {
                // Log seg_reduce_region_by_key_kernel configuration
                if (debug_synchronous) CubLog("Invoking seg_reduce_region_by_key_kernel<<<%d, %d, 0, %lld>>>(), %d items per thread\n",
                    1, seg_reduce_region_by_key_config.block_threads, (long long) stream, seg_reduce_region_by_key_config.items_per_thread);

                // Invoke seg_reduce_region_by_key_kernel
                seg_reduce_region_by_key_kernel<<<1, seg_reduce_region_by_key_config.block_threads, 0, stream>>>(
                    d_tuple_partials,
                    d_output,
                    num_segments,
                    num_tuple_partials,
                    identity,
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
     * Internal dispatch routine for computing a device-wide segmented reduction.
     */
    __host__ __device__ __forceinline__
    static cudaError_t Dispatch(
        void                            *d_temp_storage,                        ///< [in] %Device allocation of temporary storage.  When NULL, the required allocation size is returned in \p temp_storage_bytes and no work is done.
        size_t                          &temp_storage_bytes,                    ///< [in,out] Size in bytes of \p d_temp_storage allocation.
        ValueIterator                   d_values,                               ///< [in] A sequence of \p num_values data to reduce
        SegmentOffsetIterator           d_segment_offsets,                      ///< [in] A sequence of (\p num_segments + 1) segment offsets
        OutputIterator                  d_output,                               ///< [out] A sequence of \p num_segments segment totals
        Offset                          num_values,                             ///< [in] Total number of values to reduce
        Offset                          num_segments,                           ///< [in] Number of segments being reduced
        Value                           identity,                               ///< [in] Identity value (for zero-length segments)
        ReductionOp                     reduction_op,                           ///< [in] Reduction operator
        cudaStream_t                    stream,                                 ///< [in] <b>[optional]</b> CUDA stream to launch kernels within.  Default is stream<sub>0</sub>.
        bool                            debug_synchronous)                      ///< [in] <b>[optional]</b> Whether or not to synchronize the stream after every kernel launch to check for errors.  Also causes launch configurations to be printed to the console.  Default is \p false.
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
            SegReduceKernelConfig seg_reduce_region_config;
            SegReduceByKeyKernelConfig seg_reduce_region_by_key_config;

            InitConfigs(ptx_version, seg_reduce_region_config, seg_reduce_region_by_key_config);

            // Dispatch
            if (CubDebug(error = Dispatch(
                d_temp_storage,
                temp_storage_bytes,
                d_values,
                d_segment_offsets,
                d_output,
                num_values,
                num_segments,
                identity,
                reduction_op,
                stream,
                debug_synchronous,
                ptx_version,            // Use PTX version instead of SM version because, as a statically known quantity, this improves device-side launch dramatically but at the risk of imprecise occupancy calculation for mismatches
                SegReduceRegionKernel<PtxSegReduceRegionPolicy, SegmentOffsetIterator, ValueIterator, OutputIterator, ReductionOp, Offset, Value>,
                SegReduceRegionByKeyKernel<PtxSegReduceRegionByKeyPolicy, KeyValuePair*, OutputIterator, ReductionOp, Offset, Value>,
                seg_reduce_region_config,
                seg_reduce_region_by_key_config))) break;
        }
        while (0);

        return error;

    }
};




/******************************************************************************
 * DeviceSegReduce
 *****************************************************************************/

/**
 * \brief DeviceSegReduce provides operations for computing a device-wide, parallel segmented reduction across data items residing within global memory.
 * \ingroup DeviceModule
 *
 * \par Overview
 * A <a href="http://en.wikipedia.org/wiki/Reduce_(higher-order_function)"><em>reduction</em></a> (or <em>fold</em>)
 * uses a binary combining operator to compute a single aggregate from a list of input elements.
 *
 * \par Usage Considerations
 * \cdp_class{DeviceReduce}
 *
 */
struct DeviceSegReduce
{
    /**
     * \brief Computes a device-wide segmented reduction using the specified binary \p reduction_op functor.
     *
     * \par
     * Does not support non-commutative reduction operators.
     *
     * \devicestorage
     *
     * \cdp
     *
     * \iterator
     *
     * \tparam ValueIterator            <b>[inferred]</b> Random-access input iterator type for reading values
     * \tparam SegmentOffsetIterator    <b>[inferred]</b> Random-access input iterator type for reading segment end-offsets
     * \tparam OutputIterator           <b>[inferred]</b> Random-access output iterator type for writing segment reductions
     * \tparam Value                    <b>[inferred]</b> Value type
     * \tparam ReductionOp              <b>[inferred]</b> Binary reduction operator type having member <tt>T operator()(const T &a, const T &b)</tt>
     */
    template <
        typename                ValueIterator,
        typename                SegmentOffsetIterator,
        typename                OutputIterator,
        typename                Value,
        typename                ReductionOp>
    __host__ __device__ __forceinline__
    static cudaError_t Reduce(
        void                    *d_temp_storage,                        ///< [in] %Device allocation of temporary storage.  When NULL, the required allocation size is returned in \p temp_storage_bytes and no work is done.
        size_t                  &temp_storage_bytes,                    ///< [in,out] Size in bytes of \p d_temp_storage allocation.
        ValueIterator           d_values,                               ///< [in] A sequence of \p num_values data to reduce
        SegmentOffsetIterator   d_segment_offsets,                      ///< [in] A sequence of (\p num_segments + 1) segment offsets
        OutputIterator          d_output,                               ///< [out] A sequence of \p num_segments segment totals
        int                     num_values,                             ///< [in] Total number of values to reduce
        int                     num_segments,                           ///< [in] Number of segments being reduced
        Value                   identity,                               ///< [in] Identity value (for zero-length segments)
        ReductionOp             reduction_op,                           ///< [in] Reduction operator
        cudaStream_t            stream              = 0,                ///< [in] <b>[optional]</b> CUDA stream to launch kernels within.  Default is stream<sub>0</sub>.
        bool                    debug_synchronous   = false)            ///< [in] <b>[optional]</b> Whether or not to synchronize the stream after every kernel launch to check for errors.  Also causes launch configurations to be printed to the console.  Default is \p false.
    {
        // Signed integer type for global offsets
        typedef int Offset;

        typedef DeviceSegReduceDispatch<
                ValueIterator,
                SegmentOffsetIterator,
                OutputIterator,
                ReductionOp,
                Offset>
            DeviceSegReduceDispatch;

        return DeviceSegReduceDispatch::Dispatch(
            d_temp_storage,
            temp_storage_bytes,
            d_values,
            d_segment_offsets,
            d_output,
            num_values,
            num_segments,
            identity,
            reduction_op,
            stream,
            debug_synchronous);
    }


    /**
     * \brief Computes a device-wide segmented sum using the addition ('+') operator.
     *
     * \par
     * Does not support non-commutative summation.
     *
     * \devicestorage
     *
     * \cdp
     *
     * \iterator
     *
     * \tparam ValueIterator            <b>[inferred]</b> Random-access input iterator type for reading values
     * \tparam SegmentOffsetIterator    <b>[inferred]</b> Random-access input iterator type for reading segment end-offsets
     * \tparam OutputIterator           <b>[inferred]</b> Random-access output iterator type for writing segment reductions
     */
    template <
        typename                ValueIterator,
        typename                SegmentOffsetIterator,
        typename                OutputIterator>
    __host__ __device__ __forceinline__
    static cudaError_t Sum(
        void                    *d_temp_storage,                        ///< [in] %Device allocation of temporary storage.  When NULL, the required allocation size is returned in \p temp_storage_bytes and no work is done.
        size_t                  &temp_storage_bytes,                    ///< [in,out] Size in bytes of \p d_temp_storage allocation.
        ValueIterator           d_values,                               ///< [in] A sequence of \p num_values data to reduce
        SegmentOffsetIterator   d_segment_offsets,                      ///< [in] A sequence of (\p num_segments + 1) segment offsets
        OutputIterator          d_output,                               ///< [out] A sequence of \p num_segments segment totals
        int                     num_values,                             ///< [in] Total number of values to reduce
        int                     num_segments,                           ///< [in] Number of segments being reduced
        cudaStream_t            stream              = 0,                ///< [in] <b>[optional]</b> CUDA stream to launch kernels within.  Default is stream<sub>0</sub>.
        bool                    debug_synchronous   = false)            ///< [in] <b>[optional]</b> Whether or not to synchronize the stream after every kernel launch to check for errors.  Also causes launch configurations to be printed to the console.  Default is \p false.
    {
        // Signed integer type for global offsets
        typedef int Offset;

        // Value type
        typedef typename std::iterator_traits<ValueIterator>::value_type Value;

        Value identity = Value();
        cub::Sum reduction_op;

        typedef DeviceSegReduceDispatch<
                ValueIterator,
                SegmentOffsetIterator,
                OutputIterator,
                cub::Sum,
                Offset>
            DeviceSegReduceDispatch;

        return DeviceSegReduceDispatch::Dispatch(
            d_temp_storage,
            temp_storage_bytes,
            d_values,
            d_segment_offsets,
            d_output,
            num_values,
            num_segments,
            identity,
            reduction_op,
            stream,
            debug_synchronous);
    }
};




//---------------------------------------------------------------------
// Test generation
//---------------------------------------------------------------------

/**
 * Initialize problem
 */
template <typename Offset, typename Value>
void Initialize(
    GenMode         gen_mode,
    Value           *h_values,
    vector<Offset>  &segment_offsets,
    int             num_values,
    int             avg_segment_size)
{
    // Initialize values
//    if (g_verbose) printf("Values: ");
    for (int i = 0; i < num_values; ++i)
    {
        InitValue(gen_mode, h_values[i], i);
//        if (g_verbose) std::cout << h_values[i] << ", ";
    }
//    if (g_verbose) printf("\n\n");

    // Initialize segment lengths
    const unsigned int  MAX_INTEGER         = -1u;
    const unsigned int  MAX_SEGMENT_LENGTH  = avg_segment_size * 2;
    const double        SCALE_FACTOR        = double(MAX_SEGMENT_LENGTH) / double(MAX_INTEGER);

    segment_offsets.push_back(0);

    Offset consumed = 0;
    Offset remaining = num_values;
    while (remaining > 0)
    {
        // Randomly sample a 32-bit unsigned int
        unsigned int segment_length;
        RandomBits(segment_length);

        // Scale to maximum segment length
        segment_length = (unsigned int) (double(segment_length) * SCALE_FACTOR);
        segment_length = CUB_MIN(segment_length, remaining);

        consumed += segment_length;
        remaining -= segment_length;

        segment_offsets.push_back(consumed);
    }
}


/**
 * Compute reference answer
 */
template <typename Offset, typename Value>
void ComputeReference(
    Value       *h_values,
    Offset      *h_segment_offsets,
    Value       *h_reference,
    int         num_segments,
    Value       identity)
{
    if (g_verbose) printf("%d segment reductions: ", num_segments);
    for (int segment = 0; segment < num_segments; ++segment)
    {
        h_reference[segment] = identity;

        for (int i = h_segment_offsets[segment]; i < h_segment_offsets[segment + 1]; ++i)
        {
            h_reference[segment] += h_values[i];
        }
        if (g_verbose) std::cout << h_reference[segment] << ", ";
    }
    if (g_verbose) printf("\n\n");
}


/**
 * Simple test of device
 */
template <
    bool            CDP,
    typename        Offset,
    typename        Value,
    typename        ReductionOp>
void Test(
    Offset          num_values,
    int             avg_segment_size,
    ReductionOp     reduction_op,
    Value           identity,
    char*           type_string)
{
    Value   *h_values = NULL;
    Value   *h_reference = NULL;
    Offset  *h_segment_offsets = NULL;

    printf("%d\n", num_values);

    // Initialize problem on host
    h_values = new Value[num_values];
    vector<Offset> segment_offsets;
    Initialize(UNIFORM, h_values, segment_offsets, num_values, avg_segment_size);

    // Allocate simple offsets array and copy STL vector into it
    h_segment_offsets = new Offset[segment_offsets.size()];
    for (int i = 0; i < segment_offsets.size(); ++i)
        h_segment_offsets[i] = segment_offsets[i];

    Offset num_segments = segment_offsets.size() - 1;
    if (g_verbose)
    {
        printf("%d segment offsets: ", num_segments);
        for (int i = 0; i < num_segments; ++i)
            std::cout << h_segment_offsets[i] << "(" << h_segment_offsets[i + 1] - h_segment_offsets[i] << "), ";
        if (g_verbose) std::cout << std::endl << std::endl;
    }

    // Solve problem on host
    h_reference = new Value[num_segments];
    ComputeReference(h_values, h_segment_offsets, h_reference, num_segments, identity);

    printf("\n\n%s cub::DeviceSegReduce::%s %d items (%d-byte %s), %d segments (%d-byte offset indices)\n",
        (CDP) ? "CDP device invoked" : "Host-invoked",
        (Equals<ReductionOp, Sum>::VALUE) ? "Sum" : "Reduce",
        num_values, (int) sizeof(Value), type_string,
        num_segments, (int) sizeof(Offset));
    fflush(stdout);

    // Allocate and initialize problem on device
    Value   *d_values = NULL;
    Offset  *d_segment_offsets = NULL;
    Value   *d_output = NULL;
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_values, sizeof(Value) * num_values));
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_segment_offsets, sizeof(Offset) * (num_segments + 1)));
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_output, sizeof(Value) * num_segments));
    CubDebugExit(cudaMemcpy(d_values, h_values, sizeof(Value) * num_values, cudaMemcpyHostToDevice));
    CubDebugExit(cudaMemcpy(d_segment_offsets, h_segment_offsets, sizeof(Offset) * (num_segments + 1), cudaMemcpyHostToDevice));

    // Request and allocate temporary storage
    void    *d_temp_storage = NULL;
    size_t  temp_storage_bytes = 0;
    CubDebugExit(DeviceSegReduce::Sum(d_temp_storage, temp_storage_bytes, d_values, d_segment_offsets, d_output, num_values, num_segments, 0, false));
    CubDebugExit(g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes));

    // Clear device output
    CubDebugExit(cudaMemset(d_output, 0, sizeof(Value) * num_segments));

    // Run warmup/correctness iteration
    CubDebugExit(DeviceSegReduce::Sum(d_temp_storage, temp_storage_bytes, d_values, d_segment_offsets, d_output, num_values, num_segments, 0, true));

    // Check for correctness (and display results, if specified)
    int compare = CompareDeviceResults(h_reference, d_output, num_segments, g_verbose, g_verbose);
    printf("\t%s", compare ? "FAIL" : "PASS");

    // Flush any stdout/stderr
    fflush(stdout);
    fflush(stderr);

    // Performance
    GpuTimer gpu_timer;
    gpu_timer.Start();
    for (int i = 0; i < g_timing_iterations; ++i)
    {
        CubDebugExit(DeviceSegReduce::Sum(d_temp_storage, temp_storage_bytes, d_values, d_segment_offsets, d_output, num_values, num_segments, 0, false));
    }
    gpu_timer.Stop();
    float elapsed_millis = gpu_timer.ElapsedMillis();

    // Display performance
    if (g_timing_iterations > 0)
    {
        float avg_millis = elapsed_millis / g_timing_iterations;
        float grate = float(num_values) / avg_millis / 1000.0 / 1000.0;
        float gbandwidth = grate * sizeof(Value);
        printf(", %.3f avg ms, %.3f billion items/s, %.3f logical GB/s", avg_millis, grate, gbandwidth);
    }

    // Device cleanup
    if (d_values) CubDebugExit(g_allocator.DeviceFree(d_values));
    if (d_segment_offsets) CubDebugExit(g_allocator.DeviceFree(d_segment_offsets));
    if (d_output) CubDebugExit(g_allocator.DeviceFree(d_output));
    if (d_temp_storage) CubDebugExit(g_allocator.DeviceFree(d_temp_storage));

    // Host cleanup
    if (h_values)           delete[] h_values;
    if (h_segment_offsets)  delete[] h_segment_offsets;
    if (h_reference)        delete[] h_reference;
}


/**
 * Main
 */
int main(int argc, char** argv)
{
    int num_values          = 32 * 1024 * 1024;
    int avg_segment_size    = 500;

    // Initialize command line
    CommandLineArgs args(argc, argv);
    g_verbose = args.CheckCmdLineFlag("v");
    args.GetCmdLineArgument("n", num_values);
    args.GetCmdLineArgument("ss", avg_segment_size);
    args.GetCmdLineArgument("i", g_timing_iterations);

    // Print usage
    if (args.CheckCmdLineFlag("help"))
    {
        printf("%s "
            "[--device=<device-id>] "
            "[--v] "
            "[--i=<timing iterations>] "
            "[--n=<input samples>]\n"
            "[--ss=<average segment size>]\n"
            "\n", argv[0]);
        exit(0);
    }

    // Initialize device
    CubDebugExit(args.DeviceInit());

    Test<false>((int) num_values, avg_segment_size, Sum(), (double) 0, CUB_TYPE_STRING(double));

    return 0;
}



