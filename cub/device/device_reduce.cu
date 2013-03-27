

#include <stdio.h>

#include <iterator>

#define CUB_STDERR

#include <cub.cuh>
#include <test_util.h>

using namespace cub;


/******************************************************************************
 * ReduceKernel
 *****************************************************************************/

enum DeviceReduceDistribution
{
    /**
     * \brief An "even-share" strategy for commutative reduction operators.
     *
     * \par Overview
     * The input is evenly partitioned into \p p segments, where \p p is
     * constant and corresponds loosely to the number of thread blocks that may
     * actively reside on the target device. Each segment is comprised of
     * consecutive tiles, where a tile is a small, constant-sized unit of input
     * to be processed to completion before the thread block terminates or
     * obtains more work.  The first kernel invokes \p p thread blocks, each
     * of which iteratively reduces a segment of <em>n</em>/<em>p</em> elements
     * in tile-size increments.  As tiles are consumed, each thread privately
     * aggregates its own partial reduction from the items it reads. The
     * per-thread partials are only cooperatively reduced in each thread block
     * after the last tile is consumed, at which point the per-block aggregates
     * are written out for further reduction. A second reduction kernel
     * consisting of a single block of threads is dispatched to reduce the
     * per-block partials, producing the final result aggregate.
     *
     * \par Usage Considerations
     * - Requires a constant <em>O</em>(<em>p</em>) temporary storage
     * - Does not support non-commutative reduction because the operator is
     *   applied to non-adjacent input elements
     * - May suffer from numerical instability for floating point data
     *   because partial sums are accumulated sequentially by thread blocks
     *   that process multiple tiles
     *
     * \par Performance Considerations
     * - Performs <em>O</em>(<em>n</em>) work and thus runs in <em>O</em>(<em>n</em>) time
     */
    DEVICE_REDUCE_EVEN_SHARE,


    /**
     * \brief A dynamic "queue-based" strategy for commutative reduction operators.
     *
     * \par Overview
     * The input is treated as a queue to be dynamically consumed by a grid of
     * thread blocks.  Work is atomically dequeued in tiles, where a tile is a
     * unit of input to be processed to completion before the thread block
     * terminates or obtains more work.  The grid size \p p is constant,
     * loosely corresponding to the number of thread blocks that may actively
     * reside on the target device.  As tiles are dequeued, each thread
     * privately aggregates its own partial reduction from the items it reads.
     * The per-thread partials are only cooperatively reduced in each thread
     * block when the queue becomes empty, at which point the per-block
     * aggregates are written out for further reduction. A second reduction
     * kernel consisting of a single block of threads is then dispatched to
     * reduce the per-block partials, producing the final result aggregate.
     *
     * \par Usage Considerations
     * - Requires a constant <em>O</em>(<em>p</em>) temporary storage
     * - Does not support non-commutative reduction because the operator is
     *   applied to non-adjacent input elements
     * - May suffer from numerical instability for floating point data
     *   because partial sums are accumulated sequentially by thread blocks
     *   that process multiple tiles
     *
     * \par Performance Considerations
     * - Performs <em>O</em>(<em>n</em>) work and thus runs in <em>O</em>(<em>n</em>) time
     * - May outperform DEVICE_REDUCE_EVEN_SHARE in the event that thread
     *   block scheduling is not uniformly fair
     */
    DEVICE_REDUCE_DYNAMIC,


    /**
     * \brief A recursive strategy emphasizing numerical stability for non-commutative reduction operators.
     *
     * \par Overview
     * The input is evenly partitioned into \p p segments of consecutive tiles
     * (where a tile is a small, constant-sized unit of input to be processed
     * to completion before the thread block terminates or obtains more work).
     * The number of segments \p p is constant, loosely corresponding to
     * the number of thread blocks that may actively reside on the target
     * device. The first kernel invokes \p p thread blocks, each assigned a
     * segment of <em>n</em>/<em>p</em> tiles to iteratively reduce.  The
     * per-thread partials are cooperatively reduced for each tile, thus
     * ensuring the reduction operator is only applied to adjacent inputs (or
     * to adjacent partial reductions).  The per-tile aggregate is then
     * written out for further reduction.  This process is repeated for
     * log<sub><em>t</em></sub>(<em>n</em>) kernel invocations (where
     * <em>t</em> is the tile size) until a final result aggregate is produced.
     *
     * \par Usage Considerations
     * - Requires <em>O</em>(<em>n</em>/<em>t</em>) temporary storage
     *
     * \par Performance Considerations
     * - Performs <em>O</em>(<em>n</em>) work and thus runs in <em>O</em>(<em>n</em>) time
     * - May underperform DEVICE_REDUCE_EVEN_SHARE or DEVICE_REDUCE_DYNAMIC from repeated kernel
     *   invocations that are not large enough to saturate the device.
     */
    DEVICE_REDUCE_STABLE,
};


/**
 * Tuning policy for BlockReduceTiles
 */
template <
    int                         _BLOCK_THREADS,
    int                         _ITEMS_PER_THREAD,
    DeviceReduceDistribution    _DISTRIBUTION,
    BlockLoadPolicy             _LOAD_POLICY,
    PtxLoadModifier             _LOAD_MODIFIER>
struct BlockReduceTilesPolicy
{
    enum
    {
        BLOCK_THREADS       = _BLOCK_THREADS,
        ITEMS_PER_THREAD    = _ITEMS_PER_THREAD,
    };

    static const DeviceReduceDistribution   DISTRIBUTION       = _DISTRIBUTION;
    static const BlockLoadPolicy            LOAD_POLICY        = _LOAD_POLICY;
    static const PtxLoadModifier            LOAD_MODIFIER      = _LOAD_MODIFIER;
};


/**
 * BlockReduceTiles threadblock abstraction
 */
template <
    typename BlockReduceTilesPolicy,
    typename InputIterator,
    typename SizeT>
struct BlockReduceTiles
{
    //---------------------------------------------------------------------
    // Types and constants
    //---------------------------------------------------------------------

    // Data type of input iterator
    typedef typename std::iterator_traits<InputIterator>::value_type T;

    // Constants
    enum
    {
        BLOCK_THREADS       = BlockReduceTilesPolicy::BLOCK_THREADS,
        ITEMS_PER_THREAD    = BlockReduceTilesPolicy::ITEMS_PER_THREAD,
        TILE_ITEMS          = BLOCK_THREADS * ITEMS_PER_THREAD,
    };

    static const BlockLoadPolicy LOAD_POLICY        = BlockReduceTilesPolicy::LOAD_POLICY;
    static const PtxLoadModifier LOAD_MODIFIER      = BlockReduceTilesPolicy::LOAD_MODIFIER;

    // Parameterized BlockLoad primitive
    typedef BlockLoad<InputIterator, BLOCK_THREADS, ITEMS_PER_THREAD, LOAD_POLICY, LOAD_MODIFIER> BlockLoadT;

    // Parameterized BlockReduce primitive
    typedef BlockReduce<T, BLOCK_THREADS> BlockReduceT;

    // Shared memory type for this threadblock
    struct SmemStorage
    {
        SizeT block_offset;                                 // Location where to dequeue input for dynamic operation
        union
        {
            typename BlockLoadT::SmemStorage    load;       // Smem needed for cooperative loading
            typename BlockReduceT::SmemStorage  reduce;     // Smem needed for cooperative reduction
        };
    };


    //---------------------------------------------------------------------
    // Operations
    //---------------------------------------------------------------------

    /**
     * Process a single, full tile
     *
     * Each thread reduces only the values it loads.
     */
    template <
        bool FIRST_TILE,
        typename SizeT,
        typename ReductionOp>
    static __device__ __forceinline__ void ConsumeFullTile(
        SmemStorage             &smem_storage,
        InputIterator           d_in,
        SizeT                   block_offset,
        ReductionOp             &reduction_op,
        T                       &thread_aggregate)      ///< [in, out]
    {
        T items[ITEMS_PER_THREAD];

        BlockLoadT::Load(smem_storage.load, d_in + block_offset, items);
        T partial = ThreadReduce(items, reduction_op);

        thread_aggregate = (FIRST_TILE) ?
            partial :
            reduction_op(thread_aggregate, partial);
    }

    /**
     * Process a single, partial tile
     *
     * Each thread reduces only the strided values it loads.
     */
    template <
        bool FIRST_TILE,
        typename SizeT,
        typename ReductionOp>
    static __device__ __forceinline__ void ConsumePartialTile(
        SmemStorage             &smem_storage,
        InputIterator           d_in,
        SizeT                   block_offset,
        SizeT                   block_oob,
        ReductionOp             &reduction_op,
        T                       &thread_aggregate)      ///< [in, out]
    {
        // Process loads singly
        SizeT thread_offset = block_offset + threadIdx.x;

        if ((FIRST_TILE) && (thread_offset < block_oob))
        {
            thread_aggregate = ThreadLoad<LOAD_MODIFIER>(d_in + thread_offset);
            thread_offset += BLOCK_THREADS;
        }

        while (thread_offset < block_oob)
        {
            T item = ThreadLoad<LOAD_MODIFIER>(d_in + thread_offset);
            thread_aggregate = reduction_op(thread_aggregate, item);
            thread_offset += BLOCK_THREADS;
        }
    }


    /**
     * Process tiles using even-share
     */
    template <
        typename OutputIterator,
        typename SizeT,
        typename ReductionOp>
    static __device__ __forceinline__ void ProcessTiles(
        SmemStorage             &smem_storage,
        InputIterator           d_in,
        OutputIterator          d_out,
        SizeT                   block_offset,
        const SizeT             &block_oob,
        ReductionOp             &reduction_op)
    {
        if (block_offset + TILE_ITEMS <= block_oob)
        {
            // We have at least one full tile to consume
            T thread_aggregate;
            ConsumeFullTile<true>(smem_storage, d_in, block_offset, reduction_op, thread_aggregate);
            block_offset += TILE_ITEMS;

            __syncthreads();

            // Consume any other full tiles
            while (block_offset + TILE_ITEMS <= block_oob)
            {
                ConsumeFullTile<false>(smem_storage, d_in, block_offset, reduction_op, thread_aggregate);
                block_offset += TILE_ITEMS;

                __syncthreads();
            }

            // Consume any remaining input
            ConsumePartialTile<false>(smem_storage, d_in, block_offset, block_oob, reduction_op, thread_aggregate);

            __syncthreads();

            // Compute and output the block-wide reduction (every thread has a valid input)
            T block_aggregate = BlockReduceT::Reduce(smem_storage, thread_aggregate, reduction_op);
            if (threadIdx.x == 0)
            {
                d_out[blockIdx.x] = block_aggregate;
            }

        }
        else
        {
            // We have less than a full tile to consume
            T thread_aggregate;
            ConsumePartialTile<true>(smem_storage, d_in, block_offset, block_oob, reduction_op, thread_aggregate);

            __syncthreads();

            // Compute and output the block-wide reduction  (up to block_items threads have valid inputs)
            SizeT block_items = block_oob - block_offset;
            T block_aggregate = BlockReduceT::Reduce(smem_storage, thread_aggregate, reduction_op, block_items);
            if (threadIdx.x == 0)
            {
                d_out[blockIdx.x] = block_aggregate;
            }
        }
    }


    /**
     * Process tiles using dynamic dequeuing
     */
    template <
        typename OutputIterator,
        typename SizeT,
        typename ReductionOp>
    static __device__ __forceinline__ void ProcessTiles(
        SmemStorage             &smem_storage,
        InputIterator           d_in,
        OutputIterator          d_out,
        SizeT                   num_items,
        GridQueue<SizeT>        &queue,
        ReductionOp             &reduction_op)
    {
        // Each thread block is statically assigned at least one (potentially
        // partial) tile of input, otherwise its block_aggregate will be undefined.
        SizeT block_offset = SizeT(blockIdx.x) * TILE_ITEMS;
        SizeT block_oob = num_items;

        if (block_offset + TILE_ITEMS <= block_oob)
        {
            // We have a full tile to consume
            T thread_aggregate;
            ConsumeFullTile<true>(smem_storage, d_in, block_offset, reduction_op, thread_aggregate);
            block_offset += TILE_ITEMS;

            __syncthreads();

            // Dynamically consume other tiles
            while (true)
            {
                // Dequeue up to TILE_ITEMS
                if (threadIdx.x == 0)
                {
                    smem_storage.block_offset = queue.Drain(TILE_ITEMS);
                }

                __syncthreads();

                block_offset = smem_storage.block_offset;
                if (block_offset + TILE_ITEMS <= block_oob)
                {
                    // We have a full tile to consume
                    ConsumeFullTile<false>(smem_storage, d_in, block_offset, reduction_op, thread_aggregate);
                }
                else
                {
                    // We have less than a full tile to consume
                    ConsumePartialTile<false>(smem_storage, d_in, block_offset, block_oob, reduction_op, thread_aggregate);
                }

                __syncthreads();
            }

            // Compute and output the block-wide reduction (every thread has a valid input)
            T block_aggregate = BlockReduceT::Reduce(smem_storage, thread_aggregate, reduction_op);
            if (threadIdx.x == 0)
            {
                d_out[blockIdx.x] = block_aggregate;
            }
        }
        else
        {
            // We have less than a full tile to consume
            T thread_aggregate;
            SizeT block_items = block_oob - block_offset;
            ConsumePartialTile<true>(smem_storage, d_in, block_offset, block_oob, reduction_op, thread_aggregate);

            __syncthreads();

            // Compute and output the block-wide reduction  (up to block_items threads have valid inputs)
            T block_aggregate = BlockReduceT::Reduce(smem_storage, thread_aggregate, reduction_op, block_items);
            if (threadIdx.x == 0)
            {
                d_out[blockIdx.x] = block_aggregate;
            }
        }
    }


    /**
     * Process tiles
     */
    template <
        typename OutputIterator,
        typename SizeT,
        typename ReductionOp>
    static __device__ __forceinline__ void ProcessTiles(
        SmemStorage             &smem_storage,
        InputIterator           d_in,
        OutputIterator          d_out,
        SizeT                   num_items,
        GridEvenShare<SizeT>    &even_share,
        GridQueue<SizeT>        &queue,
        ReductionOp             &reduction_op)
    {
        if (BlockReduceTilesPolicy::DISTRIBUTION == DEVICE_REDUCE_EVEN_SHARE)
        {
            // Even share
            even_share.BlockInit();
            ProcessTiles(smem_storage, d_in, d_out, even_share.block_offset, even_share.block_oob, reduction_op);
        }
        else
        {
            // Dynamically dequeue
            ProcessTiles(smem_storage, d_in, d_out, num_items, queue, reduction_op);
        }
    }

};


/******************************************************************************
 * Kernels
 *****************************************************************************/

/**
 * \brief Reduction kernel entry point.
 */
template <
    typename BlockReduceTilesPolicy,
    typename InputIterator,
    typename OutputIterator,
    typename SizeT,
    typename ReductionOp>
__global__ void ReduceKernel(
    InputIterator           d_in,
    OutputIterator          d_out,
    SizeT                   num_items,
    GridEvenShare<SizeT>    even_share,
    GridQueue<SizeT>        queue,
    ReductionOp             reduction_op)
{
    typedef BlockReduceTiles <BlockReduceTilesPolicy, InputIterator, SizeT> BlockReduceTilesT;

    __shared__ typename BlockReduceTilesT::SmemStorage smem_storage;

    BlockReduceTilesT::ProcessTiles(smem_storage, d_in, d_out, num_items, even_share);
}


/**
 * \brief Single-block reduction kernel entry point.
 */
template <
    typename BlockReduceTilesPolicy,
    typename InputIterator,
    typename OutputIterator,
    typename SizeT,
    typename ReductionOp>
__global__ void SingleReduceKernel(
    InputIterator           d_in,
    OutputIterator          d_out,
    SizeT                   num_items,
    ReductionOp             reduction_op)
{
    typedef BlockReduceTiles <BlockReduceTilesPolicy, InputIterator, SizeT> BlockReduceTilesT;

    __shared__ typename BlockReduceTilesT::SmemStorage smem_storage;

    BlockReduceTilesT::ProcessTiles(smem_storage, d_in, d_out, 0, num_items, reduction_op);
}



/******************************************************************************
 * DeviceFoo
 *****************************************************************************/

/**
 * Provides Foo operations on device-global data sets.
 */
struct DeviceFoo
{
    struct DispatchPolicy
    {
        int reduce_block_threads;
        int reduce_items_per_thread;
        int downsweep_block_threads;
        int downsweep_items_per_thread;

        template <typename BlockReduceTilesPolicy, typename ReduceSingleKernelPolicy>
        __host__ __device__ __forceinline__
        void Init()
        {
            reduce_block_threads       = BlockReduceTilesPolicy::BLOCK_THREADS;
            reduce_items_per_thread    = BlockReduceTilesPolicy::ITEMS_PER_THREAD;
            downsweep_block_threads     = ReduceSingleKernelPolicy::BLOCK_THREADS;
            downsweep_items_per_thread  = ReduceSingleKernelPolicy::ITEMS_PER_THREAD;
        }
    };


    // Default tuning policy types
    template <
        typename T,
        typename SizeT>
    struct DefaultPolicy
    {

        typedef BlockReduceTilesPolicy<     64,     1>      BlockReduceTilesPolicy300;
        typedef ReduceSingleKernelPolicy<   64,     1>      ReduceSingleKernelPolicy300;

        typedef BlockReduceTilesPolicy<     128,    1>      BlockReduceTilesPolicy200;
        typedef ReduceSingleKernelPolicy<   128,    1>      ReduceSingleKernelPolicy200;

        typedef BlockReduceTilesPolicy<     256,    1>      BlockReduceTilesPolicy100;
        typedef ReduceSingleKernelPolicy<   256,    1>      ReduceSingleKernelPolicy100;

    #if CUB_PTX_ARCH >= 300
        struct PtxBlockReduceTilesPolicy :      BlockReduceTilesPolicy300 {};
        struct PtxReduceSingleKernelPolicy :    ReduceSingleKernelPolicy300 {};
    #elif CUB_PTX_ARCH >= 200
        struct PtxBlockReduceTilesPolicy :      BlockReduceTilesPolicy200 {};
        struct PtxReduceSingleKernelPolicy :    ReduceSingleKernelPolicy200 {};
    #else
        struct PtxBlockReduceTilesPolicy :      BlockReduceTilesPolicy100 {};
        struct PtxReduceSingleKernelPolicy :    ReduceSingleKernelPolicy100 {};
    #endif

        /**
         * Initialize dispatch policy
         */
        static void InitDispatchPolicy(int device_arch, DispatchPolicy &dispatch_policy)
        {
            if (device_arch >= 300)
                dispatch_policy.Init<BlockReduceTilesPolicy300, ReduceSingleKernelPolicy300>();
            else if (device_arch >= 200)
                dispatch_policy.Init<BlockReduceTilesPolicy200, ReduceSingleKernelPolicy200>();
            else
                dispatch_policy.Init<BlockReduceTilesPolicy100, ReduceSingleKernelPolicy100>();
        }
    };


    // Internal Foo dispatch
    template <
        typename ReduceKernelPtr,
        typename ReduceSingleKernelPtr,
        typename T,
        typename SizeT>
    __host__ __device__ __forceinline__
    static cudaError_t Foo(
        ReduceKernelPtr     reduce_kernel_ptr,
        ReduceSingleKernelPtr   downsweep_kernel_ptr,
        DispatchPolicy          &dispatch_policy,
        T                       *d_in,
        T                       *d_out,
        SizeT                   num_elements)
    {
    #if !CUB_CNP_ENABLED

        // Kernel launch not supported from this device
        return cudaErrorInvalidConfiguration;

    #else

        cudaError error = cudaSuccess;
        do
        {
            // Get GPU ordinal
            int device_ordinal;
            if ((error = CubDebug(cudaGetDevice(&device_ordinal)))) break;

            // Get SM count
            int sm_count;
            if (CubDebug(error = cudaDeviceGetAttribute (&sm_count, cudaDevAttrMultiProcessorCount, device_ordinal))) break;

            // Rough estimate of maximum SM occupancies based upon PTX assembly
            int reduce_sm_occupancy        = ArchProps<CUB_PTX_ARCH>::MAX_SM_THREADBLOCKS;
            int downsweep_sm_occupancy      = ArchProps<CUB_PTX_ARCH>::MAX_SM_THREADBLOCKS;
            int oversubscription            = ArchProps<CUB_PTX_ARCH>::OVERSUBSCRIPTION;

        #if (CUB_PTX_ARCH == 0)

            // We're on the host, so come up with a more accurate estimate of SM occupancies from actual device properties
            DeviceProps device_props;
            if (CubDebug(error = device_props.Init(device_ordinal))) break;
            oversubscription = device_props.oversubscription;

            if (CubDebug(error = device_props.MaxSmOccupancy(
                reduce_sm_occupancy,
                reduce_kernel_ptr,
                dispatch_policy.reduce_block_threads))) break;

            if (CubDebug(error = device_props.MaxSmOccupancy(
                downsweep_sm_occupancy,
                downsweep_kernel_ptr,
                dispatch_policy.downsweep_block_threads))) break;
        #endif

            // Construct work distributions
            GridEvenShare<SizeT> reduce_distrib(
                num_elements,
                reduce_sm_occupancy * sm_count * oversubscription,
                dispatch_policy.reduce_block_threads * dispatch_policy.reduce_items_per_thread);

            GridEvenShare<SizeT> downsweep_distrib(
                num_elements,
                downsweep_sm_occupancy * sm_count * oversubscription,
                dispatch_policy.downsweep_block_threads * dispatch_policy.downsweep_items_per_thread);

            printf("Invoking Reduce<<<%d, %d>>>()\n",
                reduce_distrib.grid_size,
                dispatch_policy.reduce_block_threads);

            // Invoke Reduce
            reduce_kernel_ptr<<<reduce_distrib.grid_size, dispatch_policy.reduce_block_threads>>>(
                d_in, d_out, num_elements);

            printf("Invoking ReduceSingle<<<%d, %d>>>()\n",
                downsweep_distrib.grid_size,
                dispatch_policy.downsweep_block_threads);

            // Invoke ReduceSingle
            downsweep_kernel_ptr<<<downsweep_distrib.grid_size, dispatch_policy.downsweep_block_threads>>>(
                d_in, d_out, num_elements);
        }
        while (0);
        return error;

    #endif
    }


    //---------------------------------------------------------------------
    // Public interface
    //---------------------------------------------------------------------


    /**
     * Invoke Foo operation with custom policies
     */
    template <
        typename BlockReduceTilesPolicy,
        typename ReduceSingleKernelPolicy,
        typename T,
        typename SizeT>
    __host__ __device__ __forceinline__
    static cudaError_t Foo(
        T       *d_in,
        T       *d_out,
        SizeT   num_elements)
    {
        DispatchPolicy dispatch_policy;
        dispatch_policy.Init<BlockReduceTilesPolicy, ReduceSingleKernelPolicy>();

        return Foo(
            ReduceKernel<BlockReduceTilesPolicy, T, SizeT>,
            ReduceSingleKernel<ReduceSingleKernelPolicy, T, SizeT>,
            dispatch_policy,
            d_in,
            d_out,
            num_elements);
    }


    /**
     * Invoke Foo operation with default policies
     */
    template <
        typename T,
        typename SizeT>
    __host__ __device__ __forceinline__
    static cudaError_t Foo(
        T       *d_in,
        T       *d_out,
        SizeT   num_elements)
    {
        cudaError error = cudaSuccess;
        do
        {
            typedef typename DefaultPolicy<T, SizeT>::PtxBlockReduceTilesPolicy PtxBlockReduceTilesPolicy;
            typedef typename DefaultPolicy<T, SizeT>::PtxReduceSingleKernelPolicy PtxReduceSingleKernelPolicy;

            // Initialize dispatch policy
            DispatchPolicy dispatch_policy;

        #if CUB_PTX_ARCH > 0

            // We're on the device, so initialize the tuned dispatch policy based upon PTX arch
            dispatch_policy.Init<PtxBlockReduceTilesPolicy, PtxReduceSingleKernelPolicy>();

        #else

            // We're on the host, so initialize the tuned dispatch policy based upon the device's SM arch
            int device_ordinal;
            if (CubDebug(error = cudaGetDevice(&device_ordinal))) break;

            int major, minor;
            if (CubDebug(error = cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device_ordinal))) break;
            if (CubDebug(error = cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device_ordinal))) break;
            int device_arch = major * 100 + minor * 10;

            DefaultPolicy<T, SizeT>::InitDispatchPolicy(device_arch, dispatch_policy);

        #endif

            // Dispatch
            if (CubDebug(error = Foo(
                ReduceKernel<PtxBlockReduceTilesPolicy, T, SizeT>,
                ReduceSingleKernel<PtxReduceSingleKernelPolicy, T, SizeT>,
                dispatch_policy,
                d_in,
                d_out,
                num_elements))) break;
        }
        while (0);

        return error;
    }

};


/******************************************************************************
 * User kernel
 *****************************************************************************/

/**
 * User kernel for testing nested invocation of DeviceFoo::Foo
 */
template <typename T, typename SizeT>
__global__ void UserKernel(T *d_in, T *d_out, SizeT num_elements)
{
    DeviceFoo::Foo(d_in, d_out, num_elements);
}


/******************************************************************************
 * Main
 *****************************************************************************/

/**
 * Main
 */
int main(int argc, char** argv)
{
    typedef int T;
    typedef int SizeT;

    // Initialize command line
    CommandLineArgs args(argc, argv);

    // Initialize device
    CubDebugExit(args.DeviceInit());

    T *d_in             = NULL;
    T *d_out            = NULL;
    SizeT num_elements  = 1024 * 1024;

    // Invoke Foo from host
    DeviceFoo::Foo(d_in, d_out, num_elements);

    // Invoke Foo from device
    UserKernel<<<1,1>>>(d_in, d_out, num_elements);

    CubDebug(cudaDeviceSynchronize());

    return 0;
}



