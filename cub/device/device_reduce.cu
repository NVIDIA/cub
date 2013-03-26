

#include <stdio.h>

#define CUB_STDERR

#include <cub.cuh>
#include <test_util.h>

using namespace cub;


/******************************************************************************
 * ReduceKernel
 *****************************************************************************/

enum DeviceReducePolicy
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
     * \brief An "even-share" strategy for non-commutative reduction operators.
     *
     * \par Overview
     * The input is evenly partitioned into \p p segments, where \p p is
     * constant and corresponds loosely to the number of thread blocks that may
     * actively reside on the target device. Each segment is comprised of
     * consecutive tiles, where a tile is a small, constant-sized unit of input
     * to be processed to completion before the thread block terminates or
     * obtains more work.  The first kernel invokes \p p thread blocks, each
     * of which iteratively reduces a segment of <em>n</em>/<em>p</em> elements
     * in tile-size increments.  The per-thread partials are cooperatively
     * reduced for each tile, thus ensuring the reduction operator is only
     * applied to adjacent inputs (or to adjacent partial reductions).  A
     * running block-wide aggregate is curried between tiles as they are
     * consumed.   When the segment is finished, the per-block aggregate is
     * written out for further reduction. A second reduction kernel consisting
     * of a single block of threads is dispatched to reduce the per-block
     * partials, producing the final result aggregate.
     *
     * \par Usage Considerations
     * - Requires a constant <em>O</em>(<em>p</em>) temporary storage
     * - May suffer from numerical instability for floating point data
     *   because partial sums are accumulated sequentially by thread blocks
     *   that process multiple tiles
     *
     * \par Performance Considerations
     * - Performs <em>O</em>(<em>n</em>) work and thus runs in <em>O</em>(<em>n</em>) time
     * - May underperform DEVICE_REDUCE_EVEN_SHARE or DEVICE_REDUCE_DEQUEUE due to
     *   the overhead of cooperatively reducing every tile
     */
    DEVICE_REDUCE_NON_COMMUTATIVE,


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
     * - May underperform DEVICE_REDUCE_NON_COMMUTATIVE from repeated kernel
     *   invocations that are not large enough to saturate the device.
     */
    DEVICE_REDUCE_STABLE,
};



// ReduceKernel tuning policy
template <
    int _BLOCK_THREADS,
    int _ITEMS_PER_THREAD>
struct ReduceKernelPolicy
{
    enum
    {
        BLOCK_THREADS      = _BLOCK_THREADS,
        ITEMS_PER_THREAD   = _ITEMS_PER_THREAD,
    };
};


//
/**
 * \brief Reduction kernel entry point.
 *
 * Each thread block produces (at least) one partial reduction.
 */
template <
    typename ReduceKernelPolicy,
    typename T,
    typename SizeT>
__global__ void ReduceKernel(T *d_in, T *d_out, SizeT num_elements)
{
    if ((blockIdx.x == 0) && (threadIdx.x == 0)) printf("ReduceKernel BLOCK_THREADS(%d) ITEMS_PER_THREAD(%d)\n",
        ReduceKernelPolicy::BLOCK_THREADS,
        ReduceKernelPolicy::ITEMS_PER_THREAD);
}


/******************************************************************************
 * ReduceSingleKernel
 *****************************************************************************/


// ReduceSingleKernel tuning policy
template <
    int _BLOCK_THREADS,
    int _ITEMS_PER_THREAD>
struct ReduceSingleKernelPolicy
{
    enum
    {
        BLOCK_THREADS      = _BLOCK_THREADS,
        ITEMS_PER_THREAD   = _ITEMS_PER_THREAD,
    };
};


// ReduceSingleKernel entry point
template <
    typename ReduceSingleKernelPolicy,
    typename T,
    typename SizeT>
__global__ void ReduceSingleKernel(T *d_in, T *d_out, SizeT num_elements)
{
    if ((blockIdx.x == 0) && (threadIdx.x == 0)) printf("ReduceSingleKernel BLOCK_THREADS(%d) ITEMS_PER_THREAD(%d)\n",
        ReduceSingleKernelPolicy::BLOCK_THREADS,
        ReduceSingleKernelPolicy::ITEMS_PER_THREAD);
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

        template <typename ReduceKernelPolicy, typename ReduceSingleKernelPolicy>
        __host__ __device__ __forceinline__
        void Init()
        {
            reduce_block_threads       = ReduceKernelPolicy::BLOCK_THREADS;
            reduce_items_per_thread    = ReduceKernelPolicy::ITEMS_PER_THREAD;
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

        typedef ReduceKernelPolicy<     64,     1>      ReduceKernelPolicy300;
        typedef ReduceSingleKernelPolicy<   64,     1>      ReduceSingleKernelPolicy300;

        typedef ReduceKernelPolicy<     128,    1>      ReduceKernelPolicy200;
        typedef ReduceSingleKernelPolicy<   128,    1>      ReduceSingleKernelPolicy200;

        typedef ReduceKernelPolicy<     256,    1>      ReduceKernelPolicy100;
        typedef ReduceSingleKernelPolicy<   256,    1>      ReduceSingleKernelPolicy100;

    #if CUB_PTX_ARCH >= 300
        struct PtxReduceKernelPolicy :      ReduceKernelPolicy300 {};
        struct PtxReduceSingleKernelPolicy :    ReduceSingleKernelPolicy300 {};
    #elif CUB_PTX_ARCH >= 200
        struct PtxReduceKernelPolicy :      ReduceKernelPolicy200 {};
        struct PtxReduceSingleKernelPolicy :    ReduceSingleKernelPolicy200 {};
    #else
        struct PtxReduceKernelPolicy :      ReduceKernelPolicy100 {};
        struct PtxReduceSingleKernelPolicy :    ReduceSingleKernelPolicy100 {};
    #endif

        /**
         * Initialize dispatch policy
         */
        static void InitDispatchPolicy(int device_arch, DispatchPolicy &dispatch_policy)
        {
            if (device_arch >= 300)
                dispatch_policy.Init<ReduceKernelPolicy300, ReduceSingleKernelPolicy300>();
            else if (device_arch >= 200)
                dispatch_policy.Init<ReduceKernelPolicy200, ReduceSingleKernelPolicy200>();
            else
                dispatch_policy.Init<ReduceKernelPolicy100, ReduceSingleKernelPolicy100>();
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
        typename ReduceKernelPolicy,
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
        dispatch_policy.Init<ReduceKernelPolicy, ReduceSingleKernelPolicy>();

        return Foo(
            ReduceKernel<ReduceKernelPolicy, T, SizeT>,
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
            typedef typename DefaultPolicy<T, SizeT>::PtxReduceKernelPolicy PtxReduceKernelPolicy;
            typedef typename DefaultPolicy<T, SizeT>::PtxReduceSingleKernelPolicy PtxReduceSingleKernelPolicy;

            // Initialize dispatch policy
            DispatchPolicy dispatch_policy;

        #if CUB_PTX_ARCH > 0

            // We're on the device, so initialize the tuned dispatch policy based upon PTX arch
            dispatch_policy.Init<PtxReduceKernelPolicy, PtxReduceSingleKernelPolicy>();

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
                ReduceKernel<PtxReduceKernelPolicy, T, SizeT>,
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



