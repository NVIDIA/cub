

#include <stdio.h>

#define CUB_STDERR

#include <cub.cuh>
#include <test_util.h>

using namespace cub;


/******************************************************************************
 * FooUpsweepKernel
 *****************************************************************************/


// FooUpsweepKernel tuning policy
template <
    int _BLOCK_THREADS,
    int _ITEMS_PER_THREAD>
struct FooUpsweepKernelPolicy
{
    enum
    {
        BLOCK_THREADS      = _BLOCK_THREADS,
        ITEMS_PER_THREAD   = _ITEMS_PER_THREAD,
    };
};


// FooUpsweepKernel entry point
template <
    typename FooUpsweepKernelPolicy,
    typename T,
    typename SizeT>
__global__ void FooUpsweepKernel(T *d_in, T *d_out, SizeT num_elements)
{
    if ((blockIdx.x == 0) && (threadIdx.x == 0)) printf("FooUpsweepKernel BLOCK_THREADS(%d) ITEMS_PER_THREAD(%d)\n",
        FooUpsweepKernelPolicy::BLOCK_THREADS,
        FooUpsweepKernelPolicy::ITEMS_PER_THREAD);
}


/******************************************************************************
 * FooDownsweepKernel
 *****************************************************************************/


// FooDownsweepKernel tuning policy
template <
    int _BLOCK_THREADS,
    int _ITEMS_PER_THREAD>
struct FooDownsweepKernelPolicy
{
    enum
    {
        BLOCK_THREADS      = _BLOCK_THREADS,
        ITEMS_PER_THREAD   = _ITEMS_PER_THREAD,
    };
};


// FooDownsweepKernel entry point
template <
    typename FooDownsweepKernelPolicy,
    typename T,
    typename SizeT>
__global__ void FooDownsweepKernel(T *d_in, T *d_out, SizeT num_elements)
{
    if ((blockIdx.x == 0) && (threadIdx.x == 0)) printf("FooDownsweepKernel BLOCK_THREADS(%d) ITEMS_PER_THREAD(%d)\n",
        FooDownsweepKernelPolicy::BLOCK_THREADS,
        FooDownsweepKernelPolicy::ITEMS_PER_THREAD);
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
        int upsweep_block_threads;
        int upsweep_items_per_thread;
        int downsweep_block_threads;
        int downsweep_items_per_thread;

        template <typename FooUpsweepKernelPolicy, typename FooDownsweepKernelPolicy>
        __host__ __device__ __forceinline__
        void Init()
        {
            upsweep_block_threads       = FooUpsweepKernelPolicy::BLOCK_THREADS;
            upsweep_items_per_thread    = FooUpsweepKernelPolicy::ITEMS_PER_THREAD;
            downsweep_block_threads     = FooDownsweepKernelPolicy::BLOCK_THREADS;
            downsweep_items_per_thread  = FooDownsweepKernelPolicy::ITEMS_PER_THREAD;
        }
    };


    // Default tuning policy types
    template <
        typename T,
        typename SizeT>
    struct DefaultPolicy
    {

        typedef FooUpsweepKernelPolicy<     64,     1>      FooUpsweepKernelPolicy300;
        typedef FooDownsweepKernelPolicy<   64,     1>      FooDownsweepKernelPolicy300;

        typedef FooUpsweepKernelPolicy<     128,    1>      FooUpsweepKernelPolicy200;
        typedef FooDownsweepKernelPolicy<   128,    1>      FooDownsweepKernelPolicy200;

        typedef FooUpsweepKernelPolicy<     256,    1>      FooUpsweepKernelPolicy100;
        typedef FooDownsweepKernelPolicy<   256,    1>      FooDownsweepKernelPolicy100;

    #if CUB_PTX_ARCH >= 300
        struct PtxFooUpsweepKernelPolicy :      FooUpsweepKernelPolicy300 {};
        struct PtxFooDownsweepKernelPolicy :    FooDownsweepKernelPolicy300 {};
    #elif CUB_PTX_ARCH >= 200
        struct PtxFooUpsweepKernelPolicy :      FooUpsweepKernelPolicy200 {};
        struct PtxFooDownsweepKernelPolicy :    FooDownsweepKernelPolicy200 {};
    #else
        struct PtxFooUpsweepKernelPolicy :      FooUpsweepKernelPolicy100 {};
        struct PtxFooDownsweepKernelPolicy :    FooDownsweepKernelPolicy100 {};
    #endif

        /**
         * Initialize dispatch policy
         */
        static void InitDispatchPolicy(int device_arch, DispatchPolicy &dispatch_policy)
        {
            if (device_arch >= 300)
                dispatch_policy.Init<FooUpsweepKernelPolicy300, FooDownsweepKernelPolicy300>();
            else if (device_arch >= 200)
                dispatch_policy.Init<FooUpsweepKernelPolicy200, FooDownsweepKernelPolicy200>();
            else
                dispatch_policy.Init<FooUpsweepKernelPolicy100, FooDownsweepKernelPolicy100>();
        }
    };


    // Internal Foo dispatch
    template <
        typename FooUpsweepKernelPtr,
        typename FooDownsweepKernelPtr,
        typename T,
        typename SizeT>
    __host__ __device__ __forceinline__
    static cudaError_t Foo(
        FooUpsweepKernelPtr     upsweep_kernel_ptr,
        FooDownsweepKernelPtr   downsweep_kernel_ptr,
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
            int upsweep_sm_occupancy        = ArchProps<CUB_PTX_ARCH>::MAX_SM_THREADBLOCKS;
            int downsweep_sm_occupancy      = ArchProps<CUB_PTX_ARCH>::MAX_SM_THREADBLOCKS;
            int oversubscription            = ArchProps<CUB_PTX_ARCH>::OVERSUBSCRIPTION;

        #if (CUB_PTX_ARCH == 0)

            // We're on the host, so come up with a more accurate estimate of SM occupancies from actual device properties
            Device device_props;
            if (CubDebug(error = device_props.Init(device_ordinal))) break;
            oversubscription = device_props.oversubscription;

            if (CubDebug(error = device_props.MaxSmOccupancy(
                upsweep_sm_occupancy,
                upsweep_kernel_ptr,
                dispatch_policy.upsweep_block_threads))) break;

            if (CubDebug(error = device_props.MaxSmOccupancy(
                downsweep_sm_occupancy,
                downsweep_kernel_ptr,
                dispatch_policy.downsweep_block_threads))) break;
        #endif

            // Construct work distributions
            GridEvenShare<SizeT> upsweep_distrib(
                num_elements,
                upsweep_sm_occupancy * sm_count * oversubscription,
                dispatch_policy.upsweep_block_threads * dispatch_policy.upsweep_items_per_thread);

            GridEvenShare<SizeT> downsweep_distrib(
                num_elements,
                downsweep_sm_occupancy * sm_count * oversubscription,
                dispatch_policy.downsweep_block_threads * dispatch_policy.downsweep_items_per_thread);

            printf("Invoking FooUpsweep<<<%d, %d>>>()\n",
                upsweep_distrib.grid_size,
                dispatch_policy.upsweep_block_threads);

            // Invoke FooUpsweep
            upsweep_kernel_ptr<<<upsweep_distrib.grid_size, dispatch_policy.upsweep_block_threads>>>(
                d_in, d_out, num_elements);

            printf("Invoking FooDownsweep<<<%d, %d>>>()\n",
                downsweep_distrib.grid_size,
                dispatch_policy.downsweep_block_threads);

            // Invoke FooDownsweep
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
        typename FooUpsweepKernelPolicy,
        typename FooDownsweepKernelPolicy,
        typename T,
        typename SizeT>
    __host__ __device__ __forceinline__
    static cudaError_t Foo(
        T       *d_in,
        T       *d_out,
        SizeT   num_elements)
    {
        DispatchPolicy dispatch_policy;
        dispatch_policy.Init<FooUpsweepKernelPolicy, FooDownsweepKernelPolicy>();

        return Foo(
            FooUpsweepKernel<FooUpsweepKernelPolicy, T, SizeT>,
            FooDownsweepKernel<FooDownsweepKernelPolicy, T, SizeT>,
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
            typedef typename DefaultPolicy<T, SizeT>::PtxFooUpsweepKernelPolicy PtxFooUpsweepKernelPolicy;
            typedef typename DefaultPolicy<T, SizeT>::PtxFooDownsweepKernelPolicy PtxFooDownsweepKernelPolicy;

            // Initialize dispatch policy
            DispatchPolicy dispatch_policy;

        #if CUB_PTX_ARCH > 0

            // We're on the device, so initialize the tuned dispatch policy based upon PTX arch
            dispatch_policy.Init<PtxFooUpsweepKernelPolicy, PtxFooDownsweepKernelPolicy>();

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
                FooUpsweepKernel<PtxFooUpsweepKernelPolicy, T, SizeT>,
                FooDownsweepKernel<PtxFooDownsweepKernelPolicy, T, SizeT>,
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



