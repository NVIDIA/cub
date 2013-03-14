

#include <stdio.h>

#define CUB_STDERR

#include <cub.cuh>
#include <test_util.h>

using namespace cub;


/******************************************************************************
 * FooKernel
 *****************************************************************************/


// FooKernel tuning policy
template <
    int _BLOCK_THREADS,
    int _ITEMS_PER_THREAD>
struct FooKernelPolicy
{
    enum
    {
        BLOCK_THREADS      = _BLOCK_THREADS,
        ITEMS_PER_THREAD   = _ITEMS_PER_THREAD,
    };
};


// FooKernel entrypoint
template <
    typename FooKernelPolicy,
    typename T>
__global__ void FooKernel(T *d_in, T *d_out)
{
    if ((blockIdx.x == 0) && (threadIdx.x == 0)) printf("FooKernel BLOCK_THREADS(%d) ITEMS_PER_THREAD(%d)\n",
        FooKernelPolicy::BLOCK_THREADS,
        FooKernelPolicy::ITEMS_PER_THREAD);
}


/******************************************************************************
 * BarKernel
 *****************************************************************************/


// BarKernel tuning policy
template <
    int _BLOCK_THREADS,
    int _ITEMS_PER_THREAD>
struct BarKernelPolicy
{
    enum
    {
        BLOCK_THREADS      = _BLOCK_THREADS,
        ITEMS_PER_THREAD   = _ITEMS_PER_THREAD,
    };
};


// BarKernel entrypoint
template <
    typename BarKernelPolicy,
    typename T>
__global__ void BarKernel(T *d_in, T *d_out)
{
    if ((blockIdx.x == 0) && (threadIdx.x == 0)) printf("BarKernel BLOCK_THREADS(%d) ITEMS_PER_THREAD(%d)\n",
        BarKernelPolicy::BLOCK_THREADS,
        BarKernelPolicy::ITEMS_PER_THREAD);
}



/******************************************************************************
 * DeviceBaz
 *****************************************************************************/



/**
 * Provides BAZ operations on device-global data sets.
 */
struct DeviceBaz
{
    // Configuration context.  These can be configured and then executed later.
    // For example, an autotuning framework may create a large list of
    // different contexts.
    template <
        typename T,
        typename SizeT>
    struct Context
    {
        //---------------------------------------------------------------------
        // Default tuning types
        //---------------------------------------------------------------------

        typedef FooKernelPolicy<64,     1>      FooKernelPolicy300;
        typedef BarKernelPolicy<64,     1>      BarKernelPolicy300;

        typedef FooKernelPolicy<128,    1>      FooKernelPolicy200;
        typedef BarKernelPolicy<128,    1>      BarKernelPolicy200;

        typedef FooKernelPolicy<256,    1>      FooKernelPolicy100;
        typedef BarKernelPolicy<256,    1>      BarKernelPolicy100;

    #if CUB_PTX_ARCH >= 300
        struct PtxFooKernelPolicy : FooKernelPolicy300 {};
        struct PtxBarKernelPolicy : BarKernelPolicy300 {};
    #elif CUB_PTX_ARCH >= 200
        struct PtxFooKernelPolicy : FooKernelPolicy200 {};
        struct PtxBarKernelPolicy : BarKernelPolicy200 {};
    #else
        struct PtxFooKernelPolicy : FooKernelPolicy100 {};
        struct PtxBarKernelPolicy : BarKernelPolicy100 {};
    #endif

        //---------------------------------------------------------------------
        // Kernel context types
        //---------------------------------------------------------------------

        // Foo kernel context
        struct FooKernelContext
        {
            void (*kernel_ptr)(T *d_in, T *d_out);
            int tile_items;
            int block_threads;

            // Initializer
            template <typename KernelPolicy, typename OpaqueKernelPolicy>
            __host__ __device__ __forceinline__
            cudaError_t Init()
            {
                kernel_ptr      = FooKernel<OpaqueKernelPolicy>;
                block_threads   = KernelPolicy::BLOCK_THREADS;
                tile_items      = KernelPolicy::BLOCK_THREADS * KernelPolicy::ITEMS_PER_THREAD;

                return cudaSuccess;
            }
        };

        // Bar kernel context
        struct BarKernelContext
        {
            void (*kernel_ptr)(T *d_in, T *d_out);
            int tile_items;
            int block_threads;

            // Initializer
            template <typename KernelPolicy, typename OpaqueKernelPolicy>
            __host__ __device__ __forceinline__
            cudaError_t Init()
            {
                kernel_ptr      = BarKernel<OpaqueKernelPolicy>;
                block_threads   = KernelPolicy::BLOCK_THREADS;
                tile_items      = KernelPolicy::BLOCK_THREADS * KernelPolicy::ITEMS_PER_THREAD;

                return cudaSuccess;
            }
        };


        //---------------------------------------------------------------------
        // Fields
        //---------------------------------------------------------------------

        // Foo kernel context
        FooKernelContext foo_kernel_context;

        // Bar kernel context
        BarKernelContext bar_kernel_context;


        //---------------------------------------------------------------------
        // Public interface
        //---------------------------------------------------------------------

        // Initializer (for custom tuning that may generate foo and
        // bar kernel contexts separately)
        template <typename FooKernelPolicy, typename BarKernelPolicy>
        __host__ __device__ __forceinline__
        cudaError_t Init(
            const FooKernelContext &foo_kernel_context,
            const BarKernelContext &bar_kernel_context)
        {
            this->foo_kernel_context = foo_kernel_context;
            this->bar_kernel_context = bar_kernel_context;
            return cudaSuccess;
        }


        // Initializer (for default tuning)
        __host__ __device__ __forceinline__
        cudaError_t Init()
        {
            cudaError error = cudaSuccess;
            do
            {

            #if CUB_PTX_ARCH > 0

                // We're on the device, so we know we can simply use the policy compiled in the current PTX bundle
                if ((error = foo_kernel_context.template Init<PtxFooKernelPolicy, PtxFooKernelPolicy>())) break;
                if ((error = bar_kernel_context.template Init<PtxBarKernelPolicy, PtxBarKernelPolicy>())) break;

            #else
                // We're on the host, so determine which tuned variant to initialize
                int device_ordinal;
                if (CubDebug(error = cudaGetDevice(&device_ordinal))) break;

                int major, minor;
                if (CubDebug(error = cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device_ordinal))) break;
                if (CubDebug(error = cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device_ordinal))) break;
                int device_arch = major * 100 + minor * 10;

                // Initialize kernel contexts
                if (device_arch >= 300)
                {
                    if ((error = foo_kernel_context.template Init<FooKernelPolicy300, PtxFooKernelPolicy>())) break;
                    if ((error = bar_kernel_context.template Init<BarKernelPolicy300, PtxBarKernelPolicy>())) break;
                }
                else if (device_arch >= 200)
                {
                    if ((error = foo_kernel_context.template Init<FooKernelPolicy200, PtxFooKernelPolicy>())) break;
                    if ((error = bar_kernel_context.template Init<BarKernelPolicy200, PtxBarKernelPolicy>())) break;
                }
                else
                {
                    if ((error = foo_kernel_context.template Init<FooKernelPolicy100, PtxFooKernelPolicy>())) break;
                    if ((error = bar_kernel_context.template Init<BarKernelPolicy100, PtxBarKernelPolicy>())) break;
                }
            #endif
            }
            while (0);

            return error;
        }


        // Invoke operation (instance must be initialized first)
        __host__ __device__ __forceinline__
        cudaError_t Baz(T *d_in, T *d_out, SizeT num_elements)
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

                // Rough estimate of maximum SM occupancies
                int foo_sm_occupancy   = ArchProps<CUB_PTX_ARCH>::MAX_SM_THREADBLOCKS;
                int bar_sm_occupancy   = ArchProps<CUB_PTX_ARCH>::MAX_SM_THREADBLOCKS;
                int oversubscription   = ArchProps<CUB_PTX_ARCH>::OVERSUBSCRIPTION;

            #if (CUB_PTX_ARCH == 0)

                // On the host, come up with a more accurate estimate of SM occupancies
                DeviceProps device_props;
                if (CubDebug(error = device_props.Init(device_ordinal))) break;
                oversubscription = device_props.oversubscription;

                if (CubDebug(error = device_props.MaxSmOccupancy(
                    foo_sm_occupancy,
                    foo_kernel_context.kernel_ptr,
                    foo_kernel_context.block_threads))) break;

                if (CubDebug(error = device_props.MaxSmOccupancy(
                    bar_sm_occupancy,
                    bar_kernel_context.kernel_ptr,
                    bar_kernel_context.block_threads))) break;

            #endif

                // Construct work distributions
                GridEvenShare<SizeT> foo_even_share(
                    num_elements,
                    foo_sm_occupancy * sm_count * oversubscription,
                    foo_kernel_context.tile_items);

                GridEvenShare<SizeT> bar_even_share(
                    num_elements,
                    bar_sm_occupancy * sm_count * oversubscription,
                    bar_kernel_context.tile_items);

                printf("Invoking Foo<<<%d, %d>>>()\n", foo_even_share.grid_size, foo_kernel_context.block_threads);

                // Invoke Foo
                foo_kernel_context.kernel_ptr<<<foo_even_share.grid_size, foo_kernel_context.block_threads>>>(
                    d_in,
                    d_out);

                printf("Invoking Bar<<<%d, %d>>>()\n", bar_even_share.grid_size, bar_kernel_context.block_threads);

                // Invoke Bar
                bar_kernel_context.kernel_ptr<<<bar_even_share.grid_size, bar_kernel_context.block_threads>>>(
                    d_in,
                    d_out);
            }
            while (0);
            return error;

        #endif
        }

    };


    /**
     * Baz
     */
    template <typename T, typename SizeT>
    __host__ __device__ __forceinline__
    static cudaError_t Baz(T *d_in, T *d_out, SizeT num_elements)
    {
        cudaError_t error = cudaSuccess;
        do
        {
            // Create configuration instance
            Context<T, SizeT> config_instance;
            if ((error = config_instance.Init())) break;

            // Run
            if ((error = config_instance.Baz(d_in, d_out, num_elements))) break;
        }
        while (0);

        return error;
    }

};


/******************************************************************************
 * User kernel
 *****************************************************************************/

template <typename T, typename SizeT>
__global__ void UserKernel(T *d_in, T *d_out, SizeT num_elements)
{
    // Cuda nested kernel invocation
    DeviceBaz::Baz(d_in, d_out, num_elements);
}


/******************************************************************************
 * Main
 *****************************************************************************/


/// Main
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

    // Invoke Baz from host
    DeviceBaz::Baz(d_in, d_out, num_elements);

    // Invoke Baz from device
    UserKernel<<<1,1>>>(d_in, d_out, num_elements);

    CubDebug(cudaDeviceSynchronize());

    return 0;
}



