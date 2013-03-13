

#include <stdio.h>

#include "cub.cuh"

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
    if (threadIdx.x == 0) printf("FooKernel BLOCK_THREADS(%d) ITEMS_PER_THREAD(%d)\n",
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
    if (threadIdx.x == 0) printf("BarKernel BLOCK_THREADS(%d) ITEMS_PER_THREAD(%d)\n",
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
    template <typename T>
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
            __host__ __device__ __forceinline__ cudaError_t
            Init()
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
            FooKernelContext foo_kernel_context,
            BarKernelContext bar_kernel_context)
        {
            this->foo_kernel_context = foo_kernel_context;
            this->bar_kernel_context = bar_kernel_context;
            return cudaSuccess;
        }


        // Initializer (for default tuning)
        __host__ __device__ __forceinline__
        cudaError_t Init()
        {
            cudaError retval = cudaSuccess;
            do
            {

            #if CUB_PTX_ARCH > 0

                // We're on the device, so we know we can simply use the policy compiled in the current PTX bundle
                if ((retval = foo_kernel_context.template Init<PtxFooKernelPolicy, PtxFooKernelPolicy>())) break;
                if ((retval = bar_kernel_context.template Init<PtxBarKernelPolicy, PtxBarKernelPolicy>())) break;

            #else
                // We're on the host, so determine which tuned variant to initialize
                int device_ordinal;
                if (CubDebug(retval = cudaGetDevice(&device_ordinal))) break;

                int major, minor;
                if (CubDebug(retval = cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device_ordinal))) break;
                if (CubDebug(retval = cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device_ordinal))) break;
                int device_arch = major * 100 + minor * 10;

                // Initialize kernel contexts
                if (device_arch >= 300)
                {
                    if ((retval = foo_kernel_context.template Init<FooKernelPolicy300, PtxFooKernelPolicy>())) break;
                    if ((retval = bar_kernel_context.template Init<BarKernelPolicy300, PtxBarKernelPolicy>())) break;
                }
                else if (device_arch >= 200)
                {
                    if ((retval = foo_kernel_context.template Init<FooKernelPolicy200, PtxFooKernelPolicy>())) break;
                    if ((retval = bar_kernel_context.template Init<BarKernelPolicy200, PtxBarKernelPolicy>())) break;
                }
                else
                {
                    if ((retval = foo_kernel_context.template Init<FooKernelPolicy100, PtxFooKernelPolicy>())) break;
                    if ((retval = bar_kernel_context.template Init<BarKernelPolicy100, PtxBarKernelPolicy>())) break;
                }
            #endif
            }
            while (0);

            return retval;
        }


        // Invoke operation (instance must be initialized first)
        __host__ __device__ __forceinline__
        cudaError_t Baz(T *d_in, T *d_out)
        {

        #if CNP_ENABLED

            cudaError retval = cudaSuccess;
            do
            {
                int grid_size = 1;
                foo_kernel_context.kernel_ptr<<<grid_size, foo_kernel_context.block_threads>>>(d_in, d_out);
                bar_kernel_context.kernel_ptr<<<grid_size, bar_kernel_context.block_threads>>>(d_in, d_out);
            }
            while (0);
            return retval;

        #else

            // Kernel launch not supported from this device
            return cudaErrorInvalidConfiguration;

        #endif
        }

    };

    // Baz
    template <typename T>
    __host__ __device__ __forceinline__
    static cudaError_t Baz(T *d_in, T *d_out)
    {
        cudaError_t retval = cudaSuccess;
        do
        {
            // Create configuration instance
            Context<T> config_instance;
            if ((retval = config_instance.Init())) break;

            // Run
            if ((retval = config_instance.Baz(d_in, d_out))) break;
        }
        while (0);

        return retval;
    }

};


/******************************************************************************
 * Main
 *****************************************************************************/

template <typename T>
__global__ void WrapperKernel(T *d_in, T *d_out)
{
    if (threadIdx.x == 0) printf("WrapperKernel\n");

#if CNP_ENABLED
    if ((blockIdx.x == 0) && (threadIdx.x == 0))
    {
        // Cuda nested kernel invocation
        DeviceBaz::Baz(d_in, d_out);
    }

    int device_ordinal;
    CubDebug(cudaGetDevice(&device_ordinal));

    int sm_count;
    CubDebug(cudaDeviceGetAttribute (&sm_count, cudaDevAttrMultiProcessorCount, device_ordinal));
    printf("Sm count: %d\n", sm_count);

#endif
}


/// Main
int main()
{
    typedef int T;

    T *d_in = NULL;
    T *d_out = NULL;

    DeviceBaz::Baz(d_in, d_out);

    WrapperKernel<<<1,1>>>(d_in, d_out);

    CubDebug(cudaDeviceSynchronize());

    return 0;
}



