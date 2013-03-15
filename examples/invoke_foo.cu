/**
 * nvcc invoke_foo.cu -gencode=arch=compute_35,code=\"sm_35,compute_35\" -gencode=arch=compute_20,code=\"sm_20,compute_20\" -Xptxas -v
 * nvcc invoke_foo.cu -gencode=arch=compute_35,code=\"sm_35,compute_35\" -Xptxas -v -rdc=true -lcudadevrt
 */



#include <stdio.h>

//---------------------------------------------------------------------
// Macro definitions
//---------------------------------------------------------------------

#ifdef __CUDA_ARCH__
#define PTX_ARCH __CUDA_ARCH__
#else
#define PTX_ARCH 0
#endif
#define CNP_ENABLED ((PTX_ARCH == 0) || ((PTX_ARCH >= 350) && defined(__BUILDING_CNPRT__)))


//---------------------------------------------------------------------
// Foo-related kernels and policies
//---------------------------------------------------------------------

/**
 * FooKernel tuning policy
 */
template <int _BLOCK_THREADS, int _ITEMS_PER_THREAD>
struct FooKernelPolicy
{
    enum
    {
        BLOCK_THREADS      = _BLOCK_THREADS,
        ITEMS_PER_THREAD   = _ITEMS_PER_THREAD,
    };
};


/**
 * FooKernel kernel entrypoint
 */
template <typename FooKernelPolicy, typename T>
__launch_bounds__ (FooKernelPolicy::BLOCK_THREADS, 1)
__global__ void FooKernel(T *d_in, T *d_out, int num_elements)
{
    if ((blockIdx.x == 0) && (threadIdx.x == 0)) printf("FooKernel<<<%d, %d>>>, ITEMS_PER_THREAD(%d)\n",
        gridDim.x,
        FooKernelPolicy::BLOCK_THREADS,
        FooKernelPolicy::ITEMS_PER_THREAD);
}


//---------------------------------------------------------------------
// Foo wrapper
//---------------------------------------------------------------------

/**
 * Wrapper for all foo-related entrypoints
 */
struct Foo
{

    /**
     * Invoke foo operation with custom policy
     */
    template <typename FooKernelPolicyT, typename T>
    __host__ __device__ __forceinline__
    static cudaError_t Invoke(
        T *d_in,
        T *d_out,
        int num_elements,
        void (*foo_kernel_ptr)(T*, T*, int) = FooKernel<FooKernelPolicyT, T>)
    {
        // Preconfigured tuning policies
        typedef FooKernelPolicy<64,     1>      FooKernelPolicy300;
        typedef FooKernelPolicy<128,    1>      FooKernelPolicy200;
        typedef FooKernelPolicy<256,    1>      FooKernelPolicy100;

        // PTX-specific default policy
    #if PTX_ARCH >= 300
        struct PtxFooKernelPolicy : FooKernelPolicy300 {};
    #elif PTX_ARCH >= 200
        struct PtxFooKernelPolicy : FooKernelPolicy200 {};
    #else
        struct PtxFooKernelPolicy : FooKernelPolicy100 {};
    #endif


        if (foo_kernel_ptr == NULL) foo_kernel_ptr = FooKernel<PtxFooKernelPolicy, T>;

    #if !CNP_ENABLED
        // CUDA API calls and kernel launch not supported from this device
        return cudaErrorInvalidConfiguration;
    #else
        // Determine grid size
        const int TILE_SIZE = FooKernelPolicyT::BLOCK_THREADS * FooKernelPolicyT::ITEMS_PER_THREAD;
        int grid_size = (num_elements + TILE_SIZE - 1) / TILE_SIZE;

        // Invoke kernel
        foo_kernel_ptr<<<grid_size, FooKernelPolicyT::BLOCK_THREADS>>>(d_in, d_out, num_elements);

        return cudaSuccess;
    #endif
    }


    /**
     * Invoke foo operation with default policy
     */
    template <typename T>
    __host__ __device__ __forceinline__
    static cudaError_t Invoke(T *d_in, T *d_out, int num_elements)
    {
        // Preconfigured tuning policies
        typedef FooKernelPolicy<64,     1>      FooKernelPolicy300;
        typedef FooKernelPolicy<128,    1>      FooKernelPolicy200;
        typedef FooKernelPolicy<256,    1>      FooKernelPolicy100;

        // PTX-specific default policy
    #if PTX_ARCH >= 300
        struct PtxFooKernelPolicy : FooKernelPolicy300 {};
    #elif PTX_ARCH >= 200
        struct PtxFooKernelPolicy : FooKernelPolicy200 {};
    #else
        struct PtxFooKernelPolicy : FooKernelPolicy100 {};
    #endif

    #if !CNP_ENABLED

        // CUDA API calls and kernel launch not supported from this device
        return cudaErrorInvalidConfiguration;

    #else

        // We're on the host, so determine which tuned variant to initialize
        int device_ordinal;
        cudaGetDevice(&device_ordinal);

        int major, minor;
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device_ordinal);
        cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device_ordinal);
        int device_arch = major * 100 + minor * 10;

        // Our PTX-specific foo kernel function pointer
        void (*foo_kernel_ptr)(T*, T*, int) = FooKernel<PtxFooKernelPolicy, T>;

        // Dispatch with explicit policy
        if (device_arch >= 300)
            return Invoke<FooKernelPolicy300>(d_in, d_out, num_elements, foo_kernel_ptr);
        else if (device_arch >= 200)
            return Invoke<FooKernelPolicy200>(d_in, d_out, num_elements, foo_kernel_ptr);
        else
            return Invoke<FooKernelPolicy100>(d_in, d_out, num_elements, foo_kernel_ptr);

    #endif
    }
};



//---------------------------------------------------------------------
// User kernel for dispatching Foo from device
//---------------------------------------------------------------------

/**
 * User kernel for nested invocation of foo
 */
template <typename T>
__global__ void UserKernel(T *d_in, T *d_out, int num_elements)
{
    // Invoke Foo
    Foo::Invoke(d_in, d_out, num_elements);
}


//---------------------------------------------------------------------
// Main
//---------------------------------------------------------------------

/**
 * Main
 */
int main(int argc, char **argv)
{
    typedef int T;

    T *d_in = NULL;
    T *d_out = NULL;
    int num_elements = 1024 * 1024;

    int dev = 0;
    if (argc > 1)
    {
        dev = atoi(argv[1]);
    }
    cudaSetDevice(dev);
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, dev);
    printf("Using device %s\n\n", props.name);
    fflush(stdout);

    // Test1: Dispatch Foo from host
    Foo::Invoke(d_in, d_out, num_elements);

    // Test2: Dispatch Foo with custom policy
    Foo::Invoke<FooKernelPolicy<96, 17> >(d_in, d_out, num_elements);

    // Test3: Dispatch user kernel that dispatches Foo from device
    UserKernel<<<1,1>>>(d_in, d_out, num_elements);

    cudaDeviceSynchronize();

    return 0;
}
