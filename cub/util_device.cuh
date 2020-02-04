/******************************************************************************
 * Copyright (c) 2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2020, NVIDIA CORPORATION.  All rights reserved.
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
 * Properties of a given CUDA device and the corresponding PTX bundle
 */

#pragma once

#include "util_type.cuh"
#include "util_arch.cuh"
#include "util_debug.cuh"
#include "util_namespace.cuh"
#include "util_macro.cuh"

/// Optional outer namespace(s)
CUB_NS_PREFIX

/// CUB namespace
namespace cub {


/**
 * \addtogroup UtilMgmt
 * @{
 */

#ifndef DOXYGEN_SHOULD_SKIP_THIS // Do not document


/**
 * \brief Alias temporaries to externally-allocated device storage (or simply return the amount of storage needed).
 */
template <int ALLOCATIONS>
__host__ __device__ __forceinline__
cudaError_t AliasTemporaries(
    void    *d_temp_storage,                    ///< [in] %Device-accessible allocation of temporary storage.  When NULL, the required allocation size is written to \p temp_storage_bytes and no work is done.
    size_t  &temp_storage_bytes,                ///< [in,out] Size in bytes of \t d_temp_storage allocation
    void*   (&allocations)[ALLOCATIONS],        ///< [in,out] Pointers to device allocations needed
    size_t  (&allocation_sizes)[ALLOCATIONS])   ///< [in] Sizes in bytes of device allocations needed
{
    const int ALIGN_BYTES   = 256;
    const int ALIGN_MASK    = ~(ALIGN_BYTES - 1);

    // Compute exclusive prefix sum over allocation requests
    size_t allocation_offsets[ALLOCATIONS];
    size_t bytes_needed = 0;
    for (int i = 0; i < ALLOCATIONS; ++i)
    {
        size_t allocation_bytes = (allocation_sizes[i] + ALIGN_BYTES - 1) & ALIGN_MASK;
        allocation_offsets[i] = bytes_needed;
        bytes_needed += allocation_bytes;
    }
    bytes_needed += ALIGN_BYTES - 1;

    // Check if the caller is simply requesting the size of the storage allocation
    if (!d_temp_storage)
    {
        temp_storage_bytes = bytes_needed;
        return cudaSuccess;
    }

    // Check if enough storage provided
    if (temp_storage_bytes < bytes_needed)
    {
        return CubDebug(cudaErrorInvalidValue);
    }

    // Alias
    d_temp_storage = (void *) ((size_t(d_temp_storage) + ALIGN_BYTES - 1) & ALIGN_MASK);
    for (int i = 0; i < ALLOCATIONS; ++i)
    {
        allocations[i] = static_cast<char*>(d_temp_storage) + allocation_offsets[i];
    }

    return cudaSuccess;
}


/**
 * \brief Empty kernel for querying PTX manifest metadata (e.g., version) for the current device
 */
template <typename T>
__global__ void EmptyKernel(void) { }

#endif  // DOXYGEN_SHOULD_SKIP_THIS

/**
 * \brief Returns the current device or -1 if an error occurred.
 */
CUB_RUNTIME_FUNCTION __forceinline__ int CurrentDevice()
{
#if defined(CUB_RUNTIME_ENABLED) // Host code or device code with the CUDA runtime.

    int device = -1;
    if (CubDebug(cudaGetDevice(&device))) return -1;
    return device;

#else // Device code without the CUDA runtime.

    return -1;

#endif
}

/**
 * \brief RAII helper which saves the current device and switches to the
 *        specified device on construction and switches to the saved device on
 *        destruction.
 */
struct SwitchDevice
{
private:
    int const old_device;

public:
    __host__ __forceinline__ SwitchDevice(int new_device)
      : old_device(CurrentDevice())
    {
        CubDebug(cudaSetDevice(new_device));
    }

    __host__ __forceinline__ ~SwitchDevice()
    {
        CubDebug(cudaSetDevice(old_device));
    }
};

/**
 * \brief Per-device cache for a CUDA attribute value; the attribute is queried
 *        and stored for each device upon construction.
 */
struct PerDeviceAttributeCache
{
    int attribute[CUB_MAX_DEVICES];
    cudaError_t error[CUB_MAX_DEVICES];

    template <typename UncachedFunction>
    __host__ __forceinline__ PerDeviceAttributeCache(UncachedFunction uncached_function)
    {
        int num_devices = 0;

        cudaError_t non_existent_error = cudaErrorUnknown;

        if (!CubDebug(non_existent_error = cudaGetDeviceCount(&num_devices)))
            non_existent_error = cudaErrorInvalidDevice;

        int device = 0;

        for (; device < num_devices; ++device)
        {
            // If this fails, we haven't compiled device code that can run on
            // this device. This is only an error if we actually use this device,
            // so we don't use CubDebug here.
            if (error[device] = uncached_function(attribute[device], device)) {
              // Clear the global CUDA error state which may have been set by
              // the last call. Otherwise, errors may "leak" to unrelated
              // kernel launches.
              cudaGetLastError();
              break;
            }
        }

        // Make sure the entries for non-existent devices are initialized.
        for (; device < CUB_MAX_DEVICES; ++device)
        {
            attribute[device] = -1;
            error[device] = non_existent_error;
        }
    }
};

/**
 * \brief Retrieves the PTX version that will be used on the current device (major * 100 + minor * 10).
 */
CUB_RUNTIME_FUNCTION __forceinline__ cudaError_t PtxVersionUncached(int &ptx_version)
{
    // Instantiate `EmptyKernel<void>` in both host and device code to ensure
    // it can be called.
    typedef void (*EmptyKernelPtr)();
    EmptyKernelPtr empty_kernel = EmptyKernel<void>;

    (void)(empty_kernel);

#if (CUB_PTX_ARCH == 0) // Host code.

    cudaError_t error = cudaSuccess;
    cudaFuncAttributes empty_kernel_attrs;

    do {
        // We do not `CubDebug` here because failure is not a hard error.
        // We may be querying a device that we do not have code for but
        // never use.
        if (error = cudaFuncGetAttributes(&empty_kernel_attrs, empty_kernel)) {
            // Clear the global CUDA error state which may have been set by
            // the last call. Otherwise, errors may "leak" to unrelated
            // kernel launches.
            cudaGetLastError();
            break;
        }
    }
    while(0);

    ptx_version = empty_kernel_attrs.ptxVersion * 10;

    return error;

#else // Device code.

    // The `reinterpret_cast` is necessary to suppress a set-but-unused warnings.
    // This is a meme now: https://twitter.com/blelbach/status/1222391615576100864
    (void)reinterpret_cast<EmptyKernelPtr>(empty_kernel);

    ptx_version = CUB_PTX_ARCH;

    return cudaSuccess;

#endif
}

/**
 * \brief Retrieves the PTX version that will be used on \p device (major * 100 + minor * 10).
 */
__host__ __forceinline__ cudaError_t PtxVersionUncached(int &ptx_version, int device)
{
    SwitchDevice sd(device);
    // Neither the single argument version of `PtxVersionUncached` or this
    // overload passes the `cudaError_t` through `CubDebug` because failure
    // is not a hard error here. We may be querying a device that we do not
    // have code for but never use.
    return PtxVersionUncached(ptx_version);
}

/**
 * \brief Retrieves the PTX version that will be used on \p device (major * 100 + minor * 10).
 *
 * \note This function may cache the result internally.
 */
__host__ __forceinline__ cudaError_t PtxVersion(int &ptx_version, int device)
{
#if __cplusplus >= 201103L // C++11 and later.

    using FunctionPointer = cudaError_t(*)(int &, int);
    FunctionPointer fun_ptr = PtxVersionUncached;

    // C++11 guarantees that initialization of static locals is thread safe.
    static const PerDeviceAttributeCache cache(fun_ptr);

    if (!CubDebug(cache.error[device]))
        ptx_version = cache.attribute[device];

    return cache.error[device];

#else // Pre C++11.

    return CubDebug(PtxVersionUncached(ptx_version, device));

#endif
}

/**
 * \brief Retrieves the PTX version that will be used on the current device (major * 100 + minor * 10).
 *
 * \note This function may cache the result internally.
 */
CUB_RUNTIME_FUNCTION __forceinline__ cudaError_t PtxVersion(int &ptx_version)
{
#if __cplusplus >= 201103L && (CUB_PTX_ARCH == 0) // Host code and C++11.

    return PtxVersion(ptx_version, CurrentDevice());

#else // Device code or host code before C++11.

    // Avoid an unnecessary set/reset of the CUDA current device.
    return CubDebug(PtxVersionUncached(ptx_version));

#endif
}

/**
 * \brief Retrieves the SM version of \p device (major * 100 + minor * 10)
 */
CUB_RUNTIME_FUNCTION __forceinline__ cudaError_t SmVersionUncached(int &sm_version, int device = CurrentDevice())
{
#if defined(CUB_RUNTIME_ENABLED) // Host code or device code with the CUDA runtime.

    cudaError_t error = cudaSuccess;
    do
    {
        int major = 0, minor = 0;
        if (CubDebug(error = cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device))) break;
        if (CubDebug(error = cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device))) break;
        sm_version = major * 100 + minor * 10;
    }
    while (0);

    return error;

#else // Device code without the CUDA runtime.

    (void)sm_version;
    (void)device;

    // CUDA API calls are not supported from this device.
    return CubDebug(cudaErrorInvalidConfiguration);

#endif
}

/**
 * \brief Retrieves the SM version of \p device (major * 100 + minor * 10)
 *
 * \note This function may cache the result internally.
 */
CUB_RUNTIME_FUNCTION __forceinline__ cudaError_t SmVersion(int &sm_version, int device = CurrentDevice())
{
#if __cplusplus >= 201103L && (CUB_PTX_ARCH == 0) // Host code and C++11.

    using FunctionPointer = cudaError_t(*)(int &, int);
    FunctionPointer fun_ptr = SmVersionUncached;

    // C++11 guarantees that initialization of static locals is thread safe.
    static const PerDeviceAttributeCache cache(fun_ptr);

    if (!CubDebug(cache.error[device]))
        sm_version = cache.attribute[device];

    return cache.error[device];

#else // Device code or host code before C++11.

    return SmVersionUncached(sm_version, device);

#endif
}

/**
 * Synchronize the specified \p stream.
 */
CUB_RUNTIME_FUNCTION __forceinline__ cudaError_t SyncStream(cudaStream_t stream)
{
#if (CUB_PTX_ARCH == 0) // Host code.

    return CubDebug(cudaStreamSynchronize(stream));

#elif defined(CUB_RUNTIME_ENABLED) // Device code with the CUDA runtime.

    (void)stream;
    // Device can't yet sync on a specific stream
    return CubDebug(cudaDeviceSynchronize());

#else // Device code without the CUDA runtime.

    (void)stream;
    // CUDA API calls are not supported from this device.
    return CubDebug(cudaErrorInvalidConfiguration);

#endif
}


/**
 * \brief Computes maximum SM occupancy in thread blocks for executing the given kernel function pointer \p kernel_ptr on the current device with \p block_threads per thread block.
 *
 * \par Snippet
 * The code snippet below illustrates the use of the MaxSmOccupancy function.
 * \par
 * \code
 * #include <cub/cub.cuh>   // or equivalently <cub/util_device.cuh>
 *
 * template <typename T>
 * __global__ void ExampleKernel()
 * {
 *     // Allocate shared memory for BlockScan
 *     __shared__ volatile T buffer[4096];
 *
 *        ...
 * }
 *
 *     ...
 *
 * // Determine SM occupancy for ExampleKernel specialized for unsigned char
 * int max_sm_occupancy;
 * MaxSmOccupancy(max_sm_occupancy, ExampleKernel<unsigned char>, 64);
 *
 * // max_sm_occupancy  <-- 4 on SM10
 * // max_sm_occupancy  <-- 8 on SM20
 * // max_sm_occupancy  <-- 12 on SM35
 *
 * \endcode
 *
 */
template <typename KernelPtr>
CUB_RUNTIME_FUNCTION __forceinline__
cudaError_t MaxSmOccupancy(
    int                 &max_sm_occupancy,          ///< [out] maximum number of thread blocks that can reside on a single SM
    KernelPtr           kernel_ptr,                 ///< [in] Kernel pointer for which to compute SM occupancy
    int                 block_threads,              ///< [in] Number of threads per thread block
    int                 dynamic_smem_bytes = 0)
{
#ifndef CUB_RUNTIME_ENABLED

    (void)dynamic_smem_bytes;
    (void)block_threads;
    (void)kernel_ptr;
    (void)max_sm_occupancy;

    // CUDA API calls not supported from this device
    return CubDebug(cudaErrorInvalidConfiguration);

#else

    return CubDebug(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_sm_occupancy,
        kernel_ptr,
        block_threads,
        dynamic_smem_bytes));

#endif  // CUB_RUNTIME_ENABLED
}


/******************************************************************************
 * Policy management
 ******************************************************************************/

/**
 * Kernel dispatch configuration
 */
struct KernelConfig
{
    int block_threads;
    int items_per_thread;
    int tile_size;
    int sm_occupancy;

    CUB_RUNTIME_FUNCTION __forceinline__
    KernelConfig() : block_threads(0), items_per_thread(0), tile_size(0), sm_occupancy(0) {}

    template <typename AgentPolicyT, typename KernelPtrT>
    CUB_RUNTIME_FUNCTION __forceinline__
    cudaError_t Init(KernelPtrT kernel_ptr)
    {
        block_threads        = AgentPolicyT::BLOCK_THREADS;
        items_per_thread     = AgentPolicyT::ITEMS_PER_THREAD;
        tile_size            = block_threads * items_per_thread;
        cudaError_t retval   = MaxSmOccupancy(sm_occupancy, kernel_ptr, block_threads);
        return retval;
    }
};



/// Helper for dispatching into a policy chain
template <int PTX_VERSION, typename PolicyT, typename PrevPolicyT>
struct ChainedPolicy
{
   /// The policy for the active compiler pass
   typedef typename If<(CUB_PTX_ARCH < PTX_VERSION), typename PrevPolicyT::ActivePolicy, PolicyT>::Type ActivePolicy;

   /// Specializes and dispatches op in accordance to the first policy in the chain of adequate PTX version
   template <typename FunctorT>
   CUB_RUNTIME_FUNCTION __forceinline__
   static cudaError_t Invoke(int ptx_version, FunctorT &op)
   {
       if (ptx_version < PTX_VERSION) {
           return PrevPolicyT::Invoke(ptx_version, op);
       }
       return op.template Invoke<PolicyT>();
   }
};

/// Helper for dispatching into a policy chain (end-of-chain specialization)
template <int PTX_VERSION, typename PolicyT>
struct ChainedPolicy<PTX_VERSION, PolicyT, PolicyT>
{
    /// The policy for the active compiler pass
    typedef PolicyT ActivePolicy;

    /// Specializes and dispatches op in accordance to the first policy in the chain of adequate PTX version
    template <typename FunctorT>
    CUB_RUNTIME_FUNCTION __forceinline__
    static cudaError_t Invoke(int /*ptx_version*/, FunctorT &op) {
        return op.template Invoke<PolicyT>();
    }
};




/** @} */       // end group UtilMgmt

}               // CUB namespace
CUB_NS_POSTFIX  // Optional outer namespace(s)
