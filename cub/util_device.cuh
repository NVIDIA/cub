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

#include "detail/device_synchronize.cuh"

#include "util_type.cuh"
#include "util_arch.cuh"
#include "util_debug.cuh"
#include "util_cpp_dialect.cuh"
#include "util_namespace.cuh"
#include "util_macro.cuh"

#include <atomic>
#include <array>
#include <cassert>

CUB_NAMESPACE_BEGIN


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
    void    *d_temp_storage,                    ///< [in] Device-accessible allocation of temporary storage.  When NULL, the required allocation size is written to \p temp_storage_bytes and no work is done.
    size_t& temp_storage_bytes,                 ///< [in,out] Size in bytes of \t d_temp_storage allocation
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

namespace TemporaryStorage
{

/**
 * @brief Named memory region of a temporary storage slot
 *
 * @par Overview
 * This class provides a typed wrapper of a temporary slot memory region.
 * It can be considered as a field in the C++ union. It's only possible to
 * increase the array size.
 *
 * @warning Array lifetime is defined by the lifetime of the associated slot
 *          It's impossible to grow the array if the layout is already mapped.
 */
template <typename T>
class Array
{
  void *&slot_pointer;
  std::size_t &slot_bytes;
  std::size_t elements{};

public:
  Array() = delete;

  __host__ __device__
  explicit Array(void *&slot_pointer, std::size_t &slot_bytes)
      : slot_pointer(slot_pointer)
      , slot_bytes(slot_bytes)
  {}

  __host__ __device__
  explicit Array(void *&slot_pointer,
                 std::size_t &slot_bytes,
                 std::size_t elements)
      : slot_pointer(slot_pointer)
      , slot_bytes(slot_bytes)
      , elements(elements)
  {
    UpdateSlot();
  }

  /**
   * @brief Increases the number of elements
   *
   * @warning This method should be called before temporary storage mapping stage.
   *
   * @param[in] new_elements Increases the memory region occupied in the
   *                         temporary slot to fit up to @p new_elements items
   *                         of type @p T.
   */
  __host__ __device__
  void Grow(std::size_t new_elements)
  {
    elements = new_elements;
    UpdateSlot();
  }

  /**
   * @brief Returns pointer to array
   *
   * If the @p elements number is equal to zero, or storage layout isn't mapped
   * @p nullptr is returned.
   */
  __host__ __device__
  T *Get() const
  {
    if (elements == 0)
    {
      return nullptr;
    }

    return reinterpret_cast<T *>(slot_pointer);
  }

private:
  __host__ __device__
  void UpdateSlot()
  {
    slot_bytes = (max)(slot_bytes, elements * sizeof(T));
  }
};

/**
 * @brief Temporary storage slot that can be considered a C++ union with an
 *        arbitrary fields count.
 *
 * @warning Slot lifetime is defined by the lifetime of the associated layout.
 *          It's impossible to request new array if layout is already mapped.
 *
 * @par A Simple Example
 * @code
 * auto slot = temporary_storage.GetSlot(0);
 *
 * // Add fields into the slot
 * auto int_array = slot->GetAlias<int>();
 * auto double_array = slot->GetAlias<double>(2);
 * auto empty_array = slot->GetAlias<char>();
 * // Slot size is defined by double_array size (2 * sizeof(double))
 *
 * if (condition)
 * {
 *   int_array.Grow(42);
 *   // Now slot size is defined by int_array size (42 * sizeof(int))
 * }
 *
 * // Temporary storage mapping
 * // ...

 * int *d_int_array = int_array.Get();
 * double *d_double_array = double_array.Get();
 * char *d_empty_array = empty_array.Get(); // Guaranteed to return nullptr
 * @endcode
 */
class Slot
{
  std::size_t size{};
  void *pointer{};

public:
  Slot() = default;

  __host__ __device__
  std::size_t GetBytesRequired() const { return size; }

  __host__ __device__
  void SetStorage(void *ptr) { pointer = ptr; }

  /**
   * @brief Returns empty array of type @p T
   */
  template <typename T>
  __host__ __device__
  Array<T> GetAlias()
  {
    return Array<T>(pointer, size);
  }

  /**
   * @brief Returns an array of type @p T and length @p elements
   */
  template <typename T>
  __host__ __device__
  Array<T> GetAlias(std::size_t elements)
  {
    return Array<T>(pointer, size, elements);
  }
};

/**
 * @brief Temporary storage layout represents a structure with
 *        @p SlotsCount union-like fields
 *
 * @par A Simple Example
 * @code
 * cub::TemporaryStorage::Layout<3> temporary_storage;
 *
 * auto slot_0 = temporary_storage.GetSlot(0);
 * auto slot_1 = temporary_storage.GetSlot(1);
 *
 * // Add fields into the first slot
 * auto int_array = slot_1->GetAlias<int>(1);
 * auto double_array = slot_1->GetAlias<double>(2);
 *
 * // Add fields into the second slot
 * auto char_array = slot_2->GetAlias<char>();
 *
 * // The third slot is empty
 *
 * // Temporary storage mapping
 * if (d_temp_storage == nullptr)
 * {
 *   temp_storage_bytes = temporary_storage.GetSize();
 *   return;
 * }
 * else
 * {
 *   temporary_storage.MapToBuffer(d_temp_storage, temp_storage_bytes);
 * }
 *
 * // Use pointers
 * int *d_int_array = int_array.Get();
 * double *d_double_array = double_array.Get();
 * char *d_char_array = char_array.Get();
 * @endcode
 */
template <int SlotsCount>
class Layout
{
  Slot slots[SlotsCount];

  std::size_t sizes[SlotsCount];
  void *pointers[SlotsCount];

public:
  Layout() = default;

  __host__ __device__
  Slot *GetSlot(int slot_id)
  {
    if (slot_id < SlotsCount)
    {
      return &slots[slot_id];
    }

    return nullptr;
  }

  /**
   * @brief Returns required temporary storage size in bytes
   */
  __host__ __device__
  std::size_t GetSize()
  {
    PrepareInterface();

    // AliasTemporaries can return error only in mapping stage,
    // so it's safe to ignore it here.
    std::size_t temp_storage_bytes{};
    AliasTemporaries(nullptr, temp_storage_bytes, pointers, sizes);

    if (temp_storage_bytes == 0)
    {
      return 1;
    }

    return temp_storage_bytes;
  }

  /**
   * @brief Maps the layout to the temporary storage buffer.
   */
  __host__ __device__
  cudaError_t MapToBuffer(void *d_temp_storage, std::size_t temp_storage_bytes)
  {
    PrepareInterface();

    cudaError_t error = cudaSuccess;
    if ((error = AliasTemporaries(d_temp_storage,
                                  temp_storage_bytes,
                                  pointers,
                                  sizes)))
    {
      return error;
    }

    for (std::size_t slot_id = 0; slot_id < SlotsCount; slot_id++)
    {
      slots[slot_id].SetStorage(pointers[slot_id]);
    }

    return error;
  }

private:
  __host__ __device__
  void PrepareInterface()
  {
    for (std::size_t slot_id = 0; slot_id < SlotsCount; slot_id++)
    {
      const std::size_t slot_size = slots[slot_id].GetBytesRequired();

      sizes[slot_id]    = slot_size;
      pointers[slot_id] = nullptr;
    }
  }
};

} // namespace TemporaryStorage

/**
 * \brief Empty kernel for querying PTX manifest metadata (e.g., version) for the current device
 */
template <typename T>
__global__ void EmptyKernel(void) { }

#endif  // DOXYGEN_SHOULD_SKIP_THIS

/**
 * \brief Returns the current device or -1 if an error occurred.
 */
CUB_RUNTIME_FUNCTION inline int CurrentDevice()
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
    bool const needs_reset;
public:
    __host__ inline SwitchDevice(int new_device)
      : old_device(CurrentDevice()), needs_reset(old_device != new_device)
    {
        if (needs_reset)
            CubDebug(cudaSetDevice(new_device));
    }

    __host__ inline ~SwitchDevice()
    {
        if (needs_reset)
            CubDebug(cudaSetDevice(old_device));
    }
};

/**
 * \brief Returns the number of CUDA devices available or -1 if an error
 *        occurred.
 */
CUB_RUNTIME_FUNCTION inline int DeviceCountUncached()
{
#if defined(CUB_RUNTIME_ENABLED) // Host code or device code with the CUDA runtime.

    int count = -1;
    if (CubDebug(cudaGetDeviceCount(&count)))
        // CUDA makes no guarantees about the state of the output parameter if
        // `cudaGetDeviceCount` fails; in practice, they don't, but out of
        // paranoia we'll reset `count` to `-1`.
        count = -1;
    return count;

#else // Device code without the CUDA runtime.

    return -1;

#endif
}

#if CUB_CPP_DIALECT >= 2011 // C++11 and later.

/**
 * \brief Cache for an arbitrary value produced by a nullary function.
 */
template <typename T, T(*Function)()>
struct ValueCache
{
    T const value;

    /**
     * \brief Call the nullary function to produce the value and construct the
     *        cache.
     */
    __host__ inline ValueCache() : value(Function()) {}
};

#endif

#if CUB_CPP_DIALECT >= 2011
// Host code, only safely usable in C++11 or newer, where thread-safe
// initialization of static locals is guaranteed.  This is a separate function
// to avoid defining a local static in a host/device function.
__host__ inline int DeviceCountCachedValue()
{
    static ValueCache<int, DeviceCountUncached> cache;
    return cache.value;
}
#endif

/**
 * \brief Returns the number of CUDA devices available.
 *
 * \note This function may cache the result internally.
 *
 * \note This function is thread safe.
 */
CUB_RUNTIME_FUNCTION inline int DeviceCount()
{
    int result = -1;
    if (CUB_IS_HOST_CODE) {
        #if CUB_INCLUDE_HOST_CODE
            #if CUB_CPP_DIALECT >= 2011
                // Host code and C++11.
                result = DeviceCountCachedValue();
            #else
                // Host code and C++98.
                result = DeviceCountUncached();
            #endif
        #endif
    } else {
        #if CUB_INCLUDE_DEVICE_CODE
            // Device code.
            result = DeviceCountUncached();
        #endif
    }
    return result;
}

#if CUB_CPP_DIALECT >= 2011 // C++11 and later.

/**
 * \brief Per-device cache for a CUDA attribute value; the attribute is queried
 *        and stored for each device upon construction.
 */
struct PerDeviceAttributeCache
{
    struct DevicePayload
    {
        int         attribute;
        cudaError_t error;
    };

    // Each entry starts in the `DeviceEntryEmpty` state, then proceeds to the
    // `DeviceEntryInitializing` state, and then proceeds to the
    // `DeviceEntryReady` state. These are the only state transitions allowed;
    // e.g. a linear sequence of transitions.
    enum DeviceEntryStatus
    {
        DeviceEntryEmpty = 0,
        DeviceEntryInitializing,
        DeviceEntryReady
    };

    struct DeviceEntry
    {
        std::atomic<DeviceEntryStatus> flag;
        DevicePayload                  payload;
    };

private:
    std::array<DeviceEntry, CUB_MAX_DEVICES> entries_;

public:
    /**
     * \brief Construct the cache.
     */
    __host__ inline PerDeviceAttributeCache() : entries_()
    {
        assert(DeviceCount() <= CUB_MAX_DEVICES);
    }

    /**
     * \brief Retrieves the payload of the cached function \p f for \p device.
     *
     * \note You must pass a morally equivalent function in to every call or
     *       this function has undefined behavior.
     */
    template <typename Invocable>
    __host__ DevicePayload operator()(Invocable&& f, int device)
    {
        if (device >= DeviceCount())
            return DevicePayload{0, cudaErrorInvalidDevice};

        auto& entry   = entries_[device];
        auto& flag    = entry.flag;
        auto& payload = entry.payload;

        DeviceEntryStatus old_status = DeviceEntryEmpty;

        // First, check for the common case of the entry being ready.
        if (flag.load(std::memory_order_acquire) != DeviceEntryReady)
        {
            // Assume the entry is empty and attempt to lock it so we can fill
            // it by trying to set the state from `DeviceEntryReady` to
            // `DeviceEntryInitializing`.
            if (flag.compare_exchange_strong(old_status, DeviceEntryInitializing,
                                             std::memory_order_acq_rel,
                                             std::memory_order_acquire))
            {
                // We successfully set the state to `DeviceEntryInitializing`;
                // we have the lock and it's our job to initialize this entry
                // and then release it.

                // We don't use `CubDebug` here because we let the user code
                // decide whether or not errors are hard errors.
                payload.error = std::forward<Invocable>(f)(payload.attribute);
                if (payload.error)
                    // Clear the global CUDA error state which may have been
                    // set by the last call. Otherwise, errors may "leak" to
                    // unrelated kernel launches.
                    cudaGetLastError();

                // Release the lock by setting the state to `DeviceEntryReady`.
                flag.store(DeviceEntryReady, std::memory_order_release);
            }

            // If the `compare_exchange_weak` failed, then `old_status` has
            // been updated with the value of `flag` that it observed.

            else if (old_status == DeviceEntryInitializing)
            {
                // Another execution agent is initializing this entry; we need
                // to wait for them to finish; we'll know they're done when we
                // observe the entry status as `DeviceEntryReady`.
                do { old_status = flag.load(std::memory_order_acquire); }
                while (old_status != DeviceEntryReady);
                // FIXME: Use `atomic::wait` instead when we have access to
                // host-side C++20 atomics. We could use libcu++, but it only
                // supports atomics for SM60 and up, even if you're only using
                // them in host code.
            }
        }

        // We now know that the state of our entry is `DeviceEntryReady`, so
        // just return the entry's payload.
        return entry.payload;
    }
};

#endif

/**
 * \brief Retrieves the PTX version that will be used on the current device (major * 100 + minor * 10).
 */
CUB_RUNTIME_FUNCTION inline cudaError_t PtxVersionUncached(int& ptx_version)
{
    // Instantiate `EmptyKernel<void>` in both host and device code to ensure
    // it can be called.
    typedef void (*EmptyKernelPtr)();
    EmptyKernelPtr empty_kernel = EmptyKernel<void>;

    // This is necessary for unused variable warnings in host compilers. The
    // usual syntax of (void)empty_kernel; was not sufficient on MSVC2015.
    (void)reinterpret_cast<void*>(empty_kernel);

    cudaError_t result = cudaSuccess;
    if (CUB_IS_HOST_CODE) {
       #if CUB_INCLUDE_HOST_CODE
            cudaFuncAttributes empty_kernel_attrs;

            do {
                if (CubDebug(result = cudaFuncGetAttributes(&empty_kernel_attrs, empty_kernel)))
                    break;
            }
            while(0);

            ptx_version = empty_kernel_attrs.ptxVersion * 10;
        #endif
    } else {
        #if CUB_INCLUDE_DEVICE_CODE
            // This is necessary to ensure instantiation of EmptyKernel in device code.
            // The `reinterpret_cast` is necessary to suppress a set-but-unused warnings.
            // This is a meme now: https://twitter.com/blelbach/status/1222391615576100864
            (void)reinterpret_cast<EmptyKernelPtr>(empty_kernel);

            ptx_version = CUB_PTX_ARCH;
        #endif
    }
    return result;
}

/**
 * \brief Retrieves the PTX version that will be used on \p device (major * 100 + minor * 10).
 */
__host__ inline cudaError_t PtxVersionUncached(int& ptx_version, int device)
{
    SwitchDevice sd(device);
    (void)sd;
    return PtxVersionUncached(ptx_version);
}

#if CUB_CPP_DIALECT >= 2011 // C++11 and later.
template <typename Tag>
__host__ inline PerDeviceAttributeCache& GetPerDeviceAttributeCache()
{
    // C++11 guarantees that initialization of static locals is thread safe.
    static PerDeviceAttributeCache cache;
    return cache;
}

struct PtxVersionCacheTag {};
struct SmVersionCacheTag {};
#endif

/**
 * \brief Retrieves the PTX version that will be used on \p device (major * 100 + minor * 10).
 *
 * \note This function may cache the result internally.
 *
 * \note This function is thread safe.
 */
__host__ inline cudaError_t PtxVersion(int& ptx_version, int device)
{
#if CUB_CPP_DIALECT >= 2011 // C++11 and later.

    auto const payload = GetPerDeviceAttributeCache<PtxVersionCacheTag>()(
      // If this call fails, then we get the error code back in the payload,
      // which we check with `CubDebug` below.
      [=] (int& pv) { return PtxVersionUncached(pv, device); },
      device);

    if (!CubDebug(payload.error))
        ptx_version = payload.attribute;

    return payload.error;

#else // Pre C++11.

    return PtxVersionUncached(ptx_version, device);

#endif
}

/**
 * \brief Retrieves the PTX version that will be used on the current device (major * 100 + minor * 10).
 *
 * \note This function may cache the result internally.
 *
 * \note This function is thread safe.
 */
CUB_RUNTIME_FUNCTION inline cudaError_t PtxVersion(int& ptx_version)
{
    cudaError_t result = cudaErrorUnknown;
    if (CUB_IS_HOST_CODE) {
        #if CUB_INCLUDE_HOST_CODE
            #if CUB_CPP_DIALECT >= 2011
                // Host code and C++11.
                auto const device = CurrentDevice();

                auto const payload = GetPerDeviceAttributeCache<PtxVersionCacheTag>()(
                  // If this call fails, then we get the error code back in the payload,
                  // which we check with `CubDebug` below.
                  [=] (int& pv) { return PtxVersionUncached(pv, device); },
                  device);

                if (!CubDebug(payload.error))
                    ptx_version = payload.attribute;

                result = payload.error;
            #else
                // Host code and C++98.
                result = PtxVersionUncached(ptx_version);
            #endif
        #endif
    } else {
        #if CUB_INCLUDE_DEVICE_CODE
            // Device code.
            result = PtxVersionUncached(ptx_version);
        #endif
    }
    return result;
}

/**
 * \brief Retrieves the SM version of \p device (major * 100 + minor * 10)
 */
CUB_RUNTIME_FUNCTION inline cudaError_t SmVersionUncached(int& sm_version, int device = CurrentDevice())
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
 *
 * \note This function is thread safe.
 */
CUB_RUNTIME_FUNCTION inline cudaError_t SmVersion(int& sm_version, int device = CurrentDevice())
{
    cudaError_t result = cudaErrorUnknown;
    if (CUB_IS_HOST_CODE) {
        #if CUB_INCLUDE_HOST_CODE
            #if CUB_CPP_DIALECT >= 2011
                // Host code and C++11
                auto const payload = GetPerDeviceAttributeCache<SmVersionCacheTag>()(
                  // If this call fails, then we get the error code back in the payload,
                  // which we check with `CubDebug` below.
                  [=] (int& pv) { return SmVersionUncached(pv, device); },
                  device);

                if (!CubDebug(payload.error))
                    sm_version = payload.attribute;

                result = payload.error;
            #else
                // Host code and C++98
                result = SmVersionUncached(sm_version, device);
            #endif
        #endif
    } else {
        #if CUB_INCLUDE_DEVICE_CODE
            result = SmVersionUncached(sm_version, device);
        #endif
    }
    return result;
}

/**
 * Synchronize the specified \p stream.
 */
CUB_RUNTIME_FUNCTION inline cudaError_t SyncStream(cudaStream_t stream)
{
    cudaError_t result = cudaErrorUnknown;
    if (CUB_IS_HOST_CODE) {
        #if CUB_INCLUDE_HOST_CODE
            result = CubDebug(cudaStreamSynchronize(stream));
        #endif
    } else {
        #if CUB_INCLUDE_DEVICE_CODE
            #if defined(CUB_RUNTIME_ENABLED) // Device code with the CUDA runtime.
                (void)stream;
                // Device can't yet sync on a specific stream
                result = CubDebug(cub::detail::device_synchronize());
            #else // Device code without the CUDA runtime.
                (void)stream;
                // CUDA API calls are not supported from this device.
                result = CubDebug(cudaErrorInvalidConfiguration);
            #endif
        #endif
    }
    return result;
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
CUB_RUNTIME_FUNCTION inline
cudaError_t MaxSmOccupancy(
    int&                max_sm_occupancy,          ///< [out] maximum number of thread blocks that can reside on a single SM
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
   static cudaError_t Invoke(int ptx_version, FunctorT& op)
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
    static cudaError_t Invoke(int /*ptx_version*/, FunctorT& op) {
        return op.template Invoke<PolicyT>();
    }
};




/** @} */       // end group UtilMgmt

CUB_NAMESPACE_END
