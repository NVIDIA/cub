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
 * Simple caching allocator for device memory allocations. The allocator is
 * thread-safe and capable of managing device allocations on multiple devices.
 ******************************************************************************/

#pragma once

#include <math.h>
#include <set>
#include <map>

#include "util_namespace.cuh"
#include "util_arch.cuh"
#include "util_debug.cuh"

#include "host/spinlock.cuh"

/// Optional outer namespace(s)
CUB_NS_PREFIX

/// CUB namespace
namespace cub {


/**
 * \addtogroup UtilModule
 * @{
 */


/******************************************************************************
 * DeviceAllocator abstract base class
 ******************************************************************************/

/**
 * \brief Abstract base allocator class for device memory allocations.
 */
class DeviceAllocator
{
public:

    /**
     * Provides a suitable allocation of device memory for the given size
     * on the current device
     */
    __host__ __device__ virtual cudaError_t DeviceAllocate(void** d_ptr, size_t bytes) = 0;


    /**
     * Frees a live allocation of device memory on the current device, returning it to the
     * allocator
     */
    __host__ __device__ virtual cudaError_t DeviceFree(void* d_ptr) = 0;

    /**
     * Destructor
     */
    CUB_DESTRUCTOR virtual ~DeviceAllocator() {};

};



/******************************************************************************
 * CachingDeviceAllocator (host use)
 ******************************************************************************/

/**
 * \brief A simple caching allocator for device memory allocations.
 *
 * \par Overview
 * The allocator is thread-safe and is capable of managing cached device allocations
 * on multiple devices.  It behaves as follows:
 *
 * \par
 * - Allocations categorized by bin size.
 * - Bin sizes progress geometrically in accordance with the growth factor
 *   \p bin_growth provided during construction.  Unused device allocations within
 *   a larger bin cache are not reused for allocation requests that categorize to
 *   smaller bin sizes.
 * - Allocation requests below (\p bin_growth ^ \p min_bin) are rounded up to
 *   (\p bin_growth ^ \p min_bin).
 * - Allocations above (\p bin_growth ^ \p max_bin) are not rounded up to the nearest
 *   bin and are simply freed when they are deallocated instead of being returned
 *   to a bin-cache.
 * - %If the total storage of cached allocations on a given device will exceed
 *   \p max_cached_bytes, allocations for that device are simply freed when they are
 *   deallocated instead of being returned to their bin-cache.
 *
 * \par
 * For example, the default-constructed CachingDeviceAllocator is configured with:
 * - \p bin_growth = 8
 * - \p min_bin = 3
 * - \p max_bin = 7
 * - \p max_cached_bytes = 6MB - 1B
 *
 * \par
 * which delineates five bin-sizes: 512B, 4KB, 32KB, 256KB, and 2MB
 * and sets a maximum of 6,291,455 cached bytes per device
 *
 */
struct CachingDeviceAllocator : DeviceAllocator
{
#ifndef DOXYGEN_SHOULD_SKIP_THIS    // Do not document


    //---------------------------------------------------------------------
    // Type definitions and constants
    //---------------------------------------------------------------------

    /**
     * Integer pow function for unsigned base and exponent
     */
    static __host__ __device__ __forceinline__ unsigned int IntPow(
        unsigned int base,
        unsigned int exp)
    {
        unsigned int retval = 1;
        while (exp > 0)
        {
            if (exp & 1) {
                retval = retval * base;        // multiply the result by the current base
            }
            base = base * base;                // square the base
            exp = exp >> 1;                    // divide the exponent in half
        }
        return retval;
    }


    /**
     * Round up to the nearest power-of
     */
    static __host__ __device__ __forceinline__ void NearestPowerOf(
        unsigned int &power,
        size_t &rounded_bytes,
        unsigned int base,
        size_t value)
    {
        power = 0;
        rounded_bytes = 1;

        while (rounded_bytes < value)
        {
            rounded_bytes *= base;
            power++;
        }
    }

    /**
     * Descriptor for device memory allocations
     */
    struct BlockDescriptor
    {
        int   device;        // device ordinal
        void*           d_ptr;      // Device pointer
        size_t          bytes;      // Size of allocation in bytes
        unsigned int    bin;        // Bin enumeration

        // Constructor
        BlockDescriptor(void *d_ptr, int device) :
            d_ptr(d_ptr),
            bytes(0),
            bin(0),
            device(device) {}

        // Constructor
        BlockDescriptor(size_t bytes, unsigned int bin, int device) :
            d_ptr(NULL),
            bytes(bytes),
            bin(bin),
            device(device) {}

        // Comparison functor for comparing device pointers
        static bool PtrCompare(const BlockDescriptor &a, const BlockDescriptor &b)
        {
            if (a.device < b.device) {
                return true;
            } else if (a.device > b.device) {
                return false;
            } else {
                return (a.d_ptr < b.d_ptr);
            }
        }

        // Comparison functor for comparing allocation sizes
        static bool SizeCompare(const BlockDescriptor &a, const BlockDescriptor &b)
        {
            if (a.device < b.device) {
                return true;
            } else if (a.device > b.device) {
                return false;
            } else {
                return (a.bytes < b.bytes);
            }
        }
    };

    /// BlockDescriptor comparator function interface
    typedef bool (*Compare)(const BlockDescriptor &, const BlockDescriptor &);

    /// Set type for cached blocks (ordered by size)
    typedef std::multiset<BlockDescriptor, Compare> CachedBlocks;

    /// Set type for live blocks (ordered by ptr)
    typedef std::multiset<BlockDescriptor, Compare> BusyBlocks;

    /// Map type of device ordinals to the number of cached bytes cached by each device
    typedef std::map<int, size_t> GpuCachedBytes;


    //---------------------------------------------------------------------
    // Fields
    //---------------------------------------------------------------------

    Spinlock        spin_lock;          /// Spinlock for thread-safety

    unsigned int    bin_growth;         /// Geometric growth factor for bin-sizes
    unsigned int    min_bin;            /// Minimum bin enumeration
    unsigned int    max_bin;            /// Maximum bin enumeration

    size_t          min_bin_bytes;      /// Minimum bin size
    size_t          max_bin_bytes;      /// Maximum bin size
    size_t          max_cached_bytes;   /// Maximum aggregate cached bytes per device

    bool            debug;              /// Whether or not to print (de)allocation events to stdout
    bool            skip_cleanup;       /// Whether or not to skip a call to FreeAllCached() when destructor is called.  (The CUDA runtime may have already shut down for statically declared allocators)

#ifndef __CUDA_ARCH__   // Only define STL container members in host code

    GpuCachedBytes  cached_bytes;       /// Map of device ordinal to aggregate cached bytes on that device
    CachedBlocks    cached_blocks;      /// Set of cached device allocations available for reuse
    BusyBlocks      live_blocks;        /// Set of live device allocations currently in use

#endif // __CUDA_ARCH__

#endif // DOXYGEN_SHOULD_SKIP_THIS

    //---------------------------------------------------------------------
    // Methods
    //---------------------------------------------------------------------

    /**
     * \brief Constructor.
     */
    __host__ __device__ __forceinline__ CachingDeviceAllocator(
        unsigned int bin_growth,    ///< Geometric growth factor for bin-sizes
        unsigned int min_bin,       ///< Minimum bin
        unsigned int max_bin,       ///< Maximum bin
        size_t max_cached_bytes)    ///< Maximum aggregate cached bytes per device
    :
    #ifndef __CUDA_ARCH__   // Only define STL container members in host code
            cached_blocks(BlockDescriptor::SizeCompare),
            live_blocks(BlockDescriptor::PtrCompare),
    #endif
            debug(false),
            spin_lock(0),
            bin_growth(bin_growth),
            min_bin(min_bin),
            max_bin(max_bin),
            min_bin_bytes(IntPow(bin_growth, min_bin)),
            max_bin_bytes(IntPow(bin_growth, max_bin)),
            max_cached_bytes(max_cached_bytes)
    {}


    /**
     * \brief Default constructor.
     *
     * Configured with:
     * \par
     * - \p bin_growth = 8
     * - \p min_bin = 3
     * - \p max_bin = 7
     * - \p max_cached_bytes = (\p bin_growth ^ \p max_bin) * 3) - 1 = 6,291,455 bytes
     *
     * which delineates five bin-sizes: 512B, 4KB, 32KB, 256KB, and 2MB and
     * sets a maximum of 6,291,455 cached bytes per device
     */
    __host__ __device__ __forceinline__ CachingDeviceAllocator(bool skip_cleanup = false) :
    #ifndef __CUDA_ARCH__   // Only define STL container members in host code
        cached_blocks(BlockDescriptor::SizeCompare),
        live_blocks(BlockDescriptor::PtrCompare),
    #endif
        skip_cleanup(skip_cleanup),
        debug(false),
        spin_lock(0),
        bin_growth(8),
        min_bin(3),
        max_bin(7),
        min_bin_bytes(IntPow(bin_growth, min_bin)),
        max_bin_bytes(IntPow(bin_growth, max_bin)),
        max_cached_bytes((max_bin_bytes * 3) - 1)
    {}


    /**
     * \brief Sets the limit on the number bytes this allocator is allowed to cache per device.
     */
    __host__ __device__ __forceinline__ cudaError_t SetMaxCachedBytes(
        size_t max_cached_bytes)
    {
    #ifdef __CUDA_ARCH__
        // Caching functionality only defined on host
        return CubDebug(cudaErrorInvalidConfiguration);
    #else

        // Lock
        Lock(&spin_lock);

        this->max_cached_bytes = max_cached_bytes;

        if (debug) printf("New max_cached_bytes(%lld)\n", (long long) max_cached_bytes);

        // Unlock
        Unlock(&spin_lock);

        return cudaSuccess;

    #endif  // __CUDA_ARCH__
    }


    /**
     * \brief Provides a suitable allocation of device memory for the given size on the specified device
     */
    __host__ __device__ __forceinline__ cudaError_t DeviceAllocate(
        void** d_ptr,
        size_t bytes,
        int device)
    {
    #ifdef __CUDA_ARCH__
        // Caching functionality only defined on host
        return CubDebug(cudaErrorInvalidConfiguration);
    #else

        bool locked                     = false;
        int entrypoint_device = INVALID_DEVICE_ORDINAL;
        cudaError_t error               = cudaSuccess;

        // Round up to nearest bin size
        unsigned int bin;
        size_t bin_bytes;
        NearestPowerOf(bin, bin_bytes, bin_growth, bytes);
        if (bin < min_bin) {
            bin = min_bin;
            bin_bytes = min_bin_bytes;
        }

        // Check if bin is greater than our maximum bin
        if (bin > max_bin)
        {
            // Allocate the request exactly and give out-of-range bin
            bin = (unsigned int) -1;
            bin_bytes = bytes;
        }

        BlockDescriptor search_key(bin_bytes, bin, device);

        // Lock
        if (!locked) {
            Lock(&spin_lock);
            locked = true;
        }

        do {
            // Find a free block big enough within the same bin on the same device
            CachedBlocks::iterator block_itr = cached_blocks.lower_bound(search_key);
            if ((block_itr != cached_blocks.end()) &&
                (block_itr->device == device) &&
                (block_itr->bin == search_key.bin))
            {
                // Reuse existing cache block.  Insert into live blocks.
                search_key = *block_itr;
                live_blocks.insert(search_key);

                // Remove from free blocks
                cached_blocks.erase(block_itr);
                cached_bytes[device] -= search_key.bytes;

                if (debug) printf("\tdevice %d reused cached block (%lld bytes). %lld available blocks cached (%lld bytes), %lld live blocks outstanding.\n",
                    device, (long long) search_key.bytes, (long long) cached_blocks.size(), (long long) cached_bytes[device], (long long) live_blocks.size());
            }
            else
            {
                // Need to allocate a new cache block. Unlock.
                if (locked) {
                    Unlock(&spin_lock);
                    locked = false;
                }

                // Set to specified device
                if (CubDebug(error = cudaGetDevice(&entrypoint_device))) break;
                if (CubDebug(error = cudaSetDevice(device))) break;

                // Allocate
                if (CubDebug(error = cudaMalloc(&search_key.d_ptr, search_key.bytes))) break;

                // Lock
                if (!locked) {
                    Lock(&spin_lock);
                    locked = true;
                }

                // Insert into live blocks
                live_blocks.insert(search_key);

                if (debug) printf("\tdevice %d allocating new device block %lld bytes. %lld available blocks cached (%lld bytes), %lld live blocks outstanding.\n",
                    device, (long long) search_key.bytes, (long long) cached_blocks.size(), (long long) cached_bytes[device], (long long) live_blocks.size());
            }
        } while(0);

        // Unlock
        if (locked) {
            Unlock(&spin_lock);
            locked = false;
        }

        // Copy device pointer to output parameter (NULL on error)
        *d_ptr = search_key.d_ptr;

        // Attempt to revert back to previous device if necessary
        if (entrypoint_device != INVALID_DEVICE_ORDINAL)
        {
            if (CubDebug(error = cudaSetDevice(entrypoint_device))) return error;
        }

        return error;

    #endif  // __CUDA_ARCH__
    }


    /**
     * \brief Provides a suitable allocation of device memory for the given size on the current device
     */
    __host__ __device__ __forceinline__ cudaError_t DeviceAllocate(
        void** d_ptr,
        size_t bytes)
    {
    #ifdef __CUDA_ARCH__
        // Caching functionality only defined on host
        return CubDebug(cudaErrorInvalidConfiguration);
    #else
        cudaError_t error = cudaSuccess;
        do {
            int current_device;
            if (CubDebug(error = cudaGetDevice(&current_device))) break;
            if (CubDebug(error = DeviceAllocate(d_ptr, bytes, current_device))) break;
        } while(0);

        return error;

    #endif  // __CUDA_ARCH__
    }


    /**
     * \brief Frees a live allocation of device memory on the specified device, returning it to the allocator
     */
    __host__ __device__ __forceinline__ cudaError_t DeviceFree(
        void* d_ptr,
        int device)
    {
    #ifdef __CUDA_ARCH__
        // Caching functionality only defined on host
        return CubDebug(cudaErrorInvalidConfiguration);
    #else

        bool locked                     = false;
        int entrypoint_device = INVALID_DEVICE_ORDINAL;
        cudaError_t error               = cudaSuccess;

        BlockDescriptor search_key(d_ptr, device);

        // Lock
        if (!locked) {
            Lock(&spin_lock);
            locked = true;
        }

        do {
            // Find corresponding block descriptor
            BusyBlocks::iterator block_itr = live_blocks.find(search_key);
            if (block_itr == live_blocks.end())
            {
                // Cannot find pointer
                if (CubDebug(error = cudaErrorUnknown)) break;
            }
            else
            {
                // Remove from live blocks
                search_key = *block_itr;
                live_blocks.erase(block_itr);

                // Check if we should keep the returned allocation
                if (cached_bytes[device] + search_key.bytes <= max_cached_bytes)
                {
                    // Insert returned allocation into free blocks
                    cached_blocks.insert(search_key);
                    cached_bytes[device] += search_key.bytes;

                    if (debug) printf("\tdevice %d returned %lld bytes. %lld available blocks cached (%lld bytes), %lld live blocks outstanding.\n",
                        device, (long long) search_key.bytes, (long long) cached_blocks.size(), (long long) cached_bytes[device], (long long) live_blocks.size());
                }
                else
                {
                    // Free the returned allocation.  Unlock.
                    if (locked) {
                        Unlock(&spin_lock);
                        locked = false;
                    }

                    // Set to specified device
                    if (CubDebug(error = cudaGetDevice(&entrypoint_device))) break;
                    if (CubDebug(error = cudaSetDevice(device))) break;

                    // Free device memory
                    if (CubDebug(error = cudaFree(d_ptr))) break;

                    if (debug) printf("\tdevice %d freed %lld bytes.  %lld available blocks cached (%lld bytes), %lld live blocks outstanding.\n",
                        device, (long long) search_key.bytes, (long long) cached_blocks.size(), (long long) cached_bytes[device], (long long) live_blocks.size());
                }
            }
        } while (0);

        // Unlock
        if (locked) {
            Unlock(&spin_lock);
            locked = false;
        }

        // Attempt to revert back to entry-point device if necessary
        if (entrypoint_device != INVALID_DEVICE_ORDINAL)
        {
            if (CubDebug(error = cudaSetDevice(entrypoint_device))) return error;
        }

        return error;

    #endif  // __CUDA_ARCH__
    }


    /**
     * \brief Frees a live allocation of device memory on the current device, returning it to the allocator
     */
    __host__ __device__ __forceinline__ cudaError_t DeviceFree(
        void* d_ptr)
    {
    #ifdef __CUDA_ARCH__
        // Caching functionality only defined on host
        return CubDebug(cudaErrorInvalidConfiguration);
    #else

        int current_device;
        cudaError_t error = cudaSuccess;

        do {
            if (CubDebug(error = cudaGetDevice(&current_device))) break;
            if (CubDebug(error = DeviceFree(d_ptr, current_device))) break;
        } while(0);

        return error;

    #endif  // __CUDA_ARCH__
    }


    /**
     * \brief Frees all cached device allocations on all devices
     */
    __host__ __device__ __forceinline__ cudaError_t FreeAllCached()
    {
    #ifdef __CUDA_ARCH__
        // Caching functionality only defined on host
        return CubDebug(cudaErrorInvalidConfiguration);
    #else

        cudaError_t error                   = cudaSuccess;
        bool locked                         = false;
        int entrypoint_device     = INVALID_DEVICE_ORDINAL;
        int current_device        = INVALID_DEVICE_ORDINAL;

        // Lock
        if (!locked) {
            Lock(&spin_lock);
            locked = true;
        }

        while (!cached_blocks.empty())
        {
            // Get first block
            CachedBlocks::iterator begin = cached_blocks.begin();

            // Get entry-point device ordinal if necessary
            if (entrypoint_device == INVALID_DEVICE_ORDINAL)
            {
                if (CubDebug(error = cudaGetDevice(&entrypoint_device))) break;
            }

            // Set current device ordinal if necessary
            if (begin->device != current_device)
            {
                if (CubDebug(error = cudaSetDevice(begin->device))) break;
                current_device = begin->device;
            }

            // Free device memory
            if (CubDebug(error = cudaFree(begin->d_ptr))) break;

            // Reduce balance and erase entry
            cached_bytes[current_device] -= begin->bytes;
            cached_blocks.erase(begin);

            if (debug) printf("\tdevice %d freed %lld bytes.  %lld available blocks cached (%lld bytes), %lld live blocks outstanding.\n",
                current_device, (long long) begin->bytes, (long long) cached_blocks.size(), (long long) cached_bytes[current_device], (long long) live_blocks.size());
        }

        // Unlock
        if (locked) {
            Unlock(&spin_lock);
            locked = false;
        }

        // Attempt to revert back to entry-point device if necessary
        if (entrypoint_device != INVALID_DEVICE_ORDINAL)
        {
            if (CubDebug(error = cudaSetDevice(entrypoint_device))) return error;
        }

        return error;

    #endif  // __CUDA_ARCH__
    }


    /**
     * \brief Destructor
     */
    CUB_DESTRUCTOR __forceinline__ virtual ~CachingDeviceAllocator()
    {
        if (!skip_cleanup)
            FreeAllCached();
    }

};


/******************************************************************************
 * DefaultDeviceAllocator (generic use)
 ******************************************************************************/

/// Anonymous namespace to prevent multiple symbol definition errors
namespace {

    /// Singleton, thread-safe caching allocator (one per compilation unit)
    CachingDeviceAllocator host_allocator_singleton(true);

}               // anonymous namespace

#ifndef __CUDA_ARCH__
    /**
     * \brief The default allocator for host and device usage.
     *
     * When compiled for the host, it delegates to a single
     * CachingDeviceAllocator instance.  Returns NULL When compiled for the
     * device.
     *
     * NOTE: Because CUB is implemented exclusively in header files, there will
     * be one CachingDeviceAllocator per compilation unit.
     */
    #define DefaultDeviceAllocator() (&host_allocator_singleton)
#else
    #define DefaultDeviceAllocator() NULL
#endif



/******************************************************************************
 * Shorthand CUB allocation operations
 ******************************************************************************/

#ifndef DOXYGEN_SHOULD_SKIP_THIS    // Do not document


/**
 * Provides a suitable allocation of device memory for the given size
 * on the current device
 */
__host__ __device__ __forceinline__ cudaError_t DeviceAllocate(
    void** d_ptr,
    size_t bytes,
    DeviceAllocator *device_allocator = DefaultDeviceAllocator())
{
    if (device_allocator == NULL)
    {
    #ifndef CUB_RUNTIME_ENABLED
        // CUDA API not supported from this device
        return CubDebug(cudaErrorInvalidConfiguration);
    #else
        return cudaMalloc(d_ptr, bytes);
    #endif
    }
    return device_allocator->DeviceAllocate(d_ptr, bytes);
}


/**
 * Frees a live allocation of device memory on the current device, returning it to the
 * allocator
 */
__host__ __device__ __forceinline__ cudaError_t DeviceFree(
    void* d_ptr,
    DeviceAllocator *device_allocator = DefaultDeviceAllocator())
{
    if (device_allocator == NULL)
    {
    #ifndef CUB_RUNTIME_ENABLED
        // CUDA API not supported from this device
        return CubDebug(cudaErrorInvalidConfiguration);
    #else
        return cudaFree(d_ptr);
    #endif
    }
    return device_allocator->DeviceFree(d_ptr);
}

#endif // DOXYGEN_SHOULD_SKIP_THIS



/** @} */       // end group UtilModule

}               // CUB namespace
CUB_NS_POSTFIX  // Optional outer namespace(s)
