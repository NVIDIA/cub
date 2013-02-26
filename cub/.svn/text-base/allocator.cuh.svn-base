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
 * thread-safe and capable of managing device allocations on multiple GPUs.
 ******************************************************************************/

#pragma once

#include <math.h>
#include <set>
#include <map>

#include "ns_wrapper.cuh"
#include "device_props.cuh"
#include "debug.cuh"

#include "host/spinlock.cuh"

CUB_NS_PREFIX
namespace cub {
namespace {         // Anonymous namespace to prevent multiple symbol definition errors


/******************************************************************************
 * DeviceAllocator abstract base class
 ******************************************************************************/

/**
 * Abstract base allocator class for device memory allocations.
 */
class DeviceAllocator
{
public:

    /**
     * Sets the limit on the number bytes this allocator is allowed to
     * cache per GPU.
     */
    virtual cudaError_t SetMaxCachedBytes(size_t max_cached_bytes) = 0;


    /**
     * Provides a suitable allocation of device memory for the given size
     * on the specified GPU
     */
    virtual cudaError_t DeviceAllocate(void** d_ptr, size_t bytes, GpuOrdinal gpu) = 0;


    /**
     * Provides a suitable allocation of device memory for the given size
     * on the current GPU
     */
    virtual cudaError_t DeviceAllocate(void** d_ptr, size_t bytes) = 0;


    /**
     * Frees a live allocation of GPU memory on the specified GPU, returning it to
     * the allocator
     */
    virtual cudaError_t DeviceFree(void* d_ptr, GpuOrdinal gpu) = 0;


    /**
     * Frees a live allocation of GPU memory on the current GPU, returning it to the
     * allocator
     */
    virtual cudaError_t DeviceFree(void* d_ptr) = 0;


    /**
     * Frees all cached device allocations on all GPUs
     */
    virtual cudaError_t FreeAllCached() = 0;
};



/******************************************************************************
 * CachingDeviceAllocator
 ******************************************************************************/


/**
 * Simple caching allocator for device memory allocations. The allocator is
 * thread-safe and is capable of managing cached device allocations on multiple GPUs.
 *
 * Allocations are rounded up to and categorized by bin size.  Bin sizes progress
 * geometrically in accordance with the growth factor "bin_growth" provided during
 * construction.  Unused device allocations within a larger bin cache are not
 * reused for allocation requests that categorize to smaller bin sizes.
 *
 * Allocation requests below (bin_growth ^ min_bin) are rounded up to
 * (bin_growth ^ min_bin).
 *
 * Allocations above (bin_growth ^ max_bin) are not rounded up to the nearest
 * bin and are simply freed when they are deallocated instead of being returned
 * to a bin-cache.
 *
 * If the total storage of cached allocations on a given GPU will exceed
 * (max_cached_bytes), allocations for that GPU are simply freed when they are
 * deallocated instead of being returned to their bin-cache.
 *
 * For example, the default-constructed CachingDeviceAllocator is configured with:
 *         bin_growth = 8
 *         min_bin = 3
 *         max_bin = 7
 *         max_cached_bytes = (bin_growth ^ max_bin) * 3) - 1 = 6,291,455 bytes
 *
 * which delineates five bin-sizes: 512B, 4KB, 32KB, 256KB, and 2MB
 * and sets a maximum of 6,291,455 cached bytes per GPU
 *
 */
struct CachingDeviceAllocator : public DeviceAllocator
{
    //---------------------------------------------------------------------
    // Type definitions and constants
    //---------------------------------------------------------------------

    /**
     * Integer pow function for unsigned base and exponent
     */
    static __forceinline__ unsigned int IntPow(
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
    static __forceinline__ void NearestPowerOf(
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
        GpuOrdinal            gpu;        // GPU ordinal
        void*                 d_ptr;        // Device pointer
        size_t                bytes;        // Size of allocation in bytes
        unsigned int        bin;        // Bin enumeration

        // Constructor
        BlockDescriptor(void *d_ptr, GpuOrdinal gpu) :
            d_ptr(d_ptr),
            bytes(0),
            bin(0),
            gpu(gpu) {}

        // Constructor
        BlockDescriptor(size_t bytes, unsigned int bin, GpuOrdinal gpu) :
            d_ptr(NULL),
            bytes(bytes),
            bin(bin),
            gpu(gpu) {}

        // Comparison functor for comparing device pointers
        static bool PtrCompare(const BlockDescriptor &a, const BlockDescriptor &b)
        {
            if (a.gpu < b.gpu) {
                return true;
            } else if (a.gpu > b.gpu) {
                return false;
            } else {
                return (a.d_ptr < b.d_ptr);
            }
        }

        // Comparison functor for comparing allocation sizes
        static bool SizeCompare(const BlockDescriptor &a, const BlockDescriptor &b)
        {
            if (a.gpu < b.gpu) {
                return true;
            } else if (a.gpu > b.gpu) {
                return false;
            } else {
                return (a.bytes < b.bytes);
            }
        }
    };

    // BlockDescriptor comparator function interface
    typedef bool (*Compare)(const BlockDescriptor &, const BlockDescriptor &);

    // Set type for cached blocks (ordered by size)
    typedef std::multiset<BlockDescriptor, Compare> CachedBlocks;

    // Set type for live blocks (ordered by ptr)
    typedef std::multiset<BlockDescriptor, Compare> BusyBlocks;

    // Map type of gpu ordinals to the number of cached bytes cached by each GPU
    typedef std::map<GpuOrdinal, size_t> GpuCachedBytes;


    //---------------------------------------------------------------------
    // Fields
    //---------------------------------------------------------------------

    Spinlock        spin_lock;            // Spinlock for thread-safety

    CachedBlocks    cached_blocks;        // Set of cached device allocations available for reuse
    BusyBlocks      live_blocks;        // Set of live device allocations currently in use

    unsigned int    bin_growth;            // Geometric growth factor for bin-sizes
    unsigned int    min_bin;            // Minimum bin enumeration
    unsigned int    max_bin;            // Maximum bin enumeration

    size_t          min_bin_bytes;        // Minimum bin size
    size_t          max_bin_bytes;        // Maximum bin size
    size_t          max_cached_bytes;    // Maximum aggregate cached bytes per GPU

    GpuCachedBytes  cached_bytes;        // Map of GPU ordinal to aggregate cached bytes on that GPU

    bool            debug;


    //---------------------------------------------------------------------
    // Methods
    //---------------------------------------------------------------------

    /**
     * Constructor.
     */
    CachingDeviceAllocator(
        unsigned int bin_growth,        // Geometric growth factor for bin-sizes
        unsigned int min_bin,            // Minimum bin
        unsigned int max_bin,            // Maximum bin
        size_t max_cached_bytes) :        // Maximum aggregate cached bytes per GPU
            debug(false),
            spin_lock(0),
            cached_blocks(BlockDescriptor::SizeCompare),
            live_blocks(BlockDescriptor::PtrCompare),
            bin_growth(bin_growth),
            min_bin(min_bin),
            max_bin(max_bin),
            min_bin_bytes(IntPow(bin_growth, min_bin)),
            max_bin_bytes(IntPow(bin_growth, max_bin)),
            max_cached_bytes(max_cached_bytes)
    {}


    /**
     * Constructor.  Configured with:
     *         bin_growth = 8
     *         min_bin = 3
     *         max_bin = 7
     *         max_cached_bytes = (bin_growth ^ max_bin) * 3) - 1 = 6,291,455 bytes
     *
     *     which delineates five bin-sizes: 512B, 4KB, 32KB, 256KB, and 2MB
     *     and sets a maximum of 6,291,455 cached bytes per GPU
     */
    CachingDeviceAllocator() :
        debug(false),
        spin_lock(0),
        cached_blocks(BlockDescriptor::SizeCompare),
        live_blocks(BlockDescriptor::PtrCompare),
        bin_growth(8),
        min_bin(3),
        max_bin(7),
        min_bin_bytes(IntPow(bin_growth, min_bin)),
        max_bin_bytes(IntPow(bin_growth, max_bin)),
        max_cached_bytes((max_bin_bytes * 3) - 1)
    {}


    /**
     * Sets the limit on the number bytes this allocator is allowed to
     * cache per GPU.
     */
    cudaError_t SetMaxCachedBytes(size_t max_cached_bytes)
    {
        // Lock
        Lock(&spin_lock);

        this->max_cached_bytes = max_cached_bytes;

        if (debug) printf("New max_cached_bytes(%d)\n", max_cached_bytes);

        // Unlock
        Unlock(&spin_lock);

        return cudaSuccess;
    }


    /**
     * Provides a suitable allocation of device memory for the given size
     * on the specified GPU
     */
    cudaError_t DeviceAllocate(void** d_ptr, size_t bytes, GpuOrdinal gpu)
    {
        bool locked                     = false;
        GpuOrdinal entrypoint_gpu         = INVALID_GPU_ORDINAL;
        cudaError_t error                 = cudaSuccess;

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

        BlockDescriptor search_key(bin_bytes, bin, gpu);

        // Lock
        if (!locked) {
            Lock(&spin_lock);
            locked = true;
        }

        do {
            // Find a free block big enough within the same bin on the same GPU
            CachedBlocks::iterator block_itr = cached_blocks.lower_bound(search_key);
            if ((block_itr != cached_blocks.end()) &&
                (block_itr->gpu == gpu) &&
                (block_itr->bin == search_key.bin))
            {
                // Reuse existing cache block.  Insert into live blocks.
                search_key = *block_itr;
                live_blocks.insert(search_key);

                // Remove from free blocks
                cached_blocks.erase(block_itr);
                cached_bytes[gpu] -= search_key.bytes;

                if (debug) printf("\tGPU %d reused cached block (%d bytes). %d available blocks cached (%d bytes), %d live blocks outstanding.\n",
                    gpu, search_key.bytes, cached_blocks.size(), cached_bytes[gpu], live_blocks.size());
            }
            else
            {
                // Need to allocate a new cache block. Unlock.
                if (locked) {
                    Unlock(&spin_lock);
                    locked = false;
                }

                // Set to specified GPU
                if (error = CubDebug(cudaGetDevice(&entrypoint_gpu))) break;
                if (error = CubDebug(cudaSetDevice(gpu))) break;

                // Allocate
                if (error = CubDebug(cudaMalloc(&search_key.d_ptr, search_key.bytes))) break;

                // Lock
                if (!locked) {
                    Lock(&spin_lock);
                    locked = true;
                }

                // Insert into live blocks
                live_blocks.insert(search_key);

                if (debug) printf("\tGPU %d allocating new device block %d bytes. %d available blocks cached (%d bytes), %d live blocks outstanding.\n",
                    gpu, search_key.bytes, cached_blocks.size(), cached_bytes[gpu], live_blocks.size());
            }
        } while(0);

        // Unlock
        if (locked) {
            Unlock(&spin_lock);
            locked = false;
        }

        // Copy device pointer to output parameter (NULL on error)
        *d_ptr = search_key.d_ptr;

        // Attempt to revert back to previous GPU if necessary
        if (entrypoint_gpu != INVALID_GPU_ORDINAL)
        {
            if (error = CubDebug(cudaSetDevice(entrypoint_gpu))) return error;
        }

        return error;
    }


    /**
     * Provides a suitable allocation of device memory for the given size
     * on the current GPU
     */
    cudaError_t DeviceAllocate(void** d_ptr, size_t bytes)
    {
        cudaError_t error = cudaSuccess;
        do {
            GpuOrdinal current_gpu;
            if (error = CubDebug(cudaGetDevice(&current_gpu))) break;
            if (error = DeviceAllocate(d_ptr, bytes, current_gpu)) break;
        } while(0);

        return error;
    }


    /**
     * Frees a live allocation of GPU memory on the specified GPU, returning it to
     * the allocator
     */
    cudaError_t DeviceFree(void* d_ptr, GpuOrdinal gpu)
    {
        bool locked                     = false;
        GpuOrdinal entrypoint_gpu         = INVALID_GPU_ORDINAL;
        cudaError_t error                 = cudaSuccess;

        BlockDescriptor search_key(d_ptr, gpu);

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
                if (error = CubDebug(cudaErrorUnknown)) break;
            }
            else
            {
                // Remove from live blocks
                search_key = *block_itr;
                live_blocks.erase(block_itr);

                // Check if we should keep the returned allocation
                if (cached_bytes[gpu] + search_key.bytes <= max_cached_bytes)
                {
                    // Insert returned allocation into free blocks
                    cached_blocks.insert(search_key);
                    cached_bytes[gpu] += search_key.bytes;

                    if (debug) printf("\tGPU %d returned %d bytes. %d available blocks cached (%d bytes), %d live blocks outstanding.\n",
                        gpu, search_key.bytes, cached_blocks.size(), cached_bytes[gpu], live_blocks.size());
                }
                else
                {
                    // Free the returned allocation.  Unlock.
                    if (locked) {
                        Unlock(&spin_lock);
                        locked = false;
                    }

                    // Set to specified GPU
                    if (error = CubDebug(cudaGetDevice(&entrypoint_gpu))) break;
                    if (error = CubDebug(cudaSetDevice(gpu))) break;

                    // Free device memory
                    if (error = CubDebug(cudaFree(d_ptr))) break;

                    if (debug) printf("\tGPU %d freed %d bytes.  %d available blocks cached (%d bytes), %d live blocks outstanding.\n",
                        gpu, search_key.bytes, cached_blocks.size(), cached_bytes[gpu], live_blocks.size());
                }
            }
        } while (0);

        // Unlock
        if (locked) {
            Unlock(&spin_lock);
            locked = false;
        }

        // Attempt to revert back to entry-point GPU if necessary
        if (entrypoint_gpu != INVALID_GPU_ORDINAL)
        {
            if (error = CubDebug(cudaSetDevice(entrypoint_gpu))) return error;
        }

        return error;
    }


    /**
     * Frees a live allocation of GPU memory on the current GPU, returning it to the
     * allocator
     */
    cudaError_t DeviceFree(void* d_ptr)
    {
        GpuOrdinal current_gpu;
        cudaError_t error = cudaSuccess;

        do {
            if (error = CubDebug(cudaGetDevice(&current_gpu))) break;
            if (error = DeviceFree(d_ptr, current_gpu)) break;
        } while(0);

        return error;
    }


    /**
     * Frees all cached device allocations on all GPUs
     */
    cudaError_t FreeAllCached()
    {
        cudaError_t error                 = cudaSuccess;
        bool locked                     = false;
        GpuOrdinal entrypoint_gpu         = INVALID_GPU_ORDINAL;
        GpuOrdinal current_gpu            = INVALID_GPU_ORDINAL;

        // Lock
        if (!locked) {
            Lock(&spin_lock);
            locked = true;
        }

        while (!cached_blocks.empty())
        {
            // Get first block
            CachedBlocks::iterator begin = cached_blocks.begin();

            // Get entry-point GPU ordinal if necessary
            if (entrypoint_gpu == INVALID_GPU_ORDINAL)
            {
                if (error = CubDebug(cudaGetDevice(&entrypoint_gpu))) break;
            }

            // Set current GPU ordinal if necessary
            if (begin->gpu != current_gpu)
            {
                if (error = CubDebug(cudaSetDevice(begin->gpu))) break;
                current_gpu = begin->gpu;
            }

            // Free device memory
            if (error = CubDebug(cudaFree(begin->d_ptr))) break;

            // Reduce balance and erase entry
            cached_bytes[current_gpu] -= begin->bytes;
            cached_blocks.erase(begin);

            if (debug) printf("\tGPU %d freed %d bytes.  %d available blocks cached (%d bytes), %d live blocks outstanding.\n",
                current_gpu, begin->bytes, cached_blocks.size(), cached_bytes[current_gpu], live_blocks.size());
        }

        // Unlock
        if (locked) {
            Unlock(&spin_lock);
            locked = false;
        }

        // Attempt to revert back to entry-point GPU if necessary
        if (entrypoint_gpu != INVALID_GPU_ORDINAL)
        {
            if (error = CubDebug(cudaSetDevice(entrypoint_gpu))) return error;
        }

        return error;
    }

} default_allocator;        /// Default caching allocator instance



/******************************************************************************
 * CUB allocation operations
 ******************************************************************************/

// Pointer to device allocator
DeviceAllocator *allocator = &default_allocator;


/**
 * Sets the default CUB device allocator to the specified instance
 */
cudaError_t SetDeviceAllocator(DeviceAllocator *new_allocator)
{
#ifndef __CUDA_ARCH__
    allocator = new_allocator;
    return cudaSuccess;
#else
    // Custom allocators not available on device code
    return cudaErrorInvalidResourceHandle;
#endif
}


/**
 * Returns a pointer to the default CUB device allocator
 */
cudaError_t GetDeviceAllocator(DeviceAllocator** current_allocator)
{
#ifndef __CUDA_ARCH__
    *current_allocator = allocator;
    return cudaSuccess;
#else
    // Custom allocators not available on device code
    return cudaErrorInvalidResourceHandle;
#endif
}


/**
 *
 * Sets the limit on the number bytes this allocator is allowed to
 * cache per GPU.
 */
cudaError_t SetMaxCachedBytes(size_t max_cached_bytes)
{
#ifndef __CUDA_ARCH__
    if (allocator == NULL) return cudaErrorInvalidResourceHandle;
    return allocator->SetMaxCachedBytes(max_cached_bytes);
#else
    // Custom allocators not available on device code
    return cudaErrorInvalidResourceHandle;
#endif
}


/**
 * Frees all cached device allocations on all GPUs
 */
cudaError_t FreeAllCached()
{
#ifndef __CUDA_ARCH__
    if (allocator == NULL) return cudaErrorInvalidResourceHandle;
    return allocator->FreeAllCached();
#else
    // Custom allocators not available on device code
    return cudaErrorInvalidResourceHandle;
#endif
}


/**
 * Provides a suitable allocation of device memory for the given size
 * on the specified GPU
 */
__host__ __device__ __forceinline__ cudaError_t DeviceAllocate(void** d_ptr, size_t bytes, GpuOrdinal gpu)
{
#ifndef __CUDA_ARCH__

    // Use CUDA if no default allocator present
    if (allocator == NULL)
    {
        cudaError_t error = cudaSuccess;
        GpuOrdinal entrypoint_gpu = INVALID_GPU_ORDINAL;

        do
        {
            // Set to specified GPU
            if (error = CubDebug(cudaGetDevice(&entrypoint_gpu))) break;
            if (error = CubDebug(cudaSetDevice(gpu))) break;

            // Allocate device memory
            if (error = CubDebug(cudaMalloc(&d_ptr, bytes))) break;

        } while (0);

        // Attempt to revert back to entry-point GPU if necessary
        if (entrypoint_gpu != INVALID_GPU_ORDINAL)
        {
            error = CubDebug(cudaSetDevice(entrypoint_gpu));
        }

        return error;
    }

    return allocator->DeviceAllocate(d_ptr, bytes, gpu);
#else
    // Cannot allocate on GPUs from within device code
    return cudaErrorInvalidDevice;
#endif
}


/**
 * Provides a suitable allocation of device memory for the given size
 * on the current GPU
 */
__host__ __device__ __forceinline__ cudaError_t DeviceAllocate(void** d_ptr, size_t bytes)
{
#ifndef __CUDA_ARCH__
    // Use CUDA if no default allocator present
    if (allocator == NULL) return CubDebug(cudaMalloc(&d_ptr, bytes));
    return allocator->DeviceAllocate(d_ptr, bytes);
#elif __CUDA_ARCH >= 350
    // Use CUDA (custom allocators are unavailable from within device code)
    return CubDebug(cudaMalloc(&d_ptr, bytes));
#else
    // Cannot allocate on GPUs from within device code
    return cudaErrorInvalidDevice;
#endif
}


/**
 * Frees a live allocation of GPU memory on the specified GPU, returning it to
 * the allocator
 */
__host__ __device__ __forceinline__ cudaError_t DeviceFree(void* d_ptr, GpuOrdinal gpu)
{
#ifndef __CUDA_ARCH__

    // Use CUDA if no default allocator present
    if (allocator == NULL)
    {
        cudaError_t error = cudaSuccess;
        GpuOrdinal entrypoint_gpu = INVALID_GPU_ORDINAL;

        do
        {
            // Set to specified GPU
            if (error = CubDebug(cudaGetDevice(&entrypoint_gpu))) break;
            if (error = CubDebug(cudaSetDevice(gpu))) break;

            // Free device memory
            if (error = CubDebug(cudaFree(d_ptr))) break;

        } while (0);

        // Attempt to revert back to entry-point GPU if necessary
        if (entrypoint_gpu != INVALID_GPU_ORDINAL)
        {
            error = CubDebug(cudaSetDevice(entrypoint_gpu));
        }

        return error;
    }

    // Use default allocator
    return allocator->DeviceFree(d_ptr, gpu);

#else
    // Cannot allocate on GPUs from within device code
    return cudaErrorInvalidDevice;
#endif
}


/**
 * Frees a live allocation of GPU memory on the current GPU, returning it to the
 * allocator
 */
__host__ __device__ __forceinline__ cudaError_t DeviceFree(void* d_ptr)
{
#ifndef __CUDA_ARCH__
    // Use CUDA if no default allocator present
    if (allocator == NULL) return CubDebug(cudaFree(d_ptr));
    return allocator->DeviceFree(d_ptr);
#elif __CUDA_ARCH >= 350
    // Use CUDA (custom allocators are unavailable from within device code)
    return CubDebug(cudaFree(d_ptr));
#else
    // Cannot allocate on GPUs from within device code
    return cudaErrorInvalidDevice;
#endif
}


} // anonymous namespace
} // namespace cub
CUB_NS_POSTFIX
