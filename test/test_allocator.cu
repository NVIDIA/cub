/******************************************************************************
 * Copyright (c) 2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2014, NVIDIA CORPORATION.  All rights reserved.
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
 * Test evaluation for caching allocator of device memory
 ******************************************************************************/

// Ensure printing of CUDA runtime errors to console
#define CUB_STDERR

#include <stdio.h>

#include <cub/util_allocator.cuh>
#include "test_util.h"

using namespace cub;


//---------------------------------------------------------------------
// Main
//---------------------------------------------------------------------

/**
 * Main
 */
int main(int argc, char** argv)
{
    // Initialize command line
    CommandLineArgs args(argc, argv);

    // Print usage
    if (args.CheckCmdLineFlag("help"))
    {
        printf("%s "
            "[--device=<device-id>]"
            "\n", argv[0]);
        exit(0);
    }

#if (CUB_PTX_VERSION == 0)

    // Initialize device
    CubDebugExit(args.DeviceInit());

    // Get number of GPUs and current GPU
    int num_gpus, initial_gpu;
    if (CubDebug(cudaGetDeviceCount(&num_gpus))) exit(1);
    if (CubDebug(cudaGetDevice(&initial_gpu))) exit(1);

    // Create default allocator (caches up to 6MB in device allocations per GPU)
    CachingDeviceAllocator allocator;
    allocator.debug = true;

    printf("Running single-gpu tests...\n"); fflush(stdout);

    //
    // Test1
    //

    // Allocate 5 bytes on the current gpu
    char *d_5B;
    allocator.DeviceAllocate((void **) &d_5B, 5);

    // Check that that we have zero bytes allocated on the initial GPU
    AssertEquals(allocator.cached_bytes[initial_gpu], 0);

    // Check that that we have 1 live block on the initial GPU
    AssertEquals(allocator.live_blocks.size(), 1);

    //
    // Test2
    //

    // Allocate 4096 bytes on the current gpu
    char *d_4096B;
    allocator.DeviceAllocate((void **) &d_4096B, 4096);

    // Check that that we have 2 live blocks on the initial GPU
    AssertEquals(allocator.live_blocks.size(), 2);

    //
    // Test3
    //

    // DeviceFree d_5B
    allocator.DeviceFree(d_5B);

    // Check that that we have min_bin_bytes free bytes cached on the initial gpu
    AssertEquals(allocator.cached_bytes[initial_gpu], allocator.min_bin_bytes);

    // Check that that we have 1 live block on the initial GPU
    AssertEquals(allocator.live_blocks.size(), 1);

    // Check that that we have 1 cached block on the initial GPU
    AssertEquals(allocator.cached_blocks.size(), 1);

    //
    // Test4
    //

    // DeviceFree d_4096B
    allocator.DeviceFree(d_4096B);

    // Check that that we have the 4096 + min_bin free bytes cached on the initial gpu
    AssertEquals(allocator.cached_bytes[initial_gpu], allocator.min_bin_bytes + 4096);

    // Check that that we have 0 live block on the initial GPU
    AssertEquals(allocator.live_blocks.size(), 0);

    // Check that that we have 2 cached block on the initial GPU
    AssertEquals(allocator.cached_blocks.size(), 2);

    //
    // Test5
    //

    // Allocate 768 bytes on the current gpu
    char *d_768B;
    allocator.DeviceAllocate((void **) &d_768B, 768);

    // Check that that we have the min_bin free bytes cached on the initial gpu (4096 was reused)
    AssertEquals(allocator.cached_bytes[initial_gpu], allocator.min_bin_bytes);

    // Check that that we have 1 live block on the initial GPU
    AssertEquals(allocator.live_blocks.size(), 1);

    // Check that that we have 1 cached block on the initial GPU
    AssertEquals(allocator.cached_blocks.size(), 1);

    //
    // Test6
    //

    // Allocate max_cached_bytes on the current gpu
    char *d_max_cached;
    allocator.DeviceAllocate((void **) &d_max_cached, allocator.max_cached_bytes);

    // DeviceFree d_max_cached
    allocator.DeviceFree(d_max_cached);

    // Check that that we have the min_bin free bytes cached on the initial gpu (max cached was not returned because we went over)
    AssertEquals(allocator.cached_bytes[initial_gpu], allocator.min_bin_bytes);

    // Check that that we have 1 live block on the initial GPU
    AssertEquals(allocator.live_blocks.size(), 1);

    // Check that that we still have 1 cached block on the initial GPU
    AssertEquals(allocator.cached_blocks.size(), 1);

    //
    // Test7
    //

    // Free all cached blocks on all GPUs
    allocator.FreeAllCached();

    // Check that that we have 0 bytes cached on the initial GPU
    AssertEquals(allocator.cached_bytes[initial_gpu], 0);

    // Check that that we have 0 cached blocks across all GPUs
    AssertEquals(allocator.cached_blocks.size(), 0);

    // Check that that still we have 1 live block across all GPUs
    AssertEquals(allocator.live_blocks.size(), 1);

    //
    // Test8
    //

    // Allocate max cached bytes + 1 on the current gpu
    char *d_max_cached_plus;
    allocator.DeviceAllocate((void **) &d_max_cached_plus, allocator.max_cached_bytes + 1);

    // DeviceFree max cached bytes
    allocator.DeviceFree(d_max_cached_plus);

    // DeviceFree d_768B
    allocator.DeviceFree(d_768B);

    unsigned int power;
    size_t rounded_bytes;
    allocator.NearestPowerOf(power, rounded_bytes, allocator.bin_growth, 768);

    // Check that that we have 4096 free bytes cached on the initial gpu
    AssertEquals(allocator.cached_bytes[initial_gpu], rounded_bytes);

    // Check that that we have 1 cached blocks across all GPUs
    AssertEquals(allocator.cached_blocks.size(), 1);

    // Check that that still we have 0 live block across all GPUs
    AssertEquals(allocator.live_blocks.size(), 0);

#ifndef CUB_CDP
    // BUG: find out why these tests fail when one GPU is CDP compliant and the other is not

    if (num_gpus > 1)
    {
        printf("Running multi-gpu tests...\n"); fflush(stdout);

        //
        // Test9
        //

        // Allocate 768 bytes on the next gpu
        int next_gpu = (initial_gpu + 1) % num_gpus;
        char *d_768B_2;
        allocator.DeviceAllocate((void **) &d_768B_2, 768, next_gpu);

        // DeviceFree d_768B on the next gpu
        allocator.DeviceFree(d_768B_2, next_gpu);

        // Check that that we have 4096 free bytes cached on the initial gpu
        AssertEquals(allocator.cached_bytes[initial_gpu], rounded_bytes);

        // Check that that we have 4096 free bytes cached on the second gpu
        AssertEquals(allocator.cached_bytes[next_gpu], rounded_bytes);

        // Check that that we have 2 cached blocks across all GPUs
        AssertEquals(allocator.cached_blocks.size(), 2);

        // Check that that still we have 0 live block across all GPUs
        AssertEquals(allocator.live_blocks.size(), 0);
    }
#endif  // CUB_CDP
#endif

    printf("Success\n");
    return 0;
}

