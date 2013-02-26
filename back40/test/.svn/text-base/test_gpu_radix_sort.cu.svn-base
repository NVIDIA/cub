/******************************************************************************
 *
 * Copyright (c) 2010-2012, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2012, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License. 
 * 
 ******************************************************************************/


/******************************************************************************
 * Simple test driver program for radix sort.
 ******************************************************************************/

// Ensure printing of CUDA runtime errors to console
#define CUB_STDERR

#include <stdio.h> 
#include <algorithm>
#include <iostream>

// Sorting includes
#include <back40/gpu_radix_sort.cuh>

// Test utils
#include "cub/cub.cuh"
#include "cub/test/test_util.h"


template <typename T, typename S>
void Assign(T &t, S &s)
{
    t = s;
}

template <typename S>
void Assign(cub::NullType, S &s) {}

template <typename T>
int CastInt(T &t)
{
    return (int) t;
}

int CastInt(cub::NullType)
{
    return 0;
}


/******************************************************************************
 * Main
 ******************************************************************************/

int main(int argc, char** argv)
{
//    typedef unsigned long long        KeyType;
//    typedef float                    KeyType;
//    typedef char                    KeyType;
//    typedef int                        KeyType;
    typedef unsigned int             KeyType;
//    typedef unsigned short             KeyType;
    typedef cub::NullType             ValueType;
//    typedef unsigned long long         ValueType;
//    typedef unsigned int            ValueType;

    const int       START_BIT            = 0;
    const int       KEY_BITS             = sizeof(KeyType) * 8;
    const bool      KEYS_ONLY            = cub::Equals<ValueType, cub::NullType>::VALUE;
    int             num_elements         = 1024 * 1024 * 8;
    unsigned int    max_ctas             = 0;
    int             iterations           = 0;
    int             entropy_reduction    = 0;

    // Initialize command line
    CommandLineArgs args(argc, argv);
    CubDebugExit(args.DeviceInit());

    // Usage/help
    if (args.CheckCmdLineFlag("help") || args.CheckCmdLineFlag("h"))
    {
        printf("\n%s [--schmoo] [--device=<device index>] [--v] [--n=<elements>] "
                "[--max-ctas=<max-thread-blocks>] [--i=<iterations>] "
                "[--zeros | --regular] [--entropy-reduction=<random &'ing rounds>\n",
                argv[0]);
        return 0;
    }

    // Parse commandline args
    bool verbose = args.CheckCmdLineFlag("v");
    bool zeros = args.CheckCmdLineFlag("zeros");
    bool regular = args.CheckCmdLineFlag("regular");
    bool schmoo = args.CheckCmdLineFlag("schmoo");
    args.GetCmdLineArgument("n", num_elements);
    args.GetCmdLineArgument("i", iterations);
    args.GetCmdLineArgument("max-ctas", max_ctas);
    args.GetCmdLineArgument("entropy-reduction", entropy_reduction);

    // Print header
    printf("Initializing problem instance ");
    if (zeros) printf("(zeros)\n");
    else if (regular) printf("(%d-bit mod-%llu)\n", KEY_BITS, 1ull << KEY_BITS);
    else printf("(%d-bit random)\n", KEY_BITS);
    fflush(stdout);

    // Allocate and initialize host problem data and host reference solution.
    // Only use RADIX_BITS effective bits (remaining high order bits
    // are left zero): we only want to perform one sorting pass

    KeyType         *h_keys             = new KeyType[num_elements];
    KeyType         *h_reference_keys   = new KeyType[num_elements];
    ValueType       *h_values           = NULL;
    if (!KEYS_ONLY)
    {
        h_values = new ValueType[num_elements];
    }

    if (verbose) printf("Original:\n");
    for (int i = 0; i < num_elements; ++i)
    {
        if (regular) {
            h_keys[i] = i & ((1ull << KEY_BITS) - 1);
        } else if (zeros) {
            h_keys[i] = 0;
        } else {
            RandomBits(h_keys[i], entropy_reduction, START_BIT, START_BIT + KEY_BITS);
        }
        h_reference_keys[i] = h_keys[i];

        if (!KEYS_ONLY) {
            Assign(h_values[i], i);
        }

        if (verbose) {
            std::cout << h_keys[i] << ", ";
        }
    }
    if (verbose) printf("\n");
    printf("Done.\n"); fflush(stdout);

    // Compute reference solution
    printf("Computing reference solution on CPU..."); fflush(stdout);
    std::sort(h_reference_keys, h_reference_keys + num_elements);
    printf(" Done.\n"); fflush(stdout);

    // Allocate device data.
    KeyType         *d_keys = NULL;
    ValueType       *d_values = NULL;
    CubDebugExit(cudaMalloc((void**) &d_keys, sizeof(KeyType) * num_elements));
    if (!KEYS_ONLY)
    {
        CubDebugExit(cudaMalloc((void**) &d_values, sizeof(ValueType) * num_elements));
    }

    // Resize max cached bytes to 512MB
    cub::SetMaxCachedBytes(512 * 1024 * 1024);
//  cub::allocator->debug = true;

    //
    // Perform one sorting pass for correctness/warmup
    //

    // Copy problem to GPU
    CubDebugExit(cudaMemcpy(d_keys, h_keys, sizeof(KeyType) * num_elements, cudaMemcpyHostToDevice));
    if (!KEYS_ONLY)
    {
        CubDebugExit(cudaMemcpy(d_values, h_values, sizeof(ValueType) * num_elements, cudaMemcpyHostToDevice));
    }

    // Sort
    CubDebugExit(back40::GpuRadixSort(
        d_keys,
        d_values,
        num_elements,
        START_BIT,
        KEY_BITS,
        0,
        max_ctas,
        true));

    // Check key results
    CompareDeviceResults(h_reference_keys, d_keys, num_elements, true, verbose);
    printf("\n");

    // Check value results
    if (!KEYS_ONLY)
    {
        CubDebugExit(cudaMemcpy(h_values, d_values, sizeof(ValueType) * num_elements, cudaMemcpyDeviceToHost));

        printf("\n\nValues: ");
        if (verbose) {
            for (int i = 0; i < num_elements; ++i) {
                std::cout << h_values[i] << ", ";
            }
            printf("\n\n");
        }

        bool correct = true;
        for (int i = 0; i < num_elements; ++i) {
            if (h_keys[CastInt(h_values[i])] != h_reference_keys[i])
            {
                std::cout << "Incorrect: [" << i << "]: " << h_keys[CastInt(h_values[i])] << " != " << h_reference_keys[i] << std::endl << std::endl;
                correct = false;
                break;
            }
        }
        if (correct) {
            printf("Correct\n\n");
        }
    }

    // Flush any stdio from the kernel
    CubDebugExit(cudaThreadSynchronize());
    fflush(stdout);

/*
    // Print column headers
    if (schmoo) {
        printf("Iteration, Elements, Elapsed (ms), Throughput (MKeys/s)\n");
    }


    //
    // Iterate for timing results
    //

    GpuTimer gpu_timer;
    double max_exponent         = log2(double(num_elements)) - 5.0;
    unsigned int max_int         = (unsigned int) -1;
    float elapsed                 = 0;

    for (int i = 0; i < iterations; i++) {

        // Copy problem to GPU
        CubDebugExit(cudaMemcpy(d_keys, h_keys, sizeof(KeyType) * num_elements, cudaMemcpyHostToDevice));
        if (!KEYS_ONLY)
        {
            CubDebugExit(cudaMemcpy(d_values, h_values, sizeof(ValueType) * num_elements, cudaMemcpyHostToDevice));
        }

        if (schmoo)
        {
            // Sample a problem size
            unsigned int sample;
            RandomBits(sample);
            double scale = double(sample) / max_int;
            int elements = (i < iterations / 2) ?
                pow(2.0, (max_exponent * scale) + 5.0) :            // log bias
                elements = scale * num_elements;                    // uniform bias

            gpu_timer.Start();

            printf("%d, %d", i, elements);
            fflush(stdout);

            // Sort
            CubDebugExit(back40::GpuRadixSort(
                d_keys,
                d_values,
                elements,
                START_BIT,
                KEY_BITS,
                0,
                max_ctas,
                false));

            gpu_timer.Stop();

            float millis = gpu_timer.ElapsedMillis();
            printf(", %.3f, %.2f\n",
                millis,
                float(elements) / millis / 1000.f);
            fflush(stdout);
        }
        else
        {
            // Regular iteration
            gpu_timer.Start();

            // Sort
            CubDebugExit(back40::GpuRadixSort(
                d_keys,
                d_values,
                num_elements,
                START_BIT,
                KEY_BITS,
                0,
                max_ctas,
                false));

            gpu_timer.Stop();

            elapsed += gpu_timer.ElapsedMillis();
        }
    }

    // Display output
    if ((!schmoo) && (iterations > 0))
    {
        float avg_elapsed = elapsed / float(iterations);
        printf("Elapsed millis: %f, avg elapsed: %f, throughput: %.2f Mkeys/s\n",
            elapsed,
            avg_elapsed,
            float(num_elements) / avg_elapsed / 1000.f);
    }
*/

    // Cleanup device storage
    if (d_keys) CubDebugExit(cudaFree(d_keys));
    if (d_values) CubDebugExit(cudaFree(d_values));

    // Cleanup other
    delete[] h_keys;
    delete[] h_reference_keys;
    delete[] h_values;

    return 0;
}

