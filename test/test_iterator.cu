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
 * Test of iterator utilities
 ******************************************************************************/

// Ensure printing of CUDA runtime errors to console
#define CUB_STDERR

#include <iterator>
#include <stdio.h>

#include <cub/util_type.cuh>
#include <cub/util_allocator.cuh>
#include <cub/util_iterator.cuh>

#include "test_util.h"

using namespace cub;


//---------------------------------------------------------------------
// Globals, constants and typedefs
//---------------------------------------------------------------------

bool                    g_verbose = false;
CachingDeviceAllocator  g_allocator(true);

template <typename T>
struct TransformOp
{
    // Increment transform
    __host__ __device__ __forceinline__ T operator()(const T input)
    {
        T addend;
        InitValue(SEQ_INC, addend, 1);
        return input + addend;
    }
};


//---------------------------------------------------------------------
// Test kernels
//---------------------------------------------------------------------

/**
 * Test random access input iterator
 */
template <
    typename InputIterator,
    typename T>
__global__ void Kernel(
    InputIterator     d_in,
    T                   *d_out,
    InputIterator     *d_itrs)
{
    d_out[0] = *d_in;               // Value at offset 0
    d_out[1] = d_in[100];           // Value at offset 100
    d_out[2] = *(d_in + 1000);      // Value at offset 1000
    d_out[3] = *(d_in + 10000);   // Value at offset 10000

    d_in++;
    d_out[4] = d_in[0];             // Value at offset 1

    d_in += 20;
    d_out[5] = d_in[0];             // Value at offset 21
    d_itrs[0] = d_in;               // Iterator at offset 21

    d_in -= 10;
    d_out[6] = d_in[0];             // Value at offset 11;

    d_in -= 11;
    d_out[7] = d_in[0];             // Value at offset 0
    d_itrs[1] = d_in;               // Iterator at offset 0
}



//---------------------------------------------------------------------
// Host testing subroutines
//---------------------------------------------------------------------


/**
 * Run iterator test on device
 */
template <
    typename    InputIterator,
    typename    T,
    int         TEST_VALUES>
void Test(
    InputIterator     d_in,
    T                   (&h_reference)[TEST_VALUES])
{
    // Allocate device arrays
    T                   *d_out = NULL;
    InputIterator     *d_itrs = NULL;
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_out, sizeof(T) * TEST_VALUES));
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_itrs, sizeof(InputIterator) * 2));

    int compare;

    // Run unguarded kernel
    Kernel<<<1, 1>>>(d_in, d_out, d_itrs);
    CubDebugExit(cudaDeviceSynchronize());

    // Check results
    compare = CompareDeviceResults(h_reference, d_out, TEST_VALUES, g_verbose, g_verbose);
    printf("\tValues: %s\n", (compare) ? "FAIL" : "PASS");
    AssertEquals(0, compare);

    // Check iterator at offset 21
    InputIterator h_itr = d_in + 21;
    compare = CompareDeviceResults(&h_itr, d_itrs, 1, g_verbose, g_verbose);
    printf("\tIterators: %s\n", (compare) ? "FAIL" : "PASS");
    AssertEquals(0, compare);

    // Check iterator at offset 0
    compare = CompareDeviceResults(&d_in, d_itrs + 1, 1, g_verbose, g_verbose);
    printf("\tIterators: %s\n", (compare) ? "FAIL" : "PASS");
    AssertEquals(0, compare);

    // Cleanup
    if (d_out) CubDebugExit(g_allocator.DeviceFree(d_out));
    if (d_itrs) CubDebugExit(g_allocator.DeviceFree(d_itrs));
}


/**
 * Test constant iterator
 */
template <typename T>
void TestConstant(T base, char *type_string)
{
    printf("\nTesting constant iterator on type %s (base: %d)\n", type_string, base); fflush(stdout);

    T h_reference[8] = {base, base, base, base, base, base, base, base};

    Test(ConstantInputIterator<T>(base), h_reference);
}


/**
 * Test counting iterator
 */
template <typename T>
void TestCounting(T base, char *type_string)
{
    printf("\nTesting counting iterator on type %s (base: %d) \n", type_string, base); fflush(stdout);

    // Initialize reference data
    T h_reference[8];
    h_reference[0] = base + 0;          // Value at offset 0
    h_reference[1] = base + 100;        // Value at offset 100
    h_reference[2] = base + 1000;       // Value at offset 1000
    h_reference[3] = base + 10000;      // Value at offset 10000
    h_reference[4] = base + 1;          // Value at offset 1
    h_reference[5] = base + 21;         // Value at offset 21
    h_reference[6] = base + 11;         // Value at offset 11
    h_reference[7] = base + 0;          // Value at offset 0;

    Test(CountingInputIterator<T>(base), h_reference);
}


/**
 * Test modified iterator
 */
template <typename T>
void TestModified(char *type_string)
{
    printf("\nTesting cache-modified iterator on type %s\n", type_string); fflush(stdout);

    const unsigned int TEST_VALUES = 11000;

    T *h_data = new T[TEST_VALUES];
    for (int i = 0; i < TEST_VALUES; ++i)
    {
        RandomBits(h_data[i]);
    }

    // Allocate device arrays
    T *d_data = NULL;
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_data, sizeof(T) * TEST_VALUES));
    CubDebugExit(cudaMemcpy(d_data, h_data, sizeof(T) * TEST_VALUES, cudaMemcpyHostToDevice));

    // Initialize reference data
    T h_reference[8];
    h_reference[0] = h_data[0];          // Value at offset 0
    h_reference[1] = h_data[100];        // Value at offset 100
    h_reference[2] = h_data[1000];       // Value at offset 1000
    h_reference[3] = h_data[10000];      // Value at offset 10000
    h_reference[4] = h_data[1];          // Value at offset 1
    h_reference[5] = h_data[21];         // Value at offset 21
    h_reference[6] = h_data[11];         // Value at offset 11
    h_reference[7] = h_data[0];          // Value at offset 0;

    Test(CacheModifiedInputIterator<LOAD_DEFAULT, T>(d_data), h_reference);
    Test(CacheModifiedInputIterator<LOAD_CA, T>(d_data), h_reference);
    Test(CacheModifiedInputIterator<LOAD_CG, T>(d_data), h_reference);
    Test(CacheModifiedInputIterator<LOAD_CS, T>(d_data), h_reference);
    Test(CacheModifiedInputIterator<LOAD_CV, T>(d_data), h_reference);
    Test(CacheModifiedInputIterator<LOAD_LDG, T>(d_data), h_reference);
    Test(CacheModifiedInputIterator<LOAD_VOLATILE, T>(d_data), h_reference);

    // Cleanup
    if (h_data) delete[] h_data;
    if (d_data) CubDebugExit(g_allocator.DeviceFree(d_data));
}


/**
 * Test transform iterator
 */
template <typename T>
void TestTransform(char *type_string)
{
    printf("\nTesting transform iterator on type %s\n", type_string); fflush(stdout);

    const unsigned int TEST_VALUES = 11000;

    T *h_data = new T[TEST_VALUES];
    for (int i = 0; i < TEST_VALUES; ++i)
    {
        RandomBits(h_data[i]);
    }

    // Allocate device arrays
    T *d_data = NULL;
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_data, sizeof(T) * TEST_VALUES));
    CubDebugExit(cudaMemcpy(d_data, h_data, sizeof(T) * TEST_VALUES, cudaMemcpyHostToDevice));

    TransformOp<T> op;

    // Initialize reference data
    T h_reference[8];
    h_reference[0] = op(h_data[0]);          // Value at offset 0
    h_reference[1] = op(h_data[100]);        // Value at offset 100
    h_reference[2] = op(h_data[1000]);       // Value at offset 1000
    h_reference[3] = op(h_data[10000]);      // Value at offset 10000
    h_reference[4] = op(h_data[1]);          // Value at offset 1
    h_reference[5] = op(h_data[21]);         // Value at offset 21
    h_reference[6] = op(h_data[11]);         // Value at offset 11
    h_reference[7] = op(h_data[0]);          // Value at offset 0;

    Test(TransformInputIterator<T, TransformOp<T>, T*>(d_data, op), h_reference);

    // Cleanup
    if (h_data) delete[] h_data;
    if (d_data) CubDebugExit(g_allocator.DeviceFree(d_data));
}


/**
 * Test texture iterator
 */
template <typename T>
void TestTexture(char *type_string)
{
    printf("\nTesting texture iterator on type %s\n", type_string); fflush(stdout);

    const unsigned int TEST_VALUES          = 11000;
    const unsigned int DUMMY_OFFSET         = 500;
    const unsigned int DUMMY_TEST_VALUES    = TEST_VALUES - DUMMY_OFFSET;

    T *h_data = new T[TEST_VALUES];
    for (int i = 0; i < TEST_VALUES; ++i)
    {
        RandomBits(h_data[i]);
    }

    // Allocate device arrays
    T *d_data   = NULL;
    T *d_dummy  = NULL;
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_data, sizeof(T) * TEST_VALUES));
    CubDebugExit(cudaMemcpy(d_data, h_data, sizeof(T) * TEST_VALUES, cudaMemcpyHostToDevice));

    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_dummy, sizeof(T) * DUMMY_TEST_VALUES));
    CubDebugExit(cudaMemcpy(d_dummy, h_data + DUMMY_OFFSET, sizeof(T) * DUMMY_TEST_VALUES, cudaMemcpyHostToDevice));

    // Initialize reference data
    T h_reference[8];
    h_reference[0] = h_data[0];          // Value at offset 0
    h_reference[1] = h_data[100];        // Value at offset 100
    h_reference[2] = h_data[1000];       // Value at offset 1000
    h_reference[3] = h_data[10000];      // Value at offset 10000
    h_reference[4] = h_data[1];          // Value at offset 1
    h_reference[5] = h_data[21];         // Value at offset 21
    h_reference[6] = h_data[11];         // Value at offset 11
    h_reference[7] = h_data[0];          // Value at offset 0;

    // Create and bind ref-based test iterator
    TexRefInputIterator<T, __LINE__> d_ref_itr;
    CubDebugExit(d_ref_itr.BindTexture(d_data, sizeof(T) * TEST_VALUES));

    // Create and bind dummy iterator of same type to check with interferance
    TexRefInputIterator<T, __LINE__> d_ref_itr2;
    CubDebugExit(d_ref_itr2.BindTexture(d_dummy, sizeof(T) * DUMMY_TEST_VALUES));

    Test(d_ref_itr, h_reference);

    // Cleanup
    CubDebugExit(d_ref_itr.UnbindTexture());
    CubDebugExit(d_ref_itr2.UnbindTexture());

#ifdef CUB_CDP

    // Create and bind obj-based test iterator
    TexObjInputIterator<T> d_obj_itr;
    CubDebugExit(d_obj_itr.BindTexture(d_data, sizeof(T) * TEST_VALUES));

    Test(d_obj_itr, h_reference);

    // Cleanup
    CubDebugExit(d_obj_itr.UnbindTexture());

#endif


    // Cleanup
    if (h_data) delete[] h_data;
    if (d_data) CubDebugExit(g_allocator.DeviceFree(d_data));
    if (d_dummy) CubDebugExit(g_allocator.DeviceFree(d_dummy));
}


/**
 * Test texture transform iterator
 */
template <typename T>
void TestTexTransform(char *type_string)
{
    printf("\nTesting tex-transform iterator on type %s\n", type_string); fflush(stdout);

    const unsigned int TEST_VALUES = 11000;

    T *h_data = new T[TEST_VALUES];
    for (int i = 0; i < TEST_VALUES; ++i)
    {
        RandomBits(h_data[i]);
    }

    // Allocate device arrays
    T *d_data = NULL;
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_data, sizeof(T) * TEST_VALUES));
    CubDebugExit(cudaMemcpy(d_data, h_data, sizeof(T) * TEST_VALUES, cudaMemcpyHostToDevice));

    TransformOp<T> op;

    // Initialize reference data
    T h_reference[8];
    h_reference[0] = op(h_data[0]);          // Value at offset 0
    h_reference[1] = op(h_data[100]);        // Value at offset 100
    h_reference[2] = op(h_data[1000]);       // Value at offset 1000
    h_reference[3] = op(h_data[10000]);      // Value at offset 10000
    h_reference[4] = op(h_data[1]);          // Value at offset 1
    h_reference[5] = op(h_data[21]);         // Value at offset 21
    h_reference[6] = op(h_data[11]);         // Value at offset 11
    h_reference[7] = op(h_data[0]);          // Value at offset 0;

    // Create and bind texture iterator
    typedef TexRefInputIterator<T, __LINE__> TextureIterator;

    TextureIterator d_tex_itr;
    CubDebugExit(d_tex_itr.BindTexture(d_data, sizeof(T) * TEST_VALUES));

    // Create transform iterator
    TransformInputIterator<T, TransformOp<T>, TextureIterator> xform_itr(d_tex_itr, op);

    Test(xform_itr, h_reference);

    // Cleanup
    CubDebugExit(d_tex_itr.UnbindTexture());
    if (h_data) delete[] h_data;
    if (d_data) CubDebugExit(g_allocator.DeviceFree(d_data));
}



/**
 * Run non-integer tests
 */
template <typename T>
void TestInteger(Int2Type<false> is_integer, char *type_string)
{
    TestModified<T>(type_string);
    TestTransform<T>(type_string);
    TestTexture<T>(type_string);
    TestTexTransform<T>(type_string);
}

/**
 * Run integer tests
 */
template <typename T>
void TestInteger(Int2Type<true> is_integer, char *type_string)
{
    TestConstant<T>(0, type_string);
    TestConstant<T>(99, type_string);

    TestCounting<T>(0, type_string);
    TestCounting<T>(99, type_string);

    // Run non-integer tests
    TestInteger<T>(Int2Type<false>(), type_string);
}

/**
 * Run tests
 */
template <typename T>
void Test(char *type_string)
{
    enum {
        IS_INTEGER = (Traits<T>::CATEGORY == SIGNED_INTEGER) || (Traits<T>::CATEGORY == UNSIGNED_INTEGER)
    };
    TestInteger<T>(Int2Type<IS_INTEGER>(), type_string);
}


/**
 * Main
 */
int main(int argc, char** argv)
{
    // Initialize command line
    CommandLineArgs args(argc, argv);
    g_verbose = args.CheckCmdLineFlag("v");

    // Print usage
    if (args.CheckCmdLineFlag("help"))
    {
        printf("%s "
            "[--device=<device-id>] "
            "[--v] "
            "\n", argv[0]);
        exit(0);
    }

    // Initialize device
    CubDebugExit(args.DeviceInit());

    // Evaluate different data types
    Test<char>(CUB_TYPE_STRING(char));
    Test<short>(CUB_TYPE_STRING(short));
    Test<long>(CUB_TYPE_STRING(long));
    Test<long long>(CUB_TYPE_STRING(long long));
    Test<float>(CUB_TYPE_STRING(float));
    Test<double>(CUB_TYPE_STRING(double));

    Test<char2>(CUB_TYPE_STRING(char2));
    Test<short2>(CUB_TYPE_STRING(short2));
    Test<int2>(CUB_TYPE_STRING(int2));
    Test<long2>(CUB_TYPE_STRING(long2));
    Test<longlong2>(CUB_TYPE_STRING(longlong2));
    Test<float2>(CUB_TYPE_STRING(float2));
    Test<double2>(CUB_TYPE_STRING(double2));

    Test<char3>(CUB_TYPE_STRING(char3));
    Test<short3>(CUB_TYPE_STRING(short3));
    Test<int3>(CUB_TYPE_STRING(int3));
    Test<long3>(CUB_TYPE_STRING(long3));
    Test<longlong3>(CUB_TYPE_STRING(longlong3));
    Test<float3>(CUB_TYPE_STRING(float3));
    Test<double3>(CUB_TYPE_STRING(double3));

    Test<char4>(CUB_TYPE_STRING(char4));
    Test<short4>(CUB_TYPE_STRING(short4));
    Test<int4>(CUB_TYPE_STRING(int4));
    Test<long4>(CUB_TYPE_STRING(long4));
    Test<longlong4>(CUB_TYPE_STRING(longlong4));
    Test<float4>(CUB_TYPE_STRING(float4));
    Test<double4>(CUB_TYPE_STRING(double4));

    Test<TestFoo>(CUB_TYPE_STRING(TestFoo));
    Test<TestBar>(CUB_TYPE_STRING(TestBar));

    printf("\nTest complete\n"); fflush(stdout);

    return 0;
}



