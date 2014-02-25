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
 * Test of iterator utilities
 ******************************************************************************/

// Ensure printing of CUDA runtime errors to console
#define CUB_STDERR

#include <iterator>
#include <stdio.h>

#include <cub/util_type.cuh>
#include <cub/util_allocator.cuh>

#include <cub/iterator/arg_index_input_iterator.cuh>
#include <cub/iterator/cache_modified_input_iterator.cuh>
#include <cub/iterator/cache_modified_output_iterator.cuh>
#include <cub/iterator/constant_input_iterator.cuh>
#include <cub/iterator/counting_input_iterator.cuh>
#include <cub/iterator/tex_obj_input_iterator.cuh>
#include <cub/iterator/tex_ref_input_iterator.cuh>
#include <cub/iterator/transform_input_iterator.cuh>

#include <thrust/device_ptr.h>
#include <thrust/copy.h>

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
    __host__ __device__ __forceinline__ T operator()(T input) const
    {
        T addend;
        InitValue(INTEGER_SEED, addend, 1);
        return input + addend;
    }
};

struct SelectOp
{
    template <typename T>
    __host__ __device__ __forceinline__ bool operator()(T input)
    {
        return true;;
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
    T                 *d_out,
    InputIterator     *d_itrs)
{
    d_out[0] = *d_in;               // Value at offset 0
    d_out[1] = d_in[100];           // Value at offset 100
    d_out[2] = *(d_in + 1000);      // Value at offset 1000
    d_out[3] = *(d_in + 10000);     // Value at offset 10000

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
    typename        InputIterator,
    typename        T,
    int             TEST_VALUES>
void Test(
    InputIterator   d_in,
    T               (&h_reference)[TEST_VALUES])
{
    // Allocate device arrays
    T                 *d_out    = NULL;
    InputIterator     *d_itrs   = NULL;
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_out,     sizeof(T) * TEST_VALUES));
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_itrs,    sizeof(InputIterator) * 2));

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
    printf("\nTesting constant iterator on type %s (base: %d)\n", type_string, int(base)); fflush(stdout);

    //
    // Test iterator manipulation in kernel
    //

    T h_reference[8] = {base, base, base, base, base, base, base, base};
    ConstantInputIterator<T> d_itr(base);
    Test(d_itr, h_reference);

#if (THRUST_VERSION >= 100700)  // Thrust 1.7 or newer

    //
    // Test with thrust::copy_if()
    //

    int copy_items  = 100;
    T   *h_copy     = new T[copy_items];
    T   *d_copy     = NULL;

    for (int i = 0; i < copy_items; ++i)
        h_copy[i] = d_itr[i];

    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_copy, sizeof(T) * copy_items));
    thrust::device_ptr<T> d_copy_wrapper(d_copy);

    thrust::copy_if(d_itr, d_itr + copy_items, d_copy_wrapper, SelectOp());

    int compare = CompareDeviceResults(h_copy, d_copy, copy_items, g_verbose, g_verbose);
    printf("\tthrust::copy_if(): %s\n", (compare) ? "FAIL" : "PASS");
    AssertEquals(0, compare);

    if (h_copy) delete[] h_copy;
    if (d_copy) CubDebugExit(g_allocator.DeviceFree(d_copy));

#endif // THRUST_VERSION
}


/**
 * Test counting iterator
 */
template <typename T>
void TestCounting(T base, char *type_string)
{
    printf("\nTesting counting iterator on type %s (base: %d) \n", type_string, int(base)); fflush(stdout);

    //
    // Test iterator manipulation in kernel
    //

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

    CountingInputIterator<T> d_itr(base);
    Test(d_itr, h_reference);

#if (THRUST_VERSION >= 100700)  // Thrust 1.7 or newer

    //
    // Test with thrust::copy_if()
    //

    unsigned int max_items  = (unsigned int) ((1ull << ((sizeof(T) * 8) - 1)) - 1);
    int copy_items          = CUB_MIN(max_items - base, 100);     // potential issue with differencing overflows when T is a smaller type than can handle the offset
    T   *h_copy             = new T[copy_items];
    T   *d_copy             = NULL;

    for (int i = 0; i < copy_items; ++i)
        h_copy[i] = d_itr[i];

    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_copy, sizeof(T) * copy_items));
    thrust::device_ptr<T> d_copy_wrapper(d_copy);
    thrust::copy_if(d_itr, d_itr + copy_items, d_copy_wrapper, SelectOp());

    int compare = CompareDeviceResults(h_copy, d_copy, copy_items, g_verbose, g_verbose);
    printf("\tthrust::copy_if(): %s\n", (compare) ? "FAIL" : "PASS");
    AssertEquals(0, compare);

    if (h_copy) delete[] h_copy;
    if (d_copy) CubDebugExit(g_allocator.DeviceFree(d_copy));

#endif // THRUST_VERSION
}


/**
 * Test modified iterator
 */
template <typename T>
void TestModified(char *type_string)
{
    printf("\nTesting cache-modified iterator on type %s\n", type_string); fflush(stdout);

    //
    // Test iterator manipulation in kernel
    //

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

#if (THRUST_VERSION >= 100700)  // Thrust 1.7 or newer

    //
    // Test with thrust::copy_if()
    //

    T *d_copy = NULL;
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_copy, sizeof(T) * TEST_VALUES));

    CacheModifiedInputIterator<LOAD_CG, T> d_in_itr(d_data);
    CacheModifiedOutputIterator<STORE_CG, T> d_out_itr(d_copy);

    thrust::copy_if(d_in_itr, d_in_itr + TEST_VALUES, d_out_itr, SelectOp());

    int compare = CompareDeviceResults(h_data, d_copy, TEST_VALUES, g_verbose, g_verbose);
    printf("\tthrust::copy_if(): %s\n", (compare) ? "FAIL" : "PASS");
    AssertEquals(0, compare);

    // Cleanup
    if (d_copy) CubDebugExit(g_allocator.DeviceFree(d_copy));

#endif // THRUST_VERSION

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

    //
    // Test iterator manipulation in kernel
    //

    const unsigned int TEST_VALUES = 11000;

    T *h_data = new T[TEST_VALUES];
    for (int i = 0; i < TEST_VALUES; ++i)
    {
        InitValue(INTEGER_SEED, h_data[i], i);
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

    TransformInputIterator<T, TransformOp<T>, T*> d_itr(d_data, op);
    Test(d_itr, h_reference);

#if (THRUST_VERSION >= 100700)  // Thrust 1.7 or newer

    //
    // Test with thrust::copy_if()
    //

    T *h_copy = new T[TEST_VALUES];
    for (int i = 0; i < TEST_VALUES; ++i)
        h_copy[i] = op(h_data[i]);

    T *d_copy = NULL;
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_copy, sizeof(T) * TEST_VALUES));
    thrust::device_ptr<T> d_copy_wrapper(d_copy);

    thrust::copy_if(d_itr, d_itr + TEST_VALUES, d_copy_wrapper, SelectOp());

    int compare = CompareDeviceResults(h_copy, d_copy, TEST_VALUES, g_verbose, g_verbose);
    printf("\tthrust::copy_if(): %s\n", (compare) ? "FAIL" : "PASS");
    AssertEquals(0, compare);

    // Cleanup
    if (h_copy) delete[] h_copy;
    if (d_copy) CubDebugExit(g_allocator.DeviceFree(d_copy));

#endif // THRUST_VERSION

    if (h_data) delete[] h_data;
    if (d_data) CubDebugExit(g_allocator.DeviceFree(d_data));
}


/**
 * Test tex-obj texture iterator
 */
template <typename T>
void TestTexObj(char *type_string)
{
    printf("\nTesting tex-obj iterator on type %s\n", type_string); fflush(stdout);

    //
    // Test iterator manipulation in kernel
    //

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

    // Create and bind obj-based test iterator
    TexObjInputIterator<T> d_obj_itr;
    CubDebugExit(d_obj_itr.BindTexture(d_data, sizeof(T) * TEST_VALUES));

    Test(d_obj_itr, h_reference);

#if (THRUST_VERSION >= 100700)  // Thrust 1.7 or newer

    //
    // Test with thrust::copy_if()
    //

    T *d_copy = NULL;
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_copy, sizeof(T) * TEST_VALUES));
    thrust::device_ptr<T> d_copy_wrapper(d_copy);

    CubDebugExit(cudaMemset(d_copy, 0, sizeof(T) * TEST_VALUES));
    thrust::copy_if(d_obj_itr, d_obj_itr + TEST_VALUES, d_copy_wrapper, SelectOp());

    int compare = CompareDeviceResults(h_data, d_copy, TEST_VALUES, g_verbose, g_verbose);
    printf("\tthrust::copy_if(): %s\n", (compare) ? "FAIL" : "PASS");
    AssertEquals(0, compare);

    // Cleanup
    CubDebugExit(d_obj_itr.UnbindTexture());

    if (d_copy) CubDebugExit(g_allocator.DeviceFree(d_copy));

#endif  // THRUST_VERSION

    if (h_data) delete[] h_data;
    if (d_data) CubDebugExit(g_allocator.DeviceFree(d_data));
    if (d_dummy) CubDebugExit(g_allocator.DeviceFree(d_dummy));
}


#if CUDA_VERSION >= 5050

/**
 * Test tex-ref texture iterator
 */
template <typename T>
void TestTexRef(char *type_string)
{
    printf("\nTesting tex-ref iterator on type %s\n", type_string); fflush(stdout);

    //
    // Test iterator manipulation in kernel
    //

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

#if (THRUST_VERSION >= 100700)  // Thrust 1.7 or newer

    //
    // Test with thrust::copy_if()
    //

    T *d_copy = NULL;
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_copy, sizeof(T) * TEST_VALUES));
    thrust::device_ptr<T> d_copy_wrapper(d_copy);

    CubDebugExit(cudaMemset(d_copy, 0, sizeof(T) * TEST_VALUES));
    thrust::copy_if(d_ref_itr, d_ref_itr + TEST_VALUES, d_copy_wrapper, SelectOp());

    int compare = CompareDeviceResults(h_data, d_copy, TEST_VALUES, g_verbose, g_verbose);
    printf("\tthrust::copy_if(): %s\n", (compare) ? "FAIL" : "PASS");
    AssertEquals(0, compare);

    if (d_copy) CubDebugExit(g_allocator.DeviceFree(d_copy));

#endif  // THRUST_VERSION

    CubDebugExit(d_ref_itr.UnbindTexture());
    CubDebugExit(d_ref_itr2.UnbindTexture());

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

    //
    // Test iterator manipulation in kernel
    //

    const unsigned int TEST_VALUES = 11000;

    T *h_data = new T[TEST_VALUES];
    for (int i = 0; i < TEST_VALUES; ++i)
    {
        InitValue(INTEGER_SEED, h_data[i], i);
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

#if (THRUST_VERSION >= 100700)  // Thrust 1.7 or newer

    //
    // Test with thrust::copy_if()
    //

    T *h_copy = new T[TEST_VALUES];
    for (int i = 0; i < TEST_VALUES; ++i)
        h_copy[i] = op(h_data[i]);

    T *d_copy = NULL;
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_copy, sizeof(T) * TEST_VALUES));
    thrust::device_ptr<T> d_copy_wrapper(d_copy);

    thrust::copy_if(xform_itr, xform_itr + TEST_VALUES, d_copy_wrapper, SelectOp());

    int compare = CompareDeviceResults(h_copy, d_copy, TEST_VALUES, g_verbose, g_verbose);
    printf("\tthrust::copy_if(): %s\n", (compare) ? "FAIL" : "PASS");
    AssertEquals(0, compare);

    // Cleanup
    if (h_copy) delete[] h_copy;
    if (d_copy) CubDebugExit(g_allocator.DeviceFree(d_copy));

#endif  // THRUST_VERSION

    CubDebugExit(d_tex_itr.UnbindTexture());
    if (h_data) delete[] h_data;
    if (d_data) CubDebugExit(g_allocator.DeviceFree(d_data));
}

#endif  // CUDA_VERSION




/**
 * Run non-integer tests
 */
template <typename T>
void TestInteger(Int2Type<false> is_integer, char *type_string)
{
    TestModified<T>(type_string);
    TestTransform<T>(type_string);

#if CUB_CDP
    // Test tex-obj iterators if CUDA dynamic parallelism enabled
    TestTexObj<T>(type_string);
#endif  // CUB_CDP

#if CUDA_VERSION >= 5050
    // Test tex-ref iterators for CUDA 5.5
    TestTexRef<T>(type_string);
    TestTexTransform<T>(type_string);
#endif  // CUDA_VERSION
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

    // Get ptx version
    int ptx_version;
    CubDebugExit(PtxVersion(ptx_version));

    // Evaluate different data types
    Test<char>(CUB_TYPE_STRING(char));
    Test<short>(CUB_TYPE_STRING(short));
    Test<int>(CUB_TYPE_STRING(int));
    Test<long>(CUB_TYPE_STRING(long));
    Test<long long>(CUB_TYPE_STRING(long long));
    Test<float>(CUB_TYPE_STRING(float));
    if (ptx_version > 100)                          // Don't check doubles on PTX100 because they're down-converted
        Test<double>(CUB_TYPE_STRING(double));

    Test<char2>(CUB_TYPE_STRING(char2));
    Test<short2>(CUB_TYPE_STRING(short2));
    Test<int2>(CUB_TYPE_STRING(int2));
    Test<long2>(CUB_TYPE_STRING(long2));
    Test<longlong2>(CUB_TYPE_STRING(longlong2));
    Test<float2>(CUB_TYPE_STRING(float2));
    if (ptx_version > 100)                          // Don't check doubles on PTX100 because they're down-converted
        Test<double2>(CUB_TYPE_STRING(double2));

    Test<char3>(CUB_TYPE_STRING(char3));
    Test<short3>(CUB_TYPE_STRING(short3));
    Test<int3>(CUB_TYPE_STRING(int3));
    Test<long3>(CUB_TYPE_STRING(long3));
    Test<longlong3>(CUB_TYPE_STRING(longlong3));
    Test<float3>(CUB_TYPE_STRING(float3));
    if (ptx_version > 100)                          // Don't check doubles on PTX100 because they're down-converted
        Test<double3>(CUB_TYPE_STRING(double3));

    Test<char4>(CUB_TYPE_STRING(char4));
    Test<short4>(CUB_TYPE_STRING(short4));
    Test<int4>(CUB_TYPE_STRING(int4));
    Test<long4>(CUB_TYPE_STRING(long4));
    Test<longlong4>(CUB_TYPE_STRING(longlong4));
    Test<float4>(CUB_TYPE_STRING(float4));
    if (ptx_version > 100)                          // Don't check doubles on PTX100 because they're down-converted
        Test<double4>(CUB_TYPE_STRING(double4));

    Test<TestFoo>(CUB_TYPE_STRING(TestFoo));
    Test<TestBar>(CUB_TYPE_STRING(TestBar));

    printf("\nTest complete\n"); fflush(stdout);

    return 0;
}



