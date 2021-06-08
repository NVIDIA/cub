/******************************************************************************
 * Copyright (c) 2011-2021, NVIDIA CORPORATION. All rights reserved.
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
 * Test of BlockShuffle utilities
 ******************************************************************************/

// Ensure printing of CUDA runtime errors to console
#define CUB_STDERR

#include <stdio.h>
#include <limits>
#include <typeinfo>
#include <memory>

#include <cub/util_allocator.cuh>
#include <cub/block/block_shuffle.cuh>

#include "test_util.h"

using namespace cub;

template<typename DataType>
__global__ void IotaKernel(
        const unsigned int num_items,
        DataType *data)
{
    const unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i < num_items)
    {
        data[i] = i;
    }
}

template<typename DataType>
void Iota(
        const unsigned int num_items,
        DataType *data)
{
    const unsigned int ThreadsPerBlock = 256;
    const unsigned int blocks_per_grid = (num_items + ThreadsPerBlock - 1) / ThreadsPerBlock;

    IotaKernel<<<blocks_per_grid, ThreadsPerBlock>>>(num_items, data);

    CubDebugExit(cudaPeekAtLastError());
    CubDebugExit(cudaDeviceSynchronize());
}

template <
        typename DataType,
        unsigned int ThreadsInBlock,
        unsigned int ItemsPerThread,
        typename ActionType>
__global__ void BlockShuffleTestKernel(
        DataType *data,
        ActionType action)
{
    typedef cub::BlockShuffle<DataType, ThreadsInBlock> BlockShuffle;

    __shared__ typename BlockShuffle::TempStorage temp_storage_shuffle;

    DataType thread_data[ItemsPerThread];

    data += threadIdx.x * ItemsPerThread;
    for (unsigned int item = 0; item < ItemsPerThread; item++)
    {
        thread_data[item] = data[item];
    }
    __syncthreads();

    BlockShuffle block_shuffle(temp_storage_shuffle);
    action(block_shuffle, thread_data);

    for (unsigned int item = 0; item < ItemsPerThread; item++)
    {
        data[item] = thread_data[item];
    }
}

template<
        typename DataType,
        unsigned int ItemsPerThread,
        unsigned int ThreadsInBlock,
        typename ActionType>
void BlockShuffleTest(DataType *data, ActionType action)
{
    BlockShuffleTestKernel<DataType, ThreadsInBlock, ItemsPerThread><<<1, ThreadsInBlock>>> (data, action);

    CubDebugExit(cudaPeekAtLastError());
    CubDebugExit(cudaDeviceSynchronize());
}

template <
        typename DataType,
        unsigned int ItemsPerThread,
        unsigned int ThreadsInBlock>
struct UpTest
{
    __device__ void operator()(
            BlockShuffle<DataType, ThreadsInBlock> &block_shuffle,
            DataType (&thread_data)[ItemsPerThread]) const
    {
        block_shuffle.Up(thread_data, thread_data);
    }

    static __host__ bool check(const DataType *data, int i)
    {
        if (i == 0)
        {
            return data[i] == 0;
        }

        return data[i] == i - 1;
    }
};

template <
        typename DataType,
        unsigned int ItemsPerThread,
        unsigned int ThreadsInBlock>
struct DownTest
{
    __device__ void operator()(
            BlockShuffle<DataType, ThreadsInBlock> &block_shuffle,
            DataType (&thread_data)[ItemsPerThread]) const
    {
        block_shuffle.Down(thread_data, thread_data);
    }

    static __host__ bool check(const DataType *data, int i)
    {
        if (i == ItemsPerThread * ThreadsInBlock - 1)
        {
            return data[i] == i;
        }

        return data[i] == i + 1;
    }
};

template<typename DataType,
         unsigned int ThreadsInBlock,
         int offset>
struct OffsetTestBase
{
    static constexpr unsigned int ItemsPerThread = 1;

    __device__ void operator()(
            BlockShuffle<DataType, ThreadsInBlock> &block_shuffle,
            DataType (&thread_data)[ItemsPerThread]) const
    {
        block_shuffle.Offset(thread_data[0], thread_data[0], offset);
    }
};

template <typename DataType,
          unsigned int ThreadsInBlock>
struct OffsetUpTest : public OffsetTestBase<DataType, ThreadsInBlock, -1 /* offset */>
{
    static __host__ bool check(const DataType *data, int i)
    {
        return UpTest<DataType, 1 /* ItemsPerThread */, ThreadsInBlock>::check (data, i);
    }
};

template<typename DataType,
         unsigned int ThreadsInBlock>
struct OffsetDownTest : public OffsetTestBase<DataType, ThreadsInBlock, 1 /* offset */>
{
    static __host__ bool check(const DataType *data, int i)
    {
        return DownTest<DataType, 1 /* ItemsPerThread */, ThreadsInBlock>::check (data, i);
    }
};

template<typename DataType,
         unsigned int ThreadsInBlock,
         unsigned int offset>
struct RotateTestBase
{
    static constexpr unsigned int ItemsPerThread = 1;

    __device__ void operator()(
            BlockShuffle<DataType, ThreadsInBlock> &block_shuffle,
            DataType (&thread_data)[ItemsPerThread]) const
    {
        block_shuffle.Rotate(thread_data[0], thread_data[0], offset);
    }

    static __host__ bool check(const DataType *data, int i)
    {
        return data[i] == static_cast<DataType>((i + offset) % ThreadsInBlock);
    }
};

template<typename DataType,
         unsigned int ThreadsInBlock>
struct RotateUpTest : public RotateTestBase<DataType, ThreadsInBlock, 1 /* offset */>
{ };

template<typename DataType,
         unsigned int ThreadsInBlock>
struct RotateTest : public RotateTestBase<DataType, ThreadsInBlock, 24 /* offset */>
{ };


template <typename DataType, typename TestType>
int CheckResult(
        int num_items,
        const DataType *d_output,
        DataType *h_output,
        const TestType &test)
{
    CubDebugExit(cudaMemcpy(h_output, d_output, num_items * sizeof (DataType), cudaMemcpyDeviceToHost));

    for (int i = 0; i < num_items; i++)
    {
        if (!test.check (h_output, i))
        {
            return 1;
        }
    }

    return 0;
}

template <
        typename DataType,
        unsigned int ItemsPerThread,
        unsigned int ThreadsInBlock,
        template<typename, unsigned int, unsigned int> class TestType>
void Test(unsigned int num_items,
          DataType *d_data,
          DataType *h_data)
{
    TestType<DataType, ItemsPerThread, ThreadsInBlock> test;

    Iota(num_items, d_data);
    BlockShuffleTest<DataType, ItemsPerThread, ThreadsInBlock>(d_data, test);
    AssertEquals(0, CheckResult(num_items, d_data, h_data, test));
}

/**
 * Some methods of it only support a single element per thread.
 * This structure skips tests for unsupported cases.
 */
template <
        typename DataType,
        unsigned int ItemsPerThread,
        unsigned int ThreadsInBlock,
        template<typename, unsigned int> class TestType>
struct SingleItemTestHelper
{
    static void run(unsigned int /* num_items */,
                    DataType * /* d_data */,
                    DataType * /* h_data */)
    {
    }
};

template <
        typename DataType,
        unsigned int ThreadsInBlock,
        template<typename, unsigned int> class TestType>
struct SingleItemTestHelper<DataType, 1, ThreadsInBlock, TestType>
{
    static void run(unsigned int num_items,
                   DataType *d_data,
                   DataType *h_data)
    {
        TestType<DataType, ThreadsInBlock> test;

        Iota(num_items, d_data);
        BlockShuffleTest<DataType, 1 /* ItemsPerThread */, ThreadsInBlock>(d_data, test);
        AssertEquals(0, CheckResult(num_items, d_data, h_data, test));
    }
};


template <
        typename DataType,
        unsigned int ItemsPerThread,
        unsigned int ThreadsInBlock>
void Test(
    CachingDeviceAllocator &g_allocator
)
{
    const unsigned int num_items = ItemsPerThread * ThreadsInBlock;

    DataType *d_data = nullptr;
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_data, sizeof(DataType) * num_items));

    std::unique_ptr<DataType[]> h_data(new DataType[num_items]);

    Test<DataType, ItemsPerThread, ThreadsInBlock, UpTest>(num_items, d_data, h_data.get());
    Test<DataType, ItemsPerThread, ThreadsInBlock, DownTest>(num_items, d_data, h_data.get());

    SingleItemTestHelper<DataType, ItemsPerThread, ThreadsInBlock, OffsetUpTest>().run(num_items,
                                                                                       d_data,
                                                                                       h_data.get());

    SingleItemTestHelper<DataType, ItemsPerThread, ThreadsInBlock, OffsetDownTest>().run(num_items,
                                                                                         d_data,
                                                                                         h_data.get());

    SingleItemTestHelper<DataType, ItemsPerThread, ThreadsInBlock, RotateUpTest>().run(num_items,
                                                                                       d_data,
                                                                                       h_data.get());

    SingleItemTestHelper<DataType, ItemsPerThread, ThreadsInBlock, RotateTest>().run(num_items,
                                                                                     d_data,
                                                                                     h_data.get());

    if (d_data)
    {
        CubDebugExit(g_allocator.DeviceFree(d_data));
    }
}

template <unsigned int ItemsPerThread>
void Test(CachingDeviceAllocator &g_allocator)
{
    Test<int16_t, ItemsPerThread, 32>(g_allocator);
    Test<int32_t, ItemsPerThread, 32>(g_allocator);
    Test<int32_t, ItemsPerThread, 512>(g_allocator);
    Test<int64_t, ItemsPerThread, 512>(g_allocator);
    Test<int64_t, ItemsPerThread, 1024>(g_allocator);
}

int main(int argc, char** argv)
{
    CommandLineArgs args(argc, argv);

    // Initialize device
    CubDebugExit(args.DeviceInit());

    CachingDeviceAllocator g_allocator(true);

    Test<1> (g_allocator);
    Test<2> (g_allocator);
    Test<15> (g_allocator);

    return 0;
}
