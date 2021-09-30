# CUB: Collective Primitives for CUDA

CUB provides a toolkit of primitives for crafting parallel algorithms using CUDA,
NVIDIA's GPU programming model:
- **Device-wide primitives**:
  - Sort, prefix scan, reduction, histogram, etc.
  - Compatible with CUDA dynamic parallelism.
- **Block-wide collective primitives**:
  - I/O, sort, prefix scan, reduction, histogram, etc.
  - Compatible with arbitrary thread block sizes and types.
- **Warp-wide "collective" primitives**:
  - Warp-wide prefix scan, reduction, etc.
  - Safe and architecture-specific.
- **Thread and resource utilities**:
  - PTX intrinsics, device reflection, texture-caching iterators, caching memory allocators, etc.

![Orientation of collective primitives within the CUDA software stack](http://nvlabs.github.io/cub/assets/images/cub_overview.png)

CUB is included in the NVIDIA HPC SDK and the CUDA Toolkit.

## Examples

CUB is best learned through examples:

```cuda
#include <cub/cub.cuh>

// Block-sorting CUDA kernel.
__global__ void BlockSortKernel(int* d_in, int* d_out) {
  // Specialize BlockRadixSort, BlockLoad, and BlockStore for 128 threads
  // owning 16 integer items each.
  using BlockRadixSort = cub::BlockRadixSort<int, 128, 16>;
  using BlockLoad = cub::BlockLoad<int, 128, 16, BLOCK_LOAD_TRANSPOSE>;
  using BlockStore = cub::BlockStore<int, 128, 16, BLOCK_STORE_TRANSPOSE>;

  // Allocate shared memory.
  __shared__ union {
    BlockRadixSort::TempStorage  sort;
    BlockLoad::TempStorage       load;
    BlockStore::TempStorage      store;
  } temp_storage;

  int block_offset = blockIdx.x * (128 * 16);	// OffsetT for this block.

  // Obtain a segment of 2048 consecutive keys that are blocked across threads.
  int thread_keys[16];
  BlockLoad(temp_storage.load).Load(d_in + block_offset, thread_keys);
  __syncthreads();

  // Collectively sort the keys
  BlockRadixSort(temp_storage.sort).Sort(thread_keys);
  __syncthreads();

  // Store the sorted segment
  BlockStore(temp_storage.store).Store(d_out + block_offset, thread_keys);
}
```

Each thread block uses `cub::BlockRadixSort` to collectively sort
  its own input segment.
The class is specialized by the data type being sorted, by the number of
  threads per block, by the number of keys per thread, and implicitly by the
  targeted compilation architecture.

The `cub::BlockLoad` and `cub::BlockStore` classes are similarly specialized.
Furthermore, to provide coalesced accesses to device memory, these primitives
  are configured to access memory using a striped access pattern (where
  consecutive threads simultaneously access consecutive items) and then
  transpose the keys into a blocked arrangement of elements across threads.

Once specialized, these classes expose opaque `TempStorage` member types.
The thread block uses these storage types to statically allocate the union of
  shared memory needed by the thread block.
Alternatively these storage types could be aliased to global memory allocations.

## Developing CUB

CUB and [Thrust] (the C++ parallel algorithms library) depend on each other.
It is recommended to clone Thrust and build CUB as a component of Thrust.

CUB uses the [CMake build system] to build unit tests, examples, and header
  tests.
To build CUB as a developer, the following
  recipe should be followed:

```
# Clone Thrust and CUB from Github. CUB is located in Thrust's
# `dependencies/cub` submodule.
git clone --recursive https://github.com/NVIDIA/thrust.git
cd thrust

# Create build directory:
mkdir build
cd build

# Configure -- use one of the following:
cmake -DTHRUST_INCLUDE_CUB_CMAKE=ON ..   # Command line interface.
ccmake -DTHRUST_INCLUDE_CUB_CMAKE=ON ..  # ncurses GUI (Linux only)
cmake-gui  # Graphical UI, set source/build directories and options in the app

# Build:
cmake --build . -j <num jobs>   # invokes make (or ninja, etc)

# Run tests and examples:
ctest
```

By default, the C++14 standard is targeted, but this can be changed in CMake.

More information on configuring your CUB build and creating a pull request
  can be found in the [contributing section].

## Licensing

CUB is an open source project developed on [GitHub].
CUB is distributed under the [BSD 3-Clause License];
  some parts are distributed under the [Apache License v2.0 with LLVM Exceptions].



[GitHub]: https://github.com/nvidia/cub
[Thrust]: https://github.com/nvidia/thrust

[CMake section]: https://nvidia.github.io/cub/setup/cmake_options.html
[contributing section]: https://nvidia.github.io/cub/contributing.html

[CMake build system]: https://cmake.org

[Apache License v2.0 with LLVM Exceptions]: https://llvm.org/LICENSE.txt
[BSD 3-Clause License]: https://opensource.org/licenses/BSD-3-Clause

