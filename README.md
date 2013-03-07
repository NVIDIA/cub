<hr>
<h3>About CUB</h3>

We recommend the [CUB Project Website](http://nvlabs.github.com/cub) and the [cub-users discussion forum](http://groups.google.com/group/cub-users) for further information and examples.

CUB is a library of cooperative threadblock primitives and other high performance
utilities for CUDA kernel programming.  CUB enhances productivity, performance, and portability by
providing an abstraction layer over complex threadblock, warp, and thread-level operations.

![SIMT abstraction layer](http://nvlabs.github.com/cub/simt_abstraction.png)

<br><hr>
<h3>Recent news</h3>

- CUB v0.9 "preview" release (3/6/2013). CUB is the first durable, high-performance library of cooperative threadblock, warp, and thread primitives for CUDA kernel programming. More primitives and examples coming soon!

<br><hr>
<h3>A Simple Example</h3>

```C++
#include <cub.cuh>
 
// An exclusive prefix sum CUDA kernel (for a single-threadblock grid)
template <
    int         BLOCK_THREADS,              // Threads per threadblock
    int         ITEMS_PER_THREAD,           // Items per thread
    typename    T>                          // Data type
__global__ void PrefixSumKernel(T *d_in, T *d_out)
{
    using namespace cub;
 
    // Parameterize BlockScan for the current execution context
    typedef BlockScan<T, BLOCK_THREADS> BlockScan;
 
    // The shared memory needed by BlockScan
    __shared__ typename BlockScan::SmemStorage smem_storage;
 
    // A segment of data items per thread
    T data[ITEMS_PER_THREAD];
 
    // Load a tile of data using vector-load instructions
    BlockLoadVectorized(data, d_in, 0);
 
    // Perform an exclusive prefix sum across the tile of data
    BlockScan::ExclusiveSum(smem_storage, data, data);

    // Store a tile of data using vector-load instructions
    BlockStoreVectorized(data, d_out, 0);
}
```

<br><hr>
<h3>Contributors</h3>

CUB is developed as an open-source project by [NVIDIA Research](http://research.nvidia.com).  The primary contributor is [Duane Merrill](http://github.com/dumerrill).

<br><hr>
<h3>Open Source License</h3>

CUB is available under the "New BSD" open-source license:

```
Copyright (c) 2010-2011, Duane Merrill.  All rights reserved.
Copyright (c) 2011-2013, NVIDIA CORPORATION.  All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
   *  Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
   *  Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
   *  Neither the name of the NVIDIA CORPORATION nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```
