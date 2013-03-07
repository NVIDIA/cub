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

/**
 * \file
 * CUB umbrella include file
 */

#pragma once

// Grid
#include "grid/grid_barrier.cuh"
#include "grid/grid_queue.cuh"
#include "grid/grid_even_share.cuh"

// Threadblock
#include "block/block_load.cuh"
#include "block/block_radix_rank.cuh"
#include "block/block_radix_sort.cuh"
#include "block/block_reduce.cuh"
#include "block/block_scan.cuh"
#include "block/block_store.cuh"
#include "block/block_discontinuity.cuh"

// Warp
#include "warp/warp_scan.cuh"

// Thread
#include "thread/thread_load.cuh"
#include "thread/thread_reduce.cuh"
#include "thread/thread_scan.cuh"
#include "thread/thread_store.cuh"

// Host
#include "host/cuda_props.cuh"
#include "host/kernel_props.cuh"
#include "host/multi_buffer.cuh"
#include "host/spinlock.cuh"

// General
#include "debug.cuh"
#include "macro_utils.cuh"
#include "device_props.cuh"
#include "operators.cuh"
#include "ptx_intrinsics.cuh"
#include "type_utils.cuh"
#include "vector_type.cuh"
#include "allocator.cuh"


/**
 * \mainpage
 *
 * \tableofcontents
 *
 * \htmlonly
 * <a href="https://github.com/NVlabs/CUB/archive/0.9.zip"><img src="download-icon.png" style="position:relative; bottom:-10px;"/></a>
 * &nbsp;&nbsp;
 * <a href="https://github.com/NVlabs/CUB/archive/0.9.zip">Download CUB!</a>
 * <br>
 * <a href="https://github.com/NVlabs/CUB"><img src="github-icon-747d8b799a48162434b2c0595ba1317e.png" style="position:relative; bottom:-10px;"/></a>
 * &nbsp;&nbsp;
 * <a href="https://github.com/NVlabs/CUB">Fork CUB at GitHub!</a>
 * <br>
 * <a href="http://groups.google.com/group/cub-users"><img src="groups-icon.png" style="position:relative; bottom:-10px;"/></a>
 * &nbsp;&nbsp;
 * <a href="http://groups.google.com/group/cub-users">Join the cub-users discussion forum!</a>
 * \endhtmlonly
 *
 * \section sec0 (1) What is CUB?
 *
 * \par
 * CUB is a flexible library of cooperative threadblock primitives and other utilities for CUDA kernel programming.
 * CUB enhances productivity and portability by providing an abstraction layer over complex
 * threadblock, warp, and thread-level operations.
 *
 * \par
 * CUB's primitives are not bound to any particular width-of-parallelism or data type.  This allows them
 * to be flexible and tunable to fit your kernel needs.
 * Thus CUB is [<b>C</b>uda <b>U</b>n<b>b</b>ound](index.html).
 *
 * \image html simt_abstraction.png
 *
 * \par
 * Browse our collections of:
 * - [<b>Cooperative primitives</b>](group___simt_coop.html):
 *   - Threadblock operations (e.g., BlockRadixSort, BlockScan, BlockReduce, etc.)
 *   - Warp operations (e.g., WarpScan, etc.)
 *   - etc.
 * - [<b>SIMT utilities</b>](group___simt_utils.html):
 *   - Tile-based I/O utilities for performing vectorized|coalesced data movement of blocked|striped data tiles
 *   - Low-level thread-I/O using cache-modifiers (e.g., ThreadLoad & ThreadStore intrinsics)
 *   - Abstractions for threadblock work distribution (GridQueue, GridEvenShare, etc.)
 *   - etc.
 * - [<b>Host utilities</b>](group___host_util.html):
 *   - Caching allocator for quick management of device temporaries
 *   - Device reflection
 *   - etc.
 *
 * \section sec2 (2) Recent news
 *
 * \par
 * - [<b><em>CUB v0.9 "preview" release</em></b>](https://github.com/NVlabs/CUB/archive/0.9.zip) (3/6/2013).  CUB is the first durable, high-performance
 *   library of cooperative threadblock, warp, and thread primitives for CUDA kernel
 *   programming.  More primitives and examples coming soon!
 *
 * \section sec3 (3) A simple example
 *
 * \par
 * The following code snippet illustrates a simple CUDA kernel for sorting a threadblock's data:
 *
 * \par
 * \code
 * #include <cub.cuh>
 *
 * // An tile-sorting CUDA kernel
 * template <
 *      int         BLOCK_THREADS,              // Threads per threadblock
 *      int         ITEMS_PER_THREAD,           // Items per thread
 *      typename    T>                          // Numeric data type
 * __global__ void TileSortKernel(T *d_in, T *d_out)
 * {
 *      using namespace cub;
 *      const int TILE_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD;
 *
 *      // Parameterize cub::BlockRadixSort for the parallel execution context
 *      typedef BlockRadixSort<T, BLOCK_THREADS> BlockRadixSort;
 *
 *      // Declare the shared memory needed by BlockRadixSort
 *      __shared__ typename BlockRadixSort::SmemStorage smem_storage;
 *
 *      // A segment of data items per thread
 *      T data[ITEMS_PER_THREAD];
 *
 *      // Load a tile of data using vector-load instructions
 *      BlockLoadVectorized(data, d_in + (blockIdx.x * TILE_SIZE));
 *
 *      // Sort data in ascending order
 *      BlockRadixSort::SortBlocked(smem_storage, data);
 *
 *      // Store the sorted tile using vector-store instructions
 *      BlockStoreVectorized(data, d_out + (blockIdx.x * TILE_SIZE));
 * }
 * \endcode
 *
 * \par
 * The cub::BlockRadixSort type performs a cooperative radix sort across the
 * threadblock's data items.  Its implementation is parameterized by the number of threadblock threads and the aggregate
 * data type \p T, and is specialized for the underlying architecture.
 *
 * \par
 * Once instantiated, the cub::BlockRadixSort type exposes an opaque cub::BlockRadixSort::SmemStorage
 * member type.  The threadblock uses this storage type to allocate the shared memory needed by the
 * primitive.  This storage type can be aliased or <tt>union</tt>'d with other types so that the
 * shared memory can be reused for other purposes.
 *
 * \par
 * Furthermore, the kernel uses CUB's primitives for vectorizing global
 * loads and stores.  For example, <tt>ld.global.v4.s32</tt> PTX instructions
 * will be generated when \p T = \p int and \p ITEMS_PER_THREAD is a multiple of 4.
 *
 * \section sec4 (4) Why do you need CUB?
 *
 * \par
 * Kernel development is perhaps the most challenging, time-consuming,
 * and costly aspect of CUDA programming.  It is where the complexity of parallelism is
 * expressed.  Kernel code is often the most performance-sensitive and
 * difficult-to-maintain layer in the CUDA software stack.  This is particularly true for
 * complex parallel algorithms having intricate dependences between threads.
 *
 * \par
 * However, with the exception of CUB, there are few (if any) software libraries of
 * reusable kernel primitives. In the CUDA ecosystem, CUB is unique in this regard.
 * As a SIMT library and software abstraction layer, CUB gives you:
 * -# <b>The ease of sequencing.</b>  Parallel primitives within
 *    kernels can be simply sequenced together (similar to programming with
 *    [<b>Thrust</b>](http://http://thrust.github.com/) in the host program).
 * -# <b>Simple performance tuning.</b>  CUB primitives provide consistent functional
 *    abstractions whose performance behavior can be statically tuned.  For example,
 *    most CUB primitives support alternative algorithmic strategies and variable grain
 *    sizes (i.e., threads per threadblock, items per thread, etc.).
 * -# <b>Performance-portability.</b> CUB primitives are specialized to match
 *    the target hardware.  Furthermore, CUDA kernels can be made future-proof by
 *    simply recompiling against new CUB releases to leverage new algorithmic
 *    developments, hardware instructions, etc.  (Currently, kernels are often hand-rewritten
 *    for every new architecture.)
 * -# <b>Robustness and durability.</b> CUB primitives are designed to function properly for
 *    arbitrary data types and widths-of-parallelism (not just for the built-in C++ types
 *    and power-of-two threadblocks).
 *
 * \section sec5 (5) Where is CUB positioned in the CUDA ecosystem?
 *
 * \par
 * CUDA's programming model exposes three different levels of execution and their corresponding
 * abstraction layers (i.e., the "black boxes" shown in the software stacks below):
 *
 * <table border="0px" cellpadding="10px" cellspacing="0px"><tr>
 * <td width="50%">
 * - <b>Grid kernel (scalar)</b>.  A single thread invokes a CUDA grid to perform some
 *    data parallel function.  This is the highest and most common level of CUDA software
 *    abstraction.  Programmers do not have to reason about parallel CUDA
 *    threads.  Libraries targeting this level include:
 *    - [<b>CUBLAS</b>](https://developer.nvidia.com/cublas)
 *    - [<b>CUFFT</b>](https://developer.nvidia.com/cufft)
 *    - [<b>CUSPARSE</b>](https://developer.nvidia.com/cusparse)
 *    - [<b>Thrust</b>](http://thrust.github.com/)
 * </td>
 * <td width="50%">
 *    \htmlonly
 *    <a href="kernel_abstraction.png"><center><img src="kernel_abstraction.png" width="100%"/></center></a>
 *    \endhtmlonly
 * </td>
 * </tr><tr>
 * <td>
 * - <b>Threadblock / warp (SIMT)</b>.  A threadblock or warp of threads collectively invokes some
 *    cooperative function.  This is the least common level of CUDA software reuse.
 *    Libraries targeting this level include:
 *    - [<b>CUB</b>](index.html)
 * </td>
 * <td>
 *    \htmlonly
 *    <a href="simt_abstraction.png"><center><img src="simt_abstraction.png" width="100%"/></center></a>
 *    \endhtmlonly
 * </td>
 * </tr><tr>
 * <td>
 * - <b>Device thread (scalar)</b>.  A single CUDA thread invokes some scalar function.
 *    This is the lowest level of CUDA software abstraction.  Programmers do not have to reason about
 *    the interaction of parallel CUDA threads.  Libraries targeting this level include:
 *    - <b>CUDA API</b> (e.g., \p text1D(), \p atomicAdd(), \p popc(), etc.)
 *    - [<b>CUB</b>](index.html)
 * </td>
 * <td>
 *    \htmlonly
 *    <a href="devfun_abstraction.png"><center><img src="devfun_abstraction.png" width="100%"/></center></a>
 *    \endhtmlonly
 * </td>
 * </tr></table>
 *
 *
 * \section sec6 (6) How does CUB work?
 *
 * \par
 * CUB leverages the following programming idioms:
 * -# [<b>C++ templates</b>](index.html#sec3sec1)
 * -# [<b>Reflective type structure</b>](index.html#sec3sec2)
 * -# [<b>Flexible data mapping</b>](index.html#sec3sec3)
 *
 * \subsection sec3sec1 4.1 &nbsp;&nbsp; C++ templates
 *
 * \par
 * As a SIMT library, CUB must be flexible enough to accommodate a wide spectrum
 * of <em>parallel execution contexts</em>,
 * i.e., specific:
 *    - Data types
 *    - Widths of parallelism (threadblock threads)
 *    - Grain sizes (data items per thread)
 *    - Underlying architectures (special instructions, warp width, rules for bank conflicts, etc.)
 *    - Tuning requirements (e.g., latency vs. throughput)
 *
 * \par
 * To provide this flexibility, CUB is implemented as a C++ template library.
 * C++ templates are a way to write generic algorithms and data structures.
 * There is no need to build CUB separately.  You simply #<tt>include</tt> the
 * <tt>cub.cuh</tt> header file into your <tt>.cu</tt> or <tt>.cpp</tt> sources
 * and compile with CUDA's <tt>nvcc</tt> compiler.
 *
 * \subsection sec3sec2 4.2 &nbsp;&nbsp; Reflective type structure
 *
 * \par
 * Cooperation requires shared memory for communicating between threads.
 * However, the specific size and layout of the memory needed by a given
 * primitive will be specific to the details of its parallel execution context (e.g., how
 * many threads are calling into it, how many items per thread, etc.).  Furthermore,
 * this shared memory must be allocated externally to the component if it is to be
 * reused elsewhere by the threadblock.
 *
 * \par
 * \code
 * // Parameterize a BlockScan type for use with 128 threads
 * // and 4 items per thread
 * typedef cub::BlockScan<unsigned int, 128, 4> BlockScan;
 *
 * // Declare shared memory for BlockScan
 * __shared__ typename BlockScan::SmemStorage smem_storage;
 *
 * // A segment of consecutive input items per thread
 * int data[4];
 *
 * // Obtain data in blocked order
 * ...
 *
 * // Perform an exclusive prefix sum across the tile of data
 * BlockScan::ExclusiveSum(smem_storage, data, data);
 *
 * \endcode
 *
 * \par
 * To address this issue, we encapsulate cooperative procedures within
 * <em>reflective type structure</em> (C++ classes).  As illustrated in the
 * cub::BlockScan example above, these primitives are C++ classes with
 * interfaces that expose both (1) procedural methods as well as (2) the opaque
 * shared memory types needed for their operation.
 *
 * \subsection sec3sec3 4.3 &nbsp;&nbsp; Flexible data mapping
 *
 * \par
 * We often design kernels such that each threadblock is assigned a "tile" of data
 * items for processing.  When the tile size equals the threadblock size, the
 * mapping of data onto threads is straightforward (1:1 items
 * to threadblock threads). Alternatively, it is often desirable to processes more
 * than one datum per thread.  When doing so, we must decide how
 * to partition this "tile" of items across the threadblock.
 *
 * \par
 * CUB primitives support the following data arrangements:
 * - <b><em>Blocked arrangement</em></b>.  The aggregate tile of items is partitioned
 *   evenly across threads in "blocked" fashion with thread<sub><em>i</em></sub>
 *   owning the <em>i</em><sup>th</sup> segment of consecutive elements.
 * - <b><em>Striped arrangement</em></b>.  The aggregate tile of items is partitioned across
 *   threads in "striped" fashion, i.e., the \p ITEMS_PER_THREAD items owned by
 *   each thread have logical stride \p BLOCK_THREADS between them.
 * <br><br>
 * \image html thread_data_1.png
 * <div class="centercaption">Blocked vs. striped arrangements with \p BLOCK_THREADS = 4 and
 * \p ITEMS_PER_THREAD = 2, emphasis on items owned by <em>thread</em><sub>0</sub></div>
 * <br>
 *
 * \par
 * The benefits of processing multiple items per thread (a.k.a., <em>register blocking</em>, <em>granularity coarsening</em>, etc.) include:
 * - <b>Algorithmic efficiency</b>.  Sequential work over multiple items in
 *   thread-private registers is cheaper than synchronized, cooperative
 *   work through shared memory spaces
 * - <b>Data occupancy</b>.  The number of items that can be resident on-chip in
 *   thread-private register storage is often greater than the number of
 *   schedulable threads
 * - <b>Instruction-level parallelism</b>.  Multiple items per thread also
 *   facilitates greater ILP for improved throughput and utilization
 *
 * \par
 * The cub::BlockExchange primitive provides operations for converting between blocked
 * and striped arrangements. Blocked arrangements are often desirable for
 * algorithmic benefits (where long sequences of items can be processed sequentially
 * within each thread).  Striped arrangements are often desirable for data movement
 * through global memory (where read/write coalescing is a important performance
 * consideration).
 *
 * \section sec7 (7) Contributors
 *
 * \par
 * CUB is developed as an open-source project by [NVIDIA Research](http://research.nvidia.com).
 * The primary contributor is [Duane Merrill](http://github.com/dumerrill).
 *
 * \section sec8 (8) Open Source License
 *
 * \par
 * CUB is available under the "New BSD" open-source license:
 *
 * \par
 * \code
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
 * \endcode
 *
 */


/**
 * \defgroup Simt SIMT Primitives
 */

/**
 * \defgroup SimtCoop Cooperative SIMT Operations
 * \ingroup Simt
 */

/**
 * \defgroup SimtUtils SIMT Utilities
 * \ingroup Simt
 */

/**
 * \defgroup HostUtil Host Utilities
 */
