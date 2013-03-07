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

// Thread block
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
 * <a href="https://github.com/NVlabs/cub/archive/0.9.zip"><img src="download-icon.png" style="position:relative; bottom:-10px; border:0px;"/></a>
 * &nbsp;&nbsp;
 * <a href="https://github.com/NVlabs/cub/archive/0.9.zip">Download CUB!</a>
 * <br>
 * <a href="https://github.com/NVlabs/cub"><img src="github-icon-747d8b799a48162434b2c0595ba1317e.png" style="position:relative; bottom:-10px; border:0px;"/></a>
 * &nbsp;&nbsp;
 * <a href="https://github.com/NVlabs/cub">Fork CUB at GitHub!</a>
 * <br>
 * <a href="http://groups.google.com/group/cub-users"><img src="groups-icon.png" style="position:relative; bottom:-10px; border:0px;"/></a>
 * &nbsp;&nbsp;
 * <a href="http://groups.google.com/group/cub-users">Join the cub-users discussion forum!</a>
 * \endhtmlonly
 *
 * \section sec0 (1) What is CUB?
 *
 * \par
 * CUB is a library of high-performance parallel primitives and other utilities for
 * building CUDA kernel software. CUB enhances productivity, performance, and portability
 * by providing an abstraction layer over complex
 * [block-level] (http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#programming-model),
 * [warp-level] (http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#hardware-implementation), and
 * [thread-level](http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#programming-model) operations.
 *
 * \par
 * CUB's primitives are not bound to any particular width of parallelism or to any particular
 * data type.  This allows them to be flexible and tunable to fit your kernels' needs.
 * Thus CUB is [<b>C</b>UDA <b>U</b>n<b>b</b>ound](index.html).
 *
 * \image html cub_overview.png
 *
 * \par
 * Browse our collections of:
 * - [<b>Cooperative primitives</b>](group___simt_coop.html), including:
 *   - Thread block operations (e.g., radix sort, prefix scan, reduction, etc.)
 *   - Warp operations (e.g., prefix scan)
 * - [<b>SIMT utilities</b>](group___simt_utils.html), including:
 *   - Tile-based I/O utilities (e.g., for performing {vectorized, coalesced} data movement of {blocked, striped} data tiles)
 *   - Low-level thread I/O using cache-modifiers
 *   - Abstractions for thread block work distribution (e.g., work-stealing, even-share, etc.)
 * - [<b>Host utilities</b>](group___host_util.html), including:
 *   - Caching allocator for quick management of device temporaries
 *   - Device reflection
 *
 * \section sec2 (2) Recent news
 *
 * \par
 * - [<b><em>CUB v0.9 "preview" release</em></b>](https://github.com/NVlabs/cub/archive/0.9.zip) (3/7/2013).  CUB is the first durable, high-performance
 *   library of cooperative block-level, warp-level, and thread-level primitives for CUDA kernel
 *   programming.  More primitives and examples coming soon!
 *
 * \section sec3 (3) A simple example
 *
 * \par
 * The following code snippet illustrates a simple CUDA kernel for sorting a thread block's data:
 *
 * \par
 * \code
 * #include <cub.cuh>
 *
 * // An tile-sorting CUDA kernel
 * template <
 *      int         BLOCK_THREADS,              // Threads per block
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
 * thread block's data items.  Its implementation is parameterized by the number of threads per block and the aggregate
 * data type \p T and is specialized for the underlying architecture.
 *
 * \par
 * Once instantiated, the cub::BlockRadixSort type exposes an opaque cub::BlockRadixSort::SmemStorage
 * member type.  The thread block uses this storage type to allocate the shared memory needed by the
 * primitive.  This storage type can be aliased or <tt>union</tt>'d with other types so that the
 * shared memory can be reused for other purposes.
 *
 * \par
 * Furthermore, the kernel uses CUB's primitives for vectorizing global
 * loads and stores.  For example, lower-level <tt>ld.global.v4.s32</tt>
 * [PTX instructions](http://docs.nvidia.com/cuda/parallel-thread-execution)
 * will be generated when \p T = \p int and \p ITEMS_PER_THREAD is a multiple of 4.
 *
 * \section sec4 (4) Why do you need CUB?
 *
 * \par
 * CUDA kernel software is where the complexity of parallelism is expressed.
 * Programmers must reason about deadlock, livelock, synchronization, race conditions,
 * shared memory layout, plurality of state, granularity, throughput, latency,
 * memory bottlenecks, etc. Constructing and fine-tuning kernel code is perhaps the
 * most challenging, time-consuming aspect of CUDA programming.
 *
 * \par
 * However, with the exception of CUB, there are few (if any) software libraries of
 * reusable kernel primitives. In the CUDA ecosystem, CUB is unique in this regard.
 * As a [SIMT](http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#hardware-implementation)
 * library and software abstraction layer, CUB provides:
 * -# <b>Simplicity of composition.</b>  Parallel CUB primitives can be simply sequenced
 *    together in kernel code.  (This convenience is analogous to programming with
 *    [<b>Thrust</b>](http://thrust.github.com/) primitives in the host program.)
 * -# <b>High performance.</b> CUB simplifies high performance kernel development by
 *    taking care to implement and make available the fastest available algorithms,
 *    strategies, and techniques.
 * -# <b>Performance portability.</b> CUB primitives are specialized to match
 *    the target hardware.  Furthermore, the CUB library continually evolves to accommodate new
 *    algorithmic developments, hardware instructions, etc.
 * -# <b>Simplicity of performance tuning.</b>  CUB primitives provide parallel abstractions
 *    whose performance behavior can be statically tuned.  For example, most CUB primitives
 *    support alternative algorithmic strategies and variable grain sizes (threads per block,
 *    items per thread, etc.).
 * -# <b>Robustness and durability.</b> CUB primitives are designed to function properly for
 *    arbitrary data types and widths of parallelism (not just for the built-in C++ types
 *    or for powers-of-two threads per block).
 *
 * \section sec5 (5) Where is CUB positioned in the CUDA ecosystem?
 *
 * \par
 * CUDA's programming model embodies three different levels of program execution, each
 * engendering its own abstraction layer in the CUDA software stack (i.e., the "black boxes"
 * below):
 *
 * <table border="0px" style="padding:0px; border:0px; margin:0px;"><tr>
 * <td width="50%">
 * \par
 * <b>CUDA kernel</b>.  A CPU program invokes a CUDA kernel to perform
 * some data-parallel function.  Reuse of entire kernels (by incorporating them into
 * libraries) is the most common form of code reuse for CUDA.  Libraries of CUDA kernels include
 * the following:
 * - [<b>cuBLAS</b>](https://developer.nvidia.com/cublas)
 * - [<b>cuFFT</b>](https://developer.nvidia.com/cufft)
 * - [<b>cuSPARSE</b>](https://developer.nvidia.com/cusparse)
 * - [<b>Thrust</b>](http://thrust.github.com/)
 * </td>
 * <td width="50%">
 * \htmlonly
 * <a href="kernel_abstraction.png"><center><img src="kernel_abstraction.png" width="100%"/></center></a>
 * \endhtmlonly
 * </td>
 * </tr><tr>
 * <td>
 * \par
 * <b>Thread blocks (SIMT)</b>.  Each kernel invocation comprises some number of parallel threads.  Threads
 * are grouped into blocks, and the threads within a block can communicate and synchronize with each other
 * to perform some cooperative function.  There has historically been very little reuse of cooperative SIMT
 * software within CUDA kernel.  Libraries of thread-block primitives include the following:
 * - [<b>CUB</b>](index.html)
 * </td>
 * <td>
 * \htmlonly
 * <a href="simt_abstraction.png"><center><img src="simt_abstraction.png" width="100%"/></center></a>
 * \endhtmlonly
 * </td>
 * </tr><tr>
 * <td>
 * \par
 * <b>CUDA thread (scalar)</b>.  A single CUDA thread invokes some scalar function.
 * This is the lowest level of CUDA software abstraction, and is useful when there is no
 * need to reason about the interaction of parallel threads.  CUDA libraries of
 * purely data-parallel functions include the following:
 * - [<b>CUDA Math Library</b>](https://developer.nvidia.com/cuda-math-library) (e.g., \p text1D(), \p atomicAdd(), \p popc(), etc.)
 * - [<b>cuRAND</b>](https://developer.nvidia.com/curand)'s device-code interface
 * - [<b>CUB</b>](index.html)
 * </td>
 * <td>
 * \htmlonly
 * <a href="devfun_abstraction.png"><center><img src="devfun_abstraction.png" width="100%"/></center></a>
 * \endhtmlonly
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
 * \subsection sec3sec1 6.1 &nbsp;&nbsp; C++ templates
 *
 * \par
 * As a SIMT library, CUB must be flexible enough to accommodate a wide spectrum
 * of <em>parallel execution contexts</em>,
 * i.e., specific:
 *    - Data types
 *    - Widths of parallelism (threads per block)
 *    - Grain sizes (data items per thread)
 *    - Underlying architectures (special instructions, warp size, rules for bank conflicts, etc.)
 *    - Tuning requirements (e.g., latency vs. throughput)
 *
 * \par
 * To provide this flexibility, CUB is implemented as a C++ template library.
 * C++ templates are a way to write generic algorithms and data structures.
 * There is no need to build CUB separately.  You simply <tt>\#include</tt> the
 * <tt>cub.cuh</tt> header file into your <tt>.cu</tt> CUDA C++ sources
 * and compile with NVIDIA's <tt>nvcc</tt> compiler.
 *
 * \subsection sec3sec2 6.2 &nbsp;&nbsp; Reflective type structure
 *
 * \par
 * Cooperation within a thread block requires shared memory for communicating between threads.
 * However, the specific size and layout of the memory needed by a given
 * primitive will be specific to the details of its parallel execution context (e.g., how
 * many threads are calling into it, how many items are processed per thread, etc.).  Furthermore,
 * this shared memory must be allocated outside of the component itself if it is to be
 * reused elsewhere by the thread block.
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
 * \subsection sec3sec3 6.3 &nbsp;&nbsp; Flexible data mapping
 *
 * \par
 * We often design kernels such that each thread block is assigned a "tile" of data
 * items for processing.  When the tile size equals the thread block size, the
 * mapping of data onto threads is straightforward (one datum per thread).
 * However, it is often desirable for performance reasons to process more
 * than one datum per thread.  When doing so, we must decide how
 * to partition this "tile" of items across the thread block.
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
 *   work through shared memory spaces.
 * - <b>Data occupancy</b>.  The number of items that can be resident on-chip in
 *   thread-private register storage is often greater than the number of
 *   schedulable threads.
 * - <b>Instruction-level parallelism</b>.  Multiple items per thread also
 *   facilitates greater ILP for improved throughput and utilization.
 *
 * \par
 * The cub::BlockExchange primitive provides operations for converting between blocked
 * and striped arrangements. Blocked arrangements are often desirable for
 * algorithmic benefits (where long sequences of items can be processed sequentially
 * within each thread).  Striped arrangements are often desirable for data movement
 * through global memory (where
 * [read/write coalescing](http://docs.nvidia.com/cuda/cuda-c-best-practices-guide/#coalesced-access-global-memory)</a>
 * is an important performance consideration).
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
