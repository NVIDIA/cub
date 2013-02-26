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

// Threadblock
#include "block/block_load.cuh"
#include "block/block_even_share.cuh"
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
 * \section sec1 (1) What is CUB?
 *
 * \par
 * CUB is a library of SIMT primitives for CUDA kernel
 * programming. CUB enhances productivity and portability
 * by providing commonplace threadblock-wide, warp-wide, and thread-level operations that
 * are flexible and tunable to fit your kernel needs.
 *
 * \par
 * Browse our collections of:
 * - [<b>SIMT cooperative primitives</b>](annotated.html)
 *   - BlockRadixSort, BlockReduce, WarpScan, etc.
 * - [<b>SIMT utilities</b>](group___simt_utils.html)
 *   - threadblock loads/stores in blocked/striped arrangements (vectorized, coalesced, etc.)
 *   - Sequential ThreadScan, ThreadReduce, etc.
 *   - Cache-modified ThreadLoad/ThreadStore
 * - [<b>Host utilities</b>](group___host_util.html)
 *   - Caching allocators, error handling, etc.
 *
 * \section sec2 (2) A simple example
 *
 * \par
 * The following snippet illustrates the simplicity of using CUB primitives
 * to construct a CUDA kernel for computing prefix sum:
 *
 * \par
 * \code
 * #include <cub.cuh>
 *
 * // An exclusive prefix sum CUDA kernel (for a single-threadblock grid)
 * template <
 *      int         BLOCK_THREADS,              // Threads per threadblock
 *      int         ITEMS_PER_THREAD,           // Items per thread
 *      typename    T>                          // Data type
 * __global__ void PrefixSumKernel(T *d_in, T *d_out)
 * {
 *      using namespace cub;
 *
 *      // Parameterize a BlockScan type for use in the current execution context
 *      typedef BlockScan<T, BLOCK_THREADS> BlockScan;
 *
 *      // The shared memory needed by the cooperative BlockScan
 *      __shared__ typename BlockScan::SmemStorage smem_storage;
 *
 *      // A segment of data items per thread
 *      T data[ITEMS_PER_THREAD];
 *
 *      // Load a tile of data using vector-load instructions
 *      BlockLoadVectorized(data, d_in, 0);
 *
 *      // Perform an exclusive prefix sum across the tile of data
 *      BlockScan::ExclusiveSum(smem_storage, data, data);
 *
 *      // Store a tile of data using vector-load instructions
 *      BlockStoreVectorized(data, d_out, 0);
 * }
 * \endcode
 *
 * \par
 * The cub::BlockScan type performs a cooperative prefix sum across the
 * threadblock's data items.  Its implementation is parameterized by the number of threadblock threads and the aggregate
 * data type \p T, and is specialized for the underlying architecture.  Once
 * instantiated, cub::BlockScan exposes the opaque cub::BlockScan::SmemStorage
 * type which allows us to allocate the shared memory needed by the primitive.
 *
 * \par
 * Furthermore, the kernel uses CUB's primitives for vectorizing global
 * loads and stores.  For example, <tt>ld.global.v4.s32</tt> will be generated when
 * \p T = \p int and \p ITEMS_PER_THREAD is a multiple of 4.
 *
 * \section sec3 (3) Why do you need CUB?
 *
 * \par
 * With the exception of CUB, there are few (if any) software libraries of
 * reusable threadblock-level primitives.  This is unfortunate, especially for
 * complex algorithms with intricate dependences between threads.  For cooperative
 * problems, the SIMT kernel is often the most complex and performance-sensitive
 * layer in the CUDA software stack.  Best practices would have us
 * leverage libraries and abstraction layers to help  mitigate the complexity,
 * risks, and maintenance costs of this software.
 *
 * \par
 * As a SIMT library and software abstraction layer, CUB gives you:
 * -# <b>The ease of sequential programming.</b>  Parallel primitives within
 * kernels can be simply sequenced together (similar to Thrust programming on
 * the host).
 * -# <b>The benefits of transparent performance-portability.</b> Kernels can
 * be simply recompiled against new CUB releases (instead of hand-rewritten)
 * to leverage new algorithmic developments, hardware instructions, etc.
 *
 * \section sec4 (4) How does CUB work?
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
 * of <em>execution contexts</em>,
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
 * primitive will be specific to the details of its execution context (e.g., how
 * many threads are calling into it, how many items per thread, etc.).  Furthermore,
 * this shared memory must be allocated externally to the component if it is to be
 * reused elsewhere by the threadblock.
 *
 * \par
 * \code
 * // Parameterize a BlockRadixSort type for use with 128 threads
 * // and 4 items per thread
 * typedef cub::BlockRadixSort<unsigned int, 128, 4> BlockRadixSort;
 *
 * // Declare shared memory for BlockRadixSort
 * __shared__ typename BlockRadixSort::SmemStorage smem_storage;
 *
 * // A segment of consecutive input items per thread
 * int keys[4];
 *
 * // Obtain keys in blocked order
 * ...
 *
 * // Sort keys in ascending order
 * BlockRadixSort::SortBlocked(smem_storage, keys);
 *
 * \endcode
 *
 * \par
 * To address this issue, we encapsulate cooperative procedures within
 * <em>reflective type structure</em> (C++ classes).  As illustrated in the
 * cub::BlockRadixSort example above, these primitives are C++ classes with
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
 * <center><b>Blocked vs. striped arrangements with \p BLOCK_THREADS = 4 and \p ITEMS_PER_THREAD = 2, emphasis on items owned by <em>thread</em><sub>0</sub></b></center>
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
