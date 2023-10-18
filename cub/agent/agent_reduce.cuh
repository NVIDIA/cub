/******************************************************************************
 * Copyright (c) 2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2022, NVIDIA CORPORATION.  All rights reserved.
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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" 
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE 
 * ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

/**
 * @file cub::AgentReduce implements a stateful abstraction of CUDA thread 
 *       blocks for participating in device-wide reduction.
 */

#pragma once

#include <iterator>

#include <cub/block/block_load.cuh>
#include <cub/block/block_reduce.cuh>
#include <cub/config.cuh>
#include <cub/detail/type_traits.cuh>
#include <cub/grid/grid_even_share.cuh>
#include <cub/grid/grid_mapping.cuh>
#include <cub/iterator/cache_modified_input_iterator.cuh>
#include <cub/util_type.cuh>

CUB_NAMESPACE_BEGIN

/******************************************************************************
 * Tuning policy types
 ******************************************************************************/

/**
 * Parameterizable tuning policy type for AgentReduce
 * @tparam NOMINAL_BLOCK_THREADS_4B Threads per thread block
 * @tparam NOMINAL_ITEMS_PER_THREAD_4B Items per thread (per tile of input)
 * @tparam ComputeT Dominant compute type
 * @tparam _VECTOR_LOAD_LENGTH Number of items per vectorized load
 * @tparam _BLOCK_ALGORITHM Cooperative block-wide reduction algorithm to use
 * @tparam _LOAD_MODIFIER Cache load modifier for reading input elements
 */
template <int NOMINAL_BLOCK_THREADS_4B,
          int NOMINAL_ITEMS_PER_THREAD_4B,
          typename ComputeT,
          int _VECTOR_LOAD_LENGTH,
          BlockReduceAlgorithm _BLOCK_ALGORITHM,
          CacheLoadModifier _LOAD_MODIFIER,
          typename ScalingType = MemBoundScaling<NOMINAL_BLOCK_THREADS_4B,
                                                 NOMINAL_ITEMS_PER_THREAD_4B,
                                                 ComputeT>>
struct AgentReducePolicy : ScalingType
{
  /// Number of items per vectorized load
  static constexpr int VECTOR_LOAD_LENGTH = _VECTOR_LOAD_LENGTH;

  /// Cooperative block-wide reduction algorithm to use
  static constexpr BlockReduceAlgorithm BLOCK_ALGORITHM = _BLOCK_ALGORITHM;

  /// Cache load modifier for reading input elements
  static constexpr CacheLoadModifier LOAD_MODIFIER = _LOAD_MODIFIER;
};

template <int BLOCK_THREADS,
          int NOMINAL_WARP_THREADS_4B,
          int NOMINAL_ITEMS_PER_THREAD_4B,
          typename ComputeT,
          int _VECTOR_LOAD_LENGTH,
          CacheLoadModifier _LOAD_MODIFIER>
struct AgentWarpReducePolicy 
{
  // TODO MemBoundScaling-like computation
  static constexpr int ITEMS_PER_THREAD = NOMINAL_ITEMS_PER_THREAD_4B;

  static constexpr int WARP_THREADS = NOMINAL_WARP_THREADS_4B;

  /// Number of items per vectorized load
  static constexpr int VECTOR_LOAD_LENGTH = _VECTOR_LOAD_LENGTH;

  /// Cache load modifier for reading input elements
  static constexpr CacheLoadModifier LOAD_MODIFIER = _LOAD_MODIFIER;

  constexpr static int ITEMS_PER_TILE = ITEMS_PER_THREAD * WARP_THREADS; 

  constexpr static int SEGMENTS_PER_BLOCK = BLOCK_THREADS / WARP_THREADS;  

  static_assert((BLOCK_THREADS % WARP_THREADS) == 0, "Block should be multiple of warp");
};

/******************************************************************************
 * Thread block abstractions
 ******************************************************************************/

/**
 * @brief AgentReduce implements a stateful abstraction of CUDA thread blocks
 *        for participating in device-wide reduction .
 *
 * Each thread reduces only the values it loads. If `FIRST_TILE`, this partial
 * reduction is stored into `thread_aggregate`. Otherwise it is accumulated
 * into `thread_aggregate`.
 *
 * @tparam AgentReducePolicy
 *   Parameterized AgentReducePolicy tuning policy type
 *
 * @tparam InputIteratorT
 *   Random-access iterator type for input
 *
 * @tparam OutputIteratorT
 *   Random-access iterator type for output
 *
 * @tparam OffsetT
 *   Signed integer type for global offsets
 *
 * @tparam ReductionOp
 *   Binary reduction operator type having member
 *   `auto operator()(T &&a, U &&b)`
 *
 * @tparam AccumT
 *   The type of intermediate accumulator (according to P2322R6)
 */
template <typename AgentReducePolicy,
          typename InputIteratorT,
          typename OutputIteratorT,
          typename OffsetT,
          typename ReductionOp,
          typename AccumT,
          typename CollectiveReduceT,
          int THREADS>
struct AgentReduceImpl
{
  //---------------------------------------------------------------------
  // Types and constants
  //---------------------------------------------------------------------

  /// The input value type
  using InputT = cub::detail::value_t<InputIteratorT>;

  /// Vector type of InputT for data movement
  using VectorT =
    typename CubVector<InputT, AgentReducePolicy::VECTOR_LOAD_LENGTH>::Type;

  /// Input iterator wrapper type (for applying cache modifier)
  // Wrap the native input pointer with CacheModifiedInputIterator
  // or directly use the supplied input iterator type
  using WrappedInputIteratorT = cub::detail::conditional_t<
    std::is_pointer<InputIteratorT>::value,
    CacheModifiedInputIterator<AgentReducePolicy::LOAD_MODIFIER, InputT, OffsetT>,
    InputIteratorT>;

  /// Constants
  static constexpr int ITEMS_PER_THREAD   = AgentReducePolicy::ITEMS_PER_THREAD;
  static constexpr int TILE_ITEMS         = THREADS * ITEMS_PER_THREAD;
  static constexpr int VECTOR_LOAD_LENGTH =
    CUB_MIN(ITEMS_PER_THREAD, AgentReducePolicy::VECTOR_LOAD_LENGTH);

  // Can vectorize according to the policy if the input iterator is a native
  // pointer to a primitive type
  static constexpr bool ATTEMPT_VECTORIZATION = (VECTOR_LOAD_LENGTH > 1) &&
                            (ITEMS_PER_THREAD % VECTOR_LOAD_LENGTH == 0) &&
                            (std::is_pointer<InputIteratorT>::value) &&
                            Traits<InputT>::PRIMITIVE;

  static constexpr CacheLoadModifier LOAD_MODIFIER =
    AgentReducePolicy::LOAD_MODIFIER;

  /// Shared memory type required by this thread block
  struct _TempStorage
  {
    typename CollectiveReduceT::TempStorage reduce;
  };

  /// Alias wrapper allowing storage to be unioned
  struct TempStorage : Uninitialized<_TempStorage>
  {};

  //---------------------------------------------------------------------
  // Per-thread fields
  //---------------------------------------------------------------------

  _TempStorage &temp_storage;         ///< Reference to temp_storage
  unsigned int lane_id;
  WrappedInputIteratorT d_wrapped_in; ///< Wrapped input data to reduce
  ReductionOp reduction_op;           ///< Binary reduction operator
  InputIteratorT d_in;                ///< Input data to reduce

  //---------------------------------------------------------------------
  // Utility
  //---------------------------------------------------------------------

  // Whether or not the input is aligned with the vector type (specialized for
  // types we can vectorize)
  template <typename Iterator>
  static __device__ __forceinline__ bool
  IsAligned(Iterator d_in, Int2Type<true> /*can_vectorize*/)
  {
    return (size_t(d_in) & (sizeof(VectorT) - 1)) == 0;
  }

  // Whether or not the input is aligned with the vector type (specialized for
  // types we cannot vectorize)
  template <typename Iterator>
  static __device__ __forceinline__ bool
  IsAligned(Iterator /*d_in*/, Int2Type<false> /*can_vectorize*/)
  {
    return false;
  }

  //---------------------------------------------------------------------
  // Constructor
  //---------------------------------------------------------------------

  /**
   * @brief Constructor
   * @param temp_storage Reference to temp_storage
   * @param d_in Input data to reduce
   * @param reduction_op Binary reduction operator
   */
  __device__ __forceinline__ AgentReduceImpl(TempStorage &temp_storage,
                                             InputIteratorT d_in,
                                             ReductionOp reduction_op,
                                             unsigned int lane_id)
      : temp_storage(temp_storage.Alias())
      , d_in(d_in)
      , d_wrapped_in(d_in)
      , reduction_op(reduction_op)
      , lane_id(lane_id)
  {}

  //---------------------------------------------------------------------
  // Tile consumption
  //---------------------------------------------------------------------

  /**
   * @brief Consume a full tile of input (non-vectorized)
   * @param block_offset The offset the tile to consume
   * @param valid_items The number of valid items in the tile
   * @param is_full_tile Whether or not this is a full tile
   * @param can_vectorize Whether or not we can vectorize loads
   */
  template <int IS_FIRST_TILE>
  __device__ __forceinline__ void ConsumeTile(AccumT &thread_aggregate,
                                              OffsetT block_offset,
                                              int /*valid_items*/,
                                              Int2Type<true> /*is_full_tile*/,
                                              Int2Type<false> /*can_vectorize*/)
  {
    AccumT items[ITEMS_PER_THREAD];

    // Load items in striped fashion
    LoadDirectStriped<THREADS>(lane_id,
                               d_wrapped_in + block_offset,
                               items);

    // Reduce items within each thread stripe
    thread_aggregate =
      (IS_FIRST_TILE)
        ? internal::ThreadReduce(items, reduction_op)
        : internal::ThreadReduce(items, reduction_op, thread_aggregate);
  }

  /**
   * Consume a full tile of input (vectorized)
   * @param block_offset The offset the tile to consume
   * @param valid_items The number of valid items in the tile
   * @param is_full_tile Whether or not this is a full tile
   * @param can_vectorize Whether or not we can vectorize loads
   */
  template <int IS_FIRST_TILE>
  __device__ __forceinline__ void ConsumeTile(AccumT &thread_aggregate,
                                              OffsetT block_offset,
                                              int /*valid_items*/,
                                              Int2Type<true> /*is_full_tile*/,
                                              Int2Type<true> /*can_vectorize*/)
  {
    // Alias items as an array of VectorT and load it in striped fashion
    enum
    {
      WORDS = ITEMS_PER_THREAD / VECTOR_LOAD_LENGTH
    };

    // Fabricate a vectorized input iterator
    InputT *d_in_unqualified = const_cast<InputT *>(d_in) + block_offset +
                               (lane_id * VECTOR_LOAD_LENGTH);
    CacheModifiedInputIterator<AgentReducePolicy::LOAD_MODIFIER, VectorT, OffsetT>
      d_vec_in(reinterpret_cast<VectorT *>(d_in_unqualified));

    // Load items as vector items
    InputT input_items[ITEMS_PER_THREAD];
    VectorT *vec_items = reinterpret_cast<VectorT *>(input_items);
#pragma unroll
    for (int i = 0; i < WORDS; ++i)
    {
      vec_items[i] = d_vec_in[THREADS * i];
    }

    // Convert from input type to output type
    AccumT items[ITEMS_PER_THREAD];
#pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; ++i)
    {
      items[i] = input_items[i];
    }

    // Reduce items within each thread stripe
    thread_aggregate =
      (IS_FIRST_TILE)
        ? internal::ThreadReduce(items, reduction_op)
        : internal::ThreadReduce(items, reduction_op, thread_aggregate);
  }

  /**
   * Consume a partial tile of input
   * @param block_offset The offset the tile to consume
   * @param valid_items The number of valid items in the tile
   * @param is_full_tile Whether or not this is a full tile
   * @param can_vectorize Whether or not we can vectorize loads
   */
  template <int IS_FIRST_TILE, int CAN_VECTORIZE>
  __device__ __forceinline__ void
  ConsumeTile(AccumT &thread_aggregate,
              OffsetT block_offset,
              int valid_items,
              Int2Type<false> /*is_full_tile*/,
              Int2Type<CAN_VECTORIZE> /*can_vectorize*/)
  {
    // Partial tile
    int thread_offset = lane_id;

    // Read first item
    if ((IS_FIRST_TILE) && (thread_offset < valid_items))
    {
      thread_aggregate = d_wrapped_in[block_offset + thread_offset];
      thread_offset += THREADS;
    }

    // Continue reading items (block-striped)
    while (thread_offset < valid_items)
    {
      InputT item(d_wrapped_in[block_offset + thread_offset]);

      thread_aggregate = reduction_op(thread_aggregate, item);
      thread_offset += THREADS;
    }
  }

  //---------------------------------------------------------------
  // Consume a contiguous segment of tiles
  //---------------------------------------------------------------------

  /**
   * @brief Reduce a contiguous segment of input tiles
   * @param even_share GridEvenShare descriptor
   * @param can_vectorize Whether or not we can vectorize loads
   */
  template <int CAN_VECTORIZE>
  __device__ __forceinline__ AccumT
  ConsumeRange(GridEvenShare<OffsetT> &even_share,
               Int2Type<CAN_VECTORIZE> can_vectorize)
  {
    AccumT thread_aggregate{};

    if (even_share.block_end - even_share.block_offset < TILE_ITEMS)
    {
      // First tile isn't full (not all threads have valid items)
      int valid_items = even_share.block_end - even_share.block_offset;
      ConsumeTile<true>(thread_aggregate,
                        even_share.block_offset,
                        valid_items,
                        Int2Type<false>(),
                        can_vectorize);

      // TODO Extract clamping into the SFINAE to keep block version as is
      int num_valid = (THREADS <= valid_items) ? THREADS : valid_items;

      return CollectiveReduceT(temp_storage.reduce)
        .Reduce(thread_aggregate, reduction_op, num_valid);
    }

    // Extracting this into a function saves 8% of generated kernel size by allowing to reuse 
    // the block reduction below. This also workaround hang in nvcc.
    ConsumeFullTileRange(thread_aggregate, even_share, can_vectorize);

    // Compute block-wide reduction (all threads have valid items)
    return CollectiveReduceT(temp_storage.reduce)
      .Reduce(thread_aggregate, reduction_op);
  }

  /**
   * @brief Reduce a contiguous segment of input tiles
   * @param[in] block_offset Threadblock begin offset (inclusive)
   * @param[in] block_end Threadblock end offset (exclusive)
   */
  __device__ __forceinline__ AccumT ConsumeRange(OffsetT block_offset,
                                                 OffsetT block_end)
  {
    GridEvenShare<OffsetT> even_share;
    even_share.template BlockInit<TILE_ITEMS>(block_offset, block_end);

    return (IsAligned(d_in + block_offset, Int2Type<ATTEMPT_VECTORIZATION>()))
             ? ConsumeRange(even_share,
                            Int2Type < true && ATTEMPT_VECTORIZATION > ())
             : ConsumeRange(even_share,
                            Int2Type < false && ATTEMPT_VECTORIZATION > ());
  }

  /**
   * Reduce a contiguous segment of input tiles
   * @param[in] even_share GridEvenShare descriptor
   */
  __device__ __forceinline__ AccumT
  ConsumeTiles(GridEvenShare<OffsetT> &even_share)
  {
    // Initialize GRID_MAPPING_STRIP_MINE even-share descriptor for this thread block
    even_share.template BlockInit<TILE_ITEMS, GRID_MAPPING_STRIP_MINE>();

    return (IsAligned(d_in, Int2Type<ATTEMPT_VECTORIZATION>()))
             ? ConsumeRange(even_share,
                            Int2Type < true && ATTEMPT_VECTORIZATION > ())
             : ConsumeRange(even_share,
                            Int2Type < false && ATTEMPT_VECTORIZATION > ());
  }

private:
  /**
   * @brief Reduce a contiguous segment of input tiles with more than `TILE_ITEMS` elements
   * @param even_share GridEvenShare descriptor
   * @param can_vectorize Whether or not we can vectorize loads
   */
  template <int CAN_VECTORIZE>
  __device__ __forceinline__ void
  ConsumeFullTileRange(AccumT &thread_aggregate,
                       GridEvenShare<OffsetT> &even_share,
                       Int2Type<CAN_VECTORIZE> can_vectorize)
  {
    // At least one full block
    ConsumeTile<true>(thread_aggregate,
                      even_share.block_offset,
                      TILE_ITEMS,
                      Int2Type<true>(),
                      can_vectorize);

    if (even_share.block_end - even_share.block_offset < even_share.block_stride)
    {
      // Exit early to handle offset overflow
      return;
    }

    even_share.block_offset += even_share.block_stride;

    // Consume subsequent full tiles of input, at least one full tile was processed, so 
    // `even_share.block_end >= TILE_ITEMS`
    while (even_share.block_offset <= even_share.block_end - TILE_ITEMS)
    {
      ConsumeTile<false>(thread_aggregate,
                         even_share.block_offset,
                         TILE_ITEMS,
                         Int2Type<true>(),
                         can_vectorize);

      if (even_share.block_end - even_share.block_offset < even_share.block_stride)
      {
        // Exit early to handle offset overflow
        return;
      }

      even_share.block_offset += even_share.block_stride;
    }

    // Consume a partially-full tile
    if (even_share.block_offset < even_share.block_end)
    {
      int valid_items = even_share.block_end - even_share.block_offset;
      ConsumeTile<false>(thread_aggregate,
                         even_share.block_offset,
                         valid_items,
                         Int2Type<false>(),
                         can_vectorize);
    }
  }
};

template <typename AgentReducePolicy,
          typename InputIteratorT,
          typename OutputIteratorT,
          typename OffsetT,
          typename ReductionOp,
          typename AccumT>
struct AgentReduce : AgentReduceImpl<AgentReducePolicy, 
                                     InputIteratorT, 
                                     OutputIteratorT, 
                                     OffsetT, 
                                     ReductionOp, 
                                     AccumT, 
                                     BlockReduce<AccumT, 
                                                 AgentReducePolicy::BLOCK_THREADS, 
                                                 AgentReducePolicy::BLOCK_ALGORITHM>,
                                     AgentReducePolicy::BLOCK_THREADS>
{
  using base_t = AgentReduceImpl<AgentReducePolicy, 
                                     InputIteratorT, 
                                     OutputIteratorT, 
                                     OffsetT, 
                                     ReductionOp, 
                                     AccumT, 
                                     BlockReduce<AccumT, 
                                                 AgentReducePolicy::BLOCK_THREADS, 
                                                 AgentReducePolicy::BLOCK_ALGORITHM>,
                                     AgentReducePolicy::BLOCK_THREADS>;

  __device__ __forceinline__ AgentReduce(typename base_t::TempStorage &temp_storage,
                                         InputIteratorT d_in,
                                         ReductionOp reduction_op)
    : base_t(temp_storage, d_in, reduction_op, threadIdx.x) 
  {
  }
};

template <typename AgentReducePolicy,
          typename InputIteratorT,
          typename OutputIteratorT,
          typename OffsetT,
          typename ReductionOp,
          typename AccumT>
struct AgentWarpReduce : AgentReduceImpl<AgentReducePolicy, 
                                         InputIteratorT, 
                                         OutputIteratorT, 
                                         OffsetT, 
                                         ReductionOp, 
                                         AccumT, 
                                         WarpReduce<AccumT, 
                                                    AgentReducePolicy::WARP_THREADS>,
                                         AgentReducePolicy::WARP_THREADS>
{
  using base_t = AgentReduceImpl<AgentReducePolicy, 
                                     InputIteratorT, 
                                     OutputIteratorT, 
                                     OffsetT, 
                                     ReductionOp, 
                                     AccumT, 
                                     WarpReduce<AccumT, 
                                                AgentReducePolicy::WARP_THREADS>,
                                     AgentReducePolicy::WARP_THREADS>;

  __device__ __forceinline__ AgentWarpReduce(typename base_t::TempStorage &temp_storage,
                                             InputIteratorT d_in,
                                             ReductionOp reduction_op,
                                             int lane_id)
    : base_t(temp_storage, d_in, reduction_op, lane_id) 
  {
  }
};

CUB_NAMESPACE_END

