/******************************************************************************
 * Copyright (c) 2011-2021, NVIDIA CORPORATION.  All rights reserved.
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

#pragma once

#include "../../config.cuh"
#include "../../util_math.cuh"
#include "../../util_device.cuh"
#include "../../agent/agent_adjacent_difference.cuh"

#include <thrust/system/cuda/detail/core/triple_chevron_launch.h>

/// Optional outer namespace(s)
CUB_NS_PREFIX

/// CUB namespace
namespace cub {

template <typename AgentDifferenceInitT,
          typename InputIteratorT,
          typename OutputIteratorT,
          typename OffsetT>
void __global__ DeviceAdjacentDifferenceInitKernel(InputIteratorT first,
                                                   OutputIteratorT result,
                                                   OffsetT num_tiles,
                                                   int items_per_tile)
{
  const int tile_idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  AgentDifferenceInitT::Process(tile_idx,
                                first,
                                result,
                                num_tiles,
                                items_per_tile);
}

template <typename Policy,
          typename InputIteratorT,
          typename OutputIteratorT,
          typename FlagOpT,
          typename OffsetT,
          typename InputT,
          bool InPlace,
          bool ReadLeft>
void __global__
DeviceAdjacentDifferenceDifferenceKernel(InputIteratorT input,
                                         InputT *first_tile_previous,
                                         OutputIteratorT result,
                                         FlagOpT flag_op,
                                         OffsetT num_items)
{
  using Agent = AgentDifference<Policy,
                                InputIteratorT,
                                OutputIteratorT,
                                FlagOpT,
                                OffsetT,
                                InputT,
                                InPlace,
                                ReadLeft>;

  extern __shared__ char shmem[];
  typename Agent::TempStorage &storage =
    *reinterpret_cast<typename Agent::TempStorage *>(shmem);

  Agent agent(storage, input, first_tile_previous, result, flag_op, num_items);

  int tile_idx = static_cast<int>(blockIdx.x);
  OffsetT tile_base  = static_cast<OffsetT>(tile_idx) * Policy::ITEMS_PER_TILE;

  agent.Process(tile_idx, tile_base);
}

template <typename InputIteratorT>
struct DeviceAdjacentDifferencePolicy
{
  using ValueT = typename std::iterator_traits<InputIteratorT>::value_type;

  //------------------------------------------------------------------------------
  // Architecture-specific tuning policies
  //------------------------------------------------------------------------------

  struct Policy300 : ChainedPolicy<300, Policy300, Policy300>
  {
    using AdjacentDifferencePolicy =
      AgentAdjacentDifferencePolicy<128,
                                    Nominal8BItemsToItems<ValueT>(7),
                                    cub::BLOCK_LOAD_WARP_TRANSPOSE,
                                    cub::LOAD_DEFAULT,
                                    cub::BLOCK_STORE_WARP_TRANSPOSE>;
  };

  struct Policy350 : ChainedPolicy<350, Policy350, Policy300>
  {
    using AdjacentDifferencePolicy =
      AgentAdjacentDifferencePolicy<128,
                                    Nominal8BItemsToItems<ValueT>(7),
                                    cub::BLOCK_LOAD_WARP_TRANSPOSE,
                                    cub::LOAD_LDG,
                                    cub::BLOCK_STORE_WARP_TRANSPOSE>;
  };

  using MaxPolicy = Policy350;
};

template <typename InputIteratorT,
          typename OutputIteratorT,
          typename FlagOpT,
          typename OffsetT,
          bool InPlace,
          bool ReadLeft,
          typename SelectedPolicy =
            DeviceAdjacentDifferencePolicy<InputIteratorT>>
struct DispatchAdjacentDifference : public SelectedPolicy
{
  using ValueT = typename std::iterator_traits<InputIteratorT>::value_type;

  void *d_temp_storage;
  std::size_t &temp_storage_bytes;
  InputIteratorT d_input;
  OutputIteratorT d_output;
  OffsetT num_items;
  FlagOpT flag_op;
  cudaStream_t stream;
  bool debug_synchronous;

  CUB_RUNTIME_FUNCTION __forceinline__
  DispatchAdjacentDifference(void *d_temp_storage,
                             std::size_t &temp_storage_bytes,
                             InputIteratorT d_input,
                             OutputIteratorT d_output,
                             OffsetT num_items,
                             FlagOpT flag_op,
                             cudaStream_t stream,
                             bool debug_synchronous)
      : d_temp_storage(d_temp_storage)
      , temp_storage_bytes(temp_storage_bytes)
      , d_input(d_input)
      , d_output(d_output)
      , num_items(num_items)
      , flag_op(flag_op)
      , stream(stream)
      , debug_synchronous(debug_synchronous)
  {}

  /// Invocation
  template <typename ActivePolicyT>
  CUB_RUNTIME_FUNCTION __forceinline__ cudaError_t Invoke()
  {
    using AdjacentDifferencePolicyT =
      typename ActivePolicyT::AdjacentDifferencePolicy;

    using AgentDifferenceT = AgentDifference<AdjacentDifferencePolicyT,
                                             InputIteratorT,
                                             OutputIteratorT,
                                             FlagOpT,
                                             OffsetT,
                                             ValueT,
                                             InPlace,
                                             ReadLeft>;

    cudaError error = cudaSuccess;

    do
    {
      OffsetT tile_size = AdjacentDifferencePolicyT::ITEMS_PER_TILE;
      OffsetT num_tiles = cub::DivideAndRoundUp(num_items, tile_size);

      int shmem_size = AgentDifferenceT::SHARED_MEMORY_SIZE;

      std::size_t first_tile_previous_size = InPlace * (num_tiles) * sizeof(ValueT);

      void *allocations[1]            = {nullptr};
      std::size_t allocation_sizes[1] = {first_tile_previous_size};

      if (InPlace)
      {
        if (CubDebug(error = AliasTemporaries(d_temp_storage,
                                              temp_storage_bytes,
                                              allocations,
                                              allocation_sizes)))
        {
          break;
        }

        if (d_temp_storage == nullptr)
        {
          // Return if the caller is simply requesting the size of the storage
          // allocation

          break;
        }
      }

      auto first_tile_previous = (ValueT *)allocations[0];

      if (InPlace)
      {
        using AgentDifferenceInitT = typename If<
          ReadLeft,
          AgentDifferenceInitLeft<InputIteratorT, OutputIteratorT, OffsetT>,
          AgentDifferenceInitRight<InputIteratorT, OutputIteratorT, OffsetT>>::Type;

        const unsigned int init_block_size =
          AgentDifferenceInitT::BLOCK_THREADS;

        const unsigned int init_grid_size =
          cub::DivideAndRoundUp(num_tiles, init_block_size);

        thrust::cuda_cub::launcher::triple_chevron(init_grid_size,
                                                   init_block_size,
                                                   0,
                                                   stream)
          .doit(DeviceAdjacentDifferenceInitKernel<AgentDifferenceInitT,
                                                   InputIteratorT,
                                                   OutputIteratorT,
                                                   OffsetT>,
                d_input,
                first_tile_previous,
                num_tiles,
                tile_size);

        if (debug_synchronous)
        {
          if (CubDebug(error = SyncStream(stream)))
          {
            break;
          }
        }

        // Check for failure to launch
        if (CubDebug(error = cudaPeekAtLastError()))
        {
          break;
        }
      }

      thrust::cuda_cub::launcher::triple_chevron(
        num_tiles,
        AdjacentDifferencePolicyT::BLOCK_THREADS,
        shmem_size,
        stream)
        .doit(DeviceAdjacentDifferenceDifferenceKernel<AdjacentDifferencePolicyT,
                                                       InputIteratorT,
                                                       OutputIteratorT,
                                                       FlagOpT,
                                                       OffsetT,
                                                       ValueT,
                                                       InPlace,
                                                       ReadLeft>,
              d_input,
              first_tile_previous,
              d_output,
              flag_op,
              num_items);

      if (debug_synchronous)
      {
        if (CubDebug(error = SyncStream(stream)))
        {
          break;
        }
      }

      // Check for failure to launch
      if (CubDebug(error = cudaPeekAtLastError()))
      {
        break;
      }
    } while (0);

    return error;
  }

  CUB_RUNTIME_FUNCTION
  static cudaError_t Dispatch(void *d_temp_storage,
                              std::size_t &temp_storage_bytes,
                              InputIteratorT d_input,
                              OutputIteratorT d_output,
                              OffsetT num_items,
                              FlagOpT flag_op,
                              cudaStream_t stream,
                              bool debug_synchronous)
  {
    using MaxPolicyT = typename DispatchAdjacentDifference::MaxPolicy;

    cudaError error = cudaSuccess;
    do
    {
      // Get PTX version
      int ptx_version = 0;
      if (CubDebug(error = PtxVersion(ptx_version)))
      {
        break;
      }

      if (num_items == OffsetT(0))
      {
        break;
      }

      // Create dispatch functor
      DispatchAdjacentDifference dispatch(d_temp_storage,
                                          temp_storage_bytes,
                                          d_input,
                                          d_output,
                                          num_items,
                                          flag_op,
                                          stream,
                                          debug_synchronous);

      // Dispatch to chained policy
      if (CubDebug(error = MaxPolicyT::Invoke(ptx_version, dispatch)))
      {
        break;
      }
    } while (0);

    return error;
  }
};

}               // CUB namespace
CUB_NS_POSTFIX  // Optional outer namespace(s)
