/******************************************************************************
 * Copyright (c) 2011-2020, NVIDIA CORPORATION.  All rights reserved.
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
 * radix_rank_sort_operations.cuh contains common abstractions, definitions and
 * operations used for radix sorting and ranking.
 */

#pragma once

#include "../config.cuh"
#include "../util_type.cuh"


/// Optional outer namespace(s)
CUB_NS_PREFIX

/// CUB namespace
namespace cub {

/** \brief Twiddling keys for radix sort. */
template <bool IS_DESCENDING, typename KeyT>
struct RadixSortTwiddle
{
    typedef Traits<KeyT> TraitsT;
    typedef typename TraitsT::UnsignedBits UnsignedBits;
    static __host__ __device__ __forceinline__ UnsignedBits In(UnsignedBits key)
    {
        key = TraitsT::TwiddleIn(key);
        if (IS_DESCENDING) key = ~key;
        return key;
    }
    static __host__ __device__ __forceinline__ UnsignedBits Out(UnsignedBits key)
    {
        if (IS_DESCENDING) key = ~key;
        key = TraitsT::TwiddleOut(key);
        return key;
    }
    static __host__ __device__ __forceinline__ UnsignedBits DefaultKey()
    {
        return Out(~UnsignedBits(0));
    }
};


/** \brief Stateful abstraction to extract digits. */
template <typename UnsignedBits>
struct DigitExtractor
{
    int current_bit, mask;
    __device__ __forceinline__ DigitExtractor() : current_bit(0), mask(0) {}
    __device__ __forceinline__ DigitExtractor(int current_bit, int num_bits)
        : current_bit(current_bit), mask((1 << num_bits) - 1)
    { }

    __device__ __forceinline__ int Digit(UnsignedBits key)
    {
        return int(key >> UnsignedBits(current_bit)) & mask;
    }
};

}               // CUB namespace
CUB_NS_POSTFIX  // Optional outer namespace(s)
