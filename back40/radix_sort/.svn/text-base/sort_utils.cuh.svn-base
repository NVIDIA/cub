/******************************************************************************
 * 
 * Copyright (c) 2010-2012, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2012, NVIDIA CORPORATION.  All rights reserved.
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License. 
 * 
 ******************************************************************************/
/******************************************************************************
 * Common utilities for radix sorting
 ******************************************************************************/

#pragma once

#include "../ns_wrapper.cuh"

BACK40_NS_PREFIX
namespace back40 {
namespace radix_sort {



/******************************************************************************
 * BinDescriptor descriptor
 ******************************************************************************/

enum BinState
{
    INVALID,
    UPSWEEP,
    SCAN,
    DOWNSWEEP,
    LOCAL
};


/**
 * BinDescriptor descriptor
 */
template <typename SizeT>
struct BinDescriptor
{
    int             buffer_selector;        // Which buffer the bin lies in
    SizeT           buffer_offset;          // Offset from beginning of buffer
    SizeT           num_items;              // Number of keys in the bin
    unsigned int    current_bit;            // The current (most-significant) bit
    BinState        bin_state;              // The status of this bin
};



/******************************************************************************
 * Traits for converting for converting signed and floating point types
 * to unsigned types suitable for radix sorting
 ******************************************************************************/

/**
 * Specialization for unsigned signed integers
 */
template <typename _UnsignedBits>
struct UnsignedKeyTraits
{
    typedef _UnsignedBits UnsignedBits;

    static const UnsignedBits MIN_KEY = UnsignedBits(0);
    static const UnsignedBits MAX_KEY = UnsignedBits(-1);

    static __device__ __forceinline__ UnsignedBits TwiddleIn(UnsignedBits key)
    {
        return key;
    }

    static __device__ __forceinline__ UnsignedBits TwiddleOut(UnsignedBits key)
    {
        return key;
    }
};


/**
 * Specialization for signed integers
 */
template <typename _UnsignedBits>
struct SignedKeyTraits
{
    typedef _UnsignedBits UnsignedBits;

    static const UnsignedBits HIGH_BIT = UnsignedBits(1) << ((sizeof(UnsignedBits) * 8) - 1);
    static const UnsignedBits MIN_KEY = HIGH_BIT;
    static const UnsignedBits MAX_KEY = UnsignedBits(-1) ^ HIGH_BIT;

    static __device__ __forceinline__ UnsignedBits TwiddleIn(UnsignedBits key)
    {
        return key ^ HIGH_BIT;
    };

    static __device__ __forceinline__ UnsignedBits TwiddleOut(UnsignedBits key)
    {
        return key ^ HIGH_BIT;
    };
};


/**
 * Specialization for floating point
 */
template <typename _UnsignedBits>
struct FloatKeyTraits
{
    typedef _UnsignedBits     UnsignedBits;

    static const UnsignedBits HIGH_BIT = UnsignedBits(1) << ((sizeof(UnsignedBits) * 8) - 1);
    static const UnsignedBits MIN_KEY = UnsignedBits(-1);
    static const UnsignedBits MAX_KEY = UnsignedBits(-1) ^ HIGH_BIT;

    static __device__ __forceinline__ UnsignedBits TwiddleIn(UnsignedBits key)
    {
        UnsignedBits mask = (key & HIGH_BIT) ? UnsignedBits(-1) : HIGH_BIT;
        return key ^ mask;
    };

    static __device__ __forceinline__ UnsignedBits TwiddleOut(UnsignedBits key)
    {
        UnsignedBits mask = (key & HIGH_BIT) ? HIGH_BIT : UnsignedBits(-1);
        return key ^ mask;
    };

};




// Default unsigned types
template <typename T>
struct KeyTraits : UnsignedKeyTraits<T> {};

// char
template <> struct KeyTraits<char> : SignedKeyTraits<unsigned char> {};

// signed char
template <> struct KeyTraits<signed char> : SignedKeyTraits<unsigned char> {};

// short
template <> struct KeyTraits<short> : SignedKeyTraits<unsigned short> {};

// int
template <> struct KeyTraits<int> : SignedKeyTraits<unsigned int> {};

// long
template <> struct KeyTraits<long> : SignedKeyTraits<unsigned long> {};

// long long
template <> struct KeyTraits<long long> : SignedKeyTraits<unsigned long long> {};

// float
template <> struct KeyTraits<float> : FloatKeyTraits<unsigned int> {};

// double
template <> struct KeyTraits<double> : FloatKeyTraits<unsigned long long> {};




} // namespace radix_sort
} // namespace back40
BACK40_NS_POSTFIX
