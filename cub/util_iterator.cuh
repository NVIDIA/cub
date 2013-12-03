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
 * Random-access iterator types
 */

#pragma once

#include <iterator>
#include <iostream>

#include "thread/thread_load.cuh"
#include "thread/thread_store.cuh"
#include "util_device.cuh"
#include "util_debug.cuh"
#include "util_namespace.cuh"

/// Optional outer namespace(s)
CUB_NS_PREFIX

/// CUB namespace
namespace cub {


/******************************************************************************
 * Static file-scope Tesla/Fermi-style texture references
 *****************************************************************************/

#ifndef DOXYGEN_SHOULD_SKIP_THIS    // Do not document

// Anonymous namespace
namespace {

/// Global texture reference specialized by type
template <typename T>
struct IteratorTexRef
{
    /// And by unique ID
    template <int UNIQUE_ID>
    struct TexId
    {
        // Largest texture word we can use in device
        typedef typename UnitWord<T>::DeviceWord DeviceWord;
        typedef typename UnitWord<T>::TextureWord TextureWord;

        // Number of texture words per T
        enum {
            DEVICE_MULTIPLE = sizeof(T) / sizeof(DeviceWord),
            TEXTURE_MULTIPLE = sizeof(T) / sizeof(TextureWord)
        };

        // Texture reference type
        typedef texture<TextureWord> TexRef;

        // Texture reference
        static TexRef ref;

        /// Bind texture
        static cudaError_t BindTexture(void *d_in)
        {
            printf("binding texture element of size %d to %d words of size %d\n",
                sizeof(T),
                TEXTURE_MULTIPLE,
                sizeof(TextureWord));

            if (d_in)
            {
                cudaChannelFormatDesc tex_desc = cudaCreateChannelDesc<TextureWord>();
                return (CubDebug(cudaBindTexture(NULL, ref, d_in, tex_desc)));
            }

            return cudaSuccess;
        }

        /// Unbind texture
        static cudaError_t UnbindTexture()
        {
            return CubDebug(cudaUnbindTexture(ref));
        }

        /// Fetch element
        template <typename Distance>
        static __device__ __forceinline__ T Fetch(Distance offset)
        {
            DeviceWord temp[DEVICE_MULTIPLE];
            TextureWord *words = reinterpret_cast<TextureWord*>(temp);

            #pragma unroll
            for (int i = 0; i < TEXTURE_MULTIPLE; ++i)
            {
                words[i] = tex1Dfetch(ref, (offset * TEXTURE_MULTIPLE) + i);
            }

            return reinterpret_cast<T&>(temp);
        }
    };
};

// Texture reference definitions
template <typename  T>
template <int       UNIQUE_ID>
typename IteratorTexRef<T>::template TexId<UNIQUE_ID>::TexRef IteratorTexRef<T>::template TexId<UNIQUE_ID>::ref = 0;


} // Anonymous namespace


#endif // DOXYGEN_SHOULD_SKIP_THIS







/**
 * \addtogroup UtilIterator
 * @{
 */


/******************************************************************************
 * Iterators
 *****************************************************************************/

/**
 * \brief A random-access input generator for dereferencing a sequence of homogeneous values
 *
 * \par Overview
 * - Read references to a ConstantInputIterator iterator always return the supplied constant
 *   of type \p ValueType.
 * - Can be used with any data type.
 * - Can be constructed, manipulated, dereferenced, and exchanged within and between host and device
 *   functions.
 *
 * \par Example
 * The code snippet below illustrates the use of \p ConstantInputIterator to
 * dereference a sequence of homogeneous doubles.
 * \par
 * \code
 * #include <cub/cub.cuh>
 *
 * cub::ConstantInputIterator<double> itr(5.0);
 *
 * printf("%f\n", itr[0]);      // 5.0
 * printf("%f\n", itr[1]);      // 5.0
 * printf("%f\n", itr[2]);      // 5.0
 * printf("%f\n", itr[50]);     // 5.0
 *
 * \endcode
 *
 * \tparam ValueType            The value type of this iterator
 * \tparam Offset               The difference type of this iterator (Default: \p ptrdiff_t)
 */
template <
    typename ValueType,
    typename Offset = ptrdiff_t>
class ConstantInputIterator
{
public:

    // Required iterator traits
    typedef ConstantInputIterator               self_type;              ///< My own type
    typedef Offset                              difference_type;        ///< Type to express the result of subtracting one iterator from another
    typedef ValueType                           value_type;             ///< The type of the element the iterator can point to
    typedef ValueType*                          pointer;                ///< The type of a pointer to an element the iterator can point to
    typedef ValueType                           reference;              ///< The type of a reference to an element the iterator can point to
    typedef std::random_access_iterator_tag     iterator_category;      ///< The iterator category

private:

    ValueType val;

public:

    /// Constructor
    __host__ __device__ __forceinline__ ConstantInputIterator(
        const ValueType &val)          ///< Constant value for the iterator instance to report
    :
        val(val)
    {}

    /// Postfix increment
    __host__ __device__ __forceinline__ self_type operator++(int)
    {
        self_type retval = *this;
        return retval;
    }

    /// Indirection
    __host__ __device__ __forceinline__ reference operator*()
    {
        return val;
    }

    /// Addition
    template <typename Distance>
    __host__ __device__ __forceinline__ self_type operator+(Distance n)
    {
        return self_type(val);
    }

    /// Addition assignment
    template <typename Distance>
    __host__ __device__ __forceinline__ self_type& operator+=(Distance n)
    {
        return *this;
    }

    /// Subtraction
    template <typename Distance>
    __host__ __device__ __forceinline__ self_type operator-(Distance n)
    {
        return self_type(val);
    }

    /// Subtraction assignment
    template <typename Distance>
    __host__ __device__ __forceinline__ self_type& operator-=(Distance n)
    {
        return *this;
    }

    /// Array subscript
    template <typename Distance>
    __host__ __device__ __forceinline__ reference operator[](Distance n)
    {
        return val;
    }

    /// Structure dereference
    __host__ __device__ __forceinline__ pointer operator->()
    {
        return &val;
    }

    /// Equal to
    __host__ __device__ __forceinline__ bool operator==(const self_type& rhs)
    {
        return (val == rhs.val);
    }

    /// Not equal to
    __host__ __device__ __forceinline__ bool operator!=(const self_type& rhs)
    {
        return (val != rhs.val);
    }

    /// ostream operator
    friend std::ostream& operator<<(std::ostream& os, const self_type& itr)
    {
        return os;
    }

};



/**
 * \brief A random-access input generator for dereferencing a sequence of incrementing integer values.
 *
 * \par Overview
 * - After initializing a CountingInputIterator to a certain integer \p base, read references
 *   at \p offset will return the value \p base + \p offset.
 * - Can be constructed, manipulated, dereferenced, and exchanged within and between host and device
 *   functions.
 *
 * \par Example
 * The code snippet below illustrates the use of \p CountingInputIterator to
 * dereference a sequence of incrementing integers.
 * \par
 * \code
 * #include <cub/cub.cuh>
 *
 * cub::CountingInputIterator<int> itr(5);
 *
 * printf("%d\n", itr[0]);      // 5
 * printf("%d\n", itr[1]);      // 6
 * printf("%d\n", itr[2]);      // 7
 * printf("%d\n", itr[50]);     // 55
 *
 * \endcode
 *
 * \tparam ValueType            The value type of this iterator
 * \tparam Offset               The difference type of this iterator (Default: \p ptrdiff_t)
 */
template <
    typename ValueType,
    typename Offset = ptrdiff_t>
class CountingInputIterator
{
public:

    // Required iterator traits
    typedef CountingInputIterator               self_type;              ///< My own type
    typedef Offset                              difference_type;        ///< Type to express the result of subtracting one iterator from another
    typedef ValueType                           value_type;             ///< The type of the element the iterator can point to
    typedef ValueType*                          pointer;                ///< The type of a pointer to an element the iterator can point to
    typedef ValueType                           reference;              ///< The type of a reference to an element the iterator can point to
    typedef std::random_access_iterator_tag     iterator_category;      ///< The iterator category

private:

    ValueType val;

public:

    /// Constructor
    __host__ __device__ __forceinline__ CountingInputIterator(
        const ValueType &val)          ///< Starting value for the iterator instance to report
    :
        val(val)
    {}

    /// Postfix increment
    __host__ __device__ __forceinline__ self_type operator++(int)
    {
        self_type retval = *this;
        val++;
        return retval;
    }

    /// Indirection
    __host__ __device__ __forceinline__ reference operator*()
    {
        return val;
    }

    /// Addition
    template <typename Distance>
    __host__ __device__ __forceinline__ self_type operator+(Distance n)
    {
        self_type retval(val + n);
        return retval;
    }

    /// Addition assignment
    template <typename Distance>
    __host__ __device__ __forceinline__ self_type& operator+=(Distance n)
    {
        val += n;
        return *this;
    }

    /// Subtraction
    template <typename Distance>
    __host__ __device__ __forceinline__ self_type operator-(Distance n)
    {
        self_type retval(val - n);
        return retval;
    }

    /// Subtraction assignment
    template <typename Distance>
    __host__ __device__ __forceinline__ self_type& operator-=(Distance n)
    {
        val -= n;
        return *this;
    }

    /// Array subscript
    template <typename Distance>
    __host__ __device__ __forceinline__ reference operator[](Distance n)
    {
        return val + n;
    }

    /// Structure dereference
    __host__ __device__ __forceinline__ pointer operator->()
    {
        return &val;
    }

    /// Equal to
    __host__ __device__ __forceinline__ bool operator==(const self_type& rhs)
    {
        return (val == rhs.val);
    }

    /// Not equal to
    __host__ __device__ __forceinline__ bool operator!=(const self_type& rhs)
    {
        return (val != rhs.val);
    }

    /// ostream operator
    friend std::ostream& operator<<(std::ostream& os, const self_type& itr)
    {
        return os;
    }

};


/**
 * \brief A random-access input wrapper for dereferencing array values using a PTX cache load modifier.
 *
 * \par Overview
 * - CacheModifiedInputIterator is a random-access input iterator that wraps a native
 *   device pointer of type <tt>ValueType*</tt>. \p ValueType references are
 *   made by reading \p ValueType values through loads modified by \p MODIFIER.
 * - Can be used to load any data type from memory using PTX cache load modifiers (e.g., "LOAD_LDG",
 *   "LOAD_CG", "LOAD_CA", "LOAD_CS", "LOAD_CV", etc.).
 * - Can be constructed, manipulated, and exchanged within and between host and device
 *   functions, but can only be dereferenced within device functions.
 *
 * \par Example
 * The code snippet below illustrates the use of \p CacheModifiedInputIterator to
 * dereference a device array of double using the "ldg" PTX load modifier
 * (i.e., load values through texture cache).
 * \par
 * \code
 * #include <cub/cub.cuh>
 *
 * // Declare, allocate, and initialize a device array
 * double *d_in;            // e.g., [8.0, 6.0, 7.0, 5.0, 3.0, 0.0, 9.0]
 *
 * // Create an iterator wrapper
 * cub::CacheModifiedInputIterator<cub::LOAD_LDG, double> itr(d_in);
 *
 * // Within device code:
 * printf("%f\n", itr[0]);  // 8.0
 * printf("%f\n", itr[1]);  // 6.0
 * printf("%f\n", itr[6]);  // 9.0
 *
 * \endcode
 *
 * \tparam CacheLoadModifier    The cub::CacheLoadModifier to use when accessing data
 * \tparam ValueType            The value type of this iterator
 * \tparam Offset               The difference type of this iterator (Default: \p ptrdiff_t)
 */
template <
    CacheLoadModifier   MODIFIER,
    typename            ValueType,
    typename            Offset = ptrdiff_t>
class CacheModifiedInputIterator
{
public:

    // Required iterator traits
    typedef CacheModifiedInputIterator          self_type;              ///< My own type
    typedef Offset                              difference_type;        ///< Type to express the result of subtracting one iterator from another
    typedef ValueType                           value_type;             ///< The type of the element the iterator can point to
    typedef ValueType*                          pointer;                ///< The type of a pointer to an element the iterator can point to
    typedef ValueType                           reference;              ///< The type of a reference to an element the iterator can point to
    typedef std::random_access_iterator_tag     iterator_category;      ///< The iterator category

private:

    ValueType* ptr;

public:

    /// Constructor
    __host__ __device__ __forceinline__ CacheModifiedInputIterator(
        ValueType* ptr)     ///< Native pointer to wrap
    :
        ptr(ptr)
    {}

    /// Postfix increment
    __host__ __device__ __forceinline__ self_type operator++(int)
    {
        self_type retval = *this;
        ptr++;
        return retval;
    }

    /// Indirection
    __host__ __device__ __forceinline__ reference operator*()
    {
        return ThreadLoad<MODIFIER>(ptr);
    }

    /// Addition
    template <typename Distance>
    __host__ __device__ __forceinline__ self_type operator+(Distance n)
    {
        self_type retval(ptr + n);
        return retval;
    }

    /// Addition assignment
    template <typename Distance>
    __host__ __device__ __forceinline__ self_type& operator+=(Distance n)
    {
        ptr += n;
        return *this;
    }

    /// Subtraction
    template <typename Distance>
    __host__ __device__ __forceinline__ self_type operator-(Distance n)
    {
        self_type retval(ptr - n);
        return retval;
    }

    /// Subtraction assignment
    template <typename Distance>
    __host__ __device__ __forceinline__ self_type& operator-=(Distance n)
    {
        ptr -= n;
        return *this;
    }

    /// Array subscript
    template <typename Distance>
    __host__ __device__ __forceinline__ reference operator[](Distance n)
    {
        return ThreadLoad<MODIFIER>(ptr + n);
    }

    /// Structure dereference
    __host__ __device__ __forceinline__ pointer operator->()
    {
        return &ThreadLoad<MODIFIER>(ptr);
    }

    /// Equal to
    __host__ __device__ __forceinline__ bool operator==(const self_type& rhs)
    {
        return (ptr == rhs.ptr);
    }

    /// Not equal to
    __host__ __device__ __forceinline__ bool operator!=(const self_type& rhs)
    {
        return (ptr != rhs.ptr);
    }

    /// ostream operator
    friend std::ostream& operator<<(std::ostream& os, const self_type& itr)
    {
        return os;
    }
};


/**
 * \brief A random-access output wrapper for storing array values using a PTX cache-modifier.
 *
 * \par Overview
 * - CacheModifiedOutputIterator is a random-access output iterator that wraps a native
 *   device pointer of type <tt>ValueType*</tt>. \p ValueType references are
 *   made by writing \p ValueType values through stores modified by \p MODIFIER.
 * - Can be used to store any data type to memory using PTX cache store modifiers (e.g., "STORE_WB",
 *   "STORE_CG", "STORE_CS", "STORE_WT", etc.).
 * - Can be constructed, manipulated, and exchanged within and between host and device
 *   functions, but can only be dereferenced within device functions.
 *
 * \par Example
 * The code snippet below illustrates the use of \p CacheModifiedOutputIterator to
 * dereference a device array of doubles using the "wt" PTX load modifier
 * (i.e., write-through to system memory).
 * \par
 * \code
 * #include <cub/cub.cuh>
 *
 * // Declare, allocate, and initialize a device array
 * double *d_out;              // e.g., [, , , , , , ]
 *
 * // Create an iterator wrapper
 * cub::CacheModifiedOutputIterator<cub::STORE_WT, double> itr(d_out);
 *
 * // Within device code:
 * itr[0]  = 8.0;
 * itr[1]  = 66.0;
 * itr[55] = 24.0;
 *
 * \endcode
 *
 * \par Usage Considerations
 * - Can only be dereferenced within device code
 *
 * \tparam CacheStoreModifier     The cub::CacheStoreModifier to use when accessing data
 * \tparam ValueType            The value type of this iterator
 * \tparam Offset               The difference type of this iterator (Default: \p ptrdiff_t)
 */
template <
    CacheStoreModifier  MODIFIER,
    typename            ValueType,
    typename            Offset = ptrdiff_t>
class CacheModifiedOutputIterator
{
private:

    // Proxy object
    struct Reference
    {
        ValueType* ptr;

        /// Constructor
        __host__ __device__ __forceinline__ Reference(ValueType* ptr) : ptr(ptr) {}

        /// Assignment
        __host__ __device__ __forceinline__ ValueType operator =(ValueType val)
        {
            ThreadStore<MODIFIER>(ptr, val);
            return val;
        }
    };

public:

    // Required iterator traits
    typedef CacheModifiedOutputIterator         self_type;              ///< My own type
    typedef Offset                              difference_type;        ///< Type to express the result of subtracting one iterator from another
    typedef ValueType                           value_type;             ///< The type of the element the iterator can point to
    typedef ValueType*                          pointer;                ///< The type of a pointer to an element the iterator can point to
    typedef Reference                           reference;              ///< The type of a reference to an element the iterator can point to
    typedef std::random_access_iterator_tag     iterator_category;      ///< The iterator category

private:

    ValueType* ptr;

public:

    /// Constructor
    __host__ __device__ __forceinline__ CacheModifiedOutputIterator(
        ValueType* ptr)     ///< Native pointer to wrap
    :
        ptr(ptr)
    {}

    /// Postfix increment
    __host__ __device__ __forceinline__ self_type operator++(int)
    {
        self_type retval = *this;
        ptr++;
        return retval;
    }

    /// Indirection
    __host__ __device__ __forceinline__ reference operator*()
    {
        return Reference(ptr);
    }

    /// Addition
    template <typename Distance>
    __host__ __device__ __forceinline__ self_type operator+(Distance n)
    {
        self_type retval(ptr + n);
        return retval;
    }

    /// Addition assignment
    template <typename Distance>
    __host__ __device__ __forceinline__ self_type& operator+=(Distance n)
    {
        ptr += n;
        return *this;
    }

    /// Subtraction
    template <typename Distance>
    __host__ __device__ __forceinline__ self_type operator-(Distance n)
    {
        self_type retval(ptr - n);
        return retval;
    }

    /// Subtraction assignment
    template <typename Distance>
    __host__ __device__ __forceinline__ self_type& operator-=(Distance n)
    {
        ptr -= n;
        return *this;
    }

    /// Array subscript
    template <typename Distance>
    __host__ __device__ __forceinline__ reference operator[](Distance n)
    {
        return Reference(ptr + n);
    }

    /// Equal to
    __host__ __device__ __forceinline__ bool operator==(const self_type& rhs)
    {
        return (ptr == rhs.ptr);
    }

    /// Not equal to
    __host__ __device__ __forceinline__ bool operator!=(const self_type& rhs)
    {
        return (ptr != rhs.ptr);
    }

    /// ostream operator
    friend std::ostream& operator<<(std::ostream& os, const self_type& itr)
    {
        return os;
    }
};


/**
 * \brief A random-access input wrapper for pairing dereferenced values with their corresponding indices (forming \p ItemOffsetPair tuples).
 *
 * \par Overview
 * - ArgIndexInputIterator wraps a random access input iterator \p itr of type \p InputIterator.
 *   Dereferencing an ArgIndexInputIterator at offset \p i produces a \p ItemOffsetPair value whose
 *   \p offset field is \p i and whose \p item field is <tt>itr[i]</tt>.
 * - Can be used with any data type.
 * - Can be constructed, manipulated, and exchanged within and between host and device
 *   functions.  Wrapped host memory can only be dereferenced on the host, and wrapped
 *   device memory can only be dereferenced on the device.
 *
 * \par Example
 * The code snippet below illustrates the use of \p ArgIndexInputIterator to
 * dereference an array of doubles
 * \par
 * \code
 * #include <cub/cub.cuh>
 *
 * // Declare, allocate, and initialize a device array
 * double *d_in;         // e.g., [8.0, 6.0, 7.0, 5.0, 3.0, 0.0, 9.0]
 *
 * // Create an iterator wrapper
 * cub::ArgIndexInputIterator<double> itr(d_in);
 *
 * // Within device code:
 * typedef typename cub::ArgIndexInputIterator<double>::value_type Tuple;
 * Tuple item_offset_pair.offset = *itr;
 * printf("%f @ %d\n",
 *  item_offset_pair.value,
 *  item_offset_pair.offset);   // 8.0 @ 0
 *
 * itr = itr + 6;
 * item_offset_pair.offset = *itr;
 * printf("%f @ %d\n",
 *  item_offset_pair.value,
 *  item_offset_pair.offset);   // 9.0 @ 6
 *
 * \endcode
 *
 * \tparam InputIterator        The type of the wrapped input iterator
 * \tparam Offset               The difference type of this iterator (Default: \p ptrdiff_t)
 */
template <
    typename    InputIterator,
    typename    Offset = ptrdiff_t>
class ArgIndexInputIterator
{
private:

    // Data type of input iterator
    typedef typename std::iterator_traits<InputIterator>::value_type T;

public:


    // Required iterator traits
    typedef ArgIndexInputIterator               self_type;              ///< My own type
    typedef Offset                              difference_type;        ///< Type to express the result of subtracting one iterator from another
    typedef ItemOffsetPair<T, difference_type>  value_type;             ///< The type of the element the iterator can point to
    typedef value_type*                         pointer;                ///< The type of a pointer to an element the iterator can point to
    typedef value_type                          reference;              ///< The type of a reference to an element the iterator can point to
    typedef std::random_access_iterator_tag     iterator_category;      ///< The iterator category

private:

    InputIterator   itr;
    difference_type offset;

public:

    /// Constructor
    __host__ __device__ __forceinline__ ArgIndexInputIterator(
        InputIterator   itr,            ///< Input iterator to wrap
        difference_type offset = 0)     ///< Offset (in items) from \p itr denoting the position of the iterator
    :
        itr(itr),
        offset(offset)
    {}

    /// Postfix increment
    __host__ __device__ __forceinline__ self_type operator++(int)
    {
        self_type retval = *this;
        offset++;
        return retval;
    }

    /// Indirection
    __host__ __device__ __forceinline__ reference operator*()
    {
        value_type retval;
        retval.value = itr[offset];
        retval.offset = offset;
        return retval;
    }

    /// Addition
    template <typename Distance>
    __host__ __device__ __forceinline__ self_type operator+(Distance n)
    {
        self_type retval(itr, offset + n);
        return retval;
    }

    /// Addition assignment
    template <typename Distance>
    __host__ __device__ __forceinline__ self_type& operator+=(Distance n)
    {
        offset += n;
        return *this;
    }

    /// Subtraction
    template <typename Distance>
    __host__ __device__ __forceinline__ self_type operator-(Distance n)
    {
        self_type retval(itr, offset - n);
        return retval;
    }

    /// Subtraction assignment
    template <typename Distance>
    __host__ __device__ __forceinline__ self_type& operator-=(Distance n)
    {
        offset -= n;
        return *this;
    }

    /// Array subscript
    template <typename Distance>
    __host__ __device__ __forceinline__ reference operator[](Distance n)
    {
        return *(*this + n);
    }

    /// Structure dereference
    __host__ __device__ __forceinline__ pointer operator->()
    {
        return &(*(*this));
    }

    /// Equal to
    __host__ __device__ __forceinline__ bool operator==(const self_type& rhs)
    {
        return ((itr == rhs.itr) && (offset == rhs.offset));
    }

    /// Not equal to
    __host__ __device__ __forceinline__ bool operator!=(const self_type& rhs)
    {
        return ((itr != rhs.itr) || (offset != rhs.offset));
    }

    /// ostream operator
    friend std::ostream& operator<<(std::ostream& os, const self_type& itr)
    {
        return os;
    }
};


/**
 * \brief A random-access input wrapper for transforming dereferenced values.
 *
 * \par Overview
 * - TransformInputIterator wraps a unary conversion functor of type \p
 *   ConversionOp and a random-access input iterator of type <tt>InputIterator</tt>,
 *   using the former to produce references of type \p ValueType from the latter.
 * - Can be used with any data type.
 * - Can be constructed, manipulated, and exchanged within and between host and device
 *   functions.  Wrapped host memory can only be dereferenced on the host, and wrapped
 *   device memory can only be dereferenced on the device.
 *
 * \par Example
 * The code snippet below illustrates the use of \p TransformInputIterator to
 * dereference an array of integers, tripling the values and converting them to doubles.
 * \par
 * \code
 * #include <cub/cub.cuh>
 *
 * // Functor for tripling integer values and converting to doubles
 * struct TripleDoubler
 * {
 *     __host__ __device__ __forceinline__
 *     double operator()(const int &a) const {
 *         return double(a * 2);
 *     }
 * };
 *
 * // Declare, allocate, and initialize a device array
 * int *d_in;                   // e.g., [8, 6, 7, 5, 3, 0, 9]
 * TripleDoubler conversion_op;
 *
 * // Create an iterator wrapper
 * cub::TransformInputIterator<double, TripleDoubler, int*> itr(d_in, conversion_op);
 *
 * // Within device code:
 * printf("%f\n", itr[0]);  // 24.0
 * printf("%f\n", itr[1]);  // 18.0
 * printf("%f\n", itr[6]);  // 27.0
 *
 * \endcode
 *
 * \tparam ValueType            The value type of this iterator
 * \tparam ConversionOp         Unary functor type for mapping objects of type \p InputType to type \p ValueType.  Must have member <tt>ValueType operator()(const InputType &datum)</tt>.
 * \tparam InputIterator        The type of the wrapped input iterator
 * \tparam Offset               The difference type of this iterator (Default: \p ptrdiff_t)
 *
 */
template <
    typename ValueType,
    typename ConversionOp,
    typename InputIterator,
    typename Offset = ptrdiff_t>
class TransformInputIterator
{
public:

    // Required iterator traits
    typedef TransformInputIterator              self_type;              ///< My own type
    typedef Offset                              difference_type;        ///< Type to express the result of subtracting one iterator from another
    typedef ValueType                           value_type;             ///< The type of the element the iterator can point to
    typedef ValueType*                          pointer;                ///< The type of a pointer to an element the iterator can point to
    typedef ValueType                           reference;              ///< The type of a reference to an element the iterator can point to
    typedef std::random_access_iterator_tag     iterator_category;      ///< The iterator category

private:

    ConversionOp  conversion_op;
    InputIterator input_itr;

public:

    /// Constructor
    __host__ __device__ __forceinline__ TransformInputIterator(
        InputIterator       input_itr,          ///< Input iterator to wrap
        ConversionOp        conversion_op)      ///< Conversion functor to wrap
    :
        conversion_op(conversion_op),
        input_itr(input_itr)
    {}

    /// Postfix increment
    __host__ __device__ __forceinline__ self_type operator++(int)
    {
        self_type retval = *this;
        input_itr++;
        return retval;
    }

    /// Indirection
    __host__ __device__ __forceinline__ reference operator*()
    {
        return conversion_op(*input_itr);
    }

    /// Addition
    template <typename Distance>
    __host__ __device__ __forceinline__ self_type operator+(Distance n)
    {
        self_type retval(input_itr + n, conversion_op);
        return retval;
    }

    /// Addition assignment
    template <typename Distance>
    __host__ __device__ __forceinline__ self_type& operator+=(Distance n)
    {
        input_itr += n;
        return *this;
    }

    /// Subtraction
    template <typename Distance>
    __host__ __device__ __forceinline__ self_type operator-(Distance n)
    {
        self_type retval(input_itr - n, conversion_op);
        return retval;
    }

    /// Subtraction assignment
    template <typename Distance>
    __host__ __device__ __forceinline__ self_type& operator-=(Distance n)
    {
        input_itr -= n;
        return *this;
    }

    /// Array subscript
    template <typename Distance>
    __host__ __device__ __forceinline__ reference operator[](Distance n)
    {
        return conversion_op(input_itr[n]);
    }

    /// Structure dereference
    __host__ __device__ __forceinline__ pointer operator->()
    {
        return &conversion_op(*input_itr);
    }

    /// Equal to
    __host__ __device__ __forceinline__ bool operator==(const self_type& rhs)
    {
        return (input_itr == rhs.input_itr);
    }

    /// Not equal to
    __host__ __device__ __forceinline__ bool operator!=(const self_type& rhs)
    {
        return (input_itr != rhs.input_itr);
    }

    /// ostream operator
    friend std::ostream& operator<<(std::ostream& os, const self_type& itr)
    {
        return os;
    }
};



/**
 * \brief A random-access input wrapper for dereferencing array values through texture cache.  Uses older Tesla/Fermi-style texture references.
 *
 * \par Overview
 * - TexRefInputIterator wraps a native device pointer of type <tt>ValueType*</tt>. References
 *   to elements are to be loaded through texture cache.
 * - Can be used to load any data type from memory through texture cache.
 * - Can be manipulated and exchanged within and between host and device
 *   functions, can only be constructed within host functions, and can only be
 *   dereferenced within device functions.
 * - The \p UNIQUE_ID template parameter is used to statically name the underlying texture
 *   reference.  Only one TexRefInputIterator instance can be bound at any given time for a
 *   specific combination of (1) data type \p T, (2) \p UNIQUE_ID, (3) host
 *   thread, and (4) compilation .o unit.
 * - With regard to nested/dynamic parallelism, TexRefInputIterator iterators may only be
 *   created by the host thread and used by a top-level kernel (i.e. the one which is launched
 *   from the host).
 *
 * \par Example
 * The code snippet below illustrates the use of \p TexRefInputIterator to
 * dereference a device array of doubles through texture cache.
 * \par
 * \code
 * #include <cub/cub.cuh>
 *
 * // Declare, allocate, and initialize a device array
 * int num_items;   // e.g., 7
 * double *d_in;    // e.g., [8.0, 6.0, 7.0, 5.0, 3.0, 0.0, 9.0]
 *
 * // Create an iterator wrapper
 * cub::TexRefInputIterator<double, __LINE__> itr;
 * itr.BindTexture(d_in, sizeof(double) * num_items);
 * ...
 *
 * // Within device code:
 * printf("%f\n", itr[0]);      // 8.0
 * printf("%f\n", itr[1]);      // 6.0
 * printf("%f\n", itr[6]);      // 9.0
 *
 * ...
 * itr.UnbindTexture();
 *
 * \endcode
 *
 * \tparam T                    The value type of this iterator
 * \tparam UNIQUE_ID            A globally-unique identifier (within the compilation unit) to name the underlying texture reference
 * \tparam Offset               The difference type of this iterator (Default: \p ptrdiff_t)
 */
template <
    typename    T,
    int         UNIQUE_ID,
    typename    Offset = ptrdiff_t>
class TexRefInputIterator
{
public:

    // Required iterator traits
    typedef TexRefInputIterator                 self_type;              ///< My own type
    typedef Offset                              difference_type;        ///< Type to express the result of subtracting one iterator from another
    typedef T                                   value_type;             ///< The type of the element the iterator can point to
    typedef T*                                  pointer;                ///< The type of a pointer to an element the iterator can point to
    typedef T                                   reference;              ///< The type of a reference to an element the iterator can point to
    typedef std::random_access_iterator_tag     iterator_category;      ///< The iterator category

private:

    T*              ptr;
    difference_type tex_offset;

    // Texture reference wrapper (old Tesla/Fermi-style textures)
    typedef typename IteratorTexRef<T>::template TexId<UNIQUE_ID> TexId;

public:

    /// Constructor
    __host__ __device__ __forceinline__ TexRefInputIterator()
    :
        ptr(NULL),
        tex_offset(0)
    {}

    /// Use this iterator to bind \p ptr with a texture reference
    cudaError_t BindTexture(
        T               *ptr,                   ///< Native pointer to wrap that is aligned to cudaDeviceProp::textureAlignment
        size_t          bytes,                  ///< Number of bytes in the range
        size_t          tex_offset = 0)         ///< Offset (in items) from \p ptr denoting the position of the iterator
    {
        this->ptr = ptr;
        this->tex_offset = tex_offset;
        return TexId::BindTexture(ptr);
    }

    /// Unbind this iterator from its texture reference
    cudaError_t UnbindTexture()
    {
        return TexId::UnbindTexture();
    }

    /// Postfix increment
    __host__ __device__ __forceinline__ self_type operator++(int)
    {
        self_type retval = *this;
        tex_offset++;
        return retval;
    }

    /// Indirection
    __host__ __device__ __forceinline__ reference operator*()
    {
#if (CUB_PTX_VERSION == 0)
        // Simply dereference the pointer on the host
        return ptr[tex_offset];
#else
        // Use the texture reference
        return TexId::Fetch(tex_offset);
#endif
    }

    /// Addition
    template <typename Distance>
    __host__ __device__ __forceinline__ self_type operator+(Distance n)
    {
        self_type retval;
        retval.ptr = ptr;
        retval.tex_offset = tex_offset + n;
        return retval;
    }

    /// Addition assignment
    template <typename Distance>
    __host__ __device__ __forceinline__ self_type& operator+=(Distance n)
    {
        tex_offset += n;
        return *this;
    }

    /// Subtraction
    template <typename Distance>
    __host__ __device__ __forceinline__ self_type operator-(Distance n)
    {
        self_type retval;
        retval.ptr = ptr;
        retval.tex_offset = tex_offset - n;
        return retval;
    }

    /// Subtraction assignment
    template <typename Distance>
    __host__ __device__ __forceinline__ self_type& operator-=(Distance n)
    {
        tex_offset -= n;
        return *this;
    }

    /// Array subscript
    template <typename Distance>
    __host__ __device__ __forceinline__ reference operator[](Distance n)
    {
        return *(*this + n);
    }

    /// Structure dereference
    __host__ __device__ __forceinline__ pointer operator->()
    {
        return &(*(*this));
    }

    /// Equal to
    __host__ __device__ __forceinline__ bool operator==(const self_type& rhs)
    {
        return ((ptr == rhs.ptr) && (tex_offset == rhs.tex_offset));
    }

    /// Not equal to
    __host__ __device__ __forceinline__ bool operator!=(const self_type& rhs)
    {
        return ((ptr != rhs.ptr) || (tex_offset != rhs.tex_offset));
    }

    /// ostream operator
    friend std::ostream& operator<<(std::ostream& os, const self_type& itr)
    {
        return os;
    }

};


/**
 * \brief A random-access input wrapper for dereferencing array values through texture cache.  Uses newer Kepler-style texture objects.
 *
 * \par Overview
 * - TexObjInputIterator wraps a native device pointer of type <tt>ValueType*</tt>. References
 *   to elements are to be loaded through texture cache.
 * - Can be used to load any data type from memory through texture cache.
 * - Can be manipulated and exchanged within and between host and device
 *   functions, can only be constructed within host functions, and can only be
 *   dereferenced within device functions.
 * - With regard to nested/dynamic parallelism, TexObjInputIterator iterators may only be
 *   created by the host thread, but can be used by any descendant kernel.
 *
 * \par Example
 * The code snippet below illustrates the use of \p TexRefInputIterator to
 * dereference a device array of doubles through texture cache.
 * \par
 * \code
 * #include <cub/cub.cuh>
 *
 * // Declare, allocate, and initialize a device array
 * int num_items;   // e.g., 7
 * double *d_in;    // e.g., [8.0, 6.0, 7.0, 5.0, 3.0, 0.0, 9.0]
 *
 * // Create an iterator wrapper
 * cub::TexObjInputIterator<double> itr;
 * itr.BindTexture(d_in, sizeof(double) * num_items);
 * ...
 *
 * // Within device code:
 * printf("%f\n", itr[0]);      // 8.0
 * printf("%f\n", itr[1]);      // 6.0
 * printf("%f\n", itr[6]);      // 9.0
 *
 * ...
 * itr.UnbindTexture();
 *
 * \endcode
 *
 * \tparam T                    The value type of this iterator
 * \tparam Offset               The difference type of this iterator (Default: \p ptrdiff_t)
 */
template <
    typename    T,
    typename    Offset = ptrdiff_t>
class TexObjInputIterator
{
public:

    // Required iterator traits
    typedef TexObjInputIterator                 self_type;              ///< My own type
    typedef Offset                              difference_type;        ///< Type to express the result of subtracting one iterator from another
    typedef T                                   value_type;             ///< The type of the element the iterator can point to
    typedef T*                                  pointer;                ///< The type of a pointer to an element the iterator can point to
    typedef T                                   reference;              ///< The type of a reference to an element the iterator can point to
    typedef std::random_access_iterator_tag     iterator_category;      ///< The iterator category

private:

    // Largest texture word we can use in device
    typedef typename UnitWord<T>::TextureWord TextureWord;

    // Number of texture words per T
    enum {
        TEXTURE_MULTIPLE = UnitWord<T>::TEXTURE_MULTIPLE
    };

private:

    T*                  ptr;
    difference_type     tex_offset;
    cudaTextureObject_t tex_obj;

public:

    /// Constructor
    __host__ __device__ __forceinline__ TexObjInputIterator()
    :
        ptr(NULL),
        tex_offset(0),
        tex_obj(0)
    {}

    /// Use this iterator to bind \p ptr with a texture reference
    cudaError_t BindTexture(
        T               *ptr,               ///< Native pointer to wrap that is aligned to cudaDeviceProp::textureAlignment
        size_t          bytes,              ///< Number of bytes in the range
        size_t          tex_offset = 0)     ///< Offset (in items) from \p ptr denoting the position of the iterator
    {
        this->ptr = ptr;
        this->tex_offset = tex_offset;

        cudaChannelFormatDesc   channel_desc = cudaCreateChannelDesc<TextureWord>();
        cudaResourceDesc        res_desc;
        cudaTextureDesc         tex_desc;
        memset(&res_desc, 0, sizeof(cudaResourceDesc));
        memset(&tex_desc, 0, sizeof(cudaTextureDesc));
        res_desc.resType                = cudaResourceTypeLinear;
        res_desc.res.linear.devPtr      = ptr;
        res_desc.res.linear.desc        = channel_desc;
        res_desc.res.linear.sizeInBytes = bytes;
        tex_desc.readMode               = cudaReadModeElementType;
        return cudaCreateTextureObject(&tex_obj, &res_desc, &tex_desc, NULL);
    }

    /// Unbind this iterator from its texture reference
    cudaError_t UnbindTexture()
    {
        return cudaDestroyTextureObject(tex_obj);
    }

    /// Postfix increment
    __host__ __device__ __forceinline__ self_type operator++(int)
    {
        self_type retval = *this;
        tex_offset++;
        return retval;
    }

    /// Indirection
    __host__ __device__ __forceinline__ reference operator*()
    {
#if (CUB_PTX_VERSION == 0)
        // Simply dereference the pointer on the host
        return ptr[tex_offset];
#else
        // Move array of uninitialized words, then alias and assign to return value
        TextureWord words[TEXTURE_MULTIPLE];

        #pragma unroll
        for (int i = 0; i < TEXTURE_MULTIPLE; ++i)
        {
            words[i] = tex1Dfetch<TextureWord>(
                tex_obj,
                (tex_offset * TEXTURE_MULTIPLE) + i);
        }

        // Load from words
        return *reinterpret_cast<T*>(words);
#endif
    }

    /// Addition
    template <typename Distance>
    __host__ __device__ __forceinline__ self_type operator+(Distance n)
    {
        self_type retval;
        retval.ptr          = ptr;
        retval.tex_obj      = tex_obj;
        retval.tex_offset   = tex_offset + n;
        return retval;
    }

    /// Addition assignment
    template <typename Distance>
    __host__ __device__ __forceinline__ self_type& operator+=(Distance n)
    {
        tex_offset += n;
        return *this;
    }

    /// Subtraction
    template <typename Distance>
    __host__ __device__ __forceinline__ self_type operator-(Distance n)
    {
        self_type retval;
        retval.ptr          = ptr;
        retval.tex_obj      = tex_obj;
        retval.tex_offset   = tex_offset - n;
        return retval;
    }

    /// Subtraction assignment
    template <typename Distance>
    __host__ __device__ __forceinline__ self_type& operator-=(Distance n)
    {
        tex_offset -= n;
        return *this;
    }

    /// Array subscript
    template <typename Distance>
    __host__ __device__ __forceinline__ reference operator[](Distance n)
    {
        return *(*this + n);
    }

    /// Structure dereference
    __host__ __device__ __forceinline__ pointer operator->()
    {
        return &(*(*this));
    }

    /// Equal to
    __host__ __device__ __forceinline__ bool operator==(const self_type& rhs)
    {
        return ((ptr == rhs.ptr) && (tex_offset == rhs.tex_offset) && (tex_obj == rhs.tex_obj));
    }

    /// Not equal to
    __host__ __device__ __forceinline__ bool operator!=(const self_type& rhs)
    {
        return ((ptr != rhs.ptr) || (tex_offset != rhs.tex_offset) || (tex_obj != rhs.tex_obj));
    }

    /// ostream operator
    friend std::ostream& operator<<(std::ostream& os, const self_type& itr)
    {
        return os;
    }

};



/** @} */       // end group UtilIterator

}               // CUB namespace
CUB_NS_POSTFIX  // Optional outer namespace(s)
