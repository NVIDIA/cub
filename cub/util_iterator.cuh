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
template <typename  TextureWord>
template <int       UNIQUE_ID>
typename IteratorTexRef<TextureWord>::TexId<UNIQUE_ID>::TexRef IteratorTexRef<TextureWord>::TexId<UNIQUE_ID>::ref = 0;

} // Anonymous namespace


#endif // DOXYGEN_SHOULD_SKIP_THIS







/**
 * \addtogroup UtilModule
 * @{
 */


/******************************************************************************
 * Iterators
 *****************************************************************************/

/**
 * \brief A random-access input iterator for loading a range of constant values
 *
 * \par Overview
 * Read references to a ConstantInputIterator iterator always return the supplied constant
 * of type \p ValueType.
 *
 * \tparam ValueType            The value type of this iterator
 * \tparam difference_type      The difference type of this iterator (Default: \p ptrdiff_t)
 */
template <
    typename ValueType,
    typename difference_type = ptrdiff_t>
class ConstantInputIterator
{
public:

    // Required iterator traits
    typedef ConstantInputIterator               self_type;              ///< My own type
    typedef difference_type                     difference_type;        ///< Type to express the result of subtracting one iterator from another
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
 * \brief A random-access input iterator for loading a range of sequential integer values.
 *
 * \par Overview
 * After initializing a CountingInputIterator to a certain integer \p base, read references
 * at \p offset will return the value \p base + \p offset.
 *
 * \tparam ValueType            The value type of this iterator
 * \tparam difference_type      The difference type of this iterator (Default: \p ptrdiff_t)
 */
template <
    typename ValueType,
    typename difference_type = ptrdiff_t>
class CountingInputIterator
{
public:

    // Required iterator traits
    typedef CountingInputIterator               self_type;              ///< My own type
    typedef difference_type                     difference_type;        ///< Type to express the result of subtracting one iterator from another
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
 * \brief A random-access input iterator for loading array elements through a PTX cache modifier.
 *
 * \par Overview
 * CacheModifiedInputIterator is a random-access input iterator that wraps a native
 * device pointer of type <tt>ValueType*</tt>. \p ValueType references are
 * made by reading \p ValueType values through loads modified by \p MODIFIER.
 *
 * \tparam CacheLoadModifier    The cub::CacheLoadModifier to use when accessing data
 * \tparam ValueType            The value type of this iterator
 * \tparam difference_type      The difference type of this iterator (Default: \p ptrdiff_t)
 */
template <
    CacheLoadModifier   MODIFIER,
    typename            ValueType,
    typename            difference_type = ptrdiff_t>
class CacheModifiedInputIterator
{
public:

    // Required iterator traits
    typedef CacheModifiedInputIterator          self_type;              ///< My own type
    typedef difference_type                     difference_type;        ///< Type to express the result of subtracting one iterator from another
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
 * \brief A random-access output iterator for storing array elements through a PTX cache modifier.
 *
 * \par Overview
 * CacheModifiedOutputIterator is a random-access output iterator that wraps a native
 * device pointer of type <tt>ValueType*</tt>. \p ValueType references are
 * made by writing \p ValueType values through stores modified by \p MODIFIER.
 *
 * \par Usage Considerations
 * - Can only be dereferenced within device code
 *
 * \tparam CacheStoreModifier     The cub::CacheStoreModifier to use when accessing data
 * \tparam ValueType            The value type of this iterator
 * \tparam difference_type      The difference type of this iterator (Default: \p ptrdiff_t)
 */
template <
    CacheStoreModifier  MODIFIER,
    typename            ValueType,
    typename            difference_type = ptrdiff_t>
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
    typedef difference_type                     difference_type;        ///< Type to express the result of subtracting one iterator from another
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
 * \brief A random-access input iterator for applying a transformation operator to another random-access input iterator.
 *
 * \par Overview
 * TransformInputIterator wraps a unary conversion functor of type \p ConversionOp and a random-access
 * input iterator of type <tt>InputIterator</tt>, using the former to produce
 * references of type \p ValueType from the latter.
 *
 * \tparam ValueType            The value type of this iterator
 * \tparam ConversionOp         Unary functor type for mapping objects of type \p InputType to type \p ValueType.  Must have member <tt>ValueType operator()(const InputType &datum)</tt>.
 * \tparam difference_type      The difference type of this iterator (Default: \p ptrdiff_t)
 *
 */
template <
    typename ValueType,
    typename ConversionOp,
    typename InputIterator,
    typename difference_type = ptrdiff_t>
class TransformInputIterator
{
public:

    // Required iterator traits
    typedef TransformInputIterator              self_type;              ///< My own type
    typedef difference_type                     difference_type;        ///< Type to express the result of subtracting one iterator from another
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
 * \brief A random-access input iterator for loading primitive array elements through texture cache.  Uses older Tesla/Fermi-style texture references.
 *
 * \par Overview
 * TexRefInputIterator wraps a native device pointer of type <tt>ValueType*</tt>. References
 * to elements are to be pulled through texture cache.  Works with any \p ValueType.
 *
 * \par Usage Considerations
 * - The \p UNIQUE_ID template parameter is used to statically name the underlying texture
 *   reference.  For a given data type \p T and \p UNIQUE_ID, only one TexRefInputIterator
 *   instance can be bound at any given time (per host thread, per compilation .o unit)
 * - With regard to nested/dynamic parallelism, TexRefInputIterator iterators may only be
 *   created by the host thread and used by a top-level kernel (i.e. the one which is launched
 *   from the host).
 *
 * \p UNIQUE_ID template parameter is used to statically name the underlying texture
 *   reference.  For a given \p UNIQUE_ID, only one TexRefInputIterator instance can be
 *   bound at any given time (per host thread, per compilation .o unit)
 *
 * \tparam T                    The value type of this iterator
 * \tparam UNIQUE_ID            A globally-unique identifier (within the compilation unit) to name the underlying texture reference
 * \tparam difference_type      The difference type of this iterator (Default: \p ptrdiff_t)
 */
template <
    typename    T,
    int         UNIQUE_ID,
    typename    difference_type = ptrdiff_t>
class TexRefInputIterator
{
public:

    // Required iterator traits
    typedef TexRefInputIterator                 self_type;              ///< My own type
    typedef difference_type                     difference_type;        ///< Type to express the result of subtracting one iterator from another
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
 * \brief A random-access input iterator for loading primitive array elements through texture cache.  Uses newer Kepler-style texture objects.
 *
 * \par Overview
 * TexObjInputIterator wraps a native device pointer of type <tt>ValueType*</tt>. References
 * to elements are to be pulled through texture cache.  Works with any \p ValueType.
 *
 * \par Usage Considerations
 * - With regard to nested/dynamic parallelism, TexObjInputIterator iterators may only be
 *   created by the host thread, but can be used by any descendant kernel.
 *
 * \tparam T                    The value type of this iterator
 * \tparam difference_type      The difference type of this iterator (Default: \p ptrdiff_t)
 */
template <
    typename    T,
    typename    difference_type = ptrdiff_t>
class TexObjInputIterator
{
public:

    // Required iterator traits
    typedef TexObjInputIterator                 self_type;              ///< My own type
    typedef difference_type                     difference_type;        ///< Type to express the result of subtracting one iterator from another
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



/** @} */       // end group UtilModule

}               // CUB namespace
CUB_NS_POSTFIX  // Optional outer namespace(s)
