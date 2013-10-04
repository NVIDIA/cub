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
#include "util_device.cuh"
#include "util_debug.cuh"
#include "util_namespace.cuh"

/// Optional outer namespace(s)
CUB_NS_PREFIX

/// CUB namespace
namespace cub {


/******************************************************************************
 * Texture references
 *****************************************************************************/

#ifndef DOXYGEN_SHOULD_SKIP_THIS    // Do not document

// Anonymous namespace
namespace {

/// Templated texture reference type

template <int UNIQUE_ID>
struct Foo
{
    template <typename TextureWord>
    struct TexIteratorRef
    {
        static texture<TextureWord> ref;

        static cudaError_t BindTexture(void *d_in)
        {
            cudaChannelFormatDesc tex_desc = cudaCreateChannelDesc<TextureWord>();
            if (d_in)
                return (CubDebug(cudaBindTexture(NULL, ref, d_in, tex_desc)));

            return cudaSuccess;
        }

        static cudaError_t UnbindTexture()
        {
            return CubDebug(cudaUnbindTexture(ref));
        }
    };
};

// Texture reference definitions
template <int UNIQUE_ID>
template <typename TextureWord>
texture<TextureWord> Foo<UNIQUE_ID>::TexIteratorRef<TextureWord>::ref = 0;

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
 * \brief A random-access input iterator for referencing a range of constant values
 *
 * \par Overview
 * Read references to a ConstantIteratorRA iterator always return the supplied constant
 * of type \p ValueType.
 *
 * \tparam ValueType            The value type of this iterator
 * \tparam difference_type      The difference type of this iterator (Default: \p ptrdiff_t)
 */
template <
    typename ValueType,
    typename difference_type = ptrdiff_t>
class ConstantIteratorRA
{
public:

    // Required iterator traits
    typedef ConstantIteratorRA                  self_type;              ///< My own type
    typedef difference_type                     difference_type;        ///< Type to express the result of subtracting one iterator from another
    typedef ValueType                           value_type;             ///< The type of the element the iterator can point to
    typedef ValueType*                          pointer;                ///< The type of a pointer to an element the iterator can point to
    typedef ValueType                           reference;              ///< The type of a reference to an element the iterator can point to
    typedef std::random_access_iterator_tag     iterator_category;      ///< The iterator category

private:

    ValueType val;

public:

    /// Constructor
    __host__ __device__ __forceinline__ ConstantIteratorRA(
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
 * \brief A random-access input iterator for referencing a range of sequential integer values.
 *
 * \par Overview
 * After initializing a CountingIteratorRA to a certain integer \p base, read references
 * at \p offset will return the value \p base + \p offset.
 *
 * \tparam ValueType            The value type of this iterator
 * \tparam difference_type      The difference type of this iterator (Default: \p ptrdiff_t)
 */
template <
    typename ValueType,
    typename difference_type = ptrdiff_t>
class CountingIteratorRA
{
public:

    // Required iterator traits
    typedef CountingIteratorRA                  self_type;              ///< My own type
    typedef difference_type                     difference_type;        ///< Type to express the result of subtracting one iterator from another
    typedef ValueType                           value_type;             ///< The type of the element the iterator can point to
    typedef ValueType*                          pointer;                ///< The type of a pointer to an element the iterator can point to
    typedef ValueType                           reference;              ///< The type of a reference to an element the iterator can point to
    typedef std::random_access_iterator_tag     iterator_category;      ///< The iterator category

private:

    ValueType val;

public:

    /// Constructor
    __host__ __device__ __forceinline__ CountingIteratorRA(
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
 * \brief A random-access input iterator for referencing array elements through a PTX cache modifier.
 *
 * \par Overview
 * CacheModifiedIteratorRA is a random-access input iterator that wraps a native
 * device pointer of type <tt>ValueType*</tt>. \p ValueType references are
 * made by reading \p ValueType values through loads modified by \p MODIFIER.
 *
 * \tparam PtxLoadModifier      The cub::PtxLoadModifier to use when accessing data
 * \tparam ValueType            The value type of this iterator
 * \tparam difference_type      The difference type of this iterator (Default: \p ptrdiff_t)
 */
template <
    PtxLoadModifier     MODIFIER,
    typename            ValueType,
    typename            difference_type = ptrdiff_t>
class CacheModifiedIteratorRA
{
public:

    // Required iterator traits
    typedef CacheModifiedIteratorRA             self_type;              ///< My own type
    typedef difference_type                     difference_type;        ///< Type to express the result of subtracting one iterator from another
    typedef ValueType                           value_type;             ///< The type of the element the iterator can point to
    typedef ValueType*                          pointer;                ///< The type of a pointer to an element the iterator can point to
    typedef ValueType                           reference;              ///< The type of a reference to an element the iterator can point to
    typedef std::random_access_iterator_tag     iterator_category;      ///< The iterator category

private:

    ValueType* ptr;

public:

    /// Constructor
    __host__ __device__ __forceinline__ CacheModifiedIteratorRA(
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
 * \brief A random-access input iterator for applying a transformation operator to another random-access input iterator.
 *
 * \par Overview
 * TransformIteratorRA wraps a unary conversion functor of type \p ConversionOp and a random-access
 * input iterator of type <tt>InputIteratorRA</tt>, using the former to produce
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
    typename InputIteratorRA,
    typename difference_type = ptrdiff_t>
class TransformIteratorRA
{
public:

    // Required iterator traits
    typedef TransformIteratorRA                 self_type;              ///< My own type
    typedef difference_type                     difference_type;        ///< Type to express the result of subtracting one iterator from another
    typedef ValueType                           value_type;             ///< The type of the element the iterator can point to
    typedef ValueType*                          pointer;                ///< The type of a pointer to an element the iterator can point to
    typedef ValueType                           reference;              ///< The type of a reference to an element the iterator can point to
    typedef std::random_access_iterator_tag     iterator_category;      ///< The iterator category

private:

    ConversionOp    conversion_op;
    InputIteratorRA input_itr;

public:

    /// Constructor
    __host__ __device__ __forceinline__ TransformIteratorRA(
        InputIteratorRA     input_itr,          ///< Input iterator to wrap
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
 * \brief A random-access input iterator for referencing primitive array elements through texture cache.
 *
 * \par Overview
 * TexIteratorRA wraps a native device pointer of type <tt>ValueType*</tt>. References
 * to elements are to be pulled through texture cache.  Works with any \p ValueType.
 *
 * \par Usage Considerations
 * - Only one TexIteratorRA or TexIteratorRA of a certain \p ValueType can be bound at any given time (per host thread)
 *
 * \tparam T            The value type of this iterator
 * \tparam difference_type      The difference type of this iterator (Default: \p ptrdiff_t)
 */
template <
    typename    T,
    int         UNIQUE_ID,
    typename    difference_type = ptrdiff_t>
class TexIteratorRA
{
public:

    // Required iterator traits
    typedef TexIteratorRA                       self_type;              ///< My own type
    typedef difference_type                     difference_type;        ///< Type to express the result of subtracting one iterator from another
    typedef T                                   value_type;             ///< The type of the element the iterator can point to
    typedef T*                                  pointer;                ///< The type of a pointer to an element the iterator can point to
    typedef T                                   reference;              ///< The type of a reference to an element the iterator can point to
    typedef std::random_access_iterator_tag     iterator_category;      ///< The iterator category

private:

    // Largest texture word we can use
    typedef typename WordAlignment<T>::TextureWord TextureWord;

    // Texture reference wrapper
    typedef typename Foo<UNIQUE_ID>::template TexIteratorRef<TextureWord> TexIteratorRef;


    // Number of texture words per T
    enum {
        TEXTURE_MULTIPLE = WordAlignment<T>::TEXTURE_MULTIPLE
    };

    T*                  ptr;
    size_t              tex_align_offset;
    cudaTextureObject_t tex_obj;

public:

    /// Constructor
    __host__ __device__ __forceinline__ TexIteratorRA()
    :
        ptr(NULL),
        tex_align_offset(0),
        tex_obj(0)
    {}

    /// Use this iterator to bind \p ptr with a texture reference
    cudaError_t BindTexture(
        T               *ptr,                   ///< Native pointer to wrap that is aligned to cudaDeviceProp::textureAlignment
        size_t          bytes,                  ///< Number of bytes in the range
        size_t          tex_align_offset = 0)   ///< Offset (in items) from \p ptr denoting the position of the iterator
    {

        this->ptr = ptr;
        this->tex_align_offset = tex_align_offset * TEXTURE_MULTIPLE;

        TextureWord *tex_ptr = reinterpret_cast<TextureWord*>(ptr);

        int ptx_version;
        cudaError_t error = cudaSuccess;
        if (CubDebug(error = PtxVersion(ptx_version))) return error;
        if (ptx_version >= 300)
        {
            // Use texture object
            cudaChannelFormatDesc   channel_desc = cudaCreateChannelDesc<TextureWord>();
            cudaResourceDesc        res_desc;
            cudaTextureDesc         tex_desc;
            memset(&res_desc, 0, sizeof(cudaResourceDesc));
            memset(&tex_desc, 0, sizeof(cudaTextureDesc));
            res_desc.resType                = cudaResourceTypeLinear;
            res_desc.res.linear.devPtr      = tex_ptr;
            res_desc.res.linear.desc        = channel_desc;
            res_desc.res.linear.sizeInBytes = bytes;
            tex_desc.readMode               = cudaReadModeElementType;
            return cudaCreateTextureObject(&tex_obj, &res_desc, &tex_desc, NULL);
        }
        else
        {
            // Use texture reference
            return TexIteratorRef::BindTexture(tex_ptr);
        }
    }

    /// Unbind this iterator from its texture reference
    cudaError_t UnbindTexture()
    {
        int ptx_version;
        cudaError_t error = cudaSuccess;
        if (CubDebug(error = PtxVersion(ptx_version))) return error;
        if (ptx_version < 300)
        {
            // Use texture reference
            return TexIteratorRef::UnbindTexture();
        }
        else
        {
            // Use texture object
            return cudaDestroyTextureObject(tex_obj);
        }
    }

    /// Postfix increment
    __host__ __device__ __forceinline__ self_type operator++(int)
    {
        self_type retval = *this;
        ptr++;
        tex_align_offset += TEXTURE_MULTIPLE;
        return retval;
    }

    /// Indirection
    __host__ __device__ __forceinline__ reference operator*()
    {
#if (CUB_PTX_VERSION == 0)

        // Simply dereference the pointer on the host
        return *ptr;

#else

        // Move array of uninitialized words, then alias and assign to return value
        typename WordAlignment<T>::UninitializedTextureWords words;

        #pragma unroll
        for (int i = 0; i < TEXTURE_MULTIPLE; ++i)
#if (CUB_PTX_VERSION < 300)
            words.buf[i] = tex1Dfetch(TexIteratorRef::ref, tex_align_offset + i);
#else
            words.buf[i] = tex1Dfetch<TextureWord>(tex_obj, tex_align_offset + i);
#endif

        // Load from words
        return *reinterpret_cast<T*>(words.buf);
#endif
    }

    /// Addition
    template <typename Distance>
    __host__ __device__ __forceinline__ self_type operator+(Distance n)
    {
        self_type retval;
        retval.ptr = ptr + n;
        retval.tex_align_offset = tex_align_offset + (n * TEXTURE_MULTIPLE);
        return retval;
    }

    /// Addition assignment
    template <typename Distance>
    __host__ __device__ __forceinline__ self_type& operator+=(Distance n)
    {
        ptr += n;
        tex_align_offset += (n * TEXTURE_MULTIPLE);
        return *this;
    }

    /// Subtraction
    template <typename Distance>
    __host__ __device__ __forceinline__ self_type operator-(Distance n)
    {
        self_type retval;
        retval.ptr = ptr - n;
        retval.tex_align_offset = tex_align_offset - (n * TEXTURE_MULTIPLE);
        return retval;
    }

    /// Subtraction assignment
    template <typename Distance>
    __host__ __device__ __forceinline__ self_type& operator-=(Distance n)
    {
        ptr -= n;
        tex_align_offset -= (n * TEXTURE_MULTIPLE);
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
 * \brief A random-access input iterator for applying a transformation operator to primitive array elements referenced through texture cache.
 *
 * \par Overview
 * TransformIteratorRA wraps a unary conversion functor of type \p ConversionOp and a
 * native device pointer of type <tt>T*</tt>.  \p ValueType references are produced by
 * applying the former to elements of the latter read through texture cache.
 *
 * \par Usage Considerations
 * - Can only be used with primitive types (e.g., \p char, \p int, \p float), with the exception of \p double
 * - Only one TexIteratorRA or TexTransformIteratorRA of a certain \p InputType can be bound at any given time (per host thread)
 *
 * \tparam InputType            The value type of the pointer being wrapped
 * \tparam ConversionOp         Unary functor type for mapping objects of type \p InputType to type \p OutputType.  Must have member <tt>OutputType operator()(const InputType &datum)</tt>.
 * \tparam OutputType           The value type of this iterator
 */
template <
    typename    ValueType,
    typename    ConversionOp,
    typename    T,
    int         UNIQUE_ID,
    typename    difference_type = ptrdiff_t>
class TexTransformIteratorRA
{
public:

    // Required iterator traits
    typedef TexTransformIteratorRA              self_type;              ///< My own type
    typedef difference_type                     difference_type;        ///< Type to express the result of subtracting one iterator from another
    typedef ValueType                           value_type;             ///< The type of the element the iterator can point to
    typedef ValueType*                          pointer;                ///< The type of a pointer to an element the iterator can point to
    typedef ValueType                           reference;              ///< The type of a reference to an element the iterator can point to
    typedef std::random_access_iterator_tag     iterator_category;      ///< The iterator category

private:

    TexIteratorRA<T, UNIQUE_ID, difference_type>   tex_itr;
    ConversionOp                        conversion_op;

public:

    /// Constructor
    __host__ __device__ __forceinline__ TexTransformIteratorRA(
        ConversionOp conversion_op)          ///< Binary transformation functor
    :
        conversion_op(conversion_op)
    {}

    /// Use this iterator to bind \p ptr with a texture reference
    cudaError_t BindTexture(
        T       *ptr,                   ///< Native pointer to wrap that is aligned to cudaDeviceProp::textureAlignment
        size_t  bytes,                  ///< Number of bytes in the range
        size_t  tex_align_offset = 0)   ///< Offset (in items) from \p ptr denoting the position of the iterator
    {
        return tex_itr.BindTexture(ptr, bytes, tex_align_offset);
    }

    /// Unbind this iterator from its texture reference
    cudaError_t UnbindTexture()
    {
        return tex_itr.UnbindTexture();
    }

    /// Postfix increment
    __host__ __device__ __forceinline__ self_type operator++(int)
    {
        self_type retval = *this;
        tex_itr++;
        return retval;
    }

    /// Indirection
    __host__ __device__ __forceinline__ reference operator*()
    {
        return conversion_op(*tex_itr);
    }

    /// Addition
    template <typename Distance>
    __host__ __device__ __forceinline__ self_type operator+(Distance n)
    {
        self_type retval(conversion_op);
        retval.tex_itr = tex_itr + n;
        return retval;
    }

    /// Addition assignment
    template <typename Distance>
    __host__ __device__ __forceinline__ self_type& operator+=(Distance n)
    {
        tex_itr += n;
        return *this;
    }

    /// Subtraction
    template <typename Distance>
    __host__ __device__ __forceinline__ self_type operator-(Distance n)
    {
        self_type retval(conversion_op);
        retval.tex_itr = tex_itr - n;
        return retval;
    }

    /// Subtraction assignment
    template <typename Distance>
    __host__ __device__ __forceinline__ self_type& operator-=(Distance n)
    {
        tex_itr -= n;
        return *this;
    }

    /// Array subscript
    template <typename Distance>
    __host__ __device__ __forceinline__ reference operator[](Distance n)
    {
        return conversion_op(tex_itr[n]);
    }

    /// Structure dereference
    __host__ __device__ __forceinline__ pointer operator->()
    {
        return &conversion_op(*tex_itr);
    }

    /// Equal to
    __host__ __device__ __forceinline__ bool operator==(const self_type& rhs)
    {
        return (tex_itr == rhs.tex_itr);
    }

    /// Not equal to
    __host__ __device__ __forceinline__ bool operator!=(const self_type& rhs)
    {
        return (tex_itr != rhs.tex_itr);
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
