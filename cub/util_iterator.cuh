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

#include "thread/thread_load.cuh"
#include "util_debug.cuh"
#include "util_namespace.cuh"

/// Optional outer namespace(s)
CUB_NS_PREFIX

/// CUB namespace
namespace cub {

/**
 * \addtogroup UtilModule
 * @{
 */

/**
 * \brief A simple random-access transform iterator for applying a transformation operator.
 *
 * \par Overview
 * TransformIteratorRA is a random-access iterator that wraps both a native
 * device pointer of type <tt>InputType*</tt> and a unary conversion functor of
 * type \p ConversionOp. \p OutputType references are made by pulling \p InputType
 * values through the \p ConversionOp instance.
 *
 * \tparam InputType            The value type of the pointer being wrapped
 * \tparam ConversionOp         Unary functor type for mapping objects of type \p InputType to type \p OutputType.  Must have member <tt>OutputType operator()(const InputType &datum)</tt>.
 * \tparam OutputType           The value type of this iterator
 */
template <typename OutputType, typename ConversionOp, typename InputType>
class TransformIteratorRA
{
public:

#ifndef DOXYGEN_SHOULD_SKIP_THIS    // Do not document

    typedef TransformIteratorRA                 self_type;
    typedef OutputType                          value_type;
    typedef OutputType                          reference;
    typedef OutputType*                         pointer;
    typedef std::random_access_iterator_tag     iterator_category;
    typedef int                                 difference_type;

#endif  // DOXYGEN_SHOULD_SKIP_THIS

private:

    ConversionOp    conversion_op;
    InputType*      ptr;

public:

    /**
     * \brief Constructor
     * @param ptr Native pointer to wrap
     * @param conversion_op Binary transformation functor
     */
    __host__ __device__ __forceinline__ TransformIteratorRA(InputType* ptr, ConversionOp conversion_op) :
        conversion_op(conversion_op),
        ptr(ptr) {}

#ifndef DOXYGEN_SHOULD_SKIP_THIS    // Do not document

    __host__ __device__ __forceinline__ self_type operator++()
    {
        self_type i = *this;
        ptr++;
        return i;
    }

    __host__ __device__ __forceinline__ self_type operator++(int junk)
    {
        ptr++;
        return *this;
    }

    __host__ __device__ __forceinline__ reference operator*()
    {
        return conversion_op(*ptr);
    }

    template <typename SizeT>
    __host__ __device__ __forceinline__ self_type operator+(SizeT n)
    {
        TransformIteratorRA retval(ptr + n, conversion_op);
        return retval;
    }

    template <typename SizeT>
    __host__ __device__ __forceinline__ self_type operator-(SizeT n)
    {
        TransformIteratorRA retval(ptr - n, conversion_op);
        return retval;
    }

    template <typename SizeT>
    __host__ __device__ __forceinline__ reference operator[](SizeT n)
    {
        return conversion_op(ptr[n]);
    }

    __host__ __device__ __forceinline__ pointer operator->()
    {
        return &conversion_op(*ptr);
    }

    __host__ __device__ __forceinline__ bool operator==(const self_type& rhs)
    {
        return (ptr == rhs.ptr);
    }

    __host__ __device__ __forceinline__ bool operator!=(const self_type& rhs)
    {
        return (ptr != rhs.ptr);
    }

#endif // DOXYGEN_SHOULD_SKIP_THIS

};



/******************************************************************************
 * Texture references
 *****************************************************************************/

#ifndef DOXYGEN_SHOULD_SKIP_THIS    // Do not document

// Anonymous namespace
namespace {

/// Templated Texture reference type for multiplicand vector
template <typename T>
struct TexIteratorRef
{
    // Texture reference type
    typedef texture<T, cudaTextureType1D, cudaReadModeElementType> TexRef;

    static TexRef ref;

    /**
     * Bind texture
     */
    static cudaError_t BindTexture(void *d_in, size_t &offset)
    {
        cudaChannelFormatDesc tex_desc = cudaCreateChannelDesc<T>();
        if (d_in)
        {
            return (CubDebug(cudaBindTexture(&offset, ref, d_in, tex_desc)));
        }
        return cudaSuccess;
    }

    /**
     * Unbind textures
     */
    static cudaError_t UnbindTexture()
    {
        return CubDebug(cudaUnbindTexture(ref));
    }
};

// Texture reference definitions
template <typename Value>
typename TexIteratorRef<Value>::TexRef TexIteratorRef<Value>::ref = 0;

} // Anonymous namespace



/**
 * Define HasTexBinding structure for testing the presence of nested
 * TexBindingTag type names within data types
 */
CUB_DEFINE_DETECT_NESTED_TYPE(HasTexBinding, TexBindingTag)


/// Helper for (un)binding iterator textures (specialized for iterators that can have texture bound)
template <
    typename InputIteratorRA,
    bool HAS_TEX_BINDING = HasTexBinding<InputIteratorRA>::VALUE>
struct TexIteratorBinder
{
    static cudaError_t Bind(InputIteratorRA d_itr)
    {
        return d_itr.BindTexture();
    }

    static cudaError_t Unbind(InputIteratorRA d_itr)
    {
        return d_itr.UnindTexture();
    }
};

/// Helper for (un)binding iterator textures (specialized for iterators that cannot have texture bound)
template <typename InputIteratorRA>
struct TexIteratorBinder<InputIteratorRA, false>
{
    static cudaError_t Bind(InputIteratorRA d_itr)
    {
        return cudaSuccess;
    }

    static cudaError_t Unbind(InputIteratorRA d_itr)
    {
        return cudaSuccess;
    }
};

/// Bind iterator texture if supported (otherwise do nothing)
template <typename InputIteratorRA>
__host__ cudaError_t BindIteratorTexture(InputIteratorRA d_itr)
{
    return TexIteratorBinder<InputIteratorRA>::Bind(d_itr);
}

/// Unbind iterator texture if supported (otherwise do nothing)
template <typename InputIteratorRA>
__host__ cudaError_t UnbindIteratorTexture(InputIteratorRA d_itr)
{
    return TexIteratorBinder<InputIteratorRA>::Bind(d_itr);
}

#endif // DOXYGEN_SHOULD_SKIP_THIS



/**
 * \brief A simple random-access iterator for loading primitive values through texture cache.
 *
 * \par Overview
 * TexIteratorRA is a random-access iterator that wraps a native
 * device pointer of type <tt>T*</tt>. References made through TexIteratorRA
 * causes values to be pulled through texture cache.
 *
 * \par Usage Considerations
 * - Can only be used with primitive types (e.g., \p char, \p int, \p float), with the exception of \p double
 * - Only one TexIteratorRA or TexIteratorRA of a certain \p InputType can be bound at any given time (per host thread)
 *
 * \tparam InputType            The value type of the pointer being wrapped
 * \tparam ConversionOp         Unary functor type for mapping objects of type \p InputType to type \p OutputType.  Must have member <tt>OutputType operator()(const InputType &datum)</tt>.
 * \tparam OutputType           The value type of this iterator
 */
template <typename T>
class TexIteratorRA
{
public:
#ifndef DOXYGEN_SHOULD_SKIP_THIS    // Do not document

    typedef TexIteratorRA                       self_type;
    typedef T                                   value_type;
    typedef T                                   reference;
    typedef T*                                  pointer;
    typedef std::random_access_iterator_tag     iterator_category;
    typedef int                                 difference_type;

#endif // DOXYGEN_SHOULD_SKIP_THIS

    /// Tag identifying iterator type as being texture-bindable
    typedef void TexBindingTag;

private:

    T*      ptr;
    size_t  offset;

public:

    /**
     * \brief Constructor
     * @param ptr Native pointer to wrap
     */
    __host__ __device__ __forceinline__ TexIteratorRA(T* ptr) :
        ptr(ptr),
        offset(0) {}

    /// \brief Bind iterator to texture reference
    cudaError_t BindTexture()
    {
        return TexIteratorRef<T>::BindTexture(ptr, offset);
    }


    /// \brief Unbind iterator to texture reference
    cudaError_t UnbindTexture()
    {
        return TexIteratorRef<T>::UnbindTexture();
    }

#ifndef DOXYGEN_SHOULD_SKIP_THIS    // Do not document

    __host__ __device__ __forceinline__ self_type operator++()
    {
        self_type i = *this;
        ptr++;
        offset++;
        return i;
    }

    __host__ __device__ __forceinline__ self_type operator++(int junk)
    {
        ptr++;
        offset++;
        return *this;
    }

    __host__ __device__ __forceinline__ reference operator*()
    {
#if (CUB_PTX_ARCH == 0)
        // Simply dereference the pointer on the host
        return *ptr;
#elif (CUB_PTX_ARCH < 350)
        // Use the texture reference
        return tex1Dfetch(TexIteratorRef<T>::ref, offset);
#else
        // Use LDG
        return ThreadLoad<PTX_LOAD_LDG>(ptr);
#endif
    }

    template <typename SizeT>
    __host__ __device__ __forceinline__ self_type operator+(SizeT n)
    {
        TexIteratorRA retval(ptr + n);
        retval.offset = offset + n;
        return retval;
    }

    template <typename SizeT>
    __host__ __device__ __forceinline__ self_type operator-(SizeT n)
    {
        TexIteratorRA retval(ptr - n);
        retval.offset = offset - n;
        return retval;
    }

    template <typename SizeT>
    __host__ __device__ __forceinline__ reference operator[](SizeT n)
    {
#if (CUB_PTX_ARCH == 0)
        // Simply dereference the pointer on the host
        return ptr[n];
#elif (CUB_PTX_ARCH < 350)
        // Use the texture reference
        return tex1Dfetch(TexIteratorRef<T>::ref, offset + n);
#else
        // Use LDG
        return ThreadLoad<PTX_LOAD_LDG>(ptr + n);
#endif
    }

    __host__ __device__ __forceinline__ pointer operator->()
    {
#if (CUB_PTX_ARCH == 0)
        // Simply dereference the pointer on the host
        return &(*ptr);
#elif (CUB_PTX_ARCH < 350)
        // Use the texture reference
        return &(tex1Dfetch(TexIteratorRef<T>::ref, offset));
#else
        // Use LDG
        return &(ThreadLoad<PTX_LOAD_LDG>(ptr));
#endif
    }

    __host__ __device__ __forceinline__ bool operator==(const self_type& rhs)
    {
        return (ptr == rhs.ptr);
    }

    __host__ __device__ __forceinline__ bool operator!=(const self_type& rhs)
    {
        return (ptr != rhs.ptr);
    }

#endif // DOXYGEN_SHOULD_SKIP_THIS

};


/**
 * \brief A simple random-access transform iterator for loading primitive values through texture cache and and subsequently applying a transformation operator.
 *
 * \par Overview
 * TexTransformIteratorRA is a random-access iterator that wraps both a native
 * device pointer of type <tt>InputType*</tt> and a unary conversion functor of
 * type \p ConversionOp. \p OutputType references are made by pulling \p InputType
 * values through the texture cache and then transformed them using the
 * \p ConversionOp instance.
 *
 * \par Usage Considerations
 * - Can only be used with primitive types (e.g., \p char, \p int, \p float), with the exception of \p double
 * - Only one TexIteratorRA or TexTransformIteratorRA of a certain \p InputType can be bound at any given time (per host thread)
 *
 * \tparam InputType            The value type of the pointer being wrapped
 * \tparam ConversionOp         Unary functor type for mapping objects of type \p InputType to type \p OutputType.  Must have member <tt>OutputType operator()(const InputType &datum)</tt>.
 * \tparam OutputType           The value type of this iterator
 */
template <typename OutputType, typename ConversionOp, typename InputType>
class TexTransformIteratorRA
{
public:

#ifndef DOXYGEN_SHOULD_SKIP_THIS    // Do not document

    typedef TexTransformIteratorRA              self_type;
    typedef OutputType                          value_type;
    typedef OutputType                          reference;
    typedef OutputType*                         pointer;
    typedef std::random_access_iterator_tag     iterator_category;
    typedef int                                 difference_type;

#endif  // DOXYGEN_SHOULD_SKIP_THIS

    /// Tag identifying iterator type as being texture-bindable
    typedef void TexBindingTag;

private:

    ConversionOp    conversion_op;
    InputType*      ptr;
    size_t          offset;

public:

    /**
     * \brief Constructor
     * @param ptr Native pointer to wrap
     * @param conversion_op Binary transformation functor
     */
    __host__ __device__ __forceinline__ TexTransformIteratorRA(InputType* ptr, ConversionOp conversion_op) :
        conversion_op(conversion_op),
        ptr(ptr),
        offset(0) {}

    /// \brief Bind iterator to texture reference
    cudaError_t BindTexture()
    {
        return TexIteratorRef<InputType>::BindTexture(ptr, offset);
    }

    /// \brief Unbind iterator to texture reference
    cudaError_t UnbindTexture()
    {
        return TexIteratorRef<InputType>::UnbindTexture();
    }

#ifndef DOXYGEN_SHOULD_SKIP_THIS    // Do not document

    __host__ __device__ __forceinline__ self_type operator++()
    {
        self_type i = *this;
        ptr++;
        offset++;
        return i;
    }

    __host__ __device__ __forceinline__ self_type operator++(int junk)
    {
        ptr++;
        offset++;
        return *this;
    }

    __host__ __device__ __forceinline__ reference operator*()
    {
#if (CUB_PTX_ARCH == 0)
        // Simply dereference the pointer on the host
        return conversion_op(*ptr);
/*#elif (CUB_PTX_ARCH <= 350)
        // Use LDG
        return conversion_op(ThreadLoad<PTX_LOAD_LDG>(ptr));
*/#else
        // Use the texture reference
        return conversion_op(tex1Dfetch(TexIteratorRef<InputType>::ref, offset));
#endif
    }

    template <typename SizeT>
    __host__ __device__ __forceinline__ self_type operator+(SizeT n)
    {
        TexTransformIteratorRA retval(ptr + n, conversion_op);
        retval.offset = offset + n;
        return retval;
    }

    template <typename SizeT>
    __host__ __device__ __forceinline__ self_type operator-(SizeT n)
    {
        TexTransformIteratorRA retval(ptr - n, conversion_op);
        retval.offset = offset - n;
        return retval;
    }

    template <typename SizeT>
    __host__ __device__ __forceinline__ reference operator[](SizeT n)
    {
#if (CUB_PTX_ARCH == 0)
        // Simply dereference the pointer on the host
        return conversion_op(ptr[n]);
/*#elif (CUB_PTX_ARCH >= 350)
        // Use LDG
        return conversion_op(ThreadLoad<PTX_LOAD_LDG>(ptr + n));
*/#else
        // Use the texture reference
        return conversion_op(tex1Dfetch(TexIteratorRef<InputType>::ref, offset + n));
#endif
    }

    __host__ __device__ __forceinline__ pointer operator->()
    {
#if (CUB_PTX_ARCH == 0)
        // Simply dereference the pointer on the host
        return &conversion_op(*ptr);
/*#elif (CUB_PTX_ARCH >= 350)
        // Use LDG
        return &conversion_op(ThreadLoad<PTX_LOAD_LDG>(ptr));
*/#else
        // Use the texture reference
        return &conversion_op(tex1Dfetch(TexIteratorRef<InputType>::ref, offset));
#endif
    }

    __host__ __device__ __forceinline__ bool operator==(const self_type& rhs)
    {
        return (ptr == rhs.ptr);
    }

    __host__ __device__ __forceinline__ bool operator!=(const self_type& rhs)
    {
        return (ptr != rhs.ptr);
    }

#endif // DOXYGEN_SHOULD_SKIP_THIS

};




/** @} */       // end group UtilModule

}               // CUB namespace
CUB_NS_POSTFIX  // Optional outer namespace(s)
