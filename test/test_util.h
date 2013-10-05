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


#pragma once

#if defined(_WIN32) || defined(_WIN64)
    #include <windows.h>
    #undef small            // Windows is terrible for polluting macro namespace
#else
    #include <sys/resource.h>
#endif

#include <cuda_runtime.h>

#include <stdio.h>
#include <math.h>
#include <float.h>

#include <string>
#include <vector>
#include <sstream>
#include <iostream>
#include <limits>

#include <cub/util_debug.cuh>
#include <cub/util_device.cuh>
#include <cub/util_type.cuh>

/******************************************************************************
 * Assertion macros
 ******************************************************************************/

/**
 * Assert equals
 */
#define AssertEquals(a, b) if ((a) != (b)) { std::cerr << "\n(" << __FILE__ << ": " << __LINE__ << ")\n"; exit(1);}


/******************************************************************************
 * Command-line parsing functionality
 ******************************************************************************/

/**
 * Utility for parsing command line arguments
 */
class CommandLineArgs
{
protected:

    std::vector<std::string> keys;
    std::vector<std::string> values;

public:

    /**
     * Constructor
     */
    CommandLineArgs(int argc, char **argv) :
        keys(10),
        values(10)
    {
        using namespace std;

        for (int i = 1; i < argc; i++)
        {
            string arg = argv[i];

            if ((arg[0] != '-') || (arg[1] != '-'))
            {
                continue;
            }

            string::size_type pos;
            string key, val;
            if ((pos = arg.find( '=')) == string::npos) {
                key = string(arg, 2, arg.length() - 2);
                val = "";
            } else {
                key = string(arg, 2, pos - 2);
                val = string(arg, pos + 1, arg.length() - 1);
            }

            keys.push_back(key);
            values.push_back(val);
        }
    }


    /**
     * Checks whether a flag "--<flag>" is present in the commandline
     */
    bool CheckCmdLineFlag(const char* arg_name)
    {
        using namespace std;

        for (int i = 0; i < keys.size(); ++i)
        {
            if (keys[i] == string(arg_name))
                return true;
        }
        return false;
    }


    /**
     * Returns the value specified for a given commandline parameter --<flag>=<value>
     */
    template <typename T>
    void GetCmdLineArgument(const char *arg_name, T &val)
    {
        using namespace std;

        for (int i = 0; i < keys.size(); ++i)
        {
            if (keys[i] == string(arg_name))
            {
                istringstream str_stream(values[i]);
                str_stream >> val;
            }
        }
    }


    /**
     * Returns the values specified for a given commandline parameter --<flag>=<value>,<value>*
     */
    template <typename T>
    void GetCmdLineArguments(const char *arg_name, std::vector<T> &vals)
    {
        using namespace std;

        if (CheckCmdLineFlag(arg_name))
        {
            // Clear any default values
            vals.clear();

            // Recover from multi-value string
            for (int i = 0; i < keys.size(); ++i)
            {
                if (keys[i] == string(arg_name))
                {
                    string val_string(values[i]);
                    istringstream str_stream(val_string);
                    string::size_type old_pos = 0;
                    string::size_type new_pos = 0;

                    // Iterate comma-separated values
                    T val;
                    while ((new_pos = val_string.find(',', old_pos)) != string::npos)
                    {
                        if (new_pos != old_pos)
                        {
                            str_stream.width(new_pos - old_pos);
                            str_stream >> val;
                            vals.push_back(val);
                        }

                        // skip over comma
                        str_stream.ignore(1);
                        old_pos = new_pos + 1;
                    }

                    // Read last value
                    str_stream >> val;
                    vals.push_back(val);
                }
            }
        }
    }


    /**
     * The number of pairs parsed
     */
    int ParsedArgc()
    {
        return keys.size();
    }

    /**
     * Initialize device
     */
    cudaError_t DeviceInit(int dev = -1)
    {
        cudaError_t error = cudaSuccess;

        do
        {
            int deviceCount;
            error = CubDebug(cudaGetDeviceCount(&deviceCount));
            if (error) break;

            if (deviceCount == 0) {
                fprintf(stderr, "No devices supporting CUDA.\n");
                exit(1);
            }
            if (dev < 0)
            {
                GetCmdLineArgument("device", dev);
            }
            if ((dev > deviceCount - 1) || (dev < 0))
            {
                dev = 0;
            }

            error = CubDebug(cudaSetDevice(dev));
            if (error) break;

            size_t free_physmem, total_physmem;
            CubDebugExit(cudaMemGetInfo(&free_physmem, &total_physmem));

            int ptx_version;
            error = CubDebug(cub::PtxVersion(ptx_version));
            if (error) break;

            cudaDeviceProp deviceProp;
            error = CubDebug(cudaGetDeviceProperties(&deviceProp, dev));
            if (error) break;

            if (deviceProp.major < 1) {
                fprintf(stderr, "Device does not support CUDA.\n");
                exit(1);
            }
            if (!CheckCmdLineFlag("quiet")) {
                printf("Using device %d: %s (PTX version %d, SM%d, %d SMs, %lld free / %lld total MB physmem, ECC %s)\n",
                    dev,
                    deviceProp.name,
                    ptx_version,
                    deviceProp.major * 100 + deviceProp.minor * 10,
                    deviceProp.multiProcessorCount,
                    (unsigned long long) free_physmem / 1024 / 1024,
                    (unsigned long long) total_physmem / 1024 / 1024,
                    (deviceProp.ECCEnabled) ? "on" : "off");
                fflush(stdout);
            }

        } while (0);

        return error;
    }
};


/******************************************************************************
 * Random bits generator
 ******************************************************************************/

/**
 * Generates random keys.
 *
 * We always take the second-order byte from rand() because the higher-order
 * bits returned by rand() are commonly considered more uniformly distributed
 * than the lower-order bits.
 *
 * We can decrease the entropy level of keys by adopting the technique
 * of Thearling and Smith in which keys are computed from the bitwise AND of
 * multiple random samples:
 *
 * entropy_reduction    | Effectively-unique bits per key
 * -----------------------------------------------------
 * -1                   | 0
 * 0                    | 32
 * 1                    | 25.95
 * 2                    | 17.41
 * 3                    | 10.78
 * 4                    | 6.42
 * ...                  | ...
 *
 */
template <typename K>
void RandomBits(
    K &key,
    int entropy_reduction = 0,
    int begin_bit = 0,
    int end_bit = sizeof(K) * 8)
{
    const int NUM_BYTES = sizeof(K);
    const int NUM_BITS = NUM_BYTES * 8;

    unsigned char byte_buff[NUM_BYTES];

    if (entropy_reduction == -1)
    {
        memset((void *) &key, 0, sizeof(key));
        return;
    }

    do {
        // Generate random byte_buff
        for (int j = 0; j < NUM_BYTES; j++)
        {
            const int BYTES_REMAINING = NUM_BYTES - j - 1;
            unsigned char byte = 0xff;

            byte <<= CUB_MAX(0, begin_bit - (j * 8));
            byte >>= CUB_MAX(0, NUM_BITS - end_bit - (BYTES_REMAINING * 8));

            for (int i = 0; i <= entropy_reduction; i++)
            {
                byte &= (rand() >> 7);
            }
            byte_buff[j] = byte;
        }

        memcpy(&key, byte_buff, sizeof(K));

    } while (key != key);        // avoids NaNs when generating random floating point numbers
}



/******************************************************************************
 * Console printing utilities
 ******************************************************************************/

/**
 * Helper for casting character types to integers for cout printing
 */
template <typename T>
T CoutCast(T val) { return val; }

int CoutCast(char val) { return val; }

int CoutCast(unsigned char val) { return val; }

int CoutCast(signed char val) { return val; }


/******************************************************************************
 * Test value initialization utilities
 ******************************************************************************/

/**
 * Test problem generation options
 */
enum GenMode
{
    UNIFORM,            // All 2s
    SEQ_INC,            // Sequentially incrementing
    RANDOM,             // Random
};

/**
 * Initialize value
 */
template <typename T>
__host__ __device__ __forceinline__ void InitValue(GenMode gen_mode, T &value, int index = 0)
{
    switch (gen_mode)
    {
#ifndef __CUDA_ARCH__
    case RANDOM:
         RandomBits(value);
         break;
#endif
     case UNIFORM:
        value = 2;
        break;
    case SEQ_INC:
    default:
         value = index;
        break;
    }
}


/******************************************************************************
 * Comparison and ostream operators for CUDA vector types
 ******************************************************************************/

/**
 * Vector1 overloads
 */
#define CUB_VEC_OVERLOAD_1(T)                               \
    /* Ostream output */                                    \
    std::ostream& operator<<(                               \
        std::ostream& os,                                   \
        const T& val)                                       \
    {                                                       \
        os << '(' << CoutCast(val.x) << ')';                \
        return os;                                          \
    }                                                       \
    /* Inequality */                                        \
    __host__ __device__ __forceinline__ bool operator!=(    \
        const T &a,                                         \
        const T &b)                                         \
    {                                                       \
        return (a.x != b.x);                                \
    }                                                       \
    /* Test initialization */                               \
    __host__ __device__ __forceinline__ void InitValue(GenMode gen_mode, T &value, int index = 0)   \
    {                                                       \
        InitValue(gen_mode, value.x, index);                \
    }                                                       \
    /* Max */                                               \
    __host__ __device__ __forceinline__ bool operator>(     \
        const T &a,                                         \
        const T &b)                                         \
    {                                                       \
        return (a.x > b.x);                                 \
    }                                                       \
    /* Summation */                                         \
    __host__ __device__ __forceinline__ T operator+(        \
        const T &a,                                         \
        const T &b)                                         \
    {                                                       \
        T retval = {a.x + b.x};                             \
        return retval;                                      \
    }

/**
 * Vector2 overloads
 */
#define CUB_VEC_OVERLOAD_2(T)                               \
    /* Ostream output */                                    \
    std::ostream& operator<<(                               \
        std::ostream& os,                                   \
        const T& val)                                       \
    {                                                       \
        os << '('                                           \
            << CoutCast(val.x) << ','                       \
            << CoutCast(val.y) << ')';                      \
        return os;                                          \
    }                                                       \
    /* Inequality */                                        \
    __host__ __device__ __forceinline__ bool operator!=(    \
        const T &a,                                         \
        const T &b)                                         \
    {                                                       \
        return (a.x != b.x) &&                              \
            (a.y != b.y);                                   \
    }                                                       \
    /* Test initialization */                               \
    __host__ __device__ __forceinline__ void InitValue(GenMode gen_mode, T &value, int index = 0)   \
    {                                                       \
        InitValue(gen_mode, value.x, index);                \
        InitValue(gen_mode, value.y, index);                \
    }                                                       \
    /* Max */                                               \
    __host__ __device__ __forceinline__ bool operator>(     \
        const T &a,                                         \
        const T &b)                                         \
    {                                                       \
        return (a.x > b.x);                                 \
    }                                                       \
    /* Summation */                                         \
    __host__ __device__ __forceinline__ T operator+(        \
        const T &a,                                         \
        const T &b)                                         \
    {                                                       \
        T retval = {                                        \
            a.x + b.x,                                      \
            a.y + b.y};                                     \
        return retval;                                      \
    }


/**
 * Vector3 overloads
 */
#define CUB_VEC_OVERLOAD_3(T)                               \
    /* Ostream output */                                    \
    std::ostream& operator<<(                               \
        std::ostream& os,                                   \
        const T& val)                                       \
    {                                                       \
        os << '('                                           \
            << CoutCast(val.x) << ','                       \
            << CoutCast(val.y) << ','                       \
            << CoutCast(val.z) << ')';                      \
        return os;                                          \
    }                                                       \
    /* Inequality */                                        \
    __host__ __device__ __forceinline__ bool operator!=(    \
        const T &a,                                         \
        const T &b)                                         \
    {                                                       \
        return (a.x != b.x) &&                              \
            (a.y != b.y) &&                                 \
            (a.z != b.z);                                   \
    }                                                       \
    /* Test initialization */                               \
    __host__ __device__ __forceinline__ void InitValue(GenMode gen_mode, T &value, int index = 0)   \
    {                                                       \
        InitValue(gen_mode, value.x, index);                \
        InitValue(gen_mode, value.y, index);                \
        InitValue(gen_mode, value.z, index);                \
    }                                                       \
    /* Max */                                               \
    __host__ __device__ __forceinline__ bool operator>(     \
        const T &a,                                         \
        const T &b)                                         \
    {                                                       \
        return (a.x > b.x);                                 \
    }                                                       \
    /* Summation */                                         \
    __host__ __device__ __forceinline__ T operator+(        \
        const T &a,                                         \
        const T &b)                                         \
    {                                                       \
        T retval = {                                        \
            a.x + b.x,                                      \
            a.y + b.y,                                      \
            a.z + b.z};                                     \
        return retval;                                      \
    }

/**
 * Vector4 overloads
 */
#define CUB_VEC_OVERLOAD_4(T)                               \
    /* Ostream output */                                    \
    std::ostream& operator<<(                               \
        std::ostream& os,                                   \
        const T& val)                                       \
    {                                                       \
        os << '('                                           \
            << CoutCast(val.x) << ','                       \
            << CoutCast(val.y) << ','                       \
            << CoutCast(val.z) << ','                       \
            << CoutCast(val.w) << ')';                      \
        return os;                                          \
    }                                                       \
    /* Inequality */                                        \
    __host__ __device__ __forceinline__ bool operator!=(    \
        const T &a,                                         \
        const T &b)                                         \
    {                                                       \
        return (a.x != b.x) &&                              \
            (a.y != b.y) &&                                 \
            (a.z != b.z) &&                                 \
            (a.w != b.w);                                   \
    }                                                       \
    /* Test initialization */                               \
    __host__ __device__ __forceinline__ void InitValue(GenMode gen_mode, T &value, int index = 0)   \
    {                                                       \
        InitValue(gen_mode, value.x, index);                \
        InitValue(gen_mode, value.y, index);                \
        InitValue(gen_mode, value.z, index);                \
        InitValue(gen_mode, value.w, index);                \
    }                                                       \
    /* Max */                                               \
    __host__ __device__ __forceinline__ bool operator>(     \
        const T &a,                                         \
        const T &b)                                         \
    {                                                       \
    	return (a.x > b.x);                                 \
    }                                                       \
    /* Summation */                                         \
    __host__ __device__ __forceinline__ T operator+(        \
        const T &a,                                         \
        const T &b)                                         \
    {                                                       \
        T retval = {                                        \
            a.x + b.x,                                      \
            a.y + b.y,                                      \
            a.z + b.z,                                      \
            a.w + b.w};                                     \
        return retval;                                      \
    }

/**
 * All vector overloads
 */
#define CUB_VEC_OVERLOAD(BASE_T)                            \
    CUB_VEC_OVERLOAD_1(BASE_T##1)                           \
    CUB_VEC_OVERLOAD_2(BASE_T##2)                           \
    CUB_VEC_OVERLOAD_3(BASE_T##3)                           \
    CUB_VEC_OVERLOAD_4(BASE_T##4)

/**
 * Define for types
 */
CUB_VEC_OVERLOAD(char)
CUB_VEC_OVERLOAD(short)
CUB_VEC_OVERLOAD(int)
CUB_VEC_OVERLOAD(long)
CUB_VEC_OVERLOAD(longlong)
CUB_VEC_OVERLOAD(uchar)
CUB_VEC_OVERLOAD(ushort)
CUB_VEC_OVERLOAD(uint)
CUB_VEC_OVERLOAD(ulong)
CUB_VEC_OVERLOAD(ulonglong)
CUB_VEC_OVERLOAD(float)
CUB_VEC_OVERLOAD(double)


//---------------------------------------------------------------------
// Complex data type TestFoo
//---------------------------------------------------------------------

/**
 * TestFoo complex data type
 */
struct TestFoo
{
    long long   x;
    int         y;
    short       z;
    char        w;

    // Factory
    static __host__ __device__ __forceinline__ TestFoo MakeTestFoo(long long x, int y, short z, char w)
    {
        TestFoo retval = {x, y, z, w};
        return retval;
    }

    // Assignment from int operator
    __host__ __device__ __forceinline__ TestFoo operator =(int b)
    {
        x = b;
        y = b;
        z = b;
        w = b;
        return *this;
    }

    // Summation operator
    __host__ __device__ __forceinline__ TestFoo operator+(const TestFoo &b) const
    {
        return MakeTestFoo(x + b.x, y + b.y, z + b.z, w + b.w);
    }

    // Inequality operator
    __host__ __device__ __forceinline__ bool operator !=(const TestFoo &b)
    {
        return (x != b.x) || (y != b.y) || (z != b.z) || (w != b.w);
    }
};


/**
 * TestFoo max operator
 */
bool __host__ __device__ __forceinline__ operator>(const TestFoo& first, const TestFoo& second)
{
    return (first.x > second.x);
}


/**
 * TestFoo ostream operator
 */
std::ostream& operator<<(std::ostream& os, const TestFoo& val)
{
    os << '(' << val.x << ',' << val.y << ',' << val.z << ',' << CoutCast(val.w) << ')';
    return os;
}

/**
 * TestFoo test initialization
 */
__host__ __device__ __forceinline__ void InitValue(GenMode gen_mode, TestFoo &value, int index = 0)
{
    InitValue(gen_mode, value.x, index);
    InitValue(gen_mode, value.y, index);
    InitValue(gen_mode, value.z, index);
    InitValue(gen_mode, value.w, index);
}


//---------------------------------------------------------------------
// Complex data type TestBar (with optimizations for fence-free warp-synchrony)
//---------------------------------------------------------------------

/**
 * TestBar complex data type
 */
struct TestBar
{
    long long       x;
    int             y;

    // Constructor
    __host__ __device__ __forceinline__ TestBar() : x(0), y(0)
    {}

    // Constructor
    __host__ __device__ __forceinline__ TestBar(long long x, int y) : x(x), y(y)
    {}

    // Assignment from int operator
    __host__ __device__ __forceinline__ TestBar operator =(int b)
    {
        x = b;
        y = b;
        return *this;
    }

    // Summation operator
    __host__ __device__ __forceinline__ TestBar operator+(const TestBar &b) const
    {
        return TestBar(x + b.x, y + b.y);
    }

    // Inequality operator
    __host__ __device__ __forceinline__ bool operator !=(const TestBar &b)
    {
        return (x != b.x) || (y != b.y);
    }
};

/* todo: uncomment once Fermi codegen bug is fixed

/// Load (simply defer to loading individual items)
template <cub::PtxLoadModifier MODIFIER>
__device__ __forceinline__ TestBar ThreadLoad(TestBar *ptr)
{
    TestBar retval;
    retval.x = cub::ThreadLoad<MODIFIER>(&(ptr->x));
    retval.y = cub::ThreadLoad<MODIFIER>(&(ptr->y));
    return retval;
}

 /// Store (simply defer to storing individual items)
template <cub::PtxLoadModifier MODIFIER>
__device__ __forceinline__ void ThreadStore(
    TestBar *ptr,
    TestBar val)
{
    // Always write partial first
    cub::ThreadStore<MODIFIER>(&(ptr->x), val.x);
    cub::ThreadStore<MODIFIER>(&(ptr->y), val.y);
}

*/

/**
 * TestBar max operator
 */
bool __host__ __device__ __forceinline__ operator>(const TestBar& first, const TestBar& second)
{
    return (first.x > second.x);
}

/**
 * TestBar ostream operator
 */
std::ostream& operator<<(std::ostream& os, const TestBar& val)
{
    os << '(' << val.x << ',' << val.y << ')';
    return os;
}

/**
 * TestBar test initialization
 */
__host__ __device__ __forceinline__ void InitValue(GenMode gen_mode, TestBar &value, int index = 0)
{
    InitValue(gen_mode, value.x, index);
    InitValue(gen_mode, value.y, index);
}


/******************************************************************************
 * Helper routines for list comparison and display
 ******************************************************************************/


/**
 * Compares the equivalence of two arrays
 */
template <typename S, typename T, typename SizeT>
int CompareResults(T* computed, S* reference, SizeT len, bool verbose = true)
{
    for (SizeT i = 0; i < len; i++)
    {
        if (computed[i] != reference[i])
        {
            if (verbose) std::cout << "INCORRECT: [" << i << "]: "
                << CoutCast(computed[i]) << " != "
                << CoutCast(reference[i]);
            return 1;
        }
    }
    return 0;
}


/**
 * Compares the equivalence of two arrays
 */
template <typename SizeT>
int CompareResults(float* computed, float* reference, SizeT len, bool verbose = true)
{
    int retval = 0;
    for (SizeT i = 0; i < len; i++)
    {
        float difference = std::abs(computed[i]-reference[i]);
        float fraction = difference / std::abs(reference[i]);

        if (fraction > 0.0001)
        {
            if (verbose) std::cout << "INCORRECT: [" << i << "]: "
                << CoutCast(computed[i]) << " != "
                << CoutCast(reference[i]) << " (difference:" << difference << ", fraction: " << fraction << ")";
            return 1;
        }
    }

    if (!retval) printf("CORRECT\n");
    return retval;
}


/**
 * Compares the equivalence of two arrays
 */
template <typename SizeT>
int CompareResults(cub::NullType* computed, cub::NullType* reference, SizeT len, bool verbose = true)
{
    printf("CORRECT\n");
    return 0;
}

/**
 * Compares the equivalence of two arrays
 */
template <typename SizeT>
int CompareResults(double* computed, double* reference, SizeT len, bool verbose = true)
{
    int retval = 0;
    for (SizeT i = 0; i < len; i++)
    {
        double difference = std::abs(computed[i]-reference[i]);
        double fraction = difference / std::abs(reference[i]);

        if (fraction > 0.0001)
        {
            if (verbose) std::cout << "INCORRECT: [" << i << "]: "
                << CoutCast(computed[i]) << " != "
                << CoutCast(reference[i]) << " (difference:" << difference << ", fraction: " << fraction << ")";
            return 1;
        }
    }

    if (!retval) printf("CORRECT\n");
    return retval;
}


/**
 * Verify the contents of a device array match those
 * of a host array
 */
int CompareDeviceResults(
    cub::NullType *h_reference,
    cub::NullType *d_data,
    size_t num_items,
    bool verbose = true,
    bool display_data = false)
{
    return 0;
}


/**
 * Verify the contents of a device array match those
 * of a host array
 */
template <typename S, typename T>
int CompareDeviceResults(
    S *h_reference,
    T *d_data,
    size_t num_items,
    bool verbose = true,
    bool display_data = false)
{
    // Allocate array on host
    T *h_data = (T*) malloc(num_items * sizeof(T));

    // Copy data back
    cudaMemcpy(h_data, d_data, sizeof(T) * num_items, cudaMemcpyDeviceToHost);

    // Display data
    if (display_data)
    {
        printf("Reference:\n");
        for (int i = 0; i < num_items; i++)
        {
            std::cout << CoutCast(h_reference[i]) << ", ";
        }
        printf("\n\nData:\n");
        for (int i = 0; i < num_items; i++)
        {
            std::cout << CoutCast(h_data[i]) << ", ";
        }
        printf("\n\n");
    }

    // Check
    int retval = CompareResults(h_data, h_reference, num_items, verbose);

    // Cleanup
    if (h_data) free(h_data);

    return retval;
}


/**
 * Verify the contents of a device array match those
 * of a device array
 */
template <typename T>
int CompareDeviceDeviceResults(
    T *d_reference,
    T *d_data,
    size_t num_items,
    bool verbose = true,
    bool display_data = false)
{
    // Allocate array on host
    T *h_reference = (T*) malloc(num_items * sizeof(T));
    T *h_data = (T*) malloc(num_items * sizeof(T));

    // Copy data back
    cudaMemcpy(h_reference, d_reference, sizeof(T) * num_items, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_data, d_data, sizeof(T) * num_items, cudaMemcpyDeviceToHost);

    // Display data
    if (display_data) {
        printf("Reference:\n");
        for (int i = 0; i < num_items; i++)
        {
            std::cout << CoutCast(h_reference[i]) << ", ";
        }
        printf("\n\nData:\n");
        for (int i = 0; i < num_items; i++)
        {
            std::cout << CoutCast(h_data[i]) << ", ";
        }
        printf("\n\n");
    }

    // Check
    int retval = CompareResults(h_data, h_reference, num_items, verbose);

    // Cleanup
    if (h_reference) free(h_reference);
    if (h_data) free(h_data);

    return retval;
}


/**
 * Print the contents of a host array
 */
void DisplayResults(
    cub::NullType   *h_data,
    size_t          num_items)
{}


/**
 * Print the contents of a host array
 */
template <typename T>
void DisplayResults(
    T *h_data,
    size_t num_items)
{
    // Display data
    for (int i = 0; i < num_items; i++)
    {
        std::cout << CoutCast(h_data[i]) << ", ";
    }
    printf("\n");
}


/**
 * Print the contents of a device array
 */
template <typename T>
void DisplayDeviceResults(
    T *d_data,
    size_t num_items)
{
    // Allocate array on host
    T *h_data = (T*) malloc(num_items * sizeof(T));

    // Copy data back
    cudaMemcpy(h_data, d_data, sizeof(T) * num_items, cudaMemcpyDeviceToHost);

    DisplayResults(h_data, num_items);

    // Cleanup
    if (h_data) free(h_data);
}



/******************************************************************************
 * Timing
 ******************************************************************************/


struct CpuTimer
{
#if defined(_WIN32) || defined(_WIN64)

    LARGE_INTEGER ll_freq;
    LARGE_INTEGER ll_start;
    LARGE_INTEGER ll_stop;

    CpuTimer()
    {
        QueryPerformanceFrequency(&ll_freq);
    }

    void Start()
    {
        QueryPerformanceCounter(&ll_start);
    }

    void Stop()
    {
        QueryPerformanceCounter(&ll_stop);
    }

    float ElapsedMillis()
    {
        double start = double(ll_start.QuadPart) / double(ll_freq.QuadPart);
        double stop  = double(ll_stop.QuadPart) / double(ll_freq.QuadPart);

        return (stop - start) * 1000;
    }

#else

    rusage start;
    rusage stop;

    void Start()
    {
        getrusage(RUSAGE_SELF, &start);
    }

    void Stop()
    {
        getrusage(RUSAGE_SELF, &stop);
    }

    float ElapsedMillis()
    {
        float sec = stop.ru_utime.tv_sec - start.ru_utime.tv_sec;
        float usec = stop.ru_utime.tv_usec - start.ru_utime.tv_usec;

        return (sec * 1000) + (usec / 1000);
    }

#endif
};

struct GpuTimer
{
    cudaEvent_t start;
    cudaEvent_t stop;

    GpuTimer()
    {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }

    ~GpuTimer()
    {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    void Start()
    {
        cudaEventRecord(start, 0);
    }

    void Stop()
    {
        cudaEventRecord(stop, 0);
    }

    float ElapsedMillis()
    {
        float elapsed;
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed, start, stop);
        return elapsed;
    }
};
