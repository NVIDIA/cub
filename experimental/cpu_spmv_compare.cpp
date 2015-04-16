/******************************************************************************
 * Copyright (c) 2011-2015, NVIDIA CORPORATION.  All rights reserved.
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
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIAeBILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

/******************************************************************************
 *
 *
 * cl.exe cpu_spmv_compare.cpp -DCUB_MKL mkl_intel_lp64.lib mkl_intel_thread.lib  mkl_core.lib libiomp5md.lib /fp:strict /MT /O2 /openmp
 * cl.exe cpu_spmv_compare.cpp /fp:strict /MT /O2 /openmp
 *
 * g++ cpu_spmv_compare.cpp -DCUB_MKL -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm -ffloat-store -O3 -fopenmp
 * g++ cpu_spmv_compare.cpp -lm -ffloat-store -O3 -fopenmp
 *
 * icpc cpu_spmv_compare.cpp -mkl -openmp -DCUB_MKL -O3 -o spmv_omp.out -lrt
 * export KMP_AFFINITY=granularity=core,compact
 *
 *
 *
 *
 ******************************************************************************/


//---------------------------------------------------------------------
// SpMV comparison tool
//---------------------------------------------------------------------

#if defined(_WIN32) || defined(_WIN64)
    #include <windows.h>
    #undef min       // Windows is terrible for polluting macro namespace
    #undef max       // Windows is terrible for polluting macro namespace
    #undef small     // Windows is terrible for polluting macro namespace
#else
    #include <sys/resource.h>
    #include <time.h>
#endif

#include <omp.h>

#include <stdio.h>
#include <map>
#include <vector>
#include <algorithm>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <iostream>

#ifdef CUB_MKL
    #include <mkl.h>
#endif


#include "sparse_matrix.h"


//---------------------------------------------------------------------
// Globals, constants, and type declarations
//---------------------------------------------------------------------

bool                    g_quiet         = false;        // Whether to display stats in CSV format
bool                    g_verbose       = false;        // Whether to display output to console
bool                    g_verbose2      = false;        // Whether to display input to console
int                     g_omp_threads   = -1;         // Number of openMP threads



//---------------------------------------------------------------------
// Utility types
//---------------------------------------------------------------------

struct int2
{
    int x;
    int y;
};


/**
 * Counting iterator
 */
template <
    typename ValueType,
    typename OffsetT = ptrdiff_t>
struct CountingInputIterator
{
    // Required iterator traits
    typedef CountingInputIterator               self_type;              ///< My own type
    typedef OffsetT                             difference_type;        ///< Type to express the result of subtracting one iterator from another
    typedef ValueType                           value_type;             ///< The type of the element the iterator can point to
    typedef ValueType*                          pointer;                ///< The type of a pointer to an element the iterator can point to
    typedef ValueType                           reference;              ///< The type of a reference to an element the iterator can point to
    typedef std::random_access_iterator_tag     iterator_category;      ///< The iterator category

    ValueType val;

    /// Constructor
    inline CountingInputIterator(
        const ValueType &val)          ///< Starting value for the iterator instance to report
    :
        val(val)
    {}

    /// Postfix increment
    inline self_type operator++(int)
    {
        self_type retval = *this;
        val++;
        return retval;
    }

    /// Prefix increment
    inline self_type operator++()
    {
        val++;
        return *this;
    }

    /// Indirection
    inline reference operator*() const
    {
        return val;
    }

    /// Addition
    template <typename Distance>
    inline self_type operator+(Distance n) const
    {
        self_type retval(val + n);
        return retval;
    }

    /// Addition assignment
    template <typename Distance>
    inline self_type& operator+=(Distance n)
    {
        val += n;
        return *this;
    }

    /// Subtraction
    template <typename Distance>
    inline self_type operator-(Distance n) const
    {
        self_type retval(val - n);
        return retval;
    }

    /// Subtraction assignment
    template <typename Distance>
    inline self_type& operator-=(Distance n)
    {
        val -= n;
        return *this;
    }

    /// Distance
    inline difference_type operator-(self_type other) const
    {
        return val - other.val;
    }

    /// Array subscript
    template <typename Distance>
    inline reference operator[](Distance n) const
    {
        return val + n;
    }

    /// Structure dereference
    inline pointer operator->()
    {
        return &val;
    }

    /// Equal to
    inline bool operator==(const self_type& rhs)
    {
        return (val == rhs.val);
    }

    /// Not equal to
    inline bool operator!=(const self_type& rhs)
    {
        return (val != rhs.val);
    }

    /// ostream operator
    friend std::ostream& operator<<(std::ostream& os, const self_type& itr)
    {
        os << "[" << itr.val << "]";
        return os;
    }
};


/**
 * Utility for parsing command line arguments
 */
struct CommandLineArgs
{

    std::vector<std::string>    keys;
    std::vector<std::string>    values;
    std::vector<std::string>    args;

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
                args.push_back(arg);
                continue;
            }

            string::size_type pos;
            string key, val;
            if ((pos = arg.find('=')) == string::npos) {
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

        for (int i = 0; i < int(keys.size()); ++i)
        {
            if (keys[i] == string(arg_name))
                return true;
        }
        return false;
    }


    /**
     * Returns number of naked (non-flag and non-key-value) commandline parameters
     */
    template <typename T>
    int NumNakedArgs()
    {
        return args.size();
    }


    /**
     * Returns the commandline parameter for a given index (not including flags)
     */
    template <typename T>
    void GetCmdLineArgument(int index, T &val)
    {
        using namespace std;
        if (index < args.size()) {
            istringstream str_stream(args[index]);
            str_stream >> val;
        }
    }

    /**
     * Returns the value specified for a given commandline parameter --<flag>=<value>
     */
    template <typename T>
    void GetCmdLineArgument(const char *arg_name, T &val)
    {
        using namespace std;

        for (int i = 0; i < int(keys.size()); ++i)
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
        return (int) keys.size();
    }
};


/**
 * CPU timer
 */
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

        return float((stop - start) * 1000);
    }

#else

    timespec start;
    timespec stop;

    void Start()
    {
        clock_gettime(CLOCK_MONOTONIC, &start);
    }

    void Stop()
    {
        clock_gettime(CLOCK_MONOTONIC, &stop);
    }

    float ElapsedMillis()
    {
        return ((float(stop.tv_sec - start.tv_sec) * 1000) + (float(stop.tv_nsec - start.tv_nsec) / 1000000L));
    }

#endif
};



//---------------------------------------------------------------------
// MergePath Search
//---------------------------------------------------------------------


/**
 * Computes the begin offsets into A and B for the specific diagonal
 */
template <
    typename AIteratorT,
    typename BIteratorT,
    typename OffsetT,
    typename CoordinateT>
inline void MergePathSearch(
    OffsetT         diagonal,
    AIteratorT      a,
    BIteratorT      b,
    OffsetT         a_len,
    OffsetT         b_len,
    CoordinateT&    path_coordinate)
{
    /// The value type of the input iterator
    typedef typename std::iterator_traits<AIteratorT>::value_type T;

    OffsetT split_min = std::max(diagonal - b_len, 0);
    OffsetT split_max = std::min(diagonal, a_len);

    while (split_min < split_max)
    {
        OffsetT split_pivot = (split_min + split_max) >> 1;
        if (a[split_pivot] <= b[diagonal - split_pivot - 1])
        {
            // Move candidate split range up A, down B
            split_min = split_pivot + 1;
        }
        else
        {
            // Move candidate split range up B, down A
            split_max = split_pivot;
        }
    }

    path_coordinate.x = std::min(split_min, a_len);
    path_coordinate.y = diagonal - split_min;
}


//---------------------------------------------------------------------
// SpMV verification
//---------------------------------------------------------------------

// Compute reference SpMV y = Ax
template <
    typename ValueT,
    typename OffsetT>
void SpmvGold(
    CsrMatrix<ValueT, OffsetT>&     a,
    ValueT*                         vector_x,
    ValueT*                         vector_y_in,
    ValueT*                         vector_y_out,
    ValueT                          alpha,
    ValueT                          beta)
{
    for (OffsetT row = 0; row < a.num_rows; ++row)
    {
        ValueT partial = beta * vector_y_in[row];
        for (
            OffsetT offset = a.row_offsets[row];
            offset < a.row_offsets[row + 1];
            ++offset)
        {
            partial += alpha * a.values[offset] * vector_x[a.column_indices[offset]];
        }
        vector_y_out[row] = partial;
    }
}


/**
 * Compares the equivalence of two arrays
 */
template <typename S, typename T, typename OffsetT>
int CompareResults(T* computed, S* reference, OffsetT len, bool verbose = true)
{
    for (OffsetT i = 0; i < len; i++)
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
template <typename OffsetT>
int CompareResults(float* computed, float* reference, OffsetT len, bool verbose = true)
{
    for (OffsetT i = 0; i < len; i++)
    {
        if (computed[i] != reference[i])
        {
            float difference = std::abs(computed[i]-reference[i]);
            float fraction = difference / std::abs(reference[i]);

            if (fraction > 0.0001)
            {
                if (verbose) std::cout << "INCORRECT: [" << i << "]: "
                    << computed[i] << " != "
                    << reference[i] << " (difference:" << difference << ", fraction: " << fraction << ")";
                return 1;
            }
        }
    }
    return 0;
}


/**
 * Compares the equivalence of two arrays
 */
template <typename OffsetT>
int CompareResults(double* computed, double* reference, OffsetT len, bool verbose = true)
{
    for (OffsetT i = 0; i < len; i++)
    {
        if (computed[i] != reference[i])
        {
            double difference = std::abs(computed[i]-reference[i]);
            double fraction = difference / std::abs(reference[i]);

            if (fraction > 0.0001)
            {
                if (verbose) std::cout << "INCORRECT: [" << i << "]: "
                    << computed[i] << " != "
                    << reference[i] << " (difference:" << difference << ", fraction: " << fraction << ")";
                return 1;
            }
        }
    }
    return 0;
}


//---------------------------------------------------------------------
// CPU SpMV CSR I/O proxy
//---------------------------------------------------------------------

/**
 * OpenMP CPU merge-based SpMV
 */
template <
    typename ValueT,
    typename OffsetT>
void OmpCsrIoProxy(
    int                             num_threads,
    CsrMatrix<ValueT, OffsetT>&     a,
    ValueT*                         vector_x,
    ValueT*                         vector_y_out)
{
    OffsetT nnz_per_thread = (a.num_nonzeros + num_threads - 1) / num_threads;
    OffsetT nr_per_thread = (a.num_rows + num_threads - 1) / num_threads;

    #pragma omp parallel for
    for (int tid = 0; tid < num_threads; tid++)
    {
        // Stream nonzeros

        int start_idx   = nnz_per_thread * tid;
        int end_idx     = std::min(start_idx + nnz_per_thread, a.num_nonzeros);

        ValueT running_total = ValueT(0.0);
        for (int nonzero_idx = start_idx; nonzero_idx < end_idx; ++nonzero_idx)
        {
            running_total += a.values[nonzero_idx] * vector_x[a.column_indices[nonzero_idx]];
        }

        // Stream rows

        start_idx   = nr_per_thread * tid;
        end_idx     = std::min(start_idx + nr_per_thread, a.num_rows);

        for (int row_idx = start_idx; row_idx < end_idx; ++row_idx)
        {
            vector_y_out[row_idx] = ValueT(a.row_offsets[row_idx]) + running_total;
        }
    }

}


/**
 * Run OmpCsrIoProxy
 */
template <
    typename ValueT,
    typename OffsetT>
float TestOmpCsrIoProxy(
    CsrMatrix<ValueT, OffsetT>&     a,
    ValueT*                         vector_x,
    int                             timing_iterations)
{
    ValueT* vector_y_out = new ValueT[a.num_rows];

    if (g_omp_threads == -1)
        g_omp_threads = omp_get_num_procs();

    omp_set_num_threads(g_omp_threads);
    omp_set_dynamic(0);

    if (!g_quiet)
    {
        printf("\tUsing %d threads on %d procs\n", g_omp_threads, omp_get_num_procs());
    }

    // Warmup
    OmpCsrIoProxy(g_omp_threads, a, vector_x, vector_y_out);
    OmpCsrIoProxy(g_omp_threads, a, vector_x, vector_y_out);
    OmpCsrIoProxy(g_omp_threads, a, vector_x, vector_y_out);

    // Timing
    float elapsed_millis = 0.0;
    CpuTimer timer;
    timer.Start();
    for(int it = 0; it < timing_iterations; ++it)
    {
        OmpCsrIoProxy(g_omp_threads, a, vector_x, vector_y_out);
    }
    timer.Stop();
    elapsed_millis += timer.ElapsedMillis();

    delete[] vector_y_out;

    return elapsed_millis / timing_iterations;
}



//---------------------------------------------------------------------
// CPU merge-based SpMV
//---------------------------------------------------------------------

// OpenMP CPU merge-based SpMV
template <
    typename ValueT,
    typename OffsetT>
void OmpMergeCsrmv(
    int                             num_threads,
    CsrMatrix<ValueT, OffsetT>&     a,
    ValueT*                         vector_x,
    ValueT*                         vector_y_out)
{
    OffsetT row_carry_out[256];
    ValueT value_carry_out[256];

    OffsetT                         num_merge_items     = a.num_rows + a.num_nonzeros;
    OffsetT                         items_per_thread    = (num_merge_items + num_threads - 1) / num_threads;
    OffsetT*                        row_end_offsets     = a.row_offsets + 1;
    CountingInputIterator<OffsetT>  nonzero_indices(0);

    #pragma omp parallel for
    for (int tid = 0; tid < num_threads; tid++)
    {
        int start_diagonal  = items_per_thread * tid;
        int end_diagonal    = std::min(start_diagonal + items_per_thread, num_merge_items);

        // Search for starting coordinates
        int2 thread_coord;
        MergePathSearch(start_diagonal, row_end_offsets, nonzero_indices, a.num_rows, a.num_nonzeros, thread_coord);

        ValueT  running_total   = ValueT(0.0);
        OffsetT row_end_offset  = row_end_offsets[thread_coord.x];
        OffsetT nonzero_idx     = nonzero_indices[thread_coord.y];

        // Merge items
        for (int merge_item = start_diagonal; merge_item < end_diagonal; ++merge_item)
        {

            if (nonzero_idx < row_end_offset)
            {
                // Move down (accumulate)
                ValueT nonzero = a.values[nonzero_idx] * vector_x[a.column_indices[nonzero_idx]];
                running_total += nonzero;
                ++thread_coord.y;
                nonzero_idx = nonzero_indices[thread_coord.y];
            }
            else
            {
                // Move right (reset)
                vector_y_out[thread_coord.x] = running_total;
                running_total = ValueT(0.0);
                ++thread_coord.x;
                row_end_offset = row_end_offsets[thread_coord.x];
            }
        }

        // Save carry-outs
        row_carry_out[tid] = thread_coord.x;
        value_carry_out[tid] = running_total;
    }

    // Carry-out fix-up
    for (int tid = 0; tid < num_threads - 1; ++tid)
    {
        if (row_carry_out[tid] < a.num_rows)
            vector_y_out[row_carry_out[tid]] += value_carry_out[tid];
    }
}


/**
 * Run OmpMergeCsrmv
 */
template <
    typename ValueT,
    typename OffsetT>
float TestOmpMergeCsrmv(
    CsrMatrix<ValueT, OffsetT>&     a,
    ValueT*                         vector_x,
    ValueT*                         reference_vector_y_out,
    int                             timing_iterations)
{
    ValueT* vector_y_out = new ValueT[a.num_rows];

    if (g_omp_threads == -1)
        g_omp_threads = omp_get_num_procs();

    omp_set_num_threads(g_omp_threads);
    omp_set_dynamic(0);

    if (!g_quiet)
    {
        printf("\tUsing %d threads on %d procs\n", g_omp_threads, omp_get_num_procs());
    }

    // Warmup
    OmpMergeCsrmv(g_omp_threads, a, vector_x, vector_y_out);
    if (!g_quiet)
    {
        int compare = CompareResults(reference_vector_y_out, vector_y_out, a.num_rows, true);
        printf("\t%s\n", compare ? "FAIL" : "PASS"); fflush(stdout);
    }
    OmpMergeCsrmv(g_omp_threads, a, vector_x, vector_y_out);
    OmpMergeCsrmv(g_omp_threads, a, vector_x, vector_y_out);

    // Timing
    float elapsed_millis = 0.0;
    CpuTimer timer;
    timer.Start();
    for(int it = 0; it < timing_iterations; ++it)
    {
        OmpMergeCsrmv(g_omp_threads, a, vector_x, vector_y_out);
    }
    timer.Stop();
    elapsed_millis += timer.ElapsedMillis();

    delete[] vector_y_out;

    return elapsed_millis / timing_iterations;
}



//---------------------------------------------------------------------
// MKL SpMV
//---------------------------------------------------------------------

/**
 * Run MKL SpMV (specialized for fp32)
 */
template <
    typename OffsetT>
float TestMklCsrmv(
    CsrMatrix<float, OffsetT>&      a,
    float*                          vector_x,
    float*                          reference_vector_y_out,
    int                             timing_iterations)
{
    float* vector_y_out = new float[a.num_rows];

    // Warmup
    mkl_cspblas_scsrgemv("n", &a.num_rows, a.values, a.row_offsets, a.column_indices, vector_x, vector_y_out);
    if (!g_quiet)
    {
        int compare = CompareResults(reference_vector_y_out, vector_y_out, a.num_rows, true);
        printf("\t%s\n", compare ? "FAIL" : "PASS"); fflush(stdout);
    }
    mkl_cspblas_scsrgemv("n", &a.num_rows, a.values, a.row_offsets, a.column_indices, vector_x, vector_y_out);
    mkl_cspblas_scsrgemv("n", &a.num_rows, a.values, a.row_offsets, a.column_indices, vector_x, vector_y_out);

    // Timing
    float elapsed_millis    = 0.0;
    CpuTimer timer;
    timer.Start();
    for(int it = 0; it < timing_iterations; ++it)
    {
        mkl_cspblas_scsrgemv("n", &a.num_rows, a.values, a.row_offsets, a.column_indices, vector_x, vector_y_out);
    }
    timer.Stop();
    elapsed_millis += timer.ElapsedMillis();

    delete[] vector_y_out;

    return elapsed_millis / timing_iterations;
}


/**
 * Run MKL SpMV (specialized for fp64)
 */
template <
    typename OffsetT>
float TestMklCsrmv(
    CsrMatrix<double, OffsetT>&     a,
    double*                         vector_x,
    double*                         reference_vector_y_out,
    int                             timing_iterations)
{
    double* vector_y_out = new double[a.num_rows];

    // Warmup
    mkl_cspblas_dcsrgemv("n", &a.num_rows, a.values, a.row_offsets, a.column_indices, vector_x, vector_y_out);
    if (!g_quiet)
    {
        int compare = CompareResults(reference_vector_y_out, vector_y_out, a.num_rows, true);
        printf("\t%s\n", compare ? "FAIL" : "PASS"); fflush(stdout);
    }
    mkl_cspblas_dcsrgemv("n", &a.num_rows, a.values, a.row_offsets, a.column_indices, vector_x, vector_y_out);
    mkl_cspblas_dcsrgemv("n", &a.num_rows, a.values, a.row_offsets, a.column_indices, vector_x, vector_y_out);

    // Timing
    float elapsed_millis = 0.0;
    CpuTimer timer;
    timer.Start();
    for(int it = 0; it < timing_iterations; ++it)
    {
        mkl_cspblas_dcsrgemv("n", &a.num_rows, a.values, a.row_offsets, a.column_indices, vector_x, vector_y_out);
    }
    timer.Stop();
    elapsed_millis += timer.ElapsedMillis();

    delete[] vector_y_out;

    return elapsed_millis / timing_iterations;
}

//---------------------------------------------------------------------
// Test generation
//---------------------------------------------------------------------

/**
 * Display perf
 */
template <typename ValueT, typename OffsetT>
void DisplayPerf(
    float                           device_giga_bandwidth,
    double                          avg_millis,
    CsrMatrix<ValueT, OffsetT>&     csr_matrix)
{
    double nz_throughput, effective_bandwidth;
    size_t total_bytes = (csr_matrix.num_nonzeros * (sizeof(ValueT) * 2 + sizeof(OffsetT))) +
        (csr_matrix.num_rows) * (sizeof(OffsetT) + sizeof(ValueT));

    nz_throughput       = double(csr_matrix.num_nonzeros) / avg_millis / 1.0e6;
    effective_bandwidth = double(total_bytes) / avg_millis / 1.0e6;

    if (!g_quiet)
        printf("fp%d: %.4f avg ms, %.5f gflops, %.3lf effective GB/s (%.2f%% peak)\n",
            int(sizeof(ValueT) * 8),
            avg_millis,
            2 * nz_throughput,
            effective_bandwidth,
            effective_bandwidth / device_giga_bandwidth * 100);
    else
        printf("%.5f, %.6f, %.3lf, %.2f%%, ",
            avg_millis,
            2 * nz_throughput,
            effective_bandwidth,
            effective_bandwidth / device_giga_bandwidth * 100);

    fflush(stdout);
}


/**
 * Run tests
 */
template <
    typename ValueT,
    typename OffsetT>
void RunTests(
    bool                rcm_relabel,
    ValueT              alpha,
    ValueT              beta,
    const std::string&  mtx_filename,
    int                 grid2d,
    int                 grid3d,
    int                 wheel,
    int                 dense,
    int                 timing_iterations,
    CommandLineArgs&    args)
{
    // Initialize matrix in COO form
    CooMatrix<ValueT, OffsetT> coo_matrix;

    if (!mtx_filename.empty())
    {
        // Parse matrix market file
        printf("%s, ", mtx_filename.c_str()); fflush(stdout);
        coo_matrix.InitMarket(mtx_filename, 1.0, !g_quiet);
    }
    else if (grid2d > 0)
    {
        // Generate 2D lattice
        printf("grid2d_%d, ", grid2d); fflush(stdout);
        coo_matrix.InitGrid2d(grid2d, false);
    }
    else if (grid3d > 0)
    {
        // Generate 3D lattice
        printf("grid3d_%d, ", grid3d); fflush(stdout);
        coo_matrix.InitGrid3d(grid3d, false);
    }
    else if (wheel > 0)
    {
        // Generate wheel graph
        printf("wheel_%d, ", grid2d); fflush(stdout);
        coo_matrix.InitWheel(wheel);
    }
    else if (dense > 0)
    {
        // Generate dense graph
        OffsetT rows = (1<<24) / dense;               // 16M nnz
        printf("dense_%d_x_%d, ", rows, dense); fflush(stdout);
        coo_matrix.InitDense(rows, dense);
    }
    else
    {
        fprintf(stderr, "No graph type specified.\n");
        exit(1);
    }

    CsrMatrix<ValueT, OffsetT> csr_matrix;
    csr_matrix.FromCoo(coo_matrix);

    // Relabel
    if (rcm_relabel)
    {
        if (!g_quiet)
        {
            csr_matrix.Stats().Display();
            printf("\n");
            csr_matrix.DisplayHistogram();
            printf("\n");
            if (g_verbose2)
                csr_matrix.Display();
            printf("\n");
        }

        RcmRelabel(csr_matrix, !g_quiet);

        if (!g_quiet)
        printf("\n");
    }

    // Display matrix info
    csr_matrix.Stats().Display(!g_quiet);
    if (!g_quiet)
    {
        printf("\n");
        csr_matrix.DisplayHistogram();
        printf("\n");
        if (g_verbose2)
            csr_matrix.Display();
        printf("\n");
    }
    fflush(stdout);

    // Adaptive timing iterations: run two billion nonzeros through
    if (timing_iterations == -1)
    {
        timing_iterations = std::max(5, std::min(OffsetT(1e5), (OffsetT(2e9) / csr_matrix.num_nonzeros)));
        if (!g_quiet)
            printf("\t%d timing iterations\n", timing_iterations);
    }

    // Allocate input and output vectors
    ValueT* vector_x        = new ValueT[csr_matrix.num_cols];
    ValueT* vector_y_in     = new ValueT[csr_matrix.num_rows];
    ValueT* vector_y_out    = new ValueT[csr_matrix.num_rows];

    for (int col = 0; col < csr_matrix.num_cols; ++col)
        vector_x[col] = 1.0;

    for (int row = 0; row < csr_matrix.num_rows; ++row)
        vector_y_in[row] = 1.0;

    // Compute reference answer
    SpmvGold(csr_matrix, vector_x, vector_y_in, vector_y_out, alpha, beta);

    float avg_millis;

#ifdef CUB_MKL

    if (!g_quiet) {
        printf("\n\nMKL SpMV: "); fflush(stdout);
    }
    avg_millis = TestMklCsrmv(csr_matrix, vector_x, vector_y_out, timing_iterations);
    DisplayPerf(30, avg_millis, csr_matrix);

#endif

    if (!g_quiet) {
        printf("\n\nCPU CSR I/O Proxy: "); fflush(stdout);
    }
    avg_millis = TestOmpCsrIoProxy(csr_matrix, vector_x, timing_iterations);
    DisplayPerf(30, avg_millis, csr_matrix);

    if (!g_quiet) {
        printf("\n\nOMP SpMV: "); fflush(stdout);
    }
    avg_millis = TestOmpMergeCsrmv(csr_matrix, vector_x, vector_y_out, timing_iterations);
    DisplayPerf(30, avg_millis, csr_matrix);

    if (vector_x)                   delete[] vector_x;
    if (vector_y_in)                delete[] vector_y_in;
    if (vector_y_out)               delete[] vector_y_out;
}



/**
 * Main
 */
int main(int argc, char **argv)
{
    // Initialize command line
    CommandLineArgs args(argc, argv);
    if (args.CheckCmdLineFlag("help"))
    {
        printf(
            "%s "
            "[--device=<device-id>] "
            "[--quiet] "
            "[--v] "
            "[--i=<timing iterations>] "
            "[--fp64] "
            "[--rcm] "
            "[--alpha=<alpha scalar (default: 1.0)>] "
            "[--beta=<beta scalar (default: 0.0)>] "
            "\n\t"
                "--mtx=<matrix market file> "
            "\n\t"
                "--dense=<cols>"
            "\n\t"
                "--grid2d=<width>"
            "\n\t"
                "--grid3d=<width>"
            "\n\t"
                "--wheel=<spokes>"
            "\n", argv[0]);
        exit(0);
    }

    bool                fp64;
    bool                rcm_relabel;
    std::string         mtx_filename;
    int                 grid2d              = -1;
    int                 grid3d              = -1;
    int                 wheel               = -1;
    int                 dense               = -1;
    int                 timing_iterations   = -1;
    float               alpha               = 1.0;
    float               beta                = 0.0;

    g_verbose = args.CheckCmdLineFlag("v");
    g_verbose2 = args.CheckCmdLineFlag("v2");
    g_quiet = args.CheckCmdLineFlag("quiet");
    fp64 = args.CheckCmdLineFlag("fp64");
    rcm_relabel = args.CheckCmdLineFlag("rcm");
    args.GetCmdLineArgument("i", timing_iterations);
    args.GetCmdLineArgument("mtx", mtx_filename);
    args.GetCmdLineArgument("grid2d", grid2d);
    args.GetCmdLineArgument("grid3d", grid3d);
    args.GetCmdLineArgument("wheel", wheel);
    args.GetCmdLineArgument("dense", dense);
    args.GetCmdLineArgument("alpha", alpha);
    args.GetCmdLineArgument("beta", beta);
    args.GetCmdLineArgument("threads", g_omp_threads);

    // Run test(s)
    if (fp64)
    {
        RunTests<double, int>(rcm_relabel, alpha, beta, mtx_filename, grid2d, grid3d, wheel, dense, timing_iterations, args);
    }
    else
    {
        RunTests<float, int>(rcm_relabel, alpha, beta, mtx_filename, grid2d, grid3d, wheel, dense, timing_iterations, args);
    }

    printf("\n");

    return 0;
}
