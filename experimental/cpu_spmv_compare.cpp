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
 * icpc cpu_spmv_compare.cpp -mkl -openmp -DCUB_MKL -O3 -o spmv_omp.out -lrt -fno-alias -xHost
 * export KMP_AFFINITY=granularity=core,scatter
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
int                     g_omp_threads   = -1;           // Number of openMP threads
int                     g_omp_oversub   = 1;            // Factor of over-subscription
int                     g_ht_threshold  = 1*1024*1024;  // Hyperthread threshold (num nonzeros)


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
    double start;
    double stop;

    void Start()
    {
        start = omp_get_wtime();
    }

    void Stop()
    {
        stop = omp_get_wtime();
    }

    float ElapsedMillis()
    {
        return (stop - start) * 1000;
    }

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

    #pragma omp parallel for schedule(static) num_threads(num_threads)
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
    ValueT* vector_y_out = (ValueT*) mkl_malloc(sizeof(ValueT) * a.num_rows, 4096);

    if (g_omp_threads == -1)
        g_omp_threads = omp_get_num_procs(); 

    if (!g_quiet)
        printf("\tUsing %d threads on %d procs\n", g_omp_threads, omp_get_num_procs());

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

    mkl_free(vector_y_out);

    return elapsed_millis / timing_iterations;
}



//---------------------------------------------------------------------
// CPU merge-based SpMV
//---------------------------------------------------------------------

/*
// Doc version
template <typename CsrMatrix>
// OpenMP CPU merge-based SpMV (y = Ax)
void OmpMergeCsrmv(int num_threads, CsrMatrix& A, double* x, double* y)
{
    int* row_end_offsets = A.row_offsets + 1;           // Merge list A (row end-offsets)
    CountingInputIterator<int> nz_indices(0);           // Merge list B (NZ indices)

    int num_merge_items = A.num_rows + A.num_nonzeros;  // Merge path length
    int items_per_thread = (num_merge_items + num_threads - 1) / num_threads;

    int row_carry_out[num_threads];
    double value_carry_out[num_threads];

    #pragma omp parallel for schedule(static) num_threads(num_threads)
    for (int tid = 0; tid < num_threads; tid++)
    {
        // Find starting and ending MergePath coordinates (row-idx, nonzero-idx) for each thread
        int2 thread_coord, thread_coord_end;
        int merge_item      = std::min(items_per_thread * tid, num_merge_items);
        int merge_item_end  = std::min(merge_item + items_per_thread, num_merge_items);
        MergePathSearch(merge_item, row_end_offsets, nz_indices, A.num_rows, A.num_nonzeros, thread_coord);
        MergePathSearch(merge_item_end, row_end_offsets, nz_indices, A.num_rows, A.num_nonzeros, thread_coord_end);

        // Consume whole rows
        double running_total = 0.0;
        for (; thread_coord.x < thread_coord_end.x; ++thread_coord.x)
        {
            for (; thread_coord.y < row_end_offsets[thread_coord.x]; ++thread_coord.y)
                running_total += A.values[thread_coord.y] * x[A.column_indices[thread_coord.y]];

            y[thread_coord.x] = running_total;
            running_total = 0.0;
        }

        // Consume partial portion of thread's last row
        for (; thread_coord.y < thread_coord_end.y; ++thread_coord.y)
            running_total += A.values[thread_coord.y] * x[A.column_indices[thread_coord.y]];

        // Save carry-outs
        row_carry_out[tid] = thread_coord_end.x;
        value_carry_out[tid] = running_total;
    }

    // Carry-out fix-up (rows spanning multiple threads)
    for (int tid = 0; tid < num_threads - 1; ++tid)
        if (row_carry_out[tid] < A.num_rows)
            y[row_carry_out[tid]] += value_carry_out[tid];
}
*/


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
    OffsetT     row_carry_out[1024];
    ValueT      value_carry_out[1024];

    int         slices = num_threads * g_omp_oversub;

    #pragma omp parallel for schedule(static) num_threads(num_threads)
    for (int tid = 0; tid < slices; tid++)
    {
        OffsetT                         num_merge_items     = a.num_rows + a.num_nonzeros;
        OffsetT                         items_per_thread    = (num_merge_items + num_threads - 1) / num_threads;
        OffsetT*                        row_end_offsets     = a.row_offsets + 1;
        CountingInputIterator<OffsetT>  nonzero_indices(0);

        int start_diagonal      = std::min(items_per_thread * tid, num_merge_items);
        int end_diagonal        = std::min(start_diagonal + items_per_thread, num_merge_items);

        int2 thread_coord;
        int2 thread_coord_end;

        MergePathSearch(start_diagonal, row_end_offsets, nonzero_indices, a.num_rows, a.num_nonzeros, thread_coord);
        MergePathSearch(end_diagonal, row_end_offsets, nonzero_indices, a.num_rows, a.num_nonzeros, thread_coord_end);

        ValueT running_total = ValueT(0.0);

        // Consume whole rows
        for (; thread_coord.x < thread_coord_end.x; ++thread_coord.x)
        {
            for (; thread_coord.y + 4 <= row_end_offsets[thread_coord.x]; thread_coord.y += 4)
            {
                ValueT v0 = a.values[thread_coord.y + 0] * vector_x[a.column_indices[thread_coord.y + 0]];
                ValueT v1 = a.values[thread_coord.y + 1] * vector_x[a.column_indices[thread_coord.y + 1]];
                ValueT v2 = a.values[thread_coord.y + 2] * vector_x[a.column_indices[thread_coord.y + 2]];
                ValueT v3 = a.values[thread_coord.y + 3] * vector_x[a.column_indices[thread_coord.y + 3]];

                running_total += v0 + v1 + v2 + v3;
            }

            for (; thread_coord.y < row_end_offsets[thread_coord.x]; ++thread_coord.y)
            {
                running_total += a.values[thread_coord.y] * vector_x[a.column_indices[thread_coord.y]];
            }

            vector_y_out[thread_coord.x] = running_total;
            running_total = ValueT(0.0);
        }

        // Consume partial portion of thread's last row
        for (; thread_coord.y < thread_coord_end.y; ++thread_coord.y)
        {
            running_total += a.values[thread_coord.y] * vector_x[a.column_indices[thread_coord.y]];
        }

        // Save carry-outs
        row_carry_out[tid] = thread_coord_end.x;
        value_carry_out[tid] = running_total;
    }

    // Carry-out fix-up
    for (int tid = 0; tid < slices - 1; ++tid)
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
    ValueT*                         vector_y_out,
    int                             timing_iterations)
{
    if (g_omp_threads == -1)
        g_omp_threads = omp_get_num_procs(); 

    if (!g_quiet)
        printf("\tUsing %d threads on %d procs\n", g_omp_threads, omp_get_num_procs());

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

    return elapsed_millis / timing_iterations;
}


//---------------------------------------------------------------------
// Autotuned MKL SpMV
//---------------------------------------------------------------------

int g_expected_calls = 1000000;

/**
 * Run MKL SpMV (specialized for fp32)
 */
template <
    typename OffsetT>
float TestMklCsrmvTuned(
    CsrMatrix<float, OffsetT>&      a,
    float*                          vector_x,
    float*                          reference_vector_y_out,
    float*                          vector_y_out,
    int                             timing_iterations)
{
    float alpha = 1.0;
    float beta = 0.0;

    sparse_status_t status;

    CpuTimer timer;
    timer.Start();

    // Create CSR handle
    sparse_matrix_t mkl_a;
    status = mkl_sparse_s_create_csr(
        &mkl_a,
        SPARSE_INDEX_BASE_ZERO,
        a.num_rows,
        a.num_cols,
        a.row_offsets,
        a.row_offsets + 1,
        a.column_indices,
        a.values);
    if (status != SPARSE_STATUS_SUCCESS) {
        printf("Could not create handle: %d\n", status); exit(1);
    }

    // Set MV hint
    matrix_descr mkl_a_desc;
    mkl_a_desc.type = SPARSE_MATRIX_TYPE_GENERAL;
    status = mkl_sparse_set_mv_hint(mkl_a, SPARSE_OPERATION_NON_TRANSPOSE, mkl_a_desc, g_expected_calls);
    if (status != SPARSE_STATUS_SUCCESS) {
        printf("Could not set hint: %d\n", status); exit(1);
    }

    // Optimize
    status = mkl_sparse_optimize(mkl_a);
    if (status != SPARSE_STATUS_SUCCESS) {
        printf("Could not optimize: %d\n", status); exit(1);
    }

    timer.Stop();
    float elapsed_millis = timer.ElapsedMillis();
    printf("mkl tune ms, %.5f, ", elapsed_millis);

    // Warmup
    status = mkl_sparse_s_mv(SPARSE_OPERATION_NON_TRANSPOSE, alpha, mkl_a, mkl_a_desc, vector_x, beta, vector_y_out);
    if (status != SPARSE_STATUS_SUCCESS) {
        printf("SpMV failed: %d\n", status); exit(1);
    }
    if (!g_quiet)
    {
        int compare = CompareResults(reference_vector_y_out, vector_y_out, a.num_rows, true);
        printf("\t%s\n", compare ? "FAIL" : "PASS"); fflush(stdout);
    }

    // Timing
    timer.Start();
    for(int it = 0; it < timing_iterations; ++it)
    {
        status = mkl_sparse_s_mv(SPARSE_OPERATION_NON_TRANSPOSE, alpha, mkl_a, mkl_a_desc, vector_x, beta, vector_y_out);
    }
    timer.Stop();
    elapsed_millis = timer.ElapsedMillis();

    // Cleanup
    status = mkl_sparse_destroy(mkl_a);
    if (status != SPARSE_STATUS_SUCCESS) {
        printf("Cleanup failed: %d\n", status); exit(1);
    }

    return elapsed_millis / timing_iterations;
}

/**
 * Run MKL SpMV (specialized for fp64)
 */
template <
    typename OffsetT>
float TestMklCsrmvTuned(
    CsrMatrix<double, OffsetT>&      a,
    double*                          vector_x,
    double*                          reference_vector_y_out,
    double*                          vector_y_out,
    int                             timing_iterations)
{
    double alpha = 1.0;
    double beta = 0.0;

    sparse_status_t status;

    CpuTimer timer;
    timer.Start();

    // Create CSR handle
    matrix_descr mkl_a_desc;
    mkl_a_desc.type = SPARSE_MATRIX_TYPE_GENERAL;
    sparse_matrix_t mkl_a;
    status = mkl_sparse_d_create_csr(
        &mkl_a,
        SPARSE_INDEX_BASE_ZERO,
        a.num_rows,
        a.num_cols,
        a.row_offsets,
        a.row_offsets + 1,
        a.column_indices,
        a.values);
    if (status != SPARSE_STATUS_SUCCESS) {
        printf("Could not create handle: %d\n", status); exit(1);
    }

    // Set MV hint
    status = mkl_sparse_set_mv_hint(mkl_a, SPARSE_OPERATION_NON_TRANSPOSE, mkl_a_desc, g_expected_calls);
    if (status != SPARSE_STATUS_SUCCESS) {
        printf("Could not set hint: %d\n", status); exit(1);
    }

    // Optimize
    status = mkl_sparse_optimize(mkl_a);
    if (status != SPARSE_STATUS_SUCCESS) {
        printf("Could not optimize: %d\n", status); exit(1);
    }

    timer.Stop();
    float elapsed_millis = timer.ElapsedMillis();
    printf("mkl tune ms, %.5f, ", elapsed_millis);

    // Warmup
    status = mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, alpha, mkl_a, mkl_a_desc, vector_x, beta, vector_y_out);
    if (status != SPARSE_STATUS_SUCCESS) {
        printf("SpMV failed: %d\n", status); exit(1);
    }
    if (!g_quiet)
    {
        int compare = CompareResults(reference_vector_y_out, vector_y_out, a.num_rows, true);
        printf("\t%s\n", compare ? "FAIL" : "PASS"); fflush(stdout);
    }

    // Timing
    timer.Start();
    for(int it = 0; it < timing_iterations; ++it)
    {
        status = mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, alpha, mkl_a, mkl_a_desc, vector_x, beta, vector_y_out);
    }
    timer.Stop();
    elapsed_millis = timer.ElapsedMillis();

    // Cleanup
    status = mkl_sparse_destroy(mkl_a);
    if (status != SPARSE_STATUS_SUCCESS) {
        printf("Cleanup failed: %d\n", status); exit(1);
    }

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
    float*                          vector_y_out,
    int                             timing_iterations)
{
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
    double*                         vector_y_out,
    int                             timing_iterations)
{
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
    double                          avg_millis,
    CsrMatrix<ValueT, OffsetT>&     csr_matrix)
{
    double nz_throughput, effective_bandwidth;
    size_t total_bytes = (csr_matrix.num_nonzeros * (sizeof(ValueT) * 2 + sizeof(OffsetT))) +
        (csr_matrix.num_rows) * (sizeof(OffsetT) + sizeof(ValueT));

    nz_throughput       = double(csr_matrix.num_nonzeros) / avg_millis / 1.0e6;
    effective_bandwidth = double(total_bytes) / avg_millis / 1.0e6;

    if (!g_quiet)
        printf("fp%d: %.4f avg ms, %.5f gflops, %.3lf effective GB/s\n",
            int(sizeof(ValueT) * 8),
            avg_millis,
            2 * nz_throughput,
            effective_bandwidth);
    else
        printf("%.5f, %.6f, %.3lf, ",
            avg_millis,
            2 * nz_throughput,
            effective_bandwidth);

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

    // Adaptive timing iterations: run 16 billion nonzeros through
    if (timing_iterations == -1)
    {
        timing_iterations = std::min(50000ull, std::max(100ull, ((16ull << 30) / csr_matrix.num_nonzeros)));
        if (!g_quiet)
            printf("\t%d timing iterations\n", timing_iterations);
    }

    // Allocate input and output vectors
    ValueT* vector_x                = (ValueT*) mkl_malloc(sizeof(ValueT) * csr_matrix.num_cols, 4096);
    ValueT* vector_y_in             = (ValueT*) mkl_malloc(sizeof(ValueT) * csr_matrix.num_rows, 4096);
    ValueT* reference_vector_y_out  = (ValueT*) mkl_malloc(sizeof(ValueT) * csr_matrix.num_rows, 4096);
    ValueT* vector_y_out            = (ValueT*) mkl_malloc(sizeof(ValueT) * csr_matrix.num_rows, 4096);

    for (int col = 0; col < csr_matrix.num_cols; ++col)
        vector_x[col] = 1.0;

    for (int row = 0; row < csr_matrix.num_rows; ++row)
        vector_y_in[row] = 1.0;

    // Compute reference answer
    SpmvGold(csr_matrix, vector_x, vector_y_in, reference_vector_y_out, alpha, beta);

    float avg_millis;

    // MKL SpMV Tuned
    if (!g_quiet) {
        printf("\n\nTuned MKL SpMV: "); fflush(stdout);
    }
    avg_millis = TestMklCsrmvTuned(csr_matrix, vector_x, reference_vector_y_out, vector_y_out, timing_iterations);
    DisplayPerf(avg_millis, csr_matrix);

    // MKL SpMV
    if (!g_quiet) {
        printf("\n\nMKL SpMV: "); fflush(stdout);
    }
    avg_millis = TestMklCsrmv(csr_matrix, vector_x, reference_vector_y_out, vector_y_out, timing_iterations);
    DisplayPerf(avg_millis, csr_matrix);

    //  IO Proxy
    if (!g_quiet) {
        printf("\n\nCPU CSR I/O Proxy: "); fflush(stdout);
    }
    avg_millis = TestOmpCsrIoProxy(csr_matrix, vector_x, timing_iterations);
    DisplayPerf(avg_millis, csr_matrix);

    // Merge SpMV
    if (!g_quiet) {
        printf("\n\nOMP SpMV: "); fflush(stdout);
    }
    avg_millis = TestOmpMergeCsrmv(csr_matrix, vector_x, reference_vector_y_out, vector_y_out, timing_iterations);
    DisplayPerf(avg_millis, csr_matrix);

    if (vector_x)                   mkl_free(vector_x);
    if (vector_y_in)                mkl_free(vector_y_in);
    if (reference_vector_y_out)     mkl_free(reference_vector_y_out);
    if (vector_y_out)               mkl_free(vector_y_out);
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
    args.GetCmdLineArgument("oversub", g_omp_oversub);

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
