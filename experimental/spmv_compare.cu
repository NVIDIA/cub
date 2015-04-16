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

//---------------------------------------------------------------------
// SpMV comparison tool
//---------------------------------------------------------------------

#include <stdio.h>
#include <map>
#include <vector>
#include <algorithm>
#include <cstdio>
#include <fstream>

#ifdef CUB_MKL
    #include <mkl.h>
#endif

#include <omp.h>

#include <cusparse.h>

#include "sparse_matrix.h"

// Ensure printing of CUDA runtime errors to console
#define CUB_STDERR

#include <cub/device/device_spmv.cuh>
#include <cub/util_allocator.cuh>
#include <cub/iterator/tex_ref_input_iterator.cuh>
#include <test/test_util.h>

using namespace cub;


//---------------------------------------------------------------------
// Globals, constants, and type declarations
//---------------------------------------------------------------------

bool                    g_quiet     = false;        // Whether to display stats in CSV format
bool                    g_verbose   = false;        // Whether to display output to console
bool                    g_verbose2  = false;        // Whether to display input to console
bool                    g_gpu       = false;        // Whether to time the CPU impls
bool                    g_cpu       = false;        // Whether to time the GPU impls
CachingDeviceAllocator  g_allocator(true);          // Caching allocator for device memory
int                     g_omp_threads = -1;         // Number of openMP threads


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


//---------------------------------------------------------------------
// GPU I/O proxy
//---------------------------------------------------------------------

/**
 * Read every matrix nonzero value, read every corresponding vector value
 */
template <
    int         BLOCK_THREADS,
    int         ITEMS_PER_THREAD,
    typename    ValueT,
    typename    OffsetT,
    typename    VectorItr>
__global__ void NonZeroIoKernel(
    SpmvParams<ValueT, OffsetT> params,
    VectorItr                   d_vector_x)
{
    OffsetT block_offset = blockIdx.x * BLOCK_THREADS * ITEMS_PER_THREAD;

    ValueT nonzero = 0.0;

    #pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
    {
        OffsetT nonzero_idx = block_offset + (ITEM * BLOCK_THREADS) + threadIdx.x;
        nonzero_idx = CUB_MIN(nonzero_idx, params.num_nonzeros - 1);

        OffsetT     column_idx      = ThreadLoad<LOAD_LDG>(params.d_column_indices + nonzero_idx);
        ValueT      value           = ThreadLoad<LOAD_LDG>(params.d_values + nonzero_idx);

#if (CUB_PTX_ARCH >= 350)
        ValueT      vector_value    = ThreadLoad<LOAD_LDG>(params.d_vector_x + column_idx);
#else
        ValueT      vector_value    = d_vector_x[column_idx];
#endif

        nonzero                     += vector_value * value;
    }

    if (block_offset < params.num_rows)
    {
        #pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
        {
            OffsetT row_idx = block_offset + (ITEM * BLOCK_THREADS) + threadIdx.x;
            row_idx = CUB_MIN(row_idx, params.num_rows - 1);

#if (CUB_PTX_ARCH >= 350)
            OffsetT nonzero_idx = ThreadLoad<LOAD_LDG>(params.d_row_end_offsets + row_idx);
#else
            OffsetT nonzero_idx = params.d_row_end_offsets[row_idx];
#endif

            if (nonzero_idx > 0)
                params.d_vector_y[row_idx] = nonzero;
        }
    }
}


/**
 * Run GPU I/O proxy
 */
template <
    typename ValueT,
    typename OffsetT>
float TestGpuCsrIoProxy(
    SpmvParams<ValueT, OffsetT>&    params,
    int                             timing_iterations)
{
    enum {
        BLOCK_THREADS       = 128,
        ITEMS_PER_THREAD    = 7,
        TILE_SIZE           = BLOCK_THREADS * ITEMS_PER_THREAD,
    };

    unsigned int nonzero_blocks = (params.num_nonzeros + TILE_SIZE - 1) / TILE_SIZE;
    unsigned int row_blocks = (params.num_rows + TILE_SIZE - 1) / TILE_SIZE;
    unsigned int blocks = std::max(nonzero_blocks, row_blocks);

    TexRefInputIterator<ValueT, 1234, int> x_itr;
    CubDebugExit(x_itr.BindTexture(params.d_vector_x));

    // Warmup
    NonZeroIoKernel<BLOCK_THREADS, ITEMS_PER_THREAD><<<blocks, BLOCK_THREADS>>>(
        params,
        x_itr);

    // Check for failures
    CubDebugExit(cudaPeekAtLastError());
    CubDebugExit(SyncStream(0));

    // Timing
    GpuTimer timer;
    float elapsed_millis = 0.0;
    timer.Start();
    for (int it = 0; it < timing_iterations; ++it)
    {
        NonZeroIoKernel<BLOCK_THREADS, ITEMS_PER_THREAD><<<blocks, BLOCK_THREADS>>>(
            params,
            x_itr);
    }
    timer.Stop();
    elapsed_millis += timer.ElapsedMillis();

    CubDebugExit(x_itr.UnbindTexture());

    return elapsed_millis / timing_iterations;
}

//---------------------------------------------------------------------
// CPU merge-based SpMV
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
 * Run OmpMergeCsrmv
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
        g_omp_threads = omp_get_num_procs() * 2;

    omp_set_num_threads(g_omp_threads);
    omp_set_dynamic(0);

    if (!g_quiet)
    {
        printf("\tUsing %d threads on %d procs\n", g_omp_threads, omp_get_num_procs());
    }

    // Warmup
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
        ValueT  nonzero         = a.values[nonzero_idx] * vector_x[a.column_indices[nonzero_idx]];

        // Merge items
        for (int merge_item = start_diagonal; merge_item < end_diagonal; ++merge_item)
        {
            if (nonzero_idx < row_end_offset)
            {
                // Move down (accumulate)
                running_total += nonzero;
                ++thread_coord.y;
                nonzero_idx = nonzero_indices[thread_coord.y];
                nonzero = a.values[nonzero_idx] * vector_x[a.column_indices[nonzero_idx]];
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
    for (int tid = 0; tid < num_threads; ++tid)
    {
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
        g_omp_threads = omp_get_num_procs() * 2;

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
// cuSparse SpMV
//---------------------------------------------------------------------

/**
 * Run cuSparse SpMV (specialized for fp32)
 */
template <
    typename OffsetT>
float TestCusparseCsrmv(
    float*                          vector_y_in,
    float*                          reference_vector_y_out,
    SpmvParams<float, OffsetT>&     params,
    int                             timing_iterations,
    cusparseHandle_t                cusparse)
{
    cusparseMatDescr_t desc;
    cusparseCreateMatDescr(&desc);

    // Reset input/output vector y
    CubDebugExit(cudaMemcpy(params.d_vector_y, vector_y_in, sizeof(float) * params.num_rows, cudaMemcpyHostToDevice));

    // Warmup
    AssertEquals(CUSPARSE_STATUS_SUCCESS, cusparseScsrmv(
        cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
        params.num_rows, params.num_cols, params.num_nonzeros, &params.alpha, desc,
        params.d_values, params.d_row_end_offsets, params.d_column_indices,
        params.d_vector_x, &params.beta, params.d_vector_y));

    if (!g_quiet)
    {
        int compare = CompareDeviceResults(reference_vector_y_out, params.d_vector_y, params.num_rows, true, g_verbose);
        printf("\t%s\n", compare ? "FAIL" : "PASS"); fflush(stdout);
    }

    // Timing
    float elapsed_millis    = 0.0;
    GpuTimer timer;

    timer.Start();
    for(int it = 0; it < timing_iterations; ++it)
    {
        AssertEquals(CUSPARSE_STATUS_SUCCESS, cusparseScsrmv(
            cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
            params.num_rows, params.num_cols, params.num_nonzeros, &params.alpha, desc,
            params.d_values, params.d_row_end_offsets, params.d_column_indices,
            params.d_vector_x, &params.beta, params.d_vector_y));
    }
    timer.Stop();
    elapsed_millis += timer.ElapsedMillis();

    cusparseDestroyMatDescr(desc);
    return elapsed_millis / timing_iterations;
}


/**
 * Run cuSparse SpMV (specialized for fp64)
 */
template <
    typename OffsetT>
float TestCusparseCsrmv(
    double*                         vector_y_in,
    double*                         reference_vector_y_out,
    SpmvParams<double, OffsetT>&    params,
    int                             timing_iterations,
    cusparseHandle_t                cusparse)
{
    cusparseMatDescr_t desc;
    cusparseCreateMatDescr(&desc);

    // Reset input/output vector y
    CubDebugExit(cudaMemcpy(params.d_vector_y, vector_y_in, sizeof(float) * params.num_rows, cudaMemcpyHostToDevice));

    // Warmup
    AssertEquals(CUSPARSE_STATUS_SUCCESS, cusparseDcsrmv(
        cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
        params.num_rows, params.num_cols, params.num_nonzeros, &params.alpha, desc,
        params.d_values, params.d_row_end_offsets, params.d_column_indices,
        params.d_vector_x, &params.beta, params.d_vector_y));

    if (!g_quiet)
    {
        int compare = CompareDeviceResults(reference_vector_y_out, params.d_vector_y, params.num_rows, true, g_verbose);
        printf("\t%s\n", compare ? "FAIL" : "PASS"); fflush(stdout);
    }

    // Timing
    float elapsed_millis = 0.0;
    GpuTimer timer;
    timer.Start();
    for(int it = 0; it < timing_iterations; ++it)
    {
        AssertEquals(CUSPARSE_STATUS_SUCCESS, cusparseDcsrmv(
            cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
            params.num_rows, params.num_cols, params.num_nonzeros, &params.alpha, desc,
            params.d_values, params.d_row_end_offsets, params.d_column_indices,
            params.d_vector_x, &params.beta, params.d_vector_y));

    }
    timer.Stop();
    elapsed_millis += timer.ElapsedMillis();

    cusparseDestroyMatDescr(desc);
    return elapsed_millis / timing_iterations;
}

//---------------------------------------------------------------------
// GPU Merge-based SpMV
//---------------------------------------------------------------------

/**
 * Run CUB SpMV
 */
template <
    typename ValueT,
    typename OffsetT>
float TestGpuMergeCsrmv(
    ValueT*                         vector_y_in,
    ValueT*                         reference_vector_y_out,
    SpmvParams<ValueT, OffsetT>&    params,
    int                             timing_iterations)
{
    // Allocate temporary storage
    size_t temp_storage_bytes = 0;
    void *d_temp_storage = NULL;

    // Get amount of temporary storage needed
    CubDebugExit(DeviceSpmv::CsrMV(
        d_temp_storage, temp_storage_bytes,
        params.d_values, params.d_row_end_offsets, params.d_column_indices,
        params.d_vector_x, params.d_vector_y,
        params.num_rows, params.num_cols, params.num_nonzeros, params.alpha, params.beta,
        (cudaStream_t) 0, false));

    // Allocate
    CubDebugExit(g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes));

    // Reset input/output vector y
    CubDebugExit(cudaMemcpy(params.d_vector_y, vector_y_in, sizeof(ValueT) * params.num_rows, cudaMemcpyHostToDevice));

    // Warmup
    CubDebugExit(DeviceSpmv::CsrMV(
        d_temp_storage, temp_storage_bytes,
        params.d_values, params.d_row_end_offsets, params.d_column_indices,
        params.d_vector_x, params.d_vector_y,
        params.num_rows, params.num_cols, params.num_nonzeros, params.alpha, params.beta,
        (cudaStream_t) 0, !g_quiet));

    if (!g_quiet)
    {
        int compare = CompareDeviceResults(reference_vector_y_out, params.d_vector_y, params.num_rows, true, g_verbose);
        printf("\t%s\n", compare ? "FAIL" : "PASS"); fflush(stdout);
    }

    // Timing
    GpuTimer timer;
    float elapsed_millis = 0.0;

    timer.Start();
    for(int it = 0; it < timing_iterations; ++it)
    {
        CubDebugExit(DeviceSpmv::CsrMV(
            d_temp_storage, temp_storage_bytes,
            params.d_values, params.d_row_end_offsets, params.d_column_indices,
            params.d_vector_x, params.d_vector_y,
            params.num_rows, params.num_cols, params.num_nonzeros, params.alpha, params.beta,
            (cudaStream_t) 0, false));
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
            sizeof(ValueT) * 8,
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
    CommandLineArgs&    args,
    cusparseHandle_t    cusparse)
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
        timing_iterations = std::max(5, std::min(OffsetT(1e4), (OffsetT(2e9) / csr_matrix.num_nonzeros)));
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

    if (g_cpu)
    {
        if (!g_quiet) {
            printf("\n\nCPU CSR I/O Proxy: "); fflush(stdout);
        }
        avg_millis = TestOmpCsrIoProxy(csr_matrix, vector_x, timing_iterations);
        DisplayPerf(-1, avg_millis, csr_matrix);

    #ifdef CUB_MKL

        if (!g_quiet) {
            printf("\n\nMKL SpMV: "); fflush(stdout);
        }
        avg_millis = TestMklCsrmv(csr_matrix, vector_x, vector_y_out, timing_iterations);
        DisplayPerf(-1, avg_millis, csr_matrix);

    #endif

        if (!g_quiet) {
            printf("\n\nOMP SpMV: "); fflush(stdout);
        }
        avg_millis = TestOmpMergeCsrmv(csr_matrix, vector_x, vector_y_out, timing_iterations);
        DisplayPerf(-1, avg_millis, csr_matrix);
    }

    if (g_gpu)
    {
        if (g_quiet) {
            printf("%s, %s, ", args.deviceProp.name, (sizeof(ValueT) > 4) ? "fp64" : "fp32"); fflush(stdout);
        }

        // Get GPU device bandwidth (GB/s)
        float device_giga_bandwidth = args.device_giga_bandwidth;

        if (((size_t(csr_matrix.num_rows) * (sizeof(OffsetT) + (sizeof(ValueT) * 2))) +
            (size_t(csr_matrix.num_nonzeros) * (sizeof(OffsetT) + sizeof(ValueT)))) > args.device_free_physmem)
        {
            // Won't fit
            printf("Too big\n"); fflush(stdout);
        }
        else
        {
            // Will fit

            // Allocate and initialize GPU problem
            SpmvParams<ValueT, OffsetT> params;

            g_allocator.DeviceAllocate((void **) &params.d_values,          sizeof(ValueT) * csr_matrix.num_nonzeros);
            g_allocator.DeviceAllocate((void **) &params.d_row_end_offsets, sizeof(OffsetT) * (csr_matrix.num_rows + 1));
            g_allocator.DeviceAllocate((void **) &params.d_column_indices,  sizeof(OffsetT) * csr_matrix.num_nonzeros);
            g_allocator.DeviceAllocate((void **) &params.d_vector_x,        sizeof(ValueT) * csr_matrix.num_cols);
            g_allocator.DeviceAllocate((void **) &params.d_vector_y,        sizeof(ValueT) * csr_matrix.num_rows);
            params.num_rows = csr_matrix.num_rows;
            params.num_cols = csr_matrix.num_cols;
            params.num_nonzeros = csr_matrix.num_nonzeros;
            params.alpha = alpha;
            params.beta = beta;

            CubDebugExit(cudaMemcpy(params.d_values,            csr_matrix.values,          sizeof(ValueT) * csr_matrix.num_nonzeros, cudaMemcpyHostToDevice));
            CubDebugExit(cudaMemcpy(params.d_row_end_offsets,   csr_matrix.row_offsets,     sizeof(OffsetT) * (csr_matrix.num_rows + 1), cudaMemcpyHostToDevice));
            CubDebugExit(cudaMemcpy(params.d_column_indices,    csr_matrix.column_indices,  sizeof(OffsetT) * csr_matrix.num_nonzeros, cudaMemcpyHostToDevice));
            CubDebugExit(cudaMemcpy(params.d_vector_x,          vector_x,                   sizeof(ValueT) * csr_matrix.num_cols, cudaMemcpyHostToDevice));

            if (!g_quiet) {
                printf("\n\nGPU CSR I/O Prox: "); fflush(stdout);
            }
            avg_millis = TestGpuCsrIoProxy(params, timing_iterations);
            DisplayPerf(device_giga_bandwidth, avg_millis, csr_matrix);

            if (!g_quiet) {
                printf("\n\nCusparse: "); fflush(stdout);
            }
            avg_millis = TestCusparseCsrmv(vector_y_in, vector_y_out, params, timing_iterations, cusparse);
            DisplayPerf(device_giga_bandwidth, avg_millis, csr_matrix);

            if (!g_quiet) {
                printf("\n\nCUB: "); fflush(stdout);
            }
            avg_millis = TestGpuMergeCsrmv(vector_y_in, vector_y_out, params, timing_iterations);
            DisplayPerf(device_giga_bandwidth, avg_millis, csr_matrix);

            // Cleanup
            if (params.d_values)            g_allocator.DeviceFree(params.d_values);
            if (params.d_row_end_offsets)   g_allocator.DeviceFree(params.d_row_end_offsets);
            if (params.d_column_indices)    g_allocator.DeviceFree(params.d_column_indices);
            if (params.d_vector_x)          g_allocator.DeviceFree(params.d_vector_x);
            if (params.d_vector_y)          g_allocator.DeviceFree(params.d_vector_y);
        }
    }

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
            "[--cpu] "
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
    g_gpu = args.CheckCmdLineFlag("gpu");
    g_cpu = args.CheckCmdLineFlag("cpu");
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

    // Initialize device
    CubDebugExit(args.DeviceInit());

    // Initalize cuSparse
    cusparseHandle_t cusparse;
    AssertEquals(CUSPARSE_STATUS_SUCCESS, cusparseCreate(&cusparse));

    // Run test(s)
    if (fp64)
    {
        RunTests<double, int>(rcm_relabel, alpha, beta, mtx_filename, grid2d, grid3d, wheel, dense, timing_iterations, args, cusparse);
    }
    else
    {
        RunTests<float, int>(rcm_relabel, alpha, beta, mtx_filename, grid2d, grid3d, wheel, dense, timing_iterations, args, cusparse);
    }

    CubDebugExit(cudaDeviceSynchronize());
    printf("\n");

    return 0;
}
