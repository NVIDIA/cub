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
bool                    g_cpu       = false;        // Whether to time the sequential CPU impl
CachingDeviceAllocator  g_allocator(true);          // Caching allocator for device memory


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
    if ((alpha == 1.0) && (beta == 0.0))
    {
        for (OffsetT row = 0; row < a.num_rows; ++row)
        {
            ValueT partial = 0.0;
            for (
                OffsetT offset = a.row_offsets[row];
                offset < a.row_offsets[row + 1];
                ++offset)
            {
                 partial += a.values[offset] * vector_x[a.column_indices[offset]];
            }
            vector_y_out[row] = partial;
        }
    }
    else
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

}


//---------------------------------------------------------------------
// SpMV I/O proxy
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
__global__ void NonZeroIO(
    int                 num_nonzeros,
    ValueT*             d_values,
    OffsetT*            d_column_indices,
    VectorItr           d_vector_x)
{
    __shared__ volatile ValueT no_elide[1];

    OffsetT block_offset = blockIdx.x * BLOCK_THREADS * ITEMS_PER_THREAD;

    ValueT nonzero = 0.0;

    #pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
    {
        OffsetT item_idx = block_offset + (ITEM * BLOCK_THREADS) + threadIdx.x;
        item_idx = CUB_MIN(item_idx, num_nonzeros - 1);

        OffsetT     column_idx      = d_column_indices[item_idx];
        ValueT      value           = d_values[item_idx];
        ValueT      vector_value    = d_vector_x[column_idx];
        nonzero         += vector_value * value;
    }

    if (num_nonzeros < 0)
        no_elide[threadIdx.x] = nonzero;
}


/**
 * Read every row-offset, write every row-sum
 */
template <
    int         BLOCK_THREADS,
    typename    ValueT,
    typename    OffsetT>
__global__ void RowIO(
    int         num_rows,
    OffsetT*    d_row_offsets,
    ValueT*     d_vector_y)
{
    OffsetT item_idx = (blockIdx.x * BLOCK_THREADS) + threadIdx.x;
    item_idx = CUB_MIN(item_idx, num_rows - 1);
    d_vector_y[item_idx] = (ValueT) d_row_offsets[num_rows];
}


/**
 * Run GPU I/O proxy
 */
template <
    typename ValueT,
    typename OffsetT>
float IoSpmv(
    SpmvParams<ValueT, OffsetT>&    params,
    int                             timing_iterations)
{
    enum {
        BLOCK_THREADS       = 128,
        ITEMS_PER_THREAD    = 7,
        TILE_SIZE           = BLOCK_THREADS * ITEMS_PER_THREAD,
    };

    unsigned int nonzero_blocks = (params.num_nonzeros + TILE_SIZE - 1) / TILE_SIZE;
    unsigned int row_blocks = (params.num_rows + BLOCK_THREADS - 1) / BLOCK_THREADS;

    TexRefInputIterator<ValueT, 1234, int> x_itr;
    CubDebugExit(x_itr.BindTexture(params.d_vector_x));

    // Warmup
    NonZeroIO<BLOCK_THREADS, ITEMS_PER_THREAD><<<nonzero_blocks, BLOCK_THREADS>>>(
        params.num_nonzeros,
        params.d_values,
        params.d_column_indices,
        x_itr);

    // Check for failures
    CubDebugExit(cudaPeekAtLastError());
    CubDebugExit(SyncStream(0));

    RowIO<BLOCK_THREADS><<<row_blocks, BLOCK_THREADS>>>(
        params.num_rows,
        params.d_row_end_offsets,
        params.d_vector_y);

    // Check for failures
    CubDebugExit(cudaPeekAtLastError());
    CubDebugExit(SyncStream(0));

    // Timing
    GpuTimer gpu_timer;
    float elapsed_millis = 0.0;

    // Reset input/output vector y
    gpu_timer.Start();
    for (int it = 0; it < timing_iterations; ++it)
    {

        NonZeroIO<BLOCK_THREADS, ITEMS_PER_THREAD><<<nonzero_blocks, BLOCK_THREADS>>>(
            params.num_nonzeros,
            params.d_values,
            params.d_column_indices,
            x_itr);

        RowIO<BLOCK_THREADS><<<row_blocks, BLOCK_THREADS>>>(
            params.num_rows,
            params.d_row_end_offsets,
            params.d_vector_y);
    }
    gpu_timer.Stop();
    elapsed_millis += gpu_timer.ElapsedMillis();

    CubDebugExit(x_itr.UnbindTexture());

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
float CusparseSpmv(
    float*                          vector_y_in,
    float*                          vector_y_out,
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
        int compare = CompareDeviceResults(vector_y_out, params.d_vector_y, params.num_rows, true, g_verbose);
        printf("\t%s\n", compare ? "FAIL" : "PASS"); fflush(stdout);
    }

    // Timing
    float elapsed_millis    = 0.0;
    GpuTimer gpu_timer;

    gpu_timer.Start();
    for(int it = 0; it < timing_iterations; ++it)
    {
        AssertEquals(CUSPARSE_STATUS_SUCCESS, cusparseScsrmv(
            cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
            params.num_rows, params.num_cols, params.num_nonzeros, &params.alpha, desc,
            params.d_values, params.d_row_end_offsets, params.d_column_indices,
            params.d_vector_x, &params.beta, params.d_vector_y));
    }
    gpu_timer.Stop();
    elapsed_millis += gpu_timer.ElapsedMillis();

    cusparseDestroyMatDescr(desc);
    return elapsed_millis / timing_iterations;
}


/**
 * Run cuSparse SpMV (specialized for fp64)
 */
template <
    typename OffsetT>
float CusparseSpmv(
    double*                         vector_y_in,
    double*                         vector_y_out,
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
        int compare = CompareDeviceResults(vector_y_out, params.d_vector_y, params.num_rows, true, g_verbose);
        printf("\t%s\n", compare ? "FAIL" : "PASS"); fflush(stdout);
    }

    // Timing
    float elapsed_millis    = 0.0;
    GpuTimer gpu_timer;

    gpu_timer.Start();
    for(int it = 0; it < timing_iterations; ++it)
    {
        AssertEquals(CUSPARSE_STATUS_SUCCESS, cusparseDcsrmv(
            cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
            params.num_rows, params.num_cols, params.num_nonzeros, &params.alpha, desc,
            params.d_values, params.d_row_end_offsets, params.d_column_indices,
            params.d_vector_x, &params.beta, params.d_vector_y));

    }
    gpu_timer.Stop();
    elapsed_millis += gpu_timer.ElapsedMillis();

    cusparseDestroyMatDescr(desc);
    return elapsed_millis / timing_iterations;
}

//---------------------------------------------------------------------
// CUB SpMV
//---------------------------------------------------------------------

/**
 * Run CUB SpMV
 */
template <
    typename ValueT,
    typename OffsetT>
float CubSpmv(
    ValueT*                         vector_y_in,
    ValueT*                         vector_y_out,
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
        int compare = CompareDeviceResults(vector_y_out, params.d_vector_y, params.num_rows, true, g_verbose);
        printf("\t%s\n", compare ? "FAIL" : "PASS"); fflush(stdout);
    }

    // Timing
    GpuTimer gpu_timer;
    float elapsed_millis = 0.0;

    gpu_timer.Start();
    for(int it = 0; it < timing_iterations; ++it)
    {
        CubDebugExit(DeviceSpmv::CsrMV(
            d_temp_storage, temp_storage_bytes,
            params.d_values, params.d_row_end_offsets, params.d_column_indices,
            params.d_vector_x, params.d_vector_y,
            params.num_rows, params.num_cols, params.num_nonzeros, params.alpha, params.beta,
            (cudaStream_t) 0, false));
    }
    gpu_timer.Stop();
    elapsed_millis += gpu_timer.ElapsedMillis();

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
    float                           bandwidth_GBs,
    double                          avg_millis,
    CsrMatrix<ValueT, OffsetT>&     csr_matrix)
{
    double nz_throughput, effective_bandwidth;
    size_t total_bytes = (csr_matrix.num_nonzeros * (sizeof(ValueT) * 2 + sizeof(OffsetT))) +
        (csr_matrix.num_rows) * (sizeof(OffsetT) + sizeof(ValueT));

    nz_throughput       = double(csr_matrix.num_nonzeros) / avg_millis / 1.0e6;
    effective_bandwidth = double(total_bytes) / avg_millis / 1.0e6;

    if (!g_quiet)
        printf("fp%d: %.3f avg ms, %.3f gflops, %.3lf effective GB/s (%.1f%% peak)\n",
            sizeof(ValueT) * 8,
            avg_millis,
            2 * nz_throughput,
            effective_bandwidth,
            effective_bandwidth / bandwidth_GBs * 100);
    else
        printf("%.3f, %.3f, %.3lf, %.1f%%, ",
            avg_millis,
            2 * nz_throughput,
            effective_bandwidth,
            effective_bandwidth / bandwidth_GBs * 100);

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
    float               bandwidth_GBs,
    cusparseHandle_t    cusparse)
{
    // Initialize matrix in COO form
    CooMatrix<ValueT, OffsetT> coo_matrix;

    if (!mtx_filename.empty())
    {
        // Parse matrix market file
        printf("%s, ", mtx_filename.c_str());
        coo_matrix.InitMarket(mtx_filename, 1.0, !g_quiet);
    }
    else if (grid2d > 0)
    {
        // Generate 2D lattice
        printf("grid2d_%d, ", grid2d);
        coo_matrix.InitGrid2d(grid2d, false);
    }
    else if (grid3d > 0)
    {
        // Generate 3D lattice
        printf("grid3d_%d, ", grid3d);
        coo_matrix.InitGrid3d(grid3d, false);
    }
    else if (wheel > 0)
    {
        // Generate wheel graph
        printf("wheel_%d, ", grid2d);
        coo_matrix.InitWheel(wheel);
    }
    else if (dense > 0)
    {
        // Generate dense graph
        OffsetT rows = (1<<24) / dense;               // 16M nnz
        printf("dense_%d_x_%d, ", rows, dense);
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
    if (g_cpu)
    {
        CpuTimer cpu_timer;
        cpu_timer.Start();
        for (int it = 0; it < timing_iterations; ++it)
        {
            SpmvGold(csr_matrix, vector_x, vector_y_in, vector_y_out, alpha, beta);
        }
        cpu_timer.Stop();
        float avg_millis = cpu_timer.ElapsedMillis() / timing_iterations;
        DisplayPerf(bandwidth_GBs, avg_millis, csr_matrix);
    }

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
        printf("\n\nIO Proxy: "); fflush(stdout);
    }
    float avg_millis = IoSpmv(params, timing_iterations);
    DisplayPerf(bandwidth_GBs, avg_millis, csr_matrix);

    if (!g_quiet) {
        printf("\n\nCusparse: "); fflush(stdout);
    }
    avg_millis = CusparseSpmv(vector_y_in, vector_y_out, params, timing_iterations, cusparse);
    DisplayPerf(bandwidth_GBs, avg_millis, csr_matrix);

    if (!g_quiet) {
        printf("\n\nCUB: "); fflush(stdout);
    }
    avg_millis = CubSpmv(vector_y_in, vector_y_out, params, timing_iterations);
    DisplayPerf(bandwidth_GBs, avg_millis, csr_matrix);

    // Cleanup
    if (vector_x)                   delete[] vector_x;
    if (vector_y_in)                delete[] vector_y_in;
    if (vector_y_out)               delete[] vector_y_out;
    if (params.d_values)            g_allocator.DeviceFree(params.d_values);
    if (params.d_row_end_offsets)   g_allocator.DeviceFree(params.d_row_end_offsets);
    if (params.d_column_indices)    g_allocator.DeviceFree(params.d_column_indices);
    if (params.d_vector_x)          g_allocator.DeviceFree(params.d_vector_x);
    if (params.d_vector_y)          g_allocator.DeviceFree(params.d_vector_y);
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
    int                 timing_iterations   = 100;
    float               alpha               = 1.0;
    float               beta                = 0.0;

    g_verbose = args.CheckCmdLineFlag("v");
    g_verbose2 = args.CheckCmdLineFlag("v2");
    g_quiet = args.CheckCmdLineFlag("quiet");
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

    // Initialize device
    CubDebugExit(args.DeviceInit());

    // Initalize cuSparse
    cusparseHandle_t cusparse;
    AssertEquals(CUSPARSE_STATUS_SUCCESS, cusparseCreate(&cusparse));

    // Get GPU device bandwidth (GB/s)
    int device_ordinal, bus_width, mem_clock_khz;
    CubDebugExit(cudaGetDevice(&device_ordinal));
    CubDebugExit(cudaDeviceGetAttribute(&bus_width, cudaDevAttrGlobalMemoryBusWidth, device_ordinal));
    CubDebugExit(cudaDeviceGetAttribute(&mem_clock_khz, cudaDevAttrMemoryClockRate, device_ordinal));
    float bandwidth_GBs = float(bus_width) * mem_clock_khz * 2 / 8 / 1000 / 1000;

    // Run test(s)
    if (fp64)
    {
        RunTests<double, int>(rcm_relabel, alpha, beta, mtx_filename, grid2d, grid3d, wheel, dense, timing_iterations, bandwidth_GBs, cusparse);
    }
    else
    {
        RunTests<float, int>(rcm_relabel, alpha, beta, mtx_filename, grid2d, grid3d, wheel, dense, timing_iterations, bandwidth_GBs, cusparse);
    }

    CubDebugExit(cudaDeviceSynchronize());
    printf("\n");

    return 0;
}
