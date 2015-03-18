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
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
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

#include "matrix.h"

// Ensure printing of CUDA runtime errors to console
#define CUB_STDERR

#include <cub/device/device_spmv.cuh>
#include <cub/util_allocator.cuh>
#include <test/test_util.h>

using namespace cub;


//---------------------------------------------------------------------
// Globals, constants, and type declarations
//---------------------------------------------------------------------


bool                    g_verbose   = false;      // Whether to display output to console
bool                    g_verbose2  = false;     // Whether to display input to console
CachingDeviceAllocator  g_allocator(true);      // Caching allocator for device memory


//---------------------------------------------------------------------
// SpMV verification
//---------------------------------------------------------------------

// Compute reference SpMV y = Ax
template <
    typename ValueT,
    typename OffsetT>
void SpmvGold(
    CsrMatrix<ValueT, OffsetT>&     matrix_a,
    ValueT*                         vector_x,
    ValueT*                         vector_y_in,
    ValueT*                         vector_y_out,
    ValueT                          alpha,
    ValueT                          beta)
{
    for (OffsetT row = 0; row < matrix_a.num_rows; ++row)
    {
        vector_y_out[row] = beta * vector_y_in[row];
        for (
            OffsetT offset = matrix_a.row_offsets[row];
            offset < matrix_a.row_offsets[row + 1];
            ++offset)
        {
            vector_y_out[row] += alpha * matrix_a.values[offset] * vector_x[matrix_a.column_indices[offset]];
        }
    }
}


//---------------------------------------------------------------------
// GPU SpMV execution
//---------------------------------------------------------------------

/**
 * Run cuSparse SpMV (specialized for fp32)
 */
template <
    typename OffsetT>
float CusparseSpmv(
    float*              vector_y_in,
    float*              d_matrix_values,
    OffsetT*            d_matrix_row_offsets,
    OffsetT*            d_matrix_column_indices,
    float*              d_vector_x,
    float*              d_vector_y,
    int                 num_rows,
    int                 num_cols,
    int                 num_nonzeros,
    float               alpha,
    float               beta,
    int                 timing_iterations,
    cusparseHandle_t    cusparse)
{
    cusparseMatDescr_t desc;
    cusparseCreateMatDescr(&desc);

    // Reset input/output vector y
    CubDebugExit(cudaMemcpy(d_vector_y, vector_y_in, sizeof(float) * num_rows, cudaMemcpyHostToDevice));

    // Warmup
    AssertEquals(CUSPARSE_STATUS_SUCCESS, cusparseScsrmv(
        cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
        num_rows, num_cols, num_nonzeros, &alpha, desc,
        d_matrix_values, d_matrix_row_offsets, d_matrix_column_indices,
        d_vector_x, &beta, d_vector_y));

    // Timing
    float elapsed_millis    = 0.0;
    GpuTimer gpu_timer;

    for(int it = 0; it < timing_iterations; ++it)
    {
        // Reset input/output vector y
        CubDebugExit(cudaMemcpy(d_vector_y, vector_y_in, sizeof(float) * num_rows, cudaMemcpyHostToDevice));

        gpu_timer.Start();

        cusparseScsrmv(
            cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
            num_rows, num_cols, num_nonzeros, &alpha, desc,
            d_matrix_values, d_matrix_row_offsets, d_matrix_column_indices,
            d_vector_x, &beta, d_vector_y);

        gpu_timer.Stop();
        elapsed_millis += gpu_timer.ElapsedMillis();
    }

    cusparseDestroyMatDescr(desc);
    return elapsed_millis / timing_iterations;
}


/**
 * Run cuSparse SpMV (specialized for fp64)
 */
template <
    typename OffsetT>
float CusparseSpmv(
    double*             vector_y_in,
    double*             d_matrix_values,
    OffsetT*            d_matrix_row_offsets,
    OffsetT*            d_matrix_column_indices,
    double*             d_vector_x,
    double*             d_vector_y,
    int                 num_rows,
    int                 num_cols,
    int                 num_nonzeros,
    double              alpha,
    double              beta,
    int                 timing_iterations,
    cusparseHandle_t    cusparse)
{
    cusparseMatDescr_t desc;
    cusparseCreateMatDescr(&desc);

    // Reset input/output vector y
    CubDebugExit(cudaMemcpy(d_vector_y, vector_y_in, sizeof(double) * num_rows, cudaMemcpyHostToDevice));

    // Warmup
    AssertEquals(CUSPARSE_STATUS_SUCCESS, cusparseDcsrmv(
        cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
        num_rows, num_cols, num_nonzeros, &alpha, desc,
        d_matrix_values, d_matrix_row_offsets, d_matrix_column_indices,
        d_vector_x, &beta, d_vector_y));

    // Timing
    float elapsed_millis    = 0.0;
    GpuTimer gpu_timer;

    for(int it = 0; it < timing_iterations; ++it)
    {
        // Reset input/output vector y
        CubDebugExit(cudaMemcpy(d_vector_y, vector_y_in, sizeof(double) * num_rows, cudaMemcpyHostToDevice));

        gpu_timer.Start();

        cusparseDcsrmv(
            cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
            num_rows, num_cols, num_nonzeros, &alpha, desc,
            d_matrix_values, d_matrix_row_offsets, d_matrix_column_indices,
            d_vector_x, &beta, d_vector_y);

        gpu_timer.Stop();
        elapsed_millis += gpu_timer.ElapsedMillis();
    }

    cusparseDestroyMatDescr(desc);
    return elapsed_millis / timing_iterations;
}


/**
 * Run CUB SpMV
 */
template <
    typename ValueT,
    typename OffsetT>
float CubSpmv(
    ValueT*             vector_y_in,
    ValueT*             d_matrix_values,
    OffsetT*            d_matrix_row_offsets,
    OffsetT*            d_matrix_column_indices,
    ValueT*             d_vector_x,
    ValueT*             d_vector_y,
    int                 num_rows,
    int                 num_cols,
    int                 num_nonzeros,
    ValueT              alpha,
    ValueT              beta,
    int                 timing_iterations)
{
    // Allocate temporary storage
    size_t temp_storage_bytes = 0;
    void *d_temp_storage = NULL;

    // Get amount of temporary storage needed
    CubDebugExit(DeviceSpmv::CsrMV(
        d_temp_storage, temp_storage_bytes,
        d_matrix_values, d_matrix_row_offsets, d_matrix_column_indices,
        d_vector_x, d_vector_y,
        num_rows, num_cols, num_nonzeros, alpha, beta,
        (cudaStream_t) 0, false));

    // Allocate
    CubDebugExit(g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes));

    // Reset input/output vector y
    CubDebugExit(cudaMemcpy(d_vector_y, vector_y_in, sizeof(ValueT) * num_rows, cudaMemcpyHostToDevice));

    // Warmup
    CubDebugExit(DeviceSpmv::CsrMV(
        d_temp_storage, temp_storage_bytes,
        d_matrix_values, d_matrix_row_offsets, d_matrix_column_indices,
        d_vector_x, d_vector_y,
        num_rows, num_cols, num_nonzeros, alpha, beta,
        (cudaStream_t) 0, true));

    // Timing
    GpuTimer gpu_timer;
    float elapsed_millis = 0.0;

    for(int it = 0; it < timing_iterations; ++it)
    {
        // Reset input/output vector y
        CubDebugExit(cudaMemcpy(d_vector_y, vector_y_in, sizeof(ValueT) * num_rows, cudaMemcpyHostToDevice));

        gpu_timer.Start();

        CubDebugExit(DeviceSpmv::CsrMV(
            d_temp_storage, temp_storage_bytes,
            d_matrix_values, d_matrix_row_offsets, d_matrix_column_indices,
            d_vector_x, d_vector_y,
            num_rows, num_cols, num_nonzeros, alpha, beta,
            (cudaStream_t) 0, false));

        gpu_timer.Stop();
        elapsed_millis += gpu_timer.ElapsedMillis();
    }

    return elapsed_millis / timing_iterations;
}

//---------------------------------------------------------------------
// Test generation
//---------------------------------------------------------------------

/**
 * Run tests
 */
template <
    typename ValueT,
    typename OffsetT>
void RunTests(
    ValueT              alpha,
    ValueT              beta,
    std::string         &mtx_filename,
    int                 grid2d,
    int                 grid3d,
    int                 wheel,
    int                 timing_iterations,
    float               bandwidth_GBs,
    cusparseHandle_t    cusparse)
{
    // Initialize matrix in COO form
    CooMatrix<OffsetT, ValueT> coo_matrix;

    if (!mtx_filename.empty())
    {
        // Parse matrix market file
        cout << "Reading matrix market file " << mtx_filename << "... "; fflush(stdout);
        coo_matrix.InitMarket(mtx_filename);
        cout << "done.\n"; fflush(stdout);
    }
    else if (grid2d > 0)
    {
        // Generate 2D lattice
        coo_matrix.InitGrid2d(grid2d, false);
    }
    else if (grid3d > 0)
    {
        // Generate 3D lattice
        coo_matrix.InitGrid3d(grid3d, false);
    }
    else if (wheel > 0)
    {
        // Generate wheel graph
        coo_matrix.InitWheel(wheel);
    }
    else
    {
        fprintf(stderr, "No graph type specified.\n");
        exit(1);
    }

    CsrMatrix<ValueT, OffsetT> csr_matrix;
    csr_matrix.FromCoo(coo_matrix);

    // Display matrix info
    csr_matrix.DisplayHistogram();
    if (g_verbose2) csr_matrix.Display();
    printf("\n");

    // Allocate input and output vectors
    ValueT* vector_x        = new ValueT[csr_matrix.num_cols];
    ValueT* vector_y_in     = new ValueT[csr_matrix.num_rows];
    ValueT* vector_y_out    = new ValueT[csr_matrix.num_rows];

    for (int col = 0; col < csr_matrix.num_cols; ++col)
        vector_x[col] = 1.0;

    for (int row = 0; row < csr_matrix.num_cols; ++row)
        vector_y_in[row] = 1.0;

    // Compute reference answer
    SpmvGold(csr_matrix, vector_x, vector_y_in, vector_y_out, alpha, beta);

    // Allocate and initialize GPU problem
    ValueT*             d_matrix_values;
    OffsetT*            d_matrix_row_offsets;
    OffsetT*            d_matrix_column_indices;
    ValueT*             d_vector_x;
    ValueT*             d_vector_y;

    g_allocator.DeviceAllocate((void **) &d_matrix_values,          sizeof(ValueT) * csr_matrix.num_nonzeros);
    g_allocator.DeviceAllocate((void **) &d_matrix_row_offsets,     sizeof(OffsetT) * (csr_matrix.num_rows + 1));
    g_allocator.DeviceAllocate((void **) &d_matrix_column_indices,  sizeof(OffsetT) * csr_matrix.num_nonzeros);
    g_allocator.DeviceAllocate((void **) &d_vector_x,               sizeof(ValueT) * csr_matrix.num_cols);
    g_allocator.DeviceAllocate((void **) &d_vector_y,               sizeof(ValueT) * csr_matrix.num_rows);

    CubDebugExit(cudaMemcpy(d_matrix_values,            csr_matrix.values,          sizeof(ValueT) * csr_matrix.num_nonzeros, cudaMemcpyHostToDevice));
    CubDebugExit(cudaMemcpy(d_matrix_row_offsets,       csr_matrix.row_offsets,     sizeof(OffsetT) * (csr_matrix.num_rows + 1), cudaMemcpyHostToDevice));
    CubDebugExit(cudaMemcpy(d_matrix_column_indices,    csr_matrix.column_indices,  sizeof(OffsetT) * csr_matrix.num_nonzeros, cudaMemcpyHostToDevice));
    CubDebugExit(cudaMemcpy(d_vector_x,                 vector_x,                   sizeof(ValueT) * csr_matrix.num_cols, cudaMemcpyHostToDevice));

    double avg_millis, nz_throughput, effective_bandwidth;
    int compare = 0;
    size_t total_bytes = (csr_matrix.num_nonzeros * (sizeof(ValueT) * 2 + sizeof(OffsetT))) +
        (csr_matrix.num_rows) * (sizeof(OffsetT) + sizeof(ValueT));

    // Run problem on cuSparse

    CubDebugExit(cudaMemset(d_vector_y, 0, sizeof(ValueT) * csr_matrix.num_rows));

    avg_millis = CusparseSpmv(
        vector_y_in,
        d_matrix_values, d_matrix_row_offsets, d_matrix_column_indices, d_vector_x, d_vector_y,
        csr_matrix.num_rows, csr_matrix.num_cols, csr_matrix.num_nonzeros,
        alpha, beta,
        timing_iterations, cusparse);

    nz_throughput       = double(csr_matrix.num_nonzeros) / avg_millis / 1.0e6;
    effective_bandwidth = double(total_bytes) / avg_millis / 1.0e6;

    printf("\n\n%s fp%d: %.3f avg ms, %.3f gflops, %.3lf effective GB/s (%.1f%% peak)\n",
        "cuSparse",
        sizeof(ValueT) * 8,
        avg_millis,
        2 * nz_throughput,
        effective_bandwidth,
        effective_bandwidth / bandwidth_GBs * 100);

    compare = CompareDeviceResults(vector_y_out, d_vector_y, csr_matrix.num_rows, true, g_verbose);
    printf("\t%s\n", compare ? "FAIL" : "PASS"); fflush(stdout);
//    AssertEquals(0, compare);

    // Run problem on CUB

    CubDebugExit(cudaMemset(d_vector_y, 0, sizeof(ValueT) * csr_matrix.num_rows));

    avg_millis = CubSpmv(
        vector_y_in,
        d_matrix_values, d_matrix_row_offsets, d_matrix_column_indices, d_vector_x, d_vector_y,
        csr_matrix.num_rows, csr_matrix.num_cols, csr_matrix.num_nonzeros,
        alpha, beta,
        timing_iterations);

    nz_throughput       = double(csr_matrix.num_nonzeros) / avg_millis / 1.0e6;
    effective_bandwidth = double(total_bytes) / avg_millis / 1.0e6;

    printf("\n\n%s fp%d: %.3f avg ms, %.3f gflops, %.3lf effective GB/s (%.1f%% peak)\n",
        "CUB",
        sizeof(ValueT) * 8,
        avg_millis,
        2 * nz_throughput,
        effective_bandwidth,
        effective_bandwidth / bandwidth_GBs * 100);

    compare = CompareDeviceResults(vector_y_out, d_vector_y, csr_matrix.num_rows, true, g_verbose);
    printf("\t%s\n", compare ? "FAIL" : "PASS"); fflush(stdout);
    AssertEquals(0, compare);

    // Cleanup
    if (vector_x)                   delete[] vector_x;
    if (vector_y_in)                delete[] vector_y_in;
    if (vector_y_out)               delete[] vector_y_out;
    if (d_matrix_values)            g_allocator.DeviceFree(d_matrix_values);
    if (d_matrix_row_offsets)       g_allocator.DeviceFree(d_matrix_row_offsets);
    if (d_matrix_column_indices)    g_allocator.DeviceFree(d_matrix_column_indices);
    if (d_vector_x)                 g_allocator.DeviceFree(d_vector_x);
    if (d_vector_y)                 g_allocator.DeviceFree(d_vector_y);
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
            "[--v] "
            "[--i=<timing iterations>] "
            "[--fp64] "
            "[--alpha=<alpha scalar (default: 1.0)>] "
            "[--beta=<beta scalar (default: 0.0)>] "
            "\n\t"
                "--mtx=<matrix market file> "
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
    std::string         mtx_filename;
    int                 grid2d              = -1;
    int                 grid3d              = -1;
    int                 wheel               = -1;
    int                 timing_iterations   = 100;
    float               alpha               = 1.0;
    float               beta                = 0.0;

    g_verbose = args.CheckCmdLineFlag("v");
    g_verbose2 = args.CheckCmdLineFlag("v2");
    fp64 = args.CheckCmdLineFlag("fp64");
    args.GetCmdLineArgument("i", timing_iterations);
    args.GetCmdLineArgument("mtx", mtx_filename);
    args.GetCmdLineArgument("grid2d", grid2d);
    args.GetCmdLineArgument("grid3d", grid3d);
    args.GetCmdLineArgument("wheel", wheel);
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

    printf("Peak global B/W: %.3f GB/s (%d bits, %d kHz clock)\n", bandwidth_GBs, bus_width, mem_clock_khz);

    // Run test(s)
    if (fp64)
    {
        RunTests<double, int>(alpha, beta, mtx_filename, grid2d, grid3d, wheel, timing_iterations, bandwidth_GBs, cusparse);
    }
    else
    {
        RunTests<float, int>(alpha, beta, mtx_filename, grid2d, grid3d, wheel, timing_iterations, bandwidth_GBs, cusparse);
    }

    CubDebugExit(cudaDeviceSynchronize());
    printf("\n\n");

    return 0;
}
