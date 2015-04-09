/******************************************************************************
 * Copyright (c) 2011, Duane Merrill.  All rights reserved.
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

/******************************************************************************
 * Matrix data structures and parsing logic
 ******************************************************************************/

#pragma once

#include <iterator>
#include <string>
#include <algorithm>
#include <iostream>
#include <queue>
#include <set>
#include <fstream>
#include <stdio.h>

using namespace std;

/******************************************************************************
 * COO matrix type
 ******************************************************************************/

struct GraphStats
{
    int         num_rows;
    int         num_cols;
    int         num_nonzeros;

    long long   row_profile;
    double      log_row_profile_occupancy;

    long long   intersection_profile;
    double      log_intersection_profile_occupancy;

    double      row_length_mean;        // mean
    double      row_length_variance;    // sample variance
    double      row_length_variation;   // coefficient of variation
    double      row_length_skewness;    // skewness

    void Display(bool show_labels = true)
    {
        if (show_labels)
            printf("\n"
                "\tnum_rows: %d\n"
                "\tnum_cols: %d\n"
                "\tnum_nonzeros: %d\n"
                "\trow_profile: %lld\n"
                "\tlog_row_profile_occupancy: %.10f\n"
                "\tintersection_profile: %lld\n"
                "\tlog_intersection_profile_occupancy: %.10f\n"
                "\trow_length_mean: %.5f\n"
                "\trow_length_variance: %.5f\n"
                "\trow_length_variation: %.5f\n"
                "\trow_length_skewness: %.5f\n",
                    num_rows,
                    num_cols,
                    num_nonzeros,
                    row_profile,
                    log_row_profile_occupancy,
                    intersection_profile,
                    log_intersection_profile_occupancy,
                    row_length_mean,
                    row_length_variance,
                    row_length_variation,
                    row_length_skewness);
        else
            printf(
                "%d, "
                "%d, "
                "%d, "
//                "%lld, "
                "%.10f, "
//                "%lld, "
                "%.10f, "
                "%.5f, "
                "%.5f, "
                "%.5f, "
                "%.5f, ",
                    num_rows,
                    num_cols,
                    num_nonzeros,
//                    row_profile,
                    log_row_profile_occupancy,
//                    intersection_profile,
                    log_intersection_profile_occupancy,
                    row_length_mean,
                    row_length_variance,
                    row_length_variation,
                    row_length_skewness);
    }
};



/******************************************************************************
 * COO matrix type
 ******************************************************************************/


/**
 * COO matrix type.  A COO matrix is just a vector of edge tuples.  Tuples are sorted
 * first by row, then by column.
 */
template<typename ValueT, typename OffsetT>
struct CooMatrix
{
    //---------------------------------------------------------------------
    // Type definitions and constants
    //---------------------------------------------------------------------

    // COO edge tuple
    struct CooTuple
    {
        OffsetT            row;
        OffsetT            col;
        ValueT             val;

        CooTuple() {}
        CooTuple(OffsetT row, OffsetT col) : row(row), col(col) {}
        CooTuple(OffsetT row, OffsetT col, ValueT val) : row(row), col(col), val(val) {}

        /**
         * Comparator for sorting COO sparse format num_nonzeros
         */
        bool operator<(const CooTuple &other) const
        {
            if ((row < other.row) || ((row == other.row) && (col < other.col)))
            {
                return true;
            }

            return false;
        }
    };


    //---------------------------------------------------------------------
    // Data members
    //---------------------------------------------------------------------

    // Fields
    int                 num_rows;
    int                 num_cols;
    int                 num_nonzeros;
    CooTuple*           coo_tuples;

    //---------------------------------------------------------------------
    // Methods
    //---------------------------------------------------------------------

    // Constructor
    CooMatrix() : num_rows(0), num_cols(0), num_nonzeros(0), coo_tuples(NULL) {}


    /**
     * Clear
     */
    void Clear()
    {
        if (coo_tuples) delete[] coo_tuples;
        coo_tuples = NULL;
    }


    // Destructor
    ~CooMatrix()
    {
        Clear();
    }


    // Display matrix to stdout
    void Display()
    {
        cout << "COO Matrix (" << num_rows << " rows, " << num_cols << " columns, " << num_nonzeros << " non-zeros):\n";
        cout << "Ordinal, Row, Column, Value\n";
        for (int i = 0; i < num_nonzeros; i++)
        {
            cout << '\t' << i << ',' << coo_tuples[i].row << ',' << coo_tuples[i].col << ',' << coo_tuples[i].val << "\n";
        }
    }


    /**
     * Builds a symmetric COO sparse from an asymmetric CSR matrix.
     */
    template <typename CsrMatrixT>
    void InitCsrSymmetric(CsrMatrixT &csr_matrix)
    {
        if (coo_tuples)
        {
            fprintf(stderr, "Matrix already constructed\n");
            exit(1);
        }

        num_rows        = csr_matrix.num_cols;
        num_cols        = csr_matrix.num_rows;
        num_nonzeros    = csr_matrix.num_nonzeros * 2;
        coo_tuples      = new CooTuple[num_nonzeros];

        for (OffsetT row = 0; row < csr_matrix.num_rows; ++row)
        {
            for (OffsetT nonzero = csr_matrix.row_offsets[row]; nonzero < csr_matrix.row_offsets[row + 1]; ++nonzero)
            {
                coo_tuples[nonzero].row = row;
                coo_tuples[nonzero].col = csr_matrix.column_indices[nonzero];
                coo_tuples[nonzero].val = csr_matrix.values[nonzero];

                coo_tuples[csr_matrix.num_nonzeros + nonzero].row = coo_tuples[nonzero].col;
                coo_tuples[csr_matrix.num_nonzeros + nonzero].col = coo_tuples[nonzero].row;
                coo_tuples[csr_matrix.num_nonzeros + nonzero].val = csr_matrix.values[nonzero];

            }
        }

        // Sort by rows, then columns
        std::stable_sort(coo_tuples, coo_tuples + num_nonzeros);
    }

    /**
     * Builds a COO sparse from a relabeled CSR matrix.
     */
    template <typename CsrMatrixT>
    void InitCsrRelabel(CsrMatrixT &csr_matrix, OffsetT* relabel_indices)
    {
        if (coo_tuples)
        {
            fprintf(stderr, "Matrix already constructed\n");
            exit(1);
        }

        num_rows        = csr_matrix.num_rows;
        num_cols        = csr_matrix.num_cols;
        num_nonzeros    = csr_matrix.num_nonzeros;
        coo_tuples      = new CooTuple[num_nonzeros];

        for (OffsetT row = 0; row < num_rows; ++row)
        {
            for (OffsetT nonzero = csr_matrix.row_offsets[row]; nonzero < csr_matrix.row_offsets[row + 1]; ++nonzero)
            {
                coo_tuples[nonzero].row = relabel_indices[row];
                coo_tuples[nonzero].col = relabel_indices[csr_matrix.column_indices[nonzero]];
                coo_tuples[nonzero].val = csr_matrix.values[nonzero];
            }
        }

        // Sort by rows, then columns
        std::stable_sort(coo_tuples, coo_tuples + num_nonzeros);
    }



    /**
     * Builds a METIS COO sparse from the given file.
     */
    void InitMetis(const string &metis_filename)
    {
        if (coo_tuples)
        {
            fprintf(stderr, "Matrix already constructed\n");
            exit(1);
        }

        // TODO
    }


    /**
     * Builds a MARKET COO sparse from the given file.
     */
    void InitMarket(
        const string&   market_filename,
        ValueT          default_value       = 1.0,
        bool            verbose             = false)
    {
        if (verbose) {
            printf("Reading... "); fflush(stdout);
        }

        if (coo_tuples)
        {
            fprintf(stderr, "Matrix already constructed\n");
            exit(1);
        }

        // Read from file
        FILE *f_in = fopen(market_filename.c_str(), "r");
        if (!f_in)
        {
            fprintf(stderr, "Could not open input file\n");
            exit(1);
        }

/*
        std::ifstream ifs;
        ifs.open(market_filename.c_str(), std::ifstream::in);
*/
        bool    symmetric = false;
        bool    skew = false;
        int     current_edge = -1;
        char    line[1024];

        if (verbose) {
            printf("Parsing... "); fflush(stdout);
        }

        while (true)
        {
/*
            ifs.getline(line, 1024);
            if (!ifs.good())
                break;
*/

            if (fscanf(f_in, "%[^\n]\n", line) <= 0)
            {
                // Done
                break;
            }

            if (line[0] == '%')
            {
                // Comment
                if (line[1] == '%')
                {
                    // Banner
                    symmetric = (strstr(line, "symmetric") != NULL);
                    skew = (strstr(line, "skew") != NULL);

                    if (verbose) {
                        printf("(symmetric: %d, skew: %d) ", symmetric, skew); fflush(stdout);
                    }
                }
            }
            else if (current_edge == -1)
            {
                // Problem description
                if (sscanf(line, "%d %d %d", &num_rows, &num_cols, &num_nonzeros) != 3)
                {
                    fprintf(stderr, "Error parsing MARKET matrix: invalid problem description\n");
                    exit(1);
                }

                if (symmetric)
                    num_nonzeros *= 2;

                // Allocate coo matrix
                coo_tuples = new CooTuple[num_nonzeros];
                current_edge = 0;
            }
            else
            {
                // Edge
                if (current_edge >= num_nonzeros)
                {
                    fprintf(stderr, "Error parsing MARKET matrix: encountered more than %d num_nonzeros\n", num_nonzeros);
                    exit(1);
                }

                int row, col;
                double val;
                int nparsed = sscanf(line, "%d %d %lf", &row, &col, &val);

                if (nparsed == 2)
                {
                    // No value specified
                    val = default_value;
                }
                else if (nparsed != 3)
                {
                    fprintf(stderr, "Error parsing MARKET matrix: badly formed current_edge\n", num_nonzeros);
                    exit(1);
                }

                coo_tuples[current_edge] = CooTuple(row - 1, col - 1, val);    // Convert indices to zero-based
                current_edge++;

                if (symmetric && (row != col))
                {
                    coo_tuples[current_edge].row = coo_tuples[current_edge - 1].col;
                    coo_tuples[current_edge].col = coo_tuples[current_edge - 1].row;
                    coo_tuples[current_edge].val = coo_tuples[current_edge - 1].val * (skew ? -1 : 1);
                    current_edge++;
                }
            }
        }

        // Adjust nonzero count (nonzeros along the diagonal aren't reversed)
        num_nonzeros = current_edge;

        if (verbose) {
            printf("done. Ordering..."); fflush(stdout);
        }

        // Sort by rows, then columns
        std::stable_sort(coo_tuples, coo_tuples + num_nonzeros);

        if (verbose) {
            printf("done. "); fflush(stdout);
        }

        fclose(f_in);
//        ifs.close();
    }


    /**
     * Builds a dense matrix
     */
    int InitDense(
        OffsetT     num_rows,
        OffsetT     num_cols,
        ValueT      default_value   = 1.0,
        bool        verbose         = false)
    {
        if (coo_tuples)
        {
            fprintf(stderr, "Matrix already constructed\n");
            exit(1);
        }

        this->num_rows  = num_rows;
        this->num_cols  = num_cols;

        num_nonzeros    = num_rows * num_cols;
        coo_tuples      = new CooTuple[num_nonzeros];

        for (OffsetT row = 0; row < num_rows; ++row)
        {
            for (OffsetT col = 0; col < num_cols; ++col)
            {
                coo_tuples[(row * num_cols) + col] = CooTuple(row, col, default_value);
            }
        }

        // Sort by rows, then columns
        std::stable_sort(coo_tuples, coo_tuples + num_nonzeros);

        return 0;
    }

    /**
     * Builds a wheel COO sparse matrix having spokes spokes.
     */
    int InitWheel(
        OffsetT     spokes,
        ValueT      default_value   = 1.0,
        bool        verbose         = false)
    {
        if (coo_tuples)
        {
            fprintf(stderr, "Matrix already constructed\n");
            exit(1);
        }

        num_rows        = spokes + 1;
        num_cols        = num_rows;
        num_nonzeros    = spokes * 2;
        coo_tuples      = new CooTuple[num_nonzeros];

        // Add spoke num_nonzeros
        int current_edge = 0;
        for (OffsetT i = 0; i < spokes; i++)
        {
            coo_tuples[current_edge] = CooTuple(0, i + 1, default_value);
            current_edge++;
        }

        // Add rim
        for (OffsetT i = 0; i < spokes; i++)
        {
            OffsetT dest = (i + 1) % spokes;
            coo_tuples[current_edge] = CooTuple(i + 1, dest + 1, default_value);
            current_edge++;
        }

        // Sort by rows, then columns
        std::stable_sort(coo_tuples, coo_tuples + num_nonzeros);

        return 0;
    }


    /**
     * Builds a square 2D grid CSR matrix.  Interior num_vertices have degree 5 when including
     * a self-loop.
     *
     * Returns 0 on success, 1 on failure.
     */
    int InitGrid2d(OffsetT width, bool self_loop, ValueT default_value = 1.0)
    {
        if (coo_tuples)
        {
            fprintf(stderr, "Matrix already constructed\n");
            exit(1);
        }

        int     interior_nodes  = (width - 2) * (width - 2);
        int     edge_nodes      = (width - 2) * 4;
        int     corner_nodes    = 4;
        num_rows                       = width * width;
        num_cols                       = num_rows;
        num_nonzeros                   = (interior_nodes * 4) + (edge_nodes * 3) + (corner_nodes * 2);

        if (self_loop)
            num_nonzeros += num_rows;

        coo_tuples          = new CooTuple[num_nonzeros];
        int current_edge    = 0;

        for (OffsetT j = 0; j < width; j++)
        {
            for (OffsetT k = 0; k < width; k++)
            {
                OffsetT me = (j * width) + k;

                // West
                OffsetT neighbor = (j * width) + (k - 1);
                if (k - 1 >= 0) {
                    coo_tuples[current_edge] = CooTuple(me, neighbor, default_value);
                    current_edge++;
                }

                // East
                neighbor = (j * width) + (k + 1);
                if (k + 1 < width) {
                    coo_tuples[current_edge] = CooTuple(me, neighbor, default_value);
                    current_edge++;
                }

                // North
                neighbor = ((j - 1) * width) + k;
                if (j - 1 >= 0) {
                    coo_tuples[current_edge] = CooTuple(me, neighbor, default_value);
                    current_edge++;
                }

                // South
                neighbor = ((j + 1) * width) + k;
                if (j + 1 < width) {
                    coo_tuples[current_edge] = CooTuple(me, neighbor, default_value);
                    current_edge++;
                }

                if (self_loop)
                {
                    neighbor = me;
                    coo_tuples[current_edge] = CooTuple(me, neighbor, default_value);
                    current_edge++;
                }
            }
        }

        // Sort by rows, then columns, update dims
        std::stable_sort(coo_tuples, coo_tuples + num_nonzeros);

        return 0;
    }


    /**
     * Builds a square 3D grid COO sparse matrix.  Interior num_vertices have degree 7 when including
     * a self-loop.  Values are unintialized, coo_tuples are sorted.
     */
    int InitGrid3d(OffsetT width, bool self_loop, ValueT default_value = 1.0)
    {
        if (coo_tuples)
        {
            fprintf(stderr, "Matrix already constructed\n");
            return -1;
        }

        OffsetT interior_nodes  = (width - 2) * (width - 2) * (width - 2);
        OffsetT face_nodes      = (width - 2) * (width - 2) * 6;
        OffsetT edge_nodes      = (width - 2) * 12;
        OffsetT corner_nodes    = 8;
        num_cols                       = width * width * width;
        num_rows                       = num_cols;
        num_nonzeros                     = (interior_nodes * 6) + (face_nodes * 5) + (edge_nodes * 4) + (corner_nodes * 3);

        if (self_loop)
            num_nonzeros += num_rows;

        coo_tuples          = new CooTuple[num_nonzeros];
        int current_edge    = 0;

        for (OffsetT i = 0; i < width; i++)
        {
            for (OffsetT j = 0; j < width; j++)
            {
                for (OffsetT k = 0; k < width; k++)
                {

                    OffsetT me = (i * width * width) + (j * width) + k;

                    // Up
                    OffsetT neighbor = (i * width * width) + (j * width) + (k - 1);
                    if (k - 1 >= 0) {
                        coo_tuples[current_edge] = CooTuple(me, neighbor, default_value);
                        current_edge++;
                    }

                    // Down
                    neighbor = (i * width * width) + (j * width) + (k + 1);
                    if (k + 1 < width) {
                        coo_tuples[current_edge] = CooTuple(me, neighbor, default_value);
                        current_edge++;
                    }

                    // West
                    neighbor = (i * width * width) + ((j - 1) * width) + k;
                    if (j - 1 >= 0) {
                        coo_tuples[current_edge] = CooTuple(me, neighbor, default_value);
                        current_edge++;
                    }

                    // East
                    neighbor = (i * width * width) + ((j + 1) * width) + k;
                    if (j + 1 < width) {
                        coo_tuples[current_edge] = CooTuple(me, neighbor, default_value);
                        current_edge++;
                    }

                    // North
                    neighbor = ((i - 1) * width * width) + (j * width) + k;
                    if (i - 1 >= 0) {
                        coo_tuples[current_edge] = CooTuple(me, neighbor, default_value);
                        current_edge++;
                    }

                    // South
                    neighbor = ((i + 1) * width * width) + (j * width) + k;
                    if (i + 1 < width) {
                        coo_tuples[current_edge] = CooTuple(me, neighbor, default_value);
                        current_edge++;
                    }

                    if (self_loop)
                    {
                        neighbor = me;
                        coo_tuples[current_edge] = CooTuple(me, neighbor, default_value);
                        current_edge++;
                    }
                }
            }
        }

        // Sort by rows, then columns, update dims
        std::stable_sort(coo_tuples, coo_tuples + num_nonzeros);

        return 0;
    }
};



/******************************************************************************
 * COO matrix type
 ******************************************************************************/


/**
 * CSR sparse format matrix
 */
template<
    typename ValueT,
    typename OffsetT>
struct CsrMatrix
{
    int         num_rows;
    int         num_cols;
    int         num_nonzeros;
    OffsetT*    row_offsets;
    OffsetT*    column_indices;
    ValueT*     values;

    /**
     * Constructor
     */
    CsrMatrix() : num_rows(0), num_cols(0), num_nonzeros(0), row_offsets(NULL), column_indices(NULL), values(NULL) {}


    /**
     * Clear
     */
    void Clear()
    {
        if (row_offsets)    delete[] row_offsets;
        if (column_indices) delete[] column_indices;
        if (values)         delete[] values;

        row_offsets = NULL;
        column_indices = NULL;
        values = NULL;
    }

    /**
     * Destructor
     */
    ~CsrMatrix()
    {
        Clear();
    }

    GraphStats Stats()
    {
        GraphStats stats;
        stats.num_rows = num_rows;
        stats.num_cols = num_cols;
        stats.num_nonzeros = num_nonzeros;

/*
        // Compute diagonal profile

        OffsetT num_diags = num_rows + num_cols;
        OffsetT* diag_min = new OffsetT[num_diags];       // upper right
        OffsetT* diag_max = new OffsetT[num_diags];       // lower left
        for (OffsetT diag = 0; diag < num_diags; ++diag)
        {
            diag_min[diag] = num_rows;
            diag_max[diag] = -1;
        }
        for (OffsetT row = 0; row < num_rows; ++row)
        {
            for (OffsetT column_idx = row_offsets[row]; column_idx < row_offsets[row + 1]; ++column_idx)
            {
                OffsetT nz_diag = row + column_indices[column_idx];
                diag_min[nz_diag] = std::min(diag_min[nz_diag], row);
                diag_max[nz_diag] = std::max(diag_max[nz_diag], row);
            }
        }
        stats.counter_diagonal_profile = 0;
        for (OffsetT diag = 0; diag < num_diags; ++diag)
        {
            OffsetT delta = diag_max[diag] - diag_min[diag];
            if (delta > 0)
            {
                stats.counter_diagonal_profile += delta;
            }
        }
        stats.log_counter_diagonal_occupancy = pow(2.0, double(num_nonzeros) / double(stats.counter_diagonal_profile));
*/
        // Compute row profile

        stats.row_profile = 0;
        for (OffsetT row = 0; row < num_rows; ++row)
        {
            if (row_offsets[row + 1] > row_offsets[row])
            {
                OffsetT row_segment = column_indices[row_offsets[row + 1] - 1] - column_indices[row_offsets[row]] + 1;
                stats.row_profile += row_segment;
            }
        }
        stats.log_row_profile_occupancy = pow(2.0, double(num_nonzeros) / double(stats.row_profile));

        // Compute column profile

        OffsetT* min_row = new OffsetT[num_cols];
        OffsetT* max_row = new OffsetT[num_cols];

        for (OffsetT col = 0; col < num_cols; ++col)
        {
            min_row[col] = num_rows;
            max_row[col] = -1;
        }
        for (OffsetT row = 0; row < num_rows; ++row)
        {
            for (OffsetT column_idx = row_offsets[row]; column_idx < row_offsets[row + 1]; ++column_idx)
            {
                OffsetT col = column_indices[column_idx];
                min_row[col] = std::min(min_row[col], row);
                max_row[col] = std::max(max_row[col], row);
            }
        }

        long long col_profile = 0;
        for (OffsetT col = 0; col < num_cols; ++col)
        {
            OffsetT delta = max_row[col] - min_row[col] + 1;
            if (delta > 0)
                col_profile += delta;
        }

        // Compute average profile

        stats.intersection_profile = (stats.row_profile + col_profile) / 2;
        stats.log_intersection_profile_occupancy = pow(2.0, double(num_nonzeros) / double(stats.intersection_profile));

/*
        // Compute intersection profile

        // Compute the min and max row for each column

        stats.intersection_profile = 0;
        for (OffsetT row = 0; row < num_rows; ++row)
        {
            if (row_offsets[row] < row_offsets[row + 1])
            {
                // Non-empty row
                OffsetT first_col = column_indices[row_offsets[row]];
                OffsetT last_col  = column_indices[row_offsets[row + 1] - 1];

                for (OffsetT col = first_col; col <= last_col; ++col)
                {
                    if ((min_row[col] <= row) && (max_row[col] >= row))
                    {
                        ++stats.intersection_profile;
                    }
                }
            }
        }
        stats.log_intersection_profile_occupancy = pow(2.0, double(num_nonzeros) / double(stats.intersection_profile));
*/

        // Compute row-length statistics

        // Sample mean
        stats.row_length_mean       = float(num_nonzeros) / num_rows;
        stats.row_length_variance   = 0.0;
        stats.row_length_skewness   = 0.0;
        for (int row = 0; row < num_rows; ++row)
        {
            int length                  = row_offsets[row + 1] - row_offsets[row];
            double delta                = length - stats.row_length_mean;
            stats.row_length_variance   += delta * delta;
            stats.row_length_skewness   += delta * delta * delta;
        }
        stats.row_length_variance   /= (num_rows - 1);
        double std_dev              = sqrt(stats.row_length_variance);
        stats.row_length_skewness   /= (num_rows);
        stats.row_length_skewness   = stats.row_length_skewness / pow(std_dev, 3.0);
        stats.row_length_variation  = std_dev / stats.row_length_mean;

        // Cleanup
        delete[] min_row;
        delete[] max_row;

        return stats;
    }

    /**
     * Build CSR matrix from sorted COO matrix
     */
    void FromCoo(const CooMatrix<ValueT, OffsetT> &coo_matrix)
    {
        num_rows        = coo_matrix.num_rows;
        num_cols        = coo_matrix.num_cols;
        num_nonzeros    = coo_matrix.num_nonzeros;

        row_offsets     = new OffsetT[num_rows + 1];
        column_indices  = new OffsetT[num_nonzeros];
        values          = new ValueT[num_nonzeros];

        OffsetT prev_row = -1;
        for (OffsetT current_edge = 0; current_edge < num_nonzeros; current_edge++)
        {
            OffsetT current_row = coo_matrix.coo_tuples[current_edge].row;

            // Fill in rows up to and including the current row
            for (OffsetT row = prev_row + 1; row <= current_row; row++)
            {
                row_offsets[row] = current_edge;
            }
            prev_row = current_row;

            column_indices[current_edge]    = coo_matrix.coo_tuples[current_edge].col;
            values[current_edge]            = coo_matrix.coo_tuples[current_edge].val;
        }

        // Fill out any trailing edgeless vertices (and the end-of-list element)
        for (OffsetT row = prev_row + 1; row <= num_rows; row++)
        {
            row_offsets[row] = num_nonzeros;
        }
    }


    /**
     * Display log-histogram to stdout
     */
    void DisplayHistogram()
    {
        // Initialize
        int log_counts[9];
        for (int i = 0; i < 9; i++)
        {
            log_counts[i] = 0;
        }

        // Scan
        int max_log_length = -1;
        for (OffsetT row = 0; row < num_rows; row++)
        {
            OffsetT length = row_offsets[row + 1] - row_offsets[row];

            int log_length = -1;
            while (length > 0)
            {
                length /= 10;
                log_length++;
            }
            if (log_length > max_log_length)
            {
                max_log_length = log_length;
            }

            log_counts[log_length + 1]++;
        }
        printf("CSR matrix (%d rows, %d columns, %d non-zeros):\n", (int) num_rows, (int) num_cols, (int) num_nonzeros);
        for (int i = -1; i < max_log_length + 1; i++)
        {
            printf("\tDegree 1e%d: \t%d (%.2f%%)\n", i, log_counts[i + 1], (float) log_counts[i + 1] * 100.0 / num_cols);
        }
        fflush(stdout);
    }


    /**
     * Display matrix to stdout
     */
    void Display()
    {
        printf("Input Matrix:\n");
        for (OffsetT row = 0; row < num_rows; row++)
        {
            printf("%d [@%d, #%d]: ", row, row_offsets[row], row_offsets[row + 1] - row_offsets[row]);
            for (OffsetT current_edge = row_offsets[row]; current_edge < row_offsets[row + 1]; current_edge++)
            {
                printf("%d (%f), ", column_indices[current_edge], values[current_edge]);
            }
            printf("\n");
        }
        fflush(stdout);
    }


};



/******************************************************************************
 * Matrix transformations
 ******************************************************************************/

// Comparator for ordering rows by degree (lowest first), then by row-id (lowest first)
template <typename OffsetT>
struct OrderByLow
{
    OffsetT* row_degrees;
    OrderByLow(OffsetT* row_degrees) : row_degrees(row_degrees) {}

    bool operator()(const OffsetT &a, const OffsetT &b)
    {
        if (row_degrees[a] < row_degrees[b])
            return true;
        else if (row_degrees[a] > row_degrees[b])
            return false;
        else
            return (a < b);
    }
};

// Comparator for ordering rows by degree (highest first), then by row-id (lowest first)
template <typename OffsetT>
struct OrderByHigh
{
    OffsetT* row_degrees;
    OrderByHigh(OffsetT* row_degrees) : row_degrees(row_degrees) {}

    bool operator()(const OffsetT &a, const OffsetT &b)
    {
        if (row_degrees[a] > row_degrees[b])
            return true;
        else if (row_degrees[a] < row_degrees[b])
            return false;
        else
            return (a < b);
    }
};



/**
 * Reverse Cuthill-McKee
 */
template <typename ValueT, typename OffsetT>
void RcmRelabel(
    CsrMatrix<ValueT, OffsetT>&     matrix,
    OffsetT*                        relabel_indices)
{
    // Initialize row degrees
    OffsetT* row_degrees_in     = new OffsetT[matrix.num_rows];
    OffsetT* row_degrees_out    = new OffsetT[matrix.num_rows];
    for (OffsetT row = 0; row < matrix.num_rows; ++row)
    {
        row_degrees_in[row]         = 0;
        row_degrees_out[row]        = matrix.row_offsets[row + 1] - matrix.row_offsets[row];
    }
    for (OffsetT nonzero = 0; nonzero < matrix.num_nonzeros; ++nonzero)
    {
        row_degrees_in[matrix.column_indices[nonzero]]++;
    }

    // Initialize unlabeled set 
    typedef std::set<OffsetT, OrderByLow<OffsetT> > UnlabeledSet;
    typename UnlabeledSet::key_compare  unlabeled_comp(row_degrees_in);
    UnlabeledSet                        unlabeled(unlabeled_comp);
    for (OffsetT row = 0; row < matrix.num_rows; ++row)
    {
        relabel_indices[row]    = -1;
        unlabeled.insert(row);
    }

    // Initialize queue set
    std::deque<OffsetT> q;

    // Process unlabeled vertices (traverse connected components)
    OffsetT relabel_idx = 0;
    while (!unlabeled.empty())
    {
        // Seed the unvisited frontier queue with the unlabeled vertex of lowest-degree
        OffsetT vertex = *unlabeled.begin();
        q.push_back(vertex);

        while (!q.empty())
        {
            vertex = q.front();
            q.pop_front();

            if (relabel_indices[vertex] == -1)
            {
                // Update this vertex
                unlabeled.erase(vertex);
                relabel_indices[vertex] = relabel_idx;
                relabel_idx++;

                // Sort neighbors by degree
                OrderByLow<OffsetT> neighbor_comp(row_degrees_in);
                std::sort(
                    matrix.column_indices + matrix.row_offsets[vertex],
                    matrix.column_indices + matrix.row_offsets[vertex + 1],
                    neighbor_comp);

                // Inspect neighbors, adding to the out frontier if unlabeled
                for (OffsetT neighbor_idx = matrix.row_offsets[vertex];
                    neighbor_idx < matrix.row_offsets[vertex + 1];
                    ++neighbor_idx)
                {
                    OffsetT neighbor = matrix.column_indices[neighbor_idx];
                    q.push_back(neighbor);
                }
            }
        }
    }

/*
    // Reverse labels
    for (int row = 0; row < matrix.num_rows; ++row)
    {
        relabel_indices[row] = matrix.num_rows - relabel_indices[row] - 1;
    }
*/

    // Cleanup
    if (row_degrees_in) delete[] row_degrees_in;
    if (row_degrees_out) delete[] row_degrees_out;
}


/**
 * Reverse Cuthill-McKee
 */
template <typename ValueT, typename OffsetT>
void RcmRelabel(
    CsrMatrix<ValueT, OffsetT>&     matrix,
    bool                            verbose = false)
{
    // Do not process if not square
    if (matrix.num_cols != matrix.num_rows)
    {
        if (verbose) {
            printf("RCM transformation ignored (not square)\n"); fflush(stdout);
        }
        return;
    }

    // Initialize relabel indices
    OffsetT* relabel_indices = new OffsetT[matrix.num_rows];

    if (verbose) {
        printf("RCM relabeling... "); fflush(stdout);
    }

    RcmRelabel(matrix, relabel_indices);

    if (verbose) {
        printf("done. Reconstituting... "); fflush(stdout);
    }

    // Create a COO matrix from the relabel indices
    CooMatrix<ValueT, OffsetT> coo_matrix;
    coo_matrix.InitCsrRelabel(matrix, relabel_indices);

    // Reconstitute the CSR matrix from the sorted COO tuples
    if (relabel_indices) delete[] relabel_indices;
    matrix.Clear();
    matrix.FromCoo(coo_matrix);

    if (verbose) {
        printf("done. "); fflush(stdout);
    }
}




