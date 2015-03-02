/******************************************************************************
 * Copyright (c) 2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2014, NVIDIA CORPORATION.  All rights reserved.
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
#include <stdio.h>

using namespace std;



/******************************************************************************
 * COO matrix type
 ******************************************************************************/


/**
 * COO matrix type.  A COO matrix is just a vector of edge tuples.  Tuples are sorted
 * first by row, then by column.
 */
template<typename OffsetT, typename ValueT>
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


    // Destructor
    ~CooMatrix()
    {
        if (coo_tuples)
        {
            delete[] coo_tuples;
        }
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
    void InitMarket(const string &market_filename, ValueT default_value = 1.0)
    {
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

        int     current_edge = -1;
        char    line[1024];

        while(true)
        {
            if (fscanf(f_in, "%[^\n]\n", line) <= 0)
            {
                break;
            }
            if (line[0] == '%')
            {
                // Comment
            }
            else if (current_edge == -1)
            {
                // Problem description
                if (sscanf(line, "%d %d %d", &num_rows, &num_cols, &num_nonzeros) != 3)
                {
                    fprintf(stderr, "Error parsing MARKET matrix: invalid problem description\n");
                    exit(1);
                }

                // Allocate coo matrix
                coo_tuples = new CooTuple[num_nonzeros];
                current_edge = 0;
            }
            else
            {
                if (current_edge >= num_nonzeros)
                {
                    fprintf(stderr, "Error parsing MARKET matrix: encountered more than %d num_nonzeros\n", num_nonzeros);
                    exit(1);
                }

                int row, col;
                ValueT val;
                int nparsed = sscanf(line, "%d %d %f", &row, &col, &val);
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
            }
        }

        if (current_edge != num_nonzeros)
        {
            fprintf(stderr, "Error parsing MARKET matrix: only %d/%d num_nonzeros read\n", current_edge, num_nonzeros);
            exit(1);
        }

        // Sort by rows, then columns
        std::stable_sort(coo_tuples, coo_tuples + num_nonzeros);

        fclose(f_in);
    }


    /**
     * Builds a wheel COO sparse matrix having spokes spokes.
     */
    int InitWheel(OffsetT spokes, ValueT default_value = 1.0)
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
     * Destructor
     */
    ~CsrMatrix()
    {
        if (row_offsets)    delete[] row_offsets;
        if (column_indices) delete[] column_indices;
        if (values)         delete[] values;
    }


    /**
     * Build CSR matrix from sorted COO matrix
     */
    void FromCoo(const CooMatrix<OffsetT, ValueT> &coo_matrix)
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
        fflush(stdout);

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
        cout << "Input Matrix:\n";
        for (OffsetT row = 0; row < num_rows; row++)
        {
            cout << row << "[@" << row_offsets[row] << "]: ";
            for (OffsetT current_edge = row_offsets[row]; current_edge < row_offsets[row + 1]; current_edge++)
            {
                cout << column_indices[current_edge] << " (" << values[current_edge] << "), ";
            }
            cout << "\n";
        }
    }


};

