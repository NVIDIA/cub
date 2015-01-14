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
 * COO graph data structure
 ******************************************************************************/

#pragma once

#include <iterator>
#include <vector>
#include <string>
#include <algorithm>
#include <iostream>
#include <stdio.h>

#include <cub/cub.cuh>

using namespace std;


/**
 * COO graph type.  A COO graph is just a vector of edge tuples.
 */
template<typename VertexId, typename Value>
struct CooGraph
{
    /**
     * COO edge tuple.  (A COO graph is just a vector of these.)
     */
    struct CooTuple
    {
        VertexId            row;
        VertexId            col;
        Value               val;

        CooTuple() {}
        CooTuple(VertexId row, VertexId col) : row(row), col(col) {}
        CooTuple(VertexId row, VertexId col, Value val) : row(row), col(col), val(val) {}
    };

    /**
     * Comparator for sorting COO sparse format edges
     */
    static bool CooTupleCompare (const CooTuple &elem1, const CooTuple &elem2)
    {
        if (elem1.row < elem2.row)
        {
            return true;
        }
        else if ((elem1.row == elem2.row) && (elem1.col < elem2.col))
        {
            return true;
        }

        return false;
    }


    /**
     * Fields
     */
    int                 row_dim;        // Num rows
    int                 col_dim;        // Num cols
    vector<CooTuple>    coo_tuples;     // Non-zero entries


    /**
     * CooGraph ostream operator
     */
    friend std::ostream& operator<<(std::ostream& os, const CooGraph& coo_graph)
    {
        os << "Sparse COO (" << coo_graph.row_dim << " rows, " << coo_graph.col_dim << " cols, " << coo_graph.coo_tuples.size() << " nonzeros):\n";
        os << "Ordinal, Row, Col, Val\n";
        for (int i = 0; i < coo_graph.coo_tuples.size(); i++)
        {
            os << i << ',' << coo_graph.coo_tuples[i].row << ',' << coo_graph.coo_tuples[i].col << ',' << coo_graph.coo_tuples[i].val << "\n";
        }
        return os;
    }

    /**
     * Update graph dims based upon COO tuples
     */
    void UpdateDims()
    {
        row_dim = -1;
        col_dim = -1;

        for (int i = 0; i < coo_tuples.size(); i++)
        {
            row_dim = CUB_MAX(row_dim, coo_tuples[i].row);
            col_dim = CUB_MAX(col_dim, coo_tuples[i].col);
        }

        row_dim++;
        col_dim++;
    }


    /**
     * Builds a METIS COO sparse from the given file.
     */
    int InitMetis(const string &metis_filename)
    {
        coo_tuples.clear();


        return 0;
    }


    /**
     * Builds a MARKET COO sparse from the given file.
     */
    int InitMarket(const string &market_filename)
    {
        coo_tuples.clear();

        // Read from file
        FILE *f_in = fopen(market_filename.c_str(), "r");
        if (!f_in) return -1;

        int edges_read = -1;
        int edges = 0;

        char line[1024];

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
            else if (edges_read == -1)
            {
                // Problem description
                long long ll_nodes_x, ll_nodes_y, ll_edges;
                if (sscanf(line, "%lld %lld %lld", &ll_nodes_x, &ll_nodes_y, &ll_edges) != 3)
                {
                    fprintf(stderr, "Error parsing MARKET graph: invalid problem description\n");
                    return -1;
                }

                edges = ll_edges;

                printf(" (%lld nodes, %lld directed edges)... ",
                    (unsigned long long) ll_nodes_x,
                    (unsigned long long) ll_edges);
                fflush(stdout);

                // Allocate coo graph
                coo_tuples.reserve(edges);
                edges_read++;
            }
            else
            {
                if (edges_read >= edges)
                {
                    fprintf(stderr, "Error parsing MARKET graph: encountered more than %d edges\n", edges);
                    fclose(f_in);
                    return -1;
                }

                long long ll_row, ll_col;
                Value val;
                int nparsed = sscanf(line, "%lld %lld %f", &ll_col, &ll_row, &val);
                if (nparsed == 2)
                {
                    // No value
                    val = 1.0;
                }
                else if (nparsed != 3)
                {
                    fprintf(stderr, "Error parsing MARKET graph: badly formed edge\n", edges);
                    fclose(f_in);
                    return -1;
                }

                ll_row -= 1;
                ll_col -= 1;

                coo_tuples.push_back(CooTuple(ll_row, ll_col, val));    // zero-based array
                edges_read++;
            }
        }

        if (edges_read != edges)
        {
            fprintf(stderr, "Error parsing MARKET graph: only %d/%d edges read\n", edges_read, edges);
            fclose(f_in);
            return -1;
        }

        // Sort by rows, then columns, update dims
        std::stable_sort(coo_tuples.begin(), coo_tuples.end(), CooTupleCompare);
        UpdateDims();

        fclose(f_in);

        return 0;
    }


    /**
     * Builds a wheel COO sparse graph having spokes spokes.
     */
    int InitWheel(VertexId spokes)
    {
        VertexId edges  = spokes + (spokes - 1);

        coo_tuples.clear();
        coo_tuples.reserve(edges);

        // Add spoke edges
        for (VertexId i = 0; i < spokes; i++)
        {
            coo_tuples.push_back(CooTuple(0, i + 1));
        }

        // Add rim
        for (VertexId i = 0; i < spokes; i++)
        {
            VertexId dest = (i + 1) % spokes;
            coo_tuples.push_back(CooTuple(i + 1, dest + 1));
        }

        // Sort by rows, then columns, update dims
        std::stable_sort(coo_tuples.begin(), coo_tuples.end(), CooTupleCompare);
        UpdateDims();

        // Assign arbitrary values to graph vertices
        for (int i = 0; i < coo_tuples.size(); i++)
        {
            coo_tuples[i].val = i % 21;
        }

        return 0;
    }


    /**
     * Builds a square 3D grid COO sparse graph.  Interior nodes have degree 7 when including
     * a self-loop.  Values are unintialized, coo_tuples are sorted.
     */
    int InitGrid3d(VertexId width, bool self_loop)
    {
        VertexId interior_nodes        = (width - 2) * (width - 2) * (width - 2);
        VertexId face_nodes            = (width - 2) * (width - 2) * 6;
        VertexId edge_nodes            = (width - 2) * 12;
        VertexId corner_nodes          = 8;
        VertexId nodes                 = width * width * width;
        VertexId edges                 = (interior_nodes * 6) + (face_nodes * 5) + (edge_nodes * 4) + (corner_nodes * 3);

        if (self_loop) edges += nodes;

        coo_tuples.clear();
        coo_tuples.reserve(edges);

        for (VertexId i = 0; i < width; i++) {
            for (VertexId j = 0; j < width; j++) {
                for (VertexId k = 0; k < width; k++) {

                    VertexId me = (i * width * width) + (j * width) + k;

                    VertexId neighbor = (i * width * width) + (j * width) + (k - 1);
                    if (k - 1 >= 0) {
                        coo_tuples.push_back(CooTuple(me, neighbor));
                    }

                    neighbor = (i * width * width) + (j * width) + (k + 1);
                    if (k + 1 < width) {
                        coo_tuples.push_back(CooTuple(me, neighbor));
                    }

                    neighbor = (i * width * width) + ((j - 1) * width) + k;
                    if (j - 1 >= 0) {
                        coo_tuples.push_back(CooTuple(me, neighbor));
                    }

                    neighbor = (i * width * width) + ((j + 1) * width) + k;
                    if (j + 1 < width) {
                        coo_tuples.push_back(CooTuple(me, neighbor));
                    }

                    neighbor = ((i - 1) * width * width) + (j * width) + k;
                    if (i - 1 >= 0) {
                        coo_tuples.push_back(CooTuple(me, neighbor));
                    }

                    neighbor = ((i + 1) * width * width) + (j * width) + k;
                    if (i + 1 < width) {
                        coo_tuples.push_back(CooTuple(me, neighbor));
                    }

                    if (self_loop)
                    {
                        neighbor = me;
                        coo_tuples.push_back(CooTuple(me, neighbor));
                    }
                }
            }
        }

        // Sort by rows, then columns, update dims
        std::stable_sort(coo_tuples.begin(), coo_tuples.end(), CooTupleCompare);
        UpdateDims();

        // Assign arbitrary values to graph vertices
        for (int i = 0; i < coo_tuples.size(); i++)
        {
            coo_tuples[i].val = i % 21;
        }

        return 0;
    }


    /**
     * Builds a square 2D grid CSR graph.  Interior nodes have degree 5 when including
     * a self-loop.
     *
     * Returns 0 on success, 1 on failure.
     */
    int InitGrid2d(VertexId width, bool self_loop)
    {
        VertexId interior_nodes         = (width - 2) * (width - 2);
        VertexId edge_nodes             = (width - 2) * 4;
        VertexId corner_nodes           = 4;
        VertexId nodes                  = width * width;
        VertexId edges                  = (interior_nodes * 4) + (edge_nodes * 3) + (corner_nodes * 2);

        if (self_loop) edges += nodes;

        coo_tuples.clear();
        coo_tuples.reserve(edges);

        for (VertexId j = 0; j < width; j++) {
            for (VertexId k = 0; k < width; k++) {

                VertexId me = (j * width) + k;

                VertexId neighbor = (j * width) + (k - 1);
                if (k - 1 >= 0) {
                    coo_tuples.push_back(CooTuple(me, neighbor));
                }

                neighbor = (j * width) + (k + 1);
                if (k + 1 < width) {
                    coo_tuples.push_back(CooTuple(me, neighbor));
                }

                neighbor = ((j - 1) * width) + k;
                if (j - 1 >= 0) {
                    coo_tuples.push_back(CooTuple(me, neighbor));
                }

                neighbor = ((j + 1) * width) + k;
                if (j + 1 < width) {
                    coo_tuples.push_back(CooTuple(me, neighbor));
                }

                if (self_loop)
                {
                    neighbor = me;
                    coo_tuples.push_back(CooTuple(me, neighbor));
                }
            }
        }

        // Sort by rows, then columns, update dims
        std::stable_sort(coo_tuples.begin(), coo_tuples.end(), CooTupleCompare);
        UpdateDims();

        // Assign arbitrary values to graph vertices
        for (int i = 0; i < coo_tuples.size(); i++)
        {
            coo_tuples[i].val = i % 21;
        }

        return 0;
    }
};


