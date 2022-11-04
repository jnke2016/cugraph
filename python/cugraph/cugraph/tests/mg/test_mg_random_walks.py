# Copyright (c) 2022, NVIDIA CORPORATION.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import cugraph.dask as dcg
import gc
import pytest
import cugraph
import dask_cudf
import random

# from cugraph.dask.common.mg_utils import is_single_gpu
from cugraph.testing import utils

from cugraph.experimental.datasets import DATASETS_SMALL, karate_asymmetric


# =============================================================================
# Pytest Setup / Teardown - called for each test function
# =============================================================================


def setup_function():
    gc.collect()


IS_DIRECTED = [True]


# =============================================================================
# Pytest fixtures
# =============================================================================

datasets = DATASETS_SMALL + [karate_asymmetric]
#datasets = [karate_asymmetric]

fixture_params = utils.genFixtureParamsProduct(
    (datasets, "graph_file"),
    (IS_DIRECTED, "directed"),
)


def calc_random_walks(G):
    """
    compute random walks

    parameters
    ----------
    G : cuGraph.Graph or networkx.Graph
        The graph can be either directed (DiGraph) or undirected (Graph).
        Weights in the graph are ignored.
        Use weight parameter if weights need to be considered
        (currently not supported)

    Returns
    -------
    vertex_paths : cudf.Series or cudf.DataFrame
        Series containing the vertices of edges/paths in the random walk.

    edge_weight_paths: cudf.Series
        Series containing the edge weights of edges represented by the
        returned vertex_paths

    max_path_length : int
        The maximum path length
    
    start_vertices : list
        Roots for the random walks
    
    max_depth : int
    """
    k = random.randint(1, 4)
    random_walks_type = "uniform"
    max_depth = random.randint(2, 4)
    start_vertices = G.nodes().compute().sample(k).reset_index(drop=True)

    vertex_paths, edge_weights, max_path_length = dcg.random_walks(
        G, random_walks_type, start_vertices, max_depth
    )

    return (vertex_paths, edge_weights, max_path_length), start_vertices, max_depth


def check_random_walks(G, path_data, seeds, max_depth, df_G=None):
    invalid_edge = 0
    invalid_seeds = 0
    offsets_idx = 0
    next_path_idx = 0
    v_paths = path_data[0].compute()

    max_path_length = path_data[2]
    sizes = max_path_length
    
    for _ in range(len(seeds)):
        for i in range(next_path_idx, next_path_idx + sizes - 1):
            src, dst = v_paths.iloc[i], v_paths.iloc[i + 1]

            if i == next_path_idx and src not in seeds.values:
                invalid_seeds += 1
                print(
                    "[ERR] Invalid seed: "
                    " src {} != src {}".format(src, seeds)
                )

            else:
                # If everything is good proceed to the next part
                # now check the destination
         
                # find the src out_degree to ensure it effectively has no outgoing edges
                # No need to check for -1 values, move to the next iteration
                if src != -1:
                    src_degree = G.out_degree([src])["degree"].compute()[0]
                    if dst == -1  and src_degree == 0:
                        # No need to check the next element as 'dst' will become 'src'
                        i += 1
                    else:
                        exp_edge = df_G.loc[(df_G["src"] == (src)) & (df_G["dst"] == (dst))].\
                            reset_index(drop=True)

                        if len(exp_edge) == 0:
                            print(
                                "[ERR] Invalid edge: " "There is no edge src {} dst {}".\
                                    format(src, dst)
                            )
                            invalid_edge += 1

        offsets_idx += 1
        next_path_idx += sizes + 1

    assert invalid_edge == 0
    assert invalid_seeds == 0
    assert max_path_length == max_depth


@pytest.fixture(scope="module", params=fixture_params)
def input_graph(request):
    """
    Simply return the current combination of params as a dictionary for use in
    tests or other parameterized fixtures.
    """
    parameters = dict(zip(("graph_file", "directed"), request.param))
    input_data_path = parameters["graph_file"].get_path()
    directed = parameters["directed"]

    chunksize = dcg.get_chunksize(input_data_path)
    ddf = dask_cudf.read_csv(
        input_data_path,
        chunksize=chunksize,
        delimiter=" ",
        names=["src", "dst", "value"],
        dtype=["int32", "int32", "float32"],
    )
    dg = cugraph.Graph(directed=directed)
    dg.from_dask_cudf_edgelist(
        ddf,
        source="src",
        destination="dst",
        edge_attr="value",
        renumber=True,
        legacy_renum_only=True,
        store_transposed=True,
    )

    return dg


def test_dask_random_walks(dask_client, benchmark, input_graph):
    path_data, seeds, max_depth = calc_random_walks(input_graph)
    df_G = input_graph.input_df.compute().reset_index(drop=True)
    check_random_walks(input_graph, path_data, seeds, max_depth, df_G)


