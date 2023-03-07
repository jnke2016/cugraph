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
import cupy


# from cugraph.dask.common.mg_utils import is_single_gpu
from cugraph.testing import utils
from pylibcugraph.testing import gen_fixture_params_product


# =============================================================================
# Pytest Setup / Teardown - called for each test function
# =============================================================================


def setup_function():
    gc.collect()

IS_DIRECTED = [True, False]


# =============================================================================
# Pytest fixtures
# =============================================================================

datasets = utils.DATASETS_UNDIRECTED + [
    utils.RAPIDS_DATASET_ROOT_DIR_PATH / "email-Eu-core.csv"
]

fixture_params = gen_fixture_params_product(
    (datasets, "graph_file"),
    ([False, True], "normalized"),
    ([False, True], "endpoints"),
    ([42], "subset_seed"),
    ([None, 15], "subset_size"),
    (IS_DIRECTED, "directed"),
)


@pytest.fixture(scope="module", params=fixture_params)
def input_combo(request):
    """
    Simply return the current combination of params as a dictionary for use in
    tests or other parameterized fixtures.
    """
    parameters = dict(
        zip(
            ("graph_file", "normalized", "endpoints", "subset_seed", "subset_size", "directed"),
            request.param,
        )
    )

    return parameters


@pytest.fixture(scope="module")
def input_expected_output(input_combo):
    """
    This fixture returns the inputs and expected results from the
    betweenness_centrality algo based on cuGraph betweenness_centrality) which can
    be used for validation.
    """

    input_data_path = input_combo["graph_file"]
    normalized = input_combo["normalized"]
    endpoints = input_combo["endpoints"]
    random_state = input_combo["subset_seed"]
    subset_size = input_combo["subset_size"]
    directed = input_combo["directed"]

    G = utils.generate_cugraph_graph_from_file(input_data_path, directed=directed)

    if subset_size is None:
        k = subset_size
    elif isinstance(subset_size, int):
        # Select random vertices
        k = G.select_random_vertices(random_state=random_state, num_vertices=subset_size)
        print("the seeds are \n", k)
    
    input_combo["k"] = k

    sg_cugraph_bc = cugraph.plc_betweenness_centrality(
        G, k=k, normalized=normalized, endpoints=endpoints
    )
    # Save the results back to the input_combo dictionary to prevent redundant
    # cuGraph runs. Other tests using the input_combo fixture will look for
    # them, and if not present they will have to re-run the same cuGraph call.
    sg_cugraph_bc = sg_cugraph_bc.sort_values("vertex").reset_index(drop=True)

    input_combo["sg_cugraph_results"] = sg_cugraph_bc
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

    input_combo["MGGraph"] = dg

    return input_combo


# =============================================================================
# Tests
# =============================================================================


# @pytest.mark.skipif(
#    is_single_gpu(), reason="skipping MG testing on Single GPU system"
# )

def test_dask_betweenness_centrality(dask_client, benchmark, input_expected_output):

    dg = input_expected_output["MGGraph"]
    k = input_expected_output["k"]
    endpoints = input_expected_output["endpoints"]
    normalized = input_expected_output["normalized"]
    mg_bc_results = benchmark(
        dcg.betweenness_centrality,
        dg,
        k=k,
        normalized=normalized,
        endpoints=endpoints,
    )

    mg_bc_results = (
        mg_bc_results.compute()
        .sort_values("vertex")
        .reset_index(drop=True)
    )["betweenness_centrality"].to_cupy()
    
    sg_bc_results = (
        input_expected_output["sg_cugraph_results"]
        .sort_values("vertex")
        .reset_index(drop=True)
    )["betweenness_centrality"].to_cupy()

    diff = cupy.isclose(mg_bc_results, sg_bc_results)

    assert diff.all() == True