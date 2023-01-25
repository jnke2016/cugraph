# Copyright (c) 2019-2022, NVIDIA CORPORATION.
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

from pylibcugraph import (
    betweenness_centrality as pylibcugraph_betweenness_centrality,
    ResourceHandle,
)

from cugraph.utilities import (
    ensure_cugraph_obj_for_nx,
    df_score_to_dictionary,
)
import cudf
import warnings
import numpy as np


def plc_betweenness_centrality(
    G,
    k=None,
    normalized=True,
    weight=None,
    endpoints=False,
    seed=None,
    result_dtype=np.float64,
):
    """
    Compute the betweenness centrality for all vertices of the graph G.
    Betweenness centrality is a measure of the number of shortest paths that
    pass through a vertex.  A vertex with a high betweenness centrality score
    has more paths passing through it and is therefore believed to be more
    important.

    To improve performance. rather than doing an all-pair shortest path,
    a sample of k starting vertices can be used.

    CuGraph does not currently support the 'endpoints' and 'weight' parameters
    as seen in the corresponding networkX call.

    Parameters
    ----------
    G : cuGraph.Graph or networkx.Graph
        The graph can be either directed (Graph(directed=True)) or undirected.
        Weights in the graph are ignored, the current implementation uses a parallel
        variation of the Brandes Algorithm (2001) to compute exact or approximate
        betweenness. If weights are provided in the edgelist, they will not be
        used.

    k : int or list or None, optional (default=None)
        If k is not None, use k node samples to estimate betweenness.  Higher
        values give better approximation.  If k is a list, use the content
        of the list for estimation: the list should contain vertex
        identifiers. If k is None (the default), all the vertices are used
        to estimate betweenness.  Vertices obtained through sampling or
        defined as a list will be used as sources for traversals inside the
        algorithm.

    normalized : bool, optional (default=True)
        If true, the betweenness values are normalized by
        __2 / ((n - 1) * (n - 2))__ for undirected Graphs, and
        __1 / ((n - 1) * (n - 2))__ for directed Graphs
        where n is the number of nodes in G.
        Normalization will ensure that values are in [0, 1],
        this normalization scales for the highest possible value where one
        node is crossed by every single shortest path.

    weight : cudf.DataFrame, optional (default=None)
        Specifies the weights to be used for each edge.
        Should contain a mapping between
        edges and weights.

        (Not Supported): But if weights are provided at the Graph creation,
        they will be used

    endpoints : bool, optional (default=False)
        If true, include the endpoints in the shortest path counts.

    seed : optional (default=None)
        if k is specified and k is an integer, use seed to initialize the
        random number generator.
        Using None defaults to a hash of process id, time, and hostname
        If k is either None or list: seed parameter is ignored

    result_dtype : np.float32 or np.float64, optional, default=np.float64
        Indicate the data type of the betweenness centrality scores

    Returns
    -------
    df : cudf.DataFrame or Dictionary if using NetworkX
        GPU data frame containing two cudf.Series of size V: the vertex
        identifiers and the corresponding betweenness centrality values.
        Please note that the resulting the 'vertex' column might not be
        in ascending order.  The Dictionary contains the same two columns

        df['vertex'] : cudf.Series
            Contains the vertex identifiers
        df['betweenness_centrality'] : cudf.Series
            Contains the betweenness centrality of vertices

    Examples
    --------
    >>> from cugraph.experimental.datasets import karate
    >>> G = karate.get_graph(fetch=True)
    >>> bc = cugraph.betweenness_centrality(G)

    """
    # vertex_list is intended to be a cuDF series or dataframe that contains a
    # sampling of k vertices out of the graph.

    G, isNx = ensure_cugraph_obj_for_nx(G)

    # FIXME: Should we raise an error if the graph created is weighted?
    if weight is not None:
        raise NotImplementedError(
            "weighted implementation of betweenness "
            "centrality not currently supported"
        )

    if G.store_transposed is True:
        warning_msg = (
            "Betweenness centrality expects the 'store_transposed' flag "
            "to be set to 'False' for optimal performance during "
            "the graph creation"
        )
        warnings.warn(warning_msg, UserWarning)

    # FIXME: Should we now remove this paramter?
    if result_dtype not in [np.float32, np.float64]:
        raise TypeError("result type can only be np.float32 or np.float64")
    else:
        warning_msg = (
            "This parameter is deprecated and will be remove " "in the next release."
        )
        warnings.warn(warning_msg, PendingDeprecationWarning)

    # FIXME: Do not deprecate this parameter
    if seed is not None:
        warning_msg = (
            "This parameter is deprecated and will be remove " "in the next release."
        )
        warnings.warn(warning_msg, PendingDeprecationWarning)


    if isinstance(k, list):
        k = _initialize_vertices_from_identifiers_list(G, k)
    
    vertices, values = pylibcugraph_betweenness_centrality(
        resource_handle=ResourceHandle(),
        graph=G._plc_graph,
        k=k,
        seed=seed,
        normalized=normalized,
        include_endpoints=endpoints,
        do_expensive_check=False,
    )

    vertices = cudf.Series(vertices)
    values = cudf.Series(values)

    df = cudf.DataFrame()
    df["vertex"] = vertices
    df["betweenness_centrality"] = values

    if G.renumbered:
        df = G.unrenumber(df, "vertex")

    if isNx is True:
        dict = df_score_to_dictionary(df, "betweenness_centrality")
        return dict
    else:
        return df


def _initialize_vertices_from_identifiers_list(G, identifiers):
    vertices = identifiers
    if G.renumbered:
        vertices = G.lookup_internal_vertex_id(cudf.Series(vertices))

    return vertices