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

from cugraph.utilities import (
    ensure_cugraph_obj_for_nx,
    df_edge_score_to_dictionary,
    renumber_vertex_pair,
)
import cudf

from pylibcugraph.experimental import (
    overlap_coefficients as pylibcugraph_overlap_coefficients,
)
from pylibcugraph import ResourceHandle


def EXPERIMENTAL__overlap_coefficient(G, ebunch=None):
    """
    For NetworkX Compatability.  See `overlap`

    Parameters
    ----------
    graph : cugraph.Graph
        cuGraph Graph instance, should contain the connectivity information
        as an edge list (edge weights are not used for this algorithm). The
        graph should be undirected where an undirected edge is represented by a
        directed edge in both direction. The adjacency list will be computed if
        not already present.

    ebunch : cudf.DataFrame, optional (default=None)
        A GPU dataframe consisting of two columns representing pairs of
        vertices. If provided, the Overlap coefficient is computed for the
        given vertex pairs.  If the vertex_pair is not provided then the
        current implementation computes the overlap coefficient for all
        adjacent vertices in the graph.

    Returns
    -------
    df  : cudf.DataFrame
        GPU data frame of size E (the default) or the size of the given pairs
        (first, second) containing the overlap weights. The ordering is
        relative to the adjacency list, or that given by the specified vertex
        pairs.

        df['source'] : cudf.Series
            The source vertex ID (will be identical to first if specified)
        df['destination'] : cudf.Series
            The destination vertex ID (will be identical to second if
            specified)
        df['overlap_coeff'] : cudf.Series
            The computed Overlap coefficient between the source and destination
            vertices

    Examples
    --------
    >>> from cugraph.experimental.datasets import karate
    >>> G = karate.get_graph(fetch=True)
    >>> df = cugraph.overlap_coefficient(G)
    """
    vertex_pair = None

    G, isNx = ensure_cugraph_obj_for_nx(G)

    if isNx is True and ebunch is not None:
        vertex_pair = cudf.DataFrame(ebunch)

    df = EXPERIMENTAL__overlap(G, vertex_pair)

    if isNx is True:
        df = df_edge_score_to_dictionary(
            df, k="overlap_coeff", src="source", dst="destination"
        )

    return df


def EXPERIMENTAL__overlap(G, vertex_pair=None):
    """
    Compute the Overlap Coefficient between each pair of vertices connected by
    an edge, or between arbitrary pairs of vertices specified by the user.
    Overlap Coefficient is defined between two sets as the ratio of the volume
    of their intersection divided by the smaller of their two volumes. In the
    context of graphs, the neighborhood of a vertex is seen as a set. The
    Overlap Coefficient weight of each edge represents the strength of
    connection between vertices based on the relative similarity of their
    neighbors. If first is specified but second is not, or vice versa, an
    exception will be thrown.

    cugraph.overlap, in the absence of a specified vertex pair list, will
    compute the two_hop_neighbors of the entire graph to construct a vertex pair
    list and will return the overlap coefficient for those vertex pairs. This is
    not advisable as the vertex_pairs can grow exponentially with respect to the
    size of the datasets

    Parameters
    ----------
    G : cugraph.Graph
        cuGraph Graph instance, should contain the connectivity information
        as an edge list (edge weights are not used for this algorithm). The
        adjacency list will be computed if not already present.

    vertex_pair : cudf.DataFrame, optional (default=None)
        A GPU dataframe consisting of two columns representing pairs of
        vertices. If provided, the overlap coefficient is computed for the
        given vertex pairs, else, it is computed for all vertex pairs.

    Returns
    -------
    df : cudf.DataFrame
        GPU data frame of size E (the default) or the size of the given pairs
        (first, second) containing the Overlap coefficients. The ordering is
        relative to the adjacency list, or that given by the specified vertex
        pairs.

        df['source'] : cudf.Series
            The source vertex ID (will be identical to first if specified).
        df['destination'] : cudf.Series
            The destination vertex ID (will be identical to second if
            specified).
        df['overlap_coeff'] : cudf.Series
            The computed Overlap coefficient between the source and destination
            vertices.

    Examples
    --------
    >>> from cugraph.experimental.datasets import karate
    >>> G = karate.get_graph(fetch=True)
    >>> df = cugraph.overlap(G)

    """

    if G.is_directed():
        raise ValueError("Input must be an undirected Graph.")

    # FIXME: Add warning if there are weight in the PLC stating that
    # they will not be used, to called woverlap instead

    # FIXME: Update docstrings explaining that and the implication
    if vertex_pair is None:
        # Call two_hop neighbor of the entire graph
        vertex_pair = G.get_two_hop_neighbors()

    v_p_num_col = len(vertex_pair.columns)

    if isinstance(vertex_pair, cudf.DataFrame):
        vertex_pair = renumber_vertex_pair(G, vertex_pair)
        src_col_name = vertex_pair.columns[0]
        dst_col_name = vertex_pair.columns[1]
        first = vertex_pair[src_col_name]
        second = vertex_pair[dst_col_name]
    
    # FIXME: Match vertex_pair type to edgelist type and set renumber=False
    # to check

    elif vertex_pair is not None:
        raise ValueError("vertex_pair must be a cudf dataframe")

    # 'use_weight' is set to False by default for overlap and True
    # for 'woverlap'
    use_weight = False
    first, second, overlap_coeff = pylibcugraph_overlap_coefficients(
        resource_handle=ResourceHandle(),
        graph=G._plc_graph,
        first=first,
        second=second,
        use_weight=use_weight,
        do_expensive_check=False,
    )

    if G.renumbered:
        vertex_pair = G.unrenumber(vertex_pair, src_col_name, preserve_order=True)
        vertex_pair = G.unrenumber(vertex_pair, dst_col_name, preserve_order=True)

    if v_p_num_col == 2:
        # single column vertex
        vertex_pair = vertex_pair.rename(
            columns={src_col_name: "source", dst_col_name: "destination"}
        )

    df = vertex_pair
    df["overlap_coeff"] = cudf.Series(overlap_coeff)

    return df