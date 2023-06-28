# Copyright (c) 2023, NVIDIA CORPORATION.
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

# Have cython use python 3 syntax
# cython: language_level = 3


from pylibcugraph._cugraph_c.resource_handle cimport (
    bool_t,
    cugraph_resource_handle_t,
)
from pylibcugraph._cugraph_c.error cimport (
    cugraph_error_code_t,
    cugraph_error_t,
)
from pylibcugraph._cugraph_c.array cimport (
    cugraph_type_erased_device_array_view_t,
    cugraph_type_erased_device_array_view_free,
)
from pylibcugraph._cugraph_c.graph cimport (
    cugraph_graph_t,
)
from pylibcugraph._cugraph_c.centrality_algorithms cimport (
    cugraph_edge_centrality_result_t,
    cugraph_edge_betweenness_centrality,
    cugraph_edge_centrality_result_get_src_vertices,
    cugraph_edge_centrality_result_get_dst_vertices,
    cugraph_edge_centrality_result_get_values,
    cugraph_edge_centrality_result_get_edge_ids,
    cugraph_edge_centrality_result_get_values,
    cugraph_edge_centrality_result_free,
)
from pylibcugraph.resource_handle cimport (
    ResourceHandle,
)
from pylibcugraph.graphs cimport (
    _GPUGraph,
)
from pylibcugraph.utils cimport (
    assert_success,
    copy_to_cupy_array,
    create_cugraph_type_erased_device_array_view_from_py_obj,
)
from pylibcugraph.select_random_vertices import (
    select_random_vertices
)


def edge_betweenness_centrality(ResourceHandle resource_handle,
                                _GPUGraph graph,
                                k,
                                random_state,
                                bool_t normalized,
                                bool_t do_expensive_check):
    """
    Compute the edge betweenness centrality for all edges of the graph G.
    Betweenness centrality is a measure of the number of shortest paths
    that pass over an edge.  An edge with a high betweenness centrality
    score has more paths passing over it and is therefore believed to be
    more important.


    Parameters
    ----------
    resource_handle : ResourceHandle
        Handle to the underlying device resources needed for referencing data
        and running algorithms.

    graph : SGGraph or MGGraph
        The input graph, for either Single or Multi-GPU operations.

    k : int or device array type or None, optional (default=None)
        If k is not None, use k node samples to estimate the edge betweenness.
        Higher values give better approximation.  If k is a device array type,
        use the content of the list for estimation: the list should contain
        vertex identifiers. If k is None (the default), all the vertices are
        used to estimate the edge betweenness.  Vertices obtained through sampling or
        defined as a list will be used as sources for traversals inside the
        algorithm.

    random_state : int, optional (default=None)
        if k is specified and k is an integer, use random_state to initialize the
        random number generator.
        Using None defaults to a hash of process id, time, and hostname
        If k is either None or list or cudf objects: random_state parameter is
        ignored.
    
    normalized : bool_t
        Normalization will ensure that values are in [0, 1].

    do_expensive_check : bool_t
        A flag to run expensive checks for input arguments if True.
    
    Returns
    -------

    Examples
    --------

    """

    if isinstance(k, int):
        # randomly select vertices
        
        #'select_random_vertices' internally creates a
        # 'pylibcugraph.random.CuGraphRandomState'
        vertex_list = select_random_vertices(
            resource_handle, graph, random_state, k)
    else:
        # FiXME: Add CAPI check ensuring that k is a cuda array interface
        vertex_list = k

    cdef cugraph_resource_handle_t* c_resource_handle_ptr = \
        resource_handle.c_resource_handle_ptr
    cdef cugraph_graph_t* c_graph_ptr = graph.c_graph_ptr

    cdef cugraph_edge_centrality_result_t* result_ptr
    cdef cugraph_error_code_t error_code
    cdef cugraph_error_t* error_ptr

    cdef cugraph_type_erased_device_array_view_t* \
        vertex_list_view_ptr = \
            create_cugraph_type_erased_device_array_view_from_py_obj(
                vertex_list)

    error_code = cugraph_edge_betweenness_centrality(c_resource_handle_ptr,
                                                c_graph_ptr,
                                                vertex_list_view_ptr,
                                                normalized,
                                                do_expensive_check,
                                                &result_ptr,
                                                &error_ptr)
    assert_success(error_code, error_ptr, "cugraph_edge_betweenness_centrality")

    # Extract individual device array pointers from result and copy to cupy
    # arrays for returning.
    cdef cugraph_type_erased_device_array_view_t* src_ptr = \
        cugraph_edge_centrality_result_get_src_vertices(result_ptr)
    cdef cugraph_type_erased_device_array_view_t* dst_ptr = \
        cugraph_edge_centrality_result_get_dst_vertices(result_ptr)
    cdef cugraph_type_erased_device_array_view_t* edge_ids_ptr = \
        cugraph_edge_centrality_result_get_values(result_ptr)
    cdef cugraph_type_erased_device_array_view_t* values_ptr = \
        cugraph_edge_centrality_result_get_values(result_ptr)
    
    cupy_src = copy_to_cupy_array(c_resource_handle_ptr, src_ptr)
    cupy_dst = copy_to_cupy_array(c_resource_handle_ptr, dst_ptr)
    cupy_edge_ids = copy_to_cupy_array(c_resource_handle_ptr, edge_ids_ptr)
    cupy_values = copy_to_cupy_array(c_resource_handle_ptr, values_ptr)

    cugraph_edge_centrality_result_free(result_ptr)
    cugraph_type_erased_device_array_view_free(vertex_list_view_ptr)

    return (cupy_src, cupy_dst, cupy_edge_ids, cupy_values)
