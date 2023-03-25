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
)
from pylibcugraph._cugraph_c.graph cimport (
    cugraph_graph_t,
)
from pylibcugraph._cugraph_c.community_algorithms cimport (
    cugraph_clustering_result_t,
    cugraph_analyze_clustering_modularity,
    cugraph_clustering_free,
    cugraph_clustering_result_get_vertices,
    cugraph_clustering_result_get_clusters,
    cugraph_create_clustering,
    cugraph_clustering_result_free,
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
    create_cugraph_type_erased_device_array_view_from_py_obj
)


def analyze_clustering_modularity(ResourceHandle resource_handle,
                                  _GPUGraph graph,
                                  num_clusters,
                                  vertex,
                                  cluster,
                                  double score,
                                  ):
    """
    Compute the modularity score for a given partitioning/clustering.

    Parameters
    ----------
    resource_handle : ResourceHandle
        Handle to the underlying device resources needed for referencing data
        and running algorithms.

    graph : SGGraph
        The input graph.

    num_clusters : size_t
        Specifies the number of clusters to find, must be greater than 1

    num_eigen_vects : size_t
        Specifies the number of eigenvectors to use. Must be lower or equal to
        num_clusters.

    evs_tolerance: double
        Specifies the tolerance to use in the eigensolver.

    evs_max_iter: size_t
        Specifies the maximum number of iterations for the eigensolver.

    kmean_tolerance: double
        Specifies the tolerance to use in the k-means solver.

    kmean_max_iter: size_t
        Specifies the maximum number of iterations for the k-means solver.
    
    do_expensive_check : bool_t
        If True, performs more extensive tests on the inputs to ensure
        validitity, at the expense of increased run time.

    Returns
    -------
    A tuple containing the clustering vertices, clusters

    Examples
    --------
    >>> import pylibcugraph, cupy, numpy
    >>> srcs = cupy.asarray([0, 1, 2], dtype=numpy.int32)
    >>> dsts = cupy.asarray([1, 2, 0], dtype=numpy.int32)
    >>> weights = cupy.asarray([1.0, 1.0, 1.0], dtype=numpy.float32)
    >>> resource_handle = pylibcugraph.ResourceHandle()
    >>> graph_props = pylibcugraph.GraphProperties(
    ...     is_symmetric=True, is_multigraph=False)
    >>> G = pylibcugraph.SGGraph(
    ...     resource_handle, graph_props, srcs, dsts, weights,
    ...     store_transposed=True, renumber=False, do_expensive_check=False)
    >>> (vertices, clusters) = pylibcugraph.analyze_clustering_modularity(
    ...     resource_handle, G, num_clusters=5, num_eigen_vects=2, evs_tolerance=0.00001
    ...     evs_max_iter=100, kmean_tolerance=0.00001, kmean_max_iter=100)
    # FIXME: Fix dockstring result.
    >>> vertices
    ############
    >>> clusters
    ############

    """

    cdef cugraph_resource_handle_t* c_resource_handle_ptr = \
        resource_handle.c_resource_handle_ptr
    cdef cugraph_graph_t* c_graph_ptr = graph.c_graph_ptr
    cdef cugraph_clustering_result_t* result_ptr
    cdef cugraph_error_code_t error_code
    cdef cugraph_error_t* error_ptr

    cdef cugraph_clustering_result_t* clustering_ptr

    #cdef double* score_ptr

    # 'first' is a required parameter
    cdef cugraph_type_erased_device_array_view_t* \
        vertex_view_ptr = \
            create_cugraph_type_erased_device_array_view_from_py_obj(
                vertex)

    # 'second' is a required parameter
    cdef cugraph_type_erased_device_array_view_t* \
        cluster_view_ptr = \
            create_cugraph_type_erased_device_array_view_from_py_obj(
                cluster)

    error_code = cugraph_create_clustering(c_resource_handle_ptr,
                                             c_graph_ptr,
                                             vertex_view_ptr,
                                             cluster_view_ptr,
                                             &clustering_ptr,
                                             &error_ptr)
    assert_success(error_code, error_ptr, "clustering")

    error_code = cugraph_analyze_clustering_modularity(c_resource_handle_ptr,
                                                       c_graph_ptr,
                                                       num_clusters,
                                                       clustering_ptr,
                                                       &score,
                                                       &error_ptr)
    assert_success(error_code, error_ptr, "cugraph_analyze_clustering_modularity")

    # Extract individual device array pointers from result and copy to cupy
    # arrays for returning.
    """
    cdef cugraph_type_erased_device_array_view_t* vertices_ptr = \
        cugraph_clustering_result_get_vertices(result_ptr)
    cdef cugraph_type_erased_device_array_view_t* clusters_ptr = \
        cugraph_clustering_result_get_clusters(result_ptr)

    cupy_vertices = copy_to_cupy_array(c_resource_handle_ptr, vertices_ptr)
    cupy_clusters = copy_to_cupy_array(c_resource_handle_ptr, clusters_ptr)
    """

    # cugraph_clustering_result_free(result_ptr)
    cugraph_clustering_free(clustering_ptr)

    return score
