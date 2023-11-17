# Copyright (c) 2022-2023, NVIDIA CORPORATION.
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

from pylibcugraph._cugraph_c.error cimport (
    cugraph_error_code_t,
    cugraph_error_t,
)
from cython.operator cimport dereference as deref
from pylibcugraph._cugraph_c.array cimport (
    cugraph_type_erased_device_array_view_t,
    cugraph_type_erased_device_array_view_free,
)
from pylibcugraph._cugraph_c.graph cimport (
    cugraph_sg_graph_create,
    cugraph_graph_create_sg,
    cugraph_mg_graph_create,
    cugraph_graph_create_mg,
    cugraph_sg_graph_create_from_csr, #FIXME: Remove this once
    # 'cugraph_graph_create_sg_from_csr' is exposed
    cugraph_graph_create_sg_from_csr,
    cugraph_sg_graph_free, #FIXME: Remove this
    cugraph_graph_free,
    cugraph_mg_graph_free, #FIXME: Remove this
)
from pylibcugraph.resource_handle cimport (
    ResourceHandle,
)
from pylibcugraph.graph_properties cimport (
    GraphProperties,
)
from pylibcugraph.utils cimport (
    assert_success,
    assert_CAI_type,
    create_cugraph_type_erased_device_array_view_from_py_obj,
)


cdef class SGGraph(_GPUGraph):
    """
    RAII-stye Graph class for use with single-GPU APIs that manages the
    individual create/free calls and the corresponding cugraph_graph_t pointer.

    Parameters
    ----------
    resource_handle : ResourceHandle
        Handle to the underlying device resources needed for referencing data
        and running algorithms.

    graph_properties : GraphProperties
        Object defining intended properties for the graph.

    src_or_offset_array : device array type
        Device array containing either the vertex identifiers of the source of 
        each directed edge if represented in COO format or the offset if
        CSR format. In the case of a COO, the order of the array corresponds to
        the ordering of the dst_or_index_array, where the ith item in
        src_offset_array and the ith item in dst_index_array define the ith edge
        of the graph.

    dst_or_index_array : device array type
        Device array containing the vertex identifiers of the destination of
        each directed edge if represented in COO format or the index if
        CSR format. In the case of a COO, The order of the array corresponds to
        the ordering of the src_offset_array, where the ith item in src_offset_array
        and the ith item in dst_index_array define the ith edge of the graph.
    
    vertices_array : device array type
        Device array containing the isolated vertices of the graph.

    weight_array : device array type
        Device array containing the weight values of each directed edge. The
        order of the array corresponds to the ordering of the src_array and
        dst_array arrays, where the ith item in weight_array is the weight value
        of the ith edge of the graph.

    store_transposed : bool
        Set to True if the graph should be transposed. This is required for some
        algorithms, such as pagerank.

    renumber : bool
        Set to True to indicate the vertices used in src_array and dst_array are
        not appropriate for use as internal array indices, and should be mapped
        to continuous integers starting from 0.

    do_expensive_check : bool
        If True, performs more extensive tests on the inputs to ensure
        validitity, at the expense of increased run time.
    
    edge_id_array : device array type
        Device array containing the edge ids of each directed edge.  Must match
        the ordering of the src/dst arrays.  Optional (may be null).  If
        provided, edge_type_array must also be provided.
    
    edge_type_array : device array type
        Device array containing the edge types of each directed edge.  Must
        match the ordering of the src/dst/edge_id arrays.  Optional (may be
        null).  If provided, edge_id_array must be provided.
    
    input_array_format: str, optional (default='COO')
        Input representation used to construct a graph
            COO: arrays represent src_array and dst_array
            CSR: arrays represent offset_array and index_array
    
    drop_self_loops : bool, optional (default='False')
        If true, drop any self loops that exist in the provided edge list.
    
    drop_multi_edges: bool, optional (default='False')
        If true, drop any multi edges that exist in the provided edge list

    Examples
    ---------
    >>> import pylibcugraph, cupy, numpy
    >>> srcs = cupy.asarray([0, 1, 2], dtype=numpy.int32)
    >>> dsts = cupy.asarray([1, 2, 3], dtype=numpy.int32)
    >>> seeds = cupy.asarray([0, 0, 1], dtype=numpy.int32)
    >>> weights = cupy.asarray([1.0, 1.0, 1.0], dtype=numpy.float32)
    >>> resource_handle = pylibcugraph.ResourceHandle()
    >>> graph_props = pylibcugraph.GraphProperties(
    ...     is_symmetric=False, is_multigraph=False)
    >>> G = pylibcugraph.SGGraph(
    ...     resource_handle, graph_props, srcs, dsts, weights,
    ...     store_transposed=False, renumber=False, do_expensive_check=False)

    """
    def __cinit__(self,
                  ResourceHandle resource_handle,
                  GraphProperties graph_properties,
                  src_or_offset_array,
                  dst_or_index_array,
                  vertices_array=None,
                  weight_array=None,
                  store_transposed=False,
                  renumber=False,
                  do_expensive_check=False,
                  edge_id_array=None,
                  edge_type_array=None,
                  input_array_format="COO",
                  drop_self_loops=False,
                  drop_multi_edges=False):
        
        print("entering", flush=True)
        # FIXME: add tests for these
        if not(isinstance(store_transposed, (int, bool))):
            raise TypeError("expected int or bool for store_transposed, got "
                            f"{type(store_transposed)}")
        if not(isinstance(renumber, (int, bool))):
            raise TypeError("expected int or bool for renumber, got "
                            f"{type(renumber)}")
        if not(isinstance(do_expensive_check, (int, bool))):
            raise TypeError("expected int or bool for do_expensive_check, got "
                            f"{type(do_expensive_check)}")
        assert_CAI_type(src_or_offset_array, "src_or_offset_array")
        assert_CAI_type(dst_or_index_array, "dst_or_index_array")
        assert_CAI_type(vertices_array, "vertices_array", True)
        assert_CAI_type(weight_array, "weight_array", True)
        assert_CAI_type(edge_id_array, "edge_id_array", True)
        assert_CAI_type(edge_type_array, "edge_type_array", True)

        # FIXME: assert that src_or_offset_array and dst_or_index_array have the same type

        cdef cugraph_error_t* error_ptr
        cdef cugraph_error_code_t error_code

        cdef cugraph_type_erased_device_array_view_t* srcs_or_offsets_view_ptr = \
            create_cugraph_type_erased_device_array_view_from_py_obj(
                src_or_offset_array
            )
        
        cdef cugraph_type_erased_device_array_view_t* dsts_or_indices_view_ptr = \
            create_cugraph_type_erased_device_array_view_from_py_obj(
                dst_or_index_array
            )

        cdef cugraph_type_erased_device_array_view_t* vertices_view_ptr = \
            create_cugraph_type_erased_device_array_view_from_py_obj(
                vertices_array
            )

        self.weights_view_ptr = create_cugraph_type_erased_device_array_view_from_py_obj(
                weight_array
            )

        self.edge_id_view_ptr = create_cugraph_type_erased_device_array_view_from_py_obj(
                edge_id_array
            )

        
        cdef cugraph_type_erased_device_array_view_t* edge_type_view_ptr = \
            create_cugraph_type_erased_device_array_view_from_py_obj(
                edge_type_array
            )

        if input_array_format == "COO":
            print("calling SG graph create", flush=True)
            error_code = cugraph_graph_create_sg(
                resource_handle.c_resource_handle_ptr,
                &(graph_properties.c_graph_properties),
                vertices_view_ptr,
                srcs_or_offsets_view_ptr,
                dsts_or_indices_view_ptr,
                self.weights_view_ptr,
                self.edge_id_view_ptr,
                edge_type_view_ptr,
                store_transposed,
                renumber,
                drop_self_loops,
                drop_multi_edges,
                do_expensive_check,
                &(self.c_graph_ptr),
                &error_ptr)
            assert_success(error_code, error_ptr,
                       "cugraph_graph_create_sg()")
            print("Done calling SG graph create", flush=True)

        elif input_array_format == "CSR":
            error_code = cugraph_sg_graph_create_from_csr(
                resource_handle.c_resource_handle_ptr,
                &(graph_properties.c_graph_properties),
                srcs_or_offsets_view_ptr,
                dsts_or_indices_view_ptr,
                self.weights_view_ptr,
                self.edge_id_view_ptr,
                edge_type_view_ptr,
                store_transposed,
                renumber,
                # drop_self_loops, #FIXME: Not supported yet
                # drop_multi_edges, #FIXME: Not supported yet
                do_expensive_check,
                &(self.c_graph_ptr),
                &error_ptr)

            assert_success(error_code, error_ptr,
                       "cugraph_sg_graph_create_from_csr()")
        
        else:
            raise ValueError("invalid 'input_array_format'. Only "
                "'COO' and 'CSR' format are supported."
            )

        cugraph_type_erased_device_array_view_free(srcs_or_offsets_view_ptr)
        cugraph_type_erased_device_array_view_free(dsts_or_indices_view_ptr)
        #cugraph_type_erased_device_array_view_free(self.weights_view_ptr)
        if self.weights_view_ptr is not NULL:
            cugraph_type_erased_device_array_view_free(self.weights_view_ptr)
        if self.edge_id_view_ptr is not NULL:
            cugraph_type_erased_device_array_view_free(self.edge_id_view_ptr)
        if edge_type_view_ptr is not NULL:
            cugraph_type_erased_device_array_view_free(edge_type_view_ptr)

    def __dealloc__(self):
        if self.c_graph_ptr is not NULL:
            cugraph_sg_graph_free(self.c_graph_ptr)


cdef class MGGraph(_GPUGraph):
    """
    RAII-stye Graph class for use with multi-GPU APIs that manages the
    individual create/free calls and the corresponding cugraph_graph_t pointer.

    Parameters
    ----------
    resource_handle : ResourceHandle
        Handle to the underlying device resources needed for referencing data
        and running algorithms.

    graph_properties : GraphProperties
        Object defining intended properties for the graph.

    src_array : device array type
        Device array containing the vertex identifiers of the source of each
        directed edge. The order of the array corresponds to the ordering of the
        dst_array, where the ith item in src_array and the ith item in dst_array
        define the ith edge of the graph.

    dst_array : device array type
        Device array containing the vertex identifiers of the destination of
        each directed edge. The order of the array corresponds to the ordering
        of the src_array, where the ith item in src_array and the ith item in
        dst_array define the ith edge of the graph.
    
    vertices_array : device array type
        Device array containing the isolated vertices of the graph.

    weight_array : device array type
        Device array containing the weight values of each directed edge. The
        order of the array corresponds to the ordering of the src_array and
        dst_array arrays, where the ith item in weight_array is the weight value
        of the ith edge of the graph.

    store_transposed : bool
        Set to True if the graph should be transposed. This is required for some
        algorithms, such as pagerank.
    
    num_edges : int
        Number of edges
    
    do_expensive_check : bool
        If True, performs more extensive tests on the inputs to ensure
        validitity, at the expense of increased run time.

    edge_id_array : device array type
        Device array containing the edge ids of each directed edge.  Must match
        the ordering of the src/dst arrays.  Optional (may be null).  If
        provided, edge_type_array must also be provided.
    
    edge_type_array : device array type
        Device array containing the edge types of each directed edge.  Must
        match the ordering of the src/dst/edge_id arrays.  Optional (may be
        null).  If provided, edge_id_array must be provided.
    
    drop_self_loops : bool, optional (default='False')
        If true, drop any self loops that exist in the provided edge list.
    
    drop_multi_edges: bool, optional (default='False')
        If true, drop any multi edges that exist in the provided edge list
    """
    def __cinit__(self,
                  ResourceHandle resource_handle,
                  GraphProperties graph_properties,
                  src_array,
                  dst_array,
                  vertices_array=None,
                  weight_array=None,
                  store_transposed=False,
                  size_t num_arrays=1, # default value to not break users
                  do_expensive_check=True,
                  edge_id_array=None,
                  edge_type_array=None,
                  drop_self_loops=False,
                  drop_multi_edges=False):

        print("entering mg", flush=True)
        # FIXME: add tests for these
        if not(isinstance(store_transposed, (int, bool))):
            raise TypeError("expected int or bool for store_transposed, got "
                            f"{type(store_transposed)}")
        """
        if not(isinstance(num_edges, (int))):
            raise TypeError("expected int for num_edges, got "
                            f"{type(num_edges)}")
        if num_edges < 0:
            raise TypeError("num_edges must be > 0")
        """
        if not(isinstance(do_expensive_check, (int, bool))):
            raise TypeError("expected int or bool for do_expensive_check, got "
                            f"{type(do_expensive_check)}")
        assert_CAI_type(src_array, "src_array")
        assert_CAI_type(dst_array, "dst_array")
        assert_CAI_type(weight_array, "weight_array", True)
        assert_CAI_type(vertices_array, "vertices_array", True)

        assert_CAI_type(edge_id_array, "edge_id_array", True)
        if edge_id_array is not None and len(edge_id_array) != len(src_array):
            raise ValueError('Edge id array must be same length as edgelist')
        
        assert_CAI_type(edge_type_array, "edge_type_array", True)
        if edge_type_array is not None and len(edge_type_array) != len(src_array):
            raise ValueError('Edge type array must be same length as edgelist')

        # FIXME: assert that src_array and dst_array have the same type

        cdef cugraph_error_t* error_ptr
        cdef cugraph_error_code_t error_code

        # FIXME: Assert that the 'vertices', 'src', 'dst', 'weights', 'edge_ids'
        # and 'edge_type_ids' are of size 'num_arrays'.

        #cdef cugraph_type_erased_device_array_view_t* srcs_view_ptr[0] = \
        #    create_cugraph_type_erased_device_array_view_from_py_obj(
        #        src_array
        #    )

        cdef cugraph_type_erased_device_array_view_t** srcs_view_ptr_ptr = NULL
        cdef cugraph_type_erased_device_array_view_t* srcs_view_ptr = NULL

        cdef cugraph_type_erased_device_array_view_t** dsts_view_ptr_ptr = NULL
        cdef cugraph_type_erased_device_array_view_t* dsts_view_ptr = NULL

        cdef cugraph_type_erased_device_array_view_t** vertices_view_ptr_ptr = NULL
        cdef cugraph_type_erased_device_array_view_t* vertices_view_ptr = NULL

        #cdef cugraph_type_erased_device_array_view_t** weights_view_ptr_ptr = NULL
        #cdef cugraph_type_erased_device_array_view_t* weights_view_ptr = NULL

        #cdef cugraph_type_erased_device_array_view_t** edge_id_view_ptr_ptr = NULL
        #cdef cugraph_type_erased_device_array_view_t* edge_id_view_ptr = NULL

        cdef cugraph_type_erased_device_array_view_t** edge_type_view_ptr_ptr = NULL
        cdef cugraph_type_erased_device_array_view_t* edge_type_view_ptr = NULL


        for i in range(num_arrays):
            srcs_view_ptr = create_cugraph_type_erased_device_array_view_from_py_obj(
                    src_array
                )
            srcs_view_ptr_ptr = &(srcs_view_ptr)
            srcs_view_ptr_ptr += 1

            dsts_view_ptr = create_cugraph_type_erased_device_array_view_from_py_obj(
                    dst_array
                )
            dsts_view_ptr_ptr = &(dsts_view_ptr)
            dsts_view_ptr_ptr += 1

            if vertices_array is not None:
                vertices_view_ptr = create_cugraph_type_erased_device_array_view_from_py_obj(
                        vertices_array
                    )
                vertices_view_ptr_ptr = &(vertices_view_ptr)
                vertices_view_ptr_ptr += 1

            # FIXME: When checking wethere a graph has weights or edge ids, properly handle
            # SG VS MG as SG will have 'self.weights_view_ptr_ptr' non 'NULL'.

            if weight_array is not None:
                self.weights_view_ptr = create_cugraph_type_erased_device_array_view_from_py_obj(
                        weight_array
                    )
                self.weights_view_ptr_ptr = &(self.weights_view_ptr)
                self.weights_view_ptr_ptr += 1

            if edge_id_array is not None:
                self.edge_id_view_ptr = create_cugraph_type_erased_device_array_view_from_py_obj(
                        edge_id_array
                    )
                self.edge_id_view_ptr_ptr = &(self.edge_id_view_ptr)
                self.edge_id_view_ptr_ptr += 1

            if edge_type_array is not None:
                edge_type_view_ptr = create_cugraph_type_erased_device_array_view_from_py_obj(
                        edge_type_array
                    )
                edge_type_view_ptr_ptr = &(edge_type_view_ptr)
                edge_type_view_ptr_ptr += 1
        

        srcs_view_ptr_ptr -= num_arrays
        dsts_view_ptr_ptr -= num_arrays
        if vertices_array:
            vertices_view_ptr_ptr -= num_arrays
        if weight_array:
            self.weights_view_ptr_ptr -= num_arrays
        if edge_id_array:
            self.edge_id_view_ptr_ptr -= num_arrays
        if edge_type_array:
            self.edge_type_view_ptr_ptr -= num_arrays

        #num_arrays = 1
        print("calling MG graph create", flush=True)
        error_code = cugraph_graph_create_mg(
            resource_handle.c_resource_handle_ptr,
            &(graph_properties.c_graph_properties),
            vertices_view_ptr_ptr,
            srcs_view_ptr_ptr,
            dsts_view_ptr_ptr,
            self.weights_view_ptr_ptr,
            self.edge_id_view_ptr_ptr,
            edge_type_view_ptr_ptr,
            store_transposed,
            num_arrays,
            do_expensive_check,
            drop_self_loops,
            drop_multi_edges,
            &(self.c_graph_ptr),
            &error_ptr)

        print("Done calling MG graph create", flush=True)
        assert_success(error_code, error_ptr,
                       "cugraph_mg_graph_create()")

        cugraph_type_erased_device_array_view_free(srcs_view_ptr_ptr[0])
        cugraph_type_erased_device_array_view_free(dsts_view_ptr_ptr[0])
        if self.weights_view_ptr is not NULL:
            cugraph_type_erased_device_array_view_free(self.weights_view_ptr_ptr[0])
        if self.edge_id_view_ptr is not NULL:
            cugraph_type_erased_device_array_view_free(self.edge_id_view_ptr_ptr[0])
        if edge_type_view_ptr is not NULL:
            cugraph_type_erased_device_array_view_free(edge_type_view_ptr_ptr[0])

    def __dealloc__(self):
        if self.c_graph_ptr is not NULL:
            cugraph_mg_graph_free(self.c_graph_ptr)




"""
    cdef cugraph_error_code_t \
        cugraph_graph_create_sg_from_csr(
            const cugraph_resource_handle_t* handle,
            const cugraph_graph_properties_t* properties,
            const cugraph_type_erased_device_array_view_t* offsets,
            const cugraph_type_erased_device_array_view_t* indices,
            const cugraph_type_erased_device_array_view_t* weights,
            const cugraph_type_erased_device_array_view_t* edge_ids,
            const cugraph_type_erased_device_array_view_t* edge_type_ids,
            bool_t store_transposed,
            bool_t renumber,
            bool_t check,
            cugraph_graph_t** graph,
            cugraph_error_t** error
        )
"""