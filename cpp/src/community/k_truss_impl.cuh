/*
 * Copyright (c) 2020-2024, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include "prims/extract_transform_v_frontier_outgoing_e.cuh"
#include "prims/edge_bucket.cuh"
#include "prims/extract_transform_e.cuh"
#include "prims/fill_edge_property.cuh"
#include "prims/transform_e.cuh"
#include "prims/transform_reduce_dst_nbr_intersection_of_e_endpoints_by_v.cuh"
#include "prims/update_edge_src_dst_property.cuh"

#include <cugraph/algorithms.hpp>
#include <cugraph/detail/shuffle_wrappers.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/utilities/error.hpp>

#include <raft/util/integer_utils.hpp>

#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/distance.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/optional.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>

namespace cugraph {

template <typename vertex_t, typename edge_t>
struct extract_weak_edges {
  edge_t k{};
  __device__ thrust::optional<thrust::tuple<vertex_t, vertex_t, edge_t>> operator()(
    vertex_t src, vertex_t dst, thrust::nullopt_t, thrust::nullopt_t, edge_t count) const
  {
    //printf("\nsrc = %d, dst = %d, count = %d\n", src, dst, count);
    return count < k - 2
             ? thrust::optional<thrust::tuple<vertex_t, vertex_t, edge_t>>{thrust::make_tuple(src, dst, count)}
             : thrust::nullopt;
  }
};

template <typename vertex_t, typename edge_t>
struct extract_edges {
  __device__ thrust::optional<thrust::tuple<vertex_t, vertex_t, edge_t>> operator()(
    
    auto src, auto dst, thrust::nullopt_t, thrust::nullopt_t, auto count) const
  {
    //printf("\nchecking the count - src = %d, dst = %d, count = %d\n", src, dst, count);
    return thrust::make_tuple(src, dst, count);
  }
};

template <typename vertex_t>
struct extract_edges_to_q_r {

  raft::device_span<vertex_t const> vertex_q_r{};
  __device__ thrust::optional<thrust::tuple<vertex_t, vertex_t>> operator()(
  
  auto src, auto dst, thrust::nullopt_t, thrust::nullopt_t, thrust::nullopt_t) const
  {
    //printf("\nchecking the count - src = %d, dst = %d, count = %d\n", src, dst, count);
    auto itr = thrust::find(
        thrust::seq, vertex_q_r.begin(), vertex_q_r.end(), src);
    return (itr != vertex_q_r.end() && *itr == src)
              ? thrust::optional<thrust::tuple<vertex_t, vertex_t>>{thrust::make_tuple(src, dst)}
              : thrust::nullopt;
  }
};

namespace {

template <typename vertex_t>
struct exclude_self_loop_t {
  __device__ thrust::optional<thrust::tuple<vertex_t, vertex_t>> operator()(
    vertex_t src, vertex_t dst, thrust::nullopt_t, thrust::nullopt_t, thrust::nullopt_t) const
  {
    return src != dst
             ? thrust::optional<thrust::tuple<vertex_t, vertex_t>>{thrust::make_tuple(src, dst)}
             : thrust::nullopt;
  }
};

template <typename vertex_t, typename weight_t, typename edge_t>
struct extract_low_to_high_degree_weighted_edges_t {
  __device__ thrust::optional<thrust::tuple<vertex_t, vertex_t, weight_t>> operator()(
    vertex_t src, vertex_t dst, edge_t src_out_degree, edge_t dst_out_degree, weight_t wgt) const
  {
    return (src_out_degree < dst_out_degree)
             ? thrust::optional<thrust::tuple<vertex_t, vertex_t, weight_t>>{thrust::make_tuple(
                 src, dst, wgt)}
             : (((src_out_degree == dst_out_degree) &&
                 (src < dst) /* tie-breaking using vertex ID */)
                  ? thrust::optional<
                      thrust::tuple<vertex_t, vertex_t, weight_t>>{thrust::make_tuple(
                      src, dst, wgt)}
                  : thrust::nullopt);
  }
};

template <typename vertex_t, typename edge_t>
struct extract_low_to_high_degree_edges_t {
  __device__ thrust::optional<thrust::tuple<vertex_t, vertex_t>> operator()(vertex_t src,
                                                                            vertex_t dst,
                                                                            edge_t src_out_degree,
                                                                            edge_t dst_out_degree,
                                                                            thrust::nullopt_t) const
  {
    return (src_out_degree < dst_out_degree)
             ? thrust::optional<thrust::tuple<vertex_t, vertex_t>>{thrust::make_tuple(src, dst)}
             : (((src_out_degree == dst_out_degree) &&
                 (src < dst) /* tie-breaking using vertex ID */)
                  ? thrust::optional<thrust::tuple<vertex_t, vertex_t>>{thrust::make_tuple(src,
                                                                                           dst)}
                  : thrust::nullopt);
  }
};

template <typename vertex_t, typename edge_t, bool generate_p_r>
struct generate_p_r_or_q_r_from_p_q {
  size_t chunk_start{};
  raft::device_span<size_t const> intersection_offsets{};
  raft::device_span<vertex_t const> intersection_indices{};
  raft::device_span<vertex_t const> invalid_srcs{};
  raft::device_span<vertex_t const> invalid_dsts{};

  __device__ thrust::tuple<vertex_t, vertex_t> operator()(edge_t i) const
  {
    auto itr = thrust::upper_bound(
      thrust::seq, intersection_offsets.begin() + 1, intersection_offsets.end(), i);
    auto idx = thrust::distance(intersection_offsets.begin() + 1, itr);

    if constexpr (generate_p_r) {
      return thrust::make_tuple(invalid_srcs[chunk_start + idx], intersection_indices[i]);

    } else {
      return thrust::make_tuple(invalid_dsts[chunk_start + idx], intersection_indices[i]);
    }
  }
};

template <typename vertex_t, typename edge_t>
struct extract_q_idx {
  using return_type = thrust::optional<thrust::tuple<vertex_t, edge_t>>;

  return_type __device__ operator()(thrust::tuple<vertex_t, edge_t> tagged_src,
                                    vertex_t dst,
                                    thrust::nullopt_t,
                                    thrust::nullopt_t,
                                    thrust::nullopt_t) const
  {
    printf("\n dst = %d, tag = %d\n", dst, thrust::get<1>(tagged_src));
    return thrust::make_optional(thrust::make_tuple(dst, thrust::get<1>(tagged_src)));
  }
};

template <typename vertex_t, typename edge_t>
struct extract_q_idx_closing {
  using return_type = thrust::optional<thrust::tuple<vertex_t, edge_t>>;
  raft::device_span<vertex_t const> weak_edgelist_dsts;

  return_type __device__ operator()(thrust::tuple<vertex_t, edge_t> tagged_src,
                                    vertex_t dst,
                                    thrust::nullopt_t,
                                    thrust::nullopt_t,
                                    thrust::nullopt_t) const
  {
    //printf("\n dst = %d, tag = %d\n", dst, thrust::get<1>(tagged_src));
    edge_t idx = thrust::get<1>(tagged_src);
    if (dst == weak_edgelist_dsts[idx]){
      //printf("\nsrc = %d ---  dst = %d, tag = %d\n", thrust::get<0>(tagged_src), dst, thrust::get<1>(tagged_src));
    }
    return dst == weak_edgelist_dsts[idx]
             ? thrust::make_optional(thrust::make_tuple(thrust::get<0>(tagged_src), idx))
             : thrust::nullopt;
  }
};

template <typename vertex_t, typename edge_t>
struct generate_p_q {
  size_t chunk_start{};
  raft::device_span<size_t const> intersection_offsets{};
  raft::device_span<vertex_t const> intersection_indices{};
  raft::device_span<vertex_t const> invalid_srcs{};
  raft::device_span<vertex_t const> invalid_dsts{};

  __device__ thrust::tuple<vertex_t, vertex_t> operator()(edge_t i) const
  {
    auto itr = thrust::upper_bound(
      thrust::seq, intersection_offsets.begin() + 1, intersection_offsets.end(), i);
    auto idx = thrust::distance(intersection_offsets.begin() + 1, itr);
    
    return thrust::make_tuple(invalid_srcs[chunk_start + idx], invalid_dsts[chunk_start + idx]);
  }
};

template <typename vertex_t, typename edge_t, typename EdgeIterator>
struct generate_p_r {
  EdgeIterator invalid_edge{};
  raft::device_span<edge_t const> invalid_edge_idx{};

  __device__ thrust::tuple<vertex_t, vertex_t> operator()(edge_t i) const
  {
    return *(invalid_edge + invalid_edge_idx[i]);
  }
};

template <typename vertex_t, typename edge_t, typename EdgeIterator, bool generate_p_q>
struct generate_p_q_q_r {
  EdgeIterator invalid_edge{};
  raft::device_span<vertex_t const> q_closing{};
  raft::device_span<edge_t const> invalid_edge_idx{};

  __device__ thrust::tuple<vertex_t, vertex_t> operator()(edge_t i) const
  {

    if constexpr (generate_p_q) {
      return thrust::make_tuple(thrust::get<0>(*(invalid_edge + invalid_edge_idx[i])), q_closing[i]);
    } else {
      return thrust::make_tuple(q_closing[i], thrust::get<1>(*(invalid_edge + invalid_edge_idx[i])));
    }
  }
};

// FIXME: remove 'EdgeIterator' template
template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu>
void unroll_p_q_p_r_edges(raft::handle_t const& handle,
                          graph_view_t<vertex_t, edge_t, false, multi_gpu> & cur_graph_view,
                          //thrust::optional(q_r_graph)
                          edge_property_t<graph_view_t<vertex_t, edge_t, false, multi_gpu>, edge_t> & e_property_triangle_count,
                          raft::device_span<vertex_t const> vertex_pair_buffer_first,
                          raft::device_span<vertex_t const> vertex_pair_buffer_last
                          //EdgeIterator vertex_pair_buffer,
                          //vertex_t buffer_size
                          ) {
  

  cugraph::edge_bucket_t<vertex_t, void, true, multi_gpu, true> edges_to_decrement_count(handle);
      edges_to_decrement_count.insert(vertex_pair_buffer_first.begin(),
                                  vertex_pair_buffer_first.end(),
                                  vertex_pair_buffer_last.begin());


  printf("\nupdating count\n");
  auto vertex_pair_buffer_begin = thrust::make_zip_iterator(vertex_pair_buffer_first.begin(), vertex_pair_buffer_last.begin());
  cugraph::transform_e(
    handle,
    cur_graph_view,
    edges_to_decrement_count,
    cugraph::edge_src_dummy_property_t{}.view(),
    cugraph::edge_dst_dummy_property_t{}.view(),
    e_property_triangle_count.view(),
    [
      vertex_pair_buffer_begin = vertex_pair_buffer_begin,
      vertex_pair_buffer_end = vertex_pair_buffer_begin + vertex_pair_buffer_first.size()
    ]
    __device__(auto src, auto dst, thrust::nullopt_t, thrust::nullopt_t, edge_t count) {
      //printf("\nsrc = %d, dst = %d, count = %d\n", src, dst, count);
      auto e = thrust::make_tuple(src, dst);
      auto itr = thrust::lower_bound(
        thrust::seq, vertex_pair_buffer_begin, vertex_pair_buffer_end, e);
      
      if ((itr != vertex_pair_buffer_end) && (*itr == e)) {
        //printf("\nupdating the count - src = %d, dst = %d, count = %d\n", src, dst, count);
        return count - 1;
      } 
      
      return count;

    },
    e_property_triangle_count.mutable_view(),
    false);
  

};




template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu, bool is_p_q_edge>
vertex_t find_unroll_p_q_q_r_edges(raft::handle_t const& handle,
                                   graph_view_t<vertex_t, edge_t, false, multi_gpu> & cur_graph_view,
                                   //thrust::optional(q_r_graph)
                                   edge_property_t<graph_view_t<vertex_t, edge_t, false, multi_gpu>, edge_t> & e_property_triangle_count,
                                   raft::device_span<vertex_t const> weak_edgelist_srcs,
                                   raft::device_span<vertex_t const> weak_edgelist_dsts,
                                   bool do_expensive_check) {

  size_t prev_chunk_size         = 0;
  size_t chunk_num_invalid_edges = weak_edgelist_srcs.size();
  size_t edges_to_intersect_per_iteration =
        static_cast<size_t>(handle.get_device_properties().multiProcessorCount) * (1 << 17);
  auto num_chunks =
    raft::div_rounding_up_safe(weak_edgelist_srcs.size(), edges_to_intersect_per_iteration);
  
  auto invalid_edge_first = thrust::make_zip_iterator(weak_edgelist_srcs.begin(), weak_edgelist_dsts.begin());

  
  for (size_t i = 0; i < num_chunks; ++i) {
      auto chunk_size = std::min(edges_to_intersect_per_iteration, chunk_num_invalid_edges);
      auto [intersection_offsets, intersection_indices] =
        detail::nbr_intersection(handle,
                                cur_graph_view,
                                cugraph::edge_dummy_property_t{}.view(),
                                invalid_edge_first + prev_chunk_size,
                                invalid_edge_first + prev_chunk_size + chunk_size,
                                std::array<bool, 2>{true, true},
                                do_expensive_check);
    
      raft::print_device_vector("intersection_offsets", intersection_offsets.data(), intersection_offsets.size(), std::cout);
      raft::print_device_vector("intersection_indices", intersection_indices.data(), intersection_indices.size(), std::cout);
      
      // Generate (p, q) edges
      // FIXME: Should this array be reduced? an edge can have an intersection size  > 1
      auto vertex_pair_buffer_p_q =
        allocate_dataframe_buffer<thrust::tuple<vertex_t, vertex_t>>(intersection_indices.size(),
                                                                     handle.get_stream());
  
      thrust::tabulate(
        handle.get_thrust_policy(),
        get_dataframe_buffer_begin(vertex_pair_buffer_p_q),
        get_dataframe_buffer_end(vertex_pair_buffer_p_q),
        generate_p_q<vertex_t, edge_t>{
          prev_chunk_size,
          raft::device_span<size_t const>(intersection_offsets.data(),
                                          intersection_offsets.size()),
          raft::device_span<vertex_t const>(intersection_indices.data(),
                                            intersection_indices.size()),
          weak_edgelist_srcs,
          weak_edgelist_dsts
          });
      
      raft::print_device_vector("vertex_pair_buffer_p_q", std::get<0>(vertex_pair_buffer_p_q).data(), std::get<0>(vertex_pair_buffer_p_q).size(), std::cout);
      raft::print_device_vector("vertex_pair_buffer_p_q", std::get<1>(vertex_pair_buffer_p_q).data(), std::get<1>(vertex_pair_buffer_p_q).size(), std::cout);

      // unroll (p, q) edges
      // FIXME: remove 'EdgeIterator' template
      unroll_p_q_p_r_edges<vertex_t, edge_t, weight_t, multi_gpu>(
        handle,
        cur_graph_view,
        e_property_triangle_count,
        raft::device_span<vertex_t>(std::get<0>(vertex_pair_buffer_p_q).data(), std::get<0>(vertex_pair_buffer_p_q).size()),
        raft::device_span<vertex_t>(std::get<1>(vertex_pair_buffer_p_q).data(), std::get<1>(vertex_pair_buffer_p_q).size())
      );
  
      
      /*
      // Unroll (p, q) edges
      cugraph::edge_bucket_t<vertex_t, void, true, multi_gpu, true> invalid_edges_bucket(handle);
      invalid_edges_bucket.insert(weak_edgelist_srcs.begin(),
                                    weak_edgelist_srcs.end(),
                                    weak_edgelist_dsts.begin());

      printf("\nupdating count\n");
      cugraph::transform_e(
        handle,
        cur_graph_view,
        invalid_edges_bucket,
        cugraph::edge_src_dummy_property_t{}.view(),
        cugraph::edge_dst_dummy_property_t{}.view(),
        e_property_triangle_count.view(),
        [
          vertex_pair_buffer_p_q_begin = get_dataframe_buffer_begin(vertex_pair_buffer_p_q),
          vertex_pair_buffer_p_q_end = get_dataframe_buffer_end(vertex_pair_buffer_p_q)
        ]
        __device__(auto src, auto dst, thrust::nullopt_t, thrust::nullopt_t, edge_t count) {
          //printf("\nsrc = %d, dst = %d, count = %d\n", src, dst, count);
          auto e = thrust::make_tuple(src, dst);
          auto itr = thrust::lower_bound(
            thrust::seq, vertex_pair_buffer_p_q_begin, vertex_pair_buffer_p_q_end, e);
          
          if ((itr != vertex_pair_buffer_p_q_end) && (*itr == e)) {
            //printf("\nupdating the count - src = %d, dst = %d, count = %d\n", src, dst, count);
            return count - 1;
          } 
          
          return count;
    
        },
        e_property_triangle_count.mutable_view(),
        false);
      
    

      //#if 0
      auto [srcs, dsts, count] = extract_transform_e(handle,
                                            cur_graph_view,
                                            cugraph::edge_src_dummy_property_t{}.view(),
                                            cugraph::edge_dst_dummy_property_t{}.view(),
                                            //view_concat(e_property_triangle_count.view(), modified_triangle_count.view()),
                                            e_property_triangle_count.view(),
                                            extract_edges<vertex_t, edge_t>{});

      printf("\nafter unrolling (p, q) edges\n");
      raft::print_device_vector("srcs", srcs.data(), srcs.size(), std::cout);
      raft::print_device_vector("dsts", dsts.data(), dsts.size(), std::cout);
      raft::print_device_vector("count", count.data(), count.size(), std::cout);
      //#endif
      */

      auto vertex_pair_buffer_p_r_edge_p_q =
          allocate_dataframe_buffer<thrust::tuple<vertex_t, vertex_t>>(intersection_indices.size(),
                                                                       handle.get_stream());
      thrust::tabulate(
        handle.get_thrust_policy(),
        get_dataframe_buffer_begin(vertex_pair_buffer_p_r_edge_p_q),
        get_dataframe_buffer_end(vertex_pair_buffer_p_r_edge_p_q),
        generate_p_r_or_q_r_from_p_q<vertex_t, edge_t, true>{
          prev_chunk_size,
          raft::device_span<size_t const>(intersection_offsets.data(),
                                          intersection_offsets.size()),
          raft::device_span<vertex_t const>(intersection_indices.data(),
                                            intersection_indices.size()),
          weak_edgelist_srcs,
          weak_edgelist_dsts});
      
      unroll_p_q_p_r_edges<vertex_t, edge_t, weight_t, multi_gpu>(
        handle,
        cur_graph_view,
        e_property_triangle_count,
        raft::device_span<vertex_t>(std::get<0>(vertex_pair_buffer_p_r_edge_p_q).data(), std::get<0>(vertex_pair_buffer_p_r_edge_p_q).size()),
        raft::device_span<vertex_t>(std::get<1>(vertex_pair_buffer_p_r_edge_p_q).data(), std::get<1>(vertex_pair_buffer_p_r_edge_p_q).size())
      );
      
      /*
      //invalid_edges
      cugraph::transform_e(
        handle,
        cur_graph_view,
        cugraph::edge_src_dummy_property_t{}.view(),
        cugraph::edge_dst_dummy_property_t{}.view(),
        e_property_triangle_count.view(),
        [
          vertex_pair_buffer_p_r_edge_p_q_begin = get_dataframe_buffer_begin(vertex_pair_buffer_p_r_edge_p_q),
          vertex_pair_buffer_p_r_edge_p_q_end = get_dataframe_buffer_end(vertex_pair_buffer_p_r_edge_p_q)
        ]
        __device__(auto src, auto dst, thrust::nullopt_t, thrust::nullopt_t, edge_t count) {
          //printf("\nsrc = %d, dst = %d, count = %d\n", src, dst, count);
          auto e = thrust::make_tuple(src, dst);
          auto itr = thrust::lower_bound(
            thrust::seq, vertex_pair_buffer_p_r_edge_p_q_begin, vertex_pair_buffer_p_r_edge_p_q_end, e);
          
          if ((itr != vertex_pair_buffer_p_r_edge_p_q_end) && (*itr == e)) {
            //printf("\nupdating the count - src = %d, dst = %d, count = %d\n", src, dst, count);
            return count - 1;
          } 
          
          return count;
    
        },
        e_property_triangle_count.mutable_view(),
        false);
      printf("\nafter unrolling (p, q) edge (p, r) edges\n");
      */

      

      auto vertex_pair_buffer_q_r_edge_p_q =
          allocate_dataframe_buffer<thrust::tuple<vertex_t, vertex_t>>(intersection_indices.size(),
                                                                       handle.get_stream());
      thrust::tabulate(
        handle.get_thrust_policy(),
        get_dataframe_buffer_begin(vertex_pair_buffer_q_r_edge_p_q),
        get_dataframe_buffer_end(vertex_pair_buffer_q_r_edge_p_q),
        generate_p_r_or_q_r_from_p_q<vertex_t, edge_t, false>{
          prev_chunk_size,
          raft::device_span<size_t const>(intersection_offsets.data(),
                                          intersection_offsets.size()),
          raft::device_span<vertex_t const>(intersection_indices.data(),
                                            intersection_indices.size()),
          weak_edgelist_srcs,
          weak_edgelist_dsts});
      
      unroll_p_q_p_r_edges<vertex_t, edge_t, weight_t, multi_gpu>(
        handle,
        cur_graph_view,
        e_property_triangle_count,
        raft::device_span<vertex_t>(std::get<0>(vertex_pair_buffer_q_r_edge_p_q).data(), std::get<0>(vertex_pair_buffer_q_r_edge_p_q).size()),
        raft::device_span<vertex_t>(std::get<1>(vertex_pair_buffer_q_r_edge_p_q).data(), std::get<1>(vertex_pair_buffer_q_r_edge_p_q).size())
      );
      
      /*
      // invalid_edges
      cugraph::transform_e(
        handle,
        cur_graph_view,
        cugraph::edge_src_dummy_property_t{}.view(),
        cugraph::edge_dst_dummy_property_t{}.view(),
        e_property_triangle_count.view(),
        [
          vertex_pair_buffer_q_r_edge_p_q_begin = get_dataframe_buffer_begin(vertex_pair_buffer_q_r_edge_p_q),
          vertex_pair_buffer_q_r_edge_p_q_end = get_dataframe_buffer_end(vertex_pair_buffer_q_r_edge_p_q)
        ]
        __device__(auto src, auto dst, thrust::nullopt_t, thrust::nullopt_t, edge_t count) {
          //printf("\nsrc = %d, dst = %d, count = %d\n", src, dst, count);
          auto e = thrust::make_tuple(src, dst);
          auto itr = thrust::lower_bound(
            thrust::seq, vertex_pair_buffer_q_r_edge_p_q_begin, vertex_pair_buffer_q_r_edge_p_q_end, e);
          
          if ((itr != vertex_pair_buffer_q_r_edge_p_q_end) && (*itr == e)) {
            //printf("\nupdating the count - src = %d, dst = %d, count = %d\n", src, dst, count);
            return count - 1;
          } 
          
          return count;
    
        },
        e_property_triangle_count.mutable_view(),
        false);




      #if 0
      auto [srcs_, dsts_, count_] = extract_transform_e(handle,
                                            cur_graph_view,
                                            cugraph::edge_src_dummy_property_t{}.view(),
                                            cugraph::edge_dst_dummy_property_t{}.view(),
                                            e_property_triangle_count.view(),
                                            extract_edges<vertex_t, edge_t>{});
      
      printf("\nafter unrolling (p, q) edges from (p, r)\n");
      raft::print_device_vector("srcs", srcs_.data(), srcs_.size(), std::cout);
      raft::print_device_vector("dsts", dsts_.data(), dsts_.size(), std::cout);
      raft::print_device_vector("count", count_.data(), count_.size(), std::cout);
      #endif
      */
    
      if constexpr (is_p_q_edge) {
        // FIXME: This might not work when chunking because the invalid (p. q) edges should be
        // temporarily masked at the end when completly unrolling (p, q) edges. Failing to do
        // this might cause some invalid edges (p, q) to not have their count decremented
      //cugraph::edge_property_t<graph_view_t<vertex_t, edge_t, false, multi_gpu>, bool> tmp_edge_mask(handle, cur_graph_view);
      //cugraph::fill_edge_property(handle, cur_graph_view, true, tmp_edge_mask);
      /*
      cur_graph_view.attach_edge_mask(tmp_edge_mask.view());
        cugraph::edge_bucket_t<vertex_t, void, true, multi_gpu, true> edges_to_tmp_mask(handle);
        edges_to_tmp_mask.clear(); // Continuously mask (p, q) edges as they are processed in chunks
        edges_to_tmp_mask.insert(std::get<0>(vertex_pair_buffer_p_q).begin(),
                                 std::get<0>(vertex_pair_buffer_p_q).end(),
                                 std::get<1>(vertex_pair_buffer_p_q).begin());

        cugraph::transform_e(
              handle,
              cur_graph_view,
              edges_to_tmp_mask,
              cugraph::edge_src_dummy_property_t{}.view(),
              cugraph::edge_dst_dummy_property_t{}.view(),
              cugraph::edge_dummy_property_t{}.view(),
              [] __device__(auto src, auto dst, thrust::nullopt_t, thrust::nullopt_t, auto wgt) {
                return false;
              },
              tmp_edge_mask.mutable_view(),
              false);
        
        cur_graph_view.attach_edge_mask(tmp_edge_mask.view());
        */
      }
      


    
      prev_chunk_size += chunk_size;
      chunk_num_invalid_edges -= chunk_size;
    }

  return 0;
}
}  // namespace







template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           std::optional<rmm::device_uvector<weight_t>>>
k_truss(raft::handle_t const& handle,
        graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
        std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
        edge_t k,
        bool do_expensive_check)
{
  // 1. Check input arguments.

  CUGRAPH_EXPECTS(!graph_view.has_edge_mask(), "unimplemented.");

  CUGRAPH_EXPECTS(graph_view.is_symmetric(),
                  "Invalid input arguments: K-truss currently supports undirected graphs only.");
  CUGRAPH_EXPECTS(!graph_view.is_multigraph(),
                  "Invalid input arguments: K-truss currently does not support multi-graphs.");

  if (do_expensive_check) {
    // nothing to do
  }

  std::optional<graph_t<vertex_t, edge_t, false, multi_gpu>> modified_graph{std::nullopt};
  std::optional<graph_view_t<vertex_t, edge_t, false, multi_gpu>> modified_graph_view{std::nullopt};
  std::optional<rmm::device_uvector<vertex_t>> renumber_map{std::nullopt};
  std::optional<edge_property_t<graph_view_t<vertex_t, edge_t, false, multi_gpu>, weight_t>>
    edge_weight{std::nullopt};
  std::optional<rmm::device_uvector<weight_t>> wgts{std::nullopt};

  if (graph_view.count_self_loops(handle) > edge_t{0}) {
    auto [srcs, dsts] = extract_transform_e(handle,
                                            graph_view,
                                            edge_src_dummy_property_t{}.view(),
                                            edge_dst_dummy_property_t{}.view(),
                                            edge_dummy_property_t{}.view(),
                                            exclude_self_loop_t<vertex_t>{});

    if constexpr (multi_gpu) {
      std::tie(srcs, dsts, std::ignore, std::ignore, std::ignore) =
        detail::shuffle_ext_vertex_pairs_with_values_to_local_gpu_by_edge_partitioning<vertex_t,
                                                                                       edge_t,
                                                                                       weight_t,
                                                                                       int32_t>(
          handle, std::move(srcs), std::move(dsts), std::nullopt, std::nullopt, std::nullopt);
    }

    std::tie(*modified_graph, std::ignore, std::ignore, std::ignore, renumber_map) =
      create_graph_from_edgelist<vertex_t, edge_t, weight_t, edge_t, int32_t, false, multi_gpu>(
        handle,
        std::nullopt,
        std::move(srcs),
        std::move(dsts),
        std::nullopt,
        std::nullopt,
        std::nullopt,
        cugraph::graph_properties_t{true, graph_view.is_multigraph()},
        true);

    modified_graph_view = (*modified_graph).view();
  }

  // 3. Find (k-1)-core and exclude edges that do not belong to (k-1)-core
  #if 0
  {
    auto cur_graph_view = modified_graph_view ? *modified_graph_view : graph_view;

    auto vertex_partition_range_lasts =
      renumber_map
        ? std::make_optional<std::vector<vertex_t>>(cur_graph_view.vertex_partition_range_lasts())
        : std::nullopt;

    rmm::device_uvector<edge_t> core_numbers(cur_graph_view.number_of_vertices(),
                                             handle.get_stream());
    core_number(
      handle, cur_graph_view, core_numbers.data(), k_core_degree_type_t::OUT, size_t{2}, size_t{2});

    raft::device_span<edge_t const> core_number_span{core_numbers.data(), core_numbers.size()};

    rmm::device_uvector<vertex_t> srcs{0, handle.get_stream()};
    rmm::device_uvector<vertex_t> dsts{0, handle.get_stream()};
    std::tie(srcs, dsts, wgts) = k_core(handle,
                                        cur_graph_view,
                                        edge_weight_view,
                                        k - 1,
                                        std::make_optional(k_core_degree_type_t::OUT),
                                        std::make_optional(core_number_span));

    if constexpr (multi_gpu) {
      std::tie(srcs, dsts, std::ignore, std::ignore, std::ignore) =
        detail::shuffle_ext_vertex_pairs_with_values_to_local_gpu_by_edge_partitioning<vertex_t,
                                                                                       edge_t,
                                                                                       weight_t,
                                                                                       int32_t>(
          handle, std::move(srcs), std::move(dsts), std::nullopt, std::nullopt, std::nullopt);
    }

    std::optional<rmm::device_uvector<vertex_t>> tmp_renumber_map{std::nullopt};

    std::tie(*modified_graph, edge_weight, std::ignore, std::ignore, tmp_renumber_map) =
      create_graph_from_edgelist<vertex_t, edge_t, weight_t, edge_t, int32_t, false, multi_gpu>(
        handle,
        std::nullopt,
        std::move(srcs),
        std::move(dsts),
        std::move(wgts),
        std::nullopt,
        std::nullopt,
        cugraph::graph_properties_t{true, graph_view.is_multigraph()},
        false);

    modified_graph_view = (*modified_graph).view();

    if (renumber_map) {  // collapse renumber_map
      unrenumber_int_vertices<vertex_t, multi_gpu>(handle,
                                                   (*tmp_renumber_map).data(),
                                                   (*tmp_renumber_map).size(),
                                                   (*renumber_map).data(),
                                                   *vertex_partition_range_lasts);
    }
    renumber_map = std::move(tmp_renumber_map);
  }
  #endif

  // 4. Keep only the edges from a low-degree vertex to a high-degree vertex.

  {
    auto cur_graph_view = modified_graph_view ? *modified_graph_view : graph_view;

    auto vertex_partition_range_lasts =
      renumber_map
        ? std::make_optional<std::vector<vertex_t>>(cur_graph_view.vertex_partition_range_lasts())
        : std::nullopt;

    auto out_degrees = cur_graph_view.compute_out_degrees(handle);
    edge_src_property_t<decltype(cur_graph_view), edge_t> edge_src_out_degrees(handle,
                                                                               cur_graph_view);
    edge_dst_property_t<decltype(cur_graph_view), edge_t> edge_dst_out_degrees(handle,
                                                                               cur_graph_view);
    update_edge_src_property(handle, cur_graph_view, out_degrees.begin(), edge_src_out_degrees);
    update_edge_dst_property(handle, cur_graph_view, out_degrees.begin(), edge_dst_out_degrees);

    rmm::device_uvector<vertex_t> srcs(0, handle.get_stream());
    rmm::device_uvector<vertex_t> dsts(0, handle.get_stream());

    edge_weight_view =
      edge_weight ? std::make_optional((*edge_weight).view())
                  : std::optional<edge_property_view_t<edge_t, weight_t const*>>{std::nullopt};
    if (edge_weight_view) {
      std::tie(srcs, dsts, wgts) = extract_transform_e(
        handle,
        cur_graph_view,
        edge_src_out_degrees.view(),
        edge_dst_out_degrees.view(),
        *edge_weight_view,
        extract_low_to_high_degree_weighted_edges_t<vertex_t, weight_t, edge_t>{});
    } else {
      std::tie(srcs, dsts) =
        extract_transform_e(handle,
                            cur_graph_view,
                            edge_src_out_degrees.view(),
                            edge_dst_out_degrees.view(),
                            edge_dummy_property_t{}.view(),
                            extract_low_to_high_degree_edges_t<vertex_t, edge_t>{});
    }

    if constexpr (multi_gpu) {
      std::tie(srcs, dsts, wgts, std::ignore, std::ignore) =
        detail::shuffle_ext_vertex_pairs_with_values_to_local_gpu_by_edge_partitioning<vertex_t,
                                                                                       edge_t,
                                                                                       weight_t,
                                                                                       int32_t>(
          handle, std::move(srcs), std::move(dsts), std::move(wgts), std::nullopt, std::nullopt);
    }

    std::optional<rmm::device_uvector<vertex_t>> tmp_renumber_map{std::nullopt};

    std::tie(*modified_graph, edge_weight, std::ignore, std::ignore, tmp_renumber_map) =
      create_graph_from_edgelist<vertex_t, edge_t, weight_t, edge_t, int32_t, false, multi_gpu>(
        handle,
        std::nullopt,
        std::move(srcs),
        std::move(dsts),
        std::move(wgts),
        std::nullopt,
        std::nullopt,
        cugraph::graph_properties_t{false /* now asymmetric */, cur_graph_view.is_multigraph()},
        false);

    modified_graph_view = (*modified_graph).view();
    if (renumber_map) {  // collapse renumber_map
      unrenumber_int_vertices<vertex_t, multi_gpu>(handle,
                                                   (*tmp_renumber_map).data(),
                                                   (*tmp_renumber_map).size(),
                                                   (*renumber_map).data(),
                                                   *vertex_partition_range_lasts);
    }
    renumber_map = std::move(tmp_renumber_map);
  }

  // 5. Decompress the resulting graph to an edges list and ind intersection of edges endpoints
  // for each partition using detail::nbr_intersection

  {
    auto cur_graph_view = modified_graph_view ? *modified_graph_view : graph_view;
    /*
    Design
    1) create a new graph with with the edge property from which we will iterate
      a) Directly update the property of the edges
      a) How do you traverse the graph?
    */ 

    auto e_property_triangle_count = edge_triangle_count<vertex_t, edge_t, false>(handle, cur_graph_view);

    cugraph::edge_property_t<decltype(cur_graph_view), bool> tmp_edge_mask(handle, cur_graph_view);
    cugraph::fill_edge_property(handle, cur_graph_view, true, tmp_edge_mask);
    cur_graph_view.attach_edge_mask(tmp_edge_mask.view());
    
    // extract the edges that have counts less than k - 2. THose edges will be unrolled
    std::cout<< "before calling extract transform_e" << std::endl;
    auto [weak_edgelist_srcs, weak_edgelist_dsts, triangle_count] = extract_transform_e(handle,
                                                                              cur_graph_view,
                                                                              edge_src_dummy_property_t{}.view(),
                                                                              edge_dst_dummy_property_t{}.view(),
                                                                              e_property_triangle_count.view(),
                                                                              extract_weak_edges<vertex_t, edge_t>{k});
    
    auto invalid_edge_first = thrust::make_zip_iterator(weak_edgelist_srcs.begin(), weak_edgelist_dsts.begin());
    
    raft::print_device_vector("srcs", weak_edgelist_srcs.data(), weak_edgelist_srcs.size(), std::cout);
    raft::print_device_vector("dsts", weak_edgelist_dsts.data(), weak_edgelist_dsts.size(), std::cout);
    raft::print_device_vector("n_tr", triangle_count.data(), triangle_count.size(), std::cout);

    // Call nbr_intersection unroll (p, q) edges
    size_t edges_to_intersect_per_iteration =
        static_cast<size_t>(handle.get_device_properties().multiProcessorCount) * (1 << 17);

    size_t prev_chunk_size         = 0;
    size_t chunk_num_invalid_edges = weak_edgelist_srcs.size();

    auto num_chunks =
      raft::div_rounding_up_safe(weak_edgelist_srcs.size(), edges_to_intersect_per_iteration);

    edge_property_t<decltype(cur_graph_view), edge_t> modified_triangle_count(handle, cur_graph_view);
    
    // find intersection edges
    /* 
    find_unroll_p_q_q_r_edges<vertex_t, edge_t, weight_t, multi_gpu, true>(
      handle,
      cur_graph_view,
      e_property_triangle_count,
      raft::device_span<vertex_t const>(weak_edgelist_srcs.data(), weak_edgelist_srcs.size()),
      raft::device_span<vertex_t const>(weak_edgelist_dsts.data(), weak_edgelist_dsts.size()),
      do_expensive_check
      //invalid_edge_first,
      //weak_edgelist_srcs.size()
    );
    */

  
  
    /*
    for (size_t i = 0; i < num_chunks; ++i) {
      auto chunk_size = std::min(edges_to_intersect_per_iteration, chunk_num_invalid_edges);
      auto [intersection_offsets, intersection_indices] =
        detail::nbr_intersection(handle,
                                cur_graph_view,
                                cugraph::edge_dummy_property_t{}.view(),
                                invalid_edge_first + prev_chunk_size,
                                invalid_edge_first + prev_chunk_size + chunk_size,
                                std::array<bool, 2>{true, true},
                                do_expensive_check);
    
      raft::print_device_vector("intersection_offsets", intersection_offsets.data(), intersection_offsets.size(), std::cout);
      raft::print_device_vector("intersection_indices", intersection_indices.data(), intersection_indices.size(), std::cout);
      
      // Generate (p, q) edges
      // FIXME: Should this array be reduced? an edge can have an intersection size  > 1
      auto vertex_pair_buffer_p_q =
        allocate_dataframe_buffer<thrust::tuple<vertex_t, vertex_t>>(intersection_indices.size(),
                                                                     handle.get_stream());
      
      thrust::tabulate(
        handle.get_thrust_policy(),
        get_dataframe_buffer_begin(vertex_pair_buffer_p_q),
        get_dataframe_buffer_end(vertex_pair_buffer_p_q),
        generate_p_q<vertex_t, edge_t>{
          prev_chunk_size,
          raft::device_span<size_t const>(intersection_offsets.data(),
                                          intersection_offsets.size()),
          raft::device_span<vertex_t const>(intersection_indices.data(),
                                            intersection_indices.size()),
          raft::device_span<vertex_t>(weak_edgelist_srcs.data(), weak_edgelist_srcs.size()),
          raft::device_span<vertex_t>(weak_edgelist_dsts.data(), weak_edgelist_dsts.size())});
      
      raft::print_device_vector("vertex_pair_buffer_p_q", std::get<0>(vertex_pair_buffer_p_q).data(), std::get<0>(vertex_pair_buffer_p_q).size(), std::cout);
      raft::print_device_vector("vertex_pair_buffer_p_q", std::get<1>(vertex_pair_buffer_p_q).data(), std::get<1>(vertex_pair_buffer_p_q).size(), std::cout);

      // Unroll (p, q) edges
      cugraph::edge_bucket_t<vertex_t, void, true, multi_gpu, true> invalid_edges_bucket(handle);
      invalid_edges_bucket.insert(weak_edgelist_srcs.begin(),
                                    weak_edgelist_srcs.end(),
                                    weak_edgelist_dsts.begin());

      printf("\nupdating count\n");
      cugraph::transform_e(
        handle,
        cur_graph_view,
        invalid_edges_bucket,
        cugraph::edge_src_dummy_property_t{}.view(),
        cugraph::edge_dst_dummy_property_t{}.view(),
        e_property_triangle_count.view(),
        [
          vertex_pair_buffer_p_q_begin = get_dataframe_buffer_begin(vertex_pair_buffer_p_q),
          vertex_pair_buffer_p_q_end = get_dataframe_buffer_end(vertex_pair_buffer_p_q)
        ]
        __device__(auto src, auto dst, thrust::nullopt_t, thrust::nullopt_t, edge_t count) {
          //printf("\nsrc = %d, dst = %d, count = %d\n", src, dst, count);
          auto e = thrust::make_tuple(src, dst);
          auto itr = thrust::lower_bound(
            thrust::seq, vertex_pair_buffer_p_q_begin, vertex_pair_buffer_p_q_end, e);
          
          if ((itr != vertex_pair_buffer_p_q_end) && (*itr == e)) {
            //printf("\nupdating the count - src = %d, dst = %d, count = %d\n", src, dst, count);
            return count - 1;
          } 
          
          return count;
    
        },
        e_property_triangle_count.mutable_view(),
        false);
    

      #if 0
      auto [srcs, dsts, count] = extract_transform_e(handle,
                                            cur_graph_view,
                                            cugraph::edge_src_dummy_property_t{}.view(),
                                            cugraph::edge_dst_dummy_property_t{}.view(),
                                            //view_concat(e_property_triangle_count.view(), modified_triangle_count.view()),
                                            e_property_triangle_count.view(),
                                            extract_edges<vertex_t, edge_t>{});

      printf("\nafter unrolling (p, q) edges\n");
      raft::print_device_vector("srcs", srcs.data(), srcs.size(), std::cout);
      raft::print_device_vector("dsts", dsts.data(), dsts.size(), std::cout);
      raft::print_device_vector("count", count.data(), count.size(), std::cout);
      #endif

      auto vertex_pair_buffer_p_r_edge_p_q =
          allocate_dataframe_buffer<thrust::tuple<vertex_t, vertex_t>>(intersection_indices.size(),
                                                                       handle.get_stream());
      thrust::tabulate(
        handle.get_thrust_policy(),
        get_dataframe_buffer_begin(vertex_pair_buffer_p_r_edge_p_q),
        get_dataframe_buffer_end(vertex_pair_buffer_p_r_edge_p_q),
        generate_p_r_or_q_r_from_p_q<vertex_t, edge_t, true>{
          prev_chunk_size,
          raft::device_span<size_t const>(intersection_offsets.data(),
                                          intersection_offsets.size()),
          raft::device_span<vertex_t const>(intersection_indices.data(),
                                            intersection_indices.size()),
          raft::device_span<vertex_t>(weak_edgelist_srcs.data(), weak_edgelist_srcs.size()),
          raft::device_span<vertex_t>(weak_edgelist_dsts.data(), weak_edgelist_dsts.data())});
      
      //invalid_edges
      cugraph::transform_e(
        handle,
        cur_graph_view,
        cugraph::edge_src_dummy_property_t{}.view(),
        cugraph::edge_dst_dummy_property_t{}.view(),
        e_property_triangle_count.view(),
        [
          vertex_pair_buffer_p_r_edge_p_q_begin = get_dataframe_buffer_begin(vertex_pair_buffer_p_r_edge_p_q),
          vertex_pair_buffer_p_r_edge_p_q_end = get_dataframe_buffer_end(vertex_pair_buffer_p_r_edge_p_q)
        ]
        __device__(auto src, auto dst, thrust::nullopt_t, thrust::nullopt_t, edge_t count) {
          //printf("\nsrc = %d, dst = %d, count = %d\n", src, dst, count);
          auto e = thrust::make_tuple(src, dst);
          auto itr = thrust::lower_bound(
            thrust::seq, vertex_pair_buffer_p_r_edge_p_q_begin, vertex_pair_buffer_p_r_edge_p_q_end, e);
          
          if ((itr != vertex_pair_buffer_p_r_edge_p_q_end) && (*itr == e)) {
            //printf("\nupdating the count - src = %d, dst = %d, count = %d\n", src, dst, count);
            return count - 1;
          } 
          
          return count;
    
        },
        e_property_triangle_count.mutable_view(),
        false);

      

      auto vertex_pair_buffer_q_r_edge_p_q =
          allocate_dataframe_buffer<thrust::tuple<vertex_t, vertex_t>>(intersection_indices.size(),
                                                                       handle.get_stream());
      thrust::tabulate(
        handle.get_thrust_policy(),
        get_dataframe_buffer_begin(vertex_pair_buffer_q_r_edge_p_q),
        get_dataframe_buffer_end(vertex_pair_buffer_q_r_edge_p_q),
        generate_p_r_or_q_r_from_p_q<vertex_t, edge_t, false>{
          prev_chunk_size,
          raft::device_span<size_t const>(intersection_offsets.data(),
                                          intersection_offsets.size()),
          raft::device_span<vertex_t const>(intersection_indices.data(),
                                            intersection_indices.size()),
          raft::device_span<vertex_t>(weak_edgelist_srcs.data(), weak_edgelist_srcs.size()),
          raft::device_span<vertex_t>(weak_edgelist_dsts.data(), weak_edgelist_dsts.data())});
      
      // invalid_edges
      cugraph::transform_e(
        handle,
        cur_graph_view,
        cugraph::edge_src_dummy_property_t{}.view(),
        cugraph::edge_dst_dummy_property_t{}.view(),
        e_property_triangle_count.view(),
        [
          vertex_pair_buffer_q_r_edge_p_q_begin = get_dataframe_buffer_begin(vertex_pair_buffer_q_r_edge_p_q),
          vertex_pair_buffer_q_r_edge_p_q_end = get_dataframe_buffer_end(vertex_pair_buffer_q_r_edge_p_q)
        ]
        __device__(auto src, auto dst, thrust::nullopt_t, thrust::nullopt_t, edge_t count) {
          //printf("\nsrc = %d, dst = %d, count = %d\n", src, dst, count);
          auto e = thrust::make_tuple(src, dst);
          auto itr = thrust::lower_bound(
            thrust::seq, vertex_pair_buffer_q_r_edge_p_q_begin, vertex_pair_buffer_q_r_edge_p_q_end, e);
          
          if ((itr != vertex_pair_buffer_q_r_edge_p_q_end) && (*itr == e)) {
            //printf("\nupdating the count - src = %d, dst = %d, count = %d\n", src, dst, count);
            return count - 1;
          } 
          
          return count;
    
        },
        e_property_triangle_count.mutable_view(),
        false);
  




    
      auto [srcs_, dsts_, count_] = extract_transform_e(handle,
                                            cur_graph_view,
                                            cugraph::edge_src_dummy_property_t{}.view(),
                                            cugraph::edge_dst_dummy_property_t{}.view(),
                                            e_property_triangle_count.view(),
                                            extract_edges<vertex_t, edge_t>{});
      
      printf("\nafter unrolling (p, q) edges from (p, r)\n");
      raft::print_device_vector("srcs", srcs_.data(), srcs_.size(), std::cout);
      raft::print_device_vector("dsts", dsts_.data(), dsts_.size(), std::cout);
      raft::print_device_vector("count", count_.data(), count_.size(), std::cout);
      

    
      prev_chunk_size += chunk_size;
      chunk_num_invalid_edges -= chunk_size;
    }
    */
  

    // case 2: unroll (q, r)
    // temporarily mask (p, q) edges
    /*
    cugraph::edge_bucket_t<vertex_t, void, true, multi_gpu, true> edges_to_tmp_mask(handle);
      edges_to_tmp_mask.insert(weak_edgelist_srcs.begin(),
                                    weak_edgelist_srcs.end(),
                                    weak_edgelist_dsts.begin());

    cugraph::transform_e(
          handle,
          cur_graph_view,
          edges_to_tmp_mask,
          cugraph::edge_src_dummy_property_t{}.view(),
          cugraph::edge_dst_dummy_property_t{}.view(),
          cugraph::edge_dummy_property_t{}.view(),
          [] __device__(auto src, auto dst, thrust::nullopt_t, thrust::nullopt_t, auto wgt) {
            return false;
          },
          tmp_edge_mask.mutable_view(),
          false);
    
    cur_graph_view.attach_edge_mask(tmp_edge_mask.view());
    */
  

    // FIXME: memory footprint overhead
    rmm::device_uvector<vertex_t> vertex_q_r(weak_edgelist_srcs.size() * 2, handle.get_stream());
      
    // Iterate over unique vertices that appear as either q or r
    thrust::merge(handle.get_thrust_policy(),
                  weak_edgelist_srcs.begin(),
                  weak_edgelist_srcs.end(),
                  weak_edgelist_dsts.begin(),
                  weak_edgelist_dsts.end(),
                  vertex_q_r.begin());
    thrust::sort(handle.get_thrust_policy(), vertex_q_r.begin(), vertex_q_r.end());
    auto invalid_unique_v_end =  thrust::unique(
                                    handle.get_thrust_policy(),
                                    vertex_q_r.begin(),
                                    vertex_q_r.end());
    
    vertex_q_r.resize(thrust::distance(vertex_q_r.begin(), invalid_unique_v_end), handle.get_stream());

    auto invalid_edgelist = thrust::make_zip_iterator(weak_edgelist_srcs.begin(), weak_edgelist_dsts.begin());
    // raft::device_span<vertex_t const>(vertex_q_r.data(), vertex_q_r.size())
    auto [srcs_to_q_r, dsts_to_q_r] = extract_transform_e(handle,
                                              cur_graph_view,
                                              cugraph::edge_src_dummy_property_t{}.view(),
                                              cugraph::edge_dst_dummy_property_t{}.view(),
                                              //view_concat(e_property_triangle_count.view(), modified_triangle_count.view()),
                                              cugraph::edge_dummy_property_t{}.view(),
                                              extract_edges_to_q_r<vertex_t>{raft::device_span<vertex_t const>(vertex_q_r.data(), vertex_q_r.size())});

    printf("\nunrolling q, r edges\n");
    raft::print_device_vector("srcs", srcs_to_q_r.data(), srcs_to_q_r.size(), std::cout);
    raft::print_device_vector("dsts", dsts_to_q_r.data(), dsts_to_q_r.size(), std::cout);

    std::optional<graph_t<vertex_t, edge_t, false, multi_gpu>> graph_q_r{std::nullopt};
        std::optional<rmm::device_uvector<vertex_t>> renumber_map_q_r{std::nullopt};
        std::tie(*graph_q_r, std::ignore, std::ignore, std::ignore, renumber_map_q_r) =
          create_graph_from_edgelist<vertex_t, edge_t, weight_t, edge_t, int32_t, false, multi_gpu>(
            handle,
            std::nullopt,
            std::move(dsts_to_q_r),
            std::move(srcs_to_q_r),
            std::nullopt,
            std::nullopt,
            std::nullopt,
            cugraph::graph_properties_t{true, graph_view.is_multigraph()},
            false);
    
    auto [srcs__, dsts__] = extract_transform_e(handle,
                                              (*graph_q_r).view(),
                                              cugraph::edge_src_dummy_property_t{}.view(),
                                              cugraph::edge_dst_dummy_property_t{}.view(),
                                              //view_concat(e_property_triangle_count.view(), modified_triangle_count.view()),
                                              cugraph::edge_dummy_property_t{}.view(),
                                              extract_edges_to_q_r<vertex_t>{raft::device_span<vertex_t const>(vertex_q_r.data(), vertex_q_r.size())});

    printf("\nq, r edge graph\n");
    raft::print_device_vector("srcs__", srcs__.data(), srcs__.size(), std::cout);
    raft::print_device_vector("dsts__", dsts__.data(), dsts__.size(), std::cout);
    

    // *********************************************************************************************************
    printf("\nbefore crash\n");
    raft::print_device_vector("srcs", weak_edgelist_srcs.data(), weak_edgelist_srcs.size(), std::cout);
    raft::print_device_vector("dsts", weak_edgelist_dsts.data(), weak_edgelist_srcs.size(), std::cout);
    
    prev_chunk_size         = 0;
    chunk_num_invalid_edges = weak_edgelist_srcs.size();

    num_chunks =
      raft::div_rounding_up_safe(weak_edgelist_srcs.size(), edges_to_intersect_per_iteration);
    
    for (size_t i = 0; i < num_chunks; ++i) {
      
      auto chunk_size = std::min(edges_to_intersect_per_iteration, chunk_num_invalid_edges);
      auto [intersection_offsets, intersection_indices] =
        detail::nbr_intersection(handle,
                                (*graph_q_r).view(),
                                //cur_graph_view,
                                cugraph::edge_dummy_property_t{}.view(),
                                invalid_edge_first + prev_chunk_size,
                                invalid_edge_first + prev_chunk_size + chunk_size,
                                //thrust::make_zip_iterator(weak_edgelist_srcs.begin(), weak_edgelist_dsts.begin()),
                                //thrust::make_zip_iterator(weak_edgelist_srcs.end(), weak_edgelist_dsts.end()),
                                std::array<bool, 2>{true, true},
                                true);

      // clear mask. 
      cur_graph_view.clear_edge_mask();

      printf("\n**********intersection when unrolling q, r edges\n");
      raft::print_device_vector("intersection_offsets", intersection_offsets.data(), intersection_offsets.size(), std::cout);
      raft::print_device_vector("intersection_indices", intersection_indices.data(), intersection_indices.size(), std::cout);
      
      // Generate (p, q) edges
      // FIXME: Should this array be reduced? an edge can have an intersection size  > 1
      auto vertex_pair_buffer_p_q =
        allocate_dataframe_buffer<thrust::tuple<vertex_t, vertex_t>>(intersection_indices.size(),
                                                                     handle.get_stream());

      thrust::tabulate(
        handle.get_thrust_policy(),
        get_dataframe_buffer_begin(vertex_pair_buffer_p_q),
        get_dataframe_buffer_end(vertex_pair_buffer_p_q),
        generate_p_q<vertex_t, edge_t>{
          prev_chunk_size,
          raft::device_span<size_t const>(intersection_offsets.data(),
                                          intersection_offsets.size()),
          raft::device_span<vertex_t const>(intersection_indices.data(),
                                            intersection_indices.size()),
          raft::device_span<vertex_t const>(weak_edgelist_srcs.data(), weak_edgelist_srcs.size()),
          raft::device_span<vertex_t const>(weak_edgelist_dsts.data(), weak_edgelist_dsts.size())});
      
      raft::print_device_vector("vertex_pair_buffer_p_q", std::get<0>(vertex_pair_buffer_p_q).data(), std::get<0>(vertex_pair_buffer_p_q).size(), std::cout);
      raft::print_device_vector("vertex_pair_buffer_p_q", std::get<1>(vertex_pair_buffer_p_q).data(), std::get<1>(vertex_pair_buffer_p_q).size(), std::cout);

      // Unroll (p, q) edges
      cugraph::edge_bucket_t<vertex_t, void, true, multi_gpu, true> invalid_edges_bucket(handle);
      invalid_edges_bucket.insert(weak_edgelist_srcs.begin(),
                                  weak_edgelist_srcs.end(),
                                  weak_edgelist_dsts.begin());

      printf("\nupdating count\n");
      cugraph::transform_e(
        handle,
        cur_graph_view,
        invalid_edges_bucket,
        cugraph::edge_src_dummy_property_t{}.view(),
        cugraph::edge_dst_dummy_property_t{}.view(),
        e_property_triangle_count.view(),
        [
          vertex_pair_buffer_p_q_begin = get_dataframe_buffer_begin(vertex_pair_buffer_p_q),
          vertex_pair_buffer_p_q_end = get_dataframe_buffer_end(vertex_pair_buffer_p_q)
        ]
        __device__(auto src, auto dst, thrust::nullopt_t, thrust::nullopt_t, edge_t count) {
          //printf("\nsrc = %d, dst = %d, count = %d\n", src, dst, count);
          auto e = thrust::make_tuple(src, dst);
          auto itr = thrust::lower_bound(
            thrust::seq, vertex_pair_buffer_p_q_begin, vertex_pair_buffer_p_q_end, e);
          
          if ((itr != vertex_pair_buffer_p_q_end) && (*itr == e)) {
            //printf("\nupdating the count - src = %d, dst = %d, count = %d\n", src, dst, count);
            return count - 1;
          } 
          
          return count;
    
        },
        e_property_triangle_count.mutable_view(),
        false);
    

      //#if 0
      auto [srcs, dsts, count] = extract_transform_e(handle,
                                            cur_graph_view,
                                            cugraph::edge_src_dummy_property_t{}.view(),
                                            cugraph::edge_dst_dummy_property_t{}.view(),
                                            //view_concat(e_property_triangle_count.view(), modified_triangle_count.view()),
                                            e_property_triangle_count.view(),
                                            extract_edges<vertex_t, edge_t>{});

      //printf("\nafter unrolling (p, q) edges\n");
      //raft::print_device_vector("srcs", srcs.data(), srcs.size(), std::cout);
      //raft::print_device_vector("dsts", dsts.data(), dsts.size(), std::cout);
      //raft::print_device_vector("count", count.data(), count.size(), std::cout);
      //#endif

      auto vertex_pair_buffer_p_r_edge_p_q =
          allocate_dataframe_buffer<thrust::tuple<vertex_t, vertex_t>>(intersection_indices.size(),
                                                                       handle.get_stream());
      thrust::tabulate(
        handle.get_thrust_policy(),
        get_dataframe_buffer_begin(vertex_pair_buffer_p_r_edge_p_q),
        get_dataframe_buffer_end(vertex_pair_buffer_p_r_edge_p_q),
        generate_p_r_or_q_r_from_p_q<vertex_t, edge_t, true>{
          prev_chunk_size,
          raft::device_span<size_t const>(intersection_offsets.data(),
                                          intersection_offsets.size()),
          raft::device_span<vertex_t const>(intersection_indices.data(),
                                            intersection_indices.size()),
          raft::device_span<vertex_t>(weak_edgelist_srcs.data(), weak_edgelist_srcs.size()),
          raft::device_span<vertex_t>(weak_edgelist_dsts.data(), weak_edgelist_dsts.data())});
      
      //invalid_edges
      cugraph::transform_e(
        handle,
        cur_graph_view,
        cugraph::edge_src_dummy_property_t{}.view(),
        cugraph::edge_dst_dummy_property_t{}.view(),
        e_property_triangle_count.view(),
        [
          vertex_pair_buffer_p_r_edge_p_q_begin = get_dataframe_buffer_begin(vertex_pair_buffer_p_r_edge_p_q),
          vertex_pair_buffer_p_r_edge_p_q_end = get_dataframe_buffer_end(vertex_pair_buffer_p_r_edge_p_q)
        ]
        __device__(auto src, auto dst, thrust::nullopt_t, thrust::nullopt_t, edge_t count) {
          //printf("\nsrc = %d, dst = %d, count = %d\n", src, dst, count);
          auto e = thrust::make_tuple(src, dst);
          auto itr = thrust::lower_bound(
            thrust::seq, vertex_pair_buffer_p_r_edge_p_q_begin, vertex_pair_buffer_p_r_edge_p_q_end, e);
          
          if ((itr != vertex_pair_buffer_p_r_edge_p_q_end) && (*itr == e)) {
            //printf("\nupdating the count - src = %d, dst = %d, count = %d\n", src, dst, count);
            return count - 1;
          } 
          
          return count;
    
        },
        e_property_triangle_count.mutable_view(),
        false);

      

      auto vertex_pair_buffer_q_r_edge_p_q =
          allocate_dataframe_buffer<thrust::tuple<vertex_t, vertex_t>>(intersection_indices.size(),
                                                                       handle.get_stream());
      thrust::tabulate(
        handle.get_thrust_policy(),
        get_dataframe_buffer_begin(vertex_pair_buffer_q_r_edge_p_q),
        get_dataframe_buffer_end(vertex_pair_buffer_q_r_edge_p_q),
        generate_p_r_or_q_r_from_p_q<vertex_t, edge_t, false>{
          prev_chunk_size,
          raft::device_span<size_t const>(intersection_offsets.data(),
                                          intersection_offsets.size()),
          raft::device_span<vertex_t const>(intersection_indices.data(),
                                            intersection_indices.size()),
          raft::device_span<vertex_t>(weak_edgelist_srcs.data(), weak_edgelist_srcs.size()),
          raft::device_span<vertex_t>(weak_edgelist_dsts.data(), weak_edgelist_dsts.data())});
      
      // invalid_edges
      cugraph::transform_e(
        handle,
        cur_graph_view,
        cugraph::edge_src_dummy_property_t{}.view(),
        cugraph::edge_dst_dummy_property_t{}.view(),
        e_property_triangle_count.view(),
        [
          vertex_pair_buffer_q_r_edge_p_q_begin = get_dataframe_buffer_begin(vertex_pair_buffer_q_r_edge_p_q),
          vertex_pair_buffer_q_r_edge_p_q_end = get_dataframe_buffer_end(vertex_pair_buffer_q_r_edge_p_q)
        ]
        __device__(auto src, auto dst, thrust::nullopt_t, thrust::nullopt_t, edge_t count) {
          //printf("\nsrc = %d, dst = %d, count = %d\n", src, dst, count);
          auto e = thrust::make_tuple(src, dst);
          auto itr = thrust::lower_bound(
            thrust::seq, vertex_pair_buffer_q_r_edge_p_q_begin, vertex_pair_buffer_q_r_edge_p_q_end, e);
          
          if ((itr != vertex_pair_buffer_q_r_edge_p_q_end) && (*itr == e)) {
            //printf("\nupdating the count - src = %d, dst = %d, count = %d\n", src, dst, count);
            return count - 1;
          } 
          
          return count;
    
        },
        e_property_triangle_count.mutable_view(),
        false);
    
      prev_chunk_size += chunk_size;
      chunk_num_invalid_edges -= chunk_size;



      printf("\nafter unrolling (p, q) edges\n");
      raft::print_device_vector("srcs", srcs.data(), srcs.size(), std::cout);
      raft::print_device_vector("dsts", dsts.data(), dsts.size(), std::cout);
      raft::print_device_vector("count", count.data(), count.size(), std::cout);
    }
    // ****************************************************************************


    // Unrolling p, r edges
    // create pair invalid_src, invalid_edge_idx
    // create a dataframe buffer of size invalid_edge_size
    // FIXME: No need to create a dataframe buffer. We can just zip weak_edgelist_srcs
    // with a vector counting from 0 .. 
    auto vertex_pair_buffer_p_tag =
      allocate_dataframe_buffer<thrust::tuple<vertex_t, edge_t>>(weak_edgelist_srcs.size(),
                                                                   handle.get_stream());
    
    thrust::tabulate(
          handle.get_thrust_policy(),
          get_dataframe_buffer_begin(vertex_pair_buffer_p_tag),
          get_dataframe_buffer_end(vertex_pair_buffer_p_tag),
          [
            p = weak_edgelist_srcs.begin()
          ] __device__(auto idx) {
            return thrust::make_tuple(p[idx], idx);
            });
    
    raft::print_device_vector("vertex_pair_buffer_p_tag", std::get<0>(vertex_pair_buffer_p_tag).data(), std::get<0>(vertex_pair_buffer_p_tag).size(), std::cout);
    raft::print_device_vector("vertex_pair_buffer_p_tag", std::get<1>(vertex_pair_buffer_p_tag).data(), std::get<1>(vertex_pair_buffer_p_tag).size(), std::cout);
    
    
    vertex_frontier_t<vertex_t, edge_t, multi_gpu, false> vertex_frontier(handle, 1);
    vertex_frontier.bucket(0).insert(
    thrust::make_zip_iterator(std::get<0>(vertex_pair_buffer_p_tag).begin(), std::get<1>(vertex_pair_buffer_p_tag).begin()),
    thrust::make_zip_iterator(std::get<0>(vertex_pair_buffer_p_tag).end(), std::get<1>(vertex_pair_buffer_p_tag).end()));

    printf("\nsize after inserting - part 1 = %d\n", vertex_frontier.bucket(0).size());

    /*
    std::tie(edge_majors, edge_minors, *edge_weights, subgraph_edge_graph_ids) =
      cugraph::extract_transform_v_frontier_outgoing_e(
        handle,
        cur_graph_view,
        vertex_frontier.bucket(0),
        cugraph::edge_src_dummy_property_t{}.view(),
        cugraph::edge_dst_dummy_property_t{}.view(),
        cugraph::edge_dummy_property_t{}.view(),
        [] __device__(
          thrust::tuple<vertex_t, edge_t> tagged_src,
          vertex_t dst,
          thrust::nullopt_t,
          thrust::nullopt_t,
          thrust::nullopt_t) {
            printf("\n dst = %d, tag = %d\n", dst, thrust::get<1>(tagged_src));

          },
        do_expensive_check);
    */

    auto [q, idx] =
      cugraph::extract_transform_v_frontier_outgoing_e(
        handle,
        cur_graph_view,
        vertex_frontier.bucket(0),
        cugraph::edge_src_dummy_property_t{}.view(),
        cugraph::edge_dst_dummy_property_t{}.view(),
        cugraph::edge_dummy_property_t{}.view(),
        extract_q_idx<vertex_t, edge_t>{},
        do_expensive_check);
    
    raft::print_device_vector("q", q.data(), q.size(), std::cout);
    raft::print_device_vector("i", idx.data(), idx.size(), std::cout);
    vertex_frontier.bucket(0).clear();
    printf("\nsize after clearning = %d\n", vertex_frontier.bucket(0).size());

    vertex_frontier.bucket(0).insert(
    thrust::make_zip_iterator(q.begin(), idx.begin()),
    thrust::make_zip_iterator(q.end(), idx.end()));

    printf("\nsize after inserting - part 2 = %d\n", vertex_frontier.bucket(0).size());
    // FIXME: Need to mask (p, q) and (q, r) edges before unrolling (p, r) edges to avoid overcompensating
    auto [q_closing, idx_closing] =
      cugraph::extract_transform_v_frontier_outgoing_e(
        handle,
        cur_graph_view,
        vertex_frontier.bucket(0),
        cugraph::edge_src_dummy_property_t{}.view(),
        cugraph::edge_dst_dummy_property_t{}.view(),
        cugraph::edge_dummy_property_t{}.view(),
        extract_q_idx_closing<vertex_t, edge_t>{raft::device_span<vertex_t>(weak_edgelist_dsts.data(), weak_edgelist_dsts.size())},
        do_expensive_check);
    
    raft::print_device_vector("q_closing", q_closing.data(), q_closing.size(), std::cout);
    raft::print_device_vector("i_closing", idx_closing.data(), idx_closing.size(), std::cout);

    
    // extract pair (p, r)
    auto vertex_pair_buffer_p_r =
      allocate_dataframe_buffer<thrust::tuple<vertex_t, vertex_t>>(q_closing.size(),
                                                                   handle.get_stream());
    // construct pair (p, q)
    // construct pair (q, r)
    thrust::tabulate(
        handle.get_thrust_policy(),
        get_dataframe_buffer_begin(vertex_pair_buffer_p_r),
        get_dataframe_buffer_end(vertex_pair_buffer_p_r),
        generate_p_r<vertex_t, edge_t, decltype(invalid_edgelist)>{
          invalid_edgelist,
          raft::device_span<edge_t const>(idx_closing.data(),
                                          idx_closing.size())
          });
    
    // construct pair (p, q)
    auto vertex_pair_buffer_p_q_for_p_r =
      allocate_dataframe_buffer<thrust::tuple<vertex_t, vertex_t>>(q_closing.size(),
                                                                   handle.get_stream());
    thrust::tabulate(
        handle.get_thrust_policy(),
        get_dataframe_buffer_begin(vertex_pair_buffer_p_q_for_p_r),
        get_dataframe_buffer_end(vertex_pair_buffer_p_q_for_p_r),
        generate_p_q_q_r<vertex_t, edge_t, decltype(invalid_edgelist), true>{
          invalid_edgelist,
          raft::device_span<edge_t const>(q_closing.data(),
                                          q_closing.size()),
          raft::device_span<edge_t const>(idx_closing.data(),
                                          idx_closing.size())
          });

    // construct pair (q, r)
    auto vertex_pair_buffer_q_r_for_p_r =
      allocate_dataframe_buffer<thrust::tuple<vertex_t, vertex_t>>(q_closing.size(),
                                                                   handle.get_stream());
    thrust::tabulate(
        handle.get_thrust_policy(),
        get_dataframe_buffer_begin(vertex_pair_buffer_q_r_for_p_r),
        get_dataframe_buffer_end(vertex_pair_buffer_q_r_for_p_r),
        generate_p_q_q_r<vertex_t, edge_t, decltype(invalid_edgelist), false>{
          invalid_edgelist,
          raft::device_span<edge_t const>(q_closing.data(),
                                          q_closing.size()),
          raft::device_span<edge_t const>(idx_closing.data(),
                                          idx_closing.size())
          });

    raft::print_device_vector("vertex_pair_buffer_p_r", std::get<0>(vertex_pair_buffer_p_r).data(), std::get<0>(vertex_pair_buffer_p_r).size(), std::cout);
    raft::print_device_vector("vertex_pair_buffer_p_r", std::get<1>(vertex_pair_buffer_p_r).data(), std::get<1>(vertex_pair_buffer_p_r).size(), std::cout);

    raft::print_device_vector("vertex_pair_buffer_p_q_for_p_r", std::get<0>(vertex_pair_buffer_p_q_for_p_r).data(), std::get<0>(vertex_pair_buffer_p_q_for_p_r).size(), std::cout);
    raft::print_device_vector("vertex_pair_buffer_p_q_for_p_r", std::get<1>(vertex_pair_buffer_p_q_for_p_r).data(), std::get<1>(vertex_pair_buffer_p_q_for_p_r).size(), std::cout);

    raft::print_device_vector("vertex_pair_buffer_q_r_for_p_r", std::get<0>(vertex_pair_buffer_q_r_for_p_r).data(), std::get<0>(vertex_pair_buffer_q_r_for_p_r).size(), std::cout);
    raft::print_device_vector("vertex_pair_buffer_q_r_for_p_r", std::get<1>(vertex_pair_buffer_q_r_for_p_r).data(), std::get<1>(vertex_pair_buffer_q_r_for_p_r).size(), std::cout);

    //rmm::device_uvector<vertex_t> weak_edgelist_srcs(0, handle.get_stream());
    //rmm::device_uvector<vertex_t> weak_edgelist_dsts(0, handle.get_stream());
    std::optional<rmm::device_uvector<weight_t>> edgelist_wgts{std::nullopt};
    return std::make_tuple(
      std::move(weak_edgelist_srcs), std::move(weak_edgelist_dsts), std::move(edgelist_wgts));
    
  
  
  
  }

}
}  // namespace cugraph
