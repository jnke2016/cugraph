/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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
#include <community/detail/common_methods.cuh>

namespace cugraph {
namespace detail {

template rmm::device_uvector<double>
cugraph::detail::lookup_primitive_values_for_keys<int32_t, double, true>(
  raft::handle_t const&,
  rmm::device_uvector<int32_t>&,
  rmm::device_uvector<double>&,
  rmm::device_uvector<int32_t>&);
template rmm::device_uvector<float>
cugraph::detail::lookup_primitive_values_for_keys<int32_t, float, true>(
  raft::handle_t const&,
  rmm::device_uvector<int32_t>&,
  rmm::device_uvector<float>&,
  rmm::device_uvector<int32_t>&);

template rmm::device_uvector<int32_t>
cugraph::detail::lookup_primitive_values_for_keys<int32_t, int32_t, true>(
  raft::handle_t const&,
  rmm::device_uvector<int32_t>&,
  rmm::device_uvector<int32_t>&,
  rmm::device_uvector<int32_t>&);

template rmm::device_uvector<double>
cugraph::detail::lookup_primitive_values_for_keys<int64_t, double, true>(
  raft::handle_t const&,
  rmm::device_uvector<int64_t>&,
  rmm::device_uvector<double>&,
  rmm::device_uvector<int64_t>&);

template rmm::device_uvector<float>
cugraph::detail::lookup_primitive_values_for_keys<int64_t, float, true>(
  raft::handle_t const&,
  rmm::device_uvector<int64_t>&,
  rmm::device_uvector<float>&,
  rmm::device_uvector<int64_t>&);

template rmm::device_uvector<int64_t>
cugraph::detail::lookup_primitive_values_for_keys<int64_t, int64_t, true>(
  raft::handle_t const&,
  rmm::device_uvector<int64_t>&,
  rmm::device_uvector<int64_t>&,
  rmm::device_uvector<int64_t>&);

template float compute_modularity(
  raft::handle_t const& handle,
  cugraph::graph_view_t<int32_t, int32_t, false, true> const& graph_view,
  std::optional<edge_property_view_t<int32_t, float const*>> edge_weight_view,
  edge_src_property_t<cugraph::graph_view_t<int32_t, int32_t, false, true>, int32_t> const&
    src_clusters_cache,
  edge_dst_property_t<cugraph::graph_view_t<int32_t, int32_t, false, true>, int32_t> const&
    dst_clusters_cache,
  rmm::device_uvector<int32_t> const& next_clusters,
  rmm::device_uvector<float> const& cluster_weights,
  float total_edge_weight,
  float resolution);

template float compute_modularity(
  raft::handle_t const& handle,
  cugraph::graph_view_t<int32_t, int64_t, false, true> const& graph_view,
  std::optional<edge_property_view_t<int64_t, float const*>> edge_weight_view,
  edge_src_property_t<cugraph::graph_view_t<int32_t, int64_t, false, true>, int32_t> const&
    src_clusters_cache,
  edge_dst_property_t<cugraph::graph_view_t<int32_t, int64_t, false, true>, int32_t> const&
    dst_clusters_cache,
  rmm::device_uvector<int32_t> const& next_clusters,
  rmm::device_uvector<float> const& cluster_weights,
  float total_edge_weight,
  float resolution);

template float compute_modularity(
  raft::handle_t const& handle,
  cugraph::graph_view_t<int64_t, int64_t, false, true> const& graph_view,
  std::optional<edge_property_view_t<int64_t, float const*>> edge_weight_view,
  edge_src_property_t<cugraph::graph_view_t<int64_t, int64_t, false, true>, int64_t> const&
    src_clusters_cache,
  edge_dst_property_t<cugraph::graph_view_t<int64_t, int64_t, false, true>, int64_t> const&
    dst_clusters_cache,
  rmm::device_uvector<int64_t> const& next_clusters,
  rmm::device_uvector<float> const& cluster_weights,
  float total_edge_weight,
  float resolution);

template double compute_modularity(
  raft::handle_t const& handle,
  cugraph::graph_view_t<int32_t, int32_t, false, true> const& graph_view,
  std::optional<edge_property_view_t<int32_t, double const*>> edge_weight_view,
  edge_src_property_t<cugraph::graph_view_t<int32_t, int32_t, false, true>, int32_t> const&
    src_clusters_cache,
  edge_dst_property_t<cugraph::graph_view_t<int32_t, int32_t, false, true>, int32_t> const&
    dst_clusters_cache,
  rmm::device_uvector<int32_t> const& next_clusters,
  rmm::device_uvector<double> const& cluster_weights,
  double total_edge_weight,
  double resolution);

template double compute_modularity(
  raft::handle_t const& handle,
  cugraph::graph_view_t<int32_t, int64_t, false, true> const& graph_view,
  std::optional<edge_property_view_t<int64_t, double const*>> edge_weight_view,
  edge_src_property_t<cugraph::graph_view_t<int32_t, int64_t, false, true>, int32_t> const&
    src_clusters_cache,
  edge_dst_property_t<cugraph::graph_view_t<int32_t, int64_t, false, true>, int32_t> const&
    dst_clusters_cache,
  rmm::device_uvector<int32_t> const& next_clusters,
  rmm::device_uvector<double> const& cluster_weights,
  double total_edge_weight,
  double resolution);

template double compute_modularity(
  raft::handle_t const& handle,
  cugraph::graph_view_t<int64_t, int64_t, false, true> const& graph_view,
  std::optional<edge_property_view_t<int64_t, double const*>> edge_weight_view,
  edge_src_property_t<cugraph::graph_view_t<int64_t, int64_t, false, true>, int64_t> const&
    src_clusters_cache,
  edge_dst_property_t<cugraph::graph_view_t<int64_t, int64_t, false, true>, int64_t> const&
    dst_clusters_cache,
  rmm::device_uvector<int64_t> const& next_clusters,
  rmm::device_uvector<double> const& cluster_weights,
  double total_edge_weight,
  double resolution);

template std::tuple<
  cugraph::graph_t<int32_t, int32_t, false, true>,
  std::optional<edge_property_t<graph_view_t<int32_t, int32_t, false, true>, float>>,
  std::optional<rmm::device_uvector<int32_t>>>
graph_contraction(raft::handle_t const& handle,
                  cugraph::graph_view_t<int32_t, int32_t, false, true> const& graph_view,
                  std::optional<edge_property_view_t<int32_t, float const*>> edge_weights,
                  raft::device_span<int32_t> labels);

template std::tuple<
  cugraph::graph_t<int32_t, int64_t, false, true>,
  std::optional<edge_property_t<graph_view_t<int32_t, int64_t, false, true>, float>>,
  std::optional<rmm::device_uvector<int32_t>>>
graph_contraction(raft::handle_t const& handle,
                  cugraph::graph_view_t<int32_t, int64_t, false, true> const& graph_view,
                  std::optional<edge_property_view_t<int64_t, float const*>> edge_weights,
                  raft::device_span<int32_t> labels);

template std::tuple<
  cugraph::graph_t<int64_t, int64_t, false, true>,
  std::optional<edge_property_t<graph_view_t<int64_t, int64_t, false, true>, float>>,
  std::optional<rmm::device_uvector<int64_t>>>
graph_contraction(raft::handle_t const& handle,
                  cugraph::graph_view_t<int64_t, int64_t, false, true> const& graph_view,
                  std::optional<edge_property_view_t<int64_t, float const*>> edge_weights,
                  raft::device_span<int64_t> labels);

template std::tuple<
  cugraph::graph_t<int32_t, int32_t, false, true>,
  std::optional<edge_property_t<graph_view_t<int32_t, int32_t, false, true>, double>>,
  std::optional<rmm::device_uvector<int32_t>>>
graph_contraction(raft::handle_t const& handle,
                  cugraph::graph_view_t<int32_t, int32_t, false, true> const& graph_view,
                  std::optional<edge_property_view_t<int32_t, double const*>> edge_weights,
                  raft::device_span<int32_t> labels);

template std::tuple<
  cugraph::graph_t<int32_t, int64_t, false, true>,
  std::optional<edge_property_t<graph_view_t<int32_t, int64_t, false, true>, double>>,
  std::optional<rmm::device_uvector<int32_t>>>
graph_contraction(raft::handle_t const& handle,
                  cugraph::graph_view_t<int32_t, int64_t, false, true> const& graph_view,
                  std::optional<edge_property_view_t<int64_t, double const*>> edge_weights,
                  raft::device_span<int32_t> labels);

template std::tuple<
  cugraph::graph_t<int64_t, int64_t, false, true>,
  std::optional<edge_property_t<graph_view_t<int64_t, int64_t, false, true>, double>>,
  std::optional<rmm::device_uvector<int64_t>>>
graph_contraction(raft::handle_t const& handle,
                  cugraph::graph_view_t<int64_t, int64_t, false, true> const& graph_view,
                  std::optional<edge_property_view_t<int64_t, double const*>> edge_weights,
                  raft::device_span<int64_t> labels);

template rmm::device_uvector<int32_t> update_clustering_by_delta_modularity(
  raft::handle_t const& handle,
  cugraph::graph_view_t<int32_t, int32_t, false, true> const& graph_view,
  std::optional<edge_property_view_t<int32_t, float const*>> edge_weight_view,
  float total_edge_weight,
  float resolution,
  rmm::device_uvector<float> const& vertex_weights_v,
  rmm::device_uvector<int32_t>&& cluster_keys_v,
  rmm::device_uvector<float>&& cluster_weights_v,
  rmm::device_uvector<int32_t>&& next_clusters_v,
  edge_src_property_t<cugraph::graph_view_t<int32_t, int32_t, false, true>, float> const&
    src_vertex_weights_cache,
  edge_src_property_t<cugraph::graph_view_t<int32_t, int32_t, false, true>, int32_t> const&
    src_clusters_cache,
  edge_dst_property_t<cugraph::graph_view_t<int32_t, int32_t, false, true>, int32_t> const&
    dst_clusters_cache,
  bool up_down);

template rmm::device_uvector<int32_t> update_clustering_by_delta_modularity(
  raft::handle_t const& handle,
  cugraph::graph_view_t<int32_t, int64_t, false, true> const& graph_view,
  std::optional<edge_property_view_t<int64_t, float const*>> edge_weight_view,
  float total_edge_weight,
  float resolution,
  rmm::device_uvector<float> const& vertex_weights_v,
  rmm::device_uvector<int32_t>&& cluster_keys_v,
  rmm::device_uvector<float>&& cluster_weights_v,
  rmm::device_uvector<int32_t>&& next_clusters_v,
  edge_src_property_t<cugraph::graph_view_t<int32_t, int64_t, false, true>, float> const&
    src_vertex_weights_cache,
  edge_src_property_t<cugraph::graph_view_t<int32_t, int64_t, false, true>, int32_t> const&
    src_clusters_cache,
  edge_dst_property_t<cugraph::graph_view_t<int32_t, int64_t, false, true>, int32_t> const&
    dst_clusters_cache,
  bool up_down);

template rmm::device_uvector<int64_t> update_clustering_by_delta_modularity(
  raft::handle_t const& handle,
  cugraph::graph_view_t<int64_t, int64_t, false, true> const& graph_view,
  std::optional<edge_property_view_t<int64_t, float const*>> edge_weight_view,
  float total_edge_weight,
  float resolution,
  rmm::device_uvector<float> const& vertex_weights_v,
  rmm::device_uvector<int64_t>&& cluster_keys_v,
  rmm::device_uvector<float>&& cluster_weights_v,
  rmm::device_uvector<int64_t>&& next_clusters_v,
  edge_src_property_t<cugraph::graph_view_t<int64_t, int64_t, false, true>, float> const&
    src_vertex_weights_cache,
  edge_src_property_t<cugraph::graph_view_t<int64_t, int64_t, false, true>, int64_t> const&
    src_clusters_cache,
  edge_dst_property_t<cugraph::graph_view_t<int64_t, int64_t, false, true>, int64_t> const&
    dst_clusters_cache,
  bool up_down);

template rmm::device_uvector<int32_t> update_clustering_by_delta_modularity(
  raft::handle_t const& handle,
  cugraph::graph_view_t<int32_t, int32_t, false, true> const& graph_view,
  std::optional<edge_property_view_t<int32_t, double const*>> edge_weight_view,
  double total_edge_weight,
  double resolution,
  rmm::device_uvector<double> const& vertex_weights_v,
  rmm::device_uvector<int32_t>&& cluster_keys_v,
  rmm::device_uvector<double>&& cluster_weights_v,
  rmm::device_uvector<int32_t>&& next_clusters_v,
  edge_src_property_t<cugraph::graph_view_t<int32_t, int32_t, false, true>, double> const&
    src_vertex_weights_cache,
  edge_src_property_t<cugraph::graph_view_t<int32_t, int32_t, false, true>, int32_t> const&
    src_clusters_cache,
  edge_dst_property_t<cugraph::graph_view_t<int32_t, int32_t, false, true>, int32_t> const&
    dst_clusters_cache,
  bool up_down);

template rmm::device_uvector<int32_t> update_clustering_by_delta_modularity(
  raft::handle_t const& handle,
  cugraph::graph_view_t<int32_t, int64_t, false, true> const& graph_view,
  std::optional<edge_property_view_t<int64_t, double const*>> edge_weight_view,
  double total_edge_weight,
  double resolution,
  rmm::device_uvector<double> const& vertex_weights_v,
  rmm::device_uvector<int32_t>&& cluster_keys_v,
  rmm::device_uvector<double>&& cluster_weights_v,
  rmm::device_uvector<int32_t>&& next_clusters_v,
  edge_src_property_t<cugraph::graph_view_t<int32_t, int64_t, false, true>, double> const&
    src_vertex_weights_cache,
  edge_src_property_t<cugraph::graph_view_t<int32_t, int64_t, false, true>, int32_t> const&
    src_clusters_cache,
  edge_dst_property_t<cugraph::graph_view_t<int32_t, int64_t, false, true>, int32_t> const&
    dst_clusters_cache,
  bool up_down);

template rmm::device_uvector<int64_t> update_clustering_by_delta_modularity(
  raft::handle_t const& handle,
  cugraph::graph_view_t<int64_t, int64_t, false, true> const& graph_view,
  std::optional<edge_property_view_t<int64_t, double const*>> edge_weight_view,
  double total_edge_weight,
  double resolution,
  rmm::device_uvector<double> const& vertex_weights_v,
  rmm::device_uvector<int64_t>&& cluster_keys_v,
  rmm::device_uvector<double>&& cluster_weights_v,
  rmm::device_uvector<int64_t>&& next_clusters_v,
  edge_src_property_t<cugraph::graph_view_t<int64_t, int64_t, false, true>, double> const&
    src_vertex_weights_cache,
  edge_src_property_t<cugraph::graph_view_t<int64_t, int64_t, false, true>, int64_t> const&
    src_clusters_cache,
  edge_dst_property_t<cugraph::graph_view_t<int64_t, int64_t, false, true>, int64_t> const&
    dst_clusters_cache,
  bool up_down);

template std::tuple<rmm::device_uvector<int32_t>, rmm::device_uvector<float>>
compute_cluster_keys_and_values(
  raft::handle_t const& handle,
  cugraph::graph_view_t<int32_t, int32_t, false, true> const& graph_view,
  std::optional<edge_property_view_t<int32_t, float const*>> edge_weight_view,
  rmm::device_uvector<int32_t> const& next_clusters_v,
  edge_src_property_t<cugraph::graph_view_t<int32_t, int32_t, false, true>, int32_t> const&
    src_clusters_cache);

template std::tuple<rmm::device_uvector<int32_t>, rmm::device_uvector<float>>
compute_cluster_keys_and_values(
  raft::handle_t const& handle,
  cugraph::graph_view_t<int32_t, int64_t, false, true> const& graph_view,
  std::optional<edge_property_view_t<int64_t, float const*>> edge_weight_view,
  rmm::device_uvector<int32_t> const& next_clusters_v,
  edge_src_property_t<cugraph::graph_view_t<int32_t, int64_t, false, true>, int32_t> const&
    src_clusters_cache);

template std::tuple<rmm::device_uvector<int64_t>, rmm::device_uvector<float>>
compute_cluster_keys_and_values(
  raft::handle_t const& handle,
  cugraph::graph_view_t<int64_t, int64_t, false, true> const& graph_view,
  std::optional<edge_property_view_t<int64_t, float const*>> edge_weight_view,
  rmm::device_uvector<int64_t> const& next_clusters_v,
  edge_src_property_t<cugraph::graph_view_t<int64_t, int64_t, false, true>, int64_t> const&
    src_clusters_cache);

template std::tuple<rmm::device_uvector<int32_t>, rmm::device_uvector<double>>
compute_cluster_keys_and_values(
  raft::handle_t const& handle,
  cugraph::graph_view_t<int32_t, int32_t, false, true> const& graph_view,
  std::optional<edge_property_view_t<int32_t, double const*>> edge_weight_view,
  rmm::device_uvector<int32_t> const& next_clusters_v,
  edge_src_property_t<cugraph::graph_view_t<int32_t, int32_t, false, true>, int32_t> const&
    src_clusters_cache);

template std::tuple<rmm::device_uvector<int32_t>, rmm::device_uvector<double>>
compute_cluster_keys_and_values(
  raft::handle_t const& handle,
  cugraph::graph_view_t<int32_t, int64_t, false, true> const& graph_view,
  std::optional<edge_property_view_t<int64_t, double const*>> edge_weight_view,
  rmm::device_uvector<int32_t> const& next_clusters_v,
  edge_src_property_t<cugraph::graph_view_t<int32_t, int64_t, false, true>, int32_t> const&
    src_clusters_cache);

template std::tuple<rmm::device_uvector<int64_t>, rmm::device_uvector<double>>
compute_cluster_keys_and_values(
  raft::handle_t const& handle,
  cugraph::graph_view_t<int64_t, int64_t, false, true> const& graph_view,
  std::optional<edge_property_view_t<int64_t, double const*>> edge_weight_view,
  rmm::device_uvector<int64_t> const& next_clusters_v,
  edge_src_property_t<cugraph::graph_view_t<int64_t, int64_t, false, true>, int64_t> const&
    src_clusters_cache);

}  // namespace detail
}  // namespace cugraph
