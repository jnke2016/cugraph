/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "prims/detail/nbr_intersection_for_each.cuh"

#include <cugraph/edge_partition_device_view.cuh>
#include <cugraph/edge_partition_edge_property_device_view.cuh>
#include <cugraph/graph_view.hpp>
#include <cugraph/utilities/packed_bool_utils.hpp>

#include <raft/core/handle.hpp>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/for_each.h>
#include <thrust/functional.h>
#include <thrust/gather.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/scan.h>
#include <thrust/transform.h>

namespace cugraph {

namespace detail {

template <typename vertex_t, typename edge_t>
struct csr_to_pair_t {
  edge_t const* offsets;
  vertex_t const* indices;
  vertex_t num_vertices;

  __device__ cuda::std::tuple<vertex_t, vertex_t> operator()(edge_t edge_idx) const
  {
    auto src = static_cast<vertex_t>(
      thrust::upper_bound(thrust::seq, offsets, offsets + num_vertices + 1, edge_idx) - offsets - 1);
    auto dst = indices[edge_idx];
    return cuda::std::make_tuple(src, dst);
  }
};

template <typename edge_t>
struct edge_active_flag_t {
  uint32_t const* mask_ptr;
  __device__ bool operator()(edge_t e) const
  {
    return static_cast<bool>(mask_ptr[packed_bool_offset(e)] & packed_bool_mask(e));
  }
};

template <typename edge_t>
struct compute_min_degree_t {
  edge_t const* offsets;
  vertex_t const* indices;
  edge_t const* vertex_degrees;
  vertex_t num_vertices;

  __device__ edge_t operator()(edge_t edge_idx) const
  {
    auto src = static_cast<vertex_t>(
      thrust::upper_bound(thrust::seq, offsets, offsets + num_vertices + 1, edge_idx) - offsets - 1);
    auto dst = indices[edge_idx];
    return cuda::minimum<edge_t>{}(vertex_degrees[src], vertex_degrees[dst]);
  }
};

}  // namespace detail

/**
 * @brief 3-arg overload: iterate ALL edges in the graph (CSR-direct), compute
 * destination neighbor intersection for each, and go per
 * common neighbor.  SG only.
 */
template <typename GraphViewType, typename IntersectionOp>
void per_v_pair_dst_nbr_intersection_for_each(raft::handle_t const& handle,
                                              GraphViewType const& graph_view,
                                              IntersectionOp intersection_op,
                                              bool do_expensive_check = false)
{
  using vertex_t = typename GraphViewType::vertex_type;
  using edge_t   = typename GraphViewType::edge_type;

  static_assert(!GraphViewType::is_storage_transposed);
  static_assert(!GraphViewType::is_multi_gpu);

  auto edge_mask_view = graph_view.edge_mask_view();
  auto edge_partition =
    edge_partition_device_view_t<vertex_t, edge_t, false>(
      graph_view.local_edge_partition_view(size_t{0}));
  auto edge_partition_e_mask =
    edge_mask_view
      ? cuda::std::make_optional<
          detail::edge_partition_edge_property_device_view_t<edge_t, uint32_t const*, bool>>(
          *edge_mask_view, 0)
      : cuda::std::nullopt;

  auto stream = handle.get_stream();
  auto num_vertices = graph_view.number_of_vertices();
  auto num_edges    = edge_partition.number_of_edges();

  uint32_t const* mask_ptr = edge_partition_e_mask
    ? (*edge_partition_e_mask).value_first()
    : nullptr;

  bool const has_mask = (mask_ptr != nullptr);

  auto offsets_ptr = edge_partition.offsets();
  auto indices_ptr = edge_partition.indices();

  auto vertex_pair_first = thrust::make_transform_iterator(
    thrust::make_counting_iterator(edge_t{0}),
    detail::csr_to_pair_t<vertex_t, edge_t>{offsets_ptr, indices_ptr, static_cast<vertex_t>(num_vertices)});

  // Compute per-vertex degrees (masked when mask present).
  rmm::device_uvector<edge_t> vertex_degrees(0, stream);
  if (has_mask) {
    vertex_degrees = edge_partition.compute_local_degrees_with_mask(mask_ptr, stream);
  } else {
    vertex_degrees = edge_partition.compute_local_degrees(stream);
  }

  // Compute per-edge min-degree.
  rmm::device_uvector<edge_t> min_degrees(num_edges, stream);
  thrust::transform(handle.get_thrust_policy(),
                    thrust::make_counting_iterator(edge_t{0}),
                    thrust::make_counting_iterator(static_cast<edge_t>(num_edges)),
                    min_degrees.begin(),
                    detail::compute_min_degree_t<edge_t>{
                      offsets_ptr, indices_ptr, vertex_degrees.data(),
                      static_cast<vertex_t>(num_vertices)});

  vertex_degrees.resize(0, stream);
  vertex_degrees.shrink_to_fit(stream);

  // Bin edges by min-degree (only unmasked edges when mask present).
  auto min_deg_ptr = min_degrees.data();
  auto counting    = thrust::make_counting_iterator(size_t{0});

  auto is_active = [mask_ptr] __device__(size_t e) -> bool {
    if (mask_ptr == nullptr) return true;
    return static_cast<bool>(mask_ptr[packed_bool_offset(e)] & packed_bool_mask(e));
  };

  auto num_low = thrust::count_if(
    handle.get_thrust_policy(), counting, counting + num_edges,
    [min_deg_ptr, is_active] __device__(size_t i) {
      return is_active(i) && min_deg_ptr[i] < static_cast<edge_t>(detail::low_degree_threshold);
    });
  auto num_high = thrust::count_if(
    handle.get_thrust_policy(), counting, counting + num_edges,
    [min_deg_ptr, is_active] __device__(size_t i) {
      return is_active(i) && min_deg_ptr[i] >= static_cast<edge_t>(detail::mid_degree_threshold);
    });
  auto num_active = thrust::count_if(
    handle.get_thrust_policy(), counting, counting + num_edges, is_active);
  auto num_mid = num_active - num_low - num_high;

  rmm::device_uvector<size_t> low_indices(num_low, stream);
  rmm::device_uvector<size_t> mid_indices(num_mid, stream);
  rmm::device_uvector<size_t> high_indices(num_high, stream);

  if (num_low > 0) {
    thrust::copy_if(
      handle.get_thrust_policy(), counting, counting + num_edges, low_indices.data(),
      [min_deg_ptr, is_active] __device__(size_t i) {
        return is_active(i) && min_deg_ptr[i] < static_cast<edge_t>(detail::low_degree_threshold);
      });
  }
  if (num_mid > 0) {
    thrust::copy_if(
      handle.get_thrust_policy(), counting, counting + num_edges, mid_indices.data(),
      [min_deg_ptr, is_active] __device__(size_t i) {
        return is_active(i) &&
               min_deg_ptr[i] >= static_cast<edge_t>(detail::low_degree_threshold) &&
               min_deg_ptr[i] < static_cast<edge_t>(detail::mid_degree_threshold);
      });
  }
  if (num_high > 0) {
    thrust::copy_if(
      handle.get_thrust_policy(), counting, counting + num_edges, high_indices.data(),
      [min_deg_ptr, is_active] __device__(size_t i) {
        return is_active(i) && min_deg_ptr[i] >= static_cast<edge_t>(detail::mid_degree_threshold);
      });
  }

  auto max_grid_size = handle.get_device_properties().maxGridSize[0];

  // Launch low-degree kernel (thread-per-pair)
  if (num_low > 0) {
    raft::grid_1d_thread_t grid(
      num_low, detail::intersection_kernel_block_size, max_grid_size);
    if (has_mask) {
      detail::intersection_low_degree<true, vertex_t, edge_t>
        <<<grid.num_blocks, grid.block_size, 0, stream>>>(
          edge_partition, vertex_pair_first, low_indices.data(),
          static_cast<size_t>(num_low), intersection_op, mask_ptr);
    } else {
      detail::intersection_low_degree<false, vertex_t, edge_t>
        <<<grid.num_blocks, grid.block_size, 0, stream>>>(
          edge_partition, vertex_pair_first, low_indices.data(),
          static_cast<size_t>(num_low), intersection_op, mask_ptr);
    }
  }

  // Launch mid-degree kernel (warp-per-pair)
  if (num_mid > 0) {
    raft::grid_1d_warp_t grid(
      num_mid, detail::intersection_kernel_block_size, max_grid_size);
    if (has_mask) {
      detail::intersection_mid_degree<true, vertex_t, edge_t>
        <<<grid.num_blocks, grid.block_size, 0, stream>>>(
          edge_partition, vertex_pair_first, mid_indices.data(),
          static_cast<size_t>(num_mid), intersection_op, mask_ptr);
    } else {
      detail::intersection_mid_degree<false, vertex_t, edge_t>
        <<<grid.num_blocks, grid.block_size, 0, stream>>>(
          edge_partition, vertex_pair_first, mid_indices.data(),
          static_cast<size_t>(num_mid), intersection_op, mask_ptr);
    }
  }

  // Launch high-degree kernel (block-per-pair)
  if (num_high > 0) {
    raft::grid_1d_block_t grid(
      num_high, detail::intersection_kernel_block_size, max_grid_size);
    if (has_mask) {
      detail::intersection_high_degree<true, vertex_t, edge_t>
        <<<grid.num_blocks, grid.block_size, 0, stream>>>(
          edge_partition, vertex_pair_first, high_indices.data(),
          static_cast<size_t>(num_high), intersection_op, mask_ptr);
    } else {
      detail::intersection_high_degree<false, vertex_t, edge_t>
        <<<grid.num_blocks, grid.block_size, 0, stream>>>(
          edge_partition, vertex_pair_first, high_indices.data(),
          static_cast<size_t>(num_high), intersection_op, mask_ptr);
    }
  }
}

}  // namespace cugraph
