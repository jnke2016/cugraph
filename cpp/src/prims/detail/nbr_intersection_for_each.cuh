/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cugraph/edge_partition_device_view.cuh>
#include <cugraph/edge_partition_edge_property_device_view.cuh>
#include <cugraph/utilities/packed_bool_utils.hpp>

#include <raft/util/cudart_utils.hpp>

#include <cuda/std/optional>
#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>

namespace cugraph {
namespace detail {

constexpr size_t intersection_kernel_block_size = 256;
constexpr size_t low_degree_threshold           = 32;
constexpr size_t mid_degree_threshold           = 1024;

template <typename edge_t>
__device__ __forceinline__ bool is_edge_unmasked(uint32_t const* edge_mask, edge_t offset)
{
  return edge_mask == nullptr ||
         static_cast<bool>(edge_mask[packed_bool_offset(offset)] & packed_bool_mask(offset));
}

// Thread-per-pair kernel: one thread handles one pair using binary search.
template <bool check_mask,
          typename vertex_t,
          typename edge_t,
          typename VertexPairIterator,
          typename IntersectionOp>
__global__ static void intersection_low_degree(
  edge_partition_device_view_t<vertex_t, edge_t, false> edge_partition,
  VertexPairIterator vertex_pair_first,
  size_t const* pair_index_first,
  size_t num_pairs,
  IntersectionOp intersection_op,
  uint32_t const* edge_mask)
{
  auto const tid = threadIdx.x + static_cast<size_t>(blockIdx.x) * blockDim.x;
  size_t idx     = tid;

  while (idx < num_pairs) {
    auto i    = pair_index_first[idx];
    auto pair = *(vertex_pair_first + i);
    auto p    = cuda::std::get<0>(pair);
    auto q    = cuda::std::get<1>(pair);

    auto p_idx            = edge_partition.major_offset_from_major_nocheck(p);
    edge_t local_offset_p = edge_partition.local_offset(p_idx);
    edge_t local_degree_p = edge_partition.local_degree(p_idx);

    auto q_idx            = edge_partition.major_offset_from_major_nocheck(q);
    edge_t local_offset_q = edge_partition.local_offset(q_idx);
    edge_t local_degree_q = edge_partition.local_degree(q_idx);

    auto indices = edge_partition.indices();

    auto pq_itr = thrust::lower_bound(
      thrust::seq, indices + local_offset_p, indices + local_offset_p + local_degree_p, q);
    edge_t pq_edge_offset = static_cast<edge_t>(pq_itr - indices);

    bool p_is_short    = (local_degree_p <= local_degree_q);
    auto short_offset  = p_is_short ? local_offset_p : local_offset_q;
    auto short_degree  = p_is_short ? local_degree_p : local_degree_q;
    auto long_offset   = p_is_short ? local_offset_q : local_offset_p;
    auto long_degree   = p_is_short ? local_degree_q : local_degree_p;

    for (edge_t si = 0; si < short_degree; ++si) {
      if constexpr (check_mask) {
        if (!is_edge_unmasked(edge_mask, short_offset + si)) continue;
      }
      auto r = indices[short_offset + si];

      edge_t lo = long_offset;
      edge_t hi = long_offset + long_degree;
      while (lo < hi) {
        auto mid = lo + (hi - lo) / 2;
        if (indices[mid] < r) lo = mid + 1; else hi = mid;
      }
      if (lo < long_offset + long_degree && indices[lo] == r) {
        if constexpr (check_mask) {
          if (!is_edge_unmasked(edge_mask, lo)) continue;
        }
        edge_t pr_offset = p_is_short ? (short_offset + si) : lo;
        edge_t qr_offset = p_is_short ? lo : (short_offset + si);
        intersection_op(p, q, r, pq_edge_offset, pr_offset, qr_offset);
      }
    }

    idx += static_cast<size_t>(gridDim.x) * blockDim.x;
  }
}

// Warp-per-pair kernel: 32 threads cooperate on one pair via parallel binary search.
template <bool check_mask,
          typename vertex_t,
          typename edge_t,
          typename VertexPairIterator,
          typename IntersectionOp>
__global__ static void intersection_mid_degree(
  edge_partition_device_view_t<vertex_t, edge_t, false> edge_partition,
  VertexPairIterator vertex_pair_first,
  size_t const* pair_index_first,
  size_t num_pairs,
  IntersectionOp intersection_op,
  uint32_t const* edge_mask)
{
  auto const tid     = threadIdx.x + static_cast<size_t>(blockIdx.x) * blockDim.x;
  auto const lane_id = static_cast<edge_t>(tid % raft::warp_size());
  size_t idx         = tid / raft::warp_size();

  while (idx < num_pairs) {
    auto i    = pair_index_first[idx];
    auto pair = *(vertex_pair_first + i);
    auto p    = cuda::std::get<0>(pair);
    auto q    = cuda::std::get<1>(pair);

    auto p_idx            = edge_partition.major_offset_from_major_nocheck(p);
    edge_t local_offset_p = edge_partition.local_offset(p_idx);
    edge_t local_degree_p = edge_partition.local_degree(p_idx);

    auto q_idx            = edge_partition.major_offset_from_major_nocheck(q);
    edge_t local_offset_q = edge_partition.local_offset(q_idx);
    edge_t local_degree_q = edge_partition.local_degree(q_idx);

    auto indices = edge_partition.indices();

    edge_t pq_edge_offset{};
    if (lane_id == 0) {
      auto pq_itr = thrust::lower_bound(
        thrust::seq, indices + local_offset_p, indices + local_offset_p + local_degree_p, q);
      pq_edge_offset = static_cast<edge_t>(pq_itr - indices);
    }
    pq_edge_offset = __shfl_sync(0xFFFFFFFF, pq_edge_offset, 0);

    bool p_is_short    = (local_degree_p <= local_degree_q);
    auto short_offset  = p_is_short ? local_offset_p : local_offset_q;
    auto short_degree  = p_is_short ? local_degree_p : local_degree_q;
    auto long_offset   = p_is_short ? local_offset_q : local_offset_p;
    auto long_degree   = p_is_short ? local_degree_q : local_degree_p;

    for (edge_t si = lane_id; si < short_degree; si += raft::warp_size()) {
      if constexpr (check_mask) {
        if (!is_edge_unmasked(edge_mask, short_offset + si)) continue;
      }
      auto r = indices[short_offset + si];

      edge_t lo = long_offset;
      edge_t hi = long_offset + long_degree;
      while (lo < hi) {
        auto mid = lo + (hi - lo) / 2;
        if (indices[mid] < r) lo = mid + 1; else hi = mid;
      }
      if (lo < long_offset + long_degree && indices[lo] == r) {
        if constexpr (check_mask) {
          if (!is_edge_unmasked(edge_mask, lo)) continue;
        }
        edge_t pr_offset = p_is_short ? (short_offset + si) : lo;
        edge_t qr_offset = p_is_short ? lo : (short_offset + si);
        intersection_op(p, q, r, pq_edge_offset, pr_offset, qr_offset);
      }
    }

    idx += static_cast<size_t>(gridDim.x) * (blockDim.x / raft::warp_size());
  }
}

// Block-per-pair kernel: an entire block cooperates on one pair via parallel binary search.
template <bool check_mask,
          typename vertex_t,
          typename edge_t,
          typename VertexPairIterator,
          typename IntersectionOp>
__global__ static void intersection_high_degree(
  edge_partition_device_view_t<vertex_t, edge_t, false> edge_partition,
  VertexPairIterator vertex_pair_first,
  size_t const* pair_index_first,
  size_t num_pairs,
  IntersectionOp intersection_op,
  uint32_t const* edge_mask)
{
  size_t idx = static_cast<size_t>(blockIdx.x);

  while (idx < num_pairs) {
    auto i    = pair_index_first[idx];
    auto pair = *(vertex_pair_first + i);
    auto p    = cuda::std::get<0>(pair);
    auto q    = cuda::std::get<1>(pair);

    auto p_idx            = edge_partition.major_offset_from_major_nocheck(p);
    edge_t local_offset_p = edge_partition.local_offset(p_idx);
    edge_t local_degree_p = edge_partition.local_degree(p_idx);

    auto q_idx            = edge_partition.major_offset_from_major_nocheck(q);
    edge_t local_offset_q = edge_partition.local_offset(q_idx);
    edge_t local_degree_q = edge_partition.local_degree(q_idx);

    auto indices = edge_partition.indices();

    __shared__ edge_t shared_pq_offset;
    if (threadIdx.x == 0) {
      auto pq_itr = thrust::lower_bound(
        thrust::seq, indices + local_offset_p, indices + local_offset_p + local_degree_p, q);
      shared_pq_offset = static_cast<edge_t>(pq_itr - indices);
    }
    __syncthreads();
    edge_t pq_edge_offset = shared_pq_offset;

    bool p_is_short    = (local_degree_p <= local_degree_q);
    auto short_offset  = p_is_short ? local_offset_p : local_offset_q;
    auto short_degree  = p_is_short ? local_degree_p : local_degree_q;
    auto long_offset   = p_is_short ? local_offset_q : local_offset_p;
    auto long_degree   = p_is_short ? local_degree_q : local_degree_p;

    for (edge_t si = static_cast<edge_t>(threadIdx.x); si < short_degree;
         si += static_cast<edge_t>(blockDim.x)) {
      if constexpr (check_mask) {
        if (!is_edge_unmasked(edge_mask, short_offset + si)) continue;
      }
      auto r = indices[short_offset + si];

      edge_t lo = long_offset;
      edge_t hi = long_offset + long_degree;
      while (lo < hi) {
        auto mid = lo + (hi - lo) / 2;
        if (indices[mid] < r) lo = mid + 1; else hi = mid;
      }
      if (lo < long_offset + long_degree && indices[lo] == r) {
        if constexpr (check_mask) {
          if (!is_edge_unmasked(edge_mask, lo)) continue;
        }
        edge_t pr_offset = p_is_short ? (short_offset + si) : lo;
        edge_t qr_offset = p_is_short ? lo : (short_offset + si);
        intersection_op(p, q, r, pq_edge_offset, pr_offset, qr_offset);
      }
    }

    idx += static_cast<size_t>(gridDim.x);
  }
}

}  // namespace detail
}  // namespace cugraph
