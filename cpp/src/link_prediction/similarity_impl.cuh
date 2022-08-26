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
#pragma once

#include <raft/handle.hpp>
#include <raft/core/span.hpp>

#include <optional>
#include <tuple>

namespace cugraph {
namespace detail {

template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu, typename functor_t>
std::tuple<rmm::device_uvector<vertex_t>, rmm::device_uvector<vertex_t>, rmm::device_uvector<weight_t>>
similarity(raft::handle_t const& handle,
           graph_view_t<vertex_t, edge_t, weight_t, false, multi_gpu> const& graph_view,
           std::optional<raft::device_span<vertex_t const>> first,
           std::optional<raft::device_span<vertex_t const>> second,
           bool use_weights,
           functor_t functor)
{
  CUGRAPH_FAIL("not implemented");

  // Implementation, using primitives, that computes:
  //   For use_weights == False:  cardinality of A intersect B
  //   For use_weights == True:   sum of minimum weight in A intersect B, sum of maximum weight in A intersect B
  //
  // Then use the functor to compute the score
  //
}

}  // namespace detail
}  // namespace cugraph
