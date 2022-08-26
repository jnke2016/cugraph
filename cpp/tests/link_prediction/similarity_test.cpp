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
 * See the License for the specific language governin_from_mtxg permissions and
 * limitations under the License.
 */
#include <utilities/base_fixture.hpp>
#include <utilities/device_comm_wrapper.hpp>
#include <utilities/high_res_clock.h>
#include <utilities/test_graphs.hpp>
#include <utilities/test_utilities.hpp>
#include <utilities/thrust_wrapper.hpp>

#include <cugraph/algorithms.hpp>

#include <link_prediction/similarity_compare.hpp>

#include <gtest/gtest.h>

struct Similarity_Usecase {
  bool use_weights{false};
  bool check_correctness{true};
};

template <typename input_usecase_t>
class Tests_Similarity
  : public ::testing::TestWithParam<std::tuple<Similarity_Usecase, input_usecase_t>> {
 public:
  Tests_Similarity() {}

  static void SetUpTestCase() {}
  static void TearDownTestCase() {}

  virtual void SetUp() {}
  virtual void TearDown() {}

  template <typename vertex_t, typename edge_t, typename weight_t, typename test_functor_t>
  void run_current_test(std::tuple<Similarity_Usecase const&, input_usecase_t const&> const& param,
                        test_functor_t const& test_functor)
  {
    constexpr bool renumber                  = true;
    auto [similarity_usecase, input_usecase] = param;

    // 1. initialize handle

    raft::handle_t handle{};
    HighResClock hr_clock{};

    // 2. create SG graph

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_clock.start();
    }

    auto [graph, d_renumber_map_labels] =
      cugraph::test::construct_graph<vertex_t, edge_t, weight_t, false, false>(
        handle, input_usecase, false, renumber);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      double elapsed_time{0.0};
      hr_clock.stop(&elapsed_time);
      std::cout << "construct_graph took " << elapsed_time * 1e-6 << " s.\n";
    }

    // 3. run similarity

    auto graph_view = graph.view();

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_clock.start();
    }

    // FIXME:  Need to add some tests that specify first/second
    std::optional<raft::device_span<vertex_t const>> first  = std::nullopt;
    std::optional<raft::device_span<vertex_t const>> second = std::nullopt;

#if 0
    auto [result_src, result_dst, result_score] =
      test_functor.run(handle, graph_view, first, second, similarity_usecase.use_weights);
#else
    EXPECT_THROW(
      test_functor.run(handle, graph_view, first, second, similarity_usecase.use_weights),
      std::exception);
#endif
    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      double elapsed_time{0.0};
      hr_clock.stop(&elapsed_time);
      std::cout << test_functor.testname << " took " << elapsed_time * 1e-6 << " s.\n";
    }

    if (similarity_usecase.check_correctness) {
#if 0
      auto [src, dst, wgt] = cugraph::test::graph_to_host_coo(handle, graph_view);

      std::vector<vertex_t> h_result_src(result_src.size());
      std::vector<vertex_t> h_result_dst(result_dst.size());
      std::vector<weight_t> h_result_score(result_score.size());

      raft::update_host(
        h_result_src.data(), result_src.data(), result_src.size(), handle.get_stream());
      raft::update_host(
        h_result_dst.data(), result_dst.data(), result_dst.size(), handle.get_stream());
      raft::update_host(
        h_result_score.data(), result_score.data(), result_score.size(), handle.get_stream());

      similarity_compare(graph_view.number_of_vertices(),
                         std::move(src),
                         std::move(dst),
                         std::move(wgt),
                         std::move(h_result_src),
                         std::move(h_result_dst),
                         std::move(h_result_score),
                         test_functor);
#endif
    }
  }
};

using Tests_Similarity_File = Tests_Similarity<cugraph::test::File_Usecase>;
using Tests_Similarity_Rmat = Tests_Similarity<cugraph::test::Rmat_Usecase>;

TEST_P(Tests_Similarity_File, CheckInt32Int32FloatJaccard)
{
  run_current_test<int32_t, int32_t, float>(
    override_File_Usecase_with_cmd_line_arguments(GetParam()), cugraph::test::test_jaccard_t{});
}

TEST_P(Tests_Similarity_Rmat, CheckInt32Int32FloatJaccard)
{
  run_current_test<int32_t, int32_t, float>(
    override_Rmat_Usecase_with_cmd_line_arguments(GetParam()), cugraph::test::test_jaccard_t{});
}

TEST_P(Tests_Similarity_Rmat, CheckInt32Int64FloatJaccard)
{
  run_current_test<int32_t, int64_t, float>(
    override_Rmat_Usecase_with_cmd_line_arguments(GetParam()), cugraph::test::test_jaccard_t{});
}

TEST_P(Tests_Similarity_Rmat, CheckInt64Int64FloatJaccard)
{
  run_current_test<int64_t, int64_t, float>(
    override_Rmat_Usecase_with_cmd_line_arguments(GetParam()), cugraph::test::test_jaccard_t{});
}

TEST_P(Tests_Similarity_File, CheckInt32Int32FloatSorensen)
{
  run_current_test<int32_t, int32_t, float>(
    override_File_Usecase_with_cmd_line_arguments(GetParam()), cugraph::test::test_sorensen_t{});
}

TEST_P(Tests_Similarity_Rmat, CheckInt32Int32FloatSorensen)
{
  run_current_test<int32_t, int32_t, float>(
    override_Rmat_Usecase_with_cmd_line_arguments(GetParam()), cugraph::test::test_sorensen_t{});
}

TEST_P(Tests_Similarity_Rmat, CheckInt32Int64FloatSorensen)
{
  run_current_test<int32_t, int64_t, float>(
    override_Rmat_Usecase_with_cmd_line_arguments(GetParam()), cugraph::test::test_sorensen_t{});
}

TEST_P(Tests_Similarity_Rmat, CheckInt64Int64FloatSorensen)
{
  run_current_test<int64_t, int64_t, float>(
    override_Rmat_Usecase_with_cmd_line_arguments(GetParam()), cugraph::test::test_sorensen_t{});
}

TEST_P(Tests_Similarity_File, CheckInt32Int32FloatOverlap)
{
  run_current_test<int32_t, int32_t, float>(
    override_File_Usecase_with_cmd_line_arguments(GetParam()), cugraph::test::test_overlap_t{});
}

TEST_P(Tests_Similarity_Rmat, CheckInt32Int32FloatOverlap)
{
  run_current_test<int32_t, int32_t, float>(
    override_Rmat_Usecase_with_cmd_line_arguments(GetParam()), cugraph::test::test_overlap_t{});
}

TEST_P(Tests_Similarity_Rmat, CheckInt32Int64FloatOverlap)
{
  run_current_test<int32_t, int64_t, float>(
    override_Rmat_Usecase_with_cmd_line_arguments(GetParam()), cugraph::test::test_overlap_t{});
}

TEST_P(Tests_Similarity_Rmat, CheckInt64Int64FloatOverlap)
{
  run_current_test<int64_t, int64_t, float>(
    override_Rmat_Usecase_with_cmd_line_arguments(GetParam()), cugraph::test::test_overlap_t{});
}

INSTANTIATE_TEST_SUITE_P(
  file_test,
  Tests_Similarity_File,
  ::testing::Combine(
    // enable correctness checks
    ::testing::Values(Similarity_Usecase{true}),
    ::testing::Values(cugraph::test::File_Usecase("test/datasets/karate.mtx"),
                      cugraph::test::File_Usecase("test/datasets/dolphins.mtx"))));

INSTANTIATE_TEST_SUITE_P(rmat_small_test,
                         Tests_Similarity_Rmat,
                         // enable correctness checks
                         ::testing::Combine(::testing::Values(Similarity_Usecase{true}),
                                            ::testing::Values(cugraph::test::Rmat_Usecase(
                                              10, 16, 0.57, 0.19, 0.19, 0, false, false))));

INSTANTIATE_TEST_SUITE_P(
  file_benchmark_test, /* note that the test filename can be overridden in benchmarking (with
                          --gtest_filter to select only the file_benchmark_test with a specific
                          vertex & edge type combination) by command line arguments and do not
                          include more than one File_Usecase that differ only in filename
                          (to avoid running same benchmarks more than once) */
  Tests_Similarity_File,
  ::testing::Combine(
    // disable correctness checks
    ::testing::Values(Similarity_Usecase{false}),
    ::testing::Values(cugraph::test::File_Usecase("test/datasets/karate.mtx"))));

INSTANTIATE_TEST_SUITE_P(
  rmat_benchmark_test, /* note that scale & edge factor can be overridden in benchmarking (with
                          --gtest_filter to select only the rmat_benchmark_test with a specific
                          vertex & edge type combination) by command line arguments and do not
                          include more than one Rmat_Usecase that differ only in scale or edge
                          factor (to avoid running same benchmarks more than once) */
  Tests_Similarity_Rmat,
  // disable correctness checks for large graphs
  ::testing::Combine(
    ::testing::Values(Similarity_Usecase{false}),
    ::testing::Values(cugraph::test::Rmat_Usecase(10, 16, 0.57, 0.19, 0.19, 0, false, false))));

CUGRAPH_TEST_PROGRAM_MAIN()
