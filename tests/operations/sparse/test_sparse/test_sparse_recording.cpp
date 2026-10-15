/**************************************************************************/
/* test_sparse_recording.cpp                                              */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                SushiBLAS                               */
/*                https://github.com/SushiSystems/SushiBLAS               */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present Mustafa Garip & Sushi Systems               */
/*                                                                        */
/* Licensed under the PolyForm Noncommercial License 1.0.0 (the           */
/* "License"); you may not use this file except in compliance with the    */
/* License. You may obtain a copy of the License at                       */
/*                                                                        */
/*     https://polyformproject.org/licenses/noncommercial/1.0.0           */
/*                                                                        */
/* Noncommercial use is free. Commercial use requires a separate licence  */
/* from Sushi Systems; see COMMERCIAL.md. The software is provided        */
/* "as is", without warranty of any kind.                                 */
/**************************************************************************/

/**
 * @file test_sparse_recording.cpp
 * @brief Tests that a recorded sparse task computes its result, keeps its operands and is
 *        ordered behind the task that wrote them.
 * @author Mustafa Garip
 */

#include <atomic>
#include <vector>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.hpp>
#include <SushiBLAS/engine_fixture/sushiblas_test.hpp>
#include <SushiBLAS/tensor_expectations/tensor_comparison.hpp>
#include <SushiBLAS/tensor_expectations/tensor_readback.hpp>
#include <SushiBLAS/test_runner/test_environment.hpp>
#include <SushiBLAS/graph_probe/graph_probe.hpp>
#include <SushiBLAS/sparse_fixtures/sparse_samples.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            /// @brief Absolute error allowed on a result; every expectation is a small integer.
            constexpr double K_TOLERANCE = 1e-5;
        } // namespace

        /**
         * @brief Records sparse calls on FLOAT32 operands with INT32 indices and reads the graph.
         */
        class Regression_SparseRecording : public SushiBLASTest
        {
            protected:
                /** @brief Returns the CSR matrix of FLOAT32 host triplets, zero-based. */
                [[nodiscard]] SparseMatrix build(const HostTriplets<float>& host)
                {
                    const Core::IndexBase base = Core::IndexBase::Zero;
                    const SparseBuilder builder(engine->tensor_factory());
                    return builder.matrix_from_triplets(
                        host.rows, host.cols, stored_indices<int32_t>(host.row_indices, base),
                        stored_indices<int32_t>(host.column_indices, base), host.values);
                }

                /** @brief Returns the 4 x 5 sample matrix. */
                [[nodiscard]] SparseMatrix sample()
                {
                    return build(sample_matrix<float>());
                }

                /**
                 * @brief Returns a 4 x 4 lower triangle L with rows (2), (1, 3), (., 4, 5),
                 *        (6, ., 7, 8).
                 * @return L * (1, 2, 3, 4) is (2, 7, 23, 59); mirrored, the product is
                 *         (28, 19, 51, 59).
                 */
                [[nodiscard]] SparseMatrix lower_triangle()
                {
                    HostTriplets<float> host;
                    host.rows = 4;
                    host.cols = 4;
                    host.row_indices = {0, 1, 1, 2, 2, 3, 3, 3};
                    host.column_indices = {0, 0, 1, 1, 2, 0, 2, 3};
                    host.values = {2.0f, 1.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
                    return build(host);
                }

                /** @brief Returns a FLOAT32 vector holding values. */
                [[nodiscard]] Tensor vector_of(const std::vector<float>& values)
                {
                    return create_initialized_tensor<float>(
                        {static_cast<int64_t>(values.size())}, values);
                }

                /** @brief Returns how many USM allocations of the test context are alive. */
                [[nodiscard]] static std::size_t live_allocations()
                {
                    return TestEnvironment::live_allocation_count();
                }

                /**
                 * @brief Expects the one task record adds to compute what verify compares.
                 * @param record Callable `(TaskRecorder&) -> TaskHandle` that records one task.
                 * @param verify Callable `() -> void` comparing the task's output with the host
                 *        formula; it runs after the probe has run the task.
                 */
                template<typename Record, typename Verify>
                void expect_task_computes(const char* routine, Record&& record, Verify&& verify)
                {
                    SCOPED_TRACE(routine);

                    GraphProbe probe(ExecutionMode::ForceSYCL);
                    ASSERT_TRUE(record(probe.recorder()).is_valid());

                    ASSERT_EQ(1u, probe.recorded_count());
                    const auto tasks = probe.recorded_tasks();
                    ASSERT_TRUE(tasks.has_value()) << tasks.error().message;
                    ASSERT_EQ(1u, tasks->size());
                    EXPECT_EQ(0u, (*tasks)[0].dependency_count);

                    const SushiRuntime::Result<void> ran = probe.execute();
                    ASSERT_TRUE(ran.has_value()) << ran.error().message;
                    verify();
                }
        };

        TEST_F(Regression_SparseRecording, ARecordedTaskComputesItsResult)
        {
            const std::vector<float> four = {1.0f, 2.0f, 3.0f, 4.0f};
            const std::vector<float> five = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
            const std::vector<float> transposed = {25.0f, 9.0f, 12.0f, 2.0f, 43.0f};

            {
                const SparseBuilder builder(engine->tensor_factory());
                const SparseVector x = builder.vector_from_entries(
                    8, std::vector<int32_t>{6, 1, 4}, std::vector<float>{10.0f, 20.0f, 30.0f});
                const Tensor y = vector_of({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f});
                Tensor result = vector_of({0.0f});
                expect_task_computes("doti",
                    [&](TaskRecorder& r) { return SparseOps(r).doti(x, y, result); },
                    [&] { verify_tensor(result, std::vector<float>{260.0f}, K_TOLERANCE); });
            }

            {
                const SparseMatrix A = sample();
                const Tensor x = vector_of(five);
                Tensor y = vector_of({0.0f, 0.0f, 0.0f, 0.0f});
                expect_task_computes("spmv under NO_TRANS",
                    [&](TaskRecorder& r) { return SparseOps(r).spmv(A, x, y); },
                    [&] { verify_tensor(y, std::vector<float>{9, 0, 43, 41}, K_TOLERANCE); });
            }

            {
                const SparseMatrix A = sample();
                const Tensor x = vector_of(four);
                Tensor y = vector_of({0.0f, 0.0f, 0.0f, 0.0f, 0.0f});
                expect_task_computes("spmv under TRANS",
                    [&](TaskRecorder& r)
                    {
                        return SparseOps(r).spmv(A, x, y, Core::Transpose::Trans);
                    },
                    [&] { verify_tensor(y, transposed, K_TOLERANCE); });
            }

            {
                const SparseMatrix A = lower_triangle();
                const Tensor x = vector_of(four);
                Tensor y = vector_of({0.0f, 0.0f, 0.0f, 0.0f});
                expect_task_computes("spsymv",
                    [&](TaskRecorder& r)
                    {
                        return SparseOps(r).spsymv(A, x, y, Core::Uplo::Lower);
                    },
                    [&] { verify_tensor(y, std::vector<float>{28, 19, 51, 59}, K_TOLERANCE); });
            }

            {
                // Column 0 of B is (1, 2, 3, 4) and column 1 is zero, row by row.
                const SparseMatrix A = sample();
                const Tensor B =
                    create_initialized_tensor<float>({4, 2}, {1, 0, 2, 0, 3, 0, 4, 0});
                Tensor C = create_initialized_tensor<float>({5, 2}, std::vector<float>(10, 0.0f));
                expect_task_computes("spmm under TRANS",
                    [&](TaskRecorder& r)
                    {
                        return SparseOps(r).spmm(A, B, C, Core::Transpose::Trans);
                    },
                    [&]
                    {
                        verify_tensor(C, std::vector<float>{25, 0, 9, 0, 12, 0, 2, 0, 43, 0},
                                      K_TOLERANCE);
                    });
            }

            {
                const SparseMatrix A = lower_triangle();
                Tensor b = vector_of({2.0f, 7.0f, 23.0f, 59.0f});
                expect_task_computes("sptrsv",
                    [&](TaskRecorder& r) { return SparseOps(r).sptrsv(A, b, Core::Uplo::Lower); },
                    [&] { verify_tensor(b, four, K_TOLERANCE); });
            }
        }

        TEST_F(Regression_SparseRecording, AMatrixDestroyedBeforeExecuteIsKept)
        {
            const Tensor x = vector_of({1.0f, 2.0f, 3.0f, 4.0f, 5.0f});
            Tensor y = vector_of({0.0f, 0.0f, 0.0f, 0.0f});
            std::size_t live_with_matrix = 0;

            {
                const SparseMatrix A = sample();
                live_with_matrix = live_allocations();
                SparseOps(*engine).spmv(A, x, y);
            }

            EXPECT_GE(live_allocations(), live_with_matrix)
                << "The recorded task is the last holder of the three arrays and must keep them.";

            engine->execute().value();

            verify_tensor(y, std::vector<float>{9, 0, 43, 41}, K_TOLERANCE);
        }

        TEST_F(Regression_SparseRecording, AProductIsOrderedBehindTheTaskThatWroteItsValues)
        {
            const Tensor ones = vector_of(std::vector<float>(7, 1.0f));
            const Tensor x = vector_of({1.0f, 2.0f, 3.0f, 4.0f, 5.0f});

            {
                GraphProbe probe;
                SparseMatrix A = sample();
                Tensor y = vector_of({0.0f, 0.0f, 0.0f, 0.0f});

                ElementwiseOps(probe.recorder()).mul(ones, ones, A.values);
                SparseOps(probe.recorder()).spmv(A, x, y);

                ASSERT_EQ(2u, probe.recorded_count());
                const auto tasks = probe.recorded_tasks();
                ASSERT_TRUE(tasks.has_value()) << tasks.error().message;
                EXPECT_EQ(1u, (*tasks)[1].dependency_count)
                    << "The product reads A.values and must depend on the task that wrote them.";
            }

            {
                GraphProbe probe;
                const SparseMatrix A = sample();
                Tensor y = vector_of({0.0f, 0.0f, 0.0f, 0.0f});
                Tensor unrelated = vector_of(std::vector<float>(7, 0.0f));

                ElementwiseOps(probe.recorder()).mul(ones, ones, unrelated);
                SparseOps(probe.recorder()).spmv(A, x, y);

                ASSERT_EQ(2u, probe.recorded_count());
                const auto tasks = probe.recorded_tasks();
                ASSERT_TRUE(tasks.has_value()) << tasks.error().message;
                EXPECT_EQ(0u, (*tasks)[1].dependency_count)
                    << "The product reads nothing the first task wrote.";
            }
        }

        TEST_F(Regression_SparseRecording, TheSameCallsReplayIntoASecondGraph)
        {
            const SparseMatrix A = sample();
            const Tensor x = vector_of({1.0f, 2.0f, 3.0f, 4.0f});
            const std::vector<float> expected = {25.0f, 9.0f, 12.0f, 2.0f, 43.0f};

            GraphProbe first;
            GraphProbe second;
            Tensor y_first = vector_of({0.0f, 0.0f, 0.0f, 0.0f, 0.0f});
            Tensor y_second = vector_of({0.0f, 0.0f, 0.0f, 0.0f, 0.0f});

            SparseOps(first.recorder()).spmv(A, x, y_first, Core::Transpose::Trans);
            SparseOps(second.recorder()).spmv(A, x, y_second, Core::Transpose::Trans);
            first.execute().value();
            second.execute().value();

            verify_tensor(y_first, expected, K_TOLERANCE);
            EXPECT_EQ(read_tensor<float>(y_first), read_tensor<float>(y_second));
        }
    } // namespace Test
} // namespace SushiBLAS
