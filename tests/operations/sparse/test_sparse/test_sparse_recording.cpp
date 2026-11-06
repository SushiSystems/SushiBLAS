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
/* Licensed under the Apache License, Version 2.0 (the "License"); you    */
/* may not use this file except in compliance with the License. You may   */
/* obtain a copy of the License at                                        */
/*                                                                        */
/*     https://www.apache.org/licenses/LICENSE-2.0                        */
/*                                                                        */
/* Unless required by applicable law or agreed to in writing, software    */
/* distributed under the License is distributed on an "AS IS" BASIS,      */
/* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        */
/* implied. See the License for the specific language governing           */
/* permissions and limitations under the License.                         */
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
#include <SushiBLAS/tensor_expectations/tolerance_bounds.hpp>
#include <SushiBLAS/tensor_expectations/tensor_readback.hpp>
#include <SushiBLAS/test_runner/test_environment.hpp>
#include <SushiBLAS/graph_probe/graph_probe.hpp>
#include <SushiBLAS/sparse_fixtures/sparse_samples.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /**
         * @brief Records sparse calls on FLOAT32 operands with INT32 indices and reads the graph.
         */
        class Medium_SparseRecording : public SushiBLASTest
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
        };

        TEST_F(Medium_SparseRecording, AProductIsOrderedBehindTheTaskThatWroteItsValues)
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
    } // namespace Test
} // namespace SushiBLAS
