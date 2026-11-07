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
 * @brief Tests that a recorded sparse product is ordered behind the task that wrote the
 *        values of its matrix and behind no task that wrote anything else.
 * @author Mustafa Garip
 */

#include <string>
#include <vector>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.hpp>
#include <SushiBLAS/engine_fixture/sushiblas_test.hpp>
#include <SushiBLAS/tensor_expectations/tensor_readback.hpp>
#include <SushiBLAS/graph_probe/graph_probe.hpp>
#include <SushiBLAS/sparse_fixtures/sparse_samples.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            /** @brief Holds which tensor the first task writes and the edges the product gets. */
            struct OrderRow
            {
                const char* name;      /**< The row suffix. */
                bool writes_values;    /**< Whether the first task writes A.values. */
                std::size_t dependencies; /**< The tasks the product must wait for. */
            };

            /** @brief Returns a writer of the matrix values and a writer of an unrelated tensor. */
            std::vector<OrderRow> order_rows()
            {
                return {{"AWriterOfTheMatrixValues", true, 1},
                        {"AWriterOfAnUnrelatedTensor", false, 0}};
            }

            /// @brief Names a row, which gtest takes as the test suffix.
            std::string order_row_name(const ::testing::TestParamInfo<OrderRow>& info)
            {
                return info.param.name;
            }
        } // namespace

        /**
         * @brief Records one sparse product after one writer, on FLOAT32 operands with INT32
         *        indices, and reads the graph.
         */
        class Medium_SparseRecording : public SushiBLASTest,
                                       public ::testing::WithParamInterface<OrderRow>
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

                /** @brief Returns a FLOAT32 vector holding values. */
                [[nodiscard]] Tensor vector_of(const std::vector<float>& values)
                {
                    return create_initialized_tensor<float>(
                        {static_cast<int64_t>(values.size())}, values);
                }
        };

        TEST_P(Medium_SparseRecording, AProductIsOrderedBehindTheTaskThatWroteItsValues)
        {
            const OrderRow& row = GetParam();
            const Tensor ones = vector_of(std::vector<float>(7, 1.0f));
            const Tensor x = vector_of({1.0f, 2.0f, 3.0f, 4.0f, 5.0f});
            GraphProbe probe;
            SparseMatrix A = build(sample_matrix<float>());
            Tensor y = vector_of({0.0f, 0.0f, 0.0f, 0.0f});
            Tensor unrelated = vector_of(std::vector<float>(7, 0.0f));
            Tensor& written = row.writes_values ? A.values : unrelated;

            ElementwiseOps(probe.recorder()).mul(ones, ones, written);
            SparseOps(probe.recorder()).spmv(A, x, y);

            ASSERT_EQ(2u, probe.recorded_count());
            const auto tasks = probe.recorded_tasks();
            ASSERT_TRUE(tasks.has_value()) << tasks.error().message;
            EXPECT_EQ(row.dependencies, (*tasks)[1].dependency_count)
                << "The product reads A.values and depends on a task only when it wrote them.";
        }

        INSTANTIATE_TEST_SUITE_P(, Medium_SparseRecording, ::testing::ValuesIn(order_rows()),
                                 order_row_name);
    } // namespace Test
} // namespace SushiBLAS
