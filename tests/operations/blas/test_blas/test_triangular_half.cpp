/**************************************************************************/
/* test_triangular_half.cpp                                               */
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
 * @file test_triangular_half.cpp
 * @brief Tests that the blocked HALF TPSV and the HALF TRMV carry their sums in FLOAT32.
 * @author Mustafa Garip
 */

#include <string>
#include <vector>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include "triangular_support.hpp"

namespace SushiBLAS
{
    namespace Test
    {
        using namespace TriangularSupport;

        /** @brief Gives the HALF triangular tests their operands, and skips a device without fp16. */
        class TriangularHalfBase : public SushiBLASTest
        {
            protected:
                /** @brief Builds the engine and skips the case on a device without fp16. */
                void SetUp() override
                {
                    SushiBLASTest::SetUp();

                    if (!test_queue().get_device().has(sycl::aspect::fp16))
                        GTEST_SKIP() << "the device does not report the fp16 aspect";
                }

                /** @brief Returns the row-major unit-lower A of order K_HALF_ORDER, zero above. */
                [[nodiscard]] static std::vector<sycl::half> unit_lower_values()
                {
                    const std::size_t order = static_cast<std::size_t>(K_HALF_ORDER);
                    std::vector<sycl::half> values(order * order, sycl::half(0.0f));
                    for (std::size_t i = 0; i < order; ++i)
                    {
                        for (std::size_t k = 0; k < i; ++k)
                        {
                            const int64_t step = static_cast<int64_t>((i * 7u + k * 13u) % 17u) - 8;
                            values[i * order + k] = sycl::half(static_cast<float>(step) / 64.0f);
                        }
                        values[i * order + i] = sycl::half(1.0f);
                    }
                    return values;
                }

                /** @brief Returns K_HALF_ORDER right-hand-side elements, each a multiple of 1/8. */
                [[nodiscard]] static std::vector<sycl::half> right_hand_side_values()
                {
                    std::vector<sycl::half> values(static_cast<std::size_t>(K_HALF_ORDER));
                    for (std::size_t i = 0; i < values.size(); ++i)
                    {
                        const int64_t step = static_cast<int64_t>((i * 5u) % 23u) - 11;
                        values[i] = sycl::half(static_cast<float>(step) / 8.0f);
                    }
                    return values;
                }

                /**
                 * @brief Returns the host solution of the unit-lower system of one vector, summed
                 *        and carried in float, subtracting in the order of the kernel's
                 *        substitution.
                 * @return Each solved element rounded once to HALF.
                 */
                [[nodiscard]] static std::vector<sycl::half> substitute_in_float(
                    const std::vector<sycl::half>& a, const std::vector<sycl::half>& b)
                {
                    std::vector<float> carried(b.size());
                    std::vector<sycl::half> solution(b.size());

                    for (int64_t i = 0; i < K_HALF_ORDER; ++i)
                    {
                        float sum = static_cast<float>(b[static_cast<std::size_t>(i)]);
                        for (int64_t k = 0; k < i; ++k)
                        {
                            const std::size_t at = static_cast<std::size_t>(i * K_HALF_ORDER + k);
                            sum -= static_cast<float>(a[at]) * carried[static_cast<std::size_t>(k)];
                        }
                        carried[static_cast<std::size_t>(i)] = sum;
                        solution[static_cast<std::size_t>(i)] = sycl::half(sum);
                    }
                    return solution;
                }
        };

        /** @brief Runs the HALF TPSV at a block width, one width per row. */
        class Medium_TriangularHalfWidth : public TriangularHalfBase,
                                           public ::testing::WithParamInterface<int64_t>
        {

        };

        /** @brief Runs the HALF TRMV in a kernel, one kernel per row. */
        class Medium_TriangularHalfMode : public TriangularHalfBase,
                                          public ::testing::WithParamInterface<ExecutionMode>
        {

        };

        /** @brief Tests that a blocked HALF TPSV carries its solution in FLOAT32 at a width. */
        TEST_P(Medium_TriangularHalfWidth, ABlockedHalfVectorSolveCarriesTheSolutionInFloat)
        {
            TriangularFlags flags;
            flags.diag = Core::Diag::Unit;

            const std::vector<sycl::half> a = unit_lower_values();
            const std::vector<sycl::half> b = right_hand_side_values();

            Tensor AP = device_buffer<sycl::half>(*engine, packed_storage(flags, K_HALF_ORDER, a));
            Tensor x = device_buffer<sycl::half>(*engine, b);
            Tensor work = device_buffer<float>(*engine, std::vector<float>(b.size(), 0.0f));

            Kernels::SYCL::tpsv<sycl::half>(
                test_queue(), Kernels::SYCL::TriangularSolvePlan{GetParam()}, flags.layout,
                flags.uplo, flags.trans, flags.diag, K_HALF_ORDER, pointer_of<const sycl::half>(AP),
                pointer_of<sycl::half>(x), 1, pointer_of<float>(work), {})
                .wait_and_throw();
            expect_same_bits(read_buffer<sycl::half>(x), substitute_in_float(a, b));
        }

        /** @brief Tests that a HALF TRMV sums a row in FLOAT32, where a HALF sum loses the 1. */
        TEST_P(Medium_TriangularHalfMode, HalfTrmvSumsInFloat)
        {
            const sycl::half nan(std::numeric_limits<float>::quiet_NaN());
            const sycl::half one(1.0f);
            const std::vector<sycl::half> a = {one, nan, nan,
                                               one, one, nan,
                                               sycl::half(2048.0f), one, sycl::half(-2048.0f)};
            set_mode(GetParam());

            Tensor a_buffer = device_buffer<sycl::half>(*engine, a);
            Tensor x = device_buffer<sycl::half>(*engine, std::vector<sycl::half>(3, one));
            Tensor A = matrix_view(a_buffer, {3, 3}, Core::Layout::RowMajor);

            SushiBLAS::BLASOps(recorder()).trmv(A, x, Core::Uplo::Lower);
            recording->run().value();

            const std::vector<sycl::half> product = read_buffer<sycl::half>(x);
            ASSERT_EQ(std::size_t{3}, product.size());
            EXPECT_EQ(1.0f, static_cast<float>(product[2]));
        }

        INSTANTIATE_TEST_SUITE_P(, Medium_TriangularHalfWidth,
                                 ::testing::ValuesIn(values_of(K_HALF_WIDTHS)),
                                 [](const ::testing::TestParamInfo<int64_t>& info) {
                                     return "w" + std::to_string(static_cast<std::size_t>(info.param));
                                 });
        INSTANTIATE_TEST_SUITE_P(, Medium_TriangularHalfMode,
                                 ::testing::Values(ExecutionMode::ForceReference,
                                                   ExecutionMode::ForceSYCL),
                                 [](const ::testing::TestParamInfo<ExecutionMode>& info) {
                                     return word(info.param);
                                 });
    } // namespace Test
} // namespace SushiBLAS
