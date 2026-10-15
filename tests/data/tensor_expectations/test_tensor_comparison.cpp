/**************************************************************************/
/* test_tensor_comparison.cpp                                             */
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
 * @file test_tensor_comparison.cpp
 * @brief Tests that the tensor comparisons accept a match and report a mismatch, NaN included.
 * @author Mustafa Garip
 */

#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>
#include <gtest/gtest.h>
#include <gtest/gtest-spi.h>
#include <SushiRuntime/sushiruntime.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/tensor/tensor_factory.hpp>
#include <SushiBLAS/tensor_expectations/tensor_comparison.hpp>
#include <SushiBLAS/test_runner/test_environment.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /// @brief Returns a FLOAT32 tensor holding values.
        static Tensor float_tensor(const std::vector<float>& values)
        {
            const TensorFactory factory(TestEnvironment::runtime());
            const int64_t dims[1] = {static_cast<int64_t>(values.size())};
            Tensor tensor = factory.create(SushiRuntime::span<const int64_t>(dims, 1));
            TestEnvironment::queue()
                .memcpy(tensor.data(), values.data(), values.size() * sizeof(float))
                .wait();
            return tensor;
        }

        TEST(Medium_TensorComparison, AcceptsValuesWithinTheTolerance)
        {
            const Tensor tensor = float_tensor({1.0f, 2.0f, 3.0f});

            EXPECT_TENSOR_NEAR(tensor, (std::vector<float>{1.0f, 2.00001f, 3.0f}), 1e-4);
            SB_ASSERT_TENSOR_NEAR(tensor, float_tensor({1.0f, 2.0f, 3.0f}), 0.0, float);
        }

        TEST(Medium_TensorComparison, ReportsAValueOutsideTheTolerance)
        {
            const Tensor tensor = float_tensor({1.0f, 2.0f});

            EXPECT_NONFATAL_FAILURE(verify_tensor(tensor, std::vector<float>{1.0f, 2.5f}, 1e-4),
                                    "Mismatch (value) at index 1");
        }

        TEST(Medium_TensorComparison, ANaNMatchesOnlyANaN)
        {
            const float nan = std::numeric_limits<float>::quiet_NaN();
            const Tensor tensor = float_tensor({nan, 1.0f});

            verify_tensor(tensor, std::vector<float>{nan, 1.0f}, 0.0);
            EXPECT_NONFATAL_FAILURE(verify_tensor(float_tensor({0.0f, 1.0f}),
                                                  std::vector<float>{nan, 1.0f}, 0.0),
                                    "Expected NaN");
        }

        TEST(Medium_TensorComparison, AnInfinityMatchesOnlyItself)
        {
            const float infinity = std::numeric_limits<float>::infinity();

            verify_tensor(float_tensor({infinity}), std::vector<float>{infinity}, 0.0);
            EXPECT_NONFATAL_FAILURE(verify_tensor(float_tensor({-infinity}),
                                                  std::vector<float>{infinity}, 0.0),
                                    "Infinity mismatch");
        }
    } // namespace Test
} // namespace SushiBLAS
