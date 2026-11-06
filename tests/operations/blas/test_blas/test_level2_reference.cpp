/**************************************************************************/
/* test_level2_reference.cpp                                              */
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
 * @file test_level2_reference.cpp
 * @brief Tests the BLAS Level 2 reference kernels against hand-computed results.
 * @author Mustafa Garip
 */

#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.hpp>
#include <SushiBLAS/engine_fixture/sushiblas_test.hpp>
#include <SushiBLAS/tensor_expectations/tensor_comparison.hpp>
#include <SushiBLAS/tensor_expectations/tolerance_bounds.hpp>
namespace SushiBLAS
{
    namespace Test
    {
        /// @brief Absolute tolerance: 16 FLOAT32 roundings of a value of order one,
        /// for every comparison in this file.
        constexpr double K_TOLERANCE = absolute_tolerance<float>(16);

        /// @brief Fills the triangle a call must neither read nor write.
        constexpr float K_POISON = 999.0f;

        /**
         * @brief Runs every Level 2 call of a test on the reference kernels.
         */
        class Medium_Level2ReferenceTest : public SushiBLASTest
        {
            protected:
                /** @brief Builds the engine and forces the reference path. */
                void SetUp() override
                {
                    SushiBLASTest::SetUp();
                    set_mode(ExecutionMode::ForceReference);
                }

                /** @brief Returns U = [[1,2,3],[0,4,5],[0,0,6]] with its lower part poisoned. */
                [[nodiscard]] static std::vector<float> upper_matrix()
                {
                    return {1.0f, 2.0f, 3.0f,
                            K_POISON, 4.0f, 5.0f,
                            K_POISON, K_POISON, 6.0f};
                }

                /** @brief Returns L = U^T with its upper part poisoned. */
                [[nodiscard]] static std::vector<float> lower_matrix()
                {
                    return {1.0f, K_POISON, K_POISON,
                            2.0f, 4.0f, K_POISON,
                            3.0f, 5.0f, 6.0f};
                }
        };

        TEST_F(Medium_Level2ReferenceTest, TrsvUpperNoTranspose)
        {
            const std::vector<float> K_EXPECTED = {1.0f, 1.0f, 2.0f};

            auto A = create_initialized_tensor<float>({3, 3}, upper_matrix());
            auto b = create_initialized_tensor<float>({3}, {9.0f, 14.0f, 12.0f});

            SushiBLAS::BLASOps(recorder()).trsv(A, b, Core::Uplo::Upper,
                                             Core::Transpose::NoTrans,
                                             Core::Diag::NonUnit);
            recording->run().value();

            EXPECT_TENSOR_NEAR(b, K_EXPECTED, K_TOLERANCE);
        }

        TEST_F(Medium_Level2ReferenceTest, TrsvUpperTransposed)
        {
            const std::vector<float> K_EXPECTED = {1.0f, 1.0f, 2.0f};

            auto A = create_initialized_tensor<float>({3, 3}, upper_matrix());
            auto b = create_initialized_tensor<float>({3}, {1.0f, 6.0f, 20.0f});

            SushiBLAS::BLASOps(recorder()).trsv(A, b, Core::Uplo::Upper,
                                             Core::Transpose::Trans,
                                             Core::Diag::NonUnit);
            recording->run().value();

            EXPECT_TENSOR_NEAR(b, K_EXPECTED, K_TOLERANCE);
        }

        TEST_F(Medium_Level2ReferenceTest, TrsvUpperUnitDiagonal)
        {
            const std::vector<float> K_EXPECTED = {1.0f, 1.0f, 2.0f};

            auto A = create_initialized_tensor<float>({3, 3}, upper_matrix());
            auto b = create_initialized_tensor<float>({3}, {9.0f, 11.0f, 2.0f});

            SushiBLAS::BLASOps(recorder()).trsv(A, b, Core::Uplo::Upper,
                                             Core::Transpose::NoTrans,
                                             Core::Diag::Unit);
            recording->run().value();

            EXPECT_TENSOR_NEAR(b, K_EXPECTED, K_TOLERANCE);
        }

        TEST_F(Medium_Level2ReferenceTest, TrsvLowerNoTranspose)
        {
            const std::vector<float> K_EXPECTED = {1.0f, 1.0f, 2.0f};

            auto A = create_initialized_tensor<float>({3, 3}, lower_matrix());
            auto b = create_initialized_tensor<float>({3}, {1.0f, 6.0f, 20.0f});

            SushiBLAS::BLASOps(recorder()).trsv(A, b, Core::Uplo::Lower,
                                             Core::Transpose::NoTrans,
                                             Core::Diag::NonUnit);
            recording->run().value();

            EXPECT_TENSOR_NEAR(b, K_EXPECTED, K_TOLERANCE);
        }

        TEST_F(Medium_Level2ReferenceTest, TrsvLowerTransposed)
        {
            const std::vector<float> K_EXPECTED = {1.0f, 1.0f, 2.0f};

            auto A = create_initialized_tensor<float>({3, 3}, lower_matrix());
            auto b = create_initialized_tensor<float>({3}, {9.0f, 14.0f, 12.0f});

            SushiBLAS::BLASOps(recorder()).trsv(A, b, Core::Uplo::Lower,
                                             Core::Transpose::Trans,
                                             Core::Diag::NonUnit);
            recording->run().value();

            EXPECT_TENSOR_NEAR(b, K_EXPECTED, K_TOLERANCE);
        }

        TEST_F(Medium_Level2ReferenceTest, TrsvLowerTransposedUnitDiagonal)
        {
            const std::vector<float> K_EXPECTED = {1.0f, 1.0f, 2.0f};

            auto A = create_initialized_tensor<float>({3, 3}, lower_matrix());
            auto b = create_initialized_tensor<float>({3}, {9.0f, 11.0f, 2.0f});

            SushiBLAS::BLASOps(recorder()).trsv(A, b, Core::Uplo::Lower,
                                             Core::Transpose::Trans,
                                             Core::Diag::Unit);
            recording->run().value();

            EXPECT_TENSOR_NEAR(b, K_EXPECTED, K_TOLERANCE);
        }
    } // namespace Test
} // namespace SushiBLAS
