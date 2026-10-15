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
 * @file test_level2_reference.cpp
 * @brief Tests the BLAS Level 2 reference kernels against hand-computed results.
 * @author Mustafa Garip
 */

#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.h>
#include <SushiBLAS/engine_fixture/sushiblas_test.hpp>
#include <SushiBLAS/tensor_expectations/tensor_comparison.hpp>
namespace SushiBLAS
{
    namespace Test
    {
        /// @brief Absolute tolerance of every comparison in this file.
        constexpr double K_TOLERANCE = 1e-5;

        /// @brief Fills the triangle a call must neither read nor write.
        constexpr float K_POISON = 999.0f;

        /**
         * @brief Runs every Level 2 call of a test on the reference kernels.
         */
        class Unit_Level2ReferenceTest : public SushiBLASTest
        {
            protected:
                /** @brief Builds the engine and forces the reference path. */
                void SetUp() override
                {
                    SushiBLASTest::SetUp();
                    engine->set_mode(ExecutionMode::ForceReference);
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

        TEST_F(Unit_Level2ReferenceTest, GemvScalesAndAccumulates)
        {
            const std::vector<float> K_EXPECTED = {1.0f, 6.0f};

            auto A = create_initialized_tensor<float>({2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
            auto x = create_initialized_tensor<float>({3}, {1.0f, 0.0f, -1.0f});
            auto y = create_initialized_tensor<float>({2}, {10.0f, 20.0f});

            SushiBLAS::BLASOps(*engine).gemv(A, x, y, Core::Transpose::NoTrans, 2.0, 0.5);
            engine->execute().value();

            EXPECT_TENSOR_NEAR(y, K_EXPECTED, K_TOLERANCE);
        }

        TEST_F(Unit_Level2ReferenceTest, GemvTransposed)
        {
            const std::vector<float> K_EXPECTED = {-2.0f, -1.0f, 0.0f};

            auto A = create_initialized_tensor<float>({2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
            auto x = create_initialized_tensor<float>({2}, {2.0f, -1.0f});
            auto y = create_initialized_tensor<float>({3}, {0.0f, 0.0f, 0.0f});

            SushiBLAS::BLASOps(*engine).gemv(A, x, y, Core::Transpose::Trans, 1.0, 0.0);
            engine->execute().value();

            EXPECT_TENSOR_NEAR(y, K_EXPECTED, K_TOLERANCE);
        }

        TEST_F(Unit_Level2ReferenceTest, GerAddsTheScaledOuterProduct)
        {
            const std::vector<float> K_EXPECTED = {7.0f, -1.0f, 2.0f,
                                                   13.0f, -3.0f, 3.0f};

            auto x = create_initialized_tensor<float>({2}, {1.0f, 2.0f});
            auto y = create_initialized_tensor<float>({3}, {3.0f, -1.0f, 0.5f});
            auto A = create_initialized_tensor<float>({2, 3}, std::vector<float>(6, 1.0f));

            SushiBLAS::BLASOps(*engine).ger(x, y, A, 2.0);
            engine->execute().value();

            EXPECT_TENSOR_NEAR(A, K_EXPECTED, K_TOLERANCE);
        }

        TEST_F(Unit_Level2ReferenceTest, SymvReadsOnlyTheUpperTriangle)
        {
            const std::vector<float> K_EXPECTED = {5.0f, 8.0f, 10.0f};

            auto A = create_initialized_tensor<float>({3, 3}, {1.0f, 2.0f, 3.0f,
                                                               K_POISON, 4.0f, 5.0f,
                                                               K_POISON, K_POISON, 6.0f});
            auto x = create_initialized_tensor<float>({3}, {1.0f, -1.0f, 2.0f});
            auto y = create_initialized_tensor<float>({3}, {0.0f, 0.0f, 0.0f});

            SushiBLAS::BLASOps(*engine).symv(A, x, y, Core::Uplo::Upper, 1.0, 0.0);
            engine->execute().value();

            EXPECT_TENSOR_NEAR(y, K_EXPECTED, K_TOLERANCE);
        }

        TEST_F(Unit_Level2ReferenceTest, SymvReadsOnlyTheLowerTriangleAndAccumulates)
        {
            const std::vector<float> K_EXPECTED = {11.0f, 17.0f, 21.0f};

            auto A = create_initialized_tensor<float>({3, 3}, {1.0f, K_POISON, K_POISON,
                                                               2.0f, 4.0f, K_POISON,
                                                               3.0f, 5.0f, 6.0f});
            auto x = create_initialized_tensor<float>({3}, {1.0f, -1.0f, 2.0f});
            auto y = create_initialized_tensor<float>({3}, {1.0f, 1.0f, 1.0f});

            SushiBLAS::BLASOps(*engine).symv(A, x, y, Core::Uplo::Lower, 2.0, 1.0);
            engine->execute().value();

            EXPECT_TENSOR_NEAR(y, K_EXPECTED, K_TOLERANCE);
        }

        TEST_F(Unit_Level2ReferenceTest, SyrWritesOnlyTheUpperTriangle)
        {
            const std::vector<float> K_EXPECTED = {1.0f, 2.0f, 3.0f,
                                                   K_POISON, 4.0f, 6.0f,
                                                   K_POISON, K_POISON, 9.0f};

            auto x = create_initialized_tensor<float>({3}, {1.0f, 2.0f, 3.0f});
            auto A = create_initialized_tensor<float>({3, 3}, {0.0f, 0.0f, 0.0f,
                                                               K_POISON, 0.0f, 0.0f,
                                                               K_POISON, K_POISON, 0.0f});

            SushiBLAS::BLASOps(*engine).syr(x, A, Core::Uplo::Upper, 1.0);
            engine->execute().value();

            EXPECT_TENSOR_NEAR(A, K_EXPECTED, K_TOLERANCE);
        }

        TEST_F(Unit_Level2ReferenceTest, SyrWritesOnlyTheLowerTriangle)
        {
            const std::vector<float> K_EXPECTED = {3.0f, K_POISON, K_POISON,
                                                   5.0f, 9.0f, K_POISON,
                                                   7.0f, 13.0f, 19.0f};

            auto x = create_initialized_tensor<float>({3}, {1.0f, 2.0f, 3.0f});
            auto A = create_initialized_tensor<float>({3, 3}, {1.0f, K_POISON, K_POISON,
                                                               1.0f, 1.0f, K_POISON,
                                                               1.0f, 1.0f, 1.0f});

            SushiBLAS::BLASOps(*engine).syr(x, A, Core::Uplo::Lower, 2.0);
            engine->execute().value();

            EXPECT_TENSOR_NEAR(A, K_EXPECTED, K_TOLERANCE);
        }

        TEST_F(Unit_Level2ReferenceTest, Syr2WritesOnlyTheUpperTriangle)
        {
            const std::vector<float> K_EXPECTED = {0.0f, 1.0f, 1.0f,
                                                   K_POISON, 0.0f, 2.0f,
                                                   K_POISON, K_POISON, 4.0f};

            auto x = create_initialized_tensor<float>({3}, {1.0f, 0.0f, 2.0f});
            auto y = create_initialized_tensor<float>({3}, {0.0f, 1.0f, 1.0f});
            auto A = create_initialized_tensor<float>({3, 3}, {0.0f, 0.0f, 0.0f,
                                                               K_POISON, 0.0f, 0.0f,
                                                               K_POISON, K_POISON, 0.0f});

            SushiBLAS::BLASOps(*engine).syr2(x, y, A, Core::Uplo::Upper, 1.0);
            engine->execute().value();

            EXPECT_TENSOR_NEAR(A, K_EXPECTED, K_TOLERANCE);
        }

        TEST_F(Unit_Level2ReferenceTest, Syr2WritesOnlyTheLowerTriangle)
        {
            const std::vector<float> K_EXPECTED = {0.0f, K_POISON, K_POISON,
                                                   0.5f, 0.0f, K_POISON,
                                                   0.5f, 1.0f, 2.0f};

            auto x = create_initialized_tensor<float>({3}, {1.0f, 0.0f, 2.0f});
            auto y = create_initialized_tensor<float>({3}, {0.0f, 1.0f, 1.0f});
            auto A = create_initialized_tensor<float>({3, 3}, {0.0f, K_POISON, K_POISON,
                                                               0.0f, 0.0f, K_POISON,
                                                               0.0f, 0.0f, 0.0f});

            SushiBLAS::BLASOps(*engine).syr2(x, y, A, Core::Uplo::Lower, 0.5);
            engine->execute().value();

            EXPECT_TENSOR_NEAR(A, K_EXPECTED, K_TOLERANCE);
        }

        TEST_F(Unit_Level2ReferenceTest, TrmvUpperNoTranspose)
        {
            const std::vector<float> K_EXPECTED = {9.0f, 14.0f, 12.0f};

            auto A = create_initialized_tensor<float>({3, 3}, upper_matrix());
            auto x = create_initialized_tensor<float>({3}, {1.0f, 1.0f, 2.0f});

            SushiBLAS::BLASOps(*engine).trmv(A, x, Core::Uplo::Upper,
                                             Core::Transpose::NoTrans,
                                             Core::Diag::NonUnit);
            engine->execute().value();

            EXPECT_TENSOR_NEAR(x, K_EXPECTED, K_TOLERANCE);
        }

        TEST_F(Unit_Level2ReferenceTest, TrmvUpperTransposed)
        {
            const std::vector<float> K_EXPECTED = {1.0f, 6.0f, 20.0f};

            auto A = create_initialized_tensor<float>({3, 3}, upper_matrix());
            auto x = create_initialized_tensor<float>({3}, {1.0f, 1.0f, 2.0f});

            SushiBLAS::BLASOps(*engine).trmv(A, x, Core::Uplo::Upper,
                                             Core::Transpose::Trans,
                                             Core::Diag::NonUnit);
            engine->execute().value();

            EXPECT_TENSOR_NEAR(x, K_EXPECTED, K_TOLERANCE);
        }

        TEST_F(Unit_Level2ReferenceTest, TrmvUpperUnitDiagonal)
        {
            const std::vector<float> K_EXPECTED = {9.0f, 11.0f, 2.0f};

            auto A = create_initialized_tensor<float>({3, 3}, upper_matrix());
            auto x = create_initialized_tensor<float>({3}, {1.0f, 1.0f, 2.0f});

            SushiBLAS::BLASOps(*engine).trmv(A, x, Core::Uplo::Upper,
                                             Core::Transpose::NoTrans,
                                             Core::Diag::Unit);
            engine->execute().value();

            EXPECT_TENSOR_NEAR(x, K_EXPECTED, K_TOLERANCE);
        }

        TEST_F(Unit_Level2ReferenceTest, TrmvLowerNoTranspose)
        {
            const std::vector<float> K_EXPECTED = {1.0f, 6.0f, 20.0f};

            auto A = create_initialized_tensor<float>({3, 3}, lower_matrix());
            auto x = create_initialized_tensor<float>({3}, {1.0f, 1.0f, 2.0f});

            SushiBLAS::BLASOps(*engine).trmv(A, x, Core::Uplo::Lower,
                                             Core::Transpose::NoTrans,
                                             Core::Diag::NonUnit);
            engine->execute().value();

            EXPECT_TENSOR_NEAR(x, K_EXPECTED, K_TOLERANCE);
        }

        TEST_F(Unit_Level2ReferenceTest, TrmvLowerTransposed)
        {
            const std::vector<float> K_EXPECTED = {9.0f, 14.0f, 12.0f};

            auto A = create_initialized_tensor<float>({3, 3}, lower_matrix());
            auto x = create_initialized_tensor<float>({3}, {1.0f, 1.0f, 2.0f});

            SushiBLAS::BLASOps(*engine).trmv(A, x, Core::Uplo::Lower,
                                             Core::Transpose::Trans,
                                             Core::Diag::NonUnit);
            engine->execute().value();

            EXPECT_TENSOR_NEAR(x, K_EXPECTED, K_TOLERANCE);
        }

        TEST_F(Unit_Level2ReferenceTest, TrmvLowerTransposedUnitDiagonal)
        {
            const std::vector<float> K_EXPECTED = {9.0f, 11.0f, 2.0f};

            auto A = create_initialized_tensor<float>({3, 3}, lower_matrix());
            auto x = create_initialized_tensor<float>({3}, {1.0f, 1.0f, 2.0f});

            SushiBLAS::BLASOps(*engine).trmv(A, x, Core::Uplo::Lower,
                                             Core::Transpose::Trans,
                                             Core::Diag::Unit);
            engine->execute().value();

            EXPECT_TENSOR_NEAR(x, K_EXPECTED, K_TOLERANCE);
        }

        TEST_F(Unit_Level2ReferenceTest, TrsvUpperNoTranspose)
        {
            const std::vector<float> K_EXPECTED = {1.0f, 1.0f, 2.0f};

            auto A = create_initialized_tensor<float>({3, 3}, upper_matrix());
            auto b = create_initialized_tensor<float>({3}, {9.0f, 14.0f, 12.0f});

            SushiBLAS::BLASOps(*engine).trsv(A, b, Core::Uplo::Upper,
                                             Core::Transpose::NoTrans,
                                             Core::Diag::NonUnit);
            engine->execute().value();

            EXPECT_TENSOR_NEAR(b, K_EXPECTED, K_TOLERANCE);
        }

        TEST_F(Unit_Level2ReferenceTest, TrsvUpperTransposed)
        {
            const std::vector<float> K_EXPECTED = {1.0f, 1.0f, 2.0f};

            auto A = create_initialized_tensor<float>({3, 3}, upper_matrix());
            auto b = create_initialized_tensor<float>({3}, {1.0f, 6.0f, 20.0f});

            SushiBLAS::BLASOps(*engine).trsv(A, b, Core::Uplo::Upper,
                                             Core::Transpose::Trans,
                                             Core::Diag::NonUnit);
            engine->execute().value();

            EXPECT_TENSOR_NEAR(b, K_EXPECTED, K_TOLERANCE);
        }

        TEST_F(Unit_Level2ReferenceTest, TrsvUpperUnitDiagonal)
        {
            const std::vector<float> K_EXPECTED = {1.0f, 1.0f, 2.0f};

            auto A = create_initialized_tensor<float>({3, 3}, upper_matrix());
            auto b = create_initialized_tensor<float>({3}, {9.0f, 11.0f, 2.0f});

            SushiBLAS::BLASOps(*engine).trsv(A, b, Core::Uplo::Upper,
                                             Core::Transpose::NoTrans,
                                             Core::Diag::Unit);
            engine->execute().value();

            EXPECT_TENSOR_NEAR(b, K_EXPECTED, K_TOLERANCE);
        }

        TEST_F(Unit_Level2ReferenceTest, TrsvLowerNoTranspose)
        {
            const std::vector<float> K_EXPECTED = {1.0f, 1.0f, 2.0f};

            auto A = create_initialized_tensor<float>({3, 3}, lower_matrix());
            auto b = create_initialized_tensor<float>({3}, {1.0f, 6.0f, 20.0f});

            SushiBLAS::BLASOps(*engine).trsv(A, b, Core::Uplo::Lower,
                                             Core::Transpose::NoTrans,
                                             Core::Diag::NonUnit);
            engine->execute().value();

            EXPECT_TENSOR_NEAR(b, K_EXPECTED, K_TOLERANCE);
        }

        TEST_F(Unit_Level2ReferenceTest, TrsvLowerTransposed)
        {
            const std::vector<float> K_EXPECTED = {1.0f, 1.0f, 2.0f};

            auto A = create_initialized_tensor<float>({3, 3}, lower_matrix());
            auto b = create_initialized_tensor<float>({3}, {9.0f, 14.0f, 12.0f});

            SushiBLAS::BLASOps(*engine).trsv(A, b, Core::Uplo::Lower,
                                             Core::Transpose::Trans,
                                             Core::Diag::NonUnit);
            engine->execute().value();

            EXPECT_TENSOR_NEAR(b, K_EXPECTED, K_TOLERANCE);
        }

        TEST_F(Unit_Level2ReferenceTest, TrsvLowerTransposedUnitDiagonal)
        {
            const std::vector<float> K_EXPECTED = {1.0f, 1.0f, 2.0f};

            auto A = create_initialized_tensor<float>({3, 3}, lower_matrix());
            auto b = create_initialized_tensor<float>({3}, {9.0f, 11.0f, 2.0f});

            SushiBLAS::BLASOps(*engine).trsv(A, b, Core::Uplo::Lower,
                                             Core::Transpose::Trans,
                                             Core::Diag::Unit);
            engine->execute().value();

            EXPECT_TENSOR_NEAR(b, K_EXPECTED, K_TOLERANCE);
        }
    } // namespace Test
} // namespace SushiBLAS
