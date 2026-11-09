/**************************************************************************/
/* test_blas_validation.cpp                                               */
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
 * @file test_blas_validation.cpp
 * @brief Tests the record-time operand checks of the BLAS operations and of tensor creation.
 * @author Mustafa Garip
 */

#include <vector>
#include <string>
#include <iterator>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.hpp>
#include <SushiBLAS/engine_fixture/sushiblas_test.hpp>
#include <SushiBLAS/tensor_expectations/tensor_comparison.hpp>
#include <SushiBLAS/tensor_expectations/tolerance_bounds.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        using namespace SushiRuntime::Literals;

        namespace
        {
            /// @brief Absolute tolerance: 16 FLOAT32 roundings of a value of order one;
            /// every value is a small integer.
            constexpr double K_TOLERANCE = absolute_tolerance<float>(16);

            /// @brief Two row-major 2x2 matrices: [[1,2],[3,4]] and [[5,6],[7,8]].
            const std::vector<float> K_BATCHED_A = {1, 2, 3, 4, 5, 6, 7, 8};

            /// @brief One row-major 2x2 matrix [[1,1],[0,1]], shared by both batches.
            const std::vector<float> K_SHARED_B = {1, 1, 0, 1};

            /// @brief A0 * B = [[1,3],[3,7]] and A1 * B = [[5,11],[7,15]].
            const std::vector<float> K_BROADCAST_C = {1, 3, 3, 7, 5, 11, 7, 15};

            /// @brief A scalar with a non-zero imaginary part, which a real tensor refuses.
            const BLASScalar K_COMPLEX_ALPHA = std::complex<double>(1.0, 2.0);

            /** @brief Holds a level 2 routine and records it on a matrix and two vectors. */
            struct Level2Routine
            {
                const char* name;   /**< The routine as a call spells it. */
                bool needs_square;  /**< Whether the matrix it takes must be square. */
                void (*record)(BLASOps&, Tensor&, Tensor&, Tensor&); /**< Records it on A, x, y. */
            };

            /// @brief The seven level 2 routines that take a dense matrix.
            const Level2Routine K_LEVEL2[] = {
                {"gemv", false,
                 [](BLASOps& ops, Tensor& A, Tensor& x, Tensor& y) { (void)ops.gemv(A, x, y); }},
                {"ger", false,
                 [](BLASOps& ops, Tensor& A, Tensor& x, Tensor& y) { (void)ops.ger(x, y, A); }},
                {"symv", true,
                 [](BLASOps& ops, Tensor& A, Tensor& x, Tensor& y)
                 {
                     (void)ops.symv(A, x, y, Core::Uplo::Lower);
                 }},
                {"syr", true,
                 [](BLASOps& ops, Tensor& A, Tensor& x, Tensor&)
                 {
                     (void)ops.syr(x, A, Core::Uplo::Lower);
                 }},
                {"syr2", true,
                 [](BLASOps& ops, Tensor& A, Tensor& x, Tensor& y)
                 {
                     (void)ops.syr2(x, y, A, Core::Uplo::Lower);
                 }},
                {"trmv", true,
                 [](BLASOps& ops, Tensor& A, Tensor& x, Tensor&)
                 {
                     (void)ops.trmv(A, x, Core::Uplo::Lower);
                 }},
                {"trsv", true,
                 [](BLASOps& ops, Tensor& A, Tensor& x, Tensor&)
                 {
                     (void)ops.trsv(A, x, Core::Uplo::Lower);
                 }}};

            /** @brief Holds a level 1 routine and records it on two vectors and a result. */
            struct Level1Routine
            {
                const char* name; /**< The routine as a call spells it. */
                void (*record)(BLASOps&, Tensor&, Tensor&, Tensor&); /**< Records it on x, y, result. */
            };

            /// @brief The five level 1 routines that take two vectors.
            const Level1Routine K_LEVEL1[] = {
                {"axpy",
                 [](BLASOps& ops, Tensor& x, Tensor& y, Tensor&) { (void)ops.axpy(2.0, x, y); }},
                {"copy", [](BLASOps& ops, Tensor& x, Tensor& y, Tensor&) { (void)ops.copy(x, y); }},
                {"swap", [](BLASOps& ops, Tensor& x, Tensor& y, Tensor&) { (void)ops.swap(x, y); }},
                {"rot",
                 [](BLASOps& ops, Tensor& x, Tensor& y, Tensor&)
                 {
                     (void)ops.rot(x, y, 0.6, 0.8);
                 }},
                {"dot",
                 [](BLASOps& ops, Tensor& x, Tensor& y, Tensor& result)
                 {
                     (void)ops.dot(x, y, result);
                 }}};

            /** @brief Holds a routine with a scalar result and records it on a vector pair. */
            struct ScalarResultRoutine
            {
                const char* name; /**< The routine as a call spells it. */
                void (*record)(BLASOps&, Tensor&, Tensor&, Tensor&, Tensor&); /**< On x, y, wide, wide_index. */
            };

            /// @brief The four routines that write one scalar.
            const ScalarResultRoutine K_SCALAR_RESULTS[] = {
                {"dot",
                 [](BLASOps& ops, Tensor& x, Tensor& y, Tensor& wide, Tensor&)
                 {
                     (void)ops.dot(x, y, wide);
                 }},
                {"asum",
                 [](BLASOps& ops, Tensor& x, Tensor&, Tensor& wide, Tensor&)
                 {
                     (void)ops.asum(x, wide);
                 }},
                {"nrm2",
                 [](BLASOps& ops, Tensor& x, Tensor&, Tensor& wide, Tensor&)
                 {
                     (void)ops.nrm2(x, wide);
                 }},
                {"iamax",
                 [](BLASOps& ops, Tensor& x, Tensor&, Tensor&, Tensor& wide_index)
                 {
                     (void)ops.iamax(x, wide_index);
                 }}};

            /** @brief Holds a routine that takes a scalar and records it with a complex one. */
            struct ScalarRoutine
            {
                const char* name; /**< The routine as a call spells it. */
                void (*record)(BLASOps&, Tensor&, Tensor&, Tensor&, Tensor&, Tensor&); /**< On A, B, C, x, y. */
            };

            /// @brief The six routines refused for a complex scalar on a real tensor.
            const ScalarRoutine K_SCALAR_ROUTINES[] = {
                {"axpy",
                 [](BLASOps& ops, Tensor&, Tensor&, Tensor&, Tensor& x, Tensor& y)
                 {
                     (void)ops.axpy(K_COMPLEX_ALPHA, x, y);
                 }},
                {"scal",
                 [](BLASOps& ops, Tensor&, Tensor&, Tensor&, Tensor& x, Tensor&)
                 {
                     (void)ops.scal(K_COMPLEX_ALPHA, x);
                 }},
                {"gemv",
                 [](BLASOps& ops, Tensor& A, Tensor&, Tensor&, Tensor& x, Tensor& y)
                 {
                     (void)ops.gemv(A, x, y, Core::Transpose::NoTrans, K_COMPLEX_ALPHA);
                 }},
                {"gemm",
                 [](BLASOps& ops, Tensor& A, Tensor& B, Tensor& C, Tensor&, Tensor&)
                 {
                     (void)ops.gemm(A, B, C, Core::Transpose::NoTrans, Core::Transpose::NoTrans,
                                    K_COMPLEX_ALPHA);
                 }},
                {"syrk",
                 [](BLASOps& ops, Tensor& A, Tensor&, Tensor& C, Tensor&, Tensor&)
                 {
                     (void)ops.syrk(A, C, Core::Uplo::Lower, Core::Transpose::NoTrans,
                                    K_COMPLEX_ALPHA);
                 }},
                {"trsm",
                 [](BLASOps& ops, Tensor& A, Tensor& B, Tensor&, Tensor&, Tensor&)
                 {
                     (void)ops.trsm(A, B, Core::Side::Left, Core::Uplo::Lower,
                                    Core::Transpose::NoTrans, Core::Diag::NonUnit,
                                    K_COMPLEX_ALPHA);
                 }}};

            /** @brief Returns the indices of the level 2 routines that need a square matrix. */
            [[nodiscard]] std::vector<std::size_t> square_level2_routines()
            {
                std::vector<std::size_t> indices;
                for (std::size_t index = 0; index < std::size(K_LEVEL2); ++index)
                    if (K_LEVEL2[index].needs_square)
                        indices.push_back(index);
                return indices;
            }

            /** @brief Returns the name of a level 2 row. */
            [[nodiscard]] std::string level2_name(const ::testing::TestParamInfo<std::size_t>& info)
            {
                return K_LEVEL2[info.param].name;
            }

            /** @brief Returns the name of a level 1 row. */
            [[nodiscard]] std::string level1_name(const ::testing::TestParamInfo<std::size_t>& info)
            {
                return K_LEVEL1[info.param].name;
            }

            /** @brief Returns the name of a scalar result row. */
            [[nodiscard]] std::string scalar_result_name(
                const ::testing::TestParamInfo<std::size_t>& info)
            {
                return K_SCALAR_RESULTS[info.param].name;
            }

            /** @brief Returns the name of a complex scalar row. */
            [[nodiscard]] std::string scalar_routine_name(
                const ::testing::TestParamInfo<std::size_t>& info)
            {
                return K_SCALAR_ROUTINES[info.param].name;
            }

            /// @brief The names of the two shapes of B: one matrix of rank 2, and a batch of one.
            const char* const K_BROADCAST_NAMES[] = {"RankTwo", "BatchOfOne"};

            /** @brief Returns the name of a broadcast row. */
            [[nodiscard]] std::string broadcast_name(
                const ::testing::TestParamInfo<std::size_t>& info)
            {
                return K_BROADCAST_NAMES[info.param];
            }
        } // namespace

        /// @brief Fixture for the BLAS record-time validation tests.
        class Medium_BlasValidationTest : public SushiBLASTest
        {

        };

        /// @brief Runs a refusal once per level 2 routine.
        class Medium_BlasLevel2Validation : public Medium_BlasValidationTest,
                                            public ::testing::WithParamInterface<std::size_t>
        {

        };

        /// @brief Runs a refusal once per level 2 routine that needs a square matrix.
        class Medium_BlasLevel2SquareValidation : public Medium_BlasValidationTest,
                                                  public ::testing::WithParamInterface<std::size_t>
        {

        };

        /// @brief Runs a refusal once per level 1 routine that takes two vectors.
        class Medium_BlasLevel1Validation : public Medium_BlasValidationTest,
                                            public ::testing::WithParamInterface<std::size_t>
        {

        };

        /// @brief Runs a refusal once per routine that writes one scalar.
        class Medium_BlasScalarResultValidation : public Medium_BlasValidationTest,
                                                  public ::testing::WithParamInterface<std::size_t>
        {

        };

        /// @brief Runs a refusal once per routine that takes a scalar factor.
        class Medium_BlasComplexScalarValidation : public Medium_BlasValidationTest,
                                                   public ::testing::WithParamInterface<std::size_t>
        {

        };

        /// @brief Runs the broadcast of B over a batched A once per shape of B.
        class Medium_BlasBroadcast : public Medium_BlasValidationTest,
                                     public ::testing::WithParamInterface<std::size_t>
        {

        };

        TEST_P(Medium_BlasScalarResultValidation, RejectsAResultOfMoreThanOneElement)
        {
            auto x = engine->create_tensor({4});
            auto y = engine->create_tensor({4});
            auto wide = engine->create_tensor({2});
            auto wide_index = engine->create_tensor({2}, Core::DataType::Int64);
            SushiBLAS::BLASOps ops(recorder());

            EXPECT_THROW(K_SCALAR_RESULTS[GetParam()].record(ops, x, y, wide, wide_index),
                         std::invalid_argument);
        }

        TEST_P(Medium_BlasBroadcast, GemmBroadcastsAnOperandOverBatchedA)
        {
            auto A = create_initialized_tensor<float>({2, 2, 2}, K_BATCHED_A);
            auto B = (GetParam() == 0) ? create_initialized_tensor<float>({2, 2}, K_SHARED_B)
                                       : create_initialized_tensor<float>({1, 2, 2}, K_SHARED_B);
            auto C = engine->create_tensor({2, 2, 2});

            SushiBLAS::BLASOps(recorder()).gemm(A, B, C);
            recording->run().value();

            expect_tensor(C, K_BROADCAST_C, K_TOLERANCE);
        }

        TEST_P(Medium_BlasLevel2Validation, RejectsAMatrixOfRankThree)
        {
            auto A = engine->create_tensor({1, 3, 3});
            auto x = engine->create_tensor({3});
            auto y = engine->create_tensor({3});
            SushiBLAS::BLASOps ops(recorder());

            EXPECT_THROW(K_LEVEL2[GetParam()].record(ops, A, x, y), std::invalid_argument);
        }

        TEST_P(Medium_BlasLevel2Validation, RejectsAMatrixOfRankOne)
        {
            auto A = engine->create_tensor({9});
            auto x = engine->create_tensor({3});
            auto y = engine->create_tensor({3});
            SushiBLAS::BLASOps ops(recorder());

            EXPECT_THROW(K_LEVEL2[GetParam()].record(ops, A, x, y), std::invalid_argument);
        }

        TEST_P(Medium_BlasLevel2Validation, RejectsAVectorOfTheWrongLength)
        {
            auto A = engine->create_tensor({3, 3});
            auto x = engine->create_tensor({2});
            auto y = engine->create_tensor({3});
            SushiBLAS::BLASOps ops(recorder());

            EXPECT_THROW(K_LEVEL2[GetParam()].record(ops, A, x, y), std::invalid_argument);
        }

        TEST_P(Medium_BlasLevel2SquareValidation, RejectsANonSquareMatrix)
        {
            auto A = engine->create_tensor({2, 3});
            auto x = engine->create_tensor({3});
            auto y = engine->create_tensor({3});
            SushiBLAS::BLASOps ops(recorder());

            EXPECT_THROW(K_LEVEL2[GetParam()].record(ops, A, x, y), std::invalid_argument);
        }

        TEST_P(Medium_BlasLevel1Validation, RejectsOperandsOfDifferentDtypes)
        {
            auto x = engine->create_tensor({4});
            auto y = engine->create_tensor({4}, Core::DataType::Float64);
            auto result = engine->create_tensor({1});
            SushiBLAS::BLASOps ops(recorder());

            EXPECT_THROW(K_LEVEL1[GetParam()].record(ops, x, y, result), std::invalid_argument);
        }

        TEST_P(Medium_BlasLevel1Validation, RejectsOperandsOfDifferentLengths)
        {
            auto x = engine->create_tensor({4});
            auto y = engine->create_tensor({5});
            auto result = engine->create_tensor({1});
            SushiBLAS::BLASOps ops(recorder());

            EXPECT_THROW(K_LEVEL1[GetParam()].record(ops, x, y, result), std::invalid_argument);
        }

        TEST_F(Medium_BlasValidationTest, SymvRejectsAUploThatIsNoEnumerator)
        {
            auto A = engine->create_tensor({3, 3});
            auto x = engine->create_tensor({3});
            auto y = engine->create_tensor({3});

            EXPECT_THROW(SushiBLAS::BLASOps(recorder()).symv(A, x, y, static_cast<Core::Uplo>(0)),
                         std::invalid_argument);
        }

        TEST_F(Medium_BlasValidationTest, GemvRejectsATransposeThatIsNoEnumerator)
        {
            auto A = engine->create_tensor({3, 3});
            auto x = engine->create_tensor({3});
            auto y = engine->create_tensor({3});

            EXPECT_THROW(SushiBLAS::BLASOps(recorder())
                             .gemv(A, x, y, static_cast<Core::Transpose>(0)),
                         std::invalid_argument);
        }

        TEST_F(Medium_BlasValidationTest, TrmvRejectsADiagThatIsNoEnumerator)
        {
            auto A = engine->create_tensor({3, 3});
            auto x = engine->create_tensor({3});

            EXPECT_THROW(SushiBLAS::BLASOps(recorder()).trmv(A, x, Core::Uplo::Lower,
                                                          Core::Transpose::NoTrans,
                                                          static_cast<Core::Diag>(0)),
                         std::invalid_argument);
        }

        TEST_F(Medium_BlasValidationTest, TrsmRejectsASideThatIsNoEnumerator)
        {
            auto A = engine->create_tensor({3, 3});
            auto B = engine->create_tensor({3, 3});

            EXPECT_THROW(SushiBLAS::BLASOps(recorder()).trsm(A, B, static_cast<Core::Side>(0),
                                                          Core::Uplo::Lower),
                         std::invalid_argument);
        }

        TEST_P(Medium_BlasComplexScalarValidation, RealTensorsRejectAScalarWithAnImaginaryPart)
        {
            auto A = engine->create_tensor({3, 3});
            auto B = engine->create_tensor({3, 3});
            auto C = engine->create_tensor({3, 3});
            auto x = engine->create_tensor({3});
            auto y = engine->create_tensor({3});
            SushiBLAS::BLASOps ops(recorder());

            EXPECT_THROW(K_SCALAR_ROUTINES[GetParam()].record(ops, A, B, C, x, y),
                         std::invalid_argument);
        }

        TEST_F(Medium_BlasValidationTest, DotRejectsComplexTensors)
        {
            auto x = engine->create_tensor({3}, Core::DataType::Complex32);
            auto y = engine->create_tensor({3}, Core::DataType::Complex32);
            auto result = engine->create_tensor({1}, Core::DataType::Complex32);

            EXPECT_THROW(SushiBLAS::BLASOps(recorder()).dot(x, y, result), std::invalid_argument);
        }

        TEST_F(Medium_BlasValidationTest, GerRejectsComplexTensors)
        {
            auto A = engine->create_tensor({3, 3}, Core::DataType::Complex32);
            auto x = engine->create_tensor({3}, Core::DataType::Complex32);
            auto y = engine->create_tensor({3}, Core::DataType::Complex32);

            EXPECT_THROW(SushiBLAS::BLASOps(recorder()).ger(x, y, A), std::invalid_argument);
        }

        TEST_F(Medium_BlasValidationTest, SyrkRejectsConjTransOnComplexTensors)
        {
            auto A = engine->create_tensor({3, 2}, Core::DataType::Complex32);
            auto C = engine->create_tensor({2, 2}, Core::DataType::Complex32);

            EXPECT_THROW(SushiBLAS::BLASOps(recorder()).syrk(A, C, Core::Uplo::Lower,
                                                          Core::Transpose::ConjTrans),
                         std::invalid_argument);
        }

        TEST_F(Medium_BlasValidationTest, HerkRejectsTransOnComplexTensors)
        {
            auto A = engine->create_tensor({3, 2}, Core::DataType::Complex32);
            auto C = engine->create_tensor({2, 2}, Core::DataType::Complex32);

            EXPECT_THROW(SushiBLAS::BLASOps(recorder()).herk(A, C, Core::Uplo::Lower,
                                                          Core::Transpose::Trans),
                         std::invalid_argument);
        }

        INSTANTIATE_TEST_SUITE_P(
            , Medium_BlasLevel2Validation,
            ::testing::Range<std::size_t>(0, std::size(K_LEVEL2)), level2_name);
        INSTANTIATE_TEST_SUITE_P(
            , Medium_BlasLevel2SquareValidation,
            ::testing::ValuesIn(square_level2_routines()), level2_name);
        INSTANTIATE_TEST_SUITE_P(
            , Medium_BlasLevel1Validation,
            ::testing::Range<std::size_t>(0, std::size(K_LEVEL1)), level1_name);
        INSTANTIATE_TEST_SUITE_P(
            , Medium_BlasScalarResultValidation,
            ::testing::Range<std::size_t>(0, std::size(K_SCALAR_RESULTS)), scalar_result_name);
        INSTANTIATE_TEST_SUITE_P(
            , Medium_BlasComplexScalarValidation,
            ::testing::Range<std::size_t>(0, std::size(K_SCALAR_ROUTINES)),
            scalar_routine_name);
        INSTANTIATE_TEST_SUITE_P(
            , Medium_BlasBroadcast,
            ::testing::Range<std::size_t>(0, std::size(K_BROADCAST_NAMES)), broadcast_name);
    } // namespace Test
} // namespace SushiBLAS
