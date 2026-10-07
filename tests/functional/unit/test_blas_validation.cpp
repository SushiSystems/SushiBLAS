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
 * @file test_blas_validation.cpp
 * @brief Tests the record-time operand checks of the BLAS operations and of tensor creation.
 * @author Mustafa Garip
 */

#include <array>
#include <vector>
#include <complex>
#include <cstdint>
#include <stdexcept>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.h>
#include <SushiBLAS/engine/blas/dispatch.hpp>
#include "../../common/test_utils.hpp"

namespace SushiBLAS
{
    namespace Test
    {
        using namespace SushiRuntime::Graph::Literals;

        namespace
        {
            /// @brief Absolute tolerance of the broadcast GEMM check; every value is a small integer.
            constexpr double K_TOLERANCE = 1e-5;

            /// @brief Two row-major 2x2 matrices: [[1,2],[3,4]] and [[5,6],[7,8]].
            const std::vector<float> K_BATCHED_A = {1, 2, 3, 4, 5, 6, 7, 8};

            /// @brief One row-major 2x2 matrix [[1,1],[0,1]], shared by both batches.
            const std::vector<float> K_SHARED_B = {1, 1, 0, 1};

            /// @brief A0 * B = [[1,3],[3,7]] and A1 * B = [[5,11],[7,15]].
            const std::vector<float> K_BROADCAST_C = {1, 3, 3, 7, 5, 11, 7, 15};

            /// @brief A scalar with a non-zero imaginary part, which a real tensor refuses.
            const BlasScalar K_COMPLEX_ALPHA = std::complex<double>(1.0, 2.0);
        } // namespace

        /// @brief Fixture for the BLAS record-time validation tests.
        class Unit_BlasValidationTest : public SushiBLASTest
        {

        };

        TEST_F(Unit_BlasValidationTest, GemmRejectsInnerDimensionMismatch)
        {
            auto A = engine->create_tensor({2, 3});
            auto B = engine->create_tensor({4, 2});
            auto C = engine->create_tensor({2, 2});

            EXPECT_THROW(SushiBLAS::BLASOps(*engine).gemm(A, B, C), std::invalid_argument);
        }

        TEST_F(Unit_BlasValidationTest, GemmRejectsMixedLayouts)
        {
            auto A = engine->create_tensor({2, 2});
            auto row_major_B = engine->create_tensor({2, 2});
            auto C = engine->create_tensor({2, 2});
            const Tensor B(row_major_B.storage, {2, 2}, Core::DataType::Float32, 0,
                           Core::Layout::ColumnMajor);

            EXPECT_THROW(SushiBLAS::BLASOps(*engine).gemm(A, B, C), std::invalid_argument);
        }

        TEST_F(Unit_BlasValidationTest, GemmRejectsBatchCountMismatch)
        {
            auto A = engine->create_tensor({3, 2, 2});
            auto B = engine->create_tensor({2, 2, 2});
            auto C = engine->create_tensor({2, 2, 2});

            EXPECT_THROW(SushiBLAS::BLASOps(*engine).gemm(A, B, C), std::invalid_argument);
        }

        TEST_F(Unit_BlasValidationTest, GemmRejectsTransposedView)
        {
            auto parent = engine->create_tensor({2, 3});
            const Tensor A = parent.transpose(0, 1);
            auto B = engine->create_tensor({2, 2});
            auto C = engine->create_tensor({3, 2});

            EXPECT_THROW(SushiBLAS::BLASOps(*engine).gemm(A, B, C), std::invalid_argument);
        }

        TEST_F(Unit_BlasValidationTest, GemmBroadcastsRankTwoBOverBatchedA)
        {
            auto A = create_initialized_tensor<float>({2, 2, 2}, K_BATCHED_A);
            auto B = create_initialized_tensor<float>({2, 2}, K_SHARED_B);
            auto C = engine->create_tensor({2, 2, 2});

            SushiBLAS::BLASOps(*engine).gemm(A, B, C);
            engine->execute();

            verify_tensor(C, K_BROADCAST_C, K_TOLERANCE);
        }

        TEST_F(Unit_BlasValidationTest, SyrkRejectsNonSquareC)
        {
            auto A = engine->create_tensor({3, 2});
            auto C = engine->create_tensor({3, 2});

            EXPECT_THROW(SushiBLAS::BLASOps(*engine).syrk(A, C, Core::Uplo::Lower),
                         std::invalid_argument);
        }

        TEST_F(Unit_BlasValidationTest, TrsmRejectsNonSquareA)
        {
            auto A = engine->create_tensor({2, 3});
            auto B = engine->create_tensor({2, 2});

            EXPECT_THROW(SushiBLAS::BLASOps(*engine).trsm(A, B, Core::Side::Left,
                                                          Core::Uplo::Lower),
                         std::invalid_argument);
        }

        TEST_F(Unit_BlasValidationTest, DotRejectsResultOfWrongDtype)
        {
            auto x = engine->create_tensor({4});
            auto y = engine->create_tensor({4});
            auto result = engine->create_tensor({1}, Core::DataType::Half);

            EXPECT_THROW(SushiBLAS::BLASOps(*engine).dot(x, y, result), std::invalid_argument);
        }

        TEST_F(Unit_BlasValidationTest, ScalarResultsRejectMoreThanOneElement)
        {
            auto x = engine->create_tensor({4});
            auto y = engine->create_tensor({4});
            auto wide = engine->create_tensor({2});
            auto wide_index = engine->create_tensor({2}, Core::DataType::Int64);

            EXPECT_THROW(SushiBLAS::BLASOps(*engine).dot(x, y, wide), std::invalid_argument);
            EXPECT_THROW(SushiBLAS::BLASOps(*engine).asum(x, wide), std::invalid_argument);
            EXPECT_THROW(SushiBLAS::BLASOps(*engine).nrm2(x, wide), std::invalid_argument);
            EXPECT_THROW(SushiBLAS::BLASOps(*engine).iamax(x, wide_index), std::invalid_argument);
        }

        TEST_F(Unit_BlasValidationTest, GemmRejectsBatchAxesThatDifferWithEqualProduct)
        {
            auto A = engine->create_tensor({2, 3, 2, 2});
            auto B = engine->create_tensor({2, 2});
            auto C = engine->create_tensor({3, 2, 2, 2});

            EXPECT_THROW(SushiBLAS::BLASOps(*engine).gemm(A, B, C), std::invalid_argument);
        }

        TEST_F(Unit_BlasValidationTest, SyrkAndTrsmRejectBatchAxesThatDiffer)
        {
            auto A = engine->create_tensor({2, 3, 2, 2});
            auto C = engine->create_tensor({6, 2, 2});

            EXPECT_THROW(SushiBLAS::BLASOps(*engine).syrk(A, C, Core::Uplo::Lower),
                         std::invalid_argument);
            EXPECT_THROW(SushiBLAS::BLASOps(*engine).trsm(A, C, Core::Side::Left,
                                                          Core::Uplo::Lower),
                         std::invalid_argument);
        }

        TEST_F(Unit_BlasValidationTest, GemmBroadcastsOperandHoldingOneMatrix)
        {
            auto A = create_initialized_tensor<float>({2, 2, 2}, K_BATCHED_A);
            auto B = create_initialized_tensor<float>({1, 2, 2}, K_SHARED_B);
            auto C = engine->create_tensor({2, 2, 2});

            SushiBLAS::BLASOps(*engine).gemm(A, B, C);
            engine->execute();

            verify_tensor(C, K_BROADCAST_C, K_TOLERANCE);
        }

        TEST_F(Unit_BlasValidationTest, Level2RejectsMatrixAboveRankTwo)
        {
            auto A = engine->create_tensor({1, 3, 3});
            auto x = engine->create_tensor({3});
            auto y = engine->create_tensor({3});

            EXPECT_THROW(SushiBLAS::BLASOps(*engine).gemv(A, x, y), std::invalid_argument);
            EXPECT_THROW(SushiBLAS::BLASOps(*engine).ger(x, y, A), std::invalid_argument);
            EXPECT_THROW(SushiBLAS::BLASOps(*engine).symv(A, x, y, Core::Uplo::Lower),
                         std::invalid_argument);
            EXPECT_THROW(SushiBLAS::BLASOps(*engine).syr(x, A, Core::Uplo::Lower),
                         std::invalid_argument);
            EXPECT_THROW(SushiBLAS::BLASOps(*engine).syr2(x, y, A, Core::Uplo::Lower),
                         std::invalid_argument);
            EXPECT_THROW(SushiBLAS::BLASOps(*engine).trmv(A, x, Core::Uplo::Lower),
                         std::invalid_argument);
            EXPECT_THROW(SushiBLAS::BLASOps(*engine).trsv(A, x, Core::Uplo::Lower),
                         std::invalid_argument);
        }

        TEST_F(Unit_BlasValidationTest, TensorCreationRejectsNegativeDimension)
        {
            EXPECT_THROW((void)engine->create_tensor({-1, 4}), std::runtime_error);
            EXPECT_THROW((void)engine->create_tensor({4, -1}, Core::DataType::Float64),
                         std::runtime_error);
        }

        TEST_F(Unit_BlasValidationTest, TensorCreationRejectsOverflowingSize)
        {
            constexpr int64_t K_HUGE = int64_t{1} << 40;

            EXPECT_THROW((void)engine->create_tensor({K_HUGE, K_HUGE}), std::runtime_error);
            EXPECT_THROW((void)engine->create_tensor({int64_t{1} << 62}, Core::DataType::Float64),
                         std::runtime_error);
        }

        TEST_F(Unit_BlasValidationTest, ScalRejectsVectorOfStrideZero)
        {
            auto parent = engine->create_tensor({4});
            const std::vector<int64_t> dims = {4};
            const std::array<int64_t, Core::K_MAX_TENSOR_RANK> strides = {0};
            Tensor x(parent.storage, dims, strides, Core::DataType::Float32, 0,
                     Core::Layout::RowMajor);

            EXPECT_THROW(SushiBLAS::BLASOps(*engine).scal(2.0, x), std::invalid_argument);
        }

        TEST_F(Unit_BlasValidationTest, Level1RejectsOperandsOfDifferentDtypes)
        {
            auto x = engine->create_tensor({4});
            auto y = engine->create_tensor({4}, Core::DataType::Float64);
            auto result = engine->create_tensor({1});

            EXPECT_THROW(SushiBLAS::BLASOps(*engine).axpy(2.0, x, y), std::invalid_argument);
            EXPECT_THROW(SushiBLAS::BLASOps(*engine).copy(x, y), std::invalid_argument);
            EXPECT_THROW(SushiBLAS::BLASOps(*engine).swap(x, y), std::invalid_argument);
            EXPECT_THROW(SushiBLAS::BLASOps(*engine).rot(x, y, 0.6, 0.8), std::invalid_argument);
            EXPECT_THROW(SushiBLAS::BLASOps(*engine).dot(x, y, result), std::invalid_argument);
        }

        TEST_F(Unit_BlasValidationTest, Level1RejectsOperandsOfDifferentLengths)
        {
            auto x = engine->create_tensor({4});
            auto y = engine->create_tensor({5});
            auto result = engine->create_tensor({1});

            EXPECT_THROW(SushiBLAS::BLASOps(*engine).axpy(2.0, x, y), std::invalid_argument);
            EXPECT_THROW(SushiBLAS::BLASOps(*engine).copy(x, y), std::invalid_argument);
            EXPECT_THROW(SushiBLAS::BLASOps(*engine).swap(x, y), std::invalid_argument);
            EXPECT_THROW(SushiBLAS::BLASOps(*engine).rot(x, y, 0.6, 0.8), std::invalid_argument);
            EXPECT_THROW(SushiBLAS::BLASOps(*engine).dot(x, y, result), std::invalid_argument);
        }

        TEST_F(Unit_BlasValidationTest, ScalRejectsNonContiguousVectorOfRankTwo)
        {
            auto parent = engine->create_tensor({2, 3});
            Tensor x = parent.transpose(0, 1);
            ASSERT_FALSE(x.is_contiguous());

            EXPECT_THROW(SushiBLAS::BLASOps(*engine).scal(2.0, x), std::invalid_argument);
        }

        TEST_F(Unit_BlasValidationTest, Level2RejectsMatrixBelowRankTwo)
        {
            auto A = engine->create_tensor({9});
            auto x = engine->create_tensor({3});
            auto y = engine->create_tensor({3});

            EXPECT_THROW(SushiBLAS::BLASOps(*engine).gemv(A, x, y), std::invalid_argument);
            EXPECT_THROW(SushiBLAS::BLASOps(*engine).ger(x, y, A), std::invalid_argument);
            EXPECT_THROW(SushiBLAS::BLASOps(*engine).symv(A, x, y, Core::Uplo::Lower),
                         std::invalid_argument);
            EXPECT_THROW(SushiBLAS::BLASOps(*engine).syr(x, A, Core::Uplo::Lower),
                         std::invalid_argument);
            EXPECT_THROW(SushiBLAS::BLASOps(*engine).syr2(x, y, A, Core::Uplo::Lower),
                         std::invalid_argument);
            EXPECT_THROW(SushiBLAS::BLASOps(*engine).trmv(A, x, Core::Uplo::Lower),
                         std::invalid_argument);
            EXPECT_THROW(SushiBLAS::BLASOps(*engine).trsv(A, x, Core::Uplo::Lower),
                         std::invalid_argument);
        }

        TEST_F(Unit_BlasValidationTest, SquareLevel2RoutinesRejectNonSquareA)
        {
            auto A = engine->create_tensor({2, 3});
            auto x = engine->create_tensor({3});
            auto y = engine->create_tensor({3});

            EXPECT_THROW(SushiBLAS::BLASOps(*engine).symv(A, x, y, Core::Uplo::Lower),
                         std::invalid_argument);
            EXPECT_THROW(SushiBLAS::BLASOps(*engine).syr(x, A, Core::Uplo::Lower),
                         std::invalid_argument);
            EXPECT_THROW(SushiBLAS::BLASOps(*engine).syr2(x, y, A, Core::Uplo::Lower),
                         std::invalid_argument);
            EXPECT_THROW(SushiBLAS::BLASOps(*engine).trmv(A, x, Core::Uplo::Lower),
                         std::invalid_argument);
            EXPECT_THROW(SushiBLAS::BLASOps(*engine).trsv(A, x, Core::Uplo::Lower),
                         std::invalid_argument);
        }

        TEST_F(Unit_BlasValidationTest, Level2RejectsVectorOfWrongLength)
        {
            auto A = engine->create_tensor({3, 3});
            auto x = engine->create_tensor({2});
            auto y = engine->create_tensor({3});

            EXPECT_THROW(SushiBLAS::BLASOps(*engine).gemv(A, x, y), std::invalid_argument);
            EXPECT_THROW(SushiBLAS::BLASOps(*engine).ger(x, y, A), std::invalid_argument);
            EXPECT_THROW(SushiBLAS::BLASOps(*engine).symv(A, x, y, Core::Uplo::Lower),
                         std::invalid_argument);
            EXPECT_THROW(SushiBLAS::BLASOps(*engine).syr(x, A, Core::Uplo::Lower),
                         std::invalid_argument);
            EXPECT_THROW(SushiBLAS::BLASOps(*engine).syr2(x, y, A, Core::Uplo::Lower),
                         std::invalid_argument);
            EXPECT_THROW(SushiBLAS::BLASOps(*engine).trmv(A, x, Core::Uplo::Lower),
                         std::invalid_argument);
            EXPECT_THROW(SushiBLAS::BLASOps(*engine).trsv(A, x, Core::Uplo::Lower),
                         std::invalid_argument);
        }

        TEST_F(Unit_BlasValidationTest, TrsvRejectsStridedB)
        {
            auto A = engine->create_tensor({3, 3});
            auto parent = engine->create_tensor({6});
            const std::vector<int64_t> dims = {3};
            const std::array<int64_t, Core::K_MAX_TENSOR_RANK> strides = {2};
            Tensor b(parent.storage, dims, strides, Core::DataType::Float32, 0,
                     Core::Layout::RowMajor);

            EXPECT_THROW(SushiBLAS::BLASOps(*engine).trsv(A, b, Core::Uplo::Lower),
                         std::invalid_argument);
        }

        TEST_F(Unit_BlasValidationTest, FlagsRejectValueThatIsNoEnumerator)
        {
            auto A = engine->create_tensor({3, 3});
            auto B = engine->create_tensor({3, 3});
            auto x = engine->create_tensor({3});
            auto y = engine->create_tensor({3});

            EXPECT_THROW(SushiBLAS::BLASOps(*engine).symv(A, x, y, static_cast<Core::Uplo>(0)),
                         std::invalid_argument);
            EXPECT_THROW(SushiBLAS::BLASOps(*engine).gemv(A, x, y, static_cast<Core::Transpose>(0)),
                         std::invalid_argument);
            EXPECT_THROW(SushiBLAS::BLASOps(*engine).trmv(A, x, Core::Uplo::Lower,
                                                          Core::Transpose::NoTrans,
                                                          static_cast<Core::Diag>(0)),
                         std::invalid_argument);
            EXPECT_THROW(SushiBLAS::BLASOps(*engine).trsm(A, B, static_cast<Core::Side>(0),
                                                          Core::Uplo::Lower),
                         std::invalid_argument);
        }

        TEST_F(Unit_BlasValidationTest, RealTensorsRejectScalarWithImaginaryPart)
        {
            auto A = engine->create_tensor({3, 3});
            auto B = engine->create_tensor({3, 3});
            auto C = engine->create_tensor({3, 3});
            auto x = engine->create_tensor({3});
            auto y = engine->create_tensor({3});

            EXPECT_THROW(SushiBLAS::BLASOps(*engine).axpy(K_COMPLEX_ALPHA, x, y),
                         std::invalid_argument);
            EXPECT_THROW(SushiBLAS::BLASOps(*engine).scal(K_COMPLEX_ALPHA, x),
                         std::invalid_argument);
            EXPECT_THROW(SushiBLAS::BLASOps(*engine).gemv(A, x, y, Core::Transpose::NoTrans,
                                                          K_COMPLEX_ALPHA),
                         std::invalid_argument);
            EXPECT_THROW(SushiBLAS::BLASOps(*engine).gemm(A, B, C, Core::Transpose::NoTrans,
                                                          Core::Transpose::NoTrans,
                                                          K_COMPLEX_ALPHA),
                         std::invalid_argument);
            EXPECT_THROW(SushiBLAS::BLASOps(*engine).syrk(A, C, Core::Uplo::Lower,
                                                          Core::Transpose::NoTrans,
                                                          K_COMPLEX_ALPHA),
                         std::invalid_argument);
            EXPECT_THROW(SushiBLAS::BLASOps(*engine).trsm(A, B, Core::Side::Left,
                                                          Core::Uplo::Lower,
                                                          Core::Transpose::NoTrans,
                                                          Core::Diag::NonUnit, K_COMPLEX_ALPHA),
                         std::invalid_argument);
        }

        TEST_F(Unit_BlasValidationTest, DotAndGerRejectComplexTensors)
        {
            auto A = engine->create_tensor({3, 3}, Core::DataType::Complex32);
            auto x = engine->create_tensor({3}, Core::DataType::Complex32);
            auto y = engine->create_tensor({3}, Core::DataType::Complex32);
            auto result = engine->create_tensor({1}, Core::DataType::Complex32);

            EXPECT_THROW(SushiBLAS::BLASOps(*engine).dot(x, y, result), std::invalid_argument);
            EXPECT_THROW(SushiBLAS::BLASOps(*engine).ger(x, y, A), std::invalid_argument);
        }

        TEST_F(Unit_BlasValidationTest, SyrkAndHerkRejectTheOtherTransposeOnComplexTensors)
        {
            auto A = engine->create_tensor({3, 2}, Core::DataType::Complex32);
            auto C = engine->create_tensor({2, 2}, Core::DataType::Complex32);

            EXPECT_THROW(SushiBLAS::BLASOps(*engine).syrk(A, C, Core::Uplo::Lower,
                                                          Core::Transpose::ConjTrans),
                         std::invalid_argument);
            EXPECT_THROW(SushiBLAS::BLASOps(*engine).herk(A, C, Core::Uplo::Lower,
                                                          Core::Transpose::Trans),
                         std::invalid_argument);
        }

        TEST_F(Unit_BlasValidationTest, ExecuteGemmUnderTheRealDomainRejectsComplexTensors)
        {
            auto A = engine->create_tensor({2, 2}, Core::DataType::Complex32);
            auto B = engine->create_tensor({2, 2}, Core::DataType::Complex32);
            auto C = engine->create_tensor({2, 2}, Core::DataType::Complex32);

            EXPECT_THROW(Dispatch::execute_gemm<Dispatch::Domain::Real>(
                             engine->get_recorder(), A, B, C, Core::Transpose::NoTrans,
                             Core::Transpose::NoTrans, 1.0, 0.0, "test.gemm_real_domain",
                             "test.gemm_real_domain"_op, Kernels::GemmIdentityEpilogue{}),
                         std::invalid_argument);
        }
    } // namespace Test
} // namespace SushiBLAS
