/**************************************************************************/
/* test_shape_check.cpp                                                   */
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
 * @file test_shape_check.cpp
 * @brief Tests that the shape checks of the dense BLAS routines refuse what the routines refuse,
 *        on tensors that own no storage.
 * @author Mustafa Garip
 */

#include <array>
#include <string>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/blas/shape_check.hpp>
#include <SushiBLAS/tensor/tensor.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            using Core::DataType;
            using Core::Layout;

            /** @brief Returns a dense tensor of a shape that owns no storage. */
            Tensor shape_of(const std::vector<int64_t>& dims, DataType dtype = DataType::Float32,
                            Layout layout = Layout::RowMajor)
            {
                return Tensor::make_shape_only(
                    SushiRuntime::span<const int64_t>(dims.data(), dims.size()), dtype, layout);
            }

            /** @brief Returns a rank-1 shape-only view whose elements lie stride apart. */
            Tensor strided_vector(int64_t length, int64_t stride)
            {
                const std::vector<int64_t> dims = {length};
                std::array<int64_t, Core::K_MAX_TENSOR_RANK> strides{};
                strides[0] = stride;
                return Tensor::make_shape_only(
                    SushiRuntime::span<const int64_t>(dims.data(), dims.size()), strides,
                    DataType::Float32, Layout::RowMajor);
            }

            /**
             * @brief Expects check to throw std::invalid_argument naming the operation, the
             *        operand and a fragment of the rule.
             */
            void expect_refusal(const std::function<void()>& check, const std::string& op,
                                const std::string& operand, const std::string& fragment)
            {
                try
                {
                    check();
                    ADD_FAILURE() << "the check accepted the operands";
                }
                catch (const std::invalid_argument& error)
                {
                    const std::string message = error.what();
                    EXPECT_NE(message.find(op + ": operand '" + operand + "'"), std::string::npos)
                        << message;
                    EXPECT_NE(message.find(fragment), std::string::npos) << message;
                }
            }

            constexpr const char* K_GEMM = "blas.lvl3.gemm";
            constexpr const char* K_SYRK = "blas.lvl3.syrk";
            constexpr const char* K_TRSM = "blas.lvl3.trsm";
            constexpr const char* K_TRSV = "blas.lvl2.trsv";
            constexpr const char* K_DOT = "blas.lvl1.dot";
            constexpr const char* K_SCAL = "blas.lvl1.scal";
            constexpr Core::Transpose K_NO = Core::Transpose::NoTrans;
        } // namespace

        TEST(Small_ShapeCheckGemm, AnInnerDimensionThatDiffersIsRefused)
        {
            const Tensor A = shape_of({2, 3});
            const Tensor B = shape_of({4, 2});
            const Tensor C = shape_of({2, 2});

            expect_refusal([&] { (void)Dispatch::check_gemm_shapes(A, B, C, K_NO, K_NO, K_GEMM); },
                           K_GEMM, "B", "rows in op(B)");
        }

        TEST(Small_ShapeCheckGemm, OperandsOfMixedLayoutsAreRefused)
        {
            const Tensor A = shape_of({2, 2});
            const Tensor B = shape_of({2, 2}, DataType::Float32, Layout::ColumnMajor);
            const Tensor C = shape_of({2, 2});

            expect_refusal([&] { (void)Dispatch::check_gemm_shapes(A, B, C, K_NO, K_NO, K_GEMM); },
                           K_GEMM, "B", "share a layout");
        }

        TEST(Small_ShapeCheckGemm, ABatchCountThatDiffersIsRefused)
        {
            const Tensor A = shape_of({3, 2, 2});
            const Tensor B = shape_of({2, 2, 2});
            const Tensor C = shape_of({2, 2, 2});

            expect_refusal([&] { (void)Dispatch::check_gemm_shapes(A, B, C, K_NO, K_NO, K_GEMM); },
                           K_GEMM, "A", "holds 3 matrices");
        }

        TEST(Small_ShapeCheckGemm, ATransposedViewIsRefused)
        {
            const Tensor A = shape_of({2, 3}).transpose(0, 1);
            const Tensor B = shape_of({2, 2});
            const Tensor C = shape_of({3, 2});

            expect_refusal([&] { (void)Dispatch::check_gemm_shapes(A, B, C, K_NO, K_NO, K_GEMM); },
                           K_GEMM, "A", "strided views");
        }

        TEST(Small_ShapeCheckGemm, BatchAxesThatDifferWithAnEqualProductAreRefused)
        {
            const Tensor A = shape_of({2, 3, 2, 2});
            const Tensor B = shape_of({2, 2});
            const Tensor C = shape_of({3, 2, 2, 2});

            expect_refusal([&] { (void)Dispatch::check_gemm_shapes(A, B, C, K_NO, K_NO, K_GEMM); },
                           K_GEMM, "A", "batch axes that differ");
        }

        TEST(Small_ShapeCheckGemm, ARealDomainRefusesAComplexTensor)
        {
            const Tensor A = shape_of({2, 2}, DataType::Complex32);

            expect_refusal([&] { Dispatch::require_gemm_domain(true, A, K_GEMM); }, K_GEMM, "A",
                           "declared for real tensors");
        }

        TEST(Small_ShapeCheckSyrk, ANonSquareCIsRefused)
        {
            const Tensor A = shape_of({3, 2});
            const Tensor C = shape_of({3, 2});

            expect_refusal([&] { (void)Dispatch::check_syrk_shapes(A, C, K_NO, K_SYRK); }, K_SYRK,
                           "C", "");
        }

        TEST(Small_ShapeCheckSyrk, BatchAxesThatDifferAreRefused)
        {
            const Tensor A = shape_of({2, 3, 2, 2});
            const Tensor C = shape_of({6, 2, 2});

            expect_refusal([&] { (void)Dispatch::check_syrk_shapes(A, C, K_NO, K_SYRK); }, K_SYRK,
                           "A", "batch axes that differ");
        }

        TEST(Small_ShapeCheckTrsm, ANonSquareAIsRefused)
        {
            const Tensor A = shape_of({2, 3});
            const Tensor B = shape_of({2, 2});

            expect_refusal(
                [&] { (void)Dispatch::check_trsm_shapes(A, B, Core::Side::Left, K_TRSM); }, K_TRSM,
                "A", "");
        }

        TEST(Small_ShapeCheckTrsm, BatchAxesThatDifferAreRefused)
        {
            const Tensor A = shape_of({2, 3, 2, 2});
            const Tensor B = shape_of({6, 2, 2});

            expect_refusal(
                [&] { (void)Dispatch::check_trsm_shapes(A, B, Core::Side::Left, K_TRSM); }, K_TRSM,
                "A", "batch axes that differ");
        }

        TEST(Small_ShapeCheckTrsv, AStridedRightHandSideIsRefused)
        {
            const Tensor A = shape_of({3, 3});
            const Tensor b = strided_vector(3, 2);

            expect_refusal([&] { (void)Dispatch::check_trsv_shapes(A, b, K_TRSV); }, K_TRSV, "b",
                           "contiguous vector");
        }

        TEST(Small_ShapeCheckDot, AResultOfAnotherDtypeIsRefused)
        {
            const Tensor x = shape_of({4});
            const Tensor y = shape_of({4});
            const Tensor result = shape_of({1}, DataType::Half);

            expect_refusal([&] { (void)Dispatch::check_dot_shapes(x, y, result, K_DOT); }, K_DOT,
                           "result", "has dtype");
        }

        TEST(Small_ShapeCheckScal, AVectorOfStrideZeroIsRefused)
        {
            const Tensor x = strided_vector(4, 0);

            expect_refusal([&] { (void)Dispatch::describe_vector_shape(x, K_SCAL, "x"); }, K_SCAL,
                           "x", "must be positive");
        }

        TEST(Small_ShapeCheckScal, ANonContiguousVectorOfRankTwoIsRefused)
        {
            const Tensor x = shape_of({2, 3}).transpose(0, 1);

            expect_refusal([&] { (void)Dispatch::describe_vector_shape(x, K_SCAL, "x"); }, K_SCAL,
                           "x", "has rank 2");
        }
    } // namespace Test
} // namespace SushiBLAS
