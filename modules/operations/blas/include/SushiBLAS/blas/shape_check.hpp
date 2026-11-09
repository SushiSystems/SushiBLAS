/**************************************************************************/
/* shape_check.hpp                                                        */
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
 * @file shape_check.hpp
 * @brief Declares the shape checks of the dense BLAS routines, which read shape, strides and
 *        dtype and need no queue.
 * @author Mustafa Garip
 */

#pragma once

#include <utility>
#include <cstdint>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/core/blas_types.hpp>
#include <SushiBLAS/engine/operand_check.hpp>
#include <SushiBLAS/operands/argument_check.hpp>
#include <SushiBLAS/operands/scalar_result.hpp>
#include <SushiBLAS/operands/matrix_operand.hpp>
#include <SushiBLAS/operands/vector_operand.hpp>

namespace SushiBLAS
{
    namespace Dispatch
    {
        /**
         * @brief Throws unless a tensor has storage on a device that computes in its dtype.
         * @throws std::invalid_argument on missing storage.
         * @throws UnsupportedExecution on a dtype the tensor's device cannot compute in.
         */
        inline void require_operand_storage(const Tensor& t, const char* op, const char* operand)
        {
            require_storage(t, op, operand);
            require_device_dtype(t, op, operand);
        }

        /** @brief Holds what GEMM takes from the shapes of its three operands. */
        struct GemmGeometry
        {
            int64_t m = 0;          /**< Rows of op(A) and C. */
            int64_t n = 0;          /**< Columns of op(B) and C. */
            int64_t k = 0;          /**< Columns of op(A), rows of op(B). */
            int64_t batch_size = 1; /**< Matrices the call computes. */
            int64_t stride_a = 0;   /**< Element distance between matrices of A. */
            int64_t stride_b = 0;   /**< Element distance between matrices of B. */
            int64_t stride_c = 0;   /**< Element distance between matrices of C. */
            MatrixOperand a;        /**< The geometry of A. */
            MatrixOperand b;        /**< The geometry of B. */
            MatrixOperand c;        /**< The geometry of C. */
        };

        /**
         * @brief Throws unless C = op(A) * op(B) has matching layouts, extents and batch axes.
         * @return The extents, strides and operand geometry the call records with.
         * @throws std::invalid_argument naming the operand and the rule that fails.
         */
        [[nodiscard]] inline GemmGeometry check_gemm_shapes(const Tensor& A, const Tensor& B,
                                                            const Tensor& C,
                                                            Core::Transpose transA,
                                                            Core::Transpose transB,
                                                            const char* name)
        {
            GemmGeometry g;
            g.a = describe_matrix_shape(A, name, "A");
            g.b = describe_matrix_shape(B, name, "B");
            g.c = describe_matrix_shape(C, name, "C");
            require_same_layout(g.a, g.c, name, "A", "C");
            require_same_layout(g.b, g.c, name, "B", "C");

            const bool a_transposed = Core::is_transposed(transA);
            const bool b_transposed = Core::is_transposed(transB);
            g.m = g.c.rows;
            g.n = g.c.cols;
            g.k = a_transposed ? g.a.rows : g.a.cols;
            require_matrix_extent(a_transposed ? g.a.cols : g.a.rows, g.m, "rows in op(A)", name,
                                  "A");
            require_matrix_extent(b_transposed ? g.b.cols : g.b.rows, g.k, "rows in op(B)", name,
                                  "B");
            require_matrix_extent(b_transposed ? g.b.rows : g.b.cols, g.n, "columns in op(B)", name,
                                  "B");

            g.batch_size = g.c.batch_count;
            g.stride_a = batch_stride(g.a, g.batch_size, name, "A");
            g.stride_b = batch_stride(g.b, g.batch_size, name, "B");
            g.stride_c = batch_stride(g.c, g.batch_size, name, "C");
            require_same_batch_axes(A, g.a, C, g.c, name, "A", "C");
            require_same_batch_axes(B, g.b, C, g.c, name, "B", "C");
            return g;
        }

        /** @brief Holds what SYRK and HERK take from the shapes of their two operands. */
        struct SyrkGeometry
        {
            int64_t n = 0;          /**< Order of C. */
            int64_t k = 0;          /**< The summed extent of op(A). */
            int64_t batch_size = 1; /**< Matrices the call computes. */
            int64_t stride_a = 0;   /**< Element distance between matrices of A. */
            int64_t stride_c = 0;   /**< Element distance between matrices of C. */
            MatrixOperand a;        /**< The geometry of A. */
            MatrixOperand c;        /**< The geometry of C. */
        };

        /**
         * @brief Throws unless C is square, shares the layout of A and the rows of op(A), and the
         *        batch axes agree.
         * @throws std::invalid_argument naming the operand and the rule that fails.
         */
        [[nodiscard]] inline SyrkGeometry check_syrk_shapes(const Tensor& A, const Tensor& C,
                                                            Core::Transpose trans, const char* op)
        {
            SyrkGeometry g;
            g.a = describe_matrix_shape(A, op, "A");
            g.c = describe_matrix_shape(C, op, "C");
            require_same_layout(g.a, g.c, op, "A", "C");
            require_square(g.c, op, "C");

            const bool transposed = Core::is_transposed(trans);
            g.n = g.c.rows;
            g.k = transposed ? g.a.rows : g.a.cols;
            require_matrix_extent(transposed ? g.a.cols : g.a.rows, g.n, "rows in op(A)", op, "A");

            g.batch_size = g.c.batch_count;
            g.stride_a = batch_stride(g.a, g.batch_size, op, "A");
            g.stride_c = batch_stride(g.c, g.batch_size, op, "C");
            require_same_batch_axes(A, g.a, C, g.c, op, "A", "C");
            return g;
        }

        /** @brief Holds what TRSM takes from the shapes of its two operands. */
        struct TrsmGeometry
        {
            int64_t m = 0;          /**< Rows of B. */
            int64_t n = 0;          /**< Columns of B. */
            int64_t batch_size = 1; /**< Matrices the call computes. */
            int64_t stride_a = 0;   /**< Element distance between matrices of A. */
            int64_t stride_b = 0;   /**< Element distance between matrices of B. */
            MatrixOperand a;        /**< The geometry of A. */
            MatrixOperand b;        /**< The geometry of B. */
        };

        /**
         * @brief Throws unless A is square, shares the layout of B and the order the side needs,
         *        and the batch axes agree.
         * @throws std::invalid_argument naming the operand and the rule that fails.
         */
        [[nodiscard]] inline TrsmGeometry check_trsm_shapes(const Tensor& A, const Tensor& B,
                                                            Core::Side side, const char* op)
        {
            TrsmGeometry g;
            g.a = describe_matrix_shape(A, op, "A");
            g.b = describe_matrix_shape(B, op, "B");
            require_same_layout(g.a, g.b, op, "A", "B");
            require_square(g.a, op, "A");

            const bool left = (side == Core::Side::Left);
            g.m = g.b.rows;
            g.n = g.b.cols;
            require_matrix_extent(g.a.rows, left ? g.m : g.n,
                                  left ? "rows against the rows of B"
                                       : "rows against the columns of B",
                                  op, "A");

            g.batch_size = g.b.batch_count;
            g.stride_a = batch_stride(g.a, g.batch_size, op, "A");
            g.stride_b = batch_stride(g.b, g.batch_size, op, "B");
            require_same_batch_axes(A, g.a, B, g.b, op, "A", "B");
            return g;
        }

        /**
         * @brief Throws unless A is an unbatched square matrix and b a contiguous vector of its
         *        order.
         * @return The geometry of A and of b, in that order.
         * @throws std::invalid_argument naming the operand and the rule that fails.
         */
        [[nodiscard]] inline std::pair<MatrixOperand, VectorOperand> check_trsv_shapes(
            const Tensor& A, const Tensor& b, const char* op)
        {
            const MatrixOperand a = describe_matrix_shape(A, op, "A");
            require_unbatched(a, op, "A");
            require_square(a, op, "A");
            const VectorOperand vb = describe_vector_shape(b, op, "b");
            require_unit_increment(vb, op, "b");
            require_vector_length(vb, a.cols, "the order of A", op, "b");
            return {a, vb};
        }

        /**
         * @brief Throws unless x and y share a dtype and an element count, and result is a scalar
         *        of that dtype.
         * @return The geometry of x and of y, in that order.
         * @throws std::invalid_argument naming the operand and the rule that fails.
         */
        [[nodiscard]] inline std::pair<VectorOperand, VectorOperand> check_dot_shapes(
            const Tensor& x, const Tensor& y, const Tensor& result, const char* op)
        {
            require_dtype(y, x.dtype, op, "y");
            const VectorOperand vx = describe_vector_shape(x, op, "x");
            const VectorOperand vy = describe_vector_shape(y, op, "y");
            require_same_element_count(x, y, op, "x", "y");
            Internal::require_scalar_result_shape(result, x.dtype, op);
            return {vx, vy};
        }

        /**
         * @brief Throws if a GEMM declared for real tensors is given a complex one.
         * @param real_only Whether the epilogue was declared for real tensors alone.
         * @throws std::invalid_argument naming the dtype.
         */
        inline void require_gemm_domain(bool real_only, const Tensor& A, const char* name)
        {
            if (real_only && Core::is_complex_dtype(A.dtype))
                Detail::throw_operand_error(name, "A",
                                            "has dtype {}, but the epilogue of this call was "
                                            "declared for real tensors; pass "
                                            "Domain::RealAndComplex when the epilogue is "
                                            "well-formed for std::complex",
                                            Core::to_string(A.dtype));
        }
    } // namespace Dispatch
} // namespace SushiBLAS
