/**************************************************************************/
/* level3.hpp                                                             */
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
 * @file level3.hpp
 * @brief Declares Level3, the matrix-matrix BLAS operations.
 * @author Mustafa Garip
 */

#pragma once

#include <SushiBLAS/tensor.hpp>
#include <SushiBLAS/engine/task_handle.hpp>
#include <SushiBLAS/engine/task_recorder.hpp>

namespace SushiBLAS
{
    /**
     * @brief Records the Level 3 BLAS routines.
     * @see include/SushiBLAS/engine/blas/README.md
     */
    class Level3
    {
        public:
            /**
             * @brief Computes C = alpha * op(A) * op(B) + beta * C, batched when the operands have rank > 2.
             */
            TaskHandle gemm(const Tensor& A, const Tensor& B, Tensor& C,
                            bool transA = false, bool transB = false,
                            double alpha = 1.0, double beta = 0.0);

            /**
             * @brief Solves a triangular system with multiple right-hand sides, overwriting B with X.
             * @param left_side Solves op(A) * X = alpha * B when true, X * op(A) = alpha * B when false.
             * @param upper Treats A as upper triangular when true, lower when false.
             */
            TaskHandle trsm(const Tensor& A, Tensor& B,
                            bool left_side = true, bool upper = false,
                            bool transA = false, bool unit_diag = false,
                            double alpha = 1.0);

            /**
             * @brief Computes the symmetric rank-k update of C from A.
             * @param upper Stores the result in the upper triangle of C when true, the lower when false.
             * @param transA Computes alpha * A^T * A + beta * C when true, alpha * A * A^T + beta * C when false.
             */
            TaskHandle syrk(const Tensor& A, Tensor& C,
                            bool upper = false, bool transA = false,
                            double alpha = 1.0, double beta = 0.0);

        protected:
            /** @brief Binds the level to the recorder its routines record through. */
            explicit Level3(TaskRecorder& recorder) noexcept : recorder_(recorder) {}

            Level3(const Level3&) = default;
            Level3& operator=(const Level3&) = delete;
            Level3(Level3&&) = default;
            Level3& operator=(Level3&&) = delete;
            ~Level3() = default;

        private:
            /// @brief The recorder every routine of this level records through.
            TaskRecorder& recorder_;
    };
} // namespace SushiBLAS
