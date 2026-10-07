/**************************************************************************/
/* level2.hpp                                                             */
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
 * @file level2.hpp
 * @brief Declares Level2, the matrix-vector BLAS operations.
 * @author Mustafa Garip
 */

#pragma once

#include <SushiBLAS/tensor.hpp>
#include <SushiBLAS/engine/task_handle.hpp>
#include <SushiBLAS/engine/task_recorder.hpp>

namespace SushiBLAS
{
    /**
     * @brief Records the Level 2 BLAS routines.
     * @see include/SushiBLAS/engine/blas/README.md
     */
    class Level2
    {
        public:
            /**
             * @brief Computes y = alpha * op(A) * x + beta * y.
             * @param transA Uses A^T as op(A) when true.
             */
            TaskHandle gemv(const Tensor& A, const Tensor& x, Tensor& y,
                            bool transA = false, double alpha = 1.0, double beta = 0.0);

            /**
             * @brief Computes the rank-1 update A = alpha * x * y^H + A.
             */
            TaskHandle ger(const Tensor& x, const Tensor& y, Tensor& A, double alpha = 1.0);

            /**
             * @brief Computes y = alpha * A * x + beta * y for a symmetric A.
             * @param upper Reads the upper triangle of A when true, the lower when false.
             */
            TaskHandle symv(const Tensor& A, const Tensor& x, Tensor& y,
                            bool upper = false, double alpha = 1.0, double beta = 0.0);

            /**
             * @brief Computes x = op(A) * x for a triangular A.
             * @param upper Treats A as upper triangular when true, lower when false.
             * @param unit_diag Treats the diagonal of A as all ones when true.
             */
            TaskHandle trmv(const Tensor& A, Tensor& x,
                            bool upper = false, bool transA = false, bool unit_diag = false);

            /**
             * @brief Solves op(A) * x = b for a triangular A, overwriting b with x.
             * @param upper Treats A as upper triangular when true, lower when false.
             * @param unit_diag Treats the diagonal of A as all ones when true.
             */
            TaskHandle trsv(const Tensor& A, Tensor& b,
                            bool upper = false, bool transA = false, bool unit_diag = false);

            /**
             * @brief Computes the symmetric rank-1 update A = alpha * x * x^H + A.
             * @param upper Updates the upper triangle when true, the lower when false.
             */
            TaskHandle syr(const Tensor& x, Tensor& A, bool upper = false, double alpha = 1.0);

            /**
             * @brief Computes the symmetric rank-2 update A = alpha * x * y^H + alpha * y * x^H + A.
             * @param upper Updates the upper triangle when true, the lower when false.
             */
            TaskHandle syr2(const Tensor& x, const Tensor& y, Tensor& A, bool upper = false, double alpha = 1.0);

        protected:
            /** @brief Binds the level to the recorder its routines record through. */
            explicit Level2(TaskRecorder& recorder) noexcept : recorder_(recorder) {}

            Level2(const Level2&) = default;
            Level2& operator=(const Level2&) = delete;
            Level2(Level2&&) = default;
            Level2& operator=(Level2&&) = delete;
            ~Level2() = default;

        private:
            /// @brief The recorder every routine of this level records through.
            TaskRecorder& recorder_;
    };
} // namespace SushiBLAS
