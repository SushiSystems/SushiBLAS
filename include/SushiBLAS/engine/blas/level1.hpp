/**************************************************************************/
/* level1.hpp                                                             */
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
 * @file level1.hpp
 * @brief Declares Level1, the vector-vector BLAS operations.
 * @author Mustafa Garip
 */

#pragma once

#include <SushiBLAS/tensor.hpp>
#include <SushiBLAS/engine/task_recorder.hpp>
#include <SushiBLAS/engine/blas/scalar.hpp>

namespace SushiBLAS
{
    /**
     * @brief Records the BLAS Level 1 vector-vector operations through a task recorder.
     * @see include/SushiBLAS/engine/blas/README.md
     */
    class Level1
    {
        public:
            /**
             * @brief Records y = alpha * x + y.
             * @throws std::invalid_argument on an operand or a scalar the call does not accept.
             */
            TaskHandle axpy(BlasScalar alpha, const Tensor& x, Tensor& y);

            /**
             * @brief Records x = alpha * x.
             * @throws std::invalid_argument on an operand or a scalar the call does not accept.
             */
            TaskHandle scal(BlasScalar alpha, Tensor& x);

            /**
             * @brief Records a copy of every element of x into y.
             * @throws std::invalid_argument on an operand the call does not accept.
             */
            TaskHandle copy(const Tensor& x, Tensor& y);

            /**
             * @brief Records an exchange of the elements of x and y.
             * @throws std::invalid_argument on an operand the call does not accept.
             */
            TaskHandle swap(Tensor& x, Tensor& y);

            /**
             * @brief Records the dot product x^T * y of two real vectors into result.
             * @param result Scalar tensor of the operands' dtype.
             * @throws std::invalid_argument on a complex operand, or one the call does not accept.
             */
            TaskHandle dot(const Tensor& x, const Tensor& y, Tensor& result);

            /**
             * @brief Records the unconjugated dot product x^T * y into result.
             * @param result Scalar tensor of the operands' dtype.
             * @throws std::invalid_argument on an operand the call does not accept.
             */
            TaskHandle dotu(const Tensor& x, const Tensor& y, Tensor& result);

            /**
             * @brief Records the conjugated dot product x^H * y into result.
             * @param result Scalar tensor of the operands' dtype.
             * @throws std::invalid_argument on an operand the call does not accept.
             */
            TaskHandle dotc(const Tensor& x, const Tensor& y, Tensor& result);

            /**
             * @brief Records the Euclidean norm sqrt(sum(|x_i|^2)) into result.
             * @param result Scalar tensor of the real component type of x.
             * @throws std::invalid_argument on an operand the call does not accept.
             */
            TaskHandle nrm2(const Tensor& x, Tensor& result);

            /**
             * @brief Records sum(|Re(x_i)| + |Im(x_i)|) into result.
             * @param result Scalar tensor of the real component type of x.
             * @throws std::invalid_argument on an operand the call does not accept.
             */
            TaskHandle asum(const Tensor& x, Tensor& result);

            /**
             * @brief Records the smallest index at which |Re(x_i)| + |Im(x_i)| is largest.
             * @param result Scalar INT64 tensor; the index is zero-based.
             * @throws std::invalid_argument on an operand the call does not accept.
             */
            TaskHandle iamax(const Tensor& x, Tensor& result);

            /**
             * @brief Records the plane rotation x_i = c * x_i + s * y_i, y_i = c * y_i - s * x_i.
             * @param c Real for every dtype, as is s.
             * @throws std::invalid_argument on an operand the call does not accept.
             */
            TaskHandle rot(Tensor& x, Tensor& y, double c, double s);

        protected:
            /** @brief Binds the recorder every operation records on; it must outlive this view. */
            explicit Level1(TaskRecorder& recorder) noexcept
                : recorder_(recorder)
            {

            }

            /** @brief Copies the binding. */
            Level1(const Level1&) noexcept = default;

            /** @brief Moves the binding. */
            Level1(Level1&&) noexcept = default;

            /** @brief Refused: a view cannot be re-bound. */
            Level1& operator=(const Level1&) = delete;

            /** @brief Refused: a view cannot be re-bound. */
            Level1& operator=(Level1&&) = delete;

            /** @brief Destroys the view; the recorder is not owned. */
            ~Level1() = default;

        private:
            /// @brief The recorder every operation of this level records on.
            TaskRecorder& recorder_;
    };
} // namespace SushiBLAS
