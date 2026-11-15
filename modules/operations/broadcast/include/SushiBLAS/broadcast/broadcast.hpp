/**************************************************************************/
/* broadcast.hpp                                                          */
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
 * @file broadcast.hpp
 * @brief Declares BroadcastOps, the arithmetic between a tensor and a vector broadcast along its rows or columns.
 * @author Mustafa Garip
 */

#pragma once

#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiBLAS/engine/task_handle.hpp>
#include <SushiBLAS/engine/task_recorder.hpp>
#include <SushiBLAS/broadcast/broadcast_operand.hpp>

namespace SushiBLAS
{
    /**
     * @class BroadcastOps
     * @brief Records arithmetic between a tensor and a vector broadcast along its rows or columns.
     * @note A, C and the vector are row-major, dense and of one dtype. C is A itself or lies
     * apart from it and from the vector. The vector's element count is a product of leading
     * extents of A's shape for a leading vector, or that of the rest for a trailing one.
     * @see modules/operations/broadcast/README.md
     */
    class BroadcastOps final
    {
        public:
            /** @brief Builds a view that records through recorder, which must outlive the view. */
            explicit BroadcastOps(TaskRecorder& recorder) noexcept : recorder_(recorder) {}

            BroadcastOps(const BroadcastOps&) = default;
            BroadcastOps& operator=(const BroadcastOps&) = delete;
            BroadcastOps(BroadcastOps&&) = default;
            BroadcastOps& operator=(BroadcastOps&&) = delete;
            ~BroadcastOps() = default;

            /**
             * @brief Computes `C = A + v`.
             * @param axis Leading: `v[o]` meets every element of row o. Trailing: `v[n]` meets
             *        element n of every row.
             * @return The handle of the recorded task, or an empty one when A has no element.
             * @throws std::invalid_argument at record time on an operand violation.
             */
            TaskHandle add(const Tensor& A, const Tensor& v, Tensor& C, BroadcastAxis axis);

            /** @brief Computes `C = A - v`; the parameters and the refusals are those of add. */
            TaskHandle sub(const Tensor& A, const Tensor& v, Tensor& C, BroadcastAxis axis);

            /** @brief Computes `C = A * v`; the parameters and the refusals are those of add. */
            TaskHandle mul(const Tensor& A, const Tensor& v, Tensor& C, BroadcastAxis axis);

            /** @brief Computes `C = A / v`; the parameters and the refusals are those of add. */
            TaskHandle div(const Tensor& A, const Tensor& v, Tensor& C, BroadcastAxis axis);

        private:
            /// @brief The recorder every call records through.
            TaskRecorder& recorder_;
    };
} // namespace SushiBLAS
