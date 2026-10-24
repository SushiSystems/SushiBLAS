/**************************************************************************/
/* blend.hpp                                                              */
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

/**
 * @file blend.hpp
 * @brief Declares BlendOps, the operations that blend weighted transforms and apply them to points.
 * @author Mustafa Garip
 */

#pragma once

#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiBLAS/engine/task_handle.hpp>
#include <SushiBLAS/engine/task_recorder.hpp>

namespace SushiBLAS
{
    /**
     * @class BlendOps
     * @brief Records operations that move each point by a weighted blend of indexed transforms.
     */
    class BlendOps final
    {
        public:
            /** @brief Builds a view that records through recorder, which must outlive the view. */
            explicit BlendOps(TaskRecorder& recorder) noexcept : recorder_(recorder) {}

            BlendOps(const BlendOps&) = default;
            BlendOps& operator=(const BlendOps&) = delete;
            BlendOps(BlendOps&&) = default;
            BlendOps& operator=(BlendOps&&) = delete;
            ~BlendOps() = default;

            /**
             * @brief Moves every point by a weighted sum of transforms: `out[p] = sum_j weights[p][j] * transforms[indices[p][j]] * points[p]`.
             * @param points Row-major `[P, 3]` positions (taken as `w = 1`) or `[P, 4]` homogeneous points.
             * @param transforms Row-major `[T, 4, 4]` or `[T, 3, 4]`, applied to a column vector; a 3-row transform ends in the implicit row `(0, 0, 0, 1)`.
             * @param indices The `[P, J]` INT32 or INT64 transform of each influence; one outside `[0, T)` adds no term. weights has its shape and the dtype of points.
             * @param out Shaped like points; it may share no byte with an input.
             * @throws std::invalid_argument at record time on a rank, dtype or extent mismatch, an operand that is not dense and row-major, or an out that overlaps an input.
             */
            TaskHandle blend_transforms(const Tensor& points, const Tensor& transforms,
                                        const Tensor& indices, const Tensor& weights,
                                        Tensor& out);

        private:
            /// @brief The recorder every operation of this view records through.
            TaskRecorder& recorder_;
    };
} // namespace SushiBLAS
