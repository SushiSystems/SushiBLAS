/**************************************************************************/
/* orthonormal_sweep.hpp                                                  */
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
 * @file orthonormal_sweep.hpp
 * @brief Declares the vector layout and the constants the orthonormalisation kernels of both back ends share.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>

namespace SushiBLAS
{
    namespace Kernels
    {
        /**
         * @brief Holds the strided layout of the vectors a sweep orthonormalises.
         * @see modules/foundation/orthonormal/README.md
         */
        struct VectorSet
        {
            int64_t count = 0;       /**< Number of vectors. */
            int64_t length = 0;      /**< Elements per vector. */
            int64_t elem_stride = 0; /**< Element-to-element step within one vector. */
            int64_t vec_stride = 0;  /**< First-element-to-first-element step between vectors. */
        };

        /** @brief Holds how many Gram-Schmidt sweeps run before the vectors are normalised. */
        inline constexpr int K_ORTHONORMAL_SWEEPS = 2;

        /** @brief Returns the squared norm at or below which a vector counts as zero. */
        template<typename T>
        [[nodiscard]] inline constexpr T tiny_norm_sq() noexcept
        {
            return static_cast<T>(1e-12) * static_cast<T>(1e-12);
        }

        /**
         * @brief Returns the coefficient that projects a vector of squared norm @p norm_sq out of another.
         * @param dot The dot product of the two vectors.
         * @return `dot / norm_sq`, or zero when the vector counts as zero.
         */
        template<typename T>
        [[nodiscard]] inline constexpr T projection_coefficient(T dot, T norm_sq) noexcept
        {
            return norm_sq > tiny_norm_sq<T>() ? dot / norm_sq : static_cast<T>(0);
        }

        /**
         * @brief Returns the factor that scales a vector of squared norm @p norm_sq to norm @p gain.
         * @param root The square root of @p norm_sq, taken by the caller with its own library.
         * @return `gain / root`, or @p gain itself when the vector counts as zero.
         */
        template<typename T>
        [[nodiscard]] inline constexpr T normalising_scale(T gain, T norm_sq, T root) noexcept
        {
            return norm_sq > tiny_norm_sq<T>() ? gain / root : gain;
        }
    } // namespace Kernels
} // namespace SushiBLAS
