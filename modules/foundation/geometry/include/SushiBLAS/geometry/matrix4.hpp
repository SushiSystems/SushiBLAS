/**************************************************************************/
/* matrix4.hpp                                                            */
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
 * @file matrix4.hpp
 * @brief Declares the column-major 4x4 matrix and its product.
 * @author Mustafa Garip
 */

#pragma once

#include <type_traits>

namespace SushiBLAS
{
    namespace Geometry
    {
        /**
         * @brief Holds a 4x4 matrix in column-major storage, element (r, c) at m[c * 4 + r].
         * @tparam T A floating-point element type.
         * @see modules/foundation/geometry/README.md
         */
        template<typename T>
        struct Matrix4T
        {
            /// @brief The sixteen elements, column by column; the identity by default.
            T m[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
        };

        static_assert(std::is_trivially_copyable<Matrix4T<float>>::value &&
                          std::is_trivially_copyable<Matrix4T<double>>::value,
                      "Matrix4T is captured by SYCL kernels and must be trivially copyable.");
        static_assert(std::is_standard_layout<Matrix4T<float>>::value &&
                          std::is_standard_layout<Matrix4T<double>>::value,
                      "Matrix4T is stored in buffers and must be standard layout.");
        static_assert(sizeof(Matrix4T<float>) == 16 * sizeof(float) &&
                          sizeof(Matrix4T<double>) == 16 * sizeof(double),
                      "Matrix4T holds sixteen elements and no padding.");

        /** @brief Returns the matrix product a * b. */
        template<typename T>
        inline Matrix4T<T> mul(const Matrix4T<T>& a, const Matrix4T<T>& b) noexcept
        {
            Matrix4T<T> r{};
            for (int c = 0; c < 4; ++c)
                for (int row = 0; row < 4; ++row)
                {
                    T sum = 0;
                    for (int k = 0; k < 4; ++k)
                        sum += a.m[k * 4 + row] * b.m[c * 4 + k];
                    r.m[c * 4 + row] = sum;
                }
            return r;
        }
    } // namespace Geometry
} // namespace SushiBLAS
