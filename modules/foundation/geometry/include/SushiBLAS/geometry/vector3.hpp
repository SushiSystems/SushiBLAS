/**************************************************************************/
/* vector3.hpp                                                            */
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
 * @file vector3.hpp
 * @brief Declares the 3-component vector and its dot, cross, length and normalisation.
 * @author Mustafa Garip
 */

#pragma once

#include <type_traits>
#include <SushiBLAS/support/device_math.hpp>

namespace SushiBLAS
{
    /**
     * @brief Holds the fixed-size geometric value types and the functions on them.
     * @see modules/foundation/geometry/README.md
     */
    namespace Geometry
    {
        /**
         * @brief Holds three components of element type T.
         * @tparam T A floating-point element type.
         * @see modules/foundation/geometry/README.md
         */
        template<typename T>
        struct Vector3T
        {
            T x = 0; /**< The first component. */
            T y = 0; /**< The second component. */
            T z = 0; /**< The third component. */

            /** @brief Returns the componentwise sum of this vector and @p o. */
            constexpr Vector3T operator+(const Vector3T& o) const noexcept
            {
                return Vector3T{x + o.x, y + o.y, z + o.z};
            }

            /** @brief Returns this vector with every component multiplied by @p s. */
            constexpr Vector3T operator*(T s) const noexcept
            {
                return Vector3T{x * s, y * s, z * s};
            }

            /** @brief Returns the componentwise difference, this vector minus @p o. */
            constexpr Vector3T operator-(const Vector3T& o) const noexcept
            {
                return Vector3T{x - o.x, y - o.y, z - o.z};
            }
        };

        static_assert(std::is_trivially_copyable<Vector3T<float>>::value &&
                          std::is_trivially_copyable<Vector3T<double>>::value,
                      "Vector3T is captured by SYCL kernels and must be trivially copyable.");
        static_assert(std::is_standard_layout<Vector3T<float>>::value &&
                          std::is_standard_layout<Vector3T<double>>::value,
                      "Vector3T is stored in buffers and must be standard layout.");
        static_assert(sizeof(Vector3T<float>) == 3 * sizeof(float) &&
                          sizeof(Vector3T<double>) == 3 * sizeof(double),
                      "Vector3T holds three elements and no padding.");

        /** @brief Returns the dot product of two vectors. */
        template<typename T>
        inline T dot(const Vector3T<T>& a, const Vector3T<T>& b) noexcept
        {
            return a.x * b.x + a.y * b.y + a.z * b.z;
        }

        /** @brief Returns the right-handed cross product a x b. */
        template<typename T>
        inline Vector3T<T> cross(const Vector3T<T>& a, const Vector3T<T>& b) noexcept
        {
            return Vector3T<T>{a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z,
                               a.x * b.y - a.y * b.x};
        }

        /** @brief Returns the Euclidean length of a vector. */
        template<typename T>
        inline T length(const Vector3T<T>& v) noexcept
        {
            return Math::sqrt(dot(v, v));
        }

        /**
         * @brief Returns the unit vector in the direction of @p v.
         * @return @p v unchanged when its length is not above zero.
         */
        template<typename T>
        inline Vector3T<T> normalize(const Vector3T<T>& v) noexcept
        {
            const T vector_length = length(v);
            return vector_length > T(0) ? v * (T(1) / vector_length) : v;
        }
    } // namespace Geometry
} // namespace SushiBLAS
