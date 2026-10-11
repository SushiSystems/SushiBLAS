/**************************************************************************/
/* quaternion.hpp                                                         */
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
 * @file quaternion.hpp
 * @brief Declares the rotation quaternion, its product, conjugate, norm and vector rotation.
 * @author Mustafa Garip
 */

#pragma once

#include <cmath>
#include <type_traits>
#include <SushiBLAS/support/device_math.hpp>
#include <SushiBLAS/geometry/vector3.hpp>

namespace SushiBLAS
{
    namespace Geometry
    {
        /**
         * @brief Holds a rotation quaternion: the vector part (x, y, z) and the scalar w.
         * @tparam T A floating-point element type.
         * @see modules/foundation/geometry/README.md
         */
        template<typename T>
        struct QuaternionT
        {
            T x = 0; /**< The first component of the vector part. */
            T y = 0; /**< The second component of the vector part. */
            T z = 0; /**< The third component of the vector part. */
            T w = 1; /**< The scalar part. */
        };

        static_assert(std::is_trivially_copyable<QuaternionT<float>>::value &&
                          std::is_trivially_copyable<QuaternionT<double>>::value,
                      "QuaternionT is captured by SYCL kernels and must be trivially copyable.");
        static_assert(std::is_standard_layout<QuaternionT<float>>::value &&
                          std::is_standard_layout<QuaternionT<double>>::value,
                      "QuaternionT is stored in buffers and must be standard layout.");
        static_assert(sizeof(QuaternionT<float>) == 4 * sizeof(float) &&
                          sizeof(QuaternionT<double>) == 4 * sizeof(double),
                      "QuaternionT holds four elements and no padding.");

        /**
         * @brief Returns the quaternion of @p angle radians about @p axis.
         * @param axis The rotation axis; must be unit length.
         */
        template<typename T>
        inline QuaternionT<T> quaternion_axis_angle(const Vector3T<T>& axis, T angle) noexcept
        {
            const T half = angle * T(0.5);
            const T s = std::sin(half);
            return QuaternionT<T>{axis.x * s, axis.y * s, axis.z * s, std::cos(half)};
        }

        /** @brief Returns the Hamilton product a * b, the rotation b followed by a. */
        template<typename T>
        inline QuaternionT<T> mul(const QuaternionT<T>& a, const QuaternionT<T>& b) noexcept
        {
            return QuaternionT<T>{
                a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
                a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
                a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
                a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z};
        }

        /** @brief Returns the conjugate of @p q, its inverse when @p q is unit length. */
        template<typename T>
        inline QuaternionT<T> conjugate(const QuaternionT<T>& q) noexcept
        {
            return QuaternionT<T>{-q.x, -q.y, -q.z, q.w};
        }

        /**
         * @brief Returns @p v rotated by @p q.
         * @param q The rotation; must be unit length.
         * @see modules/foundation/geometry/README.md
         */
        template<typename T>
        inline Vector3T<T> rotate(const QuaternionT<T>& q, const Vector3T<T>& v) noexcept
        {
            const Vector3T<T> qv{q.x, q.y, q.z};
            const Vector3T<T> t = cross(qv, v) * T(2);
            return v + t * q.w + cross(qv, t);
        }

        /**
         * @brief Returns @p q scaled to unit length.
         * @return The identity when the length of @p q is not above zero.
         */
        template<typename T>
        inline QuaternionT<T> normalize(const QuaternionT<T>& q) noexcept
        {
            const T quaternion_length = Math::sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
            if (quaternion_length <= T(0))
                return QuaternionT<T>{};
            const T inv = T(1) / quaternion_length;
            return QuaternionT<T>{q.x * inv, q.y * inv, q.z * inv, q.w * inv};
        }

        /** @brief Returns the dot product of two quaternions as 4-vectors. */
        template<typename T>
        inline T dot(const QuaternionT<T>& a, const QuaternionT<T>& b) noexcept
        {
            return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
        }
    } // namespace Geometry
} // namespace SushiBLAS
