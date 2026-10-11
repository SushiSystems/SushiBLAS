/**************************************************************************/
/* interpolation.hpp                                                      */
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
 * @file interpolation.hpp
 * @brief Declares the linear blend of scalars and vectors and the two blends of rotations.
 * @author Mustafa Garip
 */

#pragma once

#include <cmath>
#include <SushiBLAS/geometry/vector3.hpp>
#include <SushiBLAS/geometry/quaternion.hpp>

namespace SushiBLAS
{
    namespace Geometry
    {
        /**
         * @brief Returns the componentwise linear blend `a + (b - a) * t` of two vectors.
         * @param t The blend parameter; not clamped to [0, 1].
         */
        template<typename T>
        inline Vector3T<T> lerp(const Vector3T<T>& a, const Vector3T<T>& b, T t) noexcept
        {
            return a + (b - a) * t;
        }

        /**
         * @brief Returns the linear blend `a + (b - a) * t` of two scalars.
         * @param t The blend parameter; not clamped to [0, 1].
         */
        template<typename T>
        inline T lerp(T a, T b, T t) noexcept
        {
            return a + (b - a) * t;
        }

        /**
         * @brief Returns the normalised linear blend of two rotations along the short arc.
         * @param t The blend parameter, in [0, 1].
         * @see modules/foundation/geometry/README.md
         */
        template<typename T>
        inline QuaternionT<T> nlerp(const QuaternionT<T>& a, QuaternionT<T> b, T t) noexcept
        {
            if (dot(a, b) < T(0))
                b = QuaternionT<T>{-b.x, -b.y, -b.z, -b.w};
            return normalize(QuaternionT<T>{a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t,
                                            a.z + (b.z - a.z) * t, a.w + (b.w - a.w) * t});
        }

        /**
         * @brief Returns the blend of two rotations at constant angular rate along the short arc.
         * @param t The blend parameter, in [0, 1].
         * @see modules/foundation/geometry/README.md
         */
        template<typename T>
        inline QuaternionT<T> slerp(const QuaternionT<T>& a, QuaternionT<T> b, T t) noexcept
        {
            T cos_omega = dot(a, b);
            if (cos_omega < T(0))
            {
                b = QuaternionT<T>{-b.x, -b.y, -b.z, -b.w};
                cos_omega = -cos_omega;
            }
            if (cos_omega > T(1) - T(1e-6))
                return nlerp(a, b, t);
            const T omega = std::acos(cos_omega);
            const T sin_omega = std::sin(omega);
            const T s0 = std::sin((T(1) - t) * omega) / sin_omega;
            const T s1 = std::sin(t * omega) / sin_omega;
            return QuaternionT<T>{a.x * s0 + b.x * s1, a.y * s0 + b.y * s1,
                                  a.z * s0 + b.z * s1, a.w * s0 + b.w * s1};
        }
    } // namespace Geometry
} // namespace SushiBLAS
