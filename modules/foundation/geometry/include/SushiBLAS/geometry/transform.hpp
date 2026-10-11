/**************************************************************************/
/* transform.hpp                                                          */
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
 * @file transform.hpp
 * @brief Declares the functions that build a 4x4 transform matrix or take one apart.
 * @author Mustafa Garip
 */

#pragma once

#include <cmath>
#include <SushiBLAS/geometry/vector3.hpp>
#include <SushiBLAS/geometry/quaternion.hpp>
#include <SushiBLAS/geometry/matrix4.hpp>

namespace SushiBLAS
{
    namespace Geometry
    {
        /** @brief Returns the matrix that translates by @p t. */
        template<typename T>
        inline Matrix4T<T> translation(const Vector3T<T>& t) noexcept
        {
            Matrix4T<T> r{};
            r.m[12] = t.x;
            r.m[13] = t.y;
            r.m[14] = t.z;
            return r;
        }

        /** @brief Returns the matrix that scales each axis by the matching component of @p s. */
        template<typename T>
        inline Matrix4T<T> scaling(const Vector3T<T>& s) noexcept
        {
            Matrix4T<T> r{};
            r.m[0] = s.x;
            r.m[5] = s.y;
            r.m[10] = s.z;
            return r;
        }

        /**
         * @brief Returns a right-handed reverse-Z perspective matrix with an infinite far plane.
         * @param near_plane The near clip distance; must be above zero.
         * @param far_plane Ignored; the far plane is at infinity.
         * @pre The pipeline clears depth to 0 and compares with GREATER_OR_EQUAL.
         * @see modules/foundation/geometry/README.md
         */
        template<typename T>
        inline Matrix4T<T> perspective(T fovy_radians, T aspect, T near_plane, T far_plane) noexcept
        {
            (void)far_plane;
            const T f = T(1) / std::tan(fovy_radians * T(0.5));
            Matrix4T<T> r{};
            for (T& value : r.m)
                value = 0;
            r.m[0] = f / aspect;
            r.m[5] = -f;
            r.m[10] = 0;
            r.m[11] = -1;
            r.m[14] = near_plane;
            return r;
        }

        /**
         * @brief Returns the right-handed view matrix that looks from @p eye toward @p center.
         * @param up The world up direction; must not be parallel to the view direction.
         */
        template<typename T>
        inline Matrix4T<T> look_at(const Vector3T<T>& eye, const Vector3T<T>& center,
                                   const Vector3T<T>& up) noexcept
        {
            const Vector3T<T> f = normalize(center - eye);
            const Vector3T<T> s = normalize(cross(f, up));
            const Vector3T<T> u = cross(s, f);
            Matrix4T<T> r{};
            r.m[0] = s.x;  r.m[4] = s.y;  r.m[8] = s.z;
            r.m[1] = u.x;  r.m[5] = u.y;  r.m[9] = u.z;
            r.m[2] = -f.x; r.m[6] = -f.y; r.m[10] = -f.z;
            r.m[12] = -dot(s, eye);
            r.m[13] = -dot(u, eye);
            r.m[14] = dot(f, eye);
            return r;
        }

        /**
         * @brief Returns the rotation matrix of @p q.
         * @param q The rotation; must be unit length.
         */
        template<typename T>
        inline Matrix4T<T> matrix4_from_quaternion(const QuaternionT<T>& q) noexcept
        {
            const T xx = q.x * q.x, yy = q.y * q.y, zz = q.z * q.z;
            const T xy = q.x * q.y, xz = q.x * q.z, yz = q.y * q.z;
            const T wx = q.w * q.x, wy = q.w * q.y, wz = q.w * q.z;
            Matrix4T<T> r{};
            r.m[0] = 1 - 2 * (yy + zz); r.m[1] = 2 * (xy + wz);     r.m[2] = 2 * (xz - wy);
            r.m[4] = 2 * (xy - wz);     r.m[5] = 1 - 2 * (xx + zz); r.m[6] = 2 * (yz + wx);
            r.m[8] = 2 * (xz + wy);     r.m[9] = 2 * (yz - wx);     r.m[10] = 1 - 2 * (xx + yy);
            return r;
        }

        /**
         * @brief Returns the model matrix translation * rotation * scale.
         * @param rotation The orientation; must be unit length.
         */
        template<typename T>
        inline Matrix4T<T> compose_transform(const Vector3T<T>& position,
                                             const QuaternionT<T>& rotation,
                                             const Vector3T<T>& scale) noexcept
        {
            return mul(translation(position),
                       mul(matrix4_from_quaternion(rotation), scaling(scale)));
        }

        /**
         * @brief Returns the unit quaternion of the rotation in the upper-left 3x3 of @p matrix.
         * @param matrix An affine matrix whose 3x3 part is orthonormal.
         * @see modules/foundation/geometry/README.md
         */
        template<typename T>
        inline QuaternionT<T> quaternion_from_matrix(const Matrix4T<T>& matrix) noexcept
        {
            // mRC names element (row R, column C), stored at m[C * 4 + R].
            const T m00 = matrix.m[0], m10 = matrix.m[1], m20 = matrix.m[2];
            const T m01 = matrix.m[4], m11 = matrix.m[5], m21 = matrix.m[6];
            const T m02 = matrix.m[8], m12 = matrix.m[9], m22 = matrix.m[10];
            const T trace = m00 + m11 + m22;
            QuaternionT<T> q{};
            if (trace > T(0))
            {
                T s = std::sqrt(trace + T(1)) * T(2); // s = 4 * w
                q.w = T(0.25) * s;
                q.x = (m21 - m12) / s;
                q.y = (m02 - m20) / s;
                q.z = (m10 - m01) / s;
            }
            else if (m00 > m11 && m00 > m22)
            {
                T s = std::sqrt(T(1) + m00 - m11 - m22) * T(2); // s = 4 * x
                q.w = (m21 - m12) / s;
                q.x = T(0.25) * s;
                q.y = (m01 + m10) / s;
                q.z = (m02 + m20) / s;
            }
            else if (m11 > m22)
            {
                T s = std::sqrt(T(1) + m11 - m00 - m22) * T(2); // s = 4 * y
                q.w = (m02 - m20) / s;
                q.x = (m01 + m10) / s;
                q.y = T(0.25) * s;
                q.z = (m12 + m21) / s;
            }
            else
            {
                T s = std::sqrt(T(1) + m22 - m00 - m11) * T(2); // s = 4 * z
                q.w = (m10 - m01) / s;
                q.x = (m02 + m20) / s;
                q.y = (m12 + m21) / s;
                q.z = T(0.25) * s;
            }
            return normalize(q);
        }

        /**
         * @brief Returns the inverse of an affine matrix.
         * @param matrix A matrix whose last row is (0, 0, 0, 1).
         * @return The identity when the 3x3 part of @p matrix is singular.
         * @see modules/foundation/geometry/README.md
         */
        template<typename T>
        inline Matrix4T<T> affine_inverse(const Matrix4T<T>& matrix) noexcept
        {
            const T a = matrix.m[0], b = matrix.m[4], c = matrix.m[8];
            const T d = matrix.m[1], e = matrix.m[5], f = matrix.m[9];
            const T g = matrix.m[2], h = matrix.m[6], i = matrix.m[10];
            const T det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
            if (det == T(0))
                return Matrix4T<T>{};
            const T inv_det = T(1) / det;
            Matrix4T<T> r{};
            // r.m[col * 4 + row] holds element (row, col) of the inverse 3x3.
            r.m[0] = (e * i - f * h) * inv_det;
            r.m[1] = (f * g - d * i) * inv_det;
            r.m[2] = (d * h - e * g) * inv_det;
            r.m[4] = (c * h - b * i) * inv_det;
            r.m[5] = (a * i - c * g) * inv_det;
            r.m[6] = (b * g - a * h) * inv_det;
            r.m[8] = (b * f - c * e) * inv_det;
            r.m[9] = (c * d - a * f) * inv_det;
            r.m[10] = (a * e - b * d) * inv_det;
            // The inverse translation is the original translation through the negated inverse 3x3.
            const T tx = matrix.m[12], ty = matrix.m[13], tz = matrix.m[14];
            r.m[12] = -(r.m[0] * tx + r.m[4] * ty + r.m[8] * tz);
            r.m[13] = -(r.m[1] * tx + r.m[5] * ty + r.m[9] * tz);
            r.m[14] = -(r.m[2] * tx + r.m[6] * ty + r.m[10] * tz);
            return r;
        }

        /**
         * @brief Splits a translation-rotation-scale matrix into its three parts.
         * @param matrix An affine matrix without shear.
         * @param out_rotation Receives a unit quaternion.
         * @param out_scale Receives a negative x when @p matrix holds a reflection.
         * @see modules/foundation/geometry/README.md
         */
        template<typename T>
        inline void decompose_transform(const Matrix4T<T>& matrix, Vector3T<T>& out_position,
                                        QuaternionT<T>& out_rotation,
                                        Vector3T<T>& out_scale) noexcept
        {
            out_position = Vector3T<T>{matrix.m[12], matrix.m[13], matrix.m[14]};
            Vector3T<T> col0{matrix.m[0], matrix.m[1], matrix.m[2]};
            Vector3T<T> col1{matrix.m[4], matrix.m[5], matrix.m[6]};
            Vector3T<T> col2{matrix.m[8], matrix.m[9], matrix.m[10]};
            T sx = length(col0), sy = length(col1), sz = length(col2);
            // A negative determinant is a reflection; the x scale carries it.
            const T det = dot(col0, cross(col1, col2));
            if (det < T(0))
                sx = -sx;
            out_scale = Vector3T<T>{sx, sy, sz};
            Matrix4T<T> rotation{};
            if (sx != T(0))
            {
                rotation.m[0] = col0.x / sx;
                rotation.m[1] = col0.y / sx;
                rotation.m[2] = col0.z / sx;
            }
            if (sy != T(0))
            {
                rotation.m[4] = col1.x / sy;
                rotation.m[5] = col1.y / sy;
                rotation.m[6] = col1.z / sy;
            }
            if (sz != T(0))
            {
                rotation.m[8] = col2.x / sz;
                rotation.m[9] = col2.y / sz;
                rotation.m[10] = col2.z / sz;
            }
            out_rotation = quaternion_from_matrix(rotation);
        }
    } // namespace Geometry
} // namespace SushiBLAS
