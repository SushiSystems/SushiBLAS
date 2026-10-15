/**************************************************************************/
/* orthonormalize.hpp                                                     */
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
 * @file orthonormalize.hpp
 * @brief Defines the reference kernel of the orthonormalisation sweep.
 * @author Mustafa Garip
 */

#pragma once

#include <cmath>
#include <cstdint>
#include <SushiBLAS/orthonormal/orthonormal_sweep.hpp>
#include <SushiBLAS/reference/reference_accumulator.hpp>

namespace SushiBLAS
{
    namespace Kernels
    {
        namespace Reference
        {
            namespace Random
            {
                /** @brief Holds the helpers of the reference orthonormalisation. */
                namespace Detail
                {
                    /** @brief Returns the dot product of two strided vectors of set, narrowed to T. */
                    template<typename T>
                    [[nodiscard]] inline T strided_dot(const T* a, const T* b, const VectorSet& set)
                    {
                        using RW = ReferenceAccumulator<T>;

                        ReferenceSum<T> sum;
                        for (int64_t j = 0; j < set.length; ++j)
                            sum.add(RW(a[j * set.elem_stride]) * RW(b[j * set.elem_stride]));
                        return static_cast<T>(sum.value());
                    }
                } // namespace Detail

                /**
                 * @brief Orthonormalises a strided set of vectors in place by modified Gram-Schmidt and scales them by gain.
                 * @tparam T The real element type, float or double.
                 * @see kernels/README.md
                 */
                template<typename T>
                inline void orthonormalize(T* p, const VectorSet& set, T gain)
                {
                    for (int sweep = 0; sweep < K_ORTHONORMAL_SWEEPS; ++sweep)
                    {
                        for (int64_t i = 0; i + 1 < set.count; ++i)
                        {
                            const T* vi = p + i * set.vec_stride;
                            const T norm_sq = Detail::strided_dot(vi, vi, set);

                            // Vector i is projected out of every later vector, in ascending order.
                            for (int64_t m = i + 1; m < set.count; ++m)
                            {
                                T* vm = p + m * set.vec_stride;
                                const T dot = Detail::strided_dot(vi, vm, set);
                                const T coef = norm_sq > tiny_norm_sq<T>() ? dot / norm_sq
                                                                           : static_cast<T>(0);

                                for (int64_t j = 0; j < set.length; ++j)
                                    vm[j * set.elem_stride] -= coef * vi[j * set.elem_stride];
                            }
                        }
                    }

                    for (int64_t v = 0; v < set.count; ++v)
                    {
                        T* vector = p + v * set.vec_stride;
                        const T norm_sq = Detail::strided_dot(vector, vector, set);
                        const T scale = norm_sq > tiny_norm_sq<T>() ? gain / std::sqrt(norm_sq)
                                                                    : gain;

                        for (int64_t j = 0; j < set.length; ++j)
                            vector[j * set.elem_stride] *= scale;
                    }
                }
            } // namespace Random
        } // namespace Reference
    } // namespace Kernels
} // namespace SushiBLAS
