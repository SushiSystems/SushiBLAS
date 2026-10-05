/****************************************************************/
/* orthonormalize.hpp                                           */
/* SushiBLAS - https://github.com/SushiSystems/SushiBLAS        */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

/**
 * @file orthonormalize.hpp
 * @brief Defines the modified Gram-Schmidt kernel behind RandomOps::orthogonal.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <sycl/sycl.hpp>

namespace SushiBLAS
{
    namespace Kernels
    {
        namespace SYCL
        {
            namespace Random
            {
                /**
                 * @brief Orthonormalises a strided set of vectors in place and scales them.
                 * @tparam T The real element type, float or double.
                 * @param elem_stride Element-to-element step within one vector.
                 * @param vec_stride First-element-to-first-element step between vectors.
                 * @see kernels/README.md
                 */
                template<typename T>
                inline void orthonormalize_mgs(T* p, int64_t count, int64_t length,
                                                int64_t elem_stride, int64_t vec_stride, T gain)
                {
                    // A norm at or below tiny leaves its vector unscaled; see kernels/README.md.
                    const T tiny = static_cast<T>(1e-12);

                    for (int64_t i = 0; i < count; ++i)
                    {
                        T* vi = p + i * vec_stride;

                        T norm_sq = static_cast<T>(0);
                        for (int64_t j = 0; j < length; ++j)
                        {
                            T v = vi[j * elem_stride];
                            norm_sq += v * v;
                        }

                        T norm = sycl::sqrt(norm_sq);
                        if (norm > tiny)
                        {
                            T inv = static_cast<T>(1) / norm;
                            for (int64_t j = 0; j < length; ++j)
                            {
                                vi[j * elem_stride] *= inv;
                            }
                        }

                        // Project the now-unit v_i out of every vector still to be processed.
                        for (int64_t m = i + 1; m < count; ++m)
                        {
                            T* vm = p + m * vec_stride;

                            T dot = static_cast<T>(0);
                            for (int64_t j = 0; j < length; ++j)
                            {
                                dot += vi[j * elem_stride] * vm[j * elem_stride];
                            }

                            for (int64_t j = 0; j < length; ++j)
                            {
                                vm[j * elem_stride] -= dot * vi[j * elem_stride];
                            }
                        }
                    }

                    if (gain != static_cast<T>(1))
                    {
                        for (int64_t i = 0; i < count; ++i)
                        {
                            T* vi = p + i * vec_stride;
                            for (int64_t j = 0; j < length; ++j)
                            {
                                vi[j * elem_stride] *= gain;
                            }
                        }
                    }
                }

            } // namespace Random
        } // namespace SYCL
    } // namespace Kernels
} // namespace SushiBLAS
