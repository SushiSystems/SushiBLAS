/****************************************************************/
/* orthonormalize.hpp                                           */
/* SushiBLAS - https://github.com/SushiSystems/SushiBLAS        */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

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
                 * @brief Orthonormalizes a strided set of vectors in place by modified
                 * Gram-Schmidt, then scales the result.
                 *
                 * Backs `RandomOps::orthogonal`. Orthogonal weight initialization is normally
                 * written as "fill with N(0,1), take the Q of a QR decomposition"; SushiBLAS
                 * has no QR (LinalgOps is quarantined and unimplemented), so this computes the
                 * same Q directly. MGS and Householder QR produce the same column space and
                 * differ only in how they get there - MGS is chosen because it is a dozen
                 * lines with no reflector storage, and because it re-orthogonalizes against
                 * the *updated* vectors rather than the originals, which keeps it accurate
                 * enough for initialization purposes on well-conditioned Gaussian input.
                 *
                 * Not a general-purpose QR: it returns Q only, discards R, and offers no
                 * pivoting or rank detection. It is not exported as a linear-algebra
                 * primitive for exactly that reason.
                 *
                 * @tparam T The real element type (float or double).
                 * @param p Pointer to the first element of the first vector.
                 * @param count Number of vectors to orthonormalize.
                 * @param length Number of elements in each vector.
                 * @param elem_stride Element-to-element step within one vector.
                 * @param vec_stride First-element-to-first-element step between vectors.
                 * @param gain Scalar applied to every element once the set is orthonormal.
                 */
                template<typename T>
                inline void orthonormalize_mgs(T* p, int64_t count, int64_t length,
                                                int64_t elem_stride, int64_t vec_stride, T gain)
                {
                    // Below this a vector is numerically indistinguishable from zero and
                    // dividing by its norm would produce infinities. Cannot happen for the
                    // Gaussian input this is used with (probability zero), but a NaN-filled
                    // weight matrix is a bad way to find out otherwise.
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
