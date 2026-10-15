/**************************************************************************/
/* test_symmetric_access.cpp                                              */
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
 * @file test_symmetric_access.cpp
 * @brief Tests SymmetricAccess reads of both triangles, both layouts and both conjugation policies.
 * @author Mustafa Garip
 */

#include <limits>
#include <vector>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <algorithm>
#include <type_traits>
#include <gtest/gtest.h>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/core/traits.hpp>
#include <SushiBLAS/core/blas_types.hpp>
#include <SushiBLAS/blas_seam_kernels/conjugation.hpp>
#include <SushiBLAS/blas_seam_kernels/operand_access.hpp>
#include <SushiBLAS/arithmetic/complex_arithmetic.hpp>
#include <SushiBLAS/blas_seam_kernels/symmetric_access.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            /// @brief Order of the matrix every run-time case reads.
            constexpr int64_t K_ORDER = 4;

            /// @brief Leading dimension of the array, one more than the order so padding exists.
            constexpr int64_t K_LEADING_DIMENSION = 5;

            /// @brief Imaginary part given to a stored diagonal that a Hermitian read must drop.
            constexpr float K_DIAGONAL_IMAGINARY = 7.0f;

            /// @brief Order of the matrix the compile-time case reads.
            constexpr int64_t K_CONSTANT_ORDER = 3;

            /// @brief A row-major 3 x 3 array whose upper triangle is the stored one.
            constexpr float K_CONSTANT_MATRIX[] = {1.0f, 2.0f, 3.0f,
                                                   -1.0f, 4.0f, 5.0f,
                                                   -1.0f, -1.0f, 6.0f};

            /// @brief The two layouts every run-time case reads in.
            constexpr Core::Layout K_LAYOUTS[] = {Core::Layout::RowMajor,
                                                  Core::Layout::ColumnMajor};

            /// @brief The two triangles every run-time case stores.
            constexpr Core::Uplo K_TRIANGLES[] = {Core::Uplo::Upper, Core::Uplo::Lower};

            /** @brief Returns the value that marks an element no read may return. */
            template<typename T>
            [[nodiscard]] T poison() noexcept
            {
                using Real = Kernels::RealType<T>;
                if constexpr (Core::is_complex_v<T>)
                    return T(std::numeric_limits<Real>::quiet_NaN(),
                             std::numeric_limits<Real>::quiet_NaN());
                else
                    return std::numeric_limits<Real>::quiet_NaN();
            }

            /** @brief Returns whether (r, c) lies in the stored triangle. */
            [[nodiscard]] constexpr bool is_stored(Core::Uplo uplo, int64_t r, int64_t c) noexcept
            {
                return uplo == Core::Uplo::Upper ? r <= c : r >= c;
            }

            /**
             * @brief Returns a poisoned array of order K_ORDER whose stored triangle holds
             *        value(r, c).
             * @param value Callable `(int64_t r, int64_t c) -> T`.
             */
            template<typename T, typename Value>
            [[nodiscard]] std::vector<T> stored_triangle(Core::Layout layout, Core::Uplo uplo,
                                                         Value value)
            {
                const Kernels::OperandAccess at = Kernels::operand_access(
                    layout, Core::Transpose::NoTrans, K_LEADING_DIMENSION);
                std::vector<T> array(static_cast<std::size_t>(K_ORDER * K_LEADING_DIMENSION),
                                     poison<T>());
                for (int64_t r = 0; r < K_ORDER; ++r)
                {
                    for (int64_t c = 0; c < K_ORDER; ++c)
                    {
                        if (is_stored(uplo, r, c))
                            array[static_cast<std::size_t>(at.index(r, c))] = value(r, c);
                    }
                }
                return array;
            }

            /** @brief Returns the complex element the Hermitian cases store at (r, c). */
            [[nodiscard]] std::complex<float> hermitian_element(int64_t r, int64_t c) noexcept
            {
                return std::complex<float>(static_cast<float>(10 * r + c + 1),
                                           static_cast<float>(r - c) + 0.5f);
            }
        } // namespace

        TEST(Small_SymmetricAccess, ReadsTheStoredTriangleOnly)
        {
            for (const Core::Layout layout : K_LAYOUTS)
            {
                for (const Core::Uplo uplo : K_TRIANGLES)
                {
                    const std::vector<float> array = stored_triangle<float>(
                        layout, uplo,
                        [](int64_t r, int64_t c) { return static_cast<float>(10 * r + c + 1); });
                    const auto access = Kernels::symmetric_access<Kernels::Unconjugated>(
                        layout, uplo, K_LEADING_DIMENSION);

                    for (int64_t r = 0; r < K_ORDER; ++r)
                    {
                        for (int64_t c = 0; c < K_ORDER; ++c)
                        {
                            const int64_t low = std::min(r, c);
                            const int64_t high = std::max(r, c);
                            const int64_t expected = uplo == Core::Uplo::Upper
                                                         ? 10 * low + high + 1
                                                         : 10 * high + low + 1;
                            EXPECT_EQ(access.at(array.data(), r, c), static_cast<float>(expected))
                                << Core::to_string(uplo) << " (" << r << ", " << c << ")";
                        }
                    }
                }
            }
        }

        TEST(Small_SymmetricAccess, AHermitianMirrorIsConjugated)
        {
            for (const Core::Layout layout : K_LAYOUTS)
            {
                for (const Core::Uplo uplo : K_TRIANGLES)
                {
                    const std::vector<std::complex<float>> array =
                        stored_triangle<std::complex<float>>(layout, uplo, &hermitian_element);
                    const auto access = Kernels::symmetric_access<Kernels::Conjugated>(
                        layout, uplo, K_LEADING_DIMENSION);

                    for (int64_t r = 0; r < K_ORDER; ++r)
                    {
                        for (int64_t c = 0; c < K_ORDER; ++c)
                        {
                            if (r == c)
                                continue;

                            const std::complex<float> expected =
                                is_stored(uplo, r, c) ? hermitian_element(r, c)
                                                      : std::conj(hermitian_element(c, r));
                            EXPECT_EQ(access.at(array.data(), r, c), expected)
                                << Core::to_string(uplo) << " (" << r << ", " << c << ")";
                        }
                    }
                }
            }
        }

        TEST(Small_SymmetricAccess, AHermitianDiagonalIsReadAsItsRealPart)
        {
            const auto diagonal_element = [](int64_t r, int64_t c)
            {
                return std::complex<float>(static_cast<float>(10 * r + c + 1),
                                           K_DIAGONAL_IMAGINARY);
            };

            for (const Core::Layout layout : K_LAYOUTS)
            {
                for (const Core::Uplo uplo : K_TRIANGLES)
                {
                    const std::vector<std::complex<float>> array =
                        stored_triangle<std::complex<float>>(layout, uplo, diagonal_element);
                    const auto hermitian = Kernels::symmetric_access<Kernels::Conjugated>(
                        layout, uplo, K_LEADING_DIMENSION);
                    const auto symmetric = Kernels::symmetric_access<Kernels::Unconjugated>(
                        layout, uplo, K_LEADING_DIMENSION);

                    for (int64_t d = 0; d < K_ORDER; ++d)
                    {
                        const float real = static_cast<float>(11 * d + 1);
                        EXPECT_EQ(hermitian.at(array.data(), d, d),
                                  std::complex<float>(real, 0.0f))
                            << Core::to_string(uplo) << " diagonal " << d;
                        EXPECT_EQ(symmetric.at(array.data(), d, d),
                                  std::complex<float>(real, K_DIAGONAL_IMAGINARY))
                            << Core::to_string(uplo) << " diagonal " << d;
                    }
                }
            }
        }

        TEST(Small_SymmetricAccess, ARealElementIgnoresThePolicy)
        {
            static_assert(
                std::is_same_v<
                    Kernels::SymmetricAccess<Kernels::ConjugationFor<float, Kernels::Conjugated>>,
                    Kernels::SymmetricAccess<Kernels::Unconjugated>>,
                "A real element type reads a Hermitian matrix as a symmetric one.");
            static_assert(
                std::is_same_v<
                    Kernels::SymmetricAccess<
                        Kernels::ConjugationFor<std::complex<float>, Kernels::Conjugated>>,
                    Kernels::SymmetricAccess<Kernels::Conjugated>>,
                "A complex element type keeps the policy it was given.");
            SUCCEED();
        }

        TEST(Small_SymmetricAccess, IsAConstantExpression)
        {
            constexpr auto access = Kernels::symmetric_access<Kernels::Unconjugated>(
                Core::Layout::RowMajor, Core::Uplo::Upper, K_CONSTANT_ORDER);

            static_assert(access.at(K_CONSTANT_MATRIX, 0, 2) == 3.0f,
                          "A stored element is read where it lies.");
            static_assert(access.at(K_CONSTANT_MATRIX, 2, 0) == 3.0f,
                          "A mirrored element is read from the stored triangle.");
            static_assert(access.at(K_CONSTANT_MATRIX, 2, 1) == 5.0f,
                          "A mirrored element is read from the stored triangle.");
            static_assert(access.at(K_CONSTANT_MATRIX, 1, 1) == 4.0f,
                          "A diagonal element is read where it lies.");
            SUCCEED();
        }
    } // namespace Test
} // namespace SushiBLAS
