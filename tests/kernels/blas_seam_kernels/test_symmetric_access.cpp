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
 * @file test_symmetric_access.cpp
 * @brief Tests SymmetricAccess reads of both triangles, both layouts and both conjugation policies.
 * @author Mustafa Garip
 */

#include <array>
#include <limits>
#include <string>
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

            /** @brief Holds one layout and one stored triangle of a run-time case. */
            struct StoredTriangle
            {
                Core::Layout layout; /**< The layout of the array. */
                Core::Uplo uplo;     /**< The triangle the array stores. */
            };

            /** @brief Returns the four layout and triangle pairs every case reads. */
            [[nodiscard]] std::array<StoredTriangle, 4> stored_triangles()
            {
                return {{{Core::Layout::RowMajor, Core::Uplo::Upper},
                         {Core::Layout::RowMajor, Core::Uplo::Lower},
                         {Core::Layout::ColumnMajor, Core::Uplo::Upper},
                         {Core::Layout::ColumnMajor, Core::Uplo::Lower}}};
            }

            /** @brief Names a layout and triangle pair for the test name. */
            std::string stored_triangle_name(const ::testing::TestParamInfo<StoredTriangle>& info)
            {
                return std::string(info.param.layout == Core::Layout::RowMajor ? "RowMajor"
                                                                                : "ColumnMajor") +
                       (info.param.uplo == Core::Uplo::Upper ? "Upper" : "Lower");
            }

            /** @brief Expects that a symmetric read returns the stored triangle's element. */
            void expect_the_stored_triangle_is_read(const StoredTriangle& row)
            {
                const std::vector<float> array = stored_triangle<float>(
                    row.layout, row.uplo,
                    [](int64_t r, int64_t c) { return static_cast<float>(10 * r + c + 1); });
                const auto access = Kernels::symmetric_access<Kernels::Unconjugated>(
                    row.layout, row.uplo, K_LEADING_DIMENSION);

                for (int64_t r = 0; r < K_ORDER; ++r)
                {
                    for (int64_t c = 0; c < K_ORDER; ++c)
                    {
                        const int64_t low = std::min(r, c);
                        const int64_t high = std::max(r, c);
                        const int64_t expected = row.uplo == Core::Uplo::Upper
                                                     ? 10 * low + high + 1
                                                     : 10 * high + low + 1;
                        EXPECT_EQ(access.at(array.data(), r, c), static_cast<float>(expected))
                            << "(" << r << ", " << c << ")";
                    }
                }
            }

            /** @brief Expects that a Hermitian read conjugates the mirrored off-diagonal. */
            void expect_the_hermitian_mirror_is_conjugated(const StoredTriangle& row)
            {
                const std::vector<std::complex<float>> array =
                    stored_triangle<std::complex<float>>(row.layout, row.uplo,
                                                         &hermitian_element);
                const auto access = Kernels::symmetric_access<Kernels::Conjugated>(
                    row.layout, row.uplo, K_LEADING_DIMENSION);

                for (int64_t r = 0; r < K_ORDER; ++r)
                {
                    for (int64_t c = 0; c < K_ORDER; ++c)
                    {
                        if (r == c)
                            continue;

                        const std::complex<float> expected =
                            is_stored(row.uplo, r, c) ? hermitian_element(r, c)
                                                      : std::conj(hermitian_element(c, r));
                        EXPECT_EQ(access.at(array.data(), r, c), expected)
                            << "(" << r << ", " << c << ")";
                    }
                }
            }

            /** @brief Returns the complex element of the diagonal cases, with its imaginary part. */
            [[nodiscard]] std::complex<float> diagonal_element(int64_t r, int64_t c) noexcept
            {
                return std::complex<float>(static_cast<float>(10 * r + c + 1),
                                           K_DIAGONAL_IMAGINARY);
            }

            /** @brief Expects that a read of the diagonal gives its value under a policy. */
            template<typename Policy>
            void expect_the_diagonal_reads(const StoredTriangle& row, float imaginary)
            {
                const std::vector<std::complex<float>> array =
                    stored_triangle<std::complex<float>>(row.layout, row.uplo, &diagonal_element);
                const auto access =
                    Kernels::symmetric_access<Policy>(row.layout, row.uplo, K_LEADING_DIMENSION);

                for (int64_t d = 0; d < K_ORDER; ++d)
                {
                    const float real = static_cast<float>(11 * d + 1);
                    EXPECT_EQ(access.at(array.data(), d, d), std::complex<float>(real, imaginary))
                        << "diagonal " << d;
                }
            }
        } // namespace

        /** @brief Reads a stored triangle once per layout and triangle. */
        class Small_SymmetricAccess : public ::testing::TestWithParam<StoredTriangle>
        {

        };

        TEST_P(Small_SymmetricAccess, ReadsTheStoredTriangleOnly)
        {
            expect_the_stored_triangle_is_read(GetParam());
        }

        TEST_P(Small_SymmetricAccess, AHermitianMirrorIsConjugated)
        {
            expect_the_hermitian_mirror_is_conjugated(GetParam());
        }

        TEST_P(Small_SymmetricAccess, AHermitianDiagonalIsReadAsItsRealPart)
        {
            expect_the_diagonal_reads<Kernels::Conjugated>(GetParam(), 0.0f);
        }

        TEST_P(Small_SymmetricAccess, ASymmetricDiagonalKeepsItsImaginaryPart)
        {
            expect_the_diagonal_reads<Kernels::Unconjugated>(GetParam(), K_DIAGONAL_IMAGINARY);
        }

        INSTANTIATE_TEST_SUITE_P(, Small_SymmetricAccess, ::testing::ValuesIn(stored_triangles()),
                                 stored_triangle_name);

        TEST(Small_SymmetricAccessPolicy, ARealElementIgnoresThePolicy)
        {
            const bool real_reads_as_symmetric = std::is_same_v<
                Kernels::SymmetricAccess<Kernels::ConjugationFor<float, Kernels::Conjugated>>,
                Kernels::SymmetricAccess<Kernels::Unconjugated>>;
            const bool complex_keeps_the_policy = std::is_same_v<
                Kernels::SymmetricAccess<
                    Kernels::ConjugationFor<std::complex<float>, Kernels::Conjugated>>,
                Kernels::SymmetricAccess<Kernels::Conjugated>>;

            EXPECT_TRUE(real_reads_as_symmetric);
            EXPECT_TRUE(complex_keeps_the_policy);
        }
    } // namespace Test
} // namespace SushiBLAS
