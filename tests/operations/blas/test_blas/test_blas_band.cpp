/**************************************************************************/
/* test_blas_band.cpp                                                     */
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
 * @file test_blas_band.cpp
 * @brief Tests the band routines against hand-computed results, their dense siblings and
 *        each other's kernels, one row per layout, kernel and element type.
 * @author Mustafa Garip
 */

#include <array>
#include <cmath>
#include <random>
#include <string>
#include <vector>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <type_traits>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <SushiBLAS/SushiBLAS.hpp>
#include <SushiBLAS/core/traits.hpp>
#include <SushiBLAS/core/blas_types.hpp>
#include <SushiBLAS/arithmetic/complex_arithmetic.hpp>
#include <SushiBLAS/engine_fixture/sushiblas_test.hpp>
#include <SushiBLAS/tensor_expectations/tensor_comparison.hpp>
#include <SushiBLAS/tensor_expectations/tensor_readback.hpp>
#include <SushiBLAS/tensor_expectations/scalar_dtype.hpp>
#include <SushiBLAS/test_runner/test_environment.hpp>
#include <SushiBLAS/blas_test_oracle/blas_structured.hpp>
#include "support/tolerance.hpp"
#include "blas_rows.hpp"

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            /// @brief The type hand-written and integer test data is stated in.
            using Wide = std::complex<double>;

            /// @brief Seed of every random draw in this file.
            constexpr std::uint32_t K_SEED = 20261007u;

            /// @brief Fills both parts of every slot a call must neither read nor write.
            constexpr double K_POISON = 999.0;

            /// @brief Imaginary part given to a stored Hermitian diagonal, which is not read.
            constexpr double K_DIAGONAL_IMAGINARY = 9.0;

            /// @brief Scales the product in the cases that compare two results.
            constexpr double K_ALPHA = 1.5;

            /// @brief Scales the prior y in the cases that compare two results.
            constexpr double K_BETA = -0.5;

            /// @brief Rows of the worked matrix of the design, section 2.1.
            constexpr int64_t K_WORKED_ROWS = 4;

            /// @brief Columns of the worked matrix.
            constexpr int64_t K_WORKED_COLS = 5;

            /// @brief Sub-diagonals of the worked matrix.
            constexpr int64_t K_WORKED_KL = 1;

            /// @brief Super-diagonals of the worked matrix.
            constexpr int64_t K_WORKED_KU = 2;

            /// @brief Elements per stored line of the worked matrix: kl + ku + 1.
            constexpr int64_t K_WORKED_LD = 4;

            /// @brief Order of the 3 x 3 triangular and Hermitian hand cases.
            constexpr int64_t K_HAND_ORDER = 3;

            /// @brief Order of the tridiagonal symmetric hand case.
            constexpr int64_t K_TRIDIAGONAL_ORDER = 4;

            /// @brief Rows of the general matrix compared against gemv.
            constexpr int64_t K_SIBLING_ROWS = 9;

            /// @brief Columns of the general matrix compared against gemv.
            constexpr int64_t K_SIBLING_COLS = 13;

            /// @brief Order of the square matrices compared against their dense siblings.
            constexpr int64_t K_SIBLING_ORDER = 17;

            /// @brief Order at which a device that plans a work-group per row runs one.
            constexpr int64_t K_LARGE_ORDER = 2048;

            /// @brief Order of the matrices the two kernels are compared on.
            constexpr int64_t K_MODE_ORDER = 64;

            /**
             * @brief Magnitude of one product of two entries drawn from [-1, 1] in both parts:
             *        the bound of |a| * |b| for complex entries of modulus at most 2^(1/2).
             */
            constexpr double K_ENTRY_PRODUCT_MAGNITUDE = 2.0;

            /**
             * @brief Ratio of error to rounding bound above which two kernels disagree: the
             *        reference-tester threshold 16 times 4 for the growth of a tbsv solve whose
             *        diagonal k + 2 dominates k off-diagonals of at most 1.
             */
            constexpr double K_AGREEMENT_THRESHOLD = 64.0;

            /// @brief Rounded operations on the path to a HALF result: the sum is carried in FLOAT32.
            constexpr std::size_t K_HALF_ROUNDINGS = 2;

            /// @brief The worked matrix, row by row.
            const std::vector<double> K_WORKED_DENSE = {11, 12, 13, 0,  0,
                                                        21, 22, 23, 24, 0,
                                                        0,  32, 33, 34, 35,
                                                        0,  0,  43, 44, 45};

            /// @brief The worked matrix as COLUMN_MAJOR band storage, shape (4, 5).
            const std::vector<double> K_WORKED_COLUMN_MAJOR = {K_POISON, K_POISON, 11, 21,
                                                               K_POISON, 12, 22, 32,
                                                               13, 23, 33, 43,
                                                               24, 34, 44, K_POISON,
                                                               35, 45, K_POISON, K_POISON};

            /// @brief The worked matrix as ROW_MAJOR band storage, shape (4, 4).
            const std::vector<double> K_WORKED_ROW_MAJOR = {K_POISON, 11, 12, 13,
                                                            21, 22, 23, 24,
                                                            32, 33, 34, 35,
                                                            43, 44, 45, K_POISON};

            /**
             * @brief The tridiagonal [1 2 0 0; 2 3 4 0; 0 4 5 6; 0 0 6 7] with its diagonal
             *        last in each line of two: UPPER in COLUMN_MAJOR, LOWER in ROW_MAJOR.
             */
            const std::vector<double> K_TRIDIAGONAL_DIAGONAL_LAST = {K_POISON, 1, 2, 3,
                                                                     4,        5, 6, 7};

            /**
             * @brief The same tridiagonal with its diagonal first in each line: UPPER in
             *        ROW_MAJOR, LOWER in COLUMN_MAJOR.
             */
            const std::vector<double> K_TRIDIAGONAL_DIAGONAL_FIRST = {1, 2, 3, 4,
                                                                      5, 6, 7, K_POISON};

            /// @brief U = [1 2 0; 0 3 4; 0 0 5], row by row.
            const std::vector<double> K_HAND_UPPER = {1, 2, 0, 0, 3, 4, 0, 0, 5};

            /// @brief L = [1 0 0; 2 3 0; 0 4 5], the transpose of K_HAND_UPPER.
            const std::vector<double> K_HAND_LOWER = {1, 0, 0, 2, 3, 0, 0, 4, 5};

            /// @brief Band widths the two kernels are compared at.
            const std::vector<int64_t> K_MODE_WIDTHS = {0, 1, 5, 63};

            /// @brief Both triangles a band can hold.
            const std::vector<Core::Uplo> K_TRIANGLES = {Core::Uplo::Upper, Core::Uplo::Lower};

            /// @brief Every form op(A) takes.
            const std::vector<Core::Transpose> K_TRANSPOSES = {Core::Transpose::NoTrans,
                                                               Core::Transpose::Trans,
                                                               Core::Transpose::ConjTrans};

            /// @brief Both ways a triangular diagonal is read.
            constexpr Core::Diag K_DIAGONALS[] = {Core::Diag::NonUnit, Core::Diag::Unit};

            /// @brief Both kernels an operation has.
            const std::vector<ExecutionMode> K_BOTH_MODES = {ExecutionMode::ForceReference,
                                                             ExecutionMode::ForceSYCL};

            /// @brief The kernel the cases that compare the device with itself run.
            const std::vector<ExecutionMode> K_DEVICE_MODE = {ExecutionMode::ForceSYCL};

            /// @brief The five element types the band routines accept.
            const std::vector<Core::DataType> K_ALL_TYPES = {
                Core::DataType::Half, Core::DataType::Float32, Core::DataType::Float64,
                Core::DataType::Complex32, Core::DataType::Complex64};

            /// @brief The complex element types, the only ones a conjugation changes.
            const std::vector<Core::DataType> K_COMPLEX_TYPES = {Core::DataType::Complex32,
                                                                 Core::DataType::Complex64};

            /// @brief The element type of the HALF accumulation cases.
            const std::vector<Core::DataType> K_HALF_TYPE = {Core::DataType::Half};

            /// @brief The element type of the cases of order K_LARGE_ORDER.
            const std::vector<Core::DataType> K_FLOAT_TYPE = {Core::DataType::Float32};

            /** @brief Holds the flags of one triangular product of [1, 2, 3] and its result. */
            struct TriangularRow
            {
                Core::Uplo uplo;              /**< The triangle: K_HAND_UPPER or K_HAND_LOWER. */
                Core::Transpose trans;        /**< The form of op(A). */
                Core::Diag diag;              /**< How the diagonal is read. */
                double product[K_HAND_ORDER]; /**< op(A) * [1, 2, 3]. */
            };

            /// @brief The triangular products of the design, section 10.3.
            const std::vector<TriangularRow> K_TRIANGULAR_ROWS = {
                {Core::Uplo::Upper, Core::Transpose::NoTrans, Core::Diag::NonUnit, {5, 18, 15}},
                {Core::Uplo::Upper, Core::Transpose::Trans, Core::Diag::NonUnit, {1, 8, 23}},
                {Core::Uplo::Upper, Core::Transpose::NoTrans, Core::Diag::Unit, {5, 14, 3}},
                {Core::Uplo::Upper, Core::Transpose::Trans, Core::Diag::Unit, {1, 4, 11}},
                {Core::Uplo::Lower, Core::Transpose::NoTrans, Core::Diag::NonUnit, {1, 8, 23}},
                {Core::Uplo::Lower, Core::Transpose::Trans, Core::Diag::NonUnit, {5, 18, 15}}};

            /** @brief Holds the flags of one call on a triangle band. */
            struct TriangleFlags
            {
                Core::Uplo uplo;       /**< The triangle. */
                Core::Transpose trans; /**< The form of op(A). */
                Core::Diag diag;       /**< How the diagonal is read. */
            };

            /** @brief Holds the extents and the band widths of one general band matrix. */
            struct BandShape
            {
                int64_t m;  /**< Rows. */
                int64_t n;  /**< Columns. */
                int64_t kl; /**< Sub-diagonals. */
                int64_t ku; /**< Super-diagonals. */
            };

            /**
             * @brief Shapes at the edge of what gbmv accepts: one element, a band that leaves
             *        the matrix on one side, and widths larger than the matrix.
             */
            const std::vector<BandShape> K_EDGE_SHAPES = {{1, 1, 0, 0},
                                                          {5, 2, 3, 0},
                                                          {2, 5, 0, 3},
                                                          {3, 4, 7, 7}};

            /** @brief Returns every combination of uplo, trans and diag. */
            [[nodiscard]] std::vector<TriangleFlags> all_triangle_flags()
            {
                std::vector<TriangleFlags> flags;
                for (const Core::Uplo uplo : K_TRIANGLES)
                    for (const Core::Transpose trans : K_TRANSPOSES)
                        for (const Core::Diag diag : K_DIAGONALS)
                            flags.push_back({uplo, trans, diag});
                return flags;
            }

            /** @brief Returns the tail of a row named by the flags of a call. */
            [[nodiscard]] std::string axis_name(const TriangleFlags& flags)
            {
                return "_" + word(flags.uplo) + "_" + word(flags.trans) + "_" + word(flags.diag);
            }

            /** @brief Returns the tail of a row named by the flags of a triangular product. */
            [[nodiscard]] std::string axis_name(const TriangularRow& row)
            {
                return axis_name(TriangleFlags{row.uplo, row.trans, row.diag});
            }

            /** @brief Returns the tail of a row named by the shape of a band matrix. */
            [[nodiscard]] std::string axis_name(const BandShape& shape)
            {
                return "_" + std::to_string(shape.m) + "x" + std::to_string(shape.n) + "_kl" +
                       std::to_string(shape.kl) + "_ku" + std::to_string(shape.ku);
            }

            /**
             * @brief Returns the rounded operations on the path to one result of two kernels
             *        that sum 2 * K_MODE_ORDER terms: the sum, and the carried solve.
             */
            template<typename R>
            [[nodiscard]] constexpr std::size_t agreement_roundings()
            {
                if constexpr (std::is_same_v<R, sycl::half>)
                    return K_HALF_ROUNDINGS;
                else
                    return static_cast<std::size_t>(2 * K_MODE_ORDER);
            }

            /** @brief Returns the form of op(A) as the index into the three expectations of gbmv. */
            [[nodiscard]] std::size_t form_index(Core::Transpose trans)
            {
                if (trans == Core::Transpose::NoTrans)
                    return 0;
                return trans == Core::Transpose::Trans ? 1 : 2;
            }

            /** @brief Returns value in T; a real T drops the imaginary part. */
            template<typename T>
            [[nodiscard]] T narrowed(const Wide& value)
            {
                if constexpr (Core::is_complex_v<T>)
                {
                    using R = typename T::value_type;
                    return T(static_cast<R>(value.real()), static_cast<R>(value.imag()));
                }
                else if constexpr (std::is_same_v<T, sycl::half>)
                    return sycl::half(static_cast<float>(value.real()));
                else
                    return static_cast<T>(value.real());
            }

            /** @brief Returns value as a complex double; a real T has a zero imaginary part. */
            template<typename T>
            [[nodiscard]] Wide widened(const T& value)
            {
                if constexpr (Core::is_complex_v<T>)
                    return Detail::to_complex_double(value);
                else
                    return Wide(Detail::to_double(value), 0.0);
            }

            /** @brief Returns values in T. */
            template<typename T>
            [[nodiscard]] std::vector<T> narrowed_all(const std::vector<Wide>& values)
            {
                std::vector<T> out;
                out.reserve(values.size());
                for (const Wide& value : values)
                    out.push_back(narrowed<T>(value));
                return out;
            }

            /** @brief Returns real values in T. */
            template<typename T>
            [[nodiscard]] std::vector<T> elements(const std::vector<double>& values)
            {
                std::vector<T> out;
                out.reserve(values.size());
                for (const double value : values)
                    out.push_back(narrowed<T>(Wide(value, 0.0)));
                return out;
            }

            /** @brief Returns the value that marks a slot no kernel may read or write. */
            template<typename T>
            [[nodiscard]] T poison()
            {
                return narrowed<T>(Wide(K_POISON, K_POISON));
            }

            /** @brief Returns count elements with every part drawn from [-1, 1]. */
            template<typename T>
            [[nodiscard]] std::vector<T> random_elements(std::mt19937& generator, int64_t count)
            {
                std::uniform_real_distribution<double> distribution(-1.0, 1.0);
                std::vector<T> out(static_cast<std::size_t>(count));
                for (T& element : out)
                {
                    const double real = distribution(generator);
                    const double imaginary = distribution(generator);
                    element = narrowed<T>(Wide(real, imaginary));
                }
                return out;
            }

            /** @brief Returns an integer drawn from low to low + count - 1. */
            [[nodiscard]] double draw_integer(std::mt19937& generator, int low, unsigned count)
            {
                return static_cast<double>(low + static_cast<int>(generator() % count));
            }

            /**
             * @brief Holds a triangular system whose every product and solve is exact in HALF.
             */
            struct ExactTriangle
            {
                std::vector<Wide> stored; /**< The matrix a call receives, row by row. */
                std::vector<Wide> x;      /**< The vector of the product, the solution. */
                std::vector<Wide> b;      /**< op(T) * x, the right-hand side of the solve. */
            };

            /**
             * @brief Returns a triangle of small integers with a diagonal of 1, 2, -2 or 4, an
             *        integer x, and op(T) * x formed in integers.
             * @param complex_parts Whether the off-diagonal elements and x have imaginary parts.
             * @return stored holds K_POISON in the other triangle, and on the diagonal under UNIT.
             */
            [[nodiscard]] ExactTriangle exact_triangle(int64_t n, Core::Uplo uplo,
                                                       Core::Transpose trans, Core::Diag diag,
                                                       bool complex_parts)
            {
                constexpr double diagonals[] = {1.0, 2.0, -2.0, 4.0};
                const bool upper = (uplo == Core::Uplo::Upper);
                const bool unit = (diag == Core::Diag::Unit);
                const auto at = [n](int64_t r, int64_t c)
                {
                    return static_cast<std::size_t>(r * n + c);
                };

                std::mt19937 generator(K_SEED);
                ExactTriangle out;
                out.stored.assign(static_cast<std::size_t>(n * n), Wide(K_POISON, K_POISON));
                std::vector<Wide> model(static_cast<std::size_t>(n * n));

                for (int64_t r = 0; r < n; ++r)
                {
                    for (int64_t c = 0; c < n; ++c)
                    {
                        if (upper ? (r > c) : (r < c))
                            continue;

                        if (r == c)
                        {
                            const Wide value(diagonals[generator() % 4u], 0.0);
                            model[at(r, c)] = unit ? Wide(1.0, 0.0) : value;
                            if (!unit)
                                out.stored[at(r, c)] = value;
                            continue;
                        }

                        const double real = draw_integer(generator, -2, 5u);
                        const double imaginary = draw_integer(generator, -2, 5u);
                        model[at(r, c)] = Wide(real, complex_parts ? imaginary : 0.0);
                        out.stored[at(r, c)] = model[at(r, c)];
                    }
                }

                out.x.resize(static_cast<std::size_t>(n));
                for (Wide& element : out.x)
                {
                    const double real = draw_integer(generator, -3, 7u);
                    const double imaginary = draw_integer(generator, -3, 7u);
                    element = Wide(real, complex_parts ? imaginary : 0.0);
                }

                out.b.assign(static_cast<std::size_t>(n), Wide());
                for (int64_t r = 0; r < n; ++r)
                {
                    for (int64_t c = 0; c < n; ++c)
                    {
                        Wide element = Core::is_transposed(trans) ? model[at(c, r)]
                                                                  : model[at(r, c)];
                        if (Core::is_conjugated(trans))
                            element = std::conj(element);
                        out.b[static_cast<std::size_t>(r)] +=
                            element * out.x[static_cast<std::size_t>(c)];
                    }
                }
                return out;
            }

            /** @brief Expects both parts of every element to equal its expectation exactly. */
            template<typename T>
            void expect_equal(const std::vector<T>& actual, const std::vector<T>& expected)
            {
                ASSERT_EQ(actual.size(), expected.size());
                for (std::size_t i = 0; i < actual.size(); ++i)
                {
                    const Wide a = widened(actual[i]);
                    const Wide e = widened(expected[i]);
                    EXPECT_EQ(a.real(), e.real()) << "real part of element " << i;
                    EXPECT_EQ(a.imag(), e.imag()) << "imaginary part of element " << i;
                }
            }

            /**
             * @brief Expects both parts of every element within the rounding bound two kernels
             *        of the component type of T may differ by.
             *
             * The bound is rounded operations * unit roundoff * magnitude, measured through
             * TestSupport::measure_error; the magnitude is the largest of 1 and the expected
             * modulus, times K_ENTRY_PRODUCT_MAGNITUDE.
             */
            template<typename T>
            void expect_near(const std::vector<T>& actual, const std::vector<T>& expected)
            {
                using R = Kernels::RealType<T>;

                ASSERT_EQ(actual.size(), expected.size());
                for (std::size_t i = 0; i < actual.size(); ++i)
                {
                    const Wide a = widened(actual[i]);
                    const Wide e = widened(expected[i]);
                    const double magnitude =
                        K_ENTRY_PRODUCT_MAGNITUDE * std::max(1.0, std::abs(e));
                    const double bound = static_cast<double>(agreement_roundings<R>()) *
                                         unit_roundoff_of<R>() * magnitude;

                    EXPECT_TRUE(TestSupport::is_within(TestSupport::measure_error(
                        a.real(), e.real(), bound, K_AGREEMENT_THRESHOLD)))
                        << "real part of element " << i;
                    EXPECT_TRUE(TestSupport::is_within(TestSupport::measure_error(
                        a.imag(), e.imag(), bound, K_AGREEMENT_THRESHOLD)))
                        << "imaginary part of element " << i;
                }
            }
        } // namespace

        /**
         * @brief Builds band, dense and vector operands of any element type and records the
         *        band routines on them.
         */
        class BlasBandCase : public BlasEngineStart
        {
            protected:
                /** @brief Returns a tensor of dims that holds stored in storage order. */
                template<typename T>
                [[nodiscard]] Tensor tensor_of(const std::vector<int64_t>& dims,
                                               const std::vector<T>& stored)
                {
                    Tensor tensor = this->engine->create_tensor(
                        SushiRuntime::span<const int64_t>(dims.data(), dims.size()),
                        sushiblas_dtype<T>());
                    EXPECT_EQ(stored.size(), static_cast<std::size_t>(tensor.num_elements));
                    if (!stored.empty())
                        TestEnvironment::queue()
                            .memcpy(tensor.data(), stored.data(), stored.size() * sizeof(T))
                            .wait();
                    return tensor;
                }

                /** @brief Returns a rank-1 tensor that holds values. */
                template<typename T>
                [[nodiscard]] Tensor vector_of(const std::vector<T>& values)
                {
                    return tensor_of<T>({static_cast<int64_t>(values.size())}, values);
                }

                /** @brief Returns a rows x cols tensor that holds the row-ordered values. */
                template<typename T>
                [[nodiscard]] Tensor dense_of(int64_t rows, int64_t cols,
                                              const std::vector<T>& row_ordered)
                {
                    return tensor_of<T>({rows, cols}, in_layout(row_ordered, rows, cols,
                                                                this->engine->layout()));
                }

                /**
                 * @brief Returns the tensor of the band storage of an m x n matrix.
                 * @param band n lines of ld elements under COLUMN_MAJOR, m lines under ROW_MAJOR.
                 */
                template<typename T>
                [[nodiscard]] Tensor band_of(const std::vector<T>& band, int64_t m, int64_t n,
                                             int64_t ld)
                {
                    if (this->engine->layout() == Core::Layout::ColumnMajor)
                        return tensor_of<T>({ld, n}, band);
                    return tensor_of<T>({m, ld}, band);
                }

                /** @brief Returns the band tensor of one triangle of a dense n x n matrix. */
                template<typename T>
                [[nodiscard]] Tensor triangle_band_of(const std::vector<T>& dense, int64_t n,
                                                      Core::Uplo uplo, int64_t k)
                {
                    return band_of<T>(to_triangle_band(dense, n, uplo, k, k + 1,
                                                       this->engine->layout(), poison<T>()),
                                      n, n, k + 1);
                }

                /** @brief Returns a view of every second element of a rank-1 parent. */
                [[nodiscard]] static Tensor every_second_element(const Tensor& parent)
                {
                    const int64_t dims[] = {parent.num_elements / 2};
                    std::array<int64_t, Core::K_MAX_TENSOR_RANK> strides{};
                    strides[0] = 2;
                    return Tensor(parent.storage, SushiRuntime::span<const int64_t>(dims, 1),
                                  strides, parent.dtype, 0, parent.layout);
                }

                /**
                 * @brief Returns y after gbmv on the worked matrix.
                 * @param band Its band storage in the layout of the engine, ld elements a line.
                 */
                template<typename T>
                [[nodiscard]] std::vector<T> worked_gbmv(const std::vector<T>& band, int64_t ld,
                                                         Core::Transpose trans,
                                                         const std::vector<double>& x,
                                                         const std::vector<T>& y,
                                                         BLASScalar alpha, BLASScalar beta)
                {
                    const Tensor A = band_of<T>(band, K_WORKED_ROWS, K_WORKED_COLS, ld);
                    const Tensor x_tensor = vector_of<T>(elements<T>(x));
                    Tensor y_tensor = vector_of<T>(y);

                    SushiBLAS::BLASOps(this->recorder()).gbmv(A, x_tensor, y_tensor, K_WORKED_KL,
                                                           K_WORKED_KU, trans, alpha, beta);
                    this->recording->run().value();
                    return read_tensor<T>(y_tensor);
                }

                /** @brief Returns the literal band storage of the worked matrix for a layout. */
                template<typename T>
                [[nodiscard]] static std::vector<T> worked_band(Core::Layout layout)
                {
                    return elements<T>(layout == Core::Layout::ColumnMajor
                                           ? K_WORKED_COLUMN_MAJOR
                                           : K_WORKED_ROW_MAJOR);
                }

                /** @brief Returns the band tensor of K_HAND_UPPER or K_HAND_LOWER. */
                template<typename T>
                [[nodiscard]] Tensor hand_triangle(Core::Uplo uplo)
                {
                    return triangle_band_of<T>(
                        elements<T>(uplo == Core::Uplo::Upper ? K_HAND_UPPER : K_HAND_LOWER),
                        K_HAND_ORDER, uplo, 1);
                }

                /**
                 * @brief Records two calls on copies of initial in one graph, runs it and expects
                 *        the same elements from both.
                 * @param first Callable `(Tensor&)` that records on the first copy.
                 * @param second Callable `(Tensor&)` that records on the second copy.
                 * @return What the first call left.
                 */
                template<typename T, typename First, typename Second>
                std::vector<T> expect_same_result(const std::vector<T>& initial, First&& first,
                                                  Second&& second)
                {
                    Tensor from_first = vector_of<T>(initial);
                    Tensor from_second = vector_of<T>(initial);
                    first(from_first);
                    second(from_second);
                    this->recording->run().value();

                    const std::vector<T> result = read_tensor<T>(from_first);
                    expect_equal(result, read_tensor<T>(from_second));
                    return result;
                }

                /**
                 * @brief Returns y of gbmv on a full band of order K_LARGE_ORDER, then y of gemv
                 *        on the same matrix.
                 */
                template<typename T>
                [[nodiscard]] std::array<std::vector<T>, 2> large_general_products()
                {
                    const int64_t n = K_LARGE_ORDER;
                    const int64_t ld = 2 * n - 1;
                    std::mt19937 generator(K_SEED);
                    const std::vector<T> dense = random_elements<T>(generator, n * n);

                    const Tensor band = band_of<T>(
                        to_band(dense, n, n, n - 1, n - 1, ld, this->engine->layout(),
                                poison<T>()),
                        n, n, ld);
                    const Tensor full = dense_of<T>(n, n, dense);
                    const Tensor x = vector_of<T>(random_elements<T>(generator, n));
                    const std::vector<T> y = random_elements<T>(generator, n);
                    Tensor from_band = vector_of<T>(y);
                    Tensor from_dense = vector_of<T>(y);

                    SushiBLAS::BLASOps ops(this->recorder());
                    ops.gbmv(band, x, from_band, n - 1, n - 1, Core::Transpose::NoTrans, K_ALPHA,
                             K_BETA);
                    ops.gemv(full, x, from_dense, Core::Transpose::NoTrans, K_ALPHA, K_BETA);
                    this->recording->run().value();
                    return {{read_tensor<T>(from_band), read_tensor<T>(from_dense)}};
                }

                /**
                 * @brief Returns y of sbmv on a full upper band of order K_LARGE_ORDER, then y of
                 *        symv on the same matrix.
                 */
                template<typename T>
                [[nodiscard]] std::array<std::vector<T>, 2> large_symmetric_products()
                {
                    const int64_t n = K_LARGE_ORDER;
                    std::mt19937 generator(K_SEED);
                    const std::vector<T> dense = random_elements<T>(generator, n * n);

                    const Tensor band = triangle_band_of<T>(dense, n, Core::Uplo::Upper, n - 1);
                    const Tensor full = dense_of<T>(n, n, dense);
                    const Tensor x = vector_of<T>(random_elements<T>(generator, n));
                    const std::vector<T> y = random_elements<T>(generator, n);
                    Tensor from_band = vector_of<T>(y);
                    Tensor from_dense = vector_of<T>(y);

                    SushiBLAS::BLASOps ops(this->recorder());
                    ops.sbmv(band, x, from_band, Core::Uplo::Upper, n - 1, K_ALPHA, K_BETA);
                    ops.symv(full, x, from_dense, Core::Uplo::Upper, K_ALPHA, K_BETA);
                    this->recording->run().value();
                    return {{read_tensor<T>(from_band), read_tensor<T>(from_dense)}};
                }

                /**
                 * @brief Returns the results of gbmv, sbmv, hbmv, tbmv and tbsv, one after
                 *        another, on random matrices of order K_MODE_ORDER with k off-diagonals.
                 */
                template<typename T>
                [[nodiscard]] std::vector<T> banded_results(int64_t k)
                {
                    const int64_t n = K_MODE_ORDER;
                    const Core::Layout layout = this->engine->layout();
                    std::mt19937 generator(K_SEED);

                    const std::vector<T> general = random_elements<T>(generator, n * n);
                    std::vector<T> dominant = random_elements<T>(generator, n * n);
                    for (int64_t i = 0; i < n; ++i)
                        dominant[static_cast<std::size_t>(i * n + i)] =
                            narrowed<T>(Wide(static_cast<double>(k + 2), 0.0));
                    const std::vector<T> x = random_elements<T>(generator, n);
                    const std::vector<T> y = random_elements<T>(generator, n);

                    const Tensor band = band_of<T>(
                        to_band(general, n, n, k, k, 2 * k + 1, layout, poison<T>()), n, n,
                        2 * k + 1);
                    const Tensor upper = triangle_band_of<T>(general, n, Core::Uplo::Upper, k);
                    const Tensor lower = triangle_band_of<T>(general, n, Core::Uplo::Lower, k);
                    const Tensor upper_dominant =
                        triangle_band_of<T>(dominant, n, Core::Uplo::Upper, k);
                    const Tensor lower_dominant =
                        triangle_band_of<T>(dominant, n, Core::Uplo::Lower, k);
                    const Tensor x_tensor = vector_of<T>(x);

                    std::vector<Tensor> results;
                    for (int routine = 0; routine < 3; ++routine)
                        results.push_back(vector_of<T>(y));
                    for (int routine = 0; routine < 2; ++routine)
                        results.push_back(vector_of<T>(x));

                    SushiBLAS::BLASOps ops(this->recorder());
                    ops.gbmv(band, x_tensor, results[0], k, k, Core::Transpose::ConjTrans,
                             K_ALPHA, K_BETA);
                    ops.sbmv(upper, x_tensor, results[1], Core::Uplo::Upper, k, K_ALPHA, K_BETA);
                    ops.hbmv(lower, x_tensor, results[2], Core::Uplo::Lower, k, K_ALPHA, K_BETA);
                    ops.tbmv(upper_dominant, results[3], Core::Uplo::Upper, k,
                             Core::Transpose::Trans, Core::Diag::NonUnit);
                    ops.tbsv(lower_dominant, results[4], Core::Uplo::Lower, k,
                             Core::Transpose::NoTrans, Core::Diag::NonUnit);
                    this->recording->run().value();

                    std::vector<T> all;
                    for (const Tensor& result : results)
                    {
                        const std::vector<T> values = read_tensor<T>(result);
                        all.insert(all.end(), values.begin(), values.end());
                    }
                    return all;
                }
        };

        /**
         * @brief Runs a case once per row of the BlasBandCase suite.
         * @tparam Axis NoAxis, or the type of the value that completes each row.
         */
        template<typename Axis>
        using BandRowCase = BlasRowCase<BlasBandCase, Axis>;

        /** @brief Runs the worked-matrix, sibling and hand cases over the five element types. */
        class Medium_BlasBand : public BandRowCase<NoAxis>
        {

        };

        /** @brief Runs the gbmv edge shapes. */
        class Medium_BlasBandShape : public BandRowCase<BandShape>
        {

        };

        /** @brief Runs the cases that take one triangle. */
        class Medium_BlasBandTriangle : public BandRowCase<Core::Uplo>
        {

        };

        /** @brief Runs the cases that take one form of op(A). */
        class Medium_BlasBandForm : public BandRowCase<Core::Transpose>
        {

        };

        /** @brief Runs the hand products of the triangular routines. */
        class Medium_BlasBandProduct : public BandRowCase<TriangularRow>
        {

        };

        /** @brief Runs the full-triangle comparisons over every flag combination. */
        class Medium_BlasBandFlags : public BandRowCase<TriangleFlags>
        {

        };

        /** @brief Runs the comparison of the two kernels at each band width. */
        class Medium_BlasBandWidth : public BandRowCase<int64_t>
        {

        };

        /** @brief Runs the conjugating band cases over the complex element types. */
        class Medium_BlasBandComplex : public BandRowCase<NoAxis>
        {

        };

        /** @brief Runs the conjugating cases that take one triangle over the complex types. */
        class Medium_BlasBandComplexTriangle : public BandRowCase<Core::Uplo>
        {

        };

        /** @brief Runs the conjugating cases that take one form of op(A) over the complex types. */
        class Medium_BlasBandComplexForm : public BandRowCase<Core::Transpose>
        {

        };

        /** @brief Runs the HALF accumulation cases. */
        class Medium_BlasBandHalf : public BandRowCase<NoAxis>
        {

        };

        /** @brief Runs the FLOAT32 cases of order K_LARGE_ORDER. */
        class Medium_BlasBandLarge : public BandRowCase<NoAxis>
        {

        };

        /** @brief Tests gbmv on the literal band arrays of the worked matrix. */
        TEST_P(Medium_BlasBand, GbmvMatchesTheWorkedMatrix)
        {
            with_element(GetParam().type, [&](auto element)
            {
                using T = typename decltype(element)::type;

                const std::vector<T> y = worked_gbmv<T>(
                    worked_band<T>(GetParam().layout), K_WORKED_LD, Core::Transpose::NoTrans,
                    {1, 2, 3, 4, 5}, std::vector<T>(4, poison<T>()), 1.0, 0.0);

                expect_equal(y, elements<T>({74, 230, 474, 530}));
            });
        }

        /** @brief Tests that gbmv scales the product by alpha and the prior y by beta. */
        TEST_P(Medium_BlasBand, GbmvScalesAndAccumulates)
        {
            with_element(GetParam().type, [&](auto element)
            {
                using T = typename decltype(element)::type;

                const std::vector<T> y = worked_gbmv<T>(
                    worked_band<T>(GetParam().layout), K_WORKED_LD, Core::Transpose::NoTrans,
                    {1, 2, 3, 4, 5}, elements<T>({1, 1, 1, 1}), 2.0, -1.0);

                expect_equal(y, elements<T>({147, 459, 947, 1059}));
            });
        }

        /** @brief Tests gbmv on the transpose of the worked matrix. */
        TEST_P(Medium_BlasBand, GbmvTransposed)
        {
            with_element(GetParam().type, [&](auto element)
            {
                using T = typename decltype(element)::type;

                const std::vector<T> y = worked_gbmv<T>(
                    worked_band<T>(GetParam().layout), K_WORKED_LD, Core::Transpose::Trans,
                    {1, 2, 3, 4}, std::vector<T>(5, poison<T>()), 1.0, 0.0);

                expect_equal(y, elements<T>({53, 152, 330, 326, 285}));
            });
        }

        /** @brief Tests that gbmv reads no slot past the band of a line. */
        TEST_P(Medium_BlasBand, GbmvIgnoresSpareSlots)
        {
            constexpr int64_t wide_ld = K_WORKED_LD + 2;

            with_element(GetParam().type, [&](auto element)
            {
                using T = typename decltype(element)::type;

                const std::vector<T> band =
                    to_band(elements<T>(K_WORKED_DENSE), K_WORKED_ROWS, K_WORKED_COLS,
                            K_WORKED_KL, K_WORKED_KU, wide_ld, GetParam().layout, poison<T>());
                const std::vector<T> y = worked_gbmv<T>(
                    band, wide_ld, Core::Transpose::NoTrans, {1, 2, 3, 4, 5},
                    std::vector<T>(4, poison<T>()), 1.0, 0.0);

                expect_equal(y, elements<T>({74, 230, 474, 530}));
            });
        }

        /** @brief Tests gbmv on a shape at the edge of what it accepts against a host loop. */
        TEST_P(Medium_BlasBandShape, GbmvMatchesAHostLoop)
        {
            const BandShape shape = GetParam().axis;
            const int64_t ld = shape.kl + shape.ku + 1;

            std::vector<double> dense(static_cast<std::size_t>(shape.m * shape.n));
            std::vector<double> x(static_cast<std::size_t>(shape.n));
            std::vector<double> expected(static_cast<std::size_t>(shape.m), 0.0);
            for (int64_t j = 0; j < shape.n; ++j)
                x[static_cast<std::size_t>(j)] = static_cast<double>(j + 1);
            for (int64_t i = 0; i < shape.m; ++i)
            {
                for (int64_t j = 0; j < shape.n; ++j)
                {
                    const std::size_t at = static_cast<std::size_t>(i * shape.n + j);
                    dense[at] = static_cast<double>((i * shape.n + j) % 5 + 1);
                    if (j - i <= shape.ku && i - j <= shape.kl)
                        expected[static_cast<std::size_t>(i)] +=
                            dense[at] * x[static_cast<std::size_t>(j)];
                }
            }

            with_element(GetParam().type, [&](auto element)
            {
                using T = typename decltype(element)::type;

                const Tensor A = band_of<T>(
                    to_band(elements<T>(dense), shape.m, shape.n, shape.kl, shape.ku, ld,
                            GetParam().layout, poison<T>()),
                    shape.m, shape.n, ld);
                const Tensor x_tensor = vector_of<T>(elements<T>(x));
                Tensor y = vector_of<T>(
                    std::vector<T>(static_cast<std::size_t>(shape.m), poison<T>()));

                SushiBLAS::BLASOps(recorder()).gbmv(A, x_tensor, y, shape.kl, shape.ku);
                recording->run().value();

                expect_equal(read_tensor<T>(y), elements<T>(expected));
            });
        }

        /** @brief Tests that gbmv on a matrix of no columns only scales y by beta. */
        TEST_P(Medium_BlasBand, GbmvOfNoColumnsScalesY)
        {
            with_element(GetParam().type, [&](auto element)
            {
                using T = typename decltype(element)::type;
                const bool by_columns = (GetParam().layout == Core::Layout::ColumnMajor);

                const Tensor A = band_of<T>(
                    std::vector<T>(by_columns ? std::size_t{0} : std::size_t{2}, poison<T>()),
                    2, 0, 1);
                const Tensor x = vector_of<T>(std::vector<T>{});
                Tensor y = vector_of<T>(elements<T>({1, 2}));

                SushiBLAS::BLASOps(recorder()).gbmv(A, x, y, 0, 0, Core::Transpose::NoTrans,
                                                   1.0, 3.0);
                recording->run().value();

                expect_equal(read_tensor<T>(y), elements<T>({3, 6}));
            });
        }

        /**
         * @brief Tests sbmv on the literal band arrays of a tridiagonal matrix, where a
         *        mistaken layout or triangle reads another matrix.
         */
        TEST_P(Medium_BlasBandTriangle, SbmvMatchesTheTridiagonal)
        {
            const int64_t n = K_TRIDIAGONAL_ORDER;
            const Core::Uplo uplo = GetParam().axis;
            const bool diagonal_last =
                (uplo == Core::Uplo::Upper) == (GetParam().layout == Core::Layout::ColumnMajor);

            with_element(GetParam().type, [&](auto element)
            {
                using T = typename decltype(element)::type;

                const Tensor A = band_of<T>(
                    elements<T>(diagonal_last ? K_TRIDIAGONAL_DIAGONAL_LAST
                                              : K_TRIDIAGONAL_DIAGONAL_FIRST),
                    n, n, 2);
                const Tensor x = vector_of<T>(elements<T>({1, 2, 3, 4}));
                Tensor y = vector_of<T>(std::vector<T>(4, poison<T>()));

                SushiBLAS::BLASOps(recorder()).sbmv(A, x, y, uplo, 1);
                recording->run().value();

                expect_equal(read_tensor<T>(y), elements<T>({5, 20, 47, 46}));
            });
        }

        /** @brief Tests tbmv on one hand triangle, transposed or with a unit diagonal. */
        TEST_P(Medium_BlasBandProduct, TbmvMatchesByHand)
        {
            const TriangularRow row = GetParam().axis;

            with_element(GetParam().type, [&](auto element)
            {
                using T = typename decltype(element)::type;

                const Tensor A = hand_triangle<T>(row.uplo);
                Tensor x = vector_of<T>(elements<T>({1, 2, 3}));

                SushiBLAS::BLASOps(recorder()).tbmv(A, x, row.uplo, 1, row.trans, row.diag);
                recording->run().value();

                expect_equal(read_tensor<T>(x),
                             elements<T>({row.product[0], row.product[1], row.product[2]}));
            });
        }

        /** @brief Tests that tbsv recovers [1, 2, 3] from the product of TbmvMatchesByHand. */
        TEST_P(Medium_BlasBandProduct, TbsvInvertsTbmv)
        {
            const TriangularRow row = GetParam().axis;

            with_element(GetParam().type, [&](auto element)
            {
                using T = typename decltype(element)::type;

                const Tensor A = hand_triangle<T>(row.uplo);
                Tensor b = vector_of<T>(
                    elements<T>({row.product[0], row.product[1], row.product[2]}));

                SushiBLAS::BLASOps(recorder()).tbsv(A, b, row.uplo, 1, row.trans, row.diag);
                recording->run().value();

                expect_equal(read_tensor<T>(b), elements<T>({1, 2, 3}));
            });
        }

        /**
         * @brief Tests that tbsv solves in a view of every second element and leaves the
         *        elements between as they were.
         */
        TEST_P(Medium_BlasBand, TbsvTakesAStridedVector)
        {
            with_element(GetParam().type, [&](auto element)
            {
                using T = typename decltype(element)::type;

                const Tensor A = hand_triangle<T>(Core::Uplo::Upper);
                Tensor parent = vector_of<T>(
                    elements<T>({5, K_POISON, 18, K_POISON, 15, K_POISON}));
                Tensor b = every_second_element(parent);

                SushiBLAS::BLASOps(recorder()).tbsv(A, b, Core::Uplo::Upper, 1);
                recording->run().value();

                expect_equal(read_tensor<T>(parent),
                             elements<T>({1, K_POISON, 2, K_POISON, 3, K_POISON}));
            });
        }

        /** @brief Tests that a band wide enough to hold the whole matrix gives the elements of gemv. */
        TEST_P(Medium_BlasBandForm, AFullGeneralBandEqualsGemv)
        {
            const Core::Transpose trans = GetParam().axis;
            const bool transposed = Core::is_transposed(trans);

            with_element(GetParam().type, [&](auto element)
            {
                using T = typename decltype(element)::type;
                std::mt19937 generator(K_SEED);
                SushiBLAS::BLASOps ops(recorder());

                const int64_t m = K_SIBLING_ROWS;
                const int64_t n = K_SIBLING_COLS;
                const int64_t ld = m + n - 1;
                const std::vector<T> general = random_elements<T>(generator, m * n);
                const Tensor band = band_of<T>(
                    to_band(general, m, n, m - 1, n - 1, ld, GetParam().layout, poison<T>()), m, n,
                    ld);
                const Tensor full = dense_of<T>(m, n, general);
                const Tensor x = vector_of<T>(random_elements<T>(generator, transposed ? m : n));
                const std::vector<T> y = random_elements<T>(generator, transposed ? n : m);

                expect_same_result<T>(
                    y,
                    [&](Tensor& out)
                    {
                        ops.gbmv(band, x, out, m - 1, n - 1, trans, K_ALPHA, K_BETA);
                    },
                    [&](Tensor& out) { ops.gemv(full, x, out, trans, K_ALPHA, K_BETA); });
            });
        }

        /** @brief Tests that a full symmetric band gives the elements of symv. */
        TEST_P(Medium_BlasBandTriangle, AFullSymmetricBandEqualsSymv)
        {
            const Core::Uplo uplo = GetParam().axis;

            with_element(GetParam().type, [&](auto element)
            {
                using T = typename decltype(element)::type;
                std::mt19937 generator(K_SEED);
                SushiBLAS::BLASOps ops(recorder());

                const int64_t order = K_SIBLING_ORDER;
                const std::vector<T> square = random_elements<T>(generator, order * order);
                const Tensor square_full = dense_of<T>(order, order, square);
                const Tensor triangle = triangle_band_of<T>(square, order, uplo, order - 1);
                const Tensor x = vector_of<T>(random_elements<T>(generator, order));
                const std::vector<T> y = random_elements<T>(generator, order);

                expect_same_result<T>(
                    y,
                    [&](Tensor& out)
                    {
                        ops.sbmv(triangle, x, out, uplo, order - 1, K_ALPHA, K_BETA);
                    },
                    [&](Tensor& out) { ops.symv(square_full, x, out, uplo, K_ALPHA, K_BETA); });
            });
        }

        /** @brief Tests that a full Hermitian band gives the elements of hemv. */
        TEST_P(Medium_BlasBandTriangle, AFullHermitianBandEqualsHemv)
        {
            const Core::Uplo uplo = GetParam().axis;

            with_element(GetParam().type, [&](auto element)
            {
                using T = typename decltype(element)::type;
                std::mt19937 generator(K_SEED);
                SushiBLAS::BLASOps ops(recorder());

                const int64_t order = K_SIBLING_ORDER;
                const std::vector<T> square = random_elements<T>(generator, order * order);
                const Tensor square_full = dense_of<T>(order, order, square);
                const Tensor triangle = triangle_band_of<T>(square, order, uplo, order - 1);
                const Tensor x = vector_of<T>(random_elements<T>(generator, order));
                const std::vector<T> y = random_elements<T>(generator, order);

                expect_same_result<T>(
                    y,
                    [&](Tensor& out)
                    {
                        ops.hbmv(triangle, x, out, uplo, order - 1, K_ALPHA, K_BETA);
                    },
                    [&](Tensor& out) { ops.hemv(square_full, x, out, uplo, K_ALPHA, K_BETA); });
            });
        }

        /**
         * @brief Tests that tbmv and tbsv on a full triangle band give the elements of trmv and
         *        trsv, and the product and the solution formed in integers on the host.
         */
        TEST_P(Medium_BlasBandFlags, AFullTriangleBandEqualsTrmvAndTrsv)
        {
            const int64_t n = K_SIBLING_ORDER;
            const TriangleFlags flags = GetParam().axis;

            with_element(GetParam().type, [&](auto element)
            {
                using T = typename decltype(element)::type;
                SushiBLAS::BLASOps ops(recorder());

                const ExactTriangle exact = exact_triangle(n, flags.uplo, flags.trans, flags.diag,
                                                           Core::is_complex_v<T>);
                const std::vector<T> stored = narrowed_all<T>(exact.stored);
                const std::vector<T> x = narrowed_all<T>(exact.x);
                const std::vector<T> b = narrowed_all<T>(exact.b);

                const Tensor band = triangle_band_of<T>(stored, n, flags.uplo, n - 1);
                const Tensor full = dense_of<T>(n, n, stored);

                const std::vector<T> product = expect_same_result<T>(
                    x,
                    [&](Tensor& out)
                    {
                        ops.tbmv(band, out, flags.uplo, n - 1, flags.trans, flags.diag);
                    },
                    [&](Tensor& out) { ops.trmv(full, out, flags.uplo, flags.trans, flags.diag); });
                expect_equal(product, b);

                const std::vector<T> solution = expect_same_result<T>(
                    b,
                    [&](Tensor& out)
                    {
                        ops.tbsv(band, out, flags.uplo, n - 1, flags.trans, flags.diag);
                    },
                    [&](Tensor& out) { ops.trsv(full, out, flags.uplo, flags.trans, flags.diag); });
                expect_equal(solution, x);
            });
        }

        /**
         * @brief Tests that the two kernels agree within the rounding bound at one band width.
         *
         * The row names the SYCL kernel; the case runs ForceReference itself to compare it with.
         */
        TEST_P(Medium_BlasBandWidth, BothModesAgree)
        {
            const int64_t k = GetParam().axis;

            with_element(GetParam().type, [&](auto element)
            {
                using T = typename decltype(element)::type;

                start(GetParam().layout, ExecutionMode::ForceReference);
                const std::vector<T> reference = banded_results<T>(k);
                start(GetParam().layout, ExecutionMode::ForceSYCL);
                const std::vector<T> device = banded_results<T>(k);

                expect_near(device, reference);
            });
        }

        /** @brief Tests gbmv under one form of op(A) on a complex band matrix. */
        TEST_P(Medium_BlasBandComplexForm, GbmvConjTransposed)
        {
            const std::vector<Wide> dense = {Wide(1, 2), Wide(3, -1), Wide(0, 0), Wide(4, 1)};
            const std::vector<Wide> x = {Wide(1, 0), Wide(0, 1)};
            const std::vector<Wide> expected[] = {{Wide(2, 5), Wide(-1, 4)},
                                                  {Wide(1, 2), Wide(2, 3)},
                                                  {Wide(1, -2), Wide(4, 5)}};
            const Core::Transpose trans = GetParam().axis;

            with_element(GetParam().type, [&](auto element)
            {
                using T = typename decltype(element)::type;

                const Tensor A = band_of<T>(
                    to_band(narrowed_all<T>(dense), 2, 2, 0, 1, 2, GetParam().layout, poison<T>()),
                    2, 2, 2);
                const Tensor x_tensor = vector_of<T>(narrowed_all<T>(x));
                Tensor y = vector_of<T>(std::vector<T>(2, poison<T>()));

                SushiBLAS::BLASOps(recorder()).gbmv(A, x_tensor, y, 0, 1, trans);
                recording->run().value();

                expect_equal(read_tensor<T>(y), narrowed_all<T>(expected[form_index(trans)]));
            });
        }

        /**
         * @brief Tests that hbmv conjugates the mirrored triangle and reads no imaginary
         *        diagonal.
         */
        TEST_P(Medium_BlasBandComplexTriangle, HbmvConjugatesTheMirror)
        {
            const int64_t n = K_HAND_ORDER;
            const std::vector<Wide> hermitian = {
                Wide(1, K_DIAGONAL_IMAGINARY), Wide(2, 1),  Wide(0, 0),
                Wide(2, -1), Wide(3, K_DIAGONAL_IMAGINARY), Wide(4, -2),
                Wide(0, 0),  Wide(4, 2), Wide(5, K_DIAGONAL_IMAGINARY)};
            const std::vector<Wide> x = {Wide(1, 0), Wide(0, 1), Wide(2, 0)};

            with_element(GetParam().type, [&](auto element)
            {
                using T = typename decltype(element)::type;

                const Tensor x_tensor = vector_of<T>(narrowed_all<T>(x));
                const Tensor A =
                    triangle_band_of<T>(narrowed_all<T>(hermitian), n, GetParam().axis, 1);
                Tensor y = vector_of<T>(std::vector<T>(3, poison<T>()));

                SushiBLAS::BLASOps(recorder()).hbmv(A, x_tensor, y, GetParam().axis, 1);
                recording->run().value();

                expect_equal(read_tensor<T>(y),
                             narrowed_all<T>({Wide(0, 2), Wide(10, -2), Wide(8, 4)}));
            });
        }

        /** @brief Tests that sbmv on the Hermitian storage mirrors without conjugating. */
        TEST_P(Medium_BlasBandComplex, SbmvMirrorsWithoutConjugating)
        {
            const int64_t n = K_HAND_ORDER;
            const std::vector<Wide> real_diagonal = {Wide(1, 0), Wide(2, 1),  Wide(0, 0),
                                                     Wide(2, -1), Wide(3, 0), Wide(4, -2),
                                                     Wide(0, 0),  Wide(4, 2), Wide(5, 0)};
            const std::vector<Wide> x = {Wide(1, 0), Wide(0, 1), Wide(2, 0)};

            with_element(GetParam().type, [&](auto element)
            {
                using T = typename decltype(element)::type;

                const Tensor x_tensor = vector_of<T>(narrowed_all<T>(x));
                const Tensor A = triangle_band_of<T>(narrowed_all<T>(real_diagonal), n,
                                                     Core::Uplo::Upper, 1);
                Tensor y = vector_of<T>(std::vector<T>(3, poison<T>()));

                SushiBLAS::BLASOps(recorder()).sbmv(A, x_tensor, y, Core::Uplo::Upper, 1);
                recording->run().value();

                expect_equal(read_tensor<T>(y),
                             narrowed_all<T>({Wide(0, 2), Wide(10, 0), Wide(12, 4)}));
            });
        }

        /**
         * @brief Tests that gbmv, sbmv and tbmv sum a HALF row in FLOAT32: 2048 + 1 + 1 - 2048
         *        is 2 there and 0 in HALF.
         */
        TEST_P(Medium_BlasBandHalf, HalfSumsInFloat)
        {
            using T = sycl::half;
            constexpr int64_t n = 4;
            const std::vector<double> row = {2048, 1, 1, -2048};
            std::vector<double> square(static_cast<std::size_t>(n * n), 0.0);
            std::copy(row.begin(), row.end(), square.begin());

            SushiBLAS::BLASOps ops(recorder());
            const Tensor ones = vector_of<T>(elements<T>({1, 1, 1, 1}));

            const Tensor general = band_of<T>(
                to_band(elements<T>(row), 1, n, 0, n - 1, n, GetParam().layout, poison<T>()), 1, n,
                n);
            Tensor from_gbmv = vector_of<T>(std::vector<T>(1, poison<T>()));
            ops.gbmv(general, ones, from_gbmv, 0, n - 1);

            const Tensor triangle =
                triangle_band_of<T>(elements<T>(square), n, Core::Uplo::Upper, n - 1);
            Tensor from_sbmv = vector_of<T>(std::vector<T>(4, poison<T>()));
            ops.sbmv(triangle, ones, from_sbmv, Core::Uplo::Upper, n - 1);

            Tensor from_tbmv = vector_of<T>(elements<T>({1, 1, 1, 1}));
            ops.tbmv(triangle, from_tbmv, Core::Uplo::Upper, n - 1);
            recording->run().value();

            expect_equal(read_tensor<T>(from_gbmv), elements<T>({2}));
            expect_equal(read_tensor<T>(from_sbmv), elements<T>({2, 1, 1, -2048}));
            expect_equal(read_tensor<T>(from_tbmv), elements<T>({2, 0, 0, 0}));
        }

        /**
         * @brief Tests that tbsv carries a HALF solution in FLOAT32: the middle element is
         *        2049 there, stored as 2048, and the last is 2050 - 2049.
         */
        TEST_P(Medium_BlasBandHalf, HalfSolveCarriesTheSolutionInFloat)
        {
            using T = sycl::half;
            const std::vector<double> lower = {1, 0, 0, -1, 1, 0, 0, 1, 1};

            const Tensor A =
                triangle_band_of<T>(elements<T>(lower), K_HAND_ORDER, Core::Uplo::Lower, 1);
            Tensor b = vector_of<T>(elements<T>({2048, 1, 2050}));

            SushiBLAS::BLASOps(recorder()).tbsv(A, b, Core::Uplo::Lower, 1);
            recording->run().value();

            expect_equal(read_tensor<T>(b), elements<T>({2048, 2048, 1}));
        }

        /**
         * @brief Tests gbmv and sbmv against gemv and symv at an order where a device that
         *        plans a work-group per row runs one.
         */
        TEST_P(Medium_BlasBandLarge, AFullBandEqualsTheDenseSiblingLarge)
        {
            const std::array<std::vector<float>, 2> general = large_general_products<float>();
            expect_equal(general[0], general[1]);

            const std::array<std::vector<float>, 2> symmetric = large_symmetric_products<float>();
            expect_equal(symmetric[0], symmetric[1]);
        }

        /** @brief Tests that gbmv and sbmv give the same bits when the device runs them twice. */
        TEST_P(Medium_BlasBandLarge, RepeatsToTheBit)
        {
            const auto same_bits = [](const std::vector<float>& first,
                                      const std::vector<float>& second)
            {
                return first.size() == second.size() &&
                       std::memcmp(first.data(), second.data(),
                                   first.size() * sizeof(float)) == 0;
            };

            const std::vector<float> general = large_general_products<float>()[0];
            EXPECT_TRUE(same_bits(general, large_general_products<float>()[0])) << "gbmv";

            const std::vector<float> symmetric = large_symmetric_products<float>()[0];
            EXPECT_TRUE(same_bits(symmetric, large_symmetric_products<float>()[0])) << "sbmv";
        }

        INSTANTIATE_TEST_SUITE_P(
            , Medium_BlasBand, ::testing::ValuesIn(plain_rows(K_ALL_TYPES, K_BOTH_MODES)),
            row_name<NoAxis>);
        INSTANTIATE_TEST_SUITE_P(
            , Medium_BlasBandShape,
            ::testing::ValuesIn(rows_of(K_ALL_TYPES, K_BOTH_MODES, K_EDGE_SHAPES)),
            row_name<BandShape>);
        INSTANTIATE_TEST_SUITE_P(
            , Medium_BlasBandTriangle,
            ::testing::ValuesIn(rows_of(K_ALL_TYPES, K_BOTH_MODES, K_TRIANGLES)),
            row_name<Core::Uplo>);
        INSTANTIATE_TEST_SUITE_P(
            , Medium_BlasBandForm,
            ::testing::ValuesIn(rows_of(K_ALL_TYPES, K_BOTH_MODES, K_TRANSPOSES)),
            row_name<Core::Transpose>);
        INSTANTIATE_TEST_SUITE_P(
            , Medium_BlasBandProduct,
            ::testing::ValuesIn(rows_of(K_ALL_TYPES, K_BOTH_MODES, K_TRIANGULAR_ROWS)),
            row_name<TriangularRow>);
        INSTANTIATE_TEST_SUITE_P(
            , Medium_BlasBandFlags,
            ::testing::ValuesIn(rows_of(K_ALL_TYPES, K_BOTH_MODES, all_triangle_flags())),
            row_name<TriangleFlags>);
        INSTANTIATE_TEST_SUITE_P(
            , Medium_BlasBandWidth,
            ::testing::ValuesIn(rows_of(K_ALL_TYPES, K_DEVICE_MODE, K_MODE_WIDTHS)),
            row_name<int64_t>);
        INSTANTIATE_TEST_SUITE_P(
            , Medium_BlasBandComplex,
            ::testing::ValuesIn(plain_rows(K_COMPLEX_TYPES, K_BOTH_MODES)), row_name<NoAxis>);
        INSTANTIATE_TEST_SUITE_P(
            , Medium_BlasBandComplexTriangle,
            ::testing::ValuesIn(rows_of(K_COMPLEX_TYPES, K_BOTH_MODES, K_TRIANGLES)),
            row_name<Core::Uplo>);
        INSTANTIATE_TEST_SUITE_P(
            , Medium_BlasBandComplexForm,
            ::testing::ValuesIn(rows_of(K_COMPLEX_TYPES, K_BOTH_MODES, K_TRANSPOSES)),
            row_name<Core::Transpose>);
        INSTANTIATE_TEST_SUITE_P(
            , Medium_BlasBandHalf, ::testing::ValuesIn(plain_rows(K_HALF_TYPE, K_BOTH_MODES)),
            row_name<NoAxis>);
        INSTANTIATE_TEST_SUITE_P(
            , Medium_BlasBandLarge, ::testing::ValuesIn(plain_rows(K_FLOAT_TYPE, K_BOTH_MODES)),
            row_name<NoAxis>);
    } // namespace Test
} // namespace SushiBLAS
