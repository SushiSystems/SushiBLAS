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
 * @file test_blas_band.cpp
 * @brief Tests the band routines against hand-computed results, their dense siblings and
 *        each other's kernels.
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
#include <SushiBLAS/typed_suite_naming/default_type_names.hpp>
#include <SushiBLAS/test_runner/test_environment.hpp>
#include <SushiBLAS/blas_test_oracle/blas_structured.hpp>

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

            /// @brief Band widths the two kernels are compared at.
            constexpr int64_t K_MODE_WIDTHS[] = {0, 1, 5, 63};

            /// @brief Both kernels an operation has.
            constexpr ExecutionMode K_MODES[] = {ExecutionMode::ForceReference,
                                                 ExecutionMode::ForceSYCL};

            /// @brief Both element orders a band can be stored in.
            constexpr Core::Layout K_LAYOUTS[] = {Core::Layout::RowMajor,
                                                  Core::Layout::ColumnMajor};

            /// @brief Both triangles a band can hold.
            constexpr Core::Uplo K_TRIANGLES[] = {Core::Uplo::Upper, Core::Uplo::Lower};

            /// @brief Every form op(A) takes.
            constexpr Core::Transpose K_TRANSPOSES[] = {Core::Transpose::NoTrans,
                                                        Core::Transpose::Trans,
                                                        Core::Transpose::ConjTrans};

            /// @brief Both ways a triangular diagonal is read.
            constexpr Core::Diag K_DIAGONALS[] = {Core::Diag::NonUnit, Core::Diag::Unit};

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

            /** @brief Holds the flags of one triangular product of [1, 2, 3] and its result. */
            struct TriangularRow
            {
                Core::Uplo uplo;              /**< The triangle: K_HAND_UPPER or K_HAND_LOWER. */
                Core::Transpose trans;        /**< The form of op(A). */
                Core::Diag diag;              /**< How the diagonal is read. */
                double product[K_HAND_ORDER]; /**< op(A) * [1, 2, 3]. */
            };

            /// @brief The triangular products of the design, section 10.3.
            constexpr TriangularRow K_TRIANGULAR_ROWS[] = {
                {Core::Uplo::Upper, Core::Transpose::NoTrans, Core::Diag::NonUnit, {5, 18, 15}},
                {Core::Uplo::Upper, Core::Transpose::Trans, Core::Diag::NonUnit, {1, 8, 23}},
                {Core::Uplo::Upper, Core::Transpose::NoTrans, Core::Diag::Unit, {5, 14, 3}},
                {Core::Uplo::Upper, Core::Transpose::Trans, Core::Diag::Unit, {1, 4, 11}},
                {Core::Uplo::Lower, Core::Transpose::NoTrans, Core::Diag::NonUnit, {1, 8, 23}},
                {Core::Uplo::Lower, Core::Transpose::Trans, Core::Diag::NonUnit, {5, 18, 15}}};

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
            constexpr BandShape K_EDGE_SHAPES[] = {{1, 1, 0, 0},
                                                   {5, 2, 3, 0},
                                                   {2, 5, 0, 3},
                                                   {3, 4, 7, 7}};

            /**
             * @brief Holds how far the results of two kernels may differ, as a fraction of the
             *        larger of 1 and the expected magnitude.
             * @tparam R sycl::half, float or double.
             */
            template<typename R>
            struct ModeTolerance;

            /** @brief Allows two units in the last place of HALF. */
            template<>
            struct ModeTolerance<sycl::half>
            {
                static constexpr double K_VALUE = 0.001953125; /**< 2^-9. */
            };

            /** @brief Allows the rounding of a sum of 127 FLOAT32 products of unit size. */
            template<>
            struct ModeTolerance<float>
            {
                static constexpr double K_VALUE = 1e-4; /**< About 800 epsilon. */
            };

            /** @brief Allows the rounding of a sum of 127 FLOAT64 products of unit size. */
            template<>
            struct ModeTolerance<double>
            {
                static constexpr double K_VALUE = 1e-11; /**< About 45000 epsilon. */
            };

            /** @brief Returns the name of a mode and a layout for a failure message. */
            [[nodiscard]] std::string engine_name(Core::Layout layout, ExecutionMode mode)
            {
                return std::string(mode == ExecutionMode::ForceSYCL ? "ForceSYCL"
                                                                     : "ForceReference") +
                       (layout == Core::Layout::RowMajor ? ", ROW_MAJOR" : ", COLUMN_MAJOR");
            }

            /** @brief Returns the names of the flags of a triangular call. */
            [[nodiscard]] std::string flags_name(Core::Uplo uplo, Core::Transpose trans,
                                                 Core::Diag diag)
            {
                return std::string(Core::to_string(uplo)) + ", " + Core::to_string(trans) + ", " +
                       Core::to_string(diag);
            }

            /** @brief Returns whether the test device computes in the component type of T. */
            template<typename T>
            [[nodiscard]] bool device_runs()
            {
                using R = Kernels::RealType<T>;
                const sycl::device device =
                    TestEnvironment::queue().get_device();
                if constexpr (std::is_same_v<R, sycl::half>)
                    return device.has(sycl::aspect::fp16);
                else if constexpr (std::is_same_v<R, double>)
                    return device.has(sycl::aspect::fp64);
                else
                    return true;
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

            /** @brief Expects both parts of every element within ModeTolerance of expected. */
            template<typename T>
            void expect_near(const std::vector<T>& actual, const std::vector<T>& expected)
            {
                const double tolerance = ModeTolerance<Kernels::RealType<T>>::K_VALUE;

                ASSERT_EQ(actual.size(), expected.size());
                for (std::size_t i = 0; i < actual.size(); ++i)
                {
                    const Wide a = widened(actual[i]);
                    const Wide e = widened(expected[i]);
                    const double bound = tolerance * std::max(1.0, std::abs(e));
                    EXPECT_LE(std::abs(a.real() - e.real()), bound)
                        << "real part of element " << i;
                    EXPECT_LE(std::abs(a.imag() - e.imag()), bound)
                        << "imaginary part of element " << i;
                }
            }
        } // namespace

        /**
         * @brief Builds band, dense and vector operands of element type T and records the band
         *        routines on them.
         */
        template<typename T>
        class BLASBandCases : public SushiBLASTest
        {
            protected:
                /** @brief Skips the case on a device that lacks the component type of T. */
                void SetUp() override
                {
                    SushiBLASTest::SetUp();
                    if (!device_runs<T>())
                        GTEST_SKIP() << "the device does not compute in this element type";
                }

                /** @brief Rebuilds the engine in a layout and selects a kernel. */
                void start(Core::Layout layout, ExecutionMode mode)
                {
                    this->reinit_engine(layout);
                    this->set_mode(mode);
                }

                /**
                 * @brief Runs body once in each layout under each kernel.
                 * @param body Callable `(Core::Layout)`, called after the engine is rebuilt.
                 */
                template<typename Body>
                void for_each_engine(Body&& body)
                {
                    for (const Core::Layout layout : K_LAYOUTS)
                    {
                        for (const ExecutionMode mode : K_MODES)
                        {
                            SCOPED_TRACE(engine_name(layout, mode));
                            start(layout, mode);
                            body(layout);
                        }
                    }
                }

                /** @brief Returns a tensor of dims that holds stored in storage order. */
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
                [[nodiscard]] Tensor vector_of(const std::vector<T>& values)
                {
                    return tensor_of({static_cast<int64_t>(values.size())}, values);
                }

                /** @brief Returns a rows x cols tensor that holds the row-ordered values. */
                [[nodiscard]] Tensor dense_of(int64_t rows, int64_t cols,
                                              const std::vector<T>& row_ordered)
                {
                    return tensor_of({rows, cols}, in_layout(row_ordered, rows, cols,
                                                             this->engine->layout()));
                }

                /**
                 * @brief Returns the tensor of the band storage of an m x n matrix.
                 * @param band n lines of ld elements under COLUMN_MAJOR, m lines under ROW_MAJOR.
                 */
                [[nodiscard]] Tensor band_of(const std::vector<T>& band, int64_t m, int64_t n,
                                             int64_t ld)
                {
                    if (this->engine->layout() == Core::Layout::ColumnMajor)
                        return tensor_of({ld, n}, band);
                    return tensor_of({m, ld}, band);
                }

                /** @brief Returns the band tensor of one triangle of a dense n x n matrix. */
                [[nodiscard]] Tensor triangle_band_of(const std::vector<T>& dense, int64_t n,
                                                      Core::Uplo uplo, int64_t k)
                {
                    return band_of(to_triangle_band(dense, n, uplo, k, k + 1,
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
                [[nodiscard]] std::vector<T> worked_gbmv(const std::vector<T>& band, int64_t ld,
                                                         Core::Transpose trans,
                                                         const std::vector<double>& x,
                                                         const std::vector<T>& y,
                                                         BLASScalar alpha, BLASScalar beta)
                {
                    const Tensor A = band_of(band, K_WORKED_ROWS, K_WORKED_COLS, ld);
                    const Tensor x_tensor = vector_of(elements<T>(x));
                    Tensor y_tensor = vector_of(y);

                    SushiBLAS::BLASOps(this->recorder()).gbmv(A, x_tensor, y_tensor, K_WORKED_KL,
                                                           K_WORKED_KU, trans, alpha, beta);
                    this->recording->run().value();
                    return read_tensor<T>(y_tensor);
                }

                /** @brief Returns the literal band storage of the worked matrix for a layout. */
                [[nodiscard]] static std::vector<T> worked_band(Core::Layout layout)
                {
                    return elements<T>(layout == Core::Layout::ColumnMajor
                                           ? K_WORKED_COLUMN_MAJOR
                                           : K_WORKED_ROW_MAJOR);
                }

                /** @brief Returns the band tensor of K_HAND_UPPER or K_HAND_LOWER. */
                [[nodiscard]] Tensor hand_triangle(Core::Uplo uplo)
                {
                    return triangle_band_of(
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
                template<typename First, typename Second>
                std::vector<T> expect_same_result(const std::vector<T>& initial, First&& first,
                                                  Second&& second)
                {
                    Tensor from_first = vector_of(initial);
                    Tensor from_second = vector_of(initial);
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
                [[nodiscard]] std::array<std::vector<T>, 2> large_general_products()
                {
                    const int64_t n = K_LARGE_ORDER;
                    const int64_t ld = 2 * n - 1;
                    std::mt19937 generator(K_SEED);
                    const std::vector<T> dense = random_elements<T>(generator, n * n);

                    const Tensor band = band_of(
                        to_band(dense, n, n, n - 1, n - 1, ld, this->engine->layout(),
                                poison<T>()),
                        n, n, ld);
                    const Tensor full = dense_of(n, n, dense);
                    const Tensor x = vector_of(random_elements<T>(generator, n));
                    const std::vector<T> y = random_elements<T>(generator, n);
                    Tensor from_band = vector_of(y);
                    Tensor from_dense = vector_of(y);

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
                [[nodiscard]] std::array<std::vector<T>, 2> large_symmetric_products()
                {
                    const int64_t n = K_LARGE_ORDER;
                    std::mt19937 generator(K_SEED);
                    const std::vector<T> dense = random_elements<T>(generator, n * n);

                    const Tensor band = triangle_band_of(dense, n, Core::Uplo::Upper, n - 1);
                    const Tensor full = dense_of(n, n, dense);
                    const Tensor x = vector_of(random_elements<T>(generator, n));
                    const std::vector<T> y = random_elements<T>(generator, n);
                    Tensor from_band = vector_of(y);
                    Tensor from_dense = vector_of(y);

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

                    const Tensor band = band_of(
                        to_band(general, n, n, k, k, 2 * k + 1, layout, poison<T>()), n, n,
                        2 * k + 1);
                    const Tensor upper = triangle_band_of(general, n, Core::Uplo::Upper, k);
                    const Tensor lower = triangle_band_of(general, n, Core::Uplo::Lower, k);
                    const Tensor upper_dominant =
                        triangle_band_of(dominant, n, Core::Uplo::Upper, k);
                    const Tensor lower_dominant =
                        triangle_band_of(dominant, n, Core::Uplo::Lower, k);
                    const Tensor x_tensor = vector_of(x);

                    std::vector<Tensor> results;
                    for (int routine = 0; routine < 3; ++routine)
                        results.push_back(vector_of(y));
                    for (int routine = 0; routine < 2; ++routine)
                        results.push_back(vector_of(x));

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

        /** @brief Runs the band routines over the five element types. */
        template<typename T>
        class Medium_BlasBand : public BLASBandCases<T>
        {

        };

        /** @brief Runs the conjugating band cases over the complex element types. */
        template<typename T>
        class Medium_BlasBandComplex : public BLASBandCases<T>
        {

        };

        /// @brief Names the suite of the HALF accumulation cases.
        using Medium_BlasBandHalf = BLASBandCases<sycl::half>;

        /// @brief Names the suite of the FLOAT32 cases of order K_LARGE_ORDER.
        using Medium_BlasBandLarge = BLASBandCases<float>;

        /// @brief Names the five element types the band routines accept.
        using BLASBandTypes = ::testing::Types<sycl::half, float, double, std::complex<float>,
                                               std::complex<double>>;
        /// @brief Names the complex element types, the only ones a conjugation changes.
        using BLASBandComplexTypes = ::testing::Types<std::complex<float>, std::complex<double>>;
        SB_TYPED_TEST_SUITE(Medium_BlasBand, BLASBandTypes);
        SB_TYPED_TEST_SUITE(Medium_BlasBandComplex, BLASBandComplexTypes);

        /**
         * @brief Tests gbmv on the literal band arrays of the worked matrix.
         */
        TYPED_TEST(Medium_BlasBand, GbmvMatchesTheWorkedMatrix)
        {
            using T = TypeParam;

            this->for_each_engine([&](Core::Layout layout)
            {
                const std::vector<T> y = this->worked_gbmv(
                    TestFixture::worked_band(layout), K_WORKED_LD, Core::Transpose::NoTrans,
                    {1, 2, 3, 4, 5}, std::vector<T>(4, poison<T>()), 1.0, 0.0);

                expect_equal(y, elements<T>({74, 230, 474, 530}));
            });
        }

        /**
         * @brief Tests that gbmv scales the product by alpha and the prior y by beta.
         */
        TYPED_TEST(Medium_BlasBand, GbmvScalesAndAccumulates)
        {
            using T = TypeParam;

            this->for_each_engine([&](Core::Layout layout)
            {
                const std::vector<T> y = this->worked_gbmv(
                    TestFixture::worked_band(layout), K_WORKED_LD, Core::Transpose::NoTrans,
                    {1, 2, 3, 4, 5}, elements<T>({1, 1, 1, 1}), 2.0, -1.0);

                expect_equal(y, elements<T>({147, 459, 947, 1059}));
            });
        }

        /**
         * @brief Tests gbmv on the transpose of the worked matrix.
         */
        TYPED_TEST(Medium_BlasBand, GbmvTransposed)
        {
            using T = TypeParam;

            this->for_each_engine([&](Core::Layout layout)
            {
                const std::vector<T> y = this->worked_gbmv(
                    TestFixture::worked_band(layout), K_WORKED_LD, Core::Transpose::Trans,
                    {1, 2, 3, 4}, std::vector<T>(5, poison<T>()), 1.0, 0.0);

                expect_equal(y, elements<T>({53, 152, 330, 326, 285}));
            });
        }

        /**
         * @brief Tests that gbmv reads no slot past the band of a line.
         */
        TYPED_TEST(Medium_BlasBand, GbmvIgnoresSpareSlots)
        {
            using T = TypeParam;
            constexpr int64_t wide_ld = K_WORKED_LD + 2;

            this->for_each_engine([&](Core::Layout layout)
            {
                const std::vector<T> band =
                    to_band(elements<T>(K_WORKED_DENSE), K_WORKED_ROWS, K_WORKED_COLS,
                            K_WORKED_KL, K_WORKED_KU, wide_ld, layout, poison<T>());
                const std::vector<T> y = this->worked_gbmv(
                    band, wide_ld, Core::Transpose::NoTrans, {1, 2, 3, 4, 5},
                    std::vector<T>(4, poison<T>()), 1.0, 0.0);

                expect_equal(y, elements<T>({74, 230, 474, 530}));
            });
        }

        /**
         * @brief Tests gbmv on shapes at the edge of what it accepts against a host loop.
         */
        TYPED_TEST(Medium_BlasBand, GbmvEdgeShapes)
        {
            using T = TypeParam;

            this->for_each_engine([&](Core::Layout layout)
            {
                for (const BandShape& shape : K_EDGE_SHAPES)
                {
                    SCOPED_TRACE(::testing::Message() << shape.m << " x " << shape.n << ", kl "
                                                      << shape.kl << ", ku " << shape.ku);
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

                    const Tensor A = this->band_of(
                        to_band(elements<T>(dense), shape.m, shape.n, shape.kl, shape.ku, ld,
                                layout, poison<T>()),
                        shape.m, shape.n, ld);
                    const Tensor x_tensor = this->vector_of(elements<T>(x));
                    Tensor y = this->vector_of(
                        std::vector<T>(static_cast<std::size_t>(shape.m), poison<T>()));

                    SushiBLAS::BLASOps(this->recorder()).gbmv(A, x_tensor, y, shape.kl, shape.ku);
                    this->recording->run().value();

                    expect_equal(read_tensor<T>(y), elements<T>(expected));
                }
            });
        }

        /**
         * @brief Tests that gbmv on a matrix of no columns only scales y by beta.
         */
        TYPED_TEST(Medium_BlasBand, GbmvOfNoColumnsScalesY)
        {
            using T = TypeParam;

            this->for_each_engine([&](Core::Layout layout)
            {
                const bool by_columns = (layout == Core::Layout::ColumnMajor);
                const Tensor A = this->band_of(
                    std::vector<T>(by_columns ? std::size_t{0} : std::size_t{2}, poison<T>()),
                    2, 0, 1);
                const Tensor x = this->vector_of({});
                Tensor y = this->vector_of(elements<T>({1, 2}));

                SushiBLAS::BLASOps(this->recorder()).gbmv(A, x, y, 0, 0, Core::Transpose::NoTrans,
                                                       1.0, 3.0);
                this->recording->run().value();

                expect_equal(read_tensor<T>(y), elements<T>({3, 6}));
            });
        }

        /**
         * @brief Tests sbmv on the literal band arrays of a tridiagonal matrix, where a
         *        mistaken layout or triangle reads another matrix.
         */
        TYPED_TEST(Medium_BlasBand, SbmvMatchesTheTridiagonal)
        {
            using T = TypeParam;
            const int64_t n = K_TRIDIAGONAL_ORDER;

            this->for_each_engine([&](Core::Layout layout)
            {
                for (const Core::Uplo uplo : K_TRIANGLES)
                {
                    SCOPED_TRACE(Core::to_string(uplo));
                    const bool diagonal_last = (uplo == Core::Uplo::Upper) ==
                                               (layout == Core::Layout::ColumnMajor);
                    const Tensor A = this->band_of(
                        elements<T>(diagonal_last ? K_TRIDIAGONAL_DIAGONAL_LAST
                                                  : K_TRIDIAGONAL_DIAGONAL_FIRST),
                        n, n, 2);
                    const Tensor x = this->vector_of(elements<T>({1, 2, 3, 4}));
                    Tensor y = this->vector_of(std::vector<T>(4, poison<T>()));

                    SushiBLAS::BLASOps(this->recorder()).sbmv(A, x, y, uplo, 1);
                    this->recording->run().value();

                    expect_equal(read_tensor<T>(y), elements<T>({5, 20, 47, 46}));
                }
            });
        }

        /**
         * @brief Tests tbmv on both hand triangles, transposed and with a unit diagonal.
         */
        TYPED_TEST(Medium_BlasBand, TbmvMatchesByHand)
        {
            using T = TypeParam;

            this->for_each_engine([&](Core::Layout)
            {
                for (const TriangularRow& row : K_TRIANGULAR_ROWS)
                {
                    SCOPED_TRACE(flags_name(row.uplo, row.trans, row.diag));
                    const Tensor A = this->hand_triangle(row.uplo);
                    Tensor x = this->vector_of(elements<T>({1, 2, 3}));

                    SushiBLAS::BLASOps(this->recorder()).tbmv(A, x, row.uplo, 1, row.trans,
                                                           row.diag);
                    this->recording->run().value();

                    expect_equal(read_tensor<T>(x),
                                 elements<T>({row.product[0], row.product[1], row.product[2]}));
                }
            });
        }

        /**
         * @brief Tests that tbsv recovers [1, 2, 3] from each product of TbmvMatchesByHand.
         */
        TYPED_TEST(Medium_BlasBand, TbsvInvertsTbmv)
        {
            using T = TypeParam;

            this->for_each_engine([&](Core::Layout)
            {
                for (const TriangularRow& row : K_TRIANGULAR_ROWS)
                {
                    SCOPED_TRACE(flags_name(row.uplo, row.trans, row.diag));
                    const Tensor A = this->hand_triangle(row.uplo);
                    Tensor b = this->vector_of(
                        elements<T>({row.product[0], row.product[1], row.product[2]}));

                    SushiBLAS::BLASOps(this->recorder()).tbsv(A, b, row.uplo, 1, row.trans,
                                                           row.diag);
                    this->recording->run().value();

                    expect_equal(read_tensor<T>(b), elements<T>({1, 2, 3}));
                }
            });
        }

        /**
         * @brief Tests that tbsv solves in a view of every second element and leaves the
         *        elements between as they were.
         */
        TYPED_TEST(Medium_BlasBand, TbsvTakesAStridedVector)
        {
            using T = TypeParam;

            this->for_each_engine([&](Core::Layout)
            {
                const Tensor A = this->hand_triangle(Core::Uplo::Upper);
                Tensor parent = this->vector_of(
                    elements<T>({5, K_POISON, 18, K_POISON, 15, K_POISON}));
                Tensor b = TestFixture::every_second_element(parent);

                SushiBLAS::BLASOps(this->recorder()).tbsv(A, b, Core::Uplo::Upper, 1);
                this->recording->run().value();

                expect_equal(read_tensor<T>(parent),
                             elements<T>({1, K_POISON, 2, K_POISON, 3, K_POISON}));
            });
        }

        /**
         * @brief Tests that a band wide enough to hold the whole matrix gives the elements of
         *        gemv, symv and hemv.
         */
        TYPED_TEST(Medium_BlasBand, AFullBandEqualsTheDenseSibling)
        {
            using T = TypeParam;

            this->for_each_engine([&](Core::Layout layout)
            {
                std::mt19937 generator(K_SEED);
                SushiBLAS::BLASOps ops(this->recorder());

                const int64_t m = K_SIBLING_ROWS;
                const int64_t n = K_SIBLING_COLS;
                const int64_t ld = m + n - 1;
                const std::vector<T> general = random_elements<T>(generator, m * n);
                const Tensor band = this->band_of(
                    to_band(general, m, n, m - 1, n - 1, ld, layout, poison<T>()), m, n, ld);
                const Tensor full = this->dense_of(m, n, general);

                for (const Core::Transpose trans : K_TRANSPOSES)
                {
                    SCOPED_TRACE(std::string("gbmv, ") + Core::to_string(trans));
                    const bool transposed = Core::is_transposed(trans);
                    const Tensor x =
                        this->vector_of(random_elements<T>(generator, transposed ? m : n));
                    const std::vector<T> y = random_elements<T>(generator, transposed ? n : m);

                    this->expect_same_result(
                        y,
                        [&](Tensor& out)
                        {
                            ops.gbmv(band, x, out, m - 1, n - 1, trans, K_ALPHA, K_BETA);
                        },
                        [&](Tensor& out) { ops.gemv(full, x, out, trans, K_ALPHA, K_BETA); });
                }

                const int64_t order = K_SIBLING_ORDER;
                const std::vector<T> square = random_elements<T>(generator, order * order);
                const Tensor square_full = this->dense_of(order, order, square);

                for (const Core::Uplo uplo : K_TRIANGLES)
                {
                    SCOPED_TRACE(Core::to_string(uplo));
                    const Tensor triangle = this->triangle_band_of(square, order, uplo, order - 1);
                    const Tensor x = this->vector_of(random_elements<T>(generator, order));
                    const std::vector<T> y = random_elements<T>(generator, order);

                    this->expect_same_result(
                        y,
                        [&](Tensor& out)
                        {
                            ops.sbmv(triangle, x, out, uplo, order - 1, K_ALPHA, K_BETA);
                        },
                        [&](Tensor& out)
                        {
                            ops.symv(square_full, x, out, uplo, K_ALPHA, K_BETA);
                        });
                    this->expect_same_result(
                        y,
                        [&](Tensor& out)
                        {
                            ops.hbmv(triangle, x, out, uplo, order - 1, K_ALPHA, K_BETA);
                        },
                        [&](Tensor& out)
                        {
                            ops.hemv(square_full, x, out, uplo, K_ALPHA, K_BETA);
                        });
                }
            });
        }

        /**
         * @brief Tests that tbmv and tbsv on a full triangle band give the elements of trmv and
         *        trsv, and the product and the solution formed in integers on the host.
         */
        TYPED_TEST(Medium_BlasBand, AFullTriangleBandEqualsTheDenseSibling)
        {
            using T = TypeParam;
            const int64_t n = K_SIBLING_ORDER;

            this->for_each_engine([&](Core::Layout)
            {
                SushiBLAS::BLASOps ops(this->recorder());

                for (const Core::Uplo uplo : K_TRIANGLES)
                {
                    for (const Core::Transpose trans : K_TRANSPOSES)
                    {
                        for (const Core::Diag diag : K_DIAGONALS)
                        {
                            SCOPED_TRACE(flags_name(uplo, trans, diag));
                            const ExactTriangle exact =
                                exact_triangle(n, uplo, trans, diag, Core::is_complex_v<T>);
                            const std::vector<T> stored = narrowed_all<T>(exact.stored);
                            const std::vector<T> x = narrowed_all<T>(exact.x);
                            const std::vector<T> b = narrowed_all<T>(exact.b);

                            const Tensor band = this->triangle_band_of(stored, n, uplo, n - 1);
                            const Tensor full = this->dense_of(n, n, stored);

                            const std::vector<T> product = this->expect_same_result(
                                x,
                                [&](Tensor& out) { ops.tbmv(band, out, uplo, n - 1, trans, diag); },
                                [&](Tensor& out) { ops.trmv(full, out, uplo, trans, diag); });
                            expect_equal(product, b);

                            const std::vector<T> solution = this->expect_same_result(
                                b,
                                [&](Tensor& out) { ops.tbsv(band, out, uplo, n - 1, trans, diag); },
                                [&](Tensor& out) { ops.trsv(full, out, uplo, trans, diag); });
                            expect_equal(solution, x);
                        }
                    }
                }
            });
        }

        /**
         * @brief Tests that the two kernels agree within ModeTolerance at four band widths.
         */
        TYPED_TEST(Medium_BlasBand, BothModesAgree)
        {
            using T = TypeParam;

            for (const Core::Layout layout : K_LAYOUTS)
            {
                for (const int64_t k : K_MODE_WIDTHS)
                {
                    SCOPED_TRACE(::testing::Message()
                                 << engine_name(layout, ExecutionMode::ForceSYCL) << ", k " << k);

                    this->start(layout, ExecutionMode::ForceReference);
                    const std::vector<T> reference = this->banded_results(k);
                    this->start(layout, ExecutionMode::ForceSYCL);
                    const std::vector<T> device = this->banded_results(k);

                    expect_near(device, reference);
                }
            }
        }

        /**
         * @brief Tests gbmv under each transpose on a complex band matrix.
         */
        TYPED_TEST(Medium_BlasBandComplex, GbmvConjTransposed)
        {
            using T = TypeParam;
            const std::vector<Wide> dense = {Wide(1, 2), Wide(3, -1), Wide(0, 0), Wide(4, 1)};
            const std::vector<Wide> x = {Wide(1, 0), Wide(0, 1)};
            const std::vector<Wide> expected[] = {{Wide(2, 5), Wide(-1, 4)},
                                                  {Wide(1, 2), Wide(2, 3)},
                                                  {Wide(1, -2), Wide(4, 5)}};

            this->for_each_engine([&](Core::Layout layout)
            {
                const Tensor A = this->band_of(
                    to_band(narrowed_all<T>(dense), 2, 2, 0, 1, 2, layout, poison<T>()), 2, 2, 2);
                const Tensor x_tensor = this->vector_of(narrowed_all<T>(x));

                for (std::size_t form = 0; form < 3; ++form)
                {
                    SCOPED_TRACE(Core::to_string(K_TRANSPOSES[form]));
                    Tensor y = this->vector_of(std::vector<T>(2, poison<T>()));

                    SushiBLAS::BLASOps(this->recorder()).gbmv(A, x_tensor, y, 0, 1,
                                                           K_TRANSPOSES[form]);
                    this->recording->run().value();

                    expect_equal(read_tensor<T>(y), narrowed_all<T>(expected[form]));
                }
            });
        }

        /**
         * @brief Tests that hbmv conjugates the mirrored triangle and reads no imaginary
         *        diagonal, and that sbmv on the same storage mirrors without conjugating.
         */
        TYPED_TEST(Medium_BlasBandComplex, HbmvConjugatesTheMirror)
        {
            using T = TypeParam;
            const int64_t n = K_HAND_ORDER;
            const std::vector<Wide> hermitian = {
                Wide(1, K_DIAGONAL_IMAGINARY), Wide(2, 1),  Wide(0, 0),
                Wide(2, -1), Wide(3, K_DIAGONAL_IMAGINARY), Wide(4, -2),
                Wide(0, 0),  Wide(4, 2), Wide(5, K_DIAGONAL_IMAGINARY)};
            const std::vector<Wide> real_diagonal = {Wide(1, 0), Wide(2, 1),  Wide(0, 0),
                                                     Wide(2, -1), Wide(3, 0), Wide(4, -2),
                                                     Wide(0, 0),  Wide(4, 2), Wide(5, 0)};
            const std::vector<Wide> x = {Wide(1, 0), Wide(0, 1), Wide(2, 0)};

            this->for_each_engine([&](Core::Layout)
            {
                const Tensor x_tensor = this->vector_of(narrowed_all<T>(x));

                for (const Core::Uplo uplo : K_TRIANGLES)
                {
                    SCOPED_TRACE(Core::to_string(uplo));
                    const Tensor A =
                        this->triangle_band_of(narrowed_all<T>(hermitian), n, uplo, 1);
                    Tensor y = this->vector_of(std::vector<T>(3, poison<T>()));

                    SushiBLAS::BLASOps(this->recorder()).hbmv(A, x_tensor, y, uplo, 1);
                    this->recording->run().value();

                    expect_equal(read_tensor<T>(y),
                                 narrowed_all<T>({Wide(0, 2), Wide(10, -2), Wide(8, 4)}));
                }

                const Tensor A = this->triangle_band_of(narrowed_all<T>(real_diagonal), n,
                                                        Core::Uplo::Upper, 1);
                Tensor y = this->vector_of(std::vector<T>(3, poison<T>()));

                SushiBLAS::BLASOps(this->recorder()).sbmv(A, x_tensor, y, Core::Uplo::Upper, 1);
                this->recording->run().value();

                expect_equal(read_tensor<T>(y),
                             narrowed_all<T>({Wide(0, 2), Wide(10, 0), Wide(12, 4)}));
            });
        }

        /**
         * @brief Tests that gbmv, sbmv and tbmv sum a HALF row in FLOAT32: 2048 + 1 + 1 - 2048
         *        is 2 there and 0 in HALF.
         */
        TEST_F(Medium_BlasBandHalf, HalfSumsInFloat)
        {
            using T = sycl::half;
            constexpr int64_t n = 4;
            const std::vector<double> row = {2048, 1, 1, -2048};
            std::vector<double> square(static_cast<std::size_t>(n * n), 0.0);
            std::copy(row.begin(), row.end(), square.begin());

            for_each_engine([&](Core::Layout layout)
            {
                SushiBLAS::BLASOps ops(recorder());
                const Tensor ones = vector_of(elements<T>({1, 1, 1, 1}));

                const Tensor general = band_of(
                    to_band(elements<T>(row), 1, n, 0, n - 1, n, layout, poison<T>()), 1, n, n);
                Tensor from_gbmv = vector_of(std::vector<T>(1, poison<T>()));
                ops.gbmv(general, ones, from_gbmv, 0, n - 1);

                const Tensor triangle =
                    triangle_band_of(elements<T>(square), n, Core::Uplo::Upper, n - 1);
                Tensor from_sbmv = vector_of(std::vector<T>(4, poison<T>()));
                ops.sbmv(triangle, ones, from_sbmv, Core::Uplo::Upper, n - 1);

                Tensor from_tbmv = vector_of(elements<T>({1, 1, 1, 1}));
                ops.tbmv(triangle, from_tbmv, Core::Uplo::Upper, n - 1);
                recording->run().value();

                expect_equal(read_tensor<T>(from_gbmv), elements<T>({2}));
                expect_equal(read_tensor<T>(from_sbmv), elements<T>({2, 1, 1, -2048}));
                expect_equal(read_tensor<T>(from_tbmv), elements<T>({2, 0, 0, 0}));
            });
        }

        /**
         * @brief Tests that tbsv carries a HALF solution in FLOAT32: the middle element is
         *        2049 there, stored as 2048, and the last is 2050 - 2049.
         */
        TEST_F(Medium_BlasBandHalf, HalfSolveCarriesTheSolutionInFloat)
        {
            using T = sycl::half;
            const std::vector<double> lower = {1, 0, 0, -1, 1, 0, 0, 1, 1};

            for_each_engine([&](Core::Layout)
            {
                const Tensor A =
                    triangle_band_of(elements<T>(lower), K_HAND_ORDER, Core::Uplo::Lower, 1);
                Tensor b = vector_of(elements<T>({2048, 1, 2050}));

                SushiBLAS::BLASOps(recorder()).tbsv(A, b, Core::Uplo::Lower, 1);
                recording->run().value();

                expect_equal(read_tensor<T>(b), elements<T>({2048, 2048, 1}));
            });
        }

        /**
         * @brief Tests gbmv and sbmv against gemv and symv at an order where a device that
         *        plans a work-group per row runs one.
         */
        TEST_F(Medium_BlasBandLarge, AFullBandEqualsTheDenseSiblingLarge)
        {
            for (const ExecutionMode mode : K_MODES)
            {
                SCOPED_TRACE(engine_name(Core::Layout::RowMajor, mode));
                start(Core::Layout::RowMajor, mode);
                const std::array<std::vector<float>, 2> general = large_general_products();
                expect_equal(general[0], general[1]);
            }

            for_each_engine([&](Core::Layout)
            {
                const std::array<std::vector<float>, 2> symmetric = large_symmetric_products();
                expect_equal(symmetric[0], symmetric[1]);
            });
        }

        /**
         * @brief Tests that gbmv and sbmv give the same bits when the device runs them twice.
         */
        TEST_F(Medium_BlasBandLarge, RepeatsToTheBit)
        {
            const auto same_bits = [](const std::vector<float>& first,
                                      const std::vector<float>& second)
            {
                return first.size() == second.size() &&
                       std::memcmp(first.data(), second.data(),
                                   first.size() * sizeof(float)) == 0;
            };

            start(Core::Layout::RowMajor, ExecutionMode::ForceSYCL);
            const std::vector<float> general = large_general_products()[0];
            EXPECT_TRUE(same_bits(general, large_general_products()[0])) << "gbmv";

            for (const Core::Layout layout : K_LAYOUTS)
            {
                SCOPED_TRACE(engine_name(layout, ExecutionMode::ForceSYCL));
                start(layout, ExecutionMode::ForceSYCL);
                const std::vector<float> symmetric = large_symmetric_products()[0];
                EXPECT_TRUE(same_bits(symmetric, large_symmetric_products()[0])) << "sbmv";
            }
        }
    } // namespace Test
} // namespace SushiBLAS
