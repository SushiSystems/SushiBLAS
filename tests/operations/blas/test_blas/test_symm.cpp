/**************************************************************************/
/* test_symm.cpp                                                          */
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
 * @file test_symm.cpp
 * @brief Tests symm and hemm against hand-computed products, gemm on the full matrix and the
 *        device-only operand refusal; the record-time refusals are in test_symm_refusals.cpp.
 * @author Mustafa Garip
 */

#include <cmath>
#include <limits>
#include <random>
#include <string>
#include <vector>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <initializer_list>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <SushiBLAS/SushiBLAS.hpp>
#include <SushiBLAS/core/errors.hpp>
#include <SushiBLAS/core/traits.hpp>
#include <SushiBLAS/arithmetic/complex_arithmetic.hpp>
#include <SushiBLAS/engine_fixture/sushiblas_test.hpp>
#include <SushiBLAS/tensor_expectations/tensor_readback.hpp>
#include <SushiBLAS/tensor_expectations/scalar_dtype.hpp>
#include <SushiBLAS/test_runner/test_environment.hpp>
#include "blas_rows.hpp"

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            /// @brief Rows of B and C in the multi-tile cases.
            constexpr int64_t K_ROWS = 70;

            /// @brief Columns of B and C in the multi-tile cases.
            constexpr int64_t K_COLS = 37;

            /// @brief Matrices in the batched case.
            constexpr int64_t K_BATCH = 4;

            /// @brief Rows of one matrix of B and C in the batched case.
            constexpr int64_t K_BATCH_ROWS = 5;

            /// @brief Columns of one matrix of B and C in the batched case.
            constexpr int64_t K_BATCH_COLS = 3;

            /// @brief Seed of the random operands of this file.
            constexpr std::uint32_t K_SEED = 20261007u;

            /// @brief Scales the product in the cases that compare against gemm.
            constexpr double K_ALPHA = 1.5;

            /// @brief Scales the prior C in the cases that compare against gemm.
            constexpr double K_BETA = 0.5;

            /// @brief Imaginary part given to the stored diagonal of a Hermitian A.
            constexpr double K_DIAGONAL_IMAGINARY = 7.0;

            /// @brief The two kernels every case runs on.
            const std::vector<ExecutionMode> K_BOTH_MODES = {ExecutionMode::ForceSYCL,
                                                             ExecutionMode::ForceReference};

            /// @brief The element types symm accepts that the helpers of this file instantiate.
            const std::vector<Core::DataType> K_SYMM_TYPES = {
                Core::DataType::Float32, Core::DataType::Float64, Core::DataType::Complex32,
                Core::DataType::Complex64};

            /// @brief The two sides A can stand on.
            const std::vector<Core::Side> K_SIDES = {Core::Side::Left, Core::Side::Right};

            /// @brief The two triangles A can be stored in.
            const std::vector<Core::Uplo> K_TRIANGLES = {Core::Uplo::Upper, Core::Uplo::Lower};

            /** @brief Names the routine a shared case calls. */
            enum class Routine : std::uint8_t
            {
                Symm, /**< The symmetric product. */
                Hemm  /**< The Hermitian product. */
            };

            /** @brief Names the side and the triangle of one product. */
            struct SideTriangle
            {
                Core::Side side; /**< The side A stands on. */
                Core::Uplo uplo; /**< The triangle A is stored in. */
            };

            /** @brief Returns the tail of a row named by a side and a triangle. */
            [[nodiscard]] std::string axis_name(const SideTriangle& form)
            {
                return "_" + word(form.side) + "_" + word(form.uplo);
            }

            /** @brief Names the side of a batched product and whether A carries the batch axis. */
            struct BatchForm
            {
                Core::Side side;  /**< The side A stands on. */
                bool a_is_batched; /**< True when A has one matrix per product. */
            };

            /** @brief Returns the tail of a row named by a batched form. */
            [[nodiscard]] std::string axis_name(const BatchForm& form)
            {
                return "_" + word(form.side) + (form.a_is_batched ? "_ABatched" : "_ARank2");
            }

            /** @brief Names an empty product by the extents of B and C and the side. */
            struct EmptyForm
            {
                int64_t rows;    /**< Rows of B and C. */
                int64_t cols;    /**< Columns of B and C. */
                Core::Side side; /**< The side A stands on. */
            };

            /** @brief Returns the tail of a row named by an empty product. */
            [[nodiscard]] std::string axis_name(const EmptyForm& form)
            {
                return "_" + std::to_string(form.rows) + "x" + std::to_string(form.cols) + "_" +
                       word(form.side);
            }

            /** @brief Returns the rows of a suite that run in the row-major layout alone. */
            template<typename Axis>
            [[nodiscard]] std::vector<Row<Axis>> row_major(const std::vector<Row<Axis>>& rows)
            {
                std::vector<Row<Axis>> kept;
                for (const Row<Axis>& row : rows)
                    if (row.layout == Core::Layout::RowMajor)
                        kept.push_back(row);
                return kept;
            }

            /** @brief Returns the side and triangle pairs, side first. */
            [[nodiscard]] std::vector<SideTriangle> side_triangles()
            {
                std::vector<SideTriangle> forms;
                for (const Core::Side side : K_SIDES)
                    for (const Core::Uplo uplo : K_TRIANGLES)
                        forms.push_back({side, uplo});
                return forms;
            }

            /** @brief Returns the batched forms: each side with A batched and with A of rank 2. */
            [[nodiscard]] std::vector<BatchForm> batch_forms()
            {
                std::vector<BatchForm> forms;
                for (const Core::Side side : K_SIDES)
                    for (const bool a_is_batched : {true, false})
                        forms.push_back({side, a_is_batched});
                return forms;
            }

            /** @brief Returns the empty products: B and C of no row or of no column, each side. */
            [[nodiscard]] std::vector<EmptyForm> empty_forms()
            {
                std::vector<EmptyForm> forms;
                for (const Core::Side side : K_SIDES)
                {
                    forms.push_back({0, 3, side});
                    forms.push_back({3, 0, side});
                }
                return forms;
            }

            /** @brief Returns the value that marks an element no kernel may read. */
            template<typename T>
            [[nodiscard]] T poison() noexcept
            {
                if constexpr (Core::is_complex_v<T>)
                {
                    using Real = typename T::value_type;
                    return T(std::numeric_limits<Real>::quiet_NaN(),
                             std::numeric_limits<Real>::quiet_NaN());
                }
                else
                    return static_cast<T>(std::numeric_limits<float>::quiet_NaN());
            }

            /** @brief Returns whether either part of value is NaN. */
            template<typename T>
            [[nodiscard]] bool is_nan(T value) noexcept
            {
                if constexpr (Core::is_complex_v<T>)
                    return std::isnan(value.real()) || std::isnan(value.imag());
                else
                    return std::isnan(static_cast<double>(value));
            }

            /** @brief Returns value with its imaginary part replaced; a real value is unchanged. */
            template<typename T>
            [[nodiscard]] T with_imaginary(T value, [[maybe_unused]] double imaginary) noexcept
            {
                if constexpr (Core::is_complex_v<T>)
                    return T(value.real(), static_cast<typename T::value_type>(imaginary));
                else
                    return value;
            }

            /** @brief Returns values as elements of T, a NaN becoming poison. */
            template<typename T>
            [[nodiscard]] std::vector<T> elements(std::initializer_list<double> values)
            {
                std::vector<T> out;
                out.reserve(values.size());
                for (const double value : values)
                    out.push_back(std::isnan(value) ? poison<T>() : static_cast<T>(
                        static_cast<Kernels::RealType<T>>(value)));
                return out;
            }

            /** @brief Returns count elements drawn uniformly from [-1, 1] in each part. */
            template<typename T>
            [[nodiscard]] std::vector<T> random_elements(std::mt19937& generator, int64_t count)
            {
                using Real = Kernels::RealType<T>;
                std::uniform_real_distribution<Real> distribution(Real(-1), Real(1));
                std::vector<T> out(static_cast<std::size_t>(count));
                for (T& element : out)
                {
                    if constexpr (Core::is_complex_v<T>)
                    {
                        const Real real = distribution(generator);
                        const Real imaginary = distribution(generator);
                        element = T(real, imaginary);
                    }
                    else
                        element = distribution(generator);
                }
                return out;
            }

            /** @brief Returns count integers from -3 to 3, Gaussian integers for a complex T. */
            template<typename T>
            [[nodiscard]] std::vector<T> integer_elements(std::size_t seed, int64_t count)
            {
                using Real = Kernels::RealType<T>;
                std::vector<T> out(static_cast<std::size_t>(count));
                for (std::size_t i = 0; i < out.size(); ++i)
                {
                    const Real real = static_cast<Real>(
                        static_cast<int64_t>(((i + seed) * 37u) % 7u) - 3);
                    const Real imaginary = static_cast<Real>(
                        static_cast<int64_t>(((i + seed) * 53u + 2u) % 7u) - 3);
                    out[i] = with_imaginary(T(real), static_cast<double>(imaginary));
                }
                return out;
            }

            /** @brief Returns whether (r, c) lies in the stored triangle. */
            [[nodiscard]] constexpr bool is_stored(Core::Uplo uplo, int64_t r, int64_t c) noexcept
            {
                return uplo == Core::Uplo::Upper ? r <= c : r >= c;
            }

            /**
             * @brief Returns the full row-major matrix a routine reads from one triangle of a.
             * @param a Row-major, of the given order; only its stored triangle is read.
             */
            template<typename T>
            [[nodiscard]] std::vector<T> mirrored(const std::vector<T>& a, int64_t order,
                                                  Core::Uplo uplo, Routine routine)
            {
                const bool hermitian = routine == Routine::Hemm;
                std::vector<T> full(a.size());
                for (int64_t r = 0; r < order; ++r)
                {
                    for (int64_t c = 0; c < order; ++c)
                    {
                        const bool stored = is_stored(uplo, r, c);
                        const T value = a[static_cast<std::size_t>(stored ? r * order + c
                                                                          : c * order + r)];
                        T& out = full[static_cast<std::size_t>(r * order + c)];
                        if (!hermitian)
                            out = value;
                        else if (r == c)
                            out = Kernels::real_part(value);
                        else
                            out = stored ? value : Kernels::conjugate(value);
                    }
                }
                return full;
            }

            /** @brief Poisons the triangle of a row-major a that a routine must not read. */
            template<typename T>
            void poison_unread_triangle(std::vector<T>& a, int64_t order, Core::Uplo uplo)
            {
                for (int64_t r = 0; r < order; ++r)
                {
                    for (int64_t c = 0; c < order; ++c)
                    {
                        if (!is_stored(uplo, r, c))
                            a[static_cast<std::size_t>(r * order + c)] = poison<T>();
                    }
                }
            }

            /** @brief Returns a tensor of dims that holds values. */
            template<typename T>
            [[nodiscard]] Tensor upload(Engine& engine, std::initializer_list<int64_t> dims,
                                        const std::vector<T>& values)
            {
                Tensor tensor = engine.create_tensor(dims, sushiblas_dtype<T>());
                EXPECT_EQ(values.size(), static_cast<std::size_t>(tensor.num_elements));
                TestEnvironment::queue()
                    .memcpy(tensor.data(), values.data(), values.size() * sizeof(T)).wait();
                return tensor;
            }

            /** @brief Returns the count elements of values that hold matrix index of a batch. */
            template<typename T>
            [[nodiscard]] std::vector<T> matrix_of(const std::vector<T>& values, int64_t index,
                                                   int64_t count)
            {
                const auto first = values.begin() + static_cast<std::ptrdiff_t>(index * count);
                return std::vector<T>(first, first + static_cast<std::ptrdiff_t>(count));
            }

            /** @brief Expects two results to hold the same bytes. */
            template<typename T>
            void expect_same_bytes(const std::vector<T>& actual, const std::vector<T>& expected)
            {
                ASSERT_EQ(actual.size(), expected.size());
                EXPECT_EQ(0, std::memcmp(actual.data(), expected.data(),
                                         actual.size() * sizeof(T)));
            }

            /** @brief Expects every element to equal its expectation with no tolerance. */
            template<typename T>
            void expect_equal(const std::vector<T>& actual, const std::vector<T>& expected)
            {
                ASSERT_EQ(actual.size(), expected.size());
                for (std::size_t i = 0; i < actual.size(); ++i)
                    EXPECT_EQ(actual[i], expected[i]) << "index " << i;
            }

            /** @brief Records the product of a routine on recorder. */
            TaskHandle record_product(TaskRecorder& recorder, Routine routine, const Tensor& A,
                                      const Tensor& B, Tensor& C, Core::Side side,
                                      Core::Uplo uplo, BLASScalar alpha, BLASScalar beta)
            {
                SushiBLAS::BLASOps ops(recorder);
                return routine == Routine::Hemm ? ops.hemm(A, B, C, side, uplo, alpha, beta)
                                                : ops.symm(A, B, C, side, uplo, alpha, beta);
            }

            /**
             * @brief Expects a routine to leave the bytes gemm leaves on the mirrored full matrix.
             * @param form The side and the triangle of the product.
             */
            template<typename T>
            void expect_product_equals_gemm(Engine& engine, RecordingFixture& recording,
                                            Routine routine, const SideTriangle& form)
            {
                std::mt19937 generator(K_SEED);
                const bool left = form.side == Core::Side::Left;
                const int64_t order = left ? K_ROWS : K_COLS;
                std::vector<T> a = random_elements<T>(generator, order * order);
                const std::vector<T> b = random_elements<T>(generator, K_ROWS * K_COLS);
                const std::vector<T> c = random_elements<T>(generator, K_ROWS * K_COLS);

                if (routine == Routine::Hemm)
                {
                    for (int64_t d = 0; d < order; ++d)
                    {
                        T& diagonal = a[static_cast<std::size_t>(d * order + d)];
                        diagonal = with_imaginary(diagonal, K_DIAGONAL_IMAGINARY);
                    }
                }
                const std::vector<T> full = mirrored(a, order, form.uplo, routine);
                poison_unread_triangle(a, order, form.uplo);

                const Tensor A = upload(engine, {order, order}, a);
                const Tensor S = upload(engine, {order, order}, full);
                const Tensor B = upload(engine, {K_ROWS, K_COLS}, b);
                Tensor from_routine = upload(engine, {K_ROWS, K_COLS}, c);
                Tensor from_gemm = upload(engine, {K_ROWS, K_COLS}, c);

                TaskRecorder& recorder = recording.recorder();
                record_product(recorder, routine, A, B, from_routine, form.side, form.uplo,
                               K_ALPHA, K_BETA);
                SushiBLAS::BLASOps(recorder).gemm(left ? S : B, left ? B : S, from_gemm,
                                                  Core::Transpose::NoTrans,
                                                  Core::Transpose::NoTrans, K_ALPHA, K_BETA);
                recording.run().value();

                expect_same_bytes(read_tensor<T>(from_routine), read_tensor<T>(from_gemm));
            }

            /**
             * @brief Expects a batched symm to leave, for each matrix, the bytes the same call on
             *        that matrix alone leaves.
             */
            template<typename T>
            void expect_batch_matches_each_matrix(Engine& engine, RecordingFixture& recording,
                                                  const BatchForm& form)
            {
                const int64_t matrix = K_BATCH_ROWS * K_BATCH_COLS;
                const int64_t order =
                    form.side == Core::Side::Left ? K_BATCH_ROWS : K_BATCH_COLS;
                const std::vector<T> a = integer_elements<T>(0, K_BATCH * order * order);
                const std::vector<T> b = integer_elements<T>(11, K_BATCH * matrix);
                const std::vector<T> c = integer_elements<T>(23, K_BATCH * matrix);

                const Tensor A = form.a_is_batched
                    ? upload(engine, {K_BATCH, order, order}, a)
                    : upload(engine, {order, order}, matrix_of(a, 0, order * order));
                const Tensor B = upload(engine, {K_BATCH, K_BATCH_ROWS, K_BATCH_COLS}, b);
                Tensor C = upload(engine, {K_BATCH, K_BATCH_ROWS, K_BATCH_COLS}, c);
                SushiBLAS::BLASOps(recording.recorder()).symm(A, B, C, form.side,
                                                              Core::Uplo::Upper, 2.0, -1.0);
                recording.run().value();
                const std::vector<T> batched = read_tensor<T>(C);

                for (int64_t index = 0; index < K_BATCH; ++index)
                {
                    SCOPED_TRACE("matrix " + std::to_string(index));
                    const Tensor one_a = upload(
                        engine, {order, order},
                        matrix_of(a, form.a_is_batched ? index : 0, order * order));
                    const Tensor one_b = upload(engine, {K_BATCH_ROWS, K_BATCH_COLS},
                                                matrix_of(b, index, matrix));
                    Tensor one_c = upload(engine, {K_BATCH_ROWS, K_BATCH_COLS},
                                          matrix_of(c, index, matrix));
                    SushiBLAS::BLASOps(recording.recorder()).symm(one_a, one_b, one_c, form.side,
                                                                  Core::Uplo::Upper, 2.0, -1.0);
                    recording.run().value();

                    expect_equal(matrix_of(batched, index, matrix), read_tensor<T>(one_c));
                }
            }

            /**
             * @brief Expects symm with a zero beta to give the same bytes over a poisoned C as
             *        over a zeroed one, and no NaN.
             */
            template<typename T>
            void expect_zero_beta_ignores_c(Engine& engine, RecordingFixture& recording,
                                            Core::Side side)
            {
                std::mt19937 generator(K_SEED);
                const int64_t order = side == Core::Side::Left ? K_ROWS : K_COLS;
                const std::vector<T> a = random_elements<T>(generator, order * order);
                const std::vector<T> b = random_elements<T>(generator, K_ROWS * K_COLS);
                const std::size_t size = static_cast<std::size_t>(K_ROWS * K_COLS);

                const Tensor A = upload(engine, {order, order}, a);
                const Tensor B = upload(engine, {K_ROWS, K_COLS}, b);
                Tensor over_poison = upload(engine, {K_ROWS, K_COLS},
                                            std::vector<T>(size, poison<T>()));
                Tensor over_zero = upload(engine, {K_ROWS, K_COLS}, std::vector<T>(size, T{}));

                SushiBLAS::BLASOps ops(recording.recorder());
                ops.symm(A, B, over_poison, side, Core::Uplo::Lower, K_ALPHA, 0.0);
                ops.symm(A, B, over_zero, side, Core::Uplo::Lower, K_ALPHA, 0.0);
                recording.run().value();

                const std::vector<T> result = read_tensor<T>(over_poison);
                for (std::size_t i = 0; i < result.size(); ++i)
                    EXPECT_FALSE(is_nan(result[i])) << "index " << i;
                expect_same_bytes(result, read_tensor<T>(over_zero));
            }
        } // namespace

        /** @brief Holds the suites that run one case per kernel, in the row-major layout. */
        template<typename Axis>
        using SymmCase = BlasRowCase<BlasEngineStart, Axis>;

        /** @brief Runs the hand-computed symm products over the element types. */
        class Medium_Symm : public SymmCase<NoAxis>
        {

        };

        TEST_P(Medium_Symm, LeftSideMatchesAHandComputedProduct)
        {
            with_non_half_element(GetParam().type, [&](auto element)
            {
                using T = typename decltype(element)::type;
                const double unread = std::numeric_limits<double>::quiet_NaN();
                const std::vector<T> a = elements<T>({1, 2, 3,
                                                      unread, 4, 5,
                                                      unread, unread, 6});
                const std::vector<T> b = elements<T>({1, 2, 3, 4, 5, 6});
                const std::vector<T> ones = elements<T>({1, 1, 1, 1, 1, 1});

                const Tensor A = upload(*this->engine, {3, 3}, a);
                const Tensor B = upload(*this->engine, {3, 2}, b);
                Tensor C = upload(*this->engine, {3, 2}, ones);

                SushiBLAS::BLASOps(this->recorder()).symm(A, B, C, Core::Side::Left,
                                                          Core::Uplo::Upper, 2.0, -1.0);
                this->recording->run().value();

                // A * B is {22 28, 39 50, 48 62}; C is twice that minus one.
                expect_equal(read_tensor<T>(C), elements<T>({43, 55, 77, 99, 95, 123}));
            });
        }

        TEST_P(Medium_Symm, RightSideMatchesAHandComputedProduct)
        {
            with_non_half_element(GetParam().type, [&](auto element)
            {
                using T = typename decltype(element)::type;
                const double unread = std::numeric_limits<double>::quiet_NaN();
                const std::vector<T> a = elements<T>({1, unread,
                                                      2, 3});
                const std::vector<T> b = elements<T>({1, 2, 3, 4, 5, 6});

                const Tensor A = upload(*this->engine, {2, 2}, a);
                const Tensor B = upload(*this->engine, {3, 2}, b);
                Tensor C = upload(*this->engine, {3, 2}, std::vector<T>(6, poison<T>()));

                SushiBLAS::BLASOps(this->recorder()).symm(A, B, C, Core::Side::Right,
                                                          Core::Uplo::Lower, 1.0, 0.0);
                this->recording->run().value();

                expect_equal(read_tensor<T>(C), elements<T>({5, 8, 11, 18, 17, 28}));
            });
        }

        INSTANTIATE_TEST_SUITE_P(
            , Medium_Symm,
            ::testing::ValuesIn(row_major(plain_rows(K_SYMM_TYPES, K_BOTH_MODES))),
            row_name<NoAxis>);

        /** @brief Runs symm against gemm on the mirrored matrix, per side and triangle. */
        class Medium_SymmGemm : public SymmCase<SideTriangle>
        {

        };

        TEST_P(Medium_SymmGemm, EqualsGemmOnTheFullMatrixToTheBit)
        {
            with_non_half_element(GetParam().type, [&](auto element)
            {
                using T = typename decltype(element)::type;
                expect_product_equals_gemm<T>(*this->engine, *this->recording, Routine::Symm,
                                              GetParam().axis);
            });
        }

        INSTANTIATE_TEST_SUITE_P(
            , Medium_SymmGemm,
            ::testing::ValuesIn(row_major(rows_of(K_SYMM_TYPES, K_BOTH_MODES, side_triangles()))),
            row_name<SideTriangle>);

        /** @brief Runs a batched symm against the same call on each matrix alone. */
        class Medium_SymmBatch : public SymmCase<BatchForm>
        {

        };

        TEST_P(Medium_SymmBatch, ABatchIsMultipliedMatrixByMatrix)
        {
            with_non_half_element(GetParam().type, [&](auto element)
            {
                using T = typename decltype(element)::type;
                expect_batch_matches_each_matrix<T>(*this->engine, *this->recording,
                                                    GetParam().axis);
            });
        }

        INSTANTIATE_TEST_SUITE_P(
            , Medium_SymmBatch,
            ::testing::ValuesIn(row_major(rows_of(K_SYMM_TYPES, K_BOTH_MODES, batch_forms()))),
            row_name<BatchForm>);

        /** @brief Runs symm with a zero beta over a poisoned C, per side. */
        class Medium_SymmZeroBeta : public SymmCase<Core::Side>
        {

        };

        TEST_P(Medium_SymmZeroBeta, AZeroBetaDoesNotReadC)
        {
            with_non_half_element(GetParam().type, [&](auto element)
            {
                using T = typename decltype(element)::type;
                expect_zero_beta_ignores_c<T>(*this->engine, *this->recording,
                                              GetParam().axis);
            });
        }

        INSTANTIATE_TEST_SUITE_P(
            , Medium_SymmZeroBeta,
            ::testing::ValuesIn(row_major(rows_of(K_SYMM_TYPES, K_BOTH_MODES, K_SIDES))),
            row_name<Core::Side>);

        /** @brief Runs symm on a product with no row or no column. */
        class Medium_SymmEmpty : public SymmCase<EmptyForm>
        {

        };

        TEST_P(Medium_SymmEmpty, AnEmptyProductRecordsAndRuns)
        {
            with_non_half_element(GetParam().type, [&](auto element)
            {
                using T = typename decltype(element)::type;
                const EmptyForm& form = GetParam().axis;
                const Core::DataType dtype = sushiblas_dtype<T>();
                const int64_t order = form.side == Core::Side::Left ? form.rows : form.cols;

                const Tensor A = this->engine->create_tensor({order, order}, dtype);
                const Tensor B = this->engine->create_tensor({form.rows, form.cols}, dtype);
                Tensor C = this->engine->create_tensor({form.rows, form.cols}, dtype);

                EXPECT_NO_THROW(SushiBLAS::BLASOps(this->recorder()).symm(
                    A, B, C, form.side, Core::Uplo::Upper));
                EXPECT_TRUE(this->recording->run().has_value());
            });
        }

        INSTANTIATE_TEST_SUITE_P(
            , Medium_SymmEmpty,
            ::testing::ValuesIn(row_major(rows_of(K_SYMM_TYPES, K_BOTH_MODES, empty_forms()))),
            row_name<EmptyForm>);

        /** @brief Runs the hand-computed hemm product over the complex element types. */
        class Medium_Hemm : public SymmCase<NoAxis>
        {

        };

        TEST_P(Medium_Hemm, MatchesAHandComputedProduct)
        {
            with_complex_element(GetParam().type, [&](auto element)
            {
                using T = typename decltype(element)::type;
                const std::vector<T> a = {T(2, 9), T(1, 1),
                                          poison<T>(), T(3, 9)};
                const std::vector<T> b = {T(1, 0), T(0, 1)};

                const Tensor A = upload(*this->engine, {2, 2}, a);
                const Tensor B = upload(*this->engine, {2, 1}, b);
                Tensor C = upload(*this->engine, {2, 1}, std::vector<T>(2, poison<T>()));

                SushiBLAS::BLASOps(this->recorder()).hemm(A, B, C, Core::Side::Left,
                                                          Core::Uplo::Upper, 1.0, 0.0);
                this->recording->run().value();

                // Row 0 is 2 * 1 + (1 + i) * i; row 1 is conj(1 + i) * 1 + 3 * i.
                expect_equal(read_tensor<T>(C), std::vector<T>{T(1, 1), T(1, 2)});
            });
        }

        INSTANTIATE_TEST_SUITE_P(
            , Medium_Hemm,
            ::testing::ValuesIn(row_major(plain_rows(K_COMPLEX_TYPES, K_BOTH_MODES))),
            row_name<NoAxis>);

        /** @brief Runs hemm against gemm on the mirrored Hermitian matrix. */
        class Medium_HemmGemm : public SymmCase<SideTriangle>
        {

        };

        TEST_P(Medium_HemmGemm, EqualsGemmOnTheFullHermitianMatrixToTheBit)
        {
            with_complex_element(GetParam().type, [&](auto element)
            {
                using T = typename decltype(element)::type;
                expect_product_equals_gemm<T>(*this->engine, *this->recording, Routine::Hemm,
                                              GetParam().axis);
            });
        }

        INSTANTIATE_TEST_SUITE_P(
            , Medium_HemmGemm,
            ::testing::ValuesIn(row_major(rows_of(K_COMPLEX_TYPES, K_BOTH_MODES,
                                                  side_triangles()))),
            row_name<SideTriangle>);

        /** @brief Runs hemm on a real tensor, which forwards to symm. */
        class Medium_SymmHemmForward : public SymmCase<NoAxis>
        {

        };

        TEST_P(Medium_SymmHemmForward, HemmOnARealTensorIsSymm)
        {
            std::mt19937 generator(K_SEED);
            const std::vector<float> a = random_elements<float>(generator, K_ROWS * K_ROWS);
            const std::vector<float> b = random_elements<float>(generator, K_ROWS * K_COLS);
            const std::vector<float> c = random_elements<float>(generator, K_ROWS * K_COLS);

            const Tensor A = upload(*this->engine, {K_ROWS, K_ROWS}, a);
            const Tensor B = upload(*this->engine, {K_ROWS, K_COLS}, b);
            Tensor from_hemm = upload(*this->engine, {K_ROWS, K_COLS}, c);
            Tensor from_symm = upload(*this->engine, {K_ROWS, K_COLS}, c);

            record_product(this->recorder(), Routine::Hemm, A, B, from_hemm, Core::Side::Left,
                           Core::Uplo::Upper, K_ALPHA, K_BETA);
            record_product(this->recorder(), Routine::Symm, A, B, from_symm, Core::Side::Left,
                           Core::Uplo::Upper, K_ALPHA, K_BETA);
            this->recording->run().value();

            expect_same_bytes(read_tensor<float>(from_hemm), read_tensor<float>(from_symm));
        }

        INSTANTIATE_TEST_SUITE_P(
            , Medium_SymmHemmForward,
            ::testing::ValuesIn(row_major(plain_rows({Core::DataType::Float32}, K_BOTH_MODES))),
            row_name<NoAxis>);

        /** @brief Runs the HALF row sums of symm; the fixture skips a device without fp16. */
        class Medium_SymmHalf : public SymmCase<NoAxis>
        {

        };

        TEST_P(Medium_SymmHalf, HalfSumsInFloat)
        {
            const double unread = std::numeric_limits<double>::quiet_NaN();
            const std::vector<sycl::half> a = elements<sycl::half>({2048, 1, -2048,
                                                                    unread, 0, 0,
                                                                    unread, unread, 0});
            const std::vector<sycl::half> ones = elements<sycl::half>({1, 1, 1});

            const Tensor A = upload(*this->engine, {3, 3}, a);
            const Tensor B = upload(*this->engine, {3, 1}, ones);
            Tensor C = upload(*this->engine, {3, 1},
                              std::vector<sycl::half>(3, poison<sycl::half>()));

            SushiBLAS::BLASOps(this->recorder()).symm(A, B, C, Core::Side::Left,
                                                      Core::Uplo::Upper, 1.0, 0.0);
            this->recording->run().value();

            // Row 0 sums 2048 + 1 - 2048, which is 1 in FLOAT32 and 0 in HALF.
            const std::vector<sycl::half> result = read_tensor<sycl::half>(C);
            ASSERT_EQ(3u, result.size());
            EXPECT_EQ(static_cast<float>(result[0]), 1.0f);
            EXPECT_EQ(static_cast<float>(result[1]), 1.0f);
            EXPECT_EQ(static_cast<float>(result[2]), -2048.0f);
        }

        INSTANTIATE_TEST_SUITE_P(
            , Medium_SymmHalf,
            ::testing::ValuesIn(row_major(plain_rows({Core::DataType::Half}, K_BOTH_MODES))),
            row_name<NoAxis>);

        /**
         * @brief Runs the ForceReference refusal of a device-only operand; it needs device-only
         *        USM, so it cannot run on a host tensor (decision 75).
         */
        class Medium_SymmDeviceOnly : public BlasEngineStart
        {
            protected:
                /** @brief Builds the device-only operand, or skips on a device without it. */
                void SetUp() override
                {
                    BlasEngineStart::SetUp();
                    this->start(Core::Layout::RowMajor, ExecutionMode::ForceReference);
                    device_only_ = make_device_tensor({3, 3});
                    if (!device_only_)
                        GTEST_SKIP() << "This device offers no device-only USM.";
                }

                /**
                 * @brief Returns the message of the UnsupportedExecution that record throws, or an
                 *        empty string when it throws nothing else.
                 */
                template<typename Record>
                [[nodiscard]] std::string refusal_message(Record&& record)
                {
                    try
                    {
                        record();
                    }
                    catch (const UnsupportedExecution& refusal)
                    {
                        return refusal.what();
                    }
                    return "";
                }

                /// @brief A FLOAT32 tensor in device-only USM.
                std::optional<Tensor> device_only_;

            private:
                /**
                 * @brief Returns a FLOAT32 tensor of dims in device-only USM, or nothing without
                 *        such USM.
                 */
                [[nodiscard]] std::optional<Tensor>
                make_device_tensor(std::initializer_list<int64_t> dims)
                {
                    try
                    {
                        Tensor tensor = engine->create_tensor(dims, Core::DataType::Float32,
                                                              SushiRuntime::Residency::Device);
                        const sycl::context context = TestEnvironment::queue().get_context();
                        if (sycl::get_pointer_type(tensor.data(), context) !=
                            sycl::usm::alloc::device)
                            return std::nullopt;
                        return tensor;
                    }
                    catch (const std::exception&)
                    {
                        return std::nullopt;
                    }
                }
        };

        TEST_F(Medium_SymmDeviceOnly, ForceReferenceRefusesADeviceOnlyOperand)
        {
            const Tensor B = engine->create_tensor({3, 2});
            Tensor C = engine->create_tensor({3, 2});

            const std::string message = refusal_message([&]
            {
                SushiBLAS::BLASOps(recorder()).symm(*device_only_, B, C, Core::Side::Left,
                                                    Core::Uplo::Upper);
            });

            EXPECT_NE(message.find("blas.lvl3.symm"), std::string::npos) << message;
        }
    } // namespace Test
} // namespace SushiBLAS
