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
 * @file test_symm.cpp
 * @brief Tests symm and hemm against hand-computed products, gemm on the full matrix and refusals.
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
#include <SushiBLAS/core/traits.hpp>
#include <SushiBLAS/arithmetic/complex_arithmetic.hpp>
#include <SushiBLAS/engine_fixture/sushiblas_test.hpp>
#include <SushiBLAS/tensor_expectations/tensor_readback.hpp>
#include <SushiBLAS/tensor_expectations/scalar_dtype.hpp>
#include <SushiBLAS/typed_suite_naming/default_type_names.hpp>
#include <SushiBLAS/test_runner/test_environment.hpp>
#include <SushiBLAS/graph_probe/graph_probe.hpp>

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

            /// @brief The two modes every case runs in.
            constexpr ExecutionMode K_MODES[] = {ExecutionMode::ForceSYCL,
                                                 ExecutionMode::ForceReference};

            /// @brief The two sides A can stand on.
            constexpr Core::Side K_SIDES[] = {Core::Side::Left, Core::Side::Right};

            /// @brief The two triangles A can be stored in.
            constexpr Core::Uplo K_TRIANGLES[] = {Core::Uplo::Upper, Core::Uplo::Lower};

            /// @brief A scalar with a non-zero imaginary part, which a real tensor refuses.
            const BLASScalar K_COMPLEX_SCALAR = std::complex<double>(1.0, 2.0);

            /** @brief Names the routine a shared case calls. */
            enum class Routine : std::uint8_t
            {
                Symm, /**< The symmetric product. */
                Hemm  /**< The Hermitian product. */
            };

            /** @brief Returns the name of a mode for a failure message. */
            [[nodiscard]] const char* mode_name(ExecutionMode mode) noexcept
            {
                return mode == ExecutionMode::ForceSYCL ? "ForceSYCL" : "ForceReference";
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
             * @brief Expects a routine to leave the bytes gemm leaves on the mirrored full matrix,
             *        for each side, triangle and mode.
             */
            template<typename T>
            void expect_product_equals_gemm(Engine& engine, RecordingFixture& recording,
                                            Routine routine)
            {
                std::mt19937 generator(K_SEED);
                for (const Core::Side side : K_SIDES)
                {
                    for (const Core::Uplo uplo : K_TRIANGLES)
                    {
                        SCOPED_TRACE(std::string(Core::to_string(side)) + " " +
                                     Core::to_string(uplo));

                        const bool left = side == Core::Side::Left;
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
                        const std::vector<T> full = mirrored(a, order, uplo, routine);
                        poison_unread_triangle(a, order, uplo);

                        for (const ExecutionMode mode : K_MODES)
                        {
                            SCOPED_TRACE(mode_name(mode));
                            engine.set_mode(mode);
                            recording.recorder().set_mode(mode);

                            const Tensor A = upload(engine, {order, order}, a);
                            const Tensor S = upload(engine, {order, order}, full);
                            const Tensor B = upload(engine, {K_ROWS, K_COLS}, b);
                            Tensor from_routine = upload(engine, {K_ROWS, K_COLS}, c);
                            Tensor from_gemm = upload(engine, {K_ROWS, K_COLS}, c);

                            TaskRecorder& recorder = recording.recorder();
                            record_product(recorder, routine, A, B, from_routine, side, uplo,
                                           K_ALPHA, K_BETA);
                            SushiBLAS::BLASOps(recorder).gemm(left ? S : B, left ? B : S,
                                                              from_gemm, Core::Transpose::NoTrans,
                                                              Core::Transpose::NoTrans,
                                                              K_ALPHA, K_BETA);
                            recording.run().value();

                            expect_same_bytes(read_tensor<T>(from_routine),
                                              read_tensor<T>(from_gemm));
                        }
                    }
                }
            }
        } // namespace

        /** @brief Runs symm over the real and complex element types. */
        template<typename T>
        class Medium_Symm : public SushiBLASTest
        {

        };

        using SymmTypes = ::testing::Types<float, double, std::complex<float>,
                                           std::complex<double>>;
        SB_TYPED_TEST_SUITE(Medium_Symm, SymmTypes);

        TYPED_TEST(Medium_Symm, LeftSideMatchesAHandComputedProduct)
        {
            using T = TypeParam;
            const double unread = std::numeric_limits<double>::quiet_NaN();
            const std::vector<T> a = elements<T>({1, 2, 3,
                                                  unread, 4, 5,
                                                  unread, unread, 6});
            const std::vector<T> b = elements<T>({1, 2, 3, 4, 5, 6});
            const std::vector<T> ones = elements<T>({1, 1, 1, 1, 1, 1});

            for (const ExecutionMode mode : K_MODES)
            {
                SCOPED_TRACE(mode_name(mode));
                this->set_mode(mode);
                const Tensor A = upload(*this->engine, {3, 3}, a);
                const Tensor B = upload(*this->engine, {3, 2}, b);
                Tensor C = upload(*this->engine, {3, 2}, ones);

                SushiBLAS::BLASOps(this->recorder()).symm(A, B, C, Core::Side::Left,
                                                       Core::Uplo::Upper, 2.0, -1.0);
                this->recording->run().value();

                // A * B is {22 28, 39 50, 48 62}; C is twice that minus one.
                expect_equal(read_tensor<T>(C), elements<T>({43, 55, 77, 99, 95, 123}));
            }
        }

        TYPED_TEST(Medium_Symm, RightSideMatchesAHandComputedProduct)
        {
            using T = TypeParam;
            const double unread = std::numeric_limits<double>::quiet_NaN();
            const std::vector<T> a = elements<T>({1, unread,
                                                  2, 3});
            const std::vector<T> b = elements<T>({1, 2, 3, 4, 5, 6});

            for (const ExecutionMode mode : K_MODES)
            {
                SCOPED_TRACE(mode_name(mode));
                this->set_mode(mode);
                const Tensor A = upload(*this->engine, {2, 2}, a);
                const Tensor B = upload(*this->engine, {3, 2}, b);
                Tensor C = upload(*this->engine, {3, 2}, std::vector<T>(6, poison<T>()));

                SushiBLAS::BLASOps(this->recorder()).symm(A, B, C, Core::Side::Right,
                                                       Core::Uplo::Lower, 1.0, 0.0);
                this->recording->run().value();

                expect_equal(read_tensor<T>(C), elements<T>({5, 8, 11, 18, 17, 28}));
            }
        }

        TYPED_TEST(Medium_Symm, EqualsGemmOnTheFullMatrixToTheBit)
        {
            expect_product_equals_gemm<TypeParam>(*this->engine, *this->recording, Routine::Symm);
        }

        TYPED_TEST(Medium_Symm, ABatchIsMultipliedMatrixByMatrix)
        {
            using T = TypeParam;
            const int64_t matrix = K_BATCH_ROWS * K_BATCH_COLS;

            for (const Core::Side side : K_SIDES)
            {
                const int64_t order = side == Core::Side::Left ? K_BATCH_ROWS : K_BATCH_COLS;
                const std::vector<T> a = integer_elements<T>(0, K_BATCH * order * order);
                const std::vector<T> b = integer_elements<T>(11, K_BATCH * matrix);
                const std::vector<T> c = integer_elements<T>(23, K_BATCH * matrix);

                for (const bool a_is_batched : {true, false})
                {
                    for (const ExecutionMode mode : K_MODES)
                    {
                        SCOPED_TRACE(std::string(Core::to_string(side)) + " " + mode_name(mode) +
                                     (a_is_batched ? ", A batched" : ", A of rank 2"));
                        this->set_mode(mode);

                        const Tensor A = a_is_batched
                            ? upload(*this->engine, {K_BATCH, order, order}, a)
                            : upload(*this->engine, {order, order},
                                     matrix_of(a, 0, order * order));
                        const Tensor B =
                            upload(*this->engine, {K_BATCH, K_BATCH_ROWS, K_BATCH_COLS}, b);
                        Tensor C =
                            upload(*this->engine, {K_BATCH, K_BATCH_ROWS, K_BATCH_COLS}, c);
                        SushiBLAS::BLASOps(this->recorder()).symm(A, B, C, side, Core::Uplo::Upper,
                                                               2.0, -1.0);
                        this->recording->run().value();
                        const std::vector<T> batched = read_tensor<T>(C);

                        for (int64_t index = 0; index < K_BATCH; ++index)
                        {
                            SCOPED_TRACE("matrix " + std::to_string(index));
                            const Tensor one_a = upload(
                                *this->engine, {order, order},
                                matrix_of(a, a_is_batched ? index : 0, order * order));
                            const Tensor one_b = upload(*this->engine,
                                                        {K_BATCH_ROWS, K_BATCH_COLS},
                                                        matrix_of(b, index, matrix));
                            Tensor one_c = upload(*this->engine, {K_BATCH_ROWS, K_BATCH_COLS},
                                                  matrix_of(c, index, matrix));
                            SushiBLAS::BLASOps(this->recorder()).symm(one_a, one_b, one_c, side,
                                                                   Core::Uplo::Upper, 2.0, -1.0);
                            this->recording->run().value();

                            expect_equal(matrix_of(batched, index, matrix),
                                         read_tensor<T>(one_c));
                        }
                    }
                }
            }
        }

        TYPED_TEST(Medium_Symm, AZeroBetaDoesNotReadC)
        {
            using T = TypeParam;
            std::mt19937 generator(K_SEED);

            for (const Core::Side side : K_SIDES)
            {
                const int64_t order = side == Core::Side::Left ? K_ROWS : K_COLS;
                const std::vector<T> a = random_elements<T>(generator, order * order);
                const std::vector<T> b = random_elements<T>(generator, K_ROWS * K_COLS);

                for (const ExecutionMode mode : K_MODES)
                {
                    SCOPED_TRACE(std::string(Core::to_string(side)) + " " + mode_name(mode));
                    this->set_mode(mode);
                    const Tensor A = upload(*this->engine, {order, order}, a);
                    const Tensor B = upload(*this->engine, {K_ROWS, K_COLS}, b);
                    Tensor C = upload(*this->engine, {K_ROWS, K_COLS},
                                      std::vector<T>(static_cast<std::size_t>(K_ROWS * K_COLS),
                                                     poison<T>()));

                    SushiBLAS::BLASOps(this->recorder()).symm(A, B, C, side, Core::Uplo::Lower,
                                                           K_ALPHA, 0.0);
                    this->recording->run().value();

                    const std::vector<T> result = read_tensor<T>(C);
                    for (std::size_t i = 0; i < result.size(); ++i)
                        EXPECT_FALSE(is_nan(result[i])) << "index " << i;
                }
            }
        }

        TYPED_TEST(Medium_Symm, AnEmptyProductRecordsAndRuns)
        {
            const Core::DataType dtype = sushiblas_dtype<TypeParam>();
            const int64_t shapes[][2] = {{0, 3}, {3, 0}};

            for (const auto& shape : shapes)
            {
                const int64_t rows = shape[0];
                const int64_t cols = shape[1];

                for (const Core::Side side : K_SIDES)
                {
                    const int64_t order = side == Core::Side::Left ? rows : cols;

                    for (const ExecutionMode mode : K_MODES)
                    {
                        SCOPED_TRACE(std::string(Core::to_string(side)) + " " + mode_name(mode));
                        this->set_mode(mode);
                        const Tensor A = this->engine->create_tensor({order, order}, dtype);
                        const Tensor B = this->engine->create_tensor({rows, cols}, dtype);
                        Tensor C = this->engine->create_tensor({rows, cols}, dtype);

                        EXPECT_NO_THROW(SushiBLAS::BLASOps(this->recorder()).symm(
                            A, B, C, side, Core::Uplo::Upper));
                        EXPECT_TRUE(this->recording->run().has_value());
                    }
                }
            }
        }

        /** @brief Runs hemm over the complex element types. */
        template<typename T>
        class Medium_Hemm : public SushiBLASTest
        {

        };

        using HemmTypes = ::testing::Types<std::complex<float>, std::complex<double>>;
        SB_TYPED_TEST_SUITE(Medium_Hemm, HemmTypes);

        TYPED_TEST(Medium_Hemm, EqualsGemmOnTheFullHermitianMatrixToTheBit)
        {
            expect_product_equals_gemm<TypeParam>(*this->engine, *this->recording, Routine::Hemm);
        }

        TYPED_TEST(Medium_Hemm, MatchesAHandComputedProduct)
        {
            using T = TypeParam;
            const std::vector<T> a = {T(2, 9), T(1, 1),
                                      poison<T>(), T(3, 9)};
            const std::vector<T> b = {T(1, 0), T(0, 1)};

            for (const ExecutionMode mode : K_MODES)
            {
                SCOPED_TRACE(mode_name(mode));
                this->set_mode(mode);
                const Tensor A = upload(*this->engine, {2, 2}, a);
                const Tensor B = upload(*this->engine, {2, 1}, b);
                Tensor C = upload(*this->engine, {2, 1}, std::vector<T>(2, poison<T>()));

                SushiBLAS::BLASOps(this->recorder()).hemm(A, B, C, Core::Side::Left,
                                                       Core::Uplo::Upper, 1.0, 0.0);
                this->recording->run().value();

                // Row 0 is 2 * 1 + (1 + i) * i; row 1 is conj(1 + i) * 1 + 3 * i.
                expect_equal(read_tensor<T>(C), std::vector<T>{T(1, 1), T(1, 2)});
            }
        }

        /** @brief Runs the cases of symm and hemm that are not typed over the element type. */
        class Medium_SymmHemmEdges : public SushiBLASTest
        {
            protected:
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
                        const sycl::context context =
                            TestEnvironment::queue().get_context();
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

                /**
                 * @brief Expects record to throw std::invalid_argument and to add no task.
                 * @param record Callable `(SushiBLAS::BLASOps&) -> TaskHandle`.
                 */
                template<typename Record>
                void expect_refused(const char* what, Record&& record)
                {
                    SCOPED_TRACE(what);
                    SushiBLAS::BLASOps ops(probe.recorder());
                    EXPECT_THROW((void)record(ops), std::invalid_argument);
                    EXPECT_EQ(0u, probe.take_recorded_count());
                }

                /// @brief The graph the refusal cases record on and read.
                GraphProbe probe;
        };

        TEST_F(Medium_SymmHemmEdges, HemmOnARealTensorIsSymm)
        {
            std::mt19937 generator(K_SEED);
            const std::vector<float> a = random_elements<float>(generator, K_ROWS * K_ROWS);
            const std::vector<float> b = random_elements<float>(generator, K_ROWS * K_COLS);
            const std::vector<float> c = random_elements<float>(generator, K_ROWS * K_COLS);

            for (const ExecutionMode mode : K_MODES)
            {
                SCOPED_TRACE(mode_name(mode));
                set_mode(mode);
                const Tensor A = upload(*engine, {K_ROWS, K_ROWS}, a);
                const Tensor B = upload(*engine, {K_ROWS, K_COLS}, b);
                Tensor from_hemm = upload(*engine, {K_ROWS, K_COLS}, c);
                Tensor from_symm = upload(*engine, {K_ROWS, K_COLS}, c);

                record_product(recorder(), Routine::Hemm, A, B, from_hemm, Core::Side::Left,
                               Core::Uplo::Upper, K_ALPHA, K_BETA);
                record_product(recorder(), Routine::Symm, A, B, from_symm, Core::Side::Left,
                               Core::Uplo::Upper, K_ALPHA, K_BETA);
                recording->run().value();

                expect_same_bytes(read_tensor<float>(from_hemm), read_tensor<float>(from_symm));
            }
        }

        TEST_F(Medium_SymmHemmEdges, HalfSumsInFloat)
        {
            const sycl::queue& queue = TestEnvironment::queue();
            if (!queue.get_device().has(sycl::aspect::fp16))
                GTEST_SKIP() << "the device does not report the fp16 aspect";

            const double unread = std::numeric_limits<double>::quiet_NaN();
            const std::vector<sycl::half> a = elements<sycl::half>({2048, 1, -2048,
                                                                    unread, 0, 0,
                                                                    unread, unread, 0});
            const std::vector<sycl::half> ones = elements<sycl::half>({1, 1, 1});

            for (const ExecutionMode mode : K_MODES)
            {
                SCOPED_TRACE(mode_name(mode));
                set_mode(mode);
                const Tensor A = upload(*engine, {3, 3}, a);
                const Tensor B = upload(*engine, {3, 1}, ones);
                Tensor C = upload(*engine, {3, 1},
                                  std::vector<sycl::half>(3, poison<sycl::half>()));

                SushiBLAS::BLASOps(recorder()).symm(A, B, C, Core::Side::Left, Core::Uplo::Upper,
                                                 1.0, 0.0);
                recording->run().value();

                // Row 0 sums 2048 + 1 - 2048, which is 1 in FLOAT32 and 0 in HALF.
                const std::vector<sycl::half> result = read_tensor<sycl::half>(C);
                ASSERT_EQ(3u, result.size());
                EXPECT_EQ(static_cast<float>(result[0]), 1.0f);
                EXPECT_EQ(static_cast<float>(result[1]), 1.0f);
                EXPECT_EQ(static_cast<float>(result[2]), -2048.0f);
            }
        }

        TEST_F(Medium_SymmHemmEdges, EveryRefusalIsAnInvalidArgument)
        {
            const Tensor A = engine->create_tensor({3, 3});
            const Tensor B = engine->create_tensor({3, 2});
            Tensor C = engine->create_tensor({3, 2});
            const Tensor wide_B = engine->create_tensor({2, 3});
            Tensor wide_C = engine->create_tensor({2, 3});
            const Tensor A_of_order_4 = engine->create_tensor({4, 4});

            expect_refused("side holding 0", [&](SushiBLAS::BLASOps& ops)
            {
                return ops.symm(A, B, C, static_cast<Core::Side>(0), Core::Uplo::Upper);
            });
            expect_refused("uplo holding 0", [&](SushiBLAS::BLASOps& ops)
            {
                return ops.symm(A, B, C, Core::Side::Left, static_cast<Core::Uplo>(0));
            });
            expect_refused("B of another dtype", [&](SushiBLAS::BLASOps& ops)
            {
                const Tensor other = engine->create_tensor({3, 2}, Core::DataType::Float64);
                return ops.symm(A, other, C, Core::Side::Left, Core::Uplo::Upper);
            });
            expect_refused("C of another dtype", [&](SushiBLAS::BLASOps& ops)
            {
                Tensor other = engine->create_tensor({3, 2}, Core::DataType::Float64);
                return ops.symm(A, B, other, Core::Side::Left, Core::Uplo::Upper);
            });
            expect_refused("a complex alpha on FLOAT32", [&](SushiBLAS::BLASOps& ops)
            {
                return ops.symm(A, B, C, Core::Side::Left, Core::Uplo::Upper, K_COMPLEX_SCALAR);
            });
            expect_refused("a complex beta on FLOAT32", [&](SushiBLAS::BLASOps& ops)
            {
                return ops.symm(A, B, C, Core::Side::Left, Core::Uplo::Upper, 1.0,
                                K_COMPLEX_SCALAR);
            });
            expect_refused("a 3 x 4 A", [&](SushiBLAS::BLASOps& ops)
            {
                const Tensor other = engine->create_tensor({3, 4});
                return ops.symm(other, B, C, Core::Side::Left, Core::Uplo::Upper);
            });
            expect_refused("A of order 4 with a 3-row C under LEFT", [&](SushiBLAS::BLASOps& ops)
            {
                return ops.symm(A_of_order_4, B, C, Core::Side::Left, Core::Uplo::Upper);
            });
            expect_refused("A of order 4 with a 3-column C under RIGHT",
                           [&](SushiBLAS::BLASOps& ops)
            {
                return ops.symm(A_of_order_4, wide_B, wide_C, Core::Side::Right,
                                Core::Uplo::Upper);
            });
            expect_refused("B of another shape than C", [&](SushiBLAS::BLASOps& ops)
            {
                const Tensor other = engine->create_tensor({3, 3});
                return ops.symm(A, other, C, Core::Side::Left, Core::Uplo::Upper);
            });
            expect_refused("a column-major A with a row-major C", [&](SushiBLAS::BLASOps& ops)
            {
                const Tensor other(A.storage, {3, 3}, Core::DataType::Float32, 0,
                                   Core::Layout::ColumnMajor);
                return ops.symm(other, B, C, Core::Side::Left, Core::Uplo::Upper);
            });
            expect_refused("a batch of 3 against a batch of 4", [&](SushiBLAS::BLASOps& ops)
            {
                const Tensor batch_A = engine->create_tensor({3, 3, 3});
                const Tensor batch_B = engine->create_tensor({4, 3, 2});
                Tensor batch_C = engine->create_tensor({4, 3, 2});
                return ops.symm(batch_A, batch_B, batch_C, Core::Side::Left, Core::Uplo::Upper);
            });
            expect_refused("an INT32 A", [&](SushiBLAS::BLASOps& ops)
            {
                const Tensor int_A = engine->create_tensor({3, 3}, Core::DataType::Int32);
                const Tensor int_B = engine->create_tensor({3, 2}, Core::DataType::Int32);
                Tensor int_C = engine->create_tensor({3, 2}, Core::DataType::Int32);
                return ops.symm(int_A, int_B, int_C, Core::Side::Left, Core::Uplo::Upper);
            });
        }

        TEST_F(Medium_SymmHemmEdges, ForceReferenceRefusesADeviceOnlyOperand)
        {
            std::optional<Tensor> A = make_device_tensor({3, 3});
            if (!A)
                GTEST_SKIP() << "This device offers no device-only USM.";
            const Tensor B = engine->create_tensor({3, 2});
            Tensor C = engine->create_tensor({3, 2});

            set_mode(ExecutionMode::ForceReference);
            try
            {
                SushiBLAS::BLASOps(recorder()).symm(*A, B, C, Core::Side::Left, Core::Uplo::Upper);
                FAIL() << "ForceReference must refuse a device-only operand.";
            }
            catch (const std::invalid_argument& e)
            {
                const std::string message = e.what();
                EXPECT_NE(message.find("blas.lvl3.symm"), std::string::npos) << message;
            }
        }
    } // namespace Test
} // namespace SushiBLAS
