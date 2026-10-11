/**************************************************************************/
/* test_sparse_spsymv.cpp                                                 */
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
 * @file test_sparse_spsymv.cpp
 * @brief Tests SparseOps::spsymv, sphemv and their SYCL kernel against mirrored host products.
 * @author Mustafa Garip
 */

#include <cmath>
#include <limits>
#include <vector>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <utility>
#include <type_traits>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <SushiBLAS/SushiBLAS.h>
#include <SushiBLAS/sparse_kernels/sparse.hpp>
#include "test_utils.hpp"
#include "sparse_samples.hpp"

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            /** @brief A host vector of expectations or of operand values. */
            using Wide = std::vector<std::complex<double>>;

            /// @brief Absolute error allowed on an element of a hand-worked complex product.
            constexpr double K_SAMPLE_TOLERANCE = 1e-4;

            /// @brief Order of the integer-valued matrix; above one work-group on every device.
            constexpr int64_t K_EXACT_ORDER = 1000;

            /// @brief Entries drawn for each row of the integer-valued matrix.
            constexpr int64_t K_EXACT_PER_ROW = 8;

            /// @brief Seed of the integer-valued matrix.
            constexpr std::uint32_t K_EXACT_SEED = 20261007u;

            /// @brief The two triangles a one-triangle product reads.
            constexpr Core::Uplo K_BOTH_TRIANGLES[2] = {Core::Uplo::Lower, Core::Uplo::Upper};

            /** @brief Selects which of the two one-triangle products a case records. */
            enum class Mirror : std::uint8_t
            {
                Symmetric, /**< spsymv: the mirror of an entry is the entry. */
                Hermitian  /**< sphemv: the mirror of an entry is its conjugate. */
            };

            /**
             * @brief Returns the lower triangle of the 4 x 4 symmetric sample S.
             * @return Rows (2), (1, 3), (., ., 5), (4, ., 6, 7); S * (1, 2, 3, 4) is
             *         (20, 7, 39, 50).
             */
            template<typename T>
            [[nodiscard]] HostTriplets<T> symmetric_sample_lower()
            {
                HostTriplets<T> m;
                m.rows = 4;
                m.cols = 4;
                m.row_indices = {3, 0, 1, 3, 1, 2, 3};
                m.column_indices = {2, 0, 0, 0, 1, 2, 3};
                for (const double value : {6.0, 2.0, 1.0, 4.0, 3.0, 5.0, 7.0})
                    m.values.push_back(narrowed<T>(std::complex<double>(value, 0.0)));
                return m;
            }

            /** @brief Returns the transpose of a matrix: every entry with its indices exchanged. */
            template<typename T>
            [[nodiscard]] HostTriplets<T> transpose_of(const HostTriplets<T>& host)
            {
                HostTriplets<T> m = host;
                std::swap(m.rows, m.cols);
                std::swap(m.row_indices, m.column_indices);
                return m;
            }

            /** @brief Returns the entries of a matrix on and below its diagonal. */
            template<typename T>
            [[nodiscard]] HostTriplets<T> lower_part(const HostTriplets<T>& host)
            {
                HostTriplets<T> m;
                m.rows = host.rows;
                m.cols = host.cols;
                for (std::size_t k = 0; k < host.values.size(); ++k)
                {
                    if (host.row_indices[k] < host.column_indices[k])
                        continue;

                    m.row_indices.push_back(host.row_indices[k]);
                    m.column_indices.push_back(host.column_indices[k]);
                    m.values.push_back(host.values[k]);
                }
                return m;
            }

            /**
             * @brief Returns a lower triangle together with its mirror above the diagonal.
             * @param above The value every entry above the diagonal takes, or nullptr to mirror
             *        the entry below.
             */
            template<typename T>
            [[nodiscard]] HostTriplets<T> stored_in_full(const HostTriplets<T>& lower,
                                                         const T* above)
            {
                HostTriplets<T> m = lower;
                for (std::size_t k = 0; k < lower.values.size(); ++k)
                {
                    if (lower.row_indices[k] == lower.column_indices[k])
                        continue;

                    m.row_indices.push_back(lower.column_indices[k]);
                    m.column_indices.push_back(lower.row_indices[k]);
                    m.values.push_back(above != nullptr ? *above : lower.values[k]);
                }
                return m;
            }

            /**
             * @brief Returns alpha * A * x + beta * y on the host, for the A one triangle gives.
             * @param triangle Entries outside the triangle uplo names are skipped.
             * @param y The value before the call.
             */
            template<typename T>
            [[nodiscard]] Wide host_mirrored_product(const HostTriplets<T>& triangle,
                                                     Core::Uplo uplo, Mirror mirror,
                                                     std::complex<double> alpha, const Wide& x,
                                                     std::complex<double> beta, const Wide& y)
            {
                const bool lower = (uplo == Core::Uplo::Lower);
                const bool hermitian = (mirror == Mirror::Hermitian);

                Wide sums(y.size(), std::complex<double>(0.0, 0.0));
                for (std::size_t k = 0; k < triangle.values.size(); ++k)
                {
                    const std::size_t i = static_cast<std::size_t>(triangle.row_indices[k]);
                    const std::size_t j = static_cast<std::size_t>(triangle.column_indices[k]);
                    const std::complex<double> value = widened(triangle.values[k]);

                    if (i == j)
                    {
                        sums[i] += (hermitian ? std::complex<double>(value.real(), 0.0) : value) *
                                   x[i];
                    }
                    else if (lower ? j < i : j > i)
                    {
                        sums[i] += value * x[j];
                        sums[j] += (hermitian ? std::conj(value) : value) * x[i];
                    }
                }

                Wide result(y.size());
                for (std::size_t i = 0; i < y.size(); ++i)
                    result[i] = alpha * sums[i] + beta * y[i];
                return result;
            }
        } // namespace

        /**
         * @brief Builds CSR triangles from host triplets and runs spsymv or sphemv in a chosen
         *        execution mode.
         */
        template<typename TypeParam>
        class Unit_SparseSpsymvTest : public SushiBLASTest
        {
            protected:
                /** @brief The element type under test. */
                using T = typename TypeParam::Value;

                /** @brief The index type under test. */
                using Index = typename TypeParam::IndexType;

                /** @brief Skips the HALF suite on a device without the fp16 aspect. */
                void SetUp() override
                {
                    SushiBLASTest::SetUp();
                    if (std::is_same_v<T, sycl::half> &&
                        !TestEnvironment::get_context().devices().queue(0).get_device().has(
                            sycl::aspect::fp16))
                        GTEST_SKIP() << "the device does not report the fp16 aspect";
                }

                /** @brief Returns the CSR matrix of host triplets, stored with base. */
                [[nodiscard]] SparseMatrix build(const HostTriplets<T>& host, Core::IndexBase base)
                {
                    const SparseBuilder builder(engine->get_tensor_factory());
                    return builder.matrix_from_triplets(
                        host.rows, host.cols, stored_indices<Index>(host.row_indices, base),
                        stored_indices<Index>(host.column_indices, base), host.values, base);
                }

                /** @brief Returns a rank-1 tensor holding values narrowed to T. */
                [[nodiscard]] Tensor vector_of(const Wide& values)
                {
                    std::vector<T> host(values.size());
                    for (std::size_t i = 0; i < values.size(); ++i)
                        host[i] = narrowed<T>(values[i]);

                    const int64_t count = static_cast<int64_t>(values.size());
                    return create_initialized_tensor<T>({count}, host);
                }

                /** @brief Returns y after y = alpha * A * x + beta * y recorded in mode. */
                [[nodiscard]] std::vector<T> run(ExecutionMode mode, Mirror mirror,
                                                 const HostTriplets<T>& host, Core::IndexBase base,
                                                 Core::Uplo uplo, std::complex<double> alpha,
                                                 const Wide& x, std::complex<double> beta,
                                                 const Wide& y)
                {
                    engine->set_mode(mode);
                    const SparseMatrix A = build(host, base);
                    const Tensor tx = vector_of(x);
                    Tensor ty = vector_of(y);

                    SparseOps ops(*engine);
                    if (mirror == Mirror::Hermitian)
                        ops.sphemv(A, tx, ty, uplo, BLASScalar(alpha), BLASScalar(beta));
                    else
                        ops.spsymv(A, tx, ty, uplo, BLASScalar(alpha), BLASScalar(beta));
                    engine->execute();
                    return read_tensor<T>(ty);
                }

                /** @brief Expects every element within tolerance of its host expectation. */
                static void expect_near(const std::vector<T>& actual, const Wide& expected,
                                        double tolerance)
                {
                    ASSERT_EQ(actual.size(), expected.size());
                    for (std::size_t i = 0; i < actual.size(); ++i)
                    {
                        const std::complex<double> got = widened(actual[i]);
                        EXPECT_NEAR(got.real(), expected[i].real(), tolerance) << "element " << i;
                        EXPECT_NEAR(got.imag(), expected[i].imag(), tolerance) << "element " << i;
                    }
                }

                /** @brief Returns the triangle uplo names of the symmetric sample S. */
                [[nodiscard]] static HostTriplets<T> sample_triangle(Core::Uplo uplo)
                {
                    const HostTriplets<T> lower = symmetric_sample_lower<T>();
                    return uplo == Core::Uplo::Lower ? lower : transpose_of(lower);
                }
        };

        SB_TYPED_TEST_SUITE(Unit_SparseSpsymvTest, SparseCases);

        TYPED_TEST(Unit_SparseSpsymvTest, EitherStoredTriangleGivesTheProductOfTheSample)
        {
            const Wide x = counting(4, 1.0);
            const Wide zeros(4, std::complex<double>(0.0, 0.0));
            const Wide ones(4, std::complex<double>(1.0, 0.0));
            const std::complex<double> one(1.0, 0.0);
            const std::complex<double> zero(0.0, 0.0);
            const std::complex<double> two(2.0, 0.0);
            const std::complex<double> minus_one(-1.0, 0.0);

            const Wide plain = {{20.0, 0.0}, {7.0, 0.0}, {39.0, 0.0}, {50.0, 0.0}};
            const Wide scaled = {{39.0, 0.0}, {13.0, 0.0}, {77.0, 0.0}, {99.0, 0.0}};

            for (const ExecutionMode mode : K_BOTH_MODES)
            {
                for (const Core::IndexBase base : K_BOTH_BASES)
                {
                    for (const Core::Uplo uplo : K_BOTH_TRIANGLES)
                    {
                        const auto triangle = this->sample_triangle(uplo);
                        this->expect_near(this->run(mode, Mirror::Symmetric, triangle, base, uplo,
                                                    one, x, zero, zeros),
                                          plain, 0.0);
                        this->expect_near(this->run(mode, Mirror::Symmetric, triangle, base, uplo,
                                                    two, x, minus_one, ones),
                                          scaled, 0.0);
                    }
                }
            }
        }

        TYPED_TEST(Unit_SparseSpsymvTest, AMatrixStoredInFullGivesTheSameUnderEitherUplo)
        {
            using T = typename TypeParam::Value;

            const HostTriplets<T> full = stored_in_full<T>(symmetric_sample_lower<T>(), nullptr);
            const Wide x = counting(4, 1.0);
            const Wide zeros(4, std::complex<double>(0.0, 0.0));
            const std::complex<double> one(1.0, 0.0);
            const std::complex<double> zero(0.0, 0.0);
            const Wide expected = {{20.0, 0.0}, {7.0, 0.0}, {39.0, 0.0}, {50.0, 0.0}};

            for (const ExecutionMode mode : K_BOTH_MODES)
            {
                for (const Core::Uplo uplo : K_BOTH_TRIANGLES)
                    this->expect_near(this->run(mode, Mirror::Symmetric, full,
                                                Core::IndexBase::Zero, uplo, one, x, zero, zeros),
                                      expected, 0.0);
            }
        }

        TYPED_TEST(Unit_SparseSpsymvTest, TheOtherTriangleIsNotRead)
        {
            using T = typename TypeParam::Value;

            const double nan = std::numeric_limits<double>::quiet_NaN();
            const T poison = narrowed<T>(std::complex<double>(nan, nan));
            const HostTriplets<T> poisoned =
                stored_in_full<T>(symmetric_sample_lower<T>(), &poison);
            const Wide x = counting(4, 1.0);
            const Wide zeros(4, std::complex<double>(0.0, 0.0));
            const std::complex<double> one(1.0, 0.0);
            const std::complex<double> zero(0.0, 0.0);
            const Wide expected = {{20.0, 0.0}, {7.0, 0.0}, {39.0, 0.0}, {50.0, 0.0}};

            for (const ExecutionMode mode : K_BOTH_MODES)
            {
                for (const Core::IndexBase base : K_BOTH_BASES)
                    this->expect_near(this->run(mode, Mirror::Symmetric, poisoned, base,
                                                Core::Uplo::Lower, one, x, zero, zeros),
                                      expected, 0.0);
            }
        }

        TYPED_TEST(Unit_SparseSpsymvTest, ABetaOfZeroDoesNotReadY)
        {
            const double nan = std::numeric_limits<double>::quiet_NaN();
            const Wide x = counting(4, 1.0);
            const Wide poisoned(4, std::complex<double>(nan, nan));
            const std::complex<double> one(1.0, 0.0);
            const std::complex<double> zero(0.0, 0.0);
            const Wide expected = {{20.0, 0.0}, {7.0, 0.0}, {39.0, 0.0}, {50.0, 0.0}};

            for (const ExecutionMode mode : K_BOTH_MODES)
            {
                for (const Mirror mirror : {Mirror::Symmetric, Mirror::Hermitian})
                    this->expect_near(this->run(mode, mirror,
                                                this->sample_triangle(Core::Uplo::Lower),
                                                Core::IndexBase::Zero, Core::Uplo::Lower, one, x,
                                                zero, poisoned),
                                      expected, 0.0);
            }
        }

        TYPED_TEST(Unit_SparseSpsymvTest, SphemvConjugatesTheMirrorAndReadsARealDiagonal)
        {
            using T = typename TypeParam::Value;

            if (!Core::is_complex_v<T>)
                GTEST_SKIP() << "the case needs a complex element type";

            // The lower triangle (0,0) = 2 + 9i, (1,0) = 1 + 2i, (1,1) = 3, against x = (1, i).
            HostTriplets<T> lower;
            lower.rows = 2;
            lower.cols = 2;
            lower.row_indices = {0, 1, 1};
            lower.column_indices = {0, 0, 1};
            lower.values = {narrowed<T>({2.0, 9.0}), narrowed<T>({1.0, 2.0}),
                            narrowed<T>({3.0, 0.0})};

            const Wide x = {{1.0, 0.0}, {0.0, 1.0}};
            const Wide zeros(2, std::complex<double>(0.0, 0.0));
            const std::complex<double> one(1.0, 0.0);
            const std::complex<double> zero(0.0, 0.0);

            for (const ExecutionMode mode : K_BOTH_MODES)
            {
                for (const Core::IndexBase base : K_BOTH_BASES)
                {
                    this->expect_near(this->run(mode, Mirror::Hermitian, lower, base,
                                                Core::Uplo::Lower, one, x, zero, zeros),
                                      {{4.0, 1.0}, {1.0, 5.0}}, K_SAMPLE_TOLERANCE);
                    this->expect_near(this->run(mode, Mirror::Symmetric, lower, base,
                                                Core::Uplo::Lower, one, x, zero, zeros),
                                      {{0.0, 10.0}, {1.0, 5.0}}, K_SAMPLE_TOLERANCE);
                }
            }
        }

        TYPED_TEST(Unit_SparseSpsymvTest, OnARealTensorSphemvHoldsTheBitsOfSpsymv)
        {
            using T = typename TypeParam::Value;

            if (Core::is_complex_v<T>)
                GTEST_SKIP() << "the case needs a real element type";

            const Wide x = counting(4, 1.0);
            const Wide ones(4, std::complex<double>(1.0, 0.0));
            const std::complex<double> two(2.0, 0.0);
            const std::complex<double> minus_one(-1.0, 0.0);

            for (const ExecutionMode mode : K_BOTH_MODES)
            {
                for (const Core::Uplo uplo : K_BOTH_TRIANGLES)
                {
                    const auto triangle = this->sample_triangle(uplo);
                    const std::vector<T> symmetric =
                        this->run(mode, Mirror::Symmetric, triangle, Core::IndexBase::Zero, uplo,
                                  two, x, minus_one, ones);
                    const std::vector<T> hermitian =
                        this->run(mode, Mirror::Hermitian, triangle, Core::IndexBase::Zero, uplo,
                                  two, x, minus_one, ones);

                    ASSERT_EQ(symmetric.size(), hermitian.size());
                    EXPECT_EQ(std::memcmp(symmetric.data(), hermitian.data(),
                                          symmetric.size() * sizeof(T)),
                              0);
                }
            }
        }

        TYPED_TEST(Unit_SparseSpsymvTest, IntegerDataIsExactInBothModesForEitherTriangle)
        {
            using T = typename TypeParam::Value;

            const HostTriplets<T> lower = lower_part(integer_valued_matrix<T>(
                K_EXACT_ORDER, K_EXACT_ORDER, K_EXACT_PER_ROW, K_EXACT_SEED));
            const std::complex<double> one(1.0, 0.0);
            const std::complex<double> minus_one(-1.0, 0.0);

            // x cycles through -2, -1, 0, 1, 2 and y starts at 1.
            Wide x(static_cast<std::size_t>(K_EXACT_ORDER));
            for (std::size_t i = 0; i < x.size(); ++i)
                x[i] = std::complex<double>(static_cast<double>(i % 5) - 2.0, 0.0);
            const Wide y(x.size(), one);

            for (const Core::Uplo uplo : K_BOTH_TRIANGLES)
            {
                const HostTriplets<T> triangle =
                    uplo == Core::Uplo::Lower ? lower : transpose_of(lower);
                const Wide expected = host_mirrored_product(triangle, uplo, Mirror::Symmetric, one,
                                                            x, minus_one, y);

                for (const ExecutionMode mode : K_BOTH_MODES)
                    this->expect_near(this->run(mode, Mirror::Symmetric, triangle,
                                                Core::IndexBase::Zero, uplo, one, x, minus_one, y),
                                      expected, 0.0);
            }
        }

        TYPED_TEST(Unit_SparseSpsymvTest, AMatrixOfOrderZeroRecordsAndRuns)
        {
            using T = typename TypeParam::Value;

            const HostTriplets<T> empty;
            for (const ExecutionMode mode : K_BOTH_MODES)
            {
                this->engine->set_mode(mode);
                const SparseMatrix A = this->build(empty, Core::IndexBase::Zero);
                const Tensor x = this->engine->create_tensor({0}, get_sushiblas_dtype<T>());
                Tensor y = this->engine->create_tensor({0}, get_sushiblas_dtype<T>());

                SparseOps ops(*this->engine);
                EXPECT_NO_THROW(ops.spsymv(A, x, y, Core::Uplo::Lower));
                EXPECT_NO_THROW(ops.sphemv(A, x, y, Core::Uplo::Upper));
                EXPECT_NO_THROW(this->engine->execute());
            }
        }

        /**
         * @brief Runs the HALF product whose mirrored sum a HALF accumulator would round away.
         */
        class Unit_SparseSpsymvHalf : public Unit_SparseSpsymvTest<SparseCase<sycl::half, int64_t>>
        {

        };

        TEST_F(Unit_SparseSpsymvHalf, TheMirroredEntriesAreSummedInFloat32)
        {
            // Column 0 holds 2048, 1 and -2048 below the diagonal; their mirrors all land on y[0].
            HostTriplets<sycl::half> lower;
            lower.rows = 4;
            lower.cols = 4;
            lower.row_indices = {1, 2, 3};
            lower.column_indices = {0, 0, 0};
            lower.values = {sycl::half(2048.0f), sycl::half(1.0f), sycl::half(-2048.0f)};

            const std::complex<double> one(1.0, 0.0);
            const std::complex<double> zero(0.0, 0.0);
            const Wide ones(4, one);
            const Wide zeros(4, zero);
            const Wide expected = {one, {2048.0, 0.0}, one, {-2048.0, 0.0}};

            for (const ExecutionMode mode : K_BOTH_MODES)
                expect_near(run(mode, Mirror::Symmetric, lower, Core::IndexBase::Zero,
                                Core::Uplo::Lower, one, ones, zero, zeros),
                            expected, 0.0);
        }

        /**
         * @brief Calls the symmetric SYCL kernel with a block count the test chooses.
         */
        class Unit_SparseSpsymvKernel : public SushiBLASTest
        {
            protected:
                /** @brief Returns A * x from the SYCL kernel run with blocks row blocks. */
                [[nodiscard]] std::vector<float> mirrored_product(const HostTriplets<float>& lower,
                                                                  const std::vector<float>& x,
                                                                  std::size_t blocks)
                {
                    const Core::IndexBase base = Core::IndexBase::Zero;
                    const SparseBuilder builder(engine->get_tensor_factory());
                    const SparseMatrix A = builder.matrix_from_triplets(
                        lower.rows, lower.cols, stored_indices<int32_t>(lower.row_indices, base),
                        stored_indices<int32_t>(lower.column_indices, base), lower.values);
                    const Tensor tx = create_initialized_tensor<float>({lower.rows}, x);
                    Tensor ty = engine->create_tensor({lower.rows});
                    Tensor partials =
                        engine->create_tensor({static_cast<int64_t>(blocks) * lower.rows});

                    const Kernels::CsrPattern<int32_t> pattern{
                        A.rows, A.cols, A.nnz(), 0, A.row_offsets.data_as<int32_t>(),
                        A.column_indices.data_as<int32_t>()};
                    Kernels::SYCL::spsymv<float, int32_t>(
                        TestEnvironment::get_context().devices().queue(0), Core::Uplo::Lower, pattern,
                        A.values.data_as<float>(), 1.0f, tx.data_as<float>(), 1, 0.0f,
                        ty.data_as<float>(), 1, partials.data_as<float>(), blocks, {}).wait();
                    return read_tensor<float>(ty);
                }
        };

        TEST_F(Unit_SparseSpsymvKernel, EveryBlockCountGivesTheExactIntegerResult)
        {
            const HostTriplets<float> lower = lower_part(integer_valued_matrix<float>(
                K_EXACT_ORDER, K_EXACT_ORDER, K_EXACT_PER_ROW, K_EXACT_SEED));
            std::vector<float> x(static_cast<std::size_t>(lower.rows));
            Wide wide_x(x.size());
            for (std::size_t i = 0; i < x.size(); ++i)
            {
                x[i] = static_cast<float>(i % 5) - 2.0f;
                wide_x[i] = std::complex<double>(static_cast<double>(x[i]), 0.0);
            }

            const std::complex<double> one(1.0, 0.0);
            const std::complex<double> zero(0.0, 0.0);
            const Wide expected =
                host_mirrored_product(lower, Core::Uplo::Lower, Mirror::Symmetric, one, wide_x,
                                      zero, Wide(x.size(), zero));

            const std::size_t block_counts[6] = {1, 2, 3, 7, 64, 256};
            for (const std::size_t blocks : block_counts)
            {
                const std::vector<float> actual = mirrored_product(lower, x, blocks);
                ASSERT_EQ(actual.size(), expected.size());
                for (std::size_t i = 0; i < actual.size(); ++i)
                    EXPECT_EQ(static_cast<double>(actual[i]), expected[i].real())
                        << "row " << i << " with " << blocks << " blocks";
            }
        }
    } // namespace Test
} // namespace SushiBLAS
