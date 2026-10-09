/**************************************************************************/
/* test_sparse_spmv.cpp                                                   */
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
 * @file test_sparse_spmv.cpp
 * @brief Tests SparseOps::spmv and the transposed SYCL kernel against host products.
 * @author Mustafa Garip
 */

#include <array>
#include <cmath>
#include <limits>
#include <vector>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <SushiBLAS/SushiBLAS.h>
#include <SushiBLAS/support/device_policy.hpp>
#include <SushiBLAS/sparse_kernels/sparse.hpp>
#include "test_utils.hpp"
#include "sparse_samples.hpp"

namespace SushiBLAS
{
    namespace Test
    {
        /// @brief Absolute error allowed on an element of the 4 x 5 sample in single precision.
        constexpr double K_SAMPLE_TOLERANCE = 1e-4;

        /// @brief Rows of the integer-valued matrix; above one work-group on every device.
        constexpr int64_t K_EXACT_ROWS = 1000;

        /// @brief Columns of the integer-valued matrix; not equal to its rows.
        constexpr int64_t K_EXACT_COLS = 777;

        /// @brief Entries drawn for each row of the integer-valued matrix.
        constexpr int64_t K_EXACT_PER_ROW = 8;

        /// @brief Seed of the integer-valued matrix.
        constexpr std::uint32_t K_EXACT_SEED = 20261007u;

        /// @brief The three values of Core::Transpose.
        constexpr Core::Transpose K_EVERY_TRANSPOSE[3] = {
            Core::Transpose::NoTrans, Core::Transpose::Trans, Core::Transpose::ConjTrans};

        /// @brief The value the elements between those of a strided view hold.
        constexpr double K_FILLER = -7.0;

        /// @brief Empty rows before the first stored entry of the block-partition matrix.
        constexpr int64_t K_LEADING_EMPTY_ROWS = 3;

        /// @brief Rows of the block-partition matrix that store entries.
        constexpr int64_t K_STORED_ROWS = 100;

        /// @brief The stored row of the block-partition matrix that holds K_HEAVY_ENTRIES.
        constexpr int64_t K_HEAVY_ROW = 50;

        /// @brief Entries of the heavy row; every other stored row holds one.
        constexpr int64_t K_HEAVY_ENTRIES = 900;

        /// @brief Empty rows after the last stored entry of the block-partition matrix.
        constexpr int64_t K_TRAILING_EMPTY_ROWS = 2;

        /**
         * @brief Builds CSR matrices from host triplets and runs spmv in a chosen execution mode.
         */
        template<typename TypeParam>
        class Unit_SparseSpmvTest : public SushiBLASTest
        {
            protected:
                /** @brief The element type under test. */
                using T = typename TypeParam::Value;

                /** @brief The index type under test. */
                using Index = typename TypeParam::IndexType;

                /** @brief A host vector of expectations or of operand values. */
                using Wide = std::vector<std::complex<double>>;

                /** @brief Skips the HALF suite on a device without the fp16 aspect. */
                void SetUp() override
                {
                    SushiBLASTest::SetUp();
                    if (std::is_same_v<T, sycl::half> &&
                        !TestEnvironment::get_context().get_queue().get_device().has(
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

                /** @brief Returns a rank-1 tensor of T that holds no element. */
                [[nodiscard]] Tensor vector_without_elements()
                {
                    return engine->create_tensor({0}, get_sushiblas_dtype<T>());
                }

                /**
                 * @brief Returns a CSR matrix whose three arrays the caller gives as they are.
                 * @param offsets Zero-based; rows + 1 of them.
                 * @param columns Zero-based; a (row, column) pair may be given twice.
                 */
                [[nodiscard]] SparseMatrix matrix_of_arrays(int64_t rows, int64_t cols,
                                                            const std::vector<int64_t>& offsets,
                                                            const std::vector<int64_t>& columns,
                                                            const Wide& values,
                                                            Core::IndexBase base)
                {
                    SparseMatrix A;
                    A.rows = rows;
                    A.cols = cols;
                    A.index_base = base;
                    A.row_offsets = create_initialized_tensor<Index>(
                        {static_cast<int64_t>(offsets.size())},
                        stored_indices<Index>(offsets, base));
                    A.column_indices = create_initialized_tensor<Index>(
                        {static_cast<int64_t>(columns.size())},
                        stored_indices<Index>(columns, base));
                    A.values = vector_of(values);
                    return A;
                }

                /** @brief Returns a tensor of twice the length: values at its even elements. */
                [[nodiscard]] Tensor interleaved(const Wide& values)
                {
                    Wide spread(2 * values.size(), std::complex<double>(K_FILLER, 0.0));
                    for (std::size_t i = 0; i < values.size(); ++i)
                        spread[2 * i] = values[i];
                    return vector_of(spread);
                }

                /** @brief Returns the stride-2 view of the even elements of a tensor. */
                [[nodiscard]] static Tensor every_second(const Tensor& wide)
                {
                    const int64_t dims[1] = {wide.num_elements / 2};
                    std::array<int64_t, Core::K_MAX_TENSOR_RANK> strides{};
                    strides[0] = 2;
                    return Tensor(wide.storage, SushiRuntime::span<const int64_t>(dims, 1),
                                  strides, wide.dtype, 0, wide.layout);
                }

                /** @brief Returns y after y = alpha * op(A) * x + beta * y recorded in mode. */
                [[nodiscard]] std::vector<T> run(ExecutionMode mode, const HostTriplets<T>& host,
                                                 Core::IndexBase base, Core::Transpose trans,
                                                 std::complex<double> alpha, const Wide& x,
                                                 std::complex<double> beta, const Wide& y)
                {
                    engine->set_mode(mode);
                    const SparseMatrix A = build(host, base);
                    const Tensor tx = vector_of(x);
                    Tensor ty = vector_of(y);
                    SushiBLAS::SparseOps(*engine).spmv(A, tx, ty, trans, BlasScalar(alpha),
                                                       BlasScalar(beta));
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

                /**
                 * @brief Expects the even elements near expected and the odd ones still K_FILLER.
                 * @param wide The elements of a tensor interleaved() made, after the call.
                 */
                static void expect_interleaved(const std::vector<T>& wide, const Wide& expected,
                                               double tolerance)
                {
                    ASSERT_EQ(wide.size(), 2 * expected.size());
                    std::vector<T> written(expected.size());
                    std::vector<T> skipped(expected.size());
                    for (std::size_t i = 0; i < expected.size(); ++i)
                    {
                        written[i] = wide[2 * i];
                        skipped[i] = wide[2 * i + 1];
                    }

                    expect_near(written, expected, tolerance);
                    expect_near(skipped,
                                Wide(expected.size(), std::complex<double>(K_FILLER, 0.0)), 0.0);
                }
        };

        SB_TYPED_TEST_SUITE(Unit_SparseSpmvTest, SparseCases);

        TYPED_TEST(Unit_SparseSpmvTest, TheSampleProductMatchesTheHostInEveryModeAndBase)
        {
            using T = typename TypeParam::Value;
            using Wide = std::vector<std::complex<double>>;

            const HostTriplets<T> host = sample_matrix<T>();
            const Wide x = counting(5, 1.0);
            const Wide y(4, std::complex<double>(1.0, 0.0));
            const std::complex<double> alpha(2.0, 0.0);
            const std::complex<double> beta(-1.0, 0.0);
            const Core::Transpose trans = Core::Transpose::NoTrans;

            const Wide expected = host_product(host, trans, alpha, x, beta, y);
            if (!Core::is_complex_v<T>)
            {
                // A * x is (9, 0, 43, 41) for x = (1, 2, 3, 4, 5).
                const double by_hand[4] = {17.0, -1.0, 85.0, 81.0};
                for (std::size_t i = 0; i < 4; ++i)
                    ASSERT_EQ(expected[i].real(), by_hand[i]);
            }

            for (const ExecutionMode mode : K_BOTH_MODES)
            {
                for (const Core::IndexBase base : K_BOTH_BASES)
                    this->expect_near(this->run(mode, host, base, trans, alpha, x, beta, y),
                                      expected, K_SAMPLE_TOLERANCE);
            }
        }

        TYPED_TEST(Unit_SparseSpmvTest, TheTransposedSampleProductMatchesTheHost)
        {
            using T = typename TypeParam::Value;
            using Wide = std::vector<std::complex<double>>;

            const HostTriplets<T> host = sample_matrix<T>();
            const Wide x = counting(4, 1.0);
            const Wide y(5, std::complex<double>(0.0, 0.0));
            const std::complex<double> one(1.0, 0.0);
            const std::complex<double> zero(0.0, 0.0);
            const Core::Transpose trans = Core::Transpose::Trans;

            const Wide expected = host_product(host, trans, one, x, zero, y);
            if (!Core::is_complex_v<T>)
            {
                const double by_hand[5] = {25.0, 9.0, 12.0, 2.0, 43.0};
                for (std::size_t i = 0; i < 5; ++i)
                    ASSERT_EQ(expected[i].real(), by_hand[i]);
            }

            for (const ExecutionMode mode : K_BOTH_MODES)
            {
                for (const Core::IndexBase base : K_BOTH_BASES)
                    this->expect_near(this->run(mode, host, base, trans, one, x, zero, y),
                                      expected, K_SAMPLE_TOLERANCE);
            }
        }

        TYPED_TEST(Unit_SparseSpmvTest, ConjTransConjugatesAComplexMatrixAndNothingElse)
        {
            using T = typename TypeParam::Value;
            using Wide = std::vector<std::complex<double>>;

            const HostTriplets<T> host = sample_matrix<T>();
            const Wide y(5, std::complex<double>(0.0, 0.0));
            const std::complex<double> one(1.0, 0.0);
            const std::complex<double> zero(0.0, 0.0);
            const Core::IndexBase base = Core::IndexBase::Zero;

            // x selects row 0, whose entries are 1 + 2i in column 0 and 2 - i in column 3.
            Wide x(4, zero);
            x[0] = Core::is_complex_v<T> ? std::complex<double>(1.0, 1.0) : one;

            for (const ExecutionMode mode : K_BOTH_MODES)
            {
                const std::vector<T> plain =
                    this->run(mode, host, base, Core::Transpose::Trans, one, x, zero, y);
                const std::vector<T> conjugated =
                    this->run(mode, host, base, Core::Transpose::ConjTrans, one, x, zero, y);

                if (Core::is_complex_v<T>)
                {
                    this->expect_near(plain, {{-1.0, 3.0}, zero, zero, {3.0, 1.0}, zero},
                                      K_SAMPLE_TOLERANCE);
                    this->expect_near(conjugated, {{3.0, -1.0}, zero, zero, {1.0, 3.0}, zero},
                                      K_SAMPLE_TOLERANCE);
                }
                else
                {
                    this->expect_near(plain, {one, zero, zero, {2.0, 0.0}, zero}, 0.0);
                    this->expect_near(conjugated, {one, zero, zero, {2.0, 0.0}, zero}, 0.0);
                }
            }
        }

        TYPED_TEST(Unit_SparseSpmvTest, IntegerDataIsExactInBothModesForEveryTranspose)
        {
            using T = typename TypeParam::Value;
            using Wide = std::vector<std::complex<double>>;

            const HostTriplets<T> host = integer_valued_matrix<T>(K_EXACT_ROWS, K_EXACT_COLS,
                                                                  K_EXACT_PER_ROW, K_EXACT_SEED);
            const std::complex<double> one(1.0, 0.0);
            const std::complex<double> minus_one(-1.0, 0.0);
            const Core::IndexBase base = Core::IndexBase::Zero;

            for (const Core::Transpose trans : K_EVERY_TRANSPOSE)
            {
                const bool transposed = Core::is_transposed(trans);
                const int64_t inputs = transposed ? host.rows : host.cols;
                const int64_t outputs = transposed ? host.cols : host.rows;

                // x cycles through -2, -1, 0, 1, 2 and y starts at 1.
                Wide x(static_cast<std::size_t>(inputs));
                for (std::size_t i = 0; i < x.size(); ++i)
                    x[i] = std::complex<double>(static_cast<double>(i % 5) - 2.0, 0.0);
                const Wide y(static_cast<std::size_t>(outputs), one);

                const Wide expected = host_product(host, trans, one, x, minus_one, y);
                for (const ExecutionMode mode : K_BOTH_MODES)
                    this->expect_near(this->run(mode, host, base, trans, one, x, minus_one, y),
                                      expected, 0.0);
            }
        }

        TYPED_TEST(Unit_SparseSpmvTest, ABetaOfZeroDoesNotReadY)
        {
            using T = typename TypeParam::Value;
            using Wide = std::vector<std::complex<double>>;

            const HostTriplets<T> host = sample_matrix<T>();
            const std::complex<double> one(1.0, 0.0);
            const std::complex<double> zero(0.0, 0.0);
            const double nan = std::numeric_limits<double>::quiet_NaN();
            const Core::Transpose both[2] = {Core::Transpose::NoTrans, Core::Transpose::Trans};

            for (const Core::Transpose trans : both)
            {
                const bool transposed = Core::is_transposed(trans);
                const std::size_t inputs = transposed ? 4 : 5;
                const std::size_t outputs = transposed ? 5 : 4;

                const Wide x = counting(inputs, 1.0);
                const Wide poisoned(outputs, std::complex<double>(nan, nan));
                const Wide expected = host_product(host, trans, one, x, zero, Wide(outputs, zero));

                for (const ExecutionMode mode : K_BOTH_MODES)
                    this->expect_near(this->run(mode, host, Core::IndexBase::Zero, trans, one, x,
                                                zero, poisoned),
                                      expected, K_SAMPLE_TOLERANCE);
            }
        }

        TYPED_TEST(Unit_SparseSpmvTest, AMatrixWithoutEntriesScalesYByBeta)
        {
            using T = typename TypeParam::Value;
            using Wide = std::vector<std::complex<double>>;

            HostTriplets<T> host;
            host.rows = 3;
            host.cols = 4;
            const std::complex<double> alpha(3.0, 0.0);
            const std::complex<double> beta(-2.0, 0.0);

            for (const Core::Transpose trans : K_EVERY_TRANSPOSE)
            {
                const bool transposed = Core::is_transposed(trans);
                const Wide x = counting(transposed ? 3 : 4, 1.0);
                const Wide y = counting(transposed ? 4 : 3, 1.0);

                const Wide expected = host_product(host, trans, alpha, x, beta, y);
                for (const ExecutionMode mode : K_BOTH_MODES)
                {
                    for (const Core::IndexBase base : K_BOTH_BASES)
                        this->expect_near(this->run(mode, host, base, trans, alpha, x, beta, y),
                                          expected, 0.0);
                }
            }
        }

        TYPED_TEST(Unit_SparseSpmvTest, AMatrixWithoutRowsOrColumnsScalesItsOtherSide)
        {
            using T = typename TypeParam::Value;
            using Wide = std::vector<std::complex<double>>;

            HostTriplets<T> no_rows;
            no_rows.cols = 5;
            HostTriplets<T> no_columns;
            no_columns.rows = 5;

            const BlasScalar alpha = 3.0;
            const BlasScalar beta = -2.0;
            const Wide five = counting(5, 1.0);
            const Wide scaled = {{-2.0, 0.0}, {-4.0, 0.0}, {-6.0, 0.0}, {-8.0, 0.0}, {-10.0, 0.0}};

            for (const ExecutionMode mode : K_BOTH_MODES)
            {
                this->engine->set_mode(mode);
                const SparseMatrix wide_matrix = this->build(no_rows, Core::IndexBase::Zero);
                const SparseMatrix tall_matrix = this->build(no_columns, Core::IndexBase::One);

                // The side of length 5 is y under TRANS for 0 x 5 and under NO_TRANS for 5 x 0.
                const Tensor nothing = this->vector_without_elements();
                Tensor no_result = this->vector_without_elements();
                const Tensor x = this->vector_of(five);
                Tensor y_wide = this->vector_of(five);
                Tensor y_tall = this->vector_of(five);

                SparseOps ops(*this->engine);
                ops.spmv(wide_matrix, x, no_result, Core::Transpose::NoTrans, alpha, beta);
                ops.spmv(wide_matrix, nothing, y_wide, Core::Transpose::Trans, alpha, beta);
                ops.spmv(tall_matrix, nothing, y_tall, Core::Transpose::NoTrans, alpha, beta);
                ops.spmv(tall_matrix, x, no_result, Core::Transpose::Trans, alpha, beta);
                this->engine->execute();

                this->expect_near(read_tensor<T>(y_wide), scaled, 0.0);
                this->expect_near(read_tensor<T>(y_tall), scaled, 0.0);
            }
        }

        TYPED_TEST(Unit_SparseSpmvTest, StridedVectorsGiveTheSampleProduct)
        {
            using T = typename TypeParam::Value;
            using Wide = std::vector<std::complex<double>>;

            const HostTriplets<T> host = sample_matrix<T>();
            const Wide x = counting(5, 1.0);
            const Wide y(4, std::complex<double>(1.0, 0.0));
            const std::complex<double> alpha(2.0, 0.0);
            const std::complex<double> beta(-1.0, 0.0);
            const Core::Transpose trans = Core::Transpose::NoTrans;
            const Wide expected = host_product(host, trans, alpha, x, beta, y);

            for (const ExecutionMode mode : K_BOTH_MODES)
            {
                for (const Core::IndexBase base : K_BOTH_BASES)
                {
                    this->engine->set_mode(mode);
                    const SparseMatrix A = this->build(host, base);
                    const Tensor wide_x = this->interleaved(x);
                    const Tensor wide_y = this->interleaved(y);
                    const Tensor tx = this->every_second(wide_x);
                    Tensor ty = this->every_second(wide_y);

                    SparseOps(*this->engine).spmv(A, tx, ty, trans, BlasScalar(alpha),
                                                  BlasScalar(beta));
                    this->engine->execute();

                    this->expect_interleaved(read_tensor<T>(wide_y), expected,
                                             K_SAMPLE_TOLERANCE);
                }
            }
        }

        TYPED_TEST(Unit_SparseSpmvTest, APairStoredTwiceCountsTwice)
        {
            using T = typename TypeParam::Value;
            using Wide = std::vector<std::complex<double>>;

            // Row 0 stores column 1 twice, as 2 and as 3; row 1 stores 4 in column 2.
            const std::vector<int64_t> offsets = {0, 2, 3};
            const std::vector<int64_t> columns = {1, 1, 2};
            const Wide values = {{2.0, 0.0}, {3.0, 0.0}, {4.0, 0.0}};
            const std::complex<double> zero(0.0, 0.0);

            for (const ExecutionMode mode : K_BOTH_MODES)
            {
                for (const Core::IndexBase base : K_BOTH_BASES)
                {
                    this->engine->set_mode(mode);
                    const SparseMatrix A =
                        this->matrix_of_arrays(2, 3, offsets, columns, values, base);
                    const Tensor by_columns =
                        this->vector_of({{1.0, 0.0}, {10.0, 0.0}, {100.0, 0.0}});
                    const Tensor by_rows = this->vector_of({{1.0, 0.0}, {10.0, 0.0}});
                    Tensor product = this->vector_of(Wide(2, zero));
                    Tensor transposed = this->vector_of(Wide(3, zero));

                    SparseOps ops(*this->engine);
                    ops.spmv(A, by_columns, product);
                    ops.spmv(A, by_rows, transposed, Core::Transpose::Trans);
                    this->engine->execute();

                    this->expect_near(read_tensor<T>(product), {{50.0, 0.0}, {400.0, 0.0}}, 0.0);
                    this->expect_near(read_tensor<T>(transposed),
                                      {zero, {5.0, 0.0}, {40.0, 0.0}}, 0.0);
                }
            }
        }

        /**
         * @brief Runs the HALF products whose sum a HALF accumulator would round away.
         */
        class Unit_SparseSpmvHalf : public Unit_SparseSpmvTest<SparseCase<sycl::half, int64_t>>
        {

        };

        TEST_F(Unit_SparseSpmvHalf, ARowAndAColumnAreSummedInFloat32)
        {
            const std::complex<double> one(1.0, 0.0);
            const std::complex<double> zero(0.0, 0.0);
            const Wide ones(3, one);
            const Wide start(1, zero);

            // 2048 + 1 rounds back to 2048 in HALF, so a HALF sum of the three entries is 0.
            HostTriplets<sycl::half> row;
            row.rows = 1;
            row.cols = 3;
            row.row_indices = {0, 0, 0};
            row.column_indices = {0, 1, 2};
            row.values = {sycl::half(2048.0f), sycl::half(1.0f), sycl::half(-2048.0f)};

            HostTriplets<sycl::half> column;
            column.rows = 3;
            column.cols = 1;
            column.row_indices = {0, 1, 2};
            column.column_indices = {0, 0, 0};
            column.values = row.values;

            for (const ExecutionMode mode : K_BOTH_MODES)
            {
                expect_near(run(mode, row, Core::IndexBase::Zero, Core::Transpose::NoTrans, one,
                                ones, zero, start),
                            Wide(1, one), 0.0);
                expect_near(run(mode, column, Core::IndexBase::Zero, Core::Transpose::Trans, one,
                                ones, zero, start),
                            Wide(1, one), 0.0);
            }
        }

        /**
         * @brief Calls the transposed SYCL kernel with a block count the test chooses.
         */
        class Unit_SparseSpmvKernel : public SushiBLASTest
        {
            protected:
                /** @brief Returns A^T * x from the SYCL kernel run with blocks row blocks. */
                [[nodiscard]] std::vector<float> transposed_product(const HostTriplets<float>& host,
                                                                    const std::vector<float>& x,
                                                                    std::size_t blocks)
                {
                    const Core::IndexBase base = Core::IndexBase::Zero;
                    const SparseBuilder builder(engine->get_tensor_factory());
                    const SparseMatrix A = builder.matrix_from_triplets(
                        host.rows, host.cols, stored_indices<int32_t>(host.row_indices, base),
                        stored_indices<int32_t>(host.column_indices, base), host.values);
                    const Tensor tx = create_initialized_tensor<float>({host.rows}, x);
                    Tensor ty = engine->create_tensor({host.cols});
                    Tensor partials =
                        engine->create_tensor({static_cast<int64_t>(blocks) * host.cols});

                    const Kernels::CsrPattern<int32_t> pattern{
                        A.rows, A.cols, A.nnz(), 0, A.row_offsets.data_as<int32_t>(),
                        A.column_indices.data_as<int32_t>()};
                    Kernels::SYCL::spmv<float, int32_t>(
                        TestEnvironment::get_context().get_queue(), Core::Transpose::Trans, pattern,
                        A.values.data_as<float>(), 1.0f, tx.data_as<float>(), 1, 0.0f,
                        ty.data_as<float>(), 1, partials.data_as<float>(), blocks, {}).wait();
                    return read_tensor<float>(ty);
                }
        };

        TEST_F(Unit_SparseSpmvKernel, EveryBlockCountGivesTheExactIntegerResult)
        {
            using Wide = std::vector<std::complex<double>>;

            const HostTriplets<float> host = integer_valued_matrix<float>(
                K_EXACT_ROWS, K_EXACT_COLS, K_EXACT_PER_ROW, K_EXACT_SEED);
            std::vector<float> x(static_cast<std::size_t>(host.rows));
            Wide wide_x(x.size());
            for (std::size_t i = 0; i < x.size(); ++i)
            {
                x[i] = static_cast<float>(i % 5) - 2.0f;
                wide_x[i] = std::complex<double>(static_cast<double>(x[i]), 0.0);
            }

            const std::complex<double> one(1.0, 0.0);
            const std::complex<double> zero(0.0, 0.0);
            const Wide expected = host_product(host, Core::Transpose::Trans, one, wide_x, zero,
                                               Wide(static_cast<std::size_t>(host.cols), zero));

            const std::size_t block_counts[6] = {1, 2, 3, 7, 64, 256};
            for (const std::size_t blocks : block_counts)
            {
                const std::vector<float> actual = transposed_product(host, x, blocks);
                ASSERT_EQ(actual.size(), expected.size());
                for (std::size_t j = 0; j < actual.size(); ++j)
                    EXPECT_EQ(static_cast<double>(actual[j]), expected[j].real())
                        << "column " << j << " with " << blocks << " blocks";
            }
        }

        TEST_F(Unit_SparseSpmvKernel, TheRowBlocksCoverEveryEntryOnce)
        {
            namespace Blocks = Kernels::SYCL::Detail;

            // Three empty rows, a hundred stored rows of which one holds 900, two empty rows.
            std::vector<int32_t> offsets(static_cast<std::size_t>(K_LEADING_EMPTY_ROWS) + 1, 0);
            for (int64_t row = 0; row < K_STORED_ROWS; ++row)
                offsets.push_back(offsets.back() +
                                  static_cast<int32_t>(row == K_HEAVY_ROW ? K_HEAVY_ENTRIES : 1));
            offsets.insert(offsets.end(), static_cast<std::size_t>(K_TRAILING_EMPTY_ROWS),
                           offsets.back());

            const int64_t rows = static_cast<int64_t>(offsets.size()) - 1;
            const int64_t nnz = offsets.back();
            const std::vector<int32_t> columns(static_cast<std::size_t>(nnz), 0);
            const Kernels::CsrPattern<int32_t> pattern{rows, 1, nnz, 0, offsets.data(),
                                                       columns.data()};
            ASSERT_EQ(nnz, K_STORED_ROWS - 1 + K_HEAVY_ENTRIES);

            const int64_t block_counts[4] = {1, 2, 7, 256};
            for (const int64_t blocks : block_counts)
            {
                SCOPED_TRACE(blocks);
                EXPECT_EQ(Blocks::block_first_row(pattern, 0, blocks), 0);
                EXPECT_EQ(Blocks::block_end_row(pattern, blocks - 1, blocks), rows);

                int64_t previous_first = 0;
                int64_t covered = 0;
                for (int64_t block = 0; block < blocks; ++block)
                {
                    const int64_t first = Blocks::block_first_row(pattern, block, blocks);
                    const int64_t end = Blocks::block_end_row(pattern, block, blocks);
                    ASSERT_GE(first, previous_first) << "block " << block;
                    ASSERT_GE(end, first) << "block " << block;
                    ASSERT_LE(end, rows) << "block " << block;

                    covered += offsets[static_cast<std::size_t>(end)] -
                               offsets[static_cast<std::size_t>(first)];
                    previous_first = first;
                }
                EXPECT_EQ(covered, nnz);
            }
        }

        TEST_F(Unit_SparseSpmvKernel, TheBlockCountFollowsTheProfileAndTheDensity)
        {
            using Kernels::SYCL::scatter_blocks;

            Device::Profile cpu;
            cpu.is_cpu = true;
            cpu.compute_units = 12;

            Device::Profile gpu;
            gpu.is_cpu = false;
            gpu.compute_units = 4;

            Device::Profile large_gpu;
            large_gpu.is_cpu = false;
            large_gpu.compute_units = 64;

            EXPECT_EQ(scatter_blocks(cpu, 0, 100), 0u);
            EXPECT_EQ(scatter_blocks(cpu, 10, 0), 1u);
            EXPECT_EQ(scatter_blocks(cpu, 10, 1000), 12u);
            EXPECT_EQ(scatter_blocks(gpu, 10, 1000), 4u * Device::K_WAVES_PER_COMPUTE_UNIT);

            // 2 * 250 / 100 partial sums per output are affordable, below the 12 compute units.
            EXPECT_EQ(scatter_blocks(cpu, 100, 250), 5u);
            EXPECT_EQ(scatter_blocks(large_gpu, 10, 100000), Kernels::SYCL::K_MAX_SCATTER_BLOCKS);
        }
    } // namespace Test
} // namespace SushiBLAS
