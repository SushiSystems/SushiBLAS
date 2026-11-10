/**************************************************************************/
/* test_level2_sycl_parity.cpp                                            */
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
 * @file test_level2_sycl_parity.cpp
 * @brief Tests that the GEMV and SYMV kernels stay within their rounding bound of a
 *        double-precision truth, and that the SYCL kernels repeat to the bit.
 * @author Mustafa Garip
 */

#include <cmath>
#include <string>
#include <cstdint>
#include <cstring>
#include <vector>
#include <cstddef>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.hpp>
#include <SushiBLAS/arithmetic/accumulator.hpp>
#include <SushiBLAS/reference/reference_accumulator.hpp>
#include <SushiBLAS/engine_fixture/sushiblas_test.hpp>
#include <SushiBLAS/tensor_expectations/scalar_dtype.hpp>
#include <SushiBLAS/test_runner/test_environment.hpp>
#include <SushiBLAS/blas_test_oracle/blas_flags.hpp>
#include <SushiBLAS/blas_test_oracle/blas_truth.hpp>
#include "blas_rows.hpp"

namespace SushiBLAS
{
    namespace Test
    {
        /// @brief Edge of the small square case; below every work-group-per-row crossover.
        constexpr int64_t K_SMALL_N = 100;

        /// @brief Edge of the large square case.
        constexpr int64_t K_LARGE_N = 4096;

        /// @brief Rows of the wide GEMV case.
        constexpr int64_t K_WIDE_ROWS = 4;

        /// @brief Columns of the wide GEMV case; at the CPU two-stage reduction threshold.
        constexpr int64_t K_WIDE_COLS = 16384;

        /// @brief Index of a data set at which the elements of A start.
        constexpr std::size_t K_SEED_A = 0;

        /// @brief Index of a data set at which the elements of x start.
        constexpr std::size_t K_SEED_X = 11;

        /// @brief Index of a data set at which the elements of y start.
        constexpr std::size_t K_SEED_Y = 23;

        /// @brief Roundings a kernel makes whatever the length, and two of the truth itself.
        constexpr std::size_t K_FIXED_ROUNDINGS = 6;

        /// @brief Roundings counted for a compensated sum, which does not grow with the length.
        constexpr std::size_t K_COMPENSATED_ROUNDINGS = 2;

        /**
         * @brief Returns a fixed, sign-mixed value in [-1, 1] for index i.
         */
        [[nodiscard]] inline double pattern_value(std::size_t i) noexcept
        {
            return static_cast<double>(static_cast<int64_t>((i * 37u) % 101u) - 50) / 50.0;
        }

        /**
         * @brief Returns a fixed integer in [-3, 3] for index i.
         */
        [[nodiscard]] inline double integer_value(std::size_t i) noexcept
        {
            return static_cast<double>(static_cast<int64_t>((i * 37u) % 7u) - 3);
        }

        /** @brief Holds the values a case fills its operands with and the scalars of its call. */
        struct ParityData
        {
            double (*value)(std::size_t) noexcept; /**< The value at an index. */
            double alpha;                          /**< Scales op(A) * x. */
            double beta;                           /**< Scales the prior y. */
        };

        /// @brief Rounded data: sign-mixed fractions, alpha 1.5 and beta -0.5.
        constexpr ParityData K_PATTERNED{&pattern_value, 1.5, -0.5};

        /// @brief Exact data: every partial sum of every case is an integer below 2^24.
        constexpr ParityData K_INTEGER_VALUED{&integer_value, 2.0, -1.0};

        /** @brief Names the product a row runs; a SYMV reads one triangle of A. */
        enum class ParityProduct
        {
            Gemv,      /**< y = alpha * A * x + beta * y for an m x n A. */
            GemvTransposed, /**< y = alpha * A^T * x + beta * y for an m x n A: n sums of m terms. */
            SymvUpper, /**< The same for a symmetric A of order n, read from its upper triangle. */
            SymvLower  /**< The same, read from its lower triangle. */
        };

        /** @brief Holds a product and the extent of its A; a SYMV takes m equal to n. */
        struct ParityShape
        {
            ParityProduct product; /**< The product the row runs. */
            int64_t m;             /**< The rows of A. */
            int64_t n;             /**< The columns of A, and the terms of every sum. */
        };

        /** @brief Holds one row: the element type, the kernel and the shape. */
        struct ParityRow
        {
            Core::DataType type; /**< FLOAT32 or FLOAT64. */
            ExecutionMode mode;  /**< The kernel the engine selects. */
            ParityShape shape;   /**< The product and its extent. */
        };

        /** @brief Returns every row over the two element types, the given kernels and shapes. */
        [[nodiscard]] inline std::vector<ParityRow> parity_rows(
            const std::vector<ExecutionMode>& modes, const std::vector<ParityShape>& shapes)
        {
            std::vector<ParityRow> rows;
            for (const Core::DataType type : {Core::DataType::Float32, Core::DataType::Float64})
                for (const ExecutionMode mode : modes)
                    for (const ParityShape& shape : shapes)
                        rows.push_back({type, mode, shape});
            return rows;
        }

        /** @brief Returns the word that names a product in a row name. */
        [[nodiscard]] inline const char* product_name(ParityProduct product) noexcept
        {
            switch (product)
            {
                case ParityProduct::Gemv:
                    return "Gemv";
                case ParityProduct::GemvTransposed:
                    return "GemvTransposed";
                case ParityProduct::SymvUpper:
                    return "SymvUpper";
                case ParityProduct::SymvLower:
                    break;
            }
            return "SymvLower";
        }

        /** @brief Returns the name of a row: type, kernel, product and extent. */
        [[nodiscard]] inline std::string parity_row_name(
            const ::testing::TestParamInfo<ParityRow>& info)
        {
            const ParityShape& shape = info.param.shape;
            return std::string(Core::to_string(info.param.type)) + "_" + word(info.param.mode) + "_" +
                   product_name(shape.product) +
                   "_" + std::to_string(shape.m) + "x" + std::to_string(shape.n);
        }

        /// @brief The kernels a truth case runs in.
        const std::vector<ExecutionMode> K_TRUTH_MODES = {ExecutionMode::ForceReference,
                                                          ExecutionMode::ForceSYCL};

        /// @brief The shapes the bound is measured at.
        const std::vector<ParityShape> K_BOUND_SHAPES = {
            {ParityProduct::Gemv, K_SMALL_N, K_SMALL_N},
            {ParityProduct::Gemv, K_LARGE_N, K_LARGE_N},
            {ParityProduct::Gemv, K_WIDE_ROWS, K_WIDE_COLS},
            {ParityProduct::SymvUpper, K_SMALL_N, K_SMALL_N},
            {ParityProduct::SymvLower, K_SMALL_N, K_SMALL_N},
            {ParityProduct::SymvUpper, K_LARGE_N, K_LARGE_N},
            {ParityProduct::SymvLower, K_LARGE_N, K_LARGE_N}};

        /// @brief The shapes that are exact on small integers.
        const std::vector<ParityShape> K_EXACT_SHAPES = {
            {ParityProduct::Gemv, K_LARGE_N, K_LARGE_N},
            {ParityProduct::Gemv, K_WIDE_ROWS, K_WIDE_COLS},
            {ParityProduct::GemvTransposed, K_WIDE_COLS, K_WIDE_ROWS},
            {ParityProduct::SymvUpper, K_LARGE_N, K_LARGE_N},
            {ParityProduct::SymvLower, K_LARGE_N, K_LARGE_N}};

        /// @brief The shapes that repeat to the bit: the two-stage reduction and the large SYMV.
        const std::vector<ParityShape> K_REPEAT_SHAPES = {
            {ParityProduct::Gemv, K_WIDE_ROWS, K_WIDE_COLS},
            {ParityProduct::SymvUpper, K_LARGE_N, K_LARGE_N}};

        /**
         * @brief Runs GEMV and SYMV in a chosen execution mode and measures them against a truth.
         * @see tests/operations/blas/README.md
         */
        class Level2Parity : public SushiBLASTest
        {
            protected:
                /** @brief Returns the element a tensor of T holds at index of data, as a double. */
                template<typename T>
                [[nodiscard]] static double element(const ParityData& data,
                                                    std::size_t index) noexcept
                {
                    return static_cast<double>(static_cast<T>(data.value(index)));
                }

                /** @brief Returns a tensor of dims filled from data, starting at index seed. */
                template<typename T>
                [[nodiscard]] Tensor filled(std::initializer_list<int64_t> dims,
                                            const ParityData& data, std::size_t seed)
                {
                    Tensor t = engine->create_tensor(dims, sushiblas_dtype<T>());
                    std::vector<T> host(static_cast<std::size_t>(t.num_elements));
                    for (std::size_t i = 0; i < host.size(); ++i)
                        host[i] = static_cast<T>(data.value(i + seed));
                    TestEnvironment::queue()
                        .memcpy(t.data(), host.data(), host.size() * sizeof(T)).wait();
                    return t;
                }

                /** @brief Copies a tensor to the host. */
                template<typename T>
                [[nodiscard]] static std::vector<T> read_back(const Tensor& t)
                {
                    std::vector<T> host(static_cast<std::size_t>(t.num_elements));
                    TestEnvironment::queue()
                        .memcpy(host.data(), t.data(), host.size() * sizeof(T)).wait();
                    return host;
                }

                /** @brief Returns y = alpha * A * x + beta * y for an m x n A, computed in mode. */
                template<typename T>
                [[nodiscard]] std::vector<T> run_gemv(ExecutionMode mode, int64_t m, int64_t n,
                                                      const ParityData& data)
                {
                    set_mode(mode);
                    Tensor A = filled<T>({m, n}, data, K_SEED_A);
                    Tensor x = filled<T>({n}, data, K_SEED_X);
                    Tensor y = filled<T>({m}, data, K_SEED_Y);
                    SushiBLAS::BLASOps(recorder()).gemv(A, x, y, Core::Transpose::NoTrans,
                                                     data.alpha, data.beta);
                    recording->run().value();
                    return read_back<T>(y);
                }

                /** @brief Returns y = alpha * A^T * x + beta * y for an m x n A, computed in mode. */
                template<typename T>
                [[nodiscard]] std::vector<T> run_gemv_transposed(ExecutionMode mode, int64_t m,
                                                                 int64_t n, const ParityData& data)
                {
                    set_mode(mode);
                    Tensor A = filled<T>({m, n}, data, K_SEED_A);
                    Tensor x = filled<T>({m}, data, K_SEED_X);
                    Tensor y = filled<T>({n}, data, K_SEED_Y);
                    SushiBLAS::BLASOps(recorder()).gemv(A, x, y, Core::Transpose::Trans,
                                                     data.alpha, data.beta);
                    recording->run().value();
                    return read_back<T>(y);
                }

                /** @brief Returns y = alpha * A * x + beta * y for a symmetric A, run in mode. */
                template<typename T>
                [[nodiscard]] std::vector<T> run_symv(ExecutionMode mode, int64_t n, bool upper,
                                                      const ParityData& data)
                {
                    set_mode(mode);
                    Tensor A = filled<T>({n, n}, data, K_SEED_A);
                    Tensor x = filled<T>({n}, data, K_SEED_X);
                    Tensor y = filled<T>({n}, data, K_SEED_Y);
                    SushiBLAS::BLASOps(recorder()).symv(A, x, y, uplo_of(upper), data.alpha,
                                                     data.beta);
                    recording->run().value();
                    return read_back<T>(y);
                }

                /** @brief Returns the product of a shape on data, computed in mode. */
                template<typename T>
                [[nodiscard]] std::vector<T> run_product(const ParityShape& shape,
                                                         ExecutionMode mode, const ParityData& data)
                {
                    if (shape.product == ParityProduct::Gemv)
                        return run_gemv<T>(mode, shape.m, shape.n, data);
                    if (shape.product == ParityProduct::GemvTransposed)
                        return run_gemv_transposed<T>(mode, shape.m, shape.n, data);
                    return run_symv<T>(mode, shape.n, shape.product == ParityProduct::SymvUpper, data);
                }

                /** @brief Returns the exact result of run_gemv on data, element by element. */
                template<typename T>
                [[nodiscard]] static std::vector<TruthValue> truth_gemv(int64_t m, int64_t n,
                                                                        const ParityData& data)
                {
                    const std::size_t rows = static_cast<std::size_t>(m);
                    const std::size_t cols = static_cast<std::size_t>(n);
                    std::vector<TruthValue> truth(rows);
                    for (std::size_t i = 0; i < rows; ++i)
                    {
                        TruthSum sum;
                        for (std::size_t j = 0; j < cols; ++j)
                            sum.add_product(data.alpha * element<T>(data, K_SEED_A + i * cols + j),
                                            element<T>(data, K_SEED_X + j));
                        sum.add_product(data.beta, element<T>(data, K_SEED_Y + i));
                        truth[i] = sum.value();
                    }
                    return truth;
                }

                /** @brief Returns the exact result of run_gemv_transposed on data. */
                template<typename T>
                [[nodiscard]] static std::vector<TruthValue> truth_gemv_transposed(
                    int64_t m, int64_t n, const ParityData& data)
                {
                    const std::size_t terms = static_cast<std::size_t>(m);
                    const std::size_t cols = static_cast<std::size_t>(n);
                    std::vector<TruthValue> truth(cols);
                    for (std::size_t j = 0; j < cols; ++j)
                    {
                        TruthSum sum;
                        for (std::size_t i = 0; i < terms; ++i)
                            sum.add_product(data.alpha * element<T>(data, K_SEED_A + i * cols + j),
                                            element<T>(data, K_SEED_X + i));
                        sum.add_product(data.beta, element<T>(data, K_SEED_Y + j));
                        truth[j] = sum.value();
                    }
                    return truth;
                }

                /** @brief Returns the exact result of run_symv on data, read from one triangle. */
                template<typename T>
                [[nodiscard]] static std::vector<TruthValue> truth_symv(int64_t n, bool upper,
                                                                        const ParityData& data)
                {
                    const std::size_t order = static_cast<std::size_t>(n);
                    std::vector<TruthValue> truth(order);
                    for (std::size_t i = 0; i < order; ++i)
                    {
                        TruthSum sum;
                        for (std::size_t j = 0; j < order; ++j)
                        {
                            const bool stored = upper ? i <= j : i >= j;
                            const std::size_t index = stored ? i * order + j : j * order + i;
                            sum.add_product(data.alpha * element<T>(data, K_SEED_A + index),
                                            element<T>(data, K_SEED_X + j));
                        }
                        sum.add_product(data.beta, element<T>(data, K_SEED_Y + i));
                        truth[i] = sum.value();
                    }
                    return truth;
                }

                /** @brief Returns the exact result of the product of a shape on data. */
                template<typename T>
                [[nodiscard]] static std::vector<TruthValue> truth_of(const ParityShape& shape,
                                                                      const ParityData& data)
                {
                    if (shape.product == ParityProduct::Gemv)
                        return truth_gemv<T>(shape.m, shape.n, data);
                    if (shape.product == ParityProduct::GemvTransposed)
                        return truth_gemv_transposed<T>(shape.m, shape.n, data);
                    return truth_symv<T>(shape.n, shape.product == ParityProduct::SymvUpper, data);
                }

                /** @brief Returns the unit roundoff of the type the kernel of mode sums T in. */
                template<typename T>
                [[nodiscard]] static double summing_unit(ExecutionMode mode) noexcept
                {
                    return mode == ExecutionMode::ForceReference
                               ? unit_roundoff<Kernels::ReferenceAccumulator<T>>()
                               : unit_roundoff<Kernels::Accumulator<T>>();
                }

                /** @brief Returns the roundings the kernel of mode makes on a sum of len terms. */
                template<typename T>
                [[nodiscard]] static std::size_t roundings(ExecutionMode mode,
                                                           int64_t len) noexcept
                {
                    const bool compensated = mode == ExecutionMode::ForceReference &&
                                             Kernels::ReferenceSum<T>::K_IS_COMPENSATED;
                    return (compensated ? K_COMPENSATED_ROUNDINGS : static_cast<std::size_t>(len)) +
                           K_FIXED_ROUNDINGS;
                }

                /** @brief Expects every element within the bound of mode for a sum of len terms. */
                template<typename T>
                static void expect_within_bound(const std::vector<T>& actual,
                                                const std::vector<TruthValue>& truth, int64_t len,
                                                ExecutionMode mode)
                {
                    ASSERT_EQ(actual.size(), truth.size());
                    const double unit = summing_unit<T>(mode);
                    const std::size_t count = roundings<T>(mode, len);
                    for (std::size_t i = 0; i < actual.size(); ++i)
                    {
                        const double bound = rounding_bound(count, unit, truth[i].magnitude) +
                                             unit_roundoff<T>() * std::fabs(truth[i].value);
                        EXPECT_NEAR(static_cast<double>(actual[i]), truth[i].value, bound)
                            << "index " << i;
                    }
                }

                /** @brief Expects every element to equal the truth with no tolerance. */
                template<typename T>
                static void expect_exact(const std::vector<T>& actual,
                                         const std::vector<TruthValue>& truth)
                {
                    ASSERT_EQ(actual.size(), truth.size());
                    for (std::size_t i = 0; i < actual.size(); ++i)
                        EXPECT_EQ(static_cast<double>(actual[i]), truth[i].value) << "index " << i;
                }

                /** @brief Expects two results to hold the same bits. */
                template<typename T>
                static void expect_same_bits(const std::vector<T>& first,
                                             const std::vector<T>& second)
                {
                    ASSERT_EQ(first.size(), second.size());
                    EXPECT_EQ(0, std::memcmp(first.data(), second.data(), first.size() * sizeof(T)));
                }
        };

        /** @brief Runs a case once per row, and skips a device that lacks the type of the row. */
        class Level2ParityRow : public Level2Parity, public ::testing::WithParamInterface<ParityRow>
        {
            protected:
                /** @brief Skips the case when the device does not compute in the type of the row. */
                void SetUp() override
                {
                    Level2Parity::SetUp();
                    if (!device_runs(GetParam().type))
                        GTEST_SKIP() << "the device does not compute in this element type";
                }
        };

        /** @brief Measures the kernels against the truth on rounded data. */
        class Medium_Level2ParityBound : public Level2ParityRow
        {
        };

        /** @brief Measures the kernels against the truth on data whose sums are exact. */
        class Medium_Level2ParityExact : public Level2ParityRow
        {
        };

        /** @brief Runs the SYCL kernels twice and compares the bits. */
        class Medium_Level2ParityRepeat : public Level2ParityRow
        {
        };

        TEST_P(Medium_Level2ParityBound, TheKernelIsWithinItsBoundOfTheTruth)
        {
            const ParityRow& row = GetParam();
            with_single_or_double_element(row.type, [&](auto element)
            {
                using T = typename decltype(element)::type;
                expect_within_bound<T>(run_product<T>(row.shape, row.mode, K_PATTERNED),
                                       truth_of<T>(row.shape, K_PATTERNED), row.shape.n, row.mode);
            });
        }

        TEST_P(Medium_Level2ParityExact, TheKernelIsExactOnSmallIntegers)
        {
            const ParityRow& row = GetParam();
            with_single_or_double_element(row.type, [&](auto element)
            {
                using T = typename decltype(element)::type;
                expect_exact<T>(run_product<T>(row.shape, row.mode, K_INTEGER_VALUED),
                                truth_of<T>(row.shape, K_INTEGER_VALUED));
            });
        }

        TEST_P(Medium_Level2ParityRepeat, TheKernelRepeatsToTheBit)
        {
            const ParityRow& row = GetParam();
            with_single_or_double_element(row.type, [&](auto element)
            {
                using T = typename decltype(element)::type;
                const auto first = run_product<T>(row.shape, row.mode, K_PATTERNED);
                const auto second = run_product<T>(row.shape, row.mode, K_PATTERNED);
                expect_same_bits<T>(first, second);
            });
        }

        INSTANTIATE_TEST_SUITE_P(
            , Medium_Level2ParityBound,
            ::testing::ValuesIn(parity_rows(K_TRUTH_MODES, K_BOUND_SHAPES)), parity_row_name);

        INSTANTIATE_TEST_SUITE_P(
            , Medium_Level2ParityExact,
            ::testing::ValuesIn(parity_rows(K_TRUTH_MODES, K_EXACT_SHAPES)), parity_row_name);

        INSTANTIATE_TEST_SUITE_P(
            , Medium_Level2ParityRepeat,
            ::testing::ValuesIn(parity_rows({ExecutionMode::ForceSYCL}, K_REPEAT_SHAPES)),
            parity_row_name);
    } // namespace Test
} // namespace SushiBLAS
