/**************************************************************************/
/* test_linalg.cpp                                                        */
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
 * @file test_linalg.cpp
 * @brief Tests LinalgOps::lu_factor, lu_solve and gmres in both forced modes against P L U
 * reconstruction and a direct solve, and their refusals.
 * @author Mustafa Garip
 */

#include <SushiBLAS/experimental/lapack/linalg.hpp>

#include <cmath>
#include <limits>
#include <string>
#include <vector>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <algorithm>
#include <stdexcept>
#include <type_traits>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <SushiBLAS/SushiBLAS.hpp>
#include <SushiBLAS/engine_fixture/sushiblas_test.hpp>
#include <SushiBLAS/tensor_expectations/scalar_dtype.hpp>
#include <SushiBLAS/typed_suite_naming/default_type_names.hpp>
#include <SushiBLAS/test_runner/test_environment.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            /// @brief The forced modes every case runs in.
            constexpr ExecutionMode K_FORCED_MODES[] = {ExecutionMode::ForceSYCL,
                                                        ExecutionMode::ForceReference};

            /** @brief Returns count fixed pseudo-random values in [-1, 1) from a seed. */
            std::vector<double> values(std::size_t count, unsigned seed)
            {
                std::vector<double> out(count);
                unsigned state = seed * 2654435761u + 1u;
                for (double& v : out)
                {
                    state = state * 1664525u + 1013904223u;
                    v = static_cast<double>(state >> 8) / static_cast<double>(1u << 24) * 2.0 - 1.0;
                }
                return out;
            }

            /** @brief Adds n to the diagonal of each of batch n * n matrices. */
            void make_dominant(std::vector<double>& a, std::size_t n, std::size_t batch)
            {
                for (std::size_t b = 0; b < batch; ++b)
                    for (std::size_t i = 0; i < n; ++i)
                        a[b * n * n + i * n + i] += static_cast<double>(n);
            }

            /** @brief Returns the solution of the n * n system A x = b in long double. */
            std::vector<double> direct_solve(const std::vector<double>& a,
                                             const std::vector<double>& b, std::size_t n)
            {
                std::vector<long double> m(a.begin(), a.end());
                std::vector<long double> x(b.begin(), b.end());
                for (std::size_t k = 0; k < n; ++k)
                {
                    std::size_t best = k;
                    for (std::size_t i = k + 1; i < n; ++i)
                        if (std::fabs(m[i * n + k]) > std::fabs(m[best * n + k]))
                            best = i;
                    for (std::size_t j = 0; j < n; ++j)
                        std::swap(m[k * n + j], m[best * n + j]);
                    std::swap(x[k], x[best]);
                    for (std::size_t i = k + 1; i < n; ++i)
                    {
                        const long double f = m[i * n + k] / m[k * n + k];
                        for (std::size_t j = k; j < n; ++j)
                            m[i * n + j] -= f * m[k * n + j];
                        x[i] -= f * x[k];
                    }
                }
                for (std::size_t i = n; i-- > 0;)
                {
                    for (std::size_t j = i + 1; j < n; ++j)
                        x[i] -= m[i * n + j] * x[j];
                    x[i] /= m[i * n + i];
                }
                return std::vector<double>(x.begin(), x.end());
            }

            /** @brief Returns P^T L U of one factored n * n matrix. */
            template<typename T>
            std::vector<double> reconstruct(const T* lu, const std::uint32_t* pivots,
                                            std::size_t n)
            {
                std::vector<double> product(n * n, 0.0);
                for (std::size_t i = 0; i < n; ++i)
                    for (std::size_t j = 0; j < n; ++j)
                        for (std::size_t k = 0; k <= std::min(i, j); ++k)
                        {
                            const double l = k == i ? 1.0 : static_cast<double>(lu[i * n + k]);
                            product[i * n + j] += l * static_cast<double>(lu[k * n + j]);
                        }
                for (std::size_t k = n; k-- > 0;)
                    for (std::size_t j = 0; j < n; ++j)
                        std::swap(product[k * n + j], product[pivots[k] * n + j]);
                return product;
            }

            /** @brief Returns a rank-1 tensor holding elements. */
            template<typename E>
            Tensor tensor_of(Engine& engine, const std::vector<E>& elements)
            {
                Tensor t = engine.create_tensor({static_cast<int64_t>(elements.size())},
                                                sushiblas_dtype<E>());
                SushiBLAS::IO().from_vector(t, elements);
                return t;
            }

            /** @brief Returns values rounded to T. */
            template<typename T>
            std::vector<T> rounded(const std::vector<double>& in)
            {
                return std::vector<T>(in.begin(), in.end());
            }

            /** @brief Returns true when the test device can compute in T. */
            template<typename T>
            bool device_computes()
            {
                if constexpr (std::is_same_v<T, double>)
                    return TestEnvironment::queue().get_device().has(sycl::aspect::fp64);
                else
                    return true;
            }

            /** @brief Returns the name a failure message gives a mode. */
            const char* trace_of(ExecutionMode mode)
            {
                return mode == ExecutionMode::ForceReference ? "ForceReference" : "ForceSYCL";
            }

            /**
             * @brief Expects record to throw std::invalid_argument naming op and operand.
             * @tparam Record Callable `() -> void` that records one malformed call.
             */
            template<typename Record>
            void expect_refusal(Record&& record, const char* op, const char* operand)
            {
                try
                {
                    record();
                    ADD_FAILURE() << op << " accepted a malformed '" << operand << "'";
                }
                catch (const std::invalid_argument& error)
                {
                    const std::string text = error.what();
                    EXPECT_NE(text.find(op), std::string::npos) << text;
                    EXPECT_NE(text.find("'" + std::string(operand) + "'"), std::string::npos)
                        << text;
                }
            }
        } // namespace

        /** @brief Runs the solvers in one scalar type, skipped where the device lacks it. */
        template<typename T>
        class Medium_LinalgTyped : public SushiBLASTest
        {
            protected:
                /** @brief Builds the engine and skips the case on a device without T's aspect. */
                void SetUp() override
                {
                    SushiBLASTest::SetUp();
                    if (!device_computes<T>())
                        GTEST_SKIP() << "the device does not report the aspect this dtype needs";
                }
        };

        using LinalgScalars = ::testing::Types<float, double>;
        SB_TYPED_TEST_SUITE(Medium_LinalgTyped, LinalgScalars);

        /** @brief Runs the refusals, which fix one scalar type. */
        class Medium_Linalg : public SushiBLASTest
        {

        };

        TYPED_TEST(Medium_LinalgTyped, FactorReconstructsPLUAndFlagsTheSingularMatrix)
        {
            using T = TypeParam;
            Engine& engine = *this->engine;
            constexpr std::size_t K_N = 9;
            constexpr std::size_t K_BATCH = 3;
            std::vector<double> raw = values(K_BATCH * K_N * K_N, 61);
            // Matrix 1 repeats its first row, so one pivot vanishes.
            for (std::size_t j = 0; j < K_N; ++j)
                raw[K_N * K_N + K_N + j] = raw[K_N * K_N + j];
            const std::vector<T> original = rounded<T>(raw);
            const double bound = 64.0 * K_N * std::numeric_limits<T>::epsilon();

            for (const ExecutionMode mode : K_FORCED_MODES)
            {
                SCOPED_TRACE(trace_of(mode));
                engine.set_mode(mode);
                Tensor a = tensor_of(engine, original);
                Tensor pivots = tensor_of(engine, std::vector<std::uint32_t>(K_BATCH * K_N, 99u));
                Tensor singular = tensor_of(engine, std::vector<std::uint32_t>(K_BATCH, 7u));
                RecordingFixture graph(engine);
                LinalgOps(graph.recorder()).lu_factor(a, pivots, singular, K_N, K_BATCH);
                graph.run().value();

                const std::vector<T> lu = SushiBLAS::IO().to_vector<T>(a);
                const std::vector<std::uint32_t> p =
                    SushiBLAS::IO().to_vector<std::uint32_t>(pivots);
                EXPECT_EQ(SushiBLAS::IO().to_vector<std::uint32_t>(singular),
                          (std::vector<std::uint32_t>{0u, 1u, 0u}));
                for (const std::size_t b : {std::size_t{0}, std::size_t{2}})
                {
                    const std::vector<double> product =
                        reconstruct(lu.data() + b * K_N * K_N, p.data() + b * K_N, K_N);
                    for (std::size_t e = 0; e < K_N * K_N; ++e)
                        EXPECT_NEAR(product[e], original[b * K_N * K_N + e], bound)
                            << b << " " << e;
                }
            }
        }

        TYPED_TEST(Medium_LinalgTyped, SolveMatchesTheDirectSolve)
        {
            using T = TypeParam;
            Engine& engine = *this->engine;
            constexpr std::size_t K_N = 8;
            constexpr std::size_t K_BATCH = 2;
            std::vector<double> raw = values(K_BATCH * K_N * K_N, 62);
            make_dominant(raw, K_N, K_BATCH);
            const std::vector<T> a_values = rounded<T>(raw);
            const std::vector<T> b_values = rounded<T>(values(K_BATCH * K_N, 63));
            const double bound = 64.0 * K_N * std::numeric_limits<T>::epsilon();

            for (const ExecutionMode mode : K_FORCED_MODES)
            {
                SCOPED_TRACE(trace_of(mode));
                engine.set_mode(mode);
                Tensor a = tensor_of(engine, a_values);
                Tensor pivots = tensor_of(engine, std::vector<std::uint32_t>(K_BATCH * K_N));
                Tensor singular = tensor_of(engine, std::vector<std::uint32_t>(K_BATCH));
                Tensor x = tensor_of(engine, b_values);
                RecordingFixture graph(engine);
                LinalgOps view(graph.recorder());
                view.lu_factor(a, pivots, singular, K_N, K_BATCH);
                view.lu_solve(a, pivots, x, K_N, 1, K_BATCH);
                graph.run().value();

                const std::vector<T> got = SushiBLAS::IO().to_vector<T>(x);
                for (std::size_t b = 0; b < K_BATCH; ++b)
                {
                    const std::vector<double> matrix(a_values.begin() + b * K_N * K_N,
                                                     a_values.begin() + (b + 1) * K_N * K_N);
                    const std::vector<double> rhs(b_values.begin() + b * K_N,
                                                  b_values.begin() + (b + 1) * K_N);
                    const std::vector<double> expected = direct_solve(matrix, rhs, K_N);
                    for (std::size_t i = 0; i < K_N; ++i)
                        EXPECT_NEAR(got[b * K_N + i], expected[i], bound) << b << " " << i;
                }
            }
        }

        TYPED_TEST(Medium_LinalgTyped, GmresMatchesTheDirectSolveAndReportsItsWork)
        {
            using T = TypeParam;
            Engine& engine = *this->engine;
            constexpr std::size_t K_N = 16;
            std::vector<double> raw = values(K_N * K_N, 64);
            make_dominant(raw, K_N, 1);
            const std::vector<T> a_values = rounded<T>(raw);
            const std::vector<T> b_values = rounded<T>(values(K_N, 65));
            const std::vector<double> expected =
                direct_solve(std::vector<double>(a_values.begin(), a_values.end()),
                             std::vector<double>(b_values.begin(), b_values.end()), K_N);
            GmresOptions options;
            options.restart = 5;
            options.block_size = 4;
            options.relative_tolerance = std::is_same_v<T, double> ? 1e-12 : 1e-5;

            for (const ExecutionMode mode : K_FORCED_MODES)
            {
                SCOPED_TRACE(trace_of(mode));
                engine.set_mode(mode);
                const Tensor a = tensor_of(engine, a_values);
                const Tensor b = tensor_of(engine, b_values);
                Tensor x = tensor_of(engine, std::vector<T>(K_N, T(0)));
                Tensor iterations = tensor_of(engine, std::vector<std::uint32_t>(1, 0u));
                Tensor residual = tensor_of(engine, std::vector<T>(1, T(-1)));
                RecordingFixture graph(engine);
                LinalgOps(graph.recorder()).gmres(a, b, x, iterations, residual, options);
                graph.run().value();

                const std::vector<T> got = SushiBLAS::IO().to_vector<T>(x);
                const std::uint32_t taken =
                    SushiBLAS::IO().to_vector<std::uint32_t>(iterations)[0];
                EXPECT_GE(taken, 1u);
                EXPECT_LE(taken, options.max_iterations);
                EXPECT_LE(SushiBLAS::IO().to_vector<T>(residual)[0],
                          static_cast<T>(options.relative_tolerance));
                const double bound = 100.0 * options.relative_tolerance;
                for (std::size_t i = 0; i < K_N; ++i)
                    EXPECT_NEAR(got[i], expected[i], bound) << i;
            }
        }

        TEST_F(Medium_Linalg, FactorRefusesAWrongPivotCount)
        {
            Engine& engine = *this->engine;
            Tensor a = tensor_of(engine, std::vector<float>(4, 1.0f));
            Tensor pivots = tensor_of(engine, std::vector<std::uint32_t>(3));
            Tensor singular = tensor_of(engine, std::vector<std::uint32_t>(1));
            RecordingFixture graph(engine);

            expect_refusal(
                [&] { LinalgOps(graph.recorder()).lu_factor(a, pivots, singular, 2, 1); },
                "linalg.lu_factor", "pivots");
        }

        TEST_F(Medium_Linalg, SolveRefusesAMismatchedDtype)
        {
            Engine& engine = *this->engine;
            const Tensor a = tensor_of(engine, std::vector<float>(4, 1.0f));
            const Tensor pivots = tensor_of(engine, std::vector<std::uint32_t>(2));
            Tensor rhs = tensor_of(engine, std::vector<std::int32_t>(2));
            RecordingFixture graph(engine);

            expect_refusal([&] { LinalgOps(graph.recorder()).lu_solve(a, pivots, rhs, 2, 1, 1); },
                           "linalg.lu_solve", "rhs");
        }

        TEST_F(Medium_Linalg, GmresRefusesABlockThatDoesNotDivideTheOrder)
        {
            Engine& engine = *this->engine;
            const Tensor a = tensor_of(engine, std::vector<float>(9, 1.0f));
            const Tensor b = tensor_of(engine, std::vector<float>(3, 1.0f));
            Tensor x = tensor_of(engine, std::vector<float>(3, 0.0f));
            Tensor iterations = tensor_of(engine, std::vector<std::uint32_t>(1));
            Tensor residual = tensor_of(engine, std::vector<float>(1));
            GmresOptions options;
            options.block_size = 2;
            RecordingFixture graph(engine);

            expect_refusal(
                [&] { LinalgOps(graph.recorder()).gmres(a, b, x, iterations, residual, options); },
                "linalg.gmres", "options");
        }

        TEST_F(Medium_Linalg, GmresRefusesAnIntegerMatrix)
        {
            Engine& engine = *this->engine;
            const Tensor a = tensor_of(engine, std::vector<std::int32_t>(4, 1));
            const Tensor b = tensor_of(engine, std::vector<float>(2, 1.0f));
            Tensor x = tensor_of(engine, std::vector<float>(2, 0.0f));
            Tensor iterations = tensor_of(engine, std::vector<std::uint32_t>(1));
            Tensor residual = tensor_of(engine, std::vector<float>(1));
            RecordingFixture graph(engine);

            expect_refusal([&]
                           {
                               LinalgOps(graph.recorder())
                                   .gmres(a, b, x, iterations, residual, GmresOptions{});
                           },
                           "linalg.gmres", "matrix");
        }
    } // namespace Test
} // namespace SushiBLAS
