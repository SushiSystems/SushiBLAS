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
#include "support/tolerance.hpp"
namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            /// @brief The forced modes every case runs in.
            constexpr ExecutionMode K_FORCED_MODES[] = {ExecutionMode::ForceSYCL,
                                                        ExecutionMode::ForceReference};

            /// @brief The order of each matrix the factorisation test factors.
            constexpr std::size_t K_FACTOR_ORDER = 9;

            /// @brief How many matrices the factorisation test factors; matrix 1 is singular.
            constexpr std::size_t K_FACTOR_BATCH = 3;

            /// @brief The order and the batch of the solve test.
            constexpr std::size_t K_SOLVE_ORDER = 8;
            constexpr std::size_t K_SOLVE_BATCH = 2;

            /// @brief The order of the system GMRES solves.
            constexpr std::size_t K_GMRES_ORDER = 16;

            /**
             * @brief The GMRES relative tolerance in units of the unit roundoff of FLOAT64: 1e-12 is
             *        about 9007 of them.
             */
            constexpr double K_GMRES_UNITS_DOUBLE = 9007.0;

            /**
             * @brief The GMRES relative tolerance in units of the unit roundoff of FLOAT32: 1e-5 is
             *        about 168 of them.
             */
            constexpr double K_GMRES_UNITS_FLOAT = 168.0;

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

            /** @brief Holds P^T L U of one factored matrix and the sum of |L||U| that bounds its rounding. */
            struct Reconstruction
            {
                std::vector<double> product;   /**< P^T L U. */
                std::vector<double> magnitude; /**< P^T |L||U|, entry by entry. */
            };

            /** @brief Returns P^T L U and P^T |L||U| of one factored n * n matrix. */
            template<typename T>
            Reconstruction reconstruct(const T* lu, const std::uint32_t* pivots, std::size_t n)
            {
                Reconstruction r{std::vector<double>(n * n, 0.0), std::vector<double>(n * n, 0.0)};
                for (std::size_t i = 0; i < n; ++i)
                    for (std::size_t j = 0; j < n; ++j)
                        for (std::size_t k = 0; k <= std::min(i, j); ++k)
                        {
                            const double l = k == i ? 1.0 : static_cast<double>(lu[i * n + k]);
                            const double u = static_cast<double>(lu[k * n + j]);
                            r.product[i * n + j] += l * u;
                            r.magnitude[i * n + j] += std::fabs(l * u);
                        }
                for (std::size_t k = n; k-- > 0;)
                    for (std::size_t j = 0; j < n; ++j)
                    {
                        std::swap(r.product[k * n + j], r.product[pivots[k] * n + j]);
                        std::swap(r.magnitude[k * n + j], r.magnitude[pivots[k] * n + j]);
                    }
                return r;
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

            /** @brief Returns the largest magnitude among the values. */
            double largest_magnitude(const std::vector<double>& in)
            {
                return std::fabs(*std::max_element(in.begin(), in.end(),
                                                   [](double a, double b)
                                                   { return std::fabs(a) < std::fabs(b); }));
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

            /** @brief Runs check once for each forced mode. */
            template<typename Check>
            void in_each_forced_mode(Check&& check)
            {
                for (const ExecutionMode mode : K_FORCED_MODES)
                {
                    SCOPED_TRACE(trace_of(mode));
                    check(mode);
                }
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

            /** @brief Holds what lu_factor wrote for the matrices of the factorisation test. */
            template<typename T>
            struct FactorRun
            {
                std::vector<T> original;          /**< The matrices before the factorisation. */
                std::vector<T> lu;                /**< The factors, packed. */
                std::vector<std::uint32_t> pivots;   /**< The pivot rows. */
                std::vector<std::uint32_t> singular; /**< The singular flag of each matrix. */
            };

            /** @brief Factors K_FACTOR_BATCH matrices, matrix 1 repeating its first row, in mode. */
            template<typename T>
            FactorRun<T> run_factor(Engine& engine, ExecutionMode mode)
            {
                std::vector<double> raw = values(K_FACTOR_BATCH * K_FACTOR_ORDER * K_FACTOR_ORDER, 61);
                // Matrix 1 repeats its first row, so one pivot vanishes.
                for (std::size_t j = 0; j < K_FACTOR_ORDER; ++j)
                    raw[K_FACTOR_ORDER * K_FACTOR_ORDER + K_FACTOR_ORDER + j] =
                        raw[K_FACTOR_ORDER * K_FACTOR_ORDER + j];

                FactorRun<T> run;
                run.original = rounded<T>(raw);
                engine.set_mode(mode);
                Tensor a = tensor_of(engine, run.original);
                Tensor pivots = tensor_of(
                    engine, std::vector<std::uint32_t>(K_FACTOR_BATCH * K_FACTOR_ORDER, 99u));
                Tensor singular = tensor_of(engine, std::vector<std::uint32_t>(K_FACTOR_BATCH, 7u));
                RecordingFixture graph(engine);
                LinalgOps(graph.recorder())
                    .lu_factor(a, pivots, singular, K_FACTOR_ORDER, K_FACTOR_BATCH);
                graph.run().value();

                run.lu = SushiBLAS::IO().to_vector<T>(a);
                run.pivots = SushiBLAS::IO().to_vector<std::uint32_t>(pivots);
                run.singular = SushiBLAS::IO().to_vector<std::uint32_t>(singular);
                return run;
            }

            /** @brief Expects P^T L U of the matrix `slot` of a run to equal the matrix within n·u·|L||U|. */
            template<typename T>
            void expect_reconstructs(const FactorRun<T>& run, std::size_t slot)
            {
                const std::size_t entries = K_FACTOR_ORDER * K_FACTOR_ORDER;
                const Reconstruction r = reconstruct(run.lu.data() + slot * entries,
                                                     run.pivots.data() + slot * K_FACTOR_ORDER,
                                                     K_FACTOR_ORDER);
                for (std::size_t e = 0; e < entries; ++e)
                    EXPECT_TRUE(TestSupport::is_within(TestSupport::measure_sum_error<T>(
                        r.product[e], static_cast<double>(run.original[slot * entries + e]),
                        r.magnitude[e], K_FACTOR_ORDER)))
                        << slot << " " << e;
            }

            /** @brief Expects the matrices 0 and 2 of a run to reconstruct, in mode. */
            template<typename T>
            void expect_factor_reconstructs(Engine& engine, ExecutionMode mode)
            {
                const FactorRun<T> run = run_factor<T>(engine, mode);
                expect_reconstructs(run, 0);
                expect_reconstructs(run, 2);
            }

            /** @brief Expects the singular flags of the run in mode to mark only matrix 1. */
            template<typename T>
            void expect_factor_flags_singular(Engine& engine, ExecutionMode mode)
            {
                EXPECT_EQ(run_factor<T>(engine, mode).singular,
                          (std::vector<std::uint32_t>{0u, 1u, 0u}));
            }

            /**
             * @brief Expects lu_factor then lu_solve to match the direct solve in mode.
             * @note The matrices are diagonally dominant with a diagonal of n over entries below one,
             *       so the condition number is below 2n and the error of the solve stays within
             *       n * n roundings of the largest solution entry.
             */
            template<typename T>
            void expect_solve_matches_direct(Engine& engine, ExecutionMode mode)
            {
                constexpr std::size_t K_N = K_SOLVE_ORDER;
                std::vector<double> raw = values(K_SOLVE_BATCH * K_N * K_N, 62);
                make_dominant(raw, K_N, K_SOLVE_BATCH);
                const std::vector<T> a_values = rounded<T>(raw);
                const std::vector<T> b_values = rounded<T>(values(K_SOLVE_BATCH * K_N, 63));

                engine.set_mode(mode);
                Tensor a = tensor_of(engine, a_values);
                Tensor pivots = tensor_of(engine, std::vector<std::uint32_t>(K_SOLVE_BATCH * K_N));
                Tensor singular = tensor_of(engine, std::vector<std::uint32_t>(K_SOLVE_BATCH));
                Tensor x = tensor_of(engine, b_values);
                RecordingFixture graph(engine);
                LinalgOps view(graph.recorder());
                view.lu_factor(a, pivots, singular, K_N, K_SOLVE_BATCH);
                view.lu_solve(a, pivots, x, K_N, 1, K_SOLVE_BATCH);
                graph.run().value();

                const std::vector<T> got = SushiBLAS::IO().to_vector<T>(x);
                for (std::size_t b = 0; b < K_SOLVE_BATCH; ++b)
                {
                    const std::vector<double> matrix(a_values.begin() + b * K_N * K_N,
                                                     a_values.begin() + (b + 1) * K_N * K_N);
                    const std::vector<double> rhs(b_values.begin() + b * K_N,
                                                  b_values.begin() + (b + 1) * K_N);
                    const std::vector<double> expected = direct_solve(matrix, rhs, K_N);
                    const double bound = static_cast<double>(K_N * K_N) *
                                         TestSupport::unit_roundoff<T>() * largest_magnitude(expected);
                    for (std::size_t i = 0; i < K_N; ++i)
                        EXPECT_TRUE(TestSupport::is_within(TestSupport::measure_error(
                            static_cast<double>(got[b * K_N + i]), expected[i], bound)))
                            << b << " " << i;
                }
            }

            /** @brief Returns the GMRES relative tolerance for T. */
            template<typename T>
            double gmres_tolerance()
            {
                const double units = std::is_same_v<T, double> ? K_GMRES_UNITS_DOUBLE
                                                               : K_GMRES_UNITS_FLOAT;
                return units * TestSupport::unit_roundoff<T>();
            }

            /** @brief Holds what gmres wrote for the system of the GMRES tests. */
            template<typename T>
            struct GmresRun
            {
                std::vector<double> expected; /**< The direct solution. */
                std::vector<T> solution;      /**< The solution gmres wrote. */
                std::uint32_t iterations = 0; /**< The iterations it reports. */
                T residual = T(-1);           /**< The relative residual it reports. */
                GmresOptions options;         /**< The options it ran with. */
            };

            /** @brief Solves the dominant system of order K_GMRES_ORDER with gmres in mode. */
            template<typename T>
            GmresRun<T> run_gmres(Engine& engine, ExecutionMode mode)
            {
                constexpr std::size_t K_N = K_GMRES_ORDER;
                std::vector<double> raw = values(K_N * K_N, 64);
                make_dominant(raw, K_N, 1);
                const std::vector<T> a_values = rounded<T>(raw);
                const std::vector<T> b_values = rounded<T>(values(K_N, 65));

                GmresRun<T> run;
                run.expected = direct_solve(std::vector<double>(a_values.begin(), a_values.end()),
                                            std::vector<double>(b_values.begin(), b_values.end()),
                                            K_N);
                run.options.restart = 5;
                run.options.block_size = 4;
                run.options.relative_tolerance = gmres_tolerance<T>();

                engine.set_mode(mode);
                const Tensor a = tensor_of(engine, a_values);
                const Tensor b = tensor_of(engine, b_values);
                Tensor x = tensor_of(engine, std::vector<T>(K_N, T(0)));
                Tensor iterations = tensor_of(engine, std::vector<std::uint32_t>(1, 0u));
                Tensor residual = tensor_of(engine, std::vector<T>(1, T(-1)));
                RecordingFixture graph(engine);
                LinalgOps(graph.recorder()).gmres(a, b, x, iterations, residual, run.options);
                graph.run().value();

                run.solution = SushiBLAS::IO().to_vector<T>(x);
                run.iterations = SushiBLAS::IO().to_vector<std::uint32_t>(iterations)[0];
                run.residual = SushiBLAS::IO().to_vector<T>(residual)[0];
                return run;
            }

            /**
             * @brief Expects the solution of gmres to match the direct solve in mode.
             * @note The residual is at most tolerance times |b|, |b| is at most 2n times the largest
             *       solution entry for this matrix, and |A^-1| is below one, which bounds the error.
             */
            template<typename T>
            void expect_gmres_solution(Engine& engine, ExecutionMode mode)
            {
                const GmresRun<T> run = run_gmres<T>(engine, mode);
                const double bound = run.options.relative_tolerance *
                                     static_cast<double>(2 * K_GMRES_ORDER) *
                                     largest_magnitude(run.expected);
                for (std::size_t i = 0; i < K_GMRES_ORDER; ++i)
                    EXPECT_TRUE(TestSupport::is_within(TestSupport::measure_error(
                        static_cast<double>(run.solution[i]), run.expected[i], bound)))
                        << i;
            }

            /** @brief Expects gmres to report at least one iteration and no more than its limit in mode. */
            template<typename T>
            void expect_gmres_iterations(Engine& engine, ExecutionMode mode)
            {
                const GmresRun<T> run = run_gmres<T>(engine, mode);
                EXPECT_GE(run.iterations, 1u);
                EXPECT_LE(run.iterations, run.options.max_iterations);
            }

            /** @brief Expects gmres to report a relative residual within its tolerance in mode. */
            template<typename T>
            void expect_gmres_residual(Engine& engine, ExecutionMode mode)
            {
                const GmresRun<T> run = run_gmres<T>(engine, mode);
                EXPECT_LE(run.residual, static_cast<T>(run.options.relative_tolerance));
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

        /** @brief Tests that P^T L U of a factored matrix equals the matrix within the rounding bound. */
        TYPED_TEST(Medium_LinalgTyped, FactorReconstructsPLU)
        {
            in_each_forced_mode([&](ExecutionMode mode)
                                { expect_factor_reconstructs<TypeParam>(*this->engine, mode); });
        }

        /** @brief Tests that the singular flag marks the matrix with a vanishing pivot and no other. */
        TYPED_TEST(Medium_LinalgTyped, FactorFlagsTheSingularMatrix)
        {
            in_each_forced_mode([&](ExecutionMode mode)
                                { expect_factor_flags_singular<TypeParam>(*this->engine, mode); });
        }

        TYPED_TEST(Medium_LinalgTyped, SolveMatchesTheDirectSolve)
        {
            in_each_forced_mode([&](ExecutionMode mode)
                                { expect_solve_matches_direct<TypeParam>(*this->engine, mode); });
        }

        /** @brief Tests that gmres writes the solution of the direct solve within its tolerance. */
        TYPED_TEST(Medium_LinalgTyped, GmresMatchesTheDirectSolve)
        {
            in_each_forced_mode([&](ExecutionMode mode)
                                { expect_gmres_solution<TypeParam>(*this->engine, mode); });
        }

        /** @brief Tests that gmres reports an iteration count between one and its limit. */
        TYPED_TEST(Medium_LinalgTyped, GmresReportsItsIterationCount)
        {
            in_each_forced_mode([&](ExecutionMode mode)
                                { expect_gmres_iterations<TypeParam>(*this->engine, mode); });
        }

        /** @brief Tests that gmres reports a residual within its tolerance. */
        TYPED_TEST(Medium_LinalgTyped, GmresReportsAResidualWithinItsTolerance)
        {
            in_each_forced_mode([&](ExecutionMode mode)
                                { expect_gmres_residual<TypeParam>(*this->engine, mode); });
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
