/**************************************************************************/
/* test_packed_product.cpp                                                */
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
 * @file test_packed_product.cpp
 * @brief Tests the packed product kernel against the reference GEMM under injected blockings.
 * @author Mustafa Garip
 */

#include <array>
#include <cmath>
#include <string>
#include <vector>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/core/traits.hpp>
#include <SushiBLAS/core/blas_types.hpp>
#include <SushiBLAS/arithmetic/accumulator.hpp>
#include <SushiBLAS/launch/device_policy.hpp>
#include <SushiBLAS/arithmetic/complex_arithmetic.hpp>
#include <SushiBLAS/blas/dispatch.hpp>
#include <SushiBLAS/blas_seam_kernels/level3/gemm_epilogue.hpp>
#include <SushiBLAS/blas_seam_kernels/level3/channel_bias_epilogue.hpp>
#include <SushiBLAS/blas_seam_kernels/level3/packed_product_plan.hpp>
#include "../common/test_utils.hpp"
#include "../common/blas_truth.hpp"
#include "../common/packed_product_harness.hpp"

namespace SushiBLAS
{
    namespace Test
    {
        using namespace SushiRuntime::Literals;

        namespace
        {
            /// @brief Holds the seed of the operands whose products round.
            constexpr std::uint32_t K_UNIFORM_SEED = 20261007u;

            /// @brief Holds the rounded operations a parity bound allows beyond one per term.
            constexpr std::size_t K_FIXED_ROUNDINGS = 6;

            /// @brief Holds the step between the orders tried for a product the profile packs.
            constexpr int64_t K_SEAM_ORDER_STEP = 64;

            /// @brief Holds the largest order tried for a product the profile packs.
            constexpr int64_t K_LARGEST_SEAM_ORDER = 768;

            /// @brief Holds the order of the product that stays below every packing threshold.
            constexpr int64_t K_SMALL_ORDER = 8;

            /// @brief Holds the largest magnitude of a bias element of the epilogue test.
            constexpr double K_BIAS_MAGNITUDE = 1.0;

            /**
             * @brief Holds the relative error of storing a part of T: 2^-11 for HALF, the unit
             *        roundoff of the accumulation type otherwise.
             */
            template<typename T>
            constexpr double K_STORE_ROUNDOFF =
                std::is_same_v<T, sycl::half>
                    ? 0x1p-11
                    : unit_roundoff<Kernels::RealType<Kernels::Accumulator<T>>>();

            /** @brief Holds one injected blocking and the compute units it is planned for. */
            struct InjectedBlocking
            {
                const char* name;               /**< Names the blocking in a failure message. */
                Device::CacheBlocking blocking; /**< The budgets handed to the plan. */
                std::size_t compute_units;      /**< The compute units handed to the plan. */
            };

            /** @brief Returns the blockings every correctness test runs under, for type T. */
            template<typename T>
            [[nodiscard]] std::vector<InjectedBlocking> injected_blockings(const sycl::queue& q)
            {
                Device::CacheBlocking device_default = Device::profile_for(q).blocking;
                device_default.min_operand_bytes = 0;
                device_default.min_register_blocks = 0;

                Device::CacheBlocking crowded = ragged_blocking<T>();
                crowded.workers_per_compute_unit = 4;

                return {
                    {"smallest, one unit", smallest_blocking(), 1},
                    {"smallest, five units", smallest_blocking(), 5},
                    {"widest, one unit", widest_blocking(), 1},
                    {"widest, seven units", widest_blocking(), 7},
                    {"ragged, three units", ragged_blocking<T>(), 3},
                    {"ragged, four workers per unit", crowded, 3},
                    {"device default, two units", device_default, 2},
                };
            }

            /** @brief Returns alpha for the exact-data tests: 2, or 1 + i for a complex T. */
            template<typename T>
            [[nodiscard]] T exact_alpha() noexcept
            {
                if constexpr (Core::is_complex_v<T>)
                    return T(1, 1);
                else
                    return T(2);
            }

            /** @brief Returns beta for the exact-data tests: -1, or -i for a complex T. */
            template<typename T>
            [[nodiscard]] T exact_beta() noexcept
            {
                if constexpr (Core::is_complex_v<T>)
                    return T(0, -1);
                else
                    return T(-1);
            }

            /** @brief Returns the real and the imaginary part of value; a real T has no second. */
            template<typename T>
            [[nodiscard]] std::array<double, 2> parts_of(T value) noexcept
            {
                if constexpr (Core::is_complex_v<T>)
                    return {static_cast<double>(value.real()), static_cast<double>(value.imag())};
                else
                    return {Detail::to_double(value), 0.0};
            }

            /** @brief Returns the absolute value of a real or complex scalar. */
            template<typename T>
            [[nodiscard]] double magnitude_of(T value) noexcept
            {
                const std::array<double, 2> parts = parts_of(value);
                return std::hypot(parts[0], parts[1]);
            }

            /** @brief Returns whether a part of value is a NaN. */
            template<typename T>
            [[nodiscard]] bool is_nan_element(T value) noexcept
            {
                const std::array<double, 2> parts = parts_of(value);
                return std::isnan(parts[0]) || std::isnan(parts[1]);
            }

            /**
             * @brief Returns how far a part of one GEMM element may lie from the reference.
             * @param terms The terms of each sum.
             * @param alpha_abs The absolute value of alpha; operand parts are at most 1.
             * @param added_abs The largest magnitude added to the product: beta times C, or a bias.
             */
            template<typename T>
            [[nodiscard]] double parity_bound(int64_t terms, double alpha_abs,
                                              double added_abs) noexcept
            {
                using Real = Kernels::RealType<Kernels::Accumulator<T>>;

                const double parts = Core::is_complex_v<T> ? 2.0 : 1.0;
                const double magnitude =
                    parts * (alpha_abs * static_cast<double>(terms) + added_abs);
                return rounding_bound(static_cast<std::size_t>(terms) + K_FIXED_ROUNDINGS,
                                      unit_roundoff<Real>(), magnitude) +
                       K_STORE_ROUNDOFF<T> * magnitude;
            }

            /** @brief Returns how many parts of got lie beyond bound from want, a NaN included. */
            template<typename T>
            [[nodiscard]] std::size_t count_parts_beyond(const std::vector<T>& got,
                                                         const std::vector<T>& want,
                                                         double bound) noexcept
            {
                std::size_t beyond = 0;
                for (std::size_t i = 0; i < want.size(); ++i)
                {
                    const std::array<double, 2> got_parts = parts_of(got[i]);
                    const std::array<double, 2> want_parts = parts_of(want[i]);
                    for (std::size_t part = 0; part < got_parts.size(); ++part)
                        if (!(std::fabs(got_parts[part] - want_parts[part]) <= bound))
                            ++beyond;
                }
                return beyond;
            }

            /** @brief Returns the index of element (row, col) of a row-major C of n columns. */
            [[nodiscard]] std::size_t output_index(int64_t n, int64_t row, int64_t col) noexcept
            {
                return static_cast<std::size_t>(row * (n + K_PACKED_TEST_LD_PADDING) + col);
            }

            /** @brief Returns whether the device computes in the precision T needs. */
            template<typename T>
            [[nodiscard]] bool device_runs(const Device::Profile& profile) noexcept
            {
                if (std::is_same_v<T, sycl::half>)
                    return profile.supports_fp16;
                if (std::is_same_v<Kernels::RealType<T>, double>)
                    return profile.supports_fp64;
                return true;
            }

            /**
             * @brief Returns the smallest order the profile packs a square product of T at.
             * @return Zero when no order up to K_LARGEST_SEAM_ORDER is packed.
             */
            template<typename T>
            [[nodiscard]] int64_t packed_order(const Device::Profile& profile) noexcept
            {
                for (int64_t s = K_SEAM_ORDER_STEP; s <= K_LARGEST_SEAM_ORDER;
                     s += K_SEAM_ORDER_STEP)
                    if (Kernels::plan_packed_product<T>(profile, s, s, s, 1).selected)
                        return s;
                return 0;
            }

            /**
             * @brief Returns why a seam test of T cannot run on the device of q.
             * @return An empty string when the device runs T and packs some order of it.
             */
            template<typename T>
            [[nodiscard]] std::string seam_obstacle(const sycl::queue& q)
            {
                const Device::Profile& profile = Device::profile_for(q);
                const std::string dtype = Core::to_string(sushiblas_dtype<T>());

                if (!device_runs<T>(profile))
                    return dtype + ": the device does not compute in this precision. ";
                if (packed_order<T>(profile) == 0)
                    return dtype + ": no order up to " + std::to_string(K_LARGEST_SEAM_ORDER) +
                           " selects the packed kernel; cache_bytes=" +
                           std::to_string(profile.cache_bytes) + ". ";
                return std::string();
            }

            /** @brief Adds 8 * row + col to an element, so each output names its coordinates. */
            struct CoordinateEpilogue
            {
                /** @brief Returns value plus 8 * row + col. */
                template<typename T>
                [[nodiscard]] T operator()(T value, int64_t row, int64_t col) const noexcept
                {
                    return value + static_cast<T>(static_cast<float>(8 * row + col));
                }
            };
        } // namespace

        /** @brief Gives the packed product tests the queue they submit to, per element type. */
        template<typename T>
        class Unit_PackedProduct : public ::testing::Test
        {
            protected:
                /** @brief Skips the test when the device does not compute in T's precision. */
                void SetUp() override
                {
                    if (!device_runs<T>(Device::profile_for(queue())))
                        GTEST_SKIP() << "the device does not compute in this precision";
                }

                /** @brief Returns the queue of the test environment. */
                [[nodiscard]] static sycl::queue& queue()
                {
                    return TestEnvironment::queue();
                }

                /** @brief Expects the packed result to equal the reference under every blocking. */
                static void expect_exact(GemmProblem<T>& problem, T alpha, T beta)
                {
                    const std::vector<T> want = problem.expected(alpha, beta);
                    for (const InjectedBlocking& injected : injected_blockings<T>(queue()))
                    {
                        SCOPED_TRACE(injected.name);
                        const std::vector<T> got =
                            problem.packed(injected.blocking, injected.compute_units, alpha, beta);
                        EXPECT_EQ(count_bit_mismatches(got.data(), want.data(), want.size()),
                                  std::size_t{0});
                    }
                }
        };

        using PackedProductTypes = ::testing::Types<sycl::half, float, double, std::complex<float>,
                                                    std::complex<double>>;
        SB_TYPED_TEST_SUITE(Unit_PackedProduct, PackedProductTypes);

        /** @brief Tests that every blocking gives the reference result on a ragged shape. */
        TYPED_TEST(Unit_PackedProduct, EveryBlockingEqualsTheReferenceOnExactData)
        {
            GemmProblem<TypeParam> problem(this->queue(), Core::Layout::RowMajor,
                                           Core::Transpose::NoTrans, Core::Transpose::NoTrans,
                                           37, 45, 23, 1);
            this->expect_exact(problem, exact_alpha<TypeParam>(), exact_beta<TypeParam>());
        }

        /** @brief Tests every pair of Transpose values in both layouts against the reference. */
        TYPED_TEST(Unit_PackedProduct, EveryTransposeAndLayoutEqualsTheReference)
        {
            for (const Core::Layout layout : {Core::Layout::RowMajor, Core::Layout::ColumnMajor})
            {
                for (const Core::Transpose transA : {Core::Transpose::NoTrans,
                                                     Core::Transpose::Trans,
                                                     Core::Transpose::ConjTrans})
                {
                    for (const Core::Transpose transB : {Core::Transpose::NoTrans,
                                                         Core::Transpose::Trans,
                                                         Core::Transpose::ConjTrans})
                    {
                        SCOPED_TRACE(Core::to_string(transA));
                        SCOPED_TRACE(Core::to_string(transB));
                        GemmProblem<TypeParam> problem(this->queue(), layout, transA, transB,
                                                       19, 21, 17, 1);
                        this->expect_exact(problem, exact_alpha<TypeParam>(),
                                           exact_beta<TypeParam>());
                    }
                }
            }
        }

        /** @brief Tests shapes at, one past and far from the register block for exact parity. */
        TYPED_TEST(Unit_PackedProduct, EdgeShapesEqualTheReference)
        {
            using Block = Kernels::PackedRegisterBlock<Kernels::Accumulator<TypeParam>>;
            const int64_t rows = Block::K_ROWS;
            const int64_t cols = Block::K_COLS;

            const std::array<std::array<int64_t, 3>, 6> shapes = {{
                {1, 1, 1},
                {rows, cols, 1},
                {rows + 1, cols + 1, 2},
                {64, 3, 40},
                {3, 64, 40},
                {5 * rows, 5 * cols, 64},
            }};

            for (const std::array<int64_t, 3>& shape : shapes)
            {
                SCOPED_TRACE(::testing::Message()
                             << shape[0] << " x " << shape[1] << " x " << shape[2]);
                GemmProblem<TypeParam> problem(this->queue(), Core::Layout::RowMajor,
                                               Core::Transpose::NoTrans,
                                               Core::Transpose::NoTrans,
                                               shape[0], shape[1], shape[2], 1);
                this->expect_exact(problem, exact_alpha<TypeParam>(), exact_beta<TypeParam>());
            }
        }

        /** @brief Tests that a batch of three equals the reference, gaps included, per layout. */
        TYPED_TEST(Unit_PackedProduct, ABatchKeepsItsMatricesApart)
        {
            for (const Core::Layout layout : {Core::Layout::RowMajor, Core::Layout::ColumnMajor})
            {
                SCOPED_TRACE(layout == Core::Layout::RowMajor ? "row-major" : "column-major");
                GemmProblem<TypeParam> problem(this->queue(), layout, Core::Transpose::NoTrans,
                                               Core::Transpose::NoTrans, 13, 18, 11, 3);
                this->expect_exact(problem, exact_alpha<TypeParam>(), exact_beta<TypeParam>());
            }
        }

        /** @brief Tests that a product of no terms stores beta * C, and zero under a zero beta. */
        TYPED_TEST(Unit_PackedProduct, ZeroTermsScaleTheOutput)
        {
            const int64_t m = 19;
            const int64_t n = 21;
            const TypeParam alpha = exact_alpha<TypeParam>();
            const TypeParam zero = TypeParam(0);

            GemmProblem<TypeParam> problem(this->queue(), Core::Layout::RowMajor,
                                           Core::Transpose::NoTrans, Core::Transpose::NoTrans,
                                           m, n, 0, 1);
            this->expect_exact(problem, alpha, exact_beta<TypeParam>());
            this->expect_exact(problem, alpha, zero);

            for (const InjectedBlocking& injected : injected_blockings<TypeParam>(this->queue()))
            {
                SCOPED_TRACE(injected.name);
                const std::vector<TypeParam> got =
                    problem.packed(injected.blocking, injected.compute_units, alpha, zero);

                std::size_t non_zero = 0;
                for (int64_t row = 0; row < m; ++row)
                    for (int64_t col = 0; col < n; ++col)
                        if (!(got[output_index(n, row, col)] == zero))
                            ++non_zero;
                EXPECT_EQ(non_zero, std::size_t{0});
            }
        }

        /** @brief Tests that a zero beta stores no NaN from a C that holds nothing else. */
        TYPED_TEST(Unit_PackedProduct, ZeroBetaNeverReadsTheOutput)
        {
            const int64_t m = 37;
            const int64_t n = 45;
            const TypeParam alpha = exact_alpha<TypeParam>();
            const TypeParam zero = TypeParam(0);

            GemmProblem<TypeParam> problem(this->queue(), Core::Layout::RowMajor,
                                           Core::Transpose::NoTrans, Core::Transpose::NoTrans,
                                           m, n, 23, 1);
            problem.poison_output();

            // The padding is NaN in both buffers with the same bits, so the whole buffer compares.
            const std::vector<TypeParam> want = problem.expected(alpha, zero);
            for (const InjectedBlocking& injected : injected_blockings<TypeParam>(this->queue()))
            {
                SCOPED_TRACE(injected.name);
                const std::vector<TypeParam> got =
                    problem.packed(injected.blocking, injected.compute_units, alpha, zero);
                EXPECT_EQ(count_bit_mismatches(got.data(), want.data(), want.size()),
                          std::size_t{0});

                std::size_t nans = 0;
                for (int64_t row = 0; row < m; ++row)
                    for (int64_t col = 0; col < n; ++col)
                        if (is_nan_element(got[output_index(n, row, col)]))
                            ++nans;
                EXPECT_EQ(nans, std::size_t{0});
            }
        }

        /** @brief Tests that the epilogue runs once per element with that element's coordinates. */
        TYPED_TEST(Unit_PackedProduct, AnEpilogueSeesEveryElementOnceAtItsCoordinates)
        {
            const TypeParam alpha = exact_alpha<TypeParam>();
            const TypeParam beta = exact_beta<TypeParam>();

            GemmProblem<TypeParam> problem(this->queue(), Core::Layout::RowMajor,
                                           Core::Transpose::NoTrans, Core::Transpose::NoTrans,
                                           37, 45, 23, 1);

            const std::vector<TypeParam> want = problem.expected(alpha, beta, CoordinateEpilogue{});
            for (const InjectedBlocking& injected : injected_blockings<TypeParam>(this->queue()))
            {
                SCOPED_TRACE(injected.name);
                const std::vector<TypeParam> got =
                    problem.packed(injected.blocking, injected.compute_units, alpha, beta, nullptr,
                                   CoordinateEpilogue{});
                EXPECT_EQ(count_bit_mismatches(got.data(), want.data(), want.size()),
                          std::size_t{0});
            }
        }

        /** @brief Tests that an output of no rows or no columns plans no block and keeps C. */
        TYPED_TEST(Unit_PackedProduct, AnEmptyOutputLaunchesNothing)
        {
            const std::array<std::array<int64_t, 2>, 2> shapes = {{{0, 21}, {19, 0}}};

            for (const std::array<int64_t, 2>& shape : shapes)
            {
                SCOPED_TRACE(::testing::Message() << shape[0] << " x " << shape[1]);
                GemmProblem<TypeParam> problem(this->queue(), Core::Layout::RowMajor,
                                               Core::Transpose::NoTrans,
                                               Core::Transpose::NoTrans,
                                               shape[0], shape[1], 17, 1);

                // A fresh problem fills C with salt 2 over a leading dimension of n plus padding.
                const int64_t ldc = shape[1] + K_PACKED_TEST_LD_PADDING;
                std::vector<TypeParam> initial(problem.output_size());
                for (std::size_t i = 0; i < initial.size(); ++i)
                    initial[i] = patterned_element<TypeParam>(static_cast<int64_t>(i) / ldc,
                                                              static_cast<int64_t>(i) % ldc, 2);

                for (const InjectedBlocking& injected :
                     injected_blockings<TypeParam>(this->queue()))
                {
                    SCOPED_TRACE(injected.name);
                    Kernels::PackedProductPlan plan;
                    const std::vector<TypeParam> got =
                        problem.packed(injected.blocking, injected.compute_units,
                                       exact_alpha<TypeParam>(), exact_beta<TypeParam>(), &plan);
                    EXPECT_EQ(plan.blocks, 0);
                    ASSERT_EQ(got.size(), initial.size());
                    EXPECT_EQ(count_bit_mismatches(got.data(), initial.data(), initial.size()),
                              std::size_t{0});
                }
            }
        }

        /** @brief Tests that no blocking changes a bit of a product whose sums round. */
        TYPED_TEST(Unit_PackedProduct, BlockingsAgreeToTheBitOnInexactData)
        {
            const int64_t k = 64;
            const TypeParam alpha = exact_alpha<TypeParam>();
            const TypeParam beta = exact_beta<TypeParam>();

            GemmProblem<TypeParam> problem(this->queue(), Core::Layout::RowMajor,
                                           Core::Transpose::NoTrans, Core::Transpose::NoTrans,
                                           61, 70, k, 1);
            problem.fill_uniform(K_UNIFORM_SEED);

            const std::vector<TypeParam> want = problem.expected(alpha, beta);
            const double bound =
                parity_bound<TypeParam>(k, magnitude_of(alpha), magnitude_of(beta));

            std::vector<TypeParam> first;
            for (const InjectedBlocking& injected : injected_blockings<TypeParam>(this->queue()))
            {
                SCOPED_TRACE(injected.name);
                const std::vector<TypeParam> got =
                    problem.packed(injected.blocking, injected.compute_units, alpha, beta);
                if (first.empty())
                    first = got;

                EXPECT_EQ(count_bit_mismatches(got.data(), first.data(), first.size()),
                          std::size_t{0});
                EXPECT_EQ(count_parts_beyond(got, want, bound), std::size_t{0});
            }
        }

        /** @brief Tests that one blocking run twice on the same operands gives the same bits. */
        TYPED_TEST(Unit_PackedProduct, ARunRepeatsToTheBit)
        {
            const TypeParam alpha = exact_alpha<TypeParam>();
            const TypeParam beta = exact_beta<TypeParam>();

            GemmProblem<TypeParam> problem(this->queue(), Core::Layout::RowMajor,
                                           Core::Transpose::NoTrans, Core::Transpose::NoTrans,
                                           61, 70, 64, 1);
            problem.fill_uniform(K_UNIFORM_SEED);

            const std::vector<TypeParam> once =
                problem.packed(ragged_blocking<TypeParam>(), 3, alpha, beta);
            const std::vector<TypeParam> twice =
                problem.packed(ragged_blocking<TypeParam>(), 3, alpha, beta);
            EXPECT_EQ(count_bit_mismatches(twice.data(), once.data(), once.size()),
                      std::size_t{0});
        }

        /** @brief Tests that the packed kernel keeps the bits of the kernel without scratch. */
        TYPED_TEST(Unit_PackedProduct, ThePackedAndTheUnpackedKernelAgreeToTheBit)
        {
            if (!Device::profile_for(this->queue()).is_cpu)
                GTEST_SKIP() << "the device is not a CPU, where the two kernels sum in one order";

            const TypeParam alpha = exact_alpha<TypeParam>();
            const TypeParam beta = exact_beta<TypeParam>();

            GemmProblem<TypeParam> problem(this->queue(), Core::Layout::RowMajor,
                                           Core::Transpose::NoTrans, Core::Transpose::NoTrans,
                                           61, 70, 64, 1);
            problem.fill_uniform(K_UNIFORM_SEED);

            const std::vector<TypeParam> packed =
                problem.packed(ragged_blocking<TypeParam>(), 3, alpha, beta);
            const std::vector<TypeParam> unpacked = problem.unpacked(alpha, beta);
            EXPECT_EQ(count_bit_mismatches(packed.data(), unpacked.data(), unpacked.size()),
                      std::size_t{0});
        }

        /** @brief Runs square GEMMs through the engine at the order the device profile packs. */
        class Unit_PackedGemm : public SushiBLASTest
        {
            protected:
                /** @brief Returns the queue of the test environment. */
                [[nodiscard]] static sycl::queue& queue()
                {
                    return TestEnvironment::queue();
                }

                /** @brief Returns a square tensor of T of order s with elements in [-1, 1]. */
                template<typename T>
                [[nodiscard]] Tensor random_square(int64_t s)
                {
                    Tensor tensor = engine->create_tensor({s, s}, sushiblas_dtype<T>());
                    fill_random<T>(tensor);
                    return tensor;
                }

                /**
                 * @brief Expects two results of one product of s terms to lie within the bound.
                 * @param added_abs The largest magnitude the epilogue adds to an element.
                 */
                template<typename T>
                static void expect_parity(const Tensor& sycl_result, const Tensor& reference_result,
                                          int64_t s, double added_abs)
                {
                    const std::vector<T> got = read_tensor<T>(sycl_result);
                    const std::vector<T> want = read_tensor<T>(reference_result);
                    ASSERT_EQ(got.size(), want.size());
                    EXPECT_EQ(count_parts_beyond(got, want, parity_bound<T>(s, 1.0, added_abs)),
                              std::size_t{0});
                }

                /**
                 * @brief Runs one product of T under both modes and expects parity, or appends the
                 *        reason it cannot run to skipped.
                 */
                template<typename T>
                void expect_seam_parity(std::string& skipped)
                {
                    const std::string obstacle = seam_obstacle<T>(queue());
                    if (!obstacle.empty())
                    {
                        skipped += obstacle;
                        return;
                    }

                    const int64_t s = packed_order<T>(Device::profile_for(queue()));
                    const Tensor A = random_square<T>(s);
                    const Tensor B = random_square<T>(s);
                    Tensor sycl_result = engine->create_tensor({s, s}, sushiblas_dtype<T>());
                    Tensor reference_result =
                        engine->create_tensor({s, s}, sushiblas_dtype<T>());

                    engine->set_mode(ExecutionMode::ForceSYCL);
                    SushiBLAS::BLASOps(*engine).gemm(A, B, sycl_result);
                    engine->execute().value();

                    engine->set_mode(ExecutionMode::ForceReference);
                    SushiBLAS::BLASOps(*engine).gemm(A, B, reference_result);
                    engine->execute().value();

                    expect_parity<T>(sycl_result, reference_result, s, 0.0);
                }
        };

        /** @brief Tests that recording a packed GEMM leases one scratch block, on SYCL alone. */
        TEST_F(Unit_PackedGemm, TheSeamLeasesScratchWhenTheProfileSelectsThePackedKernel)
        {
            const std::string obstacle = seam_obstacle<float>(queue());
            if (!obstacle.empty())
                GTEST_SKIP() << obstacle;

            const int64_t s = packed_order<float>(Device::profile_for(queue()));
            const Tensor A = random_square<float>(s);
            const Tensor B = random_square<float>(s);
            Tensor sycl_result = engine->create_tensor({s, s}, Core::DataType::Float32);
            Tensor reference_result = engine->create_tensor({s, s}, Core::DataType::Float32);

            engine->set_mode(ExecutionMode::ForceSYCL);
            const std::size_t live_before_sycl = TestEnvironment::live_allocation_count();
            SushiBLAS::BLASOps(*engine).gemm(A, B, sycl_result);
            EXPECT_EQ(TestEnvironment::live_allocation_count(), live_before_sycl + 1);
            engine->execute().value();

            engine->set_mode(ExecutionMode::ForceReference);
            const std::size_t live_before_reference = TestEnvironment::live_allocation_count();
            SushiBLAS::BLASOps(*engine).gemm(A, B, reference_result);
            EXPECT_EQ(TestEnvironment::live_allocation_count(), live_before_reference);
            engine->execute().value();

            expect_parity<float>(sycl_result, reference_result, s, 0.0);
        }

        /** @brief Tests that a product below the thresholds records without leasing scratch. */
        TEST_F(Unit_PackedGemm, ASmallProductKeepsTheUnpackedKernel)
        {
            const Tensor A = random_square<float>(K_SMALL_ORDER);
            const Tensor B = random_square<float>(K_SMALL_ORDER);
            Tensor C = engine->create_tensor({K_SMALL_ORDER, K_SMALL_ORDER},
                                             Core::DataType::Float32);

            engine->set_mode(ExecutionMode::ForceSYCL);
            const std::size_t live_before = TestEnvironment::live_allocation_count();
            SushiBLAS::BLASOps(*engine).gemm(A, B, C);
            EXPECT_EQ(TestEnvironment::live_allocation_count(), live_before);
            engine->execute().value();
        }

        /** @brief Tests HALF and COMPLEX64 through the packed seam against the reference path. */
        TEST_F(Unit_PackedGemm, HalfAndComplexTakeThePackedSeam)
        {
            std::string skipped;
            expect_seam_parity<sycl::half>(skipped);
            expect_seam_parity<std::complex<double>>(skipped);

            if (!skipped.empty())
                GTEST_SKIP() << skipped;
        }

        /** @brief Tests that an epilogue bundle with a declared read runs on the packed seam. */
        TEST_F(Unit_PackedGemm, AnEpilogueBundleRidesThePackedSeam)
        {
            const std::string obstacle = seam_obstacle<float>(queue());
            if (!obstacle.empty())
                GTEST_SKIP() << obstacle;

            const int64_t s = packed_order<float>(Device::profile_for(queue()));
            const Tensor A = random_square<float>(s);
            const Tensor B = random_square<float>(s);
            Tensor sycl_result = engine->create_tensor({s, s}, Core::DataType::Float32);
            Tensor reference_result = engine->create_tensor({s, s}, Core::DataType::Float32);
            Tensor bias = engine->create_tensor({s}, Core::DataType::Float32);
            fill_random<float>(bias);

            // (0, 1): the bias element is chosen by the column of C.
            const Kernels::ChannelBiasEpilogue epilogue{bias.data(), 0, 1};

            engine->set_mode(ExecutionMode::ForceSYCL);
            Dispatch::execute_gemm(engine->recorder(), A, B, sycl_result,
                                   Core::Transpose::NoTrans, Core::Transpose::NoTrans, 1.0, 0.0,
                                   "test.packed_gemm_bias", "test.packed_gemm_bias"_op,
                                   Dispatch::gemm_epilogue(epilogue).reads(bias));
            engine->execute().value();

            engine->set_mode(ExecutionMode::ForceReference);
            Dispatch::execute_gemm(engine->recorder(), A, B, reference_result,
                                   Core::Transpose::NoTrans, Core::Transpose::NoTrans, 1.0, 0.0,
                                   "test.packed_gemm_bias_reference",
                                   "test.packed_gemm_bias_reference"_op,
                                   Dispatch::gemm_epilogue(epilogue).reads(bias));
            engine->execute().value();

            expect_parity<float>(sycl_result, reference_result, s, K_BIAS_MAGNITUDE);
        }
    } // namespace Test
} // namespace SushiBLAS
