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
#include <iterator>
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
#include <SushiBLAS/SushiBLAS.hpp>
#include <SushiBLAS/engine_fixture/sushiblas_test.hpp>
#include <SushiBLAS/tensor_expectations/tensor_comparison.hpp>
#include <SushiBLAS/tensor_expectations/tensor_fill.hpp>
#include <SushiBLAS/tensor_expectations/tensor_readback.hpp>
#include <SushiBLAS/tensor_expectations/scalar_dtype.hpp>
#include <SushiBLAS/test_runner/test_environment.hpp>
#include <SushiBLAS/blas_test_oracle/blas_truth.hpp>
#include <SushiBLAS/packed_product_harness/packed_product_harness.hpp>
#include "../test_blas/blas_rows.hpp"

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

            /// @brief Holds the largest magnitude of a bias element of the epilogue test.
            constexpr double K_BIAS_MAGNITUDE = 1.0;

            /// @brief Holds the rows of the output of the tests that scale it.
            constexpr int64_t K_OUTPUT_ROWS = 19;

            /// @brief Holds the columns of the output of the tests that scale it.
            constexpr int64_t K_OUTPUT_COLS = 21;

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

            /** @brief Returns how many elements of the m by n block of got differ from zero. */
            template<typename T>
            [[nodiscard]] std::size_t count_non_zero(const std::vector<T>& got, int64_t m,
                                                     int64_t n) noexcept
            {
                std::size_t non_zero = 0;
                for (int64_t row = 0; row < m; ++row)
                    for (int64_t col = 0; col < n; ++col)
                        if (!(got[output_index(n, row, col)] == T(0)))
                            ++non_zero;
                return non_zero;
            }

            /** @brief Returns how many elements of the m by n block of got hold a NaN. */
            template<typename T>
            [[nodiscard]] std::size_t count_nans(const std::vector<T>& got, int64_t m,
                                                 int64_t n) noexcept
            {
                std::size_t nans = 0;
                for (int64_t row = 0; row < m; ++row)
                    for (int64_t col = 0; col < n; ++col)
                        if (is_nan_element(got[output_index(n, row, col)]))
                            ++nans;
                return nans;
            }

            /**
             * @brief Returns the output a fresh problem starts from: salt 2 over a leading
             *        dimension of n plus the padding.
             * @param size The elements of the output buffer.
             * @param n The columns of the output.
             */
            template<typename T>
            [[nodiscard]] std::vector<T> fresh_output(std::size_t size, int64_t n)
            {
                const int64_t ldc = n + K_PACKED_TEST_LD_PADDING;
                std::vector<T> initial(size);
                for (std::size_t i = 0; i < initial.size(); ++i)
                    initial[i] = patterned_element<T>(static_cast<int64_t>(i) / ldc,
                                                      static_cast<int64_t>(i) % ldc, 2);
                return initial;
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

                if (!device_runs(sushiblas_dtype<T>()))
                    return dtype + ": the device does not compute in this precision.";
                if (packed_order<T>(profile) == 0)
                    return dtype + ": no order up to " + std::to_string(K_LARGEST_SEAM_ORDER) +
                           " selects the packed kernel; cache_bytes=" +
                           std::to_string(profile.cache_bytes) + ".";
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

            /** @brief Holds the layout and the two forms of one product. */
            struct Orientation
            {
                Core::Layout layout; /**< The element order of the operands. */
                Core::Transpose a;   /**< The form of op(A). */
                Core::Transpose b;   /**< The form of op(B). */
            };

            /** @brief Names the shapes that sit at, past and far from a register block. */
            enum class EdgeShape
            {
                OneByOneByOne,
                OneBlock,
                PastOneBlock,
                TallAndThin,
                WideAndThin,
                FiveBlocks
            };

            /** @brief Names the two outputs that hold no element. */
            enum class EmptyOutput
            {
                NoRows,
                NoColumns
            };

            /** @brief Returns every pair of forms in both layouts. */
            [[nodiscard]] std::vector<Orientation> all_orientations()
            {
                const Core::Transpose forms[] = {Core::Transpose::NoTrans, Core::Transpose::Trans,
                                                 Core::Transpose::ConjTrans};
                std::vector<Orientation> orientations;
                for (const Core::Layout layout : K_ROW_LAYOUTS)
                    for (const Core::Transpose a : forms)
                        for (const Core::Transpose b : forms)
                            orientations.push_back({layout, a, b});
                return orientations;
            }

            /** @brief Returns the shapes of the edge cases. */
            [[nodiscard]] std::vector<EdgeShape> all_edge_shapes()
            {
                return {EdgeShape::OneByOneByOne, EdgeShape::OneBlock,    EdgeShape::PastOneBlock,
                        EdgeShape::TallAndThin,   EdgeShape::WideAndThin, EdgeShape::FiveBlocks};
            }

            /** @brief Returns the two layouts as the values of an axis. */
            [[nodiscard]] std::vector<Core::Layout> all_layouts()
            {
                return std::vector<Core::Layout>(std::begin(K_ROW_LAYOUTS),
                                                 std::end(K_ROW_LAYOUTS));
            }

            /** @brief Returns the rows, columns and terms of an edge shape, for the block of T. */
            template<typename T>
            [[nodiscard]] std::array<int64_t, 3> edge_extents(EdgeShape shape) noexcept
            {
                using Block = Kernels::PackedRegisterBlock<Kernels::Accumulator<T>>;
                const int64_t rows = Block::K_ROWS;
                const int64_t cols = Block::K_COLS;

                switch (shape)
                {
                    case EdgeShape::OneByOneByOne:
                        return {1, 1, 1};
                    case EdgeShape::OneBlock:
                        return {rows, cols, 1};
                    case EdgeShape::PastOneBlock:
                        return {rows + 1, cols + 1, 2};
                    case EdgeShape::TallAndThin:
                        return {64, 3, 40};
                    case EdgeShape::WideAndThin:
                        return {3, 64, 40};
                    case EdgeShape::FiveBlocks:
                        break;
                }
                return {5 * rows, 5 * cols, 64};
            }

            /** @brief Returns the rows and columns of an output that holds no element. */
            [[nodiscard]] std::array<int64_t, 2> empty_extents(EmptyOutput output) noexcept
            {
                if (output == EmptyOutput::NoRows)
                    return {0, K_OUTPUT_COLS};
                return {K_OUTPUT_ROWS, 0};
            }

            /** @brief Returns the tail of a row named by a layout. */
            [[nodiscard]] std::string axis_name(Core::Layout layout)
            {
                return "_" + word(layout);
            }

            /** @brief Returns the tail of a row named by an orientation. */
            [[nodiscard]] std::string axis_name(const Orientation& orientation)
            {
                return "_" + word(orientation.layout) + "_" + word(orientation.a) + "_" +
                       word(orientation.b);
            }

            /** @brief Returns the tail of a row named by an edge shape. */
            [[nodiscard]] std::string axis_name(EdgeShape shape)
            {
                switch (shape)
                {
                    case EdgeShape::OneByOneByOne:
                        return "_OneByOneByOne";
                    case EdgeShape::OneBlock:
                        return "_OneBlock";
                    case EdgeShape::PastOneBlock:
                        return "_PastOneBlock";
                    case EdgeShape::TallAndThin:
                        return "_TallAndThin";
                    case EdgeShape::WideAndThin:
                        return "_WideAndThin";
                    case EdgeShape::FiveBlocks:
                        break;
                }
                return "_FiveBlocks";
            }

            /** @brief Returns the tail of a row named by an empty output. */
            [[nodiscard]] std::string axis_name(EmptyOutput output)
            {
                return output == EmptyOutput::NoRows ? "_NoRows" : "_NoColumns";
            }

            /**
             * @brief Holds one row of a packed product suite: the element type and the one axis
             *        the cases of the suite vary besides it.
             * @tparam Axis NoAxis, or the type of the value that completes the row.
             */
            template<typename Axis>
            struct PackedCase
            {
                Core::DataType type; /**< The element type of every operand. */
                Axis axis;           /**< The value that completes the row. */
            };

            /** @brief Returns every row over the five element types and the given axis values. */
            template<typename Axis>
            [[nodiscard]] std::vector<PackedCase<Axis>> cases_of(const std::vector<Axis>& axes)
            {
                std::vector<PackedCase<Axis>> cases;
                for (const Core::DataType type : K_ALL_TYPES)
                    for (const Axis& axis : axes)
                        cases.push_back({type, axis});
                return cases;
            }

            /** @brief Returns the rows of a suite that has no axis of its own. */
            [[nodiscard]] std::vector<PackedCase<NoAxis>> plain_cases()
            {
                return cases_of(std::vector<NoAxis>{NoAxis{}});
            }

            /** @brief Returns the name of a row for GoogleTest: element type, then the axis. */
            template<typename Axis>
            [[nodiscard]] std::string case_name(
                const ::testing::TestParamInfo<PackedCase<Axis>>& info)
            {
                return Core::to_string(info.param.type) + axis_name(info.param.axis);
            }
        } // namespace

        /**
         * @brief Gives the packed product tests the queue they submit to and the checks they
         *        share, once per row.
         * @tparam Axis NoAxis, or the type of the value that completes each row.
         */
        template<typename Axis>
        class PackedProductCase : public ::testing::TestWithParam<PackedCase<Axis>>
        {
            protected:
                /** @brief Skips the row when the device does not compute in its precision. */
                void SetUp() override
                {
                    if (!device_runs(this->GetParam().type))
                        GTEST_SKIP() << "the device does not compute in this precision";
                }

                /** @brief Returns the queue of the test environment. */
                [[nodiscard]] static sycl::queue& queue()
                {
                    return TestEnvironment::queue();
                }

                /** @brief Expects the packed result to equal the reference under every blocking. */
                template<typename T>
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

                /** @brief Expects every blocking to store zeros in the m by n output. */
                template<typename T>
                static void expect_zero_output(GemmProblem<T>& problem, T alpha, int64_t m,
                                               int64_t n)
                {
                    for (const InjectedBlocking& injected : injected_blockings<T>(queue()))
                    {
                        SCOPED_TRACE(injected.name);
                        const std::vector<T> got = problem.packed(
                            injected.blocking, injected.compute_units, alpha, T(0));
                        EXPECT_EQ(count_non_zero(got, m, n), std::size_t{0});
                    }
                }

                /**
                 * @brief Expects every blocking to give the reference bits of a poisoned output
                 *        under a zero beta, and no NaN in the m by n block. The padding is NaN in both
                 *        buffers with the same bits, so the whole buffer compares.
                 */
                template<typename T>
                static void expect_poison_unread(GemmProblem<T>& problem, T alpha, int64_t m,
                                                 int64_t n)
                {
                    const std::vector<T> want = problem.expected(alpha, T(0));
                    for (const InjectedBlocking& injected : injected_blockings<T>(queue()))
                    {
                        SCOPED_TRACE(injected.name);
                        const std::vector<T> got = problem.packed(
                            injected.blocking, injected.compute_units, alpha, T(0));
                        EXPECT_EQ(count_bit_mismatches(got.data(), want.data(), want.size()),
                                  std::size_t{0});
                        EXPECT_EQ(count_nans(got, m, n), std::size_t{0});
                    }
                }

                /** @brief Expects every blocking to give the reference bits under the epilogue. */
                template<typename T>
                static void expect_epilogue_exact(GemmProblem<T>& problem, T alpha, T beta)
                {
                    const std::vector<T> want =
                        problem.expected(alpha, beta, CoordinateEpilogue{});
                    for (const InjectedBlocking& injected : injected_blockings<T>(queue()))
                    {
                        SCOPED_TRACE(injected.name);
                        const std::vector<T> got = problem.packed(
                            injected.blocking, injected.compute_units, alpha, beta, nullptr,
                            CoordinateEpilogue{});
                        EXPECT_EQ(count_bit_mismatches(got.data(), want.data(), want.size()),
                                  std::size_t{0});
                    }
                }

                /** @brief Expects every blocking to leave an output of no element untouched. */
                template<typename T>
                static void expect_output_kept(GemmProblem<T>& problem, int64_t n)
                {
                    const std::vector<T> initial = fresh_output<T>(problem.output_size(), n);
                    for (const InjectedBlocking& injected : injected_blockings<T>(queue()))
                    {
                        SCOPED_TRACE(injected.name);
                        Kernels::PackedProductPlan plan;
                        const std::vector<T> got = problem.packed(
                            injected.blocking, injected.compute_units, exact_alpha<T>(),
                            exact_beta<T>(), &plan);
                        EXPECT_EQ(plan.blocks, 0);
                        ASSERT_EQ(got.size(), initial.size());
                        EXPECT_EQ(count_bit_mismatches(got.data(), initial.data(), initial.size()),
                                  std::size_t{0});
                    }
                }

                /**
                 * @brief Expects every blocking to give the bits of the first one, and every
                 *        result to lie within the parity bound of the reference.
                 */
                template<typename T>
                static void expect_blockings_agree(GemmProblem<T>& problem, T alpha, T beta,
                                                   int64_t terms)
                {
                    const std::vector<InjectedBlocking> blockings =
                        injected_blockings<T>(queue());
                    const std::vector<T> want = problem.expected(alpha, beta);
                    const std::vector<T> first = problem.packed(
                        blockings.front().blocking, blockings.front().compute_units, alpha, beta);
                    const double bound =
                        parity_bound<T>(terms, magnitude_of(alpha), magnitude_of(beta));

                    for (const InjectedBlocking& injected : blockings)
                    {
                        SCOPED_TRACE(injected.name);
                        const std::vector<T> got = problem.packed(
                            injected.blocking, injected.compute_units, alpha, beta);
                        EXPECT_EQ(count_bit_mismatches(got.data(), first.data(), first.size()),
                                  std::size_t{0});
                        EXPECT_EQ(count_parts_beyond(got, want, bound), std::size_t{0});
                    }
                }
        };

        /** @brief Runs the cases that take no axis besides the element type. */
        class Medium_PackedProduct : public PackedProductCase<NoAxis>
        {

        };

        /** @brief Runs the cases that take a pair of forms and a layout. */
        class Medium_PackedProductOrientation : public PackedProductCase<Orientation>
        {

        };

        /** @brief Runs the cases that take a shape at the edge of a register block. */
        class Medium_PackedProductEdge : public PackedProductCase<EdgeShape>
        {

        };

        /** @brief Runs the cases that take a layout. */
        class Medium_PackedProductLayout : public PackedProductCase<Core::Layout>
        {

        };

        /** @brief Runs the cases that take an output of no element. */
        class Medium_PackedProductEmpty : public PackedProductCase<EmptyOutput>
        {

        };

        /** @brief Runs the cases that compare the two kernels, which sum in one order on a CPU. */
        class Medium_PackedProductOnCpu : public PackedProductCase<NoAxis>
        {
            protected:
                /** @brief Skips the row on a device that is not a CPU. */
                void SetUp() override
                {
                    PackedProductCase<NoAxis>::SetUp();
                    if (!IsSkipped() && !Device::profile_for(queue()).is_cpu)
                        GTEST_SKIP()
                            << "the device is not a CPU, where the two kernels sum in one order";
                }
        };

        /** @brief Tests that every blocking gives the reference result on a ragged shape. */
        TEST_P(Medium_PackedProduct, EveryBlockingEqualsTheReferenceOnExactData)
        {
            with_element(GetParam().type, [&](auto element) {
                using T = typename decltype(element)::type;
                GemmProblem<T> problem(queue(), Core::Layout::RowMajor, Core::Transpose::NoTrans,
                                       Core::Transpose::NoTrans, 37, 45, 23, 1);
                expect_exact(problem, exact_alpha<T>(), exact_beta<T>());
            });
        }

        /** @brief Tests a pair of forms in a layout against the reference. */
        TEST_P(Medium_PackedProductOrientation, EveryTransposeAndLayoutEqualsTheReference)
        {
            with_element(GetParam().type, [&](auto element) {
                using T = typename decltype(element)::type;
                const Orientation& orientation = GetParam().axis;
                GemmProblem<T> problem(queue(), orientation.layout, orientation.a, orientation.b,
                                       19, 21, 17, 1);
                expect_exact(problem, exact_alpha<T>(), exact_beta<T>());
            });
        }

        /** @brief Tests a shape at, one past or far from the register block for exact parity. */
        TEST_P(Medium_PackedProductEdge, EdgeShapesEqualTheReference)
        {
            with_element(GetParam().type, [&](auto element) {
                using T = typename decltype(element)::type;
                const std::array<int64_t, 3> shape = edge_extents<T>(GetParam().axis);
                GemmProblem<T> problem(queue(), Core::Layout::RowMajor, Core::Transpose::NoTrans,
                                       Core::Transpose::NoTrans, shape[0], shape[1], shape[2], 1);
                expect_exact(problem, exact_alpha<T>(), exact_beta<T>());
            });
        }

        /** @brief Tests that a batch of three equals the reference, gaps included. */
        TEST_P(Medium_PackedProductLayout, ABatchKeepsItsMatricesApart)
        {
            with_element(GetParam().type, [&](auto element) {
                using T = typename decltype(element)::type;
                GemmProblem<T> problem(queue(), GetParam().axis, Core::Transpose::NoTrans,
                                       Core::Transpose::NoTrans, 13, 18, 11, 3);
                expect_exact(problem, exact_alpha<T>(), exact_beta<T>());
            });
        }

        /** @brief Tests that a product of no terms stores beta * C. */
        TEST_P(Medium_PackedProduct, ZeroTermsScaleTheOutput)
        {
            with_element(GetParam().type, [&](auto element) {
                using T = typename decltype(element)::type;
                GemmProblem<T> problem(queue(), Core::Layout::RowMajor, Core::Transpose::NoTrans,
                                       Core::Transpose::NoTrans, K_OUTPUT_ROWS, K_OUTPUT_COLS, 0,
                                       1);
                expect_exact(problem, exact_alpha<T>(), exact_beta<T>());
            });
        }

        /** @brief Tests that a product of no terms stores zero under a zero beta. */
        TEST_P(Medium_PackedProduct, ZeroTermsUnderZeroBetaStoreZero)
        {
            with_element(GetParam().type, [&](auto element) {
                using T = typename decltype(element)::type;
                GemmProblem<T> problem(queue(), Core::Layout::RowMajor, Core::Transpose::NoTrans,
                                       Core::Transpose::NoTrans, K_OUTPUT_ROWS, K_OUTPUT_COLS, 0,
                                       1);
                expect_exact(problem, exact_alpha<T>(), T(0));
                expect_zero_output(problem, exact_alpha<T>(), K_OUTPUT_ROWS, K_OUTPUT_COLS);
            });
        }

        /** @brief Tests that a zero beta stores no NaN from a C that holds nothing else. */
        TEST_P(Medium_PackedProduct, ZeroBetaNeverReadsTheOutput)
        {
            with_element(GetParam().type, [&](auto element) {
                using T = typename decltype(element)::type;
                GemmProblem<T> problem(queue(), Core::Layout::RowMajor, Core::Transpose::NoTrans,
                                       Core::Transpose::NoTrans, 37, 45, 23, 1);
                problem.poison_output();
                expect_poison_unread(problem, exact_alpha<T>(), 37, 45);
            });
        }

        /** @brief Tests that the epilogue runs once per element with that element's coordinates. */
        TEST_P(Medium_PackedProduct, AnEpilogueSeesEveryElementOnceAtItsCoordinates)
        {
            with_element(GetParam().type, [&](auto element) {
                using T = typename decltype(element)::type;
                GemmProblem<T> problem(queue(), Core::Layout::RowMajor, Core::Transpose::NoTrans,
                                       Core::Transpose::NoTrans, 37, 45, 23, 1);
                expect_epilogue_exact(problem, exact_alpha<T>(), exact_beta<T>());
            });
        }

        /** @brief Tests that an output of no rows or no columns plans no block and keeps C. */
        TEST_P(Medium_PackedProductEmpty, AnEmptyOutputLaunchesNothing)
        {
            with_element(GetParam().type, [&](auto element) {
                using T = typename decltype(element)::type;
                const std::array<int64_t, 2> shape = empty_extents(GetParam().axis);
                GemmProblem<T> problem(queue(), Core::Layout::RowMajor, Core::Transpose::NoTrans,
                                       Core::Transpose::NoTrans, shape[0], shape[1], 17, 1);
                expect_output_kept(problem, shape[1]);
            });
        }

        /** @brief Tests that no blocking changes a bit of a product whose sums round. */
        TEST_P(Medium_PackedProduct, BlockingsAgreeToTheBitOnInexactData)
        {
            with_element(GetParam().type, [&](auto element) {
                using T = typename decltype(element)::type;
                const int64_t k = 64;
                GemmProblem<T> problem(queue(), Core::Layout::RowMajor, Core::Transpose::NoTrans,
                                       Core::Transpose::NoTrans, 61, 70, k, 1);
                problem.fill_uniform(K_UNIFORM_SEED);
                expect_blockings_agree(problem, exact_alpha<T>(), exact_beta<T>(), k);
            });
        }

        /** @brief Tests that one blocking run twice on the same operands gives the same bits. */
        TEST_P(Medium_PackedProduct, ARunRepeatsToTheBit)
        {
            with_element(GetParam().type, [&](auto element) {
                using T = typename decltype(element)::type;
                GemmProblem<T> problem(queue(), Core::Layout::RowMajor, Core::Transpose::NoTrans,
                                       Core::Transpose::NoTrans, 61, 70, 64, 1);
                problem.fill_uniform(K_UNIFORM_SEED);

                const std::vector<T> once = problem.packed(ragged_blocking<T>(), 3,
                                                           exact_alpha<T>(), exact_beta<T>());
                const std::vector<T> twice = problem.packed(ragged_blocking<T>(), 3,
                                                            exact_alpha<T>(), exact_beta<T>());
                EXPECT_EQ(count_bit_mismatches(twice.data(), once.data(), once.size()),
                          std::size_t{0});
            });
        }

        /** @brief Tests that the packed kernel keeps the bits of the kernel without scratch. */
        TEST_P(Medium_PackedProductOnCpu, ThePackedAndTheUnpackedKernelAgreeToTheBit)
        {
            with_element(GetParam().type, [&](auto element) {
                using T = typename decltype(element)::type;
                GemmProblem<T> problem(queue(), Core::Layout::RowMajor, Core::Transpose::NoTrans,
                                       Core::Transpose::NoTrans, 61, 70, 64, 1);
                problem.fill_uniform(K_UNIFORM_SEED);

                const std::vector<T> packed = problem.packed(ragged_blocking<T>(), 3,
                                                             exact_alpha<T>(), exact_beta<T>());
                const std::vector<T> unpacked =
                    problem.unpacked(exact_alpha<T>(), exact_beta<T>());
                EXPECT_EQ(count_bit_mismatches(packed.data(), unpacked.data(), unpacked.size()),
                          std::size_t{0});
            });
        }

        INSTANTIATE_TEST_SUITE_P(, Medium_PackedProduct, ::testing::ValuesIn(plain_cases()),
                                 case_name<NoAxis>);
        INSTANTIATE_TEST_SUITE_P(, Medium_PackedProductOrientation,
                                 ::testing::ValuesIn(cases_of(all_orientations())),
                                 case_name<Orientation>);
        INSTANTIATE_TEST_SUITE_P(, Medium_PackedProductEdge,
                                 ::testing::ValuesIn(cases_of(all_edge_shapes())),
                                 case_name<EdgeShape>);
        INSTANTIATE_TEST_SUITE_P(, Medium_PackedProductLayout,
                                 ::testing::ValuesIn(cases_of(all_layouts())),
                                 case_name<Core::Layout>);
        INSTANTIATE_TEST_SUITE_P(
            , Medium_PackedProductEmpty,
            ::testing::ValuesIn(cases_of(
                std::vector<EmptyOutput>{EmptyOutput::NoRows, EmptyOutput::NoColumns})),
            case_name<EmptyOutput>);
        INSTANTIATE_TEST_SUITE_P(, Medium_PackedProductOnCpu, ::testing::ValuesIn(plain_cases()),
                                 case_name<NoAxis>);

        /** @brief Holds the tensors and the runs the packed GEMM seam tests share. */
        class PackedGemmSeam : public SushiBLASTest
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
                static void expect_parity(const Tensor& sycl_result,
                                          const Tensor& reference_result, int64_t s,
                                          double added_abs)
                {
                    const std::vector<T> got = read_tensor<T>(sycl_result);
                    const std::vector<T> want = read_tensor<T>(reference_result);
                    ASSERT_EQ(got.size(), want.size());
                    EXPECT_EQ(count_parts_beyond(got, want, parity_bound<T>(s, 1.0, added_abs)),
                              std::size_t{0});
                }

                /**
                 * @brief Runs one product of T under both modes.
                 * @return How many parts of the SYCL result lie beyond the parity bound of the
                 *         reference path.
                 */
                template<typename T>
                std::size_t seam_parts_beyond()
                {
                    const int64_t s = packed_order<T>(Device::profile_for(queue()));
                    const Tensor A = random_square<T>(s);
                    const Tensor B = random_square<T>(s);
                    Tensor sycl_result = engine->create_tensor({s, s}, sushiblas_dtype<T>());
                    Tensor reference_result =
                        engine->create_tensor({s, s}, sushiblas_dtype<T>());

                    set_mode(ExecutionMode::ForceSYCL);
                    SushiBLAS::BLASOps(recorder()).gemm(A, B, sycl_result);
                    recording->run().value();

                    set_mode(ExecutionMode::ForceReference);
                    SushiBLAS::BLASOps(recorder()).gemm(A, B, reference_result);
                    recording->run().value();

                    return count_parts_beyond(read_tensor<T>(sycl_result),
                                              read_tensor<T>(reference_result),
                                              parity_bound<T>(s, 1.0, 0.0));
                }
        };

        /** @brief Runs the packed GEMM seam once per element type the profile can pack. */
        class Medium_PackedGemm : public PackedGemmSeam,
                                  public ::testing::WithParamInterface<Core::DataType>
        {
            protected:
                /** @brief Skips the row when the device cannot pack a product of its type. */
                void SetUp() override
                {
                    PackedGemmSeam::SetUp();
                    std::string obstacle;
                    with_element(GetParam(), [&](auto element) {
                        obstacle = seam_obstacle<typename decltype(element)::type>(queue());
                    });
                    if (!obstacle.empty())
                        GTEST_SKIP() << obstacle;
                }
        };

        /** @brief Runs the packed GEMM seam with an epilogue bundle. */
        class Medium_PackedGemmEpilogue : public PackedGemmSeam
        {
            protected:
                /** @brief Skips the test when the device cannot pack a product of FLOAT32. */
                void SetUp() override
                {
                    PackedGemmSeam::SetUp();
                    const std::string obstacle = seam_obstacle<float>(queue());
                    if (!obstacle.empty())
                        GTEST_SKIP() << obstacle;
                }
        };

        /** @brief Tests that the packed seam gives the result of the reference path. */
        TEST_P(Medium_PackedGemm, TheSeamMatchesTheReferencePath)
        {
            with_element(GetParam(), [&](auto element) {
                EXPECT_EQ(seam_parts_beyond<typename decltype(element)::type>(), std::size_t{0});
            });
        }

        /** @brief Tests that an epilogue bundle with a declared read runs on the packed seam. */
        TEST_F(Medium_PackedGemmEpilogue, AnEpilogueBundleRidesThePackedSeam)
        {
            const int64_t s = packed_order<float>(Device::profile_for(queue()));
            const Tensor A = random_square<float>(s);
            const Tensor B = random_square<float>(s);
            Tensor sycl_result = engine->create_tensor({s, s}, Core::DataType::Float32);
            Tensor reference_result = engine->create_tensor({s, s}, Core::DataType::Float32);
            Tensor bias = engine->create_tensor({s}, Core::DataType::Float32);
            fill_random<float>(bias);

            // (0, 1): the bias element is chosen by the column of C.
            const Kernels::ChannelBiasEpilogue epilogue{bias.data(), 0, 1};

            set_mode(ExecutionMode::ForceSYCL);
            Dispatch::execute_gemm(recorder(), A, B, sycl_result,
                                   Core::Transpose::NoTrans, Core::Transpose::NoTrans, 1.0, 0.0,
                                   "test.packed_gemm_bias", "test.packed_gemm_bias"_op,
                                   Dispatch::gemm_epilogue(epilogue).reads(bias));
            recording->run().value();

            set_mode(ExecutionMode::ForceReference);
            Dispatch::execute_gemm(recorder(), A, B, reference_result,
                                   Core::Transpose::NoTrans, Core::Transpose::NoTrans, 1.0, 0.0,
                                   "test.packed_gemm_bias_reference",
                                   "test.packed_gemm_bias_reference"_op,
                                   Dispatch::gemm_epilogue(epilogue).reads(bias));
            recording->run().value();

            expect_parity<float>(sycl_result, reference_result, s, K_BIAS_MAGNITUDE);
        }

        INSTANTIATE_TEST_SUITE_P(
            , Medium_PackedGemm,
            ::testing::Values(Core::DataType::Half, Core::DataType::Float32,
                              Core::DataType::Complex64),
            [](const ::testing::TestParamInfo<Core::DataType>& info) {
                return Core::to_string(info.param);
            });
    } // namespace Test
} // namespace SushiBLAS
