/**************************************************************************/
/* test_gemm.cpp                                                          */
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
 * @file test_gemm.cpp
 * @brief Tests gemm against the reference kernel, hand-computed results and its epilogue seam.
 * @author Mustafa Garip
 */

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <vector>
#include <complex>
#include <utility>
#include <algorithm>
#include <type_traits>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <SushiBLAS/SushiBLAS.hpp>
#include <SushiBLAS/blas/dispatch.hpp>
#include <SushiBLAS/core/traits.hpp>
#include <SushiBLAS/blas_seam_kernels/sycl/level3/gemm.hpp>
#include <SushiBLAS/blas_seam_kernels/reference/level3/gemm.hpp>
#include <SushiRuntime/core/sushi_ptr.hpp>
#include <SushiBLAS/engine_fixture/sushiblas_test.hpp>
#include <SushiBLAS/tensor_expectations/tensor_comparison.hpp>
#include <SushiBLAS/tensor_expectations/tensor_fill.hpp>
#include <SushiBLAS/tensor_expectations/scalar_dtype.hpp>
#include <SushiBLAS/test_runner/test_environment.hpp>
#include "support/tolerance.hpp"
#include "blas_rows.hpp"
#include "gemm_seam_fixtures.hpp"

namespace SushiBLAS
{
    namespace Test
    {
        using namespace SushiRuntime::Literals;

        namespace
        {
            /**
             * @brief Names one product the kernels are run on: the forms of A and B, the extents, the
             *        batch and the scalars. The layout and the element type come from the row.
             */
            struct GemmCase
            {
                Core::Transpose trans_a; /**< How A is read. */
                Core::Transpose trans_b; /**< How B is read. */
                int64_t m;               /**< Rows of C. */
                int64_t n;               /**< Columns of C. */
                int64_t k;               /**< Depth of the product. */
                int64_t batch;           /**< Products in the batch, stored back to back. */
                double alpha;            /**< Scale of the product. */
                double beta;             /**< Scale of the old C; exactly zero never reads it. */
            };

            /** @brief Returns the tail of a row named by the product it runs. */
            [[nodiscard]] inline std::string axis_name(const GemmCase& c)
            {
                std::string name = "_" + word(c.trans_a) + "_" + word(c.trans_b) + "_" +
                                   std::to_string(c.m) + "x" + std::to_string(c.n) + "x" +
                                   std::to_string(c.k);
                if (c.batch > 1)
                    name += "_b" + std::to_string(c.batch);
                if (c.alpha != 1.0)
                    name += "_scaled";
                if (c.beta == 0.0)
                    name += "_beta0";
                return name;
            }
        } // namespace

        /**
         * @brief A deliberately non-trivial epilogue: two broadcasts and a clamp.
         *
         * The per-row and per-column terms prove both coordinates reach the functor, and the
         * clamp makes it non-linear, so it cannot be absorbed into alpha or beta.
         */
        struct BiasReLUEpilogue
        {
            /** @brief One value per row of C, broadcast across its columns. */
            const float* row_bias;

            /** @brief One value per column of C, broadcast down its rows. */
            const float* col_bias;

            /**
             * @brief Adds both biases to value and clamps the result at zero.
             * @return max(value + row_bias[row] + col_bias[col], 0).
             */
            template<typename T>
            [[nodiscard]] T operator()(T value, int64_t row, int64_t col) const noexcept
            {
                const T biased = value + static_cast<T>(row_bias[row]) + static_cast<T>(col_bias[col]);
                return biased > T(0) ? biased : T(0);
            }
        };

        namespace
        {
            using Core::DataType;
            using Core::Layout;
            using Core::Transpose;

            /**
             * @brief Magnitude of one product of two entries drawn from [-1, 1] in both parts: the
             *        bound of |a| * |b| for complex entries of modulus at most 2^(1/2).
             */
            constexpr double K_ENTRY_PRODUCT_MAGNITUDE = 2.0;

            /// @brief Rounded operations on the path to a HALF result: the sum is carried in FLOAT32.
            constexpr std::size_t K_HALF_ROUNDINGS = 2;

            /** @brief Returns the component type of an element type. */
            template<typename T, bool = Core::is_complex_v<T>>
            struct ComponentOf
            {
                using type = T; /**< The element type of a real tensor. */
            };

            /** @brief Returns the component type of a complex element type. */
            template<typename T>
            struct ComponentOf<T, true>
            {
                using type = typename T::value_type; /**< The type of the real and imaginary part. */
            };

            /** @brief Returns the rounded operations on the path to one result of depth k. */
            template<typename T>
            [[nodiscard]] std::size_t roundings_of(int64_t k)
            {
                if constexpr (std::is_same_v<typename ComponentOf<T>::type, sycl::half>)
                    return K_HALF_ROUNDINGS;
                else
                    return static_cast<std::size_t>(k) + 2;
            }

            /** @brief Returns a value of T from a double; a complex T has a zero imaginary part. */
            template<typename T>
            [[nodiscard]] T element_of(double value)
            {
                using R = typename ComponentOf<T>::type;
                if constexpr (Core::is_complex_v<T>)
                    return T(static_cast<R>(value), R(0));
                else if constexpr (std::is_same_v<T, sycl::half>)
                    return sycl::half(static_cast<float>(value));
                else
                    return static_cast<T>(value);
            }

            /** @brief Returns the real and the imaginary part of an element as doubles. */
            template<typename T>
            [[nodiscard]] std::pair<double, double> parts_of(const T& value)
            {
                if constexpr (Core::is_complex_v<T>)
                    return {static_cast<double>(value.real()), static_cast<double>(value.imag())};
                else if constexpr (std::is_same_v<T, sycl::half>)
                    return {static_cast<double>(static_cast<float>(value)), 0.0};
                else
                    return {static_cast<double>(value), 0.0};
            }

            /** @brief Returns the modulus of an element as an element with no imaginary part. */
            template<typename T>
            [[nodiscard]] T modulus_of(const T& value)
            {
                const auto [re, im] = parts_of(value);
                return element_of<T>(std::hypot(re, im));
            }

            /** @brief Copies every element of a tensor of T to the host. */
            template<typename T>
            [[nodiscard]] std::vector<T> download_as(const Tensor& t)
            {
                std::vector<T> host(static_cast<std::size_t>(t.num_elements));
                TestEnvironment::queue().memcpy(host.data(), t.data(), host.size() * sizeof(T)).wait();
                return host;
            }

            /** @brief Holds the leading dimensions and batch strides of one case as stored. */
            struct CaseStorage
            {
                int64_t lda;   /**< Leading dimension of A as stored. */
                int64_t ldb;   /**< Leading dimension of B as stored. */
                int64_t ldc;   /**< Leading dimension of C. */
                int64_t str_a; /**< Elements of one A. */
                int64_t str_b; /**< Elements of one B. */
                int64_t str_c; /**< Elements of one C. */
                int64_t a_rows; /**< Stored rows of A. */
                int64_t a_cols; /**< Stored columns of A. */
                int64_t b_rows; /**< Stored rows of B. */
                int64_t b_cols; /**< Stored columns of B. */
            };

            /** @brief Returns how the operands of a case are stored, back to back per batch. */
            [[nodiscard]] CaseStorage storage_of(Layout layout, const GemmCase& c)
            {
                // A is stored k x m when it is read transposed and m x k otherwise; B likewise.
                const bool row_major = layout == Layout::RowMajor;
                CaseStorage s{};
                s.a_rows = Core::is_transposed(c.trans_a) ? c.k : c.m;
                s.a_cols = Core::is_transposed(c.trans_a) ? c.m : c.k;
                s.b_rows = Core::is_transposed(c.trans_b) ? c.n : c.k;
                s.b_cols = Core::is_transposed(c.trans_b) ? c.k : c.n;
                s.lda = row_major ? s.a_cols : s.a_rows;
                s.ldb = row_major ? s.b_cols : s.b_rows;
                s.ldc = row_major ? c.n : c.m;
                s.str_a = s.a_rows * s.a_cols;
                s.str_b = s.b_rows * s.b_cols;
                s.str_c = c.m * c.n;
                return s;
            }

            /** @brief Holds the reference product of a case and the magnitude its error scales with. */
            template<typename T>
            struct HostProduct
            {
                std::vector<T> value;            /**< alpha * op(A) * op(B) + beta * C, per element. */
                std::vector<double> magnitude;   /**< The sum of |a| * |b| of each element. */
            };

            /**
             * @brief Computes the reference product of a case on the host.
             * @param c0 The old contents of C.
             */
            template<typename T>
            [[nodiscard]] HostProduct<T> host_product(Layout layout, const GemmCase& c,
                                                      const std::vector<T>& a,
                                                      const std::vector<T>& b,
                                                      const std::vector<T>& c0)
            {
                const CaseStorage s = storage_of(layout, c);
                std::vector<T> abs_a(a.size());
                std::vector<T> abs_b(b.size());
                std::transform(a.begin(), a.end(), abs_a.begin(), [](const T& v) { return modulus_of(v); });
                std::transform(b.begin(), b.end(), abs_b.begin(), [](const T& v) { return modulus_of(v); });

                HostProduct<T> out;
                out.value = c0;
                std::vector<T> sums(c0.size(), T(0));
                for (int64_t i = 0; i < c.batch; ++i)
                {
                    Kernels::Reference::gemm<T>(layout, c.trans_a, c.trans_b, c.m, c.n, c.k,
                                                element_of<T>(c.alpha), a.data() + i * s.str_a, s.lda,
                                                b.data() + i * s.str_b, s.ldb, element_of<T>(c.beta),
                                                out.value.data() + i * s.str_c, s.ldc);
                    Kernels::Reference::gemm<T>(layout, c.trans_a, c.trans_b, c.m, c.n, c.k, T(1),
                                                abs_a.data() + i * s.str_a, s.lda,
                                                abs_b.data() + i * s.str_b, s.ldb, T(0),
                                                sums.data() + i * s.str_c, s.ldc);
                }
                for (const T& sum : sums)
                    out.magnitude.push_back(parts_of(sum).first);
                return out;
            }

            /**
             * @brief Expects every element of actual to lie within the rounding bound of expected.
             *
             * The bound is the rounded operations on the path to a result times the unit roundoff
             * times |alpha| * (the sum of |a| * |b|) plus |beta| times the magnitude of an old C entry.
             */
            template<typename T>
            void expect_close(const std::vector<T>& actual, const std::vector<T>& expected,
                              const std::vector<double>& magnitude, const GemmCase& c)
            {
                ASSERT_EQ(actual.size(), expected.size());
                ASSERT_EQ(actual.size(), magnitude.size());
                for (std::size_t i = 0; i < actual.size(); ++i)
                {
                    const double scale = std::abs(c.alpha) * magnitude[i] +
                                         std::abs(c.beta) * K_ENTRY_PRODUCT_MAGNITUDE;
                    const double bound = static_cast<double>(roundings_of<T>(c.k)) *
                                         unit_roundoff_of<typename ComponentOf<T>::type>() * scale;
                    const auto [got_re, got_im] = parts_of(actual[i]);
                    const auto [want_re, want_im] = parts_of(expected[i]);

                    EXPECT_TRUE(TestSupport::is_within(
                        TestSupport::measure_error(got_re, want_re, bound)))
                        << "real part of element " << i;
                    EXPECT_TRUE(TestSupport::is_within(
                        TestSupport::measure_error(got_im, want_im, bound)))
                        << "imaginary part of element " << i;
                }
            }

            /** @brief Returns the dims of a stored operand: rank 2, or rank 3 for a batch. */
            [[nodiscard]] std::vector<int64_t> operand_dims(int64_t rows, int64_t cols, int64_t batch)
            {
                if (batch > 1)
                    return {batch, rows, cols};
                return {rows, cols};
            }

            /** @brief Returns a tensor of dims and dtype from the engine of a fixture. */
            [[nodiscard]] Tensor tensor_of_dims(Engine& engine, const std::vector<int64_t>& dims,
                                                DataType dtype)
            {
                return engine.create_tensor(
                    SushiRuntime::span<const int64_t>(dims.data(), dims.size()), dtype);
            }

            /** @brief Shapes the SYCL kernel is compared with the reference on, in both layouts. */
            const std::vector<GemmCase> K_AGREEMENT_CASES = {
                // Small, rectangular, ragged and short-K shapes.
                {Transpose::NoTrans, Transpose::NoTrans, 4, 4, 4, 1, 1.0, 0.0},
                {Transpose::NoTrans, Transpose::NoTrans, 8, 16, 4, 1, 1.0, 0.0},
                {Transpose::NoTrans, Transpose::NoTrans, 37, 23, 13, 1, 1.0, 0.0},
                {Transpose::NoTrans, Transpose::NoTrans, 40, 40, 3, 1, 1.0, 0.0},
                {Transpose::NoTrans, Transpose::NoTrans, 1, 1, 1, 1, 1.0, 0.0},
                {Transpose::NoTrans, Transpose::NoTrans, 65, 2, 33, 1, 1.0, 0.0},
                {Transpose::NoTrans, Transpose::NoTrans, 2, 65, 33, 1, 1.0, 0.0},
                // Beta zero over a C that starts full of NaN.
                {Transpose::NoTrans, Transpose::NoTrans, 37, 23, 13, 1, 1.5, 0.0},
                // Deep products and a result wide enough for sixteen column lanes.
                {Transpose::NoTrans, Transpose::NoTrans, 5, 7, 4133, 1, 1.0 / 4096.0, 0.0},
                {Transpose::NoTrans, Transpose::Trans, 5, 7, 4133, 1, 1.0 / 4096.0, 0.5},
                {Transpose::Trans, Transpose::Trans, 9, 4, 4133, 1, 1.0 / 4096.0, 0.5},
                {Transpose::NoTrans, Transpose::NoTrans, 130, 131, 4133, 1, 1.0 / 4096.0, 0.5},
                {Transpose::Trans, Transpose::NoTrans, 41, 70, 4133, 1, 1.0 / 4096.0, 0.0},
                {Transpose::NoTrans, Transpose::NoTrans, 5, 6, 4100, 3, 1.0 / 4096.0, 0.5},
                // The four forms of op(A) and op(B) under scalars that scale both terms.
                {Transpose::NoTrans, Transpose::NoTrans, 17, 19, 11, 1, 1.5, 0.5},
                {Transpose::Trans, Transpose::NoTrans, 17, 19, 11, 1, 1.5, 0.5},
                {Transpose::NoTrans, Transpose::Trans, 17, 19, 11, 1, 1.5, 0.5},
                {Transpose::Trans, Transpose::Trans, 17, 19, 11, 1, 1.5, 0.5},
                // Distinct m, n and k give the three operands three distinct batch strides.
                {Transpose::NoTrans, Transpose::NoTrans, 13, 7, 5, 4, 1.5, 0.5},
                {Transpose::NoTrans, Transpose::NoTrans, 33, 17, 23, 3, 1.0, 0.0}};

            /** @brief Returns the rows of every element type, layout and agreement case. */
            [[nodiscard]] std::vector<Row<GemmCase>> agreement_rows()
            {
                return rows_of({DataType::Half, DataType::Float32, DataType::Float64,
                                DataType::Complex32, DataType::Complex64},
                               {ExecutionMode::ForceSYCL}, K_AGREEMENT_CASES);
            }

            /** @brief Returns whether a device hosts the local-tiled work-group of T. */
            template<typename T>
            [[nodiscard]] bool hosts_local_tiled(const Device::Profile& profile)
            {
                using Configuration = Kernels::GemmTile<T, Kernels::GemmTileRole::Balanced>;
                return Device::hosts_local_tile(
                    profile, {Configuration::local_mem_bytes(),
                              static_cast<std::size_t>(Configuration::K_WG_ITEMS)});
            }

            /** @brief Returns whether a device hosts the work-group of one tile shape. */
            template<typename Shape>
            [[nodiscard]] bool hosts_shape(const Device::Profile& profile)
            {
                return Device::hosts_local_tile(
                    profile, {Shape::local_mem_bytes(), static_cast<std::size_t>(Shape::K_WG_ITEMS)});
            }

            /** @brief Shapes the local-tiled kernel is compared with the reference on. */
            const std::vector<GemmCase> K_LOCAL_TILED_CASES = {
                // Each case mixes whole and edge tiles.
                {Transpose::NoTrans, Transpose::NoTrans, 128, 136, 40, 2, 1.0, 0.0},
                {Transpose::NoTrans, Transpose::NoTrans, 130, 72, 40, 1, 1.0, 0.0},
                {Transpose::NoTrans, Transpose::NoTrans, 72, 128, 40, 1, 1.0, 0.0},
                {Transpose::Trans, Transpose::NoTrans, 128, 72, 40, 1, 1.0, 0.0},
                {Transpose::Trans, Transpose::Trans, 72, 68, 48, 1, 1.0, 0.0},
                // A ragged shape under alpha, and transposes on odd extents.
                {Transpose::NoTrans, Transpose::NoTrans, 37, 45, 23, 1, 1.5, 0.0},
                {Transpose::Trans, Transpose::Trans, 19, 21, 17, 1, 1.0, 0.0}};

            /** @brief Shapes every tile shape of the catalog is compared with Balanced on. */
            [[nodiscard]] std::vector<GemmCase> tile_shape_cases()
            {
                std::vector<GemmCase> cases;
                for (const Transpose trans_a : {Transpose::NoTrans, Transpose::Trans})
                    for (const Transpose trans_b : {Transpose::NoTrans, Transpose::Trans})
                    {
                        cases.push_back({trans_a, trans_b, 256, 128, 64, 1, 1.0, 0.0});
                        cases.push_back({trans_a, trans_b, 131, 70, 45, 2, 1.0, 0.0});
                    }
                return cases;
            }
        } // namespace

        /** @brief Compares the SYCL kernel with the reference one on random operands. */
        class GemmAgreementCase : public BlasEngineStart
        {
            protected:
                /**
                 * @brief Differential check of the SYCL kernel against the reference one.
                 *
                 * The reference kernel is the specification of the operation, so every shape is
                 * judged against it. A defect the two share is caught by the hand-computed rows.
                 */
                template<typename T>
                void expect_agreement(Layout layout, const GemmCase& c)
                {
                    const DataType dtype = sushiblas_dtype<T>();
                    const CaseStorage s = storage_of(layout, c);

                    Tensor A = tensor_of_dims(*engine, operand_dims(s.a_rows, s.a_cols, c.batch), dtype);
                    Tensor B = tensor_of_dims(*engine, operand_dims(s.b_rows, s.b_cols, c.batch), dtype);
                    Tensor C_actual = tensor_of_dims(*engine, operand_dims(c.m, c.n, c.batch), dtype);
                    Tensor C_expected = tensor_of_dims(*engine, operand_dims(c.m, c.n, c.batch), dtype);
                    fill_random<T>(A);
                    fill_random<T>(B);
                    seed_output<T>(C_actual, C_expected, c.beta == 0.0);

                    set_mode(ExecutionMode::ForceSYCL);
                    SushiBLAS::BLASOps(recorder()).gemm(A, B, C_actual, c.trans_a, c.trans_b,
                                                        c.alpha, c.beta);
                    recording->run().value();

                    set_mode(ExecutionMode::ForceReference);
                    SushiBLAS::BLASOps(recorder()).gemm(A, B, C_expected, c.trans_a, c.trans_b,
                                                        c.alpha, c.beta);
                    recording->run().value();

                    const std::vector<T> zeros(static_cast<std::size_t>(C_actual.num_elements), T(0));
                    const HostProduct<T> truth =
                        host_product<T>(layout, c, download_as<T>(A), download_as<T>(B), zeros);
                    expect_close<T>(download_as<T>(C_actual), download_as<T>(C_expected),
                                    truth.magnitude, c);
                }

            private:
                /**
                 * @brief Gives both output tensors the same starting contents.
                 * @param poison Whether to fill with NaN, which must never survive a beta of
                 *        zero: reading C at all is a defect there.
                 */
                template<typename T>
                void seed_output(Tensor& actual, Tensor& expected, bool poison)
                {
                    if (poison)
                    {
                        const std::vector<float> nans(static_cast<std::size_t>(actual.num_elements),
                                                      std::numeric_limits<float>::quiet_NaN());
                        fill_tensor(actual, nans);
                        fill_tensor(expected, nans);
                        return;
                    }

                    fill_random<T>(actual);
                    TestEnvironment::queue()
                        .memcpy(expected.data(), actual.data(),
                                static_cast<std::size_t>(actual.num_elements) * sizeof(T))
                        .wait();
                }
        };

        /** @brief Runs the agreement cases over layout and element type. */
        class Medium_GemmAgreement : public BlasRowCase<GemmAgreementCase, GemmCase>
        {

        };

        TEST_P(Medium_GemmAgreement, TheSyclKernelAgreesWithTheReferenceKernel)
        {
            with_element(GetParam().type, [&](auto element)
            {
                using T = typename decltype(element)::type;
                expect_agreement<T>(GetParam().layout, GetParam().axis);
            });
        }

        INSTANTIATE_TEST_SUITE_P(
            , Medium_GemmAgreement, ::testing::ValuesIn(agreement_rows()), row_name<GemmCase>);

        namespace
        {
            /**
             * @brief Returns the storage-order values of a row-ordered matrix, or of its transpose.
             * @param row_ordered The matrix, row by row.
             * @param transposed Whether the stored matrix is the transpose of the given one.
             */
            [[nodiscard]] std::vector<float> stored_values(const std::vector<float>& row_ordered,
                                                          int64_t rows, int64_t cols, Layout layout,
                                                          bool transposed)
            {
                const int64_t out_rows = transposed ? cols : rows;
                const int64_t out_cols = transposed ? rows : cols;
                std::vector<float> stored(row_ordered.size());
                for (int64_t i = 0; i < out_rows; ++i)
                    for (int64_t j = 0; j < out_cols; ++j)
                    {
                        const float value = transposed ? row_ordered[static_cast<std::size_t>(j * cols + i)]
                                                       : row_ordered[static_cast<std::size_t>(i * cols + j)];
                        const int64_t at = layout == Layout::RowMajor ? i * out_cols + j : j * out_rows + i;
                        stored[static_cast<std::size_t>(at)] = value;
                    }
                return stored;
            }
        } // namespace

        /** @brief Pins the product of two small integer matrices on both kernels and layouts. */
        class Medium_GemmHandComputed : public BlasRowCase<BlasEngineStart, Core::Transpose>
        {

        };

        TEST_P(Medium_GemmHandComputed, TheProductOfTwoSmallMatricesIsTheHandComputedOne)
        {
            // A (2x3) = [[1,2,3],[4,5,6]], B (3x2) = [[1,0],[0,1],[1,1]], C = [[4,5],[10,11]].
            const bool a_stored_transposed = GetParam().axis == Transpose::Trans;
            const Layout layout = GetParam().layout;
            auto A = a_stored_transposed
                         ? create_initialized_tensor<float>(
                               {3, 2}, stored_values({1, 2, 3, 4, 5, 6}, 2, 3, layout, true))
                         : create_initialized_tensor<float>(
                               {2, 3}, stored_values({1, 2, 3, 4, 5, 6}, 2, 3, layout, false));
            auto B = create_initialized_tensor<float>(
                {3, 2}, stored_values({1, 0, 0, 1, 1, 1}, 3, 2, layout, false));
            auto C = engine->create_tensor({2, 2}, DataType::Float32);

            SushiBLAS::BLASOps(recorder()).gemm(A, B, C, GetParam().axis, Transpose::NoTrans);
            recording->run().value();

            expect_tensor_equal(C, stored_values({4, 5, 10, 11}, 2, 2, layout, false));
        }

        TEST_P(Medium_GemmHandComputed, AlphaScalesTheProductAndBetaTheOldC)
        {
            // The product [[4,5],[10,11]] of the test above, with alpha 2 and beta -1 on a C of ones: [[7,9],[19,21]].
            const bool a_stored_transposed = GetParam().axis == Transpose::Trans;
            const Layout layout = GetParam().layout;
            auto A = a_stored_transposed
                         ? create_initialized_tensor<float>(
                               {3, 2}, stored_values({1, 2, 3, 4, 5, 6}, 2, 3, layout, true))
                         : create_initialized_tensor<float>(
                               {2, 3}, stored_values({1, 2, 3, 4, 5, 6}, 2, 3, layout, false));
            auto B = create_initialized_tensor<float>(
                {3, 2}, stored_values({1, 0, 0, 1, 1, 1}, 3, 2, layout, false));
            auto C = create_initialized_tensor<float>({2, 2}, stored_values({1, 1, 1, 1}, 2, 2, layout, false));

            SushiBLAS::BLASOps(recorder()).gemm(A, B, C, GetParam().axis, Transpose::NoTrans, 2.0, -1.0);
            recording->run().value();

            expect_tensor_equal(C, stored_values({7, 9, 19, 21}, 2, 2, layout, false));
        }

        INSTANTIATE_TEST_SUITE_P(
            , Medium_GemmHandComputed,
            ::testing::ValuesIn(rows_of({DataType::Float32},
                                        {ExecutionMode::ForceSYCL, ExecutionMode::ForceReference},
                                        std::vector<Core::Transpose>{Transpose::NoTrans,
                                                                     Transpose::Trans})),
            row_name<Core::Transpose>);

        /** @brief Starts the device engine of the fused seam tests over their data. */
        class GemmSeamValues : public BlasEngineStart, protected GemmSeamData
        {

        };

        /** @brief Runs the fused epilogue once per kernel, in the row-major layout. */
        class Medium_GemmEpilogueModes : public BlasRowCase<GemmSeamValues, NoAxis>
        {

        };

        TEST_P(Medium_GemmEpilogueModes, ExecuteGemmFusesBiasBroadcastAndReLU)
        {
            auto A = create_initialized_tensor<float>({3, 2}, a_values());
            auto B = create_initialized_tensor<float>({2, 3}, b_values());
            auto C = engine->create_tensor({3, 3}, DataType::Float32);
            auto row_bias = create_initialized_tensor<float>({3}, {0.0f, -6.0f, 5.0f});
            auto col_bias = create_initialized_tensor<float>({3}, {1.0f, -8.0f, 3.0f});
            const BiasReLUEpilogue epilogue{row_bias.data_as<float>(), col_bias.data_as<float>()};

            Dispatch::execute_gemm(recorder(), A, B, C, Transpose::NoTrans, Transpose::NoTrans,
                                   1.0, 0.0, "test.gemm_bias_relu", "test.gemm_bias_relu"_op,
                                   epilogue);
            recording->run().value();

            expect_tensor_equal(C, expected());
        }

        namespace
        {
            /** @brief Returns the row-major rows of the float suites that run one kernel each. */
            [[nodiscard]] std::vector<Row<NoAxis>> kernel_rows()
            {
                std::vector<Row<NoAxis>> rows;
                for (const Row<NoAxis>& row :
                     plain_rows({DataType::Float32},
                                {ExecutionMode::ForceSYCL, ExecutionMode::ForceReference}))
                    if (row.layout == Layout::RowMajor)
                        rows.push_back(row);
                return rows;
            }
        } // namespace

        INSTANTIATE_TEST_SUITE_P(
            , Medium_GemmEpilogueModes, ::testing::ValuesIn(kernel_rows()), row_name<NoAxis>);

        /** @brief Runs the fused seam and the dependency probes on the default engine. */
        class Medium_GEMMSeamTest : public GemmSeamValues
        {

        };

        TEST_F(Medium_GEMMSeamTest, ExecuteGemmWithTheIdentityEpilogueIsPlainGemm)
        {
            // Level3::gemm is the seam with the identity.
            auto A = engine->create_tensor({13, 7}, DataType::Float32);
            auto B = engine->create_tensor({7, 11}, DataType::Float32);
            auto via_api = engine->create_tensor({13, 11}, DataType::Float32);
            auto via_seam = engine->create_tensor({13, 11}, DataType::Float32);
            fill_random<float>(A);
            fill_random<float>(B);

            SushiBLAS::BLASOps(recorder()).gemm(A, B, via_api, Transpose::NoTrans,
                                                Transpose::NoTrans, 1.5, 0.0);
            Dispatch::execute_gemm(recorder(), A, B, via_seam, Transpose::NoTrans,
                                   Transpose::NoTrans, 1.5, 0.0, "test.gemm_identity",
                                   "test.gemm_identity"_op, Kernels::GemmIdentityEpilogue{});
            recording->run().value();

            expect_tensor_equal(via_seam, SushiBLAS::IO().to_vector<float>(via_api));
        }

        TEST_F(Medium_GEMMSeamTest, DeclaringOperandsChangesNoArithmetic)
        {
            // Both spellings must agree to the bit.
            auto A = create_initialized_tensor<float>({3, 2}, a_values());
            auto B = create_initialized_tensor<float>({2, 3}, b_values());
            auto bias = create_initialized_tensor<float>({3}, {1.0f, -8.0f, 3.0f});
            auto bare = engine->create_tensor({3, 3}, DataType::Float32);
            auto declared = engine->create_tensor({3, 3}, DataType::Float32);

            set_mode(ExecutionMode::ForceSYCL);
            Dispatch::execute_gemm(recorder(), A, B, bare, Transpose::NoTrans, Transpose::NoTrans,
                                   1.0, 0.0, "test.gemm_bias_bare", "test.gemm_bias_bare"_op,
                                   BiasColumnEpilogue{bias.data_as<float>()});
            Dispatch::execute_gemm(recorder(), A, B, declared, Transpose::NoTrans,
                                   Transpose::NoTrans, 1.0, 0.0, "test.gemm_bias_declared",
                                   "test.gemm_bias_declared"_op,
                                   Dispatch::gemm_epilogue(
                                       BiasColumnEpilogue{bias.data_as<float>()}).reads(bias));
            recording->run().value();

            // A*B = [[1,2,3], [4,5,6], [5,7,9]] plus the column bias [1, -8, 3].
            expect_tensor_equal(declared, std::vector<float>{2, -6, 6,
                                                             5, -3, 9,
                                                             6, -1, 12});
            expect_tensor_equal(declared, SushiBLAS::IO().to_vector<float>(bare));
        }

        TEST_F(Medium_GEMMSeamTest, DeclaringAnOperandWithNoStorageIsRefused)
        {
            // A base-less region orders against nothing.
            Tensor nothing;

            EXPECT_THROW(Dispatch::gemm_epilogue(BiasColumnEpilogue{nullptr}).reads(nothing),
                         std::invalid_argument);
            EXPECT_THROW(Dispatch::gemm_epilogue(BiasColumnEpilogue{nullptr}).writes(nothing),
                         std::invalid_argument);
        }

        /** @brief Holds the direct calls to the kernels of the SYCL GEMM. */
        class Medium_GEMMKernelTest : public BlasEngineStart
        {
        };

        TEST_F(Medium_GEMMKernelTest, EpilogueFusesBiasAndReLUIntoTheStore)
        {
            // 2*(A*B), plus both biases, clamped at 0.
            auto A = create_initialized_tensor<float>({2, 2}, {1, 2, 3, 4});
            auto B = create_initialized_tensor<float>({2, 3}, {1, 0, -1, 0, 1, -1});
            auto C = engine->create_tensor({2, 3}, DataType::Float32);
            auto row_bias = create_initialized_tensor<float>({2}, {0.0f, 1.0f});
            auto col_bias = create_initialized_tensor<float>({3}, {0.5f, -1.0f, 5.0f});

            const BiasReLUEpilogue epilogue{static_cast<const float*>(row_bias.data()),
                                            static_cast<const float*>(col_bias.data())};

            Kernels::SYCL::gemm<float>(
                TestEnvironment::queue(), Layout::RowMajor, Transpose::NoTrans, Transpose::NoTrans,
                2, 3, 2, 2.0f, static_cast<const float*>(A.data()), 2, 0,
                static_cast<const float*>(B.data()), 3, 0, 0.0f, static_cast<float*>(C.data()), 3, 0,
                1, {}, epilogue).wait();

            expect_tensor_equal(C, std::vector<float>{2.5f, 3.0f, 0.0f, 7.5f, 8.0f, 0.0f});
        }

        TEST_F(Medium_GEMMKernelTest, NoLocalVariantMatchesReferenceOnRaggedShape)
        {
            const GemmCase c{Transpose::NoTrans, Transpose::NoTrans, 37, 45, 23, 1, 1.5, 0.0};
            const CaseStorage s = storage_of(Layout::RowMajor, c);

            Tensor A = engine->create_tensor({c.m, c.k}, DataType::Float32);
            Tensor B = engine->create_tensor({c.k, c.n}, DataType::Float32);
            Tensor C = engine->create_tensor({c.m, c.n}, DataType::Float32);
            fill_random<float>(A);
            fill_random<float>(B);

            const std::vector<float> zeros(static_cast<std::size_t>(s.str_c), 0.0f);
            const HostProduct<float> truth =
                host_product<float>(Layout::RowMajor, c, download_as<float>(A), download_as<float>(B),
                                    zeros);

            Kernels::SYCL::gemm<float>(
                TestEnvironment::queue(), Layout::RowMajor, Transpose::NoTrans, Transpose::NoTrans,
                c.m, c.n, c.k, static_cast<float>(c.alpha), static_cast<const float*>(A.data()),
                c.k, 0, static_cast<const float*>(B.data()), c.n, 0,
                0.0f, static_cast<float*>(C.data()), c.n, 0,
                1, {}, Kernels::GemmIdentityEpilogue{}, Kernels::GemmStrategy::no_local()).wait();

            expect_close<float>(download_as<float>(C), truth.value, truth.magnitude, c);
        }

        /** @brief Runs the local-tiled kernel once per layout, element type and product. */
        using LocalTiledRow = BlasRowCase<BlasEngineStart, GemmCase>;

        /** @brief Compares the local-tiled kernel with the reference GEMM. */
        class Medium_GemmLocalTiledAgreement : public LocalTiledRow
        {
            protected:
                /** @brief Runs the tiled kernel on a case and compares it with the reference. */
                template<typename T>
                void expect_matches_reference(Layout layout, const GemmCase& c)
                {
                    if (!hosts_local_tiled<T>(Device::profile_for(TestEnvironment::queue())))
                    {
                        GTEST_SKIP() << "device cannot host the local-tiled work-group";
                    }
                    const CaseStorage s = storage_of(layout, c);
                    const DataType dtype = sushiblas_dtype<T>();
                    Tensor A = engine->create_tensor({c.batch * s.str_a}, dtype);
                    Tensor B = engine->create_tensor({c.batch * s.str_b}, dtype);
                    Tensor C = engine->create_tensor({c.batch * s.str_c}, dtype);
                    fill_random<T>(A);
                    fill_random<T>(B);

                    const std::vector<T> zeros(static_cast<std::size_t>(c.batch * s.str_c), T(0));
                    const HostProduct<T> truth =
                        host_product<T>(layout, c, download_as<T>(A), download_as<T>(B), zeros);

                    Kernels::SYCL::gemm<T>(
                        TestEnvironment::queue(), layout, c.trans_a, c.trans_b, c.m, c.n, c.k,
                        element_of<T>(c.alpha), static_cast<const T*>(A.data()), s.lda, s.str_a,
                        static_cast<const T*>(B.data()), s.ldb, s.str_b,
                        T(0), static_cast<T*>(C.data()), s.ldc, s.str_c,
                        c.batch, {}, Kernels::GemmIdentityEpilogue{},
                        Kernels::GemmStrategy::local_tiled()).wait();

                    expect_close<T>(download_as<T>(C), truth.value, truth.magnitude, c);
                }
        };

        TEST_P(Medium_GemmLocalTiledAgreement, TheLocalTiledKernelAgreesWithTheReferenceKernel)
        {
            with_element(GetParam().type, [&](auto element)
            {
                using T = typename decltype(element)::type;
                expect_matches_reference<T>(GetParam().layout, GetParam().axis);
            });
        }

        INSTANTIATE_TEST_SUITE_P(
            , Medium_GemmLocalTiledAgreement,
            ::testing::ValuesIn(rows_of({DataType::Float32, DataType::Float64, DataType::Half},
                                        {ExecutionMode::ForceSYCL}, K_LOCAL_TILED_CASES)),
            row_name<GemmCase>);

        /** @brief Compares every tile shape of the catalog with Balanced, bit for bit. */
        class Medium_GemmTileShapeBits : public LocalTiledRow
        {
            protected:
                /** @brief Runs the tiled kernel in the shape of a role on the operands and returns C. */
                template<typename T>
                [[nodiscard]] std::vector<T> local_tiled_with(Kernels::GemmTileRole role,
                                                              Layout layout, const GemmCase& c,
                                                              const Tensor& A, const Tensor& B)
                {
                    const CaseStorage s = storage_of(layout, c);
                    Tensor C = engine->create_tensor({c.batch * s.str_c}, sushiblas_dtype<T>());

                    Kernels::SYCL::gemm<T>(
                        TestEnvironment::queue(), layout, c.trans_a, c.trans_b, c.m, c.n, c.k, T(1),
                        static_cast<const T*>(A.data()), s.lda, s.str_a,
                        static_cast<const T*>(B.data()), s.ldb, s.str_b,
                        T(0), static_cast<T*>(C.data()), s.ldc, s.str_c,
                        c.batch, {}, Kernels::GemmIdentityEpilogue{},
                        Kernels::GemmStrategy::local_tiled(role)).wait();
                    return download_as<T>(C);
                }

                /** @brief Expects the Wide and Compact shapes to give the bits of Balanced. */
                template<typename T>
                void expect_every_shape_gives_balanced_bits(Layout layout, const GemmCase& c)
                {
                    using Wide = Kernels::GemmTile<T, Kernels::GemmTileRole::Wide>;
                    using Compact = Kernels::GemmTile<T, Kernels::GemmTileRole::Compact>;

                    const Device::Profile& profile = Device::profile_for(TestEnvironment::queue());
                    if (!hosts_local_tiled<T>(profile))
                    {
                        GTEST_SKIP() << "device cannot host the local-tiled work-group";
                    }
                    const CaseStorage s = storage_of(layout, c);
                    Tensor A = engine->create_tensor({c.batch * s.str_a}, sushiblas_dtype<T>());
                    Tensor B = engine->create_tensor({c.batch * s.str_b}, sushiblas_dtype<T>());
                    fill_random<T>(A);
                    fill_random<T>(B);

                    // One fused multiply-add per k, k rising, in every shape.
                    const std::vector<T> want = local_tiled_with<T>(Kernels::GemmTileRole::Balanced, layout, c, A, B);
                    const std::size_t bytes = want.size() * sizeof(T);
                    if (hosts_shape<Wide>(profile))
                    {
                        const std::vector<T> wide = local_tiled_with<T>(Kernels::GemmTileRole::Wide, layout, c, A, B);
                        EXPECT_EQ(std::memcmp(wide.data(), want.data(), bytes), 0) << "Wide";
                    }
                    if (hosts_shape<Compact>(profile))
                    {
                        const std::vector<T> compact = local_tiled_with<T>(Kernels::GemmTileRole::Compact, layout, c, A, B);
                        EXPECT_EQ(std::memcmp(compact.data(), want.data(), bytes), 0) << "Compact";
                    }
                }
        };

        TEST_P(Medium_GemmTileShapeBits, EveryTileShapeGivesTheBitsOfBalanced)
        {
            with_element(GetParam().type, [&](auto element)
            {
                using T = typename decltype(element)::type;
                expect_every_shape_gives_balanced_bits<T>(GetParam().layout, GetParam().axis);
            });
        }

        INSTANTIATE_TEST_SUITE_P(
            , Medium_GemmTileShapeBits,
            ::testing::ValuesIn(rows_of({DataType::Float32, DataType::Float64, DataType::Complex32},
                                        {ExecutionMode::ForceSYCL}, tile_shape_cases())),
            row_name<GemmCase>);
    } // namespace Test
} // namespace SushiBLAS
