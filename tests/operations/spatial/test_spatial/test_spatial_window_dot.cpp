/**************************************************************************/
/* test_spatial_window_dot.cpp                                            */
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
 * @file test_spatial_window_dot.cpp
 * @brief Tests the per-channel weighted window sum and its two gradients in both forced modes.
 * @author Mustafa Garip
 */

#include <cmath>
#include <string>
#include <vector>
#include <functional>
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <SushiBLAS/SushiBLAS.hpp>
#include <SushiBLAS/arithmetic/accumulator.hpp>
#include <SushiBLAS/reduction/reduce_axis.hpp>
#include <SushiBLAS/launch/device_policy.hpp>
#include <SushiBLAS/engine_fixture/host_engine_test.hpp>
#include <SushiBLAS/engine_fixture/sushiblas_test.hpp>
#include <SushiBLAS/tensor_expectations/tensor_comparison.hpp>
#include <SushiBLAS/tensor_expectations/tensor_fill.hpp>
#include <SushiBLAS/tensor_expectations/tolerance_bounds.hpp>
#include <SushiBLAS/tensor_expectations/scalar_dtype.hpp>
#include <SushiBLAS/typed_suite_naming/default_type_names.hpp>
#include <SushiBLAS/test_runner/test_environment.hpp>
#include <SushiBLAS/blas_test_oracle/blas_truth.hpp>
#include "support/forced_mode_checks.hpp"

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            /// @brief The value an output holds before its operation runs.
            constexpr double K_SENTINEL = 7.0;

            /// @brief The largest magnitude up to which HALF holds every integer.
            constexpr double K_HALF_EXACT_INTEGER_LIMIT = 2048.0;

            /** @brief Returns the name a failure message gives a forced mode. */
            const char* forced_mode_name(ExecutionMode mode) noexcept
            {
                return mode == ExecutionMode::ForceReference ? "ForceReference" : "ForceSYCL";
            }

            /** @brief Returns how many elements a tensor of dims holds. */
            std::size_t element_count(const std::vector<int64_t>& dims) noexcept
            {
                int64_t count = 1;
                for (const int64_t extent : dims)
                    count *= extent;
                return static_cast<std::size_t>(count);
            }

            /**
             * @brief Holds one geometry with the extents of the three tensors it relates.
             */
            struct WindowCase
            {
                /** @brief Returns the NHWC extents of the tensor in the input role. */
                [[nodiscard]] std::vector<int64_t> input_dims() const
                {
                    return {batch, in_h, in_w, channels};
                }

                /** @brief Returns the NHWC extents of the tensor in the output role. */
                [[nodiscard]] std::vector<int64_t> output_dims() const
                {
                    return {batch, out_h, out_w, channels};
                }

                /** @brief Returns the `[kernel_h, kernel_w, C]` extents of the taps. */
                [[nodiscard]] std::vector<int64_t> tap_dims() const
                {
                    return {geometry.kernel_h, geometry.kernel_w, channels};
                }

                /** @brief Returns how many windows one tap of one channel is summed over. */
                [[nodiscard]] std::size_t windows() const noexcept
                {
                    return static_cast<std::size_t>(batch * out_h * out_w);
                }

                const char* id = "";            /**< The case's name in a failure message. */
                Spatial::ConvGeometry geometry; /**< Window, stride, dilation, leading padding. */
                int64_t batch = 0;              /**< N. */
                int64_t in_h = 0;               /**< H. */
                int64_t in_w = 0;               /**< W. */
                int64_t channels = 0;           /**< C. */
                int64_t out_h = 0;              /**< H_o, stated and not derived. */
                int64_t out_w = 0;              /**< W_o, stated and not derived. */
            };

            /** @brief Returns a geometry from its eight fields, window first and padding last. */
            Spatial::ConvGeometry geometry_of(int64_t kernel_h, int64_t kernel_w, int64_t stride_h,
                                              int64_t stride_w, int64_t dilation_h,
                                              int64_t dilation_w, int64_t pad_top,
                                              int64_t pad_left) noexcept
            {
                Spatial::ConvGeometry g;
                g.kernel_h = kernel_h;
                g.kernel_w = kernel_w;
                g.stride_h = stride_h;
                g.stride_w = stride_w;
                g.dilation_h = dilation_h;
                g.dilation_w = dilation_w;
                g.pad_top = pad_top;
                g.pad_left = pad_left;
                return g;
            }

            /** @brief Returns the five table-driven geometries, G1 to G5. */
            std::vector<WindowCase> table_cases()
            {
                using Spatial::output_extent;

                return {
                    {"G1", geometry_of(3, 3, 1, 1, 1, 1, 1, 1), 2, 5, 6, 3,
                     output_extent(5, 1, 1, 1, 3, 1), output_extent(6, 1, 1, 1, 3, 1)},
                    {"G2", geometry_of(2, 3, 2, 1, 1, 2, 1, 2), 2, 7, 5, 4,
                     output_extent(7, 1, 1, 1, 2, 2), output_extent(5, 2, 2, 2, 3, 1)},
                    {"G3", geometry_of(3, 1, 1, 3, 2, 1, 0, 1), 3, 6, 7, 1,
                     output_extent(6, 0, 0, 2, 3, 1), output_extent(7, 1, 1, 1, 1, 3)},
                    {"G4", geometry_of(1, 1, 1, 1, 1, 1, 0, 0), 1, 3, 4, 5, 3, 4},
                    {"G5", geometry_of(3, 3, 2, 2, 1, 1, 1, 1), 2, 7, 7, 2,
                     output_extent(7, 1, 0, 1, 3, 2), output_extent(7, 1, 0, 1, 3, 2)},
                };
            }

            /**
             * @brief Visits every tap of every window that lies inside the input.
             * @tparam Visit Callable `(std::size_t output, std::size_t input, std::size_t tap)`
             *         taking the three dense element indices.
             */
            template<typename Visit>
            void for_each_live_tap(const WindowCase& c, Visit&& visit)
            {
                const Spatial::ConvGeometry& g = c.geometry;

                for (int64_t n = 0; n < c.batch; ++n)
                for (int64_t i = 0; i < c.out_h; ++i)
                for (int64_t j = 0; j < c.out_w; ++j)
                for (int64_t r = 0; r < g.kernel_h; ++r)
                for (int64_t s = 0; s < g.kernel_w; ++s)
                {
                    const int64_t h = i * g.stride_h - g.pad_top + r * g.dilation_h;
                    const int64_t w = j * g.stride_w - g.pad_left + s * g.dilation_w;
                    if (h < 0 || h >= c.in_h || w < 0 || w >= c.in_w)
                        continue;

                    const int64_t output = ((n * c.out_h + i) * c.out_w + j) * c.channels;
                    const int64_t input = ((n * c.in_h + h) * c.in_w + w) * c.channels;
                    const int64_t tap = (r * g.kernel_w + s) * c.channels;

                    for (int64_t ch = 0; ch < c.channels; ++ch)
                        visit(static_cast<std::size_t>(output + ch),
                              static_cast<std::size_t>(input + ch),
                              static_cast<std::size_t>(tap + ch));
                }
            }

            /** @brief Returns the window sums the formula gives for x and weights. */
            std::vector<double> forward_by_formula(const WindowCase& c,
                                                   const std::vector<double>& x,
                                                   const std::vector<double>& weights)
            {
                std::vector<double> out(element_count(c.output_dims()), 0.0);
                for_each_live_tap(c, [&](std::size_t output, std::size_t input, std::size_t tap)
                {
                    out[output] += x[input] * weights[tap];
                });
                return out;
            }

            /** @brief Returns the input gradient a scatter of `dy * weights` over taps gives. */
            std::vector<double> input_gradient_by_scatter(const WindowCase& c,
                                                          const std::vector<double>& dy,
                                                          const std::vector<double>& weights)
            {
                std::vector<double> dx(element_count(c.input_dims()), 0.0);
                for_each_live_tap(c, [&](std::size_t output, std::size_t input, std::size_t tap)
                {
                    dx[input] += dy[output] * weights[tap];
                });
                return dx;
            }

            /** @brief Returns the weight gradient the formula gives for dy and x. */
            std::vector<double> weight_gradient_by_formula(const WindowCase& c,
                                                           const std::vector<double>& dy,
                                                           const std::vector<double>& x)
            {
                std::vector<double> dweights(element_count(c.tap_dims()), 0.0);
                for_each_live_tap(c, [&](std::size_t output, std::size_t input, std::size_t tap)
                {
                    dweights[tap] += dy[output] * x[input];
                });
                return dweights;
            }

            /** @brief Returns `((i * multiplier) % modulus) - shift` for i below count. */
            std::vector<double> residue_pattern(std::size_t count, int64_t multiplier,
                                                int64_t modulus, int64_t shift)
            {
                std::vector<double> values(count);
                for (std::size_t i = 0; i < count; ++i)
                    values[i] = static_cast<double>(
                        (static_cast<int64_t>(i) * multiplier) % modulus - shift);
                return values;
            }

            /** @brief Returns the integer input pattern, values from -11 to 11. */
            std::vector<double> input_pattern(const std::vector<int64_t>& dims)
            {
                return residue_pattern(element_count(dims), 37, 23, 11);
            }

            /** @brief Returns the integer tap pattern, values from -3 to 3. */
            std::vector<double> tap_pattern(const std::vector<int64_t>& dims)
            {
                return residue_pattern(element_count(dims), 13, 7, 3);
            }

            /** @brief Returns the integer output-gradient pattern, values from -2 to 2. */
            std::vector<double> gradient_pattern(const std::vector<int64_t>& dims)
            {
                return residue_pattern(element_count(dims), 29, 5, 2);
            }

            /** @brief Returns `0.37 * v + 0.011` for every v; no element type holds it exactly. */
            std::vector<double> fractional(std::vector<double> values)
            {
                for (double& v : values)
                    v = 0.37 * v + 0.011;
                return values;
            }

            /** @brief Returns values rounded to T. */
            template<typename T>
            std::vector<T> elements_from(const std::vector<double>& values)
            {
                std::vector<T> elements(values.size());
                for (std::size_t i = 0; i < values.size(); ++i)
                {
                    if constexpr (std::is_same_v<T, sycl::half>)
                        elements[i] = sycl::half(static_cast<float>(values[i]));
                    else
                        elements[i] = static_cast<T>(values[i]);
                }
                return elements;
            }

            /** @brief Returns elements widened to double. */
            template<typename T>
            std::vector<double> widened(const std::vector<T>& elements)
            {
                std::vector<double> values(elements.size());
                for (std::size_t i = 0; i < elements.size(); ++i)
                    values[i] = Detail::to_double(elements[i]);
                return values;
            }

            /**
             * @brief Returns a dense tensor of dims holding values rounded to T.
             * @param values One value per element; empty when dims holds a zero.
             */
            template<typename T>
            Tensor tensor_from(Engine& engine, const std::vector<int64_t>& dims,
                               const std::vector<double>& values)
            {
                Tensor t = engine.create_tensor(
                    SushiRuntime::span<const int64_t>(dims.data(), dims.data() + dims.size()),
                    sushiblas_dtype<T>());

                if (t.num_elements > 0)
                    SushiBLAS::IO().from_vector(t, elements_from<T>(values));
                return t;
            }

            /** @brief Returns a dense tensor of dims with every element at K_SENTINEL. */
            template<typename T>
            Tensor sentinel_tensor(Engine& engine, const std::vector<int64_t>& dims)
            {
                return tensor_from<T>(engine, dims,
                                      std::vector<double>(element_count(dims), K_SENTINEL));
            }

            /** @brief Returns the window sums the library writes for in and weights. */
            template<typename T>
            std::vector<T> forward_of(Engine& engine, const WindowCase& c, const Tensor& in,
                                      const Tensor& weights)
            {
                Tensor out = sentinel_tensor<T>(engine, c.output_dims());
                RecordingFixture graph(TestEnvironment::runtime(), engine);
                SushiBLAS::SpatialOps(graph.recorder()).window_dot(in, weights, out, c.geometry);
                graph.run().value();
                return SushiBLAS::IO().to_vector<T>(out);
            }

            /** @brief Returns the input gradient the library writes for dy and weights. */
            template<typename T>
            std::vector<T> input_gradient_of(Engine& engine, const WindowCase& c, const Tensor& dy,
                                             const Tensor& weights)
            {
                Tensor dx = sentinel_tensor<T>(engine, c.input_dims());
                RecordingFixture graph(TestEnvironment::runtime(), engine);
                SushiBLAS::SpatialOps(graph.recorder())
                    .window_dot_backward(dy, weights, dx, c.geometry);
                graph.run().value();
                return SushiBLAS::IO().to_vector<T>(dx);
            }

            /** @brief Returns the weight gradient the library writes for dy and in. */
            template<typename T>
            std::vector<T> weight_gradient_of(Engine& engine, const WindowCase& c, const Tensor& dy,
                                              const Tensor& in)
            {
                Tensor dweights = sentinel_tensor<T>(engine, c.tap_dims());
                RecordingFixture graph(TestEnvironment::runtime(), engine);
                SushiBLAS::SpatialOps(graph.recorder())
                    .window_dot_weights_backward(dy, in, dweights, c.geometry);
                graph.run().value();
                return SushiBLAS::IO().to_vector<T>(dweights);
            }

            /**
             * @brief Expects every element to equal its expectation, with no tolerance.
             * @param expected Every value must be exact in T.
             */
            template<typename T>
            void expect_exact(const std::vector<T>& got, const std::vector<double>& expected,
                              const char* quantity)
            {
                ASSERT_EQ(got.size(), expected.size()) << quantity;
                for (std::size_t i = 0; i < expected.size(); ++i)
                    EXPECT_EQ(Detail::to_double(got[i]), expected[i])
                        << quantity << " element " << i;
            }

            /// @brief The smallest normal HALF magnitude, 2^-14.
            constexpr double K_HALF_MIN_NORMAL = 0x1p-14;

            /// @brief Magnitude below which the host conversion to HALF truncates where a device rounds.
            constexpr double K_HOST_HALF_TRUNCATION_LIMIT = 2.0 * K_HALF_MIN_NORMAL;

            /// @brief The HALF spacing below the limit: 2^-14 times 2^-10, which is twice the unit roundoff.
            constexpr double K_HALF_STEP_BELOW_LIMIT = K_HALF_MIN_NORMAL * 2.0 * K_HALF_UNIT_ROUNDOFF;

            /**
             * @brief Returns whether a reference and a device element agree.
             * @note A HALF pair below K_HOST_HALF_TRUNCATION_LIMIT may differ by one HALF step, the
             *       host conversion truncating where the device rounds; every other pair holds the
             *       same bits.
             * @see tests/operations/spatial/README.md
             */
            template<typename T>
            bool holds_equal_bits(const T& reference, const T& device)
            {
                if (std::memcmp(&reference, &device, sizeof(T)) == 0)
                    return true;
                if (!std::is_same_v<T, sycl::half>)
                    return false;

                const double reference_value = Detail::to_double(reference);
                const double device_value = Detail::to_double(device);
                return std::fabs(device_value) < K_HOST_HALF_TRUNCATION_LIMIT &&
                       std::fabs(device_value - reference_value) <= K_HALF_STEP_BELOW_LIMIT;
            }

            /** @brief Expects two results to hold the same bits, element for element. */
            template<typename T>
            void expect_equal_bits(const std::vector<T>& reference, const std::vector<T>& device,
                                   const char* quantity)
            {
                ASSERT_EQ(reference.size(), device.size()) << quantity;
                for (std::size_t i = 0; i < reference.size(); ++i)
                    EXPECT_TRUE(holds_equal_bits(reference[i], device[i]))
                        << quantity << " element " << i << ": reference "
                        << Detail::to_double(reference[i]) << ", device "
                        << Detail::to_double(device[i]);
            }

            /** @brief Returns the sum of the elementwise products of two equally long vectors. */
            double dot_of(const std::vector<double>& a, const std::vector<double>& b)
            {
                double sum = 0.0;
                for (std::size_t i = 0; i < a.size(); ++i)
                    sum += a[i] * b[i];
                return sum;
            }

            /**
             * @brief Returns the bound on a weight-gradient element's distance from its truth.
             * @param terms How many products the element sums.
             * @see kernels/README.md
             */
            template<typename T>
            double weight_gradient_bound(ExecutionMode mode, std::size_t terms,
                                         const TruthValue& truth)
            {
                const double store = unit_roundoff<T>() * std::fabs(truth.value);

                if (mode == ExecutionMode::ForceSYCL)
                    return rounding_bound(16, unit_roundoff<Kernels::Accumulator<T>>(),
                                          truth.magnitude) + store;

                const std::size_t roundings = std::is_same_v<T, double> ? 5 : terms + 2;
                return rounding_bound(roundings, unit_roundoff<double>(), truth.magnitude) + store;
            }

            /** @brief Expects every weight-gradient element to lie within its bound of the truth. */
            template<typename T>
            void expect_within_bounds(ExecutionMode mode, const std::vector<double>& got,
                                      const std::vector<TruthSum>& truths, std::size_t terms)
            {
                ASSERT_EQ(got.size(), truths.size());
                for (std::size_t tap = 0; tap < truths.size(); ++tap)
                {
                    const TruthValue truth = truths[tap].value();
                    EXPECT_LE(std::fabs(got[tap] - truth.value),
                              weight_gradient_bound<T>(mode, terms, truth))
                        << "dweights element " << tap << ": got " << got[tap] << ", truth "
                        << truth.value << ", magnitude " << truth.magnitude;
                }
            }

            /** @brief Expects every value in expected to be an integer HALF holds exactly. */
            void expect_integers_exact_in_half(const std::vector<double>& expected, const char* id)
            {
                for (const double truth : expected)
                    EXPECT_LT(std::fabs(truth), K_HALF_EXACT_INTEGER_LIMIT) << id;
            }

            /** @brief Returns the one-row window of the hand-worked values. */
            WindowCase hand_worked_case()
            {
                return WindowCase{"hand", geometry_of(1, 2, 1, 2, 1, 2, 0, 1), 1, 1, 5, 1,
                                  1, Spatial::output_extent(5, 1, 1, 2, 2, 2)};
            }

            /**
             * @brief Returns the dense copy of the interior of a larger input.
             * @param larger_values The values of an input with one more image and two more columns.
             * @details The copy skips image 0 and one column on each side of every row.
             */
            std::vector<double> interior_copy(const WindowCase& c,
                                              const std::vector<double>& larger_values)
            {
                std::vector<double> dense_values;
                dense_values.reserve(element_count(c.input_dims()));
                for (int64_t n = 1; n <= c.batch; ++n)
                    for (int64_t h = 0; h < c.in_h; ++h)
                        for (int64_t w = 1; w <= c.in_w; ++w)
                            for (int64_t ch = 0; ch < c.channels; ++ch)
                                dense_values.push_back(larger_values[static_cast<std::size_t>(
                                    ((n * c.in_h + h) * (c.in_w + 2) + w) * c.channels + ch)]);
                return dense_values;
            }

            /** @brief Names one malformed window_dot call and the operand its refusal names. */
            struct MalformedRow
            {
                /** @brief The row name, which gtest takes as the suffix of the test. */
                std::string name;

                /** @brief The operand the message names. */
                const char* operand;

                /** @brief Records the malformed call. */
                std::function<void(SpatialOps&, Engine&)> record;
            };

            /** @brief Returns a 1 x 4 x 4 x 2 float tensor of the given dtype and batch. */
            Tensor malformed_image(Engine& e, Core::DataType dtype, int64_t batch)
            {
                return e.create_tensor({batch, 4, 4, 2}, dtype);
            }

            /** @brief Returns the 3 x 3 float taps of the given channel count. */
            Tensor malformed_taps(Engine& e, int64_t channels)
            {
                return e.create_tensor({3, 3, channels}, Core::DataType::Float32);
            }

            /** @brief Returns the rows of the malformed window_dot calls. */
            std::vector<MalformedRow> malformed_rows()
            {
                const Spatial::ConvGeometry g = geometry_of(3, 3, 1, 1, 1, 1, 1, 1);

                return {
                    {"WeightsOfRankTwo", "weights",
                     [g](SpatialOps& s, Engine& e)
                     {
                         Tensor flat = e.create_tensor({3, 3}, Core::DataType::Float32);
                         Tensor in = malformed_image(e, Core::DataType::Float32, 1);
                         Tensor out = malformed_image(e, Core::DataType::Float32, 1);
                         s.window_dot(in, flat, out, g);
                     }},
                    {"WeightsOfTheWrongChannelCount", "weights",
                     [g](SpatialOps& s, Engine& e)
                     {
                         Tensor in = malformed_image(e, Core::DataType::Float32, 1);
                         Tensor wide = malformed_taps(e, 3);
                         Tensor out = malformed_image(e, Core::DataType::Float32, 1);
                         s.window_dot(in, wide, out, g);
                     }},
                    {"OutputOfAnotherDtype", "out",
                     [g](SpatialOps& s, Engine& e)
                     {
                         Tensor in = malformed_image(e, Core::DataType::Float32, 1);
                         Tensor weights = malformed_taps(e, 2);
                         Tensor out = malformed_image(e, Core::DataType::Float64, 1);
                         s.window_dot(in, weights, out, g);
                     }},
                    {"OutputOfAnotherBatch", "out",
                     [g](SpatialOps& s, Engine& e)
                     {
                         Tensor in = malformed_image(e, Core::DataType::Float32, 1);
                         Tensor weights = malformed_taps(e, 2);
                         Tensor out = malformed_image(e, Core::DataType::Float32, 2);
                         s.window_dot(in, weights, out, g);
                     }},
                    {"OutputAliasingTheInput", "out",
                     [g](SpatialOps& s, Engine& e)
                     {
                         Tensor in = malformed_image(e, Core::DataType::Float32, 1);
                         Tensor weights = malformed_taps(e, 2);
                         Tensor aliased = in;
                         s.window_dot(in, weights, aliased, g);
                     }},
                    {"GeometryOfStrideZero", "geometry",
                     [g](SpatialOps& s, Engine& e)
                     {
                         Tensor in = malformed_image(e, Core::DataType::Float32, 1);
                         Tensor weights = malformed_taps(e, 2);
                         Tensor out = malformed_image(e, Core::DataType::Float32, 1);
                         Spatial::ConvGeometry unstrided = g;
                         unstrided.stride_h = 0;
                         s.window_dot(in, weights, out, unstrided);
                     }}};
            }

            /** @brief Names a row by its name. */
            std::string malformed_row_name(const ::testing::TestParamInfo<MalformedRow>& info)
            {
                return info.param.name;
            }
        } // namespace

        /**
         * @brief Runs the window-sum operations in one element type, skipped where the device
         *        lacks it.
         */
        template<typename T>
        class Medium_SpatialWindowDotTyped : public SushiBLASTest
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

        using WindowDotTypes = ::testing::Types<sycl::half, float, double>;
        SB_TYPED_TEST_SUITE(Medium_SpatialWindowDotTyped, WindowDotTypes);

        /**
         * @brief Runs the window-sum cases that fix one element type.
         */
        class Medium_SpatialWindowDot : public SushiBLASTest
        {
        };

        /** @brief Runs the adjoint check in FLOAT64, skipped where the device lacks it. */
        class Medium_SpatialWindowDotAdjoint : public SushiBLASTest
        {
            protected:
                /** @brief Builds the engine and skips the case on a device without fp64. */
                void SetUp() override
                {
                    SushiBLASTest::SetUp();

                    if (!device_computes<double>())
                        GTEST_SKIP() << "the device does not report the fp64 aspect";
                }
        };

        /** @brief Runs the window-sum operations on the host engine. */
        class Small_SpatialWindowDot : public HostEngineTest
        {
        };

        /** @brief Records one malformed window_dot call per row on a host engine. */
        class Small_SpatialWindowDotMalformed : public HostEngineTest,
                                                public ::testing::WithParamInterface<MalformedRow>
        {
        };

        /** @brief Holds a device-only input, skipped where the device gives no such allocation. */
        class Medium_SpatialWindowDotDeviceOnly : public SushiBLASTest
        {
            protected:
                /** @brief Builds the engine, the device-only input and the other operands. */
                void SetUp() override
                {
                    SushiBLASTest::SetUp();

                    const WindowCase c = table_cases()[0];
                    geometry = c.geometry;

                    try
                    {
                        input = engine->create_tensor({c.batch, c.in_h, c.in_w, c.channels},
                                                      Core::DataType::Float32,
                                                      SushiRuntime::Residency::Device);
                    }
                    catch (const std::exception& error)
                    {
                        GTEST_SKIP() << "the device gives no device-only USM: " << error.what();
                    }

                    const sycl::context context = TestEnvironment::queue().get_context();
                    if (sycl::get_pointer_type(input->data(), context) != sycl::usm::alloc::device)
                        GTEST_SKIP() << "the allocation is not device-only USM";

                    fill_tensor(*input, std::vector<float>(element_count(c.input_dims()), 1.0f));
                    weights = tensor_from<float>(*engine, c.tap_dims(), tap_pattern(c.tap_dims()));
                    output = sentinel_tensor<float>(*engine, c.output_dims());
                }

                /** @brief The geometry of the call. */
                Spatial::ConvGeometry geometry;

                /** @brief The device-only input. */
                std::optional<Tensor> input;

                /** @brief The taps. */
                Tensor weights;

                /** @brief The output. */
                Tensor output;
        };


        /** @brief Tests that a strided, dilated, padded 1 by 2 window gives the hand-worked window sums. */
        TYPED_TEST(Medium_SpatialWindowDotTyped, TheOneRowWindowGivesItsHandWorkedForward)
        {
            using T = TypeParam;
            Engine& engine = *this->engine;
            const WindowCase c = hand_worked_case();
            ASSERT_EQ(c.out_w, 3);

            for (const ExecutionMode mode : K_FORCED_MODES)
            {
                SCOPED_TRACE(forced_mode_name(mode));
                engine.set_mode(mode);

                const Tensor in = tensor_from<T>(engine, c.input_dims(), {1, 2, 3, 4, 5});
                const Tensor weights = tensor_from<T>(engine, c.tap_dims(), {10, 1});

                expect_exact(forward_of<T>(engine, c, in, weights), {2, 24, 40}, "out");
            }
        }

        /** @brief Tests that the same window gives the hand-worked input gradient. */
        TYPED_TEST(Medium_SpatialWindowDotTyped, TheOneRowWindowGivesItsHandWorkedInputGradient)
        {
            using T = TypeParam;
            Engine& engine = *this->engine;
            const WindowCase c = hand_worked_case();

            for (const ExecutionMode mode : K_FORCED_MODES)
            {
                SCOPED_TRACE(forced_mode_name(mode));
                engine.set_mode(mode);

                const Tensor weights = tensor_from<T>(engine, c.tap_dims(), {10, 1});
                const Tensor dy = tensor_from<T>(engine, c.output_dims(), {1, 2, 3});

                expect_exact(input_gradient_of<T>(engine, c, dy, weights), {0, 21, 0, 32, 0}, "dx");
            }
        }

        /** @brief Tests that the same window gives the hand-worked weight gradient. */
        TYPED_TEST(Medium_SpatialWindowDotTyped, TheOneRowWindowGivesItsHandWorkedWeightGradient)
        {
            using T = TypeParam;
            Engine& engine = *this->engine;
            const WindowCase c = hand_worked_case();

            for (const ExecutionMode mode : K_FORCED_MODES)
            {
                SCOPED_TRACE(forced_mode_name(mode));
                engine.set_mode(mode);

                const Tensor in = tensor_from<T>(engine, c.input_dims(), {1, 2, 3, 4, 5});
                const Tensor dy = tensor_from<T>(engine, c.output_dims(), {1, 2, 3});

                expect_exact(weight_gradient_of<T>(engine, c, dy, in), {16, 10}, "dweights");
            }
        }

        /** @brief Tests that the forward equals a host loop written from its formula, G1 to G5. */
        TYPED_TEST(Medium_SpatialWindowDotTyped, ForwardMatchesTheFormula)
        {
            using T = TypeParam;
            Engine& engine = *this->engine;

            for (const WindowCase& c : table_cases())
            {
                const std::vector<double> x = input_pattern(c.input_dims());
                const std::vector<double> w = tap_pattern(c.tap_dims());
                const std::vector<double> expected = forward_by_formula(c, x, w);

                for (const ExecutionMode mode : K_FORCED_MODES)
                {
                    SCOPED_TRACE(std::string(c.id) + " " + forced_mode_name(mode));
                    engine.set_mode(mode);

                    const Tensor in = tensor_from<T>(engine, c.input_dims(), x);
                    const Tensor weights = tensor_from<T>(engine, c.tap_dims(), w);

                    expect_exact(forward_of<T>(engine, c, in, weights), expected, "out");
                }
            }
        }

        /** @brief Tests that the gathered input gradient equals a host scatter, for G1 to G5. */
        TYPED_TEST(Medium_SpatialWindowDotTyped, InputGradientMatchesAHostScatter)
        {
            using T = TypeParam;
            Engine& engine = *this->engine;

            for (const WindowCase& c : table_cases())
            {
                const std::vector<double> g = gradient_pattern(c.output_dims());
                const std::vector<double> w = tap_pattern(c.tap_dims());
                const std::vector<double> expected = input_gradient_by_scatter(c, g, w);

                for (const ExecutionMode mode : K_FORCED_MODES)
                {
                    SCOPED_TRACE(std::string(c.id) + " " + forced_mode_name(mode));
                    engine.set_mode(mode);

                    const Tensor dy = tensor_from<T>(engine, c.output_dims(), g);
                    const Tensor weights = tensor_from<T>(engine, c.tap_dims(), w);

                    expect_exact(input_gradient_of<T>(engine, c, dy, weights), expected, "dx");
                }
            }
        }

        /** @brief Tests that the weight gradient equals a host loop of its formula, G1 to G5. */
        TYPED_TEST(Medium_SpatialWindowDotTyped, WeightGradientMatchesTheFormula)
        {
            using T = TypeParam;
            Engine& engine = *this->engine;

            for (const WindowCase& c : table_cases())
            {
                const std::vector<double> g = gradient_pattern(c.output_dims());
                const std::vector<double> x = input_pattern(c.input_dims());
                const std::vector<double> expected = weight_gradient_by_formula(c, g, x);

                expect_integers_exact_in_half(expected, c.id);

                for (const ExecutionMode mode : K_FORCED_MODES)
                {
                    SCOPED_TRACE(std::string(c.id) + " " + forced_mode_name(mode));
                    engine.set_mode(mode);

                    const Tensor dy = tensor_from<T>(engine, c.output_dims(), g);
                    const Tensor in = tensor_from<T>(engine, c.input_dims(), x);

                    expect_exact(weight_gradient_of<T>(engine, c, dy, in), expected, "dweights");
                }
            }
        }

        /** @brief Tests that the three kernels give one inner product, as adjoints must. */
        TEST_F(Medium_SpatialWindowDotAdjoint, TheThreeKernelsAreAdjoint)
        {
            for (const WindowCase& c : table_cases())
            {
                const std::vector<double> x = input_pattern(c.input_dims());
                const std::vector<double> w = tap_pattern(c.tap_dims());
                const std::vector<double> g = gradient_pattern(c.output_dims());

                for (const ExecutionMode mode : K_FORCED_MODES)
                {
                    SCOPED_TRACE(std::string(c.id) + " " + forced_mode_name(mode));
                    set_mode(mode);

                    const Tensor in = tensor_from<double>(*engine, c.input_dims(), x);
                    const Tensor weights = tensor_from<double>(*engine, c.tap_dims(), w);
                    const Tensor dy = tensor_from<double>(*engine, c.output_dims(), g);

                    const double through_out =
                        dot_of(forward_of<double>(*engine, c, in, weights), g);
                    const double through_in =
                        dot_of(x, input_gradient_of<double>(*engine, c, dy, weights));
                    const double through_weights =
                        dot_of(w, weight_gradient_of<double>(*engine, c, dy, in));

                    EXPECT_EQ(through_out, through_in);
                    EXPECT_EQ(through_out, through_weights);
                }
            }
        }

        /** @brief Tests that the two modes write equal forward bits on non-integer data. */
        TYPED_TEST(Medium_SpatialWindowDotTyped, ReferenceAndSYCLAgreeToTheBitOnTheForward)
        {
            using T = TypeParam;
            Engine& engine = *this->engine;

            for (const WindowCase& c : table_cases())
            {
                SCOPED_TRACE(c.id);

                const Tensor in = tensor_from<T>(engine, c.input_dims(),
                                                 fractional(input_pattern(c.input_dims())));
                const Tensor weights =
                    tensor_from<T>(engine, c.tap_dims(), fractional(tap_pattern(c.tap_dims())));

                engine.set_mode(ExecutionMode::ForceReference);
                const std::vector<T> reference_out = forward_of<T>(engine, c, in, weights);

                engine.set_mode(ExecutionMode::ForceSYCL);
                const std::vector<T> device_out = forward_of<T>(engine, c, in, weights);

                expect_equal_bits(reference_out, device_out, "out");
            }
        }

        /** @brief Tests that the two modes write equal input-gradient bits on non-integer data. */
        TYPED_TEST(Medium_SpatialWindowDotTyped, ReferenceAndSYCLAgreeToTheBitOnTheInputGradient)
        {
            using T = TypeParam;
            Engine& engine = *this->engine;

            for (const WindowCase& c : table_cases())
            {
                SCOPED_TRACE(c.id);

                const Tensor weights =
                    tensor_from<T>(engine, c.tap_dims(), fractional(tap_pattern(c.tap_dims())));
                const Tensor dy = tensor_from<T>(engine, c.output_dims(),
                                                 fractional(gradient_pattern(c.output_dims())));

                engine.set_mode(ExecutionMode::ForceReference);
                const std::vector<T> reference_dx = input_gradient_of<T>(engine, c, dy, weights);

                engine.set_mode(ExecutionMode::ForceSYCL);
                const std::vector<T> device_dx = input_gradient_of<T>(engine, c, dy, weights);

                expect_equal_bits(reference_dx, device_dx, "dx");
            }
        }

        /** @brief Tests that a weight gradient of 4096 non-integer terms is within its bound. */
        TYPED_TEST(Medium_SpatialWindowDotTyped, WeightGradientIsWithinItsBound)
        {
            using T = TypeParam;
            Engine& engine = *this->engine;

            const WindowCase c{"bound", geometry_of(3, 3, 1, 1, 1, 1, 1, 1), 4, 32, 32, 3, 32, 32};

            const std::vector<double> x =
                widened(elements_from<T>(fractional(input_pattern(c.input_dims()))));
            const std::vector<double> g =
                widened(elements_from<T>(fractional(gradient_pattern(c.output_dims()))));

            std::vector<TruthSum> truths(element_count(c.tap_dims()));
            for_each_live_tap(c, [&](std::size_t output, std::size_t input, std::size_t tap)
            {
                truths[tap].add_product(g[output], x[input]);
            });

            for (const ExecutionMode mode : K_FORCED_MODES)
            {
                SCOPED_TRACE(forced_mode_name(mode));
                engine.set_mode(mode);

                const Tensor dy = tensor_from<T>(engine, c.output_dims(), g);
                const Tensor in = tensor_from<T>(engine, c.input_dims(), x);
                const std::vector<double> got = widened(weight_gradient_of<T>(engine, c, dy, in));

                expect_within_bounds<T>(mode, got, truths, c.windows());
            }
        }

        /** @brief Returns a CPU profile with 8 compute units and a work-group limit of 64. */
        static Device::Profile window_dot_cpu_profile()
        {
            Device::Profile profile;
            profile.is_cpu = true;
            profile.compute_units = 8;
            profile.max_work_group_size = 64;
            profile.sub_group_size = 1;
            return profile;
        }

        /** @brief Tests that 2 segments of 256 windows take the cooperative launch shape. */
        TEST(Small_SpatialWindowDotPlan, TwoSegmentsOfManyWindowsAreFoldedByAWorkGroup)
        {
            const Reduce::AxisPlan plan = Reduce::plan_axis(window_dot_cpu_profile(), 2, 256);

            EXPECT_TRUE(plan.cooperative());
        }

        /** @brief Tests that 144 segments, more than the 8 resident, take one work-item each. */
        TEST(Small_SpatialWindowDotPlan, ManySegmentsAreFoldedOneWorkItemEach)
        {
            const Reduce::AxisPlan plan = Reduce::plan_axis(window_dot_cpu_profile(), 144, 72);

            EXPECT_FALSE(plan.cooperative());
        }

        /** @brief Tests that an integer weight gradient is exact with 2 segments and with 144. */
        TEST_F(Medium_SpatialWindowDot, WeightGradientIsExactInBothLaunchShapes)
        {
            const WindowCase cases[] = {
                {"two_segments", geometry_of(1, 2, 1, 1, 1, 1, 0, 1), 4, 8, 8, 1, 8,
                 Spatial::output_extent(8, 1, 0, 1, 2, 1)},
                {"many_segments", geometry_of(3, 3, 1, 1, 1, 1, 1, 1), 2, 6, 6, 16, 6, 6},
            };
            ASSERT_EQ(cases[0].windows(), std::size_t{256});
            ASSERT_EQ(element_count(cases[1].tap_dims()), std::size_t{144});

            const Device::Profile& profile =
                Device::profile_for(TestEnvironment::queue());

            for (const WindowCase& c : cases)
            {
                const Reduce::AxisPlan plan =
                    Reduce::plan_axis(profile, element_count(c.tap_dims()), c.windows());
                RecordProperty(std::string("cooperative_") + c.id,
                               plan.cooperative() ? "true" : "false");

                const std::vector<double> g = gradient_pattern(c.output_dims());
                const std::vector<double> x = input_pattern(c.input_dims());
                const std::vector<double> expected = weight_gradient_by_formula(c, g, x);

                for (const ExecutionMode mode : K_FORCED_MODES)
                {
                    SCOPED_TRACE(std::string(c.id) + " " + forced_mode_name(mode));
                    set_mode(mode);

                    const Tensor dy = tensor_from<float>(*engine, c.output_dims(), g);
                    const Tensor in = tensor_from<float>(*engine, c.input_dims(), x);

                    expect_exact(weight_gradient_of<float>(*engine, c, dy, in), expected,
                                 "dweights");
                }
            }
        }

        /** @brief Returns the window whose first two output columns lie wholly in the padding. */
        static WindowCase window_dot_padding_case()
        {
            // Output column j reads input column j - 2, so columns 0 and 1 read only padding.
            return WindowCase{"padding", geometry_of(1, 1, 1, 1, 1, 1, 0, 2), 1, 1, 3, 1, 1, 4};
        }

        /** @brief Tests that a window lying wholly in the padding gives a forward of zero. */
        TEST_F(Medium_SpatialWindowDot, AWindowOfOnlyPaddingGivesAZeroForward)
        {
            const WindowCase c = window_dot_padding_case();

            for (const ExecutionMode mode : K_FORCED_MODES)
            {
                SCOPED_TRACE(forced_mode_name(mode));
                set_mode(mode);

                const Tensor in = tensor_from<float>(*engine, c.input_dims(), {3, 5, 7});
                const Tensor weights = tensor_from<float>(*engine, c.tap_dims(), {2});

                expect_exact(forward_of<float>(*engine, c, in, weights), {0, 0, 6, 10}, "out");
            }
        }

        /** @brief Tests that a window lying wholly in the padding adds nothing to the input gradient. */
        TEST_F(Medium_SpatialWindowDot, AWindowOfOnlyPaddingAddsNothingToTheInputGradient)
        {
            const WindowCase c = window_dot_padding_case();

            for (const ExecutionMode mode : K_FORCED_MODES)
            {
                SCOPED_TRACE(forced_mode_name(mode));
                set_mode(mode);

                const Tensor weights = tensor_from<float>(*engine, c.tap_dims(), {2});
                const Tensor dy = tensor_from<float>(*engine, c.output_dims(), {1, 2, 3, 4});

                expect_exact(input_gradient_of<float>(*engine, c, dy, weights), {6, 8, 0}, "dx");
            }
        }

        /** @brief Tests that a window lying wholly in the padding adds nothing to the weight gradient. */
        TEST_F(Medium_SpatialWindowDot, AWindowOfOnlyPaddingAddsNothingToTheWeightGradient)
        {
            const WindowCase c = window_dot_padding_case();

            for (const ExecutionMode mode : K_FORCED_MODES)
            {
                SCOPED_TRACE(forced_mode_name(mode));
                set_mode(mode);

                const Tensor in = tensor_from<float>(*engine, c.input_dims(), {3, 5, 7});
                const Tensor dy = tensor_from<float>(*engine, c.output_dims(), {1, 2, 3, 4});

                expect_exact(weight_gradient_of<float>(*engine, c, dy, in), {29}, "dweights");
            }
        }

        /** @brief Tests that an offset, row-strided input view gives the dense copy's result. */
        TEST_F(Medium_SpatialWindowDot, AStridedInputIsReadThroughItsStrides)
        {
            const WindowCase c = table_cases()[1];
            const std::vector<int64_t> larger_dims = {c.batch + 1, c.in_h, c.in_w + 2, c.channels};
            const std::vector<double> larger_values = input_pattern(larger_dims);
            const std::vector<double> dense_values = interior_copy(c, larger_values);

            for (const ExecutionMode mode : K_FORCED_MODES)
            {
                SCOPED_TRACE(forced_mode_name(mode));
                set_mode(mode);

                const Tensor larger = tensor_from<float>(*engine, larger_dims, larger_values);
                const Tensor view = larger.slice(0, 1, c.batch + 1).slice(2, 1, c.in_w + 1);
                const Tensor dense = tensor_from<float>(*engine, c.input_dims(), dense_values);
                const Tensor weights =
                    tensor_from<float>(*engine, c.tap_dims(), tap_pattern(c.tap_dims()));
                const Tensor dy =
                    tensor_from<float>(*engine, c.output_dims(), gradient_pattern(c.output_dims()));

                ASSERT_FALSE(view.is_contiguous());

                expect_equal_bits(forward_of<float>(*engine, c, dense, weights),
                                  forward_of<float>(*engine, c, view, weights), "out");
                expect_equal_bits(weight_gradient_of<float>(*engine, c, dy, dense),
                                  weight_gradient_of<float>(*engine, c, dy, view), "dweights");
            }
        }

        /** @brief Tests that a batch of zero images writes a weight gradient of zeros. */
        TEST_F(Small_SpatialWindowDot, AZeroBatchWritesAZeroWeightGradient)
        {
            const Tensor dy = engine->create_tensor({0, 4, 4, 2}, Core::DataType::Float32);
            const Tensor in = engine->create_tensor({0, 4, 4, 2}, Core::DataType::Float32);
            Tensor dweights = engine->create_tensor({3, 3, 2}, Core::DataType::Float32);
            fill_tensor(dweights, std::vector<float>(18, static_cast<float>(K_SENTINEL)));

            SushiBLAS::SpatialOps(recorder()).window_dot_weights_backward(
                dy, in, dweights, geometry_of(3, 3, 1, 1, 1, 1, 1, 1));
            ASSERT_TRUE(recording->run_succeeds());

            expect_tensor_equal(dweights, std::vector<float>(18, 0.0f));
        }

        /** @brief Tests that each malformed call throws naming the operation and the operand. */
        TEST_P(Small_SpatialWindowDotMalformed, TheRefusalNamesTheOperationAndTheOperand)
        {
            const MalformedRow& row = GetParam();
            SushiBLAS::SpatialOps spatial(recorder());

            expect_refusal([&] { row.record(spatial, *engine); }, "spatial.window_dot",
                           row.operand);
        }

        INSTANTIATE_TEST_SUITE_P(Operands, Small_SpatialWindowDotMalformed,
                                 ::testing::ValuesIn(malformed_rows()), malformed_row_name);

        /** @brief Tests that ForceReference refuses a device-only input. */
        TEST_F(Medium_SpatialWindowDotDeviceOnly, ForceReferenceRefusesADeviceOnlyInput)
        {
            SushiBLAS::SpatialOps spatial(recorder());

            set_mode(ExecutionMode::ForceReference);
            EXPECT_THROW(spatial.window_dot(*input, weights, output, geometry),
                         std::invalid_argument);
        }

        /** @brief Tests that Auto records a device-only input and the window sums of ones come out. */
        TEST_F(Medium_SpatialWindowDotDeviceOnly, AutoRecordsADeviceOnlyInput)
        {
            SushiBLAS::SpatialOps spatial(recorder());

            set_mode(ExecutionMode::Auto);
            spatial.window_dot(*input, weights, output, geometry);
            recording->run().value();

            const WindowCase c = table_cases()[0];
            expect_exact(SushiBLAS::IO().to_vector<float>(output),
                         forward_by_formula(c, std::vector<double>(element_count(c.input_dims()), 1.0),
                                            tap_pattern(c.tap_dims())),
                         "out");
        }
    }
}
