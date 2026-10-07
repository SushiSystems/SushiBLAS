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
 * @file test_spatial_window_dot.cpp
 * @brief Tests the per-channel weighted window sum and its two gradients in both forced modes.
 * @author Mustafa Garip
 */

#include <cmath>
#include <string>
#include <vector>
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <SushiBLAS/SushiBLAS.h>
#include <SushiBLAS/support/accumulator.hpp>
#include <SushiBLAS/support/reduce_axis.hpp>
#include <SushiBLAS/support/device_policy.hpp>
#include "../../common/test_utils.hpp"
#include "../../common/blas_truth.hpp"

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            /// @brief The forced modes every case runs in.
            constexpr ExecutionMode K_FORCED_MODES[] = {ExecutionMode::ForceSycl,
                                                        ExecutionMode::ForceReference};

            /// @brief The value an output holds before its operation runs.
            constexpr double K_SENTINEL = 7.0;

            /// @brief The largest magnitude up to which HALF holds every integer.
            constexpr double K_HALF_EXACT_INTEGER_LIMIT = 2048.0;

            /** @brief Returns the name a failure message gives a forced mode. */
            const char* forced_mode_name(ExecutionMode mode) noexcept
            {
                return mode == ExecutionMode::ForceReference ? "FORCE_REFERENCE" : "FORCE_SYCL";
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
                    get_sushiblas_dtype<T>());

                if (t.num_elements > 0)
                    SushiBLAS::IO(engine).from_vector(t, elements_from<T>(values));
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
                SushiBLAS::SpatialOps(engine).window_dot(in, weights, out, c.geometry);
                return SushiBLAS::IO(engine).to_vector<T>(out);
            }

            /** @brief Returns the input gradient the library writes for dy and weights. */
            template<typename T>
            std::vector<T> input_gradient_of(Engine& engine, const WindowCase& c, const Tensor& dy,
                                             const Tensor& weights)
            {
                Tensor dx = sentinel_tensor<T>(engine, c.input_dims());
                SushiBLAS::SpatialOps(engine).window_dot_backward(dy, weights, dx, c.geometry);
                return SushiBLAS::IO(engine).to_vector<T>(dx);
            }

            /** @brief Returns the weight gradient the library writes for dy and in. */
            template<typename T>
            std::vector<T> weight_gradient_of(Engine& engine, const WindowCase& c, const Tensor& dy,
                                              const Tensor& in)
            {
                Tensor dweights = sentinel_tensor<T>(engine, c.tap_dims());
                SushiBLAS::SpatialOps(engine).window_dot_weights_backward(dy, in, dweights,
                                                                          c.geometry);
                return SushiBLAS::IO(engine).to_vector<T>(dweights);
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

            /** @brief Expects two results to hold the same bits, element for element. */
            template<typename T>
            void expect_equal_bits(const std::vector<T>& reference, const std::vector<T>& device,
                                   const char* quantity)
            {
                ASSERT_EQ(reference.size(), device.size()) << quantity;
                for (std::size_t i = 0; i < reference.size(); ++i)
                    EXPECT_EQ(std::memcmp(&reference[i], &device[i], sizeof(T)), 0)
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

                if (mode == ExecutionMode::ForceSycl)
                    return rounding_bound(16, unit_roundoff<Kernels::Accumulator<T>>(),
                                          truth.magnitude) + store;

                const std::size_t roundings = std::is_same_v<T, double> ? 5 : terms + 2;
                return rounding_bound(roundings, unit_roundoff<double>(), truth.magnitude) + store;
            }

            /** @brief Returns true when the test device can compute in T. */
            template<typename T>
            bool device_computes()
            {
                const sycl::device device = TestEnvironment::get_context().get_queue().get_device();

                if constexpr (std::is_same_v<T, sycl::half>)
                    return device.has(sycl::aspect::fp16);
                else if constexpr (std::is_same_v<T, double>)
                    return device.has(sycl::aspect::fp64);
                else
                    return true;
            }

            /**
             * @brief Expects record to throw std::invalid_argument naming operation and operand.
             * @tparam Record Callable `() -> void` that records one malformed call.
             */
            template<typename Record>
            void expect_refusal(Record&& record, const char* operation, const char* operand)
            {
                try
                {
                    record();
                    ADD_FAILURE() << operation << " accepted a malformed '" << operand << "'";
                }
                catch (const std::invalid_argument& error)
                {
                    const std::string text = error.what();
                    EXPECT_NE(text.find(operation), std::string::npos) << text;
                    EXPECT_NE(text.find("'" + std::string(operand) + "'"), std::string::npos)
                        << text;
                }
            }
        } // namespace

        /**
         * @brief Runs the window-sum operations in one element type, skipped where the device
         *        lacks it.
         */
        template<typename T>
        class Unit_SpatialWindowDotTyped : public SushiBLASTest
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
        SB_TYPED_TEST_SUITE(Unit_SpatialWindowDotTyped, WindowDotTypes);

        /**
         * @brief Runs the window-sum cases that fix one element type.
         */
        class Unit_SpatialWindowDot : public SushiBLASTest
        {
        };

        /** @brief Tests that a strided, dilated, padded 1 by 2 window gives hand-worked values. */
        TYPED_TEST(Unit_SpatialWindowDotTyped, TheOneRowWindowGivesItsHandWorkedValues)
        {
            using T = TypeParam;
            Engine& engine = *this->engine;

            const WindowCase c{"hand", geometry_of(1, 2, 1, 2, 1, 2, 0, 1), 1, 1, 5, 1,
                               1, Spatial::output_extent(5, 1, 1, 2, 2, 2)};
            ASSERT_EQ(c.out_w, 3);

            for (const ExecutionMode mode : K_FORCED_MODES)
            {
                SCOPED_TRACE(forced_mode_name(mode));
                engine.set_mode(mode);

                const Tensor in = tensor_from<T>(engine, c.input_dims(), {1, 2, 3, 4, 5});
                const Tensor weights = tensor_from<T>(engine, c.tap_dims(), {10, 1});
                const Tensor dy = tensor_from<T>(engine, c.output_dims(), {1, 2, 3});

                expect_exact(forward_of<T>(engine, c, in, weights), {2, 24, 40}, "out");
                expect_exact(input_gradient_of<T>(engine, c, dy, weights), {0, 21, 0, 32, 0}, "dx");
                expect_exact(weight_gradient_of<T>(engine, c, dy, in), {16, 10}, "dweights");
            }
        }

        /** @brief Tests that the forward equals a host loop written from its formula, G1 to G5. */
        TYPED_TEST(Unit_SpatialWindowDotTyped, ForwardMatchesTheFormula)
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
        TYPED_TEST(Unit_SpatialWindowDotTyped, InputGradientMatchesAHostScatter)
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
        TYPED_TEST(Unit_SpatialWindowDotTyped, WeightGradientMatchesTheFormula)
        {
            using T = TypeParam;
            Engine& engine = *this->engine;

            for (const WindowCase& c : table_cases())
            {
                const std::vector<double> g = gradient_pattern(c.output_dims());
                const std::vector<double> x = input_pattern(c.input_dims());
                const std::vector<double> expected = weight_gradient_by_formula(c, g, x);

                for (const double truth : expected)
                    ASSERT_LT(std::fabs(truth), K_HALF_EXACT_INTEGER_LIMIT) << c.id;

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
        TEST_F(Unit_SpatialWindowDot, TheThreeKernelsAreAdjoint)
        {
            if (!device_computes<double>())
                GTEST_SKIP() << "the device does not report the fp64 aspect";

            for (const WindowCase& c : table_cases())
            {
                const std::vector<double> x = input_pattern(c.input_dims());
                const std::vector<double> w = tap_pattern(c.tap_dims());
                const std::vector<double> g = gradient_pattern(c.output_dims());

                for (const ExecutionMode mode : K_FORCED_MODES)
                {
                    SCOPED_TRACE(std::string(c.id) + " " + forced_mode_name(mode));
                    engine->set_mode(mode);

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

        /** @brief Tests that the two gathers write equal bits in both modes on non-integer data. */
        TYPED_TEST(Unit_SpatialWindowDotTyped, ReferenceAndSyclAgreeToTheBitOnTheGathers)
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
                const Tensor dy = tensor_from<T>(engine, c.output_dims(),
                                                 fractional(gradient_pattern(c.output_dims())));

                engine.set_mode(ExecutionMode::ForceReference);
                const std::vector<T> reference_out = forward_of<T>(engine, c, in, weights);
                const std::vector<T> reference_dx = input_gradient_of<T>(engine, c, dy, weights);

                engine.set_mode(ExecutionMode::ForceSycl);
                const std::vector<T> device_out = forward_of<T>(engine, c, in, weights);
                const std::vector<T> device_dx = input_gradient_of<T>(engine, c, dy, weights);

                expect_equal_bits(reference_out, device_out, "out");
                expect_equal_bits(reference_dx, device_dx, "dx");
            }
        }

        /** @brief Tests that a weight gradient of 4096 non-integer terms is within its bound. */
        TYPED_TEST(Unit_SpatialWindowDotTyped, WeightGradientIsWithinItsBound)
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

                ASSERT_EQ(got.size(), truths.size());
                for (std::size_t tap = 0; tap < truths.size(); ++tap)
                {
                    const TruthValue truth = truths[tap].value();
                    EXPECT_LE(std::fabs(got[tap] - truth.value),
                              weight_gradient_bound<T>(mode, c.windows(), truth))
                        << "dweights element " << tap << ": got " << got[tap] << ", truth "
                        << truth.value << ", magnitude " << truth.magnitude;
                }
            }
        }

        /** @brief Tests that an integer weight gradient is exact with 2 segments and with 144. */
        TEST_F(Unit_SpatialWindowDot, WeightGradientIsExactInBothLaunchShapes)
        {
            const WindowCase cases[] = {
                {"two_segments", geometry_of(1, 2, 1, 1, 1, 1, 0, 1), 4, 8, 8, 1, 8,
                 Spatial::output_extent(8, 1, 0, 1, 2, 1)},
                {"many_segments", geometry_of(3, 3, 1, 1, 1, 1, 1, 1), 2, 6, 6, 16, 6, 6},
            };
            ASSERT_EQ(cases[0].windows(), std::size_t{256});
            ASSERT_EQ(element_count(cases[1].tap_dims()), std::size_t{144});

            const Device::Profile& profile =
                Device::profile_for(TestEnvironment::get_context().get_queue());

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
                    engine->set_mode(mode);

                    const Tensor dy = tensor_from<float>(*engine, c.output_dims(), g);
                    const Tensor in = tensor_from<float>(*engine, c.input_dims(), x);

                    expect_exact(weight_gradient_of<float>(*engine, c, dy, in), expected,
                                 "dweights");
                }
            }
        }

        /** @brief Tests that a window lying wholly in the padding is accepted and adds nothing. */
        TEST_F(Unit_SpatialWindowDot, AWindowOfOnlyPaddingGivesZero)
        {
            // Output column j reads input column j - 2, so columns 0 and 1 read only padding.
            const WindowCase c{"padding", geometry_of(1, 1, 1, 1, 1, 1, 0, 2), 1, 1, 3, 1, 1, 4};

            for (const ExecutionMode mode : K_FORCED_MODES)
            {
                SCOPED_TRACE(forced_mode_name(mode));
                engine->set_mode(mode);

                const Tensor in = tensor_from<float>(*engine, c.input_dims(), {3, 5, 7});
                const Tensor weights = tensor_from<float>(*engine, c.tap_dims(), {2});
                const Tensor dy = tensor_from<float>(*engine, c.output_dims(), {1, 2, 3, 4});

                expect_exact(forward_of<float>(*engine, c, in, weights), {0, 0, 6, 10}, "out");
                expect_exact(input_gradient_of<float>(*engine, c, dy, weights), {6, 8, 0}, "dx");
                expect_exact(weight_gradient_of<float>(*engine, c, dy, in), {29}, "dweights");
            }
        }

        /** @brief Tests that an offset, row-strided input view gives the dense copy's result. */
        TEST_F(Unit_SpatialWindowDot, AStridedInputIsReadThroughItsStrides)
        {
            const WindowCase c = table_cases()[1];
            const std::vector<int64_t> larger_dims = {c.batch + 1, c.in_h, c.in_w + 2, c.channels};
            const std::vector<double> larger_values = input_pattern(larger_dims);

            // The view skips batch 0 and one column on each side of every row.
            std::vector<double> dense_values;
            dense_values.reserve(element_count(c.input_dims()));
            for (int64_t n = 1; n <= c.batch; ++n)
            for (int64_t h = 0; h < c.in_h; ++h)
            for (int64_t w = 1; w <= c.in_w; ++w)
            for (int64_t ch = 0; ch < c.channels; ++ch)
                dense_values.push_back(larger_values[static_cast<std::size_t>(
                    ((n * c.in_h + h) * (c.in_w + 2) + w) * c.channels + ch)]);

            for (const ExecutionMode mode : K_FORCED_MODES)
            {
                SCOPED_TRACE(forced_mode_name(mode));
                engine->set_mode(mode);

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
        TEST_F(Unit_SpatialWindowDot, AZeroBatchWritesAZeroWeightGradient)
        {
            const WindowCase c{"zero_batch", geometry_of(3, 3, 1, 1, 1, 1, 1, 1), 0, 4, 4, 2, 4, 4};

            for (const ExecutionMode mode : K_FORCED_MODES)
            {
                SCOPED_TRACE(forced_mode_name(mode));
                engine->set_mode(mode);

                const Tensor dy = tensor_from<float>(*engine, c.output_dims(), {});
                const Tensor in = tensor_from<float>(*engine, c.input_dims(), {});

                expect_exact(weight_gradient_of<float>(*engine, c, dy, in),
                             std::vector<double>(element_count(c.tap_dims()), 0.0), "dweights");
            }
        }

        /** @brief Tests that each malformed call throws naming the operation and the operand. */
        TEST_F(Unit_SpatialWindowDot, MalformedOperandsNameTheOperand)
        {
            static constexpr const char* NAME = "spatial.window_dot";
            const int64_t N = 1, H = 4, W = 4, C = 2;
            const Spatial::ConvGeometry g = geometry_of(3, 3, 1, 1, 1, 1, 1, 1);

            Tensor in = engine->create_tensor({N, H, W, C}, Core::DataType::Float32);
            Tensor weights = engine->create_tensor({3, 3, C}, Core::DataType::Float32);
            Tensor out = engine->create_tensor({N, H, W, C}, Core::DataType::Float32);
            SushiBLAS::SpatialOps spatial(*engine);

            Tensor flat_weights = engine->create_tensor({3, 3}, Core::DataType::Float32);
            expect_refusal([&] { spatial.window_dot(in, flat_weights, out, g); }, NAME, "weights");

            Tensor wide_weights = engine->create_tensor({3, 3, C + 1}, Core::DataType::Float32);
            expect_refusal([&] { spatial.window_dot(in, wide_weights, out, g); }, NAME, "weights");

            Tensor double_out = engine->create_tensor({N, H, W, C}, Core::DataType::Float64);
            expect_refusal([&] { spatial.window_dot(in, weights, double_out, g); }, NAME, "out");

            Tensor batched_out = engine->create_tensor({N + 1, H, W, C}, Core::DataType::Float32);
            expect_refusal([&] { spatial.window_dot(in, weights, batched_out, g); }, NAME, "out");

            Tensor aliased_out = in;
            expect_refusal([&] { spatial.window_dot(in, weights, aliased_out, g); }, NAME, "out");

            Spatial::ConvGeometry unstrided = g;
            unstrided.stride_h = 0;
            expect_refusal([&] { spatial.window_dot(in, weights, out, unstrided); }, NAME,
                           "geometry");
        }

        /** @brief Tests that FORCE_REFERENCE refuses a device-only input and AUTO records it. */
        TEST_F(Unit_SpatialWindowDot, ADeviceOnlyOperandIsRefusedUnderForceReference)
        {
            const WindowCase c = table_cases()[0];
            const Spatial::ConvGeometry g = c.geometry;

            std::optional<Tensor> in;
            try
            {
                in = engine->create_tensor({c.batch, c.in_h, c.in_w, c.channels},
                                           Core::DataType::Float32,
                                           SushiRuntime::API::Residency::Device);
            }
            catch (const std::exception& error)
            {
                GTEST_SKIP() << "the device gives no device-only USM: " << error.what();
            }

            const sycl::context context = TestEnvironment::get_context().get_queue().get_context();
            if (sycl::get_pointer_type(in->data(), context) != sycl::usm::alloc::device)
                GTEST_SKIP() << "the allocation is not device-only USM";

            fill_tensor(*in, std::vector<float>(element_count(c.input_dims()), 1.0f));

            const Tensor weights =
                tensor_from<float>(*engine, c.tap_dims(), tap_pattern(c.tap_dims()));
            Tensor out = sentinel_tensor<float>(*engine, c.output_dims());
            SushiBLAS::SpatialOps spatial(*engine);

            engine->set_mode(ExecutionMode::ForceReference);
            EXPECT_THROW(spatial.window_dot(*in, weights, out, g), std::invalid_argument);

            engine->set_mode(ExecutionMode::Auto);
            EXPECT_NO_THROW(spatial.window_dot(*in, weights, out, g));
            engine->execute();
        }
    }
}
