/**************************************************************************/
/* test_spatial_upsample_bilinear.cpp                                     */
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
 * @file test_spatial_upsample_bilinear.cpp
 * @brief Tests SpatialOps::upsample_bilinear and upsample_bilinear_backward.
 * @author Mustafa Garip
 */

#include <array>
#include <cmath>
#include <string>
#include <vector>
#include <limits>
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <algorithm>
#include <utility>
#include <functional>
#include <stdexcept>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.hpp>
#include <SushiBLAS/engine_fixture/sushiblas_test.hpp>
#include <SushiBLAS/tensor_expectations/tensor_readback.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            /** @brief Returns the bit pattern of a float, so -0 and NaN compare as what they are. */
            uint32_t bilinear_float_bits(float value) noexcept
            {
                uint32_t bits = 0;
                std::memcpy(&bits, &value, sizeof(bits));
                return bits;
            }

            /**
             * @brief Holds the rounding of one blend of a unit-gauge convex combination, in units of u.
             *
             * One blend is `p * (1 - t) + q * t`: two products and one sum round once each, and the
             * two weights carry one rounding each, so it errs by at most 5 u times the largest
             * input. The value is two blends over one blend: 10 u.
             */
            constexpr double K_VALUE_UNITS = 10.0;

            /**
             * @brief Holds the rounding of a gradient's summation, in units of u, before the term count.
             *
             * Each term of a gradient is a product of a gradient, two axis weights and their
             * product: 4 roundings. The term count is added by the caller.
             */
            constexpr double K_TERM_UNITS = 4.0;

            /** @brief Returns the unit roundoff of T, half its machine epsilon. */
            template<typename T>
            constexpr double unit_roundoff() noexcept
            {
                return static_cast<double>(std::numeric_limits<T>::epsilon()) / 2.0;
            }

            /** @brief Returns floor of s, as a double, for the oracle. */
            int64_t floor_of(double s) noexcept
            {
                return static_cast<int64_t>(std::floor(s));
            }

            /** @brief Returns v clamped to `[0, last]`. */
            int64_t clamped(int64_t v, int64_t last) noexcept
            {
                return std::max<int64_t>(0, std::min<int64_t>(v, last));
            }

            /**
             * @brief Interpolates an NHWC image by the formula of the design, in double.
             * @return The `[N, H * sh, W * sw, C]` result in memory order.
             */
            std::vector<double> host_bilinear(const std::vector<float>& in, int64_t N, int64_t H,
                                              int64_t W, int64_t C, int64_t sh, int64_t sw)
            {
                const int64_t out_h = H * sh;
                const int64_t out_w = W * sw;
                std::vector<double> out(static_cast<std::size_t>(N * out_h * out_w * C));

                for (int64_t n = 0; n < N; ++n)
                {
                    for (int64_t h_o = 0; h_o < out_h; ++h_o)
                    {
                        const double s_h = (static_cast<double>(h_o) + 0.5) / sh - 0.5;
                        const int64_t h0 = floor_of(s_h);
                        const double t_h = s_h - static_cast<double>(h0);

                        for (int64_t w_o = 0; w_o < out_w; ++w_o)
                        {
                            const double s_w = (static_cast<double>(w_o) + 0.5) / sw - 0.5;
                            const int64_t w0 = floor_of(s_w);
                            const double t_w = s_w - static_cast<double>(w0);

                            for (int64_t c = 0; c < C; ++c)
                            {
                                auto pixel = [&](int64_t r, int64_t q)
                                {
                                    const int64_t row = clamped(r, H - 1);
                                    const int64_t col = clamped(q, W - 1);
                                    return static_cast<double>(
                                        in[static_cast<std::size_t>(((n * H + row) * W + col) * C
                                                                    + c)]);
                                };

                                const double top = pixel(h0, w0) * (1 - t_w)
                                                   + pixel(h0, w0 + 1) * t_w;
                                const double bottom = pixel(h0 + 1, w0) * (1 - t_w)
                                                      + pixel(h0 + 1, w0 + 1) * t_w;

                                out[static_cast<std::size_t>(((n * out_h + h_o) * out_w + w_o) * C
                                                             + c)] = top * (1 - t_h)
                                                                     + bottom * t_h;
                            }
                        }
                    }
                }

                return out;
            }

            /** @brief Returns the largest magnitude of values. */
            double largest_magnitude(const std::vector<float>& values)
            {
                double largest = 0.0;
                for (float v : values)
                    largest = std::max(largest, std::fabs(static_cast<double>(v)));
                return largest;
            }

            /** @brief Returns an integer-valued pattern in `[-8, 8]` that repeats every 17 elements. */
            std::vector<float> small_integers(std::size_t count, std::size_t step)
            {
                std::vector<float> values(count);
                for (std::size_t i = 0; i < count; ++i)
                    values[i] = static_cast<float>(static_cast<int>((i * step) % 17) - 8);
                return values;
            }
        } // namespace

        class Medium_SpatialUpsampleBilinearTest : public SushiBLASTest
        {
            protected:
                /** @brief Records the forward on in and runs it in the given mode. */
                Tensor forward(const Tensor& in, int64_t sh, int64_t sw, ExecutionMode mode)
                {
                    auto out = engine->create_tensor(
                        {in.shape[0], in.shape[1] * sh, in.shape[2] * sw, in.shape[3]},
                        in.dtype);

                    set_mode(mode);
                    SushiBLAS::SpatialOps(recorder()).upsample_bilinear(in, out, sh, sw);
                    recording->run().value();
                    return out;
                }

                /** @brief Records the gradient of a forward of the given scales and runs it. */
                Tensor backward(const Tensor& dy, int64_t sh, int64_t sw, ExecutionMode mode)
                {
                    auto dx = engine->create_tensor(
                        {dy.shape[0], dy.shape[1] / sh, dy.shape[2] / sw, dy.shape[3]},
                        dy.dtype);

                    set_mode(mode);
                    SushiBLAS::SpatialOps(recorder()).upsample_bilinear_backward(dy, dx, sh, sw);
                    recording->run().value();
                    return dx;
                }

                /** @brief Asserts that call throws std::invalid_argument naming the operation. */
                void expect_refusal(const std::string& operation, const std::function<void()>& call)
                {
                    try
                    {
                        call();
                        ADD_FAILURE() << operation << " accepted an operand it must refuse";
                    }
                    catch (const std::invalid_argument& e)
                    {
                        EXPECT_NE(std::string(e.what()).find(operation), std::string::npos)
                            << "the message does not name the operation: " << e.what();
                    }
                }

                /**
                 * @brief Asserts `sum(dy * f(x)) == sum(b(dy) * x)` within the derived bound.
                 *
                 * The bound is the roundoff of the two sides: each value of f errs by
                 * K_VALUE_UNITS u times the largest input, each gradient element sums at most
                 * `4 * sh * sw` terms of K_TERM_UNITS roundings each, and each side is a sum of
                 * as many terms as it has elements, held in double.
                 */
                template<typename T>
                void expect_adjoint(ExecutionMode mode, int64_t sh, int64_t sw)
                {
                    const int64_t N = 2, H = 3, W = 4, C = 2;
                    const std::vector<float> x = small_integers(N * H * W * C, 5);
                    const std::vector<float> g = small_integers(N * H * sh * W * sw * C, 7);

                    auto in = create_initialized_tensor<T>({N, H, W, C},
                                                           std::vector<T>(x.begin(), x.end()));
                    auto dy = create_initialized_tensor<T>({N, H * sh, W * sw, C},
                                                           std::vector<T>(g.begin(), g.end()));

                    const std::vector<T> y = read_tensor<T>(forward(in, sh, sw, mode));
                    const std::vector<T> dx = read_tensor<T>(backward(dy, sh, sw, mode));

                    double lhs = 0.0;
                    for (std::size_t i = 0; i < y.size(); ++i)
                        lhs += static_cast<double>(g[i]) * static_cast<double>(y[i]);

                    double rhs = 0.0;
                    for (std::size_t i = 0; i < dx.size(); ++i)
                        rhs += static_cast<double>(dx[i]) * static_cast<double>(x[i]);

                    double gradient_mass = 0.0;
                    for (float v : g)
                        gradient_mass += std::fabs(static_cast<double>(v));

                    const double gauge = gradient_mass * largest_magnitude(x);
                    const double terms = static_cast<double>(y.size() + dx.size());
                    const double units = K_VALUE_UNITS
                                         + K_TERM_UNITS * static_cast<double>(4 * sh * sw) + terms;

                    EXPECT_LE(std::fabs(lhs - rhs), units * unit_roundoff<T>() * gauge)
                        << "lhs " << lhs << ", rhs " << rhs;
                }
        };

        TEST_F(Medium_SpatialUpsampleBilinearTest, ATwoByTwoImageAtScaleTwoGivesTheSixteenValues)
        {
            // At scale 2 the rows (0, 8) and (16, 40) give (0, 2, 6, 8) and (16, 22, 34, 40) as columns.
            auto in = create_initialized_tensor<float>({1, 2, 2, 1}, {0.0f, 8.0f, 16.0f, 40.0f});

            const std::vector<float> expected = {0, 2, 6, 8,
                                                 4, 7, 13, 16,
                                                 12, 17, 27, 32,
                                                 16, 22, 34, 40};

            const std::vector<float> got =
                read_tensor<float>(forward(in, 2, 2, ExecutionMode::ForceReference));
            ASSERT_EQ(got.size(), expected.size());
            EXPECT_EQ(got, expected);
        }

        TEST_F(Medium_SpatialUpsampleBilinearTest, ABackwardOfOnesGivesEachPixelTheScaleProduct)
        {
            // The weights of a pixel sum to 2 per axis over the output, so it receives 2 * 2 = 4.
            auto dy = create_initialized_tensor<float>({1, 4, 4, 1}, std::vector<float>(16, 1.0f));

            const std::vector<float> got =
                read_tensor<float>(backward(dy, 2, 2, ExecutionMode::ForceReference));

            EXPECT_EQ(got, std::vector<float>(4, 4.0f));
        }

        TEST_F(Medium_SpatialUpsampleBilinearTest, AConstantImageStaysConstant)
        {
            // Scales 2 and 4 give weights that are multiples of 1/8, so 3.5 is reproduced exactly.
            auto in = create_initialized_tensor<float>({2, 3, 3, 2},
                                                       std::vector<float>(36, 3.5f));

            const std::vector<float> got =
                read_tensor<float>(forward(in, 2, 4, ExecutionMode::ForceReference));

            EXPECT_EQ(got, std::vector<float>(2 * 6 * 12 * 2, 3.5f));
        }

        TEST_F(Medium_SpatialUpsampleBilinearTest, AScaleOfOneOnBothAxesCopiesTheBitPatterns)
        {
            const std::vector<float> values = {1.5f, -0.0f, std::numeric_limits<float>::infinity(),
                                               -2.0f, 0.0f, 7.25f};
            auto in = create_initialized_tensor<float>({1, 2, 3, 1}, values);

            const std::vector<float> got =
                read_tensor<float>(forward(in, 1, 1, ExecutionMode::ForceReference));

            ASSERT_EQ(got.size(), values.size());
            EXPECT_EQ(bilinear_float_bits(got[0]), bilinear_float_bits(values[0]));
            EXPECT_EQ(bilinear_float_bits(got[1]), bilinear_float_bits(values[1]));
            EXPECT_EQ(bilinear_float_bits(got[2]), bilinear_float_bits(values[2]));
            EXPECT_EQ(bilinear_float_bits(got[3]), bilinear_float_bits(values[3]));
            EXPECT_EQ(bilinear_float_bits(got[4]), bilinear_float_bits(values[4]));
            EXPECT_EQ(bilinear_float_bits(got[5]), bilinear_float_bits(values[5]));
        }

        class Medium_SpatialUpsampleBilinearScalesTest
            : public Medium_SpatialUpsampleBilinearTest,
              public ::testing::WithParamInterface<std::pair<int64_t, int64_t>>
        {
        };

        TEST_P(Medium_SpatialUpsampleBilinearScalesTest, EachScalePairMatchesTheDoubleFormula)
        {
            const int64_t sh = GetParam().first;
            const int64_t sw = GetParam().second;
            const int64_t N = 2, H = 3, W = 2, C = 2;

            const std::vector<float> x = small_integers(N * H * W * C, 3);
            auto in = create_initialized_tensor<float>({N, H, W, C}, x);

            const std::vector<float> got =
                read_tensor<float>(forward(in, sh, sw, ExecutionMode::ForceReference));
            const std::vector<double> expected = host_bilinear(x, N, H, W, C, sh, sw);

            const double bound =
                K_VALUE_UNITS * unit_roundoff<float>() * largest_magnitude(x);
            ASSERT_EQ(got.size(), expected.size());
            EXPECT_TRUE(std::equal(got.begin(), got.end(), expected.begin(),
                                   [bound](float a, double b)
                                   {
                                       return std::fabs(static_cast<double>(a) - b) <= bound;
                                   }));
        }

        INSTANTIATE_TEST_SUITE_P(
            Scales, Medium_SpatialUpsampleBilinearScalesTest,
            ::testing::Values(std::make_pair<int64_t, int64_t>(1, 2),
                              std::make_pair<int64_t, int64_t>(3, 2),
                              std::make_pair<int64_t, int64_t>(2, 3),
                              std::make_pair<int64_t, int64_t>(4, 1),
                              std::make_pair<int64_t, int64_t>(3, 4)));

        TEST_F(Medium_SpatialUpsampleBilinearTest, TheForwardAndItsGradientAreAdjointInDouble)
        {
            expect_adjoint<double>(ExecutionMode::ForceReference, 3, 2);
        }

        TEST_F(Medium_SpatialUpsampleBilinearTest, TheForwardAndItsGradientAreAdjointInFloat)
        {
            expect_adjoint<float>(ExecutionMode::ForceReference, 2, 4);
        }

        TEST_F(Medium_SpatialUpsampleBilinearTest, TheForwardAndItsGradientAreAdjointOnTheDevice)
        {
            expect_adjoint<float>(ExecutionMode::ForceSYCL, 3, 2);
        }

        TEST_F(Medium_SpatialUpsampleBilinearTest, TheDeviceForwardMatchesTheReferenceForward)
        {
            const int64_t N = 2, H = 5, W = 3, C = 3, sh = 3, sw = 2;
            const std::vector<float> x = small_integers(N * H * W * C, 11);
            auto in = create_initialized_tensor<float>({N, H, W, C}, x);

            const std::vector<float> reference =
                read_tensor<float>(forward(in, sh, sw, ExecutionMode::ForceReference));
            const std::vector<float> device =
                read_tensor<float>(forward(in, sh, sw, ExecutionMode::ForceSYCL));

            const double bound =
                K_VALUE_UNITS * unit_roundoff<float>() * largest_magnitude(x);
            ASSERT_EQ(device.size(), reference.size());
            EXPECT_TRUE(std::equal(device.begin(), device.end(), reference.begin(),
                                   [bound](float a, float b)
                                   {
                                       return std::fabs(static_cast<double>(a) - b) <= bound;
                                   }));
        }

        TEST_F(Medium_SpatialUpsampleBilinearTest, TheDeviceGradientMatchesTheReferenceGradient)
        {
            const int64_t N = 2, H = 5, W = 3, C = 3, sh = 3, sw = 2;
            const std::vector<float> g = small_integers(N * H * sh * W * sw * C, 13);
            auto dy = create_initialized_tensor<float>({N, H * sh, W * sw, C}, g);

            const std::vector<float> reference =
                read_tensor<float>(backward(dy, sh, sw, ExecutionMode::ForceReference));
            const std::vector<float> device =
                read_tensor<float>(backward(dy, sh, sw, ExecutionMode::ForceSYCL));

            // A pixel sums at most 4 * sh * sw terms of at most 2 * sh * 2 * sw total weight each.
            const double terms = static_cast<double>(4 * sh * sw);
            const double bound = (K_TERM_UNITS + terms) * unit_roundoff<float>() * terms
                                 * largest_magnitude(g);
            ASSERT_EQ(device.size(), reference.size());
            EXPECT_TRUE(std::equal(device.begin(), device.end(), reference.begin(),
                                   [bound](float a, float b)
                                   {
                                       return std::fabs(static_cast<double>(a) - b) <= bound;
                                   }));
        }

        TEST_F(Medium_SpatialUpsampleBilinearTest, RefusesAScaleOfZero)
        {
            auto in = create_initialized_tensor<float>({1, 2, 2, 1}, std::vector<float>(4, 1.0f));
            auto out = engine->create_tensor({1, 4, 4, 1}, Core::DataType::Float32);

            expect_refusal("spatial.upsample_bilinear", [&]
            {
                SushiBLAS::SpatialOps(recorder()).upsample_bilinear(in, out, 0, 2);
            });
        }

        TEST_F(Medium_SpatialUpsampleBilinearTest, RefusesANegativeScaleOnTheGradient)
        {
            auto dy = create_initialized_tensor<float>({1, 4, 4, 1}, std::vector<float>(16, 1.0f));
            auto dx = engine->create_tensor({1, 2, 2, 1}, Core::DataType::Float32);

            expect_refusal("spatial.upsample_bilinear_backward", [&]
            {
                SushiBLAS::SpatialOps(recorder()).upsample_bilinear_backward(dy, dx, 2, -2);
            });
        }

        TEST_F(Medium_SpatialUpsampleBilinearTest, RefusesAnOutputOfTheWrongExtent)
        {
            auto in = create_initialized_tensor<float>({1, 2, 2, 1}, std::vector<float>(4, 1.0f));
            auto out = engine->create_tensor({1, 4, 5, 1}, Core::DataType::Float32);

            expect_refusal("spatial.upsample_bilinear", [&]
            {
                SushiBLAS::SpatialOps(recorder()).upsample_bilinear(in, out, 2, 2);
            });
        }

        TEST_F(Medium_SpatialUpsampleBilinearTest, RefusesAnOutputOfAnotherDtype)
        {
            auto in = create_initialized_tensor<float>({1, 2, 2, 1}, std::vector<float>(4, 1.0f));
            auto out = engine->create_tensor({1, 4, 4, 1}, Core::DataType::Float64);

            expect_refusal("spatial.upsample_bilinear", [&]
            {
                SushiBLAS::SpatialOps(recorder()).upsample_bilinear(in, out, 2, 2);
            });
        }

        TEST_F(Medium_SpatialUpsampleBilinearTest, RefusesAnInputOfTheWrongRank)
        {
            auto in = engine->create_tensor({2, 2, 1}, Core::DataType::Float32);
            auto out = engine->create_tensor({1, 4, 4, 1}, Core::DataType::Float32);

            expect_refusal("spatial.upsample_bilinear", [&]
            {
                SushiBLAS::SpatialOps(recorder()).upsample_bilinear(in, out, 2, 2);
            });
        }

        TEST_F(Medium_SpatialUpsampleBilinearTest, RefusesAnOutputThatOverlapsItsInput)
        {
            auto parent = engine->create_tensor({1, 4, 4, 1}, Core::DataType::Float32);

            const std::vector<int64_t> dims = {1, 2, 2, 1};
            std::array<int64_t, Core::K_MAX_TENSOR_RANK> strides{};
            strides[0] = 16;
            strides[1] = 4;
            strides[2] = 1;
            strides[3] = 1;
            const Tensor in(parent.storage, dims, strides, Core::DataType::Float32,
                            parent.storage_offset, parent.layout);

            expect_refusal("spatial.upsample_bilinear", [&]
            {
                SushiBLAS::SpatialOps(recorder()).upsample_bilinear(in, parent, 2, 2);
            });
        }

        TEST_F(Medium_SpatialUpsampleBilinearTest, RefusesAGradientThatOverlapsItsSource)
        {
            auto parent = engine->create_tensor({1, 4, 4, 1}, Core::DataType::Float32);

            const std::vector<int64_t> dims = {1, 2, 2, 1};
            std::array<int64_t, Core::K_MAX_TENSOR_RANK> strides{};
            strides[0] = 16;
            strides[1] = 4;
            strides[2] = 1;
            strides[3] = 1;
            Tensor dx(parent.storage, dims, strides, Core::DataType::Float32,
                      parent.storage_offset, parent.layout);

            expect_refusal("spatial.upsample_bilinear_backward", [&]
            {
                SushiBLAS::SpatialOps(recorder()).upsample_bilinear_backward(parent, dx, 2, 2);
            });
        }
    }
}
