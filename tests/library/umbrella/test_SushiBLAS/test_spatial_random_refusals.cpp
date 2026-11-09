/**************************************************************************/
/* test_spatial_random_refusals.cpp                                       */
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
 * @file test_spatial_random_refusals.cpp
 * @brief Tests, on a host engine, which spatial geometries and random fills are refused at record time.
 * @author Mustafa Garip
 */

#include <cmath>
#include <string>
#include <vector>
#include <cstdint>
#include <limits>
#include <functional>
#include <stdexcept>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.hpp>
#include <SushiBLAS/engine_fixture/host_engine_test.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            /// @brief A Poisson mean above K_MAX_POISSON_LAMBDA.
            constexpr double K_LAMBDA_ABOVE_CAP = 2.0e7;

            /// @brief A negative Poisson mean.
            constexpr double K_LAMBDA_NEGATIVE = -1.0;

            /// @brief The bound FLOAT32 discrete_uniform accepts in magnitude, 2^24.
            constexpr int32_t K_FLOAT32_EXACT_LIMIT = 16777216;

            /// @brief A float bound whose doubled magnitude overflows FLOAT32.
            constexpr double K_LARGE_FLOAT_BOUND = 3.0e38;

            /// @brief A double bound whose doubled magnitude overflows FLOAT64.
            constexpr double K_LARGE_DOUBLE_BOUND = 1.0e308;

            /// @brief What recording one call does.
            enum class Ending : std::uint8_t
            {
                Accepted, /**< The call records without throwing. */
                Refused   /**< The call throws std::invalid_argument. */
            };

            /** @brief Names one record call and how it must end. */
            struct RefusalRow
            {
                /** @brief The row name, which gtest takes as the suffix of the test. */
                std::string name;

                /** @brief Records the call over tensors of the engine. */
                std::function<void(TaskRecorder&, Engine&)> record;

                /** @brief How the call must end. */
                Ending expected;
            };

            /** @brief Returns how record ends; any other exception leaves the test. */
            Ending ending_of(const std::function<void()>& record)
            {
                try
                {
                    record();
                }
                catch (const std::invalid_argument&)
                {
                    return Ending::Refused;
                }

                return Ending::Accepted;
            }

            /** @brief Returns a FLOAT32 NHWC tensor of 1 x 2 x 2 x 1. */
            Tensor small_image(Engine& engine)
            {
                return engine.create_tensor({1, 2, 2, 1}, Core::DataType::Float32);
            }

            /** @brief Returns a 2 x 2 window geometry. */
            Spatial::ConvGeometry window_geometry()
            {
                Spatial::ConvGeometry g;
                g.kernel_h = 2;
                g.kernel_w = 2;
                return g;
            }

            /** @brief Returns the rows of the spatial geometry and index tensor checks. */
            std::vector<RefusalRow> spatial_rows()
            {
                return {
                    {"GeometryWithZeroStride",
                     [](TaskRecorder& r, Engine& e)
                     {
                         Tensor in = small_image(e);
                         Tensor out = e.create_tensor({1, 1, 1, 1}, Core::DataType::Float32);
                         Spatial::ConvGeometry g = window_geometry();
                         g.stride_h = 0;
                         SpatialOps(r).pool_avg(in, out, g);
                     },
                     Ending::Refused},
                    {"PoolingWindowMadeOnlyOfPadding",
                     [](TaskRecorder& r, Engine& e)
                     {
                         // Window 0 covers rows -2 and -1 of a 2-row input: both taps are padding.
                         Tensor in = small_image(e);
                         Tensor out = e.create_tensor({1, 1, 1, 1}, Core::DataType::Float32);
                         Tensor indices = e.create_tensor({1, 1, 1, 1}, Core::DataType::Int64);
                         Spatial::ConvGeometry g = window_geometry();
                         g.pad_top = 2;
                         SpatialOps(r).pool_max(in, out, indices, g);
                     },
                     Ending::Refused},
                    {"PoolMaxStridedIndexTensor",
                     [](TaskRecorder& r, Engine& e)
                     {
                         Tensor in = small_image(e);
                         Tensor out = small_image(e);
                         Tensor dense = e.create_tensor({1, 2, 2, 1}, Core::DataType::Int64);
                         Tensor indices = dense.transpose(1, 2);
                         SpatialOps(r).pool_max(in, out, indices, Spatial::ConvGeometry{});
                     },
                     Ending::Refused},
                    {"PoolMaxBackwardTransposedOutputGradient",
                     [](TaskRecorder& r, Engine& e)
                     {
                         Tensor dy = small_image(e).transpose(1, 2);
                         Tensor indices = e.create_tensor({1, 2, 2, 1}, Core::DataType::Int64);
                         Tensor dx = small_image(e);
                         SpatialOps(r).pool_max_backward(dy, indices, dx, Spatial::ConvGeometry{});
                     },
                     Ending::Refused}};
            }

            /** @brief Returns a row that fills a tensor of four elements of dtype through call. */
            RefusalRow random_row(const char* name, Core::DataType dtype,
                                  std::function<void(RandomOps&, Tensor&)> call, Ending expected)
            {
                return {name,
                        [dtype, call](TaskRecorder& r, Engine& e)
                        {
                            Tensor t = e.create_tensor({4}, dtype);
                            RandomOps ops(r, e.rng());
                            call(ops, t);
                        },
                        expected};
            }

            /** @brief Returns the rows of the random fill checks. */
            std::vector<RefusalRow> random_rows()
            {
                const double inf = std::numeric_limits<double>::infinity();
                const double nan = std::numeric_limits<double>::quiet_NaN();
                const Core::DataType f32 = Core::DataType::Float32;
                const Core::DataType f64 = Core::DataType::Float64;
                const Core::DataType c32 = Core::DataType::Complex32;

                return {
                    random_row("PoissonLambdaAboveTheCap", f32,
                               [](RandomOps& o, Tensor& t) { o.poisson(t, K_LAMBDA_ABOVE_CAP); },
                               Ending::Refused),
                    random_row("PoissonLambdaAtTheCap", f32,
                               [](RandomOps& o, Tensor& t) { o.poisson(t, K_MAX_POISSON_LAMBDA); },
                               Ending::Accepted),
                    random_row("PoissonLambdaJustAboveTheCap", f32,
                               [=](RandomOps& o, Tensor& t)
                               { o.poisson(t, std::nextafter(K_MAX_POISSON_LAMBDA, inf)); },
                               Ending::Refused),
                    random_row("PoissonNegativeLambda", f32,
                               [](RandomOps& o, Tensor& t) { o.poisson(t, K_LAMBDA_NEGATIVE); },
                               Ending::Refused),
                    random_row("UniformBoundsOutOfOrder", f32,
                               [](RandomOps& o, Tensor& t) { o.uniform(t, 1.0, 0.0); },
                               Ending::Refused),
                    random_row("UniformInfiniteBound", f64,
                               [=](RandomOps& o, Tensor& t) { o.uniform(t, 0.0, inf); },
                               Ending::Refused),
                    random_row("UniformNaNBound", f64,
                               [=](RandomOps& o, Tensor& t) { o.uniform(t, nan, 1.0); },
                               Ending::Refused),
                    random_row("UniformWidthOverflowsFloat32", f32,
                               [](RandomOps& o, Tensor& t)
                               { o.uniform(t, -K_LARGE_FLOAT_BOUND, K_LARGE_FLOAT_BOUND); },
                               Ending::Refused),
                    random_row("UniformWidthOverflowsFloat64", f64,
                               [](RandomOps& o, Tensor& t)
                               { o.uniform(t, -K_LARGE_DOUBLE_BOUND, K_LARGE_DOUBLE_BOUND); },
                               Ending::Refused),
                    random_row("DiscreteUniformAtTheFloat32Limit", f32,
                               [](RandomOps& o, Tensor& t)
                               { o.discrete_uniform(t, -K_FLOAT32_EXACT_LIMIT, K_FLOAT32_EXACT_LIMIT); },
                               Ending::Accepted),
                    random_row("DiscreteUniformUpperBoundPastTheLimit", f32,
                               [](RandomOps& o, Tensor& t)
                               { o.discrete_uniform(t, 0, K_FLOAT32_EXACT_LIMIT + 1); },
                               Ending::Refused),
                    random_row("DiscreteUniformLowerBoundPastTheLimit", f32,
                               [](RandomOps& o, Tensor& t)
                               { o.discrete_uniform(t, -K_FLOAT32_EXACT_LIMIT - 1, 0); },
                               Ending::Refused),
                    random_row("PoissonIntoAComplexTensor", c32,
                               [](RandomOps& o, Tensor& t) { o.poisson(t, 1.0); },
                               Ending::Refused),
                    random_row("DiscreteUniformIntoAComplexTensor", c32,
                               [](RandomOps& o, Tensor& t) { o.discrete_uniform(t, 0, 10); },
                               Ending::Refused),
                    random_row("BernoulliIntoAComplexTensor", c32,
                               [](RandomOps& o, Tensor& t) { o.bernoulli(t, 0.5); },
                               Ending::Refused)};
            }

            /** @brief Names a row by its name. */
            std::string refusal_row_name(const ::testing::TestParamInfo<RefusalRow>& info)
            {
                return info.param.name;
            }
        } // namespace

        /// @brief Records one call per row on a host engine; each is decided on shape, stride or value.
        class Small_RecordRefusal : public HostEngineTest,
                                    public ::testing::WithParamInterface<RefusalRow> {};

        /// @brief Tests that a call ends as its row names and a refused call records no task.
        TEST_P(Small_RecordRefusal, EndsAsTheRowNames)
        {
            const RefusalRow& row = GetParam();

            EXPECT_EQ(ending_of([&] { row.record(recorder(), *engine); }), row.expected);
            if (row.expected == Ending::Refused)
                EXPECT_EQ(recording->recorded_count(), 0u);
        }

        INSTANTIATE_TEST_SUITE_P(Spatial, Small_RecordRefusal,
                                 ::testing::ValuesIn(spatial_rows()), refusal_row_name);

        INSTANTIATE_TEST_SUITE_P(Random, Small_RecordRefusal,
                                 ::testing::ValuesIn(random_rows()), refusal_row_name);
    } // namespace Test
} // namespace SushiBLAS
