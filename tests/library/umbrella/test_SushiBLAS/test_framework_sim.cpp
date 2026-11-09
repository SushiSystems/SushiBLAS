/**************************************************************************/
/* test_framework_sim.cpp                                                 */
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
 * @file test_framework_sim.cpp
 * @brief Tests a ten-layer forward pass and 24 independent models through the engine.
 * @author Mustafa Garip
 */

#include <gtest/gtest.h>
#include <vector>
#include <SushiBLAS/SushiBLAS.hpp>
#include <SushiBLAS/engine_fixture/sushiblas_test.hpp>
#include <SushiBLAS/tensor_expectations/tensor_comparison.hpp>
#include <SushiBLAS/tensor_expectations/tensor_fill.hpp>

namespace sb = SushiBLAS;
using namespace SushiBLAS::Test;

class Medium_FrameworkSimTest : public SushiBLASTest
{
    protected:
        /** @brief Returns a features x features weight matrix whose even columns hold +weight and odd columns -weight. */
        static std::vector<float> alternating_weights(int features, float weight)
        {
            std::vector<float> values(static_cast<std::size_t>(features) * features);
            for (int i = 0; i < features * features; ++i)
                values[i] = ((i % features) % 2 == 0) ? weight : -weight;

            return values;
        }

        /** @brief Creates one weight matrix per layer, filled with the same values. */
        std::vector<sb::Tensor> layer_weights(int layers, int features, const std::vector<float>& values)
        {
            std::vector<sb::Tensor> weights;
            for (int i = 0; i < layers; ++i)
            {
                weights.push_back(engine->create_tensor({features, features}));
                fill_tensor(weights.back(), values);
            }

            return weights;
        }

        /** @brief Creates the output of each layer, after the input, which is activations[0]. */
        std::vector<sb::Tensor> layer_activations(const sb::Tensor& input, int layers, int batch, int features)
        {
            std::vector<sb::Tensor> activations{input};
            for (int i = 0; i < layers; ++i)
                activations.push_back(engine->create_tensor({batch, features}));

            return activations;
        }

        /** @brief Records H_{i+1} = H_i * W_i, with a ReLU after every layer but the last. */
        void record_layers(std::vector<sb::Tensor>& activations,
                           const std::vector<sb::Tensor>& weights)
        {
            const std::size_t layers = weights.size();
            for (std::size_t i = 0; i < layers; ++i)
            {
                SushiBLAS::BLASOps(recorder()).gemm(activations[i], weights[i], activations[i + 1]);

                if (i + 1 < layers)
                    SushiBLAS::NonLinearOps(recorder()).relu(activations[i + 1]);
            }
        }

        /** @brief Returns the magnitude of every element of the last layer of an all-ones input. */
        static double last_layer_magnitude(int layers, int features, float weight)
        {
            // Layer 0 sums all features; later layers see only the even half that survived ReLU.
            double magnitude = static_cast<double>(features) * static_cast<double>(weight);
            for (int i = 1; i < layers; ++i)
                magnitude *= static_cast<double>(features / 2) * static_cast<double>(weight);

            return magnitude;
        }

        /** @brief Returns the last layer: +magnitude in even columns, -magnitude in odd ones, as it has no ReLU. */
        static std::vector<float> last_layer(int batch, int features, double magnitude)
        {
            std::vector<float> expected(static_cast<std::size_t>(batch) * features);
            for (int i = 0; i < batch * features; ++i)
                expected[i] = static_cast<float>(((i % features) % 2 == 0) ? magnitude : -magnitude);

            return expected;
        }
};

/**
 * @brief Tests a chain of ten GEMMs, each followed by ReLU except the last, against a host-computed result.
 * @see tests/library/umbrella/README.md
 */
TEST_F(Medium_FrameworkSimTest, DeepForwardPass)
{
    const int batch_size = 512;
    const int features = 1024;
    const int num_layers = 10;
    const float weight = 0.01f;

    auto X = engine->create_tensor({batch_size, features});
    fill_tensor(X, std::vector<float>(batch_size * features, 1.0f));
    const std::vector<sb::Tensor> weights =
        layer_weights(num_layers, features, alternating_weights(features, weight));
    std::vector<sb::Tensor> activations =
        layer_activations(X, num_layers, batch_size, features);

    record_layers(activations, weights);
    recording->run().value();

    const double magnitude = last_layer_magnitude(num_layers, features, weight);
    expect_tensor(activations.back(), last_layer(batch_size, features, magnitude),
                  absolute_tolerance<float>(2048, magnitude));
}

/** @brief Holds 24 independent GEMM and ReLU pairs: model i has X = i + 1 and W = +2 for even i, -2 for odd i. */
class Medium_ParallelModels : public Medium_FrameworkSimTest
{
    protected:
        static constexpr int K_MODELS = 24;
        static constexpr int K_M = 256;
        static constexpr int K_N = 512;
        static constexpr int K_K = 256;

        /** @brief Returns the value every element of the input of model i holds. */
        static float input_of(int i)
        {
            return static_cast<float>(i + 1);
        }

        /** @brief Returns the value every element of the weights of model i holds; ReLU zeroes the odd models. */
        static float weight_of(int i)
        {
            return (i % 2 == 0) ? 2.0f : -2.0f;
        }

        /** @brief Records every model into the graph and returns the output tensors. */
        std::vector<sb::Tensor> record_models()
        {
            std::vector<sb::Tensor> inputs;
            std::vector<sb::Tensor> weights;
            std::vector<sb::Tensor> outputs;
            for (int i = 0; i < K_MODELS; ++i)
            {
                inputs.push_back(engine->create_tensor({K_M, K_K}));
                fill_tensor(inputs.back(), std::vector<float>(K_M * K_K, input_of(i)));
                weights.push_back(engine->create_tensor({K_K, K_N}));
                fill_tensor(weights.back(), std::vector<float>(K_K * K_N, weight_of(i)));
                outputs.push_back(engine->create_tensor({K_M, K_N}));
            }

            for (int i = 0; i < K_MODELS; ++i)
            {
                SushiBLAS::BLASOps(recorder()).gemm(inputs[i], weights[i], outputs[i]);
                SushiBLAS::NonLinearOps(recorder()).relu(outputs[i]);
            }

            return outputs;
        }

        /** @brief Returns relu(k * input_of(i) * weight_of(i)) for every element of model i, exact in float. */
        static std::vector<float> expected_output(int i)
        {
            const float pre_activation = static_cast<float>(K_K) * input_of(i) * weight_of(i);

            return std::vector<float>(K_M * K_N, pre_activation > 0.0f ? pre_activation : 0.0f);
        }

        /** @brief Expects every output to hold the result of its own model. */
        static void expect_every_model(const std::vector<sb::Tensor>& outputs)
        {
            for (int i = 0; i < K_MODELS; ++i)
                expect_tensor_equal(outputs[i], expected_output(i));
        }
};

/**
 * @brief Tests 24 independent GEMM and ReLU pairs submitted to one graph against their exact results.
 * @see tests/library/umbrella/README.md
 */
TEST_F(Medium_ParallelModels, EveryModelOfOneGraphHoldsItsOwnResult)
{
    const std::vector<sb::Tensor> outputs = record_models();

    recording->run().value();

    expect_every_model(outputs);
}
