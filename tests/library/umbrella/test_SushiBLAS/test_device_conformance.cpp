/**************************************************************************/
/* test_device_conformance.cpp                                            */
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
 * @file test_device_conformance.cpp
 * @brief Tests that sum, dot, gemm, softmax and layer norm give on every device of the runtime
 *        the CPU's bytes, or differ by a stated bound that the test names.
 * @author Mustafa Garip
 */

#include <cmath>
#include <string>
#include <vector>
#include <random>
#include <cstring>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <optional>
#include <algorithm>
#include <initializer_list>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.hpp>
#include "support/tolerance.hpp"
#include <SushiBLAS/engine_fixture/sushiblas_test.hpp>
#include <SushiBLAS/test_runner/test_environment.hpp>
#include <SushiBLAS/tensor_expectations/tensor_copy.hpp>
#include <SushiBLAS/tensor_expectations/tensor_readback.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            /// @brief Device slots the suite covers; a runtime that offers more fails the coverage test.
            constexpr std::size_t K_DEVICE_SLOTS = 4;

            /// @brief Seed of every input; each operand adds its own salt.
            constexpr std::uint32_t K_INPUT_SEED = 20261102u;

            /// @brief Terms of a sum and of a dot; above the CPU two-stage threshold of 16384.
            constexpr int64_t K_VECTOR_LENGTH = 65536;

            /// @brief Rows of the GEMM product.
            constexpr int64_t K_GEMM_M = 96;

            /// @brief Columns of the GEMM product.
            constexpr int64_t K_GEMM_N = 80;

            /// @brief Terms of each GEMM element.
            constexpr int64_t K_GEMM_K = 64;

            /// @brief Elements of the softmax and layer norm rows.
            constexpr int64_t K_ROW_LENGTH = 1024;

            /// @brief The layer norm epsilon added to the variance before the square root.
            constexpr float K_LAYER_NORM_EPSILON = 1.0e-5f;

            /// @brief SYCL 2020 maximum error of exp in float, in ulp.
            constexpr std::size_t K_EXP_ULPS = 4;

            /// @brief SYCL 2020 maximum error of sqrt and of division in float, in ulp, rounded up.
            constexpr std::size_t K_SQRT_AND_DIVIDE_ULPS = 3;

            /// @brief Both the CPU's value and the device's value carry the roundings of an operation.
            constexpr std::size_t K_RESULTS_COMPARED = 2;

            /// @brief The operations of the kernel set.
            enum class Operation
            {
                Sum,       /**< The sum of a vector. */
                Dot,       /**< The dot product of two vectors. */
                Gemm,      /**< The product of two matrices. */
                Softmax,   /**< exp(x - max) over its sum, composed from reductions and elementwise operations. */
                LayerNorm  /**< (x - mean) / sqrt(var + eps) * gamma + beta, composed the same way. */
            };

            /// @brief Names the values a row feeds its operation.
            enum class Data
            {
                Integer, /**< Integers in [-3, 3]: every partial sum is exact, so bytes must match. */
                Seeded   /**< Seeded reals: rounding differs by device, so a bound holds. */
            };

            /// @brief One operation on one kind of data.
            struct ConformanceRow
            {
                Operation operation; /**< The operation the row runs. */
                Data data;           /**< The values the row feeds it. */
            };

            /// @brief One device slot and one row.
            struct ConformanceParam
            {
                std::size_t slot;   /**< The index of the runtime device. */
                ConformanceRow row; /**< The operation and its data. */
            };

            /// @brief Holds the host operands of a row.
            struct Inputs
            {
                std::vector<float> first;  /**< x, or A. */
                std::vector<float> second; /**< y, or B. */
                std::vector<float> gamma;  /**< The layer norm scale. */
                std::vector<float> beta;   /**< The layer norm shift. */
            };

            /// @brief Returns the element count of an extent.
            [[nodiscard]] std::size_t count_of(int64_t extent)
            {
                return static_cast<std::size_t>(extent);
            }

            /// @brief Returns the name of an operation in a test name and a report.
            [[nodiscard]] const char* operation_name(Operation operation)
            {
                switch (operation)
                {
                    case Operation::Sum: return "Sum";
                    case Operation::Dot: return "Dot";
                    case Operation::Gemm: return "Gemm";
                    case Operation::Softmax: return "Softmax";
                    case Operation::LayerNorm: return "LayerNorm";
                }
                return "Unknown";
            }

            /// @brief Returns why bytes may differ for a row, or why they may not.
            [[nodiscard]] const char* difference_reason(const ConformanceRow& row)
            {
                if (row.data == Data::Integer)
                {
                    return "none expected: every partial sum is an integer below 2^24, exact in any order";
                }
                switch (row.operation)
                {
                    case Operation::Sum:
                    case Operation::Dot:
                        return "reduction order: each device folds the terms in its own tree";
                    case Operation::Gemm:
                        return "reduction order and fused multiply-add: a device may round a*b+c once";
                    case Operation::Softmax:
                        return "exp and divide ulp, and the reduction order of the denominator";
                    case Operation::LayerNorm:
                        return "reduction order of mean and variance, and sqrt and divide ulp";
                }
                return "unknown";
            }

            /// @brief Returns the values of one operand, from the seed and the operand's salt.
            [[nodiscard]] std::vector<float> operand_values(std::size_t count, std::uint32_t salt, Data data,
                                                            float centre, float spread)
            {
                std::mt19937 generator(K_INPUT_SEED + salt);
                std::vector<float> values(count);
                if (data == Data::Integer)
                {
                    std::uniform_int_distribution<int> distribution(-3, 3);
                    for (float& value : values)
                        value = static_cast<float>(distribution(generator));
                    return values;
                }
                std::uniform_real_distribution<float> distribution(centre - spread, centre + spread);
                for (float& value : values)
                    value = distribution(generator);
                return values;
            }

            /// @brief Returns the host operands of a row.
            [[nodiscard]] Inputs make_inputs(const ConformanceRow& row)
            {
                Inputs inputs;
                switch (row.operation)
                {
                    case Operation::Sum:
                        inputs.first = operand_values(count_of(K_VECTOR_LENGTH), 1, row.data, 0.0f, 1.0f);
                        break;
                    case Operation::Dot:
                        inputs.first = operand_values(count_of(K_VECTOR_LENGTH), 2, row.data, 0.0f, 1.0f);
                        inputs.second = operand_values(count_of(K_VECTOR_LENGTH), 3, row.data, 0.0f, 1.0f);
                        break;
                    case Operation::Gemm:
                        inputs.first = operand_values(count_of(K_GEMM_M * K_GEMM_K), 4, row.data, 0.0f, 1.0f);
                        inputs.second = operand_values(count_of(K_GEMM_K * K_GEMM_N), 5, row.data, 0.0f, 1.0f);
                        break;
                    case Operation::Softmax:
                        inputs.first = operand_values(count_of(K_ROW_LENGTH), 6, row.data, 0.0f, 2.0f);
                        break;
                    case Operation::LayerNorm:
                        inputs.first = operand_values(count_of(K_ROW_LENGTH), 7, row.data, 0.25f, 1.0f);
                        inputs.gamma = operand_values(count_of(K_ROW_LENGTH), 8, row.data, 1.0f, 0.5f);
                        inputs.beta = operand_values(count_of(K_ROW_LENGTH), 9, row.data, 0.0f, 0.5f);
                        break;
                }
                return inputs;
            }

            /// @brief Returns the roundings one result of an operation carries, the count its bound scales with.
            [[nodiscard]] std::size_t rounding_count(Operation operation)
            {
                switch (operation)
                {
                    case Operation::Sum:
                    case Operation::Dot:
                        return count_of(K_VECTOR_LENGTH);
                    case Operation::Gemm:
                        return count_of(K_GEMM_K);
                    case Operation::Softmax:
                        // The shift, exp, the denominator's sum over the row, one divide.
                        return 1 + K_EXP_ULPS + count_of(K_ROW_LENGTH) + K_SQRT_AND_DIVIDE_ULPS;
                    case Operation::LayerNorm:
                        // The mean and variance sums, sqrt, divide, then the subtraction, the scale and the shift.
                        return 2 * count_of(K_ROW_LENGTH) + 2 * K_SQRT_AND_DIVIDE_ULPS + 3;
                }
                return 0;
            }

            /**
             * @brief Returns, per result element, the magnitude the rounding bound scales with.
             *
             * A sum, dot or GEMM element scales with the sum of its absolute terms. A softmax
             * element is positive and scales with its own value. A layer norm element scales with
             * the largest output, because an error in the mean or the variance reaches every
             * element at that size.
             */
            [[nodiscard]] std::vector<double> bound_gauge(const ConformanceRow& row, const Inputs& inputs,
                                                          const std::vector<float>& cpu_values)
            {
                std::vector<double> gauge(cpu_values.size(), 0.0);
                switch (row.operation)
                {
                    case Operation::Sum:
                        for (const float value : inputs.first)
                            gauge[0] += std::fabs(static_cast<double>(value));
                        break;
                    case Operation::Dot:
                        for (std::size_t i = 0; i < inputs.first.size(); ++i)
                            gauge[0] += std::fabs(static_cast<double>(inputs.first[i]) * inputs.second[i]);
                        break;
                    case Operation::Gemm:
                        for (int64_t i = 0; i < K_GEMM_M; ++i)
                            for (int64_t j = 0; j < K_GEMM_N; ++j)
                                for (int64_t k = 0; k < K_GEMM_K; ++k)
                                    gauge[count_of(i * K_GEMM_N + j)] += std::fabs(
                                        static_cast<double>(inputs.first[count_of(i * K_GEMM_K + k)]) *
                                        inputs.second[count_of(k * K_GEMM_N + j)]);
                        break;
                    case Operation::Softmax:
                        for (std::size_t i = 0; i < gauge.size(); ++i)
                            gauge[i] = std::fabs(static_cast<double>(cpu_values[i]));
                        break;
                    case Operation::LayerNorm:
                    {
                        double largest = 0.0;
                        for (const float value : cpu_values)
                            largest = std::max(largest, std::fabs(static_cast<double>(value)));
                        std::fill(gauge.begin(), gauge.end(), largest);
                        break;
                    }
                }
                return gauge;
            }

            /// @brief Returns the slot, name and platform of a device, for a report.
            [[nodiscard]] std::string describe_device(std::size_t slot)
            {
                const sycl::device device = TestEnvironment::queue(slot).get_device();
                return "slot " + std::to_string(slot) + " " + device.get_info<sycl::info::device::name>() + " (" +
                       device.get_platform().get_info<sycl::info::platform::name>() + ")";
            }

            /// @brief Returns the slot of the first CPU device of the runtime, if it offers one.
            [[nodiscard]] std::optional<std::size_t> find_cpu_slot()
            {
                for (std::size_t slot = 0; slot < TestEnvironment::device_count(); ++slot)
                {
                    if (TestEnvironment::queue(slot).get_device().is_cpu())
                        return slot;
                }
                return std::nullopt;
            }

            /// @brief Names a case from its slot, operation and data, for the test report.
            [[nodiscard]] std::string case_name(const ::testing::TestParamInfo<ConformanceParam>& info)
            {
                return "Slot" + std::to_string(static_cast<std::size_t>(info.param.slot)) + "_" + operation_name(info.param.row.operation) +
                       (info.param.row.data == Data::Integer ? "_Integer" : "_Seeded");
            }

            /// @brief Returns every device slot crossed with every row.
            [[nodiscard]] std::vector<ConformanceParam> conformance_params()
            {
                const std::vector<ConformanceRow> rows = {
                    {Operation::Sum, Data::Integer},    {Operation::Sum, Data::Seeded},
                    {Operation::Dot, Data::Integer},    {Operation::Dot, Data::Seeded},
                    {Operation::Gemm, Data::Integer},   {Operation::Gemm, Data::Seeded},
                    {Operation::Softmax, Data::Seeded}, {Operation::LayerNorm, Data::Seeded}};
                std::vector<ConformanceParam> params;
                for (std::size_t slot = 0; slot < K_DEVICE_SLOTS; ++slot)
                    for (const ConformanceRow& row : rows)
                        params.push_back({slot, row});
                return params;
            }

            /// @brief What comparing one device's result with the CPU's found.
            struct Difference
            {
                std::size_t count = 0;           /**< Elements compared. */
                std::size_t differing = 0;       /**< Elements whose bytes differ. */
                std::size_t first_differing = 0; /**< Index of the first such element. */
                double largest_error = 0.0;      /**< The largest absolute difference. */
                double largest_ratio = 0.0;      /**< The largest error over its bound. */
                bool within = true;              /**< Whether every differing element is within its bound. */
            };

            /// @brief Compares a device's values with the CPU's, element by element, in bytes first.
            [[nodiscard]] Difference compare(const std::vector<float>& cpu, const std::vector<float>& device,
                                             const std::vector<double>& gauge, std::size_t roundings)
            {
                Difference difference;
                difference.count = cpu.size();
                for (std::size_t i = 0; i < cpu.size(); ++i)
                {
                    if (std::memcmp(&cpu[i], &device[i], sizeof(float)) == 0)
                        continue;
                    if (difference.differing == 0)
                        difference.first_differing = i;
                    ++difference.differing;
                    const TestSupport::ErrorRatio measured =
                        TestSupport::measure_sum_error<float>(device[i], cpu[i], gauge[i], roundings);
                    difference.largest_error = std::max(difference.largest_error, measured.error);
                    difference.largest_ratio = std::max(difference.largest_ratio, measured.ratio);
                    difference.within = difference.within && measured.is_within();
                }
                return difference;
            }
        } // namespace

        /**
         * @brief Runs a row on one device of the runtime and compares it with the CPU device.
         *
         * Softmax and layer norm are composed from the public reductions and elementwise
         * operations, one run per stage, because the library has no fused operation for either.
         * A scalar a stage produces is read back and broadcast by a filled tensor, because the
         * elementwise operations do not broadcast.
         */
        class Large_DeviceConformance : public SushiBLASTest, public ::testing::WithParamInterface<ConformanceParam>
        {
            protected:
                /** @brief Skips the case, naming what is absent, when the slot or the CPU is missing. */
                void SetUp() override
                {
                    SushiBLASTest::SetUp();
                    if (GetParam().slot >= TestEnvironment::device_count())
                        GTEST_SKIP() << "device slot " << GetParam().slot << " is absent: the runtime offers "
                                     << TestEnvironment::device_count() << " devices";
                    if (!find_cpu_slot().has_value())
                        GTEST_SKIP() << "the CPU reference device is absent from the runtime";
                    set_mode(ExecutionMode::ForceSYCL);
                }

                /** @brief Fails when the device's result leaves the row's rule; prints each difference with its reason. */
                void expect_conforms(const ConformanceRow& row)
                {
                    const Inputs inputs = make_inputs(row);
                    const std::vector<float> cpu = run_on(row, inputs, *find_cpu_slot());
                    const std::vector<float> device = run_on(row, inputs, GetParam().slot);
                    ASSERT_EQ(cpu.size(), device.size());

                    const Difference difference =
                        compare(cpu, device, bound_gauge(row, inputs, cpu),
                                rounding_count(row.operation) * K_RESULTS_COMPARED);
                    std::cout << "[conformance] " << operation_name(row.operation) << " "
                              << (row.data == Data::Integer ? "integer" : "seeded") << " on "
                              << describe_device(GetParam().slot) << " against "
                              << describe_device(*find_cpu_slot()) << ": " << difference.differing << " of "
                              << difference.count << " elements differ in bytes";
                    if (difference.differing != 0)
                        std::cout << ", first at " << difference.first_differing << ", largest error "
                                  << difference.largest_error << ", largest error over bound "
                                  << difference.largest_ratio << ", reason: " << difference_reason(row);
                    std::cout << "\n";

                    EXPECT_TRUE(difference.within)
                        << operation_name(row.operation) << " on " << describe_device(GetParam().slot)
                        << " leaves its bound; the expected difference is: " << difference_reason(row);
                    if (row.data == Data::Integer)
                        EXPECT_EQ(difference.differing, 0u) << "exact data must give the CPU's bytes";
                }

            private:
                /** @brief Returns a tensor on a slot holding the host values. */
                Tensor upload(const std::vector<float>& values, std::initializer_list<int64_t> dims, std::size_t slot)
                {
                    Tensor tensor =
                        engine->create_tensor(dims, Core::DataType::Float32, SushiRuntime::Residency::Device, slot);
                    copy_host_to_tensor(tensor, values.data(), values.size() * sizeof(float));
                    return tensor;
                }

                /** @brief Returns a tensor of extent elements on a slot, all equal to value. */
                Tensor constant(float value, int64_t extent, std::size_t slot)
                {
                    return upload(std::vector<float>(count_of(extent), value), {extent}, slot);
                }

                /** @brief Returns a one-element tensor on a slot. */
                Tensor scalar_tensor(std::size_t slot)
                {
                    return engine->create_tensor({1}, Core::DataType::Float32, SushiRuntime::Residency::Device, slot);
                }

                /** @brief Returns the one element of a tensor. */
                static float scalar_of(const Tensor& tensor)
                {
                    return read_tensor<float>(tensor).front();
                }

                /** @brief Runs everything recorded so far. */
                void run_recorded()
                {
                    recording->run().value();
                }

                /** @brief Returns the result of a row computed on a slot. */
                std::vector<float> run_on(const ConformanceRow& row, const Inputs& inputs, std::size_t slot)
                {
                    switch (row.operation)
                    {
                        case Operation::Sum: return run_sum(inputs, slot);
                        case Operation::Dot: return run_dot(inputs, slot);
                        case Operation::Gemm: return run_gemm(inputs, slot);
                        case Operation::Softmax: return run_softmax(inputs, slot);
                        case Operation::LayerNorm: return run_layer_norm(inputs, slot);
                    }
                    return {};
                }

                /** @brief Returns the sum of x. */
                std::vector<float> run_sum(const Inputs& inputs, std::size_t slot)
                {
                    Tensor x = upload(inputs.first, {K_VECTOR_LENGTH}, slot);
                    Tensor result = scalar_tensor(slot);
                    ReductionOps(recorder()).sum(x, result);
                    run_recorded();
                    return read_tensor<float>(result);
                }

                /** @brief Returns the dot product of x and y. */
                std::vector<float> run_dot(const Inputs& inputs, std::size_t slot)
                {
                    Tensor x = upload(inputs.first, {K_VECTOR_LENGTH}, slot);
                    Tensor y = upload(inputs.second, {K_VECTOR_LENGTH}, slot);
                    Tensor result = scalar_tensor(slot);
                    BLASOps(recorder()).dot(x, y, result);
                    run_recorded();
                    return read_tensor<float>(result);
                }

                /** @brief Returns C = A * B. */
                std::vector<float> run_gemm(const Inputs& inputs, std::size_t slot)
                {
                    Tensor a = upload(inputs.first, {K_GEMM_M, K_GEMM_K}, slot);
                    Tensor b = upload(inputs.second, {K_GEMM_K, K_GEMM_N}, slot);
                    Tensor c = engine->create_tensor({K_GEMM_M, K_GEMM_N}, Core::DataType::Float32,
                                                     SushiRuntime::Residency::Device, slot);
                    BLASOps(recorder()).gemm(a, b, c);
                    run_recorded();
                    return read_tensor<float>(c);
                }

                /** @brief Returns exp(x - max(x)) divided by its sum. */
                std::vector<float> run_softmax(const Inputs& inputs, std::size_t slot)
                {
                    Tensor x = upload(inputs.first, {K_ROW_LENGTH}, slot);
                    Tensor maximum = scalar_tensor(slot);
                    ReductionOps(recorder()).max(x, maximum);
                    run_recorded();

                    Tensor maximum_row = constant(scalar_of(maximum), K_ROW_LENGTH, slot);
                    Tensor shifted = constant(0.0f, K_ROW_LENGTH, slot);
                    ElementwiseOps(recorder()).sub(x, maximum_row, shifted);
                    ElementwiseOps(recorder()).exp(shifted);
                    Tensor total = scalar_tensor(slot);
                    ReductionOps(recorder()).sum(shifted, total);
                    run_recorded();

                    Tensor total_row = constant(scalar_of(total), K_ROW_LENGTH, slot);
                    Tensor result = constant(0.0f, K_ROW_LENGTH, slot);
                    ElementwiseOps(recorder()).div(shifted, total_row, result);
                    run_recorded();
                    return read_tensor<float>(result);
                }

                /** @brief Returns (x - mean) / sqrt(var + eps) * gamma + beta. */
                std::vector<float> run_layer_norm(const Inputs& inputs, std::size_t slot)
                {
                    Tensor x = upload(inputs.first, {K_ROW_LENGTH}, slot);
                    Tensor gamma = upload(inputs.gamma, {K_ROW_LENGTH}, slot);
                    Tensor beta = upload(inputs.beta, {K_ROW_LENGTH}, slot);
                    Tensor mean = scalar_tensor(slot);
                    Tensor variance = scalar_tensor(slot);
                    ReductionOps(recorder()).mean(x, mean);
                    ReductionOps(recorder()).var(x, variance);
                    run_recorded();

                    Tensor mean_row = constant(scalar_of(mean), K_ROW_LENGTH, slot);
                    Tensor deviation = constant(scalar_of(variance) + K_LAYER_NORM_EPSILON, K_ROW_LENGTH, slot);
                    Tensor centred = constant(0.0f, K_ROW_LENGTH, slot);
                    ElementwiseOps(recorder()).sub(x, mean_row, centred);
                    ElementwiseOps(recorder()).sqrt(deviation);
                    Tensor normalised = constant(0.0f, K_ROW_LENGTH, slot);
                    ElementwiseOps(recorder()).div(centred, deviation, normalised);
                    Tensor scaled = constant(0.0f, K_ROW_LENGTH, slot);
                    ElementwiseOps(recorder()).mul(normalised, gamma, scaled);
                    Tensor result = constant(0.0f, K_ROW_LENGTH, slot);
                    ElementwiseOps(recorder()).add(scaled, beta, result);
                    run_recorded();
                    return read_tensor<float>(result);
                }
        };

        TEST_P(Large_DeviceConformance, TheDeviceResultMatchesTheCPUOrStaysWithinItsNamedBound)
        {
            expect_conforms(GetParam().row);
        }

        INSTANTIATE_TEST_SUITE_P(, Large_DeviceConformance, ::testing::ValuesIn(conformance_params()), case_name);

        /** @brief Holds the coverage promise: every device the runtime offers has a slot above. */
        class Large_DeviceConformanceCoverage : public ::testing::Test
        {
        };

        TEST_F(Large_DeviceConformanceCoverage, EveryDeviceOfTheRuntimeHasASlotInTheConformanceSuite)
        {
            EXPECT_LE(TestEnvironment::device_count(), K_DEVICE_SLOTS)
                << "the runtime offers more devices than the conformance suite has slots; raise K_DEVICE_SLOTS";
        }
    } // namespace Test
} // namespace SushiBLAS
