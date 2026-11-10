/**************************************************************************/
/* test_random_reference.cpp                                              */
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
 * @file test_random_reference.cpp
 * @brief Tests that each random operation draws one stream under both forced modes.
 * @author Mustafa Garip
 */

#include <cmath>
#include <limits>
#include <string>
#include <vector>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <functional>
#include <type_traits>
#include <initializer_list>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <SushiBLAS/SushiBLAS.hpp>
#include <SushiBLAS/orthonormal/orthonormal_sweep.hpp>
#include <SushiBLAS/engine_fixture/sushiblas_test.hpp>
#include <SushiBLAS/tensor_expectations/scalar_dtype.hpp>
#include <SushiBLAS/test_runner/test_environment.hpp>
#include "support/forced_mode_checks.hpp"
#include <SushiBLAS/tensor_expectations/tolerance_bounds.hpp>
namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            /// @brief Holds the element count of a fill: 4 * 1024 + 3, a partial last block.
            constexpr int64_t K_ELEMENTS = 4099;

            /// @brief Holds the seed of every case but the large-lambda Poisson one.
            constexpr uint64_t K_SEED = 20261007;

            /// @brief Holds the seed the large-lambda Poisson case is pinned to.
            constexpr uint64_t K_LARGE_LAMBDA_SEED = 42;

            /// @brief Holds the value a tensor carries before a fill; no fill here draws it.
            constexpr float K_UNWRITTEN = -77.0f;

            /// @brief Holds the value constant fills with: exact in HALF, 2 in an integer.
            constexpr double K_CONSTANT = 2.5;

            /// @brief Holds the lower bound of the uniform fills.
            constexpr double K_LOW = -2.0;

            /// @brief Holds the upper bound of the uniform fills.
            constexpr double K_HIGH = 6.0;

            /// @brief Holds the lowest integer discrete_uniform draws.
            constexpr int32_t K_LOWEST_INTEGER = -7;

            /// @brief Holds the highest integer discrete_uniform draws.
            constexpr int32_t K_HIGHEST_INTEGER = 9;

            /// @brief Holds the probability bernoulli draws a 1 with.
            constexpr double K_PROBABILITY = 0.3;

            /// @brief Holds the fan-in of the He fills.
            constexpr int64_t K_HE_FAN_IN = 8;

            /// @brief Holds the fan-in of the Xavier fills.
            constexpr int64_t K_XAVIER_FAN_IN = 5;

            /// @brief Holds the fan-out of the Xavier fills.
            constexpr int64_t K_XAVIER_FAN_OUT = 3;

            /// @brief Holds sqrt(2 / 8), the standard deviation of both fan-scaled normal fills.
            constexpr double K_FAN_STDDEV = 0.5;

            /// @brief Holds the mean of the normal fill.
            constexpr double K_NORMAL_MEAN = 0.5;

            /// @brief Holds the standard deviation of the normal fill.
            constexpr double K_NORMAL_STDDEV = 2.0;

            /// @brief Holds the rate of the exponential fill.
            constexpr double K_EXPONENTIAL_RATE = 1.5;

            /// @brief Holds the mean of the normal under the log-normal fill.
            constexpr double K_LOG_NORMAL_MEAN = 0.0;

            /// @brief Holds the standard deviation of the normal under the log-normal fill.
            constexpr double K_LOG_NORMAL_STDDEV = 0.5;

            /// @brief Holds a Poisson mean below 10, where the draw is Knuth's product.
            constexpr double K_SMALL_LAMBDA = 4.0;

            /// @brief Holds a Poisson mean above 10, where the draw is PTRS.
            constexpr double K_LARGE_LAMBDA = 50.0;

            /// @brief Holds the counts a PTRS draw may differ by between the paths: one.
            constexpr double K_PTRS_COUNT_SLACK = 1.0;

            /// @brief Holds the counts a Knuth draw may differ by between the paths: none.
            constexpr double K_KNUTH_COUNT_SLACK = 0.0;

            /// @brief Holds the lower cut of the truncated normal fill, in standard deviations.
            constexpr double K_LOWER_CUT = -2.0;

            /// @brief Holds the upper cut of the truncated normal fill, in standard deviations.
            constexpr double K_UPPER_CUT = 2.0;

            /// @brief Holds the FLOAT32 roundings a Gram entry may carry, per unit of gain squared.
            constexpr std::size_t K_GRAM_OPERATIONS = 128;

            /// @brief Holds the largest radius a Box-Muller draw reaches in double.
            constexpr double K_BOX_MULLER_RADIUS = 9.0;

            /// @brief Holds the bit pattern of 1.0 in HALF.
            constexpr uint16_t K_HALF_ONE_BITS = 0x3C00;

            /** @brief Names a callable that records one random operation on a tensor. */
            using Operation = std::function<void(RandomOps&, Tensor&)>;

            /** @brief Holds what one operation left in two tensors, one per forced mode. */
            template<typename T>
            struct ModeDraws
            {
                std::vector<T> reference; /**< The elements written under ForceReference. */
                std::vector<T> device;    /**< The elements written under ForceSYCL. */
            };

            /** @brief Holds the allowance of a comparison: absolute + relative * |reference|. */
            struct Tolerance
            {
                double absolute = 0.0; /**< The part that does not depend on the value. */
                double relative = 0.0; /**< The part multiplied by the reference's magnitude. */
            };

            /** @brief Holds how two results differ when their elements are compared bit by bit. */
            struct BitDifference
            {
                std::size_t count = 0; /**< How many elements differ. */
                std::size_t first = 0; /**< The first differing element; the size when none. */
            };

            /** @brief Names one matrix shape and the vectors orthogonal makes orthonormal in it. */
            struct OrthogonalCase
            {
                const char* name;           /**< The shape, for the failure message. */
                int64_t rows;               /**< The row count. */
                int64_t cols;               /**< The column count. */
                double gain;                /**< The gain the matrix is scaled by. */
                Kernels::VectorSet vectors; /**< The shorter axis's vectors, row-major. */
            };

            /// @brief Holds the wide and the tall matrix the two orthogonal paths are compared on, and one vector at a gain of 3.
            constexpr OrthogonalCase K_ORTHOGONAL_CASES[] = {
                {"Matrix_8_by_16", 8, 16, 1.0, {8, 16, 1, 16}},
                {"Matrix_16_by_8", 16, 8, 1.0, {8, 16, 8, 1}},
                {"Matrix_1_by_4_at_gain_3", 1, 4, 3.0, {1, 4, 1, 4}},
            };

            /** @brief Returns single for FLOAT32 and wide for FLOAT64. */
            template<typename T>
            [[nodiscard]] constexpr double by_precision(double single, double wide) noexcept
            {
                return std::is_same_v<T, float> ? single : wide;
            }

            /** @brief Returns 16 epsilon (|mean| + 9 |stddev|), the allowance of a normal draw. */
            template<typename T>
            [[nodiscard]] double box_muller_allowance(double mean, double stddev) noexcept
            {
                const double epsilon = static_cast<double>(std::numeric_limits<T>::epsilon());
                return 16.0 * epsilon * (std::abs(mean) + K_BOX_MULLER_RADIUS * std::abs(stddev));
            }

            /** @brief Returns 8 epsilon, the relative allowance of one log1p or exp of T. */
            template<typename T>
            [[nodiscard]] double builtin_allowance() noexcept
            {
                return 8.0 * static_cast<double>(std::numeric_limits<T>::epsilon());
            }

            /**
             * @brief Returns count distinct values of T.
             * @return Element i holds i; a HALF element holds the bits of 1.0 plus i.
             */
            template<typename T>
            [[nodiscard]] std::vector<T> distinct_values(std::size_t count)
            {
                std::vector<T> values(count);
                for (std::size_t i = 0; i < count; ++i)
                {
                    if constexpr (std::is_same_v<T, sycl::half>)
                    {
                        static_assert(sizeof(T) == sizeof(uint16_t), "HALF is 16 bits wide");

                        const uint16_t bits = static_cast<uint16_t>(K_HALF_ONE_BITS + i);
                        std::memcpy(static_cast<void*>(&values[i]), &bits, sizeof(bits));
                    }
                    else
                    {
                        values[i] = static_cast<T>(i);
                    }
                }
                return values;
            }

            /** @brief Reports whether two elements hold the same bits. */
            template<typename T>
            [[nodiscard]] bool has_equal_bits(const T& a, const T& b) noexcept
            {
                return std::memcmp(static_cast<const void*>(&a), static_cast<const void*>(&b),
                                   sizeof(T)) == 0;
            }

            /**
             * @brief Returns how many elements of two results differ in their bits, and the first.
             * @pre Both results hold the same number of elements.
             */
            template<typename T>
            [[nodiscard]] BitDifference bit_difference(const std::vector<T>& expected,
                                                       const std::vector<T>& actual)
            {
                BitDifference difference;
                difference.first = expected.size();

                for (std::size_t i = 0; i < expected.size(); ++i)
                {
                    if (has_equal_bits(expected[i], actual[i]))
                        continue;

                    if (difference.count == 0)
                        difference.first = i;
                    ++difference.count;
                }
                return difference;
            }

            /** @brief Returns how many elements hold the bits of value. */
            template<typename T>
            [[nodiscard]] std::size_t count_holding(const std::vector<T>& values, const T& value)
            {
                std::size_t count = 0;
                for (const T& element : values)
                    count += has_equal_bits(element, value) ? 1 : 0;
                return count;
            }

            /** @brief Expects two results to hold the same bits in every element. */
            template<typename T>
            void expect_equal_bits(const std::vector<T>& reference, const std::vector<T>& device)
            {
                ASSERT_EQ(reference.size(), device.size());

                const BitDifference difference = bit_difference(reference, device);
                EXPECT_EQ(difference.count, std::size_t{0})
                    << difference.count << " of " << reference.size()
                    << " elements differ; the first is element " << difference.first;
            }

            /** @brief Expects every device element within tolerance of its reference element. */
            template<typename T>
            void expect_within(const std::vector<T>& reference, const std::vector<T>& device,
                               const Tolerance& tolerance)
            {
                ASSERT_EQ(reference.size(), device.size());
                ASSERT_FALSE(reference.empty());

                std::size_t outside = 0;
                std::size_t worst = 0;
                double worst_difference = 0.0;
                for (std::size_t i = 0; i < reference.size(); ++i)
                {
                    const double expected = static_cast<double>(reference[i]);
                    const double difference = std::abs(static_cast<double>(device[i]) - expected);
                    const double allowed =
                        tolerance.absolute + tolerance.relative * std::abs(expected);

                    // Both comparisons are written so that a NaN difference fails them.
                    if (!(difference <= allowed))
                        ++outside;
                    if (!(difference <= worst_difference))
                    {
                        worst = i;
                        worst_difference = difference;
                    }
                }

                EXPECT_EQ(outside, std::size_t{0})
                    << "the largest difference is " << worst_difference << " at element " << worst
                    << ", where the reference holds " << static_cast<double>(reference[worst]);
            }

            /** @brief Expects the Gram matrix of set's vectors to be diagonal times identity. */
            template<typename T>
            void expect_orthonormal(const std::vector<T>& matrix, const Kernels::VectorSet& set,
                                    double diagonal)
            {
                const auto element = [&matrix, &set](int64_t vector, int64_t k)
                {
                    const int64_t index = vector * set.vec_stride + k * set.elem_stride;
                    return static_cast<double>(matrix[static_cast<std::size_t>(index)]);
                };

                const double tolerance = absolute_tolerance<float>(K_GRAM_OPERATIONS, diagonal);

                for (int64_t i = 0; i < set.count; ++i)
                {
                    for (int64_t j = i; j < set.count; ++j)
                    {
                        double dot = 0.0;
                        for (int64_t k = 0; k < set.length; ++k)
                            dot += element(i, k) * element(j, k);

                        EXPECT_NEAR(dot, i == j ? diagonal : 0.0, tolerance)
                            << "Gram entry (" << i << ", " << j << ")";
                    }
                }
            }
        } // namespace

        /** @brief Runs one random operation under both forced modes from one seed. */
        class Medium_RandomReference : public SushiBLASTest
        {
            protected:
                /** @brief Builds the engine and skips the case on a device without fp64. */
                void SetUp() override
                {
                    SushiBLASTest::SetUp();

                    if (!device_has(sycl::aspect::fp64))
                        GTEST_SKIP() << "the device does not report the fp64 aspect";
                }

                /** @brief Reports whether the device of the test context has aspect. */
                [[nodiscard]] static bool device_has(sycl::aspect aspect)
                {
                    return TestEnvironment::queue().get_device().has(aspect);
                }

                /**
                 * @brief Returns a FLOAT32 tensor of dims in device-only USM, or nothing.
                 * @return Empty when the allocation fails or the pointer is not usm::alloc::device.
                 */
                [[nodiscard]] std::optional<Tensor> device_only_tensor(
                    std::initializer_list<int64_t> dims)
                {
                    try
                    {
                        Tensor t = engine->create_tensor(dims, Core::DataType::Float32,
                                                         SushiRuntime::Residency::Device);
                        const sycl::context context =
                            TestEnvironment::queue().get_context();
                        if (sycl::get_pointer_type(t.data(), context) != sycl::usm::alloc::device)
                            return std::nullopt;
                        return t;
                    }
                    catch (const std::exception&)
                    {
                        return std::nullopt;
                    }
                }

                /**
                 * @brief Returns what operation leaves in a tensor of dims that held initial.
                 * @param seed The stream is reseeded with it before operation records.
                 * @param initial Must hold one element per element of dims.
                 */
                template<typename T>
                [[nodiscard]] std::vector<T> run_under(ExecutionMode mode, uint64_t seed,
                                                       std::initializer_list<int64_t> dims,
                                                       const std::vector<T>& initial,
                                                       const Operation& operation)
                {
                    set_mode(mode);

                    Tensor t = engine->create_tensor(dims, sushiblas_dtype<T>());
                    SushiBLAS::IO io;
                    io.from_vector(t, initial);

                    RandomOps random(recorder(), engine->rng());
                    random.set_seed(seed);
                    operation(random, t);
                    recording->run().value();
                    return io.to_vector<T>(t);
                }

                /** @brief Returns what operation leaves under each of the two forced modes. */
                template<typename T>
                [[nodiscard]] ModeDraws<T> run_in_both_modes(uint64_t seed,
                                                             std::initializer_list<int64_t> dims,
                                                             const std::vector<T>& initial,
                                                             const Operation& operation)
                {
                    ModeDraws<T> draws;
                    draws.reference =
                        run_under(ExecutionMode::ForceReference, seed, dims, initial, operation);
                    draws.device =
                        run_under(ExecutionMode::ForceSYCL, seed, dims, initial, operation);
                    return draws;
                }

                /** @brief Returns both modes' fills and expects each to write every element. */
                template<typename T>
                [[nodiscard]] ModeDraws<T> fill_in_both_modes(uint64_t seed,
                                                              std::initializer_list<int64_t> dims,
                                                              const Operation& fill)
                {
                    std::size_t count = 1;
                    for (const int64_t dim : dims)
                        count *= static_cast<std::size_t>(dim);

                    const T unwritten = static_cast<T>(K_UNWRITTEN);
                    const ModeDraws<T> draws =
                        run_in_both_modes(seed, dims, std::vector<T>(count, unwritten), fill);

                    EXPECT_EQ(count_holding(draws.reference, unwritten), std::size_t{0})
                        << "ForceReference left elements unwritten";
                    EXPECT_EQ(count_holding(draws.device, unwritten), std::size_t{0})
                        << "ForceSYCL left elements unwritten";
                    return draws;
                }

                /** @brief Expects fill to write equal bits to K_ELEMENTS elements on both paths. */
                template<typename T>
                void expect_same_bits(const char* operation, uint64_t seed, const Operation& fill)
                {
                    SCOPED_TRACE(trace_of<T>(operation));

                    const ModeDraws<T> draws = fill_in_both_modes<T>(seed, {K_ELEMENTS}, fill);
                    expect_equal_bits(draws.reference, draws.device);
                }

                /** @brief Expects fill to write K_ELEMENTS elements that agree within tolerance. */
                template<typename T>
                void expect_close(const char* operation, const Tolerance& tolerance,
                                  const Operation& fill)
                {
                    SCOPED_TRACE(trace_of<T>(operation));

                    const ModeDraws<T> draws = fill_in_both_modes<T>(K_SEED, {K_ELEMENTS}, fill);
                    expect_within(draws.reference, draws.device, tolerance);
                }

                /** @brief Expects constant to write equal bits of T on both paths. */
                template<typename T>
                void expect_same_constant()
                {
                    expect_same_bits<T>("constant", K_SEED, [](RandomOps& random, Tensor& t)
                    {
                        random.constant(t, K_CONSTANT);
                    });
                }

                /** @brief Expects shuffle to permute distinct values of T alike on both paths. */
                template<typename T>
                void expect_same_shuffle()
                {
                    SCOPED_TRACE(trace_of<T>("shuffle"));

                    const std::vector<T> ordered =
                        distinct_values<T>(static_cast<std::size_t>(K_ELEMENTS));
                    const ModeDraws<T> draws =
                        run_in_both_modes(K_SEED, {K_ELEMENTS}, ordered,
                                          [](RandomOps& random, Tensor& t) { random.shuffle(t); });

                    EXPECT_GT(bit_difference(ordered, draws.reference).count, std::size_t{0})
                        << "ForceReference left every element in place";
                    expect_equal_bits(draws.reference, draws.device);
                }

                /** @brief Expects uniform on [K_LOW, K_HIGH) to write equal bits of T on both paths. */
                template<typename T>
                void expect_same_uniform()
                {
                    expect_same_bits<T>("uniform", K_SEED, [](RandomOps& random, Tensor& t)
                    {
                        random.uniform(t, K_LOW, K_HIGH);
                    });
                }

                /** @brief Expects discrete_uniform to write equal bits of T on both paths. */
                template<typename T>
                void expect_same_discrete_uniform()
                {
                    expect_same_bits<T>("discrete_uniform", K_SEED, [](RandomOps& random, Tensor& t)
                    {
                        random.discrete_uniform(t, K_LOWEST_INTEGER, K_HIGHEST_INTEGER);
                    });
                }

                /** @brief Expects bernoulli to write equal bits of T on both paths. */
                template<typename T>
                void expect_same_bernoulli()
                {
                    expect_same_bits<T>("bernoulli", K_SEED, [](RandomOps& random, Tensor& t)
                    {
                        random.bernoulli(t, K_PROBABILITY);
                    });
                }

                /** @brief Expects he_uniform and xavier_uniform to write equal bits of T on both paths. */
                template<typename T>
                void expect_same_uniform_initializers()
                {
                    expect_same_bits<T>("he_uniform", K_SEED, [](RandomOps& random, Tensor& t)
                    {
                        random.he_uniform(t, K_HE_FAN_IN);
                    });
                    expect_same_bits<T>("xavier_uniform", K_SEED, [](RandomOps& random, Tensor& t)
                    {
                        random.xavier_uniform(t, K_XAVIER_FAN_IN, K_XAVIER_FAN_OUT);
                    });
                }

                /**
                 * @brief Expects poisson(lambda) to agree on both paths within slack counts.
                 * @note A PTRS draw may differ by one count, because the device log and lgamma can
                 *       flip the acceptance test of an element. A Knuth draw calls no such builtin
                 *       and agrees exactly.
                 */
                template<typename T>
                void expect_poisson_counts_agree(double lambda, uint64_t seed, double slack)
                {
                    SCOPED_TRACE(trace_of<T>("poisson"));

                    const ModeDraws<T> draws = fill_in_both_modes<T>(
                        seed, {K_ELEMENTS}, [lambda](RandomOps& random, Tensor& t)
                    {
                        random.poisson(t, lambda);
                    });
                    expect_within(draws.reference, draws.device, Tolerance{slack, 0.0});
                }

                /** @brief Expects normal to agree in T within the Box-Muller allowance. */
                template<typename T>
                void expect_normal_agrees()
                {
                    expect_close<T>("normal",
                                    Tolerance{box_muller_allowance<T>(K_NORMAL_MEAN, K_NORMAL_STDDEV), 0.0},
                                    [](RandomOps& random, Tensor& t)
                                    {
                                        random.normal(t, K_NORMAL_MEAN, K_NORMAL_STDDEV);
                                    });
                }

                /** @brief Expects he_normal and xavier_normal to agree in T within the Box-Muller allowance. */
                template<typename T>
                void expect_normal_initializers_agree()
                {
                    const Tolerance fan_tolerance{box_muller_allowance<T>(0.0, K_FAN_STDDEV), 0.0};

                    expect_close<T>("he_normal", fan_tolerance, [](RandomOps& random, Tensor& t)
                    {
                        random.he_normal(t, K_HE_FAN_IN);
                    });
                    expect_close<T>("xavier_normal", fan_tolerance, [](RandomOps& random, Tensor& t)
                    {
                        random.xavier_normal(t, K_XAVIER_FAN_IN, K_XAVIER_FAN_OUT);
                    });
                }

                /** @brief Expects exponential to agree in T within the relative allowance of one log1p. */
                template<typename T>
                void expect_exponential_agrees()
                {
                    expect_close<T>("exponential", Tolerance{0.0, builtin_allowance<T>()},
                                    [](RandomOps& random, Tensor& t)
                                    {
                                        random.exponential(t, K_EXPONENTIAL_RATE);
                                    });
                }

                /** @brief Expects log_normal to agree in T; the error of the normal passes through exp as a relative error. */
                template<typename T>
                void expect_log_normal_agrees()
                {
                    const Tolerance tolerance{
                        0.0, box_muller_allowance<T>(K_LOG_NORMAL_MEAN, K_LOG_NORMAL_STDDEV) +
                                 builtin_allowance<T>()};

                    expect_close<T>("log_normal", tolerance, [](RandomOps& random, Tensor& t)
                    {
                        random.log_normal(t, K_LOG_NORMAL_MEAN, K_LOG_NORMAL_STDDEV);
                    });
                }

                /** @brief Expects truncated_normal to agree in T; the bounds are tested in test_distributions.cpp. */
                template<typename T>
                void expect_truncated_normal_agrees()
                {
                    SCOPED_TRACE(trace_of<T>("truncated_normal"));

                    const ModeDraws<T> draws = fill_in_both_modes<T>(
                        K_SEED, {K_ELEMENTS}, [](RandomOps& random, Tensor& t)
                        {
                            random.truncated_normal(t, 0.0, 1.0, K_LOWER_CUT, K_UPPER_CUT);
                        });

                    expect_within(draws.reference, draws.device,
                                  Tolerance{by_precision<T>(absolute_tolerance<float>(2048), absolute_tolerance<double>(1048576)), 0.0});
                }

                /** @brief Expects both paths of orthogonal to be orthonormal and to agree in T. */
                template<typename T>
                void expect_orthogonal_agrees(const OrthogonalCase& matrix)
                {
                    SCOPED_TRACE(trace_of<T>(matrix.name));

                    const ModeDraws<T> draws = fill_in_both_modes<T>(
                        K_SEED, {matrix.rows, matrix.cols}, [&matrix](RandomOps& random, Tensor& t)
                        {
                            random.orthogonal(t, matrix.gain);
                        });

                    expect_orthonormal(draws.reference, matrix.vectors, matrix.gain * matrix.gain);
                    expect_orthonormal(draws.device, matrix.vectors, matrix.gain * matrix.gain);
                    expect_within(draws.reference, draws.device,
                                  Tolerance{by_precision<T>(absolute_tolerance<float>(2048), absolute_tolerance<double>(65536)) * matrix.gain, 0.0});
                }
        };

        /** @brief Tests that uniform has equal bits on both paths for real and complex elements. */
        TEST_F(Medium_RandomReference, UniformDrawsAreTheSameStream)
        {
            expect_same_uniform<float>();
            expect_same_uniform<double>();
            expect_same_uniform<std::complex<float>>();
        }

        /** @brief Tests that constant has equal bits on both paths for every dtype but HALF. */
        TEST_F(Medium_RandomReference, ConstantFillsAreTheSameStream)
        {
            expect_same_constant<float>();
            expect_same_constant<double>();
            expect_same_constant<std::complex<float>>();
            expect_same_constant<std::complex<double>>();
            expect_same_constant<int32_t>();
            expect_same_constant<int64_t>();
        }

        /** @brief Tests that discrete_uniform has equal bits on both paths. */
        TEST_F(Medium_RandomReference, DiscreteUniformDrawsAreTheSameStream)
        {
            expect_same_discrete_uniform<float>();
            expect_same_discrete_uniform<double>();
        }

        /** @brief Tests that bernoulli has equal bits on both paths. */
        TEST_F(Medium_RandomReference, BernoulliDrawsAreTheSameStream)
        {
            expect_same_bernoulli<float>();
            expect_same_bernoulli<double>();
        }

        /** @brief Tests that the uniform initializers have equal bits on both paths. */
        TEST_F(Medium_RandomReference, UniformInitializersAreTheSameStream)
        {
            expect_same_uniform_initializers<float>();
            expect_same_uniform_initializers<double>();
        }

        /** @brief Tests that shuffle permutes alike on both paths. */
        TEST_F(Medium_RandomReference, ShuffleIsTheSamePermutation)
        {
            expect_same_shuffle<float>();
            expect_same_shuffle<double>();
        }

        /** @brief Runs the cases that need the device to compute in HALF. */
        class Medium_RandomReferenceHalf : public Medium_RandomReference
        {
            protected:
                /** @brief Builds the engine and skips the case on a device without fp16. */
                void SetUp() override
                {
                    Medium_RandomReference::SetUp();

                    if (!IsSkipped() && !device_has(sycl::aspect::fp16))
                        GTEST_SKIP() << "the device does not report the fp16 aspect";
                }
        };

        /** @brief Tests that constant leaves equal bits in HALF on both paths. */
        TEST_F(Medium_RandomReferenceHalf, ConstantIsMovedToTheSameBits)
        {
            expect_same_constant<sycl::half>();
        }

        /** @brief Tests that shuffle leaves equal bits in HALF on both paths. */
        TEST_F(Medium_RandomReferenceHalf, ShuffleIsMovedToTheSameBits)
        {
            expect_same_shuffle<sycl::half>();
        }

        /** @brief Names a Poisson mean, its seed and the count slack between the paths. */
        struct PoissonRow
        {
            const char* name; /**< The row name a failure shows. */
            double lambda;    /**< The mean of the draw. */
            uint64_t seed;    /**< The seed of both runs. */
            double slack;     /**< The counts the paths may differ by. */
        };

        /** @brief Returns the Knuth row, which agrees exactly, and the PTRS row, which agrees within one count. */
        std::vector<PoissonRow> poisson_rows()
        {
            return {{"KnuthLambdaFour", K_SMALL_LAMBDA, K_SEED, K_KNUTH_COUNT_SLACK},
                    {"PtrsLambdaFifty", K_LARGE_LAMBDA, K_LARGE_LAMBDA_SEED, K_PTRS_COUNT_SLACK}};
        }

        /** @brief Runs one row of poisson_rows(). */
        class Medium_RandomReferencePoisson : public Medium_RandomReference,
                                              public ::testing::WithParamInterface<PoissonRow>
        {

        };

        /**
         * @brief Tests that poisson gives both paths counts within the slack of the row.
         * @note A PTRS failure names an element whose acceptance test sits inside the difference
         *       between the device log or lgamma and the host one. It does not show a wrong stream.
         */
        TEST_P(Medium_RandomReferencePoisson, BothPathsDrawTheSameCounts)
        {
            const PoissonRow row = GetParam();

            expect_poisson_counts_agree<float>(row.lambda, row.seed, row.slack);
            expect_poisson_counts_agree<double>(row.lambda, row.seed, row.slack);
        }

        INSTANTIATE_TEST_SUITE_P(
            , Medium_RandomReferencePoisson, ::testing::ValuesIn(poisson_rows()),
            [](const ::testing::TestParamInfo<PoissonRow>& info)
            { return std::string(info.param.name); });

        /** @brief Tests that normal agrees on both paths within its bound. */
        TEST_F(Medium_RandomReference, NormalDrawsAgreeWithinTheirBound)
        {
            expect_normal_agrees<float>();
            expect_normal_agrees<double>();
        }

        /** @brief Tests that he_normal and xavier_normal agree on both paths within their bound. */
        TEST_F(Medium_RandomReference, NormalInitializersAgreeWithinTheirBound)
        {
            expect_normal_initializers_agree<float>();
            expect_normal_initializers_agree<double>();
        }

        /** @brief Tests that exponential agrees on both paths within its bound. */
        TEST_F(Medium_RandomReference, ExponentialDrawsAgreeWithinTheirBound)
        {
            expect_exponential_agrees<float>();
            expect_exponential_agrees<double>();
        }

        /** @brief Tests that log_normal agrees on both paths within its bound. */
        TEST_F(Medium_RandomReference, LogNormalDrawsAgreeWithinTheirBound)
        {
            expect_log_normal_agrees<float>();
            expect_log_normal_agrees<double>();
        }

        /**
         * @brief Tests that truncated_normal agrees on both paths, which differ by at most 2048
         *        FLOAT32 roundings or 2^20 FLOAT64 roundings of a value of order one.
         */
        TEST_F(Medium_RandomReference, TruncatedNormalAgreesOnBothPaths)
        {
            expect_truncated_normal_agrees<float>();
            expect_truncated_normal_agrees<double>();
        }

        /**
         * @brief Tests that each path of orthogonal is orthonormal at its gain, and that the two
         *        differ by at most 2048 FLOAT32 roundings or 65536 FLOAT64 roundings of a value of
         *        order one, scaled by the gain.
         */
        TEST_F(Medium_RandomReference, BothOrthogonalPathsAreOrthonormal)
        {
            for (const OrthogonalCase& matrix : K_ORTHOGONAL_CASES)
            {
                expect_orthogonal_agrees<float>(matrix);
                expect_orthogonal_agrees<double>(matrix);
            }
        }

        /** @brief Holds two device-only tensors for the cases that check what a refused call reserves. */
        class Medium_RandomReferenceDeviceOnly : public Medium_RandomReference
        {
            protected:
                /** @brief Builds the engine and the tensors, and skips the case on a device without device-only USM. */
                void SetUp() override
                {
                    Medium_RandomReference::SetUp();

                    if (IsSkipped())
                        return;

                    values_ = device_only_tensor({4});
                    matrix_ = device_only_tensor({3, 3});
                    if (!values_ || !matrix_)
                        GTEST_SKIP() << "This device offers no device-only USM.";
                }

                /** @brief Holds the vector the refused calls are made on. */
                std::optional<Tensor> values_;

                /** @brief Holds the matrix the refused orthogonal call is made on. */
                std::optional<Tensor> matrix_;
        };

        /** @brief Tests that a call refused for a device-only tensor reserves no offset. */
        TEST_F(Medium_RandomReferenceDeviceOnly, ARefusedCallReservesNoOffset)
        {
            RandomOps random(recorder(), engine->rng());
            random.set_seed(K_SEED);

            set_mode(ExecutionMode::ForceReference);
            const uint64_t offset_before = engine->rng().reserve(0).offset;

            EXPECT_THROW(random.uniform(*values_), std::invalid_argument);
            EXPECT_THROW(random.shuffle(*values_), std::invalid_argument);
            EXPECT_THROW(random.orthogonal(*matrix_), std::invalid_argument);
            EXPECT_EQ(offset_before, engine->rng().reserve(0).offset);
        }

        /** @brief Tests that an accepted call under ForceSYCL reserves exactly one offset. */
        TEST_F(Medium_RandomReferenceDeviceOnly, AnAcceptedCallUnderForceSYCLReservesOneOffset)
        {
            RandomOps random(recorder(), engine->rng());
            random.set_seed(K_SEED);

            set_mode(ExecutionMode::ForceSYCL);
            const uint64_t offset_before = engine->rng().reserve(0).offset;

            random.uniform(*values_);

            EXPECT_EQ(offset_before + 1, engine->rng().reserve(0).offset);
            recording->run().value();
        }
    } // namespace Test
} // namespace SushiBLAS
