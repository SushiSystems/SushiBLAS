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
#include <SushiBLAS/SushiBLAS.h>
#include <SushiBLAS/support/orthonormal_sweep.hpp>
#include "../../common/test_utils.hpp"

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

            /// @brief Holds the lower cut of the truncated normal fill, in standard deviations.
            constexpr double K_LOWER_CUT = -2.0;

            /// @brief Holds the upper cut of the truncated normal fill, in standard deviations.
            constexpr double K_UPPER_CUT = 2.0;

            /// @brief Holds the gain of the orthogonal fills the two paths are compared on.
            constexpr double K_GAIN = 1.0;

            /// @brief Holds the gain of the single-vector orthogonal fill.
            constexpr double K_SINGLE_VECTOR_GAIN = 3.0;

            /// @brief Holds the tolerance test_orthogonal.cpp allows a Gram entry at gain 1.
            constexpr double K_GRAM_TOLERANCE = 1e-4;

            /// @brief Holds the tolerance test_orthogonal.cpp allows the norm of one vector.
            constexpr double K_NORM_TOLERANCE = 1e-4;

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
                std::vector<T> reference; /**< The elements written under FORCE_REFERENCE. */
                std::vector<T> device;    /**< The elements written under FORCE_SYCL. */
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
                Kernels::VectorSet vectors; /**< The shorter axis's vectors, row-major. */
            };

            /// @brief Holds the wide and the tall matrix the two orthogonal paths are compared on.
            constexpr OrthogonalCase K_ORTHOGONAL_CASES[] = {
                {"8 by 16", 8, 16, {8, 16, 1, 16}},
                {"16 by 8", 16, 8, {8, 16, 8, 1}},
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

            /** @brief Returns "<operation> over <the dtype of T>" for a failure message. */
            template<typename T>
            [[nodiscard]] std::string trace_of(const char* operation)
            {
                const Core::DataType dtype = get_sushiblas_dtype<T>();
                return std::string(operation) + " over " + Core::to_string(dtype);
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

            /** @brief Returns the Euclidean norm of the elements, summed in double. */
            template<typename T>
            [[nodiscard]] double norm_of(const std::vector<T>& values)
            {
                double squares = 0.0;
                for (const T& value : values)
                    squares += static_cast<double>(value) * static_cast<double>(value);
                return std::sqrt(squares);
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

            /** @brief Expects every element inside [low, high]. */
            template<typename T>
            void expect_all_inside(const std::vector<T>& values, double low, double high)
            {
                std::size_t outside = 0;
                for (const T& value : values)
                {
                    const double element = static_cast<double>(value);
                    outside += (element >= low && element <= high) ? 0 : 1;
                }

                EXPECT_EQ(outside, std::size_t{0})
                    << "elements lie outside [" << low << ", " << high << "]";
            }

            /** @brief Expects the Gram matrix of set's vectors to be diagonal times identity. */
            template<typename T>
            void expect_orthonormal(const std::vector<T>& matrix, const Kernels::VectorSet& set,
                                    double diagonal, double tolerance)
            {
                const auto element = [&matrix, &set](int64_t vector, int64_t k)
                {
                    const int64_t index = vector * set.vec_stride + k * set.elem_stride;
                    return static_cast<double>(matrix[static_cast<std::size_t>(index)]);
                };

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
        class Unit_RandomReference : public SushiBLASTest
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
                    return TestEnvironment::get_context().get_queue().get_device().has(aspect);
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
                                                         SushiRuntime::API::Residency::Device);
                        const sycl::context context =
                            TestEnvironment::get_context().get_queue().get_context();
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
                    engine->set_mode(mode);

                    Tensor t = engine->create_tensor(dims, get_sushiblas_dtype<T>());
                    SushiBLAS::IO io(*engine);
                    io.from_vector(t, initial);

                    RandomOps random(*engine);
                    random.set_seed(seed);
                    operation(random, t);
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
                        run_under(ExecutionMode::ForceSycl, seed, dims, initial, operation);
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
                        << "FORCE_REFERENCE left elements unwritten";
                    EXPECT_EQ(count_holding(draws.device, unwritten), std::size_t{0})
                        << "FORCE_SYCL left elements unwritten";
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
                        << "FORCE_REFERENCE left every element in place";
                    expect_equal_bits(draws.reference, draws.device);
                }

                /** @brief Expects the draws that call no math builtin to have equal bits of T. */
                template<typename T>
                void expect_exact_draws_agree()
                {
                    expect_same_shuffle<T>();

                    expect_same_bits<T>("discrete_uniform", K_SEED, [](RandomOps& random, Tensor& t)
                    {
                        random.discrete_uniform(t, K_LOWEST_INTEGER, K_HIGHEST_INTEGER);
                    });
                    expect_same_bits<T>("bernoulli", K_SEED, [](RandomOps& random, Tensor& t)
                    {
                        random.bernoulli(t, K_PROBABILITY);
                    });
                    expect_same_bits<T>("uniform", K_SEED, [](RandomOps& random, Tensor& t)
                    {
                        random.uniform(t, K_LOW, K_HIGH);
                    });
                    expect_same_bits<T>("he_uniform", K_SEED, [](RandomOps& random, Tensor& t)
                    {
                        random.he_uniform(t, K_HE_FAN_IN);
                    });
                    expect_same_bits<T>("xavier_uniform", K_SEED, [](RandomOps& random, Tensor& t)
                    {
                        random.xavier_uniform(t, K_XAVIER_FAN_IN, K_XAVIER_FAN_OUT);
                    });
                }

                /** @brief Expects poisson(lambda) to write equal bits of T on both paths. */
                template<typename T>
                void expect_same_poisson(double lambda, uint64_t seed)
                {
                    expect_same_bits<T>("poisson", seed, [lambda](RandomOps& random, Tensor& t)
                    {
                        random.poisson(t, lambda);
                    });
                }

                /** @brief Expects the draws built on a math builtin to agree in T within bounds. */
                template<typename T>
                void expect_normal_draws_agree()
                {
                    const Tolerance normal_tolerance{
                        box_muller_allowance<T>(K_NORMAL_MEAN, K_NORMAL_STDDEV), 0.0};
                    const Tolerance fan_tolerance{box_muller_allowance<T>(0.0, K_FAN_STDDEV), 0.0};
                    const Tolerance exponential_tolerance{0.0, builtin_allowance<T>()};

                    // The error of the normal draw passes through exp as a relative error.
                    const Tolerance log_normal_tolerance{
                        0.0, box_muller_allowance<T>(K_LOG_NORMAL_MEAN, K_LOG_NORMAL_STDDEV) +
                                 builtin_allowance<T>()};

                    expect_close<T>("normal", normal_tolerance, [](RandomOps& random, Tensor& t)
                    {
                        random.normal(t, K_NORMAL_MEAN, K_NORMAL_STDDEV);
                    });
                    expect_close<T>("he_normal", fan_tolerance, [](RandomOps& random, Tensor& t)
                    {
                        random.he_normal(t, K_HE_FAN_IN);
                    });
                    expect_close<T>("xavier_normal", fan_tolerance, [](RandomOps& random, Tensor& t)
                    {
                        random.xavier_normal(t, K_XAVIER_FAN_IN, K_XAVIER_FAN_OUT);
                    });
                    expect_close<T>("exponential", exponential_tolerance,
                                    [](RandomOps& random, Tensor& t)
                                    {
                                        random.exponential(t, K_EXPONENTIAL_RATE);
                                    });
                    expect_close<T>("log_normal", log_normal_tolerance,
                                    [](RandomOps& random, Tensor& t)
                                    {
                                        random.log_normal(t, K_LOG_NORMAL_MEAN,
                                                          K_LOG_NORMAL_STDDEV);
                                    });
                }

                /** @brief Expects truncated_normal to stay inside its cuts and agree in T. */
                template<typename T>
                void expect_truncated_normal_agrees()
                {
                    SCOPED_TRACE(trace_of<T>("truncated_normal"));

                    const ModeDraws<T> draws = fill_in_both_modes<T>(
                        K_SEED, {K_ELEMENTS}, [](RandomOps& random, Tensor& t)
                        {
                            random.truncated_normal(t, 0.0, 1.0, K_LOWER_CUT, K_UPPER_CUT);
                        });

                    expect_all_inside(draws.reference, K_LOWER_CUT, K_UPPER_CUT);
                    expect_all_inside(draws.device, K_LOWER_CUT, K_UPPER_CUT);
                    expect_within(draws.reference, draws.device,
                                  Tolerance{by_precision<T>(1e-3, 1e-9), 0.0});
                }

                /** @brief Expects both paths of orthogonal to be orthonormal and to agree in T. */
                template<typename T>
                void expect_orthogonal_agrees(const OrthogonalCase& matrix)
                {
                    SCOPED_TRACE(trace_of<T>(matrix.name));

                    const ModeDraws<T> draws = fill_in_both_modes<T>(
                        K_SEED, {matrix.rows, matrix.cols}, [](RandomOps& random, Tensor& t)
                        {
                            random.orthogonal(t, K_GAIN);
                        });

                    expect_orthonormal(draws.reference, matrix.vectors, K_GAIN * K_GAIN,
                                       K_GRAM_TOLERANCE);
                    expect_orthonormal(draws.device, matrix.vectors, K_GAIN * K_GAIN,
                                       K_GRAM_TOLERANCE);
                    expect_within(draws.reference, draws.device,
                                  Tolerance{by_precision<T>(1e-3, 1e-10) * K_GAIN, 0.0});
                }
        };

        /** @brief Tests that the draws that call no math builtin have equal bits on both paths. */
        TEST_F(Unit_RandomReference, IntegerAndUniformDrawsAreTheSameStream)
        {
            expect_exact_draws_agree<float>();
            expect_exact_draws_agree<double>();

            expect_same_bits<std::complex<float>>("uniform", K_SEED,
                                                  [](RandomOps& random, Tensor& t)
                                                  {
                                                      random.uniform(t, K_LOW, K_HIGH);
                                                  });

            expect_same_constant<float>();
            expect_same_constant<double>();
            expect_same_constant<std::complex<float>>();
            expect_same_constant<std::complex<double>>();
            expect_same_constant<int32_t>();
            expect_same_constant<int64_t>();
        }

        /** @brief Tests that constant and shuffle leave equal bits in HALF on both paths. */
        TEST_F(Unit_RandomReference, HalfValuesAreMovedToTheSameBits)
        {
            if (!device_has(sycl::aspect::fp16))
                GTEST_SKIP() << "the device does not report the fp16 aspect";

            expect_same_constant<sycl::half>();
            expect_same_shuffle<sycl::half>();
        }

        /** @brief Tests that poisson(4), drawn by Knuth's product, has equal bits on both paths. */
        TEST_F(Unit_RandomReference, SmallLambdaPoissonIsTheSameStream)
        {
            expect_same_poisson<float>(K_SMALL_LAMBDA, K_SEED);
            expect_same_poisson<double>(K_SMALL_LAMBDA, K_SEED);
        }

        /**
         * @brief Tests that poisson(50), drawn by PTRS, gives both paths the same counts.
         * @note A failure names an element whose acceptance test sits inside the difference between
         *       the device's log or lgamma and the host's. It does not show a wrong stream.
         */
        TEST_F(Unit_RandomReference, LargeLambdaPoissonGivesTheSameCounts)
        {
            expect_same_poisson<float>(K_LARGE_LAMBDA, K_LARGE_LAMBDA_SEED);
            expect_same_poisson<double>(K_LARGE_LAMBDA, K_LARGE_LAMBDA_SEED);
        }

        /** @brief Tests that the draws built on a math builtin agree within their bounds. */
        TEST_F(Unit_RandomReference, NormalDrawsAgreeWithinTheirBound)
        {
            expect_normal_draws_agree<float>();
            expect_normal_draws_agree<double>();
        }

        /**
         * @brief Tests that truncated_normal stays in [-2, 2] on both paths, which differ by at
         *        most 1e-3 in FLOAT32 and 1e-9 in FLOAT64.
         */
        TEST_F(Unit_RandomReference, TruncatedNormalAgreesInsideItsBounds)
        {
            expect_truncated_normal_agrees<float>();
            expect_truncated_normal_agrees<double>();
        }

        /**
         * @brief Tests that each path of orthogonal is orthonormal, and that the two differ by at
         *        most 1e-3 in FLOAT32 and 1e-10 in FLOAT64.
         */
        TEST_F(Unit_RandomReference, BothOrthogonalPathsAreOrthonormal)
        {
            for (const OrthogonalCase& matrix : K_ORTHOGONAL_CASES)
            {
                expect_orthogonal_agrees<float>(matrix);
                expect_orthogonal_agrees<double>(matrix);
            }
        }

        /** @brief Tests that orthogonal scales the one vector of a 1 by 4 matrix to the gain. */
        TEST_F(Unit_RandomReference, ASingleVectorIsScaledToTheGain)
        {
            const ModeDraws<float> draws =
                fill_in_both_modes<float>(K_SEED, {1, 4}, [](RandomOps& random, Tensor& t)
                {
                    random.orthogonal(t, K_SINGLE_VECTOR_GAIN);
                });

            EXPECT_NEAR(norm_of(draws.reference), K_SINGLE_VECTOR_GAIN, K_NORM_TOLERANCE)
                << "FORCE_REFERENCE";
            EXPECT_NEAR(norm_of(draws.device), K_SINGLE_VECTOR_GAIN, K_NORM_TOLERANCE)
                << "FORCE_SYCL";
        }

        /** @brief Tests that a call refused for a device-only tensor reserves no offset. */
        TEST_F(Unit_RandomReference, ARefusedCallReservesNoOffset)
        {
            std::optional<Tensor> values = device_only_tensor({4});
            std::optional<Tensor> matrix = device_only_tensor({3, 3});
            if (!values || !matrix)
                GTEST_SKIP() << "This device offers no device-only USM.";

            RandomOps random(*engine);
            random.set_seed(K_SEED);

            engine->set_mode(ExecutionMode::ForceReference);
            const uint64_t offset_before = engine->get_rng().reserve(0).offset;

            EXPECT_THROW(random.uniform(*values), std::invalid_argument);
            EXPECT_THROW(random.shuffle(*values), std::invalid_argument);
            EXPECT_THROW(random.orthogonal(*matrix), std::invalid_argument);
            EXPECT_EQ(offset_before, engine->get_rng().reserve(0).offset);

            // The same call reserves one offset once the mode lets it record.
            engine->set_mode(ExecutionMode::ForceSycl);
            random.uniform(*values);
            EXPECT_EQ(offset_before + 1, engine->get_rng().reserve(0).offset);
            engine->execute();
        }

        /** @brief Tests that two FORCE_REFERENCE runs of normal from one seed write equal bits. */
        TEST_F(Unit_RandomReference, TheSameSeedGivesTheSameReferenceStreamTwice)
        {
            const Operation fill = [](RandomOps& random, Tensor& t)
            {
                random.normal(t, K_NORMAL_MEAN, K_NORMAL_STDDEV);
            };
            const std::vector<float> unwritten(static_cast<std::size_t>(K_ELEMENTS), K_UNWRITTEN);

            const std::vector<float> first = run_under(ExecutionMode::ForceReference, K_SEED,
                                                       {K_ELEMENTS}, unwritten, fill);
            const std::vector<float> second = run_under(ExecutionMode::ForceReference, K_SEED,
                                                        {K_ELEMENTS}, unwritten, fill);

            EXPECT_EQ(count_holding(first, K_UNWRITTEN), std::size_t{0});
            expect_equal_bits(first, second);
        }
    } // namespace Test
} // namespace SushiBLAS
