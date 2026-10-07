/**************************************************************************/
/* test_utils.hpp                                                         */
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
 * @file test_utils.hpp
 * @brief Declares the tensor fill and comparison helpers, the base fixture and the typed-suite macro.
 * @author Mustafa Garip
 * @see tests/common/README.md
 */

#pragma once

#include <string>
#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <complex>
#include <algorithm>
#include <memory>
#include <random>
#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include <SushiBLAS/SushiBLAS.h>
#include <SushiBLAS/core/traits.hpp>
#include "test_environment.hpp"

namespace SushiBLAS
{
    namespace Test
    {
        // The complex-type trait used here is Core::is_complex_v from <SushiBLAS/core/traits.hpp>.

        /**
        * @brief Comparison utility for SushiBLAS Tensors.
        * 
        * Automatically handles USM memory types, data types, and synchronization.
        */
        template <typename T>
        void assert_tensor_near(const Tensor& actual, const Tensor& expected, double tolerance = 1e-4)
        {
            ASSERT_EQ(actual.dtype, expected.dtype) << "Tensor data type mismatch.";
            ASSERT_EQ(actual.shape, expected.shape) << "Tensor shape mismatch.";
            
            size_t size = static_cast<size_t>(actual.num_elements);

            std::vector<T> host_actual(size);
            std::vector<T> host_expected(size);

            // Fetch runtime queue for synchronization and copy
            auto& q = TestEnvironment::get_context().get_queue();

            q.memcpy(host_actual.data(), actual.data(), size * sizeof(T)).wait();
            q.memcpy(host_expected.data(), expected.data(), size * sizeof(T)).wait();

            for (size_t i = 0; i < size; ++i)
            {
                if constexpr (std::is_same_v<T, sycl::half>)
                {
                    float a = static_cast<float>(host_actual[i]);
                    float e = static_cast<float>(host_expected[i]);
                    EXPECT_NEAR(a, e, tolerance) << "Mismatch at index " << i;
                }
                else if constexpr (Core::is_complex_v<T>)
                {
                    auto a = host_actual[i];
                    auto e = host_expected[i];
                    EXPECT_NEAR(a.real(), e.real(), tolerance) << "Real part mismatch at index " << i;
                    EXPECT_NEAR(a.imag(), e.imag(), tolerance) << "Imaginary part mismatch at index " << i;
                }
                else
                {
                    double a = static_cast<double>(host_actual[i]);
                    double e = static_cast<double>(host_expected[i]);
                    EXPECT_NEAR(a, e, tolerance) << "Mismatch at index " << i;
                }
            }
        }

        /**
        * @brief Maps a C++ scalar type to its corresponding SushiBLAS DataType enum value.
        */
        template<typename T> Core::DataType get_sushiblas_dtype();
        template<> inline Core::DataType get_sushiblas_dtype<sycl::half>() { return Core::DataType::Half; }
        template<> inline Core::DataType get_sushiblas_dtype<float>() { return Core::DataType::Float32; }
        template<> inline Core::DataType get_sushiblas_dtype<double>() { return Core::DataType::Float64; }
        template<> inline Core::DataType get_sushiblas_dtype<std::complex<float>>() { return Core::DataType::Complex32; }
        template<> inline Core::DataType get_sushiblas_dtype<std::complex<double>>() { return Core::DataType::Complex64; }
        template<> inline Core::DataType get_sushiblas_dtype<int32_t>() { return Core::DataType::Int32; }
        template<> inline Core::DataType get_sushiblas_dtype<int64_t>() { return Core::DataType::Int64; }

        /**
        * @brief Writes a flat host-side value list into a tensor, dispatching on the tensor's
        * runtime dtype (values are cast to the element type on write).
        */
        inline void fill_tensor(Tensor& tensor, const std::vector<float>& values)
        {
            size_t size = static_cast<size_t>(tensor.num_elements);
            ASSERT_EQ(values.size(), size) << "fill_tensor: value count does not match tensor size.";

            auto& q = TestEnvironment::get_context().get_queue();
            switch (tensor.dtype)
            {
                case Core::DataType::Half:
                {
                    std::vector<sycl::half> host(values.begin(), values.end());
                    q.memcpy(tensor.data(), host.data(), size * sizeof(sycl::half)).wait();
                    break;
                }
                case Core::DataType::Float32:
                {
                    std::vector<float> host(values.begin(), values.end());
                    q.memcpy(tensor.data(), host.data(), size * sizeof(float)).wait();
                    break;
                }
                case Core::DataType::Float64:
                {
                    std::vector<double> host(values.begin(), values.end());
                    q.memcpy(tensor.data(), host.data(), size * sizeof(double)).wait();
                    break;
                }
                case Core::DataType::Complex32:
                {
                    std::vector<std::complex<float>> host(values.begin(), values.end());
                    q.memcpy(tensor.data(), host.data(), size * sizeof(std::complex<float>)).wait();
                    break;
                }
                case Core::DataType::Complex64:
                {
                    std::vector<std::complex<double>> host(values.begin(), values.end());
                    q.memcpy(tensor.data(), host.data(), size * sizeof(std::complex<double>)).wait();
                    break;
                }
                // The integer arms truncate the given values, as every other arm casts them.
                case Core::DataType::Int32:
                {
                    std::vector<int32_t> host(values.begin(), values.end());
                    q.memcpy(tensor.data(), host.data(), size * sizeof(int32_t)).wait();
                    break;
                }
                case Core::DataType::Int64:
                {
                    std::vector<int64_t> host(values.begin(), values.end());
                    q.memcpy(tensor.data(), host.data(), size * sizeof(int64_t)).wait();
                    break;
                }
            }
        }

        namespace Detail
        {
            /**
             * @brief Expects one scalar to match its expectation, treating NaN and infinity exactly.
             * @param expected A NaN expects a NaN; an infinity expects the same infinity.
             * @param tolerance Absolute tolerance for a finite expectation.
             * @param part Names the compared part in the failure message.
             */
            inline void expect_scalar_near(double actual, double expected, double tolerance,
                                           size_t index, const char* part)
            {
                if (std::isnan(expected))
                    EXPECT_TRUE(std::isnan(actual)) << "Expected NaN (" << part << ") at index " << index << ", got " << actual;
                else if (std::isinf(expected))
                    EXPECT_EQ(actual, expected) << "Infinity mismatch (" << part << ") at index " << index;
                else
                    EXPECT_NEAR(actual, expected, tolerance) << "Mismatch (" << part << ") at index " << index;
            }

            /**
             * @brief Widens a real or complex expectation to std::complex<double>.
             * @tparam E float, double, std::complex<float> or std::complex<double>.
             */
            template <typename E>
            std::complex<double> to_complex_double(const E& value)
            {
                if constexpr (Core::is_complex_v<E>)
                    return std::complex<double>(static_cast<double>(value.real()), static_cast<double>(value.imag()));
                else
                    return std::complex<double>(static_cast<double>(value), 0.0);
            }

            /**
             * @brief Widens a real host element to double.
             * @tparam H A real tensor scalar type, sycl::half included.
             */
            template <typename H>
            double to_double(const H& value)
            {
                if constexpr (std::is_same_v<H, sycl::half>)
                    return static_cast<double>(static_cast<float>(value));
                else
                    return static_cast<double>(value);
            }

            /**
             * @brief Copies a dense tensor of element type H to the host and compares every element.
             * @tparam H The tensor's element type.
             * @tparam E The expectation's element type; a real E expects a zero imaginary part.
             */
            template <typename H, typename E>
            void compare_host_elements(const Tensor& tensor, const std::vector<E>& expected, double tolerance)
            {
                const size_t size = static_cast<size_t>(tensor.num_elements);
                std::vector<H> host(size);
                TestEnvironment::get_context().get_queue().memcpy(host.data(), tensor.data(), size * sizeof(H)).wait();

                for (size_t i = 0; i < size; ++i)
                {
                    const std::complex<double> e = to_complex_double(expected[i]);
                    if constexpr (Core::is_complex_v<H>)
                    {
                        expect_scalar_near(static_cast<double>(host[i].real()), e.real(), tolerance, i, "real");
                        expect_scalar_near(static_cast<double>(host[i].imag()), e.imag(), tolerance, i, "imaginary");
                    }
                    else if constexpr (std::is_integral_v<H>)
                        EXPECT_EQ(host[i], static_cast<H>(e.real())) << "Mismatch at index " << i;
                    else
                        expect_scalar_near(to_double(host[i]), e.real(), tolerance, i, "value");
                }
            }

            /**
             * @brief Compares a dense tensor against expectations, dispatching on its runtime dtype.
             * @tparam E float, double, std::complex<float> or std::complex<double>.
             */
            template <typename E>
            void verify_tensor_elements(const Tensor& tensor, const std::vector<E>& expected, double tolerance)
            {
                ASSERT_EQ(expected.size(), static_cast<size_t>(tensor.num_elements))
                    << "verify_tensor: expected value count does not match tensor size.";

                switch (tensor.dtype)
                {
                    case Core::DataType::Half:      compare_host_elements<sycl::half>(tensor, expected, tolerance); break;
                    case Core::DataType::Float32:   compare_host_elements<float>(tensor, expected, tolerance); break;
                    case Core::DataType::Float64:   compare_host_elements<double>(tensor, expected, tolerance); break;
                    case Core::DataType::Complex32: compare_host_elements<std::complex<float>>(tensor, expected, tolerance); break;
                    case Core::DataType::Complex64: compare_host_elements<std::complex<double>>(tensor, expected, tolerance); break;
                    case Core::DataType::Int32:     compare_host_elements<int32_t>(tensor, expected, tolerance); break;
                    case Core::DataType::Int64:     compare_host_elements<int64_t>(tensor, expected, tolerance); break;
                }
            }
        } // namespace Detail

        /**
        * @brief Compares a dense tensor against a flat host-side list of real expectations.
        *
        * A complex element must match on its real part and have a zero imaginary part. An integer
        * element is compared exactly. A NaN expectation expects a NaN.
        */
        inline void verify_tensor(const Tensor& tensor, const std::vector<float>& expected, double tolerance = 1e-4)
        {
            Detail::verify_tensor_elements(tensor, expected, tolerance);
        }

        /**
        * @brief Compares a dense tensor against a flat host-side list of double or complex expectations.
        * @tparam E double, std::complex<float> or std::complex<double>; a complex E checks both parts.
        */
        template <typename E>
        void verify_tensor(const Tensor& tensor, const std::vector<E>& expected, double tolerance = 1e-4)
        {
            Detail::verify_tensor_elements(tensor, expected, tolerance);
        }

        /**
         * @brief Copies a dense tensor's elements into a host vector.
         * @tparam T The tensor's element type.
         * @throws std::invalid_argument when T does not map to the tensor's dtype.
         */
        template <typename T>
        std::vector<T> read_tensor(const Tensor& tensor)
        {
            if (get_sushiblas_dtype<T>() != tensor.dtype)
                throw std::invalid_argument("read_tensor: T does not match the tensor's dtype " +
                                            std::string(Core::to_string(tensor.dtype)) + ".");

            const size_t size = static_cast<size_t>(tensor.num_elements);
            std::vector<T> host(size);
            TestEnvironment::get_context().get_queue()
                .memcpy(host.data(), tensor.data(), size * sizeof(T))
                .wait();
            return host;
        }

        /// @brief Holds the seed fill_random's generator starts from.
        constexpr std::uint32_t K_FILL_RANDOM_SEED = 20261006u;

        /**
         * @brief Returns the generator fill_random draws from.
         */
        inline std::mt19937& fill_random_generator()
        {
            static std::mt19937 generator(K_FILL_RANDOM_SEED);
            return generator;
        }

        /**
         * @brief Restarts fill_random's generator from K_FILL_RANDOM_SEED.
         */
        inline void reset_fill_random_generator()
        {
            fill_random_generator().seed(K_FILL_RANDOM_SEED);
        }

        /**
         * @brief Fills a dense tensor with uniform values drawn from fill_random_generator().
         * @see tests/common/README.md
         */
        template <typename T>
        void fill_random(Tensor& tensor, float min_val = -1.0f, float max_val = 1.0f)
        {
            size_t size = static_cast<size_t>(tensor.num_elements);

            std::vector<T> host_data(size);
            std::uniform_real_distribution<float> distribution(min_val, max_val);
            for (size_t i = 0; i < size; ++i)
            {
                auto get_rand = [&]()
                {
                    return distribution(fill_random_generator());
                };

                if constexpr (std::is_same_v<T, sycl::half>)
                    host_data[i] = sycl::vec<float, 1>(get_rand())
                                       .convert<sycl::half, sycl::rounding_mode::automatic>()[0];
                else if constexpr (Core::is_complex_v<T>)
                {
                    using BaseT = typename T::value_type;
                    const BaseT real_part = static_cast<BaseT>(get_rand());
                    const BaseT imag_part = static_cast<BaseT>(get_rand());
                    host_data[i] = T(real_part, imag_part);
                }
                else
                    host_data[i] = static_cast<T>(get_rand());
            }

            auto& q = TestEnvironment::get_context().get_queue();
            q.memcpy(tensor.data(), host_data.data(), size * sizeof(T)).wait();
        }

        /**
        * @brief Universal test fixture for SushiBLAS tests.
        */
        class SushiBLASTest : public ::testing::Test 
        {
            protected:
                std::unique_ptr<Engine> engine;

                void SetUp() override 
                {
                    engine = std::make_unique<Engine>(TestEnvironment::get_context(), Core::Layout::RowMajor);
                    reset_fill_random_generator();
                }

                void TearDown() override 
                {
                    engine.reset();
                }

                void reinit_engine(Core::Layout layout)
                {
                    engine = std::make_unique<Engine>(TestEnvironment::get_context(), layout);
                }

                /**
                * @brief Creates a tensor of dtype `T` and initializes it from a host-side vector.
                */
                template<typename T>
                Tensor create_initialized_tensor(std::initializer_list<int64_t> dims, const std::vector<T>& data)
                {
                    auto t = engine->create_tensor(dims, get_sushiblas_dtype<T>());
                    auto& q = TestEnvironment::get_context().get_queue();
                    q.memcpy(t.data(), data.data(), data.size() * sizeof(T)).wait();
                    return t;
                }
        };

        /**
         * @brief Names each type parameter by its index, as GoogleTest does by default.
         * @see tests/common/README.md
         */
        struct DefaultTypeNames
        {
            /**
             * @brief The suffix for the @p i-th type parameter.
             * @tparam T The type parameter; unused, as in GoogleTest's own default.
             * @param i The type's index within the suite's type list.
             * @return @p i rendered as a decimal string.
             */
            template<typename T>
            static std::string GetName(int i) { return std::to_string(i); }
        };

    } //namespace Test
} // namespace SushiBLAS

/**
 * @brief `TYPED_TEST_SUITE` with the name generator spelled out.
 *
 * See @ref SushiBLAS::Test::DefaultTypeNames for why the two-argument form cannot be
 * written directly on a C++17 baseline. Behaviour is identical.
 */
#define SB_TYPED_TEST_SUITE(SuiteName, Types) \
    TYPED_TEST_SUITE(SuiteName, Types, SushiBLAS::Test::DefaultTypeNames)

#define SB_ASSERT_TENSOR_NEAR(actual, expected, tol, type) \
    SushiBLAS::Test::assert_tensor_near<type>(actual, expected, tol)

#define EXPECT_TENSOR_NEAR(tensor, expected_list, tol) \
    SushiBLAS::Test::verify_tensor(tensor, expected_list, tol)
