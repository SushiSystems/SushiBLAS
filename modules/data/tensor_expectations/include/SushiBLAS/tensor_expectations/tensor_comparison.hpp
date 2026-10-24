/**************************************************************************/
/* tensor_comparison.hpp                                                  */
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
 * @file tensor_comparison.hpp
 * @brief Declares verify_tensor and assert_tensor_near, which compare a dense tensor with host expectations.
 * @author Mustafa Garip
 * @see modules/data/tensor_expectations/README.md
 */

#pragma once

#include <cmath>
#include <vector>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/core/traits.hpp>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiBLAS/test_runner/test_environment.hpp>

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
            auto& q = TestEnvironment::queue();

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
                TestEnvironment::queue()
                    .memcpy(host.data(), tensor.data(), size * sizeof(H))
                    .wait();

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
                    case Core::DataType::UInt32:
                        compare_host_elements<uint32_t>(tensor, expected, tolerance);
                        break;
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
    } // namespace Test
} // namespace SushiBLAS

#define SB_ASSERT_TENSOR_NEAR(actual, expected, tol, type) \
    SushiBLAS::Test::assert_tensor_near<type>(actual, expected, tol)

#define EXPECT_TENSOR_NEAR(tensor, expected_list, tol) \
    SushiBLAS::Test::verify_tensor(tensor, expected_list, tol)

