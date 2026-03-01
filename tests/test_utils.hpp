/**************************************************************************/
/* test_utils.hpp                                                         */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                SushiBLAS                               */
/**************************************************************************/

#pragma once

#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <complex>
#include <algorithm>
#include <memory>
#include <SushiBLAS/SushiBLAS.h>
#include "test_environment.hpp"

namespace SushiBLAS
{
    namespace Test
    {
        /**
        * @brief Helper trait to check for complex types.
        */
        template<typename T> struct is_complex : std::false_type {};
        template<typename T> struct is_complex<std::complex<T>> : std::true_type {};
        template<typename T> inline constexpr bool is_complex_v = is_complex<T>::value;

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
            
            size_t size = 1;
            for (auto s : actual.shape) size *= s;

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
                else if constexpr (is_complex_v<T>)
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
        * @brief Random data generator for testing.
        */
        template <typename T>
        void fill_random(Tensor& tensor, float min_val = -1.0f, float max_val = 1.0f)
        {
            size_t size = 1;
            for (auto s : tensor.shape) size *= s;
            
            std::vector<T> host_data(size);
            for (size_t i = 0; i < size; ++i)
            {
                auto get_rand = [&]() {
                    float r = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
                    return min_val + r * (max_val - min_val);
                };

                if constexpr (std::is_same_v<T, sycl::half>)
                    host_data[i] = sycl::vec<float, 1>(get_rand()).convert<sycl::half, sycl::rounding_mode::automatic>()[0];
                else if constexpr (is_complex_v<T>)
                {
                    using BaseT = typename T::value_type;
                    host_data[i] = T(static_cast<BaseT>(get_rand()), static_cast<BaseT>(get_rand()));
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
                    engine = std::make_unique<Engine>(TestEnvironment::get_context(), Core::Layout::ROW_MAJOR);
                }

                void TearDown() override 
                {
                    engine.reset();
                }

                void reinit_engine(Core::Layout layout) 
                {
                    engine = std::make_unique<Engine>(TestEnvironment::get_context(), layout);
                }
        };

    } //namespace Test
} // namespace SushiBLAS

#define SB_ASSERT_TENSOR_NEAR(actual, expected, tol, type) \
    SushiBLAS::Test::assert_tensor_near<type>(actual, expected, tol)
