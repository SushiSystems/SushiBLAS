/****************************************************************/
/* test_utils.hpp                                               */
/* SushiBLAS - https://github.com/SushiSystems/SushiBLAS        */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

#pragma once

#include <string>
#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <complex>
#include <algorithm>
#include <memory>
#include <SushiBLAS/SushiBLAS.h>
#include <SushiBLAS/core/traits.hpp>
#include "test_environment.hpp"

namespace SushiBLAS
{
    namespace Test
    {
        // The complex-type trait these helpers need is the library's own,
        // SushiBLAS::Core::is_complex_v (<SushiBLAS/core/traits.hpp>). This file used to
        // carry a fourth copy of it; there is exactly one in the project now.

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
        template<> inline Core::DataType get_sushiblas_dtype<sycl::half>() { return Core::DataType::HALF; }
        template<> inline Core::DataType get_sushiblas_dtype<float>() { return Core::DataType::FLOAT32; }
        template<> inline Core::DataType get_sushiblas_dtype<double>() { return Core::DataType::FLOAT64; }
        template<> inline Core::DataType get_sushiblas_dtype<std::complex<float>>() { return Core::DataType::COMPLEX32; }
        template<> inline Core::DataType get_sushiblas_dtype<std::complex<double>>() { return Core::DataType::COMPLEX64; }
        template<> inline Core::DataType get_sushiblas_dtype<int32_t>() { return Core::DataType::INT32; }
        template<> inline Core::DataType get_sushiblas_dtype<int64_t>() { return Core::DataType::INT64; }

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
                case Core::DataType::HALF:
                {
                    std::vector<sycl::half> host(values.begin(), values.end());
                    q.memcpy(tensor.data(), host.data(), size * sizeof(sycl::half)).wait();
                    break;
                }
                case Core::DataType::FLOAT32:
                {
                    std::vector<float> host(values.begin(), values.end());
                    q.memcpy(tensor.data(), host.data(), size * sizeof(float)).wait();
                    break;
                }
                case Core::DataType::FLOAT64:
                {
                    std::vector<double> host(values.begin(), values.end());
                    q.memcpy(tensor.data(), host.data(), size * sizeof(double)).wait();
                    break;
                }
                case Core::DataType::COMPLEX32:
                {
                    std::vector<std::complex<float>> host(values.begin(), values.end());
                    q.memcpy(tensor.data(), host.data(), size * sizeof(std::complex<float>)).wait();
                    break;
                }
                case Core::DataType::COMPLEX64:
                {
                    std::vector<std::complex<double>> host(values.begin(), values.end());
                    q.memcpy(tensor.data(), host.data(), size * sizeof(std::complex<double>)).wait();
                    break;
                }
                // Truncating, like every other arm here: the caller states the values it
                // wants and the tensor's dtype decides how they are stored. An integer
                // tensor filled from {1.0f, 0.0f} is the natural way to write a comparison
                // operand or an expected truth vector.
                case Core::DataType::INT32:
                {
                    std::vector<int32_t> host(values.begin(), values.end());
                    q.memcpy(tensor.data(), host.data(), size * sizeof(int32_t)).wait();
                    break;
                }
                case Core::DataType::INT64:
                {
                    std::vector<int64_t> host(values.begin(), values.end());
                    q.memcpy(tensor.data(), host.data(), size * sizeof(int64_t)).wait();
                    break;
                }
            }
        }

        /**
        * @brief Compares a tensor's contents against a flat host-side expected value list,
        * dispatching on the tensor's runtime dtype. For complex dtypes, only the real part is
        * checked (matching this codebase's existing test convention for real-valued expectations).
        */
        inline void verify_tensor(const Tensor& tensor, const std::vector<float>& expected, double tolerance = 1e-4)
        {
            size_t size = static_cast<size_t>(tensor.num_elements);
            ASSERT_EQ(expected.size(), size) << "verify_tensor: expected value count does not match tensor size.";

            auto& q = TestEnvironment::get_context().get_queue();
            std::vector<float> exp(expected.begin(), expected.end());

            switch (tensor.dtype)
            {
                case Core::DataType::HALF:
                {
                    std::vector<sycl::half> host(size);
                    q.memcpy(host.data(), tensor.data(), size * sizeof(sycl::half)).wait();
                    for (size_t i = 0; i < size; ++i) EXPECT_NEAR(static_cast<float>(host[i]), exp[i], tolerance) << "Mismatch at index " << i;
                    break;
                }
                case Core::DataType::FLOAT32:
                {
                    std::vector<float> host(size);
                    q.memcpy(host.data(), tensor.data(), size * sizeof(float)).wait();
                    for (size_t i = 0; i < size; ++i) EXPECT_NEAR(host[i], exp[i], tolerance) << "Mismatch at index " << i;
                    break;
                }
                case Core::DataType::FLOAT64:
                {
                    std::vector<double> host(size);
                    q.memcpy(host.data(), tensor.data(), size * sizeof(double)).wait();
                    for (size_t i = 0; i < size; ++i) EXPECT_NEAR(host[i], exp[i], tolerance) << "Mismatch at index " << i;
                    break;
                }
                case Core::DataType::COMPLEX32:
                {
                    std::vector<std::complex<float>> host(size);
                    q.memcpy(host.data(), tensor.data(), size * sizeof(std::complex<float>)).wait();
                    for (size_t i = 0; i < size; ++i) EXPECT_NEAR(host[i].real(), exp[i], tolerance) << "Mismatch (real) at index " << i;
                    break;
                }
                case Core::DataType::COMPLEX64:
                {
                    std::vector<std::complex<double>> host(size);
                    q.memcpy(host.data(), tensor.data(), size * sizeof(std::complex<double>)).wait();
                    for (size_t i = 0; i < size; ++i) EXPECT_NEAR(host[i].real(), exp[i], tolerance) << "Mismatch (real) at index " << i;
                    break;
                }
                // Exact, not near: an integer element either is the expected value or it is
                // not, and a tolerance would only hide an off-by-one.
                case Core::DataType::INT32:
                {
                    std::vector<int32_t> host(size);
                    q.memcpy(host.data(), tensor.data(), size * sizeof(int32_t)).wait();
                    for (size_t i = 0; i < size; ++i) EXPECT_EQ(host[i], static_cast<int32_t>(exp[i])) << "Mismatch at index " << i;
                    break;
                }
                case Core::DataType::INT64:
                {
                    std::vector<int64_t> host(size);
                    q.memcpy(host.data(), tensor.data(), size * sizeof(int64_t)).wait();
                    for (size_t i = 0; i < size; ++i) EXPECT_EQ(host[i], static_cast<int64_t>(exp[i])) << "Mismatch at index " << i;
                    break;
                }
            }
        }

        /**
        * @brief Random data generator for testing.
        */
        template <typename T>
        void fill_random(Tensor& tensor, float min_val = -1.0f, float max_val = 1.0f)
        {
            size_t size = static_cast<size_t>(tensor.num_elements);

            std::vector<T> host_data(size);
            for (size_t i = 0; i < size; ++i)
            {
                auto get_rand = [&]() {
                    float r = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
                    return min_val + r * (max_val - min_val);
                };

                if constexpr (std::is_same_v<T, sycl::half>)
                    host_data[i] = sycl::vec<float, 1>(get_rand()).convert<sycl::half, sycl::rounding_mode::automatic>()[0];
                else if constexpr (Core::is_complex_v<T>)
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
         * @brief The type-parameter naming GoogleTest applies when it is given none.
         *
         * `TYPED_TEST_SUITE(Suite, Types)` looks like a two-argument macro but is declared
         * `TYPED_TEST_SUITE(CaseName, Types, ...)`, the `...` being an optional name
         * generator. Passing no argument at all for a `...` is allowed from C++20 and is a
         * diagnosed extension before it, so on this project's C++17 baseline the
         * two-argument spelling is an error under `-pedantic -Werror` - raised inside a
         * third-party header this project does not own and cannot fix.
         *
         * Naming a generator explicitly settles it, and this is the default one restated
         * against GoogleTest's public API rather than reached for in its `internal`
         * namespace: the suffix is the type's index, exactly as before, so every generated
         * test name is unchanged and no test is renamed by the fix.
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
