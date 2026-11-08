/**************************************************************************/
/* test_tensor_fill.cpp                                                   */
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
 * @file test_tensor_fill.cpp
 * @brief Tests that fill_tensor converts its values to each dtype and that fill_random repeats.
 * @author Mustafa Garip
 */

#include <complex>
#include <cstdint>
#include <random>
#include <type_traits>
#include <vector>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <SushiRuntime/sushiruntime.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/core/traits.hpp>
#include <SushiBLAS/tensor/tensor_factory.hpp>
#include <SushiBLAS/tensor_expectations/scalar_dtype.hpp>
#include <SushiBLAS/tensor_expectations/tensor_fill.hpp>
#include <SushiBLAS/tensor_expectations/tensor_readback.hpp>
#include <SushiBLAS/typed_suite_naming/default_type_names.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /// @brief Returns a rank-one tensor of host memory with four elements of dtype.
        static Tensor four_of(Core::DataType dtype)
        {
            const TensorFactory factory(TensorFactory::Host{});
            const int64_t dims[1] = {4};
            return factory.create(SushiRuntime::span<const int64_t>(dims, 1), dtype);
        }

        /// @brief Returns the four values the conversion tests fill, all non-negative.
        static std::vector<float> filled_values()
        {
            return {1.5f, 2.0f, 0.25f, 7.0f};
        }

        /// @brief Returns filled_values() as T holds them: an integer drops the fraction.
        template<typename T>
        static std::vector<T> filled_as()
        {
            if constexpr (std::is_integral_v<T>)
                return {T(1), T(2), T(0), T(7)};
            else if constexpr (Core::is_complex_v<T>)
                return {T(1.5, 0.0), T(2.0, 0.0), T(0.25, 0.0), T(7.0, 0.0)};
            else
                return {T(1.5f), T(2.0f), T(0.25f), T(7.0f)};
        }

        template<typename T>
        class Small_TensorFill : public ::testing::Test
        {
        };

        using FillTypes = ::testing::Types<sycl::half, float, double, std::complex<float>,
                                           std::complex<double>, int32_t, int64_t, uint32_t>;
        SB_TYPED_TEST_SUITE(Small_TensorFill, FillTypes);

        TYPED_TEST(Small_TensorFill, ConvertsTheValuesToTheTensorsDtype)
        {
            Tensor tensor = four_of(sushiblas_dtype<TypeParam>());

            fill_tensor(tensor, filled_values());

            EXPECT_EQ(read_tensor<TypeParam>(tensor), filled_as<TypeParam>());
        }

        /// @brief Expects every value to lie in [min_val, max_val).
        static void expect_in_half_open_range(const std::vector<float>& values, float min_val,
                                              float max_val)
        {
            for (float value : values)
            {
                EXPECT_GE(value, min_val);
                EXPECT_LT(value, max_val);
            }
        }

        /// @brief Returns four floats drawn in [2, 3) from a generator seeded with seed.
        static std::vector<float> drawn_from(std::uint32_t seed)
        {
            std::mt19937 generator(seed);
            Tensor tensor = four_of(Core::DataType::Float32);
            fill_random<float>(tensor, generator, 2.0f, 3.0f);
            return read_tensor<float>(tensor);
        }

        /// @brief The seed of the fill_random tests.
        constexpr std::uint32_t K_SEED = 20261006u;

        TEST(Small_TensorFillRandom, StaysInTheRangeItIsGiven)
        {
            expect_in_half_open_range(drawn_from(K_SEED), 2.0f, 3.0f);
        }

        TEST(Small_TensorFillRandom, RepeatsFromTheSameSeed)
        {
            EXPECT_EQ(drawn_from(K_SEED), drawn_from(K_SEED));
        }
    } // namespace Test
} // namespace SushiBLAS
