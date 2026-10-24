/**************************************************************************/
/* tensor_fill.hpp                                                        */
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
 * @file tensor_fill.hpp
 * @brief Declares fill_tensor and fill_random, which write host values into a dense tensor.
 * @author Mustafa Garip
 * @see modules/data/tensor_expectations/README.md
 */

#pragma once

#include <random>
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
        /**
        * @brief Writes a flat host-side value list into a tensor, dispatching on the tensor's
        * runtime dtype (values are cast to the element type on write).
        */
        inline void fill_tensor(Tensor& tensor, const std::vector<float>& values)
        {
            size_t size = static_cast<size_t>(tensor.num_elements);
            ASSERT_EQ(values.size(), size) << "fill_tensor: value count does not match tensor size.";

            auto& q = TestEnvironment::queue();
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
                case Core::DataType::UInt32:
                {
                    std::vector<uint32_t> host(values.begin(), values.end());
                    q.memcpy(tensor.data(), host.data(), size * sizeof(uint32_t)).wait();
                    break;
                }
            }
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
         * @see modules/data/tensor_expectations/README.md
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

            auto& q = TestEnvironment::queue();
            q.memcpy(tensor.data(), host_data.data(), size * sizeof(T)).wait();
        }
    } // namespace Test
} // namespace SushiBLAS
