/**************************************************************************/
/* test_ordered_sum_element.cpp                                           */
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
 * @file test_ordered_sum_element.cpp
 * @brief Tests the ordered_sum element functor: the type of its weights and the bits it writes.
 * @author Mustafa Garip
 */

#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>
#include <vector>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <SushiBLAS/elementwise/ordered_sum_element.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            /// @brief A weight no float holds exactly, so the product rounds.
            constexpr double K_FIRST_WEIGHT = 0.37;

            /// @brief A second weight no float holds exactly.
            constexpr double K_SECOND_WEIGHT = -1.9;
        } // namespace

        /** @brief Returns the type the kernel of element type T holds its weights in. */
        template<typename T>
        using KernelWeight =
            typename decltype(Dispatch::Detail::OrderedSumElement<T>::weights)::value_type;

        /**
         * @brief Tests that the kernel holds no double for a type that is not FLOAT64 or
         *        COMPLEX64, so a device without fp64 runs a FLOAT32, HALF or COMPLEX32 sum.
         */
        TEST(Small_Regression_OrderedSum, KernelHoldsNoDoubleOutsideTheDoubleTypes)
        {
            EXPECT_TRUE((std::is_same_v<KernelWeight<sycl::half>, float>));
            EXPECT_TRUE((std::is_same_v<KernelWeight<float>, float>));
            EXPECT_TRUE((std::is_same_v<KernelWeight<std::complex<float>>, float>));
            EXPECT_TRUE((std::is_same_v<KernelWeight<std::int32_t>, std::int32_t>));
            EXPECT_TRUE((std::is_same_v<KernelWeight<std::int64_t>, std::int64_t>));
            EXPECT_TRUE((std::is_same_v<KernelWeight<double>, double>));
            EXPECT_TRUE((std::is_same_v<KernelWeight<std::complex<double>>, double>));
        }

        /**
         * @brief Tests that a weight rounded to float once on the host gives the bits the
         *        kernel gave when it rounded the same double to float before each multiply.
         */
        TEST(Small_OrderedSumElement, AWeightRoundedOnTheHostKeepsTheSumsBits)
        {
            #pragma clang fp contract(off)
            const std::vector<float> first = {1.25f, -3.5f, 97.03f};
            const std::vector<float> second = {0.1f, 2.0e7f, -0.333f};
            std::vector<float> got(first.size(), 0.0f);
            Dispatch::Detail::OrderedSumElement<float> element{};
            element.inputs[0] = first.data();
            element.inputs[1] = second.data();
            element.weights[0] = static_cast<float>(K_FIRST_WEIGHT);
            element.weights[1] = static_cast<float>(K_SECOND_WEIGHT);
            element.output = got.data();
            element.input_count = 2;

            for (std::size_t i = 0; i < first.size(); ++i)
            {
                element(i);
                float expected = 0.0f;
                expected += static_cast<float>(K_FIRST_WEIGHT) * first[i];
                expected += static_cast<float>(K_SECOND_WEIGHT) * second[i];
                EXPECT_EQ(std::memcmp(&expected, &got[i], sizeof(float)), 0) << "element " << i;
            }
        }
    } // namespace Test
} // namespace SushiBLAS
