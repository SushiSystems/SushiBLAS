/**************************************************************************/
/* test_scalar.cpp                                                        */
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
 * @file test_scalar.cpp
 * @brief Tests that ElementScalar holds one parameter in every element type, integers saturated.
 * @author Mustafa Garip
 */

#include <cmath>
#include <cstdint>
#include <limits>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <SushiBLAS/functor_map/scalar.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        TEST(Medium_ElementScalar, HoldsTheValueInEachFloatingType)
        {
            const Dispatch::ElementScalar scalar(1.5);

            EXPECT_EQ(scalar.value(), 1.5);
            EXPECT_EQ(scalar.as<double>(), 1.5);
            EXPECT_EQ(scalar.as<float>(), 1.5f);
            EXPECT_EQ(static_cast<float>(scalar.as<sycl::half>()), 1.5f);
        }

        TEST(Medium_ElementScalar, IntegersTruncateTowardZero)
        {
            EXPECT_EQ(Dispatch::ElementScalar(2.9).as<int32_t>(), 2);
            EXPECT_EQ(Dispatch::ElementScalar(-2.9).as<int64_t>(), int64_t{-2});
        }

        TEST(Medium_ElementScalar, IntegersSaturateAndANaNBecomesZero)
        {
            EXPECT_EQ(Dispatch::ElementScalar(1e20).as<int32_t>(),
                      std::numeric_limits<int32_t>::max());
            EXPECT_EQ(Dispatch::ElementScalar(-1e20).as<int64_t>(),
                      std::numeric_limits<int64_t>::min());
            EXPECT_EQ(Dispatch::ElementScalar(std::nan("")).as<int32_t>(), 0);
        }
    } // namespace Test
} // namespace SushiBLAS
