/**************************************************************************/
/* test_scalar_binding.cpp                                                */
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
 * @file test_scalar_binding.cpp
 * @brief Tests that bound scalars reach the functor after its elements, narrowed to their type.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <gtest/gtest.h>
#include <SushiBLAS/engine/dispatch.hpp>
#include <SushiBLAS/functor_map/scalar.hpp>
#include <SushiBLAS/functor_map/scalar_binding.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /** @brief Returns x * a + b, the element first and the scalars after it. */
        struct ScaleShift
        {
            /** @brief Applies the scale and the shift to one element. */
            template<typename T>
            T operator()(T x, T a, T b) const
            {
                return x * a + b;
            }
        };

        TEST(Medium_ScalarBinding, AppendsTheScalarsInTheElementType)
        {
            const auto bound = Dispatch::bind_scalars(ScaleShift{}, Dispatch::ElementScalar(2.5),
                                                      Dispatch::ElementScalar(1.0));

            EXPECT_EQ(bound.narrow<double>()(4.0), 11.0);
            EXPECT_EQ(bound.narrow<float>()(2.0f), 6.0f);
            EXPECT_EQ(bound.narrow<int32_t>()(3), 7);
        }

        TEST(Medium_ScalarBinding, NarrowForPicksTheBoundFormOrPassesAPlainFunctor)
        {
            const auto bound = Dispatch::bind_scalars(ScaleShift{}, Dispatch::ElementScalar(3.0),
                                                      Dispatch::ElementScalar(-1.0));
            const auto plain = [](double x) { return x + 1.0; };

            EXPECT_EQ(Dispatch::Detail::narrow_for(Dispatch::ElementTag<double>{}, bound)(2.0),
                      5.0);
            EXPECT_EQ(Dispatch::Detail::narrow_for(Dispatch::ElementTag<double>{}, plain)(2.0),
                      3.0);
        }
    } // namespace Test
} // namespace SushiBLAS
