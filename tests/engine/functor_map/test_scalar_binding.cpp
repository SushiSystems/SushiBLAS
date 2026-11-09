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
 * @file test_scalar_binding.cpp
 * @brief Tests that bound scalars reach the functor after its elements, narrowed to their type.
 * @author Mustafa Garip
 */

#include <string>
#include <vector>
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

        namespace
        {
            /** @brief The functor of the narrowing rows with a scale of 2.5 and a shift of 1.0 bound. */
            using BoundScaleShift = Dispatch::ScalarBound<ScaleShift, 2>;

            /** @brief Names one narrowing of the bound scale and shift to an element type. */
            struct NarrowingRow
            {
                const char* name;                         /**< The row name a failure shows. */
                double (*apply)(const BoundScaleShift&);  /**< Narrows to the type and applies the functor. */
                double expected;                          /**< The element the functor returns. */
            };

            /** @brief Returns one row per element type. */
            std::vector<NarrowingRow> narrowing_rows()
            {
                return {{"Double",
                         [](const BoundScaleShift& bound) { return bound.narrow<double>()(4.0); },
                         11.0},
                        {"Float",
                         [](const BoundScaleShift& bound)
                         { return static_cast<double>(bound.narrow<float>()(2.0f)); },
                         6.0},
                        {"Int32",
                         [](const BoundScaleShift& bound)
                         { return static_cast<double>(bound.narrow<std::int32_t>()(3)); },
                         7.0}};
            }
        } // namespace

        /** @brief Runs one row of narrowing_rows(). */
        class Small_ScalarBinding : public ::testing::TestWithParam<NarrowingRow>
        {
        };

        TEST_P(Small_ScalarBinding, AppendsTheScalarsInTheElementType)
        {
            const BoundScaleShift bound = Dispatch::bind_scalars(
                ScaleShift{}, Dispatch::ElementScalar(2.5), Dispatch::ElementScalar(1.0));

            EXPECT_EQ(GetParam().apply(bound), GetParam().expected);
        }

        INSTANTIATE_TEST_SUITE_P(
            , Small_ScalarBinding, ::testing::ValuesIn(narrowing_rows()),
            [](const ::testing::TestParamInfo<NarrowingRow>& info)
            { return std::string(info.param.name); });

        TEST(Small_ScalarBindingNarrowFor, ABoundFunctorIsNarrowed)
        {
            const auto bound = Dispatch::bind_scalars(ScaleShift{}, Dispatch::ElementScalar(3.0),
                                                      Dispatch::ElementScalar(-1.0));

            EXPECT_EQ(Dispatch::Detail::narrow_for(Dispatch::ElementTag<double>{}, bound)(2.0),
                      5.0);
        }

        TEST(Small_ScalarBindingNarrowFor, APlainFunctorPassesThrough)
        {
            const auto plain = [](double x) { return x + 1.0; };

            EXPECT_EQ(Dispatch::Detail::narrow_for(Dispatch::ElementTag<double>{}, plain)(2.0),
                      3.0);
        }
    } // namespace Test
} // namespace SushiBLAS
