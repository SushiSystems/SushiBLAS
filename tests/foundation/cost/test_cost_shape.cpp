/**************************************************************************/
/* test_cost_shape.cpp                                                    */
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
 * @file test_cost_shape.cpp
 * @brief Tests that a CostShape holds one family at a time and starts empty.
 * @author Mustafa Garip
 * @see tests/foundation/cost/README.md
 */

#include <SushiBLAS/cost/cost_shape.hpp>

#include <variant>

#include <gtest/gtest.h>

namespace SushiBLAS
{
    namespace Test
    {
        TEST(Small_CostShape, ADefaultShapeStatesNothing)
        {
            const CostShape shape;
            EXPECT_TRUE(std::holds_alternative<std::monostate>(shape));
        }

        TEST(Small_CostShape, EachFamilyIsAnAlternative)
        {
            EXPECT_TRUE(std::holds_alternative<GemmShape>(CostShape(GemmShape{})));
            EXPECT_TRUE(std::holds_alternative<ConvolutionShape>(CostShape(ConvolutionShape{})));
            EXPECT_TRUE(std::holds_alternative<ElementwiseShape>(CostShape(ElementwiseShape{})));
            EXPECT_TRUE(std::holds_alternative<ReductionShape>(CostShape(ReductionShape{})));
            EXPECT_TRUE(std::holds_alternative<WindowShape>(CostShape(WindowShape{})));
            EXPECT_TRUE(std::holds_alternative<TransformShape>(CostShape(TransformShape{})));
            EXPECT_TRUE(std::holds_alternative<CopyShape>(CostShape(CopyShape{})));
        }

        TEST(Small_CostShape, ADefaultGemmIsOneProductWithNoEpilogue)
        {
            const GemmShape shape;
            EXPECT_EQ(shape.batch, 1);
            EXPECT_EQ(shape.epilogue_flops_per_element, 0U);
        }
    } // namespace Test
} // namespace SushiBLAS
