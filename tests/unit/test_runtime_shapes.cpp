/**************************************************************************/
/* test_runtime_shapes.cpp                                                */
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
 * @file test_runtime_shapes.cpp
 * @brief Tests create_tensor and reshape with dimensions held in a std::vector.
 * @author Mustafa Garip
 */

#include <vector>
#include <cstdint>
#include <stdexcept>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.h>
#include <SushiBLAS/engine_fixture/sushiblas_test.hpp>
namespace SushiBLAS
{
    namespace Test
    {
        /** @brief Holds a rank-3 shape known only at run time. */
        const std::vector<int64_t> K_DIMS{2, 3, 4};

        /** @brief Holds 2 * 3 * 4, the element count of K_DIMS. */
        constexpr int64_t K_ELEMENTS = 24;

        /** @brief Holds the row-major strides of K_DIMS: {3 * 4, 4, 1}. */
        const std::vector<int64_t> K_ROW_MAJOR_STRIDES{12, 4, 1};

        /** @brief Holds K_DIMS regrouped as a 6x4 matrix. */
        const std::vector<int64_t> K_RESHAPED{6, 4};

        /** @brief Supplies a row-major engine. */
        class Unit_RuntimeShapes : public SushiBLASTest
        {

        };

        TEST_F(Unit_RuntimeShapes, CreateTensorTakesAVectorShape)
        {
            const Tensor t = engine->create_tensor(K_DIMS);

            ASSERT_EQ(t.rank, static_cast<int32_t>(K_DIMS.size()));
            EXPECT_EQ(t.num_elements, K_ELEMENTS);
            EXPECT_EQ(t.dtype, Core::DataType::Float32);
            for (size_t i = 0; i < K_DIMS.size(); ++i)
            {
                EXPECT_EQ(t.shape[i], K_DIMS[i]) << "axis " << i;
                EXPECT_EQ(t.strides[i], K_ROW_MAJOR_STRIDES[i]) << "axis " << i;
            }
        }

        TEST_F(Unit_RuntimeShapes, CreateTensorTakesAVectorShapeAndDtype)
        {
            const Tensor t = engine->create_tensor(K_DIMS, Core::DataType::Float64);

            EXPECT_EQ(t.dtype, Core::DataType::Float64);
            EXPECT_EQ(t.num_elements, K_ELEMENTS);
        }

        TEST_F(Unit_RuntimeShapes, CreateTensorRejectsANegativeDimension)
        {
            const std::vector<int64_t> negative{2, -3};

            EXPECT_THROW((void)engine->create_tensor(negative), std::runtime_error);
        }

        TEST_F(Unit_RuntimeShapes, ReshapeTakesAVectorShape)
        {
            const Tensor t = engine->create_tensor(K_DIMS);
            const Tensor r = t.reshape(K_RESHAPED);

            ASSERT_EQ(r.rank, 2);
            EXPECT_EQ(r.shape[0], K_RESHAPED[0]);
            EXPECT_EQ(r.shape[1], K_RESHAPED[1]);
            EXPECT_EQ(r.num_elements, K_ELEMENTS);
        }

        TEST_F(Unit_RuntimeShapes, ReshapeRejectsAnotherElementCount)
        {
            const Tensor t = engine->create_tensor(K_DIMS);
            const std::vector<int64_t> wrong{5, 5};

            EXPECT_THROW((void)t.reshape(wrong), std::runtime_error);
        }
    } // namespace Test
} // namespace SushiBLAS
