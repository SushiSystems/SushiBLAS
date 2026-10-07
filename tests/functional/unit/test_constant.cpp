/**************************************************************************/
/* test_constant.cpp                                                      */
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
 * @file test_constant.cpp
 * @brief Tests that random().constant fills a tensor of every supported dtype with one value.
 * @author Mustafa Garip
 */

#include <complex>
#include <cstdint>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.h>
#include "../../common/test_utils.hpp"

using namespace SushiBLAS::Test;
using SushiBLAS::Core::DataType;

namespace
{
    /// @brief Holds the fill value for floating-point and complex tensors; exact in every one of them.
    constexpr double K_REAL_VALUE = 2.5;

    /// @brief Holds the fill value for integer tensors.
    constexpr double K_INTEGER_VALUE = -3.0;

    /// @brief Holds an odd element count that is not a multiple of any vector width.
    constexpr int64_t K_ODD_SIZE = 1001;
}

class Unit_ConstantTest : public SushiBLASTest
{
    protected:
        /**
         * @brief Fills a tensor of the given dtype and size with value and checks every element.
         * @param value Expected back exactly; complex elements expect a zero imaginary part.
         */
        void expect_constant_fill(DataType dtype, int64_t size, double value)
        {
            auto t = engine->create_tensor({size}, dtype);
            SushiBLAS::RandomOps(*engine).constant(t, value);
            engine->execute();
            verify_tensor(t, std::vector<double>(static_cast<size_t>(size), value), 0.0);
        }
};

TEST_F(Unit_ConstantTest, Float32)
{
    const int N = 10;
    auto t = engine->create_tensor({N});
    SushiBLAS::RandomOps(*engine).constant(t, 2.5);
    engine->execute();
    verify_tensor(t, std::vector<float>(N, 2.5f));
}

TEST_F(Unit_ConstantTest, HalfOddSize)
{
    expect_constant_fill(DataType::HALF, K_ODD_SIZE, K_REAL_VALUE);
}

TEST_F(Unit_ConstantTest, Float32OddSize)
{
    expect_constant_fill(DataType::FLOAT32, K_ODD_SIZE, K_REAL_VALUE);
}

TEST_F(Unit_ConstantTest, Float64OddSize)
{
    expect_constant_fill(DataType::FLOAT64, K_ODD_SIZE, K_REAL_VALUE);
}

TEST_F(Unit_ConstantTest, Complex32OddSizeHasZeroImaginaryPart)
{
    expect_constant_fill(DataType::COMPLEX32, K_ODD_SIZE, K_REAL_VALUE);
}

TEST_F(Unit_ConstantTest, Complex64OddSizeHasZeroImaginaryPart)
{
    expect_constant_fill(DataType::COMPLEX64, K_ODD_SIZE, K_REAL_VALUE);
}

TEST_F(Unit_ConstantTest, Int32OddSize)
{
    expect_constant_fill(DataType::INT32, K_ODD_SIZE, K_INTEGER_VALUE);
}

TEST_F(Unit_ConstantTest, Int64OddSize)
{
    expect_constant_fill(DataType::INT64, K_ODD_SIZE, K_INTEGER_VALUE);
}

TEST_F(Unit_ConstantTest, SingleElement)
{
    expect_constant_fill(DataType::FLOAT32, 1, K_REAL_VALUE);
}
