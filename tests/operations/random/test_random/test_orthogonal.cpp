/**************************************************************************/
/* test_orthogonal.cpp                                                    */
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
 * @file test_orthogonal.cpp
 * @brief Tests that RandomOps::orthogonal fills a matrix with orthonormal rows or columns.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <string>
#include <vector>
#include <stdexcept>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.hpp>
#include <SushiRuntime/core/sushi_ptr.hpp>
#include <SushiBLAS/engine_fixture/sushiblas_test.hpp>
#include <SushiBLAS/tensor_expectations/tensor_readback.hpp>
#include <SushiBLAS/tensor_expectations/tolerance_bounds.hpp>
using namespace SushiBLAS::Test;
using SushiBLAS::Core::DataType;

namespace
{
    /// @brief Holds the float roundings a Gram entry of a unit-gain matrix may carry, as the default comparison allows.
    constexpr std::size_t K_GRAM_OPERATIONS = 128;

    /** @brief Names one matrix shape, its seed and its gain. */
    struct GramRow
    {
        const char* name; /**< The row name a failure shows. */
        int64_t rows;     /**< The row count. */
        int64_t cols;     /**< The column count. */
        uint64_t seed;    /**< The seed of the fill. */
        float gain;       /**< The gain the matrix is scaled by. */
    };

    /** @brief Returns the shapes whose Gram matrix is checked; 64 is the widest work-group of the sweep. */
    std::vector<GramRow> gram_rows()
    {
        return {{"TallHasOrthonormalColumns", 8, 4, 2026, 1.0f},
                {"WideHasOrthonormalRows", 3, 7, 7, 1.0f},
                {"SquareIsFullyOrthogonal", 6, 6, 31337, 1.0f},
                {"SixtyFourSquareIsFullyOrthogonal", 64, 64, 20261006, 1.0f},
                {"GainScalesTheDiagonalQuadratically", 5, 5, 11, 2.5f},
                {"SingleVectorIsScaledToTheGain", 1, 4, 99, 3.0f}};
    }
}

/** @brief Runs one row of gram_rows(). */
class Medium_OrthogonalGram : public SushiBLASTest,
                              public ::testing::WithParamInterface<GramRow>
{

};

TEST_P(Medium_OrthogonalGram, VectorsHaveTheGramMatrixOfTheGainTimesIdentity)
{
    const GramRow row = GetParam();
    auto t = engine->create_tensor({row.rows, row.cols});
    SushiBLAS::RandomOps(recorder(), engine->rng()).set_seed(row.seed);
    SushiBLAS::RandomOps(recorder(), engine->rng()).orthogonal(t, row.gain);
    recording->run().value();

    // The shorter axis holds the vectors: rows of a wide matrix, columns of a tall one, so Q Q^T or Q^T Q is gain^2 I.
    const std::vector<float> matrix = read_tensor<float>(t);
    const bool by_rows = row.rows <= row.cols;
    const int64_t count = by_rows ? row.rows : row.cols;
    const int64_t length = by_rows ? row.cols : row.rows;
    const int64_t elem_stride = by_rows ? 1 : row.cols;
    const int64_t vec_stride = by_rows ? row.cols : 1;
    const double gain_squared = static_cast<double>(row.gain) * static_cast<double>(row.gain);
    const double tolerance = absolute_tolerance<float>(K_GRAM_OPERATIONS, gain_squared);

    for (int64_t i = 0; i < count; ++i)
    {
        for (int64_t j = i; j < count; ++j)
        {
            double dot = 0.0;
            for (int64_t k = 0; k < length; ++k)
            {
                dot += static_cast<double>(matrix[static_cast<size_t>(i * vec_stride + k * elem_stride)]) *
                       static_cast<double>(matrix[static_cast<size_t>(j * vec_stride + k * elem_stride)]);
            }
            EXPECT_NEAR(dot, i == j ? gain_squared : 0.0, tolerance)
                << "Gram entry (" << i << "," << j << ")";
        }
    }
}

INSTANTIATE_TEST_SUITE_P(
    , Medium_OrthogonalGram, ::testing::ValuesIn(gram_rows()),
    [](const ::testing::TestParamInfo<GramRow>& info) { return std::string(info.param.name); });

class Medium_OrthogonalTest : public SushiBLASTest
{

};

/** @brief Names a shape and dtype the sweep cannot orthonormalize. */
struct RefusedRow
{
    const char* name;                 /**< The row name a failure shows. */
    std::vector<int64_t> dims;        /**< The shape of the tensor. */
    DataType dtype;                   /**< The dtype of the tensor. */
};

/** @brief Returns the inputs orthogonal refuses at the call site. */
std::vector<RefusedRow> refused_rows()
{
    return {{"RankOne", {16}, DataType::Float32},
            {"RankThree", {2, 3, 4}, DataType::Float32},
            {"Half", {4, 4}, DataType::Half},
            {"Complex32", {4, 4}, DataType::Complex32}};
}

/** @brief Runs one row of refused_rows(). */
class Medium_OrthogonalRefusal : public SushiBLASTest,
                                 public ::testing::WithParamInterface<RefusedRow>
{

};

TEST_P(Medium_OrthogonalRefusal, IsRefusedWithInvalidArgument)
{
    const RefusedRow row = GetParam();
    auto t = engine->create_tensor(row.dims, row.dtype);

    EXPECT_THROW(SushiBLAS::RandomOps(recorder(), engine->rng()).orthogonal(t),
                 std::invalid_argument);
}

INSTANTIATE_TEST_SUITE_P(
    , Medium_OrthogonalRefusal, ::testing::ValuesIn(refused_rows()),
    [](const ::testing::TestParamInfo<RefusedRow>& info) { return std::string(info.param.name); });

TEST_F(Medium_OrthogonalTest, RejectsAMatrixWithoutStorage)
{
    SushiBLAS::Tensor bare(SushiRuntime::sushi_ptr<SushiBLAS::Storage>(), {4, 4});
    EXPECT_THROW(SushiBLAS::RandomOps(recorder(), engine->rng())
        .orthogonal(bare), std::invalid_argument);
}
