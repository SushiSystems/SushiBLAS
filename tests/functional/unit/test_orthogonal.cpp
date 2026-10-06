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
 * @file test_orthogonal.cpp
 * @brief Tests that random().orthogonal fills a matrix with orthonormal rows or columns.
 * @author Mustafa Garip
 */

#include <cmath>
#include <vector>
#include <stdexcept>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.h>
#include "../../common/test_utils.hpp"

using namespace SushiBLAS::Test;
using SushiBLAS::Core::DataType;

class Unit_OrthogonalTest : public SushiBLASTest
{
    protected:
        std::vector<float> read_matrix(const SushiBLAS::Tensor& t)
        {
            std::vector<float> host(static_cast<size_t>(t.num_elements));
            auto& q = TestEnvironment::get_context().get_queue();
            q.memcpy(host.data(), t.data(), host.size() * sizeof(float)).wait();
            return host;
        }

        /** @brief Asserts that the vectors' Gram matrix is expected_diag times identity. */
        void expect_orthonormal(const std::vector<float>& m, int64_t count, int64_t length,
                                 int64_t elem_stride, int64_t vec_stride,
                                 float expected_diag, float tol = 1e-4f)
        {
            for (int64_t i = 0; i < count; ++i)
            {
                for (int64_t j = i; j < count; ++j)
                {
                    double dot = 0.0;
                    for (int64_t k = 0; k < length; ++k)
                    {
                        dot += static_cast<double>(m[static_cast<size_t>(i * vec_stride + k * elem_stride)]) *
                               static_cast<double>(m[static_cast<size_t>(j * vec_stride + k * elem_stride)]);
                    }
                    const float want = (i == j) ? expected_diag : 0.0f;
                    EXPECT_NEAR(static_cast<float>(dot), want, tol)
                        << "Gram entry (" << i << "," << j << ") is not " << want;
                }
            }
        }
};

TEST_F(Unit_OrthogonalTest, TallMatrixHasOrthonormalColumns)
{
    // 8 rows cannot all be orthogonal in 4 dimensions, so the four columns are: Q^T Q = I(4).
    auto t = engine->create_tensor({8, 4});
    engine->set_seed(2026);
    engine->random().orthogonal(t);
    engine->execute().wait();

    auto m = read_matrix(t);
    expect_orthonormal(m, /*count=*/4, /*length=*/8, /*elem_stride=*/4, /*vec_stride=*/1, 1.0f);
}

TEST_F(Unit_OrthogonalTest, WideMatrixHasOrthonormalRows)
{
    // 3x7, row-major: the short axis is the rows, so Q Q^T = I(3).
    auto t = engine->create_tensor({3, 7});
    engine->set_seed(7);
    engine->random().orthogonal(t);
    engine->execute().wait();

    auto m = read_matrix(t);
    expect_orthonormal(m, /*count=*/3, /*length=*/7, /*elem_stride=*/1, /*vec_stride=*/7, 1.0f);
}

TEST_F(Unit_OrthogonalTest, SquareMatrixIsFullyOrthogonal)
{
    auto t = engine->create_tensor({6, 6});
    engine->set_seed(31337);
    engine->random().orthogonal(t);
    engine->execute().wait();

    auto m = read_matrix(t);
    expect_orthonormal(m, 6, 6, /*elem_stride=*/6, /*vec_stride=*/1, 1.0f);
}

TEST_F(Unit_OrthogonalTest, GainScalesTheGramDiagonalQuadratically)
{
    // Q is scaled by gain, so Q^T Q is gain^2 I, the convention of PyTorch's orthogonal_.
    constexpr float kGain = 2.5f;
    auto t = engine->create_tensor({5, 5});
    engine->set_seed(11);
    engine->random().orthogonal(t, kGain);
    engine->execute().wait();

    auto m = read_matrix(t);
    expect_orthonormal(m, 5, 5, 5, 1, kGain * kGain, 1e-3f);
}

TEST_F(Unit_OrthogonalTest, ResultIsNotJustAGaussianFill)
{
    // A Gaussian 6x6 has column norms near sqrt(6). See tests/functional/unit/README.md.
    auto t = engine->create_tensor({6, 6});
    engine->set_seed(4242);
    engine->random().orthogonal(t);
    engine->execute().wait();

    auto m = read_matrix(t);
    for (int64_t c = 0; c < 6; ++c)
    {
        double norm_sq = 0.0;
        for (int64_t r = 0; r < 6; ++r)
        {
            const double v = m[static_cast<size_t>(r * 6 + c)];
            norm_sq += v * v;
        }
        EXPECT_NEAR(std::sqrt(norm_sq), 1.0, 1e-4) << "Column " << c << " is not a unit vector.";
    }
}

TEST_F(Unit_OrthogonalTest, RejectsInputsItCannotOrthonormalize)
{
    // Each shape or dtype the Gram-Schmidt sweep cannot handle is reported at the call site.
    auto vec = engine->create_tensor({16});
    EXPECT_THROW(engine->random().orthogonal(vec), std::runtime_error);

    auto cube = engine->create_tensor({2, 3, 4});
    EXPECT_THROW(engine->random().orthogonal(cube), std::runtime_error);

    auto half_mat = engine->create_tensor({4, 4}, DataType::HALF);
    EXPECT_THROW(engine->random().orthogonal(half_mat), std::runtime_error);

    auto complex_mat = engine->create_tensor({4, 4}, DataType::COMPLEX32);
    EXPECT_THROW(engine->random().orthogonal(complex_mat), std::runtime_error);
}
