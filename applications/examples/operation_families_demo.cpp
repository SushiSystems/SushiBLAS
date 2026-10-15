/**************************************************************************/
/* operation_families_demo.cpp                                            */
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
 * @file operation_families_demo.cpp
 * @brief Records one operation of each family, runs them and compares each result with the host.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <iostream>
#include <vector>

#include <SushiBLAS/SushiBLAS.hpp>

namespace
{
    using Values = std::vector<float>;

    /** @brief Returns a FLOAT32 tensor of the given shape that holds values. */
    [[nodiscard]] SushiBLAS::Tensor tensor_of(SushiBLAS::Engine& engine,
                                              std::initializer_list<std::int64_t> shape,
                                              const Values& values)
    {
        SushiBLAS::Tensor tensor = engine.create_tensor(shape);
        SushiBLAS::IO(engine).from_vector(tensor, values);
        return tensor;
    }

    /** @brief Prints whether the tensor holds expected, and returns it. */
    [[nodiscard]] bool holds(SushiBLAS::Engine& engine, const char* family,
                             const SushiBLAS::Tensor& tensor, const Values& expected)
    {
        const bool equal = SushiBLAS::IO(engine).to_vector<float>(tensor) == expected;
        std::cout << family << ": " << (equal ? "ok" : "MISMATCH") << "\n";
        return equal;
    }
} // namespace

int main()
{
    SushiRuntime::Runtime runtime = SushiRuntime::Runtime::create().value();
    SushiBLAS::Engine engine(runtime);

    const SushiBLAS::Tensor x = tensor_of(engine, {4}, {1.0f, 2.0f, 3.0f, 4.0f});
    const SushiBLAS::Tensor y = tensor_of(engine, {4}, {4.0f, 3.0f, 2.0f, 1.0f});
    const SushiBLAS::Tensor matrix = tensor_of(engine, {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    const SushiBLAS::Tensor ones = tensor_of(engine, {2}, {1.0f, 1.0f});
    const SushiBLAS::Tensor image = tensor_of(engine, {1, 2, 2, 1}, {1.0f, 2.0f, 3.0f, 4.0f});
    const SushiBLAS::Tensor dense = tensor_of(engine, {3}, {1.0f, 2.0f, 3.0f});
    SushiBLAS::Tensor signed_values = tensor_of(engine, {4}, {-1.0f, 2.0f, -3.0f, 4.0f});

    // The sparse matrix is [1 0 2; 0 3 0], given as (row, column, value) triplets.
    const std::vector<std::int32_t> rows{0, 0, 1};
    const std::vector<std::int32_t> columns{0, 2, 1};
    const SushiBLAS::SparseMatrix sparse = SushiBLAS::SparseBuilder(engine.tensor_factory())
        .matrix_from_triplets(2, 3, rows, columns, Values{1.0f, 2.0f, 3.0f});

    SushiBLAS::Tensor dot = engine.create_tensor({1});
    SushiBLAS::Tensor matrix_vector = engine.create_tensor({2});
    SushiBLAS::Tensor matrix_matrix = engine.create_tensor({2, 2});
    SushiBLAS::Tensor product = engine.create_tensor({4});
    SushiBLAS::Tensor greater = engine.create_tensor({4});
    SushiBLAS::Tensor maximum = engine.create_tensor({1});
    SushiBLAS::Tensor sparse_vector = engine.create_tensor({2});
    SushiBLAS::Tensor upsampled = engine.create_tensor({1, 4, 4, 1});
    SushiBLAS::Tensor first_draw = engine.create_tensor({64});
    SushiBLAS::Tensor second_draw = engine.create_tensor({64});

    SushiBLAS::BLASOps blas(engine);
    blas.dot(x, y, dot);
    blas.gemv(matrix, ones, matrix_vector);
    blas.gemm(matrix, matrix, matrix_matrix);
    SushiBLAS::ElementwiseOps(engine).mul(x, y, product);
    SushiBLAS::NonLinearOps(engine).relu(signed_values);
    SushiBLAS::LogicOps(engine).greater(x, y, greater);
    SushiBLAS::ReductionOps(engine).max(x, maximum);
    SushiBLAS::SparseOps(engine).spmv(sparse, dense, sparse_vector);
    SushiBLAS::SpatialOps(engine).upsample_nearest(image, upsampled, 2, 2);

    // The same seed gives the same stream, so the two draws must hold the same numbers.
    SushiBLAS::RandomOps random(engine);
    random.set_seed(42);
    random.uniform(first_draw);
    random.set_seed(42);
    random.uniform(second_draw);

    engine.execute().value();

    bool correct = holds(engine, "blas level 1, dot", dot, {20.0f});
    correct = holds(engine, "blas level 2, gemv", matrix_vector, {3.0f, 7.0f}) && correct;
    correct = holds(engine, "blas level 3, gemm", matrix_matrix, {7.0f, 10.0f, 15.0f, 22.0f}) &&
              correct;
    correct = holds(engine, "elementwise, mul", product, {4.0f, 6.0f, 6.0f, 4.0f}) && correct;
    correct = holds(engine, "nonlinear, relu", signed_values, {0.0f, 2.0f, 0.0f, 4.0f}) && correct;
    correct = holds(engine, "logic, greater", greater, {0.0f, 0.0f, 1.0f, 1.0f}) && correct;
    correct = holds(engine, "reductions, max", maximum, {4.0f}) && correct;
    correct = holds(engine, "sparse, spmv", sparse_vector, {7.0f, 6.0f}) && correct;
    correct = holds(engine, "spatial, upsample_nearest", upsampled,
                    {1.0f, 1.0f, 2.0f, 2.0f, 1.0f, 1.0f, 2.0f, 2.0f,
                     3.0f, 3.0f, 4.0f, 4.0f, 3.0f, 3.0f, 4.0f, 4.0f}) && correct;

    const Values draw = SushiBLAS::IO(engine).to_vector<float>(first_draw);
    bool in_range = true;
    for (const float value : draw)
        in_range = in_range && value >= 0.0f && value < 1.0f;
    correct = holds(engine, "random, uniform with seed 42 drawn twice", second_draw, draw) &&
              in_range && correct;

    std::cout << "operation_families_demo: " << (correct ? "ok" : "MISMATCH") << "\n";
    return correct ? 0 : 1;
}
