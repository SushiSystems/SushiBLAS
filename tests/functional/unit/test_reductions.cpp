/**************************************************************************/
/* test_reductions.cpp                                                    */
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
 * @file test_reductions.cpp
 * @brief Tests each whole-tensor reduction on a FLOAT32 tensor.
 * @author Mustafa Garip
 */

#include <cmath>
#include <limits>
#include <vector>
#include <random>
#include <numeric>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.h>
#include "../../common/test_utils.hpp"
#include "../../common/reference_reductions.hpp"

using namespace SushiBLAS::Test;
using namespace SushiBLAS;

template <typename T>
class Unit_ReductionOpsTyped : public SushiBLASTest 
{
protected:
    std::vector<T> generate_random_data(size_t size) 
    {
        std::vector<T> data(size);
        std::mt19937 gen(42);
        if constexpr (std::is_integral_v<T>) 
        {
            std::uniform_int_distribution<T> dist(-10, 10);
            for (auto& v : data) v = dist(gen);
        } else 
        {
            std::uniform_real_distribution<T> dist(static_cast<T>(-10.0), static_cast<T>(10.0));
            for (auto& v : data) v = dist(gen);
        }
        return data;
    }
};

using TestTypes = ::testing::Types<float>;
SB_TYPED_TEST_SUITE(Unit_ReductionOpsTyped, TestTypes);

TYPED_TEST(Unit_ReductionOpsTyped, Sum) 
{
    using T = TypeParam;
    auto data = this->generate_random_data(100);
    T expected = ReferenceKernels::sum(data);

    auto t = this->template create_initialized_tensor<T>({100}, data);
    auto res = this->engine->create_tensor({1}, get_sushiblas_dtype<T>());

    SushiBLAS::ReductionOps(*this->engine).sum(t, res);
    this->engine->execute();

    EXPECT_TENSOR_NEAR(res, {expected}, static_cast<float>(1e-4));
}

TYPED_TEST(Unit_ReductionOpsTyped, Mean) 
{
    using T = TypeParam;
    auto data = this->generate_random_data(100);
    T expected = ReferenceKernels::mean(data);

    auto t = this->template create_initialized_tensor<T>({100}, data);
    auto res = this->engine->create_tensor({1}, get_sushiblas_dtype<T>());

    SushiBLAS::ReductionOps(*this->engine).mean(t, res);
    this->engine->execute();

    EXPECT_TENSOR_NEAR(res, {expected}, static_cast<float>(1e-4));
}

TYPED_TEST(Unit_ReductionOpsTyped, Max) 
{
    using T = TypeParam;
    auto data = this->generate_random_data(100);
    data[25] = static_cast<T>(999.0);
    T expected = ReferenceKernels::max(data);

    auto t = this->template create_initialized_tensor<T>({100}, data);
    auto res = this->engine->create_tensor({1}, get_sushiblas_dtype<T>());

    SushiBLAS::ReductionOps(*this->engine).max(t, res);
    this->engine->execute();

    EXPECT_TENSOR_NEAR(res, {expected}, static_cast<float>(1e-4));
}

TYPED_TEST(Unit_ReductionOpsTyped, Min) 
{
    using T = TypeParam;
    auto data = this->generate_random_data(100);
    data[10] = static_cast<T>(-999.0);
    T expected = ReferenceKernels::min(data);

    auto t = this->template create_initialized_tensor<T>({100}, data);
    auto res = this->engine->create_tensor({1}, get_sushiblas_dtype<T>());

    SushiBLAS::ReductionOps(*this->engine).min(t, res);
    this->engine->execute();

    EXPECT_TENSOR_NEAR(res, {expected}, static_cast<float>(1e-4));
}

TYPED_TEST(Unit_ReductionOpsTyped, ArgMax) 
{
    using T = TypeParam;
    auto data = this->generate_random_data(100);
    data[42] = static_cast<T>(999.0);
    T expected = ReferenceKernels::argmax(data);

    auto t = this->template create_initialized_tensor<T>({100}, data);

    // The index is INT64 and is compared exactly. See tests/functional/unit/README.md.
    auto res = this->engine->create_tensor({1}, Core::DataType::INT64);

    SushiBLAS::ReductionOps(*this->engine).argmax(t, res);
    this->engine->execute();

    EXPECT_TENSOR_NEAR(res, {static_cast<float>(expected)}, 0.0);
}

TYPED_TEST(Unit_ReductionOpsTyped, ArgMin) 
{
    using T = TypeParam;
    auto data = this->generate_random_data(100);
    data[7] = static_cast<T>(-999.0);
    T expected = ReferenceKernels::argmin(data);

    auto t = this->template create_initialized_tensor<T>({100}, data);

    // INT64 for the same reason as ArgMax above.
    auto res = this->engine->create_tensor({1}, Core::DataType::INT64);

    SushiBLAS::ReductionOps(*this->engine).argmin(t, res);
    this->engine->execute();

    EXPECT_TENSOR_NEAR(res, {static_cast<float>(expected)}, 0.0);
}

TYPED_TEST(Unit_ReductionOpsTyped, Var) 
{
    using T = TypeParam;
    auto data = this->generate_random_data(5);
    T expected = ReferenceKernels::var(data);

    auto t = this->template create_initialized_tensor<T>({5}, data);
    auto res = this->engine->create_tensor({1}, get_sushiblas_dtype<T>());

    SushiBLAS::ReductionOps(*this->engine).var(t, res);
    this->engine->execute();

    EXPECT_TENSOR_NEAR(res, {expected}, static_cast<float>(1e-4));
}

TYPED_TEST(Unit_ReductionOpsTyped, Std) 
{
    using T = TypeParam;
    auto data = this->generate_random_data(5);
    T expected = ReferenceKernels::std_dev(data);

    auto t = this->template create_initialized_tensor<T>({5}, data);
    auto res = this->engine->create_tensor({1}, get_sushiblas_dtype<T>());

    SushiBLAS::ReductionOps(*this->engine).std(t, res);
    this->engine->execute();

    EXPECT_TENSOR_NEAR(res, {expected}, static_cast<float>(1e-4));
}

TYPED_TEST(Unit_ReductionOpsTyped, Prod) 
{
    using T = TypeParam;
    std::vector<T> data = {static_cast<T>(1), static_cast<T>(2), static_cast<T>(3), static_cast<T>(4), static_cast<T>(5)};
    T expected = ReferenceKernels::prod(data);

    auto t = this->template create_initialized_tensor<T>({5}, data);
    auto res = this->engine->create_tensor({1}, get_sushiblas_dtype<T>());

    SushiBLAS::ReductionOps(*this->engine).prod(t, res);
    this->engine->execute();

    EXPECT_TENSOR_NEAR(res, {expected}, static_cast<float>(1e-4));
}

TYPED_TEST(Unit_ReductionOpsTyped, LogSumExp) 
{
    using T = TypeParam;
    std::vector<T> data = {static_cast<T>(1), static_cast<T>(2), static_cast<T>(3)};
    T expected = ReferenceKernels::logsumexp(data);

    auto t = this->template create_initialized_tensor<T>({3}, data);
    auto res = this->engine->create_tensor({1}, get_sushiblas_dtype<T>());

    SushiBLAS::ReductionOps(*this->engine).logsumexp(t, res);
    this->engine->execute();

    EXPECT_TENSOR_NEAR(res, {expected}, static_cast<float>(1e-4));
}

TYPED_TEST(Unit_ReductionOpsTyped, Asum) 
{
    using T = TypeParam;
    auto data = this->generate_random_data(5);
    T expected = ReferenceKernels::asum(data);

    auto t = this->template create_initialized_tensor<T>({5}, data);
    auto res = this->engine->create_tensor({1}, get_sushiblas_dtype<T>());

    SushiBLAS::ReductionOps(*this->engine).asum(t, res);
    this->engine->execute();

    EXPECT_TENSOR_NEAR(res, {expected}, static_cast<float>(1e-4));
}

TYPED_TEST(Unit_ReductionOpsTyped, Nrm2) 
{
    using T = TypeParam;
    auto data = this->generate_random_data(4);
    T expected = ReferenceKernels::nrm2(data);

    auto t = this->template create_initialized_tensor<T>({4}, data);
    auto res = this->engine->create_tensor({1}, get_sushiblas_dtype<T>());

    SushiBLAS::ReductionOps(*this->engine).nrm2(t, res);
    this->engine->execute();

    EXPECT_TENSOR_NEAR(res, {expected}, static_cast<float>(1e-4));
}
