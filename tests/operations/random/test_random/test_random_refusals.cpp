/**************************************************************************/
/* test_random_refusals.cpp                                               */
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
 * @file test_random_refusals.cpp
 * @brief Tests that RandomOps refuses a bad parameter, bound, dtype or shape at record time, on a host engine.
 * @author Mustafa Garip
 */

#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.hpp>
#include <SushiBLAS/engine_fixture/host_engine_test.hpp>
using namespace SushiBLAS::Test;
using SushiBLAS::Core::DataType;
using SushiBLAS::RandomOps;
using SushiBLAS::Tensor;

namespace
{
    /// @brief Holds a quiet NaN.
    constexpr double K_NAN = std::numeric_limits<double>::quiet_NaN();

    /// @brief Holds positive infinity.
    constexpr double K_INF = std::numeric_limits<double>::infinity();

    /** @brief Records one operation on a tensor. */
    using Record = std::function<void(RandomOps&, Tensor&)>;

    /** @brief Names one recorded call, for a row of a table of refusals. */
    struct CallRow
    {
        const char* name; /**< The row name a failure shows. */
        Record record;    /**< Records the call. */
    };

    /** @brief Returns the name of a dtype as a test name. */
    std::string name_of(DataType dtype)
    {
        return std::string(SushiBLAS::Core::to_string(dtype));
    }
} // namespace

/** @brief Gives the refusal tests a host engine and a recorder, which no refusal runs. */
class Small_RandomRefusals : public HostEngineTest
{

};

/** @brief Names a truncated normal call its record-time check refuses. */
struct RefusedBoundsRow
{
    const char* name; /**< The row name a failure shows. */
    DataType dtype;   /**< The dtype of the tensor. */
    double lower;     /**< The lower cut. */
    double upper;     /**< The upper cut. */
};

/** @brief Returns the bounds the draw cannot honour, and a complex tensor. */
std::vector<RefusedBoundsRow> refused_bounds_rows()
{
    return {{"LowerAboveUpper", DataType::Float32, 2.0, -2.0},
            {"EmptyInterval", DataType::Float32, 1.0, 1.0},
            {"IntervalFarAboveTheMean", DataType::Float32, 15.0, 20.0},
            {"IntervalFarBelowTheMean", DataType::Float32, -20.0, -15.0},
            {"ComplexTensor", DataType::Complex32, -2.0, 2.0}};
}

/** @brief Runs one row of refused_bounds_rows(). */
class Small_TruncatedNormalRefusal : public HostEngineTest,
                                     public ::testing::WithParamInterface<RefusedBoundsRow>
{

};

TEST_P(Small_TruncatedNormalRefusal, IsRefusedWithInvalidArgument)
{
    const RefusedBoundsRow row = GetParam();
    auto t = engine->create_tensor({8}, row.dtype);

    EXPECT_THROW(RandomOps(recorder(), engine->rng())
                     .truncated_normal(t, 0.0, 1.0, row.lower, row.upper),
                 std::invalid_argument);
}

INSTANTIATE_TEST_SUITE_P(
    , Small_TruncatedNormalRefusal, ::testing::ValuesIn(refused_bounds_rows()),
    [](const ::testing::TestParamInfo<RefusedBoundsRow>& info)
    { return std::string(info.param.name); });

/** @brief Returns the calls whose parameter breaks its rule; no run follows, so a throw comes from the recording call. */
std::vector<CallRow> invalid_parameter_rows()
{
    return {
        {"ExponentialZeroRate", [](RandomOps& r, Tensor& t) { r.exponential(t, 0.0); }},
        {"ExponentialNegativeRate", [](RandomOps& r, Tensor& t) { r.exponential(t, -1.0); }},
        {"ExponentialInfiniteRate", [](RandomOps& r, Tensor& t) { r.exponential(t, K_INF); }},
        {"ExponentialNaNRate", [](RandomOps& r, Tensor& t) { r.exponential(t, K_NAN); }},
        {"NormalNegativeStddev", [](RandomOps& r, Tensor& t) { r.normal(t, 0.0, -1.0); }},
        {"NormalNaNStddev", [](RandomOps& r, Tensor& t) { r.normal(t, 0.0, K_NAN); }},
        {"NormalInfiniteMean", [](RandomOps& r, Tensor& t) { r.normal(t, K_INF, 1.0); }},
        {"LogNormalNegativeStddev", [](RandomOps& r, Tensor& t) { r.log_normal(t, 0.0, -1.0); }},
        {"LogNormalInfiniteStddev", [](RandomOps& r, Tensor& t) { r.log_normal(t, 0.0, K_INF); }},
        {"LogNormalNaNMean", [](RandomOps& r, Tensor& t) { r.log_normal(t, K_NAN, 1.0); }},
        {"BernoulliNaNProbability", [](RandomOps& r, Tensor& t) { r.bernoulli(t, K_NAN); }},
        {"BernoulliNegativeProbability", [](RandomOps& r, Tensor& t) { r.bernoulli(t, -0.5); }},
        {"BernoulliProbabilityAboveOne", [](RandomOps& r, Tensor& t) { r.bernoulli(t, 1.5); }},
        {"HeUniformZeroFanIn", [](RandomOps& r, Tensor& t) { r.he_uniform(t, 0); }},
        {"HeNormalNegativeFanIn", [](RandomOps& r, Tensor& t) { r.he_normal(t, -4); }},
        {"XavierUniformZeroFans", [](RandomOps& r, Tensor& t) { r.xavier_uniform(t, 0, 0); }},
        {"XavierNormalNegativeFanOut", [](RandomOps& r, Tensor& t) { r.xavier_normal(t, 2, -2); }}};
}

/** @brief Runs one row of invalid_parameter_rows(). */
class Small_RandomParameterRefusal : public HostEngineTest,
                                     public ::testing::WithParamInterface<CallRow>
{

};

TEST_P(Small_RandomParameterRefusal, IsRefusedWithInvalidArgument)
{
    auto t = engine->create_tensor({4, 4});
    RandomOps random(recorder(), engine->rng());

    EXPECT_THROW(GetParam().record(random, t), std::invalid_argument);
}

INSTANTIATE_TEST_SUITE_P(
    , Small_RandomParameterRefusal, ::testing::ValuesIn(invalid_parameter_rows()),
    [](const ::testing::TestParamInfo<CallRow>& info) { return std::string(info.param.name); });

/** @brief Runs one dtype that every value draw refuses. */
class Small_ValueDrawDtypeRefusal : public HostEngineTest,
                                    public ::testing::WithParamInterface<DataType>
{

};

TEST_P(Small_ValueDrawDtypeRefusal, EveryValueDrawRefusesHalfAndComplex)
{
    auto t = engine->create_tensor({4, 4}, GetParam());
    RandomOps random(recorder(), engine->rng());

    EXPECT_THROW(random.log_normal(t), std::invalid_argument);
    EXPECT_THROW(random.exponential(t), std::invalid_argument);
    EXPECT_THROW(random.poisson(t), std::invalid_argument);
    EXPECT_THROW(random.bernoulli(t), std::invalid_argument);
    EXPECT_THROW(random.discrete_uniform(t, 0, 10), std::invalid_argument);
    EXPECT_THROW(random.truncated_normal(t), std::invalid_argument);
    EXPECT_THROW(random.xavier_uniform(t, 4, 4), std::invalid_argument);
    EXPECT_THROW(random.xavier_normal(t, 4, 4), std::invalid_argument);
    EXPECT_THROW(random.he_uniform(t, 4), std::invalid_argument);
    EXPECT_THROW(random.he_normal(t, 4), std::invalid_argument);
    EXPECT_THROW(random.orthogonal(t), std::invalid_argument);
}

INSTANTIATE_TEST_SUITE_P(
    , Small_ValueDrawDtypeRefusal,
    ::testing::Values(DataType::Half, DataType::Complex32, DataType::Complex64),
    [](const ::testing::TestParamInfo<DataType>& info) { return name_of(info.param); });

TEST_F(Small_RandomRefusals, UniformAndNormalRefuseHalf)
{
    auto half = engine->create_tensor({4}, DataType::Half);
    RandomOps random(recorder(), engine->rng());

    EXPECT_THROW(random.uniform(half), std::invalid_argument);
    EXPECT_THROW(random.normal(half), std::invalid_argument);
}

/** @brief Runs one integer dtype that shuffle refuses. */
class Small_ShuffleDtypeRefusal : public HostEngineTest,
                                  public ::testing::WithParamInterface<DataType>
{

};

TEST_P(Small_ShuffleDtypeRefusal, ShuffleRefusesIntegers)
{
    auto t = engine->create_tensor({4}, GetParam());

    EXPECT_THROW(RandomOps(recorder(), engine->rng()).shuffle(t), std::invalid_argument);
}

INSTANTIATE_TEST_SUITE_P(
    , Small_ShuffleDtypeRefusal, ::testing::Values(DataType::Int32, DataType::Int64),
    [](const ::testing::TestParamInfo<DataType>& info) { return name_of(info.param); });

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
class Small_OrthogonalRefusal : public HostEngineTest,
                                public ::testing::WithParamInterface<RefusedRow>
{

};

TEST_P(Small_OrthogonalRefusal, IsRefusedWithInvalidArgument)
{
    const RefusedRow row = GetParam();
    auto t = engine->create_tensor(row.dims, row.dtype);

    EXPECT_THROW(SushiBLAS::RandomOps(recorder(), engine->rng()).orthogonal(t),
                 std::invalid_argument);
}

INSTANTIATE_TEST_SUITE_P(
    , Small_OrthogonalRefusal, ::testing::ValuesIn(refused_rows()),
    [](const ::testing::TestParamInfo<RefusedRow>& info) { return std::string(info.param.name); });

TEST_F(Small_RandomRefusals, OrthogonalRejectsAMatrixWithoutStorage)
{
    SushiBLAS::Tensor bare(SushiRuntime::sushi_ptr<SushiBLAS::Storage>(), {4, 4});
    EXPECT_THROW(SushiBLAS::RandomOps(recorder(), engine->rng())
        .orthogonal(bare), std::invalid_argument);
}
