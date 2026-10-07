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
 * @brief Tests each whole-tensor reduction, its refusals and its two-stage results.
 * @author Mustafa Garip
 */

#include <cmath>
#include <limits>
#include <string>
#include <vector>
#include <random>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <type_traits>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.h>
#include "../../common/test_utils.hpp"
#include "../../common/graph_probe.hpp"
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

namespace
{
    /// @brief Holds a quiet NaN.
    const float K_NAN = std::numeric_limits<float>::quiet_NaN();

    /// @brief Holds positive infinity.
    const float K_INF = std::numeric_limits<float>::infinity();

    /// @brief Holds an element count a CPU folds in one sequential pass.
    constexpr std::size_t K_SINGLE_PASS_SIZE = 7;

    /// @brief Holds an element count past the CPU's two-stage threshold.
    constexpr std::size_t K_TWO_STAGE_SIZE = 100000;

    /// @brief Holds the seed of the generator the moment cases draw from.
    constexpr std::uint32_t K_MOMENT_SEED = 20261007u;

    /// @brief Holds how many units in the last place of the element type a moment may be off by.
    constexpr long double K_MOMENT_ULPS = 4.0L;

    /// @brief Holds the absolute tolerance of the finite logsumexp expectation.
    constexpr double K_LOGSUMEXP_TOLERANCE = 1e-3;

    /// @brief Holds a value as hi + lo, where lo is the rounding error of hi.
    struct Paired
    {
        long double hi; /**< The leading part. */
        long double lo; /**< What the leading part rounded away. */
    };

    /** @brief Returns a + b and the rounding error of that sum, exactly. */
    Paired exact_sum(long double a, long double b)
    {
        const long double sum = a + b;
        const long double b_seen = sum - a;
        const long double error = (a - (sum - b_seen)) + (b - b_seen);
        return Paired{sum, error};
    }

    /** @brief Returns a + b to twice the precision of long double. */
    Paired paired_sum(Paired a, Paired b)
    {
        const Paired leading = exact_sum(a.hi, b.hi);
        return exact_sum(leading.hi, leading.lo + a.lo + b.lo);
    }

    /** @brief Returns a * a to twice the precision of long double. */
    Paired paired_square(Paired a)
    {
        const long double product = a.hi * a.hi;
        const long double error = std::fma(a.hi, a.hi, -product) + 2.0L * a.hi * a.lo;
        return exact_sum(product, error);
    }

    /** @brief Returns a / n to twice the precision of long double. */
    Paired paired_quotient(Paired a, long double n)
    {
        const long double quotient = a.hi / n;
        const long double remainder = std::fma(-quotient, n, a.hi) + a.lo;
        return exact_sum(quotient, remainder / n);
    }

    /** @brief Widens one element to long double, which holds every element type exactly. */
    template<typename T>
    long double to_long_double(T value)
    {
        if constexpr (std::is_same_v<T, sycl::half>)
            return static_cast<long double>(static_cast<float>(value));
        else
            return static_cast<long double>(value);
    }

    /** @brief Returns the population variance of values from a two-pass host computation. */
    template<typename T>
    long double host_variance(const std::vector<T>& values)
    {
        const long double n = static_cast<long double>(values.size());

        Paired total{0.0L, 0.0L};
        for (const T& value : values)
            total = paired_sum(total, Paired{to_long_double(value), 0.0L});

        const Paired mean = paired_quotient(total, n);
        const Paired minus_mean{-mean.hi, -mean.lo};

        Paired squares{0.0L, 0.0L};
        for (const T& value : values)
        {
            const Paired deviation = paired_sum(Paired{to_long_double(value), 0.0L}, minus_mean);
            squares = paired_sum(squares, paired_square(deviation));
        }

        return paired_quotient(squares, n).hi;
    }

    /** @brief Returns one unit in the last place of T at magnitude, which must not be zero. */
    template<typename T>
    long double ulp_at(long double magnitude)
    {
        return std::ldexp(1.0L, std::ilogb(magnitude) - (std::numeric_limits<T>::digits - 1));
    }

    /** @brief Returns the message of the std::invalid_argument record throws, or "" if none. */
    template<typename Record>
    std::string refusal_of(Record&& record)
    {
        try
        {
            record();
        }
        catch (const std::invalid_argument& refusal)
        {
            return std::string(refusal.what());
        }
        return std::string();
    }
}

/// @brief Runs the reduction cases that take their own element types.
class Unit_ReductionOps : public SushiBLASTest
{
    protected:
        /** @brief Expects record to throw a std::invalid_argument that names FORCE_REFERENCE. */
        template<typename Record>
        void expect_mode_refusal(const char* name, Record&& record)
        {
            SCOPED_TRACE(name);
            const std::string message = refusal_of(record);
            EXPECT_NE(std::string::npos, message.find("FORCE_REFERENCE")) << message;
        }

        /** @brief Reduces FLOAT32 values with logsumexp and checks the result. */
        void expect_logsumexp(const std::vector<float>& values, float expected)
        {
            auto t = engine->create_tensor({static_cast<int64_t>(values.size())});
            fill_tensor(t, values);

            auto result = engine->create_tensor({1});
            SushiBLAS::ReductionOps(*engine).logsumexp(t, result);
            engine->execute();

            verify_tensor(result, {expected}, K_LOGSUMEXP_TOLERANCE);
        }

        /** @brief Reduces values with argmax and argmin and checks both indices. */
        template<typename T>
        void expect_arg_indices(const std::vector<T>& values, int64_t expected_max,
                                int64_t expected_min)
        {
            auto t = create_initialized_tensor<T>({static_cast<int64_t>(values.size())}, values);
            auto at_max = engine->create_tensor({1}, Core::DataType::INT64);
            auto at_min = engine->create_tensor({1}, Core::DataType::INT64);

            SushiBLAS::ReductionOps(*engine).argmax(t, at_max);
            SushiBLAS::ReductionOps(*engine).argmin(t, at_min);
            engine->execute();

            EXPECT_EQ(expected_max, read_tensor<int64_t>(at_max)[0]);
            EXPECT_EQ(expected_min, read_tensor<int64_t>(at_min)[0]);
        }

        /**
         * @brief Checks the tie rule of argmax and argmin on size elements of T.
         * @param size At least 7, so the four marked positions are distinct.
         */
        template<typename T>
        void expect_ties_take_the_lowest_index(std::size_t size)
        {
            const std::size_t first_peak = size / 3;
            const std::size_t first_trough = size / 5;

            // The maximum sits at size / 3 and size - 2, the minimum at size / 5 and size - 1.
            std::vector<T> values(size, static_cast<T>(1));
            values[first_peak] = static_cast<T>(5);
            values[size - 2] = static_cast<T>(5);
            values[first_trough] = static_cast<T>(-5);
            values[size - 1] = static_cast<T>(-5);

            expect_arg_indices(values, static_cast<int64_t>(first_peak),
                               static_cast<int64_t>(first_trough));
        }
};

/**
 * @brief Tests that the real-only reductions refuse integer and complex tensors by dtype name.
 */
TEST_F(Unit_ReductionOps, NonRealDtypesAreRefusedByName)
{
    SushiBLAS::ReductionOps reductions(*engine);

    for (Core::DataType dtype : {Core::DataType::INT32, Core::DataType::INT64})
    {
        SCOPED_TRACE(Core::to_string(dtype));

        auto t = engine->create_tensor({4}, dtype);
        auto result = engine->create_tensor({1}, dtype);
        const std::string name = Core::to_string(dtype);

        EXPECT_NE(std::string::npos, refusal_of([&] { reductions.sum(t, result); }).find(name));
        EXPECT_NE(std::string::npos, refusal_of([&] { reductions.nrm2(t, result); }).find(name));
        EXPECT_NE(std::string::npos, refusal_of([&] { reductions.mean(t, result); }).find(name));
    }

    auto z = engine->create_tensor({4}, Core::DataType::COMPLEX32);
    auto z_result = engine->create_tensor({1}, Core::DataType::COMPLEX32);
    const std::string complex_name = Core::to_string(Core::DataType::COMPLEX32);

    EXPECT_NE(std::string::npos,
              refusal_of([&] { reductions.sum(z, z_result); }).find(complex_name));
    EXPECT_NE(std::string::npos,
              refusal_of([&] { reductions.max(z, z_result); }).find(complex_name));
}

/**
 * @brief Tests that every reduction refuses FORCE_REFERENCE before it leases scratch.
 */
TEST_F(Unit_ReductionOps, ForceReferenceRefusesEveryReduction)
{
    GraphProbe probe(ExecutionMode::FORCE_REFERENCE);
    SushiBLAS::ReductionOps reductions(probe.get_recorder());

    auto t = engine->create_tensor({static_cast<int64_t>(K_TWO_STAGE_SIZE)});
    auto scalar = engine->create_tensor({1});
    auto index = engine->create_tensor({1}, Core::DataType::INT64);
    auto matrix = engine->create_tensor({2, 4});
    auto per_row = engine->create_tensor({2});
    auto image = engine->create_tensor({1, 2, 2, 3});
    auto per_channel = engine->create_tensor({1, 3});

    auto& context = TestEnvironment::get_context();
    const std::size_t idle_before = probe.get_scratch_pool().idle_block_count();
    const std::size_t live_before = context.live_usm_count();

    expect_mode_refusal("sum", [&] { reductions.sum(t, scalar); });
    expect_mode_refusal("mean", [&] { reductions.mean(t, scalar); });
    expect_mode_refusal("max", [&] { reductions.max(t, scalar); });
    expect_mode_refusal("min", [&] { reductions.min(t, scalar); });
    expect_mode_refusal("argmax", [&] { reductions.argmax(t, index); });
    expect_mode_refusal("argmin", [&] { reductions.argmin(t, index); });
    expect_mode_refusal("var", [&] { reductions.var(t, scalar); });
    expect_mode_refusal("std", [&] { reductions.std(t, scalar); });
    expect_mode_refusal("prod", [&] { reductions.prod(t, scalar); });
    expect_mode_refusal("logsumexp", [&] { reductions.logsumexp(t, scalar); });
    expect_mode_refusal("max_along", [&] { reductions.max_along(matrix, 1, per_row); });
    expect_mode_refusal("min_along", [&] { reductions.min_along(matrix, 1, per_row); });
    expect_mode_refusal("sum_along", [&] { reductions.sum_along(matrix, 1, per_row); });
    expect_mode_refusal("mean_spatial", [&] { reductions.mean_spatial(image, per_channel); });
    expect_mode_refusal("asum", [&] { reductions.asum(t, scalar); });
    expect_mode_refusal("nrm2", [&] { reductions.nrm2(t, scalar); });
    expect_mode_refusal("count_nonfinite", [&] { reductions.count_nonfinite(t, index); });

    EXPECT_TRUE(probe.take_nodes().empty());
    EXPECT_EQ(idle_before, probe.get_scratch_pool().idle_block_count());
    EXPECT_EQ(live_before, context.live_usm_count());
}

/**
 * @brief Tests that the mode is reported before a malformed operand is.
 */
TEST_F(Unit_ReductionOps, ForceReferenceIsReportedBeforeTheOperands)
{
    GraphProbe probe(ExecutionMode::FORCE_REFERENCE);
    SushiBLAS::ReductionOps reductions(probe.get_recorder());

    auto vector = engine->create_tensor({4});
    auto wide = engine->create_tensor({3});

    // A rank-1 operand is malformed for every axis reduction.
    expect_mode_refusal("sum_along", [&] { reductions.sum_along(vector, 0, wide); });
    expect_mode_refusal("max_along", [&] { reductions.max_along(vector, 0, wide); });

    // A three-element result is not a scalar.
    expect_mode_refusal("sum", [&] { reductions.sum(vector, wide); });
    expect_mode_refusal("argmax", [&] { reductions.argmax(vector, wide); });
}

/**
 * @brief Tests logsumexp on an infinite maximum, on a NaN and on two large equal elements.
 */
TEST_F(Unit_ReductionOps, LogSumExpHandlesInfiniteAndNaNInputs)
{
    expect_logsumexp({0.0f, K_INF}, K_INF);
    expect_logsumexp({-K_INF, -K_INF}, -K_INF);
    expect_logsumexp({1.0f, K_NAN}, K_NAN);

    // Two equal elements give their value plus log(2).
    expect_logsumexp({1000.0f, 1000.0f}, 1000.0f + std::log(2.0f));
}

/**
 * @brief Tests the tie and NaN rules of argmax and argmin in both launch shapes.
 */
TEST_F(Unit_ReductionOps, ArgMaxAndArgMinFollowTheScanRule)
{
    for (std::size_t size : {K_SINGLE_PASS_SIZE, K_TWO_STAGE_SIZE})
    {
        SCOPED_TRACE(size);

        expect_ties_take_the_lowest_index<float>(size);
        expect_ties_take_the_lowest_index<int32_t>(size);
        expect_ties_take_the_lowest_index<int64_t>(size);

        expect_arg_indices(std::vector<float>(size, 2.5f), 0, 0);

        // A NaN at index 0 is the answer of both, whatever follows it.
        std::vector<float> nan_first(size, 1.0f);
        nan_first[0] = K_NAN;
        nan_first[size / 3] = 5.0f;
        nan_first[size / 5] = -5.0f;
        expect_arg_indices(nan_first, 0, 0);

        // A NaN past index 0 is never chosen.
        std::vector<float> nan_later(size, 1.0f);
        nan_later[size / 2] = K_NAN;
        nan_later[size / 3] = 5.0f;
        nan_later[size / 5] = -5.0f;
        expect_arg_indices(nan_later, static_cast<int64_t>(size / 3),
                           static_cast<int64_t>(size / 5));
    }
}

/// @brief Runs the var and std cases once per real element type.
template<typename T>
class Unit_ReductionMomentsTyped : public SushiBLASTest
{

};

/// @brief Names the element types the moment cases run over.
using MomentTypes = ::testing::Types<sycl::half, float, double>;
SB_TYPED_TEST_SUITE(Unit_ReductionMomentsTyped, MomentTypes);

/**
 * @brief Tests that var and std stay within four ulp of a two-pass host computation.
 */
TYPED_TEST(Unit_ReductionMomentsTyped, VarianceMatchesATwoPassHostReference)
{
    using T = TypeParam;

    std::mt19937 generator(K_MOMENT_SEED);
    std::uniform_real_distribution<float> distribution(-10.0f, 10.0f);

    for (std::size_t size : {std::size_t{5}, std::size_t{1000}, K_TWO_STAGE_SIZE})
    {
        SCOPED_TRACE(size);

        std::vector<T> values(size);
        for (T& value : values)
            value = static_cast<T>(distribution(generator));

        auto t = this->template create_initialized_tensor<T>({static_cast<int64_t>(size)}, values);
        auto variance = this->engine->create_tensor({1}, get_sushiblas_dtype<T>());
        auto deviation = this->engine->create_tensor({1}, get_sushiblas_dtype<T>());

        SushiBLAS::ReductionOps(*this->engine).var(t, variance);
        SushiBLAS::ReductionOps(*this->engine).std(t, deviation);
        this->engine->execute();

        const long double expected_variance = host_variance(values);
        const long double expected_deviation = std::sqrt(expected_variance);
        const long double got_variance = to_long_double(read_tensor<T>(variance)[0]);
        const long double got_deviation = to_long_double(read_tensor<T>(deviation)[0]);

        EXPECT_LE(std::fabs(got_variance - expected_variance),
                  K_MOMENT_ULPS * ulp_at<T>(expected_variance))
            << "var gave " << got_variance << ", the host " << expected_variance;
        EXPECT_LE(std::fabs(got_deviation - expected_deviation),
                  K_MOMENT_ULPS * ulp_at<T>(expected_deviation))
            << "std gave " << got_deviation << ", the host " << expected_deviation;
    }
}
