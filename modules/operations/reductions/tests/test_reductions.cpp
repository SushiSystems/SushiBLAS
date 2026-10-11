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
 * @brief Tests each reduction's results and refusals, and that its host and SYCL paths agree.
 * @author Mustafa Garip
 */

#include <cmath>
#include <limits>
#include <string>
#include <vector>
#include <random>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <optional>
#include <algorithm>
#include <stdexcept>
#include <functional>
#include <type_traits>
#include <initializer_list>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.h>
#include <SushiBLAS/reductions/dispatch.hpp>
#include "test_utils.hpp"
#include "blas_truth.hpp"
#include "graph_probe.hpp"
#include "reference_reductions.hpp"

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

    // The index is INT64 and is compared exactly. See tests/unit/README.md.
    auto res = this->engine->create_tensor({1}, Core::DataType::Int64);

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
    auto res = this->engine->create_tensor({1}, Core::DataType::Int64);

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

    /// @brief Holds how many rounded operations the SYCL fold of a sum spends on one result.
    constexpr std::size_t K_SUM_ROUNDINGS = 16;

    /// @brief Holds how many rounded operations the SYCL tree of a norm is allowed.
    constexpr std::size_t K_NORM_ROUNDINGS = 64;

    /// @brief Holds how many values the product cases multiply.
    constexpr std::size_t K_PRODUCT_TERMS = 64;

    /// @brief Holds how many epsilons of the element type two logsumexp results may differ by.
    constexpr double K_LOGSUMEXP_PARITY_EPSILONS = 8.0;

    /// @brief Holds the value a result tensor starts from, which no case writes.
    constexpr float K_UNWRITTEN = 7.0f;

    /// @brief Holds the shapes one data set takes as a matrix and as an NHWC image.
    struct ParityShape
    {
        int64_t rows;     /**< The rows of the rank-2 tensor. */
        int64_t columns;  /**< The columns of the rank-2 tensor. */
        int64_t images;   /**< N of the NHWC tensor. */
        int64_t height;   /**< H of the NHWC tensor. */
        int64_t width;    /**< W of the NHWC tensor. */
        int64_t channels; /**< C of the NHWC tensor. */

        /** @brief Returns the element count, which both shapes share. */
        [[nodiscard]] constexpr std::size_t size() const noexcept
        {
            return static_cast<std::size_t>(rows * columns);
        }
    };

    /// @brief Holds the shapes of six elements, the shortest small-integer case.
    constexpr ParityShape K_SIX_ELEMENTS{1, 6, 1, 6, 1, 1};

    /// @brief Holds the shapes of K_SINGLE_PASS_SIZE elements.
    constexpr ParityShape K_SINGLE_PASS_SHAPE{1, 7, 1, 7, 1, 1};

    /// @brief Holds the shapes of a thousand elements, the longest small-integer case HALF holds.
    constexpr ParityShape K_THOUSAND_ELEMENTS{4, 250, 2, 5, 25, 4};

    /// @brief Holds the shapes of K_TWO_STAGE_SIZE elements.
    constexpr ParityShape K_TWO_STAGE_SHAPE{4, 25000, 2, 100, 125, 4};

    static_assert(K_SINGLE_PASS_SHAPE.size() == K_SINGLE_PASS_SIZE &&
                  K_TWO_STAGE_SHAPE.size() == K_TWO_STAGE_SIZE,
                  "The parity shapes must hold the element counts the launch shapes are named for.");

    /// @brief Holds what one reduction wrote under each forced mode.
    template<typename Out>
    struct ModeResults
    {
        std::vector<Out> reference; /**< The result under ForceReference. */
        std::vector<Out> device;    /**< The result under ForceSycl. */
    };

    /// @brief Names one reduction call and the shape of the result it writes.
    struct ReductionCall
    {
        const char* name;                 /**< The call's name in a failure message. */
        std::vector<int64_t> result_dims; /**< The shape of the result tensor. */
        std::function<void(ReductionOps&, Tensor&)> record; /**< Records the call into a result. */
    };

    /// @brief Names one sum-class call and how its bound reads the data.
    struct SumClassCall
    {
        ReductionCall call;                                /**< The call and its result shape. */
        std::function<std::size_t(std::size_t)> result_of; /**< The result element i is folded into. */
        double scale;                                      /**< 1 for a sum, 1 / terms for a mean. */
    };

    /// @brief Holds one data set as the three tensors the reductions fold it through.
    struct ParityOperands
    {
        Tensor flat;   /**< The values as a rank-1 tensor. */
        Tensor matrix; /**< The values as rows by columns. */
        Tensor image;  /**< The values as an NHWC tensor. */
    };

    /** @brief Widens one element to double, which holds every element type the cases use. */
    template<typename T>
    double to_double(T value)
    {
        return static_cast<double>(to_long_double(value));
    }

    /** @brief Narrows a double to T; HALF is reached through FLOAT32, as its constructor takes one. */
    template<typename T>
    T from_double(double value)
    {
        if constexpr (std::is_same_v<T, sycl::half>)
            return sycl::half(static_cast<float>(value));
        else
            return static_cast<T>(value);
    }

    /** @brief Returns size non-integers between 0.125 and 0.5625 in magnitude, every fourth negative. */
    template<typename T>
    std::vector<T> parity_values(std::size_t size)
    {
        std::vector<T> values(size);
        for (std::size_t i = 0; i < size; ++i)
        {
            const double magnitude = 0.125 + 0.015 * static_cast<double>((i * 37) % 29);
            values[i] = from_double<T>(i % 4 == 3 ? -magnitude : magnitude);
        }
        return values;
    }

    /** @brief Returns size integers that walk from -3 to 3 and start again. */
    template<typename T>
    std::vector<T> small_integers(std::size_t size)
    {
        std::vector<T> values(size);
        for (std::size_t i = 0; i < size; ++i)
            values[i] = from_double<T>(static_cast<double>(i % 7) - 3.0);
        return values;
    }

    /** @brief Returns size zeros of alternating sign, element 0 negative. */
    template<typename T>
    std::vector<T> signed_zeros(std::size_t size)
    {
        std::vector<T> values(size);
        for (std::size_t i = 0; i < size; ++i)
            values[i] = from_double<T>(i % 2 == 0 ? -0.0 : 0.0);
        return values;
    }

    /**
     * @brief Returns small_integers with a NaN at index size / 2 and an infinity at size / 3.
     * @param size At least 7, so both marks miss index 0 and each other.
     */
    template<typename T>
    std::vector<T> marked_integers(std::size_t size)
    {
        std::vector<T> values = small_integers<T>(size);
        values[size / 2] = from_double<T>(std::numeric_limits<double>::quiet_NaN());
        values[size / 3] = from_double<T>(std::numeric_limits<double>::infinity());
        return values;
    }

    /** @brief Returns K_PRODUCT_TERMS values within 11/256 of 1, each exact in HALF. */
    template<typename T>
    std::vector<T> near_one_values()
    {
        std::vector<T> values(K_PRODUCT_TERMS);
        for (std::size_t i = 0; i < K_PRODUCT_TERMS; ++i)
            values[i] = from_double<T>(1.0 + (static_cast<double>((i * 37) % 23) - 11.0) / 256.0);
        return values;
    }

    /** @brief Returns K_PRODUCT_TERMS powers of two from 1/4 to 4 whose product is 1/4. */
    template<typename T>
    std::vector<T> powers_of_two()
    {
        std::vector<T> values(K_PRODUCT_TERMS);
        for (std::size_t i = 0; i < K_PRODUCT_TERMS; ++i)
            values[i] = from_double<T>(std::ldexp(1.0, static_cast<int>(i % 5) - 2));
        return values;
    }

    /** @brief Returns the sum of the absolute values of values. */
    template<typename T>
    double magnitude_of(const std::vector<T>& values)
    {
        double magnitude = 0.0;
        for (const T& value : values)
            magnitude += std::fabs(to_double(value));
        return magnitude;
    }

    /** @brief Returns a tensor of dims that holds values in memory order. */
    template<typename T>
    Tensor tensor_of(Engine& engine, const std::vector<int64_t>& dims, const std::vector<T>& values)
    {
        Tensor t = engine.create_tensor(SushiRuntime::span<const int64_t>(dims.data(), dims.size()),
                                        get_sushiblas_dtype<T>());
        SushiBLAS::IO(engine).from_vector(t, values);
        return t;
    }

    /** @brief Returns values as the flat, the matrix and the image tensor of shape. */
    template<typename T>
    ParityOperands operands_of(Engine& engine, const std::vector<T>& values,
                               const ParityShape& shape)
    {
        return ParityOperands{
            tensor_of(engine, {static_cast<int64_t>(shape.size())}, values),
            tensor_of(engine, {shape.rows, shape.columns}, values),
            tensor_of(engine, {shape.images, shape.height, shape.width, shape.channels}, values)};
    }

    /** @brief Records call under each forced mode on engine, runs both and returns what each wrote. */
    template<typename Out>
    ModeResults<Out> in_both_modes(Engine& engine, const ReductionCall& call)
    {
        ModeResults<Out> results;

        for (ExecutionMode mode : {ExecutionMode::ForceReference, ExecutionMode::ForceSycl})
        {
            engine.set_mode(mode);

            Tensor result = engine.create_tensor(
                SushiRuntime::span<const int64_t>(call.result_dims.data(), call.result_dims.size()),
                get_sushiblas_dtype<Out>());
            fill_tensor(result, std::vector<float>(static_cast<std::size_t>(result.num_elements),
                                                   K_UNWRITTEN));

            ReductionOps reductions(engine);
            call.record(reductions, result);
            engine.execute();

            std::vector<Out>& written =
                mode == ExecutionMode::ForceReference ? results.reference : results.device;
            written = read_tensor<Out>(result);
        }

        return results;
    }

    /// @brief Names a ReductionOps member that reduces a whole tensor into a scalar.
    using WholeReduction = TaskHandle (ReductionOps::*)(const Tensor&, Tensor&);

    /// @brief Names a ReductionOps member that reduces one axis of a matrix.
    using AxisReduction = TaskHandle (ReductionOps::*)(const Tensor&, int, Tensor&);

    /** @brief Returns the call of member over t into a one-element result. */
    ReductionCall whole_call(const char* name, WholeReduction member, const Tensor& t)
    {
        return {name, {1}, [member, t](ReductionOps& ops, Tensor& out) { (ops.*member)(t, out); }};
    }

    /**
     * @brief Returns the call of member along axis of matrix.
     * @param kept The extent of the axis that survives, which is the result's length.
     */
    ReductionCall axis_call(const char* name, AxisReduction member, const Tensor& matrix, int axis,
                            int64_t kept)
    {
        return {name, {kept}, [member, matrix, axis](ReductionOps& ops, Tensor& out)
                              {
                                  (ops.*member)(matrix, axis, out);
                              }};
    }

    /** @brief Returns sum, asum, mean, sum_along on both axes and mean_spatial over operands. */
    std::vector<SumClassCall> sum_class_calls(const ParityOperands& operands,
                                              const ParityShape& shape)
    {
        const Tensor image = operands.image;

        const std::size_t columns = static_cast<std::size_t>(shape.columns);
        const std::size_t channels = static_cast<std::size_t>(shape.channels);
        const std::size_t window = static_cast<std::size_t>(shape.height * shape.width);
        const std::size_t per_image = window * channels;

        const auto whole = [](std::size_t) { return std::size_t{0}; };
        const double per_mean = 1.0 / static_cast<double>(shape.size());
        const double per_window = 1.0 / static_cast<double>(window);

        return {
            {whole_call("sum", &ReductionOps::sum, operands.flat), whole, 1.0},
            {whole_call("asum", &ReductionOps::asum, operands.flat), whole, 1.0},
            {whole_call("mean", &ReductionOps::mean, operands.flat), whole, per_mean},
            {axis_call("sum_along, axis 0", &ReductionOps::sum_along, operands.matrix, 0,
                       shape.columns),
             [columns](std::size_t i) { return i % columns; }, 1.0},
            {axis_call("sum_along, axis 1", &ReductionOps::sum_along, operands.matrix, 1,
                       shape.rows),
             [columns](std::size_t i) { return i / columns; }, 1.0},
            {{"mean_spatial", {shape.images, shape.channels},
              [image](ReductionOps& ops, Tensor& out) { ops.mean_spatial(image, out); }},
             [per_image, channels](std::size_t i)
             {
                 return (i / per_image) * channels + i % channels;
             },
             per_window},
        };
    }

    /** @brief Returns max, min, then max_along and min_along on both axes, over operands. */
    std::vector<ReductionCall> ordered_value_calls(const ParityOperands& operands,
                                                   const ParityShape& shape)
    {
        return {
            whole_call("max", &ReductionOps::max, operands.flat),
            whole_call("min", &ReductionOps::min, operands.flat),
            axis_call("max_along, axis 0", &ReductionOps::max_along, operands.matrix, 0,
                      shape.columns),
            axis_call("max_along, axis 1", &ReductionOps::max_along, operands.matrix, 1,
                      shape.rows),
            axis_call("min_along, axis 0", &ReductionOps::min_along, operands.matrix, 0,
                      shape.columns),
            axis_call("min_along, axis 1", &ReductionOps::min_along, operands.matrix, 1,
                      shape.rows),
        };
    }

    /**
     * @brief Returns, per result, the scaled sum of the absolute values folded into it.
     * @param call Its result_of must map every index of values below the result count.
     */
    template<typename T>
    std::vector<double> folded_magnitudes(const std::vector<T>& values, const SumClassCall& call)
    {
        std::size_t results = 1;
        for (int64_t extent : call.call.result_dims)
            results *= static_cast<std::size_t>(extent);

        std::vector<double> magnitudes(results, 0.0);
        for (std::size_t i = 0; i < values.size(); ++i)
            magnitudes[call.result_of(i)] += std::fabs(to_double(values[i])) * call.scale;

        return magnitudes;
    }

    /** @brief Returns whether a and b hold the same bits, a NaN matching a NaN of any payload. */
    template<typename Out>
    bool same_bits(const Out& a, const Out& b)
    {
        if (std::isnan(to_double(a)) && std::isnan(to_double(b)))
            return true;

        return std::memcmp(&a, &b, sizeof(Out)) == 0;
    }

    /** @brief Expects both paths of one reduction to have written the same bits. */
    template<typename Out>
    void expect_equal_bits(const char* name, const ModeResults<Out>& results)
    {
        SCOPED_TRACE(name);
        ASSERT_EQ(results.reference.size(), results.device.size());

        for (std::size_t i = 0; i < results.reference.size(); ++i)
            EXPECT_TRUE(same_bits(results.reference[i], results.device[i]))
                << "result " << i << ": the host path gave " << to_double(results.reference[i])
                << ", the SYCL path " << to_double(results.device[i]);
    }

    /**
     * @brief Expects the two paths of one reduction to differ by no more than their rounding allows.
     * @param roundings Rounded operations the paths spend on one result, in the accumulation type.
     * @param magnitudes Per result, the magnitude those roundings scale by.
     */
    template<typename T>
    void expect_within_rounding(const char* name, const ModeResults<T>& results,
                                std::size_t roundings, const std::vector<double>& magnitudes)
    {
        SCOPED_TRACE(name);
        ASSERT_EQ(magnitudes.size(), results.reference.size());
        ASSERT_EQ(magnitudes.size(), results.device.size());

        const double folding_unit = unit_roundoff<Reduce::AccumulationOf<T>>();
        const double storing_unit = unit_roundoff<T>();

        for (std::size_t i = 0; i < magnitudes.size(); ++i)
        {
            const double reference = to_double(results.reference[i]);
            const double device = to_double(results.device[i]);

            // Each path rounds its result once when it stores it in T.
            const double bound = rounding_bound(roundings, folding_unit, magnitudes[i]) +
                                 storing_unit * (std::fabs(reference) + std::fabs(device));

            EXPECT_LE(std::fabs(device - reference), bound)
                << "result " << i << ": the host path gave " << reference
                << ", the SYCL path " << device;
        }
    }
}

/// @brief Runs the reduction cases that take their own element types.
class Unit_ReductionOps : public SushiBLASTest
{
    protected:
        /** @brief Expects record to throw a std::invalid_argument that names ForceReference. */
        template<typename Record>
        void expect_mode_refusal(const char* name, Record&& record)
        {
            SCOPED_TRACE(name);
            const std::string message = refusal_of(record);
            EXPECT_NE(std::string::npos, message.find("ForceReference")) << message;
        }

        /**
         * @brief Expects record to be refused with one message under both forced modes.
         * @param record Callable `(ReductionOps&)`; it must record nothing.
         */
        template<typename Record>
        void expect_refusal_in_both_modes(const char* name, Record&& record)
        {
            SCOPED_TRACE(name);

            GraphProbe on_host(ExecutionMode::ForceReference);
            GraphProbe on_device(ExecutionMode::ForceSycl);
            ReductionOps host_reductions(on_host.get_recorder());
            ReductionOps device_reductions(on_device.get_recorder());

            const std::string host_message = refusal_of([&] { record(host_reductions); });
            const std::string device_message = refusal_of([&] { record(device_reductions); });

            EXPECT_FALSE(host_message.empty());
            EXPECT_EQ(host_message, device_message);
            EXPECT_TRUE(on_host.take_nodes().empty());
            EXPECT_TRUE(on_device.take_nodes().empty());
        }

        /** @brief Returns a FLOAT32 tensor of dims in device-only USM, or nothing where none can be had. */
        std::optional<Tensor> device_only_tensor(std::initializer_list<int64_t> dims)
        {
            try
            {
                return engine->create_tensor(dims, Core::DataType::Float32,
                                             SushiRuntime::API::Residency::Device);
            }
            catch (const std::exception&)
            {
                return std::nullopt;
            }
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
            auto at_max = engine->create_tensor({1}, Core::DataType::Int64);
            auto at_min = engine->create_tensor({1}, Core::DataType::Int64);

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

    for (Core::DataType dtype : {Core::DataType::Int32, Core::DataType::Int64})
    {
        SCOPED_TRACE(Core::to_string(dtype));

        auto t = engine->create_tensor({4}, dtype);
        auto result = engine->create_tensor({1}, dtype);
        const std::string name = Core::to_string(dtype);

        EXPECT_NE(std::string::npos, refusal_of([&] { reductions.sum(t, result); }).find(name));
        EXPECT_NE(std::string::npos, refusal_of([&] { reductions.nrm2(t, result); }).find(name));
        EXPECT_NE(std::string::npos, refusal_of([&] { reductions.mean(t, result); }).find(name));
    }

    auto z = engine->create_tensor({4}, Core::DataType::Complex32);
    auto z_result = engine->create_tensor({1}, Core::DataType::Complex32);
    const std::string complex_name = Core::to_string(Core::DataType::Complex32);

    EXPECT_NE(std::string::npos,
              refusal_of([&] { reductions.sum(z, z_result); }).find(complex_name));
    EXPECT_NE(std::string::npos,
              refusal_of([&] { reductions.max(z, z_result); }).find(complex_name));
}

/**
 * @brief Tests that a device-only operand is refused under ForceReference on every reduction seam.
 */
TEST_F(Unit_ReductionOps, ADeviceOnlyOperandIsRefusedOnEverySeam)
{
    std::optional<Tensor> t = device_only_tensor({4});
    std::optional<Tensor> matrix = device_only_tensor({2, 4});
    std::optional<Tensor> image = device_only_tensor({1, 2, 2, 3});
    if (!t || !matrix || !image)
        GTEST_SKIP() << "the device hands out no device-only USM";

    auto scalar = engine->create_tensor({1});
    auto index = engine->create_tensor({1}, Core::DataType::Int64);
    auto per_row = engine->create_tensor({2});
    auto per_channel = engine->create_tensor({1, 3});

    GraphProbe on_host(ExecutionMode::ForceReference);
    ReductionOps refused(on_host.get_recorder());

    // One call per seam: folding, centred, index, counting, and the span seam twice.
    expect_mode_refusal("sum", [&] { refused.sum(*t, scalar); });
    expect_mode_refusal("var", [&] { refused.var(*t, scalar); });
    expect_mode_refusal("argmax", [&] { refused.argmax(*t, index); });
    expect_mode_refusal("count_nonfinite", [&] { refused.count_nonfinite(*t, index); });
    expect_mode_refusal("sum_along", [&] { refused.sum_along(*matrix, 1, per_row); });
    expect_mode_refusal("mean_spatial", [&] { refused.mean_spatial(*image, per_channel); });
    EXPECT_TRUE(on_host.take_nodes().empty());

    GraphProbe on_device(ExecutionMode::ForceSycl);
    ReductionOps recorded(on_device.get_recorder());

    EXPECT_NO_THROW(recorded.sum(*t, scalar));
    EXPECT_NO_THROW(recorded.var(*t, scalar));
    EXPECT_NO_THROW(recorded.argmax(*t, index));
    EXPECT_NO_THROW(recorded.count_nonfinite(*t, index));
    EXPECT_NO_THROW(recorded.sum_along(*matrix, 1, per_row));
    EXPECT_NO_THROW(recorded.mean_spatial(*image, per_channel));
    EXPECT_EQ(6u, on_device.take_nodes().size());
}

/**
 * @brief Tests that a malformed operand is refused with the same message under both forced modes.
 */
TEST_F(Unit_ReductionOps, MalformedOperandsAreRefusedAlikeInBothModes)
{
    auto vector = engine->create_tensor({4});
    auto wide = engine->create_tensor({3});

    // A rank-1 operand is malformed for every axis reduction.
    expect_refusal_in_both_modes("sum_along",
                                 [&](ReductionOps& ops) { ops.sum_along(vector, 0, wide); });
    expect_refusal_in_both_modes("max_along",
                                 [&](ReductionOps& ops) { ops.max_along(vector, 0, wide); });

    // A three-element result is not a scalar.
    expect_refusal_in_both_modes("sum", [&](ReductionOps& ops) { ops.sum(vector, wide); });
    expect_refusal_in_both_modes("argmax", [&](ReductionOps& ops) { ops.argmax(vector, wide); });
}

/**
 * @brief Tests that an empty operand writes the identity or is refused, alike under both forced modes.
 */
TEST_F(Unit_ReductionOps, EmptyOperandsBehaveAlikeInBothModes)
{
    const Tensor empty(SushiRuntime::sushi_ptr<Storage>(), {0, 3});
    auto scalar = engine->create_tensor({1});

    const ModeResults<float> sums = in_both_modes<float>(
        *engine, whole_call("sum", &ReductionOps::sum, empty));
    const ModeResults<float> products = in_both_modes<float>(
        *engine, whole_call("prod", &ReductionOps::prod, empty));

    // Each result started from K_UNWRITTEN, so an identity here was written by the fold.
    EXPECT_TRUE(same_bits(0.0f, sums.reference[0])) << sums.reference[0];
    EXPECT_TRUE(same_bits(0.0f, sums.device[0])) << sums.device[0];
    EXPECT_TRUE(same_bits(1.0f, products.reference[0])) << products.reference[0];
    EXPECT_TRUE(same_bits(1.0f, products.device[0])) << products.device[0];

    expect_refusal_in_both_modes("mean", [&](ReductionOps& ops) { ops.mean(empty, scalar); });
    expect_refusal_in_both_modes("max", [&](ReductionOps& ops) { ops.max(empty, scalar); });
}

/**
 * @brief Tests that a reduction recorded under ForceReference leases no scratch and allocates none.
 */
TEST_F(Unit_ReductionOps, AReferenceReductionLeasesNoScratch)
{
    auto t = create_initialized_tensor<float>({static_cast<int64_t>(K_TWO_STAGE_SIZE)},
                                              parity_values<float>(K_TWO_STAGE_SIZE));
    if (Dispatch::planned_partials_for(t) == 0)
        GTEST_SKIP() << "the device plans no partials, so the SYCL path would lease none either";

    auto total = engine->create_tensor({1});
    auto variance = engine->create_tensor({1});
    auto index = engine->create_tensor({1}, Core::DataType::Int64);
    auto count = engine->create_tensor({1}, Core::DataType::Int64);

    GraphProbe probe(ExecutionMode::ForceReference);
    ReductionOps reductions(probe.get_recorder());

    auto& context = TestEnvironment::get_context();
    const std::size_t idle_before = probe.get_scratch_pool().idle_block_count();
    const std::size_t live_before = context.memory().live_count();

    reductions.sum(t, total);
    reductions.var(t, variance);
    reductions.argmax(t, index);
    reductions.count_nonfinite(t, count);

    // A lease would be a live allocation from the moment its task is recorded.
    EXPECT_EQ(live_before, context.memory().live_count());

    probe.execute();

    EXPECT_EQ(idle_before, probe.get_scratch_pool().idle_block_count());
    EXPECT_EQ(live_before, context.memory().live_count());
    EXPECT_EQ(0, read_tensor<int64_t>(count)[0]);
}

/**
 * @brief Tests that the host mean of FLOAT32 terms divides its FLOAT64 sum before it narrows.
 */
TEST_F(Unit_ReductionOps, AHostMeanDividesBeforeItNarrows)
{
    // The sum is 25165831, which FLOAT32 rounds to 25165832; a third of each rounds differently.
    const std::vector<float> values = {16777216.0f, 8388608.0f, 7.0f};
    const float divided_first = 8388610.0f;
    const float narrowed_first = 8388611.0f;

    engine->set_mode(ExecutionMode::ForceReference);

    auto t = create_initialized_tensor<float>({3}, values);
    auto image = create_initialized_tensor<float>({1, 3, 1, 1}, values);
    auto mean = engine->create_tensor({1});
    auto spatial_mean = engine->create_tensor({1, 1});

    ReductionOps(*engine).mean(t, mean);
    ReductionOps(*engine).mean_spatial(image, spatial_mean);
    engine->execute();

    const float got_mean = read_tensor<float>(mean)[0];
    const float got_spatial_mean = read_tensor<float>(spatial_mean)[0];

    EXPECT_TRUE(same_bits(divided_first, got_mean))
        << "mean gave " << got_mean << "; narrowing first gives " << narrowed_first;
    EXPECT_TRUE(same_bits(divided_first, got_spatial_mean))
        << "mean_spatial gave " << got_spatial_mean << "; narrowing first gives " << narrowed_first;
}

/**
 * @brief Tests that argmax and argmin of integers elect the same index on both paths.
 */
TEST_F(Unit_ReductionOps, IndexReductionsOfIntegersAgreeToTheBit)
{
    for (std::size_t size : {K_SINGLE_PASS_SIZE, K_TWO_STAGE_SIZE})
    {
        SCOPED_TRACE(size);

        const Tensor narrow = tensor_of(*engine, {static_cast<int64_t>(size)},
                                        small_integers<int32_t>(size));
        const Tensor wide = tensor_of(*engine, {static_cast<int64_t>(size)},
                                      small_integers<int64_t>(size));

        for (const Tensor& t : {narrow, wide})
        {
            const ModeResults<int64_t> at_max = in_both_modes<int64_t>(
                *engine, whole_call("argmax", &ReductionOps::argmax, t));
            const ModeResults<int64_t> at_min = in_both_modes<int64_t>(
                *engine, whole_call("argmin", &ReductionOps::argmin, t));

            expect_equal_bits("argmax", at_max);
            expect_equal_bits("argmin", at_min);

            // The walk from -3 to 3 first peaks at index 6 and first bottoms at index 0.
            EXPECT_EQ(6, at_max.reference[0]);
            EXPECT_EQ(0, at_min.reference[0]);
        }
    }
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

/// @brief Runs each parity case once per real element type, on both paths of a reduction.
template<typename T>
class Unit_ReductionReferenceTyped : public SushiBLASTest
{
    protected:
        /** @brief Builds the engine and skips a HALF case on a device without fp16. */
        void SetUp() override
        {
            SushiBLASTest::SetUp();

            const sycl::device device = TestEnvironment::get_context().devices().queue(0).get_device();
            if (std::is_same_v<T, sycl::half> && !device.has(sycl::aspect::fp16))
                GTEST_SKIP() << "the device does not report the fp16 aspect";
        }
};

/// @brief Names the element types the parity cases run over.
using ParityTypes = ::testing::Types<sycl::half, float, double>;
SB_TYPED_TEST_SUITE(Unit_ReductionReferenceTyped, ParityTypes);

/**
 * @brief Tests that the SYCL result of each sum is within the rounding bound of its host result.
 */
TYPED_TEST(Unit_ReductionReferenceTyped, SumsAgreeWithTheirReferencePathWithinTheBound)
{
    using T = TypeParam;
    Engine& engine = *this->engine;

    for (const ParityShape& shape : {K_SINGLE_PASS_SHAPE, K_TWO_STAGE_SHAPE})
    {
        SCOPED_TRACE(shape.size());

        const std::vector<T> values = parity_values<T>(shape.size());
        const ParityOperands operands = operands_of(engine, values, shape);

        for (const SumClassCall& sum : sum_class_calls(operands, shape))
        {
            const ModeResults<T> results = in_both_modes<T>(engine, sum.call);
            expect_within_rounding(sum.call.name, results, K_SUM_ROUNDINGS,
                                   folded_magnitudes(values, sum));
        }
    }
}

/**
 * @brief Tests that sums of small integers, which no path rounds, come out with equal bits.
 */
TYPED_TEST(Unit_ReductionReferenceTyped, SumsOfSmallIntegersAgreeToTheBit)
{
    using T = TypeParam;
    Engine& engine = *this->engine;

    const double exact_integers = std::ldexp(1.0, std::numeric_limits<T>::digits);

    for (const ParityShape& shape : {K_SIX_ELEMENTS, K_THOUSAND_ELEMENTS, K_TWO_STAGE_SHAPE})
    {
        SCOPED_TRACE(shape.size());

        const std::vector<T> values = small_integers<T>(shape.size());

        // T holds every integer below 2^digits, so HALF leaves the longest tensor to the wider types.
        if (magnitude_of(values) >= exact_integers)
            continue;

        const ParityOperands operands = operands_of(engine, values, shape);

        for (const SumClassCall& sum : sum_class_calls(operands, shape))
            expect_equal_bits(sum.call.name, in_both_modes<T>(engine, sum.call));
    }
}

/**
 * @brief Tests that the reductions whose result does not depend on the fold order agree to the bit.
 */
TYPED_TEST(Unit_ReductionReferenceTyped, OrderFreeReductionsAgreeToTheBit)
{
    using T = TypeParam;
    Engine& engine = *this->engine;

    for (const ParityShape& shape : {K_SINGLE_PASS_SHAPE, K_TWO_STAGE_SHAPE})
    {
        SCOPED_TRACE(shape.size());

        const std::size_t size = shape.size();
        const ParityOperands zeros = operands_of(engine, signed_zeros<T>(size), shape);
        const ParityOperands marked = operands_of(engine, marked_integers<T>(size), shape);

        for (const ParityOperands& operands : {zeros, marked})
        {
            for (const ReductionCall& call : ordered_value_calls(operands, shape))
                expect_equal_bits(call.name, in_both_modes<T>(engine, call));

            for (const ReductionCall& call :
                 {whole_call("argmax", &ReductionOps::argmax, operands.flat),
                  whole_call("argmin", &ReductionOps::argmin, operands.flat),
                  whole_call("count_nonfinite", &ReductionOps::count_nonfinite, operands.flat)})
                expect_equal_bits(call.name, in_both_modes<int64_t>(engine, call));
        }

        // +0 is the maximum of zeros of both signs and -0 their minimum.
        const std::vector<ReductionCall> extremes = ordered_value_calls(zeros, shape);
        EXPECT_FALSE(std::signbit(to_double(in_both_modes<T>(engine, extremes[0]).reference[0])));
        EXPECT_TRUE(std::signbit(to_double(in_both_modes<T>(engine, extremes[1]).reference[0])));

        // The NaN makes the maximum NaN; the infinity is the largest element, the first -3 the smallest.
        const ModeResults<T> largest = in_both_modes<T>(
            engine, whole_call("max", &ReductionOps::max, marked.flat));
        const ModeResults<int64_t> at_largest = in_both_modes<int64_t>(
            engine, whole_call("argmax", &ReductionOps::argmax, marked.flat));
        const ModeResults<int64_t> at_smallest = in_both_modes<int64_t>(
            engine, whole_call("argmin", &ReductionOps::argmin, marked.flat));
        const ModeResults<int64_t> non_finite = in_both_modes<int64_t>(
            engine, whole_call("count_nonfinite", &ReductionOps::count_nonfinite, marked.flat));

        EXPECT_TRUE(std::isnan(to_double(largest.reference[0])));
        EXPECT_EQ(static_cast<int64_t>(size / 3), at_largest.reference[0]);
        EXPECT_EQ(0, at_smallest.reference[0]);
        EXPECT_EQ(2, non_finite.reference[0]);
    }
}

/**
 * @brief Tests that var and std of the two paths differ by no more than both allowances together.
 */
TYPED_TEST(Unit_ReductionReferenceTyped, MomentsAgreeWithTheirReferencePath)
{
    using T = TypeParam;
    Engine& engine = *this->engine;

    std::mt19937 generator(K_MOMENT_SEED);
    std::uniform_real_distribution<float> distribution(-10.0f, 10.0f);

    for (std::size_t size : {std::size_t{5}, std::size_t{1000}, K_TWO_STAGE_SIZE})
    {
        SCOPED_TRACE(size);

        std::vector<T> values(size);
        for (T& value : values)
            value = static_cast<T>(distribution(generator));

        const Tensor t = tensor_of(engine, {static_cast<int64_t>(size)}, values);

        for (const ReductionCall& call : {whole_call("var", &ReductionOps::var, t),
                                          whole_call("std", &ReductionOps::std, t)})
        {
            SCOPED_TRACE(call.name);

            const ModeResults<T> results = in_both_modes<T>(engine, call);
            const long double reference = to_long_double(results.reference[0]);
            const long double device = to_long_double(results.device[0]);

            // Each path is allowed K_MOMENT_ULPS against the two-pass truth.
            EXPECT_LE(std::fabs(device - reference),
                      2.0L * K_MOMENT_ULPS * ulp_at<T>(reference))
                << "the host path gave " << reference << ", the SYCL path " << device;
        }
    }
}

/**
 * @brief Tests that prod and nrm2 of the two paths stay within their rounding allowances.
 */
TYPED_TEST(Unit_ReductionReferenceTyped, ProductAndNormAgreeWithinTheirBounds)
{
    using T = TypeParam;
    Engine& engine = *this->engine;

    const std::vector<int64_t> terms = {static_cast<int64_t>(K_PRODUCT_TERMS)};
    const Tensor near_one = tensor_of(engine, terms, near_one_values<T>());
    const Tensor powers = tensor_of(engine, terms, powers_of_two<T>());

    // Each path multiplies the values in its own order and may round once per value.
    const ModeResults<T> rounded = in_both_modes<T>(
        engine, whole_call("prod", &ReductionOps::prod, near_one));
    expect_within_rounding("prod", rounded, 2 * K_PRODUCT_TERMS,
                           {std::fabs(to_double(rounded.reference[0]))});

    // A product of powers of two is exact in every order.
    const ModeResults<T> exact = in_both_modes<T>(
        engine, whole_call("prod", &ReductionOps::prod, powers));
    expect_equal_bits("prod of powers of two", exact);
    EXPECT_TRUE(same_bits(from_double<T>(0.25), exact.reference[0]))
        << to_double(exact.reference[0]);

    for (std::size_t size : {K_SINGLE_PASS_SIZE, K_TWO_STAGE_SIZE})
    {
        SCOPED_TRACE(size);

        const Tensor t = tensor_of(engine, {static_cast<int64_t>(size)}, parity_values<T>(size));
        const ModeResults<T> norms = in_both_modes<T>(
            engine, whole_call("nrm2", &ReductionOps::nrm2, t));

        expect_within_rounding("nrm2", norms, K_NORM_ROUNDINGS,
                               {std::fabs(to_double(norms.reference[0]))});
    }
}

/**
 * @brief Tests that logsumexp of the two paths agrees, on finite terms and on its special values.
 */
TYPED_TEST(Unit_ReductionReferenceTyped, LogSumExpAgreesWithItsReferencePath)
{
    using T = TypeParam;
    Engine& engine = *this->engine;

    const double epsilon = to_double(std::numeric_limits<T>::epsilon());

    for (std::size_t size : {K_SINGLE_PASS_SIZE, K_TWO_STAGE_SIZE})
    {
        SCOPED_TRACE(size);

        const Tensor t = tensor_of(engine, {static_cast<int64_t>(size)}, parity_values<T>(size));
        const ModeResults<T> results = in_both_modes<T>(
            engine, whole_call("logsumexp", &ReductionOps::logsumexp, t));

        const double reference = to_double(results.reference[0]);
        const double device = to_double(results.device[0]);

        EXPECT_LE(std::fabs(device - reference),
                  K_LOGSUMEXP_PARITY_EPSILONS * epsilon * std::max(1.0, std::fabs(reference)))
            << "the host path gave " << reference << ", the SYCL path " << device;
    }

    const double infinity = std::numeric_limits<double>::infinity();
    const double not_a_number = std::numeric_limits<double>::quiet_NaN();

    const auto of_pair = [&engine](double first, double second)
    {
        const Tensor t = tensor_of(engine, {2},
                                   std::vector<T>{from_double<T>(first), from_double<T>(second)});
        return in_both_modes<T>(engine, whole_call("logsumexp", &ReductionOps::logsumexp, t));
    };

    const ModeResults<T> above = of_pair(0.0, infinity);
    const ModeResults<T> below = of_pair(-infinity, -infinity);
    const ModeResults<T> undefined = of_pair(1.0, not_a_number);

    expect_equal_bits("{0, +inf}", above);
    expect_equal_bits("{-inf, -inf}", below);
    expect_equal_bits("{1, NaN}", undefined);

    EXPECT_TRUE(std::isinf(to_double(above.reference[0])) && to_double(above.reference[0]) > 0.0);
    EXPECT_TRUE(std::isinf(to_double(below.reference[0])) && to_double(below.reference[0]) < 0.0);
    EXPECT_TRUE(std::isnan(to_double(undefined.reference[0])));
}
