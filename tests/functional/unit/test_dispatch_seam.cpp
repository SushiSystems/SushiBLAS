/**************************************************************************/
/* test_dispatch_seam.cpp                                                 */
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
 * @file test_dispatch_seam.cpp
 * @brief Tests the public Dispatch seam with functors written outside the library.
 * @author Mustafa Garip
 */

#include <array>
#include <vector>
#include <cstdint>
#include <cstring>
#include <string>
#include <stdexcept>
#include <type_traits>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.h>
#include <SushiBLAS/engine/dispatch.hpp>
#include <SushiBLAS/engine/task.hpp>
#include <SushiBLAS/engine/logic/dispatch.hpp>
#include <SushiBLAS/engine/math/safe_math.hpp>
#include <SushiBLAS/engine/math/elementwise/dispatch.hpp>
#include <SushiBLAS/engine/math/random/dispatch.hpp>
#include <SushiBLAS/engine/math/reductions/dispatch.hpp>
#include "../../common/test_utils.hpp"

namespace sb = SushiBLAS;

using namespace SushiBLAS::Test;
using namespace SushiRuntime::Graph::Literals;
using SushiBLAS::Core::DataType;

namespace
{
    /**
     * @brief Returns the bit pattern of a float.
     * @see tests/functional/unit/README.md
     */
    uint32_t float_bits(float value) noexcept
    {
        uint32_t bits = 0;
        std::memcpy(&bits, &value, sizeof(bits));
        return bits;
    }
} // namespace

class Unit_DispatchSeamTest : public SushiBLASTest {};

TEST_F(Unit_DispatchSeamTest, FusesAConsumerFunctorIntoOneBinaryKernel)
{
    auto a = engine->create_tensor({4});
    auto b = engine->create_tensor({4});
    auto c = engine->create_tensor({4});
    fill_tensor(a, {1.0f, 2.0f, 3.0f, 4.0f});
    fill_tensor(b, {2.0f, 4.0f, 6.0f, 8.0f});

    // mul, add and div in one functor: one task where the public API records three.
    sb::Dispatch::execute_binary(*engine, a, b, c, "test.fused_binary", "test.fused_binary"_op,
        [](auto x, auto y) { return (x * y + decltype(x)(1)) / decltype(x)(2); });
    engine->execute().wait();

    verify_tensor(c, {1.5f, 4.5f, 9.5f, 16.5f});
}

TEST_F(Unit_DispatchSeamTest, FusesAConsumerFunctorIntoOneUnaryKernel)
{
    auto t = engine->create_tensor({4});
    fill_tensor(t, {0.0f, 1.0f, 2.0f, 3.0f});

    // A consumer's own softplus-like activation, written against safe_math.
    sb::Dispatch::execute_unary_inplace(*engine, t, "test.fused_unary", "test.fused_unary"_op,
        [](auto x) { return sb::Dispatch::safe_log(decltype(x)(1) + sb::Dispatch::safe_exp(x)); });
    engine->execute().wait();

    verify_tensor(t, {0.693147f, 1.313262f, 2.126928f, 3.048587f}, 1e-4);
}

TEST_F(Unit_DispatchSeamTest, ConsumerFunctorCanOptIntoComplex)
{
    using cf = std::complex<float>;

    auto a = create_initialized_tensor<cf>({2}, {cf(1.0f, 1.0f), cf(2.0f, -1.0f)});
    auto b = create_initialized_tensor<cf>({2}, {cf(0.0f, 1.0f), cf(1.0f, 1.0f)});
    auto c = engine->create_tensor({2}, DataType::COMPLEX32);

    // (1+i)(i) + (1+i) = 2i and (2-i)(1+i) + (2-i) = 5.
    sb::Dispatch::execute_binary<sb::Dispatch::Domain::REAL_AND_COMPLEX>(
        *engine, a, b, c, "test.fused_complex", "test.fused_complex"_op,
        [](auto x, auto y) { return x * y + x; });
    engine->execute().wait();

    auto expected = create_initialized_tensor<cf>({2}, {cf(0.0f, 2.0f), cf(5.0f, 0.0f)});
    SB_ASSERT_TENSOR_NEAR(c, expected, 1e-4, cf);
}

TEST_F(Unit_DispatchSeamTest, ConsumerCanDriveTheLogicAndReductionSeamsToo)
{
    auto a = engine->create_tensor({4});
    auto b = engine->create_tensor({4});
    auto mask = engine->create_tensor({4});
    fill_tensor(a, {1.0f, 5.0f, 2.0f, 9.0f});
    fill_tensor(b, {3.0f, 3.0f, 3.0f, 3.0f});

    // "within 2 of b" - a predicate SushiBLAS does not ship.
    sb::Dispatch::execute_logic_binary(*engine, a, b, mask, "test.near", "test.near"_op,
        [](auto x, auto y)
        {
            auto d = x - y;
            return (d < decltype(x)(0) ? -d : d) <= decltype(x)(2) ? decltype(x)(1) : decltype(x)(0);
        });
    engine->execute().wait();
    verify_tensor(mask, {1.0f, 1.0f, 1.0f, 0.0f});

    auto total = engine->create_tensor({1});
    sb::Dispatch::execute_reduction(*engine, mask, total, "test.count", "test.count"_op,
        [](sycl::queue& q, const auto* pT, auto* pR, int64_t size,
           [[maybe_unused]] auto scratch,
           const std::vector<sycl::event>& deps)
        {
            using T = std::decay_t<decltype(*pT)>;

            return q.submit([&](sycl::handler& h)
            {
                h.depends_on(deps);
                h.parallel_for(sycl::range<1>(size),
                    sycl::reduction(pR, T(0), sycl::plus<T>(), sycl::property::reduction::initialize_to_identity{}),
                    [=](sycl::id<1> idx, auto& reducer) { reducer.combine(pT[idx[0]]); });
            });
        });
    engine->execute().wait();
    verify_tensor(total, {3.0f});
}

TEST_F(Unit_DispatchSeamTest, RealOnlyDomainRejectsComplexAtRecordTime)
{
    auto a = engine->create_tensor({2}, DataType::COMPLEX32);
    auto b = engine->create_tensor({2}, DataType::COMPLEX32);
    auto c = engine->create_tensor({2}, DataType::COMPLEX32);

    // The default domain is REAL. See tests/functional/unit/README.md.
    EXPECT_THROW(
        sb::Dispatch::execute_binary(*engine, a, b, c, "test.real_only", "test.real_only"_op,
                                     [](auto x, auto y) { return x + y; }),
        std::runtime_error);
}

TEST_F(Unit_DispatchSeamTest, FusesAFiveTensorAdamWUpdateIntoOneKernel)
{
    // 1023 is a multiple of no work-group or vector width. See tests/functional/unit/README.md.
    constexpr int64_t k_n = 1023;

    const float b1 = 0.9f;
    const float b2 = 0.999f;
    const float lr = 0.01f;
    const float wd = 0.01f;
    const float eps = 1e-8f;
    const float ema_decay = 0.99f;
    const float inv_bc1 = 1.0f / (1.0f - b1);   // bias correction at step 1
    const float inv_bc2 = 1.0f / (1.0f - b2);

    std::vector<float> g0(k_n), th0(k_n), m0(k_n), v0(k_n);
    for (int64_t i = 0; i < k_n; ++i)
    {
        g0[static_cast<size_t>(i)]  = 0.01f * static_cast<float>((i % 17) - 8);
        th0[static_cast<size_t>(i)] = 0.001f * static_cast<float>((i % 23) - 11);
        m0[static_cast<size_t>(i)]  = 0.002f * static_cast<float>((i % 13) - 6);
        v0[static_cast<size_t>(i)]  = 1e-4f * static_cast<float>((i % 7) + 1);
    }

    auto seeded = [&](const std::vector<float>& values)
    {
        auto t = engine->create_tensor({k_n});
        fill_tensor(t, values);
        return t;
    };

    auto g = seeded(g0);
    auto th_fused = seeded(th0), m_fused = seeded(m0), v_fused = seeded(v0), ema_fused = seeded(th0);
    auto th_steps = seeded(th0), m_steps = seeded(m0), v_steps = seeded(v0), ema_steps = seeded(th0);

    // g is an operand although the functor only reads it. See tests/functional/unit/README.md.
    sb::Dispatch::execute_zip_inplace(*engine,
        std::array{&th_fused, &g, &m_fused, &v_fused, &ema_fused},
        "test.adamw_fused", "test.adamw_fused"_op,
        [b1, b2, lr, wd, eps, inv_bc1, inv_bc2, ema_decay]
        (auto& theta, auto& grad, auto& m, auto& v, auto& ema)
        {
            using T = std::decay_t<decltype(theta)>;
            m = T(b1) * m + T(1.0f - b1) * grad;
            v = T(b2) * v + T(1.0f - b2) * (grad * grad);
            const T denom = sb::Dispatch::safe_sqrt(v * T(inv_bc2)) + T(eps);
            theta = theta * T(1.0f - lr * wd) - T(lr) * ((m * T(inv_bc1)) / denom);
            ema = T(ema_decay) * ema + T(1.0f - ema_decay) * theta;
        });

    // The same arithmetic as fifteen library tasks. See tests/functional/unit/README.md.
    auto scratch_a = engine->create_tensor({k_n});
    auto scratch_b = engine->create_tensor({k_n});
    auto eps_vec = engine->create_tensor({k_n});
    fill_tensor(eps_vec, std::vector<float>(static_cast<size_t>(k_n), eps));

    engine->blas().scal(static_cast<double>(b1), m_steps);
    engine->blas().axpy(static_cast<double>(1.0f - b1), g, m_steps);

    engine->blas().copy(g, scratch_a);
    engine->elementwise().square(scratch_a);
    engine->blas().scal(static_cast<double>(b2), v_steps);
    engine->blas().axpy(static_cast<double>(1.0f - b2), scratch_a, v_steps);

    engine->blas().copy(v_steps, scratch_a);
    engine->blas().scal(static_cast<double>(inv_bc2), scratch_a);
    engine->elementwise().sqrt(scratch_a);
    engine->elementwise().add(scratch_a, eps_vec, scratch_a);

    engine->blas().copy(m_steps, scratch_b);
    engine->blas().scal(static_cast<double>(inv_bc1), scratch_b);
    engine->elementwise().div(scratch_b, scratch_a, scratch_b);

    engine->blas().scal(static_cast<double>(1.0f - lr * wd), th_steps);
    engine->blas().axpy(static_cast<double>(-lr), scratch_b, th_steps);

    engine->blas().scal(static_cast<double>(ema_decay), ema_steps);
    engine->blas().axpy(static_cast<double>(1.0f - ema_decay), th_steps, ema_steps);

    engine->execute().wait();

    SB_ASSERT_TENSOR_NEAR(m_fused, m_steps, 1e-6, float);
    SB_ASSERT_TENSOR_NEAR(v_fused, v_steps, 1e-6, float);
    SB_ASSERT_TENSOR_NEAR(th_fused, th_steps, 1e-6, float);
    SB_ASSERT_TENSOR_NEAR(ema_fused, ema_steps, 1e-6, float);

    // A read-only operand comes back bit for bit. See tests/functional/unit/README.md.
    verify_tensor(g, g0, 0.0);
}

TEST_F(Unit_DispatchSeamTest, FusedInPlaceUpdateIsRepeatableToTheByte)
{
    // Two identical fused updates must agree exactly. See tests/functional/unit/README.md.
    constexpr int64_t k_n = 517;
    std::vector<float> seed(static_cast<size_t>(k_n));
    for (int64_t i = 0; i < k_n; ++i) seed[static_cast<size_t>(i)] = 0.001f * static_cast<float>((i % 31) - 15);

    auto run = [&]()
    {
        auto a = engine->create_tensor({k_n});
        auto b = engine->create_tensor({k_n});
        auto c = engine->create_tensor({k_n});
        fill_tensor(a, seed);
        fill_tensor(b, seed);
        fill_tensor(c, seed);

        sb::Dispatch::execute_zip_inplace(*engine, std::array{&a, &b, &c},
            "test.zip_determinism", "test.zip_determinism"_op,
            [](auto& x, auto& y, auto& z)
            {
                using T = std::decay_t<decltype(x)>;
                x = x * T(1.0000001) + y * T(0.3333333) + z * T(0.1666667);
                y = y - x * T(0.25);
                z = sb::Dispatch::safe_sqrt(x * x + y * y);
            });
        engine->execute().wait();

        std::vector<float> out(static_cast<size_t>(k_n) * 3);
        auto& q = SushiBLAS::Test::TestEnvironment::get_context().get_queue();
        q.memcpy(out.data(), a.data(), static_cast<size_t>(k_n) * sizeof(float)).wait();
        q.memcpy(out.data() + k_n, b.data(), static_cast<size_t>(k_n) * sizeof(float)).wait();
        q.memcpy(out.data() + 2 * k_n, c.data(), static_cast<size_t>(k_n) * sizeof(float)).wait();
        return out;
    };

    const std::vector<float> first = run();
    const std::vector<float> second = run();

    ASSERT_EQ(first.size(), second.size());
    for (size_t i = 0; i < first.size(); ++i)
    {
        EXPECT_EQ(float_bits(first[i]), float_bits(second[i]))
            << "Fused update was not bit-reproducible at index " << i;
    }
}

TEST_F(Unit_DispatchSeamTest, FusedInPlaceUpdateRejectsDisagreeingOperands)
{
    auto four = engine->create_tensor({4});
    auto eight = engine->create_tensor({8});
    auto wide = engine->create_tensor({4}, DataType::FLOAT64);
    auto cplx = engine->create_tensor({4}, DataType::COMPLEX32);
    sb::Tensor* missing = nullptr;

    auto accumulate = [](auto& x, auto& y) { x = x + y; };

    // The message names the disagreeing operand; "sizes must match" is no help at N = 5.
    EXPECT_THROW(sb::Dispatch::execute_zip_inplace(*engine, std::array{&four, &eight},
                     "test.zip_size", "test.zip_size"_op, accumulate),
                 std::invalid_argument);

    EXPECT_THROW(sb::Dispatch::execute_zip_inplace(*engine, std::array{&four, &wide},
                     "test.zip_dtype", "test.zip_dtype"_op, accumulate),
                 std::runtime_error);

    EXPECT_THROW(sb::Dispatch::execute_zip_inplace(*engine, std::array{&four, missing},
                     "test.zip_null", "test.zip_null"_op, accumulate),
                 std::runtime_error);

    // Default domain is REAL, exactly as for execute_unary_inplace/execute_binary.
    EXPECT_THROW(sb::Dispatch::execute_zip_inplace(*engine, std::array{&cplx, &cplx},
                     "test.zip_complex", "test.zip_complex"_op, accumulate),
                 std::runtime_error);
}

TEST_F(Unit_DispatchSeamTest, RngMetadataSlotsAreInRangeAndRoundTrip)
{
    // The seed and offset slots read back exactly. See tests/functional/unit/README.md.
    constexpr uint64_t kSeed = 0xDEADBEEFCAFEF00Dull;
    constexpr uint64_t kOffset = 42;

    static_assert(sb::Dispatch::K_RNG_SEED_PARAM < static_cast<int>(SushiRuntime::Graph::TaskMetadata::K_MAX_PARAMS));
    static_assert(sb::Dispatch::K_RNG_OFFSET_PARAM < static_cast<int>(SushiRuntime::Graph::TaskMetadata::K_MAX_PARAMS));

    SushiRuntime::Graph::TaskMetadata meta;
    sb::Dispatch::set_rng_params(meta, kSeed, kOffset);

    EXPECT_EQ(meta.get_param<uint64_t>(sb::Dispatch::K_RNG_SEED_PARAM), kSeed);
    EXPECT_EQ(meta.get_param<uint64_t>(sb::Dispatch::K_RNG_OFFSET_PARAM), kOffset);
}

TEST_F(Unit_DispatchSeamTest, DistributionParamsDoNotCollideWithTheRngSlots)
{
    SushiRuntime::Graph::TaskMetadata meta;
    const std::vector<double> params = {1.0, 2.0, 3.0, 4.0};

    sb::Dispatch::set_task_params(meta, params, "test.params",
                                  SushiRuntime::Graph::TaskMetadata::K_MAX_PARAMS - sb::Dispatch::K_MAX_RNG_PARAMS);
    sb::Dispatch::set_rng_params(meta, 7, 8);

    for (size_t i = 0; i < params.size(); ++i)
    {
        EXPECT_DOUBLE_EQ(meta.get_param<double>(static_cast<int>(i)), params[i]);
    }
    EXPECT_EQ(meta.get_param<uint64_t>(sb::Dispatch::K_RNG_SEED_PARAM), 7u);
    EXPECT_EQ(meta.get_param<uint64_t>(sb::Dispatch::K_RNG_OFFSET_PARAM), 8u);
}

TEST_F(Unit_DispatchSeamTest, TooManyTaskParamsThrowsInsteadOfDroppingThem)
{
    auto t = engine->create_tensor({4});
    const std::vector<float> too_many(SushiRuntime::Graph::TaskMetadata::K_MAX_PARAMS + 1, 1.0f);

    EXPECT_THROW(
        sb::Dispatch::execute_unary_inplace(*engine, t, "test.overflow", "test.overflow"_op,
                                            [](auto x) { return x; }, too_many),
        std::runtime_error);
}

/**
 * @brief Tests that a typed task reports the family its caller declared.
 * @see tests/functional/unit/README.md
 */
TEST_F(Unit_DispatchSeamTest, ATypedTaskReportsTheFamilyItsCallerDeclared)
{
    auto t = engine->create_tensor({4}, DataType::COMPLEX32);

    SushiRuntime::Graph::TaskMetadata meta;
    meta.name = "spatial.im2col";
    meta.task_type = SushiRuntime::Graph::TaskType::MATH_OP;

    try
    {
        (void)sb::Internal::execute_typed_task<sb::Dispatch::Domain::REAL>(
            *engine, meta, {}, {}, t.dtype, "Spatial operation",
            [](auto, sycl::queue&, const std::vector<sycl::event>&) -> sycl::event
            {
                return sycl::event();
            });
        FAIL() << "a complex dtype must be refused by a REAL-only task";
    }
    catch (const std::exception& e)
    {
        const std::string message = e.what();
        EXPECT_NE(message.find("Spatial operation"), std::string::npos) << message;
        EXPECT_EQ(message.find("BLAS operation"), std::string::npos) << message;
        EXPECT_NE(message.find("spatial.im2col"), std::string::npos) << message;
    }
}
