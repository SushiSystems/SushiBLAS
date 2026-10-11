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
#include <cmath>
#include <atomic>
#include <vector>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <optional>
#include <exception>
#include <stdexcept>
#include <type_traits>
#include <initializer_list>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <SushiBLAS/SushiBLAS.h>
#include <SushiBLAS/engine/dispatch.hpp>
#include <SushiBLAS/engine/task.hpp>
#include <SushiBLAS/logic/dispatch.hpp>
#include <SushiBLAS/support/safe_math.hpp>
#include <SushiBLAS/elementwise/dispatch.hpp>
#include <SushiBLAS/nonlinear/dispatch.hpp>
#include <SushiBLAS/random/dispatch.hpp>
#include <SushiBLAS/reductions/dispatch.hpp>
#include <SushiBLAS/functor_map/read_binding.hpp>
#include <SushiRuntime/SushiRuntime.h>
#include "../common/test_utils.hpp"
#include "../common/graph_probe.hpp"

namespace sb = SushiBLAS;

using namespace SushiBLAS::Test;
using namespace SushiRuntime::Graph::Literals;
using SushiBLAS::Core::DataType;

namespace
{
    /**
     * @brief Returns the bit pattern of a float.
     * @see tests/unit/README.md
     */
    uint32_t float_bits(float value) noexcept
    {
        uint32_t bits = 0;
        std::memcpy(&bits, &value, sizeof(bits));
        return bits;
    }

    /** @brief Absolute tolerance on a FLOAT32 or FLOAT64 result, as the fixed-value tests use. */
    constexpr double K_REAL_TOLERANCE = 1e-4;

    /** @brief Absolute tolerance on a HALF result: two units in the last place at magnitude 8. */
    constexpr double K_HALF_TOLERANCE = 1.5625e-2;

    /** @brief Lower clamp bound; it truncates to -1 for an integer element. */
    constexpr double K_CLAMP_LOW = -1.3;

    /** @brief Upper clamp bound; it truncates to 2 for an integer element. */
    constexpr double K_CLAMP_HIGH = 2.7;

    /** @brief Exponent of the pow case. */
    constexpr double K_POW_EXPONENT = 3.0;

    /** @brief Slope of the elu and leaky_relu cases; binary16 does not hold it exactly. */
    constexpr double K_ALPHA = 0.3;

    /** @brief Returns values as elements of T; every value given is exact in every dtype. */
    template<typename T>
    std::vector<T> exact_values(const std::vector<float>& values)
    {
        std::vector<T> typed;
        typed.reserve(values.size());
        for (float value : values)
            typed.push_back(static_cast<T>(value));
        return typed;
    }

    /** @brief Widens a HALF or FLOAT32 element to double. */
    template<typename T>
    double widen(T value)
    {
        return static_cast<double>(static_cast<float>(value));
    }

    /** @brief Returns a FLOAT64 element unchanged. */
    inline double widen(double value)
    {
        return value;
    }

    /** @brief Absolute tolerance on a result made of small whole numbers: it is exact. */
    constexpr double K_EXACT_TOLERANCE = 0.0;

    /** @brief Element count of a gated update; a multiple of no work-group or vector width. */
    constexpr int64_t K_GATED_ELEMENTS = 1023;

    /** @brief The FLOAT32 step a gated update reads from its first declared tensor. */
    constexpr float K_GATED_STEP = 0.5f;

    /** @brief Sums a tensor with a plain SYCL reduction, as a consumer's own dispatcher would. */
    struct PlainSumDispatch
    {
        /** @brief Submits the SYCL fold behind deps; it stages no partials and uses no scratch. */
        template<typename T, typename Scratch>
        sycl::event operator()(sycl::queue& q, const T* pT, T* pR, int64_t size, Scratch,
                               const std::vector<sycl::event>& deps) const
        {
            return q.submit([&](sycl::handler& h)
            {
                h.depends_on(deps);
                h.parallel_for(sycl::range<1>(size),
                    sycl::reduction(pR, T(0), sycl::plus<T>(),
                                    sycl::property::reduction::initialize_to_identity{}),
                    [=](sycl::id<1> idx, auto& reducer) { reducer.combine(pT[idx[0]]); });
            });
        }

        /** @brief Runs the same sum on the host, in index order. */
        template<typename T>
        void reference(const T* pT, T* pR, int64_t size) const
        {
            T total = T(0);
            for (int64_t i = 0; i < size; ++i)
                total = total + pT[i];
            pR[0] = total;
        }
    };

    /** @brief Multiplies an element by the first element of the tensor its functor declared. */
    struct ScaledByRead
    {
        /** @brief Returns `value * factor[0]`. */
        template<typename T>
        T operator()(T value, const T* factor) const
        {
            return value * factor[0];
        }
    };

    /** @brief Adds `scalars[0] * g` to theta unless the gate of its slot is set. */
    struct GatedStep
    {
        /** @brief Holds the index of this parameter's gate. */
        int64_t slot;

        /** @brief Updates one element; the two pointers are the declared reads, in bound order. */
        template<typename T>
        void operator()(T& theta, T& g, const float* scalars, const int64_t* gate) const
        {
            if (gate[slot] != 0)
                return;

            theta = theta + static_cast<T>(scalars[0]) * g;
        }
    };

    /** @brief Holds the two FLOAT64 parameters a gated update leaves, and what it started from. */
    struct GatedUpdate
    {
        std::vector<double> theta;    /**< Both parameters before the update. */
        std::vector<double> gradient; /**< The gradient both updates read. */
        std::vector<double> gated;    /**< The parameter whose gate is set. */
        std::vector<double> stepped;  /**< The parameter whose gate is clear. */
    };

    /** @brief Returns what record throws as std::invalid_argument, or an empty string. */
    template<typename Record>
    std::string refusal_of(const Record& record)
    {
        try
        {
            record();
        }
        catch (const std::invalid_argument& error)
        {
            return error.what();
        }

        return std::string();
    }
} // namespace

class Unit_DispatchSeamTest : public SushiBLASTest
{
    protected:
        /** @brief Expects clamp over T to equal the host clamp with narrowed bounds, bit for bit. */
        template<typename T>
        void expect_clamp_matches_host()
        {
            const std::vector<T> input = exact_values<T>({-3.0f, -1.0f, 0.0f, 2.0f, 3.0f, 5.0f});
            auto t = create_initialized_tensor<T>({6}, input);

            sb::ElementwiseOps(*engine).clamp(t, K_CLAMP_LOW, K_CLAMP_HIGH);
            engine->execute();

            const T low = sb::Dispatch::ElementScalar(K_CLAMP_LOW).as<T>();
            const T high = sb::Dispatch::ElementScalar(K_CLAMP_HIGH).as<T>();
            std::vector<T> expected = input;
            for (T& value : expected)
                value = value < low ? low : (high < value ? high : value);

            const std::vector<T> actual = read_tensor<T>(t);
            ASSERT_EQ(actual.size(), expected.size());
            EXPECT_EQ(std::memcmp(actual.data(), expected.data(), actual.size() * sizeof(T)), 0)
                << "clamp differs from the host clamp in dtype " << sb::Core::to_string(t.dtype);
        }

        /** @brief Expects out to hold formula(x[i]) at every index, within tolerance. */
        template<typename T, typename Formula>
        void expect_elements(const sb::Tensor& out, const std::vector<T>& x, double tolerance,
                             const char* op, Formula&& formula)
        {
            const std::vector<T> actual = read_tensor<T>(out);
            ASSERT_EQ(actual.size(), x.size());
            for (size_t i = 0; i < x.size(); ++i)
            {
                EXPECT_NEAR(widen(actual[i]), formula(widen(x[i])), tolerance)
                    << op << " differs at index " << i << " in dtype "
                    << sb::Core::to_string(out.dtype);
            }
        }

        /** @brief Expects pow, elu, leaky_relu and both backward passes to match the host over T. */
        template<typename T>
        void expect_real_scalar_ops_match_host(double tolerance)
        {
            const std::vector<T> x = exact_values<T>({-2.0f, -0.5f, 0.5f, 2.0f});
            const std::vector<T> positive = exact_values<T>({0.5f, 1.0f, 1.5f, 2.0f});
            const std::vector<T> ones = exact_values<T>({1.0f, 1.0f, 1.0f, 1.0f});
            const double exponent = widen(sb::Dispatch::ElementScalar(K_POW_EXPONENT).as<T>());
            const double alpha = widen(sb::Dispatch::ElementScalar(K_ALPHA).as<T>());

            auto raised = create_initialized_tensor<T>({4}, positive);
            auto elu = create_initialized_tensor<T>({4}, x);
            auto leaky = create_initialized_tensor<T>({4}, x);
            auto saved = create_initialized_tensor<T>({4}, x);
            auto dy = create_initialized_tensor<T>({4}, ones);
            auto elu_dx = engine->create_tensor({4}, get_sushiblas_dtype<T>());
            auto leaky_dx = engine->create_tensor({4}, get_sushiblas_dtype<T>());

            sb::ElementwiseOps(*engine).pow(raised, K_POW_EXPONENT);
            sb::NonLinearOps(*engine).elu(elu, K_ALPHA);
            sb::NonLinearOps(*engine).elu_backward(dy, saved, elu_dx, K_ALPHA);
            sb::NonLinearOps(*engine).leaky_relu(leaky, K_ALPHA);
            sb::NonLinearOps(*engine).leaky_relu_backward(dy, saved, leaky_dx, K_ALPHA);
            engine->execute();

            expect_elements(raised, positive, tolerance, "pow",
                            [=](double v) { return std::pow(v, exponent); });
            expect_elements(elu, x, tolerance, "elu",
                            [=](double v) { return v > 0.0 ? v : alpha * (std::exp(v) - 1.0); });
            expect_elements(elu_dx, x, tolerance, "elu_backward",
                            [=](double v) { return v > 0.0 ? 1.0 : alpha * std::exp(v); });
            expect_elements(leaky, x, tolerance, "leaky_relu",
                            [=](double v) { return v > 0.0 ? v : v * alpha; });
            expect_elements(leaky_dx, x, tolerance, "leaky_relu_backward",
                            [=](double v) { return v > 0.0 ? 1.0 : alpha; });
        }

        /** @brief Returns the in-degree of every node a probe holds, in recording order. */
        [[nodiscard]] static std::vector<int32_t> in_degrees_of(GraphProbe& probe)
        {
            std::vector<int32_t> degrees;
            for (const auto& node : probe.take_nodes())
                degrees.push_back(node->in_degree.load(std::memory_order_relaxed));
            return degrees;
        }

        /**
         * @brief Returns a FLOAT32 tensor of dims in device-only USM, or nothing without such USM.
         * @return Empty when the allocation fails or the pointer is not usm::alloc::device.
         */
        std::optional<sb::Tensor> make_device_tensor(std::initializer_list<int64_t> dims)
        {
            try
            {
                sb::Tensor t = engine->create_tensor(dims, DataType::Float32,
                                                     SushiRuntime::API::Residency::Device);
                const sycl::context context =
                    TestEnvironment::get_context().devices().queue(0).get_context();
                if (sycl::get_pointer_type(t.data(), context) != sycl::usm::alloc::device)
                    return std::nullopt;
                return t;
            }
            catch (const std::exception&)
            {
                return std::nullopt;
            }
        }

        /** @brief Records `y[i] = y[i] * factor[0]` with factor as a declared element read. */
        static sb::TaskHandle record_scaled(sb::TaskRecorder& recorder, sb::Tensor& y,
                                            const sb::Tensor& factor)
        {
            return sb::Dispatch::execute_unary_inplace(
                recorder, y, "test.scaled_by_read", "test.scaled_by_read"_op,
                sb::Dispatch::bind_reads(ScaledByRead{}, sb::Dispatch::read_as_element(factor)));
        }

        /**
         * @brief Runs a FLOAT64 update under mode on two parameters, gated by slots 0 and 1.
         * @return The inputs and both parameters; slot 0's gate is set and slot 1's is clear.
         */
        GatedUpdate run_gated_update(sb::ExecutionMode mode)
        {
            const size_t count = static_cast<size_t>(K_GATED_ELEMENTS);

            GatedUpdate update;
            update.theta.resize(count);
            update.gradient.resize(count);
            for (size_t i = 0; i < count; ++i)
            {
                update.theta[i] = 0.125 * (static_cast<double>(i % 23) - 11.0);
                update.gradient[i] = 0.25 * (static_cast<double>(i % 17) - 8.0);
            }

            auto scalars = create_initialized_tensor<float>({1}, {K_GATED_STEP});
            auto gate = create_initialized_tensor<int64_t>({2}, {1, 0});
            auto gradient = create_initialized_tensor<double>({K_GATED_ELEMENTS}, update.gradient);
            auto gated = create_initialized_tensor<double>({K_GATED_ELEMENTS}, update.theta);
            auto stepped = create_initialized_tensor<double>({K_GATED_ELEMENTS}, update.theta);

            GraphProbe probe(mode);
            const std::array<sb::Tensor*, 2> parameters = {&gated, &stepped};
            for (size_t slot = 0; slot < parameters.size(); ++slot)
            {
                sb::Dispatch::execute_zip_inplace(
                    probe.get_recorder(), std::array{parameters[slot], &gradient},
                    "test.gated_update", "test.gated_update"_op,
                    sb::Dispatch::bind_reads(GatedStep{static_cast<int64_t>(slot)},
                                             sb::Dispatch::read_as<float>(scalars),
                                             sb::Dispatch::read_as<int64_t>(gate)));
            }
            probe.execute();

            update.gated = read_tensor<double>(gated);
            update.stepped = read_tensor<double>(stepped);
            return update;
        }

        /** @brief Expects two FLOAT64 buffers to hold the same bits. */
        static void expect_same_bits(const std::vector<double>& actual,
                                     const std::vector<double>& expected, const char* what)
        {
            ASSERT_EQ(actual.size(), expected.size()) << what;

            const size_t bytes = actual.size() * sizeof(double);
            EXPECT_EQ(std::memcmp(actual.data(), expected.data(), bytes), 0)
                << what << " differs in its bits";
        }
};

TEST_F(Unit_DispatchSeamTest, FusesAConsumerFunctorIntoOneBinaryKernel)
{
    auto a = engine->create_tensor({4});
    auto b = engine->create_tensor({4});
    auto c = engine->create_tensor({4});
    fill_tensor(a, {1.0f, 2.0f, 3.0f, 4.0f});
    fill_tensor(b, {2.0f, 4.0f, 6.0f, 8.0f});

    // mul, add and div in one functor: one task where the public API records three.
    sb::Dispatch::execute_binary(
        engine->get_recorder(), a, b, c, "test.fused_binary", "test.fused_binary"_op,
        [](auto x, auto y) { return (x * y + decltype(x)(1)) / decltype(x)(2); });
    engine->execute();

    verify_tensor(c, {1.5f, 4.5f, 9.5f, 16.5f});
}

TEST_F(Unit_DispatchSeamTest, FusesAConsumerFunctorIntoOneUnaryKernel)
{
    auto t = engine->create_tensor({4});
    fill_tensor(t, {0.0f, 1.0f, 2.0f, 3.0f});

    // A consumer's own softplus-like activation, written against safe_math.
    sb::Dispatch::execute_unary_inplace(
        engine->get_recorder(), t, "test.fused_unary", "test.fused_unary"_op,
        [](auto x) { return sb::Dispatch::safe_log(decltype(x)(1) + sb::Dispatch::safe_exp(x)); });
    engine->execute();

    verify_tensor(t, {0.693147f, 1.313262f, 2.126928f, 3.048587f}, 1e-4);
}

TEST_F(Unit_DispatchSeamTest, ConsumerFunctorCanOptIntoComplex)
{
    using cf = std::complex<float>;

    auto a = create_initialized_tensor<cf>({2}, {cf(1.0f, 1.0f), cf(2.0f, -1.0f)});
    auto b = create_initialized_tensor<cf>({2}, {cf(0.0f, 1.0f), cf(1.0f, 1.0f)});
    auto c = engine->create_tensor({2}, DataType::Complex32);

    // (1+i)(i) + (1+i) = 2i and (2-i)(1+i) + (2-i) = 5.
    sb::Dispatch::execute_binary<sb::Dispatch::Domain::RealAndComplex>(
        engine->get_recorder(), a, b, c, "test.fused_complex", "test.fused_complex"_op,
        [](auto x, auto y) { return x * y + x; });
    engine->execute();

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
    sb::Dispatch::execute_logic_binary(
        engine->get_recorder(), a, b, mask, "test.near", "test.near"_op,
        [](auto x, auto y)
        {
            auto d = x - y;
            return (d < decltype(x)(0) ? -d : d) <= decltype(x)(2) ? decltype(x)(1) : decltype(x)(0);
        });
    engine->execute();
    verify_tensor(mask, {1.0f, 1.0f, 1.0f, 0.0f});

    auto total = engine->create_tensor({1});
    sb::Dispatch::execute_reduction(
        engine->get_recorder(), mask, total, "test.count", "test.count"_op, PlainSumDispatch{});
    engine->execute();
    verify_tensor(total, {3.0f});
}

TEST_F(Unit_DispatchSeamTest, RealOnlyDomainRejectsComplexAtRecordTime)
{
    auto a = engine->create_tensor({2}, DataType::Complex32);
    auto b = engine->create_tensor({2}, DataType::Complex32);
    auto c = engine->create_tensor({2}, DataType::Complex32);

    // The default domain is REAL. See tests/unit/README.md.
    EXPECT_THROW(
        sb::Dispatch::execute_binary(
            engine->get_recorder(), a, b, c, "test.real_only", "test.real_only"_op,
            [](auto x, auto y) { return x + y; }),
        std::invalid_argument);
}

TEST_F(Unit_DispatchSeamTest, FusesAFiveTensorAdamWUpdateIntoOneKernel)
{
    // 1023 is a multiple of no work-group or vector width. See tests/unit/README.md.
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

    // g is an operand although the functor only reads it. See tests/unit/README.md.
    sb::Dispatch::execute_zip_inplace(engine->get_recorder(),
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

    // The same arithmetic as fifteen library tasks. See tests/unit/README.md.
    auto scratch_a = engine->create_tensor({k_n});
    auto scratch_b = engine->create_tensor({k_n});
    auto eps_vec = engine->create_tensor({k_n});
    fill_tensor(eps_vec, std::vector<float>(static_cast<size_t>(k_n), eps));

    SushiBLAS::BLASOps(*engine).scal(static_cast<double>(b1), m_steps);
    SushiBLAS::BLASOps(*engine).axpy(static_cast<double>(1.0f - b1), g, m_steps);

    SushiBLAS::BLASOps(*engine).copy(g, scratch_a);
    SushiBLAS::ElementwiseOps(*engine).square(scratch_a);
    SushiBLAS::BLASOps(*engine).scal(static_cast<double>(b2), v_steps);
    SushiBLAS::BLASOps(*engine).axpy(static_cast<double>(1.0f - b2), scratch_a, v_steps);

    SushiBLAS::BLASOps(*engine).copy(v_steps, scratch_a);
    SushiBLAS::BLASOps(*engine).scal(static_cast<double>(inv_bc2), scratch_a);
    SushiBLAS::ElementwiseOps(*engine).sqrt(scratch_a);
    SushiBLAS::ElementwiseOps(*engine).add(scratch_a, eps_vec, scratch_a);

    SushiBLAS::BLASOps(*engine).copy(m_steps, scratch_b);
    SushiBLAS::BLASOps(*engine).scal(static_cast<double>(inv_bc1), scratch_b);
    SushiBLAS::ElementwiseOps(*engine).div(scratch_b, scratch_a, scratch_b);

    SushiBLAS::BLASOps(*engine).scal(static_cast<double>(1.0f - lr * wd), th_steps);
    SushiBLAS::BLASOps(*engine).axpy(static_cast<double>(-lr), scratch_b, th_steps);

    SushiBLAS::BLASOps(*engine).scal(static_cast<double>(ema_decay), ema_steps);
    SushiBLAS::BLASOps(*engine).axpy(static_cast<double>(1.0f - ema_decay), th_steps, ema_steps);

    engine->execute();

    SB_ASSERT_TENSOR_NEAR(m_fused, m_steps, 1e-6, float);
    SB_ASSERT_TENSOR_NEAR(v_fused, v_steps, 1e-6, float);
    SB_ASSERT_TENSOR_NEAR(th_fused, th_steps, 1e-6, float);
    SB_ASSERT_TENSOR_NEAR(ema_fused, ema_steps, 1e-6, float);

    // A read-only operand comes back bit for bit. See tests/unit/README.md.
    verify_tensor(g, g0, 0.0);
}

TEST_F(Unit_DispatchSeamTest, FusedInPlaceUpdateIsRepeatableToTheByte)
{
    // Two identical fused updates must agree exactly. See tests/unit/README.md.
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

        sb::Dispatch::execute_zip_inplace(engine->get_recorder(), std::array{&a, &b, &c},
            "test.zip_determinism", "test.zip_determinism"_op,
            [](auto& x, auto& y, auto& z)
            {
                using T = std::decay_t<decltype(x)>;
                x = x * T(1.0000001) + y * T(0.3333333) + z * T(0.1666667);
                y = y - x * T(0.25);
                z = sb::Dispatch::safe_sqrt(x * x + y * y);
            });
        engine->execute();

        std::vector<float> out(static_cast<size_t>(k_n) * 3);
        auto& q = SushiBLAS::Test::TestEnvironment::get_context().devices().queue(0);
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
    auto wide = engine->create_tensor({4}, DataType::Float64);
    auto cplx = engine->create_tensor({4}, DataType::Complex32);
    sb::Tensor* missing = nullptr;

    auto accumulate = [](auto& x, auto& y) { x = x + y; };

    // The message names the disagreeing operand; "sizes must match" is no help at N = 5.
    EXPECT_THROW(
        sb::Dispatch::execute_zip_inplace(
            engine->get_recorder(), std::array{&four, &eight},
            "test.zip_size", "test.zip_size"_op, accumulate),
        std::invalid_argument);

    EXPECT_THROW(
        sb::Dispatch::execute_zip_inplace(
            engine->get_recorder(), std::array{&four, &wide},
            "test.zip_dtype", "test.zip_dtype"_op, accumulate),
        std::invalid_argument);

    EXPECT_THROW(
        sb::Dispatch::execute_zip_inplace(
            engine->get_recorder(), std::array{&four, missing},
            "test.zip_null", "test.zip_null"_op, accumulate),
        std::invalid_argument);

    // Default domain is REAL, exactly as for execute_unary_inplace/execute_binary.
    EXPECT_THROW(
        sb::Dispatch::execute_zip_inplace(
            engine->get_recorder(), std::array{&cplx, &cplx},
            "test.zip_complex", "test.zip_complex"_op, accumulate),
        std::invalid_argument);
}

TEST_F(Unit_DispatchSeamTest, RngMetadataSlotsAreInRangeAndRoundTrip)
{
    // The seed and offset slots read back exactly. See tests/unit/README.md.
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
    const std::initializer_list<double> params = {1.0, 2.0, 3.0, 4.0};

    sb::Dispatch::set_task_params(meta, params, "test.params",
                                  SushiRuntime::Graph::TaskMetadata::K_MAX_PARAMS - sb::Dispatch::K_MAX_RNG_PARAMS);
    sb::Dispatch::set_rng_params(meta, 7, 8);

    for (size_t i = 0; i < params.size(); ++i)
    {
        EXPECT_DOUBLE_EQ(meta.get_param<double>(static_cast<int>(i)), params.begin()[i]);
    }
    EXPECT_EQ(meta.get_param<uint64_t>(sb::Dispatch::K_RNG_SEED_PARAM), 7u);
    EXPECT_EQ(meta.get_param<uint64_t>(sb::Dispatch::K_RNG_OFFSET_PARAM), 8u);
}

TEST_F(Unit_DispatchSeamTest, TooManyTaskParamsThrowsInsteadOfDroppingThem)
{
    auto t = engine->create_tensor({4});

    static_assert(SushiRuntime::Graph::TaskMetadata::K_MAX_PARAMS == 8, "the list below is one longer than the capacity");
    EXPECT_THROW(
        sb::Dispatch::execute_unary_inplace(
            engine->get_recorder(), t, "test.overflow", "test.overflow"_op,
            [](auto x) { return x; },
            {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f}),
        std::invalid_argument);
}

/**
 * @brief Tests that every operation with a scalar parameter matches its host formula per dtype.
 * @see tests/unit/README.md
 */
TEST_F(Unit_DispatchSeamTest, ScalarOperationsMatchTheirHostFormula)
{
    expect_clamp_matches_host<sycl::half>();
    expect_clamp_matches_host<float>();
    expect_clamp_matches_host<double>();
    expect_clamp_matches_host<int32_t>();
    expect_clamp_matches_host<int64_t>();

    expect_real_scalar_ops_match_host<sycl::half>(K_HALF_TOLERANCE);
    expect_real_scalar_ops_match_host<float>(K_REAL_TOLERANCE);
    expect_real_scalar_ops_match_host<double>(K_REAL_TOLERANCE);
}

/**
 * @brief Tests that a fused update takes a functor bound to a scalar parameter.
 * @see tests/unit/README.md
 */
TEST_F(Unit_DispatchSeamTest, ABoundFunctorWorksInAFusedUpdate)
{
    auto y = create_initialized_tensor<float>({4}, {1.0f, 2.0f, 3.0f, 4.0f});
    auto x = create_initialized_tensor<float>({4}, {2.0f, 4.0f, 6.0f, 8.0f});

    sb::Dispatch::execute_zip_inplace(
        engine->get_recorder(), std::array{&y, &x}, "test.zip_bound", "test.zip_bound"_op,
        sb::Dispatch::bind_scalars([](auto& yi, auto& xi, auto a) { yi = yi + a * xi; },
                                   sb::Dispatch::ElementScalar(0.5)));
    engine->execute();

    verify_tensor(y, {2.0f, 4.0f, 6.0f, 8.0f}, K_REAL_TOLERANCE);
    verify_tensor(x, {2.0f, 4.0f, 6.0f, 8.0f}, K_REAL_TOLERANCE);
}

/**
 * @brief Tests that a typed task reports the family its caller declared.
 * @see tests/unit/README.md
 */
TEST_F(Unit_DispatchSeamTest, ATypedTaskReportsTheFamilyItsCallerDeclared)
{
    auto t = engine->create_tensor({4}, DataType::Complex32);

    SushiRuntime::Graph::TaskMetadata meta;
    meta.name = "spatial.im2col";
    meta.task_type = SushiRuntime::Graph::TaskType::MATH_OP;

    try
    {
        (void)sb::Internal::execute_typed_task<sb::Dispatch::Domain::Real>(
            engine->get_recorder(), meta, SushiBLAS::OperandSet(), t.dtype, "Spatial operation",
            [](auto, const sb::Internal::KernelPairDispatcher&, sycl::queue&,
               const std::vector<sycl::event>&) -> sycl::event
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

/**
 * @brief Tests that a declared read orders its task behind the tensor's writer, and a captured
 *        pointer does not.
 * @see tests/unit/README.md
 */
TEST_F(Unit_DispatchSeamTest, ADeclaredReadIsOrderedBehindItsWriter)
{
    auto factor = engine->create_tensor({1});
    auto y = engine->create_tensor({4});

    GraphProbe declared;
    sb::RandomOps(declared.get_recorder(), engine->get_rng()).constant(factor, 2.0);
    record_scaled(declared.get_recorder(), y, factor);
    EXPECT_EQ(in_degrees_of(declared), (std::vector<int32_t>{0, 1}));

    GraphProbe captured;
    sb::RandomOps(captured.get_recorder(), engine->get_rng()).constant(factor, 2.0);
    sb::Dispatch::execute_unary_inplace(
        captured.get_recorder(), y, "test.captured_pointer", "test.captured_pointer"_op,
        [pointer = factor.data_as<float>()](auto v) { return v * decltype(v)(pointer[0]); });
    EXPECT_EQ(in_degrees_of(captured), (std::vector<int32_t>{0, 0}));
}

/**
 * @brief Tests that a task keeps the storage of a tensor its functor declared.
 * @see tests/unit/README.md
 */
TEST_F(Unit_DispatchSeamTest, ADeclaredReadIsKeptAllocated)
{
    SushiRuntime::Execution::RuntimeContext& context = TestEnvironment::get_context();

    auto y = create_initialized_tensor<float>({4}, {1.0f, 2.0f, 3.0f, 4.0f});
    std::optional<sb::Tensor> factor(create_initialized_tensor<float>({1}, {2.0f}));

    record_scaled(engine->get_recorder(), y, *factor);

    const size_t live_before = context.memory().live_count();
    factor.reset();
    EXPECT_EQ(context.memory().live_count(), live_before);

    engine->execute();
    verify_tensor(y, {2.0f, 4.0f, 6.0f, 8.0f}, K_EXACT_TOLERANCE);
}

/**
 * @brief Tests that a read declared in another type than the operation's reaches the functor typed.
 * @see tests/unit/README.md
 */
TEST_F(Unit_DispatchSeamTest, ADeclaredReadOfAnotherTypeArrivesTyped)
{
    const GatedUpdate update = run_gated_update(sb::ExecutionMode::Auto);

    std::vector<double> expected = update.theta;
    for (size_t i = 0; i < expected.size(); ++i)
        expected[i] = update.theta[i] + static_cast<double>(K_GATED_STEP) * update.gradient[i];

    expect_same_bits(update.gated, update.theta, "the parameter whose gate is set");
    expect_same_bits(update.stepped, expected, "the parameter whose gate is clear");
}

/**
 * @brief Tests that each of the other six functor seams hands a declared read to its functor.
 * @see tests/unit/README.md
 */
TEST_F(Unit_DispatchSeamTest, ADeclaredReadWorksOnEverySeam)
{
    const std::vector<float> a_values = {1.0f, 2.0f, 3.0f, 4.0f};
    const std::vector<float> b_values = {2.0f, 3.0f, 6.0f, 7.0f};

    auto factor = create_initialized_tensor<float>({1}, {2.0f});
    auto a = create_initialized_tensor<float>({4}, a_values);
    auto b = create_initialized_tensor<float>({4}, b_values);
    auto condition = create_initialized_tensor<float>({4}, {1.0f, 1.0f, 0.0f, 0.0f});
    auto forward = create_initialized_tensor<float>({4}, a_values);
    auto binary = engine->create_tensor({4});
    auto backward = engine->create_tensor({4});
    auto logic_unary = engine->create_tensor({4});
    auto logic_binary = engine->create_tensor({4});
    auto logic_ternary = engine->create_tensor({4});

    sb::TaskRecorder& recorder = engine->get_recorder();
    const auto read = sb::Dispatch::read_as_element(factor);

    sb::Dispatch::execute_binary(
        recorder, a, b, binary, "test.read_binary", "test.read_binary"_op,
        sb::Dispatch::bind_reads([](auto x, auto y, const auto* f) { return (x + y) * f[0]; },
                                 read));
    sb::Dispatch::execute_nonlinear_forward(
        recorder, forward, "test.read_forward", "test.read_forward"_op,
        sb::Dispatch::bind_reads(ScaledByRead{}, read));
    sb::Dispatch::execute_nonlinear_backward(
        recorder, a, b, backward, "test.read_backward", "test.read_backward"_op,
        sb::Dispatch::bind_reads([](auto dy, auto x, const auto* f) { return dy * x * f[0]; },
                                 read));
    sb::Dispatch::execute_logic_unary(
        recorder, a, logic_unary, "test.read_logic_unary", "test.read_logic_unary"_op,
        sb::Dispatch::bind_reads(
            [](auto x, const auto* f) { return x > f[0] ? decltype(x)(1) : decltype(x)(0); },
            read));
    sb::Dispatch::execute_logic_binary(
        recorder, a, b, logic_binary, "test.read_logic_binary", "test.read_logic_binary"_op,
        sb::Dispatch::bind_reads(
            [](auto x, auto y, const auto* f)
            {
                return x * f[0] > y ? decltype(x)(1) : decltype(x)(0);
            },
            read));
    sb::Dispatch::execute_logic_ternary(
        recorder, condition, a, b, logic_ternary, "test.read_logic_ternary",
        "test.read_logic_ternary"_op,
        sb::Dispatch::bind_reads(
            [](auto c, auto x, auto y, const auto* f)
            {
                return c != decltype(c)(0) ? x * f[0] : y;
            },
            read));
    engine->execute();

    verify_tensor(binary, {6.0f, 10.0f, 18.0f, 22.0f}, K_EXACT_TOLERANCE);
    verify_tensor(forward, {2.0f, 4.0f, 6.0f, 8.0f}, K_EXACT_TOLERANCE);
    verify_tensor(backward, {4.0f, 12.0f, 36.0f, 56.0f}, K_EXACT_TOLERANCE);
    verify_tensor(logic_unary, {0.0f, 0.0f, 1.0f, 1.0f}, K_EXACT_TOLERANCE);
    verify_tensor(logic_binary, {0.0f, 1.0f, 0.0f, 1.0f}, K_EXACT_TOLERANCE);
    verify_tensor(logic_ternary, {2.0f, 4.0f, 6.0f, 7.0f}, K_EXACT_TOLERANCE);
}

/**
 * @brief Tests that a seam refuses a declared read of a tensor the operation writes.
 * @see tests/unit/README.md
 */
TEST_F(Unit_DispatchSeamTest, ADeclaredReadMayNotBeAnOutput)
{
    GraphProbe probe;
    auto y = engine->create_tensor({4});

    const std::string message = refusal_of([&] { record_scaled(probe.get_recorder(), y, y); });

    EXPECT_NE(message.find("'read 0'"), std::string::npos) << message;
    EXPECT_TRUE(probe.take_nodes().empty());
}

/**
 * @brief Tests that ForceReference refuses a declared read in device-only USM and ForceSycl
 *        records it.
 * @see tests/unit/README.md
 */
TEST_F(Unit_DispatchSeamTest, ADeviceOnlyReadIsRefusedUnderForceReference)
{
    std::optional<sb::Tensor> factor = make_device_tensor({1});
    if (!factor)
        GTEST_SKIP() << "This device offers no device-only USM.";
    auto y = engine->create_tensor({4});

    GraphProbe reference(sb::ExecutionMode::ForceReference);
    const std::string message =
        refusal_of([&] { record_scaled(reference.get_recorder(), y, *factor); });
    EXPECT_NE(message.find("ForceReference"), std::string::npos) << message;
    EXPECT_TRUE(reference.take_nodes().empty());

    GraphProbe device(sb::ExecutionMode::ForceSycl);
    EXPECT_NO_THROW(record_scaled(device.get_recorder(), y, *factor));
    EXPECT_EQ(device.take_nodes().size(), 1u);
}

/**
 * @brief Tests that the host path hands a functor the same declared reads as the SYCL path.
 * @see tests/unit/README.md
 */
TEST_F(Unit_DispatchSeamTest, ADeclaredReadReachesTheHostPath)
{
    const GatedUpdate reference = run_gated_update(sb::ExecutionMode::ForceReference);
    const GatedUpdate device = run_gated_update(sb::ExecutionMode::ForceSycl);

    expect_same_bits(reference.gated, device.gated, "the parameter whose gate is set");
    expect_same_bits(reference.stepped, device.stepped, "the parameter whose gate is clear");
    ASSERT_EQ(reference.stepped.size(), reference.theta.size());
    EXPECT_NE(std::memcmp(reference.stepped.data(), reference.theta.data(),
                          reference.theta.size() * sizeof(double)), 0)
        << "the update under ForceReference changed nothing";
}
