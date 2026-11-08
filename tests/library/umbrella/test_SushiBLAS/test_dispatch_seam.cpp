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
 * @file test_dispatch_seam.cpp
 * @brief Tests the public Dispatch seam with functors written outside the library.
 * @author Mustafa Garip
 */

#include <array>
#include <map>
#include <functional>
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
#include <SushiBLAS/SushiBLAS.hpp>
#include <SushiBLAS/engine/dispatch.hpp>
#include <SushiBLAS/engine/task.hpp>
#include <SushiBLAS/logic/dispatch.hpp>
#include <SushiBLAS/arithmetic/safe_math.hpp>
#include <SushiBLAS/elementwise/dispatch.hpp>
#include <SushiBLAS/nonlinear/dispatch.hpp>
#include <SushiBLAS/random/dispatch.hpp>
#include <SushiBLAS/reductions/dispatch.hpp>
#include <SushiBLAS/functor_map/read_binding.hpp>
#include <SushiBLAS/engine/task_description.hpp>
#include <SushiRuntime/sushiruntime.hpp>
#include <SushiBLAS/engine_fixture/sushiblas_test.hpp>
#include <SushiBLAS/tensor_expectations/tensor_comparison.hpp>
#include <SushiBLAS/tensor_expectations/tensor_fill.hpp>
#include <SushiBLAS/tensor_expectations/tensor_readback.hpp>
#include <SushiBLAS/tensor_expectations/scalar_dtype.hpp>
#include <SushiBLAS/test_runner/test_environment.hpp>
#include <SushiBLAS/graph_probe/graph_probe.hpp>

namespace sb = SushiBLAS;

using namespace SushiBLAS::Test;
using namespace SushiRuntime::Literals;
using SushiBLAS::Core::DataType;

namespace
{
    /**
     * @brief Returns the bit pattern of a float.
     * @see tests/library/umbrella/README.md
     */
    uint32_t float_bits(float value) noexcept
    {
        uint32_t bits = 0;
        std::memcpy(&bits, &value, sizeof(bits));
        return bits;
    }

    /** @brief Absolute tolerance: 128 float roundings of a value of order one. */
    constexpr double K_REAL_TOLERANCE = absolute_tolerance<float>(128);

    /** @brief Absolute tolerance: 2 half roundings of a value of order one. */
    constexpr double K_HALF_TOLERANCE = half_absolute_tolerance(2);

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

    /** @brief Returns the parameter slots 0 to count - 1 of description as doubles. */
    std::vector<double> distribution_parameters(const sb::TaskDescription& description, std::size_t count)
    {
        std::vector<double> parameters;
        for (std::size_t slot = 0; slot < count; ++slot)
            parameters.push_back(description.parameter<double>(static_cast<int>(slot)));
        return parameters;
    }

    /** @brief Returns the values the repeatability update starts from, a sawtooth of count elements. */
    std::vector<float> repeatable_seed(int64_t count)
    {
        std::vector<float> seed(static_cast<size_t>(count));
        for (int64_t i = 0; i < count; ++i)
            seed[static_cast<size_t>(i)] = 0.001f * static_cast<float>((i % 31) - 15);
        return seed;
    }

    /** @brief Expects two FLOAT32 buffers to hold the same bits, element by element. */
    void expect_same_float_bits(const std::vector<float>& actual, const std::vector<float>& expected)
    {
        ASSERT_EQ(actual.size(), expected.size());
        for (size_t i = 0; i < actual.size(); ++i)
        {
            EXPECT_EQ(float_bits(actual[i]), float_bits(expected[i]))
                << "Fused update was not bit-reproducible at index " << i;
        }
    }

    /** @brief The inputs and the output of one declared-read seam row. */
    struct SeamInputs
    {
        sb::Tensor factor;    /**< The declared read, {2}. */
        sb::Tensor a;         /**< {1, 2, 3, 4}. */
        sb::Tensor b;         /**< {2, 3, 6, 7}. */
        sb::Tensor condition; /**< {1, 1, 0, 0}. */
        sb::Tensor out;       /**< Starts as a copy of a, since the forward seam works in place. */
    };

    /** @brief One functor seam: its name, the call that records it on SeamInputs, and the expected output. */
    struct SeamRow
    {
        std::string name;
        std::function<void(sb::TaskRecorder&, SeamInputs&)> record;
        std::vector<float> expected;
    };

    /** @brief Names a row by its name, which gtest takes as the test's suffix. */
    std::string seam_row_name(const ::testing::TestParamInfo<SeamRow>& info)
    {
        return info.param.name;
    }
} // namespace

class Medium_DispatchSeamTest : public SushiBLASTest
{
    protected:
        /** @brief The scalars of one AdamW step at step 1. */
        struct AdamWScalars
        {
            float b1 = 0.9f;
            float b2 = 0.999f;
            float lr = 0.01f;
            float wd = 0.01f;
            float eps = 1e-8f;
            float ema_decay = 0.99f;
            float inv_bc1 = 1.0f / (1.0f - 0.9f);   /**< Bias correction of the first moment at step 1. */
            float inv_bc2 = 1.0f / (1.0f - 0.999f); /**< Bias correction of the second moment at step 1. */
        };

        /** @brief The four tensors an AdamW step updates in place. */
        struct AdamWState
        {
            sb::Tensor theta;
            sb::Tensor m;
            sb::Tensor v;
            sb::Tensor ema;
        };

        /** @brief Returns a state whose four tensors hold the given values. */
        AdamWState seeded_adamw_state(const std::vector<float>& theta, const std::vector<float>& m,
                                      const std::vector<float>& v)
        {
            const int64_t n = static_cast<int64_t>(theta.size());
            AdamWState state{engine->create_tensor({n}), engine->create_tensor({n}),
                             engine->create_tensor({n}), engine->create_tensor({n})};
            fill_tensor(state.theta, theta);
            fill_tensor(state.m, m);
            fill_tensor(state.v, v);
            fill_tensor(state.ema, theta);
            return state;
        }

        /** @brief Records the whole update as one fused task; g is an operand although the functor only reads it. */
        void record_fused_adamw(AdamWState& state, sb::Tensor& g, const AdamWScalars& p)
        {
            sb::Dispatch::execute_zip_inplace(recorder(),
                std::array{&state.theta, &g, &state.m, &state.v, &state.ema},
                "test.adamw_fused", "test.adamw_fused"_op,
                [b1 = p.b1, b2 = p.b2, lr = p.lr, wd = p.wd, eps = p.eps, inv_bc1 = p.inv_bc1, inv_bc2 = p.inv_bc2,
                 ema_decay = p.ema_decay]
                (auto& theta, auto& grad, auto& m, auto& v, auto& ema)
                {
                    using T = std::decay_t<decltype(theta)>;
                    m = T(b1) * m + T(1.0f - b1) * grad;
                    v = T(b2) * v + T(1.0f - b2) * (grad * grad);
                    const T denom = sb::Dispatch::safe_sqrt(v * T(inv_bc2)) + T(eps);
                    theta = theta * T(1.0f - lr * wd) - T(lr) * ((m * T(inv_bc1)) / denom);
                    ema = T(ema_decay) * ema + T(1.0f - ema_decay) * theta;
                });
        }

        /** @brief Records the same arithmetic as fifteen library tasks. */
        void record_adamw_steps(AdamWState& state, sb::Tensor& g, const AdamWScalars& p)
        {
            const int64_t n = g.shape[0];
            auto scratch_a = engine->create_tensor({n});
            auto scratch_b = engine->create_tensor({n});
            auto eps_vec = engine->create_tensor({n});
            fill_tensor(eps_vec, std::vector<float>(static_cast<size_t>(n), p.eps));

            SushiBLAS::BLASOps(recorder()).scal(static_cast<double>(p.b1), state.m);
            SushiBLAS::BLASOps(recorder()).axpy(static_cast<double>(1.0f - p.b1), g, state.m);

            SushiBLAS::BLASOps(recorder()).copy(g, scratch_a);
            SushiBLAS::ElementwiseOps(recorder()).square(scratch_a);
            SushiBLAS::BLASOps(recorder()).scal(static_cast<double>(p.b2), state.v);
            SushiBLAS::BLASOps(recorder()).axpy(static_cast<double>(1.0f - p.b2), scratch_a, state.v);

            SushiBLAS::BLASOps(recorder()).copy(state.v, scratch_a);
            SushiBLAS::BLASOps(recorder()).scal(static_cast<double>(p.inv_bc2), scratch_a);
            SushiBLAS::ElementwiseOps(recorder()).sqrt(scratch_a);
            SushiBLAS::ElementwiseOps(recorder()).add(scratch_a, eps_vec, scratch_a);

            SushiBLAS::BLASOps(recorder()).copy(state.m, scratch_b);
            SushiBLAS::BLASOps(recorder()).scal(static_cast<double>(p.inv_bc1), scratch_b);
            SushiBLAS::ElementwiseOps(recorder()).div(scratch_b, scratch_a, scratch_b);

            SushiBLAS::BLASOps(recorder()).scal(static_cast<double>(1.0f - p.lr * p.wd), state.theta);
            SushiBLAS::BLASOps(recorder()).axpy(static_cast<double>(-p.lr), scratch_b, state.theta);

            SushiBLAS::BLASOps(recorder()).scal(static_cast<double>(p.ema_decay), state.ema);
            SushiBLAS::BLASOps(recorder()).axpy(static_cast<double>(1.0f - p.ema_decay), state.theta, state.ema);
        }

        /** @brief Expects the four tensors of two states to agree within two float roundings. */
        void expect_adamw_states_match(const AdamWState& actual, const AdamWState& expected)
        {
            SB_ASSERT_TENSOR_NEAR(actual.m, expected.m, absolute_tolerance<float>(2), float);
            SB_ASSERT_TENSOR_NEAR(actual.v, expected.v, absolute_tolerance<float>(2), float);
            SB_ASSERT_TENSOR_NEAR(actual.theta, expected.theta, absolute_tolerance<float>(2), float);
            SB_ASSERT_TENSOR_NEAR(actual.ema, expected.ema, absolute_tolerance<float>(2), float);
        }

        /** @brief Runs two identical fused updates' worth of work once on seed and returns a, b and c. */
        std::vector<float> run_zip_determinism_update(const std::vector<float>& seed)
        {
            const int64_t count = static_cast<int64_t>(seed.size());
            auto a = engine->create_tensor({count});
            auto b = engine->create_tensor({count});
            auto c = engine->create_tensor({count});
            fill_tensor(a, seed);
            fill_tensor(b, seed);
            fill_tensor(c, seed);

            sb::Dispatch::execute_zip_inplace(recorder(), std::array{&a, &b, &c},
                "test.zip_determinism", "test.zip_determinism"_op,
                [](auto& x, auto& y, auto& z)
                {
                    using T = std::decay_t<decltype(x)>;
                    x = x * T(1.0000001) + y * T(0.3333333) + z * T(0.1666667);
                    y = y - x * T(0.25);
                    z = sb::Dispatch::safe_sqrt(x * x + y * y);
                });
            recording->run().value();

            std::vector<float> out = read_tensor<float>(a);
            for (const sb::Tensor* t : {&b, &c})
            {
                const std::vector<float> part = read_tensor<float>(*t);
                out.insert(out.end(), part.begin(), part.end());
            }
            return out;
        }

        /** @brief Expects clamp over T to equal the host clamp with narrowed bounds, bit for bit. */
        template<typename T>
        void expect_clamp_matches_host()
        {
            const std::vector<T> input = exact_values<T>({-3.0f, -1.0f, 0.0f, 2.0f, 3.0f, 5.0f});
            auto t = create_initialized_tensor<T>({6}, input);

            sb::ElementwiseOps(recorder()).clamp(t, K_CLAMP_LOW, K_CLAMP_HIGH);
            recording->run().value();

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
            auto elu_dx = engine->create_tensor({4}, sushiblas_dtype<T>());
            auto leaky_dx = engine->create_tensor({4}, sushiblas_dtype<T>());

            sb::ElementwiseOps(recorder()).pow(raised, K_POW_EXPONENT);
            sb::NonLinearOps(recorder()).elu(elu, K_ALPHA);
            sb::NonLinearOps(recorder()).elu_backward(dy, saved, elu_dx, K_ALPHA);
            sb::NonLinearOps(recorder()).leaky_relu(leaky, K_ALPHA);
            sb::NonLinearOps(recorder()).leaky_relu_backward(dy, saved, leaky_dx, K_ALPHA);
            recording->run().value();

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

        /**
         * @brief Returns a FLOAT32 tensor of dims in device-only USM, or nothing without such USM.
         * @return Empty when the allocation fails or the pointer is not usm::alloc::device.
         */
        std::optional<sb::Tensor> make_device_tensor(std::initializer_list<int64_t> dims)
        {
            try
            {
                sb::Tensor t = engine->create_tensor(dims, DataType::Float32,
                                                     SushiRuntime::Residency::Device);
                const sycl::context context =
                    TestEnvironment::queue().get_context();
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
                    probe.recorder(), std::array{parameters[slot], &gradient},
                    "test.gated_update", "test.gated_update"_op,
                    sb::Dispatch::bind_reads(GatedStep{static_cast<int64_t>(slot)},
                                             sb::Dispatch::read_as<float>(scalars),
                                             sb::Dispatch::read_as<int64_t>(gate)));
            }
            probe.execute().value();

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

TEST_F(Medium_DispatchSeamTest, FusesAConsumerFunctorIntoOneBinaryKernel)
{
    auto a = engine->create_tensor({4});
    auto b = engine->create_tensor({4});
    auto c = engine->create_tensor({4});
    fill_tensor(a, {1.0f, 2.0f, 3.0f, 4.0f});
    fill_tensor(b, {2.0f, 4.0f, 6.0f, 8.0f});

    // mul, add and div in one functor: one task where the public API records three.
    sb::Dispatch::execute_binary(
        recorder(), a, b, c, "test.fused_binary", "test.fused_binary"_op,
        [](auto x, auto y) { return (x * y + decltype(x)(1)) / decltype(x)(2); });
    EXPECT_EQ(recording->recorded_count(), 1u) << "the three operations are one task";
    recording->run().value();

    expect_tensor(c, {1.5f, 4.5f, 9.5f, 16.5f});
}

TEST_F(Medium_DispatchSeamTest, FusesAConsumerFunctorIntoOneUnaryKernel)
{
    const std::vector<float> inputs = {0.0f, 1.0f, 2.0f, 3.0f};
    auto t = engine->create_tensor({4});
    fill_tensor(t, inputs);

    // A consumer's own softplus-like activation, written against safe_math.
    sb::Dispatch::execute_unary_inplace(
        recorder(), t, "test.fused_unary", "test.fused_unary"_op,
        [](auto x) { return sb::Dispatch::safe_log(decltype(x)(1) + sb::Dispatch::safe_exp(x)); });
    EXPECT_EQ(recording->recorded_count(), 1u) << "the activation is one task";
    recording->run().value();

    std::vector<float> expected;
    for (float x : inputs)
        expected.push_back(static_cast<float>(std::log1p(std::exp(static_cast<double>(x)))));
    expect_tensor(t, expected, K_REAL_TOLERANCE);
}

TEST_F(Medium_DispatchSeamTest, ConsumerFunctorCanOptIntoComplex)
{
    using cf = std::complex<float>;

    auto a = create_initialized_tensor<cf>({2}, {cf(1.0f, 1.0f), cf(2.0f, -1.0f)});
    auto b = create_initialized_tensor<cf>({2}, {cf(0.0f, 1.0f), cf(1.0f, 1.0f)});
    auto c = engine->create_tensor({2}, DataType::Complex32);

    // (1+i)(i) + (1+i) = 2i and (2-i)(1+i) + (2-i) = 5.
    sb::Dispatch::execute_binary<sb::Dispatch::Domain::RealAndComplex>(
        recorder(), a, b, c, "test.fused_complex", "test.fused_complex"_op,
        [](auto x, auto y) { return x * y + x; });
    recording->run().value();

    auto expected = create_initialized_tensor<cf>({2}, {cf(0.0f, 2.0f), cf(5.0f, 0.0f)});
    SB_ASSERT_TENSOR_NEAR(c, expected, K_DEFAULT_TOLERANCE, cf);
}

TEST_F(Medium_DispatchSeamTest, ConsumerCanDriveTheLogicSeam)
{
    auto a = engine->create_tensor({4});
    auto b = engine->create_tensor({4});
    auto mask = engine->create_tensor({4});
    fill_tensor(a, {1.0f, 5.0f, 2.0f, 9.0f});
    fill_tensor(b, {3.0f, 3.0f, 3.0f, 3.0f});

    // "within 2 of b" - a predicate SushiBLAS does not ship.
    sb::Dispatch::execute_logic_binary(
        recorder(), a, b, mask, "test.near", "test.near"_op,
        [](auto x, auto y)
        {
            auto d = x - y;
            return (d < decltype(x)(0) ? -d : d) <= decltype(x)(2) ? decltype(x)(1) : decltype(x)(0);
        });
    recording->run().value();
    expect_tensor_equal(mask, {1.0f, 1.0f, 1.0f, 0.0f});
}

TEST_F(Medium_DispatchSeamTest, ConsumerCanDriveTheReductionSeam)
{
    auto mask = engine->create_tensor({4});
    fill_tensor(mask, {1.0f, 1.0f, 1.0f, 0.0f});

    auto total = engine->create_tensor({1});
    sb::Dispatch::execute_reduction(
        recorder(), mask, total, "test.count", "test.count"_op, PlainSumDispatch{});
    recording->run().value();
    expect_tensor_equal(total, {3.0f});
}

TEST_F(Medium_DispatchSeamTest, FusesAFiveTensorAdamWUpdateIntoOneKernel)
{
    // 1023 is a multiple of no work-group or vector width. See tests/library/umbrella/README.md.
    constexpr int64_t k_n = 1023;

    std::vector<float> g0(k_n), th0(k_n), m0(k_n), v0(k_n);
    for (int64_t i = 0; i < k_n; ++i)
    {
        g0[static_cast<size_t>(i)]  = 0.01f * static_cast<float>((i % 17) - 8);
        th0[static_cast<size_t>(i)] = 0.001f * static_cast<float>((i % 23) - 11);
        m0[static_cast<size_t>(i)]  = 0.002f * static_cast<float>((i % 13) - 6);
        v0[static_cast<size_t>(i)]  = 1e-4f * static_cast<float>((i % 7) + 1);
    }

    auto g = engine->create_tensor({k_n});
    fill_tensor(g, g0);
    AdamWState fused = seeded_adamw_state(th0, m0, v0);
    AdamWState steps = seeded_adamw_state(th0, m0, v0);
    const AdamWScalars scalars;

    record_fused_adamw(fused, g, scalars);
    EXPECT_EQ(recording->recorded_count(), 1u) << "the update is one task";
    record_adamw_steps(steps, g, scalars);
    recording->run().value();

    expect_adamw_states_match(fused, steps);

    // A read-only operand comes back bit for bit. See tests/library/umbrella/README.md.
    expect_tensor_equal(g, g0);
}

TEST_F(Medium_DispatchSeamTest, FusedInPlaceUpdateIsRepeatableToTheByte)
{
    // Two identical fused updates must agree exactly. See tests/library/umbrella/README.md.
    constexpr int64_t k_n = 517;
    const std::vector<float> seed = repeatable_seed(k_n);

    const std::vector<float> first = run_zip_determinism_update(seed);
    const std::vector<float> second = run_zip_determinism_update(seed);

    expect_same_float_bits(first, second);
}

/** @brief Runs one operand mix per row; every row must be refused. */
class Medium_DispatchSeamRefusalTest : public Medium_DispatchSeamTest, public ::testing::WithParamInterface<std::string> {};

TEST_P(Medium_DispatchSeamRefusalTest, FusedInPlaceUpdateRejectsDisagreeingOperands)
{
    auto four = engine->create_tensor({4});
    auto eight = engine->create_tensor({8});
    auto wide = engine->create_tensor({4}, DataType::Float64);
    auto cplx = engine->create_tensor({4}, DataType::Complex32);
    sb::Tensor* missing = nullptr;
    const std::map<std::string, std::array<sb::Tensor*, 2>> operands = {
        {"Size", {&four, &eight}}, {"Dtype", {&four, &wide}}, {"Null", {&four, missing}},
        {"ComplexUnderTheDefaultDomain", {&cplx, &cplx}}};

    // The message names the disagreeing operand; "sizes must match" is no help at N = 5.
    EXPECT_THROW(
        sb::Dispatch::execute_zip_inplace(
            recorder(), operands.at(GetParam()),
            "test.zip_refusal", "test.zip_refusal"_op, [](auto& x, auto& y) { x = x + y; }),
        std::invalid_argument);
}

INSTANTIATE_TEST_SUITE_P(Operands, Medium_DispatchSeamRefusalTest,
                         ::testing::Values("Size", "Dtype", "Null", "ComplexUnderTheDefaultDomain"));

TEST_F(Medium_DispatchSeamTest, DistributionParamsDoNotCollideWithTheRNGSlots)
{
    sb::TaskDescription description;
    const std::vector<double> params = {1.0, 2.0, 3.0, 4.0};

    sb::Dispatch::set_task_params(
        description, std::initializer_list<double>{1.0, 2.0, 3.0, 4.0}, "test.params",
        sb::TaskDescription::K_MAX_PARAMETERS - sb::Dispatch::K_MAX_RNG_PARAMS);
    sb::Dispatch::set_rng_params(description, 7, 8);

    EXPECT_EQ(distribution_parameters(description, params.size()), params);
    EXPECT_EQ(description.parameter<uint64_t>(sb::Dispatch::K_RNG_SEED_PARAM), 7u);
    EXPECT_EQ(description.parameter<uint64_t>(sb::Dispatch::K_RNG_OFFSET_PARAM), 8u);
}

TEST_F(Medium_DispatchSeamTest, ClampMatchesTheHostFormulaOnHalf)
{
    expect_clamp_matches_host<sycl::half>();
}

TEST_F(Medium_DispatchSeamTest, ClampMatchesTheHostFormulaOnFloat32)
{
    expect_clamp_matches_host<float>();
}

TEST_F(Medium_DispatchSeamTest, ClampMatchesTheHostFormulaOnFloat64)
{
    expect_clamp_matches_host<double>();
}

TEST_F(Medium_DispatchSeamTest, ClampMatchesTheHostFormulaOnInt32)
{
    expect_clamp_matches_host<int32_t>();
}

TEST_F(Medium_DispatchSeamTest, ClampMatchesTheHostFormulaOnInt64)
{
    expect_clamp_matches_host<int64_t>();
}

TEST_F(Medium_DispatchSeamTest, RealScalarOperationsMatchTheirHostFormulaOnHalf)
{
    expect_real_scalar_ops_match_host<sycl::half>(K_HALF_TOLERANCE);
}

TEST_F(Medium_DispatchSeamTest, RealScalarOperationsMatchTheirHostFormulaOnFloat32)
{
    expect_real_scalar_ops_match_host<float>(K_REAL_TOLERANCE);
}

TEST_F(Medium_DispatchSeamTest, RealScalarOperationsMatchTheirHostFormulaOnFloat64)
{
    expect_real_scalar_ops_match_host<double>(K_REAL_TOLERANCE);
}

/**
 * @brief Tests that a fused update takes a functor bound to a scalar parameter.
 * @see tests/library/umbrella/README.md
 */
TEST_F(Medium_DispatchSeamTest, ABoundFunctorWorksInAFusedUpdate)
{
    auto y = create_initialized_tensor<float>({4}, {1.0f, 2.0f, 3.0f, 4.0f});
    auto x = create_initialized_tensor<float>({4}, {2.0f, 4.0f, 6.0f, 8.0f});

    sb::Dispatch::execute_zip_inplace(
        recorder(), std::array{&y, &x}, "test.zip_bound", "test.zip_bound"_op,
        sb::Dispatch::bind_scalars([](auto& yi, auto& xi, auto a) { yi = yi + a * xi; },
                                   sb::Dispatch::ElementScalar(0.5)));
    recording->run().value();

    expect_tensor(y, {2.0f, 4.0f, 6.0f, 8.0f}, K_REAL_TOLERANCE);
    expect_tensor(x, {2.0f, 4.0f, 6.0f, 8.0f}, K_REAL_TOLERANCE);
}

TEST_F(Medium_DispatchSeamTest, ATypedTaskReportsTheFamilyItsCallerDeclared)
{
    auto t = engine->create_tensor({4}, DataType::Complex32);

    SushiBLAS::TaskDescription description;
    description.name = "spatial.im2col";

    const std::string message = refusal_of([&]
    {
        (void)sb::Internal::execute_typed_task<sb::Dispatch::Domain::Real>(
            recorder(), description, SushiBLAS::OperandSet(), t.dtype, "Spatial operation",
            [](auto, const sb::KernelPairDispatcher&,
               const std::vector<sycl::event>&) -> sycl::event
            {
                return sycl::event();
            });
    });

    ASSERT_FALSE(message.empty()) << "a complex dtype must be refused by a REAL-only task";
    EXPECT_NE(message.find("Spatial operation"), std::string::npos) << message;
    EXPECT_EQ(message.find("BLAS operation"), std::string::npos) << message;
    EXPECT_NE(message.find("spatial.im2col"), std::string::npos) << message;
}

/**
 * @brief Tests that a declared read orders its task behind the tensor's writer.
 * @see tests/library/umbrella/README.md
 */
TEST_F(Medium_DispatchSeamTest, ADeclaredReadIsOrderedBehindItsWriter)
{
    auto factor = engine->create_tensor({1});
    auto y = engine->create_tensor({4});

    GraphProbe declared;
    sb::RandomOps(declared.recorder(), engine->rng()).constant(factor, 2.0);
    record_scaled(declared.recorder(), y, factor);
    const auto declared_counts = declared.dependency_counts();
    ASSERT_TRUE(declared_counts.has_value()) << declared_counts.error().message;
    EXPECT_EQ(*declared_counts, (std::vector<std::size_t>{0, 1}));
}

/**
 * @brief Tests that a pointer a functor captured gets no edge: the library cannot see it.
 * @see tests/library/umbrella/README.md
 */
TEST_F(Medium_DispatchSeamTest, ACapturedPointerGetsNoDependencyEdge)
{
    auto factor = engine->create_tensor({1});
    auto y = engine->create_tensor({4});

    GraphProbe captured;
    sb::RandomOps(captured.recorder(), engine->rng()).constant(factor, 2.0);
    sb::Dispatch::execute_unary_inplace(
        captured.recorder(), y, "test.captured_pointer", "test.captured_pointer"_op,
        [pointer = factor.data_as<float>()](auto v) { return v * decltype(v)(pointer[0]); });
    const auto captured_counts = captured.dependency_counts();
    ASSERT_TRUE(captured_counts.has_value()) << captured_counts.error().message;
    EXPECT_EQ(*captured_counts, (std::vector<std::size_t>{0, 0}));
}

/** @brief Records one functor seam per row, each with the same declared read. */
class Medium_DispatchSeamReadTest : public Medium_DispatchSeamTest, public ::testing::WithParamInterface<SeamRow> {};

/**
 * @brief Tests that each of the other six functor seams hands a declared read to its functor.
 * @see tests/library/umbrella/README.md
 */
TEST_P(Medium_DispatchSeamReadTest, HandsTheDeclaredReadToItsFunctor)
{
    const std::vector<float> a_values = {1.0f, 2.0f, 3.0f, 4.0f};
    SeamInputs inputs{create_initialized_tensor<float>({1}, {2.0f}),
                      create_initialized_tensor<float>({4}, a_values),
                      create_initialized_tensor<float>({4}, {2.0f, 3.0f, 6.0f, 7.0f}),
                      create_initialized_tensor<float>({4}, {1.0f, 1.0f, 0.0f, 0.0f}),
                      create_initialized_tensor<float>({4}, a_values)};

    GetParam().record(recorder(), inputs);
    recording->run().value();

    expect_tensor_equal(inputs.out, GetParam().expected);
}

INSTANTIATE_TEST_SUITE_P(
    Seams, Medium_DispatchSeamReadTest,
    ::testing::Values(
        SeamRow{"Binary",
                [](sb::TaskRecorder& recorder, SeamInputs& in)
                {
                    sb::Dispatch::execute_binary(
                        recorder, in.a, in.b, in.out, "test.read_binary", "test.read_binary"_op,
                        sb::Dispatch::bind_reads([](auto x, auto y, const auto* f) { return (x + y) * f[0]; },
                                                 sb::Dispatch::read_as_element(in.factor)));
                },
                {6.0f, 10.0f, 18.0f, 22.0f}},
        SeamRow{"NonlinearForward",
                [](sb::TaskRecorder& recorder, SeamInputs& in)
                {
                    sb::Dispatch::execute_nonlinear_forward(
                        recorder, in.out, "test.read_forward", "test.read_forward"_op,
                        sb::Dispatch::bind_reads(ScaledByRead{}, sb::Dispatch::read_as_element(in.factor)));
                },
                {2.0f, 4.0f, 6.0f, 8.0f}},
        SeamRow{"NonlinearBackward",
                [](sb::TaskRecorder& recorder, SeamInputs& in)
                {
                    sb::Dispatch::execute_nonlinear_backward(
                        recorder, in.a, in.b, in.out, "test.read_backward", "test.read_backward"_op,
                        sb::Dispatch::bind_reads([](auto dy, auto x, const auto* f) { return dy * x * f[0]; },
                                                 sb::Dispatch::read_as_element(in.factor)));
                },
                {4.0f, 12.0f, 36.0f, 56.0f}},
        SeamRow{"LogicUnary",
                [](sb::TaskRecorder& recorder, SeamInputs& in)
                {
                    sb::Dispatch::execute_logic_unary(
                        recorder, in.a, in.out, "test.read_logic_unary", "test.read_logic_unary"_op,
                        sb::Dispatch::bind_reads(
                            [](auto x, const auto* f) { return x > f[0] ? decltype(x)(1) : decltype(x)(0); },
                            sb::Dispatch::read_as_element(in.factor)));
                },
                {0.0f, 0.0f, 1.0f, 1.0f}},
        SeamRow{"LogicBinary",
                [](sb::TaskRecorder& recorder, SeamInputs& in)
                {
                    sb::Dispatch::execute_logic_binary(
                        recorder, in.a, in.b, in.out, "test.read_logic_binary", "test.read_logic_binary"_op,
                        sb::Dispatch::bind_reads(
                            [](auto x, auto y, const auto* f)
                            {
                                return x * f[0] > y ? decltype(x)(1) : decltype(x)(0);
                            },
                            sb::Dispatch::read_as_element(in.factor)));
                },
                {0.0f, 1.0f, 0.0f, 1.0f}},
        SeamRow{"LogicTernary",
                [](sb::TaskRecorder& recorder, SeamInputs& in)
                {
                    sb::Dispatch::execute_logic_ternary(
                        recorder, in.condition, in.a, in.b, in.out, "test.read_logic_ternary",
                        "test.read_logic_ternary"_op,
                        sb::Dispatch::bind_reads(
                            [](auto c, auto x, auto y, const auto* f)
                            {
                                return c != decltype(c)(0) ? x * f[0] : y;
                            },
                            sb::Dispatch::read_as_element(in.factor)));
                },
                {2.0f, 4.0f, 6.0f, 7.0f}}),
    seam_row_name);

/** @brief Holds a declared read in device-only USM, and skips each test when the device offers none. */
class Medium_DispatchSeamDeviceUsmTest : public Medium_DispatchSeamTest
{
    protected:
        /** @brief Allocates the factor in device-only USM or skips the test. */
        void SetUp() override
        {
            Medium_DispatchSeamTest::SetUp();
            factor = make_device_tensor({1});
            if (!factor)
                GTEST_SKIP() << "This device offers no device-only USM.";
        }

        /// @brief The declared read, in device-only USM.
        std::optional<sb::Tensor> factor;
};

/**
 * @brief Tests that ForceSYCL records a declared read in device-only USM; the refusal under
 *        ForceReference is checked by ADeviceOnlyOperandIsRefusedInEveryFamily.
 * @see tests/library/umbrella/README.md
 */
TEST_F(Medium_DispatchSeamDeviceUsmTest, ADeviceOnlyReadIsRecordedUnderForceSYCL)
{
    auto y = engine->create_tensor({4});

    GraphProbe device(sb::ExecutionMode::ForceSYCL);
    EXPECT_NO_THROW(record_scaled(device.recorder(), y, *factor));
    EXPECT_EQ(device.take_recorded_count(), 1u);
}

/**
 * @brief Tests that the host path hands a functor the same declared reads as the SYCL path.
 * @see tests/library/umbrella/README.md
 */
TEST_F(Medium_DispatchSeamTest, ADeclaredReadReachesTheHostPath)
{
    const GatedUpdate reference = run_gated_update(sb::ExecutionMode::ForceReference);
    const GatedUpdate device = run_gated_update(sb::ExecutionMode::ForceSYCL);

    expect_same_bits(reference.gated, device.gated, "the parameter whose gate is set");
    expect_same_bits(reference.stepped, device.stepped, "the parameter whose gate is clear");
    ASSERT_EQ(reference.stepped.size(), reference.theta.size());
    EXPECT_NE(std::memcmp(reference.stepped.data(), reference.theta.data(),
                          reference.theta.size() * sizeof(double)), 0)
        << "the update under ForceReference changed nothing";
}
