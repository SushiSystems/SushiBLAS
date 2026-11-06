/**************************************************************************/
/* main.cpp                                                               */
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
 * @file main.cpp
 * @brief Runs AXPY and three consumer-authored fused kernels against an installed SushiBLAS.
 * @author Mustafa Garip
 * @see tests/package/README.md
 */

#include <array>
#include <cstdio>
#include <vector>
#include <cstdint>
#include <type_traits>
#include <SushiBLAS/SushiBLAS.hpp>
#include <SushiBLAS/blas/dispatch.hpp>
#include <SushiBLAS/elementwise/dispatch.hpp>

namespace sb = SushiBLAS;
namespace sr = SushiRuntime;
using namespace SushiRuntime::Literals;

/**
 * @brief Returns whether every actual value equals its expected value exactly.
 * @param what Names the checked result in the failure line.
 * @param actual The values the program read back.
 * @param expected The values the arithmetic gives.
 * @note Every expected value is a small integer or a half of one, which float holds exactly, so no
 *       tolerance is needed and a rounding difference is a defect.
 */
bool matches(const char* what, const std::vector<float>& actual, const std::vector<float>& expected)
{
    for (std::size_t i = 0; i < expected.size(); ++i)
    {
        if (actual[i] != expected[i])
        {
            std::printf("FAIL: %s produced %f at index %zu, expected %f\n", what,
                        static_cast<double>(actual[i]), i, static_cast<double>(expected[i]));
            return false;
        }
    }
    return true;
}

/**
 * @brief Adds a row bias and a column bias to a GEMM element and clamps the sum at zero.
 * @warning Fused into a forward training pass, it needs the pre-activation spilled too.
 * @see tests/package/README.md
 */
struct ConsumerBiasReLU
{
    /** @brief One value per row of C. */
    const float* row_bias;

    /** @brief One value per column of C. */
    const float* col_bias;

    /**
     * @brief Adds both biases to @p value and clamps at zero.
     * @tparam T The element type of C.
     * @param value The fully formed alpha*(A*B) + beta*C element.
     * @param row The element's row in C.
     * @param col The element's column in C.
     * @return max(value + row_bias[row] + col_bias[col], 0).
     */
    template<typename T>
    T operator()(T value, std::int64_t row, std::int64_t col) const
    {
        const T y = value + static_cast<T>(row_bias[row]) + static_cast<T>(col_bias[col]);
        return y > T(0) ? y : T(0);
    }
};

int main()
{
    sr::Runtime runtime = sr::Runtime::create().value();
    sb::Engine engine(runtime);
    sr::Graph graph = runtime.graph();
    sb::LateCountHold late_counts(runtime);
    sb::TaskRecorder recorder(graph.recorder(), engine.scratch_pool(), late_counts,
                              engine.mode());

    constexpr int64_t N = 64;
    auto x = engine.create_tensor({N});
    auto y = engine.create_tensor({N});

    SushiBLAS::BLASOps(recorder).axpy(2.0f, x, y);
    graph.run().value();
    graph.clear();
    runtime.wait().value();

    // One consumer-authored functor runs multiply, add and divide as a single task.
    auto a = engine.create_tensor({N});
    auto b = engine.create_tensor({N});
    auto c = engine.create_tensor({N});

    auto& q = runtime.advanced().queue(sr::DeviceIndex{0});
    std::vector<float> host_a(N), host_b(N);
    for (int64_t i = 0; i < N; ++i)
    {
        host_a[static_cast<size_t>(i)] = static_cast<float>(i);
        host_b[static_cast<size_t>(i)] = 2.0f;
    }
    q.memcpy(a.data(), host_a.data(), N * sizeof(float)).wait();
    q.memcpy(b.data(), host_b.data(), N * sizeof(float)).wait();

    sb::Dispatch::execute_binary(
        recorder, a, b, c, "consumer.fused", "consumer.fused"_op,
        [](auto p, auto r) { return (p * r + decltype(p)(1)) / decltype(p)(2); });
    graph.run().value();
    graph.clear();
    runtime.wait().value();

    std::vector<float> host_c(N);
    q.memcpy(host_c.data(), c.data(), N * sizeof(float)).wait();

    std::vector<float> want_fused(N);
    for (int64_t i = 0; i < N; ++i)
        want_fused[static_cast<size_t>(i)] = (static_cast<float>(i) * 2.0f + 1.0f) / 2.0f;
    if (!matches("fused dispatch", host_c, want_fused))
        return 1;

    // One consumer-authored functor advances three tensors in a single task.
    sb::Dispatch::execute_zip_inplace(recorder, std::array{&a, &b, &c},
        "consumer.zip", "consumer.zip"_op,
        [](auto& p, auto& r, auto& s)
        {
            using T = std::decay_t<decltype(p)>;
            s = p + r;
            p = p * T(2);
        });
    graph.run().value();
    graph.clear();
    runtime.wait().value();

    std::vector<float> host_p(N), host_s(N);
    q.memcpy(host_p.data(), a.data(), N * sizeof(float)).wait();
    q.memcpy(host_s.data(), c.data(), N * sizeof(float)).wait();

    std::vector<float> want_sum(N), want_double(N);
    for (int64_t i = 0; i < N; ++i)
    {
        want_sum[static_cast<size_t>(i)] = static_cast<float>(i) + 2.0f;
        want_double[static_cast<size_t>(i)] = static_cast<float>(i) * 2.0f;
    }
    if (!matches("fused in-place sum", host_s, want_sum) ||
        !matches("fused in-place double", host_p, want_double))
        return 1;

    // C is square, so a seam that swapped row and col would change the answer.
    auto gemm_a = engine.create_tensor({3, 2});
    auto gemm_b = engine.create_tensor({2, 3});
    auto gemm_c = engine.create_tensor({3, 3});
    auto row_bias = engine.create_tensor({3});
    auto col_bias = engine.create_tensor({3});

    const std::vector<float> host_gemm_a{1, 0, 0, 1, 1, 1};
    const std::vector<float> host_gemm_b{1, 2, 3, 4, 5, 6};
    const std::vector<float> host_row_bias{0.0f, -6.0f, 5.0f};
    const std::vector<float> host_col_bias{1.0f, -8.0f, 3.0f};

    q.memcpy(gemm_a.data(), host_gemm_a.data(), host_gemm_a.size() * sizeof(float)).wait();
    q.memcpy(gemm_b.data(), host_gemm_b.data(), host_gemm_b.size() * sizeof(float)).wait();
    q.memcpy(row_bias.data(), host_row_bias.data(), host_row_bias.size() * sizeof(float)).wait();
    q.memcpy(col_bias.data(), host_col_bias.data(), host_col_bias.size() * sizeof(float)).wait();

    sb::Dispatch::execute_gemm(
        recorder, gemm_a, gemm_b, gemm_c,
        sb::Core::Transpose::NoTrans, sb::Core::Transpose::NoTrans, 1.0, 0.0,
        "consumer.linear_relu", "consumer.linear_relu"_op,
        ConsumerBiasReLU{row_bias.data_as<float>(), col_bias.data_as<float>()});
    graph.run().value();
    graph.clear();
    runtime.wait().value();

    const std::vector<float> want_gemm{2, 0, 6, 0, 0, 3, 11, 4, 17};
    std::vector<float> host_gemm_c(want_gemm.size());
    q.memcpy(host_gemm_c.data(), gemm_c.data(), want_gemm.size() * sizeof(float)).wait();

    if (!matches("fused GEMM epilogue", host_gemm_c, want_gemm))
        return 1;

    std::printf("OK: SushiBLAS %s ran AXPY, a consumer-authored fused kernel, a "
                "fused in-place update and a GEMM with a consumer-authored epilogue "
                "against an installed package\n",
                SUSHIBLAS_VERSION_STRING);
    return 0;
}
