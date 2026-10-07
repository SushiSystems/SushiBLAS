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
 * @file main.cpp
 * @brief Runs AXPY and three consumer-authored fused kernels against an installed SushiBLAS.
 * @author Mustafa Garip
 * @see tests/package/README.md
 */

#include <array>
#include <cmath>
#include <cstdio>
#include <vector>
#include <cstdint>
#include <type_traits>
#include <SushiBLAS/SushiBLAS.h>
#include <SushiBLAS/engine/blas/dispatch.hpp>
#include <SushiBLAS/engine/math/elementwise/dispatch.hpp>

namespace sb = SushiBLAS;
using namespace SushiRuntime::Graph::Literals;

/**
 * @brief Adds a row bias and a column bias to a GEMM element and clamps the sum at zero.
 * @warning Fused into a forward training pass, it needs the pre-activation spilled too.
 * @see tests/package/README.md
 */
struct ConsumerBiasRelu
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
    sr::Execution::RuntimeContext ctx;
    sb::Engine engine(ctx);

    constexpr int64_t N = 64;
    auto x = engine.create_tensor({N});
    auto y = engine.create_tensor({N});

    SushiBLAS::BLASOps(engine).axpy(2.0f, x, y);
    engine.execute();
    ctx.wait_all();

    // One consumer-authored functor runs multiply, add and divide as a single task.
    auto a = engine.create_tensor({N});
    auto b = engine.create_tensor({N});
    auto c = engine.create_tensor({N});

    auto& q = ctx.get_queue();
    std::vector<float> host_a(N), host_b(N);
    for (int64_t i = 0; i < N; ++i)
    {
        host_a[static_cast<size_t>(i)] = static_cast<float>(i);
        host_b[static_cast<size_t>(i)] = 2.0f;
    }
    q.memcpy(a.data(), host_a.data(), N * sizeof(float)).wait();
    q.memcpy(b.data(), host_b.data(), N * sizeof(float)).wait();

    sb::Dispatch::execute_binary(
        engine.get_recorder(), a, b, c, "consumer.fused", "consumer.fused"_op,
        [](auto p, auto r) { return (p * r + decltype(p)(1)) / decltype(p)(2); });
    engine.execute();
    ctx.wait_all();

    std::vector<float> host_c(N);
    q.memcpy(host_c.data(), c.data(), N * sizeof(float)).wait();

    for (int64_t i = 0; i < N; ++i)
    {
        const float expected = (static_cast<float>(i) * 2.0f + 1.0f) / 2.0f;
        const float actual = host_c[static_cast<size_t>(i)];
        if (std::fabs(actual - expected) > 1e-4f)
        {
            std::printf("FAIL: fused dispatch produced %f at index %lld, expected %f\n",
                        static_cast<double>(actual), static_cast<long long>(i),
                        static_cast<double>(expected));
            return 1;
        }
    }

    // One consumer-authored functor advances three tensors in a single task.
    sb::Dispatch::execute_zip_inplace(engine.get_recorder(), std::array{&a, &b, &c},
        "consumer.zip", "consumer.zip"_op,
        [](auto& p, auto& r, auto& s)
        {
            using T = std::decay_t<decltype(p)>;
            s = p + r;
            p = p * T(2);
        });
    engine.execute();
    ctx.wait_all();

    std::vector<float> host_p(N), host_s(N);
    q.memcpy(host_p.data(), a.data(), N * sizeof(float)).wait();
    q.memcpy(host_s.data(), c.data(), N * sizeof(float)).wait();

    for (int64_t i = 0; i < N; ++i)
    {
        const float base = static_cast<float>(i);
        if (std::fabs(host_s[static_cast<size_t>(i)] - (base + 2.0f)) > 1e-4f ||
            std::fabs(host_p[static_cast<size_t>(i)] - (base * 2.0f)) > 1e-4f)
        {
            std::printf("FAIL: fused in-place update wrong at index %lld\n",
                        static_cast<long long>(i));
            return 1;
        }
    }

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
        engine.get_recorder(), gemm_a, gemm_b, gemm_c, false, false, 1.0, 0.0,
        "consumer.linear_relu", "consumer.linear_relu"_op,
        ConsumerBiasRelu{row_bias.data_as<float>(), col_bias.data_as<float>()});
    engine.execute();
    ctx.wait_all();

    const std::vector<float> want_gemm{2, 0, 6, 0, 0, 3, 11, 4, 17};
    std::vector<float> host_gemm_c(want_gemm.size());
    q.memcpy(host_gemm_c.data(), gemm_c.data(), want_gemm.size() * sizeof(float)).wait();

    for (std::size_t i = 0; i < want_gemm.size(); ++i)
    {
        if (std::fabs(host_gemm_c[i] - want_gemm[i]) > 1e-4f)
        {
            std::printf("FAIL: fused GEMM epilogue produced %f at index %zu, expected %f\n",
                        static_cast<double>(host_gemm_c[i]), i,
                        static_cast<double>(want_gemm[i]));
            return 1;
        }
    }

    std::printf("OK: SushiBLAS %s ran AXPY, a consumer-authored fused kernel, a "
                "fused in-place update and a GEMM with a consumer-authored epilogue "
                "against an installed package\n",
                SUSHIBLAS_VERSION_STRING);
    return 0;
}
