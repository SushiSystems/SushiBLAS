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
/* Licensed under the Apache License, Version 2.0 (the "License");        */
/* you may not use this file except in compliance with the License.       */
/* You may obtain a copy of the License at                                */
/*                                                                        */
/*     http://www.apache.org/licenses/LICENSE-2.0                         */
/*                                                                        */
/* Unless required by applicable law or agreed to in writing, software    */
/* distributed under the License is distributed on an "AS IS" BASIS,      */
/* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        */
/* implied. See the License for the specific language governing           */
/* permissions and limitations under the License.                         */
/**************************************************************************/

// The smallest thing that proves the package works: a real op (AXPY, one of
// SB-3's from-scratch kernels), recorded and run against an installed
// SushiBLAS, from a project that has no access to sushiblas's source tree.
//
// It is small on purpose - anything more would be testing SushiBLAS itself,
// which its own suite does. What is under test here is the package: that the
// headers are found, that the kernel template compiles in THIS project's
// translation unit under the toolchain the consumer chose, and that the
// library links and runs end to end through both SushiBLAS and its
// SushiRuntime dependency.
//
// Second thing under test, added when the five *_internal.hpp op-dispatch
// helpers were promoted out of src/ and installed: that SushiBLAS::Dispatch is
// genuinely reachable from here. Handing execute_binary a lambda THIS file
// wrote, and having it compile and run, is the difference between "the fusion
// seam was moved" and "the fusion seam is public API". Nothing in sushiblas's
// own suite can show that - in-tree, every header is on the include path
// whether it gets installed or not.

#include <array>
#include <cmath>
#include <cstdio>
#include <vector>
#include <type_traits>
#include <SushiBLAS/SushiBLAS.h>
#include <SushiBLAS/engine/math/elementwise/dispatch.hpp>

using namespace SushiRuntime::Graph::Literals;

int main()
{
    sr::Execution::RuntimeContext ctx;
    sb::Engine engine(ctx);

    constexpr int64_t N = 64;
    auto x = engine.create_tensor({N});
    auto y = engine.create_tensor({N});

    engine.blas().axpy(2.0f, x, y);
    engine.execute();
    ctx.wait_all();

    // A consumer-authored functor fused into a single kernel: multiply, add and
    // divide as one device-side expression, with no SYCL written here and one
    // task in the graph instead of three.
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

    sb::Dispatch::execute_binary(engine, a, b, c, "consumer.fused", "consumer.fused"_op,
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

    // The variadic in-place seam, from out of tree: three tensors advanced
    // together by one consumer-authored functor, in one task. This also proves
    // <SushiBLAS/engine/launch_1d.hpp> - the header the seam's launch geometry
    // lives in - actually reaches an installed consumer, which nothing in-tree
    // can show.
    sb::Dispatch::execute_zip_inplace(engine, std::array{&a, &b, &c},
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

    std::printf("OK: SushiBLAS %s ran AXPY, a consumer-authored fused kernel and a "
                "fused in-place update against an installed package\n",
                SUSHIBLAS_VERSION_STRING);
    return 0;
}
