/**************************************************************************/
/* test_storage_offset.cpp                                                */
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
 * @file test_storage_offset.cpp
 * @brief Tests every operation family on views at a non-zero storage offset.
 * @author Mustafa Garip
 */

#include <string>
#include <vector>
#include <utility>
#include <cstddef>
#include <algorithm>
#include <cstdint>
#include <gtest/gtest.h>
#include <SushiBLAS/tensor/runtime_block.hpp>
#include <SushiBLAS/SushiBLAS.hpp>
#include <SushiRuntime/core/sushi_ptr.hpp>
#include <SushiBLAS/engine_fixture/sushiblas_test.hpp>
#include <SushiBLAS/tensor_expectations/tensor_comparison.hpp>
#include <SushiBLAS/tensor_expectations/tensor_fill.hpp>
#include <SushiBLAS/test_runner/test_environment.hpp>
namespace SushiBLAS
{
    namespace Test
    {
        class Medium_StorageOffsetTest : public SushiBLASTest
        {
            protected:
                /** @brief Elements of poison placed before and after every view. */
                static constexpr int64_t K_PAD = 16;

                /** @brief Byte pattern the padding is filled with. */
                static constexpr int K_POISON_BYTE = 0x5A;

                /**
                 * @brief A tensor that is a view at element `K_PAD` of an over-sized,
                 *        fully poisoned allocation.
                 *
                 * The allocation holds `K_PAD + n + K_PAD` elements, so there is poison on
                 * both sides of the view: the head catches an operation that ignored the
                 * offset, the tail catches one that ran off the end.
                 */
                Tensor offset_view(std::initializer_list<int64_t> dims,
                                   Core::DataType dtype = Core::DataType::Float32)
                {
                    int64_t n = 1;
                    for (auto d : dims)
                        n *= d;

                    const size_t bpe = Core::bytes_per_element(dtype);
                    auto storage = make_runtime_storage(TestEnvironment::runtime(),
                        static_cast<size_t>(K_PAD + n + K_PAD) * bpe,
                        SushiRuntime::Residency::Shared);

                    // Poisons the view too; each test writes its inputs first.
                    auto& q = TestEnvironment::queue();
                    q.memset(storage->allocation_base(), K_POISON_BYTE, storage->size_bytes()).wait();

                    return Tensor(storage, dims, dtype, K_PAD, engine->layout());
                }

                /**
                 * @brief Asserts the padding on both sides of a view was not written.
                 *
                 * The assertion that actually catches an ignored offset. A kernel handed
                 * the allocation base instead of the view writes its output over the head
                 * padding, which this reports as the first differing byte.
                 */
                ::testing::AssertionResult padding_intact(const Tensor& v, const char* what) const
                {
                    const size_t bpe = Core::bytes_per_element(v.dtype);
                    const size_t total = v.storage->size_bytes();
                    const size_t head = static_cast<size_t>(v.storage_offset) * bpe;
                    const size_t tail = head + static_cast<size_t>(v.num_elements) * bpe;

                    std::vector<unsigned char> host(total);
                    auto& q = TestEnvironment::queue();
                    q.memcpy(host.data(), v.storage->allocation_base(), total).wait();

                    for (size_t i = 0; i < total; ++i)
                    {
                        if (i >= head && i < tail)
                            continue;
                        if (host[i] != static_cast<unsigned char>(K_POISON_BYTE))
                        {
                            return ::testing::AssertionFailure()
                                << what << ": byte " << i << " of the allocation was written, but the view "
                                << "occupies bytes [" << head << ", " << tail << "). The operation ignored "
                                << "storage_offset and addressed the allocation base instead.";
                        }
                    }
                    return ::testing::AssertionSuccess();
                }
        };

        TEST_F(Medium_StorageOffsetTest, Level1AxpyHonoursOffset)
        {
            auto x = offset_view({4});
            auto y = offset_view({4});
            fill_tensor(x, {1, 2, 3, 4});
            fill_tensor(y, {10, 20, 30, 40});

            SushiBLAS::BLASOps(recorder()).axpy(2.0, x, y);
            recording->run().value();

            expect_tensor_equal(y, std::vector<float>({12, 24, 36, 48}));
            EXPECT_TRUE(padding_intact(x, "axpy x"));
            EXPECT_TRUE(padding_intact(y, "axpy y"));
        }

        TEST_F(Medium_StorageOffsetTest, Level1ScalHonoursOffset)
        {
            auto x = offset_view({4});
            fill_tensor(x, {1, 2, 3, 4});

            SushiBLAS::BLASOps(recorder()).scal(3.0, x);
            recording->run().value();

            expect_tensor_equal(x, std::vector<float>({3, 6, 9, 12}));
            EXPECT_TRUE(padding_intact(x, "scal x"));
        }

        TEST_F(Medium_StorageOffsetTest, Level1CopyHonoursOffset)
        {
            auto x = offset_view({4});
            auto y = offset_view({4});
            fill_tensor(x, {5, 6, 7, 8});
            fill_tensor(y, {0, 0, 0, 0});

            SushiBLAS::BLASOps(recorder()).copy(x, y);
            recording->run().value();

            expect_tensor_equal(y, std::vector<float>({5, 6, 7, 8}));
            EXPECT_TRUE(padding_intact(x, "copy x"));
            EXPECT_TRUE(padding_intact(y, "copy y"));
        }

        TEST_F(Medium_StorageOffsetTest, Level2GemvHonoursOffset)
        {
            // beta != 0, so y is read as well as written.
            auto A = offset_view({2, 3});
            auto x = offset_view({3});
            auto y = offset_view({2});
            fill_tensor(A, {1, 2, 3, 4, 5, 6});
            fill_tensor(x, {1, 1, 1});
            fill_tensor(y, {100, 200});

            SushiBLAS::BLASOps(recorder()).gemv(
                A, x, y, Core::Transpose::NoTrans, /*alpha=*/1.0, /*beta=*/2.0);
            recording->run().value();

            // A*x = {6, 15}; + 2*{100, 200} = {206, 415}
            expect_tensor_equal(y, std::vector<float>({206, 415}));
            EXPECT_TRUE(padding_intact(A, "gemv A"));
            EXPECT_TRUE(padding_intact(x, "gemv x"));
            EXPECT_TRUE(padding_intact(y, "gemv y"));
        }

        TEST_F(Medium_StorageOffsetTest, Level2GerHonoursOffset)
        {
            auto x = offset_view({2});
            auto y = offset_view({3});
            auto A = offset_view({2, 3});
            fill_tensor(x, {1, 2});
            fill_tensor(y, {10, 20, 30});
            fill_tensor(A, {1, 1, 1, 1, 1, 1});

            SushiBLAS::BLASOps(recorder()).ger(x, y, A, /*alpha=*/1.0);
            recording->run().value();

            // A += x y^T = {{10,20,30},{20,40,60}}
            expect_tensor_equal(A, std::vector<float>({11, 21, 31, 21, 41, 61}));
            EXPECT_TRUE(padding_intact(x, "ger x"));
            EXPECT_TRUE(padding_intact(y, "ger y"));
            EXPECT_TRUE(padding_intact(A, "ger A"));
        }

        TEST_F(Medium_StorageOffsetTest, Level3GemmHonoursOffset)
        {
            auto A = offset_view({2, 2});
            auto B = offset_view({2, 2});
            auto C = offset_view({2, 2});
            fill_tensor(A, {1, 2, 3, 4});
            fill_tensor(B, {1, 0, 0, 1});
            fill_tensor(C, {0, 0, 0, 0});

            SushiBLAS::BLASOps(recorder()).gemm(A, B, C);
            recording->run().value();

            expect_tensor_equal(C, std::vector<float>({1, 2, 3, 4}));
            EXPECT_TRUE(padding_intact(A, "gemm A"));
            EXPECT_TRUE(padding_intact(B, "gemm B"));
            EXPECT_TRUE(padding_intact(C, "gemm C"));
        }

        TEST_F(Medium_StorageOffsetTest, ElementwiseAddHonoursOffset)
        {
            auto A = offset_view({4});
            auto B = offset_view({4});
            auto C = offset_view({4});
            fill_tensor(A, {1, 2, 3, 4});
            fill_tensor(B, {10, 10, 10, 10});
            fill_tensor(C, {0, 0, 0, 0});

            SushiBLAS::ElementwiseOps(recorder()).add(A, B, C);   // binary seam
            recording->run().value();

            expect_tensor_equal(C, std::vector<float>({11, 12, 13, 14}));
            EXPECT_TRUE(padding_intact(A, "add A"));
            EXPECT_TRUE(padding_intact(B, "add B"));
            EXPECT_TRUE(padding_intact(C, "add C"));
        }

        TEST_F(Medium_StorageOffsetTest, ElementwiseNegHonoursOffset)
        {
            auto C = offset_view({4});
            fill_tensor(C, {11, 12, 13, 14});

            SushiBLAS::ElementwiseOps(recorder()).neg(C);         // unary in-place seam
            recording->run().value();

            expect_tensor_equal(C, std::vector<float>({-11, -12, -13, -14}));
            EXPECT_TRUE(padding_intact(C, "neg C"));
        }

        TEST_F(Medium_StorageOffsetTest, NonlinearHonoursOffset)
        {
            auto t = offset_view({4});
            fill_tensor(t, {-2, -1, 1, 2});

            SushiBLAS::NonLinearOps(recorder()).relu(t);          // forward seam
            recording->run().value();

            expect_tensor_equal(t, std::vector<float>({0, 0, 1, 2}));
            EXPECT_TRUE(padding_intact(t, "relu t"));
        }

        TEST_F(Medium_StorageOffsetTest, NonlinearBackwardHonoursOffset)
        {
            auto dy = offset_view({4});
            auto x = offset_view({4});
            auto dx = offset_view({4});
            fill_tensor(dy, {1, 1, 1, 1});
            fill_tensor(x, {-2, -1, 1, 2});
            fill_tensor(dx, {0, 0, 0, 0});

            SushiBLAS::NonLinearOps(recorder()).relu_backward(dy, x, dx);
            recording->run().value();

            expect_tensor_equal(dx, std::vector<float>({0, 0, 1, 1}));
            EXPECT_TRUE(padding_intact(dy, "relu_backward dy"));
            EXPECT_TRUE(padding_intact(x, "relu_backward x"));
            EXPECT_TRUE(padding_intact(dx, "relu_backward dx"));
        }

        TEST_F(Medium_StorageOffsetTest, LogicHonoursOffset)
        {
            auto A = offset_view({4});
            auto B = offset_view({4});
            auto R = offset_view({4});
            fill_tensor(A, {1, 5, 3, 7});
            fill_tensor(B, {2, 2, 2, 2});
            fill_tensor(R, {0, 0, 0, 0});

            SushiBLAS::LogicOps(recorder()).greater(A, B, R);
            recording->run().value();

            expect_tensor_equal(R, std::vector<float>({0, 1, 1, 1}));
            EXPECT_TRUE(padding_intact(A, "greater A"));
            EXPECT_TRUE(padding_intact(B, "greater B"));
            EXPECT_TRUE(padding_intact(R, "greater R"));
        }

        TEST_F(Medium_StorageOffsetTest, ReductionHonoursOffset)
        {
            auto t = offset_view({4});
            auto r = offset_view({1});
            fill_tensor(t, {1, 2, 3, 4});
            fill_tensor(r, {0});

            SushiBLAS::ReductionOps(recorder()).sum(t, r);
            recording->run().value();

            expect_tensor_equal(r, std::vector<float>({10}));
            EXPECT_TRUE(padding_intact(t, "sum t"));
            EXPECT_TRUE(padding_intact(r, "sum result"));
        }

        TEST_F(Medium_StorageOffsetTest, RandomConstantHonoursOffset)
        {
            auto c = offset_view({8});

            SushiBLAS::RandomOps(recorder(), engine->rng()).constant(c, 7.0);    // the q.fill path
            recording->run().value();

            expect_tensor_equal(c, std::vector<float>(8, 7.0f));
            EXPECT_TRUE(padding_intact(c, "constant"));
        }

        TEST_F(Medium_StorageOffsetTest, RandomUniformHonoursOffset)
        {
            auto u = offset_view({8});

            SushiBLAS::RandomOps(recorder(), engine->rng())
                .uniform(u, 0.0, 1.0); // the Philox dispatch path
            recording->run().value();

            // The draw is checked for landing in the view and for its range only.
            EXPECT_TRUE(padding_intact(u, "uniform"));
            std::vector<float> host(8);
            TestEnvironment::queue().memcpy(host.data(), u.data(), host.size() * sizeof(float)).wait();
            EXPECT_TRUE(std::all_of(host.begin(), host.end(),
                                    [](float v) { return v >= 0.0f && v <= 1.0f; }));
        }

        TEST_F(Medium_StorageOffsetTest, OffsetIsCountedInElementsNotBytes)
        {
            // storage_offset counts elements, not bytes. See tests/library/umbrella/README.md.
            auto f64 = offset_view({4}, Core::DataType::Float64);

            const auto* base = static_cast<const char*>(f64.storage->allocation_base());
            const auto* view = static_cast<const char*>(f64.data());

            EXPECT_EQ(view - base, K_PAD * static_cast<int64_t>(sizeof(double)))
                << "storage_offset must be scaled by the element width, not used as a byte count.";
            EXPECT_EQ(reinterpret_cast<std::uintptr_t>(view) % sizeof(double), 0u)
                << "an element-counted offset over an aligned allocation is always naturally aligned.";
        }

        TEST_F(Medium_StorageOffsetTest, TheRegionOfAViewDescribesTheSameBytes)
        {
            auto f64 = offset_view({4}, Core::DataType::Float64);

            // The region handed to the dependency tracker describes the bytes of the view.
            const auto region = f64.region();
            EXPECT_EQ(region.base, f64.storage->allocation_base());
            EXPECT_EQ(region.offset, static_cast<size_t>(K_PAD) * sizeof(double));
            EXPECT_EQ(region.length, 4u * sizeof(double));
        }

        /// @brief Names one residency and an element count to allocate.
        struct AlignmentRow
        {
            /** @brief The row name, which gtest takes as the suffix of the test. */
            std::string name;

            /** @brief Where the allocation lives. */
            SushiRuntime::Residency residency;

            /** @brief The element count of the FLOAT32 tensor over it. */
            int64_t count;
        };

        /// @brief Returns one row per residency and per element count.
        static std::vector<AlignmentRow> alignment_rows()
        {
            std::vector<AlignmentRow> rows;
            const std::pair<const char*, SushiRuntime::Residency> residencies[] = {
                {"Shared", SushiRuntime::Residency::Shared},
                {"Host", SushiRuntime::Residency::Host},
                {"Device", SushiRuntime::Residency::Device}};
            for (const auto& residency : residencies)
            {
                for (const int64_t count : {int64_t{1}, int64_t{7}, int64_t{1000}})
                    rows.push_back({std::string(residency.first) + "_" + std::to_string(count),
                                    residency.second, count});
            }
            return rows;
        }

        /// @brief Allocates one storage per row.
        class Medium_StorageOffsetAlignment : public Medium_StorageOffsetTest,
                                              public ::testing::WithParamInterface<AlignmentRow> {};

        TEST_P(Medium_StorageOffsetAlignment, TheAllocationBaseIsAlignedForEveryResidency)
        {
            auto storage = make_runtime_storage(TestEnvironment::runtime(),
                static_cast<size_t>(GetParam().count) * sizeof(float), GetParam().residency);
            const Tensor t(storage, {GetParam().count}, Core::DataType::Float32);

            const auto base = reinterpret_cast<std::uintptr_t>(storage->allocation_base());
            EXPECT_EQ(base % K_STORAGE_ALIGNMENT, 0u);
            EXPECT_TRUE(t.is_aligned());
        }

        INSTANTIATE_TEST_SUITE_P(Rows, Medium_StorageOffsetAlignment,
                                 ::testing::ValuesIn(alignment_rows()),
                                 [](const ::testing::TestParamInfo<AlignmentRow>& info)
                                 { return info.param.name; });

        TEST_F(Medium_StorageOffsetTest, ATensorsDeviceIsTheDeviceItsStorageWasMadeOn)
        {
            auto storage = make_runtime_storage(TestEnvironment::runtime(), 4 * sizeof(float));
            const Tensor t(storage, {4}, Core::DataType::Float32);

            EXPECT_EQ(t.device(), TestEnvironment::queue(storage->device_index()).get_device());
        }

    } // namespace Test
} // namespace SushiBLAS
