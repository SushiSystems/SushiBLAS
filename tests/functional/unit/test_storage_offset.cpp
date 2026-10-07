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
 * @file test_storage_offset.cpp
 * @brief Tests every operation family on views at a non-zero storage offset.
 * @author Mustafa Garip
 */

#include <vector>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.h>
#include "../../common/test_utils.hpp"

namespace SushiBLAS
{
    namespace Test
    {
        class Unit_StorageOffsetTest : public SushiBLASTest
        {
            protected:
                /** @brief Elements of poison placed before and after every view. */
                static constexpr int64_t kPad = 16;

                /** @brief Byte pattern the padding is filled with. */
                static constexpr int kPoisonByte = 0x5A;

                /**
                 * @brief A tensor that is a view at element `kPad` of an over-sized,
                 *        fully poisoned allocation.
                 *
                 * The allocation holds `kPad + n + kPad` elements, so there is poison on
                 * both sides of the view: the head catches an operation that ignored the
                 * offset, the tail catches one that ran off the end.
                 */
                Tensor offset_view(std::initializer_list<int64_t> dims,
                                   Core::DataType dtype = Core::DataType::FLOAT32)
                {
                    int64_t n = 1;
                    for (auto d : dims) n *= d;

                    const size_t bpe = Core::bytes_per_element(dtype);
                    auto storage = SushiRuntime::make_sushi<Storage>(
                        TestEnvironment::get_context(),
                        static_cast<size_t>(kPad + n + kPad) * bpe,
                        SushiRuntime::API::Residency::Shared);

                    // Poisons the view too; each test writes its inputs first.
                    auto& q = TestEnvironment::get_context().get_queue();
                    q.memset(storage->allocation_base(), kPoisonByte, storage->size_bytes()).wait();

                    return Tensor(storage, dims, dtype, kPad, engine->get_layout());
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
                    auto& q = TestEnvironment::get_context().get_queue();
                    q.memcpy(host.data(), v.storage->allocation_base(), total).wait();

                    for (size_t i = 0; i < total; ++i)
                    {
                        if (i >= head && i < tail) continue;
                        if (host[i] != static_cast<unsigned char>(kPoisonByte))
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

        TEST_F(Unit_StorageOffsetTest, Level1AxpyHonoursOffset)
        {
            auto x = offset_view({4});
            auto y = offset_view({4});
            fill_tensor(x, {1, 2, 3, 4});
            fill_tensor(y, {10, 20, 30, 40});

            SushiBLAS::BLASOps(*engine).axpy(2.0, x, y);
            engine->execute();

            EXPECT_TENSOR_NEAR(y, std::vector<float>({12, 24, 36, 48}), 1e-5);
            EXPECT_TRUE(padding_intact(x, "axpy x"));
            EXPECT_TRUE(padding_intact(y, "axpy y"));
        }

        TEST_F(Unit_StorageOffsetTest, Level1ScalHonoursOffset)
        {
            auto x = offset_view({4});
            fill_tensor(x, {1, 2, 3, 4});

            SushiBLAS::BLASOps(*engine).scal(3.0, x);
            engine->execute();

            EXPECT_TENSOR_NEAR(x, std::vector<float>({3, 6, 9, 12}), 1e-5);
            EXPECT_TRUE(padding_intact(x, "scal x"));
        }

        TEST_F(Unit_StorageOffsetTest, Level1CopyHonoursOffset)
        {
            auto x = offset_view({4});
            auto y = offset_view({4});
            fill_tensor(x, {5, 6, 7, 8});
            fill_tensor(y, {0, 0, 0, 0});

            SushiBLAS::BLASOps(*engine).copy(x, y);
            engine->execute();

            EXPECT_TENSOR_NEAR(y, std::vector<float>({5, 6, 7, 8}), 1e-5);
            EXPECT_TRUE(padding_intact(x, "copy x"));
            EXPECT_TRUE(padding_intact(y, "copy y"));
        }

        TEST_F(Unit_StorageOffsetTest, Level2GemvHonoursOffset)
        {
            // beta != 0, so y is read as well as written.
            auto A = offset_view({2, 3});
            auto x = offset_view({3});
            auto y = offset_view({2});
            fill_tensor(A, {1, 2, 3, 4, 5, 6});
            fill_tensor(x, {1, 1, 1});
            fill_tensor(y, {100, 200});

            SushiBLAS::BLASOps(*engine).gemv(
                A, x, y, /*transA=*/false, /*alpha=*/1.0, /*beta=*/2.0);
            engine->execute();

            // A*x = {6, 15}; + 2*{100, 200} = {206, 415}
            EXPECT_TENSOR_NEAR(y, std::vector<float>({206, 415}), 1e-4);
            EXPECT_TRUE(padding_intact(A, "gemv A"));
            EXPECT_TRUE(padding_intact(x, "gemv x"));
            EXPECT_TRUE(padding_intact(y, "gemv y"));
        }

        TEST_F(Unit_StorageOffsetTest, Level2GerHonoursOffset)
        {
            auto x = offset_view({2});
            auto y = offset_view({3});
            auto A = offset_view({2, 3});
            fill_tensor(x, {1, 2});
            fill_tensor(y, {10, 20, 30});
            fill_tensor(A, {1, 1, 1, 1, 1, 1});

            SushiBLAS::BLASOps(*engine).ger(x, y, A, /*alpha=*/1.0);
            engine->execute();

            // A += x y^T = {{10,20,30},{20,40,60}}
            EXPECT_TENSOR_NEAR(A, std::vector<float>({11, 21, 31, 21, 41, 61}), 1e-4);
            EXPECT_TRUE(padding_intact(x, "ger x"));
            EXPECT_TRUE(padding_intact(y, "ger y"));
            EXPECT_TRUE(padding_intact(A, "ger A"));
        }

        TEST_F(Unit_StorageOffsetTest, Level3GemmHonoursOffset)
        {
            auto A = offset_view({2, 2});
            auto B = offset_view({2, 2});
            auto C = offset_view({2, 2});
            fill_tensor(A, {1, 2, 3, 4});
            fill_tensor(B, {1, 0, 0, 1});
            fill_tensor(C, {0, 0, 0, 0});

            SushiBLAS::BLASOps(*engine).gemm(A, B, C);
            engine->execute();

            EXPECT_TENSOR_NEAR(C, std::vector<float>({1, 2, 3, 4}), 1e-4);
            EXPECT_TRUE(padding_intact(A, "gemm A"));
            EXPECT_TRUE(padding_intact(B, "gemm B"));
            EXPECT_TRUE(padding_intact(C, "gemm C"));
        }

        TEST_F(Unit_StorageOffsetTest, ElementwiseHonoursOffset)
        {
            auto A = offset_view({4});
            auto B = offset_view({4});
            auto C = offset_view({4});
            fill_tensor(A, {1, 2, 3, 4});
            fill_tensor(B, {10, 10, 10, 10});
            fill_tensor(C, {0, 0, 0, 0});

            SushiBLAS::ElementwiseOps(*engine).add(A, B, C);   // binary seam
            SushiBLAS::ElementwiseOps(*engine).neg(C);         // unary in-place seam
            engine->execute();

            EXPECT_TENSOR_NEAR(C, std::vector<float>({-11, -12, -13, -14}), 1e-5);
            EXPECT_TRUE(padding_intact(A, "add A"));
            EXPECT_TRUE(padding_intact(B, "add B"));
            EXPECT_TRUE(padding_intact(C, "add/neg C"));
        }

        TEST_F(Unit_StorageOffsetTest, NonlinearHonoursOffset)
        {
            auto t = offset_view({4});
            fill_tensor(t, {-2, -1, 1, 2});

            SushiBLAS::NonLinearOps(*engine).relu(t);          // forward seam
            engine->execute();

            EXPECT_TENSOR_NEAR(t, std::vector<float>({0, 0, 1, 2}), 1e-5);
            EXPECT_TRUE(padding_intact(t, "relu t"));
        }

        TEST_F(Unit_StorageOffsetTest, NonlinearBackwardHonoursOffset)
        {
            auto dy = offset_view({4});
            auto x = offset_view({4});
            auto dx = offset_view({4});
            fill_tensor(dy, {1, 1, 1, 1});
            fill_tensor(x, {-2, -1, 1, 2});
            fill_tensor(dx, {0, 0, 0, 0});

            SushiBLAS::NonLinearOps(*engine).relu_backward(dy, x, dx);
            engine->execute();

            EXPECT_TENSOR_NEAR(dx, std::vector<float>({0, 0, 1, 1}), 1e-5);
            EXPECT_TRUE(padding_intact(dy, "relu_backward dy"));
            EXPECT_TRUE(padding_intact(x, "relu_backward x"));
            EXPECT_TRUE(padding_intact(dx, "relu_backward dx"));
        }

        TEST_F(Unit_StorageOffsetTest, LogicHonoursOffset)
        {
            auto A = offset_view({4});
            auto B = offset_view({4});
            auto R = offset_view({4});
            fill_tensor(A, {1, 5, 3, 7});
            fill_tensor(B, {2, 2, 2, 2});
            fill_tensor(R, {0, 0, 0, 0});

            SushiBLAS::LogicOps(*engine).greater(A, B, R);
            engine->execute();

            EXPECT_TENSOR_NEAR(R, std::vector<float>({0, 1, 1, 1}), 1e-5);
            EXPECT_TRUE(padding_intact(A, "greater A"));
            EXPECT_TRUE(padding_intact(B, "greater B"));
            EXPECT_TRUE(padding_intact(R, "greater R"));
        }

        TEST_F(Unit_StorageOffsetTest, ReductionHonoursOffset)
        {
            auto t = offset_view({4});
            auto r = offset_view({1});
            fill_tensor(t, {1, 2, 3, 4});
            fill_tensor(r, {0});

            SushiBLAS::ReductionOps(*engine).sum(t, r);
            engine->execute();

            EXPECT_TENSOR_NEAR(r, std::vector<float>({10}), 1e-5);
            EXPECT_TRUE(padding_intact(t, "sum t"));
            EXPECT_TRUE(padding_intact(r, "sum result"));
        }

        TEST_F(Unit_StorageOffsetTest, RandomHonoursOffset)
        {
            auto c = offset_view({8});
            auto u = offset_view({8});

            SushiBLAS::RandomOps(*engine).constant(c, 7.0);    // the q.fill path
            SushiBLAS::RandomOps(*engine).uniform(u, 0.0, 1.0); // the Philox dispatch path
            engine->execute();

            EXPECT_TENSOR_NEAR(c, std::vector<float>(8, 7.0f), 1e-6);
            EXPECT_TRUE(padding_intact(c, "constant"));
            EXPECT_TRUE(padding_intact(u, "uniform"));

            // The uniform draw is checked for range only; test_uniform.cpp owns the distribution.
            std::vector<float> host(8);
            auto& q = TestEnvironment::get_context().get_queue();
            q.memcpy(host.data(), u.data(), host.size() * sizeof(float)).wait();
            for (float v : host)
            {
                EXPECT_GE(v, 0.0f);
                EXPECT_LE(v, 1.0f);
            }
        }

        TEST_F(Unit_StorageOffsetTest, OffsetIsCountedInElementsNotBytes)
        {
            // storage_offset counts elements, not bytes. See tests/functional/unit/README.md.
            auto f64 = offset_view({4}, Core::DataType::FLOAT64);

            const auto* base = static_cast<const char*>(f64.storage->allocation_base());
            const auto* view = static_cast<const char*>(f64.data());

            EXPECT_EQ(view - base, kPad * static_cast<int64_t>(sizeof(double)))
                << "storage_offset must be scaled by the element width, not used as a byte count.";
            EXPECT_EQ(reinterpret_cast<std::uintptr_t>(view) % sizeof(double), 0u)
                << "an element-counted offset over an aligned allocation is always naturally aligned.";

            // And the region handed to the dependency tracker describes the same bytes.
            const auto region = f64.region();
            EXPECT_EQ(region.base, f64.storage->allocation_base());
            EXPECT_EQ(region.offset, static_cast<size_t>(kPad) * sizeof(double));
            EXPECT_EQ(region.length, 4u * sizeof(double));
        }

        TEST_F(Unit_StorageOffsetTest, DisjointViewsOfOneAllocationDoNotAlias)
        {
            // Disjoint views of one arena must not alias. See tests/functional/unit/README.md.
            auto storage = SushiRuntime::make_sushi<Storage>(
                TestEnvironment::get_context(),
                32 * sizeof(float),
                SushiRuntime::API::Residency::Shared);

            Tensor first(storage, {8}, Core::DataType::FLOAT32, 0);
            Tensor second(storage, {8}, Core::DataType::FLOAT32, 8);
            Tensor overlapping(storage, {8}, Core::DataType::FLOAT32, 4);

            EXPECT_FALSE(first.region().overlaps(second.region()));
            EXPECT_TRUE(first.region().overlaps(overlapping.region()));
            EXPECT_TRUE(second.region().overlaps(overlapping.region()));
            EXPECT_TRUE(first.region().overlaps(first.region()));
        }

        TEST_F(Unit_StorageOffsetTest, TheAllocationBaseIsAlignedForEveryResidency)
        {
            const SushiRuntime::API::Residency residencies[] = {
                SushiRuntime::API::Residency::Shared,
                SushiRuntime::API::Residency::Host,
                SushiRuntime::API::Residency::Device};
            const int64_t element_counts[] = {1, 7, 1000};

            for (const SushiRuntime::API::Residency residency : residencies)
            {
                for (const int64_t count : element_counts)
                {
                    auto storage = SushiRuntime::make_sushi<Storage>(
                        TestEnvironment::get_context(),
                        static_cast<size_t>(count) * sizeof(float),
                        residency);
                    const Tensor t(storage, {count}, Core::DataType::FLOAT32);

                    SCOPED_TRACE(::testing::Message() << "residency " << static_cast<int>(residency)
                                                      << ", " << count << " elements");

                    const auto base = reinterpret_cast<std::uintptr_t>(storage->allocation_base());
                    EXPECT_EQ(base % K_STORAGE_ALIGNMENT, 0u);
                    EXPECT_TRUE(t.is_aligned());
                }
            }
        }

        TEST_F(Unit_StorageOffsetTest, TheAllocatedSizeIsAMultipleOfTheAlignment)
        {
            auto one_byte = SushiRuntime::make_sushi<Storage>(TestEnvironment::get_context(), 1);
            auto past_one_block =
                SushiRuntime::make_sushi<Storage>(TestEnvironment::get_context(), 129);

            EXPECT_EQ(one_byte->size_bytes(), 128u);
            EXPECT_EQ(past_one_block->size_bytes(), 256u);
        }

        TEST_F(Unit_StorageOffsetTest, AViewOffTheBaseReportsItsOwnAlignment)
        {
            const int64_t n = 64;
            auto storage = SushiRuntime::make_sushi<Storage>(
                TestEnvironment::get_context(), static_cast<size_t>(n) * sizeof(float));
            const Tensor whole(storage, {n}, Core::DataType::FLOAT32);

            // One float past the base is 4 bytes in; 32 floats past it are 128 bytes in.
            EXPECT_FALSE(whole.slice(0, 1, n).is_aligned());
            EXPECT_TRUE(whole.slice(0, 32, n).is_aligned());
        }

        TEST_F(Unit_StorageOffsetTest, GetDeviceAnswersWithoutTheRegistry)
        {
            auto storage = SushiRuntime::make_sushi<Storage>(
                TestEnvironment::get_context(), 4 * sizeof(float));
            const Tensor t(storage, {4}, Core::DataType::FLOAT32);

            auto& context = TestEnvironment::get_context();
            EXPECT_EQ(t.get_device(), context.get_queue(storage->get_device_index()).get_device());
        }

    } // namespace Test
} // namespace SushiBLAS
