/**************************************************************************/
/* test_gemm_split.cpp                                                    */
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
 * @file test_gemm_split.cpp
 * @brief Tests the axis, the sub-problem, the offset epilogue and the slice of a GEMM split.
 * @author Mustafa Garip
 */

#include <atomic>
#include <thread>
#include <vector>
#include <optional>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <SushiBLAS/blas/gemm_split.hpp>
#include <SushiBLAS/blas_seam_kernels/reference/level3/gemm.hpp>
#include <SushiBLAS/test_runner/test_environment.hpp>
#include <SushiRuntime/handle/bound_operands.hpp>
#include "gemm_split_cases.hpp"

namespace SushiBLAS
{
    namespace Test
    {
        using Core::Layout;
        using Core::Transpose;
        using Dispatch::GemmGeometry;
        using Dispatch::GemmSplitAxis;

        using GemmSplitCases::PositionEpilogue;
        using GemmSplitCases::SplitCase;
        using GemmSplitCases::geometry_of;
        using GemmSplitCases::values;

        TEST(Small_GemmSplitAxis, TheLongerSideOfCIsCut)
        {
            GemmGeometry tall;
            tall.m = 8;
            tall.n = 3;
            GemmGeometry wide;
            wide.m = 3;
            wide.n = 8;

            EXPECT_EQ(Dispatch::gemm_split_axis(tall), GemmSplitAxis::Rows);
            EXPECT_EQ(Dispatch::gemm_split_axis(wide), GemmSplitAxis::Columns);
        }

        TEST(Small_GemmSplitAxis, ASquareCIsCutAlongItsRows)
        {
            GemmGeometry square;
            square.m = 4;
            square.n = 4;

            EXPECT_EQ(Dispatch::gemm_split_axis(square), GemmSplitAxis::Rows);
        }

        TEST(Small_GemmSplitAxis, TheExtentIsTheLengthOfTheAxis)
        {
            GemmGeometry geometry;
            geometry.m = 8;
            geometry.n = 3;

            EXPECT_EQ(Dispatch::gemm_split_extent(geometry, GemmSplitAxis::Rows), int64_t{8});
            EXPECT_EQ(Dispatch::gemm_split_extent(geometry, GemmSplitAxis::Columns), int64_t{3});
        }

        TEST(Small_GemmSlice, ARowPartOfARowMajorProductStepsAByItsLeadingDimension)
        {
            const SplitCase test_case{Layout::RowMajor, Transpose::NoTrans, Transpose::NoTrans,
                                      6, 4, 3};

            const Dispatch::GemmSlice slice = Dispatch::slice_gemm(
                geometry_of(test_case), test_case.layout, test_case.trans_a, test_case.trans_b,
                GemmSplitAxis::Rows, 2, 5);

            EXPECT_EQ(slice.m, int64_t{3});
            EXPECT_EQ(slice.n, int64_t{4});
            EXPECT_EQ(slice.a_offset, int64_t{2 * 3});
            EXPECT_EQ(slice.b_offset, int64_t{0});
            EXPECT_EQ(slice.c_offset, int64_t{2 * 4});
        }

        TEST(Small_GemmSlice, AColumnPartOfARowMajorProductStepsBByOne)
        {
            const SplitCase test_case{Layout::RowMajor, Transpose::NoTrans, Transpose::NoTrans,
                                      3, 6, 4};

            const Dispatch::GemmSlice slice = Dispatch::slice_gemm(
                geometry_of(test_case), test_case.layout, test_case.trans_a, test_case.trans_b,
                GemmSplitAxis::Columns, 2, 5);

            EXPECT_EQ(slice.m, int64_t{3});
            EXPECT_EQ(slice.n, int64_t{3});
            EXPECT_EQ(slice.a_offset, int64_t{0});
            EXPECT_EQ(slice.b_offset, int64_t{2});
            EXPECT_EQ(slice.c_offset, int64_t{2});
        }

        TEST(Small_GemmSlice, ATransposedAStepsByOneRowMajor)
        {
            const SplitCase test_case{Layout::RowMajor, Transpose::Trans, Transpose::NoTrans,
                                      6, 4, 3};

            const Dispatch::GemmSlice slice = Dispatch::slice_gemm(
                geometry_of(test_case), test_case.layout, test_case.trans_a, test_case.trans_b,
                GemmSplitAxis::Rows, 2, 5);

            EXPECT_EQ(slice.a_offset, int64_t{2});
        }

        TEST(Small_GemmSlice, AColumnMajorProductStepsCByOneAlongItsRows)
        {
            const SplitCase test_case{Layout::ColumnMajor, Transpose::NoTrans, Transpose::NoTrans,
                                      6, 4, 3};

            const Dispatch::GemmSlice slice = Dispatch::slice_gemm(
                geometry_of(test_case), test_case.layout, test_case.trans_a, test_case.trans_b,
                GemmSplitAxis::Rows, 2, 5);

            EXPECT_EQ(slice.a_offset, int64_t{2});
            EXPECT_EQ(slice.c_offset, int64_t{2});
        }

        TEST(Small_OffsetEpilogue, TheFunctorSeesThePositionInTheWholeMatrix)
        {
            Dispatch::OffsetEpilogue<PositionEpilogue> epilogue(PositionEpilogue{}, 3, 5);

            EXPECT_EQ(epilogue(1.0f, 2, 4), 1.0f + static_cast<float>(100 * 5 + 9));
        }

        /** @brief Runs a product whole on the host and in parts on the queue, and compares the two. */
        class Medium_GemmSplitSlice : public ::testing::TestWithParam<SplitCase>
        {
            protected:
                /** @brief Where a part reads its operands from. */
                enum class Operands
                {
                    Recorded, /**< The pointers the call recorded; the runtime bound none. */
                    Bound     /**< The pointers the runtime bound; the call recorded null. */
                };

                /** @brief Returns the queue of the test environment. */
                [[nodiscard]] static sycl::queue& queue()
                {
                    return TestEnvironment::queue();
                }

                /**
                 * @brief Returns the C that three consecutive parts of the declared split compute.
                 * @param test_case The product to compute.
                 * @param a The elements of A.
                 * @param b The elements of B.
                 * @param source Whether the parts read the recorded or the bound pointers.
                 */
                [[nodiscard]] static std::vector<float> in_parts(const SplitCase& test_case,
                                                                 const std::vector<float>& a,
                                                                 const std::vector<float>& b,
                                                                 Operands source)
                {
                    sycl::queue& q = queue();
                    const std::size_t c_count = static_cast<std::size_t>(test_case.m * test_case.n);
                    float* device_a = sycl::malloc_shared<float>(a.size(), q);
                    float* device_b = sycl::malloc_shared<float>(b.size(), q);
                    float* device_c = sycl::malloc_shared<float>(c_count, q);
                    q.memcpy(device_a, a.data(), a.size() * sizeof(float)).wait();
                    q.memcpy(device_b, b.data(), b.size() * sizeof(float)).wait();
                    q.memset(device_c, 0, c_count * sizeof(float)).wait();

                    const bool bound = source == Operands::Bound;
                    const auto split = Dispatch::make_gemm_split<float>(
                        geometry_of(test_case), test_case.layout, test_case.trans_a,
                        test_case.trans_b, 1.0f, bound ? nullptr : device_a,
                        bound ? nullptr : device_b, 0.0f, bound ? nullptr : device_c,
                        PositionEpilogue{}, Dispatch::GemmOperandRegions{},
                        bound ? std::optional<Dispatch::GemmBindings>(Dispatch::GemmBindings{
                                    OperandBinding{0, 0}, OperandBinding{1, 0},
                                    OperandBinding{2, 0}})
                              : std::nullopt);
                    void* pointers[Dispatch::K_GEMM_OPERAND_COUNT] = {device_a, device_b, device_c};
                    const SushiRuntime::BoundOperands table =
                        bound ? SushiRuntime::BoundOperands::of_bytes(
                                    pointers, Dispatch::K_GEMM_OPERAND_COUNT)
                              : SushiRuntime::BoundOperands();

                    const std::uint64_t extent = split->extent;
                    const std::uint64_t first = extent / 3;
                    const std::uint64_t second = extent - extent / 3;
                    const std::uint64_t bounds[4] = {0, first, second, extent};
                    std::vector<sycl::event> parts;
                    for (std::size_t index = 0; index < 3; ++index)
                    {
                        parts.push_back(split->slice(SushiRuntime::SplitPart{split->axis, bounds[index],
                                                                             bounds[index + 1], index, 3},
                                                     q, SushiRuntime::span<const sycl::event>(), table));
                    }
                    sycl::event::wait(parts);

                    std::vector<float> c(c_count);
                    q.memcpy(c.data(), device_c, c_count * sizeof(float)).wait();
                    sycl::free(device_a, q);
                    sycl::free(device_b, q);
                    sycl::free(device_c, q);
                    return c;
                }

                /** @brief Returns the C that the whole product computes on the host. */
                [[nodiscard]] static std::vector<float> whole(const SplitCase& test_case,
                                                              const std::vector<float>& a,
                                                              const std::vector<float>& b)
                {
                    std::vector<float> c(static_cast<std::size_t>(test_case.m * test_case.n), 0.0f);
                    const Dispatch::GemmGeometry geometry = geometry_of(test_case);
                    Kernels::Reference::gemm<float>(
                        test_case.layout, test_case.trans_a, test_case.trans_b, test_case.m,
                        test_case.n, test_case.k, 1.0f, a.data(), geometry.a.leading_dimension,
                        b.data(), geometry.b.leading_dimension, 0.0f, c.data(),
                        geometry.c.leading_dimension, PositionEpilogue{});
                    return c;
                }
        };

        TEST_P(Medium_GemmSplitSlice, ThePartsOnTheQueueComputeTheWholeProduct)
        {
            const SplitCase test_case = GetParam();
            const std::vector<float> a = values(static_cast<std::size_t>(test_case.m * test_case.k), 1);
            const std::vector<float> b = values(static_cast<std::size_t>(test_case.k * test_case.n), 4);

            EXPECT_EQ(in_parts(test_case, a, b, Operands::Recorded), whole(test_case, a, b));
        }

        TEST_P(Medium_GemmSplitSlice, ThePartsReadThePointersTheRuntimeBound)
        {
            const SplitCase test_case = GetParam();
            const std::vector<float> a = values(static_cast<std::size_t>(test_case.m * test_case.k), 1);
            const std::vector<float> b = values(static_cast<std::size_t>(test_case.k * test_case.n), 4);

            EXPECT_EQ(in_parts(test_case, a, b, Operands::Bound), whole(test_case, a, b));
        }

        /**
         * @brief Tests that two parts of a batched row-major GEMM, cut along the columns of C, give
         *        the product the host reference gives.
         */
        TEST(Medium_GemmSplitBatched, TwoColumnPartsOfABatchedProductGiveTheHostReference)
        {
            constexpr int64_t K_M = 3;
            constexpr int64_t K_N = 7;
            constexpr int64_t K_K = 4;
            constexpr int64_t K_BATCH = 2;
            GemmGeometry geometry = geometry_of(SplitCase{Layout::RowMajor, Transpose::NoTrans,
                                                          Transpose::NoTrans, K_M, K_N, K_K});
            geometry.batch_size = K_BATCH;
            geometry.stride_a = K_M * K_K;
            geometry.stride_b = K_K * K_N;
            geometry.stride_c = K_M * K_N;
            const std::vector<float> a = values(static_cast<std::size_t>(K_BATCH * K_M * K_K), 1);
            const std::vector<float> b = values(static_cast<std::size_t>(K_BATCH * K_K * K_N), 4);
            const std::size_t c_count = static_cast<std::size_t>(K_BATCH * K_M * K_N);
            std::vector<float> expected(c_count, 0.0f);
            Kernels::Reference::gemm<float>(Layout::RowMajor, Transpose::NoTrans, Transpose::NoTrans,
                                            K_M, K_N, K_K, 1.0f, a.data(), geometry.a.leading_dimension,
                                            geometry.stride_a, b.data(), geometry.b.leading_dimension,
                                            geometry.stride_b, 0.0f, expected.data(),
                                            geometry.c.leading_dimension, geometry.stride_c, K_BATCH,
                                            PositionEpilogue{});
            sycl::queue& q = TestEnvironment::queue();
            float* device_a = sycl::malloc_shared<float>(a.size(), q);
            float* device_b = sycl::malloc_shared<float>(b.size(), q);
            float* device_c = sycl::malloc_shared<float>(c_count, q);
            q.memcpy(device_a, a.data(), a.size() * sizeof(float)).wait();
            q.memcpy(device_b, b.data(), b.size() * sizeof(float)).wait();
            q.memset(device_c, 0, c_count * sizeof(float)).wait();
            const auto split = Dispatch::make_gemm_split<float>(
                geometry, Layout::RowMajor, Transpose::NoTrans, Transpose::NoTrans, 1.0f, device_a,
                device_b, 0.0f, device_c, PositionEpilogue{}, Dispatch::GemmOperandRegions{});

            const std::vector<sycl::event> parts = {
                split->slice(SushiRuntime::SplitPart{split->axis, 0, 4, 0, 2}, q,
                             SushiRuntime::span<const sycl::event>(), SushiRuntime::BoundOperands()),
                split->slice(SushiRuntime::SplitPart{split->axis, 4, 7, 1, 2}, q,
                             SushiRuntime::span<const sycl::event>(), SushiRuntime::BoundOperands())};
            sycl::event::wait(parts);

            std::vector<float> c(c_count);
            q.memcpy(c.data(), device_c, c_count * sizeof(float)).wait();
            sycl::free(device_a, q);
            sycl::free(device_b, q);
            sycl::free(device_c, q);
            EXPECT_EQ(split->axis, static_cast<std::uint32_t>(GemmSplitAxis::Columns));
            EXPECT_EQ(c, expected);
        }

        /**
         * @brief Tests that a slice returns its part's event without waiting, and that the part
         *        runs only after the event it is given.
         */
        TEST(Medium_GemmSplitEvent, ASliceReturnsItsEventWithoutWaiting)
        {
            const SplitCase test_case{Layout::RowMajor, Transpose::NoTrans, Transpose::NoTrans, 4, 3, 2};
            const std::vector<float> a = values(static_cast<std::size_t>(test_case.m * test_case.k), 1);
            const std::vector<float> b = values(static_cast<std::size_t>(test_case.k * test_case.n), 4);
            const std::size_t c_count = static_cast<std::size_t>(test_case.m * test_case.n);
            const GemmGeometry geometry = geometry_of(test_case);
            std::vector<float> expected(c_count, 0.0f);
            Kernels::Reference::gemm<float>(test_case.layout, test_case.trans_a, test_case.trans_b, test_case.m,
                                            test_case.n, test_case.k, 1.0f, a.data(), geometry.a.leading_dimension,
                                            b.data(), geometry.b.leading_dimension, 0.0f, expected.data(),
                                            geometry.c.leading_dimension, PositionEpilogue{});
            sycl::queue& q = TestEnvironment::queue();
            float* device_a = sycl::malloc_shared<float>(a.size(), q);
            float* device_b = sycl::malloc_shared<float>(b.size(), q);
            float* device_c = sycl::malloc_shared<float>(c_count, q);
            q.memcpy(device_a, a.data(), a.size() * sizeof(float)).wait();
            q.memcpy(device_b, b.data(), b.size() * sizeof(float)).wait();
            q.memset(device_c, 0, c_count * sizeof(float)).wait();
            const auto split = Dispatch::make_gemm_split<float>(
                geometry, test_case.layout, test_case.trans_a, test_case.trans_b, 1.0f, device_a, device_b,
                0.0f, device_c, PositionEpilogue{}, Dispatch::GemmOperandRegions{});

            // A host step that holds until the test opens it stands for a producer still running.
            std::atomic<bool> open{false};
            const sycl::event producer = q.submit([&open](sycl::handler& handler)
            {
                const auto hold = [&open]()
                {
                    while (!open.load(std::memory_order_acquire))
                        std::this_thread::yield();
                };
#if defined(__ACPP__) || defined(__HIPSYCL__)
                handler.AdaptiveCpp_enqueue_custom_operation([hold](auto&&) { hold(); });
#else
                handler.host_task(hold);
#endif
            });
            sycl::event part = split->slice(
                SushiRuntime::SplitPart{split->axis, 0, split->extent, 0, 1}, q,
                SushiRuntime::span<const sycl::event>(&producer, 1), SushiRuntime::BoundOperands());

            EXPECT_NE(part.get_info<sycl::info::event::command_execution_status>(),
                      sycl::info::event_command_status::complete);
            open.store(true, std::memory_order_release);
            part.wait();

            std::vector<float> c(c_count);
            q.memcpy(c.data(), device_c, c_count * sizeof(float)).wait();
            sycl::free(device_a, q);
            sycl::free(device_b, q);
            sycl::free(device_c, q);
            EXPECT_EQ(c, expected);
        }

        INSTANTIATE_TEST_SUITE_P(
            , Medium_GemmSplitSlice,
            ::testing::Values(
                SplitCase{Core::Layout::RowMajor, Core::Transpose::NoTrans, Core::Transpose::NoTrans, 9, 4, 5},
                SplitCase{Core::Layout::RowMajor, Core::Transpose::NoTrans, Core::Transpose::NoTrans, 4, 9, 5},
                SplitCase{Core::Layout::RowMajor, Core::Transpose::Trans, Core::Transpose::NoTrans, 9, 4, 5},
                SplitCase{Core::Layout::RowMajor, Core::Transpose::NoTrans, Core::Transpose::Trans, 4, 9, 5},
                SplitCase{Core::Layout::ColumnMajor, Core::Transpose::NoTrans, Core::Transpose::NoTrans, 9, 4, 5},
                SplitCase{Core::Layout::ColumnMajor, Core::Transpose::NoTrans, Core::Transpose::NoTrans, 4, 9, 5},
                SplitCase{Core::Layout::ColumnMajor, Core::Transpose::Trans, Core::Transpose::Trans, 9, 4, 5},
                SplitCase{Core::Layout::ColumnMajor, Core::Transpose::Trans, Core::Transpose::Trans, 4, 9, 5}));
    } // namespace Test
} // namespace SushiBLAS
