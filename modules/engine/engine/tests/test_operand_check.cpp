/**************************************************************************/
/* test_operand_check.cpp                                                 */
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
 * @file test_operand_check.cpp
 * @brief Tests the operand checks, the scalar binding and the Tensor constructor and transpose guards.
 * @author Mustafa Garip
 */

#include <array>
#include <cstddef>
#include <string>
#include <cstdint>
#include <stdexcept>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.h>
#include <SushiBLAS/engine/dispatch.hpp>
#include <SushiBLAS/engine/operand_check.hpp>
#include <SushiBLAS/launch/device_policy.hpp>
#include <SushiBLAS/functor_map/scalar.hpp>
#include <SushiBLAS/functor_map/scalar_binding.hpp>
#include <SushiRuntime/core/sushi_ptr.hpp>
#include "test_utils.hpp"

namespace SushiBLAS
{
    namespace Test
    {
        class Unit_OperandCheckTest : public SushiBLASTest
        {
            protected:
                /** @brief Returns a storage of `elements` FLOAT32 elements in shared memory. */
                SushiRuntime::sushi_ptr<Storage> make_storage(int64_t elements)
                {
                    return SushiRuntime::make_sushi<Storage>(
                        TestEnvironment::runtime(),
                        static_cast<size_t>(elements) * sizeof(float),
                        SushiRuntime::Residency::Shared);
                }

                /** @brief Returns a dense FLOAT32 tensor over a storage of its own. */
                Tensor make_tensor(std::initializer_list<int64_t> dims,
                                   Core::Layout layout = Core::Layout::RowMajor)
                {
                    int64_t n = 1;
                    for (auto d : dims)
                        n *= d;
                    return Tensor(make_storage(n), dims, Core::DataType::Float32, 0, layout);
                }

                /** @brief Returns the text of the std::invalid_argument a check throws, or "". */
                template<typename Check>
                static std::string message_of(const Check& check)
                {
                    try
                    {
                        check();
                    }
                    catch (const std::invalid_argument& error)
                    {
                        return error.what();
                    }

                    return std::string();
                }
        };

        TEST_F(Unit_OperandCheckTest, RequireStorageAcceptsTensorWithStorage)
        {
            auto t = make_tensor({4});
            EXPECT_NO_THROW(Dispatch::require_storage(t, "op", "x"));
        }

        TEST_F(Unit_OperandCheckTest, RequireStorageRejectsTensorWithoutStorage)
        {
            Tensor t(SushiRuntime::sushi_ptr<Storage>(), {4});
            EXPECT_THROW(Dispatch::require_storage(t, "op", "x"), std::invalid_argument);
        }

        TEST_F(Unit_OperandCheckTest, RequireDtypeAcceptsMatchingDtype)
        {
            auto t = make_tensor({4});
            EXPECT_NO_THROW(Dispatch::require_dtype(t, Core::DataType::Float32, "op", "x"));
        }

        TEST_F(Unit_OperandCheckTest, RequireDtypeRejectsOtherDtype)
        {
            auto t = make_tensor({4});
            EXPECT_THROW(Dispatch::require_dtype(t, Core::DataType::Float64, "op", "x"),
                         std::invalid_argument);
        }

        TEST_F(Unit_OperandCheckTest, RequireSameShapeAcceptsEqualShapes)
        {
            auto a = make_tensor({2, 3});
            auto b = make_tensor({2, 3});
            EXPECT_NO_THROW(Dispatch::require_same_shape(a, b, "op", "a", "b"));
        }

        TEST_F(Unit_OperandCheckTest, RequireSameShapeRejectsSameCountOtherShape)
        {
            auto a = make_tensor({2, 3});
            auto b = make_tensor({3, 2});
            EXPECT_THROW(Dispatch::require_same_shape(a, b, "op", "a", "b"), std::invalid_argument);
        }

        TEST_F(Unit_OperandCheckTest, RequireSameLayoutAcceptsEqualLayouts)
        {
            auto a = make_tensor({2, 3}, Core::Layout::ColumnMajor);
            auto b = make_tensor({2, 3}, Core::Layout::ColumnMajor);
            EXPECT_NO_THROW(Dispatch::require_same_layout(a, b, "op", "a", "b"));
        }

        TEST_F(Unit_OperandCheckTest, RequireSameLayoutAcceptsRankOneOperands)
        {
            auto a = make_tensor({4});
            auto b = make_tensor({4}, Core::Layout::ColumnMajor);
            EXPECT_NO_THROW(Dispatch::require_same_layout(a, b, "op", "a", "b"));
        }

        TEST_F(Unit_OperandCheckTest, RequireSameLayoutRejectsOtherLayout)
        {
            auto a = make_tensor({2, 3});
            auto b = make_tensor({2, 3}, Core::Layout::ColumnMajor);
            EXPECT_THROW(Dispatch::require_same_layout(a, b, "op", "a", "b"), std::invalid_argument);
        }

        TEST_F(Unit_OperandCheckTest, RequireSameElementCountAcceptsReshapedOperand)
        {
            auto a = make_tensor({2, 3});
            auto b = make_tensor({6});
            EXPECT_NO_THROW(Dispatch::require_same_element_count(a, b, "op", "a", "b"));
        }

        TEST_F(Unit_OperandCheckTest, RequireSameElementCountRejectsOtherCount)
        {
            auto a = make_tensor({2, 3});
            auto b = make_tensor({5});
            EXPECT_THROW(Dispatch::require_same_element_count(a, b, "op", "a", "b"),
                         std::invalid_argument);
        }

        TEST_F(Unit_OperandCheckTest, RequireMinElementsAcceptsExactMinimum)
        {
            auto t = make_tensor({4});
            EXPECT_NO_THROW(Dispatch::require_min_elements(t, 4, "op", "x"));
        }

        TEST_F(Unit_OperandCheckTest, RequireMinElementsRejectsTooFew)
        {
            auto t = make_tensor({4});
            EXPECT_THROW(Dispatch::require_min_elements(t, 5, "op", "x"), std::invalid_argument);
        }

        TEST_F(Unit_OperandCheckTest, RequireContiguousAcceptsRowMajorTensor)
        {
            auto t = make_tensor({2, 3});
            EXPECT_NO_THROW(Dispatch::require_contiguous(t, "op", "x"));
        }

        TEST_F(Unit_OperandCheckTest, RequireContiguousAcceptsColumnMajorTensor)
        {
            auto t = make_tensor({2, 3}, Core::Layout::ColumnMajor);
            ASSERT_EQ(t.strides[0], 1);
            ASSERT_EQ(t.strides[1], 2);
            EXPECT_NO_THROW(Dispatch::require_contiguous(t, "op", "x"));
        }

        TEST_F(Unit_OperandCheckTest, RequireContiguousAcceptsStorageOffsetView)
        {
            Tensor t(make_storage(16), {2, 3}, Core::DataType::Float32, 8);
            EXPECT_NO_THROW(Dispatch::require_contiguous(t, "op", "x"));
        }

        TEST_F(Unit_OperandCheckTest, RequireContiguousRejectsTransposeView)
        {
            // Strides {3, 1} swap to {1, 3} on shape {3, 2}; dense row-major needs {2, 1}.
            auto t = make_tensor({2, 3}).transpose(0, 1);
            EXPECT_THROW(Dispatch::require_contiguous(t, "op", "x"), std::invalid_argument);
        }

        TEST_F(Unit_OperandCheckTest, RequireContiguousRejectsStrideZeroView)
        {
            const std::array<int64_t, Core::K_MAX_TENSOR_RANK> strides{0};
            const std::array<int64_t, 1> dims{4};
            Tensor t(make_storage(1), SushiRuntime::span<const int64_t>(dims.data(), dims.size()),
                     strides, Core::DataType::Float32, 0, Core::Layout::RowMajor);
            EXPECT_THROW(Dispatch::require_contiguous(t, "op", "x"), std::invalid_argument);
        }

        TEST_F(Unit_OperandCheckTest, RequireContiguousRejectsSlicedView)
        {
            // Shape {4, 2} keeps strides {4, 1}; dense row-major needs {2, 1}.
            auto t = make_tensor({4, 4}).slice(1, 0, 2);
            EXPECT_THROW(Dispatch::require_contiguous(t, "op", "x"), std::invalid_argument);
        }

        TEST_F(Unit_OperandCheckTest, RequireRankNamesBothRanks)
        {
            auto t = make_tensor({2, 3});
            EXPECT_NO_THROW(Dispatch::require_rank(t, 2, "op", "x"));
            EXPECT_THROW(Dispatch::require_rank(t, 1, "op", "x"), std::invalid_argument);
            EXPECT_EQ(message_of([&] { Dispatch::require_rank(t, 1, "op", "x"); }),
                      "op: operand 'x' has rank 2, expected 1");
        }

        TEST_F(Unit_OperandCheckTest, RequireScalarRejectsZeroAndTwoElements)
        {
            Tensor none(SushiRuntime::sushi_ptr<Storage>(), {0});
            auto one = make_tensor({1});
            auto two = make_tensor({2});
            EXPECT_THROW(Dispatch::require_scalar(none, "op", "x"), std::invalid_argument);
            EXPECT_NO_THROW(Dispatch::require_scalar(one, "op", "x"));
            EXPECT_THROW(Dispatch::require_scalar(two, "op", "x"), std::invalid_argument);
        }

        TEST_F(Unit_OperandCheckTest, RequireExtentNamesTheExtent)
        {
            EXPECT_NO_THROW(Dispatch::require_extent(4, 4, "channels", "op", "x"));
            EXPECT_THROW(Dispatch::require_extent(3, 4, "channels", "op", "x"),
                         std::invalid_argument);
            EXPECT_EQ(message_of([] { Dispatch::require_extent(3, 4, "channels", "op", "x"); }),
                      "op: operand 'x' has 3 channels, expected 4");
        }

        TEST_F(Unit_OperandCheckTest, RequireOperandPassesItsFormatThrough)
        {
            EXPECT_NO_THROW(Dispatch::require_operand(true, "op", "x", "is {}", 7));
            EXPECT_THROW(Dispatch::require_operand(false, "op", "x", "is {}", 7),
                         std::invalid_argument);
            EXPECT_EQ(message_of([] { Dispatch::require_operand(false, "op", "x", "is {}", 7); }),
                      "op: operand 'x' is 7");
        }

        TEST_F(Unit_OperandCheckTest, ADtypeErrorIsAnInvalidArgument)
        {
            const auto raise = []
            {
                Dispatch::Detail::throw_dtype_error("Elementwise operation", "sqrt",
                                                    Core::DataType::Int32);
            };
            EXPECT_THROW(raise(), std::invalid_argument);

            const std::string message = message_of(raise);
            EXPECT_NE(message.find("Elementwise operation"), std::string::npos);
            EXPECT_NE(message.find("'sqrt'"), std::string::npos);
            EXPECT_NE(message.find(Core::to_string(Core::DataType::Int32)), std::string::npos);
        }

        TEST_F(Unit_OperandCheckTest, RequireDeviceDtypeAcceptsWhatTheDeviceOffers)
        {
            auto narrow = make_tensor({4});
            EXPECT_NO_THROW(Dispatch::require_device_dtype(narrow, "op", "x"));

            Tensor detached(SushiRuntime::sushi_ptr<Storage>(), {4}, Core::DataType::Float64);
            EXPECT_NO_THROW(Dispatch::require_device_dtype(detached, "op", "x"));

            // Eight FLOAT32 elements are the 32 bytes four FLOAT64 elements need.
            Tensor wide(make_storage(8), {4}, Core::DataType::Float64);
            if (Device::profile_for(wide.device()).supports_fp64)
                EXPECT_NO_THROW(Dispatch::require_device_dtype(wide, "op", "x"));
            else
                EXPECT_THROW(Dispatch::require_device_dtype(wide, "op", "x"),
                             std::invalid_argument);
        }

        TEST_F(Unit_OperandCheckTest, ANarrowedOpHoldsOnlyTheElementType)
        {
            const auto clamp = [](auto x, auto low, auto high)
            {
                return x < low ? low : (x > high ? high : x);
            };
            const auto bound = Dispatch::bind_scalars(clamp, Dispatch::ElementScalar(0.0),
                                                      Dispatch::ElementScalar(1.0));

            EXPECT_LE(sizeof(bound.narrow<float>()), 3 * sizeof(float));
            EXPECT_FLOAT_EQ(bound.narrow<float>()(3.0f), 1.0f);
        }

        TEST_F(Unit_OperandCheckTest, APlainFunctorPassesThroughNarrowFor)
        {
            const auto f = [](float x) { return x; };
            EXPECT_EQ(&Dispatch::Detail::narrow_for(Dispatch::ElementTag<float>{}, f), &f);
        }

        TEST_F(Unit_OperandCheckTest, TensorRejectsNegativeDimension)
        {
            EXPECT_THROW(Tensor(SushiRuntime::sushi_ptr<Storage>(), {-1, 2}), std::runtime_error);
        }

        TEST_F(Unit_OperandCheckTest, TensorRejectsOverflowingElementCount)
        {
            const int64_t big = int64_t(1) << 32;
            EXPECT_THROW(Tensor(SushiRuntime::sushi_ptr<Storage>(), {big, big}), std::runtime_error);
        }

        TEST_F(Unit_OperandCheckTest, TensorRejectsRankAboveLimit)
        {
            EXPECT_THROW(Tensor(SushiRuntime::sushi_ptr<Storage>(), {1, 1, 1, 1, 1, 1, 1}),
                         std::runtime_error);
        }

        TEST_F(Unit_OperandCheckTest, TransposeRejectsNegativeDimensionIndex)
        {
            auto t = make_tensor({2, 3});
            EXPECT_THROW(t.transpose(-1, 0), std::runtime_error);
        }

        TEST_F(Unit_OperandCheckTest, TransposeRejectsDimensionIndexAtRank)
        {
            auto t = make_tensor({2, 3});
            EXPECT_THROW(t.transpose(0, 2), std::runtime_error);
        }

        TEST_F(Unit_OperandCheckTest, RequireDeviceDtypeAcceptsHalfWhereTheDeviceOffersIt)
        {
            Tensor detached(SushiRuntime::sushi_ptr<Storage>(), {4}, Core::DataType::Half);
            EXPECT_NO_THROW(Dispatch::require_device_dtype(detached, "op", "x"));

            // Two FLOAT32 elements are the 8 bytes four HALF elements need.
            Tensor half(make_storage(2), {4}, Core::DataType::Half);
            const auto check = [&] { Dispatch::require_device_dtype(half, "op", "x"); };
            if (Device::profile_for(half.device()).supports_fp16)
            {
                EXPECT_NO_THROW(check());
            }
            else
            {
                EXPECT_THROW(check(), std::invalid_argument);
                EXPECT_NE(message_of(check).find("fp16"), std::string::npos);
            }
        }

        TEST_F(Unit_OperandCheckTest, TheProfileReportsTheFp16AspectOfItsDevice)
        {
            const sycl::queue& q = TestEnvironment::queue();
            EXPECT_EQ(Device::profile_for(q).supports_fp16,
                      q.get_device().has(sycl::aspect::fp16));
        }

    } // namespace Test
} // namespace SushiBLAS
