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
 * @file test_operand_check.cpp
 * @brief Tests the operand checks, the scalar binding and the Tensor constructor and transpose guards.
 * @author Mustafa Garip
 */

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/tensor/host_block.hpp>
#include <SushiBLAS/tensor/runtime_block.hpp>
#include <SushiBLAS/SushiBLAS.hpp>
#include <SushiBLAS/core/errors.hpp>
#include <SushiBLAS/engine/dispatch.hpp>
#include <SushiBLAS/engine/operand_check.hpp>
#include <SushiBLAS/launch/device_policy.hpp>
#include <SushiBLAS/functor_map/scalar.hpp>
#include <SushiBLAS/functor_map/scalar_binding.hpp>
#include <SushiRuntime/core/sushi_ptr.hpp>
#include <SushiBLAS/engine_fixture/sushiblas_test.hpp>
#include <SushiBLAS/test_runner/test_environment.hpp>
namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            /** @brief Names how a check ended. */
            enum class Ending : std::uint8_t
            {
                Accepted, /**< The check threw nothing. */
                Refused   /**< The check threw std::invalid_argument. */
            };

            /** @brief Returns how a check ended; any other exception leaves the test. */
            Ending ending_of(const std::function<void()>& check)
            {
                try
                {
                    check();
                }
                catch (const std::invalid_argument&)
                {
                    return Ending::Refused;
                }

                return Ending::Accepted;
            }

            /** @brief Returns the text of the std::invalid_argument a check throws, or "". */
            std::string message_of(const std::function<void()>& check)
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

            /** @brief Returns a dense FLOAT32 tensor over host storage of its own. */
            Tensor host_tensor(std::initializer_list<int64_t> dims,
                               Core::Layout layout = Core::Layout::RowMajor)
            {
                int64_t count = 1;
                for (const int64_t dim : dims)
                    count *= dim;
                return Tensor(make_host_storage(static_cast<std::size_t>(count) * sizeof(float)),
                              dims, Core::DataType::Float32, 0, layout);
            }

            /** @brief Returns a FLOAT32 tensor with no storage. */
            Tensor detached_tensor(std::initializer_list<int64_t> dims)
            {
                return Tensor(SushiRuntime::sushi_ptr<Storage>(), dims, Core::DataType::Float32);
            }

            /** @brief Names one check and how it must end. */
            struct CheckRow
            {
                const char* name;            /**< The row name a failure shows. */
                std::function<void()> check; /**< The call under test. */
                Ending expected;             /**< How the call must end. */
            };

            /** @brief Returns a row from a name, a check and its ending. */
            CheckRow make_row(const char* name, std::function<void()> check, Ending expected)
            {
                return {name, std::move(check), expected};
            }

            /** @brief Returns the rows of the storage, dtype, shape, layout and count checks. */
            std::vector<CheckRow> value_check_rows()
            {
                const Tensor with_storage = host_tensor({4});
                const Tensor without_storage = detached_tensor({4});
                const Tensor row_major = host_tensor({2, 3});
                const Tensor transposed_shape = host_tensor({3, 2});
                const Tensor column_major = host_tensor({2, 3}, Core::Layout::ColumnMajor);
                const Tensor flat_six = host_tensor({6});
                const Tensor flat_five = host_tensor({5});
                const Tensor flat_column = host_tensor({4}, Core::Layout::ColumnMajor);
                return {
                    make_row("StorageAcceptsATensorWithStorage",
                        [=] { Dispatch::require_storage(with_storage, "op", "x"); },
                        Ending::Accepted),
                    make_row("StorageRefusesATensorWithoutStorage",
                        [=] { Dispatch::require_storage(without_storage, "op", "x"); },
                        Ending::Refused),
                    make_row("DtypeAcceptsTheMatchingDtype",
                        [=] {
                            Dispatch::require_dtype(with_storage, Core::DataType::Float32, "op",
                                                    "x");
                        },
                        Ending::Accepted),
                    make_row("DtypeRefusesAnotherDtype",
                        [=] {
                            Dispatch::require_dtype(with_storage, Core::DataType::Float64, "op",
                                                    "x");
                        },
                        Ending::Refused),
                    make_row("SameShapeAcceptsEqualShapes",
                        [=] { Dispatch::require_same_shape(row_major, row_major, "op", "a", "b"); },
                        Ending::Accepted),
                    make_row("SameShapeRefusesTheSameCountInAnotherShape",
                        [=] {
                            Dispatch::require_same_shape(row_major, transposed_shape, "op", "a",
                                                         "b");
                        },
                        Ending::Refused),
                    make_row("SameLayoutAcceptsEqualLayouts",
                        [=] {
                            Dispatch::require_same_layout(column_major, column_major, "op", "a",
                                                          "b");
                        },
                        Ending::Accepted),
                    make_row("SameLayoutAcceptsRankOneOperands",
                        [=] {
                            Dispatch::require_same_layout(with_storage, flat_column, "op", "a",
                                                          "b");
                        },
                        Ending::Accepted),
                    make_row("SameLayoutRefusesAnotherLayout",
                        [=] {
                            Dispatch::require_same_layout(row_major, column_major, "op", "a", "b");
                        },
                        Ending::Refused),
                    make_row("SameElementCountAcceptsAReshapedOperand",
                        [=] {
                            Dispatch::require_same_element_count(row_major, flat_six, "op", "a",
                                                                 "b");
                        },
                        Ending::Accepted),
                    make_row("SameElementCountRefusesAnotherCount",
                        [=] {
                            Dispatch::require_same_element_count(row_major, flat_five, "op", "a",
                                                                 "b");
                        },
                        Ending::Refused),
                    make_row("MinElementsAcceptsTheExactMinimum",
                        [=] { Dispatch::require_min_elements(with_storage, 4, "op", "x"); },
                        Ending::Accepted),
                    make_row("MinElementsRefusesFewer",
                        [=] { Dispatch::require_min_elements(with_storage, 5, "op", "x"); },
                        Ending::Refused),
                    make_row("ScalarRefusesZeroElements",
                        [=] { Dispatch::require_scalar(detached_tensor({0}), "op", "x"); },
                        Ending::Refused),
                    make_row("ScalarAcceptsOneElement",
                        [=] { Dispatch::require_scalar(host_tensor({1}), "op", "x"); },
                        Ending::Accepted),
                    make_row("ScalarRefusesTwoElements",
                        [=] { Dispatch::require_scalar(host_tensor({2}), "op", "x"); },
                        Ending::Refused),
                    make_row("RankAcceptsTheExpectedRank",
                        [=] { Dispatch::require_rank(row_major, 2, "op", "x"); },
                        Ending::Accepted),
                    make_row("RankRefusesAnotherRank",
                        [=] { Dispatch::require_rank(row_major, 1, "op", "x"); },
                        Ending::Refused),
                    make_row("ExtentAcceptsEqualExtents",
                        [] { Dispatch::require_extent(4, 4, "channels", "op", "x"); },
                        Ending::Accepted),
                    make_row("ExtentRefusesUnequalExtents",
                        [] { Dispatch::require_extent(3, 4, "channels", "op", "x"); },
                        Ending::Refused),
                    make_row("OperandAcceptsATrueCondition",
                        [] { Dispatch::require_operand(true, "op", "x", "is {}", 7); },
                        Ending::Accepted),
                    make_row("OperandRefusesAFalseCondition",
                        [] { Dispatch::require_operand(false, "op", "x", "is {}", 7); },
                        Ending::Refused)};
            }

            /** @brief Returns the rows of the contiguity check. */
            std::vector<CheckRow> contiguity_rows()
            {
                const Tensor row_major = host_tensor({2, 3});
                const Tensor column_major = host_tensor({2, 3}, Core::Layout::ColumnMajor);
                // Eight elements leave room for the offset view that starts at element 2.
                const Tensor offset_view(make_host_storage(8 * sizeof(float)), {2, 3},
                                         Core::DataType::Float32, 2);
                // Strides {3, 1} swap to {1, 3} on shape {3, 2}; dense row-major needs {2, 1}.
                const Tensor transposed = host_tensor({2, 3}).transpose(0, 1);
                // Shape {4, 2} keeps strides {4, 1}; dense row-major needs {2, 1}.
                const Tensor sliced = host_tensor({4, 4}).slice(1, 0, 2);
                const std::array<int64_t, Core::K_MAX_TENSOR_RANK> zero_strides{0};
                const std::array<int64_t, 1> four{4};
                const Tensor broadcast(
                    make_host_storage(sizeof(float)),
                    SushiRuntime::span<const int64_t>(four.data(), four.size()), zero_strides,
                    Core::DataType::Float32, 0, Core::Layout::RowMajor);
                return {
                    make_row("RowMajorTensorIsContiguous",
                        [=] { Dispatch::require_contiguous(row_major, "op", "x"); },
                        Ending::Accepted),
                    make_row("ColumnMajorTensorIsContiguous",
                        [=] { Dispatch::require_contiguous(column_major, "op", "x"); },
                        Ending::Accepted),
                    make_row("StorageOffsetViewIsContiguous",
                        [=] { Dispatch::require_contiguous(offset_view, "op", "x"); },
                        Ending::Accepted),
                    make_row("TransposeViewIsRefused",
                        [=] { Dispatch::require_contiguous(transposed, "op", "x"); },
                        Ending::Refused),
                    make_row("StrideZeroViewIsRefused",
                        [=] { Dispatch::require_contiguous(broadcast, "op", "x"); },
                        Ending::Refused),
                    make_row("SlicedViewIsRefused",
                        [=] { Dispatch::require_contiguous(sliced, "op", "x"); },
                        Ending::Refused)};
            }
        } // namespace

        /** @brief Runs one row of a check list and compares how it ended. */
        class Small_OperandCheck : public ::testing::TestWithParam<CheckRow>
        {

        };

        TEST_P(Small_OperandCheck, EndsAsTheRowNames)
        {
            EXPECT_EQ(ending_of(GetParam().check), GetParam().expected);
        }

        INSTANTIATE_TEST_SUITE_P(
            Value, Small_OperandCheck, ::testing::ValuesIn(value_check_rows()),
            [](const ::testing::TestParamInfo<CheckRow>& info)
            { return std::string(info.param.name); });

        INSTANTIATE_TEST_SUITE_P(
            Contiguity, Small_OperandCheck, ::testing::ValuesIn(contiguity_rows()),
            [](const ::testing::TestParamInfo<CheckRow>& info)
            { return std::string(info.param.name); });

        /** @brief Tests that require_rank names both ranks. */
        TEST(Small_OperandCheckMessage, RankNamesBothRanks)
        {
            const Tensor matrix = host_tensor({2, 3});

            EXPECT_EQ(message_of([&] { Dispatch::require_rank(matrix, 1, "op", "x"); }),
                      "op: operand 'x' has rank 2, expected 1");
        }

        /** @brief Tests that require_extent names the extent. */
        TEST(Small_OperandCheckMessage, ExtentNamesTheExtent)
        {
            EXPECT_EQ(message_of([] { Dispatch::require_extent(3, 4, "channels", "op", "x"); }),
                      "op: operand 'x' has 3 channels, expected 4");
        }

        /** @brief Tests that require_operand passes its format through. */
        TEST(Small_OperandCheckMessage, OperandPassesItsFormatThrough)
        {
            EXPECT_EQ(message_of([] { Dispatch::require_operand(false, "op", "x", "is {}", 7); }),
                      "op: operand 'x' is 7");
        }

        /** @brief Tests that the dtype error is an invalid_argument naming family, name and dtype. */
        TEST(Small_OperandCheckMessage, ADtypeErrorNamesFamilyNameAndDtype)
        {
            const auto raise = []
            {
                Dispatch::Detail::throw_dtype_error("Elementwise operation", "sqrt",
                                                    Core::DataType::Int32);
            };

            const std::string message = message_of(raise);

            EXPECT_NE(message.find("Elementwise operation"), std::string::npos);
            EXPECT_NE(message.find("'sqrt'"), std::string::npos);
            EXPECT_NE(message.find(Core::to_string(Core::DataType::Int32)), std::string::npos);
        }

        /** @brief Tests that a narrowed bound functor holds at most its three float scalars. */
        TEST(Small_OperandCheckFunctor, ANarrowedOpHoldsOnlyTheElementType)
        {
            const auto clamp = [](auto x, auto low, auto high)
            {
                return x < low ? low : (x > high ? high : x);
            };
            const auto bound = Dispatch::bind_scalars(clamp, Dispatch::ElementScalar(0.0),
                                                      Dispatch::ElementScalar(1.0));

            EXPECT_LE(sizeof(bound.narrow<float>()), 3 * sizeof(float));
        }

        /** @brief Tests that the Tensor constructor refuses a negative dimension. */
        TEST(Small_OperandCheckTensor, RejectsANegativeDimension)
        {
            EXPECT_THROW(Tensor(SushiRuntime::sushi_ptr<Storage>(), {-1, 2}), std::runtime_error);
        }

        /** @brief Tests that the Tensor constructor refuses an element count that overflows. */
        TEST(Small_OperandCheckTensor, RejectsAnOverflowingElementCount)
        {
            const int64_t big = int64_t(1) << 32;

            EXPECT_THROW(Tensor(SushiRuntime::sushi_ptr<Storage>(), {big, big}),
                         std::runtime_error);
        }

        /** @brief Tests that the Tensor constructor refuses a rank above the limit. */
        TEST(Small_OperandCheckTensor, RejectsARankAboveTheLimit)
        {
            EXPECT_THROW(Tensor(SushiRuntime::sushi_ptr<Storage>(), {1, 1, 1, 1, 1, 1, 1}),
                         std::runtime_error);
        }

        /** @brief Tests that transpose refuses a negative dimension index. */
        TEST(Small_OperandCheckTensor, TransposeRejectsANegativeDimensionIndex)
        {
            const Tensor matrix = host_tensor({2, 3});

            EXPECT_THROW((void)matrix.transpose(-1, 0), std::runtime_error);
        }

        namespace
        {
            /** @brief Names one require_device_dtype call and the device aspect it needs. */
            struct DeviceDtypeRow
            {
                const char* name;      /**< The row name a failure shows. */
                Core::DataType dtype;  /**< The dtype of the tensor. */
                bool has_storage;      /**< Whether the tensor owns storage; one without needs no aspect. */
                bool Device::Profile::* aspect; /**< The profile aspect the dtype needs, or null for none. */
            };

            /** @brief Returns FLOAT32 on a device, then FLOAT64 and HALF with and without storage. */
            std::vector<DeviceDtypeRow> device_dtype_rows()
            {
                return {{"Float32OnTheDevice", Core::DataType::Float32, true, nullptr},
                        {"Float64WithoutStorage", Core::DataType::Float64, false, nullptr},
                        {"HalfWithoutStorage", Core::DataType::Half, false, nullptr},
                        {"Float64OnTheDevice", Core::DataType::Float64, true,
                         &Device::Profile::supports_fp64},
                        {"HalfOnTheDevice", Core::DataType::Half, true,
                         &Device::Profile::supports_fp16}};
            }

            /** @brief Returns the text of the UnsupportedExecution a call throws, or "". */
            std::string refusal_text_of(const std::function<void()>& call)
            {
                try
                {
                    call();
                }
                catch (const UnsupportedExecution& refusal)
                {
                    return refusal.what();
                }

                return std::string();
            }
        } // namespace

        /** @brief Runs one row of device_dtype_rows() against the process device. */
        class Medium_OperandCheckDeviceDtype : public SushiBLASTest,
                                               public ::testing::WithParamInterface<DeviceDtypeRow>
        {

        };

        TEST_P(Medium_OperandCheckDeviceDtype, RefusesWhatTheDeviceLacks)
        {
            const DeviceDtypeRow row = GetParam();
            // Eight FLOAT32 elements are the 32 bytes four FLOAT64 elements need.
            const Tensor tensor(row.has_storage
                                    ? make_runtime_storage(TestEnvironment::runtime(),
                                                           8 * sizeof(float),
                                                           SushiRuntime::Residency::Shared)
                                    : SushiRuntime::sushi_ptr<Storage>(),
                                {4}, row.dtype);
            const bool offered = row.aspect == nullptr ||
                                 Device::profile_for(tensor.device()).*row.aspect;
            const std::string refusal = refusal_text_of(
                [&] { Dispatch::require_device_dtype(tensor, "op", "x"); });

            EXPECT_EQ(refusal.empty(), offered) << refusal;
        }

        INSTANTIATE_TEST_SUITE_P(
            , Medium_OperandCheckDeviceDtype, ::testing::ValuesIn(device_dtype_rows()),
            [](const ::testing::TestParamInfo<DeviceDtypeRow>& info)
            { return std::string(info.param.name); });

        /** @brief Groups the tests that read the device's own aspects. */
        class Medium_OperandCheckProfile : public SushiBLASTest
        {

        };

        TEST_F(Medium_OperandCheckProfile, TheProfileReportsTheFp16AspectOfItsDevice)
        {
            const sycl::queue& q = TestEnvironment::queue();
            EXPECT_EQ(Device::profile_for(q).supports_fp16,
                      q.get_device().has(sycl::aspect::fp16));
        }
    } // namespace Test
} // namespace SushiBLAS
