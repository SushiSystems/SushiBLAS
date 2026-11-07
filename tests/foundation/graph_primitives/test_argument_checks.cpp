/**************************************************************************/
/* test_argument_checks.cpp                                               */
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
/**
 * @file test_argument_checks.cpp
 * @brief Tests the value-level argument checks the recording primitives run first.
 * @author Mustafa Garip
 * @see tests/foundation/graph_primitives/README.md
 */

#include <cstddef>
#include <optional>
#include <string>

#include <gtest/gtest.h>

#include <SushiRuntime/api/data/buffer.hpp>
#include <SushiRuntime/core/determinism.hpp>
#include <SushiRuntime/core/error.hpp>

#include <SushiBLAS/graph_primitives/argument_checks.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            using GraphPrimitives::PrimitiveOptions;
            using GraphPrimitives::Detail::CountingSortShape;
            using GraphPrimitives::Detail::Operand;
            using GraphPrimitives::Detail::PrefixScanShape;
            using GraphPrimitives::Detail::ReduceShape;
            using GraphPrimitives::Detail::SegmentedReduceShape;
            using GraphPrimitives::Detail::validate_counting_sort;
            using GraphPrimitives::Detail::validate_prefix_scan;
            using GraphPrimitives::Detail::validate_reduce;
            using GraphPrimitives::Detail::validate_segmented_reduce;

            /** @brief Returns an operand of size elements on device 0 with a window at offset. */
            Operand operand(std::size_t size, std::size_t offset = 0, std::size_t device = 0)
            {
                return Operand{size, offset, SushiRuntime::DeviceIndex{device}};
            }

            /** @brief Returns a sort of eight keys over four buckets whose operands all fit. */
            CountingSortShape fitting_sort()
            {
                CountingSortShape shape;
                shape.n_capacity = 8;
                shape.key_capacity = 4;
                shape.keys = operand(8);
                shape.indices = operand(8);
                shape.offsets = operand(5);
                shape.total = operand(1);
                shape.counts = operand(4);
                return shape;
            }

            /** @brief Returns a scan of eight counts whose operands all fit. */
            PrefixScanShape fitting_scan()
            {
                PrefixScanShape shape;
                shape.capacity = 8;
                shape.counts = operand(8);
                shape.offsets = operand(9);
                shape.total = operand(1);
                return shape;
            }

            /** @brief Returns a reduction of eight values whose operands all fit. */
            ReduceShape fitting_reduce()
            {
                ReduceShape shape;
                shape.count = 8;
                shape.input = operand(8);
                shape.output = operand(1);
                return shape;
            }

            /** @brief Returns a segmented reduction of four segments whose operands all fit. */
            SegmentedReduceShape fitting_segments()
            {
                SegmentedReduceShape shape;
                shape.segment_count = 4;
                shape.values = operand(10);
                shape.values_count = 10;
                shape.segment_offsets = operand(5);
                shape.output = operand(4);
                return shape;
            }

            /** @brief Returns whether result is an InvalidGraph error whose message holds word. */
            ::testing::AssertionResult refuses(const SushiRuntime::Result<void>& result,
                                               const std::string& word)
            {
                if (result.has_value())
                    return ::testing::AssertionFailure() << "the call was accepted";
                if (result.error().code != SushiRuntime::ErrorCode::InvalidGraph)
                    return ::testing::AssertionFailure() << "the code is not InvalidGraph";
                if (result.error().message.find(word) == std::string::npos)
                    return ::testing::AssertionFailure()
                           << "'" << word << "' is missing from: " << result.error().message;
                return ::testing::AssertionSuccess();
            }

            /** @brief Returns options whose name is empty. */
            std::optional<PrimitiveOptions> unnamed_options()
            {
                return PrimitiveOptions{};
            }

            /** @brief Returns named options that carry no class of the runtime. */
            std::optional<PrimitiveOptions> unknown_class_options()
            {
                return PrimitiveOptions{"node", static_cast<SushiRuntime::Determinism>(250)};
            }
        } // namespace

        /** @brief Tests that operands which all fit are accepted, with and without counts. */
        TEST(Small_GraphPrimitivesArgumentChecks, AcceptsOperandsThatFit)
        {
            CountingSortShape without_counts = fitting_sort();
            without_counts.counts.reset();

            EXPECT_TRUE(validate_counting_sort(fitting_sort(), std::nullopt).has_value());
            EXPECT_TRUE(validate_counting_sort(without_counts, std::nullopt).has_value());
            EXPECT_TRUE(validate_prefix_scan(fitting_scan(), std::nullopt).has_value());
            EXPECT_TRUE(validate_reduce(fitting_reduce(), std::nullopt).has_value());
            EXPECT_TRUE(validate_segmented_reduce(fitting_segments(), std::nullopt).has_value());
        }

        /** @brief Tests that a key range capacity of zero is refused, having no bucket. */
        TEST(Small_GraphPrimitivesArgumentChecks, CountingSortRefusesAKeyRangeOfZero)
        {
            CountingSortShape shape = fitting_sort();
            shape.key_capacity = 0;

            EXPECT_TRUE(refuses(validate_counting_sort(shape, std::nullopt),
                                "key_range"));
        }

        /** @brief Tests that a buffer one element short of its window is refused by name. */
        TEST(Small_GraphPrimitivesArgumentChecks, CountingSortRefusesAShortOperand)
        {
            CountingSortShape keys = fitting_sort();
            keys.keys.size = 7;
            CountingSortShape indices = fitting_sort();
            indices.indices.size = 7;
            CountingSortShape offsets = fitting_sort();
            offsets.offsets.size = 4;
            CountingSortShape total = fitting_sort();
            total.total.size = 0;
            CountingSortShape counts = fitting_sort();
            counts.counts->size = 3;

            EXPECT_TRUE(refuses(validate_counting_sort(keys, std::nullopt), "keys"));
            EXPECT_TRUE(refuses(validate_counting_sort(indices, std::nullopt), "indices"));
            EXPECT_TRUE(refuses(validate_counting_sort(offsets, std::nullopt), "offsets"));
            EXPECT_TRUE(refuses(validate_counting_sort(total, std::nullopt), "total"));
            EXPECT_TRUE(refuses(validate_counting_sort(counts, std::nullopt), "counts"));
        }

        /** @brief Tests that a window of one element past the end of its buffer is refused. */
        TEST(Small_GraphPrimitivesArgumentChecks, CountingSortRefusesAWindowPastItsBuffer)
        {
            CountingSortShape shape = fitting_sort();
            shape.keys.offset = 1;
            EXPECT_TRUE(refuses(validate_counting_sort(shape, std::nullopt), "keys"));

            shape = fitting_sort();
            shape.indices.offset = 1;
            EXPECT_TRUE(refuses(validate_counting_sort(shape, std::nullopt), "indices"));

            shape = fitting_sort();
            shape.offsets.offset = 1;
            EXPECT_TRUE(refuses(validate_counting_sort(shape, std::nullopt), "offsets"));

            shape = fitting_sort();
            shape.total.offset = 1;
            EXPECT_TRUE(refuses(validate_counting_sort(shape, std::nullopt), "total"));

            shape = fitting_sort();
            shape.counts->offset = 1;
            EXPECT_TRUE(refuses(validate_counting_sort(shape, std::nullopt), "counts"));
        }

        /** @brief Tests that a sort whose operands live on two devices is refused. */
        TEST(Small_GraphPrimitivesArgumentChecks, CountingSortRefusesOperandsOnTwoDevices)
        {
            CountingSortShape indices = fitting_sort();
            indices.indices.device = SushiRuntime::DeviceIndex{1};
            CountingSortShape offsets = fitting_sort();
            offsets.offsets.device = SushiRuntime::DeviceIndex{1};
            CountingSortShape counts = fitting_sort();
            counts.counts->device = SushiRuntime::DeviceIndex{1};

            EXPECT_TRUE(refuses(validate_counting_sort(indices, std::nullopt), "same device"));
            EXPECT_TRUE(refuses(validate_counting_sort(offsets, std::nullopt), "same device"));
            EXPECT_TRUE(refuses(validate_counting_sort(counts, std::nullopt), "same device"));
        }

        /** @brief Tests that options which name nothing are refused before any operand. */
        TEST(Small_GraphPrimitivesArgumentChecks, EveryPrimitiveRefusesOptionsThatNameNothing)
        {
            CountingSortShape short_sort = fitting_sort();
            short_sort.keys.size = 0;

            for (const std::optional<PrimitiveOptions>& options :
                 {unnamed_options(), unknown_class_options()})
            {
                EXPECT_TRUE(refuses(validate_counting_sort(short_sort, options),
                                    "'options'"));
                EXPECT_TRUE(refuses(validate_prefix_scan(fitting_scan(), options),
                                    "'options'"));
                EXPECT_TRUE(refuses(validate_reduce(fitting_reduce(), options),
                                    "'options'"));
                EXPECT_TRUE(refuses(
                    validate_segmented_reduce(fitting_segments(), options),
                    "'options'"));
            }
        }

        /** @brief Tests that a scan refuses a short operand, a late window and two devices. */
        TEST(Small_GraphPrimitivesArgumentChecks, PrefixScanRefusesWhatDoesNotFit)
        {
            PrefixScanShape counts = fitting_scan();
            counts.counts.size = 7;
            PrefixScanShape sentinel = fitting_scan();
            sentinel.offsets.size = 8;
            PrefixScanShape total = fitting_scan();
            total.total.size = 0;
            PrefixScanShape window = fitting_scan();
            window.offsets.offset = 1;
            PrefixScanShape device = fitting_scan();
            device.offsets.device = SushiRuntime::DeviceIndex{1};

            EXPECT_TRUE(refuses(validate_prefix_scan(counts, std::nullopt), "counts"));
            EXPECT_TRUE(refuses(validate_prefix_scan(sentinel, std::nullopt), "offsets"));
            EXPECT_TRUE(refuses(validate_prefix_scan(total, std::nullopt), "total"));
            EXPECT_TRUE(refuses(validate_prefix_scan(window, std::nullopt), "offsets"));
            EXPECT_TRUE(refuses(validate_prefix_scan(device, std::nullopt), "same device"));
        }

        /** @brief Tests that a reduction refuses a short or late window. */
        TEST(Small_GraphPrimitivesArgumentChecks, ReduceRefusesWhatDoesNotFit)
        {
            ReduceShape input = fitting_reduce();
            input.input.size = 7;
            ReduceShape output = fitting_reduce();
            output.output.size = 0;
            ReduceShape window = fitting_reduce();
            window.input.offset = 1;
            ReduceShape out_window = fitting_reduce();
            out_window.output.offset = 1;

            EXPECT_TRUE(refuses(validate_reduce(input, std::nullopt), "input"));
            EXPECT_TRUE(refuses(validate_reduce(output, std::nullopt), "output"));
            EXPECT_TRUE(refuses(validate_reduce(window, std::nullopt), "input"));
            EXPECT_TRUE(refuses(validate_reduce(out_window, std::nullopt), "output"));
        }

        /** @brief Tests that a reduction of nothing needs only its output to fit. */
        TEST(Small_GraphPrimitivesArgumentChecks, ReduceOfZeroValuesNeedsNoInput)
        {
            ReduceShape shape = fitting_reduce();
            shape.count = 0;
            shape.input = operand(0);

            EXPECT_TRUE(validate_reduce(shape, std::nullopt).has_value());
        }

        /** @brief Tests that a segmented reduction refuses what does not fit, by operand. */
        TEST(Small_GraphPrimitivesArgumentChecks, SegmentedReduceRefusesWhatDoesNotFit)
        {
            SegmentedReduceShape values = fitting_segments();
            values.values.size = 9;
            SegmentedReduceShape offsets = fitting_segments();
            offsets.segment_offsets.size = 4;
            SegmentedReduceShape output = fitting_segments();
            output.output.size = 3;
            SegmentedReduceShape both = fitting_segments();
            both.segment_offsets.size = 2;
            both.output.size = 1;
            SegmentedReduceShape window = fitting_segments();
            window.output.offset = 1;

            EXPECT_TRUE(refuses(validate_segmented_reduce(values, std::nullopt), "values"));
            EXPECT_TRUE(
                refuses(validate_segmented_reduce(offsets, std::nullopt), "segment_offsets"));
            EXPECT_TRUE(refuses(validate_segmented_reduce(output, std::nullopt), "output"));
            EXPECT_TRUE(
                refuses(validate_segmented_reduce(both, std::nullopt), "segment_offsets"));
            EXPECT_TRUE(refuses(validate_segmented_reduce(window, std::nullopt), "output"));
        }

    } // namespace Test
} // namespace SushiBLAS
