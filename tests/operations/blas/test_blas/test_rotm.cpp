/**************************************************************************/
/* test_rotm.cpp                                                          */
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
 * @file test_rotm.cpp
 * @brief Tests rotm on each flag, on strided and empty vectors, across kernels and after rotmg.
 * @author Mustafa Garip
 */

#include <array>
#include <cmath>
#include <limits>
#include <random>
#include <string>
#include <vector>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <SushiBLAS/SushiBLAS.hpp>
#include <SushiBLAS/engine_fixture/sushiblas_test.hpp>
#include <SushiBLAS/tensor_expectations/tensor_comparison.hpp>
#include <SushiBLAS/tensor_expectations/tensor_readback.hpp>
#include <SushiBLAS/tensor_expectations/scalar_dtype.hpp>
#include <SushiBLAS/typed_suite_naming/default_type_names.hpp>
#include <SushiBLAS/test_runner/test_environment.hpp>
namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            /// @brief Seed of every random draw in this file.
            constexpr std::uint32_t K_SEED = 20261007u;

            /// @brief Elements of a rotm parameter tensor: the flag and the four entries of H.
            constexpr int64_t K_PARAMETERS = 5;

            /// @brief Fills a parameter slot the flag implies, which a kernel must not read.
            constexpr double K_UNREAD = 99.0;

            /// @brief Fills the elements a strided view skips in x.
            constexpr double K_SKIPPED_IN_X = 77.0;

            /// @brief Fills the elements a strided view skips in y.
            constexpr double K_SKIPPED_IN_Y = 88.0;

            /// @brief Pairs the two kernels are compared on.
            constexpr int64_t K_LARGE_LENGTH = 100000;

            /// @brief A full H, flag -1, whose entries HALF holds exactly.
            constexpr double K_FULL_PARAM[K_PARAMETERS] = {-1.0, 0.5, -0.25, 0.75, 1.25};

            /// @brief Both kernels an operation has.
            constexpr ExecutionMode K_MODES[] = {ExecutionMode::ForceReference,
                                                 ExecutionMode::ForceSYCL};

            /// @brief Both element orders an engine can create tensors in.
            constexpr Core::Layout K_LAYOUTS[] = {Core::Layout::RowMajor,
                                                  Core::Layout::ColumnMajor};

            /** @brief Holds one parameter array and what it makes of x = [1, 2], y = [3, 4]. */
            struct RotmRow
            {
                double param[K_PARAMETERS]; /**< The flag, then h11, h21, h12 and h22. */
                double x[2];                /**< x after the call. */
                double y[2];                /**< y after the call. */
            };

            /// @brief The table of the design, section 10.3.
            constexpr RotmRow K_ROWS[] = {
                {{-1.0, 2.0, 4.0, 3.0, 5.0}, {11.0, 16.0}, {19.0, 28.0}},
                {{0.0, K_UNREAD, 4.0, 3.0, K_UNREAD}, {10.0, 14.0}, {7.0, 12.0}},
                {{1.0, 2.0, K_UNREAD, K_UNREAD, 5.0}, {5.0, 8.0}, {14.0, 18.0}},
                {{-2.0, K_UNREAD, K_UNREAD, K_UNREAD, K_UNREAD}, {1.0, 2.0}, {3.0, 4.0}}};

            /** @brief Returns the name of a mode and a layout for a failure message. */
            [[nodiscard]] std::string engine_name(Core::Layout layout, ExecutionMode mode)
            {
                return std::string(mode == ExecutionMode::ForceSYCL ? "ForceSYCL"
                                                                     : "ForceReference") +
                       (layout == Core::Layout::RowMajor ? ", ROW_MAJOR" : ", COLUMN_MAJOR");
            }

            /** @brief Returns whether the test device computes in T. */
            template<typename T>
            [[nodiscard]] bool device_runs()
            {
                const sycl::device device =
                    TestEnvironment::queue().get_device();
                if constexpr (std::is_same_v<T, sycl::half>)
                    return device.has(sycl::aspect::fp16);
                else if constexpr (std::is_same_v<T, double>)
                    return device.has(sycl::aspect::fp64);
                else
                    return true;
            }

            /** @brief Returns value in the real element type T. */
            template<typename T>
            [[nodiscard]] T make(double value)
            {
                if constexpr (std::is_same_v<T, sycl::half>)
                    return sycl::half(static_cast<float>(value));
                else
                    return static_cast<T>(value);
            }

            /** @brief Returns values in the real element type T. */
            template<typename T, std::size_t N>
            [[nodiscard]] std::vector<T> make_all(const double (&values)[N])
            {
                std::vector<T> out;
                out.reserve(N);
                for (const double value : values)
                    out.push_back(make<T>(value));
                return out;
            }

            /** @brief Expects every element to equal its expectation with no tolerance. */
            template<typename T>
            void expect_equal(const std::vector<T>& actual, const std::vector<T>& expected)
            {
                ASSERT_EQ(actual.size(), expected.size());
                for (std::size_t i = 0; i < actual.size(); ++i)
                    EXPECT_EQ(Detail::to_double(actual[i]), Detail::to_double(expected[i]))
                        << "index " << i;
            }
        } // namespace

        /**
         * @brief Builds the vectors and the parameter tensor of a rotm call in element type T.
         */
        template<typename T>
        class RotmCases : public SushiBLASTest
        {
            protected:
                /** @brief Skips the case on a device that does not compute in T. */
                void SetUp() override
                {
                    SushiBLASTest::SetUp();
                    if (!device_runs<T>())
                        GTEST_SKIP() << "the device does not compute in this element type";
                }

                /** @brief Rebuilds the engine in a layout and selects a kernel. */
                void start(Core::Layout layout, ExecutionMode mode)
                {
                    this->reinit_engine(layout);
                    this->engine->set_mode(mode);
                }

                /** @brief Returns a rank-1 tensor that holds values. */
                [[nodiscard]] Tensor vector_of(const std::vector<T>& values)
                {
                    Tensor tensor = this->engine->create_tensor(
                        {static_cast<int64_t>(values.size())}, sushiblas_dtype<T>());
                    if (!values.empty())
                        TestEnvironment::queue()
                            .memcpy(tensor.data(), values.data(), values.size() * sizeof(T))
                            .wait();
                    return tensor;
                }

                /** @brief Returns a view of every second element of a rank-1 parent. */
                [[nodiscard]] static Tensor every_second_element(const Tensor& parent)
                {
                    const int64_t dims[] = {parent.num_elements / 2};
                    std::array<int64_t, Core::K_MAX_TENSOR_RANK> strides{};
                    strides[0] = 2;
                    return Tensor(parent.storage, SushiRuntime::span<const int64_t>(dims, 1),
                                  strides, parent.dtype, 0, parent.layout);
                }

                /** @brief Returns count values drawn from [-1, 1]. */
                [[nodiscard]] static std::vector<T> random_values(std::mt19937& generator,
                                                                  int64_t count)
                {
                    std::uniform_real_distribution<double> distribution(-1.0, 1.0);
                    std::vector<T> values(static_cast<std::size_t>(count));
                    for (T& value : values)
                        value = make<T>(distribution(generator));
                    return values;
                }
        };

        /** @brief Runs rotm over the real element types. */
        template<typename T>
        class Medium_Rotm : public RotmCases<T>
        {

        };

        /** @brief Runs rotm after rotmg over the element types rotmg accepts. */
        template<typename T>
        class Medium_RotmChained : public RotmCases<T>
        {

        };

        /// @brief Names the real element types rotm accepts.
        using RotmTypes = ::testing::Types<sycl::half, float, double>;
        /// @brief Names the element types both rotm and rotmg accept.
        using RotmChainedTypes = ::testing::Types<float, double>;
        SB_TYPED_TEST_SUITE(Medium_Rotm, RotmTypes);
        SB_TYPED_TEST_SUITE(Medium_RotmChained, RotmChainedTypes);

        /**
         * @brief Tests each flag against its hand-computed result; an implied slot holds
         *        K_UNREAD.
         */
        TYPED_TEST(Medium_Rotm, EachFlagAppliesOnlyTheEntriesItDoesNotImply)
        {
            using T = TypeParam;

            for (const Core::Layout layout : K_LAYOUTS)
            {
                for (const ExecutionMode mode : K_MODES)
                {
                    SCOPED_TRACE(engine_name(layout, mode));
                    this->start(layout, mode);

                    for (const RotmRow& row : K_ROWS)
                    {
                        SCOPED_TRACE(::testing::Message() << "flag " << row.param[0]);
                        Tensor x = this->vector_of({make<T>(1.0), make<T>(2.0)});
                        Tensor y = this->vector_of({make<T>(3.0), make<T>(4.0)});
                        const Tensor param = this->vector_of(make_all<T>(row.param));

                        SushiBLAS::BLASOps(*this->engine).rotm(x, y, param);
                        this->engine->execute().value();

                        expect_equal(read_tensor<T>(x), make_all<T>(row.x));
                        expect_equal(read_tensor<T>(y), make_all<T>(row.y));
                    }
                }
            }
        }

        /**
         * @brief Tests that flag -2 writes nothing: an infinity stays and makes no NaN.
         */
        TYPED_TEST(Medium_Rotm, TheIdentityFlagLeavesAnInfinityAlone)
        {
            using T = TypeParam;
            const double identity[K_PARAMETERS] = {-2.0, K_UNREAD, K_UNREAD, K_UNREAD, K_UNREAD};

            for (const Core::Layout layout : K_LAYOUTS)
            {
                for (const ExecutionMode mode : K_MODES)
                {
                    SCOPED_TRACE(engine_name(layout, mode));
                    this->start(layout, mode);

                    Tensor x = this->vector_of(
                        {make<T>(std::numeric_limits<double>::infinity()), make<T>(2.0)});
                    Tensor y = this->vector_of({make<T>(3.0), make<T>(4.0)});
                    const Tensor param = this->vector_of(make_all<T>(identity));

                    SushiBLAS::BLASOps(*this->engine).rotm(x, y, param);
                    this->engine->execute().value();

                    const std::vector<T> x_after = read_tensor<T>(x);
                    const std::vector<T> y_after = read_tensor<T>(y);
                    EXPECT_TRUE(std::isinf(Detail::to_double(x_after[0])));
                    EXPECT_GT(Detail::to_double(x_after[0]), 0.0);
                    EXPECT_EQ(Detail::to_double(x_after[1]), 2.0);
                    EXPECT_EQ(Detail::to_double(y_after[0]), 3.0);
                    EXPECT_EQ(Detail::to_double(y_after[1]), 4.0);
                }
            }
        }

        /**
         * @brief Tests that a view of every second element is transformed and the elements
         *        between are left as they were.
         */
        TYPED_TEST(Medium_Rotm, StridedVectorsLeaveTheSkippedElementsAlone)
        {
            using T = TypeParam;
            const double x_values[] = {1.0, K_SKIPPED_IN_X, 2.0, K_SKIPPED_IN_X};
            const double y_values[] = {3.0, K_SKIPPED_IN_Y, 4.0, K_SKIPPED_IN_Y};
            const double x_expected[] = {11.0, K_SKIPPED_IN_X, 16.0, K_SKIPPED_IN_X};
            const double y_expected[] = {19.0, K_SKIPPED_IN_Y, 28.0, K_SKIPPED_IN_Y};

            for (const Core::Layout layout : K_LAYOUTS)
            {
                for (const ExecutionMode mode : K_MODES)
                {
                    SCOPED_TRACE(engine_name(layout, mode));
                    this->start(layout, mode);

                    Tensor x_parent = this->vector_of(make_all<T>(x_values));
                    Tensor y_parent = this->vector_of(make_all<T>(y_values));
                    Tensor x = TestFixture::every_second_element(x_parent);
                    Tensor y = TestFixture::every_second_element(y_parent);
                    const Tensor param = this->vector_of(make_all<T>(K_ROWS[0].param));

                    SushiBLAS::BLASOps(*this->engine).rotm(x, y, param);
                    this->engine->execute().value();

                    expect_equal(read_tensor<T>(x_parent), make_all<T>(x_expected));
                    expect_equal(read_tensor<T>(y_parent), make_all<T>(y_expected));
                }
            }
        }

        /**
         * @brief Tests that vectors of no elements record and run.
         */
        TYPED_TEST(Medium_Rotm, AnEmptyPairRecordsAndRuns)
        {
            using T = TypeParam;

            for (const Core::Layout layout : K_LAYOUTS)
            {
                for (const ExecutionMode mode : K_MODES)
                {
                    SCOPED_TRACE(engine_name(layout, mode));
                    this->start(layout, mode);

                    Tensor x = this->vector_of({});
                    Tensor y = this->vector_of({});
                    const Tensor param = this->vector_of(make_all<T>(K_FULL_PARAM));

                    EXPECT_NO_THROW(SushiBLAS::BLASOps(*this->engine).rotm(x, y, param));
                    EXPECT_TRUE(this->engine->execute().has_value());
                    expect_equal(read_tensor<T>(param), make_all<T>(K_FULL_PARAM));
                }
            }
        }

        /**
         * @brief Tests that the two kernels give the same elements on K_LARGE_LENGTH pairs
         *        under a full H.
         */
        TYPED_TEST(Medium_Rotm, BothModesAgreeElementForElement)
        {
            using T = TypeParam;
            std::mt19937 generator(K_SEED);
            const std::vector<T> x_values = TestFixture::random_values(generator, K_LARGE_LENGTH);
            const std::vector<T> y_values = TestFixture::random_values(generator, K_LARGE_LENGTH);

            for (const Core::Layout layout : K_LAYOUTS)
            {
                SCOPED_TRACE(layout == Core::Layout::RowMajor ? "ROW_MAJOR" : "COLUMN_MAJOR");
                std::vector<T> results[2][2];

                for (std::size_t mode = 0; mode < 2; ++mode)
                {
                    this->start(layout, K_MODES[mode]);
                    Tensor x = this->vector_of(x_values);
                    Tensor y = this->vector_of(y_values);
                    const Tensor param = this->vector_of(make_all<T>(K_FULL_PARAM));

                    SushiBLAS::BLASOps(*this->engine).rotm(x, y, param);
                    this->engine->execute().value();

                    results[mode][0] = read_tensor<T>(x);
                    results[mode][1] = read_tensor<T>(y);
                }

                for (std::size_t vector = 0; vector < 2; ++vector)
                {
                    const std::vector<T>& reference = results[0][vector];
                    const std::vector<T>& device = results[1][vector];
                    ASSERT_EQ(device.size(), reference.size());

                    std::size_t differing = 0;
                    std::size_t first = 0;
                    for (std::size_t i = 0; i < device.size(); ++i)
                    {
                        if (Detail::to_double(device[i]) == Detail::to_double(reference[i]))
                            continue;
                        if (differing == 0)
                            first = i;
                        ++differing;
                    }
                    EXPECT_EQ(differing, 0u) << (vector == 0 ? "x" : "y")
                                             << " differs first at index " << first;
                }
            }
        }

        /**
         * @brief Tests that rotm reads the param an earlier rotmg of the same graph writes.
         */
        TYPED_TEST(Medium_RotmChained, AParamWrittenByRotmgIsRead)
        {
            using T = TypeParam;
            const double unwritten[K_PARAMETERS] = {K_UNREAD, K_UNREAD, K_UNREAD, K_UNREAD,
                                                    K_UNREAD};

            for (const Core::Layout layout : K_LAYOUTS)
            {
                for (const ExecutionMode mode : K_MODES)
                {
                    SCOPED_TRACE(engine_name(layout, mode));
                    this->start(layout, mode);

                    Tensor d1 = this->vector_of({make<T>(5.0)});
                    Tensor d2 = this->vector_of({make<T>(5.0)});
                    Tensor x1 = this->vector_of({make<T>(4.0)});
                    const Tensor y1 = this->vector_of({make<T>(2.0)});
                    Tensor param = this->vector_of(make_all<T>(unwritten));
                    Tensor x = this->vector_of({make<T>(4.0)});
                    Tensor y = this->vector_of({make<T>(2.0)});

                    SushiBLAS::BLASOps ops(*this->engine);
                    ops.rotmg(d1, d2, x1, y1, param);
                    ops.rotm(x, y, param);
                    this->engine->execute().value();

                    // H is [1, 0.5; -0.5, 1], so (4, 2) becomes (5, 0).
                    expect_equal(read_tensor<T>(x), std::vector<T>{make<T>(5.0)});
                    expect_equal(read_tensor<T>(y), std::vector<T>{make<T>(0.0)});
                }
            }
        }
    } // namespace Test
} // namespace SushiBLAS
