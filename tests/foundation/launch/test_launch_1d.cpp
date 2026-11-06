/**************************************************************************/
/* test_launch_1d.cpp                                                     */
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
 * @file test_launch_1d.cpp
 * @brief Tests the shape-taking launch primitives.
 * @author Mustafa Garip
 */

#include <array>
#include <complex>
#include <limits>
#include <vector>
#include <cstddef>
#include <cstring>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <SushiBLAS/launch/launch_1d.hpp>
#include <SushiBLAS/launch/device_policy.hpp>
#include <SushiBLAS/host_device/shared_array.hpp>
#include <SushiBLAS/test_runner/test_environment.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            /** @brief Holds the element count at which every work-item of a GPU-like grid loops over its stride. */
            inline constexpr std::size_t K_LARGE_COUNT = 100000;

            /** @brief Holds the modulus that keeps every generated value an integer a float holds exactly. */
            inline constexpr std::size_t K_VALUE_PERIOD = 1024;

            /** @brief Returns the shape a non-CPU device with one compute unit plans for n elements of width. */
            Launch::Shape1D gpu_like_shape(const sycl::queue& q, std::size_t n, std::size_t width)
            {
                Device::Profile profile = Device::profile_for(q);
                profile.is_cpu = false;
                profile.compute_units = 1;
                return Launch::plan_for(profile, n, width);
            }

            /** @brief Returns the i-th generated input value, an integer below K_VALUE_PERIOD. */
            template<typename T>
            [[nodiscard]] T value_at(std::size_t i) noexcept
            {
                return static_cast<T>(i % K_VALUE_PERIOD);
            }

            /** @brief Sets element i of array to generate(i) for every i below n. */
            template<typename T, typename Generator>
            void fill_from(SharedArray<T>& array, std::size_t n, Generator generate)
            {
                for (std::size_t i = 0; i < n; ++i)
                    array.get()[i] = generate(i);
            }

            /** @brief Returns the n values generate(0) to generate(n - 1). */
            template<typename T, typename Generator>
            [[nodiscard]] std::vector<T> vector_from(std::size_t n, Generator generate)
            {
                std::vector<T> values;
                values.reserve(n);
                for (std::size_t i = 0; i < n; ++i)
                    values.push_back(generate(i));
                return values;
            }

            /** @brief Returns how many elements of got differ in their bits from the same element of want. */
            template<typename T>
            [[nodiscard]] std::size_t count_mismatches(const T* got, const std::vector<T>& want)
            {
                std::size_t mismatches = 0;
                for (std::size_t i = 0; i < want.size(); ++i)
                    if (std::memcmp(got + i, &want[i], sizeof(T)) != 0)
                        ++mismatches;
                return mismatches;
            }
        } // namespace

        /** @brief Gives the launch-shape tests the queue they submit to. */
        class Medium_LaunchShapes : public ::testing::Test
        {
            protected:
                /** @brief Returns the queue of the test environment. */
                [[nodiscard]] static sycl::queue& queue()
                {
                    return TestEnvironment::queue();
                }
        };

        /** @brief Gives a launch-shape test one element count to run at. */
        class LaunchAtCount : public ::testing::TestWithParam<std::size_t>
        {
            protected:
                /** @brief Returns the queue of the test environment. */
                [[nodiscard]] static sycl::queue& queue()
                {
                    return TestEnvironment::queue();
                }
        };

        /// @brief Runs the wide pass of a map at one element count.
        class Medium_LaunchWidePass : public LaunchAtCount
        {
        };

        /** @brief Tests that a wide pass and its scalar tail together write every element of a map. */
        TEST_P(Medium_LaunchWidePass, TheWidePassAndItsTailCoverEveryElement)
        {
            sycl::queue& q = queue();
            const std::size_t n = GetParam();
            SharedArray<float> in(q, n);
            SharedArray<float> out(q, n);
            fill_from(in, n, [](std::size_t i) { return value_at<float>(i); });
            fill_from(out, n, [](std::size_t) { return -1.0f; });
            const std::vector<float> want =
                vector_from<float>(n, [](std::size_t i) { return value_at<float>(i) * 2.0f + 1.0f; });

            const Launch::Shape1D shape = gpu_like_shape(q, n, 4);
            ASSERT_EQ(shape.vector_steps, n / 4);
            ASSERT_FALSE(shape.flat);

            const float* in_ptr = in.get();
            float* out_ptr = out.get();
            q.submit([&](sycl::handler& h)
            {
                Launch::map(h, shape, n, [](float x) { return x * 2.0f + 1.0f; },
                            out_ptr, in_ptr);
            }).wait();

            EXPECT_EQ(count_mismatches(out.get(), want), std::size_t{0});
        }

        INSTANTIATE_TEST_SUITE_P(, Medium_LaunchWidePass,
                                 ::testing::Values(1, 3, 4, 5, 127, 1000, 1023, K_LARGE_COUNT));

        /** @brief Tests that a grid-stride map without a wide pass updates every element exactly once. */
        TEST_F(Medium_LaunchShapes, TheGridStrideLoopVisitsEveryElementOnce)
        {
            sycl::queue& q = queue();
            const std::size_t n = K_LARGE_COUNT;
            SharedArray<float> data(q, n);
            fill_from(data, n, [](std::size_t i) { return value_at<float>(i); });
            const std::vector<float> want =
                vector_from<float>(n, [](std::size_t i) { return value_at<float>(i) + 1.0f; });

            const Launch::Shape1D shape = gpu_like_shape(q, n, 1);
            ASSERT_EQ(shape.vector_steps, std::size_t{0});
            ASSERT_FALSE(shape.flat);
            ASSERT_LT(shape.grid.stride(), n);

            float* ptr = data.get();
            const float* in_ptr = ptr;
            q.submit([&](sycl::handler& h)
            {
                // The map runs in place, so an element visited twice would be incremented twice.
                Launch::map(h, shape, n, [](float x) { return x + 1.0f; }, ptr, in_ptr);
            }).wait();

            EXPECT_EQ(count_mismatches(data.get(), want), std::size_t{0});
        }

        /// @brief Runs a fused in-place update at one element count.
        class Medium_LaunchFusedUpdate : public LaunchAtCount
        {
        };

        /** @brief Tests that a fused in-place update gives the host result. */
        TEST_P(Medium_LaunchFusedUpdate, AFusedUpdateTakesTheWidePass)
        {
            sycl::queue& q = queue();
            const std::size_t n = GetParam();
            SharedArray<float> first(q, n);
            SharedArray<float> second(q, n);
            fill_from(first, n, [](std::size_t i) { return value_at<float>(i); });
            fill_from(second, n, [](std::size_t i) { return value_at<float>(i + 7); });
            const std::vector<float> want_first = vector_from<float>(
                n, [](std::size_t i) { return value_at<float>(i + 7) * 2.0f; });
            const std::vector<float> want_second = vector_from<float>(
                n, [](std::size_t i) { return value_at<float>(i) * 3.0f; });

            const std::array<float*, 2> ptrs{first.get(), second.get()};
            q.submit([&](sycl::handler& h)
            {
                Launch::zip_inplace(h, gpu_like_shape(q, n, 4), n, ptrs, [](float& a, float& b)
                {
                    const float held = a;
                    a = b * 2.0f;
                    b = held * 3.0f;
                });
            }).wait();

            EXPECT_EQ(count_mismatches(first.get(), want_first), std::size_t{0});
            EXPECT_EQ(count_mismatches(second.get(), want_second), std::size_t{0});
        }

        INSTANTIATE_TEST_SUITE_P(, Medium_LaunchFusedUpdate,
                                 ::testing::Values(5, 1023, K_LARGE_COUNT));

        /// @brief Runs an indexed call at one element count.
        class Medium_LaunchIndexed : public LaunchAtCount
        {
        };

        /** @brief Tests that indexed calls its operation for every index of a grid-stride shape. */
        TEST_P(Medium_LaunchIndexed, IndexedCoversEveryIndexInAGridStrideShape)
        {
            sycl::queue& q = queue();
            const std::size_t n = GetParam();
            SharedArray<std::size_t> out(q, n);
            fill_from(out, n, [](std::size_t) { return std::numeric_limits<std::size_t>::max(); });
            const std::vector<std::size_t> want =
                vector_from<std::size_t>(n, [](std::size_t i) { return i; });

            const Launch::Shape1D shape = gpu_like_shape(q, n, 1);
            ASSERT_FALSE(shape.flat);

            std::size_t* out_ptr = out.get();
            q.submit([&](sycl::handler& h)
            {
                Launch::indexed(h, shape, n, [out_ptr](std::size_t i) { out_ptr[i] = i; });
            }).wait();

            EXPECT_EQ(count_mismatches(out.get(), want), std::size_t{0});
        }

        INSTANTIATE_TEST_SUITE_P(, Medium_LaunchIndexed,
                                 ::testing::Values(1, 1023, K_LARGE_COUNT));

        /** @brief Tests that a type of vector width one writes every element of a shape that asks for a wide pass. */
        TEST_F(Medium_LaunchShapes, AWidthOneTypeIgnoresAWidePass)
        {
            using Element = std::complex<float>;
            sycl::queue& q = queue();

            ASSERT_EQ(Launch::K_VECTOR_WIDTH<Element>, 1);

            const std::size_t n = K_LARGE_COUNT;
            SharedArray<Element> in(q, n);
            SharedArray<Element> out(q, n);
            fill_from(in, n, [](std::size_t i) { return Element(value_at<float>(i), 2.0f); });
            fill_from(out, n, [](std::size_t) { return Element(-1.0f, -1.0f); });
            const std::vector<Element> want = vector_from<Element>(
                n, [](std::size_t i) { return Element(value_at<float>(i) + 1.0f, 2.0f); });

            // The promise is about a requested wide pass, so the request is forced here.
            Launch::Shape1D shape = gpu_like_shape(q, n, 1);
            shape.vector_steps = n / 4;

            const Element* in_ptr = in.get();
            Element* out_ptr = out.get();
            q.submit([&](sycl::handler& h)
            {
                Launch::map(h, shape, n,
                            [](Element x) { return Element(x.real() + 1.0f, x.imag()); },
                            out_ptr, in_ptr);
            }).wait();

            EXPECT_EQ(count_mismatches(out.get(), want), std::size_t{0});
        }

        /// @brief Runs the queue form of map at one element count.
        class Medium_LaunchQueueForm : public LaunchAtCount
        {
        };

        /** @brief Tests that the queue form of map writes the same bits as the shape form given the planned shape. */
        TEST_P(Medium_LaunchQueueForm, TheQueueFormMatchesTheShapeItPlans)
        {
            sycl::queue& q = queue();
            const std::size_t n = GetParam();
            SharedArray<float> in(q, n);
            SharedArray<float> by_queue(q, n);
            SharedArray<float> by_shape(q, n);
            fill_from(in, n, [](std::size_t i) { return value_at<float>(i); });
            fill_from(by_queue, n, [](std::size_t) { return -1.0f; });
            fill_from(by_shape, n, [](std::size_t) { return -2.0f; });

            const float* in_ptr = in.get();
            float* queue_ptr = by_queue.get();
            float* shape_ptr = by_shape.get();
            const bool aligned = Launch::vector_aligned(in_ptr, queue_ptr, shape_ptr);
            ASSERT_TRUE(aligned);

            const auto op = [](float x) { return x * 2.0f + 1.0f; };
            q.submit([&](sycl::handler& h)
            {
                Launch::map(h, q, n, aligned, op, queue_ptr, in_ptr);
            }).wait();
            q.submit([&](sycl::handler& h)
            {
                Launch::map(h, Launch::plan<float>(q, n, aligned), n, op, shape_ptr, in_ptr);
            }).wait();

            EXPECT_EQ(0, std::memcmp(queue_ptr, shape_ptr, n * sizeof(float)));
        }

        INSTANTIATE_TEST_SUITE_P(, Medium_LaunchQueueForm,
                                 ::testing::Values(5, 1023, K_LARGE_COUNT));
    } // namespace Test
} // namespace SushiBLAS
