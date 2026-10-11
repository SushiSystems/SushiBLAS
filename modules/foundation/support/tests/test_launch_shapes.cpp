/**************************************************************************/
/* test_launch_shapes.cpp                                                 */
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
 * @file test_launch_shapes.cpp
 * @brief Tests the shape-taking launch primitives and the local tile fit check.
 * @author Mustafa Garip
 */

#include <new>
#include <array>
#include <complex>
#include <limits>
#include <vector>
#include <cstddef>
#include <cstring>
#include <initializer_list>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <SushiBLAS/support/launch_1d.hpp>
#include <SushiBLAS/support/device_policy.hpp>
#include "test_environment.hpp"

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            /** @brief Holds the alignment, in bytes, of every array a test launches over. */
            inline constexpr std::size_t K_TEST_ALIGNMENT = 128;

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

            /**
             * @brief Owns one 128-byte aligned shared USM array and frees it with `sycl::free`.
             * @tparam T The element type; trivially copyable.
             */
            template<typename T>
            class SharedArray final
            {
                public:
                    /** @brief Allocates count elements on the device of q. */
                    SharedArray(sycl::queue& q, std::size_t count)
                        : queue_(q),
                          data_(sycl::aligned_alloc_shared<T>(K_TEST_ALIGNMENT, count, q))
                    {
                        if (data_ == nullptr)
                            throw std::bad_alloc();
                    }

                    /** @brief Frees the array. */
                    ~SharedArray()
                    {
                        sycl::free(data_, queue_);
                    }

                    SharedArray(const SharedArray&) = delete;
                    SharedArray& operator=(const SharedArray&) = delete;
                    SharedArray(SharedArray&&) = delete;
                    SharedArray& operator=(SharedArray&&) = delete;

                    /** @brief Returns the first element. */
                    [[nodiscard]] T* get() const noexcept
                    {
                        return data_;
                    }

                private:
                    /** @brief The queue whose context owns the array. */
                    sycl::queue& queue_;

                    /** @brief The first element. */
                    T* data_;
            };

            /** @brief Returns the i-th generated input value, an integer below K_VALUE_PERIOD. */
            template<typename T>
            [[nodiscard]] T value_at(std::size_t i) noexcept
            {
                return static_cast<T>(i % K_VALUE_PERIOD);
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
        class Unit_LaunchShapes : public ::testing::Test
        {
            protected:
                /** @brief Returns the queue of the test environment. */
                [[nodiscard]] static sycl::queue& queue()
                {
                    return TestEnvironment::queue();
                }
        };

        /** @brief Tests that a wide pass and its scalar tail together write every element of a map. */
        TEST_F(Unit_LaunchShapes, TheWidePassAndItsTailCoverEveryElement)
        {
            sycl::queue& q = queue();

            for (const std::size_t n : {std::size_t{1}, std::size_t{3}, std::size_t{4}, std::size_t{5},
                                        std::size_t{127}, std::size_t{1000}, std::size_t{1023},
                                        K_LARGE_COUNT})
            {
                SCOPED_TRACE(n);
                SharedArray<float> in(q, n);
                SharedArray<float> out(q, n);
                std::vector<float> want(n);
                for (std::size_t i = 0; i < n; ++i)
                {
                    in.get()[i] = value_at<float>(i);
                    out.get()[i] = -1.0f;
                    want[i] = value_at<float>(i) * 2.0f + 1.0f;
                }

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
        }

        /** @brief Tests that a grid-stride map without a wide pass updates every element exactly once. */
        TEST_F(Unit_LaunchShapes, TheGridStrideLoopVisitsEveryElementOnce)
        {
            sycl::queue& q = queue();
            const std::size_t n = K_LARGE_COUNT;

            SharedArray<float> data(q, n);
            std::vector<float> want(n);
            for (std::size_t i = 0; i < n; ++i)
            {
                data.get()[i] = value_at<float>(i);
                want[i] = value_at<float>(i) + 1.0f;
            }

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

        /** @brief Tests that a fused in-place update gives the host result when it takes the wide pass. */
        TEST_F(Unit_LaunchShapes, AFusedUpdateTakesTheWidePass)
        {
            sycl::queue& q = queue();

            for (const std::size_t n : {std::size_t{5}, std::size_t{1023}, K_LARGE_COUNT})
            {
                SCOPED_TRACE(n);
                SharedArray<float> first(q, n);
                SharedArray<float> second(q, n);
                std::vector<float> want_first(n);
                std::vector<float> want_second(n);
                for (std::size_t i = 0; i < n; ++i)
                {
                    first.get()[i] = value_at<float>(i);
                    second.get()[i] = value_at<float>(i + 7);
                    want_first[i] = value_at<float>(i + 7) * 2.0f;
                    want_second[i] = value_at<float>(i) * 3.0f;
                }

                const Launch::Shape1D shape = gpu_like_shape(q, n, 4);
                ASSERT_EQ(shape.vector_steps, n / 4);

                const std::array<float*, 2> ptrs{first.get(), second.get()};
                q.submit([&](sycl::handler& h)
                {
                    Launch::zip_inplace(h, shape, n, ptrs, [](float& a, float& b)
                    {
                        const float held = a;
                        a = b * 2.0f;
                        b = held * 3.0f;
                    });
                }).wait();

                EXPECT_EQ(count_mismatches(first.get(), want_first), std::size_t{0});
                EXPECT_EQ(count_mismatches(second.get(), want_second), std::size_t{0});
            }
        }

        /** @brief Tests that indexed calls its operation for every index of a grid-stride shape. */
        TEST_F(Unit_LaunchShapes, IndexedCoversEveryIndexInAGridStrideShape)
        {
            sycl::queue& q = queue();

            for (const std::size_t n : {std::size_t{1}, std::size_t{1023}, K_LARGE_COUNT})
            {
                SCOPED_TRACE(n);
                SharedArray<std::size_t> out(q, n);
                std::vector<std::size_t> want(n);
                for (std::size_t i = 0; i < n; ++i)
                {
                    out.get()[i] = std::numeric_limits<std::size_t>::max();
                    want[i] = i;
                }

                const Launch::Shape1D shape = gpu_like_shape(q, n, 1);
                ASSERT_FALSE(shape.flat);

                std::size_t* out_ptr = out.get();
                q.submit([&](sycl::handler& h)
                {
                    Launch::indexed(h, shape, n, [out_ptr](std::size_t i) { out_ptr[i] = i; });
                }).wait();

                EXPECT_EQ(count_mismatches(out.get(), want), std::size_t{0});
            }
        }

        /** @brief Tests that a type of vector width one writes every element of a shape that asks for a wide pass. */
        TEST_F(Unit_LaunchShapes, AWidthOneTypeIgnoresAWidePass)
        {
            using Element = std::complex<float>;
            sycl::queue& q = queue();

            ASSERT_EQ(Launch::K_VECTOR_WIDTH<Element>, 1);

            const std::size_t n = K_LARGE_COUNT;
            SharedArray<Element> in(q, n);
            SharedArray<Element> out(q, n);
            std::vector<Element> want(n);
            for (std::size_t i = 0; i < n; ++i)
            {
                in.get()[i] = Element(value_at<float>(i), 2.0f);
                out.get()[i] = Element(-1.0f, -1.0f);
                want[i] = Element(value_at<float>(i) + 1.0f, 2.0f);
            }

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

        /** @brief Tests that the queue form of map writes the same bits as the shape form given the planned shape. */
        TEST_F(Unit_LaunchShapes, TheQueueFormMatchesTheShapeItPlans)
        {
            sycl::queue& q = queue();

            for (const std::size_t n : {std::size_t{5}, std::size_t{1023}, K_LARGE_COUNT})
            {
                SCOPED_TRACE(n);
                SharedArray<float> in(q, n);
                SharedArray<float> by_queue(q, n);
                SharedArray<float> by_shape(q, n);
                for (std::size_t i = 0; i < n; ++i)
                {
                    in.get()[i] = value_at<float>(i);
                    by_queue.get()[i] = -1.0f;
                    by_shape.get()[i] = -2.0f;
                }

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
        }

        /** @brief Tests that a local tile fits a device only when its bytes and its work-group both fit. */
        TEST_F(Unit_LaunchShapes, ALocalTileFitsOnlyWhereBothLimitsHold)
        {
            Device::Profile profile;
            profile.local_mem_bytes = 4096;
            profile.max_work_group_size = 64;

            EXPECT_TRUE(Device::hosts_local_tile(profile, Device::LocalTileDemand{4096, 64}));
            EXPECT_TRUE(Device::hosts_local_tile(profile, Device::LocalTileDemand{}));
            EXPECT_FALSE(Device::hosts_local_tile(profile, Device::LocalTileDemand{4097, 64}));
            EXPECT_FALSE(Device::hosts_local_tile(profile, Device::LocalTileDemand{4096, 65}));
        }

    } // namespace Test
} // namespace SushiBLAS
