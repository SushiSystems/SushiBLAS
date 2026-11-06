/**************************************************************************/
/* test_orthonormalize.cpp                                                */
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
 * @file test_orthonormalize.cpp
 * @brief Tests the work-group SYCL sweep against the reference sweep on the host device.
 * @author Mustafa Garip
 */

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <SushiBLAS/orthonormal/orthonormal_sweep.hpp>
#include <SushiBLAS/random_kernels/reference/orthonormalize.hpp>
#include <SushiBLAS/random_kernels/sycl/orthonormalize.hpp>
#include <SushiBLAS/host_device/host_device_test.hpp>
#include <SushiBLAS/host_device/shared_array.hpp>
#include <SushiBLAS/small_integers/small_integers.hpp>
#include "support/tolerance.hpp"

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            /// @brief The number of vectors the sweep runs on.
            constexpr int64_t K_VECTORS = 6;

            /// @brief The length of each vector.
            constexpr int64_t K_LENGTH = 50;

            /// @brief The norm the sweep scales every vector to.
            constexpr double K_GAIN = 3.0;

            /// @brief Returns small integers with the diagonal raised, so the set is not rank deficient.
            std::vector<double> diagonally_boosted_vectors()
            {
                std::vector<double> p = small_integers<double>(K_VECTORS * K_LENGTH, 1);
                for (int64_t v = 0; v < K_VECTORS; ++v)
                    p[static_cast<std::size_t>(v * K_LENGTH + v)] += 9.0;
                return p;
            }

            /**
             * @brief Expects every element of the device result to match the reference within a summation bound.
             *
             * The tree sum and the serial sum add in different orders. Each element depends on every
             * dot product before it, so the terms are the scalars of the set, and no element exceeds
             * the gain, so the products are bounded by its square.
             */
            void expect_within_summation_order(const double* device, const std::vector<double>& reference)
            {
                const std::size_t terms = static_cast<std::size_t>(K_VECTORS * K_LENGTH);
                for (std::size_t i = 0; i < reference.size(); ++i)
                    EXPECT_TRUE(TestSupport::is_within(TestSupport::measure_sum_error<double>(
                        device[i], reference[i], K_GAIN * K_GAIN, terms)))
                        << "element " << i;
            }
        } // namespace

        /** @brief Runs the sweep on the host CPU queue. */
        class Medium_SYCLOrthonormalizeOnDevice : public HostDeviceTest
        {

        };

        TEST_F(Medium_SYCLOrthonormalizeOnDevice, MatchesTheReferenceWithinItsSummationOrder)
        {
            const Kernels::VectorSet set{K_VECTORS, K_LENGTH, 1, K_LENGTH};
            std::vector<double> reference = diagonally_boosted_vectors();
            SharedArray<double> device(queue(), reference);

            Kernels::SYCL::Random::orthonormalize(queue(), {}, device.get(), set, K_GAIN).wait();
            Kernels::Reference::Random::orthonormalize(reference.data(), set, K_GAIN);

            expect_within_summation_order(device.get(), reference);
        }

        TEST_F(Medium_SYCLOrthonormalizeOnDevice, TheGroupIsTheLargestPowerOfTwoTheDeviceAccepts)
        {
            const std::size_t device_limit =
                queue().get_device().get_info<sycl::info::device::max_work_group_size>();
            const std::size_t limit = std::min(Kernels::SYCL::Random::Detail::K_ORTHO_MAX_GROUP, device_limit);

            const std::size_t group = Kernels::SYCL::Random::Detail::ortho_group_size(queue());

            EXPECT_EQ(group & (group - 1), std::size_t{0});
            EXPECT_LE(group, limit);
            EXPECT_GT(2 * group, limit);
        }
    } // namespace Test
} // namespace SushiBLAS
