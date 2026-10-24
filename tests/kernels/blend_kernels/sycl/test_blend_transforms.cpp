/**************************************************************************/
/* test_blend_transforms.cpp                                              */
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

/**
 * @file test_blend_transforms.cpp
 * @brief Tests the SYCL blend of transforms against the reference one.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <vector>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <SushiBLAS/blend_kernels/reference/blend_transforms.hpp>
#include <SushiBLAS/blend_kernels/sycl/blend_transforms.hpp>
#include <SushiBLAS/host_device/host_device_test.hpp>
#include <SushiBLAS/host_device/shared_array.hpp>
#include <SushiBLAS/small_integers/small_integers.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            /**
             * @brief Returns indices that run from -1 to 4, so each blend meets both ends of an
             *        out-of-range one.
             */
            template<typename Index>
            std::vector<Index> indices_around(std::size_t count)
            {
                std::vector<Index> indices(count);
                for (std::size_t i = 0; i < count; ++i)
                    indices[i] = static_cast<Index>(static_cast<int64_t>(i % 6) - 1);
                return indices;
            }
        } // namespace

        /** @brief Runs the kernel on the host CPU queue. */
        class Medium_SYCLBlendTransforms : public HostDeviceTest
        {

        };

        TEST_F(Medium_SYCLBlendTransforms, MatchesTheReferenceInEveryShapeAndIndexType)
        {
            // Every value is a small integer, so both kernels compute without rounding.
            constexpr int64_t K_POINTS = 37;
            constexpr int64_t K_TRANSFORMS = 4;
            constexpr int64_t K_INFLUENCES = 4;

            for (const int64_t components : {int64_t{3}, int64_t{4}})
            {
                for (const int64_t rows : {int64_t{3}, int64_t{4}})
                {
                    const std::vector<double> points =
                        small_integers<double>(static_cast<std::size_t>(K_POINTS * components), 1);
                    const std::vector<double> matrices = small_integers<double>(
                        static_cast<std::size_t>(K_TRANSFORMS * rows * 4), 2);
                    const std::vector<double> weights = small_integers<double>(
                        static_cast<std::size_t>(K_POINTS * K_INFLUENCES), 3);
                    const std::vector<int32_t> narrow =
                        indices_around<int32_t>(static_cast<std::size_t>(K_POINTS * K_INFLUENCES));
                    const std::vector<int64_t> wide =
                        indices_around<int64_t>(static_cast<std::size_t>(K_POINTS * K_INFLUENCES));
                    const std::size_t outputs = static_cast<std::size_t>(K_POINTS * components);

                    const SharedArray<double> device_points(queue(), points);
                    const SharedArray<double> device_matrices(queue(), matrices);
                    const SharedArray<double> device_weights(queue(), weights);
                    const SharedArray<int32_t> device_narrow(queue(), narrow);
                    const SharedArray<int64_t> device_wide(queue(), wide);
                    SharedArray<double> out_narrow(queue(), outputs);
                    SharedArray<double> out_wide(queue(), outputs);
                    std::vector<double> expected(outputs);

                    Kernels::Reference::blend_transforms(K_POINTS, components, rows, K_INFLUENCES,
                                                         K_TRANSFORMS, points.data(),
                                                         matrices.data(), wide.data(),
                                                         weights.data(), expected.data());
                    Kernels::SYCL::blend_transforms(queue(), K_POINTS, components, rows,
                                                    K_INFLUENCES, K_TRANSFORMS,
                                                    device_points.get(), device_matrices.get(),
                                                    device_narrow.get(), device_weights.get(),
                                                    out_narrow.get(), {}).wait();
                    Kernels::SYCL::blend_transforms(queue(), K_POINTS, components, rows,
                                                    K_INFLUENCES, K_TRANSFORMS,
                                                    device_points.get(), device_matrices.get(),
                                                    device_wide.get(), device_weights.get(),
                                                    out_wide.get(), {}).wait();

                    EXPECT_EQ(out_narrow.values(), expected)
                        << components << " values, " << rows << " rows, INT32";
                    EXPECT_EQ(out_wide.values(), expected)
                        << components << " values, " << rows << " rows, INT64";
                }
            }
        }

        TEST_F(Medium_SYCLBlendTransforms, MatchesTheReferenceWithOneInfluenceAndOnePoint)
        {
            const std::vector<float> points{1.0f, 2.0f, 3.0f};
            const std::vector<float> matrices{1, 0, 0, 1, 0, 1, 0, 2, 0, 0, 1, 3};
            const std::vector<int32_t> indices{0};
            const std::vector<float> weights{1.0f};
            const SharedArray<float> device_points(queue(), points);
            const SharedArray<float> device_matrices(queue(), matrices);
            const SharedArray<int32_t> device_indices(queue(), indices);
            const SharedArray<float> device_weights(queue(), weights);
            SharedArray<float> device_out(queue(), 3);

            Kernels::SYCL::blend_transforms(queue(), 1, 3, 3, 1, 1, device_points.get(),
                                            device_matrices.get(), device_indices.get(),
                                            device_weights.get(), device_out.get(), {}).wait();

            EXPECT_EQ(device_out.values(), (std::vector<float>{2.0f, 4.0f, 6.0f}));
        }
    } // namespace Test
} // namespace SushiBLAS
