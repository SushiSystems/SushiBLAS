/**************************************************************************/
/* test_convolve_1d.cpp                                                   */
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
 * @file test_convolve_1d.cpp
 * @brief Tests the SYCL valid-mode convolution against the reference one.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <SushiBLAS/signal_kernels/reference/convolve_1d.hpp>
#include <SushiBLAS/signal_kernels/sycl/convolve_1d.hpp>
#include <SushiBLAS/host_device/host_device_test.hpp>
#include <SushiBLAS/host_device/shared_array.hpp>
#include <SushiBLAS/small_integers/small_integers.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            /// @brief Holds the shape of one convolution the device and the reference both run.
            struct ConvolutionShape
            {
                    const char* name;
                    int64_t rows;
                    int64_t length;
                    int64_t taps;
            };

            /// @brief Returns batched rows, one tap and taps as long as the row.
            std::vector<ConvolutionShape> convolution_shapes()
            {
                return {{"BatchedRows", 6, 37, 9},
                        {"OneTap", 4, 7, 1},
                        {"TapsAsLongAsTheRow", 4, 7, 7}};
            }

            /// @brief Names a row, which gtest takes as the test's suffix.
            std::string convolution_shape_name(const ::testing::TestParamInfo<ConvolutionShape>& info)
            {
                return info.param.name;
            }
        } // namespace

        /** @brief Runs the kernel on the host CPU queue, one shape at a time. */
        class Medium_SYCLConvolve1D : public HostDeviceTest, public ::testing::WithParamInterface<ConvolutionShape>
        {

        };

        TEST_P(Medium_SYCLConvolve1D, MatchesTheReferenceOnBatchedRows)
        {
            // Every value is a small integer, so both kernels compute without rounding.
            const ConvolutionShape& shape = GetParam();
            const std::size_t outputs = static_cast<std::size_t>(shape.rows * (shape.length - shape.taps + 1));
            const std::vector<double> signal =
                small_integers<double>(static_cast<std::size_t>(shape.rows * shape.length), 1);
            const std::vector<double> taps = small_integers<double>(static_cast<std::size_t>(shape.taps), 2);
            const SharedArray<double> device_signal(queue(), signal);
            const SharedArray<double> device_taps(queue(), taps);
            SharedArray<double> device_out(queue(), outputs);
            std::vector<double> out(outputs);

            Kernels::SYCL::convolve_1d(queue(), shape.rows, shape.length, shape.taps, device_signal.get(),
                                       device_taps.get(), device_out.get(), {}).wait();
            Kernels::Reference::convolve_1d(shape.rows, shape.length, shape.taps, signal.data(), taps.data(),
                                            out.data());

            EXPECT_EQ(device_out.values(), out);
        }

        INSTANTIATE_TEST_SUITE_P(, Medium_SYCLConvolve1D, ::testing::ValuesIn(convolution_shapes()),
                                 convolution_shape_name);
    } // namespace Test
} // namespace SushiBLAS
