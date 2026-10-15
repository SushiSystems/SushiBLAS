/**************************************************************************/
/* test_gemm.cpp                                                          */
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
 * @file test_gemm.cpp
 * @brief Tests the unpacked SYCL gemm against the reference gemm on the host device.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <vector>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <SushiBLAS/core/blas_types.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/blas_seam_kernels/sycl/level3/gemm.hpp>
#include <SushiBLAS/blas_test_oracle/blas_flags.hpp>
#include <SushiBLAS/host_device/host_device_test.hpp>
#include <SushiBLAS/packed_product_harness/packed_product_harness.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /** @brief Runs the unpacked gemm on the host CPU queue. */
        class Device_SYCLGemm : public HostDeviceTest
        {

        };

        TEST_F(Device_SYCLGemm, MatchesTheReferenceInEveryLayoutTransposeAndBatch)
        {
            // The patterned operands are small integers, so both kernels compute exactly.
            for (const Core::Layout layout : {Core::Layout::RowMajor, Core::Layout::ColumnMajor})
            {
                for (const bool trans_a : {false, true})
                {
                    for (const bool trans_b : {false, true})
                    {
                        SCOPED_TRACE(2 * trans_a + trans_b);
                        GemmProblem<double> problem(queue(), layout, transpose_of(trans_a),
                                                    transpose_of(trans_b), 37, 29, 41, 2);

                        EXPECT_EQ(problem.unpacked(2.0, -1.0), problem.expected(2.0, -1.0));
                    }
                }
            }
        }

        TEST_F(Device_SYCLGemm, FloatAndAZeroBetaMatchTheReference)
        {
            GemmProblem<float> problem(queue(), Core::Layout::RowMajor, Core::Transpose::NoTrans,
                                       Core::Transpose::NoTrans, 64, 48, 33, 1);
            problem.poison_output();
            const std::vector<float> got = problem.unpacked(1.0f, 0.0f);
            const std::vector<float> want = problem.expected(1.0f, 0.0f);

            // The padding keeps its NaN poison, and NaN never compares equal, so compare bits.
            ASSERT_EQ(got.size(), want.size());
            EXPECT_EQ(count_bit_mismatches(got.data(), want.data(), got.size()), std::size_t{0});
        }
    } // namespace Test
} // namespace SushiBLAS
