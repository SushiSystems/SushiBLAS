/**************************************************************************/
/* test_gemv.cpp                                                          */
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
 * @file test_gemv.cpp
 * @brief Tests the SYCL gemv against the reference gemv on the host device.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <vector>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <SushiBLAS/core/blas_types.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/blas_kernels/reference/level2/gemv.hpp>
#include <SushiBLAS/blas_kernels/sycl/level2/gemv.hpp>
#include <SushiBLAS/blas_test_oracle/blas_flags.hpp>
#include <SushiBLAS/host_device/host_device_test.hpp>
#include <SushiBLAS/host_device/shared_array.hpp>
#include <SushiBLAS/small_integers/small_integers.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /// @brief Rows of the matrix; odd, so no launch shape divides it.
        constexpr int64_t K_ROWS = 37;

        /// @brief Columns of the matrix.
        constexpr int64_t K_COLUMNS = 53;

        /** @brief Runs gemv on the host CPU queue. */
        class Medium_SYCLGemv : public HostDeviceTest
        {

        };

        TEST_F(Medium_SYCLGemv, MatchesTheReferenceInEveryLayoutAndTranspose)
        {
            // Every operand is a small integer, so both kernels compute without rounding.
            const std::vector<double> a = small_integers<double>(K_ROWS * K_COLUMNS, 1);
            const SharedArray<double> device_a(queue(), a);
            for (const Core::Layout layout : {Core::Layout::RowMajor, Core::Layout::ColumnMajor})
            {
                for (const bool transposed : {false, true})
                {
                    SCOPED_TRACE(transposed);
                    const int64_t in_length = transposed ? K_ROWS : K_COLUMNS;
                    const int64_t out_length = transposed ? K_COLUMNS : K_ROWS;
                    const int64_t lda = (layout == Core::Layout::RowMajor) ? K_COLUMNS : K_ROWS;
                    const std::vector<double> x = small_integers<double>(2 * in_length, 2);
                    std::vector<double> y = small_integers<double>(out_length, 3);
                    const SharedArray<double> device_x(queue(), x);
                    SharedArray<double> device_y(queue(), y);

                    Kernels::SYCL::gemv(queue(), layout, transpose_of(transposed), K_ROWS,
                                        K_COLUMNS, 3.0, device_a.get(), lda, device_x.get(), 2,
                                        -2.0, device_y.get(), 1, {}).wait();
                    Kernels::Reference::gemv(layout, transpose_of(transposed), K_ROWS, K_COLUMNS,
                                             3.0, a.data(), lda, x.data(), 2, -2.0, y.data(), 1);

                    EXPECT_EQ(device_y.values(), y);
                }
            }
        }

        TEST_F(Medium_SYCLGemv, FloatMatchesTheReferenceToo)
        {
            const std::vector<float> a = small_integers<float>(K_ROWS * K_COLUMNS, 4);
            const std::vector<float> x = small_integers<float>(K_COLUMNS, 5);
            std::vector<float> y(K_ROWS, 0.0f);
            const SharedArray<float> device_a(queue(), a);
            const SharedArray<float> device_x(queue(), x);
            SharedArray<float> device_y(queue(), y);

            Kernels::SYCL::gemv(queue(), Core::Layout::RowMajor, Core::Transpose::NoTrans,
                                K_ROWS, K_COLUMNS, 1.0f, device_a.get(), K_COLUMNS,
                                device_x.get(), 1, 0.0f, device_y.get(), 1, {}).wait();
            Kernels::Reference::gemv(Core::Layout::RowMajor, Core::Transpose::NoTrans, K_ROWS,
                                     K_COLUMNS, 1.0f, a.data(), K_COLUMNS, x.data(), 1, 0.0f,
                                     y.data(), 1);

            EXPECT_EQ(device_y.values(), y);
        }
    } // namespace Test
} // namespace SushiBLAS
