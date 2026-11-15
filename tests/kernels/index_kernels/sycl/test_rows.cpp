/**************************************************************************/
/* test_rows.cpp                                                          */
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
 * @file test_rows.cpp
 * @brief Tests the SYCL take_rows and add_rows against the reference ones.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <vector>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <SushiBLAS/arithmetic/accumulator.hpp>
#include <SushiBLAS/index_kernels/reference/rows.hpp>
#include <SushiBLAS/index_kernels/sycl/rows.hpp>
#include <SushiBLAS/host_device/host_device_test.hpp>
#include <SushiBLAS/host_device/shared_array.hpp>
#include <SushiBLAS/small_integers/small_integers.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /** @brief Runs the kernel on the host CPU queue. */
        class Medium_SYCLRows : public HostDeviceTest
        {
            protected:
                /** @brief Expects the SYCL add_rows to equal the reference one for a beta. */
                void expect_add_rows_matches_reference(double beta);
        };

        namespace
        {
            /** @brief Holds the extents of a 5 x 3 table and 7 rows, with repeated and absent indices. */
            struct RowsCase
            {
                Kernels::MatrixSpan table{5, 3, 3, 1};
                Kernels::MatrixSpan rows{7, 3, 3, 1};
                std::vector<int64_t> index{4, 0, 4, -1, 2, 5, 0};
            };
        } // namespace

        void Medium_SYCLRows::expect_add_rows_matches_reference(double beta)
        {
            // Every value is a small integer and beta is 0 or 0.5, so both kernels add exactly.
            const RowsCase shape;
            const std::vector<double> rows = small_integers<double>(21, 2);
            const std::vector<double> start = small_integers<double>(15, 3);
            const SharedArray<double> device_rows(queue(), rows);
            const SharedArray<int64_t> device_index(queue(), shape.index);
            SharedArray<double> device_table(queue(), start);
            std::vector<double> table = start;

            Kernels::SYCL::add_rows<double, int64_t>(
                queue(), shape.rows, device_rows.get(), device_index.get(), 1, shape.table,
                device_table.get(), beta, static_cast<Kernels::Accumulator<double>*>(nullptr), {}).wait();
            Kernels::Reference::add_rows<double, int64_t>(shape.rows, rows.data(),
                                                          shape.index.data(), 1, shape.table,
                                                          table.data(), beta);

            EXPECT_EQ(device_table.values(), table);
        }

        TEST_F(Medium_SYCLRows, TakeRowsMatchesTheReference)
        {
            const RowsCase shape;
            const std::vector<double> table = small_integers<double>(15, 1);
            const SharedArray<double> device_table(queue(), table);
            const SharedArray<int64_t> device_index(queue(), shape.index);
            SharedArray<double> device_rows(queue(), static_cast<std::size_t>(21));
            std::vector<double> rows(21);

            Kernels::SYCL::take_rows<double, int64_t>(queue(), shape.table, device_table.get(),
                                                      device_index.get(), 1, shape.rows,
                                                      device_rows.get(), {}).wait();
            Kernels::Reference::take_rows<double, int64_t>(shape.table, table.data(),
                                                           shape.index.data(), 1, shape.rows,
                                                           rows.data());

            EXPECT_EQ(device_rows.values(), rows);
        }

        TEST_F(Medium_SYCLRows, AddRowsWithBetaZeroMatchesTheReference)
        {
            expect_add_rows_matches_reference(0.0);
        }

        TEST_F(Medium_SYCLRows, AddRowsWithBetaHalfMatchesTheReference)
        {
            expect_add_rows_matches_reference(0.5);
        }
    } // namespace Test
} // namespace SushiBLAS
