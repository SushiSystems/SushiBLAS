/**************************************************************************/
/* test_sptrsv.cpp                                                        */
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
 * @file test_sptrsv.cpp
 * @brief Tests the SYCL sptrsv against the reference sptrsv for every flag on the host device.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <string>
#include <cstdint>
#include <vector>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <SushiBLAS/core/blas_types.hpp>
#include <SushiBLAS/arithmetic/accumulator.hpp>
#include <SushiBLAS/sparse_kernels/csr_pattern.hpp>
#include <SushiBLAS/sparse_kernels/reference/sptrsv.hpp>
#include <SushiBLAS/sparse_kernels/sycl/sptrsv.hpp>
#include <SushiBLAS/blas_test_oracle/blas_flags.hpp>
#include <SushiBLAS/blas_test_oracle/dense_oracle.hpp>
#include <SushiBLAS/host_device/host_device_test.hpp>
#include <SushiBLAS/host_device/shared_array.hpp>
#include <SushiBLAS/small_integers/small_integers.hpp>
#include <SushiBLAS/sparse_fixtures/integer_csr.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            /** @brief One flag set of the triangular solve. */
            struct FlagRow
            {
                    const char* name; /**< The row's suffix. */
                    bool upper;       /**< Whether the upper triangle is solved. */
                    bool transposed;  /**< Whether op(A) is the transpose. */
                    bool unit;        /**< Whether the diagonal is taken as one. */
            };

            /** @brief Returns the eight combinations of triangle, transposition and diagonal. */
            std::vector<FlagRow> flag_rows()
            {
                return {{"LowerPlainStored", false, false, false},
                        {"LowerPlainUnit", false, false, true},
                        {"LowerTransposedStored", false, true, false},
                        {"LowerTransposedUnit", false, true, true},
                        {"UpperPlainStored", true, false, false},
                        {"UpperPlainUnit", true, false, true},
                        {"UpperTransposedStored", true, true, false},
                        {"UpperTransposedUnit", true, true, true}};
            }

            /// @brief Names a row, which gtest takes as the test's suffix.
            std::string flag_row_name(const ::testing::TestParamInfo<FlagRow>& info)
            {
                return info.param.name;
            }
        } // namespace

        /** @brief Runs sptrsv on the host CPU queue, one flag set at a time. */
        class Medium_SYCLSptrsv : public HostDeviceTest, public ::testing::WithParamInterface<FlagRow>
        {

        };

        TEST_P(Medium_SYCLSptrsv, MatchesTheReferenceForEveryFlag)
        {
            const FlagRow& row = GetParam();
            const int64_t n = 64;
            // A stored unit diagonal keeps every substitution exact in integers.
            const IntegerCsr a = integer_csr(n, n, 7, true);
            const SharedArray<int64_t> offsets(queue(), a.offsets);
            const SharedArray<int64_t> columns(queue(), a.columns);
            const SharedArray<double> values(queue(), a.values);
            const Kernels::CsrPattern<int64_t> device_pattern{n, n, a.nnz(), 0, offsets.get(),
                                                              columns.get()};
            const Kernels::CsrPattern<int64_t> host_pattern{n, n, a.nnz(), 0, a.offsets.data(),
                                                            a.columns.data()};
            SharedArray<Kernels::Accumulator<double>> work(queue(), static_cast<std::size_t>(n));
            const std::vector<double> solution = small_integers<double>(n, 8);
            std::vector<double> b = dense_matvec(triangle_of(a.dense(), n, row.upper, row.unit), n,
                                                 n, row.transposed, 1.0, solution, 0.0, solution);
            SharedArray<double> device_b(queue(), b);

            Kernels::SYCL::sptrsv(queue(), uplo_of(row.upper), transpose_of(row.transposed),
                                  diag_of(row.unit), device_pattern, values.get(), device_b.get(),
                                  work.get(), {}).wait();
            Kernels::Reference::sptrsv(uplo_of(row.upper), transpose_of(row.transposed),
                                       diag_of(row.unit), host_pattern, a.values.data(), b.data());

            EXPECT_EQ(device_b.values(), b);
        }

        INSTANTIATE_TEST_SUITE_P(, Medium_SYCLSptrsv, ::testing::ValuesIn(flag_rows()),
                                 flag_row_name);
    } // namespace Test
} // namespace SushiBLAS
