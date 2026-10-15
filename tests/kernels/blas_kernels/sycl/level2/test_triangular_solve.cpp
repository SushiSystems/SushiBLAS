/**************************************************************************/
/* test_triangular_solve.cpp                                              */
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
 * @file test_triangular_solve.cpp
 * @brief Tests the blocked SYCL triangular solve against the reference one through a packed access.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <vector>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <SushiBLAS/core/blas_types.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/arithmetic/accumulator.hpp>
#include <SushiBLAS/blas_seam_kernels/packed_access.hpp>
#include <SushiBLAS/blas_kernels/reference/level2/triangular_solve.hpp>
#include <SushiBLAS/blas_kernels/sycl/level2/triangular_solve.hpp>
#include <SushiBLAS/blas_kernels/sycl/level3/triangular_plan.hpp>
#include <SushiBLAS/blas_test_oracle/blas_flags.hpp>
#include <SushiBLAS/blas_test_oracle/blas_structured.hpp>
#include <SushiBLAS/blas_test_oracle/dense_oracle.hpp>
#include <SushiBLAS/host_device/host_device_test.hpp>
#include <SushiBLAS/host_device/shared_array.hpp>
#include <SushiBLAS/small_integers/small_integers.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /** @brief Runs the triangular solve on the host CPU queue. */
        class Medium_SYCLTriangularSolve : public HostDeviceTest
        {

        };

        TEST_F(Medium_SYCLTriangularSolve, RecoversTheSolutionTheReferenceRecovers)
        {
            const int64_t n = 70;
            // A unit diagonal keeps every substitution exact in integers.
            const std::vector<double> a = unit_diagonal_matrix<double>(n, 1);
            const std::vector<double> solution = small_integers<double>(n, 2);
            SharedArray<Kernels::Accumulator<double>> work(queue(), static_cast<std::size_t>(n));
            for (const int64_t block : {int64_t{1}, int64_t{16}, n})
            {
                const Kernels::SYCL::TriangularSolvePlan plan{block};
                for (const bool upper : {false, true})
                {
                    for (const bool transposed : {false, true})
                    {
                        SCOPED_TRACE(100 * block + 2 * upper + transposed);
                        const std::vector<double> packed =
                            to_packed(a, n, uplo_of(upper), Core::Layout::RowMajor);
                        const SharedArray<double> device_packed(queue(), packed);
                        const Kernels::PackedAccess access =
                            Kernels::packed_access(Core::Layout::RowMajor, uplo_of(upper),
                                                   transpose_of(transposed), n);
                        std::vector<double> x =
                            dense_matvec(triangle_of(a, n, upper, false), n, n, transposed, 1.0,
                                         solution, 0.0, solution);
                        SharedArray<double> device_x(queue(), x);

                        Kernels::SYCL::Detail::triangular_solve(
                            queue(), plan, access, transposed != upper, false, n, n,
                            device_packed.get(), device_x.get(), 1, work.get(), {}).wait();
                        Kernels::Reference::Detail::triangular_solve(
                            access, transposed != upper, false, n, packed.data(), x.data(), 1);

                        EXPECT_EQ(device_x.values(), x);
                        EXPECT_EQ(x, solution);
                    }
                }
            }
        }
    } // namespace Test
} // namespace SushiBLAS
