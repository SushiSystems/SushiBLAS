/**************************************************************************/
/* test_syrk.cpp                                                          */
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
 * @file test_syrk.cpp
 * @brief Tests blas().syrk against the reference kernel over dtypes, layouts and triangles.
 * @author Mustafa Garip
 */

#include <vector>
#include <complex>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.h>
#include "../../common/test_utils.hpp"
#include "../../common/blas_flags.hpp"

namespace SushiBLAS
{
    namespace Test
    {
        template <typename T>
        class Unit_SYRKTest : public SushiBLASTest 
        {
            public:
                void test_accuracy(int64_t n, int64_t k, Core::Layout layout = Core::Layout::ROW_MAJOR, bool upper = false, bool transA = false) 
                {
                    reinit_engine(layout);

                    auto dtype = Core::DataType::FLOAT32;
                    if constexpr (std::is_same_v<T, double>) dtype = Core::DataType::FLOAT64;
                    if constexpr (std::is_same_v<T, std::complex<float>>) dtype = Core::DataType::COMPLEX32;
                    if constexpr (std::is_same_v<T, std::complex<double>>) dtype = Core::DataType::COMPLEX64;

                    int64_t a_rows = transA ? k : n;
                    int64_t a_cols = transA ? n : k;

                    // 1. Prepare Inputs
                    auto A = engine->create_tensor({a_rows, a_cols}, dtype);
                    auto C_actual = engine->create_tensor({n, n}, dtype);
                    auto C_expected = engine->create_tensor({n, n}, dtype);

                    fill_random<T>(A);
                    fill_random<T>(C_actual);
                    
                    // Copy C_actual to C_expected so they start exactly the same (important if beta != 0)
                    auto& q = TestEnvironment::get_context().get_queue();
                    size_t c_size_bytes = n * n * sizeof(T);
                    q.memcpy(C_expected.data(), C_actual.data(), c_size_bytes).wait();

                    double alpha = 1.5;
                    double beta = 0.5;

                    // 2. Compute SYCL (Optimized) result
                    engine->set_mode(ExecutionMode::FORCE_SYCL);
                    SushiBLAS::BLASOps(*engine).syrk(A, C_actual, uplo_of(upper),
                                                     transpose_of(transA), alpha, beta);
                    engine->execute();

                    // 3. Compute Reference (CPU) result
                    engine->set_mode(ExecutionMode::FORCE_REFERENCE);
                    SushiBLAS::BLASOps(*engine).syrk(A, C_expected, uplo_of(upper),
                                                     transpose_of(transA), alpha, beta);
                    engine->execute();

                    // 4. Compare
                    double tol = (dtype == Core::DataType::HALF) ? 1e-2 : 1e-4;
                    SB_ASSERT_TENSOR_NEAR(C_actual, C_expected, tol, T);
                }
        };

        using DataTypes = ::testing::Types<float, double, std::complex<float>, std::complex<double>>;
        SB_TYPED_TEST_SUITE(Unit_SYRKTest, DataTypes);

        TYPED_TEST(Unit_SYRKTest, SmallMatrixLowerNoTrans) 
        {
            this->test_accuracy(4, 3, Core::Layout::ROW_MAJOR, false, false);
        }

        TYPED_TEST(Unit_SYRKTest, SmallMatrixUpperNoTrans) 
        {
            this->test_accuracy(4, 3, Core::Layout::ROW_MAJOR, true, false);
        }

        TYPED_TEST(Unit_SYRKTest, SmallMatrixLowerTrans) 
        {
            this->test_accuracy(4, 3, Core::Layout::ROW_MAJOR, false, true);
        }

        TYPED_TEST(Unit_SYRKTest, SmallMatrixUpperTrans) 
        {
            this->test_accuracy(4, 3, Core::Layout::ROW_MAJOR, true, true);
        }

        TYPED_TEST(Unit_SYRKTest, SmallMatrixColMajor) 
        {
            this->test_accuracy(4, 3, Core::Layout::COLUMN_MAJOR, false, false);
        }

        TYPED_TEST(Unit_SYRKTest, Rectangular) 
        {
            this->test_accuracy(8, 16, Core::Layout::ROW_MAJOR, false, false);
        }
    }
}
