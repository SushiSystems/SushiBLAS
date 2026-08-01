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
/* Licensed under the Apache License, Version 2.0 (the "License");        */
/* you may not use this file except in compliance with the License.       */
/* You may obtain a copy of the License at                                */
/*                                                                        */
/*     http://www.apache.org/licenses/LICENSE-2.0                         */
/*                                                                        */
/* Unless required by applicable law or agreed to in writing, software    */
/* distributed under the License is distributed on an "AS IS" BASIS,      */
/* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        */
/* implied. See the License for the specific language governing           */
/* permissions and limitations under the License.                         */
/**************************************************************************/

#include <vector>
#include <complex>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.h>
#include "../../common/test_utils.hpp"

namespace SushiBLAS
{
    namespace Test
    {
        template <typename T>
        class Unit_GEMMTest : public SushiBLASTest 
        {
            public:
                void test_accuracy(int64_t m, int64_t n, int64_t k, Core::Layout layout = Core::Layout::ROW_MAJOR)
                {
                    reinit_engine(layout);

                    auto dtype = Core::DataType::FLOAT32;
                    if constexpr (std::is_same_v<T, double>) dtype = Core::DataType::FLOAT64;
                    if constexpr (std::is_same_v<T, sycl::half>) dtype = Core::DataType::HALF;
                    if constexpr (std::is_same_v<T, std::complex<float>>) dtype = Core::DataType::COMPLEX32;
                    if constexpr (std::is_same_v<T, std::complex<double>>) dtype = Core::DataType::COMPLEX64;

                    // 1. Prepare Inputs
                    auto A = engine->create_tensor({m, k}, dtype);
                    auto B = engine->create_tensor({k, n}, dtype);
                    auto C_actual = engine->create_tensor({m, n}, dtype);
                    auto C_expected = engine->create_tensor({m, n}, dtype);

                    fill_random<T>(A);
                    fill_random<T>(B);
                    
                    // 2. Compute SYCL (Optimized) result
                    engine->set_mode(ExecutionMode::FORCE_SYCL);
                    engine->blas().gemm(A, B, C_actual);
                    engine->execute().wait();

                    // 3. Compute Reference (CPU) result
                    engine->set_mode(ExecutionMode::FORCE_REFERENCE);
                    engine->blas().gemm(A, B, C_expected);
                    engine->execute().wait();
                    
                    // 4. Compare
                    double tol = (dtype == Core::DataType::HALF) ? 1e-2 : 1e-4;
                    SB_ASSERT_TENSOR_NEAR(C_actual, C_expected, tol, T);
                }
            };

            using DataTypes = ::testing::Types<float, double, sycl::half, std::complex<float>, std::complex<double>>;
            TYPED_TEST_SUITE(Unit_GEMMTest, DataTypes);

            TYPED_TEST(Unit_GEMMTest, SmallMatrix)
            {
                this->test_accuracy(4, 4, 4);
            }

            TYPED_TEST(Unit_GEMMTest, SmallMatrixColMajor)
            {
                this->test_accuracy(4, 4, 4, Core::Layout::COLUMN_MAJOR);
            }

            TYPED_TEST(Unit_GEMMTest, Rectangular)
            {
                this->test_accuracy(8, 16, 4);
            }

            TYPED_TEST(Unit_GEMMTest, RectangularColMajor)
            {
                this->test_accuracy(8, 16, 4, Core::Layout::COLUMN_MAJOR);
            }

    } // namespace Test
} // namespace SushiBLAS::Test
