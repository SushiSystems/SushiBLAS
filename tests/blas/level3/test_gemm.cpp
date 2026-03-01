/**************************************************************************/
/* test_gemm.cpp                                                          */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                SushiBLAS                               */
/*                https://github.com/SushiSystems/SushiBLAS               */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present  Mustafa Garip & Sushi Systems              */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#include <vector>
#include <complex>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.h>
#include "../../test_utils.hpp"

namespace SushiBLAS
{
    namespace Test
    {
        template <typename T>
        class GEMMTest : public SushiBLASTest 
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
            TYPED_TEST_SUITE(GEMMTest, DataTypes);

            TYPED_TEST(GEMMTest, SmallMatrix)
            {
                this->test_accuracy(4, 4, 4);
            }

            TYPED_TEST(GEMMTest, SmallMatrixColMajor)
            {
                this->test_accuracy(4, 4, 4, Core::Layout::COLUMN_MAJOR);
            }

            TYPED_TEST(GEMMTest, Rectangular)
            {
                this->test_accuracy(8, 16, 4);
            }

            TYPED_TEST(GEMMTest, RectangularColMajor)
            {
                this->test_accuracy(8, 16, 4, Core::Layout::COLUMN_MAJOR);
            }

    } // namespace Test
} // namespace SushiBLAS::Test
