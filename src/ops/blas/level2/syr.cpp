/**************************************************************************/
/* syr.cpp                                                                */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                SushiBLAS                               */
/*                https://github.com/SushiSystems/SushiBLAS               */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present  Mustafa Garip & Sushi Systems              */
/**************************************************************************/

#include <complex>
#include <oneapi/mkl.hpp>
#include <SushiBLAS/engine.hpp>
#include <SushiBLAS/ops/blas/level2.hpp>
#include <SushiBLAS/ops/blas/utils.hpp>

namespace SushiBLAS 
{
    namespace
    {
        template<typename T>
        void syr_dispatch(sycl::queue& queue, Core::Layout layout,
                          oneapi::mkl::uplo uplo,
                          int64_t n, T alpha, const T* x, int64_t incx,
                          T* a, int64_t lda) 
        {
            if (layout == Core::Layout::ROW_MAJOR) 
            {
                SB_LOG_INFO("MKL SYR [Row-Major]: {}x{}", n, n);
                // Note: SYR is defined for real numbers in standard BLAS. 
                // For complex, HER is used if Hermitian, but oneMKL does provide syr for complex in some versions or it's just 'syr'.
                // Standard BLAS SYR works for float/double/complex.
                oneapi::mkl::blas::row_major::syr(queue, uplo, n, alpha, x, incx, a, lda);
            } 
            else 
            {
                SB_LOG_INFO("MKL SYR [Col-Major]: {}x{}", n, n);
                oneapi::mkl::blas::column_major::syr(queue, uplo, n, alpha, x, incx, a, lda);
            }
        }
    }

    void Level2::syr(const Tensor& x, Tensor& A, bool upper, float alpha) 
    {
        SB_THROW_IF(A.rank < 2, "SYR requires A to be at least a 2D matrix.");
        SB_THROW_IF(A.dtype != x.dtype, "Data type mismatch in SYR.");

        int32_t rA = A.rank;
        int64_t n = A.shape[rA - 1]; // Symmetric matrix must be n x n
        SB_THROW_IF(A.shape[rA - 2] != n, "SYR requires A to be a square matrix.");

        int64_t lda = (A.layout == Core::Layout::ROW_MAJOR) ? A.shape[rA - 1] : A.shape[rA - 2];

        int64_t nx, incx;
        Internal::get_vec_params(x, nx, incx);

        SB_THROW_IF(nx != n, "Dimension mismatch for vector x in SYR.");

        auto& queue = engine_.get_context().get_queue();
        auto mkl_uplo = upper ? oneapi::mkl::uplo::upper : oneapi::mkl::uplo::lower;

        switch (A.dtype) 
        {
            case Core::DataType::FLOAT32:
                syr_dispatch<float>(queue, A.layout, mkl_uplo, n, alpha, x.data_as<float>(), incx, A.data_as<float>(), lda);
                break;
            case Core::DataType::FLOAT64:
                syr_dispatch<double>(queue, A.layout, mkl_uplo, n, static_cast<double>(alpha), x.data_as<double>(), incx, A.data_as<double>(), lda);
                break;
            case Core::DataType::COMPLEX32:
                syr_dispatch<std::complex<float>>(queue, A.layout, mkl_uplo, n, std::complex<float>(alpha, 0.0f), x.data_as<std::complex<float>>(), incx, A.data_as<std::complex<float>>(), lda);
                break;
            case Core::DataType::COMPLEX64:
                syr_dispatch<std::complex<double>>(queue, A.layout, mkl_uplo, n, std::complex<double>(alpha, 0.0), x.data_as<std::complex<double>>(), incx, A.data_as<std::complex<double>>(), lda);
                break;
            default:
                SB_THROW_IF(true, "Unsupported data type for SYR.");
        }
    }
}
