/**************************************************************************/
/* symv.cpp                                                               */
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
        void symv_dispatch(sycl::queue& queue, Core::Layout layout,
                           oneapi::mkl::uplo uplo,
                           int64_t n, T alpha, const T* a, int64_t lda,
                           const T* x, int64_t incx,
                           T beta, T* y, int64_t incy) 
        {
            if (layout == Core::Layout::ROW_MAJOR) 
            {
                SB_LOG_INFO("MKL SYMV [Row-Major]: {}x{}", n, n);
                oneapi::mkl::blas::row_major::symv(queue, uplo, n, alpha, a, lda, x, incx, beta, y, incy);
            } 
            else 
            {
                SB_LOG_INFO("MKL SYMV [Col-Major]: {}x{}", n, n);
                oneapi::mkl::blas::column_major::symv(queue, uplo, n, alpha, a, lda, x, incx, beta, y, incy);
            }
        }
    }

    void Level2::symv(const Tensor& A, const Tensor& x, Tensor& y, 
                      bool upper, float alpha, float beta) 
    {
        SB_THROW_IF(A.rank < 2, "SYMV requires A to be at least a 2D matrix.");
        SB_THROW_IF(A.dtype != x.dtype || A.dtype != y.dtype, "Data type mismatch in SYMV.");

        int32_t rA = A.rank;
        int64_t n = A.shape[rA - 1]; // Symmetric matrix must be n x n
        SB_THROW_IF(A.shape[rA - 2] != n, "SYMV requires A to be a square matrix.");

        int64_t lda = (A.layout == Core::Layout::ROW_MAJOR) ? A.shape[rA - 1] : A.shape[rA - 2];

        int64_t nx, incx;
        Internal::get_vec_params(x, nx, incx);
        
        int64_t ny, incy;
        Internal::get_vec_params(y, ny, incy);

        SB_THROW_IF(nx != n, "Dimension mismatch for vector x in SYMV.");
        SB_THROW_IF(ny != n, "Dimension mismatch for vector y in SYMV.");

        auto& queue = engine_.get_context().get_queue();
        auto mkl_uplo = upper ? oneapi::mkl::uplo::upper : oneapi::mkl::uplo::lower;

        switch (A.dtype) 
        {
            case Core::DataType::FLOAT32:
                symv_dispatch<float>(queue, A.layout, mkl_uplo, n, alpha, A.data_as<float>(), lda, x.data_as<float>(), incx, beta, y.data_as<float>(), incy);
                break;
            case Core::DataType::FLOAT64:
                symv_dispatch<double>(queue, A.layout, mkl_uplo, n, static_cast<double>(alpha), A.data_as<double>(), lda, x.data_as<double>(), incx, static_cast<double>(beta), y.data_as<double>(), incy);
                break;
            case Core::DataType::COMPLEX32: // Note: For complex numbers, HEMV is usually used for Hermitian matrices, but SYMV exists for complex symmetric. We use SYMV here.
                symv_dispatch<std::complex<float>>(queue, A.layout, mkl_uplo, n, std::complex<float>(alpha, 0.0f), A.data_as<std::complex<float>>(), lda, x.data_as<std::complex<float>>(), incx, std::complex<float>(beta, 0.0f), y.data_as<std::complex<float>>(), incy);
                break;
            case Core::DataType::COMPLEX64:
                symv_dispatch<std::complex<double>>(queue, A.layout, mkl_uplo, n, std::complex<double>(alpha, 0.0), A.data_as<std::complex<double>>(), lda, x.data_as<std::complex<double>>(), incx, std::complex<double>(beta, 0.0), y.data_as<std::complex<double>>(), incy);
                break;
            default:
                SB_THROW_IF(true, "Unsupported data type for SYMV.");
        }
    }
}
