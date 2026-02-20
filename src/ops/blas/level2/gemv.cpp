/**************************************************************************/
/* gemv.cpp                                                               */
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
        void gemv_dispatch(sycl::queue& queue, Core::Layout layout,
                           oneapi::mkl::transpose transA,
                           int64_t m, int64_t n,
                           T alpha, const T* a, int64_t lda,
                           const T* x, int64_t incx,
                           T beta, T* y, int64_t incy) 
        {
            if (layout == Core::Layout::ROW_MAJOR) 
            {
                SB_LOG_INFO("MKL GEMV [Row-Major]: {}x{}", m, n);
                oneapi::mkl::blas::row_major::gemv(queue, transA, m, n, alpha, a, lda, x, incx, beta, y, incy);
            } 
            else 
            {
                SB_LOG_INFO("MKL GEMV [Col-Major]: {}x{}", m, n);
                oneapi::mkl::blas::column_major::gemv(queue, transA, m, n, alpha, a, lda, x, incx, beta, y, incy);
            }
        }
    }

    void Level2::gemv(const Tensor& A, const Tensor& x, Tensor& y,
                      bool transA, float alpha, float beta) 
    {
        // Validation
        SB_THROW_IF(A.rank < 2, "GEMV requires A to be at least a 2D matrix.");
        SB_THROW_IF(A.dtype != x.dtype || A.dtype != y.dtype, "Data type mismatch in GEMV.");

        int32_t rA = A.rank;
        int64_t m = A.shape[rA - 2];
        int64_t n = A.shape[rA - 1];

        int64_t lda = (A.layout == Core::Layout::ROW_MAJOR) ? A.shape[rA - 1] : A.shape[rA - 2];

        int64_t nx, incx;
        Internal::get_vec_params(x, nx, incx);
        
        int64_t ny, incy;
        Internal::get_vec_params(y, ny, incy);

        int64_t expected_nx = transA ? m : n;
        int64_t expected_ny = transA ? n : m;

        SB_THROW_IF(nx != expected_nx, "Dimension mismatch for vector x in GEMV.");
        SB_THROW_IF(ny != expected_ny, "Dimension mismatch for vector y in GEMV.");

        auto& queue = engine_.get_context().get_queue();
        auto mkl_transA = transA ? oneapi::mkl::transpose::trans : oneapi::mkl::transpose::nontrans;

        switch (A.dtype) 
        {
            case Core::DataType::FLOAT32:
                gemv_dispatch<float>(queue, A.layout, mkl_transA, m, n, alpha, A.data_as<float>(), lda, x.data_as<float>(), incx, beta, y.data_as<float>(), incy);
                break;
            case Core::DataType::FLOAT64:
                gemv_dispatch<double>(queue, A.layout, mkl_transA, m, n, static_cast<double>(alpha), A.data_as<double>(), lda, x.data_as<double>(), incx, static_cast<double>(beta), y.data_as<double>(), incy);
                break;
            case Core::DataType::COMPLEX32:
                gemv_dispatch<std::complex<float>>(queue, A.layout, mkl_transA, m, n, std::complex<float>(alpha, 0.0f), A.data_as<std::complex<float>>(), lda, x.data_as<std::complex<float>>(), incx, std::complex<float>(beta, 0.0f), y.data_as<std::complex<float>>(), incy);
                break;
            case Core::DataType::COMPLEX64:
                gemv_dispatch<std::complex<double>>(queue, A.layout, mkl_transA, m, n, std::complex<double>(alpha, 0.0), A.data_as<std::complex<double>>(), lda, x.data_as<std::complex<double>>(), incx, std::complex<double>(beta, 0.0), y.data_as<std::complex<double>>(), incy);
                break;
            default:
                SB_THROW_IF(true, "Unsupported data type for GEMV.");
        }
    }
}
