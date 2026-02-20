/**************************************************************************/
/* ger.cpp                                                                */
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
        void ger_dispatch(sycl::queue& queue, Core::Layout layout,
                          int64_t m, int64_t n,
                          T alpha, const T* x, int64_t incx,
                          const T* y, int64_t incy,
                          T* a, int64_t lda) 
        {
            if (layout == Core::Layout::ROW_MAJOR) 
            {
                SB_LOG_INFO("MKL GER [Row-Major]: {}x{}", m, n);
                // Note: ger doesn't distinguish between geru and gerc in float/double, but in complex it does. 
                // We use geru as the generic ger.
                oneapi::mkl::blas::row_major::geru(queue, m, n, alpha, x, incx, y, incy, a, lda);
            } 
            else 
            {
                SB_LOG_INFO("MKL GER [Col-Major]: {}x{}", m, n);
                oneapi::mkl::blas::column_major::geru(queue, m, n, alpha, x, incx, y, incy, a, lda);
            }
        }
        
        // Float specific specializations (oneMKL ger for real numbers is just ger, not geru)
        template<>
        void ger_dispatch<float>(sycl::queue& queue, Core::Layout layout, int64_t m, int64_t n, float alpha, const float* x, int64_t incx, const float* y, int64_t incy, float* a, int64_t lda) 
        {
            if (layout == Core::Layout::ROW_MAJOR) 
            {
                SB_LOG_INFO("MKL GER [Row-Major]: {}x{}", m, n);
                oneapi::mkl::blas::row_major::ger(queue, m, n, alpha, x, incx, y, incy, a, lda);
            } else {
                SB_LOG_INFO("MKL GER [Col-Major]: {}x{}", m, n);
                oneapi::mkl::blas::column_major::ger(queue, m, n, alpha, x, incx, y, incy, a, lda);
            }
        }
        
        template<>
        void ger_dispatch<double>(sycl::queue& queue, Core::Layout layout, int64_t m, int64_t n, double alpha, const double* x, int64_t incx, const double* y, int64_t incy, double* a, int64_t lda) 
        {
            if (layout == Core::Layout::ROW_MAJOR) 
            {
                SB_LOG_INFO("MKL GER [Row-Major]: {}x{}", m, n);
                oneapi::mkl::blas::row_major::ger(queue, m, n, alpha, x, incx, y, incy, a, lda);
            } else {
                SB_LOG_INFO("MKL GER [Col-Major]: {}x{}", m, n);
                oneapi::mkl::blas::column_major::ger(queue, m, n, alpha, x, incx, y, incy, a, lda);
            }
        }
    }

    void Level2::ger(const Tensor& x, const Tensor& y, Tensor& A, float alpha) 
    {
        SB_THROW_IF(A.rank < 2, "GER requires A to be at least a 2D matrix.");
        SB_THROW_IF(A.dtype != x.dtype || A.dtype != y.dtype, "Data type mismatch in GER.");

        int32_t rA = A.rank;
        int64_t m = A.shape[rA - 2];
        int64_t n = A.shape[rA - 1];

        int64_t lda = (A.layout == Core::Layout::ROW_MAJOR) ? A.shape[rA - 1] : A.shape[rA - 2];

        int64_t nx, incx;
        Internal::get_vec_params(x, nx, incx);
        
        int64_t ny, incy;
        Internal::get_vec_params(y, ny, incy);

        SB_THROW_IF(nx != m, "Dimension mismatch for vector x in GER.");
        SB_THROW_IF(ny != n, "Dimension mismatch for vector y in GER.");

        auto& queue = engine_.get_context().get_queue();

        switch (A.dtype) 
        {
            case Core::DataType::FLOAT32:
                ger_dispatch<float>(queue, A.layout, m, n, alpha, x.data_as<float>(), incx, y.data_as<float>(), incy, A.data_as<float>(), lda);
                break;
            case Core::DataType::FLOAT64:
                ger_dispatch<double>(queue, A.layout, m, n, static_cast<double>(alpha), x.data_as<double>(), incx, y.data_as<double>(), incy, A.data_as<double>(), lda);
                break;
            case Core::DataType::COMPLEX32:
                ger_dispatch<std::complex<float>>(queue, A.layout, m, n, std::complex<float>(alpha, 0.0f), x.data_as<std::complex<float>>(), incx, y.data_as<std::complex<float>>(), incy, A.data_as<std::complex<float>>(), lda);
                break;
            case Core::DataType::COMPLEX64:
                ger_dispatch<std::complex<double>>(queue, A.layout, m, n, std::complex<double>(alpha, 0.0), x.data_as<std::complex<double>>(), incx, y.data_as<std::complex<double>>(), incy, A.data_as<std::complex<double>>(), lda);
                break;
            default:
                SB_THROW_IF(true, "Unsupported data type for GER.");
        }
    }
}
