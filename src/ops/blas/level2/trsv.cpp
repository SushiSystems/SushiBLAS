/**************************************************************************/
/* trsv.cpp                                                               */
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
        void trsv_dispatch(sycl::queue& queue, Core::Layout layout,
                           oneapi::mkl::uplo uplo, oneapi::mkl::transpose trans, oneapi::mkl::diag diag,
                           int64_t n, const T* a, int64_t lda,
                           T* x, int64_t incx) 
        {
            if (layout == Core::Layout::ROW_MAJOR) 
            {
                SB_LOG_INFO("MKL TRSV [Row-Major]: {}x{}", n, n);
                oneapi::mkl::blas::row_major::trsv(queue, uplo, trans, diag, n, a, lda, x, incx);
            } 
            else 
            {
                SB_LOG_INFO("MKL TRSV [Col-Major]: {}x{}", n, n);
                oneapi::mkl::blas::column_major::trsv(queue, uplo, trans, diag, n, a, lda, x, incx);
            }
        }
    }

    void Level2::trsv(const Tensor& A, Tensor& b, 
                      bool upper, bool transA, bool unit_diag) 
    {
        SB_THROW_IF(A.rank < 2, "TRSV requires A to be at least a 2D matrix.");
        SB_THROW_IF(A.dtype != b.dtype, "Data type mismatch in TRSV.");

        int32_t rA = A.rank;
        int64_t n = A.shape[rA - 1]; // Triangular matrix must be n x n
        SB_THROW_IF(A.shape[rA - 2] != n, "TRSV requires A to be a square matrix.");

        int64_t lda = (A.layout == Core::Layout::ROW_MAJOR) ? A.shape[rA - 1] : A.shape[rA - 2];

        int64_t nx, incx;
        Internal::get_vec_params(b, nx, incx);

        SB_THROW_IF(nx != n, "Dimension mismatch for vector b in TRSV.");

        auto& queue = engine_.get_context().get_queue();
        auto mkl_uplo = upper ? oneapi::mkl::uplo::upper : oneapi::mkl::uplo::lower;
        auto mkl_trans = transA ? oneapi::mkl::transpose::trans : oneapi::mkl::transpose::nontrans;
        auto mkl_diag = unit_diag ? oneapi::mkl::diag::unit : oneapi::mkl::diag::nonunit;

        switch (A.dtype) 
        {
            case Core::DataType::FLOAT32:
                trsv_dispatch<float>(queue, A.layout, mkl_uplo, mkl_trans, mkl_diag, n, A.data_as<float>(), lda, b.data_as<float>(), incx);
                break;
            case Core::DataType::FLOAT64:
                trsv_dispatch<double>(queue, A.layout, mkl_uplo, mkl_trans, mkl_diag, n, A.data_as<double>(), lda, b.data_as<double>(), incx);
                break;
            case Core::DataType::COMPLEX32:
                trsv_dispatch<std::complex<float>>(queue, A.layout, mkl_uplo, mkl_trans, mkl_diag, n, A.data_as<std::complex<float>>(), lda, b.data_as<std::complex<float>>(), incx);
                break;
            case Core::DataType::COMPLEX64:
                trsv_dispatch<std::complex<double>>(queue, A.layout, mkl_uplo, mkl_trans, mkl_diag, n, A.data_as<std::complex<double>>(), lda, b.data_as<std::complex<double>>(), incx);
                break;
            default:
                SB_THROW_IF(true, "Unsupported data type for TRSV.");
        }
    }
}
