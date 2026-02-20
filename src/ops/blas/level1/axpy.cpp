#include <complex>
#include <oneapi/mkl.hpp>
#include <SushiBLAS/engine.hpp>
#include <SushiBLAS/ops/blas/level1.hpp>
#include <SushiBLAS/ops/blas/utils.hpp>

namespace SushiBLAS 
{
    namespace
    {
        template<typename T>
        void axpy_dispatch(sycl::queue& queue, int64_t n, T alpha, const T* x, int64_t incx, T* y, int64_t incy) 
        {
            SB_LOG_INFO("MKL AXPY: {} elements", n);
            oneapi::mkl::blas::column_major::axpy(queue, n, alpha, x, incx, y, incy);
        }
    }

    void Level1::axpy(float alpha, const Tensor& x, Tensor& y) 
    {
        SB_THROW_IF(x.dtype != y.dtype, "Data type mismatch in AXPY.");
        int64_t n, incx, incy;
        Internal::get_vec_params(x, n, incx);
        int64_t ny; Internal::get_vec_params(y, ny, incy);
        SB_THROW_IF(n != ny, "AXPY requires tensors of the same number of elements.");

        auto& queue = engine_.get_context().get_queue();
        switch (x.dtype) 
        {
            case Core::DataType::FLOAT32: 
                axpy_dispatch<float>(queue, n, alpha, x.data_as<float>(), incx, y.data_as<float>(), incy); break;
            case Core::DataType::FLOAT64: 
                axpy_dispatch<double>(queue, n, static_cast<double>(alpha), x.data_as<double>(), incx, y.data_as<double>(), incy); break;
            case Core::DataType::COMPLEX32: 
                axpy_dispatch<std::complex<float>>(queue, n, std::complex<float>(alpha, 0.0f), x.data_as<std::complex<float>>(), incx, y.data_as<std::complex<float>>(), incy); break;
            case Core::DataType::COMPLEX64: 
                axpy_dispatch<std::complex<double>>(queue, n, std::complex<double>(alpha, 0.0), x.data_as<std::complex<double>>(), incx, y.data_as<std::complex<double>>(), incy); break;
            default: 
                SB_THROW_IF(true, "Unsupported data type for AXPY.");
        }
    }
}
