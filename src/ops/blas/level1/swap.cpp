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
        void swap_dispatch(sycl::queue& queue, int64_t n, T* x, int64_t incx, T* y, int64_t incy) 
        {
            SB_LOG_INFO("MKL SWAP: {} elements", n);
            oneapi::mkl::blas::column_major::swap(queue, n, x, incx, y, incy);
        }
    }

    void Level1::swap(Tensor& x, Tensor& y) 
    {
        SB_THROW_IF(x.dtype != y.dtype, "Data type mismatch in SWAP.");
        int64_t n, incx, incy;
        Internal::get_vec_params(x, n, incx);
        int64_t ny; Internal::get_vec_params(y, ny, incy);
        SB_THROW_IF(n != ny, "SWAP requires tensors of the same number of elements.");

        auto& queue = engine_.get_context().get_queue();
        switch (x.dtype) 
        {
            case Core::DataType::FLOAT32: 
                swap_dispatch<float>(queue, n, x.data_as<float>(), incx, y.data_as<float>(), incy); 
                break;
            case Core::DataType::FLOAT64: 
                swap_dispatch<double>(queue, n, x.data_as<double>(), incx, y.data_as<double>(), incy); 
                break;
            case Core::DataType::COMPLEX32: 
                swap_dispatch<std::complex<float>>(queue, n, x.data_as<std::complex<float>>(), incx, y.data_as<std::complex<float>>(), incy); 
                break;
            case Core::DataType::COMPLEX64: 
                swap_dispatch<std::complex<double>>(queue, n, x.data_as<std::complex<double>>(), incx, y.data_as<std::complex<double>>(), incy); 
                break;
            default: 
                SB_THROW_IF(true, "Unsupported data type for SWAP.");
        }
    }
}
