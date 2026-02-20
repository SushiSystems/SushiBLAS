#include <complex>
#include <oneapi/mkl.hpp>
#include <SushiBLAS/engine.hpp>
#include <SushiBLAS/ops/blas/level1.hpp>
#include <SushiBLAS/ops/blas/utils.hpp>

namespace SushiBLAS 
{
    namespace
    {
        template<typename T, typename ResT = T>
        float dot_dispatch(sycl::queue& queue, int64_t n, const T* x, int64_t incx, const T* y, int64_t incy) 
        {
            SB_LOG_INFO("MKL DOT: {} elements", n);
            ResT* res = sycl::malloc_shared<ResT>(1, queue);
            oneapi::mkl::blas::column_major::dot(queue, n, x, incx, y, incy, res).wait();
            float ret = static_cast<float>(std::abs(*res));
            sycl::free(res, queue);
            return ret;
        }

        template<>
        float dot_dispatch<std::complex<float>, std::complex<float>>(sycl::queue& queue, int64_t n, const std::complex<float>* x, int64_t incx, const std::complex<float>* y, int64_t incy) 
        {
            SB_LOG_INFO("MKL DOTC (Complex32): {} elements", n);
            std::complex<float>* res = sycl::malloc_shared<std::complex<float>>(1, queue);
            oneapi::mkl::blas::column_major::dotc(queue, n, x, incx, y, incy, res).wait();
            float ret = std::abs(*res);
            sycl::free(res, queue);
            return ret;
        }

        template<>
        float dot_dispatch<std::complex<double>, std::complex<double>>(sycl::queue& queue, int64_t n, const std::complex<double>* x, int64_t incx, const std::complex<double>* y, int64_t incy) 
        {
            SB_LOG_INFO("MKL DOTC (Complex64): {} elements", n);
            std::complex<double>* res = sycl::malloc_shared<std::complex<double>>(1, queue);
            oneapi::mkl::blas::column_major::dotc(queue, n, x, incx, y, incy, res).wait();
            float ret = static_cast<float>(std::abs(*res));
            sycl::free(res, queue);
            return ret;
        }
    }

    float Level1::dot(const Tensor& x, const Tensor& y) 
    {
        SB_THROW_IF(x.dtype != y.dtype, "Data type mismatch in DOT.");
        int64_t n, incx, incy;
        Internal::get_vec_params(x, n, incx);
        int64_t ny; 
        Internal::get_vec_params(y, ny, incy);
        SB_THROW_IF(n != ny, "DOT requires tensors of the same number of elements.");

        auto& queue = engine_.get_context().get_queue();
        switch (x.dtype) 
        {
            case Core::DataType::FLOAT32: 
                return dot_dispatch<float>(queue, n, x.data_as<float>(), incx, y.data_as<float>(), incy);
            case Core::DataType::FLOAT64: 
                return dot_dispatch<double>(queue, n, x.data_as<double>(), incx, y.data_as<double>(), incy);
            case Core::DataType::COMPLEX32: 
                return dot_dispatch<std::complex<float>>(queue, n, x.data_as<std::complex<float>>(), incx, y.data_as<std::complex<float>>(), incy);
            case Core::DataType::COMPLEX64: 
                return dot_dispatch<std::complex<double>>(queue, n, x.data_as<std::complex<double>>(), incx, y.data_as<std::complex<double>>(), incy);
            default: 
                SB_THROW_IF(true, "Unsupported data type for DOT."); return 0.0f;
        }
    }
}
