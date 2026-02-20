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
        void scal_dispatch(sycl::queue& queue, int64_t n, T alpha, T* x, int64_t incx) 
        {
            SB_LOG_INFO("MKL SCAL: {} elements", n);
            oneapi::mkl::blas::column_major::scal(queue, n, alpha, x, incx);
        }
    }

    void Level1::scal(float alpha, Tensor& x) 
    {
        int64_t n, incx;
        Internal::get_vec_params(x, n, incx);
        auto& queue = engine_.get_context().get_queue();

        switch (x.dtype) 
        {
            case Core::DataType::FLOAT32: 
                scal_dispatch<float>(queue, n, alpha, x.data_as<float>(), incx); 
                break;
            case Core::DataType::FLOAT64: 
                scal_dispatch<double>(queue, n, static_cast<double>(alpha), x.data_as<double>(), incx); 
                break;
            case Core::DataType::COMPLEX32: 
                scal_dispatch<std::complex<float>>(queue, n, std::complex<float>(alpha, 0.0f), x.data_as<std::complex<float>>(), incx); 
                break;
            case Core::DataType::COMPLEX64: 
                scal_dispatch<std::complex<double>>(queue, n, std::complex<double>(alpha, 0.0), x.data_as<std::complex<double>>(), incx); 
                break;
            default: 
                SB_THROW_IF(true, "Unsupported data type for SCAL.");
        }
    }
}
