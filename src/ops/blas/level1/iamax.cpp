#include <SushiBLAS/ops/blas/utils.hpp>
#include <complex>
#include <oneapi/mkl.hpp>
#include <SushiBLAS/engine.hpp>
#include <SushiBLAS/ops/blas/level1.hpp>

namespace SushiBLAS 
{
    namespace
    {
        template<typename T>
        int64_t iamax_dispatch(sycl::queue& queue, int64_t n, const T* x, int64_t incx) 
        {
            SB_LOG_INFO("MKL IAMAX: {} elements", n);
            int64_t* res = sycl::malloc_shared<int64_t>(1, queue);
            oneapi::mkl::blas::column_major::iamax(queue, n, x, incx, res).wait();
            int64_t ret = *res;
            sycl::free(res, queue);
            return ret;
        }
    }

    int64_t Level1::iamax(const Tensor& x) 
    {
        int64_t n, incx;
        Internal::get_vec_params(x, n, incx);
        auto& queue = engine_.get_context().get_queue();

        switch (x.dtype) 
        {
            case Core::DataType::FLOAT32: 
                return iamax_dispatch<float>(queue, n, x.data_as<float>(), incx);
            case Core::DataType::FLOAT64: 
                return iamax_dispatch<double>(queue, n, x.data_as<double>(), incx);
            case Core::DataType::COMPLEX32: 
                return iamax_dispatch<std::complex<float>>(queue, n, x.data_as<std::complex<float>>(), incx);
            case Core::DataType::COMPLEX64: 
                return iamax_dispatch<std::complex<double>>(queue, n, x.data_as<std::complex<double>>(), incx);
            default: 
                SB_THROW_IF(true, "Unsupported data type for IAMAX."); 
                return 0;
        }
    }
}
