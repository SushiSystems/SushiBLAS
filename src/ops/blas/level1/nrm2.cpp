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
        float nrm2_dispatch(sycl::queue& queue, int64_t n, const T* x, int64_t incx) 
        {
            SB_LOG_INFO("MKL NRM2: {} elements", n);
            ResT* res = sycl::malloc_shared<ResT>(1, queue);
            oneapi::mkl::blas::column_major::nrm2(queue, n, x, incx, res).wait();
            float ret = static_cast<float>(*res);
            sycl::free(res, queue);
            return ret;
        }
    }

    float Level1::nrm2(const Tensor& x) 
    {
        int64_t n, incx;
        Internal::get_vec_params(x, n, incx);
        auto& queue = engine_.get_context().get_queue();

        switch (x.dtype) 
        {
            case Core::DataType::FLOAT32: 
                return nrm2_dispatch<float>(queue, n, x.data_as<float>(), incx);
            case Core::DataType::FLOAT64: 
                return nrm2_dispatch<double, double>(queue, n, x.data_as<double>(), incx);
            case Core::DataType::COMPLEX32: 
                return nrm2_dispatch<std::complex<float>, float>(queue, n, x.data_as<std::complex<float>>(), incx);
            case Core::DataType::COMPLEX64: 
                return nrm2_dispatch<std::complex<double>, double>(queue, n, x.data_as<std::complex<double>>(), incx);
            default: 
                SB_THROW_IF(true, "Unsupported data type for NRM2."); 
                return 0.0f;
        }
    }
}
