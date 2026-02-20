#include <complex>
#include <oneapi/mkl.hpp>
#include <SushiBLAS/engine.hpp>
#include <SushiBLAS/ops/blas/level1.hpp>
#include <SushiBLAS/ops/blas/utils.hpp>

namespace SushiBLAS 
{
    namespace
    {
        template<typename T, typename Tc, typename Ts>
        void rot_dispatch(sycl::queue& queue, int64_t n, T* x, int64_t incx, T* y, int64_t incy, Tc c, Ts s) 
        {
            SB_LOG_INFO("MKL ROT: {} elements", n);
            oneapi::mkl::blas::column_major::rot(queue, n, x, incx, y, incy, c, s);
        }
    }

    void Level1::rot(Tensor& x, Tensor& y, float c, float s) 
    {
        SB_THROW_IF(x.dtype != y.dtype, "Data type mismatch in ROT.");
        int64_t n, incx, incy;
        Internal::get_vec_params(x, n, incx);
        int64_t ny; Internal::get_vec_params(y, ny, incy);
        SB_THROW_IF(n != ny, "ROT requires tensors of the same number of elements.");

        auto& queue = engine_.get_context().get_queue();
        switch (x.dtype) 
        {
            case Core::DataType::FLOAT32: 
                rot_dispatch(queue, n, x.data_as<float>(), incx, y.data_as<float>(), incy, c, s); 
                break;
            case Core::DataType::FLOAT64: 
                rot_dispatch(queue, n, x.data_as<double>(), incx, y.data_as<double>(), incy, static_cast<double>(c), static_cast<double>(s)); 
                break;
            case Core::DataType::COMPLEX32: 
                // For complex, c is real (float), s is complex.
                rot_dispatch(queue, n, x.data_as<std::complex<float>>(), incx, y.data_as<std::complex<float>>(), incy, c, std::complex<float>(s, 0.0f)); 
                break;
            case Core::DataType::COMPLEX64: 
                // For complex, c is real (double), s is complex.
                rot_dispatch(queue, n, x.data_as<std::complex<double>>(), incx, y.data_as<std::complex<double>>(), incy, static_cast<double>(c), std::complex<double>(s, 0.0)); 
                break;
            default: 
                SB_THROW_IF(true, "Unsupported data type for ROT.");
        }
    }
}
