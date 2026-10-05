/****************************************************************/
/* transforms.hpp                                               */
/* SushiBLAS - https://github.com/SushiSystems/SushiBLAS        */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

/**
 * @file transforms.hpp
 * @brief Declares TransformsOps, the experimental signal transforms that have no implementation yet.
 * @author Mustafa Garip
 */

#pragma once

#include <sycl/sycl.hpp>
#include <SushiBLAS/tensor.hpp>

namespace SushiBLAS
{
    class Engine;

    /**
     * @class TransformsOps
     * @brief Signal transformation operations (FFT, etc.).
     *
     * Provides interfaces for Fast Fourier Transforms (FFT) in multiple dimensions.
     * Uses oneMKL's high-performance FFT domain underneath.
     * @see include/SushiBLAS/experimental/README.md
     */
    class TransformsOps
    {
        public:
            explicit TransformsOps(Engine& e) : engine_(e) {}

            /**
             * @brief 1D Forward Fast Fourier Transform.
             * @param in Input tensor (time domain).
             * @param out Output tensor (frequency domain, complex).
             * @return sycl::event.
             */
            sycl::event fft1d(const Tensor& in, Tensor& out);

            /**
             * @brief 1D Inverse Fast Fourier Transform.
             * @param in Input tensor (frequency domain).
             * @param out Output tensor (time domain).
             * @return sycl::event.
             */
            sycl::event ifft1d(const Tensor& in, Tensor& out);

            /**
             * @brief 2D Forward Fast Fourier Transform (Gratings/Images).
             * @param in Input tensor.
             * @param out Output complex tensor.
             * @return sycl::event.
             */
            sycl::event fft2d(const Tensor& in, Tensor& out);

            /**
             * @brief 2D Inverse Fast Fourier Transform.
             * @param in Input complex tensor.
             * @param out Output tensor.
             * @return sycl::event.
             */
            sycl::event ifft2d(const Tensor& in, Tensor& out);

        private:
            Engine& engine_;
    };

} // namespace SushiBLAS
