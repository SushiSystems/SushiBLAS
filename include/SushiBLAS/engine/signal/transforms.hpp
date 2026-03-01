/**************************************************************************/
/* transforms.hpp                                                         */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                SushiBLAS                               */
/*                https://github.com/SushiSystems/SushiBLAS               */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present  Mustafa Garip & Sushi Systems              */
/*                                                                   	  */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

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
