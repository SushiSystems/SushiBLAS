/****************************************************************/
/* transforms.hpp                                               */
/* SushiBLAS - https://github.com/SushiSystems/SushiBLAS        */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

#pragma once

// EXPERIMENTAL - quarantined per SB-2/D1. Not included from engine.hpp and not
// reachable via Engine::signal() - that accessor was removed. No .cpp
// implementation exists for any method declared here. The FFT backend
// strategy is a named follow-up (P8), not solved by this file's presence.

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
