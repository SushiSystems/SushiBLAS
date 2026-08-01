/**************************************************************************/
/* transforms.hpp                                                         */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                SushiBLAS                               */
/*                https://github.com/SushiSystems/SushiBLAS               */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present Mustafa Garip & Sushi Systems               */
/*                                                                        */
/* Licensed under the Apache License, Version 2.0 (the "License");        */
/* you may not use this file except in compliance with the License.       */
/* You may obtain a copy of the License at                                */
/*                                                                        */
/*     http://www.apache.org/licenses/LICENSE-2.0                         */
/*                                                                        */
/* Unless required by applicable law or agreed to in writing, software    */
/* distributed under the License is distributed on an "AS IS" BASIS,      */
/* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        */
/* implied. See the License for the specific language governing           */
/* permissions and limitations under the License.                         */
/**************************************************************************/

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
