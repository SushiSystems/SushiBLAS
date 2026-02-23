/**************************************************************************/
/* random.hpp                                                             */
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
     * @class RandomOps
     * @brief Random number generators for tensor initialization.
     * 
     * Uses oneMKL's Vector Statistics Library (VSL) to generate numbers 
     * from various distributions directly on the accelerator.
     */
    class RandomOps 
    {
        public:
            explicit RandomOps(Engine& e) : engine_(e) {}

            /**
             * @brief Fill tensor with values from a uniform distribution [min, max).
             * @param t Tensor to fill.
             * @param min Minimum value (default 0.0).
             * @param max Maximum value (default 1.0).
             * @return sycl::event.
             */
            sycl::event uniform(Tensor& t, float min = 0.0f, float max = 1.0f);

            /**
             * @brief Fill tensor with values from a normal (Gaussian) distribution.
             * @param t Tensor to fill.
             * @param mean Mean of the distribution (default 0.0).
             * @param stddev Standard deviation (default 1.0).
             * @return sycl::event.
             */
            sycl::event normal(Tensor& t, float mean = 0.0f, float stddev = 1.0f);

            /**
             * @brief Xavier/Glorot initialization for neural network weights.
             * @param t Tensor to initialize.
             * @param n_in Number of input features.
             * @param n_out Number of output features.
             * @return sycl::event.
             */
            sycl::event xavier(Tensor& t, int64_t n_in, int64_t n_out);

            /**
             * @brief He (Kaiming) initialization for ReLU-based networks.
             * @param t Tensor to initialize.
             * @param n_in Number of input features.
             * @return sycl::event.
             */
            sycl::event he(Tensor& t, int64_t n_in);

        private:
            Engine& engine_;
    };

} // namespace SushiBLAS
