/**************************************************************************/
/* level1.hpp                                                             */
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
     * @class Level1
     * @brief Vector-Vector operations (BLAS Level 1).
     * 
     * This class provides an interface for standard BLAS Level 1 operations, 
     * which involve scalar-vector and vector-vector computations. All operations 
     * are executed asynchronously using the SushiBLAS task graph system.
     */
    class Level1 
    {
        protected:
            explicit Level1(Engine& e) : engine_(e) {}
            Engine& engine_;

        public:
            /**
             * @brief Vector-Scalar Product and Addition (AXPY).
             * 
             * Computes the operation: y = alpha * x + y.
             * This is a fundamental linear algebra operation that scales vector x 
             * by scalar alpha and adds it to vector y.
             * 
             * @param alpha Scalar multiplier.
             * @param x Input vector x.
             * @param y Input/Output vector y (result is stored in y).
             * @return sycl::event representing the completion of the operation.
             */
            sycl::event axpy(float alpha, const Tensor& x, Tensor& y);

            /**
             * @brief Vector Dot Product (DOT).
             * 
             * Computes the dot product of two vectors: result = x^T * y.
             * For complex vectors, this computes the unconjugated dot product.
             * For complex conjugate dot product, see DOTC.
             * 
             * @param x Input vector x.
             * @param y Input vector y.
             * @param result Scalar output tensor where the result will be stored.
             * @return sycl::event representing the completion of the operation.
             */
            sycl::event dot(const Tensor& x, const Tensor& y, Tensor& result);

            /**
             * @brief Vector Scaling (SCAL).
             * 
             * Computes the operation: x = alpha * x.
             * Scales every element of vector x by the scalar factor alpha.
             * 
             * @param alpha Scalar factor.
             * @param x Input/Output vector to scale.
             * @return sycl::event representing the completion of the operation.
             */
            sycl::event scal(float alpha, Tensor& x);

            /**
             * @brief Vector Copy (COPY).
             * 
             * Copies all elements from vector x to vector y: y = x.
             * 
             * @param x Source vector.
             * @param y Destination vector.
             * @return sycl::event representing the completion of the operation.
             */
            sycl::event copy(const Tensor& x, Tensor& y);

            /**
             * @brief Vector Swap (SWAP).
             * 
             * Interchange the elements of vector x and vector y.
             * 
             * @param x First vector.
             * @param y Second vector.
             * @return sycl::event representing the completion of the operation.
             */
            sycl::event swap(Tensor& x, Tensor& y);

            /**
             * @brief Euclidean Norm (NRM2).
             * 
             * Computes the L2 norm (Euclidean norm) of a vector: result = sqrt(sum(|x_i|^2)).
             * This operation is more robust against overflow/underflow than a direct implementation.
             * 
             * @param x Input vector.
             * @param result Scalar output tensor where the norm will be stored.
             * @return sycl::event representing the completion of the operation.
             */
            sycl::event nrm2(const Tensor& x, Tensor& result);

            /**
             * @brief Sum of Absolute Values (ASUM).
             * 
             * Computes the L1 norm of a vector: result = sum(|Re(x_i)| + |Im(x_i)|).
             * For real vectors, this is simply the sum of absolute values.
             * 
             * @param x Input vector.
             * @param result Scalar output tensor where the sum will be stored.
             * @return sycl::event representing the completion of the operation.
             */
            sycl::event asum(const Tensor& x, Tensor& result);

            /**
             * @brief Index of Absolute Maximum (IAMAX).
             * 
             * Finds the smallest index i such that |Re(x_i)| + |Im(x_i)| is maximal.
             * This is often used for pivoting in matrix factorizations.
             * 
             * @param x Input vector.
             * @param result Scalar output tensor (stores the 1-based or 0-based index depending on implementation).
             * @return sycl::event representing the completion of the operation.
             */
            sycl::event iamax(const Tensor& x, Tensor& result);

            /**
             * @brief Givens Rotation (ROT).
             * 
             * Applies a plane rotation to points in vectors x and y:
             * x_i = c * x_i + s * y_i
             * y_i = -s * x_i + c * y_i
             * 
             * @param x Vector x.
             * @param y Vector y.
             * @param c Cosine component of the rotation.
             * @param s Sine component of the rotation.
             * @return sycl::event representing the completion of the operation.
             */
            sycl::event rot(Tensor& x, Tensor& y, float c, float s);
    };
}
