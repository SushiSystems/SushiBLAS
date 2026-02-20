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

#include <SushiBLAS/tensor.hpp>

namespace SushiBLAS 
{
    class Engine;

    /**
     * @brief Vector-Vector operations.
     */
    class Level1 
    {
        protected:
            explicit Level1(Engine& e) : engine_(e) {}
            Engine& engine_;

        public:
            /** @brief y = alpha*x + y 
            
            @param alpha Scalar multiplier
            @param x Input vector
            @param y Input/output vector
            */
            void axpy(float alpha, const Tensor& x, Tensor& y);
            
            /** @brief dot = x^T * y 
            
            @param x Input vector
            @param y Input vector
            @return dot product of x and y
            */
            float dot(const Tensor& x, const Tensor& y);

            /** @brief x = alpha*x 
            
            @param alpha Scalar multiplier
            @param x Input/output vector
            */
            void scal(float alpha, Tensor& x);

            /** @brief returns the euclidean norm of a vector (L2 norm)
            
            @param x Input vector
            @return Euclidean norm of x
            */
            float nrm2(const Tensor& x);

            /** @brief returns the sum of absolute values (L1 norm)
            
            @param x Input vector
            @return L1 norm of x
            */
            float asum(const Tensor& x);

            /** @brief returns the index of the element with the maximum absolute value
            
            @param x Input vector
            @return 0-based index of the maximum absolute value
            */
            int64_t iamax(const Tensor& x);

            /** @brief y = x (copy vector)
            
            @param x Source vector
            @param y Destination vector
            */
            void copy(const Tensor& x, Tensor& y);

            /** @brief swap x and y
            
            @param x Vector to be interchanged
            @param y Vector to be interchanged
            */
            void swap(Tensor& x, Tensor& y);

            /** @brief Performs Givens rotation of points in the plane
            
            @param x Input vector
            @param y Input vector
            @param c Cosine component
            @param s Sine component
            */
            void rot(Tensor& x, Tensor& y, float c, float s);

    };
}
