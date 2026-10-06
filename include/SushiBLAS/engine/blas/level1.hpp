/**************************************************************************/
/* level1.hpp                                                             */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                SushiBLAS                               */
/*                https://github.com/SushiSystems/SushiBLAS               */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present Mustafa Garip & Sushi Systems               */
/*                                                                        */
/* Licensed under the PolyForm Noncommercial License 1.0.0 (the           */
/* "License"); you may not use this file except in compliance with the    */
/* License. You may obtain a copy of the License at                       */
/*                                                                        */
/*     https://polyformproject.org/licenses/noncommercial/1.0.0           */
/*                                                                        */
/* Noncommercial use is free. Commercial use requires a separate licence  */
/* from Sushi Systems; see COMMERCIAL.md. The software is provided        */
/* "as is", without warranty of any kind.                                 */
/**************************************************************************/

/**
 * @file level1.hpp
 * @brief Declares Level1, the vector-vector BLAS operations.
 * @author Mustafa Garip
 */

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
             * @brief Computes y = alpha * x + y.
             * @param alpha Taken as double and narrowed to the element type at the kernel boundary.
             * @see include/SushiBLAS/engine/blas/README.md
             */
            sycl::event axpy(double alpha, const Tensor& x, Tensor& y);

            /**
             * @brief Computes the unconjugated dot product x^T * y into result.
             * @param result Scalar output tensor.
             */
            sycl::event dot(const Tensor& x, const Tensor& y, Tensor& result);

            /**
             * @brief Scales every element of x by alpha.
             * @param alpha Taken as double and narrowed at the kernel boundary; see axpy.
             */
            sycl::event scal(double alpha, Tensor& x);

            /**
             * @brief Copies every element of x into y.
             */
            sycl::event copy(const Tensor& x, Tensor& y);

            /**
             * @brief Exchanges the elements of x and y.
             */
            sycl::event swap(Tensor& x, Tensor& y);

            /**
             * @brief Computes the Euclidean norm sqrt(sum(|x_i|^2)) into result.
             * @param result Scalar output tensor.
             * @see include/SushiBLAS/engine/blas/README.md
             */
            sycl::event nrm2(const Tensor& x, Tensor& result);

            /**
             * @brief Computes sum(|Re(x_i)| + |Im(x_i)|) into result.
             * @param result Scalar output tensor.
             */
            sycl::event asum(const Tensor& x, Tensor& result);

            /**
             * @brief Finds the smallest index i at which |Re(x_i)| + |Im(x_i)| is largest.
             * @see include/SushiBLAS/engine/blas/README.md
             */
            sycl::event iamax(const Tensor& x, Tensor& result);

            /**
             * @brief Applies the plane rotation x_i = c * x_i + s * y_i, y_i = -s * x_i + c * y_i.
             * @param c Taken as double, like s, and narrowed at the kernel boundary; see axpy.
             */
            sycl::event rot(Tensor& x, Tensor& y, double c, double s);
    };
}
