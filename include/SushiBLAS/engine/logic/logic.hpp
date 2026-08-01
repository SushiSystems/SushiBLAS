/**************************************************************************/
/* logic.hpp                                                              */
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

#include <sycl/sycl.hpp>
#include <SushiBLAS/tensor.hpp>

namespace SushiBLAS 
{
    class Engine;

    /**
     * @class LogicOps
     * @brief Logical and comparison operations for tensors.
     * 
     * These operations are essential for masking, filtering, and 
     * implementing conditional logic in computational graphs.
     */
    class LogicOps 
    {
        public:
            explicit LogicOps(Engine& e) : engine_(e) {}

            /**
             * @brief Element-wise equality comparison: A == B.
             * @param A First input tensor.
             * @param B Second input tensor.
             * @param result Output tensor (1 if true, 0 if false).
             * @return sycl::event representing the operation.
             */
            sycl::event equal(const Tensor& A, const Tensor& B, Tensor& result);

            /**
             * @brief Element-wise inequality comparison: A != B.
             * @param A First input tensor.
             * @param B Second input tensor.
             * @param result Output tensor (1 if true, 0 if false).
             * @return sycl::event representing the operation.
             */
            sycl::event not_equal(const Tensor& A, const Tensor& B, Tensor& result);

            /**
             * @brief Element-wise comparison: A > B.
             * @param A First input tensor.
             * @param B Second input tensor.
             * @param result Output tensor (1 if true, 0 if false).
             * @return sycl::event representing the operation.
             */
            sycl::event greater(const Tensor& A, const Tensor& B, Tensor& result);

            /**
             * @brief Element-wise comparison: A < B.
             * @param A First input tensor.
             * @param B Second input tensor.
             * @param result Output tensor (1 if true, 0 if false).
             * @return sycl::event representing the operation.
             */
            sycl::event less(const Tensor& A, const Tensor& B, Tensor& result);

            /**
             * @brief Element-wise comparison: A >= B.
             * @param A First input tensor.
             * @param B Second input tensor.
             * @param result Output tensor (1 if true, 0 if false).
             * @return sycl::event representing the operation.
             */
            sycl::event greater_equal(const Tensor& A, const Tensor& B, Tensor& result);

            /**
             * @brief Element-wise comparison: A <= B.
             * @param A First input tensor.
             * @param B Second input tensor.
             * @param result Output tensor (1 if true, 0 if false).
             * @return sycl::event representing the operation.
             */
            sycl::event less_equal(const Tensor& A, const Tensor& B, Tensor& result);

            /**
             * @brief Conditional selection (Like numpy.where).
             * Computes: out = condition ? A : B
             * @param condition Mask tensor (non-zero treated as true).
             * @param A Selection if true.
             * @param B Selection if false.
             * @param out Output tensor.
             * @return sycl::event representing the operation.
             */
            sycl::event where(const Tensor& condition, const Tensor& A, const Tensor& B, Tensor& out);

            /**
             * @brief Element-wise logical AND of two tensors.
             * @param A Input tensor A.
             * @param B Input tensor B.
             * @param out Output tensor (1 if both non-zero, else 0).
             * @return sycl::event representing the operation.
             */
            sycl::event logical_and(const Tensor& A, const Tensor& B, Tensor& out);

            /**
             * @brief Element-wise logical OR of two tensors.
             * @param A Input tensor A.
             * @param B Input tensor B.
             * @param out Output tensor (1 if either non-zero, else 0).
             * @return sycl::event representing the operation.
             */
            sycl::event logical_or(const Tensor& A, const Tensor& B, Tensor& out);

            /**
             * @brief Element-wise logical XOR of two tensors.
             * @param A Input tensor A.
             * @param B Input tensor B.
             * @param out Output tensor (1 if exactly one is non-zero, else 0).
             * @return sycl::event representing the operation.
             */
            sycl::event logical_xor(const Tensor& A, const Tensor& B, Tensor& out);

            /**
             * @brief Element-wise logical NOT of a tensor.
             * @param A Input tensor.
             * @param out Output tensor (1 if zero, 0 if non-zero).
             * @return sycl::event representing the operation.
             */
            sycl::event logical_not(const Tensor& A, Tensor& out);

            /**
             * @brief Check if all elements are non-zero (Logical ALL reduction).
             * @param t Input tensor.
             * @param result Scalar output (1 if all true, else 0).
             * @return sycl::event representing the operation.
             */
            sycl::event all(const Tensor& t, Tensor& result);

            /**
             * @brief Check if any element is non-zero (Logical ANY reduction).
             * @param t Input tensor.
             * @param result Scalar output (1 if any true, else 0).
             * @return sycl::event representing the operation.
             */
            sycl::event any(const Tensor& t, Tensor& result);

        private:
            Engine& engine_;
    };

} // namespace SushiBLAS
