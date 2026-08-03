/**************************************************************************/
/* reductions.hpp                                                         */
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
     * @class ReductionOps
     * @brief Parallel tensor reduction operations.
     * 
     * These operations aggregate many elements into a single scalar value. 
     * All results are returned as SYCL events as they are computed asynchronously 
     * and stored in the provided scalar tensor.
     */
    class ReductionOps 
    {
        public:
            /**
             * @brief Construct ReductionOps with a reference to the engine.
             * @param e The SushiBLAS engine.
             */
            explicit ReductionOps(Engine& e) : engine_(e) {}

            /** 
             * @brief Sum of all elements.
             * Computes result = sum(t_i).
             * @param t Input tensor.
             * @param result Scalar output tensor.
             * @return sycl::event representing task completion.
             */
            sycl::event sum(const Tensor& t, Tensor& result);

            /** 
             * @brief Mean of all elements.
             * Computes result = sum(t_i) / num_elements.
             * @param t Input tensor.
             * @param result Scalar output tensor.
             * @return sycl::event.
             */
            sycl::event mean(const Tensor& t, Tensor& result);

            /** 
             * @brief Maximum value in the tensor.
             * Finds result = max(t_i).
             * @param t Input tensor.
             * @param result Scalar output tensor.
             * @return sycl::event.
             */
            sycl::event max(const Tensor& t, Tensor& result);

            /** 
             * @brief Minimum value in the tensor.
             * Finds result = min(t_i).
             * @param t Input tensor.
             * @param result Scalar output tensor.
             * @return sycl::event.
             */
            sycl::event min(const Tensor& t, Tensor& result);

            /** 
             * @brief Index of the maximum value.
             * @param t Input tensor.
             * @param result Scalar output tensor (stores index as int64_t/float).
             * @return sycl::event.
             */
            sycl::event argmax(const Tensor& t, Tensor& result);

            /** 
             * @brief Index of the minimum value.
             * @param t Input tensor.
             * @param result Scalar output tensor.
             * @return sycl::event.
             */
            sycl::event argmin(const Tensor& t, Tensor& result);

            /** 
             * @brief Variance of all elements.
             * @param t Input tensor.
             * @param result Scalar output tensor.
             * @return sycl::event.
             */
            sycl::event var(const Tensor& t, Tensor& result);

            /** 
             * @brief Standard Deviation of all elements.
             * @param t Input tensor.
             * @param result Scalar output tensor.
             * @return sycl::event.
             */
            sycl::event std(const Tensor& t, Tensor& result);

            /** 
             * @brief Product of all elements.
             * @param t Input tensor.
             * @param result Scalar output tensor.
             * @return sycl::event.
             */
            sycl::event prod(const Tensor& t, Tensor& result);

            /** 
             * @brief Log-Sum-Exp reduction.
             * Computes result = log(sum(exp(t_i - max(t)))).
             * Used for numerical stability in softmax calculations.
             * @param t Input tensor.
             * @param result Scalar output tensor.
             * @return sycl::event.
             */
            sycl::event logsumexp(const Tensor& t, Tensor& result);

            /**
             * @brief Maximum along one axis of a rank-2 tensor.
             *
             * The axis-wise counterpart of @ref max: instead of one scalar for the
             * whole tensor, one value per surviving position. `axis == 1` folds each
             * row to its own maximum, which is the shift a per-row log-sum-exp needs
             * - shifting a batch by its *global* maximum is exact in real arithmetic
             * and returns `+inf` in binary32 for any row about 104 below it, because
             * every exponential in that row underflows and its row sum becomes
             * exactly zero.
             *
             * Deterministic by construction, like every other folding reduction here;
             * see <SushiBLAS/engine/reduce_axis.hpp>.
             *
             * @param t Input tensor; must be rank 2.
             * @param axis Which axis to fold away: 0 for a per-column maximum (result
             *        length `t.shape[1]`), 1 for a per-row maximum (result length
             *        `t.shape[0]`).
             * @param result Rank-1 output tensor of the surviving length, sharing
             *        `t`'s dtype.
             * @return sycl::event.
             * @throws std::runtime_error on a rank, axis, length or dtype mismatch.
             */
            sycl::event max_along(const Tensor& t, int axis, Tensor& result);

            /**
             * @brief Sum along one axis of a rank-2 tensor.
             *
             * The axis-wise counterpart of @ref sum, and compensated the same way:
             * each segment is folded with Neumaier's correction in an order fixed by
             * the launch geometry, so the answer is both accurate and reproducible to
             * the bit.
             *
             * @param t Input tensor; must be rank 2.
             * @param axis Which axis to fold away: 0 for column sums (result length
             *        `t.shape[1]`), 1 for row sums (result length `t.shape[0]`).
             * @param result Rank-1 output tensor of the surviving length, sharing
             *        `t`'s dtype.
             * @return sycl::event.
             * @throws std::runtime_error on a rank, axis, length or dtype mismatch.
             */
            sycl::event sum_along(const Tensor& t, int axis, Tensor& result);

            /**
             * @brief L1 Norm (Sum of absolute values).
             * @param t Input tensor.
             * @param result Scalar output tensor.
             * @return sycl::event.
             */
            sycl::event asum(const Tensor& t, Tensor& result);

            /** 
             * @brief L2 Norm (Euclidean norm).
             * @param t Input tensor.
             * @param result Scalar output tensor.
             * @return sycl::event.
             */
            sycl::event nrm2(const Tensor& t, Tensor& result);

        private:
            Engine& engine_;
    };

} // namespace SushiBLAS
