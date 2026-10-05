/****************************************************************/
/* reductions.hpp                                               */
/* SushiBLAS - https://github.com/SushiSystems/SushiBLAS        */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

/**
 * @file reductions.hpp
 * @brief Declares ReductionOps, the whole-tensor and per-axis reductions.
 * @author Mustafa Garip
 */

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
             * @brief Computes the maximum along one axis of a rank-2 tensor.
             * @param axis 0 for a per-column maximum, 1 for a per-row maximum.
             * @param result Rank-1 tensor of the surviving length, sharing t's dtype.
             * @throws std::runtime_error on a rank, axis, length or dtype mismatch.
             * @see include/SushiBLAS/engine/math/README.md
             */
            sycl::event max_along(const Tensor& t, int axis, Tensor& result);

            /**
             * @brief Computes the Neumaier-compensated sum along one axis of a rank-2 tensor.
             * @param axis 0 for column sums, 1 for row sums.
             * @param result Rank-1 tensor of the surviving length, sharing t's dtype.
             * @throws std::runtime_error on a rank, axis, length or dtype mismatch.
             * @see include/SushiBLAS/engine/math/README.md
             */
            sycl::event sum_along(const Tensor& t, int axis, Tensor& result);

            /**
             * @brief Computes the mean over the two spatial axes of an NHWC tensor: `[N, H, W, C] -> [N, C]`.
             * @param t Must be rank 4, NHWC, with rows contiguous with one another.
             * @param result Rank-2 `[N, C]` tensor sharing t's dtype.
             * @throws std::runtime_error on a rank, shape, stride or dtype mismatch.
             * @see include/SushiBLAS/engine/math/README.md
             */
            sycl::event mean_spatial(const Tensor& t, Tensor& result);

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

            /**
             * @brief Counts the elements of t that are infinite or NaN into an INT64 scalar.
             * @param t Must be HALF, FLOAT32 or FLOAT64.
             * @param result Must hold exactly one element of dtype INT64.
             * @throws std::runtime_error if result is not an INT64 scalar or t's dtype is not real.
             * @see include/SushiBLAS/engine/math/README.md
             */
            sycl::event count_nonfinite(const Tensor& t, Tensor& result);

        private:
            Engine& engine_;
    };

} // namespace SushiBLAS
