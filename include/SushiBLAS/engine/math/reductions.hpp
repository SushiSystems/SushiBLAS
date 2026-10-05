/****************************************************************/
/* reductions.hpp                                               */
/* SushiBLAS - https://github.com/SushiSystems/SushiBLAS        */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

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
             * @brief Mean over the two spatial axes of an NHWC tensor:
             * `[N, H, W, C] -> [N, C]`.
             *
             * The per-channel, per-sample average a normalisation layer and a
             * global-average-pool both want, in one launch. It is not @ref mean of a
             * reshaped view and it is not @ref sum_along of one either: the surviving
             * position is a *pair* `(n, c)`, so the input offset of a fold is
             * `n * strides[0] + c * strides[3]` and no single segment stride produces
             * that from one index. Folding both spatial axes as one run of `H * W`
             * elements is the easy half; the pair is the reason this exists.
             *
             * Neumaier-compensated and deterministic like every other folding
             * reduction here - the division is the finalizer, so it costs the same
             * launch as the sum would; see <SushiBLAS/engine/reduce_axis.hpp>.
             *
             * @param t Input tensor; must be rank 4 and NHWC, and its rows must be
             *        contiguous with one another (a row-sliced view is rejected by
             *        name rather than read past).
             * @param result Rank-2 `[N, C]` output tensor sharing `t`'s dtype.
             * @return sycl::event.
             * @throws std::runtime_error on a rank, shape, stride or dtype mismatch.
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
             * @brief How many elements of @p t are infinite or NaN.
             *
             * The screen a loss-scaling step runs before it applies an update: if the
             * answer is not zero the gradients overflowed at the current scale, so the
             * step is skipped and the scale halved. `count_nonfinite(t, r) == 0` is the
             * boolean "all finite"; the count itself is what says how badly, which is
             * strictly more than the boolean and costs the same fold.
             *
             * **Both** kinds of non-finite value, and they need separate tests: `x != x`
             * is true only of a NaN, and an infinity compares equal to itself perfectly
             * happily. Overflow in fp16 produces infinities and division by an overflowed
             * quantity produces NaNs, so a screen that caught one and not the other would
             * pass exactly half the runs it exists to stop.
             *
             * **The result is an INT64 scalar**, and the reason is the consumer rather
             * than tidiness: the caller reads this value from a *device* kernel with no
             * host synchronisation, because a sync inside the optimiser step would drain
             * the graph the whole architecture exists to keep full. So the answer has to
             * be a device-resident value a kernel can load and branch on, which an INT64
             * tensor element is. It must also be **exact**, which is what rules out
             * putting the count in a tensor of the input's own dtype: a count is exact in
             * FLOAT32 only below 2^24 and in HALF only below 2048, and a HALF count past
             * 65504 is an infinity - so the tensors most worth counting are precisely the
             * ones whose count would come back wrong. That is the same trap
             * `Dispatch::execute_index_reduction` was cut out of `execute_reduction` to
             * escape, arriving by a different route.
             *
             * **Why not `asum`.** It nearly works - `sum(|x|)` is non-finite whenever any
             * element is, so a caller could have tested `isfinite(asum(g))` and shipped
             * nothing new. It is rejected for one decisive reason and two supporting ones.
             * The decisive one: `asum` accumulates in the *input's* dtype, so on a HALF
             * tensor the running total can leave binary16's range all by itself. Eight
             * elements of 20000 - each exactly representable, not one of them remotely bad
             * - give a non-finite total, at which point the screen reports an overflow that
             * never happened, the scale is halved, and it is halved again on the next step,
             * for ever, on a run that was fine. The op meant to rescue a mixed-precision
             * run would be the thing that stalls it.
             * `Unit_CountNonFiniteTest.AsumReportsOverflowOnAHalfTensorThisOpCallsClean` is
             * that exact tensor put through both ops, so this is a measurement rather than
             * a worry. The supporting ones:
             * even in fp32 a loss-scaled gradient is *deliberately* pushed up in magnitude,
             * which is the one situation where a sum of magnitudes overflows without any
             * element doing so, and `asum` therefore conflates "an element is bad" with
             * "the total is big"; and its answer is a float in the input's dtype, so a
             * consumer branching on it does floating-point comparisons where an integer
             * `!= 0` would do. A predicate count cannot produce a false positive, and that
             * is the property that matters for something whose job is to decide whether to
             * throw a training step away.
             *
             * **Deterministic**, like every folding reduction here - and for a stronger
             * reason. The fold is `Reduce::ExactSum` over integers, which is exactly
             * associative and exactly commutative, so no combination order can change the
             * answer at all; the fixed-order shape of
             * <SushiBLAS/engine/reduce_1d.hpp> is inherited from the shared seam rather
             * than being what makes this op reproducible.
             *
             * @param t Input tensor; HALF, FLOAT32 or FLOAT64. The integer and complex
             *        dtypes throw by name rather than answering: an integer element cannot
             *        be infinite or NaN, so a confident count of zero would tell a caller
             *        their tensor had been screened when the question was never asked of
             *        it.
             * @param result Scalar output tensor; must be rank-anything with exactly one
             *        element and dtype INT64.
             * @return sycl::event.
             * @throws std::runtime_error if @p result is not a scalar or not INT64, or if
             *         @p t's dtype is not one of HALF / FLOAT32 / FLOAT64.
             */
            sycl::event count_nonfinite(const Tensor& t, Tensor& result);

        private:
            Engine& engine_;
    };

} // namespace SushiBLAS
