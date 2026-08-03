/**************************************************************************/
/* dispatch.hpp                                                           */
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

// The BLAS member of SushiBLAS's generic op-dispatch seam - see
// <SushiBLAS/engine/dispatch.hpp> for the shared vocabulary and for why the
// family exists at all.
//
// It has exactly one executor, and unlike the five math/logic families the
// functor it takes is not a per-element operation but a GEMM *epilogue*: a
// callable applied to each element of C, with that element's coordinates, in
// the instant before it is stored. The five other families fuse a chain of
// elementwise ops into one launch; this one fuses the elementwise work into the
// GEMM that produced its input, which is one launch fewer still and is where a
// training step actually spends its time.
//
// `Kernels::SYCL::gemm` has carried the `Epilogue` template parameter since
// SB-8, but the kernel header was not installed and `Engine::blas().gemm(...)`
// has no epilogue argument, so nothing outside this library could reach it - a
// fusion hook that exists and is unreachable is a fusion hook that does not
// exist. This header is the reach.

#include <vector>
#include <cstdint>
#include <sycl/sycl.hpp>
#include <SushiBLAS/engine.hpp>
#include <SushiBLAS/tensor.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/core/logger.hpp>
#include <SushiBLAS/engine/dispatch.hpp>
#include <SushiBLAS/engine/blas/utils.hpp>
#include <SushiRuntime/core/resource_region.hpp>
#include <SushiRuntime/graph/task_types.hpp>
#include <SushiBLAS/kernels/blas/level3/gemm_epilogue.hpp>
#include <SushiBLAS/kernels/sycl/blas/level3/gemm.hpp>
#include <SushiBLAS/kernels/reference/blas/level3/gemm.hpp>

namespace SushiBLAS
{
    namespace Dispatch
    {
        /**
         * @brief Records a GEMM with a caller-supplied epilogue fused into its store:
         * `C = epilogue(alpha * op(A) * op(B) + beta * C, row, col)`.
         *
         * The generic seam behind `Level3::gemm`, which is this function with
         * @ref SushiBLAS::Kernels::GemmIdentityEpilogue. Everything `Level3::gemm` does -
         * shape and leading-dimension derivation, batch strides, the reads/writes it
         * declares on the graph, the runtime-dtype switch, the `ExecutionMode` triad -
         * happens here, once, so the fused and unfused paths cannot drift apart.
         *
         * The epilogue is applied to the **fully formed** element, after alpha and after
         * `beta * C`, immediately before the store. Its `row` and `col` arguments are
         * what make it more than an elementwise pass: with them, a bias broadcast is
         * expressible, and a bias-add plus an activation that would otherwise be two more
         * kernel launches over the whole of C become zero.
         *
         * @code
         * // One launch: the product, the row and column biases, and the ReLU.
         * struct BiasRelu
         * {
         *     const float* bias;   // one value per column of C
         *
         *     template<typename T>
         *     T operator()(T value, int64_t row, int64_t col) const
         *     {
         *         const T y = value + static_cast<T>(bias[col]);
         *         return y > T(0) ? y : T(0);
         *     }
         * };
         *
         * Dispatch::execute_gemm(engine, X, W, Y, false, false, 1.0, 0.0,
         *                        "nn.linear_relu", "nn.linear_relu"_op,
         *                        BiasRelu{bias.data_as<float>()});
         * @endcode
         *
         * @tparam Dom Which element types to instantiate @p epilogue for. Defaults to
         *         Domain::REAL, exactly as @ref execute_binary does and for the same
         *         reason: a functor written for real scalars always compiles under it,
         *         while `std::complex` has no ordering, so the `value > 0` at the heart
         *         of every activation is ill-formed for it. Pass
         *         Domain::REAL_AND_COMPLEX only when @p epilogue is genuinely well-formed
         *         for `std::complex` - `Level3::gemm` does, because the identity epilogue
         *         is.
         * @tparam Epilogue The epilogue functor type, deduced. Must provide
         *         `T operator()(T value, int64_t row, int64_t col) const` for every
         *         element type inside @p Dom - a template or generic-lambda call operator
         *         is the usual spelling, since the dtype is a runtime value and every arm
         *         inside the domain is instantiated.
         * @param engine The engine whose task graph the op is recorded on.
         * @param A The left operand.
         * @param B The right operand.
         * @param C The output; must not overlap @p A or @p B.
         * @param transA Whether A is used transposed.
         * @param transB Whether B is used transposed.
         * @param alpha Scale applied to the product.
         * @param beta Scale applied to the incoming C; exactly zero skips reading C.
         * @param name Human-readable op name, used for logging and failure messages.
         * @param op_id Stable operation identifier for the task graph.
         * @param epilogue The per-element epilogue functor.
         * @return An empty sycl::event; the work is deferred until `Engine::execute()`.
         *
         * @warning This is a low-level generic seam, not a convenience wrapper. Same
         *          device-kernel constraints as @ref execute_binary: @p epilogue is
         *          inlined into a SYCL device kernel, so it must be device-copyable and
         *          trivially copyable, must capture by value only, must not dereference
         *          host pointers or touch host-only state, must not allocate, and must
         *          not throw - none of which the compiler will always diagnose for you.
         *          A pointer it carries (a bias vector, say) must be a device-accessible
         *          USM pointer, and it must stay alive until `Engine::execute()` has
         *          completed, since the task holds the functor by value and runs later.
         *
         * @warning **An activation epilogue in a forward *training* pass is silently
         *          wrong** unless it also writes its pre-activation input to an auxiliary
         *          buffer, or is invertible. The backward pass has to differentiate the
         *          activation at its input; if the only thing that reached memory is the
         *          activation's output, that input no longer exists and the gradient is
         *          computed at the wrong point. Nothing throws, nothing produces NaN -
         *          the model simply trains towards the wrong place, which is the worst
         *          failure mode a numerical library has, because there is no symptom to
         *          debug. This is exactly why cuBLASLt exposes `GELU_AUX_BIAS`, which
         *          spills the pre-activation, and not merely `GELU_BIAS`. Inference-only
         *          fusion carries no such constraint, and neither does a bias-only
         *          epilogue, which is invertible. If you fuse an activation into a
         *          training forward pass, the epilogue must write `value` out to an
         *          auxiliary buffer before it returns the activated result.
         *
         * @warning Under `ExecutionMode::FORCE_REFERENCE` and `VERIFY`, @p epilogue is
         *          also invoked on the **host**, by `Kernels::Reference::gemm`. Any
         *          pointer it carries must therefore be host-readable in those modes -
         *          which shared USM is and device USM is not. That is the same constraint
         *          `FORCE_REFERENCE` already places on the operands themselves.
         *
         * @throws std::runtime_error if any operand has rank < 2, if the dtypes disagree,
         *         if the dtype falls outside @p Dom, or if the dtype is one
         *         `execute_blas_task` does not support.
         */
        template<Domain Dom = Domain::REAL, typename Epilogue>
        sycl::event execute_gemm(Engine& engine, const Tensor& A, const Tensor& B, Tensor& C,
                                 bool transA, bool transB, double alpha, double beta,
                                 const char* name, SushiRuntime::Graph::OpID op_id,
                                 Epilogue epilogue)
        {
            SB_THROW_IF(A.rank < 2 || B.rank < 2 || C.rank < 2,
                        "GEMM operation '{}' requires at least 2D tensors.", name);
            SB_THROW_IF(A.dtype != B.dtype || A.dtype != C.dtype,
                        "Data type mismatch in GEMM operation '{}'.", name);

            if constexpr (Dom == Domain::REAL)
            {
                // Rejected here, at record time, rather than at execution time: the
                // caller learns from the call that recorded the mistake instead of from
                // a later Engine::execute() that names a task it did not write.
                SB_THROW_IF(Core::is_complex_dtype(A.dtype),
                            "GEMM operation '{}' is not declared for complex tensors (dtype {}): its "
                            "epilogue was instantiated for real scalars only. Pass "
                            "Domain::REAL_AND_COMPLEX if the epilogue is well-formed for std::complex.",
                            name, Core::to_string(A.dtype));
            }

            const int32_t rA = A.rank, rB = B.rank, rC = C.rank;
            const int64_t m = C.shape[rC - 2];
            const int64_t n = C.shape[rC - 1];
            const int64_t k = transA ? A.shape[rA - 2] : A.shape[rA - 1];

            // Batch & stride calculation.
            int64_t batch_size = 1;
            for (int i = 0; i < rC - 2; ++i) batch_size *= C.shape[i];

            const int64_t lda = (A.layout == Core::Layout::ROW_MAJOR) ? A.shape[rA - 1] : A.shape[rA - 2];
            const int64_t ldb = (B.layout == Core::Layout::ROW_MAJOR) ? B.shape[rB - 1] : B.shape[rB - 2];
            const int64_t ldc = (C.layout == Core::Layout::ROW_MAJOR) ? C.shape[rC - 1] : C.shape[rC - 2];

            const int64_t str_a = (batch_size > 1) ? A.strides[rA - 3] : 0;
            const int64_t str_b = (batch_size > 1) ? B.strides[rB - 3] : 0;
            const int64_t str_c = (batch_size > 1) ? C.strides[rC - 3] : 0;

            // Capture data safely for asynchronous execution.
            const auto layout = A.layout;
            void* read_A = A.data_or_null();
            void* read_B = B.data_or_null();
            void* rw_C = C.data_or_null();

            // `C = alpha * op(A) * op(B) + beta * C` reads C whenever beta is non-zero -
            // the kernels branch around the read on exactly `beta == 0`. Declaring that
            // read makes the access set true rather than merely survivable via the
            // tracker's WAW edge.
            //
            // Note what is NOT declared here: anything the epilogue reads. A bias vector
            // the functor dereferences is invisible to this layer - the functor is an
            // opaque callable - so the caller owns ordering it against whatever produced
            // it. Recording the producer first is not enough; the graph will happily run
            // them concurrently. See the class docs above.
            std::vector<SushiRuntime::Core::ResourceRegion> reads = {};
            if (read_A) reads.push_back(A.region());
            if (read_B) reads.push_back(B.region());
            if (rw_C && beta != 0.0) reads.push_back(C.region());
            std::vector<SushiRuntime::Core::ResourceRegion> writes = {};
            if (rw_C) writes.push_back(C.region());

            SushiRuntime::Graph::TaskMetadata meta;
            meta.name = name;
            meta.task_type = SushiRuntime::Graph::TaskType::MATH_OP;
            meta.op_id = op_id;
            meta.set_param(0, alpha);
            meta.set_param(1, beta);
            meta.set_param(2, transA);
            meta.set_param(3, transB);

            const auto mode = engine.get_mode();

            return Internal::execute_blas_task<Dom>(engine, meta, reads, writes, A.dtype,
                [=](auto scalar_type, sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event
                {
                    using T = decltype(scalar_type);

                    auto pA = static_cast<const T*>(read_A);
                    auto pB = static_cast<const T*>(read_B);
                    auto pC = static_cast<T*>(rw_C);
                    auto alpha_t = static_cast<T>(alpha);
                    auto beta_t = static_cast<T>(beta);

                    return Internal::dispatch_with_mode(mode, q, deps,
                        [=]
                        {
                            Kernels::Reference::gemm<T>(layout, transA, transB, m, n, k, alpha_t,
                                                        pA, lda, str_a, pB, ldb, str_b,
                                                        beta_t, pC, ldc, str_c, batch_size, epilogue);
                        },
                        [=](sycl::queue& q2, const std::vector<sycl::event>& deps2) -> sycl::event
                        {
                            return Kernels::SYCL::gemm<T>(q2, layout, transA, transB, m, n, k, alpha_t,
                                                          pA, lda, str_a, pB, ldb, str_b,
                                                          beta_t, pC, ldc, str_c, batch_size, deps2, epilogue);
                        });
                });
        }

    } // namespace Dispatch
} // namespace SushiBLAS
