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
//
// The epilogue is opaque to this layer on purpose - that is what lets a consumer
// write one without writing SYCL - so the memory it touches is opaque too.
// `Dispatch::GemmEpilogue` is how the caller hands that back, in both directions.

#include <vector>
#include <cstdint>
#include <utility>
#include <type_traits>
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
        // WHY THE ACCESS SET IS BUNDLED WITH THE FUNCTOR RATHER THAN PASSED BESIDE IT
        //
        // The obvious alternative is a twelfth, defaulted parameter on `execute_gemm`
        // carrying the regions. It is the smaller diff, it adds no public type and it
        // needs no deduction layer - and it was rejected for two costs it cannot avoid.
        //
        // The first is that an omitted defaulted argument is invisible. The failure this
        // whole facility exists to close is a *silent* one: an undeclared read is not a
        // wrong dependency edge, it is no edge, so the graph runs the epilogue
        // concurrently with whatever writes its operand and the run merely trains
        // towards the wrong place. Putting the fix behind an argument the caller can
        // simply not write leaves it exactly as easy to omit as it was before the fix
        // existed, and leaves the omission invisible in a diff. Here the operands travel
        // *inside* the epilogue value, so wherever the functor goes - returned from a
        // helper, chosen in a switch, stored and forwarded - its access set goes with
        // it. A separate parameter cannot have that property, because it splits one
        // indivisible fact into two independently-losable ones.
        //
        // The second is that the access set belongs to the *spelling*, not to the call.
        // A caller choosing among several epilogues in a switch (SushiAI's lowering
        // does; a GELU_AUX_BIAS arm would add one whose *write* set differs from every
        // other arm's) would have to hoist the regions out of the arm that knows them
        // and keep the two in step by hand. Bundled, each arm states its own and they
        // cannot drift.
        //
        // What the rejected shape would have bought is given up only partly: a bare
        // functor is still accepted and still means "declares nothing", so every
        // existing call site - `Level3::gemm` with the identity epilogue included -
        // compiles and behaves exactly as before. See @ref Detail::append_epilogue_access.

        /**
         * @class GemmEpilogue
         * @brief A GEMM epilogue functor together with the memory it touches.
         *
         * @ref execute_gemm builds its access set from A, B and C, and it cannot do
         * better on its own: the epilogue is an opaque callable by design - that is what
         * lets a consumer fuse without writing SYCL - so a bias vector the functor
         * dereferences is invisible to this layer. The caller always knows, because the
         * caller built the functor around that pointer. This is where it says so.
         *
         * It carries **both directions**. An epilogue may also write: the pre-activation
         * spill that makes an activation safe to fuse into a *training* forward pass -
         * what cuBLASLt's `GELU_AUX_BIAS` exists to produce, and what the second warning
         * on @ref execute_gemm demands - is an auxiliary write, and whatever reads that
         * buffer afterwards needs an edge for exactly the same reason the bias's producer
         * does.
         *
         * @code
         * // One launch, and the task that writes `bias` is ordered before it.
         * Dispatch::execute_gemm(engine, X, W, Y, false, false, 1.0, 0.0,
         *                        "nn.linear_relu", "nn.linear_relu"_op,
         *                        Dispatch::gemm_epilogue(BiasRelu{bias.data_as<float>()})
         *                            .reads(bias));
         * @endcode
         *
         * **Operands are named as tensors, never as regions.** A `ResourceRegion`
         * converts implicitly from a bare pointer, and a bare pointer widens to the whole
         * allocation of whatever it points at - so two views of one arena become two
         * unrelated resources, `overlaps` says false, and the tracker emits no edge at
         * all. That is precisely the mistake that cost a training run in the reduction
         * seam (see <SushiBLAS/engine/math/reductions/dispatch.hpp>), and taking a
         * `Tensor` makes it unspellable here: the region comes from @ref Tensor::region,
         * once, inside this class. A caller holding memory no `Tensor` describes needs a
         * new overload rather than a way around this one.
         *
         * @tparam Epilogue The epilogue functor type; the same contract @ref execute_gemm
         *         states for a bare one, unchanged. Only the functor reaches the device -
         *         the regions are read off on the host and left behind.
         */
        template<typename Epilogue>
        class GemmEpilogue
        {
            public:
                /**
                 * @brief Wraps @p functor, declaring nothing yet.
                 * @param functor The epilogue functor.
                 */
                explicit GemmEpilogue(Epilogue functor) : functor_(std::move(functor)) {}

                /**
                 * @brief Declares that the epilogue reads @p operand.
                 *
                 * @param operand The tensor the functor dereferences.
                 * @return `*this`, so declarations chain.
                 * @throws std::runtime_error if @p operand has no storage.
                 */
                GemmEpilogue& reads(const Tensor& operand)
                {
                    reads_.push_back(declared_region(operand, "reads"));
                    return *this;
                }

                /**
                 * @brief Declares that the epilogue writes @p operand.
                 *
                 * For an auxiliary buffer the epilogue stores to on its way past - a
                 * pre-activation spill, an index, a running statistic. Whatever reads it
                 * later is then ordered behind this GEMM, and whatever wrote it before is
                 * ordered in front.
                 *
                 * @param operand The tensor the functor stores into.
                 * @return `*this`, so declarations chain.
                 * @throws std::runtime_error if @p operand has no storage.
                 */
                GemmEpilogue& writes(const Tensor& operand)
                {
                    writes_.push_back(declared_region(operand, "writes"));
                    return *this;
                }

                /** @brief The functor itself - the only part that reaches the device. */
                [[nodiscard]] const Epilogue& functor() const noexcept { return functor_; }

                /** @brief The regions declared through @ref reads. */
                [[nodiscard]] const std::vector<SushiRuntime::Core::ResourceRegion>&
                read_regions() const noexcept { return reads_; }

                /** @brief The regions declared through @ref writes. */
                [[nodiscard]] const std::vector<SushiRuntime::Core::ResourceRegion>&
                write_regions() const noexcept { return writes_; }

            private:
                /**
                 * @brief @p operand's region, refusing one that keys on no allocation.
                 *
                 * A storage-less tensor yields a base-less region, which the tracker
                 * would order against nothing - the declaration would compile, read as a
                 * fix, and be a no-op. That is the failure mode this whole class exists
                 * to remove, so it throws instead.
                 *
                 * @param operand The tensor being declared.
                 * @param direction "reads" or "writes", for the message.
                 * @return The operand's byte interval.
                 * @throws std::runtime_error if @p operand has no storage.
                 */
                [[nodiscard]] static SushiRuntime::Core::ResourceRegion
                declared_region(const Tensor& operand, const char* direction)
                {
                    const SushiRuntime::Core::ResourceRegion region = operand.region();
                    SB_THROW_IF(!region.valid(),
                                "A GEMM epilogue declared a tensor it {}, but that tensor has no "
                                "storage. Its region keys on no allocation, so the declaration "
                                "would order it against nothing - which is the silent hazard "
                                "declaring it was meant to close.",
                                direction);
                    return region;
                }

                Epilogue functor_;
                std::vector<SushiRuntime::Core::ResourceRegion> reads_;
                std::vector<SushiRuntime::Core::ResourceRegion> writes_;
        };

        /**
         * @brief Bundles @p functor into a @ref GemmEpilogue whose operands are declared next.
         *
         * The spelling a call site uses, because it deduces @p Epilogue where naming the
         * class would not (C++17 has class template argument deduction, but a function
         * template is what the rest of this seam is written in and what an aggregate
         * epilogue deduces from without a deduction guide).
         *
         * @tparam Epilogue The epilogue functor type, deduced.
         * @param functor The epilogue functor.
         * @return A bundle declaring nothing yet; chain @ref GemmEpilogue::reads and
         *         @ref GemmEpilogue::writes onto it.
         */
        template<typename Epilogue>
        [[nodiscard]] inline GemmEpilogue<Epilogue> gemm_epilogue(Epilogue functor)
        {
            return GemmEpilogue<Epilogue>(std::move(functor));
        }

        namespace Detail
        {
            /**
             * @brief The functor of a bare epilogue: itself.
             * @tparam Epilogue The epilogue functor type.
             * @param epilogue The epilogue.
             * @return @p epilogue.
             */
            template<typename Epilogue>
            [[nodiscard]] inline const Epilogue& epilogue_functor(const Epilogue& epilogue) noexcept
            {
                return epilogue;
            }

            /**
             * @brief The functor inside a @ref GemmEpilogue bundle.
             *
             * More specialised than the overload above, so partial ordering picks it for a
             * bundle and the other one for everything else.
             *
             * @tparam Epilogue The wrapped functor type.
             * @param bundle The bundle.
             * @return The functor it carries.
             */
            template<typename Epilogue>
            [[nodiscard]] inline const Epilogue&
            epilogue_functor(const GemmEpilogue<Epilogue>& bundle) noexcept
            {
                return bundle.functor();
            }

            /**
             * @brief Adds nothing: a bare epilogue declares no operands.
             *
             * This is what keeps "an epilogue that declares nothing behaves exactly as
             * before" a property of the code rather than a claim about it - the access set
             * `execute_gemm` builds is byte for byte the one it built before this facility
             * existed.
             *
             * @tparam Epilogue The epilogue functor type.
             */
            template<typename Epilogue>
            inline void append_epilogue_access(const Epilogue&,
                                               std::vector<SushiRuntime::Core::ResourceRegion>&,
                                               std::vector<SushiRuntime::Core::ResourceRegion>&) noexcept
            {
            }

            /**
             * @brief Appends a bundle's declared operands to a task's access set.
             *
             * @tparam Epilogue The wrapped functor type.
             * @param bundle The bundle whose declarations to copy.
             * @param reads The task's read set, appended to.
             * @param writes The task's write set, appended to.
             */
            template<typename Epilogue>
            inline void append_epilogue_access(const GemmEpilogue<Epilogue>& bundle,
                                               std::vector<SushiRuntime::Core::ResourceRegion>& reads,
                                               std::vector<SushiRuntime::Core::ResourceRegion>& writes)
            {
                reads.insert(reads.end(),
                             bundle.read_regions().begin(), bundle.read_regions().end());
                writes.insert(writes.end(),
                              bundle.write_regions().begin(), bundle.write_regions().end());
            }
        } // namespace Detail

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
         *                        Dispatch::gemm_epilogue(BiasRelu{bias.data_as<float>()})
         *                            .reads(bias));
         * @endcode
         *
         * The `.reads(bias)` is not decoration. Without it the task declares A, B and C
         * and nothing else, `bias` is a resource this GEMM never mentions, and the graph
         * is free to run the task that writes it at the same time. See @ref GemmEpilogue.
         *
         * @tparam Dom Which element types to instantiate @p epilogue for. Defaults to
         *         Domain::REAL, exactly as @ref execute_binary does and for the same
         *         reason: a functor written for real scalars always compiles under it,
         *         while `std::complex` has no ordering, so the `value > 0` at the heart
         *         of every activation is ill-formed for it. Pass
         *         Domain::REAL_AND_COMPLEX only when @p epilogue is genuinely well-formed
         *         for `std::complex` - `Level3::gemm` does, because the identity epilogue
         *         is.
         * @tparam Epilogue The epilogue functor type, deduced - or a
         *         @ref GemmEpilogue wrapping one. The functor must provide
         *         `T operator()(T value, int64_t row, int64_t col) const` for every
         *         element type inside @p Dom - a template or generic-lambda call operator
         *         is the usual spelling, since the dtype is a runtime value and every arm
         *         inside the domain is instantiated. A bare functor declares no operands
         *         of its own and produces exactly the access set this seam has always
         *         built; a @ref GemmEpilogue adds whatever the caller declared on it.
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
         * @param epilogue The per-element epilogue functor, bare or bundled with its
         *        operands by @ref gemm_epilogue.
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
         *          Staying alive is not the same as being *ordered*: declare the memory
         *          through @ref gemm_epilogue as well, or nothing sequences this task
         *          against whatever else touches it.
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
         *          auxiliary buffer before it returns the activated result - and it must
         *          declare that buffer through @ref GemmEpilogue::writes, or the backward
         *          pass that reads it races the GEMM that filled it and the spill buys
         *          nothing.
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
            std::vector<SushiRuntime::Core::ResourceRegion> reads = {};
            if (read_A) reads.push_back(A.region());
            if (read_B) reads.push_back(B.region());
            if (rw_C && beta != 0.0) reads.push_back(C.region());
            std::vector<SushiRuntime::Core::ResourceRegion> writes = {};
            if (rw_C) writes.push_back(C.region());

            // Anything the epilogue itself touches. This layer still cannot discover it -
            // the functor is an opaque callable and that is the point of it - but the
            // caller can state it, and @ref GemmEpilogue is how. A bare functor appends
            // nothing, so the access set above is the whole of it, exactly as before.
            Detail::append_epilogue_access(epilogue, reads, writes);

            // The functor alone, never the bundle. A GemmEpilogue owns std::vectors and
            // the closure below is copied into a SYCL kernel, where a std::vector is not
            // device-copyable - so the access set is read off here, on the host, and left
            // behind. Naming this instead of `epilogue` inside the closure is what keeps
            // it out.
            const auto kernel_epilogue = Detail::epilogue_functor(epilogue);

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
                                                        beta_t, pC, ldc, str_c, batch_size, kernel_epilogue);
                        },
                        [=](sycl::queue& q2, const std::vector<sycl::event>& deps2) -> sycl::event
                        {
                            return Kernels::SYCL::gemm<T>(q2, layout, transA, transB, m, n, k, alpha_t,
                                                          pA, lda, str_a, pB, ldb, str_b,
                                                          beta_t, pC, ldc, str_c, batch_size, deps2, kernel_epilogue);
                        });
                });
        }

    } // namespace Dispatch
} // namespace SushiBLAS
