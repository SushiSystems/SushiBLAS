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
/* Licensed under the Apache License, Version 2.0 (the "License"); you    */
/* may not use this file except in compliance with the License. You may   */
/* obtain a copy of the License at                                        */
/*                                                                        */
/*     https://www.apache.org/licenses/LICENSE-2.0                        */
/*                                                                        */
/* Unless required by applicable law or agreed to in writing, software    */
/* distributed under the License is distributed on an "AS IS" BASIS,      */
/* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        */
/* implied. See the License for the specific language governing           */
/* permissions and limitations under the License.                         */
/**************************************************************************/

/**
 * @file dispatch.hpp
 * @brief Declares the GEMM executor of the dispatch seam and the epilogue bundle it accepts.
 * @author Mustafa Garip
 */

#pragma once

#include <vector>
#include <cstdint>
#include <utility>
#include <type_traits>
#include <sycl/sycl.hpp>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/core/blas_types.hpp>
#include <SushiBLAS/engine/dispatch.hpp>
#include <SushiBLAS/engine/operand_set.hpp>
#include <SushiBLAS/engine/operand_check.hpp>
#include <SushiBLAS/engine/task_recorder.hpp>
#include <SushiBLAS/blas/blas_task.hpp>
#include <SushiBLAS/blas/gemm_recording.hpp>
#include <SushiBLAS/operands/scalar.hpp>
#include <SushiRuntime/region/buffer_region.hpp>
#include <SushiRuntime/core/operation_id.hpp>
#include <SushiBLAS/blas_seam_kernels/level3/gemm_epilogue.hpp>
#include <SushiBLAS/blas_seam_kernels/sycl/level3/gemm.hpp>
#include <SushiBLAS/blas_seam_kernels/sycl/level3/gemm_packed.hpp>
#include <SushiBLAS/blas_seam_kernels/reference/level3/gemm.hpp>

namespace SushiBLAS
{
    namespace Dispatch
    {
        /**
         * @brief Holds a GEMM epilogue functor together with the tensors it reads and writes.
         * @tparam Epilogue The functor type, under the contract execute_gemm states for a bare one.
         * @see modules/operations/blas/README.md
         */
        template<typename Epilogue>
        class GemmEpilogue
        {
            public:
                /** @brief Wraps functor, declaring nothing yet. */
                explicit GemmEpilogue(Epilogue functor)
                    : functor_(std::move(functor))
                {

                }

                /**
                 * @brief Declares that the epilogue reads operand.
                 * @return `*this`, so declarations chain.
                 * @throws std::invalid_argument if operand has no storage.
                 */
                GemmEpilogue& reads(const Tensor& operand)
                {
                    require_declared_storage(operand, "reads");
                    operands_.reads(operand);
                    return *this;
                }

                /**
                 * @brief Declares that the epilogue writes operand.
                 * @return `*this`, so declarations chain.
                 * @throws std::invalid_argument if operand has no storage.
                 */
                GemmEpilogue& writes(const Tensor& operand)
                {
                    require_declared_storage(operand, "writes");
                    operands_.writes(operand);
                    return *this;
                }

                /** @brief Returns the functor, the only part that reaches the device. */
                [[nodiscard]] const Epilogue& functor() const noexcept
                {
                    return functor_;
                }

                /** @brief Returns the tensors declared through reads and writes. */
                [[nodiscard]] const OperandSet& operands() const noexcept
                {
                    return operands_;
                }

                /** @brief Returns the regions declared through reads. */
                [[nodiscard]] const std::vector<SushiRuntime::BufferRegion>&
                read_regions() const noexcept
                {
                    return operands_.read_regions();
                }

                /** @brief Returns the regions declared through writes. */
                [[nodiscard]] const std::vector<SushiRuntime::BufferRegion>&
                write_regions() const noexcept
                {
                    return operands_.write_regions();
                }

            private:
                /**
                 * @brief Throws if a declared tensor has no storage, so its region keys on nothing.
                 * @param direction "reads" or "writes", for the message.
                 */
                static void require_declared_storage(const Tensor& operand, const char* direction)
                {
                    if (!operand.storage)
                        Detail::throw_operand_error("Dispatch::GemmEpilogue", direction,
                                                    "was given a tensor without storage; its "
                                                    "region keys on no allocation, so the "
                                                    "declaration would order it against nothing");
                }

                /// @brief The epilogue functor.
                Epilogue functor_;

                /// @brief The tensors the functor dereferences, with their storage retained.
                OperandSet operands_;
        };

        /**
         * @brief Bundles functor into a GemmEpilogue that declares nothing yet.
         * @see modules/operations/blas/README.md
         */
        template<typename Epilogue>
        [[nodiscard]] inline GemmEpilogue<Epilogue> gemm_epilogue(Epilogue functor)
        {
            return GemmEpilogue<Epilogue>(std::move(functor));
        }

        namespace Detail
        {
            /**
             * @brief Holds the floating-point operations a GEMM epilogue does on each element of C.
             *
             * An epilogue states it as `static constexpr std::uint32_t K_FLOPS_PER_ELEMENT`; one
             * that states none counts 0, which is the identity epilogue's cost and a lower bound
             * for any other.
             * @see modules/foundation/cost/README.md
             */
            template<typename Epilogue, typename = void>
            struct EpilogueFlops
            {
                /// @brief The count an epilogue with no stated cost takes.
                static constexpr std::uint32_t value = 0;
            };

            /** @brief Reads the count an epilogue states. */
            template<typename Epilogue>
            struct EpilogueFlops<Epilogue, std::void_t<decltype(Epilogue::K_FLOPS_PER_ELEMENT)>>
            {
                /// @brief The count the epilogue states.
                static constexpr std::uint32_t value = Epilogue::K_FLOPS_PER_ELEMENT;
            };

            /** @brief Reads the count of the functor a bundle carries. */
            template<typename Epilogue>
            struct EpilogueFlops<GemmEpilogue<Epilogue>> : EpilogueFlops<Epilogue>
            {
            };

            /** @brief Returns the functor of a bare epilogue: itself. */
            template<typename Epilogue>
            [[nodiscard]] inline const Epilogue& epilogue_functor(const Epilogue& epilogue) noexcept
            {
                return epilogue;
            }

            /** @brief Returns the functor a GemmEpilogue bundle carries. */
            template<typename Epilogue>
            [[nodiscard]] inline const Epilogue&
            epilogue_functor(const GemmEpilogue<Epilogue>& bundle) noexcept
            {
                return bundle.functor();
            }

            /** @brief Adds nothing, because a bare epilogue declares no operands. */
            template<typename Epilogue>
            inline void append_epilogue_access(const Epilogue&, OperandSet&) noexcept
            {

            }

            /** @brief Adds a bundle's declared operands to a task's operand set. */
            template<typename Epilogue>
            inline void append_epilogue_access(const GemmEpilogue<Epilogue>& bundle,
                                               OperandSet& operands)
            {
                operands.merge(bundle.operands());
            }
        } // namespace Detail

        /**
         * @brief Returns what a bare epilogue declares: nothing, and the flops it states.
         * @see modules/operations/blas/README.md
         */
        template<typename Epilogue>
        [[nodiscard]] inline GemmEpilogueAccess epilogue_access(const Epilogue& epilogue)
        {
            GemmEpilogueAccess access;
            access.flops_per_element = Detail::EpilogueFlops<Epilogue>::value;
            Detail::append_epilogue_access(epilogue, access.operands);
            return access;
        }

        /**
         * @brief Records a GEMM whose task runs the reference kernel or the SYCL kernel, with the
         *        epilogue applied to each element of C.
         * @tparam Dom The element types the task instantiates its kernels for.
         * @tparam Epilogue The functor type; it enters the kernel closure, never its bundle.
         * @see modules/operations/blas/README.md
         */
        template<Domain Dom, typename Epilogue>
        class KernelGemmPort final : public GemmTaskPort
        {
            public:
                /** @brief Holds the functor the kernels apply. */
                explicit KernelGemmPort(Epilogue epilogue)
                    : epilogue_(std::move(epilogue))
                {

                }

                /** @brief Records gemm as one task over the reference and SYCL kernels. */
                TaskHandle record(TaskRecorder& recorder, const PreparedGemm& gemm) override
                {
                    const Epilogue kernel_epilogue = epilogue_;
                    const ScratchBuffer pack = gemm.pack;
                    const int64_t m = gemm.geometry.m;
                    const int64_t n = gemm.geometry.n;
                    const int64_t k = gemm.geometry.k;
                    const int64_t batch_size = gemm.geometry.batch_size;
                    const int64_t str_a = gemm.geometry.stride_a;
                    const int64_t str_b = gemm.geometry.stride_b;
                    const int64_t str_c = gemm.geometry.stride_c;
                    const int64_t lda = gemm.geometry.a.leading_dimension;
                    const int64_t ldb = gemm.geometry.b.leading_dimension;
                    const int64_t ldc = gemm.geometry.c.leading_dimension;
                    const Core::Layout layout = gemm.layout;
                    const Core::Transpose transA = gemm.trans_a;
                    const Core::Transpose transB = gemm.trans_b;
                    const BLASScalar alpha = gemm.alpha;
                    const BLASScalar beta = gemm.beta;
                    const void* read_A = gemm.read_a;
                    const void* read_B = gemm.read_b;
                    void* rw_C = gemm.write_c;

                    return Internal::execute_blas_task<Dom>(recorder, gemm.description,
                        gemm.operands, gemm.dtype,
                        [=](auto scalar_type, const KernelPairDispatcher& dispatcher,
                            const std::vector<sycl::event>& deps) -> sycl::event
                        {
                            using T = decltype(scalar_type);

                            auto pA = static_cast<const T*>(read_A);
                            auto pB = static_cast<const T*>(read_B);
                            auto pC = static_cast<T*>(rw_C);
                            const T alpha_t = alpha.as<T>();
                            const T beta_t = beta.as<T>();

                            return dispatcher.run(deps,
                                [=]
                                {
                                    Kernels::Reference::gemm<T>(layout, transA, transB, m, n, k,
                                                                alpha_t, pA, lda, str_a, pB, ldb,
                                                                str_b, beta_t, pC, ldc, str_c,
                                                                batch_size, kernel_epilogue);
                                },
                                [=](sycl::queue& q2,
                                    const std::vector<sycl::event>& deps2) -> sycl::event
                                {
                                    return Kernels::SYCL::gemm<T>(q2, layout, transA, transB, m, n,
                                                                  k, alpha_t, pA, lda, str_a, pB,
                                                                  ldb, str_b, beta_t, pC, ldc,
                                                                  str_c, batch_size, pack.data(),
                                                                  pack.bytes(), deps2,
                                                                  kernel_epilogue);
                                });
                        });
                }

            private:
                /// @brief The functor the kernels apply.
                Epilogue epilogue_;
        };

        /**
         * @brief Records `C = epilogue(alpha * op(A) * op(B) + beta * C, row, col)` as one task.
         * @param C Must not overlap A or B.
         * @param epilogue Must be device-copyable, capture by value, and never allocate or throw.
         * @throws std::invalid_argument on an operand, flag or scalar the call does not accept.
         * @warning A bare functor that hides a pointer is neither kept alive nor host-checked.
         * @see modules/operations/blas/README.md
         */
        template<Domain Dom = Domain::Real, typename Epilogue>
        TaskHandle execute_gemm(TaskRecorder& recorder, const Tensor& A, const Tensor& B, Tensor& C,
                                Core::Transpose transA, Core::Transpose transB,
                                BLASScalar alpha, BLASScalar beta,
                                const char* name, SushiRuntime::OperationID op_id,
                                Epilogue epilogue)
        {
            // Only the functor enters the kernel closure, never the bundle and its operand set.
            KernelGemmPort<Dom, std::decay_t<decltype(Detail::epilogue_functor(epilogue))>> port(
                Detail::epilogue_functor(epilogue));

            return record_gemm(port, recorder, A, B, C, transA, transB, alpha, beta, name, op_id,
                               Dom == Domain::Real, epilogue_access(epilogue));
        }
    } // namespace Dispatch
} // namespace SushiBLAS
