/****************************************************************/
/* dispatch.hpp                                                 */
/* SushiBLAS - https://github.com/SushiSystems/SushiBLAS        */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

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
        // The access set travels in the epilogue, so it cannot be lost; see include/SushiBLAS/engine/blas/README.md.

        /**
         * @brief Holds a GEMM epilogue functor together with the regions it reads and writes.
         * @tparam Epilogue The functor type, under the contract execute_gemm states for a bare one.
         * @see include/SushiBLAS/engine/blas/README.md
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
                 * @brief Declares that the epilogue writes operand.
                 * @return `*this`, so declarations chain.
                 * @throws std::runtime_error if operand has no storage.
                 * @see include/SushiBLAS/engine/blas/README.md
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
                 * @brief Returns the region of operand, refusing one that keys on no allocation.
                 * @param direction "reads" or "writes", for the message.
                 * @throws std::runtime_error if operand has no storage.
                 * @see include/SushiBLAS/engine/blas/README.md
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
         * @brief Bundles functor into a GemmEpilogue that declares nothing yet.
         * @see include/SushiBLAS/engine/blas/README.md
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
             * @brief Returns the functor a GemmEpilogue bundle carries.
             * @see include/SushiBLAS/engine/blas/README.md
             */
            template<typename Epilogue>
            [[nodiscard]] inline const Epilogue&
            epilogue_functor(const GemmEpilogue<Epilogue>& bundle) noexcept
            {
                return bundle.functor();
            }

            /**
             * @brief Adds nothing, because a bare epilogue declares no operands.
             * @see include/SushiBLAS/engine/blas/README.md
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
         * @brief Records `C = epilogue(alpha * op(A) * op(B) + beta * C, row, col)` as one task.
         * @param C Must not overlap A or B.
         * @param epilogue Must be device-copyable, capture by value, and neither allocate nor throw.
         * @throws std::runtime_error on rank < 2, a dtype mismatch, or a dtype outside Dom.
         * @see include/SushiBLAS/engine/blas/README.md
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
                // A dtype outside the domain is rejected at record time; see include/SushiBLAS/engine/blas/README.md.
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

            // C is declared as a read whenever beta is non-zero; see include/SushiBLAS/engine/blas/README.md.
            std::vector<SushiRuntime::Core::ResourceRegion> reads = {};
            if (read_A) reads.push_back(A.region());
            if (read_B) reads.push_back(B.region());
            if (rw_C && beta != 0.0) reads.push_back(C.region());
            std::vector<SushiRuntime::Core::ResourceRegion> writes = {};
            if (rw_C) writes.push_back(C.region());

            // The epilogue's declared operands join the access set here; see include/SushiBLAS/engine/blas/README.md.
            Detail::append_epilogue_access(epilogue, reads, writes);

            // Only the functor enters the kernel closure, not the bundle; see include/SushiBLAS/engine/blas/README.md.
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
