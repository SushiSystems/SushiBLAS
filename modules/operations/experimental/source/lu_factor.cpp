/**************************************************************************/
/* lu_factor.cpp                                                          */
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
 * @file lu_factor.cpp
 * @brief Defines LinalgOps::lu_factor, which records the batched in-place LU factorisation with
 * partial pivoting.
 * @author Mustafa Garip
 */

#include <SushiBLAS/experimental/lapack/linalg.hpp>

#include <string>
#include <vector>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/core/logger.hpp>
#include <SushiBLAS/engine/task.hpp>
#include <SushiBLAS/engine/operand_set.hpp>
#include <SushiBLAS/engine/operand_check.hpp>
#include <SushiBLAS/engine/task_description.hpp>
#include <SushiBLAS/linalg_kernels/linalg.hpp>
#include <SushiBLAS/experimental/dense_operand.hpp>

namespace SushiBLAS
{
    using namespace SushiRuntime::Literals;

    namespace
    {
        /// @brief The operation id, which is also the node's name when a call states none.
        constexpr const char* K_OPERATION = "linalg.lu_factor";
    } // namespace

    TaskHandle LinalgOps::lu_factor(Tensor& matrices, Tensor& pivots, Tensor& singular,
                                    std::size_t n, std::size_t batch)
    {
        return lu_factor(matrices, pivots, singular, n, batch, OperationOptions{K_OPERATION});
    }

    TaskHandle LinalgOps::lu_factor(Tensor& matrices, Tensor& pivots, Tensor& singular,
                                    std::size_t n, std::size_t batch,
                                    const OperationOptions& options)
    {
        static constexpr const char* NAME = K_OPERATION;

        Dispatch::require_operation_options(options, NAME);
        Dispatch::require_operand(n >= 1, NAME, "n", "is 0; a matrix needs at least one row");
        Internal::require_float_dtype(matrices, NAME, "matrices");
        Dispatch::require_dtype(pivots, Core::DataType::UInt32, NAME, "pivots");
        Dispatch::require_dtype(singular, Core::DataType::UInt32, NAME, "singular");
        Internal::require_dense_elements(matrices, batch * n * n, NAME, "matrices");
        Internal::require_dense_elements(pivots, batch * n, NAME, "pivots");
        Internal::require_dense_elements(singular, batch, NAME, "singular");
        Dispatch::require_operand(!pivots.region().overlaps(matrices.region()), NAME, "pivots",
                                  "overlaps 'matrices' in memory");
        Dispatch::require_operand(!singular.region().overlaps(matrices.region()) &&
                                      !singular.region().overlaps(pivots.region()),
                                  NAME, "singular", "overlaps another operand in memory");

        OperandSet operands;
        operands.updates(matrices).writes(pivots).writes(singular);

        const std::string label(options.name);
        TaskDescription description;
        description.name = label.c_str();
        description.operation = "linalg.lu_factor"_op;
        description.determinism = options.determinism;

        void* rw_matrices = matrices.data_or_null();
        auto* write_pivots = static_cast<std::uint32_t*>(pivots.data_or_null());
        auto* write_singular = static_cast<std::uint32_t*>(singular.data_or_null());

        return Internal::execute_typed_task<Dispatch::Domain::Real>(
            recorder_, description, operands, matrices.dtype, "Linear algebra operation",
            [=](auto scalar_type, const KernelPairDispatcher& dispatcher, sycl::queue& q,
                const std::vector<sycl::event>& deps) -> sycl::event
            {
                using T = decltype(scalar_type);
                if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>)
                {
                    auto a = static_cast<T*>(rw_matrices);
                    return dispatcher.run(q, deps,
                        [=]
                        {
                            Kernels::Reference::lu_factor<T>(a, write_pivots, write_singular, n,
                                                             batch);
                        },
                        [=](sycl::queue& q2, const std::vector<sycl::event>& deps2) -> sycl::event
                        {
                            return Kernels::SYCL::lu_factor<T>(q2, a, write_pivots, write_singular,
                                                               n, batch, deps2);
                        });
                }
                else
                {
                    // require_float_dtype admits FLOAT32 and FLOAT64 only.
                    SB_THROW_IF(true, "{} does not support dtype HALF.", NAME);
                    return sycl::event();
                }
            });
    }
} // namespace SushiBLAS
