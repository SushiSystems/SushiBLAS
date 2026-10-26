/**************************************************************************/
/* lu_solve.cpp                                                           */
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
 * @file lu_solve.cpp
 * @brief Defines LinalgOps::lu_solve, which records the in-place solve over the factors
 * lu_factor wrote.
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
        constexpr const char* K_OPERATION = "linalg.lu_solve";
    } // namespace

    TaskHandle LinalgOps::lu_solve(const Tensor& factors, const Tensor& pivots, Tensor& rhs,
                                   std::size_t n, std::size_t rhs_count, std::size_t batch)
    {
        return lu_solve(factors, pivots, rhs, n, rhs_count, batch, OperationOptions{K_OPERATION});
    }

    TaskHandle LinalgOps::lu_solve(const Tensor& factors, const Tensor& pivots, Tensor& rhs,
                                   std::size_t n, std::size_t rhs_count, std::size_t batch,
                                   const OperationOptions& options)
    {
        static constexpr const char* NAME = K_OPERATION;

        Dispatch::require_operation_options(options, NAME);
        Dispatch::require_operand(n >= 1, NAME, "n", "is 0; a matrix needs at least one row");
        Internal::require_float_dtype(factors, NAME, "factors");
        Dispatch::require_dtype(rhs, factors.dtype, NAME, "rhs");
        Dispatch::require_dtype(pivots, Core::DataType::UInt32, NAME, "pivots");
        Internal::require_dense_elements(factors, batch * n * n, NAME, "factors");
        Internal::require_dense_elements(pivots, batch * n, NAME, "pivots");
        Internal::require_dense_elements(rhs, batch * n * rhs_count, NAME, "rhs");
        Dispatch::require_operand(!rhs.region().overlaps(factors.region()) &&
                                      !rhs.region().overlaps(pivots.region()),
                                  NAME, "rhs", "overlaps another operand in memory");

        OperandSet operands;
        operands.reads(factors).reads(pivots).updates(rhs);

        const std::string label(options.name);
        TaskDescription description;
        description.name = label.c_str();
        description.operation = "linalg.lu_solve"_op;
        description.determinism = options.determinism;

        const void* read_factors = factors.data_or_null();
        const auto* read_pivots = static_cast<const std::uint32_t*>(pivots.data_or_null());
        void* rw_rhs = rhs.data_or_null();

        return Internal::execute_typed_task<Dispatch::Domain::Real>(
            recorder_, description, operands, factors.dtype, "Linear algebra operation",
            [=](auto scalar_type, const KernelPairDispatcher& dispatcher, sycl::queue& q,
                const std::vector<sycl::event>& deps) -> sycl::event
            {
                using T = decltype(scalar_type);
                if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>)
                {
                    auto lu = static_cast<const T*>(read_factors);
                    auto x = static_cast<T*>(rw_rhs);
                    return dispatcher.run(q, deps,
                        [=]
                        {
                            Kernels::Reference::lu_solve<T>(lu, read_pivots, x, n, rhs_count,
                                                            batch);
                        },
                        [=](sycl::queue& q2, const std::vector<sycl::event>& deps2) -> sycl::event
                        {
                            return Kernels::SYCL::lu_solve<T>(q2, lu, read_pivots, x, n, rhs_count,
                                                              batch, deps2);
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
