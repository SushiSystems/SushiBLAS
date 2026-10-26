/**************************************************************************/
/* gmres.cpp                                                              */
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
 * @file gmres.cpp
 * @brief Defines LinalgOps::gmres, which records a restarted, block-Jacobi preconditioned GMRES
 * solve over a workspace leased once at record time.
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
#include <SushiBLAS/engine/scratch.hpp>
#include <SushiBLAS/engine/operand_set.hpp>
#include <SushiBLAS/engine/operand_check.hpp>
#include <SushiBLAS/engine/task_description.hpp>
#include <SushiBLAS/operands/scratch_lease.hpp>
#include <SushiBLAS/linalg_kernels/linalg.hpp>
#include <SushiBLAS/experimental/dense_operand.hpp>

namespace SushiBLAS
{
    using namespace SushiRuntime::Literals;

    TaskHandle LinalgOps::gmres(const Tensor& matrix, const Tensor& rhs, Tensor& solution,
                                Tensor& iterations, Tensor& residual, const GmresOptions& options)
    {
        static constexpr const char* NAME = "linalg.gmres";

        Dispatch::require_operation_options(options.node, NAME);
        Internal::require_float_dtype(matrix, NAME, "matrix");
        Dispatch::require_dtype(rhs, matrix.dtype, NAME, "rhs");
        Dispatch::require_dtype(solution, matrix.dtype, NAME, "solution");
        Dispatch::require_dtype(residual, matrix.dtype, NAME, "residual");
        Dispatch::require_dtype(iterations, Core::DataType::UInt32, NAME, "iterations");

        const std::size_t n = static_cast<std::size_t>(rhs.num_elements);
        Dispatch::require_operand(n >= 1, NAME, "rhs", "is empty; the system needs one unknown");
        Dispatch::require_operand(options.restart >= 1, NAME, "options",
                                  "has restart 0; at least 1 is required");
        Dispatch::require_operand(options.max_iterations >= 1, NAME, "options",
                                  "has max_iterations 0; at least 1 is required");
        Dispatch::require_operand(options.block_size >= 1 && n % options.block_size == 0, NAME,
                                  "options", "has block_size {}, which does not divide n = {}",
                                  options.block_size, n);
        Internal::require_dense_elements(matrix, n * n, NAME, "matrix");
        Internal::require_dense_elements(rhs, n, NAME, "rhs");
        Internal::require_dense_elements(solution, n, NAME, "solution");
        Internal::require_dense_elements(iterations, 1, NAME, "iterations");
        Internal::require_dense_elements(residual, 1, NAME, "residual");
        Dispatch::require_operand(!solution.region().overlaps(matrix.region()) &&
                                      !solution.region().overlaps(rhs.region()),
                                  NAME, "solution", "overlaps an input in memory");

        Kernels::GmresSettings settings;
        settings.restart = options.restart;
        settings.max_iterations = options.max_iterations;
        settings.relative_tolerance = options.relative_tolerance;
        settings.block_size = options.block_size;

        OperandSet operands;
        operands.reads(matrix).reads(rhs).updates(solution).writes(iterations).writes(residual);

        const std::size_t element_bytes =
            matrix.dtype == Core::DataType::Float64 ? sizeof(double) : sizeof(float);
        const Kernels::GmresLayout layout(n, settings.restart, settings.block_size);
        const ScratchBuffer workspace = Internal::allocate_blas_scratch(
            recorder_, matrix, layout.bytes(element_bytes), operands);

        const std::string label(options.node.name);
        TaskDescription description;
        description.name = label.c_str();
        description.operation = "linalg.gmres"_op;
        description.determinism = options.node.determinism;

        const void* read_matrix = matrix.data_or_null();
        const void* read_rhs = rhs.data_or_null();
        void* rw_solution = solution.data_or_null();
        auto* write_iterations = static_cast<std::uint32_t*>(iterations.data_or_null());
        void* write_residual = residual.data_or_null();

        return Internal::execute_typed_task<Dispatch::Domain::Real>(
            recorder_, description, operands, matrix.dtype, "Linear algebra operation",
            [=](auto scalar_type, const KernelPairDispatcher& dispatcher, sycl::queue& q,
                const std::vector<sycl::event>& deps) -> sycl::event
            {
                using T = decltype(scalar_type);
                if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>)
                {
                    auto a = static_cast<const T*>(read_matrix);
                    auto b = static_cast<const T*>(read_rhs);
                    auto x = static_cast<T*>(rw_solution);
                    auto r = static_cast<T*>(write_residual);
                    return dispatcher.run(q, deps,
                        [=]
                        {
                            Kernels::Reference::gmres<T>(a, b, x, write_iterations, r, n,
                                                         settings);
                        },
                        [=](sycl::queue& q2, const std::vector<sycl::event>& deps2) -> sycl::event
                        {
                            return Kernels::SYCL::gmres<T>(q2, a, b, x, write_iterations, r, n,
                                                           settings, workspace.data(), deps2);
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
