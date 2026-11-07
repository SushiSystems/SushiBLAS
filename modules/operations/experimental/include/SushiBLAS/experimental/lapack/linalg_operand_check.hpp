/**************************************************************************/
/* linalg_operand_check.hpp                                               */
/**************************************************************************/
/*                          This file is part of:                         */
/*                               SushiSkills                              */
/*               https://github.com/SushiSystems/SushiSkills              */
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

/**
 * @file linalg_operand_check.hpp
 * @brief Declares the record-time operand checks of LinalgOps::lu_factor, lu_solve and gmres,
 *        which read shapes and storage facts and so run without a runtime.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <SushiBLAS/engine/operand_check.hpp>
#include <SushiBLAS/experimental/dense_operand.hpp>
#include <SushiBLAS/experimental/lapack/linalg.hpp>

namespace SushiBLAS
{
    namespace Internal
    {
        /**
         * @brief Throws unless the operands of a batched LU factorisation are a well-formed call.
         * @param name The operation id the messages name.
         * @throws std::invalid_argument naming the operand and the rule it breaks;
         *         UnsupportedExecution for a dtype the device of an operand cannot compute.
         * @see modules/operations/experimental/README.md
         */
        inline void check_lu_factor_operands(const Dispatch::PlacedTensor& matrices,
                                             const Dispatch::PlacedTensor& pivots,
                                             const Dispatch::PlacedTensor& singular,
                                             std::size_t n, std::size_t batch, const char* name)
        {
            Dispatch::require_operand(n >= 1, name, "n", "is 0; a matrix needs at least one row");
            require_float_dtype(matrices.tensor, name, "matrices");
            Dispatch::require_dtype(pivots.tensor, Core::DataType::UInt32, name, "pivots");
            Dispatch::require_dtype(singular.tensor, Core::DataType::UInt32, name, "singular");
            require_dense_elements(matrices, batch * n * n, name, "matrices");
            require_dense_elements(pivots, batch * n, name, "pivots");
            require_dense_elements(singular, batch, name, "singular");
            Dispatch::require_disjoint(pivots.storage, matrices.storage, name, "pivots",
                                       "matrices");
            Dispatch::require_operand(!singular.storage.region.overlaps(matrices.storage.region) &&
                                          !singular.storage.region.overlaps(pivots.storage.region),
                                      name, "singular", "overlaps another operand in memory");
        }

        /**
         * @brief Throws unless the operands of a batched LU solve are a well-formed call.
         * @param name The operation id the messages name.
         * @throws std::invalid_argument naming the operand and the rule it breaks;
         *         UnsupportedExecution for a dtype the device of an operand cannot compute.
         * @see modules/operations/experimental/README.md
         */
        inline void check_lu_solve_operands(const Dispatch::PlacedTensor& factors,
                                            const Dispatch::PlacedTensor& pivots,
                                            const Dispatch::PlacedTensor& rhs, std::size_t n,
                                            std::size_t rhs_count, std::size_t batch,
                                            const char* name)
        {
            Dispatch::require_operand(n >= 1, name, "n", "is 0; a matrix needs at least one row");
            require_float_dtype(factors.tensor, name, "factors");
            Dispatch::require_dtype(rhs.tensor, factors.tensor.dtype, name, "rhs");
            Dispatch::require_dtype(pivots.tensor, Core::DataType::UInt32, name, "pivots");
            require_dense_elements(factors, batch * n * n, name, "factors");
            require_dense_elements(pivots, batch * n, name, "pivots");
            require_dense_elements(rhs, batch * n * rhs_count, name, "rhs");
            Dispatch::require_operand(!rhs.storage.region.overlaps(factors.storage.region) &&
                                          !rhs.storage.region.overlaps(pivots.storage.region),
                                      name, "rhs", "overlaps another operand in memory");
        }

        /**
         * @brief Throws unless the operands and the options of a GMRES solve are well formed.
         * @param name The operation id the messages name.
         * @return The number of unknowns n, which is the element count of rhs.
         * @throws std::invalid_argument naming the operand and the rule it breaks;
         *         UnsupportedExecution for a dtype the device of an operand cannot compute.
         */
        inline std::size_t check_gmres_operands(const Dispatch::PlacedTensor& matrix,
                                                const Dispatch::PlacedTensor& rhs,
                                                const Dispatch::PlacedTensor& solution,
                                                const Dispatch::PlacedTensor& iterations,
                                                const Dispatch::PlacedTensor& residual,
                                                const GmresOptions& options, const char* name)
        {
            require_float_dtype(matrix.tensor, name, "matrix");
            Dispatch::require_dtype(rhs.tensor, matrix.tensor.dtype, name, "rhs");
            Dispatch::require_dtype(solution.tensor, matrix.tensor.dtype, name, "solution");
            Dispatch::require_dtype(residual.tensor, matrix.tensor.dtype, name, "residual");
            Dispatch::require_dtype(iterations.tensor, Core::DataType::UInt32, name,
                                    "iterations");

            const std::size_t n = static_cast<std::size_t>(rhs.tensor.num_elements);
            Dispatch::require_operand(n >= 1, name, "rhs",
                                      "is empty; the system needs one unknown");
            Dispatch::require_operand(options.restart >= 1, name, "options",
                                      "has restart 0; at least 1 is required");
            Dispatch::require_operand(options.max_iterations >= 1, name, "options",
                                      "has max_iterations 0; at least 1 is required");
            Dispatch::require_operand(options.block_size >= 1 && n % options.block_size == 0,
                                      name, "options",
                                      "has block_size {}, which does not divide n = {}",
                                      options.block_size, n);
            require_dense_elements(matrix, n * n, name, "matrix");
            require_dense_elements(rhs, n, name, "rhs");
            require_dense_elements(solution, n, name, "solution");
            require_dense_elements(iterations, 1, name, "iterations");
            require_dense_elements(residual, 1, name, "residual");
            Dispatch::require_operand(!solution.storage.region.overlaps(matrix.storage.region) &&
                                          !solution.storage.region.overlaps(rhs.storage.region),
                                      name, "solution", "overlaps an input in memory");
            return n;
        }
    } // namespace Internal
} // namespace SushiBLAS
