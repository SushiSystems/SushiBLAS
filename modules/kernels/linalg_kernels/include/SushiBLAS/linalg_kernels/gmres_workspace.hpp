/**************************************************************************/
/* gmres_workspace.hpp                                                    */
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
 * @file gmres_workspace.hpp
 * @brief Declares GmresSettings and GmresLayout, the stopping rule and scratch shape of a solve.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <SushiBLAS/linalg_kernels/chunked_dot.hpp>

namespace SushiBLAS
{
    namespace Kernels
    {
        /**
         * @brief Holds the stopping rule and the preconditioner block of one GMRES solve.
         * @see modules/kernels/linalg_kernels/README.md
         */
        struct GmresSettings
        {
                std::size_t restart = 30;         /**< Krylov vectors kept before a restart. */
                std::size_t max_iterations = 300; /**< Inner iterations across all restarts. */
                double relative_tolerance = 1e-6; /**< Stop once |r| / |b| falls below it. */
                std::size_t block_size = 1;       /**< Diagonal block of block-Jacobi. */
        };

        /**
         * @brief Holds where each array of a GMRES workspace starts, in T elements.
         * @see modules/kernels/linalg_kernels/README.md
         */
        struct GmresLayout
        {
                /**
                 * @brief Lays out the workspace of order @p n with @p kept vectors and @p block.
                 * @pre block divides n; kept and block are at least 1.
                 */
                GmresLayout(std::size_t n, std::size_t kept, std::size_t block) noexcept
                    : order(n), restart(kept), block_size(block)
                {
                    basis = 0;
                    hessenberg = basis + (kept + 1) * n;
                    cosines = hessenberg + (kept + 1) * kept;
                    sines = cosines + kept;
                    projection = sines + kept;
                    coefficients = projection + kept + 1;
                    residual = coefficients + kept;
                    work = residual + n;
                    preconditioned = work + n;
                    factors = preconditioned + n;
                    partials = factors + n * block;
                    scalars = partials + dot_chunks(n);
                }

                /** @brief Returns the bytes of the workspace for elements @p element_bytes wide. */
                [[nodiscard]] std::size_t bytes(std::size_t element_bytes) const noexcept
                {
                    return scalars * element_bytes + order * sizeof(std::uint32_t);
                }

                std::size_t order = 0;          /**< Order of the matrix. */
                std::size_t restart = 0;        /**< Krylov vectors kept before a restart. */
                std::size_t block_size = 0;     /**< Order of each diagonal block. */
                std::size_t basis = 0;          /**< restart + 1 basis vectors. */
                std::size_t hessenberg = 0;     /**< (restart + 1) * restart, row-major. */
                std::size_t cosines = 0;        /**< Cosine of each Givens rotation. */
                std::size_t sines = 0;          /**< Sine of each Givens rotation. */
                std::size_t projection = 0;     /**< Rotated beta e1. */
                std::size_t coefficients = 0;   /**< Least-squares solution of one cycle. */
                std::size_t residual = 0;       /**< b - A x at the start of a cycle. */
                std::size_t work = 0;           /**< Product and correction scratch. */
                std::size_t preconditioned = 0; /**< M^-1 of a basis vector or the correction. */
                std::size_t factors = 0;        /**< LU factors of each diagonal block. */
                std::size_t partials = 0;       /**< One sum per chunk of a dot product. */
                std::size_t scalars = 0;        /**< T elements in all; the pivots follow. */
        };
    } // namespace Kernels
} // namespace SushiBLAS
