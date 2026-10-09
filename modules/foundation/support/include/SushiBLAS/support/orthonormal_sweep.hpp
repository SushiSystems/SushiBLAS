/**************************************************************************/
/* orthonormal_sweep.hpp                                                  */
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
 * @file orthonormal_sweep.hpp
 * @brief Declares the vector layout and the constants the orthonormalisation kernels of both back ends share.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>

namespace SushiBLAS
{
    namespace Kernels
    {
        /**
         * @brief Holds the strided layout of the vectors a sweep orthonormalises.
         * @see modules/foundation/support/README.md
         */
        struct VectorSet
        {
            int64_t count = 0;       /**< Number of vectors. */
            int64_t length = 0;      /**< Elements per vector. */
            int64_t elem_stride = 0; /**< Element-to-element step within one vector. */
            int64_t vec_stride = 0;  /**< First-element-to-first-element step between vectors. */
        };

        /** @brief Holds how many Gram-Schmidt sweeps run before the vectors are normalised. */
        inline constexpr int K_ORTHONORMAL_SWEEPS = 2;

        /** @brief Returns the squared norm at or below which a vector counts as zero. */
        template<typename T>
        [[nodiscard]] inline constexpr T tiny_norm_sq() noexcept
        {
            return static_cast<T>(1e-12) * static_cast<T>(1e-12);
        }
    } // namespace Kernels
} // namespace SushiBLAS
