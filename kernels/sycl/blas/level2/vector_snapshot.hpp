/**************************************************************************/
/* vector_snapshot.hpp                                                    */
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
 * @file vector_snapshot.hpp
 * @brief Defines the copy of a strided vector the in-place triangular products read.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <cstdint>

namespace SushiBLAS
{
    namespace Kernels
    {
        namespace SYCL
        {
            namespace Detail
            {
                /**
                 * @brief Copies element i of a strided vector into slot i of a contiguous snapshot.
                 * @see kernels/README.md
                 */
                template<typename T>
                struct VectorSnapshot
                {
                    const T* x = nullptr;  /**< The vector the product overwrites. */
                    int64_t incx = 1;      /**< Distance between consecutive elements of x. */
                    T* snapshot = nullptr; /**< Receives one element per element of x. */

                    /** @brief Copies element index. */
                    void operator()(std::size_t index) const
                    {
                        snapshot[index] = x[static_cast<int64_t>(index) * incx];
                    }
                };
            } // namespace Detail
        } // namespace SYCL
    } // namespace Kernels
} // namespace SushiBLAS
