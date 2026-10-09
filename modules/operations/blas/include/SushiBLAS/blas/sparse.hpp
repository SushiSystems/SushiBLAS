/**************************************************************************/
/* sparse.hpp                                                             */
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
 * @file sparse.hpp
 * @brief Declares SparseBLAS, the placeholder for sparse BLAS operations.
 * @author Mustafa Garip
 */

#pragma once

#include <sycl/sycl.hpp>
#include <SushiBLAS/tensor/tensor.hpp>

namespace SushiBLAS 
{
    class Engine;

    /**
     * @class SparseBLAS
     * @brief Sparse matrix operations.
     * 
     * This module handles matrices where most elements are zero.
     * It uses specialized storage formats like CSR, CSC, and COO.
     */
    class SparseBLAS 
    {
        public:
            explicit SparseBLAS(Engine& e) : engine_(e) {}

        private:
            // Stays unread until this class declares an operation; see modules/operations/blas/README.md.
            [[maybe_unused]] Engine& engine_;
    };

} // namespace SushiBLAS
