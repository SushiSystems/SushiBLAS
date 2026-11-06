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
 * @file sparse.hpp
 * @brief Declares SparseBLAS, the placeholder for sparse BLAS operations.
 * @author Mustafa Garip
 */

#pragma once

#include <SushiBLAS/engine/task_recorder.hpp>
#include <SushiBLAS/tensor/tensor.hpp>

namespace SushiBLAS 
{
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
            explicit SparseBLAS(TaskRecorder& recorder) noexcept : recorder_(recorder) {}

        private:
            // Stays unread until this class declares an operation; see modules/operations/blas/README.md.
            [[maybe_unused]] TaskRecorder& recorder_;
    };

} // namespace SushiBLAS
