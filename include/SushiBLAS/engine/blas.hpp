/**************************************************************************/
/* blas.hpp                                                               */
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
 * @file blas.hpp
 * @brief Declares BLASOps, the single access point for the Level 1, 2, 3 and sparse BLAS operations.
 * @author Mustafa Garip
 */

#pragma once

#include <SushiBLAS/engine/blas/level1.hpp>
#include <SushiBLAS/engine/blas/level2.hpp>
#include <SushiBLAS/engine/blas/level3.hpp>
#include <SushiBLAS/engine/blas/sparse.hpp>

namespace SushiBLAS 
{
    class Engine;

    /**
     * @class BLASOps
     * @brief A unified interface for all BLAS operations (Standard & Sparse).
     * 
     * This class inherits from Level1, Level2, Level3, and SparseBLAS to provide 
     * a single point of access for all BLAS functionality within SushiBLAS.
     */
    class BLASOps : public Level1, public Level2, public Level3, public SparseBLAS
    {
        public:
            /**
             * @brief Construct BLASOps with a reference to the engine.
             * @param e The SushiBLAS engine.
             */
            explicit BLASOps(Engine& e) : Level1(e), Level2(e), Level3(e), SparseBLAS(e) {}
    };

} // namespace SushiBLAS
