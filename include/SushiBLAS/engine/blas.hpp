/****************************************************************/
/* blas.hpp                                                     */
/* SushiBLAS - https://github.com/SushiSystems/SushiBLAS        */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

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
