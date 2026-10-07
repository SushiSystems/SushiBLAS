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
 * @brief Declares BLASOps, the single access point for the Level 1, 2 and 3 BLAS operations.
 * @author Mustafa Garip
 */

#pragma once

#include <SushiBLAS/engine.hpp>
#include <SushiBLAS/engine/task_handle.hpp>
#include <SushiBLAS/engine/task_recorder.hpp>
#include <SushiBLAS/engine/blas/level1.hpp>
#include <SushiBLAS/engine/blas/level2.hpp>
#include <SushiBLAS/engine/blas/level3.hpp>

namespace SushiBLAS
{
    /**
     * @class BLASOps
     * @brief Gives one point of access to the Level 1, 2 and 3 BLAS routines.
     */
    class BLASOps final : public Level1, public Level2, public Level3
    {
        public:
            /** @brief Builds a view that records through recorder, which must outlive the view. */
            explicit BLASOps(TaskRecorder& recorder) noexcept
                : Level1(recorder), Level2(recorder), Level3(recorder)
            {

            }

            /** @brief Builds a view that records on engine's own graph. */
            explicit BLASOps(Engine& engine) noexcept : BLASOps(engine.get_recorder()) {}
    };
} // namespace SushiBLAS
