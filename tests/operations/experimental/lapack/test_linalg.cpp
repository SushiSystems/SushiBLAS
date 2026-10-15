/**************************************************************************/
/* test_linalg.cpp                                                        */
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
 * @file test_linalg.cpp
 * @brief Tests the shape of the quarantined LinalgOps declaration, which no source implements.
 * @author Mustafa Garip
 */

#include <type_traits>
#include <gtest/gtest.h>
#include <SushiBLAS/engine/task_recorder.hpp>
#include <SushiBLAS/experimental/lapack/linalg.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        static_assert(std::is_nothrow_constructible_v<LinalgOps, TaskRecorder&>,
                      "LinalgOps is a view over a recorder.");
        static_assert(!std::is_convertible_v<TaskRecorder&, LinalgOps>,
                      "A recorder does not turn into LinalgOps without being named.");
        static_assert(std::is_copy_constructible_v<LinalgOps> &&
                          !std::is_copy_assignable_v<LinalgOps>,
                      "LinalgOps copies but never rebinds.");

        TEST(Unit_Linalg, TheDeclarationKeepsTheShapeOfARecorderView)
        {
            SUCCEED();
        }
    } // namespace Test
} // namespace SushiBLAS
