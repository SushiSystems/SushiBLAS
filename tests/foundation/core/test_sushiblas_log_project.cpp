/**************************************************************************/
/* test_sushiblas_log_project.cpp                                         */
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
 * @file test_sushiblas_log_project.cpp
 * @brief Tests that K_SUSHIBLAS_LOG_PROJECT names SushiBLAS and its level variable.
 * @author Mustafa Garip
 */

#include <gtest/gtest.h>
#include <SushiBLAS/core/sushiblas_log_project.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        TEST(Small_SushiBLASLogProject, TheProjectIsReadFromSbLogLevel)
        {
            static_assert(K_SUSHIBLAS_LOG_PROJECT.name() == "sushiblas", "the project's name");
            static_assert(K_SUSHIBLAS_LOG_PROJECT.level_variable() == "SB_LOG_LEVEL",
                          "the variable that sets its levels");
            SUCCEED();
        }
    } // namespace Test
} // namespace SushiBLAS
