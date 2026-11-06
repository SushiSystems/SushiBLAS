/**************************************************************************/
/* test_reductions_log_category.cpp                                       */
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
 * @file test_reductions_log_category.cpp
 * @brief Tests that K_REDUCTIONS_LOG_CATEGORY names the reductions module of SushiBLAS.
 * @author Mustafa Garip
 */

#include <gtest/gtest.h>
#include <SushiBLAS/reductions/reductions_log_category.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        TEST(Small_ReductionsLogCategory, TheCategoryIsNamedAfterTheModule)
        {
            static_assert(K_REDUCTIONS_LOG_CATEGORY.module() == "reductions", "the folder name of the module");
            static_assert(K_REDUCTIONS_LOG_CATEGORY.project().name() == "sushiblas", "the project of SushiBLAS");
        }
    } // namespace Test
} // namespace SushiBLAS
