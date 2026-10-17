/**************************************************************************/
/* test_blas_kernels_log_category.cpp                                     */
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
 * @file test_blas_kernels_log_category.cpp
 * @brief Tests that K_BLAS_KERNELS_LOG_CATEGORY names the blas_kernels module of SushiBLAS.
 * @author Mustafa Garip
 */

#include <gtest/gtest.h>
#include <SushiBLAS/blas_kernels/blas_kernels_log_category.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        TEST(Small_BlasKernelsLogCategory, TheCategoryIsNamedAfterTheModule)
        {
            static_assert(K_BLAS_KERNELS_LOG_CATEGORY.module() == "blas_kernels", "the folder name of the module");
            static_assert(K_BLAS_KERNELS_LOG_CATEGORY.project().name() == "sushiblas", "the project of SushiBLAS");
            SUCCEED();
        }
    } // namespace Test
} // namespace SushiBLAS
