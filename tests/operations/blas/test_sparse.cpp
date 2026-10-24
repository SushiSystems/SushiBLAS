/**************************************************************************/
/* test_sparse.cpp                                                        */
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
 * @file test_sparse.cpp
 * @brief Tests that SparseBLAS is built from a recorder and only from one.
 * @author Mustafa Garip
 */

#include <type_traits>
#include <gtest/gtest.h>
#include <SushiBLAS/blas/sparse.hpp>
#include <SushiBLAS/engine_fixture/sushiblas_test.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        static_assert(std::is_constructible_v<SparseBLAS, TaskRecorder&>,
                      "SparseBLAS is built over a recorder.");
        static_assert(!std::is_convertible_v<TaskRecorder&, SparseBLAS>,
                      "A recorder does not turn into SparseBLAS without being named.");
        static_assert(!std::is_default_constructible_v<SparseBLAS>,
                      "SparseBLAS has no recorder-free form.");

        /** @brief Builds the facade over the fixture's recorder. */
        class Medium_SparseBLAS : public SushiBLASTest
        {

        };

        TEST_F(Medium_SparseBLAS, IsBuiltOverARecorder)
        {
            const SparseBLAS sparse(recorder());

            (void)sparse;
            SUCCEED();
        }
    } // namespace Test
} // namespace SushiBLAS
