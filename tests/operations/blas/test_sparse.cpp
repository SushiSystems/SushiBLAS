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
 * @brief Tests that SparseBLAS is built from an engine and only from one.
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
        static_assert(std::is_constructible_v<SparseBLAS, Engine&>,
                      "SparseBLAS is built over an engine.");
        static_assert(!std::is_convertible_v<Engine&, SparseBLAS>,
                      "An engine does not turn into SparseBLAS without being named.");
        static_assert(!std::is_default_constructible_v<SparseBLAS>,
                      "SparseBLAS has no engine-free form.");

        /** @brief Builds the facade over the fixture's engine. */
        class Unit_SparseBLAS : public SushiBLASTest
        {

        };

        TEST_F(Unit_SparseBLAS, IsBuiltOverAnEngine)
        {
            const SparseBLAS sparse(*engine);

            (void)sparse;
            SUCCEED();
        }
    } // namespace Test
} // namespace SushiBLAS
