/**************************************************************************/
/* default_type_names.hpp                                                 */
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
 * @file default_type_names.hpp
 * @brief Declares DefaultTypeNames and SB_TYPED_TEST_SUITE, a typed suite with its name generator spelled out.
 * @author Mustafa Garip
 * @see modules/foundation/typed_suite_naming/README.md
 */

#pragma once

#include <string>
#include <gtest/gtest.h>

namespace SushiBLAS
{
    namespace Test
    {
        /**
         * @brief Names each type parameter by its index, as GoogleTest does by default.
         * @see modules/foundation/typed_suite_naming/README.md
         */
        struct DefaultTypeNames
        {
            /**
             * @brief The suffix for the @p i-th type parameter.
             * @tparam T The type parameter; unused, as in GoogleTest's own default.
             * @param i The type's index within the suite's type list.
             * @return @p i rendered as a decimal string.
             */
            template<typename T>
            static std::string GetName(int i) { return std::to_string(i); }
        };
    } // namespace Test
} // namespace SushiBLAS

/**
 * @brief `TYPED_TEST_SUITE` with the name generator spelled out.
 *
 * See @ref SushiBLAS::Test::DefaultTypeNames for why the two-argument form cannot be
 * written directly on a C++17 baseline. Behaviour is identical.
 */
#define SB_TYPED_TEST_SUITE(SuiteName, Types) \
    TYPED_TEST_SUITE(SuiteName, Types, SushiBLAS::Test::DefaultTypeNames)
