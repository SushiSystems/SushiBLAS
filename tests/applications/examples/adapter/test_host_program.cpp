/**************************************************************************/
/* test_host_program.cpp                                                  */
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
 * @file test_host_program.cpp
 * @brief Tests the host program's status names and a multiply-add through the SushiBLAS backend.
 * @author Mustafa Garip
 */

#include <set>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include "host_program.hpp"

namespace HostProgram
{
    namespace Test
    {
        TEST(Small_HostProgram, EveryStatusCodeHasItsOwnName)
        {
            const StatusCode codes[] = {StatusCode::Ok, StatusCode::InvalidRequest,
                                        StatusCode::Unsupported, StatusCode::OutOfMemory,
                                        StatusCode::BackendFailed};
            std::set<std::string> names;

            for (const StatusCode code : codes)
                names.insert(to_string(code));

            EXPECT_EQ(names.size(), sizeof(codes) / sizeof(codes[0]));
            EXPECT_EQ(Status{}.code, StatusCode::Ok);
            EXPECT_TRUE(Status{}.message.empty());
        }

        TEST(Medium_HostProgram, MultiplyAddGivesTheShiftedProductAndItsSum)
        {
            BackendHandle backend;
            ASSERT_EQ(create_backend(backend).code, StatusCode::Ok);
            const Matrix a{2, 2, {1, 2, 3, 4}};
            const Matrix b{2, 2, {0, 1, 1, 0}};
            const Matrix offset{2, 2, {1, 1, 1, 1}};
            Matrix result;
            float sum = 0;

            const Status status = multiply_add(*backend, a, b, offset, result, sum);

            ASSERT_EQ(status.code, StatusCode::Ok) << status.message;
            EXPECT_EQ(result.values, (std::vector<float>{3, 2, 5, 4}));
            EXPECT_EQ(sum, 14.0f);
        }

        TEST(Medium_HostProgram, MismatchedShapesAreAnInvalidRequest)
        {
            BackendHandle backend;
            ASSERT_EQ(create_backend(backend).code, StatusCode::Ok);
            const Matrix a{2, 3, {1, 2, 3, 4, 5, 6}};
            const Matrix b{2, 2, {1, 0, 0, 1}};
            const Matrix offset{2, 2, {0, 0, 0, 0}};
            Matrix result;
            float sum = 0;

            EXPECT_EQ(multiply_add(*backend, a, b, offset, result, sum).code,
                      StatusCode::InvalidRequest);
        }
    } // namespace Test
} // namespace HostProgram
