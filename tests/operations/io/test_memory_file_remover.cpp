/**************************************************************************/
/* test_memory_file_remover.cpp                                                  */
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
 * @file test_memory_file_remover.cpp
 * @brief Tests that MemoryFileRemover holds each path on its own.
 * @author Mustafa Garip
 * @see tests/operations/io/README.md
 */

#include <gtest/gtest.h>
#include <SushiBLAS/io/memory_file_remover.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        TEST(Small_MemoryFileRemover, RemovingOnePathLeavesTheOthersHeld)
        {
            IOFiles::MemoryFileRemover remover;
            remover.add("first.bin");
            remover.add("second.bin");

            EXPECT_TRUE(remover.remove("first.bin"));
            EXPECT_FALSE(remover.has("first.bin"));
            EXPECT_TRUE(remover.has("second.bin"));
        }
    } // namespace Test
} // namespace SushiBLAS
