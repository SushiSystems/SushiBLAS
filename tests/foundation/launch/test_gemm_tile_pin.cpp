/**************************************************************************/
/* test_gemm_tile_pin.cpp                                                 */
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
 * @file test_gemm_tile_pin.cpp
 * @brief Tests the read of the GEMM tile pin from a variable's value.
 * @author Mustafa Garip
 */

#include <string>
#include <gtest/gtest.h>
#include <SushiBLAS/launch/gemm_tile_pin.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        TEST(Small_GemmTilePin, AnUnsetVariableGivesAnEmptyPin)
        {
            EXPECT_TRUE(Device::gemm_tile_pin_from(nullptr).empty());
        }

        TEST(Small_GemmTilePin, ASetVariableKeepsItsValueUnchanged)
        {
            EXPECT_EQ(Device::gemm_tile_pin_from("Wide"), std::string("Wide"));
        }

        TEST(Small_GemmTilePin, AnEmptyValueGivesAnEmptyPin)
        {
            EXPECT_TRUE(Device::gemm_tile_pin_from("").empty());
        }

        TEST(Small_GemmTilePin, TheVariableIsNamedSbGemmTile)
        {
            EXPECT_STREQ(Device::K_GEMM_TILE_PIN_VARIABLE, "SB_GEMM_TILE");
        }

        TEST(Small_GemmTilePin, TheProcessPinIsReadOnce)
        {
            EXPECT_EQ(&Device::gemm_tile_pin(), &Device::gemm_tile_pin());
        }
    } // namespace Test
} // namespace SushiBLAS
