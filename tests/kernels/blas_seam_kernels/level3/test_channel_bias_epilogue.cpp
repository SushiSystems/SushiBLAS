/**************************************************************************/
/* test_channel_bias_epilogue.cpp                                         */
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
 * @file test_channel_bias_epilogue.cpp
 * @brief Tests that the channel bias epilogue adds the bias its stride pair names.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/blas_seam_kernels/level3/channel_bias_epilogue.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        TEST(Small_ChannelBiasEpilogue, AColumnChannelAddsTheBiasOfTheColumn)
        {
            const std::vector<float> bias{10.0f, 20.0f, 30.0f};
            const Kernels::ChannelBiasEpilogue epilogue{bias.data(), 0, 1};

            EXPECT_EQ(epilogue(1.0f, 5, 2), 31.0f);
            EXPECT_EQ(epilogue(1.0f, 0, 0), 11.0f);
        }

        TEST(Small_ChannelBiasEpilogue, ARowChannelAddsTheBiasOfTheRow)
        {
            const std::vector<double> bias{1.0, 2.0};
            const Kernels::ChannelBiasEpilogue epilogue{bias.data(), 1, 0};

            EXPECT_EQ(epilogue(5.0, 1, 7), 7.0);
        }

        TEST(Small_ChannelBiasEpilogue, BothStridesAddressATwoDimensionalBias)
        {
            const std::vector<double> bias{1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
            const Kernels::ChannelBiasEpilogue epilogue{bias.data(), 3, 1};

            EXPECT_EQ(epilogue(0.0, 1, 2), 6.0);
        }
    } // namespace Test
} // namespace SushiBLAS
