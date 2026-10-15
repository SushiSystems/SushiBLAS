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
 * @brief Tests that sparse.hpp alone reaches the structure checks and both kinds of kernel.
 * @author Mustafa Garip
 */

#include <SushiBLAS/sparse_kernels/sparse.hpp>

#include <cstdint>
#include <vector>
#include <gtest/gtest.h>

namespace SushiBLAS
{
    namespace Test
    {
        TEST(Small_SparseKernels, TheStructureCheckAndAReferenceKernelAreReachable)
        {
            const std::vector<int64_t> offsets{0, 1, 2};
            const std::vector<int64_t> columns{1, 0};
            const std::vector<double> values{2.0, 3.0};
            const Kernels::CsrPattern<int64_t> pattern{2, 2, 2, 0, offsets.data(), columns.data()};
            const std::vector<double> x{1.0, 1.0};
            std::vector<double> y{0.0, 0.0};

            Kernels::Reference::spmv(Core::Transpose::NoTrans, pattern, values.data(), 1.0,
                                     x.data(), 1, 0.0, y.data(), 1);

            EXPECT_EQ(Kernels::Reference::find_defect(pattern),
                      Kernels::Reference::StructureDefect::None);
            EXPECT_EQ(y, (std::vector<double>{2.0, 3.0}));
        }
    } // namespace Test
} // namespace SushiBLAS
