/**************************************************************************/
/* test_include_order.cpp                                                 */
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
 * @file test_include_order.cpp
 * @brief Tests that a seam defined before write_binding.hpp still checks a bundle's reads.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <stdexcept>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/tensor/storage.hpp>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiBLAS/functor_map/read_binding.hpp>
#include <SushiRuntime/core/span.hpp>
#include <SushiRuntime/core/sushi_ptr.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            /**
             * @brief Checks the declared reads of op as a seam does, by qualified name.
             *
             * It is defined before write_binding.hpp is included, so it sees only what
             * read_binding.hpp declares.
             */
            template<typename Op>
            void check_reads_as_a_seam_does(const Op& op)
            {
                Dispatch::Detail::require_declared_reads(op, Core::DataType::Float32, "seam");
            }

            /** @brief Returns a FLOAT32 tensor of four elements with no storage. */
            Tensor tensor_without_storage()
            {
                const std::vector<std::int64_t> dims = {4};
                return Tensor(SushiRuntime::sushi_ptr<Storage>(),
                              SushiRuntime::span<const std::int64_t>(dims.data(), dims.size()),
                              Core::DataType::Float32);
            }
        } // namespace
    } // namespace Test
} // namespace SushiBLAS

// This include follows the seam above on purpose: the test is about that order.
#include <SushiBLAS/functor_map/write_binding.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /**
         * @brief Tests that a seam defined before write_binding.hpp finds the write bundle's
         *        overload and refuses its read without storage, instead of checking nothing.
         */
        TEST(Small_Regression_BindingIncludeOrder, ASeamDefinedFirstChecksAWriteBundlesReads)
        {
            const Tensor read = tensor_without_storage();
            Tensor written = tensor_without_storage();
            const auto bundle = Dispatch::bind_writes(
                Dispatch::bind_reads([](float, const float*, float*) {},
                                     Dispatch::read_as_element(read)),
                Dispatch::write_as_element(written));

            EXPECT_THROW(check_reads_as_a_seam_does(bundle), std::invalid_argument);
        }
    } // namespace Test
} // namespace SushiBLAS
