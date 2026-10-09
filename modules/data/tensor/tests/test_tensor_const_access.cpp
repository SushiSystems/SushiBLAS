/**************************************************************************/
/* test_tensor_const_access.cpp                                           */
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
 * @file test_tensor_const_access.cpp
 * @brief Tests that Tensor's const and non-const data accessors return matching pointers.
 * @author Mustafa Garip
 */

#include <utility>
#include <type_traits>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.h>
#include "test_utils.hpp"

namespace SushiBLAS
{
    namespace Test
    {
        static_assert(std::is_same<decltype(std::declval<Tensor&>().data()), void*>::value,
                      "data() on a non-const Tensor returns void*");
        static_assert(std::is_same<decltype(std::declval<const Tensor&>().data()), const void*>::value,
                      "data() on a const Tensor returns const void*");
        static_assert(std::is_same<decltype(std::declval<Tensor&>().data_as<float>()), float*>::value,
                      "data_as<T>() on a non-const Tensor returns T*");
        static_assert(std::is_same<decltype(std::declval<const Tensor&>().data_as<float>()), const float*>::value,
                      "data_as<T>() on a const Tensor returns const T*");
        static_assert(std::is_same<decltype(std::declval<Tensor&>().data_or_null()), void*>::value,
                      "data_or_null() on a non-const Tensor returns void*");
        static_assert(std::is_same<decltype(std::declval<const Tensor&>().data_or_null()), const void*>::value,
                      "data_or_null() on a const Tensor returns const void*");

        /** @brief Fixture for the Tensor data accessor tests. */
        class Unit_TensorConstAccessTest : public SushiBLASTest
        {

        };

        /**
         * @brief Checks that every const accessor returns the address its non-const twin returns.
         */
        static void expect_same_addresses(Tensor& t)
        {
            const Tensor& view = t;
            EXPECT_EQ(view.data(), static_cast<const void*>(t.data()));
            EXPECT_EQ(view.data_as<float>(), static_cast<const float*>(t.data_as<float>()));
            EXPECT_EQ(view.data_or_null(), static_cast<const void*>(t.data_or_null()));
        }

        TEST_F(Unit_TensorConstAccessTest, ConstAndMutableAccessorsShareAnAddress)
        {
            Tensor t = engine->create_tensor({4, 8}, Core::DataType::Float32);
            expect_same_addresses(t);
        }

        TEST_F(Unit_TensorConstAccessTest, ConstAndMutableAccessorsShareAnAddressOnAnOffsetView)
        {
            Tensor t = engine->create_tensor({4, 8}, Core::DataType::Float32);
            Tensor view = t.slice(0, 1, 3);
            ASSERT_GT(view.storage_offset, 0);
            expect_same_addresses(view);

            const Tensor& const_view = view;
            EXPECT_EQ(static_cast<const char*>(const_view.data()),
                      static_cast<const char*>(static_cast<const Tensor&>(t).data()) + view.storage_offset * 4);
        }

        TEST_F(Unit_TensorConstAccessTest, DataOrNullIsNullWithoutStorage)
        {
            Tensor t;
            const Tensor& view = t;
            EXPECT_EQ(t.data_or_null(), nullptr);
            EXPECT_EQ(view.data_or_null(), nullptr);
        }
    } // namespace Test
} // namespace SushiBLAS
