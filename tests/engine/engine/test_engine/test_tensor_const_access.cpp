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
 * @file test_tensor_const_access.cpp
 * @brief Tests that Tensor's const and non-const data accessors return matching pointers.
 * @author Mustafa Garip
 */

#include <utility>
#include <type_traits>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.hpp>
#include <SushiBLAS/engine_fixture/host_engine_test.hpp>
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

        /** @brief Fixture for the Tensor data accessor tests, on a host engine. */
        class Small_TensorConstAccessTest : public HostEngineTest
        {

        };

        /**
         * @brief Tests that the const data() returns the allocation base of a tensor that owns its
         *        storage whole, the address the storage states independently of the accessor.
         */
        TEST_F(Small_TensorConstAccessTest, ConstDataIsTheStorageBase)
        {
            Tensor t = engine->create_tensor({4, 8}, Core::DataType::Float32);
            const Tensor& view = t;

            EXPECT_EQ(view.data(), static_cast<const void*>(t.storage->allocation_base()));
        }

        /** @brief Tests that the mutable data() returns the allocation base of the storage. */
        TEST_F(Small_TensorConstAccessTest, MutableDataIsTheStorageBase)
        {
            Tensor t = engine->create_tensor({4, 8}, Core::DataType::Float32);

            EXPECT_EQ(t.data(), t.storage->allocation_base());
        }

        /** @brief Tests that data_as<float>() in both forms is the allocation base. */
        TEST_F(Small_TensorConstAccessTest, TypedDataIsTheStorageBase)
        {
            Tensor t = engine->create_tensor({4, 8}, Core::DataType::Float32);
            const Tensor& view = t;

            EXPECT_EQ(static_cast<const void*>(view.data_as<float>()),
                      static_cast<const void*>(t.storage->allocation_base()));
            EXPECT_EQ(static_cast<void*>(t.data_as<float>()), t.storage->allocation_base());
        }

        /** @brief Tests that data_or_null() in both forms is the allocation base of a stored tensor. */
        TEST_F(Small_TensorConstAccessTest, DataOrNullIsTheStorageBaseOfAStoredTensor)
        {
            Tensor t = engine->create_tensor({4, 8}, Core::DataType::Float32);
            const Tensor& view = t;

            EXPECT_EQ(view.data_or_null(), static_cast<const void*>(t.storage->allocation_base()));
            EXPECT_EQ(t.data_or_null(), t.storage->allocation_base());
        }
    } // namespace Test
} // namespace SushiBLAS
