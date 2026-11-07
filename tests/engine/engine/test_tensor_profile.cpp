/**************************************************************************/
/* test_tensor_profile.cpp                                                */
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


/**
 * @file test_tensor_profile.cpp
 * @brief Tests that profile_of() gives a host tensor the host Profile and refuses a tensor with no storage.
 * @author Mustafa Garip
 */

#include <stdexcept>
#include <gtest/gtest.h>
#include <SushiBLAS/engine/engine.hpp>
#include <SushiBLAS/engine/tensor_profile.hpp>
#include <SushiBLAS/launch/device_policy.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /** @brief Tests that a tensor a host engine made is sized for the host Profile, with no device asked. */
        TEST(Small_TensorProfile, AHostTensorGetsTheHostProfile)
        {
            Engine engine(Engine::Host{});
            const Tensor tensor = engine.create_tensor({4});

            EXPECT_EQ(&Dispatch::profile_of(tensor), &Device::host_profile());
        }

        /** @brief Tests that a tensor with no storage has no Profile. */
        TEST(Small_TensorProfile, ATensorWithNoStorageIsRefused)
        {
            const Tensor shape_only;

            EXPECT_THROW((void)Dispatch::profile_of(shape_only), std::runtime_error);
        }
    } // namespace Test
} // namespace SushiBLAS
