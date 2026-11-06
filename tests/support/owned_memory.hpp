/**************************************************************************/
/* owned_memory.hpp                                                       */
/**************************************************************************/
/*                          This file is part of:                         */
/*                               SushiSkills                              */
/*               https://github.com/SushiSystems/SushiSkills              */
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
 * @file owned_memory.hpp
 * @brief Declares OwnedMemory, shared USM that stands in for memory another owner keeps alive.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <SushiRuntime/sushiruntime.hpp>
#include <SushiBLAS/test_runner/test_environment.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /** @brief Holds USM that stands in for memory another owner keeps alive. */
        class OwnedMemory final
        {
            public:
                /** @brief Allocates n_bytes of shared USM on the queue of the first device. */
                explicit OwnedMemory(std::size_t n_bytes)
                    : address_(sycl::malloc_shared<std::byte>(n_bytes, TestEnvironment::queue(0))),
                      token_(SushiRuntime::make_liveness())
                {

                }

                OwnedMemory(const OwnedMemory&) = delete;
                OwnedMemory& operator=(const OwnedMemory&) = delete;

                /** @brief Frees the USM. */
                ~OwnedMemory()
                {
                    sycl::free(address_, TestEnvironment::queue(0));
                }

                /** @brief Returns the first byte. */
                [[nodiscard]] std::byte* data() const noexcept
                {
                    return address_;
                }

                /** @brief Returns count bytes from offset, borrowed through Interop. */
                [[nodiscard]] SushiRuntime::Buffer<std::byte> borrow(std::size_t offset,
                                                                     std::size_t count) const
                {
                    return TestEnvironment::runtime().interop().borrow<std::byte>(
                        address_ + offset, count, SushiRuntime::DeviceIndex{0},
                        SushiRuntime::Residency::Shared, token_).value();
                }

                /** @brief Lets go of the memory, as its owner does when it frees it. */
                void release()
                {
                    token_.reset();
                }

            private:
                /** @brief The USM. */
                std::byte* address_;

                /** @brief Lives as long as the memory is valid. */
                SushiRuntime::LivenessToken token_;
        };
    } // namespace Test
} // namespace SushiBLAS
