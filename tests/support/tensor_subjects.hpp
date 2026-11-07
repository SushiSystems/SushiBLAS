/**************************************************************************/
/* tensor_subjects.hpp                                                    */
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
 * @file tensor_subjects.hpp
 * @brief Declares the subjects a StorageBlock and a TensorFactory contract runs on: the host adapter and the shared runtime.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <memory>
#include <SushiRuntime/sushiruntime.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/tensor/host_block.hpp>
#include <SushiBLAS/tensor/runtime_block.hpp>
#include <SushiBLAS/tensor/storage.hpp>
#include <SushiBLAS/tensor/storage_block.hpp>
#include <SushiBLAS/tensor/tensor_factory.hpp>
#include <SushiBLAS/test_runner/test_environment.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /// @brief The device count of the host adapter, which has one.
        inline constexpr std::size_t K_HOST_DEVICE_COUNT = 1;

        /** @brief Builds StorageBlocks of host memory through HostBlockMaker. */
        struct HostBlockSubject
        {
            /** @brief Returns a block of n_bytes. */
            [[nodiscard]] std::unique_ptr<StorageBlock> block_of(std::size_t n_bytes) const
            {
                return HostBlockMaker().make(n_bytes, SushiRuntime::Residency::Host, 0);
            }

            /** @brief Returns the residency a block of this adapter reports. */
            [[nodiscard]] static SushiRuntime::Residency residency() noexcept
            {
                return SushiRuntime::Residency::Host;
            }

            /** @brief Returns whether a block of this adapter has a queue. */
            [[nodiscard]] static bool has_queue() noexcept { return false; }
        };

        /** @brief Builds StorageBlocks of runtime buffers through RuntimeBlockMaker. */
        struct RuntimeBlockSubject
        {
            /** @brief Returns a block of n_bytes on the first device. */
            [[nodiscard]] std::unique_ptr<StorageBlock> block_of(std::size_t n_bytes) const
            {
                return RuntimeBlockMaker(TestEnvironment::runtime())
                    .make(n_bytes, SushiRuntime::Residency::Shared, 0);
            }

            /** @brief Returns the residency a block of this adapter reports. */
            [[nodiscard]] static SushiRuntime::Residency residency() noexcept
            {
                return SushiRuntime::Residency::Shared;
            }

            /** @brief Returns whether a block of this adapter has a queue. */
            [[nodiscard]] static bool has_queue() noexcept { return true; }
        };

        /** @brief Builds storages and factories on the host, with no runtime. */
        struct HostTensorSubject
        {
            /// @brief The factory under test.
            TensorFactory factory;

            /// @brief The layout the factory is built with.
            static constexpr Core::Layout K_LAYOUT = Core::Layout::ColumnMajor;

            /** @brief Builds the factory with no runtime. */
            HostTensorSubject() : factory(TensorFactory::Host{}, K_LAYOUT) {}

            /** @brief Returns a storage of n_bytes. */
            [[nodiscard]] SushiRuntime::sushi_ptr<Storage> storage_of(std::size_t n_bytes) const
            {
                return make_host_storage(n_bytes);
            }

            /** @brief Returns the residency a storage of this adapter reports. */
            [[nodiscard]] static SushiRuntime::Residency residency() noexcept
            {
                return SushiRuntime::Residency::Host;
            }

            /** @brief Returns the number of devices the adapter has. */
            [[nodiscard]] static std::size_t device_count() noexcept
            {
                return K_HOST_DEVICE_COUNT;
            }
        };

        /** @brief Builds storages and factories on the runtime the test program made. */
        struct RuntimeTensorSubject
        {
            /// @brief The layout the factory is built with.
            static constexpr Core::Layout K_LAYOUT = Core::Layout::ColumnMajor;

            /// @brief The factory under test.
            TensorFactory factory{TestEnvironment::runtime(), K_LAYOUT};

            /** @brief Returns a storage of n_bytes on the first device. */
            [[nodiscard]] SushiRuntime::sushi_ptr<Storage> storage_of(std::size_t n_bytes) const
            {
                return make_runtime_storage(TestEnvironment::runtime(), n_bytes);
            }

            /** @brief Returns the residency a storage of this adapter reports. */
            [[nodiscard]] static SushiRuntime::Residency residency() noexcept
            {
                return SushiRuntime::Residency::Shared;
            }

            /** @brief Returns the number of devices the adapter has. */
            [[nodiscard]] static std::size_t device_count()
            {
                return TestEnvironment::device_count();
            }
        };

    } // namespace Test
} // namespace SushiBLAS
