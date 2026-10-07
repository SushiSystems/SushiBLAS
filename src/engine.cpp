/**************************************************************************/
/* engine.cpp                                                             */
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
 * @file engine.cpp
 * @brief Defines the Engine constructor, its mode setter and its tensor creation.
 * @author Mustafa Garip
 */

#include <SushiBLAS/engine.hpp>

#include <vector>
#include <stdexcept>

namespace SushiBLAS
{
    namespace
    {
        /**
         * @brief Returns the byte count a dense tensor of dims and dtype needs.
         * @throws std::runtime_error on a negative dimension or an overflowing count.
         */
        [[nodiscard]] size_t checked_allocation_bytes(const std::vector<int64_t>& dims,
                                                      Core::DataType dtype)
        {
            SB_THROW_IF(dims.size() > Core::MAX_TENSOR_RANK, "Rank {} exceeds limit of {}",
                        dims.size(), Core::MAX_TENSOR_RANK);

            int64_t elements = 1;
            size_t axis = 0;

            for (const int64_t d : dims)
            {
                SB_THROW_IF(d < 0, "Dimension {} is negative ({})", axis, d);
                elements = Detail::checked_multiply(elements, d, "Element count");
                ++axis;
            }

            const int64_t bpe = static_cast<int64_t>(Core::bytes_per_element(dtype));
            return static_cast<size_t>(Detail::checked_multiply(elements, bpe, "Byte size"));
        }
    } // namespace

    Engine::Engine(SushiRuntime::Execution::RuntimeContext& ctx, Core::Layout layout)
        : context_(ctx), graph_(ctx), default_layout_(layout)
    {
        SB_LOG_INFO("SushiBLAS Engine initialized with {} layout.",
                    layout == Core::Layout::ROW_MAJOR ? "Row-Major" : "Column-Major");
    }

    void Engine::set_mode(ExecutionMode mode)
    {
        if (mode == ExecutionMode::VERIFY)
            throw std::logic_error("ExecutionMode::VERIFY is not implemented. Run the operation "
                                   "once under ExecutionMode::FORCE_REFERENCE and once under "
                                   "ExecutionMode::FORCE_SYCL and compare the results.");
        mode_ = mode;
    }

    Tensor Engine::create_tensor(std::initializer_list<int64_t> dims,
                                 SushiRuntime::API::Residency residency)
    {
        return create_tensor(std::vector<int64_t>(dims), Core::DataType::FLOAT32, residency);
    }

    Tensor Engine::create_tensor(std::initializer_list<int64_t> dims,
                                 Core::DataType dtype,
                                 SushiRuntime::API::Residency residency)
    {
        return create_tensor(std::vector<int64_t>(dims), dtype, residency);
    }

    Tensor Engine::create_tensor(const std::vector<int64_t>& dims,
                                 SushiRuntime::API::Residency residency)
    {
        return create_tensor(dims, Core::DataType::FLOAT32, residency);
    }

    Tensor Engine::create_tensor(const std::vector<int64_t>& dims,
                                 Core::DataType dtype,
                                 SushiRuntime::API::Residency residency)
    {
        const size_t bytes = checked_allocation_bytes(dims, dtype);

        auto storage = SushiRuntime::make_sushi<Storage>(context_, bytes, residency);

        return Tensor(storage, SushiRuntime::span<const int64_t>(dims.data(), dims.size()), dtype,
                      0, default_layout_);
    }

} // namespace SushiBLAS
