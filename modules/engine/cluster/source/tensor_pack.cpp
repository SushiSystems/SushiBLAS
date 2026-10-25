/**************************************************************************/
/* tensor_pack.cpp                                                        */
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
 * @file tensor_pack.cpp
 * @brief Defines TensorPack: the shared allocation and the recorded gather and scatter.
 * @author Mustafa Garip
 */

#include <SushiBLAS/cluster/tensor_pack.hpp>

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>
#include <sycl/sycl.hpp>
#include <SushiBLAS/core/logger.hpp>
#include <SushiBLAS/engine/kernel_pair_dispatcher.hpp>
#include <SushiBLAS/engine/operand_set.hpp>
#include <SushiBLAS/engine/task_description.hpp>

namespace SushiBLAS
{
    namespace
    {
        /** @brief Names which way a recorded copy moves bytes. */
        enum class CopyDirection : std::uint8_t
        {
            IntoPack,  /**< From the tensors into the pack. */
            OutOfPack, /**< From the pack into the tensors. */
        };

        /** @brief Describes one copy between a tensor and its place in a pack. */
        struct CopyStep
        {
            void* destination = nullptr;      /**< Where the bytes go. */
            const void* source = nullptr;     /**< Where the bytes come from. */
            std::size_t size_bytes = 0;       /**< How many bytes move. */
        };

        /** @brief Holds the copies of one task, run in order on the task's queue. */
        struct CopyTask
        {
            /// @brief The copies, run in order.
            std::vector<CopyStep> steps;

            /** @brief Submits each copy after the one before it and returns the last event. */
            sycl::event operator()(const KernelPairDispatcher&, sycl::queue& queue,
                                   const std::vector<sycl::event>& dependencies) const
            {
                std::vector<sycl::event> previous = dependencies;
                sycl::event last;
                for (const CopyStep& step : steps)
                {
                    last = queue.submit([&](sycl::handler& handler)
                    {
                        handler.depends_on(previous);
                        handler.memcpy(step.destination, step.source, step.size_bytes);
                    });
                    previous.assign(1, last);
                }
                return last;
            }
        };

        /** @brief Records the copies between a pack and its tensors as one task. */
        TaskHandle record_copies(TaskRecorder& recorder, const PackLayout& layout,
                                 const Tensor& pack, SushiRuntime::span<const Tensor> tensors,
                                 CopyDirection direction, const char* name)
        {
            const std::vector<PackEntry>& entries = layout.entries();
            SB_THROW_IF(tensors.size() != entries.size(),
                        "{}: the pack holds {} tensors and {} were given", name, entries.size(),
                        tensors.size());

            std::byte* base = static_cast<std::byte*>(pack.storage->allocation_base());
            CopyTask task;
            task.steps.reserve(entries.size());
            OperandSet operands;
            if (direction == CopyDirection::IntoPack)
                operands.writes(pack);
            else
                operands.reads(pack);

            for (std::size_t index = 0; index < entries.size(); ++index)
            {
                const PackEntry& entry = entries[index];
                const Tensor& tensor = tensors[index];
                SB_THROW_IF(!PackLayout::matches(entry, tensor),
                            "{}: tensor {} differs from its entry in type, layout or shape", name,
                            index);
                if (entry.size_bytes == 0)
                    continue;

                SB_THROW_IF(!tensor.storage, "{}: tensor {} has no storage", name, index);
                SB_THROW_IF(!tensor.is_contiguous(), "{}: tensor {} is not contiguous", name,
                            index);
                SB_THROW_IF(tensor.storage->device_index() != pack.storage->device_index(),
                            "{}: tensor {} lives on device {} and the pack on device {}; a pack "
                            "copies within one device",
                            name, index, tensor.storage->device_index(),
                            pack.storage->device_index());

                std::byte* place = base + entry.offset_bytes;
                // Tensor::data() is const-qualified only on its view; the bytes are writable.
                void* tensor_bytes = const_cast<void*>(tensor.data());
                CopyStep step;
                step.size_bytes = entry.size_bytes;
                step.destination = direction == CopyDirection::IntoPack ? place : tensor_bytes;
                step.source = direction == CopyDirection::IntoPack ? tensor_bytes : place;
                task.steps.push_back(step);
                if (direction == CopyDirection::IntoPack)
                    operands.reads(tensor);
                else
                    operands.writes(tensor);
            }

            TaskDescription description;
            description.name = name;
            return recorder.record_pair(description, operands, "TensorPack", std::move(task));
        }
    } // namespace

    TensorPack::TensorPack(SushiRuntime::Runtime& runtime, PackLayout layout,
                           std::size_t device_index)
        : layout_(std::move(layout)),
          storage_(SushiRuntime::make_sushi<Storage>(runtime, layout_.size_bytes(),
                                                     SushiRuntime::Residency::Host,
                                                     device_index))
    {

    }

    Tensor TensorPack::tensor(std::size_t index) const
    {
        const PackEntry& entry = layout_.entries().at(index);
        // An entry lies at a multiple of its element size: its segment starts on 128 bytes.
        return Tensor(storage_,
                      SushiRuntime::span<const std::int64_t>(entry.shape.data(),
                                                             static_cast<std::size_t>(entry.rank)),
                      entry.dtype,
                      static_cast<std::int64_t>(entry.offset_bytes /
                                                Core::bytes_per_element(entry.dtype)),
                      entry.layout);
    }

    Tensor TensorPack::segment_tensor(Core::DataType dtype) const
    {
        const PackSegment& segment = layout_.segment(dtype);
        const std::int64_t dims[] = {static_cast<std::int64_t>(segment.element_count)};
        // A segment starts on 128 bytes, so on a multiple of its element size.
        return Tensor(storage_, SushiRuntime::span<const std::int64_t>(dims, 1), dtype,
                      static_cast<std::int64_t>(segment.offset_bytes /
                                                Core::bytes_per_element(dtype)));
    }

    TaskHandle TensorPack::record_gather(TaskRecorder& recorder,
                                         SushiRuntime::span<const Tensor> sources) const
    {
        return record_copies(recorder, layout_, as_operand(), sources, CopyDirection::IntoPack,
                             "sushiblas.pack.gather");
    }

    TaskHandle TensorPack::record_scatter(TaskRecorder& recorder,
                                          SushiRuntime::span<const Tensor> targets) const
    {
        return record_copies(recorder, layout_, as_operand(), targets, CopyDirection::OutOfPack,
                             "sushiblas.pack.scatter");
    }

    Tensor TensorPack::as_operand() const
    {
        // A layout's size is a multiple of K_PACK_SEGMENT_ALIGNMENT, so of sizeof(float).
        const std::int64_t dims[] = {
            static_cast<std::int64_t>(layout_.size_bytes() / sizeof(float))};
        return Tensor(storage_, SushiRuntime::span<const std::int64_t>(dims, 1),
                      Core::DataType::Float32);
    }
} // namespace SushiBLAS
