/**************************************************************************/
/* ordered_sum.cpp                                                        */
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
 * @file ordered_sum.cpp
 * @brief Defines ElementwiseOps::ordered_sum, which records one kernel that adds n tensors in index order.
 * @author Mustafa Garip
 */

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include <sycl/sycl.hpp>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/core/logger.hpp>
#include <SushiBLAS/elementwise/elementwise_log_category.hpp>
#include <SushiBLAS/launch/launch_1d.hpp>
#include <SushiBLAS/reference/serial_1d.hpp>
#include <SushiBLAS/engine/dispatch.hpp>
#include <SushiBLAS/engine/operand_set.hpp>
#include <SushiBLAS/engine/task_handle.hpp>
#include <SushiBLAS/engine/operand_check.hpp>
#include <SushiBLAS/engine/task_recorder.hpp>
#include <SushiBLAS/elementwise/elementwise.hpp>
#include <SushiBLAS/elementwise/ordered_sum_element.hpp>
#include <SushiBLAS/engine/kernel_pair_dispatcher.hpp>
#include <SushiBLAS/engine/task_description.hpp>
#include <SushiRuntime/core/operation_id.hpp>

namespace SushiBLAS
{
    using namespace SushiRuntime::Literals;

    namespace
    {
        /** @brief Names the operation in the task, the log line and every message. */
        constexpr const char* K_NAME = "math.ew.ordered_sum";

        /** @brief Holds the capacity of the input table a kernel carries by value. */
        constexpr std::size_t K_CAPACITY = Dispatch::Detail::K_ORDERED_SUM_CAPACITY;

        /** @brief Holds the addresses and weights of one recorded sum, before the element type is known. */
        struct SumTable
        {
            /** @brief Holds the first element of each input, in index order. */
            std::array<const void*, K_CAPACITY> inputs{};
            /** @brief Holds the weight of each input as the caller gave it, in index order. */
            std::array<double, K_CAPACITY> weights{};
            /** @brief Holds the first element of the output. */
            void* output = nullptr;
            /** @brief Holds the number of inputs. */
            std::size_t input_count = 0;
            /** @brief Holds the number of elements of every tensor. */
            std::size_t element_count = 0;
        };

        /**
         * @brief Records the task for one concrete element type and returns its handle.
         * @throws UnsupportedExecution under ForceReference when an operand is device-only USM.
         */
        template<typename T>
        TaskHandle record_sum(TaskRecorder& recorder, const TaskDescription& description,
                              const OperandSet& operands, const SumTable& table)
        {
            using Element = Dispatch::Detail::OrderedSumElement<T>;
            Element element{};
            for (std::size_t k = 0; k < table.input_count; ++k)
            {
                element.inputs[k] = static_cast<const T*>(table.inputs[k]);
                element.weights[k] = static_cast<typename Element::Weight>(table.weights[k]);
            }
            element.output = static_cast<T*>(table.output);
            element.input_count = table.input_count;
            const std::size_t count = table.element_count;

            return recorder.record_pair(description, operands, "Elementwise operation",
                [count, element](const KernelPairDispatcher& pair, sycl::queue& q,
                                 const std::vector<sycl::event>& deps) -> sycl::event
                {
                    SB_LOG_TRACE(K_ELEMENTWISE_LOG_CATEGORY,
                                 "Elementwise ordered_sum: {} elements over {} inputs", count,
                                 element.input_count);

                    return pair.run(q, deps,
                        [count, element] { Serial::indexed(count, element); },
                        [count, element](sycl::queue& q2,
                                         const std::vector<sycl::event>& deps2) -> sycl::event
                        {
                            return q2.submit([&](sycl::handler& h)
                            {
                                h.depends_on(deps2);
                                Launch::indexed(h, q2, count, element);
                            });
                        });
                });
        }

        /** @brief Records the sum for the element type a with_element_type arm names. */
        struct RecordArm
        {
            /** @brief Holds the recorder the task goes to. */
            TaskRecorder& recorder;
            /** @brief Holds the name and operation of the task. */
            const TaskDescription& description;
            /** @brief Holds the reads and the write of the task. */
            const OperandSet& operands;
            /** @brief Holds the addresses and weights the kernel carries. */
            const SumTable& table;
            /** @brief Receives the handle of the recorded task. */
            TaskHandle& handle;

            /** @brief Records the task for the element type the tag names. */
            template<typename Tag>
            void operator()(Tag) const
            {
                handle = record_sum<typename Tag::type>(recorder, description, operands, table);
            }
        };
    } // namespace

    TaskHandle ElementwiseOps::ordered_sum(SushiRuntime::span<const Tensor> inputs,
                                           SushiRuntime::span<const double> weights, Tensor& out)
    {
        Dispatch::require_operand(!inputs.empty(), K_NAME, "inputs",
                                  "is empty; a sum needs at least one input");
        Dispatch::require_operand(inputs.size() <= K_CAPACITY, K_NAME, "inputs",
                                  "holds {} tensors; a sum takes at most {}", inputs.size(),
                                  K_CAPACITY);
        Dispatch::require_operand(weights.size() == inputs.size(), K_NAME, "weights",
                                  "holds {} weights for {} inputs", weights.size(), inputs.size());

        const Tensor& first = inputs[0];
        for (std::size_t k = 0; k < inputs.size(); ++k)
        {
            const std::string operand = "inputs[" + std::to_string(k) + "]";
            Dispatch::require_storage(inputs[k], K_NAME, operand.c_str());
            Dispatch::require_device_dtype(inputs[k], K_NAME, operand.c_str());
            Dispatch::require_contiguous(inputs[k], K_NAME, operand.c_str());
            Dispatch::require_dtype(inputs[k], first.dtype, K_NAME, operand.c_str());
            Dispatch::require_same_shape(inputs[k], first, K_NAME, operand.c_str(), "inputs[0]");
            Dispatch::require_same_layout(inputs[k], first, K_NAME, operand.c_str(), "inputs[0]");
        }

        Dispatch::require_storage(out, K_NAME, "out");
        Dispatch::require_device_dtype(out, K_NAME, "out");
        Dispatch::require_contiguous(out, K_NAME, "out");
        Dispatch::require_dtype(out, first.dtype, K_NAME, "out");
        Dispatch::require_same_shape(out, first, K_NAME, "out", "inputs[0]");
        Dispatch::require_same_layout(out, first, K_NAME, "out", "inputs[0]");

        if (Core::is_integer_dtype(first.dtype))
        {
            for (std::size_t k = 0; k < weights.size(); ++k)
            {
                const std::string operand = "weights[" + std::to_string(k) + "]";
                Dispatch::require_operand(weights[k] == 1.0, K_NAME, operand.c_str(),
                                          "is {}; an integer sum adds exactly and takes weight 1",
                                          weights[k]);
            }
        }

        if (out.num_elements == 0)
            return TaskHandle();

        for (std::size_t k = 0; k < inputs.size(); ++k)
        {
            const std::string operand = "inputs[" + std::to_string(k) + "]";
            Dispatch::require_operand(!inputs[k].region().overlaps(out.region()), K_NAME,
                                      "out", "overlaps '{}' in memory; the sum cannot run in place",
                                      operand);
        }

        SumTable table;
        OperandSet operands;
        for (std::size_t k = 0; k < inputs.size(); ++k)
        {
            table.inputs[k] = inputs[k].data();
            table.weights[k] = weights[k];
            operands.reads(inputs[k]);
        }
        operands.writes(out);
        table.output = out.data();
        table.input_count = inputs.size();
        table.element_count = static_cast<std::size_t>(out.num_elements);

        TaskDescription description;
        description.name = K_NAME;
        description.operation = "math.ew.ordered_sum"_op;

        TaskHandle handle;
        Dispatch::with_element_type<Dispatch::Domain::All>(
            first.dtype, "Elementwise operation", K_NAME,
            RecordArm{recorder_, description, operands, table, handle});
        return handle;
    }
} // namespace SushiBLAS
