/**************************************************************************/
/* constant.cpp                                                           */
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
 * @file constant.cpp
 * @brief Defines RandomOps::constant, which records the constant fill of a tensor.
 * @author Mustafa Garip
 */

#include <vector>
#include <complex>
#include <cstddef>
#include <algorithm>
#include <sycl/sycl.hpp>
#include <SushiBLAS/engine/operand_check.hpp>
#include <SushiBLAS/core/logger.hpp>
#include <SushiBLAS/engine/math/random.hpp>
#include <SushiBLAS/engine/math/random/dispatch.hpp>
#include <SushiBLAS/engine/kernel_pair_dispatcher.hpp>
#include <SushiRuntime/core/resource_region.hpp>
#include <SushiRuntime/graph/task_types.hpp>

namespace SushiBLAS 
{
    using namespace SushiRuntime::Graph::Literals;

    namespace
    {
        /**
         * @brief Records the fill of size elements of T with value, on the device or on the host.
         * @return The handle of the recorded task.
         * @throws UnsupportedExecution under ForceReference when the tensor is device-only USM.
         */
        template<typename T>
        TaskHandle record_fill(TaskRecorder& recorder, const SushiRuntime::Graph::TaskMetadata& meta,
                               const OperandSet& operands, T* out, T value, int64_t size)
        {
            return recorder.record_pair(meta, operands, "RNG operation",
                [out, value, size](const Internal::KernelPairDispatcher& pair, sycl::queue& q,
                                   const std::vector<sycl::event>& deps) -> sycl::event
                {
                    SB_LOG_INFO("RandomOps: constant ({} elements)", size);

                    const std::size_t count = static_cast<std::size_t>(size);
                    return pair.run(q, deps,
                        [out, value, count] { std::fill_n(out, count, value); },
                        [out, value, count](sycl::queue& q2,
                                            const std::vector<sycl::event>& deps2) -> sycl::event
                        {
                            return q2.fill(out, value, count, deps2);
                        });
                });
        }
    } // namespace

    TaskHandle RandomOps::constant(Tensor& t, double value) 
    {
        auto op_id = "random.constant"_op;
        SushiRuntime::Graph::TaskMetadata meta;
        meta.name = "random.constant";
        meta.task_type = SushiRuntime::Graph::TaskType::MATH_OP;
        meta.op_id = op_id;
        meta.set_param(0, value);

        Dispatch::require_random_dtype(t, Dispatch::Domain::All, Dispatch::Sampling::MovesValues, "random.constant");
        Dispatch::require_storage(t, "random.constant", "t");
        Dispatch::require_device_dtype(t, "random.constant", "t");
        Dispatch::require_contiguous(t, "random.constant", "t");

        const int64_t size = t.num_elements;

        OperandSet operands;
        operands.writes(t);

        switch (t.dtype)
        {
            case Core::DataType::Half:
                return record_fill(recorder_, meta, operands, t.data_as<sycl::half>(),
                                   static_cast<sycl::half>(value), size);
            case Core::DataType::Float32:
                return record_fill(recorder_, meta, operands, t.data_as<float>(),
                                   static_cast<float>(value), size);
            case Core::DataType::Float64:
                return record_fill(recorder_, meta, operands, t.data_as<double>(), value, size);
            case Core::DataType::Complex32:
                return record_fill(recorder_, meta, operands, t.data_as<std::complex<float>>(),
                                   std::complex<float>(static_cast<float>(value), 0.0f), size);
            case Core::DataType::Complex64:
                return record_fill(recorder_, meta, operands, t.data_as<std::complex<double>>(),
                                   std::complex<double>(value, 0.0), size);
            // An integer tensor stores the truncated double; see src/README.md.
            case Core::DataType::Int32:
                return record_fill(recorder_, meta, operands, t.data_as<int32_t>(),
                                   static_cast<int32_t>(value), size);
            case Core::DataType::Int64:
                return record_fill(recorder_, meta, operands, t.data_as<int64_t>(),
                                   static_cast<int64_t>(value), size);
            default:
                SB_THROW_IF(true, "Operation 'random.constant' does not support dtype {}.",
                            Core::to_string(t.dtype));
        }

        return TaskHandle();
    }
} // namespace SushiBLAS
