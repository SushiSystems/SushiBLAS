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

#include <complex>
#include <sycl/sycl.hpp>
#include <SushiBLAS/engine/operand_check.hpp>
#include <SushiBLAS/core/logger.hpp>
#include <SushiBLAS/engine/math/random.hpp>
#include <SushiBLAS/engine/math/random/dispatch.hpp>
#include <SushiRuntime/core/resource_region.hpp>
#include <SushiRuntime/graph/task_types.hpp>

namespace SushiBLAS 
{
    using namespace SushiRuntime::Graph::Literals;

    TaskHandle RandomOps::constant(Tensor& t, double value) 
    {
        auto op_id = "random.constant"_op;
        SushiRuntime::Graph::TaskMetadata meta;
        meta.name = "random.constant";
        meta.task_type = SushiRuntime::Graph::TaskType::MATH_OP;
        meta.op_id = op_id;
        meta.set_param(0, value);

        Dispatch::require_random_dtype(t, Dispatch::Domain::ALL, Dispatch::Sampling::MOVES_VALUES, "random.constant");
        Dispatch::require_storage(t, "random.constant", "t");
        Dispatch::require_contiguous(t, "random.constant", "t");

        const int64_t size = t.num_elements;

        OperandSet operands;
        operands.writes(t);

        switch (t.dtype)
        {
            case Core::DataType::HALF:
            {
                return recorder_.record(meta, operands, "RNG operation",
                    [size, value, pT = t.data_as<sycl::half>()](sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event
                    {
                        SB_LOG_INFO("RandomOps: constant ({} elements, value: {:.4f})", size, value);
                        return q.fill(pT, static_cast<sycl::half>(value), size, deps);
                    });
            }
            case Core::DataType::FLOAT32:
            {
                return recorder_.record(meta, operands, "RNG operation",
                    [size, value, pT = t.data_as<float>()](sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event
                    {
                        SB_LOG_INFO("RandomOps: constant ({} elements, value: {:.4f})", size, value);
                        return q.fill(pT, static_cast<float>(value), size, deps);
                    });
            }
            case Core::DataType::FLOAT64:
            {
                return recorder_.record(meta, operands, "RNG operation",
                    [size, value, pT = t.data_as<double>()](sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event
                    {
                        SB_LOG_INFO("RandomOps: constant ({} elements, value: {:.4f})", size, value);
                        return q.fill(pT, value, size, deps);
                    });
            }
            case Core::DataType::COMPLEX32:
            {
                return recorder_.record(meta, operands, "RNG operation",
                    [size, value, pT = t.data_as<std::complex<float>>()](sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event
                    {
                        SB_LOG_INFO("RandomOps: constant complex32 ({} elements, value: {:.4f})", size, value);
                        return q.fill(pT, std::complex<float>(static_cast<float>(value), 0.0f), size, deps);
                    });
            }
            case Core::DataType::COMPLEX64:
            {
                return recorder_.record(meta, operands, "RNG operation",
                    [size, value, pT = t.data_as<std::complex<double>>()](sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event
                    {
                        SB_LOG_INFO("RandomOps: constant complex64 ({} elements, value: {:.4f})", size, value);
                        return q.fill(pT, std::complex<double>(value, 0.0), size, deps);
                    });
            }
            // An integer tensor stores the truncated double; see src/README.md.
            case Core::DataType::INT32:
            {
                return recorder_.record(meta, operands, "RNG operation",
                    [size, value, pT = t.data_as<int32_t>()](sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event
                    {
                        SB_LOG_INFO("RandomOps: constant int32 ({} elements, value: {:.4f})", size, value);
                        return q.fill(pT, static_cast<int32_t>(value), size, deps);
                    });
            }
            case Core::DataType::INT64:
            {
                return recorder_.record(meta, operands, "RNG operation",
                    [size, value, pT = t.data_as<int64_t>()](sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event
                    {
                        SB_LOG_INFO("RandomOps: constant int64 ({} elements, value: {:.4f})", size, value);
                        return q.fill(pT, static_cast<int64_t>(value), size, deps);
                    });
            }
            default:
                SB_THROW_IF(true, "Operation 'random.constant' does not support dtype {}.",
                            Core::to_string(t.dtype));
        }

        return TaskHandle();
    }
} // namespace SushiBLAS
