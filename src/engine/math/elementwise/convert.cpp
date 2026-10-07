/**************************************************************************/
/* convert.cpp                                                            */
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
 * @file convert.cpp
 * @brief Defines ElementwiseOps::convert, which records the dtype-converting copy between real
 * tensors.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <vector>
#include <sycl/sycl.hpp>
#include <SushiBLAS/tensor.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/core/logger.hpp>
#include <SushiBLAS/support/launch_1d.hpp>
#include <SushiBLAS/engine/operand_set.hpp>
#include <SushiBLAS/engine/task_handle.hpp>
#include <SushiBLAS/engine/operand_check.hpp>
#include <SushiBLAS/engine/task_recorder.hpp>
#include <SushiBLAS/engine/math/elementwise.hpp>
#include <SushiRuntime/core/resource_region.hpp>
#include <SushiRuntime/graph/task_types.hpp>

namespace SushiBLAS
{
    using namespace SushiRuntime::Graph::Literals;

    namespace
    {
        /** @brief The op's name, used for the task, the log line and every message. */
        constexpr const char* K_NAME = "math.ew.convert";

        /**
         * @brief Records the task for one concrete (source, destination) type pair.
         * @param operands Must declare the source as read and the destination as written.
         * @return The handle of the recorded task.
         * @throws std::invalid_argument under FORCE_REFERENCE.
         * @see src/README.md
         */
        template<typename Src, typename Dst>
        TaskHandle record_convert(TaskRecorder& recorder,
                                  const SushiRuntime::Graph::TaskMetadata& meta,
                                  const OperandSet& operands,
                                  const void* src_ptr, void* dst_ptr, int64_t n)
        {
            return recorder.record(meta, operands, "Elementwise operation",
                [n, pS = static_cast<const Src*>(src_ptr), pD = static_cast<Dst*>(dst_ptr)]
                (sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event
                {
                    SB_LOG_INFO("Elementwise convert: {} elements", n);

                    return q.submit([&](sycl::handler& h)
                    {
                        h.depends_on(deps);

                        // Indexed, because Launch::map needs one element type; see src/README.md.
                        Launch::indexed(h, q, static_cast<std::size_t>(n),
                            [pS, pD](std::size_t i)
                            {
                                // static_cast rounds to nearest even; see src/README.md.
                                pD[i] = static_cast<Dst>(pS[i]);
                            });
                    });
                });
        }

        /**
         * @brief Selects the destination type at runtime for a fixed source type.
         * @return The handle of the recorded task.
         * @throws std::runtime_error if @p dst_dtype is outside `Domain::REAL`.
         * @throws std::invalid_argument under FORCE_REFERENCE.
         * @see src/README.md
         */
        template<typename Src>
        TaskHandle dispatch_destination(Core::DataType dst_dtype, TaskRecorder& recorder,
                                        const SushiRuntime::Graph::TaskMetadata& meta,
                                        const OperandSet& operands,
                                        const void* src_ptr, void* dst_ptr, int64_t n)
        {
            switch (dst_dtype)
            {
                case Core::DataType::HALF:
                    return record_convert<Src, sycl::half>(recorder, meta, operands,
                                                           src_ptr, dst_ptr, n);
                case Core::DataType::FLOAT32:
                    return record_convert<Src, float>(recorder, meta, operands,
                                                      src_ptr, dst_ptr, n);
                case Core::DataType::FLOAT64:
                    return record_convert<Src, double>(recorder, meta, operands,
                                                       src_ptr, dst_ptr, n);
                default:
                    break;
            }

            // Unreachable after the caller's dtype check, and it still throws; see src/README.md.
            SB_THROW_IF(true, "Elementwise operation '{}' does not support destination dtype {}.",
                        K_NAME, Core::to_string(dst_dtype));
            return TaskHandle();
        }
    } // namespace

    TaskHandle ElementwiseOps::convert(const Tensor& src, Tensor& dst)
    {
        // The message names both dtypes; see src/README.md.
        SB_THROW_IF(!Core::is_real_dtype(src.dtype) || !Core::is_real_dtype(dst.dtype),
                    "Elementwise operation '{}' converts between the real floating-point "
                    "types only (HALF, FLOAT32, FLOAT64), and was asked for {} -> {}. A "
                    "complex operand has no single real value to narrow to, and a "
                    "float-to-integer conversion is a rounding policy (truncate? "
                    "round-half-even? saturate?) this layer has no basis to choose for "
                    "you - an op that does it belongs under a name that states which.",
                    K_NAME, Core::to_string(src.dtype), Core::to_string(dst.dtype));

        Dispatch::require_storage(src, K_NAME, "src");
        Dispatch::require_storage(dst, K_NAME, "dst");
        Dispatch::require_contiguous(src, K_NAME, "src");
        Dispatch::require_contiguous(dst, K_NAME, "dst");
        Dispatch::require_same_shape(src, dst, K_NAME, "src", "dst");
        Dispatch::require_same_layout(src, dst, K_NAME, "src", "dst");

        if (dst.num_elements == 0)
            return TaskHandle();

        // Overlapping operands race between work-items, so they are refused; see src/README.md.
        SB_THROW_IF(src.region().overlaps(dst.region()),
                    "Elementwise operation '{}': the source and destination overlap in memory. "
                    "A converting copy reads {}-byte elements and writes {}-byte ones, so an "
                    "overlap is a race between work-items, not an in-place update.",
                    K_NAME, Core::bytes_per_element(src.dtype), Core::bytes_per_element(dst.dtype));

        const void* read_src = src.data();
        void* write_dst = dst.data();
        const int64_t n = src.num_elements;

        OperandSet operands;
        operands.reads(src).writes(dst);

        SushiRuntime::Graph::TaskMetadata meta;
        meta.name = K_NAME;
        meta.task_type = SushiRuntime::Graph::TaskType::MATH_OP;
        meta.op_id = "math.ew.convert"_op;

        // The outer switch picks the source type; see src/README.md.
        switch (src.dtype)
        {
            case Core::DataType::HALF:
                return dispatch_destination<sycl::half>(dst.dtype, recorder_, meta, operands,
                                                        read_src, write_dst, n);
            case Core::DataType::FLOAT32:
                return dispatch_destination<float>(dst.dtype, recorder_, meta, operands,
                                                   read_src, write_dst, n);
            case Core::DataType::FLOAT64:
                return dispatch_destination<double>(dst.dtype, recorder_, meta, operands,
                                                    read_src, write_dst, n);
            default:
                break;
        }

        SB_THROW_IF(true, "Elementwise operation '{}' does not support source dtype {}.",
                    K_NAME, Core::to_string(src.dtype));
        return TaskHandle();
    }
} // namespace SushiBLAS
