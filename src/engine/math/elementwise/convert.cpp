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
#include <SushiBLAS/engine.hpp>
#include <SushiBLAS/tensor.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/core/logger.hpp>
#include <SushiBLAS/support/launch_1d.hpp>
#include <SushiBLAS/engine/operand_check.hpp>
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
         * @param reads The read set; must be `src.region()`, never a bare pointer.
         * @param writes The write set; must be `dst.region()`, never a bare pointer.
         * @see src/README.md
         */
        template<typename Src, typename Dst>
        void record_convert(Engine& engine, const SushiRuntime::Graph::TaskMetadata& meta,
                            const std::vector<SushiRuntime::Core::ResourceRegion>& reads,
                            const std::vector<SushiRuntime::Core::ResourceRegion>& writes,
                            const void* src_ptr, void* dst_ptr, int64_t n)
        {
            engine.get_graph().add_task(meta, reads, writes,
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
         * @throws std::runtime_error if @p dst_dtype is outside `Domain::REAL`.
         * @see src/README.md
         */
        template<typename Src>
        void dispatch_destination(Core::DataType dst_dtype, Engine& engine,
                                  const SushiRuntime::Graph::TaskMetadata& meta,
                                  const std::vector<SushiRuntime::Core::ResourceRegion>& reads,
                                  const std::vector<SushiRuntime::Core::ResourceRegion>& writes,
                                  const void* src_ptr, void* dst_ptr, int64_t n)
        {
            switch (dst_dtype)
            {
                case Core::DataType::HALF:
                    record_convert<Src, sycl::half>(engine, meta, reads, writes, src_ptr, dst_ptr, n);
                    return;
                case Core::DataType::FLOAT32:
                    record_convert<Src, float>(engine, meta, reads, writes, src_ptr, dst_ptr, n);
                    return;
                case Core::DataType::FLOAT64:
                    record_convert<Src, double>(engine, meta, reads, writes, src_ptr, dst_ptr, n);
                    return;
                default:
                    break;
            }

            // Unreachable after the caller's dtype check, and it still throws; see src/README.md.
            SB_THROW_IF(true, "Elementwise operation '{}' does not support destination dtype {}.",
                        K_NAME, Core::to_string(dst_dtype));
        }
    } // namespace

    sycl::event ElementwiseOps::convert(const Tensor& src, Tensor& dst)
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
            return sycl::event();

        // Overlapping operands race between work-items, so they are refused; see src/README.md.
        SB_THROW_IF(src.region().overlaps(dst.region()),
                    "Elementwise operation '{}': the source and destination overlap in memory. "
                    "A converting copy reads {}-byte elements and writes {}-byte ones, so an "
                    "overlap is a race between work-items, not an in-place update.",
                    K_NAME, Core::bytes_per_element(src.dtype), Core::bytes_per_element(dst.dtype));

        const void* read_src = src.data();
        void* write_dst = dst.data();
        const int64_t n = src.num_elements;

        // The access sets hold Tensor::region(), never data(); see src/README.md.
        const std::vector<SushiRuntime::Core::ResourceRegion> reads = {src.region()};
        const std::vector<SushiRuntime::Core::ResourceRegion> writes = {dst.region()};

        SushiRuntime::Graph::TaskMetadata meta;
        meta.name = K_NAME;
        meta.task_type = SushiRuntime::Graph::TaskType::MATH_OP;
        meta.op_id = "math.ew.convert"_op;

        // The outer switch picks the source type; see src/README.md.
        switch (src.dtype)
        {
            case Core::DataType::HALF:
                dispatch_destination<sycl::half>(dst.dtype, engine_, meta, reads, writes, read_src, write_dst, n);
                return sycl::event();
            case Core::DataType::FLOAT32:
                dispatch_destination<float>(dst.dtype, engine_, meta, reads, writes, read_src, write_dst, n);
                return sycl::event();
            case Core::DataType::FLOAT64:
                dispatch_destination<double>(dst.dtype, engine_, meta, reads, writes, read_src, write_dst, n);
                return sycl::event();
            default:
                break;
        }

        SB_THROW_IF(true, "Elementwise operation '{}' does not support source dtype {}.",
                    K_NAME, Core::to_string(src.dtype));
        return sycl::event();
    }
} // namespace SushiBLAS
