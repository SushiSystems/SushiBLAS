/**************************************************************************/
/* utils.hpp                                                              */
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
 * @file utils.hpp
 * @brief Declares the BLAS helpers that read vector parameters and record a typed BLAS task.
 * @author Mustafa Garip
 */

#pragma once

#include <vector>
#include <cstddef>
#include <complex>
#include <utility>
#include <sycl/sycl.hpp>
#include <SushiBLAS/engine.hpp>
#include <SushiBLAS/tensor.hpp>
#include <SushiBLAS/core/traits.hpp>
#include <SushiBLAS/engine/task.hpp>
#include <SushiBLAS/engine/scratch.hpp>
#include <SushiBLAS/engine/dispatch.hpp>
#include <SushiBLAS/engine/operand_check.hpp>
#include <SushiBLAS/support/reduce_1d.hpp>
#include <SushiBLAS/support/device_policy.hpp>
#include <SushiRuntime/core/resource_region.hpp>
#include <SushiRuntime/graph/task_types.hpp>

namespace SushiBLAS
{
    namespace Internal
    {
        /**
         * @brief Reads the element count and the positive increment of a vector operand.
         * @param n Receives the number of elements.
         * @param inc Receives the distance between consecutive elements.
         * @throws std::runtime_error on a non-contiguous tensor of rank > 1.
         * @throws std::invalid_argument on missing storage or a rank-1 stride of zero or less.
         */
        inline void get_vec_params(const Tensor& t, int64_t& n, int64_t& inc, const char* op,
                                   const char* operand)
        {
            Dispatch::require_storage(t, op, operand);

            n = t.num_elements;
            if (t.is_contiguous())
            {
                inc = 1;
            }
            else
            {
                SB_THROW_IF(t.rank > 1, "Level 1 BLAS expects contiguous memory for rank > 1 tensors.");
                if (t.strides[0] <= 0)
                    Dispatch::Detail::throw_operand_error(op, operand,
                                                          "has increment {}; a vector increment "
                                                          "must be positive",
                                                          t.strides[0]);
                inc = t.strides[0];
            }
        }

        /**
         * @brief Throws unless a Level 2 matrix operand has rank 2 or less.
         * @throws std::invalid_argument on a rank above 2.
         */
        inline void require_level2_matrix_rank(const Tensor& A, const char* op)
        {
            if (A.rank > 2)
                Dispatch::Detail::throw_operand_error(op, "A",
                                                      "has rank {}; a Level 2 matrix has rank 2",
                                                      A.rank);
        }

        /**
         * @brief Returns the dtype a norm or absolute-sum result holds for an operand of dtype.
         * @return The real component type of a complex dtype, dtype itself otherwise.
         */
        [[nodiscard]] inline constexpr Core::DataType real_result_dtype(Core::DataType dtype)
        {
            if (dtype == Core::DataType::COMPLEX32)
                return Core::DataType::FLOAT32;
            if (dtype == Core::DataType::COMPLEX64)
                return Core::DataType::FLOAT64;
            return dtype;
        }

        /**
         * @brief Throws unless a scalar result tensor has storage, the expected dtype and exactly one element.
         * @throws std::invalid_argument naming the first rule that fails.
         */
        inline void require_scalar_result(const Tensor& result, Core::DataType expected,
                                          const char* op)
        {
            Dispatch::require_storage(result, op, "result");
            Dispatch::require_dtype(result, expected, op, "result");
            if (result.num_elements != 1)
                Dispatch::Detail::throw_operand_error(op, "result",
                                                      "holds {} elements; a scalar result holds "
                                                      "exactly one",
                                                      result.num_elements);
        }

        /**
         * @brief Allocates the partials scratch a Level 1 reduction over x plans on x's device,
         *        and appends its write region to writes.
         * @param elements Values the reduction folds: n, or 2n for a complex nrm2.
         * @param partial_bytes Bytes one partial takes.
         * @return The scratch, empty when x has no storage or nothing is planned.
         */
        [[nodiscard]] inline ScratchBuffer allocate_level1_scratch(
            Engine& engine, const Tensor& x, std::size_t elements, std::size_t partial_bytes,
            std::vector<SushiRuntime::Core::ResourceRegion>& writes)
        {
            if (!x.storage || elements == 0)
                return ScratchBuffer();

            const std::size_t partials =
                Reduce::planned_partials(Device::profile_for(x.get_device()), elements);
            ScratchBuffer scratch(engine, partials * partial_bytes, x.storage->get_device_index());
            if (!scratch.is_empty())
                writes.push_back(scratch.region());
            return scratch;
        }

        /**
         * @brief Records one dtype-dispatched BLAS task whose body runs a reference and SYCL kernel pair.
         * @param host_operands Every pointer the reference kernel reads or writes.
         * @param op_func Callable taking a scalar of the element type, a const KernelPairDispatcher&,
         *        the queue and the dependencies.
         * @throws std::runtime_error at record time if dtype falls outside Dom.
         * @throws std::invalid_argument at record time, as require_host_reachable states.
         * @see include/SushiBLAS/engine/blas/README.md
         */
        template<Dispatch::Domain Dom = Dispatch::Domain::REAL_AND_COMPLEX, typename Func>
        sycl::event execute_blas_task(Engine& engine, const SushiRuntime::Graph::TaskMetadata& meta,
                                       const std::vector<SushiRuntime::Core::ResourceRegion>& reads,
                                       const std::vector<SushiRuntime::Core::ResourceRegion>& writes,
                                       Core::DataType dtype, const ReferenceOperands& host_operands,
                                       Func&& op_func)
        {
            return execute_typed_task<Dom>(engine, meta, reads, writes, dtype, "BLAS operation",
                                           host_operands, std::forward<Func>(op_func));
        }
    }
}
