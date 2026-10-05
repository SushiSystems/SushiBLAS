/****************************************************************/
/* utils.hpp                                                    */
/* SushiBLAS - https://github.com/SushiSystems/SushiBLAS        */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

/**
 * @file utils.hpp
 * @brief Declares the BLAS helpers that read vector parameters and record a typed BLAS task.
 * @author Mustafa Garip
 */

#pragma once

#include <vector>
#include <complex>
#include <utility>
#include <sycl/sycl.hpp>
#include <SushiBLAS/engine.hpp>
#include <SushiBLAS/tensor.hpp>
#include <SushiBLAS/core/traits.hpp>
#include <SushiBLAS/engine/task.hpp>
#include <SushiBLAS/engine/dispatch.hpp>
#include <SushiRuntime/core/resource_region.hpp>
#include <SushiRuntime/graph/task_types.hpp>

namespace SushiBLAS
{
    namespace Internal
    {
        /**
         * @brief Helper to extract 1D vector parameters (size and increment) from a Tensor.
         *
         * @param t The input tensor.
         * @param n Output: Number of elements.
         * @param inc Output: Increment (stride).
         */
        inline void get_vec_params(const Tensor& t, int64_t& n, int64_t& inc)
        {
            n = t.num_elements;
            if (t.is_contiguous())
            {
                inc = 1;
            }
            else
            {
                SB_THROW_IF(t.rank > 1, "Level 1 BLAS expects contiguous memory for rank > 1 tensors.");
                inc = t.strides[0];
            }
        }

        /**
         * @brief Records one dtype-dispatched BLAS task on the engine's graph.
         * @param op_func Callable taking a scalar of the element type, the queue and the dependencies.
         * @throws std::runtime_error at record time if dtype falls outside Dom.
         * @see include/SushiBLAS/engine/blas/README.md
         */
        template<Dispatch::Domain Dom = Dispatch::Domain::REAL_AND_COMPLEX, typename Func>
        sycl::event execute_blas_task(Engine& engine, const SushiRuntime::Graph::TaskMetadata& meta,
                                       const std::vector<SushiRuntime::Core::ResourceRegion>& reads,
                                       const std::vector<SushiRuntime::Core::ResourceRegion>& writes,
                                       Core::DataType dtype, Func&& op_func)
        {
            return execute_typed_task<Dom>(engine, meta, reads, writes, dtype, "BLAS operation",
                                           std::forward<Func>(op_func));
        }
    }
}
