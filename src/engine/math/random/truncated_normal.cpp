/****************************************************************/
/* truncated_normal.cpp                                         */
/* SushiBLAS - https://github.com/SushiSystems/SushiBLAS        */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

#include <SushiBLAS/engine.hpp>
#include <SushiBLAS/core/logger.hpp>
#include <SushiBLAS/engine/math/random.hpp>
#include <SushiRuntime/graph/task_types.hpp>
#include <SushiBLAS/engine/math/random/dispatch.hpp>

namespace SushiBLAS 
{
    using namespace SushiRuntime::Graph::Literals;

    sycl::event RandomOps::truncated_normal(Tensor& t, double mean, double stddev, double a, double b) 
    {
        (void)a;
        (void)b;
        SB_LOG_INFO("RandomOps: truncated_normal ({} elements, mean: {:.4f}, stddev: {:.4f}) - Falling back to normal", (uint64_t)t.num_elements, mean, stddev);
        // TODO: Implement proper rejection sampling or inverse CDF for truncated normal.
        return normal(t, mean, stddev);
    }
} // namespace SushiBLAS
