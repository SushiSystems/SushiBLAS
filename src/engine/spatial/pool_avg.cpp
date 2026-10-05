/****************************************************************/
/* pool_avg.cpp                                                 */
/* SushiBLAS - https://github.com/SushiSystems/SushiBLAS        */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

/**
 * @file pool_avg.cpp
 * @brief Defines SpatialOps::pool_avg and pool_avg_backward, which record average pooling and its
 * gradient.
 * @author Mustafa Garip
 */

#include <vector>
#include <SushiBLAS/engine.hpp>
#include <SushiBLAS/engine/task.hpp>
#include <SushiBLAS/engine/spatial.hpp>
#include <SushiBLAS/engine/spatial/window.hpp>
#include "kernels/spatial.hpp"

namespace SushiBLAS
{
    using namespace SushiRuntime::Graph::Literals;

    sycl::event SpatialOps::pool_avg(const Tensor& in, Tensor& out, const Spatial::ConvGeometry& g)
    {
        static constexpr const char* NAME = "spatial.pool_avg";

        SB_THROW_IF(in.dtype != out.dtype,
                    "Spatial operation '{}' has a {} input and a {} output.",
                    NAME, Core::to_string(in.dtype), Core::to_string(out.dtype));

        const Spatial::WindowSpan span = Spatial::window_span_for(in, g, out, NAME);

        void* read_in = in.data_or_null();
        void* write_out = out.data_or_null();

        std::vector<SushiRuntime::Core::ResourceRegion> reads = {};
        if (read_in) reads.push_back(in.region());
        std::vector<SushiRuntime::Core::ResourceRegion> writes = {};
        if (write_out) writes.push_back(out.region());

        SushiRuntime::Graph::TaskMetadata meta;
        meta.name = NAME;
        meta.task_type = SushiRuntime::Graph::TaskType::MATH_OP;
        meta.op_id = "spatial.pool_avg"_op;

        const auto mode = engine_.get_mode();

        return Internal::execute_typed_task<Dispatch::Domain::REAL>(
            engine_, meta, reads, writes, in.dtype, "Spatial operation",
            [=](auto scalar_type, sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event
            {
                using T = decltype(scalar_type);

                auto source = static_cast<const T*>(read_in);
                auto destination = static_cast<T*>(write_out);

                return Internal::dispatch_with_mode(mode, q, deps,
                    [=] { Kernels::Reference::pool_avg<T>(span, source, destination); },
                    [=](sycl::queue& q2, const std::vector<sycl::event>& deps2) -> sycl::event
                    {
                        return Kernels::SYCL::pool_avg<T>(q2, span, source, destination, deps2);
                    });
            });
    }

    sycl::event SpatialOps::pool_avg_backward(const Tensor& dy, Tensor& dx,
                                              const Spatial::ConvGeometry& g)
    {
        static constexpr const char* NAME = "spatial.pool_avg_backward";

        SB_THROW_IF(dy.dtype != dx.dtype,
                    "Spatial operation '{}' has a {} output gradient and a {} input gradient.",
                    NAME, Core::to_string(dy.dtype), Core::to_string(dx.dtype));

        // dx plays the input role and dy the output role, as in the forward; see src/README.md.
        const Spatial::WindowSpan span = Spatial::window_span_for(dx, g, dy, NAME);

        void* read_dy = dy.data_or_null();
        void* write_dx = dx.data_or_null();

        std::vector<SushiRuntime::Core::ResourceRegion> reads = {};
        if (read_dy) reads.push_back(dy.region());
        std::vector<SushiRuntime::Core::ResourceRegion> writes = {};
        if (write_dx) writes.push_back(dx.region());

        SushiRuntime::Graph::TaskMetadata meta;
        meta.name = NAME;
        meta.task_type = SushiRuntime::Graph::TaskType::MATH_OP;
        meta.op_id = "spatial.pool_avg_backward"_op;

        const auto mode = engine_.get_mode();

        return Internal::execute_typed_task<Dispatch::Domain::REAL>(
            engine_, meta, reads, writes, dy.dtype, "Spatial operation",
            [=](auto scalar_type, sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event
            {
                using T = decltype(scalar_type);

                auto source = static_cast<const T*>(read_dy);
                auto destination = static_cast<T*>(write_dx);

                return Internal::dispatch_with_mode(mode, q, deps,
                    [=] { Kernels::Reference::pool_avg_backward<T>(span, source, destination); },
                    [=](sycl::queue& q2, const std::vector<sycl::event>& deps2) -> sycl::event
                    {
                        return Kernels::SYCL::pool_avg_backward<T>(q2, span, source,
                                                                   destination, deps2);
                    });
            });
    }
} // namespace SushiBLAS
