/****************************************************************/
/* col2im.cpp                                                   */
/* SushiBLAS - https://github.com/SushiSystems/SushiBLAS        */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

/**
 * @file col2im.cpp
 * @brief Defines SpatialOps::col2im, which records the scatter of a column matrix back onto an
 * image.
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

    sycl::event SpatialOps::col2im(const Tensor& cols, Tensor& image, const Spatial::ConvGeometry& g,
                                   int64_t out_h, int64_t out_w)
    {
        static constexpr const char* NAME = "spatial.col2im";

        SB_THROW_IF(image.dtype != cols.dtype,
                    "Spatial operation '{}' has a {} column matrix and a {} image.",
                    NAME, Core::to_string(cols.dtype), Core::to_string(image.dtype));

        // image plays the input role in the span; see src/README.md.
        const Spatial::WindowSpan span = Spatial::window_span_for(image, g, out_h, out_w, NAME);
        const Spatial::ColumnSpan column = Spatial::column_span_for(cols, span, NAME);

        void* read_cols = cols.data_or_null();
        void* write_image = image.data_or_null();

        std::vector<SushiRuntime::Core::ResourceRegion> reads = {};
        if (read_cols) reads.push_back(cols.region());
        std::vector<SushiRuntime::Core::ResourceRegion> writes = {};
        if (write_image) writes.push_back(image.region());

        SushiRuntime::Graph::TaskMetadata meta;
        meta.name = NAME;
        meta.task_type = SushiRuntime::Graph::TaskType::MATH_OP;
        meta.op_id = "spatial.col2im"_op;

        const auto mode = engine_.get_mode();

        return Internal::execute_typed_task<Dispatch::Domain::REAL>(
            engine_, meta, reads, writes, image.dtype, "Spatial operation",
            [=](auto scalar_type, sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event
            {
                using T = decltype(scalar_type);

                auto source = static_cast<const T*>(read_cols);
                auto destination = static_cast<T*>(write_image);

                return Internal::dispatch_with_mode(mode, q, deps,
                    [=] { Kernels::Reference::col2im<T>(span, column, source, destination); },
                    [=](sycl::queue& q2, const std::vector<sycl::event>& deps2) -> sycl::event
                    {
                        return Kernels::SYCL::col2im<T>(q2, span, column, source, destination, deps2);
                    });
            });
    }
} // namespace SushiBLAS
