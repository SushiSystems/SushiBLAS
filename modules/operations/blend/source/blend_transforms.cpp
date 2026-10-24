/**************************************************************************/
/* blend_transforms.cpp                                                   */
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

/**
 * @file blend_transforms.cpp
 * @brief Defines BlendOps::blend_transforms, which records the weighted blend of indexed
 * transforms applied to every point.
 * @author Mustafa Garip
 */

#include <vector>
#include <cstdint>
#include <SushiBLAS/engine/task.hpp>
#include <SushiBLAS/engine/operand_check.hpp>
#include <SushiBLAS/blend/blend.hpp>
#include <SushiBLAS/blend_kernels/blend.hpp>
#include <SushiBLAS/engine/task_description.hpp>

namespace SushiBLAS
{
    using namespace SushiRuntime::Literals;

    namespace
    {
        /** @brief Holds the extents of one recorded blend, captured by value. */
        struct BlendShape
        {
            int64_t point_count;     /**< P. */
            int64_t components;      /**< The values per point and per output, 3 or 4. */
            int64_t transform_rows;  /**< The rows per transform, 3 or 4. */
            int64_t influences;      /**< J. */
            int64_t transform_count; /**< T. */
        };

        /** @brief Holds the untyped pointers of one recorded blend, captured by value. */
        struct BlendPointers
        {
            const void* points;
            const void* transforms;
            const void* indices;
            const void* weights;
            void* out;
        };

        /**
         * @brief Throws unless a tensor is dense and row-major, so its rows are contiguous.
         * @throws std::invalid_argument on a view, a slice or a column-major tensor.
         * @see modules/operations/blend/README.md
         */
        void require_dense_row_major(const Tensor& t, const char* name, const char* operand)
        {
            Dispatch::require_contiguous(t, name, operand);
            Dispatch::require_operand(t.layout == Core::Layout::RowMajor, name, operand,
                                      "is column-major; it must be dense and row-major");
        }

        /**
         * @brief Runs the reference or the SYCL kernel for element type T and index type Index.
         * @return The event of the SYCL kernel; the reference path returns a completed event.
         */
        template<typename T, typename Index>
        sycl::event run_blend(const KernelPairDispatcher& dispatcher, sycl::queue& q,
                              const std::vector<sycl::event>& deps, const BlendShape& shape,
                              const BlendPointers& data)
        {
            auto points = static_cast<const T*>(data.points);
            auto transforms = static_cast<const T*>(data.transforms);
            auto indices = static_cast<const Index*>(data.indices);
            auto weights = static_cast<const T*>(data.weights);
            auto out = static_cast<T*>(data.out);

            return dispatcher.run(q, deps,
                [=]
                {
                    Kernels::Reference::blend_transforms<T, Index>(
                        shape.point_count, shape.components, shape.transform_rows,
                        shape.influences, shape.transform_count, points, transforms, indices,
                        weights, out);
                },
                [=](sycl::queue& q2, const std::vector<sycl::event>& deps2) -> sycl::event
                {
                    return Kernels::SYCL::blend_transforms<T, Index>(
                        q2, shape.point_count, shape.components, shape.transform_rows,
                        shape.influences, shape.transform_count, points, transforms, indices,
                        weights, out, deps2);
                });
        }
    } // namespace

    TaskHandle BlendOps::blend_transforms(const Tensor& points, const Tensor& transforms,
                                          const Tensor& indices, const Tensor& weights,
                                          Tensor& out)
    {
        static constexpr const char* NAME = "blend.blend_transforms";

        Dispatch::require_dtype(transforms, points.dtype, NAME, "transforms");
        Dispatch::require_dtype(weights, points.dtype, NAME, "weights");
        Dispatch::require_dtype(out, points.dtype, NAME, "out");
        Dispatch::require_operand(indices.dtype == Core::DataType::Int32 ||
                                      indices.dtype == Core::DataType::Int64,
                                  NAME, "indices", "has dtype {}; indices are INT32 or INT64",
                                  Core::to_string(indices.dtype));

        Dispatch::require_rank(points, 2, NAME, "points");
        Dispatch::require_rank(transforms, 3, NAME, "transforms");
        Dispatch::require_rank(indices, 2, NAME, "indices");
        Dispatch::require_rank(weights, 2, NAME, "weights");
        Dispatch::require_rank(out, 2, NAME, "out");

        const int64_t point_count = points.shape[0];
        const int64_t components = points.shape[1];
        const int64_t transform_count = transforms.shape[0];
        const int64_t transform_rows = transforms.shape[1];
        const int64_t influences = indices.shape[1];

        Dispatch::require_operand(components == 3 || components == 4, NAME, "points",
                                  "has {} values per point; a position has 3 and a homogeneous "
                                  "point 4", components);
        Dispatch::require_operand(point_count >= 1, NAME, "points", "holds no points");
        Dispatch::require_operand(transform_count >= 1, NAME, "transforms", "holds no transforms");
        Dispatch::require_operand(transform_rows == 3 || transform_rows == 4, NAME, "transforms",
                                  "has {} rows per transform; a transform has 3 or 4",
                                  transform_rows);
        Dispatch::require_extent(transforms.shape[2], 4, "columns per transform", NAME,
                                 "transforms");
        Dispatch::require_operand(influences >= 1, NAME, "indices",
                                  "holds no influences per point; at least 1 is required");
        Dispatch::require_extent(indices.shape[0], point_count, "rows, one per point", NAME,
                                 "indices");
        Dispatch::require_same_shape(weights, indices, NAME, "weights", "indices");
        Dispatch::require_same_shape(out, points, NAME, "out", "points");

        Dispatch::require_storage(points, NAME, "points");
        Dispatch::require_storage(transforms, NAME, "transforms");
        Dispatch::require_storage(indices, NAME, "indices");
        Dispatch::require_storage(weights, NAME, "weights");
        Dispatch::require_storage(out, NAME, "out");
        require_dense_row_major(points, NAME, "points");
        require_dense_row_major(transforms, NAME, "transforms");
        require_dense_row_major(indices, NAME, "indices");
        require_dense_row_major(weights, NAME, "weights");
        require_dense_row_major(out, NAME, "out");
        Dispatch::require_device_dtype(points, NAME, "points");
        Dispatch::require_device_dtype(transforms, NAME, "transforms");
        Dispatch::require_device_dtype(weights, NAME, "weights");
        Dispatch::require_device_dtype(out, NAME, "out");

        // Each output point is written while its inputs are read, so no byte may be shared.
        Dispatch::require_operand(!out.region().overlaps(points.region()), NAME, "out",
                                  "overlaps 'points' in memory");
        Dispatch::require_operand(!out.region().overlaps(transforms.region()), NAME, "out",
                                  "overlaps 'transforms' in memory");
        Dispatch::require_operand(!out.region().overlaps(indices.region()), NAME, "out",
                                  "overlaps 'indices' in memory");
        Dispatch::require_operand(!out.region().overlaps(weights.region()), NAME, "out",
                                  "overlaps 'weights' in memory");

        const BlendShape shape{point_count, components, transform_rows, influences,
                               transform_count};
        const BlendPointers data{points.data_or_null(), transforms.data_or_null(),
                                 indices.data_or_null(), weights.data_or_null(),
                                 out.data_or_null()};
        const bool wide_indices = indices.dtype == Core::DataType::Int64;

        OperandSet operands;
        operands.reads(points).reads(transforms).reads(indices).reads(weights).writes(out);

        TaskDescription description;
        description.name = NAME;
        description.operation = "blend.blend_transforms"_op;

        return Internal::execute_typed_task<Dispatch::Domain::Real>(
            recorder_, description, operands, points.dtype, "Blend operation",
            [=](auto scalar_type, const KernelPairDispatcher& dispatcher,
                sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event
            {
                using T = decltype(scalar_type);

                if (wide_indices)
                    return run_blend<T, int64_t>(dispatcher, q, deps, shape, data);
                return run_blend<T, int32_t>(dispatcher, q, deps, shape, data);
            });
    }
} // namespace SushiBLAS
