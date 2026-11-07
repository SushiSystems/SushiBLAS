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
/* Licensed under the Apache License, Version 2.0 (the "License"); you    */
/* may not use this file except in compliance with the License. You may   */
/* obtain a copy of the License at                                        */
/*                                                                        */
/*     https://www.apache.org/licenses/LICENSE-2.0                        */
/*                                                                        */
/* Unless required by applicable law or agreed to in writing, software    */
/* distributed under the License is distributed on an "AS IS" BASIS,      */
/* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        */
/* implied. See the License for the specific language governing           */
/* permissions and limitations under the License.                         */

/**
 * @file blend_transforms.cpp
 * @brief Defines BlendOps::blend_transforms, which records the weighted blend of indexed
 * transforms applied to every point.
 * @author Mustafa Garip
 */

#include <string>
#include <vector>
#include <cstdint>
#include <SushiBLAS/engine/task.hpp>
#include <SushiBLAS/engine/operand_check.hpp>
#include <SushiBLAS/blend/blend.hpp>
#include <SushiBLAS/blend/blend_operand_check.hpp>
#include <SushiBLAS/blend_kernels/blend.hpp>
#include <SushiBLAS/engine/task_description.hpp>

namespace SushiBLAS
{
    using namespace SushiRuntime::Literals;

    namespace
    {
        /** @brief Holds the operation id, which is also the node's name when a call states none. */
        constexpr const char* K_OPERATION = "blend.blend_transforms";

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
        return blend_transforms(points, transforms, indices, weights, out,
                                OperationOptions{K_OPERATION});
    }

    TaskHandle BlendOps::blend_transforms(const Tensor& points, const Tensor& transforms,
                                          const Tensor& indices, const Tensor& weights,
                                          Tensor& out, const OperationOptions& options)
    {
        static constexpr const char* NAME = K_OPERATION;

        Dispatch::require_operation_options(options, NAME);
        const std::string label(options.name);

        Internal::check_blend_operands(Dispatch::placed(points), Dispatch::placed(transforms),
                                       Dispatch::placed(indices), Dispatch::placed(weights),
                                       Dispatch::placed(out), NAME);

        const BlendShape shape{points.shape[0], points.shape[1], transforms.shape[1],
                               indices.shape[1], transforms.shape[0]};
        const BlendPointers data{points.data_or_null(), transforms.data_or_null(),
                                 indices.data_or_null(), weights.data_or_null(),
                                 out.data_or_null()};
        const Core::DataType index_dtype = indices.dtype;

        OperandSet operands;
        operands.reads(points).reads(transforms).reads(indices).reads(weights).writes(out);

        TaskDescription description;
        description.name = label.c_str();
        description.operation = "blend.blend_transforms"_op;
        description.determinism = options.determinism;

        return Internal::execute_typed_task<Dispatch::Domain::Real>(
            recorder_, description, operands, points.dtype, "Blend operation",
            [=](auto scalar_type, const KernelPairDispatcher& dispatcher,
                sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event
            {
                using T = decltype(scalar_type);

                switch (index_dtype)
                {
                    case Core::DataType::Int64:
                        return run_blend<T, int64_t>(dispatcher, q, deps, shape, data);
                    case Core::DataType::UInt32:
                        return run_blend<T, uint32_t>(dispatcher, q, deps, shape, data);
                    default:
                        return run_blend<T, int32_t>(dispatcher, q, deps, shape, data);
                }
            });
    }
} // namespace SushiBLAS
