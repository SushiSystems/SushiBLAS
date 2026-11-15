/**************************************************************************/
/* broadcast_checks.hpp                                                   */
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
/**************************************************************************/

/**
 * @file broadcast_checks.hpp
 * @brief Declares the operand checks of a broadcast call and the extents they return.
 * @author Mustafa Garip
 * @see modules/operations/broadcast/README.md
 */

#pragma once

#include <string>
#include <cstddef>
#include <cstdint>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/engine/operand_check.hpp>
#include <SushiBLAS/functor_map/map_operand.hpp>
#include <SushiBLAS/broadcast/broadcast_operand.hpp>

namespace SushiBLAS
{
    namespace Dispatch
    {
        namespace Detail
        {
            /** @brief Holds the extents of the matrix a broadcast call walks. */
            struct BroadcastGeometry
            {
                /** @brief Holds the element count of the leading vectors. */
                std::size_t outer;
                /** @brief Holds the element count of the trailing vectors. */
                std::size_t inner;
            };

            /** @brief Returns the shape of t as text, such as "[2, 3, 4]". */
            [[nodiscard]] inline std::string broadcast_shape_text(const Tensor& t)
            {
                std::string text = "[";
                for (int32_t d = 0; d < t.rank; ++d)
                {
                    if (d > 0)
                        text += ", ";
                    text += std::to_string(t.shape[d]);
                }
                return text + "]";
            }

            /** @brief Reports whether outer is the product of the first k extents of t, for some k. */
            [[nodiscard]] inline bool is_leading_product(const Tensor& t, int64_t outer) noexcept
            {
                int64_t product = 1;
                if (product == outer)
                    return true;

                for (int32_t d = 0; d < t.rank && product < outer; ++d)
                {
                    product *= t.shape[d];
                    if (product == outer)
                        return true;
                }
                return false;
            }

            /** @brief Throws unless t is row-major; a column-major flat order is no prefix by suffix. */
            inline void require_broadcast_layout(const Tensor& t, const char* name,
                                                 const char* operand)
            {
                require_operand(t.layout == Core::Layout::RowMajor, name, operand,
                                "is column-major; a broadcast needs a row-major operand, because "
                                "only its flat order is a prefix by a suffix");
            }

            /**
             * @brief Checks the operands of a broadcast call and returns the extents it walks.
             * @tparam Bind True when the task binds its operands, so a vector may be planned.
             * @pre a holds at least one element.
             * @throws std::invalid_argument naming the operand and the rule, for each refusal the
             *         broadcast design lists.
             * @see modules/operations/broadcast/README.md
             */
            template<bool Bind>
            [[nodiscard]] inline BroadcastGeometry require_broadcast_operands(
                const Tensor& a, const BroadcastOperand* vectors, std::size_t count, const Tensor& c,
                const char* name)
            {
                std::int64_t leading = -1;
                std::int64_t trailing = -1;

                for (std::size_t k = 0; k < count; ++k)
                {
                    const std::string operand = "vector " + std::to_string(k);
                    require_operand(vectors[k].vector != nullptr, name, operand.c_str(),
                                    "is null");

                    const Tensor& v = *vectors[k].vector;
                    require_map_storage<Bind>(v, name, operand.c_str());
                    require_device_dtype(v, name, operand.c_str());
                    require_broadcast_layout(v, name, operand.c_str());
                    require_contiguous(v, name, operand.c_str());
                    require_dtype(v, a.dtype, name, operand.c_str());
                    require_operand(v.num_elements > 0, name, operand.c_str(),
                                    "holds no element, but the matrix holds {}", a.num_elements);
                    require_operand(!c.region().overlaps(v.region()), name, operand.c_str(),
                                    "overlaps 'C' in memory");

                    std::int64_t& expected =
                        vectors[k].axis == BroadcastAxis::Leading ? leading : trailing;
                    if (expected < 0)
                        expected = v.num_elements;
                    else
                        require_operand(v.num_elements == expected, name, operand.c_str(),
                                        "holds {} elements, but an earlier {} vector holds {}",
                                        v.num_elements,
                                        vectors[k].axis == BroadcastAxis::Leading ? "leading"
                                                                                  : "trailing",
                                        expected);
                }

                const std::int64_t total = a.num_elements;
                const std::string shape = broadcast_shape_text(a);
                std::int64_t outer = leading;
                std::int64_t inner = trailing;

                if (leading >= 0 && trailing >= 0)
                {
                    require_operand(leading * trailing == total, name, "vectors",
                                    "leading count {} times trailing count {} is not the {} "
                                    "elements of shape {}",
                                    leading, trailing, total, shape);
                }
                else if (leading >= 0)
                {
                    require_operand(total % leading == 0, name, "vectors",
                                    "leading count {} does not divide the {} elements of shape {}",
                                    leading, total, shape);
                    inner = total / leading;
                }
                else
                {
                    require_operand(total % trailing == 0, name, "vectors",
                                    "trailing count {} does not divide the {} elements of shape {}",
                                    trailing, total, shape);
                    outer = total / trailing;
                }

                require_operand(is_leading_product(a, outer), name, "vectors",
                                "count {} is no product of leading extents of shape {}", outer,
                                shape);

                return BroadcastGeometry{static_cast<std::size_t>(outer),
                                         static_cast<std::size_t>(inner)};
            }

            /** @brief Throws unless c is a itself or lies apart from it in memory. */
            inline void require_broadcast_output(const Tensor& a, const Tensor& c, const char* name)
            {
                const SushiRuntime::BufferRegion region_a = a.region();
                const SushiRuntime::BufferRegion region_c = c.region();
                if (!region_c.overlaps(region_a))
                    return;

                require_operand(region_c.base == region_a.base &&
                                    region_c.offset == region_a.offset &&
                                    region_c.length == region_a.length,
                                name, "C",
                                "partly overlaps 'A' in memory; C may be A itself or lie apart "
                                "from it");
            }
        } // namespace Detail
    } // namespace Dispatch
} // namespace SushiBLAS
