/**************************************************************************/
/* cost.hpp                                                               */
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
 * @file cost.hpp
 * @brief Declares Cost, what one recorded task states about its work, and why it may state none.
 * @author Mustafa Garip
 * @see modules/foundation/cost/README.md
 */

#pragma once

#include <cstdint>
#include <SushiBLAS/core/common.hpp>

namespace SushiBLAS
{
    /**
     * @brief Names why a task states no cost.
     * @see modules/foundation/cost/README.md
     */
    enum class UncostedReason : std::uint8_t
    {
        None,          /**< The task is costed. */
        NoRule,        /**< The operation id has no row in the cost table. */
        NoShape,       /**< The operation has a row but its call site stated no shape. */
        DataDependent, /**< The work depends on data or a functor the recorder cannot see. */
        Unaudited      /**< The row is declared, and its body is not yet read for a rule. */
    };

    /**
     * @brief Returns the name of a reason, for a report.
     * @return A string literal; "unknown" outside the enumeration.
     */
    [[nodiscard]] constexpr const char* to_string(UncostedReason reason) noexcept
    {
        switch (reason)
        {
            case UncostedReason::None:
                return "none";
            case UncostedReason::NoRule:
                return "no_rule";
            case UncostedReason::NoShape:
                return "no_shape";
            case UncostedReason::DataDependent:
                return "data_dependent";
            case UncostedReason::Unaudited:
                return "unaudited";
        }
        return "unknown";
    }

    /**
     * @brief Returns the real type a dtype is judged in: a complex type counts in its real class.
     * @see modules/foundation/cost/README.md
     */
    [[nodiscard]] constexpr Core::DataType real_class_of(Core::DataType dtype) noexcept
    {
        switch (dtype)
        {
            case Core::DataType::Complex32:
                return Core::DataType::Float32;
            case Core::DataType::Complex64:
                return Core::DataType::Float64;
            default:
                return dtype;
        }
    }

    /**
     * @brief Holds the work one task states: floating-point operations, bytes and the dtype class.
     *
     * A task with a reason other than None states no work; its flops and bytes are 0 and mean
     * "not stated", never "zero work".
     * @see modules/foundation/cost/README.md
     */
    struct Cost
    {
        /** @brief Floating-point operations the task performs, by the conventions of the cost README. */
        std::uint64_t flops = 0;

        /** @brief Bytes the operands the task declared span, an updated operand counted twice. */
        std::uint64_t bytes = 0;

        /** @brief The real type the task's operands are judged in; meaningless when the task is uncosted. */
        Core::DataType dtype = Core::DataType::Float32;

        /** @brief Why the task states no cost; None when it states one. */
        UncostedReason reason = UncostedReason::None;

        /** @brief Returns whether the task states a cost. */
        [[nodiscard]] constexpr bool is_costed() const noexcept
        {
            return reason == UncostedReason::None;
        }

        /** @brief Builds the cost of a task that states none, for @p why. */
        [[nodiscard]] static constexpr Cost uncosted(UncostedReason why) noexcept
        {
            Cost cost;
            cost.reason = why;
            return cost;
        }

        /** @brief Returns this cost with the bytes the operands span; an uncosted cost stays as it is. */
        [[nodiscard]] constexpr Cost with_bytes(std::uint64_t operand_bytes) const noexcept
        {
            Cost cost = *this;
            if (cost.is_costed())
                cost.bytes = operand_bytes;
            return cost;
        }
    };
} // namespace SushiBLAS
