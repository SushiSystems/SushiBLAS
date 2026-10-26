/**************************************************************************/
/* cost_table.hpp                                                         */
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
 * @file cost_table.hpp
 * @brief Declares the table of cost rows and cost_of, the one function that applies a rule.
 * @author Mustafa Garip
 * @see modules/foundation/cost/README.md
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <SushiBLAS/cost/cost.hpp>
#include <SushiBLAS/cost/cost_shape.hpp>
#include <SushiRuntime/core/operation_id.hpp>

namespace SushiBLAS
{
    /**
     * @brief Holds what the rules know about one operation.
     *
     * The weights are operations of the operation's formula, a transcendental counted as 1; which
     * of them a family reads is stated in the cost README.
     * @see modules/foundation/cost/README.md
     */
    struct CostRow
    {
        SushiRuntime::OperationID id;                 /**< The operation the row describes. */
        std::uint32_t per_element = 0;                /**< Work applied to each input element. */
        std::uint32_t per_combine = 0;                /**< Work to fold one element into a result. */
        std::uint32_t per_output = 0;                 /**< Work done once for each output element. */
        UncostedReason reason = UncostedReason::None; /**< Why the operation states no cost, or None. */
    };

    /** @brief Names a run of rows in the table, sorted by id. */
    struct CostRows
    {
        const CostRow* first = nullptr; /**< The first row. */
        std::size_t count = 0;          /**< How many rows. */
    };

    /** @brief Returns every row of the table, sorted by operation id, with no id twice. */
    [[nodiscard]] CostRows cost_rows() noexcept;

    /**
     * @brief Returns the row of an operation, or null when the table has none.
     * @return A pointer into the table, valid for the life of the process.
     */
    [[nodiscard]] const CostRow* find_cost_row(SushiRuntime::OperationID id) noexcept;

    /**
     * @brief Returns the cost the table states for an operation over a shape; bytes are 0.
     * @return Cost::uncosted(NoRule) without a row, the row's reason for a row that declares one,
     *         and Cost::uncosted(NoShape) for an empty shape.
     * @see modules/foundation/cost/README.md
     */
    [[nodiscard]] Cost cost_of(SushiRuntime::OperationID id, const CostShape& shape) noexcept;
} // namespace SushiBLAS
