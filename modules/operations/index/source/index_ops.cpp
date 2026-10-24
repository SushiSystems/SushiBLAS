/**************************************************************************/
/* index_ops.cpp                                                          */
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
 * @file index_ops.cpp
 * @brief Defines the checks and the operation id IndexOps shares across its templates.
 * @author Mustafa Garip
 */

#include <SushiBLAS/index/index_ops.hpp>

#include <SushiBLAS/engine/operand_check.hpp>

namespace SushiBLAS
{
    using namespace SushiRuntime::Literals;

    SushiRuntime::OperationID IndexOps::for_each_operation() noexcept
    {
        return "index.for_each"_op;
    }

    void IndexOps::require_count_not_written(const Tensor& count, const OperandSet& operands,
                                             const char* name)
    {
        const SushiRuntime::BufferRegion region = count.region();
        for (const SushiRuntime::BufferRegion& written : operands.write_regions())
            Dispatch::require_operand(!region.overlaps(written), name, "count",
                                      "overlaps a tensor the operation writes");
    }
} // namespace SushiBLAS
