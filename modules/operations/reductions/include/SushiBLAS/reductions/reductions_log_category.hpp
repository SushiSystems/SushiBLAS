/**************************************************************************/
/* reductions_log_category.hpp                                            */
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
 * @file reductions_log_category.hpp
 * @brief Declares K_REDUCTIONS_LOG_CATEGORY, the category the reductions module logs under.
 * @author Mustafa Garip
 * @see modules/operations/reductions/README.md
 */

#pragma once

#include <SushiLog/log/log_category.hpp>

#include <SushiBLAS/core/sushiblas_log_project.hpp>

namespace SushiBLAS
{
    /** @brief Names the records the reductions module logs: reductions as they are recorded. */
    inline constexpr SushiLog::LogCategory K_REDUCTIONS_LOG_CATEGORY =
        SushiLog::make_log_category(K_SUSHIBLAS_LOG_PROJECT, "reductions");
} // namespace SushiBLAS
