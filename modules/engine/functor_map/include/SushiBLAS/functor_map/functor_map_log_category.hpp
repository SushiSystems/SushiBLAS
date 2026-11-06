/**************************************************************************/
/* functor_map_log_category.hpp                                           */
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
 * @file functor_map_log_category.hpp
 * @brief Declares K_FUNCTOR_MAP_LOG_CATEGORY, the category the functor_map module logs under.
 * @author Mustafa Garip
 * @see modules/engine/functor_map/README.md
 */

#pragma once

#include <SushiLog/log/log_category.hpp>

#include <SushiBLAS/core/sushiblas_log_project.hpp>

namespace SushiBLAS
{
    /** @brief Names the records the functor_map module logs: element-wise maps as they are recorded. */
    inline constexpr SushiLog::LogCategory K_FUNCTOR_MAP_LOG_CATEGORY =
        SushiLog::make_log_category(K_SUSHIBLAS_LOG_PROJECT, "functor_map");
} // namespace SushiBLAS
