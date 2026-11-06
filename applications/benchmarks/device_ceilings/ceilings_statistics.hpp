/**************************************************************************/
/* ceilings_statistics.hpp                                                */
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
 * @file ceilings_statistics.hpp
 * @brief Declares the median and the spread of the timed runs of one ceiling.
 * @author Mustafa Garip
 * @see applications/benchmarks/device_ceilings/README.md
 */

#pragma once

#include <vector>
#include <cstddef>
#include <algorithm>

namespace SushiBLAS
{
    namespace Ceilings
    {
        /** @brief Holds what the timed runs of one measurement came to. */
        struct RunStatistics
        {
            double median = 0.0;  /**< The median of the runs, in the unit they were given in. */
            double spread = 0.0;  /**< (maximum - minimum) / median; 0 for no run or a zero median. */
        };

        /**
         * @brief Returns the median and the spread of @p runs.
         * @param runs Any number of values; empty gives a zero result.
         */
        [[nodiscard]] inline RunStatistics summarize(std::vector<double> runs)
        {
            RunStatistics statistics;
            if (runs.empty())
                return statistics;

            std::sort(runs.begin(), runs.end());
            const std::size_t middle = runs.size() / 2;
            statistics.median = runs.size() % 2 == 0 ? 0.5 * (runs[middle - 1] + runs[middle])
                                                     : runs[middle];
            if (statistics.median != 0.0)
                statistics.spread = (runs.back() - runs.front()) / statistics.median;
            return statistics;
        }
    } // namespace Ceilings
} // namespace SushiBLAS
