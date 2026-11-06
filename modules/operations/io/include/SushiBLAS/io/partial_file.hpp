/**************************************************************************/
/* partial_file.hpp                                                       */
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
 * @file partial_file.hpp
 * @brief Declares remove_partial_file, which deletes the file a failed save left and says when it cannot.
 * @author Mustafa Garip
 * @see modules/operations/io/README.md
 */

#pragma once

#include <cstdio>
#include <string>
#include <SushiBLAS/core/logger.hpp>
#include <SushiBLAS/io/io_log_category.hpp>

namespace SushiBLAS
{
    namespace IOFiles
    {
        /**
         * @brief Deletes the partial file a failed save left at @p path.
         * @return True when the file is gone; false when the delete failed, after a WARN that
         *         names the path and says the file stays on disk.
         * @see modules/operations/io/README.md
         */
        inline bool remove_partial_file(const std::string& path)
        {
            if (std::remove(path.c_str()) == 0)
                return true;

            SB_LOG_WARN(K_IO_LOG_CATEGORY,
                        "io: could not remove the partial file {} after a failed save; "
                        "it stays on disk and may be incomplete",
                        path);
            return false;
        }
    } // namespace IOFiles
} // namespace SushiBLAS
