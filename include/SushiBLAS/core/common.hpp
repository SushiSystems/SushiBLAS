/**************************************************************************/
/* common.hpp                                                             */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                SushiBLAS                               */
/*                https://github.com/SushiSystems/SushiBLAS               */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present Mustafa Garip & Sushi Systems               */
/*                                                                        */
/* Licensed under the Apache License, Version 2.0 (the "License");        */
/* you may not use this file except in compliance with the License.       */
/* You may obtain a copy of the License at                                */
/*                                                                        */
/*     http://www.apache.org/licenses/LICENSE-2.0                         */
/*                                                                        */
/* Unless required by applicable law or agreed to in writing, software    */
/* distributed under the License is distributed on an "AS IS" BASIS,      */
/* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        */
/* implied. See the License for the specific language governing           */
/* permissions and limitations under the License.                         */
/**************************************************************************/

#pragma once

#include <cstddef>
#include <cstdint>

namespace SushiBLAS 
{
    namespace Core
    {
        /** @brief Memory layout strategies. */
        enum class Layout : uint8_t
        {
            ROW_MAJOR,
            COLUMN_MAJOR
        };


        /** @brief Supported data types in SushiBLAS. */
        enum class DataType : uint8_t 
        {
            HALF,
            FLOAT32,
            FLOAT64,
            COMPLEX32,
            COMPLEX64
        };

        /** @brief Maximum number of supported tensor ranks. */
        inline constexpr size_t MAX_TENSOR_RANK = 6;

        /**
         * @brief Renders a DataType as a human-readable name, for diagnostics.
         *
         * Exists so that a "this operation does not support that dtype" failure can name
         * the offending type instead of printing an integer the reader has to decode.
         *
         * @param dtype The data type to name.
         * @return A static, null-terminated name such as "FLOAT32"; "UNKNOWN" for a value
         *         outside the enumeration.
         */
        inline constexpr const char* to_string(DataType dtype)
        {
            switch (dtype)
            {
                case DataType::HALF:
                    return "HALF";
                case DataType::FLOAT32:
                    return "FLOAT32";
                case DataType::FLOAT64:
                    return "FLOAT64";
                case DataType::COMPLEX32:
                    return "COMPLEX32";
                case DataType::COMPLEX64:
                    return "COMPLEX64";
            }
            return "UNKNOWN";
        }

        /**
         * @brief Whether a DataType names one of the complex element types.
         *
         * @param dtype The data type to classify.
         * @return True for COMPLEX32 / COMPLEX64, false otherwise.
         */
        inline constexpr bool is_complex_dtype(DataType dtype)
        {
            return dtype == DataType::COMPLEX32 || dtype == DataType::COMPLEX64;
        }

        /**
         * @brief Size in bytes of one element of a DataType.
         *
         * The one place the dtype-to-width mapping is written down. It used to be
         * open-coded as a four-branch `size_t bpe = 4; if (...)` chain in three separate
         * places (the Tensor constructor's capacity check, Tensor::data(), and io.cpp),
         * which is three chances for a new dtype to be added to two of them.
         *
         * @param dtype The data type to measure.
         * @return The element width in bytes.
         */
        inline constexpr size_t bytes_per_element(DataType dtype)
        {
            switch (dtype)
            {
                case DataType::HALF:
                    return 2;
                case DataType::FLOAT32:
                    return 4;
                case DataType::FLOAT64:
                    return 8;
                case DataType::COMPLEX32:
                    return 8;
                case DataType::COMPLEX64:
                    return 16;
            }
            // Unreachable for any enumerator; matches to_string()'s shape, which needs a
            // trailing return for a DataType holding a value outside the enumeration.
            return 4;
        }

    } // namespace Core
} // namespace SushiBLAS

namespace sb = SushiBLAS;
