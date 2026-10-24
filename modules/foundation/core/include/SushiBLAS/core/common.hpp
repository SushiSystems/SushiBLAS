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
 * @file common.hpp
 * @brief Declares the layout and element-type enumerations and the helpers that classify them.
 * @author Mustafa Garip
 */

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
            RowMajor = 0,   /**< Elements of one row are adjacent. */
            ColumnMajor = 1 /**< Elements of one column are adjacent. */
        };


        /**
         * @brief Enumerates the element types a tensor can hold.
         * @warning The values are a file-format contract: none changes, and a new type is appended.
         * @see modules/foundation/core/README.md
         */
        enum class DataType : uint8_t
        {
            Half = 0,      /**< A 16-bit IEEE float. */
            Float32 = 1,   /**< A 32-bit IEEE float. */
            Float64 = 2,   /**< A 64-bit IEEE float. */
            Complex32 = 3, /**< A complex number of two 32-bit floats. */
            Complex64 = 4, /**< A complex number of two 64-bit floats. */
            Int32 = 5,     /**< A signed 32-bit integer. */
            Int64 = 6,     /**< A signed 64-bit integer. */
            UInt32 = 7     /**< An unsigned 32-bit integer; no operation computes on it yet. */
        };

        /** @brief Holds how many enumerators DataType has; the values run from 0 to one less. */
        inline constexpr size_t K_DATA_TYPE_COUNT = 8;

        /** @brief Maximum number of supported tensor ranks. */
        inline constexpr size_t K_MAX_TENSOR_RANK = 6;

        /**
         * @brief Returns the name of a DataType for diagnostics, or "UNKNOWN" outside the enumeration.
         * @see modules/foundation/core/README.md
         */
        inline constexpr const char* to_string(DataType dtype)
        {
            switch (dtype)
            {
                case DataType::Half:
                    return "HALF";
                case DataType::Float32:
                    return "FLOAT32";
                case DataType::Float64:
                    return "FLOAT64";
                case DataType::Complex32:
                    return "COMPLEX32";
                case DataType::Complex64:
                    return "COMPLEX64";
                case DataType::Int32:
                    return "INT32";
                case DataType::Int64:
                    return "INT64";
                case DataType::UInt32:
                    return "UINT32";
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
            return dtype == DataType::Complex32 || dtype == DataType::Complex64;
        }

        /**
         * @brief Returns whether a DataType is the signed INT32 or INT64; UINT32 is not one.
         * @see modules/foundation/core/README.md
         */
        inline constexpr bool is_integer_dtype(DataType dtype)
        {
            return dtype == DataType::Int32 || dtype == DataType::Int64;
        }

        /**
         * @brief Whether a DataType names one of the real floating-point element types.
         *
         * @param dtype The data type to classify.
         * @return True for HALF / FLOAT32 / FLOAT64, false otherwise.
         */
        inline constexpr bool is_real_dtype(DataType dtype)
        {
            return dtype == DataType::Half || dtype == DataType::Float32 ||
                   dtype == DataType::Float64;
        }

        /**
         * @brief Returns the size in bytes of one element of a DataType.
         * @see modules/foundation/core/README.md
         */
        inline constexpr size_t bytes_per_element(DataType dtype)
        {
            switch (dtype)
            {
                case DataType::Half:
                    return 2;
                case DataType::Float32:
                    return 4;
                case DataType::Float64:
                    return 8;
                case DataType::Complex32:
                    return 8;
                case DataType::Complex64:
                    return 16;
                case DataType::Int32:
                    return 4;
                case DataType::Int64:
                    return 8;
                case DataType::UInt32:
                    return 4;
            }
            // Reached only by a DataType holding a value outside the enumeration; see modules/foundation/core/README.md.
            return 4;
        }

    } // namespace Core
} // namespace SushiBLAS
