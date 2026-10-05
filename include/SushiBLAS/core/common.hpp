/****************************************************************/
/* common.hpp                                                   */
/* SushiBLAS - https://github.com/SushiSystems/SushiBLAS        */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

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
            ROW_MAJOR,
            COLUMN_MAJOR
        };


        /**
         * @brief Enumerates the element types a tensor can hold.
         * @see include/SushiBLAS/core/README.md
         */
        enum class DataType : uint8_t
        {
            HALF,
            FLOAT32,
            FLOAT64,
            COMPLEX32,
            COMPLEX64,
            INT32,
            INT64
        };

        /** @brief Maximum number of supported tensor ranks. */
        inline constexpr size_t MAX_TENSOR_RANK = 6;

        /**
         * @brief Returns the name of a DataType for diagnostics, or "UNKNOWN" outside the enumeration.
         * @see include/SushiBLAS/core/README.md
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
                case DataType::INT32:
                    return "INT32";
                case DataType::INT64:
                    return "INT64";
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
         * @brief Returns whether a DataType is INT32 or INT64.
         * @see include/SushiBLAS/core/README.md
         */
        inline constexpr bool is_integer_dtype(DataType dtype)
        {
            return dtype == DataType::INT32 || dtype == DataType::INT64;
        }

        /**
         * @brief Whether a DataType names one of the real floating-point element types.
         *
         * @param dtype The data type to classify.
         * @return True for HALF / FLOAT32 / FLOAT64, false otherwise.
         */
        inline constexpr bool is_real_dtype(DataType dtype)
        {
            return !is_complex_dtype(dtype) && !is_integer_dtype(dtype);
        }

        /**
         * @brief Returns the size in bytes of one element of a DataType.
         * @see include/SushiBLAS/core/README.md
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
                case DataType::INT32:
                    return 4;
                case DataType::INT64:
                    return 8;
            }
            // Reached only by a DataType holding a value outside the enumeration; see include/SushiBLAS/core/README.md.
            return 4;
        }

    } // namespace Core
} // namespace SushiBLAS

namespace sb = SushiBLAS;
