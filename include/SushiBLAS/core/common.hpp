/****************************************************************/
/* common.hpp                                                   */
/* SushiBLAS - https://github.com/SushiSystems/SushiBLAS        */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

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
         * @brief Supported data types in SushiBLAS.
         *
         * **The integer types are not general-purpose arithmetic types here.** They exist
         * because some results are indices rather than numbers - `iamax`, `argmax` and
         * `argmin` all answer "which element", and answering that in a FLOAT32 tensor
         * meant either writing eight bytes into a four-byte element (which is what
         * `iamax` did, corrupting whatever a memory planner had packed next to it) or
         * rounding the index through a mantissa (which is what `argmax` did, silently
         * losing exactness past 2^24). They are also the right element type for a
         * comparison result, which is a truth value and not a float.
         *
         * So only the operations for which an integer element means something accept
         * them: `copy`, `constant`, the comparison and logic family, the index
         * reductions, and `io`. Everything else - every elementwise transcendental,
         * every non-linearity, every BLAS product, every folding reduction - rejects
         * them by name at record time. That is deliberate and it is the *point*: a dtype
         * that is quietly accepted and then silently mishandled is worse than one that
         * is not accepted at all. See `SushiBLAS::Dispatch::Domain`, which is how each
         * op states which element types it will instantiate for.
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
         * @brief Whether a DataType names one of the integer element types.
         *
         * The counterpart to @ref is_complex_dtype, and used the same way: to reject a
         * dtype at record time in the handful of places that check a whole family at
         * once rather than switching arm by arm.
         *
         * @param dtype The data type to classify.
         * @return True for INT32 / INT64, false otherwise.
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
                case DataType::INT32:
                    return 4;
                case DataType::INT64:
                    return 8;
            }
            // Unreachable for any enumerator; matches to_string()'s shape, which needs a
            // trailing return for a DataType holding a value outside the enumeration.
            return 4;
        }

    } // namespace Core
} // namespace SushiBLAS

namespace sb = SushiBLAS;
