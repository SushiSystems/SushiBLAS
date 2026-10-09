/**************************************************************************/
/* argument_check.hpp                                                     */
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
 * @file argument_check.hpp
 * @brief Declares the record-time checks on the flags, scalars and routine domain of a BLAS call.
 * @author Mustafa Garip
 */

#pragma once

#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/core/blas_types.hpp>
#include <SushiBLAS/operands/scalar.hpp>
#include <SushiBLAS/engine/operand_check.hpp>

namespace SushiBLAS
{
    namespace Dispatch
    {
        /**
         * @brief Throws unless a flag holds one of its type's enumerators.
         * @tparam Flag Core::Transpose, Core::Uplo, Core::Diag or Core::Side.
         * @throws std::invalid_argument naming the argument and its numeric value.
         */
        template<typename Flag>
        inline void require_flag(Flag flag, const char* op, const char* argument)
        {
            if (!Core::is_valid(flag))
                Detail::throw_operand_error(op, argument,
                                            "holds the value {}, which is not an enumerator of "
                                            "its type",
                                            static_cast<int>(flag));
        }

        /**
         * @brief Throws if a scalar with a non-zero imaginary part is given to a real dtype.
         * @throws std::invalid_argument naming the imaginary part and the dtype.
         */
        inline void require_real_scalar(const BlasScalar& value, Core::DataType dtype,
                                        const char* op, const char* argument)
        {
            if (!Core::is_complex_dtype(dtype) && !value.is_real())
                Detail::throw_operand_error(op, argument,
                                            "has imaginary part {}; a {} tensor takes a real "
                                            "scalar",
                                            value.imag(), Core::to_string(dtype));
        }

        /**
         * @brief Throws if a routine defined for real tensors is given a complex one.
         * @param siblings Names the routines that take a complex tensor, such as "dotu or dotc".
         * @throws std::invalid_argument naming the dtype and the siblings.
         */
        inline void require_real_dtype(const Tensor& t, const char* op, const char* operand,
                                       const char* siblings)
        {
            if (Core::is_complex_dtype(t.dtype))
                Detail::throw_operand_error(op, operand,
                                            "has dtype {}; this routine is defined for real "
                                            "tensors, use {}",
                                            Core::to_string(t.dtype), siblings);
        }

        /**
         * @brief Throws if trans is the one value a routine does not define for a complex tensor.
         * @param sibling Names the routine that takes the refused value.
         * @throws std::invalid_argument naming the value and the sibling.
         */
        inline void require_transpose_not(Core::Transpose trans, Core::Transpose refused,
                                          const char* op, const char* sibling)
        {
            if (trans == refused)
                Detail::throw_operand_error(op, "trans",
                                            "is {}, which this routine does not define for a "
                                            "complex tensor; {} takes it",
                                            Core::to_string(trans), sibling);
        }
    } // namespace Dispatch
} // namespace SushiBLAS
