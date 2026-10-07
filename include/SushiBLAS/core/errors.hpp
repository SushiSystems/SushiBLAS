/**************************************************************************/
/* errors.hpp                                                             */
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
 * @file errors.hpp
 * @brief Declares UnsupportedExecution, the exception a mode or device refusal throws.
 * @author Mustafa Garip
 */

#pragma once

#include <stdexcept>

namespace SushiBLAS
{
    /**
     * @brief Reports that an operation cannot run under the execution mode or on the device.
     */
    class UnsupportedExecution final : public std::invalid_argument
    {
        public:
            /// @brief Takes the message what() returns, as std::invalid_argument does.
            using std::invalid_argument::invalid_argument;
    };
} // namespace SushiBLAS
