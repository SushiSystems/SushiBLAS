/**************************************************************************/
/* execution_mode.hpp                                                     */
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
 * @file execution_mode.hpp
 * @brief Declares ExecutionMode, which names the kernel a recorded task runs.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>

namespace SushiBLAS
{
    /**
     * @brief Names which kernel of an operation a task recorded under the mode runs.
     * @see include/SushiBLAS/engine/README.md
     */
    enum class ExecutionMode : std::uint8_t
    {
        Auto,           /**< Lets the library choose; it chooses the SYCL kernel. */
        ForceSycl,     /**< Runs the SYCL kernel. */
        ForceReference /**< Runs the reference path on the host: a reference kernel, or a serial loop over the operation's functor. */
    };
} // namespace SushiBLAS
