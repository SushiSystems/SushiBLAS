/**************************************************************************/
/* signal.hpp                                                             */
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

/**
 * @file signal.hpp
 * @brief Includes every signal kernel, reference and SYCL.
 * @author Mustafa Garip
 */

#pragma once

// Every signal kernel is private to this tree; see kernels/README.md.

#include <SushiBLAS/signal_kernels/reference/convolve_1d.hpp>
#include <SushiBLAS/signal_kernels/sycl/convolve_1d.hpp>
#include <SushiBLAS/signal_kernels/fft_plan.hpp>
#include <SushiBLAS/signal_kernels/reference/fft.hpp>
#include <SushiBLAS/signal_kernels/sycl/fft.hpp>
