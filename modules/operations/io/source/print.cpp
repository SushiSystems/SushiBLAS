/**************************************************************************/
/* print.cpp                                                              */
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
 * @file print.cpp
 * @brief Implements IO::print, the one IO call that writes to stdout.
 * @author Mustafa Garip
 * @see modules/operations/io/README.md
 */

#include <iostream>

#include <SushiBLAS/io/io.hpp>

namespace SushiBLAS
{
    void IO::print(const Tensor& t, int precision, int edge_items)
    {
        std::cout << to_string(t, precision, edge_items);
    }
} // namespace SushiBLAS
