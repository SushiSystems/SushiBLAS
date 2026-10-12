/**************************************************************************/
/* adapter_demo.cpp                                                       */
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
 * @file adapter_demo.cpp
 * @brief Runs a matrix product, an addition and a sum through the host program's own interface.
 * @author Mustafa Garip
 */

#include "host_program.hpp"

#include <cstddef>
#include <iostream>

namespace
{
    /** @brief Returns a * b + offset, computed on the host with three loops. */
    [[nodiscard]] HostProgram::Matrix host_multiply_add(const HostProgram::Matrix& a,
                                                        const HostProgram::Matrix& b,
                                                        const HostProgram::Matrix& offset)
    {
        HostProgram::Matrix result{a.rows, b.columns, offset.values};
        for (std::size_t row = 0; row < a.rows; ++row)
            for (std::size_t column = 0; column < b.columns; ++column)
                for (std::size_t inner = 0; inner < a.columns; ++inner)
                    result.values[row * b.columns + column] +=
                        a.values[row * a.columns + inner] * b.values[inner * b.columns + column];

        return result;
    }
} // namespace

int main()
{
    HostProgram::BackendHandle backend;
    const HostProgram::Status created = HostProgram::create_backend(backend);
    if (created.code != HostProgram::StatusCode::Ok)
    {
        std::cerr << "create_backend: " << HostProgram::to_string(created.code) << ": "
                  << created.message << "\n";
        return 1;
    }

    const HostProgram::Matrix a{2, 3, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}};
    const HostProgram::Matrix b{3, 2, {7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f}};
    const HostProgram::Matrix offset{2, 2, {0.5f, 0.5f, 0.5f, 0.5f}};

    HostProgram::Matrix result;
    float sum = 0.0f;
    const HostProgram::Status ran = HostProgram::multiply_add(*backend, a, b, offset, result, sum);
    if (ran.code != HostProgram::StatusCode::Ok)
    {
        std::cerr << "multiply_add: " << HostProgram::to_string(ran.code) << ": " << ran.message
                  << "\n";
        return 1;
    }

    const HostProgram::Matrix expected = host_multiply_add(a, b, offset);
    float expected_sum = 0.0f;
    for (const float value : expected.values)
        expected_sum += value;

    std::cout << "a * b + offset = [" << result.values[0] << ", " << result.values[1] << "; "
              << result.values[2] << ", " << result.values[3] << "], sum = " << sum << "\n";

    // a has three columns and two rows, so a * a has no meaning and the backend refuses it.
    HostProgram::Matrix unused;
    const HostProgram::Status refused =
        HostProgram::multiply_add(*backend, a, a, offset, unused, sum);
    std::cout << "a * a: " << HostProgram::to_string(refused.code) << ": " << refused.message
              << "\n";

    const bool correct = result.values == expected.values && sum == expected_sum &&
                         refused.code == HostProgram::StatusCode::InvalidRequest;
    std::cout << (correct ? "adapter_demo: ok\n" : "adapter_demo: MISMATCH\n");
    return correct ? 0 : 1;
}
