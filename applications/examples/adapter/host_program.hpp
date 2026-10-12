/**************************************************************************/
/* host_program.hpp                                                       */
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
 * @file host_program.hpp
 * @brief Declares the types and the numeric operations of a host program that names no library.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

/** @brief Holds what the host program knows about its numeric backend. */
namespace HostProgram
{
    /** @brief Names the outcome of a backend call in the host's own terms. */
    enum class StatusCode : std::uint8_t
    {
        Ok,             /**< The call did what it was asked. */
        InvalidRequest, /**< The host asked for something the backend refuses. */
        Unsupported,    /**< The machine or the mode cannot run the request. */
        OutOfMemory,    /**< An allocation failed. */
        BackendFailed,  /**< The backend failed while it created or ran the work. */
    };

    /** @brief Carries the outcome of a backend call and the backend's message. */
    struct Status
    {
        StatusCode code = StatusCode::Ok; /**< The outcome. */
        std::string message;              /**< The backend's sentence; empty on Ok. */
    };

    /** @brief Returns the spelling of a status code. */
    [[nodiscard]] inline const char* to_string(StatusCode code) noexcept
    {
        switch (code)
        {
            case StatusCode::Ok:             return "Ok";
            case StatusCode::InvalidRequest: return "InvalidRequest";
            case StatusCode::Unsupported:    return "Unsupported";
            case StatusCode::OutOfMemory:    return "OutOfMemory";
            case StatusCode::BackendFailed:  return "BackendFailed";
        }

        return "Unknown";
    }

    /** @brief Holds a dense matrix of the host, one row after another. */
    struct Matrix
    {
        std::size_t rows = 0;      /**< The number of rows. */
        std::size_t columns = 0;   /**< The number of columns. */
        std::vector<float> values; /**< rows * columns elements, row by row. */
    };

    /** @brief Names the backend; the adapter alone defines it. */
    class Backend;

    /** @brief Destroys a backend where its definition is visible. */
    struct BackendDeleter
    {
        /** @brief Destroys the backend. */
        void operator()(Backend* backend) const noexcept;
    };

    /** @brief Owns a backend. */
    using BackendHandle = std::unique_ptr<Backend, BackendDeleter>;

    /**
     * @brief Creates the backend into backend.
     * @param backend Left empty when the status is not Ok.
     */
    [[nodiscard]] Status create_backend(BackendHandle& backend);

    /**
     * @brief Computes result = a * b + offset and the sum of the elements of result.
     * @param result Written only when the status is Ok, as is result_sum.
     */
    [[nodiscard]] Status multiply_add(Backend& backend, const Matrix& a, const Matrix& b,
                                      const Matrix& offset, Matrix& result, float& result_sum);
} // namespace HostProgram
