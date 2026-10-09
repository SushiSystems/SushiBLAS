/**************************************************************************/
/* io_status.hpp                                                          */
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
 * @file io_status.hpp
 * @brief Declares IoStatus, the result of a file operation of IO.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>

namespace SushiBLAS
{
    /** @brief Says whether a file operation of IO succeeded, and why it did not. */
    enum class IoStatus : std::uint8_t
    {
        Ok,             /**< The file was written or loaded whole. */
        FileUnreadable, /**< The file could not be opened, measured or read. */
        FileUnwritable, /**< The file could not be opened, written or closed; it is removed. */
        MalformedFile,  /**< The magic, the format version or the header breaks the format. */
        Truncated,      /**< The file ends before the bytes its header or the tensor calls for. */
        ShapeMismatch,  /**< The file's rank or a dimension differs from the tensor's. */
        DtypeMismatch,  /**< The file's element type differs from the tensor's. */
        LayoutMismatch, /**< The file's memory order differs from the tensor's layout. */
    };

    /** @brief Returns the enumerator's name. */
    [[nodiscard]] inline constexpr const char* to_string(IoStatus status) noexcept
    {
        switch (status)
        {
            case IoStatus::Ok:             return "Ok";
            case IoStatus::FileUnreadable: return "FileUnreadable";
            case IoStatus::FileUnwritable: return "FileUnwritable";
            case IoStatus::MalformedFile:  return "MalformedFile";
            case IoStatus::Truncated:      return "Truncated";
            case IoStatus::ShapeMismatch:  return "ShapeMismatch";
            case IoStatus::DtypeMismatch:  return "DtypeMismatch";
            case IoStatus::LayoutMismatch: return "LayoutMismatch";
        }
        return "Unknown";
    }
} // namespace SushiBLAS
