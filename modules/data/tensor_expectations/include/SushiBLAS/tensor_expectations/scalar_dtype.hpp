/**************************************************************************/
/* scalar_dtype.hpp                                                       */
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
 * @file scalar_dtype.hpp
 * @brief Declares sushiblas_dtype, the DataType of a C++ element type.
 * @author Mustafa Garip
 */

#pragma once

#include <complex>
#include <cstdint>
#include <sycl/sycl.hpp>
#include <SushiBLAS/core/common.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /**
        * @brief Maps a C++ scalar type to its corresponding SushiBLAS DataType enum value.
        */
        template<typename T> Core::DataType sushiblas_dtype();
        template<> inline Core::DataType sushiblas_dtype<sycl::half>() { return Core::DataType::Half; }
        template<> inline Core::DataType sushiblas_dtype<float>() { return Core::DataType::Float32; }
        template<> inline Core::DataType sushiblas_dtype<double>() { return Core::DataType::Float64; }
        template<> inline Core::DataType sushiblas_dtype<std::complex<float>>() { return Core::DataType::Complex32; }
        template<> inline Core::DataType sushiblas_dtype<std::complex<double>>() { return Core::DataType::Complex64; }
        template<> inline Core::DataType sushiblas_dtype<int32_t>() { return Core::DataType::Int32; }
        template<> inline Core::DataType sushiblas_dtype<int64_t>() { return Core::DataType::Int64; }
        template<> inline Core::DataType sushiblas_dtype<uint32_t>() { return Core::DataType::UInt32; }
    } // namespace Test
} // namespace SushiBLAS
