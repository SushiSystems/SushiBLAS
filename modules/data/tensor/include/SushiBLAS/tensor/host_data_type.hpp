/**************************************************************************/
/* host_data_type.hpp                                                     */
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
 * @file host_data_type.hpp
 * @brief Declares HostDataType, which maps a host element type to its DataType.
 * @author Mustafa Garip
 */

#pragma once

#include <complex>
#include <cstdint>
#include <sycl/sycl.hpp>
#include <SushiBLAS/core/common.hpp>

namespace SushiBLAS
{
    /** @brief Holds the host-type traits the host transfers use. */
    namespace Detail
    {
        /**
         * @brief Maps a host element type to the DataType a tensor of it carries.
         * @tparam T One of float, double, sycl::half, std::complex<float>,
         * std::complex<double>, int32_t, int64_t, uint32_t.
         */
        template<typename T> struct HostDataType;

        /** @brief Maps float to FLOAT32. */
        template<> struct HostDataType<float>
        {
            /** @brief The matching DataType. */
            static constexpr Core::DataType K_VALUE = Core::DataType::Float32;
        };

        /** @brief Maps double to FLOAT64. */
        template<> struct HostDataType<double>
        {
            /** @brief The matching DataType. */
            static constexpr Core::DataType K_VALUE = Core::DataType::Float64;
        };

        /** @brief Maps sycl::half to HALF. */
        template<> struct HostDataType<sycl::half>
        {
            /** @brief The matching DataType. */
            static constexpr Core::DataType K_VALUE = Core::DataType::Half;
        };

        /** @brief Maps std::complex<float> to COMPLEX32. */
        template<> struct HostDataType<std::complex<float>>
        {
            /** @brief The matching DataType. */
            static constexpr Core::DataType K_VALUE = Core::DataType::Complex32;
        };

        /** @brief Maps std::complex<double> to COMPLEX64. */
        template<> struct HostDataType<std::complex<double>>
        {
            /** @brief The matching DataType. */
            static constexpr Core::DataType K_VALUE = Core::DataType::Complex64;
        };

        /** @brief Maps int32_t to INT32. */
        template<> struct HostDataType<int32_t>
        {
            /** @brief The matching DataType. */
            static constexpr Core::DataType K_VALUE = Core::DataType::Int32;
        };

        /** @brief Maps int64_t to INT64. */
        template<> struct HostDataType<int64_t>
        {
            /** @brief The matching DataType. */
            static constexpr Core::DataType K_VALUE = Core::DataType::Int64;
        };

        /** @brief Maps uint32_t to UINT32. */
        template<> struct HostDataType<uint32_t>
        {
            /** @brief The matching DataType. */
            static constexpr Core::DataType K_VALUE = Core::DataType::UInt32;
        };
    } // namespace Detail
} // namespace SushiBLAS
