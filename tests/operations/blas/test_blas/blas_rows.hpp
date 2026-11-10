/**************************************************************************/
/* blas_rows.hpp                                                          */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                SushiBLAS                               */
/*                https://github.com/SushiSystems/SushiBLAS               */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present Mustafa Garip & Sushi Systems               */
/*                                                                        */
/* Licensed under the Apache License, Version 2.0 (the "License"); you    */
/* may not use this file except in compliance with the License. You may   */
/* obtain a copy of the License at                                        */
/*                                                                        */
/*     https://www.apache.org/licenses/LICENSE-2.0                        */
/*                                                                        */
/* Unless required by applicable law or agreed to in writing, software    */
/* distributed under the License is distributed on an "AS IS" BASIS,      */
/* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        */
/* implied. See the License for the specific language governing           */
/* permissions and limitations under the License.                         */
/**************************************************************************/

/**
 * @file blas_rows.hpp
 * @brief Declares the parameter row of the BLAS suites, the words that name a row, the
 *        dispatch from a data type to its element type and the fixture of a row.
 * @author Mustafa Garip
 */

#pragma once

#include <string>
#include <vector>
#include <complex>
#include <cstdint>
#include <type_traits>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/core/blas_types.hpp>
#include <SushiBLAS/engine/execution_mode.hpp>
#include <SushiBLAS/engine_fixture/sushiblas_test.hpp>
#include <SushiBLAS/test_runner/test_environment.hpp>
#include <SushiBLAS/tensor_expectations/tolerance_bounds.hpp>
#include "support/tolerance.hpp"

namespace SushiBLAS
{
    namespace Test
    {
        /** @brief Holds the layouts a row runs under, in the order the rows list them. */
        inline constexpr Core::Layout K_ROW_LAYOUTS[] = {Core::Layout::RowMajor,
                                                         Core::Layout::ColumnMajor};

        /** @brief Holds the five element types the BLAS routines accept. */
        inline const std::vector<Core::DataType> K_ALL_TYPES = {
            Core::DataType::Half, Core::DataType::Float32, Core::DataType::Float64,
            Core::DataType::Complex32, Core::DataType::Complex64};

        /** @brief Holds the two complex element types, the only ones a conjugation changes. */
        inline const std::vector<Core::DataType> K_COMPLEX_TYPES = {Core::DataType::Complex32,
                                                                    Core::DataType::Complex64};

        /** @brief Returns the word that names a triangle in a row name. */
        [[nodiscard]] inline std::string word(Core::Uplo uplo)
        {
            return uplo == Core::Uplo::Upper ? "Upper" : "Lower";
        }

        /** @brief Returns the word that names a side in a row name. */
        [[nodiscard]] inline std::string word(Core::Side side)
        {
            return side == Core::Side::Left ? "Left" : "Right";
        }

        /** @brief Returns the word that names a form of op(A) in a row name. */
        [[nodiscard]] inline std::string word(Core::Transpose trans)
        {
            if (trans == Core::Transpose::NoTrans)
                return "NoTrans";
            return trans == Core::Transpose::Trans ? "Trans" : "ConjTrans";
        }

        /** @brief Returns the word that names a diagonal in a row name. */
        [[nodiscard]] inline std::string word(Core::Diag diag)
        {
            return diag == Core::Diag::Unit ? "Unit" : "NonUnit";
        }

        /** @brief Returns the word that names a layout in a row name. */
        [[nodiscard]] inline std::string word(Core::Layout layout)
        {
            return layout == Core::Layout::RowMajor ? "RowMajor" : "ColumnMajor";
        }

        /** @brief Returns the word that names a kernel in a row name. */
        [[nodiscard]] inline std::string word(ExecutionMode mode)
        {
            return mode == ExecutionMode::ForceSYCL ? "ForceSYCL" : "ForceReference";
        }

        /** @brief Marks a row that varies over layout, kernel and element type alone. */
        struct NoAxis
        {
        };

        /** @brief Returns the empty tail of a row that has no further axis. */
        [[nodiscard]] inline std::string axis_name(const NoAxis&)
        {
            return "";
        }

        /** @brief Returns the tail of a row named by a triangle. */
        [[nodiscard]] inline std::string axis_name(Core::Uplo uplo)
        {
            return "_" + word(uplo);
        }

        /** @brief Returns the tail of a row named by a side. */
        [[nodiscard]] inline std::string axis_name(Core::Side side)
        {
            return "_" + word(side);
        }

        /** @brief Returns the tail of a row named by a form of op(A). */
        [[nodiscard]] inline std::string axis_name(Core::Transpose trans)
        {
            return "_" + word(trans);
        }

        /** @brief Returns the tail of a row named by a band width. */
        [[nodiscard]] inline std::string axis_name(int64_t width)
        {
            return "_k" + std::to_string(width);
        }

        /**
         * @brief Holds one row of a suite: the engine layout, the kernel, the element type and
         *        the one axis the cases of the suite vary besides those.
         * @tparam Axis NoAxis, or the type of the value that completes the row.
         */
        template<typename Axis>
        struct Row
        {
            Core::Layout layout; /**< The element order the engine is rebuilt in. */
            ExecutionMode mode;  /**< The kernel the engine selects. */
            Core::DataType type; /**< The element type of every operand. */
            Axis axis;           /**< The value that completes the row. */
        };

        /**
         * @brief Returns every row over the layouts, the given kernels and types and axes.
         * @param types The element types of the suite.
         * @param modes The kernels of the suite.
         * @param axes The values of the suite's own axis.
         */
        template<typename Axis>
        [[nodiscard]] std::vector<Row<Axis>> rows_of(const std::vector<Core::DataType>& types,
                                                     const std::vector<ExecutionMode>& modes,
                                                     const std::vector<Axis>& axes)
        {
            std::vector<Row<Axis>> rows;
            for (const Core::DataType type : types)
                for (const Core::Layout layout : K_ROW_LAYOUTS)
                    for (const ExecutionMode mode : modes)
                        for (const Axis& axis : axes)
                            rows.push_back({layout, mode, type, axis});
            return rows;
        }

        /** @brief Returns the rows of a suite that has no axis of its own. */
        [[nodiscard]] inline std::vector<Row<NoAxis>> plain_rows(
            const std::vector<Core::DataType>& types, const std::vector<ExecutionMode>& modes)
        {
            return rows_of(types, modes, std::vector<NoAxis>{NoAxis{}});
        }

        /** @brief Returns the name of a row for GoogleTest: layout, kernel, type, axis. */
        template<typename Axis>
        [[nodiscard]] std::string row_name(const ::testing::TestParamInfo<Row<Axis>>& info)
        {
            return word(info.param.layout) + "_" + word(info.param.mode) + "_" +
                   Core::to_string(info.param.type) + axis_name(info.param.axis);
        }

        /** @brief Carries an element type to a generic lambda. */
        template<typename T>
        struct Element
        {
            using type = T; /**< The element type. */
        };

        /**
         * @brief Calls body with the Element of a data type, so one row body serves every type.
         * @param type One of the five types the BLAS routines accept.
         * @param body Callable taking an Element.
         */
        template<typename Body>
        void with_element(Core::DataType type, Body&& body)
        {
            switch (type)
            {
                case Core::DataType::Half:
                    body(Element<sycl::half>{});
                    break;
                case Core::DataType::Float32:
                    body(Element<float>{});
                    break;
                case Core::DataType::Float64:
                    body(Element<double>{});
                    break;
                case Core::DataType::Complex32:
                    body(Element<std::complex<float>>{});
                    break;
                case Core::DataType::Complex64:
                    body(Element<std::complex<double>>{});
                    break;
                default:
                    FAIL() << "a BLAS routine takes no such element type";
            }
        }

        /**
         * @brief Calls body with the Element of FLOAT32, FLOAT64, COMPLEX32 or COMPLEX64.
         *
         * HALF is left out for a case whose helpers have no HALF instantiation, such as
         * `random_elements`, which has no `uniform_real_distribution` for `sycl::half`.
         * @param type One of the four types named above.
         * @param body Callable taking an Element.
         */
        template<typename Body>
        void with_non_half_element(Core::DataType type, Body&& body)
        {
            switch (type)
            {
                case Core::DataType::Float32:
                    body(Element<float>{});
                    break;
                case Core::DataType::Float64:
                    body(Element<double>{});
                    break;
                case Core::DataType::Complex32:
                    body(Element<std::complex<float>>{});
                    break;
                case Core::DataType::Complex64:
                    body(Element<std::complex<double>>{});
                    break;
                default:
                    FAIL() << "this case takes FLOAT32, FLOAT64, COMPLEX32 or COMPLEX64";
            }
        }

        /**
         * @brief Calls body with the Element of a real data type.
         * @param type HALF, FLOAT32 or FLOAT64.
         * @param body Callable taking an Element.
         */
        template<typename Body>
        void with_real_element(Core::DataType type, Body&& body)
        {
            switch (type)
            {
                case Core::DataType::Half:
                    body(Element<sycl::half>{});
                    break;
                case Core::DataType::Float32:
                    body(Element<float>{});
                    break;
                case Core::DataType::Float64:
                    body(Element<double>{});
                    break;
                default:
                    FAIL() << "this case takes a real element type";
            }
        }

        /**
         * @brief Calls body with the Element of a complex data type.
         * @param type Complex64 selects the double pair; any other type selects the float pair.
         * @param body Callable taking an Element.
         */
        template<typename Body>
        void with_complex_element(Core::DataType type, Body&& body)
        {
            if (type == Core::DataType::Complex64)
                body(Element<std::complex<double>>{});
            else
                body(Element<std::complex<float>>{});
        }

        /** @brief Returns the unit roundoff of the component type R of an element type. */
        template<typename R>
        [[nodiscard]] constexpr double unit_roundoff_of()
        {
            if constexpr (std::is_same_v<R, sycl::half>)
                return K_HALF_UNIT_ROUNDOFF;
            else
                return TestSupport::unit_roundoff<R>();
        }

        /** @brief Returns whether the test device computes in the component type of a data type. */
        [[nodiscard]] inline bool device_runs(Core::DataType type)
        {
            const sycl::device device = TestEnvironment::queue().get_device();
            if (type == Core::DataType::Half)
                return device.has(sycl::aspect::fp16);
            if (type == Core::DataType::Float64 || type == Core::DataType::Complex64)
                return device.has(sycl::aspect::fp64);
            return true;
        }

        /** @brief Starts the engine of a suite in a layout with a kernel selected. */
        class BlasEngineStart : public SushiBLASTest
        {
            protected:
                /** @brief Rebuilds the engine in a layout and selects a kernel. */
                void start(Core::Layout layout, ExecutionMode mode)
                {
                    this->reinit_engine(layout);
                    this->set_mode(mode);
                }
        };

        /**
         * @brief Runs a case once per row: skips a device that lacks the element type of the row,
         *        then starts the engine in the layout and kernel of the row through start.
         * @tparam Base The suite fixture; it provides start(layout, mode).
         * @tparam Axis NoAxis, or the type of the value that completes each row.
         */
        template<typename Base, typename Axis>
        class BlasRowCase : public Base, public ::testing::WithParamInterface<Row<Axis>>
        {
            protected:
                /** @brief Starts the engine of the row, or skips the case. */
                void SetUp() override
                {
                    Base::SetUp();
                    if (!device_runs(this->GetParam().type))
                        GTEST_SKIP() << "the device does not compute in this element type";
                    this->start(this->GetParam().layout, this->GetParam().mode);
                }
        };
    } // namespace Test
} // namespace SushiBLAS
