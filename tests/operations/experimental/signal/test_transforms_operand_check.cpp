/**************************************************************************/
/* test_transforms_operand_check.cpp                                      */
/**************************************************************************/
/*                          This file is part of:                         */
/*                               SushiSkills                              */
/*               https://github.com/SushiSystems/SushiSkills              */
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

/**
 * @file test_transforms_operand_check.cpp
 * @brief Tests that check_transform_operands accepts a well-formed forward or inverse transform
 *        and refuses each malformed one naming the operand, on operands built by hand.
 * @author Mustafa Garip
 */

#include <SushiBLAS/experimental/signal/transforms_operand_check.hpp>

#include <string>
#include <vector>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include "support/forced_mode_checks.hpp"
#include "support/hand_built_operand.hpp"

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            /// @brief The made-up allocation of the input and of the output of a well-formed call.
            constexpr std::size_t K_INPUT_ALLOCATION = 1;
            constexpr std::size_t K_OUTPUT_ALLOCATION = 2;

            /// @brief The rows, the columns and the elements of a transform the calls below describe.
            constexpr std::size_t K_ROWS = 2;
            constexpr std::size_t K_COLUMNS = 4;
            constexpr int64_t K_SAMPLES = 8;
            constexpr int64_t K_FULL_SPECTRUM = 8;
            constexpr int64_t K_HALF_SPECTRUM = 6;

            /**
             * @brief Holds the two operands and the shape of one transform of two rows of four
             *        samples, forward or inverse, complex or real.
             */
            struct TransformCall
            {
                HandBuiltOperand input;
                HandBuiltOperand output;
                FftShape shape;
                bool forward = true;
                bool wide = false;

                /** @brief Builds a well-formed call of the kind and direction, in FLOAT64 parts when wide. */
                static TransformCall make(bool forward, FftKind kind, bool wide = false)
                {
                    const Core::DataType real = wide ? Core::DataType::Float64 : Core::DataType::Float32;
                    const Core::DataType complex = wide ? Core::DataType::Complex64
                                                        : Core::DataType::Complex32;
                    const bool halved = kind == FftKind::RealToComplex;
                    const Core::DataType sample_dtype = halved ? real : complex;
                    const int64_t spectrum_elements = halved ? K_HALF_SPECTRUM : K_FULL_SPECTRUM;

                    HandBuiltOperand samples({K_SAMPLES}, sample_dtype,
                                             forward ? K_INPUT_ALLOCATION : K_OUTPUT_ALLOCATION);
                    HandBuiltOperand spectrum({spectrum_elements}, complex,
                                              forward ? K_OUTPUT_ALLOCATION : K_INPUT_ALLOCATION);
                    return TransformCall{forward ? samples : spectrum, forward ? spectrum : samples,
                                         FftShape{K_ROWS, K_COLUMNS, 1, kind}, forward, wide};
                }

                /** @brief Returns the operation id the messages name. */
                const char* op() const
                {
                    return forward ? "signal.fft" : "signal.inverse_fft";
                }

                /** @brief Returns the operand that holds the samples. */
                HandBuiltOperand& samples()
                {
                    return forward ? input : output;
                }

                /** @brief Returns the operand that holds the spectrum. */
                HandBuiltOperand& spectrum()
                {
                    return forward ? output : input;
                }

                /** @brief Returns the operand that holds the samples. */
                const HandBuiltOperand& samples() const
                {
                    return forward ? input : output;
                }

                /** @brief Returns the operand that holds the spectrum. */
                const HandBuiltOperand& spectrum() const
                {
                    return forward ? output : input;
                }

                /** @brief Returns the name a message gives the samples. */
                const char* samples_name() const
                {
                    return forward ? "input" : "output";
                }

                /** @brief Returns the name a message gives the spectrum. */
                const char* spectrum_name() const
                {
                    return forward ? "output" : "input";
                }

                /** @brief Returns the label of the call for a failure message. */
                std::string label() const
                {
                    return std::string(forward ? "forward " : "inverse ") +
                           (shape.kind == FftKind::RealToComplex ? "real" : "complex") +
                           (wide ? " wide" : "");
                }

                /** @brief Runs the check and returns the dtype of the parts of the spectrum. */
                Core::DataType check() const
                {
                    return Internal::check_transform_operands(input.placed(), output.placed(),
                                                              shape, forward, op());
                }
            };

            /** @brief Returns the four calls: forward and inverse, complex and real. */
            std::vector<TransformCall> every_call(bool wide = false)
            {
                return {TransformCall::make(true, FftKind::ComplexToComplex, wide),
                        TransformCall::make(true, FftKind::RealToComplex, wide),
                        TransformCall::make(false, FftKind::ComplexToComplex, wide),
                        TransformCall::make(false, FftKind::RealToComplex, wide)};
            }
        } // namespace

        /** @brief Tests that a well-formed call of each direction and kind passes, in both widths, until it runs in place. */
        TEST(Small_TransformsOperandCheck, AWellFormedCallIsAcceptedUntilItRunsInPlace)
        {
            for (const bool wide : {false, true})
            {
                for (const TransformCall& call : every_call(wide))
                {
                    SCOPED_TRACE(call.label());
                    TransformCall in_place = call;
                    in_place.output.in_allocation(call.input.allocation());

                    EXPECT_NO_THROW(call.check());
                    EXPECT_THROW(in_place.check(), std::invalid_argument);
                }
            }
        }

        /** @brief Tests that the check reports the dtype of the real parts of the spectrum. */
        TEST(Small_TransformsOperandCheck, ThePartsOfTheSpectrumGiveThePlanDtype)
        {
            for (const TransformCall& call : every_call())
                EXPECT_EQ(std::string(Core::to_string(call.check())),
                          std::string(Core::to_string(Core::DataType::Float32))) << call.label();
            for (const TransformCall& call : every_call(true))
                EXPECT_EQ(std::string(Core::to_string(call.check())),
                          std::string(Core::to_string(Core::DataType::Float64))) << call.label();
        }

        /** @brief Tests that a shape with no rows or no columns is refused naming the shape. */
        TEST(Small_TransformsOperandCheck, AnEmptyShapeIsRefusedNamingTheShape)
        {
            for (const TransformCall& call : every_call())
            {
                SCOPED_TRACE(call.label());
                TransformCall no_columns = call;
                no_columns.shape.columns = 0;
                TransformCall no_rows = call;
                no_rows.shape.rows = 0;

                expect_refusal([&] { no_columns.check(); }, call.op(), "shape");
                expect_refusal([&] { no_rows.check(); }, call.op(), "shape");
            }
        }

        /** @brief Tests that samples whose dtype is not the dtype of the parts of the spectrum are refused naming them. */
        TEST(Small_TransformsOperandCheck, SamplesOfAnotherPrecisionAreRefusedNamingThem)
        {
            for (TransformCall call : every_call())
            {
                SCOPED_TRACE(call.label());
                const bool real = call.shape.kind == FftKind::RealToComplex;
                const Core::DataType other = real ? Core::DataType::Float64 : Core::DataType::Complex64;
                const std::size_t allocation = call.samples().allocation();
                call.samples() = HandBuiltOperand({K_SAMPLES}, other, allocation);

                expect_refusal([&] { call.check(); }, call.op(), call.samples_name());
            }
        }

        /** @brief Tests that a spectrum that is not complex is refused naming it. */
        TEST(Small_TransformsOperandCheck, ASpectrumThatIsNotComplexIsRefusedNamingIt)
        {
            for (TransformCall call : every_call())
            {
                SCOPED_TRACE(call.label());
                const int64_t elements = call.spectrum().tensor().num_elements;
                const std::size_t allocation = call.spectrum().allocation();
                call.spectrum() = HandBuiltOperand({elements}, Core::DataType::Float32, allocation);

                expect_refusal([&] { call.check(); }, call.op(), call.spectrum_name());
            }
        }

        /** @brief Tests that samples or a spectrum of the wrong element count are refused naming them. */
        TEST(Small_TransformsOperandCheck, AWrongElementCountIsRefusedNamingTheOperand)
        {
            for (const TransformCall& call : every_call())
            {
                SCOPED_TRACE(call.label());
                TransformCall short_samples = call;
                short_samples.samples() = HandBuiltOperand({K_SAMPLES - 1}, call.samples().tensor().dtype,
                                                           call.samples().allocation());
                TransformCall short_spectrum = call;
                short_spectrum.spectrum() = HandBuiltOperand({K_HALF_SPECTRUM + 1},
                                                             call.spectrum().tensor().dtype,
                                                             call.spectrum().allocation());

                expect_refusal([&] { short_samples.check(); }, call.op(), call.samples_name());
                expect_refusal([&] { short_spectrum.check(); }, call.op(), call.spectrum_name());
            }
        }

        /** @brief Tests that the half spectrum of a real transform is not a full spectrum. */
        TEST(Small_TransformsOperandCheck, ARealTransformRefusesAFullSpectrum)
        {
            for (const bool forward : {true, false})
            {
                TransformCall call = TransformCall::make(forward, FftKind::RealToComplex);
                call.spectrum() = HandBuiltOperand({K_FULL_SPECTRUM}, Core::DataType::Complex32,
                                                   call.spectrum().allocation());

                expect_refusal([&] { call.check(); }, call.op(), call.spectrum_name());
            }
        }

        /** @brief Tests that an operand without storage is refused naming it. */
        TEST(Small_TransformsOperandCheck, AnOperandWithoutStorageIsRefusedNamingIt)
        {
            for (const TransformCall& call : every_call())
            {
                SCOPED_TRACE(call.label());
                TransformCall no_input = call;
                no_input.input.without_storage();
                TransformCall no_output = call;
                no_output.output.without_storage();

                expect_refusal([&] { no_input.check(); }, call.op(), "input");
                expect_refusal([&] { no_output.check(); }, call.op(), "output");
            }
        }

        /** @brief Tests that a strided view is refused naming the operand. */
        TEST(Small_TransformsOperandCheck, AStridedViewIsRefusedNamingIt)
        {
            for (const TransformCall& call : every_call())
            {
                SCOPED_TRACE(call.label());
                TransformCall strided_input = call;
                strided_input.input.with_stride(0, 2);
                TransformCall strided_output = call;
                strided_output.output.with_stride(0, 2);

                expect_refusal([&] { strided_input.check(); }, call.op(), "input");
                expect_refusal([&] { strided_output.check(); }, call.op(), "output");
            }
        }

        /** @brief Tests that an operand of FLOAT64 parts on a device without fp64 is refused naming it. */
        TEST(Small_TransformsOperandCheck, AnOperandOnADeviceWithoutFp64IsRefusedNamingIt)
        {
            for (const TransformCall& call : every_call(true))
            {
                SCOPED_TRACE(call.label());
                TransformCall input_aside = call;
                input_aside.input.on_device_without_fp64();
                TransformCall output_aside = call;
                output_aside.output.on_device_without_fp64();

                EXPECT_THROW(input_aside.check(), UnsupportedExecution);
                EXPECT_THROW(output_aside.check(), UnsupportedExecution);
                expect_refusal([&] { input_aside.check(); }, call.op(), "input");
                expect_refusal([&] { output_aside.check(); }, call.op(), "output");
            }
        }

        /** @brief Tests that an output that shares a byte with the input, as an in-place transform does, is refused. */
        TEST(Small_TransformsOperandCheck, AnOutputThatSharesBytesWithTheInputIsRefusedNamingIt)
        {
            for (TransformCall call : every_call())
            {
                SCOPED_TRACE(call.label());
                call.output.in_allocation(call.input.allocation());

                expect_refusal([&] { call.check(); }, call.op(), "output");
            }
        }

        /** @brief Tests that an output starting where the input ends passes and one a byte earlier does not. */
        TEST(Small_TransformsOperandCheck, AnOutputStartingWhereTheInputEndsIsAcceptedAndOneByteEarlierIsRefused)
        {
            const std::size_t input_bytes = static_cast<std::size_t>(K_SAMPLES) * 8;
            TransformCall adjacent = TransformCall::make(true, FftKind::ComplexToComplex);
            adjacent.output.in_allocation(adjacent.input.allocation()).at_byte_offset(input_bytes);
            TransformCall overlapping = TransformCall::make(true, FftKind::ComplexToComplex);
            overlapping.output.in_allocation(overlapping.input.allocation())
                .at_byte_offset(input_bytes - 1);

            EXPECT_NO_THROW(adjacent.check());
            expect_refusal([&] { overlapping.check(); }, overlapping.op(), "output");
        }
    } // namespace Test
} // namespace SushiBLAS
