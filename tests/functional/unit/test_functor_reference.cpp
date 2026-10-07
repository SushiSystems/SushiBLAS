/**************************************************************************/
/* test_functor_reference.cpp                                             */
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
 * @file test_functor_reference.cpp
 * @brief Tests that the functor seams give the same result on the host path as on the SYCL path.
 * @author Mustafa Garip
 */

#include <array>
#include <cmath>
#include <limits>
#include <string>
#include <vector>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <utility>
#include <optional>
#include <exception>
#include <stdexcept>
#include <functional>
#include <type_traits>
#include <initializer_list>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <SushiBLAS/SushiBLAS.h>
#include <SushiBLAS/engine/math/elementwise/dispatch.hpp>
#include <SushiRuntime/SushiRuntime.h>
#include "../../common/test_utils.hpp"
#include "../../common/graph_probe.hpp"

namespace SushiBLAS
{
    namespace Test
    {
        using namespace SushiRuntime::Graph::Literals;

        namespace
        {
            /// @brief The forced modes every case is recorded under.
            constexpr ExecutionMode K_FORCED_MODES[] = {ExecutionMode::ForceReference,
                                                        ExecutionMode::ForceSycl};

            /// @brief The modes that run the SYCL kernel.
            constexpr ExecutionMode K_SYCL_MODES[] = {ExecutionMode::Auto,
                                                      ExecutionMode::ForceSycl};

            /// @brief Element count of every case but the fused update.
            constexpr int64_t K_ELEMENTS = 1000;

            /// @brief Element count of the fused update.
            constexpr int64_t K_FUSED_ELEMENTS = 100000;

            /// @brief The FLOAT32 tolerance the exp, log, sqrt, sin and cos test files name.
            constexpr double K_NAMED_FLOAT32_TOLERANCE = 1e-6;

            /// @brief The FLOAT64 tolerance the same five files name.
            constexpr double K_NAMED_FLOAT64_TOLERANCE = 1e-12;

            /// @brief verify_tensor's default; every other math operation's value test uses it.
            constexpr double K_DEFAULT_TOLERANCE = 1e-4;

            /// @brief Absolute tolerance on a sum of small whole numbers: it is exact.
            constexpr double K_EXACT_TOLERANCE = 0.0;

            /// @brief Lower clamp bound; it truncates to -1 for an integer element.
            constexpr double K_CLAMP_LOW = -1.3;

            /// @brief Upper clamp bound; it truncates to 2 for an integer element.
            constexpr double K_CLAMP_HIGH = 2.7;

            /// @brief Slope of the leaky_relu and elu cases.
            constexpr double K_SLOPE = 0.3;

            /// @brief Exponent of the pow case.
            constexpr double K_EXPONENT = 3.0;

            /** @brief Holds the tensors one recorded case may touch. */
            struct CaseTensors
            {
                Tensor a;      /**< First input; a group's cases share it and none writes it. */
                Tensor b;      /**< Second input, shared and read-only as a is. */
                Tensor c;      /**< Third input, shared and read-only as a is. */
                Tensor t;      /**< In-place operand of this case alone. */
                Tensor out;    /**< Output of this case alone, as long as t. */
                Tensor scalar; /**< One-element output of this case alone. */
            };

            /** @brief Names one operation call and records it on the tensors it is given. */
            struct FunctorCase
            {
                /** @brief The operation's name in a failure message. */
                const char* name;

                /** @brief Records the call and returns the tensor that receives its result. */
                std::function<Tensor(TaskRecorder&, CaseTensors&)> record;
            };

            /** @brief Pairs an operation call with the tolerance its own value test allows. */
            struct ToleratedCase
            {
                /** @brief The call. */
                FunctorCase call;

                /** @brief Absolute tolerance between the two paths in FLOAT32. */
                double float32_tolerance;

                /** @brief Absolute tolerance between the two paths in FLOAT64. */
                double float64_tolerance;
            };

            /** @brief Holds the host values the shared inputs of a group of cases start from. */
            template<typename T>
            struct HostInputs
            {
                std::vector<T> a; /**< Values of CaseTensors::a and of every in-place operand. */
                std::vector<T> b; /**< Values of CaseTensors::b. */
                std::vector<T> c; /**< Values of CaseTensors::c; empty when no case reads it. */
            };

            /** @brief Holds what one case wrote under each forced mode. */
            template<typename T>
            struct ModeResults
            {
                /** @brief Returns the elements the given forced mode wrote. */
                [[nodiscard]] std::vector<T>& in(ExecutionMode mode) noexcept
                {
                    return mode == ExecutionMode::ForceReference ? reference : device;
                }

                const char* name = "";    /**< The operation's name in a failure message. */
                std::vector<T> reference; /**< The result under FORCE_REFERENCE. */
                std::vector<T> device;    /**< The result under FORCE_SYCL. */
            };

            /** @brief Returns value as an element of T, through FLOAT32 for HALF. */
            template<typename T>
            T element_of(double value)
            {
                if constexpr (std::is_same_v<T, sycl::half>)
                    return sycl::half(static_cast<float>(value));
                else
                    return static_cast<T>(value);
            }

            /** @brief Returns an element in a type a failure message can print. */
            template<typename T>
            auto printable(T value)
            {
                if constexpr (std::is_same_v<T, sycl::half>)
                    return static_cast<float>(value);
                else
                    return value;
            }

            /** @brief Reports whether an element is a NaN; an integer never is. */
            template<typename T>
            bool is_nan(T value)
            {
                if constexpr (std::is_integral_v<T>)
                    return false;
                else
                    return std::isnan(static_cast<double>(printable(value)));
            }

            /**
             * @brief Returns count elements of T, element i holding `value_at(i)`.
             * @param value_at Callable `(int64_t) -> T`.
             */
            template<typename T, typename ValueAt>
            std::vector<T> filled(int64_t count, ValueAt value_at)
            {
                std::vector<T> values(static_cast<std::size_t>(count));
                for (int64_t i = 0; i < count; ++i)
                    values[static_cast<std::size_t>(i)] = value_at(i);
                return values;
            }

            /** @brief Returns `(((i * 37) % 23) - 11.5) / divisor`, which is never zero. */
            template<typename T>
            std::vector<T> signed_pattern(int64_t count, double divisor = 1.0)
            {
                return filled<T>(count, [divisor](int64_t i) -> T
                {
                    return element_of<T>((static_cast<double>((i * 37) % 23) - 11.5) / divisor);
                });
            }

            /** @brief Returns `(((i * 17) % 13) - 6.25) / divisor`, which is never zero. */
            template<typename T>
            std::vector<T> offset_pattern(int64_t count, double divisor = 1.0)
            {
                return filled<T>(count, [divisor](int64_t i) -> T
                {
                    return element_of<T>((static_cast<double>((i * 17) % 13) - 6.25) / divisor);
                });
            }

            /** @brief Returns `(((i * 37) % 23) + 4) / 16`, between 0.25 and 1.625. */
            template<typename T>
            std::vector<T> positive_pattern(int64_t count)
            {
                return filled<T>(count, [](int64_t i) -> T
                {
                    return element_of<T>((static_cast<double>((i * 37) % 23) + 4.0) / 16.0);
                });
            }

            /** @brief Returns `first + (i % modulus)`. */
            template<typename T>
            std::vector<T> cycle(int64_t count, int64_t modulus, int64_t first = 0)
            {
                return filled<T>(count, [modulus, first](int64_t i) -> T
                {
                    return element_of<T>(static_cast<double>(first + i % modulus));
                });
            }

            /** @brief Returns integers at both ends of T's range and around zero, by i modulo 3. */
            template<typename T>
            std::vector<T> wrapping_pattern(int64_t count)
            {
                return filled<T>(count, [](int64_t i) -> T
                {
                    const T step = static_cast<T>((i * 37) % 23);
                    if (i % 3 == 0)
                        return static_cast<T>(std::numeric_limits<T>::max() - step);
                    if (i % 3 == 1)
                        return static_cast<T>(std::numeric_limits<T>::min() + step);
                    return static_cast<T>(i % 7 - 3);
                });
            }

            /**
             * @brief Returns first's element at every fifth index and, elsewhere, a step of 1 to 13
             *        whose sign alternates.
             */
            template<typename T>
            std::vector<T> step_pattern(const std::vector<T>& first)
            {
                return filled<T>(static_cast<int64_t>(first.size()), [&first](int64_t i) -> T
                {
                    if (i % 5 == 0)
                        return first[static_cast<std::size_t>(i)];

                    const T step = static_cast<T>((i * 17) % 13 + 1);
                    return i % 2 == 0 ? step : static_cast<T>(-step);
                });
            }

            /** @brief Returns the case `out = op(a, b)` of a view's three-tensor operation. */
            template<typename Ops>
            FunctorCase binary_case(const char* name,
                                    TaskHandle (Ops::*op)(const Tensor&, const Tensor&, Tensor&))
            {
                return {name, [op](TaskRecorder& recorder, CaseTensors& on)
                {
                    (Ops(recorder).*op)(on.a, on.b, on.out);
                    return on.out;
                }};
            }

            /** @brief Returns the case `t = op(t)` of a view's in-place operation. */
            template<typename Ops>
            FunctorCase inplace_case(const char* name, TaskHandle (Ops::*op)(Tensor&))
            {
                return {name, [op](TaskRecorder& recorder, CaseTensors& on)
                {
                    (Ops(recorder).*op)(on.t);
                    return on.t;
                }};
            }

            /** @brief Returns the case `out = op(a)` of a view's two-tensor operation. */
            template<typename Ops>
            FunctorCase unary_case(const char* name, TaskHandle (Ops::*op)(const Tensor&, Tensor&))
            {
                return {name, [op](TaskRecorder& recorder, CaseTensors& on)
                {
                    (Ops(recorder).*op)(on.a, on.out);
                    return on.out;
                }};
            }

            /** @brief Returns the case `scalar = op(a)` of a view's scan over one tensor. */
            template<typename Ops>
            FunctorCase scan_case(const char* name, TaskHandle (Ops::*op)(const Tensor&, Tensor&))
            {
                return {name, [op](TaskRecorder& recorder, CaseTensors& on)
                {
                    (Ops(recorder).*op)(on.a, on.scalar);
                    return on.scalar;
                }};
            }

            /** @brief Returns the exact operations over signed values; c selects for where. */
            std::vector<FunctorCase> arithmetic_cases()
            {
                return {
                    binary_case("add", &ElementwiseOps::add),
                    binary_case("sub", &ElementwiseOps::sub),
                    binary_case("mul", &ElementwiseOps::mul),
                    binary_case("div", &ElementwiseOps::div),
                    binary_case("min", &ElementwiseOps::min),
                    binary_case("max", &ElementwiseOps::max),
                    binary_case("fmod", &ElementwiseOps::fmod),
                    binary_case("remainder", &ElementwiseOps::remainder),
                    inplace_case("neg", &ElementwiseOps::neg),
                    inplace_case("abs", &ElementwiseOps::abs),
                    inplace_case("square", &ElementwiseOps::square),
                    inplace_case("reciprocal", &ElementwiseOps::reciprocal),
                    inplace_case("floor", &ElementwiseOps::floor),
                    inplace_case("ceil", &ElementwiseOps::ceil),
                    inplace_case("round", &ElementwiseOps::round),
                    {"clamp", [](TaskRecorder& recorder, CaseTensors& on)
                     {
                         ElementwiseOps(recorder).clamp(on.t, K_CLAMP_LOW, K_CLAMP_HIGH);
                         return on.t;
                     }},
                    inplace_case("relu", &NonLinearOps::relu),
                    binary_case("relu_backward", &NonLinearOps::relu_backward),
                    {"leaky_relu", [](TaskRecorder& recorder, CaseTensors& on)
                     {
                         NonLinearOps(recorder).leaky_relu(on.t, K_SLOPE);
                         return on.t;
                     }},
                    {"leaky_relu_backward", [](TaskRecorder& recorder, CaseTensors& on)
                     {
                         NonLinearOps(recorder).leaky_relu_backward(on.a, on.b, on.out, K_SLOPE);
                         return on.out;
                     }},
                    binary_case("sigmoid_backward", &NonLinearOps::sigmoid_backward),
                    binary_case("tanh_backward", &NonLinearOps::tanh_backward),
                    {"where", [](TaskRecorder& recorder, CaseTensors& on)
                     {
                         LogicOps(recorder).where(on.c, on.a, on.b, on.out);
                         return on.out;
                     }},
                    scan_case("all of non-zeros", &LogicOps::all),
                    scan_case("any of non-zeros", &LogicOps::any),
                };
            }

            /** @brief Returns the comparisons and truth operations on whole numbers; c is zero. */
            std::vector<FunctorCase> truth_cases()
            {
                return {
                    binary_case("equal", &LogicOps::equal),
                    binary_case("not_equal", &LogicOps::not_equal),
                    binary_case("greater", &LogicOps::greater),
                    binary_case("less", &LogicOps::less),
                    binary_case("greater_equal", &LogicOps::greater_equal),
                    binary_case("less_equal", &LogicOps::less_equal),
                    binary_case("logical_and", &LogicOps::logical_and),
                    binary_case("logical_or", &LogicOps::logical_or),
                    binary_case("logical_xor", &LogicOps::logical_xor),
                    unary_case("logical_not", &LogicOps::logical_not),
                    scan_case("all of a mix", &LogicOps::all),
                    scan_case("any of a mix", &LogicOps::any),
                    {"all of zeros", [](TaskRecorder& recorder, CaseTensors& on)
                     {
                         LogicOps(recorder).all(on.c, on.scalar);
                         return on.scalar;
                     }},
                    {"any of zeros", [](TaskRecorder& recorder, CaseTensors& on)
                     {
                         LogicOps(recorder).any(on.c, on.scalar);
                         return on.scalar;
                     }},
                };
            }

            /** @brief Returns the operations an integer dtype serves; c holds no zero. */
            std::vector<FunctorCase> integer_cases()
            {
                return {
                    binary_case("add", &ElementwiseOps::add),
                    binary_case("sub", &ElementwiseOps::sub),
                    binary_case("mul", &ElementwiseOps::mul),
                    {"clamp", [](TaskRecorder& recorder, CaseTensors& on)
                     {
                         ElementwiseOps(recorder).clamp(on.t, K_CLAMP_LOW, K_CLAMP_HIGH);
                         return on.t;
                     }},
                    binary_case("equal", &LogicOps::equal),
                    binary_case("not_equal", &LogicOps::not_equal),
                    binary_case("greater", &LogicOps::greater),
                    binary_case("less", &LogicOps::less),
                    binary_case("greater_equal", &LogicOps::greater_equal),
                    binary_case("less_equal", &LogicOps::less_equal),
                    scan_case("all of a mix", &LogicOps::all),
                    scan_case("any of a mix", &LogicOps::any),
                    {"all of non-zeros", [](TaskRecorder& recorder, CaseTensors& on)
                     {
                         LogicOps(recorder).all(on.c, on.scalar);
                         return on.scalar;
                     }},
                };
            }

            /** @brief Returns the HALF operations whose host result is correctly rounded. */
            std::vector<FunctorCase> half_cases()
            {
                return {
                    binary_case("add", &ElementwiseOps::add),
                    binary_case("sub", &ElementwiseOps::sub),
                    binary_case("mul", &ElementwiseOps::mul),
                    binary_case("div", &ElementwiseOps::div),
                    inplace_case("relu", &NonLinearOps::relu),
                };
            }

            /** @brief Returns the math-library operations defined on negative inputs. */
            std::vector<ToleratedCase> math_cases()
            {
                const auto named = [](FunctorCase call)
                {
                    return ToleratedCase{std::move(call), K_NAMED_FLOAT32_TOLERANCE,
                                         K_NAMED_FLOAT64_TOLERANCE};
                };
                const auto defaulted = [](FunctorCase call)
                {
                    return ToleratedCase{std::move(call), K_DEFAULT_TOLERANCE, K_DEFAULT_TOLERANCE};
                };

                return {
                    named(inplace_case("exp", &ElementwiseOps::exp)),
                    named(inplace_case("sin", &ElementwiseOps::sin)),
                    named(inplace_case("cos", &ElementwiseOps::cos)),
                    defaulted(inplace_case("tan", &ElementwiseOps::tan)),
                    defaulted(inplace_case("atan", &ElementwiseOps::atan)),
                    defaulted(inplace_case("tanh", &NonLinearOps::tanh)),
                    defaulted(inplace_case("sigmoid", &NonLinearOps::sigmoid)),
                    defaulted(inplace_case("silu", &NonLinearOps::silu)),
                    defaulted(inplace_case("gelu", &NonLinearOps::gelu)),
                    defaulted(inplace_case("softplus", &NonLinearOps::softplus)),
                    defaulted({"elu", [](TaskRecorder& recorder, CaseTensors& on)
                               {
                                   NonLinearOps(recorder).elu(on.t, K_SLOPE);
                                   return on.t;
                               }}),
                    defaulted({"elu_backward", [](TaskRecorder& recorder, CaseTensors& on)
                               {
                                   NonLinearOps(recorder).elu_backward(on.a, on.b, on.out, K_SLOPE);
                                   return on.out;
                               }}),
                    defaulted(binary_case("silu_backward", &NonLinearOps::silu_backward)),
                    defaulted(binary_case("gelu_backward", &NonLinearOps::gelu_backward)),
                    defaulted(binary_case("softplus_backward", &NonLinearOps::softplus_backward)),
                };
            }

            /** @brief Returns the math-library operations that need a positive input. */
            std::vector<ToleratedCase> positive_math_cases()
            {
                return {
                    {inplace_case("log", &ElementwiseOps::log), K_NAMED_FLOAT32_TOLERANCE,
                     K_NAMED_FLOAT64_TOLERANCE},
                    {inplace_case("sqrt", &ElementwiseOps::sqrt), K_NAMED_FLOAT32_TOLERANCE,
                     K_NAMED_FLOAT64_TOLERANCE},
                    {{"pow", [](TaskRecorder& recorder, CaseTensors& on)
                      {
                          ElementwiseOps(recorder).pow(on.t, K_EXPONENT);
                          return on.t;
                      }},
                     K_DEFAULT_TOLERANCE, K_DEFAULT_TOLERANCE},
                };
            }

            /** @brief Adds half of x to y, then negates x. */
            struct StepAndFlip
            {
                /** @brief Updates one element of each operand. */
                template<typename T>
                void operator()(T& y, T& x) const
                {
                    y = y + T(0.5) * x;
                    x = -x;
                }
            };

            /** @brief Returns what record throws as std::invalid_argument, or an empty string. */
            template<typename Record>
            std::string refusal_of(const Record& record)
            {
                try
                {
                    record();
                }
                catch (const std::invalid_argument& error)
                {
                    return error.what();
                }

                return std::string();
            }
        } // namespace

        /** @brief Records operation calls under both forced modes and compares what they wrote. */
        class Unit_FunctorReference : public SushiBLASTest
        {
            protected:
                /** @brief Reports whether the test device computes in HALF. */
                [[nodiscard]] static bool device_has_fp16()
                {
                    return TestEnvironment::get_context().get_queue().get_device().has(
                        sycl::aspect::fp16);
                }

                /** @brief Returns a vector tensor of values, or one without storage for none. */
                template<typename T>
                [[nodiscard]] Tensor tensor_of(const std::vector<T>& values)
                {
                    if (values.empty())
                        return Tensor();

                    return create_initialized_tensor<T>({static_cast<int64_t>(values.size())},
                                                        values);
                }

                /**
                 * @brief Returns a FLOAT32 vector of count elements in device-only USM, or nothing.
                 * @return Empty when the allocation fails or the pointer is not usm::alloc::device.
                 */
                [[nodiscard]] std::optional<Tensor> device_only_vector(int64_t count)
                {
                    try
                    {
                        Tensor t = engine->create_tensor({count}, Core::DataType::Float32,
                                                         SushiRuntime::API::Residency::Device);
                        const sycl::context context =
                            TestEnvironment::get_context().get_queue().get_context();
                        if (sycl::get_pointer_type(t.data(), context) != sycl::usm::alloc::device)
                            return std::nullopt;
                        return t;
                    }
                    catch (const std::exception&)
                    {
                        return std::nullopt;
                    }
                }

                /**
                 * @brief Records every case under both forced modes, runs both graphs and returns
                 *        what each case wrote.
                 * @param inputs Supplies a, b and c; every case gets its own t, out and scalar.
                 * @param seed The values each case's in-place operand starts from; its size is the
                 *        element count of out.
                 */
                template<typename T>
                [[nodiscard]] std::vector<ModeResults<T>> run_in_both_modes(
                    const std::vector<FunctorCase>& cases, const CaseTensors& inputs,
                    const std::vector<T>& seed)
                {
                    std::vector<ModeResults<T>> results(cases.size());
                    for (std::size_t k = 0; k < cases.size(); ++k)
                        results[k].name = cases[k].name;

                    for (const ExecutionMode mode : K_FORCED_MODES)
                    {
                        GraphProbe probe(mode);

                        std::vector<Tensor> written;
                        written.reserve(cases.size());
                        for (const FunctorCase& c : cases)
                        {
                            CaseTensors on = inputs;
                            on.t = tensor_of(seed);
                            on.out = tensor_of(std::vector<T>(seed.size()));
                            on.scalar = tensor_of(std::vector<T>(1));
                            written.push_back(c.record(probe.get_recorder(), on));
                        }
                        probe.execute();

                        for (std::size_t k = 0; k < cases.size(); ++k)
                            results[k].in(mode) = read_tensor<T>(written[k]);
                    }

                    return results;
                }

                /** @brief Runs the cases as the other overload does, on inputs built from host. */
                template<typename T>
                [[nodiscard]] std::vector<ModeResults<T>> run_in_both_modes(
                    const std::vector<FunctorCase>& cases, const HostInputs<T>& host)
                {
                    CaseTensors inputs;
                    inputs.a = tensor_of(host.a);
                    inputs.b = tensor_of(host.b);
                    inputs.c = tensor_of(host.c);

                    return run_in_both_modes<T>(cases, inputs, host.a);
                }

                /** @brief Expects both modes to have written equal bits, a NaN matching any NaN. */
                template<typename T>
                static void expect_equal_bits(const ModeResults<T>& result)
                {
                    ASSERT_EQ(result.reference.size(), result.device.size()) << result.name;

                    for (std::size_t i = 0; i < result.reference.size(); ++i)
                    {
                        const T reference = result.reference[i];
                        const T device = result.device[i];
                        if (std::memcmp(&reference, &device, sizeof(T)) == 0)
                            continue;
                        if (is_nan(reference) && is_nan(device))
                            continue;

                        ADD_FAILURE() << result.name << " differs at element " << i
                                      << ": FORCE_REFERENCE wrote " << printable(reference)
                                      << ", FORCE_SYCL wrote " << printable(device);
                        return;
                    }
                }

                /** @brief Expects every element of the two modes to differ by at most tolerance. */
                template<typename T>
                static void expect_within(const ModeResults<T>& result, double tolerance)
                {
                    ASSERT_EQ(result.reference.size(), result.device.size()) << result.name;

                    for (std::size_t i = 0; i < result.reference.size(); ++i)
                    {
                        const double reference = static_cast<double>(result.reference[i]);
                        const double device = static_cast<double>(result.device[i]);
                        if (std::abs(reference - device) <= tolerance)
                            continue;

                        ADD_FAILURE() << result.name << " differs at element " << i
                                      << " by more than " << tolerance << ": FORCE_REFERENCE wrote "
                                      << reference << ", FORCE_SYCL wrote " << device;
                        return;
                    }
                }

                /** @brief Expects every exact operation to agree to the bit in real dtype T. */
                template<typename T>
                void expect_exact_operations_agree()
                {
                    SCOPED_TRACE(Core::to_string(get_sushiblas_dtype<T>()));

                    const HostInputs<T> arithmetic{signed_pattern<T>(K_ELEMENTS),
                                                   offset_pattern<T>(K_ELEMENTS),
                                                   cycle<T>(K_ELEMENTS, 3)};
                    for (const ModeResults<T>& result :
                         run_in_both_modes(arithmetic_cases(), arithmetic))
                        expect_equal_bits(result);

                    const HostInputs<T> truth{cycle<T>(K_ELEMENTS, 3), cycle<T>(K_ELEMENTS, 2),
                                              cycle<T>(K_ELEMENTS, 1)};
                    for (const ModeResults<T>& result : run_in_both_modes(truth_cases(), truth))
                        expect_equal_bits(result);
                }

                /** @brief Expects every integer operation to agree to the bit in dtype T. */
                template<typename T>
                void expect_integer_operations_agree()
                {
                    SCOPED_TRACE(Core::to_string(get_sushiblas_dtype<T>()));

                    HostInputs<T> host;
                    host.a = wrapping_pattern<T>(K_ELEMENTS);
                    host.b = step_pattern<T>(host.a);
                    host.c = cycle<T>(K_ELEMENTS, 3, 1);

                    for (const ModeResults<T>& result : run_in_both_modes(integer_cases(), host))
                        expect_equal_bits(result);
                }

                /** @brief Expects every math-library operation to agree within its tolerance. */
                template<typename T>
                void expect_math_operations_agree()
                {
                    SCOPED_TRACE(Core::to_string(get_sushiblas_dtype<T>()));

                    // Every input keeps exp, log and sqrt below 2, where their tolerance was set.
                    const HostInputs<T> around_zero{signed_pattern<T>(K_ELEMENTS, 16.0),
                                                    offset_pattern<T>(K_ELEMENTS, 8.0), {}};
                    expect_within_tolerances(math_cases(), around_zero);

                    const HostInputs<T> positive{positive_pattern<T>(K_ELEMENTS), {}, {}};
                    expect_within_tolerances(positive_math_cases(), positive);
                }

                /** @brief Runs the tolerated cases on host and holds each to its tolerance. */
                template<typename T>
                void expect_within_tolerances(const std::vector<ToleratedCase>& tolerated,
                                              const HostInputs<T>& host)
                {
                    std::vector<FunctorCase> cases;
                    cases.reserve(tolerated.size());
                    for (const ToleratedCase& c : tolerated)
                        cases.push_back(c.call);

                    const std::vector<ModeResults<T>> results = run_in_both_modes(cases, host);
                    for (std::size_t k = 0; k < results.size(); ++k)
                    {
                        expect_within(results[k], std::is_same_v<T, float>
                                                      ? tolerated[k].float32_tolerance
                                                      : tolerated[k].float64_tolerance);
                    }
                }

                /** @brief Expects convert from Src to Dst to agree to the bit across both modes. */
                template<typename Src, typename Dst>
                void expect_convert_agrees()
                {
                    SCOPED_TRACE(std::string(Core::to_string(get_sushiblas_dtype<Src>())) + " to " +
                                 Core::to_string(get_sushiblas_dtype<Dst>()));

                    CaseTensors inputs;
                    inputs.a = tensor_of(signed_pattern<Src>(K_ELEMENTS, 3.0));

                    const std::vector<FunctorCase> cases = {
                        {"convert", [](TaskRecorder& recorder, CaseTensors& on)
                         {
                             ElementwiseOps(recorder).convert(on.a, on.out);
                             return on.out;
                         }},
                    };

                    const std::vector<Dst> seed(static_cast<std::size_t>(K_ELEMENTS));
                    for (const ModeResults<Dst>& result :
                         run_in_both_modes<Dst>(cases, inputs, seed))
                        expect_equal_bits(result);
                }
        };

        /** @brief Tests that the operations built from exact arithmetic agree to the bit. */
        TEST_F(Unit_FunctorReference, ExactOperationsAgreeToTheBit)
        {
            expect_exact_operations_agree<float>();
            expect_exact_operations_agree<double>();
        }

        /** @brief Tests that integer operations agree to the bit, wrapping included. */
        TEST_F(Unit_FunctorReference, IntegerOperationsAgreeToTheBit)
        {
            expect_integer_operations_agree<int32_t>();
            expect_integer_operations_agree<int64_t>();
        }

        /** @brief Tests that convert agrees to the bit for each ordered pair of real dtypes. */
        TEST_F(Unit_FunctorReference, ConvertAgreesToTheBit)
        {
            expect_convert_agrees<float, double>();
            expect_convert_agrees<double, float>();

            if (!device_has_fp16())
                GTEST_SKIP() << "the device does not report the fp16 aspect, so no HALF pair ran";

            expect_convert_agrees<sycl::half, float>();
            expect_convert_agrees<float, sycl::half>();
            expect_convert_agrees<sycl::half, double>();
            expect_convert_agrees<double, sycl::half>();
        }

        /** @brief Tests that HALF arithmetic agrees to the bit on normal and zero results. */
        TEST_F(Unit_FunctorReference, HalfOperationsAgreeOutsideTheSubnormalRange)
        {
            if (!device_has_fp16())
                GTEST_SKIP() << "the device does not report the fp16 aspect";

            HostInputs<sycl::half> host;
            host.a = signed_pattern<sycl::half>(K_ELEMENTS, 3.0);
            host.b = filled<sycl::half>(K_ELEMENTS, [](int64_t i) -> sycl::half
            {
                return element_of<sycl::half>((static_cast<double>((i * 17) % 13) + 1.25) / 7.0);
            });

            for (const ModeResults<sycl::half>& result : run_in_both_modes(half_cases(), host))
                expect_equal_bits(result);
        }

        /** @brief Tests that the math-library operations agree within their own tolerance. */
        TEST_F(Unit_FunctorReference, MathOperationsAgreeWithinTheirOwnTolerance)
        {
            expect_math_operations_agree<float>();
            expect_math_operations_agree<double>();
        }

        /** @brief Tests that a fused update over 100 000 elements agrees to the bit. */
        TEST_F(Unit_FunctorReference, AFusedUpdateAgreesToTheBit)
        {
            const std::vector<float> y_values = signed_pattern<float>(K_FUSED_ELEMENTS);
            const std::vector<float> x_values = offset_pattern<float>(K_FUSED_ELEMENTS, 3.0);

            ModeResults<float> y_result;
            ModeResults<float> x_result;
            y_result.name = "the stepped operand";
            x_result.name = "the negated operand";

            for (const ExecutionMode mode : K_FORCED_MODES)
            {
                GraphProbe probe(mode);
                Tensor y = tensor_of(y_values);
                Tensor x = tensor_of(x_values);

                Dispatch::execute_zip_inplace(probe.get_recorder(), std::array{&y, &x},
                                              "test.step_and_flip", "test.step_and_flip"_op,
                                              StepAndFlip{});
                probe.execute();

                y_result.in(mode) = read_tensor<float>(y);
                x_result.in(mode) = read_tensor<float>(x);
            }

            expect_equal_bits(y_result);
            expect_equal_bits(x_result);
        }

        /** @brief Tests that AUTO takes the device-only operand that FORCE_REFERENCE refuses. */
        TEST_F(Unit_FunctorReference, AutoRunsTheSyclPath)
        {
            std::optional<Tensor> a = device_only_vector(4);
            if (!a)
                GTEST_SKIP() << "This device offers no device-only USM.";
            fill_tensor(*a, {1.0f, 2.0f, 3.0f, 4.0f});
            const Tensor b = tensor_of(std::vector<float>{10.0f, 20.0f, 30.0f, 40.0f});

            for (const ExecutionMode mode : K_SYCL_MODES)
            {
                GraphProbe probe(mode);
                Tensor c = engine->create_tensor({4});

                EXPECT_NO_THROW(ElementwiseOps(probe.get_recorder()).add(*a, b, c));
                probe.execute();

                verify_tensor(c, {11.0f, 22.0f, 33.0f, 44.0f}, K_EXACT_TOLERANCE);
            }

            GraphProbe reference(ExecutionMode::ForceReference);
            Tensor c = engine->create_tensor({4});
            const std::string message =
                refusal_of([&] { ElementwiseOps(reference.get_recorder()).add(*a, b, c); });

            EXPECT_NE(message.find("FORCE_REFERENCE"), std::string::npos) << message;
            EXPECT_TRUE(reference.take_nodes().empty());
        }

        /** @brief Tests that add over empty tensors records no task under either forced mode. */
        TEST_F(Unit_FunctorReference, AnEmptyTensorRecordsNothingInEitherMode)
        {
            for (const ExecutionMode mode : K_FORCED_MODES)
            {
                GraphProbe probe(mode);
                Tensor a = engine->create_tensor({0, 3});
                Tensor b = engine->create_tensor({0, 3});
                Tensor c = engine->create_tensor({0, 3});

                EXPECT_FALSE(ElementwiseOps(probe.get_recorder()).add(a, b, c).is_valid());
                EXPECT_TRUE(probe.take_nodes().empty());
            }
        }
    } // namespace Test
} // namespace SushiBLAS
