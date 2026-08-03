/**************************************************************************/
/* dispatch.hpp                                                           */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                SushiBLAS                               */
/*                https://github.com/SushiSystems/SushiBLAS               */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present Mustafa Garip & Sushi Systems               */
/*                                                                        */
/* Licensed under the Apache License, Version 2.0 (the "License");        */
/* you may not use this file except in compliance with the License.       */
/* You may obtain a copy of the License at                                */
/*                                                                        */
/*     http://www.apache.org/licenses/LICENSE-2.0                         */
/*                                                                        */
/* Unless required by applicable law or agreed to in writing, software    */
/* distributed under the License is distributed on an "AS IS" BASIS,      */
/* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        */
/* implied. See the License for the specific language governing           */
/* permissions and limitations under the License.                         */
/**************************************************************************/

#pragma once

#include <vector>
#include <cstddef>
#include <cstdint>
#include <complex>
#include <sycl/sycl.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/core/logger.hpp>
#include <SushiBLAS/engine/device_policy.hpp>
#include <SushiRuntime/graph/task_types.hpp>

// The shared vocabulary of SushiBLAS's generic op-dispatch seam.
//
// `SushiBLAS::Dispatch` is the low-level, functor-templated layer every op in
// this library is built on: one `execute_*` helper per op family that records a
// task on the engine's graph, switches on the tensor's runtime dtype, and
// inlines a caller-supplied functor into a single SYCL kernel. The five family
// headers are:
//
//   <SushiBLAS/engine/math/elementwise/dispatch.hpp>
//   <SushiBLAS/engine/math/nonlinear/dispatch.hpp>
//   <SushiBLAS/engine/math/reductions/dispatch.hpp>
//   <SushiBLAS/engine/math/random/dispatch.hpp>
//   <SushiBLAS/engine/logic/dispatch.hpp>
//
// These used to be private `*_internal.hpp` headers under src/, uninstalled and
// therefore unreachable from outside this tree. They are public now because
// they are the seam a layer above (an autodiff/NN framework) needs in order to
// fuse: composing several math ops into one functor and handing it to
// `execute_binary` produces ONE kernel launch instead of one per op, and it does
// so without the consumer writing a line of SYCL - the functor is a plain C++
// lambda. Keeping that capability private would have forced every consumer to
// either eat a kernel launch per elementwise op or reimplement this dispatch
// layer itself.
//
// This header carries only what all five families share; include the family
// header for the executors themselves.

namespace SushiBLAS
{
    namespace Dispatch
    {
        /**
         * @brief The set of element types a dispatch helper is allowed to instantiate its
         * kernel for.
         *
         * Not a runtime filter: it selects, at compile time, which `case` arms of the
         * dtype switch actually instantiate the caller's functor. That distinction matters
         * because the functor is a generic lambda whose body is only well-formed for some
         * element types - `sycl::max(x, y)` and `sycl::floor(x)`, for instance, reject
         * `std::complex`, and a generic lambda gives no SFINAE-visible signal of that
         * (return-type deduction instantiates the body, which is not the immediate
         * context). So the caller states the domain, the dispatch instantiates only the
         * arms inside it, and every arm outside it throws naming the op and the dtype.
         *
         * **A bit set over element-type families, not a list of named combinations.** It
         * began as the two-valued `{REAL, REAL_AND_COMPLEX}` this codebase still spells
         * everywhere, and the arrival of `INT32`/`INT64` would have turned that into six
         * enumerators whose relationships only the reader can see. As flags, a new family
         * costs one bit and the query stays @ref domain_admits; the two original names
         * survive as the combinations they always were, so no existing call site changes.
         *
         * @warning REAL is the default deliberately: a functor written for real scalars
         *          always compiles under it. Widen it only when the functor is genuinely
         *          well-formed for the family being added - see the `safe_*` helpers in
         *          <SushiBLAS/engine/math/safe_math.hpp> for the complex case, and note
         *          that an integer functor must not use `sycl::` floating-point builtins.
         */
        enum class Domain : uint8_t
        {
            REAL              = 1u << 0,  /**< HALF, FLOAT32, FLOAT64. */
            COMPLEX           = 1u << 1,  /**< COMPLEX32, COMPLEX64. */
            INTEGRAL          = 1u << 2,  /**< INT32, INT64. */

            REAL_AND_COMPLEX  = REAL | COMPLEX,   /**< Everything but the integers. */
            REAL_AND_INTEGRAL = REAL | INTEGRAL,  /**< Everything an ordering is defined on. */
            ALL               = REAL | COMPLEX | INTEGRAL /**< Every element type. */
        };

        /**
         * @brief Whether @p allowed includes the element-type family @p family.
         *
         * The one query every dispatch arm asks, always in an `if constexpr`, so an arm
         * outside the domain is never instantiated - only compiled as a throw.
         *
         * @param allowed The domain the caller declared.
         * @param family A single-bit @ref Domain naming one family.
         * @return True when @p allowed contains @p family.
         */
        [[nodiscard]] inline constexpr bool domain_admits(Domain allowed, Domain family) noexcept
        {
            return (static_cast<uint8_t>(allowed) & static_cast<uint8_t>(family)) != 0;
        }

        /**
         * @brief The element-type family a runtime dtype belongs to.
         *
         * The runtime companion to @ref domain_admits, for the few places that must
         * reject a whole family before the switch rather than inside it - a shape or
         * dtype-agreement check that runs at record time, where the compile-time domain
         * is known but the arm has not been chosen yet.
         *
         * @param dtype The data type to classify.
         * @return REAL, COMPLEX or INTEGRAL - never a combination.
         */
        [[nodiscard]] inline constexpr Domain domain_of(Core::DataType dtype) noexcept
        {
            if (Core::is_complex_dtype(dtype)) return Domain::COMPLEX;
            if (Core::is_integer_dtype(dtype)) return Domain::INTEGRAL;
            return Domain::REAL;
        }

        /**
         * @brief Throws unless @p dtype is one @p Dom admits.
         *
         * @ref with_element_type would reject it too, but not until the task runs, which is
         * a worse place to find out: by then the graph is recorded, the caller is inside
         * `Engine::execute()`, and the failure is attributed to a submission rather than to
         * the line that queued it. This is the same question asked at record time, where a
         * caller can still act on the answer.
         *
         * @tparam Dom Which element-type families the operation admits.
         * @param dtype The operand dtype to check.
         * @param family What kind of operation this is ("Elementwise operation", ...).
         * @param name The op's own name.
         * @throws std::runtime_error naming @p family, @p name and @p dtype when the dtype
         *         falls outside @p Dom.
         */
        template<Domain Dom>
        inline void require_domain(Core::DataType dtype, const char* family, const char* name)
        {
            SB_THROW_IF(!domain_admits(Dom, domain_of(dtype)),
                        "{} '{}' does not support dtype {}. Complex operands need an ordering "
                        "or a real-only builtin they do not provide; integer operands are "
                        "accepted only where an integer element means something (copy, fill, "
                        "the comparison and logic family, and the index reductions).",
                        family, name, Core::to_string(dtype));
        }

        /**
         * @brief Runs @p arm instantiated for the C++ type @p dtype denotes, or throws.
         *
         * The dtype switch, once. Every per-element executor in this library used to
         * write its own - seven `case` arms each, four of them differing only in a type
         * name, two of them wrapped in an `if constexpr` domain guard, and one `default`
         * carrying the failure message. Eight copies of that is eight places to forget an
         * arm when a `DataType` is added, and eight slightly different wordings when one
         * is rejected. This is the whole thing in one function: the caller supplies a
         * templated lambda and says nothing about dtypes at all.
         *
         * An arm outside @p Dom is never instantiated - the `if constexpr` sees to that,
         * which is what lets a functor that is only well-formed for real scalars be
         * dispatched on a tensor whose dtype *could* have been complex.
         *
         * @code
         * Dispatch::with_element_type<Dom>(dtype, "Elementwise operation", name,
         *     [&]<typename T>() { Launch::map(h, q, n, ok, op, (T*)out, (const T*)in); });
         * @endcode
         *
         * @tparam Dom Which element-type families to instantiate @p arm for.
         * @tparam Arm The templated-lambda type, deduced.
         * @param dtype The runtime element type to dispatch on.
         * @param family What kind of operation this is, for the failure message
         *        ("Elementwise operation", "Logic operation", ...).
         * @param name The op's own name, for the failure message.
         * @param arm Callable as `arm.template operator()<T>()` for every element type in
         *        @p Dom.
         * @throws std::runtime_error naming @p family, @p name and @p dtype when the dtype
         *         falls outside @p Dom - or is not an element type at all.
         */
        namespace Detail
        {
            /**
             * @brief Runs @p arm for element type @p T if @p Dom admits @p Family.
             *
             * Out of line from the switch in @ref with_element_type, and not inlined back
             * into it, because an `if constexpr` written directly inside a `switch` in a
             * template makes this project's clang lose the switch: instantiating it reports
             * "'case' statement not in switch statement" for every label after the first.
             * Hoisting the compile-time test into its own function leaves the switch a
             * plain switch of plain calls, which it parses correctly.
             *
             * @tparam Dom The domain the caller declared.
             * @tparam Family The single family this arm belongs to.
             * @tparam T The element type to instantiate @p arm for.
             * @tparam Arm The templated-lambda type, deduced.
             * @param arm The caller's arm.
             * @return True when the arm ran; false when @p Family is outside @p Dom.
             */
            template<Domain Dom, Domain Family, typename T, typename Arm>
            [[nodiscard]] inline bool run_arm_if_admitted(Arm& arm)
            {
                if constexpr (domain_admits(Dom, Family))
                {
                    arm.template operator()<T>();
                    return true;
                }
                else
                {
                    return false;
                }
            }
        } // namespace Detail

        template<Domain Dom, typename Arm>
        inline void with_element_type(Core::DataType dtype, const char* family,
                                      const char* name, Arm&& arm)
        {
            bool served = false;

            switch (dtype)
            {
                case Core::DataType::HALF:
                    served = Detail::run_arm_if_admitted<Dom, Domain::REAL, sycl::half>(arm);
                    break;
                case Core::DataType::FLOAT32:
                    served = Detail::run_arm_if_admitted<Dom, Domain::REAL, float>(arm);
                    break;
                case Core::DataType::FLOAT64:
                    served = Detail::run_arm_if_admitted<Dom, Domain::REAL, double>(arm);
                    break;
                case Core::DataType::COMPLEX32:
                    served = Detail::run_arm_if_admitted<Dom, Domain::COMPLEX, std::complex<float>>(arm);
                    break;
                case Core::DataType::COMPLEX64:
                    served = Detail::run_arm_if_admitted<Dom, Domain::COMPLEX, std::complex<double>>(arm);
                    break;
                case Core::DataType::INT32:
                    served = Detail::run_arm_if_admitted<Dom, Domain::INTEGRAL, int32_t>(arm);
                    break;
                case Core::DataType::INT64:
                    served = Detail::run_arm_if_admitted<Dom, Domain::INTEGRAL, int64_t>(arm);
                    break;
            }

            // Not served means a dtype outside Dom, or a DataType holding a value outside
            // the enumeration at all. Both are "this op cannot do that", and both say so by
            // name - a silent no-op here is how seven operations came to do nothing at all.
            SB_THROW_IF(!served, "{} '{}' does not support dtype {}.",
                        family, name, Core::to_string(dtype));
        }

        /**
         * @brief Copies a caller-supplied parameter list into a task's inline metadata slots.
         *
         * `TaskMetadata::set_param` silently ignores an out-of-range index, so a caller that
         * writes past `K_MAX_PARAMS` loses the value with no diagnostic at all. This wrapper
         * turns that into a loud failure at record time (P1: fail loudly, never silently).
         *
         * @tparam T The parameter element type; must fit in 64 bits.
         * @param meta The metadata block to populate, starting at slot 0.
         * @param params The values to store.
         * @param name The op name, used only in the failure message.
         * @param reserved_slots Number of trailing slots the caller keeps for its own use
         *        (for example the RNG seed/offset pair) and which `params` must not reach.
         */
        template<typename T>
        inline void set_task_params(SushiRuntime::Graph::TaskMetadata& meta, const std::vector<T>& params,
                                     const char* name, size_t reserved_slots = 0)
        {
            constexpr size_t k_capacity = SushiRuntime::Graph::TaskMetadata::K_MAX_PARAMS;
            SB_THROW_IF(params.size() + reserved_slots > k_capacity,
                        "Operation '{}' passes {} task parameters (+{} reserved) but TaskMetadata holds only {}; "
                        "the excess would be silently dropped.",
                        name, params.size(), reserved_slots, k_capacity);

            for (size_t i = 0; i < params.size(); ++i)
            {
                meta.set_param(static_cast<int>(i), params[i]);
            }
        }

    } // namespace Dispatch
} // namespace SushiBLAS
