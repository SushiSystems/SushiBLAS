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
 * @file dispatch.hpp
 * @brief Declares what every op-dispatch family shares: the element domain and the dtype switch.
 * @author Mustafa Garip
 */

#pragma once

#include <vector>
#include <cstddef>
#include <cstdint>
#include <complex>
#include <utility>
#include <type_traits>
#include <initializer_list>
#include <sycl/sycl.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/core/logger.hpp>
#include <SushiBLAS/support/device_policy.hpp>
#include <SushiBLAS/engine/operand_check.hpp>
#include <SushiBLAS/engine/task_description.hpp>
#include <SushiRuntime/core/operation_id.hpp>

namespace SushiBLAS
{
    namespace Dispatch
    {
        /**
         * @brief Enumerates the element-type families a dispatch helper may instantiate its kernel for.
         * @warning Widen it from REAL only when the functor is well-formed for the family being added.
         * @see modules/engine/engine/README.md
         */
        enum class Domain : uint8_t
        {
            Real              = 1u << 0,  /**< HALF, FLOAT32, FLOAT64. */
            Complex           = 1u << 1,  /**< COMPLEX32, COMPLEX64. */
            Integral          = 1u << 2,  /**< INT32, INT64. */

            RealAndComplex  = Real | Complex,   /**< Everything but the integers. */
            RealAndIntegral = Real | Integral,  /**< Everything an ordering is defined on. */
            All               = Real | Complex | Integral /**< Every element type. */
        };

        /**
         * @brief Returns whether allowed includes the element-type family.
         * @param family A single-bit Domain naming one family.
         * @see modules/engine/engine/README.md
         */
        [[nodiscard]] inline constexpr bool domain_admits(Domain allowed, Domain family) noexcept
        {
            return (static_cast<uint8_t>(allowed) & static_cast<uint8_t>(family)) != 0;
        }

        /**
         * @brief Returns the element-type family a runtime dtype belongs to.
         * @return REAL, COMPLEX or INTEGRAL, never a combination.
         * @see modules/engine/engine/README.md
         */
        [[nodiscard]] inline constexpr Domain domain_of(Core::DataType dtype) noexcept
        {
            if (Core::is_complex_dtype(dtype))
                return Domain::Complex;
            if (Core::is_integer_dtype(dtype))
                return Domain::Integral;
            return Domain::Real;
        }

        /**
         * @brief Throws unless dtype is one Dom admits.
         * @throws std::invalid_argument naming family, name and dtype when the dtype falls outside Dom.
         * @see modules/engine/engine/README.md
         */
        template<Domain Dom>
        inline void require_domain(Core::DataType dtype, const char* family, const char* name)
        {
            if (!domain_admits(Dom, domain_of(dtype)))
                Detail::throw_dtype_error(family, name, dtype);
        }

        /**
         * @brief Carries one element type to a with_element_type arm as a value.
         * @see modules/engine/engine/README.md
         */
        template<typename T>
        struct ElementTag
        {
            /** @brief The element type this tag denotes. */
            using type = T;
        };

        /**
         * @brief Holds the helpers with_element_type uses to detect and invoke an arm.
         * @see modules/engine/engine/README.md
         */
        namespace Detail
        {
            /**
             * @brief Reports whether Arm accepts the tag form, `arm(ElementTag<T>{})`.
             * @pre T is an element type the caller's domain admits.
             * @see modules/engine/engine/README.md
             */
            template<typename Arm, typename T, typename = void>
            struct arm_takes_tag : std::false_type {};

            template<typename Arm, typename T>
            struct arm_takes_tag<Arm, T,
                                 std::void_t<decltype(std::declval<Arm&>()(ElementTag<T>{}))>>
                : std::true_type {};

            /**
             * @brief Whether @p Arm accepts the explicit form, `arm.template operator()<T>()`.
             * @tparam Arm The arm's type.
             * @tparam T The element type.
             */
            template<typename Arm, typename T, typename = void>
            struct arm_takes_explicit_type : std::false_type {};

            template<typename Arm, typename T>
            struct arm_takes_explicit_type<Arm, T,
                                           std::void_t<decltype(std::declval<Arm&>().template operator()<T>())>>
                : std::true_type {};

            /**
             * @brief Calls @p arm for element type @p T, whichever form it is written in.
             *
             * @tparam T The element type to instantiate @p arm for.
             * @tparam Arm The arm's type, deduced.
             * @param arm The caller's arm.
             */
            template<typename T, typename Arm>
            inline void invoke_arm(Arm& arm)
            {
                if constexpr (arm_takes_tag<Arm, T>::value)
                {
                    arm(ElementTag<T>{});
                }
                else
                {
                    // The static_assert states the arm contract in a sentence; see modules/engine/engine/README.md.
                    static_assert(arm_takes_explicit_type<Arm, T>::value,
                                  "A with_element_type arm must be callable either as "
                                  "arm(SushiBLAS::Dispatch::ElementTag<T>{}) - a generic lambda "
                                  "'[&](auto tag) { using T = typename decltype(tag)::type; ... }' - "
                                  "or as arm.template operator()<T>() - a C++20 templated lambda "
                                  "'[&]<typename T>() { ... }'. This one is neither; check its "
                                  "parameter list.");
                    arm.template operator()<T>();
                }
            }

            /**
             * @brief Runs arm for element type T if Dom admits Family.
             * @return True when the arm ran; false when Family is outside Dom.
             * @see modules/engine/engine/README.md
             */
            template<Domain Dom, Domain Family, typename T, typename Arm>
            [[nodiscard]] inline bool run_arm_if_admitted(Arm& arm)
            {
                if constexpr (domain_admits(Dom, Family))
                {
                    invoke_arm<T>(arm);
                    return true;
                }
                else
                {
                    return false;
                }
            }
        } // namespace Detail

        /**
         * @brief Runs arm instantiated for the C++ type dtype denotes, or throws.
         * @param arm Callable as `arm(ElementTag<T>{})` or `arm.template operator()<T>()` for each type in Dom.
         * @throws std::runtime_error naming family, name and dtype when the dtype falls outside Dom.
         * @see modules/engine/engine/README.md
         */
        template<Domain Dom, typename Arm>
        inline void with_element_type(Core::DataType dtype, const char* family,
                                      const char* name, Arm&& arm)
        {
            bool served = false;

            switch (dtype)
            {
                case Core::DataType::Half:
                    served = Detail::run_arm_if_admitted<Dom, Domain::Real, sycl::half>(arm);
                    break;
                case Core::DataType::Float32:
                    served = Detail::run_arm_if_admitted<Dom, Domain::Real, float>(arm);
                    break;
                case Core::DataType::Float64:
                    served = Detail::run_arm_if_admitted<Dom, Domain::Real, double>(arm);
                    break;
                case Core::DataType::Complex32:
                    served = Detail::run_arm_if_admitted<Dom, Domain::Complex, std::complex<float>>(arm);
                    break;
                case Core::DataType::Complex64:
                    served = Detail::run_arm_if_admitted<Dom, Domain::Complex, std::complex<double>>(arm);
                    break;
                case Core::DataType::Int32:
                    served = Detail::run_arm_if_admitted<Dom, Domain::Integral, int32_t>(arm);
                    break;
                case Core::DataType::Int64:
                    served = Detail::run_arm_if_admitted<Dom, Domain::Integral, int64_t>(arm);
                    break;
            }

            // An unserved dtype throws by name; it is never a silent no-op; see modules/engine/engine/README.md.
            SB_THROW_IF(!served, "{} '{}' does not support dtype {}.",
                        family, name, Core::to_string(dtype));
        }

        /**
         * @brief Copies a parameter list into a task's inline metadata slots, starting at slot 0.
         * @tparam T The parameter element type; must fit in 64 bits.
         * @param reserved_slots Trailing slots the caller keeps for itself, which params must not reach.
         * @throws std::invalid_argument if params and the reserved slots exceed the capacity.
         * @see modules/engine/engine/README.md
         */
        template<typename T>
        inline void set_task_params(TaskDescription& description,
                                    std::initializer_list<T> params, const char* name,
                                    std::size_t reserved_slots = 0)
        {
            constexpr std::size_t k_capacity = TaskDescription::K_MAX_PARAMETERS;
            if (params.size() + reserved_slots > k_capacity)
                Detail::throw_operand_error(name, "params",
                                            "lists {} task parameters (+{} reserved) but "
                                            "TaskDescription holds only {}",
                                            params.size(), reserved_slots, k_capacity);

            int slot = 0;
            for (const T& value : params)
                description.set_parameter(slot++, value);
        }

    } // namespace Dispatch
} // namespace SushiBLAS
