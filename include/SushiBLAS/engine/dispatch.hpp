/****************************************************************/
/* dispatch.hpp                                                 */
/* SushiBLAS - https://github.com/SushiSystems/SushiBLAS        */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

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
#include <sycl/sycl.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/core/logger.hpp>
#include <SushiBLAS/engine/device_policy.hpp>
#include <SushiRuntime/graph/task_types.hpp>

namespace SushiBLAS
{
    namespace Dispatch
    {
        /**
         * @brief Enumerates the element-type families a dispatch helper may instantiate its kernel for.
         * @warning Widen it from REAL only when the functor is well-formed for the family being added.
         * @see include/SushiBLAS/engine/README.md
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
         * @brief Returns whether allowed includes the element-type family.
         * @param family A single-bit Domain naming one family.
         * @see include/SushiBLAS/engine/README.md
         */
        [[nodiscard]] inline constexpr bool domain_admits(Domain allowed, Domain family) noexcept
        {
            return (static_cast<uint8_t>(allowed) & static_cast<uint8_t>(family)) != 0;
        }

        /**
         * @brief Returns the element-type family a runtime dtype belongs to.
         * @return REAL, COMPLEX or INTEGRAL, never a combination.
         * @see include/SushiBLAS/engine/README.md
         */
        [[nodiscard]] inline constexpr Domain domain_of(Core::DataType dtype) noexcept
        {
            if (Core::is_complex_dtype(dtype)) return Domain::COMPLEX;
            if (Core::is_integer_dtype(dtype)) return Domain::INTEGRAL;
            return Domain::REAL;
        }

        /**
         * @brief Throws unless dtype is one Dom admits.
         * @throws std::runtime_error naming family, name and dtype when the dtype falls outside Dom.
         * @see include/SushiBLAS/engine/README.md
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
         * @brief Carries one element type to a with_element_type arm as a value.
         * @see include/SushiBLAS/engine/README.md
         */
        template<typename T>
        struct ElementTag
        {
            /** @brief The element type this tag denotes. */
            using type = T;
        };

        /**
         * @brief Holds the helpers with_element_type uses to detect and invoke an arm.
         * @see include/SushiBLAS/engine/README.md
         */
        namespace Detail
        {
            /**
             * @brief Reports whether Arm accepts the tag form, `arm(ElementTag<T>{})`.
             * @pre T is an element type the caller's domain admits.
             * @see include/SushiBLAS/engine/README.md
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
                    // The static_assert states the arm contract in a sentence; see include/SushiBLAS/engine/README.md.
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
             * @see include/SushiBLAS/engine/README.md
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
         * @see include/SushiBLAS/engine/README.md
         */
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

            // An unserved dtype throws by name; it is never a silent no-op; see include/SushiBLAS/engine/README.md.
            SB_THROW_IF(!served, "{} '{}' does not support dtype {}.",
                        family, name, Core::to_string(dtype));
        }

        /**
         * @brief Copies a parameter list into a task's inline metadata slots, starting at slot 0.
         * @tparam T The parameter element type; must fit in 64 bits.
         * @param reserved_slots Trailing slots the caller keeps for itself, which params must not reach.
         * @see include/SushiBLAS/engine/README.md
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
