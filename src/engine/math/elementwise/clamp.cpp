/****************************************************************/
/* clamp.cpp                                                    */
/* SushiBLAS - https://github.com/SushiSystems/SushiBLAS        */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

/**
 * @file clamp.cpp
 * @brief Defines ElementwiseOps::clamp, which records the in-place elementwise clamp task.
 * @author Mustafa Garip
 */

#include <SushiBLAS/engine/math/elementwise.hpp>
#include <SushiBLAS/engine/math/elementwise/dispatch.hpp>

namespace SushiBLAS 
{
    using namespace SushiRuntime::Graph::Literals;

    sycl::event ElementwiseOps::clamp(Tensor& t, float min_val, float max_val) 
    {
        return Dispatch::execute_unary_inplace(engine_, t, "math.ew.clamp", "math.ew.clamp"_op, 
            [min_val, max_val](auto x) { return sycl::clamp(x, (decltype(x))min_val, (decltype(x))max_val); },
            {min_val, max_val});
    }
} // namespace SushiBLAS
