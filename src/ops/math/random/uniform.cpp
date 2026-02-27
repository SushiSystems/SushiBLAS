/**************************************************************************/
/* uniform.cpp                                                            */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                SushiBLAS                               */
/*                https://github.com/SushiSystems/SushiBLAS               */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present  Mustafa Garip & Sushi Systems              */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#include <SushiBLAS/engine.hpp>
#include <SushiBLAS/core/logger.hpp>
#include <SushiBLAS/ops/math/random.hpp>
#include <SushiRuntime/graph/task_types.hpp>
#include "random_internal.hpp"

namespace SushiBLAS 
{
    using namespace SushiRuntime::Graph::Literals;

    sycl::event RandomOps::uniform(Tensor& t, double min, double max) 
    {
        return Internal::execute_random(engine_, t, "random.uniform", "random.uniform"_op, {min, max},
            [min, max](auto scalar_type, sycl::queue& q, uint64_t seed, uint64_t offset, int64_t size, auto* pT, const std::vector<sycl::event>& deps) -> sycl::event 
            {
                using T = decltype(scalar_type);
                oneapi::mkl::rng::philox4x32x10 engine_obj(q, seed);
                
                if constexpr (Internal::is_complex_v<T>) 
                {
                    using RealT = typename T::value_type;
                    oneapi::mkl::rng::skip_ahead(engine_obj, offset * size * 2);
                    return oneapi::mkl::rng::generate(
                        oneapi::mkl::rng::uniform<RealT>(static_cast<RealT>(min), static_cast<RealT>(max)), 
                        engine_obj, size * 2, reinterpret_cast<RealT*>(pT), deps);
                } 
                else 
                {
                    oneapi::mkl::rng::skip_ahead(engine_obj, offset * size);
                    return oneapi::mkl::rng::generate(
                        oneapi::mkl::rng::uniform<T>(static_cast<T>(min), static_cast<T>(max)), 
                        engine_obj, size, pT, deps);
                }
            });
    }
} // namespace SushiBLAS
