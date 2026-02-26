/**************************************************************************/
/* he_normal.cpp                                                          */
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
#include <SushiBLAS/ops/math/random.hpp>
#include <SushiRuntime/graph/task_types.hpp>
#include "random_internal.hpp"

namespace SushiBLAS 
{
    using namespace SushiRuntime::Graph::Literals;

    sycl::event RandomOps::he_normal(Tensor& t, int64_t n_in) 
    {
        const double stddev = std::sqrt(2.0 / n_in);

        if (t.dtype == Core::DataType::FLOAT32)
        {
            return Internal::add_rng_task<float>(engine_, t, "he_normal", "random.he_normal"_op, oneapi::mkl::rng::gaussian<float>(0.0f, static_cast<float>(stddev)));
        }
        else if (t.dtype == Core::DataType::FLOAT64)
        {
            return Internal::add_rng_task<double>(engine_, t, "he_normal", "random.he_normal"_op, oneapi::mkl::rng::gaussian<double>(0.0, stddev));
        }

        return normal(t, 0.0, stddev);
    }
}
