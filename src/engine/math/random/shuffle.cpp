/****************************************************************/
/* shuffle.cpp                                                  */
/* SushiBLAS - https://github.com/SushiSystems/SushiBLAS        */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

/**
 * @file shuffle.cpp
 * @brief Defines RandomOps::shuffle, which records the in-place shuffle of a tensor's elements.
 * @author Mustafa Garip
 */

#include <vector>
#include <complex>
#include <sycl/sycl.hpp>
#include <SushiBLAS/engine.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/core/logger.hpp>
#include <SushiBLAS/engine/math/random.hpp>
#include <SushiBLAS/engine/math/random/dispatch.hpp>
#include <SushiRuntime/core/resource_region.hpp>
#include <SushiRuntime/graph/task_types.hpp>
#include <kernels/random.hpp>

namespace SushiBLAS
{
    using namespace SushiRuntime::Graph::Literals;

    namespace
    {
        // Fisher-Yates runs in one sequential task, one instantiation per dtype; see src/README.md.
        template<typename T>
        sycl::event shuffle_dispatch(sycl::queue& q, T* p, int64_t size, uint64_t seed, uint64_t offset,
                                      const std::vector<sycl::event>& deps)
        {
            return q.submit([&](sycl::handler& h)
            {
                h.depends_on(deps);
                h.single_task([=]()
                {
                    for (int64_t i = size - 1; i > 0; --i)
                    {
                        int64_t j = static_cast<int64_t>(Kernels::SYCL::Random::draw_index(
                            seed, offset, static_cast<uint64_t>(i), static_cast<uint64_t>(i) + 1));

                        T tmp = p[i];
                        p[i] = p[j];
                        p[j] = tmp;
                    }
                });
            });
        }
    }

    sycl::event RandomOps::shuffle(Tensor& t)
    {
        const char* name = "random.shuffle";

        const int64_t size = t.num_elements;
        void* ptr = t.data_or_null();

        // A shuffle reads what it writes, so the region is read-write; see src/README.md.
        std::vector<SushiRuntime::Core::ResourceRegion> rw = {};
        if (ptr) rw.push_back(t.region());

        SushiRuntime::Graph::TaskMetadata meta;
        meta.name = name;
        meta.task_type = SushiRuntime::Graph::TaskType::MATH_OP;
        meta.op_id = "random.shuffle"_op;

        const uint64_t seed = engine_.get_seed();
        const uint64_t offset = engine_.get_and_increment_rng_offset();
        Dispatch::set_rng_params(meta, seed, offset);

        engine_.get_graph().add_task(meta, rw, rw,
            [dtype = t.dtype, size, seed, offset, ptr, name](sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event
            {
                SB_LOG_INFO("RandomOps: {} ({} elements, seed: {}, offset: {})", name, size, seed, offset);

                switch (dtype)
                {
                    case Core::DataType::HALF:
                        return shuffle_dispatch(q, static_cast<sycl::half*>(ptr), size, seed, offset, deps);
                    case Core::DataType::FLOAT32:
                        return shuffle_dispatch(q, static_cast<float*>(ptr), size, seed, offset, deps);
                    case Core::DataType::FLOAT64:
                        return shuffle_dispatch(q, static_cast<double*>(ptr), size, seed, offset, deps);
                    case Core::DataType::COMPLEX32:
                        return shuffle_dispatch(q, static_cast<std::complex<float>*>(ptr), size, seed, offset, deps);
                    case Core::DataType::COMPLEX64:
                        return shuffle_dispatch(q, static_cast<std::complex<double>*>(ptr), size, seed, offset, deps);
                    default:
                        SB_THROW_IF(true, "Operation '{}' received an unrecognized dtype ({}).",
                                    name, static_cast<int>(dtype));
                        return sycl::event();
                }
            }
        );

        return sycl::event();
    }
} // namespace SushiBLAS
