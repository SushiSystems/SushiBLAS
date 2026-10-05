/****************************************************************/
/* constant.cpp                                                 */
/* SushiBLAS - https://github.com/SushiSystems/SushiBLAS        */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

#include <complex>
#include <sycl/sycl.hpp>
#include <SushiBLAS/engine.hpp>
#include <SushiBLAS/core/logger.hpp>
#include <SushiBLAS/engine/math/random.hpp>
#include <SushiRuntime/core/resource_region.hpp>
#include <SushiRuntime/graph/task_types.hpp>

namespace SushiBLAS 
{
    using namespace SushiRuntime::Graph::Literals;

    sycl::event RandomOps::constant(Tensor& t, double value) 
    {
        auto op_id = "random.constant"_op;
        SushiRuntime::Graph::TaskMetadata meta;
        meta.name = "random.constant";
        meta.task_type = SushiRuntime::Graph::TaskType::MATH_OP;
        meta.op_id = op_id;
        meta.set_param(0, value);

        const int64_t size = t.num_elements;
        void* ptr = t.data_or_null();
        std::vector<SushiRuntime::Core::ResourceRegion> reads = {};
        std::vector<SushiRuntime::Core::ResourceRegion> writes = {};
        if (ptr) writes.push_back(t.region());

        switch (t.dtype)
        {
            case Core::DataType::HALF:
            {
                engine_.get_graph().add_task(meta, reads, writes,
                    [size, value, pT = t.data_as<sycl::half>()](sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event 
                    {
                        SB_LOG_INFO("RandomOps: constant ({} elements, value: {:.4f})", size, value);
                        return q.fill(pT, static_cast<sycl::half>(value), size, deps);
                    }
                );
                break;
            }
            case Core::DataType::FLOAT32:
            {
                engine_.get_graph().add_task(meta, reads, writes,
                    [size, value, pT = t.data_as<float>()](sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event 
                    {
                        SB_LOG_INFO("RandomOps: constant ({} elements, value: {:.4f})", size, value);
                        return q.fill(pT, static_cast<float>(value), size, deps);
                    }
                );
                break;
            }
            case Core::DataType::FLOAT64:
            {
                engine_.get_graph().add_task(meta, reads, writes,
                    [size, value, pT = t.data_as<double>()](sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event 
                    {
                        SB_LOG_INFO("RandomOps: constant ({} elements, value: {:.4f})", size, value);
                        return q.fill(pT, value, size, deps);
                    }
                );
                break;
            }
            case Core::DataType::COMPLEX32:
            {
                engine_.get_graph().add_task(meta, reads, writes,
                    [size, value, pT = t.data_as<std::complex<float>>()](sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event 
                    {
                        SB_LOG_INFO("RandomOps: constant complex32 ({} elements, value: {:.4f})", size, value);
                        return q.fill(pT, std::complex<float>(static_cast<float>(value), 0.0f), size, deps);
                    }
                );
                break;
            }
            case Core::DataType::COMPLEX64:
            {
                engine_.get_graph().add_task(meta, reads, writes,
                    [size, value, pT = t.data_as<std::complex<double>>()](sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event 
                    {
                        SB_LOG_INFO("RandomOps: constant complex64 ({} elements, value: {:.4f})", size, value);
                        return q.fill(pT, std::complex<double>(value, 0.0), size, deps);
                    }
                );
                break;
            }
            // The integer arms are not an oddity here: `constant` is how an index tensor
            // gets a known value before an op writes over it, and how a comparison operand
            // gets a threshold, and both of those live in an integer tensor now. Truncating
            // the double parameter is the same convention the HALF and FLOAT32 arms use -
            // the caller states a value, the tensor's dtype decides how it is stored.
            case Core::DataType::INT32:
            {
                engine_.get_graph().add_task(meta, reads, writes,
                    [size, value, pT = t.data_as<int32_t>()](sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event
                    {
                        SB_LOG_INFO("RandomOps: constant int32 ({} elements, value: {:.4f})", size, value);
                        return q.fill(pT, static_cast<int32_t>(value), size, deps);
                    }
                );
                break;
            }
            case Core::DataType::INT64:
            {
                engine_.get_graph().add_task(meta, reads, writes,
                    [size, value, pT = t.data_as<int64_t>()](sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event
                    {
                        SB_LOG_INFO("RandomOps: constant int64 ({} elements, value: {:.4f})", size, value);
                        return q.fill(pT, static_cast<int64_t>(value), size, deps);
                    }
                );
                break;
            }
            default:
                SB_THROW_IF(true, "Operation 'random.constant' does not support dtype {}.",
                            Core::to_string(t.dtype));
        }

        return sycl::event();
    }
} // namespace SushiBLAS
