/****************************************************************/
/* convert.cpp                                                  */
/* SushiBLAS - https://github.com/SushiSystems/SushiBLAS        */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

// The dtype-converting copy. See the `@brief` on ElementwiseOps::convert in
// <SushiBLAS/engine/math/elementwise.hpp> for why it lives in this family, why
// the dtype pair is bounded to REAL x REAL, and why a same-dtype conversion is
// legal rather than an error. What is recorded here is the mechanics.
//
// **No public `Dispatch::execute_convert` seam, deliberately.** Every other
// executor in `SushiBLAS::Dispatch` is public because it takes a caller-written
// *functor*: that is what lets a consumer fuse several math ops into one kernel
// without writing SYCL, and it is the entire reason those headers were promoted
// out of src/. This op has no functor - the conversion is `static_cast`, fixed by
// the pair of dtypes - so a seam would expose a parameterisation that does not
// exist. Keeping the implementation in this translation unit also physically
// confines the nine instantiations to one object file, which is the bound the
// header argues for rather than merely asserts.

#include <cstddef>
#include <cstdint>
#include <vector>
#include <sycl/sycl.hpp>
#include <SushiBLAS/engine.hpp>
#include <SushiBLAS/tensor.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/core/logger.hpp>
#include <SushiBLAS/engine/launch_1d.hpp>
#include <SushiBLAS/engine/math/elementwise.hpp>
#include <SushiRuntime/core/resource_region.hpp>
#include <SushiRuntime/graph/task_types.hpp>

namespace SushiBLAS
{
    using namespace SushiRuntime::Graph::Literals;

    namespace
    {
        /** @brief The op's name, used for the task, the log line and every message. */
        constexpr const char* K_NAME = "math.ew.convert";

        /**
         * @brief Records the task for one concrete (source, destination) type pair.
         *
         * One of the nine arms. An ordinary function template rather than a lambda,
         * because the two-level dispatch below has to be two plain `switch` statements:
         * a generic lambda nested inside a generic lambda is the shape that crashes this
         * project's clang during instantiation (see the note at
         * <SushiBLAS/engine/blas/utils.hpp>), and a converting op is exactly where that
         * nesting would otherwise be reached for.
         *
         * @tparam Src The source element type.
         * @tparam Dst The destination element type.
         * @param engine The engine whose task graph the op is recorded on.
         * @param meta The task metadata.
         * @param reads The read set - `src.region()`, never a bare pointer.
         * @param writes The write set - `dst.region()`, never a bare pointer.
         * @param src_ptr Address of the source's first element.
         * @param dst_ptr Address of the destination's first element.
         * @param n How many elements to convert.
         */
        template<typename Src, typename Dst>
        void record_convert(Engine& engine, const SushiRuntime::Graph::TaskMetadata& meta,
                            const std::vector<SushiRuntime::Core::ResourceRegion>& reads,
                            const std::vector<SushiRuntime::Core::ResourceRegion>& writes,
                            const void* src_ptr, void* dst_ptr, int64_t n)
        {
            engine.get_graph().add_task(meta, reads, writes,
                [n, pS = static_cast<const Src*>(src_ptr), pD = static_cast<Dst*>(dst_ptr)]
                (sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event
                {
                    SB_LOG_INFO("Elementwise convert: {} elements", n);

                    return q.submit([&](sycl::handler& h)
                    {
                        h.depends_on(deps);

                        // `Launch::indexed`, not `Launch::map`. `map` static_asserts that
                        // every operand shares one element type, and it is right to: its
                        // wide pass loads and stores `sycl::vec<T, W>` at one width, and
                        // two operands of different widths do not have one W. So this op
                        // gets the device split - flat on a CPU, grid-stride on a GPU -
                        // and no vector path, which is the honest shape rather than a
                        // fast one that would be reading half of every second element.
                        Launch::indexed(h, q, static_cast<std::size_t>(n),
                            [pS, pD](std::size_t i)
                            {
                                // The conversion itself. `static_cast` is IEEE-754
                                // round-to-nearest-even for every pair here, and it is
                                // what makes an fp32 value past 65504 arrive as an fp16
                                // infinity rather than as a wrapped or clamped number -
                                // the behaviour a loss-scaling backoff is built to detect.
                                //
                                // One honest caveat, for FLOAT64 -> HALF only: `sycl::half`
                                // has no direct double constructor on every toolchain, so
                                // that pair may round twice (double -> float -> half), and
                                // a double-rounding differs from a single correctly-rounded
                                // step for the handful of values that land exactly halfway
                                // after the first step. Not measured, not relied on, and
                                // not on the mixed-precision path, which is fp32 <-> fp16.
                                pD[i] = static_cast<Dst>(pS[i]);
                            });
                    });
                });
        }

        /**
         * @brief The inner half of the dtype switch: source type fixed, destination
         * chosen at runtime.
         *
         * Three arms, instantiated once per source type, for nine in total. Written out
         * rather than generated so the instantiation count is a thing a reader can
         * count.
         *
         * @tparam Src The source element type, already resolved.
         * @param dst_dtype The destination tensor's runtime dtype.
         * @param engine The engine whose task graph the op is recorded on.
         * @param meta The task metadata.
         * @param reads The read set.
         * @param writes The write set.
         * @param src_ptr Address of the source's first element.
         * @param dst_ptr Address of the destination's first element.
         * @param n How many elements to convert.
         * @throws std::runtime_error if @p dst_dtype is outside `Domain::REAL`.
         */
        template<typename Src>
        void dispatch_destination(Core::DataType dst_dtype, Engine& engine,
                                  const SushiRuntime::Graph::TaskMetadata& meta,
                                  const std::vector<SushiRuntime::Core::ResourceRegion>& reads,
                                  const std::vector<SushiRuntime::Core::ResourceRegion>& writes,
                                  const void* src_ptr, void* dst_ptr, int64_t n)
        {
            switch (dst_dtype)
            {
                case Core::DataType::HALF:
                    record_convert<Src, sycl::half>(engine, meta, reads, writes, src_ptr, dst_ptr, n);
                    return;
                case Core::DataType::FLOAT32:
                    record_convert<Src, float>(engine, meta, reads, writes, src_ptr, dst_ptr, n);
                    return;
                case Core::DataType::FLOAT64:
                    record_convert<Src, double>(engine, meta, reads, writes, src_ptr, dst_ptr, n);
                    return;
                default:
                    break;
            }

            // Unreachable: the caller rejected every non-REAL dtype at record time. It
            // ends in a throw all the same, because a switch that can fall out of the
            // bottom is how a silent no-op gets in - the same reasoning
            // `Internal::execute_blas_task` gives for its own trailing throw.
            SB_THROW_IF(true, "Elementwise operation '{}' does not support destination dtype {}.",
                        K_NAME, Core::to_string(dst_dtype));
        }
    } // namespace

    sycl::event ElementwiseOps::convert(const Tensor& src, Tensor& dst)
    {
        // Both dtypes, named together, because "which one was wrong" is the first thing
        // a caller wants and the pair is what was actually rejected.
        SB_THROW_IF(!Core::is_real_dtype(src.dtype) || !Core::is_real_dtype(dst.dtype),
                    "Elementwise operation '{}' converts between the real floating-point "
                    "types only (HALF, FLOAT32, FLOAT64), and was asked for {} -> {}. A "
                    "complex operand has no single real value to narrow to, and a "
                    "float-to-integer conversion is a rounding policy (truncate? "
                    "round-half-even? saturate?) this layer has no basis to choose for "
                    "you - an op that does it belongs under a name that states which.",
                    K_NAME, Core::to_string(src.dtype), Core::to_string(dst.dtype));

        SB_THROW_IF(src.rank != dst.rank,
                    "Elementwise operation '{}' copies element for element, so its operands "
                    "must have the same shape: source is rank {}, destination is rank {}.",
                    K_NAME, src.rank, dst.rank);

        for (int32_t i = 0; i < src.rank; ++i)
        {
            SB_THROW_IF(src.shape[i] != dst.shape[i],
                        "Elementwise operation '{}': source and destination disagree on "
                        "dimension {} - source has {}, destination has {}. Reinterpreting "
                        "the element count is what Tensor::reshape is for, and saying so "
                        "explicitly is better than having a conversion do it invisibly.",
                        K_NAME, i, src.shape[i], dst.shape[i]);
        }

        // Redundant given the shape check above, and kept because it is the invariant the
        // kernel's loop bound actually rests on. If shape equality ever stops implying it,
        // this is the assertion that fires instead of the kernel running off an operand.
        SB_THROW_IF(src.num_elements != dst.num_elements,
                    "Elementwise operation '{}': source holds {} elements, destination holds {}.",
                    K_NAME, src.num_elements, dst.num_elements);

        SB_THROW_IF(!src.storage, "Elementwise operation '{}': the source tensor has no storage.", K_NAME);
        SB_THROW_IF(!dst.storage, "Elementwise operation '{}': the destination tensor has no storage.", K_NAME);

        // The kernel walks a flat index, so a view whose elements are not laid out flatly
        // would be read at the wrong addresses. That is a wrong answer with nothing to
        // report it, so it is refused by name instead.
        SB_THROW_IF(!src.is_contiguous(),
                    "Elementwise operation '{}': the source tensor is not contiguous. This op "
                    "walks a flat index; an inner-axis slice does not have its elements there.",
                    K_NAME);
        SB_THROW_IF(!dst.is_contiguous(),
                    "Elementwise operation '{}': the destination tensor is not contiguous. This "
                    "op walks a flat index; an inner-axis slice does not have its elements there.",
                    K_NAME);

        // Each work-item reads one element and writes another of a *different* width, so
        // overlapping operands are a race between work-items rather than the well-defined
        // same-index aliasing `Launch::map`'s in-place callers rely on. Refused for the
        // same-dtype case too: the widths agreeing does not make a self-convert useful,
        // and a rule with an exception in it is one a caller has to remember.
        SB_THROW_IF(src.region().overlaps(dst.region()),
                    "Elementwise operation '{}': the source and destination overlap in memory. "
                    "A converting copy reads {}-byte elements and writes {}-byte ones, so an "
                    "overlap is a race between work-items, not an in-place update.",
                    K_NAME, Core::bytes_per_element(src.dtype), Core::bytes_per_element(dst.dtype));

        const void* read_src = src.data();
        void* write_dst = dst.data();
        const int64_t n = src.num_elements;

        // `Tensor::region()`, never `data()`. A ResourceRegion converts implicitly from a
        // bare pointer and a bare pointer widens to the whole allocation, so two views of
        // one arena would become two unrelated resources, `overlaps` would answer false,
        // and the tracker would emit NO edge - see the header, and the two training runs
        // this has already cost the stack.
        const std::vector<SushiRuntime::Core::ResourceRegion> reads = {src.region()};
        const std::vector<SushiRuntime::Core::ResourceRegion> writes = {dst.region()};

        SushiRuntime::Graph::TaskMetadata meta;
        meta.name = K_NAME;
        meta.task_type = SushiRuntime::Graph::TaskType::MATH_OP;
        meta.op_id = "math.ew.convert"_op;

        // The outer half of the switch. Two plain switches over function templates, nine
        // instantiations, no lambda nested in a lambda.
        switch (src.dtype)
        {
            case Core::DataType::HALF:
                dispatch_destination<sycl::half>(dst.dtype, engine_, meta, reads, writes, read_src, write_dst, n);
                return sycl::event();
            case Core::DataType::FLOAT32:
                dispatch_destination<float>(dst.dtype, engine_, meta, reads, writes, read_src, write_dst, n);
                return sycl::event();
            case Core::DataType::FLOAT64:
                dispatch_destination<double>(dst.dtype, engine_, meta, reads, writes, read_src, write_dst, n);
                return sycl::event();
            default:
                break;
        }

        SB_THROW_IF(true, "Elementwise operation '{}' does not support source dtype {}.",
                    K_NAME, Core::to_string(src.dtype));
        return sycl::event();
    }
} // namespace SushiBLAS
