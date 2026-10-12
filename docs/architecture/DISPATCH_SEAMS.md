# 6. The dispatch seams and the fusion hooks

A seam is the function a view method calls to record its task. Seams take a `TaskRecorder&` and
return a `TaskHandle`. There are two kinds.

**The kernel-pair seam** serves the dense BLAS, sparse and spatial families.
`Internal::execute_typed_task<Dom>(recorder, description, operands, dtype, family, op_func)`
(`engine/task.hpp`) checks the dtype against the domain and records one task whose body switches
on the runtime dtype and calls `op_func(T{}, dispatcher, q, deps)` with a value of the matching
element type. `op_func` calls `dispatcher.run` with its reference kernel and its SYCL kernel.
`Internal::execute_blas_task` (`blas/blas_task.hpp`) is the same call with the family string
`"BLAS operation"`. Section 5 of [CONTRIBUTING.md](../CONTRIBUTING.md) shows a whole operation
written on it.

**The functor seams** are the public `SushiBLAS::Dispatch::execute_*` functions. Each takes a
caller-written functor, inlines it into one SYCL kernel, and runs the same functor in a serial
host loop under `ForceReference`:

| Header | Executors |
| --- | --- |
| `elementwise/dispatch.hpp` | `execute_unary_inplace`, `execute_binary`, `execute_zip_inplace` |
| `nonlinear/dispatch.hpp` | `execute_nonlinear_forward`, `execute_nonlinear_backward` |
| `logic/dispatch.hpp` | `execute_logic_unary`, `execute_logic_binary`, `execute_logic_ternary` |
| `reductions/dispatch.hpp` | `execute_reduction`, `execute_index_reduction`, `execute_counting_reduction`, `execute_folding_reduction`, `execute_folding_span_reduction`, `execute_folding_axis_reduction` |
| `random/dispatch.hpp` | `execute_random` |
| `blas/dispatch.hpp` | `execute_gemm` |

The umbrella header does not include the `dispatch.hpp` headers; a caller of a seam includes the
one it uses, as `tests/package/main.cpp` does.

A functor that reaches a device is device-copyable and trivially copyable, captures by value,
dereferences no host pointer, allocates nothing and throws nothing. The compiler does not
diagnose all of these.

**`Dispatch::Domain`** (`engine/dispatch.hpp`) is a bit set over element-type families: `Real`,
`Complex`, `Integral` and the unions `RealAndComplex`, `RealAndIntegral` and `All`. It says
which element types a functor may be instantiated for. `require_domain<Dom>` rejects a dtype
outside the domain at record time with `std::invalid_argument`. `with_element_type<Dom>` is the
dtype switch of the functor seams: an arm outside the domain is never instantiated, which is what
lets a functor that is ill-formed for `std::complex` be dispatched on a runtime dtype. An arm is
written as a generic lambda over `Dispatch::ElementTag<T>`, or, at C++20, as a templated lambda.
`execute_typed_task` spells the same seven arms out in a named class, `Detail::TypedTaskBody`,
because its body takes a value and returns an event.

## The fusion hooks

**No automatic fusion pass exists.** SushiBLAS does not inspect the graph, match a chain of
operations or merge two tasks. Two recorded tasks are two launches. Fusion is something the
caller writes: one functor that does the work of several operations, handed to one seam call.
The library supplies the places to hand it to.

*Composing a functor.* A consumer that wants `x * scale + shift` as one kernel writes it as one
functor and calls `execute_unary_inplace` once. `execute_zip_inplace` does the same over N
same-shaped tensors, each passed to the functor as an lvalue, so an optimiser step that reads and
updates five tensors is one task. All N are declared as both read and written, since the seam
cannot see which the functor touched.

*`bind_scalars`* (`functor_map/scalar_binding.hpp`) attaches scalar parameters to a
functor. The scalars are `Dispatch::ElementScalar` values, narrowed on the host once per element
type, and the functor receives them after its elements in that type. A HALF or FLOAT32 kernel
then does no `double` arithmetic for a parameter the public signature takes as `double`.

*`bind_reads`* (`functor_map/read_binding.hpp`) declares the tensors a functor reads
at execution time beyond the seam's own operands: a table, a gain, a mask. `read_as_element(t)`
hands the functor a `const T*` in the operation's element type and `read_as<U>(t)` a `const U*`.
The seam checks each declared read (storage, at least one element, the declared dtype, a
device that computes in that dtype, contiguous, no overlap with a tensor the operation writes), adds it to the operand set so the
graph orders against it and retains it, and passes the functor only the pointer. The two bindings
nest: `bind_reads(bind_scalars(fn, ...), ...)`, pointers before scalars.

```cpp
TaskHandle ElementwiseOps::clamp(Tensor& t, double min_val, double max_val)
{
    Dispatch::require_operand(!(min_val > max_val), "math.ew.clamp", "min_val",
                              "is {}, above max_val {}", min_val, max_val);

    return Dispatch::execute_unary_inplace<Dispatch::Domain::RealAndIntegral>(
        recorder_, t, "math.ew.clamp", "math.ew.clamp"_op,
        Dispatch::bind_scalars([](auto x, auto lo, auto hi) { return sycl::clamp(x, lo, hi); },
                               Dispatch::ElementScalar(min_val),
                               Dispatch::ElementScalar(max_val)),
        {min_val, max_val});
}
```

Source: `modules/operations/elementwise/source/clamp.cpp`

The two scalars reach the functor in
the element type of each dtype arm, and the trailing list records them in the task's parameter
slots. The `"..."_op` literal comes from `SushiRuntime::Literals` and gives the
`SushiRuntime::OperationID` of the name.

A bundle of declared reads goes where the functor goes. With scalars inside it:

```cpp
Dispatch::bind_reads(Dispatch::bind_scalars(fn, Dispatch::ElementScalar(2.0)),
                     Dispatch::read_as_element(a))
```

Source: `modules/engine/functor_map/tests/test_read_binding.cpp`

There `fn` takes `(float element, const float* pointer, float scalar)`.

*The GEMM epilogue.* `Dispatch::execute_gemm` records
`C = epilogue(alpha * op(A) * op(B) + beta * C, row, col)` as one task. The epilogue is a functor
`T operator()(T value, int64_t row, int64_t col) const`, applied to each finished element just
before the store. The coordinates make a bias broadcast expressible, so a product, a bias-add and
an activation are one task. `Level3::gemm` is this function with
`Kernels::GemmIdentityEpilogue`, and the reference kernel takes the same parameter, so
`ForceReference` runs a fused GEMM too.

```cpp
struct BiasColumnEpilogue
{
    const float* bias;

    template<typename T>
    [[nodiscard]] T operator()(T value, int64_t, int64_t col) const noexcept
    {
        return value + static_cast<T>(bias[col]);
    }
};

// ...

Dispatch::execute_gemm(probe.recorder(), A, B, C,
                       Core::Transpose::NoTrans, Core::Transpose::NoTrans,
                       1.0, 0.0,
                       "test.gemm_declared_bias", "test.gemm_declared_bias"_op,
                       Dispatch::gemm_epilogue(
                           BiasColumnEpilogue{bias.data_as<float>()}).reads(bias));
```

Source: `modules/operations/blas/tests/test_gemm.cpp`

`probe.recorder()` is a `TaskRecorder&`. The product and the bias broadcast are one task. `tests/package/main.cpp` has
an epilogue that also clamps at zero, handed over as a bare functor.

`Dispatch::gemm_epilogue(functor)` bundles the functor with the tensors it touches, and
`.reads(t)` and `.writes(t)` declare them. `execute_gemm` merges those declarations into the
task's operand set, so the bias is ordered against whatever writes it, retained until the task is
dropped, and refused under `ForceReference` when it is device-only. A bare functor is accepted
and declares nothing. `execute_gemm` defaults to `Domain::Real`; an epilogue that is well-formed
for `std::complex` passes `Domain::RealAndComplex`.

Two warnings travel with the epilogue. An activation fused into the forward pass of training
gives a wrong gradient unless the epilogue also writes its pre-activation value to a buffer the
backward pass reads, declared through `.writes`; nothing throws when this is missing. And under
`ForceReference` the epilogue runs on the host, so a pointer it carries must be host-readable.
