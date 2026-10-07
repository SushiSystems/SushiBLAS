# Logic headers

`include/SushiBLAS/engine/logic` declares the comparison and logical operations.

| File | Declares |
| --- | --- |
| `logic.hpp` | `LogicOps`, the view that records the comparison and logical operations |
| `dispatch.hpp` | `execute_logic_unary`, `execute_logic_binary`, `execute_logic_ternary` and the comparison helpers they share |

Every executor takes a `TaskRecorder&` and returns a `TaskHandle`. The three executors record the
same `Detail::MapTask` the elementwise family does, so they accept a `bind_reads` bundle and have
the serial host path under `ForceReference`; `include/SushiBLAS/engine/math/README.md` describes
both under "The per-element seam".

Each section below holds the reasoning a header carried in its comments until 2026-10-05, under the
name of the file and of the symbol it belongs to. The header states the rule in one line and cites
this page.

## `dispatch.hpp`

### Empty operands

Each executor checks shapes, layouts, dtypes and the domain, then returns a default `TaskHandle`
without recording a task when `result` holds no elements. A tensor with zero elements needs no
storage. `include/SushiBLAS/engine/math/README.md` states the same rule for the elementwise family.

### `execute_logic_unary`

Logic results are carried in the operand's own dtype (1 for true, 0 for false) rather than a
dedicated boolean tensor type, so `result` must match `x`.

Warning: This is a low-level generic seam. `op_func` is inlined into a SYCL device kernel: it must
be device-copyable, capture by value only, hold no host pointers, allocate nothing and throw
nothing.

Throws `std::invalid_argument` on an operand or declared-read violation or too many `params`, and
`UnsupportedExecution` on a device-only tensor under `ForceReference` or a dtype a tensor's
device cannot compute in.

### Ordering helpers

The ordering helpers have no complex overloads, on purpose. `<`, `<=`, `>` and `>=` have no meaning
on complex numbers; these helpers used to supply a lexicographic stand-in, which made `less(a, b)`
on a COMPLEX32 tensor return a confident, well-formed, mathematically meaningless answer. The
ordering ops now declare Domain::Real, so a complex operand throws at record time instead of being
quietly re-interpreted.

## `logic.hpp`

### `all` and `any` of an empty tensor

An input with zero elements needs no storage. `all` still records its task and writes 1 to
`result`, `any` writes 0: each is its reduction's identity, and the kernel reads no input element.
