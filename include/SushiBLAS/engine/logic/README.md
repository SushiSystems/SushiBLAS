# Logic headers

`include/SushiBLAS/engine/logic` declares `LogicOps` (`logic.hpp`) and the logic executors of the
dispatch seam (`dispatch.hpp`).

Each section below holds the reasoning a header carried in its comments until 2026-10-05, under the
name of the file and of the symbol it belongs to. The header states the rule in one line and cites
this page.

## `dispatch.hpp`

### `execute_logic_unary`

Logic results are carried in the operand's own dtype (1 for true, 0 for false) rather than a
dedicated boolean tensor type, so `result` must match `x`.

Warning: This is a low-level generic seam. `op_func` is inlined into a SYCL device kernel: it must
be device-copyable, capture by value only, hold no host pointers, allocate nothing and throw
nothing.

### Ordering helpers

The ordering helpers have no complex overloads, on purpose. `<`, `<=`, `>` and `>=` have no meaning
on complex numbers; these helpers used to supply a lexicographic stand-in, which made `less(a, b)`
on a COMPLEX32 tensor return a confident, well-formed, mathematically meaningless answer. The
ordering ops now declare Domain::REAL, so a complex operand throws at record time instead of being
quietly re-interpreted.
