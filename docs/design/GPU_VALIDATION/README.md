# GPU validation checklist

**Status:** Open — the suite has run on one card; G2 to G7 wait for a profiled run (RW-23 to RW-30)
**About:** Several changes in this repository and in SushiAI were made for a GPU, justified by argument and by arithmetic, and checked only as far as a CPU can check them.
**Modules:** modules/foundation/launch

## Problem

Several changes in this repository and in SushiAI were made for a GPU, justified by argument and
by arithmetic, and checked only as far as a CPU can check them. Until 2026-10-07 no usable GPU
was at hand. The suite has since run on one card, an NVIDIA GeForce RTX 3080 Ti; those runs are in
the GPU runs report. A passing functional suite shows
that the GPU-shaped paths the tests reach gave the expected values. It measures no bank conflict,
no register spill and no crossover.

## Decision

This page separates what was observed from what is believed, so that the first person who
profiles on real hardware knows where to look. Nothing below reports a passing test; it lists
claims that wait for one.

- A build that fails at G1 stops the checklist. Report the driver error; do not work around it.
- A determinism regression that fails on a GPU is reported with the test and the byte. The
  expectation is not re-recorded.
- A constant named below as "the first thing to change" is changed only after a measurement on
  the card.

## Design

### Get a CUDA-capable build

The default build emits SPIR-V and JITs it, and that never reaches an NVIDIA GPU: NVIDIA's
OpenCL driver does not implement `clCreateProgramWithIL`, so it cannot take SPIR-V. A binary
that seems to fall back to the CPU on a machine with an NVIDIA card never saw the card.

```bash
sb build --type release -D SB_SYCL_TARGETS="spir64;nvidia_gpu_sm_86"
```

Substitute your architecture. Keep `spir64` in the list so that the same binary reaches the CPU
device; without it the build is GPU-only.

The flag cannot check two prerequisites:

- The compiler must be an intel-llvm built with `--cuda`. A stock oneAPI release has no CUDA
  backend and rejects the triple.
- intel-llvm documents support for `sm_75` and newer. An older architecture is below the tested
  floor even where the flag is accepted.

Configure prints the active targets either way, so an unreachable device is known at build time.
Expected: configure prints `SushiBLAS: AOT device targets = spir64;nvidia_gpu_sm_86` and the
build succeeds.

### Confirm the device profile is what the kernels assume

Every architecture decision in this library goes through `SushiBLAS::Device::Profile`
(`modules/foundation/launch/include/SushiBLAS/launch/device_policy.hpp`). On the development CPU it reports
`has_dedicated_local_memory = false`, so the GPU-shaped code paths were never selected there.

On an NVIDIA GPU it should report `is_cpu = false`, `has_dedicated_local_memory = true`,
`sub_group_size = 32`, `local_mem_bytes` of 48 KiB or more, and `compute_units` equal to the SM
count.

`has_dedicated_local_memory` gates the tiled GEMM and the grid-stride launch shape. If it comes
back false on a GPU, the dispatch is inverted and everything below runs the CPU shape without
saying so.

### Claims that were derived, not measured

#### GEMM bank-conflict arithmetic

`modules/kernels/blas_seam_kernels/include/SushiBLAS/blas_seam_kernels/sycl/level3/gemm.hpp` uses a padded tile (`K_TILE_K + 1 = 17`)
and a strided ownership lattice in place of contiguous ownership. The reasoning, for 32 banks of
4 bytes with an 8×8 register block:

- A 32-lane sub-group spans 4 distinct `local_row` values, so the 8 lanes that share one are a
  broadcast and not a conflict.
- Contiguous ownership strides `8 × 16 = 128`, and `128 mod 32 == 0` puts all four on one bank,
  a 4-way conflict.
- `gcd(17, 32) == 1` gives banks `{0, 17, 2, 19}`, which do not conflict, on AMD wave64 either.
- Padding cannot help the B tile, because its conflict is between lanes that vary `local_col`
  at a fixed `kk`. That needed the strided lattice, which also coalesces the C store.

Unverified. `ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared` measures it directly.

#### When an axis reduction switches to a work-group per segment

`Reduce::plan_axis` (`modules/foundation/reduction/include/SushiBLAS/reduction/reduce_axis.hpp`) picks between one work-item
per segment and one work-group per segment. It takes the item-per-segment shape whenever there
are `compute_units × K_WAVES_PER_COMPUTE_UNIT` segments or more, and otherwise takes the
work-group shape only if each segment is at least one full group-width long.

The floor is an argument: a group whose segment is shorter than the group holds idle lanes that
pay `log2(local)` pairs of barriers. The crossover is not measured on a GPU. Both shapes were
measured for correctness and for bit-stability on the OpenCL CPU device, where
`Medium_AxisReductions` exercises both by driving each kernel directly.

If a row-wise reduction underperforms on real hardware, this threshold is the first thing to
change. It cannot make an answer wrong: the shape is a pure function of the device profile and
the span, so it is the same on every run of one binary, and the two shapes compute the same
value.

A row-wise reduction over a batch, the case the seam was written for, has far more segments
than a GPU has resident groups, so it takes the item-per-segment shape on every device. Only a
few very long segments reach the work-group shape.

That has an unmeasured cost on a GPU. In the item-per-segment shape, work-item `s` walks segment
`s`, so for a row-wise (axis 1) reduction consecutive work-items read addresses one whole row
apart. A CPU wants that pattern, because each thread streams a contiguous run. A GPU does not:
the reads are uncoalesced, and 32 lanes issue 32 transactions per step where one would do. The
work-group shape is coalesced, since consecutive lanes read consecutive elements within a
segment, but `plan_axis` does not select it for a batch-sized row count, because that would
leave one group per row and most of the machine idle.

Both shapes give the same answer, so nothing is wrong because of this. If a row-wise reduction
is slow on real hardware, the likely fix is a third shape, a sub-group per segment, which is
coalesced and keeps enough segments in flight. It was not written, because a shape that cannot
be measured cannot be justified.

Column-wise (axis 0) reductions do not have this problem in either shape: consecutive segments
are adjacent in memory, so consecutive work-items read consecutive addresses.

#### Register pressure on the 8×8 block

The register block is 8×8 for 4-byte types, 256 bytes of accumulator, which matches portBLAS's
`Tile<8,8,8,8>` default, and 4×4 for wider types. A block that spills its accumulator to local
memory loses most of what blocking buys.

```bash
sb build --type release -D SB_SYCL_TARGETS=nvidia_gpu_sm_86 -D SB_SYCL_DEVICE_FLAGS="-Xcuda-ptxas;-v"
```

Read the register and spill counts of each kernel. A spill count above zero on the GEMM kernel
means the 8×8 block is too wide for the architecture, and the byte budget of
`register_block_dim` should come down.

#### The speed of the tiled GEMM on scratchpad memory is unmeasured

`gemm_local_tiled` ran on the development machine: the OpenCL CPU device reports enough
`local_mem_size` for two tests to call it directly. That shows it compiles, launches,
synchronises and is numerically correct on ragged shapes, both layouts and all four transpose
combinations.

It is a correctness result. It says nothing about speed on real scratchpad hardware, and speed
there is the reason the variant exists.

#### The tiled GEMM is slower at order 2048

The tiled GEMM stages each tile into one local buffer, with 16-byte loads of whole aligned blocks.
On the RTX 3080 Ti it is 18 to 42 % faster than the scalar kernel it replaced up to order 1024 in
FLOAT32, and 14 % slower in ColumnMajor and 24 % slower in RowMajor at 2048. The measurements are
in `docs/agent/2026_10_19_KERNEL_AUDIT/REPORT.md`. Turning the 16-byte loads off made every
order slower, and two local buffers made 2048 slower still, so neither is the cause. The two
candidates left are the contiguous-axis staging order and the whole-tile path; a register and
occupancy read of the compiled kernel, under RW-92, decides between them.

The tiled GEMM now runs one of three shapes per device. Until RW-110 measures the card, the
3080 Ti takes the model's choice, `Wide` at order 2048, which nobody has timed.

### Code paths the development CPU never took

The GPU runs expect both paths to be taken. Which paths those runs took was not recorded, so the
two items stay open until a run prints it.

#### The 128-bit vector path

`modules/foundation/launch/include/SushiBLAS/launch/launch_1d.hpp` issues `sycl::vec<T, W>` loads and stores where
`Device::vector_width` says a wide access pays. That function returns 1 on a CPU by design: the
OpenCL CPU backend already vectorises across work-items, and a hand-written `sycl::vec` competes
with it for the same lanes. On the development machine `vector_steps` was therefore always 0 and
the wide pass never ran.

On a GPU it runs for every 4-byte and 8-byte elementwise, nonlinear and logic operation. Check
numerical agreement against a CPU run first. A wrong wide path is most likely to show at a size
that is not a multiple of the vector width, where the scalar tail takes over.

#### The grid-stride launch shape

On a CPU the library takes a flat `range<1>`, because that measured about 19% faster there at
1M elements and a CPU has nothing to hold resident. The grid-stride `nd_range` path, with
`K_WAVES_PER_COMPUTE_UNIT = 8` as its occupancy target, is taken only on a GPU and is
unmeasured.

If elementwise operations underperform, that constant is the first thing to change.

### The determinism claim

Of the claims on this page, this one is the most likely to break on a GPU.

SushiBLAS links `sushiruntime_deterministic_fp` `PUBLIC`, and its guarantee is byte-for-byte
equality for the same binary on the same device. SushiAI has regression tests that check it:
two identical training runs compared as raw bytes, with no tolerance.

Those tests pass on the development CPU. No run of them on a GPU is recorded.

Three things were done so that they keep passing there:

- Gradient accumulation uses no float atomics. It accumulates through the graph's ordered
  edges, because the hardware chooses the combination order of a float atomic.
- Reductions do not use `sycl::reduction`. `sum`, `mean`, `asum`, `nrm2` and `prod` run a
  hand-written fold in a fixed order: a Neumaier-compensated grid-strided pass, a halving tree
  over local memory that does not call `reduce_over_group`, then a final pass over a fixed
  partial count in ascending index order.
- The axis reductions follow the same rule. `max_along` and `sum_along` (`reduce_axis.hpp`)
  never had a `sycl::reduction` to give up, since it has no segmented form, and they do not use
  `reduce_over_group` for the in-group step. Their item-per-segment shape combines nothing
  across work-items; their work-group shape uses the same hand-written halving tree with fixed
  offsets. `Medium_AxisReductions.BothShapesAreBitwiseStable` compares repeated runs as bytes at
  both shapes.

The second item rests on an argument from the SYCL specification, which gives no deterministic
combination order for a non-associative operator such as floating-point addition. It does not
rest on an observation. On the development CPU `sycl::reduction` happened to be deterministic,
so the change removed a hazard that had never fired. A GPU is where it would fire.

Run these first:

```bash
sa test --suite regression
```

`Regression_Determinism` holds the tests in question: two SGD runs, two AdamW runs,
initialisation, the dataset and a bare replay of a compiled step, all compared byte for byte.

One limit was already found on the CPU. With compensation removed, the sequential `single_task`
fallback returned a wrong but reproducible answer, because the OpenCL CPU JIT vectorised and
reassociated the loop. The same binary on the same device still gives the same bits, so the
contract holds. The code cannot claim that one work-item means index order, and the comment at
that branch says so.

### Full validation run

```bash
# SushiBLAS
sb build --type release -D SB_SYCL_TARGETS="spir64;nvidia_gpu_sm_86"
sb test --suite all

# SushiAI (builds SushiBLAS from source as a subdirectory)
sa build --type release -D SB_SYCL_TARGETS="spir64;nvidia_gpu_sm_86"
sa test --suite all
sa demo mlp
```

The demo must report `compiles: 1`, a loss that falls from `ln C` toward zero, and 100% accuracy
on the synthetic problem. `compiles: 1` means one lowering, one compile and N replays, which is
what the architecture assumes. A GPU figure for `ms/step` would be this project's first speed
figure from a GPU; the CPU baseline is in the GPU runs report.

### Items that are not GPU-specific

The open items that are not GPU-specific are in [`REMAINING_WORK.md`](../../backlog/README.md), under
"Kernels and reductions", so that a GPU run is not blamed for them. The log-softmax fix, whose
measurements other documents cite, is in the GPU runs report.

## Phases

| Id | Phase | State |
| --- | --- | --- |
| G1 | A CUDA-capable build reaches the card | built |
| G2 | Confirm the device profile on the card (RW-23) | planned |
| G3 | Measure the tiled GEMM's bank conflicts and register spill (RW-24) | planned |
| G4 | Run the 128-bit vector path and the grid-stride launch (RW-25) | planned |
| G5 | Measure where an axis reduction switches to a work-group per segment (RW-26) | planned |
| G6 | Run SushiAI's determinism regressions on the GPU (RW-27) | planned |
| G7 | Run the full validation run with a profiled `ms/step` | planned |
| G8 | The reference sums in double or compensated, after the SYMV parity failure | built |
| G9 | Log-softmax with a per-row shift | built |

### Work

<!-- work:begin -->
No open work.
<!-- work:end -->

## Open questions

- Does the 2048-order slowdown of the tiled GEMM come from the contiguous-axis staging order or
  from the whole-tile path? RW-92 decides it.
- Does a row-wise reduction over a batch need a sub-group-per-segment shape on a GPU?
