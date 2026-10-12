# GPU validation checklist

**Status:** Open. Functional runs on one card exist (section 0); nothing in sections 2 to 5 has been measured.

Until 2026-10-07 everything in this repository and in SushiAI was developed on a machine with no
usable GPU: `sycl-ls` reported one device, a Ryzen CPU through Intel OpenCL, and the NVIDIA path
could not be compiled. Sections 1 to 7 were written in that state. Section 0 records the GPU
runs made since.

Several changes were made for a GPU, justified by argument and by arithmetic, and checked only
as far as a CPU can check them. This page separates what was observed from what is believed, so
that the first person who profiles on real hardware knows where to look.

Outside section 0, nothing below reports a passing test. It lists claims that wait for one.

## 0. The GPU runs

Both runs were on one card, an NVIDIA GeForce RTX 3080 Ti, which `sycl-ls` names
`[cuda:gpu][cuda:0] NVIDIA CUDA BACKEND, NVIDIA GeForce RTX 3080 Ti 8.6 [CUDA 13.3]`.

| | First run | Second run |
| --- | --- | --- |
| Date | 2026-10-07, at 14:15 and repeated at 14:53 | 2026-10-09 |
| Tree | commit `dd5cdfc`, before the first commit of the v0.1.0 work | the v0.1.0 tree of that day, after the interface break |
| Result | 790 of 792 tests passed | 2 047 of 2 047 tests passed |
| Skipped | `Regression_LoggerDeathTest.AssertWritesItsMessageBeforeAborting`, a death test that needs a debug build | CTest lists some tests as skipped; their count and their reason were not recorded |
| Failed | `Unit_Level2SyclParityTest.SymvMatchesReferenceLarge<float>` | none |

The failure of the first run was a finding about the reference kernel. For n = 4096 and the
lower triangle, at element 1148:

| | Value | Error |
| --- | --- | --- |
| Exact, from the test's data | 400.839198 | |
| SYCL kernel on the GPU | 400.839203 | 5e-6, the float nearest the exact value |
| Reference kernel | 400.848022 | 8.8e-3 |
| The test's allowance | | 8.1e-3 |

The reference `symv` added 4096 `FLOAT32` products one after another in a `FLOAT32`, and a sum of
that length may lose about `4096 * 2^-24 * sum(|term|)`, 0.42 for this row. The GPU kernel
reduces the row in work-group partials and landed on the nearest float. The test passed on the
CPU device because a row of 4096 is below the CPU's two-stage threshold there, so the SYCL kernel
summed in the reference's order and shared its error. The three numbers were reproduced on the
host from the test's data pattern.

The fix is in the tree the second run tested. Every running sum of a reference BLAS kernel is a
`Kernels::ReferenceSum<T>`: double for `HALF`, `FLOAT32` and `COMPLEX32` terms, a compensated
double sum for `FLOAT64` and `COMPLEX64` (`modules/foundation/support/include/SushiBLAS/support/reference_accumulator.hpp`).
The parity test compares each kernel with a double-precision truth under a bound derived from
the term count (`tests/common/blas_truth.hpp`,
`modules/operations/blas/tests/test_level2_sycl_parity.cpp`).

What the runs do not show:

- Which of the items of sections 2 to 5 they exercised was not recorded. A passing functional
  suite shows that the GPU-shaped paths the tests reach gave the expected values; it measures no
  bank conflict, no register spill and no crossover.
- The device profile of section 2 was not printed in either run.
- The tree has run on no other card and, since the break, not on the OpenCL CPU device.
  [REMAINING_WORK.md](REMAINING_WORK.md) lists the runs that settle this.

## 1. Get a CUDA-capable build

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
build succeeds. Both builds behind section 0 produced a binary that reached the card; what
configure printed for them was not kept.

If the build fails here, nothing below applies. Report the driver error; do not work around it.

## 2. Confirm the device profile is what the kernels assume

Every architecture decision in this library goes through `SushiBLAS::Device::Profile`
(`modules/foundation/support/include/SushiBLAS/support/device_policy.hpp`). On the development CPU it reports
`has_dedicated_local_memory = false`, so the GPU-shaped code paths were never selected there.

On an NVIDIA GPU it should report `is_cpu = false`, `has_dedicated_local_memory = true`,
`sub_group_size = 32`, `local_mem_bytes` of 48 KiB or more, and `compute_units` equal to the SM
count.

`has_dedicated_local_memory` gates the tiled GEMM and the grid-stride launch shape. If it comes
back false on a GPU, the dispatch is inverted and everything below runs the CPU shape without
saying so.

## 3. Claims that were derived, not measured

### 3.1 GEMM bank-conflict arithmetic

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

### 3.1a When an axis reduction switches to a work-group per segment

`Reduce::plan_axis` (`modules/foundation/support/include/SushiBLAS/support/reduce_axis.hpp`) picks between one work-item
per segment and one work-group per segment. It takes the item-per-segment shape whenever there
are `compute_units × K_WAVES_PER_COMPUTE_UNIT` segments or more, and otherwise takes the
work-group shape only if each segment is at least one full group-width long.

The floor is an argument: a group whose segment is shorter than the group holds idle lanes that
pay `log2(local)` pairs of barriers. The crossover is not measured on a GPU. Both shapes were
measured for correctness and for bit-stability on the OpenCL CPU device, where
`Unit_AxisReductions` exercises both by driving each kernel directly.

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

### 3.2 Register pressure on the 8×8 block

The register block is 8×8 for 4-byte types, 256 bytes of accumulator, which matches portBLAS's
`Tile<8,8,8,8>` default, and 4×4 for wider types. A block that spills its accumulator to local
memory loses most of what blocking buys.

```bash
sb build --type release -D SB_SYCL_TARGETS=nvidia_gpu_sm_86 -D SB_SYCL_DEVICE_FLAGS="-Xcuda-ptxas;-v"
```

Read the register and spill counts of each kernel. A spill count above zero on the GEMM kernel
means the 8×8 block is too wide for the architecture, and the byte budget of
`Detail::register_block_dim` should come down.

### 3.3 The speed of the tiled GEMM on scratchpad memory is unmeasured

`gemm_local_tiled` ran on the development machine: the OpenCL CPU device reports enough
`local_mem_size` for two tests to call it directly. That shows it compiles, launches,
synchronises and is numerically correct on ragged shapes, both layouts and all four transpose
combinations.

It is a correctness result. It says nothing about speed on real scratchpad hardware, and speed
there is the reason the variant exists.

### 3.4 Double buffering was left out

Double buffering could not be verified on the development machine, its gain over a single-buffered
kernel with correct bank padding is modest, and restructuring the barrier sequence untested was
the larger risk. The kernel records the reason. If a profile shows the tiled GEMM bound by the
latency of its tile loads, double buffering is the next thing to try.

## 4. Code paths the development CPU never took

The runs of section 0 were on a GPU, where this section expects both paths to be taken. Which
paths those runs took was not recorded, so the two items stay open until a run prints it.

### 4.1 The 128-bit vector path

`modules/foundation/support/include/SushiBLAS/support/launch_1d.hpp` issues `sycl::vec<T, W>` loads and stores where
`Device::vector_width` says a wide access pays. That function returns 1 on a CPU by design: the
OpenCL CPU backend already vectorises across work-items, and a hand-written `sycl::vec` competes
with it for the same lanes. On the development machine `vector_steps` was therefore always 0 and
the wide pass never ran.

On a GPU it runs for every 4-byte and 8-byte elementwise, nonlinear and logic operation. Check
numerical agreement against a CPU run first. A wrong wide path is most likely to show at a size
that is not a multiple of the vector width, where the scalar tail takes over.

### 4.2 The grid-stride launch shape

On a CPU the library takes a flat `range<1>`, because that measured about 19% faster there at
1M elements and a CPU has nothing to hold resident. The grid-stride `nd_range` path, with
`K_WAVES_PER_COMPUTE_UNIT = 8` as its occupancy target, is taken only on a GPU and is
unmeasured.

If elementwise operations underperform, that constant is the first thing to change.

## 5. The determinism claim

Of the claims on this page, this one is the most likely to break on a GPU.

SushiBLAS links `sushiruntime_deterministic_fp` `PUBLIC`, and its guarantee is byte-for-byte
equality for the same binary on the same device. SushiAI has regression tests that check it:
two identical training runs compared as raw bytes, with no tolerance.

Those tests pass on the development CPU. No run of them on a GPU is recorded here.

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
  offsets. `Unit_AxisReductions.BothShapesAreBitwiseStable` compares repeated runs as bytes at
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

If one of them fails on a GPU, report which one and at which byte. Do not re-record the
expectation.

One limit was already found on the CPU. With compensation removed, the sequential `single_task`
fallback returned a wrong but reproducible answer, because the OpenCL CPU JIT vectorised and
reassociated the loop. The same binary on the same device still gives the same bits, so the
contract holds. The code cannot claim that one work-item means index order, and the comment at
that branch says so.

## 6. Full validation run

```bash
# SushiBLAS
sb build --type release -D SB_SYCL_TARGETS="spir64;nvidia_gpu_sm_86"
sb test --suite all          # 403/403 on the development CPU, 2026-08-03

# SushiAI (builds SushiBLAS from source as a subdirectory)
sa build --type release -D SB_SYCL_TARGETS="spir64;nvidia_gpu_sm_86"
sa test --suite all          # 582/582 when last recorded here; not re-run since,
                              # and this repo has gained tests since, so expect more
sa demo mlp
```

The demo must report `compiles: 1`, a loss that falls from `ln C` toward zero, and 100% accuracy
on the synthetic problem. `compiles: 1` means one lowering, one compile and N replays, which is
what the architecture assumes.

A GPU figure for `ms/step` would be this project's first speed figure from a GPU. The CPU
baseline is about 7.5 ms/step on a 12-core Ryzen for a 32→128→6 MLP at batch 32.

## 7. Items that are not GPU-specific

The open items this section listed are in [`REMAINING_WORK.md`](REMAINING_WORK.md), under
"Kernels and reductions", so that a GPU run is not blamed for them. One closed item stays here
because other documents cite its measurements.

Log-softmax returned `+inf` for a row far below the batch maximum, and is fixed. The shift used
the global maximum; it now uses a per-row maximum, computed by `ReductionOps::max_along`. The
measurement was repeated on binary32, on a [2, 3] input whose true log-probabilities run from
−0.41 to −2.41. The quantity below is the worst absolute deviation of the row's
log-probabilities from the answer the same code gives at a gap of 0. It is the one figure
measured on both sides, by the `EXPECT_EQ` of
`Regression_LogSoftmaxStability.TheAnswerDoesNotDependOnTheGapAtAll`:

| Gap between the row and the batch maximum | Global shift, before | Per-row shift, after |
| ---: | ---: | ---: |
| 90 | 2.15e−06 | 0, exactly |
| 100 | 2.70e−02 | 0, exactly |
| 104 | `inf` | 0, exactly |
| 200 | `inf` | 0, exactly |

The after column is zero at every gap because the gap does not enter the arithmetic, and the
assertion is an equality for that reason.

Measured against the hand-computed expectation, and not against the answer at gap 0, the
gap-100 case came out 2.7033e−02 off, which is 6.6% of the largest true log-probability
(−0.40761). That is the "about 7% low at 100" an earlier text of this entry recorded.

The residual at the gaps that pass is not stated on either side, because it was not measured: a
passing `EXPECT_NEAR` bounds it under 1e−4 and says no more. An earlier revision of this entry
and of the test's header both carried invented figures for it.
