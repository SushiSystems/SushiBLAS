# GPU validation checklist

Everything in this repository and in SushiAI was developed on a machine with **no usable GPU**. `sycl-ls` there reports exactly one device — a Ryzen CPU via Intel OpenCL — and there is no CUDA toolkit, so the NVIDIA path could not be compiled, let alone run.

That is not a small caveat, and it is the reason this file exists. Several changes were made *for* a GPU, justified by argument and by arithmetic, and validated only to the extent a CPU can validate them. This page separates what is known from what is merely believed, so the first person with real hardware knows exactly where to look.

**Nothing below is a report of a passing test. It is a list of claims awaiting one.**

---

## 1. Get a CUDA-capable build

The default build emits SPIR-V and JITs it, which reaches an NVIDIA GPU **not at all**: NVIDIA's OpenCL driver has never implemented `clCreateProgramWithIL`, so it cannot ingest SPIR-V. A binary that "falls back to the CPU" on a machine with a good NVIDIA card is not falling back — the CPU was the only device it could ever see.

```bash
sb build --type release -D SB_SYCL_TARGETS="spir64;nvidia_gpu_sm_86"
```

Substitute your architecture. Keep `spir64` in the list so the same binary still reaches the CPU device; without it you get a GPU-only build.

Two prerequisites the flag cannot check for you:

- the compiler must be an **intel-llvm built with `--cuda`**. A stock oneAPI release has no CUDA backend and will reject the triple.
- intel-llvm documents support for **`sm_75` and newer**. Anything older is below the tested floor even where the flag is accepted.

Configure prints the active targets either way, so an unreachable device is a build-time fact rather than a runtime mystery.

**Expected:** configure prints `SushiBLAS: AOT device targets = spir64;nvidia_gpu_sm_86`, and the build succeeds. **Never compiled on the development machine — this step is the first thing that could fail.**

If it fails here, everything below is moot; report the driver error rather than working around it.

---

## 2. Confirm the device profile is what the kernels assume

Every architecture decision in this library routes through `SushiBLAS::Device::Profile` (`include/SushiBLAS/engine/device_policy.hpp`). On the development CPU it reports `has_dedicated_local_memory = false`, which is why the GPU-shaped code paths were never selected there.

On an NVIDIA GPU it should report `is_cpu = false`, `has_dedicated_local_memory = true`, `sub_group_size = 32`, `local_mem_bytes` of 48 KiB or more, and `compute_units` equal to the SM count.

`has_dedicated_local_memory` is the one that matters: it gates the tiled GEMM and the grid-stride launch shape. If it comes back false on a real GPU, the whole dispatch is inverted and everything below will silently run the CPU shape.

---

## 3. Claims that were derived, not measured

### 3.1 GEMM bank-conflict arithmetic

`kernels/sycl/blas/level3/gemm.hpp` uses a padded tile (`tile_k + 1 = 17`) and a **strided ownership lattice** rather than contiguous ownership. The reasoning, for 32 banks of 4 bytes with an 8×8 register block:

- a 32-lane sub-group spans only 4 distinct `local_row`, so the 8 lanes sharing one are a broadcast rather than a conflict;
- contiguous ownership strides `8 × 16 = 128`, and `128 mod 32 == 0` puts all four on one bank — a 4-way conflict;
- `gcd(17, 32) == 1` gives banks `{0, 17, 2, 19}` — conflict-free, and still so on AMD wave64;
- **padding cannot help the B tile at all**, because its conflict is between lanes varying `local_col` at fixed `kk`. That needed the strided lattice, which coalesces the C store as a side effect.

**Unverified.** Profile it. `ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared` is the direct measurement.

### 3.1a When an axis reduction switches to a work-group per segment

`Reduce::plan_axis` (`include/SushiBLAS/engine/reduce_axis.hpp`) picks between one work-item per segment and one work-group per segment. The rule is: take the cheap item-per-segment shape whenever there are already `compute_units × K_WAVES_PER_COMPUTE_UNIT` segments or more, and otherwise take the work-group shape only if each segment is at least one full group-width long.

The *floor* is an argument — a group whose segment is shorter than the group is idle lanes paying `log2(local)` pairs of barriers — but the **crossover is not measured on a GPU**, because there is no GPU here. Both shapes were measured for correctness and for bit-stability on the OpenCL CPU device, and both are exercised there by `Unit_AxisReductions` (which drives each kernel directly rather than relying on the planner to reach it).

If a row-wise reduction underperforms on real hardware, this is the first knob. It cannot make an answer wrong: the shape is a pure function of the device profile and the span, so it is the same on every run of the same binary, and the two shapes compute the same thing.

A row-wise reduction over a batch — the case the seam was written for — has far more segments than any GPU has resident groups, so it takes the item-per-segment shape on every device. The work-group shape is only reached by a few very long segments.

**That has an unmeasured cost on a GPU, and it is worth stating plainly.** In the item-per-segment shape, work-item `s` walks segment `s`, so for a *row-wise* (axis 1) reduction consecutive work-items read addresses one whole row apart. That is the pattern a CPU wants — each thread streams a contiguous run — and it is exactly the **uncoalesced** pattern a GPU does not: 32 lanes issue 32 separate transactions per step instead of one. The work-group shape is fully coalesced (consecutive lanes read consecutive elements within a segment), but `plan_axis` will not select it for a batch-sized row count, because that would leave one group per row and most of the machine idle.

Nothing here is wrong because of it; both shapes give the same answer. But if a row-wise reduction is slow on real hardware, the fix is likely a third shape — a sub-group per segment, coalesced *and* with enough segments in flight — rather than a tweak to the threshold. That was not written, because a shape that cannot be measured is a shape that cannot be justified.

**Column-wise (axis 0) reductions do not have this problem** in either shape: consecutive segments are adjacent in memory, so consecutive work-items already read consecutive addresses.

### 3.2 Register pressure on the 8×8 block

The register block is 8×8 for 4-byte types — 256 bytes of accumulator, matching portBLAS's `Tile<8,8,8,8>` default — and 4×4 for wider types. A block that spills its accumulator to local memory loses most of the point of blocking.

```bash
sb build --type release -D SB_SYCL_TARGETS=nvidia_gpu_sm_86 -D SB_SYCL_DEVICE_FLAGS="-Xcuda-ptxas;-v"
```

Read the per-kernel register and spill counts. **Nonzero spill on the GEMM kernel means the 8×8 block is too wide for your architecture** and `Detail::register_block_dim`'s byte budget should come down.

### 3.3 The tiled GEMM has never run on scratchpad memory

`gemm_local_tiled` *was* executed on the development machine — the OpenCL CPU device reports enough `local_mem_size` for two tests to call it directly, which proves it compiles, launches, synchronises and is numerically correct on ragged shapes, both layouts and all four transpose combinations.

That is a correctness result and nothing else. It says nothing about performance on real scratchpad hardware, which is the only reason the variant exists.

### 3.4 Double buffering was deliberately omitted

Not an oversight. It could not be verified here, the win over a correctly bank-padded single-buffered kernel is modest, and restructuring the barrier sequence untested was the wrong risk. The reason is recorded in the kernel. If you profile the tiled GEMM and find it latency-bound on the tile loads, this is the next thing to try.

---

## 4. Code paths that have never executed at all

### 4.1 The 128-bit vector path

`include/SushiBLAS/engine/launch_1d.hpp` issues `sycl::vec<T, W>` loads and stores where `Device::vector_width` says a wide access is worth it. **That function returns 1 on a CPU by design** — the OpenCL CPU backend already vectorises across work-items, and a hand-written `sycl::vec` competes with it for the same lanes. So on the development machine `vector_steps` was always 0 and **the wide pass never ran once**.

On a GPU it will run for every 4-byte and 8-byte elementwise, nonlinear and logic op. Check numerical agreement against a CPU run first; a wrong wide path is most likely to show up at a size that is not a multiple of the vector width, where the scalar tail takes over.

### 4.2 The grid-stride launch shape

Also never exercised. On a CPU the library deliberately takes a flat `range<1>` instead, because that measured faster there (~19% at 1M elements) and a CPU has nothing to hold resident. The grid-stride `nd_range` path — including `K_WAVES_PER_COMPUTE_UNIT = 8` as the occupancy target — is GPU-only and unmeasured.

If elementwise ops underperform, that constant is the first knob.

---

## 5. The determinism claim is the one most likely to break

This is the most important item on the page.

SushiBLAS links `sushiruntime_deterministic_fp` PUBLIC, and its guarantee is **same-binary, same-device, byte-for-byte equality**. SushiAI has regression tests that check exactly that — two identical training runs compared as raw bytes, not with a tolerance.

Those tests pass on the development CPU. They have never run on a GPU.

Two things were done specifically so they would keep passing there:

- **gradient accumulation never uses float atomics.** It accumulates through the graph's own ordered edges instead, because a float atomic's combination order is whatever the hardware chooses.
- **reductions no longer use `sycl::reduction`.** `sum`, `mean`, `asum`, `nrm2` and `prod` run a hand-written fixed-order fold — a Neumaier-compensated grid-strided pass, a halving tree over local memory (**not** `reduce_over_group`), then a final pass over a fixed partial count in ascending index order.
- **the axis reductions were built to the same rule.** `max_along` and `sum_along` (`reduce_axis.hpp`) never had a `sycl::reduction` to give up — there is no segmented form of it — but they could have used `reduce_over_group` for the in-group step and deliberately do not. Their item-per-segment shape has no cross-work-item combination at all; their work-group shape uses the same hand-written fixed-offset halving tree. `Unit_AxisReductions.BothShapesAreBitwiseStable` compares repeated runs as bytes at both shapes.

The justification for the second one is an argument from the SYCL specification — it gives no deterministic combination order for a non-associative operator, and floating-point addition is one — **not an observation**. On the development CPU `sycl::reduction` happened to be deterministic, so the change fixed a hazard that had never actually fired. A GPU is where it would.

Run these first:

```bash
sa test --suite regression
```

`Regression_Determinism` is the suite that matters: two SGD runs, two AdamW runs, initialisation, the dataset, and a bare compiled-step replay, all compared byte for byte.

**If any of them fail on a GPU, that is a real finding and worth reporting precisely** — which one, and at which byte. Do not re-record the expectation.

One honest limit already found on CPU: with compensation removed, the *sequential* `single_task` fallback returned a wrong but reproducible answer, because the OpenCL CPU JIT vectorised and reassociated the loop. Same binary and same device still give the same bits, so the contract holds — but "one work-item, therefore provably index order" is not a claim this code can make, and it says so at that branch.

---

## 6. Full validation run

```bash
# SushiBLAS
sb build --type release -D SB_SYCL_TARGETS="spir64;nvidia_gpu_sm_86"
sb test --suite all          # expect 356/356 as of this writing

# SushiAI (builds SushiBLAS from source as a subdirectory)
sa build --type release -D SB_SYCL_TARGETS="spir64;nvidia_gpu_sm_86"
sa test --suite all          # expect 582/582 (SushiAI's own plus this repo's)
sa demo mlp
```

The demo must report `compiles: 1`, a loss falling from `ln C` toward zero, and 100% accuracy on the synthetic problem. `compiles: 1` is the load-bearing line: it means one lowering, one compile, and N replays, which is the premise the whole architecture rests on.

A GPU number for `ms/step` would be the first performance figure this project has that is not a CPU number. The CPU baseline, for reference and for nothing else, is around 7.5 ms/step on a 12-core Ryzen for a 32→128→6 MLP at batch 32.

---

## 7. Known-open items that are not GPU-specific

These are recorded so a GPU run does not get blamed for them:

- ~~**Log-softmax returns `+inf` for a row far below the batch maximum.**~~ **Fixed.** The shift used the global maximum rather than a per-row one; it now uses a per-row one, computed by `ReductionOps::max_along`, which this library did not previously expose. Re-measured on binary32, on a [2, 3] input whose true log-probabilities are −0.41 to −2.41. The quantity below is the worst absolute **deviation of the row's log-probabilities from the answer the same code gives at a gap of 0** — the one figure directly measured on both sides, by `Regression_LogSoftmaxStability.TheAnswerDoesNotDependOnTheGapAtAll`'s `EXPECT_EQ`:

  | row-to-batch-max gap | global shift (before) | per-row shift (after) |
  |---------------------:|----------------------:|----------------------:|
  |                   90 |              2.15e−06 |          0 (exactly)  |
  |                  100 |              2.70e−02 |          0 (exactly)  |
  |                  104 |                `inf`  |          0 (exactly)  |
  |                  200 |                `inf`  |          0 (exactly)  |

  The after column is exactly zero at every gap because the gap no longer enters the arithmetic at all — which is why that assertion is an equality rather than a tolerance.

  Measured against the hand-computed expectation directly, rather than against the gap-0 answer, the gap-100 case came out 2.7033e−02 off — 6.6% of the largest true log-probability (−0.40761), which is the "about 7% low at 100" this entry used to record.

  **The residual at the gaps that pass is not stated, on either side, because it was not measured**: a passing `EXPECT_NEAR` bounds it under 1e−4 and says nothing more. An earlier revision of this entry, and of the test's own header, both carried invented figures for it.
- **SYRK has no local/no-local split and no epilogue.** It always takes the tiled shape, including on a CPU.
- **`nrm2` overflow behaviour** — check whether the scaling form has landed by the time you read this; the naive two-norm overflows above roughly 1e19 in fp32.
- **The batch feed is outside the compiled plan**, costing one extra dispatch round per step. It needs an operand-address late binding that SushiRuntime does not currently offer for ordinary device tasks.
