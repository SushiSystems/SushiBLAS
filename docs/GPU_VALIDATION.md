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
sb test --suite all          # expect 306/306 as of this writing

# SushiAI (builds SushiBLAS from source as a subdirectory)
sa build --type release -D SB_SYCL_TARGETS="spir64;nvidia_gpu_sm_86"
sa test --suite all          # expect 499/499
sa demo mlp
```

The demo must report `compiles: 1`, a loss falling from `ln C` toward zero, and 100% accuracy on the synthetic problem. `compiles: 1` is the load-bearing line: it means one lowering, one compile, and N replays, which is the premise the whole architecture rests on.

A GPU number for `ms/step` would be the first performance figure this project has that is not a CPU number. The CPU baseline, for reference and for nothing else, is around 7.5 ms/step on a 12-core Ryzen for a 32→128→6 MLP at batch 32.

---

## 7. Known-open items that are not GPU-specific

These are recorded so a GPU run does not get blamed for them:

- **Log-softmax returns `+inf` for a row far below the batch maximum.** The shift uses the global maximum rather than a per-row one. Measured on binary32: exact to a gap of ~90, about 7% low at 100, `+inf` from ~104. Needs a row-wise maximum, which needs an axis reduction this library does not yet expose.
- **SYRK has no local/no-local split and no epilogue.** It always takes the tiled shape, including on a CPU.
- **`nrm2` overflow behaviour** — check whether the scaling form has landed by the time you read this; the naive two-norm overflows above roughly 1e19 in fp32.
- **The batch feed is outside the compiled plan**, costing one extra dispatch round per step. It needs an operand-address late binding that SushiRuntime does not currently offer for ordinary device tasks.
