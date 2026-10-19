# Parametric GEMM tiles

**Status:** Open — specified; the owner decided section 7 on 2026-10-19 and holds the start (RW-107).

## 1. Problem

`gemm_local_tiled` runs one tile shape on every device. `GemmLocalTile<T>` in
`modules/kernels/blas_seam_kernels/include/SushiBLAS/blas_seam_kernels/sycl/level3/gemm.hpp`
fixes the shape at compile time: an 8 x 8 work-group, an 8 x 8 register block for FLOAT32, a
depth of 16, an A tile padded to 17 and a B tile padded to `K_BLOCK_N + 2`. `Device::Profile`
chooses between the GEMM variants. It does not choose a shape inside one.

The shape that is fastest depends on the device: local memory per compute unit, registers per
compute unit, sub-group width, bank count and the number of compute units. The kernel audit of
2026-10-19 measured this on the RTX 3080 Ti. A second local buffer cost 16 % at order 2048,
because the larger tile let fewer work-groups share one SM. A device with 164 or 228 KB of
local memory per compute unit would likely gain from the same buffer. One shape cannot be
right for NVIDIA, AMD and Intel GPUs at once. Today the library ships the shape that was
convenient on one card, and on every other card it leaves the gap unmeasured.

## 2. Goal

Each device runs the tile shape that suits it. The shape is chosen when the GEMM is launched,
from a small set of shapes compiled in advance, by a rule that a reader can follow:

1. a measured entry for the device's architecture and the problem's size class, when one exists;
2. otherwise an analytic choice from the device's queried traits;
3. always within hard limits: the tile fits in local memory, and the work-group fits the device
   and the compiled kernel.

A tuning program measures every shape on the device it runs on and prints the entries for the
table. A user can pin one shape for a run, for tuning and for bug reports.

## 3. What the literature and other libraries do

The survey of 2026-10-19 found:

- **Shipped databases keyed by device.** CLBlast keeps tuned parameters per device and kernel,
  with defaults for an unknown device (Nugteren 2017, arXiv 1705.05249). Tensile and hipBLASLt
  write per-`gfx` logic files offline. portBLAS picks a tuning target per vendor at build time
  and chooses among a few launchers by an arithmetic-intensity test on M, N and K.
- **Models filter, measurements pick.** BLIS derives CPU block sizes from cache geometry and
  matches tuned results (Low et al., ACM TOMS 2016). tritonBLAS reports over 95 % of autotuned
  speed from an analytic model (arXiv 2512.04226, the authors' figure). Occupancy limits are
  used as hard filters before timing, as in the UTK pruning of an SGEMM space from 388 567 to
  about 79 532 configurations.
- **A few shapes fit most devices.** On the CLBlast data, a small set of kernel variants chosen
  without per-device retuning lands near the tuned speed on unseen devices (Hochgraf and Pai,
  arXiv 2507.15277).
- **Tuning is offline.** CUTLASS profiles a candidate list. hipBLASLt and ISAAC tune ahead of
  time because a first-call search costs seconds to hours.
- **Reproducibility is per configuration.** cuBLAS and oneMKL CNR promise equal bits on one
  device class, one configuration and one library version, never across devices.

## 4. What SYCL offers here

- A tile shape sizes register arrays and local arrays, so it must be a compile-time constant.
  Specialization constants cannot size either. Each shape is therefore its own template
  instantiation and its own kernel. C++17 has no class-type non-type template parameters, so a
  shape is a type carrying `static constexpr` members.
- `info::device::architecture` from the experimental `sycl_ext_oneapi_device_architecture`
  names the device (`nvidia_gpu_sm_86`, `amd_gpu_gfx90a`, `intel_gpu_pvc`, ...) or returns
  `unknown`. The fallback is the vendor, `local_mem_size`, `max_work_group_size`,
  `sub_group_sizes` and `max_compute_units`, which `Device::Profile` partly holds today.
- `get_kernel_info<Kernel, info::kernel_device_specific::work_group_size>(queue)` from
  `sycl_ext_oneapi_get_kernel_info`, which DPC++ fully supports, gives the largest work-group a
  compiled kernel accepts, which registers limit. It is the portable hard check a shape needs
  before launch.
- `if_architecture_is` cannot resolve in the `spir64` image this project builds, so the choice
  is made on the host among kernel names, never inside a kernel.
- `sycl_ext_oneapi_launch_queries` (proposed) and `sycl_ext_oneapi_kernel_compiler`
  (experimental) are kept out of shipped paths.

## 5. Design

### 5.1 Bricks

| Brick | Module | Holds |
| --- | --- | --- |
| `Device::Architecture` and the vendor | `launch` | The architecture the device reports, or `Unknown`, and the vendor; added to `Device::Profile` |
| `GemmTileShape` | `blas_seam_kernels` | One shape as a type: work-group rows and columns, register block rows and columns, depth, A and B padding, buffer count. Its derived sizes: block M and N, local bytes, work-items |
| The shape catalog | `blas_seam_kernels` | The shapes compiled for each dtype, each with a stable name; one type list, one place to add a shape |
| `gemm_local_tiled<Shape>` | `blas_seam_kernels` | Today's kernel with the shape as a template parameter; every constant it reads comes from `Shape` |
| `GemmTileTable` | `blas_seam_kernels` | Measured entries: architecture, dtype, size class to a shape name. Plain data in one header |
| `choose_gemm_tile` | `blas_seam_kernels` | The host rule of section 2: table, else model, always the hard limits. A pure function of the profile, the dtype and the shape of the product, tested without a device |
| Pinning | `blas_seam_kernels` | `SB_GEMM_TILE=<name>` read once; a name outside the catalog or past a hard limit logs a WARN and is ignored |
| `benchmark_gemm_tiles` | `tests/operations/blas` | Times every catalog shape on the device for each size class and prints the table entries, one line each |

`choose_gemm_tile` returns a shape name, and `gemm_accessed` dispatches on it with one switch
over the catalog. The catalog is closed by its meaning and small, so a switch is the right shape
here (`sushiskills`, default 1, "not when").

### 5.2 Size classes

Three classes, each defined by how full the device is and how deep the product runs:

- **small**: fewer output tiles than compute units at the default shape;
- **skinny**: M or N under one block while K is long, the case split-K serves;
- **large**: everything else.

A class only selects among shapes. Split-K stays in `gemm_depth_split`, which the chooser
already reaches first.

### 5.3 The analytic model

For a shape S on a device D:

- S is admissible when its local bytes fit `local_mem_size`, its work-items fit
  `max_work_group_size` and the compiled kernel's `work_group_size`, and its work-group is a
  whole number of sub-groups.
- Resident work-groups per compute unit are limited by local memory per compute unit, which
  the architecture names (a table of known values, falling back to `local_mem_size`), and by
  the work-group count the kernel query allows.
- Among admissible shapes, the model takes the one with the most arithmetic per staged byte
  that still fills every compute unit at least once for the product's tile count; ties go to
  fewer local bytes.

The model is a fallback for devices nobody has measured. It is never trusted over a
measurement.

### 5.4 Bits

Every shape of `gemm_local_tiled` accumulates each element of C in the same order: one fused
multiply-add per k, k rising. The depth tile only splits that loop. A shape change therefore
gives the same bits, and the plan carries a test that holds every catalog shape to the bits of
the default one, on the GPU and on the CPU device. `gemm_depth_split` and `gemm_joint_matrix`
sum in other orders and keep their own promises.

### 5.5 What this covers and what it does not

Covered: the FLOAT32, FLOAT64 and complex `gemm_local_tiled` path and every caller of
`gemm_accessed` (gemm, symm, packed fallback).

Not covered, each its own backlog entry: the `syrk` and `syr2k` tiles, which define their own
constants; the no-local register block; the CPU packed register block per ISA (B18); a user
cache file written by the tuner; a first-call tuner; Stream-K; `joint_matrix` shapes read from
`matrix_combinations`.

## 6. Risks

- **Binary size and build time.** Each shape adds a SPIR-V, a PTX and a SASS image per dtype.
  The plan measures the cost of each added shape before the catalog grows past three per dtype.
- **One card to measure on.** Only the RTX 3080 Ti and the Ryzen 5 7600X are here. Entries for
  any other architecture come from the tuner run on that hardware, not from this plan. Until
  then those devices take the model.
- **Experimental extension.** `sycl_ext_oneapi_device_architecture` may change. It is read in
  one function in `launch`, behind its feature-test macro, and an absent extension means
  `Unknown`.

## 7. Decisions of the owner, 2026-10-19

1. `Device::Profile` gains public architecture and vendor fields. The architecture is read from
   the experimental extension behind its feature-test macro, and is `Unknown` without it.
2. A shape is pinned through the environment variable `SB_GEMM_TILE` only; no API on `Engine`.
3. The catalog holds at most three shapes per dtype until the size measurement allows more.
4. The measured table holds the RTX 3080 Ti's entries only; every other device takes the model
   until someone runs the tuner on its hardware.
