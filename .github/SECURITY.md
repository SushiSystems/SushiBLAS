# Security Policy

We take the security of SushiBLAS seriously. This document explains how to
report a vulnerability and what to expect in return.

## Reporting a vulnerability

**Please do not open a public GitHub issue for a security problem.** Public
disclosure before a fix is available puts every user at risk.

Instead, report it privately through one of:

- **Email (preferred):** **mustafagarip@sushisystems.io**
- **Discord:** reach the maintainer privately on the community server —
  invite: `https://discord.gg/y5639YuaPh`

If you prefer, you may also use GitHub's **private vulnerability reporting**
("Report a vulnerability" under the repository's Security tab).

### What to include

A good report lets us reproduce and assess quickly:

- A clear description of the issue and its impact.
- The affected component (e.g. a specific kernel, the tensor/storage layer,
  the `.sushi`/`.npy`/binary file loaders) and the version / commit hash.
- Step-by-step reproduction, ideally with a minimal proof of concept
  (including the tensor shapes/dtypes/layout involved, or the input file, if
  relevant).
- The platform, toolchain, and build configuration.
- Any suggested mitigation, if you have one.

## What to expect

- **Acknowledgement** within **72 hours**.
- An initial assessment and severity classification within **7 days**.
- Regular updates while we work on a fix, and credit in the release notes once a
  fix ships (unless you ask to remain anonymous).

Please give us a reasonable window to release a fix before any public
disclosure. We are happy to coordinate timing with you.

## Scope and threat model

A few properties are intentional design decisions rather than
vulnerabilities — please keep them in mind:

- **SushiBLAS executes the operations you ask it to.** It is a compute
  library, not a sandbox: calling `BLASOps(engine).gemm(...)` runs a genuine
  matrix multiplication against the memory you gave it. We are interested in
  memory-safety bugs *inside* SushiBLAS's own kernels and dispatch code
  (out-of-bounds reads/writes from a valid-looking but malicious shape/stride
  combination, a `Tensor`/`Storage` lifetime bug), not in the fact that the
  library computes what it is told to compute.
- **Tensor construction validates shape and storage bounds** (`SB_THROW_IF`
  checks in `Tensor`'s constructor and accessors). A crafted shape/stride/
  offset combination that bypasses these checks and reaches an out-of-bounds
  memory access in a kernel is squarely in scope.

In scope, and especially valued:

- Memory-safety bugs in any kernel (`modules/kernels/reference/`, `modules/kernels/sycl/`) —
  out-of-bounds access, use-after-free through `Storage`'s reference
  counting, an integer overflow in a shape/stride/offset computation.
- Parsing flaws in the tensor file loaders (`IO::load`, `load_npy`,
  `load_bin`) — a malformed `.sushi`, `.npy`, or raw binary file that causes
  a buffer overrun, an out-of-bounds read, or a crash rather than a clean
  error.
- Any bypass of `Tensor`'s own bounds/shape validation that reaches an
  out-of-bounds kernel access.

Out of scope:

- The fact that a caller can request arbitrary compute over memory it
  legitimately owns.
- Timing or performance characteristics of a kernel.

## Supported versions

SushiBLAS is pre-1.0 and under active development. Security fixes are applied
to the latest `main`; there is no long-term support branch yet. Always test
against the current `main` before reporting.

## Contact

For anything that isn't sensitive, a GitHub issue or Discussion is fine. For
security matters, use the private channels above or email
**mustafagarip@sushisystems.io**.
