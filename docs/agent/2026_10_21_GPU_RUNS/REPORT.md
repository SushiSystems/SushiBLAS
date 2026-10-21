# GPU runs and the log-softmax measurement

**Status:** Shipped — 2026-10-21

The records of the GPU runs of the suite and of one closed fix. The design document
[GPU validation checklist](../../design/GPU_VALIDATION.md) took them out when it took the design
format.

## The GPU runs

All three runs were on one card, an NVIDIA GeForce RTX 3080 Ti, which `sycl-ls` named in the first
two as
`[cuda:gpu][cuda:0] NVIDIA CUDA BACKEND, NVIDIA GeForce RTX 3080 Ti 8.6 [CUDA 13.3]`.

| | First run | Second run | Third run |
| --- | --- | --- | --- |
| Date | 2026-10-07, at 14:15 and repeated at 14:53 | 2026-10-09 | 2026-10-12 |
| Tree | commit `dd5cdfc`, before the first commit of the v0.1.0 work | the v0.1.0 tree of that day, after the interface break | branch `feature/api_refinement`, built by `sb build` on Windows with the pinned intel-llvm |
| Result | 790 of 792 tests passed | 2 047 of 2 047 tests passed | 2 207 tests ran, none failed |
| Skipped | `Medium_LoggerDeathTest.AssertWritesItsMessageBeforeAborting`, a death test that needs a debug build | CTest lists some tests as skipped; their count and their reason were not recorded | not recorded |
| Failed | `Medium_Level2SyclParityTest.SymvMatchesReferenceLarge<float>` | none | none |

Before 2026-10-07 everything in this repository and in SushiAI was developed on a machine with no
usable GPU: `sycl-ls` reported one device, a Ryzen CPU through Intel OpenCL, and the NVIDIA path
could not be compiled.

What the runs do not show:

- Which checklist items they exercised was not recorded.
- The device profile was not printed in either run.
- The tree has run on no other card and, since the break, not on the OpenCL CPU device.

## The failure of the first run

For n = 4096 and the lower triangle, at element 1148:

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

## Log-softmax far below the batch maximum

Log-softmax returned `+inf` for a row far below the batch maximum, and is fixed. The shift used
the global maximum; it now uses a per-row maximum, computed by `ReductionOps::max_along`. The
measurement was repeated on binary32, on a [2, 3] input whose true log-probabilities run from
−0.41 to −2.41. The quantity below is the worst absolute deviation of the row's log-probabilities
from the answer the same code gives at a gap of 0. It is the one figure measured on both sides, by
the `EXPECT_EQ` of `Regression_LogSoftmaxStability.TheAnswerDoesNotDependOnTheGapAtAll`:

| Gap between the row and the batch maximum | Global shift, before | Per-row shift, after |
| ---: | ---: | ---: |
| 90 | 2.15e−06 | 0, exactly |
| 100 | 2.70e−02 | 0, exactly |
| 104 | `inf` | 0, exactly |
| 200 | `inf` | 0, exactly |

The after column is zero at every gap because the gap does not enter the arithmetic, and the
assertion is an equality for that reason.

Measured against the hand-computed expectation, and not against the answer at gap 0, the gap-100
case came out 2.7033e−02 off, which is 6.6% of the largest true log-probability (−0.40761).

The residual at the gaps that pass is not stated on either side, because it was not measured: a
passing `EXPECT_NEAR` bounds it under 1e−4 and says no more.

## The last full run on the development CPU

`sb test --suite all` passed 403 of 403 on 2026-08-03. SushiAI's `sa test --suite all` passed
582 of 582 when last recorded, and has not been re-run since; both repositories have gained tests.
The CPU baseline of the MLP demo is about 7.5 ms/step on a 12-core Ryzen for a 32→128→6 MLP at
batch 32.
