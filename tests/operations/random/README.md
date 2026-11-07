# Tests of random

The tests of the `random` module, built into `sushiblas_random_test_objects` and linked
into `sushiblas_operations_tests` by `sushiblas_add_test_program()` in `cmake/SushiBLASTests.cmake`. The source comments state
what a test checks; a section below holds why a test is written the way it is.
[`tests/README.md`](../../README.md) holds the reasoning the tests share, and
`modules/engine/engine_fixture/README.md` describes the fixture.

| File | Tests |
| --- | --- |
| `random_statistics.hpp` | Holds the laws of the draws and the standard-error checks the statistical tests share |
| `test_dispatch.cpp` | Tests the dtypes a random task admits, its seed slots and the fill on both paths |
| `test_random/test_constant.cpp` | Tests, one row per dtype, that RandomOps::constant fills a tensor with one value, `Small_` on a host engine |
| `test_random/test_distributions.cpp` | Tests the log-normal, exponential, Poisson, Bernoulli, discrete and truncated draws against their laws, `Small_` on a host engine |
| `test_random/test_initializers.cpp` | Tests the Xavier and He initializers of RandomOps, `Small_` on a host engine |
| `test_random/test_normal.cpp` | Tests RandomOps::normal's sample moments, spread and seed determinism, `Small_` on a host engine |
| `test_random/test_orthogonal.cpp` | Tests, one row per shape and gain, that RandomOps::orthogonal fills a matrix with orthonormal rows or columns, `Small_` on a host engine |
| `test_random/test_random_large_lambda.cpp` | Tests the Poisson moments for lambda 50, 5000 and 1e6, and the replay of the Knuth and PTRS methods, `Small_` on a host engine |
| `test_random/test_random_reference.cpp` | Tests that each random operation family draws one stream under both forced modes |
| `test_random/test_random_refusals.cpp` | Tests, on a host engine, that RandomOps refuses a bad parameter, truncated-normal bound, dtype or shape at record time |
| `test_random/test_shuffle.cpp` | Tests that RandomOps::shuffle permutes its input and repeats for one seed and offset, `Small_` on a host engine |
| `test_random/test_uniform.cpp` | Tests RandomOps::uniform's bounds, sample moments, spread and seed determinism, `Small_` on a host engine |
| `compile_random_log_category.cpp` | Checks that `random_log_category.hpp` compiles on its own and names `K_RANDOM_LOG_CATEGORY`; compiled, never run |
| `benchmark_random.cpp` | Times the random draws, shuffle and orthogonal and prints Melem/s per length |

## test_random/test_orthogonal.cpp

`RandomOps::orthogonal` was declared in an installed header, implemented as a forward to
`normal()` and commented out of the build. A consumer calling it failed at link, and anyone who
un-commented it got a Gaussian matrix under an orthogonal name. It is now a modified
Gram-Schmidt orthonormalization, and the tests check the only property that distinguishes it
from the fallback it replaced.

`expect_orthonormal` is the whole contract: an orthogonal matrix is exactly one whose vectors
have unit norm and vanishing pairwise inner products.

The Gram rows are the regression the rewrite was made for. The earlier implementation returned
`normal(0, 1)`, whose columns are neither unit-norm nor orthogonal. A Gaussian 6x6 has column
norms near sqrt(6), about 2.45, so the unit diagonal of the square row cannot be an accident.
Each row's tolerance is `absolute_tolerance<float>(128, gain^2)`: 128 float roundings of a value
of the size of the diagonal.

## Statistical tests

The statistical tests draw from a fixed seed and compare the sample mean and the sample variance
with the law's, in `random_statistics.hpp`. A bound is five standard errors, computed from the
law's variance and fourth central moment and the sample count, so no typed epsilon appears. At five
standard errors the chance that a fixed seed fails by accident is about 6e-7 per moment.

## test_random/test_shuffle.cpp

`RandomOps::shuffle` was declared in an installed header, implemented as a no-op stub and
commented out of the build, so any consumer that called it failed at link. It is now a
Fisher-Yates permutation, and the tests are the properties that make it one.

`ResultIsAPermutationOfTheInput`: being a permutation is the defining property, and the old
stub satisfied it trivially by doing nothing. Hence `ResultIsNotTheIdentity`, which asserts
that something moved. Identity is a legal permutation, but the odds of drawing it from 64! are
not distinguishable from zero, so seeing it means the kernel never ran.

`SameSeedReproducesTheSamePermutation`: a fresh engine restarts the RNG offset counter, so the
second run uses the same (seed, offset) pair as the first. That is the reproducibility contract
the header advertises.

`ConsecutiveShufflesOnOneEngineDiffer` guards against the failure mode of an RNG keyed on the
seed alone, with no offset.
