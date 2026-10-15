# Tests of random

The unit tests of the `random` module, built into `sushiblas_random_tests` by the
module's `CMakeLists.txt`. The source comments state what a test checks; a section below
holds why a test is written the way it is. `tests/unit/README.md` holds the reasoning the
tests share, and `modules/engine/engine_fixture/README.md` describes the fixture.

| File | Tests |
| --- | --- |
| `test_constant.cpp` | Tests that RandomOps::constant fills a tensor of every supported dtype with one value |
| `test_distributions.cpp` | Tests the log-normal, exponential, Poisson, Bernoulli, discrete and truncated draws |
| `test_initializers.cpp` | Tests the Xavier and He initializers of RandomOps |
| `test_normal.cpp` | Tests RandomOps::normal's sample moments, spread and seed determinism |
| `test_orthogonal.cpp` | Tests that RandomOps::orthogonal fills a matrix with orthonormal rows or columns |
| `test_random_large_lambda.cpp` | Tests the Poisson draw for lambda in the PTRS range and its reproducibility |
| `test_random_reference.cpp` | Tests that each random operation draws one stream under both forced modes |
| `test_shuffle.cpp` | Tests that RandomOps::shuffle permutes its input and repeats for one seed and offset |
| `test_uniform.cpp` | Tests RandomOps::uniform's bounds, sample moments, spread and seed determinism |

## test_orthogonal.cpp

`RandomOps::orthogonal` was declared in an installed header, implemented as a forward to
`normal()` and commented out of the build. A consumer calling it failed at link, and anyone who
un-commented it got a Gaussian matrix under an orthogonal name. It is now a modified
Gram-Schmidt orthonormalization, and the tests check the only property that distinguishes it
from the fallback it replaced.

`expect_orthonormal` is the whole contract: an orthogonal matrix is exactly one whose vectors
have unit norm and vanishing pairwise inner products.

`ResultIsNotJustAGaussianFill` is the regression the rewrite was made for. The earlier
implementation returned `normal(0, 1)`, whose columns are neither unit-norm nor orthogonal. A
Gaussian 6x6 has column norms near sqrt(6), about 2.45, so a unit diagonal cannot be an
accident.

## test_shuffle.cpp

`RandomOps::shuffle` was declared in an installed header, implemented as a no-op stub and
commented out of the build, so any consumer that called it failed at link. It is now a
Fisher-Yates permutation, and the tests are the properties that make it one.

`ResultIsAPermutationOfTheInput`: being a permutation is the defining property, and the old
stub satisfied it trivially by doing nothing. Hence the second half of the test, which asserts
that something moved. Identity is a legal permutation, but the odds of drawing it from 64! are
not distinguishable from zero, so seeing it means the kernel never ran.

`SameSeedReproducesTheSamePermutation`: a fresh engine restarts the RNG offset counter, so the
second run uses the same (seed, offset) pair as the first. That is the reproducibility contract
the header advertises.

`ConsecutiveShufflesOnOneEngineDiffer` guards against the failure mode of an RNG keyed on the
seed alone, with no offset.
